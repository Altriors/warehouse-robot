"""
warehouse.py
------------
Defines the Warehouse environment: the 2D grid, cell types,
package locations, delivery stations, charging station, and obstacles.

Cell Types (used in grid):
    0 = Free space
    1 = Static obstacle / shelf wall
    2 = Charging station
    3 = Delivery station
    4 = Package location (shelf with package)
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Package:
    """Represents a package in the warehouse."""
    pkg_id: str
    location: Tuple[int, int]       # (row, col) where it sits
    destination: Tuple[int, int]    # delivery station (row, col)
    delivered: bool = False


@dataclass
class WarehouseConfig:
    """Static configuration for a warehouse layout."""
    rows: int = 12
    cols: int = 16
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    packages: List[Package] = field(default_factory=list)
    delivery_stations: List[Tuple[int, int]] = field(default_factory=list)
    charging_station: Tuple[int, int] = (0, 0)
    robot_start: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Warehouse class
# ---------------------------------------------------------------------------

class Warehouse:
    """
    The Warehouse holds the full environment state.

    The grid is a 2-D array of cell type integers.
    Packages and stations are tracked separately so they can be
    queried quickly by the planner.
    """

    # Cell-type constants
    FREE      = 0
    OBSTACLE  = 1
    CHARGING  = 2
    DELIVERY  = 3
    PKG_SHELF = 4

    def __init__(self, config: WarehouseConfig):
        self.rows = config.rows
        self.cols = config.cols
        self.charging_station = config.charging_station
        self.delivery_stations: List[Tuple[int, int]] = list(config.delivery_stations)

        # Build base grid
        self.grid: List[List[int]] = [
            [self.FREE] * self.cols for _ in range(self.rows)
        ]

        # Place static obstacles
        self.static_obstacles: set = set()
        self.dynamic_obstacles: set = set()          # added at runtime

        for obs in config.obstacles:
            self._place_obstacle(obs)
            self.static_obstacles.add(obs)

        # Mark special cells
        r, c = self.charging_station
        self.grid[r][c] = self.CHARGING

        for dr, dc in self.delivery_stations:
            self.grid[dr][dc] = self.DELIVERY

        # Packages: keyed by pkg_id
        self.packages: Dict[str, Package] = {}
        for pkg in config.packages:
            self.packages[pkg.pkg_id] = pkg
            pr, pc = pkg.location
            self.grid[pr][pc] = self.PKG_SHELF

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _place_obstacle(self, pos: Tuple[int, int]) -> None:
        r, c = pos
        if self._in_bounds(r, c):
            self.grid[r][c] = self.OBSTACLE

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, r: int, c: int) -> bool:
        """Returns True if the robot can step onto (r, c)."""
        if not self._in_bounds(r, c):
            return False
        if (r, c) in self.dynamic_obstacles:
            return False
        return self.grid[r][c] != self.OBSTACLE

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return passable neighbours (4-directional)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_passable(nr, nc):
                result.append((nr, nc))
        return result

    # ------------------------------------------------------------------
    # Dynamic obstacle management
    # ------------------------------------------------------------------

    def add_dynamic_obstacle(self, pos: Tuple[int, int]) -> bool:
        """
        Add a dynamic obstacle at pos if it is a free cell and not
        occupied by special objects.  Returns True if added.
        """
        r, c = pos
        if not self._in_bounds(r, c):
            return False
        if self.grid[r][c] in (self.OBSTACLE, self.CHARGING, self.DELIVERY):
            return False
        if pos in self.dynamic_obstacles:
            return False
        self.dynamic_obstacles.add(pos)
        return True

    def remove_dynamic_obstacle(self, pos: Tuple[int, int]) -> None:
        self.dynamic_obstacles.discard(pos)

    # ------------------------------------------------------------------
    # Package helpers
    # ------------------------------------------------------------------

    def package_at(self, pos: Tuple[int, int]) -> Optional[Package]:
        """Return the undelivered package sitting at pos, if any."""
        for pkg in self.packages.values():
            if pkg.location == pos and not pkg.delivered:
                return pkg
        return None

    def pending_packages(self) -> List[Package]:
        """All packages not yet delivered."""
        return [p for p in self.packages.values() if not p.delivered]

    def mark_delivered(self, pkg_id: str) -> None:
        """Mark a package as delivered and clear its shelf cell."""
        pkg = self.packages[pkg_id]
        pkg.delivered = True
        pr, pc = pkg.location
        # Only reset cell to FREE if no other package shares same location
        others = [p for p in self.packages.values()
                  if p.location == pkg.location and not p.delivered and p.pkg_id != pkg_id]
        if not others:
            self.grid[pr][pc] = self.FREE

    # ------------------------------------------------------------------
    # Factory: default warehouse layout
    # ------------------------------------------------------------------

    @staticmethod
    def default_warehouse() -> "Warehouse":
        """
        Build a pre-defined 12×16 warehouse.

        Layout design:
          - Two shelf blocks (left and right) made of obstacle cells
          - Packages placed on the FREE cells ADJACENT to shelves (pickup face)
          - Wide center aisle + top aisle for robot navigation
          - 3 delivery stations along the bottom row
          - Charging station top-left corner
        """
        obstacles = []

        # ── Left shelf block: rows 2-4, cols 2-3 ──────────────────────
        for r in range(2, 5):
            for c in range(2, 4):
                obstacles.append((r, c))

        # ── Left shelf block 2: rows 6-8, cols 2-3 ───────────────────
        for r in range(6, 9):
            for c in range(2, 4):
                obstacles.append((r, c))

        # ── Right shelf block: rows 2-4, cols 12-13 ──────────────────
        for r in range(2, 5):
            for c in range(12, 14):
                obstacles.append((r, c))

        # ── Right shelf block 2: rows 6-8, cols 12-13 ────────────────
        for r in range(6, 9):
            for c in range(12, 14):
                obstacles.append((r, c))

        # ── Center shelf block: rows 3-5, cols 7-8 ───────────────────
        for r in range(3, 6):
            for c in range(7, 9):
                obstacles.append((r, c))

        # Packages placed on FREE cells next to shelves (aisle-facing side)
        packages = [
            Package("P1", location=(2, 4),   destination=(11, 1)),   # right of left-top shelf
            Package("P2", location=(4, 4),   destination=(11, 7)),   # right of left-top shelf lower
            Package("P3", location=(2, 11),  destination=(11, 13)),  # left of right-top shelf
            Package("P4", location=(6, 4),   destination=(11, 1)),   # right of left-bottom shelf
            Package("P5", location=(7, 11),  destination=(11, 7)),   # left of right-bottom shelf
        ]

        delivery_stations = [(11, 1), (11, 7), (11, 13)]
        charging_station  = (0, 0)

        config = WarehouseConfig(
            rows=12, cols=16,
            obstacles=obstacles,
            packages=packages,
            delivery_stations=delivery_stations,
            charging_station=charging_station,
            robot_start=(0, 2),
        )
        return Warehouse(config)