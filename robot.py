"""
robot.py
--------
Defines the Robot agent: its position, battery level, carried package,
and the STRIPS-style actions it can perform.

STRIPS Action Model
-------------------
Each action has:
    preconditions  – conditions that must hold in the current state
    effects        – how the state changes after execution

Actions implemented:
    Move(direction)   – move one cell; costs 1 battery unit
    PickUp(pkg_id)    – pick up package from current cell
    Drop(pkg_id)      – deliver package at delivery station
    Recharge()        – restore battery at charging station
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Robot State (used by planner for state-space search)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotState:
    """
    Immutable snapshot of the robot's world state.
    Used as nodes in the state-space search graph.
    """
    position: Tuple[int, int]
    battery: int
    carrying: Optional[str]              # pkg_id or None
    delivered: frozenset = field(default_factory=frozenset)

    def __hash__(self):
        return hash((self.position, self.battery, self.carrying, self.delivered))


# ---------------------------------------------------------------------------
# Robot class (mutable, tracks live simulation state)
# ---------------------------------------------------------------------------

class Robot:
    """
    The autonomous robot agent.

    Holds live mutable state during simulation.
    The planner works on immutable RobotState snapshots;
    the Robot class applies confirmed plan steps to the real simulation.
    """

    MAX_BATTERY      = 100
    BATTERY_PER_MOVE = 1          # battery drained per cell moved
    LOW_BATTERY_THRESHOLD = 20    # triggers recharge task

    def __init__(self, start: Tuple[int, int]):
        self.position: Tuple[int, int] = start
        self.battery: int = self.MAX_BATTERY
        self.carrying: Optional[str] = None     # pkg_id being carried
        self.delivered: set = set()             # pkg_ids delivered

        # Execution log for metrics
        self.total_moves: int = 0
        self.recharge_count: int = 0

    # ------------------------------------------------------------------
    # STRIPS Actions
    # ------------------------------------------------------------------

    def can_move(self, new_pos: Tuple[int, int], warehouse) -> Tuple[bool, str]:
        """
        Preconditions for Move:
            - new_pos is passable in the warehouse
            - battery >= 0 (battery hits 0 but robot can still complete step)
        """
        if not warehouse.is_passable(*new_pos):
            return False, f"Cell {new_pos} is blocked"
        return True, ""

    def move(self, new_pos: Tuple[int, int], warehouse) -> bool:
        """
        Effect of Move:
            - position <- new_pos
            - battery  <- battery - BATTERY_PER_MOVE
        """
        ok, reason = self.can_move(new_pos, warehouse)
        if not ok:
            return False
        self.position = new_pos
        self.battery = max(0, self.battery - self.BATTERY_PER_MOVE)
        self.total_moves += 1
        return True

    def can_pickup(self, pkg_id: str, warehouse) -> Tuple[bool, str]:
        """
        Preconditions for PickUp:
            - robot is not already carrying a package
            - package pkg_id exists, is undelivered, and is at robot's position
        """
        if self.carrying is not None:
            return False, f"Already carrying {self.carrying}"
        pkg = warehouse.packages.get(pkg_id)
        if pkg is None or pkg.delivered:
            return False, f"Package {pkg_id} unavailable"
        if pkg.location != self.position:
            return False, f"Not at package location {pkg.location}"
        return True, ""

    def pickup(self, pkg_id: str, warehouse) -> bool:
        """
        Effect of PickUp:
            - carrying <- pkg_id
        """
        ok, _ = self.can_pickup(pkg_id, warehouse)
        if not ok:
            return False
        self.carrying = pkg_id
        return True

    def can_drop(self, warehouse) -> Tuple[bool, str]:
        """
        Preconditions for Drop:
            - robot is carrying a package
            - robot is at the package's destination delivery station
        """
        if self.carrying is None:
            return False, "Not carrying any package"
        pkg = warehouse.packages[self.carrying]
        if self.position != pkg.destination:
            return False, f"Not at destination {pkg.destination}"
        return True, ""

    def drop(self, warehouse) -> bool:
        """
        Effect of Drop:
            - warehouse marks package as delivered
            - carrying <- None
            - delivered set grows
        """
        ok, _ = self.can_drop(warehouse)
        if not ok:
            return False
        pkg_id = self.carrying
        warehouse.mark_delivered(pkg_id)
        self.delivered.add(pkg_id)
        self.carrying = None
        return True

    def can_recharge(self, warehouse) -> Tuple[bool, str]:
        """
        Preconditions for Recharge:
            - robot is at charging station
        """
        if self.position != warehouse.charging_station:
            return False, f"Not at charging station {warehouse.charging_station}"
        return True, ""

    def recharge(self, warehouse) -> bool:
        """
        Effect of Recharge:
            - battery <- MAX_BATTERY
        """
        ok, _ = self.can_recharge(warehouse)
        if not ok:
            return False
        self.battery = self.MAX_BATTERY
        self.recharge_count += 1
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def needs_recharge(self) -> bool:
        return self.battery <= self.LOW_BATTERY_THRESHOLD

    def snapshot(self) -> RobotState:
        """Return an immutable state snapshot for the planner."""
        return RobotState(
            position=self.position,
            battery=self.battery,
            carrying=self.carrying,
            delivered=frozenset(self.delivered),
        )

    def __repr__(self) -> str:
        return (f"Robot(pos={self.position}, battery={self.battery}, "
                f"carrying={self.carrying}, delivered={self.delivered})")