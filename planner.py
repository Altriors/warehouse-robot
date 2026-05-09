"""
planner.py
----------
Classical AI Planning engine for the warehouse robot.

Two-layer planning architecture:
    1. High-level Goal Stack Planner (STRIPS-style)
       Decomposes the overall mission into an ordered stack of sub-goals:
           [Recharge if needed] → [Go to package] → [Pick up] →
           [Go to delivery] → [Drop] → repeat for next package

    2. Low-level A* Path Planner
       For each "go to location" sub-goal, A* finds the optimal
       collision-free path through the warehouse grid.

State Space:
    Node  = (row, col) grid cell
    Edge  = one move (cost = 1 per step)
    H(n)  = Manhattan distance to goal  (admissible heuristic for A*)

Metrics tracked:
    - nodes_expanded  : total A* nodes popped from the open list
    - planning_time   : cumulative seconds spent in plan()
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple

from warehouse import Warehouse
from robot import Robot


# ---------------------------------------------------------------------------
# A* Path Planner
# ---------------------------------------------------------------------------

class AStarPlanner:
    """
    Finds the shortest path between two grid cells using A*.

    Heuristic: Manhattan distance (admissible – never overestimates
    because the robot moves one cell at a time in 4 directions).
    """

    def __init__(self):
        self.nodes_expanded: int = 0

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        warehouse: Warehouse,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* from start to goal.

        Returns the path as a list of (row, col) cells including start
        and goal, or None if no path exists.
        """
        # Priority queue entries: (f_score, g_score, position)
        open_list: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_list, (0, 0, start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_list:
            _, g, current = heapq.heappop(open_list)
            self.nodes_expanded += 1

            if current == goal:
                return self._reconstruct(came_from, goal)

            # Skip if we have already found a better path to current
            if g > g_score.get(current, float("inf")):
                continue

            for neighbor in warehouse.neighbors(*current):
                tentative_g = g_score[current] + 1  # uniform edge cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        return None  # No path found

    @staticmethod
    def _reconstruct(
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Trace back from goal to start."""
        path = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# High-level Task / Goal representation
# ---------------------------------------------------------------------------

class Task:
    """A single delivery task: pick up pkg_id and deliver it."""

    def __init__(self, pkg_id: str):
        self.pkg_id = pkg_id

    def __repr__(self):
        return f"Task(deliver={self.pkg_id})"


# ---------------------------------------------------------------------------
# STRIPS-style Goal Stack Planner
# ---------------------------------------------------------------------------

class GoalStackPlanner:
    """
    Forward planner using a Goal Stack (a classical planning technique).

    The planner breaks the full delivery mission into an ordered sequence
    of primitive sub-goals.  Each sub-goal produces a path segment or a
    single action (PickUp / Drop / Recharge).

    Plan structure returned (list of PlanStep dicts):
        {
            "action"  : "MOVE" | "PICKUP" | "DROP" | "RECHARGE",
            "path"    : [list of (r,c) for MOVE steps] | [],
            "pkg_id"  : str | None,
            "target"  : (r,c) destination,
        }
    """

    def __init__(self):
        self.astar = AStarPlanner()
        self.nodes_expanded: int = 0
        self.planning_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_mission(
        self,
        robot: Robot,
        warehouse: Warehouse,
        pkg_order: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Generate the full mission plan for all pending packages.

        pkg_order: explicit delivery order (optional).  If None, packages
                   are ordered by distance from robot (greedy nearest-first).
        """
        t0 = time.perf_counter()
        self.astar.nodes_expanded = 0  # reset for this planning call

        pending = warehouse.pending_packages()
        if not pending:
            return []

        if pkg_order is None:
            pkg_order = self._nearest_first_order(robot.position, pending)

        steps: List[dict] = []

        current_pos = robot.position
        current_battery = robot.battery
        carrying = robot.carrying

        for pkg_id in pkg_order:
            pkg = warehouse.packages[pkg_id]
            if pkg.delivered:
                continue

            # ── Sub-goal 1: Recharge if battery may not be enough ──────
            # Rough estimate: distance to pkg + distance pkg→delivery + margin
            dist_to_pkg = self._manhattan(current_pos, pkg.location)
            dist_pkg_to_del = self._manhattan(pkg.location, pkg.destination)
            est_cost = (dist_to_pkg + dist_pkg_to_del) * robot.BATTERY_PER_MOVE

            if current_battery - est_cost < robot.LOW_BATTERY_THRESHOLD:
                charge_path = self.astar.plan(
                    current_pos, warehouse.charging_station, warehouse
                )
                if charge_path:
                    steps.append({
                        "action": "MOVE",
                        "path": charge_path,
                        "pkg_id": None,
                        "target": warehouse.charging_station,
                        "label": "-> Charging Station",
                    })
                    current_pos = warehouse.charging_station
                    current_battery = robot.MAX_BATTERY

                steps.append({
                    "action": "RECHARGE",
                    "path": [],
                    "pkg_id": None,
                    "target": warehouse.charging_station,
                    "label": "[CHG] Recharging",
                })

            # ── Sub-goal 2: Navigate to package location ────────────────
            if current_pos != pkg.location:
                path_to_pkg = self.astar.plan(current_pos, pkg.location, warehouse)
                if path_to_pkg is None:
                    continue  # unreachable – skip
                steps.append({
                    "action": "MOVE",
                    "path": path_to_pkg,
                    "pkg_id": pkg_id,
                    "target": pkg.location,
                    "label": f"-> Pick up {pkg_id}",
                })
                current_battery -= len(path_to_pkg) * robot.BATTERY_PER_MOVE
                current_pos = pkg.location

            # ── Sub-goal 3: Pick up ──────────────────────────────────────
            steps.append({
                "action": "PICKUP",
                "path": [],
                "pkg_id": pkg_id,
                "target": pkg.location,
                "label": f"[PKG] Pick up {pkg_id}",
            })
            carrying = pkg_id

            # ── Sub-goal 4: Navigate to delivery station ─────────────────
            path_to_del = self.astar.plan(current_pos, pkg.destination, warehouse)
            if path_to_del is None:
                continue  # unreachable
            steps.append({
                "action": "MOVE",
                "path": path_to_del,
                "pkg_id": pkg_id,
                "target": pkg.destination,
                "label": f"-> Deliver {pkg_id}",
            })
            current_battery -= len(path_to_del) * robot.BATTERY_PER_MOVE
            current_pos = pkg.destination

            # ── Sub-goal 5: Drop / deliver ───────────────────────────────
            steps.append({
                "action": "DROP",
                "path": [],
                "pkg_id": pkg_id,
                "target": pkg.destination,
                "label": f"[OK] Delivered {pkg_id}",
            })
            carrying = None

        self.nodes_expanded += self.astar.nodes_expanded
        self.planning_time += time.perf_counter() - t0
        return steps

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        warehouse: Warehouse,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Public wrapper around A* – used by executor during replanning
        to find a new path around a dynamic obstacle.
        """
        t0 = time.perf_counter()
        self.astar.nodes_expanded = 0
        path = self.astar.plan(start, goal, warehouse)
        self.nodes_expanded += self.astar.nodes_expanded
        self.planning_time += time.perf_counter() - t0
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_first_order(
        self,
        start: Tuple[int, int],
        packages,
    ) -> List[str]:
        """
        Greedy nearest-neighbour ordering.
        Minimises total travel by always going to the closest unvisited
        package next (a simple heuristic for the TSP-like sub-problem).
        """
        remaining = list(packages)
        order = []
        current = start

        while remaining:
            remaining.sort(key=lambda p: self._manhattan(current, p.location))
            chosen = remaining.pop(0)
            order.append(chosen.pkg_id)
            current = chosen.destination  # after delivery, robot is here

        return order