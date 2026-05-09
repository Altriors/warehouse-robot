"""
metrics.py
----------
Collects and displays planning and execution metrics.

Metrics tracked:
    - planning_time     : seconds spent in the planner
    - nodes_expanded    : A* nodes popped from the open list
    - total_path_cost   : total cells traversed during execution
    - replan_count      : number of times dynamic replanning was triggered
    - packages_delivered: number of packages successfully delivered
    - total_moves       : total move actions taken by the robot
    - recharge_count    : number of times the robot recharged
"""

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class Metrics:
    """Container for all planning and execution metrics."""
    planning_time: float = 0.0
    nodes_expanded: int = 0
    total_path_cost: int = 0
    replan_count: int = 0
    packages_delivered: int = 0
    total_moves: int = 0
    recharge_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        """Wall-clock seconds since simulation started."""
        return time.perf_counter() - self.start_time

    def update_from_executor(self, executor) -> None:
        """Pull final metrics from the executor and robot."""
        self.total_path_cost = executor.total_path_cost
        self.replan_count    = executor.replan_count
        self.total_moves     = executor.robot.total_moves
        self.recharge_count  = executor.robot.recharge_count
        self.packages_delivered = len(executor.robot.delivered)

    def update_from_planner(self, planner) -> None:
        """Pull planning metrics from the planner."""
        self.planning_time  = planner.planning_time
        self.nodes_expanded = planner.nodes_expanded

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "═" * 45,
            "         PLANNING & EXECUTION METRICS",
            "═" * 45,
            f"  Planning Time       : {self.planning_time * 1000:.2f} ms",
            f"  Nodes Expanded (A*) : {self.nodes_expanded}",
            f"  Total Path Cost     : {self.total_path_cost} cells",
            f"  Replans Triggered   : {self.replan_count}",
            f"  Packages Delivered  : {self.packages_delivered}",
            f"  Total Moves         : {self.total_moves}",
            f"  Recharge Events     : {self.recharge_count}",
            f"  Wall-clock Time     : {self.elapsed():.2f} s",
            "═" * 45,
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "planning_time_ms": round(self.planning_time * 1000, 2),
            "nodes_expanded": self.nodes_expanded,
            "total_path_cost": self.total_path_cost,
            "replan_count": self.replan_count,
            "packages_delivered": self.packages_delivered,
            "total_moves": self.total_moves,
            "recharge_count": self.recharge_count,
            "wall_clock_s": round(self.elapsed(), 2),
        }