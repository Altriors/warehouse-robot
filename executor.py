"""
executor.py
-----------
Executes the plan produced by GoalStackPlanner step by step.

Dynamic Replanning:
    After every N moves the executor randomly spawns a dynamic obstacle
    on the robot's upcoming path.  If the next planned cell is blocked,
    the executor triggers A* replanning to find a detour, increments the
    replan counter, and continues execution.

The executor yields ExecutionFrame objects so the visualiser can render
each step without the executor needing to know about matplotlib.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

from warehouse import Warehouse
from robot import Robot
from planner import GoalStackPlanner


# ---------------------------------------------------------------------------
# Execution frame (one visualisation tick)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionFrame:
    """Snapshot of simulation state for one animation frame."""
    robot_pos: Tuple[int, int]
    battery: int
    carrying: Optional[str]
    delivered: set
    dynamic_obstacles: set
    action_label: str
    task_index: int
    total_tasks: int
    replan_triggered: bool = False
    mission_complete: bool = False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Walks through a plan list produced by GoalStackPlanner and applies
    each step to the live Robot + Warehouse state.

    Yields one ExecutionFrame per cell moved / action taken so the
    visualiser can animate frame-by-frame.
    """

    OBSTACLE_SPAWN_INTERVAL = 8   # spawn a dynamic obstacle every N moves
    OBSTACLE_LIFETIME = 25        # remove dynamic obstacle after N frames

    def __init__(
        self,
        robot: Robot,
        warehouse: Warehouse,
        planner: GoalStackPlanner,
    ):
        self.robot = robot
        self.warehouse = warehouse
        self.planner = planner

        # Metrics
        self.replan_count: int = 0
        self.total_path_cost: int = 0
        self._move_counter: int = 0
        self._obstacle_expiry: List[Tuple[int, "frame_number"]] = []

    # ------------------------------------------------------------------
    # Main execution generator
    # ------------------------------------------------------------------

    def execute(self, plan: List[dict]) -> Generator[ExecutionFrame, None, None]:
        """
        Execute the plan, yielding a frame after each atomic action.
        Handles dynamic obstacles and replanning transparently.
        """
        frame_number = 0
        pending = list(plan)          # mutable copy
        task_total = sum(1 for s in plan if s["action"] == "DROP")
        task_done = 0

        step_idx = 0
        while step_idx < len(pending):
            step = pending[step_idx]
            action = step["action"]
            label  = step.get("label", action)

            # ── Expire old dynamic obstacles ──────────────────────────
            self._expire_obstacles(frame_number)

            # ── Maybe spawn a new dynamic obstacle ────────────────────
            spawned = self._maybe_spawn_obstacle(step, frame_number)

            if action == "MOVE":
                path = step["path"]
                # Walk each cell in the path
                i = 0
                while i < len(path):
                    cell = path[i]

                    # Check if next cell is still passable
                    if not self.warehouse.is_passable(*cell):
                        # ── REPLAN ──────────────────────────────────
                        self.replan_count += 1
                        new_path = self.planner.plan_path(
                            self.robot.position,
                            step["target"],
                            self.warehouse,
                        )

                        frame_number += 1
                        yield ExecutionFrame(
                            robot_pos=self.robot.position,
                            battery=self.robot.battery,
                            carrying=self.robot.carrying,
                            delivered=set(self.robot.delivered),
                            dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
                            action_label=f"[!] Replanning! Obstacle at {cell}",
                            task_index=task_done,
                            total_tasks=task_total,
                            replan_triggered=True,
                        )

                        if new_path is None:
                            # Completely blocked – wait a frame and retry later
                            time.sleep(0.1)
                            break
                        else:
                            # Replace remainder of path with replanned path
                            path = new_path
                            step = {**step, "path": new_path}
                            i = 0
                            continue

                    # Normal move
                    moved = self.robot.move(cell, self.warehouse)
                    if moved:
                        self.total_path_cost += 1
                        self._move_counter += 1
                        # Try to spawn a dynamic obstacle ahead on the path
                        remaining = path[i+1:]
                        self._maybe_spawn_obstacle_on_path(remaining, frame_number)

                    frame_number += 1
                    yield ExecutionFrame(
                        robot_pos=self.robot.position,
                        battery=self.robot.battery,
                        carrying=self.robot.carrying,
                        delivered=set(self.robot.delivered),
                        dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
                        action_label=label,
                        task_index=task_done,
                        total_tasks=task_total,
                    )
                    i += 1

            elif action == "PICKUP":
                pkg_id = step["pkg_id"]
                self.robot.pickup(pkg_id, self.warehouse)
                frame_number += 1
                yield ExecutionFrame(
                    robot_pos=self.robot.position,
                    battery=self.robot.battery,
                    carrying=self.robot.carrying,
                    delivered=set(self.robot.delivered),
                    dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
                    action_label=label,
                    task_index=task_done,
                    total_tasks=task_total,
                )

            elif action == "DROP":
                self.robot.drop(self.warehouse)
                task_done += 1
                frame_number += 1
                yield ExecutionFrame(
                    robot_pos=self.robot.position,
                    battery=self.robot.battery,
                    carrying=self.robot.carrying,
                    delivered=set(self.robot.delivered),
                    dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
                    action_label=label,
                    task_index=task_done,
                    total_tasks=task_total,
                )

            elif action == "RECHARGE":
                self.robot.recharge(self.warehouse)
                frame_number += 1
                yield ExecutionFrame(
                    robot_pos=self.robot.position,
                    battery=self.robot.battery,
                    carrying=self.robot.carrying,
                    delivered=set(self.robot.delivered),
                    dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
                    action_label=label,
                    task_index=task_done,
                    total_tasks=task_total,
                )

            step_idx += 1

        # Mission complete frame
        yield ExecutionFrame(
            robot_pos=self.robot.position,
            battery=self.robot.battery,
            carrying=self.robot.carrying,
            delivered=set(self.robot.delivered),
            dynamic_obstacles=set(self.warehouse.dynamic_obstacles),
            action_label="[DONE] Mission Complete!",
            task_index=task_done,
            total_tasks=task_total,
            mission_complete=True,
        )

    # ------------------------------------------------------------------
    # Dynamic obstacle helpers
    # ------------------------------------------------------------------

    def _maybe_spawn_obstacle_on_path(self, remaining_path: list, frame_number: int) -> bool:
        """
        Every OBSTACLE_SPAWN_INTERVAL moves, place a dynamic obstacle
        a few cells ahead on the remaining path to force a replan.
        """
        if self._move_counter % self.OBSTACLE_SPAWN_INTERVAL != 0:
            return False
        if len(remaining_path) < 3:
            return False
        # Target a cell 2–4 steps ahead so robot has time to detect it
        candidates = remaining_path[2: min(5, len(remaining_path))]
        random.shuffle(candidates)
        for candidate in candidates:
            added = self.warehouse.add_dynamic_obstacle(candidate)
            if added:
                expiry = frame_number + self.OBSTACLE_LIFETIME
                self._obstacle_expiry.append((candidate, expiry))
                return True
        return False

    def _maybe_spawn_obstacle(self, step: dict, frame_number: int) -> bool:
        """Legacy stub – kept for compatibility, no-op."""
        return False

    def _expire_obstacles(self, frame_number: int) -> None:
        """Remove dynamic obstacles whose lifetime has expired."""
        still_active = []
        for pos, expiry in self._obstacle_expiry:
            if frame_number >= expiry:
                self.warehouse.remove_dynamic_obstacle(pos)
            else:
                still_active.append((pos, expiry))
        self._obstacle_expiry = still_active