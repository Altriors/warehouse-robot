"""
main.py
-------
Entry point for the Autonomous Warehouse Robot Planning System.

Run:
    python main.py

Flow:
    1. Build warehouse environment
    2. Create robot at start position
    3. Run GoalStackPlanner to generate full mission plan
    4. Execute plan step-by-step via Executor (with dynamic replanning)
    5. Collect all frames and pass to Visualizer for animation
    6. Print final metrics summary
"""

import time
from warehouse import Warehouse
from robot import Robot
from planner import GoalStackPlanner
from executor import Executor
from visualizer import Visualizer
from metrics import Metrics


def main():
    print("=" * 50)
    print("  Autonomous Warehouse Robot Planning System")
    print("  Classical AI Planning – STRIPS + A*")
    print("=" * 50)

    # ── 1. Build environment ───────────────────────────────────────────
    print("\n[1/4] Building warehouse...")
    warehouse = Warehouse.default_warehouse()
    robot_start = (0, 2)
    robot = Robot(start=robot_start)
    print(f"      Grid: {warehouse.rows}×{warehouse.cols}")
    print(f"      Packages: {len(warehouse.packages)}")
    print(f"      Delivery stations: {len(warehouse.delivery_stations)}")

    # ── 2. Plan ───────────────────────────────────────────────────────
    print("\n[2/4] Running Goal Stack Planner (STRIPS + A*)...")
    metrics = Metrics()
    planner = GoalStackPlanner()
    plan = planner.plan_mission(robot, warehouse)
    metrics.update_from_planner(planner)
    print(f"      Plan steps generated: {len(plan)}")
    print(f"      Planning time: {planner.planning_time * 1000:.2f} ms")
    print(f"      A* nodes expanded: {planner.nodes_expanded}")

    # ── 3. Execute & collect frames ───────────────────────────────────
    print("\n[3/4] Executing plan (collecting animation frames)...")
    executor = Executor(robot, warehouse, planner)
    frames = list(executor.execute(plan))
    metrics.update_from_executor(executor)
    print(f"      Frames collected: {len(frames)}")
    print(f"      Replans triggered: {executor.replan_count}")

    # ── 4. Animate ────────────────────────────────────────────────────
    print("\n[4/4] Launching animation (close window to see metrics)...")
    viz = Visualizer(
        warehouse=warehouse,
        frames=frames,
        interval_ms=180,
        robot_start=robot_start,
    )
    viz.run()

    # ── Final metrics ─────────────────────────────────────────────────
    print("\n" + metrics.summary())


if __name__ == "__main__":
    main()