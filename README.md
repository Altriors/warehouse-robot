# 🤖 Autonomous Warehouse Robot Planning System

> Classical AI Planning for Robot Task Execution — STRIPS + A* Search

---

## Overview

This project simulates an **autonomous warehouse robot** that uses **classical AI planning** to pick up packages from shelf locations and deliver them to designated drop-off stations — all without any human control.

The robot reasons about its environment using a **STRIPS-style action model**, plans optimal paths with **A\* search**, and **dynamically replans** when unexpected obstacles appear during execution.

---

## Features

- 12×16 warehouse grid with shelves, aisles, and stations
- 5 packages across different shelf locations with unique delivery destinations
- Robot carries one package at a time
- Battery system — robot must recharge when battery is low
- Dynamic obstacles appear mid-execution, forcing replanning
- Full step-by-step matplotlib animation with live status panel
- Metrics: planning time, A\* nodes expanded, path cost, replan count

---

## AI / Planning Concepts Used

### STRIPS Action Model
Each robot action is defined by:
- **Preconditions** — what must be true before the action can execute
- **Effects** — how the world state changes after execution

| Action   | Preconditions                              | Effects                        |
|----------|--------------------------------------------|--------------------------------|
| Move     | Target cell passable, battery > 0          | Position changes, battery -2   |
| PickUp   | At package location, not carrying anything | carrying = package             |
| Drop     | At package destination, carrying package   | Package delivered, carrying = None |
| Recharge | At charging station                        | battery = 100                  |

### Goal Stack Planning
The high-level planner decomposes the full mission into an ordered stack of sub-goals:

```
For each package (ordered by nearest-first heuristic):
    1. Recharge if battery estimate is too low
    2. Navigate to package location
    3. Pick up package
    4. Navigate to delivery station
    5. Drop package
```

### A\* Path Planning
For every navigation sub-goal, A\* finds the shortest collision-free path through the grid.

- **Heuristic**: Manhattan distance (admissible — never overestimates)
- **Edge cost**: 1 per cell (uniform)
- **State**: (row, col) grid cell

### Dynamic Replanning
During execution, dynamic obstacles are randomly spawned on the robot's planned path. When the next cell becomes impassable, the executor triggers a fresh A\* call from the current position to the same target, seamlessly rerouting.

---

## Project Structure

```
warehouse_robot/
├── main.py          # Entry point — builds, plans, executes, animates
├── warehouse.py     # Grid environment, packages, stations, obstacles
├── robot.py         # Robot state, STRIPS actions (move/pickup/drop/recharge)
├── planner.py       # A* path planner + Goal Stack mission planner
├── executor.py      # Step-by-step execution with dynamic replanning
├── visualizer.py    # Matplotlib animation and status panel
├── metrics.py       # Planning and execution metrics
├── requirements.txt
└── README.md
```

---

## Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/warehouse-robot-planning.git
cd warehouse-robot-planning
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python main.py
```

---

## Visualization Guide

| Colour       | Meaning                        |
|--------------|--------------------------------|
| 🔵 Blue      | Robot (empty)                  |
| 🟣 Purple    | Robot (carrying a package)     |
| 🟡 Yellow    | Package on shelf               |
| 🟢 Green     | Delivery station               |
| 🩵 Cyan      | Charging station               |
| ⬛ Dark grey | Static shelf / obstacle        |
| 🟠 Orange    | Dynamic obstacle (temporary)   |

The **right panel** shows:
- Live battery bar (green → yellow → red)
- Per-package delivery status
- Current action label
- Replan alerts (red border flash)

---

## Metrics Output

After the animation closes, the terminal prints:

```
═════════════════════════════════════════════
         PLANNING & EXECUTION METRICS
═════════════════════════════════════════════
  Planning Time       : X.XX ms
  Nodes Expanded (A*) : XXX
  Total Path Cost     : XXX cells
  Replans Triggered   : X
  Packages Delivered  : 5
  Total Moves         : XXX
  Recharge Events     : X
  Wall-clock Time     : X.XX s
═════════════════════════════════════════════
```

---

## Requirements

- Python 3.8+
- matplotlib ≥ 3.7
- numpy ≥ 1.24

---

## Course

**AI Course Project** — Classical Planning for Robot Task Execution