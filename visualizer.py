"""
visualizer.py
-------------
Renders the warehouse grid and animates the robot's task execution
using matplotlib.

Layout:
    Left panel  – warehouse grid (colour-coded cells + robot overlay)
    Right panel – live status panel (battery bar, task list, metrics)

Colour scheme:
    White       – free space
    Dark grey   – static obstacle / shelf wall
    Red/orange  – dynamic obstacle
    Green       – delivery station
    Cyan        – charging station
    Yellow      – package on shelf
    Blue        – robot
    Purple      – package being carried (shown on robot cell)
"""

import time
from typing import Generator, List, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from warehouse import Warehouse
from executor import ExecutionFrame


# ---------------------------------------------------------------------------
# Colour map indices (matches _build_color_matrix logic)
# ---------------------------------------------------------------------------
C_FREE     = 0
C_OBSTACLE = 1
C_CHARGE   = 2
C_DELIVERY = 3
C_PKG      = 4
C_ROBOT    = 5
C_CARRY    = 6   # robot carrying a package
C_DYN_OBS  = 7

COLORS = [
    "#2C2C3E",   # 0 free (dark background so yellow/colors pop)
    "#424242",   # 1 static obstacle
    "#00BCD4",   # 2 charging station
    "#43A047",   # 3 delivery station
    "#FF8F00",   # 4 package shelf - deep amber, very visible on dark bg
    "#1E88E5",   # 5 robot
    "#8E24AA",   # 6 robot + carrying
    "#FF5722",   # 7 dynamic obstacle
]


class Visualizer:
    """Drives the matplotlib animation."""

    def __init__(
        self,
        warehouse: Warehouse,
        frames: List[ExecutionFrame],
        interval_ms: int = 200,
        robot_start=None,
    ):
        self.warehouse = warehouse
        self.frames = frames
        self.interval_ms = interval_ms
        self.robot_start = robot_start

        # Build the colour lookup as a (N_COLORS, 3) float array for imshow
        self._cmap_array = np.array([
            matplotlib.colors.to_rgb(c) for c in COLORS
        ])

        self._fig = None
        self._ax_grid = None
        self._ax_status = None
        self._im = None
        self._battery_bar = None
        self._status_text = None
        self._task_texts: List = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the figure, wire up FuncAnimation, and show."""
        self._setup_figure()
        anim = FuncAnimation(
            self._fig,
            self._update,
            frames=len(self.frames),
            interval=self.interval_ms,
            repeat=False,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        self._fig = plt.figure(figsize=(16, 8), facecolor="#1A1A2E")
        gs = GridSpec(1, 2, width_ratios=[3, 1], figure=self._fig)

        # ── Left: grid ─────────────────────────────────────────────────
        self._ax_grid = self._fig.add_subplot(gs[0])
        self._ax_grid.set_facecolor("#1A1A2E")
        self._ax_grid.set_title(
            "Autonomous Warehouse Robot – Classical Planning",
            color="white", fontsize=13, fontweight="bold", pad=10
        )
        self._ax_grid.set_xticks([])
        self._ax_grid.set_yticks([])

        # Initial colour matrix
        init_matrix = self._build_color_matrix(self.frames[0])
        self._im = self._ax_grid.imshow(
            init_matrix, interpolation="nearest",
            extent=[-0.5, self.warehouse.cols - 0.5,
                    self.warehouse.rows - 0.5, -0.5]
        )
        self._draw_grid_lines()
        self._draw_legend()
        self._draw_station_labels()
        self._pkg_label_texts = self._draw_package_labels()

        # ── Right: status panel ─────────────────────────────────────────
        self._ax_status = self._fig.add_subplot(gs[1])
        self._ax_status.set_facecolor("#0D0D1A")
        self._ax_status.set_xlim(0, 1)
        self._ax_status.set_ylim(0, 1)
        self._ax_status.set_xticks([])
        self._ax_status.set_yticks([])
        self._ax_status.set_title("Status", color="white",
                                  fontsize=11, fontweight="bold")

        # Battery bar background
        self._ax_status.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, 0.88), 0.9, 0.06,
                boxstyle="round,pad=0.01",
                linewidth=1, edgecolor="#555", facecolor="#222"
            )
        )
        self._battery_bar = self._ax_status.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, 0.88), 0.9, 0.06,
                boxstyle="round,pad=0.01",
                linewidth=0, facecolor="#4CAF50"
            )
        )

        self._battery_label = self._ax_status.text(
            0.5, 0.955, "Battery: 100%",
            ha="center", va="center", color="white",
            fontsize=9, fontweight="bold"
        )

        self._action_text = self._ax_status.text(
            0.5, 0.82, "",
            ha="center", va="center", color="#FFC107",
            fontsize=8, fontweight="bold", wrap=True
        )

        # Task list placeholders (up to 7 tasks)
        self._task_texts = []
        for i in range(7):
            t = self._ax_status.text(
                0.08, 0.74 - i * 0.09, "",
                ha="left", va="center", color="#AAAAAA",
                fontsize=8
            )
            self._task_texts.append(t)

        # Metrics text block
        self._metrics_text = self._ax_status.text(
            0.5, 0.05, "",
            ha="center", va="bottom", color="#90CAF9",
            fontsize=7.5, family="monospace",
            multialignment="left"
        )

    # ------------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------------

    def _update(self, frame_idx: int) -> None:
        frame = self.frames[frame_idx]

        # Rebuild colour matrix
        matrix = self._build_color_matrix(frame)
        self._im.set_data(matrix)

        # Flash red border on replan
        if frame.replan_triggered:
            for spine in self._ax_grid.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
        else:
            for spine in self._ax_grid.spines.values():
                spine.set_edgecolor("#333")
                spine.set_linewidth(1)

        # Battery bar
        pct = frame.battery / 100.0
        self._battery_bar.set_width(0.9 * pct)
        color = "#4CAF50" if pct > 0.4 else ("#FFC107" if pct > 0.2 else "#F44336")
        self._battery_bar.set_facecolor(color)
        self._battery_label.set_text(f"Battery: {frame.battery}%")

        # Action label
        self._action_text.set_text(frame.action_label)

        # Task list
        warehouse_pkgs = list(self.warehouse.packages.values())
        for i, txt_obj in enumerate(self._task_texts):
            if i < len(warehouse_pkgs):
                pkg = warehouse_pkgs[i]
                if pkg.pkg_id in frame.delivered:
                    status = "[OK]"
                    col = "#4CAF50"
                elif frame.carrying == pkg.pkg_id:
                    status = "[>>]"
                    col = "#FFC107"
                else:
                    status = "[ ]"
                    col = "#AAAAAA"
                txt_obj.set_text(f"{status} {pkg.pkg_id}: {pkg.location}→{pkg.destination}")
                txt_obj.set_color(col)
            else:
                txt_obj.set_text("")

        # Metrics block
        metrics_str = (
            f"Tasks: {frame.task_index}/{frame.total_tasks}\n"
            f"Frame: {frame_idx}"
        )
        self._metrics_text.set_text(metrics_str)

        if frame.mission_complete:
            self._ax_grid.set_title(
                "Mission Complete! All packages delivered.",
                color="#4CAF50", fontsize=13, fontweight="bold"
            )

        # Show/hide package labels based on delivery state
        for pkg_id, txt_obj in self._pkg_label_texts.items():
            pkg = self.warehouse.packages[pkg_id]
            visible = pkg_id not in frame.delivered and frame.carrying != pkg_id
            txt_obj.set_visible(visible)

    # ------------------------------------------------------------------
    # Colour matrix builder
    # ------------------------------------------------------------------

    def _build_color_matrix(self, frame: ExecutionFrame) -> np.ndarray:
        """
        Build an (rows, cols, 3) RGB float array for imshow.
        Priority (highest first): robot > dynamic obstacle > package >
        delivery > charging > static obstacle > free
        """
        rows, cols = self.warehouse.rows, self.warehouse.cols
        matrix = np.zeros((rows, cols, 3), dtype=float)

        # Fill base grid
        for r in range(rows):
            for c in range(cols):
                ct = self.warehouse.grid[r][c]
                if ct == Warehouse.OBSTACLE:
                    idx = C_OBSTACLE
                elif ct == Warehouse.CHARGING:
                    idx = C_CHARGE
                elif ct == Warehouse.DELIVERY:
                    idx = C_DELIVERY
                elif ct == Warehouse.PKG_SHELF:
                    # Only show package cell if package is still there
                    pkg = self.warehouse.package_at((r, c))
                    idx = C_PKG if pkg and not pkg.delivered else C_FREE
                else:
                    idx = C_FREE
                matrix[r, c] = self._cmap_array[idx]

        # Dynamic obstacles
        for (dr, dc) in frame.dynamic_obstacles:
            matrix[dr, dc] = self._cmap_array[C_DYN_OBS]

        # Robot
        rr, rc = frame.robot_pos
        robot_color = C_CARRY if frame.carrying else C_ROBOT
        matrix[rr, rc] = self._cmap_array[robot_color]

        return matrix

    # ------------------------------------------------------------------
    # Static drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid_lines(self) -> None:
        rows, cols = self.warehouse.rows, self.warehouse.cols
        for x in range(cols + 1):
            self._ax_grid.axvline(x - 0.5, color="#333", linewidth=0.4)
        for y in range(rows + 1):
            self._ax_grid.axhline(y - 0.5, color="#333", linewidth=0.4)

    def _draw_legend(self) -> None:
        legend_items = [
            mpatches.Patch(color=COLORS[C_ROBOT],    label="Robot"),
            mpatches.Patch(color=COLORS[C_CARRY],    label="Robot (carrying)"),
            mpatches.Patch(color=COLORS[C_PKG],      label="Package"),
            mpatches.Patch(color=COLORS[C_DELIVERY], label="Delivery Station"),
            mpatches.Patch(color=COLORS[C_CHARGE],   label="Charging Station"),
            mpatches.Patch(color=COLORS[C_OBSTACLE], label="Shelf/Obstacle"),
            mpatches.Patch(color=COLORS[C_DYN_OBS],  label="Dynamic Obstacle"),
        ]
        self._ax_grid.legend(
            handles=legend_items,
            loc="lower left",
            fontsize=7,
            framealpha=0.8,
            facecolor="#1A1A2E",
            labelcolor="white",
            edgecolor="#555",
        )

    def _draw_station_labels(self) -> None:
        """Label delivery and charging stations directly on the grid."""
        r, c = self.warehouse.charging_station
        self._ax_grid.text(
            c, r, "CHG", ha="center", va="center",
            fontsize=5, color="white", fontweight="bold"
        )
        for i, (dr, dc) in enumerate(self.warehouse.delivery_stations):
            self._ax_grid.text(
                dc, dr, f"D{i+1}", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold"
            )

    def _draw_package_labels(self) -> dict:
        """
        Draw a text label on each package cell.
        Returns a dict of pkg_id -> Text object so _update can toggle visibility.
        """
        label_texts = {}
        for pkg_id, pkg in self.warehouse.packages.items():
            r, c = pkg.location
            txt = self._ax_grid.text(
                c, r, pkg_id,
                ha="center", va="center",
                fontsize=6, color="white", fontweight="bold",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#FF8F00",
                          edgecolor="white", linewidth=0.8, alpha=0.95)
            )
            label_texts[pkg_id] = txt
        return label_texts