from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
EDGE_CENTERS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
OUTSIDE_TOUCH = {
    "up": (-1, 1),
    "down": (3, 1),
    "left": (1, -1),
    "right": (1, 3),
}
OPPOSITE_PAIRS = [("up", "down"), ("left", "right")]
ADJACENT_PAIRS = [("up", "left"), ("up", "right"), ("down", "left"), ("down", "right")]
SLOT_SIZE = 12
SLOT_PITCH = 14
BASE_SIZE = 28
RING_MASK = np.ones((3, 3), dtype=bool)
RING_MASK[1, 1] = False


def _station_path_cells(top: int, left: int, side_a: str, side_b: str) -> set[tuple[int, int]]:
    cells = {(top + 1, left + 1)}
    for side in (side_a, side_b):
        dr, dc = EDGE_CENTERS[side]
        cells.add((top + dr, left + dc))
    return cells


def _find_stations(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid)
    h, w = g.shape
    stations: list[dict[str, Any]] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            center = int(patch[1, 1])
            if center in (BG, GREY):
                continue
            if not bool((patch[RING_MASK] == GREY).all()):
                continue

            neighbors: dict[str, int] = {}
            for side, (dr, dc) in OUTSIDE_TOUCH.items():
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    color = int(g[rr, cc])
                    if color not in (BG, GREY):
                        neighbors[side] = color

            if len(neighbors) != 2:
                continue
            if len(set(neighbors.values())) != 2:
                continue
            if center not in neighbors.values():
                continue

            input_side = next(side for side, color in neighbors.items() if color == center)
            output_side = next(side for side, color in neighbors.items() if color != center)
            stations.append(
                {
                    "top": r0,
                    "left": c0,
                    "input_side": input_side,
                    "output_side": output_side,
                    "output_color": int(neighbors[output_side]),
                }
            )

    return stations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the output-colored connector inside each grey relay station using the center color as the input-side marker.

    Input structure:
    - The background is black.
    - Colored path segments are one-pixel-wide straight horizontal or vertical stubs.
    - Each relay station is a 3x3 grey block whose center pixel is recolored to one of the two adjacent path colors.
    - A valid station has exactly two colored path endpoints touching its top, bottom, left, or right side center.

    Rule:
    - Detect every 3x3 station whose outer ring is grey and whose center is non-black and non-grey.
    - Read the two colored path endpoints touching the station; the center color marks the input side and the other path is the output side.
    - Inside the station, paint the input edge-center cell, the center cell, and the output edge-center cell.
    - Use the output path color for those painted cells and leave every other pixel unchanged.

    Key insight:
    - The center color does not tell you what color to draw; it only identifies the incoming path. The connector must be painted in the other adjacent path color.
    """
    out = np.array(grid, copy=True)
    for station in _find_stations(grid):
        for rr, cc in _station_path_cells(
            int(station["top"]),
            int(station["left"]),
            str(station["input_side"]),
            str(station["output_side"]),
        ):
            out[rr, cc] = int(station["output_color"])
    return out


def _stub_cells(side: str, length: int) -> list[tuple[int, int]]:
    if side == "up":
        return [(-step, 1) for step in range(1, length + 1)]
    if side == "down":
        return [(2 + step, 1) for step in range(1, length + 1)]
    if side == "left":
        return [(1, -step) for step in range(1, length + 1)]
    return [(1, 2 + step) for step in range(1, length + 1)]


def _widget_kinds(count: int) -> list[str]:
    if count == 2:
        kinds = ["adjacent", "opposite"]
    elif count == 3:
        kinds = ["adjacent", "opposite", random.choice(["adjacent", "opposite"])]
    else:
        kinds = ["adjacent", "adjacent", "opposite", "opposite"]
    random.shuffle(kinds)
    return kinds


def _build_widget(kind: str, colors: list[int]) -> dict[str, Any]:
    sides = list(random.choice(ADJACENT_PAIRS if kind == "adjacent" else OPPOSITE_PAIRS))
    random.shuffle(sides)
    side_to_color = {sides[0]: colors[0], sides[1]: colors[1]}
    lengths = {side: random.randint(1, 4) for side in sides}
    input_side = random.choice(sides)
    output_side = sides[1] if input_side == sides[0] else sides[0]
    center_color = side_to_color[input_side]
    output_color = side_to_color[output_side]

    cells: list[tuple[int, int, int]] = []
    for r in range(3):
        for c in range(3):
            color = center_color if (r, c) == (1, 1) else GREY
            cells.append((r, c, color))
    for side in sides:
        for r, c in _stub_cells(side, lengths[side]):
            cells.append((r, c, side_to_color[side]))

    min_r = min(r for r, _c, _color in cells)
    min_c = min(c for _r, c, _color in cells)
    normalized = [(r - min_r, c - min_c, color) for r, c, color in cells]
    height = max(r for r, _c, _color in normalized) + 1
    width = max(c for _r, c, _color in normalized) + 1
    station_top = -min_r
    station_left = -min_c

    return {
        "cells": normalized,
        "height": height,
        "width": width,
        "input_side": input_side,
        "output_side": output_side,
        "output_color": output_color,
        "station_top": station_top,
        "station_left": station_left,
    }


def _place_widgets(widgets: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    canvas = np.full((BASE_SIZE, BASE_SIZE), BG, dtype=int)
    slots = random.sample(
        [(0, 0), (0, SLOT_PITCH), (SLOT_PITCH, 0), (SLOT_PITCH, SLOT_PITCH)],
        k=len(widgets),
    )
    specs: list[dict[str, int | str]] = []

    for widget, (slot_r, slot_c) in zip(widgets, slots):
        max_r_off = SLOT_SIZE - int(widget["height"])
        max_c_off = SLOT_SIZE - int(widget["width"])
        offset_r = random.randint(0, max_r_off)
        offset_c = random.randint(0, max_c_off)
        top = slot_r + offset_r
        left = slot_c + offset_c

        for dr, dc, color in widget["cells"]:
            rr = top + int(dr)
            cc = left + int(dc)
            assert canvas[rr, cc] == BG
            canvas[rr, cc] = int(color)

        specs.append(
            {
                "top": top + int(widget["station_top"]),
                "left": left + int(widget["station_left"]),
                "input_side": str(widget["input_side"]),
                "output_side": str(widget["output_side"]),
                "output_color": int(widget["output_color"]),
            }
        )

    return canvas, specs


def _crop_canvas(
    canvas: np.ndarray, specs: list[dict[str, int | str]]
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    filled = np.argwhere(canvas != BG)
    r0, c0 = filled.min(axis=0)
    r1, c1 = filled.max(axis=0)

    top = int(r0) - random.randint(0, min(3, int(r0)))
    left = int(c0) - random.randint(0, min(3, int(c0)))
    bottom = int(r1) + random.randint(0, min(3, int(canvas.shape[0] - 1 - r1)))
    right = int(c1) + random.randint(0, min(3, int(canvas.shape[1] - 1 - c1)))

    cropped = canvas[top : bottom + 1, left : right + 1]
    shifted_specs: list[dict[str, int | str]] = []
    for spec in specs:
        shifted_specs.append(
            {
                "top": int(spec["top"]) - top,
                "left": int(spec["left"]) - left,
                "input_side": str(spec["input_side"]),
                "output_side": str(spec["output_side"]),
                "output_color": int(spec["output_color"]),
            }
        )
    return cropped, shifted_specs


def _expected_output(grid: np.ndarray, specs: list[dict[str, int | str]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        for rr, cc in _station_path_cells(
            int(spec["top"]),
            int(spec["left"]),
            str(spec["input_side"]),
            str(spec["output_side"]),
        ):
            out[rr, cc] = int(spec["output_color"])
    return out


def _generate_case() -> np.ndarray | None:
    station_count = random.randint(2, 4)
    kinds = _widget_kinds(station_count)
    colors = random.sample(PATH_COLORS, k=2 * station_count)
    widgets = [
        _build_widget(kinds[i], colors[2 * i : 2 * i + 2]) for i in range(station_count)
    ]

    canvas, specs = _place_widgets(widgets)
    grid, shifted_specs = _crop_canvas(canvas, specs)
    found = _find_stations(grid)
    if len(found) != station_count:
        return None

    expected = _expected_output(grid, shifted_specs)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    while True:
        grid = _generate_case()
        if grid is not None:
            return grid
