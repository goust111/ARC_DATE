from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load arc_utils from {_ARC_UTILS_PATH}")
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg


MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _parse_instruction_row(row: np.ndarray) -> tuple[str, int | None, int | None]:
    if np.all(row == Color.BLACK):
        return "blank", None, None
    if int(row[0]) == Color.BLACK:
        return "invalid", None, None

    color = int(row[0])
    length = 0
    width = int(row.shape[0])
    while length < width and int(row[length]) == color:
        length += 1

    if length >= width:
        return "invalid", None, None
    if np.any(row[length:] != Color.BLACK):
        return "invalid", None, None
    return "bar", color, length


def _try_parse_layout(grid: np.ndarray, panel_top: int) -> tuple[int, dict[int, int]] | None:
    top = grid[:panel_top, :]
    panel = grid[panel_top:, :]
    if top.size == 0 or panel.size == 0:
        return None
    if np.any(panel == Color.BLACK):
        return None

    panel_bg = int(get_bg(panel))
    if panel_bg == Color.BLACK or not np.any(panel != panel_bg):
        return None

    bar_lengths: dict[int, int] = {}
    used_lengths: set[int] = set()
    for row in top:
        kind, color, length = _parse_instruction_row(row)
        if kind == "invalid":
            return None
        if kind == "bar":
            assert color is not None and length is not None
            if color in bar_lengths or length in used_lengths:
                return None
            bar_lengths[int(color)] = int(length)
            used_lengths.add(int(length))

    if not bar_lengths:
        return None
    return panel_bg, bar_lengths


def _parse_layout(grid: np.ndarray) -> tuple[int, int, dict[int, int]]:
    candidates: list[tuple[int, int, dict[int, int]]] = []
    for panel_top in range(1, int(grid.shape[0])):
        parsed = _try_parse_layout(grid, panel_top)
        if parsed is not None:
            panel_bg, bar_lengths = parsed
            candidates.append((int(panel_top), int(panel_bg), bar_lengths))
    assert candidates, "expected a black instruction zone above a non-black scene panel"
    return candidates[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only panel objects whose area matches the same-color instruction-bar length.

    Input structure:
    - The top instruction zone has black background and contains one-cell-thick horizontal color bars that start in column 0.
    - Each instruction color appears at most once, and the different bar lengths are all distinct.
    - Below it is a scene panel with its own non-black background color and multiple 4-connected colored objects.

    Rule:
    - Preserve the instruction zone unchanged.
    - In the panel, inspect each monochromatic 4-connected object independently.
    - Keep an object iff its color has a matching instruction bar and the object's area equals that bar's length; otherwise repaint that object with the panel background color.

    Key insight:
    - The top bars are forward instructions about allowable object sizes, not a report about how many objects of each color exist below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    panel_top, panel_bg, bar_lengths = _parse_layout(g)
    panel = g[panel_top:, :]
    components = find_components(panel, bg=panel_bg, connectivity=4, monochromatic=True)
    assert components, "expected at least one panel object"

    out = g.copy()
    for comp in components:
        color = int(comp["color"])
        area = int(comp["area"])
        if bar_lengths.get(color) == area:
            continue
        coords = comp["coords"]
        out[panel_top + coords[:, 0], coords[:, 1]] = panel_bg

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cells)


def _random_connected_shape(area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    if random.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    for _ in range(random.randint(0, 3)):
        cells = {(c, -r) for r, c in cells}
    return _normalize_cells(cells)


def _shape_extent(cells: list[tuple[int, int]]) -> tuple[int, int]:
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    return int(height), int(width)


def _place_shape(
    panel: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    height, width = _shape_extent(cells)
    panel_h, panel_w = map(int, panel.shape)
    top_positions = list(range(1, panel_h - height))
    left_positions = list(range(1, panel_w - width))
    if not top_positions or not left_positions:
        return None

    positions = [(top, left) for top in top_positions for left in left_positions]
    random.shuffle(positions)
    for top, left in positions:
        if blocked[top - 1 : top + height + 1, left - 1 : left + width + 1].any():
            continue
        blocked[top - 1 : top + height + 1, left - 1 : left + width + 1] = True
        placed: list[tuple[int, int]] = []
        for dr, dc in cells:
            rr = int(top + dr)
            cc = int(left + dc)
            panel[rr, cc] = int(color)
            placed.append((rr, cc))
        return placed
    return None


def _sample_mismatch_area(target: int, max_area: int) -> int:
    choices = [area for area in range(1, int(max_area) + 1) if area != int(target)]
    return int(random.choice(choices))


def _plan_objects(
    bar_map: dict[int, int],
    extra_colors: list[int],
    max_area: int,
) -> tuple[list[dict[str, int | bool | str]], int]:
    bar_items = list(bar_map.items())
    random.shuffle(bar_items)
    primary_color, primary_len = bar_items[0]
    secondary_color, secondary_len = bar_items[1]

    specs: list[dict[str, int | bool | str]] = []
    primary_match_count = 2 if random.random() < 0.5 else 1
    for _ in range(primary_match_count):
        specs.append({"color": primary_color, "area": primary_len, "keep": True, "reason": "match"})
    specs.append(
        {
            "color": primary_color,
            "area": _sample_mismatch_area(primary_len, max_area),
            "keep": False,
            "reason": "size_mismatch",
        }
    )

    secondary_mismatch_count = 2 if random.random() < 0.4 else 1
    for _ in range(secondary_mismatch_count):
        specs.append(
            {
                "color": secondary_color,
                "area": _sample_mismatch_area(secondary_len, max_area),
                "keep": False,
                "reason": "size_mismatch",
            }
        )

    for color, length in bar_items[2:]:
        if random.random() < 0.65:
            keep = random.random() < 0.55
            specs.append(
                {
                    "color": color,
                    "area": length if keep else _sample_mismatch_area(length, max_area),
                    "keep": bool(keep),
                    "reason": "match" if keep else "size_mismatch",
                }
            )
        if random.random() < 0.35:
            specs.append(
                {
                    "color": color,
                    "area": _sample_mismatch_area(length, max_area),
                    "keep": False,
                    "reason": "size_mismatch",
                }
            )

    extra_count = min(len(extra_colors), 1 if random.random() < 0.55 else 2)
    for color in random.sample(extra_colors, extra_count):
        specs.append(
            {
                "color": color,
                "area": random.randint(1, max_area),
                "keep": False,
                "reason": "no_bar",
            }
        )
        if random.random() < 0.3:
            specs.append(
                {
                    "color": color,
                    "area": random.randint(1, max_area),
                    "keep": False,
                    "reason": "no_bar",
                }
            )

    random.shuffle(specs)
    return specs, int(primary_color)


def _build_expected(grid: np.ndarray, placed_objects: list[dict], panel_bg: int) -> np.ndarray:
    expected = grid.copy()
    for obj in placed_objects:
        if obj["keep"]:
            continue
        coords = np.asarray(obj["cells"], dtype=int)
        expected[coords[:, 0], coords[:, 1]] = int(panel_bg)
    return expected


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(15, 24)
    bar_count = random.randint(2, 4)
    max_bar_length = min(9, width - 2)
    possible_lengths = list(range(2, max_bar_length + 1))
    if len(possible_lengths) < bar_count:
        return None

    bar_colors = random.sample(list(PALETTE), bar_count)
    bar_lengths = random.sample(possible_lengths, bar_count)
    bar_map = {int(color): int(length) for color, length in zip(bar_colors, bar_lengths)}

    panel_bg_choices = [color for color in PALETTE if color not in bar_map]
    panel_bg = int(random.choice(panel_bg_choices))
    extra_colors = [color for color in PALETTE if color not in bar_map and color != panel_bg]
    max_area = min(9, max_bar_length + 1)
    specs, primary_color = _plan_objects(bar_map, extra_colors, max_area)

    top_h = random.randint(bar_count, min(6, 30 - 10))
    panel_h = random.randint(max(10, 5 + len(specs)), min(18, 30 - top_h))
    grid = np.full((top_h + panel_h, width), panel_bg, dtype=int)
    grid[:top_h, :] = Color.BLACK

    bar_rows = sorted(random.sample(range(top_h), bar_count))
    bar_items = list(bar_map.items())
    random.shuffle(bar_items)
    for row, (color, length) in zip(bar_rows, bar_items):
        grid[row, :length] = int(color)

    panel = grid[top_h:, :]
    blocked = np.zeros(panel.shape, dtype=bool)
    placed_objects: list[dict] = []
    for spec in sorted(specs, key=lambda item: int(item["area"]), reverse=True):
        cells = _random_connected_shape(int(spec["area"]))
        placed = _place_shape(panel, blocked, cells, int(spec["color"]))
        if placed is None:
            return None
        placed_objects.append(
            {
                "color": int(spec["color"]),
                "area": int(spec["area"]),
                "keep": bool(spec["keep"]),
                "reason": str(spec["reason"]),
                "cells": [(top_h + r, c) for r, c in placed],
            }
        )

    if int(get_bg(panel)) != panel_bg:
        return None

    actual_components = find_components(panel, bg=panel_bg, connectivity=4, monochromatic=True)
    actual_signature = sorted((int(comp["color"]), int(comp["area"])) for comp in actual_components)
    planned_signature = sorted((int(obj["color"]), int(obj["area"])) for obj in placed_objects)
    if actual_signature != planned_signature:
        return None

    if not any(bool(obj["keep"]) for obj in placed_objects):
        return None
    if not any(obj["reason"] == "size_mismatch" for obj in placed_objects):
        return None
    if not any(obj["reason"] == "no_bar" for obj in placed_objects):
        return None
    if sum(1 for obj in placed_objects if int(obj["color"]) == primary_color) < 2:
        return None

    expected = _build_expected(grid, placed_objects, panel_bg)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 044/5")
