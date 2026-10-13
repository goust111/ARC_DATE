from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
AZURE = 8
YELLOW = 4

MARKER_TO_MODE = {
    ORANGE: "orange",
    AZURE: "azure",
    YELLOW: "yellow",
}

TOWER_COLORS = [1, 2, 3, 6, 9]
GEN_ATTEMPTS = 400
DIFF_CHOICES = [7, 11, 15]
MARGINS = [1, 2, 3]
EDGES = ("top", "bottom", "left", "right")


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_towers(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    reserved = {bg, GREY, ORANGE, AZURE, YELLOW}
    towers: dict[int, dict] = {}
    for color in map(int, np.unique(g)):
        if color in reserved:
            continue
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 4, "each tower color must appear exactly four times"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert (r1 - r0, c1 - c0) == (1, 1), "tower must be a 2x2 block"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "tower block must be solid"
        towers[color] = {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "top_left": (r0, c0),
        }
    assert len(towers) >= 2, "expected at least two towers"

    ordered = sorted(towers.values(), key=lambda tower: tower["top_left"])
    for rank, tower in enumerate(ordered):
        tower["rank"] = rank
    return towers


def _trace_to_tower(
    grid: np.ndarray,
    towers: dict[int, dict],
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    h, w = grid.shape
    r, c = start[0] + dr, start[1] + dc
    seen: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == GREY:
            seen.append((r, c))
            r += dr
            c += dc
            continue
        if value in towers:
            return value, seen
        return None
    return None


def _line_cells(
    marker: tuple[int, int],
    orientation: str,
    tower_a: dict,
    tower_b: dict,
) -> list[tuple[int, int]]:
    r, c = marker
    if orientation == "horizontal":
        left, right = sorted([tower_a, tower_b], key=lambda tower: tower["bbox"][1])
        start = int(left["bbox"][3]) + 1
        end = int(right["bbox"][1]) - 1
        cells = [(r, cc) for cc in range(start, end + 1)]
        assert cells and cells[len(cells) // 2] == marker
        return cells

    top, bottom = sorted([tower_a, tower_b], key=lambda tower: tower["bbox"][0])
    start = int(top["bbox"][2]) + 1
    end = int(bottom["bbox"][0]) - 1
    cells = [(rr, c) for rr in range(start, end + 1)]
    assert cells and cells[len(cells) // 2] == marker
    return cells


def _connection_from_marker(
    grid: np.ndarray,
    towers: dict[int, dict],
    marker: tuple[int, int],
) -> dict:
    r, c = marker
    value = int(grid[r, c])
    assert value in MARKER_TO_MODE, "unexpected marker color"

    candidates = []
    for orientation, dirs in (
        ("horizontal", ((0, -1), (0, 1))),
        ("vertical", ((-1, 0), (1, 0))),
    ):
        traced = []
        ok = True
        for dr, dc in dirs:
            info = _trace_to_tower(grid, towers, marker, dr, dc)
            if info is None:
                ok = False
                break
            traced.append(info)
        if not ok:
            continue

        tower_colors = [info[0] for info in traced]
        if tower_colors[0] == tower_colors[1]:
            continue

        tower_a = towers[tower_colors[0]]
        tower_b = towers[tower_colors[1]]
        cells = _line_cells(marker, orientation, tower_a, tower_b)
        allowed = {GREY, value}
        for rr, cc in cells:
            assert int(grid[rr, cc]) in allowed, "line may only contain grey plus its marker"

        if orientation == "horizontal":
            for nr in (r - 1, r + 1):
                if 0 <= nr < grid.shape[0]:
                    assert int(grid[nr, c]) == BG, "marker orientation must be unambiguous"
            ordered = sorted([tower_a, tower_b], key=lambda tower: tower["bbox"][1])
        else:
            for nc in (c - 1, c + 1):
                if 0 <= nc < grid.shape[1]:
                    assert int(grid[r, nc]) == BG, "marker orientation must be unambiguous"
            ordered = sorted([tower_a, tower_b], key=lambda tower: tower["bbox"][0])

        candidates.append(
            {
                "mode": MARKER_TO_MODE[value],
                "orientation": orientation,
                "cells": cells,
                "tower_a": ordered[0],
                "tower_b": ordered[1],
            }
        )

    assert len(candidates) == 1, "marker must belong to exactly one straight connection"
    return candidates[0]


def _paint_connection(out: np.ndarray, connection: dict) -> None:
    cells = connection["cells"]
    tower_a = connection["tower_a"]
    tower_b = connection["tower_b"]
    color_a = int(tower_a["color"])
    color_b = int(tower_b["color"])
    mode = connection["mode"]

    if mode == "orange":
        mid = len(cells) // 2
        for index, (r, c) in enumerate(cells):
            if index < mid:
                out[r, c] = color_a
            elif index > mid:
                out[r, c] = color_b
            else:
                out[r, c] = color_a if int(tower_a["rank"]) < int(tower_b["rank"]) else color_b
        return

    if mode == "azure":
        src = color_a if int(tower_a["rank"]) > int(tower_b["rank"]) else color_b
        for r, c in cells:
            out[r, c] = src
        return

    assert mode == "yellow"
    for index, (r, c) in enumerate(cells):
        out[r, c] = color_a if index % 2 == 0 else color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each colored frequency marker on a grey tower-to-tower connection and
    recolor the line according to that marker's propagation rule.
    """
    g = np.asarray(grid, dtype=int)
    towers = _find_towers(g)
    out = g.copy()

    marker_positions = [
        tuple(map(int, coord))
        for coord in np.argwhere(np.isin(g, np.array(list(MARKER_TO_MODE), dtype=int)))
    ]
    assert marker_positions, "expected at least one frequency marker"

    for marker in marker_positions:
        connection = _connection_from_marker(g, towers, marker)
        _paint_connection(out, connection)
    return out


def _paint_tower(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r, c = top_left
    grid[r : r + 2, c : c + 2] = int(color)


def _edge_cells(
    positions: dict[str, tuple[int, int]],
    edge: str,
    lane: int,
) -> list[tuple[int, int]]:
    if edge == "top":
        r = positions["tl"][0] + lane
        c0 = positions["tl"][1] + 2
        c1 = positions["tr"][1] - 1
        return [(r, c) for c in range(c0, c1 + 1)]
    if edge == "bottom":
        r = positions["bl"][0] + lane
        c0 = positions["bl"][1] + 2
        c1 = positions["br"][1] - 1
        return [(r, c) for c in range(c0, c1 + 1)]
    if edge == "left":
        c = positions["tl"][1] + lane
        r0 = positions["tl"][0] + 2
        r1 = positions["bl"][0] - 1
        return [(r, c) for r in range(r0, r1 + 1)]
    assert edge == "right"
    c = positions["tr"][1] + lane
    r0 = positions["tr"][0] + 2
    r1 = positions["br"][0] - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _draw_connection(
    grid: np.ndarray,
    positions: dict[str, tuple[int, int]],
    edge: str,
    marker_color: int,
    lane: int,
) -> None:
    cells = _edge_cells(positions, edge, lane)
    assert len(cells) >= 5 and len(cells) % 2 == 1, "all generated lines must have an odd, readable length"
    mid = len(cells) // 2
    for index, (r, c) in enumerate(cells):
        grid[r, c] = marker_color if index == mid else GREY


def _all_transforms(grid: np.ndarray) -> list[np.ndarray]:
    rots = [np.rot90(grid, k) for k in range(4)]
    return rots + [np.fliplr(rot) for rot in rots]


def _free_tower_slots(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    slots: list[tuple[int, int]] = []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if np.all(grid[r - 1 : r + 3, c - 1 : c + 3] == BG):
                slots.append((r, c))
    return slots


def _generate_base_scene() -> np.ndarray:
    diff_row = random.choice(DIFF_CHOICES)
    diff_col = random.choice(DIFF_CHOICES)
    top_margin = random.choice(MARGINS)
    bottom_margin = random.choice(MARGINS)
    left_margin = random.choice(MARGINS)
    right_margin = random.choice(MARGINS)

    height = top_margin + diff_row + 2 + bottom_margin
    width = left_margin + diff_col + 2 + right_margin
    grid = np.full((height, width), BG, dtype=int)

    positions = {
        "tl": (top_margin, left_margin),
        "tr": (top_margin, left_margin + diff_col),
        "bl": (top_margin + diff_row, left_margin),
        "br": (top_margin + diff_row, left_margin + diff_col),
    }

    tower_colors = random.sample(TOWER_COLORS, 4)
    for name, color in zip(("tl", "tr", "bl", "br"), tower_colors):
        _paint_tower(grid, positions[name], color)

    omitted = random.choice(EDGES)
    chosen_edges = [edge for edge in EDGES if edge != omitted]
    marker_colors = random.sample([ORANGE, AZURE, YELLOW], 3)
    for edge, marker_color in zip(chosen_edges, marker_colors):
        _draw_connection(
            grid,
            positions,
            edge,
            marker_color=marker_color,
            lane=random.randint(0, 1),
        )

    extra_colors = [color for color in TOWER_COLORS if color not in tower_colors]
    if extra_colors and random.random() < 0.5:
        slots = _free_tower_slots(grid)
        if slots:
            _paint_tower(grid, random.choice(slots), extra_colors[0])

    return random.choice(_all_transforms(grid)).copy()


def _valid_generated_scene(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    marker_count = int(np.isin(grid, np.array([ORANGE, AZURE, YELLOW], dtype=int)).sum())
    if marker_count != 3:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    """
    Build a random tower network with three different frequency markers, one
    for each propagation mode, and reject layouts that violate solver checks.
    """
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_base_scene()
        if _valid_generated_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
