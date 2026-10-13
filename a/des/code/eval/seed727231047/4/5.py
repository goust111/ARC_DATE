from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]

MIN_LEN = 3
MAX_LEN = 8
STRIP_COUNT_RANGE = (3, 6)
INPUT_SIDE_BASE = (14, 24)
MAX_OUTPUT_SPAN = 20
TIE_GROUP_PROB = 0.4
MAX_GENERATION_ATTEMPTS = 400


def _rotate_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _axis_dirs(axis: str) -> list[tuple[int, int]]:
    return [(0, 1), (0, -1)] if axis == "H" else [(1, 0), (-1, 0)]


def _other_axis(axis: str) -> str:
    return "V" if axis == "H" else "H"


def _extract_strips(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    strips: list[dict] = []
    used_markers: set[tuple[int, int]] = set()

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in {BG, MAROON}):
        coords = np.argwhere(g == color)
        assert len(coords) >= 2
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1:
            row = int(rows[0])
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            assert len(coords) == c1 - c0 + 1
            assert np.array_equal(np.sort(coords[:, 1]), np.arange(c0, c1 + 1))

            marker_candidates = []
            if c0 - 1 >= 0 and int(g[row, c0 - 1]) == MAROON:
                marker_candidates.append((row, c0 - 1, (0, 1), c0 - 1, c1))
            if c1 + 1 < w and int(g[row, c1 + 1]) == MAROON:
                marker_candidates.append((row, c1 + 1, (0, -1), c0, c1 + 1))
            assert len(marker_candidates) == 1

            mr, mc, input_dir, full_c0, full_c1 = marker_candidates[0]
            length = int(full_c1 - full_c0 + 1)
            strips.append(
                {
                    "color": color,
                    "axis": "H",
                    "input_dir": input_dir,
                    "length": length,
                    "top": row,
                    "left": int(full_c0),
                    "marker": (int(mr), int(mc)),
                }
            )
            used_markers.add((int(mr), int(mc)))
        else:
            assert len(cols) == 1
            col = int(cols[0])
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            assert len(coords) == r1 - r0 + 1
            assert np.array_equal(np.sort(coords[:, 0]), np.arange(r0, r1 + 1))

            marker_candidates = []
            if r0 - 1 >= 0 and int(g[r0 - 1, col]) == MAROON:
                marker_candidates.append((r0 - 1, col, (1, 0), r0 - 1, r1))
            if r1 + 1 < h and int(g[r1 + 1, col]) == MAROON:
                marker_candidates.append((r1 + 1, col, (-1, 0), r0, r1 + 1))
            assert len(marker_candidates) == 1

            mr, mc, input_dir, full_r0, full_r1 = marker_candidates[0]
            length = int(full_r1 - full_r0 + 1)
            strips.append(
                {
                    "color": color,
                    "axis": "V",
                    "input_dir": input_dir,
                    "length": length,
                    "top": int(full_r0),
                    "left": col,
                    "marker": (int(mr), int(mc)),
                }
            )
            used_markers.add((int(mr), int(mc)))

        assert MIN_LEN <= strips[-1]["length"] <= MAX_LEN

    maroons = {tuple(map(int, pt)) for pt in np.argwhere(g == MAROON)}
    assert maroons == used_markers

    strips.sort(key=lambda strip: (-strip["length"], strip["top"], strip["left"]))
    return strips


def _connected_segments(strips: list[dict]) -> list[dict]:
    placed: list[dict] = []
    prev_axis: str | None = None
    prev_head: tuple[int, int] | None = None

    for strip in strips:
        if prev_axis is None or strip["axis"] != prev_axis:
            out_dir = strip["input_dir"]
        else:
            out_dir = _rotate_cw(strip["input_dir"])

        if prev_head is None:
            tail = (0, 0)
        else:
            tail = (prev_head[0] + out_dir[0], prev_head[1] + out_dir[1])

        cells = [
            (tail[0] + out_dir[0] * step, tail[1] + out_dir[1] * step)
            for step in range(int(strip["length"]))
        ]
        placed.append({"color": int(strip["color"]), "cells": cells, "output_dir": out_dir})
        prev_head = cells[-1]
        prev_axis = str(strip["axis"])

    return placed


def _render_connected(placed: list[dict]) -> np.ndarray:
    all_cells = [cell for seg in placed for cell in seg["cells"]]
    assert all_cells
    assert len(set(all_cells)) == len(all_cells)

    r0 = min(r for r, _ in all_cells)
    r1 = max(r for r, _ in all_cells)
    c0 = min(c for _, c in all_cells)
    c1 = max(c for _, c in all_cells)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for seg in placed:
        color = int(seg["color"])
        for r, c in seg["cells"]:
            rr = r - r0
            cc = c - c0
            assert int(out[rr, cc]) == BG
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the colored strips into a single connected snake.

    The hidden order is: longer strips first, ties by row-major position of the
    strip's full extent. Each strip is directed from its maroon endpoint toward
    its unmarked endpoint. When two consecutive input strips share the same axis,
    the later strip is turned by a fixed clockwise quarter-turn before being
    attached; otherwise it keeps its original direction.
    """
    strips = _extract_strips(grid)
    placed = _connected_segments(strips)
    return _render_connected(placed)


def _path_is_clear(existing_path: list[tuple[int, int]], new_cells: list[tuple[int, int]]) -> bool:
    if len(set(new_cells)) != len(new_cells):
        return False

    old_set = set(existing_path)
    last_old = existing_path[-1] if existing_path else None
    for j, (r, c) in enumerate(new_cells):
        if (r, c) in old_set:
            return False
        for idx, (rr, cc) in enumerate(existing_path):
            if abs(r - rr) + abs(c - cc) != 1:
                continue
            if j == 0 and idx == len(existing_path) - 1 and abs(r - rr) + abs(c - cc) == 1:
                continue
            return False

    if last_old is not None and abs(new_cells[0][0] - last_old[0]) + abs(new_cells[0][1] - last_old[1]) != 1:
        return False
    return True


def _sample_lengths(count: int) -> list[int]:
    lengths = [random.randint(MIN_LEN, MAX_LEN) for _ in range(count)]
    if count >= 2 and random.random() < TIE_GROUP_PROB:
        group_size = random.randint(2, min(3, count))
        chosen = random.sample(range(count), group_size)
        shared = random.randint(MIN_LEN, MAX_LEN)
        for idx in chosen:
            lengths[idx] = shared
    lengths.sort(reverse=True)
    return lengths


def _sample_axes(count: int) -> list[str]:
    relations = [random.choice([False, True]) for _ in range(max(0, count - 1))]
    if count >= 3 and (all(relations) or not any(relations)):
        flip = random.randrange(len(relations))
        relations[flip] = not relations[flip]

    axes = [random.choice(["H", "V"])]
    for same_as_prev in relations:
        axes.append(axes[-1] if same_as_prev else _other_axis(axes[-1]))
    return axes


def _build_random_specs() -> list[dict] | None:
    count = random.randint(*STRIP_COUNT_RANGE)
    lengths = _sample_lengths(count)
    axes = _sample_axes(count)
    colors = random.sample(PALETTE, count)

    path: list[tuple[int, int]] = []
    specs: list[dict] = []
    prev_axis: str | None = None
    prev_head: tuple[int, int] | None = None

    for length, axis, color in zip(lengths, axes, colors):
        candidates = _axis_dirs(axis)
        random.shuffle(candidates)
        valid = []
        for input_dir in candidates:
            out_dir = input_dir if prev_axis is None or axis != prev_axis else _rotate_cw(input_dir)
            if prev_head is None:
                tail = (0, 0)
            else:
                tail = (prev_head[0] + out_dir[0], prev_head[1] + out_dir[1])
            cells = [
                (tail[0] + out_dir[0] * step, tail[1] + out_dir[1] * step)
                for step in range(length)
            ]
            if not _path_is_clear(path, cells):
                continue

            trial = path + cells
            rs = [r for r, _ in trial]
            cs = [c for _, c in trial]
            if (max(rs) - min(rs) + 1) > MAX_OUTPUT_SPAN or (max(cs) - min(cs) + 1) > MAX_OUTPUT_SPAN:
                continue
            valid.append((input_dir, cells))

        if not valid:
            return None

        input_dir, cells = random.choice(valid)
        specs.append(
            {
                "color": color,
                "length": length,
                "axis": axis,
                "input_dir": input_dir,
            }
        )
        path.extend(cells)
        prev_axis = axis
        prev_head = cells[-1]

    return specs


def _make_patch(strip: dict) -> np.ndarray:
    length = int(strip["length"])
    color = int(strip["color"])
    dr, dc = strip["input_dir"]

    if strip["axis"] == "H":
        patch = np.full((1, length), color, dtype=int)
        patch[0, 0 if dc > 0 else length - 1] = MAROON
    else:
        patch = np.full((length, 1), color, dtype=int)
        patch[0 if dr > 0 else length - 1, 0] = MAROON
    return patch


def _can_place(forbidden: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    ph, pw = patch.shape
    return not bool(np.any(forbidden[r0 : r0 + ph, c0 : c0 + pw]))


def _paint_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    for rr, cc in np.argwhere(patch != BG):
        r = int(r0 + rr)
        c = int(c0 + cc)
        forbidden[max(0, r - 1) : min(grid.shape[0], r + 2), max(0, c - 1) : min(grid.shape[1], c + 2)] = True


def _scatter_input(specs: list[dict]) -> np.ndarray | None:
    max_len = max(int(strip["length"]) for strip in specs)
    side_floor = max(INPUT_SIDE_BASE[0], max_len + 4, len(specs) * 2 + 6)
    side_ceil = max(INPUT_SIDE_BASE[1], side_floor + 2)
    height = random.randint(side_floor, side_ceil)
    width = random.randint(side_floor, side_ceil)

    grid = np.full((height, width), BG, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    idx = 0
    while idx < len(specs):
        group_end = idx + 1
        while group_end < len(specs) and int(specs[group_end]["length"]) == int(specs[idx]["length"]):
            group_end += 1

        prev_key = (-1, -1)
        for strip in specs[idx:group_end]:
            patch = _make_patch(strip)
            ph, pw = patch.shape
            candidates = [
                (r, c)
                for r in range(height - ph + 1)
                for c in range(width - pw + 1)
                if (r, c) > prev_key and _can_place(forbidden, patch, r, c)
            ]
            if not candidates:
                return None

            r0, c0 = random.choice(candidates)
            _paint_patch(grid, forbidden, patch, r0, c0)
            prev_key = (r0, c0)

        idx = group_end

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        specs = _build_random_specs()
        if specs is None:
            continue

        inp = _scatter_input(specs)
        if inp is None:
            continue

        try:
            expected = _render_connected(_connected_segments(specs))
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid strip-snake input")
