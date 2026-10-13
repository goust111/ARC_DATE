from __future__ import annotations

import random

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False

    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return bool(
        mask[0, :].all()
        and mask[-1, :].all()
        and mask[:, 0].all()
        and mask[:, -1].all()
        and not mask[1:-1, 1:-1].any()
    )


def _find_legend(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[list[int]] = []

    for coords in _components(g != 0):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1:
            continue
        width = c1 - c0 + 1
        if width < 2 or int(coords.shape[0]) != width:
            continue
        if r0 not in (0, h - 1):
            continue
        if c0 != 0 and c1 != w - 1:
            continue

        run = g[r0, c0 : c1 + 1]
        if np.any(run == 0) or len(np.unique(run)) != width:
            continue
        candidates.append([int(v) for v in run.tolist()])

    assert len(candidates) == 1, "expected exactly one corner color-cycle legend"
    return candidates[0]


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []

    for color in np.unique(g):
        color = int(color)
        if color == 0:
            continue
        for coords in _components(g == color):
            if not _is_frame_component(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.any(inner != 0):
                continue
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})

    assert frames, "expected at least one rectangular frame"
    colors = [int(frame["color"]) for frame in frames]
    assert len(colors) == len(set(colors)), "frame colors must be distinct"
    frames.sort(key=lambda frame: tuple(int(v) for v in frame["bbox"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed inner pattern by cycling colors according to the frame-border color.

    Input structure
    - The background is black.
    - Several disjoint 1-pixel-thick rectangular frames appear in distinct colors.
    - Each frame contains a multicolor inner pattern that does not touch the frame.
    - A corner legend row lists a cyclic color order and includes every frame/pattern color.

    Rule
    - Read the legend row as the cycle c_0, c_1, ..., c_{N-1}.
    - For each frame, find the index k of its border color in that cycle.
    - Shift every non-black color inside that frame forward by k positions modulo N.
    - Leave the background, all frame borders, and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cycle = _find_legend(g)
    color_to_idx = {color: i for i, color in enumerate(cycle)}
    out = g.copy()

    for frame in _find_frames(g):
        border_color = int(frame["color"])
        assert border_color in color_to_idx, "frame color must appear in legend"
        shift = color_to_idx[border_color]
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        recolored = inner.copy()

        for rr, cc in np.argwhere(inner != 0):
            color = int(inner[rr, cc])
            assert color in color_to_idx, "pattern colors must appear in legend"
            recolored[rr, cc] = cycle[(color_to_idx[color] + shift) % len(cycle)]

        out[r0 + 1 : r1, c0 + 1 : c1] = recolored

    return out


# --------------- generator (o5x) ---------------

_PALETTE = tuple(range(1, 10))
_GEN_RETRIES = 400

_GRID_H_RANGE = (18, 30)
_GRID_W_RANGE = (20, 32)
_FRAME_COUNT_RANGE = (2, 4)
_FRAME_H_RANGE = (6, 10)
_FRAME_W_RANGE = (6, 10)
_PATTERN_COLOR_RANGE = (2, 4)
_PATTERN_CELL_RANGE = (4, 8)
_PLACEMENT_PADDING = 1


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _sample_legend(frame_count: int) -> list[int]:
    lo = max(4, frame_count + 1)
    hi = min(len(_PALETTE), frame_count + 4)
    return random.sample(list(_PALETTE), random.randint(lo, hi))


def _legend_layout(h: int, w: int, length: int) -> tuple[int, int]:
    vertical = random.choice(("top", "bottom"))
    horizontal = random.choice(("left", "right"))
    row = 0 if vertical == "top" else h - 1
    start = 0 if horizontal == "left" else w - length
    return row, start


def _reserve_legend_area(h: int, w: int, row: int, start: int, length: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    end = start + length - 1
    rr0 = 0 if row == 0 else h - 2
    rr1 = 2 if row == 0 else h
    cc0 = max(0, start - 1)
    cc1 = min(w, end + 2)
    mask[rr0:rr1, cc0:cc1] = True
    return mask


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rr0 = max(0, r - _PLACEMENT_PADDING)
    rr1 = min(occupied.shape[0], r + h + _PLACEMENT_PADDING)
    cc0 = max(0, c - _PLACEMENT_PADDING)
    cc1 = min(occupied.shape[1], c + w + _PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    rr0 = max(0, r - _PLACEMENT_PADDING)
    rr1 = min(occupied.shape[0], r + h + _PLACEMENT_PADDING)
    cc0 = max(0, c - _PLACEMENT_PADDING)
    cc1 = min(occupied.shape[1], c + w + _PLACEMENT_PADDING)
    occupied[rr0:rr1, cc0:cc1] = True


def _pattern_cells(height: int, width: int) -> list[tuple[int, int]]:
    zone_h = height - 4
    zone_w = width - 4
    all_cells = [(r, c) for r in range(zone_h) for c in range(zone_w)]
    target = random.randint(_PATTERN_CELL_RANGE[0], min(_PATTERN_CELL_RANGE[1], len(all_cells)))

    cells = {random.choice(all_cells)}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < zone_h and 0 <= nc < zone_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pool = list(frontier) if frontier else [cell for cell in all_cells if cell not in cells]
        cells.add(random.choice(pool))

    return [(r + 2, c + 2) for r, c in cells]


def _assign_pattern_colors(
    cycle: list[int],
    border_colors: list[int],
    patterns: list[dict],
) -> None:
    used = set(border_colors)

    for pattern in patterns:
        border_color = int(pattern["color"])
        max_colors = min(_PATTERN_COLOR_RANGE[1], len(cycle), len(pattern["cells"]))
        min_colors = min(_PATTERN_COLOR_RANGE[0], max_colors)
        color_count = random.randint(min_colors, max_colors)

        pool = [color for color in cycle if color != border_color]
        if len(pool) < color_count:
            pool = list(cycle)
        palette = random.sample(pool, color_count)
        colors = [random.choice(palette) for _ in pattern["cells"]]
        if color_count > 1:
            for i, color in enumerate(palette):
                colors[i] = color

        pattern["pattern_colors"] = [int(color) for color in colors]
        used.update(int(color) for color in colors)

    remaining = [color for color in cycle if color not in used]
    slots = [
        (pattern_index, cell_index)
        for pattern_index, pattern in enumerate(patterns)
        for cell_index in range(len(pattern["cells"]))
    ]
    random.shuffle(slots)

    assert len(remaining) <= len(slots)
    for color, (pattern_index, cell_index) in zip(remaining, slots):
        patterns[pattern_index]["pattern_colors"][cell_index] = int(color)


def _render_input(
    h: int,
    w: int,
    cycle: list[int],
    row: int,
    start: int,
    frames: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[row, start : start + len(cycle)] = np.array(cycle, dtype=int)

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        for (rr, cc), cell_color in zip(frame["cells"], frame["pattern_colors"]):
            grid[r0 + rr, c0 + cc] = int(cell_color)

    return grid


def _expected_output(inp: np.ndarray, cycle: list[int], frames: list[dict]) -> np.ndarray:
    out = inp.copy()
    color_to_idx = {color: i for i, color in enumerate(cycle)}
    for frame in frames:
        shift = color_to_idx[int(frame["color"])]
        r0, c0, _r1, _c1 = frame["bbox"]
        for (rr, cc), cell_color in zip(frame["cells"], frame["pattern_colors"]):
            out[r0 + rr, c0 + cc] = cycle[(color_to_idx[int(cell_color)] + shift) % len(cycle)]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = _randint(_GRID_H_RANGE)
    w = _randint(_GRID_W_RANGE)
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    cycle = _sample_legend(frame_count)
    row, start = _legend_layout(h, w, len(cycle))

    occupied = _reserve_legend_area(h, w, row, start, len(cycle))
    border_colors = random.sample(cycle, frame_count)
    frames: list[dict] = []

    for border_color in border_colors:
        placed = False
        for _ in range(80):
            fh = _randint(_FRAME_H_RANGE)
            fw = _randint(_FRAME_W_RANGE)
            if fh >= h - 2 or fw >= w - 2:
                continue

            r_lo = 1
            r_hi = h - fh - 2
            c_lo = 1
            c_hi = w - fw - 2
            if r_lo > r_hi or c_lo > c_hi:
                continue

            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            if not _can_place(occupied, r0, c0, fh, fw):
                continue

            _mark_occupied(occupied, r0, c0, fh, fw)
            frames.append(
                {
                    "color": int(border_color),
                    "bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                    "cells": _pattern_cells(fh, fw),
                }
            )
            placed = True
            break
        if not placed:
            return None

    _assign_pattern_colors(cycle, border_colors, frames)
    inp = _render_input(h, w, cycle, row, start, frames)
    expected = _expected_output(inp, cycle, frames)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
