import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})

    return comps


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0, "expected a non-empty palette span"
    start = int(indices[0])
    stop = int(indices[-1])
    assert np.array_equal(indices, np.arange(start, stop + 1)), "palette span must be contiguous"
    return start, stop


def _read_edge_palette(grid: np.ndarray, bg: int, side: str) -> list[int]:
    h, w = grid.shape

    if side == "top":
        mask = (grid[0, :] == grid[1, :]) & (grid[0, :] != bg)
        start, stop = _contiguous_span(np.flatnonzero(mask))
        return [int(v) for v in grid[0, start : stop + 1]]
    if side == "bottom":
        mask = (grid[h - 1, :] == grid[h - 2, :]) & (grid[h - 1, :] != bg)
        start, stop = _contiguous_span(np.flatnonzero(mask))
        return [int(v) for v in grid[h - 1, start : stop + 1]]
    if side == "left":
        mask = (grid[:, 0] == grid[:, 1]) & (grid[:, 0] != bg)
        start, stop = _contiguous_span(np.flatnonzero(mask))
        return [int(v) for v in grid[start : stop + 1, 0]]
    if side == "right":
        mask = (grid[:, w - 1] == grid[:, w - 2]) & (grid[:, w - 1] != bg)
        start, stop = _contiguous_span(np.flatnonzero(mask))
        return [int(v) for v in grid[start : stop + 1, w - 1]]
    raise ValueError(f"unknown side {side}")


def _frame_info(grid: np.ndarray, bg: int, component: dict) -> dict | None:
    h, w = grid.shape
    cells = component["cells"]
    color = int(component["color"])
    r0, c0, r1, c1 = _bbox(cells)

    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    cellset = set(cells)
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if cellset - perimeter:
        return None

    missing = perimeter - cellset
    if len(missing) != 1:
        return None

    gap = next(iter(missing))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if r1 - r0 <= 1 or c1 - c0 <= 1:
        return None
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner != bg):
        return None

    if gap[0] == r0:
        side = "top"
        sort_key = c0 + c1
    elif gap[0] == r1:
        side = "bottom"
        sort_key = c0 + c1
    elif gap[1] == c0:
        side = "left"
        sort_key = r0 + r1
    else:
        side = "right"
        sort_key = r0 + r1

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "sort_key": int(sort_key),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    palettes = {side: _read_edge_palette(g, bg, side) for side in SIDES}

    frames = []
    for comp in _monochrome_components(g, bg):
        info = _frame_info(g, bg, comp)
        if info is not None:
            frames.append(info)

    assert frames, "expected at least one frame"
    sizes = {
        (info["bbox"][2] - info["bbox"][0] + 1, info["bbox"][3] - info["bbox"][1] + 1)
        for info in frames
    }
    assert len(sizes) == 1, "all frames must share the same size"

    frames_by_side = {side: [] for side in SIDES}
    for info in frames:
        frames_by_side[info["gap_side"]].append(info)

    out = g.copy()
    for side in SIDES:
        ordered = sorted(frames_by_side[side], key=lambda item: item["sort_key"])
        sort_keys = [item["sort_key"] for item in ordered]
        assert len(sort_keys) == len(set(sort_keys)), "sorted frames must have unique order keys"
        assert len(ordered) == len(palettes[side]), "palette length must match frame count"

        for info, fill_color in zip(ordered, palettes[side]):
            r0, c0, r1, c1 = info["bbox"]
            inner = out[r0 + 1 : r1, c0 + 1 : c1]
            inner[inner == bg] = int(fill_color)

    return out


def _random_palette_sequence(rng: random.Random, pool: list[int], length: int) -> list[int]:
    seq: list[int] = []
    for _ in range(length):
        choices = [color for color in pool if not seq or color != seq[-1]]
        seq.append(int(rng.choice(choices)))
    return seq


def _draw_frame(
    grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    frame_h: int,
    frame_w: int,
    color: int,
    side: str,
    gap_index: int,
) -> None:
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        grid[r0, c0 + gap_index] = bg
    elif side == "bottom":
        grid[r1, c0 + gap_index] = bg
    elif side == "left":
        grid[r0 + gap_index, c0] = bg
    elif side == "right":
        grid[r0 + gap_index, c1] = bg
    else:
        raise ValueError(f"unknown side {side}")


def _paint_palettes(grid: np.ndarray, palettes: dict[str, list[int]], rng: random.Random) -> None:
    h, w = grid.shape

    for side, seq in palettes.items():
        if side in ("top", "bottom"):
            start = rng.randint(2, w - 2 - len(seq))
            row_a, row_b = (0, 1) if side == "top" else (h - 2, h - 1)
            for offset, color in enumerate(seq):
                c = start + offset
                grid[row_a, c] = color
                grid[row_b, c] = color
        else:
            start = rng.randint(2, h - 2 - len(seq))
            col_a, col_b = (0, 1) if side == "left" else (w - 2, w - 1)
            for offset, color in enumerate(seq):
                r = start + offset
                grid[r, col_a] = color
                grid[r, col_b] = color


def _expected_output(grid: np.ndarray, frames: list[dict], palettes: dict[str, list[int]], bg: int) -> np.ndarray:
    out = grid.copy()
    by_side = {side: [] for side in SIDES}
    for frame in frames:
        by_side[frame["gap_side"]].append(frame)

    for side in SIDES:
        ordered = sorted(by_side[side], key=lambda item: item["sort_key"])
        for frame, fill_color in zip(ordered, palettes[side]):
            r0, c0, r1, c1 = frame["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.randrange(10))
    frame_h = rng.randint(4, 6)
    frame_w = rng.randint(4, 5)

    counts = {side: 1 for side in SIDES}
    extra_frames = rng.randint(0, 3)
    for _ in range(extra_frames):
        choices = [side for side in SIDES if counts[side] < 3]
        counts[rng.choice(choices)] += 1
    n_frames = sum(counts.values())

    non_bg_colors = [color for color in range(10) if color != bg]
    if n_frames > len(non_bg_colors):
        return None

    max_horiz = max(counts["top"], counts["bottom"])
    max_vert = max(counts["left"], counts["right"])
    slot_rows = max(2, max_vert)
    slot_cols = max(2, max_horiz)

    if slot_rows < 3 and rng.random() < 0.5:
        slot_rows += 1
    if slot_cols < 3 and rng.random() < 0.5:
        slot_cols += 1
    while slot_rows * slot_cols < n_frames:
        if slot_rows <= slot_cols and slot_rows < 3:
            slot_rows += 1
        elif slot_cols < 3:
            slot_cols += 1
        elif slot_rows < 3:
            slot_rows += 1
        else:
            return None

    row_gap = rng.randint(1, 2)
    col_gap = rng.randint(1, 2)
    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 2)
    pad_right = rng.randint(1, 2)

    h = 2 + pad_top + slot_rows * frame_h + (slot_rows - 1) * row_gap + pad_bottom + 2
    w = 2 + pad_left + slot_cols * frame_w + (slot_cols - 1) * col_gap + pad_right + 2
    grid = np.full((h, w), bg, dtype=int)

    side_order = []
    for side in SIDES:
        side_order.extend([side] * counts[side])
    rng.shuffle(side_order)

    used_slots: set[tuple[int, int]] = set()
    used_axes = {side: set() for side in SIDES}
    assignments: list[tuple[str, tuple[int, int]]] = []

    for side in side_order:
        candidates = []
        for slot_r in range(slot_rows):
            for slot_c in range(slot_cols):
                if (slot_r, slot_c) in used_slots:
                    continue
                axis = slot_c if side in ("top", "bottom") else slot_r
                if axis in used_axes[side]:
                    continue
                candidates.append((slot_r, slot_c))
        if not candidates:
            return None
        slot = rng.choice(candidates)
        used_slots.add(slot)
        used_axes[side].add(slot[1] if side in ("top", "bottom") else slot[0])
        assignments.append((side, slot))

    frame_colors = rng.sample(non_bg_colors, n_frames)
    legend_pool_size = rng.randint(3, min(5, len(non_bg_colors)))
    legend_pool = rng.sample(non_bg_colors, legend_pool_size)
    palettes = {side: _random_palette_sequence(rng, legend_pool, counts[side]) for side in SIDES}

    base_row = 2 + pad_top
    base_col = 2 + pad_left
    frames: list[dict] = []

    for idx, (side, (slot_r, slot_c)) in enumerate(assignments):
        jitter_r = rng.randint(0, row_gap - 1) if row_gap > 0 else 0
        jitter_c = rng.randint(0, col_gap - 1) if col_gap > 0 else 0
        r0 = base_row + slot_r * (frame_h + row_gap) + jitter_r
        c0 = base_col + slot_c * (frame_w + col_gap) + jitter_c

        if side in ("top", "bottom"):
            gap_index = rng.randint(1, frame_w - 2)
        else:
            gap_index = rng.randint(1, frame_h - 2)

        _draw_frame(grid, bg, r0, c0, frame_h, frame_w, frame_colors[idx], side, gap_index)
        frames.append(
            {
                "bbox": (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1),
                "gap_side": side,
                "sort_key": (c0 + c0 + frame_w - 1) if side in ("top", "bottom") else (r0 + r0 + frame_h - 1),
            }
        )

    _paint_palettes(grid, palettes, rng)

    if _bg_color(grid) != bg:
        return None

    expected = _expected_output(grid, frames, palettes, bg)
    try:
        produced = main(grid)
    except Exception:
        return None

    if not np.array_equal(expected, produced):
        return None
    if np.array_equal(grid, produced):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
