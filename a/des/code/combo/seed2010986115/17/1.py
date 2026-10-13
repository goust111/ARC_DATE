import random
from itertools import combinations
from typing import Iterable

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if int(col[0]) == bg or not np.all(col == col[0]):
            continue
        if not np.any(grid[:, :c] != bg) or not np.any(grid[:, c + 1 :] != bg):
            continue
        right_non_bg = int(np.count_nonzero(grid[:, c + 1 :] != bg))
        left_non_bg = int(np.count_nonzero(grid[:, :c] != bg))
        candidates.append((right_non_bg, -left_non_bg, c))
    if not candidates:
        raise AssertionError("expected a solid vertical divider")
    return min(candidates)[2]


def _read_palette(grid: np.ndarray, divider_col: int, bg: int) -> list[int]:
    right = grid[:, divider_col + 1 :]
    palette: list[int] = []
    for r in range(right.shape[0]):
        cols = np.flatnonzero(right[r] != bg)
        if cols.size == 0:
            continue
        colors = {int(right[r, c]) for c in cols}
        if len(colors) != 1:
            raise AssertionError("each palette row must contain one color")
        palette.append(int(next(iter(colors))))
    if not palette:
        raise AssertionError("expected at least one palette color")
    return palette


def _run_end(indices: np.ndarray, start: int, max_gap: int = 2) -> int:
    pos = int(np.searchsorted(indices, start))
    if pos >= len(indices):
        raise AssertionError("start index is missing from run")
    current = int(indices[pos])
    for i in range(pos + 1, len(indices)):
        nxt = int(indices[i])
        if nxt - current > max_gap:
            break
        current = nxt
    return current


def _row_run_end(mask: np.ndarray, row: int, start_col: int) -> int:
    cols = np.flatnonzero(mask[row])
    return _run_end(cols, start_col)


def _col_run_end(mask: np.ndarray, col: int, start_row: int) -> int:
    rows = np.flatnonzero(mask[:, col])
    return _run_end(rows, start_row)


def _perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    remaining = mask.copy()
    h, w = remaining.shape
    boxes: list[tuple[int, int, int, int]] = []

    while np.any(remaining):
        rows, cols = np.where(remaining)
        idx = int(np.argmin(rows * w + cols))
        r0 = int(rows[idx])
        c0 = int(cols[idx])

        c1 = _row_run_end(remaining, r0, c0)
        r1 = max(_col_run_end(remaining, c0, r0), _col_run_end(remaining, c1, r0))
        c1 = max(c1, _row_run_end(remaining, r1, c0))
        r1 = max(r1, _col_run_end(remaining, c1, r0))

        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("frames must be at least 3x3")

        sub = remaining[r0 : r1 + 1, c0 : c1 + 1]
        perim = _perimeter_mask(r1 - r0 + 1, c1 - c0 + 1)
        if np.any(sub & ~perim):
            raise AssertionError("frames must be hollow rectangles")

        sub[perim] = False
        boxes.append((r0, c0, r1, c1))

    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame according to the number of single-pixel holes on its border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    divider_col = _find_divider_col(g, bg)
    palette = _read_palette(g, divider_col, bg)

    left_mask = g[:, :divider_col] != bg
    boxes = _frame_bboxes(left_mask)

    out = g.copy()
    for r0, c0, r1, c1 in boxes:
        perim = _perimeter_mask(r1 - r0 + 1, c1 - c0 + 1)
        present = int(left_mask[r0 : r1 + 1, c0 : c1 + 1][perim].sum())
        perimeter_len = int(perim.sum())
        hole_count = perimeter_len - present
        if not 0 <= hole_count < len(palette):
            raise AssertionError("hole count is outside palette range")
        color = int(palette[hole_count])
        box = out[r0 : r1 + 1, c0 : c1 + 1]
        box[perim] = color

    return out


GRID_H_RANGE = (14, 24)
LEFT_W_RANGE = (16, 24)
RIGHT_W_RANGE = (3, 5)
FRAME_COUNT_RANGE = (2, 4)
PALETTE_LEN_RANGE = (3, 5)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
FRAME_PAD = 2
GEN_ATTEMPTS = 400


def _perimeter_cycle_positions(height: int, width: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(1, width - 1):
        pts.append((0, c))
    for r in range(1, height - 1):
        pts.append((r, width - 1))
    for c in range(width - 2, 0, -1):
        pts.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        pts.append((r, 0))
    return pts


def _sample_holes(height: int, width: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    candidates = _perimeter_cycle_positions(height, width)
    valid = [
        list(combo)
        for combo in combinations(candidates, count)
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for i, a in enumerate(combo) for b in combo[i + 1 :])
    ]
    if not valid:
        return None
    return list(rng.choice(valid))


def _draw_frame(height: int, width: int, color: int, bg: int, holes: Iterable[tuple[int, int]]) -> np.ndarray:
    frame = np.full((height, width), bg, dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color
    for r, c in holes:
        frame[r, c] = bg
    return frame


def _expected_output(grid: np.ndarray, frames: list[dict], palette: list[int]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        h = int(frame["h"])
        w = int(frame["w"])
        color = int(palette[int(frame["holes"])])
        out[r0, c0 : c0 + w] = color
        out[r0 + h - 1, c0 : c0 + w] = color
        out[r0 : r0 + h, c0] = color
        out[r0 : r0 + h, c0 + w - 1] = color
    return out


def _build_instance(rng: random.Random) -> tuple[np.ndarray, list[dict], list[int]]:
    bg = rng.randrange(10)
    palette_len = rng.randint(*PALETTE_LEN_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    divider_color = int(colors[0])
    frame_color = int(colors[1])
    palette = [int(c) for c in colors[2 : 2 + palette_len]]

    hole_counts = [rng.randint(0, palette_len - 1) for _ in range(frame_count)]
    if frame_count >= 2 and len(set(hole_counts)) == 1 and palette_len > 1:
        hole_counts[-1] = (hole_counts[-1] + 1) % palette_len

    height = rng.randint(max(GRID_H_RANGE[0], palette_len + 4), GRID_H_RANGE[1])
    left_w = rng.randint(*LEFT_W_RANGE)
    right_w = rng.randint(*RIGHT_W_RANGE)
    width = left_w + 1 + right_w

    specs: list[dict] = []
    for holes in hole_counts:
        hole_cells = None
        for _ in range(80):
            fh = rng.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], height - 2))
            fw = rng.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], left_w - 2))
            hole_cells = _sample_holes(fh, fw, holes, rng)
            if hole_cells is not None:
                specs.append({"h": fh, "w": fw, "holes": holes, "hole_cells": hole_cells})
                break
        if hole_cells is None:
            raise RuntimeError("failed to sample frame holes")

    specs.sort(key=lambda item: int(item["h"]) * int(item["w"]), reverse=True)

    blocked = np.zeros((height, left_w), dtype=bool)
    placed: list[dict] = []
    for spec in specs:
        fh = int(spec["h"])
        fw = int(spec["w"])
        placed_here = False
        for _ in range(200):
            if height - fh - 2 < 0 or left_w - fw - 2 < 0:
                break
            r0 = rng.randint(1, height - fh - 1)
            c0 = rng.randint(1, left_w - fw - 1)
            r_lo = max(0, r0 - FRAME_PAD)
            r_hi = min(height, r0 + fh + FRAME_PAD)
            c_lo = max(0, c0 - FRAME_PAD)
            c_hi = min(left_w, c0 + fw + FRAME_PAD)
            if blocked[r_lo:r_hi, c_lo:c_hi].any():
                continue
            blocked[r_lo:r_hi, c_lo:c_hi] = True
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "h": fh,
                    "w": fw,
                    "holes": int(spec["holes"]),
                    "hole_cells": list(spec["hole_cells"]),
                }
            )
            placed_here = True
            break
        if not placed_here:
            raise RuntimeError("failed to place frames")

    step = rng.choice([1, 1, 2])
    last_row = step * (palette_len - 1)
    if last_row >= height:
        raise RuntimeError("palette does not fit vertically")
    start_row = rng.randint(0, height - 1 - last_row)
    palette_col = left_w + 1 + rng.randint(0, right_w - 1)

    grid = np.full((height, width), bg, dtype=int)
    grid[:, left_w] = divider_color

    for frame in placed:
        patch = _draw_frame(
            int(frame["h"]),
            int(frame["w"]),
            frame_color,
            bg,
            frame["hole_cells"],
        )
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    for i, color in enumerate(palette):
        grid[start_row + i * step, palette_col] = color

    return grid, placed, palette


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, _frames, _palette = _build_instance(rng)
        except RuntimeError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
