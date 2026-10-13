from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "right", "down", "left")
PATTERN_DIMS = ((3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (5, 5))
GEN_RETRIES = 500
MAX_CANVAS = 30
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _foreground_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if int(g[nr, nc]) != BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            comps.append(
                {
                    "cells": cells,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "area": int(len(cells)),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _is_frame_component(comp: dict) -> bool:
    patch = np.asarray(comp["patch"], dtype=int)
    h, w = patch.shape
    if h < 5 or w < 5:
        return False

    colors = {int(v) for v in patch.ravel() if int(v) != BLACK}
    if len(colors) != 1:
        return False

    fg = patch != BLACK
    if not np.any(patch[1:-1, 1:-1] == BLACK):
        return False

    for r, c in np.argwhere(fg):
        rr = int(r)
        cc = int(c)
        if rr not in (0, h - 1) and cc not in (0, w - 1):
            return False
    return True


def _frame_color(comp: dict) -> int:
    patch = np.asarray(comp["patch"], dtype=int)
    vals = [int(v) for v in patch.ravel() if int(v) != BLACK]
    assert vals
    return vals[0]


def _gap_sides(frame_patch: np.ndarray) -> list[str]:
    patch = np.asarray(frame_patch, dtype=int)
    h, w = patch.shape
    assert h >= 5 and w >= 5
    sides: list[str] = []
    if np.any(patch[0, 1:-1] == BLACK):
        sides.append("up")
    if np.any(patch[1:-1, -1] == BLACK):
        sides.append("right")
    if np.any(patch[-1, 1:-1] == BLACK):
        sides.append("down")
    if np.any(patch[1:-1, 0] == BLACK):
        sides.append("left")
    return sides


def _opposite(side: str) -> str:
    idx = DIRS.index(side)
    return DIRS[(idx + 2) % 4]


def _infer_pattern_side(patch: np.ndarray) -> str:
    arr = np.asarray(patch, dtype=int)
    vals, counts = np.unique(arr[arr != BLACK], return_counts=True)
    assert len(vals) >= 2
    base_color = int(vals[int(np.argmax(counts))])
    accent = (arr != BLACK) & (arr != base_color)
    assert accent.any()

    border_counts = {
        "up": int(accent[0, :].sum()),
        "right": int(accent[:, -1].sum()),
        "down": int(accent[-1, :].sum()),
        "left": int(accent[:, 0].sum()),
    }
    best_side = max(DIRS, key=lambda side: border_counts[side])
    top_score = border_counts[best_side]
    if top_score > 0 and sum(border_counts[side] == top_score for side in DIRS) == 1:
        return best_side

    coords = np.argwhere(accent)
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    h, w = accent.shape
    distances = {
        "up": center_r,
        "right": (w - 1) - center_c,
        "down": (h - 1) - center_r,
        "left": center_c,
    }
    return min(DIRS, key=lambda side: (distances[side], DIRS.index(side)))


def _rotate_to_side(patch: np.ndarray, desired_side: str) -> np.ndarray:
    for k in range(4):
        cand = np.rot90(patch, k)
        if _infer_pattern_side(cand) == desired_side:
            return cand
    raise AssertionError("could not orient pattern")


def _centered_top_left(inner_h: int, inner_w: int, patch_h: int, patch_w: int) -> tuple[int, int]:
    return (inner_h - patch_h) // 2, (inner_w - patch_w) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hollow frame with the palette pattern of equal area and rotate the
    pattern so its asymmetric accent points opposite the frame's first gap.
    """
    g = np.asarray(grid, dtype=int)
    comps = _foreground_components(g)

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in comps:
        if _is_frame_component(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    assert frames and patterns and len(frames) == len(patterns)

    patterns_by_area: dict[int, dict] = {}
    for comp in patterns:
        area = int(comp["area"])
        assert area not in patterns_by_area
        patterns_by_area[area] = comp

    out = g.copy()
    for comp in patterns:
        for r, c in comp["cells"]:
            out[int(r), int(c)] = BLACK

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        cavity_area = int(inner_h * inner_w)
        pattern = patterns_by_area[cavity_area]

        first_gap = _gap_sides(frame["patch"])[0]
        desired_side = _opposite(first_gap)
        oriented = _rotate_to_side(pattern["patch"], desired_side)
        ph, pw = oriented.shape
        assert ph <= inner_h and pw <= inner_w
        dr, dc = _centered_top_left(inner_h, inner_w, ph, pw)
        _paste(out, oriented, int(r0 + 1 + dr), int(c0 + 1 + dc))

    return out


def _build_pattern(height: int, width: int, base: int, accent: int, side: str) -> np.ndarray:
    patch = np.full((height, width), int(base), dtype=int)
    if side in {"up", "down"}:
        center = width // 2
        border_row = 0 if side == "up" else height - 1
        inner_row = 1 if side == "up" else height - 2
        if width <= 4:
            cols = [center]
        else:
            cols = [center - 1, center, center + 1]
        patch[border_row, cols] = int(accent)
        patch[inner_row, center] = int(accent)
    else:
        center = height // 2
        border_col = 0 if side == "left" else width - 1
        inner_col = 1 if side == "left" else width - 2
        if height <= 4:
            rows = [center]
        else:
            rows = [center - 1, center, center + 1]
        patch[rows, border_col] = int(accent)
        patch[center, inner_col] = int(accent)
    return patch


def _build_frame(inner_h: int, inner_w: int, color: int, gap_sides: list[str]) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)

    mid_r = (inner_h + 2) // 2
    mid_c = (inner_w + 2) // 2
    for side in gap_sides:
        if side == "up":
            patch[0, mid_c] = BLACK
        elif side == "right":
            patch[mid_r, -1] = BLACK
        elif side == "down":
            patch[-1, mid_c] = BLACK
        else:
            patch[mid_r, 0] = BLACK
    return patch


def _gap_set_for_primary(rng: random.Random, primary: str) -> list[str]:
    idx = DIRS.index(primary)
    out = [primary]
    for side in DIRS[idx + 1 :]:
        if rng.random() < 0.5:
            out.append(side)
    return out


def _line_positions(lengths: list[int], total: int, gap_min: int, rng: random.Random) -> list[int] | None:
    if not lengths:
        return []
    need = sum(lengths) + gap_min * (len(lengths) - 1)
    if need > total:
        return None
    slack = total - need
    buckets = [0] * (len(lengths) + 1)
    for _ in range(slack):
        buckets[rng.randrange(len(buckets))] += 1
    starts: list[int] = []
    cur = buckets[0]
    for i, length in enumerate(lengths):
        starts.append(cur)
        cur += int(length)
        if i + 1 < len(lengths):
            cur += gap_min + buckets[i + 1]
    return starts


def _row_groups(indices: list[int], rng: random.Random) -> list[list[int]]:
    order = indices[:]
    rng.shuffle(order)
    if len(order) <= 2 or rng.random() < 0.45:
        return [order]
    cut = rng.randrange(1, len(order))
    return [order[:cut], order[cut:]]


def _expected_output(shape: tuple[int, int], items: list[dict], frame_positions: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx, item in enumerate(items):
        frame_top, frame_left = frame_positions[idx]
        frame = item["frame_patch"]
        placed = item["placed_patch"]
        _paste(out, frame, frame_top, frame_left)
        inner_h = int(frame.shape[0] - 2)
        inner_w = int(frame.shape[1] - 2)
        dr, dc = _centered_top_left(inner_h, inner_w, int(placed.shape[0]), int(placed.shape[1]))
        _paste(out, placed, int(frame_top + 1 + dr), int(frame_left + 1 + dc))
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    dims = rng.sample(list(PATTERN_DIMS), k=n_frames)
    colors = rng.sample(list(COLOR_POOL), k=2 * n_frames)

    items: list[dict] = []
    for idx in range(n_frames):
        natural_h, natural_w = dims[idx]
        base = int(colors[2 * idx])
        accent = int(colors[2 * idx + 1])
        natural_side = rng.choice(DIRS)
        primary_gap = rng.choice(DIRS)
        desired_side = _opposite(primary_gap)

        natural_patch = _build_pattern(natural_h, natural_w, base, accent, natural_side)
        placed_patch = _rotate_to_side(natural_patch, desired_side)
        gap_sides = _gap_set_for_primary(rng, primary_gap)
        frame_patch = _build_frame(int(placed_patch.shape[0]), int(placed_patch.shape[1]), GREY, gap_sides)

        items.append(
            {
                "natural_patch": natural_patch,
                "placed_patch": placed_patch,
                "frame_patch": frame_patch,
                "area": int(natural_patch.size),
            }
        )

    row_groups = _row_groups(list(range(n_frames)), rng)
    row_gap = 2
    frame_gap = 2
    palette_gap = 1
    left_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    band_sep = 2

    row_heights = [max(int(items[idx]["frame_patch"].shape[0]) for idx in group) for group in row_groups]
    row_min_widths = [
        sum(int(items[idx]["frame_patch"].shape[1]) for idx in group) + frame_gap * (len(group) - 1)
        for group in row_groups
    ]

    palette_order = list(range(n_frames))
    rng.shuffle(palette_order)
    palette_widths = [int(items[idx]["natural_patch"].shape[1]) for idx in palette_order]
    palette_band_h = max(int(items[idx]["natural_patch"].shape[0]) for idx in palette_order) + 2
    palette_min_width = sum(palette_widths) + palette_gap * (n_frames - 1)

    inner_width = max(max(row_min_widths), palette_min_width) + rng.randint(0, 4)
    total_width = left_margin + inner_width + right_margin
    frame_block_h = sum(row_heights) + row_gap * (len(row_heights) - 1)
    total_height = top_margin + frame_block_h + band_sep + palette_band_h + bottom_margin
    if total_width > MAX_CANVAS or total_height > MAX_CANVAS:
        return None

    palette_on_top = bool(rng.randrange(2))
    if palette_on_top:
        palette_top = top_margin
        frame_top = palette_top + palette_band_h + band_sep
    else:
        frame_top = top_margin
        palette_top = frame_top + frame_block_h + band_sep

    inp = np.zeros((total_height, total_width), dtype=int)
    frame_positions: dict[int, tuple[int, int]] = {}

    cur_row_top = frame_top
    for row_idx, group in enumerate(row_groups):
        widths = [int(items[idx]["frame_patch"].shape[1]) for idx in group]
        starts = _line_positions(widths, inner_width, frame_gap, rng)
        assert starts is not None
        row_h = row_heights[row_idx]
        for idx, start in zip(group, starts):
            frame = items[idx]["frame_patch"]
            fh, fw = frame.shape
            row_offset = rng.randint(0, row_h - fh)
            r0 = cur_row_top + row_offset
            c0 = left_margin + start
            _paste(inp, frame, r0, c0)
            frame_positions[idx] = (int(r0), int(c0))
        cur_row_top += row_h + row_gap

    pattern_starts = _line_positions(palette_widths, inner_width, palette_gap, rng)
    assert pattern_starts is not None
    for idx, start in zip(palette_order, pattern_starts):
        patch = items[idx]["natural_patch"]
        ph, pw = patch.shape
        row_offset = rng.randint(0, palette_band_h - ph)
        r0 = palette_top + row_offset
        c0 = left_margin + start
        _paste(inp, patch, r0, c0)

    expected = _expected_output(inp.shape, items, frame_positions)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
