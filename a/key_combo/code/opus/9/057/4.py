from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


ORANGE = 7
RED = 2
STRIPE_PALETTE = (1, 3, 4, 6, 8, 9)
NOISE_PALETTE = (1, 3, 4, 5, 6, 8, 9)

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
STRIPE_COUNT_RANGE = (3, 5)
STRIPE_WIDTH_RANGE = (3, 5)
PAIR_COUNT_MAX = 3
NOISE_RATE_RANGE = (0.10, 0.15)
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 80


def _dominant_color(values: np.ndarray) -> int:
    vals = [int(v) for v in values.reshape(-1) if int(v) != ORANGE]
    assert vals, "stripe line must contain at least one non-orange pixel"
    counts = Counter(vals)
    best = counts.most_common()
    assert len(best) == 1 or best[0][1] != best[1][1], "expected a unique dominant color"
    return int(best[0][0])


def _mode_count(values: np.ndarray) -> int:
    vals = [int(v) for v in values.reshape(-1) if int(v) != ORANGE]
    assert vals, "stripe line must contain at least one non-orange pixel"
    return int(Counter(vals).most_common(1)[0][1])


def _orientation_score(grid: np.ndarray, *, by_rows: bool) -> int:
    if by_rows:
        return sum(_mode_count(grid[r, :]) for r in range(grid.shape[0]))
    return sum(_mode_count(grid[:, c]) for c in range(grid.shape[1]))


def _denoise(grid: np.ndarray, *, by_rows: bool) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    line_colors: list[int] = []
    if by_rows:
        for r in range(g.shape[0]):
            color = _dominant_color(g[r, :])
            line_colors.append(color)
            row_mask = out[r, :] != ORANGE
            out[r, row_mask] = color
    else:
        for c in range(g.shape[1]):
            color = _dominant_color(g[:, c])
            line_colors.append(color)
            col_mask = out[:, c] != ORANGE
            out[col_mask, c] = color
    return out, line_colors


def _extract_segments(grid: np.ndarray) -> list[dict]:
    points_by_k: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(grid == ORANGE):
        points_by_k[int(r + c)].append((int(r), int(c)))

    segments: list[dict] = []
    for k, pts in points_by_k.items():
        by_b = sorted((r - c, r, c) for r, c in pts)
        current: list[tuple[int, int, int]] = []
        for item in by_b:
            if current and item[0] != current[-1][0] + 2:
                cells = [(r, c) for _b, r, c in current]
                segments.append(
                    {
                        "k": int(k),
                        "bmin": int(current[0][0]),
                        "bmax": int(current[-1][0]),
                        "cells": cells,
                    }
                )
                current = []
            current.append(item)
        if current:
            cells = [(r, c) for _b, r, c in current]
            segments.append(
                {
                    "k": int(k),
                    "bmin": int(current[0][0]),
                    "bmax": int(current[-1][0]),
                    "cells": cells,
                }
            )

    for seg in segments:
        assert len(seg["cells"]) >= 2, "orange segments must be length >= 2"
    return segments


def _segment_stripe_color(segment: dict, line_colors: list[int], *, by_rows: bool) -> int:
    colors = {
        int(line_colors[r if by_rows else c])
        for r, c in segment["cells"]
    }
    assert len(colors) == 1, "each segment must lie inside exactly one stripe"
    return int(next(iter(colors)))


def _pair_fill_cells(seg_a: dict, seg_b: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    left, right = sorted([seg_a, seg_b], key=lambda seg: int(seg["k"]))
    b_lo = max(int(left["bmin"]), int(right["bmin"]))
    b_hi = min(int(left["bmax"]), int(right["bmax"]))
    if b_lo > b_hi:
        return set()

    cells: set[tuple[int, int]] = set()
    for b in range(b_lo, b_hi + 1):
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for r in range(r_min, r_max + 1):
            c = r - b
            k = r + c
            if int(left["k"]) < k < int(right["k"]):
                cells.add((int(r), int(c)))
    seg_cells = set(left["cells"]) | set(right["cells"])
    return {cell for cell in cells if cell not in seg_cells}


def _apply_color_pair_fill(grid: np.ndarray, segments: list[dict], colors: list[int], *, by_rows: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    by_color: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        by_color[_segment_stripe_color(seg, colors, by_rows=by_rows)].append(seg)

    for stripe_color, segs in by_color.items():
        assert len(segs) == 2, f"expected exactly one segment pair for stripe color {stripe_color}"
        for r, c in _pair_fill_cells(segs[0], segs[1], out.shape):
            if out[r, c] != ORANGE:
                out[r, c] = RED
    return out


def _apply_global_consecutive_fill(grid: np.ndarray, segments: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = [seg for seg in segments if int(seg["bmin"]) <= b <= int(seg["bmax"])]
        if len(active) < 2:
            continue
        active.sort(key=lambda seg: int(seg["k"]))
        for left, right in zip(active, active[1:]):
            for r, c in _pair_fill_cells(left, right, out.shape):
                if out[r, c] != ORANGE:
                    out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise stripe colors and use the cleaned stripe color to pair orange anti-diagonal segments before filling between them.

    Input structure:
    - The whole grid is a horizontal or vertical stripe canvas made of 3-5 contiguous color bands.
    - Non-orange cells are noisy: some stripe cells are replaced by other stripe colors or extra distractor colors.
    - Orange (7) short anti-diagonal segments sit inside stripes; each stripe has either no segment or exactly one parallel pair.

    Rule:
    - Ignore orange pixels and score the row-striped and column-striped interpretations by summing each row/column's dominant-color support.
    - Choose the uniquely better orientation, then repaint every non-orange cell in each selected row/column with that line's dominant color.
    - For every orange segment, read the denoised stripe color of the stripe containing it; segments with the same cleaned stripe color form a pair.
    - For each such pair, fill all non-orange cells strictly between the two parallel anti-diagonal segments with red (2), while keeping stripe colors and orange segment pixels otherwise unchanged.

    Key insight:
    - The orange segments are not paired by raw proximity alone: first the stripe canvas must be denoised, and the recovered stripe color becomes the pairing key that decides which diagonal region gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_score = _orientation_score(g, by_rows=True)
    col_score = _orientation_score(g, by_rows=False)
    assert row_score != col_score, "expected a unique stripe orientation"
    by_rows = bool(row_score > col_score)

    clean, line_colors = _denoise(g, by_rows=by_rows)
    segments = _extract_segments(clean)
    assert segments, "expected at least one orange segment pair"
    return _apply_color_pair_fill(clean, segments, line_colors, by_rows=by_rows)


def _random_partition(total: int, parts: int, lo: int, hi: int) -> tuple[int, ...] | None:
    options: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: tuple[int, ...]) -> None:
        if slots == 0:
            if remaining == 0:
                options.append(prefix)
            return
        min_v = max(lo, remaining - (slots - 1) * hi)
        max_v = min(hi, remaining - (slots - 1) * lo)
        for value in range(min_v, max_v + 1):
            rec(remaining - value, slots - 1, prefix + (value,))

    rec(int(total), int(parts), ())
    if not options:
        return None
    return random.choice(options)


def _make_clean_canvas(height: int, width: int, by_rows: bool, stripe_widths: tuple[int, ...], stripe_colors: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    stripes: list[dict] = []
    offset = 0
    for stripe_width, color in zip(stripe_widths, stripe_colors):
        if by_rows:
            grid[offset : offset + stripe_width, :] = int(color)
        else:
            grid[:, offset : offset + stripe_width] = int(color)
        stripes.append(
            {
                "start": int(offset),
                "end": int(offset + stripe_width - 1),
                "width": int(stripe_width),
                "color": int(color),
            }
        )
        offset += stripe_width
    return grid, stripes


def _segment_from_cells(cells: list[tuple[int, int]]) -> dict:
    bs = [r - c for r, c in cells]
    ks = {r + c for r, c in cells}
    assert len(ks) == 1
    return {
        "k": int(next(iter(ks))),
        "bmin": int(min(bs)),
        "bmax": int(max(bs)),
        "cells": [(int(r), int(c)) for r, c in cells],
    }


def _propose_pair(stripe: dict, *, by_rows: bool, height: int, width: int) -> tuple[dict, dict, set[tuple[int, int]]] | None:
    span = int(stripe["width"])
    shifts = [shift for shift in (2, 3) if shift <= span - 2]
    if not shifts:
        return None
    shift = random.choice(shifts)
    max_len = min(4, span - shift)
    if max_len < 2:
        return None
    length = random.randint(2, max_len)

    if by_rows:
        r0_min = int(stripe["start"])
        r0_max = int(stripe["end"]) - (length + shift) + 1
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(length - 1, width - 1 - shift)
    else:
        c0_min = int(stripe["start"]) + length - 1
        c0_max = int(stripe["end"]) - shift
        if c0_min > c0_max:
            return None
        c0 = random.randint(c0_min, c0_max)
        r0 = random.randint(0, height - (length + shift))

    seg_a_cells = [(r0 + i, c0 - i) for i in range(length)]
    seg_b_cells = [(r0 + shift + i, c0 + shift - i) for i in range(length)]
    seg_a = _segment_from_cells(seg_a_cells)
    seg_b = _segment_from_cells(seg_b_cells)
    fill_cells = _pair_fill_cells(seg_a, seg_b, (height, width))
    if not fill_cells:
        return None
    return seg_a, seg_b, fill_cells


def _chebyshev_clear(candidate: set[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in used:
                    return False
    return True


def _place_pairs(clean: np.ndarray, stripes: list[dict], *, by_rows: bool) -> tuple[np.ndarray, dict[int, list[dict]]] | None:
    eligible = [stripe for stripe in stripes if int(stripe["width"]) >= 4]
    if len(eligible) < 2:
        return None

    max_pairs = min(PAIR_COUNT_MAX, len(eligible), len(stripes) - 1 if len(stripes) > 2 else 2)
    pair_count = random.randint(2, max_pairs)
    chosen = random.sample(eligible, pair_count)

    grid = clean.copy()
    used_orange: set[tuple[int, int]] = set()
    used_fill: set[tuple[int, int]] = set()
    by_color: dict[int, list[dict]] = {}

    for stripe in chosen:
        for _ in range(PAIR_ATTEMPTS):
            proposal = _propose_pair(stripe, by_rows=by_rows, height=grid.shape[0], width=grid.shape[1])
            if proposal is None:
                continue
            seg_a, seg_b, fill_cells = proposal
            seg_cells = set(seg_a["cells"]) | set(seg_b["cells"])
            if not _chebyshev_clear(seg_cells, used_orange):
                continue
            if seg_cells & used_fill:
                continue
            if fill_cells & used_orange:
                continue
            if fill_cells & used_fill:
                continue
            if len(fill_cells) > max(7, grid.size // 7):
                continue

            for r, c in seg_cells:
                grid[r, c] = ORANGE
            used_orange |= seg_cells
            used_fill |= fill_cells
            by_color[int(stripe["color"])] = [seg_a, seg_b]
            break
        else:
            return None

    if len(by_color) < 2:
        return None
    return grid, by_color


def _line_noise_candidates(grid: np.ndarray, *, by_rows: bool) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if by_rows:
        for r in range(grid.shape[0]):
            slots = [(r, c) for c in range(grid.shape[1]) if int(grid[r, c]) != ORANGE]
            random.shuffle(slots)
            candidates.append(slots)
    else:
        for c in range(grid.shape[1]):
            slots = [(r, c) for r in range(grid.shape[0]) if int(grid[r, c]) != ORANGE]
            random.shuffle(slots)
            candidates.append(slots)
    return candidates


def _line_stripe_lookup(stripes: list[dict], *, by_rows: bool, shape: tuple[int, int]) -> list[int]:
    line_count = shape[0] if by_rows else shape[1]
    colors = [0] * line_count
    for stripe in stripes:
        for idx in range(int(stripe["start"]), int(stripe["end"]) + 1):
            colors[idx] = int(stripe["color"])
    return colors


def _add_noise(clean_with_orange: np.ndarray, stripes: list[dict], *, by_rows: bool) -> np.ndarray:
    inp = clean_with_orange.copy()
    line_colors = _line_stripe_lookup(stripes, by_rows=by_rows, shape=inp.shape)
    slots_by_line = _line_noise_candidates(inp, by_rows=by_rows)
    max_per_line = [max(1, (len(slots) - 1) // 3) for slots in slots_by_line]
    used_per_line = [0] * len(slots_by_line)

    total_slots = sum(len(slots) for slots in slots_by_line)
    target_noise = random.randint(
        max(1, int(total_slots * NOISE_RATE_RANGE[0])),
        max(1, int(total_slots * NOISE_RATE_RANGE[1])),
    )

    def place_on_line(line_idx: int, color: int | None = None) -> bool:
        if used_per_line[line_idx] >= max_per_line[line_idx]:
            return False
        if not slots_by_line[line_idx]:
            return False
        r, c = slots_by_line[line_idx].pop()
        base_color = int(line_colors[line_idx])
        palette = [candidate for candidate in NOISE_PALETTE if int(candidate) != base_color]
        chosen = int(color) if color is not None else int(random.choice(palette))
        if chosen == base_color or chosen == ORANGE:
            return False
        inp[r, c] = chosen
        used_per_line[line_idx] += 1
        return True

    priority_lines = list(range(len(slots_by_line)))
    random.shuffle(priority_lines)
    extra_colors = [5]
    for color in extra_colors:
        for line_idx in priority_lines:
            if place_on_line(line_idx, color=color):
                break

    for _ in range(target_noise):
        feasible = [
            idx
            for idx in range(len(slots_by_line))
            if used_per_line[idx] < max_per_line[idx] and slots_by_line[idx]
        ]
        if not feasible:
            break
        weighted = feasible[:]
        random.shuffle(weighted)
        line_idx = weighted[0]
        place_on_line(line_idx)

    return inp


def _validate_instance(inp: np.ndarray, expected: np.ndarray, clean_with_orange: np.ndarray, stripes: list[dict], *, by_rows: bool) -> bool:
    try:
        row_score = _orientation_score(inp, by_rows=True)
        col_score = _orientation_score(inp, by_rows=False)
        if row_score == col_score:
            return False
        if bool(row_score > col_score) != bool(by_rows):
            return False

        denoised, line_colors = _denoise(inp, by_rows=by_rows)
        if not np.array_equal(denoised, clean_with_orange):
            return False

        segments = _extract_segments(denoised)
        by_color: dict[int, list[dict]] = defaultdict(list)
        for seg in segments:
            by_color[_segment_stripe_color(seg, line_colors, by_rows=by_rows)].append(seg)
        used_groups = [segs for segs in by_color.values() if segs]
        if len(used_groups) < 2:
            return False
        if any(len(segs) != 2 for segs in used_groups):
            return False

        solved = _apply_color_pair_fill(denoised, segments, line_colors, by_rows=by_rows)
        if not np.array_equal(solved, expected):
            return False

        wrong = _apply_global_consecutive_fill(denoised, segments)
        if np.array_equal(wrong, expected):
            return False

        red_count = int(np.sum(expected == RED))
        if red_count < 3 or red_count > inp.size // 5:
            return False

        stripe_colors = {int(stripe["color"]) for stripe in stripes}
        if len(stripe_colors) != len(stripes):
            return False

        return bool(np.array_equal(main(inp), expected))
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    by_rows = bool(random.getrandbits(1))
    stripe_dim = height if by_rows else width

    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    stripe_widths = _random_partition(stripe_dim, stripe_count, *STRIPE_WIDTH_RANGE)
    if stripe_widths is None:
        return None
    if sum(1 for width_ in stripe_widths if width_ >= 4) < 2:
        return None

    stripe_colors = random.sample(list(STRIPE_PALETTE), stripe_count)
    clean, stripes = _make_clean_canvas(height, width, by_rows, stripe_widths, stripe_colors)

    placed = _place_pairs(clean, stripes, by_rows=by_rows)
    if placed is None:
        return None
    clean_with_orange, _pair_map = placed

    line_colors = _line_stripe_lookup(stripes, by_rows=by_rows, shape=clean.shape)
    segments = _extract_segments(clean_with_orange)
    expected = _apply_color_pair_fill(clean_with_orange, segments, line_colors, by_rows=by_rows)

    inp = _add_noise(clean_with_orange, stripes, by_rows=by_rows)
    if not _validate_instance(inp, expected, clean_with_orange, stripes, by_rows=by_rows):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy stripe pairing puzzle")
