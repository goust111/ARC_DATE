from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
PINK = 6
MOTIF_COLORS = (2, 3, 7, 8, 9)
BAR_COLORS = (1, 4, 5)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = False,
) -> list[dict]:
    comps: list[dict] = []
    if monochromatic:
        colors = sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))
        for color in colors:
            for coords in _components_from_mask(grid == color, connectivity=connectivity):
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return comps

    for coords in _components_from_mask(grid != bg, connectivity=connectivity):
        comps.append(
            {
                "color": None,
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _step_offset(row: int, r0: int, r1: int, step: int) -> int:
    if r0 <= row <= r1:
        return 0
    if row < r0:
        return ((r0 - row - 1) // step) + 1
    return ((row - r1 - 1) // step) + 1


def _render_pairs(shape: tuple[int, int], pairs: list[dict], step: int) -> np.ndarray:
    h, w = map(int, shape)
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pair in pairs:
        for row in range(h):
            offset = _step_offset(row, pair["r0"], pair["r1"], step)
            left = int(pair["left"]) - offset
            right = int(pair["right"]) + offset
            if 0 <= left < w:
                claims[(row, left)].add(int(pair["color"]))
            if 0 <= right < w:
                claims[(row, right)].add(int(pair["color"]))

    out = np.zeros((h, w), dtype=int)
    for (r, c), colors in claims.items():
        out[r, c] = int(next(iter(colors))) if len(colors) == 1 else PINK
    return out


def _motif_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    candidates: list[dict] = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            box_i = comps[i]["bbox"]
            box_j = comps[j]["bbox"]
            hi = box_i[2] - box_i[0] + 1
            wi = box_i[3] - box_i[1] + 1
            hj = box_j[2] - box_j[0] + 1
            wj = box_j[3] - box_j[1] + 1
            if (hi, wi) != (hj, wj) or hi < 3 or wi < 3:
                continue

            crop_i = _crop_bbox(grid, box_i)
            crop_j = _crop_bbox(grid, box_j)
            for complete_idx, damaged_idx, complete, damaged in (
                (i, j, crop_i, crop_j),
                (j, i, crop_j, crop_i),
            ):
                complete_mask = complete != bg
                damaged_mask = damaged != bg
                missing = complete_mask & ~damaged_mask
                extra = damaged_mask & ~complete_mask
                overlap = complete_mask & damaged_mask
                if not missing.any() or extra.any():
                    continue
                if not np.array_equal(complete[overlap], damaged[overlap]):
                    continue
                candidates.append(
                    {
                        "complete_index": complete_idx,
                        "damaged_index": damaged_idx,
                        "missing_count": int(missing.sum()),
                    }
                )
    return candidates


def _extract_motif_step(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    candidates = _motif_candidates(grid, bg)
    assert len(candidates) == 1, "expected a unique complete-vs-damaged motif pair"

    comps = _find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    motif_mask = np.zeros(grid.shape, dtype=bool)
    choice = candidates[0]
    for idx in (choice["complete_index"], choice["damaged_index"]):
        coords = comps[idx]["coords"]
        motif_mask[coords[:, 0], coords[:, 1]] = True
    return int(choice["missing_count"]), motif_mask


def _extract_bar_pairs(grid: np.ndarray, bar_mask: np.ndarray, bg: int) -> list[dict]:
    pairs: list[dict] = []
    colors = sorted(int(v) for v in np.unique(grid[bar_mask]) if int(v) != int(bg))
    assert 2 <= len(colors) <= 3, "expected 2-3 lower bar colors"
    assert PINK not in colors, "pink is reserved for overlap output"

    for color in colors:
        coords = np.argwhere((grid == int(color)) & bar_mask)
        assert coords.size > 0, "expected visible lower bars"
        cols = sorted(set(int(v) for v in coords[:, 1]))
        assert len(cols) == 2, "each lower color must occupy exactly two columns"

        bars = []
        for col in cols:
            rows = sorted(int(v) for v in coords[coords[:, 1] == col, 0])
            assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be vertically contiguous"
            bars.append((rows[0], rows[-1], col))

        left, right = bars
        assert left[0] == right[0] and left[1] == right[1], "paired bars must share a row interval"
        pairs.append(
            {
                "color": int(color),
                "r0": int(left[0]),
                "r1": int(left[1]),
                "left": int(left[2]),
                "right": int(right[2]),
            }
        )

    intervals = sorted((p["r0"], p["r1"]) for p in pairs)
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert a1 < b0 or b1 < a0, "lower bar intervals must not overlap"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Count the missing pixels between two aligned motifs and use that count as the stepped-expansion period for the lower bar pairs.

    Input structure:
    - The background is black.
    - The upper region contains two same-sized aligned multicolor motif objects; the left one is complete and the right one is a damaged copy with some foreground cells removed.
    - The lower region contains 2-3 colors, and each such color appears as exactly two vertical bars sharing one row interval.

    Rule:
    - Compare the two upper motifs after aligning their bbox top-left corners, and count the cells that are foreground in the complete motif but black in the damaged copy.
    - Use that missing-pixel count N as the common step length for every lower bar pair: inside the original bar interval the columns stay fixed, and outside it the left bar moves one column left while the right bar moves one column right after each block of N rows away from the interval.
    - Paint all in-bounds expanded cells with their pair color, recolor cells claimed by different colors to pink, and erase the two upper motifs to black.

    Key insight:
    - The upper motifs do not directly contribute visible output; they encode a hidden numeric parameter, and that count controls the geometry of the lower stepped expansions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert int(bg) == BG, "expected black background"

    step, motif_mask = _extract_motif_step(g, bg)
    assert step >= 1, "step must be positive"
    bar_mask = (g != bg) & ~motif_mask
    pairs = _extract_bar_pairs(g, bar_mask, bg)
    return _render_pairs(g.shape, pairs, step)


def _connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask, connectivity=4)
    return len(comps) == 1


def _make_motif_patch(h: int, w: int, colors: tuple[int, ...]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    style = random.choice(["rows", "cols", "quadrants", "rings"])
    if style == "rows":
        cuts = sorted(random.sample(range(1, h), len(colors) - 1))
        bands = [0, *cuts, h]
        for i, color in enumerate(colors):
            patch[bands[i] : bands[i + 1], :] = int(color)
    elif style == "cols":
        cuts = sorted(random.sample(range(1, w), len(colors) - 1))
        bands = [0, *cuts, w]
        for i, color in enumerate(colors):
            patch[:, bands[i] : bands[i + 1]] = int(color)
    elif style == "quadrants":
        split_r = random.randint(1, h - 1)
        split_c = random.randint(1, w - 1)
        a, b, *rest = colors
        c = rest[0] if rest else a
        d = rest[-1] if rest else b
        patch[:split_r, :split_c] = a
        patch[:split_r, split_c:] = b
        patch[split_r:, :split_c] = c
        patch[split_r:, split_c:] = d
    else:
        for r in range(h):
            for c in range(w):
                idx = min(r, c, h - 1 - r, w - 1 - c) % len(colors)
                patch[r, c] = int(colors[idx])
    return patch


def _choose_missing_positions(patch: np.ndarray, n_missing: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    coords = [(r, c) for r in range(h) for c in range(w) if patch[r, c] != BG]
    for _ in range(200):
        chosen = random.sample(coords, n_missing)
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) >= 2 and len(cols) >= 2:
            return chosen
    return random.sample(coords, n_missing)


def _place_intervals(h: int, lower_top: int, heights: list[int]) -> list[tuple[int, int]] | None:
    for _ in range(400):
        intervals: list[tuple[int, int]] = []
        for bar_h in heights:
            r0 = random.randint(lower_top, h - bar_h - 1)
            r1 = r0 + bar_h - 1
            if any(not (r1 < a0 or a1 < r0) for a0, a1 in intervals):
                break
            intervals.append((r0, r1))
        else:
            intervals.sort()
            if any(max(r0, h - 1 - r1) >= 6 for r0, r1 in intervals):
                return intervals
    return None


def _sample_columns(pair_count: int, width: int) -> list[tuple[int, int]] | None:
    for _ in range(400):
        pairs: list[tuple[int, int]] = []
        used: list[int] = []
        for _pair in range(pair_count):
            gap = random.randint(4, 8)
            if gap >= width - 4:
                break
            left = random.randint(2, width - gap - 3)
            right = left + gap
            cols = [left, right]
            if any(abs(col - prev) < 2 for col in cols for prev in used):
                break
            used.extend(cols)
            pairs.append((left, right))
        else:
            return pairs
    return None


def _build_input_from_spec(
    h: int,
    w: int,
    motif_patch: np.ndarray,
    missing_positions: list[tuple[int, int]],
    gap: int,
    top_r: int,
    left_c: int,
    pair_specs: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    mh, mw = motif_patch.shape
    grid[top_r : top_r + mh, left_c : left_c + mw] = motif_patch
    damaged = motif_patch.copy()
    for r, c in missing_positions:
        damaged[r, c] = BG
    right_c = left_c + mw + gap
    grid[top_r : top_r + mh, right_c : right_c + mw] = damaged

    for pair in pair_specs:
        grid[pair["r0"] : pair["r1"] + 1, pair["left"]] = int(pair["color"])
        grid[pair["r0"] : pair["r1"] + 1, pair["right"]] = int(pair["color"])
    return grid


def _validate_instance(grid: np.ndarray, expected_step: int, expected_pairs: list[dict], want_overlap: bool) -> bool:
    try:
        step, motif_mask = _extract_motif_step(grid, BG)
        if step != int(expected_step):
            return False
        bar_mask = (grid != BG) & ~motif_mask
        parsed_pairs = _extract_bar_pairs(grid, bar_mask, BG)
    except AssertionError:
        return False

    parsed_key = sorted((p["color"], p["r0"], p["r1"], p["left"], p["right"]) for p in parsed_pairs)
    expected_key = sorted((p["color"], p["r0"], p["r1"], p["left"], p["right"]) for p in expected_pairs)
    if parsed_key != expected_key:
        return False

    out = main(grid)
    oracle = _render_pairs(grid.shape, expected_pairs, expected_step)
    if not np.array_equal(out, oracle):
        return False

    non_black = int(np.count_nonzero(out))
    pink = int(np.count_nonzero(out == PINK))
    if non_black < 20 or pink > max(16, non_black // 3):
        return False
    if want_overlap != (pink > 0):
        return False

    if expected_step > 1 and np.array_equal(out, _render_pairs(grid.shape, expected_pairs, expected_step - 1)):
        return False
    if expected_step < 6 and np.array_equal(out, _render_pairs(grid.shape, expected_pairs, expected_step + 1)):
        return False
    for i in range(len(expected_pairs)):
        reduced = expected_pairs[:i] + expected_pairs[i + 1 :]
        if np.array_equal(out, _render_pairs(grid.shape, reduced, expected_step)):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    pair_count = random.choice([2, 3])
    n_missing = random.randint(2, 5)
    motif_h = random.randint(4, 6)
    motif_w = random.randint(4, 6)
    motif_colors = tuple(random.sample(MOTIF_COLORS, random.choice([2, 3])))
    motif_patch = _make_motif_patch(motif_h, motif_w, motif_colors)
    missing_positions = _choose_missing_positions(motif_patch, n_missing)

    width = random.randint(max(22, motif_w * 2 + 8), 28)
    top_gap = random.randint(2, 3)
    left_c_max = width - (motif_w * 2 + top_gap) - 1
    if left_c_max < 1:
        return None
    left_c = random.randint(1, left_c_max)
    top_r = random.randint(1, 2)
    lower_top = top_r + motif_h + random.randint(3, 4)

    heights = [random.randint(2, 4) for _ in range(pair_count)]
    min_height = lower_top + sum(heights) + pair_count + 2
    height = random.randint(max(20, min_height), min(28, max(20, min_height + 5)))
    intervals = _place_intervals(height, lower_top, heights)
    if intervals is None:
        return None

    want_overlap = random.random() < 0.55
    colors = random.sample(BAR_COLORS, pair_count)
    for _ in range(500):
        col_pairs = _sample_columns(pair_count, width)
        if col_pairs is None:
            return None
        pair_specs = [
            {
                "color": int(colors[i]),
                "r0": int(intervals[i][0]),
                "r1": int(intervals[i][1]),
                "left": int(col_pairs[i][0]),
                "right": int(col_pairs[i][1]),
            }
            for i in range(pair_count)
        ]
        grid = _build_input_from_spec(
            height,
            width,
            motif_patch,
            missing_positions,
            top_gap,
            top_r,
            left_c,
            pair_specs,
        )
        out = _render_pairs(grid.shape, pair_specs, n_missing)
        pink = int(np.count_nonzero(out == PINK))
        if want_overlap != (pink > 0):
            continue
        if _validate_instance(grid, n_missing, pair_specs, want_overlap):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
