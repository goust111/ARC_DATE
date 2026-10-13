from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _swap_colors(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    mask_a = patch == int(color_a)
    mask_b = patch == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    src = patch[pr0:pr1, pc0:pc1]
    mask = src != int(transparent)
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = src[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split one two-color object into upper/lower half-tiles and periodize four color variants across the grid.

    Input structure:
    - A dominant black/dark background contains exactly one 4-connected foreground object using exactly two colors.
    - The object's tight bounding box has even height, so a horizontal cut through its midpoint yields equal-height upper and lower half sprites.
    - The bbox upper-left corner is the phase origin for a slot lattice whose slots are half-height by full-width, separated by one blank row and one blank column.

    Rule:
    - Crop the object's tight bbox and split it horizontally at the midpoint into upper half U and lower half L, preserving any background cells inside the bbox.
    - Swap the two foreground colors inside each half to obtain U' and L'; background cells stay unchanged.
    - Starting at the original bbox upper-left corner, place slot origins every half_height+1 rows and width+1 columns.
    - Fill slot (i, j) by parity: even/even -> U, even/odd -> L', odd/even -> U', odd/odd -> L, clipping tiles at grid edges and leaving all gap rows/columns as background.

    Key insight:
    - The split halves are not reassembled into the original object; they become four tile generators, and slot-row/slot-column parity selects which variant appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components(g != int(bg))
    assert len(comps) == 1, "expected exactly one foreground object"
    r0, c0, r1, c1 = _bbox(comps[0])
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    colors = sorted(int(v) for v in np.unique(crop[crop != int(bg)]))
    assert len(colors) == 2, "expected exactly two foreground colors"
    color_a, color_b = colors

    height, width = map(int, crop.shape)
    assert height % 2 == 0, "object height must be even"
    half_h = height // 2
    upper = crop[:half_h].copy()
    lower = crop[half_h:].copy()
    assert np.any(upper != int(bg)) and np.any(lower != int(bg)), "both halves must be nonempty"

    upper_swapped = _swap_colors(upper, color_a, color_b)
    lower_swapped = _swap_colors(lower, color_a, color_b)

    out = np.full_like(g, int(bg))
    row_step = half_h + 1
    col_step = width + 1
    for i, rr in enumerate(range(r0, g.shape[0], row_step)):
        for j, cc in enumerate(range(c0, g.shape[1], col_step)):
            if i % 2 == 0 and j % 2 == 0:
                patch = upper
            elif i % 2 == 0:
                patch = lower_swapped
            elif j % 2 == 0:
                patch = upper_swapped
            else:
                patch = lower
            _paste_clip(out, patch, rr, cc, bg)
    return out


GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _runs_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _sample_half_runs(half_h: int, width: int, required_overlap: tuple[int, int] | None = None) -> list[tuple[int, int]] | None:
    min_run = max(1, width - 3)
    runs: list[tuple[int, int]] = []
    prev = required_overlap
    for _ in range(half_h):
        run_w = random.randint(min_run, width)
        valid_lefts = []
        for left in range(width - run_w + 1):
            interval = (left, left + run_w)
            if prev is None or _runs_overlap(interval, prev):
                valid_lefts.append(left)
        if not valid_lefts:
            return None
        if prev is None:
            left = random.choice(valid_lefts)
        else:
            prev_center = (prev[0] + prev[1] - 1) / 2.0
            valid_lefts.sort(key=lambda left: abs((left + left + run_w - 1) / 2.0 - prev_center))
            left = random.choice(valid_lefts[: min(3, len(valid_lefts))])
        interval = (left, left + run_w)
        runs.append(interval)
        prev = interval
    return runs


def _force_bbox_width(
    upper_runs: list[tuple[int, int]],
    lower_runs: list[tuple[int, int]],
    width: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_runs = list(upper_runs) + list(lower_runs)
    if min(left for left, _ in all_runs) > 0:
        idx = random.randrange(len(all_runs))
        _, right = all_runs[idx]
        all_runs[idx] = (0, right)
    if max(right for _, right in all_runs) < width:
        idx = random.randrange(len(all_runs))
        left, _ = all_runs[idx]
        all_runs[idx] = (left, width)
    return all_runs[: len(upper_runs)], all_runs[len(upper_runs) :]


def _runs_to_mask(runs: list[tuple[int, int]], width: int) -> np.ndarray:
    mask = np.zeros((len(runs), width), dtype=bool)
    for r, (left, right) in enumerate(runs):
        mask[r, left:right] = True
    return mask


def _border_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                out[r, c] = True
                break
    return out


def _pattern_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    h, w = mask.shape
    rr, cc = np.indices((h, w))
    if mode == "checker":
        offset = random.randint(0, 1)
        return mask & (((rr + cc + offset) % 2) == 0)
    if mode == "cols":
        pivot = random.randint(0, w - 2)
        return mask & (cc <= pivot)
    if mode == "rows":
        pivot = random.randint(0, h - 2)
        return mask & (rr <= pivot)
    if mode == "diag":
        bias = random.randint(-(w - 1), h - 1)
        return mask & ((rr - cc) <= bias)
    if mode == "anti":
        bias = random.randint(0, h + w - 2)
        return mask & ((rr + cc) <= bias)
    if mode == "border":
        return _border_mask(mask)
    raise ValueError(f"unknown mode: {mode}")


def _colorize_object(mask: np.ndarray, half_h: int, color_a: int, color_b: int) -> np.ndarray | None:
    total = int(mask.sum())
    min_count = max(1, math.ceil(0.30 * total))
    require_both_each_half = random.random() < 0.7
    modes = ["checker", "cols", "rows", "diag", "anti", "border"]

    for _ in range(80):
        raw = _pattern_mask(mask, random.choice(modes))
        for a_mask in (raw, mask & ~raw):
            count_a = int(a_mask.sum())
            count_b = total - count_a
            if count_a < min_count or count_b < min_count:
                continue
            upper_mask = mask[:half_h]
            lower_mask = mask[half_h:]
            upper_a = int(a_mask[:half_h].sum())
            lower_a = int(a_mask[half_h:].sum())
            upper_total = int(upper_mask.sum())
            lower_total = int(lower_mask.sum())
            if require_both_each_half:
                if upper_a == 0 or upper_a == upper_total:
                    continue
                if lower_a == 0 or lower_a == lower_total:
                    continue
            patch = np.full(mask.shape, GEN_BG, dtype=int)
            patch[mask] = int(color_b)
            patch[a_mask] = int(color_a)
            return patch
    return None


def _build_object_patch(half_h: int, width: int, color_a: int, color_b: int) -> np.ndarray | None:
    for _ in range(200):
        upper_runs = _sample_half_runs(half_h, width)
        if upper_runs is None:
            continue
        lower_runs = _sample_half_runs(half_h, width, required_overlap=upper_runs[-1])
        if lower_runs is None:
            continue
        upper_runs, lower_runs = _force_bbox_width(upper_runs, lower_runs, width)

        upper_mask = _runs_to_mask(upper_runs, width)
        lower_mask = _runs_to_mask(lower_runs, width)
        if np.array_equal(upper_mask, lower_mask):
            continue

        full_mask = np.vstack([upper_mask, lower_mask])
        density = float(full_mask.mean())
        if density < 0.35 or density > 0.90:
            continue
        if np.all(full_mask):
            continue

        colored = _colorize_object(full_mask, half_h, color_a, color_b)
        if colored is None:
            continue
        return colored
    return None


def _grid_size(origin: int, tile: int, full_slots: int, allow_clip: bool) -> int:
    base = origin + full_slots * tile + (full_slots - 1)
    if allow_clip and tile > 1:
        return base + 1 + random.randint(1, tile - 1)
    return base + random.randint(0, 1)


def _attempt_generate() -> np.ndarray | None:
    half_h = random.randint(2, 4)
    width = random.randint(3, 6)
    r0 = random.randint(0, 2)
    c0 = random.randint(0, 3)
    full_row_slots = random.randint(3, 4)
    full_col_slots = random.randint(3, 4)
    allow_row_clip = random.random() < 0.45
    allow_col_clip = random.random() < 0.45

    h = _grid_size(r0, half_h, full_row_slots, allow_row_clip)
    w = _grid_size(c0, width, full_col_slots, allow_col_clip)
    if h > 30 or w > 30:
        return None

    colors = random.sample(GEN_COLORS, 2)
    patch = _build_object_patch(half_h, width, colors[0], colors[1])
    if patch is None:
        return None

    grid = np.full((h, w), GEN_BG, dtype=int)
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    try:
        out = main(grid)
    except Exception:
        return None

    # Enforce the intended generation constraints.
    visible_row_starts = list(range(r0, h, half_h + 1))
    visible_col_starts = list(range(c0, w, width + 1))
    if len(visible_row_starts) < 3 or len(visible_col_starts) < 3:
        return None
    if not np.any(out != GEN_BG):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
