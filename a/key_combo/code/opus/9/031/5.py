from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = (1, 2, 3, 4, 7)
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict] = []
    for comp in _components(g == BLACK):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if not np.all(g[r0 - 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r1 + 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c0 - 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c1 + 1] == GREY):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == GREY):
            continue
        pattern_mask = (patch != BLACK) & (patch != GREY)
        if not np.any(pattern_mask):
            continue
        pr0, pc0, pr1, pc1 = _bbox(np.argwhere(pattern_mask))
        pattern = patch[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": int(len(comp)),
                "pattern": pattern,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each pattern from a grey-walled flood-fill region, then overlay them by region area.

    Input structure:
    - Black background fills the grid outside and inside a grey wall partition.
    - Grey wall segments divide the interior into 3 or 4 rectangular enclosed regions.
    - Each enclosed region contains one same-sized colored pattern made of non-black, non-grey pixels.

    Rule:
    - Flood-fill the black cells in each enclosed region using 4-connectivity, treating grey walls as barriers.
    - Use the size of each black flood-fill component as that region's area.
    - Inside each region, crop the colored pixels to their own pattern bounding box.
    - Create a black output grid of that shared pattern size and draw the region patterns from smaller area to larger area, with later pixels transparently overwriting earlier ones.

    Key insight:
    - The visible winner at any overlapping pattern cell is determined by the geometry of the enclosing region, not by the pattern itself: larger black regions draw later and cover smaller ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _extract_regions(g)
    assert 3 <= len(regions) <= 4

    areas = [region["area"] for region in regions]
    assert len(set(areas)) == len(areas)

    shapes = {tuple(region["pattern"].shape) for region in regions}
    assert len(shapes) == 1
    out = np.zeros(next(iter(shapes)), dtype=int)

    for region in sorted(regions, key=lambda item: item["area"]):
        pattern = region["pattern"]
        mask = pattern != BLACK
        out[mask] = pattern[mask]

    assert np.any(out != BLACK)
    return out


def _boundary_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                pts.append((r, c))
    return pts


def _covers_bbox(cells: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return 0 in rows and h - 1 in rows and 0 in cols and w - 1 in cols


def _choose_anchor_sets(n_regions: int, pat_h: int, pat_w: int) -> list[set[tuple[int, int]]] | None:
    boundary = _boundary_positions(pat_h, pat_w)
    candidates = []
    for size in (2, 3, 4):
        for combo in itertools.combinations(boundary, size):
            if _covers_bbox(combo, pat_h, pat_w):
                candidates.append(set(combo))
    random.shuffle(candidates)
    candidates.sort(key=len)

    chosen: list[set[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()

    def backtrack() -> bool:
        if len(chosen) == n_regions:
            return True
        options = [cand for cand in candidates if cand.isdisjoint(used)]
        random.shuffle(options)
        for cand in options[:250]:
            chosen.append(cand)
            used.update(cand)
            if backtrack():
                return True
            chosen.pop()
            used.difference_update(cand)
        return False

    if backtrack():
        return chosen
    return None


def _build_patterns(
    n_regions: int, pat_h: int, pat_w: int
) -> tuple[list[np.ndarray], list[set[tuple[int, int]]], list[tuple[int, int, tuple[int, int]]]] | None:
    anchor_sets = _choose_anchor_sets(n_regions, pat_h, pat_w)
    if anchor_sets is None:
        return None

    owners: dict[tuple[int, int], set[int]] = {}
    for idx, cells in enumerate(anchor_sets):
        for pos in cells:
            owners[pos] = {idx}

    all_positions = [(r, c) for r in range(pat_h) for c in range(pat_w)]
    interior = [(r, c) for r in range(1, pat_h - 1) for c in range(1, pat_w - 1)]
    available = [pos for pos in interior if pos not in owners]
    if len(available) < n_regions - 1:
        available = [pos for pos in all_positions if pos not in owners]
    random.shuffle(available)

    pair_conflicts: list[tuple[int, int, tuple[int, int]]] = []
    for idx in range(n_regions - 1):
        if not available:
            return None
        pos = available.pop()
        owners[pos] = {idx, idx + 1}
        pair_conflicts.append((idx, idx + 1, pos))

    counts = [len(anchor_sets[idx]) for idx in range(n_regions)]
    for left, right, _pos in pair_conflicts:
        counts[left] += 1
        counts[right] += 1

    target = max(counts) + random.randint(0, 1)
    extras_needed = sum(target - count for count in counts)
    if len(available) < extras_needed:
        spare = [pos for pos in all_positions if pos not in owners]
        random.shuffle(spare)
        available.extend(spare)
    available = [pos for pos in available if pos not in owners]
    random.shuffle(available)

    unique_positions: list[set[tuple[int, int]]] = [set(cells) for cells in anchor_sets]
    for idx in range(n_regions):
        need = target - counts[idx]
        for _ in range(need):
            if not available:
                return None
            pos = available.pop()
            owners[pos] = {idx}
            unique_positions[idx].add(pos)

    colors = random.sample(list(PATTERN_COLORS), n_regions)
    patterns = [np.zeros((pat_h, pat_w), dtype=int) for _ in range(n_regions)]
    for pos, layers in owners.items():
        for idx in layers:
            patterns[idx][pos] = colors[idx]

    if any(np.count_nonzero(pattern) != target for pattern in patterns):
        return None
    if any(len(layers) > 2 for layers in owners.values()):
        return None

    for idx in range(n_regions):
        if not any(owners[pos] == {idx} for pos in owners):
            return None
    for left, right, pos in pair_conflicts:
        if patterns[left][pos] == 0 or patterns[right][pos] == 0 or patterns[left][pos] == patterns[right][pos]:
            return None

    return patterns, unique_positions, pair_conflicts


def _sample_layout(
    n_regions: int, pat_h: int, pat_w: int
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]], tuple[int, int, int, int], list[tuple[int, int, int, int]]] | None:
    min_h = pat_h + 3
    min_w = pat_w + 3
    max_dim = 10

    if n_regions == 4:
        top_h = random.randint(min_h, max_dim)
        bottom_h = random.randint(min_h, max_dim)
        left_w = random.randint(min_w, max_dim)
        right_w = random.randint(min_w, max_dim)
        areas = [top_h * left_w, top_h * right_w, bottom_h * left_w, bottom_h * right_w]
        if len(set(areas)) != 4:
            return None
        outer_h = top_h + bottom_h + 3
        outer_w = left_w + right_w + 3
        mt, mb, ml, mr = [random.randint(1, 3) for _ in range(4)]
        h = outer_h + mt + mb
        w = outer_w + ml + mr
        if max(h, w) > 30:
            return None
        r0, c0 = mt, ml
        r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
        split_r = r0 + top_h + 1
        split_c = c0 + left_w + 1
        rects = [
            (r0 + 1, c0 + 1, top_h, left_w),
            (r0 + 1, split_c + 1, top_h, right_w),
            (split_r + 1, c0 + 1, bottom_h, left_w),
            (split_r + 1, split_c + 1, bottom_h, right_w),
        ]
        segments = [
            (split_r, c0 + 1, split_r, c1 - 1),
            (r0 + 1, split_c, r1 - 1, split_c),
        ]
        return (h, w), rects, (r0, c0, r1, c1), segments

    kind = random.choice(("v_left", "v_right", "h_top", "h_bottom"))
    a = random.randint(min_h, max_dim)
    b = random.randint(min_h, max_dim)
    c = random.randint(min_w, max_dim)
    d = random.randint(min_w, max_dim)

    mt, mb, ml, mr = [random.randint(1, 3) for _ in range(4)]
    if kind in ("v_left", "v_right"):
        left_w, right_w = c, d
        top_h, bottom_h = a, b
        tall_h = top_h + bottom_h + 1
        areas = [top_h * left_w, bottom_h * left_w, tall_h * right_w]
        if len(set(areas)) != 3:
            return None
        outer_h = tall_h + 2
        outer_w = left_w + right_w + 3
        h = outer_h + mt + mb
        w = outer_w + ml + mr
        if max(h, w) > 30:
            return None
        r0, c0 = mt, ml
        r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
        split_c = c0 + left_w + 1 if kind == "v_left" else c0 + right_w + 1
        split_r = r0 + top_h + 1
        if kind == "v_left":
            rects = [
                (r0 + 1, c0 + 1, top_h, left_w),
                (split_r + 1, c0 + 1, bottom_h, left_w),
                (r0 + 1, split_c + 1, tall_h, right_w),
            ]
            segments = [
                (r0 + 1, split_c, r1 - 1, split_c),
                (split_r, c0 + 1, split_r, split_c - 1),
            ]
        else:
            rects = [
                (r0 + 1, split_c + 1, top_h, right_w),
                (split_r + 1, split_c + 1, bottom_h, right_w),
                (r0 + 1, c0 + 1, tall_h, left_w),
            ]
            segments = [
                (r0 + 1, split_c, r1 - 1, split_c),
                (split_r, split_c + 1, split_r, c1 - 1),
            ]
        return (h, w), rects, (r0, c0, r1, c1), segments

    left_w, right_w = c, d
    top_h, bottom_h = a, b
    wide_w = left_w + right_w + 1
    areas = [top_h * left_w, top_h * right_w, bottom_h * wide_w]
    if kind == "h_top":
        areas = [top_h * wide_w, bottom_h * left_w, bottom_h * right_w]
    if len(set(areas)) != 3:
        return None
    outer_h = top_h + bottom_h + 3
    outer_w = wide_w + 2
    h = outer_h + mt + mb
    w = outer_w + ml + mr
    if max(h, w) > 30:
        return None
    r0, c0 = mt, ml
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    split_r = r0 + top_h + 1 if kind == "h_top" else r0 + bottom_h + 1
    split_c = c0 + left_w + 1
    if kind == "h_top":
        rects = [
            (r0 + 1, c0 + 1, top_h, wide_w),
            (split_r + 1, c0 + 1, bottom_h, left_w),
            (split_r + 1, split_c + 1, bottom_h, right_w),
        ]
        segments = [
            (split_r, c0 + 1, split_r, c1 - 1),
            (split_r + 1, split_c, r1 - 1, split_c),
        ]
    else:
        rects = [
            (r0 + 1, c0 + 1, top_h, left_w),
            (r0 + 1, split_c + 1, top_h, right_w),
            (split_r + 1, c0 + 1, bottom_h, wide_w),
        ]
        segments = [
            (split_r, c0 + 1, split_r, c1 - 1),
            (r0 + 1, split_c, split_r - 1, split_c),
        ]
    return (h, w), rects, (r0, c0, r1, c1), segments


def _draw_segment(grid: np.ndarray, seg: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = seg
    if r0 == r1:
        grid[r0, c0 : c1 + 1] = GREY
    else:
        grid[r0 : r1 + 1, c0] = GREY


def _overlay(patterns: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(patterns[0])
    for pattern in patterns:
        mask = pattern != 0
        out[mask] = pattern[mask]
    return out


def _visible_support(output: np.ndarray, pattern: np.ndarray) -> bool:
    mask = pattern != 0
    return bool(np.any(output[mask] == pattern[mask]))


def _attempt_generate() -> np.ndarray | None:
    n_regions = random.choice((3, 4))
    if n_regions == 4:
        pat_h = random.choice((4, 5))
        pat_w = random.choice((4, 5))
    else:
        pat_h = random.choice((4, 5))
        pat_w = random.choice((4, 5))

    built = _build_patterns(n_regions, pat_h, pat_w)
    if built is None:
        return None
    patterns, unique_positions, pair_conflicts = built

    layout = _sample_layout(n_regions, pat_h, pat_w)
    if layout is None:
        return None
    shape, rects, outer, segments = layout
    rects = sorted(rects, key=lambda item: item[2] * item[3])

    min_room_h = min(rect[2] for rect in rects)
    min_room_w = min(rect[3] for rect in rects)
    max_off_r = min_room_h - pat_h - 1
    max_off_c = min_room_w - pat_w - 1
    if max_off_r < 1 or max_off_c < 1:
        return None
    off_r = random.randint(1, min(2, max_off_r))
    off_c = random.randint(1, min(2, max_off_c))

    grid = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = outer
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    for seg in segments:
        _draw_segment(grid, seg)

    target_count = np.count_nonzero(patterns[0])
    expected_areas = []
    for idx, (rr, cc, hh, ww) in enumerate(rects):
        grid[rr : rr + hh, cc : cc + ww] = BLACK
        pr = rr + off_r
        pc = cc + off_c
        patch = patterns[idx]
        mask = patch != 0
        region = grid[pr : pr + pat_h, pc : pc + pat_w]
        region[mask] = patch[mask]
        expected_areas.append(hh * ww - target_count)

    if len(set(expected_areas)) != n_regions:
        return None

    expected = _overlay(patterns)
    try:
        actual = main(grid)
        extracted = sorted(_extract_regions(grid), key=lambda item: item["area"])
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    if len(extracted) != n_regions:
        return None
    if [region["area"] for region in extracted] != sorted(expected_areas):
        return None
    if any(not np.array_equal(region["pattern"], patterns[idx]) for idx, region in enumerate(extracted)):
        return None

    if any(not _visible_support(expected, pattern) for pattern in patterns):
        return None
    if any(np.array_equal(expected, pattern) for pattern in patterns):
        return None

    for left, right, pos in pair_conflicts:
        winner = patterns[right][pos]
        if expected[pos] != winner:
            return None
        if patterns[left][pos] == winner:
            return None

    if any(not unique_positions[idx] for idx in range(n_regions)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-priority overlay puzzle")
