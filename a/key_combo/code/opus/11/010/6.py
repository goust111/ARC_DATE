import itertools
import math
import random

import numpy as np


BG = 0
MARKER = 5
FG_COLORS = [1, 2, 3, 4, 7]
MAX_GENERATION_ATTEMPTS = 400


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _region_stats(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[list[int], list[int], list[list[int]], list[int]]:
    r0, c0, r1, c1 = bbox
    sequence: list[int] = []
    col_counts: list[int] = []
    col_chunks: list[list[int]] = []
    occupied_cols: list[int] = []
    for c in range(c0, c1 + 1):
        chunk = [
            int(grid[r, c])
            for r in range(r0, r1 + 1)
            if int(grid[r, c]) not in (BG, MARKER)
        ]
        if chunk:
            occupied_cols.append(int(c))
            col_chunks.append(chunk)
            col_counts.append(len(chunk))
            sequence.extend(chunk)
    return sequence, col_counts, col_chunks, occupied_cols


def _candidate_groups(grid: np.ndarray) -> list[dict]:
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert len(marker_coords) % 4 == 0
    assert 8 <= len(marker_coords) <= 12

    candidates: list[dict] = []
    for combo in itertools.combinations(range(len(marker_coords)), 4):
        pts = np.array([marker_coords[i] for i in combo], dtype=int)
        rows = pts[:, 0]
        cols = pts[:, 1]
        if np.sum(rows == rows.min()) != 1 or np.sum(rows == rows.max()) != 1:
            continue
        if np.sum(cols == cols.min()) != 1 or np.sum(cols == cols.max()) != 1:
            continue

        r0 = int(rows.min()) + 1
        r1 = int(rows.max()) - 1
        c0 = int(cols.min()) + 1
        c1 = int(cols.max()) - 1
        if r0 > r1 or c0 > c1:
            continue

        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(region == MARKER):
            continue
        if not np.any((region != BG) & (region != MARKER)):
            continue

        sequence, col_counts, _col_chunks, _occupied_cols = _region_stats(grid, (r0, c0, r1, c1))
        total = len(sequence)
        height = math.isqrt(total)
        if height * height != total or height <= 0:
            continue

        expected = [2 * i + 1 for i in range(height)]
        if sorted(col_counts) != expected:
            continue

        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h < max(expected):
            continue
        if box_w < expected[-1]:
            continue
        if box_w % 2 == 0:
            continue

        fg_cells = frozenset(
            (int(r), int(c))
            for r, c in np.argwhere((grid != BG) & (grid != MARKER))
            if r0 <= int(r) <= r1 and c0 <= int(c) <= c1
        )
        candidates.append(
            {
                "indices": frozenset(combo),
                "bbox": (r0, c0, r1, c1),
                "fg_cells": fg_cells,
                "height": height,
            }
        )
    return candidates


def _find_regions(grid: np.ndarray) -> list[dict]:
    all_fg = frozenset((int(r), int(c)) for r, c in np.argwhere((grid != BG) & (grid != MARKER)))
    assert all_fg

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    group_count = len(marker_coords) // 4
    candidates = _candidate_groups(grid)
    by_index: dict[int, list[dict]] = {i: [] for i in range(len(marker_coords))}
    for cand in candidates:
        for idx in cand["indices"]:
            by_index[idx].append(cand)

    solutions: list[list[dict]] = []
    full_mask = (1 << len(marker_coords)) - 1

    def backtrack(remaining_mask: int, chosen: list[dict], covered_fg: frozenset) -> None:
        if len(solutions) > 1:
            return
        if remaining_mask == 0:
            if len(chosen) == group_count and covered_fg == all_fg:
                solutions.append(chosen.copy())
            return

        first_idx = (remaining_mask & -remaining_mask).bit_length() - 1
        for cand in by_index[first_idx]:
            cand_mask = 0
            for idx in cand["indices"]:
                cand_mask |= 1 << idx
            if cand_mask & ~remaining_mask:
                continue
            if cand["fg_cells"] & covered_fg:
                continue
            if any(_boxes_overlap(cand["bbox"], prev["bbox"]) for prev in chosen):
                continue
            backtrack(
                remaining_mask ^ cand_mask,
                chosen + [cand],
                covered_fg | cand["fg_cells"],
            )

    backtrack(full_mask, [], frozenset())
    assert len(solutions) == 1, "marker grouping must be uniquely determined"
    return sorted(solutions[0], key=lambda item: item["bbox"])


def _draw_pyramid(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    sequence: list[int],
) -> None:
    r0, c0, r1, c1 = bbox
    total = len(sequence)
    height = math.isqrt(total)
    assert height * height == total

    box_w = c1 - c0 + 1
    assert box_w % 2 == 1

    idx = 0
    for layer in range(height):
        width = 2 * layer + 1
        row = r1 - height + 1 + layer
        start = c0 + (box_w - width) // 2
        out[row, start : start + width] = np.array(sequence[idx : idx + width], dtype=int)
        idx += width
    assert idx == total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each four-marker region as a local factory that turns scattered pixels into a bottom-aligned centered pyramid.

    Input structure:
    - The grid uses black background, 2-3 groups of exactly four grey marker pixels, and colored foreground pixels inside the marker-defined regions.
    - Within each marker group, the topmost, bottommost, leftmost, and rightmost marker are all unique; shrinking those extrema by one cell gives the region box.
    - Inside every region, the non-black non-grey pixels occupy columns whose population counts are a permutation of 1, 3, 5, ... and therefore sum to a perfect square.

    Rule:
    - Partition the grey markers into the unique valid groups of four, and for each group compute the inset rectangle from the marker extrema.
    - Read that rectangle's colored pixels column by column from left to right, and within each column from top to bottom.
    - Clear the original colored pixels in the rectangle, rebuild the read sequence as centered odd-width rows 1, 3, 5, ... with the base on the rectangle's bottom edge, and erase every grey marker to background.

    Key insight:
    - The grey pixels are not output geometry; they are invisible region delimiters whose only job is to tell you which scattered pixels belong to the same local pyramid and where that pyramid must be centered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _find_regions(g)
    out = g.copy()

    for region in regions:
        bbox = region["bbox"]
        sequence, _col_counts, _col_chunks, _occupied_cols = _region_stats(g, bbox)
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        _draw_pyramid(out, bbox, sequence)

    out[g == MARKER] = BG
    return out


def _choose_region_heights(region_count: int) -> list[int]:
    while True:
        heights = [random.choices([1, 2, 3, 4], weights=[1, 3, 4, 2], k=1)[0] for _ in range(region_count)]
        if len(set(heights)) >= 2 and max(heights) > 1:
            return heights


def _sample_region_shape(height: int, force_slack: bool) -> tuple[int, int]:
    base = 2 * height - 1
    width_choices = [base]
    if base + 2 <= 9:
        width_choices.append(base + 2)
    if base + 4 <= 11:
        width_choices.append(base + 4)
    if force_slack:
        width_choices = [w for w in width_choices if w > base]
    width = random.choice(width_choices)
    region_h = random.randint(base, min(10, base + 2))
    return region_h, width


def _random_chunk(count: int, palette: list[int], require_mixed: bool) -> list[int]:
    if count == 1:
        return [random.choice(palette)]

    while True:
        chunk = [random.choice(palette) for _ in range(count)]
        if require_mixed and len(set(chunk)) < 2:
            continue
        return chunk


def _build_region(height: int, force_permuted: bool, force_mixed: bool, force_slack: bool) -> dict:
    region_h, region_w = _sample_region_shape(height, force_slack)
    base = 2 * height - 1
    palette_size = random.randint(2, min(4, len(FG_COLORS))) if height > 1 else 1
    palette = random.sample(FG_COLORS, palette_size)

    odd_counts = [2 * i + 1 for i in range(height)]
    while True:
        col_counts = odd_counts.copy()
        random.shuffle(col_counts)
        if not force_permuted or height == 1 or col_counts != odd_counts:
            break

    occupied_cols = sorted(random.sample(range(region_w), height))
    local = np.zeros((region_h, region_w), dtype=int)
    sequence: list[int] = []
    mixed_seen = False

    for idx, (col, count) in enumerate(zip(occupied_cols, col_counts)):
        require_mixed = force_mixed and not mixed_seen and count > 1
        chunk = _random_chunk(count, palette, require_mixed=require_mixed)
        rows = sorted(random.sample(range(region_h), count))
        for r, color in zip(rows, chunk):
            local[r, col] = int(color)
        sequence.extend(chunk)
        mixed_seen |= len(set(chunk)) > 1

    center_col = region_w // 2
    bottom_center = (region_h - 1, center_col)
    if height == 1 and local[bottom_center] != BG and int(local.sum() > 0):
        return _build_region(height, force_permuted, force_mixed, force_slack)

    return {
        "height": height,
        "grid": local,
        "sequence": sequence,
        "col_counts": col_counts,
        "occupied_cols": occupied_cols,
        "mixed": mixed_seen,
        "slack": region_w > base,
    }


def _sample_layout(
    canvas_h: int,
    canvas_w: int,
    regions: list[dict],
) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    outer_boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        rh, rw = region["grid"].shape
        placed = False
        for _ in range(400):
            r0 = random.randint(1, canvas_h - rh - 2)
            c0 = random.randint(1, canvas_w - rw - 2)
            outer = (r0 - 1, c0 - 1, r0 + rh, c0 + rw)
            expanded = (outer[0] - 1, outer[1] - 1, outer[2] + 1, outer[3] + 1)
            if any(_boxes_overlap(expanded, prev) for prev in outer_boxes):
                continue
            placements.append((r0, c0))
            outer_boxes.append(outer)
            placed = True
            break
        if not placed:
            raise RuntimeError("layout failed")
    return placements


def _paste_region(canvas: np.ndarray, region: dict, r0: int, c0: int) -> None:
    local = region["grid"]
    rh, rw = local.shape
    mask = local != BG
    canvas[r0 : r0 + rh, c0 : c0 + rw][mask] = local[mask]

    top_c = random.randint(c0, c0 + rw - 1)
    bottom_c = random.randint(c0, c0 + rw - 1)
    left_r = random.randint(r0, r0 + rh - 1)
    right_r = random.randint(r0, r0 + rh - 1)
    canvas[r0 - 1, top_c] = MARKER
    canvas[r0 + rh, bottom_c] = MARKER
    canvas[left_r, c0 - 1] = MARKER
    canvas[right_r, c0 + rw] = MARKER


def _semantic_checks(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert np.count_nonzero(grid == MARKER) in (8, 12)
    assert np.any((grid != BG) & (grid != MARKER))

    regions = _find_regions(grid)
    assert len(regions) in (2, 3)

    heights: list[int] = []
    has_permuted = False
    has_mixed = False
    has_slack = False
    for region in regions:
        bbox = region["bbox"]
        sequence, col_counts, col_chunks, _occupied_cols = _region_stats(grid, bbox)
        height = math.isqrt(len(sequence))
        assert height * height == len(sequence)
        assert sorted(col_counts) == [2 * i + 1 for i in range(height)]
        heights.append(height)
        has_permuted |= height > 1 and col_counts != [2 * i + 1 for i in range(height)]
        has_mixed |= any(len(set(chunk)) > 1 for chunk in col_chunks if len(chunk) > 1)
        has_slack |= (bbox[3] - bbox[1] + 1) > (2 * height - 1)

    assert len(set(heights)) >= 2
    assert has_permuted
    assert has_mixed
    assert has_slack

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == MARKER)

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        before = grid[r0 : r1 + 1, c0 : c1 + 1]
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.array_equal(before, after), "each region should visibly change"


def _gen_attempt() -> np.ndarray | None:
    canvas_h = random.randint(18, 24)
    canvas_w = random.randint(18, 24)
    region_count = random.choice([2, 3])
    heights = _choose_region_heights(region_count)
    special = random.choice([i for i, h in enumerate(heights) if h > 1])

    regions: list[dict] = []
    for idx, height in enumerate(heights):
        region = _build_region(
            height=height,
            force_permuted=(idx == special),
            force_mixed=(idx == special),
            force_slack=(idx == special),
        )
        regions.append(region)

    try:
        placements = _sample_layout(canvas_h, canvas_w, regions)
    except RuntimeError:
        return None

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    for region, (r0, c0) in zip(regions, placements):
        _paste_region(grid, region, r0, c0)

    try:
        _semantic_checks(grid)
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
