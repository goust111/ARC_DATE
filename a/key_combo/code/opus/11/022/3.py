from __future__ import annotations

import random
from typing import Iterable

import numpy as np


OUTER_COLOR = 5
MIDDLE_COLOR = 2
INNER_COLOR = 1
FRAME_COLORS = (OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR)
BACKGROUND = 0
MAX_ATTEMPTS = 400


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 - 1) * (c1 - c0 - 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _is_complete_frame(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if grid[r0, c0] != color or grid[r0, c1] != color or grid[r1, c0] != color or grid[r1, c1] != color:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _find_nested_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        rows = np.flatnonzero(np.any(g == color, axis=1))
        cols = np.flatnonzero(np.any(g == color, axis=0))
        for r0 in rows:
            for r1 in rows:
                if r1 - r0 < 2:
                    continue
                for c0 in cols:
                    for c1 in cols:
                        if c1 - c0 < 2:
                            continue
                        key = (color, int(r0), int(c0), int(r1), int(c1))
                        if key in seen:
                            continue
                        seen.add(key)
                        if _is_complete_frame(g, color, int(r0), int(c0), int(r1), int(c1)):
                            candidates.append(
                                {
                                    "color": int(color),
                                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                                }
                            )

    assert len(candidates) == 3, f"expected exactly 3 frame candidates, got {len(candidates)}"
    candidates.sort(key=lambda frame: _bbox_area(frame["bbox"]), reverse=True)
    assert _contains(candidates[0]["bbox"], candidates[1]["bbox"])
    assert _contains(candidates[1]["bbox"], candidates[2]["bbox"])
    assert set(frame["color"] for frame in candidates) == set(FRAME_COLORS)
    return candidates


def _pattern_local_coords(grid: np.ndarray, frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior == color)
    assert 2 <= len(coords) <= 5, f"expected 2..5 pattern cells, got {len(coords)}"
    return coords


def _scale_coords(coords: np.ndarray, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)
    mapped = np.empty_like(coords)
    mapped[:, 0] = (coords[:, 0] * dst_h) // src_h
    mapped[:, 1] = (coords[:, 1] * dst_w) // src_w
    return mapped


def main(grid: np.ndarray) -> np.ndarray:
    """Composite the nested-frame interior patterns by depth and crop to the innermost interior.

    Input structure:
    - Black background contains exactly three one-cell-thick hollow rectangular frames.
    - The frames are strictly nested by geometric containment, and each frame color also appears as sparse interior pattern pixels inside that frame.
    - The outer, middle, and inner frames enclose different interior sizes, so their local pattern coordinates live in different reference boxes.

    Rule:
    - Detect the three complete rectangular borders and sort them by containment depth from outside to inside.
    - For each frame, collect only the same-colored pixels strictly inside its border and treat them as that layer's local pattern.
    - Create an output grid whose size equals the innermost frame interior, scale every layer's local coordinates into that size with floor-based proportional mapping, and draw the layers from outer to inner so deeper layers overwrite shallower ones.

    Key insight:
    - The nesting topology itself is the z-order: containment gives the occlusion priority, and the differing frame interiors must be normalized to the innermost coordinate system before the visible overlaps can be resolved.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_nested_frames(g)

    inner_bbox = frames[-1]["bbox"]
    out_h = inner_bbox[2] - inner_bbox[0] - 1
    out_w = inner_bbox[3] - inner_bbox[1] - 1
    assert out_h > 0 and out_w > 0
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        src_h = r1 - r0 - 1
        src_w = c1 - c0 - 1
        coords = _pattern_local_coords(g, frame)
        mapped = _scale_coords(coords, (src_h, src_w), (out_h, out_w))
        out[mapped[:, 0], mapped[:, 1]] = int(frame["color"])

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _preimage_bounds(src_len: int, dst_len: int, dst_idx: int) -> tuple[int, int]:
    lo = (dst_idx * src_len + dst_len - 1) // dst_len
    hi = ((dst_idx + 1) * src_len + dst_len - 1) // dst_len - 1
    return int(lo), int(hi)


def _choose_source_coords(
    dst_cells: Iterable[tuple[int, int]],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)
    chosen: list[tuple[int, int]] = []
    for dr, dc in sorted(dst_cells):
        rr0, rr1 = _preimage_bounds(src_h, dst_h, int(dr))
        cc0, cc1 = _preimage_bounds(src_w, dst_w, int(dc))
        rr = random.randint(rr0, rr1)
        cc = random.randint(cc0, cc1)
        chosen.append((int(rr), int(cc)))
    coords = np.array(chosen, dtype=int)
    assert len({tuple(x) for x in coords.tolist()}) == len(chosen)
    return coords


def _sample_distinct_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    return random.sample(cells, count)


def _grow_layer(
    current: set[tuple[int, int]],
    count: int,
    h: int,
    w: int,
    banned: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    pool = [(r, c) for r in range(h) for c in range(w) if (r, c) not in current and (r, c) not in banned]
    random.shuffle(pool)
    for cell in pool:
        if len(current) >= count:
            break
        current.add(cell)
    assert len(current) == count
    return current


def _build_destination_sets(out_h: int, out_w: int) -> tuple[dict[int, set[tuple[int, int]]], int]:
    triple_conflict = 1 if random.random() < 0.5 else 0
    extra_outer_inner = 1 if random.random() < 0.35 else 0
    need = 6 + triple_conflict + extra_outer_inner
    cells = _sample_distinct_cells(out_h, out_w, need)

    idx = 0
    om = cells[idx]
    idx += 1
    mi = cells[idx]
    idx += 1
    outer_only = cells[idx]
    idx += 1
    middle_only = cells[idx]
    idx += 1
    inner_only = cells[idx]
    idx += 1
    filler = cells[idx]
    idx += 1

    outer = {om, outer_only}
    middle = {om, mi, middle_only}
    inner = {mi, inner_only}

    if triple_conflict:
        tri = cells[idx]
        idx += 1
        outer.add(tri)
        middle.add(tri)
        inner.add(tri)
    if extra_outer_inner:
        oi = cells[idx]
        outer.add(oi)
        inner.add(oi)

    outer_target = random.randint(max(2, len(outer)), 5)
    middle_target = random.randint(max(2, len(middle)), 5)
    inner_target = random.randint(max(2, len(inner)), 5)

    outer = _grow_layer(outer, outer_target, out_h, out_w, {middle_only, inner_only})
    middle = _grow_layer(middle, middle_target, out_h, out_w, {outer_only})
    inner = _grow_layer(inner, inner_target, out_h, out_w, {outer_only, middle_only})

    # Ensure every layer still affects the final composite visibly.
    assert outer_only in outer and outer_only not in middle and outer_only not in inner
    assert middle_only in middle and middle_only not in inner
    assert inner_only in inner
    # Make sure each layer count is within the puzzle range.
    for cells_set in (outer, middle, inner):
        assert 2 <= len(cells_set) <= 5
    return {OUTER_COLOR: outer, MIDDLE_COLOR: middle, INNER_COLOR: inner}, filler[0]


def _compose_from_dest_sets(
    dest_sets: dict[int, set[tuple[int, int]]],
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    out = np.full(out_shape, BACKGROUND, dtype=int)
    hit_counts: dict[tuple[int, int], int] = {}
    for color in FRAME_COLORS:
        for cell in dest_sets[color]:
            hit_counts[cell] = hit_counts.get(cell, 0) + 1
            out[cell] = color
    conflicts = sum(1 for count in hit_counts.values() if count >= 2)
    return out, conflicts


def _layer_only_image(color: int, dest_sets: dict[int, set[tuple[int, int]]], out_shape: tuple[int, int]) -> np.ndarray:
    img = np.full(out_shape, BACKGROUND, dtype=int)
    for r, c in dest_sets[color]:
        img[r, c] = color
    return img


def _gen_attempt() -> np.ndarray | None:
    out_h = random.randint(4, 6)
    out_w = random.randint(4, 6)

    gap_tm = random.randint(1, 2)
    gap_bm = random.randint(1, 2)
    gap_lm = random.randint(1, 2)
    gap_rm = random.randint(1, 2)
    mid_h = out_h + 2 + gap_tm + gap_bm
    mid_w = out_w + 2 + gap_lm + gap_rm

    gap_to = random.randint(1, 2)
    gap_bo = random.randint(1, 2)
    gap_lo = random.randint(1, 2)
    gap_ro = random.randint(1, 2)
    outer_h = mid_h + 2 + gap_to + gap_bo
    outer_w = mid_w + 2 + gap_lo + gap_ro

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    grid_h = outer_h + 2 + top_margin + bottom_margin
    grid_w = outer_w + 2 + left_margin + right_margin
    if max(grid_h, grid_w) > 30:
        return None

    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_h + 1,
        left_margin + outer_w + 1,
    )
    middle_bbox = (
        outer_bbox[0] + 1 + gap_to,
        outer_bbox[1] + 1 + gap_lo,
        outer_bbox[0] + 1 + gap_to + mid_h + 1,
        outer_bbox[1] + 1 + gap_lo + mid_w + 1,
    )
    inner_bbox = (
        middle_bbox[0] + 1 + gap_tm,
        middle_bbox[1] + 1 + gap_lm,
        middle_bbox[0] + 1 + gap_tm + out_h + 1,
        middle_bbox[1] + 1 + gap_lm + out_w + 1,
    )

    if not (_contains(outer_bbox, middle_bbox) and _contains(middle_bbox, inner_bbox)):
        return None
    if middle_bbox[0] - outer_bbox[0] < 2 or inner_bbox[0] - middle_bbox[0] < 2:
        return None

    dest_sets, _ = _build_destination_sets(out_h, out_w)
    expected, conflicts = _compose_from_dest_sets(dest_sets, (out_h, out_w))
    if conflicts < 2:
        return None
    if not ({OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR} <= set(int(v) for v in np.unique(expected))):
        return None

    shapes = {
        OUTER_COLOR: (outer_h, outer_w),
        MIDDLE_COLOR: (mid_h, mid_w),
        INNER_COLOR: (out_h, out_w),
    }
    bboxes = {
        OUTER_COLOR: outer_bbox,
        MIDDLE_COLOR: middle_bbox,
        INNER_COLOR: inner_bbox,
    }

    source_coords: dict[int, np.ndarray] = {}
    for color in FRAME_COLORS:
        source_coords[color] = _choose_source_coords(dest_sets[color], shapes[color], (out_h, out_w))
        scaled_back = _scale_coords(source_coords[color], shapes[color], (out_h, out_w))
        if set(map(tuple, scaled_back.tolist())) != dest_sets[color]:
            return None

    # Force non-trivial normalization for the outer and middle layers.
    for color in (OUTER_COLOR, MIDDLE_COLOR):
        coords = source_coords[color]
        if not np.any((coords[:, 0] >= out_h) | (coords[:, 1] >= out_w)):
            return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    _draw_frame(grid, outer_bbox, OUTER_COLOR)
    _draw_frame(grid, middle_bbox, MIDDLE_COLOR)
    _draw_frame(grid, inner_bbox, INNER_COLOR)

    for color in FRAME_COLORS:
        ir0 = bboxes[color][0] + 1
        ic0 = bboxes[color][1] + 1
        for rr, cc in source_coords[color]:
            grid[ir0 + int(rr), ic0 + int(cc)] = color

    try:
        frames = _find_nested_frames(grid)
        if [frame["color"] for frame in frames] != [OUTER_COLOR, MIDDLE_COLOR, INNER_COLOR]:
            return None
        predicted = main(grid)
    except Exception:
        return None

    if not np.array_equal(predicted, expected):
        return None

    # Additional non-degeneracy checks: every layer must matter.
    if np.array_equal(expected, _layer_only_image(OUTER_COLOR, dest_sets, (out_h, out_w))):
        return None
    if np.array_equal(expected, _layer_only_image(MIDDLE_COLOR, dest_sets, (out_h, out_w))):
        return None
    if np.array_equal(expected, _layer_only_image(INNER_COLOR, dest_sets, (out_h, out_w))):
        return None
    if int(np.count_nonzero(expected == OUTER_COLOR)) == 0 or int(np.count_nonzero(expected == MIDDLE_COLOR)) == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested occlusion input")
