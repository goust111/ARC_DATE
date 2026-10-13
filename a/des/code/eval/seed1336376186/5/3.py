from __future__ import annotations

from collections import deque
import random

import numpy as np


LEAF = "leaf"
CHAIN = "chain"
SPLIT_H = "split_h"
SPLIT_V = "split_v"

MAIN_PATTERNS = (
    (CHAIN, LEAF),
    (CHAIN, (CHAIN, LEAF)),
    (SPLIT_H, LEAF, LEAF),
    (SPLIT_V, LEAF, LEAF),
    (SPLIT_H, (CHAIN, LEAF), LEAF),
    (SPLIT_V, LEAF, (CHAIN, LEAF)),
)
EXTRA_PATTERNS = (
    LEAF,
    (CHAIN, LEAF),
)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
GENERATION_RETRIES = 400
ROOT_PLACEMENT_RETRIES = 120
ROOT_MARGIN = 1
ROOT_GAP = 2
ROOT_EXTRA_MAX = 5
MAX_RING_COUNT = 7
RING_THICKNESS = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            visited[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return ~exterior


def _component_records(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _monochrome_components(g, bg)

    records: list[dict[str, object]] = []
    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        mask = _coords_mask(g.shape, coords)
        filled = _fill_holes(mask)
        inside = filled & (~mask)
        r0, c0, r1, c1 = comp["bbox"]
        records.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "mask": mask,
                "filled": filled,
                "filled_area": int(filled.sum()),
                "inside": inside,
                "parent": None,
            }
        )

    for child_idx, child in enumerate(records):
        coords = child["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        candidates: list[int] = []
        for parent_idx, parent in enumerate(records):
            if parent_idx == child_idx:
                continue
            inside = parent["inside"]
            if bool(np.all(inside[rr, cc])):
                candidates.append(parent_idx)

        if candidates:
            child["parent"] = min(
                candidates,
                key=lambda idx: (int(records[idx]["filled_area"]), int(records[idx]["bbox_area"])),
            )

    return bg, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested ring with its direct parent's original color.

    Root rings keep their own color; every deeper ring inherits the original
    color of the ring that directly encloses it.
    """
    g = np.asarray(grid, dtype=int)
    _, records = _component_records(g)
    out = g.copy()

    for record in records:
        parent_idx = record["parent"]
        color = record["color"] if parent_idx is None else records[int(parent_idx)]["color"]
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _pattern_count(pattern: object) -> int:
    if pattern == LEAF:
        return 1
    kind = pattern[0]
    if kind == CHAIN:
        return 1 + _pattern_count(pattern[1])
    return 1 + _pattern_count(pattern[1]) + _pattern_count(pattern[2])


def _pattern_min_dims(pattern: object) -> tuple[int, int]:
    if pattern == LEAF:
        return (5, 5)

    kind = pattern[0]
    if kind == CHAIN:
        ch, cw = _pattern_min_dims(pattern[1])
        return (ch + 4, cw + 4)

    ah, aw = _pattern_min_dims(pattern[1])
    bh, bw = _pattern_min_dims(pattern[2])
    if kind == SPLIT_H:
        return (max(ah, bh) + 4, aw + bw + 5)
    return (ah + bh + 5, max(aw, bw) + 4)


def _split_nonnegative(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts:
        values.append(cut - prev - 1)
        prev = cut
    values.append(total + parts - 2 - prev)
    return values


def _sample_child_box(parent_box: tuple[int, int, int, int], pattern: object) -> tuple[int, int, int, int]:
    r, c, h, w = parent_box
    inner_h = h - 2 * RING_THICKNESS
    inner_w = w - 2 * RING_THICKNESS
    min_h, min_w = _pattern_min_dims(pattern)
    max_h = inner_h - 2
    max_w = inner_w - 2
    if max_h < min_h or max_w < min_w:
        raise ValueError("child does not fit")

    child_h = random.randint(min_h, max_h)
    child_w = random.randint(min_w, max_w)
    top_extra, bottom_extra = _split_nonnegative(inner_h - child_h - 2, 2)
    left_extra, right_extra = _split_nonnegative(inner_w - child_w - 2, 2)
    child_r = r + RING_THICKNESS + 1 + top_extra
    child_c = c + RING_THICKNESS + 1 + left_extra
    assert child_r + child_h <= r + h - RING_THICKNESS - 1 + 1
    assert child_c + child_w <= c + w - RING_THICKNESS - 1 + 1
    return (child_r, child_c, child_h, child_w)


def _sample_split_h_boxes(
    parent_box: tuple[int, int, int, int], left_pattern: object, right_pattern: object
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r, c, h, w = parent_box
    inner_h = h - 2 * RING_THICKNESS
    inner_w = w - 2 * RING_THICKNESS
    lh_min, lw_min = _pattern_min_dims(left_pattern)
    rh_min, rw_min = _pattern_min_dims(right_pattern)
    if inner_h < max(lh_min, rh_min) + 2 or inner_w < lw_min + rw_min + 3:
        raise ValueError("horizontal split does not fit")

    left_h = random.randint(lh_min, inner_h - 2)
    right_h = random.randint(rh_min, inner_h - 2)

    width_budget = inner_w - 3
    extra_width = width_budget - (lw_min + rw_min)
    grow_left = random.randint(0, extra_width)
    grow_right = random.randint(0, extra_width - grow_left)
    left_w = lw_min + grow_left
    right_w = rw_min + grow_right
    left_margin_extra, gap_extra, right_margin_extra = _split_nonnegative(
        width_budget - left_w - right_w, 3
    )
    left_margin = 1 + left_margin_extra
    gap = 1 + gap_extra

    left_top_extra, _ = _split_nonnegative(inner_h - left_h - 2, 2)
    right_top_extra, _ = _split_nonnegative(inner_h - right_h - 2, 2)
    left_r = r + RING_THICKNESS + 1 + left_top_extra
    right_r = r + RING_THICKNESS + 1 + right_top_extra
    left_c = c + RING_THICKNESS + left_margin
    right_c = left_c + left_w + gap

    return (left_r, left_c, left_h, left_w), (right_r, right_c, right_h, right_w)


def _sample_split_v_boxes(
    parent_box: tuple[int, int, int, int], top_pattern: object, bottom_pattern: object
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r, c, h, w = parent_box
    inner_h = h - 2 * RING_THICKNESS
    inner_w = w - 2 * RING_THICKNESS
    th_min, tw_min = _pattern_min_dims(top_pattern)
    bh_min, bw_min = _pattern_min_dims(bottom_pattern)
    if inner_h < th_min + bh_min + 3 or inner_w < max(tw_min, bw_min) + 2:
        raise ValueError("vertical split does not fit")

    top_w = random.randint(tw_min, inner_w - 2)
    bottom_w = random.randint(bw_min, inner_w - 2)

    height_budget = inner_h - 3
    extra_height = height_budget - (th_min + bh_min)
    grow_top = random.randint(0, extra_height)
    grow_bottom = random.randint(0, extra_height - grow_top)
    top_h = th_min + grow_top
    bottom_h = bh_min + grow_bottom
    top_margin_extra, gap_extra, bottom_margin_extra = _split_nonnegative(
        height_budget - top_h - bottom_h, 3
    )
    top_margin = 1 + top_margin_extra
    gap = 1 + gap_extra

    top_left_extra, _ = _split_nonnegative(inner_w - top_w - 2, 2)
    bottom_left_extra, _ = _split_nonnegative(inner_w - bottom_w - 2, 2)
    top_r = r + RING_THICKNESS + top_margin
    bottom_r = top_r + top_h + gap
    top_c = c + RING_THICKNESS + 1 + top_left_extra
    bottom_c = c + RING_THICKNESS + 1 + bottom_left_extra

    return (top_r, top_c, top_h, top_w), (bottom_r, bottom_c, bottom_h, bottom_w)


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r, c, h, w = box
    assert h >= 5 and w >= 5
    grid[r, c : c + w] = color
    grid[r + h - 1, c : c + w] = color
    grid[r : r + h, c] = color
    grid[r : r + h, c + w - 1] = color


def _draw_pattern(
    grid: np.ndarray,
    pattern: object,
    box: tuple[int, int, int, int],
    colors: list[int],
    color_idx: int,
) -> int:
    color = int(colors[color_idx])
    color_idx += 1
    _draw_ring(grid, box, color)

    if pattern == LEAF:
        return color_idx

    kind = pattern[0]
    if kind == CHAIN:
        child_box = _sample_child_box(box, pattern[1])
        return _draw_pattern(grid, pattern[1], child_box, colors, color_idx)

    if kind == SPLIT_H:
        left_box, right_box = _sample_split_h_boxes(box, pattern[1], pattern[2])
        color_idx = _draw_pattern(grid, pattern[1], left_box, colors, color_idx)
        return _draw_pattern(grid, pattern[2], right_box, colors, color_idx)

    top_box, bottom_box = _sample_split_v_boxes(box, pattern[1], pattern[2])
    color_idx = _draw_pattern(grid, pattern[1], top_box, colors, color_idx)
    return _draw_pattern(grid, pattern[2], bottom_box, colors, color_idx)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _sample_root_boxes(h: int, w: int, patterns: list[object]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(
        range(len(patterns)),
        key=lambda idx: _pattern_min_dims(patterns[idx])[0] * _pattern_min_dims(patterns[idx])[1],
        reverse=True,
    )
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(patterns)
    placed: list[tuple[int, int, int, int]] = []

    for idx in order:
        min_h, min_w = _pattern_min_dims(patterns[idx])
        max_h = min(h - 2 * ROOT_MARGIN, min_h + ROOT_EXTRA_MAX)
        max_w = min(w - 2 * ROOT_MARGIN, min_w + ROOT_EXTRA_MAX)
        if max_h < min_h or max_w < min_w:
            return None

        for _ in range(ROOT_PLACEMENT_RETRIES):
            box_h = random.randint(min_h, max_h)
            box_w = random.randint(min_w, max_w)
            r = random.randint(ROOT_MARGIN, h - ROOT_MARGIN - box_h)
            c = random.randint(ROOT_MARGIN, w - ROOT_MARGIN - box_w)
            candidate = (r, c, box_h, box_w)
            if all(_boxes_separated(candidate, other, ROOT_GAP) for other in placed):
                boxes[idx] = candidate
                placed.append(candidate)
                break
        else:
            return None

    return [box for box in boxes if box is not None]


def _choose_patterns() -> list[object]:
    main_pattern = random.choice(MAIN_PATTERNS)
    patterns = [main_pattern]
    total = _pattern_count(main_pattern)

    for _ in range(random.randint(0, 2)):
        candidate = random.choice(EXTRA_PATTERNS)
        if total + _pattern_count(candidate) <= MAX_RING_COUNT:
            patterns.append(candidate)
            total += _pattern_count(candidate)

    random.shuffle(patterns)
    return patterns


def _valid_generated_grid(grid: np.ndarray, expected_rings: int, bg: int) -> bool:
    if _most_common_color(grid) != bg:
        return False

    counts = np.bincount(grid.ravel(), minlength=10)
    if any(i != bg and counts[i] >= counts[bg] for i in range(len(counts))):
        return False

    _, records = _component_records(grid)
    if len(records) != expected_rings:
        return False
    if not any(record["parent"] is not None for record in records):
        return False
    if any(int(record["filled_area"]) <= len(record["coords"]) for record in records):
        return False

    out = main(grid)
    for record in records:
        coords = record["coords"]
        colors = np.unique(out[coords[:, 0], coords[:, 1]])
        if len(colors) != 1:
            return False
        parent_idx = record["parent"]
        expected = int(record["color"]) if parent_idx is None else int(records[int(parent_idx)]["color"])
        if int(colors[0]) != expected:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        patterns = _choose_patterns()
        total_rings = sum(_pattern_count(pattern) for pattern in patterns)

        min_h_needed = max(GRID_H_RANGE[0], max(_pattern_min_dims(pattern)[0] + 2 for pattern in patterns))
        min_w_needed = max(GRID_W_RANGE[0], max(_pattern_min_dims(pattern)[1] + 2 for pattern in patterns))
        if min_h_needed > GRID_H_RANGE[1] or min_w_needed > GRID_W_RANGE[1]:
            continue

        h = random.randint(min_h_needed, GRID_H_RANGE[1])
        w = random.randint(min_w_needed, GRID_W_RANGE[1])
        boxes = _sample_root_boxes(h, w, patterns)
        if boxes is None:
            continue

        bg = random.randint(0, 9)
        palette = [color for color in range(10) if color != bg]
        if total_rings > len(palette):
            continue
        colors = random.sample(palette, total_rings)

        grid = np.full((h, w), bg, dtype=int)
        color_idx = 0
        try:
            for pattern, box in zip(patterns, boxes, strict=True):
                color_idx = _draw_pattern(grid, pattern, box, colors, color_idx)
        except ValueError:
            continue

        if color_idx != total_rings:
            continue
        if _valid_generated_grid(grid, total_rings, bg):
            return grid

    raise RuntimeError("failed to generate a valid nested-ring puzzle input")
