from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, label_mask


BG = 0
GEN_ATTEMPTS = 400
GRID_H_RANGE = (17, 25)
GRID_W_RANGE = (17, 27)
OBJ_H_RANGE = (5, 8)
OBJ_W_RANGE = (5, 9)
OBJ_GAP_RANGE = (2, 5)
TOP_MARGIN_RANGE = (1, 3)
SIDE_MARGIN = 1
ANCHOR_COUNT_RANGE = (3, 5)
ANCHOR_GAP_EXTRA = 2


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    _lab, n = label_mask(mask, connectivity=4)
    return n == 1


def _shape_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _random_polyomino(area: int, h: int, w: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for cell in _neighbors4(start):
        if 0 <= cell[0] < h and 0 <= cell[1] < w:
            frontier.add(cell)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nxt in _neighbors4(cell):
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in cells:
                frontier.add(nxt)
    return cells if len(cells) == area else None


def _make_object_shape(h: int, w: int) -> np.ndarray | None:
    total = h * w
    low = max(h + w - 1, int(np.ceil(total * 0.5)))
    high = min(total - 2, int(np.floor(total * 0.78)))
    if low > high:
        return None
    for _ in range(120):
        area = random.randint(low, high)
        cells = _random_polyomino(area, h, w)
        if cells is None:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) != 0 or max(rs) != h - 1 or min(cs) != 0 or max(cs) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _mask_connected(mask):
            return mask
    return None


def _sample_damage(mask: np.ndarray) -> np.ndarray | None:
    coords = _shape_coords(mask)
    want_non_rect = random.random() < 0.75
    orig_rows = {r for r, _ in coords}
    orig_cols = {c for _, c in coords}
    for _ in range(160):
        size = random.randint(2, min(5, len(coords) - 1))
        seed = random.choice(coords)
        chosen = {seed}
        frontier = {n for n in _neighbors4(seed) if 0 <= n[0] < mask.shape[0] and 0 <= n[1] < mask.shape[1] and mask[n]}
        while len(chosen) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nxt in _neighbors4(cell):
                if 0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1] and mask[nxt] and nxt not in chosen:
                    frontier.add(nxt)
        if len(chosen) != size:
            continue
        damage = np.zeros_like(mask, dtype=bool)
        for r, c in chosen:
            damage[r, c] = True
        if not _mask_connected(damage):
            continue
        remaining = mask & ~damage
        if not _mask_connected(remaining):
            continue
        remaining_coords = _shape_coords(remaining)
        rem_rows = {r for r, _ in remaining_coords}
        rem_cols = {c for _, c in remaining_coords}
        if min(rem_rows) != min(orig_rows) or max(rem_rows) != max(orig_rows):
            continue
        if min(rem_cols) != min(orig_cols) or max(rem_cols) != max(orig_cols):
            continue
        bb = bbox(damage)
        if bb is None:
            continue
        r0, c0, r1, c1 = bb
        if want_non_rect and (r1 - r0 + 1) * (c1 - c0 + 1) == size:
            continue
        return damage
    return None


def _colorize_object(mask: np.ndarray, t_color: int, accent_colors: list[int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = t_color
    coords = _shape_coords(mask)
    max_accents = min(len(coords) // 3, 5)
    if max_accents <= 0 or not accent_colors:
        return patch
    accent_count = random.randint(1, max_accents)
    for r, c in random.sample(coords, accent_count):
        patch[r, c] = random.choice(accent_colors)
    counts = Counter(map(int, patch[mask]))
    if counts[t_color] <= max(v for k, v in counts.items() if k != t_color) if len(counts) > 1 else False:
        accent_positions = [tuple(map(int, rc)) for rc in np.argwhere((patch != 0) & (patch != t_color))]
        random.shuffle(accent_positions)
        for r, c in accent_positions:
            patch[r, c] = t_color
            counts = Counter(map(int, patch[mask]))
            if all(counts[t_color] > v for k, v in counts.items() if k != t_color):
                break
    return patch


def _stamp_output(
    template_mask: np.ndarray,
    anchors: list[tuple[int, int, int]],
    t_color: int,
    o_color: int,
    shape: tuple[int, int],
    *,
    swap_colors: bool = True,
    overlap_to_o: bool = True,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)
    counts = np.zeros((h, w), dtype=int)
    first_color = np.zeros((h, w), dtype=int)
    solid = np.argwhere(template_mask)
    center_r = template_mask.shape[0] // 2
    center_c = template_mask.shape[1] // 2

    for ar, ac, anchor_color in anchors:
        paint = o_color if anchor_color == t_color else t_color
        if not swap_colors:
            paint = anchor_color
        for rr, cc in solid:
            r = int(ar + rr - center_r)
            c = int(ac + cc - center_c)
            if 0 <= r < h and 0 <= c < w and not (r == ar and c == ac):
                counts[r, c] += 1
                if counts[r, c] == 1:
                    first_color[r, c] = paint

    out[counts == 1] = first_color[counts == 1]
    if overlap_to_o:
        out[counts >= 2] = o_color
    else:
        out[counts >= 2] = first_color[counts >= 2]

    for ar, ac, anchor_color in anchors:
        out[int(ar), int(ac)] = int(anchor_color)
    return out


def _extract_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    objects = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(objects) == 2, "expected two non-singleton foreground objects"
    objects.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))

    left, right = objects
    left_bbox = tuple(map(int, left["bbox"]))
    right_bbox = tuple(map(int, right["bbox"]))
    assert crop_bbox(g, left_bbox).shape == crop_bbox(g, right_bbox).shape, "object bboxes must match"

    full_comp, damaged_comp = (left, right) if int(left["area"]) > int(right["area"]) else (right, left)
    full_bbox = tuple(map(int, full_comp["bbox"]))
    damaged_bbox = tuple(map(int, damaged_comp["bbox"]))
    full_crop = crop_bbox(g, full_bbox)
    damaged_crop = crop_bbox(g, damaged_bbox)
    assert full_crop.shape == damaged_crop.shape

    damage_mask = (full_crop != BG) & (damaged_crop == BG)
    assert 2 <= int(damage_mask.sum()) <= 5, "damage must contain 2..5 cells"
    _lab, n_damage = label_mask(damage_mask, connectivity=4)
    assert n_damage == 1, "damage mask must be 4-connected"
    damage_bbox = bbox(damage_mask)
    assert damage_bbox is not None
    template_mask = crop_bbox(damage_mask.astype(int), damage_bbox).astype(bool)

    full_colors = [int(v) for v in full_crop[full_crop != BG]]
    counts = Counter(full_colors).most_common()
    assert len(counts) >= 1
    t_color = int(counts[0][0])
    if len(counts) > 1:
        assert counts[0][1] > counts[1][1], "complete object needs a unique dominant color"

    object_coords = {
        tuple(map(int, rc))
        for comp in (full_comp, damaged_comp)
        for rc in comp["coords"]
    }
    anchors = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(g != BG)
        if (int(r), int(c)) not in object_coords
    ]
    assert len(anchors) >= 2, "expected at least two anchors"
    t_anchors = [anchor for anchor in anchors if anchor[2] == t_color]
    assert len(t_anchors) == 1, "expected exactly one T-colored anchor"
    other_anchor_colors = sorted({anchor[2] for anchor in anchors if anchor[2] != t_color})
    assert len(other_anchor_colors) == 1, "expected exactly one shared O anchor color"
    o_color = int(other_anchor_colors[0])

    return {
        "grid": g,
        "template_mask": template_mask,
        "anchors": anchors,
        "t_color": t_color,
        "o_color": o_color,
        "full_bbox": full_bbox,
        "damaged_bbox": damaged_bbox,
        "damage_mask": damage_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the damaged subshape from the aligned top copies, then stamp that mask at the lower anchor pixels.

    Input structure:
    - The background is black.
    - The upper area contains two aligned multicolor 4-connected objects with equal bbox size; one copy is intact and the other has a small connected region erased to black.
    - The lower area contains singleton anchor pixels: exactly one has the intact copy's dominant color T, while all other anchors share a different color O.

    Rule:
    - Crop the two upper objects to their own equal-sized bboxes and compare them cell by cell.
    - Collect the cells that are foreground in the intact copy and black in the damaged copy; this connected set is the damage mask.
    - Crop that mask to its own bbox to obtain the stamp template, align the template's lower-right geometric center cell to each anchor, and clip any out-of-bounds parts.
    - At the T-colored anchor, paint the template in color O; at every O-colored anchor, paint the template in color T; if multiple copies overlap, recolor the overlapping cells to O; keep anchor pixels unchanged.

    Key insight:
    - The stamped template is not explicitly drawn anywhere in the input: it must be extracted by comparing the intact and damaged copies before the anchor-guided color-swapped stamping can be done.
    """
    case = _extract_case(np.asarray(grid, dtype=int))
    return _stamp_output(
        case["template_mask"],
        case["anchors"],
        case["t_color"],
        case["o_color"],
        case["grid"].shape,
    )


def _find_overlap_offset(template_mask: np.ndarray) -> tuple[int, int] | None:
    solid = {tuple(map(int, rc)) for rc in np.argwhere(template_mask)}
    max_dim = max(template_mask.shape)
    for dr in range(-(max_dim + 1), max_dim + 2):
        for dc in range(-(max_dim + 1), max_dim + 2):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) < max_dim:
                continue
            shifted = {(r + dr, c + dc) for r, c in solid}
            if solid & shifted:
                return dr, dc
    return None


def _candidate_anchor_points(
    grid_shape: tuple[int, int],
    lower_start: int,
    template_mask: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    max_dim = max(template_mask.shape)
    points = []
    for r in range(lower_start, h):
        for c in range(w):
            near_edge = r < template_mask.shape[0] or c < template_mask.shape[1] or r >= h - template_mask.shape[0] or c >= w - template_mask.shape[1]
            if near_edge and random.random() < 0.7:
                continue
            points.append((r, c))
    random.shuffle(points)
    if points:
        return points
    return [(r, c) for r in range(lower_start, h) for c in range(w) if max_dim >= 0]


def _spaced_enough(points: list[tuple[int, int]], candidate: tuple[int, int], min_dist: int) -> bool:
    return all(abs(candidate[0] - r) + abs(candidate[1] - c) >= min_dist for r, c in points)


def _sample_anchors(
    grid_shape: tuple[int, int],
    lower_start: int,
    template_mask: np.ndarray,
    count: int,
    want_overlap: bool,
) -> list[tuple[int, int]] | None:
    min_dist = max(template_mask.shape)
    candidates = _candidate_anchor_points(grid_shape, lower_start, template_mask)
    anchors: list[tuple[int, int]] = []

    if want_overlap:
        overlap_offset = _find_overlap_offset(template_mask)
        if overlap_offset is None:
            return None
        dr, dc = overlap_offset
        for base in candidates:
            other = (base[0] + dr, base[1] + dc)
            if other[0] < lower_start or other[0] >= grid_shape[0] or other[1] < 0 or other[1] >= grid_shape[1]:
                continue
            if not _spaced_enough([], base, min_dist):
                continue
            if not _spaced_enough([base], other, min_dist):
                continue
            anchors = [base, other]
            break
        if not anchors:
            return None

    for point in candidates:
        if point in anchors:
            continue
        if _spaced_enough(anchors, point, min_dist):
            anchors.append(point)
        if len(anchors) == count:
            return anchors
    return None


def _build_case() -> dict | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    obj_h = random.randint(*OBJ_H_RANGE)
    obj_w = random.randint(*OBJ_W_RANGE)
    if obj_w * 2 + max(OBJ_GAP_RANGE) + SIDE_MARGIN * 2 > w:
        return None

    shape_mask = _make_object_shape(obj_h, obj_w)
    if shape_mask is None:
        return None
    damage_mask = _sample_damage(shape_mask)
    if damage_mask is None:
        return None

    damage_bbox = bbox(damage_mask)
    assert damage_bbox is not None
    template_mask = crop_bbox(damage_mask.astype(int), damage_bbox).astype(bool)

    palette = random.sample(range(1, 10), 4)
    t_color = int(palette[0])
    o_color = int(palette[1])
    accent_colors = [int(palette[2]), int(palette[3])]
    full_patch = _colorize_object(shape_mask, t_color, accent_colors)
    damaged_patch = full_patch.copy()
    damaged_patch[damage_mask] = BG

    top = random.randint(*TOP_MARGIN_RANGE)
    gap = random.randint(*OBJ_GAP_RANGE)
    max_left = w - SIDE_MARGIN - (obj_w * 2 + gap)
    if max_left < SIDE_MARGIN:
        return None
    left = random.randint(SIDE_MARGIN, max_left)
    right = left + obj_w + gap
    lower_start = top + obj_h + ANCHOR_GAP_EXTRA
    if lower_start >= h - 2:
        return None

    want_overlap = random.random() < 0.55
    anchor_count = random.randint(*ANCHOR_COUNT_RANGE)
    anchors_xy = _sample_anchors((h, w), lower_start, template_mask, anchor_count, want_overlap)
    if anchors_xy is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[top : top + obj_h, left : left + obj_w] = full_patch
    grid[top : top + obj_h, right : right + obj_w] = damaged_patch

    if want_overlap:
        t_anchor_idx = random.randrange(2, anchor_count)
    else:
        t_anchor_idx = random.randrange(anchor_count)
    anchors: list[tuple[int, int, int]] = []
    for i, (r, c) in enumerate(anchors_xy):
        color = t_color if i == t_anchor_idx else o_color
        if grid[r, c] != BG:
            return None
        grid[r, c] = color
        anchors.append((r, c, color))

    expected = _stamp_output(template_mask, anchors, t_color, o_color, grid.shape)
    rect_template = np.ones_like(template_mask, dtype=bool)
    rect_output = _stamp_output(rect_template, anchors, t_color, o_color, grid.shape)
    no_swap_output = _stamp_output(template_mask, anchors, t_color, o_color, grid.shape, swap_colors=False)
    full_shape_output = _stamp_output(shape_mask, anchors, t_color, o_color, grid.shape)
    overlap_output = _stamp_output(template_mask, anchors, t_color, o_color, grid.shape, overlap_to_o=False)

    if np.array_equal(expected, rect_output):
        return None
    if np.array_equal(expected, no_swap_output):
        return None
    if np.array_equal(expected, full_shape_output):
        return None
    if want_overlap and np.array_equal(expected, overlap_output):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    parsed = _extract_case(grid)
    if not np.array_equal(parsed["template_mask"], template_mask):
        return None
    if sorted(parsed["anchors"]) != sorted(anchors):
        return None

    return {"input": grid, "output": expected}


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid ARC input")
