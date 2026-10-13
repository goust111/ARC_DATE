from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_ref", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

DIFF_COLORS = (Color.BLUE, Color.RED)
SOURCE_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
GRID_SIDE_MIN = 13
GRID_SIDE_MAX = 20
MARKER_COUNT_CHOICES = (2, 3, 4)
GEN_RETRY_LIMIT = 400
MASK_RETRY_LIMIT = 200


def _rects_disjoint(a: BBox, b: BBox, pad: int = 0) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _neighbors4(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = cell
    h, w = shape
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _edge_flags(cells: set[Coord], h: int, w: int) -> tuple[bool, bool, bool, bool]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return 0 in rows, (h - 1) in rows, 0 in cols, (w - 1) in cols


def _has_full_bbox(cells: set[Coord], h: int, w: int) -> bool:
    top, bottom, left, right = _edge_flags(cells, h, w)
    return top and bottom and left and right


def _frontier(cells: set[Coord], shape: tuple[int, int], forbidden: set[Coord]) -> list[Coord]:
    candidates: set[Coord] = set()
    for cell in cells:
        for nbr in _neighbors4(cell, shape):
            if nbr not in cells and nbr not in forbidden:
                candidates.add(nbr)
    return list(candidates)


def _candidate_weight(cell: Coord, cells: set[Coord], h: int, w: int) -> float:
    r, c = cell
    current = _edge_flags(cells, h, w)
    expanded = _edge_flags(cells | {cell}, h, w)
    gain = sum(int(new and not old) for old, new in zip(current, expanded))

    pull = 0
    if not current[0]:
        pull += h - 1 - r
    if not current[1]:
        pull += r
    if not current[2]:
        pull += w - 1 - c
    if not current[3]:
        pull += c

    degree = sum(int(nbr in cells) for nbr in _neighbors4(cell, (h, w)))
    return 1.0 + gain * 20.0 + pull * 2.0 + degree + random.random()


def _gen_template_mask(h: int, w: int, center_filled: bool) -> np.ndarray:
    center = (h // 2, w // 2)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    forbidden = set() if center_filled else {center}
    starts = [cell for cell in all_cells if cell not in forbidden]
    min_area = h + w - 1
    max_area = h * w - 1

    for _ in range(MASK_RETRY_LIMIT):
        target_area = random.randint(min_area, max_area)
        start = center if center_filled else random.choice(starts)
        cells: set[Coord] = {start}

        while len(cells) < target_area or not _has_full_bbox(cells, h, w):
            frontier = _frontier(cells, (h, w), forbidden)
            if not frontier:
                break
            weights = [_candidate_weight(cell, cells, h, w) for cell in frontier]
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(cells) >= target_area and _has_full_bbox(cells, h, w):
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            assert bool(mask.any())
            assert len(components_mask(mask, connectivity=4)) == 1
            assert center_filled == bool(mask[center])
            assert bbox_h(bbox(mask)) == h
            assert bbox_w(bbox(mask)) == w
            return mask

    raise RuntimeError("failed to generate connected symmetric-difference mask")


def _grow_connected_subset(cells: set[Coord], target_size: int) -> set[Coord]:
    max_r = max(r for r, _c in cells) + 1
    max_c = max(c for _r, c in cells) + 1
    chosen = {random.choice(tuple(cells))}
    while len(chosen) < target_size:
        frontier: set[Coord] = set()
        for cell in chosen:
            for nbr in _neighbors4(cell, (max_r, max_c)):
                if nbr in cells and nbr not in chosen:
                    frontier.add(nbr)
        if not frontier:
            break
        chosen.add(random.choice(tuple(frontier)))
    return chosen


def _split_red_blue(mask: np.ndarray) -> tuple[set[Coord], set[Coord]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    min_size = max(1, len(cells) // 4)
    max_size = min(len(cells) - 1, len(cells) - min_size)
    target = random.randint(min_size, max_size)
    subset = _grow_connected_subset(cells, target)
    if len(subset) == 0 or len(subset) == len(cells):
        subset = set(list(cells)[: max(1, len(cells) // 2)])
    other = cells - subset
    if random.random() < 0.5:
        return subset, other
    return other, subset


def _green_candidates(mask: np.ndarray) -> tuple[list[Coord], list[Coord]]:
    h, w = mask.shape
    patch_h = h + 2
    patch_w = w + 2
    all_candidates: list[Coord] = []
    internal: list[Coord] = []
    diff_cells = {(r + 1, c + 1) for r, c in np.argwhere(mask)}

    for r in range(patch_h):
        for c in range(patch_w):
            if (r, c) in diff_cells:
                continue
            for nbr in _neighbors4((r, c), (patch_h, patch_w)):
                if nbr in diff_cells:
                    all_candidates.append((r, c))
                    if 1 <= r <= h and 1 <= c <= w:
                        internal.append((r, c))
                    break

    external = [cell for cell in all_candidates if cell not in internal]
    return internal, external


def _build_encoded_patch(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    red_cells, blue_cells = _split_red_blue(mask)
    for r, c in red_cells:
        patch[r + 1, c + 1] = Color.RED
    for r, c in blue_cells:
        patch[r + 1, c + 1] = Color.BLUE

    internal_green, external_green = _green_candidates(mask)
    green_count = random.randint(1, min(4, len(internal_green) + len(external_green)))
    chosen: list[Coord] = []
    if internal_green and random.random() < 0.7:
        chosen.append(random.choice(internal_green))
    if external_green and random.random() < 0.8 and len(chosen) < green_count:
        pool = [cell for cell in external_green if cell not in chosen]
        if pool:
            chosen.append(random.choice(pool))
    remaining = [cell for cell in internal_green + external_green if cell not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: green_count - len(chosen)])
    for r, c in chosen:
        patch[r, c] = Color.GREEN

    template = np.zeros((h, w), dtype=int)
    template[mask] = Color.YELLOW
    return patch, template


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, BBox, list[Coord]]:
    diff_mask = np.isin(grid, DIFF_COLORS)
    diff_components = components_mask(diff_mask, connectivity=4)
    assert len(diff_components) == 1, "expected one 4-connected symmetric-difference region"

    diff_bbox = bbox(diff_mask)
    assert diff_bbox is not None
    assert bbox_h(diff_bbox) % 2 == 1 and bbox_w(diff_bbox) % 2 == 1

    patch = crop_bbox(grid, diff_bbox)
    template = np.zeros_like(patch)
    template[np.isin(patch, DIFF_COLORS)] = Color.YELLOW

    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.GREY)]
    assert markers, "expected at least one gray marker"
    return template, diff_bbox, markers


def _stamp_cells(template: np.ndarray, center: Coord, marker: Coord) -> list[Coord]:
    center_r, center_c = center
    marker_r, marker_c = marker
    cells: list[Coord] = []
    for tr, tc in np.argwhere(template == Color.YELLOW):
        rr = marker_r + int(tr) - center_r
        cc = marker_c + int(tc) - center_c
        cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the red-blue symmetric difference as a template and stamp it onto every gray marker.

    Input structure:
    - A black background contains one mixed red/blue/green encoded region and several isolated gray marker pixels.
    - Red and blue pixels together form one 4-connected component whose bounding box has odd height and odd width.
    - Green pixels belong to the same encoded region but do not belong to the stamp template.

    Rule:
    - Treat every red or blue cell as symmetric-difference foreground and crop its bounding box.
    - Inside that cropped box, recolor the red/blue foreground to yellow and treat every other cell as transparent.
    - Center this template on each gray marker and paint yellow only onto black cells, except that the gray marker itself is replaced when the template center is yellow.

    Key insight:
    - The visible red/blue/green patch encodes two hidden overlapping masks, and only the red-plus-blue disagreement cells define the reusable stamp.
    """
    g = np.asarray(grid, dtype=int)
    template, _diff_bbox, markers = _extract_template(g)
    center = (template.shape[0] // 2, template.shape[1] // 2)

    out = g.copy()
    for marker in markers:
        for rr, cc in _stamp_cells(template, center, marker):
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            if out[rr, cc] == Color.BLACK:
                out[rr, cc] = Color.YELLOW
            elif (rr, cc) == marker and out[rr, cc] == Color.GREY:
                out[rr, cc] = Color.YELLOW
    return out


def _marker_bbox(marker: Coord, template_shape: tuple[int, int]) -> BBox:
    h, w = template_shape
    r, c = marker
    return (r - h // 2, c - w // 2, r + h // 2, c + w // 2)


def _valid_markers(
    grid_shape: tuple[int, int], template_shape: tuple[int, int], source_bbox: BBox
) -> list[Coord]:
    h, w = grid_shape
    th, tw = template_shape
    valid: list[Coord] = []
    for r in range(th // 2, h - th // 2):
        for c in range(tw // 2, w - tw // 2):
            stamp_bbox = _marker_bbox((r, c), template_shape)
            if _rects_disjoint(stamp_bbox, source_bbox, pad=1):
                valid.append((r, c))
    return valid


def _pick_markers(valid: list[Coord], template_shape: tuple[int, int], count: int) -> list[Coord] | None:
    choices = valid[:]
    random.shuffle(choices)
    picked: list[Coord] = []
    picked_boxes: list[BBox] = []
    for marker in choices:
        stamp_bbox = _marker_bbox(marker, template_shape)
        if all(_rects_disjoint(stamp_bbox, other, pad=1) for other in picked_boxes):
            picked.append(marker)
            picked_boxes.append(stamp_bbox)
            if len(picked) == count:
                return picked
    return None


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    template, _diff_bbox, markers = _extract_template(inp)
    center = (template.shape[0] // 2, template.shape[1] // 2)
    center_is_yellow = bool(template[center] == Color.YELLOW)
    source_mask = np.isin(inp, SOURCE_COLORS)

    if not np.array_equal(out[source_mask], inp[source_mask]):
        return False

    expected_yellow: set[Coord] = set()
    for marker in markers:
        for rr, cc in _stamp_cells(template, center, marker):
            tr = rr - marker[0] + center[0]
            tc = cc - marker[1] + center[1]
            if inp[rr, cc] == Color.BLACK:
                expected_yellow.add((rr, cc))
            elif (rr, cc) == marker and (tr, tc) == center and inp[rr, cc] == Color.GREY:
                expected_yellow.add((rr, cc))

    for marker in markers:
        if center_is_yellow:
            if int(out[marker]) != Color.YELLOW:
                return False
        else:
            if int(out[marker]) != Color.GREY:
                return False

    changed = {
        (r, c)
        for r, c in np.argwhere(out != inp)
    }
    if changed != expected_yellow:
        return False

    for r, c in expected_yellow:
        if int(out[r, c]) != Color.YELLOW:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template_h = random.choice((3, 5))
    template_w = random.choice((3, 5))
    center_filled = bool(random.getrandbits(1))
    mask = _gen_template_mask(template_h, template_w, center_filled)
    encoded_patch, template = _build_encoded_patch(mask)
    patch_h, patch_w = encoded_patch.shape

    grid_h = random.randint(max(GRID_SIDE_MIN, patch_h + template_h + 3), GRID_SIDE_MAX)
    grid_w = random.randint(max(GRID_SIDE_MIN, patch_w + template_w + 5), GRID_SIDE_MAX)
    if grid_h > 30 or grid_w > 30:
        return None

    max_row = grid_h - patch_h - 2
    max_col = max(1, grid_w // 2 - patch_w)
    if max_row < 1 or max_col < 1:
        return None

    patch_r = random.randint(1, max_row)
    patch_c = random.randint(1, max_col)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[patch_r : patch_r + patch_h, patch_c : patch_c + patch_w] = encoded_patch
    source_bbox = (patch_r, patch_c, patch_r + patch_h - 1, patch_c + patch_w - 1)

    marker_count = random.choice(MARKER_COUNT_CHOICES)
    valid_markers = _valid_markers(inp.shape, template.shape, source_bbox)
    markers = _pick_markers(valid_markers, template.shape, marker_count)
    if markers is None:
        return None
    for r, c in markers:
        inp[r, c] = Color.GREY

    out = main(inp)
    if not _semantic_check(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetric-difference stamp input")
