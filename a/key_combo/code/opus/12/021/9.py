from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
RED = 2
PINK = 6
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_GRID_SIZES = [13, 14, 15, 16, 17, 18, 19]
GEN_OCCLUDER_SIZES = [3, 4, 5]
GEN_BG_AND_DECO = [0, 3, 4, 5, 7, 8, 9]
GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 200
DECOR_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if connectivity != 4:
        raise ValueError("Only 4-connectivity is used here")
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _largest_solid_square(grid: np.ndarray) -> dict:
    squares = []
    color = PINK
    for coords in components_mask(grid == color, connectivity=4):
        box = bbox_coords(coords)
        h = bbox_h(box)
        w = bbox_w(box)
        if h == w and len(coords) == h * w:
            r0, c0, r1, c1 = box
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                squares.append({"color": color, "coords": coords, "bbox": box, "area": len(coords)})
    assert squares, "expected at least one solid square"
    best_side = max(bbox_h(comp["bbox"]) for comp in squares)
    largest = [comp for comp in squares if bbox_h(comp["bbox"]) == best_side]
    assert len(largest) == 1, "expected a unique largest solid square"
    return largest[0]


def _counterpart_bbox(box: tuple[int, int, int, int], shape: tuple[int, int], axis: str) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    if axis == "vertical":
        return r0, w - 1 - c1, r1, w - 1 - c0
    if axis == "horizontal":
        return h - 1 - r1, c0, h - 1 - r0, c1
    raise ValueError(axis)


def _flip_patch(patch: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        return np.fliplr(patch)
    if axis == "horizontal":
        return np.flipud(patch)
    raise ValueError(axis)


def _symmetry_score(grid: np.ndarray, box: tuple[int, int, int, int], axis: str) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    occluded = np.zeros((h, w), dtype=bool)
    occluded[r0 : r1 + 1, c0 : c1 + 1] = True
    mismatches = 0
    comparisons = 0
    for r in range(h):
        for c in range(w):
            if axis == "vertical":
                rr, cc = r, w - 1 - c
            else:
                rr, cc = h - 1 - r, c
            if (r > rr) or (r == rr and c >= cc):
                continue
            if occluded[r, c] or occluded[rr, cc]:
                continue
            comparisons += 1
            if int(grid[r, c]) != int(grid[rr, cc]):
                mismatches += 1
    return mismatches, comparisons


def _recover_patch(grid: np.ndarray, box: tuple[int, int, int, int], axis: str, occluder_color: int) -> np.ndarray:
    other = _counterpart_bbox(box, grid.shape, axis)
    patch = crop_bbox(grid, other)
    assert not np.any(patch == occluder_color), "counterpart must be fully visible"
    return _flip_patch(patch, axis)


def _valid_core_rect(
    object_cells: set[tuple[int, int]],
    missing: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    rect: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = rect

    for r, c in missing:
        distances = [r - r0, r1 - r, c - c0, c1 - c]
        if distances.count(min(distances)) != 1:
            return False

    for r, c in outside:
        side_count = int(r < r0) + int(r > r1) + int(c < c0) + int(c > c1)
        if side_count != 1:
            return False
        if r < r0:
            inner = (r + 1, c)
        elif r > r1:
            inner = (r - 1, c)
        elif c < c0:
            inner = (r, c + 1)
        else:
            inner = (r, c - 1)
        if inner not in object_cells:
            return False

    return True


def _core_rect(coords: np.ndarray) -> tuple[int, int, int, int]:
    object_cells = {tuple(map(int, p)) for p in coords}
    r0, c0, r1, c1 = bbox_coords(coords)

    candidates = []
    for rr0 in (r0, r0 + 1):
        for rr1 in (r1, r1 - 1):
            for cc0 in (c0, c0 + 1):
                for cc1 in (c1, c1 - 1):
                    if rr0 > rr1 or cc0 > cc1:
                        continue
                    rect_cells = {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}
                    missing = rect_cells - object_cells
                    outside = object_cells - rect_cells
                    rect = (rr0, cc0, rr1, cc1)
                    if not _valid_core_rect(object_cells, missing, outside, rect):
                        continue
                    score = (len(missing) + len(outside), -len(rect_cells))
                    candidates.append((score, rect))

    assert candidates, "failed to infer a near-solid core rectangle"
    best_score = min(score for score, _ in candidates)
    best_rects = [rect for score, rect in candidates if score == best_score]
    assert len(best_rects) == 1, "core rectangle must be unique"
    return best_rects[0]


def _nearest_rect_side(r: int, c: int, rect: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = rect
    distances = {
        "top": r - r0,
        "bottom": r1 - r,
        "left": c - c0,
        "right": c1 - c,
    }
    nearest = [side for side, dist in distances.items() if dist == min(distances.values())]
    assert len(nearest) == 1, "defect must have a unique nearest core side"
    return nearest[0]


def _defect_bar_cells(coords: np.ndarray, rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    object_cells = {tuple(map(int, p)) for p in coords}
    r0, c0, r1, c1 = rect
    rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    bars: set[tuple[int, int]] = set()

    for r, c in rect_cells - object_cells:
        side = _nearest_rect_side(r, c, rect)
        if side == "top":
            bars.update({(r - 1, c), (r, c)})
        elif side == "bottom":
            bars.update({(r, c), (r + 1, c)})
        elif side == "left":
            bars.update({(r, c - 1), (r, c)})
        else:
            bars.update({(r, c), (r, c + 1)})

    for r, c in object_cells - rect_cells:
        if r < r0:
            bars.update({(r, c), (r + 1, c)})
        elif r > r1:
            bars.update({(r - 1, c), (r, c)})
        elif c < c0:
            bars.update({(r, c), (r, c + 1)})
        else:
            bars.update({(r, c - 1), (r, c)})

    return bars


def _choose_blue_object(patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(patch == BLUE, connectivity=4)
    candidates = []
    for coords in comps:
        if len(coords) < 4:
            continue
        try:
            rect = _core_rect(coords)
        except AssertionError:
            continue
        rect_area = bbox_h(rect) * bbox_w(rect)
        candidates.append(((rect_area, len(coords)), coords, rect))
    assert candidates, "expected a near-solid blue rectangle inside the recovered patch"
    best = max(score for score, _, _ in candidates)
    winners = [(coords, rect) for score, coords, rect in candidates if score == best]
    assert len(winners) == 1, "blue rectangle must be uniquely identifiable"
    return winners[0]


def _mark_defects(patch: np.ndarray) -> np.ndarray:
    out = np.asarray(patch, dtype=int).copy()
    coords, rect = _choose_blue_object(out)
    for r, c in _defect_bar_cells(coords, rect):
        assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
        out[r, c] = RED
    return out


def _detect_axis(grid: np.ndarray, box: tuple[int, int, int, int]) -> str:
    candidates = []
    for axis in ("vertical", "horizontal"):
        mismatches, comparisons = _symmetry_score(grid, box, axis)
        if comparisons > 0 and mismatches == 0:
            candidates.append(axis)
    assert len(candidates) == 1, "expected exactly one usable mirror axis"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded symmetric patch, then mark the near-rectangular blue defect bars in red.

    Input structure:
    - The grid is an almost square scene with exactly one solid magenta square occluder.
    - Outside the occluder, the scene is mirror-symmetric across exactly one axis: vertical or horizontal.
    - The hidden patch contains a near-solid blue rectangle whose only irregularities are one or two two-cell defect bars.

    Rule:
    - Find the unique largest solid magenta square and determine which mirror axis still matches everywhere else.
    - Reflect the occluder bbox across that axis, copy the visible counterpart back with the inverse flip, and recover the hidden patch.
    - Inside the recovered patch, infer the largest blue rectangular core and recolor every two-cell defect bar to red.

    Key insight:
    - The visible mirror counterpart reveals the missing patch, but recovery alone is incomplete: the restored blue object must still be analyzed for edge defects and those short bars must be highlighted.
    """
    g = np.asarray(grid, dtype=int)
    occluder = _largest_solid_square(g)
    assert int(occluder["color"]) == PINK, "the occluder color is fixed and unique"
    r0, c0, r1, c1 = occluder["bbox"]
    side = bbox_h(occluder["bbox"])
    assert side == bbox_w(occluder["bbox"])
    assert np.count_nonzero(g == PINK) == side * side, "magenta must appear only in the occluder"

    axis = _detect_axis(g, occluder["bbox"])
    recovered = _recover_patch(g, occluder["bbox"], axis, PINK)
    return _mark_defects(recovered)


def _bar_cells_for_defect(kind: str, side: str, cell: tuple[int, int], rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r, c = cell
    if kind == "missing":
        if side == "top":
            return {(r - 1, c), (r, c)}
        if side == "bottom":
            return {(r, c), (r + 1, c)}
        if side == "left":
            return {(r, c - 1), (r, c)}
        return {(r, c), (r, c + 1)}
    if side == "top":
        return {(r, c), (r + 1, c)}
    if side == "bottom":
        return {(r - 1, c), (r, c)}
    if side == "left":
        return {(r, c), (r, c + 1)}
    return {(r, c - 1), (r, c)}


def _patch_candidates(size: int, rect: tuple[int, int, int, int]) -> list[tuple[str, str, tuple[int, int], set[tuple[int, int]]]]:
    r0, c0, r1, c1 = rect
    out = []
    if r0 > 0:
        for c in range(c0, c1 + 1):
            out.append(("protrusion", "top", (r0 - 1, c), {(r0 - 1, c), (r0, c)}))
        if c1 - c0 + 1 >= 3:
            for c in range(c0 + 1, c1):
                out.append(("missing", "top", (r0, c), {(r0 - 1, c), (r0, c)}))
    if r1 < size - 1:
        for c in range(c0, c1 + 1):
            out.append(("protrusion", "bottom", (r1 + 1, c), {(r1, c), (r1 + 1, c)}))
        if c1 - c0 + 1 >= 3:
            for c in range(c0 + 1, c1):
                out.append(("missing", "bottom", (r1, c), {(r1, c), (r1 + 1, c)}))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            out.append(("protrusion", "left", (r, c0 - 1), {(r, c0 - 1), (r, c0)}))
        if r1 - r0 + 1 >= 3:
            for r in range(r0 + 1, r1):
                out.append(("missing", "left", (r, c0), {(r, c0 - 1), (r, c0)}))
    if c1 < size - 1:
        for r in range(r0, r1 + 1):
            out.append(("protrusion", "right", (r, c1 + 1), {(r, c1), (r, c1 + 1)}))
        if r1 - r0 + 1 >= 3:
            for r in range(r0 + 1, r1):
                out.append(("missing", "right", (r, c1), {(r, c1), (r, c1 + 1)}))
    return out


def _make_patch(size: int, bg: int, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_ATTEMPTS):
        patch = np.full((size, size), bg, dtype=int)
        rect_h = random.randint(2, size)
        rect_w = random.randint(2, size)
        r0 = random.randint(0, size - rect_h)
        c0 = random.randint(0, size - rect_w)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        patch[r0 : rect[2] + 1, c0 : rect[3] + 1] = BLUE

        defect_choices = _patch_candidates(size, rect)
        if not defect_choices:
            continue
        random.shuffle(defect_choices)
        defect_count = 2 if len(defect_choices) >= 2 and random.random() < 0.55 else 1
        chosen = []
        used_bars: set[tuple[int, int]] = set()
        used_sides: set[str] = set()
        for kind, side, cell, bars in defect_choices:
            if side in used_sides:
                continue
            if bars & used_bars:
                continue
            chosen.append((kind, side, cell, bars))
            used_sides.add(side)
            used_bars |= bars
            if len(chosen) == defect_count:
                break
        if len(chosen) != defect_count:
            continue

        designed_bars: set[tuple[int, int]] = set()
        for kind, side, cell, bars in chosen:
            rr, cc = cell
            if kind == "missing":
                patch[rr, cc] = bg
            else:
                patch[rr, cc] = BLUE
            designed_bars |= bars

        comps = components_mask(patch == BLUE, connectivity=4)
        if len(comps) != 1:
            continue
        coords = comps[0]
        try:
            inferred = _core_rect(coords)
        except AssertionError:
            continue
        if inferred != rect:
            continue
        inferred_bars = _defect_bar_cells(coords, rect)
        if inferred_bars != designed_bars:
            continue

        expected = patch.copy()
        for rr, cc in designed_bars:
            expected[rr, cc] = RED

        flipped_expected = _flip_patch(expected, axis)
        if np.array_equal(expected, flipped_expected):
            continue
        if np.array_equal(expected, patch):
            continue
        if np.count_nonzero(expected == RED) < 2:
            continue
        return patch, expected
    return None


def _place_patch_pair(grid: np.ndarray, box: tuple[int, int, int, int], patch: np.ndarray, axis: str) -> None:
    other = _counterpart_bbox(box, grid.shape, axis)
    r0, c0, r1, c1 = box
    or0, oc0, or1, oc1 = other
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    grid[or0 : or1 + 1, oc0 : oc1 + 1] = _flip_patch(patch, axis)


def _reserve_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], pad: int = 1) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in boxes:
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h - 1, r1 + pad)
        cc1 = min(w - 1, c1 + pad)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _try_place_decor(grid: np.ndarray, reserved: np.ndarray, axis: str, color: int) -> bool:
    h, w = grid.shape
    shapes = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    dh, dw = random.choice(shapes)
    if axis == "vertical":
        max_c = (w - dw) // 2 - 1
        if max_c < 0:
            return False
        r0 = random.randint(0, h - dh)
        c0 = random.randint(0, max_c)
        left = (r0, c0, r0 + dh - 1, c0 + dw - 1)
        right = _counterpart_bbox(left, grid.shape, axis)
        if _boxes_overlap(left, right):
            return False
    else:
        max_r = (h - dh) // 2 - 1
        if max_r < 0:
            return False
        r0 = random.randint(0, max_r)
        c0 = random.randint(0, w - dw)
        left = (r0, c0, r0 + dh - 1, c0 + dw - 1)
        right = _counterpart_bbox(left, grid.shape, axis)
        if _boxes_overlap(left, right):
            return False

    for box in (left, right):
        br0, bc0, br1, bc1 = box
        if reserved[br0 : br1 + 1, bc0 : bc1 + 1].any():
            return False
        if np.any(grid[br0 : br1 + 1, bc0 : bc1 + 1] != get_bg(grid)):
            return False

    for box in (left, right):
        br0, bc0, br1, bc1 = box
        grid[br0 : br1 + 1, bc0 : bc1 + 1] = color
        reserved[br0 : br1 + 1, bc0 : bc1 + 1] = True
    return True


def _candidate_hidden_boxes(shape: tuple[int, int], size: int, axis: str) -> list[tuple[int, int, int, int]]:
    h, w = shape
    boxes = []
    for r0 in range(h - size + 1):
        for c0 in range(w - size + 1):
            box = (r0, c0, r0 + size - 1, c0 + size - 1)
            other = _counterpart_bbox(box, shape, axis)
            if box == other or _boxes_overlap(box, other):
                continue
            if axis == "vertical" and other[1] == c0:
                continue
            if axis == "horizontal" and other[0] == r0:
                continue
            boxes.append(box)
    return boxes


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GEN_GRID_SIZES)
    axis = random.choice(["vertical", "horizontal"])
    occ = random.choice(GEN_OCCLUDER_SIZES)
    bg = random.choice(GEN_BG_AND_DECO)
    patch_pair = _make_patch(occ, bg, axis)
    if patch_pair is None:
        return None
    patch, expected = patch_pair

    boxes = _candidate_hidden_boxes((size, size), occ, axis)
    random.shuffle(boxes)
    hidden = None
    for box in boxes:
        other = _counterpart_bbox(box, (size, size), axis)
        if _boxes_overlap(box, other):
            continue
        hidden = box
        break
    if hidden is None:
        return None
    visible = _counterpart_bbox(hidden, (size, size), axis)

    grid = np.full((size, size), bg, dtype=int)
    _place_patch_pair(grid, hidden, patch, axis)

    reserved = _reserve_mask(grid.shape, [hidden, visible], pad=1)
    deco_colors = [c for c in GEN_BG_AND_DECO if c != bg]
    placed = 0
    target = random.randint(5, 11)
    for _ in range(DECOR_ATTEMPTS):
        if placed >= target:
            break
        color = random.choice(deco_colors)
        if _try_place_decor(grid, reserved, axis, color):
            placed += 1
    if placed < 3:
        return None

    inp = grid.copy()
    r0, c0, r1, c1 = hidden
    inp[r0 : r1 + 1, c0 : c1 + 1] = PINK

    chosen_score = _symmetry_score(inp, hidden, axis)
    other_axis = "horizontal" if axis == "vertical" else "vertical"
    other_score = _symmetry_score(inp, hidden, other_axis)
    if chosen_score[0] != 0 or chosen_score[1] == 0:
        return None
    if other_score[0] == 0:
        return None

    visible_patch = crop_bbox(grid, visible)
    if np.array_equal(_mark_defects(visible_patch), expected):
        return None
    if np.array_equal(_mark_defects(patch), expected) is False:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
