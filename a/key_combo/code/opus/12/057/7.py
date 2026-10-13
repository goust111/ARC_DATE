from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
CYAN = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _enumerate_shapes() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 6):
        cur: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            for cell in frontier:
                grown = np.array(list(cells | {cell}), dtype=int)
                cur.add(_normalize_shape(grown))
        by_size[size] = cur

    filtered: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size in (3, 4, 5):
        shapes = []
        for shape in by_size[size]:
            h, w = _shape_dims(shape)
            if h <= 3 and w <= 3:
                shapes.append(shape)
        filtered[size] = sorted(shapes)
    return filtered


SHAPES_BY_SIZE = _enumerate_shapes()
ALL_SHAPES = [shape for size in (3, 4, 5) for shape in SHAPES_BY_SIZE[size]]


def _extract_template_shape(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    comps = _components(grid == CYAN)
    assert len(comps) == 1, "expected one cyan template"
    box = _bbox_from_coords(comps[0])
    crop = _crop(grid, box)
    gap = crop == BLACK
    assert gap.any(), "template must contain a black gap"
    gap_comps = _components(gap)
    assert len(gap_comps) == 1, "template gap must be one connected component"
    assert int(gap.sum()) + int(np.count_nonzero(crop == CYAN)) == crop.size
    return _normalize_shape(gap_comps[0])


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rect_mask = np.isin(grid, [BLUE, RED, GREEN])
    rects = []
    for coords in _components(rect_mask):
        box = _bbox_from_coords(coords)
        crop = _crop(grid, box)
        assert np.all(crop != BLACK), "foreground rectangles must be solid crops"
        assert GREEN in set(map(int, np.unique(crop))), "each rectangle must contain green fill"
        red_comps = _components(crop == RED)
        assert len(red_comps) == 1, "each rectangle must contain one red shape"
        red_shape = _normalize_shape(red_comps[0])
        blue_count = int(np.count_nonzero(crop == BLUE))
        assert 0 <= blue_count <= 3
        red_mask = crop == RED
        for br, bc in np.argwhere(crop == BLUE):
            for nr, nc in _neighbors4(int(br), int(bc)):
                if 0 <= nr < crop.shape[0] and 0 <= nc < crop.shape[1]:
                    assert not red_mask[nr, nc], "blue markers must not touch red orthogonally"
        rects.append(
            {
                "bbox": box,
                "crop": crop,
                "red_shape": red_shape,
                "red_area": len(red_shape),
                "blue_count": blue_count,
            }
        )
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _expected_from_rect(rect: dict) -> np.ndarray:
    out = rect["crop"].copy()
    if rect["blue_count"] % 2 == 1:
        out = np.fliplr(out)
    out[out == RED] = CYAN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the uniquely shape-matching rectangle, then mirror it iff its marker count is odd.

    Input structure:
    - Black background surrounds one cyan template rectangle in the upper-left area and several separate green solid rectangles elsewhere.
    - Inside the cyan template, a connected black hole encodes a small gap shape.
    - Each green rectangle contains one connected red shape and 0-3 isolated blue marker pixels.

    Rule:
    - Extract the black gap shape inside the cyan template and normalize it by translation.
    - Extract and normalize the red shape inside each green rectangle.
    - Select the unique rectangle whose red shape exactly matches the template gap shape.
    - Crop that entire rectangle; if its blue marker count is odd, flip the crop horizontally, otherwise keep it unchanged.
    - Recolor the red shape in the final crop to cyan.

    Key insight:
    - Shape matching decides which rectangle to crop, while marker counting decides only the crop orientation, so neither cue can replace the other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_shape = _extract_template_shape(g)
    rects = _extract_rectangles(g)
    assert 3 <= len(rects) <= 4, "expected 3-4 green rectangles"

    matches = [rect for rect in rects if rect["red_shape"] == template_shape]
    assert len(matches) == 1, "expected a unique complementary match"
    return _expected_from_rect(matches[0])


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_box(
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
    occupied: list[tuple[int, int, int, int]],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    gap: int,
) -> tuple[int, int, int, int] | None:
    positions = [
        (r0, c0)
        for r0 in range(row_range[0], row_range[1] + 1)
        for c0 in range(col_range[0], col_range[1] + 1)
        if r0 + h <= grid_h and c0 + w <= grid_w
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if all(_boxes_separated(box, other, gap=gap) for other in occupied):
            return box
    return None


def _random_shape_pack(n_rects: int):
    winner_area = random.choice([3, 4, 5])
    winner_shape = random.choice(SHAPES_BY_SIZE[winner_area])
    same_area_shape = random.choice([s for s in SHAPES_BY_SIZE[winner_area] if s != winner_shape])
    pool = [s for s in ALL_SHAPES if s not in {winner_shape, same_area_shape}]
    others = random.sample(pool, n_rects - 2)
    shapes = [winner_shape, same_area_shape, *others]
    random.shuffle(shapes)
    winner_idx = random.randrange(n_rects)
    if shapes[winner_idx] != winner_shape:
        other_idx = shapes.index(winner_shape)
        shapes[winner_idx], shapes[other_idx] = shapes[other_idx], shapes[winner_idx]
    same_area_idx = next(i for i, shape in enumerate(shapes) if i != winner_idx and len(shape) == winner_area)
    return shapes, winner_idx, same_area_idx


def _rect_dims_for_shape(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    sh, sw = _shape_dims(shape)
    h = random.randint(max(4, sh + 2), 6)
    w = random.randint(max(5, sw + 2), 7)
    return h, w


def _marker_count_plan(n_rects: int, winner_idx: int) -> list[int]:
    odd_branch = random.random() < 0.5
    winner_count = 1 if odd_branch else random.choice([0, 2])
    counts = [0] * n_rects
    counts[winner_idx] = winner_count
    distractors = [i for i in range(n_rects) if i != winner_idx]
    more_idx = random.choice(distractors)
    counts[more_idx] = random.randint(winner_count + 1, 3)
    for i in distractors:
        if i == more_idx:
            continue
        counts[i] = random.randint(0, 3)
    return counts


def _paint_template(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    shape: tuple[tuple[int, int], ...],
) -> None:
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), CYAN, dtype=int)
    sh, sw = _shape_dims(shape)
    offset_r = random.randint(1, patch.shape[0] - sh - 1)
    offset_c = random.randint(1, patch.shape[1] - sw - 1)
    for sr, sc in shape:
        patch[offset_r + sr, offset_c + sc] = BLACK
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _candidate_marker_cells(red_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = red_mask.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if red_mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and red_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                continue
            candidates.append((r, c))
    return candidates


def _paint_rectangle(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    shape: tuple[tuple[int, int], ...],
    marker_count: int,
) -> None:
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), GREEN, dtype=int)
    sh, sw = _shape_dims(shape)
    offset_r = random.randint(1, patch.shape[0] - sh - 1)
    offset_c = random.randint(1, patch.shape[1] - sw - 1)
    red_mask = np.zeros(patch.shape, dtype=bool)
    for sr, sc in shape:
        red_mask[offset_r + sr, offset_c + sc] = True
    patch[red_mask] = RED
    marker_cells = _candidate_marker_cells(red_mask)
    assert len(marker_cells) >= marker_count
    for rr, cc in random.sample(marker_cells, marker_count):
        patch[rr, cc] = BLUE
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _build_instance() -> np.ndarray | None:
    grid_h = random.randint(18, 22)
    grid_w = random.randint(18, 22)
    n_rects = random.randint(3, 4)
    shapes, winner_idx, same_area_idx = _random_shape_pack(n_rects)

    winner_sh, winner_sw = _shape_dims(shapes[winner_idx])
    same_sh, same_sw = _shape_dims(shapes[same_area_idx])
    winner_rect_h = random.randint(max(4, max(winner_sh, same_sh) + 2), 6)
    winner_rect_w = random.randint(max(5, max(winner_sw, same_sw) + 2), 7)
    rect_dims = []
    for i, shape in enumerate(shapes):
        if i == winner_idx or i == same_area_idx:
            rect_dims.append((winner_rect_h, winner_rect_w))
        else:
            rect_dims.append(_rect_dims_for_shape(shape))

    marker_counts = _marker_count_plan(n_rects, winner_idx)
    template_h = random.randint(max(4, _shape_dims(shapes[winner_idx])[0] + 2), 6)
    template_w = random.randint(max(4, _shape_dims(shapes[winner_idx])[1] + 2), 6)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []

    template_box = _place_box(
        template_h,
        template_w,
        grid_h,
        grid_w,
        occupied,
        row_range=(1, min(4, grid_h - template_h)),
        col_range=(1, min(4, grid_w - template_w)),
        gap=1,
    )
    if template_box is None:
        return None
    occupied.append(template_box)

    rect_boxes = [None] * n_rects
    order = list(range(n_rects))
    random.shuffle(order)
    for idx in order:
        h, w = rect_dims[idx]
        row_lo = 0
        row_hi = grid_h - h
        col_lo = 0
        col_hi = grid_w - w
        box = _place_box(
            h,
            w,
            grid_h,
            grid_w,
            occupied,
            row_range=(row_lo, row_hi),
            col_range=(col_lo, col_hi),
            gap=1,
        )
        if box is None:
            return None
        rect_boxes[idx] = box
        occupied.append(box)

    _paint_template(grid, template_box, shapes[winner_idx])
    for idx in range(n_rects):
        _paint_rectangle(grid, rect_boxes[idx], shapes[idx], marker_counts[idx])
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        assert g.ndim == 2 and max(g.shape) <= 30
        template_shape = _extract_template_shape(g)
        rects = _extract_rectangles(g)
        assert 3 <= len(rects) <= 4
        red_shapes = [rect["red_shape"] for rect in rects]
        assert len(set(red_shapes)) == len(red_shapes), "red shapes must be pairwise distinct"

        matches = [rect for rect in rects if rect["red_shape"] == template_shape]
        assert len(matches) == 1
        winner = matches[0]

        assert 3 <= len(template_shape) <= 5
        assert any(rect["blue_count"] > winner["blue_count"] for rect in rects if rect is not winner)
        assert any(
            rect["red_area"] == winner["red_area"] and rect["red_shape"] != winner["red_shape"]
            for rect in rects
            if rect is not winner
        )
        winner_h, winner_w = winner["crop"].shape
        assert any(
            rect["crop"].shape == (winner_h, winner_w)
            for rect in rects
            if rect is not winner
        )

        expected = _expected_from_rect(winner)
        actual = main(g)
        assert np.array_equal(actual, expected)
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_instance()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
