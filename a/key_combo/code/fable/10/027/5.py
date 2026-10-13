from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RETRIES = 500

BASE_L4 = [(0, 0), (1, 0), (2, 0), (2, 1)]
BASE_LINE4 = [(0, 0), (0, 1), (0, 2), (0, 3)]
BASE_DOMINO = [(0, 0), (0, 1)]


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _corners_of_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _is_three_corner_marker(points: list[tuple[int, int]]) -> bool:
    if len(points) != 3 or len(set(points)) != 3:
        return False
    r0, c0, r1, c1 = _bbox_from_points(points)
    if r0 == r1 or c0 == c1:
        return False
    return set(points).issubset(set(_corners_of_bbox((r0, c0, r1, c1))))


def _marker_windows(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    counts = Counter(map(int, grid.ravel()))
    windows: list[tuple[int, tuple[int, int, int, int]]] = []
    for color, count in counts.items():
        if color == BG or count != 3:
            continue
        points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if _is_three_corner_marker(points):
            windows.append((int(color), _bbox_from_points(points)))
    return windows


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _find_nested_windows(
    grid: np.ndarray,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    int,
    int,
]:
    windows = _marker_windows(grid)
    assert len(windows) == 2, "expected exactly two 3-corner marker colors"

    (color_a, box_a), (color_b, box_b) = windows
    if _strictly_contains(box_a, box_b):
        return box_a, box_b, color_a, color_b
    if _strictly_contains(box_b, box_a):
        return box_b, box_a, color_b, color_a
    raise AssertionError("expected one marker rectangle to strictly contain the other")


def _dest_inner_bbox(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (
        or0 + or1 - ir1,
        oc0 + oc1 - ic1,
        or0 + or1 - ir0,
        oc0 + oc1 - ic0,
    )


def _apply_layered_transform(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    row_shift = or0 + or1 - ir0 - ir1
    col_shift = oc0 + oc1 - ic0 - ic1

    out = g.copy()
    out[or0 : or1 + 1, oc0 : oc1 + 1] = BG
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                rr = r + row_shift
                cc = c + col_shift
            else:
                rr = or0 + or1 - r
                cc = oc0 + oc1 - c
            out[rr, cc] = g[r, c]
    return out


def _apply_plain_rotation(
    grid: np.ndarray, outer: tuple[int, int, int, int]
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    or0, oc0, or1, oc1 = outer
    out = g.copy()
    out[or0 : or1 + 1, oc0 : oc1 + 1] = BG
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            rr = or0 + or1 - r
            cc = oc0 + oc1 - c
            out[rr, cc] = g[r, c]
    return out


def _apply_inner_translation_only(
    grid: np.ndarray,
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    ir0, ic0, ir1, ic1 = inner
    dr0, dc0, dr1, dc1 = _dest_inner_bbox(outer, inner)

    out = g.copy()
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
    out[dr0 : dr1 + 1, dc0 : dc1 + 1] = BG
    row_shift = dr0 - ir0
    col_shift = dc0 - ic0
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            out[r + row_shift, c + col_shift] = g[r, c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the outer invisible window by 180 degrees while transporting the nested inner window without rotating its contents.

    Input structure:
    - Black background with scattered colored pixels.
    - Exactly two colors appear three times each, and each triple sits on three corners of an unseen axis-aligned rectangle.
    - One inferred rectangle lies strictly inside the other.

    Rule:
    - Recover both rectangles from their three visible corners.
    - Rotate every cell in the outer rectangle by 180 degrees about the outer rectangle center.
    - Treat the inner rectangle as a rigid stable cabin: move that whole rectangle to its 180-degree destination, but keep all pixels inside it unrotated, so they are translated together instead of flipped.
    - Leave everything outside the outer rectangle unchanged; marker pixels follow the same layered motion.

    Key insight:
    - The two marker triples do not define two independent rotations: the inner rectangle is embedded inside the outer transform and is only carried to the opposite location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    outer, inner, _outer_color, _inner_color = _find_nested_windows(g)
    return _apply_layered_transform(g, outer, inner)


def _normalize_cells(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rot90_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in cells)
    rotated = [(c, max_c - r) for r, c in cells]
    return _normalize_cells(rotated)


def _flip_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in cells)
    flipped = [(r, max_c - c) for r, c in cells]
    return _normalize_cells(flipped)


def _variants(base: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    cur = _normalize_cells(base)
    for _ in range(4):
        for candidate in (cur, _flip_cells(cur)):
            if candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
        cur = _rot90_cells(cur)
    return variants


L4_TEMPLATES = _variants(BASE_L4)
LINE4_TEMPLATES = _variants(BASE_LINE4)
DOMINO_TEMPLATES = _variants(BASE_DOMINO)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_template(
    grid: np.ndarray,
    occupied: np.ndarray,
    allowed: np.ndarray,
    templates: list[tuple[tuple[int, int], ...]],
    color: int,
) -> bool:
    h, w = grid.shape
    choices: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    for template in templates:
        max_r = max(r for r, _ in template)
        max_c = max(c for _, c in template)
        for r0 in range(h - max_r):
            for c0 in range(w - max_c):
                pts = [(r0 + dr, c0 + dc) for dr, dc in template]
                if all(allowed[r, c] and not occupied[r, c] for r, c in pts):
                    choices.append((template, r0, c0))
    if not choices:
        return False

    template, r0, c0 = random.choice(choices)
    for dr, dc in template:
        rr, cc = r0 + dr, c0 + dc
        grid[rr, cc] = color
        occupied[rr, cc] = True
    return True


def _place_marker_triplet(
    grid: np.ndarray,
    occupied: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    corners = _corners_of_bbox(bbox)
    missing = random.randrange(4)
    for idx, (r, c) in enumerate(corners):
        if idx == missing:
            continue
        grid[r, c] = color
        occupied[r, c] = True


def _choose_nested_rectangles(
    h: int, w: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    outer_h = random.randint(10, min(15, h - 4))
    outer_w = random.randint(10, min(15, w - 4))
    or0 = random.randint(2, h - outer_h - 2)
    oc0 = random.randint(2, w - outer_w - 2)
    outer = (or0, oc0, or0 + outer_h - 1, oc0 + outer_w - 1)

    inner_h = random.randint(5, min(8, outer_h - 2))
    inner_w = random.randint(5, min(8, outer_w - 2))
    ir0 = random.randint(or0 + 1, outer[2] - inner_h)
    ic0 = random.randint(oc0 + 1, outer[3] - inner_w)
    inner = (ir0, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1)
    return outer, inner


def _sample_free_cells(
    allowed: np.ndarray, occupied: np.ndarray, count: int
) -> list[tuple[int, int]] | None:
    blocked = _dilate8(occupied)
    cells = [
        (r, c)
        for r in range(allowed.shape[0])
        for c in range(allowed.shape[1])
        if allowed[r, c] and not blocked[r, c]
    ]
    if len(cells) < count:
        return None
    first = random.choice(cells)
    remaining = [
        (r, c)
        for r, c in cells
        if max(abs(r - first[0]), abs(c - first[1])) > 1
    ]
    if count == 1:
        return [first]
    if not remaining:
        return None
    second = random.choice(remaining)
    return [first, second]


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        outer, inner, outer_marker, inner_marker = _find_nested_windows(grid)
    except AssertionError:
        return False

    out = _apply_layered_transform(grid, outer, inner)
    if not np.array_equal(main(grid), out):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _apply_plain_rotation(grid, outer)):
        return False
    if np.array_equal(out, _apply_inner_translation_only(grid, outer, inner)):
        return False

    marker_mask = (grid == outer_marker) | (grid == inner_marker)
    outer_mask = _rect_mask(grid.shape, outer)
    inner_mask = _rect_mask(grid.shape, inner)
    outer_only = outer_mask & ~inner_mask

    if np.count_nonzero((grid != BG) & (~outer_mask)) < 2:
        return False
    if np.count_nonzero((grid != BG) & inner_mask & (~marker_mask)) < 5:
        return False
    if np.count_nonzero((grid != BG) & outer_only & (~marker_mask)) < 4:
        return False

    candidates = _marker_windows(grid)
    if len(candidates) != 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    outer, inner = _choose_nested_rectangles(h, w)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = random.sample(range(1, 10), 7)
    outer_marker, inner_marker, outer_a, outer_b, inner_a, inner_b, outside_a = colors

    _place_marker_triplet(grid, occupied, outer, outer_marker)
    _place_marker_triplet(grid, occupied, inner, inner_marker)

    outer_mask = _rect_mask((h, w), outer)
    inner_mask = _rect_mask((h, w), inner)
    outer_only = outer_mask & ~inner_mask
    outside = ~outer_mask

    if not _place_template(grid, occupied, outer_only, LINE4_TEMPLATES, outer_a):
        return None
    if not _place_template(grid, occupied, outer_only, DOMINO_TEMPLATES, outer_b):
        return None
    if not _place_template(grid, occupied, inner_mask, L4_TEMPLATES, inner_a):
        return None
    if not _place_template(grid, occupied, inner_mask, DOMINO_TEMPLATES, inner_b):
        return None

    outside_cells = _sample_free_cells(outside, occupied, 2)
    if outside_cells is None:
        return None
    for r, c in outside_cells:
        grid[r, c] = outside_a
        occupied[r, c] = True

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-window sample")
