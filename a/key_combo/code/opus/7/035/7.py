from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

MASK_COLORS = {BLACK, ORANGE}
MARKER_COLORS = (BLUE, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _transform_shape(shape: tuple[tuple[int, int], ...], rotations: int, flip_lr: bool) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    for _ in range(rotations % 4):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    if flip_lr:
        pts[:, 1] *= -1
    pts -= pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _build_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ]
    variants = {
        _transform_shape(shape, rotations, flip_lr)
        for shape in base_shapes
        for rotations in range(4)
        for flip_lr in (False, True)
    }
    return sorted(variants, key=lambda shape: (len(shape), _shape_dims(shape), shape))


SHAPE_POOL = _build_shape_pool()


def main(grid: np.ndarray) -> np.ndarray:
    """Merge the two orange masks and fill each merged black hole with the color of the matching marker shape.

    Input structure:
    - A full-height red column splits the grid into equal left and right halves.
    - The top rows on both sides are same-sized black/orange masks.
    - Extra rows below the masks contain several separate monochrome marker shapes in blue, green, or yellow.

    Rule:
    - Split the grid at the red separator and keep the maximal top block whose two halves use only black and orange.
    - Take the pixelwise OR of the two masks; merged orange cells become blue in the output.
    - Find each 4-connected black component in the merged mask, normalize its shape, and compare it with the normalized marker shapes.
    - Fill any hole whose shape matches a unique marker shape with that marker's color; unmatched holes stay black.

    Key insight:
    - The hole shapes are not visible in either mask alone; they emerge only after the OR merge, and the separate marker shapes provide the fill colors by normalized-shape matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_cols = np.flatnonzero(np.all(g == RED, axis=0))
    assert len(separator_cols) == 1
    sep = int(separator_cols[0])

    left_full = g[:, :sep]
    right_full = g[:, sep + 1 :]
    assert left_full.shape == right_full.shape

    mask_rows = 0
    for r in range(g.shape[0]):
        if set(np.unique(left_full[r])) <= MASK_COLORS and set(np.unique(right_full[r])) <= MASK_COLORS:
            mask_rows += 1
        else:
            break
    assert mask_rows > 0

    left = left_full[:mask_rows]
    right = right_full[:mask_rows]
    merged = (left == ORANGE) | (right == ORANGE)
    out = np.full(left.shape, BLACK, dtype=int)
    out[merged] = BLUE

    shape_to_colors: dict[tuple[tuple[int, int], ...], set[int]] = {}
    if mask_rows < g.shape[0]:
        marker_area = g[mask_rows:, :]
        for color in MARKER_COLORS:
            for comp in _components(marker_area == color):
                shape_to_colors.setdefault(_normalize_shape(comp), set()).add(color)

    unique_shape_color = {
        shape: next(iter(colors))
        for shape, colors in shape_to_colors.items()
        if len(colors) == 1
    }

    for hole in _components(~merged):
        color = unique_shape_color.get(_normalize_shape(hole))
        if color is not None:
            out[hole[:, 0], hole[:, 1]] = color
    return out


def _shape_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _choose_shapes(max_marker_h: int, max_hole_h: int, max_hole_w: int) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    candidates = [
        shape
        for shape in SHAPE_POOL
        if _shape_dims(shape)[0] <= max_marker_h
        and _shape_dims(shape)[0] <= max_hole_h - 2
        and _shape_dims(shape)[1] <= max_hole_w - 2
    ]
    unique = random.sample(candidates, 3)
    return unique[0], unique[1], unique[2]


def _place_holes(h: int, w: int, shapes: list[tuple[tuple[int, int], ...]]) -> list[list[tuple[int, int]]] | None:
    for _ in range(200):
        forbidden: set[tuple[int, int]] = set()
        placements: list[list[tuple[int, int]]] = []
        ok = True
        for shape in shapes:
            sh, sw = _shape_dims(shape)
            candidates: list[list[tuple[int, int]]] = []
            for top in range(1, h - sh):
                for left in range(1, w - sw):
                    cells = _shape_cells(shape, top, left)
                    if any(cell in forbidden for cell in cells):
                        continue
                    candidates.append(cells)
            if not candidates:
                ok = False
                break
            cells = random.choice(candidates)
            placements.append(cells)
            for r, c in cells:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if 0 <= rr < h and 0 <= cc < w:
                            forbidden.add((rr, cc))
        if ok:
            return placements
    return None


def _feasible_mask_counts(n: int, union_count: int) -> list[tuple[int, int]]:
    lo = int(np.ceil(0.40 * n))
    hi = int(np.floor(0.60 * n))
    return [
        (left_count, right_count)
        for left_count in range(lo, hi + 1)
        for right_count in range(lo, hi + 1)
        if left_count + right_count >= union_count + 1
        and left_count <= union_count - 1
        and right_count <= union_count - 1
    ]


def _split_union(union_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = union_mask.shape
    n = h * w
    active = np.flatnonzero(union_mask.reshape(-1))
    union_count = int(active.size)
    feasible = _feasible_mask_counts(n, union_count)
    if not feasible:
        return None
    left_count, right_count = random.choice(feasible)
    left_only = union_count - right_count
    right_only = union_count - left_count
    both = left_count + right_count - union_count
    if min(left_only, right_only, both) < 1:
        return None

    active = active.copy()
    np.random.shuffle(active)
    left_only_idx = active[:left_only]
    right_only_idx = active[left_only : left_only + right_only]
    both_idx = active[left_only + right_only :]

    left = np.zeros(n, dtype=int)
    right = np.zeros(n, dtype=int)
    left[left_only_idx] = ORANGE
    left[both_idx] = ORANGE
    right[right_only_idx] = ORANGE
    right[both_idx] = ORANGE
    return left.reshape(h, w), right.reshape(h, w)


def _place_markers(
    band_h: int,
    total_w: int,
    sep: int,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> np.ndarray | None:
    for _ in range(300):
        band = np.zeros((band_h, total_w), dtype=int)
        forbidden: set[tuple[int, int]] = {(r, sep) for r in range(band_h)}
        placements: list[tuple[list[tuple[int, int]], int]] = []
        ok = True
        order = list(range(len(shapes)))
        random.shuffle(order)
        for idx in order:
            shape = shapes[idx]
            color = colors[idx]
            sh, sw = _shape_dims(shape)
            candidates: list[list[tuple[int, int]]] = []
            for top in range(band_h - sh + 1):
                for left in range(total_w - sw + 1):
                    cells = _shape_cells(shape, top, left)
                    cols = [c for _, c in cells]
                    if any(c == sep for c in cols):
                        continue
                    if min(cols) < sep < max(cols):
                        continue
                    if any(cell in forbidden for cell in cells):
                        continue
                    candidates.append(cells)
            if not candidates:
                ok = False
                break
            cells = random.choice(candidates)
            placements.append((cells, color))
            for r, c in cells:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if 0 <= rr < band_h and 0 <= cc < total_w:
                            forbidden.add((rr, cc))
        if not ok:
            continue
        for cells, color in placements:
            for r, c in cells:
                band[r, c] = color
        if not np.all(np.any(band != 0, axis=1)):
            continue
        if not np.any(band[:, :sep] != 0) or not np.any(band[:, sep + 1 :] != 0):
            continue
        return band
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(6, 9)
    w = random.randint(7, 9)
    extra_h = random.choice((2, 3))

    matched_shape_a, matched_shape_b, distractor_shape = _choose_shapes(extra_h, h, w)
    hole_shapes = [matched_shape_a, matched_shape_b]
    hole_cells = _place_holes(h, w, hole_shapes)
    if hole_cells is None:
        return None

    union_mask = np.ones((h, w), dtype=bool)
    for cells in hole_cells:
        for r, c in cells:
            union_mask[r, c] = False

    split = _split_union(union_mask)
    if split is None:
        return None
    left, right = split

    marker_colors = list(MARKER_COLORS)
    random.shuffle(marker_colors)
    matched_colors = marker_colors[:2]
    band = _place_markers(
        extra_h,
        2 * w + 1,
        w,
        [matched_shape_a, matched_shape_b, distractor_shape],
        marker_colors,
    )
    if band is None:
        return None

    grid = np.zeros((h + extra_h, 2 * w + 1), dtype=int)
    grid[:, w] = RED
    grid[:h, :w] = left
    grid[:h, w + 1 :] = right
    grid[h:, :] = band
    grid[:, w] = RED

    expected = np.full((h, w), BLUE, dtype=int)
    for cells, color in zip(hole_cells, matched_colors):
        for r, c in cells:
            expected[r, c] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle")
