from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

Coord = tuple[int, int]
Shape = tuple[tuple[int, int], ...]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _normalize_coords(coords: np.ndarray | list[Coord] | set[Coord]) -> Shape:
    if isinstance(coords, np.ndarray):
        arr = coords.astype(int, copy=True)
    else:
        arr = np.array(sorted(coords), dtype=int)
    if arr.size == 0:
        return tuple()
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_to_mask(shape: Shape) -> np.ndarray:
    if not shape:
        return np.zeros((0, 0), dtype=bool)
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _unique_rotations(shape: Shape) -> list[Shape]:
    mask = _shape_to_mask(shape)
    seen: set[Shape] = set()
    out: list[Shape] = []
    for k in range(4):
        rotated = _normalize_coords(np.argwhere(np.rot90(mask, k)))
        if rotated not in seen:
            seen.add(rotated)
            out.append(rotated)
    return out


def _canonical_rotation(shape: Shape) -> Shape:
    return min(_unique_rotations(shape))


def _orbit_positions(r: int, c: int, box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    rr = r0 + r1 - r
    cc = c0 + c1 - c
    return {
        (int(r), int(c)),
        (int(rr), int(c)),
        (int(r), int(cc)),
        (int(rr), int(cc)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the symmetric blue pattern, treat the repaired green clusters as target shapes, and recolor matching lower candidates red.

    Input structure:
    - The upper part contains one black-backed blue pattern with some blue cells replaced by green placeholders.
    - The blue/green pattern is symmetric across both its horizontal and vertical bbox center lines.
    - The lower part contains several separated grey 4-connected candidate shapes on black background.

    Rule:
    - Find the bbox of the blue-and-green upper pattern and repair every green cell by copying the blue value from its reflection orbit partners.
    - The repaired cells are exactly the original green positions; cluster those positions by 4-connectivity to obtain target shapes.
    - Scan grey candidate components from left to right, and recolor a candidate red if any quarter-turn rotation exactly matches one unused target shape.
    - Keep unmatched grey candidates grey, and keep the repaired upper pattern blue.

    Key insight:
    - The lower matching templates are not fixed in advance; they are the connected shapes formed by the repaired placeholder pixels in the symmetric top pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top_box = _bbox((g == BLUE) | (g == GREEN))
    assert top_box is not None, "expected a blue/green symmetric pattern"

    out = g.copy()
    repaired_mask = np.zeros(g.shape, dtype=bool)
    green_coords = np.argwhere(g == GREEN)
    assert green_coords.size > 0, "expected at least one green placeholder"

    for r, c in green_coords:
        orbit = _orbit_positions(int(r), int(c), top_box)
        partner_values = [int(g[rr, cc]) for rr, cc in orbit if (rr, cc) != (int(r), int(c))]
        assert any(value == BLUE for value in partner_values), "each orbit must expose a blue repair value"
        out[int(r), int(c)] = BLUE
        repaired_mask[int(r), int(c)] = True

    target_shapes = [_normalize_coords(coords) for coords in _components(repaired_mask)]
    remaining_targets = list(target_shapes)

    gray_objects: list[tuple[BBox, np.ndarray, Shape]] = []
    for coords in _components(g == GREY):
        gray_objects.append((_bbox_coords(coords), coords, _normalize_coords(coords)))
    gray_objects.sort(key=lambda item: (item[0][1], item[0][0]))

    for _bbox0, coords, shape in gray_objects:
        rotations = set(_unique_rotations(shape))
        match_idx = next((idx for idx, target in enumerate(remaining_targets) if target in rotations), None)
        if match_idx is None:
            continue
        out[coords[:, 0], coords[:, 1]] = RED
        remaining_targets.pop(match_idx)

    return out


GEN_ATTEMPTS = 600
TOP_H_CHOICES = (8, 10, 12)
TOP_W_CHOICES = (10, 12, 14, 16)
BOTTOM_H_CHOICES = (6, 7, 8)
TOP_MARGIN_CHOICES = (0, 1)
SIDE_MARGIN_MAX = 3
TOP_BOTTOM_GAP = 2
OBJECT_COUNT_RANGE = (3, 5)
QUADRANTS = ("UL", "UR", "LL", "LR")


def _randint_inclusive(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    mask = _shape_to_mask(shape)
    return int(mask.shape[0]), int(mask.shape[1])


def _random_connected_shape(area: int, *, max_span: int, max_attempts: int = 200) -> Shape | None:
    for _ in range(max_attempts):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rows = [rr for rr, _ in trial]
                    cols = [cc for _, cc in trial]
                    if max(rows) - min(rows) + 1 <= max_span and max(cols) - min(cols) + 1 <= max_span:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(sorted(frontier)[int(np.random.randint(len(frontier)))])
        if len(cells) != area:
            continue
        return _normalize_coords(cells)
    return None


def _sample_target_shapes() -> list[Shape] | None:
    cluster_count = int(np.random.randint(1, 4))
    if cluster_count == 1:
        areas = [_randint_inclusive(5, 6)]
    elif cluster_count == 2:
        areas = [_randint_inclusive(3, 6), _randint_inclusive(3, 6)]
    else:
        areas = [_randint_inclusive(3, 4) for _ in range(3)]

    shapes: list[Shape] = []
    seen: set[Shape] = set()
    for area in areas:
        shape = _random_connected_shape(area, max_span=4)
        if shape is None:
            return None
        canonical = _canonical_rotation(shape)
        tries = 0
        while canonical in seen and tries < 100:
            shape = _random_connected_shape(area, max_span=4)
            if shape is None:
                return None
            canonical = _canonical_rotation(shape)
            tries += 1
        if canonical in seen:
            return None
        seen.add(canonical)
        shapes.append(shape)

    total_cells = sum(len(shape) for shape in shapes)
    if not 5 <= total_cells <= 12:
        return None
    return shapes


def _build_top_mask(height: int, width: int) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    half_h = height // 2
    half_w = width // 2
    hole_count = int(np.random.randint(0, 4))

    for _ in range(hole_count):
        max_h = min(2, half_h - 1)
        max_w = min(2, half_w - 1)
        if max_h < 1 or max_w < 1:
            continue
        hole_h = _randint_inclusive(1, max_h)
        hole_w = _randint_inclusive(1, max_w)
        r0 = _randint_inclusive(1, half_h - hole_h)
        c0 = _randint_inclusive(1, half_w - hole_w)
        local_holes = []
        for dr in range(hole_h):
            for dc in range(hole_w):
                rr = r0 + dr
                cc = c0 + dc
                local_holes.extend(
                    [
                        (rr, cc),
                        (height - 1 - rr, cc),
                        (rr, width - 1 - cc),
                        (height - 1 - rr, width - 1 - cc),
                    ]
                )
        for rr, cc in local_holes:
            mask[rr, cc] = False

    components = _components(mask)
    if len(components) != 1:
        return None
    if float(mask.mean()) < 0.65:
        return None
    return mask


def _map_quadrant(local_r: int, local_c: int, half_h: int, half_w: int, quadrant: str) -> Coord:
    if quadrant == "UL":
        return local_r, local_c
    if quadrant == "UR":
        return local_r, half_w + local_c
    if quadrant == "LL":
        return half_h + local_r, local_c
    return half_h + local_r, half_w + local_c


def _place_green_clusters(mask: np.ndarray, shapes: list[Shape], quadrant: str) -> tuple[list[list[Coord]], list[Shape]] | None:
    height, width = mask.shape
    half_h = height // 2
    half_w = width // 2
    occupied: set[Coord] = set()
    clusters: list[list[Coord]] = []
    placed_shapes: list[Shape] = []

    for base_shape in sorted(shapes, key=len, reverse=True):
        variants = _unique_rotations(base_shape)
        np.random.shuffle(variants)
        placed = False
        for shape in variants:
            sh, sw = _shape_dims(shape)
            anchors = [(r0, c0) for r0 in range(0, half_h - sh + 1) for c0 in range(0, half_w - sw + 1)]
            np.random.shuffle(anchors)
            for r0, c0 in anchors:
                local_cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                global_cells = [_map_quadrant(r, c, half_h, half_w, quadrant) for r, c in local_cells]
                if any(not mask[r, c] for r, c in global_cells):
                    continue
                if any(not all(mask[rr, cc] for rr, cc in _orbit_positions(r, c, (0, 0, height - 1, width - 1))) for r, c in global_cells):
                    continue
                bad = False
                for r, c in global_cells:
                    if (r, c) in occupied:
                        bad = True
                        break
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in occupied:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                clusters.append(global_cells)
                placed_shapes.append(_normalize_coords(np.array(local_cells, dtype=int)))
                occupied.update(global_cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return clusters, placed_shapes


def _shape_patch(shape: Shape, color: int) -> np.ndarray:
    patch = np.zeros(_shape_dims(shape), dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _bbox_gap_ok(box_a: BBox, box_b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap >= 2 or col_gap >= 2


def _place_bottom_objects(grid: np.ndarray, shapes: list[Shape], start_row: int) -> bool:
    height, width = grid.shape
    boxes: list[BBox] = []

    order = list(np.random.permutation(len(shapes)))
    for idx in order:
        shape = shapes[idx]
        sh, sw = _shape_dims(shape)
        options: list[tuple[int, int, BBox]] = []
        for r0 in range(start_row, height - sh + 1):
            for c0 in range(0, width - sw + 1):
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if any(not _bbox_gap_ok(box, other) for other in boxes):
                    continue
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(grid[r, c] != BLACK for r, c in cells):
                    continue
                options.append((r0, c0, box))
        if not options:
            return False
        r0, c0, box = options[int(np.random.randint(len(options)))]
        patch = _shape_patch(shape, GREY)
        grid[r0 : r0 + sh, c0 : c0 + sw][patch == GREY] = GREY
        boxes.append(box)
    return True


def _random_nonmatching_shape(target_shapes: list[Shape]) -> Shape | None:
    target_canonicals = {_canonical_rotation(shape) for shape in target_shapes}
    for _ in range(200):
        area = _randint_inclusive(3, 7)
        shape = _random_connected_shape(area, max_span=4)
        if shape is None:
            continue
        if _canonical_rotation(shape) not in target_canonicals:
            return shape
    return None


def _build_candidate_shapes(target_shapes: list[Shape]) -> list[Shape] | None:
    unique_match_count = _randint_inclusive(1, len(target_shapes))
    selected = list(np.random.permutation(len(target_shapes)))[:unique_match_count]

    candidates: list[Shape] = []
    for idx in selected:
        variants = _unique_rotations(target_shapes[idx])
        candidates.append(variants[int(np.random.randint(len(variants)))])

    if np.random.random() < 0.35:
        dup_idx = selected[int(np.random.randint(len(selected)))]
        dup_variants = _unique_rotations(target_shapes[dup_idx])
        candidates.append(dup_variants[int(np.random.randint(len(dup_variants)))])

    intrinsic_unmatched = _random_nonmatching_shape(target_shapes)
    if intrinsic_unmatched is None:
        return None
    candidates.append(intrinsic_unmatched)

    object_count = _randint_inclusive(max(OBJECT_COUNT_RANGE[0], len(candidates)), OBJECT_COUNT_RANGE[1])
    while len(candidates) < object_count:
        extra = _random_nonmatching_shape(target_shapes)
        if extra is None:
            return None
        variants = _unique_rotations(extra)
        candidates.append(variants[int(np.random.randint(len(variants)))])

    return candidates


def _render_attempt() -> np.ndarray | None:
    target_shapes = _sample_target_shapes()
    if target_shapes is None:
        return None

    top_h = TOP_H_CHOICES[int(np.random.randint(len(TOP_H_CHOICES)))]
    top_w = TOP_W_CHOICES[int(np.random.randint(len(TOP_W_CHOICES)))]
    bottom_h = BOTTOM_H_CHOICES[int(np.random.randint(len(BOTTOM_H_CHOICES)))]
    top_margin = TOP_MARGIN_CHOICES[int(np.random.randint(len(TOP_MARGIN_CHOICES)))]
    side_left = _randint_inclusive(1, SIDE_MARGIN_MAX)
    side_right = _randint_inclusive(1, SIDE_MARGIN_MAX)

    height = top_margin + top_h + TOP_BOTTOM_GAP + bottom_h
    width = side_left + top_w + side_right
    if max(height, width) > 30:
        return None

    top_mask = _build_top_mask(top_h, top_w)
    if top_mask is None:
        return None

    quadrant = QUADRANTS[int(np.random.randint(len(QUADRANTS)))]
    placement = _place_green_clusters(top_mask, target_shapes, quadrant)
    if placement is None:
        return None
    green_clusters, placed_shapes = placement

    grid = np.zeros((height, width), dtype=int)
    r_base = top_margin
    c_base = side_left
    grid[r_base : r_base + top_h, c_base : c_base + top_w][top_mask] = BLUE
    for cluster in green_clusters:
        for r, c in cluster:
            grid[r_base + r, c_base + c] = GREEN

    candidates = _build_candidate_shapes(placed_shapes)
    if candidates is None:
        return None

    start_row = top_margin + top_h + TOP_BOTTOM_GAP
    if not _place_bottom_objects(grid, candidates, start_row):
        return None

    green_components = _components(grid == GREEN)
    if len(green_components) != len(placed_shapes):
        return None
    if any(not 3 <= len(comp) <= 6 for comp in green_components):
        return None
    if not 5 <= int(np.count_nonzero(grid == GREEN)) <= 12:
        return None

    gray_components = _components(grid == GREY)
    if not OBJECT_COUNT_RANGE[0] <= len(gray_components) <= OBJECT_COUNT_RANGE[1]:
        return None
    if any(not 3 <= len(comp) <= 7 for comp in gray_components):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.any(out == GREEN):
        return None
    if not np.any(out == RED):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _render_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 035/1")
