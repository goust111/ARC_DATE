from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
TEAL = 8

ROTATION_TO_COLOR = {
    0: RED,
    1: YELLOW,
    2: ORANGE,
    3: BLUE,
}

NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

GRID_MIN = 13
GRID_MAX = 18
COPY_COUNT_MIN = 3
COPY_COUNT_MAX = 5
NOISE_COUNT_MIN = 2
NOISE_COUNT_MAX = 8
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    deltas = NEIGHBORS8 if connectivity == 8 else ((-1, 0), (1, 0), (0, -1), (0, 1))
    out: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in deltas:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    if arr.size == 0:
        raise ValueError("shape must be non-empty")
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    mask = _shape_mask(shape)
    return tuple(map(int, mask.shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...], quarter_turns_cw: int) -> tuple[tuple[int, int], ...]:
    turns = int(quarter_turns_cw) % 4
    mask = _shape_mask(shape)
    rotated = np.rot90(mask, -turns)
    return _normalize_shape(np.argwhere(rotated))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(_rotate_shape(shape, turns) for turns in range(4))


def _place_shape(
    shape: tuple[tuple[int, int], ...], top: int, left: int
) -> frozenset[tuple[int, int]]:
    return frozenset((top + dr, left + dc) for dr, dc in shape)


def _extract_reference_shape(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    comps = _components(grid == TEAL, connectivity=8)
    assert len(comps) == 1, "expected exactly one teal reference component"
    coords = comps[0]
    assert 4 <= len(coords) <= 6, "reference template must have 4-6 cells"
    r0, c0, _r1, _c1 = _bbox(coords)
    shape = _normalize_shape(coords - np.array([r0, c0], dtype=int))
    assert len(set(_shape_variants(shape))) == 4, "reference template must have four distinct rotations"
    return shape


def _scan_matches(
    grid: np.ndarray, reference_shape: tuple[tuple[int, int], ...]
) -> list[tuple[int, int, int, frozenset[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    variants = _shape_variants(reference_shape)
    selected_cells: set[tuple[int, int]] = set()
    matches: list[tuple[int, int, int, frozenset[tuple[int, int]]]] = []

    for top in range(h):
        for left in range(w):
            for turns, shape in enumerate(variants):
                sh, sw = _shape_dims(shape)
                if top + sh > h or left + sw > w:
                    continue
                cells = _place_shape(shape, top, left)
                if selected_cells.isdisjoint(cells) and all(g[r, c] == GREEN for r, c in cells):
                    matches.append((top, left, turns, cells))
                    selected_cells.update(cells)
                    break
    return matches


def _render_output(
    grid: np.ndarray, matches: list[tuple[int, int, int, frozenset[tuple[int, int]]]]
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for _top, _left, turns, cells in matches:
        color = ROTATION_TO_COLOR[turns]
        for r, c in cells:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find every green rotated copy of the teal template and recolor each copy by its rotation angle.

    Input structure:
    - A black background contains one small teal reference shape with 4-6 cells.
    - Elsewhere, green pixels include exact rotated copies of that reference plus extra green distractor pixels.
    - The reference shape is 8-connected and rotationally asymmetric, so its four quarter-turns are all different.

    Rule:
    - Extract the teal reference shape as a bbox-normalized mask and generate its 0/90/180/270 degree rotations.
    - Scan the grid in row-major order; at each anchor, try the four rotations in that order and select any all-green placement that does not overlap an already selected one.
    - Recolor each selected copy by its rotation: 0 degrees to red, 90 degrees to yellow, 180 degrees to orange, and 270 degrees to blue.
    - Leave the teal reference, unmatched green distractors, and black background unchanged.

    Key insight:
    - Solving requires both template matching and orientation classification, because location alone is insufficient to determine the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    reference_shape = _extract_reference_shape(g)
    matches = _scan_matches(g, reference_shape)
    return _render_output(g, matches)


def _enumerate_template_catalog() -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    catalog: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for size in range(4, 7):
        for choice in combinations(cells, size):
            shape = _normalize_shape(choice)
            mask = _shape_mask(shape)
            if len(_components(mask, connectivity=8)) != 1:
                continue
            variants = _shape_variants(shape)
            if len(set(variants)) != 4:
                continue
            key = min(variants)
            if key in seen:
                continue
            seen.add(key)
            catalog.append(shape)
    return tuple(catalog)


TEMPLATE_CATALOG = _enumerate_template_catalog()


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_anchor(
    shape: tuple[tuple[int, int], ...],
    height: int,
    width: int,
    placed_bboxes: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int] | None:
    sh, sw = _shape_dims(shape)
    anchors = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
    random.shuffle(anchors)
    for top, left in anchors:
        bbox = (top, left, top + sh - 1, left + sw - 1)
        if all(_separated(bbox, other, gap) for other in placed_bboxes):
            return top, left
    return None


def _all_candidate_records(
    height: int,
    width: int,
    variants: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[
    list[tuple[int, int, int, frozenset[tuple[int, int]]]],
    dict[tuple[int, int], list[tuple[int, int, int, frozenset[tuple[int, int]]]]],
]:
    records: list[tuple[int, int, int, frozenset[tuple[int, int]]]] = []
    by_cell: dict[tuple[int, int], list[tuple[int, int, int, frozenset[tuple[int, int]]]]] = {}
    for turns, shape in enumerate(variants):
        sh, sw = _shape_dims(shape)
        for top in range(height - sh + 1):
            for left in range(width - sw + 1):
                cells = _place_shape(shape, top, left)
                record = (top, left, turns, cells)
                records.append(record)
                for cell in cells:
                    by_cell.setdefault(cell, []).append(record)
    return records, by_cell


def _candidate_pool(
    height: int,
    width: int,
    copy_cells: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    near: list[tuple[int, int]] = []
    far: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in blocked_cells:
                continue
            adjacent = any((r + dr, c + dc) in copy_cells for dr, dc in NEIGHBORS8)
            if adjacent:
                near.append(cell)
            else:
                far.append(cell)
    random.shuffle(near)
    random.shuffle(far)
    return near + far


def _all_green_candidates(
    green_cells: set[tuple[int, int]],
    records: list[tuple[int, int, int, frozenset[tuple[int, int]]]],
) -> list[tuple[int, int, int, frozenset[tuple[int, int]]]]:
    return [record for record in records if record[3].issubset(green_cells)]


def _safe_to_add_noise(
    cell: tuple[int, int],
    green_cells: set[tuple[int, int]],
    allowed_cellsets: set[frozenset[tuple[int, int]]],
    by_cell: dict[tuple[int, int], list[tuple[int, int, int, frozenset[tuple[int, int]]]]],
) -> bool:
    trial = green_cells | {cell}
    for record in by_cell.get(cell, []):
        if record[3] not in allowed_cellsets and record[3].issubset(trial):
            return False
    return True


def _manual_output(
    grid: np.ndarray, placements: list[tuple[int, int, int, frozenset[tuple[int, int]]]]
) -> np.ndarray:
    out = grid.copy()
    for _top, _left, turns, cells in placements:
        for r, c in cells:
            out[r, c] = ROTATION_TO_COLOR[turns]
    return out


def _validate_instance(
    grid: np.ndarray,
    planned: list[tuple[int, int, int, frozenset[tuple[int, int]]]],
    noise_cells: set[tuple[int, int]],
) -> bool:
    try:
        reference = _extract_reference_shape(grid)
    except AssertionError:
        return False

    if len(planned) < COPY_COUNT_MIN:
        return False
    if len({turns for _top, _left, turns, _cells in planned}) < 2:
        return False
    if not noise_cells:
        return False

    variants = _shape_variants(reference)
    records, _by_cell = _all_candidate_records(*grid.shape, variants)
    green_cells = set(map(tuple, np.argwhere(grid == GREEN)))
    candidate_records = _all_green_candidates(green_cells, records)
    planned_cellsets = {cells for _top, _left, _turns, cells in planned}
    candidate_cellsets = {cells for _top, _left, _turns, cells in candidate_records}
    if candidate_cellsets != planned_cellsets:
        return False

    actual = _scan_matches(grid, reference)
    actual_by_cells = {cells: turns for _top, _left, turns, cells in actual}
    planned_by_cells = {cells: turns for _top, _left, turns, cells in planned}
    if actual_by_cells != planned_by_cells:
        return False

    out = main(grid)
    expected = _manual_output(grid, planned)
    if not np.array_equal(out, expected):
        return False

    for cells, turns in planned_by_cells.items():
        expected_color = ROTATION_TO_COLOR[turns]
        if any(out[r, c] != expected_color for r, c in cells):
            return False

    if any(out[r, c] != GREEN for r, c in noise_cells):
        return False
    if np.count_nonzero(out == GREEN) < len(noise_cells):
        return False
    if len({out[r, c] for _top, _left, _turns, cells in planned for r, c in cells}) < 2:
        return False
    return True


def _choose_orientations(copy_count: int) -> list[int]:
    if copy_count >= 4 and random.random() < 0.7:
        base = [0, 1, 2, 3]
        extras = [random.randrange(4) for _ in range(copy_count - 4)]
        orientations = base + extras
    else:
        distinct_count = random.randint(2, min(4, copy_count))
        chosen = random.sample(range(4), distinct_count)
        orientations = chosen[:]
        while len(orientations) < copy_count:
            orientations.append(random.choice(chosen))
    random.shuffle(orientations)
    return orientations


def _generate_one() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    reference_shape = random.choice(TEMPLATE_CATALOG)
    variants = _shape_variants(reference_shape)

    copy_count = random.randint(COPY_COUNT_MIN, COPY_COUNT_MAX)
    orientations = _choose_orientations(copy_count)

    placed_bboxes: list[tuple[int, int, int, int]] = []
    reference_anchor = _choose_anchor(reference_shape, height, width, placed_bboxes, PLACEMENT_GAP)
    if reference_anchor is None:
        return None
    ref_top, ref_left = reference_anchor
    ref_h, ref_w = _shape_dims(reference_shape)
    placed_bboxes.append((ref_top, ref_left, ref_top + ref_h - 1, ref_left + ref_w - 1))

    planned: list[tuple[int, int, int, frozenset[tuple[int, int]]]] = []
    for turns in orientations:
        shape = variants[turns]
        anchor = _choose_anchor(shape, height, width, placed_bboxes, PLACEMENT_GAP)
        if anchor is None:
            return None
        top, left = anchor
        sh, sw = _shape_dims(shape)
        placed_bboxes.append((top, left, top + sh - 1, left + sw - 1))
        planned.append((top, left, turns, _place_shape(shape, top, left)))

    grid = np.zeros((height, width), dtype=int)
    for r, c in _place_shape(reference_shape, ref_top, ref_left):
        grid[r, c] = TEAL

    copy_cells = {cell for _top, _left, _turns, cells in planned for cell in cells}
    for r, c in copy_cells:
        grid[r, c] = GREEN

    records, by_cell = _all_candidate_records(height, width, variants)
    allowed_cellsets = {cells for _top, _left, _turns, cells in planned}
    green_cells = set(copy_cells)
    blocked_cells = set(copy_cells) | set(_place_shape(reference_shape, ref_top, ref_left))
    noise_target = random.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    noise_cells: set[tuple[int, int]] = set()

    for cell in _candidate_pool(height, width, copy_cells, blocked_cells):
        if len(noise_cells) >= noise_target:
            break
        if _safe_to_add_noise(cell, green_cells, allowed_cellsets, by_cell):
            green_cells.add(cell)
            noise_cells.add(cell)
            grid[cell] = GREEN

    if len(noise_cells) < NOISE_COUNT_MIN:
        return None
    if not any(any((r + dr, c + dc) in copy_cells for dr, dc in NEIGHBORS8) for r, c in noise_cells):
        return None

    if not _validate_instance(grid, planned, noise_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
