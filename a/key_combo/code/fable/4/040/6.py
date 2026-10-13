from __future__ import annotations

from collections import deque

import numpy as np


# ARC palette roles used by the puzzle description.
BLACK = 0
BLUE = 1
GREY = 5


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        queue = deque([(start_r, start_c)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox in the arc_utils convention: (r0, c0, r1, c1)."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    """Crop coordinates to their bbox and render a normalized boolean mask."""
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _normalize(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords), "shape must contain at least one cell"
    return _mask_from_coords(coords)


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    """The distinct normalized quarter-turns of a shape (reflections excluded)."""
    variants: list[np.ndarray] = []
    for turns in range(4):
        variant = _normalize(np.rot90(mask, turns))
        if not any(np.array_equal(variant, old) for old in variants):
            variants.append(variant)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    """Rotation-invariant, but reflection-sensitive, exact shape key."""
    encodings = [
        (int(variant.shape[0]), int(variant.shape[1]), variant.astype(np.uint8).tobytes())
        for variant in _rotations(mask)
    ]
    return min(encodings)


def _extract_pockets(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    """Find edge-open black components using the blue structure's own bbox."""
    blue = np.argwhere(grid == BLUE)
    assert len(blue), "expected a blue structure"
    assert len(_components(grid == BLUE)) == 1, "expected one 4-connected blue structure"
    r0, c0, r1, c1 = _bbox(blue)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = local.shape
    pockets: list[np.ndarray] = []
    for coords in _components(local == BLACK):
        touches_edge = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == height - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == width - 1)
        )
        if touches_edge:
            pockets.append(coords + np.array([r0, c0]))
    assert pockets, "expected at least one edge-open pocket"
    return pockets, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Absorb each uniquely rotation-congruent gray block into its blue structure pocket.

    Input structure:
    - A black grid contains one upper blue 4-connected structure whose bbox edge is
      cut by several black 4-connected pockets opening in different directions.
    - Separate gray 4-connected candidate blocks lie below the structure; some are
      exact rotated copies of pockets and others are strict-fit or reflected decoys.

    Rule:
    - Compare the complete normalized cell mask of every pocket with all four
      rotations of every gray block.  For each unique exact match, erase the gray
      block to black and fill the whole matched pocket blue.  Leave every unmatched
      pocket and gray block unchanged.

    Key insight:
    - Congruence is strict equality under quarter-turn rotation: a smaller shape
      that can sit inside a pocket, or a reflected but non-rotatable copy, does not
      qualify.  Pockets are measured in the blue structure bbox reference frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, GREY})

    pockets, structure_box = _extract_pockets(g)
    _r0, _c0, structure_bottom, _c1 = structure_box
    gray_objects = _components(g == GREY)
    assert gray_objects, "expected gray candidate blocks"
    assert all(int(coords[:, 0].min()) > structure_bottom for coords in gray_objects)

    pockets_by_key: dict[tuple[int, int, bytes], list[np.ndarray]] = {}
    candidates_by_key: dict[tuple[int, int, bytes], list[np.ndarray]] = {}
    for coords in pockets:
        pockets_by_key.setdefault(_shape_key(_mask_from_coords(coords)), []).append(coords)
    for coords in gray_objects:
        candidates_by_key.setdefault(_shape_key(_mask_from_coords(coords)), []).append(coords)

    out = g.copy()
    for key, matching_pockets in pockets_by_key.items():
        matching_candidates = candidates_by_key.get(key, [])
        if not matching_candidates:
            continue
        # A valid puzzle gives a unique visible source and destination for every move.
        assert len(matching_pockets) == 1 and len(matching_candidates) == 1, (
            "rotation-congruent source/destination must be unique"
        )
        pocket = matching_pockets[0]
        candidate = matching_candidates[0]
        out[candidate[:, 0], candidate[:, 1]] = BLACK
        out[pocket[:, 0], pocket[:, 1]] = BLUE
    return out


# ------------------------------ generator ------------------------------


def _random_polyomino(rng: np.random.Generator, area: int) -> np.ndarray | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                trial = cells | {(nr, nc)}
                rows = [x for x, _ in trial]
                cols = [y for _, y in trial]
                if max(rows) - min(rows) < 4 and max(cols) - min(cols) < 4:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        choices = sorted(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])
    coords = np.asarray(sorted(cells), dtype=int)
    mask = _mask_from_coords(coords)
    if min(mask.shape) < 2 or int(mask.sum()) == int(mask.size):
        return None
    return mask


def _is_chiral(mask: np.ndarray) -> bool:
    return _shape_key(mask) != _shape_key(np.fliplr(mask))


def _make_pocket_shapes(rng: np.random.Generator, count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple[int, int, bytes]] = set()
    for index in range(count):
        found = None
        for _ in range(250):
            mask = _random_polyomino(rng, int(rng.integers(4, 8)))
            if mask is None:
                continue
            key = _shape_key(mask)
            if key in keys:
                continue
            # Pocket zero supplies both a true rotation witness and a mirror decoy.
            if index == 0 and (not _is_chiral(mask) or len(_rotations(mask)) < 4):
                continue
            found = mask
            break
        if found is None:
            return None
        shapes.append(found)
        keys.add(_shape_key(found))

    # The mirror decoy must not accidentally match any other pocket.
    if _shape_key(np.fliplr(shapes[0])) in keys:
        return None
    return shapes


def _direction_plan(rng: np.random.Generator, count: int) -> list[str]:
    directions = ["top", "bottom", "left", "right"]
    first_two = list(rng.choice(directions, size=2, replace=False))
    plan = first_two + [directions[int(rng.integers(4))] for _ in range(count - 2)]
    rng.shuffle(plan)
    return plan


def _origins_for_edge(
    mask: np.ndarray,
    direction: str,
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    height, width = mask.shape
    if direction == "top":
        return [(r0, c) for c in range(c0 + 1, c1 - width + 1)]
    if direction == "bottom":
        return [(r1 - height + 1, c) for c in range(c0 + 1, c1 - width + 1)]
    if direction == "left":
        return [(r, c0) for r in range(r0 + 1, r1 - height + 1)]
    assert direction == "right"
    return [(r, c1 - width + 1) for r in range(r0 + 1, r1 - height + 1)]


def _absolute_cells(mask: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}


def _touches_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        if any((nr, nc) in occupied for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            return True
    return False


def _carve_pockets(
    rng: np.random.Generator,
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    base_shapes: list[np.ndarray],
    directions: list[str],
) -> tuple[list[np.ndarray], list[str]] | None:
    carved: set[tuple[int, int]] = set()
    pocket_masks: list[np.ndarray] = []
    for base, direction in zip(base_shapes, directions):
        variants = _rotations(base)
        rng.shuffle(variants)
        placements: list[tuple[np.ndarray, tuple[int, int]]] = []
        for variant in variants:
            for origin in _origins_for_edge(variant, direction, box):
                cells = _absolute_cells(variant, origin)
                if not _touches_existing(cells, carved):
                    placements.append((variant, origin))
        if not placements:
            return None
        variant, origin = placements[int(rng.integers(len(placements)))]
        cells = _absolute_cells(variant, origin)
        carved |= cells
        pocket_masks.append(variant)
        for r, c in cells:
            grid[r, c] = BLACK

    if len(_components(grid == BLUE)) != 1:
        return None
    found, found_box = _extract_pockets(grid)
    if found_box != box or len(found) != len(base_shapes):
        return None
    if {_shape_key(_mask_from_coords(coords)) for coords in found} != {_shape_key(x) for x in pocket_masks}:
        return None
    return pocket_masks, directions


def _remove_one_cell(mask: np.ndarray, forbidden: set[tuple[int, int, bytes]]) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for removed in cells:
        remain = [cell for cell in cells if cell != removed]
        trial = np.zeros(mask.shape, dtype=bool)
        for r, c in remain:
            trial[r, c] = True
        if len(_components(trial)) != 1:
            continue
        trial = _normalize(trial)
        if _shape_key(trial) not in forbidden:
            return trial
    return None


def _strictly_fits(smaller: np.ndarray, pocket: np.ndarray) -> bool:
    """Independent witness that a non-congruent decoy can sit inside a pocket."""
    target = np.asarray(pocket, dtype=bool)
    if int(smaller.sum()) >= int(target.sum()):
        return False
    for variant in _rotations(smaller):
        h, w = variant.shape
        for r0 in range(target.shape[0] - h + 1):
            for c0 in range(target.shape[1] - w + 1):
                if np.all(target[r0 : r0 + h, c0 : c0 + w][variant]):
                    return True
    return False


def _candidate_locations(
    grid: np.ndarray,
    mask: np.ndarray,
    min_row: int,
) -> list[tuple[int, int]]:
    height, width = grid.shape
    h, w = mask.shape
    locations: list[tuple[int, int]] = []
    for r0 in range(min_row, height - h):
        for c0 in range(1, width - w):
            cells = _absolute_cells(mask, (r0, c0))
            clear = True
            for r, c in cells:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != BLACK:
                            clear = False
                            break
                    if not clear:
                        break
                if not clear:
                    break
            if clear:
                locations.append((r0, c0))
    return locations


def _place_candidates(
    rng: np.random.Generator,
    grid: np.ndarray,
    masks: list[np.ndarray],
    min_row: int,
) -> list[np.ndarray] | None:
    order = list(rng.permutation(len(masks)))
    placed: list[np.ndarray | None] = [None] * len(masks)
    for index in order:
        mask = masks[int(index)]
        locations = _candidate_locations(grid, mask, min_row)
        if not locations:
            return None
        origin = locations[int(rng.integers(len(locations)))]
        cells = sorted(_absolute_cells(mask, origin))
        coords = np.asarray(cells, dtype=int)
        grid[coords[:, 0], coords[:, 1]] = GREY
        placed[int(index)] = coords
    assert all(coords is not None for coords in placed)
    return [coords for coords in placed if coords is not None]


def _same_cell_sets(first: list[np.ndarray], second: list[np.ndarray]) -> bool:
    encode = lambda coords: frozenset(tuple(map(int, rc)) for rc in coords)
    return {encode(coords) for coords in first} == {encode(coords) for coords in second}


def _semantic_validation(
    grid: np.ndarray,
    expected: np.ndarray,
    intended_pockets: list[np.ndarray],
    intended_candidates: list[np.ndarray],
    matched_pocket_keys: set[tuple[int, int, bytes]],
    smaller_decoy: np.ndarray,
    smaller_source: np.ndarray,
) -> bool:
    try:
        pockets, box = _extract_pockets(grid)
        candidates = _components(grid == GREY)
        if not _same_cell_sets(pockets, intended_pockets):
            return False
        if not _same_cell_sets(candidates, intended_candidates):
            return False

        pocket_masks = [_mask_from_coords(coords) for coords in pockets]
        pocket_keys = [_shape_key(mask) for mask in pocket_masks]
        candidate_masks = [_mask_from_coords(coords) for coords in candidates]
        candidate_keys = [_shape_key(mask) for mask in candidate_masks]
        if len(pocket_keys) != len(set(pocket_keys)):
            return False
        if any(candidate_keys.count(key) > 1 for key in set(pocket_keys)):
            return False

        actual_matches = {key for key in pocket_keys if key in candidate_keys}
        if actual_matches != matched_pocket_keys:
            return False
        if not actual_matches or len(actual_matches) == len(pocket_keys):
            return False
        if not any(key not in pocket_keys for key in candidate_keys):
            return False

        # Both exact-shape and strictness counterexamples are present in every input.
        if _shape_key(smaller_decoy) in set(pocket_keys):
            return False
        if not _strictly_fits(smaller_decoy, smaller_source):
            return False
        if _shape_key(np.fliplr(smaller_source)) in set(pocket_keys):
            return False

        # At least one true match is visibly rotated relative to its destination.
        rotation_witness = False
        for pocket_mask in pocket_masks:
            key = _shape_key(pocket_mask)
            if key not in actual_matches:
                continue
            for candidate_mask in candidate_masks:
                if _shape_key(candidate_mask) == key and not np.array_equal(candidate_mask, pocket_mask):
                    rotation_witness = True
        if not rotation_witness:
            return False

        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, grid) or actual.shape != grid.shape or max(actual.shape) > 30:
            return False

        # Check semantic effects separately, rather than relying only on oracle closure.
        for coords in pockets:
            key = _shape_key(_mask_from_coords(coords))
            wanted = BLUE if key in matched_pocket_keys else BLACK
            if not np.all(actual[coords[:, 0], coords[:, 1]] == wanted):
                return False
        for coords in candidates:
            key = _shape_key(_mask_from_coords(coords))
            wanted = BLACK if key in matched_pocket_keys else GREY
            if not np.all(actual[coords[:, 0], coords[:, 1]] == wanted):
                return False
        if box[2] >= min(int(coords[:, 0].min()) for coords in candidates):
            return False
    except (AssertionError, ValueError, IndexError):
        return False
    return True


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(22, 30))
    width = int(rng.integers(18, 30))
    structure_h = int(rng.integers(7, 11))
    structure_w = int(rng.integers(13, min(21, width - 1)))
    r0 = int(rng.integers(1, 4))
    c0 = int(rng.integers(1, width - structure_w))
    r1, c1 = r0 + structure_h - 1, c0 + structure_w - 1
    if r1 + 8 >= height:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    structure_box = (r0, c0, r1, c1)

    pocket_count = int(rng.integers(3, 5))
    base_shapes = _make_pocket_shapes(rng, pocket_count)
    if base_shapes is None:
        return None
    directions = _direction_plan(rng, pocket_count)
    carved = _carve_pockets(rng, grid, structure_box, base_shapes, directions)
    if carved is None:
        return None
    pocket_masks, _directions = carved

    detected_pockets, _ = _extract_pockets(grid)
    pocket_by_key = {
        _shape_key(_mask_from_coords(coords)): coords for coords in detected_pockets
    }
    pocket_keys = list(pocket_by_key)
    first_key = _shape_key(pocket_masks[0])
    if first_key not in pocket_by_key:
        return None

    match_count = int(rng.integers(1, pocket_count))
    remaining = [key for key in pocket_keys if key != first_key]
    rng.shuffle(remaining)
    matched_keys = {first_key, *remaining[: match_count - 1]}

    candidate_masks: list[np.ndarray] = []
    for key in matched_keys:
        pocket_mask = _mask_from_coords(pocket_by_key[key])
        variants = _rotations(pocket_mask)
        if key == first_key:
            variants = [variant for variant in variants if not np.array_equal(variant, pocket_mask)]
        if not variants:
            return None
        candidate_masks.append(variants[int(rng.integers(len(variants)))])

    pocket_key_set = set(pocket_keys)
    smaller_source = _mask_from_coords(pocket_by_key[first_key])
    smaller_decoy = _remove_one_cell(smaller_source, pocket_key_set)
    if smaller_decoy is None or not _strictly_fits(smaller_decoy, smaller_source):
        return None
    candidate_masks.append(_rotations(smaller_decoy)[int(rng.integers(len(_rotations(smaller_decoy))))])

    mirror_decoy = _normalize(np.fliplr(smaller_source))
    if _shape_key(mirror_decoy) in pocket_key_set:
        return None
    candidate_masks.append(_rotations(mirror_decoy)[int(rng.integers(len(_rotations(mirror_decoy))))])

    placed_candidates = _place_candidates(rng, grid, candidate_masks, r1 + 3)
    if placed_candidates is None:
        return None

    expected = grid.copy()
    for key in matched_keys:
        coords = pocket_by_key[key]
        expected[coords[:, 0], coords[:, 1]] = BLUE
    for coords in placed_candidates:
        if _shape_key(_mask_from_coords(coords)) in matched_keys:
            expected[coords[:, 0], coords[:, 1]] = BLACK

    if not _semantic_validation(
        grid,
        expected,
        detected_pockets,
        placed_candidates,
        matched_keys,
        smaller_decoy,
        smaller_source,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, fully disambiguated example with both negative branches."""
    rng = np.random.default_rng()
    for _ in range(800):
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pocket-absorption input")
