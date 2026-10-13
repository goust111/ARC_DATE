from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
TEMPLATE_COLORS = (2, 3, 4, 6, 7, 8, 9)
TEMPLATE_OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
LEFT_TEMPLATE_SLOTS = ((-1, -1), (0, -1), (1, -1))
AXIS_TEMPLATE_SLOTS = ((-1, 0), (1, 0))
MAX_GENERATION_ATTEMPTS = 400


class Instance:
    def __init__(
        self,
        shape: tuple[int, int],
        decorated: tuple[int, int],
        ordinary: tuple[tuple[int, int], ...],
        half_template: tuple[tuple[int, int, int], ...],
    ) -> None:
        self.shape = shape
        self.decorated = decorated
        self.ordinary = ordinary
        self.half_template = half_template


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    return sorted(tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE))


def _colored_neighbors(grid: np.ndarray, center: tuple[int, int]) -> dict[tuple[int, int], int]:
    r, c = center
    found: dict[tuple[int, int], int] = {}
    for dr, dc in TEMPLATE_OFFSETS:
        if dr == 0 and dc == 0:
            continue
        rr, cc = r + dr, c + dc
        if not _in_bounds(grid.shape, rr, cc):
            continue
        color = int(grid[rr, cc])
        if color not in (BLACK, BLUE):
            found[(dr, dc)] = color
    return found


def _full_template_from_half(half_template: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    full: dict[tuple[int, int], int] = {}
    for (dr, dc), color in half_template.items():
        for key in ((dr, dc), (dr, -dc)):
            prev = full.get(key)
            assert prev in (None, color), "mirrored template colors must agree"
            full[key] = color
    return full


def _expanded_box(shape: tuple[int, int], center: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r, c = center
    return max(0, r - 2), max(0, c - 2), min(h - 1, r + 2), min(w - 1, c + 2)


def _boxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _stamp_template(
    grid: np.ndarray,
    center: tuple[int, int],
    template: dict[tuple[int, int], int],
) -> None:
    r, c = center
    for (dr, dc), color in template.items():
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc):
            grid[rr, cc] = color
    grid[r, c] = BLUE


def _paint_outline(grid: np.ndarray, center: tuple[int, int]) -> None:
    r0, c0, r1, c1 = _expanded_box(grid.shape, center)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if grid[r, c] == BLACK:
                    grid[r, c] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the decorated marker's half-template by left-right symmetry, stamp it onto every blue marker, then frame each stamped 3x3 neighborhood.

    Input structure:
    - The grid is mostly black background with several isolated blue marker pixels.
    - Exactly one blue marker is decorated: within its 3x3 neighborhood, some non-black non-blue cells appear only on the left half and/or symmetry axis.
    - Every other blue marker has a black 3x3 neighborhood and acts only as a stamp target.

    Rule:
    - Find the unique decorated blue marker by checking which marker has non-black non-blue cells in its 3x3 neighborhood.
    - Read those colored offsets, mirror them across the marker column to recover the full symmetric 3x3 template, and keep the center cell blue.
    - Stamp that recovered template around every blue marker, clipping out-of-bounds cells, then for each marker paint grey on the black perimeter of its center-aligned 5x5 expanded box.

    Key insight:
    - The visible neighborhood around the exemplar marker is only half of the real template; the missing right half must be inferred by mirror symmetry before the copied stamp and its surrounding grey frame can be produced correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _sorted_markers(g)
    assert markers, "expected at least one blue marker"

    decorated: list[tuple[tuple[int, int], dict[tuple[int, int], int]]] = []
    ordinary: list[tuple[int, int]] = []
    for marker in markers:
        offsets = _colored_neighbors(g, marker)
        if offsets:
            decorated.append((marker, offsets))
        else:
            ordinary.append(marker)

    assert len(decorated) == 1, "expected exactly one decorated marker"
    decorated_center, half_template = decorated[0]
    assert ordinary, "expected at least one ordinary marker"
    assert all(dc <= 0 for _dr, dc in half_template), "decorated input must expose only the left half of the template"

    template = _full_template_from_half(half_template)
    out = g.copy()
    for marker in markers:
        _stamp_template(out, marker, template)
    for marker in markers:
        _paint_outline(out, marker)
    return out


def _construct_input(instance: Instance) -> np.ndarray:
    grid = np.zeros(instance.shape, dtype=int)
    dr, dc = instance.decorated
    grid[dr, dc] = BLUE
    for r, c in instance.ordinary:
        grid[r, c] = BLUE
    for off_r, off_c, color in instance.half_template:
        grid[dr + off_r, dc + off_c] = color
    return grid


def _expected_output(instance: Instance) -> np.ndarray:
    grid = _construct_input(instance)
    template = _full_template_from_half({(dr, dc): color for dr, dc, color in instance.half_template})
    markers = [instance.decorated, *instance.ordinary]
    for marker in markers:
        _stamp_template(grid, marker, template)
    for marker in sorted(markers):
        _paint_outline(grid, marker)
    return grid


def _candidate_centers(
    shape: tuple[int, int],
    require_full_3x3: bool,
    require_clipped_frame: bool | None,
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if require_full_3x3 and not (1 <= r < h - 1 and 1 <= c < w - 1):
                continue
            clipped = _expanded_box(shape, (r, c)) != (r - 2, c - 2, r + 2, c + 2)
            if require_clipped_frame is not None and clipped != require_clipped_frame:
                continue
            candidates.append((r, c))
    return candidates


def _choose_center(
    rng: np.random.Generator,
    shape: tuple[int, int],
    chosen: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
) -> tuple[int, int] | None:
    boxes = [_expanded_box(shape, center) for center in chosen]
    valid = [
        center
        for center in candidates
        if all(_boxes_disjoint(_expanded_box(shape, center), box) for box in boxes)
    ]
    if not valid:
        return None
    return valid[int(rng.integers(0, len(valid)))]


def _generate_half_template(rng: np.random.Generator) -> tuple[tuple[int, int, int], ...]:
    count = int(rng.integers(2, 4))
    slots = list(LEFT_TEMPLATE_SLOTS + AXIS_TEMPLATE_SLOTS)
    while True:
        rng.shuffle(slots)
        chosen = slots[:count]
        if not any(dc == -1 for _dr, dc in chosen):
            continue
        if not any(dc == 0 for _dr, dc in chosen):
            if count == 2 and rng.random() < 0.4:
                continue
        color_count = 1 if rng.random() < 0.45 else 2
        palette = rng.choice(np.array(TEMPLATE_COLORS), size=color_count, replace=False)
        template = []
        for dr, dc in sorted(chosen):
            color = int(palette[int(rng.integers(0, len(palette)))])
            template.append((dr, dc, color))
        return tuple(template)


def _generate_instance(rng: np.random.Generator) -> Instance | None:
    shape = (int(rng.integers(15, 19)), int(rng.integers(15, 19)))
    ordinary_count = int(rng.integers(2, 5))
    decorated = _choose_center(
        rng,
        shape,
        [],
        _candidate_centers(shape, require_full_3x3=True, require_clipped_frame=False),
    )
    if decorated is None:
        return None

    chosen = [decorated]
    boundary_marker = _choose_center(
        rng,
        shape,
        chosen,
        _candidate_centers(shape, require_full_3x3=False, require_clipped_frame=True),
    )
    if boundary_marker is None:
        return None
    chosen.append(boundary_marker)

    interior_marker = _choose_center(
        rng,
        shape,
        chosen,
        _candidate_centers(shape, require_full_3x3=False, require_clipped_frame=False),
    )
    if interior_marker is None:
        return None
    chosen.append(interior_marker)

    remaining_needed = ordinary_count - 2
    if remaining_needed > 0:
        all_candidates = _candidate_centers(shape, require_full_3x3=False, require_clipped_frame=None)
        for _ in range(remaining_needed):
            extra = _choose_center(rng, shape, chosen, all_candidates)
            if extra is None:
                return None
            chosen.append(extra)

    ordinary = tuple(chosen[1:])
    half_template = _generate_half_template(rng)
    return Instance(shape=shape, decorated=decorated, ordinary=ordinary, half_template=half_template)


def _validate_instance(instance: Instance) -> bool:
    inp = _construct_input(instance)
    out = _expected_output(instance)

    markers = _sorted_markers(inp)
    if len(markers) != 1 + len(instance.ordinary):
        return False
    if markers != sorted([instance.decorated, *instance.ordinary]):
        return False

    decorated_candidates = [marker for marker in markers if _colored_neighbors(inp, marker)]
    if decorated_candidates != [instance.decorated]:
        return False

    for marker in instance.ordinary:
        if _colored_neighbors(inp, marker):
            return False

    half_dict = {(dr, dc): color for dr, dc, color in instance.half_template}
    if not (2 <= len(half_dict) <= 3):
        return False
    if not any(dc == -1 for _dr, dc in half_dict):
        return False

    full_template = _full_template_from_half(half_dict)
    if not any(dc == 1 for _dr, dc in full_template):
        return False

    decorated_r, decorated_c = instance.decorated
    mirrored_growth = False
    for (dr, dc), color in full_template.items():
        rr, cc = decorated_r + dr, decorated_c + dc
        if not _in_bounds(inp.shape, rr, cc):
            continue
        if dc > 0 and inp[rr, cc] == BLACK and out[rr, cc] == color:
            mirrored_growth = True
            break
    if not mirrored_growth:
        return False

    ordinary_boxes = [_expanded_box(inp.shape, marker) for marker in instance.ordinary]
    if not any(box[0] == 0 or box[1] == 0 or box[2] == inp.shape[0] - 1 or box[3] == inp.shape[1] - 1 for box in ordinary_boxes):
        return False
    if not any(box == (r - 2, c - 2, r + 2, c + 2) for (r, c), box in zip(instance.ordinary, ordinary_boxes)):
        return False

    for a, b in zip(markers, markers[1:]):
        if not _boxes_disjoint(_expanded_box(inp.shape, a), _expanded_box(inp.shape, b)):
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, out):
        return False

    for marker in markers:
        mr, mc = marker
        for dr, dc in TEMPLATE_OFFSETS:
            rr, cc = mr + dr, mc + dc
            if not _in_bounds(inp.shape, rr, cc):
                continue
            expected = BLUE if (dr, dc) == (0, 0) else full_template.get((dr, dc), BLACK)
            if actual[rr, cc] != expected:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _generate_instance(rng)
        if instance is None:
            continue
        if _validate_instance(instance):
            return _construct_input(instance)
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
