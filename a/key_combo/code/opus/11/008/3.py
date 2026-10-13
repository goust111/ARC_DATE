from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
PINK = 6

_QUADRANT_ORDER = ("TL", "TR", "BR", "BL")
_GEN_ROW_CHOICES = (14, 16, 18, 20)
_GEN_COL_CHOICES = (14, 16, 18, 20)
_GEN_TEMPLATE_SHAPE = (3, 4)
_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrants(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], ...]:
    h, w = map(int, shape)
    hr = h // 2
    wc = w // 2
    return (
        (0, hr, 0, wc),
        (0, hr, wc, w),
        (hr, h, wc, w),
        (hr, h, 0, wc),
    )


def _in_quadrant(cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, r1, c0, c1 = bounds
    return r0 <= r < r1 and c0 <= c < c1


def _quadrant_index_of(cell: tuple[int, int], shape: tuple[int, int]) -> int:
    for index, bounds in enumerate(_quadrants(shape)):
        if _in_quadrant(cell, bounds):
            return index
    raise ValueError(f"cell {cell} is out of bounds")


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    normalized = tuple(sorted((int(r) - min_r, int(c) - min_c) for r, c in coords))
    return normalized, (int(min_r), int(min_c))


def _orient(template: np.ndarray, k: int, mirror: bool = False) -> np.ndarray:
    base = np.fliplr(template) if mirror else template
    return np.rot90(base, k)


def _orientation_records(template: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mirror in (False, True):
        for k in range(4):
            arr = _orient(template, k, mirror)
            pink = [tuple(map(int, rc)) for rc in np.argwhere(arr == PINK)]
            yellow = [tuple(map(int, rc)) for rc in np.argwhere(arr == YELLOW)]
            pink_shape, pink_min = _normalize_coords(pink)
            yellow_rel_to_pink = tuple(
                sorted((int(r) - pink_min[0], int(c) - pink_min[1]) for r, c in yellow)
            )
            records.append(
                {
                    "array": arr,
                    "k": int(k),
                    "mirror": bool(mirror),
                    "pink": tuple(pink),
                    "yellow": tuple(yellow),
                    "pink_shape": pink_shape,
                    "pink_min": pink_min,
                    "yellow_rel_to_pink": yellow_rel_to_pink,
                }
            )
    return records


def _expected_rotation_k(template_quadrant: int, other_quadrant: int) -> int:
    delta = (int(other_quadrant) - int(template_quadrant)) % 4
    mapping = {1: 3, 2: 2, 3: 1}
    assert delta in mapping, "quadrants must differ"
    return mapping[delta]


def _quadrant_pink_coords(grid: np.ndarray, quadrant_index: int) -> list[tuple[int, int]]:
    bounds = _quadrants(grid.shape)[quadrant_index]
    r0, r1, c0, c1 = bounds
    local = np.argwhere(grid[r0:r1, c0:c1] == PINK)
    return [(r0 + int(r), c0 + int(c)) for r, c in local]


def _quadrant_candidates(
    grid: np.ndarray,
    records: list[dict[str, object]],
    quadrant_index: int,
) -> list[dict[str, object]]:
    bounds = _quadrants(grid.shape)[quadrant_index]
    observed = _quadrant_pink_coords(grid, quadrant_index)
    assert observed, f"quadrant {_QUADRANT_ORDER[quadrant_index]} must contain magenta"
    observed_shape, observed_min = _normalize_coords(observed)
    observed_set = set(observed)

    candidates: list[dict[str, object]] = []
    for record in records:
        if record["pink_shape"] != observed_shape:
            continue
        pink_min = record["pink_min"]
        top = observed_min[0] - int(pink_min[0])
        left = observed_min[1] - int(pink_min[1])

        translated_pink = {(top + int(r), left + int(c)) for r, c in record["pink"]}
        if translated_pink != observed_set:
            continue
        if any(not _in_quadrant(cell, bounds) for cell in translated_pink):
            continue

        visible_yellow = 0
        additions: list[tuple[int, int]] = []
        valid = True
        for r, c in record["yellow"]:
            rr = top + int(r)
            cc = left + int(c)
            if not _in_bounds(grid.shape, rr, cc):
                continue
            visible_yellow += 1
            if not _in_quadrant((rr, cc), bounds):
                valid = False
                break
            if int(grid[rr, cc]) == PINK:
                valid = False
                break
            if int(grid[rr, cc]) != YELLOW:
                additions.append((rr, cc))

        if not valid or visible_yellow == 0:
            continue
        candidates.append(
            {
                "k": int(record["k"]),
                "mirror": bool(record["mirror"]),
                "top": int(top),
                "left": int(left),
                "yellow": record["yellow"],
                "additions": tuple(additions),
                "visible_yellow": int(visible_yellow),
            }
        )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each magenta-only quadrant copy of the yellow/magenta template using quadrant-relative rotation to break signature ties.

    Input structure:
    - A black background is split logically into four quadrants by the grid center.
    - Exactly one quadrant contains the complete small rectangular template, tiled only by yellow and magenta.
    - Each other quadrant contains one incomplete copy that shows only the template's magenta signature pixels.

    Rule:
    - Extract the full template from the bounding box of all yellow pixels.
    - In each non-template quadrant, compare the visible magenta pattern against all 8 rotated/reflected template variants and collect every matching orientation/placement.
    - If several candidates share the same visible magenta signature, keep the one whose rotation matches that quadrant's clockwise position relative to the template quadrant: clockwise neighbor = 90 degrees clockwise, opposite = 180 degrees, counterclockwise neighbor = 90 degrees counterclockwise.
    - Paint the chosen variant's in-bounds yellow pixels yellow, discarding any yellow cells that would fall outside the grid.

    Key insight:
    - The magenta pattern identifies where a copy is, but because the signature is intentionally ambiguous under some transforms, the quadrant location is the second constraint that selects the correct rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, YELLOW, PINK}), "expected only black, yellow, and magenta"

    yellow_box = _bbox(g == YELLOW)
    assert yellow_box is not None, "expected one complete yellow/magenta template"
    tr0, tc0, tr1, tc1 = yellow_box
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert set(map(int, np.unique(template))) == {YELLOW, PINK}, "template bbox must be fully tiled by yellow/magenta"

    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    template_quadrants = {_quadrant_index_of(cell, g.shape) for cell in yellow_cells}
    assert len(template_quadrants) == 1, "yellow pixels must all lie in one quadrant"
    template_quadrant = next(iter(template_quadrants))

    records = _orientation_records(template)
    out = g.copy()
    for quadrant_index in range(4):
        if quadrant_index == template_quadrant:
            continue

        candidates = _quadrant_candidates(g, records, quadrant_index)
        assert candidates, f"quadrant {_QUADRANT_ORDER[quadrant_index]} must match at least one orientation"

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            expected_k = _expected_rotation_k(template_quadrant, quadrant_index)
            filtered = [candidate for candidate in candidates if int(candidate["k"]) == expected_k]
            assert len(filtered) == 1, "quadrant rotation must disambiguate to exactly one orientation"
            chosen = filtered[0]

        for rr, cc in chosen["additions"]:
            out[rr, cc] = YELLOW

    return out


def _template_ok(template: np.ndarray) -> bool:
    if template.shape != _GEN_TEMPLATE_SHAPE:
        return False
    if _bbox(template == YELLOW) != (0, 0, template.shape[0] - 1, template.shape[1] - 1):
        return False

    pink = np.argwhere(template == PINK)
    if not (3 <= len(pink) <= 5):
        return False
    if len(set(map(int, pink[:, 0]))) < 2 or len(set(map(int, pink[:, 1]))) < 2:
        return False

    records = _orientation_records(template)
    for required_k in (1, 2, 3):
        target = next(rec for rec in records if not rec["mirror"] and int(rec["k"]) == required_k)
        group = [rec for rec in records if rec["pink_shape"] == target["pink_shape"]]
        if len(group) != 2:
            return False
        if sum(int(rec["k"]) == required_k for rec in group) != 1:
            return False
        if len({rec["yellow_rel_to_pink"] for rec in group}) < 2:
            return False
    return True


def _build_template_catalog() -> tuple[np.ndarray, ...]:
    h, w = _GEN_TEMPLATE_SHAPE
    templates: list[np.ndarray] = []
    for bits in range(1, (1 << (h * w)) - 1):
        template = np.full((h, w), YELLOW, dtype=int)
        for idx in range(h * w):
            if (bits >> idx) & 1:
                template[idx // w, idx % w] = PINK
        if _template_ok(template):
            templates.append(template)
    assert templates, "template catalog must not be empty"
    return tuple(templates)


_TEMPLATE_CATALOG = _build_template_catalog()


def _paste_full(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _paste_masked(dst: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(patch == int(color)):
        dst[top + int(r), left + int(c)] = int(color)


def _position_choices_inside_quadrant(
    bounds: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    h, w = map(int, shape)
    row_lo = r0 + (1 if r1 - r0 - h >= 2 else 0)
    row_hi = r1 - h - (1 if r1 - r0 - h >= 2 else 0)
    col_lo = c0 + (1 if c1 - c0 - w >= 2 else 0)
    col_hi = c1 - w - (1 if c1 - c0 - w >= 2 else 0)
    return [(top, left) for top in range(row_lo, row_hi + 1) for left in range(col_lo, col_hi + 1)]


def _copy_positions(
    grid_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    patch: np.ndarray,
    require_clip: bool,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    h, w = map(int, patch.shape)
    pink = [tuple(map(int, rc)) for rc in np.argwhere(patch == PINK)]
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(patch == YELLOW)]
    positions: list[tuple[int, int]] = []

    for top in range(r0 - h + 1, r1):
        for left in range(c0 - w + 1, c1):
            translated_pink = [(top + pr, left + pc) for pr, pc in pink]
            if not translated_pink:
                continue
            if any((not _in_bounds(grid_shape, rr, cc)) or (not _in_quadrant((rr, cc), bounds)) for rr, cc in translated_pink):
                continue

            visible_yellow = []
            clipped = False
            valid = True
            for yr, yc in yellow:
                rr = top + yr
                cc = left + yc
                if not _in_bounds(grid_shape, rr, cc):
                    clipped = True
                    continue
                if not _in_quadrant((rr, cc), bounds):
                    valid = False
                    break
                visible_yellow.append((rr, cc))

            if not valid or not visible_yellow:
                continue
            if require_clip != clipped:
                continue
            positions.append((top, left))
    return positions


def _copy_additions(patch: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    cells = []
    for r, c in np.argwhere(patch == YELLOW):
        rr = top + int(r)
        cc = left + int(c)
        if _in_bounds(shape, rr, cc):
            cells.append((rr, cc))
    return tuple(cells)


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(inp == YELLOW)]
    if not yellow_cells:
        return False
    template_quadrants = {_quadrant_index_of(cell, inp.shape) for cell in yellow_cells}
    if len(template_quadrants) != 1:
        return False
    template_quadrant = next(iter(template_quadrants))

    yellow_box = _bbox(inp == YELLOW)
    assert yellow_box is not None
    r0, c0, r1, c1 = yellow_box
    template = inp[r0 : r1 + 1, c0 : c1 + 1]
    records = _orientation_records(template)

    seen_change = False
    for quadrant_index in range(4):
        bounds = _quadrants(inp.shape)[quadrant_index]
        r0q, r1q, c0q, c1q = bounds
        quad_yellow = int(np.count_nonzero(inp[r0q:r1q, c0q:c1q] == YELLOW))
        quad_pink = int(np.count_nonzero(inp[r0q:r1q, c0q:c1q] == PINK))
        if quadrant_index == template_quadrant:
            if quad_yellow == 0 or quad_pink == 0:
                return False
            continue
        if quad_yellow != 0 or quad_pink == 0:
            return False

        candidates = _quadrant_candidates(inp, records, quadrant_index)
        if len(candidates) < 2:
            return False
        expected_k = _expected_rotation_k(template_quadrant, quadrant_index)
        chosen = [candidate for candidate in candidates if int(candidate["k"]) == expected_k]
        if len(chosen) != 1:
            return False
        additions = set(chosen[0]["additions"])
        if not additions:
            return False
        for rr in range(r0q, r1q):
            for cc in range(c0q, c1q):
                in_expected = (rr, cc) in additions
                if in_expected and int(expected[rr, cc]) != YELLOW:
                    return False
                if not in_expected and int(inp[rr, cc]) != int(expected[rr, cc]):
                    return False
        seen_change = seen_change or bool(additions)

    return seen_change


def _generate_once() -> np.ndarray | None:
    h = random.choice(_GEN_ROW_CHOICES)
    w = random.choice(_GEN_COL_CHOICES)
    grid = np.full((h, w), BLACK, dtype=int)
    expected = grid.copy()

    template_seed = random.choice(_TEMPLATE_CATALOG)
    template = _orient(template_seed, random.randrange(4), bool(random.getrandbits(1)))
    template_quadrant = random.randrange(4)
    quad_bounds = _quadrants(grid.shape)

    full_positions = _position_choices_inside_quadrant(quad_bounds[template_quadrant], template.shape)
    if not full_positions:
        return None
    template_top, template_left = random.choice(full_positions)
    _paste_full(grid, template, template_top, template_left)
    _paste_full(expected, template, template_top, template_left)

    other_quadrants = [index for index in range(4) if index != template_quadrant]
    clip_quadrant = random.choice(other_quadrants) if random.random() < 0.55 else None

    for quadrant_index in other_quadrants:
        rotation_k = _expected_rotation_k(template_quadrant, quadrant_index)
        patch = np.rot90(template, rotation_k)

        require_clip = quadrant_index == clip_quadrant
        positions = _copy_positions(grid.shape, quad_bounds[quadrant_index], patch, require_clip=require_clip)
        if not positions and require_clip:
            return None
        if not positions:
            positions = _copy_positions(grid.shape, quad_bounds[quadrant_index], patch, require_clip=False)
        if not positions:
            return None

        top, left = random.choice(positions)
        _paste_masked(grid, patch, top, left, PINK)
        _paste_masked(expected, patch, top, left, PINK)
        for rr, cc in _copy_additions(patch, top, left, grid.shape):
            expected[rr, cc] = YELLOW

    if not _semantic_check(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
