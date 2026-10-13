from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 9)
LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
SLOT_SIZES = (5, 6, 7, 8, 9)
GENERATION_RETRIES = 400


def _ranges_from_dividers(dividers: list[int], extent: int) -> list[tuple[int, int]]:
    cuts = [-1, *sorted(dividers), int(extent)]
    ranges = []
    for i in range(len(cuts) - 1):
        start = cuts[i] + 1
        stop = cuts[i + 1]
        if start >= stop:
            raise AssertionError("divider lines must leave nonempty square slots")
        ranges.append((start, stop))
    return ranges


def _parse_slots(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_divs = [r for r in range(h) if np.all(g[r, :] == GREY)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == GREY)]
    row_ranges = _ranges_from_dividers(row_divs, h)
    col_ranges = _ranges_from_dividers(col_divs, w)

    row_sizes = {r1 - r0 for r0, r1 in row_ranges}
    col_sizes = {c1 - c0 for c0, c1 in col_ranges}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all slots must share one size"

    slot_size = row_sizes.pop()
    assert slot_size == col_sizes.pop(), "slots must be square"

    slots = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            slots.append((r0, r1, c0, c1))
    assert len(slots) >= 2, "expected multiple slots"
    return int(slot_size), slots


def _all_transforms(tile: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(tile, dtype=int)
    transforms = [np.rot90(base, k).copy() for k in range(4)]
    transforms.extend(np.fliplr(np.rot90(base, k)).copy() for k in range(4))
    return transforms


def _anchor_coord(tile: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(tile, dtype=int) == TEAL)
    assert coords.shape == (1, 2), "expected exactly one teal anchor pixel"
    return tuple(map(int, coords[0]))


def _foreground_coords(slot: np.ndarray) -> np.ndarray:
    return np.argwhere((slot != BLACK) & (slot != GREY))


def _solve_from_anchor(template: np.ndarray, target_anchor: tuple[int, int]) -> tuple[int, np.ndarray]:
    matches = []
    for idx, transformed in enumerate(_all_transforms(template)):
        if _anchor_coord(transformed) == tuple(map(int, target_anchor)):
            matches.append((idx, transformed))
    assert len(matches) == 1, "each lone anchor must pick exactly one orientation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the lone teal anchor in each empty slot to choose the template's landing orientation.

    Input structure:
    - Full gray divider rows and columns split the canvas into equal square slots.
    - Exactly one slot contains a multicolored square template with one teal anchor pixel.
    - Every other slot is otherwise black and contains exactly one teal lone point.

    Rule:
    - Enumerate the template slot's four rotations and four mirrored rotations.
    - For each empty slot, find the unique orientation whose teal anchor lands on that slot's lone teal point.
    - Copy the whole transformed template into that slot while leaving the original template slot unchanged.

    Key insight:
    - The lone teal point is a landing constraint, not a decoration or color source, so each slot must be solved independently rather than by any left-to-right rotation sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _slot_size, slots = _parse_slots(g)
    template_slots = []
    template = None

    for idx, (r0, r1, c0, c1) in enumerate(slots):
        slot = g[r0:r1, c0:c1]
        fg = _foreground_coords(slot)
        if fg.shape[0] > 1:
            template_slots.append(idx)
            template = slot.copy()
        else:
            assert fg.shape == (1, 2), "empty slots must contain exactly one lone point"
            rr, cc = map(int, fg[0])
            assert int(slot[rr, cc]) == TEAL, "the lone point must be teal"

    assert len(template_slots) == 1, "expected one filled template slot"
    assert template is not None
    assert np.count_nonzero(template == TEAL) == 1, "template must contain one teal anchor"

    transforms = _all_transforms(template)
    anchor_positions = [_anchor_coord(t) for t in transforms]
    assert len(set(anchor_positions)) == 8, "anchor must distinguish all eight orientations"

    out = g.copy()
    template_idx = int(template_slots[0])
    for idx, (r0, r1, c0, c1) in enumerate(slots):
        if idx == template_idx:
            continue
        slot = g[r0:r1, c0:c1]
        target = tuple(map(int, _foreground_coords(slot)[0]))
        _, transformed = _solve_from_anchor(template, target)
        out[r0:r1, c0:c1] = transformed

    return out


def _random_walk_cuts(length: int, limit: int) -> np.ndarray:
    cut = random.randint(1, limit - 1)
    cuts = []
    for _ in range(length):
        cut = max(1, min(limit - 1, cut + random.choice((-2, -1, 0, 1, 2))))
        cuts.append(cut)
    if random.random() < 0.5:
        cuts.reverse()
    return np.array(cuts, dtype=int)


def _paint_accent(pattern: np.ndarray, color: int) -> None:
    size = pattern.shape[0]
    height = random.randint(1, max(1, size // 2))
    width = random.randint(1, max(1, size // 2))
    r0 = random.randint(0, size - height)
    c0 = random.randint(0, size - width)
    pattern[r0 : r0 + height, c0 : c0 + width] = int(color)


def _pattern_row_frontier(size: int, palette: tuple[int, ...]) -> np.ndarray:
    left, right = palette[:2]
    cuts = _random_walk_cuts(size, size)
    pattern = np.where(np.arange(size)[None, :] < cuts[:, None], left, right).astype(int)
    if len(palette) == 3:
        _paint_accent(pattern, palette[2])
    return pattern


def _pattern_col_frontier(size: int, palette: tuple[int, ...]) -> np.ndarray:
    top, bottom = palette[:2]
    cuts = _random_walk_cuts(size, size)
    pattern = np.where(np.arange(size)[:, None] < cuts[None, :], top, bottom).astype(int)
    if len(palette) == 3:
        _paint_accent(pattern, palette[2])
    return pattern


def _pattern_diagonal(size: int, palette: tuple[int, ...]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    first, second = palette[:2]
    threshold = random.randint(-2, 2)
    pattern = np.where((rr - cc) <= threshold, first, second).astype(int)
    if len(palette) == 3:
        sum_threshold = random.randint(size // 2, (3 * size) // 2)
        pattern[(rr + cc) >= sum_threshold] = palette[2]
    return pattern


def _pattern_voronoi(size: int, palette: tuple[int, ...]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    seeds = random.sample([(r, c) for r in range(size) for c in range(size)], len(palette))
    dists = []
    for idx, (sr, sc) in enumerate(seeds):
        dists.append(np.abs(rr - sr) + np.abs(cc - sc) + idx * 0.01)
    pattern = np.array(palette, dtype=int)[np.argmin(np.stack(dists, axis=0), axis=0)]
    return pattern.astype(int)


def _pattern_blocks(size: int, palette: tuple[int, ...]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    row_step = random.randint(1, min(3, size - 1))
    col_step = random.randint(1, min(3, size - 1))
    pattern = np.array(palette, dtype=int)[((rr // row_step) + (cc // col_step)) % len(palette)]
    if len(palette) == 3 and random.random() < 0.7:
        _paint_accent(pattern, random.choice(palette))
    return pattern.astype(int)


def _pattern(size: int) -> np.ndarray:
    color_count = random.choice((2, 3))
    palette = tuple(random.sample(PATTERN_COLORS, color_count))
    style = random.choice(
        (
            _pattern_row_frontier,
            _pattern_col_frontier,
            _pattern_diagonal,
            _pattern_voronoi,
            _pattern_blocks,
        )
    )
    pattern = style(size, palette)
    used = set(map(int, np.unique(pattern)))
    assert used <= set(palette)
    if len(used) != len(palette):
        raise AssertionError("all sampled pattern colors must appear")
    return pattern


def _on_symmetry_axis(r: int, c: int, size: int) -> bool:
    if r == c or r + c == size - 1:
        return True
    if size % 2 == 1:
        mid = size // 2
        if r == mid or c == mid:
            return True
    return False


def _anchor_orbit(coord: tuple[int, int], size: int) -> list[tuple[int, int]]:
    marker = np.zeros((size, size), dtype=int)
    marker[coord] = 1
    orbit = []
    for transformed in _all_transforms(marker):
        point = np.argwhere(transformed == 1)
        assert point.shape == (1, 2)
        orbit.append(tuple(map(int, point[0])))
    return orbit


def _generate_template(size: int) -> np.ndarray:
    for _ in range(200):
        pattern = _pattern(size)
        values, counts = np.unique(pattern, return_counts=True)
        color_counts = {int(v): int(c) for v, c in zip(values, counts)}
        candidates = []
        for r in range(size):
            for c in range(size):
                color = int(pattern[r, c])
                if color_counts[color] <= 1:
                    continue
                if _on_symmetry_axis(r, c, size):
                    continue
                if len(set(_anchor_orbit((r, c), size))) != 8:
                    continue
                candidates.append((r, c))
        if not candidates:
            continue

        anchor_r, anchor_c = random.choice(candidates)
        template = pattern.copy()
        template[anchor_r, anchor_c] = TEAL

        if len(set(map(int, np.unique(template))) - {TEAL}) < 2:
            continue
        keys = {
            tuple(map(int, transformed.shape)) + tuple(map(int, transformed.ravel()))
            for transformed in _all_transforms(template)
        }
        if len(keys) != 8:
            continue
        return template
    raise RuntimeError("failed to generate a template with eight unique anchor landings")


def _choose_transform_indices(count: int) -> list[int]:
    if count == 1:
        return [random.randrange(8)]

    rot_pool = list(range(4))
    mir_pool = list(range(4, 8))
    random.shuffle(rot_pool)
    random.shuffle(mir_pool)

    need_rot = 2 if count >= 4 else 1
    need_mir = 2 if count >= 4 else 1
    chosen = rot_pool[:need_rot] + mir_pool[:need_mir]

    remaining = [idx for idx in range(8) if idx not in chosen]
    random.shuffle(remaining)
    while len(chosen) < count:
        chosen.append(remaining.pop())
    random.shuffle(chosen)
    return chosen


def _build_grid(rows: int, cols: int, slot_size: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = rows * slot_size + (rows - 1)
    width = cols * slot_size + (cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    for row in range(1, rows):
        grid[row * slot_size + (row - 1), :] = GREY
    for col in range(1, cols):
        grid[:, col * slot_size + (col - 1)] = GREY

    slots = []
    for row in range(rows):
        for col in range(cols):
            r0 = row * (slot_size + 1)
            c0 = col * (slot_size + 1)
            slots.append((r0, r0 + slot_size, c0, c0 + slot_size))
    return grid, slots


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = random.choice(LAYOUTS)
    slot_size = random.choice(SLOT_SIZES)
    total_slots = rows * cols
    empty_count = total_slots - 1

    template = _generate_template(slot_size)
    transforms = _all_transforms(template)
    transform_indices = _choose_transform_indices(empty_count)

    inp, slots = _build_grid(rows, cols, slot_size)
    out = inp.copy()

    template_slot = random.randrange(total_slots)
    empty_slots = [idx for idx in range(total_slots) if idx != template_slot]
    assignments = dict(zip(empty_slots, transform_indices))

    for idx, (r0, r1, c0, c1) in enumerate(slots):
        if idx == template_slot:
            inp[r0:r1, c0:c1] = template
            out[r0:r1, c0:c1] = template
            continue

        transformed = transforms[assignments[idx]]
        anchor_r, anchor_c = _anchor_coord(transformed)
        inp[anchor_r + r0, anchor_c + c0] = TEAL
        out[r0:r1, c0:c1] = transformed

    _validate_instance(inp, out, template_slot, assignments)
    return inp, out


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    template_slot: int,
    assignments: dict[int, int],
) -> None:
    slot_size, slots = _parse_slots(inp)
    assert slot_size >= 5
    assert np.array_equal(main(inp), expected), "solver must match the independently built target"

    template = inp[slots[template_slot][0] : slots[template_slot][1], slots[template_slot][2] : slots[template_slot][3]]
    assert np.count_nonzero(template == TEAL) == 1
    transforms = _all_transforms(template)
    anchor_positions = [_anchor_coord(t) for t in transforms]
    assert len(set(anchor_positions)) == 8

    used_indices = []
    for idx, (r0, r1, c0, c1) in enumerate(slots):
        slot = inp[r0:r1, c0:c1]
        fg = _foreground_coords(slot)
        if idx == template_slot:
            assert fg.shape[0] > 1
            continue
        assert fg.shape == (1, 2)
        target = tuple(map(int, fg[0]))
        matches = [k for k, pos in enumerate(anchor_positions) if pos == target]
        assert matches == [assignments[idx]]
        assert np.array_equal(expected[r0:r1, c0:c1], transforms[matches[0]])
        used_indices.append(matches[0])

    if len(used_indices) >= 2:
        assert any(idx < 4 for idx in used_indices), "need at least one pure rotation branch"
        assert any(idx >= 4 for idx in used_indices), "need at least one mirrored branch"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            grid, _expected = _build_instance()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid anchor-landing slot puzzle")
