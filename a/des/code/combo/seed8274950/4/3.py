from __future__ import annotations

import random

import numpy as np


AxisKind = str

_AXIS_ORDER: tuple[AxisKind, ...] = ("row", "col", "main_diag", "anti_diag")
_DIR4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    assert values.size > 0, "expected a non-empty grid"
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _candidate_axes(shape: tuple[int, int]) -> list[AxisKind]:
    h, w = map(int, shape)
    axes: list[AxisKind] = ["row", "col"]
    if h == w:
        axes.extend(["main_diag", "anti_diag"])
    return axes


def _reflect(axis: AxisKind, shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if axis == "row":
        return h - 1 - r, c
    if axis == "col":
        return r, w - 1 - c
    if axis == "main_diag":
        assert h == w, "main diagonal symmetry requires a square grid"
        return c, r
    if axis == "anti_diag":
        assert h == w, "anti-diagonal symmetry requires a square grid"
        return h - 1 - c, w - 1 - r
    raise AssertionError(f"unknown axis kind: {axis}")


def _iter_orbits(shape: tuple[int, int], axis: AxisKind):
    h, w = map(int, shape)
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            rr, cc = _reflect(axis, shape, r, c)
            seen[r, c] = True
            seen[rr, cc] = True
            if (r, c) <= (rr, cc):
                first, second = (r, c), (rr, cc)
            else:
                first, second = (rr, cc), (r, c)
            if first == second:
                yield (first,)
            else:
                yield (first, second)


def _axis_stats(grid: np.ndarray, bg: int, axis: AxisKind) -> dict[str, int]:
    mismatch_pairs = 0
    conflict_pairs = 0
    equal_nonbg_cells = 0
    active_orbits = 0

    for orbit in _iter_orbits(grid.shape, axis):
        if len(orbit) == 1:
            color = int(grid[orbit[0]])
            if color != bg:
                active_orbits += 1
                equal_nonbg_cells += 1
            continue

        (r1, c1), (r2, c2) = orbit
        a = int(grid[r1, c1])
        b = int(grid[r2, c2])
        if a == bg and b == bg:
            continue
        active_orbits += 1
        if a == b:
            if a != bg:
                equal_nonbg_cells += 2
        else:
            mismatch_pairs += 1
            if a != bg and b != bg:
                conflict_pairs += 1

    return {
        "mismatch_pairs": mismatch_pairs,
        "conflict_pairs": conflict_pairs,
        "equal_nonbg_cells": equal_nonbg_cells,
        "active_orbits": active_orbits,
    }


def _axis_support(grid: np.ndarray, bg: int, axis: AxisKind) -> dict[int, int]:
    support: dict[int, int] = {}
    for orbit in _iter_orbits(grid.shape, axis):
        if len(orbit) == 1:
            color = int(grid[orbit[0]])
            if color != bg:
                support[color] = support.get(color, 0) + 1
            continue

        (r1, c1), (r2, c2) = orbit
        a = int(grid[r1, c1])
        b = int(grid[r2, c2])
        if a == b and a != bg:
            support[a] = support.get(a, 0) + 2
    return support


def _axis_key(stats: dict[str, int], axis: AxisKind) -> tuple[int, int, int, int, int]:
    return (
        int(stats["mismatch_pairs"]),
        int(stats["conflict_pairs"]),
        -int(stats["equal_nonbg_cells"]),
        -int(stats["active_orbits"]),
        _AXIS_ORDER.index(axis),
    )


def _choose_axis(grid: np.ndarray, bg: int) -> AxisKind:
    ranked = sorted(
        (_axis_key(_axis_stats(grid, bg, axis), axis), axis) for axis in _candidate_axes(grid.shape)
    )
    return ranked[0][1]


def _count_of(counts: np.ndarray, color: int) -> int:
    return int(counts[color]) if 0 <= color < len(counts) else 0


def _resolve_pair(
    a: int,
    b: int,
    bg: int,
    counts: np.ndarray,
    support: dict[int, int],
) -> int:
    if a == b:
        return a
    if a == bg:
        return b
    if b == bg:
        return a

    count_a = _count_of(counts, a)
    count_b = _count_of(counts, b)
    if count_a != count_b:
        return a if count_a > count_b else b

    support_a = int(support.get(a, 0))
    support_b = int(support.get(b, 0))
    if support_a != support_b:
        return a if support_a > support_b else b
    return a


def _repair_with_axis(grid: np.ndarray, bg: int, axis: AxisKind) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    counts = np.bincount(g.ravel())
    support = _axis_support(g, bg, axis)
    out = g.copy()

    for orbit in _iter_orbits(g.shape, axis):
        if len(orbit) == 1:
            continue
        (r1, c1), (r2, c2) = orbit
        a = int(g[r1, c1])
        b = int(g[r2, c2])
        if a == b:
            continue
        color = _resolve_pair(a, b, bg, counts, support)
        out[r1, c1] = color
        out[r2, c2] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the best reflection axis and restore the nearly symmetric foreground pattern.

    Candidate axes:
    - horizontal midline
    - vertical midline
    - main diagonal
    - anti-diagonal

    Repair rule for a mismatched mirrored pair:
    - if one side is background and the other is foreground, copy the foreground color to both;
    - if both sides are non-background but differ, keep the globally more frequent color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    axis = _choose_axis(g, bg)
    return _repair_with_axis(g, bg, axis)


# --------------- generator ---------------

_PALETTE = tuple(range(10))
_GEN_RETRIES = 400
_SHAPE_RANGE = (8, 16)
_PAIR_FILL_MIN = 0.15
_PAIR_FILL_MAX = 0.34
_AXIS_FILL_PROB = 0.65
_NEW_COMPONENT_PROB = 0.28
_REMOVE_COUNT_MAX = 3
_RECOLOR_COUNT_MAX = 2
_UNIQUE_GAP_MIN = 1


def _pair_side_mask(shape: tuple[int, int], axis: AxisKind) -> np.ndarray:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    if axis == "row":
        return rr < h - 1 - rr
    if axis == "col":
        return cc < w - 1 - cc
    if axis == "main_diag":
        assert h == w
        return rr < cc
    if axis == "anti_diag":
        assert h == w
        return rr + cc < h - 1
    raise AssertionError(f"unknown axis kind: {axis}")


def _fixed_mask(shape: tuple[int, int], axis: AxisKind) -> np.ndarray:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    if axis == "row":
        return rr == h - 1 - rr
    if axis == "col":
        return cc == w - 1 - cc
    if axis == "main_diag":
        assert h == w
        return rr == cc
    if axis == "anti_diag":
        assert h == w
        return rr + cc == h - 1
    raise AssertionError(f"unknown axis kind: {axis}")


def _maybe_apply_margin(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    if h <= 6 or w <= 6:
        return mask
    margin = random.randint(0, 1)
    if margin == 0:
        return mask
    trimmed = np.zeros_like(mask, dtype=bool)
    trimmed[margin : h - margin, margin : w - margin] = True
    narrowed = mask & trimmed
    return narrowed if int(narrowed.sum()) >= 6 else mask


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = map(int, shape)
    for dr, dc in _DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _grow_cells(allowed: np.ndarray, target: int) -> list[tuple[int, int]] | None:
    allowed_coords = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if target == 0:
        return []
    if len(allowed_coords) < target:
        return None

    selected: set[tuple[int, int]] = {random.choice(allowed_coords)}
    while len(selected) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in selected:
            for rr, cc in _neighbors4(r, c, allowed.shape):
                if allowed[rr, cc] and (rr, cc) not in selected:
                    frontier.add((rr, cc))

        if frontier and random.random() >= _NEW_COMPONENT_PROB:
            selected.add(random.choice(tuple(frontier)))
            continue

        remaining = [coord for coord in allowed_coords if coord not in selected]
        if not remaining:
            return None
        selected.add(random.choice(remaining))

    return sorted(selected)


def _build_expected(shape: tuple[int, int], axis: AxisKind, bg: int, fg: int) -> np.ndarray | None:
    pair_mask = _maybe_apply_margin(_pair_side_mask(shape, axis))
    fixed = _maybe_apply_margin(_fixed_mask(shape, axis))
    pair_capacity = int(pair_mask.sum())
    if pair_capacity < 6:
        return None

    lo = max(5, int(np.ceil(pair_capacity * _PAIR_FILL_MIN)))
    hi = min(max(lo, 8), int(np.floor(pair_capacity * _PAIR_FILL_MAX)))
    hi = min(hi, pair_capacity)
    if lo > hi:
        return None
    pair_target = random.randint(lo, hi)

    pair_cells = _grow_cells(pair_mask, pair_target)
    if pair_cells is None:
        return None

    fixed_capacity = int(fixed.sum())
    fixed_cells: list[tuple[int, int]] = []
    if fixed_capacity > 0 and random.random() < _AXIS_FILL_PROB:
        fixed_target = random.randint(1, min(fixed_capacity, max(1, min(shape) // 3)))
        built = _grow_cells(fixed, fixed_target)
        if built is None:
            return None
        fixed_cells = built

    out = np.full(shape, bg, dtype=int)
    for r, c in pair_cells:
        rr, cc = _reflect(axis, shape, r, c)
        out[r, c] = fg
        out[rr, cc] = fg
    for r, c in fixed_cells:
        out[r, c] = fg
    return out


def _occupied_pair_orbits(grid: np.ndarray, bg: int, axis: AxisKind) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for orbit in _iter_orbits(grid.shape, axis):
        if len(orbit) != 2:
            continue
        (r1, c1), (r2, c2) = orbit
        if int(grid[r1, c1]) != bg and int(grid[r2, c2]) != bg:
            pairs.append(((r1, c1), (r2, c2)))
    return pairs


def _rank_axes(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int, int], AxisKind, dict[str, int]]]:
    bg = _background_color(grid)
    ranked = []
    for axis in _candidate_axes(grid.shape):
        stats = _axis_stats(grid, bg, axis)
        ranked.append((_axis_key(stats, axis), axis, stats))
    ranked.sort(key=lambda item: item[0])
    return ranked


def _is_unique_best(grid: np.ndarray, expected_axis: AxisKind) -> bool:
    ranked = _rank_axes(grid)
    if ranked[0][1] != expected_axis:
        return False
    if len(ranked) == 1:
        return True

    best = ranked[0][2]
    second = ranked[1][2]
    if ranked[1][0] == ranked[0][0]:
        return False
    if second["mismatch_pairs"] - best["mismatch_pairs"] >= _UNIQUE_GAP_MIN:
        return True
    if (
        second["mismatch_pairs"] == best["mismatch_pairs"]
        and second["equal_nonbg_cells"] < best["equal_nonbg_cells"]
    ):
        return True
    return False


def _sample_shape(axis: AxisKind) -> tuple[int, int]:
    if axis in {"main_diag", "anti_diag"}:
        n = random.randint(*_SHAPE_RANGE)
        return n, n

    if random.random() < 0.5:
        h = random.randint(*_SHAPE_RANGE)
        w = random.randint(*_SHAPE_RANGE)
        return h, w

    n = random.randint(*_SHAPE_RANGE)
    other = random.randint(_SHAPE_RANGE[0], _SHAPE_RANGE[1] - 1)
    if random.random() < 0.5:
        return n, other
    return other, n


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.choice(_AXIS_ORDER)
    shape = _sample_shape(axis)
    colors = list(_PALETTE)
    random.shuffle(colors)
    bg, fg, noise = map(int, colors[:3])

    expected = _build_expected(shape, axis, bg, fg)
    if expected is None:
        return None

    occupied_pairs = _occupied_pair_orbits(expected, bg, axis)
    if len(occupied_pairs) < 2:
        return None

    noisy = expected.copy()
    random.shuffle(occupied_pairs)

    remove_count = random.randint(1, min(_REMOVE_COUNT_MAX, len(occupied_pairs) - 1))
    recolor_count = random.randint(1, min(_RECOLOR_COUNT_MAX, len(occupied_pairs) - remove_count))

    cursor = 0
    for _ in range(remove_count):
        orbit = occupied_pairs[cursor]
        cursor += 1
        side = random.randrange(2)
        r, c = orbit[side]
        noisy[r, c] = bg

    for _ in range(recolor_count):
        orbit = occupied_pairs[cursor]
        cursor += 1
        side = random.randrange(2)
        r, c = orbit[side]
        noisy[r, c] = noise

    main_count = int(np.count_nonzero(noisy == fg))
    noise_count = int(np.count_nonzero(noisy == noise))
    if main_count <= noise_count:
        return None
    if not _is_unique_best(noisy, axis):
        return None
    if not np.array_equal(main(noisy), expected):
        return None
    return noisy, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid symmetric-noise ARC input")
