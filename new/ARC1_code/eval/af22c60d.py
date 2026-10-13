from __future__ import annotations
import numpy as np

from arc_utils import Color, paste
import random


def _best_reflection_constant(g: np.ndarray, axis: int, missing: int) -> int:
    n = g.shape[axis]
    candidates: list[tuple[int, int, int]] = []

    for s in range(2 * n - 1):
        pair_count = 0
        compared = 0
        touches_missing = 0
        ok = True

        for i in range(n):
            j = s - i
            if not (0 <= j < n) or i > j:
                continue

            a = g[i, :] if axis == 0 else g[:, i]
            b = g[j, :] if axis == 0 else g[:, j]
            known = (a != missing) & (b != missing)
            if np.any(a[known] != b[known]):
                ok = False
                break

            pair_count += 1
            compared += int(np.sum(known))
            touches_missing += int(np.sum((a == missing) | (b == missing)))

        if ok and touches_missing:
            candidates.append((pair_count, compared, s))

    assert candidates
    max_pairs = max(pair_count for pair_count, _, _ in candidates)
    widest = [candidate for candidate in candidates if candidate[0] == max_pairs]
    max_compared = max(compared for _, compared, _ in widest)
    best = [s for _, compared, s in widest if compared == max_compared]
    assert len(best) == 1
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill black gaps with the unique known color from each cell's horizontal/vertical mirror orbit.

    Input structure
    The grid is an otherwise nonblack pattern with color 0 marking missing pixels. Nonblack cells reveal a horizontal reflection over the lower row band and a vertical reflection over the right column band.

    Rule
    Infer the row-reflection and column-reflection constants whose reflected nonblack pairs agree. For each black pixel, collect its row mirror, column mirror, and combined row+column mirror positions that stay in bounds; the known colors in that orbit must be unique, and every black pixel in the orbit is recolored to it.

    Key insight
    A gap can face another gap across one mirror axis, so the fill must use the full two-axis reflection orbit rather than a single reflected cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    missing = Color.BLACK
    assert np.any(g == missing)

    h, w = g.shape
    row_s = _best_reflection_constant(g, axis=0, missing=missing)
    col_s = _best_reflection_constant(g, axis=1, missing=missing)

    out = g.copy()
    for r, c in np.argwhere(g == missing):
        rows = {int(r)}
        rr = row_s - int(r)
        if 0 <= rr < h:
            rows.add(rr)

        cols = {int(c)}
        cc = col_s - int(c)
        if 0 <= cc < w:
            cols.add(cc)

        orbit = {(orow, ocol) for orow in rows for ocol in cols}
        values = {int(g[orow, ocol]) for orow, ocol in orbit if g[orow, ocol] != missing}
        assert len(values) == 1
        value = values.pop()

        for orow, ocol in orbit:
            if out[orow, ocol] == missing:
                out[orow, ocol] = value
            else:
                assert out[orow, ocol] == value

    assert not np.any(out == missing)
    return out

# --------------- generator (o5x) ---------------

# Canvas and reflection layout.
_GEN_SIZE = 30
_GEN_REFLECT_S = 31
_GEN_PREFIX = 2
_GEN_CORE_STOP = 16

# Palette sampling and local color correlation.
_GEN_COLOR_IDS = range(1, 10)
_GEN_PALETTE_SIZE_RANGE = (5, 7)
_GEN_NEIGHBOR_COPY_PROB = 0.44

# Mask component count and black-cell budget.
_GEN_RECT_COUNT_RANGE = (3, 4)
_GEN_MIN_BLACK = 36
_GEN_MAX_BLACK = 117

# Rectangle sizes and placement padding.
_GEN_RECT_SIDE_MIN, _GEN_RECT_SIDE_MAX = 2, 11
_GEN_RECT_PADDING = 1
_GEN_RECT_MIN_AREA = _GEN_RECT_SIDE_MIN * _GEN_RECT_SIDE_MIN
_GEN_RECT_MAX_AREA = _GEN_RECT_SIDE_MAX * _GEN_RECT_SIDE_MAX
_GEN_RECT_OPTIONS = [
    (h, w, h * w) for h in range(_GEN_RECT_SIDE_MIN, _GEN_RECT_SIDE_MAX + 1)
    for w in range(_GEN_RECT_SIDE_MIN, _GEN_RECT_SIDE_MAX + 1)
]

# Rejection sampling.
_GEN_ATTEMPT_LIMIT = 300


def _gen_mirror(x: int) -> int:
    return _GEN_REFLECT_S - int(x)


def _gen_orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    rr = _gen_mirror(r)
    cc = _gen_mirror(c)
    if 0 <= rr < _GEN_SIZE:
        rows.add(rr)
    if 0 <= cc < _GEN_SIZE:
        cols.add(cc)
    return {(orow, ocol) for orow in rows for ocol in cols}


def _gen_pick_color(out: np.ndarray, palette: list[int], neighbors: list[tuple[int, int]]) -> int:
    seen = [int(out[r, c]) for r, c in neighbors if int(out[r, c]) in palette]
    if seen and random.random() < _GEN_NEIGHBOR_COPY_PROB:
        return random.choice(seen)
    return random.choice(palette)


def _gen_set_core_orbit(out: np.ndarray, r: int, c: int, color: int) -> None:
    for rr in (int(r), _gen_mirror(r)):
        for cc in (int(c), _gen_mirror(c)):
            out[rr, cc] = int(color)


def _gen_fill_prefix_rows(out: np.ndarray, palette: list[int]) -> None:
    for r in range(_GEN_PREFIX):
        for c in range(_GEN_PREFIX, _GEN_CORE_STOP):
            neighbors = []
            if c > _GEN_PREFIX:
                neighbors.append((r, c - 1))
            if r > 0:
                neighbors.append((r - 1, c))
            color = _gen_pick_color(out, palette, neighbors)
            out[r, c] = color
            out[r, _gen_mirror(c)] = color


def _gen_fill_prefix_cols(out: np.ndarray, palette: list[int]) -> None:
    for c in range(_GEN_PREFIX):
        for r in range(_GEN_PREFIX, _GEN_CORE_STOP):
            neighbors = []
            if r > _GEN_PREFIX:
                neighbors.append((r - 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            color = _gen_pick_color(out, palette, neighbors)
            out[r, c] = color
            out[_gen_mirror(r), c] = color


def _gen_fill_core(out: np.ndarray, palette: list[int]) -> None:
    for r in range(_GEN_PREFIX, _GEN_CORE_STOP):
        for c in range(_GEN_PREFIX, _GEN_CORE_STOP):
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            color = _gen_pick_color(out, palette, neighbors)
            _gen_set_core_orbit(out, r, c, color)


def _gen_force_palette_coverage(out: np.ndarray, palette: list[int]) -> None:
    present = set(map(int, out.flat))
    missing = [color for color in palette if color not in present]
    core_cells = [(r, c) for r in range(_GEN_PREFIX, _GEN_CORE_STOP) for c in range(_GEN_PREFIX, _GEN_CORE_STOP)]
    forced_cells = random.sample(core_cells, len(missing))
    for color, (r, c) in zip(missing, forced_cells):
        _gen_set_core_orbit(out, r, c, color)


def _gen_build_output() -> np.ndarray:
    palette = random.sample(_GEN_COLOR_IDS, random.randint(*_GEN_PALETTE_SIZE_RANGE))
    out = np.zeros((_GEN_SIZE, _GEN_SIZE), dtype=int)
    out[:_GEN_PREFIX, :_GEN_PREFIX] = np.random.choice(palette, size=(_GEN_PREFIX, _GEN_PREFIX))
    _gen_fill_prefix_rows(out, palette)
    _gen_fill_prefix_cols(out, palette)
    _gen_fill_core(out, palette)
    _gen_force_palette_coverage(out, palette)
    return out


def _gen_area_possible(count: int, total: int) -> bool:
    return _GEN_RECT_MIN_AREA * int(count) <= int(total) <= _GEN_RECT_MAX_AREA * int(count)


def _gen_choose_rect_dims(count: int, total: int) -> list[tuple[int, int]] | None:
    memo: set[tuple[int, int]] = set()

    def rec(k: int, remaining: int) -> list[tuple[int, int]] | None:
        if k == 0:
            return [] if remaining == 0 else None
        if not _gen_area_possible(k, remaining):
            return None
        state = (k, remaining)
        if state in memo:
            return None

        options = list(_GEN_RECT_OPTIONS)
        random.shuffle(options)
        for h, w, area in options:
            rem = remaining - area
            if _gen_area_possible(k - 1, rem):
                tail = rec(k - 1, rem)
                if tail is not None:
                    return [(h, w), *tail]

        memo.add(state)
        return None

    return rec(int(count), int(total))


def _gen_rect_has_padding(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0) + int(h)
    c1 = int(c0) + int(w)
    rr0 = max(0, int(r0) - _GEN_RECT_PADDING)
    cc0 = max(0, int(c0) - _GEN_RECT_PADDING)
    rr1 = min(_GEN_SIZE, r1 + _GEN_RECT_PADDING)
    cc1 = min(_GEN_SIZE, c1 + _GEN_RECT_PADDING)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _gen_orbits_are_solvable(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        orbit = _gen_orbit(int(r), int(c))
        if all(bool(mask[rr, cc]) for rr, cc in orbit):
            return False
    return True


def _gen_place_rect(mask: np.ndarray, h: int, w: int) -> bool:
    candidates: list[tuple[int, int]] = []
    for r0 in range(_GEN_PREFIX, _GEN_SIZE - int(h) + 1):
        for c0 in range(_GEN_PREFIX, _GEN_SIZE - int(w) + 1):
            if not _gen_rect_has_padding(mask, r0, c0, h, w):
                continue
            trial = mask.copy()
            trial[r0 : r0 + int(h), c0 : c0 + int(w)] = True
            if _gen_orbits_are_solvable(trial):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    paste(mask, np.ones((int(h), int(w)), dtype=bool), r0, c0)
    return True


def _gen_build_mask() -> np.ndarray | None:
    count = random.randint(*_GEN_RECT_COUNT_RANGE)
    target = random.randint(_GEN_MIN_BLACK, _GEN_MAX_BLACK)
    dims = _gen_choose_rect_dims(count, target)
    if dims is None:
        return None

    mask = np.zeros((_GEN_SIZE, _GEN_SIZE), dtype=bool)
    for h, w in sorted(dims, key=lambda item: item[0] * item[1], reverse=True):
        if not _gen_place_rect(mask, h, w):
            return None

    if not (_GEN_MIN_BLACK <= int(mask.sum()) <= _GEN_MAX_BLACK):
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    out = _gen_build_output()
    mask = _gen_build_mask()
    if mask is None:
        return None

    inp = out.copy()
    inp[mask] = 0

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid af22c60d input")
