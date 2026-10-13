from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 700


Coord = tuple[int, int]


@dataclass(frozen=True)
class Axis:
    color: int
    kind: str
    value: int
    endpoints: tuple[Coord, Coord]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[Coord] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        comp = [(sr, sc)]
        q: deque[Coord] = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
                    comp.append((rr, cc))
        comps.append(comp)
    return comps


def _edge_points(g: np.ndarray) -> list[Coord]:
    h, w = g.shape
    pts: list[Coord] = []
    for r, c in np.argwhere(g != BG):
        r, c = int(r), int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            pts.append((r, c))
    return pts


def _axis_cells(shape: tuple[int, int], axis: Axis) -> list[Coord]:
    h, w = shape
    cells: list[Coord] = []
    if axis.kind == "plus":
        for r in range(h):
            c = r - axis.value
            if 0 <= c < w:
                cells.append((int(r), int(c)))
    else:
        for r in range(h):
            c = axis.value - r
            if 0 <= c < w:
                cells.append((int(r), int(c)))
    return cells


def _axis_from_pair(shape: tuple[int, int], color: int, pts: list[Coord]) -> Axis:
    assert len(pts) == 2, "each endpoint color must occur twice on the edge"
    a, b = sorted((tuple(map(int, p)) for p in pts))
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert abs(dr) == abs(dc) and dr != 0, "endpoint pairs must define a 45 degree diagonal"

    if a[0] - a[1] == b[0] - b[1]:
        axis = Axis(int(color), "plus", int(a[0] - a[1]), (a, b))
    elif a[0] + a[1] == b[0] + b[1]:
        axis = Axis(int(color), "minus", int(a[0] + a[1]), (a, b))
    else:
        raise AssertionError("endpoint pair must lie on one diagonal")

    cells = _axis_cells(shape, axis)
    assert len(cells) >= 3, "mirror diagonal must span multiple cells"
    assert {cells[0], cells[-1]} == set(axis.endpoints), "edge pixels must be the diagonal's two grid-edge endpoints"
    return axis


def _parse_axes(g: np.ndarray) -> tuple[dict[int, Axis], set[Coord]]:
    endpoints = _edge_points(g)
    assert len(endpoints) == 4, "expected exactly four edge endpoint pixels"

    by_color: dict[int, list[Coord]] = {}
    for p in endpoints:
        by_color.setdefault(int(g[p]), []).append(p)
    assert len(by_color) == 2, "endpoints must form two same-color pairs"

    axes: dict[int, Axis] = {}
    kinds: set[str] = set()
    endpoint_set = set(endpoints)
    for color, pts in by_color.items():
        axis = _axis_from_pair(g.shape, int(color), pts)
        axes[int(color)] = axis
        kinds.add(axis.kind)

    assert kinds == {"plus", "minus"}, "the two hidden mirrors must have opposite diagonal directions"

    for p in endpoint_set:
        for rr, cc in _neighbors8(*p):
            if _in_bounds(g.shape, rr, cc):
                assert int(g[rr, cc]) == BG, "endpoint pixels must be isolated"

    for axis in axes.values():
        own_endpoints = set(axis.endpoints)
        for cell in _axis_cells(g.shape, axis):
            if cell in own_endpoints:
                assert int(g[cell]) == int(axis.color), "axis endpoint color mismatch"
            else:
                assert int(g[cell]) == BG, "hidden mirror axis must not be drawn or touched"

    return axes, endpoint_set


def _side_value(axis: Axis, r: int, c: int) -> int:
    if axis.kind == "plus":
        return int(r - c - axis.value)
    return int(r + c - axis.value)


def _reflect(axis: Axis, r: int, c: int) -> Coord:
    r, c = int(r), int(c)
    if axis.kind == "plus":
        return int(c + axis.value), int(r - axis.value)
    return int(axis.value - c), int(axis.value - r)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each keyed multicolor sprite across the invisible diagonal mirror selected by its key color.

    Input structure:
    - A black grid has exactly four isolated edge pixels, grouped as two same-color endpoint pairs.
    - Each endpoint pair defines one unpainted 45 degree diagonal mirror; the two mirrors have opposite diagonal directions.
    - Away from those hidden axes are separated 4-connected multicolor sprites.
    - Every sprite contains exactly one pixel whose color matches one endpoint pair, and that pixel is its key.

    Rule:
    - Recover both hidden diagonal axes from their same-color edge endpoint pairs.
    - For each sprite, use the color of its single key pixel to choose the matching axis.
    - Reflect every colored cell of the sprite across that selected diagonal, preserving each cell color.
    - Keep the original sprites and the four endpoint pixels, and leave the mirror axes unpainted.

    Key insight:
    - The nearest mirror is a decoy; a sprite chooses its invisible diagonal mirror only by the embedded key color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    axes, endpoints = _parse_axes(g)

    sprite_mask = g != BG
    for r, c in endpoints:
        sprite_mask[r, c] = False

    comps = _components(sprite_mask)
    assert comps, "expected at least one keyed sprite"

    out = g.copy()
    for comp in sorted(comps, key=lambda cells: _bbox(cells)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(comp) >= 2 and len(colors) >= 2, "each sprite must be a multicolor component"

        key_cells = [(r, c) for r, c in comp if int(g[r, c]) in axes]
        assert len(key_cells) == 1, "each sprite must contain exactly one axis-color key pixel"
        key_color = int(g[key_cells[0]])
        axis = axes[key_color]

        sides = [_side_value(axis, r, c) for r, c in comp]
        assert all(v != 0 for v in sides), "sprites may not touch their selected axis"
        assert all(v > 0 for v in sides) or all(v < 0 for v in sides), "each sprite must lie on one side of its selected axis"

        reflected: list[tuple[int, int, int]] = []
        target_coords: set[Coord] = set()
        for r, c in comp:
            rr, cc = _reflect(axis, r, c)
            assert _in_bounds(g.shape, rr, cc), "reflected sprite must remain in bounds"
            assert _side_value(axis, rr, cc) == -_side_value(axis, r, c), "reflection must cross the selected axis"
            target_coords.add((rr, cc))
            reflected.append((rr, cc, int(g[r, c])))

        assert len(target_coords) == len(comp), "reflection must keep the sprite cells distinct"
        assert target_coords.isdisjoint({(r, c) for r, c in comp}), "reflected sprite must land on the other side"

        for rr, cc, _color in reflected:
            assert int(out[rr, cc]) == BG, "reflected sprites must land on empty background"
        for rr, cc, color in reflected:
            out[rr, cc] = int(color)

    return out


def _axis_options(shape: tuple[int, int], kind: str) -> list[int]:
    h, w = shape
    if kind == "plus":
        values = range(-(w - 1), h)
    else:
        values = range(h + w - 1)
    min_len = max(8, min(h, w) - 5)
    options = []
    for value in values:
        dummy = Axis(1, kind, int(value), ((0, 0), (0, 0)))
        if len(_axis_cells(shape, dummy)) >= min_len:
            options.append(int(value))
    return options


def _axis_with_color(shape: tuple[int, int], color: int, kind: str, value: int) -> Axis:
    dummy = Axis(int(color), kind, int(value), ((0, 0), (0, 0)))
    cells = _axis_cells(shape, dummy)
    return Axis(int(color), kind, int(value), (cells[0], cells[-1]))


def _cheb(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _choose_axes(shape: tuple[int, int]) -> tuple[Axis, Axis] | None:
    plus_vals = _axis_options(shape, "plus")
    minus_vals = _axis_options(shape, "minus")
    if not plus_vals or not minus_vals:
        return None

    colors = random.sample(PALETTE, 2)
    for _ in range(200):
        plus = _axis_with_color(shape, colors[0], "plus", random.choice(plus_vals))
        minus = _axis_with_color(shape, colors[1], "minus", random.choice(minus_vals))
        endpoints = list(plus.endpoints + minus.endpoints)
        if len(set(endpoints)) != 4:
            continue
        if any(_cheb(a, b) <= 1 for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]):
            continue
        if any(_side_value(plus, *p) == 0 for p in minus.endpoints):
            continue
        if any(_side_value(minus, *p) == 0 for p in plus.endpoints):
            continue
        return plus, minus
    return None


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    r0, c0, r1, c1 = _bbox(list(cells))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    return mask


def _make_sprite(key_color: int, body_palette: list[int]) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(5, min(10, h * w))
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        degree = {}
        for r, c in coords:
            degree[(r, c)] = sum((rr, cc) in coords for rr, cc in _neighbors4(r, c))
        key_candidates = [p for p in coords if degree[p] >= 2] or coords
        key = random.choice(key_candidates)
        body_cells = [p for p in coords if p != key]
        if len(body_cells) < 4:
            continue

        n_body_colors = random.randint(2, min(4, len(body_palette), len(body_cells)))
        chosen = random.sample(body_palette, n_body_colors)
        random.shuffle(body_cells)
        colors = chosen[:]
        while len(colors) < len(body_cells):
            colors.append(random.choice(chosen))
        random.shuffle(colors)

        sprite = np.zeros(mask.shape, dtype=int)
        sprite[key] = int(key_color)
        for (r, c), color in zip(body_cells, colors):
            sprite[r, c] = int(color)

        if len(set(map(int, sprite[sprite != BG]))) >= 3:
            return sprite
    return None


def _sprite_cells(sprite: np.ndarray, r0: int, c0: int) -> list[tuple[int, int, int]]:
    cells = []
    for rr, cc in np.argwhere(sprite != BG):
        cells.append((int(r0 + rr), int(c0 + cc), int(sprite[rr, cc])))
    return cells


def _too_close_to_set(cells: set[Coord], occupied: set[Coord], gap: int = 1) -> bool:
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _sets_too_close(a: set[Coord], b: set[Coord], gap: int = 1) -> bool:
    return _too_close_to_set(a, b, gap=gap)


def _near_any_axis(cells: set[Coord], axis_sets: list[set[Coord]]) -> bool:
    for r, c in cells:
        for rr in (r - 1, r, r + 1):
            for cc in (c - 1, c, c + 1):
                if any((rr, cc) in axis_set for axis_set in axis_sets):
                    return True
    return False


def _axis_distance(axis: Axis, cells: set[Coord]) -> int:
    return min(abs(_side_value(axis, r, c)) for r, c in cells)


def _find_sprite_placement(
    shape: tuple[int, int],
    sprite: np.ndarray,
    axis: Axis,
    other_axis: Axis,
    axis_sets: list[set[Coord]],
    occupied_output: set[Coord],
    must_be_decoy: bool,
) -> tuple[Coord, list[tuple[int, int, int]], list[tuple[int, int, int]], bool] | None:
    h, w = shape
    sh, sw = sprite.shape
    origins = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(origins)

    for r0, c0 in origins:
        original = _sprite_cells(sprite, r0, c0)
        orig_coords = {(r, c) for r, c, _color in original}
        if _too_close_to_set(orig_coords, occupied_output, gap=1):
            continue
        if _near_any_axis(orig_coords, axis_sets):
            continue

        sides = [_side_value(axis, r, c) for r, c in orig_coords]
        if not (all(v > 0 for v in sides) or all(v < 0 for v in sides)):
            continue

        reflected: list[tuple[int, int, int]] = []
        ok = True
        for r, c, color in original:
            rr, cc = _reflect(axis, r, c)
            if not _in_bounds(shape, rr, cc):
                ok = False
                break
            reflected.append((rr, cc, int(color)))
        if not ok:
            continue

        refl_coords = {(r, c) for r, c, _color in reflected}
        if len(refl_coords) != len(orig_coords):
            continue
        if orig_coords & refl_coords:
            continue
        if _sets_too_close(orig_coords, refl_coords, gap=1):
            continue
        if _too_close_to_set(refl_coords, occupied_output, gap=1):
            continue
        if _near_any_axis(refl_coords, axis_sets):
            continue

        selected_distance = _axis_distance(axis, orig_coords)
        other_distance = _axis_distance(other_axis, orig_coords)
        decoy = selected_distance > other_distance
        if must_be_decoy and not decoy:
            continue

        return (int(r0), int(c0)), original, reflected, bool(decoy)
    return None


def _paint(cells: list[tuple[int, int, int]], grid: np.ndarray) -> None:
    for r, c, color in cells:
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)


def _valid_generation(inp: np.ndarray, expected: np.ndarray, axes: list[Axis], sprite_keys: list[int], decoy_seen: bool) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not decoy_seen:
        return False
    if set(sprite_keys) != {axis.color for axis in axes}:
        return False

    try:
        parsed, endpoints = _parse_axes(inp)
        solved = main(inp)
    except AssertionError:
        return False

    if set(parsed) != {axis.color for axis in axes}:
        return False
    if len(endpoints) != 4:
        return False

    added = np.count_nonzero(expected != inp)
    if added < 8:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    shape = (h, w)
    axes_pair = _choose_axes(shape)
    if axes_pair is None:
        return None
    axes = list(axes_pair)
    axes_by_color = {axis.color: axis for axis in axes}
    body_palette = [color for color in PALETTE if color not in axes_by_color]

    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    occupied_output: set[Coord] = set()
    for axis in axes:
        for r, c in axis.endpoints:
            inp[r, c] = expected[r, c] = int(axis.color)
            occupied_output.add((r, c))

    axis_sets = [set(_axis_cells(shape, axis)) for axis in axes]

    first_key = random.choice([axis.color for axis in axes])
    other_key = next(axis.color for axis in axes if axis.color != first_key)
    sprite_count = random.randint(3, 5)
    sprite_keys = [first_key, other_key]
    while len(sprite_keys) < sprite_count:
        sprite_keys.append(random.choice([axis.color for axis in axes]))

    decoy_seen = False
    placed_keys: list[int] = []
    for idx, key_color in enumerate(sprite_keys):
        axis = axes_by_color[int(key_color)]
        other_axis = next(a for a in axes if a.color != int(key_color))
        placed = None
        must_decoy = idx == 0
        for _ in range(120):
            sprite = _make_sprite(int(key_color), body_palette)
            if sprite is None:
                continue
            placed = _find_sprite_placement(
                shape,
                sprite,
                axis,
                other_axis,
                axis_sets,
                occupied_output,
                must_be_decoy=must_decoy,
            )
            if placed is not None:
                break
        if placed is None:
            return None

        _origin, original, reflected, is_decoy = placed
        _paint(original, inp)
        _paint(original, expected)
        _paint(reflected, expected)
        occupied_output |= {(r, c) for r, c, _color in original}
        occupied_output |= {(r, c) for r, c, _color in reflected}
        decoy_seen = decoy_seen or bool(is_decoy)
        placed_keys.append(int(key_color))

    if not _valid_generation(inp, expected, axes, placed_keys, decoy_seen):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for hidden diagonal mirrors")
