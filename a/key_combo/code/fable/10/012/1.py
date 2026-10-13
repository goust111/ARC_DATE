from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: set[tuple[int, int]] = set()
        while q:
            rr, cc = q.popleft()
            comp.add((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def reflect_cells(
    cells: set[tuple[int, int]], orientation: str, axis: int
) -> set[tuple[int, int]]:
    if orientation == "vertical":
        return {(r, 2 * int(axis) - c) for r, c in cells}
    return {(2 * int(axis) - r, c) for r, c in cells}


def pair_axis(pair: set[tuple[int, int]]) -> tuple[str, int]:
    assert len(pair) == 2
    (r0, c0), (r1, c1) = sorted(pair)
    if r0 == r1:
        assert c0 != c1 and (c0 + c1) % 2 == 0
        return "vertical", (c0 + c1) // 2
    if c0 == c1:
        assert r0 != r1 and (r0 + r1) % 2 == 0
        return "horizontal", (r0 + r1) // 2
    raise AssertionError("singleton pair must share a row or a column")


def _parse_color(grid: np.ndarray, color: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], str, int]:
    comps = components_of_color(grid, color)
    singletons = [comp for comp in comps if len(comp) == 1]
    sprites = [comp for comp in comps if len(comp) > 1]
    assert len(singletons) == 2, "each color must have exactly two isolated markers"
    assert len(sprites) == 1, "each color must have exactly one multi-pixel sprite"

    pair = set().union(*singletons)
    sprite = sprites[0]
    orientation, axis = pair_axis(pair)

    if orientation == "vertical":
        cols = [c for _, c in sprite]
        assert max(cols) < axis or min(cols) > axis, "sprite must lie on one side of its axis"
        assert all(c != axis for _, c in sprite), "sprite cannot touch its hidden axis"
    else:
        rows = [r for r, _ in sprite]
        assert max(rows) < axis or min(rows) > axis, "sprite must lie on one side of its axis"
        assert all(r != axis for r, _ in sprite), "sprite cannot touch its hidden axis"

    mirrored = reflect_cells(sprite, orientation, axis)
    h, w = grid.shape
    assert all(0 <= r < h and 0 <= c < w for r, c in mirrored), "mirrored sprite must fit"
    assert mirrored.isdisjoint(sprite), "input contains only one side of the sprite"
    assert mirrored.isdisjoint(pair), "markers cannot lie on the mirrored sprite"
    return pair, sprite, orientation, axis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each same-colored sprite across the private mirror axis defined by its isolated marker pair.

    Input structure:
    - A dominant black background contains several foreground colors.
    - For each foreground color, exactly two isolated single pixels of that color appear, plus one same-colored multi-pixel 4-connected sprite.
    - The two single pixels of a color lie either on one row or on one column, while that color's sprite lies strictly on one side of the axis implied by the pair.

    Rule:
    - Preserve every input pixel.
    - If a color's two isolated pixels share a row, they define a vertical mirror axis at the midpoint column between them.
    - If they share a column, they define a horizontal mirror axis at the midpoint row between them.
    - Mirror that color's multi-pixel sprite across its own axis and paint the reflected copy in the same color.

    Key insight:
    - There is no single global axis: each color carries its own hidden axis, and the pair orientation itself tells you whether that private axis is vertical or horizontal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground color"

    out = g.copy()
    seen_axes: set[tuple[str, int]] = set()
    for color in colors:
        _pair, sprite, orientation, axis = _parse_color(g, color)
        axis_sig = (orientation, axis)
        assert axis_sig not in seen_axes, "different colors must not share the same hidden axis"
        seen_axes.add(axis_sig)
        for r, c in reflect_cells(sprite, orientation, axis):
            assert out[r, c] in (bg, color), "reflections must not overwrite other colors"
            out[r, c] = int(color)
    return out


BG = 0
PALETTE = tuple(range(1, 10))
GRID_MIN = 16
GRID_MAX = 28
COLOR_COUNT_MIN = 2
COLOR_COUNT_MAX = 4
SHAPE_H_MIN = 2
SHAPE_H_MAX = 4
SHAPE_W_MIN = 2
SHAPE_W_MAX = 5
SHAPE_AREA_MIN = 3
SHAPE_AREA_MAX = 7
PAIR_DIST_MIN = 2
PAIR_DIST_MAX = 5
SPRITE_GAP_MIN = 1
SPRITE_GAP_MAX = 4
GENERATION_ATTEMPTS = 400


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _random_connected_shape() -> set[tuple[int, int]]:
    for _ in range(200):
        h = random.randint(SHAPE_H_MIN, SHAPE_H_MAX)
        w = random.randint(SHAPE_W_MIN, SHAPE_W_MAX)
        max_area = min(SHAPE_AREA_MAX, h * w - 1)
        if max_area < SHAPE_AREA_MIN:
            continue
        area = random.randint(SHAPE_AREA_MIN, max_area)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        cells = _normalize_cells(cells)
        hh, ww = _shape_dims(cells)
        if hh < 2 or ww < 2:
            continue
        if len(cells) == hh * ww:
            continue
        return cells
    raise RuntimeError("failed to sample a non-rectangular connected shape")


def _all_neighbors8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((r + dr, c + dc))
    return padded


def _pair_is_isolated(pair: set[tuple[int, int]], others: set[tuple[int, int]]) -> bool:
    for r, c in pair:
        for rr, cc in others:
            if abs(r - rr) <= 1 and abs(c - cc) <= 1:
                return False
    return True


def _sample_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("vertical", "horizontal"))]
    p = random.random()
    if p < 0.6:
        orientations = ["vertical", "horizontal"]
        orientations.extend(random.choice(("vertical", "horizontal")) for _ in range(count - 2))
        random.shuffle(orientations)
        return orientations
    if p < 0.8:
        return ["vertical"] * count
    return ["horizontal"] * count


def _candidate_spec(
    height: int,
    width: int,
    color: int,
    orientation: str,
    used_axes: set[tuple[str, int]],
    blocked_pad: set[tuple[int, int]],
) -> dict | None:
    for _ in range(250):
        shape = _random_connected_shape()
        sh, sw = _shape_dims(shape)

        placements: list[tuple[int, int, str]] = []
        if orientation == "vertical":
            for axis in range(1, width - 1):
                if ("vertical", axis) in used_axes:
                    continue
                if min(axis, width - 1 - axis) < PAIR_DIST_MIN:
                    continue
                for gap in range(SPRITE_GAP_MIN, SPRITE_GAP_MAX + 1):
                    c0_left = axis - gap - sw + 1
                    if c0_left >= 0 and axis + gap + sw - 1 < width:
                        placements.append((axis, c0_left, "left"))
                    c0_right = axis + gap
                    if c0_right + sw - 1 < width and axis - gap - sw + 1 >= 0:
                        placements.append((axis, c0_right, "right"))
        else:
            for axis in range(1, height - 1):
                if ("horizontal", axis) in used_axes:
                    continue
                if min(axis, height - 1 - axis) < PAIR_DIST_MIN:
                    continue
                for gap in range(SPRITE_GAP_MIN, SPRITE_GAP_MAX + 1):
                    r0_up = axis - gap - sh + 1
                    if r0_up >= 0 and axis + gap + sh - 1 < height:
                        placements.append((axis, r0_up, "up"))
                    r0_down = axis + gap
                    if r0_down + sh - 1 < height and axis - gap - sh + 1 >= 0:
                        placements.append((axis, r0_down, "down"))

        random.shuffle(placements)
        for axis, anchor, side in placements:
            if orientation == "vertical":
                r0 = random.randint(0, height - sh)
                c0 = anchor
                dmax = min(PAIR_DIST_MAX, axis, width - 1 - axis)
                if dmax < PAIR_DIST_MIN:
                    continue
                pair_r = random.randint(0, height - 1)
                d = random.randint(PAIR_DIST_MIN, dmax)
                pair = {(pair_r, axis - d), (pair_r, axis + d)}
                sprite = {(r0 + r, c0 + c) for r, c in shape}
            else:
                r0 = anchor
                c0 = random.randint(0, width - sw)
                dmax = min(PAIR_DIST_MAX, axis, height - 1 - axis)
                if dmax < PAIR_DIST_MIN:
                    continue
                pair_c = random.randint(0, width - 1)
                d = random.randint(PAIR_DIST_MIN, dmax)
                pair = {(axis - d, pair_c), (axis + d, pair_c)}
                sprite = {(r0 + r, c0 + c) for r, c in shape}

            mirror = reflect_cells(sprite, orientation, axis)
            final_cells = sprite | mirror | pair
            if len(final_cells) != len(sprite) * 2 + 2:
                continue
            if any(cell in blocked_pad for cell in final_cells):
                continue
            if not _pair_is_isolated(pair, sprite | mirror):
                continue
            return {
                "color": int(color),
                "pair": pair,
                "sprite": sprite,
                "mirror": mirror,
                "orientation": orientation,
                "axis": int(axis),
                "side": side,
            }
    return None


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        color = spec["color"]
        for r, c in spec["pair"] | spec["sprite"]:
            grid[r, c] = color
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = spec["color"]
        for r, c in spec["mirror"]:
            assert out[r, c] == BG
            out[r, c] = color
    return out


def _validate_semantics(inp: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    axes = {(spec["orientation"], spec["axis"]) for spec in specs}
    if len(axes) != len(specs):
        return False

    for spec in specs:
        color = spec["color"]
        pair, sprite, orientation, axis = spec["pair"], spec["sprite"], spec["orientation"], spec["axis"]
        comps = components_of_color(inp, color)
        sizes = sorted(len(comp) for comp in comps)
        if sizes != [1, 1, len(sprite)]:
            return False
        try:
            parsed_pair, parsed_sprite, parsed_orient, parsed_axis = _parse_color(inp, color)
        except AssertionError:
            return False
        if parsed_pair != pair or parsed_sprite != sprite:
            return False
        if parsed_orient != orientation or parsed_axis != axis:
            return False

        added = {
            (int(r), int(c))
            for r, c in np.argwhere((expected == color) & (inp != color))
        }
        if added != spec["mirror"]:
            return False

        for other in specs:
            if other is spec:
                continue
            alt = reflect_cells(sprite, other["orientation"], other["axis"])
            if alt == spec["mirror"]:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(PALETTE, color_count)
    orientations = _sample_orientations(color_count)

    used_axes: set[tuple[str, int]] = set()
    blocked_cells: set[tuple[int, int]] = set()
    blocked_pad: set[tuple[int, int]] = set()
    specs: list[dict] = []

    order = list(range(color_count))
    random.shuffle(order)
    for idx in order:
        spec = _candidate_spec(height, width, colors[idx], orientations[idx], used_axes, blocked_pad)
        if spec is None:
            return None
        specs.append(spec)
        used_axes.add((spec["orientation"], spec["axis"]))
        blocked_cells |= spec["pair"] | spec["sprite"] | spec["mirror"]
        blocked_pad = _all_neighbors8(blocked_cells)

    inp = _render_input(height, width, specs)
    expected = _render_expected(inp, specs)
    if not _validate_semantics(inp, specs, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis reflection input")
