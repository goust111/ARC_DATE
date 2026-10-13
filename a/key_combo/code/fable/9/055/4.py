from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 600
_SEEN_OUTPUT_KEYS: set[tuple[tuple[int, int], tuple[int, ...]]] = set()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _analyze_ring(grid: np.ndarray, component: dict[str, object], bg: int) -> dict[str, int]:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w, "every layer must be a square ring"
    side = side_h
    mask = np.zeros((side, side), dtype=bool)
    coords = np.asarray(component["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    thickness = 0
    while thickness < side and bool(np.all(mask[thickness, :])):
        thickness += 1
    assert thickness > 0 and 2 * thickness < side, "input layers must be hollow square rings"
    assert bool(np.all(mask[-thickness:, :])), "bottom band must match ring thickness"
    assert bool(np.all(mask[:, :thickness])), "left band must match ring thickness"
    assert bool(np.all(mask[:, -thickness:])), "right band must match ring thickness"

    expected = np.ones((side, side), dtype=bool)
    expected[thickness : side - thickness, thickness : side - thickness] = False
    assert np.array_equal(mask, expected), "component must be a uniform square ring"

    return {"color": color, "side": side, "thickness": thickness}


def _group_rings(grid: np.ndarray) -> list[list[dict[str, int]]]:
    bg = get_bg(grid)
    components = find_components(grid, bg)
    assert components, "input must contain foreground rings"

    grouped: dict[tuple[int, int], list[dict[str, int]]] = {}
    for comp in components:
        ring = _analyze_ring(grid, comp, bg)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        center_key = (r0 + r1, c0 + c1)
        ring["r0"] = r0
        ring["c0"] = c0
        ring["r1"] = r1
        ring["c1"] = c1
        grouped.setdefault(center_key, []).append(ring)

    assert len(grouped) == 2, "input must contain exactly two nested-ring groups"
    groups = []
    for rings in grouped.values():
        rings.sort(key=lambda ring: int(ring["side"]), reverse=True)
        assert len(rings) >= 2, "each group must have multiple layers"
        for outer, inner in zip(rings, rings[1:]):
            hole_side = int(outer["side"]) - 2 * int(outer["thickness"])
            assert int(inner["side"]) < hole_side, "nested rings must be separated by background"
        groups.append(rings)
    return groups


def _classify_groups(groups: list[list[dict[str, int]]]) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    color_groups = []
    thickness_groups = []
    for rings in groups:
        colors = [int(ring["color"]) for ring in rings]
        thicknesses = [int(ring["thickness"]) for ring in rings]
        if all(t == 1 for t in thicknesses) and len(set(colors)) == len(colors):
            color_groups.append(rings)
        if len(set(colors)) == 1 and len(set(thicknesses)) >= 2:
            thickness_groups.append(rings)

    assert len(color_groups) == 1, "color provider must be unique"
    assert len(thickness_groups) == 1, "thickness provider must be unique"
    color_group = color_groups[0]
    thickness_group = thickness_groups[0]
    assert len(color_group) == len(thickness_group), "the two groups must have the same layer count"
    return color_group, thickness_group


def _render_output(colors: list[int], thicknesses: list[int], bg: int) -> np.ndarray:
    side = 2 * sum(int(t) for t in thicknesses)
    out = np.full((side, side), bg, dtype=int)
    inset = 0
    for color, thickness in zip(colors, thicknesses):
        out[inset : side - inset, inset : side - inset] = int(color)
        inset += int(thickness)
    assert inset * 2 == side
    return out


def _parse_input(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    groups = _group_rings(np.asarray(grid, dtype=int))
    color_group, thickness_group = _classify_groups(groups)
    colors = [int(ring["color"]) for ring in color_group]
    thicknesses = [int(ring["thickness"]) for ring in thickness_group]
    return colors, thicknesses, get_bg(np.asarray(grid, dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recombine one nested square-ring object's color order with the other nested square-ring object's thickness order.

    Input structure:
    - The grid has a black background and exactly two non-touching concentric nested square-ring groups with the same layer count.
    - One group has one-cell-thick rings whose colors are all different.
    - The other group uses one color for every ring, while ring thickness varies by layer.

    Rule:
    - Identify the color-provider group and read its ring colors from outermost to innermost.
    - Identify the thickness-provider group and read its ring thicknesses from outermost to innermost.
    - Output a square of side length twice the thickness sum, using the i-th color with the i-th thickness as nested filled square bands until the innermost band fills the center.

    Key insight:
    - The two visible ring stacks each encode only half of the answer: one supplies layer order by color, the other supplies layer order by thickness, and the solution is their layerwise recombination.
    """
    colors, thicknesses, bg = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(colors, thicknesses, bg)


def _stack_sides(thicknesses: list[int], gaps: list[int], inner_hole: int) -> list[int]:
    n = len(thicknesses)
    sides = [0] * n
    hole = int(inner_hole)
    for idx in range(n - 1, -1, -1):
        side = hole + 2 * int(thicknesses[idx])
        sides[idx] = side
        if idx > 0:
            hole = side + 2 * int(gaps[idx - 1])
    return sides


def _render_ring_stack(sides: list[int], thicknesses: list[int], colors: list[int], bg: int) -> np.ndarray:
    outer_side = int(sides[0])
    patch = np.full((outer_side, outer_side), bg, dtype=int)
    for side, thickness, color in zip(sides, thicknesses, colors):
        offset = (outer_side - int(side)) // 2
        inner_start = offset + int(thickness)
        inner_end = offset + int(side) - int(thickness)
        patch[offset : offset + int(side), offset : offset + int(side)] = int(color)
        patch[inner_start:inner_end, inner_start:inner_end] = bg
    return patch


def _sample_thicknesses(layer_count: int) -> list[int]:
    high = 4 if layer_count == 2 else 3
    max_sum = 6 if layer_count == 2 else 5
    while True:
        thicknesses = [random.randint(1, high) for _ in range(layer_count)]
        if len(set(thicknesses)) < 2:
            continue
        if max(thicknesses) == 1:
            continue
        if sum(thicknesses) > max_sum:
            continue
        return thicknesses


def _sample_stack_geometry(thicknesses: list[int], outer_limit: int, gap_hi: int) -> tuple[list[int], list[int], int]:
    layer_count = len(thicknesses)
    for _ in range(200):
        gaps = [random.randint(1, gap_hi) for _ in range(layer_count - 1)]
        inner_hole = random.randint(1, 4)
        sides = _stack_sides(thicknesses, gaps, inner_hole)
        if sides[0] <= outer_limit:
            return sides, gaps, inner_hole
    raise RuntimeError("failed to sample stack geometry")


def _place_two_patches(a: np.ndarray, b: np.ndarray, bg: int) -> np.ndarray:
    ha, wa = map(int, a.shape)
    hb, wb = map(int, b.shape)
    layouts = []

    for first, second in ((a, b), (b, a)):
        h1, w1 = map(int, first.shape)
        h2, w2 = map(int, second.shape)
        for blank in range(1, 4):
            min_h = max(h1, h2) + 2
            min_w = w1 + w2 + blank + 2
            if min_h <= 30 and min_w <= 30:
                layouts.append(("horizontal", first, second, blank))
            min_h = h1 + h2 + blank + 2
            min_w = max(w1, w2) + 2
            if min_h <= 30 and min_w <= 30:
                layouts.append(("vertical", first, second, blank))

    assert layouts, "two ring groups must fit within ARC bounds"
    orientation, first, second, blank = random.choice(layouts)
    h1, w1 = map(int, first.shape)
    h2, w2 = map(int, second.shape)

    if orientation == "horizontal":
        min_h = max(h1, h2) + 2
        min_w = w1 + w2 + blank + 2
        h = min_h + random.randint(0, min(4, 30 - min_h))
        w = min_w + random.randint(0, min(5, 30 - min_w))
        grid = np.full((h, w), bg, dtype=int)
        c1 = random.randint(1, w - (w1 + blank + w2) - 1)
        c2 = random.randint(c1 + w1 + blank, w - w2 - 1)
        r1 = random.randint(1, h - h1 - 1)
        r2 = random.randint(1, h - h2 - 1)
        grid[r1 : r1 + h1, c1 : c1 + w1] = first
        grid[r2 : r2 + h2, c2 : c2 + w2] = second
        return grid

    min_h = h1 + h2 + blank + 2
    min_w = max(w1, w2) + 2
    h = min_h + random.randint(0, min(5, 30 - min_h))
    w = min_w + random.randint(0, min(4, 30 - min_w))
    grid = np.full((h, w), bg, dtype=int)
    r1 = random.randint(1, h - (h1 + blank + h2) - 1)
    r2 = random.randint(r1 + h1 + blank, h - h2 - 1)
    c1 = random.randint(1, w - w1 - 1)
    c2 = random.randint(1, w - w2 - 1)
    grid[r1 : r1 + h1, c1 : c1 + w1] = first
    grid[r2 : r2 + h2, c2 : c2 + w2] = second
    return grid


def _wrong_shortcuts_differ(colors: list[int], thicknesses: list[int], lure_color: int, bg: int, output: np.ndarray) -> bool:
    only_colors = _render_output(colors, [1] * len(colors), bg)
    only_thickness = _render_output([lure_color] * len(thicknesses), thicknesses, bg)
    swapped = _render_output([lure_color] * len(colors), [1] * len(colors), bg)
    return (
        not np.array_equal(output, only_colors)
        and not np.array_equal(output, only_thickness)
        and not np.array_equal(output, swapped)
    )


def _output_key(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, grid.shape)), tuple(map(int, grid.flatten()))


def _generate_attempt() -> np.ndarray | None:
    layer_count = random.choices([2, 3], weights=[4, 6], k=1)[0]
    colors = random.sample(PALETTE, layer_count)
    thicknesses = _sample_thicknesses(layer_count)
    output = _render_output(colors, thicknesses, BACKGROUND)
    if _output_key(output) in _SEEN_OUTPUT_KEYS:
        return None

    if random.random() < 0.5:
        lure_color = random.choice(colors)
    else:
        lure_color = random.choice([c for c in PALETTE if c not in colors])

    color_sides, _, _ = _sample_stack_geometry([1] * layer_count, outer_limit=12, gap_hi=2)
    thick_sides, _, _ = _sample_stack_geometry(thicknesses, outer_limit=15, gap_hi=2)
    color_patch = _render_ring_stack(color_sides, [1] * layer_count, colors, BACKGROUND)
    thick_patch = _render_ring_stack(thick_sides, thicknesses, [lure_color] * layer_count, BACKGROUND)

    inp = _place_two_patches(color_patch, thick_patch, BACKGROUND)

    try:
        parsed_colors, parsed_thicknesses, bg = _parse_input(inp)
    except Exception:
        return None

    if parsed_colors != colors or parsed_thicknesses != thicknesses or bg != BACKGROUND:
        return None
    if not _wrong_shortcuts_differ(colors, thicknesses, lure_color, BACKGROUND, output):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, output):
        return None

    _SEEN_OUTPUT_KEYS.add(_output_key(output))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid double-ring-fusion input")
