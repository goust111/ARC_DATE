import numpy as np


RNG = np.random.default_rng()
BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_GAP = 2


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_rectangular_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    expected_area = 2 * (h + w) - 4
    if h < 3 or w < 3 or int(comp["area"]) != expected_area:
        return False

    mask = np.zeros((h, w), dtype=bool)
    local = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_rings(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    comps = _find_components(grid, bg)
    rings = [comp for comp in comps if _is_rectangular_ring(comp)]
    assert comps and len(rings) == len(comps), "expected only rectangular ring components"
    for ring in rings:
        ring["depth"] = sum(
            1 for other in rings if _strictly_contains(other["bbox"], ring["bbox"])
        )
    return bg, rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign nested rectangular-ring colors by global color area rank.

    Rings keep their geometry. Colors are sorted by their total pixel counts
    across the whole grid, then mapped from outermost layer to innermost layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, rings = _extract_rings(g)
    colors = sorted({int(ring["color"]) for ring in rings})
    color_counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    counts = list(color_counts.values())
    assert len(set(counts)) == len(counts), "color areas must be uniquely ranked"

    max_depth = max(int(ring["depth"]) for ring in rings)
    assert len(colors) == max_depth + 1, "one shared color set is expected across depths"
    sorted_colors = sorted(colors, key=lambda color: color_counts[color], reverse=True)

    out = np.full_like(g, bg)
    for ring in rings:
        _paint_ring(out, ring["bbox"], sorted_colors[int(ring["depth"])])
    return out


def _layer_area(outer_h: int, outer_w: int, depth: int) -> int:
    h = int(outer_h - 4 * depth)
    w = int(outer_w - 4 * depth)
    return 2 * (h + w) - 4


def _sample_group_size(depth_count: int) -> tuple[int, int]:
    min_outer = 4 * (depth_count - 1) + 3
    if depth_count == 3:
        outer_h = int(RNG.integers(min_outer, min_outer + 5))
        outer_w = int(RNG.integers(min_outer, min_outer + 7))
    else:
        outer_h = int(RNG.integers(min_outer, min_outer + 4))
        outer_w = int(RNG.integers(min_outer, min_outer + 4))
    return outer_h, outer_w


def _choose_assignments(colors: list[int], group_sizes: list[tuple[int, int]]) -> list[list[int]] | None:
    depth_count = len(colors)
    layer_areas = [
        [_layer_area(outer_h, outer_w, depth) for depth in range(depth_count)]
        for outer_h, outer_w in group_sizes
    ]
    for _ in range(400):
        assignments = [list(RNG.permutation(colors)) for _ in group_sizes]
        totals = {color: 0 for color in colors}
        for group_index, group_colors in enumerate(assignments):
            for depth, color in enumerate(group_colors):
                totals[int(color)] += int(layer_areas[group_index][depth])
        if len(set(totals.values())) != depth_count:
            continue
        sorted_colors = sorted(colors, key=lambda color: totals[color], reverse=True)
        if any(
            int(group_colors[depth]) != int(sorted_colors[depth])
            for group_colors in assignments
            for depth in range(depth_count)
        ):
            return assignments
    return None


def _place_groups(group_sizes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    top = int(RNG.integers(1, 4))
    bottom = int(RNG.integers(1, 4))
    left = int(RNG.integers(1, 4))
    right = int(RNG.integers(1, 4))
    gap = int(RNG.integers(PLACEMENT_GAP, PLACEMENT_GAP + 3))
    layout = str(RNG.choice(["horizontal", "vertical"]))

    heights = [h for h, _w in group_sizes]
    widths = [w for _h, w in group_sizes]
    positions: list[tuple[int, int]] = []

    if layout == "horizontal":
        cross_slack = int(RNG.integers(0, 5))
        canvas_h = int(max(heights) + top + bottom + cross_slack)
        canvas_w = int(sum(widths) + gap * (len(group_sizes) - 1) + left + right)
        c = left
        for h, w in group_sizes:
            max_r = int(canvas_h - bottom - h)
            r = int(RNG.integers(top, max_r + 1))
            positions.append((r, c))
            c += int(w + gap)
        return canvas_h, canvas_w, positions

    cross_slack = int(RNG.integers(0, 5))
    canvas_h = int(sum(heights) + gap * (len(group_sizes) - 1) + top + bottom)
    canvas_w = int(max(widths) + left + right + cross_slack)
    r = top
    for h, w in group_sizes:
        max_c = int(canvas_w - right - w)
        c = int(RNG.integers(left, max_c + 1))
        positions.append((r, c))
        r += int(h + gap)
    return canvas_h, canvas_w, positions


def _draw_group(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    colors_by_depth: list[int],
) -> None:
    for depth, color in enumerate(colors_by_depth):
        r0 = int(top + 2 * depth)
        c0 = int(left + 2 * depth)
        r1 = int(top + outer_h - 1 - 2 * depth)
        c1 = int(left + outer_w - 1 - 2 * depth)
        _paint_ring(grid, (r0, c0, r1, c1), int(color))


def _gen_attempt() -> np.ndarray | None:
    depth_count = int(RNG.choice([3, 4]))
    group_count = int(RNG.choice([2, 3])) if depth_count == 3 else 2
    colors = list(map(int, RNG.choice(PALETTE, size=depth_count, replace=False)))
    group_sizes = [_sample_group_size(depth_count) for _ in range(group_count)]
    assignments = _choose_assignments(colors, group_sizes)
    if assignments is None:
        return None

    placement = _place_groups(group_sizes)
    if placement is None:
        return None
    canvas_h, canvas_w, positions = placement

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for (outer_h, outer_w), (top, left), group_colors in zip(group_sizes, positions, assignments):
        _draw_group(grid, top, left, outer_h, outer_w, group_colors)

    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
