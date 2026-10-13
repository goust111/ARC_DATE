import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
BG = 0
COLORS = tuple(range(1, 10))
GRID_MIN = 12
GRID_MAX = 30


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (
        or0 <= ir0
        and oc0 <= ic0
        and or1 >= ir1
        and oc1 >= ic1
        and outer != inner
    )


def _classify_component(points: list[tuple[int, int]], box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = h * w
    point_set = set(points)

    if len(points) == area:
        return "fill"

    if h >= 3 and w >= 3 and len(points) == 2 * h + 2 * w - 4:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if ((r, c) in point_set) != on_border:
                    raise AssertionError("component bbox boundary does not match a full ring")
        return "ring"

    raise AssertionError("component must be either a full ring or a solid rectangle")


def _render_piece(
    box: tuple[int, int, int, int],
    points: list[tuple[int, int]],
    color: int,
    bg: int,
) -> np.ndarray:
    r0, c0, r1, c1 = box
    piece = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in points:
        piece[r - r0, c - c0] = color
    return piece


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose nested rectangular rings and lay the layers out left-to-right by depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _components(g != bg)
    assert comps, "expected at least one non-background component"

    layers = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1, "each layer must be monochromatic"
        color = next(iter(colors))
        box = _bbox(pts)
        kind = _classify_component(pts, box)
        layers.append({"points": pts, "box": box, "color": color, "kind": kind})

    for layer in layers:
        layer["depth"] = 1 + sum(_contains(other["box"], layer["box"]) for other in layers)

    layers.sort(key=lambda layer: int(layer["depth"]))
    depths = [int(layer["depth"]) for layer in layers]
    assert depths == list(range(1, len(layers) + 1)), "layers must form one nested depth chain"
    assert layers[0]["kind"] == "ring", "outermost layer must be a ring"
    for layer in layers[:-1]:
        assert layer["kind"] == "ring", "all non-deepest layers must be rings"

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["box"]
        ir0, ic0, ir1, ic1 = inner["box"]
        assert (ir0 - or0, ic0 - oc0, or1 - ir1, oc1 - ic1) == (2, 2, 2, 2), (
            "adjacent layers must be separated by a one-cell background gap"
        )

    outer_h = layers[0]["box"][2] - layers[0]["box"][0] + 1
    piece_widths = [layer["box"][3] - layer["box"][1] + 1 for layer in layers]
    out_w = sum(piece_widths) + len(piece_widths) - 1
    out = np.full((outer_h, out_w), bg, dtype=int)

    col = 0
    for layer in layers:
        piece = _render_piece(layer["box"], layer["points"], int(layer["color"]), bg)
        ph, pw = piece.shape
        row = (outer_h - ph) // 2
        out[row : row + ph, col : col + pw] = piece
        col += pw + 1

    return out


def _draw_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _sample_layout() -> tuple[int, bool, int, int, int, int] | None:
    ring_count = random.randint(2, 5)
    has_fill = random.random() < 0.5

    if has_fill:
        max_fill_h = GRID_MAX - 4 * ring_count - 2
        max_fill_w = GRID_MAX - 4 * ring_count - 2
        if max_fill_h < 1 or max_fill_w < 1:
            return None
        fill_h = random.randint(1, min(8, max_fill_h))
        fill_w = random.randint(1, min(8, max_fill_w))
        outer_h = fill_h + 4 * ring_count
        outer_w = fill_w + 4 * ring_count
        return ring_count, has_fill, outer_h, outer_w, fill_h, fill_w

    max_inner_h = GRID_MAX - 4 * (ring_count - 1) - 2
    max_inner_w = GRID_MAX - 4 * (ring_count - 1) - 2
    if max_inner_h < 3 or max_inner_w < 3:
        return None
    inner_h = random.randint(3, min(10, max_inner_h))
    inner_w = random.randint(3, min(10, max_inner_w))
    outer_h = inner_h + 4 * (ring_count - 1)
    outer_w = inner_w + 4 * (ring_count - 1)
    return ring_count, has_fill, outer_h, outer_w, inner_h, inner_w


def _near_center_offset(total: int, size: int) -> int:
    extra = total - size
    center = extra // 2
    delta = random.randint(-min(2, center), min(2, extra - center))
    return center + delta


def _build_input(
    ring_count: int,
    has_fill: bool,
    outer_h: int,
    outer_w: int,
    inner_h: int,
    inner_w: int,
) -> np.ndarray:
    grid_h = random.randint(max(GRID_MIN, outer_h + 2), min(GRID_MAX, outer_h + 8))
    grid_w = random.randint(max(GRID_MIN, outer_w + 2), min(GRID_MAX, outer_w + 8))
    top = _near_center_offset(grid_h, outer_h)
    left = _near_center_offset(grid_w, outer_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    color_count = ring_count + int(has_fill)
    layer_colors = random.sample(COLORS, color_count)

    for depth in range(ring_count):
        layer_h = outer_h - 4 * depth
        layer_w = outer_w - 4 * depth
        layer_top = top + 2 * depth
        layer_left = left + 2 * depth
        _draw_ring(grid, layer_top, layer_left, layer_h, layer_w, layer_colors[depth])

    if has_fill:
        fill_top = top + 2 * ring_count
        fill_left = left + 2 * ring_count
        grid[fill_top : fill_top + inner_h, fill_left : fill_left + inner_w] = layer_colors[-1]

    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if int((grid == BG).sum()) <= int((grid != BG).sum()):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2


def _gen_attempt() -> np.ndarray | None:
    params = _sample_layout()
    if params is None:
        return None
    grid = _build_input(*params)
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
