from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _polyline_points(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        seg = _segment_points(vertices[i], vertices[i + 1])
        if i:
            seg = seg[1:]
        points.extend(seg)
    return points


def _reconstruct_path_mask(grid: np.ndarray, color: int) -> tuple[np.ndarray, set[int]]:
    h, w = grid.shape
    visible = grid == color
    reconstructed = visible.copy()
    hidden_under: set[int] = set()

    for r in range(h):
        for c in range(w):
            if visible[r, c]:
                continue
            if grid[r, c] == BG:
                continue

            hidden = False
            if 0 < r < h - 1 and int(grid[r - 1, c]) == color and int(grid[r + 1, c]) == color:
                hidden = True
            if 0 < c < w - 1 and int(grid[r, c - 1]) == color and int(grid[r, c + 1]) == color:
                hidden = True

            if hidden:
                reconstructed[r, c] = True
                hidden_under.add(int(grid[r, c]))

    return reconstructed, hidden_under


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the crossing stack order of colored border-to-border paths and keep only the unique topmost path.

    Input structure:
    - The background is black.
    - Each non-black color denotes one single-pixel-wide 4-connected path whose two endpoints lie on the grid border.
    - Paths may cross orthogonally, and at each crossing only the upper path's color is visible at the shared cell.

    Rule:
    - Reconstruct each path through hidden crossings by detecting cells where the same color continues straight on opposite sides of another visible color.
    - Every reconstructed hidden crossing gives a layering relation: the reconstructed color is below the visible color at that cell.
    - Find the unique path color that is never below any other path, then output only that path on a black canvas.

    Key insight:
    The geometry of a hidden lower path is recoverable locally from straight opposite-side continuations, and those local recoveries are enough to identify the one path that is never covered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one path"

    if len(colors) == 1:
        return g.copy()

    reconstructed: dict[int, np.ndarray] = {}
    hidden_under: dict[int, set[int]] = {}
    for color in colors:
        reconstructed[color], hidden_under[color] = _reconstruct_path_mask(g, color)

    top_candidates = [color for color in colors if not hidden_under[color]]
    assert len(top_candidates) == 1, "expected a unique topmost path"
    top_color = top_candidates[0]

    out = np.zeros_like(g)
    out[reconstructed[top_color]] = top_color
    return out


def _sample_spaced(
    rng: np.random.Generator, candidates: list[int], count: int, min_gap: int = 2
) -> list[int] | None:
    if count == 0:
        return []
    if count > len(candidates):
        return None

    for _ in range(200):
        chosen = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _render_paths(
    shape: tuple[int, int], paths_in_z_order: list[dict[str, object]]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for path in paths_in_z_order:
        coords = np.asarray(path["points"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(path["color"])
    return grid


def _top_mask(shape: tuple[int, int], points: list[tuple[int, int]], color: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=int)
    coords = np.asarray(points, dtype=int)
    mask[coords[:, 0], coords[:, 1]] = color
    return mask


def _gen_straight_horizontal(
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    top_row = int(rng.integers(3, h - 3))

    candidates = list(range(2, w - 2))
    max_lower = min(5, len(candidates))
    if max_lower < 2:
        return None

    lower_count = int(rng.integers(2, max_lower + 1))
    cols = _sample_spaced(rng, candidates, lower_count, min_gap=2)
    if cols is None:
        return None

    lowers = [_polyline_points([(0, c), (h - 1, c)]) for c in cols]
    top = _polyline_points([(top_row, 0), (top_row, w - 1)])
    return (h, w), lowers, top


def _gen_straight_vertical(
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    top_col = int(rng.integers(3, w - 3))

    candidates = list(range(2, h - 2))
    max_lower = min(5, len(candidates))
    if max_lower < 2:
        return None

    lower_count = int(rng.integers(2, max_lower + 1))
    rows = _sample_spaced(rng, candidates, lower_count, min_gap=2)
    if rows is None:
        return None

    lowers = [_polyline_points([(r, 0), (r, w - 1)]) for r in rows]
    top = _polyline_points([(0, top_col), (h - 1, top_col)])
    return (h, w), lowers, top


def _gen_l_shape(
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    h = int(rng.integers(15, 25))
    w = int(rng.integers(15, 25))
    bend_r = int(rng.integers(5, h - 5))
    bend_c = int(rng.integers(5, w - 5))
    quadrant = str(rng.choice(["left_top", "left_bottom", "right_top", "right_bottom"]))

    if quadrant == "left_top":
        top = _polyline_points([(bend_r, 0), (bend_r, bend_c), (0, bend_c)])
        vertical_candidates = list(range(1, bend_c - 1))
        horizontal_candidates = list(range(1, bend_r - 1))
    elif quadrant == "left_bottom":
        top = _polyline_points([(bend_r, 0), (bend_r, bend_c), (h - 1, bend_c)])
        vertical_candidates = list(range(1, bend_c - 1))
        horizontal_candidates = list(range(bend_r + 2, h - 1))
    elif quadrant == "right_top":
        top = _polyline_points([(bend_r, w - 1), (bend_r, bend_c), (0, bend_c)])
        vertical_candidates = list(range(bend_c + 2, w - 1))
        horizontal_candidates = list(range(1, bend_r - 1))
    else:
        top = _polyline_points([(bend_r, w - 1), (bend_r, bend_c), (h - 1, bend_c)])
        vertical_candidates = list(range(bend_c + 2, w - 1))
        horizontal_candidates = list(range(bend_r + 2, h - 1))

    modes: list[str] = []
    if len(vertical_candidates) >= 2:
        modes.append("vertical_only")
    if len(horizontal_candidates) >= 2:
        modes.append("horizontal_only")
    if vertical_candidates and horizontal_candidates:
        modes.append("both")
    if not modes:
        return None

    mode = str(rng.choice(modes))
    verticals: list[int] = []
    horizontals: list[int] = []

    if mode == "vertical_only":
        count = int(rng.integers(2, min(4, len(vertical_candidates)) + 1))
        sample = _sample_spaced(rng, vertical_candidates, count, min_gap=2)
        if sample is None:
            return None
        verticals = sample
    elif mode == "horizontal_only":
        count = int(rng.integers(2, min(4, len(horizontal_candidates)) + 1))
        sample = _sample_spaced(rng, horizontal_candidates, count, min_gap=2)
        if sample is None:
            return None
        horizontals = sample
    else:
        v_count = int(rng.integers(1, min(2, len(vertical_candidates)) + 1))
        h_count = int(rng.integers(1, min(2, len(horizontal_candidates)) + 1))
        v_sample = _sample_spaced(rng, vertical_candidates, v_count, min_gap=2)
        h_sample = _sample_spaced(rng, horizontal_candidates, h_count, min_gap=2)
        if v_sample is None or h_sample is None:
            return None
        verticals = v_sample
        horizontals = h_sample

    lowers = [_polyline_points([(0, c), (h - 1, c)]) for c in verticals]
    lowers.extend(_polyline_points([(r, 0), (r, w - 1)]) for r in horizontals)
    return (h, w), lowers, top


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    branch = str(rng.choice(["straight_h", "straight_v", "l_shape"]))
    if branch == "straight_h":
        scene = _gen_straight_horizontal(rng)
    elif branch == "straight_v":
        scene = _gen_straight_vertical(rng)
    else:
        scene = _gen_l_shape(rng)
    if scene is None:
        return None

    shape, lower_geometries, top_geometry = scene
    if len(lower_geometries) < 2:
        return None

    lower_geometries = list(lower_geometries)
    rng.shuffle(lower_geometries)

    path_count = len(lower_geometries) + 1
    colors = [int(c) for c in rng.choice(PALETTE, size=path_count, replace=False)]

    paths: list[dict[str, object]] = []
    for color, points in zip(colors[:-1], lower_geometries):
        paths.append({"color": color, "points": points})
    top_color = colors[-1]
    paths.append({"color": top_color, "points": top_geometry})

    grid = _render_paths(shape, paths)
    expected = _top_mask(shape, top_geometry, top_color)

    if not np.array_equal(main(grid), expected):
        return None
    if len(np.unique(grid)) != path_count + 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-stacking ARC input")
