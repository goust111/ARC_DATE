from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _cross_cells(center_r: int, center_c: int, arm: int) -> set[tuple[int, int]]:
    cells = {(int(center_r), int(c)) for c in range(int(center_c - arm), int(center_c + arm + 1))}
    cells.update((int(r), int(center_c)) for r in range(int(center_r - arm), int(center_r + arm + 1)))
    return cells


def _infer_cross(coords: np.ndarray) -> dict[str, object]:
    pts = [tuple(map(int, rc)) for rc in coords]
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    candidates: list[tuple[int, int, int]] = []

    for center_r in rows:
        for center_c in cols:
            if all(r == center_r or c == center_c for r, c in pts):
                arm = max(max(abs(r - center_r), abs(c - center_c)) for r, c in pts)
                candidates.append((int(center_r), int(center_c), int(arm)))

    unique = sorted(set(candidates))
    assert len(unique) == 1, "each color must determine one unique cross"
    center_r, center_c, arm = unique[0]
    shape = _cross_cells(center_r, center_c, arm)
    assert all((int(r), int(c)) in shape for r, c in pts)
    return {
        "center": (int(center_r), int(center_c)),
        "arm": int(arm),
        "visible": set(pts),
        "shape": shape,
        "visible_area": len(pts),
    }


def _extract_crosses(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    crosses: dict[int, dict[str, object]] = {}
    for color in colors:
        coords = np.argwhere(grid == int(color))
        crosses[int(color)] = _infer_cross(coords)
    return int(bg), crosses


def _build_occlusion_graph(
    grid: np.ndarray,
    crosses: dict[int, dict[str, object]],
) -> dict[int, set[int]]:
    colors = list(crosses)
    graph = {int(color): set() for color in colors}
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = crosses[color_a]["shape"] & crosses[color_b]["shape"]
            if not overlap:
                continue

            a_visible = any(int(grid[r, c]) == int(color_a) for r, c in overlap)
            b_visible = any(int(grid[r, c]) == int(color_b) for r, c in overlap)
            assert not (a_visible and b_visible), "pairwise overlap cannot show both colors"
            if a_visible:
                graph[int(color_a)].add(int(color_b))
            elif b_visible:
                graph[int(color_b)].add(int(color_a))
    return graph


def _topological_front_to_back(
    crosses: dict[int, dict[str, object]],
    graph: dict[int, set[int]],
) -> list[int]:
    indegree = {int(color): 0 for color in crosses}
    for src, dsts in graph.items():
        for dst in dsts:
            indegree[int(dst)] += 1

    order: list[int] = []
    remaining = set(crosses)
    while remaining:
        zeros = [color for color in remaining if indegree[color] == 0]
        assert zeros, "occlusion graph must be acyclic"
        zeros.sort(
            key=lambda color: (
                -len(graph[color]),
                -int(crosses[color]["visible_area"]),
                int(crosses[color]["center"][0]),
                int(crosses[color]["center"][1]),
                int(crosses[color]["arm"]),
            )
        )
        chosen = int(zeros[0])
        order.append(chosen)
        remaining.remove(chosen)
        for dst in graph[chosen]:
            indegree[int(dst)] -= 1
    return order


def _draw_cross(grid: np.ndarray, center_r: int, center_c: int, arm: int, color: int) -> None:
    grid[int(center_r), int(center_c - arm) : int(center_c + arm + 1)] = int(color)
    grid[int(center_r - arm) : int(center_r + arm + 1), int(center_c)] = int(color)


def _render_output(front_to_back: list[int]) -> np.ndarray:
    n = len(front_to_back)
    mid = n - 1
    out = np.zeros((2 * n - 1, 2 * n - 1), dtype=int)
    for depth, color in enumerate(reversed(front_to_back)):
        arm = n - 1 - depth
        _draw_cross(out, mid, mid, arm, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the crosses' occlusion order and redraw it as concentric cross rings.

    Input structure:
    - The background is black.
    - Several different-colored plus-shaped crosses overlap on the grid.
    - Each cross consists of one center pixel plus four 1-pixel-wide arms of equal length.
    - Different colors may have different arm lengths, and visible pixels are sufficient to infer each full cross.

    Rule:
    - For each non-background color, infer the full cross by finding the unique row/column intersection that explains all visible pixels, then set the arm length to the farthest visible axis distance from that center.
    - Compare every pair of reconstructed crosses: if their full shapes overlap and an overlap cell currently shows color A, then A occludes the other color and must be in front of it.
    - Topologically sort this occlusion graph from front to back.
    - Output a square of size 2n-1 for n colors, and paint centered crosses from back to front with arm lengths n-1, n-2, ..., 0 so the result becomes concentric cross rings.

    Key insight:
    - The input geometry is only evidence for depth order: after reconstructing the hidden full crosses, the real task is to read the occlusion layering and convert that order into a layout-independent nested-cross rendering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, crosses = _extract_crosses(g)
    graph = _build_occlusion_graph(g, crosses)
    front_to_back = _topological_front_to_back(crosses, graph)
    return _render_output(front_to_back)


COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
COUNT_CHOICES = [2, 3, 4, 5]
ARM_MIN = 1
ARM_MAX = 4
MAX_GEN_ATTEMPTS = 400


def _chain_centers(arms: list[int]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    rows = [0]
    cols = [0]
    for i in range(1, len(arms)):
        rows.append(int(rows[-1] + arms[i]))
        cols.append(int(cols[-1] + arms[i - 1]))

    centers = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False)]
    min_r = min(r - arm for (r, _), arm in zip(centers, arms, strict=False))
    max_r = max(r + arm for (r, _), arm in zip(centers, arms, strict=False))
    min_c = min(c - arm for (_, c), arm in zip(centers, arms, strict=False))
    max_c = max(c + arm for (_, c), arm in zip(centers, arms, strict=False))
    return centers, (int(min_r), int(max_r), int(min_c), int(max_c))


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    transformed = np.array(grid, copy=True)
    k = random.randrange(4)
    if k:
        transformed = np.rot90(transformed, k)
    if random.choice([False, True]):
        transformed = np.fliplr(transformed)
    return np.asarray(transformed, dtype=int)


def _build_scene(front_to_back: list[int], arms: list[int]) -> np.ndarray:
    centers, (min_r, max_r, min_c, max_c) = _chain_centers(arms)
    scene_h = max_r - min_r + 1
    scene_w = max_c - min_c + 1
    extra_h = random.randint(0, max(0, 30 - scene_h))
    extra_w = random.randint(0, max(0, 30 - scene_w))
    grid_h = scene_h + extra_h
    grid_w = scene_w + extra_w
    shift_r = -min_r + random.randint(0, extra_h)
    shift_c = -min_c + random.randint(0, extra_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for _color, (center_r, center_c), arm in zip(front_to_back, centers, arms, strict=False):
        assert 0 <= center_r + shift_r - arm
        assert center_r + shift_r + arm < grid_h
        assert 0 <= center_c + shift_c - arm
        assert center_c + shift_c + arm < grid_w

    for idx in range(len(front_to_back) - 1, -1, -1):
        center_r, center_c = centers[idx]
        _draw_cross(
            grid,
            center_r + shift_r,
            center_c + shift_c,
            arms[idx],
            int(front_to_back[idx]),
        )

    return _transform_grid(grid)


def _validate_generated_scene(grid: np.ndarray, front_to_back: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg, crosses = _extract_crosses(grid)
    if bg != 0 or set(crosses) != set(front_to_back):
        return False

    graph = _build_occlusion_graph(grid, crosses)
    edge_count = sum(len(dsts) for dsts in graph.values())
    if edge_count != len(front_to_back) - 1:
        return False
    for i in range(len(front_to_back) - 1):
        if int(front_to_back[i + 1]) not in graph[int(front_to_back[i])]:
            return False

    recovered = _topological_front_to_back(crosses, graph)
    if recovered != [int(color) for color in front_to_back]:
        return False

    expected = _render_output([int(color) for color in front_to_back])
    return bool(np.array_equal(main(grid), expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        n = random.choice(COUNT_CHOICES)
        front_to_back = random.sample(COLOR_POOL, n)
        arms = [random.randint(ARM_MIN, ARM_MAX) for _ in range(n)]

        centers, (min_r, max_r, min_c, max_c) = _chain_centers(arms)
        scene_h = max_r - min_r + 1
        scene_w = max_c - min_c + 1
        if scene_h > 30 or scene_w > 30:
            continue
        if max(scene_h, scene_w) < 5:
            continue

        grid = _build_scene(front_to_back, arms)
        if _validate_generated_scene(grid, front_to_back):
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
