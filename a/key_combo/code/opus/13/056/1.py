from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _outside_black(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros_like(grid, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == 0 and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _crossing_edges(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    edges: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center == 0:
                continue
            left = int(g[r, c - 1])
            right = int(g[r, c + 1])
            up = int(g[r - 1, c])
            down = int(g[r + 1, c])
            if left == right != 0 and up == down != 0 and left != up:
                if center == left:
                    edges.add((up, left))
                elif center == up:
                    edges.add((left, up))
                else:
                    raise AssertionError("crossing center must match one straight-through contour")
    return edges


def _unique_bottom_to_top(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    higher_from: dict[int, set[int]] = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in higher_from[lower]:
            higher_from[lower].add(upper)
            indeg[upper] += 1

    q = deque(sorted(color for color in colors if indeg[color] == 0))
    order: list[int] = []
    while q:
        if len(q) != 1:
            raise AssertionError("crossings must determine a unique full layer order")
        color = q.popleft()
        order.append(color)
        for nxt in sorted(higher_from[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(colors):
        raise AssertionError("layer constraints must be acyclic and cover all contour colors")
    return order


def _region_boundary_colors(grid: np.ndarray, region: list[tuple[int, int]]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    boundary: set[int] = set()
    for r, c in region:
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0:
                boundary.add(int(g[nr, nc]))
    return sorted(boundary)


def _modal_boundary_color(grid: np.ndarray, region: list[tuple[int, int]]) -> int | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts: Counter[int] = Counter()
    for r, c in region:
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0:
                counts[int(g[nr, nc])] += 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda color: counts[color])


def _fill_with_rank(grid: np.ndarray, rank: dict[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    outside = _outside_black(g)
    enclosed = (g == 0) & ~outside
    regions = _components(enclosed)
    out = g.copy()
    for region in regions:
        boundary_colors = _region_boundary_colors(g, region)
        if len(boundary_colors) < 2:
            raise AssertionError("every enclosed region must be bounded by at least two contour colors")
        fill = min(boundary_colors, key=lambda color: rank[color])
        for r, c in region:
            out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region with the lowest-layer contour color on its boundary.

    Input structure:
    - The grid has black background cells and 3-5 differently colored one-cell-wide closed contours.
    - Some contours cross orthogonally, and each crossing shows only the visible top contour color.
    - The contour arrangement partitions the black background into exterior space and multiple enclosed regions.

    Rule:
    - Treat every non-black cell as a wall and flood-fill from the border through black cells to mark the exterior.
    - At each orthogonal crossing, compare the visible center color with the straight horizontal and vertical neighbors to infer which contour is above the other, then combine all such constraints into one unique bottom-to-top layer order.
    - For every enclosed black region, collect the colors of the contour segments touching that region and fill the region with the lowest-ranked color among them.
    - Preserve all contour pixels and all exterior black pixels.

    Key insight:
    - The visible color at a crossing identifies the upper contour, but the required fill color is the opposite extreme: the lowest-layer color among the contours that bound each enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    colors = sorted(int(color) for color in np.unique(g) if color != 0)
    assert 3 <= len(colors) <= 5, "expected 3-5 contour colors on black background"

    edges = _crossing_edges(g)
    assert edges, "expected at least one orthogonal crossing"
    order = _unique_bottom_to_top(colors, edges)
    rank = {color: idx for idx, color in enumerate(order)}
    return _fill_with_rank(g, rank)


def _frame_patch(height: int, width: int, orientation: str) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if orientation == "h":
        split = random.randint(1, height - 2)
        left_shift_limit = min(1, width - 4)
        right_shift_limit = min(1, width - 4)
        left_shift = random.randint(0, left_shift_limit)
        right_shift = random.randint(0, right_shift_limit)
        left_inner = left_shift
        right_inner = width - 1 - right_shift
        if right_inner - left_inner < 2:
            left_inner = 0
            right_inner = width - 1
        patch[0, :] = 1
        patch[: split + 1, 0] = 1
        patch[: split + 1, width - 1] = 1
        patch[split, : left_inner + 1] = 1
        patch[split, right_inner:] = 1
        patch[split:, left_inner] = 1
        patch[split:, right_inner] = 1
        patch[height - 1, left_inner : right_inner + 1] = 1
        return patch
    if orientation == "v":
        return _frame_patch(width, height, "h").T
    raise ValueError("orientation must be 'h' or 'v'")


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    mask = patch.astype(bool)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = color


def _orientation_sequence(total: int) -> list[str]:
    start = random.choice(["h", "v"])
    other = "v" if start == "h" else "h"
    return [start if i % 2 == 0 else other for i in range(total)]


def _spaced_intervals(count: int, span_choices: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    spans = [random.randint(*span_choices) for _ in range(count)]
    gaps = [random.randint(1, 2) for _ in range(max(0, count - 1))]
    lead = random.randint(2, 4)
    trail = random.randint(2, 4)
    pos = lead
    intervals: list[tuple[int, int]] = []
    for i, span in enumerate(spans):
        intervals.append((pos, pos + span - 1))
        pos += span
        if i < len(gaps):
            pos += gaps[i]
    return intervals, pos + trail


def _gen_attempt() -> np.ndarray | None:
    total = random.randint(3, 5)
    layer_orients = _orientation_sequence(total)
    h_count = layer_orients.count("h")
    v_count = layer_orients.count("v")
    h_rows, grid_h = _spaced_intervals(h_count, (4, 5))
    v_cols, grid_w = _spaced_intervals(v_count, (4, 5))
    if grid_h > 30 or grid_w > 30:
        return None

    h_band_top = min(r0 for r0, _ in h_rows)
    h_band_bottom = max(r1 for _, r1 in h_rows)
    v_band_left = min(c0 for c0, _ in v_cols)
    v_band_right = max(c1 for _, c1 in v_cols)
    if h_band_top < 2 or v_band_left < 2 or h_band_bottom > grid_h - 3 or v_band_right > grid_w - 3:
        return None

    horizontal_frames: list[tuple[int, int, int, int, np.ndarray]] = []
    for r0, r1 in h_rows:
        c0 = random.randint(1, v_band_left - 1)
        c1 = random.randint(v_band_right + 1, grid_w - 2)
        patch = _frame_patch(r1 - r0 + 1, c1 - c0 + 1, "h")
        horizontal_frames.append((r0, c0, r1, c1, patch))

    vertical_frames: list[tuple[int, int, int, int, np.ndarray]] = []
    for c0, c1 in v_cols:
        r0 = random.randint(1, h_band_top - 1)
        r1 = random.randint(h_band_bottom + 1, grid_h - 2)
        patch = _frame_patch(r1 - r0 + 1, c1 - c0 + 1, "v")
        vertical_frames.append((r0, c0, r1, c1, patch))

    colors = random.sample(COLORS, total)
    h_queue = deque(random.sample(horizontal_frames, len(horizontal_frames)))
    v_queue = deque(random.sample(vertical_frames, len(vertical_frames)))
    truth_order: list[int] = []
    draw_plan: list[tuple[int, int, np.ndarray, int]] = []
    for color, orient in zip(colors, layer_orients):
        if orient == "h":
            r0, c0, _r1, _c1, patch = h_queue.popleft()
        else:
            r0, c0, _r1, _c1, patch = v_queue.popleft()
        truth_order.append(color)
        draw_plan.append((r0, c0, patch, color))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r0, c0, patch, color in draw_plan:
        _place_patch(grid, patch, r0, c0, color)

    try:
        inferred_order = _unique_bottom_to_top(
            sorted(int(color) for color in np.unique(grid) if color != 0),
            _crossing_edges(grid),
        )
    except AssertionError:
        return None
    if inferred_order != truth_order:
        return None

    rank = {color: idx for idx, color in enumerate(truth_order)}
    outside = _outside_black(grid)
    enclosed = (grid == 0) & ~outside
    regions = _components(enclosed)
    if len(regions) < 3:
        return None

    fill_colors: list[int] = []
    non_modal_case = False
    for region in regions:
        boundary_colors = _region_boundary_colors(grid, region)
        if len(boundary_colors) < 2:
            return None
        fill = min(boundary_colors, key=lambda color: rank[color])
        fill_colors.append(fill)
        modal = _modal_boundary_color(grid, region)
        if modal is not None and modal != fill:
            non_modal_case = True
    if len(set(fill_colors)) < 2 or not non_modal_case:
        return None

    expected = _fill_with_rank(grid, rank)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 056/1")
