from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _infer_frames(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != 0)
    assert 3 <= len(colors) <= 4, "expected 3 or 4 frame colors"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "every frame color must stay visible"
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "frames must be at least 5x5"
        mask = _frame_mask(grid.shape, bbox)
        assert np.all(mask[grid == color]), "visible pixels must lie on the inferred perimeter"
        bboxes[color] = bbox
        frames[color] = mask
    return colors, bboxes, frames


def _crossing_edges_top_to_bottom(
    grid: np.ndarray,
    colors: list[int],
    frames: dict[int, np.ndarray],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            crossing = frames[a] & frames[b]
            assert bool(crossing.any()), "every frame pair must cross"
            visible = {int(v) for v in np.unique(grid[crossing])}
            assert visible <= {a, b} and len(visible) == 1, "each crossing pair must have a single visible top color"
            top = visible.pop()
            bottom = b if top == a else a
            edges.add((top, bottom))
    return edges


def _unique_top_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for src, dst in edges:
        if dst not in outgoing[src]:
            outgoing[src].add(dst)
            indegree[dst] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = sorted(color for color in remaining if indegree[color] == 0)
        assert len(ready) == 1, "crossings must determine a unique top-to-bottom order"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for nxt in outgoing[color]:
            indegree[nxt] -= 1
    return order


def _paint_concentric_layers(
    canvas: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    requested_layers: int,
) -> None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    capacity = (min(h, w) + 1) // 2
    layers = min(int(requested_layers), int(capacity))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            if depth >= layers:
                continue
            canvas[r, c] = int(color) if depth % 2 == 0 else 0


def _render_output(
    shape: tuple[int, int],
    bboxes: dict[int, tuple[int, int, int, int]],
    top_order: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    above_count = {color: idx for idx, color in enumerate(top_order)}
    for color in reversed(top_order):
        _paint_concentric_layers(out, bboxes[color], color, above_count[color] + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite overlapping rectangular frames as concentric ring patterns whose layer counts come from occlusion depth.

    Input structure:
    - Black background with 3 or 4 distinct non-black colors.
    - Each color belongs to one 1-pixel-thick axis-aligned rectangular frame.
    - The frames partially overlap pairwise, and at each shared crossing exactly one frame color is visible.

    Rule:
    - For each color, reconstruct its full rectangle from the bounding box of its visible pixels.
    - Compare every pair of reconstructed perimeters: the visible color at their crossings is the frame above the other one.
    - These pairwise above/below relations define a unique top-to-bottom order; a frame gets `k + 1` concentric layers where `k` is the number of frames above it.
    - Draw frames from bottom to top inside their own bounding boxes, alternating visible shells as `frame color, black, frame color, ...`; if a box cannot hold all requested shells, truncate to the maximum it can fit.

    Key insight:
    - The same occlusion evidence serves two roles at once: it fixes the overlay order and also encodes how many inward shells each frame must contribute.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    colors, bboxes, frames = _infer_frames(g)
    edges = _crossing_edges_top_to_bottom(g, colors, frames)
    top_order = _unique_top_order(colors, edges)
    return _render_output(g.shape, bboxes, top_order)


def _odd_values(lo: int, hi: int) -> list[int]:
    return [v for v in range(int(lo), int(hi) + 1) if v % 2 == 1]


def _sample_centered_bboxes(
    h: int,
    w: int,
    n_frames: int,
) -> list[tuple[int, int, int, int]] | None:
    min_extent = 5 + 2 * (n_frames - 1)
    height_choices = _odd_values(min_extent, min(h, 19))
    width_choices = _odd_values(min_extent, min(w, 19))
    if not height_choices or not width_choices:
        return None

    max_h = random.choice(height_choices)
    max_w = random.choice(width_choices)
    height_pool = _odd_values(5, max_h)
    width_pool = _odd_values(5, max_w)
    if len(height_pool) < n_frames or len(width_pool) < n_frames:
        return None

    heights = sorted(random.sample(height_pool, n_frames), reverse=True)
    widths = sorted(random.sample(width_pool, n_frames))
    if len(set(heights)) != n_frames or len(set(widths)) != n_frames:
        return None

    center_r_lo = heights[0] // 2
    center_r_hi = h - 1 - heights[0] // 2
    center_c_lo = widths[-1] // 2
    center_c_hi = w - 1 - widths[-1] // 2
    if center_r_lo > center_r_hi or center_c_lo > center_c_hi:
        return None

    center_r = random.randint(center_r_lo, center_r_hi)
    center_c = random.randint(center_c_lo, center_c_hi)
    bboxes = []
    for hh, ww in zip(heights, widths):
        r0 = center_r - hh // 2
        c0 = center_c - ww // 2
        bboxes.append((r0, c0, r0 + hh - 1, c0 + ww - 1))
    return bboxes


def _render_input(
    shape: tuple[int, int],
    bboxes: dict[int, tuple[int, int, int, int]],
    top_order: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in reversed(top_order):
        grid[_frame_mask(shape, bboxes[color])] = int(color)
    return grid


def _crossing_count(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> int:
    mask_a = _frame_mask((31, 31), bbox_a)
    mask_b = _frame_mask((31, 31), bbox_b)
    return int(np.count_nonzero(mask_a & mask_b))


def _semantic_check(
    grid: np.ndarray,
    bboxes: dict[int, tuple[int, int, int, int]],
    top_order: list[int],
    expect_truncation: bool,
) -> bool:
    try:
        colors, inferred_bboxes, frames = _infer_frames(grid)
        if colors != sorted(top_order):
            return False
        for color, bbox in bboxes.items():
            if inferred_bboxes[color] != bbox:
                return False

        edges = _crossing_edges_top_to_bottom(grid, colors, frames)
        inferred_top = _unique_top_order(colors, edges)
        if inferred_top != top_order:
            return False

        for i, a in enumerate(colors):
            for b in colors[i + 1 :]:
                if _crossing_count(bboxes[a], bboxes[b]) < 4:
                    return False

        capacities = {
            color: (min(bboxes[color][2] - bboxes[color][0] + 1, bboxes[color][3] - bboxes[color][1] + 1) + 1) // 2
            for color in colors
        }
        requested = {color: inferred_top.index(color) + 1 for color in colors}
        has_truncation = any(requested[color] > capacities[color] for color in colors)
        if has_truncation != expect_truncation:
            return False

        expected = _render_output(grid.shape, bboxes, top_order)
        actual = main(grid)
        if np.array_equal(actual, grid):
            return False
        if not np.array_equal(actual, expected):
            return False

        nonzero_per_color = [int(np.count_nonzero(actual == color)) for color in colors]
        if min(nonzero_per_color) <= 0:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice([3, 4])
    expect_truncation = n_frames == 4 and random.random() < 0.6

    h = random.randint(15, 22)
    w = random.randint(15, 22)
    bbox_list = _sample_centered_bboxes(h, w, n_frames)
    if bbox_list is None:
        return None

    colors = random.sample(PALETTE, n_frames)
    random.shuffle(bbox_list)
    bboxes = {color: bbox for color, bbox in zip(colors, bbox_list)}

    top_order = random.sample(colors, len(colors))
    grid = _render_input((h, w), bboxes, top_order)
    if not _semantic_check(grid, bboxes, top_order, expect_truncation):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC input")
