from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL_COLOR = 2
FRAME_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (22, 29)
GRID_WIDTH_RANGE = (20, 29)
FRAME_COUNT_RANGE = (2, 4)
FRAME_HEIGHT_MAX = 6
FRAME_WIDTH_MAX = 10
STEP_ROW_RANGE = (1, 1)
STEP_COL_RANGE = (1, 2)
TOP_RANGE = (1, 3)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
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


def _infer_frames(
    grid: np.ndarray, bg: int, wall_color: int
) -> tuple[list[int], dict[int, np.ndarray]]:
    colors = [int(color) for color in np.unique(grid) if color not in (bg, wall_color)]
    assert colors, "expected at least one airborne frame color"

    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each frame color must remain visible"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 > r0 and c1 > c0, "each color must support a non-degenerate rectangle"
        frame = _frame_mask(grid.shape, bbox)
        visible = grid == color
        assert np.all(frame[visible]), "visible pixels must lie on the reconstructed frame perimeter"
        frames[color] = frame
    return colors, frames


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    adjacency = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for earlier, later in edges:
        if later not in adjacency[earlier]:
            adjacency[earlier].add(later)
            indegree[later] += 1

    order: list[int] = []
    available = sorted(color for color in colors if indegree[color] == 0)
    while available:
        assert len(available) == 1, "occlusion clues must determine a unique total order"
        current = available.pop()
        order.append(current)
        for nxt in sorted(adjacency[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)
        available.sort()

    assert len(order) == len(colors), "occlusion graph must cover every frame color"
    return order


def _occlusion_order(
    grid: np.ndarray, colors: list[int], frames: dict[int, np.ndarray]
) -> list[int]:
    edges: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            crossing = frames[color_a] & frames[color_b]
            if not np.any(crossing):
                continue
            visible = set(map(int, np.unique(grid[crossing]))) & {color_a, color_b}
            assert len(visible) == 1, "each pairwise crossing must expose exactly one of the two frames"
            front = next(iter(visible))
            if front == color_a:
                edges.add((color_b, color_a))
            else:
                edges.add((color_a, color_b))
    return _unique_topological_order(colors, edges)


def _drop_frames(
    shape: tuple[int, int],
    wall_color: int,
    order: list[int],
    frames: dict[int, np.ndarray],
    bg: int,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), bg, dtype=int)
    out[height - 1, :] = wall_color
    occupied = out != bg

    step = np.array([1, 0], dtype=int)
    for color in order:
        coords = np.argwhere(frames[color]).astype(int)
        while True:
            nxt = coords + step
            if np.any(nxt[:, 0] >= height) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "settled frames must not overlap"
        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replay overlapped hollow frames as a bottom-stacking gravity sequence inferred from their occlusion order.

    Input structure:
    - The bottom row is a complete red wall, and red does not appear anywhere else.
    - Above the wall sit several one-cell-thick hollow rectangular frames in distinct colors.
    - The frames overlap in the air; each color still has enough visible perimeter pixels to recover its full bbox.

    Rule:
    - Reconstruct each full rectangular frame from the bbox of its visible pixels.
    - At every pairwise frame crossing, the visible color is the front frame, so the hidden frame must settle earlier; these clues define a unique total order.
    - Clear the airborne scene, keep every frame's original columns, and drop the frames straight downward one by one in that inferred order until the next step would hit the red wall or a previously settled frame.

    Key insight:
    - The overlap colors are not damage to repair; they are timestamps that encode the exact order in which gravity should be replayed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    wall_color = int(g[-1, 0])
    assert wall_color == WALL_COLOR, "the supporting wall must be red"
    assert np.all(g[-1, :] == wall_color), "expected a complete bottom wall"
    assert int(np.sum(g == wall_color)) == g.shape[1], "red must appear only in the bottom wall"

    bg = _get_bg(g)
    colors, frames = _infer_frames(g, bg, wall_color)
    order = _occlusion_order(g, colors, frames)
    return _drop_frames(g.shape, wall_color, order, frames, bg)


def _pairwise_overlap(masks: list[np.ndarray]) -> bool:
    for i, mask_a in enumerate(masks):
        for mask_b in masks[i + 1 :]:
            if not np.any(mask_a & mask_b):
                return False
    return True


def _gen_diagonal_bboxes(
    height: int, width: int, count: int
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(80):
        step_r = random.randint(*STEP_ROW_RANGE)
        step_c = random.randint(*STEP_COL_RANGE)
        min_h = (count - 1) * step_r + 4
        min_w = (count - 1) * step_c + 5
        if min_h > FRAME_HEIGHT_MAX or min_w > FRAME_WIDTH_MAX:
            continue

        frame_h = random.randint(min_h, FRAME_HEIGHT_MAX)
        frame_w = random.randint(min_w, FRAME_WIDTH_MAX)
        max_top = height - frame_h - (count - 1) * step_r - 4
        if max_top < 1:
            continue
        top = random.randint(TOP_RANGE[0], min(TOP_RANGE[1], max_top))

        direction = random.choice((-1, 1))
        if direction == 1:
            left_lo = 1
            left_hi = width - frame_w - (count - 1) * step_c - 2
        else:
            left_lo = 1 + (count - 1) * step_c
            left_hi = width - frame_w - 2
        if left_lo > left_hi:
            continue
        left = random.randint(left_lo, left_hi)

        bboxes = []
        for idx in range(count):
            r0 = top + idx * step_r
            c0 = left + direction * idx * step_c
            bboxes.append((r0, c0, r0 + frame_h - 1, c0 + frame_w - 1))
        return bboxes
    return None


def _paint_input(
    height: int, width: int, colors: list[int], bboxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1, :] = WALL_COLOR
    masks = [_frame_mask((height, width), bbox) for bbox in bboxes]
    for color, mask in zip(colors, masks, strict=True):
        grid[mask] = color
    return grid, masks


def _visible_bboxes_match(grid: np.ndarray, colors: list[int], bboxes: list[tuple[int, int, int, int]]) -> bool:
    for color, bbox in zip(colors, bboxes, strict=True):
        coords = np.argwhere(grid == color)
        if coords.size == 0 or _bbox_from_coords(coords) != bbox:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*FRAME_COUNT_RANGE)
    colors = random.sample(FRAME_COLORS, count)

    bboxes = _gen_diagonal_bboxes(height, width, count)
    if bboxes is None:
        return None

    grid, masks = _paint_input(height, width, colors, bboxes)
    cover = np.zeros((height, width), dtype=int)
    for mask in masks:
        cover += mask.astype(int)

    if int(cover.max()) > 2:
        return None
    if not _pairwise_overlap(masks):
        return None
    if not _visible_bboxes_match(grid, colors, bboxes):
        return None

    try:
        inferred_colors, frames = _infer_frames(grid, BG, WALL_COLOR)
        order = _occlusion_order(grid, inferred_colors, frames)
        if order != colors:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-replay frame input")
