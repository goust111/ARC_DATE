from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400

BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    excluded = set(int(v) for v in exclude)
    return sorted(int(v) for v in np.unique(grid) if int(v) not in excluded)


def _frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        raise AssertionError("frame bbox must stay inside the grid")
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _grown_mask(shape: tuple[int, int], box: BBox, rank: int) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    rank = int(rank)
    out = np.zeros(shape, dtype=bool)

    er0 = max(0, r0 - rank)
    ec0 = max(0, c0 - rank)
    er1 = min(h - 1, r1 + rank)
    ec1 = min(w - 1, c1 + rank)
    out[er0 : er1 + 1, ec0 : ec1 + 1] = True

    if r1 - r0 >= 2 and c1 - c0 >= 2:
        ir0 = max(0, r0 + 1)
        ic0 = max(0, c0 + 1)
        ir1 = min(h - 1, r1 - 1)
        ic1 = min(w - 1, c1 - 1)
        if ir0 <= ir1 and ic0 <= ic1:
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return out


def _reconstruct_frames(grid: np.ndarray) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    for color in _colors_present(grid, exclude={BG}):
        visible = grid == color
        box = _bbox(visible)
        assert box is not None, "each frame color must remain visible"
        mask = _frame_mask(grid.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frames[color] = {"bbox": box, "mask": mask}
    assert 3 <= len(frames) <= 4, "expected three or four frames"
    return frames


def _overlap_edges(grid: np.ndarray, frames: dict[int, dict[str, object]]) -> set[tuple[int, int]]:
    colors = sorted(frames)
    masks = {color: frames[color]["mask"] for color in colors}
    h, w = grid.shape
    edges: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if bool(masks[color][r, c])]
            if len(covering) < 2:
                continue
            top = int(grid[r, c])
            assert top in covering, "overlap cells must show one covering frame"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    succ = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in succ[lower]:
            succ[lower].add(upper)
            indeg[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zero_indeg = sorted(color for color in remaining if indeg[color] == 0)
        assert len(zero_indeg) == 1, "layer order must be uniquely determined"
        color = zero_indeg[0]
        remaining.remove(color)
        order.append(color)
        for upper in succ[color]:
            indeg[upper] -= 1

    return order


def _solve_structure(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], list[int], set[tuple[int, int]]]:
    frames = _reconstruct_frames(grid)
    colors = sorted(frames)
    edges = _overlap_edges(grid, frames)
    assert edges, "at least one overlap is required"
    order = _unique_topological_order(colors, edges)
    return frames, order, edges


def _render_input(shape: tuple[int, int], order: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in order:
        grid[_frame_mask(shape, boxes[color])] = color
    return grid


def _render_output(shape: tuple[int, int], order: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rank, color in enumerate(order, start=1):
        out[_grown_mask(shape, boxes[color], rank)] = color
    return out


def _flip_box(box: BBox, h: int, w: int, flip_ud: bool, flip_lr: bool) -> BBox:
    r0, c0, r1, c1 = box
    if flip_ud:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_lr:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return int(r0), int(c0), int(r1), int(c1)


def _sample_boxes(rng: np.random.Generator, order: list[int]) -> tuple[tuple[int, int], dict[int, BBox]]:
    n = len(order)
    heights = sorted(int(v) for v in rng.choice(np.arange(5, 9), size=n, replace=False))
    widths = sorted(int(v) for v in rng.choice(np.arange(5, 10), size=n, replace=False))

    margin_choices = np.array([0, 1, 2, 3, 4])
    margin_probs = np.array([0.14, 0.20, 0.32, 0.20, 0.14])
    top = int(rng.choice(margin_choices, p=margin_probs))
    left = int(rng.choice(margin_choices, p=margin_probs))
    bottom = int(rng.choice(margin_choices, p=margin_probs))
    right = int(rng.choice(margin_choices, p=margin_probs))

    h = top + (n - 1) + heights[-1] + bottom
    w = left + (n - 1) + widths[-1] + right

    boxes: dict[int, BBox] = {}
    for idx, color in enumerate(order):
        r0 = top + idx
        c0 = left + idx
        r1 = r0 + heights[idx] - 1
        c1 = c0 + widths[idx] - 1
        boxes[color] = (r0, c0, r1, c1)

    flip_ud = bool(rng.integers(0, 2))
    flip_lr = bool(rng.integers(0, 2))
    if flip_ud or flip_lr:
        boxes = {
            color: _flip_box(box, h, w, flip_ud=flip_ud, flip_lr=flip_lr)
            for color, box in boxes.items()
        }

    return (h, w), boxes


def _validate_attempt(inp: np.ndarray, boxes: dict[int, BBox], order: list[int]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.count_nonzero(inp) == 0:
        return False

    expected_pairs = {(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))}
    try:
        frames, solved_order, edges = _solve_structure(inp)
    except AssertionError:
        return False

    if solved_order != order:
        return False
    if not expected_pairs.issubset(edges):
        return False

    for color, box in boxes.items():
        if frames[color]["bbox"] != box:
            return False
        visible = inp == color
        if visible.sum() < 6:
            return False

    density = float(np.count_nonzero(inp)) / float(inp.size)
    if not (0.08 <= density <= 0.42):
        return False

    if not np.array_equal(main(inp), _render_output(inp.shape, order, boxes)):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer rectangular frame stacking order and convert the rank into outward growth thickness.

    Input structure:
    - The background is black.
    - Three or four different colors each form a partially visible one-pixel-thick hollow rectangle.
    - Overlap cells show only the color of the higher frame, but each frame's visible pixels still reveal its full bbox.

    Rule:
    - Reconstruct every frame as the border of the bbox of its visible pixels.
    - Use reconstructed overlap cells to recover the unique bottom-to-top stacking order.
    - Give rank 1 to the bottom frame and rank N to the top frame, then grow each frame outward by its rank.
    - The grown shape is the solid outer band around the original frame, clipped at the grid boundary, and higher-ranked frames overwrite lower-ranked ones where the growth regions overlap.

    Key insight:
    - The same recovered layer order is used twice: once to choose how far each frame grows and again to resolve new color conflicts after growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    frames, order, _ = _solve_structure(g)
    out = np.zeros(g.shape, dtype=int)
    for rank, color in enumerate(order, start=1):
        out[_grown_mask(g.shape, frames[color]["bbox"], rank)] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        n = int(rng.choice([3, 4]))
        colors = list(int(v) for v in rng.choice(PALETTE, size=n, replace=False))
        order = colors.copy()
        rng.shuffle(order)

        shape, boxes = _sample_boxes(rng, order)
        inp = _render_input(shape, order, boxes)
        if _validate_attempt(inp, boxes, order):
            return inp

    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
