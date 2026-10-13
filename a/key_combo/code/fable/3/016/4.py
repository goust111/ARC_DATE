"""Recolor rectangular frames by shifting colors along the depth chain inferred from overlap visibility.

Input structure:
- Black or dominant background canvas.
- Several one-pixel-thick axis-aligned rectangular frames, one foreground color each.
- Frames may overlap; visible pixels are the topmost colors at intersections.
- Every frame color has at least one visible pixel on each frame side so its full bounding box is recoverable from visible pixels.

Rule:
- Recover each frame's full rectangle from the axis-aligned bounding box of its visible pixels.
- For every pair of frames, inspect perimeter intersections and add an edge from the lower frame color to the visible upper frame color.
- Determine the unique depth order path that visits every frame color once.
- Repaint each frame with the color of its immediate upper neighbor in that depth chain; the topmost frame gets the bottommost color.
- Redraw all frames on the same canvas size using the inferred depth order so occlusions match the input relations.

Key insight:
- The visible top color at any crossing point gives only local occlusion direction; chaining those directed crossings restores a unique depth permutation, and that permutation defines the color cycle.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import random
from typing import Sequence

import numpy as np


try:
    from arc_utils import (
        Color,
        bbox_coords,
        bbox_h,
        bbox_w,
        colors_present,
        get_bg,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when importing from isolated path.
    _utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _spec = importlib.util.spec_from_file_location("_arc_utils_fallback", _utils_path)
    if _spec is None or _spec.loader is None:
        raise
    _arc_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_arc_utils)  # type: ignore[union-attr]

    Color = _arc_utils.Color
    bbox_coords = _arc_utils.bbox_coords
    bbox_h = _arc_utils.bbox_h
    bbox_w = _arc_utils.bbox_w
    colors_present = _arc_utils.colors_present
    get_bg = _arc_utils.get_bg


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask of a frame perimeter with coordinates in `bbox`."""
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _infer_frames(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    """Recover each color's full rectangle bbox from its visible pixels."""
    frames: dict[int, tuple[int, int, int, int]] = {}
    colors = colors_present(grid, exclude={bg})
    assert len(colors) >= 2, "expected at least two frame colors"

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing visible pixels for color {color}"
        r0, c0, r1, c1 = bbox_coords(coords)
        assert bbox_h((r0, c0, r1, c1)) >= 3, "frame must span at least 3 rows"
        assert bbox_w((r0, c0, r1, c1)) >= 3, "frame must span at least 3 columns"
        frames[int(color)] = (int(r0), int(c0), int(r1), int(c1))

    return frames


def _occlusion_edges(
    grid: np.ndarray, frames: dict[int, tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    """Return (below, above) directed edges from perimeter crossings."""
    colors = list(frames.keys())
    if len(colors) < 2:
        return set()

    masks = {color: _frame_mask(grid.shape, frames[color]) for color in colors}
    edges: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = masks[color_a] & masks[color_b]
            if not bool(overlap.any()):
                continue
            for rr, cc in np.argwhere(overlap):
                v = int(grid[int(rr), int(cc)])
                if v == color_a:
                    edges.add((color_b, color_a))
                elif v == color_b:
                    edges.add((color_a, color_b))

    return edges


def _unique_depth_chain(
    colors: Sequence[int], edges: set[tuple[int, int]]
) -> list[int]:
    """Recover unique Hamiltonian topological order if implied by occlusion relations."""
    if len(colors) <= 1:
        return list(colors)
    colors = list(colors)

    outgoing = {color: [] for color in colors}
    for a, b in edges:
        if a in outgoing and b in outgoing:
            outgoing[a].append(b)

    paths: list[tuple[int, ...]] = []

    def dfs(path: tuple[int, ...], remaining: set[int]) -> None:
        if len(paths) > 1:
            return
        if not remaining:
            paths.append(path)
            return

        last = path[-1]
        for nxt in outgoing[last]:
            if nxt not in remaining:
                continue
            dfs(path + (nxt,), remaining - {nxt})

    color_set = set(colors)
    for start in colors:
        dfs((start,), color_set - {start})
        if len(paths) > 1:
            break

    assert len(paths) == 1, "occlusion relations do not define a unique depth chain"
    return list(paths[0])


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _bbox_has_visible_sides(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    """Require each frame side to have at least one visible pixel in the input."""
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox
    rs = coords[:, 0]
    cs = coords[:, 1]
    return (
        bool(np.any(rs == r0))
        and bool(np.any(rs == r1))
        and bool(np.any(cs == c0))
        and bool(np.any(cs == c1))
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    frames = _infer_frames(g, bg)
    edges = _occlusion_edges(g, frames)
    assert edges, "expected at least one visible frame crossing"

    depth = _unique_depth_chain(list(frames.keys()), edges)
    depth_to_next = {
        depth[idx]: depth[(idx + 1) % len(depth)] for idx in range(len(depth))
    }

    out = np.full_like(g, bg)
    for color in depth:
        _paint_frame(out, frames[color], int(depth_to_next[color]))

    return out


def _random_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    rh = int(np.random.randint(3, max(3, h - 1)))
    rw = int(np.random.randint(3, max(3, w - 1)))
    r0 = int(np.random.randint(0, h - rh + 1))
    c0 = int(np.random.randint(0, w - rw + 1))
    return (r0, c0, r0 + rh - 1, c0 + rw - 1)


def _touches_perimeter(
    shape: tuple[int, int], a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return bool(np.any(_frame_mask(shape, a) & _frame_mask(shape, b)))


def _render_candidate(grid_h: int, grid_w: int, order: Sequence[int]) -> list[tuple[int, int, int, int]]:
    """Sample frame bboxes that produce a recoverable depth relation candidate."""
    n = len(order)
    bboxes: list[tuple[int, int, int, int]] = []
    shape = (grid_h, grid_w)

    for i in range(n):
        target_mode = np.random.rand() < 0.55
        if i == 0:
            required: list[int] = []
        elif target_mode:
            required = list(range(i))
        else:
            req_count = 1 if i == 1 else int(np.random.randint(1, min(i, 3) + 1))
            required = random.sample(range(i), req_count)

        for _ in range(600):
            candidate = _random_bbox(grid_h, grid_w)
            if candidate in bboxes:
                continue

            if i == 0:
                bboxes.append(candidate)
                break

            if not any(_touches_perimeter(shape, candidate, bboxes[j]) for j in required):
                continue
            bboxes.append(candidate)
            break
        else:
            return []

    return bboxes


def _build_input_from_order(
    grid_h: int, grid_w: int, order: Sequence[int], bboxes: Sequence[tuple[int, int, int, int]]
) -> np.ndarray:
    out = np.full((grid_h, grid_w), int(Color.BLACK), dtype=int)
    for color, bbox in zip(order, bboxes):
        _paint_frame(out, bbox, int(color))
    return out


def _valid_for_main(grid: np.ndarray) -> bool:
    """Validation gate used by generate_input."""
    try:
        bg = get_bg(grid)
        frames = _infer_frames(grid, bg)
        for color, bbox in frames.items():
            if not _bbox_has_visible_sides(grid, bbox, color):
                return False

        edges = _occlusion_edges(grid, frames)
        if not edges:
            return False

        _ = _unique_depth_chain(list(frames.keys()), edges)
    except Exception:
        return False

    return True


def generate_input() -> np.ndarray:
    """Generate structured ARC-style examples matching the solver constraints."""
    max_attempts = 600
    for _ in range(max_attempts):
        h = int(np.random.randint(12, 30))
        w = int(np.random.randint(12, 30))
        n = int(np.random.randint(2, 7))

        available = list(Color.NOT_BLACK)
        random.shuffle(available)
        order = [int(c) for c in available[:n]]
        random.shuffle(order)

        bboxes = _render_candidate(h, w, order)
        if not bboxes:
            continue

        inp = _build_input_from_order(h, w, order, bboxes)
        if _valid_for_main(inp):
            return inp

    raise RuntimeError("failed to generate a valid depth-ring input")
