from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 5
FRAME = 3
RED = 2
BLUE = 1

GRID_MIN = 18
GRID_MAX = 30
MAX_DEPTH = 4
CHAIN_COUNT_CHOICES = (1, 2, 3)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary_background(grid: np.ndarray) -> int:
    boundary = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    return Counter(map(int, boundary)).most_common(1)[0][0]


def _extract_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == frame_color):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must have a non-empty interior"

        local = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True

        expected = np.zeros_like(local)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(local, expected), "each foreground component must be a 1-pixel rectangular frame"
        frames.append((r0, c0, r1, c1))

    frames.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed grey region according to how many green rectangular frames surround it.

    Input structure:
    - A grey background contains one or more 1-pixel-thick green rectangular frame outlines.
    - Frames may be nested to several levels, and disjoint top-level groups remain separated by grey background.
    - Every non-frame cell starts as grey.

    Rule:
    - Treat each green rectangle as a closed barrier and each 4-connected grey component as one region.
    - For every grey region, count how many green frames strictly contain that region.
    - Leave depth-0 background grey, fill odd depths red, fill even positive depths blue, and keep the green frame pixels unchanged.

    Key insight:
    - The target color is determined by topological containment depth, not by geometric distance to the nearest frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _boundary_background(g)
    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert len(colors) == 1, "expected exactly one frame color on the grey background"
    frame_color = colors.pop()

    frames = _extract_frames(g, frame_color)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for region in _components(g == bg):
        r, c = region[0]
        depth = sum(r0 < r < r1 and c0 < c < c1 for r0, c0, r1, c1 in frames)
        if depth == 0:
            continue
        fill = RED if depth % 2 == 1 else BLUE
        for rr, cc in region:
            out[rr, cc] = fill
    return out


def _min_outer_span(depth: int) -> int:
    return 3 + 4 * (depth - 1)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _boxes_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _make_chain_boxes(root: tuple[int, int, int, int], depth: int) -> list[tuple[int, int, int, int]]:
    boxes = [root]
    current = root
    for child_depth in range(depth - 1, 0, -1):
        r0, c0, r1, c1 = current
        cur_h = r1 - r0 + 1
        cur_w = c1 - c0 + 1
        child_min = _min_outer_span(child_depth)
        max_child_h = cur_h - 4
        max_child_w = cur_w - 4
        assert child_min <= max_child_h and child_min <= max_child_w

        child_h = random.randint(child_min, max_child_h)
        child_w = random.randint(child_min, max_child_w)

        extra_h = cur_h - child_h
        extra_w = cur_w - child_w
        top_inset = random.randint(2, extra_h - 2)
        left_inset = random.randint(2, extra_w - 2)
        bottom_inset = extra_h - top_inset
        right_inset = extra_w - left_inset

        child = (
            r0 + top_inset,
            c0 + left_inset,
            r1 - bottom_inset,
            c1 - right_inset,
        )
        boxes.append(child)
        current = child
    return boxes


def _sample_root_size(depth: int, limit_h: int, limit_w: int) -> tuple[int, int] | None:
    min_span = _min_outer_span(depth)
    max_h = min(limit_h, min_span + 8)
    max_w = min(limit_w, min_span + 8)
    if min_span > max_h or min_span > max_w:
        return None
    return random.randint(min_span, max_h), random.randint(min_span, max_w)


def _place_roots(
    shape: tuple[int, int],
    specs: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    occupied: list[tuple[int, int, int, int]] = []
    indexed_specs = sorted(
        enumerate(specs),
        key=lambda item: item[1][1] * item[1][2],
        reverse=True,
    )
    for idx, (_, box_h, box_w) in indexed_specs:
        candidates = []
        for _ in range(120):
            r0 = random.randint(1, h - box_h - 1)
            c0 = random.randint(1, w - box_w - 1)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_boxes_far_enough(box, other, gap=1) for other in occupied):
                candidates.append(box)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placed[idx] = chosen
        occupied.append(chosen)
    return [box for box in placed if box is not None]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    max_depth = min(MAX_DEPTH, (min(h, w) - 1) // 4)
    if max_depth < 1:
        return None

    chain_count = random.choice(CHAIN_COUNT_CHOICES)
    depths = [random.randint(1, max_depth)]
    while len(depths) < chain_count:
        depths.append(random.randint(1, max_depth))

    specs: list[tuple[int, int, int]] = []
    for depth in depths:
        size = _sample_root_size(depth, h - 2, w - 2)
        if size is None:
            return None
        box_h, box_w = size
        specs.append((depth, box_h, box_w))

    roots = _place_roots((h, w), specs)
    if roots is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    total_frames = 0
    for (depth, _, _), root in zip(specs, roots):
        chain = _make_chain_boxes(root, depth)
        total_frames += len(chain)
        for box in chain:
            _draw_frame(grid, box)

    frames = _extract_frames(grid, FRAME)
    if len(frames) != total_frames:
        return None

    # Make sure the generated sample exposes the intended alternating-depth rule.
    painted = main(grid)
    if not np.any(painted == RED):
        return None
    if any(depth >= 2 for depth in depths) and not np.any(painted == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
