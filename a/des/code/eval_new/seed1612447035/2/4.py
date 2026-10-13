from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if np.array_equal(mask, expected):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _draw_frame(canvas: np.ndarray, color: int) -> None:
    canvas[0, :] = color
    canvas[-1, :] = color
    canvas[:, 0] = color
    canvas[:, -1] = color


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _sorted_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        bbox = _frame_bbox(grid == color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    frames.sort(key=lambda item: (int(item["area"]), item["bbox"]))
    assert len(frames) >= 2, "expected at least two nested rectangular frames"

    for inner, outer in zip(frames, frames[1:]):
        ir0, ic0, ir1, ic1 = inner["bbox"]
        or0, oc0, or1, oc1 = outer["bbox"]
        assert ir0 - or0 >= 2 and ic0 - oc0 >= 2
        assert or1 - ir1 >= 2 and oc1 - ic1 >= 2

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a nested frame stack into horizontal layers whose copied motif borrows the next outer frame color.

    Input structure:
    - bg is the most common color.
    - The foreground contains one concentric chain of distinct-colored, one-pixel-thick rectangular hollow frames.
    - A small non-bg motif appears only inside the innermost frame.

    Rule:
    - Detect the rectangular frames and sort them from innermost to outermost.
    - Keep the innermost motif's geometry and absolute placement within the nested stack.
    - For each separated layer, draw only that layer's frame. Copy the innermost motif into the same relative location; for every non-outermost layer recolor the copied motif to the next outer frame color, while the outermost layer keeps the motif's original color(s).
    - Arrange the cropped layer grids from inner to outer, left to right, with one bg column between neighbors.

    Key insight:
    - The shape to replicate is always the innermost motif, but the fill color for each extracted layer comes from the immediately surrounding outer frame rather than from the layer itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frames = _sorted_frames(g, bg)

    ir0, ic0, ir1, ic1 = frames[0]["bbox"]
    inner_region = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
    motif_mask = inner_region != bg
    assert motif_mask.any(), "expected a non-background motif inside the innermost frame"

    motif_rel = np.argwhere(motif_mask) + np.array([ir0 + 1, ic0 + 1])
    motif_vals = g[motif_rel[:, 0], motif_rel[:, 1]]

    patches: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        fr0, fc0, fr1, fc1 = frame["bbox"]
        fh = int(fr1 - fr0 + 1)
        fw = int(fc1 - fc0 + 1)
        patch = np.full((fh, fw), bg, dtype=int)
        _draw_frame(patch, int(frame["color"]))

        rr = motif_rel[:, 0] - fr0
        cc = motif_rel[:, 1] - fc0
        if idx < len(frames) - 1:
            patch[rr, cc] = int(frames[idx + 1]["color"])
        else:
            patch[rr, cc] = motif_vals
        patches.append(patch)

    out_h = max(patch.shape[0] for patch in patches)
    out_w = sum(patch.shape[1] for patch in patches) + len(patches) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    cursor = 0
    for patch in patches:
        ph, pw = patch.shape
        out[:ph, cursor : cursor + pw] = patch
        cursor += pw + 1
    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    area_limit = min(6, height * width - 1)
    assert area_limit >= 2

    for _ in range(100):
        target = random.randint(2, area_limit)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if bbox_area == target and target > 2:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask

    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    mask[min(1, height - 1), 0] = True
    mask[min(1, height - 1), min(1, width - 1)] = True
    return mask


def _sample_dimensions(depth: int) -> tuple[list[int], list[int], list[int], list[int]] | None:
    if depth == 3:
        # Four full frame crops cannot fit side-by-side inside 30 columns, so the
        # generator uses the two feasible depths while the solver handles any chain.
        inner_h = random.randint(5, 9)
        inner_w = 5
        gap_rows = [random.randint(1, 2), random.randint(1, 2)]
        gap_cols = [1, 1]
    else:
        inner_h = random.randint(5, 10)
        inner_w = random.randint(5, 10)
        gap_rows = [random.randint(1, 3)]
        gap_cols = [random.randint(1, 3)]

    heights = [inner_h]
    widths = [inner_w]
    for gap_r, gap_c in zip(gap_rows, gap_cols):
        heights.append(heights[-1] + 2 * (gap_r + 1))
        widths.append(widths[-1] + 2 * (gap_c + 1))

    if sum(widths) + depth - 1 > 30:
        return None
    if heights[-1] > 28 or widths[-1] > 28:
        return None
    return heights, widths, gap_rows, gap_cols


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        depth = random.choice([2, 3])
        dims = _sample_dimensions(depth)
        if dims is None:
            continue

        heights, widths, gap_rows, gap_cols = dims
        outer_h = heights[-1]
        outer_w = widths[-1]

        max_top = 30 - outer_h - 1
        max_left = 30 - outer_w - 1
        if max_top < 1 or max_left < 1:
            continue

        top = random.randint(1, min(6, max_top))
        left = random.randint(1, min(6, max_left))
        bottom_max = 30 - outer_h - top
        right_max = 30 - outer_w - left
        if bottom_max < 1 or right_max < 1:
            continue
        bottom = random.randint(1, min(6, bottom_max))
        right = random.randint(1, min(6, right_max))

        grid = np.zeros((outer_h + top + bottom, outer_w + left + right), dtype=int)

        bboxes: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * depth
        bboxes[-1] = (top, left, top + outer_h - 1, left + outer_w - 1)
        for idx in range(depth - 2, -1, -1):
            dr = gap_rows[idx] + 1
            dc = gap_cols[idx] + 1
            or0, oc0, or1, oc1 = bboxes[idx + 1]
            bboxes[idx] = (or0 + dr, oc0 + dc, or1 - dr, oc1 - dc)

        colors = random.sample(range(1, 10), depth + 1)
        frame_colors = colors[:depth]
        motif_color = colors[-1]

        for bbox, color in zip(bboxes, frame_colors):
            _paint_frame(grid, bbox, color)

        ir0, ic0, ir1, ic1 = bboxes[0]
        interior_h = ir1 - ir0 - 1
        interior_w = ic1 - ic0 - 1
        if interior_h < 2 or interior_w < 2:
            continue

        motif = _random_connected_mask(interior_h, interior_w)
        grid[ir0 + 1 : ir1, ic0 + 1 : ic1][motif] = motif_color

        if _most_common_color(grid) != 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid nested-frame puzzle")
