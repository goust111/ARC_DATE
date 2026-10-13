from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[np.ndarray] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        touches_border = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        )
        if not touches_border:
            continue
        if r0 == r1 and r0 in (0, h - 1) and len(comp) == c1 - c0 + 1:
            order = comp[np.argsort(comp[:, 1])]
            candidates.append(order)
            continue
        if c0 == c1 and c0 in (0, w - 1) and len(comp) == r1 - r0 + 1:
            order = comp[np.argsort(comp[:, 0])]
            candidates.append(order)

    assert len(candidates) == 1, "expected exactly one straight border legend strip"
    order = candidates[0]
    colors = [int(grid[r, c]) for r, c in order]
    cells = [(int(r), int(c)) for r, c in order]
    return colors, cells


def _extract_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == frame_color):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        cells = {tuple(map(int, cell)) for cell in comp}
        expected = 2 * (height + width) - 4
        if len(cells) != expected:
            continue
        ok = True
        for c in range(c0, c1 + 1):
            if (r0, c) not in cells or (r1, c) not in cells:
                ok = False
                break
        if not ok:
            continue
        for r in range(r0, r1 + 1):
            if (r, c0) not in cells or (r, c1) not in cells:
                ok = False
                break
        if ok:
            frames.append((r0, c0, r1, c1))

    frames.sort(
        key=lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1),
        reverse=True,
    )
    for i in range(1, len(frames)):
        pr0, pc0, pr1, pc1 = frames[i - 1]
        r0, c0, r1, c1 = frames[i]
        assert pr0 < r0 < r1 < pr1 and pc0 < c0 < c1 < pc1, "frames must be properly nested"
    return frames


def _paste_mask(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a seed pattern by nested-frame depth and paste rotated copies onto each inner wall.

    Input structure:
    - bg is the dominant background color.
    - One foreground color forms several disconnected one-cell-thick nested rectangular frames.
    - A small multicolor seed pattern sits inside the innermost frame, and a 1xN or Nx1 strip on the grid edge lists the replacement color for each depth from outermost to innermost.

    Rule:
    - Detect the border strip, read its colors in strip order, and erase the strip in the output.
    - Sort the nested rectangular frames from outermost depth 1 to innermost depth N.
    - Treat the seed only as a non-background mask: recolor the center seed with depth N's legend color, then stamp monochrome copies onto the top, right, bottom, and left inner wall of every frame using that frame's depth color; the copies use rotations 0, 90 clockwise, 180, and 90 counterclockwise respectively.

    Key insight:
    - The nested frames provide the spatial depth index, while the edge strip is the depth-to-color lookup table; the seed's original colors only define the template shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    legend_colors, legend_cells = _detect_legend(g, bg)
    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = bg

    fg = work[work != bg]
    assert fg.size, "expected foreground after removing the legend"
    values, counts = np.unique(fg, return_counts=True)
    frame_color = int(values[np.argmax(counts)])

    frames = _extract_frames(work, frame_color)
    assert frames, "expected at least one nested frame"
    assert len(frames) == len(legend_colors), "legend length must match frame depth count"

    ir0, ic0, ir1, ic1 = frames[-1]
    inner_region = work[ir0 + 1 : ir1, ic0 + 1 : ic1]
    seed_coords = np.argwhere(inner_region != bg)
    assert seed_coords.size, "expected a seed inside the innermost frame"
    sr0, sc0, sr1, sc1 = _bbox(seed_coords)
    seed_patch = inner_region[sr0 : sr1 + 1, sc0 : sc1 + 1]
    seed_mask = seed_patch != bg
    assert len(np.unique(seed_patch[seed_mask])) >= 2, "seed must be multicolor"

    center_top = ir0 + 1 + sr0
    center_left = ic0 + 1 + sc0
    out = work.copy()
    for rr, cc in np.argwhere(seed_mask):
        out[center_top + int(rr), center_left + int(cc)] = bg

    rotated = {
        "top": seed_mask,
        "right": np.rot90(seed_mask, -1),
        "bottom": np.rot90(seed_mask, 2),
        "left": np.rot90(seed_mask, 1),
    }

    for depth, (r0, c0, r1, c1) in enumerate(frames, start=1):
        color = int(legend_colors[depth - 1])
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1

        top_mask = rotated["top"]
        right_mask = rotated["right"]
        bottom_mask = rotated["bottom"]
        left_mask = rotated["left"]

        placements = [
            (top_mask, r0 + 1, c0 + 1 + (inner_w - top_mask.shape[1]) // 2),
            (
                right_mask,
                r0 + 1 + (inner_h - right_mask.shape[0]) // 2,
                c1 - right_mask.shape[1],
            ),
            (bottom_mask, r1 - bottom_mask.shape[0], c0 + 1 + (inner_w - bottom_mask.shape[1]) // 2),
            (
                left_mask,
                r0 + 1 + (inner_h - left_mask.shape[0]) // 2,
                c0 + 1,
            ),
        ]

        for patch_mask, top, left in placements:
            _paste_mask(out, patch_mask, color, int(top), int(left))

    _paste_mask(out, seed_mask, int(legend_colors[-1]), center_top, center_left)
    return out


_GEN_RETRIES = 400
_SIDES = ("top", "bottom", "left", "right")


def _grow_connected_mask(rng: random.Random, size: int) -> np.ndarray:
    target_area = 3 if size == 2 else rng.randint(size + 1, size * size - 1)
    start = (rng.randrange(size), rng.randrange(size))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < size and 0 <= nc < size:
            frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_seed_mask(rng: random.Random, size: int) -> np.ndarray:
    for _ in range(200):
        mask = _grow_connected_mask(rng, size)
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to generate a suitable seed mask")


def _feasible_specs(side: str) -> list[tuple[int, int, int, int]]:
    specs: list[tuple[int, int, int, int]] = []
    min_top = 2 if side == "top" else 1
    min_bottom = 2 if side == "bottom" else 1
    min_left = 2 if side == "left" else 1
    min_right = 2 if side == "right" else 1
    for depth in (2, 3):
        for size in (2, 3):
            step = size + 2
            for extra_h in range(3):
                for extra_w in range(3):
                    center_h = 3 * size + 2 + extra_h
                    center_w = 3 * size + 2 + extra_w
                    outer_h = center_h + 2 + 2 * (depth - 1) * step
                    outer_w = center_w + 2 + 2 * (depth - 1) * step
                    if outer_h + min_top + min_bottom <= 30 and outer_w + min_left + min_right <= 30:
                        specs.append((depth, size, extra_h, extra_w))
    return specs


def _make_seed_patch(rng: random.Random, size: int, colors: list[int]) -> np.ndarray:
    mask = _random_seed_mask(rng, size)
    patch = np.zeros((size, size), dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(cells)
    used_colors = colors[:]
    for idx, (r, c) in enumerate(cells):
        if idx < len(used_colors):
            patch[r, c] = used_colors[idx]
        else:
            patch[r, c] = rng.choice(used_colors)
    return patch


def _gen_attempt(rng: random.Random) -> np.ndarray:
    side = rng.choice(_SIDES)
    specs = _feasible_specs(side)
    depth, size, extra_h, extra_w = rng.choice(specs)

    pool = list(range(10))
    rng.shuffle(pool)
    bg = pool.pop()
    frame_color = pool.pop()
    legend_colors = [pool.pop() for _ in range(depth)]
    seed_color_count = 2 if size == 2 else rng.randint(2, 3)
    seed_colors = [pool.pop() for _ in range(seed_color_count)]

    seed_patch = _make_seed_patch(rng, size, seed_colors)

    step = size + 2
    center_h = 3 * size + 2 + extra_h
    center_w = 3 * size + 2 + extra_w
    outer_h = center_h + 2 + 2 * (depth - 1) * step
    outer_w = center_w + 2 + 2 * (depth - 1) * step

    min_top = 2 if side == "top" else 1
    min_bottom = 2 if side == "bottom" else 1
    min_left = 2 if side == "left" else 1
    min_right = 2 if side == "right" else 1

    extra_total_h = rng.randint(0, 30 - (outer_h + min_top + min_bottom))
    extra_total_w = rng.randint(0, 30 - (outer_w + min_left + min_right))
    add_top = rng.randint(0, extra_total_h)
    add_bottom = extra_total_h - add_top
    add_left = rng.randint(0, extra_total_w)
    add_right = extra_total_w - add_left

    top_margin = min_top + add_top
    bottom_margin = min_bottom + add_bottom
    left_margin = min_left + add_left
    right_margin = min_right + add_right

    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    grid = np.full((h, w), bg, dtype=int)

    outer_r0 = top_margin
    outer_c0 = left_margin
    outer_r1 = outer_r0 + outer_h - 1
    outer_c1 = outer_c0 + outer_w - 1

    for layer in range(depth):
        r0 = outer_r0 + layer * step
        c0 = outer_c0 + layer * step
        r1 = outer_r1 - layer * step
        c1 = outer_c1 - layer * step
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

    inner_r0 = outer_r0 + (depth - 1) * step
    inner_c0 = outer_c0 + (depth - 1) * step
    seed_top = inner_r0 + 1 + (center_h - size) // 2
    seed_left = inner_c0 + 1 + (center_w - size) // 2
    seed_view = grid[seed_top : seed_top + size, seed_left : seed_left + size]
    seed_view[seed_patch != bg] = seed_patch[seed_patch != bg]

    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        start = rng.randint(0, w - depth)
        grid[row, start : start + depth] = np.array(legend_colors, dtype=int)
    else:
        col = 0 if side == "left" else w - 1
        start = rng.randint(0, h - depth)
        grid[start : start + depth, col] = np.array(legend_colors, dtype=int)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "task must produce a non-trivial output"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {_GEN_RETRIES} attempts")
