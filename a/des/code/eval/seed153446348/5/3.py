import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
LINE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300
REGION_SIZE_RANGE = (4, 8)
LAYOUTS = ((3, 2), (2, 3), (3, 3))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _region_records(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    interior = (g == BLACK) | (g == GREY)
    visited = np.zeros((h, w), dtype=bool)
    regions = []

    for r in range(h):
        for c in range(w):
            if not interior[r, c] or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert bool(interior[r0 : r1 + 1, c0 : c1 + 1].all()), "Each pane interior must be a solid rectangle"

            boundary_colors = []
            if r0 > 0:
                boundary_colors.append(_single_boundary_color(g[r0 - 1, c0 : c1 + 1]))
            if r1 < h - 1:
                boundary_colors.append(_single_boundary_color(g[r1 + 1, c0 : c1 + 1]))
            if c0 > 0:
                boundary_colors.append(_single_boundary_color(g[r0 : r1 + 1, c0 - 1]))
            if c1 < w - 1:
                boundary_colors.append(_single_boundary_color(g[r0 : r1 + 1, c1 + 1]))

            assert boundary_colors, "Each pane must touch at least one colored divider"
            regions.append(
                {
                    "bbox": (r0, r1, c0, c1),
                    "boundary_colors": tuple(boundary_colors),
                    "k": len(set(boundary_colors)),
                }
            )

    assert regions, "Expected at least one pane"
    return regions


def _single_boundary_color(values: np.ndarray) -> int:
    colors = {int(v) for v in np.asarray(values).flat if int(v) not in (BLACK, GREY)}
    assert len(colors) == 1, "Each pane side should touch exactly one divider color"
    return next(iter(colors))


def _transform_patch(patch: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return patch.copy()
    if k == 2:
        return np.fliplr(patch)
    if k == 3:
        return np.flipud(patch)
    assert k == 4, "Valid panes should see between one and four boundary colors"
    return np.rot90(patch, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each rectangular pane according to how many divider colors touch it.

    Input structure:
    - Black background with grey motifs inside rectangular panes.
    - One-pixel colored dividers split the canvas into axis-aligned panes.

    Rule:
    - Count the distinct divider colors touching the pane's top/bottom/left/right
      sides, ignoring outer grid edges with no divider.
    - K=1: unchanged; K=2: horizontal mirror; K=3: vertical mirror; K=4: rotate 180.
    - Divider lines stay fixed; only the pane interior changes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _region_records(g):
        r0, r1, c0, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _transform_patch(g[r0 : r1 + 1, c0 : c1 + 1], region["k"])
    return out


def _layout_colors(n_h_lines: int, n_v_lines: int, rng: random.Random):
    a, b, c, d = rng.sample(LINE_PALETTE, 4)
    if (n_h_lines, n_v_lines) == (3, 2):
        return [a, b, c], [c, d]
    if (n_h_lines, n_v_lines) == (2, 3):
        return [a, b], [c, d, a]
    assert (n_h_lines, n_v_lines) == (3, 3)
    return [a, b, c], [c, d, a]


def _random_sizes(count: int, rng: random.Random):
    lo, hi = REGION_SIZE_RANGE
    return [rng.randint(lo, hi) for _ in range(count)]


def _build_grid(heights, widths, h_colors, v_colors, rng: random.Random):
    total_h = sum(heights) + len(h_colors)
    total_w = sum(widths) + len(v_colors)
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_starts = []
    row_lines = []
    cursor = 0
    for idx, span in enumerate(heights):
        row_starts.append(cursor)
        cursor += span
        if idx < len(h_colors):
            row_lines.append(cursor)
            cursor += 1

    col_starts = []
    col_lines = []
    cursor = 0
    for idx, span in enumerate(widths):
        col_starts.append(cursor)
        cursor += span
        if idx < len(v_colors):
            col_lines.append(cursor)
            cursor += 1

    for row, color in zip(row_lines, h_colors):
        grid[row, :] = int(color)
    for col, color in zip(col_lines, v_colors):
        grid[:, col] = int(color)
    for row, h_color in zip(row_lines, h_colors):
        for col, v_color in zip(col_lines, v_colors):
            grid[row, col] = int(rng.choice((h_color, v_color)))

    regions = []
    for i, span_h in enumerate(heights):
        for j, span_w in enumerate(widths):
            r0 = row_starts[i]
            c0 = col_starts[j]
            r1 = r0 + span_h - 1
            c1 = c0 + span_w - 1
            regions.append((r0, r1, c0, c1))
            _paint_pattern(grid, (r0, r1, c0, c1), rng)

    return grid, regions


def _paint_pattern(grid: np.ndarray, bbox, rng: random.Random) -> None:
    r0, r1, c0, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    shape = _random_asymmetric_shape(h, w, rng)
    ph, pw = shape.shape
    off_r = rng.randint(0, h - ph)
    off_c = rng.randint(0, w - pw)
    patch = grid[r0 + off_r : r0 + off_r + ph, c0 + off_c : c0 + off_c + pw]
    patch[shape] = GREY


def _random_asymmetric_shape(max_h: int, max_w: int, rng: random.Random) -> np.ndarray:
    max_box_h = min(max_h, 6)
    max_box_w = min(max_w, 6)
    for _ in range(200):
        box_h = rng.randint(2, max_box_h)
        box_w = rng.randint(2, max_box_w)
        max_cells = min(box_h * box_w - 1, 8)
        if max_cells < 3:
            continue
        target = rng.randint(3, max_cells)

        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            shape[r - r0, c - c0] = True

        if shape.shape[0] < 2 or shape.shape[1] < 2:
            continue
        if shape.shape[0] > max_h or shape.shape[1] > max_w:
            continue
        if bool(shape.all()):
            continue
        if np.array_equal(shape, np.fliplr(shape)):
            continue
        if np.array_equal(shape, np.flipud(shape)):
            continue
        if np.array_equal(shape, np.rot90(shape, 2)):
            continue
        return shape
    raise RuntimeError("failed to build asymmetric pattern")


def _apply_dihedral(grid: np.ndarray, flip_lr: bool, rot_k: int) -> np.ndarray:
    out = np.rot90(grid, rot_k)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _branch_visibility_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    seen = {1: False, 2: False, 3: False, 4: False}
    for region in _region_records(inp):
        r0, r1, c0, c1 = region["bbox"]
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        if region["k"] == 1 and np.array_equal(before, after):
            seen[1] = True
        if region["k"] in (2, 3, 4) and not np.array_equal(before, after):
            seen[region["k"]] = True
    return all(seen.values())


def _gen_attempt(rng: random.Random):
    n_h_lines, n_v_lines = rng.choice(LAYOUTS)
    heights = _random_sizes(n_h_lines + 1, rng)
    widths = _random_sizes(n_v_lines + 1, rng)
    h_colors, v_colors = _layout_colors(n_h_lines, n_v_lines, rng)
    grid, _regions = _build_grid(heights, widths, h_colors, v_colors, rng)

    grid = _apply_dihedral(grid, rng.choice((False, True)), rng.randrange(4))

    try:
        records = _region_records(grid)
        if {region["k"] for region in records} != {1, 2, 3, 4}:
            return None
        out = main(grid)
        if not _branch_visibility_ok(grid, out):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
