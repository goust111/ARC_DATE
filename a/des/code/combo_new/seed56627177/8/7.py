from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
SEP = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in blocked]


def _segments_from_lines(size: int, lines: np.ndarray) -> list[tuple[int, int]] | None:
    indices = sorted({int(x) for x in np.asarray(lines, dtype=int).tolist()})
    segments: list[tuple[int, int]] = []
    start = 0
    for line in indices:
        if line < start or line >= size:
            return None
        if line > start:
            segments.append((start, line))
        start = line + 1
    if start < size:
        segments.append((start, size))
    if not segments:
        return None
    lengths = {b - a for a, b in segments}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        return None
    return segments


def _detect_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = grid.shape
    best: tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    for color in _colors_present(grid, exclude={BG}):
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if len(rows) == 0 and len(cols) == 0:
            continue

        row_segments = _segments_from_lines(height, rows) if len(rows) else [(0, height)]
        col_segments = _segments_from_lines(width, cols) if len(cols) else [(0, width)]
        if row_segments is None or col_segments is None:
            continue
        if len(row_segments) * len(col_segments) < 2:
            continue

        score = 3 * (len(rows) + len(cols)) + len(row_segments) + len(col_segments)
        if best is None or score > best[0]:
            best = (score, color, row_segments, col_segments)

    assert best is not None, "failed to detect the separator lattice"
    return best[1], best[2], best[3]


def _vote_patch(patches: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = patches[0].shape
    repaired = np.full((cell_h, cell_w), BG, dtype=int)
    voted_any = False

    for r in range(cell_h):
        for c in range(cell_w):
            votes = [int(patch[r, c]) for patch in patches if int(patch[r, c]) != BG]
            if votes:
                repaired[r, c] = Counter(votes).most_common(1)[0][0]
                voted_any = True

    assert voted_any, "expected at least one non-background object pixel"
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Repair every panel copy by majority-voting each aligned object pixel across the lattice.

    Input structure:
    - Black background fills the canvas outside and inside the repeated cell lattice.
    - Single-pixel maroon separator rows and columns divide the grid into equal-sized cells.
    - Each cell contains one aligned copy of the same colored object, but some object pixels are replaced by black.

    Rule:
    - Detect the separator lattice and extract every cell interior.
    - For each local cell coordinate, collect the non-black colors seen there across all cells.
    - Use the most frequent non-black color as the repaired color for that coordinate, leaving coordinates with no votes black.
    - Replace every cell interior by the repaired shared object while keeping the separator lines unchanged.

    Key insight:
    - The cells are damaged replicas of one template; black means missing data, so the original object is recovered by aligned cross-copy voting rather than from any single cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, row_segments, col_segments = _detect_lattice(g)
    assert sep_color != BG

    patches = [g[r0:r1, c0:c1].copy() for r0, r1 in row_segments for c0, c1 in col_segments]
    cell_shapes = {patch.shape for patch in patches}
    assert len(cell_shapes) == 1, "all lattice cells must have the same size"

    repaired = _vote_patch(patches)
    out = g.copy()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            out[r0:r1, c0:c1] = repaired

    if len(row_segments) > 1:
        for r in range(g.shape[0]):
            if np.all(g[r] == sep_color):
                assert np.all(out[r] == sep_color)
    if len(col_segments) > 1:
        for c in range(g.shape[1]):
            if np.all(g[:, c] == sep_color):
                assert np.all(out[:, c] == sep_color)

    return out


def _grow_blob_mask(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    frontier = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = start[0] + dr, start[1] + dc
        if 0 <= rr < height and 0 <= cc < width:
            frontier.add((rr, cc))

    while int(mask.sum()) < area and frontier:
        candidates = list(frontier)
        weights = np.array(
            [
                sum(
                    0 <= rr + dr < height and 0 <= cc + dc < width and mask[rr + dr, cc + dc]
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                for rr, cc in candidates
            ],
            dtype=float,
        )
        weights /= weights.sum()
        rr, cc = candidates[int(rng.choice(len(candidates), p=weights))]
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _frame_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if height >= 5 and rng.random() < 0.5:
        mask[height // 2, 1:-1] = True
    if width >= 5 and rng.random() < 0.5:
        mask[1:-1, width // 2] = True

    return mask


def _bar_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(50):
        mask = np.zeros((height, width), dtype=bool)
        anchor_r = int(rng.integers(height))
        anchor_c = int(rng.integers(width))

        c0 = int(rng.integers(anchor_c + 1))
        c1 = int(rng.integers(anchor_c, width))
        r0 = int(rng.integers(anchor_r + 1))
        r1 = int(rng.integers(anchor_r, height))

        mask[anchor_r, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, anchor_c] = True

        if rng.random() < 0.5:
            if rng.random() < 0.5:
                mask[anchor_r, :] = True
            else:
                mask[:, anchor_c] = True

        if int(mask.sum()) >= 6:
            return mask

    mask = np.zeros((height, width), dtype=bool)
    mask[height // 2, :] = True
    mask[:, width // 2] = True
    return mask


def _ensure_min_area(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    out = mask.copy()
    height, width = out.shape
    while int(out.sum()) < min_area:
        candidates: list[tuple[int, int, int]] = []
        for r in range(height):
            for c in range(width):
                if out[r, c]:
                    continue
                neighbors = sum(
                    0 <= r + dr < height and 0 <= c + dc < width and out[r + dr, c + dc]
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                if neighbors:
                    candidates.append((neighbors, r, c))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, r, c = candidates[0]
        out[r, c] = True
    return out


def _sample_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mode = int(rng.integers(3))
    if mode == 0:
        max_area = max(6, int(round(height * width * 0.75)))
        min_area = max(6, int(round(height * width * 0.4)))
        area = int(rng.integers(min_area, max_area + 1))
        return _ensure_min_area(_grow_blob_mask(height, width, area, rng))
    if mode == 1:
        return _ensure_min_area(_frame_mask(height, width, rng))
    return _ensure_min_area(_bar_mask(height, width, rng))


def _paint_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    colors = rng.choice(PALETTE, size=3, replace=False).tolist()
    patch = np.full(mask.shape, BG, dtype=int)
    mode = int(rng.integers(3))

    if mode == 0:
        for r, c in coords:
            patch[r, c] = colors[0]
        return patch

    if mode == 1:
        axis = int(rng.integers(2))
        ordered = sorted(coords, key=lambda rc: (rc[axis], rc[1 - axis], rc[0], rc[1]))
        split = max(1, len(ordered) // 2)
        for idx, (r, c) in enumerate(ordered):
            patch[r, c] = colors[0] if idx < split else colors[1]
        return patch

    orientation = int(rng.integers(3))
    if orientation == 0:
        ordered = sorted(coords, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    elif orientation == 1:
        ordered = sorted(coords, key=lambda rc: (rc[0], rc[1]))
    else:
        ordered = sorted(coords, key=lambda rc: (rc[1], rc[0]))
    cuts = np.linspace(0, len(ordered), num=4, dtype=int)
    for band in range(3):
        for r, c in ordered[int(cuts[band]) : int(cuts[band + 1])]:
            patch[r, c] = colors[band]
    return patch


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    for _ in range(100):
        row_count = int(rng.integers(1, 5))
        col_count = int(rng.integers(1, 5))
        if row_count * col_count < 4:
            continue

        max_cell_h = min(7, (30 - (row_count + 1)) // row_count)
        max_cell_w = min(7, (30 - (col_count + 1)) // col_count)
        if max_cell_h < 5 or max_cell_w < 5:
            continue

        cell_h = int(rng.integers(5, max_cell_h + 1))
        cell_w = int(rng.integers(5, max_cell_w + 1))
        return row_count, col_count, cell_h, cell_w
    return None


def _build_template(cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray:
    bbox_h = int(rng.integers(3, cell_h - 1))
    bbox_w = int(rng.integers(3, cell_w - 1))
    top = int(rng.integers(1, cell_h - bbox_h))
    left = int(rng.integers(1, cell_w - bbox_w))

    mask = _sample_mask(bbox_h, bbox_w, rng)
    assert int(mask.sum()) >= 6

    patch = np.full((cell_h, cell_w), BG, dtype=int)
    patch[top : top + bbox_h, left : left + bbox_w] = _paint_mask(mask, rng)
    return patch


def _damage_patches(template: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    obj_coords = np.argwhere(template != BG)
    area = int(len(obj_coords))
    assert area >= 6

    patches = [template.copy() for _ in range(count)]
    min_remove = max(1, area // 5)
    max_remove = min(area - 1, max(min_remove, (2 * area) // 5))

    for patch in patches:
        remove_count = int(rng.integers(min_remove, max_remove + 1))
        choice = rng.choice(area, size=remove_count, replace=False)
        for idx in np.atleast_1d(choice):
            r, c = obj_coords[int(idx)]
            patch[int(r), int(c)] = BG

    for r0, c0 in obj_coords:
        r = int(r0)
        c = int(c0)
        visible = [idx for idx, patch in enumerate(patches) if int(patch[r, c]) != BG]
        if not visible:
            restore_idx = int(rng.integers(count))
            patches[restore_idx][r, c] = template[r, c]
            visible = [restore_idx]
        if len(visible) == count:
            candidates = [idx for idx in visible if np.count_nonzero(patches[idx] != BG) > 1]
            if candidates:
                hide_idx = int(rng.choice(candidates))
                patches[hide_idx][r, c] = BG

    for patch in patches:
        visible_count = int(np.count_nonzero(patch != BG))
        if visible_count == 0:
            keep = int(rng.integers(area))
            r, c = obj_coords[keep]
            patch[int(r), int(c)] = template[int(r), int(c)]
            visible_count = 1
        if visible_count == area:
            hide = int(rng.integers(area))
            r, c = obj_coords[hide]
            patch[int(r), int(c)] = BG

    return patches


def _cell_segments(count: int, cell_size: int) -> list[tuple[int, int]]:
    return [(1 + i * (cell_size + 1), 1 + i * (cell_size + 1) + cell_size) for i in range(count)]


def _render_grid(
    row_count: int,
    col_count: int,
    cell_h: int,
    cell_w: int,
    patches: list[np.ndarray],
) -> np.ndarray:
    height = row_count * cell_h + row_count + 1
    width = col_count * cell_w + col_count + 1
    grid = np.full((height, width), BG, dtype=int)

    for r in range(0, height, cell_h + 1):
        grid[r, :] = SEP
    for c in range(0, width, cell_w + 1):
        grid[:, c] = SEP

    row_segments = _cell_segments(row_count, cell_h)
    col_segments = _cell_segments(col_count, cell_w)
    idx = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            grid[r0:r1, c0:c1] = patches[idx]
            idx += 1

    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout(_RNG)
    if layout is None:
        return None
    row_count, col_count, cell_h, cell_w = layout

    template = _build_template(cell_h, cell_w, _RNG)
    patches = _damage_patches(template, row_count * col_count, _RNG)
    grid = _render_grid(row_count, col_count, cell_h, cell_w, patches)
    expected = _render_grid(row_count, col_count, cell_h, cell_w, [template.copy() for _ in patches])

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
