from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
GRID_H_RANGE = (12, 20)
TRUNC_H_RANGE = (10, 12)
NONTRUNC_H_RANGE = (13, 20)
GRID_W_RANGE = (14, 26)
TRUNC_W_RANGE = (20, 26)
NONTRUNC_W_RANGE = GRID_W_RANGE
KEY_COUNT_RANGE = (2, 4)
COUNT_RANGE = (1, 4)
LEGEND_W_RANGE = (5, 7)
MOTIF_LEN_RANGE = (2, 3)
OBJECT_SIZE_RANGE = (3, 6)
OBJECT_BBOXES = [(2, 2), (2, 3), (3, 2), (3, 3)]
TRUNCATE_PROB = 0.45


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_border(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp) != expected:
        return False
    cells = set(comp)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in cells) != on_border:
                return False
    return True


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == GREY)
    candidates = [_bbox(comp) for comp in comps if _is_rect_border(comp)]
    assert len(candidates) == 1, "expected exactly one grey legend frame"
    return candidates[0]


def _extract_keys_and_motifs(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, tuple[int, ...]]]:
    r0, c0, r1, c1 = legend_bbox
    items: list[tuple[int, int, tuple[int, ...]]] = []
    for r in range(r0 + 1, r1):
        row = grid[r, c0 + 1 : c1]
        non_grey = [i for i, v in enumerate(row) if int(v) != GREY]
        assert non_grey, "legend interior row must not contain grey"
        if all(int(v) == BLACK for v in row):
            continue
        key_idx = next(i for i, v in enumerate(row) if int(v) != BLACK)
        key_color = int(row[key_idx])
        motif = []
        for idx in range(key_idx + 1, row.shape[0]):
            val = int(row[idx])
            if val == BLACK:
                break
            motif.append(val)
        assert motif, "each key row needs a motif to the right"
        assert all(int(v) != BLACK for v in motif), "motif must be a contiguous non-black run"
        assert all(int(v) == BLACK for v in row[key_idx + 1 + len(motif) :]), "motif row must end with black padding"
        items.append((r, key_color, tuple(motif)))
    assert 2 <= len(items) <= 4, "expected 2-4 legend keys"
    for (r_a, _, _), (r_b, _, _) in zip(items, items[1:]):
        assert r_b - r_a >= 2, "key rows must be separated"
    return items


def _count_outside_objects(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    color: int,
) -> int:
    r0, c0, r1, c1 = legend_bbox
    mask = grid == int(color)
    mask = mask.copy()
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return len(_components(mask))


def _render_output(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    items: list[tuple[int, int, tuple[int, ...]]],
) -> np.ndarray:
    out = np.zeros_like(grid)
    r0, c0, r1, c1 = legend_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    start_c = c1 + 1
    write_r = 0
    for _row, color, motif in items:
        reps = _count_outside_objects(grid, legend_bbox, color)
        for _ in range(reps):
            if write_r >= out.shape[0]:
                return out
            for dc, value in enumerate(motif):
                cc = start_c + dc
                if cc < out.shape[1]:
                    out[write_r, cc] = int(value)
            write_r += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve the grey legend box and stack each row motif downward a number of times equal to its outside object count.

    Input structure:
    - The grid has a black background and one grey rectangular frame on the left that serves as a legend box.
    - Inside the box, selected non-adjacent rows contain a color key followed by a short 1-row motif to its right.
    - Outside the legend, each key color appears as several separate 4-connected monochrome objects.

    Rule:
    - Find the unique grey legend frame and read its key rows from top to bottom.
    - For each key color, count the number of 4-connected objects of that color outside the legend box.
    - Produce an output of the same size with only the legend box preserved; to the right of the legend, from the top row downward, repeat each row motif exactly that many times in key-row order, truncating at the bottom if needed.

    Key insight:
    - The legend gives the color-to-motif mapping, but the repeat count comes from counting separate outside objects, so both the legend parsing and the component counting are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_bbox = _find_legend_bbox(g)
    items = _extract_keys_and_motifs(g, legend_bbox)
    return _render_output(g, legend_bbox, items)


def _sample_key_rows(height: int, k: int) -> list[int]:
    rows = list(range(1, height - 1))
    for _ in range(200):
        chosen = sorted(random.sample(rows, k))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen
    raise ValueError("not enough separated rows for legend keys")


def _sample_motif(key_color: int, used: set[tuple[int, ...]], max_len: int) -> tuple[int, ...]:
    for _ in range(200):
        length = random.randint(MOTIF_LEN_RANGE[0], min(MOTIF_LEN_RANGE[1], max_len))
        palette = random.sample(PALETTE, random.randint(2, min(4, len(PALETTE))))
        if key_color not in palette:
            palette[0] = key_color
        motif = [key_color]
        for _ in range(length - 1):
            motif.append(random.choice(palette))
        if len(set(motif)) == 1 and length > 1:
            motif[-1] = random.choice([c for c in palette if c != motif[0]])
        t = tuple(motif)
        if t not in used:
            used.add(t)
            return t
    raise ValueError("failed to sample a unique motif")


def _random_connected_shape(size: int) -> np.ndarray:
    h, w = random.choice([dims for dims in OBJECT_BBOXES if dims[0] * dims[1] >= size])
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells]
        if not neighbors:
            continue
        cells.add(random.choice(neighbors))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = 1
    return patch


def _can_place(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > grid.shape[0] or left + pw > grid.shape[1]:
        return False
    for r in range(ph):
        for c in range(pw):
            if patch[r, c] == 0:
                continue
            rr = top + r
            cc = left + c
            if occupied[rr, cc]:
                return False
            for nr in range(rr - 1, rr + 2):
                for nc in range(cc - 1, cc + 2):
                    if _in_bounds(grid.shape, nr, nc) and occupied[nr, nc]:
                        return False
    return True


def _stamp_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    patch: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    ph, pw = patch.shape
    positions = [
        (r, c)
        for r in range(grid.shape[0] - ph + 1)
        for c in range(grid.shape[1] - pw + 1)
    ]
    random.shuffle(positions)
    for top, left in positions:
        if forbidden[top : top + ph, left : left + pw].any():
            continue
        if not _can_place(grid, occupied, patch, top, left):
            continue
        for r in range(ph):
            for c in range(pw):
                if patch[r, c]:
                    rr = top + r
                    cc = left + c
                    grid[rr, cc] = int(color)
                    occupied[rr, cc] = True
        return True
    return False


def _oracle_from_layout(
    shape: tuple[int, int],
    legend_cols: tuple[int, int],
    key_rows: list[int],
    motifs: list[tuple[int, ...]],
    key_colors: list[int],
    counts: list[int],
) -> np.ndarray:
    h, w = shape
    c0, c1 = legend_cols
    grid = np.zeros((h, w), dtype=int)
    grid[:, c0] = GREY
    grid[:, c1] = GREY
    grid[0, c0 : c1 + 1] = GREY
    grid[h - 1, c0 : c1 + 1] = GREY
    for row, color, motif in zip(key_rows, key_colors, motifs):
        grid[row, c0 + 1] = int(color)
        for i, value in enumerate(motif, start=2):
            grid[row, c0 + i] = int(value)
    out = np.zeros((h, w), dtype=int)
    out[:, c0 : c1 + 1] = grid[:, c0 : c1 + 1]
    write_r = 0
    start_c = c1 + 1
    for motif, count in zip(motifs, counts):
        for _ in range(count):
            if write_r >= h:
                return out
            for dc, value in enumerate(motif):
                cc = start_c + dc
                if cc < w:
                    out[write_r, cc] = int(value)
            write_r += 1
    return out


def _validate_input(
    grid: np.ndarray,
    legend_cols: tuple[int, int],
    key_rows: list[int],
    key_colors: list[int],
    motifs: list[tuple[int, ...]],
    counts: list[int],
    expected: np.ndarray,
) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len(set(motifs)) == len(motifs), "motifs must be unique"
    assert len(set(counts)) >= 2, "counts must not all be the same"
    assert sum(counts) >= 4, "stack must be visibly non-trivial"
    legend_bbox = _find_legend_bbox(grid)
    assert legend_bbox == (0, legend_cols[0], grid.shape[0] - 1, legend_cols[1])
    items = _extract_keys_and_motifs(grid, legend_bbox)
    assert [row for row, _, _ in items] == key_rows
    assert [color for _, color, _ in items] == key_colors
    assert [motif for _, _, motif in items] == motifs
    for color, count in zip(key_colors, counts):
        assert _count_outside_objects(grid, legend_bbox, color) == count
    grey_positions = np.argwhere(grid == GREY)
    assert grey_positions.shape[0] == 2 * grid.shape[0] + 2 * (legend_cols[1] - legend_cols[0] - 1)
    solved = main(grid)
    assert np.array_equal(solved, expected), "solver mismatch"
    if sum(counts) > grid.shape[0]:
        assert np.any(expected[-1, legend_cols[1] + 1 :] != BLACK), "truncation case should reach the last row"
    else:
        assert not np.any(expected[sum(counts) :, legend_cols[1] + 1 :] != BLACK), "non-truncation case should stop exactly"


def _attempt_generate() -> np.ndarray | None:
    truncate_mode = random.random() < TRUNCATE_PROB
    h = random.randint(*(TRUNC_H_RANGE if truncate_mode else NONTRUNC_H_RANGE))
    w = random.randint(*(TRUNC_W_RANGE if truncate_mode else NONTRUNC_W_RANGE))
    if truncate_mode:
        k = random.choice([3, 4])
    else:
        k = random.randint(*KEY_COUNT_RANGE)
    legend_w = random.randint(*LEGEND_W_RANGE)
    if legend_w >= w - 3 or h < 2 * k + 3:
        return None

    key_rows = _sample_key_rows(h, k)
    key_colors = random.sample(PALETTE, k)
    max_motif_len = legend_w - 3
    if max_motif_len < MOTIF_LEN_RANGE[0]:
        return None
    used_motifs: set[tuple[int, ...]] = set()
    motifs = [_sample_motif(color, used_motifs, max_motif_len) for color in key_colors]

    if truncate_mode:
        counts = [random.randint(2, COUNT_RANGE[1]) for _ in range(k)]
    else:
        counts = [random.randint(*COUNT_RANGE) for _ in range(k)]
    if len(set(counts)) == 1:
        idx = random.randrange(k)
        choices = [n for n in range(COUNT_RANGE[0], COUNT_RANGE[1] + 1) if n != counts[idx]]
        counts[idx] = random.choice(choices)
    if truncate_mode:
        need = h + random.randint(1, max(1, k))
        while sum(counts) < need:
            idxs = [i for i, v in enumerate(counts) if v < COUNT_RANGE[1]]
            if not idxs:
                break
            counts[random.choice(idxs)] += 1
    else:
        while sum(counts) >= h:
            idxs = [i for i, v in enumerate(counts) if v > COUNT_RANGE[0]]
            if not idxs:
                break
            counts[random.choice(idxs)] -= 1
    if len(set(counts)) == 1 or sum(counts) < 4:
        return None

    legend_c0 = 0
    legend_c1 = legend_w - 1
    start_c = legend_c1 + 1
    if start_c + max(len(m) for m in motifs) > w:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, legend_c0] = GREY
    grid[:, legend_c1] = GREY
    grid[0, legend_c0 : legend_c1 + 1] = GREY
    grid[h - 1, legend_c0 : legend_c1 + 1] = GREY
    for row, color, motif in zip(key_rows, key_colors, motifs):
        grid[row, legend_c0 + 1] = int(color)
        for i, value in enumerate(motif, start=2):
            grid[row, legend_c0 + i] = int(value)

    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[:, : legend_c1 + 1] = True
    occupied = forbidden.copy()
    for r in range(h):
        for c in range(legend_c1 + 1):
            if grid[r, c] != BLACK:
                occupied[r, c] = True

    for color, count in zip(key_colors, counts):
        for _ in range(count):
            size = random.randint(3, 4) if truncate_mode else random.randint(*OBJECT_SIZE_RANGE)
            patch = _random_connected_shape(size)
            if not _stamp_object(grid, occupied, color, patch, forbidden):
                return None

    expected = _oracle_from_layout(
        grid.shape,
        (legend_c0, legend_c1),
        key_rows,
        motifs,
        key_colors,
        counts,
    )
    _validate_input(grid, (legend_c0, legend_c1), key_rows, key_colors, motifs, counts, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
