import random
from collections import deque

import numpy as np


BG = 0
BLOCK_COLOR = 5
COUNT_TO_COLOR = {1: 3, 2: 2, 3: 1, 4: 4}

GRID_SIDE_RANGE = (20, 29)
GEN_ATTEMPTS = 400
AXIS_GAP = 5

PATTERNS = (
    {"rows": [0, 0, 1, 2], "cols": [0, 1, 0, 2]},
    {"rows": [0, 0, 1, 2, 3], "cols": [0, 1, 0, 2, 3]},
    {"rows": [0, 0, 1, 1, 2], "cols": [0, 1, 0, 2, 3]},
    {"rows": [0, 0, 1, 2, 3], "cols": [0, 1, 2, 1, 3]},
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_blocks(grid: np.ndarray) -> list[dict]:
    comps = _components_mask(grid == BLOCK_COLOR)
    blocks = []
    for coords in comps:
        if coords.shape[0] != 9:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLOCK_COLOR):
            continue
        blocks.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 1, c0 + 1),
            }
        )
    return blocks


def _cross_neighbor_count(blocks: list[dict], idx: int) -> int:
    r, c = blocks[idx]["center"]
    count = 0
    for j, other in enumerate(blocks):
        if j == idx:
            continue
        rr, cc = other["center"]
        if rr != r and cc != c:
            count += 1
    return count


def _validate_blocks(blocks: list[dict], shape: tuple[int, int]) -> None:
    assert 4 <= len(blocks) <= 5, "expected 4 or 5 solid 3x3 gray blocks"
    h, w = shape
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        assert 0 <= r0 and 0 <= c0 and r1 < h and c1 < w
    shared_row = False
    shared_col = False
    has_crossing = False
    for i, bi in enumerate(blocks):
        ri, ci = bi["center"]
        for bj in blocks[i + 1 :]:
            rj, cj = bj["center"]
            dr = abs(ri - rj)
            dc = abs(ci - cj)
            assert dr > 2 or dc > 2, "blocks must not overlap"
            if ri == rj:
                shared_row = True
                assert dc >= AXIS_GAP, "same-row blocks need at least two blank columns between bodies"
            if ci == cj:
                shared_col = True
                assert dr >= AXIS_GAP, "same-column blocks need at least two blank rows between bodies"
            if ri != rj and ci != cj:
                has_crossing = True
                assert dr >= 3 and dc >= 3, "crossing points must stay outside both source blocks"
    assert shared_row and shared_col, "generator must expose both merge cases"
    assert has_crossing, "generator must also expose real cross intersections"
    counts = [_cross_neighbor_count(blocks, i) for i in range(len(blocks))]
    assert len(set(counts)) > 1, "cross-neighbor counts must not all match"
    assert all(k in COUNT_TO_COLOR for k in counts), "counts must stay in the visible recolor range"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each gray 3x3 block into a cross, clear cross intersections, and recolor each block by how many other blocks truly cross it.

    Input structure:
    - The background is black.
    - Foreground objects are 4 to 5 disconnected gray solid 3x3 blocks.
    - Some blocks may share a center row or share a center column, but no pair shares both.

    Rule:
    - For every gray 3x3 block, fill its center row across the whole grid and its center column across the whole grid with gray.
    - For every pair of blocks whose center rows differ and whose center columns differ, reset the two row-column crossing pixels to black.
    - For each block, count how many other blocks satisfy that same "different row and different column" condition, then recolor only that block's 3x3 body with K=1->green, K=2->red, K=3->blue, K=4->yellow.

    Key insight:
    - Shared center rows or shared center columns merge into one longer arm instead of creating a crossing, so each block's output color is determined by its row/column topology relative to the other blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    blocks = _find_blocks(g)
    _validate_blocks(blocks, g.shape)

    out = g.copy()
    for block in blocks:
        r, c = block["center"]
        out[r, :] = BLOCK_COLOR
        out[:, c] = BLOCK_COLOR

    for i, bi in enumerate(blocks):
        ri, ci = bi["center"]
        for bj in blocks[i + 1 :]:
            rj, cj = bj["center"]
            if ri != rj and ci != cj:
                out[ri, cj] = bg
                out[rj, ci] = bg

    for i, block in enumerate(blocks):
        k = _cross_neighbor_count(blocks, i)
        color = COUNT_TO_COLOR[k]
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _sample_axis_positions(length: int, count: int, gap: int) -> list[int] | None:
    lo = 1
    hi = length - 2
    slots = list(range(lo, hi + 1))
    if count == 0:
        return []
    for _ in range(200):
        random.shuffle(slots)
        chosen = []
        for pos in slots:
            if all(abs(pos - prev) >= gap for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _pattern_counts(pattern: dict) -> list[int]:
    rows = pattern["rows"]
    cols = pattern["cols"]
    n = len(rows)
    out = []
    for i in range(n):
        k = 0
        for j in range(n):
            if i != j and rows[i] != rows[j] and cols[i] != cols[j]:
                k += 1
        out.append(k)
    return out


def _choose_pattern() -> dict:
    candidates = list(PATTERNS)
    random.shuffle(candidates)
    return candidates[0]


def _build_grid_from_pattern(pattern: dict, height: int, width: int) -> np.ndarray | None:
    row_labels = pattern["rows"]
    col_labels = pattern["cols"]
    row_positions = _sample_axis_positions(height, len(set(row_labels)), AXIS_GAP)
    col_positions = _sample_axis_positions(width, len(set(col_labels)), AXIS_GAP)
    if row_positions is None or col_positions is None:
        return None

    row_map = {label: pos for label, pos in zip(sorted(set(row_labels)), row_positions)}
    col_map = {label: pos for label, pos in zip(sorted(set(col_labels)), col_positions)}
    grid = np.zeros((height, width), dtype=int)

    centers = []
    for row_label, col_label in zip(row_labels, col_labels):
        r = row_map[row_label]
        c = col_map[col_label]
        centers.append((r, c))
        grid[r - 1 : r + 2, c - 1 : c + 2] = BLOCK_COLOR

    blocks = _find_blocks(grid)
    if len(blocks) != len(centers):
        return None
    _validate_blocks(blocks, grid.shape)
    return grid


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    blocks = _find_blocks(inp)
    _validate_blocks(blocks, inp.shape)
    predicted = main(inp)
    assert np.array_equal(out, predicted)

    row_centers = {block["center"][0] for block in blocks}
    col_centers = {block["center"][1] for block in blocks}
    for r in row_centers:
        gray_cells = np.sum(out[r, :] == BLOCK_COLOR)
        assert gray_cells <= out.shape[1] - len(col_centers) + 1, "cross reset should remove some row-column intersections"

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        body_colors = set(map(int, np.unique(out[r0 : r1 + 1, c0 : c1 + 1])))
        assert len(body_colors) == 1 and next(iter(body_colors)) in COUNT_TO_COLOR.values()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pattern = _choose_pattern()
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        grid = _build_grid_from_pattern(pattern, height, width)
        if grid is None:
            continue
        counts = _pattern_counts(pattern)
        if len(set(counts)) <= 1:
            continue
        if not set(counts) & {1, 4} and random.random() < 0.5:
            continue
        out = main(grid)
        _semantic_check(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid cross-count input")
