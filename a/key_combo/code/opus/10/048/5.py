from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BLOCK_SIZE = 3
COMPACT_GAP = 1
OUTPUT_MARGIN = 1
INPUT_GAP_MIN = 3
INPUT_GAP_MAX = 8
JITTER_VALUES = (-1, 0, 1)
LAYOUTS = ((2, 2), (2, 3), (3, 2))
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400
SENTINEL = -1


def _solid_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    blocks: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (BLOCK_SIZE, BLOCK_SIZE):
            continue
        if int(comp["area"]) != BLOCK_SIZE * BLOCK_SIZE:
            continue
        color = int(comp["color"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == color):
            continue
        blocks.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 1, c0 + 1),
            }
        )

    colors = [block["color"] for block in blocks]
    assert 4 <= len(blocks) <= 6, "expected 4-6 solid 3x3 blocks"
    assert len(set(colors)) == len(colors), "each block color must be distinct"
    return blocks


def _cluster_axis(values: list[int]) -> list[tuple[int, int, float]]:
    ordered = sorted(int(v) for v in values)
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= 2:
            groups[-1].append(value)
        else:
            groups.append([value])
    assert 2 <= len(groups) <= 3, "expected 2-3 row/column groups"
    return [(min(group), max(group), float(sum(group)) / len(group)) for group in groups]


def _group_index(value: int, groups: list[tuple[int, int, float]]) -> int:
    candidates = []
    for idx, (lo, hi, mean) in enumerate(groups):
        inside = lo - 1 <= value <= hi + 1
        candidates.append((0 if inside else 1, abs(value - mean), idx))
    best = min(candidates)
    _, dist, idx = best
    assert dist <= 2.0, "component is too far from every inferred row/column band"
    return int(idx)


def _recover_layout(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    blocks = _solid_blocks(g, bg)

    row_groups = _cluster_axis([block["center"][0] for block in blocks])
    col_groups = _cluster_axis([block["center"][1] for block in blocks])
    rows = len(row_groups)
    cols = len(col_groups)
    assert rows * cols == len(blocks), "expected a full 2x2, 2x3, or 3x2 layout"

    layout = np.full((rows, cols), SENTINEL, dtype=int)
    for block in blocks:
        ri = _group_index(int(block["center"][0]), row_groups)
        ci = _group_index(int(block["center"][1]), col_groups)
        assert layout[ri, ci] == SENTINEL, "two blocks mapped to the same compact slot"
        layout[ri, ci] = int(block["color"])

    assert np.all(layout != SENTINEL), "missing slot in recovered compact layout"
    return bg, layout


def _compact_shape(rows: int, cols: int) -> tuple[int, int]:
    return (
        2 * OUTPUT_MARGIN + rows * BLOCK_SIZE + (rows - 1) * COMPACT_GAP,
        2 * OUTPUT_MARGIN + cols * BLOCK_SIZE + (cols - 1) * COMPACT_GAP,
    )


def _render_output(layout: np.ndarray, bg: int) -> np.ndarray:
    rows, cols = map(int, layout.shape)
    out = np.full(_compact_shape(rows, cols), int(bg), dtype=int)

    for ri in range(rows):
        for ci in range(cols):
            r0 = OUTPUT_MARGIN + ri * (BLOCK_SIZE + COMPACT_GAP)
            c0 = OUTPUT_MARGIN + ci * (BLOCK_SIZE + COMPACT_GAP)
            out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(layout[ri, ci])

    for ri in range(rows):
        for ci in range(cols - 1):
            left = int(layout[ri, ci])
            right = int(layout[ri, ci + 1])
            r0 = OUTPUT_MARGIN + ri * (BLOCK_SIZE + COMPACT_GAP)
            gap_c = OUTPUT_MARGIN + ci * (BLOCK_SIZE + COMPACT_GAP) + BLOCK_SIZE
            out[r0, gap_c] = left
            out[r0 + 2, gap_c] = right

    for ri in range(rows - 1):
        for ci in range(cols):
            top = int(layout[ri, ci])
            bottom = int(layout[ri + 1, ci])
            gap_r = OUTPUT_MARGIN + ri * (BLOCK_SIZE + COMPACT_GAP) + BLOCK_SIZE
            c0 = OUTPUT_MARGIN + ci * (BLOCK_SIZE + COMPACT_GAP)
            out[gap_r, c0] = top
            out[gap_r, c0 + 2] = bottom

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compact the scattered 3x3 blocks into a tight grid, then half-fill each new one-cell gap from both sides.

    Input structure:
    - A black-dominant background contains 4-6 solid 3x3 monochrome blocks.
    - The block colors are distinct, and the blocks form a recoverable 2x2, 2x3, or 3x2 row/column arrangement by their center coordinates.

    Rule:
    - Infer the original row order and column order from the block centers, then compact the blocks into a grid with a 1-cell outer border and exactly 1-cell gaps between neighboring blocks.
    - For each horizontal block pair in the compact grid, fill the gap column's top cell with the left block's color and bottom cell with the right block's color, leaving the middle cell black.
    - For each vertical block pair, fill the gap row's left cell with the upper block's color and right cell with the lower block's color, leaving the middle cell black; diagonal corner gap cells stay black.

    Key insight:
    - The colors that fill the gaps are determined by the neighbors created after compaction, so the solver must recover the compact row/column structure before any gap-filling rule can be applied.
    """
    bg, layout = _recover_layout(np.asarray(grid, dtype=int))
    return _render_output(layout, bg)


def _sample_axis(group_count: int) -> tuple[list[int], list[int], int]:
    for _ in range(100):
        start = random.randint(2, 6)
        end = random.randint(2, 6)
        gaps = [random.randint(INPUT_GAP_MIN, INPUT_GAP_MAX) for _ in range(group_count - 1)]
        positions = [start]
        for gap in gaps:
            positions.append(positions[-1] + BLOCK_SIZE + gap)
        size = positions[-1] + BLOCK_SIZE + end
        if size <= 30:
            return positions, gaps, size
    raise RuntimeError("failed to sample a valid axis")


def _jitter_matrix(rows: int, cols: int) -> np.ndarray:
    return np.array(
        [[random.choice(JITTER_VALUES) for _ in range(cols)] for _ in range(rows)],
        dtype=int,
    )


def _has_irregularity(
    row_gaps: list[int],
    col_gaps: list[int],
    row_jitter: np.ndarray,
    col_jitter: np.ndarray,
) -> bool:
    return (
        len(set(row_gaps)) > 1
        or len(set(col_gaps)) > 1
        or np.any(row_jitter != 0)
        or np.any(col_jitter != 0)
    )


def _render_input(
    layout: np.ndarray,
    bg: int,
    row_positions: list[int],
    col_positions: list[int],
    row_jitter: np.ndarray,
    col_jitter: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    rows, cols = map(int, layout.shape)
    grid = np.full((height, width), int(bg), dtype=int)
    for ri in range(rows):
        for ci in range(cols):
            r0 = int(row_positions[ri] + row_jitter[ri, ci])
            c0 = int(col_positions[ci] + col_jitter[ri, ci])
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(layout[ri, ci])
    return grid


def _semantic_check(inp: np.ndarray, expected: np.ndarray, rows: int, cols: int) -> bool:
    bg, recovered = _recover_layout(inp)
    if recovered.shape != (rows, cols):
        return False
    if expected.shape != _compact_shape(rows, cols):
        return False
    if not np.array_equal(_render_output(recovered, bg), expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    row_positions, row_gaps, height = _sample_axis(rows)
    col_positions, col_gaps, width = _sample_axis(cols)

    row_jitter = _jitter_matrix(rows, cols)
    col_jitter = _jitter_matrix(rows, cols)
    if not _has_irregularity(row_gaps, col_gaps, row_jitter, col_jitter):
        return None

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], rows * cols)
    layout = np.array(colors, dtype=int).reshape(rows, cols)

    inp = _render_input(
        layout,
        bg,
        row_positions,
        col_positions,
        row_jitter,
        col_jitter,
        height,
        width,
    )
    expected = _render_output(layout, bg)

    if not _semantic_check(inp, expected, rows, cols):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compact-gap-halving puzzle")
