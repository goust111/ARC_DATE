import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, draw_line


MARKER_COLORS = (Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
GRID_SIZES = (9, 12, 15, 18, 21, 24, 27, 30)
MAX_GENERATION_ATTEMPTS = 4000
SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _center_distance(size: int, r: int, c: int) -> int:
    mid = size - 1
    return abs(2 * int(r) - mid) + abs(2 * int(c) - mid)


def _validate_marker_layout(markers: list[tuple[int, int, int]], size: int) -> None:
    assert 2 <= len(markers) <= 4, "expected 2 to 4 markers"
    rows = set()
    cols = set()
    colors = set()
    dists = set()
    for r, c, color in markers:
        assert 0 < r < size - 1 and 0 < c < size - 1, "markers must stay off the border"
        assert color in MARKER_COLORS, "unexpected marker color"
        assert r not in rows, "markers must have distinct rows"
        assert c not in cols, "markers must have distinct columns"
        assert color not in colors, "markers must have distinct colors"
        rows.add(int(r))
        cols.add(int(c))
        colors.add(int(color))
        dists.add(_center_distance(size, r, c))
    assert len(dists) == len(markers), "marker center distances must be distinct"
    for i, (r0, c0, _color0) in enumerate(markers):
        for r1, c1, _color1 in markers[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "markers must be isolated pixels"


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square grid"
    assert 9 <= height <= 30 and height % 3 == 0, "expected a square size divisible by 3"

    coords = np.argwhere(g != Color.BLACK)
    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    _validate_marker_layout(markers, height)
    return sorted(markers)


def _cross_mask(shape: tuple[int, int], r: int, c: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    draw_line(canvas, r=r, c=0, direction=(0, 1), color=1)
    draw_line(canvas, r=0, c=c, direction=(1, 0), color=1)
    return canvas.astype(bool)


def _color_masks(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> dict[int, np.ndarray]:
    return {int(color): _cross_mask(shape, r, c) for r, c, color in markers}


def _render_cleared_crosses(shape: tuple[int, int], color_masks: dict[int, np.ndarray]) -> np.ndarray:
    cover_count = np.zeros(shape, dtype=int)
    first_color = np.zeros(shape, dtype=int)
    for color, mask in color_masks.items():
        first_color[(cover_count == 0) & mask] = int(color)
        cover_count += mask.astype(int)
    out = np.zeros(shape, dtype=int)
    out[cover_count == 1] = first_color[cover_count == 1]
    return out


def _pick_block_winner(counts: dict[int, int], distance_by_color: dict[int, int]) -> tuple[int, bool]:
    if not counts:
        return Color.BLACK, False
    best = max(counts.values())
    tied = [int(color) for color, count in counts.items() if count == best]
    if len(tied) == 1:
        return tied[0], False
    min_dist = min(distance_by_color[color] for color in tied)
    winners = [color for color in tied if distance_by_color[color] == min_dist]
    assert len(winners) == 1, "tie-break by center distance must be unique"
    return winners[0], True


def _compress_grid(grid: np.ndarray, distance_by_color: dict[int, int]) -> tuple[np.ndarray, int, int]:
    size = int(grid.shape[0])
    block = size // 3
    out = np.zeros((3, 3), dtype=int)
    tie_blocks = 0
    strict_blocks = 0
    for br in range(3):
        for bc in range(3):
            sub = grid[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            values, counts = np.unique(sub[sub != Color.BLACK], return_counts=True)
            block_counts = {int(value): int(count) for value, count in zip(values, counts)}
            winner, used_tie = _pick_block_winner(block_counts, distance_by_color)
            if winner != Color.BLACK:
                if used_tie:
                    tie_blocks += 1
                else:
                    strict_blocks += 1
            out[br, bc] = int(winner)
    return out, tie_blocks, strict_blocks


def _compress_preclear_masks(
    color_masks: dict[int, np.ndarray], size: int, distance_by_color: dict[int, int]
) -> np.ndarray:
    block = size // 3
    out = np.zeros((3, 3), dtype=int)
    for br in range(3):
        for bc in range(3):
            block_counts = {}
            rs = slice(br * block, (br + 1) * block)
            cs = slice(bc * block, (bc + 1) * block)
            for color, mask in color_masks.items():
                count = int(mask[rs, cs].sum())
                if count:
                    block_counts[int(color)] = count
            winner, _used_tie = _pick_block_winner(block_counts, distance_by_color)
            out[br, bc] = int(winner)
    return out


def _solve(markers: list[tuple[int, int, int]], size: int) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    distance_by_color = {int(color): _center_distance(size, r, c) for r, c, color in markers}
    color_masks = _color_masks((size, size), markers)
    cleared = _render_cleared_crosses((size, size), color_masks)
    compressed, tie_blocks, strict_blocks = _compress_grid(cleared, distance_by_color)
    preclear_compressed = _compress_preclear_masks(color_masks, size, distance_by_color)
    return cleared, compressed, tie_blocks, strict_blocks, preclear_compressed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw full-length colored crosses from isolated markers, clear cross-color overlaps, then compress to a 3x3 code.

    Input structure:
    - The grid is a square black canvas whose side length is divisible by 3.
    - There are 2 to 4 isolated non-black marker pixels, one per color, chosen from red/green/yellow/orange.
    - No two markers share a row or column, and none lies on the outer border.

    Rule:
    - For each marker, draw a full-width horizontal line and a full-height vertical line in that marker's color.
    - Any cell covered by crosses from two or more different markers is reset to black.
    - Split the cleared canvas into 3x3 equal coarse blocks.
    - Each coarse block outputs black if empty; otherwise output the most frequent non-black color in the block, breaking count ties by the marker whose original pixel is closer to the grid center in Manhattan distance.

    Key insight:
    - The answer depends on simulating the cross overlaps first, because clearing those intersections changes the coarse block color counts that decide the final 3x3 grid.
    """
    g = np.asarray(grid, dtype=int)
    markers = _extract_markers(g)
    _cleared, compressed, _tie_blocks, _strict_blocks, _preclear = _solve(markers, int(g.shape[0]))
    return compressed


def _marker_only_compression(grid: np.ndarray, distance_by_color: dict[int, int]) -> np.ndarray:
    return _compress_grid(grid, distance_by_color)[0]


def _random_markers(size: int, count: int) -> list[tuple[int, int, int]]:
    colors = random.sample(MARKER_COLORS, count)
    for _ in range(200):
        rows = random.sample(range(1, size - 1), count)
        cols = random.sample(range(1, size - 1), count)
        markers = [(int(r), int(c), int(color)) for (r, c, color) in zip(rows, cols, colors)]
        try:
            _validate_marker_layout(markers, size)
        except AssertionError:
            continue
        return markers
    raise RuntimeError("failed to place isolated markers")


def _candidate_input() -> np.ndarray:
    size = random.choice(GRID_SIZES)
    count = random.choice((2, 3, 4))
    markers = _random_markers(size, count)
    grid = np.zeros((size, size), dtype=int)
    for r, c, color in markers:
        grid[r, c] = int(color)
    return grid


def _accept_candidate(grid: np.ndarray, want_zero_block: bool) -> bool:
    markers = _extract_markers(grid)
    size = int(grid.shape[0])
    distance_by_color = {int(color): _center_distance(size, r, c) for r, c, color in markers}
    cleared, compressed, tie_blocks, strict_blocks, preclear_compressed = _solve(markers, size)

    if np.array_equal(compressed, preclear_compressed):
        return False
    if tie_blocks == 0 or strict_blocks == 0:
        return False
    if len(set(map(int, compressed.flat)) - {Color.BLACK}) < 2:
        return False
    if want_zero_block and Color.BLACK not in compressed:
        return False
    if np.array_equal(compressed, _marker_only_compression(grid, distance_by_color)):
        return False
    if np.count_nonzero(cleared) <= len(markers):
        return False
    if not np.array_equal(main(grid), compressed):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        want_zero_block = random.random() < 0.35
        grid = _candidate_input()
        in_sig = _grid_signature(grid)
        if in_sig in SEEN_INPUT_SIGS:
            continue
        if not _accept_candidate(grid, want_zero_block):
            continue
        out = main(grid)
        out_sig = _grid_signature(out)
        if out_sig in SEEN_OUTPUT_SIGS:
            continue
        SEEN_INPUT_SIGS.add(in_sig)
        SEEN_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid cross-clear-compress input")
