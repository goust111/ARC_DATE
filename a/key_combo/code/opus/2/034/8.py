from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


PLACEHOLDER = int(Color.GREY)
DEFECT_MARK = int(Color.ORANGE)
INPUT_PALETTE = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))
TILE_SIZES = (3, 4)
GENERATION_RETRIES = 300


def _periodic_grid(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    rows = np.arange(shape[0], dtype=int)[:, None] % h
    cols = np.arange(shape[1], dtype=int)[None, :] % w
    return tile[rows, cols]


def _majority_color(values: np.ndarray) -> tuple[int, int]:
    counts = np.bincount(values.astype(int), minlength=10)
    return int(np.argmax(counts)), int(counts.max())


def _candidate_tile(visible: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    tile = np.zeros((size, size), dtype=int)
    agreement = 0
    for r in range(size):
        for c in range(size):
            values = visible[r::size, c::size].ravel()
            color, count = _majority_color(values)
            tile[r, c] = color
            agreement += count
    return tile, int(agreement)


def _infer_tile(visible: np.ndarray) -> np.ndarray:
    candidates: list[tuple[int, int, np.ndarray]] = []
    for size in TILE_SIZES:
        if size > visible.shape[0] or size > visible.shape[1]:
            continue
        tile, agreement = _candidate_tile(visible, size)
        mismatches = int(visible.size - agreement)
        candidates.append((mismatches, size, tile))
    assert candidates, "expected a visible region large enough for a 3x3 or 4x4 tile"
    _mismatches, _size, tile = min(candidates, key=lambda item: (item[0], item[1]))
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the clean repeating tile, fill the gray placeholder, and mark visible color defects in orange.

    Input structure:
    - The grid contains a top-left visible rectangle made from a repeated square tile that uses 2 or 3 colors from blue/red/green/yellow.
    - Gray cells form a bottom strip, a right strip, or both, so the placeholder region is an L-shaped mask outside the visible rectangle.
    - A few visible pixels have been recolored to the wrong in-palette color, but every other visible pixel follows the same periodic tile.

    Rule:
    - Recover the ideal 3x3 or 4x4 tile by majority voting across all visible cells that share the same position modulo the tile size.
    - Periodically extend that ideal tile across the whole canvas and replace every gray placeholder cell with its expected tile color.
    - Compare the original visible cells against the ideal tile and repaint every mismatch orange.
    - Keep visible cells that already match the ideal tile unchanged.

    Key insight:
    - The wrong colors are still legal palette colors, so the tile cannot be recovered from local anomalies; it must be reconstructed globally by cross-instance majority voting, then reused both for completion and for defect detection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    visible_mask = g != PLACEHOLDER
    assert np.any(visible_mask), "expected a non-placeholder region"
    visible_box = bbox(visible_mask)
    assert visible_box is not None
    visible = crop_bbox(g, visible_box)

    tile = _infer_tile(visible)
    expected = _periodic_grid(g.shape, tile)

    out = expected.copy()
    defect_mask = visible_mask & (g != expected)
    out[defect_mask] = DEFECT_MARK
    return out


def _exact_period(tile: np.ndarray) -> tuple[int, int]:
    full_h, full_w = tile.shape
    best = (full_h, full_w)
    best_area = full_h * full_w
    for h in range(1, full_h + 1):
        for w in range(1, full_w + 1):
            candidate = _periodic_grid((full_h, full_w), tile[:h, :w])
            if np.array_equal(candidate, tile):
                area = h * w
                if area < best_area or (area == best_area and (h, w) < best):
                    best = (h, w)
                    best_area = area
    return best


def _sample_tile(rng: np.random.Generator, size: int, palette: tuple[int, ...]) -> np.ndarray:
    for _ in range(200):
        tile = rng.choice(np.array(palette, dtype=int), size=(size, size), replace=True)
        if len(np.unique(tile)) != len(palette):
            continue
        if _exact_period(tile) != (size, size):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a tile with the requested exact period")


def _sample_visible_extent(rng: np.random.Generator, size: int) -> tuple[int, int, int, int]:
    while True:
        full_rows = int(rng.integers(2, 5))
        full_cols = int(rng.integers(2, 5))
        if full_rows * full_cols >= 6:
            break
    extra_rows = int(rng.integers(0, size))
    extra_cols = int(rng.integers(0, size))
    visible_h = full_rows * size + extra_rows
    visible_w = full_cols * size + extra_cols
    return full_rows, full_cols, visible_h, visible_w


def _placeholder_padding(rng: np.random.Generator, size: int) -> tuple[int, int]:
    branch = int(rng.integers(0, 3))
    bottom = int(rng.integers(1, size + 2)) if branch in (0, 2) else 0
    right = int(rng.integers(1, size + 2)) if branch in (1, 2) else 0
    return bottom, right


def _inject_defects(
    rng: np.random.Generator,
    grid: np.ndarray,
    tile: np.ndarray,
    palette: tuple[int, ...],
    full_rows: int,
    full_cols: int,
) -> np.ndarray:
    size = tile.shape[0]
    defect_count = int(rng.integers(2, 5))
    tile_instances = [(tr, tc) for tr in range(full_rows) for tc in range(full_cols)]
    chosen_instances = rng.choice(len(tile_instances), size=defect_count, replace=False)
    chosen_offsets = rng.choice(size * size, size=defect_count, replace=False)

    defect_mask = np.zeros_like(grid, dtype=bool)
    for inst_index, offset_index in zip(chosen_instances, chosen_offsets):
        tr, tc = tile_instances[int(inst_index)]
        dr, dc = divmod(int(offset_index), size)
        r = tr * size + dr
        c = tc * size + dc
        original = int(grid[r, c])
        alternatives = [color for color in palette if color != original]
        grid[r, c] = int(rng.choice(alternatives))
        defect_mask[r, c] = True
    return defect_mask


def _candidate_mismatches(visible: np.ndarray, size: int) -> int:
    tile, agreement = _candidate_tile(visible, size)
    _ = tile
    return int(visible.size - agreement)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.choice(np.array(TILE_SIZES, dtype=int)))
    palette_size = int(rng.integers(2, 4))
    palette = tuple(int(color) for color in rng.choice(np.array(INPUT_PALETTE, dtype=int), size=palette_size, replace=False))
    tile = _sample_tile(rng, size, palette)

    full_rows, full_cols, visible_h, visible_w = _sample_visible_extent(rng, size)
    bottom_pad, right_pad = _placeholder_padding(rng, size)
    height = visible_h + bottom_pad
    width = visible_w + right_pad
    if height > 30 or width > 30:
        return None

    full = _periodic_grid((height, width), tile)
    grid = full.copy()
    placeholder_mask = np.zeros_like(grid, dtype=bool)
    if bottom_pad:
        placeholder_mask[visible_h:, :] = True
    if right_pad:
        placeholder_mask[:, visible_w:] = True
    grid[placeholder_mask] = PLACEHOLDER

    defect_mask = _inject_defects(rng, grid, tile, palette, full_rows, full_cols)
    visible = grid[:visible_h, :visible_w]

    score_by_size = {
        candidate_size: _candidate_mismatches(visible, candidate_size)
        for candidate_size in TILE_SIZES
        if candidate_size <= visible.shape[0] and candidate_size <= visible.shape[1]
    }
    best_score = min(score_by_size.values())
    if score_by_size[size] != best_score:
        return None
    if any(candidate_size != size and score == best_score for candidate_size, score in score_by_size.items()):
        return None

    expected = full.copy()
    expected[defect_mask] = DEFECT_MARK
    if not np.array_equal(main(grid), expected):
        return None
    return grid.astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
