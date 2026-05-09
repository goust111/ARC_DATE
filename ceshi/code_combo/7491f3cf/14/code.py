# Combo variant of 7491f3cf (index 14)
# combo_spec: D1/2+D14/2
# Name: 棋盘格分割+无对称性约束

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_ROOT = None
for _parent in _THIS_FILE.parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        _ROOT = _parent
        break
assert _ROOT is not None, "could not locate repo root containing solvers/arc_utils.py"
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for combo D1/2 + D14/2
# a) Concrete behavior / trace
#    - Original 7491f3cf: tile 0 contained a 5-pixel wall plus one detached indicator.
#      The solver flood-filled one side of the wall and copied tile 1 there, while the
#      opposite side used tile 2.
#    - Fused variant: tile 0 is now a full 5x5 two-color checkerboard. There is no wall,
#      no enclosed side, and no flood fill. Instead, every output pixel consults the color
#      shown at the same coordinate in tile 0: one checker color means "copy from tile 1",
#      the other means "copy from tile 2".
#    - The mapping from checker colors to sources is not fixed. A single border indicator
#      pixel above tile 0 uses one of the two checker colors; whichever checker color matches
#      that indicator becomes the tile-1 class. The other checker color becomes the tile-2
#      class. Example: if tile 0 alternates orange/teal and the lone top-border indicator is
#      teal, then every teal square in tile 0 pulls from tile 1 and every orange square pulls
#      from tile 2.
#    - Because tile 1 and tile 2 are no longer symmetric, the output becomes a visibly
#      interleaved mosaic of two arbitrary source masks rather than two mirrored half-regions.
# b) Complexity calibration
#    - The spatial rule is still low-entropy: there are only two repeated checker classes,
#      and one repeated indicator mechanism. This is enough support for human inference from
#      a few examples.
#    - The generator keeps tile 1 / tile 2 as single-foreground-color sparse masks, so the
#      main novelty is the alternating merge rule, not color chaos.
#    - To avoid trivial samples, generation rejects cases where swapping the indicator color
#      would barely change the output, or where one checker class contributes almost nothing.
# c) Original-code dependency audit
#    - "tile 0 has a background plus cue pixels": invalid now. Tile 0 is fully occupied by a
#      two-color checkerboard and has no tile-local background role.
#    - "there is a unique 8-isolated indicator inside tile 0": invalid now. The indicator has
#      moved to a fixed border cell outside tile 0.
#    - "cue pixels form a wall through the center": invalid now; removed completely.
#    - "flood-fill chooses the tile-1 side": invalid now; replaced by per-cell checker lookup.
#    - "wall cells have a special fallback from tile 2": invalid now because there is no wall.
# d) Placement legality
#    - The global canvas stays 7x25 with the same 5x5 panel boxes at columns [1:6], [7:12],
#      [13:18], [19:24]. The single indicator pixel occupies border cell (0,3), which does not
#      overlap any panel and stays within bounds.
#    - Tile 0 is a perfect 5x5 checkerboard using exactly two distinct non-border colors.
#    - Tile 3 input remains blank and is filled only by the solver output.
# e) Visual observability
#    - The checkerboard is directly visible in tile 0, and the lone colored border pixel is
#      visually salient. Across examples, the color of that lone marker flips which checker
#      squares inherit tile 1.
#    - Removing symmetry from tile 1 / tile 2 makes the alternating merge easy to see: odd
#      and even checker cells pull visibly different random-looking fragments.
# f) Visual neatness
#    - There are only six semantic colors at most: border, shared background, two checker
#      colors, and one foreground color for each source tile. No extra overlay layer is added
#      inside tile 0, so the cue remains crisp.
#    - The indicator sits in the border rather than on top of the checkerboard, which avoids
#      cluttering or breaking the repeated checker texture.
# g) Inferability of color changes
#    - No numeric color ordering is used. The indicator color is compared only by equality with
#      the two visible checker colors.
#    - The generator enforces that both checker classes have repeated visible evidence in the
#      output and that both source masks violate the original D1+D2 symmetry assumption.
# h) Fusion-behavior metric
#    - `swap_diff`: Hamming distance between the true output and the output obtained by swapping
#      the indicator color to the other checker color while keeping the same input tiles. This
#      measures whether the indicator-controlled checker mapping is behaviorally active.
#    - `nonsymmetric_sources`: how many of tile 1 / tile 2 violate the original double-diagonal
#      symmetry constraint. The generator enforces this to be 2 on every sample.
# i) New solver assumptions and generator guarantees
#    - Solver assumptions: tile 0 is an exact two-color checkerboard; there is exactly one
#      non-border marker in the frame at position (0,3); that marker uses one of the checker
#      colors; tile 1 / tile 2 / tile 3 share the same background color.
#    - Generator guarantees exactly those conditions, so main() is not relying on unsupported
#      guesses outside the produced distribution.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
PATTERN_RETRIES = 200
GEN_RETRIES = 400

MIN_SWAP_DIFF = 4
MIN_VISIBLE_PER_CLASS = 3
MIN_MASK_HAMMING = 6

INDICATOR_POS = (0, 3)
ALL_CELLS = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)]


def _tile_col(j: int) -> int:
    return 1 + 6 * j


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[1:6, _tile_col(j) : _tile_col(j) + TILE_SZ].copy() for j in range(4)]


def _build_checkerboard(color_even: int, color_odd: int) -> np.ndarray:
    tile = np.empty((TILE_SZ, TILE_SZ), dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            tile[r, c] = color_even if (r + c) % 2 == 0 else color_odd
    return tile


def _checker_colors(t0: np.ndarray) -> tuple[int, int]:
    values = {int(v) for v in t0.flatten()}
    assert len(values) == 2, f"tile 0 must contain exactly two checker colors, got {sorted(values)}"

    color_even = int(t0[0, 0])
    color_odd = int(t0[0, 1])
    assert color_even != color_odd, "checkerboard colors must differ"
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            expected = color_even if (r + c) % 2 == 0 else color_odd
            assert int(t0[r, c]) == expected, "tile 0 must be an exact checkerboard"
    return color_even, color_odd


def _is_double_diagonal_symmetric(tile: np.ndarray) -> bool:
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(tile[r, c])
            if int(tile[c, r]) != v:
                return False
            if int(tile[TILE_SZ - 1 - c, TILE_SZ - 1 - r]) != v:
                return False
    return True


def _compose_output_tile(
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
    indicator_color: int,
) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            source = t1 if int(t0[r, c]) == indicator_color else t2
            value = int(source[r, c])
            if value != tile_bg:
                out[r, c] = value
    return out


def _fusion_metrics_from_parts(
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
    indicator_color: int,
) -> dict[str, int]:
    checker_colors = _checker_colors(t0)
    assert indicator_color in checker_colors, "indicator color must equal one checker color"
    other_color = checker_colors[0] if checker_colors[1] == indicator_color else checker_colors[1]

    out = _compose_output_tile(t0, t1, t2, tile_bg, indicator_color)
    swapped = _compose_output_tile(t0, t1, t2, tile_bg, other_color)

    t1_visible = 0
    t2_visible = 0
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if int(t0[r, c]) == indicator_color:
                if int(t1[r, c]) != tile_bg:
                    t1_visible += 1
            else:
                if int(t2[r, c]) != tile_bg:
                    t2_visible += 1

    mask_hamming = int(np.count_nonzero((t1 != tile_bg) != (t2 != tile_bg)))
    return {
        "swap_diff": int(np.count_nonzero(out != swapped)),
        "t1_visible": int(t1_visible),
        "t2_visible": int(t2_visible),
        "mask_hamming": int(mask_hamming),
        "t1_d1d2": int(_is_double_diagonal_symmetric(t1)),
        "t2_d1d2": int(_is_double_diagonal_symmetric(t2)),
        "nonsymmetric_sources": int(
            (not _is_double_diagonal_symmetric(t1)) + (not _is_double_diagonal_symmetric(t2))
        ),
    }


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"
    t0, t1, t2, _t3 = _extract_tiles(g)
    tile_bg = int(get_bg(t1))
    indicator_color = int(g[INDICATOR_POS])
    return _fusion_metrics_from_parts(t0, t1, t2, tile_bg, indicator_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source tiles into the blank fourth tile using a checkerboard selector.

    Input structure:
    - Four 5x5 panels appear in one row inside the original 7x25 framed layout.
    - Tile 0 is a full two-color checkerboard. Tile 1 and tile 2 are arbitrary sparse source
      patterns sharing one background color. Tile 3 is blank in the input.
    - A single border pixel at global position (0,3) acts as the indicator; its color matches
      one of tile 0's two checker colors.

    Rule:
    - Read the indicator color from the border marker.
    - For each coordinate (r,c) in the 5x5 destination tile, compare tile 0[r,c] with the
      indicator color. Matching checker cells copy the non-background pixel from tile 1;
      non-matching checker cells copy the non-background pixel from tile 2.
    - If the chosen source has background at that coordinate, the destination keeps background.

    Key insight:
    - Tile 0 no longer divides space into two continuous regions. It is a repeated per-pixel
      routing mask, and the lone border marker tells which checker color means "take tile 1".
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"

    border = int(g[0, 0])
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            is_frame = r in {0, GRID_ROWS - 1} or c % 6 == 0 or c == GRID_COLS - 1
            if not is_frame:
                continue
            if (r, c) == INDICATOR_POS:
                continue
            assert int(g[r, c]) == border, "frame must be uniform except for the single indicator pixel"

    t0, t1, t2, t3_in = _extract_tiles(g)
    tile_bg = int(get_bg(t1))
    assert border != tile_bg, "border color must differ from tile background"
    assert int(get_bg(t2)) == tile_bg, "tile 1 and tile 2 must share a background color"
    assert np.all(t3_in == tile_bg), "tile 3 input must be blank"

    checker_colors = _checker_colors(t0)
    indicator_color = int(g[INDICATOR_POS])
    assert indicator_color in checker_colors, "indicator color must match one checker color"
    assert indicator_color != border, "indicator must differ from border color"

    out3 = _compose_output_tile(t0, t1, t2, tile_bg, indicator_color)

    out = g.copy()
    out[1:6, _tile_col(3) : _tile_col(3) + TILE_SZ] = out3
    return out


def _sample_free_pattern(
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        count = rng.randint(min_px, max_px)
        coords = rng.sample(ALL_CELLS, count)
        tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        for r, c in coords:
            tile[r, c] = fg
        if _is_double_diagonal_symmetric(tile):
            continue
        return tile
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    border, tile_bg, checker_a, checker_b, t1_fg, t2_fg = rng.sample(range(NUM_COLORS), 6)

    if rng.random() < 0.5:
        t0 = _build_checkerboard(checker_a, checker_b)
    else:
        t0 = _build_checkerboard(checker_b, checker_a)
    indicator_color = rng.choice([checker_a, checker_b])

    t1 = _sample_free_pattern(tile_bg, t1_fg, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_free_pattern(tile_bg, t2_fg, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    metrics = _fusion_metrics_from_parts(t0, t1, t2, tile_bg, indicator_color)
    if metrics["nonsymmetric_sources"] != 2:
        return None
    if metrics["swap_diff"] < MIN_SWAP_DIFF:
        return None
    if metrics["t1_visible"] < MIN_VISIBLE_PER_CLASS or metrics["t2_visible"] < MIN_VISIBLE_PER_CLASS:
        return None
    if metrics["mask_hamming"] < MIN_MASK_HAMMING:
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    out3 = _compose_output_tile(t0, t1, t2, tile_bg, indicator_color)

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, _tile_col(j) : _tile_col(j) + TILE_SZ] = tile
    g[INDICATOR_POS] = indicator_color

    expected = g.copy()
    expected[1:6, _tile_col(3) : _tile_col(3) + TILE_SZ] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
