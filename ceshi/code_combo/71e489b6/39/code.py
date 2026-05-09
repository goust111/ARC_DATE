# Combo variant of 71e489b6 (index 39)
# combo_spec: D6/0+D7/0
# Name: 不规则多边形区域+光晕限蓝色区域

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for combo D6/0 + D7/0
# a) Concrete behavior and trace
#    - Original 71e489b6 has three stages: absorb blue spur pixels inside black territory, detect black
#      anomalies (size<=3 components and leaf pixels of large black components), then paint an orange
#      Chebyshev-1 halo around every anomaly.
#    - D6/0 changes the carrier layout. Instead of axis-aligned rectangles or stripes, the blue territory is
#      a single large staircase band whose left/right border (or top/bottom border after transposition)
#      moves in orthogonal steps from block to block:
#          rectangular stripe            staircase stripe
#          00011111000                  00011111000
#          00011111000        ->        00111111000
#          00011111000                  00111110000
#          00011111000                  00011110000
#      This still partitions the grid into a few large regions, but the blue boundary now has visible bends.
#    - D7/0 changes only the painting phase. Example around a near-boundary defect:
#          clean base                    original halo             fused halo
#          0 1 1                         7 7 7                     0 7 7
#          1 0 1            defect ->   7 0 7         clipped ->  7 0 7
#          1 1 1                         7 7 7                     7 7 7
#      The target pixel stays black in both cases; the difference is that orange may no longer spill into
#      black territory. The irregular staircase border makes that clipping happen on diagonals and notches,
#      not just on flat straight edges.
# b) Complexity calibration
#    - A plain rectangle would make the D7/0 clipping easy but visually close to the source task, so the
#      generator forces at least two boundary shifts and several row blocks. That keeps the D6/0 change
#      visible.
#    - A wildly jagged one-cell zig-zag would create accidental leaf pixels in the large black background and
#      make the scene noisy. The staircase blocks are therefore at least two cells tall, and boundary shifts
#      are only a few cells at a time.
#    - Accepted boards always contain both small enclosed defects and at least one leaf protrusion, so the
#      human sees that the source task's anomaly taxonomy is still active under the new boundary rule.
# c) Hidden assumptions audit
#    - Original STEP 0 mutates the grid in scan order while absorbing blue spurs. That assumption still holds,
#      so the fused solver keeps the same in-place absorption behavior.
#    - Original halo painting assumed "any in-bounds neighbor may become orange." D7/0 breaks that; the new
#      painter must additionally require that the destination cell belongs to the surviving blue region.
#    - Original generator could rely on smooth rectangular macro-regions having no accidental leaf pixels.
#      D6/0 weakens that assumption, so generation constrains the staircase geometry to avoid one-cell black
#      tendrils and validates component sizes after construction.
# d) Placement legality
#    - The staircase band always leaves at least a two-cell black margin on both sides, so the black macro-
#      regions stay large and disconnected from each other.
#    - Defects are placed only on strict blue interiors: every non-defect 4-neighbor of every defect pixel is
#      blue before the defect is carved.
#    - Leaves are inserted only on blue cells with exactly one black 4-neighbor, guaranteeing that they become
#      true leaf pixels of a large black component instead of bridges or bulk growth.
#    - Spurs are inserted only on black cells whose effective black 4-neighbor count is already at least three,
#      and they are kept separated so the in-place absorption pass remains valid.
# e) Visual observability
#    - Each accepted sample contains at least one defect placed at a diagonal-exposed interior cell near a
#      staircase bend. On the same input, the original solver would paint extra orange onto black cells there,
#      while the fused solver clips those cells away. That makes the D7/0 change directly visible.
#    - Because the macro layout is a single bent blue stripe splitting two large black territories, the D6/0
#      geometry is readable in 2-3 examples without relying on color ids or scan order.
# f) Visual cleanliness
#    - The palette remains {0,1,7}. Orange has exactly one meaning: "blue cell within Chebyshev distance 1 of
#      a detected black anomaly."
#    - Defects and leaves are spaced apart with Chebyshev buffers so their halos do not collapse into a full
#      orange blur. The staircase is irregular but coarse, which keeps the partition legible.
# g) Quantified anti-degeneracy target
#    - After adding spurs, generation compares the fused output against the source-task output on the same
#      input. We reject the sample unless at least `MIN_CLIPPED_CELLS` orange cells that the source solver
#      would have painted are clipped away by the fused blue-only halo rule.
# h) New solver assumptions vs generator guarantees
#    - `main()` now assumes that the intended halo canvas can be reconstructed as `base == BLUE` after spur
#      absorption. `generate_input()` therefore inserts spurs only after fixing the clean base/output pair.
#    - `main()` also assumes all non-target black components larger than three cells are genuine macro-regions,
#      not accidental tiny staircase artifacts. `_build_partition()` and `_validate_partition()` enforce that.


BLUE = 1
BLACK = 0
ORANGE = 7

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 60
MIN_BLACK_REGION = 20

BLOCK_COUNT_RANGE = (4, 5)
BLOCK_MIN_HEIGHT = 2
MIN_SIDE_MARGIN = 2
MIN_BAND_WIDTH = 8
STEP_SHIFTS = (-2, -1, 0, 1, 2)
MIN_BOUNDARY_SHIFTS = 2
MIN_DISTINCT_BOUNDS = 3
MIN_DIAGONAL_INTERIOR = 4

DEFECT_COUNT_RANGE = (3, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 80
MIN_DEFECTS = 3
MIN_CLIPPED_DEFECTS = 1
MAX_CLIPPED_DEFECTS = 2

LEAF_COUNT_RANGE = (1, 2)
LEAF_CLEARANCE = 2
MIN_LEAVES = 1

SPUR_COUNT_RANGE = (2, 5)
SPUR_CLEARANCE = 1
SPUR_BUFFER = 2
MIN_SPURS = 2

MIN_CLIPPED_CELLS = 4
MAX_GEN_ATTEMPTS = 600

DEFECT_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (-1, 0), (-2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, -1), (0, -2)),
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _random_segments(total: int, parts: int, min_size: int) -> list[int]:
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _find_interior_blue(
    grid: np.ndarray,
    *,
    require_diagonal_black: bool = False,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLUE:
                continue
            ok = True
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != BLUE:
                    ok = False
                    break
            if not ok:
                continue
            if require_diagonal_black:
                diag_black = False
                for dr in (-1, 1):
                    for dc in (-1, 1):
                        rr = r + dr
                        cc = c + dc
                        if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) == BLACK:
                            diag_black = True
                            break
                    if diag_black:
                        break
                if not diag_black:
                    continue
            coords.append((r, c))
    random.shuffle(coords)
    return coords


def _count_diagonal_interior(grid: np.ndarray) -> int:
    return len(_find_interior_blue(grid, require_diagonal_black=True))


def _build_vertical_partition(h: int, w: int) -> np.ndarray | None:
    max_width = w - 2 * MIN_SIDE_MARGIN
    min_width = max(MIN_BAND_WIDTH, w // 2)
    if min_width > max_width:
        return None

    for _ in range(200):
        blocks = random.randint(*BLOCK_COUNT_RANGE)
        heights = _random_segments(h, blocks, BLOCK_MIN_HEIGHT)

        width = random.randint(min_width, max_width)
        left = random.randint(MIN_SIDE_MARGIN, w - width - MIN_SIDE_MARGIN)
        right = left + width - 1

        bounds = [(left, right)]
        shifts = 0
        for _block in range(1, blocks):
            found = False
            for _try in range(60):
                dl = random.choice(STEP_SHIFTS)
                dr = random.choice(STEP_SHIFTS)
                new_left = left + dl
                new_right = right + dr
                if new_left < MIN_SIDE_MARGIN or new_right > w - MIN_SIDE_MARGIN - 1:
                    continue
                if new_right - new_left + 1 < MIN_BAND_WIDTH:
                    continue
                left = new_left
                right = new_right
                bounds.append((left, right))
                if dl != 0 or dr != 0:
                    shifts += 1
                found = True
                break
            if not found:
                break

        if len(bounds) != blocks:
            continue
        if shifts < MIN_BOUNDARY_SHIFTS:
            continue
        if len(set(bounds)) < MIN_DISTINCT_BOUNDS:
            continue

        grid = np.zeros((h, w), dtype=int)
        row = 0
        for height, (b_left, b_right) in zip(heights, bounds):
            grid[row : row + height, b_left : b_right + 1] = BLUE
            row += height

        if _count_diagonal_interior(grid) < MIN_DIAGONAL_INTERIOR:
            continue
        return grid

    return None


def _build_partition(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        if random.random() < 0.5:
            grid = _build_vertical_partition(h, w)
        else:
            transposed = _build_vertical_partition(w, h)
            grid = None if transposed is None else transposed.T
        if grid is not None:
            return grid
    raise RuntimeError("failed to build staircase partition")


def _validate_partition(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if len(blue_comps) != 1:
        return False
    if len(blue_comps[0]) < MIN_BLUE_REGION:
        return False
    for comp in black_comps:
        if len(comp) < MIN_BLACK_REGION:
            return False
    return True


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return int(black_n)


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Match the source-task in-place spur absorption pass."""
    base = np.array(grid, dtype=int).copy()
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _collect_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Collect small black components and leaf pixels of larger black components."""
    h, w = grid.shape
    preserve: set[tuple[int, int]] = set()

    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            preserve.update(pts)
            continue

        comp_set = set(pts)
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                    black_n += 1
            if black_n <= 1:
                preserve.add((r, c))

    return preserve


def _paint_clipped_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    blue_mask = out == BLUE
    h, w = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in preserve:
                    continue
                if blue_mask[rr, cc]:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _paint_uncropped_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    h, w = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _original_main(grid: np.ndarray) -> np.ndarray:
    """Original 71e489b6 behavior, used for anti-degeneracy checks."""
    base = _absorb_blue_spurs(np.asarray(grid, dtype=int))
    preserve = _collect_targets(base)
    return _paint_uncropped_halo(base, preserve)


def _template_fits(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    defect_pixels: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLUE or (r, c) in defect_pixels:
            return False
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cell_set:
                continue
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                return False
    return True


def _place_defects(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    total_needed = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    defects_placed = 0

    clipped_needed = random.randint(MIN_CLIPPED_DEFECTS, min(MAX_CLIPPED_DEFECTS, total_needed))
    clipped_candidates = _find_interior_blue(grid, require_diagonal_black=True)
    clipped_placed = 0
    for r, c in clipped_candidates:
        if _nearby(r, c, DEFECT_CLEARANCE, defect_pixels):
            continue
        defect_pixels.add((r, c))
        defects_placed += 1
        clipped_placed += 1
        if clipped_placed >= clipped_needed:
            break
    if clipped_placed < clipped_needed:
        return set(), 0

    interior_blue = _find_interior_blue(grid)
    templates_by_size = {
        1: [tpl for tpl in DEFECT_TEMPLATES if len(tpl) == 1],
        2: [tpl for tpl in DEFECT_TEMPLATES if len(tpl) == 2],
        3: [tpl for tpl in DEFECT_TEMPLATES if len(tpl) == 3],
    }

    while defects_placed < total_needed:
        placed = False
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        templates = templates_by_size[defect_size][:]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randrange(len(interior_blue))]
            random.shuffle(templates)
            for template in templates:
                cells = [(seed_r + dr, seed_c + dc) for dr, dc in template]
                if any(_nearby(r, c, DEFECT_CLEARANCE, defect_pixels) for r, c in cells):
                    continue
                if _template_fits(grid, cells, defect_pixels):
                    defect_pixels.update(cells)
                    defects_placed += 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    for _ in range(n_leaves):
        large_black = [comp for comp in components_mask(grid == BLACK, connectivity=4) if len(comp) > 3]
        candidates: list[tuple[int, int]] = []
        for comp in large_black:
            comp_set = {(int(r), int(c)) for r, c in comp}
            for r, c in comp_set:
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                        continue
                    if int(grid[rr, cc]) != BLUE:
                        continue
                    if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                        continue
                    if _nearby(rr, cc, LEAF_CLEARANCE, defect_pixels | leaf_pixels):
                        continue
                    black_n = 0
                    for r2, c2 in _neighbors4(rr, cc):
                        if 0 <= r2 < grid.shape[0] and 0 <= c2 < grid.shape[1]:
                            if int(grid[r2, c2]) == BLACK or (r2, c2) in leaf_pixels:
                                black_n += 1
                    if black_n == 1:
                        candidates.append((rr, cc))
        if not candidates:
            break
        lr, lc = random.choice(candidates)
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK

    return leaf_pixels


def _build_output(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    return _paint_clipped_halo(grid, preserve)


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    black = grid == BLACK

    nb = np.zeros((h, w), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((h, w), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        cell = (int(r), int(c))
        if cell in exclude:
            continue
        if _nearby(cell[0], cell[1], SPUR_BUFFER, exclude):
            continue
        candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _place_spurs(grid: np.ndarray, exclude: set[tuple[int, int]]) -> int:
    candidates = _find_spur_candidates(grid, exclude)
    chosen: list[tuple[int, int]] = []
    target_count = random.randint(*SPUR_COUNT_RANGE)
    for r, c in candidates:
        if _nearby(r, c, SPUR_CLEARANCE, set(chosen)):
            continue
        chosen.append((r, c))
        if len(chosen) >= target_count:
            break
    for r, c in chosen:
        grid[r, c] = BLUE
    return len(chosen)


def _count_clipped_cells(inp: np.ndarray) -> int:
    fused = main(inp)
    source = _original_main(inp)
    return int(np.sum((source == ORANGE) & (fused != ORANGE)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight structural black anomalies inside a staircase-shaped blue region, but clip every orange halo so it
    stays inside blue territory.

    Input structure:
    - A large blue region forms an orthogonal staircase polygon that splits the board into a few large blue/black
      macro-regions instead of plain rectangles.
    - Small black defect components sit strictly inside blue territory, larger black regions may poke one-cell
      leaves into blue, and some black regions also contain isolated blue spur pixels.

    Rule:
    - First absorb any blue spur whose effective 4-neighborhood is mostly black, using the same corner-aware
      scan-order rule as the source task.
    - Then mark every black 4-component of size at most 3, plus every pixel of a larger black component that has
      at most one black 4-neighbor, as a defect center.
    - Paint orange (7) on the Chebyshev-1 halo of each defect only where the cleaned grid is blue; black cells,
      including the defect pixels themselves, never get recolored by the halo.

    Key insight:
    The anomaly detector is unchanged, but the output painter now uses the surviving blue region as a stencil, so
    defects near a bent staircase boundary produce visibly truncated halos.
    """
    base = _absorb_blue_spurs(np.asarray(grid, dtype=int))
    preserve = _collect_targets(base)
    return _paint_clipped_halo(base, preserve)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    grid = _build_partition(h, w)
    if not _validate_partition(grid):
        return None

    defect_pixels, defects_placed = _place_defects(grid)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    preserve = defect_pixels | leaf_pixels
    output = _build_output(grid, preserve)

    inp = grid.copy()
    spurs_placed = _place_spurs(inp, preserve)
    if spurs_placed < MIN_SPURS:
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    if _count_clipped_cells(inp) < MIN_CLIPPED_CELLS:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
