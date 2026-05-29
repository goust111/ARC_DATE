# Combo variant of 1818057f (index 19)
# combo_spec: D1/2+D3/3+D8/3
# Name: 双模式检测分解着色

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task only looked for local '+' motifs and painted all 5 cells teal.
#    Here the foreground contains two visually distinct target families at once:
#    (1) thin '+' motifs, and (2) solid 3x3 blocks. The '+' rule is also split by part:
#    the center becomes teal (8) while the four arms become magenta (6). Every solid 3x3
#    block becomes orange (7) as a whole.
#    A local comparison:
#        thin plus input          old output            combo output
#            . 4 .                 . 8 .                 . 6 .
#            4 4 4       ->        8 8 8       vs        6 8 6
#            . 4 .                 . 8 .                 . 6 .
#        solid block input         old output            combo output
#            4 4 4                 . . .                 7 7 7
#            4 4 4       ->        . 8 .       vs        7 7 7
#            4 4 4                 . . .                 7 7 7
#    The important change is that a solid 3x3 block is treated as its own motif family
#    rather than as a source of embedded '+' submatches.
# b) Complexity calibration:
#    The design uses only three functional output colors total, each tied to a stable
#    structural role repeated across samples:
#    - orange 7 for every cell of a solid 3x3 block,
#    - teal 8 for '+' centers,
#    - magenta 6 for '+' arms.
#    This keeps the rule visibly richer than the seed task without introducing arbitrary
#    per-object color choices or hidden ordering rules.
# c) Implicit assumption review:
#    - `get_bg(...)` still works because the input remains background-dominant.
#    - `colors[0]` as the sole foreground color still works because the input keeps only
#      one non-background color; extra colors appear only in the output.
#    - Local detection must read from the original grid `g`, not from partially recolored
#      output, because there are now two motif families.
#    - The old plain 4-neighbor '+' test is no longer sufficient on its own: every solid
#      3x3 contains a geometric '+' subpattern. To avoid ambiguity, thin '+' motifs are
#      defined as 4-neighbor crosses whose full 3x3 neighborhood is not all foreground.
# d) Placement legality:
#    The generator places pure 3x3 blocks and thin plus components with a one-cell halo
#    between designed motifs, so they do not overlap or accidentally merge into hybrid
#    targets. Random fill only happens outside reserved halos and is rejected whenever it
#    would create a new thin '+' or a new solid 3x3.
# e) Visual observability:
#    A human comparing a few training pairs can see two repeated transformations:
#    compact filled squares turn uniformly orange, while cross shapes split into a teal
#    center plus four magenta arms. The distinction is purely geometric and repeated.
# f) Visual cleanliness:
#    Intended motifs are spatially separated and random clutter is kept away from them.
#    This prevents block recolors and plus recolors from visually overlapping in the same
#    local patch, so the two rules remain legible.
# g) Color-change inferability:
#    The output colors are not arbitrary: shape type determines the palette, and within a
#    '+' the color split follows positional role (center versus arm). That structure is
#    directly visible from the grids and does not depend on numeric color ids.
# h) Generator validation strategy:
#    The generator builds a target-aware input, derives an expected output by painting only
#    the deliberately placed block cores and plus cores, and then validates that the solver
#    reproduces exactly that output. Any accidental extra motif causes rejection.
# i) New solver assumptions and generator support:
#    `main()` assumes thin pluses are distinguishable from solid blocks by the absence of
#    an all-foreground 3x3 neighborhood. The generator guarantees this by reserving plus
#    diagonals and by refusing random additions that would complete a surrounding 3x3.

PLUS_CENTER_COLOR = 8
PLUS_ARM_COLOR = 6
BLOCK_COLOR = 7
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _plus_cells(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _block_cells(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2))


def _is_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return bool(np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == fg))


def _is_thin_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != fg:
        return False
    if not all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
        return False
    return not _is_solid_3x3(grid, r, c, fg)


def _find_block_footprints(grid: np.ndarray, fg: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    footprints: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == fg and _is_solid_3x3(grid, r, c, fg):
                footprints.append(_block_cells(r, c))
    return footprints


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if _is_thin_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect solid 3x3 foreground blocks and thin '+' motifs, then recolor each family differently.

    Input structure:
    - bg is the most common color.
    - Exactly one foreground color forms a mixture of clutter, solid 3x3 squares, and thin '+' crosses on the background.
    - The '+' targets are visually cross-shaped rather than filled 3x3 patches.

    Rule:
    - Any all-foreground 3x3 neighborhood is a block target; recolor all 9 cells of that block to orange (7).
    - Any foreground cell whose 4 orthogonal neighbors are also foreground, but whose full 3x3 neighborhood is not all foreground, is a thin '+' center; recolor that center to teal (8) and its 4 orthogonal arm cells to magenta (6).
    - All remaining cells stay unchanged.

    Key insight:
    - The solver must separate two local geometries that would otherwise collide: a solid 3x3 contains an internal cross, so '+' detection must explicitly exclude solid 3x3 neighborhoods. After freezing both match sets on the original grid, the recoloring is straightforward.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    block_footprints = _find_block_footprints(g, fg)
    plus_centers = _find_plus_centers(g, fg)

    for footprint in block_footprints:
        for r, c in footprint:
            out[r, c] = BLOCK_COLOR

    for r, c in plus_centers:
        out[r, c] = PLUS_CENTER_COLOR
        for dr, dc in ORTHO:
            out[r + dr, c + dc] = PLUS_ARM_COLOR
    return out


def _halo(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], h: int, w: int, margin: int = 1) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - margin, r + margin + 1):
            for nc in range(c - margin, c + margin + 1):
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _would_create_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    h, w = grid.shape
    for cr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
                continue
            ok = True
            for rr in range(cr - 1, cr + 2):
                for nc in range(cc - 1, cc + 2):
                    val = fg if (rr, nc) == (r, c) else int(grid[rr, nc])
                    if val != fg:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _would_create_thin_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    h, w = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
            continue
        if any((cr + dr, cc + dc) == (r, c) or int(grid[cr + dr, cc + dc]) == fg for dr, dc in [(0, 0), *ORTHO]):
            center_and_arms = all(
                (cr + dr, cc + dc) == (r, c) or int(grid[cr + dr, cc + dc]) == fg
                for dr, dc in [(0, 0), *ORTHO]
            )
            if center_and_arms:
                full = True
                for rr in range(cr - 1, cr + 2):
                    for nc in range(cc - 1, cc + 2):
                        val = fg if (rr, nc) == (r, c) else int(grid[rr, nc])
                        if val != fg:
                            full = False
                            break
                    if not full:
                        break
                if not full:
                    return True
    return False


def _safe_to_add(grid: np.ndarray, r: int, c: int, bg: int, fg: int, reserved: set[tuple[int, int]]) -> bool:
    if int(grid[r, c]) != bg or (r, c) in reserved:
        return False
    return not _would_create_solid_3x3(grid, r, c, fg) and not _would_create_thin_plus(grid, r, c, fg)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_block(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    fg: int,
) -> tuple[tuple[tuple[int, int], ...], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(200):
        cr = int(rng.integers(1, h - 1))
        cc = int(rng.integers(1, w - 1))
        core = _block_cells(cr, cc)
        if any((r, c) in reserved for r, c in core):
            continue
        if any(int(grid[r, c]) != 2 for r, c in core):
            continue
        return core, _halo(core, h, w, margin=1)
    return None


def _place_plus(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    fg: int,
) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(240):
        cr = int(rng.integers(2, h - 2))
        cc = int(rng.integers(2, w - 2))
        core = _plus_cells(cr, cc)
        if any((r, c) in reserved for r, c in core):
            continue

        diag_cells = [(cr - 1, cc - 1), (cr - 1, cc + 1), (cr + 1, cc - 1), (cr + 1, cc + 1)]
        if any((r, c) in reserved for r, c in diag_cells):
            continue

        attachments: list[tuple[int, int]] = []
        dirs = list(ORTHO)
        rng.shuffle(dirs)
        n_ext = int(rng.integers(1, 3))
        for dr, dc in dirs[:n_ext]:
            max_len = 0
            for step in range(2, 5):
                nr, nc = cr + dr * step, cc + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if (nr, nc) in reserved:
                    break
                max_len = step - 1
            if max_len == 0:
                continue
            ext_len = int(rng.integers(1, max_len + 1))
            for step in range(2, ext_len + 2):
                attachments.append((cr + dr * step, cc + dc * step))

        component = list(core) + attachments
        if any((r, c) in reserved for r, c in component):
            continue
        diag_reserved = _halo(diag_cells, h, w, margin=0)
        return core, attachments, _halo(component, h, w, margin=1) | diag_reserved
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(12, 23))
    bg, fg = 2, 4

    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()
    reserved: set[tuple[int, int]] = set()

    area = h * w
    n_blocks = int(rng.integers(1, min(5, max(2, area // 55)) + 1))
    n_pluses = int(rng.integers(1, min(6, max(2, area // 45)) + 1))

    block_cores: list[tuple[tuple[int, int], ...]] = []
    plus_cores: list[tuple[tuple[int, int], ...]] = []

    for _ in range(n_blocks):
        placed = _place_block(rng, grid, reserved, fg)
        if placed is None:
            return None
        core, halo = placed
        _paint_cells(grid, core, fg)
        _paint_cells(expected, core, BLOCK_COLOR)
        reserved |= halo
        block_cores.append(core)

    for _ in range(n_pluses):
        placed = _place_plus(rng, grid, reserved, fg)
        if placed is None:
            return None
        core, attachments, halo = placed
        component = list(core) + attachments
        _paint_cells(grid, component, fg)
        _paint_cells(expected, attachments, fg)
        expected[core[0][0], core[0][1]] = PLUS_CENTER_COLOR
        for r, c in core[1:]:
            expected[r, c] = PLUS_ARM_COLOR
        reserved |= halo
        plus_cores.append(core)

    total = h * w
    target_density = float(rng.uniform(0.26, 0.38))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in reserved]
    rng.shuffle(candidates)
    for r, c in candidates:
        if current_fg >= target_fg:
            break
        if _safe_to_add(grid, r, c, bg, fg, reserved):
            grid[r, c] = fg
            expected[r, c] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if not (np.any(solved == BLOCK_COLOR) and np.any(solved == PLUS_CENTER_COLOR) and np.any(solved == PLUS_ARM_COLOR)):
        return None
    if len(_find_block_footprints(grid, fg)) != len(block_cores):
        return None
    if len(_find_plus_centers(grid, fg)) != len(plus_cores):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
