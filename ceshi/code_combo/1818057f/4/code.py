# Combo variant of 1818057f (index 4)
# combo_spec: D3/3+D8/0
# Name: 实心方块检测配高迷惑干扰

# Design analysis for the combo variant
# - a) Concretization and behavior trace:
#   The original task recolors any thin '+' center plus its 4 orthogonal neighbors. This combo
#   changes the local test and the distractor family together. The new target is a fully filled
#   3x3 foreground block:
#       ###
#       ###
#       ###
#   which now turns all 9 cells teal. The main decoy is a near-miss 3x3 region with one missing
#   corner:
#       ##.
#       ###
#       ###
#   In the original solver, the center of that 8-cell shape would still satisfy the 4-neighbor
#   '+' test and recolor 5 cells. In this combo, it stays yellow because one diagonal is absent.
#   So the changed behavior is visible in both directions: true 3x3 blocks recolor more cells than
#   before, while highly similar 8/9 blocks recolor nothing.
# - b) Complexity calibration:
#   Using only obvious isolated 3x3 squares would make the variant trivial. Each generated grid
#   therefore includes both solid 3x3 targets and several corner-gap 8/9 decoys that share the
#   same center and orthogonal cross. The rule remains a single local test with no hidden state,
#   but the input now contains strong "off-by-one" traps that force exact completeness checking.
# - c) Old implicit assumptions audited:
#   `bg = get_bg(g)` still holds because the generator keeps background as the majority color.
#   `colors[0]` remains valid because there is still exactly one non-background foreground color.
#   Scanning only interior cells still works because every candidate 3x3 center needs a 1-cell
#   margin. Reading from `g` and writing to `out` separately is still required so recoloring one
#   3x3 window cannot affect detection of another. The old 4-neighbor check is no longer sufficient:
#   every one of the 9 cells in the centered 3x3 window must be foreground.
# - d) Placement legality:
#   Deliberate motifs are placed with a reserved halo, so designed solid blocks and designed decoys
#   neither overlap nor accidentally complete one another. Thin extensions are allowed, but only if
#   every cell stays in bounds. Later random fill is filtered through a "would this create a solid
#   3x3?" check, so it cannot accidentally turn a decoy into a target.
# - e) Visual observability:
#   Across 2-4 training pairs, a human can compare solid 3x3 patches against almost-identical
#   corner-gap patches and see that only the fully filled ones turn teal. The distinction is based
#   on visible local completeness, not on color id arithmetic or any non-visual ordering.
# - f) Visual cleanliness:
#   The generator uses one foreground color, moderate density, separated motif placement, and
#   1-cell-wide extensions. This avoids noisy multi-color interactions and keeps the decisive local
#   3x3 windows easy to inspect even when the grid also contains larger blobs and scattered cells.
# - g) Change inferability:
#   Color usage stays simple: red-like background, yellow-like foreground, teal output mark. The
#   only learned mapping is structural: "all 9 cells present" implies recolor, "8 cells with a
#   missing corner" does not. Because many samples repeat this contrast, the rule is inferable.
# - h) Generator strategy adjustment:
#   An input-first generator is safer than the original output-first cross generator. Solid 3x3
#   windows and corner-gap 8/9 windows are easier to place directly, then validate with `main()`.
# - i) New `main()` assumptions and how the generator satisfies them:
#   `main()` assumes all targets are detectable by a purely local centered 3x3-all-foreground test.
#   The generator guarantees at least one such target, at least one corner-gap 8/9 decoy, exactly
#   one foreground color, and no accidental extra solid 3x3 windows from the random fill phase.

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


ORTHO_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
CORNER_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _is_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return bool(np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == fg))


def _is_corner_gap_decoy(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    window = grid[r - 1 : r + 2, c - 1 : c + 2] == fg
    if int(np.sum(window)) != 8:
        return False
    missing = {(dr - 1, dc - 1) for dr, dc in zip(*np.where(~window))}
    return len(missing) == 1 and next(iter(missing)) in set(CORNER_OFFSETS)


def _has_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect fully filled 3x3 foreground squares in a single foreground color and recolor them to teal.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms mixed blobs, short lines, solid patches, and near-solid patches.
    - Some local 3x3 regions are completely filled, while others look almost complete but miss one corner.

    Rule:
    - A target center is any foreground pixel whose entire centered 3x3 neighborhood is foreground.
    - Recolor all 9 cells of each such 3x3 neighborhood to teal (8).
    - Any incomplete 3x3 region, even one missing only a single corner, stays unchanged.

    Key insight:
    - The decisive test is full 3x3 completeness, not the thinner '+' cross hidden inside it.
      A corner-gap 8/9 block still contains the same center and orthogonal arms as a '+', but it is
      not recolored because one diagonal cell is missing.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if _is_solid_3x3(g, r, c, fg):
                out[r - 1 : r + 2, c - 1 : c + 2] = 8
    return out


def _count_solid_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) == fg and _is_solid_3x3(grid, r, c, fg):
                count += 1
    return count


def _count_corner_gap_decoys(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_corner_gap_decoy(grid, r, c, fg):
                count += 1
    return count


def _count_deceptive_plus_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _has_plus_center(grid, r, c, fg) and _is_corner_gap_decoy(grid, r, c, fg):
                count += 1
    return count


def _corner_gap_variants(grid: np.ndarray, fg: int) -> set[tuple[int, int]]:
    H, W = grid.shape
    variants: set[tuple[int, int]] = set()
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            window = grid[r - 1 : r + 2, c - 1 : c + 2] == fg
            if int(np.sum(window)) != 8:
                continue
            missing = [(dr - 1, dc - 1) for dr, dc in zip(*np.where(~window))]
            if len(missing) == 1 and missing[0] in CORNER_OFFSETS:
                variants.add(missing[0])
    return variants


def _would_create_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    for cr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
                continue
            window_ok = True
            for rr in range(cr - 1, cr + 2):
                for nc in range(cc - 1, cc + 2):
                    val = fg if (rr, nc) == (r, c) else int(grid[rr, nc])
                    if val != fg:
                        window_ok = False
                        break
                if not window_ok:
                    break
            if window_ok:
                return True
    return False


def _can_place(cells: set[tuple[int, int]], reserved: np.ndarray, pad: int = 1) -> bool:
    H, W = reserved.shape
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        r0 = max(0, r - pad)
        r1 = min(H, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(W, c + pad + 1)
        if np.any(reserved[r0:r1, c0:c1]):
            return False
    return True


def _mark_reserved(cells: set[tuple[int, int]], reserved: np.ndarray, pad: int = 1) -> None:
    H, W = reserved.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(H, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(W, c + pad + 1)
        reserved[r0:r1, c0:c1] = True


def _build_solid_shape(
    cr: int, cc: int, dirs: list[tuple[int, int]], lengths: list[int]
) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)}
    for (dr, dc), length in zip(dirs, lengths):
        for step in range(2, length + 2):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _build_corner_gap_shape(
    cr: int,
    cc: int,
    missing_corner: tuple[int, int],
    dirs: list[tuple[int, int]],
    lengths: list[int],
) -> set[tuple[int, int]]:
    cells = {
        (r, c)
        for r in range(cr - 1, cr + 2)
        for c in range(cc - 1, cc + 2)
        if (r - cr, c - cc) != missing_corner
    }
    for (dr, dc), length in zip(dirs, lengths):
        for step in range(2, length + 2):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _try_place_solids(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    count: int,
    fg: int,
) -> int:
    H, W = grid.shape
    placed = 0
    for _ in range(count * 100):
        if placed >= count:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        n_ext = int(rng.integers(1, 3))
        chosen_idx = rng.choice(len(ORTHO_DIRS), size=n_ext, replace=False)
        chosen_dirs = [ORTHO_DIRS[int(i)] for i in chosen_idx]
        lengths = [int(rng.integers(1, 4)) for _ in chosen_dirs]
        cells = _build_solid_shape(cr, cc, chosen_dirs, lengths)
        if not _can_place(cells, reserved, pad=1):
            continue
        for r, c in cells:
            grid[r, c] = fg
        _mark_reserved(cells, reserved, pad=1)
        placed += 1
    return placed


def _try_place_decoys(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    count: int,
    fg: int,
) -> int:
    H, W = grid.shape
    placed = 0
    for _ in range(count * 120):
        if placed >= count:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        missing_corner = CORNER_OFFSETS[int(rng.integers(len(CORNER_OFFSETS)))]
        n_ext = int(rng.integers(0, 3))
        if n_ext:
            chosen_idx = rng.choice(len(ORTHO_DIRS), size=n_ext, replace=False)
            chosen_dirs = [ORTHO_DIRS[int(i)] for i in chosen_idx]
            lengths = [int(rng.integers(1, 4)) for _ in chosen_dirs]
        else:
            chosen_dirs = []
            lengths = []
        cells = _build_corner_gap_shape(cr, cc, missing_corner, chosen_dirs, lengths)
        if not _can_place(cells, reserved, pad=1):
            continue
        for r, c in cells:
            grid[r, c] = fg
        _mark_reserved(cells, reserved, pad=1)
        placed += 1
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    max_solids = max(1, min(6, (H * W) // 32))
    max_decoys = max(2, min(7, (H * W) // 24))
    target_solids = int(rng.integers(1, max_solids + 1))
    target_decoys = int(rng.integers(2, max_decoys + 1))

    placed_solids = _try_place_solids(rng, grid, reserved, target_solids, fg)
    placed_decoys = _try_place_decoys(rng, grid, reserved, target_decoys, fg)
    if placed_solids == 0 or placed_decoys < 2:
        return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.38))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        rr = int(r)
        cc = int(c)
        if not _would_create_solid_3x3(grid, rr, cc, fg):
            grid[rr, cc] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    solid_centers = _count_solid_centers(grid, fg)
    corner_gap_decoys = _count_corner_gap_decoys(grid, fg)
    deceptive_plus_centers = _count_deceptive_plus_centers(grid, fg)
    corner_gap_variants = _corner_gap_variants(grid, fg)
    if solid_centers == 0:
        return None
    if corner_gap_decoys < 2 or deceptive_plus_centers < 2:
        return None
    if len(corner_gap_variants) < 2:
        return None

    out = main(grid)
    if not np.any(out == 8):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
