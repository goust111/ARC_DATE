# Variant of 1818057f
# Dimension: D1 — 核心变换操作：重着色规则 (transform)
# Change: 识别'+'字图案后，将构成'+'的5个像素从前景色(黄色4)统一重着色为青色(teal 8)，其余像素不变... → 擦除'+'字图案

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the D1 erase variant
# variant_spec:
#   core_change:
#     detection is unchanged from 1818057f: a foreground pixel is a plus-center iff
#     its four orthogonal neighbors are also foreground. The only semantic change is
#     the write-back action: those 5 cells are carved back to the background color
#     instead of being recolored to teal.
#   concretization_and_trace:
#     input still has one background color (red 2 in generated samples) and one
#     foreground color (yellow 4). Foreground forms thin lines / blobs containing
#     embedded local '+' junctions. Example:
#       input neighborhood:           output neighborhood:
#           . 4 .                         . . .
#           4 4 4          ---->          . . .
#           . 4 .                         . . .
#     but with extra foreground cells extending beyond at least one arm, so the
#     erased cross becomes a visible hole or gap inside a larger structure rather
#     than an isolated disappearing object.
#   complexity_calibration:
#     The task remains single-color and single-rule, so D1 stays the only real
#     change. To keep the erase effect observable, every generated plus gets at
#     least one extension beyond its 5-cell footprint; otherwise deleting an
#     isolated cross would leave only background and look visually inert.
#   implicit_assumptions_review:
#     - `get_bg(g)` is still valid because the variant keeps a single dominant bg.
#     - selecting the lone non-bg color as `fg` is still valid because erase does
#       not add a new color.
#     - detection must read from the original grid `g`, not the mutable output
#       `out`; otherwise erasing one plus could suppress a later nearby detection.
#     - unconditional writes are still safe because the generator keeps plus
#       footprints disjoint by spacing centers at L1 distance >= 4 and forbids
#       extra unplanned plus-centers.
#   placement_legality:
#     Planned plus-centers stay at least two cells from the border so all four arms
#     exist. The generator stamps the 5-cell plus footprint first, then adds arm
#     extensions and distractors only when the new pixel would not create any plus
#     center outside the planned set. This keeps objects in-bounds and prevents
#     accidental overlapping motifs.
#   observability_and_cleanliness:
#     Input/output pairs differ structurally, not by hidden color ids: the same
#     local '+' pattern is recognized, but output removes it. Density stays in the
#     original 26-41% range so there is enough surrounding foreground for the new
#     holes to be visually legible without turning the board into a solid mass.
#   new_solver_assumptions_to_match_generator:
#     `main()` now assumes every meaningful plus is detectable directly from the
#     input foreground layout. `generate_input()` guarantees this by validating
#     that the actual detected centers equal the planned centers before returning.

PLUS_ARMS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS_FOOTPRINT = [(0, 0), *PLUS_ARMS]


def _plus_centers(grid: np.ndarray, fg: int) -> set[tuple[int, int]]:
    """Return all interior coordinates that satisfy the local '+' test."""
    H, W = grid.shape
    centers: set[tuple[int, int]] = set()
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in PLUS_ARMS):
                centers.add((int(r), int(c)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' junctions in a single foreground color and erase them to background.

    Input structure:
    - The most frequent color is the background.
    - One other color forms sparse-to-moderate foreground lines and blobs on top of it.
    - Some foreground positions are local '+' motifs: a center pixel with all 4 orthogonal neighbors also in foreground.

    Rule:
    - Scan interior cells of the input grid.
    - Whenever a foreground pixel has foreground in all 4 orthogonal directions, treat that center plus its 4 neighbors as one '+' motif.
    - Replace those 5 pixels with the background color.
    - Leave every other cell unchanged.

    Key insight:
    - The pattern-recognition step is identical to the source task; the variant only changes the effect from additive recoloring to subtractive carving, so each detected '+' becomes a 5-cell hole or gap in the original foreground structure.
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
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in PLUS_ARMS):
                for dr, dc in PLUS_FOOTPRINT:
                    out[r + dr, c + dc] = bg
    return out


def _would_create_unplanned_plus(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    """Check whether setting (r,c) to fg would create a plus outside the planned set."""
    if int(grid[r, c]) == int(fg):
        return False

    H, W = grid.shape
    for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_FOOTPRINT:
            nr, nc = cr + dr, cc + dc
            val = int(fg) if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != int(fg):
                all_fg = False
                break
        if all_fg and (int(cr), int(cc)) not in allowed_centers:
            return True
    return False


def _gen_attempt():
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    # Plan 1-9 motif centers, each safely away from the border and mutually separated.
    K = int(rng.integers(1, 10))
    centers: list[tuple[int, int]] = []
    for _ in range(K * 50):
        if len(centers) >= K:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))

    if not centers:
        return None

    center_set = set(centers)

    for cr, cc in centers:
        for dr, dc in PLUS_FOOTPRINT:
            grid[cr + dr, cc + dc] = fg

    # Embed each plus in a larger shape by extending beyond one or more arm tips.
    for cr, cc in centers:
        target_dirs = int(rng.integers(1, 4))
        added_any = False
        used_dirs = 0

        for di in rng.permutation(len(PLUS_ARMS)):
            if used_dirs >= target_dirs:
                break
            dr, dc = PLUS_ARMS[int(di)]
            ext_len = int(rng.integers(1, 4))
            grew_this_dir = False

            for step in range(2, 2 + ext_len):
                nr = cr + dr * step
                nc = cc + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if int(grid[nr, nc]) == fg:
                    break
                if _would_create_unplanned_plus(grid, nr, nc, fg, center_set):
                    break
                grid[nr, nc] = fg
                grew_this_dir = True
                added_any = True

            if grew_this_dir:
                used_dirs += 1

        if not added_any:
            return None

    if _plus_centers(grid, fg) != center_set:
        return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)

    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_unplanned_plus(grid, int(r), int(c), fg, center_set):
            grid[int(r), int(c)] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    actual_centers = _plus_centers(grid, fg)
    if actual_centers != center_set:
        return None

    inp = grid.copy()
    out = main(inp)
    changed = inp != out

    if not np.any(changed):
        return None
    if int(np.sum(changed)) != 5 * len(center_set):
        return None
    if not np.all(inp[changed] == fg):
        return None
    if not np.all(out[changed] == bg):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
