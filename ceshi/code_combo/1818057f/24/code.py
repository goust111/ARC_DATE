# Combo variant of 1818057f (index 24)
# combo_spec: D1/0+D15/3
# Name: 演示-测试擦除范式

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for combo D1/0 + D15/3
# a) Concrete layout and behavior trace
#    The full grid is split into two equal-height bands with a 1-row background separator.
#    The upper band is not an unsolved panel: it already shows the "after" state, where
#    every local '+' motif has been erased to background. The lower band is an unsolved
#    panel that still contains intact '+' junctions. Small trace:
#      hidden upper-before:        visible upper-after:       lower input / lower output:
#          . 4 .                       . 4 .                     . 4 .        . . .
#          4 4 4         erase ->      . . .        then         4 4 4  ->    . . .
#          . 4 .                       . 4 .                     . 4 .        . . .
#      with extra foreground continuing one cell farther in all four directions, so the
#      erased 5 cells leave a plus-shaped cavity rather than a meaningless isolated loss.
#    Relative to the original task, two execution points change together: the action is
#    now erase-to-background instead of recolor-to-teal, and the actionable area is only
#    the lower half because the upper half is already a completed demonstration.
# b) Complexity calibration
#    A single solved example would be too weak if it contained only one tiny deletion, so
#    the generator enforces 2-4 erased examples in the top band and 1-4 intact '+' motifs
#    in the bottom band. Each planned '+' extends beyond all four arms by at least one cell,
#    making the upper demonstration visually repeat the same cavity pattern several times.
#    Distractor foreground cells are still sprinkled, but only when they do not create any
#    unplanned '+' centers, so the task keeps the original ambiguity level without turning
#    into a dense yellow block.
# c) Implicit assumptions review
#    - `get_bg(g)` still holds because background remains the dominant color everywhere.
#    - Choosing the lone non-bg color as `fg` still holds because erase does not introduce
#      a new output color; unlike the recolor legend variant, this combo stays 2-color.
#    - The original "scan the whole grid" assumption is no longer valid. `main()` must
#      leave the upper demonstration untouched and only search below the separator row.
#    - Detection must read from the immutable input slice rather than progressively erased
#      output so one deletion cannot suppress another. The generator keeps centers spaced
#      far enough apart that unconditional writes on the copied output are safe.
# d) Placement legality
#    Center candidates are sampled with a 2-cell margin from panel borders, which is enough
#    for the 5-cell cross plus a mandatory extra foreground cell at distance 2 in each arm
#    direction. Centers are separated by L1 distance >= 6 so those mandatory arm extensions
#    do not collide. Distractors are added only if the new cell does not create any plus
#    center outside the planned set, preventing accidental extra motifs.
# e) Visual observability
#    A human can infer the rule from the top band because the same plus-shaped cavity appears
#    multiple times inside yellow structures, while the bottom band shows the corresponding
#    intact '+' junctions in the same visual language. The separator row makes the "example
#    above / apply below" reading direct even though no extra marker color is introduced.
# f) Visual cleanliness
#    Only two colors appear in the input: red background and yellow foreground. The blank
#    separator row prevents cross-band interactions, and the top band is already solved, so
#    there is no side-by-side clutter. Repeated spacing and full-arm extensions keep each
#    erased cavity legible instead of dissolving into random background noise.
# g) Change inferability and non-degeneracy
#    The crucial measurable behavior is: top band contains visible erased plus-holes, bottom
#    band contains intact plus-centers, and `main()` erases only the bottom ones. Generator
#    validation checks that the visible top-band holes can be re-detected as 4-armed cavity
#    centers and that the bottom band still contains actionable '+' motifs. Compared with the
#    original 1818057f recolor solver, the same input yields a different output because the
#    lower-band motifs are removed instead of recolored.
# h) New main() assumptions to mirror in generation
#    `main()` assumes odd total height, a separator row at the midpoint, exactly one active
#    non-background color, and at least one intact '+' below the separator. `generate_input()`
#    constructs precisely that layout and rejects any sample whose visible top-hole count or
#    lower-band plus count does not match the intended structure.


PLUS_ARMS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS_FOOTPRINT = [(0, 0), *PLUS_ARMS]


def _plus_centers(
    grid: np.ndarray,
    fg: int,
    row_start: int = 0,
    row_end: int | None = None,
) -> set[tuple[int, int]]:
    """Return all '+' centers made of fg whose footprint lies inside the requested rows."""
    h, w = grid.shape
    if row_end is None:
        row_end = h

    centers: set[tuple[int, int]] = set()
    for r in range(max(1, row_start + 1), min(h - 1, row_end - 1)):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in PLUS_ARMS):
                centers.add((int(r), int(c)))
    return centers


def _hole_centers(
    grid: np.ndarray,
    bg: int,
    fg: int,
    row_start: int = 0,
    row_end: int | None = None,
) -> set[tuple[int, int]]:
    """
    Detect visible plus-shaped cavities: center+4 arms are bg and all distance-2 arm cells are fg.
    """
    h, w = grid.shape
    if row_end is None:
        row_end = h

    centers: set[tuple[int, int]] = set()
    for r in range(max(2, row_start + 2), min(h - 2, row_end - 2)):
        for c in range(2, w - 2):
            if int(grid[r, c]) != int(bg):
                continue
            if not all(int(grid[r + dr, c + dc]) == int(bg) for dr, dc in PLUS_ARMS):
                continue
            if all(int(grid[r + 2 * dr, c + 2 * dc]) == int(fg) for dr, dc in PLUS_ARMS):
                centers.add((int(r), int(c)))
    return centers


def _erase_rows(grid: np.ndarray, row_start: int, row_end: int, fg: int, bg: int) -> np.ndarray:
    """Erase '+' motifs inside rows [row_start, row_end)."""
    out = grid.copy()
    h, w = grid.shape
    for r in range(max(1, row_start + 1), min(h - 1, row_end - 1)):
        for c in range(1, w - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in PLUS_ARMS):
                for dr, dc in PLUS_FOOTPRINT:
                    out[r + dr, c + dc] = bg
    return out


def _solve_panel(panel: np.ndarray, fg: int, bg: int) -> np.ndarray:
    """Erase '+' motifs inside a standalone panel."""
    return _erase_rows(panel, 0, panel.shape[0], fg, bg)


def _find_demo_separator(grid: np.ndarray, bg: int, fg: int) -> int | None:
    """
    Infer the separator row from the solved upper demonstration.

    A valid split must have:
    - an all-background separator row,
    - at least two visible plus-shaped holes above it,
    - no intact '+' motifs remaining in the upper band,
    - at least one intact '+' motif below it.
    """
    h = grid.shape[0]
    best_sep: int | None = None
    best_score: tuple[int, int, int] | None = None

    for sep in range(3, h - 3):
        if not np.all(grid[sep, :] == bg):
            continue

        top_holes = _hole_centers(grid, bg, fg, 0, sep)
        if len(top_holes) < 2:
            continue
        if _plus_centers(grid, fg, 0, sep):
            continue

        bottom_pluses = _plus_centers(grid, fg, sep + 1, h)
        if not bottom_pluses:
            continue

        score = (len(top_holes), len(bottom_pluses), -abs((2 * sep + 1) - h))
        if best_score is None or score > best_score:
            best_score = score
            best_sep = sep

    return best_sep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper solved example as an erased '+' template, then erase matching '+' motifs below.

    Input structure:
    - The board is split horizontally into an upper example band and a lower task band with a background separator row between them.
    - Only one non-background color appears in the whole input; the upper band already uses the final erased form, so no extra marker color is introduced.
    - In the top half, repeated plus-shaped cavities are carved out of larger foreground structures; in the bottom half, analogous intact '+' junctions remain.

    Rule:
    - Keep the upper example band and the separator row unchanged.
    - Search only the lower band for foreground pixels whose 4 orthogonal neighbors are also foreground.
    - For every such '+' center, replace that center and its 4 orthogonal neighbors with the background color.
    - Leave all other cells unchanged.

    Key insight:
    - The top band is already post-transform, so its plus information is encoded as negative space rather than a new color. The solver must transfer that "erase this 5-cell '+' footprint" rule to the unsolved lower half instead of applying the original whole-grid recoloring behavior.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    sep = _find_demo_separator(g, bg, fg)
    if sep is None:
        return out
    return _erase_rows(g, sep + 1, g.shape[0], fg, bg)


def _would_create_unplanned_plus(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    """Check whether painting (r,c) as fg would create a plus outside the planned set."""
    if int(grid[r, c]) == int(fg):
        return False

    h, w = grid.shape
    for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
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


def _gen_panel_attempt(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    fg: int,
    want_centers: int,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    """Generate one unsolved panel plus its intended '+' centers."""
    panel = np.full((h, w), bg, dtype=int)

    row_choices = list(range(2, h - 2))
    col_choices = list(range(2, w - 2))
    if not row_choices or not col_choices:
        return None

    centers: list[tuple[int, int]] = []
    for _ in range(want_centers * 120):
        if len(centers) >= want_centers:
            break
        cr = int(rng.choice(row_choices))
        cc = int(rng.choice(col_choices))
        if all(abs(cr - er) + abs(cc - ec) >= 6 for er, ec in centers):
            centers.append((cr, cc))
    if len(centers) != want_centers:
        return None

    center_set = set(centers)

    for cr, cc in centers:
        for dr, dc in PLUS_FOOTPRINT:
            panel[cr + dr, cc + dc] = fg

    # Mandatory continuation beyond every arm makes the erased top-half cavities easy to read.
    for cr, cc in centers:
        for dr, dc in PLUS_ARMS:
            ext_len = int(rng.integers(1, 4))
            for step in range(2, 2 + ext_len):
                nr = cr + dr * step
                nc = cc + dc * step
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    if step == 2:
                        return None
                    break
                if int(panel[nr, nc]) == int(fg):
                    return None
                if _would_create_unplanned_plus(panel, nr, nc, fg, center_set):
                    return None
                panel[nr, nc] = fg

    if _plus_centers(panel, fg) != center_set:
        return None

    area = h * w
    target_density = float(rng.uniform(0.26, 0.39))
    target_fg = int(area * target_density)
    current_fg = int(np.sum(panel != bg))

    bg_positions = list(zip(*np.where(panel == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_unplanned_plus(panel, int(r), int(c), fg, center_set):
            panel[int(r), int(c)] = fg
            current_fg += 1

    if int(np.sum(panel != bg)) >= area // 2:
        return None
    if _plus_centers(panel, fg) != center_set:
        return None

    return panel, center_set


def _generate_panel(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    fg: int,
    want_centers: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    for _ in range(400):
        result = _gen_panel_attempt(rng, h, w, bg, fg, want_centers)
        if result is not None:
            return result
    raise RuntimeError("panel generation failed after 400 attempts")


def _assemble_grid(demo_solved: np.ndarray, task_panel: np.ndarray, bg: int) -> np.ndarray:
    ph, pw = demo_solved.shape
    full = np.full((2 * ph + 1, pw), bg, dtype=int)
    full[:ph, :] = demo_solved
    full[ph + 1 :, :] = task_panel
    return full


def _sample_panel_dims(
    rng: np.random.Generator,
    demo_count: int,
    task_count: int,
) -> tuple[int, int]:
    """Bias larger motif counts toward roomier panels so all count branches appear regularly."""
    max_count = max(demo_count, task_count)
    min_h = 6 + max(0, max_count - 2)
    min_w = 12 + 2 * max(0, max_count - 2)
    panel_h = int(rng.integers(min_h, 11))
    panel_w = int(rng.integers(min_w, 23))
    return panel_h, panel_w


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg, fg = 2, 4

    for _ in range(400):
        demo_count = int(rng.integers(2, 5))
        task_count = int(rng.integers(1, 5))
        panel_h, panel_w = _sample_panel_dims(rng, demo_count, task_count)  # full height stays 13..21

        demo_in, demo_centers = _generate_panel(rng, panel_h, panel_w, bg, fg, demo_count)
        task_in, task_centers = _generate_panel(rng, panel_h, panel_w, bg, fg, task_count)

        demo_out = _solve_panel(demo_in, fg, bg)
        task_out = _solve_panel(task_in, fg, bg)

        inp = _assemble_grid(demo_out, task_in, bg)
        expected = _assemble_grid(demo_out, task_out, bg)
        actual = main(inp)

        top_holes = _hole_centers(inp[:panel_h, :], bg, fg)
        bottom_pluses = _plus_centers(inp, fg, panel_h + 1, inp.shape[0])

        if not np.array_equal(actual, expected):
            continue
        if top_holes != demo_centers:
            continue
        if bottom_pluses != {(panel_h + 1 + r, c) for r, c in task_centers}:
            continue
        if _plus_centers(inp[:panel_h, :], fg):
            continue
        if np.any(inp[panel_h, :] != bg):
            continue
        if np.any(actual[: panel_h + 1, :] != inp[: panel_h + 1, :]):
            continue
        if int(np.sum(inp != expected)) != 5 * len(task_centers):
            continue
        if int(np.sum(demo_in != demo_out)) != 5 * len(demo_centers):
            continue
        if not demo_centers or not task_centers:
            continue
        return inp

    raise RuntimeError("generate_input failed after 400 attempts")
