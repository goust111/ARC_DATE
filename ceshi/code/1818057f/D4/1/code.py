# Variant of 1818057f
# Dimension: D4 — 前景色数量与背景色关系 (meta)
# Change: 恰好1种前景色(黄色4)在1种背景色(红色2)上，前景密度26-41%... → 前景色数量编码变换类型

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg


# Design analysis for the D4 variant
# analysis:
#   a_concretization_and_trace:
#     - The canvas stays a single red background (2), but the foreground is now split into
#       two visible colors: yellow (4) and green (3).
#     - Both colors form the same kind of thin '+' motif: a center cell with four same-color
#       orthogonal neighbors, usually with 1-2 extra same-color cells extending beyond one or
#       two arms so the motif is embedded inside a larger component instead of appearing as an
#       isolated 5-cell cross.
#     - Behavior trace on one local patch:
#         original: yellow '+' -> all 5 cells become teal.
#         variant: yellow '+' -> all 5 cells become teal; green '+' -> those same 5 cells
#         become background, while any extra green extension cells beyond the erased arms stay.
#     - The changed step is the per-color transform after detecting a local '+'. Input
#       construction must therefore plant both colors' motifs, solver detection must scan both
#       colors independently, and output generation must apply different write values by color.
#   b_complexity_calibration:
#     - Only two foreground colors are used. That is enough to encode two actions without
#       adding ambiguous extra branches.
#     - Every generated grid contains at least one yellow '+' and at least one green '+', so
#       the new rule is visible inside a single example and repeated across examples.
#     - Distractors of both colors remain present, so the task is still "find local pluses",
#       not "change every pixel of one color".
#   c_implicit_assumption_audit_of_original_main:
#     - `colors = ...; fg = colors[0]` assumed exactly one foreground color. This fails here
#       because one whole action branch would be ignored.
#     - `if g[r, c] != fg: continue` assumed the single chosen foreground is the only
#       transformable color. The variant needs a per-color loop.
#     - `out[...] = 8` assumed every qualifying motif shares the same output color. That is no
#       longer true because green motifs erase to background instead.
#     - Reading neighbors from `g` rather than `out` is still required; otherwise an erased
#       green '+' could prevent a nearby yellow '+' from being found. The generator keeps
#       centers separated so this stays harmless and correct.
#   d_placement_legality:
#     - Centers are kept off the border and at L1 distance >= 4 from each other, matching the
#       original spacing scale and preventing plus footprints from overlapping.
#     - Extensions stop before occupied cells or borders, so different objects do not overwrite
#       each other.
#     - Distractor cells are only added when they would not create a new '+' of that same
#       color at the placed cell or one of its orthogonal neighbors.
#   e_visual_observability:
#     - The human-visible cue is stable color-to-action mapping: yellow pluses turn teal,
#       green pluses disappear into red.
#     - Because each sample contains both colors and repeated examples of each action, the
#       mapping can be inferred from 2-3 train pairs without needing hidden metadata.
#   f_visual_cleanliness:
#     - The generator keeps the same sparse-to-medium density band as the source task and
#       limits total motif count so the two action types remain easy to pick out.
#     - Overlap between motif footprints is forbidden; only same-color extensions and
#       distractors create local clutter, as in the source task.
#   g_color_change_inference:
#     - Only two source colors participate in the action map, and each has a single consistent
#       output behavior. No ordering by color id or other invisible property is used.


RECOLOR_FG = 4
ERASE_FG = 3
BG_COLOR = 2
TEAL = 8

ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
PLUS_OFFSETS = ((0, 0),) + ORTHO


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in two foreground colors and apply the color-specific action.

    Input structure:
    - bg is the most common color.
    - Two foreground colors are mixed on that bg: yellow (4) and green (3).
    - Both colors can contain embedded thin '+' motifs inside larger same-color shapes.

    Rule:
    - A '+' center is any non-border pixel whose 4 orthogonal neighbors match the center's color.
    - For yellow '+' motifs, recolor the center and its 4 neighbors to teal (8).
    - For green '+' motifs, replace that same 5-pixel footprint with the background color.
    - All other pixels remain unchanged.

    Key insight:
    - The geometry test is identical for both branches; only the motif color chooses the output
      action, so detection must read the original grid and scan each relevant foreground color
      separately.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    for fg, replacement in ((RECOLOR_FG, TEAL), (ERASE_FG, bg)):
        for r, c in _find_plus_centers(g, fg):
            for dr, dc in PLUS_OFFSETS:
                out[r + dr, c + dc] = replacement
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether painting (r,c) with fg would create a new '+' of that same color."""
    H, W = grid.shape
    for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        if all(
            (fg if (cr + dr, cc + dc) == (r, c) else int(grid[cr + dr, cc + dc])) == fg
            for dr, dc in PLUS_OFFSETS
        ):
            return True
    return False


def _place_extensions(
    grid: np.ndarray,
    centers: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> None:
    dirs = list(ORTHO)
    H, W = grid.shape
    for fg, cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(len(dirs), size=n_ext, replace=False)
        for di in chosen:
            dr, dc = dirs[int(di)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if int(grid[nr, nc]) != BG_COLOR:
                    break
                grid[nr, nc] = fg


def _single_color_output(grid: np.ndarray, fg: int) -> np.ndarray:
    out = np.array(grid, dtype=int)
    for r, c in _find_plus_centers(out, fg):
        for dr, dc in PLUS_OFFSETS:
            out[r + dr, c + dc] = TEAL
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    grid = np.full((H, W), BG_COLOR, dtype=int)

    k_yellow = int(rng.integers(1, 5))
    k_green = int(rng.integers(1, 5))
    motif_colors = [RECOLOR_FG] * k_yellow + [ERASE_FG] * k_green
    rng.shuffle(motif_colors)

    placed: list[tuple[int, int, int]] = []
    for fg in motif_colors:
        found = False
        for _ in range(60):
            cr = int(rng.integers(2, H - 2))
            cc = int(rng.integers(2, W - 2))
            if all(abs(cr - er) + abs(cc - ec) >= 4 for _, er, ec in placed):
                placed.append((fg, cr, cc))
                found = True
                break
        if not found:
            return None

    for fg, cr, cc in placed:
        for dr, dc in PLUS_OFFSETS:
            grid[cr + dr, cc + dc] = fg

    _place_extensions(grid, placed, rng)

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != BG_COLOR))

    bg_positions = list(zip(*np.where(grid == BG_COLOR)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        for fg in rng.permutation([RECOLOR_FG, ERASE_FG]):
            fg = int(fg)
            if not _would_create_plus(grid, int(r), int(c), fg):
                grid[int(r), int(c)] = fg
                current_fg += 1
                break

    if int(np.sum(grid != BG_COLOR)) >= total // 2:
        return None

    yellow_centers = _find_plus_centers(grid, RECOLOR_FG)
    green_centers = _find_plus_centers(grid, ERASE_FG)
    if not yellow_centers or not green_centers:
        return None

    out = main(grid)

    # Non-degeneracy guard: the second color must actually trigger the new erase branch.
    if np.array_equal(out, _single_color_output(grid, RECOLOR_FG)):
        return None

    if not np.any(out == TEAL):
        return None
    if not np.any((grid == ERASE_FG) & (out == BG_COLOR)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")


# Assumption review for the new main():
# - Only yellow and green can be transform-driving colors; generate_input always plants both.
# - Plus detection is purely local and same-color; generate_input prevents accidental extra
#   same-color pluses when adding distractors, so all detected motifs are intended.
# - Writes can safely share one output grid because centers are spaced far enough that the 5-cell
#   footprints do not overlap, and detection always reads the unmodified input copy.
