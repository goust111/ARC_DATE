# Variant of 1818057f
# Dimension: D15 — 图例/符号指令 (symbol_instructions)
# Change: 无图例或符号指令，变换规则完全隐含在输入输出的差异中... → 颜色映射图例在网格边缘

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# ===============
# reference_task:
#   id: dbff022c
#   status_in_workspace: "No local solver file was available to read; only the edge-legend
#     mechanism from the prompt can be imported here."
#   retained_mechanism: "A tiny palette legend sits on the grid edge and must be read before
#     applying the structural rule."
# variant_core:
#   original_behavior:
#     - "The board had only bg + one fg color."
#     - "Every local thin '+' made of fg was recolored to a fixed constant teal(8)."
#   new_behavior:
#     - "The main canvas still contains one operative fg color and the same embedded '+' motifs."
#     - "A 2-cell legend is added on the top edge or left edge."
#     - "Top legend = vertical 2x1 strip: source above target."
#     - "Left legend = horizontal 1x2 strip: source left of target."
#     - "A full bg separator band isolates the legend from the playfield so instruction pixels
#       cannot accidentally participate in '+' detection."
# concrete_trace:
#   - "Original: if the playfield color is yellow and a plus exists, solver always writes teal."
#   - "Variant: if the top legend shows yellow above green, the exact same plus geometry now
#     becomes green; if another sample shows orange left of grey, orange pluses become grey."
#   - "So the changed step is not motif detection but how recolor_target is sourced."
# complexity_calibration:
#   - "Only one mapping pair is present per board. That keeps the task readable in 2-4 examples."
#   - "Both legend orientation branches repeat the same source->target idea, so there are only
#     two layout cases to learn."
# implicit_assumption_audit:
#   - "Original `fg = colors[0]` assumed the only non-bg color in the board was the operative
#     foreground. This breaks once the legend introduces a second non-bg color."
#   - "Original `out[...] = 8` assumed the output color was constant. This breaks when the legend
#     chooses the destination color."
#   - "Original full-grid scan assumed every non-bg pixel belonged to the drawable region. This
#     breaks once legend pixels live on the edge; the solver must isolate the playfield first."
# placement_legality:
#   - "Generator reserves either rows 0..2 or cols 0..2 for legend+separator and places the
#     original-style content strictly inside the remaining playfield."
#   - "Plus centers keep a 1-cell internal margin within the playfield, and different centers
#     keep L1 distance >= 4 so their 5-cell footprints do not overlap."
# visual_observability:
#   - "The separator band makes the legend read as a distinct instruction object instead of random
#     clutter."
#   - "Source and target colors vary across samples, preventing degeneration back to a hidden
#     fixed teal rule."
# color_inference:
#   - "The only non-bg colors outside the playfield are the two legend cells, so the mapping cue
#     is unambiguous."
# main_new_assumptions:
#   - "A valid input contains exactly one legend pair plus an all-bg separator band."
#   - "The playfield uses the legend's source color for all motifs to be recolored."
#   - "Generator enforces both assumptions directly."


PLUS_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
ARM_OFFSETS = PLUS_OFFSETS[1:]
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)


def _parse_legend(g: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    H, W = g.shape

    if H >= 4 and np.all(g[2, :] == bg):
        top_non_bg = np.flatnonzero(g[0, :] != bg)
        next_non_bg = np.flatnonzero(g[1, :] != bg)
        if len(top_non_bg) == 1 and len(next_non_bg) == 1 and int(top_non_bg[0]) == int(next_non_bg[0]):
            c = int(top_non_bg[0])
            if np.all(g[:2, np.arange(W) != c] == bg):
                src = int(g[0, c])
                dst = int(g[1, c])
                if src != bg and dst != bg and src != dst:
                    return 3, 0, src, dst

    if W >= 4 and np.all(g[:, 2] == bg):
        left_non_bg = np.flatnonzero(g[:, 0] != bg)
        right_non_bg = np.flatnonzero(g[:, 1] != bg)
        if len(left_non_bg) == 1 and len(right_non_bg) == 1 and int(left_non_bg[0]) == int(right_non_bg[0]):
            r = int(left_non_bg[0])
            if np.all(g[np.arange(H) != r, :2] == bg):
                src = int(g[r, 0])
                dst = int(g[r, 1])
                if src != bg and dst != bg and src != dst:
                    return 0, 3, src, dst

    return None


def _apply_plus_rule(playfield: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = playfield.copy()
    H, W = playfield.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(playfield[r, c]) != src:
                continue
            if all(int(playfield[r + dr, c + dc]) == src for dr, dc in ARM_OFFSETS):
                for dr, dc in PLUS_OFFSETS:
                    out[r + dr, c + dc] = dst
    return out


def _count_plus_centers(playfield: np.ndarray, src: int) -> int:
    H, W = playfield.shape
    total = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(playfield[r, c]) == src and all(int(playfield[r + dr, c + dc]) == src for dr, dc in ARM_OFFSETS):
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read an edge legend to decide how to recolor embedded '+' motifs.

    Input structure:
    - bg is the most frequent color and fills both the playfield background and a separator band.
    - A 2-cell legend sits on the top edge or left edge: top uses a vertical source-over-target pair, left uses a horizontal source-left-of-target pair.
    - Below or to the right of that separator, one operative foreground color forms mixed shapes on the bg, including embedded '+' motifs.

    Rule:
    - Parse the legend to obtain source_color -> target_color and isolate the playfield beyond the separator band.
    - In the playfield only, any source-colored pixel whose 4 orthogonal neighbors are also source-colored is a '+' center.
    - Recolor that center and its 4 orthogonal neighbors to the legend's target color.
    - Leave the legend, the separator band, and every non-matching pixel unchanged.

    Key insight:
    - The geometric search is still the original local '+' test; the new challenge is that the recolor destination is no longer a fixed constant and the instruction pixels must be excluded from the drawable area before scanning.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    parsed = _parse_legend(g, bg)
    if parsed is None:
        return out

    r0, c0, src, dst = parsed
    playfield = g[r0:, c0:]
    out[r0:, c0:] = _apply_plus_rule(playfield, src, dst)
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check if setting (r,c) to fg would create any new plus center at (r,c) or its neighbors."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _embed_with_legend(
    playfield: np.ndarray, orientation: str, src: int, dst: int, bg: int, rng: np.random.Generator
) -> np.ndarray:
    H, W = playfield.shape
    if orientation == "top":
        full = np.full((H + 3, W), bg, dtype=int)
        full[3:, :] = playfield
        c = int(rng.integers(0, W))
        full[0, c] = src
        full[1, c] = dst
        return full

    full = np.full((H, W + 3), bg, dtype=int)
    full[:, 3:] = playfield
    r = int(rng.integers(0, H))
    full[r, 0] = src
    full[r, 1] = dst
    return full


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    orientation = str(rng.choice(["top", "left"]))
    bg = 2
    src = int(rng.choice(PALETTE))
    dst = int(rng.choice([c for c in PALETTE if c != src]))

    if orientation == "top":
        H = int(rng.integers(10, 20))
        W = int(rng.integers(10, 23))
    else:
        H = int(rng.integers(10, 23))
        W = int(rng.integers(10, 20))

    playfield = np.full((H, W), bg, dtype=int)

    K = int(rng.integers(1, 10))
    centers: list[tuple[int, int]] = []
    for _ in range(K * 40):
        if len(centers) >= K:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
    if not centers:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_OFFSETS:
            playfield[cr + dr, cc + dc] = src

    dirs = list(ARM_OFFSETS)
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(len(dirs), size=n_ext, replace=False)
        for di in chosen:
            dr, dc = dirs[int(di)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr, nc = tip_r + dr * step, tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                playfield[nr, nc] = src

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(playfield != bg))

    bg_positions = list(zip(*np.where(playfield == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(playfield, int(r), int(c), src):
            playfield[int(r), int(c)] = src
            current_fg += 1

    solved_playfield = _apply_plus_rule(playfield, src, dst)
    plus_count = _count_plus_centers(playfield, src)
    if plus_count < 1 or np.array_equal(solved_playfield, playfield):
        return None

    inp = _embed_with_legend(playfield, orientation, src, dst, bg, rng)
    expected = inp.copy()
    if orientation == "top":
        expected[3:, :] = solved_playfield
    else:
        expected[:, 3:] = solved_playfield

    if int(np.sum(inp != bg)) >= inp.size // 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
