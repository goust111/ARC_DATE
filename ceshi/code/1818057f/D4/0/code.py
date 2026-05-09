# Variant of 1818057f
# Dimension: D4 — 前景色数量与背景色关系 (meta)
# Change: 恰好1种前景色(黄色4)在1种背景色(红色2)上，前景密度26-41%... → 两种前景色，各自独立检测'+'字

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


"""
design_analysis:
  variant_summary:
    - Keep the original geometric rule: a '+' is still a center pixel with 4 orthogonal
      neighbors of the same foreground color.
    - Replace the single yellow foreground layer with two simultaneous layers:
      yellow (4) and green (3) on the same red background (2).
    - Recolor yellow '+' footprints to teal (8), and green '+' footprints to orange (7).

  concretization_and_trace:
    - In the input, both yellow and green pixels form a mixed field of lines, blobs, and
      embedded plus motifs. A yellow '+' is only completed by yellow neighbors; a green '+'
      is only completed by green neighbors.
    - Original behavior at one interior cell:
      1. Pick the only non-background color.
      2. If the center and 4 orthogonal neighbors all use that color, recolor the 5 cells to teal.
    - Variant behavior at the same location:
      1. Check whether the cell is yellow with four yellow orthogonal neighbors. If yes,
         recolor that 5-cell footprint to teal.
      2. Separately check whether the cell is green with four green orthogonal neighbors. If yes,
         recolor that footprint to orange.
      3. A mixed-color cross-like shape does nothing, because the rule is color-layer-local.

  complexity_calibration:
    - The task is visibly different from the seed because both foreground colors are always present
      and both generate output changes in accepted samples.
    - The mapping is fixed and repeated across examples:
      yellow -> teal, green -> orange.
      This avoids hidden ordering logic and gives the solver repeated evidence.
    - Counts stay moderate so the grid remains readable: each sample has 2-6 total pluses, with at
      least one plus of each foreground color.

  implicit_assumption_audit:
    - Seed assumption: `colors[0]` is the only foreground color. This is invalid now, so `main()`
      must iterate explicit source->target color pairs.
    - Seed assumption: every match is recolored to 8. This is invalid now, so recoloring must depend
      on which color layer produced the match.
    - Seed assumption: distractor generation only needs to avoid accidental pluses for one color.
      This is invalid now, so the checker must be reused for both yellow and green placements.
    - Seed assumption: updating the output in place is harmless because all matches share one color.
      In the variant it is still safe only because detection always reads from the immutable input grid,
      never from partially recolored output.

  placement_legality:
    - Plus centers are sampled away from the border exactly as in the seed, so each footprint fits and
      each arm can extend by at least one extra cell.
    - Centers across both colors keep Manhattan distance >= 4, so footprints do not overlap.
    - Footprints, extensions, and distractors are only written onto background cells, so colors never
      overwrite one another.
    - Each planted plus receives at least one same-color extension, preserving the original "embedded,
      not isolated" character.

  observability_and_cleanliness:
    - The generator rejects any sample where one color would be absent from the transformed output.
    - Density remains in the original 26-41% range, background stays most common, and distractors are
      added one cell at a time only when they do not create new same-color plus centers.
    - Because the two recolorings are tied to repeated, visible shape matches, a human can infer the
      rule from a few IO pairs without relying on hidden color-id conventions.

  new_solver_assumptions_to_guarantee:
    - `main()` assumes the meaningful foreground layers are yellow and green. `generate_input()` fixes
      the palette accordingly.
    - `main()` assumes plus detection should be independent per color layer. `generate_input()` prevents
      accidental same-color pluses outside the planted motifs, so every recoloring has an intended cause.
"""

BG = 2
YELLOW = 4
GREEN = 3
TEAL = 8
ORANGE = 7
SRC_TO_DST = {YELLOW: TEAL, GREEN: ORANGE}
PLUS_DIRS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTH_DIRS = PLUS_DIRS[1:]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTH_DIRS)


def _plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect same-color '+' motifs separately in yellow and green, then recolor each layer differently.

    Input structure:
    - The background is the most common color.
    - Two foreground colors, yellow and green, are mixed on that background as scattered shapes and
      embedded '+' motifs.

    Rule:
    - Inspect yellow cells only against yellow neighbors, and green cells only against green neighbors.
    - Whenever a cell and its 4 orthogonal neighbors all share one of those foreground colors, recolor
      that 5-cell '+' footprint to the target for that layer: yellow -> teal, green -> orange.
    - Leave every other cell unchanged.

    Key insight:
    - Detection is independent per color layer. Nearby pixels of the other foreground color never help
      complete a '+', so the final output is the union of the yellow-layer hits and the green-layer hits.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    if np.all(g == bg):
        return out

    H, W = g.shape
    for fg, dst in SRC_TO_DST.items():
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if not _is_plus_center(g, r, c, fg):
                    continue
                for dr, dc in PLUS_DIRS:
                    out[r + dr, c + dc] = dst
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether writing fg at (r,c) would create any new same-color plus nearby."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_DIRS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _cross_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in PLUS_DIRS]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    total = H * W

    output = np.full((H, W), BG, dtype=int)

    max_total_pluses = min(6, max(2, total // 70 + 1))
    total_pluses = int(rng.integers(2, max_total_pluses + 1))
    plus_colors = [YELLOW, GREEN]
    for _ in range(total_pluses - 2):
        plus_colors.append(int(rng.choice([YELLOW, GREEN])))
    rng.shuffle(plus_colors)

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - er) + abs(c - ec) >= 4 for er, ec in centers):
            centers.append((r, c))
            if len(centers) == total_pluses:
                break
    if len(centers) != total_pluses:
        return None

    for (r, c), fg in zip(centers, plus_colors):
        for rr, cc in _cross_cells(r, c):
            if int(output[rr, cc]) != BG:
                return None
        for rr, cc in _cross_cells(r, c):
            output[rr, cc] = SRC_TO_DST[fg]

    center_specs = list(zip(centers, plus_colors))
    rng.shuffle(center_specs)
    for (r, c), fg in center_specs:
        feasible_runs: list[list[tuple[int, int]]] = []
        for dr, dc in ORTH_DIRS:
            run: list[tuple[int, int]] = []
            nr, nc = r + 2 * dr, c + 2 * dc
            while 0 <= nr < H and 0 <= nc < W and int(output[nr, nc]) == BG and len(run) < 3:
                run.append((nr, nc))
                nr += dr
                nc += dc
            if run:
                feasible_runs.append(run)
        if not feasible_runs:
            return None
        rng.shuffle(feasible_runs)
        n_dirs = 1 if len(feasible_runs) == 1 else int(rng.integers(1, min(2, len(feasible_runs)) + 1))
        added = 0
        for run in feasible_runs[:n_dirs]:
            ext_len = int(rng.integers(1, min(3, len(run)) + 1))
            for rr, cc in run[:ext_len]:
                if int(output[rr, cc]) != BG:
                    return None
                output[rr, cc] = fg
                added += 1
        if added == 0:
            return None

    inp = output.copy()
    for fg, dst in SRC_TO_DST.items():
        inp[inp == dst] = fg

    current_fg = int(np.sum(inp != BG))
    if current_fg >= total // 2:
        return None

    min_density = max(0.26, current_fg / total)
    if min_density > 0.41:
        return None
    target_density = float(rng.uniform(min_density, 0.41))
    target_fg = int(total * target_density)

    color_counts = {
        YELLOW: int(np.sum(inp == YELLOW)),
        GREEN: int(np.sum(inp == GREEN)),
    }
    bg_positions = list(zip(*np.where(inp == BG)))
    rng.shuffle(bg_positions)

    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        preferred = [YELLOW, GREEN] if color_counts[YELLOW] <= color_counts[GREEN] else [GREEN, YELLOW]
        if rng.random() < 0.35:
            preferred.reverse()
        for fg in preferred:
            if _would_create_plus(inp, int(r), int(c), fg):
                continue
            inp[r, c] = fg
            output[r, c] = fg
            color_counts[fg] += 1
            current_fg += 1
            break

    if not (0.26 <= current_fg / total <= 0.41):
        return None
    if current_fg >= total // 2:
        return None

    if not np.any(output == TEAL) or not np.any(output == ORANGE):
        return None

    if not _plus_centers(inp, YELLOW) or not _plus_centers(inp, GREEN):
        return None

    if not np.array_equal(main(inp), output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
