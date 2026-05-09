# Combo variant of 1818057f (index 21)
# combo_spec: D2/3+D4/0+D15/0
# Name: 图例指导双色交换

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# ===============
# concretization_and_trace:
#   - The grid now has a fixed top instruction band: rows 0..1 contain exactly two vertical
#     color pairs, and row 2 is an all-background separator. Each column encodes one mapping
#     source-over-target, so the legend explicitly tells the solver how each foreground layer
#     should transform.
#   - The playfield starts at row 3. Inside it, yellow and green both form original-style
#     mixed components with embedded thin '+' motifs. The new twist is that some isolated
#     pixels already use the legend's target colors; those pixels must swap back to their
#     paired source colors in the output.
#   - Small trace:
#       1. Seed task: yellow '+' -> teal, everything else unchanged.
#       2. This combo: suppose the top legend shows yellow over teal in one column and
#          green over magenta in the other. Then a yellow '+' becomes teal, a green '+'
#          becomes magenta, every input teal pixel becomes yellow, and every input magenta
#          pixel becomes green.
#       3. Difference: the solver must both read two mappings from the border and apply each
#          mapping bidirectionally, but only for the correct color layer.
# complexity_calibration:
#   - Keep exactly two source layers across every sample so the D4 change is unmistakable.
#   - Keep the legend to exactly two columns so the D15 cue is readable in 2-4 examples.
#   - Vary only the two target colors across samples, while keeping source colors stable
#     (yellow and green). This keeps the mapping inferable without exploding the palette.
#   - Require both mappings to be active in every generated example: each source owns at least
#     one '+' and each target already appears in the input as a swap-back distractor.
# implicit_assumption_audit:
#   - Seed assumption `colors[0]` = the only foreground color is invalid. The solver must parse
#     explicit source->target pairs from the legend and cannot infer them from non-bg colors.
#   - Seed assumption "all recolors go to teal" is invalid. The destination now depends on the
#     matching legend column.
#   - Seed assumption "every non-bg cell belongs to the drawable field" is invalid. Legend cells
#     must be excluded from motif detection, so the solver scans only rows below the separator.
#   - Seed assumption "teal-like colors are always outputs only" is invalid. Target colors may
#     already exist in the input and must be swapped back before painting new '+' footprints.
#   - Detection must read from the untouched input playfield; if it read from partially swapped
#     output, a target distractor changed back to a source color could create a fake '+'. 
# placement_legality:
#   - The generator reserves rows 0..2 for legend + separator, then places all motifs strictly
#     in the remaining playfield.
#   - Plus centers are sampled with a one-cell motif margin and at least one extra extension cell
#     available, so they fit and stay embedded instead of isolated.
#   - Centers across both colors keep Manhattan distance >= 4, preventing overlapping footprints.
#   - Target-color distractors are placed only on current background cells away from planted plus
#     neighborhoods, so they do not blur the transformed motifs.
# visual_observability:
#   - A human can see the two instruction columns directly: the colors that appear in row 1 also
#     appear inside the input playfield, then disappear into the row-0 colors in the output.
#   - Because both yellow and green generate changed '+' footprints in each sample, the task does
#     not collapse into a one-color special case.
# visual_cleanliness:
#   - The separator row prevents legend pixels from visually merging with the scene.
#   - Swap-back distractors are kept sparse and isolated, so they read as "existing target-color
#     pixels" rather than noise glued onto the main components.
# inferability_of_change:
#   - The rule is not tied to hidden color ids: the legend is the only authority for both forward
#     recolor and reverse swap, and the generator makes both directions visibly active.
# main_assumption_audit:
#   - `_parse_top_legend()` assumes exactly two vertical legend pairs and a full-bg separator row;
#     `generate_input()` constructs exactly that layout.
#   - `_apply_playfield_rule()` assumes source and target colors are pairwise distinct so reverse
#     swaps are unambiguous; the generator samples two distinct targets outside {bg, yellow, green}.
#   - The per-mapping '+' scan assumes only source colors are eligible motif layers; the generator
#     uses target colors only for legend cells and isolated swap distractors.


BG = 2
YELLOW = 4
GREEN = 3
SOURCE_COLORS = (YELLOW, GREEN)
TARGET_POOL = (1, 5, 6, 7, 8, 9)
TOP_BAND = 3
PLUS_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
ORTHO_OFFSETS = PLUS_OFFSETS[1:]


def _parse_top_legend(grid: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    if H <= TOP_BAND or not np.all(grid[2, :] == bg):
        return None

    cols = [c for c in range(W) if int(grid[0, c]) != bg or int(grid[1, c]) != bg]
    if len(cols) != 2:
        return None

    mappings: list[tuple[int, int]] = []
    for c in cols:
        src = int(grid[0, c])
        dst = int(grid[1, c])
        if src == bg or dst == bg or src == dst:
            return None
        mappings.append((src, dst))

    sources = [src for src, _dst in mappings]
    targets = [dst for _src, dst in mappings]
    if len(set(sources)) != 2 or len(set(targets)) != 2:
        return None
    if set(sources) & set(targets):
        return None
    return mappings


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_OFFSETS)


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def _apply_playfield_rule(playfield: np.ndarray, mappings: list[tuple[int, int]]) -> np.ndarray:
    out = playfield.copy()
    for src, dst in mappings:
        out[playfield == dst] = src
    for src, dst in mappings:
        for r, c in _find_plus_centers(playfield, src):
            for dr, dc in PLUS_OFFSETS:
                out[r + dr, c + dc] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a top legend with two source->target pairs, then apply bidirectional color swaps around same-color '+' motifs.

    Input structure:
    - The most common color is the background and also fills a full separator row under the legend.
    - Rows 0..1 contain exactly two vertical legend pairs, each giving one source color above its target color.
    - Below the separator, yellow and green form mixed playfield structures with embedded '+' motifs, and some pixels may already use the legend's target colors.

    Rule:
    - Parse the two legend mappings and restrict all motif detection to the playfield below the separator.
    - For each mapping independently, any playfield pixel already in that mapping's target color swaps back to the paired source color.
    - Also for each mapping independently, any source-colored pixel whose 4 orthogonal neighbors are the same source color defines a '+' footprint; recolor that 5-cell footprint to the paired target color.
    - Leave the legend, separator, and every other playfield pixel unchanged.

    Key insight: The legend controls both directions of each color pair. Target-colored input pixels are not a second motif family to detect; they are reverse-swap evidence, while '+' detection still runs only on the untouched source-color layers in the playfield.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    mappings = _parse_top_legend(g, bg)
    if mappings is None:
        return out

    playfield = g[TOP_BAND:, :]
    out[TOP_BAND:, :] = _apply_playfield_rule(playfield, mappings)
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    candidates = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
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


def _cross_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in PLUS_OFFSETS]


def _candidate_score(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int]:
    orth_non_bg = 0
    diag_non_bg = 0
    H, W = grid.shape
    for dr, dc in ORTHO_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != bg:
            orth_non_bg += 1
    for dr in (-1, 1):
        for dc in (-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != bg:
                diag_non_bg += 1
    return orth_non_bg, diag_non_bg


def _place_swap_distractors(
    input_playfield: np.ndarray,
    output_playfield: np.ndarray,
    centers: list[tuple[int, int]],
    mappings: list[tuple[int, int]],
    bg: int,
    rng: np.random.Generator,
) -> bool:
    H, W = input_playfield.shape
    protected: set[tuple[int, int]] = set()
    for cr, cc in centers:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    protected.add((nr, nc))

    placed: list[tuple[int, int]] = []
    for src, dst in mappings:
        need = int(rng.integers(2, 5))
        candidates = [
            (r, c)
            for r in range(H)
            for c in range(W)
            if int(input_playfield[r, c]) == bg and (r, c) not in protected
        ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda pos: _candidate_score(input_playfield, pos[0], pos[1], bg))

        placed_here = 0
        for r, c in candidates:
            if any(abs(r - pr) + abs(c - pc) <= 1 for pr, pc in placed):
                continue
            orth_non_bg, diag_non_bg = _candidate_score(input_playfield, r, c, bg)
            if orth_non_bg > 0 or diag_non_bg > 1:
                continue
            input_playfield[r, c] = dst
            output_playfield[r, c] = src
            placed.append((r, c))
            placed_here += 1
            if placed_here == need:
                break
        if placed_here != need:
            return False
    return True


def _embed_top_legend(
    playfield: np.ndarray,
    legend_pairs: list[tuple[int, int]],
    legend_cols: list[int],
    bg: int,
) -> np.ndarray:
    H, W = playfield.shape
    full = np.full((H + TOP_BAND, W), bg, dtype=int)
    full[TOP_BAND:, :] = playfield

    for c, (src, dst) in zip(legend_cols, legend_pairs):
        full[0, c] = src
        full[1, c] = dst
    return full


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    full_h = int(rng.integers(10, 23))
    full_w = int(rng.integers(10, 23))
    play_h = full_h - TOP_BAND
    play_w = full_w
    if play_h < 7 or play_w < 7:
        return None

    targets = [int(x) for x in rng.choice(TARGET_POOL, size=2, replace=False)]
    mappings = [(YELLOW, targets[0]), (GREEN, targets[1])]
    src_to_dst = {src: dst for src, dst in mappings}

    input_playfield = np.full((play_h, play_w), BG, dtype=int)
    output_playfield = np.full((play_h, play_w), BG, dtype=int)

    total_cells = play_h * play_w
    max_total_pluses = min(6, max(2, total_cells // 65 + 1))
    total_pluses = int(rng.integers(2, max_total_pluses + 1))
    plus_colors = [YELLOW, GREEN]
    for _ in range(total_pluses - 2):
        plus_colors.append(int(rng.choice(SOURCE_COLORS)))
    rng.shuffle(plus_colors)

    candidates = [(r, c) for r in range(2, play_h - 2) for c in range(2, play_w - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) >= 4 for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == total_pluses:
                break
    if len(centers) != total_pluses:
        return None

    center_specs = list(zip(centers, plus_colors))
    for (r, c), src in center_specs:
        dst = src_to_dst[src]
        cells = _cross_cells(r, c)
        if any(int(output_playfield[rr, cc]) != BG for rr, cc in cells):
            return None
        for rr, cc in cells:
            input_playfield[rr, cc] = src
            output_playfield[rr, cc] = dst

    shuffled_specs = list(center_specs)
    rng.shuffle(shuffled_specs)
    for (r, c), src in shuffled_specs:
        feasible_runs: list[list[tuple[int, int]]] = []
        for dr, dc in ORTHO_OFFSETS:
            run: list[tuple[int, int]] = []
            nr, nc = r + 2 * dr, c + 2 * dc
            while 0 <= nr < play_h and 0 <= nc < play_w and int(output_playfield[nr, nc]) == BG and len(run) < 3:
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
                input_playfield[rr, cc] = src
                output_playfield[rr, cc] = src
                added += 1
        if added == 0:
            return None

    if not _place_swap_distractors(input_playfield, output_playfield, centers, mappings, BG, rng):
        return None

    current_non_bg = int(np.sum(input_playfield != BG))
    if current_non_bg >= total_cells // 2:
        return None
    min_density = max(0.26, current_non_bg / total_cells)
    if min_density > 0.41:
        return None
    target_density = float(rng.uniform(min_density, 0.41))
    target_non_bg = int(total_cells * target_density)

    source_counts = {
        YELLOW: int(np.sum(input_playfield == YELLOW)),
        GREEN: int(np.sum(input_playfield == GREEN)),
    }
    bg_positions = list(zip(*np.where(input_playfield == BG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_non_bg >= target_non_bg:
            break
        preferred = [YELLOW, GREEN] if source_counts[YELLOW] <= source_counts[GREEN] else [GREEN, YELLOW]
        if rng.random() < 0.35:
            preferred.reverse()
        for src in preferred:
            if _would_create_plus(input_playfield, int(r), int(c), src):
                continue
            input_playfield[int(r), int(c)] = src
            output_playfield[int(r), int(c)] = src
            source_counts[src] += 1
            current_non_bg += 1
            break

    if not (0.26 <= current_non_bg / total_cells <= 0.41):
        return None

    legend_cols = sorted(int(x) for x in rng.choice(play_w, size=2, replace=False))
    legend_pairs = list(mappings)
    rng.shuffle(legend_pairs)
    input_grid = _embed_top_legend(input_playfield, legend_pairs, legend_cols, BG)
    output_grid = _embed_top_legend(output_playfield, legend_pairs, legend_cols, BG)

    input_bg = int(get_bg(input_grid))
    if input_bg != BG:
        return None

    parsed = _parse_top_legend(input_grid, BG)
    if parsed is None:
        return None

    for src, dst in parsed:
        if src not in SOURCE_COLORS:
            return None
        if not _find_plus_centers(input_playfield, src):
            return None
        if not np.any((input_playfield == dst) & (output_playfield == src)):
            return None
        if not np.any(output_playfield == dst):
            return None

    if not np.array_equal(main(input_grid), output_grid):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
