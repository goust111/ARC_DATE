# Combo variant of 45a5af55 (index 22)
# combo_spec: D9/1+D10/2
# Name: 数量编码旋转嵌套

from __future__ import annotations

import numpy as np

# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Base task example, bottom->top runs [(4, 2), (7, 1), (3, 2)]:
#    - original output: center 2x2 color 4, wrap 1-thick color 7 frame,
#      then wrap 2-thick color 3 frame -> a concentric square nest.
#    Required fusion:
#    - first build that same canonical nest
#    - then rotate by 90 * (number_of_runs mod 4)
#    Mathematical issue:
#    - a perfect concentric square nest is invariant under quarter-turns, so
#      the added rotation would be visually empty on every sample.
#    Non-degenerate realization used here:
#    - after building the standard nest, add one small canonical "north tab"
#      in the outermost color on a 1-cell padding border, then rotate the whole
#      marked image. The task still asks the solver to do "build first, rotate
#      second", but the rotation now has a visible carrier.
#    Small trace with 3 runs:
#    - canonical marked nest has the tab centered on the top side
#    - 3 mod 4 = 3, so np.rot90(..., 3) moves that tab to the right side
#    - the square layers stay the same, only the global orientation changes.
#
# b) Complexity calibration:
#    - The input remains visually simple: only full-width stripes.
#    - The base geometric reasoning stays intact: stripe height still encodes
#      center size / frame thickness.
#    - The only extra latent variable is low-entropy: number of stripes modulo 4.
#      The tab makes that second stage observable without adding a second object
#      family or any heavy symbolic legend.
#
# c) Implicit assumptions review against the original main():
#    - Original parsing assumes every row is uniform across the whole width.
#      Still required; generator enforces it.
#    - Original grouping assumes adjacent stripes of the same color should merge.
#      Still valid; generator avoids equal adjacent colors.
#    - Original construction assumes square side depends only on stripe heights.
#      Still valid for the canonical nest.
#    - New step adds one assumption: there is an available background color for
#      the padding border and tab carrier. We reserve 0 for that purpose and keep
#      generated stripe colors in 1..9.
#    - New rotation branch assumes the tab exists. The generator always builds
#      at least two stripes, so the marked output side is at least 5 and the tab
#      always fits cleanly.
#
# d) Placement legality:
#    - Input generator only stacks full-width stripes vertically, so there is no
#      overlap or out-of-bounds placement in the input.
#    - Output is produced analytically: canonical square side
#      h0 + 2*sum(other_heights), then padded by 1 on each side. The generator
#      rejects cases where this marked output would exceed 30x30.
#
# e) Visual observability:
#    - Across examples, stripe count visibly changes the tab orientation:
#      2 stripes -> bottom tab after 180 rotation
#      3 stripes -> right tab after 270 rotation
#      4 stripes -> top tab (no extra turn)
#      5 stripes -> left tab after 90 rotation
#    - The generator cycles through these run counts so a short sample set can
#      expose multiple branches of the count->rotation code.
#
# f) Visual cleanliness:
#    - The scene still contains just one stripe stack in the input and one nested
#      figure in the output. The tab sits on a dedicated 1-cell border, so it
#      does not clutter the frame interiors or overlap any other structure.
#
# g) Inferrability:
#    - No hidden color arithmetic is introduced. Colors keep their original role.
#      The new symbolic parameter is stripe count, which is directly visible.
#      The tab orientation is a direct geometric consequence of the count-coded
#      quarter-turn, making the fusion rule inferable from a few examples.

BG = 0
_RUN_COUNT_CYCLE = (2, 3, 4, 5)
_gen_counter = 0


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2D array")

    height, _width = g.shape
    runs: list[tuple[int, int]] = []
    r = int(height - 1)
    while r >= 0:
        color = int(g[r, 0])
        if color == BG:
            raise ValueError("input stripes must use non-background colors")
        if not bool((g[r] == color).all()):
            raise ValueError("each input row must be uniform across the full width")

        run_h = 1
        rr = int(r - 1)
        while rr >= 0:
            row_color = int(g[rr, 0])
            if not bool((g[rr] == row_color).all()):
                raise ValueError("each input row must be uniform across the full width")
            if row_color != color:
                break
            run_h += 1
            rr -= 1

        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _build_standard_nested_square(runs: list[tuple[int, int]]) -> np.ndarray:
    center_color, center_size = runs[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for color, thickness in runs[1:]:
        side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((side, side), color, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def _add_canonical_north_tab(square: np.ndarray, tab_color: int) -> np.ndarray:
    side = int(square.shape[0])
    if square.shape[0] != square.shape[1]:
        raise ValueError("expected square intermediate output")

    out = np.full((side + 2, side + 2), BG, dtype=int)
    out[1 : side + 1, 1 : side + 1] = square

    center = 1 + side // 2
    out[0, center - 1 : center + 2] = int(tab_color)
    return out


def _build_canonical_marked_output(runs: list[tuple[int, int]]) -> np.ndarray:
    square = _build_standard_nested_square(runs)
    outer_color = int(runs[-1][0])
    return _add_canonical_north_tab(square, outer_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the standard nested square from bottom->top stripe runs, then rotate
    the marked result by a quarter-turn count encoded by the number of stripes.

    Rotation convention: np.rot90(..., k) is k times 90 degrees CCW, where
    k = number_of_stripes mod 4.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    canonical = _build_canonical_marked_output(runs)
    return np.rot90(canonical, k=len(runs) % 4)


def _sample_heights(n_runs: int) -> list[int] | None:
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    side = int(heights[0] + 2 * sum(heights[1:]))
    if side + 2 > 30:
        return None
    return heights


def _gen_attempt(n_runs: int) -> np.ndarray | None:
    heights = _sample_heights(n_runs)
    if heights is None:
        return None
    if n_runs > 2 and len(set(heights)) == 1:
        return None

    width = int(np.random.randint(6, 15))
    colors = list(np.random.choice(np.arange(1, 10), size=n_runs, replace=False))
    if any(colors[i] == colors[i + 1] for i in range(n_runs - 1)):
        return None

    stripes_top_to_bottom = [
        np.full((h, width), color, dtype=int)
        for color, h in reversed(list(zip(colors, heights)))
    ]
    return np.vstack(stripes_top_to_bottom)


def generate_input() -> np.ndarray:
    global _gen_counter

    target_n_runs = int(_RUN_COUNT_CYCLE[_gen_counter % len(_RUN_COUNT_CYCLE)])
    _gen_counter += 1

    for _ in range(400):
        result = _gen_attempt(target_n_runs)
        if result is None:
            continue

        runs = _parse_runs_bottom_to_top(result)
        fused = main(result)
        canonical = _build_canonical_marked_output(runs)

        if len(runs) != target_n_runs:
            continue
        if len(runs) % 4 != 0 and np.array_equal(fused, canonical):
            continue
        return result

    raise RuntimeError("generate_input failed after 400 attempts")
