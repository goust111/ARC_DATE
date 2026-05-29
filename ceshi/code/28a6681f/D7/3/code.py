# Variant of 28a6681f
# Dimension: D7 — 盆地结构形状 (obstacle)
# Change: 2-3个底部锚定的结构：U形、L形、阶梯形或单竖壁，相邻盆地壁高不等且可相互连通允许溢流... → 悬浮平台式障碍

from __future__ import annotations

import random

import numpy as np

# Design analysis
# source_task: none
# variant_spec:
#   grid: 10x10
#   colors:
#     - background is 0
#     - water is 1
#     - three distinct solid colors are sampled for the source beam, upper cup, and lower cup
#   obstacle_layout:
#     - Use a top suspended source beam touching one side edge.
#     - Place an upper suspended asymmetric cup below the beam:
#       tall wall on the entry side, horizontal floor, and a shorter wall on the overflow side.
#     - Place a lower suspended cup one column farther toward the overflow side and several rows lower.
#       This second cup is fully walled so it visibly catches the overflow in midair.
#     - No solid pixel touches the bottom row. The floating nature must be visually obvious.
#   concrete_trace:
#     - Canonical left-to-right sample:
#       beam at row 1, cols 0..4; source water is rewound onto row 0, cols 0..3 or 0..4.
#       upper cup uses left wall col 4 rows 2..4, floor row 4 cols 4..8, short right lip col 8 rows 3..4.
#       lower cup uses walls col 5 and 9 rows 6..8 and floor row 8 cols 5..9.
#     - Water first slides off the beam's only open side, falls beside the tall wall, and fills the upper cup
#       from wall-adjacent cells so diagonal motion is unique.
#     - After the upper cup reaches its short right lip, later water spills over that lip and falls into the
#       lower suspended cup, again entering beside a wall to avoid symmetry.
#     - Old logic assumed notable non-bottom shapes near the top could be special guides; in this variant those
#       shapes are legitimate suspended basins, so that assumption must be removed from main().
#   complexity_calibration:
#     - Keep one geometric family plus a horizontal mirror. This is enough repetition for humans to infer the
#       mechanism while still showing that the cascade direction can reverse.
#     - Only two rewind counts are used (4 or 5 source pixels). This keeps the input/output relation readable.
#   implicit_assumptions_review:
#     - The old "upper non-bottom component with many diagonals is a guide" check depends on bottom-anchored
#       basins and is invalid here, so it is removed.
#     - settle() still requires each move choice to be unique. The generator guarantees this by making every
#       landing/overflow point wall-adjacent and every launch beam one-sided.
#     - The strict output enclosure check remains valid because final water is only left inside cups, never on
#       an exposed beam edge.
#   placement_legality:
#     - The beam, upper cup, and lower cup are laid out on disjoint rows and columns except for intentional
#       support adjacency; they never overlap or leave the grid.
#     - The lower cup is offset exactly one column toward the overflow side so spilled water drops into its
#       interior, not onto its rim.
#   visual_observability:
#     - In the output, water occupies two different suspended support levels: the upper cup row and the lower
#       cup row. This makes the D7 change visible even without animation.
#     - The bottom row stays empty of obstacles, which clearly distinguishes this task from the original
#       bottom-anchored basin family.
#   tidy_visuals:
#     - A blank band separates the two cups, so the cascade path is legible rather than cluttered.
#   color_rule:
#     - Colors never control motion or ordering. They only make the three suspended structures visually distinct.
#   main_assumptions_after_change:
#     - generate_input() always returns supported source water on the beam plus some already-stable carried-over
#       cup water, so main()'s supported-input assumption still holds.
#     - Because rewound water is always taken from the topmost stable cells first, no remaining input water is
#       left unsupported.

BG = 0
WATER = 1
SIZE = 10
TOTAL_WATER = 9
SOURCE_INJECT = (0, 2)


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water pixels falling through suspended cups and beams.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors are rigid obstacles that may float in midair instead of being anchored to the bottom.
    - Water starts supported on a shelf, beam, or previously settled water inside a suspended cup.

    Rule:
    - Each water pixel falls straight down through bg cells.
    - When blocked below, it may step diagonally down only through the unique open side gap.
    - When resting on a supported ledge, it slides horizontally toward the unique reachable drop column.
    - Repeating these local moves makes water collect on a floating platform, then spill from a shorter lip into lower suspended cups.

    Key insight: The physics stays local, but the obstacle geometry changes the story: stacked midair trays create visible multi-stage cascades instead of only bottom-fed basins.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [x for x in range(left, right + 1) if int(g[rr + 1, x]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted(((int(r), int(c)) for r, c in np.argwhere(g == int(WATER))), reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _build_canonical_layout(colors: tuple[int, int, int]) -> np.ndarray:
    beam_color, upper_color, lower_color = colors
    g = np.zeros((SIZE, SIZE), dtype=int)

    # One-sided launch beam along the top edge.
    g[1, 0:5] = beam_color

    # Upper suspended cup: tall entry wall at col 4, short overflow lip at col 8.
    g[4, 4:9] = upper_color
    g[2:5, 4] = upper_color
    g[3:5, 8] = upper_color

    # Lower suspended cup catches overflow beside its right wall.
    g[8, 5:10] = lower_color
    g[6:9, 5] = lower_color
    g[6:9, 9] = lower_color

    return g


def _pour_output(base: np.ndarray) -> np.ndarray:
    g = np.array(base, dtype=int).copy()
    for _ in range(TOTAL_WATER):
        sr, sc = SOURCE_INJECT
        assert int(g[sr, sc]) == int(BG), "source cell should clear after each pour"
        g[sr, sc] = int(WATER)
        g = main(g, check_input=False, check_output=False)
    return g


def _rewind_output(output: np.ndarray, rewind_count: int) -> np.ndarray:
    g = np.array(output, dtype=int).copy()
    movable = sorted((int(r), int(c)) for r, c in np.argwhere(g == int(WATER)))
    removed = movable[:rewind_count]
    assert len(removed) == rewind_count
    for r, c in removed:
        g[r, c] = int(BG)

    for c in range(rewind_count):
        assert int(g[0, c]) == int(BG)
        g[0, c] = int(WATER)

    return g


def _support_levels(output: np.ndarray) -> tuple[int, ...]:
    levels = sorted(
        {
            int(r)
            for r, c in np.argwhere(output == int(WATER))
            if int(r) < output.shape[0] - 1
            and int(output[int(r) + 1, int(c)]) not in (int(BG), int(WATER))
        }
    )
    return tuple(levels)


def generate_example(
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool | tuple[int, ...]]]:
    if rng is None:
        rng = random.Random()

    colors = tuple(rng.sample(list(range(2, 10)), 3))
    rewind_count = rng.choice((4, 5))
    mirror = rng.choice((False, True))

    output = _pour_output(_build_canonical_layout(colors))
    input_grid = _rewind_output(output, rewind_count)

    if mirror:
        output = np.fliplr(output).copy()
        input_grid = np.fliplr(input_grid).copy()

    metadata: dict[str, int | bool | tuple[int, ...]] = {
        "rewind_count": rewind_count,
        "mirror": mirror,
        "support_levels": _support_levels(output),
        "floating_obstacles_touch_bottom": bool(np.any((output[SIZE - 1] != BG) & (output[SIZE - 1] != WATER))),
    }
    return input_grid, output, metadata


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(100):
        input_grid, output, _metadata = generate_example(rng)
        try:
            if np.array_equal(main(input_grid), output):
                return input_grid
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid suspended-platform sample")
