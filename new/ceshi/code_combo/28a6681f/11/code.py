# Combo variant of 28a6681f (index 11)
# combo_spec: D3/0+D18/0
# Name: 最远落点+允许歧义消歧

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D3/0 + D18/0
# --------------------------------------
# a) Concretization + behavior trace:
#    - The fused family stays on a 10x10 board with water=1 and bg=0, but the generator now
#      centers the puzzle around one repeated shelf pattern: a supported top water block drains
#      over a solid roof row that has three vertical drop shafts.
#    - A concrete template uses rows 0..3, cols 2..4 as source water, with shaft mouths at
#      columns 0, 6, and 8. After some rightmost source cells drain away, a remaining source
#      pixel can reach shelf position (row 3, col 4) and see drop distances 4, 2, and 4.
#      * Original task: horizontal routing would pick the unique nearest shaft at col 6.
#      * D3/0 alone: it should instead target the farthest distance 4, so the candidate set is
#        {0, 8}.
#      * D18/0 fused into that step: the farthest choice is no longer required to be unique, so
#        the tie is resolved deterministically by choosing the leftmost farthest shaft, col 0.
#    - The same source block contains several water pixels, so the farthest-vs-nearest change and
#      the left tie-break both appear repeatedly inside one example rather than as a one-off move.
#
# b) Complexity calibration:
#    - The geometry family is intentionally narrow: one top source shelf and three shaft mouths.
#      This is enough to make the fused rule visible without introducing unrelated clutter.
#    - Variation is still present in repeated, human-visible axes: source height 3 or 4, source
#      width 3 or 4, several curated shaft triples, left/right mirroring, shaft depths, and 3
#      recolored solid bands. These keep examples from collapsing to one frozen bitmap.
#    - The generator rejects any candidate where the combo output is too close to either the
#      nearest-drop ablation or the right-priority ablation, so both fused dimensions stay active.
#
# c) Hidden-assumption audit of the original main():
#    - The original `settle()` assumed diagonal and horizontal ambiguity never occurs and asserted
#      `len(opts) == 1` or `len(nearest) == 1`. That is invalid here and must be replaced with an
#      explicit tie-break.
#    - The original nearest-drop branch moved only one horizontal step before recomputing. That is
#      safe for the nearest policy but can oscillate under the farthest policy. The fused solver
#      therefore commits to the chosen farthest target column for the whole supported shelf contact.
#    - The old upper-guide color sanity checks relied on task-specific assumptions about colors in
#      the top rows versus the bottom row. This generator no longer uses that guide subfamily, so
#      those checks are removed rather than becoming accidental blockers.
#    - The original final assertion that every output water cell must have non-background on the
#      left, below, and right is too strong for this shaft family: a water cell may legally keep a
#      side-adjacent empty cell when that side offers no diagonal or horizontal drop continuation.
#      The fused solver therefore checks stability by "no legal next move" rather than by pure
#      three-sided occupancy.
#
# d) Placement legality:
#    - Every layout is built from explicit roof height, source rectangle, shaft columns, and shaft
#      depths. The source rectangle occupies only rows above the roof row; shafts are carved only
#      inside the bottom solid mass. That prevents overlap and keeps everything in bounds.
#    - Mirroring happens after occupancy is drawn, so it cannot create partial off-grid objects.
#    - Recoloring touches only solid cells, never background or water.
#
# e) Visual observability:
#    - A human can infer the fused change from a few examples because the same visible shelf pattern
#      repeats: there is always a nearer shaft and also two equally far shaft mouths.
#    - Outputs differ in an interpretable way: water bypasses the nearer mouth and also prefers the
#      left far mouth over the right far mouth.
#    - The solver also resolves diagonal ties by the same left-priority rule, but the generator does
#      not depend on diagonal ambiguity in every sample. Repeated horizontal ambiguity is enough to
#      make the disambiguation policy learnable.
#
# f) Visual cleanliness:
#    - The board has only three regions: empty top space, a compact source block, and one coherent
#      lower landform with clean vertical shafts. There are no overlapping markers or legends.
#    - Three solid colors are assigned in broad column bands, so the static structure stays legible
#      while water remains the only moving color.
#
# g) Change inferability / color discipline:
#    - Colors do not encode ordering. All decisions depend only on visible geometry and absolute
#      left-versus-right position.
#    - Solid colors vary independently from the path rule, so a solver cannot rely on palette order.
#    - Generator-side metrics compare the combo output against two deterministic ablations:
#      nearest+left and farthest+right. This prevents degeneration to either single-variant behavior.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - The board is always 10x10 and always contains at least one water pixel.
#    - Source water is a supported rectangle and therefore satisfies the inherited input-side
#      support constraint.
#    - At least one horizontal ambiguity event occurs under the combo solver.
#    - The combo output differs from both the nearest-drop and right-priority ablations by at least
#      four cells, so both the farthest policy and the tie-break policy materially affect the sample.


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

LAYOUT_CHOICES = (
    {
        "roof_row": 4,
        "source_start": 2,
        "source_width": 3,
        "drops": (0, 6, 8),
        "depth_options": ((6, 6, 6), (5, 6, 6), (6, 5, 6)),
        "mirror_options": (False, True),
    },
    {
        "roof_row": 4,
        "source_start": 2,
        "source_width": 3,
        "drops": (0, 7, 8),
        "depth_options": ((6, 6, 6), (5, 6, 6), (6, 5, 6)),
        "mirror_options": (False, True),
    },
    {
        "roof_row": 4,
        "source_start": 2,
        "source_width": 3,
        "drops": (0, 1, 6),
        "depth_options": ((6, 6, 6), (5, 6, 6), (6, 5, 6)),
        "mirror_options": (False,),
    },
    {
        "roof_row": 4,
        "source_start": 2,
        "source_width": 3,
        "drops": (0, 1, 8),
        "depth_options": ((6, 6, 6), (6, 5, 6), (5, 6, 6)),
        "mirror_options": (True,),
    },
    {
        "roof_row": 3,
        "source_start": 2,
        "source_width": 4,
        "drops": (0, 6, 8),
        "depth_options": ((7, 7, 7), (7, 6, 7), (6, 7, 7)),
        "mirror_options": (False, True),
    },
    {
        "roof_row": 3,
        "source_start": 1,
        "source_width": 4,
        "drops": (0, 6, 8),
        "depth_options": ((7, 7, 7), (7, 6, 7), (6, 7, 7)),
        "mirror_options": (False, True),
    },
    {
        "roof_row": 3,
        "source_start": 2,
        "source_width": 4,
        "drops": (1, 7, 9),
        "depth_options": ((7, 7, 7), (7, 6, 7), (6, 7, 7)),
        "mirror_options": (False, True),
    },
)


def _assert_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.shape == GRID_SHAPE, f"expected grid shape {GRID_SHAPE}, got {g.shape}"
    assert np.any(g == int(WATER)), "input must contain water"
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _diagonal_options(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    opts: list[tuple[int, int]] = []
    if (
        r + 1 < h
        and c - 1 >= 0
        and int(g[r, c - 1]) == int(BG)
        and int(g[r + 1, c - 1]) == int(BG)
    ):
        opts.append((int(r + 1), int(c - 1)))
    if (
        r + 1 < h
        and c + 1 < w
        and int(g[r, c + 1]) == int(BG)
        and int(g[r + 1, c + 1]) == int(BG)
    ):
        opts.append((int(r + 1), int(c + 1)))
    return opts


def _drop_columns(grid: np.ndarray, r: int, c: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if r + 1 >= h:
        return []
    left = int(c)
    while left - 1 >= 0 and int(g[r, left - 1]) == int(BG):
        left -= 1
    right = int(c)
    while right + 1 < w and int(g[r, right + 1]) == int(BG):
        right += 1
    return [int(x) for x in range(int(left), int(right) + 1) if int(g[r + 1, int(x)]) == int(BG)]


def _simulate(
    grid: np.ndarray,
    *,
    prefer_farthest: bool,
    tie_break: str,
    check_input: bool = True,
    check_output: bool = True,
    collect_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    assert tie_break in {"left", "right"}, f"unsupported tie break: {tie_break}"
    prefer_left = tie_break == "left"
    water_before = int(np.count_nonzero(g == int(WATER)))

    metrics = {
        "diag_ties": 0,
        "horiz_ties": 0,
        "tie_break_choices": 0,
        "left_tie_breaks": 0,
        "farthest_slide_events": 0,
        "max_slide_distance": 0,
        "moved_pixels": 0,
    }

    if check_input:
        _assert_input(g)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            diagonal_options = _diagonal_options(g, int(rr), int(cc))
            if diagonal_options:
                if len(diagonal_options) > 1:
                    metrics["diag_ties"] += 1
                    metrics["tie_break_choices"] += 1
                    if prefer_left:
                        metrics["left_tie_breaks"] += 1
                rr, cc = (
                    min(diagonal_options, key=lambda rc: int(rc[1]))
                    if prefer_left
                    else max(diagonal_options, key=lambda rc: int(rc[1]))
                )
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                drops = _drop_columns(g, int(rr), int(cc))
                if not drops:
                    break

                target_distance = (
                    max(abs(int(x - cc)) for x in drops)
                    if prefer_farthest
                    else min(abs(int(x - cc)) for x in drops)
                )
                best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(target_distance)]
                if len(best_cols) > 1:
                    metrics["horiz_ties"] += 1
                    metrics["tie_break_choices"] += 1
                    if prefer_left:
                        metrics["left_tie_breaks"] += 1
                best = min(best_cols) if prefer_left else max(best_cols)
                assert best != int(cc), "supported horizontal target cannot be the current column"

                if prefer_farthest:
                    metrics["farthest_slide_events"] += 1
                    metrics["max_slide_distance"] = max(metrics["max_slide_distance"], abs(int(best - cc)))
                    for x in range(min(int(cc), int(best)) + 1, max(int(cc), int(best)) + 1):
                        assert int(g[rr, int(x)]) == int(BG), "farthest slide must stay on the same empty run"
                    cc = int(best)
                else:
                    nx = int(cc - 1 if best < int(cc) else cc + 1)
                    assert int(g[rr, nx]) == int(BG), "nearest slide must move one open step"
                    cc = int(nx)
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")

        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                moved = True
                metrics["moved_pixels"] += 1
            assert int(g[rr, cc]) == int(BG), "water paths should not overlap"
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    water_after = int(np.count_nonzero(g == int(WATER)))
    assert water_before == water_after, "water count must be conserved"

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "output water must not have empty space below"
            assert not _diagonal_options(g, int(rr), int(cc)), "output water must not have a legal diagonal move"
            assert not _drop_columns(g, int(rr), int(cc)), "output water must not have a legal horizontal drop route"

    if collect_metrics:
        return g, metrics
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate shelf-routed basin filling where supported water commits to the farthest reachable drop and resolves ties deterministically.

    Input structure:
    - Background is 0 and moving water is color 1 on a 10x10 grid.
    - All other colors are fixed solids forming one compact roof-and-shaft landform anchored at the bottom.
    - Input water appears as a supported top block resting on that roof, so several pixels must drain through the same set of shaft mouths.

    Rule:
    - Water keeps falling straight down through background cells whenever possible.
    - If blocked from below, it may move diagonally downward through an open side corridor; if both diagonals are open, choose the left one.
    - When resting on a supported horizontal background run, inspect every drop column in that run and commit to the farthest reachable one. If multiple farthest drops are tied, choose the leftmost tied target before resuming gravity.
    - Repeat the local settling pass until no water pixel moves and every water cell is trapped against solids, other water, or the grid edge.

    Key insight:
    The distinctive evidence is on the shelf: water bypasses a nearer shaft, aims for the most distant available fall, and still remains deterministic because equal farthest choices always break to the left.
    """
    result = _simulate(
        grid,
        prefer_farthest=True,
        tie_break="left",
        check_input=bool(check_input),
        check_output=bool(check_output),
        collect_metrics=False,
    )
    assert isinstance(result, np.ndarray)
    return result


def _build_case(
    *,
    roof_row: int,
    source_start: int,
    source_width: int,
    drops: tuple[int, int, int],
    depths: tuple[int, int, int],
    mirror: bool,
) -> np.ndarray:
    g = np.zeros(GRID_SHAPE, dtype=int)
    g[int(roof_row) :, :] = 2
    for col, depth in zip(drops, depths):
        for r in range(int(roof_row), min(GRID_SHAPE[0], int(roof_row + depth))):
            g[int(r), int(col)] = int(BG)
    g[: int(roof_row), int(source_start) : int(source_start + source_width)] = int(WATER)
    if mirror:
        g = np.fliplr(g)
    return g


def _recolor_structure(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    left_color, center_color, right_color = rng.sample(COLOR_POOL, 3)
    col_colors = np.array(
        [
            left_color if c <= 2 else right_color if c >= 7 else center_color
            for c in range(out.shape[1])
        ],
        dtype=int,
    )
    solid_mask = (out != int(BG)) & (out != int(WATER))
    for c in range(out.shape[1]):
        rows = solid_mask[:, int(c)]
        out[rows, int(c)] = int(col_colors[int(c)])
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    combo, combo_metrics = _simulate(
        grid,
        prefer_farthest=True,
        tie_break="left",
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    assert isinstance(combo, np.ndarray)
    nearest_left = _simulate(
        grid,
        prefer_farthest=False,
        tie_break="left",
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    farthest_right = _simulate(
        grid,
        prefer_farthest=True,
        tie_break="right",
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    assert isinstance(nearest_left, np.ndarray)
    assert isinstance(farthest_right, np.ndarray)
    return {
        **combo_metrics,
        "combo_vs_nearest_diff": int(np.sum(combo != nearest_left)),
        "combo_vs_far_right_diff": int(np.sum(combo != farthest_right)),
        "water_count": int(np.count_nonzero(combo == int(WATER))),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = rng.choice(LAYOUT_CHOICES)
        grid = _build_case(
            roof_row=int(layout["roof_row"]),
            source_start=int(layout["source_start"]),
            source_width=int(layout["source_width"]),
            drops=tuple(int(v) for v in layout["drops"]),
            depths=tuple(int(v) for v in rng.choice(layout["depth_options"])),
            mirror=bool(rng.choice(layout["mirror_options"])),
        )
        grid = _recolor_structure(grid, rng)
        metrics = _behavior_metrics(grid)
        ambiguity_events = int(metrics["diag_ties"] + metrics["horiz_ties"])
        if ambiguity_events == 0:
            continue
        if metrics["left_tie_breaks"] < 1:
            continue
        if metrics["combo_vs_nearest_diff"] < 4:
            continue
        if metrics["combo_vs_far_right_diff"] < 4:
            continue
        if metrics["farthest_slide_events"] < 6:
            continue
        if metrics["max_slide_distance"] < 4:
            continue
        if metrics["moved_pixels"] < 8:
            continue
        return grid

    raise AssertionError("failed to sample a visible D3/0 + D18/0 combo input")
