# Variant of 28a6681f
# Dimension: D18 — 歧义解消规则 (path)
# Change: 对角线移动和水平滑动都要求唯一解（双向可行时assert失败），通过盆地设计保证无歧义... → 歧义时水分裂

from __future__ import annotations

import random

import numpy as np

# Design analysis for the D18 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The original task uses supported blue source pixels that move through empty
#      background while solid colored structures stay fixed.
#    - This variant changes only the ambiguity rule: if a water pixel is blocked
#      below and both diagonals are equally open, it splits into two descendants;
#      likewise, if horizontal sliding sees equally near left/right drop columns,
#      it splits into two descendants that move one step left and right.
#    - Small trace:
#      original: water on top of a pillar with both lower diagonals open raises
#      "ambiguous diagonal moves".
#      variant: the same source reaches both side basins, so 1 input pixel can
#      become 2 settled output pixels. The rest of the gravity / diagonal / row
#      slide logic is unchanged.
# b) Complexity calibration
#    - The solver keeps the original local physics and only replaces a unique
#      target with a reachable terminal-set computation.
#    - The generator now focuses on one mixed template family where a left shelf
#      forces a horizontal split and a right pillar forces a diagonal split in
#      the same 10x10 scene.
#    - Each sample contains exactly two branch events, so the visual difference
#      from the base task is obvious and repeated rather than accidental.
# c) Old implicit assumptions that break under D18
#    - The original input check forbids water with empty cells on both horizontal
#      sides; horizontal splitters require exactly that, so that assert is removed.
#    - The original `settle` returns one coordinate and writes one water cell back.
#      Under D18 that would overwrite sibling branches; `settle` must return all
#      reachable sinks, and the pass loop must write all of them.
#    - The original ambiguity asserts in diagonal and horizontal code paths are
#      replaced by branching transitions.
# d) Placement legality
#    - All solids are bottom-anchored columns within the 10x10 grid.
#    - Source cells are always background cells with solid support directly below.
#    - Templates are chosen so sibling branches land in disjoint basins or in the
#      two-cell center basin; they do not need hidden multiplicity tracking.
# e) Visual observability
#    - Every generated sample increases visible water occupancy:
#      2 source pixels -> 4 settled pixels.
#    - Every generated sample shows both branch kinds in one scene, so a solver
#      that still expects uniqueness would fail on both the shelf tie and the
#      diagonal tie without needing cross-example luck.
# f) Visual cleanliness
#    - Water only occupies open basins/channels; solids remain simple columnar
#      walls. There is no color remapping, overlay marker, or crowded annotation.
# g) Color reasoning
#    - Colors do not encode rules. They only label fixed structures, and every
#      upper structure color also appears on the bottom row so the original guide
#      sanity checks remain harmless.
# 9) New solver assumptions and generator guarantees
#    - The solver merges sibling branches if they ever reconverge to one cell.
#      The generator avoids early reconvergence by construction.
#    - Branching is assumed to be binary because local ambiguity can only be
#      left/right in this task. The generator uses symmetric nearest-drop ties.
#    - Output stability is still checked with the original left / below / right
#      support condition, and templates are built so every final water cell passes.


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate branching water falling into bottom-anchored basins and shelves.

    Input structure:
    - Background is 0 and water is color 1.
    - The other 2-3 colors are fixed solid walls, pillars, and basin floors.
    - Water starts in supported source cells above splitters or basin entrances.

    Rule:
    - Water still falls straight down whenever the cell below is empty.
    - If downward motion is blocked, open diagonal routes are tried next; when
      both diagonals are equally open, the water splits and follows both routes.
    - If a supported water cell must slide along a row toward the nearest drop,
      an equal left/right tie also causes a split into both horizontal branches.
    - Every branch keeps moving by the same local priorities until all branches
      settle, and the process repeats for every source pixel until stable.

    Key insight: Water is no longer conserved per pixel. Ambiguous choices create
    a branching reachability tree, so the output is the union of all settled sinks
    reachable from every supported source pixel.
    """
    return _simulate(grid, check_input=check_input, check_output=check_output)[0]


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _simulate(
    grid: np.ndarray,
    check_input: bool = True,
    check_output: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    stats = {
        "input_water": int(np.count_nonzero(g == int(water))),
        "output_water": 0,
        "water_gain": 0,
        "branch_events": 0,
        "diag_splits": 0,
        "horiz_splits": 0,
        "max_terminals_per_source": 1,
        "passes": 0,
    }

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def next_positions(rr: int, cc: int) -> tuple[str | None, list[tuple[int, int]]]:
        if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
            return "down", [(int(rr + 1), int(cc))]

        diag = []
        if (
            rr + 1 < H
            and cc - 1 >= 0
            and int(g[rr, cc - 1]) == int(bg)
            and int(g[rr + 1, cc - 1]) == int(bg)
        ):
            diag.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < H
            and cc + 1 < W
            and int(g[rr, cc + 1]) == int(bg)
            and int(g[rr + 1, cc + 1]) == int(bg)
        ):
            diag.append((int(rr + 1), int(cc + 1)))
        if diag:
            return "diag", diag

        if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
            L = int(cc)
            while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                L -= 1
            R = int(cc)
            while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                R += 1

            drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
            if not drops:
                return None, []

            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            steps = sorted(
                {
                    (int(rr), int(cc + (-1 if best < cc else 1)))
                    for best in nearest
                    if int(best) != int(cc)
                }
            )
            for _r, nx in steps:
                assert int(g[rr, nx]) == int(bg)
            return "horiz", steps

        return None, []

    def settle(r: int, c: int) -> set[tuple[int, int]]:
        stack = [(int(r), int(c))]
        seen: set[tuple[int, int]] = set()
        terminals: set[tuple[int, int]] = set()
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            kind, nxts = next_positions(int(rr), int(cc))
            if len(nxts) > 1:
                stats["branch_events"] += 1
                if kind == "diag":
                    stats["diag_splits"] += 1
                elif kind == "horiz":
                    stats["horiz_splits"] += 1
            if nxts:
                stack.extend(nxts)
            else:
                terminals.add((int(rr), int(cc)))
        assert terminals, "water branch did not settle"
        stats["max_terminals_per_source"] = max(stats["max_terminals_per_source"], len(terminals))
        return terminals

    for step in range(2 * H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            terminals = settle(int(r), int(c))
            moved = bool(moved or terminals != {(int(r), int(c))})
            for rr, cc in terminals:
                g[int(rr), int(cc)] = int(water)
        if not moved:
            stats["passes"] = int(step + 1)
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(bg)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(bg)

    stats["output_water"] = int(np.count_nonzero(g == int(water)))
    stats["water_gain"] = int(stats["output_water"] - stats["input_water"])
    return g, stats


def _fill_column(grid: np.ndarray, col: int, height: int, color: int) -> None:
    if height <= 0:
        return
    H = grid.shape[0]
    grid[H - int(height) :, int(col)] = int(color)


def _mirror_case(
    grid: np.ndarray,
    sources: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mirrored = np.fliplr(np.array(grid, dtype=int))
    W = mirrored.shape[1]
    mirrored_sources = [(int(r), int(W - 1 - c)) for r, c in sources]
    return mirrored, mirrored_sources


def _build_mixed() -> tuple[np.ndarray, list[tuple[int, int]], dict[str, str]]:
    g = np.zeros((10, 10), dtype=int)
    left_color, center_color, right_color = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)

    source_row = random.choice([2, 3])
    support_height = 10 - int(source_row) - 1
    floor_height = random.choice([1, 2])
    wall_height = int(floor_height + 1)

    for col in (1, 2, 3):
        _fill_column(g, int(col), int(support_height), int(left_color))
    _fill_column(g, 0, int(floor_height), int(left_color))
    _fill_column(g, 4, int(floor_height), int(center_color))
    _fill_column(g, 5, int(wall_height), int(center_color))
    _fill_column(g, 6, int(floor_height), int(center_color))
    _fill_column(g, 7, int(support_height), int(right_color))
    _fill_column(g, 8, int(floor_height), int(right_color))
    _fill_column(g, 9, int(wall_height), int(right_color))

    sources = [(int(source_row), 2), (int(source_row), 7)]
    if random.choice([False, True]):
        g, sources = _mirror_case(g, sources)
    return g, sources, {"family": "mixed"}


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    structure, sources, meta = _build_mixed()

    expected = np.array(structure, dtype=int).copy()
    for r, c in random.sample(sources, len(sources)):
        assert int(expected[int(r), int(c)]) == 0
        expected[int(r), int(c)] = 1
        expected = _simulate(expected, check_input=False, check_output=False)[0]

    inp = np.array(structure, dtype=int).copy()
    for r, c in sources:
        assert int(inp[int(r), int(c)]) == 0
        assert int(inp[int(r) + 1, int(c)]) != 0
        inp[int(r), int(c)] = 1

    solved, stats = _simulate(inp, check_input=True, check_output=True)
    assert np.array_equal(solved, expected), "simultaneous input must match sequential pours"
    assert stats["branch_events"] >= len(sources), "each source should branch at least once"
    assert stats["water_gain"] >= len(sources), "branching must increase visible water"
    assert stats["diag_splits"] > 0 and stats["horiz_splits"] > 0

    info: dict[str, int | str] = dict(meta)
    info.update(stats)
    return inp, solved, info


def generate_input() -> np.ndarray:
    for _ in range(200):
        try:
            inp, _out, _info = _build_case()
            return inp
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid branching-water input")
