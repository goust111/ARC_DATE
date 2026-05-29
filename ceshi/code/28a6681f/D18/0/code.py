# Variant of 28a6681f
# Dimension: D18 — 歧义处理策略 (path)
# Change: 严格无歧义设计：对角方向必须恰好一个可行，水平滑动的最近可落点必须唯一——通过assert强制保证... → 确定性消歧规则

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete behavior:
#    The stable lower structure is still a 10x10 basin scene with water=1 and bg=0,
#    but the generator now deliberately creates a small top shelf plus a lower twin-basin
#    shape where one moving water pixel can see multiple legal continuations.
#    In the main template, a 3-row source block sits on a top shelf. As the block drains,
#    one pixel reaches a shelf row with two equally near drop columns, then later reaches
#    a single-cell "nose" with both down-left and down-right open. The old solver asserted
#    on either ambiguity; the new rule resolves both by choosing the leftward option.
# b) Complexity calibration:
#    The generator uses only a few prevalidated source placements and three nearby shelf
#    variants. This keeps the rule learnable while still producing repeated evidence of
#    the hidden priority. Source width is 2 or 3, and mirrored layouts are allowed only
#    where they stay visually clean.
# c) Hidden assumptions audit:
#    The old `settle()` relied on `len(opts)==1` and `len(nearest)==1`. Those lines are
#    exactly what D18 invalidates, so they are replaced by deterministic tie-breakers.
#    Input support checks remain valid because the variant changes path choice, not the
#    definition of supported source water. The output-stability checks also remain valid.
# d) Placement legality:
#    Every template is a fixed 10x10 layout. Water is only placed in top rectangles that
#    were prevalidated to be supported, non-overlapping, and to lead back into the basin.
#    Structure colors are assigned after geometry is fixed, so color changes cannot break
#    physics.
# e) Visual observability:
#    The tie-break depends only on visible left/right geometry. Some samples show both a
#    horizontal tie and a diagonal tie; mirrored samples still keep the same absolute
#    left-priority, so the output is not mirror-symmetric even when the structure nearly
#    is. That makes the new rule observable from a few examples.
# f) Visual cleanliness:
#    The upper guide/shelf uses one color, and the lower basin is split into left/right
#    colors. This keeps the source area, tie area, and resting basins easy to separate.
# g) Color-rule audit:
#    No ordering or decision depends on color ids. Colors only label static structure
#    regions; the disambiguation rule depends purely on visible geometry.
# 9) New assumptions created by the variant:
#    `generate_input()` guarantees at least one ambiguity event under the new solver and
#    keeps the upper guide color off the bottom row, so the inherited guide sanity checks
#    do not become accidental blockers.

BG = 0
WATER = 1

_TEMPLATE_ROWS = [
    [
        "0000000000",
        "0000000000",
        "0000000000",
        "0003330000",
        "0030020000",
        "0000202000",
        "0002202202",
        "0222202220",
        "0222202222",
        "2222222222",
    ],
    [
        "0000000000",
        "0000000000",
        "0000000000",
        "0003330000",
        "0033020000",
        "0000202000",
        "0002202202",
        "0222202220",
        "0222202222",
        "2222222222",
    ],
    [
        "0000000000",
        "0000000000",
        "0000000000",
        "0003330000",
        "0030320000",
        "0000202000",
        "0002202202",
        "0222202220",
        "0222202222",
        "2222222222",
    ],
]


def _rows_to_grid(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


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
    collect_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    metrics = {
        "diag_ties": 0,
        "horiz_ties": 0,
        "left_choices": 0,
        "moved_pixels": 0,
    }

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
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
                if len(opts) > 1:
                    metrics["diag_ties"] += 1
                    metrics["left_choices"] += 1
                rr, cc = min(opts, key=lambda rc: int(rc[1]))
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(BG):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(BG):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                if len(nearest) > 1:
                    metrics["horiz_ties"] += 1
                    metrics["left_choices"] += 1
                best = min(nearest)
                if best == int(cc):
                    break
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
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                moved = True
                metrics["moved_pixels"] += 1
            g[int(rr), int(cc)] = int(WATER)
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

    if collect_metrics:
        return g, metrics
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water with deterministic left-first tie-breaking.

    Input structure:
    - bg = 0 and water = 1.
    - Non-water colors are fixed solids: a small upper shelf/guide and lower basin walls.
    - Water begins in supported source blocks that are allowed to drain into the basin area.

    Rule:
    - Water keeps falling straight down through bg cells whenever possible.
    - When blocked underneath, it may slip diagonally through an open side; if both diagonals
      are open, choose down-left before down-right.
    - When resting on a supported row, it slides toward the nearest drop column; if two nearest
      drops are tied, choose the left one and continue one step at a time.
    - Repeat until every water pixel is trapped against solids, other water, or the grid edge.

    Key insight: This variant keeps the same local liquid physics, but ambiguous path choices are
    no longer forbidden; hidden geometry reveals a consistent left-priority rule.
    """
    result = _simulate(grid, check_input=check_input, check_output=check_output, collect_metrics=False)
    assert isinstance(result, np.ndarray)
    return result


def _colorize_structure(template: np.ndarray, rng: random.Random) -> np.ndarray:
    palette = [2, 3, 4, 5, 6, 7, 8, 9]
    guide_color, left_color, right_color = rng.sample(palette, 3)
    H, W = template.shape
    out = np.array(template, dtype=int).copy()
    split = W // 2
    for r in range(H):
        for c in range(W):
            if int(template[r, c]) == 2:
                out[r, c] = int(left_color if c < split else right_color)
            elif int(template[r, c]) == 3:
                out[r, c] = int(guide_color)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        template = _rows_to_grid(rng.choice(_TEMPLATE_ROWS))
        c0, width = rng.choice([(3, 2), (3, 3)])
        grid = template.copy()
        grid[:3, c0 : c0 + width] = int(WATER)
        grid = _colorize_structure(grid, rng)

        out, metrics = _simulate(grid, check_input=True, check_output=True, collect_metrics=True)
        assert isinstance(out, np.ndarray)
        assert isinstance(metrics, dict)

        if metrics["diag_ties"] + metrics["horiz_ties"] == 0:
            continue
        if metrics["moved_pixels"] < 3:
            continue
        return grid

    raise AssertionError("failed to sample a valid D18 input")
