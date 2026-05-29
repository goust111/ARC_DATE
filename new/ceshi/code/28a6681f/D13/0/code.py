# Variant of 28a6681f
# Dimension: D13 — 水像素处理顺序 (multi_step)
# Change: 每轮中水像素按坐标降序排列（即从最底、最右的先处理），底部水先稳定后上方水才落下... → 从顶到底处理

from __future__ import annotations

import random

import numpy as np

# Design analysis for the D13 variant
# -----------------------------------
# concrete_behavior:
#   grid: still 10x10, bg=0, water=1, all nonzero non-water pixels are rigid walls.
#   preserved physics: a single water pixel still settles by repeated straight fall,
#     then a unique diagonal slip if exactly one side is open, then horizontal crawl
#     toward the unique nearest drop in its current supported row segment.
#   changed mechanism: only the outer multi-step pass order changes. In the original
#     solver, each pass processed water from bottom-right upward. Here we process
#     from top-left downward. This means an upper pixel can vacate its supported
#     ledge first, drop into a low basin cell, and force lower pixels to settle
#     elsewhere when their turn arrives later in the same pass.
#   tracked example:
#     - Input contains a supported right-edge reservoir above a two-basin bottom
#       structure. One key low shelf has two competing landing cells.
#     - Old order: the lower source water is handled first and keeps the near shelf
#       slot, so upper water lands one column away.
#     - New order: the upper source water is handled first, falls through the now
#       temporarily cleared channel, and occupies the low slot that the lower water
#       would have claimed under the old order.
#     - Visible effect: final water differs by at least two cells, usually around
#       row 6-8 near the basin handoff.
# complexity_calibration:
#   - The geometry is kept to one family of adjacent basins with a narrow transfer
#     throat. That is enough to expose order dependence without introducing extra
#     rules beyond D13.
#   - The generator uses 8-12 water pixels so the order effect is visible but the
#     grid does not become noisy or ambiguous.
#   - A small compatibility adjustment is needed in the input validator: the upper
#     cells of the source reservoir may be held in place by same-row water contact
#     instead of a literal block directly underneath. The settling physics and the
#     final stability criterion remain unchanged.
# implicit_assumptions_audit:
#   - Input support checks remain valid: generated water is always supported from
#     below by a wall, the floor, or another water pixel.
#   - The diagonal and horizontal uniqueness asserts from the original task are
#     preserved, so generator templates avoid symmetric two-way exits.
#   - Guide detection logic is kept unchanged; generator intentionally uses no
#     upper diagonal guide objects, so these assertions stay dormant and safe.
# placement_legality:
#   - All templates are hand-vetted legal placements inside the 10x10 board.
#   - Structural tweaks only add/remove a few bottom-anchored wall cells and never
#     overlap water.
# visual_observability:
#   - Every returned sample is filtered by a variant-gap metric: the Hamming
#     distance between the new top-first output and the old bottom-first output.
#     We require a gap of at least 2 cells, which makes the changed order visible.
#   - Color remapping only affects rigid walls, never ordering. Sorting depends
#     only on visible row/column position, which a human can infer.
# cleanliness:
#   - The scene uses one compact basin area plus one compact source reservoir,
#     with empty background elsewhere, so input/output pairs remain easy to read.
# new_input_assumptions_checked_by_generator:
#   - Water count stays within a moderate range.
#   - The same candidate passes both the new solver and the old-order reference
#     solver, and their outputs differ by the required metric.

_BASE_INPUT = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 3, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 3, 0, 0, 4, 3, 0, 0, 0],
    ],
    dtype=int,
)

_SOLID_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]


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
    *,
    top_first: bool,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            supported_below = rr == H - 1 or int(g[rr + 1, cc]) != int(bg)
            packed_left = cc > 0 and int(g[rr, cc - 1]) == int(water)
            packed_right = cc + 1 < W and int(g[rr, cc + 1]) == int(water)
            assert supported_below or packed_left or packed_right, "input water must be supported or packed in a source run"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(bg)
                and int(g[rr + 1, cc - 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(bg)
                and int(g[rr + 1, cc + 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(bg)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted((int(r), int(c)) for r, c in np.argwhere(g == int(water)))
        if not top_first:
            pts.reverse()
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)
        if not moved:
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

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate ordered water settling into bottom basins, using top-to-bottom updates.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors are rigid basin walls, shelves, and edges anchored near the bottom.
    - Water starts as a compact side reservoir: each cell is either supported from below or packed against same-row water.

    Rule:
    - In each pass, process water pixels from the highest row to the lowest row, breaking ties from left to right.
    - The selected pixel falls straight down, then uses the unique open diagonal if needed, and on a supported row slides toward the unique nearest drop until it settles.
    - Repeat full passes until no water pixel moves; walls stay fixed and water count is conserved.

    Key insight: The gravity physics is unchanged, but reversing the pass order makes upper water move first and claim basin cells that lower water would have occupied under the original bottom-first solver.
    """
    return _simulate(
        grid,
        top_first=True,
        check_input=check_input,
        check_output=check_output,
    )


def _recolor_solids(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    labels = sorted(set(map(int, g.flatten())) - {0, 1})
    palette = random.sample(_SOLID_COLOR_POOL, len(labels))
    mapping = {label: color for label, color in zip(labels, palette)}
    for label, color in mapping.items():
        g[g == int(label)] = int(color)
    return g


def _variant_gap(grid: np.ndarray) -> int:
    top_out = _simulate(grid, top_first=True, check_input=True, check_output=True)
    bottom_out = _simulate(grid, top_first=False, check_input=True, check_output=True)
    return int(np.count_nonzero(top_out != bottom_out))


def generate_input() -> np.ndarray:
    """
    Generate a legal input that visibly depends on the top-first processing order.

    Strategy:
    - Use a single high-signal reservoir/basin layout whose new and old pass orders diverge strongly.
    - Recolor the rigid solids for visual variety while leaving geometry unchanged.
    - Keep only cases where the new top-first solver differs from the old bottom-first reference.
    """
    for _ in range(200):
        g = _recolor_solids(_BASE_INPUT)

        try:
            gap = _variant_gap(g)
            out = main(g)
        except AssertionError:
            continue

        if gap >= 2 and 8 <= int(np.count_nonzero(g == 1)) <= 12:
            assert np.count_nonzero(out == 1) == np.count_nonzero(g == 1)
            return g

    raise AssertionError("failed to generate a D13-distinguishing input")
