# Variant of 28a6681f
# Dimension: D6 — 盆地壁结构 (obstacle)
# Change: 2-3个底部锚定的盆地/沟槽结构，形状多样（U形、L形、阶梯形、单垂直壁），使用2-3种非水颜色，相邻盆地互相连通允许溢... → 悬浮障碍物
#
# Design analysis
# spec:
#   concretization:
#     grid stays 10x10. Water is still color 1 and all other non-zero colors are rigid walls.
#     The lower half keeps a connected bottom basin system, but at least one upper wall component
#     is no longer connected to the bottom row. The generator always places:
#       - a bottom-anchored basin network with a left wall, a low L-shaped divider, and a taller divider;
#       - a split floating shelf above the main basin, with a one-cell leak gap between the two halves;
#       - a second floating branch hanging from the right shelf, so the upper obstacle looks like a
#         shelf/funnel rather than a simple platform.
#     Source water starts as a supported reservoir on top of the left floating shelf. During solving,
#     it must slide toward the leak gap, fall through that suspended opening, collect in the left basin,
#     and often overflow into the neighboring basin below the hanging branch.
#   behavior_trace:
#     Small canonical trace in the unmirrored layout:
#       1. Input water occupies rows 0-2 above columns 0-3, fully supported by the floating shelf.
#       2. The rightmost source water slides right to column 4 because the shelf has a unique one-cell gap there.
#       3. It falls through the gap, lands on top of the low divider, then diagonally slips left into the basin.
#       4. As the left basin fills, later water reaches the divider height and spills rightward into the
#          neighboring pocket, now visibly sitting under the suspended branch at rows 4-5.
#     Original task behavior mostly reasoned about bottom-anchored cups; here the decisive event is the
#     mid-air leak gap. Removing the floating cells changes the output sharply or even makes the motion ambiguous.
#   complexity_calibration:
#     The layout uses one stable family of suspended shapes with only mirror/color/count variation.
#     That keeps examples legible while making the floating mechanism obvious in 2-4 samples.
#     The main branch point is only "which side is mirrored", not many unrelated subrules.
#   implicit_assumptions_review:
#     Original code had a validation block that treated non-bottom colors in the upper rows as optional
#     edge-entering guides. That relies on the old invariant "upper solids are guides, not basin walls".
#     In this variant upper solids can be arbitrary floating walls, so that check must be removed.
#     The gravity, diagonal slip, and horizontal search logic itself already works for suspended walls.
#   placement_legality:
#     The generator uses a fixed safe coordinate template, then mirrors it horizontally. Every placed
#     obstacle stays in-bounds, suspended components are disconnected from the bottom by at least one row,
#     and the water source is always supported with no cell having background on both horizontal sides.
#   visual_observability:
#     The leak gap is always visible as a one-cell opening between two colored shelf pieces high above
#     the basin. The output visibly moves water from the upper shelf region into lower basins, often with
#     overflow into a second compartment. This makes the suspended-wall rule inferable without hidden metadata.
#   cleanliness:
#     Water, floating walls, and bottom walls occupy separated vertical bands. The scene is sparse enough
#     that the eye can track the route from source shelf to leak gap to basin.
#   color_reasoning:
#     Colors only mark rigid structure pieces. No rule depends on color ordering or numeric ids.

from __future__ import annotations

import random

import numpy as np


def _simulate(
    grid: np.ndarray,
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
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

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
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(bg):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(bg):
                    right += 1

                drops = [x for x in range(int(left), int(right) + 1) if int(g[rr + 1, x]) == int(bg)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
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
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(water):
                continue
            g[r, c] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(water)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg)

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling through suspended shelves and bottom basins.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are rigid walls: some remain bottom-anchored, while at least one upper component is
      detached from the floor and forms a floating shelf, leak gap, or hanging branch.
    - Water begins as a supported reservoir perched against a shelf or wall rather than already resting in the basin.

    Rule:
    - A water pixel falls vertically through empty cells.
    - If blocked below, it may slip diagonally downward only through a unique open side channel.
    - When supported, it crawls horizontally across a shelf or ledge toward the nearest unique drop column.
    - Repeating this local motion lets water leak through suspended gaps, pass beneath overhangs, and then
      continue filling and overflowing the lower basins.

    Key insight: The transformation is still local gravity simulation, but the decisive routing now comes
    from mid-air obstacle geometry, so suspended gaps and overhangs control where the water can descend.
    """
    return _simulate(grid, check_input=check_input, check_output=check_output)


def _paint(g: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    H, W = g.shape
    for r, c in cells:
        assert 0 <= int(r) < H and 0 <= int(c) < W
        assert int(g[int(r), int(c)]) == 0
    for r, c in cells:
        g[int(r), int(c)] = int(color)


def _component_count(mask: np.ndarray) -> int:
    H, W = mask.shape
    seen: set[tuple[int, int]] = set()
    count = 0
    for r, c in np.argwhere(mask):
        root = (int(r), int(c))
        if root in seen:
            continue
        count += 1
        stack = [root]
        seen.add(root)
        while stack:
            rr, cc = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return int(count)


def _floating_mask(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    solid = (g != 0) & (g != 1)
    H, W = solid.shape
    anchored = np.zeros_like(solid, dtype=bool)
    stack = [(int(H - 1), int(c)) for c in range(W) if bool(solid[int(H - 1), int(c)])]
    for item in stack:
        anchored[item] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and bool(solid[nr, nc]) and not bool(anchored[nr, nc]):
                anchored[nr, nc] = True
                stack.append((nr, nc))
    return solid & (~anchored)


def _variant_score(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int]:
    floating = _floating_mask(inp)
    float_cells = int(np.count_nonzero(floating))
    float_components = _component_count(floating)
    if float_cells == 0:
        return 0, 0, 0
    stripped = np.array(inp, dtype=int).copy()
    stripped[floating] = 0
    try:
        stripped_out = _simulate(stripped, check_input=False, check_output=False)
        diff = int(np.count_nonzero(stripped_out != out))
    except AssertionError:
        diff = int(inp.size)
    return diff, float_cells, float_components


def _build_candidate(rng: random.Random) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    palette = rng.sample(list(range(2, 10)), k=rng.choice((2, 3)))
    c_left = int(palette[0])
    c_mid = int(palette[1 % len(palette)])
    c_right = int(palette[2 % len(palette)])

    _paint(g, [(9, c) for c in range(0, 4)], c_left)
    _paint(g, [(9, c) for c in range(4, 7)], c_mid)
    _paint(g, [(9, c) for c in range(7, 9)], c_right)
    _paint(g, [(r, 0) for r in range(6, 9)], c_left)
    _paint(g, [(7, 4), (7, 5), (8, 4), (8, 5)], c_mid)
    _paint(g, [(r, 7) for r in range(6, 9)], c_right)

    _paint(g, [(3, c) for c in range(0, 3)], c_mid)
    _paint(g, [(3, c) for c in range(4, 7)] + [(4, 5), (5, 4), (5, 5)], c_right)

    source_cells = [(r, c) for r in range(0, 3) for c in range(0, 3)]
    source_cells += [(8, 1), (8, 2)]
    source_cells += rng.choice(
        (
            [],
            [(8, 3)],
            [(7, 1)],
            [(7, 1), (7, 2)],
        )
    )

    for r, c in source_cells:
        assert int(g[r, c]) == 0
        g[r, c] = 1

    if rng.random() < 0.5:
        g = np.fliplr(g).copy()

    return g


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inp = _build_candidate(rng)
    try:
        out = main(inp)
    except AssertionError:
        return None

    diff, float_cells, float_components = _variant_score(inp, out)
    if float_cells < 6 or float_components < 2:
        return None
    if diff < 5:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
