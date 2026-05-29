# Variant of 28a6681f
# Dimension: D6 — 终止条件/稳定判据 (dst)
# Change: 水沉入盆地最低可达位置——输出中每个水像素的左、下、右都为非背景或网格边缘... → 堆叠到指定高度

from __future__ import annotations

import random
import numpy as np


# Design analysis for this D6 variant
# a) Concrete behavior and trace
#    - Keep the original visual vocabulary: 10x10 grid, blue water (1), bottom-anchored colored walls/floors.
#    - Make the source water one supported edge-touching rectangle. Its visible row count is the code.
#    - The destination is the basin strip immediately inside the source shelf. Instead of asking every
#      water pixel to wedge into the globally deepest reachable nook, fill each basin column with exactly
#      as many bottom cells as the source height.
#    - Trace: source height = 2, basin columns have floor tops [5,5,7]. Original deepest-fill prefers
#      to keep pushing one pixel into a lower column before representing the shallower last column. The
#      variant instead assigns 2 cells to each basin column, so the final pattern is three 2-high stacks.
# b) Complexity calibration
#    - Restrict to one source rectangle and one stepped basin family. This keeps the new src→dst height
#      relation legible while preserving the original basin/obstacle setting.
#    - Limit source heights to 2..4 and basin widths to 3..4. These few regimes repeat across examples,
#      which helps human solvers infer the rule from only a handful of pairs.
# c) Old-code dependency audit
#    - The original solver's core assumption was: "if a pixel can no longer move, that resting cell is
#      the answer." That assumption is exactly what D6 changes. Here the answer is no longer "deepest
#      reachable stable cell", but "bottom h cells of each target basin column", where h is the source height.
#    - The original output assertion about left/down/right support encoded the old dst and therefore must
#      be removed. A correct 3-layer stack can have side-exposed top pixels.
#    - The original generator could vary water count independently. Here count must equal basin_width * h
#      so the columnwise h-layer target is exact.
# d) Placement legality
#    - The generator places a source shelf on one side, a same-width basin strip next to it, and then a
#      closing wall. All objects are bottom-anchored or edge-anchored, do not overlap, and stay in-bounds.
#    - Because the source is a full rectangle touching the grid edge, every input water pixel is supported
#      and never has background on both horizontal sides.
# e) Visual observability
#    - Input: source height is directly visible.
#    - Output: every basin column repeats that height, even when floor heights differ. This makes the
#      encoded relation visible without any hidden ordering or color logic.
# f) Visual cleanliness
#    - Use one source rectangle, one basin strip, and 2-3 structure colors. No guides or floating markers
#      compete with the main pattern.
# g) Color inference
#    - Blue is always water, 0 is always background, and 2-3 other colors are inert solids. No behavior
#      depends on numeric color ids.
# 9) New implicit assumptions and how generate_input() guarantees them
#    - Single rectangular source: main() reads one bbox height/width; generator always makes a filled rectangle.
#    - Side-touching source: main() infers whether the basin lies to the right or left of the source; the
#      generator always anchors the source to one side edge.
#    - Same-width basin strip: main() stacks h layers in the next width-many columns; generator always builds
#      the basin with exactly the same width as the source.
#    - Non-degenerate geometry: generator samples only stair templates that were verified to differ from the
#      original deepest-fill behavior and filters again with an in-file classic simulation.


BG = 0
WATER = 1

TOP_TEMPLATES: dict[int, dict[int, list[tuple[int, ...]]]] = {
    2: {
        3: [(5, 5, 7), (5, 6, 7), (5, 5, 8)],
        4: [(5, 5, 5, 7), (5, 5, 7, 7), (5, 5, 7, 9), (5, 6, 7, 8)],
    },
    3: {
        3: [(4, 4, 6), (4, 5, 6), (4, 4, 7), (4, 5, 5)],
        4: [(4, 4, 4, 6), (4, 4, 5, 6), (4, 5, 5, 7), (4, 4, 6, 8)],
    },
    4: {
        3: [(5, 5, 7), (5, 6, 7), (5, 5, 8)],
        4: [(5, 5, 5, 7), (5, 5, 6, 7), (5, 6, 7, 8), (5, 5, 6, 8)],
    },
}


def _component_count(points: set[tuple[int, int]]) -> int:
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _source_box(g: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == int(WATER))
    assert pts.size > 0, "expected a non-empty water source"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _source_info(g: np.ndarray) -> tuple[int, int, int, int, str]:
    H, W = g.shape
    r0, c0, r1, c1 = _source_box(g)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if c0 == 0:
        side = "left"
    elif c1 == W - 1:
        side = "right"
    else:
        raise AssertionError("water source must touch exactly one side edge")
    assert not (c0 == 0 and c1 == W - 1), "source cannot touch both side edges"
    return height, width, r0, c0, side


def _basin_columns(width: int, c0: int, side: str, W: int) -> list[int]:
    if side == "left":
        start = int(c0 + width)
        cols = list(range(start, start + width))
    else:
        stop = int(c0)
        cols = list(range(stop - width, stop))
    assert cols and min(cols) >= 0 and max(cols) < W, "basin strip is out of bounds"
    return cols


def _stack_output(base: np.ndarray, height: int, width: int, c0: int, side: str) -> np.ndarray:
    H, W = base.shape
    basin_cols = _basin_columns(width, c0, side, W)
    out = base.copy()
    for c in basin_cols:
        empties = [int(r) for r in range(H - 1, -1, -1) if int(out[r, c]) == int(BG)]
        assert len(empties) >= int(height), "not enough basin capacity for encoded stack height"
        for r in empties[:height]:
            out[int(r), int(c)] = int(WATER)
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Restack source water into a stepped basin, using source height as the destination stack depth.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors are fixed solid structures. Generated samples use one edge shelf, one same-width basin strip,
      and a closing wall.
    - The source water is a single supported rectangle touching the left or right edge, and its row-count is the code.

    Rule:
    - Remove the source water from its shelf.
    - Look at the basin strip immediately inside the source. In each basin column, fill the deepest available
      background cells with exactly as many water pixels as the source object's height.
    - Structure colors stay unchanged and water count is conserved.

    Key insight: The destination is no longer "where water can sink deepest". The visible source height now directly
    prescribes how many layers must be stacked in each target basin column.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    height, width, r0, c0, side = _source_info(g)
    r1 = int(r0 + height - 1)
    c1 = int(c0 + width - 1)

    if check_input:
        source = {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}
        assert _component_count(source) == 1, "water source must be one 4-connected component"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(WATER)), "water source must be a filled rectangle"
        for r, c in source:
            assert r == H - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"
            left_bg = c > 0 and int(g[r, c - 1]) == int(BG)
            right_bg = c + 1 < W and int(g[r, c + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

        basin_cols = _basin_columns(width, c0, side, W)
        wall_col = int(max(basin_cols) + 1) if side == "left" else int(min(basin_cols) - 1)
        assert 0 <= wall_col < W, "closing wall must exist"
        assert np.all(g[:, wall_col] != int(BG)), "closing wall must span the full height"

    base = g.copy()
    base[g == int(WATER)] = int(BG)
    out = _stack_output(base, height, width, c0, side)

    if check_output:
        assert int(np.sum(g == int(WATER))) == int(np.sum(out == int(WATER))), "water count must be conserved"
        basin_cols = set(_basin_columns(width, c0, side, W))
        for c in range(W):
            col_water = int(np.sum(out[:, c] == int(WATER)))
            if c in basin_cols:
                assert col_water == int(height), "each basin column must receive exactly source-height water"
            else:
                assert col_water == 0, "water must appear only inside the target basin strip"

    return out


def _classic_main(grid: np.ndarray) -> np.ndarray:
    """Original deepest-fill behavior, kept only to filter out degenerate generator samples."""
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

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
                if len(opts) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {opts}")
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(BG):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(BG):
                    R += 1

                drops = [int(x) for x in range(L, R + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                if best == int(cc):
                    raise AssertionError("classic settle found a zero-step horizontal move")
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                if int(g[rr, nx]) != int(BG):
                    raise AssertionError("classic horizontal slide hit a blocked cell")
                cc = nx
                continue

            break
        else:
            raise AssertionError("classic water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
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
        raise AssertionError("classic water simulation did not converge")

    return g


def _build_case() -> np.ndarray:
    H = W = 10
    height = random.randint(2, 4)
    width = random.choice(sorted(TOP_TEMPLATES[height].keys()))
    tops = list(random.choice(TOP_TEMPLATES[height][width]))

    colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    shelf_color, basin_color_a, wall_color = colors

    g = np.zeros((H, W), dtype=int)
    g[height:, :width] = int(shelf_color)

    for idx, c in enumerate(range(width, 2 * width)):
        color = int(basin_color_a if idx % 2 == 0 else wall_color)
        g[tops[idx] :, c] = int(color)

    g[:, 2 * width] = int(colors[2])
    g[:height, :width] = int(WATER)

    if random.choice([False, True]):
        g = np.fliplr(g).copy()

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _build_case()
        try:
            expected = main(inp)
            classic = _classic_main(inp)
        except Exception:
            continue
        if not np.array_equal(expected, classic):
            return inp
    raise RuntimeError("failed to generate a non-degenerate D6 variant instance")
