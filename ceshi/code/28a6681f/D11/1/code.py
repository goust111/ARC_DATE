# Variant of 28a6681f
# Dimension: D11 — 输出网格形状 (grid_shape)
# Change: 输出与输入尺寸相同（10x10），结构像素不变... → 仅保留水的终态

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D11 variant
# a) Concrete behavior and flow impact:
#    - Input stays close to the original task: a 10x10 world with bg=0, water=1, and
#      2-3 other colors used as solid terrain/shelves. Water starts in a supported
#      edge reservoir resting on a top shelf, then drains into bottom basins.
#    - The physical simulation itself is unchanged: pixels fall, slide diagonally
#      through a single open side, and slide horizontally to a unique drop.
#    - The change happens after settling. Original task returned the whole 10x10
#      scene with structures preserved. This variant deletes every solid and crops
#      the output to the tight bounding box of the final settled water mask.
#    - Example trace:
#      original: input shelf-water -> simulate -> full 10x10 terrain + stable water.
#      variant:  input shelf-water -> same simulation -> erase all solids -> crop to
#      the water bounding box -> output contains only 0/1.
# b) Complexity calibration:
#    - The generator keeps two adjacent basins separated by a 2-cell plateau so the
#      cropped result is usually a non-rectangular or multi-lobed silhouette, not a
#      trivial filled rectangle.
#    - Source water is a 2-3 column edge reservoir with height 3-5, which supplies
#      enough mass to often reach both basins and make the crop visually meaningful.
# c) Implicit-assumption review of the original solver:
#    - All movement code assumes solids remain in the working grid. That is still
#      true internally, so the simulation helper is left intact.
#    - The old return value assumed "structure pixels unchanged" and relied on the
#      caller seeing the same 10x10 canvas. That assumption no longer holds, so the
#      public main() wraps the original full-grid simulation and replaces the return
#      stage with a crop of the final water mask.
#    - The old output check asserted support in the full scene. That still matters
#      for the internal stable state, so the helper keeps that assertion before the
#      crop. The cropped output then gets its own tight-bbox sanity checks.
# d) Placement legality:
#    - Terrain is generated as bottom-anchored column profiles inside a fixed
#      10x10 grid, so solids never overlap illegally or leave the grid.
#    - The top shelf is placed above the basin profile and uses a color already
#      present on the bottom row, which avoids triggering the original guide-only
#      assertions for unsupported upper decorations.
#    - Source water is a solid rectangle attached to the left or right edge; every
#      water pixel is supported by water below or by the shelf below.
# e) Visual observability:
#    - Across examples, the user can see the same physical pouring input but a very
#      different output format: the container disappears and only the final water
#      silhouette remains, tightly cropped.
#    - Requiring accepted samples to occupy both basins makes the crop visibly
#      depend on hidden structure, rather than looking like a simple falling block.
# f) Visual cleanliness:
#    - Output uses only bg and water. The crop is tight, so there is no huge empty
#      margin. Internal bg gaps are allowed and desirable because they reveal basin
#      separation after structure removal.
# g) Color inference:
#    - No new color mapping is introduced. Colors only distinguish solid obstacles
#      in the input; the output always reduces to {0,1}, so the variant's novelty is
#      purely geometric and therefore directly observable.


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
SOLID_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]


def _simulate_full(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = BG
    water = WATER

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
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

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert component_count(pts, False) == component_count(pts, True), "guide must be 4-connected"

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
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
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


def _crop_water_bbox(full_grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(full_grid == int(WATER))
    assert len(pts) > 0, "expected at least one water pixel after simulation"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(BG), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(WATER)
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate the original water physics, then discard the container and crop to the settled water.

    Input structure:
    - bg = 0 and water = 1.
    - 2-3 other colors form solid shelves, walls, and bottom-anchored basin terrain.
    - Water starts in a supported edge reservoir or on a shelf, not yet at its final resting place.

    Rule:
    - First run the same local water simulation as the base task: vertical fall, unique diagonal escape,
      then supported horizontal sliding toward the nearest reachable drop.
    - After the water stabilizes, erase every non-water structure pixel.
    - Return only the tight bounding box around the final water mask, with bg elsewhere inside that crop.

    Key insight: The target is no longer the full scene; you must infer the hidden settled water silhouette and
    reconstruct it without any of the basin walls that originally constrained it.
    """
    full = _simulate_full(grid, check_input=check_input, check_output=check_output)
    out = _crop_water_bbox(full)

    if check_output:
        vals = set(map(int, np.unique(out)))
        assert vals <= {int(BG), int(WATER)}
        assert (out == int(WATER)).any()
        assert (out[0] == int(WATER)).any()
        assert (out[-1] == int(WATER)).any()
        assert (out[:, 0] == int(WATER)).any()
        assert (out[:, -1] == int(WATER)).any()
        assert int(np.count_nonzero(out == int(WATER))) == int(np.count_nonzero(full == int(WATER)))

    return out


def _build_profile() -> list[int | None]:
    outer = random.randint(4, 5)
    separator = random.randint(outer + 1, 7)
    left_base = random.randint(separator + 1, 8)
    right_base = random.randint(separator + 1, 8)

    left_cols = [left_base, max(separator + 1, left_base - random.randint(0, 1))]
    right_cols = [right_base, max(separator + 1, right_base - random.randint(0, 1))]
    if random.random() < 0.5:
        left_cols.reverse()
    if random.random() < 0.5:
        right_cols.reverse()

    return [None, outer, left_cols[0], left_cols[1], separator, separator, right_cols[0], right_cols[1], outer, None]


def _paint_profile(grid: np.ndarray, profile: list[int | None], palette: list[int]) -> None:
    for c, top in enumerate(profile):
        if top is None:
            continue
        color = int(palette[c % len(palette)])
        grid[int(top) :, int(c)] = color


def _add_source(grid: np.ndarray, palette: list[int], side: str, profile: list[int | None]) -> None:
    shelf_color = int(palette[0])
    outer_top = int(profile[1] if side == "left" else profile[8])
    shelf_row = random.randint(max(2, outer_top - 2), outer_top - 1)
    height = random.randint(3, min(5, shelf_row + 1))
    top = int(shelf_row - height + 1)

    if side == "left":
        drop_col = random.choice([2, 3])
        grid[shelf_row + 1, 0:drop_col] = shelf_color
        grid[top : shelf_row + 1, 0:drop_col] = int(WATER)
    else:
        drop_col = random.choice([6, 7])
        grid[shelf_row + 1, drop_col + 1 :] = shelf_color
        grid[top : shelf_row + 1, drop_col + 1 :] = int(WATER)


def _basin_counts(full_grid: np.ndarray) -> tuple[int, int]:
    left = int(np.count_nonzero(full_grid[:, 2:4] == int(WATER)))
    right = int(np.count_nonzero(full_grid[:, 6:8] == int(WATER)))
    return left, right


def _interesting_sample(inp: np.ndarray, full: np.ndarray, out: np.ndarray) -> bool:
    input_water = inp == int(WATER)
    final_water = full == int(WATER)
    if np.array_equal(input_water, final_water):
        return False

    if out.shape == inp.shape:
        return False

    final_pts = np.argwhere(final_water)
    if len(final_pts) == 0 or int(final_pts[:, 0].min()) < 4:
        return False

    left_count, right_count = _basin_counts(full)
    if left_count == 0 or right_count == 0:
        return False

    row_sums = tuple(int(x) for x in np.sum(out == int(WATER), axis=1))
    col_sums = tuple(int(x) for x in np.sum(out == int(WATER), axis=0))
    if len(set(row_sums)) == 1 and len(set(col_sums)) == 1:
        return False

    if out.shape[0] < 2 or out.shape[1] < 4:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    palette = random.sample(SOLID_COLORS, k=random.choice([2, 3]))
    profile = _build_profile()
    _paint_profile(grid, profile, palette)
    _add_source(grid, palette, random.choice(["left", "right"]), profile)

    try:
        full = _simulate_full(grid, check_input=True, check_output=True)
        out = _crop_water_bbox(full)
    except Exception:
        return None

    if not _interesting_sample(grid, full, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 28a6681f D11 input")
