# Variant of 28a6681f
# Dimension: D7 — 盆地结构形状 (obstacle)
# Change: 2-3个底部锚定的结构：U形、L形、阶梯形或单竖壁，相邻盆地壁高不等且可相互连通允许溢流... → 带有孔洞的盆壁

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete layout and behavior trace
#    The variant keeps the original 10x10 gravity puzzle, but fixes the basin chain
#    to three adjacent bottom-anchored reservoirs:
#    - basin A uses columns 0..2 and is bounded on the right by a tall wall at col 3;
#    - basin B uses columns 4..5 and is bounded on the right by a shorter wall at col 6;
#    - basin C uses columns 7..8 and is bounded on the right by an outer wall at col 9.
#    The key D7 change is that the tall A/B divider is not solid: exactly one cell on
#    that wall is removed, leaving a visible one-pixel side hole. Example profile:
#    wall top at row 3, hole at row 4, lower divider top at row 5.
#    Original behavior on the same geometry without a hole:
#    water can only reach basin B after climbing above the top of the tall divider.
#    Variant behavior:
#    water can slide sideways through the hole at row 4, then fall into basin B even
#    when the final water surface never rises high enough to overtop the divider.
#    After basin B accumulates water, it still reaches basin C only by crossing the
#    top of the shorter solid wall at col 6. This makes side seepage and top overflow
#    both necessary in the same sample.
#
# b) Complexity calibration
#    The generator uses a small family of six validated wall profiles. This is enough
#    variation for the hole height and crest heights to repeat across examples without
#    becoming visually noisy. The right divider never has a hole, so any water in the
#    third basin is an unambiguous top-overflow signal. The first divider always has a
#    hole that sits strictly below its top and strictly above its base, making the new
#    mechanism visible as an actual gap rather than a merely shorter wall.
#
# c) Implicit-assumption audit for main()
#    - Vertical fall only checks for background below. A wall hole is represented as
#      background, so no new code is needed there.
#    - Diagonal motion assumes a unique open side. The generator uses edge-fed source
#      stacks and asymmetric basin widths so ambiguous two-sided slides do not occur.
#    - Horizontal sliding assumes a unique nearest drop along a supported row. The hole
#      works through exactly this branch: once water reaches the hole height, the empty
#      wall cell becomes part of the open run and leads to a unique drop into the next
#      basin. The generator keeps only one active leak wall, so there is no equal-distance
#      competition between multiple drops.
#    - Output stability checks require every final water cell to be bounded left/right
#      (unless on the edge) and below. Therefore the generator only uses wall-height /
#      water-count combinations whose final levels form full supported runs, not ragged
#      partial shelves.
#    - Guide validation in main() is unchanged; this generator simply omits guide pieces.
#
# d) Placement legality
#    All obstacle columns are fixed inside the 10x10 grid: 3, 6, 9. The hole row is
#    always between the top and the floor, so the wall still has solids above and below
#    it. Source water is created by reversing stable output water into a left-packed
#    supported reservoir in basin A; this avoids overlap with solids and preserves the
#    input-side support assertions enforced by main().
#
# e) Visual observability
#    Human-solvable signal:
#    - basin B ends up with water even though the final water surface stays below the
#      top of the tall A/B divider, so basin B cannot have been reached by crest overflow;
#      the visible hole explains the transfer.
#    - basin C also ends up with water, but the B/C divider is solid and shorter, so
#      that transfer must happen by top overflow.
#    The generator explicitly validates both facts before returning a sample.
#
# f) Visual cleanliness
#    Only four visual elements appear: black background, blue water, and three solid
#    colors for the bottom-anchored structures. There are no extra markers, no guides,
#    and no overlapping symbolic layers. The hole is a single empty pixel embedded in
#    an otherwise continuous wall, so it stands out cleanly.
#
# g) Color inference
#    Colors do not encode order or height. The three obstacle colors are chosen randomly
#    and only serve to separate adjacent structures visually. All behavioral logic is
#    driven by visible geometry: wall height, wall continuity, and the presence of the hole.


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling into chained basins, including sideways leaks through wall holes.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors draw bottom-anchored basin walls and floors; one divider may contain a
      one-cell opening, so the wall is visibly interrupted instead of completely solid.
    - Water starts as supported source stacks or runs near the basin system.

    Rule:
    - Each water pixel falls straight down through background cells.
    - If blocked below, it may move diagonally only through an open side cell.
    - When resting on a supported row, it slides along the row toward the unique reachable
      drop, which lets it pass through a side hole or over the top of a lower wall.
    - Repeated global passes continue until no water pixel can move anymore.

    Key insight: The same local gravity-and-sliding rule models two escape routes at once:
    water can spill over a low crest, and it can also leak sideways through a one-pixel wall
    gap once the waterline reaches that opening.
    """
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

                # A hole in a wall is just another background cell in this supported run.
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


_PROFILES = (
    {"w1_top": 2, "hole_row": 4, "w2_top": 5, "water_counts": (22, 24, 26)},
    {"w1_top": 2, "hole_row": 4, "w2_top": 6, "water_counts": (20, 22, 24)},
    {"w1_top": 3, "hole_row": 4, "w2_top": 5, "water_counts": (22, 24, 26)},
    {"w1_top": 3, "hole_row": 5, "w2_top": 6, "water_counts": (17, 19, 21)},
    {"w1_top": 4, "hole_row": 5, "w2_top": 6, "water_counts": (17, 19, 21)},
)
_RIGHT_WALL_TOPS = (3, 4, 5)
_OBSTACLE_COLORS = tuple(range(2, 10))


def _build_terrain(
    profile: dict[str, int | tuple[int, ...]],
    colors: tuple[int, int, int],
    right_wall_top: int,
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    left_color, mid_color, right_color = colors

    for c in range(0, 4):
        g[9, c] = left_color
    for c in range(4, 7):
        g[9, c] = mid_color
    for c in range(7, 10):
        g[9, c] = right_color

    w1_top = int(profile["w1_top"])
    hole_row = int(profile["hole_row"])
    w2_top = int(profile["w2_top"])

    for r in range(w1_top, 9):
        if r != hole_row:
            g[r, 3] = left_color
    for r in range(w2_top, 9):
        g[r, 6] = mid_color
    for r in range(int(right_wall_top), 9):
        g[r, 9] = right_color

    return g


def _pour_output(base: np.ndarray, water_count: int) -> np.ndarray:
    g = base.copy()
    for _ in range(int(water_count)):
        g[0, 0] = 1
        g = main(g, check_input=False, check_output=False)
    return g


def _variant_visible(output: np.ndarray, profile: dict[str, int | tuple[int, ...]]) -> bool:
    hole_row = int(profile["hole_row"])
    top_water_row = int(np.argwhere(output == 1)[:, 0].min())
    has_middle_basin_water = bool(np.any(output[:, 4:6] == 1))
    has_right_basin_water = bool(np.any(output[:, 7:9] == 1))
    hole_remains_open = int(output[hole_row, 3]) == 0
    return bool(has_middle_basin_water and has_right_basin_water and hole_remains_open and top_water_row > hole_row)


def _reverse_to_input(output: np.ndarray, retain_height: int) -> np.ndarray | None:
    inp = output.copy()
    retain = {
        (r, c)
        for r in range(9 - int(retain_height), 9)
        for c in range(0, 3)
        if int(output[r, c]) == 1
    }

    moved = []
    for r, c in np.argwhere(output == 1):
        rc = (int(r), int(c))
        if rc in retain:
            continue
        moved.append(rc)
        inp[rc] = 0

    source_bottom = 8 - int(retain_height)
    source_slots = []
    for r in range(source_bottom, -1, -1):
        for c in range(0, 3):
            if (r, c) not in retain:
                source_slots.append((int(r), int(c)))

    if len(moved) > len(source_slots):
        return None

    for r, c in source_slots[: len(moved)]:
        inp[r, c] = 1

    return inp


def generate_input() -> np.ndarray:
    for _ in range(100):
        profile = random.choice(_PROFILES)
        water_count = int(random.choice(profile["water_counts"]))
        right_wall_top = int(random.choice(_RIGHT_WALL_TOPS))
        colors = tuple(random.sample(_OBSTACLE_COLORS, 3))

        base = _build_terrain(profile, colors, right_wall_top)
        output = _pour_output(base, water_count)

        try:
            stable_output = main(output)
        except AssertionError:
            continue
        if not np.array_equal(stable_output, output):
            continue
        if not _variant_visible(output, profile):
            continue

        input_grid = _reverse_to_input(output, retain_height=random.choice((1, 2)))
        if input_grid is None:
            continue

        if random.random() < 0.5:
            input_grid = np.fliplr(input_grid).copy()
            output = np.fliplr(output).copy()

        try:
            solved = main(input_grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, output):
            continue
        if np.array_equal(input_grid, output):
            continue

        return input_grid

    raise AssertionError("failed to generate a valid leaking-basin example")
