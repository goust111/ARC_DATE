# Variant of 28a6681f
# Dimension: D20 — 输出中结构的保留/变换 (transform)
# Change: 所有盆壁结构在输出中完全保留不变（invariant: structure pixels unchanged）... → 水接触的壁面变色
#
# Reference-task note:
# - The workspace does not contain a standalone original solver file for 271d71e2.
# - For the required recolor baseline, this variant was calibrated against the
#   available task summary/examples for 271d71e2 and the local 271d71e2 D5 code,
#   which keeps the same "geometry unchanged, only recolor a derived subset"
#   design pattern.
#
# Design analysis for this variant
# a) Concrete behavior and trace:
#    - The original 28a6681f behavior is kept intact up to the moment when all
#      blue water pixels have settled inside the bottom basins.
#    - New output-only rule: every non-background, non-water structure pixel that
#      shares a 4-neighbor edge with final settled water is recolored to wet gray 5.
#      Water itself does not change color; only contacted structure pixels do.
#    - Small trace:
#      original logic: a 2-cell-wide basin fills with water at rows 7..9, while
#      the side walls and basin floor keep their old colors.
#      new logic: the same water cells appear, but the wall pixels at (7..9,left),
#      (7..9,right), and the floor-support pixels directly under the water all
#      change to 5. The physical trajectory is unchanged; the visible transform is
#      now "simulate first, then recolor contacted walls".
#
# b) Complexity calibration:
#    - The change is intentionally narrow: only D20 changes. Water motion,
#      conservation, basin geometry, and source placement stay in the same family
#      as the source task.
#    - To keep the new rule visible from a few examples, the generator uses only
#      three dry structure colors (2/3/4) plus one output-only wet color (5).
#      This prevents the recolor from getting lost inside too many unrelated hues.
#    - Samples are rejected unless the output contains multiple wet pixels. This
#      avoids accidental degeneration into the original invariant-structure task.
#
# c) Review of original assumptions:
#    - The original `settle()` and outer simulation loop assume that every
#      non-zero, non-water pixel is an immutable solid obstacle during motion.
#      That assumption still holds while water is moving, so the water-physics code
#      can stay unchanged.
#    - The original final write `g[rr, cc] = water` was safe because structure
#      pixels never changed. That is still safe here because recoloring happens only
#      after convergence; no recolored wall is ever treated as empty space.
#    - The original `check_output` only reasoned about stable water cells. That
#      remains valid. The new code adds a second validation layer: every wet pixel
#      must come from an original structure pixel and must touch final water.
#    - A new input-side assumption appears: wet color 5 is output-only. The
#      generator guarantees that input structures never use 5, and `check_input`
#      asserts this explicitly.
#
# d) Placement legality:
#    - The generator uses only safe, hand-screened 10x10 basin templates with
#      three bottom-anchored wall columns, giving exactly two adjacent basins.
#    - Source water sits on a supported upper shelf entering from the left or
#      right edge. Water cells are filled bottom-up from the edge inward, so every
#      input water cell is supported and never has background on both horizontal
#      sides.
#    - Layouts were filtered to avoid the original solver's ambiguous diagonal or
#      equidistant horizontal-drop asserts.
#
# e) Visual observability:
#    - The only new color in the output is wet gray 5, and it appears exactly
#      where water ends up touching the basins. Across examples, a human can infer
#      "blue water settles as before; touched walls turn gray".
#    - Overflow cases are common, so some outputs wet only one basin while others
#      wet both neighboring basins. This makes the wall-contact dependency legible.
#
# f) Visual cleanliness:
#    - Wet pixels are always a one-cell-thick contour on existing structures.
#      There is no extra overlay object, no counting marker, and no competing
#      annotation in the same basin cells as the water.
#    - The upper source shelf is kept well above the final water zone, so it does
#      not accidentally get recolored and confuse the "wet basin walls" signal.
#
# g) Color-change inferability:
#    - The recolor is not keyed by color id or by basin order. Any dry wall color
#      can become 5 if and only if final water touches it orthogonally.
#    - Because the rule depends on visible adjacency rather than hidden ordering,
#      it is recoverable from 2-4 ARC training pairs.
#
# h) Non-degeneration metric:
#    - The core D20 behavior is quantified by wet-wall count: each accepted sample
#      must produce at least 6 wet pixels in the output.
#    - A second metric is wet-basin spread: many generated samples wet both basins,
#      confirming that recoloring is tied to the simulated fill/overflow pattern.
#
# i) New solver assumptions checked against the generator:
#    - `main()` assumes wet color 5 is absent from the input.
#    - `main()` assumes top-edge source supports obey the original input support
#      rules.
#    - `generate_input()` enforces both assumptions by construction and validates
#      each candidate by calling the same water-settling routine used by `main()`.

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1
WET = 5
WALL_COLORS = (2, 3)
GUIDE_COLOR = 4

# Safe templates pre-screened against the original water solver's ambiguity asserts.
# Each entry is (wall_columns, wall_top_rows). Three bottom-anchored walls create
# exactly two adjacent basins with width 2 each.
SAFE_LAYOUTS: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((1, 4, 7), (4, 6, 4)),
    ((1, 4, 7), (4, 7, 4)),
    ((1, 4, 7), (5, 7, 5)),
    ((1, 4, 7), (4, 6, 5)),
    ((1, 4, 7), (5, 6, 4)),
    ((1, 4, 7), (4, 7, 5)),
    ((2, 5, 8), (4, 6, 4)),
    ((2, 5, 8), (4, 7, 4)),
    ((2, 5, 8), (5, 7, 5)),
    ((2, 5, 8), (4, 6, 5)),
    ((2, 5, 8), (5, 6, 4)),
    ((2, 5, 8), (4, 7, 5)),
)


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


def _settle_water(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        assert not np.any(g == int(WET)), "wet color is output-only"
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
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
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

    return g


def _mark_wet_walls(settled: np.ndarray) -> np.ndarray:
    out = np.array(settled, dtype=int).copy()
    H, W = out.shape
    wet_positions: set[tuple[int, int]] = set()
    for r, c in np.argwhere(out == int(WATER)):
        rr = int(r)
        cc = int(c)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < H and 0 <= nc < W and int(out[nr, nc]) not in (int(BG), int(WATER)):
                wet_positions.add((nr, nc))
    for r, c in wet_positions:
        out[int(r), int(c)] = int(WET)
    return out


def _basin_water_counts(grid: np.ndarray, walls: tuple[int, int, int]) -> list[int]:
    return [int(np.sum(grid[:, left + 1 : right] == int(WATER))) for left, right in zip(walls, walls[1:])]


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water exactly as in the source task, then recolor the walls that end up in direct contact with the settled water.

    Input structure:
    - Background is 0 and moving water is 1.
    - Other colors form static solid basin walls and optional upper support guides.
    - Water starts in supported source positions above or beside the basin chain, just as in the original task.

    Rule:
    - First settle every water pixel using the original gravity, diagonal slip, and supported horizontal-slide rules.
    - After the water stops moving, scan the final water cells and find every solid structure pixel that shares an edge with water.
    - Recolor exactly those contacted structure pixels to wet gray 5.
    - Water count is conserved; only the contacted wall/floor pixels change color.

    Key insight:
    - The hard part is still the original local water simulation, but the output is no longer a pure "water-only" change. The solver must derive a second object set: the final wall contour touched by the settled water.
    """
    settled = _settle_water(grid, check_input=check_input, check_output=check_output)
    out = _mark_wet_walls(settled)

    if check_output:
        dry_structure = (settled != int(BG)) & (settled != int(WATER))
        for r, c in np.argwhere(out == int(WET)):
            rr = int(r)
            cc = int(c)
            assert bool(dry_structure[rr, cc]), "wet pixels must come from original structure pixels"
            assert any(
                0 <= rr + dr < out.shape[0]
                and 0 <= cc + dc < out.shape[1]
                and int(out[rr + dr, cc + dc]) == int(WATER)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ), "wet pixels must touch water in the final output"

    return out


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()

    for _ in range(300):
        walls, tops = rng.choice(SAFE_LAYOUTS)
        first_basin_capacity = int((walls[1] - walls[0] - 1) * (10 - tops[1]))

        side_options: list[tuple[str, int, bool]] = []
        for side in ("left", "right"):
            shelf_width = walls[0] + 2 if side == "left" else 10 - (walls[-1] - 1)
            shelf_width = max(3, min(5, int(shelf_width)))
            max_source_capacity = int(3 * shelf_width)
            side_options.append((side, shelf_width, bool(max_source_capacity > first_basin_capacity)))

        want_overflow = any(can for _side, _width, can in side_options) and rng.random() < 0.55
        if want_overflow:
            capable = [item for item in side_options if item[2]]
            side, shelf_width, _can = max(capable, key=lambda item: item[1])
            shelf_row = 3
            water_count = rng.randint(max(4, first_basin_capacity + 1), shelf_row * shelf_width)
        else:
            side, shelf_width, _can = rng.choice(side_options)
            shelf_row = rng.choice([2, 3])
            high = min(first_basin_capacity, shelf_row * shelf_width)
            if high < 4:
                high = shelf_row * shelf_width
            water_count = rng.randint(4, high)

        g = np.zeros((10, 10), dtype=int)
        for idx, (col, top) in enumerate(zip(walls, tops)):
            g[int(top) :, int(col)] = int(WALL_COLORS[idx % len(WALL_COLORS)])

        if side == "left":
            g[shelf_row, :shelf_width] = int(GUIDE_COLOR)
            source_cells = [(r, c) for r in range(shelf_row - 1, -1, -1) for c in range(shelf_width)]
        else:
            g[shelf_row, 10 - shelf_width :] = int(GUIDE_COLOR)
            source_cells = [
                (r, c)
                for r in range(shelf_row - 1, -1, -1)
                for c in range(9, 10 - shelf_width - 1, -1)
            ]
        for r, c in source_cells[:water_count]:
            g[int(r), int(c)] = int(WATER)

        try:
            settled = _settle_water(g, check_input=True, check_output=True)
            out = _mark_wet_walls(settled)
        except Exception:
            continue

        if np.any(settled[: shelf_row + 1] == int(WATER)):
            continue

        wet_count = int(np.sum(out == int(WET)))
        if wet_count < 6:
            continue

        basin_counts = _basin_water_counts(settled, walls)
        touched_basins = sum(count > 0 for count in basin_counts)
        if touched_basins == 0:
            continue
        if want_overflow and touched_basins < 2:
            continue

        return g

    raise RuntimeError("failed to generate a valid D20 sample")
