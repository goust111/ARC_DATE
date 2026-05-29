# Combo variant of 28a6681f (index 18)
# combo_spec: D6/0+D20/1
# Name: 移除未使用结构+水位线限制

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete form and behavior trace
#    - The scene keeps the source-task idiom: a 10x10 grid with blue water (1), bottom-anchored
#      solid walls, and a supported source shelf near one side.
#    - The new visible control is a shared waterline marker in color 8 placed on the two outer
#      basin walls. Those marker cells are ordinary wall cells for support/deletion purposes.
#    - Water still follows the original local motion. The D6 part changes only the resting rule:
#      a moved water pixel may not finish above the marker row, so excess water remains on the
#      source shelf instead of stacking deeper into the basin network.
#    - The D20 part then runs on that limited-fill result: keep only bottom-anchored structure
#      pixels that are 4-neighbor adjacent to final water; erase every other bottom basin pixel.
#    - Small trace:
#         outer walls at columns 2 and 7, divider at column 4, marker row = 6, source on the right.
#         Original task: water fills rows 8, 7, 6, then continues into row 5 and wets more of the
#         upper wall. All wall pixels remain visible.
#         Combo task: once a pixel's best new resting place would be row 5, that move is rejected
#         and the pixel stays on the source shelf. Afterwards, wall cells on rows 2-5 that never
#         touch water are deleted, so the visible basin is truncated to the wetted portion.
#
# b) Complexity calibration
#    - Only one extra symbol is introduced: marker color 8. It always means the waterline and is
#      reused across every sample.
#    - The basin family is deliberately narrow: two adjacent cups with an optional overflow divider.
#      That is enough to expose both parts of the fusion without introducing many unrelated degrees
#      of freedom.
#    - Generator variation comes from source side, divider height, waterline row, source volume,
#      wall tops, and color remapping. The geometry is not so broad that human pattern matching
#      becomes ambiguous.
#
# c) Hidden-assumption review against the source solver
#    - Source settle() assumed every supported background cell was a legal terminal resting place.
#      That fails now: supported cells above the waterline are illegal as new destinations.
#    - Source output checks assumed every water pixel ended with left/down/right blocked. That also
#      fails now because rejected water can stay on the source shelf with air on one side.
#    - D20 adds a post-simulation deletion pass, so returning the settled grid directly is no
#      longer correct. Structure removal must happen after the limited simulation has converged.
#    - The deletion pass must inspect only bottom-anchored structure. The upper source shelf stays
#      untouched; otherwise rejected source water could lose its support.
#    - Combining both changes introduces one more assumption: the marker pixels must belong to the
#      bottom-anchored basin structure, not to a floating guide. The generator guarantees that by
#      placing them on the outer basin walls.
#
# d) Placement legality
#    - The basin uses fixed columns 2 / 4 / 7, so all candidates stay in-bounds and preserve the
#      same non-ambiguous spill geometry as the D6 reference.
#    - The shelf occupies columns 0-2 before optional mirroring. Water is inserted only in cells
#      directly above the shelf, so every source pixel is supported.
#    - The waterline row is always strictly below the source shelf and strictly above the floor.
#      Wall-top choices are always above or equal to that row, so the marker cells are valid wall
#      cells and there are dry wall segments available to delete.
#
# e) Visual observability
#    - In the input, the marker row is explicit and attached to the basin walls. In the output,
#      water visibly stops at that height while some source water remains stranded on the shelf.
#    - The deleted pixels are not arbitrary recolorings: they are precisely the dry wall segments,
#      most visibly the portions above the filled waterline and sometimes an untouched neighboring
#      basin. A human can infer that the waterline controls both fill depth and what survives.
#
# f) Visual cleanliness
#    - Functional layers are limited to background, water, ordinary walls, and marker walls.
#    - The marker is embedded in the wall rather than floating in open space, which keeps the grid
#      readable. Only bottom structure is eligible for deletion, so the upper source scaffold does
#      not fragment across outputs.
#
# g) Change inferability
#    - Color 8 is the only semantic color. Other wall colors are sampled from a small palette but
#      do not encode rules.
#    - The key relation is geometric and repeated across samples: water stops at the shared marked
#      row, and dry wall cells disappear afterwards. No hidden ordering by color id is used.
#
# h) Generator validation / non-degeneration metrics
#    - Each accepted sample must satisfy all of:
#         1. limited-fill output differs from the input,
#         2. limited-fill output differs from the unlimited-fill output,
#         3. the limited-fill output leaves more water on the source shelf than the unlimited one,
#         4. at least two above-waterline basin pixels are deleted, and
#         5. at least two structure pixels are deleted only because of the waterline limit.
#      Metric (5) is the key fusion check: it proves the deletion pattern is not just the plain D20
#      variant applied to an unchanged water simulation.
#
# i) New implicit assumptions and how generate_input() guarantees them
#    - Assumption: all markers share one row. Guaranteed by placing exactly two marker cells on the
#      two outer walls and optionally mirroring the whole scene.
#    - Assumption: rejected water can safely remain at its start cell. Guaranteed by building the
#      source as a compact shelf-supported block on the grid edge.
#    - Assumption: diagonal and horizontal choices stay unique. Guaranteed by the fixed 2/4/7 basin
#      frame and the hand-vetted divider-height templates inherited from the D6 family.


BG = 0
WATER = 1
MARKER = 8
H = 10
W = 10
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = ORTHO_DIRS

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


def _marker_row(grid: np.ndarray) -> int | None:
    rows = {int(r) for r, _c in np.argwhere(grid == int(MARKER))}
    if not rows:
        return None
    if len(rows) != 1:
        raise AssertionError("waterline markers must share a single row")
    return next(iter(rows))


def _bottom_anchored_structure(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    solid = (g != int(BG)) & (g != int(WATER))
    stack = [(H - 1, int(c)) for c in range(W) if bool(solid[H - 1, c])]
    anchored: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in anchored:
            continue
        anchored.add((int(r), int(c)))
        for dr, dc in ORTHO_DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(solid[rr, cc]) and (rr, cc) not in anchored:
                stack.append((rr, cc))
    return anchored


def _check_input_constraints(grid: np.ndarray, limit_row: int | None) -> None:
    g = np.array(grid, dtype=int)
    assert g.shape == (H, W), "grid must be 10x10"
    assert limit_row is not None, "input must contain a waterline marker"
    assert 0 < int(limit_row) < H - 1, "waterline must be inside the grid"

    anchored = _bottom_anchored_structure(g)
    markers = [(int(r), int(c)) for r, c in np.argwhere(g == int(MARKER))]
    assert len(markers) >= 2, "waterline should be visible on at least two wall cells"
    for r, c in markers:
        assert (r, c) in anchored, "waterline markers must be part of the bottom basin"
        adjacent_bg = (
            (r > 0 and int(g[r - 1, c]) == int(BG))
            or (r + 1 < H and int(g[r + 1, c]) == int(BG))
            or (c > 0 and int(g[r, c - 1]) == int(BG))
            or (c + 1 < W and int(g[r, c + 1]) == int(BG))
        )
        assert adjacent_bg, "marker should touch open space"

    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1])) - {int(BG), int(WATER)}
    top_colors = set(map(int, g[:7].flatten())) - {int(BG), int(WATER), int(MARKER)} - bottom_colors
    for color in sorted(top_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _simulate_water(grid: np.ndarray, limit_row: int | None, check_input: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    if check_input:
        _check_input_constraints(g, limit_row)

    def settle(r: int, c: int) -> tuple[int, int]:
        start = (int(r), int(c))
        rr, cc = start
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

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, x]) == int(BG)]
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

        if limit_row is not None and rr < int(limit_row) and (rr, cc) != start:
            return start
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _delete_unused_basin_pixels(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    anchored = _bottom_anchored_structure(g)
    used: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        for dr, dc in ORTHO_DIRS:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) in anchored:
                used.add((nr, nc))

    out = g.copy()
    for r, c in anchored:
        if (int(r), int(c)) not in used:
            out[int(r), int(c)] = int(BG)
    return out


def _check_settled_water_fixed_point(grid: np.ndarray, limit_row: int | None) -> None:
    replay = _simulate_water(grid, limit_row=limit_row, check_input=False)
    assert np.array_equal(replay, grid), "limited water state is not stable"


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water that fills chained basins only up to a visible line, then erase the dry basin parts.

    Input structure:
    - Background is 0, movable water is 1, and color 8 marks a shared waterline on bottom-anchored basin walls.
    - Other nonzero colors are fixed solids that form two neighboring bottom basins plus a supported source shelf near one side.
    - Water starts on that shelf or in already-settled cells, while the marker cells are part of the wall itself.

    Rule:
    - Move water with the source task physics: fall vertically through background, slip through a unique open diagonal, and while supported slide horizontally toward the nearest reachable drop.
    - A moved water pixel may not finish above the marker row. If its best new resting place would cross that height, it stays in its original supported source cell instead.
    - After the limited fill stabilizes, inspect only bottom-anchored structure pixels and delete every one that is not 4-neighbor adjacent to final water.

    Key insight:
    - The visible waterline does not merely cap the water depth; it also changes which wall pixels ever become wet, so the surviving structure can be inferred only after simulating the limited fill.
    """
    g = np.array(grid, dtype=int).copy()
    limit_row = _marker_row(g)
    if check_input:
        _check_input_constraints(g, limit_row)
    settled = _simulate_water(g, limit_row=limit_row, check_input=False)
    out = _delete_unused_basin_pixels(settled)
    if check_output:
        _check_settled_water_fixed_point(settled, limit_row=limit_row)
        assert np.array_equal(_delete_unused_basin_pixels(settled), out), "dry-structure deletion mismatch"
    return out


def _source_box(side: str, floor_row: int, height: int) -> tuple[range, range]:
    rows = range(floor_row - height, floor_row)
    if side == "left":
        return rows, range(0, 3)
    return rows, range(7, 10)


def _count_water(grid: np.ndarray, rows: range, cols: range) -> int:
    return sum(int(grid[r, c] == int(WATER)) for r in rows for c in cols)


def _build_candidate(
    rng: random.Random, want_overflow: bool
) -> tuple[np.ndarray, dict[str, int | str]]:
    g = np.zeros((H, W), dtype=int)
    wall_colors = rng.sample([2, 3, 4, 5, 6, 7, 9], 3)
    left_color, divider_color, right_color = wall_colors

    left_wall = 2
    divider = 4
    right_wall = 7
    source_side = rng.choice(["left", "right"])
    if want_overflow:
        template = rng.choice(
            [
                {"shelf_row": 3, "source_height": 3, "level_row": 7, "divider_top": 8},
                {"shelf_row": 4, "source_height": 3, "level_row": 7, "divider_top": 8},
                {"shelf_row": 4, "source_height": 4, "level_row": 6, "divider_top": 7},
                {"shelf_row": 4, "source_height": 4, "level_row": 7, "divider_top": 8},
            ]
        )
    else:
        template = rng.choice(
            [
                {"shelf_row": 3, "source_height": 2, "level_row": 5, "divider_top": 3},
                {"shelf_row": 3, "source_height": 2, "level_row": 6, "divider_top": 4},
                {"shelf_row": 4, "source_height": 3, "level_row": 6, "divider_top": 4},
                {"shelf_row": 4, "source_height": 4, "level_row": 7, "divider_top": 6},
            ]
        )

    shelf_row = int(template["shelf_row"])
    source_height = int(template["source_height"])
    level_row = int(template["level_row"])
    divider_top = int(template["divider_top"])
    left_top = rng.choice([2, 3, 4])
    right_top = rng.choice([2, 3, 4])

    g[H - 1, left_wall : divider + 1] = int(left_color)
    g[H - 1, divider : right_wall + 1] = int(right_color)
    g[left_top:H, left_wall] = int(left_color)
    g[divider_top:H, divider] = int(divider_color)
    g[right_top:H, right_wall] = int(right_color)

    g[level_row, left_wall] = int(MARKER)
    g[level_row, right_wall] = int(MARKER)

    g[shelf_row, 0:3] = int(divider_color)
    g[shelf_row - source_height : shelf_row, 0:3] = int(WATER)

    if source_side == "right":
        g = np.fliplr(g)

    meta: dict[str, int | str] = {
        "source_side": source_side,
        "shelf_row": shelf_row,
        "source_height": source_height,
        "level_row": level_row,
        "want_overflow": int(want_overflow),
    }
    return g, meta


def _variant_metrics(inp: np.ndarray, meta: dict[str, int | str]) -> dict[str, int]:
    limit_row = _marker_row(inp)
    assert limit_row is not None

    settled_limited = _simulate_water(inp, limit_row=limit_row, check_input=True)
    settled_unlimited = _simulate_water(inp, limit_row=None, check_input=False)
    final_limited = _delete_unused_basin_pixels(settled_limited)
    final_unlimited = _delete_unused_basin_pixels(settled_unlimited)

    rows, cols = _source_box(
        str(meta["source_side"]),
        int(meta["shelf_row"]),
        int(meta["source_height"]),
    )
    limited_source = _count_water(final_limited, rows, cols)
    unlimited_source = _count_water(final_unlimited, rows, cols)

    anchored = _bottom_anchored_structure(inp)
    deleted = 0
    deleted_above_line = 0
    line_forced_deleted = 0
    for r, c in anchored:
        if int(final_limited[r, c]) == int(BG):
            deleted += 1
            if r < int(limit_row):
                deleted_above_line += 1
            if int(final_unlimited[r, c]) != int(BG):
                line_forced_deleted += 1

    marker_contacts = 0
    for r, c in np.argwhere(inp == int(MARKER)):
        rr = int(r)
        cc = int(c)
        if any(
            0 <= rr + dr < H
            and 0 <= cc + dc < W
            and int(final_limited[rr + dr, cc + dc]) == int(WATER)
            for dr, dc in ORTHO_DIRS
        ):
            marker_contacts += 1

    left_basin_water = int(np.any(final_limited[5:, 3:5] == int(WATER)))
    right_basin_water = int(np.any(final_limited[5:, 5:7] == int(WATER)))

    return {
        "limited_source": int(limited_source),
        "unlimited_source": int(unlimited_source),
        "deleted": int(deleted),
        "deleted_above_line": int(deleted_above_line),
        "line_forced_deleted": int(line_forced_deleted),
        "marker_contacts": int(marker_contacts),
        "left_basin_water": int(left_basin_water),
        "right_basin_water": int(right_basin_water),
        "limited_vs_unlimited_diff": int(np.count_nonzero(final_limited != final_unlimited)),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        # Overflow candidates are harder to satisfy the stronger fusion metrics, so proposal
        # sampling is biased toward them to keep overflow as a common accepted outcome.
        want_overflow = bool(rng.random() < 0.8)
        inp, meta = _build_candidate(rng, want_overflow=want_overflow)
        try:
            out = main(inp, check_input=True, check_output=True)
            metrics = _variant_metrics(inp, meta)
        except AssertionError:
            continue

        if np.array_equal(out, inp):
            continue
        if metrics["limited_vs_unlimited_diff"] <= 0:
            continue
        if metrics["limited_source"] <= metrics["unlimited_source"]:
            continue
        if metrics["deleted"] < 4:
            continue
        if metrics["deleted_above_line"] < 2:
            continue
        if metrics["line_forced_deleted"] < 2:
            continue
        if metrics["marker_contacts"] < (2 if want_overflow else 1):
            continue

        has_overflow = bool(metrics["left_basin_water"] and metrics["right_basin_water"])
        if has_overflow != want_overflow:
            continue

        return np.array(inp, dtype=int)

    raise RuntimeError("failed to generate a non-degenerate combo sample")
