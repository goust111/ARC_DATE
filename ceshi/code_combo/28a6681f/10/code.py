# Combo variant of 28a6681f (index 10)
# combo_spec: D12/2+D20/0
# Name: 壁面侵蚀+水接触壁变色

from __future__ import annotations

import numpy as np


# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Keep the original 10x10 "adjacent basins" motif, but bias the generator
#      toward two bottom-anchored chambers separated by a one-cell-thick inner wall.
#      One chamber is shallow because it has a shelf; the neighbor is deeper and
#      initially empty.
#    - Small trace:
#      original logic: water stacked in the shallow chamber can only settle on the
#      shelf and against the divider, because the divider never changes.
#      combo logic:
#        1. run the same local water motion until the current configuration is stable;
#        2. scan wall pixels; a wall pixel erodes only when it has water on exactly
#           one horizontal side, background on the other, and that water side has a
#           vertical run of at least two settled water cells at that height or below;
#        3. erased wall pixels turn into background, opening a breach;
#        4. water re-settles through the breach into the deeper basin;
#        5. after no more motion/erosion is possible, every surviving structure pixel
#           that still touches final water orthogonally is recolored to wet color 5.
#      difference: compared with the source task, the solver must reason about both
#      dynamic geometry change and the final contact contour. The same divider can
#      therefore show all three output states: untouched dry wall above the water,
#      wet-but-intact wall near the remaining waterline, and missing cells where the
#      breach was carved.
#
# b) Complexity calibration:
#    - The erosion threshold is fixed to a single visible depth rule: "two-high water
#      stack against the wall". This is strong enough to create a breach but simple
#      enough to infer from a few pairs.
#    - Only one output-only marker color is added (5). Dry structure colors stay in
#      the small set 2/3/4, so the three wall states remain visually distinct:
#      original color, wet gray 5, or erased to 0.
#    - Layout freedom is intentionally limited to a few repeated knobs:
#      source side left/right, divider column 4/5, shelf row 7/8, and divider height
#      2..4. This avoids accidental ambiguity while still giving mirrored examples.
#
# c) Audit of original assumptions:
#    - Original `settle()` assumes every non-background, non-water cell is static.
#      That remains true inside a single sweep, but the outer loop must now continue
#      when walls erode even if no water moved in that same sweep.
#    - Original output validation expected each water cell to have no background on
#      left/below/right. After erosion, locally stable water can legally have exposed
#      sides, so validation changes to the stronger "remove one water cell and re-run
#      settle; it must land back in the same position".
#    - Original code allowed any non-zero structure color in input. The combo adds a
#      new assumption: wet color 5 is output-only. `check_input` asserts that.
#    - Recoloring cannot happen during motion, otherwise wet cells would be mistaken
#      for obstacles with special semantics. Therefore the code must compute wet marks
#      only after the erosion+motion loop has fully converged.
#
# d) Placement legality:
#    - The generator anchors outer walls on the left/right edge and never allows edge
#      columns or the bottom row to erode. This prevents punching holes out of the
#      entire container or deleting the basin floor.
#    - Source water is packed bottom-up against the divider on top of a horizontal
#      shelf, so every input water pixel is supported and never has background on
#      both horizontal sides.
#    - The target basin is deeper than the source shelf, guaranteeing that once a
#      breach opens there is an actual lower destination and the original settle rule
#      can exploit it without creating ambiguous choices.
#
# e) Visual observability:
#    - A human sees two distinct output signals at once: part of the divider has
#      disappeared, and surviving contacted walls have turned color 5.
#    - Because accepted samples require transfer into the neighboring basin, the
#      changed geometry has an obvious behavioral consequence; this prevents the task
#      from looking like a pure recolor puzzle.
#
# f) Visual cleanliness:
#    - The generator keeps all changes concentrated around the two basins. There is
#      no overlay on water itself and no scattered annotation elsewhere.
#    - The shelf stays above the deeper target floor, so the missing-wall region and
#      the wet contour do not visually collapse into a noisy mass.
#
# g) Inferability of the change:
#    - Color 5 is not tied to a specific dry color. Any surviving structure pixel can
#      become wet if and only if final water touches it orthogonally.
#    - Erosion is not keyed by color either; it depends only on a visible geometric
#      cue: a two-cell water stack pressing on one side of a thin wall with empty
#      space on the other.
#
# h) Non-degeneration metric used by the generator:
#    - Each accepted sample must remove at least 2 structure pixels, create at least
#      4 wet pixels, wet at least 1 pixel on the divider column itself, and send at
#      least 2 water cells into the formerly empty target basin.
#    - The accepted combo output must also differ from the "wet-only, no-erosion"
#      baseline on the same input. This rules out samples where the fused solver
#      behaves exactly like the recolor-only variant.
#
# i) New solver assumptions checked against the generator:
#    - Wet color 5 never appears in the input.
#    - The divider is one cell thick and bordered by background on the target side
#      before erosion, so the erosion test has a clear "water side / empty side".
#    - The source water column height is at least 3, ensuring the erosion threshold is
#      actually reachable and yields a visible breach instead of a trivial wet-only
#      case.


BG = 0
WATER = 1
LEFT_COLOR = 2
MID_COLOR = 3
RIGHT_COLOR = 4
WET = 5
GRID_SIZE = 10
EROSION_DEPTH = 2


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
    allow_erosion: bool,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
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
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER, WET} - bottom_colors):
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
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
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

    def water_run_depth(start_r: int, water_c: int) -> int:
        depth = 0
        rr = int(start_r)
        while rr < H and int(g[rr, water_c]) == int(WATER):
            depth += 1
            rr += 1
        return depth

    def erode_cells() -> list[tuple[int, int]]:
        if not allow_erosion:
            return []
        cells: list[tuple[int, int]] = []
        for r in range(H - 1):
            for c in range(1, W - 1):
                v = int(g[r, c])
                if v in (BG, WATER):
                    continue
                left = int(g[r, c - 1])
                right = int(g[r, c + 1])
                if left == int(WATER) and right == int(BG):
                    if water_run_depth(int(r), int(c - 1)) >= EROSION_DEPTH:
                        cells.append((int(r), int(c)))
                elif left == int(BG) and right == int(WATER):
                    if water_run_depth(int(r), int(c + 1)) >= EROSION_DEPTH:
                        cells.append((int(r), int(c)))
        return cells

    for _pass in range(4 * H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(WATER)

        eroded = erode_cells()
        if eroded:
            for r, c in eroded:
                g[int(r), int(c)] = int(BG)

        if not moved and not eroded:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr0 = int(r)
            cc0 = int(c)
            g[rr0, cc0] = int(BG)
            rr1, cc1 = settle(rr0, cc0)
            g[rr0, cc0] = int(WATER)
            assert (int(rr1), int(cc1)) == (rr0, cc0), "output water must already be locally stable"

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


def _solid_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero((grid != int(BG)) & (grid != int(WATER)) & (grid != int(WET))))


def _removed_solid_count(inp: np.ndarray, settled: np.ndarray) -> int:
    solid_in = (inp != int(BG)) & (inp != int(WATER)) & (inp != int(WET))
    solid_out = (settled != int(BG)) & (settled != int(WATER))
    return int(np.count_nonzero(solid_in & ~solid_out))


def _target_water_count(settled: np.ndarray, partition_col: int, source_side: str) -> int:
    if source_side == "L":
        return int(np.count_nonzero(settled[:, partition_col + 1 : 9] == int(WATER)))
    return int(np.count_nonzero(settled[:, 1:partition_col] == int(WATER)))


def _source_water_count(settled: np.ndarray, partition_col: int, source_side: str) -> int:
    if source_side == "L":
        return int(np.count_nonzero(settled[:, 1:partition_col] == int(WATER)))
    return int(np.count_nonzero(settled[:, partition_col + 1 : 9] == int(WATER)))


def _partition_wet_count(out: np.ndarray, partition_col: int) -> int:
    return int(np.count_nonzero(out[:, partition_col] == int(WET)))


def _behavior_metrics(inp: np.ndarray, out: np.ndarray, partition_col: int, source_side: str) -> dict[str, int]:
    settled_combo = np.where(out == int(WET), MID_COLOR, out)
    return {
        "removed": _removed_solid_count(inp, settled_combo),
        "wet": int(np.count_nonzero(out == int(WET))),
        "partition_wet": _partition_wet_count(out, partition_col),
        "target_water": _target_water_count(settled_combo, partition_col, source_side),
        "source_water": _source_water_count(settled_combo, partition_col, source_side),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling in neighboring basins whose thin divider can be breached, then mark the surviving contacted walls as wet.

    Input structure:
    - Background is 0, moving water is 1, and the dry basin/guide pixels use other colors.
    - The important structures are bottom-anchored basins separated by one-cell-thick walls; a shallow source chamber may already hold a supported water pile.
    - Wet marker color 5 is reserved for the output and must not appear in the input.

    Rule:
    - Water keeps the source task's local motion: fall straight down, otherwise use the unique open diagonal, otherwise slide horizontally toward the nearest reachable drop.
    - After each full settling sweep, a wall pixel erodes to background if it has water on exactly one horizontal side, background on the other, and the water side forms a vertical run of at least two settled water cells at that height or below.
    - Repeat settling and erosion until nothing changes, then recolor every surviving structure pixel that shares an edge with final water to wet color 5.

    Key insight:
    - The output is determined by a state transition on the walls, not just on the water: shallow piled water first creates enough pressure to carve an opening, and only the walls that survive that process but still touch the final water are recolored as wet.
    """
    settled = _simulate(
        grid,
        allow_erosion=True,
        check_input=check_input,
        check_output=check_output,
    )
    out = _mark_wet_walls(settled)

    if check_output:
        dry_structure = (settled != int(BG)) & (settled != int(WATER))
        for r, c in np.argwhere(out == int(WET)):
            rr = int(r)
            cc = int(c)
            assert bool(dry_structure[rr, cc]), "wet pixels must come from surviving structure pixels"
            assert any(
                0 <= rr + dr < out.shape[0]
                and 0 <= cc + dc < out.shape[1]
                and int(out[rr + dr, cc + dc]) == int(WATER)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ), "wet pixels must touch final water"

    return out


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int | str]]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    partition_col = int(rng.choice([4, 5]))
    source_side = str(rng.choice(["L", "R"]))
    shelf_row = int(rng.choice([7, 8]))
    partition_top = int(rng.integers(2, min(5, shelf_row - 1)))
    max_water_top = shelf_row - 3
    water_top = int(rng.integers(partition_top, max_water_top + 1))

    if source_side == "L":
        source_color = LEFT_COLOR
        target_color = RIGHT_COLOR
        source_cols = list(range(1, partition_col))
        source_outer_col = 0
        target_outer_col = 9
        source_fill_order = list(range(partition_col - 1, 0, -1))
    else:
        source_color = RIGHT_COLOR
        target_color = LEFT_COLOR
        source_cols = list(range(partition_col + 1, 9))
        source_outer_col = 9
        target_outer_col = 0
        source_fill_order = list(range(partition_col + 1, 9))

    source_wall_top = int(rng.integers(1, water_top + 1))
    target_wall_top = int(rng.integers(2, 7))

    g[9, :partition_col] = int(LEFT_COLOR)
    g[9, partition_col] = int(MID_COLOR)
    g[9, partition_col + 1 :] = int(RIGHT_COLOR)
    g[partition_top:9, partition_col] = int(MID_COLOR)
    g[source_wall_top:10, source_outer_col] = int(source_color)
    g[target_wall_top:10, target_outer_col] = int(target_color)
    g[shelf_row, source_cols] = int(source_color)

    rows_touching_wall = shelf_row - water_top
    top_count = int(rng.integers(1, len(source_cols) + 1))
    water_count = rows_touching_wall * len(source_cols) - (len(source_cols) - top_count)

    remaining = int(water_count)
    for r in range(shelf_row - 1, -1, -1):
        if remaining <= 0:
            break
        take = min(len(source_cols), remaining)
        for c in source_fill_order[:take]:
            g[int(r), int(c)] = int(WATER)
        remaining -= take

    assert remaining == 0
    return g, {
        "partition_col": partition_col,
        "source_side": source_side,
        "shelf_row": shelf_row,
    }


def _attempt_case(rng: np.random.Generator) -> np.ndarray | None:
    inp, meta = _build_case(rng)
    partition_col = int(meta["partition_col"])
    source_side = str(meta["source_side"])

    try:
        settled_combo = _simulate(inp, allow_erosion=True, check_input=True, check_output=True)
        out_combo = _mark_wet_walls(settled_combo)
        settled_static = _simulate(inp, allow_erosion=False, check_input=True, check_output=True)
        out_static = _mark_wet_walls(settled_static)
    except Exception:
        return None

    removed = _removed_solid_count(inp, settled_combo)
    wet_count = int(np.count_nonzero(out_combo == int(WET)))
    partition_wet = _partition_wet_count(out_combo, partition_col)
    transferred = _target_water_count(settled_combo, partition_col, source_side)
    source_remaining = _source_water_count(settled_combo, partition_col, source_side)

    if removed < 2:
        return None
    if wet_count < 4:
        return None
    if partition_wet < 1:
        return None
    if transferred < 2:
        return None
    if source_remaining < 1:
        return None
    if np.array_equal(settled_combo, settled_static):
        return None
    if np.array_equal(out_combo, out_static):
        return None
    if _solid_count(settled_combo) >= _solid_count(inp):
        return None

    return inp


def generate_input(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    for _ in range(400):
        result = _attempt_case(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed to construct a valid combo case")
