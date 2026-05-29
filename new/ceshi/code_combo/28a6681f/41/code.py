# Combo variant of 28a6681f (index 41)
# combo_spec: D1/0+D8/0
# Name: 直线射线路径+多导轨系统

from __future__ import annotations

import random

import numpy as np

# Design analysis:
# spec:
#   preserved_backbone:
#     grid: still fixed at 10x10
#     solids: lower colored structures remain bottom-anchored so the scene still reads as adjacent basins / walls
#     water: color 1 remains the only moving material
#   D1_change_path:
#     old: each water pixel repeatedly searched for a lower resting place by down / diagonal / horizontal gravity rules
#     new: each source is emitted along one straight unit ray and keeps that direction until the next step would hit a
#       solid or leave the grid
#   D8_change_obstacle:
#     old: at most one upper guide existed, mostly just steering one source into one basin
#     new: exactly 2 or 3 upper guides exist, each with its own color, each entering from the left or right border,
#       and each controlling a separate water source on the same line
#   concrete_visualization:
#     guide:
#       every guide is a straight 8-connected chain using a color absent from the bottom row
#       the border endpoint and its next cell reveal one of four allowed ray steps:
#         {right, down-right, left, down-left}
#     source:
#       a contiguous run of blue cells starts immediately after each guide tip
#       all blue cells in the whole input are exactly the union of those source runs
#     target:
#       each ray has a dedicated first blocker column anchored to the bottom
#       after movement, each source becomes a packed water segment immediately before its own blocker
#   behavior_trace:
#     example:
#       guide A: (4,0)-(4,1), source A: (4,2),(4,3), blocker A at column 8
#       guide B: (1,9)-(2,8), source B: (3,7),(4,6), blocker B at (8,2)
#       old_solver_on_same_input:
#         unsupported water would try to fall downward and ignore the visible guide orientations
#       fused_solver:
#         A continues along (0,+1) and packs into (4,6),(4,7)
#         B continues along (+1,-1) and packs into (6,4),(7,3)
#     difference:
#       the important reasoning is now "read each nozzle direction, then trace each ray independently", not "find the
#       deepest basin under gravity"
#   complexity_calibration:
#     why_not_too_simple:
#       samples always contain multiple guides, typically from both borders, and usually mix horizontal and diagonal rays
#       source lengths vary between 2 and 3 so the ray packing is visible
#       bottom scenery remains present, so this does not collapse into bare line drawing
#     why_not_too_free:
#       guide directions come from only four unit steps
#       each guide color is unique and used exactly once
#       rays are generated to stay disjoint, so assignment from source to guide is visually unambiguous
#   implicit_assumptions_review:
#     broken_old_assumptions:
#       original input validation required every water cell to be supported from below; that is invalid now because
#       source water intentionally floats on straight emitter lines
#       original settle() assumed vertical descent is meaningful; every branch in that routine depended on gravity, so
#       the whole movement core must be replaced
#     new_solver_assumptions:
#       every non-bottom color in the upper half is a guide color, not miscellaneous decoration
#       each guide is one straight chain with exactly one left/right border endpoint
#       each guide has a non-empty source run immediately after the tip
#       guide corridors and final ray segments do not interfere with each other
#     generator_guarantees:
#       guide colors never appear on the bottom row
#       all water belongs to guide-adjacent runs and every run has visible travel distance
#       selected rays are pairwise compatible with every blocker column and wall extension
#   placement_legality:
#     guides, sources, travel corridors, and blocker cells are chosen from in-bounds lattice lines
#     bottom structures are painted only outside reserved ray cells, except for the intended blocker cells
#     blocker columns are separated and checked against every other ray path to avoid accidental early collisions
#   observability_and_cleanliness:
#     humans can read each direction directly from the colored guide angle
#     repeated "guide -> blue run -> empty corridor -> collision wall" motifs make the multi-guide rule inferable from
#       a few examples
#     guide colors live in the upper half, basin colors touch the bottom row, and water stays blue, so layers remain
#       visually separated


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _sgn(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _point(start: tuple[int, int], step: tuple[int, int], t: int) -> tuple[int, int]:
    return int(start[0] + step[0] * t), int(start[1] + step[1] * t)


def _trace_line(start: tuple[int, int], step: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    r, c = start
    while _in_bounds(r, c, h, w):
        pts.append((int(r), int(c)))
        r += int(step[0])
        c += int(step[1])
    return pts


def _paint_bottom_structure(
    grid: np.ndarray,
    cols: list[int],
    heights: list[int],
    color: int,
    forbidden: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    for col, height in zip(cols, heights):
        if not (0 <= int(col) < int(w)):
            continue
        hh = max(1, min(int(h), int(height)))
        for row in range(int(h - hh), int(h)):
            cell = (int(row), int(col))
            if cell not in forbidden:
                grid[cell] = int(color)


def _parse_guide(
    grid: np.ndarray,
    color: int,
    bg: int,
    water: int,
) -> tuple[int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
    assert pts, f"guide color {color} must exist"
    assert not any(r == h - 1 for r, _c in pts), "guide color may not touch the bottom row"
    assert any(r < 7 for r, _c in pts), "guide must appear in the upper region"

    ptset = set(pts)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {pt: [] for pt in pts}
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                nxt = (int(r + dr), int(c + dc))
                if nxt in ptset:
                    neighbors[(r, c)].append(nxt)

    endpoints = [pt for pt, adj in neighbors.items() if len(adj) == 1]
    assert len(endpoints) == 2, f"guide {color} must be a simple chain"
    assert all(len(adj) <= 2 for adj in neighbors.values()), f"guide {color} may not branch"

    border_endpoints = [pt for pt in endpoints if pt[1] in (0, w - 1)]
    assert len(border_endpoints) == 1, f"guide {color} must enter from exactly one side border"
    start = border_endpoints[0]
    first = neighbors[start][0]
    step = (_sgn(first[0] - start[0]), _sgn(first[1] - start[1]))
    assert step in {(0, 1), (1, 1), (0, -1), (1, -1)}, f"unsupported guide direction {step}"

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [pt for pt in neighbors[cur] if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, f"guide {color} became ambiguous"
        nxt = nxts[0]
        got = (_sgn(nxt[0] - cur[0]), _sgn(nxt[1] - cur[1]))
        assert got == step, f"guide {color} must stay straight"
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(pts), f"guide {color} must be one straight component"
    return int(color), ordered, ordered[-1], step


def _extract_guides(
    grid: np.ndarray,
    bg: int,
    water: int,
) -> list[tuple[int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]:
    h, _w = grid.shape
    bottom_colors = set(map(int, grid[h - 1]))
    candidate_colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(water)} - bottom_colors)
    guides = [_parse_guide(grid, int(color), bg, water) for color in candidate_colors]
    assert 2 <= len(guides) <= 3, f"expected 2 or 3 guides, found {len(guides)}"
    return sorted(guides, key=lambda item: (item[2][0], item[2][1], item[3][0], item[3][1]))


def _source_after_tip(
    grid: np.ndarray,
    tip: tuple[int, int],
    step: tuple[int, int],
    water: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    r = int(tip[0] + step[0])
    c = int(tip[1] + step[1])
    while _in_bounds(r, c, h, w) and int(grid[r, c]) == int(water):
        out.append((int(r), int(c)))
        r += int(step[0])
        c += int(step[1])
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redirect several blue source runs along guide-defined straight rays until each ray reaches its first collision wall.

    Input structure:
    - bg = 0. Water pixels = color 1.
    - The lower area still contains bottom-anchored colored walls and short basin-like shelves.
    - In the upper half there are 2 or 3 distinct guide colors, each forming one straight chain from the left or right
      border, and each guide has a contiguous blue source run immediately after its tip on the same line.

    Rule:
    - Read each guide from its border endpoint toward its interior tip to obtain one unit ray direction.
    - Remove all source runs, then re-emit every source pixel along its guide direction.
    - A pixel marches only along that straight line through background cells and stops when the next step would leave
      the grid or hit a solid / previously settled water pixel.

    Key insight: The task is no longer about gravity or overflow; the visible upper guides act as multiple nozzles, so
    the solver must infer several ray directions and trace each one to its own first collision point.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    bg = 0
    water = 1

    guides = _extract_guides(g, bg, water)
    emitter_info: list[dict[str, object]] = []
    all_source_cells: set[tuple[int, int]] = set()

    for guide_color, _guide_pts, tip, step in guides:
        source = _source_after_tip(g, tip, step, water)
        assert source, f"guide {guide_color} must have water immediately after its tip"
        info = {
            "color": int(guide_color),
            "tip": tip,
            "step": step,
            "source": source,
        }
        emitter_info.append(info)
        for pt in source:
            assert pt not in all_source_cells, "guide sources must be disjoint"
            all_source_cells.add(pt)

    if check_input:
        water_pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(water))}
        assert water_pts == all_source_cells, "all water must belong to guide-adjacent source runs"

        for info in emitter_info:
            tip = info["tip"]
            step = info["step"]
            source = info["source"]
            assert isinstance(tip, tuple)
            assert isinstance(step, tuple)
            assert isinstance(source, list)

            line_after_tip = _trace_line((int(tip[0] + step[0]), int(tip[1] + step[1])), step, h, w)
            assert line_after_tip[: len(source)] == source, "source must be contiguous on the emitter line"

            probe_r = int(source[-1][0] + step[0])
            probe_c = int(source[-1][1] + step[1])
            moved_probe = False
            while _in_bounds(probe_r, probe_c, h, w) and int(g[probe_r, probe_c]) == int(bg):
                moved_probe = True
                probe_r += int(step[0])
                probe_c += int(step[1])
            assert moved_probe, "each source should have visible travel space after the source run"

    for info in emitter_info:
        source = info["source"]
        assert isinstance(source, list)
        for r, c in source:
            g[int(r), int(c)] = int(bg)

    moved_by_color: dict[int, bool] = {}
    for info in emitter_info:
        guide_color = int(info["color"])
        step = info["step"]
        source = info["source"]
        assert isinstance(step, tuple)
        assert isinstance(source, list)

        moved = False
        for r, c in reversed(source):
            rr = int(r)
            cc = int(c)
            while True:
                nr = int(rr + step[0])
                nc = int(cc + step[1])
                if not _in_bounds(nr, nc, h, w) or int(g[nr, nc]) != int(bg):
                    break
                rr, cc = nr, nc
            moved = bool(moved or ((rr, cc) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)
        moved_by_color[guide_color] = moved

    if check_output:
        final_water = [(int(r), int(c)) for r, c in np.argwhere(g == int(water))]
        assert len(final_water) == len(all_source_cells), "water count must be conserved"

        used_final: set[tuple[int, int]] = set()
        for info in emitter_info:
            guide_color = int(info["color"])
            tip = info["tip"]
            step = info["step"]
            source = info["source"]
            assert isinstance(tip, tuple)
            assert isinstance(step, tuple)
            assert isinstance(source, list)

            line = _trace_line((int(tip[0] + step[0]), int(tip[1] + step[1])), step, h, w)
            pos = {pt: idx for idx, pt in enumerate(line)}
            ray_water = [pt for pt in final_water if pt in pos]
            assert len(ray_water) == len(source), "each ray must keep its own water count"

            idxs = sorted(pos[pt] for pt in ray_water)
            assert idxs == list(range(idxs[0], idxs[-1] + 1)), "output water on each ray must stay contiguous"
            assert idxs[-1] - idxs[0] + 1 == len(source), "ray segment length must match source length"

            last = line[idxs[-1]]
            nr = int(last[0] + step[0])
            nc = int(last[1] + step[1])
            assert not _in_bounds(nr, nc, h, w) or int(g[nr, nc]) != int(bg), "ray must terminate at a collision"
            assert moved_by_color[guide_color], "generated examples should show visible motion on every guide"

            for pt in ray_water:
                assert pt not in used_final, "final ray segments must not overlap"
                used_final.add(pt)

        assert len(used_final) == len(final_water), "all output water must belong to exactly one guide ray"

    return g


def _candidate_emitters(h: int, w: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for side, step in [
        ("left", (0, 1)),
        ("left", (1, 1)),
        ("right", (0, -1)),
        ("right", (1, -1)),
    ]:
        row_choices = range(3, 7) if step[0] == 0 else range(0, 3)
        start_col = 0 if side == "left" else w - 1
        for row in row_choices:
            start = (int(row), int(start_col))
            line = _trace_line(start, step, h, w)
            tmax = len(line) - 1
            for guide_len in (2, 3):
                for source_len in (2, 3):
                    for gap in (1, 2):
                        hit_t = int(guide_len + 2 * source_len + gap)
                        if hit_t > int(tmax):
                            continue
                        guide_pts = line[:guide_len]
                        source_pts = line[guide_len : guide_len + source_len]
                        output_pts = line[hit_t - source_len : hit_t]
                        blocker = line[hit_t]
                        reserved = set(line[: hit_t + 1])
                        out.append(
                            {
                                "side": side,
                                "step": step,
                                "start": start,
                                "guide_len": int(guide_len),
                                "source_len": int(source_len),
                                "gap": int(gap),
                                "hit_t": int(hit_t),
                                "guide_pts": guide_pts,
                                "source_pts": source_pts,
                                "output_pts": output_pts,
                                "blocker": blocker,
                                "reserved": reserved,
                            }
                        )
    return out


def _emitters_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
    reserved_a = a["reserved"]
    reserved_b = b["reserved"]
    blocker_a = a["blocker"]
    blocker_b = b["blocker"]
    assert isinstance(reserved_a, set)
    assert isinstance(reserved_b, set)
    assert isinstance(blocker_a, tuple)
    assert isinstance(blocker_b, tuple)

    if reserved_a & reserved_b:
        return False
    if abs(int(blocker_a[1]) - int(blocker_b[1])) < 2:
        return False

    for r, c in reserved_a:
        if int(c) == int(blocker_b[1]) and int(r) >= int(blocker_b[0]):
            return False
    for r, c in reserved_b:
        if int(c) == int(blocker_a[1]) and int(r) >= int(blocker_a[0]):
            return False
    return True


def _pick_emitters(rng: random.Random, h: int, w: int) -> list[dict[str, object]] | None:
    candidates = _candidate_emitters(h, w)
    rng.shuffle(candidates)

    def search(target_count: int, require_two_sides: bool, require_mixed_slopes: bool) -> list[dict[str, object]] | None:
        chosen: list[dict[str, object]] = []

        def rec(start_idx: int) -> list[dict[str, object]] | None:
            if len(chosen) == target_count:
                sides = {str(item["side"]) for item in chosen}
                slopes = {int(item["step"][0]) for item in chosen}
                if require_two_sides and len(sides) < 2:
                    return None
                if require_mixed_slopes and slopes != {0, 1}:
                    return None
                return [dict(item) for item in chosen]

            for idx in range(start_idx, len(candidates)):
                cand = candidates[idx]
                if any(not _emitters_compatible(prev, cand) for prev in chosen):
                    continue
                chosen.append(cand)
                got = rec(idx + 1)
                if got is not None:
                    return got
                chosen.pop()
            return None

        return rec(0)

    for target_count in [rng.choice([2, 2, 3]), 3, 2]:
        for require_two_sides, require_mixed_slopes in [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]:
            found = search(target_count, require_two_sides, require_mixed_slopes)
            if found is not None:
                return found
    return None


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = w = 10
    bg = 0
    water = 1
    emitters = _pick_emitters(rng, h, w)
    if emitters is None:
        return None

    palette = [2, 3, 4, 5, 6, 7, 8, 9]
    if len(emitters) + 2 > len(palette):
        return None
    colors = rng.sample(palette, len(emitters) + 2)
    guide_colors = colors[: len(emitters)]
    wall_a, wall_b = colors[-2:]

    grid = np.full((h, w), bg, dtype=int)
    reserved_all: set[tuple[int, int]] = set()
    expected = grid.copy()

    for emitter, guide_color in zip(emitters, guide_colors):
        guide_pts = emitter["guide_pts"]
        source_pts = emitter["source_pts"]
        output_pts = emitter["output_pts"]
        reserved = emitter["reserved"]
        assert isinstance(guide_pts, list)
        assert isinstance(source_pts, list)
        assert isinstance(output_pts, list)
        assert isinstance(reserved, set)

        for r, c in guide_pts:
            grid[int(r), int(c)] = int(guide_color)
            expected[int(r), int(c)] = int(guide_color)
        for r, c in source_pts:
            grid[int(r), int(c)] = int(water)
            expected[int(r), int(c)] = int(bg)
        for r, c in output_pts:
            expected[int(r), int(c)] = int(water)
        reserved_all |= reserved

    blockers_sorted = sorted(
        ((idx, emitter["blocker"], emitter["step"]) for idx, emitter in enumerate(emitters)),
        key=lambda item: (item[1][1], item[1][0]),
    )

    for idx, blocker, step in blockers_sorted:
        wall_color = wall_a if idx % 2 == 0 else wall_b
        br, bc = blocker
        for row in range(int(br), h):
            grid[int(row), int(bc)] = int(wall_color)
            expected[int(row), int(bc)] = int(wall_color)

        shelf_len = rng.randint(2, 3)
        if int(step[1]) > 0:
            cols = list(range(int(bc), min(w, int(bc + shelf_len))))
        else:
            cols = list(range(max(0, int(bc - shelf_len + 1)), int(bc + 1)))
        for col in cols:
            grid[h - 1, int(col)] = int(wall_color)
            expected[h - 1, int(col)] = int(wall_color)

    blocker_cols = sorted(int(blocker[1]) for _idx, blocker, _step in blockers_sorted)
    for left, right in zip(blocker_cols, blocker_cols[1:]):
        if int(right - left) <= 2:
            continue
        bridge_color = wall_a if rng.random() < 0.5 else wall_b
        for col in range(int(left + 1), int(right)):
            if (h - 1, int(col)) not in reserved_all:
                grid[h - 1, int(col)] = int(bridge_color)
                expected[h - 1, int(col)] = int(bridge_color)

    free_cols = [c for c in range(w) if all((r, c) not in reserved_all for r in range(h - 3, h))]
    rng.shuffle(free_cols)
    n_extra = 2 if len(free_cols) >= 2 else len(free_cols)
    extra_cols = sorted(free_cols[:n_extra])
    if extra_cols:
        heights = [rng.randint(1, 3) + i for i in range(len(extra_cols))]
        extra_color = wall_b if rng.random() < 0.5 else wall_a
        _paint_bottom_structure(grid, extra_cols, heights, extra_color, reserved_all)
        _paint_bottom_structure(expected, extra_cols, heights, extra_color, reserved_all)

    if any(int(color) in set(map(int, grid[h - 1])) for color in guide_colors):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        case = _build_case(rng)
        if case is not None:
            return case
    raise RuntimeError("generate_input: failed after 800 attempts")
