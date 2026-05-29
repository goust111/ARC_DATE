# Combo variant of cb2d8a2c (index 4)
# combo_spec: D2/1+D16/1
# Name: 蓝色位置编码转弯方向+吸引型障碍条
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    - Canonical frame: the green start sits on the top edge and the route travels downward.
#      Obstacle bars are floating horizontal 1-cell segments, so both left and right bypasses
#      are physically available; otherwise D2/1 would be invisible.
#    - Blue count still encodes the per-bar distance radius `r`.
#    - Repel bars use an alternating blue motif `B R B R ... B` near one end. The path must
#      stay strictly farther than `r` from that bar, so its detour uses the column at exact
#      lateral offset `r+1`.
#    - Attract bars use a contiguous blue block `B B ... B` near one end. The path still may
#      not step on the bar itself, but now it must pass within distance `r`, so its detour
#      hugs the bar at exact lateral offset `r`.
#    - The end that carries the blue motif decides the turning side:
#      motif near the left end -> turn left first; motif near the right end -> turn right first.
#    - Example in canonical orientation with `r=3`:
#      original task on a bar spanning cols 5..11 would route outside the radius using a side
#      column 4 cells away from the touched endpoint;
#      fused repel version still does that but the blue motif chooses left vs right explicitly;
#      fused attract version instead hugs the bar at 3 cells from the endpoint, producing a
#      visibly tighter bulge on the motif-selected side.
# b) Complexity calibration:
#    - Bars use only two visual blue layouts: alternating = repel, contiguous = attract.
#      This keeps the obstacle taxonomy small and observable.
#    - Each generated sample contains both obstacle types and both turn directions, so humans
#      see repeated evidence for both subrules within a single grid.
#    - Blue counts are restricted to 2-3. This avoids the D16 type test becoming ambiguous
#      (`1` blue would look contiguous and alternating at the same time) and keeps the visual
#      differences large enough to read.
# c) Hidden-assumption audit against the original solver:
#    - Original `blocked = bar_mask | forbidden` assumed every bar repels. That fails because
#      attract bars should invite a close pass instead of forbidding a halo.
#    - Original Dijkstra also assumed bar placement alone breaks left/right symmetry. Floating
#      bars remove that assumption, so the turn side must be read from the blue motif and then
#      applied explicitly.
#    - Original bar decoding only needed blue count. The fused variant additionally assumes that
#      every bar is a single horizontal/vertical segment whose blue cells are either contiguous
#      or strictly alternating. The generator guarantees exactly those two layouts.
# d) Placement legality:
#    - Every bar is floating and leaves enough space on both sides for the larger repel detour
#      (`r+1`), so the decoded turn direction is always a genuine choice.
#    - Main-axis spacing is based on the actual detour depth per bar: `r+1` for repel, `1` for
#      attract. This keeps detour rectangles disjoint and avoids overlaps.
#    - Each bar covers the reference lane chosen by the start column, so every bar actually
#      triggers a visible detour instead of becoming irrelevant background decoration.
# e) Visual observability:
#    - The contiguous-vs-alternating blue layouts are visible without using hidden metadata.
#    - Left- and right-end motifs produce opposite-handed bends, and attract bars are visibly
#      tighter than repel bars of the same blue count.
#    - No human-invisible ordering by color id or component index is used.
# f) Visual cleanliness:
#    - Bars remain parallel 1-cell segments with ample spacing, and the path is monotone in the
#      main direction with one rectangular detour per bar. That keeps the output legible.
# g) Non-degeneracy checks:
#    - The generator rejects candidates unless the fused output differs from a same-side
#      all-repel baseline (proving the attract/repel distinction matters) and from a fixed-left
#      baseline (proving the blue-position direction code matters).

import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_CHOICES = (4, 3)
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 2, 3
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 1200

_COMBOS = [
    ("repel", "left"),
    ("repel", "right"),
    ("attract", "left"),
    ("attract", "right"),
]


def _start_edge(grid: np.ndarray) -> tuple[str, int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return "top", r, c
    if r == h - 1:
        return "bottom", r, c
    if c == 0:
        return "left", r, c
    if c == w - 1:
        return "right", r, c
    raise AssertionError("green start must lie on a border")


def _rotate_to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, 1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, -1).copy()


def _rotate_from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "right":
        return np.rot90(grid, -1).copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    return np.rot90(grid, 1).copy()


def _segment_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r0 == r1 or c0 == c1, "segments must be axis-aligned"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _append_segment(
    path: list[tuple[int, int]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> None:
    seg = _segment_coords(r0, c0, r1, c1)
    if path:
        assert path[-1] == seg[0], "segments must connect head-to-tail"
        path.extend(seg[1:])
    else:
        path.extend(seg)


def _decoded_bar_infos(grid: np.ndarray) -> list[dict[str, object]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    infos: list[dict[str, object]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert r0 == r1, "canonical bars must be single-row segments"

        blue_cols = sorted(int(c) for r, c in comp if grid[int(r), int(c)] == BLUE)
        blue_count = len(blue_cols)
        assert BLUE_PER_BAR_LO <= blue_count <= BLUE_PER_BAR_HI, "blue-count range violated"
        diffs = [b - a for a, b in zip(blue_cols, blue_cols[1:])]
        if all(d == 1 for d in diffs):
            kind = "attract"
        elif all(d == 2 for d in diffs):
            kind = "repel"
        else:
            raise AssertionError("blue layout must be contiguous or alternating")

        side = "left" if (sum(blue_cols) / blue_count) < ((c0 + c1) / 2.0) else "right"
        cells = [(r0, c) for c in range(c0, c1 + 1)]
        infos.append(
            {
                "row": r0,
                "c0": c0,
                "c1": c1,
                "blue": blue_count,
                "kind": kind,
                "side": side,
                "cells": cells,
                "cell_set": set(cells),
            }
        )
    infos.sort(key=lambda info: int(info["row"]))
    return infos


def _effective_bar(info: dict[str, object], force_kind: str | None, force_side: str | None) -> dict[str, object]:
    eff = dict(info)
    eff["kind"] = force_kind if force_kind is not None else info["kind"]
    eff["side"] = force_side if force_side is not None else info["side"]
    return eff


def _detour_depth(info: dict[str, object]) -> int:
    return int(info["blue"]) + 1 if info["kind"] == "repel" else 1


def _side_col(info: dict[str, object]) -> int:
    blue = int(info["blue"])
    if info["kind"] == "repel":
        offset = blue + 1
    else:
        offset = blue
    if info["side"] == "left":
        return int(info["c0"]) - offset
    return int(info["c1"]) + offset


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _min_path_dist(path: list[tuple[int, int]], cells: list[tuple[int, int]]) -> int:
    best = 10**9
    for pr, pc in path:
        for br, bc in cells:
            d = abs(pr - br) + abs(pc - bc)
            if d < best:
                best = d
    return best


def _validate_path(
    path: list[tuple[int, int]],
    bars: list[dict[str, object]],
    shape: tuple[int, int],
    start: tuple[int, int],
) -> list[int]:
    h, w = shape
    assert path[0] == start, "path must start at the green seed"
    assert path[-1][0] == h - 1, "canonical path must end on the bottom edge"

    bar_cells = set()
    for info in bars:
        bar_cells.update(info["cell_set"])

    for i, (r, c) in enumerate(path):
        assert 0 <= r < h and 0 <= c < w, "path stepped out of bounds"
        assert (r, c) not in bar_cells, "path may not step on obstacle cells"
        if i:
            pr, pc = path[i - 1]
            assert abs(pr - r) + abs(pc - c) == 1, "path must stay 4-connected"

    min_dists: list[int] = []
    for info in bars:
        dmin = _min_path_dist(path, info["cells"])
        min_dists.append(dmin)
        if info["kind"] == "repel":
            assert dmin > int(info["blue"]), "repel bar was approached too closely"
        else:
            assert 1 <= dmin <= int(info["blue"]), "attract bar was not passed within radius"
    return min_dists


def _solve_canonical_details(
    grid: np.ndarray,
    *,
    force_kind: str | None = None,
    force_side: str | None = None,
) -> dict[str, object]:
    g = np.array(grid, copy=True)
    h, w = g.shape
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1
    sr, sc = map(int, start_pts[0])
    assert sr == 0, "canonical solver expects the start on the top edge"

    bars = [_effective_bar(info, force_kind, force_side) for info in _decoded_bar_infos(g)]
    lane_col = sc
    cur_r = sr
    path: list[tuple[int, int]] = [(sr, sc)]
    used_sides: list[str] = []

    for info in bars:
        blue = int(info["blue"])
        row = int(info["row"])
        side_col = _side_col(info)
        clearance = _detour_depth(info)
        top_row = row - clearance
        bottom_row = row + clearance

        assert int(info["c0"]) <= lane_col <= int(info["c1"]), "generator must place every bar across the lane"
        assert 0 <= side_col < w, "decoded detour side must stay in bounds"
        assert cur_r <= top_row < bottom_row < h, "bars must be vertically ordered with enough spacing"

        _append_segment(path, cur_r, lane_col, top_row, lane_col)
        _append_segment(path, top_row, lane_col, top_row, side_col)
        _append_segment(path, top_row, side_col, bottom_row, side_col)
        _append_segment(path, bottom_row, side_col, bottom_row, lane_col)

        cur_r = bottom_row
        used_sides.append(str(info["side"]))

    _append_segment(path, cur_r, lane_col, h - 1, lane_col)
    min_dists = _validate_path(path, bars, g.shape, (sr, sc))

    return {
        "grid": g,
        "bars": bars,
        "path": path,
        "used_sides": used_sides,
        "min_dists": min_dists,
        "output": _paint_path(g, path),
    }


def _green_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero((a == GREEN) != (b == GREEN)))


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    edge, _, _ = _start_edge(inp)
    canonical = _rotate_to_canonical(inp, edge)
    fused = _solve_canonical_details(canonical)
    all_repel = _solve_canonical_details(canonical, force_kind="repel")
    all_left = _solve_canonical_details(canonical, force_side="left")

    attract_count = sum(1 for info in fused["bars"] if info["kind"] == "attract")
    repel_count = sum(1 for info in fused["bars"] if info["kind"] == "repel")
    left_count = sum(1 for side in fused["used_sides"] if side == "left")
    right_count = sum(1 for side in fused["used_sides"] if side == "right")

    return {
        "attract_count": attract_count,
        "repel_count": repel_count,
        "left_count": left_count,
        "right_count": right_count,
        "type_diff": _green_diff(fused["output"], all_repel["output"]),
        "side_diff": _green_diff(fused["output"], all_left["output"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green border seed to the opposite edge while decoding two different blue-marked obstacle behaviors at once.

    Input structure:
    - The background is azure (8), with one green (3) start pixel on a border edge.
    - Obstacles are thin red/blue bars (2/1) perpendicular to the route direction.
    - Every bar contains 2-3 blue pixels: their count gives a distance radius, their layout type distinguishes attract vs repel, and their end-position chooses the turn side.

    Rule:
    - Rotate to a canonical top-start view, decode each bar, and process the bars from top to bottom.
    - An alternating blue motif means a repel bar: the path must pass strictly farther than the encoded radius. A contiguous blue block means an attract bar: the path must pass within that radius while still staying off the bar cells.
    - The blue motif's position on the bar determines whether the detour first moves left or right, and after the whole path is drawn all blue cells recolor to red.

    Key insight: Blue pixels now carry three simultaneous meanings: how far the path must relate to the bar, which side it should detour on, and whether the bar repels or attracts at all.
    """
    g = np.asarray(grid, dtype=int)
    edge, _, _ = _start_edge(g)
    canonical = _rotate_to_canonical(g, edge)
    details = _solve_canonical_details(canonical)
    return _rotate_from_canonical(details["output"], edge)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _motif_len(kind: str, blue: int) -> int:
    return 2 * blue - 1 if kind == "repel" else blue


def _length_bounds(cross_size: int, spec: dict[str, object]) -> tuple[int, int] | None:
    blue = int(spec["blue"])
    motif_len = _motif_len(str(spec["kind"]), blue)
    margin = blue + 1
    lo = max(
        BAR_LEN_MIN,
        motif_len + 3,
        (BAR_CROSS_PCT_LO * cross_size + 99) // 100,
    )
    hi = min(
        BAR_LEN_MAX,
        cross_size - 2 * margin,
        (BAR_CROSS_PCT_HI * cross_size) // 100,
    )
    if lo > hi:
        return None
    return lo, hi


def _choose_specs(main_hi: int, cross_size: int) -> list[dict[str, object]] | None:
    for count in BAR_COUNT_CHOICES:
        for _ in range(200):
            if count == 4:
                combos = _COMBOS[:]
                random.shuffle(combos)
            else:
                combos = random.sample(_COMBOS, count)
                if {kind for kind, _ in combos} != {"repel", "attract"}:
                    continue
                if {side for _, side in combos} != {"left", "right"}:
                    continue
            specs = [
                {
                    "kind": kind,
                    "side": side,
                    "blue": random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI),
                }
                for kind, side in combos
            ]
            if any(_length_bounds(cross_size, spec) is None for spec in specs):
                continue
            clears = [int(spec["blue"]) + 1 if spec["kind"] == "repel" else 1 for spec in specs]
            if _min_main_len(clears) <= main_hi:
                return specs
    return None


def _min_main_len(clears: list[int]) -> int:
    row = clears[0] + 1
    for prev_clear, next_clear in zip(clears, clears[1:]):
        row += prev_clear + next_clear + 2
    return row + clears[-1] + 2


def _paint_bar(
    grid: np.ndarray,
    row: int,
    c0: int,
    length: int,
    *,
    kind: str,
    side: str,
    blue: int,
) -> None:
    colors = np.full(length, RED, dtype=int)
    if kind == "repel":
        motif = np.full(2 * blue - 1, RED, dtype=int)
        motif[::2] = BLUE
    else:
        motif = np.full(blue, BLUE, dtype=int)
    if side == "left":
        colors[: len(motif)] = motif
    else:
        colors[-len(motif) :] = motif
    grid[row, c0 : c0 + length] = colors


def _canonical_grid(main_size: int, cross_size: int, specs: list[dict[str, object]]) -> np.ndarray | None:
    clears = [int(spec["blue"]) + 1 if spec["kind"] == "repel" else 1 for spec in specs]
    slack = main_size - _min_main_len(clears)
    if slack < 0:
        return None

    lane_lo = cross_size // 2 - 1
    lane_hi = cross_size // 2 + 1
    lane_col = random.randint(max(2, lane_lo), min(cross_size - 3, lane_hi))

    extras = _random_composition(slack, len(specs) + 1)
    row = clears[0] + 1 + extras[0]
    rows = [row]
    for i in range(1, len(specs)):
        row += clears[i - 1] + clears[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, spec in zip(rows, specs):
        bounds = _length_bounds(cross_size, spec)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        blue = int(spec["blue"])
        margin = blue + 1
        min_c0 = max(margin, lane_col - length + 1)
        max_c0 = min(lane_col, cross_size - margin - length)
        if min_c0 > max_c0:
            return None
        c0 = random.randint(min_c0, max_c0)
        _paint_bar(
            grid,
            row,
            c0,
            length,
            kind=str(spec["kind"]),
            side=str(spec["side"]),
            blue=blue,
        )

    grid[0, lane_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.rot90(grid, 2).copy()
    if edge == "left":
        return np.rot90(grid, 1).copy()
    return np.rot90(grid, -1).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    for _ in range(120):
        cross_size = random.randint(cross_lo, cross_hi)
        specs = _choose_specs(main_hi, cross_size)
        if specs is None:
            continue
        clears = [int(spec["blue"]) + 1 if spec["kind"] == "repel" else 1 for spec in specs]
        main_size = random.randint(max(main_lo, _min_main_len(clears)), main_hi)
        canonical = _canonical_grid(main_size, cross_size, specs)
        if canonical is None:
            continue
        inp = _orient_grid(canonical, edge)
        try:
            out = main(inp)
            metrics = _fusion_metrics(inp)
            if metrics["attract_count"] < 1 or metrics["repel_count"] < 1:
                continue
            if metrics["left_count"] < 1 or metrics["right_count"] < 1:
                continue
            if metrics["type_diff"] <= 0 or metrics["side_diff"] <= 0:
                continue
            if not _reaches_opposite_edge(out, edge):
                continue
            if not np.array_equal(main(inp), out):
                continue
            return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
