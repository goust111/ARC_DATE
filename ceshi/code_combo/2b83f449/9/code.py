# Combo variant of 2b83f449 (index 9)
# combo_spec: D3/2+D8/0
# Name: 惯性移动+洞作出口
#
# Design notes for this fused variant:
# - Concrete behavior:
#   Gates still rotate exactly like the base task, and tokens still end by packing
#   at the end opposite the column where they last entered their final segment.
#   The changed part is the route simulation:
#   1) a token starts with direction "down";
#   2) if a straight vertical channel exists, it keeps falling in that column;
#   3) if straight-down is blocked, it turns inward (left source -> right, right
#      source -> left) and scans horizontally for the first place where it can
#      drop again;
#   4) a black hole in a pipe row is now a trapdoor: a horizontal token that
#      reaches that hole drops there, while a vertical token aligned to that
#      column passes through the hole and keeps falling.
# - Small example:
#   Suppose a left token reaches row r at col 0 and the only ordinary exit below
#   is at col 6. In the base task it would simply teleport to that unique exit.
#   Here it first tries to continue down at col 0, gets blocked, then scans right.
#   If row r has a hole at col 4 with an open wall below, it falls at col 4
#   before reaching the gate exit at col 6. If the next pipe row also has a hole
#   at col 4, the token does not stop there; it passes through and lands lower.
# - Complexity calibration:
#   The generator keeps the original sparse pipe/gate geometry, so the visual
#   language stays readable. It then injects a small number of targeted holes on
#   actual token routes, which makes the fusion visible without flooding rows with
#   special cases.
# - Broken base assumptions and repairs:
#   * Base assumption: each segment only needs one exit lookup. Broken by inertia.
#     Repair: the solver simulates direction state and scans horizontally.
#   * Base assumption: holes only split segments. Broken by D8/0.
#     Repair: holes are treated as solid for resting/packing, but as vertical
#     channels when encountered during motion.
#   * Base assumption: final packing can infer entry purely from topology.
#     Broken because multiple topological entries can coexist once holes become
#     outlets. Repair: each token records the column of its last vertical entry.
# - Placement legality:
#   Holes are only placed on interior pipe cells, max two per row, max four total.
#   Targeted holes are chosen from genuine motion traces so they align with open
#   walls, stay in-bounds, and do not overlap gates or edge tokens.
# - Visual observability:
#   The generator rejects samples unless at least one token horizontally drops via
#   a hole and at least one token vertically passes through a hole, and unless the
#   fused output differs from the original-base output on the same input.

from __future__ import annotations

from collections import defaultdict
import numpy as np
from arc_utils import paste

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _open_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    n_walls = (h - 1) // 2
    opened = g.copy()
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        row = g[wr]
        for c in range(1, w - 1):
            if int(row[c - 1]) == GATE and int(row[c]) == GATE and int(row[c + 1]) == GATE:
                gates[wall_idx].append(c)
        for center in gates[wall_idx]:
            opened[wr, center - 1 : center + 2] = BG
            opened[wr - 1, center] = OPEN
            opened[wr, center] = OPEN
            opened[wr + 1, center] = OPEN
    return opened, gates


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    c = 0
    w = grid.shape[1]
    while c < w:
        if int(grid[r, c]) != BG:
            c += 1
            continue
        start = c
        while c + 1 < w and int(grid[r, c + 1]) == BG:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _drop_trace(grid: np.ndarray, start_r: int, c: int) -> dict[str, object]:
    h = grid.shape[0]
    cursor = start_r
    bg_rows: list[int] = []
    hole_rows: list[int] = []
    while cursor + 2 < h and int(grid[cursor + 1, c]) == BG:
        nr = cursor + 2
        cell = int(grid[nr, c])
        if cell == BG:
            bg_rows.append(nr)
            cursor = nr
        elif cell == WALL:
            hole_rows.append(nr)
            cursor = nr
        else:
            break
    landing_r = bg_rows[-1] if bg_rows else None
    return {
        "landing_r": landing_r,
        "bg_rows": bg_rows,
        "hole_rows": hole_rows,
        "levels": len(bg_rows) + len(hole_rows),
    }


def _scan_horizontal_event(grid: np.ndarray, r: int, c: int, step: int) -> dict[str, object]:
    w = grid.shape[1]
    prev = c
    nc = c + step
    while 0 <= nc < w:
        cell = int(grid[r, nc])
        if cell == WALL:
            trace = _drop_trace(grid, r, nc)
            if trace["landing_r"] is not None:
                return {
                    "drop": True,
                    "landing_r": int(trace["landing_r"]),
                    "col": nc,
                    "via_hole": True,
                    "levels": int(trace["levels"]),
                    "hole_rows": list(trace["hole_rows"]),
                    "vertical_sites": list(trace["bg_rows"][:-1]),
                }
            return {"drop": False, "stop_c": prev}
        if cell != BG:
            return {"drop": False, "stop_c": prev}
        trace = _drop_trace(grid, r, nc)
        if trace["landing_r"] is not None:
            return {
                "drop": True,
                "landing_r": int(trace["landing_r"]),
                "col": nc,
                "via_hole": False,
                "levels": int(trace["levels"]),
                "hole_rows": list(trace["hole_rows"]),
                "vertical_sites": list(trace["bg_rows"][:-1]),
            }
        prev = nc
        nc += step
    return {"drop": False, "stop_c": prev}


def _simulate_opened(
    opened: np.ndarray,
    token_pts: list[tuple[int, int]],
    collect_sites: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    out = opened.copy()
    stats: dict[str, object] = {
        "horizontal_hole_drops": 0,
        "vertical_hole_passes": 0,
        "max_depth": 0,
        "horizontal_sites": [],
        "vertical_sites": [],
    }
    records: list[dict[str, object]] = []

    for order, (start_r, start_c) in enumerate(sorted(token_pts)):
        side = "left" if start_c <= 1 else "right" if start_c >= opened.shape[1] - 2 else "internal"
        step = 1 if side != "right" else -1
        r, c = start_r, start_c
        entry_col: int | None = None
        depth = 0

        for _ in range(opened.shape[0] * opened.shape[1] + 10):
            trace = _drop_trace(opened, r, c)
            if trace["landing_r"] is not None:
                r = int(trace["landing_r"])
                entry_col = c
                depth += int(trace["levels"])
                stats["vertical_hole_passes"] = int(stats["vertical_hole_passes"]) + len(trace["hole_rows"])
                if collect_sites:
                    cast_sites = stats["vertical_sites"]
                    assert isinstance(cast_sites, list)
                    for vr in trace["bg_rows"][:-1]:
                        cast_sites.append((int(vr), c))
                continue

            scan_r = r
            event = _scan_horizontal_event(opened, r, c, step)
            if bool(event["drop"]):
                r = int(event["landing_r"])
                c = int(event["col"])
                entry_col = c
                depth += int(event["levels"])
                stats["vertical_hole_passes"] = int(stats["vertical_hole_passes"]) + len(event["hole_rows"])
                via_hole = bool(event["via_hole"])
                if via_hole:
                    stats["horizontal_hole_drops"] = int(stats["horizontal_hole_drops"]) + 1
                if collect_sites:
                    if not via_hole:
                        cast_h = stats["horizontal_sites"]
                        assert isinstance(cast_h, list)
                        cast_h.append((scan_r, c))
                    cast_v = stats["vertical_sites"]
                    assert isinstance(cast_v, list)
                    for vr in event["vertical_sites"]:
                        cast_v.append((int(vr), c))
                continue

            c = int(event["stop_c"])
            break

        seg = _find_segment(opened, r, c)
        if seg is None:
            seg = (c, c)
        a, b = seg
        if entry_col is None:
            pack_end = a if side != "right" else b
        else:
            pack_end = b if (entry_col - a) <= (b - entry_col) else a
        stats["max_depth"] = max(int(stats["max_depth"]), depth)
        records.append(
            {
                "r": r,
                "seg": seg,
                "pack_end": pack_end,
                "order": order,
            }
        )

    buckets: dict[tuple[int, tuple[int, int], int], list[int]] = defaultdict(list)
    for rec in records:
        buckets[(int(rec["r"]), rec["seg"], int(rec["pack_end"]))].append(int(rec["order"]))

    for (r, (a, b), pack_end), orders in buckets.items():
        orders.sort()
        if pack_end == a:
            for i, _order in enumerate(orders):
                if a + i <= b:
                    out[r, a + i] = TOKEN
        else:
            for i, _order in enumerate(orders):
                if b - i >= a:
                    out[r, b - i] = TOKEN

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    opened, _gates = _open_from_input(g)
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(g == TOKEN)]
    opened[opened == TOKEN] = BG
    out, _stats = _simulate_opened(opened, token_pts, collect_sites=False)
    return out


def _build_opened_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            grid[r, c - 1] = BG
            grid[r, c] = OPEN
            grid[r, c + 1] = BG
            grid[r - 1, c] = OPEN
            grid[r + 1, c] = OPEN

    return grid


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            paste(grid, gate_patch, r, c - 1)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _base_exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _base_entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r):
            above = _base_entries_above(grid, r, seg)
            below = _base_exits_below(grid, r, seg)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(gates: list[list[int]], holes: list[list[int]], width: int) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]
    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_issues(grid)
        if not issues:
            return [sorted(row) for row in hole_sets]

        r, _seg, above, below = issues[0]
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None

        events = sorted([(c, "up") for c in above] + [(c, "down") for c in below])
        seen_up = False
        seen_down = False
        prev_col: int | None = None
        new_cols: list[int] = []

        for col, kind in events:
            duplicate = (kind == "up" and seen_up) or (kind == "down" and seen_down)
            if duplicate:
                assert prev_col is not None
                sep = _separator_column(grid, r, prev_col, col)
                if sep is None:
                    return None
                new_cols.append(sep)
                seen_up = False
                seen_down = False

            if kind == "up":
                seen_up = True
            else:
                seen_down = True
            prev_col = col

        if not new_cols:
            return None

        added_any = False
        for c in new_cols:
            before = len(hole_sets[pipe_idx])
            hole_sets[pipe_idx].add(c)
            if len(hole_sets[pipe_idx]) > before:
                added_any = True
            if len(hole_sets[pipe_idx]) > 2:
                return None

        if not added_any:
            return None

        if sum(len(row) for row in hole_sets) > 4:
            return None


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    candidates: list[int] = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue
        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            candidates.append(c)
    return candidates


def _solve_base_like(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    opened, _gates = _open_from_input(g)
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(g == TOKEN)]
    opened[opened == TOKEN] = BG
    out = opened.copy()
    records: list[tuple[int, tuple[int, int], int, int]] = []

    for start_r, start_c in sorted(token_pts):
        side = "left" if start_c <= 1 else "right"
        r, c = start_r, start_c
        entry_col: int | None = None
        while True:
            seg = _find_segment(opened, r, c)
            if seg is None:
                break
            exits = _base_exits_below(opened, r, seg)
            if len(exits) != 1:
                break
            r += 2
            c = exits[0]
            entry_col = c
        seg = _find_segment(opened, r, c)
        if seg is None:
            seg = (c, c)
        a, b = seg
        if entry_col is None:
            pack_end = a if side == "left" else b
        else:
            pack_end = b if (entry_col - a) <= (b - entry_col) else a
        records.append((r, seg, pack_end, start_r * 100 + start_c))

    buckets: dict[tuple[int, tuple[int, int], int], list[int]] = defaultdict(list)
    for r, seg, pack_end, order in records:
        buckets[(r, seg, pack_end)].append(order)
    for (r, (a, b), pack_end), orders in buckets.items():
        orders.sort()
        if pack_end == a:
            for i, _order in enumerate(orders):
                if a + i <= b:
                    out[r, a + i] = TOKEN
        else:
            for i, _order in enumerate(orders):
                if b - i >= a:
                    out[r, b - i] = TOKEN
    return out


def _add_hole_if_possible(
    holes: list[set[int]],
    pipe_idx: int,
    c: int,
    width: int,
) -> bool:
    if not (0 < c < width - 1):
        return False
    if c in holes[pipe_idx]:
        return False
    if len(holes[pipe_idx]) >= 2:
        return False
    if sum(len(row) for row in holes) >= 4:
        return False
    holes[pipe_idx].add(c)
    return True


def _aligned_vertical_sites(gates: list[list[int]], width: int) -> list[tuple[int, int]]:
    sites: list[tuple[int, int]] = []
    for pipe_idx in range(1, len(gates)):
        exits_above = {c + dc for c in gates[pipe_idx - 1] for dc in (-1, 1)}
        exits_below = {c + dc for c in gates[pipe_idx] for dc in (-1, 1)}
        for c in sorted(exits_above & exits_below):
            if 0 < c < width - 1:
                sites.append((2 * pipe_idx, c))
    return sites


def _guided_hole_injection(
    gates: list[list[int]],
    holes: list[list[int]],
    width: int,
    rng: np.random.Generator,
) -> list[list[int]] | None:
    base_holes = [set(row) for row in holes]
    opened = _build_opened_grid(gates, [sorted(row) for row in base_holes], width)
    token_pts = [(2 * pipe_idx, c) for pipe_idx in range(len(gates)) for c in (0, width - 1)]
    _out, stats = _simulate_opened(opened, token_pts, collect_sites=True)

    horizontal_sites = list(dict.fromkeys(stats["horizontal_sites"]))  # type: ignore[arg-type]
    rng.shuffle(horizontal_sites)
    vertical_sites = _aligned_vertical_sites(gates, width)

    for hr, hc in horizontal_sites:
        if opened[int(hr), int(hc)] != BG:
            continue
        preferred_verticals = [site for site in vertical_sites if site[1] == hc and site[0] > hr]
        fallback_verticals = [site for site in vertical_sites if site not in preferred_verticals]
        rng.shuffle(fallback_verticals)
        for vr, vc in preferred_verticals + fallback_verticals:
            if opened[int(vr), int(vc)] != BG:
                continue
            hole_sets = [set(row) for row in base_holes]
            if not _add_hole_if_possible(hole_sets, int(hr) // 2, int(hc), width):
                continue
            if not _add_hole_if_possible(hole_sets, int(vr) // 2, int(vc), width):
                continue
            return [sorted(row) for row in hole_sets]
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(6, 9))
    width = int(rng.integers(16, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = n_walls
    spine_start = 0
    bridge_step = int(rng.integers(1, spine_len))
    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for offset, row in enumerate(range(spine_start, spine_start + spine_len)):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif offset == bridge_step:
            forced = prev_center - 2 if go_left else prev_center + 2
            options = [forced] if 2 <= forced <= width - 3 else []
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 2)]
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left

    holes = _repair_holes(gates, holes, width)
    if holes is None or sum(len(row) for row in holes) > 2:
        return None

    total_gates = sum(len(row) for row in gates)
    if not (6 <= total_gates <= 12):
        return None

    holes = _guided_hole_injection(gates, holes, width, rng)
    if holes is None:
        return None

    inp = _build_input(gates, holes, width)
    out = main(inp)
    opened, _ = _open_from_input(inp)
    opened[opened == TOKEN] = BG
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(inp == TOKEN)]
    direct_out, stats = _simulate_opened(opened, token_pts, collect_sites=False)
    if not np.array_equal(out, direct_out):
        return None
    if int(stats["horizontal_hole_drops"]) < 1:
        return None
    if int(stats["vertical_hole_passes"]) < 1:
        return None
    if int(stats["max_depth"]) < 3:
        return None
    if np.array_equal(out, _solve_base_like(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo D3/2+D8/0")
