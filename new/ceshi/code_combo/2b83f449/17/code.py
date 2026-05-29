# Combo variant of 2b83f449 (index 17)
# combo_spec: D4/0+D5/3
# Name: 颜色令牌+近端堆积

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste


# Design analysis
# a) Concrete behavior:
#    Original: every token packs to the far end of its terminal segment.
#    Combo: token color selects packing direction after the same downward route.
#    GREEN keeps the original "far from entry" rule, BLUE uses the inverted
#    "near the entry" rule. Example: if a token enters segment [2,7] from c=2,
#    GREEN settles toward c=7 while BLUE settles toward c=2.
#    To make this visible, the generator uses both colors repeatedly and lets
#    each color appear on both left and right source edges across rows.
# b) Complexity calibration:
#    Only two token colors are used, and the mapping is fixed across all samples.
#    This keeps the branching small enough to infer from a few examples while
#    still producing visibly different output distributions from both all-far
#    and all-near baselines.
# c) Broken implicit assumptions from the original solver:
#    - "there is exactly one token color" is no longer true.
#    - "all tokens in one bucket share the same color" is no longer true.
#    - "one global packing rule applies to every token" is no longer true.
#    The combo solver therefore tracks token color per source and stores color
#    lists in packing buckets instead of mere counts.
# d) Placement legality:
#    Tokens stay on pipe-row edge cells only; gates stay interior as 1x3 bars;
#    holes remain interior single black cells. The original hole-repair logic is
#    preserved so every pipe segment still has at most one entry and one exit.
# e) Visual observability:
#    The generator rejects samples whose combo output matches either the all-far
#    baseline or the all-near baseline, so color-conditioned behavior must
#    materially affect the visible token layout.
# f) Visual cleanliness:
#    No new overlays are introduced. The only added signal is token color, which
#    reuses isolated edge pixels and settled token cells already present in the
#    base task.
# g) Inferrability of color mapping:
#    Both colors appear multiple times and on both source sides by alternating
#    row parity: even pipe rows use left=GREEN/right=BLUE, odd rows invert that.
#    This removes the confound "color == side" and leaves color as the stable
#    explanation for near-vs-far packing.


WALL = 0
BLUE = 1
GREEN = 3
OPEN = 6
GATE = 7
BG = 8

TOKEN_COLORS = (BLUE, GREEN)


def _find_gate_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.flatten())))
    for col in colors:
        if col == WALL:
            continue
        lab, n = label_mask(grid == col, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                return int(col)
    raise AssertionError("failed to infer gate color")


def _infer_bg_and_tokens(grid: np.ndarray, gate: int) -> tuple[int, list[int]]:
    colors = set(map(int, grid.flatten()))
    counts = Counter(map(int, grid.flatten()))
    bg_candidates = [c for c in colors if c not in (WALL, gate)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[c]), -int(c))))
    token_colors = sorted(c for c in colors if c not in (WALL, gate, bg))
    assert token_colors
    return bg, token_colors


def _pack_mode_for_color(color: int) -> str:
    return "near" if int(color) == BLUE else "far"


def _source_specs(n_walls: int, width: int) -> list[dict[str, int | str]]:
    specs: list[dict[str, int | str]] = []
    order = 0
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        if pipe_idx % 2 == 0:
            colors = [GREEN, BLUE]
        else:
            colors = [BLUE, GREEN]
        for c, color in zip((0, width - 1), colors):
            specs.append(
                {
                    "r": r,
                    "c": c,
                    "color": color,
                    "side": "left" if c == 0 else "right",
                    "order": order,
                }
            )
            order += 1
    return specs


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != int(bg):
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == int(bg):
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)]


def _entry_from_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r < 2:
        return None
    a, b = seg
    entries = [c for c in range(a, b + 1) if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)]
    assert len(entries) <= 1, f"segment ({a},{b}) at row {r} has {len(entries)} entries: {entries}"
    return entries[0] if entries else None


def _find_issues(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, bg):
            above = [] if r < 2 else [c for c in range(seg[0], seg[1] + 1) if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)]
            below = _exits_below(grid, r, seg, bg)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int, bg: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == int(bg)]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


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

    for spec in _source_specs(n_walls, width):
        grid[int(spec["r"]), int(spec["c"])] = int(spec["color"])

    return grid


def _repair_holes(gates: list[list[int]], holes: list[list[int]], width: int) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]

    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_issues(grid, BG)
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
                sep = _separator_column(grid, r, prev_col, col, BG)
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

        for c in new_cols:
            hole_sets[pipe_idx].add(c)
            if len(hole_sets[pipe_idx]) > 2:
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


def _simulate_output(opened: np.ndarray, mode: str = "combo") -> tuple[np.ndarray, list[dict[str, int | str]], list[int]]:
    height, width = opened.shape
    n_walls = (height - 1) // 2
    specs = _source_specs(n_walls, width)

    finals: list[dict[str, int | str]] = []
    drop_counts: list[int] = []
    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        drops = 0
        while True:
            seg = _find_segment(opened, r, c, BG)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg, BG)
            assert len(exits) <= 1, f"segment {seg} at row {r} has multiple exits {exits}"
            if not exits:
                break
            r += 2
            c = exits[0]
            drops += 1
        finals.append(
            {
                "r": r,
                "c": c,
                "color": int(spec["color"]),
                "order": int(spec["order"]),
                "drops": drops,
            }
        )
        drop_counts.append(drops)

    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], list[tuple[int, int]]] = defaultdict(list)
    for rec in finals:
        r = int(rec["r"])
        c = int(rec["c"])
        seg = _find_segment(opened, r, c, BG)
        if seg is None:
            continue
        a, b = seg
        entry = _entry_from_above(opened, r, seg, BG)
        if mode == "combo":
            pack_mode = _pack_mode_for_color(int(rec["color"]))
        else:
            pack_mode = mode

        if entry is not None:
            near_end = a if (entry - a) <= (b - entry) else b
            far_end = b if near_end == a else a
            pack_end = near_end if pack_mode == "near" else far_end
        else:
            pack_end = a if abs(c - a) <= abs(c - b) else b

        buckets[(r, seg, pack_end)].append((int(rec["order"]), int(rec["color"])))

    for (r, (a, b), pack_end), items in buckets.items():
        ordered = [color for _order, color in sorted(items)]
        if pack_end == a:
            for i, color in enumerate(ordered):
                if a + i <= b:
                    out[r, a + i] = color
        else:
            for i, color in enumerate(ordered):
                if b - i >= a:
                    out[r, b - i] = color

    return out, finals, drop_counts


def _token_diff_count(a: np.ndarray, b: np.ndarray) -> int:
    token_mask = np.isin(a, TOKEN_COLORS) | np.isin(b, TOKEN_COLORS)
    return int(np.count_nonzero((a != b) & token_mask))


def _behavior_metrics_from_opened(opened: np.ndarray) -> dict[str, int]:
    combo, records, drop_counts = _simulate_output(opened, "combo")
    far, _, _ = _simulate_output(opened, "far")
    near, _, _ = _simulate_output(opened, "near")
    blue_drops = sum(1 for rec in records if int(rec["color"]) == BLUE and int(rec["drops"]) > 0)
    green_drops = sum(1 for rec in records if int(rec["color"]) == GREEN and int(rec["drops"]) > 0)
    return {
        "max_drop": max(drop_counts, default=0),
        "blue_drops": blue_drops,
        "green_drops": green_drops,
        "diff_from_far": _token_diff_count(combo, far),
        "diff_from_near": _token_diff_count(combo, near),
    }


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    gate = _find_gate_color(np.array(grid, dtype=int))
    bg, token_colors = _infer_bg_and_tokens(np.array(grid, dtype=int), gate)
    assert bg == BG
    assert set(token_colors) == {BLUE, GREEN}
    opened = main(grid).copy()
    opened[np.isin(opened, TOKEN_COLORS)] = BG
    return _behavior_metrics_from_opened(opened)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    height, width = g.shape

    gate = _find_gate_color(g)
    bg, token_colors = _infer_bg_and_tokens(g, gate)
    open_col = 6 if int(gate) == 7 and 6 not in set(map(int, g.flatten())) else int(gate - 1) if int(gate) > 0 else 6

    out = g.copy()
    lab, n = label_mask(out == int(gate), connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        c0 = int(c0)
        center_c = c0 + 1
        out[r, c0 : c0 + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center_c] = int(open_col)

    token_pts = [(int(r), int(c), int(out[r, c])) for r, c in np.argwhere(np.isin(out, token_colors))]
    out[np.isin(out, token_colors)] = int(bg)

    finals: list[tuple[int, int, int, int]] = []
    for order, (r, c, color) in enumerate(token_pts):
        cur_r, cur_c = r, c
        if int(out[cur_r, cur_c]) != int(bg):
            finals.append((order, cur_r, cur_c, color))
            continue
        while True:
            seg = _find_segment(out, cur_r, cur_c, bg)
            if seg is None:
                break
            exits = _exits_below(out, cur_r, seg, bg)
            assert len(exits) <= 1, f"segment {seg} at row {cur_r} has multiple exits {exits}"
            if not exits:
                break
            cur_r += 2
            cur_c = exits[0]
            if cur_r >= height:
                break
        finals.append((order, cur_r, cur_c, color))

    buckets: dict[tuple[int, tuple[int, int], int], list[tuple[int, int]]] = defaultdict(list)
    for order, r, c, color in finals:
        seg = _find_segment(out, r, c, bg)
        if seg is None:
            continue
        a, b = seg
        entry = _entry_from_above(out, r, seg, bg)
        pack_mode = _pack_mode_for_color(color)
        if entry is not None:
            near_end = a if (entry - a) <= (b - entry) else b
            far_end = b if near_end == a else a
            pack_end = near_end if pack_mode == "near" else far_end
        else:
            pack_end = a if abs(c - a) <= abs(c - b) else b
        buckets[(r, seg, pack_end)].append((order, color))

    for (r, (a, b), pack_end), items in buckets.items():
        ordered = [color for _order, color in sorted(items)]
        if pack_end == a:
            for i, color in enumerate(ordered):
                if a + i <= b:
                    out[r, a + i] = color
        else:
            for i, color in enumerate(ordered):
                if b - i >= a:
                    out[r, b - i] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
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
    if holes is None:
        return None

    total_gates = sum(len(row) for row in gates)
    target_total = int(rng.integers(max(6, total_gates), 13))

    while total_gates < target_total:
        candidates: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                candidates.append((row, center))

        if not candidates:
            break

        order = rng.permutation(len(candidates))
        added = False
        for idx in order:
            row, center = candidates[int(idx)]
            new_gates = [list(cols) for cols in gates]
            new_gates[row] = sorted(new_gates[row] + [center])
            new_holes = _repair_holes(new_gates, holes, width)
            if new_holes is None:
                continue
            gates = new_gates
            holes = new_holes
            total_gates += 1
            added = True
            break

        if not added:
            break

    if not (6 <= total_gates <= 12):
        return None

    opened = _build_opened_grid(gates, holes, width)
    if _find_issues(opened, BG):
        return None

    output, _records, _drop_counts = _simulate_output(opened, "combo")
    metrics = _behavior_metrics_from_opened(opened)
    if metrics["max_drop"] < 3:
        return None
    if metrics["blue_drops"] == 0 or metrics["green_drops"] == 0:
        return None
    if metrics["diff_from_far"] < 2 or metrics["diff_from_near"] < 2:
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/17")
