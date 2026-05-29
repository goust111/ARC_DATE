# Combo variant of 2b83f449 (index 5)
# combo_spec: D2/1+D5/3
# Name: 多色柱+颜色匹配令牌
#
# Design analysis
# - Concrete behavior: the base task has one gate color and one token color. Here there
#   are two gate/bar colors and the same two token colors. Each 1x3 bar opens into a
#   vertical 3-cell post of the same color. A token may drop through a segment exit only
#   if that exit belongs to a gate of the same color. After it finishes falling, one
#   color packs to the far end of the destination segment and the other packs to the near
#   end. Example: if a magenta token enters a segment from the left opening, it stacks at
#   the right end; an orange token entering the same segment stacks at the left end.
# - Complexity calibration: using exactly two colors keeps the rule inferable, while
#   reusing the same palette for gates/posts and tokens makes the matching relation
#   explicit in the pixels instead of hidden in numeric ids. The generator swaps token
#   colors between left/right edges across rows so color is not confounded with side.
# - Hidden assumptions changed from base solver:
#   1. Base assumed a single gate color and a single token color; this variant must parse
#      multiple bar colors from odd rows and multiple token colors from even rows.
#   2. Base assumed every valid segment exit is traversable; here traversal additionally
#      depends on the color of the adjacent opened gate.
#   3. Base packed all tokens to the far end; here packing depends on token color.
#   4. Alternating gate colors every wall row would cap every path at depth 1, so the
#      generator instead assigns row colors in repeated runs, guaranteeing both colors are
#      present while still allowing some long same-color descents.
# - Placement legality: gate centers stay interior, same-row spacing is >= 5, adjacent
#   wall rows differ by >= 2 columns, holes stay off edges, and the inherited repair step
#   enforces <= 1 physical entry/exit per pipe segment.
# - Visual observability and cleanliness: opened posts are vertical colored columns, input
#   bars are horizontal colored bars, and tokens are single colored edge cells. With only
#   colors 6/7/8/0 the grids stay readable. The generator rejects samples where removing
#   color-matching or removing color-dependent packing would leave the output unchanged.

from __future__ import annotations

from collections import defaultdict

import numpy as np

WALL = 0
MAGENTA = 6
ORANGE = 7
BG = 8

TOKEN_FAR = MAGENTA
TOKEN_NEAR = ORANGE
TOKEN_COLORS = (MAGENTA, ORANGE)


def _infer_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[0::2], return_counts=True)
    candidates = [(int(cnt), int(val)) for val, cnt in zip(vals, counts) if int(val) != WALL]
    assert candidates
    return max(candidates)[1]


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segments: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != int(bg):
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == int(bg):
            c += 1
        segments.append((start, c))
        c += 1
    return segments


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


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)]


def _find_issues(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, bg):
            above = _entries_above(grid, r, seg, bg)
            below = _exits_below(grid, r, seg, bg)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int, bg: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == int(bg)]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    out: list[int] = []
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
            out.append(c)
    return out


def _assign_row_colors(n_walls: int, spine_start: int, spine_len: int, rng: np.random.Generator) -> list[int]:
    palette = [MAGENTA, ORANGE]
    start_phase = int(rng.integers(2))
    row_colors = [palette[((idx // 2) + start_phase) % 2] for idx in range(n_walls)]
    run_color = palette[int(rng.integers(2))]
    run_len = min(n_walls - spine_start, max(3, spine_len))
    for idx in range(spine_start, spine_start + run_len):
        row_colors[idx] = run_color
    if len(set(row_colors)) == 1 and n_walls >= 2:
        flip_idx = 0 if spine_start > 0 else n_walls - 1
        row_colors[flip_idx] = MAGENTA if row_colors[flip_idx] == ORANGE else ORANGE
    return row_colors


def _build_opened_grid(
    gates: list[list[int]],
    holes: list[list[int]],
    row_colors: list[int],
    width: int,
) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, centers in enumerate(gates):
        color = int(row_colors[wall_idx])
        wr = 2 * wall_idx + 1
        for center in centers:
            grid[wr - 1, center] = color
            grid[wr, center - 1] = BG
            grid[wr, center] = color
            grid[wr, center + 1] = BG
            grid[wr + 1, center] = color
    return grid


def _token_colors_for_row(pipe_idx: int) -> tuple[int, int]:
    if pipe_idx % 2 == 0:
        return MAGENTA, ORANGE
    return ORANGE, MAGENTA


def _build_input(
    gates: list[list[int]],
    holes: list[list[int]],
    row_colors: list[int],
    width: int,
) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, centers in enumerate(gates):
        color = int(row_colors[wall_idx])
        wr = 2 * wall_idx + 1
        for center in centers:
            grid[wr, center - 1 : center + 2] = color
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        left_color, right_color = _token_colors_for_row(pipe_idx)
        grid[r, 0] = left_color
        grid[r, width - 1] = right_color
    return grid


def _repair_holes(
    gates: list[list[int]],
    holes: list[list[int]],
    row_colors: list[int],
    width: int,
) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]
    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], row_colors, width)
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


def _build_structure(
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[list[int]], list[int], int, int]:
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
            raise RuntimeError("failed to seed spine")
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left

    row_colors = _assign_row_colors(n_walls, spine_start, spine_len, rng)
    repaired = _repair_holes(gates, holes, row_colors, width)
    if repaired is None:
        raise RuntimeError("failed to repair initial holes")
    holes = repaired

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
            new_holes = _repair_holes(new_gates, holes, row_colors, width)
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
        raise RuntimeError("gate count out of range")
    return gates, holes, row_colors, width, spine_len


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[dict[str, int]], int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1
    bg = _infer_bg(g)
    active_colors = {int(c) for c in np.unique(g) if int(c) not in (WALL, bg)}
    gates: list[list[tuple[int, int]]] = [[] for _ in range((h - 1) // 2)]
    for wall_idx in range(len(gates)):
        wr = 2 * wall_idx + 1
        row = g[wr]
        for c in range(1, w - 1):
            color = int(row[c])
            if color not in active_colors:
                continue
            if int(row[c - 1]) == color and int(row[c + 1]) == color:
                gates[wall_idx].append((c, color))

    sources: list[dict[str, int]] = []
    order = 0
    for pipe_idx in range((h - 1) // 2):
        r = 2 * pipe_idx
        for c in range(w):
            color = int(g[r, c])
            if color not in active_colors:
                continue
            side = -1
            if c == 0:
                side = 0
            elif c == w - 1:
                side = 1
            sources.append({"r": r, "c": c, "color": color, "side": side, "order": order})
            order += 1

    opened = g.copy()
    for wall_idx, row_gates in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center, color in row_gates:
            opened[wr, center - 1 : center + 2] = bg
            opened[wr - 1, center] = color
            opened[wr, center] = color
            opened[wr + 1, center] = color
    for src in sources:
        opened[src["r"], src["c"]] = bg
    return opened, gates, sources, bg


def _exit_color(gates: list[list[tuple[int, int]]], wall_idx: int, exit_col: int) -> int | None:
    if not (0 <= wall_idx < len(gates)):
        return None
    for center, color in gates[wall_idx]:
        if abs(int(center) - int(exit_col)) == 1:
            return int(color)
    return None


def _simulate(
    opened: np.ndarray,
    gates: list[list[tuple[int, int]]],
    sources: list[dict[str, int]],
    bg: int,
    *,
    allow_mismatch: bool,
    uniform_pack: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = opened.copy()
    records: list[dict[str, object]] = []
    for source in sources:
        cur_r = int(source["r"])
        cur_c = int(source["c"])
        depth = 0
        while True:
            seg = _find_segment(opened, cur_r, cur_c, bg)
            if seg is None:
                break
            exits = _exits_below(opened, cur_r, seg, bg)
            assert len(exits) <= 1, f"segment {seg} at row {cur_r} has exits {exits}"
            if not exits:
                break
            exit_col = int(exits[0])
            if not allow_mismatch:
                ex_color = _exit_color(gates, cur_r // 2, exit_col)
                assert ex_color is not None
                if int(ex_color) != int(source["color"]):
                    break
            cur_r += 2
            cur_c = exit_col
            depth += 1
            if cur_r >= opened.shape[0]:
                break
        seg = _find_segment(opened, cur_r, cur_c, bg)
        assert seg is not None
        records.append(
            {
                "r": cur_r,
                "seg": seg,
                "final_c": cur_c,
                "depth": depth,
                "source": source,
                "color": int(source["color"]),
            }
        )

    grouped: dict[tuple[int, tuple[int, int]], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[(int(rec["r"]), rec["seg"])].append(rec)

    for (r, seg), recs in grouped.items():
        a, b = seg
        entries = _entries_above(opened, r, seg, bg)
        assert len(entries) <= 1, f"segment {seg} at row {r} has entries {entries}"
        near_end: int | None = None
        far_end: int | None = None
        if entries:
            entry = int(entries[0])
            near_end = a if (entry - a) <= (b - entry) else b
            far_end = b if near_end == a else a
        left_recs: list[dict[str, object]] = []
        right_recs: list[dict[str, object]] = []
        for rec in sorted(recs, key=lambda item: int(item["source"]["order"])):  # type: ignore[index]
            source = rec["source"]  # type: ignore[assignment]
            if not entries:
                pack_end = a if int(source["side"]) == 0 else b
            else:
                use_far = uniform_pack or int(rec["color"]) == TOKEN_FAR
                pack_end = int(far_end if use_far else near_end)
            rec["pack_end"] = pack_end
            if pack_end == a:
                left_recs.append(rec)
            else:
                right_recs.append(rec)

        left_pos = a
        right_pos = b
        for rec in left_recs:
            if left_pos > right_pos:
                break
            out[r, left_pos] = int(rec["color"])
            left_pos += 1
        for rec in right_recs:
            if right_pos < left_pos:
                break
            out[r, right_pos] = int(rec["color"])
            right_pos -= 1

    return out, records


def _solve_ablation(
    grid: np.ndarray,
    *,
    allow_mismatch: bool,
    uniform_pack: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    opened, gates, sources, bg = _parse_input(grid)
    return _simulate(opened, gates, sources, bg, allow_mismatch=allow_mismatch, uniform_pack=uniform_pack)


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """
    Quantify whether the generated sample truly uses both fused controls.

    blocked_count:
        How many tokens stop earlier once color-matching is enforced.
    pack_switch_count:
        How many tokens change destination end once color-specific near/far packing
        is enforced.
    blocked_colors / moved_colors:
        Ensure both token colors actively participate instead of one color being
        visually present but behaviorally irrelevant.
    mixed_segment:
        At least one destination segment receives both colors and sends them to
        opposite ends, making the packing rule observable.
    """
    fused_out, fused_records = _solve_ablation(grid, allow_mismatch=False, uniform_pack=False)
    nomatch_out, nomatch_records = _solve_ablation(grid, allow_mismatch=True, uniform_pack=False)
    uniform_out, uniform_records = _solve_ablation(grid, allow_mismatch=False, uniform_pack=True)

    blocked_count = sum(
        int(fused_records[idx]["depth"]) < int(nomatch_records[idx]["depth"])
        for idx in range(len(fused_records))
    )
    blocked_colors = {
        int(fused_records[idx]["color"])
        for idx in range(len(fused_records))
        if int(fused_records[idx]["depth"]) < int(nomatch_records[idx]["depth"])
    }
    pack_switch_count = sum(
        int(fused_records[idx].get("pack_end", -1)) != int(uniform_records[idx].get("pack_end", -1))
        for idx in range(len(fused_records))
    )
    moved_colors = {int(rec["color"]) for rec in fused_records if int(rec["depth"]) > 0}

    mixed_segment = False
    grouped: dict[tuple[int, tuple[int, int]], list[dict[str, object]]] = defaultdict(list)
    for rec in fused_records:
        grouped[(int(rec["r"]), rec["seg"])].append(rec)
    for recs in grouped.values():
        colors = {int(rec["color"]) for rec in recs}
        ends = {int(rec.get("pack_end", -1)) for rec in recs}
        if len(recs) >= 2 and len(colors) >= 2 and len(ends) >= 2:
            mixed_segment = True
            break

    return {
        "fused_out": fused_out,
        "fused_records": fused_records,
        "nomatch_out": nomatch_out,
        "uniform_out": uniform_out,
        "blocked_count": blocked_count,
        "blocked_colors": blocked_colors,
        "pack_switch_count": pack_switch_count,
        "moved_colors": moved_colors,
        "mixed_segment": mixed_segment,
    }


def main(grid: np.ndarray) -> np.ndarray:
    out, _records = _solve_ablation(np.array(grid, dtype=int), allow_mismatch=False, uniform_pack=False)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    try:
        gates, holes, row_colors, width, spine_len = _build_structure(rng)
    except RuntimeError:
        return None
    opened = _build_opened_grid(gates, holes, row_colors, width)
    if _find_issues(opened, BG):
        return None

    inp = _build_input(gates, holes, row_colors, width)
    metrics = _fusion_metrics(inp)
    fused_out = metrics["fused_out"]
    fused_records = metrics["fused_records"]

    if np.array_equal(fused_out, metrics["nomatch_out"]):
        return None
    if np.array_equal(fused_out, metrics["uniform_out"]):
        return None

    depths = [int(rec["depth"]) for rec in fused_records]
    if max(depths, default=0) < min(3, spine_len):
        return None

    if metrics["blocked_colors"] != set(TOKEN_COLORS):
        return None

    if int(metrics["blocked_count"]) < 2:
        return None
    if int(metrics["pack_switch_count"]) < 2:
        return None
    if metrics["moved_colors"] != set(TOKEN_COLORS):
        return None
    if not bool(metrics["mixed_segment"]):
        return None

    if not np.array_equal(main(inp), fused_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/5")
