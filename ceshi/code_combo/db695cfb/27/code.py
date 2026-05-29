# Combo variant of db695cfb (index 27)
# combo_spec: D3/0+D7/2+D9/2
# Name: 双端生长+障碍阻断+碰撞停止

"""
Design analysis for the D3/0 + D7/2 + D9/2 combo variant.

a) Concrete behavior and trace:
   - Base vocabulary stays close to the original task: blue endpoints define 45-degree segments and magenta pixels
     can sit on those segments.
   - The fused rule changes three stages at once:
     1. A blue segment no longer gets filled in one sweep. Two growth fronts start from the two blue endpoints and
        move inward one cell per round.
     2. If a front steps onto a magenta cell, that magenta cell remains magenta, immediately emits the perpendicular
        diagonal rays, and that front stops there.
     3. Those rays no longer tunnel through structure. They paint background only and stop before the first non-bg cell.
   - Small trace on a 7-cell "\" path with a central magenta trigger:
       endpoints at indices 0 and 6, trigger at index 3, barrier segment on one emitted "/" ray.
       original task: blue fills indices 1,2,4,5 and the magenta rays go to the border.
       one-sided obstacle variant: only indices 1,2 become blue, because tracing stops when it reaches index 3.
       fused variant: round 1 paints indices 1 and 5; round 2 paints 2 and 4; round 3 both fronts hit the trigger
       at 3, stop there, and emit rays; the blocked ray truncates when it reaches the barrier endpoint.
     The visible difference is therefore twofold: blue appears on both sides of the trigger, but the trigger still
     prevents the segment from becoming continuous, and at least one magenta branch is cut short by an obstacle.

b) Complexity calibration:
   - The generator guarantees one "center-blocked" segment so D9/2 is legible: the dual-growth output must differ
     from a one-sided blocker solver on the same input.
   - It also guarantees one ray-barrier interaction so D3/0 is legible: the fused output must differ from the same
     dual-growth solver with boundary-reaching rays.
   - Extra segments are optional and triggerless. They preserve the original task's mixed-diagonal feel without hiding
     the main causal chain under many overlapping emitters.

c) Implicit-assumption audit against the original solver:
   - Original `draw_segment()` assumed the whole diagonal could be scanned in one pass. That fails here because growth
     is stateful and may stop from either side at different times.
   - Original trigger collection assumed triggers are transparent and processed after all blue drawing. That fails
     because the trigger now blocks the front and its ray must be emitted immediately.
   - Original ray loop assumed non-bg cells are ignorable obstacles. That fails here because the first non-bg cell is
     now a hard stop.
   - The original task tolerated multiple triggers on one segment because all stayed relevant. In the fused rule,
     nearer triggers can hide farther ones from one side, so the generator keeps the guaranteed interacting segment to
     a single centered trigger.
   - Because rays are now obstacles, execution order matters. Solver and generator therefore share the same simulation
     helper and the validator compares policies by rerunning that exact helper.

d) Placement legality:
   - Segment paths are pairwise disjoint.
   - The guaranteed barrier segment lies on one perpendicular ray diagonal but excludes the trigger cell itself.
   - Full segment corridors are reserved so distractors cannot appear inside blocked tails.
   - Extra segments are only placed on cells that remain background under all comparison policies, which prevents them
     from altering the guaranteed interaction.

e) Visual observability:
   - Every accepted sample must satisfy two measurable differences:
     1. fused output != one-sided obstacle output
     2. fused output != dual-growth-without-collision-stop output
   - This ensures that both the dual-growth behavior and the collision-stop behavior are visible in 2-4 examples.
   - The rule remains human-visible because it relies only on position and contact, not on hidden ordering by color id.

f) Visual cleanliness:
   - The generator isolates one main interacting pair first, then adds only safe extras.
   - Distractors are restricted to cells that stay background in all compared executions, so they cannot silently become
     blockers or accidental triggers.
   - The result keeps one dominant local story: a centered magenta blocker with blue growth on both sides and at least
     one visibly truncated magenta branch.

g) New solver assumptions and generator support:
   - Solver still assumes one color owns the longest same-color diagonal pair; generator keeps blue as that color.
   - Solver groups blue points by exact diagonal id and connects each group's extremes; generator forbids two intended
     segments from sharing a diagonal id.
   - Solver treats any non-line-color cell reached by a growth front as a blocker-and-trigger. Generator keeps the
     meaningful blockers magenta and prevents output-created overlaps from creating ambiguous secondary cases.
"""

import random

import numpy as np

from arc_utils import get_bg


_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (2, 3)
_SEG_LEN_RANGE = (3, 16)
_MAIN_SEG_LEN_RANGE = (6, 12)
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 1)
_SEGMENT_PLACE_RETRIES = 100
_PAIR_PLACE_RETRIES = 180
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 500


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _segment_slope(path: list[tuple[int, int]]) -> str:
    return "backslash" if path[1][1] > path[0][1] else "slash"


def _segment_key(path: list[tuple[int, int]]) -> tuple[str, int]:
    if _segment_slope(path) == "backslash":
        return ("backslash", path[0][0] - path[0][1])
    return ("slash", path[0][0] + path[0][1])


def _perp_dirs_for_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dc = 1 if path[1][1] > path[0][1] else -1
    return [(1, -dc), (-1, dc)]


def _cells_along_ray(
    start: tuple[int, int], delta: tuple[int, int], H: int, W: int
) -> list[tuple[int, int]]:
    dr, dc = delta
    r, c = start[0] + dr, start[1] + dc
    cells = []
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _detect_line_color(by_color: dict[int, list[tuple[int, int]]]) -> tuple[int, int] | None:
    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand
    return best


def _segment_paths_for_points(pts: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    paths: list[list[tuple[int, int]]] = []
    for key in sorted(groups_a):
        g = sorted(groups_a[key])
        if len(g) >= 2:
            paths.append(_diag_path(g[0], g[-1]))
    for key in sorted(groups_b):
        g = sorted(groups_b[key])
        if len(g) >= 2:
            paths.append(_diag_path(g[0], g[-1]))
    return paths


def _emit_rays(
    out: np.ndarray,
    bg: int,
    line_color: int,
    trigger: tuple[int, int, int],
    dirs: list[tuple[int, int]],
    *,
    stop_on_collision: bool,
    ray_records: list[dict],
) -> None:
    h, w = out.shape
    r0, c0, tcol = trigger
    for ddr, ddc in dirs:
        record = {
            "trigger": (r0, c0),
            "color": tcol,
            "dir": (ddr, ddc),
            "painted": 0,
            "blocked": False,
            "block_color": None,
            "block_kind": None,
        }
        r, c = r0 + ddr, c0 + ddc
        while 0 <= r < h and 0 <= c < w:
            v = int(out[r, c])
            if stop_on_collision and v != bg:
                record["blocked"] = True
                record["block_color"] = v
                if v == line_color:
                    record["block_kind"] = "line"
                elif v == tcol:
                    record["block_kind"] = "trigger"
                else:
                    record["block_kind"] = "other"
                break
            if v == bg:
                out[r, c] = tcol
                record["painted"] += 1
            r += ddr
            c += ddc
        ray_records.append(record)


def _solve_with_policy(
    grid: np.ndarray,
    *,
    growth_mode: str,
    stop_rays_on_collision: bool,
) -> tuple[np.ndarray, dict]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = _detect_line_color(by_color)
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    paths = _segment_paths_for_points(by_color[line_color])

    trigger_dirs: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    trigger_hits: dict[tuple[int, int, int], set[str]] = {}
    ray_records: list[dict] = []
    segment_records: list[dict] = []

    def register_trigger(
        key: tuple[int, int, int],
        dirs: list[tuple[int, int]],
        side_labels: list[str],
    ) -> None:
        hits = trigger_hits.setdefault(key, set())
        hits.update(side_labels)
        known = trigger_dirs.setdefault(key, [])
        fresh = [d for d in dirs if d not in known]
        if fresh:
            known.extend(fresh)
            _emit_rays(
                out,
                bg,
                line_color,
                key,
                fresh,
                stop_on_collision=stop_rays_on_collision,
                ray_records=ray_records,
            )

    for path in paths:
        perp_dirs = _perp_dirs_for_path(path)
        local_hits: dict[tuple[int, int, int], set[str]] = {}
        left_blue = 0
        right_blue = 0

        if growth_mode == "one_sided":
            for idx, (r, c) in enumerate(path):
                v = int(out[r, c])
                if v == bg:
                    out[r, c] = line_color
                    left_blue += 1
                    continue
                if idx > 0 and v != line_color:
                    key = (r, c, v)
                    local_hits.setdefault(key, set()).add("forward")
                    register_trigger(key, perp_dirs, ["forward"])
                    break
        elif growth_mode == "dual":
            left_idx = 0
            right_idx = len(path) - 1
            left_active = True
            right_active = True

            while left_active or right_active:
                left_next = left_idx + 1 if left_active and left_idx + 1 <= right_idx else None
                right_next = right_idx - 1 if right_active and right_idx - 1 >= left_idx else None

                if left_next is None and right_next is None:
                    break
                if left_next is not None and right_next is not None and left_next > right_next:
                    break

                round_triggers: list[tuple[tuple[int, int, int], list[str]]] = []

                if left_next is not None and right_next is not None and left_next == right_next:
                    r, c = path[left_next]
                    v = int(out[r, c])
                    if v == bg:
                        out[r, c] = line_color
                        left_blue += 1
                        right_blue += 1
                    elif v != line_color:
                        key = (r, c, v)
                        round_triggers.append((key, ["left", "right"]))
                    left_idx = left_next
                    right_idx = right_next
                    left_active = False
                    right_active = False
                else:
                    if left_next is not None:
                        r, c = path[left_next]
                        v = int(out[r, c])
                        left_idx = left_next
                        if v == bg:
                            out[r, c] = line_color
                            left_blue += 1
                        elif v != line_color:
                            key = (r, c, v)
                            round_triggers.append((key, ["left"]))
                            left_active = False
                        else:
                            left_active = False

                    if right_next is not None:
                        r, c = path[right_next]
                        v = int(out[r, c])
                        right_idx = right_next
                        if v == bg:
                            out[r, c] = line_color
                            right_blue += 1
                        elif v != line_color:
                            key = (r, c, v)
                            round_triggers.append((key, ["right"]))
                            right_active = False
                        else:
                            right_active = False

                merged_round_hits: dict[tuple[int, int, int], set[str]] = {}
                for key, labels in round_triggers:
                    merged_round_hits.setdefault(key, set()).update(labels)
                for key, labels in merged_round_hits.items():
                    local_hits.setdefault(key, set()).update(labels)
                    register_trigger(key, perp_dirs, sorted(labels))
        else:
            raise ValueError(f"Unsupported growth_mode: {growth_mode}")

        segment_records.append(
            {
                "path": path,
                "trigger_keys": sorted(local_hits),
                "left_blue": left_blue,
                "right_blue": right_blue,
                "dual_contact_trigger": any({"left", "right"} <= hits for hits in local_hits.values()),
                "two_sided_blue": bool(local_hits) and left_blue > 0 and right_blue > 0,
            }
        )

    info = {
        "bg": bg,
        "line_color": line_color,
        "trigger_count": len(trigger_hits),
        "trigger_hits": {k: set(v) for k, v in trigger_hits.items()},
        "segment_records": segment_records,
        "ray_records": ray_records,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each blue diagonal inward from both endpoints, letting magenta blockers branch immediately and rays stop on impact.

    Input structure:
    - The background is the most common color. One non-background color provides endpoint pairs on shared diagonals, and magenta pixels may lie on those diagonal corridors between the endpoints.

    Rule:
    - For every diagonal owned by the line color, start one growth front from each blue endpoint and advance both fronts one cell per round toward the middle.
    - A front paints a cell blue only if that cell is background. If it steps onto a magenta obstacle, it stops there instead of passing through, and that magenta cell immediately emits the two perpendicular diagonal rays.
    - Each emitted ray paints background cells only and halts as soon as the next cell on that ray is already occupied by any non-background pixel. Growth also stops when the two inward-moving blue fronts meet in the middle.

    Key insight: The same magenta cell now plays two roles at once: it cuts the blue connection and becomes the branching source, while the branch itself is constrained by nearby structure because collisions terminate the ray.
    """

    out, _ = _solve_with_policy(grid, growth_mode="dual", stop_rays_on_collision=True)
    return out


def _build_input(H: int, W: int, bg: int, segments: list[dict]) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        path = seg["path"]
        inp[path[0]] = _LINE_COLOR
        inp[path[-1]] = _LINE_COLOR
        for tp in seg["triggers"]:
            inp[tp] = _TRIGGER_COLOR
    return inp


def _variant_metrics(inp: np.ndarray) -> dict:
    combo_out, combo_info = _solve_with_policy(
        inp, growth_mode="dual", stop_rays_on_collision=True
    )
    one_sided_out, one_sided_info = _solve_with_policy(
        inp, growth_mode="one_sided", stop_rays_on_collision=True
    )
    no_collision_out, no_collision_info = _solve_with_policy(
        inp, growth_mode="dual", stop_rays_on_collision=False
    )

    combo_map = {
        (tuple(rec["trigger"]), tuple(rec["dir"])): rec for rec in combo_info["ray_records"]
    }
    no_collision_map = {
        (tuple(rec["trigger"]), tuple(rec["dir"])): rec for rec in no_collision_info["ray_records"]
    }
    shortened_rays = sum(
        1
        for key, rec in combo_map.items()
        if key in no_collision_map and rec["painted"] < no_collision_map[key]["painted"]
    )

    line_color = combo_info["line_color"]
    extra_blue_vs_one_sided = int(
        np.count_nonzero((combo_out == line_color) & (one_sided_out != line_color))
    )

    return {
        "combo_out": combo_out,
        "one_sided_out": one_sided_out,
        "no_collision_out": no_collision_out,
        "combo_info": combo_info,
        "one_sided_info": one_sided_info,
        "no_collision_info": no_collision_info,
        "blocked_rays": sum(1 for r in combo_info["ray_records"] if r["blocked"]),
        "blocked_by_line": sum(1 for r in combo_info["ray_records"] if r["block_kind"] == "line"),
        "shortened_rays": shortened_rays,
        "differs_from_one_sided": not np.array_equal(combo_out, one_sided_out),
        "differs_from_no_collision": not np.array_equal(combo_out, no_collision_out),
        "dual_contact_segments": sum(
            1 for seg in combo_info["segment_records"] if seg["dual_contact_trigger"]
        ),
        "two_sided_blue_segments": sum(
            1 for seg in combo_info["segment_records"] if seg["two_sided_blue"]
        ),
        "extra_blue_vs_one_sided": extra_blue_vs_one_sided,
        "line_color": line_color,
        "trigger_count": combo_info["trigger_count"],
    }


def _place_random_path(
    H: int,
    W: int,
    seg_len: int,
    *,
    slope_type: str,
) -> list[tuple[int, int]] | None:
    if slope_type == "backslash":
        if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
            return None
        r0 = random.randint(0, H - 1 - seg_len)
        c0 = random.randint(0, W - 1 - seg_len)
        return [(r0 + k, c0 + k) for k in range(seg_len + 1)]
    if H - 1 - seg_len < 0 or seg_len > W - 1:
        return None
    r0 = random.randint(0, H - 1 - seg_len)
    c0 = random.randint(seg_len, W - 1)
    return [(r0 + k, c0 - k) for k in range(seg_len + 1)]


def _place_free_segment(
    H: int,
    W: int,
    reserved_cells: set[tuple[int, int]],
    used_diagonals: set[tuple[str, int]],
    *,
    allowed_cells: set[tuple[int, int]] | None = None,
    even_length_only: bool = False,
    length_range: tuple[int, int] = _SEG_LEN_RANGE,
) -> list[tuple[int, int]] | None:
    max_seg_len = min(H - 1, W - 1, length_range[1])
    min_seg_len = length_range[0]
    if max_seg_len < min_seg_len:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(min_seg_len, max_seg_len)
        if even_length_only and seg_len % 2 == 1:
            continue
        path = _place_random_path(H, W, seg_len, slope_type=slope_type)
        if path is None:
            continue
        diag_key = _segment_key(path)
        if diag_key in used_diagonals:
            continue
        if any(rc in reserved_cells for rc in path):
            continue
        if allowed_cells is not None and any(rc not in allowed_cells for rc in path):
            continue
        return path
    return None


def _place_interacting_pair(H: int, W: int) -> list[dict] | None:
    for _ in range(_PAIR_PLACE_RETRIES):
        emitter_path = _place_free_segment(
            H,
            W,
            set(),
            set(),
            even_length_only=True,
            length_range=_MAIN_SEG_LEN_RANGE,
        )
        if emitter_path is None:
            continue

        trigger = emitter_path[len(emitter_path) // 2]
        perp_dirs = _perp_dirs_for_path(emitter_path)
        random.shuffle(perp_dirs)

        barrier_path = None
        for ray_dir in perp_dirs:
            ray_cells = _cells_along_ray(trigger, ray_dir, H, W)
            if len(ray_cells) < _SEG_LEN_RANGE[0] + 2:
                continue

            start_min = 1
            start_max = min(3, len(ray_cells) - (_SEG_LEN_RANGE[0] + 1))
            if start_max < start_min:
                continue
            start_idx = random.randint(start_min, start_max)

            remaining = len(ray_cells) - start_idx
            max_len = min(_SEG_LEN_RANGE[1], remaining - 1)
            if max_len < _SEG_LEN_RANGE[0]:
                continue

            seg_len = random.randint(_SEG_LEN_RANGE[0], max_len)
            candidate = sorted(ray_cells[start_idx : start_idx + seg_len + 1])
            if trigger in candidate:
                continue
            if any(rc in emitter_path for rc in candidate):
                continue
            barrier_path = candidate
            break

        if barrier_path is None:
            continue

        segments = [
            {"path": emitter_path, "triggers": [trigger]},
            {"path": barrier_path, "triggers": []},
        ]
        inp = _build_input(H, W, 0, segments)
        metrics = _variant_metrics(inp)
        if (
            metrics["line_color"] == _LINE_COLOR
            and metrics["trigger_count"] >= 1
            and metrics["dual_contact_segments"] >= 1
            and metrics["two_sided_blue_segments"] >= 1
            and metrics["extra_blue_vs_one_sided"] >= 2
            and metrics["blocked_rays"] >= 1
            and metrics["blocked_by_line"] >= 1
            and metrics["shortened_rays"] >= 1
            and metrics["differs_from_one_sided"]
            and metrics["differs_from_no_collision"]
        ):
            return segments
    return None


def _place_extra_segments(
    H: int,
    W: int,
    bg: int,
    base_segments: list[dict],
    total_segments: int,
) -> list[dict]:
    segments = [{"path": list(seg["path"]), "triggers": list(seg["triggers"])} for seg in base_segments]
    reserved_cells = set()
    used_diagonals = set()
    for seg in segments:
        reserved_cells.update(seg["path"])
        used_diagonals.add(_segment_key(seg["path"]))

    while len(segments) < total_segments:
        inp = _build_input(H, W, bg, segments)
        metrics = _variant_metrics(inp)
        safe_cells = {
            (r, c)
            for r in range(H)
            for c in range(W)
            if int(inp[r, c]) == bg
            and int(metrics["combo_out"][r, c]) == bg
            and int(metrics["one_sided_out"][r, c]) == bg
            and int(metrics["no_collision_out"][r, c]) == bg
        }
        path = _place_free_segment(
            H,
            W,
            reserved_cells,
            used_diagonals,
            allowed_cells=safe_cells,
        )
        if path is None:
            break
        segments.append({"path": path, "triggers": []})
        reserved_cells.update(path)
        used_diagonals.add(_segment_key(path))
    return segments


def _has_diagonal_partner(
    cell: tuple[int, int], others: list[tuple[int, int]] | set[tuple[int, int]]
) -> bool:
    r, c = cell
    return any((r, c) != (rr, cc) and abs(r - rr) == abs(c - cc) for rr, cc in others)


def _safe_distractor_cells(
    inp: np.ndarray,
    metrics: dict,
    reserved_paths: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    H, W = inp.shape
    bg = get_bg(inp)
    cells = []
    for r in range(H):
        for c in range(W):
            if (r, c) in reserved_paths or int(inp[r, c]) != bg:
                continue
            if int(metrics["combo_out"][r, c]) != bg:
                continue
            if int(metrics["one_sided_out"][r, c]) != bg:
                continue
            if int(metrics["no_collision_out"][r, c]) != bg:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    base_segments = _place_interacting_pair(H, W)
    if base_segments is None:
        return None

    total_segments = random.randint(*_SEGMENT_COUNT)
    segments = _place_extra_segments(H, W, bg, base_segments, total_segments)

    inp = _build_input(H, W, bg, segments)
    metrics = _variant_metrics(inp)
    if not (
        metrics["line_color"] == _LINE_COLOR
        and metrics["trigger_count"] >= 1
        and metrics["dual_contact_segments"] >= 1
        and metrics["two_sided_blue_segments"] >= 1
        and metrics["extra_blue_vs_one_sided"] >= 2
        and metrics["blocked_rays"] >= 1
        and metrics["blocked_by_line"] >= 1
        and metrics["shortened_rays"] >= 1
        and metrics["differs_from_one_sided"]
        and metrics["differs_from_no_collision"]
    ):
        return None

    reserved_paths = set()
    for seg in segments:
        reserved_paths.update(seg["path"])

    safe_cells = _safe_distractor_cells(inp, metrics, reserved_paths)
    line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    trigger_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        placed = False
        for r, c in safe_cells:
            if int(inp[r, c]) != bg or _has_diagonal_partner((r, c), line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            line_positions.append((r, c))
            placed = True
            break
        if not placed:
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        placed = False
        for r, c in safe_cells:
            if int(inp[r, c]) != bg:
                continue
            if _has_diagonal_partner((r, c), trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            trigger_positions.append((r, c))
            placed = True
            break
        if not placed:
            break

    if int(np.count_nonzero(inp != bg)) > 18:
        return None

    final_metrics = _variant_metrics(inp)
    if not (
        final_metrics["line_color"] == _LINE_COLOR
        and final_metrics["trigger_count"] >= 1
        and final_metrics["dual_contact_segments"] >= 1
        and final_metrics["two_sided_blue_segments"] >= 1
        and final_metrics["extra_blue_vs_one_sided"] >= 2
        and final_metrics["blocked_rays"] >= 1
        and final_metrics["blocked_by_line"] >= 1
        and final_metrics["shortened_rays"] >= 1
        and final_metrics["differs_from_one_sided"]
        and final_metrics["differs_from_no_collision"]
    ):
        return None

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, final_metrics["combo_out"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
