# Variant of db695cfb
# Dimension: D3 — 延伸终止条件 (dst)
# Change: 延伸至网格边界——射线从品红像素出发一直画到网格边缘才停止... → 遇到已有像素停止

"""
Design analysis for the D3 collision-stop variant.

variant:
  core_change:
    original: rays ignore non-background cells as obstacles; they simply skip over them and keep going.
    new: each ray advances cell-by-cell and halts immediately before the first non-background cell on that diagonal.
    consequence: the final magenta footprint depends on already-drawn blue diagonals and, in sequential cases, on earlier magenta rays.

  behavior_trace:
    concrete_layout:
      - a blue "\" segment contains a magenta trigger.
      - a blue "/" segment is placed on the trigger's perpendicular diagonal, but only on one side of the trigger.
    original_execution:
      - the perpendicular ray paints background cells until the border.
      - when it reaches the blue obstacle cell it does not overwrite it, but it continues beyond it, so cells behind the obstacle also become magenta.
    variant_execution:
      - the ray paints background cells up to the cell just before that blue obstacle.
      - at the first non-background cell it stops entirely, so nothing behind the obstacle is painted.
    visible_difference:
      - the old output shows a full ray reaching the edge.
      - the new output shows a truncated ray with a clear gap beyond the blocking blue diagonal.

  complexity_calibration:
    - The solver still uses the original "find the line color, fill the extreme same-diagonal endpoints, then launch perpendicular rays" structure.
    - To make D3 visible in 2-4 examples, the generator now guarantees at least one interacting pair of segments and at least one trigger.
    - Extra segments are allowed, but they are kept triggerless so the new stop-on-collision effect stays readable instead of turning into dense beam clutter.

  implicit_assumption_audit:
    - main() still assumes some non-background color has a diagonal pair; generator guarantees at least two blue segments.
    - main() chooses the line color via the longest diagonal-aligned same-color pair; generator checks that blue remains the chosen line color.
    - main() connects extremes of every used diagonal, so the generator forbids multiple blue segments from sharing a diagonal id.
    - With collision-stop, ray order can matter because earlier rays can become obstacles; helper simulation and validation mirror solver order exactly.
    - Distractors were harmless in the original task because rays passed through them; here they could accidentally become blockers, so the generator only places distractors on cells that stay background in both the old and new semantics.

  placement_legality:
    - Segment paths never overlap each other.
    - Trigger pixels sit strictly inside their own blue segment.
    - The guaranteed barrier segment lies on the trigger's perpendicular diagonal but excludes the trigger cell itself, so the ray collides with the barrier rather than with an invalid overlap.
    - Everything stays in-bounds, and the total number of visible input pixels stays small enough for background to remain dominant.

  visual_observability:
    - The generator accepts only inputs where the collision-stop output differs from the old boundary-reaching output.
    - It also requires at least one ray to be shortened by the new rule, so the variant does not silently degenerate back to the original task.
    - Because blockers are blue segments from the same visual vocabulary as the original task, the changed behavior is legible without introducing a new object type.

  cleanliness:
    - The interacting pair is isolated first, then extra structure is added only in untouched background regions.
    - This avoids ambiguous multi-layer crossings while still preserving the original task's mixed-segment look.
"""

import random

import numpy as np

from arc_utils import get_bg


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
    best = None  # (dist, color)
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


def _solve_with_policy(
    grid: np.ndarray, *, stop_on_collision: bool
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
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], list[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def remember_trigger(
        key: tuple[int, int, int], dirs: list[tuple[int, int]]
    ) -> None:
        bucket = triggers.setdefault(key, [])
        for d in dirs:
            if d not in bucket:
                bucket.append(d)

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                remember_trigger((r, c, v), perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    ray_records = []
    for (r0, c0, tcol), dirs in triggers.items():
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

    info = {
        "bg": bg,
        "line_color": line_color,
        "trigger_count": len(triggers),
        "ray_records": ray_records,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill same-color diagonal endpoint pairs, then grow perpendicular trigger rays that stop at the first obstacle.

    Input structure:
    - The background is the most common color. One foreground color supplies endpoint pairs on shared diagonals, and differently-colored trigger pixels may lie on those diagonals.

    Rule:
    - For each diagonal family used by the endpoint color, connect the extreme pixels with a solid blue diagonal.
    - Whenever that completed diagonal passes through a trigger pixel, launch the two perpendicular diagonal rays in the trigger color.
    - Each ray paints only background cells and halts as soon as the next cell is already occupied by any non-background pixel.

    Key insight: Triggers are still beam splitters, but the important new constraint is that rays now treat existing structure as hard blockers instead of tunneling through it.
    """

    out, _ = _solve_with_policy(grid, stop_on_collision=True)
    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (2, 4)  # raised floor: at least one emitter segment and one blocking segment
_SEG_LEN_RANGE = (3, 16)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 1)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_PAIR_PLACE_RETRIES = 160
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


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
    variant_out, variant_info = _solve_with_policy(inp, stop_on_collision=True)
    boundary_out, boundary_info = _solve_with_policy(inp, stop_on_collision=False)

    shortened_rays = 0
    for a, b in zip(variant_info["ray_records"], boundary_info["ray_records"]):
        if a["painted"] < b["painted"]:
            shortened_rays += 1

    return {
        "variant_out": variant_out,
        "boundary_out": boundary_out,
        "variant_info": variant_info,
        "boundary_info": boundary_info,
        "blocked_rays": sum(1 for r in variant_info["ray_records"] if r["blocked"]),
        "blocked_by_line": sum(1 for r in variant_info["ray_records"] if r["block_kind"] == "line"),
        "blocked_by_trigger": sum(1 for r in variant_info["ray_records"] if r["block_kind"] == "trigger"),
        "shortened_rays": shortened_rays,
        "differs_from_boundary": not np.array_equal(variant_out, boundary_out),
        "line_color": variant_info["line_color"],
        "trigger_count": variant_info["trigger_count"],
    }


def _place_free_segment(
    H: int,
    W: int,
    reserved_paths: set[tuple[int, int]],
    used_diagonals: set[tuple[str, int]],
    allowed_cells: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)

        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                continue
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            path = [(r0 + k, c0 + k) for k in range(seg_len + 1)]
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                continue
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            path = [(r0 + k, c0 - k) for k in range(seg_len + 1)]

        diag_key = _segment_key(path)
        if diag_key in used_diagonals:
            continue
        if any(rc in reserved_paths for rc in path):
            continue
        if allowed_cells is not None and any(rc not in allowed_cells for rc in path):
            continue
        return path
    return None


def _place_interacting_pair(H: int, W: int) -> list[dict] | None:
    reserved_paths: set[tuple[int, int]] = set()
    used_diagonals: set[tuple[str, int]] = set()

    for _ in range(_PAIR_PLACE_RETRIES):
        emitter_path = _place_free_segment(H, W, reserved_paths, used_diagonals)
        if emitter_path is None:
            continue

        emitter_interior = emitter_path[1:-1]
        if not emitter_interior:
            continue
        trigger = random.choice(emitter_interior)

        ray_dirs = _perp_dirs_for_path(emitter_path)
        random.shuffle(ray_dirs)
        barrier_path = None
        for ray_dir in ray_dirs:
            cells = _cells_along_ray(trigger, ray_dir, H, W)
            if len(cells) < _SEG_LEN_RANGE[0] + 1:
                continue
            start_max = min(2, len(cells) - (_SEG_LEN_RANGE[0] + 1))
            if start_max < 0:
                continue
            start_idx = random.randint(0, start_max)
            remaining = len(cells) - start_idx
            max_len = min(_SEG_LEN_RANGE[1], remaining - 1)
            if max_len < _SEG_LEN_RANGE[0]:
                continue
            seg_len = random.randint(_SEG_LEN_RANGE[0], max_len)
            candidate = sorted(cells[start_idx : start_idx + seg_len + 1])
            diag_key = _segment_key(candidate)
            if diag_key in used_diagonals:
                continue
            if any(rc in reserved_paths for rc in candidate):
                continue
            barrier_path = candidate
            break

        if barrier_path is None:
            continue

        emitter_triggers = [trigger]
        extra_emitter_slots = [p for p in emitter_interior if p != trigger]
        if extra_emitter_slots and random.random() < 0.5:
            emitter_triggers.append(random.choice(extra_emitter_slots))

        barrier_triggers = []
        barrier_interior = barrier_path[1:-1]
        if len(barrier_interior) >= 2 and random.random() < 0.4:
            barrier_options = [p for p in barrier_interior if p != barrier_path[1]]
            if barrier_options:
                barrier_triggers.append(random.choice(barrier_options))

        segments = [
            {"path": emitter_path, "triggers": emitter_triggers},
            {"path": barrier_path, "triggers": barrier_triggers},
        ]

        reserved_paths = set(emitter_path) | set(barrier_path)
        used_diagonals = {_segment_key(emitter_path), _segment_key(barrier_path)}

        inp = _build_input(H, W, 0, segments)
        inp[inp == 0] = 0
        metrics = _variant_metrics(inp)
        if (
            metrics["line_color"] == _LINE_COLOR
            and metrics["trigger_count"] >= 1
            and metrics["blocked_rays"] >= 1
            and metrics["blocked_by_line"] >= 1
            and metrics["shortened_rays"] >= 1
            and metrics["differs_from_boundary"]
        ):
            return segments
    return None


def _place_extra_segments(
    H: int,
    W: int,
    bg: int,
    base_segments: list[dict],
    total_segments: int,
) -> list[dict] | None:
    segments = [{"path": list(seg["path"]), "triggers": list(seg["triggers"])} for seg in base_segments]
    reserved_paths = set()
    used_diagonals = set()
    for seg in segments:
        reserved_paths.update(seg["path"])
        used_diagonals.add(_segment_key(seg["path"]))

    base_inp = _build_input(H, W, bg, segments)
    base_metrics = _variant_metrics(base_inp)
    safe_cells = {
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(base_inp[r, c]) == bg
        and int(base_metrics["variant_out"][r, c]) == bg
        and int(base_metrics["boundary_out"][r, c]) == bg
    }

    for _ in range(total_segments - len(segments)):
        path = _place_free_segment(H, W, reserved_paths, used_diagonals, safe_cells)
        if path is None:
            break
        seg = {"path": path, "triggers": []}
        segments.append(seg)
        reserved_paths.update(path)
        used_diagonals.add(_segment_key(path))
        for rc in path:
            safe_cells.discard(rc)

    return segments


def _safe_distractor_cells(inp: np.ndarray, metrics: dict) -> list[tuple[int, int]]:
    H, W = inp.shape
    bg = get_bg(inp)
    cells = []
    for r in range(H):
        for c in range(W):
            if int(inp[r, c]) != bg:
                continue
            if int(metrics["variant_out"][r, c]) != bg:
                continue
            if int(metrics["boundary_out"][r, c]) != bg:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _has_diagonal_partner(cell: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) == abs(c - cc) and (r, c) != (rr, cc) for rr, cc in others)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    base_segments = _place_interacting_pair(H, W)
    if base_segments is None:
        return None

    total_segments = random.randint(*_SEGMENT_COUNT)
    segments = _place_extra_segments(H, W, bg, base_segments, total_segments)
    if segments is None:
        return None

    inp = _build_input(H, W, bg, segments)
    metrics = _variant_metrics(inp)
    if not (
        metrics["line_color"] == _LINE_COLOR
        and metrics["trigger_count"] >= 1
        and metrics["blocked_rays"] >= 1
        and metrics["blocked_by_line"] >= 1
        and metrics["shortened_rays"] >= 1
        and metrics["differs_from_boundary"]
    ):
        return None

    safe_cells = _safe_distractor_cells(inp, metrics)
    line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    trigger_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))

    line_to_add = random.randint(*_UNPAIRED_LINE_PIXELS)
    for _ in range(line_to_add):
        placed = False
        for r, c in safe_cells:
            if int(inp[r, c]) != bg:
                continue
            if _has_diagonal_partner((r, c), line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            line_positions.append((r, c))
            placed = True
            break
        if not placed:
            break

    trig_to_add = random.randint(*_OFF_DIAG_TRIGGERS)
    for _ in range(trig_to_add):
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

    total_non_bg = int(np.count_nonzero(inp != bg))
    if total_non_bg > 18:
        return None

    final_metrics = _variant_metrics(inp)
    if not (
        final_metrics["line_color"] == _LINE_COLOR
        and final_metrics["trigger_count"] >= 1
        and final_metrics["blocked_rays"] >= 1
        and final_metrics["blocked_by_line"] >= 1
        and final_metrics["shortened_rays"] >= 1
        and final_metrics["differs_from_boundary"]
    ):
        return None

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, final_metrics["variant_out"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
