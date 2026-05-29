# Combo variant of db695cfb (index 2)
# combo_spec: D1/2+D3/0
# Name: 阶梯路径+碰撞停止

"""
Design analysis for the D1/2 + D3/0 combo variant.

variant:
  core_change:
    original:
      - Blue endpoints on one 45-degree diagonal are joined by a thin diagonal line.
      - Magenta triggers encountered on that line emit the perpendicular diagonal rays, and those rays keep going to the border.
    combo:
      - Each endpoint pair is joined by a canonical one-cell-wide staircase: every unit of diagonal progress is realized as one horizontal step followed by one vertical step.
      - Trigger detection follows that staircase footprint rather than the thin diagonal.
      - Trigger rays still travel on the perpendicular diagonal family, but each ray now halts immediately before the first non-background obstacle.

  behavior_trace:
    concrete_layout:
      - Suppose blue endpoints are at (1, 1) and (4, 4).
      - The old path would be (1,1),(2,2),(3,3),(4,4).
      - The new staircase is (1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4).
      - A magenta trigger at (2,3) is therefore active only in the combo path policy, because that cell is staircase-only and is not on the old thin diagonal.
      - If another blue segment occupies cells on the trigger's perpendicular slash-diagonal, the combo ray paints background cells until the first blue obstacle and then stops.
    visible_difference:
      - Compared with the original task, the output now shows a chunkier zigzag connector and can activate magenta pixels that sit one cell off the thin diagonal.
      - Compared with the D1-only staircase version, the combo additionally produces visibly truncated magenta rays instead of full border-reaching rays.

  complexity_calibration:
    - The staircase orientation is fixed and deterministic: after sorting endpoints from top to bottom, every diagonal unit expands as horizontal-then-vertical.
    - The generator guarantees at least one staircase-only trigger, so the path change is directly observable rather than hidden inside cells that would also belong to the old diagonal.
    - The generator also guarantees at least one blocking interaction, so collision-stop is not a silent no-op.
    - Extra segments are allowed, but they are kept triggerless and placed in cells that stay background under all baseline policies so the key combo remains legible.

  implicit_assumption_audit:
    - Line-color detection still assumes some non-background color has a same-color diagonal endpoint pair; the generator guarantees that blue remains the longest such pair family.
    - Connecting extremes of each used diagonal still assumes there is at most one intended segment per diagonal id; the generator forbids repeated blue diagonal ids.
    - Trigger detection now depends on the exact staircase footprint, so off-path magenta distractors must avoid every generated staircase cell; the generator reserves those cells.
    - Collision-stop makes ray order observable when rays can hit earlier drawn rays; to avoid unstable clutter, the guaranteed interaction uses a blue blocking segment, and validation mirrors solver order exactly.
    - Because the staircase set depends on endpoint ordering, both solver and generator normalize endpoints the same way: sorted by row, then column, before constructing the zigzag.

  placement_legality:
    - Every segment stays in-bounds on both its thin diagonal and its staircase footprint.
    - Staircase footprints of distinct segments never overlap.
    - The guaranteed blocking segment lies on one side of the emitter trigger's perpendicular diagonal, so it blocks a ray without sharing the trigger cell.
    - Distractors are placed only on cells that remain untouched in all simulated policies, preventing accidental triggers or blockers.

  visual_observability:
    - Acceptance requires all three comparisons to be non-degenerate on the same input:
      - staircase+boundary differs from diagonal+boundary, so D1 is visible;
      - staircase+collision differs from staircase+boundary, so D3 is visible;
      - staircase+collision differs from diagonal+collision, so the fused task is not reducible to collision-stop alone.
    - The main trigger is staircase-only, which gives the human solver a direct positional cue for why that magenta pixel matters.

  cleanliness:
    - The scene is built around one emitter segment and one blocker segment, then any extra blue segments are added only in untouched regions.
    - Trigger count is kept low so the output does not devolve into overlapping magenta beams.

  inferability:
    - Colors keep their original roles: blue still marks the segment family and magenta still marks ray sources/rays.
    - No hidden ordering by color id is introduced; all distinctions are geometric and repeated across generated examples.
"""

import random

import numpy as np

from arc_utils import get_bg


def _normalize_endpoints(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = sorted((p, q))
    return a, b


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    p, q = _normalize_endpoints(p, q)
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _stair_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    p, q = _normalize_endpoints(p, q)
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    path = [(r0, c0)]
    for k in range(1, steps + 1):
        path.append((r0 + (k - 1) * dr, c0 + k * dc))
        path.append((r0 + k * dr, c0 + k * dc))
    return path


def _segment_kind(p: tuple[int, int], q: tuple[int, int]) -> str:
    p, q = _normalize_endpoints(p, q)
    return "backslash" if (q[0] - p[0]) * (q[1] - p[1]) > 0 else "slash"


def _segment_key(p: tuple[int, int], q: tuple[int, int]) -> tuple[str, int]:
    p, q = _normalize_endpoints(p, q)
    kind = _segment_kind(p, q)
    if kind == "backslash":
        return kind, p[0] - p[1]
    return kind, p[0] + p[1]


def _perp_dirs_for_segment(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    p, q = _normalize_endpoints(p, q)
    dr = 1 if q[0] > p[0] else -1
    dc = 1 if q[1] > p[1] else -1
    return [(dr, -dc), (-dr, dc)]


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
    grid: np.ndarray, *, path_mode: str, stop_on_collision: bool
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

    def remember_trigger(key: tuple[int, int, int], dirs: list[tuple[int, int]]) -> None:
        bucket = triggers.setdefault(key, [])
        for d in dirs:
            if d not in bucket:
                bucket.append(d)

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        p, q = _normalize_endpoints(p, q)
        path = _stair_path(p, q) if path_mode == "stair" else _diag_path(p, q)
        perp_dirs = _perp_dirs_for_segment(p, q)
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
                        record["block_kind"] = "trigger_or_ray"
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
    Reconstruct diagonal endpoint pairs as staircase connectors, then emit obstacle-sensitive perpendicular rays from any trigger the staircase touches.

    Input structure:
    - The most common color is background. One foreground color marks endpoint pairs that still lie on shared 45-degree diagonals, while differently-colored trigger pixels may sit either on those diagonals or on the staircase-only offsets between them.

    Rule:
    - For every used diagonal of the endpoint color, connect its extreme pixels with the fixed zigzag path obtained by alternating one horizontal move and one vertical move for each diagonal step.
    - Whenever that staircase passes through a non-endpoint colored pixel, launch the two perpendicular diagonal rays in the trigger's color.
    - Each ray paints background cells only and stops immediately when the next cell is any non-background color.

    Key insight: The important structure is no longer the thin diagonal but the canonical staircase footprint built from the diagonal endpoints; those wider blue paths both activate extra triggers and act as blockers that can truncate magenta rays.
    """

    out, _ = _solve_with_policy(grid, path_mode="stair", stop_on_collision=True)
    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 10)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_PATH_TRIGGERS = (0, 1)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_PAIR_PLACE_RETRIES = 240
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _make_segment(
    p: tuple[int, int], q: tuple[int, int], triggers: list[tuple[int, int]] | None = None
) -> dict:
    p, q = _normalize_endpoints(p, q)
    return {
        "endpoints": (p, q),
        "diag_path": _diag_path(p, q),
        "stair_path": _stair_path(p, q),
        "key": _segment_key(p, q),
        "perp_dirs": _perp_dirs_for_segment(p, q),
        "triggers": list(triggers or []),
    }


def _build_input(H: int, W: int, bg: int, segments: list[dict]) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        p, q = seg["endpoints"]
        inp[p] = _LINE_COLOR
        inp[q] = _LINE_COLOR
        for tp in seg["triggers"]:
            inp[tp] = _TRIGGER_COLOR
    return inp


def _variant_metrics(inp: np.ndarray) -> dict:
    combo_out, combo_info = _solve_with_policy(inp, path_mode="stair", stop_on_collision=True)
    stair_boundary_out, stair_boundary_info = _solve_with_policy(inp, path_mode="stair", stop_on_collision=False)
    diag_collision_out, diag_collision_info = _solve_with_policy(inp, path_mode="diag", stop_on_collision=True)
    diag_boundary_out, diag_boundary_info = _solve_with_policy(inp, path_mode="diag", stop_on_collision=False)

    shortened_rays = 0
    for combo_rec, boundary_rec in zip(combo_info["ray_records"], stair_boundary_info["ray_records"]):
        if combo_rec["painted"] < boundary_rec["painted"]:
            shortened_rays += 1

    return {
        "combo_out": combo_out,
        "combo_info": combo_info,
        "stair_boundary_out": stair_boundary_out,
        "stair_boundary_info": stair_boundary_info,
        "diag_collision_out": diag_collision_out,
        "diag_collision_info": diag_collision_info,
        "diag_boundary_out": diag_boundary_out,
        "diag_boundary_info": diag_boundary_info,
        "blocked_rays": sum(1 for r in combo_info["ray_records"] if r["blocked"]),
        "blocked_by_line": sum(1 for r in combo_info["ray_records"] if r["block_kind"] == "line"),
        "shortened_rays": shortened_rays,
        "stair_trigger_gain": stair_boundary_info["trigger_count"] - diag_boundary_info["trigger_count"],
        "path_diff_cells": int(np.count_nonzero(stair_boundary_out != diag_boundary_out)),
        "collision_diff_cells": int(np.count_nonzero(combo_out != stair_boundary_out)),
        "combo_vs_diag_collision_cells": int(np.count_nonzero(combo_out != diag_collision_out)),
        "combo_vs_original_cells": int(np.count_nonzero(combo_out != diag_boundary_out)),
        "path_changed": not np.array_equal(stair_boundary_out, diag_boundary_out),
        "collision_changed": not np.array_equal(combo_out, stair_boundary_out),
        "combo_diff_from_diag_collision": not np.array_equal(combo_out, diag_collision_out),
        "line_color": combo_info["line_color"],
        "combo_trigger_count": combo_info["trigger_count"],
        "stair_trigger_count": stair_boundary_info["trigger_count"],
        "diag_trigger_count": diag_boundary_info["trigger_count"],
    }


def _place_free_segment(
    H: int,
    W: int,
    reserved_stairs: set[tuple[int, int]],
    used_diagonals: set[tuple[str, int]],
    allowed_cells: set[tuple[int, int]] | None = None,
) -> dict | None:
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
            p, q = (r0, c0), (r0 + seg_len, c0 + seg_len)
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                continue
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            p, q = (r0, c0), (r0 + seg_len, c0 - seg_len)

        seg = _make_segment(p, q)
        if seg["key"] in used_diagonals:
            continue
        stair_cells = set(seg["stair_path"])
        if stair_cells & reserved_stairs:
            continue
        if allowed_cells is not None and any(rc not in allowed_cells for rc in stair_cells):
            continue
        return seg
    return None


def _stair_only_trigger_cells(seg: dict) -> list[tuple[int, int]]:
    diag_cells = set(seg["diag_path"])
    return [rc for rc in seg["stair_path"][1:-1] if rc not in diag_cells]


def _place_interacting_pair(H: int, W: int) -> list[dict] | None:
    for _ in range(_PAIR_PLACE_RETRIES):
        emitter = _place_free_segment(H, W, set(), set())
        if emitter is None:
            continue

        trigger_candidates = _stair_only_trigger_cells(emitter)
        if not trigger_candidates:
            continue
        trigger = random.choice(trigger_candidates)
        emitter["triggers"] = [trigger]

        reserved_stairs = set(emitter["stair_path"])
        used_diagonals = {emitter["key"]}

        ray_dirs = list(emitter["perp_dirs"])
        random.shuffle(ray_dirs)
        barrier = None
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
            p = cells[start_idx]
            q = cells[start_idx + seg_len]
            candidate = _make_segment(p, q)
            if candidate["key"] in used_diagonals:
                continue
            if set(candidate["stair_path"]) & reserved_stairs:
                continue
            barrier = candidate
            break

        if barrier is None:
            continue

        segments = [emitter, barrier]
        inp = _build_input(H, W, 0, segments)
        metrics = _variant_metrics(inp)
        if (
            metrics["line_color"] == _LINE_COLOR
            and metrics["stair_trigger_gain"] >= 1
            and metrics["blocked_rays"] >= 1
            and metrics["blocked_by_line"] >= 1
            and metrics["shortened_rays"] >= 1
            and metrics["path_diff_cells"] >= 3
            and metrics["collision_diff_cells"] >= 1
            and metrics["combo_vs_diag_collision_cells"] >= 1
            and metrics["path_changed"]
            and metrics["collision_changed"]
            and metrics["combo_diff_from_diag_collision"]
        ):
            return segments
    return None


def _safe_background_cells(inp: np.ndarray, outputs: list[np.ndarray]) -> set[tuple[int, int]]:
    H, W = inp.shape
    bg = get_bg(inp)
    cells = set()
    for r in range(H):
        for c in range(W):
            if int(inp[r, c]) != bg:
                continue
            if all(int(out[r, c]) == bg for out in outputs):
                cells.add((r, c))
    return cells


def _place_extra_segments(H: int, W: int, bg: int, base_segments: list[dict], total_segments: int) -> list[dict]:
    segments = []
    reserved_stairs: set[tuple[int, int]] = set()
    used_diagonals: set[tuple[str, int]] = set()
    for seg in base_segments:
        copied = _make_segment(*seg["endpoints"], triggers=list(seg["triggers"]))
        segments.append(copied)
        reserved_stairs.update(copied["stair_path"])
        used_diagonals.add(copied["key"])

    base_inp = _build_input(H, W, bg, segments)
    base_metrics = _variant_metrics(base_inp)
    safe_cells = _safe_background_cells(
        base_inp,
        [
            base_metrics["combo_out"],
            base_metrics["stair_boundary_out"],
            base_metrics["diag_collision_out"],
            base_metrics["diag_boundary_out"],
        ],
    )

    for _ in range(total_segments - len(segments)):
        seg = _place_free_segment(H, W, reserved_stairs, used_diagonals, safe_cells)
        if seg is None:
            break
        segments.append(seg)
        reserved_stairs.update(seg["stair_path"])
        used_diagonals.add(seg["key"])
        for rc in seg["stair_path"]:
            safe_cells.discard(rc)

    return segments


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

    inp = _build_input(H, W, bg, segments)
    metrics = _variant_metrics(inp)
    if not (
        metrics["line_color"] == _LINE_COLOR
        and metrics["stair_trigger_gain"] >= 1
        and metrics["blocked_rays"] >= 1
        and metrics["blocked_by_line"] >= 1
        and metrics["shortened_rays"] >= 1
        and metrics["path_diff_cells"] >= 3
        and metrics["collision_diff_cells"] >= 1
        and metrics["combo_vs_diag_collision_cells"] >= 1
        and metrics["path_changed"]
        and metrics["collision_changed"]
        and metrics["combo_diff_from_diag_collision"]
    ):
        return None

    safe_cells = list(
        _safe_background_cells(
            inp,
            [
                metrics["combo_out"],
                metrics["stair_boundary_out"],
                metrics["diag_collision_out"],
                metrics["diag_boundary_out"],
            ],
        )
    )
    random.shuffle(safe_cells)

    line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    trigger_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
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

    for _ in range(random.randint(*_OFF_PATH_TRIGGERS)):
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
        and final_metrics["stair_trigger_gain"] >= 1
        and final_metrics["blocked_rays"] >= 1
        and final_metrics["blocked_by_line"] >= 1
        and final_metrics["shortened_rays"] >= 1
        and final_metrics["path_diff_cells"] >= 3
        and final_metrics["collision_diff_cells"] >= 1
        and final_metrics["combo_vs_diag_collision_cells"] >= 1
        and final_metrics["path_changed"]
        and final_metrics["collision_changed"]
        and final_metrics["combo_diff_from_diag_collision"]
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
