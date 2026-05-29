# Combo variant of db695cfb (index 13)
# combo_spec: D2/1+D3/0+D14/0
# Name: 正交延伸+碰撞停止+交叉标色

import random

import numpy as np

from arc_utils import get_bg


"""
Design analysis for the combo variant D2/1 + D3/0 + D14/0.

fusion_summary:
  - Keep the original blue "connect the diagonal endpoints" backbone.
  - Replace the old perpendicular-diagonal trigger branches with orthogonal branches:
    each on-backbone trigger emits up, down, left, and right.
  - Unlike the original task, each orthogonal branch now stops at the first occupied cell.
  - If that first occupied cell is a completed blue diagonal cell, recolor that collision
    cell with a fixed third color (red) and stop there.

behavior_trace:
  concrete_example:
    - Blue endpoints at (1,1) and (5,5) create a "\" backbone.
    - A magenta trigger sits at (3,3).
    - Another blue diagonal crosses row 3 at (3,6).
  original_task:
    - Fill the "\" diagonal.
    - From (3,3), emit the two perpendicular diagonal rays until the border.
    - A later diagonal/ray overlap is only an incidental draw-order effect.
  fused_variant:
    - Fill the "\" diagonal exactly as before.
    - From (3,3), emit four orthogonal rays: row 3 left/right and column 3 up/down.
    - The rightward ray paints (3,4),(3,5), reaches blue cell (3,6), turns that single
      cell red, and stops immediately; nothing beyond (3,6) is painted.
  visible_difference:
    - D2/1 changes the branch geometry from diagonal to axis-aligned.
    - D3/0 truncates the branch at the first obstacle instead of tunneling past it.
    - D14/0 makes that terminating line-branch collision explicit with a red anchor.

complexity_calibration:
  - Orthogonal rays occupy much more area than diagonal rays, so the generator is tightened:
    at least one true interaction pair is mandatory, but extra segments are triggerless and
    placed only in safe background regions.
  - Using a fixed red collision color keeps the added hypothesis small and human-learnable.
  - One mandatory real trigger per puzzle is enough to make the fused behavior obvious,
    while avoiding dense ray-ray interference that would make the samples messy.

implicit_assumption_audit_for_main:
  - Longest same-color diagonal pair detection still assumes one color clearly owns the
    connector role. The generator enforces that blue is the only color with a meaningful
    diagonal pair and that trigger distractors never form such pairs.
  - The original solver wrote blue directly into the output and later let rays fill only
    background. That no longer works because line-ray collisions must be detectable as
    structure, not just as "already painted" cells.
  - The fused solver therefore separates:
      1. collecting all blue diagonal cells implied by blue endpoints
      2. collecting trigger cells lying on those diagonals
      3. rendering orthogonal rays against that fixed blue-cell set
  - Collision-stop introduces order sensitivity if rays can hit earlier rays. To avoid
    accidental semantics, the solver uses a deterministic trigger/ray order, and the
    generator avoids layouts where ray-ray collisions are needed to explain the examples.
  - The new "mark the blocker red" rule is only valid when the blocker is a blue diagonal
    cell; hitting any other non-background cell still just stops without recoloring it.

placement_legality:
  - Segment paths are kept on distinct diagonal keys, so the solver's "connect extremes on
    each diagonal" logic matches generation exactly.
  - The mandatory barrier segment passes through a preselected row/column ray cell, with
    the collision cell strictly inside that barrier, not at an endpoint.
  - The generator rejects any candidate whose intended interaction would overlap an input
    pixel, go out of bounds, or fail to produce a genuine shortened branch plus red mark.
  - Extra segments and distractors are only added on cells that remain background under
    both the fused solver and the weaker orthogonal-to-border baseline, so they cannot
    silently alter the forced interaction.

visual_observability:
  - Every accepted sample has:
      * at least one on-diagonal trigger
      * at least one orthogonal branch blocked by a blue diagonal
      * at least one red collision cell
      * an output different from the weaker D2-only semantics
  - These constraints make the three-part fusion legible from a few examples: diagonal
    blue skeleton, magenta axis-aligned growth, and red collision endpoints.

cleanliness_and_inference:
  - Ray-ray crossings are not used as a teaching signal; they are avoided so red can be
    inferred as specifically "orthogonal branch hits blue diagonal".
  - Background never uses colors 1, 2, or 6, so blue / red / magenta remain visually
    distinct and structurally interpretable.
"""


_LINE_COLOR = 1
_CROSS_COLOR = 2
_TRIGGER_COLOR = 6
_ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 16)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 1)

# === CFG: retry limits ===
_PAIR_RETRIES = 220
_SEGMENT_PLACE_RETRIES = 120
_BARRIER_RETRIES = 120
_EXTRA_SEGMENT_RETRIES = 160
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _segment_key(path: list[tuple[int, int]]) -> tuple[str, int]:
    if path[1][1] > path[0][1]:
        return ("backslash", path[0][0] - path[0][1])
    return ("slash", path[0][0] + path[0][1])


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


def _collect_structure(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int]]]:
    h, w = grid.shape
    pts = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    blue_draw_cells: set[tuple[int, int]] = set()
    triggers: set[tuple[int, int, int]] = set()

    def collect_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        for r, c in _diag_path(p, q):
            v = int(grid[r, c])
            if v == bg or v == line_color:
                blue_draw_cells.add((r, c))
            elif v != bg:
                triggers.add((r, c, v))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_segment(g[0], g[-1])

    return blue_draw_cells, sorted(triggers)


def _solve_with_policy(
    grid: np.ndarray, *, stop_on_collision: bool, mark_crossings: bool
) -> tuple[np.ndarray, dict]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = _detect_line_color(by_color)
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    blue_draw_cells, triggers = _collect_structure(grid, bg, line_color)

    out = grid.copy()
    for r, c in sorted(blue_draw_cells):
        if int(out[r, c]) == bg:
            out[r, c] = line_color

    ray_records = []
    for r0, c0, tcol in triggers:
        for ddr, ddc in _ORTHO_DIRS:
            record = {
                "trigger": (r0, c0),
                "dir": (ddr, ddc),
                "painted": 0,
                "blocked": False,
                "block_kind": None,
                "cross_marked": False,
            }
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in blue_draw_cells:
                    record["blocked"] = True
                    record["block_kind"] = "line"
                    if mark_crossings:
                        out[r, c] = _CROSS_COLOR
                        record["cross_marked"] = True
                    if stop_on_collision:
                        break
                    r += ddr
                    c += ddc
                    continue

                v = int(out[r, c])
                if v != bg:
                    record["blocked"] = True
                    if v == tcol:
                        record["block_kind"] = "trigger_or_ray"
                    elif v == _CROSS_COLOR:
                        record["block_kind"] = "cross"
                    else:
                        record["block_kind"] = "other"
                    if stop_on_collision:
                        break
                else:
                    out[r, c] = tcol
                    record["painted"] += 1

                r += ddr
                c += ddc

            ray_records.append(record)

    info = {
        "bg": bg,
        "line_color": line_color,
        "trigger_count": len(triggers),
        "blue_draw_cells": blue_draw_cells,
        "ray_records": ray_records,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal endpoint pairs, then grow obstacle-aware orthogonal trigger branches with marked line collisions.

    Input structure:
    - The most common color is background. One foreground color supplies endpoint pairs on shared 45-degree diagonals, and differently-colored trigger pixels may lie on those diagonal interiors.

    Rule:
    - Connect the extreme same-color endpoints on each occupied diagonal to form the blue backbone.
    - Every non-blue pixel encountered on that backbone emits four orthogonal rays: up, down, left, and right.
    - Each ray paints background cells until the first occupied cell. If that blocker is a completed blue diagonal cell, recolor that collision cell red and stop there; any other non-background blocker just stops the ray.

    Key insight:
    - The puzzle now combines three cues at once: diagonal connectivity identifies the trigger sites, orthogonal growth switches to a different direction system, and the first blue obstacle becomes both the stopping condition and a dedicated red collision marker.
    """

    out, _ = _solve_with_policy(grid, stop_on_collision=True, mark_crossings=True)
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
    variant_out, variant_info = _solve_with_policy(inp, stop_on_collision=True, mark_crossings=True)
    d2_out, d2_info = _solve_with_policy(inp, stop_on_collision=False, mark_crossings=False)
    stop_plain_out, _ = _solve_with_policy(inp, stop_on_collision=True, mark_crossings=False)

    shortened_rays = 0
    for variant_record, d2_record in zip(variant_info["ray_records"], d2_info["ray_records"]):
        if variant_record["painted"] < d2_record["painted"]:
            shortened_rays += 1

    cross_count = int(np.count_nonzero(variant_out == _CROSS_COLOR))
    return {
        "variant_out": variant_out,
        "d2_out": d2_out,
        "stop_plain_out": stop_plain_out,
        "variant_info": variant_info,
        "d2_info": d2_info,
        "line_color": variant_info["line_color"],
        "trigger_count": variant_info["trigger_count"],
        "blocked_rays": sum(1 for r in variant_info["ray_records"] if r["blocked"]),
        "blocked_by_line": sum(1 for r in variant_info["ray_records"] if r["block_kind"] == "line"),
        "shortened_rays": shortened_rays,
        "cross_count": cross_count,
        "differs_from_d2": not np.array_equal(variant_out, d2_out),
        "differs_from_stop_plain": not np.array_equal(variant_out, stop_plain_out),
    }


def _step_limits(H: int, W: int, cell: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = cell
    dr, dc = step
    n = 0
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
        n += 1
    return n


def _cells_along_direction(
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


def _try_place_free_segment(
    H: int,
    W: int,
    blocked: set[tuple[int, int]],
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

        if _segment_key(path) in used_diagonals:
            continue
        if any(cell in blocked for cell in path):
            continue
        if allowed_cells is not None and any(cell not in allowed_cells for cell in path):
            continue
        return path

    return None


def _try_place_barrier_through_cell(
    H: int,
    W: int,
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    used_diagonals: set[tuple[str, int]],
) -> list[tuple[int, int]] | None:
    slope_types = ["backslash", "slash"]
    for _ in range(_BARRIER_RETRIES):
        random.shuffle(slope_types)
        for slope_type in slope_types:
            step = (1, 1) if slope_type == "backslash" else (1, -1)
            neg_max = _step_limits(H, W, cell, (-step[0], -step[1]))
            pos_max = _step_limits(H, W, cell, step)
            if neg_max < 1 or pos_max < 1:
                continue

            neg_choices = list(range(1, neg_max + 1))
            pos_choices = list(range(1, pos_max + 1))
            random.shuffle(neg_choices)
            random.shuffle(pos_choices)

            for neg in neg_choices:
                for pos in pos_choices:
                    seg_len = neg + pos
                    if not (_SEG_LEN_RANGE[0] <= seg_len <= _SEG_LEN_RANGE[1]):
                        continue
                    path = [
                        (cell[0] - neg * step[0] + k * step[0], cell[1] - neg * step[1] + k * step[1])
                        for k in range(seg_len + 1)
                    ]
                    if _segment_key(path) in used_diagonals:
                        continue
                    if any(p in blocked for p in path):
                        continue
                    return path
    return None


def _place_interacting_pair(H: int, W: int) -> list[dict] | None:
    for _ in range(_PAIR_RETRIES):
        emitter_path = _try_place_free_segment(H, W, set(), set())
        if emitter_path is None or len(emitter_path) < 4:
            continue

        trigger_choices = emitter_path[1:-1]
        random.shuffle(trigger_choices)
        for trigger in trigger_choices:
            ray_dirs = _ORTHO_DIRS[:]
            random.shuffle(ray_dirs)
            for delta in ray_dirs:
                ray_cells = _cells_along_direction(trigger, delta, H, W)
                if len(ray_cells) < 3:
                    continue

                hit_choices = ray_cells[1:-1]
                random.shuffle(hit_choices)
                for hit in hit_choices:
                    barrier_path = _try_place_barrier_through_cell(
                        H,
                        W,
                        hit,
                        set(emitter_path),
                        {_segment_key(emitter_path)},
                    )
                    if barrier_path is None:
                        continue

                    segments = [
                        {"path": emitter_path, "triggers": [trigger]},
                        {"path": barrier_path, "triggers": []},
                    ]
                    probe = _build_input(H, W, 0, segments)
                    metrics = _variant_metrics(probe)
                    if (
                        metrics["line_color"] == _LINE_COLOR
                        and metrics["trigger_count"] >= 1
                        and metrics["blocked_by_line"] >= 1
                        and metrics["shortened_rays"] >= 1
                        and metrics["cross_count"] >= 1
                        and metrics["differs_from_d2"]
                        and metrics["differs_from_stop_plain"]
                    ):
                        return segments

    return None


def _safe_background_cells(inp: np.ndarray, metrics: dict) -> list[tuple[int, int]]:
    H, W = inp.shape
    bg = get_bg(inp)
    cells = []
    for r in range(H):
        for c in range(W):
            if int(inp[r, c]) != bg:
                continue
            if int(metrics["variant_out"][r, c]) != bg:
                continue
            if int(metrics["d2_out"][r, c]) != bg:
                continue
            if int(metrics["stop_plain_out"][r, c]) != bg:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_extra_segments(
    H: int,
    W: int,
    bg: int,
    base_segments: list[dict],
    total_segments: int,
) -> list[dict]:
    segments = [{"path": list(seg["path"]), "triggers": list(seg["triggers"])} for seg in base_segments]
    metrics = _variant_metrics(_build_input(H, W, bg, segments))
    safe_cells = set(_safe_background_cells(_build_input(H, W, bg, segments), metrics))

    blocked = set()
    used_diagonals = set()
    for seg in segments:
        blocked.update(seg["path"])
        used_diagonals.add(_segment_key(seg["path"]))

    for _ in range(_EXTRA_SEGMENT_RETRIES):
        if len(segments) >= total_segments:
            break
        path = _try_place_free_segment(H, W, blocked, used_diagonals, safe_cells)
        if path is None:
            continue
        segments.append({"path": path, "triggers": []})
        blocked.update(path)
        used_diagonals.add(_segment_key(path))
        for cell in path:
            safe_cells.discard(cell)

    return segments


def _shares_diagonal(cell: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) == abs(c - cc) and (r, c) != (rr, cc) for rr, cc in others)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _CROSS_COLOR)])

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
        and metrics["blocked_by_line"] >= 1
        and metrics["shortened_rays"] >= 1
        and metrics["cross_count"] >= 1
        and metrics["differs_from_d2"]
        and metrics["differs_from_stop_plain"]
    ):
        return None

    safe_cells = _safe_background_cells(inp, metrics)
    line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    trigger_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        placed = False
        for r, c in safe_cells:
            if int(inp[r, c]) != bg:
                continue
            if _shares_diagonal((r, c), line_positions):
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
            if _shares_diagonal((r, c), trigger_positions):
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
        and final_metrics["blocked_by_line"] >= 1
        and final_metrics["shortened_rays"] >= 1
        and final_metrics["cross_count"] >= 1
        and final_metrics["differs_from_d2"]
        and final_metrics["differs_from_stop_plain"]
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
