# Variant of db695cfb
# Dimension: D14 — 线条交叉处理规则 (meta)
# Change: 隐含规则——蓝色对角线和品红延伸线可能交叉，交叉点的颜色由绘制顺序决定（后绘制的覆盖先绘制的）... → 交叉时两条线都中断

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for D14
# a) Concrete behavior:
#    The active blue objects are still endpoint pairs that imply full 45-degree diagonals, and magenta
#    triggers still sit on those diagonals. The new visual event happens only when a magenta ray would
#    step onto a would-be blue diagonal cell belonging to another connector: that shared cell stays
#    background, the ray stops before it, and the blue connector also leaves that one-cell hole. Small
#    trace: if a trigger would previously emit "... 6 6 6 X 6 6" across a blue path cell X, the old
#    code either preserved blue or depended on draw order; the variant instead produces "... 6 6 . 1 1"
#    or the symmetric mirror, where "." is the untouched background crossing gap.
# b) Complexity calibration:
#    The generator forces at least one real blue-ray interaction in every puzzle and randomly chooses
#    whether one or both ray directions are truncated. This keeps the new rule visible without adding
#    extra object types or hidden bookkeeping.
# c) Original assumptions that break:
#    - Original `draw_segment` wrote blue immediately into `out`; that bakes in "blue always occupies
#      every bg cell on the path". Now we must first collect all blue candidate cells, then remove the
#      ray-hit cells before drawing.
#    - Original ray emission only checked `out[r,c] == bg` and kept walking. Under D14 a ray must stop
#      at its first would-be blue cell, even though that cell is still background in the input.
#    - Using the mutable output as collision state would hide future crossings, so the new solver uses
#      geometry reconstructed from the input instead of incremental overwrite order.
# d) Placement legality:
#    The generator builds one source segment with one trigger and one or two target segments of the
#    same blue orientation. Because same-orientation diagonals never cross each other, only the magenta
#    rays can meet those target segments. Reserved cells include all blue candidate cells, all emitted
#    ray cells, and all forced crossing gaps, so distractors cannot invalidate the intended geometry.
# e) Visual observability:
#    Every generated sample contains at least one actual crossing gap, and many contain two. Some
#    samples leave the opposite ray free to reach the border, making the "stop only at blue collision"
#    rule directly visible from just a few examples.
# f) Visual cleanliness:
#    Only blue connectors, magenta rays, and a small number of same-color distractors are used. The
#    crossing cell stays blank instead of layering colors, so the output remains easy to parse.
# g) Color inference:
#    No new color mapping is introduced. Blue remains the connector color and magenta remains the ray
#    color; the novelty is entirely in the spatial interaction.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _collect_geometry(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int, int], set[tuple[int, int]]], set[tuple[int, int]]]:
    h, w = grid.shape
    pts = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    blue_paths: list[list[tuple[int, int]]] = []
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    blue_candidate_cells: set[tuple[int, int]] = set()

    def register_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        blue_paths.append(path)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(grid[r, c])
            if v == bg or v == line_color:
                blue_candidate_cells.add((r, c))
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            register_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            register_segment(g[0], g[-1])

    return blue_paths, triggers, blue_candidate_cells


def _resolve_variant(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    blue_paths: list[list[tuple[int, int]]],
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]],
    blue_candidate_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[int, int, int]]]:
    h, w = grid.shape
    out = grid.copy()
    blocked_crossings: set[tuple[int, int]] = set()
    ray_fills: list[tuple[int, int, int]] = []

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in sorted(dirs):
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in blue_candidate_cells:
                    if int(grid[r, c]) == bg:
                        blocked_crossings.add((r, c))
                    break
                if int(grid[r, c]) == bg:
                    ray_fills.append((r, c, tcol))
                r += ddr
                c += ddc

    for path in blue_paths:
        for r, c in path:
            if (r, c) in blocked_crossings:
                continue
            if int(out[r, c]) == bg:
                out[r, c] = line_color

    for r, c, tcol in ray_fills:
        if int(out[r, c]) == bg:
            out[r, c] = tcol

    return out, blocked_crossings, ray_fills


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete blue diagonals, then emit perpendicular trigger rays that stop short of blue collisions.

    Input structure:
    - The background is the majority color. Blue pixels mark diagonal endpoint pairs, and non-blue pixels sitting on those diagonals are trigger markers.
    - Blue pairs may define one or more full hidden diagonals, while unrelated blue or magenta distractors can appear away from the active geometry.

    Rule:
    - Connect the extreme blue pixels on every occupied 45-degree diagonal, just as in the base task.
    - Each trigger sends rays along the perpendicular diagonal directions, but if a ray would enter a blue connector cell, the ray stops before it and that blue cell is also left blank.

    Key insight:
    - Solve the blue diagonals first in your head, then treat the first blue cell reached by each trigger ray as a mutual exclusion gap shared by both structures.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

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
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    blue_paths, triggers, blue_candidate_cells = _collect_geometry(grid, bg, line_color)
    out, _, _ = _resolve_variant(grid, bg, line_color, blue_paths, triggers, blue_candidate_cells)
    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEG_LEN_RANGE = (3, 16)
_TARGET_SEGMENTS = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_LAYOUT_RETRIES = 140
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _max_steps(center: tuple[int, int], step: tuple[int, int], H: int, W: int) -> int:
    r, c = center
    dr, dc = step
    steps = 0
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
        steps += 1
    return steps


def _build_span(center: tuple[int, int], step: tuple[int, int], back: int, fwd: int) -> list[tuple[int, int]]:
    r0 = center[0] - back * step[0]
    c0 = center[1] - back * step[1]
    total = back + fwd
    return [(r0 + k * step[0], c0 + k * step[1]) for k in range(total + 1)]


def _sample_span(
    center: tuple[int, int], step: tuple[int, int], H: int, W: int, min_len: int, max_len: int
) -> tuple[list[tuple[int, int]], int, int] | None:
    back_cap = _max_steps(center, (-step[0], -step[1]), H, W)
    fwd_cap = _max_steps(center, step, H, W)
    max_total = min(max_len, back_cap + fwd_cap)
    options: list[tuple[int, int]] = []
    for total in range(min_len, max_total + 1):
        back_min = max(1, total - fwd_cap)
        back_max = min(back_cap, total - 1)
        for back in range(back_min, back_max + 1):
            fwd = total - back
            if 1 <= fwd <= fwd_cap:
                options.append((back, fwd))
    if not options:
        return None
    back, fwd = random.choice(options)
    return _build_span(center, step, back, fwd), back, fwd


def _segment_dict(
    path: list[tuple[int, int]], trigger_positions: list[tuple[int, int]], dc_seg: int
) -> dict[str, object]:
    return {
        "path": path,
        "trigger_positions": list(trigger_positions),
        "perp_dirs": [(1, -dc_seg), (-1, dc_seg)],
    }


def _build_expected(
    inp: np.ndarray, bg: int, segments: list[dict[str, object]]
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    blue_candidate_cells: set[tuple[int, int]] = set()
    ray_cells: set[tuple[int, int]] = set()
    blocked_crossings: set[tuple[int, int]] = set()

    for seg in segments:
        trigger_positions = set(seg["trigger_positions"])
        for cell in seg["path"]:
            if cell not in trigger_positions:
                blue_candidate_cells.add(cell)

    H, W = inp.shape
    for seg in segments:
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if (r, c) in blue_candidate_cells:
                        blocked_crossings.add((r, c))
                        break
                    ray_cells.add((r, c))
                    r += ddr
                    c += ddc

    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if (r, c) in blocked_crossings:
                continue
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR

    for r, c in ray_cells:
        if expected[r, c] == bg:
            expected[r, c] = _TRIGGER_COLOR

    return expected, blue_candidate_cells, ray_cells, blocked_crossings


def _make_active_layout(H: int, W: int) -> list[dict[str, object]] | None:
    for _ in range(_LAYOUT_RETRIES):
        dc_seg = random.choice([1, -1])
        seg_step = (1, dc_seg)
        ray_dirs = [(1, -dc_seg), (-1, dc_seg)]
        chosen_dirs = random.sample([0, 1], random.randint(*_TARGET_SEGMENTS))

        trigger = (random.randint(0, H - 1), random.randint(0, W - 1))
        source_sample = _sample_span(trigger, seg_step, H, W, _SEG_LEN_RANGE[0], _SEG_LEN_RANGE[1])
        if source_sample is None:
            continue
        source_path, _, _ = source_sample
        segments = [_segment_dict(source_path, [trigger], dc_seg)]

        success = True
        for dir_idx in chosen_dirs:
            ray_dir = ray_dirs[dir_idx]
            ray_cap = _max_steps(trigger, ray_dir, H, W)
            if ray_cap < 2:
                success = False
                break

            candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
            for dist in range(2, ray_cap + 1):
                cross = (trigger[0] + dist * ray_dir[0], trigger[1] + dist * ray_dir[1])
                target_sample = _sample_span(cross, seg_step, H, W, _SEG_LEN_RANGE[0], _SEG_LEN_RANGE[1])
                if target_sample is None:
                    continue
                target_path, _, _ = target_sample
                candidates.append((target_path, cross))

            if not candidates:
                success = False
                break

            target_path, _ = random.choice(candidates)
            segments.append(_segment_dict(target_path, [], dc_seg))

        if not success:
            continue

        all_input_cells = set()
        for seg in segments:
            endpoints = {seg["path"][0], seg["path"][-1]}
            active_inputs = endpoints | set(seg["trigger_positions"])
            if all_input_cells & active_inputs:
                success = False
                break
            all_input_cells.update(active_inputs)
        if success:
            return segments

    return None


def _try_place_line_distractor(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    all_line_positions: list[tuple[int, int]],
) -> bool:
    H, W = inp.shape
    for _ in range(_DISTRACTOR_RETRIES):
        r, c = random.randint(0, H - 1), random.randint(0, W - 1)
        if (r, c) in occupied or (r, c) in reserved:
            continue
        if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
            continue
        inp[r, c] = _LINE_COLOR
        occupied.add((r, c))
        all_line_positions.append((r, c))
        return True
    return False


def _try_place_trigger_distractor(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    all_trigger_positions: list[tuple[int, int]],
) -> bool:
    H, W = inp.shape
    for _ in range(_DISTRACTOR_RETRIES):
        r, c = random.randint(0, H - 1), random.randint(0, W - 1)
        if (r, c) in occupied or (r, c) in reserved:
            continue
        if any(abs(r - tr) == abs(c - tc) and r != tr for tr, tc in all_trigger_positions):
            continue
        inp[r, c] = _TRIGGER_COLOR
        occupied.add((r, c))
        all_trigger_positions.append((r, c))
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    segments = _make_active_layout(H, W)
    if segments is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    all_trigger_positions: list[tuple[int, int]] = []

    for seg in segments:
        ep0, ep1 = seg["path"][0], seg["path"][-1]
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        occupied.update([ep0, ep1])
        for tp in seg["trigger_positions"]:
            inp[tp] = _TRIGGER_COLOR
            occupied.add(tp)
            all_trigger_positions.append(tp)

    _, blue_candidate_cells, ray_cells, blocked_crossings = _build_expected(inp, bg, segments)
    if not blocked_crossings:
        return None

    reserved = set(blue_candidate_cells) | set(ray_cells) | set(blocked_crossings)
    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        _try_place_line_distractor(inp, occupied, reserved, all_line_positions)

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        _try_place_trigger_distractor(inp, occupied, reserved, all_trigger_positions)

    expected, _, _, _ = _build_expected(inp, bg, segments)

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
