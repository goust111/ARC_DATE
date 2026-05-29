# Variant of db695cfb
# Dimension: D14 — 线条交叉处理规则 (meta)
# Change: 隐含规则——蓝色对角线和品红延伸线可能交叉，交叉点的颜色由绘制顺序决定（后绘制的覆盖先绘制的）... → 交叉点使用特殊颜色
#
# Design analysis for the D14 variant
# ----------------------------------
# variant_spec:
#   source_task_baseline:
#     - Reference task e12f9a14 was read first to calibrate collision complexity.
#     - That task makes collisions explicit and visually legible instead of leaving them
#       as accidental draw-order side effects. This variant applies the same design idea
#       to db695cfb, but keeps the interaction simpler: only blue-line / magenta-ray
#       crossings are special, and the special mark is a fixed third color.
#   concretization:
#     - Input keeps the original object vocabulary:
#       blue endpoints (color 1), magenta trigger pixels (color 6), background, and
#       optional distractors of those same two colors.
#     - Output still completes each blue diagonal and still emits magenta rays from
#       trigger pixels along the perpendicular diagonal direction.
#     - New rule: whenever a magenta ray visits a cell that belongs to a completed blue
#       diagonal segment, that cell becomes red (color 2) instead of staying blue.
#     - Trigger cells themselves remain magenta, because they are trigger markers rather
#       than blue line cells.
#   behavior_trace:
#     - Original local behavior:
#       1. Draw the blue diagonal first.
#       2. Emit the magenta ray later.
#       3. When the ray reaches a blue cell, it skips repainting because the cell is
#          already non-background, so the crossing stays blue.
#     - Variant local behavior:
#       1. Identify the full set of cells that belong to blue diagonals.
#       2. Draw those blue cells.
#       3. Emit the magenta rays.
#       4. If a ray reaches one of the blue-diagonal cells, repaint that cell red.
#     - Concrete mini-example:
#       a source "\" segment with a magenta trigger emits a "/" ray; if that ray hits
#       another "\" segment at one interior cell, the original output keeps that cell
#       blue, while this variant turns only that crossing cell red.
#   complexity_calibration:
#     - The change is visible but narrow: it does not alter pairing, slope inference,
#       trigger detection, or ray direction.
#     - To avoid a weak or degenerate variant, generation guarantees at least one true
#       line-ray crossing per puzzle, so the special color is always exemplified.
#     - The special color is fixed to red (2), which keeps the hypothesis space small:
#       solvers only need to infer one extra interaction state.
#   implicit_assumption_audit:
#     - In the original draw_segment(), writing blue directly into out and later letting
#       rays fill only background implicitly assumed that blue-ray crossings should remain
#       blue. That assumption is invalid now.
#     - The variant therefore separates "which cells are blue-diagonal cells" from the
#       later ray-painting pass. Crossings are detected against that structural set.
#     - The old rule "any non-blue pixel on a blue diagonal is a trigger" still holds,
#       so trigger collection logic remains valid.
#     - The old rule "rays never overwrite non-background cells" is now refined to
#       "rays do not overwrite arbitrary non-background cells, except that they convert
#       blue-diagonal cells to the dedicated crossing color".
#   placement_legality:
#     - The generator constructs one mandatory collision pair:
#       a trigger-bearing source segment and a second parallel blue segment positioned so
#       that exactly one emitted ray intersects it.
#     - That crossing is placed on an interior blue cell, not on a trigger cell, so the
#       red output is unambiguous.
#     - Extra segments/triggers are only accepted when their paths and rays avoid every
#       already-occupied or already-drawn output cell, preventing accidental overlaps.
#     - Segments are kept on distinct diagonal keys within each slope family so the
#       solver's "connect extremes on a diagonal" logic still matches generation.
#   observability_and_cleanliness:
#     - Every generated puzzle contains at least one red crossing cell in the output.
#     - Crossings are point events rather than long overlaps, so the output remains easy
#       to parse: blue = completed diagonal, magenta = emitted ray, red = collision.
#     - Ray-ray crossings are not used as a new rule; they are avoided in generation.
#   color_inference:
#     - Only three task-relevant colors participate in the transformation:
#       blue (line), magenta (trigger/ray), red (line-ray crossing).
#     - Background is chosen from colors other than 1, 2, and 6 so the special red is
#       always visually distinct and never conflated with the canvas.

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_CROSS_COLOR = 2
_TRIGGER_COLOR = 6

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (2, 4)         # at least 2 so a true line-ray crossing can exist
_SEG_LEN_RANGE = (3, 16)
_EXTRA_TRIGGERS_PER_SEG = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_PAIR_RETRIES = 160
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_EXTRA_SEGMENT_RETRIES = 120
_GEN_RETRIES = 300


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _path_key(path: list[tuple[int, int]], slope_type: str) -> int:
    r, c = path[0]
    return r - c if slope_type == "backslash" else r + c


def _step_limits(H: int, W: int, cell: tuple[int, int], step: tuple[int, int]) -> int:
    r, c = cell
    dr, dc = step
    n = 0
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
        n += 1
    return n


def _segment_blue_cells(path, trigger_positions) -> set[tuple[int, int]]:
    return set(path) - set(trigger_positions)


def _render_variant(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    blue_draw_cells: set[tuple[int, int]],
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]],
) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()

    for r, c in blue_draw_cells:
        if int(out[r, c]) == bg:
            out[r, c] = line_color

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in blue_draw_cells:
                    out[r, c] = _CROSS_COLOR
                elif int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


def _render_from_segments(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    blue_draw_cells: set[tuple[int, int]] = set()
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for _, _, trigger_positions, path, perp_dirs in segments:
        blue_draw_cells.update(_segment_blue_cells(path, trigger_positions))
        for tr, tc in trigger_positions:
            triggers.setdefault((tr, tc, _TRIGGER_COLOR), set()).update(perp_dirs)
    return _render_variant(inp, bg, _LINE_COLOR, blue_draw_cells, triggers)


def _build_input(H: int, W: int, bg: int, segments) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR
    return inp


def _collect_structure_from_grid(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[set[tuple[int, int]], dict[tuple[int, int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    pts = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    blue_draw_cells: set[tuple[int, int]] = set()
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def collect_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(grid[r, c])
            if v == bg or v == line_color:
                blue_draw_cells.add((r, c))
            elif v != bg:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_segment(g[0], g[-1])

    return blue_draw_cells, triggers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal pixel pairs, then mark line-ray crossings with a third color.

    Input structure:
    - bg = most common color. Blue pixels form endpoint pairs on shared diagonals, and magenta trigger pixels may lie on those diagonals.

    Rule:
    - Complete each blue diagonal segment between its extreme aligned blue endpoints.
    - Every non-blue pixel encountered on such a diagonal emits magenta rays along the perpendicular diagonal directions.
    - If a magenta ray passes through a completed blue-diagonal cell, repaint that crossing cell red instead of leaving it blue.

    Key insight: Triggers still generate perpendicular diagonal rays, but crossings are no longer governed by draw order; blue-line / magenta-ray overlaps become explicit red collision markers.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

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

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    blue_draw_cells, triggers = _collect_structure_from_grid(grid, bg, line_color)
    return _render_variant(grid, bg, line_color, blue_draw_cells, triggers)


def _compute_ray_cells(triggers, perp_dirs, H, W):
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _try_place_segment(slope_type, seg_len, H, W, blocked, used_backslash, used_slash):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            key = r0 - c0
            if key in used_backslash:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            key = r0 + c0
            if key in used_slash:
                continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in blocked for rc in path):
            continue
        return path, dc
    return None


def _try_place_collision_pair(H: int, W: int, max_seg_len: int):
    min_pair_len = max(_SEG_LEN_RANGE[0], 4)
    if max_seg_len < min_pair_len:
        return None

    for _ in range(_PAIR_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(min_pair_len, min(max_seg_len, _SEG_LEN_RANGE[1]))
        source_result = _try_place_segment(slope_type, seg_len, H, W, set(), set(), set())
        if source_result is None:
            continue
        source_path, dc = source_result
        perp_dirs = [(1, -dc), (-1, dc)]

        trigger_choices = source_path[1:-1]
        random.shuffle(trigger_choices)
        for tr, tc in trigger_choices:
            dirs = list(perp_dirs)
            random.shuffle(dirs)
            for ddr, ddc in dirs:
                for dist in range(2, max(H, W)):
                    x = (tr + ddr * dist, tc + ddc * dist)
                    if not (0 <= x[0] < H and 0 <= x[1] < W):
                        break

                    seg_step = (1, dc)
                    neg_max = _step_limits(H, W, x, (-seg_step[0], -seg_step[1]))
                    pos_max = _step_limits(H, W, x, seg_step)
                    if neg_max < 1 or pos_max < 1:
                        continue

                    neg_options = list(range(1, neg_max + 1))
                    pos_options = list(range(1, pos_max + 1))
                    random.shuffle(neg_options)
                    random.shuffle(pos_options)
                    for neg in neg_options:
                        for pos in pos_options:
                            target_len = neg + pos
                            if not (_SEG_LEN_RANGE[0] <= target_len <= min(max_seg_len, _SEG_LEN_RANGE[1])):
                                continue

                            target_path = [
                                (x[0] - neg * seg_step[0] + k * seg_step[0], x[1] - neg * seg_step[1] + k * seg_step[1])
                                for k in range(target_len + 1)
                            ]
                            if x == target_path[0] or x == target_path[-1]:
                                continue
                            if set(target_path) & set(source_path):
                                continue

                            source_seg = (source_path[0], source_path[-1], [(tr, tc)], source_path, perp_dirs)
                            target_seg = (target_path[0], target_path[-1], [], target_path, perp_dirs)
                            return [source_seg, target_seg]
    return None


def _extract_used_keys(segments):
    used_backslash = set()
    used_slash = set()
    for _, _, _, path, _ in segments:
        if len(path) < 2:
            continue
        dc = path[1][1] - path[0][1]
        if dc == 1:
            used_backslash.add(path[0][0] - path[0][1])
        else:
            used_slash.add(path[0][0] + path[0][1])
    return used_backslash, used_slash


def _current_blocked_cells(H: int, W: int, bg: int, segments) -> set[tuple[int, int]]:
    inp = _build_input(H, W, bg, segments)
    out = _render_from_segments(inp, bg, segments)
    return set(map(tuple, np.argwhere(out != bg)))


def _current_input_cells(H: int, W: int, bg: int, segments) -> set[tuple[int, int]]:
    inp = _build_input(H, W, bg, segments)
    return set(map(tuple, np.argwhere(inp != bg)))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _CROSS_COLOR)])

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    segments = _try_place_collision_pair(H, W, max_seg_len)
    if segments is None:
        return None

    target_segments = random.randint(*_SEGMENT_COUNT)
    for _ in range(_EXTRA_SEGMENT_RETRIES):
        if len(segments) >= target_segments:
            break

        used_backslash, used_slash = _extract_used_keys(segments)
        blocked = _current_blocked_cells(H, W, bg, segments)
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, H, W, blocked, used_backslash, used_slash)
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]

        trigger_positions = []
        path_interior = path[1:-1]
        max_extra_triggers = min(_EXTRA_TRIGGERS_PER_SEG[1], len(path_interior))
        n_triggers = random.randint(_EXTRA_TRIGGERS_PER_SEG[0], max_extra_triggers)
        if n_triggers > 0:
            candidates = path_interior[:]
            random.shuffle(candidates)
            candidate_triggers = candidates[:n_triggers]
            ray_cells = _compute_ray_cells(candidate_triggers, perp_dirs, H, W)
            if any(rc in blocked for rc in ray_cells):
                continue
            trigger_positions = candidate_triggers

        segments.append((path[0], path[-1], trigger_positions, path, perp_dirs))

    inp = _build_input(H, W, bg, segments)
    expected = _render_from_segments(inp, bg, segments)

    occupied = set(map(tuple, np.argwhere(inp != bg)))
    blocked = set(map(tuple, np.argwhere(expected != bg)))

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in blocked or (r, c) in occupied:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in blocked or (r, c) in occupied:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _render_from_segments(inp, bg, segments)
    if np.array_equal(result, expected) and np.any(expected == _CROSS_COLOR):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
