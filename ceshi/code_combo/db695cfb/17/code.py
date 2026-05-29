# Combo variant of db695cfb (index 17)
# combo_spec: D3/3+D16/2
# Name: 对称长度延伸+单方向

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo D3/3 + D16/2 variant
# ===================================================
# a) Concrete behavior and trace:
# - The blue structure is unchanged: two line-color endpoints on one 45-degree diagonal
#   imply a completed blue segment between the extremes.
# - Only non-blue pixels lying on that completed segment are active triggers.
# - Fusion rule:
#   1. Order those triggers along the segment from top endpoint to bottom endpoint.
#   2. Trigger #1, #3, #5... emits only the "upward" perpendicular diagonal ray.
#      Trigger #2, #4, #6... emits only the opposite "downward" perpendicular ray.
#   3. That single ray is finite: if the trigger is k steps from one endpoint and L-k
#      from the other on a segment of length L, then its ray length is min(k, L-k).
# - Example on a backslash segment from (1,1) to (7,7), so L=6:
#   triggers at indices 1, 3, 5 correspond to cells (2,2), (4,4), (6,6).
#   Original task: each trigger would emit two full slash-diagonal rays to the border.
#   Fused task:
#     * trigger #1 uses the upward ray only, reach 1
#     * trigger #2 uses the downward ray only, reach 3
#     * trigger #3 uses the upward ray only, reach 1
#   The output becomes a one-sided spindle: only one arm per trigger survives, and the
#   arm lengths taper from short-near-end to long-near-center back to short-near-end.
#
# b) Complexity calibration:
# - If every segment had only one trigger, the direction rule could be mistaken for a
#   fixed orientation choice instead of an order-based alternation. The generator
#   therefore gives every active segment at least two triggers.
# - If all reaches were equal, the D3 change would be visually weak. The generator also
#   requires at least one showcase segment with three or more triggers and at least two
#   distinct reaches, so the tapered profile is visible.
# - I keep the degrees of freedom small: direction is binary and determined only by
#   visible order; reach is a small integer derived only from visible endpoint distance.
#
# c) Implicit-assumption audit:
# - Original `draw_segment()` assumed every trigger stores both perpendicular directions.
#   That is invalid now; direction assignment must happen after collecting the trigger
#   order along each completed segment.
# - Original emission assumed "continue until out of bounds". That is also invalid; the
#   solver must carry an explicit per-trigger reach.
# - The solver still assumes the meaningful carrier color is the one with the longest
#   diagonal-aligned same-color pair. The generator preserves that by preventing stray
#   blue pixels from creating competing diagonal pairs.
# - If an arbitrary test grid activates the same trigger from multiple segments, the
#   solver keeps the maximum reach for each assigned direction. Generated data avoids
#   such multi-activation, so each trigger behaves as a single-ray object there.
#
# d) Placement legality:
# - Segments are placed output-first and may not overlap prior occupied input pixels or
#   previously drawn output-only cells.
# - Finite single-ray cells are checked against both `occupied` and `drawn`, so rays do
#   not collide with endpoints, triggers, diagonal fills, or other rays.
# - For the chosen reach=min(k, L-k), the finite perpendicular ray stays in-bounds; it
#   is mirrored by the already in-bounds room guaranteed by the segment geometry.
#
# e) Visual observability:
# - The rule depends only on visible geometry: trigger order along a segment and
#   distance to the endpoints. No hidden use of color ids, scan order outside the
#   segment, or metadata is involved.
# - The generator enforces at least one sample object where both effects are visible:
#   some cells are missing because one side is suppressed, and additional cells are
#   missing because the surviving side stops before the border.
#
# f) Visual cleanliness:
# - Segment count is reduced relative to the source task and ray overlaps are rejected.
#   That keeps the one-sided spindle shapes legible instead of producing dense meshes.
# - Distractors remain sparse and are placed only on untouched background cells.
#
# g) Change inferability:
# - No color remapping is introduced. Blue still marks the carrier diagonal and magenta
#   still draws the secondary structure. Only the side count and finite length change.
# - Because at least one segment has repeated trigger behavior with alternating sides
#   and unequal reaches, a human can infer both dimensions from 2-4 examples.
#
# h) Post-change assumptions checked against the generator:
# - `main()` assumes at least one valid diagonal endpoint pair exists; generator always
#   places one or more segments.
# - `main()` assumes ordering is read along the completed segment from top to bottom;
#   generator constructs trigger specs in that same order.
# - `main()` assumes the fusion-relevant triggers lie on completed blue segments; the
#   generator keeps off-diagonal magenta distractors away from all drawn structures.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal endpoint pairs, then let on-segment trigger pixels fire one finite perpendicular ray each.

    Input structure:
    - The background is the majority color. One non-background color appears as endpoint pairs on shared 45-degree diagonals, while differently colored trigger pixels may lie strictly between those endpoints on the same diagonals.

    Rule:
    - Fill each qualifying diagonal between its extreme same-color endpoints.
    - Read each completed segment from its upper endpoint to its lower endpoint. Non-line pixels encountered on that path act as triggers: odd-indexed triggers emit only the upward perpendicular diagonal, even-indexed triggers emit only the downward one, and each emitted ray stops after a number of steps equal to that trigger's shorter distance to the two endpoints.

    Key insight:
    - The trigger pixels are no longer symmetric splitters and no longer reach the border. Their visible order chooses which side survives, while their symmetry position on the carrier diagonal decides how long that single surviving arm is.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    best = None  # (distance, color)
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
    pts = by_color[line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dc = 1 if q[1] > p[1] else -1
        upward_dir = (-1, dc)
        downward_dir = (1, -dc)
        segment_triggers: list[tuple[int, tuple[int, int, int]]] = []
        steps = len(path) - 1

        for path_idx, (r, c) in enumerate(path):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                segment_triggers.append((path_idx, (r, c, v)))

        for order_idx, (path_idx, key) in enumerate(segment_triggers):
            reach = min(path_idx, steps - path_idx)
            if reach <= 0:
                continue
            ray_dir = upward_dir if order_idx % 2 == 0 else downward_dir
            reach_by_dir = triggers.setdefault(key, {})
            reach_by_dir[ray_dir] = max(reach_by_dir.get(ray_dir, 0), reach)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), reach_by_dir in triggers.items():
        for (ddr, ddc), reach in reach_by_dir.items():
            for step in range(1, reach + 1):
                r = r0 + ddr * step
                c = c0 + ddc * step
                if 0 <= r < h and 0 <= c < w and int(out[r, c]) == bg:
                    out[r, c] = tcol

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (6, 16)
_TRIGGERS_PER_SEG = (2, 4)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _boundary_steps(r: int, c: int, ddr: int, ddc: int, h: int, w: int) -> int:
    steps = 0
    r += ddr
    c += ddc
    while 0 <= r < h and 0 <= c < w:
        steps += 1
        r += ddr
        c += ddc
    return steps


def _assign_trigger_specs(
    trigger_indices: list[int], path: list[tuple[int, int]], dc: int
) -> list[tuple[int, int, int, tuple[int, int]]]:
    upward_dir = (-1, dc)
    downward_dir = (1, -dc)
    seg_len = len(path) - 1
    specs = []
    for order_idx, idx in enumerate(sorted(trigger_indices)):
        ray_dir = upward_dir if order_idx % 2 == 0 else downward_dir
        r, c = path[idx]
        reach = min(idx, seg_len - idx)
        specs.append((r, c, reach, ray_dir))
    return specs


def _compute_ray_cells(
    trigger_specs: list[tuple[int, int, int, tuple[int, int]]]
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc, reach, (ddr, ddc) in trigger_specs:
        for step in range(1, reach + 1):
            cells.add((tr + ddr * step, tc + ddc * step))
    return cells


def _segment_metrics(
    trigger_specs: list[tuple[int, int, int, tuple[int, int]]], h: int, w: int
) -> tuple[int, int, int]:
    truncated_total = 0
    suppressed_total = 0
    reach_values = set()
    for tr, tc, reach, (ddr, ddc) in trigger_specs:
        truncated_total += max(0, _boundary_steps(tr, tc, ddr, ddc, h, w) - reach)
        suppressed_total += _boundary_steps(tr, tc, -ddr, -ddc, h, w)
        reach_values.add(reach)
    return truncated_total, suppressed_total, len(reach_values)


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    expected = inp.copy()
    for _, _, trigger_specs, path in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, reach, (ddr, ddc) in trigger_specs:
            for step in range(1, reach + 1):
                r = tr + ddr * step
                c = tc + ddc * step
                if expected[r, c] == bg:
                    expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    line_positions: list[tuple[int, int]],
):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            dc = 1
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        endpoints = (path[0], path[-1])
        if any(abs(r - lr) == abs(c - lc) for r, c in endpoints for lr, lc in line_positions):
            continue
        return path, dc
    return None


def _showcase_trigger_indices(seg_len: int) -> list[int]:
    mid = seg_len // 2
    candidates = {1, mid, seg_len - 1}
    if seg_len >= 8:
        candidates.add(seg_len - 2)
    return sorted(candidates)


def _random_trigger_indices(seg_len: int) -> list[int]:
    interior = list(range(1, seg_len))
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(interior))
    min_triggers = min(_TRIGGERS_PER_SEG[0], len(interior))
    if max_triggers < _TRIGGERS_PER_SEG[0]:
        return []
    n_triggers = random.randint(min_triggers, max_triggers)
    return sorted(random.sample(interior, n_triggers))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments = []
    line_positions: list[tuple[int, int]] = []

    total_triggers = 0
    total_fused_ray_cells = 0
    truncated_extra_total = 0
    suppressed_side_total = 0
    showcase_count = 0

    for seg_idx in range(n_segments):
        placed = False
        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(["backslash", "slash"])
            min_len = max(_SEG_LEN_RANGE[0], 7) if seg_idx == 0 else _SEG_LEN_RANGE[0]
            seg_len = random.randint(min_len, min(max_seg_len, _SEG_LEN_RANGE[1]))

            result = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn, line_positions)
            if result is None:
                continue
            path, dc = result

            if seg_idx == 0:
                trigger_indices = _showcase_trigger_indices(seg_len)
            else:
                trigger_indices = _random_trigger_indices(seg_len)
                if not trigger_indices:
                    continue

            trigger_specs = _assign_trigger_specs(trigger_indices, path, dc)
            fused_ray_cells = _compute_ray_cells(trigger_specs)
            if any(rc in occupied or rc in drawn for rc in fused_ray_cells):
                continue

            truncated_here, suppressed_here, distinct_reach_count = _segment_metrics(trigger_specs, h, w)
            if seg_idx == 0:
                if len(trigger_specs) < 3 or distinct_reach_count < 2:
                    continue
                if truncated_here <= 0 or suppressed_here <= 0:
                    continue

            ep0, ep1 = path[0], path[-1]
            segments.append((ep0, ep1, trigger_specs, path))
            occupied.update([ep0, ep1])
            occupied.update((tr, tc) for tr, tc, _, _ in trigger_specs)
            drawn.update(set(path[1:-1]) - {(tr, tc) for tr, tc, _, _ in trigger_specs})
            drawn.update(fused_ray_cells)
            line_positions.extend([ep0, ep1])

            total_triggers += len(trigger_specs)
            total_fused_ray_cells += len(fused_ray_cells)
            truncated_extra_total += truncated_here
            suppressed_side_total += suppressed_here
            if len(trigger_specs) >= 3 and distinct_reach_count >= 2 and truncated_here > 0 and suppressed_here > 0:
                showcase_count += 1

            placed = True
            break

        if not placed:
            return None

    if not segments:
        return None
    if total_triggers < 3 or total_fused_ray_cells == 0:
        return None
    if showcase_count == 0 or truncated_extra_total == 0 or suppressed_side_total == 0:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for ep0, ep1, trigger_specs, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, _, _ in trigger_specs:
            inp[tr, tc] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
