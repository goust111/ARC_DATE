# Variant of db695cfb
# Dimension: D3 — 延伸终止条件 (dst)
# Change: 延伸至网格边界——射线从品红像素出发一直画到网格边缘才停止... → 延伸到对称位置停止

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D3 variant
# ==================================
# variant_spec:
#   core_change:
#     original: a trigger pixel on a completed blue diagonal emits two perpendicular
#       diagonal rays that keep going until they leave the grid.
#     variant: each perpendicular arm stops after a finite number of cells. That
#       arm length is the trigger's shorter diagonal distance to the two blue
#       endpoints on the same completed diagonal.
#   concrete_shape:
#     - Blue endpoints are still connected by a filled 45-degree segment.
#     - Magenta pixels that lie on that segment still act as trigger points.
#     - If a trigger is k steps from one endpoint and L-k from the other, where L
#       is the endpoint-to-endpoint diagonal step count, then each perpendicular
#       arm has length min(k, L-k).
#     - Central triggers therefore create the longest crossbars, while near-end
#       triggers create short ones. Multiple triggers on one segment form a spindle
#       or diamond-like taper instead of full boundary-reaching X-rays.
#   behavior_trace:
#     example:
#       endpoints: (1,1) and (7,7), so L = 6 on a backslash diagonal.
#       trigger: (3,3), which is k = 2 from the first endpoint and 4 from the second.
#       original: perpendicular slash-diagonal rays run from (3,3) all the way to the
#         grid boundary in both directions.
#       variant: reach = min(2, 4) = 2, so only (4,2),(5,1) and (2,4),(1,5) are filled.
#       difference: the new stopping rule depends on the trigger's position relative
#         to the blue segment, not on the canvas edge.
#   complexity_check:
#     - The recognition logic stays the same: infer the unique line color from the
#       longest same-color diagonal pair, connect extremes on each occupied diagonal,
#       and only use off-color pixels encountered on those paths as triggers.
#     - The new free variable is only the trigger reach, a small integer derived from
#       visible geometry, so the task remains learnable from a few examples.
#   implicit_assumption_audit:
#     - The original emission loop assumed "keep stepping until out of bounds".
#       That is the only solver behavior that is no longer valid.
#     - Trigger collection previously only stored directions. In the variant it must
#       also store a per-direction finite reach. If the same trigger were ever seen
#       from multiple segments, keeping the maximum reach per direction is the safe
#       non-destructive choice.
#     - The old generator's collision checks only guarded against occupied input
#       pixels along rays because boundary rays were unavoidable. In the variant,
#       finite rays are a designed object, so we also prevent overlaps with already
#       drawn output-only cells to keep the shape legible.
#   placement_legality:
#     - For a trigger at index k on a segment of length L, reach=min(k,L-k) is always
#       in-bounds on both perpendicular directions because it mirrors the available
#       room already proven by the segment endpoints.
#     - Generator placement rejects any segment whose diagonal fill or finite magenta
#       arms would overlap prior occupied/drawn cells.
#   observability:
#     - The generator requires at least one "visible" trigger whose finite reach is
#       strictly shorter than the old boundary-reaching ray, so the dataset cannot
#       silently collapse back to the original task.
#     - Segment lengths are kept at 5+ cells and at least one trigger is required so
#       the tapering effect is visible in small sample sets.
#   cleanliness:
#     - Preventing ray overlaps keeps the output from turning into dense, ambiguous
#       magenta meshes when multiple segments are present.
#   new_solver_assumptions_checked_against_generator:
#     - main() assumes at least one diagonal-aligned line-color pair exists; generator
#       always places one or more such segments.
#     - main() assumes triggers relevant to the rule sit on the completed blue paths;
#       generator only expects on-path trigger pixels to activate, while off-path
#       trigger distractors are placed away from drawn cells.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal endpoint pairs, then emit bounded perpendicular diagonals from trigger pixels on those segments.

    Input structure:
    - The background is the majority color. One non-background color appears as endpoint pairs on shared 45-degree diagonals, while differently colored trigger pixels may lie between those endpoints on the same diagonals.

    Rule:
    - Fill each qualifying diagonal between its extreme same-color endpoints.
    - Whenever that filled diagonal passes through a trigger pixel, draw the two perpendicular diagonal arms in the trigger's color, but stop each arm after a number of steps equal to the trigger's shorter distance to the two segment endpoints.

    Key insight:
    - A trigger no longer shoots to the canvas edge; its perpendicular extension is measured by how centrally it sits on the completed diagonal, so middle triggers create longer crossbars than near-end triggers.
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
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        steps = len(path) - 1
        for idx, (r, c) in enumerate(path):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                reach = min(idx, steps - idx)
                if reach <= 0:
                    continue
                key = (r, c, v)
                reach_by_dir = triggers.setdefault(key, {})
                for perp_dir in perp_dirs:
                    reach_by_dir[perp_dir] = max(reach_by_dir.get(perp_dir, 0), reach)

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
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (5, 16)
_TRIGGERS_PER_SEG = (1, 3)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _boundary_steps(r: int, c: int, ddr: int, ddc: int, H: int, W: int) -> int:
    """Count how many cells the original boundary-reaching ray would traverse."""
    steps = 0
    r += ddr
    c += ddc
    while 0 <= r < H and 0 <= c < W:
        steps += 1
        r += ddr
        c += ddc
    return steps


def _compute_ray_cells(trigger_infos, perp_dirs):
    """Compute cells touched by finite perpendicular rays."""
    cells = set()
    for tr, tc, reach in trigger_infos:
        for ddr, ddc in perp_dirs:
            for step in range(1, reach + 1):
                cells.add((tr + ddr * step, tc + ddc * step))
    return cells


def _compute_boundary_ray_cells(trigger_infos, perp_dirs, H, W):
    """Compute cells that the original boundary-reaching rays would touch."""
    cells = set()
    for tr, tc, _ in trigger_infos:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling diagonals and emitting finite rays."""
    expected = inp.copy()
    for _, _, trigger_infos, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, reach in trigger_infos:
            for ddr, ddc in perp_dirs:
                for step in range(1, reach + 1):
                    r = tr + ddr * step
                    c = tc + ddc * step
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn):
    """Try to place one diagonal segment, returning (path, dc) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _visible_trigger_indices(path, seg_len, perp_dirs, H, W):
    """Return interior indices whose finite reach is visibly shorter than the old rays."""
    visible = []
    for idx in range(1, seg_len):
        r, c = path[idx]
        reach = min(idx, seg_len - idx)
        if reach < 2:
            continue
        if all(_boundary_steps(r, c, ddr, ddc, H, W) > reach for ddr, ddc in perp_dirs):
            visible.append(idx)
    return visible


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    segments = []
    total_triggers = 0
    visible_trigger_count = 0
    finite_ray_cells_total = 0
    boundary_only_extra_total = 0

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        perp_dirs = [(1, -dc), (-1, dc)]
        path_interior = path[1:-1]
        if not path_interior:
            continue

        visible_indices = _visible_trigger_indices(path, seg_len, perp_dirs, H, W)
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)

        if max_triggers == 0:
            continue

        n_triggers = random.randint(min_triggers, max_triggers)
        chosen_indices = []

        if visible_indices:
            first_visible = random.choice(visible_indices)
            chosen_indices.append(first_visible)
            remaining = [idx for idx in range(1, seg_len) if idx != first_visible]
            extra_needed = n_triggers - 1
        else:
            remaining = list(range(1, seg_len))
            extra_needed = n_triggers

        if extra_needed > 0:
            extra_candidates = [idx for idx in remaining if idx not in chosen_indices]
            if len(extra_candidates) < extra_needed:
                continue
            chosen_indices.extend(random.sample(extra_candidates, extra_needed))

        chosen_indices = sorted(chosen_indices)
        trigger_infos = [(path[idx][0], path[idx][1], min(idx, seg_len - idx)) for idx in chosen_indices]

        ray_cells = _compute_ray_cells(trigger_infos, perp_dirs)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_infos, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update((tr, tc) for tr, tc, _ in trigger_infos)
        drawn.update(set(path_interior) - {(tr, tc) for tr, tc, _ in trigger_infos})
        drawn.update(ray_cells)

        total_triggers += len(trigger_infos)
        finite_ray_cells_total += len(ray_cells)
        visible_here = 0
        for tr, tc, reach in trigger_infos:
            if all(_boundary_steps(tr, tc, ddr, ddc, H, W) > reach for ddr, ddc in perp_dirs):
                visible_here += 1
        visible_trigger_count += visible_here
        boundary_only_extra_total += len(_compute_boundary_ray_cells(trigger_infos, perp_dirs, H, W) - ray_cells)

    if not segments:
        return None
    if total_triggers == 0 or visible_trigger_count == 0:
        return None
    if finite_ray_cells_total == 0 or boundary_only_extra_total == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_infos, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, _ in trigger_infos:
            inp[tr, tc] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
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
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
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
