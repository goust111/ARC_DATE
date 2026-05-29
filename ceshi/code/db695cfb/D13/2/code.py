# Variant of db695cfb
# Dimension: D13 — 品红像素变换效果 (transform)
# Change: 品红像素保持不变——品红像素在输出中保留原位置和颜色，同时作为延伸线的起点... → 品红像素扩展为十字标记

import numpy as np

from arc_utils import get_bg
import random


# Design analysis for the D13 variant
# - concretization:
#   - Input stays close to the original task: blue endpoints define one or more 45-degree segments, and some magenta pixels lie on those segments.
#   - Only the magenta pixels that are actually hit by a completed blue segment are "active triggers."
#   - In the output, each active trigger becomes a 3x3 diagonal cross/X:
#     {(r, c), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)}.
#   - The two arms aligned with the segment visually interrupt the blue diagonal at the crossing.
#   - The two arms aligned with the perpendicular diagonal become the new launch pads for the magenta rays; the rays continue one more step beyond those endpoints.
#   - Off-diagonal magenta distractors are not active triggers, so they remain single pixels in the output.
#   - Example: for a "\" blue segment crossing a trigger at (4, 4), the original task emitted "/" rays from cells adjacent to (4, 4). The variant paints the X centered at (4, 4), so the "/" rays now continue outward from (3, 5) and (5, 3), beginning their new cells at (2, 6) and (6, 2).
# - complexity calibration:
#   - Keep the original segment/ray logic, but require at least one active trigger globally so the D13 change is visible in generated samples.
#   - Triggers are chosen away from borders and away from endpoints, which makes the 3x3 X readable and prevents it from swallowing the blue endpoints.
#   - Still cap active triggers at 0-2 per segment to avoid clutter and preserve the original puzzle's low-object-count feel.
# - implicit assumptions audit for main():
#   - Line-color selection still relies only on the input grid; this remains valid because triggers are still single magenta pixels in the input.
#   - Segment drawing must happen before trigger expansion; otherwise newly painted magenta arm cells could be mistaken for original triggers.
#   - Ray filling must remain background-only, so the X marker stays intact and existing colored pixels are not overwritten.
#   - Trigger collection keys on non-line pixels found along a segment path; that assumption still holds because only true on-path magenta cells should activate.
# - placement legality in the generator:
#   - Every active trigger must have its full 3x3 X in bounds.
#   - X cells outside the blue path cannot overlap previously placed pixels or prior output-only cells.
#   - Rays starting beyond the X endpoints cannot collide with any occupied or already-reserved drawn cell.
#   - Two active triggers on the same segment cannot share any X cell or ray cell.
# - visual observability:
#   - Requiring at least one active trigger makes the output visibly different from the original task.
#   - The X marker is local, repeated, and shape-based; the rule can be inferred from 2-4 examples without relying on hidden numeric properties.
# - neatness:
#   - The generator keeps trigger count small and enforces spacing from borders/endpoints, which prevents dense overlaps and keeps the blue diagonals, magenta X marks, and magenta rays visually separable.


def _cross_cells(r: int, c: int) -> set[tuple[int, int]]:
    """Return the 3x3 diagonal X centered at (r, c)."""
    return {
        (r, c),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal pixel pairs, then turn crossed trigger pixels into X-marks that launch perpendicular rays.

    Input structure:
    - The background is the most frequent color. One non-background color appears as diagonal endpoint pairs, while some magenta trigger pixels may sit directly on those diagonals.

    Rule:
    - Complete each endpoint-to-endpoint diagonal in the line color.
    - Whenever a completed diagonal passes through a differently colored trigger pixel, expand that trigger into a 3x3 diagonal cross/X in its own color.
    - From the two X arms that lie on the perpendicular diagonal, continue rays outward to the grid boundary, filling only background cells.

    Key insight:
    - On-diagonal trigger pixels are the only active ones: they become local X-shaped junction markers, and the outgoing diagonal rays start beyond the X arms rather than from the original single pixel.
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

    # Choose the line color as the color owning the longest diagonal-aligned pair.
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

    # Each active trigger remembers which perpendicular diagonal directions should emit rays.
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

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
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    # Expand active triggers into X marks before extending rays so the marker stays visible.
    for r0, c0, tcol in triggers:
        for rr, cc in _cross_cells(r0, c0):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = tcol

    # Emit perpendicular diagonal rays beyond the X endpoints.
    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + 2 * ddr, c0 + 2 * ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (D13 variant) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _trigger_is_safe_cell(r: int, c: int, H: int, W: int) -> bool:
    """Active triggers need a full 3x3 X plus one extra ray step in each diagonal direction."""
    return 2 <= r < H - 2 and 2 <= c < W - 2


def _compute_ray_cells(triggers, perp_dirs, H, W):
    """Compute all cells touched by rays that start beyond the X endpoints."""
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + 2 * ddr, tc + 2 * ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid for the D13 variant."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
    for _, _, trigger_positions, _, _ in segments:
        for tr, tc in trigger_positions:
            for r, c in _cross_cells(tr, tc):
                if 0 <= r < H and 0 <= c < W:
                    expected[r, c] = _TRIGGER_COLOR
    for _, _, trigger_positions, _, perp_dirs in segments:
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + 2 * ddr, tc + 2 * ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
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


def _choose_triggers(path, perp_dirs, H, W, occupied, drawn, must_pick_one):
    """Pick a small non-overlapping set of active triggers for one segment."""
    path_set = set(path)
    candidates = [
        p for p in path[2:-2]
        if _trigger_is_safe_cell(p[0], p[1], H, W)
    ]
    random.shuffle(candidates)

    min_needed = 1 if must_pick_one and candidates else 0
    max_allowed = min(_TRIGGERS_PER_SEG[1], len(candidates))
    if max_allowed < min_needed:
        return None
    target = random.randint(min_needed, max_allowed) if max_allowed > 0 else 0

    selected: list[tuple[int, int]] = []
    used_cross = set()
    used_ray = set()
    extra_cross = set()

    for tr, tc in candidates:
        if len(selected) >= target:
            break
        cross = _cross_cells(tr, tc)
        cross_extra = cross - path_set
        ray = _compute_ray_cells([(tr, tc)], perp_dirs, H, W)
        if cross & used_cross:
            continue
        if ray & used_cross:
            continue
        if cross_extra & occupied or cross_extra & drawn:
            continue
        if ray & occupied or ray & drawn:
            continue
        if cross_extra & used_ray or ray & extra_cross or ray & used_ray:
            continue
        selected.append((tr, tc))
        used_cross.update(cross)
        used_ray.update(ray)
        extra_cross.update(cross_extra)

    if len(selected) < min_needed:
        return None

    selected_set = set(selected)
    return selected, extra_cross, used_ray, set(path[1:-1]) - selected_set


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

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]

        chosen = _choose_triggers(
            path,
            perp_dirs,
            H,
            W,
            occupied,
            drawn,
            must_pick_one=(total_triggers == 0),
        )
        if chosen is None:
            continue
        trigger_positions, extra_cross, ray_cells, path_fill = chosen

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(path_fill)
        drawn.update(extra_cross)
        drawn.update(ray_cells)
        total_triggers += len(trigger_positions)

    if not segments or total_triggers == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

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
