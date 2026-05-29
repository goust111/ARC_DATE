# Combo variant of db695cfb (index 10)
# combo_spec: D10/0+D13/0
# Name: 裁剪输出+擦除触发

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# combo_behavior:
#   core_change:
#     - Keep the original recognition rule: find the non-background color whose
#       diagonal-aligned pair is longest, connect the extreme points on each of
#       its occupied diagonals, and treat differently-colored cells on those
#       paths as ray triggers.
#     - Merge two changes in the actual rendering:
#       1) a trigger is consumable, so after it launches the perpendicular rays
#          its own cell becomes background instead of remaining magenta;
#       2) the final answer is cropped to the smallest rectangle containing the
#          surviving construction cells, not the entire input canvas.
#   concrete_behavior_and_trace:
#     - Example: in an 8x10 grid, blue endpoints at (2,2) and (5,5) define a
#       backslash segment and a magenta trigger at (3,3) sits on that segment.
#       Original task output keeps the full canvas, fills the diagonal, keeps
#       (3,3) magenta, and emits NE/SW magenta rays from that trigger.
#     - In this combo, the same diagonal is recognized, but after the rays are
#       emitted the trigger cell itself is erased back to background. The blue
#       segment therefore has a visible hole at (3,3). The returned grid is
#       then cropped to the bbox of the surviving blue cells plus any magenta
#       ray cells. Distractors outside that bbox disappear entirely.
#     - For this source task, the erased trigger is usually interior to the
#       segment, so endpoints or ray tips often still determine the bbox. The
#       crop must nevertheless be computed from surviving visible cells, not
#       from pre-erase geometry, because the trigger cell is no longer part of
#       the output object.
#   complexity_calibration:
#     - Every generated instance must contain at least one real trigger;
#       otherwise the combo would collapse to the crop-only D10 behavior.
#     - Instances whose active construction already spans the full canvas are
#       rejected so the crop is visually observable.
#     - Distractors stay sparse and use the original palette, so the new burden
#       is understanding "trigger gets consumed" plus "frame follows the final
#       visible drawing", not decoding extra symbolic clutter.
#   implicit_assumption_review:
#     - The line-color heuristic still assumes only the true endpoint color owns
#       the longest diagonal pair. The generator preserves that by construction
#       and by oracle-checking candidate inputs.
#     - `draw_segment` assumes any non-background non-line cell on a segment is
#       a trigger. The generator forbids other occupied cells from landing on
#       segment interiors.
#     - Cropping can no longer use all traversed cells, because erased triggers
#       must not keep dead space in-frame. We therefore track only cells that
#       remain visibly non-background in the final construction.
#     - At least one segment always exists, and endpoints always survive, so the
#       visible-cell set used for cropping is guaranteed non-empty.
#   placement_legality:
#     - Segment paths cannot overlap prior occupied cells or prior drawn-only
#       cells. This prevents ambiguous multi-object reuse of a single interior.
#     - Ray trajectories cannot hit occupied input pixels. They may overlap
#       earlier drawn-only cells, which is harmless because writes only occur on
#       background cells and visible geometry is still well defined.
#     - Unpaired blue distractors are screened so they do not create competing
#       blue diagonals, and off-diagonal magenta distractors avoid all segment
#       and ray cells.
#   visual_observability_and_cleanliness:
#     - The generator prefers outside-bbox distractors so cropping is easy to
#       notice from a few examples.
#     - Trigger count stays low (1-2 per segment), which makes the erased holes
#       legible instead of turning the diagonal into noise.
#     - Because trigger cells vanish instead of being recolored, there is no
#       stacking ambiguity at those sites: solvers should read them as missing
#       pixels that launched rays, not as preserved anchors.
#   solver_assumption_audit:
#     - The segment grouping by `r-c` and `r+c` remains valid because the combo
#       does not change how true blue endpoints are encoded.
#     - Ray directions still depend only on the orientation of the segment that
#       encountered the trigger; the erase step happens after those directions
#       are collected, so no causal information is lost.
#     - Cropping by surviving construction cells means inside-bbox distractors
#       may still remain in the returned slice, but they never influence the bbox.


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_grid(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

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
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    visible_cells: set[tuple[int, int]] = set()

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
                visible_cells.add((r, c))
            elif v == line_color:
                visible_cells.add((r, c))
            else:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        out[r0, c0] = bg
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                    visible_cells.add((r, c))
                r += ddr
                c += ddc

    assert visible_cells, "at least one segment must contribute visible cells"
    return out, visible_cells, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the same diagonal pairs as the source task, but consume each trigger and return only the tight final drawing.

    Input structure:
    - The most common color is background. One non-background color marks endpoint pairs on shared 45-degree diagonals, and differently-colored pixels may sit on those diagonals as triggers. Extra distractor pixels can appear elsewhere.

    Rule:
    - Connect each extreme same-color pair on a shared diagonal.
    - Whenever a connecting diagonal encounters a trigger pixel, use that pixel to launch two rays in the perpendicular diagonal direction until they leave the canvas, then erase the trigger itself back to background.
    - After all drawing is finished, crop the result to the smallest rectangle containing the surviving constructed cells.

    Key insight:
    - The endpoint geometry still identifies the important diagonals, but trigger pixels are no longer preserved landmarks; they are temporary activators that leave holes, and the output frame follows only what remains visible after they are consumed.
    """

    out, visible_cells, _ = _solve_core(grid)
    return _crop_grid(out, _bbox_from_cells(visible_cells))


# --------------- generator (combo variant) ---------------

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


def _compute_ray_cells(
    triggers: list[tuple[int, int]], perp_dirs: list[tuple[int, int]], H: int, W: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[
        tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]
    ],
) -> tuple[np.ndarray, tuple[int, int, int, int], set[tuple[int, int]], int]:
    H, W = inp.shape
    expected = inp.copy()
    visible_cells: set[tuple[int, int]] = set()
    trigger_count = 0

    for _, _, trigger_positions, path, perp_dirs in segments:
        trigger_set = set(trigger_positions)
        trigger_count += len(trigger_positions)

        for r, c in path:
            if (r, c) in trigger_set:
                continue
            visible_cells.add((r, c))
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR

        for tr, tc in trigger_positions:
            expected[tr, tc] = bg
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                        visible_cells.add((r, c))
                    r += ddr
                    c += ddc

    bbox = _bbox_from_cells(visible_cells)
    return _crop_grid(expected, bbox), bbox, visible_cells, trigger_count


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
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


def _bbox_is_full(bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
    return bbox == (0, 0, H - 1, W - 1)


def _in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _place_distractor(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    color: int,
    prefer_outside: bool,
) -> tuple[int, int] | None:
    H, W = inp.shape
    for _ in range(_DISTRACTOR_RETRIES):
        r, c = random.randint(0, H - 1), random.randint(0, W - 1)
        if (r, c) in occupied or (r, c) in drawn:
            continue
        if prefer_outside and _in_bbox((r, c), bbox):
            continue
        inp[r, c] = color
        occupied.add((r, c))
        return (r, c)
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[
        tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]
    ] = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 and path_interior else []

        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if any(rc in occupied for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    expected, bbox, _, trigger_count = _build_expected(inp, bg, segments)
    if trigger_count == 0:
        return None
    if _bbox_is_full(bbox, H, W):
        return None

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    want_outside_line = random.random() < 0.7
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if want_outside_line and _in_bbox((r, c), bbox):
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    want_outside_trigger = random.random() < 0.7
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        _place_distractor(inp, occupied, drawn, bbox, _TRIGGER_COLOR, want_outside_trigger)

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
