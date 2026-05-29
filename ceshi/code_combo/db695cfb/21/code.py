# Combo variant of db695cfb (index 21)
# combo_spec: D9/1+D13/2
# Name: 先延伸后连接+十字标记

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the fused variant
# =====================================
# a) Concrete behavior trace
#    - Original db695cfb:
#      blue endpoints first determine a hidden diagonal segment; only after that segment is drawn
#      do magenta cells lying on it emit perpendicular diagonal rays.
#    - Single D13/2:
#      those active magenta cells become a 3x3 diagonal X:
#      {(r,c), (r-1,c-1), (r-1,c+1), (r+1,c-1), (r+1,c+1)},
#      and the perpendicular rays begin beyond the two perpendicular X arms.
#    - Fused D9/1 + D13/2:
#      1. Read the future blue segment geometry from the blue endpoints, but do not draw it yet.
#      2. Any non-blue cell on that future segment is an active trigger. First expand it into the
#         3x3 X and emit the perpendicular rays from the X's perpendicular arms.
#      3. Only after all magenta structure exists, draw the blue diagonal segment on top of it.
#    - Local example:
#      blue endpoints at (2,2) and (6,6), trigger at (4,4).
#      D13 alone would finish with a full magenta X centered at (4,4).
#      The fused rule first paints that X and the "/" rays, then redraws the "\" blue segment.
#      Therefore (3,3), (4,4), (5,5) end blue again, while (3,5) and (5,3) stay magenta.
#      The visible output is a blue main diagonal with a magenta perpendicular junction attached.
#
# b) Complexity calibration
#    - The marker stays binary in spirit: blue still means backbone, magenta still means trigger.
#      The only added geometric state is whether magenta was painted before or after blue.
#    - Trigger counts are kept small and at least one active trigger is required globally, so the
#      "blue-overwrites-the-X-center" effect is always visible and does not drown in clutter.
#    - The 3x3 marker is taken to be a diagonal X, not a plus sign, because the prompt says the
#      outgoing rays start from the marker's diagonal endpoints; a plus sign would not supply them.
#
# c) Implicit-assumption audit
#    - Original assumption broken: the solver could discover triggers while drawing the blue path.
#      That is no longer valid because the blue path is now the final step. The solver must first
#      infer the future segments from the input and collect active triggers directly from `grid`.
#    - Original assumption broken: blue only filled background cells. In this combo the later blue
#      stroke must overwrite earlier magenta paint on the main diagonal, otherwise D9/1 is inert.
#    - D13 still relies on a full 3x3 X existing before rays start, so the trigger must stay at
#      least two cells away from borders and endpoints in generated samples.
#    - Off-path magenta distractors remain inert. They have no inferred host segment, so no
#      perpendicular orientation can be assigned to them. The generator preserves this separation.
#
# d) Placement legality
#    - Each active trigger is chosen from `path[2:-2]`, guaranteeing room for the two same-diagonal
#      X arms that will later be repainted blue and for the two perpendicular arms that remain
#      magenta.
#    - The perpendicular X arms and the post-X ray cells are reserved as output-only cells, so no
#      input endpoint, distractor, or other segment may occupy them.
#    - Blue segment paths are kept mutually disjoint, avoiding ambiguous multi-segment crossings.
#    - Trigger-color distractors are kept off all planned segments/rays and are also kept pair-free
#      on diagonals, so trigger color cannot accidentally win line-color detection.
#
# e) Visual observability
#    - Every accepted sample contains at least one active trigger, so the output visibly differs
#      from both the original task and the blue-first D13 variant.
#    - A human can see the key repeated pattern: an input magenta point on the future blue segment
#      does not stay a center pixel. Instead, blue runs through the center and same-diagonal arms,
#      while magenta survives only on the perpendicular diagonal and the launched rays.
#    - Off-path magenta distractors help reveal that not every magenta pixel activates.
#
# f) Visual cleanliness
#    - The generator keeps 1-3 blue segments, at most one active trigger per segment, and only a
#      few distractors. This avoids dense overdraw and keeps the two diagonal families readable.
#    - Because segments never cross each other and rays never hit input pixels, the only intended
#      local overlap is the trigger junction itself, which is exactly the concept the combo wants.
#
# g) Inferability of the change
#    - No hidden color ordering is used. The rule is supported by visible geometry: endpoint pairs,
#      X-shaped local expansion, then blue repaint along one diagonal.
#    - The generator explicitly rejects degenerate samples whose output would match the blue-first
#      D13 variant, so both component dimensions materially affect the final result.
#
# New solver assumptions that generate_input() explicitly guarantees:
#   - There is at least one same-color diagonal blue pair long enough to identify the line color.
#   - Every active trigger lies on exactly one planned blue segment and has full X geometry in-bounds.
#   - Trigger-color pixels never form their own diagonal pair, preventing misclassification.
#   - No planned ray or perpendicular X arm collides with any input pixel or earlier reserved stroke.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (4, 16)
_TRIGGER_PROB = 0.75

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


Coord = tuple[int, int]


def _diag_path(p: Coord, q: Coord) -> list[Coord]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _cross_cells(r: int, c: int) -> set[Coord]:
    """Return the 3x3 diagonal X centered at (r, c)."""
    return {
        (r, c),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    }


def _shares_diagonal(a: Coord, b: Coord) -> bool:
    return a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _find_line_color(non_bg: list[tuple[int, int, int]]) -> int:
    by_color: dict[int, list[Coord]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best: tuple[int, int] | None = None
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
    return best[1]


def _segments_from_line_points(points: list[Coord]) -> list[dict[str, object]]:
    groups_a: dict[int, list[Coord]] = {}
    groups_b: dict[int, list[Coord]] = {}
    for r, c in points:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[dict[str, object]] = []
    for group in groups_a.values():
        if len(group) < 2:
            continue
        ordered = sorted(group)
        p, q = ordered[0], ordered[-1]
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        segments.append(
            {
                "path": _diag_path(p, q),
                "perp_dirs": [(dr, -dc), (-dr, dc)],
            }
        )

    for group in groups_b.values():
        if len(group) < 2:
            continue
        ordered = sorted(group)
        p, q = ordered[0], ordered[-1]
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        segments.append(
            {
                "path": _diag_path(p, q),
                "perp_dirs": [(dr, -dc), (-dr, dc)],
            }
        )

    return segments


def _collect_structure(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]], dict[tuple[int, int, int], set[Coord]]]:
    h, w = grid.shape
    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    line_color = _find_line_color(non_bg)
    line_points = [(r, c) for r, c, col in non_bg if col == line_color]
    segments = _segments_from_line_points(line_points)

    triggers: dict[tuple[int, int, int], set[Coord]] = {}
    for seg in segments:
        for r, c in seg["path"]:
            value = int(grid[r, c])
            if value != bg and value != line_color:
                triggers.setdefault((r, c, value), set()).update(seg["perp_dirs"])

    return bg, line_color, segments, triggers


def _paint_rays(out: np.ndarray, triggers: dict[tuple[int, int, int], set[Coord]], bg: int) -> None:
    h, w = out.shape
    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + 2 * ddr, c0 + 2 * ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc


def _blue_first_x_output(grid: np.ndarray) -> np.ndarray:
    """Single-variant D13/2 behavior, used only to reject degenerate combo samples."""
    g = np.asarray(grid, dtype=int)
    bg, line_color, segments, triggers = _collect_structure(g)
    out = g.copy()
    h, w = out.shape

    for seg in segments:
        for r, c in seg["path"]:
            if int(out[r, c]) == bg:
                out[r, c] = line_color

    for r0, c0, tcol in triggers:
        for rr, cc in _cross_cells(r0, c0):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = tcol

    _paint_rays(out, triggers, bg)
    return out


def _order_difference_metric(grid: np.ndarray) -> int:
    """Cells changed specifically by reversing the processing order after adding X markers."""
    fused = main(grid)
    baseline = _blue_first_x_output(grid)
    return int(np.count_nonzero(fused != baseline))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand on-path trigger pixels into diagonal X junctions and perpendicular rays first, then redraw the blue diagonals over them.

    Input structure:
    - The background is the majority color. One non-background color appears as diagonal endpoint pairs that imply one or more hidden 45-degree blue segments.
    - Some differently colored pixels may lie on those future segments. Those are the only active triggers; off-segment magenta pixels can still appear as inert distractors.

    Rule:
    - Infer each blue diagonal segment from the extreme same-color endpoints, but before drawing it, turn every trigger on that segment into a 3x3 diagonal X in its own color.
    - From the two X arms that lie on the perpendicular diagonal, continue rays outward to the grid boundary, filling only background cells.
    - After all trigger expansions and rays are in place, redraw every blue diagonal segment on top of the result so the blue path overwrites any magenta cells it passes through.

    Key insight:
    - The active magenta pixels are still detected by future blue-segment membership, but the construction order is reversed: magenta grows first and the blue backbone is the last paint layer, so only the perpendicular part of each X remains visible.
    """

    g = np.asarray(grid, dtype=int)
    bg, line_color, segments, triggers = _collect_structure(g)
    out = g.copy()
    h, w = out.shape

    for r0, c0, tcol in triggers:
        for rr, cc in _cross_cells(r0, c0):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = tcol

    _paint_rays(out, triggers, bg)

    for seg in segments:
        for r, c in seg["path"]:
            out[r, c] = line_color

    return out


def _trigger_is_safe_cell(r: int, c: int, h: int, w: int) -> bool:
    return 2 <= r < h - 2 and 2 <= c < w - 2


def _compute_ray_cells(trigger_positions: list[Coord], perp_dirs: list[Coord], h: int, w: int) -> set[Coord]:
    cells: set[Coord] = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + 2 * ddr, tc + 2 * ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()

    for seg in segments:
        for tr, tc in seg["trigger_positions"]:
            for r, c in _cross_cells(tr, tc):
                if 0 <= r < h and 0 <= c < w:
                    expected[r, c] = _TRIGGER_COLOR

    for seg in segments:
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + 2 * ddr, tc + 2 * ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc

    for seg in segments:
        for r, c in seg["path"]:
            expected[r, c] = _LINE_COLOR

    return expected


def _build_blue_first_expected(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()

    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR

    for seg in segments:
        for tr, tc in seg["trigger_positions"]:
            for r, c in _cross_cells(tr, tc):
                if 0 <= r < h and 0 <= c < w:
                    expected[r, c] = _TRIGGER_COLOR

    for seg in segments:
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + 2 * ddr, tc + 2 * ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc

    return expected


def _try_place_segment(slope_type: str, seg_len: int, h: int, w: int, occupied: set[Coord], drawn: set[Coord]) -> tuple[list[Coord], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if seg_len >= h or seg_len >= w:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            dc = 1
        else:
            if seg_len >= h or seg_len >= w:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(cell in occupied or cell in drawn for cell in path):
            continue
        return path, dc
    return None


def _choose_trigger(
    path: list[Coord],
    perp_dirs: list[Coord],
    h: int,
    w: int,
    occupied: set[Coord],
    drawn: set[Coord],
    existing_triggers: list[Coord],
    must_pick_one: bool,
) -> tuple[list[Coord], set[Coord], set[Coord], set[Coord]] | None:
    path_set = set(path)
    candidates = [
        p for p in path[2:-2]
        if _trigger_is_safe_cell(p[0], p[1], h, w)
        and not any(_shares_diagonal(p, other) for other in existing_triggers)
    ]
    random.shuffle(candidates)

    chosen: Coord | None = None
    if candidates and (must_pick_one or random.random() < _TRIGGER_PROB):
        for tr, tc in candidates:
            cross_extra = _cross_cells(tr, tc) - path_set
            ray_cells = _compute_ray_cells([(tr, tc)], perp_dirs, h, w)
            if cross_extra & occupied or cross_extra & drawn:
                continue
            if ray_cells & occupied or ray_cells & drawn:
                continue
            if cross_extra & ray_cells:
                continue
            chosen = (tr, tc)
            break
        if must_pick_one and chosen is None:
            return None

    trigger_positions = [chosen] if chosen is not None else []
    extra_cross = (_cross_cells(*chosen) - path_set) if chosen is not None else set()
    ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, h, w)
    path_fill = set(path[1:-1]) - set(trigger_positions)
    return trigger_positions, extra_cross, ray_cells, path_fill


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[Coord] = set()
    drawn: set[Coord] = set()
    segments: list[dict[str, object]] = []
    active_triggers: list[Coord] = []

    for _ in range(random.randint(*_SEGMENT_COUNT)):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        placed = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn)
        if placed is None:
            continue

        path, dc = placed
        perp_dirs = [(1, -dc), (-1, dc)]
        chosen = _choose_trigger(
            path,
            perp_dirs,
            h,
            w,
            occupied,
            drawn,
            active_triggers,
            must_pick_one=(len(active_triggers) == 0),
        )
        if chosen is None:
            continue
        trigger_positions, extra_cross, ray_cells, path_fill = chosen

        ep0, ep1 = path[0], path[-1]
        segments.append(
            {
                "ep0": ep0,
                "ep1": ep1,
                "path": path,
                "perp_dirs": perp_dirs,
                "trigger_positions": trigger_positions,
            }
        )
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(path_fill)
        drawn.update(extra_cross)
        drawn.update(ray_cells)
        active_triggers.extend(trigger_positions)

    if not segments or not active_triggers:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for tr in seg["trigger_positions"]:
            inp[tr] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_shares_diagonal((r, c), line_cell) for line_cell in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    trigger_positions = list(active_triggers)
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_shares_diagonal((r, c), trg) for trg in trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            trigger_positions.append((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if not np.array_equal(result, expected):
        return None

    blue_first = _build_blue_first_expected(inp, bg, segments)
    if np.array_equal(expected, blue_first):
        return None

    if _order_difference_metric(inp) <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
