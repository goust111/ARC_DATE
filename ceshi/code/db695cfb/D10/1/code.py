# Variant of db695cfb
# Dimension: D10 — 网格输出形状 (grid_shape)
# Change: 输出与输入同尺寸（same）——输出网格大小不变... → 扩展网格以容纳完整延伸

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D10 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     - Keep the original recognition task: blue endpoints define 45-degree segments,
#       magenta pixels lying on those segments trigger perpendicular diagonal rays.
#     - Replace "stop at the current grid border" with a finite, geometry-derived
#       completion rule: each trigger emits both perpendicular rays for exactly as
#       many steps as the length of its parent blue segment.
#     - If any of those completed rays leave the input rectangle, expand the canvas
#       by adding background rows/columns around the original grid. The original
#       input rectangle must remain intact inside the output; this is expansion,
#       not tight cropping.
#   concretization_and_trace:
#     - Example: a blue segment of length 5 runs from (1,1) to (6,6), and a magenta
#       trigger sits at (2,2). The original task would draw magenta only while the
#       coordinates stay inside the 8x8 input. The variant instead draws 5 cells in
#       each perpendicular direction: (1,3),(0,4),(-1,5),... and (3,1),(4,0),(5,-1),...
#       Cells with negative indices force top/left expansion. The output becomes the
#       smallest rectangle that still contains the full original 8x8 canvas plus all
#       these extra ray cells.
#   complexity_calibration:
#     - The added rule uses only one extra numeric degree of freedom: the ray reach
#       equals the already-visible blue segment length. This is observable from a few
#       examples and avoids hidden color-based branching.
#     - To keep the change visually legible, the generator guarantees at least one
#       trigger whose completed ray exits the input frame, so the larger output size
#       is not a rare accident.
#   implicit_assumption_audit:
#     - Original `out = grid.copy()` assumed all future drawing stayed in-bounds.
#       That fails once rays may extend beyond the input rectangle.
#     - Original ray loop `while 0 <= r < h and 0 <= c < w` used the old boundary as
#       the stopping condition. The variant needs a new stop rule independent of the
#       current canvas size.
#     - Original return value reused the input shape. The variant must instead render
#       in absolute coordinates first, then compute the expanded rectangle afterward.
#     - Segment detection by longest same-color diagonal pair still holds because the
#       generator preserves the original endpoint vocabulary and keeps line distractors
#       from forming diagonal pairs.
#   placement_legality:
#     - Generator tracks all endpoint/trigger cells (`occupied`) and all output-only
#       cells (`drawn`) in one absolute coordinate system that may include negatives.
#       A candidate segment is rejected if its interior fill or its completed rays
#       would overlap any previously reserved cell.
#     - Off-diagonal magenta distractors and unpaired blue distractors are placed only
#       on cells that are absent from both `occupied` and `drawn`, so solver painting
#       never overwrites input pixels.
#   visual_observability:
#     - Every generated puzzle has at least one genuine expansion event, so examples
#       do not silently collapse back to the original same-size behaviour.
#     - Ray reach depends on segment length, which is directly visible as the number
#       of blue diagonal steps between endpoints.
#   visual_cleanliness:
#     - Segment count and distractor count stay modest, and the generator forbids
#       overlaps between completed rays and other structures. This keeps expanded
#       outputs readable instead of producing dense diagonal tangles.
#   color_inference:
#     - Colors remain fixed to the original task roles: blue builds the diagonal
#       connectors, magenta marks triggers and the perpendicular completions. No
#       color-id ordering or hidden palette remapping is introduced.
#   main_assumptions_guaranteed_by_generator:
#     - At least one diagonal-aligned same-color endpoint pair exists.
#     - Any trigger that lies on a drawn segment belongs to exactly one generated
#       segment, so its segment length is unambiguous.
#     - Input non-background cells never sit on output-only fill cells, ensuring the
#       solver's "fill background only" logic matches the generator oracle.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _paint_expanded_canvas(
    base_shape: tuple[int, int],
    bg: int,
    line_color: int,
    seed_cells: dict[tuple[int, int], int],
    segment_paths: list[list[tuple[int, int]]],
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]],
) -> np.ndarray:
    h, w = base_shape
    painted = {(int(r), int(c)): int(v) for (r, c), v in seed_cells.items()}

    for path in segment_paths:
        for r, c in path:
            if painted.get((r, c), bg) == bg:
                painted[(r, c)] = int(line_color)

    for (r0, c0, tcol), dir_to_reach in trigger_reaches.items():
        for (ddr, ddc), reach in dir_to_reach.items():
            for step in range(1, int(reach) + 1):
                r = int(r0 + step * ddr)
                c = int(c0 + step * ddc)
                if painted.get((r, c), bg) == bg:
                    painted[(r, c)] = int(tcol)

    rows = [0, h - 1]
    cols = [0, w - 1]
    if painted:
        rows.extend(r for r, _ in painted)
        cols.extend(c for _, c in painted)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for (r, c), v in painted.items():
        out[r - min_r, c - min_c] = int(v)
    return out


def _compute_full_ray_cells(
    triggers: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    reach: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            for step in range(1, reach + 1):
                cells.add((tr + step * ddr, tc + step * ddc))
    return cells


def _ray_exits_input(
    trigger: tuple[int, int],
    perp_dirs: list[tuple[int, int]],
    reach: int,
    h: int,
    w: int,
) -> bool:
    tr, tc = trigger
    for ddr, ddc in perp_dirs:
        for step in range(1, reach + 1):
            r = tr + step * ddr
            c = tc + step * ddc
            if not (0 <= r < h and 0 <= c < w):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs and complete trigger rays on an expanded canvas.

    Input structure:
    - The background is the most common color. One non-background color forms endpoint pairs on 45-degree diagonals, and differently-colored trigger pixels may lie on those same diagonals.

    Rule:
    - Fill each diagonal segment between the extreme same-color endpoints.
    - Whenever such a segment passes through a trigger pixel, emit the perpendicular diagonal rays in the trigger color.
    - Each perpendicular ray runs for as many steps as the blue segment length; if those completed rays leave the input rectangle, enlarge the output grid with background padding so the full result fits.

    Key insight:
    - The trigger does not use the current border as its stopping signal anymore; its full reach is determined by the visible length of the segment that activates it, and the canvas expands afterward to hold that completed geometry.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    seed_cells: dict[tuple[int, int], int] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))
        seed_cells[(r, c)] = int(col)

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

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segment_paths: list[list[tuple[int, int]]] = []
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        segment_paths.append(path)
        seg_len = len(path) - 1
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                key = (r, c, v)
                dir_to_reach = trigger_reaches.setdefault(key, {})
                for direction in perp_dirs:
                    dir_to_reach[direction] = max(dir_to_reach.get(direction, 0), seg_len)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    return _paint_expanded_canvas((h, w), bg, line_color, seed_cells, segment_paths, trigger_reaches)


# --------------- generator (D10 variant) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 22)
_W_RANGE = (9, 20)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (3, 12)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    h, w = inp.shape
    seed_cells = {
        (int(r), int(c)): int(inp[r, c])
        for r in range(h)
        for c in range(w)
        if int(inp[r, c]) != int(bg)
    }
    segment_paths = [path for _, _, _, path, _, _ in segments]
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}
    for _, _, trigger_positions, path, perp_dirs, ray_len in segments:
        for tr, tc in trigger_positions:
            key = (int(tr), int(tc), int(_TRIGGER_COLOR))
            dir_to_reach = trigger_reaches.setdefault(key, {})
            for direction in perp_dirs:
                dir_to_reach[direction] = max(dir_to_reach.get(direction, 0), int(ray_len))
    return _paint_expanded_canvas((h, w), bg, _LINE_COLOR, seed_cells, segment_paths, trigger_reaches)


def _try_place_segment(slope_type, seg_len, h, w, occupied, drawn):
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
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments = []
    forced_expansion_satisfied = False

    for seg_index in range(n_segments):
        placed = False
        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(["backslash", "slash"])
            seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
            result = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn)
            if result is None:
                continue
            path, dc = result
            path_interior = path[1:-1]
            if not path_interior:
                continue

            perp_dirs = [(1, -dc), (-1, dc)]
            wants_forced = seg_index == 0 and not forced_expansion_satisfied
            trigger_cap = min(_TRIGGERS_PER_SEG[1], len(path_interior))
            if wants_forced and trigger_cap == 0:
                continue

            expanding_candidates = [
                pt for pt in path_interior if _ray_exits_input(pt, perp_dirs, seg_len, h, w)
            ]
            if wants_forced and not expanding_candidates:
                continue

            min_triggers = 1 if wants_forced else _TRIGGERS_PER_SEG[0]
            if trigger_cap < min_triggers:
                continue
            n_triggers = random.randint(min_triggers, trigger_cap)

            trigger_positions = []
            remaining = list(path_interior)
            if wants_forced:
                chosen = random.choice(expanding_candidates)
                trigger_positions.append(chosen)
                remaining.remove(chosen)
            if n_triggers > len(trigger_positions):
                extra = random.sample(remaining, n_triggers - len(trigger_positions))
                trigger_positions.extend(extra)

            segment_fill = set(path[1:-1]) - set(trigger_positions)
            ray_cells = _compute_full_ray_cells(trigger_positions, perp_dirs, seg_len)
            if any(rc in occupied or rc in drawn or rc in segment_fill for rc in ray_cells):
                continue
            if any(rc in occupied or rc in drawn for rc in segment_fill):
                continue

            ep0, ep1 = path[0], path[-1]
            segments.append((ep0, ep1, trigger_positions, path, perp_dirs, seg_len))
            occupied.update([ep0, ep1])
            occupied.update(trigger_positions)
            drawn.update(segment_fill)
            drawn.update(ray_cells)
            if any(_ray_exits_input(tp, perp_dirs, seg_len, h, w) for tp in trigger_positions):
                forced_expansion_satisfied = True
            placed = True
            break

        if not placed and seg_index == 0:
            return None

    if not segments or not forced_expansion_satisfied:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for ep0, ep1, trigger_positions, _path, _perp_dirs, _ray_len in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
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
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
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
    if result.shape == inp.shape:
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
