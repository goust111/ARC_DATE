# Variant of db695cfb
# Dimension: D7 — 交互触发对象 (obstacle)
# Change: 品红像素——位于主对角线路径上的品红像素触发垂直对角延伸... → 障碍物像素（阻断+延伸）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# a) Concrete behavior:
#    - Input still contains blue endpoint pairs on 45-degree diagonals and optional magenta pixels on those same paths.
#    - The original task filled the whole diagonal path and only used magenta as branching triggers.
#    - In this variant, the diagonal is traced from the upper endpoint toward the lower endpoint.
#      The first magenta encountered on that ordered path emits the same perpendicular rays as before
#      and also stops the blue trace immediately, so the lower tail of the diagonal remains blank.
#    - Small example on a backslash diagonal:
#      input:  blue at (1,1) and (6,6), magenta at (3,3)
#      original output: blue fills (2,2),(4,4),(5,5) and magenta rays branch from (3,3)
#      variant output:  blue fills only (2,2); cells (4,4),(5,5) stay background; rays still branch from (3,3)
#      The visible difference is therefore the missing continuation beyond the obstacle.
# b) Complexity calibration:
#    - To keep the new blocking rule legible, each segment is allowed at most one on-path obstacle.
#      With multiple obstacles on one ordered path, only the first one would matter, which weakens the causal story.
#    - We still allow multiple segments and both diagonal slopes so the rule is not overfit to one layout.
# c) Original assumptions that must change:
#    - Original `draw_segment()` scanned the whole path and never broke on a trigger. That assumes triggers are transparent.
#    - Original expected-output builder marked every background cell on the path blue. That also assumes transparency.
#    - Original generator allowed 0-2 triggers on a segment because every trigger on the path stayed relevant.
#      In the obstacle variant, later triggers on the same path would be hidden behind the first one, so generator logic changes.
# d) Placement legality:
#    - Segments reserve their entire diagonal corridor, not only the drawn cells. This prevents distractors from appearing
#      inside the blocked tail and keeps the obstacle effect readable.
#    - Rays are kept disjoint from other reserved structures so the solver never has to resolve ambiguous overlaps.
#    - Blue endpoints from different segments are forbidden from sharing a diagonal, so the solver's "longest diagonal pair"
#      still identifies the intended line color and intended pairing.
# e) Visual observability:
#    - Every generated puzzle contains at least one triggered segment whose obstacle is not adjacent to either endpoint.
#      That guarantees a visible blank tail behind the magenta obstacle, so the variant does not collapse back to the original task.
# f) Visual cleanliness:
#    - Full-path reservation plus non-overlapping rays avoid stacked markings in one region.
#    - Off-diagonal magenta distractors are kept away from any diagonal relation with other magenta pixels, so they remain inert.
# g) Color inference:
#    - Colors do not change roles from the original: blue still marks the diagonal family, magenta still marks the branching object.
#      The only new inference is that magenta also stops the main trace.
# h) New solver assumptions and generator support:
#    - Solver assumes the operative segment direction is "top endpoint to bottom endpoint"; generator always builds paths in that order.
#    - Solver assumes at most one relevant obstacle per segment; generator enforces that.
#    - Solver assumes rays can skip over non-background cells without overwriting; generator ensures no critical structure depends on overwrite order.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each blue diagonal from its upper endpoint, stopping at the first obstacle and branching there.

    Input structure:
    - The background is the most common color. One non-background color forms endpoint pairs on 45-degree diagonals, and magenta obstacle pixels may lie strictly between those endpoints on the same ordered diagonal path.

    Rule:
    - For every diagonal endpoint pair of the line color, draw the diagonal from the upper endpoint toward the lower one.
    - If that trace reaches a differently-colored pixel, do not pass through it. Instead, keep that obstacle pixel as-is and emit two perpendicular diagonal rays in the obstacle's color out to the grid boundary.

    Key insight:
    - Magenta is no longer a transparent trigger. It is a blocker-and-splitter: it marks where the blue diagonal terminates and where the perpendicular magenta rays begin.
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
                break

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _shares_diagonal(cell: tuple[int, int], positions: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in positions:
        if r != rr and abs(r - rr) == abs(c - cc):
            return True
    return False


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


def _compute_line_cells(path, trigger_positions):
    trigger_set = set(trigger_positions)
    cells = set()
    for r, c in path:
        if (r, c) in trigger_set:
            break
        cells.add((r, c))
    return cells


def _build_expected(inp, bg, segments):
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in _compute_line_cells(path, trigger_positions):
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, reserved):
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
        if any(rc in occupied or rc in reserved for rc in path):
            continue
        return path, dc
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    reserved = set()
    segments = []
    line_positions: list[tuple[int, int]] = []
    trigger_positions_all: list[tuple[int, int]] = []
    has_triggered_segment = False
    has_visible_block = False

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved)
        if result is None:
            continue
        path, dc = result
        ep0, ep1 = path[0], path[-1]

        if _shares_diagonal(ep0, line_positions) or _shares_diagonal(ep1, line_positions):
            continue

        # Visible blockers should leave at least one blue cell before the obstacle
        # and at least one blank tail cell behind it.
        visible_candidates = path[2:-2]
        trigger_positions = []
        if visible_candidates and (not has_triggered_segment or random.random() < 0.6):
            candidates = [cell for cell in visible_candidates if not _shares_diagonal(cell, trigger_positions_all)]
            if candidates:
                trigger_positions = [random.choice(candidates)]

        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if any(rc in occupied or rc in reserved for rc in ray_cells):
            continue

        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        reserved.update(path)
        reserved.update(ray_cells)
        line_positions.extend([ep0, ep1])
        trigger_positions_all.extend(trigger_positions)
        if trigger_positions:
            has_triggered_segment = True
            has_visible_block = True

    if not segments or not has_triggered_segment or not has_visible_block:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(line_positions)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if _shares_diagonal((r, c), all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_trigger_positions = list(trigger_positions_all)
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if _shares_diagonal((r, c), all_trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_trigger_positions.append((r, c))
            break

    if np.count_nonzero(inp != bg) > 18:
        return None

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
