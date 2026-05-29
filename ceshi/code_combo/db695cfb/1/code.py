# Combo variant of db695cfb (index 1)
# combo_spec: D1/0+D2/1
# Name: 直线路径+正交延伸

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# combo_summary:
#   - The explicit combo name and component list are "straight path + orthogonal
#     extension" (D1/0 + D2/1), so this implementation follows that rule even
#     though one prose sentence in the prompt briefly mentions diagonal
#     extension. The component specs are more precise and mutually consistent:
#       * backbone: horizontal or vertical segment between blue endpoints
#       * trigger effect: 4-way orthogonal cross from each on-path magenta pixel
# a) concretization_and_trace:
#   - Original local pattern:
#       blue at (1,1) and (4,4), magenta at (2,2)
#       -> fill diagonal (1,1)(2,2)(3,3)(4,4)
#       -> from (2,2) emit the perpendicular diagonal pair.
#   - Combo local pattern:
#       blue at (3,1) and (3,5), magenta at (3,3)
#       -> fill row 3 from col 1..5 in blue where cells are background
#       -> detect the magenta trigger on that row
#       -> emit a 4-way orthogonal cross from (3,3).
#   - Important fusion interaction:
#       for a horizontal backbone, the left/right arms of the cross travel along
#       the same row as the backbone. Because the solver only paints background
#       cells, the magenta ray skips the blue segment cells and becomes visible
#       only outside the blue endpoints; the up/down arms remain fully visible.
#       The vertical case is the symmetric analogue.
# b) complexity_calibration:
#   - Orthogonal crosses occupy much more area than the original diagonal rays,
#     so generation is tightened to 1-3 segments and at most 1 real trigger per
#     segment, with at least 1 real trigger overall so the fused behavior is
#     always observable.
#   - Blue distractors and magenta distractors remain, preserving the need to
#     infer the acting color and ignore inert same-color singletons.
# c) implicit_assumptions_review:
#   - `diag_path` and diagonal grouping are no longer valid. They are replaced by
#     same-row / same-column path construction and grouping.
#   - The original trigger-direction logic depended on diagonal slope. That
#     assumption is removed: the combo branch rule is orientation-invariant and
#     always uses up/down/left/right.
#   - The original generator could reject any ray/path overlap. That is wrong
#     here because a cross can legitimately run along the same axis as its own
#     backbone. The generator must reserve only cells that will actually be
#     newly painted, not backbone cells that rays merely pass through.
# d) placement_legality:
#   - Blue input pixels are kept pair-free across unrelated rows, columns, and
#     diagonals. This prevents accidental extra segments and also prevents the
#     sample from looking solvable by the original diagonal rule.
#   - Magenta input pixels are also kept pair-free across rows, columns, and
#     diagonals. This prevents trigger color from being mistaken as the line
#     color by either the new axis-based solver or the old diagonal solver.
#   - Every new backbone cell and every newly painted cross cell must avoid
#     earlier occupied/drawn cells, so the intended output is unambiguous.
# e) visual_observability:
#   - In 2-4 examples a human can see that blue endpoints align on rows/columns,
#     not diagonals, and that magenta spreads as full row/column crosses rather
#     than diagonal arms. At least one real trigger is forced so the D2 change is
#     always visible.
# f) cleanliness:
#   - Cross-fill cells are kept disjoint from previously reserved output cells.
#     This avoids dense overlaps between multiple branches and keeps the
#     blue-backbone / magenta-cross decomposition readable.
# g) inferability_of_change:
#   - Only two semantic foreground colors are used: blue for endpoint/backbone
#     and magenta for trigger/branch. The rule is therefore carried by geometry,
#     not hidden color-id conventions.
#
# New solver assumptions that the generator explicitly enforces:
#   - Exactly one foreground color has a same-row or same-column pair long enough
#     to win the "line color" search.
#   - Every real trigger lies on a generated backbone segment.
#   - Trigger pixels never form their own same-row/same-column pair, so they
#     cannot be selected as the line color.


_ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color straight endpoint pairs, then emit orthogonal cross rays from differently-colored pixels on those paths.

    Input structure:
    - The background is the most common color. One foreground color marks endpoint pairs that share a row or a column, while trigger-colored pixels may sit strictly between those endpoints on the same straight segment.

    Rule:
    - Fill each horizontal or vertical segment between the extreme same-color endpoints.
    - Whenever that filled segment passes through a differently-colored trigger pixel, extend that trigger color in all four orthogonal directions until the grid boundary, only painting background cells.

    Key insight:
    - Blue identifies the axis-aligned backbone, and magenta trigger pixels convert that backbone into a cross-emitter: the cross keeps propagating through the row/column geometry even when some cells on its route are already occupied by the blue segment.
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

    def straight_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            return [(r0, c) for c in range(lo, hi + 1)]
        if c0 == c1:
            lo, hi = sorted((r0, r1))
            return [(r, c0) for r in range(lo, hi + 1)]
        raise ValueError("points must share a row or column")

    best_dist = -1
    line_color: int | None = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                if (r0 == r1) == (c0 == c1):
                    continue
                dist = abs(r1 - r0) + abs(c1 - c0)
                if dist > best_dist:
                    best_dist = dist
                    line_color = col

    assert line_color is not None, "at least one straight-aligned same-color pair must exist"

    pts = by_color[line_color]

    triggers: set[tuple[int, int, int]] = set()

    row_groups: dict[int, list[tuple[int, int]]] = {}
    col_groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        row_groups.setdefault(r, []).append((r, c))
        col_groups.setdefault(c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        for r, c in straight_path(p, q):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.add((r, c, v))

    for group in row_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[1])
            draw_segment(group[0], group[-1])

    for group in col_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[0])
            draw_segment(group[0], group[-1])

    for r0, c0, tcol in triggers:
        for ddr, ddc in _ORTHO_DIRS:
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
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_SEG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _straight_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    if p[0] == q[0]:
        lo, hi = sorted((p[1], q[1]))
        return [(p[0], c) for c in range(lo, hi + 1)]
    lo, hi = sorted((p[0], q[0]))
    return [(r, p[1]) for r in range(lo, hi + 1)]


def _shares_axis_or_diagonal(cell: tuple[int, int], others: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(r == rr or c == cc or abs(r - rr) == abs(c - cc) for rr, cc in others)


def _cross_fill_cells(trigger: tuple[int, int], path_cells: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    tr, tc = trigger
    cells: set[tuple[int, int]] = set()
    for c in range(W):
        if c == tc:
            continue
        cell = (tr, c)
        if cell not in path_cells:
            cells.add(cell)
    for r in range(H):
        if r == tr:
            continue
        cell = (r, tc)
        if cell not in path_cells:
            cells.add(cell)
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, _, trigger_positions, path in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in _ORTHO_DIRS:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(orient: str, seg_len: int, H: int, W: int, occupied, drawn, line_positions):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if orient == "h":
            if W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1)
            c0 = random.randint(0, W - 1 - seg_len)
            p = (r0, c0)
            q = (r0, c0 + seg_len)
        else:
            if H - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1)
            p = (r0, c0)
            q = (r0 + seg_len, c0)

        path = _straight_path(p, q)
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        ep0, ep1 = path[0], path[-1]
        if _shares_axis_or_diagonal(ep0, line_positions) or _shares_axis_or_diagonal(ep1, line_positions):
            continue

        return path
    return None


def _allowed_trigger_positions(path_interior, existing_triggers):
    return [pos for pos in path_interior if not _shares_axis_or_diagonal(pos, existing_triggers)]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(max(H - 1, 0), max(W - 1, 0), _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    line_positions: list[tuple[int, int]] = []
    trigger_positions: list[tuple[int, int]] = []
    segments = []

    for _ in range(n_segments):
        orient = random.choice(["h", "v"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        path = _try_place_segment(orient, seg_len, H, W, occupied, drawn, line_positions)
        if path is None:
            continue
        ep0, ep1 = path[0], path[-1]
        path_set = set(path)

        path_interior = path[1:-1]
        allowed_triggers = _allowed_trigger_positions(path_interior, trigger_positions)
        if (orient == "h" and len(path) == W) or (orient == "v" and len(path) == H):
            allowed_triggers = []

        min_triggers = 1 if not trigger_positions else _TRIGGERS_PER_SEG[0]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(allowed_triggers))
        if max_triggers < min_triggers:
            continue

        n_triggers = random.randint(min_triggers, max_triggers)
        chosen_triggers = random.sample(allowed_triggers, n_triggers) if n_triggers > 0 else []

        cross_cells = set()
        for tp in chosen_triggers:
            cross_cells.update(_cross_fill_cells(tp, path_set, H, W))

        if any(rc in occupied or rc in drawn for rc in cross_cells):
            continue

        segments.append((ep0, ep1, orient, chosen_triggers, path))
        occupied.update([ep0, ep1])
        occupied.update(chosen_triggers)
        drawn.update(set(path_interior) - set(chosen_triggers))
        drawn.update(cross_cells)
        line_positions.extend([ep0, ep1])
        trigger_positions.extend(chosen_triggers)

    if not segments or not trigger_positions:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, orient, chosen_triggers, path in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in chosen_triggers:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(line_positions)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            pos = (r, c)
            if pos in occupied or pos in drawn:
                continue
            if _shares_axis_or_diagonal(pos, all_line_positions):
                continue
            inp[pos] = _LINE_COLOR
            occupied.add(pos)
            all_line_positions.append(pos)
            break

    all_trigger_positions = list(trigger_positions)
    for _ in range(random.randint(*_OFF_SEG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            pos = (r, c)
            if pos in occupied or pos in drawn:
                continue
            if _shares_axis_or_diagonal(pos, all_trigger_positions):
                continue
            inp[pos] = _TRIGGER_COLOR
            occupied.add(pos)
            all_trigger_positions.append(pos)
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
