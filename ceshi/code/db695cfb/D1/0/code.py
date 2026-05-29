# Variant of db695cfb
# Dimension: D1 — 主路径类型 (path)
# Change: 对角线连接（diagonal_line）——在两个蓝色像素之间绘制严格45度对角线... → 直线连接（水平/垂直）

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the D1 path variant
# =======================================
# variant_spec:
#   main_change:
#     - The original task used same-color endpoint pairs on 45-degree diagonals.
#     - This variant keeps the "connect endpoints, then trigger perpendicular rays"
#       template but swaps the carrier path to a straight horizontal or vertical segment.
#   concrete_geometry:
#     - Input still contains blue endpoint pixels (color 1) and optional magenta trigger
#       pixels (color 6) on the would-be carrier path.
#     - Output fills the whole row-segment or column-segment between the blue endpoints.
#     - Any magenta pixel touched by that filled segment emits two orthogonal rays:
#       vertical if the segment is horizontal, horizontal if the segment is vertical.
#   behavior_trace:
#     - Original local pattern:
#         endpoints at (1,1) and (4,4), trigger at (2,2)
#         => fill (1,1)-(4,4) diagonally, then emit rays along the other diagonal.
#     - Variant local pattern:
#         endpoints at (2,1) and (2,5), trigger at (2,3)
#         => fill row 2 from col 1..5, then emit vertical rays from (2,3).
#     - The changed step is the path construction itself; trigger detection and ray
#       emission stay conceptually the same but rotate into the orthogonal grid basis.
#   complexity_calibration:
#     - The reference task 221dfab4 uses simple row/column propagation. This variant
#       should be comparable: the main path is axis-aligned and the perpendicular effect
#       is also axis-aligned, so no diagonal stepping remains.
#     - To avoid over-simplifying into ambiguous "any aligned pair works", the generator
#       preserves 1-4 separate segments, optional triggers, and inert distractors.
#   implicit_assumptions_review:
#     - Original `diag_path` assumed abs(dr) == abs(dc); this is invalid now and must be
#       replaced by a same-row / same-column path builder.
#     - Original line-color search filtered on diagonal alignment. In the variant it must
#       filter on `dr == 0 xor dc == 0`, otherwise horizontal/vertical pairs would be
#       missed and diagonal distractors could be wrongly selected.
#     - Original grouping by `r-c` / `r+c` only finds diagonal families. The variant must
#       group blue pixels by row and by column.
#     - Original perpendicular directions were diagonal `(dr,-dc)` / `(-dr,dc)`. For a
#       horizontal segment the perpendicular directions are `(-1,0)` and `(1,0)`; for a
#       vertical segment they are `(0,-1)` and `(0,1)`.
#     - The "fill only bg cells" behavior remains necessary. It was harmless before and is
#       still required now so rays do not overwrite endpoints, triggers, or other objects.
#   placement_legality:
#     - Generator must forbid any extra blue pixels from sharing a row or column with
#       existing blue pixels unless they are the intended endpoint pair of one segment.
#       Otherwise `main()` would connect unintended extremes.
#     - Generator also forbids accidental blue-blue diagonal alignments. This is not part
#       of the rule itself, but it prevents the new samples from looking solvable by the
#       old diagonal logic and makes the D1 change visually explicit.
#     - Trigger distractors must not align by row or column with existing trigger pixels
#       from other segments, or trigger color could accidentally become the "best pair"
#       color in the solver.
#     - Paths and rays are kept disjoint from pre-placed pixels and earlier drawn cells,
#       so the solver's non-overwrite semantics match the intended output exactly.
#   observability_and_cleanliness:
#     - In 2-4 examples, a human can see blue endpoints joined by straight rows/columns
#       rather than diagonals, and magenta extensions leave at right angles.
#     - Keeping segments and rays mostly disjoint avoids clutter, so the geometric change
#       is readable without relying on hidden metadata such as color-id ordering.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color straight pixel pairs, then emit orthogonal rays from differently-colored pixels on those straight paths.

    Input structure:
    - bg = most common color. One foreground color supplies endpoint pairs that share a row or a column, and trigger-colored pixels may sit strictly between those endpoints on that same straight path.

    Rule:
    - Fill the full horizontal or vertical segment between each same-color endpoint pair.
    - Whenever that filled segment passes through a non-line pixel, keep that pixel and extend rays from it in the perpendicular orthogonal directions until the grid boundary.

    Key insight:
    - The chosen endpoint color defines an axis-aligned carrier stroke, and trigger pixels on that stroke behave like splitters that fire two straight beams at right angles.
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
    line_color = None
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

    pts = by_color[int(line_color)]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    row_groups: dict[int, list[tuple[int, int]]] = {}
    col_groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        row_groups.setdefault(r, []).append((r, c))
        col_groups.setdefault(c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = straight_path(p, q)
        if p[0] == q[0]:
            perp_dirs = [(-1, 0), (1, 0)]
        else:
            perp_dirs = [(0, -1), (0, 1)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = int(line_color)
            elif v != int(line_color):
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    for group in row_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[1])
            draw_segment(group[0], group[-1])

    for group in col_groups.values():
        if len(group) >= 2:
            group = sorted(group, key=lambda rc: rc[0])
            draw_segment(group[0], group[-1])

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
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_SEG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _straight_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    if p[0] == q[0]:
        lo, hi = sorted((p[1], q[1]))
        return [(p[0], c) for c in range(lo, hi + 1)]
    lo, hi = sorted((p[0], q[0]))
    return [(r, p[1]) for r in range(lo, hi + 1)]


def _compute_ray_cells(triggers, perp_dirs, H, W):
    """Compute all cells touched by perpendicular orthogonal rays from trigger positions."""
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling straight segments and emitting rays."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
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


def _shares_axis_or_diagonal(p, q):
    return p[0] == q[0] or p[1] == q[1] or abs(p[0] - q[0]) == abs(p[1] - q[1])


def _try_place_segment(orient, seg_len, H, W, occupied, drawn, line_positions):
    """Try to place one horizontal or vertical segment, returning (path, perp_dirs) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if orient == "h":
            if W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1)
            c0 = random.randint(0, W - 1 - seg_len)
            p = (r0, c0)
            q = (r0, c0 + seg_len)
            perp_dirs = [(-1, 0), (1, 0)]
        else:
            if H - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1)
            p = (r0, c0)
            q = (r0 + seg_len, c0)
            perp_dirs = [(0, -1), (0, 1)]

        path = _straight_path(p, q)
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        ep0, ep1 = path[0], path[-1]
        if any(_shares_axis_or_diagonal(ep0, lp) or _shares_axis_or_diagonal(ep1, lp) for lp in line_positions):
            continue

        return path, perp_dirs
    return None


def _allowed_trigger_positions(path_interior, existing_triggers):
    allowed = []
    for pos in path_interior:
        if any(pos[0] == tr or pos[1] == tc for tr, tc in existing_triggers):
            continue
        allowed.append(pos)
    return allowed


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
    line_positions = []
    trigger_positions = []
    segments = []

    for _ in range(n_segments):
        orient = random.choice(["h", "v"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(orient, seg_len, H, W, occupied, drawn, line_positions)
        if result is None:
            continue
        path, perp_dirs = result
        ep0, ep1 = path[0], path[-1]

        path_interior = path[1:-1]
        allowed_triggers = _allowed_trigger_positions(path_interior, trigger_positions)
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(allowed_triggers))
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
        chosen_triggers = random.sample(allowed_triggers, n_triggers) if n_triggers > 0 else []

        ray_cells = _compute_ray_cells(chosen_triggers, perp_dirs, H, W)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        segments.append((ep0, ep1, orient, chosen_triggers, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(chosen_triggers)
        drawn.update(set(path_interior) - set(chosen_triggers))
        drawn.update(ray_cells)
        line_positions.extend([ep0, ep1])
        trigger_positions.extend(chosen_triggers)

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, orient, chosen_triggers, path, perp_dirs in segments:
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
            if any(_shares_axis_or_diagonal(pos, lp) for lp in all_line_positions):
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
            if any(r == tr or c == tc for tr, tc in all_trigger_positions):
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
