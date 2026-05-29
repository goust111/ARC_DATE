# Variant of db695cfb
# Dimension: D2 — 延伸路径方向规则 (path)
# Change: 垂直对角延伸（perpendicular_extension）——从品红像素出发，沿原对角线旋转90度的两个方向延伸至网格边界 → 正交延伸（水平+垂直）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old: the diagonal connector hit a trigger cell and spawned two rays on the perpendicular diagonal
#     new: the diagonal connector still behaves the same, but every trigger now emits a 4-way orthogonal cross:
#       up, down, left, right until the grid boundary
#   concrete_visual:
#     input:
#       - blue endpoint pairs define one or more 45-degree segments
#       - magenta trigger pixels can lie on those diagonal interiors
#       - optional distractor blue/magenta singletons still remain inert
#     output:
#       - the diagonal segment is filled in blue
#       - each on-segment magenta trigger keeps its cell and paints its whole row/column outward in magenta
#       - the result should read as "diagonal backbone + rectilinear branches"
#   flow_trace_example:
#     original:
#       - endpoints at (2,2) and (6,6), trigger at (4,4)
#       - draw the blue diagonal (2,2)->(6,6)
#       - emit magenta on perpendicular diagonal: (3,5),(2,6)... and (5,3),(6,2)...
#     variant:
#       - keep the same blue diagonal
#       - from the same trigger emit magenta on row 4 and column 4 instead
#       - difference: branch geometry no longer depends on the segment slope; only the backbone is diagonal
#   complexity_calibration:
#     - orthogonal rays cover far more cells than diagonal rays, so keeping 1-4 segments with up to 2 triggers each
#       would produce clutter and frequent placement conflicts
#     - to keep the D2 contrast visible and samples readable, generation is tightened to 1-3 segments and at most
#       1 trigger per segment, while still requiring at least 1 real trigger overall
#   implicit_assumption_audit:
#     - longest-diagonal-pair selection in main() assumes only one color meaningfully forms endpoint pairs
#       => generator guarantees blue is the only color with diagonal-aligned same-color pairs
#     - trigger collection previously stored segment-orientation-specific perpendicular directions
#       => this must be removed because the new branch rule is orientation-invariant
#     - ray filling only writes background cells
#       => still safe, but generator must avoid placing future objects on planned row/column rays or branches become punctured
#   placement_legality:
#     - every segment path must avoid existing input pixels and already reserved output cells
#     - every orthogonal ray cell must avoid both occupied input cells and prior drawn cells
#     - trigger distractors are prevented from forming their own diagonal pair, so they cannot be mistaken for the line color
#   observability:
#     - at least one on-segment trigger is mandatory, otherwise the D2 change would be invisible
#     - branch direction is obvious from 2-3 examples because magenta expands in axis-aligned crosses rather than diagonals
#   cleanliness:
#     - cross rays are prevented from overlapping earlier diagonals/rays during generation
#     - this preserves clear separation between the blue diagonal backbone and magenta orthogonal branches
#
# New solver assumptions checked against generator:
#   - There is at least one diagonal-aligned same-color blue pair: enforced by segment placement.
#   - Non-blue colors on the chosen diagonal act as triggers: generator places at least one on-diagonal magenta trigger.
#   - Blue is the only color with a same-color diagonal pair long enough to be selected: generator forbids magenta diagonal pairs.


_ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal endpoint pairs, then emit orthogonal cross rays from trigger pixels on those diagonals.

    Input structure:
    - bg is the dominant color. One non-bg color forms endpoint pairs on shared 45-degree diagonals, while trigger-colored pixels may lie on the interiors of those diagonals.

    Rule:
    - Fill each qualifying diagonal segment between the extreme same-color endpoints.
    - Whenever that filled segment passes through a differently-colored trigger pixel, extend that trigger's color horizontally and vertically from the trigger cell to the grid boundary.

    Key insight:
    - The diagonal backbone identifies which pixels belong to the connector, but the triggered branches switch coordinate systems and expand as axis-aligned crosses rather than diagonal continuations.
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

    # Choose the non-background color that owns the longest diagonal-aligned same-color pair.
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

    triggers: set[tuple[int, int, int]] = set()

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        for r, c in diag_path(p, q):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.add((r, c, v))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

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
_H_RANGE = (8, 25)              # grid height range
_W_RANGE = (9, 21)              # grid width range
_LINE_COLOR = 1                 # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6              # color for trigger pixels and orthogonal rays
_SEGMENT_COUNT = (1, 3)         # reduced because orthogonal rays occupy full rows/cols
_SEG_LEN_RANGE = (3, 16)        # segment length in cells (inclusive)
_TRIGGERS_PER_SEG = (0, 1)      # one cross-branch per segment keeps the output legible

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)  # unpaired line_color distractor pixels
_OFF_DIAG_TRIGGERS = (0, 3)     # off-diagonal trigger_color distractor pixels

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120    # attempts to place a single segment
_DISTRACTOR_RETRIES = 80        # attempts to place a single distractor pixel
_GEN_RETRIES = 500              # outer generate_input retry limit


def _shares_diagonal(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) == abs(c - cc) and r != rr for rr, cc in others)


def _compute_ray_cells(triggers, H, W):
    """Compute all cells touched by orthogonal rays from trigger positions."""
    cells = set()
    for tr, tc in triggers:
        for r in range(0, tr):
            cells.add((r, tc))
        for r in range(tr + 1, H):
            cells.add((r, tc))
        for c in range(0, tc):
            cells.add((tr, c))
        for c in range(tc + 1, W):
            cells.add((tr, c))
    return cells


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling diagonals and emitting orthogonal rays."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in _ORTHO_DIRS:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn):
    """Try to place one diagonal segment, returning its full path or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)

        dc = 1 if slope_type == "backslash" else -1
        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()       # input pixel positions
    drawn = set()          # output-only cells (diagonal fills + orthogonal ray fills)
    trigger_pixels = set() # all input trigger-color cells; kept diagonal-pair-free
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path = result

        path_interior = path[1:-1]
        eligible_triggers = [p for p in path_interior if not _shares_diagonal(p, trigger_pixels)]

        min_triggers = 1 if not segments else _TRIGGERS_PER_SEG[0]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(eligible_triggers))
        if max_triggers < min_triggers:
            continue

        n_triggers = random.randint(min_triggers, max_triggers)
        trigger_positions = random.sample(eligible_triggers, n_triggers) if n_triggers > 0 else []

        ray_cells = _compute_ray_cells(trigger_positions, H, W)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        trigger_pixels.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments or not trigger_pixels:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, path in segments:
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
            cell = (r, c)
            if cell in occupied or cell in drawn:
                continue
            if _shares_diagonal(cell, trigger_pixels):
                continue
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            trigger_pixels.add(cell)
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
