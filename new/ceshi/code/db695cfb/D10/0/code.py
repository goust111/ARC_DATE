# Variant of db695cfb
# Dimension: D10 — 网格输出形状 (grid_shape)
# Change: 输出与输入同尺寸（same）——输出网格大小不变... → 裁剪到有效区域

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D10 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     - Keep the original recognition and drawing rule: find the longest-color
#       diagonal family, connect each extreme pair on each occupied diagonal,
#       and emit perpendicular diagonal rays from differently-colored trigger
#       pixels sitting on those segments.
#     - Change only the final output framing: after all drawing is complete,
#       crop to the minimal bounding box of the active construction
#       (segment paths plus the emitted ray trajectories).
#   concrete_behavior_and_trace:
#     - Example: on an 8x9 canvas, blue endpoints at (2,2) and (5,5) create a
#       backslash segment. A magenta trigger at (3,3) emits NE and SW rays.
#       Original task returns the full 8x9 grid after drawing. This variant
#       first draws the same cells, then trims away outer background rows and
#       columns that are untouched by the segment/rays.
#     - If an unrelated magenta distractor sits at (7,8) outside that active
#       box, it remains in the input but disappears from the cropped output.
#       This is the visible difference from the original same-size version.
#   complexity_calibration:
#     - Cropping must be based on active line geometry, not on all non-bg
#       pixels. If we cropped by all non-bg cells, outside distractors would
#       prevent any visible D10 change and the task would collapse back toward
#       the original framing.
#     - We keep the color vocabulary and segment/ray logic unchanged so the new
#       difficulty is localized to predicting output size and position.
#   implicit_assumption_review:
#     - The original `out = grid.copy()` is still valid because drawing rules do
#       not change; only the returned slice changes.
#     - The longest diagonal-pair heuristic for selecting `line_color` still
#       assumes distractor line pixels never form competing diagonal pairs.
#       The generator preserves that invariant.
#     - Grouping by `r-c` and `r+c` still assumes each real segment is encoded
#       by endpoint extremes on a shared diagonal. That remains unchanged.
#     - The new crop step assumes at least one real segment exists, so the
#       active-cell set is non-empty. The generator continues to require at
#       least one segment.
#   placement_legality:
#     - Segment paths may not overlap already occupied input cells or prior
#       drawn-only cells.
#     - Trigger rays may not hit occupied input pixels during construction.
#       They may share space with already drawn output-only cells, which is
#       harmless because colors are only written onto background cells.
#     - To make D10 observable, generated samples are rejected when the active
#       bounding box still equals the full input canvas.
#   visual_observability_and_cleanliness:
#     - The generator prefers some distractors outside the crop box, so examples
#       visibly show that irrelevant outer pixels are discarded.
#     - We keep distractor counts modest and do not add extra colors or new
#       symbolic encodings, so the visual story remains "same drawing rule,
#       tighter frame".
#   solver_assumption_audit:
#     - `draw_segment` assumes any non-bg cell on the segment is a trigger or an
#       endpoint. Generator guarantees no other colors lie on segment interiors.
#     - Ray extension loops assume grid bounds are the stopping condition.
#       Cropping happens only after all such loops finish, so no intermediate
#       coordinate logic changes.
#     - The crop bbox is computed from stored active cells, and generator-built
#       distractors never need to influence it; this matches the intended D10
#       semantics exactly.


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_grid(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the same diagonal connections as the source task, then return only the tight active window.

    Input structure:
    - The most common color is background. One non-background color forms diagonal endpoint pairs, and differently-colored trigger pixels may lie on the connecting diagonals. Extra non-background distractors can sit elsewhere.

    Rule:
    - Connect each extreme same-color pair on a shared 45-degree diagonal.
    - Whenever one of those diagonals passes through a trigger pixel, extend two rays in the perpendicular diagonal direction until they leave the canvas.
    - After all drawing is complete, crop the result to the smallest rectangle containing the constructed segment/ray pattern.

    Key insight:
    - The geometry of the blue-to-blue diagonals still determines everything, but the final answer is no longer anchored to the original canvas; only the actively drawn structure sets the output frame.
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    active_cells: set[tuple[int, int]] = set()

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        active_cells.update(path)
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

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                active_cells.add((r, c))
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    assert active_cells, "at least one segment must contribute active cells"
    return _crop_grid(out, _bbox_from_cells(active_cells))


# --------------- generator (D10 variant) ---------------

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
    """Compute all cells covered by perpendicular ray trajectories."""
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
    inp: np.ndarray, bg: int, segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]
) -> tuple[np.ndarray, tuple[int, int, int, int], set[tuple[int, int]]]:
    """Build the cropped expected output and return its active bbox."""
    H, W = inp.shape
    expected = inp.copy()
    active_cells: set[tuple[int, int]] = set()

    for _, _, trigger_positions, path, perp_dirs in segments:
        active_cells.update(path)
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    active_cells.add((r, c))
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc

    bbox = _bbox_from_cells(active_cells)
    return _crop_grid(expected, bbox), bbox, active_cells


def _try_place_segment(
    slope_type: str, seg_len: int, H: int, W: int, occupied: set[tuple[int, int]], drawn: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int] | None:
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
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]] = []

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

    _, bbox, _ = _build_expected(inp, bg, segments)
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

    try:
        result = main(inp)
    except Exception:
        return None

    expected, _, _ = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
