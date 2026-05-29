# Combo variant of db695cfb (index 30)
# combo_spec: D10/2+D13/0
# Name: 仅输出延伸区域+擦除触发

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   - D10/2 removes the full-canvas reconstruction and keeps only the emitted magenta-ray footprint.
#   - D13/0 says each on-diagonal magenta trigger is consumed after firing, so the trigger cell itself
#     must become background in the extracted output instead of remaining colored.
# concrete_trace:
#   - Example: blue endpoints at (1,1) and (5,5), trigger at (3,3).
#   - Original task:
#       1. Draw the full backslash diagonal from (1,1) to (5,5) in blue.
#       2. Keep the magenta trigger at (3,3).
#       3. Emit slash rays through (3,3) and return the entire grid.
#   - Combo task:
#       1. Infer the same hidden backslash diagonal, but never render it in the answer.
#       2. Emit the same slash rays.
#       3. Erase the trigger source itself from the final rendering.
#       4. Crop to the minimal bounding box of the emitted ray cells only.
#   - So the visible output is a magenta branch pattern with a background hole where the trigger used to be.
# complexity_calibration:
#   - If a sample had no on-diagonal trigger, the extracted output would be empty, so the generator forces
#     at least one trigger on every kept segment.
#   - D13/0 can become visually redundant inside D10/2 if every trigger falls outside the cropped region.
#     To prevent that collapse, the generator requires at least one trigger position to land inside the
#     cropped ray bbox as a background hole.
#   - Distractors stay sparse and inactive so the hidden diagonal inference is still recoverable by humans.
# hidden_assumption_review:
#   - The original `out = grid.copy()` assumed the answer preserves the input canvas and its seeds.
#     That assumption breaks here; the new solver must collect ray cells first and render a fresh cropped grid.
#   - The original ray drawing used already-painted blue diagonals as blockers via `out[r, c] == bg`.
#     Since the blue backbone is no longer output, we explicitly block with `non_bg seeds ∪ inferred line cells`.
#   - Trigger erasure means trigger positions may lie inside the answer bbox but must stay background there.
#     Therefore the renderer writes only ray cells, never the trigger cells themselves.
# placement_legality:
#   - Segment paths cannot overlap existing seeds or reserved output-only cells.
#   - Ray cells cannot overlap previously reserved paths or rays; otherwise extracted branches merge and the
#     source-hole logic becomes harder to read.
#   - Distractors are never placed on a hidden segment path or on any visible ray cell.
# visual_observability:
#   - Accepted samples always shrink from the input canvas to a smaller cropped answer.
#   - At least one accepted sample feature per grid is a visible background hole inside the cropped magenta
#     footprint, making the "trigger fired then vanished" rule legible from examples alone.
# cleanliness_and_inference:
#   - Output uses only one active color plus background, so overlap semantics stay visually clean.
#   - The only way to explain the hole pattern is to infer a hidden diagonal backbone and consumed triggers,
#     which keeps the combo nontrivial without introducing invisible metadata or color-order tricks.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _infer_segments(
    grid: np.ndarray,
) -> tuple[
    int,
    list[tuple[int, int, int]],
    int,
    set[tuple[int, int]],
    dict[tuple[int, int, int], set[tuple[int, int]]],
]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
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

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    line_cells: set[tuple[int, int]] = set()
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def register_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        line_cells.update(path)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            register_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            register_segment(group[0], group[-1])

    return bg, non_bg, line_color, line_cells, triggers


def _collect_visible_rays(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    ray_cells: dict[tuple[int, int], int] = {}
    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) not in blocked:
                    ray_cells[(r, c)] = tcol
                r += ddr
                c += ddc
    return ray_cells


def _crop_bbox(cells: set[tuple[int, int]] | dict[tuple[int, int], int]) -> tuple[int, int, int, int]:
    coords = list(cells)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solve_with_metadata(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, dict[tuple[int, int], int], tuple[int, int, int, int], list[tuple[int, int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg, non_bg, _, line_cells, triggers = _infer_segments(grid)

    blocked = {(r, c) for r, c, _ in non_bg}
    blocked.update(line_cells)
    ray_cells = _collect_visible_rays(h, w, blocked, triggers)
    assert ray_cells, "at least one trigger-aligned ray cell must exist"

    bbox = _crop_bbox(ray_cells)
    r0, c0, r1, c1 = bbox
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), col in ray_cells.items():
        out[r - r0, c - c0] = col
    return out, bg, ray_cells, bbox, list(triggers)


def _count_visible_trigger_holes(
    trigger_keys: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    ray_cells: dict[tuple[int, int], int],
) -> int:
    r0, c0, r1, c1 = bbox
    holes = 0
    for r, c, _ in trigger_keys:
        if r0 <= r <= r1 and c0 <= c <= c1 and (r, c) not in ray_cells:
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer hidden diagonal connectors, then return only the cropped extension rays with their trigger sources erased.

    Input structure:
    - The dominant color is the background. One repeated non-background color marks endpoint pairs that sit on 45-degree diagonals, while differently colored pixels on those same diagonals are the ray triggers. Extra off-diagonal distractors may also appear.

    Rule:
    - Reconstruct each endpoint diagonal by connecting its extreme same-color points, detect every trigger that lies on that hidden connector, and emit perpendicular diagonal rays from each trigger until the grid boundary.
    - The final answer keeps neither the hidden connector nor any original seed pixel. Trigger cells are treated as consumed firing points, so the output contains only the emitted ray cells inside their minimal bounding rectangle.

    Key insight:
    - The visible output is an extracted branch pattern with source holes: you must infer an invisible diagonal backbone from sparse endpoints, then notice that the trigger locations vanish and only their perpendicular extensions survive.
    """

    out, _, _, _, _ = _solve_with_metadata(grid)
    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _collect_ray_cells(
    triggers: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if (r, c) not in blocked:
                    cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _render_expected(inp: np.ndarray, bg: int, segments) -> tuple[np.ndarray, dict[str, int]]:
    H, W = inp.shape
    blocked = set(zip(*np.where(inp != bg)))
    trigger_keys = []
    for _, _, trigger_positions, path, _ in segments:
        blocked.update(path)
        for tr, tc in trigger_positions:
            trigger_keys.append((tr, tc, _TRIGGER_COLOR))

    ray_cells: dict[tuple[int, int], int] = {}
    for _, _, trigger_positions, _, perp_dirs in segments:
        for r, c in _collect_ray_cells(trigger_positions, perp_dirs, blocked, H, W):
            ray_cells[(r, c)] = _TRIGGER_COLOR

    assert ray_cells, "variant generator requires at least one visible ray cell"

    bbox = _crop_bbox(ray_cells)
    r0, c0, r1, c1 = bbox
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in ray_cells:
        out[r - r0, c - c0] = _TRIGGER_COLOR

    metrics = {
        "visible_trigger_holes": _count_visible_trigger_holes(trigger_keys, bbox, ray_cells),
        "ray_cells": len(ray_cells),
    }
    return out, metrics


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
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
        if any(cell in occupied or cell in reserved for cell in path):
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

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved)
        if result is None:
            continue
        path, dc = result

        path_interior = path[1:-1]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)
        if max_triggers <= 0:
            continue

        n_triggers = random.randint(min_triggers, max_triggers)
        trigger_positions = random.sample(path_interior, n_triggers)
        perp_dirs = [(1, -dc), (-1, dc)]

        blocked = set(path)
        blocked.update(occupied)
        ray_cells = _collect_ray_cells(trigger_positions, perp_dirs, blocked, H, W)
        if not ray_cells:
            continue
        if any(cell in reserved for cell in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        reserved.update(path)
        reserved.update(ray_cells)

    if not segments:
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
            if (r, c) in occupied or (r, c) in reserved:
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
            if (r, c) in occupied or (r, c) in reserved:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected, metrics = _render_expected(inp, bg, segments)
    if result.shape == inp.shape:
        return None
    if metrics["visible_trigger_holes"] <= 0:
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
