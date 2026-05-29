# Variant of db695cfb
# Dimension: D10 — 网格输出形状 (grid_shape)
# Change: 输出与输入同尺寸（same）——输出网格大小不变... → 只输出延伸线区域

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D10 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     original_output: full canvas with inferred blue connector diagonals plus magenta rays
#     new_output: crop to the minimal bounding box that contains only the emitted magenta ray cells
#     removed_from_output:
#       - blue reconstructed diagonals
#       - blue endpoint seeds
#       - magenta trigger seeds on the blue diagonals
#       - off-diagonal distractor seeds
#   concrete_behavior:
#     - Input still shows sparse blue endpoints and magenta trigger pixels on the same 45-degree diagonal.
#     - Solver must still infer the hidden blue connector diagonal, because only trigger pixels that lie on that
#       inferred connector emit rays.
#     - The answer discards the full original canvas and keeps only the branch footprint: magenta ray cells inside
#       their tight bounding rectangle, with background-colored gaps where no ray is present.
#   trace_example:
#     - Suppose blue endpoints are at (1,1) and (5,5), with a magenta trigger at (3,3).
#     - Original task:
#       1. Fill the backslash diagonal from (1,1) to (5,5) in blue.
#       2. Emit the perpendicular slash ray through the trigger.
#       3. Return the whole HxW grid containing both the blue backbone and the magenta branch.
#     - D10 variant:
#       1. Infer the same blue diagonal only as latent structure.
#       2. Emit the same perpendicular slash ray, but skip the trigger cell itself because it is an original seed.
#       3. Forget the original canvas and crop to the smallest rectangle covering just those emitted magenta cells.
#   complexity_calibration:
#     - The puzzle would degenerate if no trigger existed, because the new output would be empty.
#       Therefore every generated segment gets at least one on-diagonal trigger.
#     - To keep the D10 change visually obvious, the generator rejects cases where the cropped ray box has the
#       exact same shape as the input canvas.
#     - Distractors remain sparse so the human-visible rule is still recoverable from 2-4 examples.
#   hidden_assumption_review:
#     - `out = grid.copy()` from the original solver depended on preserving seeds and same-sized output.
#       That is invalid now; the new solver collects ray cells separately and renders a cropped result from scratch.
#     - Original ray emission checked `out[r,c] == bg`, which implicitly used already-drawn blue diagonals as blockers.
#       The variant must preserve that behavior even though the blue diagonal is no longer output, so we explicitly
#       build a `blocked` set = original non-bg seeds union inferred diagonal cells before emitting rays.
#     - Original solver could tolerate zero triggers because the full-grid output still existed. The new solver
#       asserts that at least one ray cell is emitted, and the generator guarantees this.
#   placement_legality:
#     - Segment paths may not overlap existing seeds or reserved drawn cells.
#     - Distractors are never placed on any inferred diagonal path or emitted ray cell, otherwise they would
#       accidentally become active triggers or alter the cropped output.
#     - Oracle validation still uses `main(inp) == expected` so rejected layouts cannot slip through.
#   visual_observability:
#     - The new output is visibly smaller than the input in accepted samples.
#     - Output contains only trigger-color rays on the original background, making the extraction/cropping rule
#       legible without relying on color-id ordering or invisible metadata.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer diagonal endpoint connectors, then return only the cropped ray footprint they activate.

    Input structure:
    - The dominant color is the background. One repeated non-background color marks blue-style endpoint pairs on 45-degree diagonals, while differently colored pixels on those diagonals act as ray triggers. Extra off-diagonal distractors may also be present.

    Rule:
    - Reconstruct each occupied diagonal segment by connecting the extreme endpoint pair, note every non-endpoint color encountered on that connector, and from each such trigger extend rays along the perpendicular diagonal directions until the grid edge.
    - Do not keep the reconstructed blue backbone or any original seed pixels in the answer; instead output only the minimal bounding rectangle covering the emitted ray cells.

    Key insight:
    - The visible output is a filtered branch extraction: the sparse input seeds still describe the hidden connector geometry, but the answer keeps only the perpendicular ray pattern produced by those triggers.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
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

    # Choose the endpoint color as the one that participates in the longest same-color diagonal pair.
    best_dist = None
    line_color = None
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
                if best_dist is None or dist > best_dist:
                    best_dist = dist
                    line_color = col

    assert line_color is not None, "at least one diagonal-aligned same-color pair must exist"
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    line_cells: set[tuple[int, int]] = set()

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def register_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        line_cells.update(path)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            register_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            register_segment(g[0], g[-1])

    blocked = {(r, c) for r, c, _ in non_bg}
    blocked.update(line_cells)

    ray_cells: dict[tuple[int, int], int] = {}
    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) not in blocked:
                    ray_cells[(r, c)] = tcol
                r += ddr
                c += ddc

    assert ray_cells, "at least one trigger-aligned ray cell must exist"

    rows = [r for r, _ in ray_cells]
    cols = [c for _, c in ray_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), col in ray_cells.items():
        out[r - r0, c - c0] = col
    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)              # grid height range
_W_RANGE = (9, 21)              # grid width range
_LINE_COLOR = 1                 # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6              # color for trigger pixels and perpendicular rays
_SEGMENT_COUNT = (1, 4)         # number of diagonal segments
_SEG_LEN_RANGE = (3, 16)        # segment length in cells (inclusive)
_TRIGGERS_PER_SEG = (1, 2)      # every kept segment must contribute visible rays

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)  # unpaired line_color distractor pixels
_OFF_DIAG_TRIGGERS = (0, 4)     # off-diagonal trigger_color distractor pixels

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80     # attempts to place a single segment
_DISTRACTOR_RETRIES = 50        # attempts to place a single distractor pixel
_GEN_RETRIES = 400              # outer generate_input retry limit


def _collect_ray_cells(
    triggers: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    """Compute visible ray cells while respecting blocking cells."""
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


def _render_cropped_rays(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    """Build the expected cropped output containing only ray cells."""
    H, W = inp.shape
    blocked = set(zip(*np.where(inp != bg)))
    for _, _, _, path, _ in segments:
        blocked.update(path)

    ray_cells = set()
    for _, _, trigger_positions, _, perp_dirs in segments:
        ray_cells.update(_collect_ray_cells(trigger_positions, perp_dirs, blocked, H, W))

    assert ray_cells, "variant generator requires at least one visible ray cell"

    rows = [r for r, _ in ray_cells]
    cols = [c for _, c in ray_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in ray_cells:
        out[r - r0, c - c0] = _TRIGGER_COLOR
    return out


def _try_place_segment(slope_type, seg_len, H, W, occupied, reserved):
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

    occupied = set()   # input pixel positions
    reserved = set()   # cells kept empty so later placements cannot change the intended logic
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

    expected = _render_cropped_rays(inp, bg, segments)
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
