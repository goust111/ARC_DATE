import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

    Input structure:
    - bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

    Rule:
    - Draw diagonal lines connecting each pair of same-color aligned pixels.
    - Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

    Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
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

    # Choose the "line color": the non-background color with the longest diagonal-aligned pair.
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

    # Collect triggers on any connecting segment. Each trigger stores which perpendicular
    # directions to extend (depends on the segment orientation).
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    # Group line_color points by both diagonal orientations.
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

    # For each diagonal that has >=2 line_color points, connect its extremes.
    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    # Emit perpendicular diagonal rays from each trigger to the grid boundary.
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
_H_RANGE = (8, 25)             # grid height range
_W_RANGE = (9, 21)             # grid width range
_LINE_COLOR = 1                # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6             # color for trigger pixels and perpendicular rays
_SEGMENT_COUNT = (1, 4)        # number of diagonal segments
_SEG_LEN_RANGE = (3, 16)      # segment length in cells (inclusive)
_TRIGGERS_PER_SEG = (0, 2)    # triggers placed on each segment interior

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2) # unpaired line_color distractor pixels
_OFF_DIAG_TRIGGERS = (0, 4)    # off-diagonal trigger_color distractor pixels

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80    # attempts to place a single segment
_DISTRACTOR_RETRIES = 50       # attempts to place a single distractor pixel
_GEN_RETRIES = 300             # outer generate_input retry limit


def _compute_ray_cells(triggers, perp_dirs, H, W):
    """Compute all cells touched by perpendicular rays from trigger positions."""
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
    """Build the expected output grid by filling diagonals and emitting rays."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
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


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn):
    """Try to place one diagonal segment, returning (path, dc) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == 'backslash':
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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()  # input pixel positions
    drawn = set()     # output-only cells (diagonal fills + ray fills)
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(['backslash', 'slash'])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        # Place triggers on segment interior
        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 and path_interior else []

        # Perpendicular directions depend on slope
        perp_dirs = [(1, -dc), (-1, dc)]

        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if any(rc in occupied for rc in ray_cells):
            continue

        # Record segment
        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments:
        return None

    # Build input grid
    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, path, perp_dirs in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    # Distractors: unpaired line_color pixels (must not diagonally align with any existing line pixel)
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

    # Distractors: off-diagonal trigger_color pixels
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    # Oracle check
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
