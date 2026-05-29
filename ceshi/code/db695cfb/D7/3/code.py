# Variant of db695cfb
# Dimension: D7 — 交互触发对象 (obstacle)
# Change: 品红像素——位于主对角线路径上的品红像素触发垂直对角延伸... → 不在路径上的邻近像素触发

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# ----------------------------------
# spec_variant:
#   preserved_core:
#     - grid size, colors, diagonal endpoint pairing, and perpendicular-ray idea stay the same.
#     - the line color is still recovered by finding the longest diagonal-aligned same-color pair.
#   changed_core:
#     - active trigger pixels are no longer on the connecting diagonal.
#     - instead, a trigger sits one perpendicular-diagonal step away from an interior line cell.
#     - the ray is emitted from the trigger pixel's own coordinates, not from the line cell it neighbors.
#   concrete_geometry:
#     - for a "\" segment with dirs (1,1), the adjacent trigger slots are the slash-neighbors
#       (r+1,c-1) and (r-1,c+1) of interior path cells.
#     - for a "/" segment with dirs (1,-1), the adjacent trigger slots are the backslash-neighbors
#       (r+1,c+1) and (r-1,c-1) of interior path cells.
#   behavior_trace:
#     - original: if a magenta pixel lies directly on the path, that exact cell becomes a splitter.
#     - variant: if the path passes through (5,5) on a "\" segment and there is a magenta at (6,4),
#       then the magenta at (6,4) is the splitter. The emitted slash ray starts from (6,4), so one
#       branch immediately meets the blue path at (5,5) and then continues beyond it without overwrite.
#     - difference: solver must inspect side-neighbor cells of the segment interior, not just cells on
#       the segment itself.
#   complexity_calibration:
#     - adjacency is restricted to exactly one perpendicular-diagonal step. This is visibly different
#       from the original, but still simple enough for 2-4 ARC examples.
#     - generator ensures at least one active adjacent trigger overall so the variant is observable.
#   implicit_assumption_audit:
#     - original draw_segment() assumed "non-line pixel on the path" implies trigger; this is no longer
#       valid, so trigger detection is moved to side-neighbor scanning over path[1:-1].
#     - original generator could place off-diagonal magenta anywhere away from drawn cells; now those
#       distractors must also avoid every segment's trigger-zone, otherwise they would accidentally fire.
#     - rays are still applied after all line segments are filled, which is required because a trigger's
#       inward branch may cross the just-drawn blue segment.
#   placement_legality:
#     - a trigger-zone is the set of all in-bounds side-neighbor cells of a segment interior.
#     - placed trigger pixels must lie in their owning segment's zone, but distractor magenta must lie
#       outside every zone.
#     - segment zones are kept disjoint from existing non-line pixels so no trigger belongs to two
#       segments or gets activated accidentally.
#   visual_observability:
#     - the offset magenta marker produces a visibly shifted ray origin compared with the original task.
#     - because the magenta is beside the blue diagonal rather than embedded in it, the new rule can be
#       inferred directly from a few examples.
#   cleanliness:
#     - generator avoids ray/path overlaps with prior output where possible and avoids diagonal-aligned
#       magenta pairs, reducing ambiguity in line-color selection.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= r < H and 0 <= c < W


def _diag_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _segment_zone(path, perp_dirs, H, W):
    zone = set()
    for r, c in path[1:-1]:
        for ddr, ddc in perp_dirs:
            nr, nc = r + ddr, c + ddc
            if _in_bounds(H, W, nr, nc):
                zone.add((nr, nc))
    return zone


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


def _build_expected(inp, bg, segments):
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


def _choose_adjacent_triggers(
    path,
    perp_dirs,
    H,
    W,
    occupied,
    drawn,
    all_trigger_positions,
    existing_trigger_zone,
):
    anchor_options = []
    for anchor in path[1:-1]:
        r, c = anchor
        options = []
        for ddr, ddc in perp_dirs:
            cell = (r + ddr, c + ddc)
            if not _in_bounds(H, W, cell[0], cell[1]):
                continue
            if cell in occupied or cell in drawn or cell in existing_trigger_zone:
                continue
            if any(_diag_aligned(cell, other) for other in all_trigger_positions):
                continue
            options.append(cell)
        if options:
            anchor_options.append((anchor, options))

    if not anchor_options:
        return []

    max_triggers = min(_TRIGGERS_PER_SEG[1], len(anchor_options))
    min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)
    n_triggers = random.randint(min_triggers, max_triggers)
    if n_triggers == 0:
        return []

    chosen = []
    for anchor, options in random.sample(anchor_options, len(anchor_options)):
        if len(chosen) >= n_triggers:
            break
        valid = [
            cell
            for cell in random.sample(options, len(options))
            if not any(_diag_aligned(cell, other) for other in chosen)
        ]
        if not valid:
            continue
        chosen.append(valid[0])

    if len(chosen) != n_triggers:
        return None
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill same-color diagonal endpoint pairs, then emit perpendicular rays from nearby off-line trigger pixels.

    Input structure:
    - The background is the dominant color. One non-background color appears as endpoint pairs on shared 45-degree diagonals, while trigger pixels of another color sit one perpendicular-diagonal step beside those segments instead of on them.

    Rule:
    - Connect each maximal diagonal endpoint pair with its line color.
    - For every non-line pixel adjacent to an interior segment cell by one perpendicular-diagonal step, extend that pixel's color along the segment's perpendicular diagonal in both directions, starting from the trigger pixel itself.

    Key insight:
    - The important interaction is no longer "diagonal hits a colored cell". The solver has to notice which nearby side pixel belongs to the diagonal, then use the diagonal's perpendicular orientation to launch a ray from that offset marker.
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

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]

        for r, c in path:
            if int(out[r, c]) == bg:
                out[r, c] = line_color

        for r, c in path[1:-1]:
            for ddr, ddc in perp_dirs:
                nr, nc = r + ddr, c + ddc
                if not _in_bounds(h, w, nr, nc):
                    continue
                v = int(grid[nr, nc])
                if v != bg and v != line_color:
                    key = (nr, nc, v)
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
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    non_line_positions = set()
    all_trigger_positions = set()
    trigger_zone = set()
    segments = []

    for _ in range(n_segments):
        placed = False
        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(["backslash", "slash"])
            seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
            result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
            if result is None:
                continue

            path, dc = result
            perp_dirs = [(1, -dc), (-1, dc)]
            zone = _segment_zone(path, perp_dirs, H, W)
            if zone & non_line_positions:
                continue

            trigger_positions = _choose_adjacent_triggers(
                path,
                perp_dirs,
                H,
                W,
                occupied,
                drawn,
                all_trigger_positions,
                trigger_zone,
            )
            if trigger_positions is None:
                continue

            ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
            current_occupied = occupied | {path[0], path[-1]} | set(trigger_positions)
            if ray_cells & current_occupied:
                continue
            if ray_cells & drawn:
                continue

            segments.append((path[0], path[-1], trigger_positions, path, perp_dirs))
            occupied.update([path[0], path[-1]])
            occupied.update(trigger_positions)
            non_line_positions.update(trigger_positions)
            all_trigger_positions.update(trigger_positions)
            drawn.update(path[1:-1])
            drawn.update(ray_cells)
            trigger_zone.update(zone)
            placed = True
            break

        if not placed:
            continue

    if not segments or not all_trigger_positions:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, path, perp_dirs in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in trigger_zone:
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
            if cell in occupied or cell in drawn or cell in trigger_zone:
                continue
            if any(_diag_aligned(cell, other) for other in all_trigger_positions):
                continue
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            non_line_positions.add(cell)
            all_trigger_positions.add(cell)
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
