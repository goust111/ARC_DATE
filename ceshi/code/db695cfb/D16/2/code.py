# Variant of db695cfb
# Dimension: D16 — 延伸线数量（每个品红像素） (meta)
# Change: 每个品红像素发射两条延伸线——垂直对角方向的正反两个方向... → 仅一条延伸线（方向由位置决定）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D16 variant
# a) Concrete behavior and trace:
# - The blue color still marks diagonal endpoints. Completing each pair still produces a full
#   45-degree segment exactly as in the source task.
# - Magenta pixels still matter only if they lie on one of those completed diagonals. The change
#   happens after detection: instead of emitting both perpendicular-diagonal rays, each trigger now
#   emits exactly one.
# - I concretize "direction decided by position" as an alternating rule along each segment:
#   traverse the completed segment from its upper endpoint to its lower endpoint; the 1st, 3rd,
#   5th... trigger fires in the perpendicular direction with row -1 ("upward"), while the 2nd,
#   4th, 6th... fires in the opposite direction with row +1 ("downward").
# - Example on a slash segment from (1,7) to (6,2): suppose triggers lie at (2,6), (3,5), (5,3).
#   Original rule would emit both NW and SE rays from all three magenta cells. The new rule emits
#   only NW from the 1st trigger, only SE from the 2nd, and only NW from the 3rd. The difference
#   is visible because each magenta location now paints only one side of the crossing family.
#
# b) Complexity calibration:
# - If a segment had only one trigger, humans could misread the rule as "pick the upward side"
#   rather than "alternate by order". To keep the D16 change visible, the generator places 2-4
#   triggers on every active segment and uses lengths >= 5 so those triggers fit cleanly.
# - I reduce distractor counts slightly so the new one-sided behavior is not buried under clutter.
#
# c) Implicit-assumption audit against the original main():
# - Original `draw_segment()` stored both perpendicular directions for every trigger. That depends
#   on the old invariant "all triggers split symmetrically". It is no longer valid.
# - Original trigger collection used a set of directions per trigger; in the variant this becomes
#   a set that normally contains one direction, but keeping the set structure still makes the solver
#   robust if an arbitrary test grid causes the same trigger to be activated by more than one line.
# - Original generator computed ray cells from a segment-wide pair of perpendicular directions.
#   That also depended on the old invariant. The variant must instead assign a direction per trigger
#   after ordering the triggers along the segment.
#
# d) Placement legality:
# - Segment paths are still placed output-first and may not intersect any already occupied or drawn
#   cell. This guarantees each trigger belongs to at most one completed segment in generated data.
# - Single-ray cells are checked against both `occupied` and `drawn`, which keeps generated outputs
#   visually disjoint: no ray crosses another ray, diagonal fill, endpoint, or distractor.
# - Off-diagonal magenta distractors and unpaired blue distractors are still allowed, but only on
#   untouched background cells so the solver can ignore them safely.
#
# e) Visual observability:
# - The rule never depends on color ids or hidden metadata. The only control variable is visible
#   order along the completed diagonal.
# - Every generated segment has at least two on-segment triggers, so both parity branches appear
#   locally inside the same object. This makes the one-ray alternation inferable from 2-4 examples.
#
# f) Visual cleanliness:
# - I keep 1-3 segments instead of up to 4 and reject ray overlaps. That preserves clear diagonal
#   structures instead of producing dense magenta tangles.
#
# g) Color-change inferability:
# - Colors do not remap. Blue still completes the carrier diagonal and magenta still paints the
#   perpendicular rays. Only the count and side-selection of magenta rays changes.
#
# h) Post-change assumptions that the generator must satisfy:
# - `main()` assumes the meaningful line color is still the color having the longest diagonal-aligned
#   same-color pair. The generator preserves that by using color 1 for all true segment endpoints
#   and avoiding diagonal alignment among blue distractors.
# - `main()` assumes trigger ordering is defined by segment traversal from top to bottom. The
#   generator builds each path in that same orientation, so its oracle and the solver agree.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal endpoint pairs, then let on-segment trigger pixels fire one alternating perpendicular ray.

    Input structure:
    - The background is the most frequent color. One non-background color appears as endpoint pairs on 45-degree diagonals, and other-colored trigger pixels may lie strictly between those endpoints on the same diagonal.

    Rule:
    - Fill each diagonal segment between the extreme aligned endpoints of the line color.
    - Read every completed segment from its upper endpoint to its lower endpoint. Non-line pixels encountered on that path act as triggers: the 1st, 3rd, 5th... trigger emits only the upward perpendicular diagonal ray, while the 2nd, 4th, 6th... trigger emits only the downward perpendicular diagonal ray, each continuing to the grid boundary without overwriting non-background cells.

    Key insight:
    - The magenta pixels are no longer symmetric splitters. They still activate only when the blue diagonal passes through them, but now their position order along that diagonal chooses which single perpendicular side is extended.
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

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dc = 1 if q[1] > p[1] else -1
        upward_dir = (-1, dc)
        downward_dir = (1, -dc)
        segment_triggers: list[tuple[int, int, int]] = []

        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                segment_triggers.append((r, c, v))

        for idx, key in enumerate(segment_triggers):
            ray_dir = upward_dir if idx % 2 == 0 else downward_dir
            triggers.setdefault(key, set()).add(ray_dir)

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
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (5, 16)
_TRIGGERS_PER_SEG = (2, 4)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _assign_trigger_dirs(
    trigger_positions: list[tuple[int, int]], path: list[tuple[int, int]], dc: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    path_index = {rc: idx for idx, rc in enumerate(path)}
    ordered = sorted(trigger_positions, key=lambda rc: path_index[rc])
    upward_dir = (-1, dc)
    downward_dir = (1, -dc)
    return [(rc, upward_dir if idx % 2 == 0 else downward_dir) for idx, rc in enumerate(ordered)]


def _compute_ray_cells(trigger_specs, H, W):
    cells = set()
    for (tr, tc), (ddr, ddc) in trigger_specs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(inp, bg, segments):
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for (tr, tc), (ddr, ddc) in trigger_specs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if expected[r, c] == bg:
                    expected[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn, line_positions):
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
        endpoints = (path[0], path[-1])
        if any(
            abs(r - lr) == abs(c - lc)
            for r, c in endpoints
            for lr, lc in line_positions
        ):
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
    drawn = set()
    segments = []
    line_positions = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn, line_positions)
        if result is None:
            continue
        path, dc = result

        path_interior = path[1:-1]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        min_triggers = min(_TRIGGERS_PER_SEG[0], len(path_interior))
        if max_triggers < _TRIGGERS_PER_SEG[0]:
            continue

        n_triggers = random.randint(min_triggers, max_triggers)
        trigger_positions = random.sample(path_interior, n_triggers)
        trigger_specs = _assign_trigger_dirs(trigger_positions, path, dc)

        ray_cells = _compute_ray_cells(trigger_specs, H, W)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)
        line_positions.extend([ep0, ep1])

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_specs, path in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp, _ in trigger_specs:
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
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
