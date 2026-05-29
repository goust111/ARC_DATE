# Variant of db695cfb
# Dimension: D2 — 延伸路径方向规则 (path)
# Change: 垂直对角延伸（perpendicular_extension）——从品红像素出发，沿原对角线旋转90度的两个方向延伸至网格边界 → 单侧延伸（基于品红像素位置）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Visible objects stay the same as the source task: color 1 endpoints mark diagonal
#      segments, color 6 trigger pixels may lie on those segments, and the segment itself is
#      filled in the output.
#    - The only changed behavior is what happens after a trigger is hit by the completed
#      segment. Originally each trigger emitted two perpendicular diagonal rays. In this
#      variant it emits only one such ray.
#    - The side choice is tied to a visible global reference: the grid main diagonal r=c.
#      If a trigger cell satisfies c>r ("above" that diagonal), it emits the upward
#      perpendicular ray. If c<r ("below"), it emits the downward perpendicular ray.
#      Triggers on r=c would be ambiguous, so valid generated examples forbid them.
#    - Small trace:
#      * Segment "\" through (2,4) uses perpendicular "/" directions.
#      * Source task: trigger at (2,4) emits both up-right and down-left.
#      * Variant: since 4>2, the same trigger is above r=c, so only the upward branch
#        survives: up-right.
#
# b) Complexity calibration
#    - A single trigger would make the new rule hard to infer. The generator therefore
#      requires at least two active triggers in each sample and insists that both sides of
#      the main diagonal appear among them, so the two branches are repeatedly supported.
#    - The rule still has only two categories (above/below), keeping inference tractable.
#
# c) Audit of source-code assumptions that break under the variant
#    - The original solver stored both perpendicular directions for every trigger.
#      That must change to storing only the direction selected from trigger position.
#    - The original expected-output builder and ray-cell planner also assumed two branches.
#      Those helpers must be updated to use the same single-direction rule as main().
#    - The original generator allowed triggers anywhere on a segment interior. That would
#      now create ambiguous r=c cases, so placement must filter them out.
#
# d) Placement legality
#    - Segment paths must remain in bounds and non-overlapping, exactly as before.
#    - Active triggers are only chosen from interior cells with c!=r.
#    - Ray cells are computed before accepting a segment; they may not collide with any
#      already occupied input cell or any previously reserved drawn cell. This keeps the
#      oracle and solver aligned and avoids accidental overwrites.
#
# e) Visual observability
#    - The main diagonal is a human-visible spatial reference; no hidden color-id ordering
#      is used.
#    - Examples are rejected unless at least one active trigger is above and one is below
#      r=c, so the "up branch vs down branch" contrast is actually visible across samples.
#
# f) Visual cleanliness
#    - Output-only structures are kept disjoint by rejecting ray/path overlaps in the
#      generator. This removes messy layered intersections that would hide the one-sided
#      behavior.
#
# g) Color inference
#    - Colors are unchanged from the source task: line color stays 1, trigger/ray color
#      stays 6. The task variation is purely directional, so no new color mapping burden is
#      introduced.


def _side_of_main_diagonal(r: int, c: int) -> int:
    """Return +1 for c>r (above r=c), -1 for c<r (below), 0 on the diagonal."""
    if c > r:
        return 1
    if c < r:
        return -1
    return 0


def _single_perp_dir_from_segment(dr: int, dc: int, r: int, c: int) -> tuple[int, int] | None:
    """
    Pick the only allowed perpendicular ray direction for a trigger on a completed segment.

    The chosen branch depends on which side of the global main diagonal the trigger occupies.
    """
    side = _side_of_main_diagonal(r, c)
    if side == 0:
        return None
    slope_sign = dr * dc  # +1 for "\", -1 for "/"
    if side > 0:
        return -1, slope_sign
    return 1, -slope_sign


def _single_perp_dir_from_dc(dc: int, r: int, c: int) -> tuple[int, int] | None:
    """Generator helper: dc is the segment's column step while row step is +1."""
    side = _side_of_main_diagonal(r, c)
    if side == 0:
        return None
    if side > 0:
        return -1, dc
    return 1, -dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal endpoint pairs, then let each trigger launch only one perpendicular diagonal ray chosen by its side of the grid main diagonal.

    Input structure:
    - The background is the dominant color. One non-background color forms diagonal endpoint pairs, while another color can appear on those diagonals as trigger pixels. Valid task instances place the active triggers off the global main diagonal r=c.

    Rule:
    - Fill each diagonal segment between the extreme same-color endpoints on that diagonal.
    - Whenever the filled segment passes through a trigger pixel, extend that trigger color along exactly one perpendicular diagonal direction until the grid edge.
    - Triggers above r=c (c>r) use the upward perpendicular branch; triggers below r=c (c<r) use the downward perpendicular branch.

    Key insight:
    - The missing half of the old two-sided beam is determined by a global geometric cue: the trigger's position relative to the grid's main diagonal selects which 90-degree-rotated branch survives.
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

    # Choose the line color: the non-background color with the longest diagonal-aligned pair.
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

    # Each trigger collects the single perpendicular direction implied by every segment that
    # crosses it. Valid generated data keeps segments disjoint, so this is usually one branch.
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
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                single_dir = _single_perp_dir_from_segment(dr, dc, r, c)
                if single_dir is None:
                    continue
                key = (r, c, v)
                triggers.setdefault(key, set()).add(single_dir)

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
_H_RANGE = (8, 25)              # grid height range
_W_RANGE = (9, 21)              # grid width range
_LINE_COLOR = 1                 # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6              # color for trigger pixels and one-sided rays
_SEGMENT_COUNT = (1, 4)         # number of diagonal segments
_SEG_LEN_RANGE = (3, 16)        # segment length in cells (inclusive)
_TRIGGERS_PER_SEG = (0, 2)      # active triggers placed on each segment interior

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)  # unpaired line_color distractor pixels
_OFF_DIAG_TRIGGERS = (0, 4)     # off-diagonal trigger_color distractor pixels

# === CFG: variant visibility guards ===
_MIN_ACTIVE_TRIGGERS = 2        # ensure the new branch-selection rule is visible
_REQUIRE_BOTH_SIDES = True      # active triggers must show both above and below r=c

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80     # attempts to place a single segment
_DISTRACTOR_RETRIES = 50        # attempts to place a single distractor pixel
_GEN_RETRIES = 400              # outer generate_input retry limit


def _compute_ray_cells(trigger_positions, dc, H, W):
    """Compute cells touched by the single allowed ray from each active trigger."""
    cells = set()
    trigger_dirs = {}
    trigger_sides = {}
    for tr, tc in trigger_positions:
        single_dir = _single_perp_dir_from_dc(dc, tr, tc)
        if single_dir is None:
            return None, None, None
        trigger_dirs[(tr, tc)] = single_dir
        trigger_sides[(tr, tc)] = _side_of_main_diagonal(tr, tc)
        ddr, ddc = single_dir
        r, c = tr + ddr, tc + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells, trigger_dirs, trigger_sides


def _build_expected(inp, bg, segments):
    """Build the expected output grid by filling diagonals and emitting one-sided rays."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, path, trigger_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for (tr, tc), (ddr, ddc) in trigger_dirs.items():
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
    all_active_sides = []
    total_active_triggers = 0

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        # Only off-main-diagonal interior cells can carry the new single-branch trigger rule.
        valid_interior = [cell for cell in path[1:-1] if _side_of_main_diagonal(*cell) != 0]
        if not valid_interior and _TRIGGERS_PER_SEG[1] > 0:
            n_triggers = 0
        else:
            n_triggers = random.randint(
                _TRIGGERS_PER_SEG[0],
                min(_TRIGGERS_PER_SEG[1], len(valid_interior)),
            )
        trigger_positions = random.sample(valid_interior, n_triggers) if n_triggers > 0 else []

        ray_cells, trigger_dirs, trigger_sides = _compute_ray_cells(trigger_positions, dc, H, W)
        if ray_cells is None:
            continue
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, path, trigger_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path[1:-1]) - set(trigger_positions))
        drawn.update(ray_cells)

        total_active_triggers += len(trigger_positions)
        all_active_sides.extend(trigger_sides.values())

    if not segments:
        return None
    if total_active_triggers < _MIN_ACTIVE_TRIGGERS:
        return None
    if _REQUIRE_BOTH_SIDES and set(all_active_sides) != {-1, 1}:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, path, trigger_dirs in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_dirs:
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
            if _side_of_main_diagonal(r, c) == 0:
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
