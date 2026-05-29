# Combo variant of db695cfb (index 29)
# combo_spec: D2/3+D17/2
# Name: 邻域编码+单侧延伸

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_UP_MARKER_COLOR = 2
_DOWN_MARKER_COLOR = 3
_TRIGGER_COLOR = 6


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The source-task skeleton stays intact: color-1 endpoint pairs define one or more
#      45-degree diagonals, and the output still fills those diagonals completely.
#    - The changed part is how an on-diagonal magenta trigger decides its perpendicular ray.
#      The D17 component adds a visible local code: a red pixel directly above the trigger
#      means "emit the upward perpendicular branch", while a green pixel directly below means
#      "emit the downward perpendicular branch".
#    - The D2 component then constrains the trigger to only one branch. In this fusion,
#      each active trigger must therefore carry exactly one micro-instruction, not zero and
#      not both.
#    - Small trace:
#      * Suppose blue endpoints at (1, 1) and (5, 5), so the completed segment is "\"
#        and its perpendicular directions are NE and SW.
#      * Original task: a magenta trigger at (3, 3) emits both NE and SW.
#      * Fused task: if (2, 3) is red and (4, 3) is not green, the same trigger emits only
#        NE; if instead (4, 3) is green and there is no red above, it emits only SW.
#      * The behavioral difference is concentrated in trigger decoding: local markers now
#        select one of the two 90-degree-rotated branches.
#
# b) Complexity calibration
#    - The fusion still uses only two instruction colors and two ray categories
#      (upward/downward), so the rule remains inferable from a few examples.
#    - To make the change visible, the generator requires at least two active triggers per
#      sample and forces both instruction types to appear somewhere in the same grid.
#    - Red and green markers are kept tightly adjacent to the trigger, which provides local,
#      repeated evidence without introducing a large new symbol vocabulary.
#
# c) Audit of source-code assumptions that break under the variant
#    - Original `elif v != line_color:` treated any non-line pixel on the completed segment
#      as a trigger. That is no longer valid because red/green are instruction symbols, not
#      emitters. The solver must only trigger on color 6.
#    - Original logic assumed every trigger always emitted both perpendicular directions.
#      The fusion invalidates that assumption: the direction must be decoded from the input
#      neighborhood and resolved to exactly one branch.
#    - Because diagonal filling mutates `out`, marker decoding cannot read from `out`; it
#      must read the untouched input `grid`.
#    - The original generator permitted trigger cells with no extra structure. Now each
#      active trigger must reserve one marker cell and exactly one ray direction.
#
# d) Placement legality
#    - Triggers remain strictly inside segment interiors, so their north/south marker cells
#      stay in bounds.
#    - Marker cells must avoid all occupied input cells and all reserved output-only cells.
#    - Single-sided ray paths are precomputed before acceptance and may not collide with any
#      input cell, previously reserved ray, or previously reserved diagonal fill.
#    - Same-color instruction markers are prevented from forming their own diagonal endpoint
#      pairs, which protects the "longest diagonal same-color pair = blue line color"
#      heuristic used by the solver.
#
# e) Visual observability
#    - The deciding cue is human-visible and local: red above vs green below a magenta cell.
#    - Because both cue types are required in accepted samples, the solver-facing contrast
#      between the two one-sided branches is visible within a single example set.
#    - No hidden ordering by numeric color id or scan order is used to choose a direction.
#
# f) Visual cleanliness
#    - No extra marker distractors are generated; red and green only appear as genuine local
#      instructions next to active magenta triggers.
#    - Ray and segment placements are rejected if they would overwrite input pixels, keeping
#      each local trigger-marker pattern readable in the final puzzle.
#    - Off-diagonal magenta distractors are still allowed, but they carry no nearby markers
#      and never lie on completed blue diagonals, so they remain inert.
#
# g) Change inferability
#    - The output color vocabulary is unchanged from the source task: blue fills diagonals
#      and magenta fills rays.
#    - The new colors have fixed semantics by position: red only means "upward branch" when
#      it sits immediately north of a trigger, and green only means "downward branch" when
#      it sits immediately south.
#    - Since every active trigger is single-sided, the difference from the original two-ray
#      behavior is structurally obvious rather than relying on subtle cell counting.
#
# 9) New implicit assumptions after fusion
#    - `main()` assumes valid active triggers have exactly one recognized north/south marker;
#      ambiguous cases with zero or two markers are ignored. The generator enforces the
#      exactly-one-marker condition for all active triggers.
#    - `main()` still assumes the longest same-color diagonal pair identifies the blue line
#      color. The generator preserves that by forbidding diagonal pairs among red/green
#      markers and oracle-checking every sample.
#    - `main()` assumes instruction markers are read in absolute north/south positions while
#      the emitted ray is interpreted relative to the segment orientation. The generator uses
#      the same convention when reserving marker cells and ray paths.


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _diag_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _conflicts_same_color_diagonal(pos: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    return any(_diag_aligned(pos, other) for other in others)


def _decode_single_dir(
    grid: np.ndarray,
    r: int,
    c: int,
    perp_dirs: list[tuple[int, int]],
) -> tuple[int, int] | None:
    h, _ = grid.shape
    up_dir = next((d for d in perp_dirs if d[0] < 0), None)
    down_dir = next((d for d in perp_dirs if d[0] > 0), None)

    has_up = up_dir is not None and r > 0 and int(grid[r - 1, c]) == _UP_MARKER_COLOR
    has_down = down_dir is not None and r + 1 < h and int(grid[r + 1, c]) == _DOWN_MARKER_COLOR

    if has_up == has_down:
        return None
    return up_dir if has_up else down_dir


def _dir_from_flag(perp_dirs: list[tuple[int, int]], emit_up: bool) -> tuple[int, int]:
    return next(d for d in perp_dirs if (d[0] < 0) == emit_up)


def _marker_spec(tr: int, tc: int, emit_up: bool) -> tuple[int, int, int]:
    if emit_up:
        return tr - 1, tc, _UP_MARKER_COLOR
    return tr + 1, tc, _DOWN_MARKER_COLOR


def _compute_ray_cells_for_dir(
    start: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells = set()
    tr, tc = start
    ddr, ddc = direction
    r, c = tr + ddr, tc + ddc
    while 0 <= r < h and 0 <= c < w:
        cells.add((r, c))
        r += ddr
        c += ddc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the same-color diagonal segments, then let each on-segment magenta trigger read one local marker and emit exactly one perpendicular diagonal ray.

    Input structure:
    - The background is the most frequent color. One non-background color forms diagonal endpoint pairs to be connected. Magenta pixels may lie on those diagonals, and an active magenta trigger carries a tiny neighborhood code: either a red pixel directly above it or a green pixel directly below it.

    Rule:
    - Fill each diagonal segment between the extreme same-color endpoints on that diagonal.
    - Whenever the completed segment passes through a magenta trigger, inspect the original input around that trigger: red above selects the upward perpendicular branch, while green below selects the downward perpendicular branch.
    - Emit only that one selected perpendicular diagonal ray until the grid boundary, filling background cells with magenta.

    Key insight:
    - The old two-sided beam splitter is replaced by a one-instruction decoder: each active magenta pixel uses its immediate north/south neighborhood to choose the single perpendicular branch that survives.
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

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

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
            if int(grid[r, c]) == _TRIGGER_COLOR:
                single_dir = _decode_single_dir(grid, r, c, perp_dirs)
                if single_dir is not None:
                    triggers.setdefault((r, c), set()).add(single_dir)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: variant visibility guards ===
_MIN_ACTIVE_TRIGGERS = 2
_REQUIRE_BOTH_DIRECTIONS = True

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, emit_up in trigger_specs:
            ddr, ddc = _dir_from_flag(perp_dirs, emit_up)
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                if expected[r, c] == bg:
                    expected[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, h, w, occupied, drawn):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            dc = 1
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _pick_trigger_specs(path, dc, h, w, occupied, drawn, marker_positions):
    path_interior = path[1:-1]
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
    empty_markers = {_UP_MARKER_COLOR: set(), _DOWN_MARKER_COLOR: set()}
    empty_reserved = set()
    if max_triggers == 0:
        return [], empty_markers, empty_reserved

    target = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
    if target == 0:
        return [], empty_markers, empty_reserved

    perp_dirs = [(1, -dc), (-1, dc)]
    trigger_specs = []
    local_input = {path[0], path[-1]}
    local_reserved = set()
    local_marker_positions = {_UP_MARKER_COLOR: set(), _DOWN_MARKER_COLOR: set()}
    candidates = path_interior[:]
    random.shuffle(candidates)

    for tr, tc in candidates:
        if len(trigger_specs) >= target:
            break
        if (tr, tc) in local_reserved:
            continue

        options = [True, False]
        random.shuffle(options)
        for emit_up in options:
            mr, mc, col = _marker_spec(tr, tc, emit_up)
            marker_cell = (mr, mc)
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            if marker_cell in occupied or marker_cell in drawn or marker_cell in local_input or marker_cell in local_reserved:
                continue

            existing_same = marker_positions[col] | local_marker_positions[col]
            if _conflicts_same_color_diagonal(marker_cell, existing_same):
                continue

            ray_dir = _dir_from_flag(perp_dirs, emit_up)
            ray_cells = _compute_ray_cells_for_dir((tr, tc), ray_dir, h, w)
            blocked = occupied | drawn | local_input | {marker_cell} | local_reserved
            if any(cell in blocked for cell in ray_cells):
                continue

            other_dir = _dir_from_flag(perp_dirs, not emit_up)
            other_ray_cells = _compute_ray_cells_for_dir((tr, tc), other_dir, h, w)
            visible_difference = other_ray_cells - blocked - ray_cells
            if not visible_difference:
                continue

            trigger_specs.append((tr, tc, emit_up))
            local_input.add((tr, tc))
            local_input.add(marker_cell)
            local_marker_positions[col].add(marker_cell)
            local_reserved.update(ray_cells)
            local_reserved.update(other_ray_cells)
            break

    return trigger_specs, local_marker_positions, local_reserved


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice(
        [
            c
            for c in range(10)
            if c not in (_LINE_COLOR, _UP_MARKER_COLOR, _DOWN_MARKER_COLOR, _TRIGGER_COLOR)
        ]
    )

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    marker_positions = {_UP_MARKER_COLOR: set(), _DOWN_MARKER_COLOR: set()}
    segments = []
    direction_flags = []
    total_active_triggers = 0

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        trigger_specs, new_marker_positions, reserved_cells = _pick_trigger_specs(
            path, dc, h, w, occupied, drawn, marker_positions
        )
        perp_dirs = [(1, -dc), (-1, dc)]

        new_input_cells = {path[0], path[-1]}
        ray_cells = set()
        for tr, tc, emit_up in trigger_specs:
            new_input_cells.add((tr, tc))
            mr, mc, _ = _marker_spec(tr, tc, emit_up)
            new_input_cells.add((mr, mc))
            ray_dir = _dir_from_flag(perp_dirs, emit_up)
            ray_cells.update(_compute_ray_cells_for_dir((tr, tc), ray_dir, h, w))

        if any(cell in occupied or cell in drawn for cell in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path, perp_dirs))
        occupied.update(new_input_cells)
        occupied.update(cell for cells in new_marker_positions.values() for cell in cells)
        drawn.update(set(path[1:-1]) - {(tr, tc) for tr, tc, _ in trigger_specs})
        drawn.update(ray_cells)
        drawn.update(reserved_cells)
        for col, cells in new_marker_positions.items():
            marker_positions[col].update(cells)

        total_active_triggers += len(trigger_specs)
        direction_flags.extend(emit_up for _, _, emit_up in trigger_specs)

    if not segments or total_active_triggers < _MIN_ACTIVE_TRIGGERS:
        return None
    if _REQUIRE_BOTH_DIRECTIONS and set(direction_flags) != {False, True}:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for ep0, ep1, trigger_specs, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, emit_up in trigger_specs:
            inp[tr, tc] = _TRIGGER_COLOR
            mr, mc, col = _marker_spec(tr, tc, emit_up)
            inp[mr, mc] = col

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_diag_aligned((r, c), (lr, lc)) for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_magenta_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_diag_aligned((r, c), pos) for pos in all_magenta_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_magenta_positions.append((r, c))
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
