# Combo variant of db695cfb (index 16)
# combo_spec: D4/0+D5/2+D13/3
# Name: 周期主色+交替延伸色+长度编码触发色

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the fused variant:
# - Concrete behavior: keep the input anchor pair blue (1) and the on-diagonal trigger
#   seeds magenta (6). If a segment path is
#   [endpoint, bg, trigger_a, bg, trigger_b, bg, endpoint],
#   the output path becomes [blue, teal, trigger-color, teal, trigger-color, teal, blue]
#   under the fixed blue/teal cycle. trigger_a emits magenta dashes with gap 1
#   (M . M . ...), trigger_b emits magenta dashes with gap 2 (M . . M . . ...).
#   Each trigger's own output color is then chosen from the full geometric length
#   of its perpendicular extension corridor: short -> red (2), medium -> yellow (4),
#   long -> green (3).
# - Complexity calibration: the line cycle is fixed to two colors and the trigger
#   length map has only three bins, so the rule is richer than the seed task but still
#   inferable from a few examples. The generator forces at least one segment with
#   multiple triggers and at least two different length bins so the three fused
#   mechanisms are actually visible.
# - Hidden assumptions changed from the seed solver:
#   1) The input no longer contains the final main-line colors; only blue anchor
#      endpoints identify the segments.
#   2) Trigger seeds stay magenta only in the input; in the output they are recolored
#      from geometry, so path painting must skip trigger cells and resolve them later.
#   3) Dash gaps depend on trigger order along each segment, so trigger ordering must
#      be preserved per segment instead of treating all triggers as an unordered set.
# - Placement legality: the generator reserves every full diagonal corridor touched by
#   a potential ray, not just the painted dash cells. That prevents overlap with other
#   segments, keeps distractors inactive, and makes the total extension length visible.
# - Visual observability and cleanliness: the main line uses a stable blue/teal rhythm,
#   real triggers are the only seeds that change away from magenta, and their rays stay
#   on empty corridors. This avoids ambiguous overdraw while still exposing three
#   learnable cues: periodic path coloring, index-based dash spacing, and
#   length-to-color mapping.


_LINE_ANCHOR = 1
_PATH_CYCLE = (1, 8)  # blue, teal
_INPUT_TRIGGER = 6    # magenta seeds in the input and magenta dash color in the output
_SHORT_TRIGGER = 2    # red
_MED_TRIGGER = 4      # yellow
_LONG_TRIGGER = 3     # green
_TRIGGER_BIN_LIMITS = (4, 8)  # <=4 short, <=8 medium, else long

_RESERVED_COLORS = {
    _LINE_ANCHOR,
    _PATH_CYCLE[1],
    _INPUT_TRIGGER,
    _SHORT_TRIGGER,
    _MED_TRIGGER,
    _LONG_TRIGGER,
}

# === CFG: canvas and structural bounds ===
_H_RANGE = (9, 25)
_W_RANGE = (10, 21)
_SEGMENT_COUNT = (1, 2)
_SEG_LEN_RANGE = (5, 14)
_PRIMARY_TRIGGER_COUNT = (2, 3)
_SECONDARY_TRIGGER_COUNT = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_TRIGGER_SELECT_RETRIES = 80
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _perp_dirs_from_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = path[0], path[-1]
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    return [(dr, -dc), (-dr, dc)]


def _full_ray_cells(start: tuple[int, int], dirs: list[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    cells = set()
    r0, c0 = start
    for ddr, ddc in dirs:
        r, c = r0 + ddr, c0 + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _ray_total_length(start: tuple[int, int], dirs: list[tuple[int, int]], H: int, W: int) -> int:
    return len(_full_ray_cells(start, dirs, H, W))


def _trigger_color_from_length(total_len: int) -> int:
    if total_len <= _TRIGGER_BIN_LIMITS[0]:
        return _SHORT_TRIGGER
    if total_len <= _TRIGGER_BIN_LIMITS[1]:
        return _MED_TRIGGER
    return _LONG_TRIGGER


def _trigger_bin_index(total_len: int) -> int:
    if total_len <= _TRIGGER_BIN_LIMITS[0]:
        return 0
    if total_len <= _TRIGGER_BIN_LIMITS[1]:
        return 1
    return 2


def _dash_cells(
    start: tuple[int, int],
    dirs: list[tuple[int, int]],
    gap_size: int,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells = set()
    period = gap_size + 1
    r0, c0 = start
    for ddr, ddc in dirs:
        step = 1
        r, c = r0 + ddr, c0 + ddc
        while 0 <= r < H and 0 <= c < W:
            if (step - 1) % period == 0:
                cells.add((r, c))
            r += ddr
            c += ddc
            step += 1
    return cells


def _apply_segment_output(
    out: np.ndarray,
    bg: int,
    path: list[tuple[int, int]],
    trigger_positions: list[tuple[int, int]],
) -> None:
    H, W = out.shape
    perp_dirs = _perp_dirs_from_path(path)
    trigger_set = set(trigger_positions)

    for idx, (r, c) in enumerate(path):
        if (r, c) not in trigger_set:
            out[r, c] = _PATH_CYCLE[idx % len(_PATH_CYCLE)]

    for order, pos in enumerate(trigger_positions, start=1):
        total_len = _ray_total_length(pos, perp_dirs, H, W)
        out[pos] = _trigger_color_from_length(total_len)
        for rr, cc in _dash_cells(pos, perp_dirs, order, H, W):
            if int(out[rr, cc]) == bg:
                out[rr, cc] = _INPUT_TRIGGER


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    expected = inp.copy()
    for seg in segments:
        _apply_segment_output(expected, bg, seg["path"], seg["triggers"])
    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the blue endpoint pairs with a periodic diagonal stripe, then convert on-path magenta seeds into length-coded emitters whose perpendicular beams become indexed magenta dashes.

    Input structure:
    - bg is the most common color and is not one of the reserved output colors.
    - Blue pixels mark segment endpoints; each real segment is defined by a same-diagonal blue pair.
    - Magenta pixels that lie on those blue-to-blue diagonals are active trigger seeds; off-diagonal magenta pixels are inert distractors.

    Rule:
    - For every diagonal formed by blue endpoints, connect the extreme blue pair and recolor the path with a fixed blue/teal repeating cycle.
    - Active trigger seeds on that path do not stay magenta: their output color is red, yellow, or green according to the total length of their perpendicular diagonal extension corridor.
    - From the first trigger on a segment, emit magenta dashes with gap 1; from the second trigger, gap 2; from the third, gap 3; continue analogously in both perpendicular diagonal directions until the grid edge.

    Key insight:
    - The input still uses only blue anchors and magenta seeds, but the output fuses three independent cues: the main diagonal's periodic color rhythm, the per-trigger dash spacing derived from trigger order along that diagonal, and the trigger pixel's own recolor derived from extension length.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = int(get_bg(grid))

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (distance, color)
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

    segments: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []

    def collect_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        triggers = [(r, c) for r, c in path if int(grid[r, c]) == _INPUT_TRIGGER]
        segments.append((path, triggers))

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            collect_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            collect_segment(group[0], group[-1])

    for path, triggers in segments:
        _apply_segment_output(out, bg, path, triggers)

    return out


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
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
        return path
    return None


def _choose_trigger_positions(
    path: list[tuple[int, int]],
    desired_count: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    interior = path[1:-1]
    if len(interior) < desired_count:
        return None

    perp_dirs = _perp_dirs_from_path(path)
    scored = []
    for pos in interior:
        corridor = _full_ray_cells(pos, perp_dirs, H, W)
        if corridor & occupied:
            continue
        if corridor & reserved:
            continue
        total_len = len(corridor)
        scored.append((pos, corridor, total_len, _trigger_bin_index(total_len)))

    if len(scored) < desired_count:
        return None

    for _ in range(_TRIGGER_SELECT_RETRIES):
        pool = scored[:]
        random.shuffle(pool)
        chosen = pool[:desired_count]
        chosen.sort(key=lambda item: path.index(item[0]))
        trigger_positions = [item[0] for item in chosen]
        ray_cells = set().union(*(item[1] for item in chosen))
        if len(ray_cells) != sum(len(item[1]) for item in chosen):
            continue
        return trigger_positions, ray_cells
    return None


def _segment_trigger_bin_set(segments: list[dict], H: int, W: int) -> set[int]:
    bins = set()
    for seg in segments:
        perp_dirs = _perp_dirs_from_path(seg["path"])
        for pos in seg["triggers"]:
            bins.add(_trigger_bin_index(_ray_total_length(pos, perp_dirs, H, W)))
    return bins


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg_choices = [c for c in range(10) if c not in _RESERVED_COLORS]
    bg = random.choice(bg_choices)

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    segments: list[dict] = []

    n_segments = random.randint(*_SEGMENT_COUNT)
    for seg_idx in range(n_segments):
        desired_count_range = _PRIMARY_TRIGGER_COUNT if seg_idx == 0 else _SECONDARY_TRIGGER_COUNT
        desired_trigger_count = random.randint(*desired_count_range)

        placed = False
        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(["backslash", "slash"])
            seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
            path = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved)
            if path is None:
                continue
            chosen = _choose_trigger_positions(path, desired_trigger_count, H, W, occupied, reserved)
            if chosen is None:
                continue
            trigger_positions, ray_cells = chosen

            segments.append({"path": path, "triggers": trigger_positions})
            occupied.update({path[0], path[-1]})
            occupied.update(trigger_positions)
            reserved.update(set(path[1:-1]) - set(trigger_positions))
            reserved.update(ray_cells)
            placed = True
            break

        if not placed and seg_idx == 0:
            return None

    if not segments:
        return None

    total_triggers = sum(len(seg["triggers"]) for seg in segments)
    if total_triggers < 2:
        return None
    if max(len(seg["triggers"]) for seg in segments) < 2:
        return None
    if len(_segment_trigger_bin_set(segments, H, W)) < 2:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        path = seg["path"]
        inp[path[0]] = _LINE_ANCHOR
        inp[path[-1]] = _LINE_ANCHOR
        for pos in seg["triggers"]:
            inp[pos] = _INPUT_TRIGGER

    all_line_positions = [tuple(pos) for pos in np.argwhere(inp == _LINE_ANCHOR)]
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_ANCHOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    active_trigger_positions = {pos for seg in segments for pos in seg["triggers"]}
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if any((r, c) in _diag_path(seg["path"][0], seg["path"][-1]) for seg in segments):
                continue
            if (r, c) in active_trigger_positions:
                continue
            inp[r, c] = _INPUT_TRIGGER
            occupied.add((r, c))
            break

    expected = _build_expected(inp, bg, segments)
    if not np.any(expected == _PATH_CYCLE[1]):
        return None
    try:
        result = main(inp)
    except Exception:
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
