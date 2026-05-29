# Combo variant of db695cfb (index 26)
# combo_spec: D3/1+D5/1
# Name: 长度编码涂色+固定长度延伸

"""
Design analysis for the D3/1 + D5/1 combo variant.

a) Concrete behavior and trace
- The blue endpoints still define 45-degree segments, and magenta pixels still act as triggers
  only when they lie on those blue diagonals. The blue segment itself is filled exactly as in the
  original task.
- The fused change is that a trigger no longer emits boundary-reaching magenta rays. Instead, its
  perpendicular rays stop after a fixed number of cells equal to the trigger's diagonal distance
  to the nearer blue endpoint, and that same length also selects the ray color.
- Example on a backslash segment from (1,1) to (7,7):
  a trigger at (3,3) has nearest-endpoint distance 2, so it emits two slash rays of length 2 and
  those rays use the short-ray color. A trigger at (4,4) on the same segment has distance 3, so
  it emits longer slash rays of length 3 and those rays use the long-ray color. In the original
  task both would have continued to the grid edge in magenta.

b) Complexity calibration
- The fusion keeps only one numeric driver: trigger-to-nearest-endpoint distance. That single
  quantity controls both geometry and color, so the task is richer than D3 alone without adding
  separate hidden mappings.
- To keep the color rule inferable, the generator always includes at least one short ray and one
  long ray in the same sample. The threshold is binary and fixed: lengths <= 2 use one color,
  lengths >= 3 use the other.
- This avoids over-fragmenting the color mapping while still making the visual difference from the
  original task obvious.

c) Implicit-assumption review of the original main()
- The longest same-color diagonal pair heuristic is preserved, so generation must still prevent
  non-blue colors from forming competing diagonal pairs in the input.
- Grouping blue points by diagonal key and connecting each group's extremes still assumes at most
  one intended segment per diagonal id for each slope. Generation enforces unique diagonal ids.
- The original trigger collector stored only perpendicular directions because all rays went to the
  border in the trigger color. That assumption breaks here: the solver must store a fixed length
  per emitted direction and derive the final paint color from that length.
- If a trigger were shared by multiple segments, different lengths could be proposed. The solver
  resolves that by keeping the maximum length per direction; generation avoids such overlaps.

d) Placement legality
- A legal trigger must support both perpendicular rays fully in-bounds for its fixed length, and
  the cell immediately beyond each endpoint must also stay in-bounds. That extra gap makes the
  finite stop visible instead of looking boundary-clipped.
- Segment paths, actual ray cells, and the post-ray gap cells are all reserved so later segments
  and distractors cannot collide with them.
- The generator also keeps magenta distractors off all intended diagonals and prevents them from
  forming diagonal pairs with one another, which protects the line-color heuristic.

e) Visual observability
- A human can read the controlling quantity directly from the picture: count from the trigger to
  the nearer blue endpoint along the filled blue segment.
- The same sample then shows that shorter counted rays and longer counted rays stop at different
  distances and also use different colors, making the dual encoding observable from structure
  rather than from raw color ids.
- The generator rejects cases where the fused output would collapse to the original boundary-based
  behavior or to a single-color fixed-length baseline.

f) Visual cleanliness
- Only two ray colors are introduced, and both are used solely on the perpendicular rays. Blue
  remains reserved for the main diagonals and magenta remains reserved for trigger dots.
- Buffers after finite rays create visible whitespace beyond the endpoints, so the stop locations
  remain legible. Overlapping rays are avoided during generation.

g) Change inferability
- The color mapping is intentionally binary: short rays use red, long rays use green. Because each
  generated example contains both categories, the viewer sees repeated support for the mapping
  instead of isolated one-off colors.
- The colors are not chosen by palette order or any non-visual attribute; they are tied to a
  visible geometric property of the rays.

i) New solver assumptions checked against generation
- main() assumes every functional trigger lies on exactly one intended blue segment and that the
  segment extremes recover the relevant nearest-endpoint distance. Generation enforces unique blue
  diagonals and forbids extra blue points on those diagonals.
- main() assumes the short/long threshold is global and fixed. Generation uses the same threshold
  when constructing expected outputs.
- main() allows clipping if malformed inputs ask for too-long rays, but generation only emits
  triggers whose full fixed rays and one-cell visibility gaps remain in-bounds.
"""

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SHORT_RAY_COLOR = 2
_LONG_RAY_COLOR = 3
_SHORT_RAY_MAX_LEN = 2


def _ray_color(ray_len: int) -> int:
    return _SHORT_RAY_COLOR if ray_len <= _SHORT_RAY_MAX_LEN else _LONG_RAY_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect diagonal blue endpoint pairs, then emit perpendicular finite rays whose length and color are both distance-coded.

    Input structure:
    - The background is the most common color. Blue pixels appear as endpoint pairs on 45-degree diagonals, and magenta trigger pixels may sit strictly between those endpoints on the same diagonals.

    Rule:
    - Fill each blue diagonal segment from one extreme endpoint to the other.
    - Whenever the segment passes through a non-blue trigger pixel, emit the two perpendicular diagonal rays for a fixed number of cells equal to that trigger's distance to the nearer blue endpoint. Paint short rays in one color and long rays in another, using the same global length threshold everywhere.

    Key insight:
    - One visible measurement drives two output properties at once: the trigger's nearest-endpoint distance determines both how far the perpendicular rays extend and which color category those rays receive.
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

    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        steps = len(path) - 1
        for idx, (r, c) in enumerate(path):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                ray_len = min(idx, steps - idx)
                if ray_len <= 0:
                    continue
                key = (r, c, v)
                dir_lengths = triggers.setdefault(key, {})
                for perp in perp_dirs:
                    dir_lengths[perp] = max(dir_lengths.get(perp, 0), ray_len)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, _tcol), dir_lengths in triggers.items():
        for (ddr, ddc), ray_len in dir_lengths.items():
            ray_col = _ray_color(ray_len)
            for step in range(1, ray_len + 1):
                r = r0 + step * ddr
                c = c0 + step * ddc
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = ray_col

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)
_MIN_VISIBLE_RAY_LEN = 2
_ANCHOR_MIN_SEG_LEN = 6

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_ANCHOR_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _trace_fixed_ray(
    start: tuple[int, int], direction: tuple[int, int], ray_len: int, H: int, W: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    r0, c0 = start
    ddr, ddc = direction
    cells = []
    for step in range(1, ray_len + 1):
        r = r0 + step * ddr
        c = c0 + step * ddc
        if not (0 <= r < H and 0 <= c < W):
            return None
        cells.append((r, c))

    gap = (r0 + (ray_len + 1) * ddr, c0 + (ray_len + 1) * ddc)
    if not (0 <= gap[0] < H and 0 <= gap[1] < W):
        return None
    return cells, gap


def _compute_trigger_bundle(
    trigger_pos: tuple[int, int], ray_len: int, perp_dirs, H: int, W: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    ray_cells = set()
    buffer_cells = set()
    for direction in perp_dirs:
        traced = _trace_fixed_ray(trigger_pos, direction, ray_len, H, W)
        if traced is None:
            return None
        cells, gap = traced
        ray_cells.update(cells)
        buffer_cells.add(gap)
    return ray_cells, buffer_cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for trigger_pos, ray_len in seg["trigger_lengths"].items():
            ray_col = _ray_color(ray_len)
            for ddr, ddc in seg["perp_dirs"]:
                for step in range(1, ray_len + 1):
                    r = trigger_pos[0] + step * ddr
                    c = trigger_pos[1] + step * ddc
                    if int(expected[r, c]) == bg:
                        expected[r, c] = ray_col
    return expected


def _build_boundary_baseline(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for trigger_pos in seg["trigger_lengths"]:
            for ddr, ddc in seg["perp_dirs"]:
                r = trigger_pos[0] + ddr
                c = trigger_pos[1] + ddc
                while 0 <= r < H and 0 <= c < W:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _build_single_color_baseline(inp: np.ndarray, bg: int, segments, ray_color: int) -> np.ndarray:
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for trigger_pos, ray_len in seg["trigger_lengths"].items():
            for ddr, ddc in seg["perp_dirs"]:
                for step in range(1, ray_len + 1):
                    r = trigger_pos[0] + step * ddr
                    c = trigger_pos[1] + step * ddc
                    if int(expected[r, c]) == bg:
                        expected[r, c] = ray_color
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            diag_id = r0 - c0
            if diag_id in used_backslash:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            diag_id = r0 + c0
            if diag_id in used_slash:
                continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in reserved for rc in path):
            continue
        return path, dc
    return None


def _candidate_triggers_for_segment(
    path: list[tuple[int, int]],
    perp_dirs,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
):
    seg_len = len(path) - 1
    candidates = []
    for idx in range(1, seg_len):
        ray_len = min(idx, seg_len - idx)
        if ray_len < _MIN_VISIBLE_RAY_LEN:
            continue
        trigger_pos = path[idx]
        bundle = _compute_trigger_bundle(trigger_pos, ray_len, perp_dirs, H, W)
        if bundle is None:
            continue
        ray_cells, buffer_cells = bundle
        if (ray_cells | buffer_cells) & (occupied | reserved):
            continue
        candidates.append(
            {
                "pos": trigger_pos,
                "ray_len": ray_len,
                "ray_cells": ray_cells,
                "buffer_cells": buffer_cells,
                "category": "short" if ray_len <= _SHORT_RAY_MAX_LEN else "long",
            }
        )
    return candidates


def _footprint(cand) -> set[tuple[int, int]]:
    return set(cand["ray_cells"]) | set(cand["buffer_cells"])


def _select_anchor_pair(candidates):
    shorts = [cand for cand in candidates if cand["category"] == "short"]
    longs = [cand for cand in candidates if cand["category"] == "long"]
    pairs = []
    for short_cand in shorts:
        short_fp = _footprint(short_cand)
        for long_cand in longs:
            if short_fp & _footprint(long_cand):
                continue
            pairs.append((short_cand, long_cand))
    if not pairs:
        return None
    return list(random.choice(pairs))


def _select_trigger_subset(candidates, max_count: int):
    shuffled = list(candidates)
    random.shuffle(shuffled)
    chosen = []
    blocked = set()
    for cand in shuffled:
        footprint = _footprint(cand)
        if footprint & blocked:
            continue
        chosen.append(cand)
        blocked.update(footprint)
        if len(chosen) >= max_count:
            break
    return chosen


def _record_segment(segments, occupied, reserved, used_backslash, used_slash, slope_type, path, dc, chosen) -> None:
    trigger_positions = [cand["pos"] for cand in chosen]
    trigger_lengths = {cand["pos"]: cand["ray_len"] for cand in chosen}
    ray_cells = set().union(*(cand["ray_cells"] for cand in chosen)) if chosen else set()
    buffer_cells = set().union(*(cand["buffer_cells"] for cand in chosen)) if chosen else set()
    ep0, ep1 = path[0], path[-1]
    segments.append(
        {
            "ep0": ep0,
            "ep1": ep1,
            "trigger_positions": trigger_positions,
            "trigger_lengths": trigger_lengths,
            "path": path,
            "perp_dirs": [(1, -dc), (-1, dc)],
        }
    )
    occupied.update([ep0, ep1])
    occupied.update(trigger_positions)
    reserved.update(set(path[1:-1]) - set(trigger_positions))
    reserved.update(ray_cells)
    reserved.update(buffer_cells)
    if slope_type == "backslash":
        used_backslash.add(path[0][0] - path[0][1])
    else:
        used_slash.add(path[0][0] + path[0][1])


def _place_anchor_segment(H, W, occupied, reserved, used_backslash, used_slash):
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _ANCHOR_MIN_SEG_LEN:
        return None

    for _ in range(_ANCHOR_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_ANCHOR_MIN_SEG_LEN, max_seg_len)
        result = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved, used_backslash, used_slash)
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]
        candidates = _candidate_triggers_for_segment(path, perp_dirs, H, W, occupied, reserved)
        chosen = _select_anchor_pair(candidates)
        if chosen is None:
            continue
        return slope_type, path, dc, chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice(
        [c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _SHORT_RAY_COLOR, _LONG_RAY_COLOR)]
    )

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _ANCHOR_MIN_SEG_LEN:
        return None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    segments = []
    used_backslash: set[int] = set()
    used_slash: set[int] = set()

    anchor = _place_anchor_segment(H, W, occupied, reserved, used_backslash, used_slash)
    if anchor is None:
        return None
    slope_type, path, dc, chosen = anchor
    _record_segment(segments, occupied, reserved, used_backslash, used_slash, slope_type, path, dc, chosen)

    for _ in range(n_segments - 1):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, H, W, occupied, reserved, used_backslash, used_slash)
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]
        candidates = _candidate_triggers_for_segment(path, perp_dirs, H, W, occupied, reserved)
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(candidates))
        if max_triggers == 0:
            chosen = []
        else:
            want = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
            chosen = _select_trigger_subset(candidates, want)
        _record_segment(segments, occupied, reserved, used_backslash, used_slash, slope_type, path, dc, chosen)

    short_count = 0
    long_count = 0
    for seg in segments:
        for ray_len in seg["trigger_lengths"].values():
            if ray_len <= _SHORT_RAY_MAX_LEN:
                short_count += 1
            else:
                long_count += 1
    if short_count == 0 or long_count == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for tp in seg["trigger_positions"]:
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

    all_trigger_positions = [pos for seg in segments for pos in seg["trigger_positions"]]
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if any(abs(r - tr) == abs(c - tc) and r != tr for tr, tc in all_trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_trigger_positions.append((r, c))
            break

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(expected, _build_boundary_baseline(inp, bg, segments)):
        return None
    if np.array_equal(expected, _build_single_color_baseline(inp, bg, segments, _SHORT_RAY_COLOR)):
        return None
    if np.array_equal(expected, _build_single_color_baseline(inp, bg, segments, _LONG_RAY_COLOR)):
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
