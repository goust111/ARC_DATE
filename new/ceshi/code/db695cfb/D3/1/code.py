# Variant of db695cfb
# Dimension: D3 — 延伸终止条件 (dst)
# Change: 延伸至网格边界——射线从品红像素出发一直画到网格边缘才停止... → 固定长度延伸

"""
Design analysis for the D3 variant.

a) Concrete behavior and trace
- Visible objects stay the same: blue endpoint pairs define 45-degree segments, and magenta
  trigger pixels lie on those segments. The output still fills each segment in blue.
- The only rule change is how magenta rays stop. In the original task, each trigger emitted
  two perpendicular diagonal rays until the grid edge. In this variant, each ray length is the
  diagonal distance from that trigger to the nearer blue endpoint on the same segment.
- Example trace on a backslash segment of 7 cells:
  blue at (1,1) and (7,7), trigger at the middle cell (4,4). Original logic drew magenta along
  the slash diagonal all the way to the border. New logic measures 3 diagonal steps to the nearer
  blue endpoint, then draws exactly 3 cells in each perpendicular slash direction and stops even
  if more empty cells remain before the border.

b) Complexity calibration
- The new parameter is numeric but still local and visually inferable: humans can count how far
  the trigger sits from the nearest blue endpoint, then expect the perpendicular magenta rays to
  have that same count.
- To keep the change observable, the generator forces at least one trigger whose rays stop before
  the border, and it prefers ray lengths of at least 2 cells. That avoids degenerate examples
  where the new output would accidentally match the boundary-based original.

c) Implicit-assumption review of the original main()
- The longest diagonal same-color pair heuristic is kept. It assumes the intended blue endpoints
  form the decisive same-color diagonal pair. The generator preserves that by preventing stray
  magenta distractors from forming diagonal pairs and by keeping blue distractors unpaired.
- Grouping blue points by diagonal and connecting each group's extremes assumes there is at most
  one intended segment per diagonal key. The generator now enforces unique diagonal ids per slope.
- The original trigger collector stored only directions because every ray ran to the border.
  That no longer holds, so this variant stores a per-direction ray length derived from the
  trigger's index along the segment.
- If a trigger were ever shared by multiple segments, lengths could differ. The solver merges
  such cases by keeping the maximum length per emitted direction, while the generator avoids
  such overlaps altogether.

d) Placement legality
- A trigger is legal only if both perpendicular rays of its fixed length stay in-bounds and there
  is at least one extra in-bounds cell beyond each ray endpoint. That guarantees the shortened
  stop is visible and prevents boundary clipping from hiding the D3 change.
- Segment paths, actual ray cells, and one-cell visibility buffers are reserved so later segments
  and distractors do not overlap them.

e) Visual observability
- The counted quantity is geometric and visible: trigger-to-nearest-endpoint distance along the
  already-drawn blue diagonal.
- The generator rejects samples where the fixed-length output would equal the original
  boundary-to-edge output. That ensures the new stopping rule is actually represented.

f) Visual cleanliness
- Reserved buffers keep an empty gap immediately after each shortened magenta ray, making the
  stop point easy to see instead of visually blending into unrelated pixels.
- Ray overlaps and segment merges are avoided in generation so the grid remains readable.

g) Color inference
- No new color mapping is introduced. Blue remains the segment color and magenta remains the
  trigger/ray color, so the only new reasoning burden is measuring distance.

i) New solver assumptions checked against generation
- main() assumes every useful trigger lies on exactly one intended blue segment; generation
  guarantees this by forbidding segment crossings through occupied/reserved cells.
- main() assumes the nearer-endpoint distance is recoverable from the segment extremes; generation
  guarantees no extra blue points appear on those diagonals.
- main() clips if a fixed-length ray would leave the grid, but generation prevents such cases for
  intended triggers so the visible behavior matches the intended fixed-length rule exactly.
"""

import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal pixel pairs, then emit perpendicular finite rays from trigger pixels on those diagonals.

    Input structure:
    - bg = most common color. Blue endpoint pairs mark one or more 45-degree segments, and magenta trigger pixels may lie strictly between those endpoints.

    Rule:
    - Fill each detected blue diagonal segment from one extreme blue endpoint to the other.
    - Whenever that segment passes through a non-blue trigger pixel, emit the two perpendicular diagonal rays in the trigger color, but only for a fixed number of cells equal to the trigger's distance to the nearer blue endpoint on that segment.

    Key insight:
    - The trigger still rotates the segment direction by 90 degrees, but the ray length is encoded by how close the trigger sits to the nearest blue endpoint rather than by the grid boundary.
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

    # Choose the line color as the color owning the longest same-color diagonal pair.
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

    # Each trigger stores the emitted perpendicular directions and the fixed length
    # attached to that direction.
    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant
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

    for (r0, c0, tcol), dir_lengths in triggers.items():
        for (ddr, ddc), ray_len in dir_lengths.items():
            for step in range(1, ray_len + 1):
                r = r0 + step * ddr
                c = c0 + step * ddc
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = tcol

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
_MIN_VISIBLE_RAY_LEN = 2

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


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
            for ddr, ddc in seg["perp_dirs"]:
                for step in range(1, ray_len + 1):
                    r = trigger_pos[0] + step * ddr
                    c = trigger_pos[1] + step * ddc
                    if not (0 <= r < H and 0 <= c < W):
                        break
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
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
            }
        )
    return candidates


def _select_trigger_subset(candidates, max_count: int):
    random.shuffle(candidates)
    chosen = []
    blocked = set()
    for cand in candidates:
        footprint = cand["ray_cells"] | cand["buffer_cells"]
        if footprint & blocked:
            continue
        chosen.append(cand)
        blocked.update(footprint)
        if len(chosen) >= max_count:
            break
    return chosen


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
    used_backslash: set[int] = set()
    used_slash: set[int] = set()

    for _ in range(n_segments):
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
                "perp_dirs": perp_dirs,
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

    total_triggers = sum(len(seg["trigger_positions"]) for seg in segments)
    if not segments or total_triggers == 0:
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

    all_trigger_positions = [
        pos for seg in segments for pos in seg["trigger_positions"]
    ]
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
