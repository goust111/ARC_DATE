# Combo variant of 5545f144 (index 7)
# combo_spec: D2/2+D17/2
# Name: 单像素精灵配方向标记+对称模板无明确头部

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg as _bg_color
from arc_utils import paste


# Design analysis for the fused variant
# a) Concrete behavior + trace:
#    Original frames contain a same-color asymmetric triangle plus same-color marker dots;
#    the triangle tip itself reveals direction. Here the visible actor is reduced to a
#    single body pixel plus one adjacent direction pixel of a different color. Example:
#    if body=B at (4,4) and direction marker=D at (4,5), the actor faces right. If the
#    nearest reachable body-colored marker is at (4,8), the body jumps there and the
#    direction pixel moves to (4,9). If the next marker lies at (1,8), the body lands
#    on (1,8) and the direction pixel turns to (0,8). The path logic is unchanged, but
#    the shape no longer encodes orientation; the color pair does.
# b) Complexity calibration:
#    A single-pixel body is much smaller than the original triangles, so the generator
#    keeps the original path complexity constraints (4-12 steps, 2-7 move distances,
#    >=2 turns, >=3 directions, multi-frame snapshots). This preserves nontrivial
#    reasoning and makes the fusion visually obvious rather than collapsing into a
#    trivial dot-chasing task.
# c) Hidden assumptions audit:
#    - Original `_find_agent` assumed the body and markers share one color and that an
#      asymmetric template can be matched directly. That fails here because the body is
#      indistinguishable from markers unless we first locate the unique direction pixel.
#    - Original fit checks depended on rotated triangle bounds. They now depend only on
#      the 2-pixel pair `{body, direction-marker}` staying inside the frame.
#    - Original output rendered only the actor color. Here the final output must retain
#      both the body color and the direction-marker color, otherwise the new rule would
#      be lost in the answer image.
# d) Placement legality:
#    The generator guarantees that every snapshot contains exactly one direction pixel,
#    that it is 4-neighbor adjacent to exactly one body-colored pixel, that no marker
#    overlaps or touches the currently visible 2-pixel actor, and that every future
#    actor state fits in-bounds after translation.
# e) Visual observability:
#    Humans can see many identical body-colored dots plus exactly one special adjacent
#    pointer-colored pair. That makes the control signal explicit in 2-4 examples: the
#    “live” dot is the one attached to the pointer, and the pointer’s side indicates the
#    facing direction after each hop.
# f) Visual cleanliness:
#    Only four colors are used at most: background, body/marker color, direction color,
#    and optional divider. Strong spacing constraints avoid clutter around the pointer,
#    so the active actor is never buried inside marker noise.
# g) Inferability of change:
#    No rule depends on raw color ids. The only color-driven distinction is structural
#    and repeated: one rare color appears once per frame and always sits next to exactly
#    one ordinary dot, making the orientation cue easy to infer from a few examples.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Split horizontally concatenated frames using solid non-background columns."""
    h, w = g.shape
    bg = _bg_color(g)
    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) != bg]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
    return frames, bg


def _pointer_cell(head: tuple[int, int], d: int) -> tuple[int, int]:
    dy, dx = DIRS4[d]
    return head[0] + dy, head[1] + dx


def _agent_cells(head: tuple[int, int], d: int) -> set[tuple[int, int]]:
    return {head, _pointer_cell(head, d)}


def _fits_sprite(head: tuple[int, int], d: int, h: int, w: int) -> bool:
    for y, x in _agent_cells(head, d):
        if not (0 <= y < h and 0 <= x < w):
            return False
    return True


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ay, ax = a
    by, bx = b
    if ax == bx and by < ay:
        return 0
    if ay == by and bx > ax:
        return 1
    if ax == bx and by > ay:
        return 2
    if ay == by and bx < ax:
        return 3
    return None


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _infer_colors(
    frame: np.ndarray, bg: int
) -> tuple[int, int]:
    """Infer `(body_color, direction_color)` from a single frame."""
    colors = [int(c) for c in np.unique(frame) if int(c) != bg]
    assert len(colors) == 2, f"expected exactly 2 non-bg colors in frame, got {colors}"
    counts = {c: int(np.count_nonzero(frame == c)) for c in colors}
    body = max(colors, key=lambda c: counts[c])
    direction = min(colors, key=lambda c: counts[c])
    assert counts[direction] == 1, f"direction color should appear once, got counts={counts}"
    return body, direction


def _find_agent(
    frame: np.ndarray, body_color: int, direction_color: int
) -> tuple[int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pointer_pts = list(map(tuple, np.argwhere(frame == direction_color)))
    if len(pointer_pts) != 1:
        return None
    py, px = map(int, pointer_pts[0])

    body_matches: list[tuple[int, tuple[int, int]]] = []
    for d, (dy, dx) in enumerate(DIRS4):
        hy, hx = py - dy, px - dx
        if 0 <= hy < frame.shape[0] and 0 <= hx < frame.shape[1] and int(frame[hy, hx]) == body_color:
            body_matches.append((d, (hy, hx)))

    if len(body_matches) != 1:
        return None

    d, head = body_matches[0]
    body_pts = set(map(tuple, np.argwhere(frame == body_color)))
    markers = body_pts - {head}
    return d, head, _agent_cells(head, d), markers


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for y, x in markers:
            if dir_idx == 0 and x == hx and y < hy:
                out.append((hy - y, (y, x)))
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
        out.sort()
        return out

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if _fits_sprite(pt, dd, h, w):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    return dd, pt


def _render_state(
    h: int,
    w: int,
    bg: int,
    body_color: int,
    direction_color: int,
    head: tuple[int, int],
    d: int,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in markers:
        frame[y, x] = body_color
    frame[head] = body_color
    frame[_pointer_cell(head, d)] = direction_color
    return frame


def _simulate_to_end(frame: np.ndarray, body_color: int, direction_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    found = _find_agent(frame, body_color, direction_color)
    if found is None:
        return frame.copy()

    d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        choice = _choose_target(head, d, markers, h, w)
        assert choice is not None, f"stuck with {len(markers)} markers at head={head}, dir={d}"
        d, head = choice
        markers.discard(head)

    return _render_state(h, w, bg, body_color, direction_color, head, d, [])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a minimal two-pixel sprite across snapshot frames until all aligned dots are consumed.

    Input structure:
    - Frames are concatenated left-to-right, optionally separated by full-height solid divider columns.
    - Inside each frame, most body-colored single pixels are markers, while the actual sprite is the
      unique body-colored pixel that sits next to a single direction-colored pixel.

    Rule:
    - Start from the rightmost frame.
    - Read the sprite direction from the side on which the direction-colored pixel touches the body.
    - At each step, markers may lie only on one cardinal half-line from the body; jump to the nearest
      marker on that half-line, consume it, and move the direction-colored pixel so it indicates the
      new facing.
    - Continue until no body-colored markers remain, then output only the final two-pixel sprite on
      a clean background.

    Key insight:
    - The body shape is now perfectly symmetric because it is only one pixel, so orientation cannot be
      recovered from shape at all; the adjacent off-color pixel is the sole cue that identifies both
      the live sprite and its facing.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    start = frames[-1]
    body_color, direction_color = _infer_colors(start, bg)
    return _simulate_to_end(start, body_color, direction_color)


# --------------- generator (fused minimal-sprite variant) ---------------

N_FRAMES_RANGE = (2, 5)          # 2..4
FRAME_MAX = 16                   # sizes sampled in [lo, 15]
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEPS_RANGE = (4, 13)            # 4..12
STEP_DIST_RANGE = (2, 8)         # 2..7
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

MAX_ATTEMPTS = 400

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        cells = list(_agent_cells((hr, hc), d))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        tr_lo = max(tr_lo, -min(rows))
        tr_hi = min(tr_hi, h - 1 - max(rows))
        tc_lo = max(tc_lo, -min(cols))
        tc_hi = min(tc_hi, w - 1 - max(cols))
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    # Keep the rightmost snapshot nontrivial: it should still expose at least two
    # future markers so humans can observe the hop-and-turn rule directly.
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0 and n // n_frames >= 2]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = list(map(int, rng.permutation(10)))
    bg, body_color, direction_color = palette[:3]
    divider = palette[3] if n_frames > 1 else None
    return n_frames, n_steps, gap, h, w, bg, body_color, direction_color, divider


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    n_turns = int(rng.integers(turn_min, turn_max + 1))
    n_segments = n_turns + 1

    segments = [int(rng.integers(0, 4))]
    for _ in range(n_segments - 1):
        segments.append(int((segments[-1] + rng.choice([1, -1])) % 4))

    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_steps - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1

    out: list[int] = []
    for d, ln in zip(segments, lengths):
        out.extend([d] * int(ln))
    return out


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
) -> bool:
    for j, old_head in enumerate(heads):
        for cell in _agent_cells(old_head, state_dirs[j]):
            if _touching(nxt, cell):
                return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    n_steps = len(step_dirs)
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [step_dirs[0]]
    placements: list[tuple[tuple[int, int], int]] = [((0, 0), step_dirs[0])]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dy, dx = DIRS4[next_dir]
        chosen = None
        for dist in rng.permutation(np.arange(*STEP_DIST_RANGE)):
            nxt = (cur[0] + int(dist) * dy, cur[1] + int(dist) * dx)
            if nxt in heads:
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs):
                continue
            trial = placements + [(nxt, next_dir)]
            if _shift_interval(trial, h, w) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        placements.append((chosen, next_dir))

    return heads, state_dirs, placements


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, []
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs)))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / len(step_dirs) <= TURN_RATIO_HI):
        return False, turns, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(len(step_dirs))]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists
    return True, turns, dists


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int | None) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    out = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(out, fr, 0, c0)
        if divider is not None and i < n_frames - 1:
            out[:, c0 + w] = divider
    return out


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        n_frames,
        n_steps,
        gap,
        h,
        w,
        bg,
        body_color,
        direction_color,
        divider,
    ) = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(placements, h, w)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [((head[0], head[1]), state_dirs[i]) for i, head in enumerate(heads)]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        markers = heads[si + 1 :]
        frame = _render_state(h, w, bg, body_color, direction_color, head, d, markers)
        found = _find_agent(frame, body_color, direction_color)
        if found is None:
            return None
        fd, fhead, _cells, fmarkers = found
        if (fd, fhead) != (d, head):
            return None
        if fmarkers != set(markers):
            return None
        frames.append(frame)

    final_head, final_dir = placements[-1]
    out_frame = _render_state(h, w, bg, body_color, direction_color, final_head, final_dir, [])
    inp = _assemble_input(frames, bg, divider)

    if not np.array_equal(main(inp), out_frame):
        return None

    last_frame = frames[-1]
    dir_pts = list(map(tuple, np.argwhere(last_frame == direction_color)))
    if len(dir_pts) != 1:
        return None
    adjacent_body = 0
    py, px = dir_pts[0]
    for dy, dx in DIRS4:
        ny, nx = py + dy, px + dx
        if 0 <= ny < h and 0 <= nx < w and int(last_frame[ny, nx]) == body_color:
            adjacent_body += 1
    if adjacent_body != 1:
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "bg": bg,
            "body_color": body_color,
            "direction_color": direction_color,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "body_pixels_in_last_frame": len(heads[(n_frames - 1) * gap + 1 :]) + 1,
            "direction_pixels_per_frame": 1,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
