# Variant of 5545f144
# Dimension: D8 — 网格结构：帧拼接方式 (grid_shape)
# Change: 输入为多帧水平拼接（分隔线列隔开），输出为单帧；帧是模拟快照，间隔均匀。... → 垂直帧拼接

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


# Design analysis for the D8 variant
# a) Concrete behavior:
#    The actor triangle and its marker dots are unchanged. The only visible rule
#    change is how time snapshots are laid out: instead of [frame|divider|frame]
#    across columns, the input becomes a top-to-bottom stack of equal-width
#    frames separated by one full-width divider row. The newest snapshot is thus
#    the bottom frame rather than the rightmost one.
#    Trace example: if three 9x8 frames used to form a 9x26 strip, the original
#    solver split on divider columns and read the last slice on the right. The
#    variant must split on divider rows and read the last slice at the bottom.
# b) Complexity calibration:
#    To keep the D8 change visible to a human, the generator only emits 2..4
#    frames. Allowing 1 frame would make horizontal vs vertical stacking
#    invisible and would weaken the intended abstraction shift.
# c) Hidden assumptions from the original main():
#    - `np.asarray(grid)` remains valid: the grid is still a plain 2D int array.
#    - `_split_frames` can no longer look for solid columns; it must infer equal
#      frame heights from solid non-background rows.
#    - Divider colors must be gathered from rows, otherwise actor detection would
#      confuse divider and sprite colors.
#    - The latest snapshot is no longer `frames[-1]` because it is "rightmost";
#      it is still `frames[-1]`, but now because it is bottommost.
# d) Placement legality:
#    The rollout, reachability, and no-overlap logic are unchanged because they
#    operate inside a single frame. The generator only changes the outer assembly
#    step, then re-validates with `main(input) == output`.
# e) Visual observability:
#    Multi-frame samples always have divider rows and the input width now matches
#    the output width, while the input height is larger. This makes the stacking
#    direction easy to spot from 2-3 examples.
# f) Visual tidiness:
#    Frames remain separated by exactly one clean divider row in a third color,
#    so the snapshot boundaries are explicit and do not overlap the actor path.
# g) Color inference:
#    No new color mapping is introduced. Background is still the modal color,
#    actor remains the unique non-bg non-divider color, and divider uses a single
#    distinct color.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: every divider is a full-width solid non-bg row and frames all
#      have equal width/height. Guarantee: `_assemble_input` constructs exactly
#      that layout.
#    - Assumption: no extra solid non-bg rows exist inside a frame, so divider
#      detection is unambiguous. Guarantee: `_build_case` rejects such frames.
#    - Assumption: the bottom frame is a valid late snapshot. Guarantee: frames
#      are still sampled at steps 0, g, 2g, ... in order, then stacked vertically.


def _find_divider_rows(g: np.ndarray, bg: int) -> list[int]:
    """Infer divider rows for a vertical equal-frame stack."""
    h, _ = g.shape
    solid_non_bg_rows = [
        r for r in range(h)
        if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != bg
    ]
    if not solid_non_bg_rows:
        return []

    max_frames = min(4, h)
    for n_frames in range(max_frames, 1, -1):
        rem = h - (n_frames - 1)
        if rem <= 0 or rem % n_frames != 0:
            continue
        frame_h = rem // n_frames
        if frame_h <= 0:
            continue
        div_rows = [i * (frame_h + 1) - 1 for i in range(1, n_frames)]
        if div_rows == solid_non_bg_rows:
            return div_rows
    return []


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int, list[int]]:
    """
    Input is multiple frames concatenated vertically, separated by full-width
    divider rows of a non-background color. Return list of frames, bg, and
    divider-row indices.
    """
    h, _ = g.shape
    bg = _bg_color(g)
    div_rows = _find_divider_rows(g, bg)

    if not div_rows:
        return [g.copy()], bg, []

    cuts = [-1] + div_rows + [h]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[a:b, :])
    return frames, bg, div_rows


# Templates are defined as offsets relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    # Hollow triangle (3 pixels)
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],  # Up
        1: [(0, 0), (-1, -1), (1, -1)],  # Right
        2: [(0, 0), (-1, -1), (-1, 1)],  # Down
        3: [(0, 0), (-1, 1), (1, 1)],  # Left
    },
    # Filled triangle (4 pixels: tip + base of 3)
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    # Arrow-like triangle (4 pixels: tip + stem + two base corners)
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
    # 3-row triangle (6 pixels): tip + filled middle row + base corners
    "midfilled6": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    },
}


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (score, template_name, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed.issubset(pts):
                    leftover = len(pts - placed)
                    score = (len(placed), -leftover)
                    if best is None or score > best[0]:
                        best = (score, tname, d, (hy, hx), placed)

    if best is None:
        return None
    _, tname, d, head, placed = best
    markers = pts - placed
    return tname, d, head, set(map(tuple, placed)), set(map(tuple, markers))


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for y, x in markers:
            if dir_idx == 0 and x == hx and y < hy:
                out.append((hy - y, (y, x)))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
        out.sort()
        return out

    def fits(target: tuple[int, int]) -> bool:
        ty, tx = target
        dy, dx = ty - hy, tx - hx
        for oy, ox in offs:
            ny = hy + oy + dy
            nx = hx + ox + dx
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, dd, pt = reachable[0]

    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}

        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through vertically stacked time snapshots until it consumes every aligned marker.

    Input structure:
    - Frames are stacked top-to-bottom and separated by solid full-width divider rows; the background is the most common color.
    - Each frame contains one triangular arrow sprite plus isolated one-pixel markers, all in the same actor color.

    Rule:
    - Begin from the bottom frame, which is the latest snapshot in the stack.
    - From the sprite head, exactly one cardinal half-line contains reachable markers; move to the nearest one on that line.
    - Landing on a marker removes it and reorients the sprite to the travel direction before the next search.
    - Continue until no markers remain, then output a single clean frame containing only the final sprite.

    Key insight: The motion logic is unchanged; the real variant is that time must be segmented by horizontal divider rows and read vertically instead of left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg, div_rows = _split_frames(g)

    div_colors = {int(g[r, 0]) for r in div_rows}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (codex gpt-5.4, adapted for vertical stacking) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)          # 2..4 frames; avoids invisible D8 degeneracy
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15           # clamps for computed H lower bound
W_FLOOR, W_CAP = 6, 15           # clamps for computed W lower bound

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # candidate total_steps (4..12 inclusive)
STEP_DIST_RANGE = (2, 8)         # per-step manhattan distance choices (2..7 inclusive)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3                 # absolute floor on #turns used during segment sampling
MIN_TURNS_ACCEPT = 2             # post-hoc minimum for accepted paths
MIN_DIR_DIVERSITY = 3            # >= 3 of {U,R,D,L} must appear
MIN_DIST_VARIETY = 2             # >= 2 distinct step distances

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 300

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(tname: str, d: int, actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = actor
    return patch, -r0, -c0


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """Cardinal direction from a to b, or None if not axis-aligned."""
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
    """Interval of (tr, tc) translations that keep every placement's template in-grid."""
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, tname)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, divider = int(palette[0]), int(palette[1]), int(palette[2])

    weights_t = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    weights_t /= weights_t.sum()
    tname = str(rng.choice(list(TEMPLATES), p=weights_t))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    """Produce a length-n sequence of step directions with turn ratio in [LO, HI]."""
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


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy path rollout. Returns (heads, state_dirs, fit_placements) or None on stuck step."""
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])

    return heads, state_dirs, fit_placements


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    """Validate candidate marker against all previously-placed heads."""
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    """Check post-hoc path-quality constraints. Returns (ok, turns, step_distances)."""
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, []
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists
    return True, turns, dists


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    return frame


def _has_solid_nonbg_row(frame: np.ndarray, bg: int) -> bool:
    return any(len(set(map(int, frame[r, :]))) == 1 and int(frame[r, 0]) != bg for r in range(frame.shape[0]))


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_h = n_frames * h + (n_frames - 1)
    inp = np.full((total_h, w), bg, dtype=int)
    for i, fr in enumerate(frames):
        r0 = i * (h + 1)
        paste(inp, fr, r0, 0)
        if i < n_frames - 1:
            inp[r0 + h, :] = divider
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1:]))

    if any(_has_solid_nonbg_row(fr, bg) for fr in frames):
        return None

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1:]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "actor": actor,
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
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "stack_axis": "vertical",
            "input_shape": tuple(map(int, inp.shape)),
            "output_shape": tuple(map(int, out_frame.shape)),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
