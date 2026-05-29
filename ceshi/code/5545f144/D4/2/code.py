# Variant of 5545f144
# Dimension: D4 — 路径规则：移动方向选择 (path)
# Change: 每步恰好一个基数方向的半线上有可达标记，精灵移向该半线上最近的标记；不允许180度反转，只允许前进或90度转弯。... → 对角线移动

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the diagonal-path variant
# a) Concrete behavior:
#    The multi-frame strip structure stays the same, but the sprite now faces and
#    moves on 45-degree rays. A remaining marker must lie on exactly one of the
#    four diagonal half-lines from the head: NE, SE, SW, or NW. The head jumps to
#    the nearest marker on that ray, consumes it, then the sprite is re-stamped in
#    the arrived direction. Small example:
#    - Old rule from head (5,5): marker at (5,8) means "move right 3".
#    - New rule from head (5,5): marker at (2,8) means "move NE 3".
#    - The difference is visible because both the alignment test and the final
#      stamped orientation are diagonal rather than row/column based.
# b) Complexity calibration:
#    The variant keeps only four movement directions, preserving the original
#    branch factor, but swaps row/column alignment for diagonal alignment. This
#    raises geometric reasoning without adding hidden cases. The sprite templates
#    are also diagonal so humans can read direction from the tip instead of having
#    to infer diagonal motion from markers alone.
# c) Implicit-assumption audit:
#    - _choose_target in the original code assumed candidate reachability could be
#      checked using row/column tests. That must become diagonal-ray tests.
#    - The original fit check translated the current orientation to the target and
#      was harmless only because the generator matched that assumption. Here the
#      correct condition is that the sprite fits after re-facing to the candidate
#      diagonal direction, so fits() now uses the candidate direction's template.
#    - _simulate_to_end used agent after the loop without initializing it when no
#      markers remained; the variant initializes the current body before stepping.
# d) Placement legality:
#    The generator still works path-first. Each future head is a single-pixel
#    marker. A candidate marker is accepted only if it lies on the intended
#    diagonal ray, does not touch any earlier sprite body, does not create a
#    nearer alternative on a past ray, and there exists a global translation that
#    keeps every stamped sprite body inside the frame.
# e) Visual observability:
#    The dominant visual difference from the source task is that marker chains run
#    along diagonals and the arrowhead itself points diagonally. Across 2-4
#    examples, a solver can see that the sprite chases the nearest dot on a 45°
#    ray rather than on the same row or column.
# f) Visual cleanliness:
#    Markers remain isolated single pixels of the actor color. The generator keeps
#    them at least Chebyshev distance 2 from any earlier marker or sprite cell, so
#    the sprite body stays legible even though sprite and markers share a color.
# g) Color inferability:
#    Colors are unchanged structurally: background, actor, optional divider. No
#    ordering or branching depends on color id, only on visible geometry.
# i) New solver assumptions and generator guarantees:
#    - _find_agent assumes one diagonal template is the unique largest actor-color
#      cluster in the active frame; generator enforces isolated future markers.
#    - _choose_target assumes all remaining markers are future heads on a single
#      valid diagonal ray at each step; rollout_path enforces that invariant.
#    - main() still assumes the rightmost frame is the latest snapshot; generator
#      keeps the original evenly-spaced cadence.

DIRS4 = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # NE, SE, SW, NW

# Direction indices: 0=NE, 1=SE, 2=SW, 3=NW


def _rot90_cw(pt: tuple[int, int]) -> tuple[int, int]:
    r, c = pt
    return c, -r


def _rotated(offs: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = list(offs)
    for _ in range(turns % 4):
        out = [_rot90_cw(pt) for pt in out]
    return out


_CANONICAL_NE: dict[str, list[tuple[int, int]]] = {
    "hollow3": [(0, 0), (1, -1), (1, 0)],
    "filled4": [(0, 0), (1, -1), (1, 0), (2, -1)],
    "stem4": [(0, 0), (1, -1), (2, -2), (2, -1)],
    "midfilled6": [(0, 0), (1, -1), (1, 0), (2, -2), (2, -1), (2, 0)],
}


def _make_templates() -> dict[str, dict[int, list[tuple[int, int]]]]:
    templates: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for tname, base in _CANONICAL_NE.items():
        dirs: dict[int, list[tuple[int, int]]] = {}
        for d in range(4):
            dirs[d] = _rotated(base, d)
        templates[tname] = dirs
    return templates


TEMPLATES = _make_templates()


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns of a non-background color. Return list of frames and bg.
    """
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


def _diag_direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    dr, dc = br - ar, bc - ac
    if dr == 0 or abs(dr) != abs(dc):
        return None
    if dr < 0 and dc > 0:
        return 0
    if dr > 0 and dc > 0:
        return 1
    if dr > 0 and dc < 0:
        return 2
    if dr < 0 and dc < 0:
        return 3
    return None


def _diag_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _fits_template(
    head: tuple[int, int], tname: str, d: int, h: int, w: int
) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][d]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for pt in markers:
            if _diag_direction(head, pt) == dir_idx:
                out.append((_diag_distance(head, pt), pt))
        out.sort()
        return out

    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        for dist, pt in candidates(dd):
            if _fits_template(pt, tname, dd, h, w):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple diagonal directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, dd, pt = reachable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, pt


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = _sprite_cells(head, tname, d)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, tname, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        d, head = choice
        agent = _sprite_cells(head, tname, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a diagonal-moving triangular sprite until every marker is consumed.

    Input structure:
    - Frames are concatenated left-to-right, with optional full-height divider columns.
    - Each frame contains one diagonal arrow sprite plus same-colored single-pixel markers on the background.

    Rule:
    - Read the rightmost frame as the latest snapshot.
    - From the sprite's head, look only along the four diagonal half-lines (NE/SE/SW/NW).
      Exactly one of those rays has reachable markers; move to the nearest marker on that ray.
    - The head lands on that marker, the marker disappears, and the sprite re-faces that diagonal direction.
      Repeat until no markers remain, then output only the final sprite on a clean background.

    Key insight: The arrowhead and the marker chain are both diagonal, so the next move is determined by the nearest dot on the unique active 45-degree ray.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (diagonal-ray variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # 1..4 frames
FRAME_MAX = 16                   # H, W sampled from [lo, 15]
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 8, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 11)            # 4..10 inclusive
STEP_DIST_RANGE = (2, 5)         # diagonal step lengths 2..4 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + n_steps))
    w_lo = min(W_CAP, max(W_FLOOR, 6 + n_steps))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


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
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

        rel_dir = _diag_direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _diag_distance(old_head, nxt) <= _diag_distance(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
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
            trial_fit = fit_placements + [(nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.append((chosen, next_dir))

    return heads, state_dirs, fit_placements


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []

    row_signs = {1 if DIRS4[d][0] > 0 else -1 for d in dir_set}
    col_signs = {1 if DIRS4[d][1] > 0 else -1 for d in dir_set}
    if row_signs != {-1, 1} or col_signs != {-1, 1}:
        return False, 0, []

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []

    dists = [_diag_distance(heads[i], heads[i + 1]) for i in range(n_steps)]
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


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int | None) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if divider is not None and i < n_frames - 1:
            inp[:, c0 + w] = divider
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

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

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
    if f_markers != set(heads[last_idx + 1 :]):
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
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
