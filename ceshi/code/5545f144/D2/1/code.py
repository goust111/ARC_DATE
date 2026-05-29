# Variant of 5545f144
# Dimension: D2 — 源对象：精灵模板类型 (src)
# Change: 三角形精灵有4种模板（hollow3, filled4, stem4, midfilled6），3-6像素，每种可旋转至... → L形精灵

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left
#
# ------------------------------ Design analysis ------------------------------
# a) Concrete variant + behavior trace
#    The moving object is now a single 5-pixel L pentomino: a 3x2 rectangle with
#    one corner missing. Rotating it yields four visually distinct, non-mirror-
#    equivalent states. We anchor the sprite at a designated head cell and define:
#      Up:    #.
#             ##
#             ##
#      Right: ###
#             ##.
#      Down:  ##
#             ##
#             .#
#      Left:  .##
#             ###
#    The head is the lone outer corner cell from which the offsets are defined.
#    Example trace: if the last snapshot shows the Up sprite with head at (4, 2)
#    and markers at (1, 2) then (1, 6), the first jump lands the head on (1, 2)
#    while keeping the Up stamp legal; the second jump lands on (1, 6) and the
#    sprite rotates to Right. The path rule itself is unchanged; only the visual
#    carrier of direction changes from a triangle tip to an L-shaped missing corner.
#
# b) Complexity calibration
#    Using one L family with four rotations keeps the task readable while making
#    the D2 change obvious. The sprite is larger than a dot (5 pixels), connected,
#    and non-axis-symmetric, so orientation cannot be inferred by cheap symmetry.
#    We do NOT introduce multiple L subclasses, extra colors, or secondary marks;
#    that would add freedom without helping human inference.
#
# c) Hidden-assumption review against the original main logic
#    - Original _find_agent searched all actor-colored pixels for any triangle
#      subset. That relied on template families whose heads were visually obvious.
#      For an L pentomino, mixing isolated dots into subset search is less safe,
#      so the new solver first isolates the unique 5-cell connected component and
#      only then matches rotation/head.
#    - Original _choose_target tested in-bounds using the current orientation's
#      offsets. That happened to be harmless because generation enforced legality.
#      Here the L footprint changes handedness across turns, so we explicitly test
#      the arrival orientation for each candidate target.
#    - Original _simulate_to_end only defined `agent` inside the loop. The last
#      input frame always had markers, but that was an implicit precondition. The
#      new code initializes the current sprite immediately so zero-marker inputs
#      remain well-defined.
#    - Color extraction in main still assumes exactly one actor color besides bg
#      and optional divider. The generator preserves that.
#
# d) Placement legality
#    Generation remains path-first. Every future marker is a future head position.
#    We reject a candidate marker if it touches any existing marker or any stamped
#    sprite cell from an earlier state. We also maintain a global translation
#    interval so every rotated L stamp fits in-grid after placement. This ensures
#    the solver's component split is valid: one 5-cell connected actor component
#    plus isolated 1-cell markers.
#
# e) Visual observability
#    The variant is visible without hidden metadata. In the snapshots and output,
#    the actor is plainly an L pentomino rather than any triangle. The missing
#    corner rotates with the path, so 90-degree turns are readable across examples.
#    No ordering or branching depends on color ids or invisible tags.
#
# f) Visual tidiness
#    Markers remain isolated single pixels, the sprite remains one compact block,
#    and divider columns remain solid and separate. Because markers never touch the
#    sprite, the start frame does not devolve into one large ambiguous cluster.
#
# g) Color-change inference
#    There is no color remapping in this variant. Background, actor, and divider
#    roles are unchanged, avoiding an orthogonal reasoning burden.
#
# 9) New main() assumptions and how the generator guarantees them
#    - Exactly one actor-colored 4-connected component has size 5 and matches an
#      L rotation. Generator enforces isolated markers, so this component is the
#      sprite.
#    - All other actor components are singleton markers. Generator's touching test
#      prevents accidental 2+ pixel marker clusters.
#    - At each simulation step, legal reachable markers lie on exactly one half-
#      line from the head, and reverse moves are absent. _marker_compatible_with_
#      existing plus path sampling preserve that invariant.


# Offsets relative to the head cell at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "ell5": {
        0: [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)],     # Up
        1: [(0, 0), (0, -1), (1, -1), (0, -2), (1, -2)],  # Right
        2: [(0, 0), (-1, 0), (-1, -1), (-2, 0), (-2, -1)],  # Down
        3: [(0, 0), (0, 1), (-1, 1), (0, 2), (-1, 2)],    # Left
    },
}
TEMPLATE_NAME = "ell5"
SPRITE_SIZE = 5


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


def _components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(pts)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (ny, nx) in remaining:
                    remaining.remove((ny, nx))
                    comp.add((ny, nx))
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _sprite_cells(head: tuple[int, int], d: int) -> set[tuple[int, int]]:
    hy, hx = head
    return {(hy + dy, hx + dx) for dy, dx in TEMPLATES[TEMPLATE_NAME][d]}


def _match_component(
    comp: set[tuple[int, int]],
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    if len(comp) != SPRITE_SIZE:
        return None

    matches: list[tuple[str, int, tuple[int, int], set[tuple[int, int]]]] = []
    for head in comp:
        hy, hx = int(head[0]), int(head[1])
        for d, offs in TEMPLATES[TEMPLATE_NAME].items():
            placed = {(hy + dy, hx + dx) for dy, dx in offs}
            if placed == comp:
                matches.append((TEMPLATE_NAME, d, (hy, hx), placed))

    if len(matches) != 1:
        return None
    return matches[0]


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    sprite_match = None
    sprite_comp = None
    for comp in _components4(pts):
        matched = _match_component(comp)
        if matched is None:
            if len(comp) != 1:
                return None
            continue
        if sprite_match is not None:
            return None
        sprite_match = matched
        sprite_comp = comp

    if sprite_match is None or sprite_comp is None:
        return None

    tname, d, head, placed = sprite_match
    markers = pts - sprite_comp
    return tname, d, head, placed, markers


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
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
        out.sort()
        return out

    def fits(target: tuple[int, int], dir_idx: int) -> bool:
        ty, tx = target
        for oy, ox in TEMPLATES[TEMPLATE_NAME][dir_idx]:
            ny = ty + oy
            nx = tx + ox
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt, dd):
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

    _tname, d, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = _sprite_cells(head, d)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = _sprite_cells(head, d)
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance an L-shaped sprite through the remaining marker trail and return its final pose.

    Input structure:
    - The input is one or more same-height frames concatenated left-to-right, with
      optional full-height divider columns in a third color.
    - Inside each frame, the actor color is shared by a connected 5-pixel L sprite
      and by isolated 1-pixel markers; the sprite's missing corner indicates facing.

    Rule:
    - Read only the rightmost frame as the current simulation state.
    - From the sprite's head cell, find aligned markers on the same row or column;
      exactly one cardinal half-line has legal targets, and the nearest marker on
      that half-line becomes the next head position.
    - After each landing, consume that marker, rotate the L to the traveled
      direction, and continue until no markers remain; output only the final sprite
      on a clean background.

    Key insight: the actor is the unique 5-cell connected L component, and its
    rotated missing corner tells you which orientation to stamp after each move.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (L-sprite variant, path-first with solver oracle) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # rng.integers low/high+1 -> 1..4 frames
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # total steps 4..12 inclusive
STEP_DIST_RANGE = (2, 8)         # jump lengths 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: retry budget ---
MAX_ATTEMPTS = 300

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _template_bounds(d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[TEMPLATE_NAME][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(d: int, actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[TEMPLATE_NAME][d]:
        patch[dr - r0, dc - c0] = actor
    return patch, -r0, -c0


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
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
    placements: list[tuple[tuple[int, int], int]], h: int, w: int
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None]:
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
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    return n_frames, n_steps, gap, h, w, bg, actor, divider


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
        sprite = _sprite_cells(old_head, state_dirs[j])
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


def _rollout_path(
    rng: np.random.Generator, step_dirs: list[int], h: int, w: int
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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs):
                continue
            trial_fit = fit_placements + [(nxt, next_dir)]
            if _shift_interval(trial_fit, h, w) is None:
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
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(d, actor, bg)
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
    n_frames, n_steps, gap, h, w, bg, actor, divider = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w)
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
        frames.append(_render_frame(h, w, bg, actor, head, d, heads[si + 1:]))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, [])
    inp = _assemble_input(frames, bg, divider)

    split_frames, _ = _split_frames(inp)
    if len(split_frames) != n_frames:
        return None

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, f_agent, f_markers = found
    expected_agent = _sprite_cells(placements[last_idx][0], placements[last_idx][1])
    if (f_tname, f_dir, f_head) != (TEMPLATE_NAME, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_agent != expected_agent:
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
            "template": TEMPLATE_NAME,
            "sprite_pixels": SPRITE_SIZE,
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
