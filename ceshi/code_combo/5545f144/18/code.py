# Combo variant of 5545f144 (index 18)
# combo_spec: D6/0+D13/2
# Name: 连续轨迹涂色+主从精灵镜像

import numpy as np

from arc_utils import Color
from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TRAIL_COLOR = int(Color.GREY)
AXES = ("vertical", "horizontal")

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Design analysis for the combo variant
# spec:
#   a_concretization_and_trace:
#     - Base 5545f144 has one triangle color and one marker set. After each move the old
#       body disappears, and only the current body remains visible.
#     - This fusion uses two synchronized sprites: a master and a mirror slave. They use
#       different actor colors so the two roles stay visually separable, while both trails
#       are painted in the same fixed grey.
#     - Small trace: if the master starts on the left facing right and its next marker is
#       three cells to the right, the old task would simply teleport the visible triangle.
#       Here the master sweeps a grey footprint across those three translated placements,
#       the slave simultaneously sweeps the mirrored grey footprint on the other half, and
#       both landing triangles remain colored at the end of the step.
#     - Remaining markers are also doubled: master-color markers on the master side and
#       slave-color markers at their reflected positions. Both disappear in lockstep.
#   b_complexity_calibration:
#     - The mirror axis is chosen from only two options, vertical or horizontal, and is
#       always the global center line of the frame. That keeps the new meta-rule visible
#       without introducing hidden per-sample offsets.
#     - Only one extra persistent color is added beyond the two actor colors: grey trail.
#       This is enough to make the D6 change obvious while keeping color inference simple.
#     - Both sprites always share one triangle template, with the slave orientation forced
#       by reflection. Markers always appear as mirrored pairs, so the D13 change is
#       repeated many times inside each example rather than appearing once.
#   c_implicit_assumption_review:
#     - Original main() assumed exactly one non-bg, non-divider color. That breaks here:
#       we now expect exactly two actor colors after removing bg, divider, and grey trail.
#     - Original _find_agent() could read the whole non-bg set at once. Here each color
#       must be analyzed separately, otherwise one sprite's markers would corrupt the
#       other's template match.
#     - Original step selection used all markers in the frame. That would be ambiguous
#       under mirror symmetry, so the solver follows only one color's marker chain and
#       validates that the other color is its exact center reflection.
#     - Original frame splitting treated any solid non-bg column as a divider. A full-height
#       grey trail column is now legal, so divider detection must explicitly ignore grey.
#     - Original output rebuilt a clean canvas containing one final sprite. The fusion
#       must preserve the existing trail, add both future trail segments, and stamp two
#       final sprites on top.
#   d_placement_legality:
#     - The generator samples a master path entirely inside one half-canvas. The slave is
#       produced by reflecting every master cell across the frame center, so the two sides
#       are automatically in-bounds and disjoint.
#     - Future markers are rejected if they would fall on existing trail or current sprite
#       cells. Because the mirror side is a separate half, the same legality then holds for
#       the slave automatically after reflection.
#   e_visual_observability:
#     - A human can see the fused rule from a few examples: there are always two colored
#       triangles, two mirrored colored marker sets, and one grey history layer linking
#       past and future states. The center reflection axis is visible from the geometry
#       itself; no color-id ordering or hidden metadata is used.
#   f_visual_cleanliness:
#     - Restricting the master path to one half of the frame prevents the two trails from
#       weaving through each other. Using one fixed grey trail and two actor colors keeps
#       the three visual layers easy to separate.
#   g_inferable_change:
#     - Colors do not undergo arbitrary remapping. Grey always means historical footprint,
#       master-color pixels belong to the master sprite/markers, and slave-color pixels
#       belong to the mirrored slave sprite/markers. That repeated structural pairing lets
#       the new rule be inferred from 2-3 training examples.


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns of a non-background color. Return list of frames and bg.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) not in {bg, TRAIL_COLOR}]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        if a < b:
            frames.append(g[:, a:b])
    return frames, bg


# Templates are defined as offsets relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],
        1: [(0, 0), (-1, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 1)],
        3: [(0, 0), (-1, 1), (1, 1)],
    },
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
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

    dirs_with_candidates = {dd for _, dd, _ in reachable}
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


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(tname: str, d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = color
    return patch, -r0, -c0


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


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


def _segment_trail_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    tname: str,
    d: int,
) -> set[tuple[int, int]]:
    move_dir = _direction(start, end)
    assert move_dir is not None, f"non-cardinal movement from {start} to {end}"
    dist = _dist(start, end)
    dr, dc = DIRS4[move_dir]

    trail: set[tuple[int, int]] = set()
    for step in range(dist):
        head = (start[0] + step * dr, start[1] + step * dc)
        trail |= _sprite_cells(head, tname, d)
    trail -= _sprite_cells(end, tname, d)
    return trail


def _mirror_dir(d: int, axis: str) -> int:
    if axis == "vertical":
        return (4 - d) % 4 if d % 2 else d
    return d if d % 2 else (2 - d) % 4


def _mirror_point(pt: tuple[int, int], axis: str, h: int, w: int) -> tuple[int, int]:
    y, x = pt
    if axis == "vertical":
        return y, w - 1 - x
    return h - 1 - y, x


def _mirror_cells(
    cells: set[tuple[int, int]], axis: str, h: int, w: int
) -> set[tuple[int, int]]:
    return {_mirror_point(cell, axis, h, w) for cell in cells}


def _mirror_list(
    pts: list[tuple[int, int]], axis: str, h: int, w: int
) -> list[tuple[int, int]]:
    return [_mirror_point(pt, axis, h, w) for pt in pts]


def _infer_axis(
    master_head: tuple[int, int],
    master_dir: int,
    master_agent: set[tuple[int, int]],
    master_markers: set[tuple[int, int]],
    slave_head: tuple[int, int],
    slave_dir: int,
    slave_agent: set[tuple[int, int]],
    slave_markers: set[tuple[int, int]],
    h: int,
    w: int,
) -> str:
    candidates: list[str] = []
    for axis in AXES:
        if _mirror_point(master_head, axis, h, w) != slave_head:
            continue
        if _mirror_dir(master_dir, axis) != slave_dir:
            continue
        if _mirror_cells(master_agent, axis, h, w) != slave_agent:
            continue
        if _mirror_cells(master_markers, axis, h, w) != slave_markers:
            continue
        candidates.append(axis)
    assert len(candidates) == 1, f"expected one mirror axis, got {candidates}"
    return candidates[0]


def _extract_pair_state(
    frame: np.ndarray, master_color: int, slave_color: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], str]:
    h, w = frame.shape
    master_found = _find_agent(frame, master_color)
    slave_found = _find_agent(frame, slave_color)
    assert master_found is not None and slave_found is not None, "both sprites must exist"

    m_tname, m_dir, m_head, m_agent, m_markers = master_found
    s_tname, s_dir, s_head, s_agent, s_markers = slave_found
    assert m_tname == s_tname, "master/slave templates must match"

    axis = _infer_axis(
        m_head,
        m_dir,
        m_agent,
        m_markers,
        s_head,
        s_dir,
        s_agent,
        s_markers,
        h,
        w,
    )
    return m_tname, m_dir, m_head, set((int(y), int(x)) for y, x in m_markers), axis


def _render_state(
    h: int,
    w: int,
    bg: int,
    master_color: int,
    slave_color: int,
    trail_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    master_markers: list[tuple[int, int]],
    axis: str,
    trail_cells: set[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for y, x in trail_cells:
        if 0 <= y < h and 0 <= x < w:
            frame[y, x] = trail_color

    patch, ar, ac = _template_patch(tname, d, master_color, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)

    slave_head = _mirror_point(head, axis, h, w)
    slave_dir = _mirror_dir(d, axis)
    patch, ar, ac = _template_patch(tname, slave_dir, slave_color, bg)
    paste(frame, patch, slave_head[0] - ar, slave_head[1] - ac, transparent=bg)

    for m in master_markers:
        frame[m] = master_color
        frame[_mirror_point(m, axis, h, w)] = slave_color
    return frame


def _simulate_to_end(
    frame: np.ndarray, master_color: int, slave_color: int, trail_color: int
) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _extract_pair_state(frame, master_color, slave_color)
    tname, d, head, markers, axis = found
    trail_cells = set(map(tuple, np.argwhere(frame == trail_color)))

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, TEMPLATES[tname][d], h, w)
        assert choice is not None, (
            f"stuck with {len(markers)} master markers at head={head} dir={d}"
        )
        nd, target = choice

        seg = _segment_trail_cells(head, target, tname, d)
        trail_cells |= seg
        trail_cells |= _mirror_cells(seg, axis, h, w)

        head = target
        d = nd
        markers.discard(head)

    master_agent = _sprite_cells(head, tname, d)
    slave_agent = _mirror_cells(master_agent, axis, h, w)

    out = np.full((h, w), bg, dtype=int)
    for y, x in trail_cells:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = trail_color
    for y, x in master_agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = master_color
    for y, x in slave_agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = slave_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a mirrored two-sprite chase and preserve both swept trails.

    Input structure:
    - One to four frames are concatenated left-to-right, optionally separated by solid divider columns whose color is distinct from the background and the grey trail.
    - Each frame contains a master-colored triangle with master-colored remaining markers, a slave-colored mirrored triangle with mirrored slave-colored markers, and any grey trail cells already painted by earlier motion.

    Rule:
    - Read the rightmost frame as the latest state.
    - Detect the two triangles separately by color, verify that one is the center reflection of the other, and use one side's markers as the controlling path.
    - At each remaining step, the master advances to the nearest reachable marker on its unique cardinal half-line without reversing 180 degrees; the slave performs the reflected move simultaneously.
    - Every intermediate placement swept by either sprite becomes grey trail. Continue until all marker pairs are consumed, then output the full trail together with both final sprite bodies.

    Key insight: color separates the two synchronized roles, while geometry shows that one side is a mirror copy of the other; once the rightmost snapshot is decoded, only one marker chain needs to be simulated and the second trajectory follows by reflection.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) not in {bg, TRAIL_COLOR}}
    colors = set(map(int, g.ravel()))
    actor_colors = sorted(colors - {bg, TRAIL_COLOR} - div_colors)
    assert len(actor_colors) == 2, f"expected two actor colors, got {actor_colors}"

    start = frames[-1]
    return _simulate_to_end(start, actor_colors[0], actor_colors[1], TRAIL_COLOR)


# --------------- generator (continuous trail + mirrored slave) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (2, 5)  # 2..4 frames so the trail is visible in the input
FULL_H_RANGE = (10, 18)  # 10..17 inclusive
FULL_W_RANGE = (10, 18)  # 10..17 inclusive
HALF_RANGE = (7, 10)     # half-canvas width/height 7..9 inclusive

# --- CFG: path complexity ---
STEPS_RANGE = (4, 8)     # 4..7 inclusive; divisible counts keep frame gaps clear
STEP_DIST_RANGE = (2, 6) # 2..5 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2
MIN_TRAIL_CELLS = 12
MIN_LAST_INPUT_TRAIL = 4

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_BASE = (1, 5, 4, 5)
TEMPLATE_WEIGHTS_HEAVY = (1, 6, 5, 6)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 2400
PATH_SEARCH_TRIES = 16

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[str, int, int, int, int, int, int, int, int, str]:
    axis = str(rng.choice(AXES))
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    if axis == "vertical":
        h = int(rng.integers(max(FULL_H_RANGE[0], 6 + (n_steps + 1) // 2), FULL_H_RANGE[1]))
        half_w = int(rng.integers(*HALF_RANGE))
        w = 2 * half_w
        mh, mw = h, half_w
    else:
        half_h = int(rng.integers(*HALF_RANGE))
        h = 2 * half_h
        w = int(rng.integers(max(FULL_W_RANGE[0], 4 + (n_steps + 2) // 2), FULL_W_RANGE[1]))
        mh, mw = half_h, w

    palette = [c for c in range(10) if c != TRAIL_COLOR]
    rng.shuffle(palette)
    bg, master_color, slave_color, divider = map(int, palette[:4])

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else TEMPLATE_WEIGHTS_BASE,
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return axis, n_frames, n_steps, gap, mh, mw, bg, master_color, slave_color, divider, tname


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
    trail_so_far: set[tuple[int, int]],
) -> bool:
    if nxt in trail_so_far:
        return False
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


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]], list[set[tuple[int, int]]]] | None:
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]
    segment_trails: list[set[tuple[int, int]]] = []
    trail_so_far: set[tuple[int, int]] = set()

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        chosen_trail = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname, trail_so_far):
                continue
            seg_trail = _segment_trail_cells(cur, nxt, tname, cur_dir)
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            chosen_trail = seg_trail
            break

        if chosen is None or chosen_trail is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])
        segment_trails.append(chosen_trail)
        trail_so_far |= chosen_trail

    return heads, state_dirs, fit_placements, segment_trails


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


def _trail_prefixes(segment_trails: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = [set()]
    acc: set[tuple[int, int]] = set()
    for seg in segment_trails:
        acc = acc | seg
        out.append(set(acc))
    return out


def _trail_is_compatible(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
    segment_trails: list[set[tuple[int, int]]],
) -> bool:
    prefixes = _trail_prefixes(segment_trails)
    for si, trail in enumerate(prefixes):
        future_markers = heads[si + 1 :]
        if any(m in trail for m in future_markers):
            return False
        sprite = _sprite_cells(heads[si], tname, state_dirs[si])
        if any(m in sprite for m in future_markers):
            return False
    return True


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if i < n_frames - 1:
            inp[:, c0 + w] = divider
    return inp


def _render_mirrored_state(
    axis: str,
    mh: int,
    mw: int,
    bg: int,
    master_color: int,
    slave_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    master_markers: list[tuple[int, int]],
    trail_prefix: set[tuple[int, int]],
) -> np.ndarray:
    if axis == "vertical":
        h, w = mh, 2 * mw
    else:
        h, w = 2 * mh, mw
    full_trail = trail_prefix | _mirror_cells(trail_prefix, axis, h, w)
    return _render_state(
        h,
        w,
        bg,
        master_color,
        slave_color,
        TRAIL_COLOR,
        head,
        d,
        tname,
        master_markers,
        axis,
        full_trail,
    )


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        axis,
        n_frames,
        n_steps,
        gap,
        mh,
        mw,
        bg,
        master_color,
        slave_color,
        divider,
        tname,
    ) = _sample_params(rng)

    path_payload = None
    for _ in range(PATH_SEARCH_TRIES):
        step_dirs = _make_step_dirs(rng, n_steps)
        rollout = _rollout_path(rng, tname, step_dirs, mh, mw)
        if rollout is None:
            continue

        heads, state_dirs, fit_placements, segment_trails = rollout
        ok, turns, dists = _validate_path(step_dirs, heads)
        if ok:
            path_payload = (
                step_dirs,
                heads,
                state_dirs,
                fit_placements,
                segment_trails,
                turns,
                dists,
            )
            break

    if path_payload is None:
        return None
    (
        step_dirs,
        heads,
        state_dirs,
        fit_placements,
        segment_trails,
        turns,
        dists,
    ) = path_payload

    shift = _shift_interval(fit_placements, mh, mw, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    segment_trails = [{(r + tr, c + tc) for r, c in seg} for seg in segment_trails]
    if not _trail_is_compatible(heads, state_dirs, tname, segment_trails):
        return None

    trail_prefixes = _trail_prefixes(segment_trails)
    full_master_trail = trail_prefixes[-1]
    last_idx = (n_frames - 1) * gap
    if len(full_master_trail) < MIN_TRAIL_CELLS:
        return None
    if len(trail_prefixes[last_idx]) < MIN_LAST_INPUT_TRAIL:
        return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(
            _render_mirrored_state(
                axis,
                mh,
                mw,
                bg,
                master_color,
                slave_color,
                head,
                d,
                tname,
                heads[si + 1 :],
                trail_prefixes[si],
            )
        )

    fhead, fdir = placements[-1]
    out_frame = _render_mirrored_state(
        axis,
        mh,
        mw,
        bg,
        master_color,
        slave_color,
        fhead,
        fdir,
        tname,
        [],
        full_master_trail,
    )
    inp = _assemble_input(frames, bg, divider)

    if _bg_color(inp) != bg:
        return None
    split_frames, split_bg = _split_frames(inp)
    if split_bg != bg or len(split_frames) != n_frames or any(fr.shape != frames[0].shape for fr in split_frames):
        return None

    found = _extract_pair_state(frames[-1], master_color, slave_color)
    f_tname, f_dir, f_head, f_markers, found_axis = found
    if (f_tname, f_dir, f_head, found_axis) != (tname, placements[last_idx][1], placements[last_idx][0], axis):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None

    h, w = frames[-1].shape
    expected_last_trail = trail_prefixes[last_idx] | _mirror_cells(trail_prefixes[last_idx], axis, h, w)
    if set(map(tuple, np.argwhere(frames[-1] == TRAIL_COLOR))) != expected_last_trail:
        return None

    if np.count_nonzero(frames[-1] == TRAIL_COLOR) <= 0:
        return None
    if np.count_nonzero(out_frame == TRAIL_COLOR) <= 0:
        return None
    if np.count_nonzero(out_frame == master_color) <= 0 or np.count_nonzero(out_frame == slave_color) <= 0:
        return None

    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "axis": axis,
            "half_height": mh,
            "half_width": mw,
            "frame_height": int(frames[0].shape[0]),
            "frame_width": int(frames[0].shape[1]),
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "master": master_color,
            "slave": slave_color,
            "divider": divider,
            "trail": TRAIL_COLOR,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame_per_side": n_steps,
            "count_in_last_input_frame_per_side": gap,
            "master_trail_cells_output": len(full_master_trail),
            "combined_trail_cells_output": int(np.count_nonzero(out_frame == TRAIL_COLOR)),
            "combined_trail_cells_last_input": int(np.count_nonzero(frames[-1] == TRAIL_COLOR)),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid fused input after {MAX_ATTEMPTS} attempts")
