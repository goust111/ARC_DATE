# Variant of 5545f144
# Dimension: D17 — 精灵模板种类与选择 (src)
# Change: 4种固定模板（hollow3/filled4/stem4/midfilled6），同一任务内所有帧使用同一模板，高步数任... → 自定义模板由输入编码

from __future__ import annotations

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the D17 variant
# a) Concrete shape / behavior:
#    The leftmost segment is now a dedicated legend panel. It contains a single
#    actor-colored sprite in canonical Up orientation and no markers. The real
#    snapshots remain on the right, still separated by full-height divider
#    columns. The sprite used in the snapshots is no longer selected from a
#    fixed lookup table; instead the solver must read the exact occupied cells
#    from the legend, rotate that template, and then find the sprite inside the
#    rightmost snapshot before continuing the original marker-consumption
#    simulation.
#    Small example: old logic matched the last frame against four baked-in
#    shapes. New logic first extracts offsets such as
#    {(0,0),(1,-1),(1,1),(2,-2),(2,0),(2,2)} from the legend, rotates those
#    offsets to U/R/D/L, and only then searches the last frame for the matching
#    monochrome sprite. The movement rule after recognition is unchanged.
# b) Complexity calibration:
#    The custom template family is still arrow/triangle-like so the task stays
#    visually close to the source puzzle, but the family is larger than the old
#    four-case catalog. Templates always have one unique top tip and 1-2
#    symmetric rows underneath, which makes the legend human-readable while
#    avoiding an explosion of arbitrary polyominoes.
# c) Hidden assumptions audited from the old solver:
#    - Old _find_agent assumed a fixed global template table. That no longer
#      holds, so template generation and matching are now driven by the legend.
#    - Old _choose_target checked in-bounds using the current orientation's
#      offsets. With variable templates this becomes brittle because turned
#      shapes can have different extents, so the fit check now uses the
#      candidate destination orientation.
#    - Actor-color inference still assumes exactly one non-bg non-divider color.
#      The generator preserves that by drawing the legend with the same actor
#      color and introducing no extra legend-only colors.
# d) Placement legality:
#    The generator still rolls out a path first, but every feasibility check now
#    uses the sampled template decoded into four rotations. Shift-interval,
#    sprite-vs-marker spacing, and render-time stamping all share that same
#    dynamic template, ensuring the legend, snapshots, and solver agree.
# e/f) Visual observability and cleanliness:
#    The legend is always a narrow left panel with a lone centered sprite and no
#    markers. That keeps it visually separate from the time snapshots and makes
#    the new rule easy to spot from a few examples. Only three colors are used:
#    bg, actor, divider. No color id participates in ordering or logic.
# g) Color inference:
#    No new color mapping rule is introduced. The only new cue is geometric:
#    the legend shape determines which actor-colored cluster inside the snapshot
#    is the sprite and how it should be rotated during simulation.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Canonical Up-oriented template grammar:
# - row 0: unique head at column 0
# - row 1: one of stem / corners / full width-3
# - row 2: absent / corners / corners+center at width-5
ROW1_OPTIONS: tuple[tuple[int, ...], ...] = (
    (-1, 1),
    (0,),
    (-1, 0, 1),
)
ROW2_OPTIONS: tuple[tuple[int, ...], ...] = (
    (),
    (-2, 2),
    (-2, 0, 2),
)
LEGEND_W = 7


def _normalize_offsets(offs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return sorted({(int(dy), int(dx)) for dy, dx in offs})


def _build_up_template(row1: tuple[int, ...], row2: tuple[int, ...]) -> list[tuple[int, int]]:
    offs = [(0, 0)]
    offs.extend((1, int(c)) for c in row1)
    offs.extend((2, int(c)) for c in row2)
    return _normalize_offsets(offs)


CUSTOM_UP_TEMPLATES: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple(_build_up_template(row1, row2))
    for row1 in ROW1_OPTIONS
    for row2 in ROW2_OPTIONS
    if 1 + len(row1) + len(row2) >= 3
)


ORIGINAL_UP_TEMPLATES = {
    tuple(_build_up_template((-1, 1), ())),
    tuple(_build_up_template((-1, 0, 1), ())),
    tuple(_build_up_template((0,), (-2, 2))),
    tuple(_build_up_template((-1, 0, 1), (-2, 2))),
}


def _template_signature(up_offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(_normalize_offsets(list(up_offsets)))


def _rotate_offset(off: tuple[int, int], d: int) -> tuple[int, int]:
    dy, dx = map(int, off)
    if d == 0:
        return dy, dx
    if d == 1:
        return dx, -dy
    if d == 2:
        return -dy, -dx
    return -dx, dy


def _orient_template(up_offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict[int, list[tuple[int, int]]]:
    return {
        d: _normalize_offsets([_rotate_offset(off, d) for off in up_offsets])
        for d in range(4)
    }


def _template_bounds(offs: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [dy for dy, _ in offs]
    cs = [dx for _, dx in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(offs: list[tuple[int, int]], actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(offs)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dy, dx in offs:
        patch[dy - r0, dx - c0] = actor
    return patch, -r0, -c0


def _sprite_cells(head: tuple[int, int], offs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    hr, hc = head
    return {(int(hr + dy), int(hc + dx)) for dy, dx in offs}


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple segments concatenated horizontally, separated by full-height
    divider columns of a non-background color. Return list of segments and bg.
    """
    _h, w = g.shape
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


def _extract_legend_template(legend: np.ndarray, actor: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(legend == actor)]
    assert pts, "legend must contain exactly one sample sprite"

    top = min(r for r, _ in pts)
    top_pts = [(r, c) for r, c in pts if r == top]
    assert len(top_pts) == 1, "legend sprite must have a unique top tip"

    hy, hx = top_pts[0]
    offs = [(int(r - hy), int(c - hx)) for r, c in pts]
    assert all(dy >= 0 for dy, _ in offs), "legend must encode the Up-oriented template"
    assert (0, 0) in offs, "legend head missing"
    return _normalize_offsets(offs)


def _find_agent(
    frame: np.ndarray,
    actor: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> tuple[int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (score, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for d, offs in template_dirs.items():
            placed = {(hy + dy, hx + dx) for dy, dx in offs}
            if placed.issubset(pts):
                leftover = len(pts - placed)
                score = (len(placed), -leftover)
                if best is None or score > best[0]:
                    best = (score, d, (hy, hx), placed)

    if best is None:
        return None
    _, d, head, placed = best
    markers = pts - placed
    return d, head, set(map(tuple, placed)), set(map(tuple, markers))


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    template_dirs: dict[int, list[tuple[int, int]]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for y, x in markers:
            if dir_idx == 0 and x == hx and y < hy:
                out.append((hy - y, (int(y), int(x))))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (int(y), int(x))))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (int(y), int(x))))
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (int(y), int(x))))
        out.sort()
        return out

    def fits(target: tuple[int, int], dir_idx: int) -> bool:
        ty, tx = target
        for oy, ox in template_dirs[dir_idx]:
            ny = ty + oy
            nx = tx + ox
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt, dd):
                reachable.append((int(dist), int(dd), pt))

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


def _simulate_to_end(
    frame: np.ndarray,
    actor: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor, template_dirs)
    if found is None:
        return frame.copy()

    d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers, template_dirs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in template_dirs[d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a legend-defined sprite and continue its marker-chasing motion.

    Input structure:
    - The leftmost segment is a compact legend panel showing the sprite in
      canonical Up orientation.
    - To the right are one or more time snapshots, horizontally concatenated and
      separated by solid divider columns; the rightmost snapshot is the latest.
    - In each snapshot, the sprite and the single-pixel markers share one actor
      color on a common background.

    Rule:
    - Read the legend to recover the exact sprite template, then rotate that
      template to all four cardinal facings.
    - Use the rightmost snapshot as the current state and identify the sprite by
      matching one rotated copy of the legend template.
    - Repeatedly move the head to the nearest reachable marker on the unique
      valid half-line, consume it, and re-face the sprite in that travel
      direction.
    - When no markers remain, return only the final sprite on a clean background.

    Key insight: The old fixed template catalog is gone; the left legend panel is
    the only reliable source for the sprite's shape, so template decoding must
    happen before any path simulation.
    """
    g = np.asarray(grid, dtype=int)
    segments, bg = _split_frames(g)
    assert len(segments) >= 2, "expected a legend panel plus at least one snapshot"

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    legend = segments[0]
    start = segments[-1]
    up_offsets = _extract_legend_template(legend, actor)
    template_dirs = _orient_template(up_offsets)
    return _simulate_to_end(start, actor, template_dirs)


# Solver/generator contract for the new legend-based assumptions:
# - _extract_legend_template assumes segment 0 is reserved for the legend only
#   and has a unique topmost actor cell. _render_legend enforces that layout.
# - _find_agent assumes the last snapshot contains exactly one rotated copy of
#   the decoded template, with all other actor cells being markers. _build_case
#   checks that oracle condition before accepting a sample.
# - _choose_target still assumes per-step half-line uniqueness and no 180-degree
#   reversals. The path generator keeps the old rollout constraints so those
#   motion assumptions remain true after the D17 template change.
#
# --------------- generator (legend-decoded custom template variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # rng.integers low/high+1 -> 1..4 actual snapshots
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # candidate total_steps (4..12 inclusive)
STEP_DIST_RANGE = (2, 8)         # per-step manhattan distance choices (2..7 inclusive)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
NOVEL_TEMPLATE_BONUS = 1.5

# --- CFG: retry budget ---
MAX_ATTEMPTS = 400

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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


def _template_extent(template_dirs: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    max_h = 0
    max_w = 0
    for offs in template_dirs.values():
        r0, r1, c0, c1 = _template_bounds(offs)
        max_h = max(max_h, r1 - r0 + 1)
        max_w = max(max_w, c1 - c0 + 1)
    return int(max_h), int(max_w)


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]],
    h: int,
    w: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(template_dirs[d])
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return int(tr_lo), int(tr_hi), int(tc_lo), int(tc_hi)


def _sample_template(rng: np.random.Generator, n_steps: int) -> list[tuple[int, int]]:
    weights = []
    for up_tpl in CUSTOM_UP_TEMPLATES:
        sig = tuple(up_tpl)
        w = NOVEL_TEMPLATE_BONUS if sig not in ORIGINAL_UP_TEMPLATES else 1.0
        if n_steps >= TEMPLATE_HEAVY_THRESHOLD:
            w *= len(up_tpl)
        weights.append(w)
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    idx = int(rng.choice(len(CUSTOM_UP_TEMPLATES), p=probs))
    return list(CUSTOM_UP_TEMPLATES[idx])


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, list[tuple[int, int]], dict[int, list[tuple[int, int]]]]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, up_offsets, dirs)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    up_offsets = _sample_template(rng, n_steps)
    template_dirs = _orient_template(up_offsets)
    max_h, max_w = _template_extent(template_dirs)

    h_lo = min(H_CAP, max(H_FLOOR, 3 + max_h + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 2 + max_w + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, divider = int(palette[0]), int(palette[1]), int(palette[2])

    return n_frames, n_steps, gap, h, w, bg, actor, divider, up_offsets, template_dirs


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
        out.extend([int(d)] * int(ln))
    return out


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    template_dirs: dict[int, list[tuple[int, int]]],
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, template_dirs[state_dirs[j]])
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
    rng: np.random.Generator,
    template_dirs: dict[int, list[tuple[int, int]]],
    step_dirs: list[int],
    h: int,
    w: int,
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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, template_dirs):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, template_dirs) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])

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
    template_dirs: dict[int, list[tuple[int, int]]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(template_dirs[d], actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[int(m[0]), int(m[1])] = actor
    return frame


def _render_legend(
    h: int,
    bg: int,
    actor: int,
    up_offsets: list[tuple[int, int]],
) -> np.ndarray:
    legend = np.full((h, LEGEND_W), bg, dtype=int)
    head = (1, LEGEND_W // 2)
    for dy, dx in up_offsets:
        legend[head[0] + dy, head[1] + dx] = actor
    return legend


def _assemble_input(legend: np.ndarray, frames: list[np.ndarray], bg: int, divider: int) -> np.ndarray:
    segments = [legend] + frames
    h = legend.shape[0]
    total_w = sum(seg.shape[1] for seg in segments) + (len(segments) - 1)
    inp = np.full((h, total_w), bg, dtype=int)

    c0 = 0
    for i, seg in enumerate(segments):
        paste(inp, seg, 0, c0)
        c0 += seg.shape[1]
        if i < len(segments) - 1:
            inp[:, c0] = divider
            c0 += 1
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, up_offsets, template_dirs = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, template_dirs, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w, template_dirs)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(int(r + tr), int(c + tc)) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, template_dirs, heads[si + 1 :]))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, template_dirs, [])
    legend = _render_legend(h, bg, actor, up_offsets)
    inp = _assemble_input(legend, frames, bg, divider)

    decoded_up = _extract_legend_template(legend, actor)
    if _template_signature(decoded_up) != _template_signature(up_offsets):
        return None
    decoded_dirs = _orient_template(decoded_up)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor, decoded_dirs)
    if found is None:
        return None
    f_dir, f_head, _f_agent, f_markers = found
    if (f_dir, f_head) != (placements[last_idx][1], placements[last_idx][0]):
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
            "legend_width": LEGEND_W,
            "n_frames": n_frames,
            "template_signature": [list(x) for x in _template_signature(up_offsets)],
            "template_size": len(up_offsets),
            "template_is_original": _template_signature(up_offsets) in ORIGINAL_UP_TEMPLATES,
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
