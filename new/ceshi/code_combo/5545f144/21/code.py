# Combo variant of 5545f144 (index 21)
# combo_spec: D17/0+D21/2
# Name: 自定义模板图例+不可消耗墙标记

from __future__ import annotations

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the fused D17/0 + D21/2 variant
# a) Concreteized behavior and trace:
#    - The leftmost segment is a legend panel that shows the exact sprite
#      template in canonical Up orientation. Real snapshots sit to its right.
#    - Inside each snapshot, actor-colored components now have three meanings:
#        1. one rotated copy of the legend template = the sprite,
#        2. isolated singleton pixels = consumable markers,
#        3. isolated 2-cell dominoes = walls that block rays and persist.
#    - Small example:
#        legend encodes offsets {(0,0),(1,-1),(1,1),(2,-2),(2,0),(2,2)}.
#        In the last frame the sprite is facing Right at head (5,4), there is a
#        marker at (5,8), another marker at (1,4), and a wall on {(3,4),(3,5)}.
#      Old task: a fixed-template matcher finds the sprite, and both Right/Up
#      look populated, so the state is ambiguous.
#      Fused task: the solver first decodes the legend, matches the custom
#      rotated sprite, then notices the Up marker is behind a wall cell on that
#      ray. Only the Right marker is reachable, so the sprite moves right.
# b) Complexity calibration:
#    - The template family stays small and arrow-like: one unique tip plus 1-2
#      symmetric support rows. That keeps the legend readable while still
#      expanding beyond the original four hard-coded shapes.
#    - Walls are sparse dominoes, not arbitrary blobs. They are required to
#      block at least one wrong-direction alignment in the starting snapshot so
#      the wall rule is visually active rather than decorative.
#    - The generator always includes a dedicated legend panel and now keeps at
#      least two snapshots, so the custom-template cue and the wall/marker
#      distinction are both observable directly from the input examples.
# c) Hidden assumptions audited against the old solver:
#    - Fixed template lookup no longer works; every sprite match and fit test
#      must use the legend-decoded offsets.
#    - "All leftover actor pixels are markers" no longer holds; actor-colored
#      components must be split into sprite / singleton / domino classes.
#    - Reachability can no longer be decided by marker alignment alone; a nearer
#      wall on the same half-line suppresses farther markers.
#    - The solver now assumes all non-sprite actor objects are isolated
#      8-components of size 1 or 2, and that the legend contains exactly one
#      Up-oriented sample with a unique top tip. The generator enforces both.
# d) Placement legality:
#    - Every rotated sprite placement must fit in-bounds for every simulated
#      state.
#    - Markers and walls must stay disjoint from all sprite cells and from each
#      other under 8-connectivity; otherwise the component parser would merge
#      them.
#    - A wall may sit on a distracting ray, but never in front of an intended
#      true next target.
# e) Visual observability:
#    - Humans can spot the legend as a dedicated left panel and infer that the
#      in-frame sprite repeats that geometry in rotated form.
#    - Humans can also distinguish singleton markers from 2-cell walls because
#      the walls persist across frames while markers disappear as consumed.
#    - The last snapshot always includes at least one blocked alternative ray, so
#      the wall behavior is already visible at the solver's starting point.
# f) Visual cleanliness:
#    - The legend contains only one sprite on blank background.
#    - Walls are kept few and separated by a blank moat from sprite cells and
#      markers, avoiding same-color merged blobs.
# g) Inferability of the change:
#    - No hidden numeric color ordering is used. Both changes are driven by
#      visible structure only: a dedicated legend region for template selection,
#      and singleton-vs-domino geometry for edible markers versus obstacles.

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

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


def _normalize_offsets(
    offs: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
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


def _template_signature(
    up_offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
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


def _orient_template(
    up_offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> dict[int, list[tuple[int, int]]]:
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


def _template_extent(template_dirs: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    max_h = 0
    max_w = 0
    for offs in template_dirs.values():
        r0, r1, c0, c1 = _template_bounds(offs)
        max_h = max(max_h, r1 - r0 + 1)
        max_w = max(max_w, c1 - c0 + 1)
    return int(max_h), int(max_w)


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
    segments: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        segments.append(g[:, a:b])
    return segments, bg


def _extract_legend_template(legend: np.ndarray, actor: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(legend == actor)]
    assert pts, "legend must contain exactly one sample sprite"

    top = min(r for r, _ in pts)
    top_pts = [(r, c) for r, c in pts if r == top]
    assert len(top_pts) == 1, "legend sprite must have a unique top tip"

    hy, hx = top_pts[0]
    offs = [(int(r - hy), int(c - hx)) for r, c in pts]
    assert (0, 0) in offs, "legend head missing"
    assert all(dy >= 0 for dy, _ in offs), "legend must encode an Up-facing template"
    return _normalize_offsets(offs)


def _components8(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    out: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if (ny, nx) == (y, x):
                        continue
                    if (ny, nx) in remaining:
                        remaining.remove((ny, nx))
                        comp.add((ny, nx))
                        stack.append((ny, nx))
        out.append(comp)
    return out


def _match_sprite_component(
    comp: set[tuple[int, int]],
    template_dirs: dict[int, list[tuple[int, int]]],
) -> tuple[int, tuple[int, int], set[tuple[int, int]]] | None:
    for head in comp:
        hy, hx = int(head[0]), int(head[1])
        for d, offs in template_dirs.items():
            if len(offs) != len(comp):
                continue
            placed = {(hy + dy, hx + dx) for dy, dx in offs}
            if placed == comp:
                return d, (hy, hx), placed
    return None


def _find_scene(
    frame: np.ndarray,
    actor: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> tuple[int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    sprite = None
    markers: set[tuple[int, int]] = set()
    wall_cells: set[tuple[int, int]] = set()

    for comp in _components8(pts):
        matched = _match_sprite_component(comp, template_dirs)
        if matched is not None:
            assert sprite is None, "multiple sprite-shaped components found"
            sprite = matched
            continue
        if len(comp) == 1:
            markers |= comp
            continue
        if len(comp) == 2:
            a, b = tuple(comp)
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                wall_cells |= comp
                continue
        raise AssertionError(f"unrecognized actor component: {sorted(comp)}")

    if sprite is None:
        return None
    d, head, placed = sprite
    return d, head, set(placed), set(markers), set(wall_cells)


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


def _nearest_wall_distance(
    head: tuple[int, int],
    dir_idx: int,
    wall_cells: set[tuple[int, int]],
) -> int | None:
    hy, hx = head
    best = None
    for y, x in wall_cells:
        if dir_idx == 0 and x == hx and y < hy:
            dist = hy - y
        elif dir_idx == 2 and x == hx and y > hy:
            dist = y - hy
        elif dir_idx == 3 and y == hy and x < hx:
            dist = hx - x
        elif dir_idx == 1 and y == hy and x > hx:
            dist = x - hx
        else:
            continue
        if best is None or dist < best:
            best = dist
    return best


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    template_dirs: dict[int, list[tuple[int, int]]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def fits(target: tuple[int, int], dir_idx: int) -> bool:
        ty, tx = target
        for oy, ox in template_dirs[dir_idx]:
            ny = ty + oy
            nx = tx + ox
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        wall_dist = _nearest_wall_distance(head, dd, wall_cells)
        cand: list[tuple[int, tuple[int, int]]] = []
        for y, x in markers:
            if dd == 0 and x == hx and y < hy:
                dist = hy - y
            elif dd == 2 and x == hx and y > hy:
                dist = y - hy
            elif dd == 3 and y == hy and x < hx:
                dist = hx - x
            elif dd == 1 and y == hy and x > hx:
                dist = x - hx
            else:
                continue
            if wall_dist is not None and dist >= wall_dist:
                continue
            cand.append((int(dist), (int(y), int(x))))
        cand.sort()
        for dist, pt in cand:
            if fits(pt, dd):
                reachable.append((int(dist), int(dd), pt))

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


def _simulate_trace(
    frame: np.ndarray,
    actor: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> dict | None:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_scene(frame, actor, template_dirs)
    if found is None:
        return None

    d, head, _agent, markers, wall_cells = found
    markers = set((int(y), int(x)) for y, x in markers)

    trace = [head]
    dirs = [d]
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        choice = _choose_target(head, d, markers, wall_cells, template_dirs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        trace.append(head)
        dirs.append(d)
        markers.discard(head)

    agent = {(head[0] + oy, head[1] + ox) for oy, ox in template_dirs[d]}
    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return {
        "trace": trace,
        "dirs": dirs,
        "output": out,
        "wall_cells": wall_cells,
    }


def _simulate_to_end(
    frame: np.ndarray,
    actor: int,
    template_dirs: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    traced = _simulate_trace(frame, actor, template_dirs)
    if traced is None:
        return frame.copy()
    return traced["output"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend sprite, ignore domino walls as targets, and finish the chase.

    Input structure:
    - The leftmost segment is a compact legend panel containing one Up-oriented
      sample of the sprite template.
    - To its right are one or more time snapshots, separated by full-height
      divider columns; the rightmost snapshot is the current state.
    - In each snapshot, the sprite, singleton markers, and 2-cell wall dominoes
      all use the same actor color on the same background.

    Rule:
    - Read the legend first to recover the exact sprite shape, then rotate that
      shape to all four cardinal facings and locate the sprite in the latest
      snapshot by exact component match.
    - Treat isolated one-pixel components as consumable markers and isolated
      horizontal/vertical dominoes as fixed walls.
    - From the sprite head, move to the nearest marker on the unique reachable
      half-line; any wall cell on that ray blocks markers behind it. Re-face the
      sprite to the travel direction and repeat until no markers remain.
    - Return only the final sprite on a clean background.

    Key insight: same-color pixels now separate by geometry twice over: the left
    legend tells you which component is the sprite template, and inside each
    snapshot singleton dots are edible while domino pairs are persistent blockers.
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


# Solver/generator contract for the fused assumptions:
# - segment 0 is reserved for the legend only; `_render_legend` guarantees a
#   single Up-oriented actor cluster with a unique top tip.
# - every non-legend actor component inside snapshots must be isolated under
#   8-connectivity and classify as exactly one of: sprite, singleton marker,
#   domino wall. `_build_case` and `_try_place_walls` enforce that geometry.
# - `_choose_target` assumes the last-frame start state has a unique reachable
#   direction after accounting for wall blocking and template fit; the path and
#   wall generator verify that by replaying whole-frame traces.

# --------------- generator (legend template + wall fusion) ---------------

# --- CFG: canvas / frame bounds ---
FRAME_COUNT_VALUES = (2, 3, 4)  # keep at least 2 snapshots so wall persistence is observable
FRAME_COUNT_WEIGHTS = (0.45, 0.35, 0.20)
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

# --- CFG: wall behavior ---
MIN_BLOCKABLE_ALIGNMENT_DIST = 4
MAX_WALL_REQUIREMENTS = 4
MAX_REQ_PER_STATE = 2
MAX_TRANSLATION_TRIES = 128

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
NOVEL_TEMPLATE_BONUS = 6.0

# --- CFG: retry budget ---
MAX_ATTEMPTS = 20000
PATH_TRIES_PER_CASE = 32

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    list[tuple[int, int]],
    dict[int, list[tuple[int, int]]],
]:
    n_frames = int(rng.choice(FRAME_COUNT_VALUES, p=np.array(FRAME_COUNT_WEIGHTS, dtype=float)))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(STEPS_RANGE[1] + 1 - n) / _STEP_AVAILABILITY[n] for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    up_offsets = _sample_template(rng, n_steps)
    template_dirs = _orient_template(up_offsets)
    max_h, max_w = _template_extent(template_dirs)

    h_lo = min(H_CAP, max(H_FLOOR, 3 + max_h + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 2 + max_w + (n_steps + 2) // 2))
    h = int(rng.integers(max(h_lo, FRAME_MAX - 3), FRAME_MAX))
    w = int(rng.integers(max(w_lo, FRAME_MAX - 3), FRAME_MAX))

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
        dist = _dist(old_head, nxt)
        if rel_dir == step_dirs[j]:
            if j < len(heads) - 1 and dist <= _dist(old_head, heads[j + 1]):
                return False
        else:
            if dist < MIN_BLOCKABLE_ALIGNMENT_DIST:
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


def _collect_wall_requirements(
    heads: list[tuple[int, int]],
    step_dirs: list[int],
) -> dict[tuple[int, int], int] | None:
    reqs: dict[tuple[int, int], int] = {}
    for j, move_dir in enumerate(step_dirs):
        next_dist = _dist(heads[j], heads[j + 1])
        for k in range(j + 2, len(heads)):
            rel_dir = _direction(heads[j], heads[k])
            if rel_dir is None:
                continue
            dist = _dist(heads[j], heads[k])
            if rel_dir == move_dir:
                if dist <= next_dist:
                    return None
                continue
            if dist < MIN_BLOCKABLE_ALIGNMENT_DIST:
                return None
            key = (j, rel_dir)
            reqs[key] = min(reqs.get(key, 10**9), dist)
    return reqs


def _actual_step_blocked(
    pair: frozenset[tuple[int, int]],
    heads: list[tuple[int, int]],
    step_dirs: list[int],
) -> bool:
    for j, move_dir in enumerate(step_dirs):
        hy, hx = heads[j]
        target_dist = _dist(heads[j], heads[j + 1])
        for y, x in pair:
            if move_dir == 0 and x == hx and 0 < hy - y < target_dist:
                return True
            if move_dir == 2 and x == hx and 0 < y - hy < target_dist:
                return True
            if move_dir == 3 and y == hy and 0 < hx - x < target_dist:
                return True
            if move_dir == 1 and y == hy and 0 < x - hx < target_dist:
                return True
    return False


def _wall_satisfies_requirement(
    pair: frozenset[tuple[int, int]],
    head: tuple[int, int],
    dir_idx: int,
    limit: int,
) -> bool:
    hy, hx = head
    for y, x in pair:
        if dir_idx == 0 and x == hx and 0 < hy - y < limit:
            return True
        if dir_idx == 2 and x == hx and 0 < y - hy < limit:
            return True
        if dir_idx == 3 and y == hy and 0 < hx - x < limit:
            return True
        if dir_idx == 1 and y == hy and 0 < x - hx < limit:
            return True
    return False


def _pair_clear_of_occupied(
    pair: frozenset[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for y, x in pair:
        if not (0 <= y < h and 0 <= x < w):
            return False
        for oy, ox in occupied:
            if _touching((y, x), (oy, ox)):
                return False
    return True


def _pair_clear_of_walls(
    pair: frozenset[tuple[int, int]],
    chosen: list[frozenset[tuple[int, int]]],
) -> bool:
    for other in chosen:
        for a in pair:
            for b in other:
                if _touching(a, b):
                    return False
    return True


def _candidate_walls_for_requirement(
    head: tuple[int, int],
    dir_idx: int,
    limit: int,
    occupied: set[tuple[int, int]],
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    h: int,
    w: int,
) -> list[frozenset[tuple[int, int]]]:
    hy, hx = head
    dr, dc = DIRS4[dir_idx]
    perps = [(dc, -dr), (-dc, dr)]
    out: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for dist in range(2, limit - 1):
        by = hy + dr * dist
        bx = hx + dc * dist
        for py, px in perps:
            pair = frozenset({(by, bx), (by + py, bx + px)})
            if len(pair) != 2 or pair in seen:
                continue
            seen.add(pair)
            if not _pair_clear_of_occupied(pair, occupied, h, w):
                continue
            if _actual_step_blocked(pair, heads, step_dirs):
                continue
            out.append(pair)
    return out


def _search_walls(
    req_items: list[tuple[tuple[int, int], int]],
    candidate_map: dict[tuple[tuple[int, int], int], list[frozenset[tuple[int, int]]]],
    heads: list[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]] | None:
    chosen: list[frozenset[tuple[int, int]]] = []

    def rec(i: int) -> bool:
        if i == len(req_items):
            return True
        key, limit = req_items[i]
        head = heads[key[0]]
        dir_idx = key[1]
        if any(_wall_satisfies_requirement(wall, head, dir_idx, limit) for wall in chosen):
            return rec(i + 1)
        for pair in candidate_map[(key, limit)]:
            if not _pair_clear_of_walls(pair, chosen):
                continue
            chosen.append(pair)
            if rec(i + 1):
                return True
            chosen.pop()
        return False

    if rec(0):
        return chosen
    return None


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    template_dirs: dict[int, list[tuple[int, int]]],
    markers: list[tuple[int, int]],
    walls: list[frozenset[tuple[int, int]]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(template_dirs[d], actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[int(m[0]), int(m[1])] = actor
    for wall in walls:
        for cell in wall:
            frame[int(cell[0]), int(cell[1])] = actor
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
        legend[int(head[0] + dy), int(head[1] + dx)] = actor
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


def _try_place_walls(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    template_dirs: dict[int, list[tuple[int, int]]],
    h: int,
    w: int,
    last_idx: int,
    reqs: dict[tuple[int, int], int],
) -> list[frozenset[tuple[int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    for head, d in zip(heads, state_dirs):
        occupied |= _sprite_cells(head, template_dirs[d])

    req_items = sorted(reqs.items(), key=lambda item: (item[0][0] != last_idx, item[1], item[0][0], item[0][1]))
    candidate_map: dict[tuple[tuple[int, int], int], list[frozenset[tuple[int, int]]]] = {}
    for key, limit in req_items:
        head = heads[key[0]]
        cand = _candidate_walls_for_requirement(head, key[1], limit, occupied, heads, step_dirs, h, w)
        if not cand:
            return None
        rng.shuffle(cand)
        candidate_map[(key, limit)] = cand
    return _search_walls(req_items, candidate_map, heads)


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, up_offsets, template_dirs = _sample_params(rng)
    template_sig = _template_signature(up_offsets)
    walls = None
    placed_heads = None
    step_dirs: list[int] | None = None
    state_dirs: list[int] | None = None
    dists: list[int] | None = None
    turns = 0
    blocked_last = 0

    for _ in range(PATH_TRIES_PER_CASE):
        trial_step_dirs = _make_step_dirs(rng, n_steps)
        rollout = _rollout_path(rng, template_dirs, trial_step_dirs, h, w)
        if rollout is None:
            continue
        heads, trial_state_dirs, fit_placements = rollout

        ok, trial_turns, trial_dists = _validate_path(trial_step_dirs, heads)
        if not ok:
            continue

        reqs = _collect_wall_requirements(heads, trial_step_dirs)
        if reqs is None or not reqs or len(reqs) > MAX_WALL_REQUIREMENTS:
            continue

        req_per_state: dict[int, int] = {}
        for (j, _dir) in reqs:
            req_per_state[j] = req_per_state.get(j, 0) + 1
        if any(v > MAX_REQ_PER_STATE for v in req_per_state.values()):
            continue

        last_idx = (n_frames - 1) * gap
        trial_blocked_last = sum(1 for (j, _dir) in reqs if j == last_idx)
        if trial_blocked_last == 0:
            continue

        shift = _shift_interval(fit_placements, h, w, template_dirs)
        if shift is None:
            continue
        tr_lo, tr_hi, tc_lo, tc_hi = shift
        translations = [(tr, tc) for tr in range(tr_lo, tr_hi + 1) for tc in range(tc_lo, tc_hi + 1)]
        if not translations:
            continue

        order = list(map(int, rng.permutation(len(translations))))
        for idx in order[: min(MAX_TRANSLATION_TRIES, len(order))]:
            tr, tc = translations[idx]
            trial_heads = [(int(r + tr), int(c + tc)) for r, c in heads]
            trial_walls = _try_place_walls(
                rng, trial_heads, trial_state_dirs, trial_step_dirs, template_dirs, h, w, last_idx, reqs
            )
            if trial_walls is None:
                continue
            walls = trial_walls
            placed_heads = trial_heads
            step_dirs = trial_step_dirs
            state_dirs = trial_state_dirs
            dists = trial_dists
            turns = trial_turns
            blocked_last = trial_blocked_last
            break
        if walls is not None:
            break

    if walls is None or placed_heads is None or step_dirs is None or state_dirs is None or dists is None:
        return None

    heads = placed_heads
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(
            _render_frame(h, w, bg, actor, head, d, template_dirs, heads[si + 1 :], walls)
        )

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, template_dirs, [], [])
    legend = _render_legend(h, bg, actor, up_offsets)
    inp = _assemble_input(legend, frames, bg, divider)

    decoded_up = _extract_legend_template(legend, actor)
    if _template_signature(decoded_up) != template_sig:
        return None
    decoded_dirs = _orient_template(decoded_up)

    try:
        for fi in range(n_frames):
            si = fi * gap
            traced = _simulate_trace(frames[fi], actor, decoded_dirs)
            if traced is None:
                return None
            if traced["trace"] != heads[si:]:
                return None
            if traced["dirs"] != state_dirs[si:]:
                return None

        found = _find_scene(frames[-1], actor, decoded_dirs)
        if found is None:
            return None
        f_dir, f_head, _f_agent, f_markers, f_wall_cells = found
    except AssertionError:
        return None

    if (f_dir, f_head) != (placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None
    if len(f_wall_cells) != 2 * len(walls):
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
            "template_signature": [list(x) for x in template_sig],
            "template_size": len(up_offsets),
            "template_is_original": template_sig in ORIGINAL_UP_TEMPLATES,
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
            "wall_count": len(walls),
            "wall_cells": [sorted(wall) for wall in walls],
            "wall_requirements": {f"{j}:{d}": limit for (j, d), limit in reqs.items()},
            "blocked_from_last_frame": blocked_last,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
