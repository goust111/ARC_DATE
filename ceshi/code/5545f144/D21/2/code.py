# Variant of 5545f144
# Dimension: D21 — 干扰元素 (meta)
# Change: 无干扰项——网格中所有非背景非分隔线像素都有功能意义（属于精灵或标记）。... → 不可消耗的'墙'标记

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Design analysis for the D21 wall variant
# a) Concrete behavior:
#    - The actor color still serves three roles inside each frame: the triangle
#      sprite, consumable one-pixel markers, and now static two-pixel walls.
#    - A wall is exactly one isolated 2x1 or 1x2 domino of the actor color.
#      Markers remain isolated single pixels. Walls are present in every input
#      frame, are never consumed, and are omitted from the output just like the
#      old task omitted the remaining scene context.
#    - The visibility rule changes at target selection time. Example:
#        head=(5,5), next marker=(5,9), future marker=(1,5),
#        wall={(3,5),(3,6)}.
#      Old logic: both Right and Up look populated, so the state is ambiguous.
#      New logic: the up-ray hits the wall cell at (3,5), so only the rightward
#      single-pixel marker is reachable; the sprite moves right.
# b) Complexity calibration:
#    - The variant is only meaningful if walls alter reachability, not merely if
#      decorative dominoes exist. The generator therefore requires at least one
#      blocked wrong-direction alignment from the rightmost input frame.
#    - Walls are kept few (typically 1-4) and isolated by at least one blank
#      cell moat so that the 1-pixel vs 2-pixel distinction is visually stable.
# c) Hidden assumptions from the original solver:
#    - `_find_agent` previously assumed "all leftover actor pixels are markers".
#      That breaks here; the new parser first splits actor-colored 8-components
#      and classifies them as sprite / singleton markers / size-2 walls.
#    - `_choose_target` previously assumed half-lines contain only markers. The
#      new version computes the nearest wall distance per direction and only
#      considers markers strictly before that wall.
#    - The new solver assumes the generator keeps every non-sprite actor object
#      as an isolated 8-component: size 1 for markers, size 2 for walls.
# d) Placement legality enforced by the generator:
#    - Every sprite placement fits in-bounds as before.
#    - Wall cells never overlap or 8-touch any sprite cell at any snapshot and
#      never 8-touch any marker position; otherwise same-color components would
#      merge and become unparsable.
#    - A wall that blocks a wrong-direction marker must place one of its cells on
#      that ray, but never on any intended next-step ray before the true target.
# e) Visual observability:
#    - Humans can distinguish single markers from walls by repeated structure:
#      singletons disappear when consumed; dominoes persist across frames.
#    - The generator insists that the last input frame already contains at least
#      one blocked alignment, so the new rule is visible where the solver starts.
# f) Visual cleanliness:
#    - Dominoes are sparse and separated from the sprite/markers, preventing
#      noisy merged blobs. The path itself still follows the original geometry.
# g) Color inferability:
#    - No new color mapping is introduced. The entire innovation is carried by
#      shape and line-of-sight, both visible directly in the grids.


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
    comp: set[tuple[int, int]]
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    for head in comp:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            if len(dirs[0]) != len(comp):
                continue
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == comp:
                    return tname, d, (hy, hx), placed
    return None


def _find_scene(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    sprite = None
    markers: set[tuple[int, int]] = set()
    wall_cells: set[tuple[int, int]] = set()

    for comp in _components8(pts):
        matched = _match_sprite_component(comp)
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
    tname, d, head, placed = sprite
    return tname, d, head, set(placed), set(markers), set(wall_cells)


def _nearest_wall_distance(
    head: tuple[int, int], dir_idx: int, wall_cells: set[tuple[int, int]]
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
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def fits(target: tuple[int, int]) -> bool:
        ty, tx = target
        dy, dx = ty - hy, tx - hx
        for oy, ox in offs:
            ny = hy + oy + dy
            nx = hx + ox + dx
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
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
            cand.append((dist, (y, x)))
        cand.sort()
        for dist, pt in cand:
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


def _simulate_trace(frame: np.ndarray, actor: int) -> dict | None:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_scene(frame, actor)
    if found is None:
        return None

    tname, d, head, _agent, markers, wall_cells = found
    markers = set((int(y), int(x)) for y, x in markers)

    trace = [head]
    dirs = [d]
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, wall_cells, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        trace.append(head)
        dirs.append(d)
        markers.discard(head)

    agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}
    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return {
        "trace": trace,
        "dirs": dirs,
        "output": out,
        "tname": tname,
        "wall_cells": wall_cells,
    }


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    traced = _simulate_trace(frame, actor)
    if traced is None:
        return frame.copy()
    return traced["output"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through the final snapshot, consuming only true markers.

    Input structure:
    - The input is a horizontal strip of snapshot frames, optionally split by full-height divider columns.
    - Each frame contains one triangle sprite plus same-colored extras of two kinds:
      isolated single pixels are consumable markers, while isolated 2x1 / 1x2 pairs are walls.

    Rule:
    - Start from the rightmost frame.
    - From the sprite head, look along the four cardinal half-lines; only single-pixel markers
      count as targets, and any wall cell on a half-line blocks markers farther along that ray.
    - Move to the nearest reachable marker, consume it, rotate the sprite to that travel direction,
      and repeat until no markers remain. Walls stay fixed and are never consumed.

    Key insight: same-color noise is not all equivalent here; singleton components are edible markers,
    but domino components are obstacles that carve visibility and decide which turn is legal.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (D21 wall variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # rng.integers low/high+1 -> 1..4 frames
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
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 20000

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

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
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
    heads: list[tuple[int, int]], step_dirs: list[int]
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


def _pair_clear_of_actors(
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
            if not _pair_clear_of_actors(pair, occupied, h, w):
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
    tname: str,
    markers: list[tuple[int, int]],
    walls: list[frozenset[tuple[int, int]]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    for wall in walls:
        for cell in wall:
            frame[cell] = actor
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


def _try_place_walls(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    last_idx: int,
    reqs: dict[tuple[int, int], int],
) -> list[frozenset[tuple[int, int]]] | None:
    placements = [(heads[i], state_dirs[i]) for i in range(len(heads))]
    occupied: set[tuple[int, int]] = set()
    for head, d in placements:
        occupied |= _sprite_cells(head, tname, d)

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
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    reqs = _collect_wall_requirements(heads, step_dirs)
    if reqs is None or not reqs or len(reqs) > MAX_WALL_REQUIREMENTS:
        return None

    req_per_state: dict[int, int] = {}
    for (j, _dir) in reqs:
        req_per_state[j] = req_per_state.get(j, 0) + 1
    if any(v > MAX_REQ_PER_STATE for v in req_per_state.values()):
        return None

    last_idx = (n_frames - 1) * gap
    blocked_last = sum(1 for (j, _dir) in reqs if j == last_idx)
    if blocked_last == 0:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    translations = [(tr, tc) for tr in range(tr_lo, tr_hi + 1) for tc in range(tc_lo, tc_hi + 1)]
    if not translations:
        return None
    order = list(map(int, rng.permutation(len(translations))))
    walls = None
    placed_heads = None
    for idx in order[: min(MAX_TRANSLATION_TRIES, len(order))]:
        tr, tc = translations[idx]
        trial_heads = [(r + tr, c + tc) for r, c in heads]
        trial_walls = _try_place_walls(rng, trial_heads, state_dirs, step_dirs, tname, h, w, last_idx, reqs)
        if trial_walls is None:
            continue
        walls = trial_walls
        placed_heads = trial_heads
        break
    if walls is None or placed_heads is None:
        return None

    heads = placed_heads
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :], walls))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [], [])
    inp = _assemble_input(frames, bg, divider)

    for fi in range(n_frames):
        si = fi * gap
        traced = _simulate_trace(frames[fi], actor)
        if traced is None:
            return None
        if traced["trace"] != heads[si:]:
            return None
        if traced["dirs"] != state_dirs[si:]:
            return None
        if traced["tname"] != tname:
            return None

    found = _find_scene(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers, f_wall_cells = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
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
