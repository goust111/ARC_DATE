# Combo variant of 5545f144 (index 25)
# combo_spec: D4/1+D6/3+D7/2
# Name: 优先级方向+方向颜色轨迹+累积旋转
#
# Design analysis for the fused variant
# =====================================
# a) Concrete behavior and trace
# - Keep the original visual grammar: one monochrome triangle sprite, same-color
#   single-pixel markers, optional divider columns, and now a persistent
#   direction-colored head trail.
# - The fused rule is:
#   1. Read the FIRST frame, because only there the shown facing is untwisted.
#   2. At each state, inspect the current DISPLAYED facing and consider only the
#      three priority directions [straight, right, left].
#   3. In the first occupied priority direction, move to the nearest marker that
#      can host the post-landing sprite orientation.
#   4. Paint the vacated head-path cells using direction colors
#      Up=2, Right=1, Down=3, Left=4.
#   5. If this move changes the movement direction relative to the previous move,
#      add one extra clockwise quarter-turn to a cumulative spin; the landed
#      sprite is stamped with facing = move_dir + spin (mod 4).
# - Small trace:
#   initial head=(6,4), shown facing Up.
#   step 0: markers visible Up and Right; straight wins, so move Up. Trail cells
#           on the segment become red. No turn yet, so landed facing stays Up.
#   step 1: markers visible Right and Left, but none Up; right wins. This is a
#           movement turn, so spin becomes 1 and the landed sprite points Down
#           even though the move itself was Right. The segment just traveled is blue.
#   step 2: priority is now computed from displayed Down, not from the previous
#           move. If both Down and Left markers are visible, Down wins. If this
#           move differs from the previous move, spin increases again.
# - Difference from the source task:
#   original = unique visible direction, no trail, facing equals move direction.
#   fused     = multiple visible directions allowed, trail persists, and the
#               facing used for later priorities drifts away from movement.
#
# b) Complexity calibration
# - Trail colors are fixed to four reused ARC colors, so the mapping is repeated
#   across many segments instead of creating a one-off color puzzle.
# - Actor / background / divider are sampled only from non-trail colors so trail
#   cells never alias with the sprite, markers, or separators.
# - To keep the fusion observable, accepted samples require:
#   - at least one ambiguous decision in the suffix visible from the last frame,
#   - a non-zero final spin so the output orientation differs from plain
#     "face the last movement direction",
#   - at least two frames, so at least one input snapshot already shows trail.
#
# c) Hidden assumptions review
# - Original `_choose_target()` assumed at most one visible direction. That is
#   invalid after D4/1 and becomes a priority scan over straight/right/left.
# - Original solver could start from the rightmost frame because facing matched
#   the last move. D7/2 breaks that: later frames show twisted facings, so the
#   solver must start from the leftmost frame where spin=0.
# - Original divider detection treated every solid non-bg column as a separator.
#   D6/3 breaks that because a full-height trail column may exist, so trail
#   colors must be excluded from divider inference.
# - Original actor extraction assumed exactly one non-bg, non-divider color.
#   Now trail colors are additional non-bg colors and must be removed first.
# - Original generator rejected any aligned future marker from old heads. That
#   becomes too strong after D4/1: lower-priority future markers are legal, so
#   compatibility must compare priority rank and distance instead.
# - New solver assumptions:
#   - step 0's shown facing equals the first movement direction,
#   - every intended move lies in the displayed facing's priority triple,
#   - every higher-priority aligned future marker is absent or farther/blocked
#     enough that the intended marker is still chosen.
#   The generator enforces all of these explicitly.
#
# d) Placement legality
# - Every visible sprite placement uses the derived displayed facing and must fit
#   in-grid after one global translation.
# - Future markers never touch any already-visible sprite body.
# - Trail cells never intersect older trail cells or any later stamped sprite
#   body, which keeps the rightmost snapshots readable.
# - Markers remain isolated single pixels and are never hidden under trail.
#
# e) Visual observability
# - Humans can see all three fused signals:
#   - ambiguous choices are exposed by multiple remaining markers on the shown
#     straight/right/left rays,
#   - colored trail segments reveal movement directions,
#   - later frames show that the triangle orientation no longer always matches
#     the incoming trail direction, revealing cumulative rotation.
# - Because the generator keeps at least one ambiguous suffix step and a nonzero
#   final spin, samples do not collapse back to the source task or to a pure
#   trail-only variant.
#
# f) Visual cleanliness
# - The trail is one-pixel wide along the head path only; this avoids dense body
#   sweeps around corners and keeps the triangle template easy to detect.
# - Trail colors are disjoint from actor/background/divider colors, and markers
#   stay off the trail, so the three element types remain visually separable.
#
# g) Change inferability
# - The color mapping is small and structural: the same four colors recur on
#   axis-aligned segments, and each segment direction is visible from geometry.
# - Priority is inferable because some last-frame suffixes intentionally contain
#   multiple visible candidate directions where the chosen continuation matches
#   straight-over-right-over-left.
# - The extra rotation is inferable because the output triangle and later input
#   frames sometimes disagree with the segment that just entered the head cell.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TRAIL_BY_DIR = {0: 2, 1: 1, 2: 3, 3: 4}
TRAIL_COLORS = set(TRAIL_BY_DIR.values())
NON_TRAIL_COLORS = (0, 5, 6, 7, 8, 9)
ACTOR_COLORS = (5, 6, 7, 8, 9)


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns. Divider columns never use the four trail colors.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [
        c for c in solid_cols if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    ]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
    return frames, bg


def _priority_dirs(facing: int) -> list[int]:
    return [facing, (facing + 1) % 4, (facing + 3) % 4]


def _priority_rank(facing: int, move_dir: int) -> int | None:
    order = _priority_dirs(facing)
    return order.index(move_dir) if move_dir in order else None


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

    best = None
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
    hy, hx = head
    return {(hy + oy, hx + ox) for oy, ox in TEMPLATES[tname][d]}


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


def _segment_cells(
    a: tuple[int, int], b: tuple[int, int], *, include_end: bool
) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = 0 if br == ar else (1 if br > ar else -1)
    dc = 0 if bc == ac else (1 if bc > ac else -1)
    dist = abs(br - ar) + abs(bc - ac)
    stop = dist + 1 if include_end else dist
    return [(ar + dr * k, ac + dc * k) for k in range(stop)]


def _extract_trail(frame: np.ndarray) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for y, x in np.argwhere(np.isin(frame, list(TRAIL_COLORS))):
        trail[(int(y), int(x))] = int(frame[y, x])
    return trail


def _fit_at(head: tuple[int, int], tname: str, facing: int, h: int, w: int) -> bool:
    hy, hx = head
    return all(0 <= hy + oy < h and 0 <= hx + ox < w for oy, ox in TEMPLATES[tname][facing])


def _aligned_markers(
    head: tuple[int, int], markers: set[tuple[int, int]]
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    hy, hx = head
    out: dict[int, list[tuple[int, tuple[int, int]]]] = {0: [], 1: [], 2: [], 3: []}
    for y, x in markers:
        if x == hx and y < hy:
            out[0].append((hy - y, (y, x)))
        elif y == hy and x > hx:
            out[1].append((x - hx, (y, x)))
        elif x == hx and y > hy:
            out[2].append((y - hy, (y, x)))
        elif y == hy and x < hx:
            out[3].append((hx - x, (y, x)))
    for dd in out:
        out[dd].sort()
    return out


def _next_spin(spin: int, step_idx: int, prev_move_dir: int, move_dir: int) -> int:
    if step_idx > 0 and move_dir != prev_move_dir:
        return (spin + 1) % 4
    return spin


def _choose_target(
    head: tuple[int, int],
    facing: int,
    prev_move_dir: int,
    spin: int,
    step_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int], int, int] | None:
    aligned = _aligned_markers(head, markers)
    for move_dir in _priority_dirs(facing):
        new_spin = _next_spin(spin, step_idx, prev_move_dir, move_dir)
        landing_facing = int((move_dir + new_spin) % 4)
        for _, pt in aligned[move_dir]:
            if _fit_at(pt, tname, landing_facing, h, w):
                return move_dir, pt, landing_facing, new_spin
    return None


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    trail = _extract_trail(frame)

    found = _find_agent(frame, actor)
    if found is None:
        out = np.full((h, w), bg, dtype=int)
        for (y, x), color in trail.items():
            out[y, x] = color
        return out

    tname, facing, head, _agent, markers = found
    markers = set((int(y), int(x)) for y, x in markers)
    agent = _sprite_cells(head, tname, facing)

    prev_move_dir = int(facing)
    spin = 0
    step_idx = 0
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, facing, prev_move_dir, spin, step_idx, markers, tname, h, w)
        assert choice is not None, (
            f"stuck with {len(markers)} markers at head={head} facing={facing} "
            f"prev_move={prev_move_dir} spin={spin}"
        )
        move_dir, target, facing, spin = choice

        for cell in _segment_cells(head, target, include_end=False):
            trail[cell] = TRAIL_BY_DIR[move_dir]

        head = target
        prev_move_dir = int(move_dir)
        agent = _sprite_cells(head, tname, facing)
        markers.discard(head)
        step_idx += 1

    out = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = color
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite puzzle where priority, trail, and facing are fused.

    Input structure:
    - Frames are concatenated left-to-right with optional solid divider columns;
      the most common color is the background.
    - Each frame contains one actor-colored triangular sprite, same-colored
      remaining markers, and any existing trail cells:
      red for Up, blue for Right, green for Down, yellow for Left.

    Rule:
    - Start from the leftmost frame, because its shown triangle orientation is
      the untwisted initial facing.
    - Repeatedly inspect markers on the displayed straight/right/left half-lines.
      Choose the nearest marker in the first occupied direction of that priority
      order, paint the vacated head path with that direction's trail color, and
      consume the marker.
    - Whenever the new movement direction differs from the previous movement
      direction, add one clockwise quarter-turn to a cumulative rotation offset.
      Stamp the landed sprite with facing = move direction + accumulated offset.
      Continue until no markers remain.

    Key insight: the trail reveals how the head moves, but later triangle
    orientations drift away from that motion; only the first frame exposes the
    untwisted starting facing that defines the priority chain.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {
        int(g[0, c])
        for c in solid_cols
        if int(g[0, c]) != bg and int(g[0, c]) not in TRAIL_COLORS
    }
    colors = set(map(int, g.ravel()))
    actor_candidates = colors - {bg} - div_colors - TRAIL_COLORS
    assert len(actor_candidates) == 1, f"expected 1 actor color, got {actor_candidates}"
    actor = next(iter(actor_candidates))

    return _simulate_to_end(frames[0], actor)


# --------------- generator (fused D4/1 + D6/3 + D7/2) ---------------

N_FRAMES_RANGE = (2, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15
BRANCH_MODES = ("straight", "right")
BRANCH_WEIGHTS = (0.55, 0.45)

STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 2
MIN_DIST_VARIETY = 2
MIN_SUFFIX_STEPS = 2
MIN_SUFFIX_DIR_DIVERSITY = 1
MIN_AMBIGUOUS_TAIL_STEPS = 1

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 1200


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
) -> tuple[int, int, int, int, int, int, str]:
    step_choices = np.arange(*STEPS_RANGE)
    step_weights = step_choices.astype(float)
    step_weights /= step_weights.sum()
    n_steps = int(rng.choice(step_choices, p=step_weights))

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(max(h_lo, 12), FRAME_MAX))
    w = int(rng.integers(max(w_lo, 10), FRAME_MAX))

    bg = int(rng.choice(NON_TRAIL_COLORS))
    actor = int(rng.choice([c for c in ACTOR_COLORS if c != bg]))
    divider = int(rng.choice([c for c in NON_TRAIL_COLORS if c not in {bg, actor}]))

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_steps, h, w, bg, actor, divider, tname


def _sample_visual_params(
    rng: np.random.Generator, n_steps: int
) -> tuple[int, int, int, int, int, str]:
    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(max(h_lo, 12), FRAME_MAX))
    w = int(rng.integers(max(w_lo, 10), FRAME_MAX))

    bg = int(rng.choice(NON_TRAIL_COLORS))
    actor = int(rng.choice([c for c in ACTOR_COLORS if c != bg]))
    divider = int(rng.choice([c for c in NON_TRAIL_COLORS if c not in {bg, actor}]))

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return h, w, bg, actor, divider, tname


def _rotate_dirs(step_dirs: list[int], rot: int) -> list[int]:
    return [int((d + rot) % 4) for d in step_dirs]


def _heads_from_steps(step_dirs: list[int], step_dists: list[int]) -> list[tuple[int, int]]:
    heads = [(0, 0)]
    for move_dir, dist in zip(step_dirs, step_dists):
        dr, dc = DIRS4[int(move_dir)]
        hr, hc = heads[-1]
        heads.append((hr + int(dist) * dr, hc + int(dist) * dc))
    return heads


def _manual_branch_path(
    rng: np.random.Generator, branch_mode: str
) -> tuple[list[int], list[int], list[int]]:
    rot = int(rng.integers(0, 4))
    if branch_mode == "straight":
        prefix = int(rng.integers(2, 5))
        forward = int(rng.integers(2, 5))
        side = int(rng.integers(4, 7))
        tail = int(rng.integers(2, side))
        step_dirs = _rotate_dirs([0, 0, 1, 2, 3], rot)
        step_dists = [prefix, forward, side, forward, tail]
        snapshot_indices = [0, 1, 3]
    else:
        step_dirs = _rotate_dirs([1, 2, 3, 0, 3, 0, 1], rot)
        step_dists = [3, 5, 3, 3, 3, 3, 6]
        snapshot_indices = [0, 1]
    return step_dirs, step_dists, snapshot_indices


def _step_branch_counts(step_dirs: list[int]) -> dict[str, int]:
    facings, _spins = _derive_facings(step_dirs)
    counts = {"straight": 0, "right": 0, "left": 0}
    for i, move_dir in enumerate(step_dirs):
        move_dir = int(move_dir)
        if move_dir == facings[i]:
            counts["straight"] += 1
        elif move_dir == (facings[i] + 1) % 4:
            counts["right"] += 1
        elif move_dir == (facings[i] + 3) % 4:
            counts["left"] += 1
    return counts


def _make_step_dirs(
    rng: np.random.Generator, n_steps: int, need_branch: str | None = None
) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    for _ in range(200):
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

        if _sequence_respects_priority(out) and (
            need_branch is None or _step_branch_counts(out)[need_branch] > 0
        ):
            return out

    return out


def _derive_facings(step_dirs: list[int]) -> tuple[list[int], list[int]]:
    facings = [int(step_dirs[0])]
    spins = [0]
    spin = 0
    prev_move = int(step_dirs[0])
    for i, move_dir in enumerate(step_dirs):
        move_dir = int(move_dir)
        if i > 0 and move_dir != prev_move:
            spin = (spin + 1) % 4
        facings.append(int((move_dir + spin) % 4))
        spins.append(int(spin))
        prev_move = move_dir
    return facings, spins


def _sequence_respects_priority(step_dirs: list[int]) -> bool:
    facings, _spins = _derive_facings(step_dirs)
    return all(step_dirs[i] in _priority_dirs(facings[i]) for i in range(len(step_dirs)))


def _trail_map(
    heads: list[tuple[int, int]], step_dirs: list[int], upto_steps: int
) -> dict[tuple[int, int], int]:
    trail: dict[tuple[int, int], int] = {}
    for i in range(upto_steps):
        for cell in _segment_cells(heads[i], heads[i + 1], include_end=False):
            trail[cell] = TRAIL_BY_DIR[step_dirs[i]]
    return trail


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, face_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False

    for j in range(len(heads) - 1):
        old_head = heads[j]
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        rel_rank = _priority_rank(face_dirs[j], rel_dir)
        if rel_rank is None:
            continue

        chosen_dir = int(step_dirs[j])
        chosen_rank = _priority_rank(face_dirs[j], chosen_dir)
        if chosen_rank is None:
            return False
        if rel_rank < chosen_rank:
            return False
        if rel_rank == chosen_rank and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False

    return True


def _has_branch_candidate(
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    spins: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    branch_mode: str,
) -> bool:
    n_steps = len(step_dirs)
    for last_idx in range(1, n_steps - MIN_SUFFIX_STEPS + 1):
        prev_move = step_dirs[last_idx - 1]
        visible = _visible_priority_dirs(
            heads[last_idx],
            face_dirs[last_idx],
            prev_move,
            spins[last_idx],
            last_idx,
            set(heads[last_idx + 1 :]),
            tname,
            h,
            w,
        )
        actual_dir = _direction(heads[last_idx], heads[last_idx + 1])
        if actual_dir is None or len(visible) < 2:
            continue
        tail_stats = _priority_stats(heads, face_dirs, spins, step_dirs, tname, h, w, last_idx)
        if tail_stats["twisted_states"] == 0 or tail_stats["ambiguous_steps"] < MIN_AMBIGUOUS_TAIL_STEPS:
            continue
        if branch_mode == "straight" and actual_dir == face_dirs[last_idx]:
            return True
        if branch_mode == "right" and actual_dir == (face_dirs[last_idx] + 1) % 4:
            if (face_dirs[last_idx] + 3) % 4 in visible:
                return True
    return False


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    step_dirs: list[int],
    h: int,
    w: int,
    branch_mode: str | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[tuple[tuple[int, int], int]], set[tuple[int, int]]] | None:
    face_dirs, spins = _derive_facings(step_dirs)
    dist_choices = [int(x) for x in np.arange(*STEP_DIST_RANGE)]

    def dfs(
        step_idx: int,
        heads: list[tuple[int, int]],
        fit_placements: list[tuple[tuple[int, int], int]],
        used_trace: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], int]], set[tuple[int, int]]] | None:
        if step_idx == len(step_dirs):
            if branch_mode is not None and not _has_branch_candidate(
                heads, face_dirs, spins, step_dirs, tname, h, w, branch_mode
            ):
                return None
            return heads, fit_placements, used_trace

        cur = heads[-1]
        next_dir = int(step_dirs[step_idx])
        dr, dc = DIRS4[next_dir]
        next_face = int(face_dirs[step_idx + 1])

        candidates = []
        for order, dist in enumerate(rng.permutation(dist_choices)):
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads or nxt in used_trace:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, face_dirs, step_dirs, tname):
                continue

            seg_trace = _segment_cells(cur, nxt, include_end=False)
            if any(cell in used_trace for cell in seg_trace):
                continue
            if any(cell in heads[:-1] for cell in seg_trace[1:]):
                continue

            trial_fit = fit_placements + [(nxt, next_face)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue

            nxt_sprite = _sprite_cells(nxt, tname, next_face)
            if nxt_sprite & used_trace:
                continue

            align_score = sum(
                int(nxt[0] == old_head[0] or nxt[1] == old_head[1]) for old_head in heads[:-1]
            )
            candidates.append(
                (
                    -align_score,
                    order,
                    nxt,
                    trial_fit,
                    used_trace | set(seg_trace),
                )
            )

        for _score, _order, nxt, trial_fit, next_trace in candidates:
            result = dfs(step_idx + 1, heads + [nxt], trial_fit, next_trace)
            if result is not None:
                return result

        return None

    result = dfs(0, [(0, 0)], [((0, 0), face_dirs[0])], set())
    if result is None:
        return None
    heads, fit_placements, used_trace = result
    return heads, face_dirs, spins, fit_placements, used_trace


def _validate_path(
    step_dirs: list[int],
    heads: list[tuple[int, int]],
    spins: list[int],
    tail_len: int,
) -> tuple[bool, int, list[int]]:
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists

    if spins[-1] == 0:
        return False, turns, dists

    suffix = step_dirs[-tail_len:]
    if len(set(suffix)) < MIN_SUFFIX_DIR_DIVERSITY:
        return False, turns, dists

    return True, turns, dists


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    facing: int,
    tname: str,
    markers: list[tuple[int, int]],
    trail: dict[tuple[int, int], int],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    for (y, x), color in trail.items():
        frame[y, x] = color
    patch, ar, ac = _template_patch(tname, facing, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    return frame


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


def _visible_priority_dirs(
    head: tuple[int, int],
    facing: int,
    prev_move_dir: int,
    spin: int,
    step_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> list[int]:
    aligned = _aligned_markers(head, markers)
    visible: list[int] = []
    for move_dir in _priority_dirs(facing):
        new_spin = _next_spin(spin, step_idx, prev_move_dir, move_dir)
        landing_facing = int((move_dir + new_spin) % 4)
        if any(_fit_at(pt, tname, landing_facing, h, w) for _, pt in aligned[move_dir]):
            visible.append(move_dir)
    return visible


def _priority_stats(
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    spins: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    start_idx: int = 0,
) -> dict[str, int]:
    stats = {
        "ambiguous_steps": 0,
        "straight_priority_wins": 0,
        "right_over_left_wins": 0,
        "twisted_states": 0,
    }
    n_steps = len(step_dirs)
    for j in range(start_idx, n_steps):
        prev_move = step_dirs[0] if j == 0 else step_dirs[j - 1]
        visible = _visible_priority_dirs(
            heads[j], face_dirs[j], prev_move, spins[j], j, set(heads[j + 1 :]), tname, h, w
        )
        actual_dir = _direction(heads[j], heads[j + 1])
        if actual_dir is None:
            continue
        if len(visible) >= 2:
            stats["ambiguous_steps"] += 1
            if actual_dir == face_dirs[j]:
                stats["straight_priority_wins"] += 1
            elif actual_dir == (face_dirs[j] + 1) % 4 and (face_dirs[j] + 3) % 4 in visible:
                stats["right_over_left_wins"] += 1
        if j > 0 and face_dirs[j] != step_dirs[j - 1]:
            stats["twisted_states"] += 1
    return stats


def _snapshot_stats(
    snapshot_indices: list[int],
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    spins: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
) -> dict[str, int]:
    stats = {
        "ambiguous_snapshots": 0,
        "priority_snapshots": 0,
        "straight_snapshots": 0,
        "right_snapshots": 0,
        "twisted_snapshots": 0,
    }
    for si in snapshot_indices:
        prev_move = step_dirs[0] if si == 0 else step_dirs[si - 1]
        visible = _visible_priority_dirs(
            heads[si], face_dirs[si], prev_move, spins[si], si, set(heads[si + 1 :]), tname, h, w
        )
        actual_dir = _direction(heads[si], heads[si + 1])
        if actual_dir is None or len(visible) < 2:
            continue
        stats["ambiguous_snapshots"] += 1
        if si > 0 and face_dirs[si] != step_dirs[si - 1]:
            stats["twisted_snapshots"] += 1
        if actual_dir == face_dirs[si]:
            stats["priority_snapshots"] += 1
            stats["straight_snapshots"] += 1
        elif actual_dir == (face_dirs[si] + 1) % 4 and (face_dirs[si] + 3) % 4 in visible:
            stats["priority_snapshots"] += 1
            stats["right_snapshots"] += 1
    return stats


def _snapshot_index_options(last_idx: int) -> list[list[int]]:
    options: list[list[int]] = []
    for n_frames in range(N_FRAMES_RANGE[0], min(N_FRAMES_RANGE[1], last_idx + 2)):
        raw = np.linspace(0, last_idx, n_frames)
        idxs = sorted({int(round(x)) for x in raw})
        if idxs[0] != 0:
            idxs.insert(0, 0)
        if idxs[-1] != last_idx:
            idxs.append(last_idx)
        for mid in range(1, last_idx):
            if len(idxs) >= n_frames:
                break
            if mid not in idxs:
                idxs.append(mid)
        idxs = sorted(idxs)
        if len(idxs) == n_frames:
            options.append(idxs)
    return options


def _choose_frame_plan(
    rng: np.random.Generator,
    n_steps: int,
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    spins: list[int],
    step_dirs: list[int],
    tname: str,
    h: int,
    w: int,
    branch_mode: str,
) -> tuple[list[int], dict[str, int], dict[str, int]] | None:
    total_stats = _priority_stats(heads, face_dirs, spins, step_dirs, tname, h, w, 0)
    candidates: list[tuple[tuple[int, int, int, int], list[int], dict[str, int], dict[str, int]]] = []
    for last_idx in range(1, n_steps - MIN_SUFFIX_STEPS + 1):
        prev_move = step_dirs[last_idx - 1]
        visible = _visible_priority_dirs(
            heads[last_idx],
            face_dirs[last_idx],
            prev_move,
            spins[last_idx],
            last_idx,
            set(heads[last_idx + 1 :]),
            tname,
            h,
            w,
        )
        actual_dir = _direction(heads[last_idx], heads[last_idx + 1])
        if actual_dir is None or len(visible) < 2:
            continue
        tail_stats = _priority_stats(heads, face_dirs, spins, step_dirs, tname, h, w, last_idx)
        if tail_stats["twisted_states"] == 0 or tail_stats["ambiguous_steps"] < MIN_AMBIGUOUS_TAIL_STEPS:
            continue
        if actual_dir == face_dirs[last_idx]:
            last_branch = "straight"
        elif actual_dir == (face_dirs[last_idx] + 1) % 4 and (face_dirs[last_idx] + 3) % 4 in visible:
            last_branch = "right"
        else:
            continue
        if last_branch != branch_mode:
            continue
        for snapshot_indices in _snapshot_index_options(last_idx):
            snap_stats = _snapshot_stats(
                snapshot_indices, heads, face_dirs, spins, step_dirs, tname, h, w
            )
            if snap_stats[f"{branch_mode}_snapshots"] == 0:
                continue
            score = (
                snap_stats[f"{branch_mode}_snapshots"],
                snap_stats["ambiguous_snapshots"],
                snap_stats["twisted_snapshots"],
                last_idx,
            )
            candidates.append((score, snapshot_indices, tail_stats, snap_stats))

    if not candidates:
        return None

    best_score = max(score for score, _, _, _ in candidates)
    best = [
        (snapshot_indices, tail_stats, snap_stats)
        for score, snapshot_indices, tail_stats, snap_stats in candidates
        if score == best_score
    ]
    snapshot_indices, tail_stats, snap_stats = best[int(rng.integers(0, len(best)))]
    total_stats = dict(total_stats)
    total_stats["visible_ambiguous_snapshots"] = snap_stats["ambiguous_snapshots"]
    total_stats["visible_priority_snapshots"] = snap_stats["priority_snapshots"]
    total_stats["visible_straight_snapshots"] = snap_stats["straight_snapshots"]
    total_stats["visible_right_snapshots"] = snap_stats["right_snapshots"]
    return snapshot_indices, total_stats, tail_stats


def _build_case(rng: np.random.Generator) -> dict | None:
    branch_mode = str(rng.choice(BRANCH_MODES, p=BRANCH_WEIGHTS))
    step_dirs, step_dists, snapshot_indices = _manual_branch_path(rng, branch_mode)
    n_steps = len(step_dirs)
    h, w, bg, actor, divider, tname = _sample_visual_params(rng, n_steps)

    if not _sequence_respects_priority(step_dirs):
        return None

    heads = _heads_from_steps(step_dirs, step_dists)
    face_dirs, spins = _derive_facings(step_dirs)
    placements = [(heads[i], face_dirs[i]) for i in range(n_steps + 1)]
    tail_len = n_steps - snapshot_indices[-1]

    ok, turns, dists = _validate_path(step_dirs, heads, spins, tail_len)
    if not ok:
        return None

    shift = _shift_interval(placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], face_dirs[i]) for i in range(n_steps + 1)]
    n_frames = len(snapshot_indices)

    total_stats = _priority_stats(heads, face_dirs, spins, step_dirs, tname, h, w, 0)
    tail_stats = _priority_stats(heads, face_dirs, spins, step_dirs, tname, h, w, snapshot_indices[-1])
    snap_stats = _snapshot_stats(snapshot_indices, heads, face_dirs, spins, step_dirs, tname, h, w)
    total_stats["visible_ambiguous_snapshots"] = snap_stats["ambiguous_snapshots"]
    total_stats["visible_priority_snapshots"] = snap_stats["priority_snapshots"]
    total_stats["visible_straight_snapshots"] = snap_stats["straight_snapshots"]
    total_stats["visible_right_snapshots"] = snap_stats["right_snapshots"]
    if total_stats["visible_ambiguous_snapshots"] < 1:
        return None
    if branch_mode == "straight" and total_stats["visible_straight_snapshots"] < 1:
        return None
    if branch_mode == "right" and total_stats["visible_right_snapshots"] < 1:
        return None

    frames = []
    for si in snapshot_indices:
        head, facing = placements[si]
        trail = _trail_map(heads, step_dirs, si)
        frames.append(_render_frame(h, w, bg, actor, head, facing, tname, heads[si + 1 :], trail))

    out_trail = _trail_map(heads, step_dirs, n_steps)
    fhead, ffacing = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, ffacing, tname, [], out_trail)
    inp = _assemble_input(frames, bg, divider)

    for si, frame in zip(snapshot_indices, frames):
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        if (f_tname, f_dir, f_head) != (tname, placements[si][1], placements[si][0]):
            return None
        if f_markers != set(heads[si + 1 :]):
            return None
        if si > 0 and not _extract_trail(frame):
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
            "snapshot_indices": list(snapshot_indices),
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "priority_branch_mode": branch_mode,
            "step_dirs": list(step_dirs),
            "face_dirs": list(face_dirs),
            "turn_offsets": list(spins),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": tail_len,
            "ambiguous_steps_total": total_stats["ambiguous_steps"],
            "ambiguous_steps_tail": tail_stats["ambiguous_steps"],
            "visible_ambiguous_snapshots": total_stats["visible_ambiguous_snapshots"],
            "visible_priority_snapshots": total_stats["visible_priority_snapshots"],
            "visible_straight_snapshots": total_stats["visible_straight_snapshots"],
            "visible_right_snapshots": total_stats["visible_right_snapshots"],
            "straight_priority_wins_tail": tail_stats["straight_priority_wins"],
            "right_over_left_wins_tail": tail_stats["right_over_left_wins"],
            "twisted_states_tail": tail_stats["twisted_states"],
            "trail_cells_output": len(out_trail),
            "final_offset": int(spins[-1]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
