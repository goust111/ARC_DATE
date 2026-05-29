# Combo variant of 5545f144 (index 20)
# combo_spec: D4/3+D7/2+D11/1
# Name: 跳跃式移动+累积旋转+螺旋路径

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior + trace
- The visible objects stay close to the source task: horizontally concatenated
  frames, one monochrome triangular sprite, optional solid divider columns, and
  single-pixel markers in the same actor color as the sprite.
- The fusion changes three linked pieces at once.
  1. D4/3: along the unique actionable half-line, the sprite skips the nearest
     marker and lands on the second-nearest one, so each real move needs a
     visible support marker in front of the consumed target.
  2. D7/2: after a turn, the displayed facing is not just the movement
     direction; it gains one extra clockwise quarter-turn per accumulated turn.
  3. D11/1: the movement directions themselves must form a one-handed spiral,
     so every non-straight turn has the same sign.
- Small concrete trace:
  start facing Right, remaining spiral moves are Right, Down, Left, Up.
  On the first ray the visible markers might be at distances 2 and 7, so the
  head jumps to 7 and leaves the distance-2 dot behind. On the second move the
  path turns clockwise to Down, so the displayed facing becomes Left
  (Down + one extra clockwise quarter-turn). The third move turns clockwise
  again to Left, so the displayed facing becomes Right. Compared with the
  source task, the consumed target is farther away, the visible orientation
  drifts away from motion, and the future targets curl around as a spiral.

b) Complexity calibration
- A weak implementation could degenerate into "just a rectangle with one dot in
  front of each corner" or make the extra rotation invisible. To avoid that, the
  generator uses exactly four spiral sides with paired lengths such as 8,8,5,5
  or 5,5,8,8, giving a clear clockwise or counterclockwise coil.
- Each move has its own dedicated skipped support marker, so the second-nearest
  rule is always visually present instead of being an accidental corner case.
- The template family is restricted to `hollow3` and `filled4` because both are
  easy for `_find_agent` to recover and their 90-degree rotations stay legible.

c) Hidden assumption audit
- The base solver starts from the rightmost frame because the displayed triangle
  still tells you the current motion direction. That breaks here: cumulative
  twist means later frames no longer reveal the untwisted motion state. The new
  solver must start from the leftmost frame.
- The base `_choose_target` assumes "nearest marker is consumed". Here a usable
  direction must have at least two reachable markers, and the second one is the
  landing cell.
- The base reversal check uses the displayed direction. After D7/2, displayed
  facing and movement direction diverge, so the no-180-degree rule must be
  applied to the previous move direction instead.
- The base path logic only forbids reversals. D11/1 adds a stronger invariant:
  once the first 90-degree turn sign is observed, all later turns must keep the
  same sign. The solver tracks that explicitly.
- D4/3 changes the stopping condition. A finite set of support markers is meant
  to remain on the board, so the rollout stops when no direction still offers a
  valid two-marker jump.

d) Placement legality
- The generator preconstructs a four-segment spiral head path, then searches one
- skipped support marker per segment.
- Support markers are rejected if they touch any sprite placement in any state,
  touch another marker, duplicate a future head, or make some earlier state gain
  a second actionable direction.
- After supports are chosen, the whole abstract configuration is translated into
  an 8-15 by 6-15 frame only if every sprite placement and every marker fits.

e) Visual observability
- A human can see all three effects directly from the pixels:
  the active ray always contains a nearer dot plus a farther consumed target;
  the remaining targets turn with one fixed handedness; and the triangle's
  stamped orientation visibly drifts from the path after each corner.
- The generator rejects any case where either a nearest-target control solver or
  a no-extra-rotation control solver would end at the same final sprite, so the
  fused mechanics stay behaviorally active rather than cosmetic.

f) Visual cleanliness
- Markers remain isolated single pixels and are kept away from every visible
  sprite body, including later twisted orientations.
- The path uses only four sides and one support per side, which keeps the
  density low enough that the spiral tail is still readable in small ARC grids.

g) Color inference
- No hidden color ordering is introduced. Background is still the most common
  color, the actor is the shared sprite/marker color, and divider columns use a
  third solid color only when multiple frames are present.

Post-change main() assumptions
- The leftmost frame is the untwisted starting state, so its visible facing
  equals the first movement direction.
- From every nonterminal state there is exactly one actionable half-line with at
  least two reachable markers.
- All movement turns, when they occur, share one sign.
- The generator enforces each of these assumptions by explicit path
  construction, exact state validation, and round-trip solver checks.
"""

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

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

FRAME_CHOICES = (1, 2, 4)
BACKBONES = (
    (7, 7, 4, 4),
    (8, 8, 5, 5),
    (9, 9, 6, 6),
    (4, 4, 7, 7),
    (5, 5, 8, 8),
    (6, 6, 9, 9),
)
TURN_SIGNS = (-1, 1)
TEMPLATE_CHOICES = ("hollow3", "filled4")
N_JUMPS = 4
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15
MAX_ATTEMPTS = 2000


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
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


def _signed_quarter_turn(prev_dir: int, next_dir: int) -> int | None:
    delta = (next_dir - prev_dir) % 4
    if delta == 0:
        return 0
    if delta == 1:
        return 1
    if delta == 3:
        return -1
    return None


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


def _ray_candidates(
    head: tuple[int, int], markers: set[tuple[int, int]], dir_idx: int
) -> list[tuple[int, tuple[int, int]]]:
    hy, hx = head
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


def _post_move_state(
    prev_move_dir: int, spin: int, step_idx: int, move_dir: int, *, use_spin: bool
) -> tuple[int, int]:
    if not use_spin:
        return spin, int(move_dir)
    next_spin = spin
    if step_idx > 0 and move_dir != prev_move_dir:
        next_spin = (spin + 1) % 4
    return int(next_spin), int((move_dir + next_spin) % 4)


def _landing_fits(
    head: tuple[int, int], tname: str, facing: int, h: int | None, w: int | None
) -> bool:
    if h is None or w is None:
        return True
    hy, hx = head
    for oy, ox in TEMPLATES[tname][facing]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _collect_actionable(
    head: tuple[int, int],
    prev_move_dir: int,
    spin: int,
    step_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
    *,
    use_jump: bool,
    use_spin: bool,
    h: int | None = None,
    w: int | None = None,
) -> list[tuple[int, list[tuple[int, tuple[int, int]]], int, int]]:
    need = 2 if use_jump else 1
    actionable: list[tuple[int, list[tuple[int, tuple[int, int]]], int, int]] = []
    for move_dir in range(4):
        if step_idx > 0 and move_dir == (prev_move_dir + 2) % 4:
            continue
        next_spin, next_facing = _post_move_state(
            prev_move_dir, spin, step_idx, move_dir, use_spin=use_spin
        )
        cands = [
            (dist, pt)
            for dist, pt in _ray_candidates(head, markers, move_dir)
            if _landing_fits(pt, tname, next_facing, h, w)
        ]
        if len(cands) >= need:
            actionable.append((move_dir, cands, next_spin, next_facing))
    return actionable


def _choose_target(
    head: tuple[int, int],
    prev_move_dir: int,
    spin: int,
    step_idx: int,
    markers: set[tuple[int, int]],
    tname: str,
    *,
    use_jump: bool,
    use_spin: bool,
    h: int | None = None,
    w: int | None = None,
) -> tuple[int, tuple[int, int], tuple[int, int] | None, int, int] | None:
    actionable = _collect_actionable(
        head,
        prev_move_dir,
        spin,
        step_idx,
        markers,
        tname,
        use_jump=use_jump,
        use_spin=use_spin,
        h=h,
        w=w,
    )
    if not actionable:
        return None

    dirs = {move_dir for move_dir, _, _, _ in actionable}
    assert len(dirs) == 1, f"head={head} step={step_idx}: multiple actionable directions {dirs}"

    move_dir, cands, next_spin, next_facing = actionable[0]
    target_idx = 1 if use_jump else 0
    skipped = cands[0][1] if use_jump else None
    return move_dir, cands[target_idx][1], skipped, next_spin, next_facing


def _simulate_to_end(
    frame: np.ndarray,
    actor: int,
    *,
    use_jump: bool,
    use_spin: bool,
    enforce_spiral: bool,
) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, facing, head, agent, markers = found
    agent = {(int(y), int(x)) for y, x in agent}
    markers = {(int(y), int(x)) for y, x in markers}

    prev_move_dir = int(facing)
    spin = 0
    step_idx = 0
    spiral_sign: int | None = None
    max_steps = h * w * 4

    for _ in range(max_steps):
        choice = _choose_target(
            head,
            prev_move_dir,
            spin,
            step_idx,
            markers,
            tname,
            use_jump=use_jump,
            use_spin=use_spin,
            h=h,
            w=w,
        )
        if choice is None:
            break

        move_dir, new_head, _skipped, next_spin, next_facing = choice
        turn = _signed_quarter_turn(prev_move_dir, move_dir) if step_idx > 0 else 0
        if turn is None:
            raise AssertionError(f"head={head} prev_move={prev_move_dir}: illegal reversal to {move_dir}")
        if enforce_spiral and turn != 0:
            if spiral_sign is None:
                spiral_sign = int(turn)
            else:
                assert turn == spiral_sign, (
                    f"head={head} prev_move={prev_move_dir}: turn sign {turn} != {spiral_sign}"
                )

        head = (int(new_head[0]), int(new_head[1]))
        prev_move_dir = int(move_dir)
        spin = int(next_spin)
        agent = _sprite_cells(head, tname, int(next_facing))
        markers.discard(head)
        step_idx += 1

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite by skipping, twisting, and following a spiral tail.

    Input structure:
    - One, two, or four same-height frames are stitched left-to-right, with optional solid divider columns in a third color.
    - Each frame contains one triangular sprite plus remaining single-pixel markers, all in one actor color, on a most-common background.

    Rule:
    - Read the leftmost frame as the untwisted starting state.
    - From the head, exactly one cardinal half-line is actionable: it must contain at least two reachable markers, the nearest one is skipped, and the second-nearest one is the landing cell.
    - Movement directions must keep one spiral handedness, so every non-straight turn is a 90-degree turn with the same sign.
    - After each turn, add one extra clockwise quarter-turn to the accumulated display offset; stamp the landed sprite with facing = move direction + accumulated offset.
    - Repeat until no two-marker jump remains, then output only the final sprite on a clean background.

    Key insight: the leftmost frame reveals the untwisted initial motion direction, while the future markers show paired skip-target dots on a one-handed spiral and the triangle's visible orientation drifts from the path after each corner.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[0]
    return _simulate_to_end(start, actor, use_jump=True, use_spin=True, enforce_spiral=True)


def _materialize_spiral(
    backbone: tuple[int, int, int, int], start_dir: int, turn_sign: int, reverse: bool
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    seg_dirs = [int((start_dir + turn_sign * i) % 4) for i in range(len(backbone))]
    heads: list[tuple[int, int]] = [(0, 0)]
    step_dirs: list[int] = []
    dists: list[int] = []

    for seg_dir, dist in zip(seg_dirs, backbone):
        dr, dc = DIRS4[seg_dir]
        hr, hc = heads[-1]
        heads.append((hr + int(dist) * dr, hc + int(dist) * dc))
        step_dirs.append(int(seg_dir))
        dists.append(int(dist))

    if reverse:
        heads = list(reversed(heads))
        step_dirs = []
        dists = []
        for a, b in zip(heads, heads[1:]):
            d = _direction(a, b)
            assert d is not None
            step_dirs.append(int(d))
            dists.append(_dist(a, b))
    return heads, step_dirs, dists


def _state_markers(
    step_idx: int, heads: list[tuple[int, int]], supports: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    return set(heads[step_idx + 1 :]) | set(supports)


def _validate_combo_abstract(
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    face_dirs: list[int],
    spins: list[int],
    supports: list[tuple[int, int]],
    tname: str,
) -> bool:
    if len(heads) != N_JUMPS + 1 or len(step_dirs) != N_JUMPS or len(supports) != N_JUMPS:
        return False

    all_markers = list(set(heads[1:]) | set(supports))
    if len(all_markers) != 2 * N_JUMPS:
        return False
    for i in range(len(all_markers)):
        for j in range(i + 1, len(all_markers)):
            if _touching(all_markers[i], all_markers[j]):
                return False

    spiral_sign: int | None = None
    for prev_dir, next_dir in zip(step_dirs, step_dirs[1:]):
        turn = _signed_quarter_turn(prev_dir, next_dir)
        if turn is None or turn == 0:
            return False
        if spiral_sign is None:
            spiral_sign = int(turn)
        elif turn != spiral_sign:
            return False

    if spins[-1] == 0 or face_dirs[-1] == step_dirs[-1]:
        return False

    for i in range(N_JUMPS + 1):
        remaining = _state_markers(i, heads, supports)
        sprite = _sprite_cells(heads[i], tname, face_dirs[i])
        if any(_touching(m, cell) for m in remaining for cell in sprite):
            return False

        prev_move_dir = step_dirs[0] if i == 0 else step_dirs[i - 1]
        choice = _choose_target(
            heads[i],
            prev_move_dir,
            spins[i],
            i,
            remaining,
            tname,
            use_jump=True,
            use_spin=True,
        )
        if i == N_JUMPS:
            if choice is not None:
                return False
            continue

        if choice is None:
            return False
        move_dir, target, skipped, next_spin, next_facing = choice
        if move_dir != step_dirs[i] or target != heads[i + 1]:
            return False
        if skipped != supports[i]:
            return False
        if next_spin != spins[i + 1] or next_facing != face_dirs[i + 1]:
            return False

    return True


def _support_options(
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    face_dirs: list[int],
    tname: str,
) -> list[list[tuple[int, int]]]:
    all_sprite_cells = set()
    for head, face in zip(heads, face_dirs):
        all_sprite_cells |= _sprite_cells(head, tname, face)

    targets = set(heads[1:])
    options: list[list[tuple[int, int]]] = []
    for i in range(N_JUMPS):
        cur = heads[i]
        move_dir = step_dirs[i]
        dist = _dist(heads[i], heads[i + 1])
        dr, dc = DIRS4[move_dir]
        step_options: list[tuple[int, int]] = []
        for skip_dist in range(2, dist - 1):
            support = (cur[0] + skip_dist * dr, cur[1] + skip_dist * dc)
            if support in targets:
                continue
            if any(_touching(support, cell) for cell in all_sprite_cells):
                continue
            step_options.append(support)
        options.append(step_options)
    return options


def _search_supports(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    step_dirs: list[int],
    face_dirs: list[int],
    spins: list[int],
    tname: str,
) -> list[tuple[int, int]] | None:
    option_lists = _support_options(heads, step_dirs, face_dirs, tname)
    if any(not opts for opts in option_lists):
        return None

    targets = set(heads[1:])
    ordered = [list(opts) for opts in option_lists]
    for opts in ordered:
        rng.shuffle(opts)

    chosen: list[tuple[int, int]] = []

    def dfs(step_idx: int) -> bool:
        if step_idx == N_JUMPS:
            return _validate_combo_abstract(heads, step_dirs, face_dirs, spins, chosen, tname)

        for support in ordered[step_idx]:
            if support in targets or support in chosen:
                continue
            if any(_touching(support, other) for other in targets | set(chosen)):
                continue
            chosen.append(support)
            if dfs(step_idx + 1):
                return True
            chosen.pop()
        return False

    if not dfs(0):
        return None
    return list(chosen)


def _layout_and_shift(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    face_dirs: list[int],
    supports: list[tuple[int, int]],
    tname: str,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    coords: set[tuple[int, int]] = set(heads) | set(supports)
    for head, face in zip(heads, face_dirs):
        coords |= _sprite_cells(head, tname, face)

    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    req_h = max(rs) - min(rs) + 1
    req_w = max(cs) - min(cs) + 1
    if req_h > H_CAP or req_w > W_CAP:
        return None

    h_lo = max(H_FLOOR, req_h)
    w_lo = max(W_FLOOR, req_w)
    if h_lo > H_CAP or w_lo > W_CAP:
        return None

    h = int(rng.integers(h_lo, H_CAP + 1))
    w = int(rng.integers(w_lo, W_CAP + 1))

    tr_lo = -min(rs)
    tr_hi = h - 1 - max(rs)
    tc_lo = -min(cs)
    tc_hi = w - 1 - max(cs)
    if tr_lo > tr_hi or tc_lo > tc_hi:
        return None

    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    shifted_heads = [(r + tr, c + tc) for r, c in heads]
    shifted_supports = [(r + tr, c + tc) for r, c in supports]
    return h, w, shifted_heads, shifted_supports


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    face: int,
    tname: str,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, face, actor, bg)
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


def _control_equals(
    frame: np.ndarray, actor: int, out_frame: np.ndarray, *, use_jump: bool, use_spin: bool
) -> bool:
    try:
        alt = _simulate_to_end(
            frame, actor, use_jump=use_jump, use_spin=use_spin, enforce_spiral=True
        )
    except Exception:
        return False
    return np.array_equal(alt, out_frame)


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int | None, str]:
    n_frames = int(rng.choice(FRAME_CHOICES))
    gap = N_JUMPS // n_frames
    palette = rng.permutation(10)
    bg = int(palette[0])
    actor = int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None
    tname = str(rng.choice(TEMPLATE_CHOICES))
    return n_frames, bg, actor, divider, tname


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, bg, actor, divider, tname = _sample_params(rng)
    gap = N_JUMPS // n_frames

    for _ in range(80):
        backbone = tuple(int(x) for x in BACKBONES[int(rng.integers(0, len(BACKBONES)))])
        start_dir = int(rng.integers(0, 4))
        turn_sign = int(rng.choice(TURN_SIGNS))
        reverse = bool(rng.integers(0, 2))

        heads, step_dirs, dists = _materialize_spiral(backbone, start_dir, turn_sign, reverse)
        face_dirs, spins = _derive_facings(step_dirs)
        supports = _search_supports(rng, heads, step_dirs, face_dirs, spins, tname)
        if supports is None:
            continue
        if not _validate_combo_abstract(heads, step_dirs, face_dirs, spins, supports, tname):
            continue

        laid_out = _layout_and_shift(rng, heads, face_dirs, supports, tname)
        if laid_out is None:
            continue
        h, w, heads, supports = laid_out
        face_dirs, spins = _derive_facings(step_dirs)
        if not _validate_combo_abstract(heads, step_dirs, face_dirs, spins, supports, tname):
            continue

        frames: list[np.ndarray] = []
        for fi in range(n_frames):
            si = fi * gap
            markers = list(heads[si + 1 :]) + list(supports)
            frames.append(_render_frame(h, w, bg, actor, heads[si], face_dirs[si], tname, markers))

        out_frame = _render_frame(h, w, bg, actor, heads[-1], face_dirs[-1], tname, [])
        inp = _assemble_input(frames, bg, divider)

        for fi, frame in enumerate(frames):
            found = _find_agent(frame, actor)
            if found is None:
                break
            f_tname, f_dir, f_head, _agent, f_markers = found
            si = fi * gap
            if (f_tname, f_dir, f_head) != (tname, face_dirs[si], heads[si]):
                break
            if f_markers != _state_markers(si, heads, supports):
                break
        else:
            if not np.array_equal(main(inp), out_frame):
                continue
            if _control_equals(frames[0], actor, out_frame, use_jump=False, use_spin=True):
                continue
            if _control_equals(frames[0], actor, out_frame, use_jump=True, use_spin=False):
                continue
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
                    "gap": gap,
                    "step_dirs": list(step_dirs),
                    "face_dirs": list(face_dirs),
                    "turn_offsets": list(spins),
                    "heads": list(heads),
                    "supports": list(supports),
                    "jump_distances": list(dists),
                    "support_distances": [_dist(heads[i], supports[i]) for i in range(N_JUMPS)],
                    "spiral_turn_sign": turn_sign,
                    "reverse": reverse,
                    "count_in_first_frame": 2 * N_JUMPS,
                    "count_in_last_input_frame": gap + N_JUMPS,
                },
            }
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid combo input after {MAX_ATTEMPTS} attempts")
