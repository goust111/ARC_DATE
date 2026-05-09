# Variant of 5545f144
# Dimension: D4 — 路径规则：移动方向选择 (path)
# Change: 每步恰好一个基数方向的半线上有可达标记，精灵移向该半线上最近的标记；不允许180度反转，只允许前进或90度转弯。... → 跳跃式移动（隔一消一）

# Design analysis for the D4 variant
# a) Concretization and behavior trace
#    The visible objects stay the same: one monochrome triangular sprite plus
#    monochrome single-pixel markers inside each frame, and solid divider
#    columns between frames. The changed behavior is not a recoloring or a new
#    object type; it is purely about which aligned marker gets chosen next.
#    Old rule on one right-facing half-line with markers at distances 2, 5, 7:
#    the head moved to distance 2. New rule: a direction is actionable only when
#    it has at least two aligned markers, the nearest one is skipped, and the
#    head lands on distance 5 instead. The distance-2 marker remains for later.
#    This affects both solver target selection and generator path construction.
# b) Complexity calibration
#    If we naively kept only one marker per move, the variant would visually
#    collapse back to the original task. Therefore every actual jump is
#    generated from a state with at least two visible aligned markers on the
#    active half-line. Other directions may contain at most isolated singletons,
#    which are inert because they do not offer a "skip one, take one" choice.
# c) Hidden assumption audit
#    The original `_choose_target` assumed "nearest marker is the consumed one".
#    The original generator also forbade any earlier marker from being nearer
#    than the intended next one. That assumption is invalid here: a nearer
#    marker is now required. A second hidden issue is logical, not cosmetic:
#    under a strict "always consume the second-nearest marker" rule, a finite
#    marker set can never be fully exhausted, because the last remaining marker
#    can never be second-nearest. The adapted termination rule is therefore:
#    stop when no direction still has two reachable markers. Generator samples
#    are built with one dedicated skipped support marker per jump; those support
#    dots remain on the board, but from the final head they never form a usable
#    two-marker jump.
# d) Placement legality
#    The generator keeps all marker cells pairwise non-touching, keeps sprite
#    placements in-bounds after a final global translation, and validates every
#    simulated state against the variant solver logic. This avoids overlaps,
#    clipped templates, and ambiguous frames that `_find_agent` could misread.
# e) Visual observability
#    The human-visible clue is the paired spacing on the active line: the sprite
#    must visibly jump past a nearer dot to a farther one. The generator favors
#    varied jump lengths and multiple turns so the final sprite position differs
#    from the original nearest-target behavior on the same input.
# f) Visual cleanliness
#    Markers remain isolated single pixels, and they are kept away from sprite
#    cells. No extra colors or overlays are introduced, so the active line and
#    skipped dot remain legible in small ARC grids.
# g) Color inference
#    No color mapping is added. Background, actor, and optional divider colors
#    keep their original roles, so the new rule is inferable from geometry
#    alone rather than hidden color ids.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


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


def _landing_fits(
    head: tuple[int, int], tname: str, dir_idx: int, h: int, w: int
) -> bool:
    hy, hx = head
    for oy, ox in TEMPLATES[tname][dir_idx]:
        ny = hy + oy
        nx = hx + ox
        if not (0 <= ny < h and 0 <= nx < w):
            return False
    return True


def _collect_actionable(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    *,
    tname: str | None = None,
    h: int | None = None,
    w: int | None = None,
) -> list[tuple[int, list[tuple[int, tuple[int, int]]]]]:
    actionable: list[tuple[int, list[tuple[int, tuple[int, int]]]]] = []
    for dd in range(4):
        cands = _ray_candidates(head, markers, dd)
        if tname is not None:
            assert h is not None and w is not None
            cands = [(dist, pt) for dist, pt in cands if _landing_fits(pt, tname, dd, h, w)]
        if len(cands) >= 2:
            actionable.append((dd, cands))
    return actionable


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    actionable = _collect_actionable(head, markers, tname=tname, h=h, w=w)
    if not actionable:
        return None

    dirs_with_choices = {dd for dd, _ in actionable}
    assert len(dirs_with_choices) == 1, (
        f"head={head} dir={d}: multiple jump directions {dirs_with_choices}"
    )

    dd, cands = actionable[0]
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )
    return dd, cands[1][1]


def _choose_target_abstract(
    head: tuple[int, int], markers: set[tuple[int, int]]
) -> tuple[int, tuple[int, int], tuple[int, int], int] | None:
    actionable = _collect_actionable(head, markers)
    if len(actionable) != 1:
        return None
    dd, cands = actionable[0]
    return dd, cands[1][1], cands[0][1], cands[1][0]


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
        choice = _choose_target(head, d, markers, tname, h, w)
        if choice is None:
            break

        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite by jumping past the nearest aligned marker.

    Input structure:
    - Several same-height frames may be stitched left-to-right, with solid divider columns between them.
    - Each frame contains one triangular sprite plus single-pixel markers in the same actor color.

    Rule:
    - Read the rightmost frame as the latest visible state.
    - From the sprite head, inspect the four cardinal half-lines. A direction is usable only if it has at least two reachable markers on that half-line.
    - Exactly one usable direction exists at each jump: ignore the nearest marker, land on the second-nearest one, consume only that landed-on marker, and re-face the travel direction.
    - Repeat until no direction still offers a two-marker jump; then output only the final sprite on a clean background.

    Key insight: the triangle tip reveals the current facing, but the move is determined by counting aligned markers and taking the second one rather than the first.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator ---------------

FRAME_CHOICES = (1, 3)
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15
JUMPS_RANGE = (3, 4)             # fixed at 3 jumps, plus one skipped support marker per jump
JUMP_DIST_RANGE = (3, 8)         # consumed target distance 3..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2
MAX_ATTEMPTS = 400
ABSTRACT_SPAN_CAP = 11
SEARCH_BRANCH_LIMIT = 18

_JUMP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*JUMPS_RANGE)}


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


def _point_span(points: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int | None, str]:
    n_frames = int(rng.choice(FRAME_CHOICES))
    n_jumps = 3
    gap = n_jumps // n_frames

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tname = str(rng.choice(["hollow3", "filled4"]))
    return n_frames, n_jumps, gap, bg, actor, divider, tname


def _state_markers(
    i: int, heads: list[tuple[int, int]], supports: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    return set(heads[i + 1:]) | set(supports)


def _make_step_dirs(rng: np.random.Generator, n_jumps: int) -> list[int]:
    assert n_jumps == 3
    start = int(rng.integers(0, 4))
    sign = int(rng.choice([-1, 1]))
    return [start, (start + sign) % 4, (start + 2 * sign) % 4]


def _marker_ok_for_past(
    marker: tuple[int, int],
    existing_markers: set[tuple[int, int]],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    target_dists: list[int],
    tname: str,
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(marker, cell) for cell in sprite):
            return False
        if j >= len(step_dirs):
            continue
        rel_dir = _direction(old_head, marker)
        if rel_dir is None:
            continue
        if rel_dir == step_dirs[j]:
            if _dist(old_head, marker) <= target_dists[j]:
                return False
            continue
        if sum(_direction(old_head, m) == rel_dir for m in existing_markers) >= 1:
            return False
    return True


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int]
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]] | None:
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [step_dirs[0]]
    supports: list[tuple[int, int]] = []
    target_dists: list[int] = []
    markers: set[tuple[int, int]] = set()

    def options(step_idx: int) -> list[tuple[int, int, tuple[int, int], tuple[int, int]]]:
        cur = heads[-1]
        next_dir = step_dirs[step_idx]
        cur_face = state_dirs[-1]
        if step_idx > 0 and next_dir == (cur_face + 2) % 4:
            return []

        dr, dc = DIRS4[next_dir]
        sprite_now = _sprite_cells(cur, tname, state_dirs[step_idx])
        out: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
        for target_dist in rng.permutation(np.arange(max(4, JUMP_DIST_RANGE[0]), JUMP_DIST_RANGE[1])):
            for support_dist in rng.permutation(np.arange(2, target_dist - 1)):
                support = (cur[0] + int(support_dist) * dr, cur[1] + int(support_dist) * dc)
                target = (cur[0] + int(target_dist) * dr, cur[1] + int(target_dist) * dc)
                if support in markers or target in markers or support == target:
                    continue
                if _touching(support, target):
                    continue
                if any(_touching(support, m) or _touching(target, m) for m in markers):
                    continue
                if any(_touching(support, cell) or _touching(target, cell) for cell in sprite_now):
                    continue
                if not _marker_ok_for_past(
                    support, markers, heads, state_dirs, step_dirs[:step_idx], target_dists, tname
                ):
                    continue
                if not _marker_ok_for_past(
                    target, markers | {support}, heads, state_dirs, step_dirs[:step_idx], target_dists, tname
                ):
                    continue
                new_head = target
                new_supports = supports + [support]
                new_sprite = _sprite_cells(new_head, tname, next_dir)
                if any(_touching(m, cell) for m in new_supports for cell in new_sprite):
                    continue
                if len(_collect_actionable(new_head, set(new_supports))) > 0:
                    continue
                span_h, span_w = _point_span(set(markers) | {support, target, (0, 0)})
                if span_h > ABSTRACT_SPAN_CAP or span_w > ABSTRACT_SPAN_CAP:
                    continue
                turn_cost = 0 if step_idx == 0 or next_dir != step_dirs[step_idx - 1] else 1
                span_cost = span_h + span_w
                out.append((turn_cost, span_cost, support, target))
        rng.shuffle(out)
        out.sort(key=lambda item: (item[0], item[1]))
        return out[:SEARCH_BRANCH_LIMIT]

    def dfs(step_idx: int) -> bool:
        if step_idx == len(step_dirs):
            return True
        for _, _, support, target in options(step_idx):
            next_dir = step_dirs[step_idx]
            supports.append(support)
            target_dists.append(_dist(heads[-1], target))
            markers.add(support)
            markers.add(target)
            heads.append(target)
            state_dirs.append(next_dir)
            if dfs(step_idx + 1):
                return True
            state_dirs.pop()
            heads.pop()
            markers.remove(target)
            markers.remove(support)
            target_dists.pop()
            supports.pop()
        return False

    if not dfs(0):
        return None
    return heads, state_dirs, supports


def _state_dirs_from_heads(heads: list[tuple[int, int]]) -> list[int]:
    step_dirs = [_direction(heads[i], heads[i + 1]) for i in range(len(heads) - 1)]
    assert all(d is not None for d in step_dirs)
    dirs = [int(step_dirs[0])]
    for d in step_dirs:
        dirs.append(int(d))
    return dirs


def _validate_path(
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    supports: list[tuple[int, int]],
    tname: str,
) -> tuple[bool, int, list[int], list[int]]:
    n_jumps = len(heads) - 1
    step_dirs = [int(_direction(heads[i], heads[i + 1])) for i in range(n_jumps)]
    if len(set(step_dirs)) < MIN_DIR_DIVERSITY:
        return False, 0, [], []
    if not (set(step_dirs) & {0, 2} and set(step_dirs) & {1, 3}):
        return False, 0, [], []

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_jumps))
    if turns < MIN_TURNS_ACCEPT:
        return False, turns, [], step_dirs
    if not (TURN_RATIO_LO <= turns / n_jumps <= TURN_RATIO_HI):
        return False, turns, [], step_dirs

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_jumps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists, step_dirs

    all_markers = set(heads[1:]) | set(supports)
    if len(all_markers) != len(heads) - 1 + len(supports):
        return False, turns, dists, step_dirs
    pts = list(all_markers)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _touching(pts[i], pts[j]):
                return False, turns, dists, step_dirs

    for i, head in enumerate(heads):
        remaining = _state_markers(i, heads, supports)
        sprite = _sprite_cells(head, tname, state_dirs[i])
        if any(_touching(m, cell) for m in remaining for cell in sprite):
            return False, turns, dists, step_dirs

        choice = _choose_target_abstract(head, remaining)
        if i == n_jumps:
            if choice is not None:
                return False, turns, dists, step_dirs
            continue

        if choice is None:
            return False, turns, dists, step_dirs
        exp_dir = step_dirs[i]
        exp_target = heads[i + 1]
        if choice[0] != exp_dir or choice[1] != exp_target:
            return False, turns, dists, step_dirs
        if i > 0 and exp_dir == (state_dirs[i] + 2) % 4:
            return False, turns, dists, step_dirs

    return True, turns, dists, step_dirs


def _layout_and_shift(
    rng: np.random.Generator,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    supports: list[tuple[int, int]],
    tname: str,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    coords: set[tuple[int, int]] = set(supports)
    for head, d in zip(heads, state_dirs):
        coords |= _sprite_cells(head, tname, d)
    coords |= set(heads)

    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    req_h = max(rs) - min(rs) + 1
    req_w = max(cs) - min(cs) + 1
    if req_h > H_CAP or req_w > W_CAP:
        return None

    h_lo = max(H_FLOOR, req_h)
    w_lo = max(W_FLOOR, req_w)
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
    n_frames, n_jumps, gap, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_jumps)
    rollout = _rollout_path(rng, tname, step_dirs)
    if rollout is None:
        return None
    heads, state_dirs, supports = rollout

    ok, turns, dists, step_dirs = _validate_path(heads, state_dirs, supports, tname)
    if not ok:
        return None

    laid_out = _layout_and_shift(rng, heads, state_dirs, supports, tname)
    if laid_out is None:
        return None
    h, w, heads, supports = laid_out
    state_dirs = _state_dirs_from_heads(heads)

    ok, turns, dists, step_dirs = _validate_path(heads, state_dirs, supports, tname)
    if not ok:
        return None

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        markers = list(heads[si + 1:]) + list(supports)
        frames.append(_render_frame(h, w, bg, actor, heads[si], state_dirs[si], tname, markers))

    out_frame = _render_frame(h, w, bg, actor, heads[-1], state_dirs[-1], tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, state_dirs[last_idx], heads[last_idx]):
        return None
    if f_markers != (set(heads[last_idx + 1:]) | set(supports)):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    skip_dists = []
    actionable_counts = []
    for i in range(n_jumps):
        remaining = _state_markers(i, heads, supports)
        choice = _choose_target_abstract(heads[i], remaining)
        if choice is None:
            return None
        actionable = _collect_actionable(heads[i], remaining)
        actionable_counts.append(len(actionable[0][1]))
        skip_dists.append(_dist(heads[i], choice[2]))

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
            "total_jumps": n_jumps,
            "initial_markers": n_jumps + len(supports),
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_jumps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "support_markers": list(supports),
            "jump_distances": dists,
            "skipped_marker_distances": skip_dists,
            "active_line_marker_counts": actionable_counts,
            "count_in_first_frame": n_jumps + len(supports),
            "count_in_last_input_frame": gap + len(supports),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
