# Variant of 5545f144
# Dimension: D3 — 目标对象：标记类型 (dst)
# Change: 散布的单像素标记点，与精灵共用同一颜色（actor色），被消耗后消失。... → 标记带有方向指示

import numpy as np

from arc_utils import components_mask
from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


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

# Marker templates are anchored at the landing cell. They are intentionally
# distinct from every actor template so color-role detection stays unambiguous.
MARKER_TEMPLATES: dict[int, list[tuple[int, int]]] = {
    0: [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)],
    1: [(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)],
    2: [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    3: [(-1, 1), (0, -1), (0, 0), (0, 1), (1, 1)],
}


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concreteization and behavior trace
# - Old marker = one actor-colored pixel at each future head. New marker = a
#   small 5-pixel arrow in a dedicated marker color; its center cell is
#   still the landing point, while the protruding arm encodes the direction to
#   face after consumption.
# - Input/output structure stays the same: horizontally concatenated snapshots as
#   input, single clean frame as output. The changed part is inside each frame:
#   the sprite keeps its original triangle templates, but the future targets are
#   no longer dots.
# - Small trace: if the current sprite faces right, the nearest marker center on
#   that rightward half-line is consumed first. In the old task the next facing
#   was inferred from geometric uniqueness among remaining dots. In this variant,
#   the consumed marker itself decides the next facing. That means future markers
#   may exist in multiple directions around the new head; only the arrow on the
#   consumed marker tells the solver which half-line matters next.
#
# b) Complexity calibration
# - The marker glyph uses only four orientations and a single marker color. That
#   is enough to make the rule visible without exploding the combinatorics.
# - We keep the original sprite templates and multi-frame cadence, so the task is
#   still recognizably the same family. The only new object type is the arrow
#   marker.
# - Final orientation is also driven by the last consumed marker, so the variant
#   can differ from the original even when the path geometry alone is similar.
#
# c) Implicit-assumption audit for the old main()
# - Old actor-color detection assumed exactly one non-bg, non-divider color in a
#   frame. That no longer holds because marker arrows have their own color.
# - Old _find_agent() treated leftover actor-colored pixels as single-pixel
#   markers. That breaks once markers are multi-pixel glyphs, so the variant must
#   separate actor-color parsing from marker-color parsing.
# - Old target choice searched all four half-lines and relied on “exactly one
#   reachable direction”. That assumption is intentionally dropped; now the sprite
#   follows its current facing only, and the consumed marker sets the next facing.
#
# d) Placement legality
# - Every sprite placement and every marker glyph must fit fully in-grid after a
#   common global translation.
# - Each marker glyph must stay disjoint from every sprite placement that occurs
#   before it is consumed, and markers must not touch one another in 8-neighborhood.
# - For any past head, a later marker on that same facing half-line may only lie
#   strictly farther than the intended next marker, otherwise the solver would
#   consume the wrong target first.
#
# e) Visual observability
# - The rule is human-visible: markers look like tiny arrows, and the sprite’s
#   final orientation can be set by the last arrow even when no further target
#   exists. This makes the “instructional marker” effect observable in only a few
#   examples.
# - No hidden ordering by color id is used. Spatial direction and marker shape are
#   the only control signals.
#
# f) Visual cleanliness
# - Markers use a dedicated color and a compact 5-cell glyph, so they are easy to
#   distinguish from the actor. Generator constraints keep glyphs from touching
#   each other or coexisting sprite placements.
#
# g) Color-change inferability
# - Only one extra color is added, and it always means “directional marker”.
#   There is no per-color mapping to learn.
# ---------------------------------------------------------------------------


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


def _template_bounds(offs: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _marker_cells(anchor: tuple[int, int], d: int) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in MARKER_TEMPLATES[d]}


def _fits_cells(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _find_agent_exact(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (size, template_name, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    score = len(placed)
                    if best is None or score > best[0]:
                        best = (score, tname, d, (hy, hx), placed)
    if best is None:
        return None
    _, tname, d, head, placed = best
    return tname, d, head, set(map(tuple, placed))


def _match_marker_component(coords: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 5:
        return None
    for anchor in pts:
        ay, ax = anchor
        for d, offs in MARKER_TEMPLATES.items():
            placed = {(ay + dy, ax + dx) for dy, dx in offs}
            if placed == pts:
                return d, (ay, ax)
    return None


def _find_markers(frame: np.ndarray, marker_color: int) -> dict[tuple[int, int], int]:
    mask = frame == marker_color
    comps = components_mask(mask, connectivity=4)
    markers: dict[tuple[int, int], int] = {}
    for coords in comps:
        found = _match_marker_component(coords)
        assert found is not None, "marker component did not match any directional glyph"
        d, anchor = found
        assert anchor not in markers, "duplicate marker anchor detected"
        markers[anchor] = d
    return markers


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: dict[tuple[int, int], int],
) -> tuple[tuple[int, int], int] | None:
    hy, hx = head
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for (y, x), nd in markers.items():
        if d == 0 and x == hx and y < hy:
            candidates.append((hy - y, (y, x), nd))
        elif d == 1 and y == hy and x > hx:
            candidates.append((x - hx, (y, x), nd))
        elif d == 2 and x == hx and y > hy:
            candidates.append((y - hy, (y, x), nd))
        elif d == 3 and y == hy and x < hx:
            candidates.append((hx - x, (y, x), nd))
    if not candidates:
        return None
    candidates.sort()
    _, anchor, nd = candidates[0]
    return anchor, nd


def _simulate_to_end(frame: np.ndarray, actor: int, marker_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent_exact(frame, actor)
    assert found is not None, "failed to detect actor sprite"
    tname, d, head, agent = found
    markers = _find_markers(frame, marker_color)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        choice = _choose_target(head, d, markers)
        assert choice is not None, f"stuck with {len(markers)} markers at head={head} dir={d}"
        head, d = choice
        agent = _sprite_cells(head, tname, d)
        assert _fits_cells(agent, h, w), f"sprite would leave grid at head={head} dir={d}"
        del markers[head]

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite through the snapshot sequence by following directional markers.

    Input structure:
    - Frames are concatenated left-to-right with solid divider columns; the background is the most frequent color.
    - Each frame contains one actor-colored triangular sprite plus marker-colored mini-arrows whose center cells are future landing points.

    Rule:
    - Start from the rightmost frame and read the sprite's facing from its triangular tip.
    - Repeatedly move the head to the nearest marker center on that facing half-line, consume the whole marker glyph, and then rotate the sprite to the consumed marker's arrow direction.
    - Continue until every marker is consumed; output only the final sprite on a clean background.

    Key insight: The sprite's current facing chooses which half-line to search, while each consumed marker dictates the next facing, so the arrows act as turn instructions rather than passive dots.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    start = frames[-1]

    colors = sorted(set(map(int, start.ravel())) - {bg})
    actor = None
    for color in colors:
        if _find_agent_exact(start, color) is not None:
            actor = int(color)
            break
    assert actor is not None, "failed to identify actor color"

    marker_colors = [int(c) for c in colors if int(c) != actor]
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker_color = marker_colors[0]
    return _simulate_to_end(start, actor, marker_color)


# --------------- generator (directional-marker variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 7, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 11)            # 4..10 inclusive
STEP_DIST_RANGE = (2, 8)         # 2..7 inclusive
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 8
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 1000

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _sprite_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    return _template_bounds(TEMPLATES[tname][d])


def _marker_bounds(d: int) -> tuple[int, int, int, int]:
    return _template_bounds(MARKER_TEMPLATES[d])


def _template_patch(tname: str, d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _sprite_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = color
    return patch, -r0, -c0


def _marker_patch(d: int, color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _marker_bounds(d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in MARKER_TEMPLATES[d]:
        patch[dr - r0, dc - c0] = color
    return patch, -r0, -c0


def _shift_interval(
    sprite_placements: list[tuple[tuple[int, int], int]],
    marker_placements: list[tuple[tuple[int, int], int]],
    h: int,
    w: int,
    tname: str,
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9

    for (hr, hc), d in sprite_placements:
        r0, r1, c0, c1 = _sprite_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None

    for (mr, mc), d in marker_placements:
        r0, r1, c0, c1 = _marker_bounds(d)
        tr_lo = max(tr_lo, -mr - r0)
        tr_hi = min(tr_hi, h - 1 - mr - r1)
        tc_lo = max(tc_lo, -mc - c0)
        tc_hi = min(tc_hi, w - 1 - mc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None

    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 5 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, marker = int(palette[0]), int(palette[1]), int(palette[2])
    divider = int(palette[3]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, marker, divider, tname


def _make_move_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
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


def _sample_final_dir(rng: np.random.Generator, last_move_dir: int) -> int:
    opts = np.array([last_move_dir, (last_move_dir + 1) % 4, (last_move_dir - 1) % 4], dtype=int)
    probs = np.array([0.25, 0.375, 0.375], dtype=float)
    return int(rng.choice(opts, p=probs))


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    next_dir: int,
    heads: list[tuple[int, int]],
    move_dirs: list[int],
    sprite_placements: list[tuple[tuple[int, int], int]],
    marker_placements: list[tuple[tuple[int, int], int]],
    tname: str,
) -> bool:
    nxt_cells = _marker_cells(nxt, next_dir)

    for head, d in sprite_placements:
        sprite = _sprite_cells(head, tname, d)
        if any(_touching(a, b) for a in nxt_cells for b in sprite):
            return False

    for anchor, d in marker_placements:
        other = _marker_cells(anchor, d)
        if any(_touching(a, b) for a in nxt_cells for b in other):
            return False

    for j, old_head in enumerate(heads[:-1]):
        rel_dir = _direction(old_head, nxt)
        if rel_dir != move_dirs[j]:
            continue
        if _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    move_dirs: list[int],
    state_dirs: list[int],
    h: int,
    w: int,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[tuple[int, int], int]],
    list[tuple[tuple[int, int], int]],
] | None:
    n_steps = len(move_dirs)
    heads: list[tuple[int, int]] = [(0, 0)]
    sprite_placements: list[tuple[tuple[int, int], int]] = [((0, 0), state_dirs[0])]
    marker_placements: list[tuple[tuple[int, int], int]] = []

    for i in range(n_steps):
        cur = heads[-1]
        move_dir = move_dirs[i]
        next_dir = state_dirs[i + 1]
        dr, dc = DIRS4[move_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None

        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if not _marker_compatible_with_existing(
                nxt,
                next_dir,
                heads,
                move_dirs,
                sprite_placements,
                marker_placements,
                tname,
            ):
                continue

            trial_sprites = sprite_placements + [(nxt, next_dir)]
            trial_markers = marker_placements + [(nxt, next_dir)]
            if _shift_interval(trial_sprites, trial_markers, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        sprite_placements.append((chosen, next_dir))
        marker_placements.append((chosen, next_dir))

    return heads, sprite_placements, marker_placements


def _validate_path(
    move_dirs: list[int],
    state_dirs: list[int],
    heads: list[tuple[int, int]],
) -> tuple[bool, int, list[int]]:
    n_steps = len(move_dirs)
    dir_set = set(move_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, []

    turns = sum(move_dirs[i] != move_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists

    if state_dirs[-1] == move_dirs[-1] and turns == MIN_TURNS_ACCEPT:
        return False, turns, dists
    return True, turns, dists


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    marker_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for anchor, md in markers:
        mp, mr, mc = _marker_patch(md, marker_color, bg)
        paste(frame, mp, anchor[0] - mr, anchor[1] - mc, transparent=bg)
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


def _branch_stats(heads: list[tuple[int, int]], state_dirs: list[int], markers: list[tuple[tuple[int, int], int]]) -> int:
    branching = 0
    for i, head in enumerate(heads[:-1]):
        facing = state_dirs[i]
        hy, hx = head
        dirs = set()
        for (y, x), _ in markers[i:]:
            rel = _direction((hy, hx), (y, x))
            if rel is not None:
                dirs.add(rel)
        if len(dirs) >= 2 and any(
            (facing == 0 and x == hx and y < hy)
            or (facing == 1 and y == hy and x > hx)
            or (facing == 2 and x == hx and y > hy)
            or (facing == 3 and y == hy and x < hx)
            for (y, x), _ in markers[i:]
        ):
            branching += 1
    return branching


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, marker_color, divider, tname = _sample_params(rng)
    move_dirs = _make_move_dirs(rng, n_steps)
    final_dir = _sample_final_dir(rng, move_dirs[-1])
    state_dirs = move_dirs + [final_dir]

    rollout = _rollout_path(rng, tname, move_dirs, state_dirs, h, w)
    if rollout is None:
        return None
    heads, sprite_placements, marker_placements = rollout

    ok, turns, dists = _validate_path(move_dirs, state_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(sprite_placements, marker_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))

    heads = [(r + tr, c + tc) for r, c in heads]
    sprite_placements = [((r + tr, c + tc), d) for (r, c), d in sprite_placements]
    marker_placements = [((r + tr, c + tc), d) for (r, c), d in marker_placements]

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = sprite_placements[si]
        frames.append(
            _render_frame(h, w, bg, actor, marker_color, head, d, tname, marker_placements[si:])
        )

    fhead, fdir = sprite_placements[-1]
    out_frame = _render_frame(h, w, bg, actor, marker_color, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    found_actor = _find_agent_exact(frames[-1], actor)
    if found_actor is None:
        return None
    f_tname, f_dir, f_head, _f_agent = found_actor
    last_idx = (n_frames - 1) * gap
    if (f_tname, f_dir, f_head) != (tname, sprite_placements[last_idx][1], sprite_placements[last_idx][0]):
        return None

    found_markers = _find_markers(frames[-1], marker_color)
    expect_markers = {anchor: d for anchor, d in marker_placements[last_idx:]}
    if found_markers != expect_markers:
        return None

    if not np.array_equal(main(inp), out_frame):
        return None

    branch_count = _branch_stats(heads, state_dirs, marker_placements)
    if branch_count == 0 and state_dirs[-1] == move_dirs[-1]:
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
            "marker_color": marker_color,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(move_dirs)),
            "move_dirs": list(move_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "branching_heads": branch_count,
            "final_dir_differs_from_last_move": state_dirs[-1] != move_dirs[-1],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
