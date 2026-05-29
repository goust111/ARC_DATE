# Variant of 5545f144
# Dimension: D1 — 匹配条件：方向对齐 (match_cond)
# Change: 精灵头部方向与标记之间的射线对齐——标记必须位于头部所朝方向的同行或同列半线上（direction_matching），... → 形状匹配消耗

import numpy as np

from arc_utils import find_components
from arc_utils import get_bg as _bg_color
from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# a) Concrete behavior and trace
# - The moving object is still one triangular sprite rendered from one of the
#   original four templates, and frames are still concatenated horizontally.
# - The change is in what counts as a consumable marker:
#   matching markers are no longer single pixels. They are disconnected
#   components in a separate marker color, each component being an exact copy of
#   the sprite's own template under some rotation.
# - The same marker color also contains distractor components such as 2x2
#   squares, 3-lines, rectangles, and plus-shapes. These are never consumed.
# - Example trace:
#   old rule: from head H, look along row/column half-lines for single pixels;
#   choose the unique reachable direction and nearest pixel.
#   new rule: extract all marker components; keep only components whose shape
#   equals the sprite template up to rotation; among those, choose the unique
#   nearest component by head-to-head Manhattan distance; jump onto that marker,
#   consume that whole component, and adopt the marker's orientation.
# - The key visible difference is that some steps move in a direction different
#   from the sprite's current facing, so the old ray rule is visibly false.
#
# b) Complexity calibration
# - I keep the original 1..4 frames, 4..12 steps, 2..7 step distances, and
#   multi-turn path structure so the task family still looks like 5545f144.
# - To avoid an over-simplified "consume everything of marker color" rule, each
#   instance includes 2..4 distractor components, with at least one distractor
#   having the same area as the true template. That forces shape matching rather
#   than color matching or pixel-count matching.
# - Marker orientations vary across a small 4-way set and repeat across a case,
#   which gives enough support for humans to notice rotation-based matching.
#
# c) Old implicit assumptions that break under the variant
# - Original `_find_agent` merged all actor-colored pixels and treated leftovers
#   as isolated marker pixels. That is invalid once markers are multi-pixel
#   components, so agent detection must work component-by-component.
# - Original `_choose_target` hard-coded same-row/same-column half-lines. That
#   assumption is exactly what D1 changes, so target choice now ignores rays and
#   instead depends on component shape matching plus nearest-distance ranking.
# - Original `main()` assumed there was exactly one non-bg non-divider color.
#   This variant needs a distinct marker color; otherwise exact-shape markers
#   would be visually and algorithmically ambiguous with the sprite itself.
#
# d) Placement legality
# - Every sprite state and every future matching marker use the same template
#   geometry, so all placements must be pairwise disjoint and 8-neighborhood
#   separated; otherwise later frames would merge components or occlude the
#   actor.
# - The generator guarantees a unique nearest future matching marker from each
#   state: for every earlier head Hj, any later head Hk must satisfy
#   dist(Hj, Hk) > dist(Hj, H(j+1)).
# - Distractors are placed after the path translation and are also kept
#   disjoint/separated from every sprite or matching-marker placement.
#
# e) Visual observability
# - Humans can see repeated copies of the sprite silhouette in the marker color
#   and can also see other small shapes of that same color that survive forever.
# - Because several steps deliberately move somewhere other than the facing ray,
#   the old "follow the arrow direction" hypothesis fails on the visible input.
#
# f) Visual cleanliness
# - All markers are disconnected components with at least a one-cell moat in the
#   8-neighborhood sense, so the grid does not collapse into large blobs.
# - Using one actor color and one marker color keeps the scene readable even
#   though multiple shape classes coexist.
#
# g) Color inference
# - There is no hidden color mapping. Actor color always means the live sprite;
#   marker color always means static candidate shapes. Shape, not color id,
#   determines whether a marker is consumable.
# ---------------------------------------------------------------------------

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Split horizontally concatenated frames separated by solid divider columns."""
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


DISTRACTOR_SHAPES: dict[str, list[tuple[int, int]]] = {
    "line3": [(0, 0), (0, 1), (0, 2)],
    "el3": [(0, 0), (1, 0), (1, 1)],
    "square4": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "zig4": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "plus5": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "rect6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
}


AREA_MATCH_DISTRACTORS = {
    3: ["line3", "el3"],
    4: ["square4", "zig4"],
    6: ["rect6"],
}


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


def _touching_point(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1


def _touching_sets(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return any(_touching_point(pa, pb) for pa in a for pb in b)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


def _component_points(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _match_component_to_template(
    coords: np.ndarray,
    allowed_tnames: list[str] | None = None,
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = _component_points(coords)
    tnames = allowed_tnames if allowed_tnames is not None else list(TEMPLATES)
    for head in pts:
        hy, hx = head
        for tname in tnames:
            for d, offs in TEMPLATES[tname].items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    return tname, d, (hy, hx), placed
    return None


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    bg = _bg_color(frame)
    comps = find_components(
        frame, bg=bg, connectivity=4, monochromatic=True, colors=[actor], return_format="dict"
    )
    if len(comps) != 1:
        return None
    match = _match_component_to_template(comps[0]["coords"])
    if match is None:
        return None
    tname, d, head, placed = match
    return tname, d, head, placed


def _identify_actor_and_marker(frame: np.ndarray, bg: int) -> tuple[int, int]:
    colors = sorted(set(map(int, frame.ravel())) - {bg})
    assert len(colors) == 2, f"expected actor and marker colors, got {colors}"

    actor = None
    for col in colors:
        comps = find_components(
            frame, bg=bg, connectivity=4, monochromatic=True, colors=[col], return_format="dict"
        )
        if len(comps) != 1:
            continue
        if _match_component_to_template(comps[0]["coords"]) is not None:
            actor = int(col)
            break

    assert actor is not None, "failed to identify actor color"
    marker = next(col for col in colors if col != actor)
    return actor, int(marker)


def _matching_marker_components(
    frame: np.ndarray, marker: int, tname: str
) -> list[dict[str, object]]:
    bg = _bg_color(frame)
    comps = find_components(
        frame, bg=bg, connectivity=4, monochromatic=True, colors=[marker], return_format="dict"
    )
    out: list[dict[str, object]] = []
    for comp in comps:
        match = _match_component_to_template(comp["coords"], [tname])
        if match is None:
            continue
        mtname, d, head, placed = match
        out.append(
            {
                "tname": mtname,
                "dir": int(d),
                "head": (int(head[0]), int(head[1])),
                "cells": set(placed),
            }
        )
    return out


def _choose_target(
    head: tuple[int, int], markers: list[dict[str, object]]
) -> dict[str, object] | None:
    if not markers:
        return None

    ranked = sorted(
        ((_dist(head, tuple(m["head"])), idx, m) for idx, m in enumerate(markers)),
        key=lambda item: (item[0], item[2]["head"][0], item[2]["head"][1], item[1]),
    )
    if len(ranked) >= 2:
        assert ranked[0][0] < ranked[1][0], (
            f"ambiguous nearest matching marker from head={head}: "
            f"{ranked[0][2]['head']} and {ranked[1][2]['head']}"
        )
    return ranked[0][2]


def _simulate_to_end(frame: np.ndarray, actor: int, marker: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent = found
    markers = _matching_marker_components(frame, marker, tname)

    for _ in range(len(markers) + 1):
        if not markers:
            break
        choice = _choose_target(head, markers)
        assert choice is not None, f"stuck with {len(markers)} matching markers"
        head = tuple(choice["head"])
        d = int(choice["dir"])
        agent = set(choice["cells"])
        markers.remove(choice)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid by consuming shape-matched markers.

    Input structure:
    - Frames are still concatenated horizontally with solid divider columns; bg is the most common color.
    - Each frame contains one actor-colored triangle sprite plus marker-colored disconnected shapes.
      Some marker components are rotated copies of the sprite template; others are distractors.

    Rule:
    - Start from the rightmost frame.
    - At each step, extract the marker-colored components and keep only the ones whose shape matches
      the sprite's own template up to rotation. Among those matches, the nearest component to the
      current head is the next target.
    - Jump the sprite onto that matched component, consume it, and adopt that component's orientation.
      Repeat until no matching components remain; output only the final sprite on a clean background.

    Key insight: The marker color contains multiple small shapes, but only rotated copies of the
    sprite silhouette are consumable, so solving the task requires component shape recognition rather
    than following the sprite's facing ray.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    start = frames[-1]
    actor, marker = _identify_actor_and_marker(start, bg)
    return _simulate_to_end(start, actor, marker)


# --------------- generator ---------------

N_FRAMES_RANGE = (1, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 7, 15

STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MIN_ORI_DIVERSITY = 3
MIN_ORI_TURNS = 2
ORI_MISMATCH_RATIO = 0.4

DISTRACTOR_COUNT_RANGE = (2, 5)  # 2..4

MAX_ATTEMPTS = 500

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 7 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 5 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor, marker = map(int, palette[:3])
    divider = int(palette[3]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, marker, divider, tname


def _make_dir_sequence(rng: np.random.Generator, n_items: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_items)))
    turn_max = min(n_items - 1, int(np.floor(TURN_RATIO_HI * n_items)))
    n_turns = int(rng.integers(turn_min, turn_max + 1))
    n_segments = n_turns + 1

    segments = [int(rng.integers(0, 4))]
    for _ in range(n_segments - 1):
        segments.append(int((segments[-1] + rng.choice([1, -1])) % 4))

    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_items - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1

    out: list[int] = []
    for d, ln in zip(segments, lengths):
        out.extend([d] * int(ln))
    return out


def _validate_move_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
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


def _validate_orientations(
    state_dirs: list[int], move_dirs: list[int]
) -> tuple[bool, int, int]:
    turns = sum(state_dirs[i] != state_dirs[i - 1] for i in range(1, len(state_dirs)))
    mismatch = sum(state_dirs[i] != move_dirs[i] for i in range(len(move_dirs)))
    if len(set(state_dirs)) < MIN_ORI_DIVERSITY:
        return False, turns, mismatch
    if turns < MIN_ORI_TURNS:
        return False, turns, mismatch
    if mismatch < max(2, int(np.ceil(ORI_MISMATCH_RATIO * len(move_dirs)))):
        return False, turns, mismatch
    return True, turns, mismatch


def _placement_compatible_with_existing(
    nxt: tuple[int, int],
    next_dir: int,
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    tname: str,
) -> bool:
    new_cells = _sprite_cells(nxt, tname, next_dir)
    for j, old_head in enumerate(heads):
        old_cells = _sprite_cells(old_head, tname, state_dirs[j])
        if new_cells & old_cells:
            return False
        if _touching_sets(new_cells, old_cells):
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    state_dirs: list[int],
    n_steps: int,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    heads: list[tuple[int, int]] = [(0, 0)]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), state_dirs[0])]
    move_dirs: list[int] = []

    for i in range(n_steps):
        cur = heads[-1]
        options = [(int(d), int(dist)) for d in rng.permutation(4) for dist in rng.permutation(np.arange(*STEP_DIST_RANGE))]
        chosen = None
        for move_dir, dist in options:
            dr, dc = DIRS4[move_dir]
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if not _placement_compatible_with_existing(nxt, state_dirs[i + 1], heads, state_dirs, tname):
                continue
            trial = fit_placements + [(nxt, state_dirs[i + 1])]
            if _shift_interval(trial, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        md = _direction(cur, chosen)
        assert md is not None
        move_dirs.append(md)
        fit_placements.append((chosen, state_dirs[i + 1]))

    return heads, move_dirs, fit_placements


def _stamp_template(
    frame: np.ndarray, head: tuple[int, int], d: int, tname: str, color: int, bg: int
) -> None:
    patch, ar, ac = _template_patch(tname, d, color, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)


def _shape_bounds(shape: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_cells(top: int, left: int, shape: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _place_distractors(
    rng: np.random.Generator,
    h: int,
    w: int,
    tname: str,
    occupied: set[tuple[int, int]],
) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    area = len(TEMPLATES[tname][0])
    primary_choices = list(AREA_MATCH_DISTRACTORS[area])
    rng.shuffle(primary_choices)
    distractor_names = [primary_choices[0]]

    others = [name for name in DISTRACTOR_SHAPES if name not in distractor_names]
    rng.shuffle(others)
    need = int(rng.integers(*DISTRACTOR_COUNT_RANGE))
    distractor_names.extend(others[: max(0, need - 1)])

    out: list[dict[str, object]] = []
    occ = set(occupied)

    for name in distractor_names:
        shape = DISTRACTOR_SHAPES[name]
        sh, sw = _shape_bounds(shape)
        candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        rng.shuffle(candidates)
        placed = None
        for top, left in candidates:
            cells = _shape_cells(top, left, shape)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if cells & occ:
                continue
            if _touching_sets(cells, occ):
                continue
            placed = {"name": name, "top": int(top), "left": int(left), "cells": cells}
            break
        if placed is None:
            return None
        out.append(placed)
        occ |= set(placed["cells"])

    return out, occ


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    marker: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    remaining_markers: list[tuple[tuple[int, int], int]],
    distractors: list[dict[str, object]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    _stamp_template(frame, head, d, tname, actor, bg)
    for mhead, md in remaining_markers:
        _stamp_template(frame, mhead, md, tname, marker, bg)
    for dd in distractors:
        for r, c in dd["cells"]:
            frame[r, c] = marker
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
    n_frames, n_steps, gap, h, w, bg, actor, marker, divider, tname = _sample_params(rng)
    state_dirs = _make_dir_sequence(rng, n_steps + 1)

    rollout = _rollout_path(rng, tname, state_dirs, n_steps, h, w)
    if rollout is None:
        return None
    heads, move_dirs, fit_placements = rollout

    ok_move, turns, dists = _validate_move_path(move_dirs, heads)
    if not ok_move:
        return None

    ok_ori, ori_turns, mismatch = _validate_orientations(state_dirs, move_dirs)
    if not ok_ori:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    occupied = set()
    for head, d in placements:
        cells = _sprite_cells(head, tname, d)
        if occupied & cells:
            return None
        if _touching_sets(cells, occupied):
            return None
        occupied |= cells

    distractor_pack = _place_distractors(rng, h, w, tname, occupied)
    if distractor_pack is None:
        return None
    distractors, _ = distractor_pack

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(
            _render_frame(
                h,
                w,
                bg,
                actor,
                marker,
                head,
                d,
                tname,
                placements[si + 1 :],
                distractors,
            )
        )

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, marker, fhead, fdir, tname, [], [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _ = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None

    found_markers = _matching_marker_components(frames[-1], marker, tname)
    expect_markers = {placements[i][0:2] for i in range(last_idx + 1, len(placements))}
    got_markers = {(tuple(m["head"]), int(m["dir"])) for m in found_markers}
    if got_markers != expect_markers:
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
            "marker": marker,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "move_turns": turns,
            "move_turn_ratio": turns / n_steps,
            "move_dirs": list(move_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "orientation_turns": ori_turns,
            "orientation_mismatch_count": mismatch,
            "distractor_names": [d["name"] for d in distractors],
            "matching_markers_in_last_frame": gap,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
