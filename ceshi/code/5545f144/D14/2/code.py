# Variant of 5545f144
# Dimension: D14 — 唯一方向约束 (match_cond)
# Change: 每步恰好一个基数方向半线上有可达标记（CONSTRAINT 1），确保移动方向无歧义。... → 标记数量消歧

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the D14 variant:
# a) Concrete behavior and trace
#    - The pixels are unchanged at the object level: each frame still contains one
#      monochrome triangle sprite plus same-color single-pixel markers, with
#      optional solid divider columns between frames.
#    - The changed behavior lives inside target selection. In the original task,
#      a state like head=(5,5) with markers at (5,7), (5,10), (3,5) was illegal
#      because both Right and Up were populated. In this variant the state is
#      legal: Right has 2 markers, Up has 1, so the sprite moves right to the
#      nearest right-side marker at (5,7). A tie such as Right=2, Up=2 would
#      still be ambiguous, so the generator avoids ties.
#    - This changes only one decision point in the solver, but it also changes
#      the generator: the old generator prohibited any future marker on a second
#      half-line from a past head; the new one must intentionally allow that and
#      guarantee a unique largest count instead.
# b) Complexity calibration
#    - To make the new rule visible, generated paths are built on a small lattice
#      of reused rows and columns, so earlier heads often see future markers on
#      multiple directions. That creates repeated count comparisons instead of
#      rare one-off accidents.
#    - The generator still keeps 4-12 total moves, >=3 directions used, varied
#      step lengths, and no 180-degree reversals, so the task remains close to
#      the original in overall complexity.
# c) Hidden assumptions in main()
#    - _split_frames assumes divider columns are solid and non-background.
#      Generator still constructs frames exactly that way.
#    - _find_agent assumes one template placement explains the sprite pixels
#      better than any alternative. Generator oracle-checks the last frame and
#      rejects cases where marker layout would confuse recovery.
#    - _choose_target used to rely on "exactly one direction present". That is
#      the core obsolete assumption. It is replaced by "the maximum reachable
#      marker count over directions is unique".
#    - The no-reversal assertion remains valid. Generator still forbids opposite
#      turns, so solver behavior stays aligned with the original task outside D14.
# d) Placement legality
#    - Head positions are sampled inside a 2-cell safety margin. Every template
#      orientation then fits everywhere the sprite lands, so count comparison is
#      not polluted by border clipping.
#    - All heads are distinct and axis-aligned with distances 2-7, so markers
#      remain single pixels and path snapshots are renderable.
# e) Visual observability
#    - The rightmost input frame is forced to contain at least two populated
#      directions from the visible head, with a unique count winner. This makes
#      the new rule readable directly from the frame that determines the output.
# f) Visual cleanliness
#    - The lattice spacing prevents marker piles and keeps dots separated.
#      The sprite and markers still share one color as in the original task, but
#      solver-oracle checks reject cluttered last frames that break detection.
# g) Color reasoning
#    - No new color logic is introduced. Colors remain purely structural:
#      background, actor, and optional divider.
# h) Generator/solver contract added by this variant
#    - Every simulated step must have at least one reachable marker.
#    - The winning direction by count must be unique.
#    - Within the winning direction, the nearest marker must be the next path
#      head, because the solver still advances to the nearest marker after the
#      count comparison chooses a direction.

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


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ay, ax = a
    by, bx = b
    if ax == bx and by < ay:
        return 0
    if ay == by and bx > ax:
        return 1
    if ax == bx and by > ay:
        return 2
    if ay == by and bx < ax:
        return 3
    return None


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
    return {(hy + dy, hx + dx) for dy, dx in TEMPLATES[tname][d]}


def _aligned_markers(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    out: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for pt in markers:
        dd = _direction(head, pt)
        if dd is None:
            continue
        out.setdefault(dd, []).append((_dist(head, pt), pt))
    for pts in out.values():
        pts.sort()
    return out


def _reachable_by_dir(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
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

    reachable: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for dd, pts in _aligned_markers(head, markers).items():
        kept = [(dist, pt) for dist, pt in pts if fits(pt)]
        if kept:
            reachable[dd] = kept
    return reachable


def _pick_target_from_reachable(
    head: tuple[int, int],
    d: int,
    reachable: dict[int, list[tuple[int, tuple[int, int]]]],
    *,
    strict: bool,
) -> tuple[int, tuple[int, int], dict[int, int]] | None:
    if not reachable:
        return None

    counts = {dd: len(pts) for dd, pts in reachable.items()}
    best_count = max(counts.values())
    winners = [dd for dd, cnt in counts.items() if cnt == best_count]
    if len(winners) != 1:
        if strict:
            raise AssertionError(
                f"head={head} dir={d}: tied direction counts {counts}, expected a unique maximum"
            )
        return None

    dd = winners[0]
    if dd == (d + 2) % 4:
        if strict:
            raise AssertionError(
                f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
            )
        return None

    return dd, reachable[dd][0][1], counts


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    reachable = _reachable_by_dir(head, markers, offs, h, w)
    picked = _pick_target_from_reachable(head, d, reachable, strict=True)
    if picked is None:
        return None
    dd, pt, _counts = picked
    return dd, pt


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
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
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
    Advance a triangular sprite through the final visible frame until every marker is consumed.

    Input structure:
    - Frames are concatenated left-to-right and separated by solid divider columns of one color.
    - Each frame contains one triangle sprite plus same-color single-pixel markers on the background.

    Rule:
    - Read only the rightmost frame as the current state.
    - From the sprite head, collect reachable markers on each cardinal half-line
      (same row or same column, with the whole translated sprite still fitting).
    - If several directions are populated, compare how many reachable markers lie
      on each half-line; move toward the direction with the unique largest count,
      then land on the nearest marker on that winning side. Consume that marker,
      rotate the sprite to the travel direction, and continue until none remain.

    Key insight: the triangle tells you where the head is, but the next move is
    chosen by counting markers per direction, not by assuming only one direction exists.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (D14 count-disambiguated variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)          # 1..4 frames
FRAME_MAX = 16                   # 11..15 once sampled below
H_FLOOR, H_CAP = 11, 15
W_FLOOR, W_CAP = 11, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # 4..12 inclusive
STEP_DIST_RANGE = (2, 8)         # allowed distances are 2..7
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: variant visibility ---
SAFE_MARGIN = 2
AXIS_MIN_GAP = 2
AXIS_COUNT_MIN, AXIS_COUNT_MAX = 4, 5
MIN_MULTI_DIR_STEPS = 2
LAST_FRAME_MIN_MARKERS = 3       # ensures the last input frame can show a count comparison

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_PATH_ATTEMPTS = 24
MAX_ATTEMPTS = 300

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _safe_head(head: tuple[int, int], h: int, w: int) -> bool:
    y, x = head
    return SAFE_MARGIN <= y < h - SAFE_MARGIN and SAFE_MARGIN <= x < w - SAFE_MARGIN


def _max_axis_count(size: int) -> int:
    usable = max(0, size - 4)
    return (usable + 1) // 2


def _axis_count_options(h: int, w: int, n_steps: int) -> list[tuple[int, int]]:
    rmax = min(AXIS_COUNT_MAX, _max_axis_count(h))
    cmax = min(AXIS_COUNT_MAX, _max_axis_count(w))
    out: list[tuple[int, int]] = []
    for rc in range(AXIS_COUNT_MIN, rmax + 1):
        for cc in range(AXIS_COUNT_MIN, cmax + 1):
            if rc * cc >= n_steps + 1:
                out.append((rc, cc))
    return out


def _sample_axis_values(
    rng: np.random.Generator, size: int, count: int
) -> list[int] | None:
    vals = list(range(SAFE_MARGIN, size - SAFE_MARGIN))
    order = [int(v) for v in rng.permutation(vals)]
    out: list[int] = []
    for v in order:
        if all(abs(v - u) >= AXIS_MIN_GAP for u in out):
            out.append(v)
            if len(out) == count:
                return sorted(out)
    return None


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None, str, tuple[int, int]] | None:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [
        n
        for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and n // n_frames >= LAST_FRAME_MIN_MARKERS
    ]
    if not multiples:
        return None

    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h = int(rng.integers(H_FLOOR, FRAME_MAX))
    w = int(rng.integers(W_FLOOR, FRAME_MAX))
    axis_options = _axis_count_options(h, w, n_steps)
    if not axis_options:
        return None

    aw = np.array([rc * cc for rc, cc in axis_options], dtype=float)
    aw /= aw.sum()
    axis_counts = axis_options[int(rng.choice(len(axis_options), p=aw))]

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname, axis_counts


def _prepend_meta(
    cand: tuple[int, int],
    suffix: list[tuple[int, int]],
) -> dict[str, int] | None:
    cur = suffix[0]
    step_dir = _direction(cand, cur)
    if step_dir is None:
        return None

    d = _dist(cand, cur)
    if not (STEP_DIST_RANGE[0] <= d < STEP_DIST_RANGE[1]):
        return None

    if len(suffix) >= 2:
        next_dir = _direction(cur, suffix[1])
        if step_dir == (next_dir + 2) % 4:
            return None

    aligned = _aligned_markers(cand, set(suffix))
    if step_dir not in aligned:
        return None

    counts = {dd: len(pts) for dd, pts in aligned.items()}
    best_count = max(counts.values())
    if counts[step_dir] != best_count:
        return None
    if sum(cnt == best_count for cnt in counts.values()) != 1:
        return None
    if aligned[step_dir][0][1] != cur:
        return None

    sorted_counts = sorted(counts.values(), reverse=True)
    count_gap = best_count - (sorted_counts[1] if len(sorted_counts) > 1 else 0)
    return {
        "step_dir": step_dir,
        "dist": d,
        "active_dirs": len(aligned),
        "max_count": best_count,
        "count_gap": count_gap,
        "multi": int(len(aligned) >= 2),
    }


def _search_path(
    rng: np.random.Generator,
    n_steps: int,
    gap: int,
    h: int,
    w: int,
    row_count: int,
    col_count: int,
) -> list[tuple[int, int]] | None:
    target_len = n_steps + 1

    def extend(
        suffix: list[tuple[int, int]],
        used: set[tuple[int, int]],
        nodes: list[tuple[int, int]],
        multi_steps: int,
    ) -> list[tuple[int, int]] | None:
        if len(suffix) == target_len:
            return suffix

        candidates: list[tuple[int, int, int, int, int, tuple[int, int]]] = []
        for cand in nodes:
            if cand in used:
                continue
            meta = _prepend_meta(cand, suffix)
            if meta is None:
                continue

            new_len = len(suffix) + 1
            if new_len == gap + 1 and meta["multi"] == 0:
                continue

            remaining = target_len - new_len
            if multi_steps + meta["multi"] + remaining < MIN_MULTI_DIR_STEPS:
                continue

            candidates.append(
                (
                    meta["multi"],
                    meta["max_count"],
                    meta["count_gap"],
                    meta["active_dirs"],
                    meta["dist"],
                    cand,
                )
            )

        if not candidates:
            return None

        rng.shuffle(candidates)
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3], -abs(4 - t[4])), reverse=True)

        for multi_flag, _mc, _gap, _dirs, _distv, cand in candidates[:18]:
            out = extend([cand] + suffix, used | {cand}, nodes, multi_steps + multi_flag)
            if out is not None:
                return out
        return None

    for _ in range(MAX_PATH_ATTEMPTS):
        rows = _sample_axis_values(rng, h, row_count)
        cols = _sample_axis_values(rng, w, col_count)
        if rows is None or cols is None:
            continue

        nodes = [(r, c) for r in rows for c in cols]
        starts = [nodes[i] for i in rng.permutation(len(nodes))]
        for start in starts:
            out = extend([start], {start}, nodes, 0)
            if out is not None:
                return out
    return None


def _validate_path(
    heads: list[tuple[int, int]],
    tname: str,
    h: int,
    w: int,
    n_frames: int,
    gap: int,
) -> tuple[bool, dict]:
    n_steps = len(heads) - 1
    if n_steps <= 0 or n_steps % n_frames != 0:
        return False, {}

    step_dirs = [_direction(heads[i], heads[i + 1]) for i in range(n_steps)]
    if any(dd is None for dd in step_dirs):
        return False, {}

    step_dirs = [int(dd) for dd in step_dirs]
    state_dirs = [step_dirs[0]] + step_dirs[:-1]

    if any(not _safe_head(head, h, w) for head in heads):
        return False, {}

    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, {}
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, {}

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, {}

    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, {}

    multi_dir_steps = 0
    visible_branch_steps = 0
    max_dir_count = 1

    for i in range(n_steps):
        markers = set(heads[i + 1 :])
        offs = TEMPLATES[tname][state_dirs[i]]
        reachable = _reachable_by_dir(heads[i], markers, offs, h, w)
        picked = _pick_target_from_reachable(heads[i], state_dirs[i], reachable, strict=False)
        if picked is None:
            return False, {}

        nd, pt, counts = picked
        if (nd, pt) != (step_dirs[i], heads[i + 1]):
            return False, {}

        sprite = _sprite_cells(heads[i], tname, state_dirs[i])
        if any(marker in sprite for marker in markers):
            return False, {}

        if len(reachable) >= 2:
            multi_dir_steps += 1
        max_dir_count = max(max_dir_count, max(counts.values()))

    snap_indices = [fi * gap for fi in range(n_frames)]
    for si in snap_indices:
        markers = set(heads[si + 1 :])
        offs = TEMPLATES[tname][state_dirs[si]]
        reachable = _reachable_by_dir(heads[si], markers, offs, h, w)
        counts = {dd: len(pts) for dd, pts in reachable.items()}
        if len(counts) >= 2:
            vals = sorted(counts.values(), reverse=True)
            if vals[0] > vals[1]:
                visible_branch_steps += 1

    last_idx = (n_frames - 1) * gap
    last_markers = set(heads[last_idx + 1 :])
    last_reachable = _reachable_by_dir(heads[last_idx], last_markers, TEMPLATES[tname][state_dirs[last_idx]], h, w)
    last_counts = {dd: len(pts) for dd, pts in last_reachable.items()}
    if len(last_counts) < 2:
        return False, {}
    last_vals = sorted(last_counts.values(), reverse=True)
    if last_vals[0] <= last_vals[1]:
        return False, {}

    if multi_dir_steps < MIN_MULTI_DIR_STEPS or visible_branch_steps < 1:
        return False, {}

    return True, {
        "step_dirs": step_dirs,
        "state_dirs": state_dirs,
        "turns": turns,
        "step_distances": dists,
        "multi_dir_steps": multi_dir_steps,
        "visible_branch_steps": visible_branch_steps,
        "last_frame_counts": last_counts,
        "max_dir_count": max_dir_count,
    }


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
    params = _sample_params(rng)
    if params is None:
        return None
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname, axis_counts = params
    row_count, col_count = axis_counts

    heads = _search_path(rng, n_steps, gap, h, w, row_count, col_count)
    if heads is None:
        return None

    ok, meta = _validate_path(heads, tname, h, w, n_frames, gap)
    if not ok:
        return None

    step_dirs: list[int] = meta["step_dirs"]
    state_dirs: list[int] = meta["state_dirs"]

    frames: list[np.ndarray] = []
    for fi in range(n_frames):
        si = fi * gap
        frames.append(
            _render_frame(h, w, bg, actor, heads[si], state_dirs[si], tname, heads[si + 1 :])
        )

    out_frame = _render_frame(h, w, bg, actor, heads[-1], step_dirs[-1], tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, state_dirs[last_idx], heads[last_idx]):
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
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "axis_counts": list(axis_counts),
            "heads": list(heads),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "turns": meta["turns"],
            "turn_ratio": meta["turns"] / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_distances": meta["step_distances"],
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "multi_dir_steps": meta["multi_dir_steps"],
            "visible_branch_steps": meta["visible_branch_steps"],
            "last_frame_counts": dict(meta["last_frame_counts"]),
            "max_direction_count": meta["max_dir_count"],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
