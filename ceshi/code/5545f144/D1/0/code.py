# Variant of 5545f144
# Dimension: D1 — 匹配条件：方向对齐 (match_cond)
# Change: 精灵头部方向与标记之间的射线对齐——标记必须位于头部所朝方向的同行或同列半线上（direction_matching），
# ... → 颜色匹配消耗
#
# Design analysis for this variant
# a) Concrete behavior:
#    The triangle sprite remains a monochrome arrow and the input still contains
#    horizontally concatenated time snapshots. The changed object is the marker:
#    every future target is a 1x2 horizontal domino [match_color][next_color].
#    The sprite may move only to the LEFT cell of the unique domino whose left
#    color equals the sprite's current body color; after landing, that domino is
#    consumed and the sprite recolors to the domino's right color. The path is
#    still axis-aligned so the arrow orientation can keep tracking travel
#    direction, but target selection is now color-driven rather than direction-
#    driven. Example: original logic from a red right-facing sprite would pick
#    the only reachable aligned dot; new logic ignores facing for target choice,
#    finds the only red-left domino anywhere among the remaining tokens, moves
#    onto its left cell, then changes to that domino's right color.
# b) Complexity calibration:
#    The rule uses a small, repeated vocabulary: one sprite color chain and many
#    two-cell markers. This avoids ambiguous free-form recoloring while making
#    the visual difference from the source task obvious. Colors are limited by
#    the ARC palette, so total steps are capped at 7 to keep the color chain
#    injective when divider/background colors also consume palette slots.
# c) Review of original assumptions that break here:
#    - The original solver inferred a single actor color by excluding bg/divider.
#      That fails once markers have multiple colors, so agent detection now
#      searches every non-bg color for a triangle template.
#    - The original leftover pixels of the actor color were all markers. That is
#      no longer true globally; only leftover pixels of the sprite color are
#      related to the current marker, while other marker colors belong to other
#      dominoes. Marker extraction therefore must parse structured 1x2 tokens.
#    - Target choice originally assumed geometry yields a unique direction.
#      Here the uniqueness assumption shifts to color: exactly one remaining
#      domino has left color == current sprite color.
# d) Placement legality:
#    Generator construction guarantees each domino fits fully in-bounds, dominoes
#    are separated from each other, and every unconsumed domino stays clear of
#    every sprite state in which it is still visible. The left cell of each
#    domino is also the future landing head, so the solver can deterministically
#    reconstruct the path.
# e) Visual observability:
#    The change is visible because markers are no longer isolated single-color
#    dots. They are explicit color-transition tokens, and intermediate colors
#    repeat structurally: a color appears as one domino's right half and the
#    next domino's left half. This supports the intended "follow the color
#    chain" inference from a few examples instead of a hidden id-based rule.
# f) Visual cleanliness:
#    Domino markers are always horizontal and well-spaced. The sprite stays
#    monochrome at each snapshot, so triangle shape and marker tokens are easy
#    to distinguish without stacked overlaps.
# g) Color-change inferability:
#    No ordering depends on raw color ids. The recolor rule is purely local and
#    visible: the consumed left cell matches the sprite, and the adjacent right
#    cell supplies the next sprite color. Because the color chain is injective,
#    each step has one unambiguous matching token.

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


def _divider_colors(g: np.ndarray, bg: int) -> set[int]:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    return {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}


def _find_agent(
    frame: np.ndarray, forbidden: set[int]
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], int] | None:
    best = None  # (score, tname, d, head, placed, color)
    for color in sorted(set(map(int, frame.ravel())) - forbidden):
        pts = set(map(tuple, np.argwhere(frame == color)))
        if len(pts) < 3:
            continue
        for head in pts:
            hy, hx = int(head[0]), int(head[1])
            for tname, dirs in TEMPLATES.items():
                for d, offs in dirs.items():
                    placed = {(hy + dy, hx + dx) for dy, dx in offs}
                    if placed.issubset(pts):
                        leftover = len(pts - placed)
                        score = (len(placed), -leftover)
                        if best is None or score > best[0]:
                            best = (score, tname, d, (hy, hx), placed, color)
    if best is None:
        return None
    _, tname, d, head, placed, color = best
    return tname, d, head, set(map(tuple, placed)), int(color)


def _extract_tokens(
    frame: np.ndarray,
    bg: int,
    sprite: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], int, int]]:
    """
    Parse horizontal 1x2 marker dominoes: [match_color][next_color].
    Returns (left_cell, match_color, next_color).
    """
    h, w = frame.shape
    occupied = {
        (int(r), int(c))
        for r in range(h)
        for c in range(w)
        if int(frame[r, c]) != bg and (r, c) not in sprite
    }
    used: set[tuple[int, int]] = set()
    tokens: list[tuple[tuple[int, int], int, int]] = []
    for r, c in sorted(occupied):
        if (r, c) in used:
            continue
        if (r, c + 1) not in occupied:
            raise AssertionError(f"unpaired marker cell at {(r, c)}")
        if (r, c - 1) in occupied or (r, c + 2) in occupied:
            raise AssertionError(f"marker token not isolated at {(r, c)}")
        left = int(frame[r, c])
        right = int(frame[r, c + 1])
        if left == right:
            raise AssertionError(f"marker token colors must differ at {(r, c)}")
        used.add((r, c))
        used.add((r, c + 1))
        tokens.append(((int(r), int(c)), left, right))
    if used != occupied:
        raise AssertionError("failed to parse all marker cells")
    return tokens


def _simulate_to_end(frame: np.ndarray, forbidden: set[int]) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    found = _find_agent(frame, forbidden | {bg})
    if found is None:
        return np.full((h, w), bg, dtype=int)

    tname, d, head, sprite, actor = found
    tokens = _extract_tokens(frame, bg, sprite)

    max_steps = h * w
    for _ in range(max_steps):
        if not tokens:
            break
        matches = [tok for tok in tokens if tok[1] == actor]
        assert len(matches) == 1, f"expected one token for color {actor}, found {len(matches)}"
        (ty, tx), _match, new_actor = matches[0]
        nd = _direction(head, (ty, tx))
        assert nd is not None, f"move from {head} to {(ty, tx)} is not axis-aligned"
        assert nd != (d + 2) % 4, f"180-degree reversal from dir {d} to {nd} is not allowed"
        head = (ty, tx)
        d = nd
        actor = new_actor
        tokens.remove(matches[0])

    agent = _sprite_cells(head, tname, d)
    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a triangular sprite across the grid by following a visible color chain.

    Input structure:
    - Frames are concatenated left-to-right and separated by solid divider columns; the background is the most common color.
    - Each frame contains one monochrome triangular sprite plus several 1x2 horizontal marker dominoes whose two cells may have different colors.

    Rule:
    - Start from the rightmost frame.
    - Each marker is read as [consume_color][next_sprite_color]. The sprite may consume only the unique marker whose left cell matches the sprite's current color.
    - The sprite moves so its head lands on that marker's left cell, turns to face the move direction, removes the whole domino, and recolors to the marker's right-cell color.
    - Repeat until no markers remain; output only the final sprite on a clean background.

    Key insight: The arrow shape still gives the sprite's current head and facing, but target selection no longer depends on directional alignment; it depends on matching the sprite color to the left half of the remaining colored domino chain.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    div_colors = _divider_colors(g, bg)
    return _simulate_to_end(frames[-1], div_colors)


# ------------------------------- generator ---------------------------------

N_FRAMES_RANGE = (1, 5)  # 1..4
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 7, 15

STEPS_RANGE = (4, 8)  # 4..7 inclusive; color-injective chain needs at most 8 states
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 7
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)
MAX_ATTEMPTS = 600

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int | None, str, list[int]]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 2) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 5 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    divider = palette[1] if n_frames > 1 else None
    usable = palette[2:] if n_frames > 1 else palette[1:]
    state_colors = usable[: n_steps + 1]

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, divider, tname, list(state_colors)


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


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]],
    token_heads: list[tuple[int, int]],
    h: int,
    w: int,
    tname: str,
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
    for hr, hc in token_heads:
        tr_lo = max(tr_lo, -hr)
        tr_hi = min(tr_hi, h - 1 - hr)
        tc_lo = max(tc_lo, -hc)
        tc_hi = min(tc_hi, w - 2 - hc)  # right cue cell must fit
    if tr_lo > tr_hi or tc_lo > tc_hi:
        return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _token_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    token_heads: list[tuple[int, int]],
    tname: str,
) -> bool:
    cells = {nxt, (nxt[0], nxt[1] + 1)}
    if any(cell in heads for cell in cells):
        return False
    for old in token_heads:
        old_cells = {old, (old[0], old[1] + 1)}
        if any(_touching(a, b) for a in cells for b in old_cells):
            return False
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(cell, sp) for cell in cells for sp in sprite):
            return False
    return True


def _rollout_path(
    rng: np.random.Generator,
    tname: str,
    step_dirs: list[int],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    n_steps = len(step_dirs)
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [step_dirs[0]]
    token_heads: list[tuple[int, int]] = []

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
            if not _token_compatible_with_existing(nxt, heads, state_dirs, token_heads, tname):
                continue
            trial_heads = heads + [nxt]
            trial_dirs = state_dirs + [next_dir]
            shift = _shift_interval(
                list(zip(trial_heads, trial_dirs)),
                token_heads + [nxt],
                h,
                w,
                tname,
            )
            if shift is None:
                continue
            chosen = nxt
            break
        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        token_heads.append(chosen)

    return heads, state_dirs


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
    tname: str,
    tokens: list[tuple[tuple[int, int], int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for (r, c), left, right in tokens:
        frame[r, c] = left
        frame[r, c + 1] = right
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
    n_frames, n_steps, gap, h, w, bg, divider, tname, state_colors = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(list(zip(heads, state_dirs)), heads[1:], h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]

    placements = list(zip(heads, state_dirs))
    tokens = [
        (heads[i + 1], state_colors[i], state_colors[i + 1])
        for i in range(n_steps)
    ]

    # Oracle cleanliness checks for every visible state.
    for i in range(n_steps):
        sprite = _sprite_cells(heads[i], tname, state_dirs[i])
        for j in range(i, n_steps):
            tok_head = heads[j + 1]
            tok_cells = {tok_head, (tok_head[0], tok_head[1] + 1)}
            if any(_touching(cell, sp) for cell in tok_cells for sp in sprite):
                return None
    for i in range(n_steps):
        cells_i = {heads[i + 1], (heads[i + 1][0], heads[i + 1][1] + 1)}
        for j in range(i + 1, n_steps):
            cells_j = {heads[j + 1], (heads[j + 1][0], heads[j + 1][1] + 1)}
            if any(_touching(a, b) for a in cells_i for b in cells_j):
                return None

    frames = []
    for fi in range(n_frames):
        si = fi * gap
        frames.append(
            _render_frame(
                h,
                w,
                bg,
                state_colors[si],
                placements[si][0],
                placements[si][1],
                tname,
                tokens[si:],
            )
        )

    out_frame = _render_frame(
        h,
        w,
        bg,
        state_colors[-1],
        placements[-1][0],
        placements[-1][1],
        tname,
        [],
    )
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], {bg} | ({divider} if divider is not None else set()))
    if found is None:
        return None
    f_tname, f_dir, f_head, f_sprite, f_color = found
    if (f_tname, f_dir, f_head, f_color) != (
        tname,
        placements[last_idx][1],
        placements[last_idx][0],
        state_colors[last_idx],
    ):
        return None
    parsed = _extract_tokens(frames[-1], bg, f_sprite)
    if parsed != tokens[last_idx:]:
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
            "state_colors": list(state_colors),
            "color_token_count": len(tokens),
            "count_in_last_input_frame": len(tokens[last_idx:]),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
