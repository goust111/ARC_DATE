from __future__ import annotations

import random

import numpy as np


Axis = tuple[str, int]


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 22
MIN_WING = 3
MIN_SPAN = 6
MAX_GENERATION_ATTEMPTS = 400
DOUBLE_NOISE_PROB = 0.35


def _mirror(axis: Axis, r: int, c: int) -> tuple[int, int]:
    kind, axis_sum = axis
    if kind == "vertical":
        return r, axis_sum - c
    return axis_sum - r, c


def _axis_candidates(shape: tuple[int, int]) -> list[Axis]:
    h, w = shape
    return [("vertical", s) for s in range(2 * w - 1)] + [("horizontal", s) for s in range(2 * h - 1)]


def _foreground_color(g: np.ndarray, bg: int) -> int | None:
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return None
    colors, counts = np.unique(non_bg, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _axis_score(g: np.ndarray, fg: int, bg: int, axis: Axis) -> tuple[int, int, int, int, int, int]:
    h, w = g.shape
    contradictions = 0
    unmatched_signal = 0
    signal_pairs = 0
    exact_pairs = 0
    repairable_pairs = 0
    axis_noise = 0

    for r in range(h):
        for c in range(w):
            rr, cc = _mirror(axis, r, c)
            if not (0 <= rr < h and 0 <= cc < w):
                if int(g[r, c]) != bg:
                    unmatched_signal += 1
                continue
            if (rr < r) or (rr == r and cc < c):
                continue

            va = int(g[r, c])
            vb = int(g[rr, cc])
            if va != bg or vb != bg:
                signal_pairs += 1

            a_known = va == bg or va == fg
            b_known = vb == bg or vb == fg
            if a_known and b_known:
                if va == vb:
                    exact_pairs += 1
                else:
                    contradictions += 1
            else:
                repairable_pairs += 1

            if r == rr and c == cc and not a_known:
                axis_noise += 1

    return (contradictions, unmatched_signal, -signal_pairs, -exact_pairs, -repairable_pairs, axis_noise)


def _sorted_axes(g: np.ndarray, fg: int, bg: int) -> list[tuple[tuple[int, int, int, int, int, int], Axis]]:
    ranked = [(_axis_score(g, fg, bg, axis), axis) for axis in _axis_candidates(g.shape)]
    ranked.sort()
    return ranked


def _repair_pair(va: int, vb: int, fg: int, bg: int) -> int:
    a_known = va == bg or va == fg
    b_known = vb == bg or vb == fg
    if a_known and b_known:
        if va == vb:
            return va
        return fg if va == fg or vb == fg else bg
    if a_known:
        return va
    if b_known:
        return vb
    return fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the hidden reflection axis of a noisy symmetric figure and repair the
    corrupted cells from their mirrored counterparts.

    The intended inputs contain:
    - black background,
    - one dominant foreground color forming a single-axis symmetric pattern,
    - two sparse side-specific noise colors that replace some cells on opposite sides.

    The solver infers the unique best horizontal/vertical reflection axis by
    minimizing contradictions among non-noise mirror pairs, then removes noise by
    copying the mirrored value. When both mirrored cells are noise, both become
    foreground.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = BACKGROUND if bool(np.any(g == BACKGROUND)) else int(np.bincount(g.ravel()).argmax())
    fg = _foreground_color(g, bg)
    if fg is None:
        return g.copy()

    non_bg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    if len(non_bg_colors) == 1:
        return np.where(g == fg, fg, bg).astype(int)

    axis = _sorted_axes(g, fg, bg)[0][1]
    h, w = g.shape
    out = np.full_like(g, bg)

    for r in range(h):
        for c in range(w):
            rr, cc = _mirror(axis, r, c)
            if not (0 <= rr < h and 0 <= cc < w):
                out[r, c] = fg if int(g[r, c]) == fg else bg
                continue
            if (rr < r) or (rr == r and cc < c):
                continue

            fill = _repair_pair(int(g[r, c]), int(g[rr, cc]), fg, bg)
            out[r, c] = fill
            out[rr, cc] = fill

    return out


def _grow_blob(height: int, width: int, target: int) -> np.ndarray | None:
    for _ in range(60):
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        add_frontier(*seed)
        while frontier and len(cells) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(*cell)

        if len(cells) == target:
            mask = np.zeros((height, width), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return mask
    return None


def _sample_offset(limit: int, span: int) -> int:
    slack = limit - span
    assert slack >= 0
    if slack <= 1:
        return 0
    return random.randint(1, slack - 1)


def _build_symmetric_local(
    orientation: str, axis_mode: str, grid_h: int, grid_w: int
) -> tuple[np.ndarray, Axis, int, int] | None:
    if orientation == "vertical":
        max_wing = (grid_w - 2 - (1 if axis_mode == "cell" else 0)) // 2
        if max_wing < MIN_WING or grid_h - 2 < MIN_SPAN:
            return None
        wing = random.randint(MIN_WING, max_wing)
        local_w = 2 * wing + (1 if axis_mode == "cell" else 0)
        local_h = random.randint(MIN_SPAN, grid_h - 2)
        canon_h, canon_w = local_h, wing + (1 if axis_mode == "cell" else 0)
        target_lo = max(7, (canon_h * canon_w) // 4)
        target_hi = max(target_lo, (canon_h * canon_w * 3) // 5)
        blob = _grow_blob(canon_h, canon_w, random.randint(target_lo, target_hi))
        if blob is None:
            return None

        local = np.zeros((local_h, local_w), dtype=bool)
        for r, c in np.argwhere(blob):
            rr = int(r)
            cc = int(c)
            local[rr, cc] = True
            local[rr, local_w - 1 - cc] = True

        row0 = _sample_offset(grid_h, local_h)
        col0 = _sample_offset(grid_w, local_w)
        axis = ("vertical", 2 * col0 + local_w - 1)
        return local, axis, row0, col0

    max_wing = (grid_h - 2 - (1 if axis_mode == "cell" else 0)) // 2
    if max_wing < MIN_WING or grid_w - 2 < MIN_SPAN:
        return None
    wing = random.randint(MIN_WING, max_wing)
    local_h = 2 * wing + (1 if axis_mode == "cell" else 0)
    local_w = random.randint(MIN_SPAN, grid_w - 2)
    canon_h, canon_w = wing + (1 if axis_mode == "cell" else 0), local_w
    target_lo = max(7, (canon_h * canon_w) // 4)
    target_hi = max(target_lo, (canon_h * canon_w * 3) // 5)
    blob = _grow_blob(canon_h, canon_w, random.randint(target_lo, target_hi))
    if blob is None:
        return None

    local = np.zeros((local_h, local_w), dtype=bool)
    for r, c in np.argwhere(blob):
        rr = int(r)
        cc = int(c)
        local[rr, cc] = True
        local[local_h - 1 - rr, cc] = True

    row0 = _sample_offset(grid_h, local_h)
    col0 = _sample_offset(grid_w, local_w)
    axis = ("horizontal", 2 * row0 + local_h - 1)
    return local, axis, row0, col0


def _pair_descriptors(
    local: np.ndarray, axis: Axis, row0: int, col0: int
) -> list[tuple[tuple[int, int], tuple[int, int], bool]]:
    local_h, local_w = local.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
    kind, _axis_sum = axis

    if kind == "vertical":
        wing = local_w // 2
        for r in range(local_h):
            for c in range(wing):
                left = (row0 + r, col0 + c)
                right = (row0 + r, col0 + local_w - 1 - c)
                pairs.append((left, right, bool(local[r, c])))
        return pairs

    wing = local_h // 2
    for r in range(wing):
        for c in range(local_w):
            top = (row0 + r, col0 + c)
            bottom = (row0 + local_h - 1 - r, col0 + c)
            pairs.append((top, bottom, bool(local[r, c])))
    return pairs


def _choose_side_indices(
    fg_pool: list[int], bg_pool: list[int], total: int, banned: set[int]
) -> set[int] | None:
    fg_choices = [idx for idx in fg_pool if idx not in banned]
    bg_choices = [idx for idx in bg_pool if idx not in banned]
    if not fg_choices:
        return None

    chosen = {random.choice(fg_choices)}
    remaining = total - 1
    pool = [idx for idx in fg_choices + bg_choices if idx not in chosen]
    if remaining > len(pool):
        return None
    if remaining > 0:
        chosen.update(random.sample(pool, remaining))
    return chosen


def _axis_is_unique(g: np.ndarray, fg: int, bg: int, expected: Axis) -> bool:
    ranked = _sorted_axes(g, fg, bg)
    return ranked[0][1] == expected and (len(ranked) == 1 or ranked[1][0] != ranked[0][0])


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid_w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    orientation = random.choice(("vertical", "horizontal"))
    axis_mode = random.choice(("cell", "gap"))

    built = _build_symmetric_local(orientation, axis_mode, grid_h, grid_w)
    if built is None:
        return None
    local, axis, row0, col0 = built

    fg, noise_a, noise_b = random.sample(PALETTE, 3)
    clean = np.zeros((grid_h, grid_w), dtype=int)
    window = clean[row0 : row0 + local.shape[0], col0 : col0 + local.shape[1]]
    window[local] = fg

    if int(np.count_nonzero(clean == fg)) < 18:
        return None
    if not _axis_is_unique(clean, fg, BACKGROUND, axis):
        return None

    pairs = _pair_descriptors(local, axis, row0, col0)
    fg_pairs = [idx for idx, (_, _, is_fg) in enumerate(pairs) if is_fg]
    bg_pairs = [idx for idx, (_, _, is_fg) in enumerate(pairs) if not is_fg]
    if len(fg_pairs) < 4:
        return None

    max_side_noise = min(8, max(2, len(pairs) // 6))
    double_noise: set[int] = set()
    if len(fg_pairs) >= 6 and random.random() < DOUBLE_NOISE_PROB:
        max_double = min(2, max(1, len(fg_pairs) // 5))
        double_noise = set(random.sample(fg_pairs, random.randint(1, max_double)))

    left_count = random.randint(1, max_side_noise)
    right_count = random.choice([count for count in range(1, max_side_noise + 1) if count != left_count])
    left_noise = _choose_side_indices(fg_pairs, bg_pairs, left_count, double_noise)
    if left_noise is None:
        return None
    right_noise = _choose_side_indices(fg_pairs, bg_pairs, right_count, double_noise | left_noise)
    if right_noise is None:
        return None

    inp = clean.copy()
    for idx in left_noise:
        (r, c), _mirror_cell, _ = pairs[idx]
        inp[r, c] = noise_a
    for idx in right_noise:
        _cell, (r, c), _ = pairs[idx]
        inp[r, c] = noise_b
    for idx in double_noise:
        (r1, c1), (r2, c2), _ = pairs[idx]
        inp[r1, c1] = noise_a
        inp[r2, c2] = noise_b

    inferred_fg = _foreground_color(inp, BACKGROUND)
    if inferred_fg != fg:
        return None
    if not _axis_is_unique(inp, fg, BACKGROUND, axis):
        return None
    if not np.array_equal(main(inp), clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
