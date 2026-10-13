import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components
import random


def _largest_solid_square(g: np.ndarray) -> dict:
    components = find_components(
        g,
        bg=-1,
        colors=colors_present(g),
        connectivity=4,
        monochromatic=True,
    )

    squares = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if h == w and int(comp["area"]) == h * w and h >= 2:
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"]))
            squares.append(comp)

    assert squares, "expected at least one solid square component"
    max_side = max(bbox_h(comp["bbox"]) for comp in squares)
    largest = [comp for comp in squares if bbox_h(comp["bbox"]) == max_side]
    assert len(largest) == 1, "expected a unique largest solid square occluder"
    return largest[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the pattern hidden by the unique solid square occluder using mirror symmetry.

    Input structure
    - The input is a square scene with one unique largest solid monochrome square occluder.
    - The occluder's color appears only in that square, and the rest of the scene is mirror-symmetric.

    Rule
    - Reflect the occluder's bbox across the vertical axis, horizontal axis, and both axes.
    - Ignore reflected regions that still contain the occluder color because they overlap the hidden square.
    - Transform each usable reflected region back into the occluder orientation and return that recovered square.

    Key insight
    - The task asks for only the occluded square, not the full repaired scene; every unoccluded mirror counterpart yields the same patch.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H == W, "expected a square input grid"

    occluder = _largest_solid_square(g)
    color = int(occluder["color"])
    r0, c0, r1, c1 = occluder["bbox"]
    h = bbox_h(occluder["bbox"])
    w = bbox_w(occluder["bbox"])
    assert np.count_nonzero(g == color) == h * w, "occluder color must occur only in the occluder"

    reflections = [
        (r0, W - 1 - c1, r1, W - 1 - c0, np.fliplr),
        (H - 1 - r1, c0, H - 1 - r0, c1, np.flipud),
        (H - 1 - r1, W - 1 - c1, H - 1 - r0, W - 1 - c0, lambda a: np.rot90(a, 2)),
    ]

    recovered = []
    for rr0, cc0, rr1, cc1, transform in reflections:
        assert 0 <= rr0 <= rr1 < H and 0 <= cc0 <= cc1 < W
        patch = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
        assert patch.shape == (h, w)
        if not np.any(patch == color):
            recovered.append(transform(patch))

    assert recovered, "expected at least one unoccluded mirror counterpart"
    assert all(np.array_equal(recovered[0], patch) for patch in recovered[1:]), "mirror counterparts must agree"
    return recovered[0].copy()

# --------------- generator (o5x) ---------------

# Canvas and occluder geometry.
_GEN_GRID_SIZE = 16
_GEN_QUAD_SIZE = 8
_GEN_OCCLUDER_SIZE = 5
_GEN_OCCLUDER_COLOR = 3

# Palette selection.
_GEN_NON3_COLORS = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_GEN_FG_COUNT_RANGE = (4, 5)

# Quadrant patterning.
_GEN_Q_MODES = ["pixel", "blocks", "mixed"]
_GEN_PIXEL_BG_PROB_RANGE = (0.43, 0.58)
_GEN_BLOCK_COUNT_RANGE = (8, 18)
_GEN_BLOCK_SIDE_RANGE = (1, 3)
_GEN_LARGE_BLOCK_PROB = 0.15
_GEN_LARGE_BLOCK_SIDE_RANGE = (3, 4)
_GEN_NOISE_COUNT_RANGE = (4, 12)
_GEN_NOISE_BG_PROB = 0.65
_GEN_BG_BALANCE_STEPS = 80
_GEN_BG_DOMINANCE_MARGIN = 7
_GEN_MIN_COLOR_CELLS_TO_REPAINT = 2
_GEN_UNIFORM_BREAK_STEPS = 40

# Occluder placement and retry limits.
_GEN_MIN_NONOVERLAP_REFLECTIONS = 2
_GEN_CROSSING_POSITION_PROB = 0.6
_GEN_RETRY_LIMIT = 300


def _gen_expand_q(q: np.ndarray) -> np.ndarray:
    top = np.concatenate([q, np.fliplr(q)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _gen_set_q_orbit(q: np.ndarray, r: int, c: int, color: int) -> None:
    q[r, c] = color
    q[c, r] = color


def _gen_paint_q_rect(q: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    q[r0 : r0 + h, c0 : c0 + w] = color
    q[c0 : c0 + w, r0 : r0 + h] = color


def _gen_force_palette(q: np.ndarray, colors: list[int]) -> None:
    used = set(int(x) for x in q.flat)
    spots = [(r, c) for r in range(_GEN_QUAD_SIZE) for c in range(r, _GEN_QUAD_SIZE)]
    random.shuffle(spots)
    for color in colors:
        if color not in used and spots:
            r, c = spots.pop()
            _gen_set_q_orbit(q, r, c, color)
            used.add(color)


def _gen_balance_bg(q: np.ndarray, bg: int, fg: list[int]) -> None:
    for _ in range(_GEN_BG_BALANCE_STEPS):
        counts = {color: int(np.count_nonzero(q == color)) for color in [bg] + fg}
        if counts[bg] >= max(counts[color] for color in fg) + _GEN_BG_DOMINANCE_MARGIN:
            return
        color = max(fg, key=lambda x: counts[x])
        coords = [(int(r), int(c)) for r, c in np.argwhere(q == color) if counts[color] > _GEN_MIN_COLOR_CELLS_TO_REPAINT]
        if not coords:
            return
        r, c = random.choice(coords)
        _gen_set_q_orbit(q, r, c, bg)


def _gen_break_uniform_5x5(q: np.ndarray, colors: list[int]) -> bool:
    for _ in range(_GEN_UNIFORM_BREAK_STEPS):
        scene = _gen_expand_q(q)
        changed = False
        for r0 in range(_GEN_GRID_SIZE - _GEN_OCCLUDER_SIZE + 1):
            for c0 in range(_GEN_GRID_SIZE - _GEN_OCCLUDER_SIZE + 1):
                patch = scene[r0 : r0 + _GEN_OCCLUDER_SIZE, c0 : c0 + _GEN_OCCLUDER_SIZE]
                color = int(patch[0, 0])
                if np.all(patch == color):
                    rr = r0 + _GEN_OCCLUDER_SIZE // 2
                    cc = c0 + _GEN_OCCLUDER_SIZE // 2
                    qr = rr if rr < _GEN_QUAD_SIZE else _GEN_GRID_SIZE - 1 - rr
                    qc = cc if cc < _GEN_QUAD_SIZE else _GEN_GRID_SIZE - 1 - cc
                    choices = [x for x in colors if x != color]
                    _gen_set_q_orbit(q, qr, qc, random.choice(choices))
                    changed = True
                    break
            if changed:
                break
        if not changed:
            return True
    return False


def _gen_make_q(bg: int, fg: list[int]) -> np.ndarray:
    q = np.full((_GEN_QUAD_SIZE, _GEN_QUAD_SIZE), bg, dtype=int)
    mode = random.choice(_GEN_Q_MODES)

    if mode in ("pixel", "mixed"):
        bg_p = random.uniform(*_GEN_PIXEL_BG_PROB_RANGE)
        for r in range(_GEN_QUAD_SIZE):
            for c in range(r, _GEN_QUAD_SIZE):
                color = bg if random.random() < bg_p else random.choice(fg)
                _gen_set_q_orbit(q, r, c, color)

    if mode in ("blocks", "mixed"):
        for _ in range(random.randint(*_GEN_BLOCK_COUNT_RANGE)):
            h = random.randint(*_GEN_BLOCK_SIDE_RANGE)
            w = random.randint(*_GEN_BLOCK_SIDE_RANGE)
            if random.random() < _GEN_LARGE_BLOCK_PROB:
                h = random.randint(*_GEN_LARGE_BLOCK_SIDE_RANGE)
            if random.random() < _GEN_LARGE_BLOCK_PROB:
                w = random.randint(*_GEN_LARGE_BLOCK_SIDE_RANGE)
            r0 = random.randint(0, _GEN_QUAD_SIZE - h)
            c0 = random.randint(0, _GEN_QUAD_SIZE - w)
            _gen_paint_q_rect(q, r0, c0, h, w, random.choice(fg))

        for _ in range(random.randint(*_GEN_NOISE_COUNT_RANGE)):
            r = random.randint(0, _GEN_QUAD_SIZE - 1)
            c = random.randint(r, _GEN_QUAD_SIZE - 1)
            _gen_set_q_orbit(q, r, c, bg if random.random() < _GEN_NOISE_BG_PROB else random.choice(fg))

    _gen_force_palette(q, [bg] + fg)
    _gen_balance_bg(q, bg, fg)
    if not _gen_break_uniform_5x5(q, [bg] + fg):
        return q
    _gen_force_palette(q, [bg] + fg)
    _gen_balance_bg(q, bg, fg)
    return q


def _gen_crosses_axis(start: int) -> bool:
    return start <= _GEN_QUAD_SIZE - 1 and start + _GEN_OCCLUDER_SIZE - 1 >= _GEN_QUAD_SIZE


def _gen_boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _gen_nonoverlap_reflections(r0: int, c0: int) -> int:
    last = _GEN_GRID_SIZE - 1
    r1 = r0 + _GEN_OCCLUDER_SIZE - 1
    c1 = c0 + _GEN_OCCLUDER_SIZE - 1
    box = (r0, c0, r1, c1)
    reflected = [
        (r0, last - c1, r1, last - c0),
        (last - r1, c0, last - r0, c1),
        (last - r1, last - c1, last - r0, last - c0),
    ]
    return sum(not _gen_boxes_overlap(box, other) for other in reflected)


def _gen_occluder_position() -> tuple[int, int]:
    crossing = []
    noncrossing = []
    for r0 in range(_GEN_GRID_SIZE - _GEN_OCCLUDER_SIZE + 1):
        for c0 in range(_GEN_GRID_SIZE - _GEN_OCCLUDER_SIZE + 1):
            row_cross = _gen_crosses_axis(r0)
            col_cross = _gen_crosses_axis(c0)
            if row_cross and col_cross:
                continue
            if _gen_nonoverlap_reflections(r0, c0) < _GEN_MIN_NONOVERLAP_REFLECTIONS:
                continue
            if row_cross or col_cross:
                crossing.append((r0, c0))
            else:
                noncrossing.append((r0, c0))
    if crossing and (not noncrossing or random.random() < _GEN_CROSSING_POSITION_PROB):
        return random.choice(crossing)
    return random.choice(noncrossing)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_GEN_NON3_COLORS)
    fg_count = random.randint(*_GEN_FG_COUNT_RANGE)
    fg = random.sample([c for c in _GEN_NON3_COLORS if c != bg], fg_count)
    q = _gen_make_q(bg, fg)
    scene = _gen_expand_q(q)

    r0, c0 = _gen_occluder_position()
    expected = scene[r0 : r0 + _GEN_OCCLUDER_SIZE, c0 : c0 + _GEN_OCCLUDER_SIZE].copy()
    inp = scene.copy()
    inp[r0 : r0 + _GEN_OCCLUDER_SIZE, c0 : c0 + _GEN_OCCLUDER_SIZE] = _GEN_OCCLUDER_COLOR

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid dc0a314f input")
