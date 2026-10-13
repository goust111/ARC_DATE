from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste


def _stamp_solid(out: np.ndarray, patch: np.ndarray, r: int, c: int, bg: int) -> None:
    H, W = out.shape
    h, w = patch.shape
    rr0 = max(0, int(r))
    cc0 = max(0, int(c))
    rr1 = min(H, int(r) + h)
    cc1 = min(W, int(c) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return

    color = int(patch[0, 0])
    window = out[rr0:rr1, cc0:cc1]
    assert np.all((window == int(bg)) | (window == color)), "zigzag bars must not conflict"
    paste(out, patch, int(r), int(c), clip=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the single rectangular bar into two diagonal zigzag chains, alternating the bar with its 90-degree rotated copy in the paired color.

    Input structure
    - The grid has one background color and exactly one 4-connected monochrome foreground object.
    - That foreground object is a solid non-square rectangle, either green or pink.
    - The foreground rectangle's height and width define the alternating step period.

    Rule
    - Keep copies of the original rectangle on both diagonals through the seed, with row and column offsets of height + width.
    - Between every adjacent pair of same-diagonal original copies, place a 90-degree rotated rectangle in the other color.
    - Continue the pattern across the whole canvas and clip any rectangle that crosses an edge.

    Key insight
    - The two visible colors encode alternating orientations of the same bar along two crossing 45-degree zigzag paths; boundary fragments are just clipped copies.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) == 1, "expected one foreground bar"

    comp = components[0]
    color = int(comp["color"])
    assert color in (Color.GREEN, Color.PINK), "bar color must be green or pink"

    r0, c0, r1, c1 = comp["bbox"]
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    assert h != w, "seed bar must be non-square"
    assert int(comp["area"]) == h * w, "seed bar must be a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "seed bbox must be solid"

    other = Color.PINK if color == Color.GREEN else Color.GREEN
    original = np.full((h, w), color, dtype=int)
    rotated = np.full((w, h), other, dtype=int)
    out = np.full_like(g, bg)

    period = h + w
    limit = g.shape[0] + g.shape[1] + period + 2
    for n in range(-limit, limit + 1):
        offset = n * period

        _stamp_solid(out, original, r0 + offset, c0 + offset, bg)
        _stamp_solid(out, original, r0 + offset, c0 - offset, bg)

        _stamp_solid(out, rotated, r0 + h + offset, c0 + w + offset, bg)
        _stamp_solid(out, rotated, r0 + h + offset, c0 - h - offset, bg)

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette controls.
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 26
BACKGROUND_COLOR, BAR_COLORS = 0, (3, 6)
# Seed rectangle size constraints.
BAR_DIM_MIN, BAR_DIM_MAX = 1, 5
SHORT_SIDE_OPTIONS = (1, 2)
LONG_SIDE_MIN, LONG_SIDE_MAX = 2, 5
MAX_BAR_AREA, MIN_GRID_PERIOD_RATIO = 10, 3.5
# Zigzag visibility and retry thresholds.
VISIBLE_STAMP_MIN, VISIBLE_STAMP_MAX = 12, 17
MIN_CHAIN_STAMPS, GENERATION_RETRIES = 5, 300
STAMP_SCAN_GRID_FACTOR, STAMP_SCAN_PADDING = 2, 2


def _gen_dimension_options(N: int) -> list[tuple[int, int]]:
    opts = []
    for h in range(BAR_DIM_MIN, BAR_DIM_MAX + 1):
        for w in range(BAR_DIM_MIN, BAR_DIM_MAX + 1):
            short, long = min(h, w), max(h, w)
            if (
                h != w
                and short in SHORT_SIDE_OPTIONS
                and LONG_SIDE_MIN <= long <= LONG_SIDE_MAX
                and h * w <= MAX_BAR_AREA
                and N / (h + w) >= MIN_GRID_PERIOD_RATIO
            ):
                opts.append((h, w))
    return opts


def _visible_stamp_counts(N: int, h: int, w: int, r0: int, c0: int) -> tuple[int, int, int]:
    period = h + w
    limit = STAMP_SCAN_GRID_FACTOR * N + period + STAMP_SCAN_PADDING
    chain_stamps = [set(), set()]

    def add_stamp(chain: int, hh: int, ww: int, r: int, c: int, kind: int) -> None:
        if max(0, r) < min(N, r + hh) and max(0, c) < min(N, c + ww):
            chain_stamps[chain].add((kind, r, c, hh, ww))

    for n in range(-limit, limit + 1):
        offset = n * period
        add_stamp(0, h, w, r0 + offset, c0 + offset, 0)
        add_stamp(1, h, w, r0 + offset, c0 - offset, 0)
        add_stamp(0, w, h, r0 + h + offset, c0 + w + offset, 1)
        add_stamp(1, w, h, r0 + h + offset, c0 - h - offset, 1)

    return len(chain_stamps[0] | chain_stamps[1]), len(chain_stamps[0]), len(chain_stamps[1])


def _gen_position_options(N: int, h: int, w: int) -> list[tuple[int, int]]:
    opts = []
    for r0 in range(N - h + 1):
        for c0 in range(N - w + 1):
            total, chain_a, chain_b = _visible_stamp_counts(N, h, w, r0, c0)
            if (
                VISIBLE_STAMP_MIN <= total <= VISIBLE_STAMP_MAX
                and chain_a >= MIN_CHAIN_STAMPS
                and chain_b >= MIN_CHAIN_STAMPS
            ):
                opts.append((r0, c0))
    return opts


def _gen_attempt() -> np.ndarray | None:
    N = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    dims = _gen_dimension_options(N)
    if not dims:
        return None

    h, w = dims[int(np.random.randint(0, len(dims)))]
    positions = _gen_position_options(N, h, w)
    if not positions:
        return None

    color = int(np.random.choice(BAR_COLORS))
    r0, c0 = positions[int(np.random.randint(0, len(positions)))]

    inp = np.full((N, N), BACKGROUND_COLOR, dtype=int)
    inp[r0 : r0 + h, c0 : c0 + w] = color

    try:
        out = main(inp)
    except Exception:
        return None
    if out.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate fd4b2b02 input")
