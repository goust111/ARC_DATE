import numpy as np


LAYER_COUNT_RANGE = (2, 5)
TOP_HEIGHT_RANGE = (2, 6)
TOP_WIDTH_RANGE = (2, 6)
EXPANSION_RANGE = (1, 4)
MAX_SCENE_SIZE = 18
TOP_PAD_RANGE = (1, 5)
GAP_TO_STRIP_RANGE = (0, 3)
WIDTH_EXTRA_RANGE = (4, 10)
STRIP_GAP_RANGE = (0, 3)
GEN_ATTEMPTS = 300

L_PATTERNS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
U_PATTERNS = (
    ("top", "left", "right"),
    ("bottom", "left", "right"),
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
)
BAR_PATTERNS = (
    ("top", "bottom"),
    ("left", "right"),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0)
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _recover_rectangle(visible: np.ndarray, covered: np.ndarray) -> tuple[int, int, int, int]:
    assert np.any(visible)
    assert not np.any(visible & covered)

    h, w = visible.shape
    vr0, vc0, vr1, vc1 = _bbox(visible)
    allowed = visible | covered
    bad_prefix = _prefix_sum(~allowed)

    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(vr0 + 1):
        for r1 in range(vr1, h):
            for c0 in range(vc0 + 1):
                for c1 in range(vc1, w):
                    if _rect_sum(bad_prefix, r0, c0, r1, c1) == 0:
                        candidates.append((r0, c0, r1, c1))
                        if len(candidates) > 1:
                            break
                if len(candidates) > 1:
                    break
            if len(candidates) > 1:
                break
        if len(candidates) > 1:
            break

    assert len(candidates) == 1, "rectangle must be uniquely recoverable"
    return candidates[0]


def _recover_rectangles(
    scene: np.ndarray, order_bottom_to_top: list[int]
) -> list[tuple[int, int, int, int]]:
    covered = np.zeros_like(scene, dtype=bool)
    rects_top_to_bottom: list[tuple[int, int, int, int]] = []
    for color in reversed(order_bottom_to_top):
        visible = scene == color
        rect = _recover_rectangle(visible, covered)
        rects_top_to_bottom.append(rect)
        r0, c0, r1, c1 = rect
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    return rects_top_to_bottom


def _pack_output(
    rects_bottom_to_top: list[tuple[int, int, int, int]],
    order_bottom_to_top: list[int],
    bg: int,
) -> np.ndarray:
    dims = []
    for rect in rects_bottom_to_top:
        r0, c0, r1, c1 = rect
        dims.append((r1 - r0 + 1, c1 - c0 + 1))

    out_h = max(h for h, _ in dims)
    out_w = sum(w for _, w in dims) + len(dims) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for color, (rect_h, rect_w) in zip(order_bottom_to_top, dims):
        out[:rect_h, col : col + rect_w] = color
        col += rect_w + 1
    return out


def _solve(scene: np.ndarray, order_bottom_to_top: list[int], bg: int) -> np.ndarray:
    rects_top_to_bottom = _recover_rectangles(scene, order_bottom_to_top)
    rects_bottom_to_top = list(reversed(rects_top_to_bottom))
    return _pack_output(rects_bottom_to_top, order_bottom_to_top, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each stacked solid rectangle from the bottom legend row and pack the
    restored layers from left to right in bottom-to-top order.

    The output row is top-aligned inside the minimal canvas of the packed rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _get_bg(g)
    scene = g[:-1]
    legend = g[-1]
    order_bottom_to_top = [int(color) for color in legend if int(color) != bg]

    assert order_bottom_to_top
    assert len(order_bottom_to_top) == len(set(order_bottom_to_top))

    scene_colors = set(map(int, np.unique(scene))) - {bg}
    assert scene_colors == set(order_bottom_to_top)

    return _solve(scene, order_bottom_to_top, bg)


def _expand_rectangle(
    rect: tuple[int, int, int, int], deltas: dict[str, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (
        r0 - deltas["top"],
        c0 - deltas["left"],
        r1 + deltas["bottom"],
        c1 + deltas["right"],
    )


def _sample_pattern(rng: np.random.Generator) -> dict[str, int]:
    family = int(rng.integers(0, 3))
    if family == 0:
        sides = L_PATTERNS[int(rng.integers(len(L_PATTERNS)))]
    elif family == 1:
        sides = U_PATTERNS[int(rng.integers(len(U_PATTERNS)))]
    else:
        sides = BAR_PATTERNS[int(rng.integers(len(BAR_PATTERNS)))]

    deltas = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    lo, hi = EXPANSION_RANGE
    for side in sides:
        deltas[side] = int(rng.integers(lo, hi + 1))
    return deltas


def _build_nested_rectangles(
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    layer_lo, layer_hi = LAYER_COUNT_RANGE
    count = int(rng.integers(layer_lo, layer_hi + 1))

    top_h = int(rng.integers(TOP_HEIGHT_RANGE[0], TOP_HEIGHT_RANGE[1] + 1))
    top_w = int(rng.integers(TOP_WIDTH_RANGE[0], TOP_WIDTH_RANGE[1] + 1))
    rects_top_to_bottom = [(0, 0, top_h - 1, top_w - 1)]

    for _ in range(count - 1):
        candidate = _expand_rectangle(rects_top_to_bottom[-1], _sample_pattern(rng))
        r0, c0, r1, c1 = candidate
        if r1 - r0 + 1 > MAX_SCENE_SIZE or c1 - c0 + 1 > MAX_SCENE_SIZE:
            return []
        rects_top_to_bottom.append(candidate)
    return rects_top_to_bottom


def _render_input(
    rects_top_to_bottom: list[tuple[int, int, int, int]],
    colors_top_to_bottom: list[int],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    all_r0 = min(rect[0] for rect in rects_top_to_bottom)
    all_c0 = min(rect[1] for rect in rects_top_to_bottom)
    all_r1 = max(rect[2] for rect in rects_top_to_bottom)
    all_c1 = max(rect[3] for rect in rects_top_to_bottom)
    scene_h = all_r1 - all_r0 + 1
    scene_w = all_c1 - all_c0 + 1

    order_bottom_to_top = list(reversed(colors_top_to_bottom))
    strip_gaps = [
        int(rng.integers(STRIP_GAP_RANGE[0], STRIP_GAP_RANGE[1] + 1))
        for _ in range(len(order_bottom_to_top) - 1)
    ]
    strip_w = len(order_bottom_to_top) + sum(strip_gaps)

    top_pad = int(rng.integers(TOP_PAD_RANGE[0], TOP_PAD_RANGE[1] + 1))
    gap_to_strip = int(rng.integers(GAP_TO_STRIP_RANGE[0], GAP_TO_STRIP_RANGE[1] + 1))
    width_extra = int(rng.integers(WIDTH_EXTRA_RANGE[0], WIDTH_EXTRA_RANGE[1] + 1))

    height = top_pad + scene_h + gap_to_strip + 1
    width = max(scene_w, strip_w) + width_extra
    grid = np.full((height, width), bg, dtype=int)

    scene_left = int(rng.integers(0, width - scene_w + 1))
    row_shift = top_pad - all_r0
    col_shift = scene_left - all_c0

    for color, rect in zip(reversed(colors_top_to_bottom), reversed(rects_top_to_bottom)):
        r0, c0, r1, c1 = rect
        grid[r0 + row_shift : r1 + row_shift + 1, c0 + col_shift : c1 + col_shift + 1] = color

    strip_left = int(rng.integers(0, width - strip_w + 1))
    col = strip_left
    for idx, color in enumerate(order_bottom_to_top):
        grid[-1, col] = color
        col += 1
        if idx < len(strip_gaps):
            col += strip_gaps[idx]

    return grid


def _expected_output(
    rects_top_to_bottom: list[tuple[int, int, int, int]],
    colors_top_to_bottom: list[int],
    bg: int,
) -> np.ndarray:
    return _pack_output(list(reversed(rects_top_to_bottom)), list(reversed(colors_top_to_bottom)), bg)


def _bg_is_most_common(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = count_map.get(bg, 0)
    return bg_count > max(count for color, count in count_map.items() if color != bg)


def _pad_for_bg_majority(grid: np.ndarray, bg: int) -> np.ndarray:
    while not _bg_is_most_common(grid, bg):
        grid = np.pad(grid, ((0, 0), (0, 1)), constant_values=bg)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rects_top_to_bottom = _build_nested_rectangles(rng)
    if not rects_top_to_bottom:
        return None

    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    colors_top_to_bottom = list(map(int, rng.choice(palette, size=len(rects_top_to_bottom), replace=False)))

    grid = _render_input(rects_top_to_bottom, colors_top_to_bottom, bg, rng)
    grid = _pad_for_bg_majority(grid, bg)
    expected = _expected_output(rects_top_to_bottom, colors_top_to_bottom, bg)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
