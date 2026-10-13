import numpy as np

from arc_utils import bbox, colors_present, get_bg
import itertools
import random


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _inset(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _side_matches(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return sum((a[0] == b[0], a[1] == b[1], a[2] == b[2], a[3] == b[3]))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete sparse nested rectangular color layers into solid rectangles.

    Input structure
    The background is the dominant color, and every foreground color marks one
    axis-aligned rectangular layer; the layer bboxes have unique areas and are
    nested from outermost to innermost.
    Rule
    Take each color's bbox as its completed rectangle, except that a child bbox
    which already matches at least three sides of its parent's one-pixel inset
    expands to that full inset; repaint the rectangles largest to smallest.
    Key insight
    The scattered pixels are samples of nested rectangular regions, so the
    output is a clean redraw of those regions with inner colors overwriting
    outer colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    layers: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        rect = bbox(g == color)
        assert rect is not None
        layers.append({"color": color, "raw": rect, "rect": rect, "area": _area(rect)})

    assert layers
    areas = [layer["area"] for layer in layers]
    assert len(set(areas)) == len(areas)
    layers.sort(key=lambda layer: int(layer["area"]), reverse=True)

    completed: list[dict[str, int | tuple[int, int, int, int]]] = []
    for layer in layers:
        raw = layer["raw"]
        assert isinstance(raw, tuple)
        if completed:
            parents = [
                parent
                for parent in completed
                if _contains(parent["rect"], raw) and parent["rect"] != raw
            ]
            assert parents
            parent = min(parents, key=lambda item: _area(item["rect"]))
            parent_inset = _inset(parent["rect"])
            assert parent_inset[0] <= parent_inset[2] and parent_inset[1] <= parent_inset[3]
            assert _contains(parent_inset, raw)
            matches = _side_matches(raw, parent_inset)
            assert matches in (0, 3, 4)
            if matches >= 3:
                layer["rect"] = parent_inset
        completed.append(layer)

    out = np.full_like(g, bg)
    for layer in layers:
        rect = layer["rect"]
        assert isinstance(rect, tuple)
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = int(layer["color"])

    assert np.all(out[g != bg] == g[g != bg])
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and layer topology.
GEN_GRID_SIZES = (10, 12)
GEN_LAYER_COUNTS = (2, 3)
GEN_OUTER_INSETS = (0, 1)
GEN_GAP_SIZES = (1, 2)
GEN_MIN_INNER_SIDE = 4

# Sparse sampling bounds and bbox-preservation controls.
GEN_MIN_SAMPLE_COUNT = 4
GEN_MIN_SAMPLE_PERCENT = 28
GEN_MAX_SAMPLE_PERCENT = 71
GEN_PERCENT_DENOMINATOR = 100
GEN_PERCENT_CEIL_OFFSET = GEN_PERCENT_DENOMINATOR - 1
GEN_SIDES = ("top", "bottom", "left", "right")
GEN_OMITTED_SIDE_CHOICES = (None, None, "top", "bottom", "left", "right")

# Color and retry controls.
GEN_COLOR_RANGE = range(1, 10)
GEN_RETRY_LIMIT = 300

GenRect = tuple[int, int, int, int]


def _rect_mask(shape: tuple[int, int], rect: GenRect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _gen_inset(rect: GenRect, amount: int) -> GenRect:
    r0, c0, r1, c1 = rect
    return r0 + amount, c0 + amount, r1 - amount, c1 - amount


def _target_rect(rect: GenRect, omitted_side: str | None) -> GenRect:
    r0, c0, r1, c1 = rect
    if omitted_side == "top":
        r0 += 1
    elif omitted_side == "bottom":
        r1 -= 1
    elif omitted_side == "left":
        c0 += 1
    elif omitted_side == "right":
        c1 -= 1
    return r0, c0, r1, c1


def _available_mask(visible: np.ndarray, rect: GenRect, omitted_side: str | None) -> np.ndarray:
    allowed = visible.copy()
    r0, c0, r1, c1 = rect
    if omitted_side == "top":
        allowed[r0, c0 : c1 + 1] = False
    elif omitted_side == "bottom":
        allowed[r1, c0 : c1 + 1] = False
    elif omitted_side == "left":
        allowed[r0 : r1 + 1, c0] = False
    elif omitted_side == "right":
        allowed[r0 : r1 + 1, c1] = False
    return allowed


def _side_candidates(allowed: np.ndarray, target: GenRect, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = target
    if side == "top":
        return [(r0, int(c)) for c in np.flatnonzero(allowed[r0, :])]
    elif side == "bottom":
        return [(r1, int(c)) for c in np.flatnonzero(allowed[r1, :])]
    elif side == "left":
        return [(int(r), c0) for r in np.flatnonzero(allowed[:, c0])]
    return [(int(r), c1) for r in np.flatnonzero(allowed[:, c1])]


def _sample_visible(visible: np.ndarray, rect: GenRect, omitted_side: str | None, count: int) -> np.ndarray | None:
    """Return an exact-count sample mask with every target bbox side represented."""
    allowed = _available_mask(visible, rect, omitted_side)
    target = _target_rect(rect, omitted_side)

    anchors: set[tuple[int, int]] = set()
    for side in GEN_SIDES:
        choices = _side_candidates(allowed, target, side)
        if not choices:
            return None
        anchors.add(random.choice(choices))

    if len(anchors) > count:
        return None

    pool = [tuple(map(int, p)) for p in np.argwhere(allowed)]
    rest = [p for p in pool if p not in anchors]
    random.shuffle(rest)
    selected = list(anchors) + rest[: count - len(anchors)]

    if len(selected) != count:
        return None

    mask = np.zeros(visible.shape, dtype=bool)
    for r, c in selected:
        mask[r, c] = True
    return mask


def _count_options(area: int, allowed_count: int) -> list[int]:
    min_by_rate = (GEN_MIN_SAMPLE_PERCENT * area + GEN_PERCENT_CEIL_OFFSET) // GEN_PERCENT_DENOMINATOR
    lo = max(GEN_MIN_SAMPLE_COUNT, min_by_rate)
    hi = min(allowed_count, (GEN_MAX_SAMPLE_PERCENT * area) // GEN_PERCENT_DENOMINATOR)
    return list(range(lo, hi + 1))


def _choose_sample_counts(n: int, areas: list[int], allowed_counts: list[int]) -> tuple[int, ...] | None:
    """Pick per-layer sample counts while keeping background the dominant color."""
    options = [_count_options(a, b) for a, b in zip(areas, allowed_counts)]
    if any(not opt for opt in options):
        return None
    valid = [
        counts
        for counts in itertools.product(*options)
        if n * n - sum(counts) > max(counts)
    ]
    if not valid:
        return None
    return random.choice(valid)


def _choose_layout() -> tuple[int, list[GenRect], tuple[int, ...]] | None:
    """Choose the canvas and nested square rects before color or sampling choices."""
    n = random.choice(GEN_GRID_SIZES)
    layer_count = random.choice(GEN_LAYER_COUNTS)
    outer_inset = random.choice(GEN_OUTER_INSETS)
    outer_side = n - 2 * outer_inset

    gap_options = [
        gaps
        for gaps in itertools.product(GEN_GAP_SIZES, repeat=layer_count - 1)
        if outer_side - 2 * sum(gaps) >= GEN_MIN_INNER_SIDE
    ]
    if not gap_options:
        return None
    gaps = random.choice(gap_options)

    rects = [(outer_inset, outer_inset, n - outer_inset - 1, n - outer_inset - 1)]
    for gap in gaps:
        rects.append(_gen_inset(rects[-1], gap))

    return n, rects, gaps


def _render_expected(n: int, colors: list[int], rects: list[GenRect]) -> np.ndarray:
    expected = np.zeros((n, n), dtype=int)
    for color, rect in zip(colors, rects):
        r0, c0, r1, c1 = rect
        expected[r0 : r1 + 1, c0 : c1 + 1] = color
    return expected


def _prepare_sampling_plan(n: int, rects: list[GenRect], gaps: tuple[int, ...]):
    """Build per-layer masks, side omissions, ring areas, and allowed counts."""
    visible_masks: list[np.ndarray] = []
    omitted_sides: list[str | None] = []
    target_areas: list[int] = []
    allowed_counts: list[int] = []
    for i, rect in enumerate(rects):
        visible = _rect_mask((n, n), rect)
        if i + 1 < len(rects):
            visible &= ~_rect_mask((n, n), rects[i + 1])
        omitted: str | None = None
        if i > 0 and gaps[i - 1] == 1:
            omitted = random.choice(GEN_OMITTED_SIDE_CHOICES)
        allowed = _available_mask(visible, rect, omitted)
        target = _target_rect(rect, omitted)
        visible_masks.append(visible)
        omitted_sides.append(omitted)
        target_areas.append(_area(target))
        allowed_counts.append(int(allowed.sum()))

    if len(set(target_areas)) != len(target_areas):
        return None

    ring_areas = [int(mask.sum()) for mask in visible_masks]
    return visible_masks, omitted_sides, ring_areas, allowed_counts


def _draw_sparse_input(n, colors, rects, visible_masks, omitted_sides, counts):
    inp = np.zeros((n, n), dtype=int)
    for color, rect, visible, omitted, count in zip(
        colors, rects, visible_masks, omitted_sides, counts
    ):
        sampled = _sample_visible(visible, rect, omitted, count)
        if sampled is None:
            return None
        inp[sampled] = color
    return inp


def _passes_solver_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    n, rects, gaps = layout

    colors = random.sample(GEN_COLOR_RANGE, len(rects))
    expected = _render_expected(n, colors, rects)

    sampling_plan = _prepare_sampling_plan(n, rects, gaps)
    if sampling_plan is None:
        return None
    visible_masks, omitted_sides, ring_areas, allowed_counts = sampling_plan

    counts = _choose_sample_counts(n, ring_areas, allowed_counts)
    if counts is None:
        return None

    inp = _draw_sparse_input(n, colors, rects, visible_masks, omitted_sides, counts)
    if inp is None:
        return None

    if not _passes_solver_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
