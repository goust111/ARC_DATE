import numpy as np


BLACK = 0
GREY = 5
SOURCE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _frame_bbox(grey_mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grey_mask)
    assert coords.size > 0, "expected a grey frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expected_frame_mask(shape: tuple[int, int], frame: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    grey_mask = g == GREY
    frame = _frame_bbox(grey_mask)
    r0, c0, r1, c1 = frame
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side >= 3, "grey object must be a square frame"

    expected_grey = _expected_frame_mask(g.shape, frame)
    assert np.array_equal(grey_mask, expected_grey), "grey pixels must form exactly one hollow square border"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "frame interior must be black in the input"

    sources = []
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        color = int(g[r, c])
        assert int(np.sum(g == color)) == 1, "each source color must appear exactly once"
        sources.append((int(r), int(c), color))

    assert sources, "expected at least one colored source pixel"
    rows = [r for r, _c, _color in sources]
    cols = [c for _r, c, _color in sources]
    colors = [color for _r, _c, color in sources]
    assert len(rows) == len(set(rows)), "source rows must be distinct"
    assert len(cols) == len(set(cols)), "source columns must be distinct"
    assert len(colors) == len(set(colors)), "source colors must be distinct"
    assert all(r > r1 for r, _c, _color in sources), "sources must be below the square frame"
    assert all(c not in (c0, c1) for _r, c, _color in sources), "sources cannot share the frame-edge columns"

    max_layers = (side - 1) // 2
    assert len(sources) <= max_layers, "too many sources for the available concentric layers"
    return frame, sources


def _render_output_with_order(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    sources: list[tuple[int, int, int]],
    ordered_colors: list[int],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)

    for r, c, color in sources:
        out[r:, c] = color

    r0, c0, r1, c1 = frame
    side = r1 - r0 + 1
    rr = np.arange(side)[:, None]
    cc = np.arange(side)[None, :]
    depth = np.minimum(np.minimum(rr, side - 1 - rr), np.minimum(cc, side - 1 - cc))

    patch = np.zeros((side, side), dtype=int)
    patch[depth == 0] = GREY

    for layer, color in enumerate(ordered_colors, start=1):
        patch[depth == layer] = color

    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _render_output(shape: tuple[int, int], frame: tuple[int, int, int, int], sources: list[tuple[int, int, int]]) -> np.ndarray:
    ordered_colors = [color for r, _c, color in sorted(sources, key=lambda item: item[0])]
    return _render_output_with_order(shape, frame, sources, ordered_colors)


def _columns_not_monotone(cols_in_height_order: list[int]) -> bool:
    inc = all(a < b for a, b in zip(cols_in_height_order, cols_in_height_order[1:]))
    dec = all(a > b for a, b in zip(cols_in_height_order, cols_in_height_order[1:]))
    return not (inc or dec)


def _semantic_generator_checks(
    inp: np.ndarray,
    frame: tuple[int, int, int, int],
    sources: list[tuple[int, int, int]],
) -> bool:
    try:
        parsed_frame, parsed_sources = _parse_input(inp)
    except AssertionError:
        return False

    if parsed_frame != frame or sorted(parsed_sources) != sorted(sources):
        return False

    r0, c0, r1, c1 = frame
    ordered = sorted(sources, key=lambda item: item[0])
    ordered_cols = [c for _r, c, _color in ordered]
    if not _columns_not_monotone(ordered_cols):
        return False

    inside = [c for _r, c, _color in sources if c0 < c < c1]
    outside = [c for _r, c, _color in sources if c < c0 or c > c1]
    if not inside or not outside:
        return False

    if min(r for r, _c, _color in sources) > inp.shape[0] - 3:
        return False

    correct = _render_output(inp.shape, frame, sources)
    by_column = _render_output_with_order(
        inp.shape,
        frame,
        sources,
        [color for _r, _c, color in sorted(sources, key=lambda item: item[1])],
    )
    if np.array_equal(correct, by_column):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend lower source pixels downward and use their ray-height order to color concentric square layers inside the grey frame.

    Input structure:
    - The grid has black background, one hollow grey square frame, and several distinct non-grey source pixels below that frame.
    - Each source is a single colored cell in its own row and column, and no source shares a column with the frame's left or right border.
    - The frame interior is black in the input, and all source pixels lie strictly below the frame.

    Rule:
    - Copy the grid, then extend every colored source straight downward in its own column to the bottom edge.
    - The height of a source ray is the number of rows from that source down to the bottom, so higher source pixels produce longer rays.
    - Sort the source colors by ray height from largest to smallest, keep the grey outer border unchanged, and fill the frame bbox by concentric square layers from outside to inside with that ordered color sequence.
    - If the ordered colors run out before reaching the center, leave the remaining inner layers black; keep all vertical rays visible in the output.

    Key insight:
    - The colored pixels below the frame are not used by position alone: once extended into downward rays, their relative lengths determine the inside-to-center ring order while the outer grey frame stays fixed.
    """
    g = np.asarray(grid, dtype=int)
    frame, sources = _parse_input(g)
    return _render_output(g.shape, frame, sources)


def _choose_frame_and_canvas(rng: np.random.Generator, source_count: int) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    side = int(rng.choice(np.array([9, 9, 11, 11], dtype=int)))
    top = int(rng.integers(1, 4 if side == 9 else 3))

    min_h = max(15, top + side + 1 + source_count)
    max_h = min(20, min_h + 2)
    if min_h > max_h:
        raise AssertionError("no valid canvas height")
    h = int(rng.integers(min_h, max_h + 1))

    min_w = max(15, side + 4)
    max_w = min(19, side + 7)
    if min_w > max_w:
        raise AssertionError("no valid canvas width")
    w = int(rng.integers(min_w, max_w + 1))

    left_min = 1
    left_max = w - side - 1
    if left_min > left_max:
        raise AssertionError("no valid horizontal padding")
    left = int(rng.integers(left_min, left_max + 1))
    return (h, w), (top, left, top + side - 1, left + side - 1)


def _choose_sources(
    rng: np.random.Generator,
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    source_count: int,
) -> list[tuple[int, int, int]] | None:
    h, w = shape
    r0, c0, r1, c1 = frame

    available_rows = list(range(r1 + 2, h))
    if len(available_rows) < source_count:
        return None
    rows = sorted(int(r) for r in rng.choice(np.array(available_rows, dtype=int), size=source_count, replace=False))

    inside_cols = [c for c in range(c0 + 1, c1) if 0 <= c < w]
    outside_cols = [c for c in range(w) if c < c0 or c > c1]
    allowed_cols = [c for c in range(w) if c not in (c0, c1)]
    if len(allowed_cols) < source_count or not inside_cols or not outside_cols:
        return None

    mandatory = {int(rng.choice(np.array(inside_cols, dtype=int))), int(rng.choice(np.array(outside_cols, dtype=int)))}
    remaining_pool = [c for c in allowed_cols if c not in mandatory]
    if len(mandatory) > source_count or len(remaining_pool) < source_count - len(mandatory):
        return None

    extra = []
    if source_count > len(mandatory):
        extra = [int(c) for c in rng.choice(np.array(remaining_pool, dtype=int), size=source_count - len(mandatory), replace=False)]
    cols_by_height = list(mandatory) + extra
    rng.shuffle(cols_by_height)
    if not _columns_not_monotone(cols_by_height):
        return None

    colors = [int(c) for c in rng.choice(np.array(SOURCE_PALETTE, dtype=int), size=source_count, replace=False)]
    rng.shuffle(colors)
    return [(rows[i], cols_by_height[i], colors[i]) for i in range(source_count)]


def _draw_input(shape: tuple[int, int], frame: tuple[int, int, int, int], sources: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    inp[_expected_frame_mask(shape, frame)] = GREY
    for r, c, color in sources:
        inp[r, c] = color
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    source_count = 4 if rng.random() < 0.6 else 3
    shape, frame = _choose_frame_and_canvas(rng, source_count)
    sources = _choose_sources(rng, shape, frame, source_count)
    if sources is None:
        return None

    inp = _draw_input(shape, frame, sources)
    if not _semantic_generator_checks(inp, frame, sources):
        return None

    expected = _render_output(shape, frame, sources)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid ARC input")
