import numpy as np


MAX_GENERATION_ATTEMPTS = 300
MIN_PANELS = 2
MAX_PANELS = 5
SEPARATOR_THICKNESS_CHOICES = (1, 2)
MAX_GRID_HEIGHT = 28
MAX_GRID_WIDTH = 28
SELECTOR_WIDTH_RANGE = (4, 9)
CONTENT_SIDE_RANGE = (2, 7)
PANEL_MARGIN_RANGE = (1, 4)
PANEL_EXTRA_WIDTH_RANGE = (0, 3)
PANEL_EXTRA_HEIGHT_RANGE = (0, 3)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).ravel()
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _mode_color(arr: np.ndarray) -> int:
    flat = np.asarray(arr, dtype=int).ravel()
    counts = np.bincount(flat)
    return int(np.argmax(counts))


def _background_color(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=int)
    corners = {
        int(arr[0, 0]),
        int(arr[0, -1]),
        int(arr[-1, 0]),
        int(arr[-1, -1]),
    }
    if len(corners) == 1:
        return next(iter(corners))
    return _mode_color(arr)


def _crop_non_background(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(arr, dtype=int) != int(bg))
    assert len(coords) > 0, "expected at least one non-background cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return np.asarray(arr[r0:r1, c0:c1], dtype=int)


def _panel_spans(right: np.ndarray, sep_color: int) -> list[tuple[int, int]]:
    row_is_separator = np.all(np.asarray(right, dtype=int) == int(sep_color), axis=1)
    return _runs(~row_is_separator)


def _find_layout(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []

    for color in map(int, np.unique(g)):
        full_columns = np.all(g == color, axis=0)
        for c0, c1 in _runs(full_columns):
            if c0 == 0 or c1 == w:
                continue
            left = g[:, :c0]
            right = g[:, c1:]
            if left.size == 0 or right.size == 0:
                continue
            if np.any(left == color):
                continue
            if len(np.unique(left)) != 2:
                continue
            spans = _panel_spans(right, color)
            if len(spans) < 2:
                continue
            if not np.any(np.all(right == color, axis=1)):
                continue
            if any(r0 >= r1 for r0, r1 in spans):
                continue
            candidates.append((color, c0, c1, spans))

    assert len(candidates) == 1, "expected a unique vertical separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count marker pixels in the selector and output the corresponding candidate panel.

    Input structure:
    - A full-height separator color splits the grid into a left selector and a right candidate area.
    - The selector contains only its background plus scattered marker pixels of one color.
    - The candidate area contains stacked panels separated by full-width separator rows.

    Rule:
    - Count the selector marker pixels.
    - Use that count as a 1-based index into the candidate panels from top to bottom.
    - Return the selected panel tightly cropped to its non-background content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, split_start, split_end, spans = _find_layout(g)
    selector = g[:, :split_start]
    selector_bg = _background_color(selector)
    index = int(np.count_nonzero(selector != selector_bg))

    right = g[:, split_end:]
    panels = [right[r0:r1, :] for r0, r1 in spans]
    assert 1 <= index <= len(panels), "selector count must pick an existing panel"

    chosen = np.asarray(panels[index - 1], dtype=int)
    panel_bg = _background_color(chosen)
    return _crop_non_background(chosen, panel_bg)


def _pick_colors(rng: np.random.Generator, count: int, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else {int(x) for x in exclude}
    pool = [color for color in range(10) if color not in banned]
    picked = rng.choice(pool, size=count, replace=False)
    return [int(x) for x in picked]


def _orient(rng: np.random.Generator, arr: np.ndarray) -> np.ndarray:
    out = np.rot90(np.asarray(arr, dtype=int), int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _family_solid(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(CONTENT_SIDE_RANGE[0], CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(CONTENT_SIDE_RANGE[0], CONTENT_SIDE_RANGE[1]))
    return np.full((h, w), colors[0], dtype=int)


def _family_frame(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((h, w), bg, dtype=int)
    arr[0, :] = colors[0]
    arr[-1, :] = colors[0]
    arr[:, 0] = colors[0]
    arr[:, -1] = colors[0]
    if h >= 5 and w >= 5 and bool(rng.integers(0, 2)):
        arr[1:-1, 1:-1] = colors[1]
        arr[2:-2, 2:-2] = bg
    else:
        arr[h // 2, w // 2] = colors[1]
    return arr


def _family_plus(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((h, w), bg, dtype=int)
    center_r = h // 2
    center_c = w // 2
    arr[center_r, :] = colors[0]
    arr[:, center_c] = colors[0]
    if bool(rng.integers(0, 2)):
        arr[center_r, center_c] = colors[1]
    return arr


def _family_l(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((h, w), bg, dtype=int)
    arr[-1, :] = colors[0]
    arr[:, 0] = colors[0]
    if bool(rng.integers(0, 2)):
        arr[0, 0] = colors[1]
    return arr


def _family_diagonals(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    n = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((n, n), bg, dtype=int)
    for i in range(n):
        arr[i, i] = colors[0]
        if bool(rng.integers(0, 2)) or i != n // 2:
            arr[i, n - 1 - i] = colors[1]
    return arr


def _family_bars(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((h, w), bg, dtype=int)
    if bool(rng.integers(0, 2)):
        for idx, row in enumerate(range(0, h, 2)):
            arr[row, :] = colors[idx % 2]
    else:
        for idx, col in enumerate(range(0, w, 2)):
            arr[:, col] = colors[idx % 2]
    return arr


def _family_blocks(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    h = int(rng.integers(4, CONTENT_SIDE_RANGE[1]))
    w = int(rng.integers(4, CONTENT_SIDE_RANGE[1]))
    arr = np.full((h, w), bg, dtype=int)
    h1 = int(rng.integers(1, max(2, h - 1)))
    w1 = int(rng.integers(1, max(2, w - 1)))
    h2 = int(rng.integers(1, max(2, h - 1)))
    w2 = int(rng.integers(1, max(2, w - 1)))
    arr[:h1, :w1] = colors[0]
    arr[-h2:, -w2:] = colors[1]
    return arr


def _family_stairs(rng: np.random.Generator, bg: int, colors: list[int]) -> np.ndarray:
    n = int(rng.integers(3, CONTENT_SIDE_RANGE[1]))
    arr = np.full((n, n), bg, dtype=int)
    for i in range(n):
        arr[i, : i + 1] = colors[0]
        if bool(rng.integers(0, 2)):
            arr[i, i] = colors[1]
    return arr


def _sample_content(rng: np.random.Generator, sep_color: int) -> tuple[np.ndarray, int]:
    bg = _pick_colors(rng, 1, exclude={sep_color})[0]
    colors = _pick_colors(rng, 3, exclude={sep_color, bg})
    families = [
        _family_solid,
        _family_frame,
        _family_plus,
        _family_l,
        _family_diagonals,
        _family_bars,
        _family_blocks,
        _family_stairs,
    ]
    family = families[int(rng.integers(0, len(families)))]
    content = _orient(rng, family(rng, bg, colors))
    cropped = _crop_non_background(content, bg)
    return cropped, bg


def _content_signature(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    data = np.asarray(arr, dtype=int)
    return tuple(map(int, data.shape)), tuple(map(int, data.ravel()))


def _embed_panel(
    rng: np.random.Generator,
    content: np.ndarray,
    bg: int,
    panel_height: int,
    panel_width: int,
    min_top: int,
    min_left: int,
) -> np.ndarray:
    content = np.asarray(content, dtype=int)
    ch, cw = map(int, content.shape)
    panel = np.full((panel_height, panel_width), int(bg), dtype=int)

    spare_h = int(panel_height - ch - min_top - 1)
    spare_w = int(panel_width - cw - min_left - 1)
    assert spare_h >= 0 and spare_w >= 0

    top = int(min_top + rng.integers(0, spare_h + 1))
    left = int(min_left + rng.integers(0, spare_w + 1))
    panel[top : top + ch, left : left + cw] = content
    return panel


def _build_right_area(
    rng: np.random.Generator,
    contents: list[np.ndarray],
    panel_bgs: list[int],
    sep_color: int,
) -> np.ndarray | None:
    panel_count = len(contents)
    sep_h = int(rng.choice(SEPARATOR_THICKNESS_CHOICES))
    top_border = bool(rng.integers(0, 2))
    bottom_border = bool(rng.integers(0, 2))

    top_margins: list[int] = []
    bottom_margins: list[int] = []
    left_margins: list[int] = []
    right_margins: list[int] = []
    panel_heights: list[int] = []
    width_needs: list[int] = []

    for content in contents:
        top = int(rng.integers(PANEL_MARGIN_RANGE[0], PANEL_MARGIN_RANGE[1]))
        bottom = int(rng.integers(PANEL_MARGIN_RANGE[0], PANEL_MARGIN_RANGE[1]))
        left = int(rng.integers(PANEL_MARGIN_RANGE[0], PANEL_MARGIN_RANGE[1]))
        right = int(rng.integers(PANEL_MARGIN_RANGE[0], PANEL_MARGIN_RANGE[1]))
        extra_h = int(rng.integers(PANEL_EXTRA_HEIGHT_RANGE[0], PANEL_EXTRA_HEIGHT_RANGE[1]))
        extra_w = int(rng.integers(PANEL_EXTRA_WIDTH_RANGE[0], PANEL_EXTRA_WIDTH_RANGE[1]))
        ch, cw = map(int, content.shape)

        top_margins.append(top)
        bottom_margins.append(bottom)
        left_margins.append(left)
        right_margins.append(right)
        panel_heights.append(ch + top + bottom + extra_h)
        width_needs.append(cw + left + right + extra_w)

    panel_width = int(max(width_needs))
    total_height = int(sum(panel_heights) + sep_h * (panel_count - 1))
    if top_border:
        total_height += sep_h
    if bottom_border:
        total_height += sep_h

    if total_height > MAX_GRID_HEIGHT or panel_width > MAX_GRID_WIDTH:
        return None

    sep_row = np.full((sep_h, panel_width), int(sep_color), dtype=int)
    rows: list[np.ndarray] = []
    if top_border:
        rows.append(sep_row.copy())

    for idx, (content, bg) in enumerate(zip(contents, panel_bgs)):
        panel = _embed_panel(
            rng,
            content,
            bg,
            panel_heights[idx],
            panel_width,
            top_margins[idx],
            left_margins[idx],
        )
        rows.append(panel)
        if idx != panel_count - 1:
            rows.append(sep_row.copy())

    if bottom_border:
        rows.append(sep_row.copy())

    return np.vstack(rows)


def _make_selector(
    rng: np.random.Generator,
    height: int,
    width: int,
    bg: int,
    marker: int,
    count: int,
) -> np.ndarray | None:
    selector = np.full((height, width), int(bg), dtype=int)
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    ]
    if len(candidates) < count:
        return None
    picks = rng.choice(len(candidates), size=count, replace=False)
    for idx in np.asarray(picks, dtype=int):
        r, c = candidates[int(idx)]
        selector[r, c] = int(marker)
    return selector


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_count = int(rng.integers(MIN_PANELS, MAX_PANELS + 1))
    target_index = int(rng.integers(1, panel_count + 1))
    sep_color, selector_bg, marker_color = _pick_colors(rng, 3)

    contents: list[np.ndarray] = []
    panel_bgs: list[int] = []
    signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()

    for _ in range(panel_count):
        for _retry in range(40):
            content, bg = _sample_content(rng, sep_color)
            sig = _content_signature(content)
            if sig in signatures:
                continue
            signatures.add(sig)
            contents.append(content)
            panel_bgs.append(bg)
            break
        else:
            return None

    right = _build_right_area(rng, contents, panel_bgs, sep_color)
    if right is None:
        return None

    height = int(right.shape[0])
    sep_w = int(rng.choice(SEPARATOR_THICKNESS_CHOICES))
    selector_width = int(rng.integers(SELECTOR_WIDTH_RANGE[0], SELECTOR_WIDTH_RANGE[1]))
    while selector_width * height < target_index + 4:
        selector_width += 1
    if selector_width + sep_w + right.shape[1] > MAX_GRID_WIDTH:
        return None

    selector = _make_selector(rng, height, selector_width, selector_bg, marker_color, target_index)
    if selector is None:
        return None

    divider = np.full((height, sep_w), int(sep_color), dtype=int)
    grid = np.hstack([selector, divider, right])

    expected = contents[target_index - 1]
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
