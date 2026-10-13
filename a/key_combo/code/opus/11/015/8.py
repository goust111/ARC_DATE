import numpy as np


MARKER = 5
BACKGROUND = 0
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
SIZE_OPTIONS = (9, 11, 13, 15)
MARKER_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
GEN_ATTEMPTS = 400
ORACLE_PRIORITY_PROB = 0.7


def _marker_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.size > 0, "expected a marker rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == MARKER), "marker cells must form a solid rectangle"
    assert coords.shape[0] == patch.size, "marker must be a single rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _restore_grid(grid: np.ndarray) -> np.ndarray:
    restored = np.array(grid, dtype=int, copy=True)
    source = np.array(grid, dtype=int, copy=False)
    height, width = source.shape
    r0, c0, r1, c1 = _marker_bbox(source)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            reflected = (
                (r, width - 1 - c),
                (height - 1 - r, c),
                (height - 1 - r, width - 1 - c),
            )
            value = None
            for rr, cc in reflected:
                if int(source[rr, cc]) != MARKER:
                    value = int(source[rr, cc])
                    break
            assert value is not None, "each masked cell must have a visible mirror value"
            restored[r, c] = value

    return restored


def _run_length_column(values: np.ndarray) -> np.ndarray:
    segments: list[tuple[int, int]] = []
    for value in map(int, values):
        if not segments or value != segments[-1][0]:
            segments.append((value, 1))
        else:
            color, count = segments[-1]
            segments[-1] = (color, count + 1)

    out = np.empty((sum(count for _color, count in segments), 1), dtype=int)
    cursor = 0
    for color, count in segments:
        out[cursor : cursor + count, 0] = color
        cursor += count
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the masked bilateral pattern, keep its center column, and emit that column as a 1-cell-wide summary strip.

    Input structure:
    - The grid has horizontal and vertical mirror symmetry except for one solid grey rectangle.
    - Grey (5) appears only as that occluding rectangle, and the rectangle overlaps the middle column.
    - The hidden pattern uses background plus several non-grey colors.

    Rule:
    - Find the grey rectangle and restore each covered cell from its left-right, up-down, or 180-degree mirror counterpart.
    - When both the horizontal and vertical mirrored cells are visible, prefer the horizontal one.
    - After restoration, extract the full middle column and run-length compress it into a 1-column output by stacking each color segment with its original length.

    Key insight:
    - The output is based on the restored middle column, not the occluded one; the grey cells are mask markers, and the 1-column run-length summary has the same vertical segment lengths as the true center column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height > 0 and width > 0 and width % 2 == 1

    restored = _restore_grid(g)
    middle = restored[:, width // 2]
    return _run_length_column(middle)


def _paint_centered_rect(grid: np.ndarray, color: int, inset_r: int, inset_c: int) -> None:
    height, width = grid.shape
    grid[inset_r : height - inset_r, inset_c : width - inset_c] = int(color)


def _sample_insets(limit: int, count: int, rng: np.random.Generator) -> list[int]:
    for _ in range(80):
        picks = sorted(int(x) for x in rng.choice(np.arange(1, limit + 1), size=count, replace=False))
        if picks[0] <= 2:
            return picks
    return sorted(int(x) for x in rng.choice(np.arange(1, limit + 1), size=count, replace=False))


def _build_pattern(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    max_layers = min(5, height // 2, width // 2)
    min_layers = min(3, max_layers)
    layer_count = int(rng.integers(min_layers, max_layers + 1))
    colors = [int(c) for c in rng.choice(PALETTE, size=layer_count, replace=False)]
    row_insets = _sample_insets(height // 2, layer_count, rng)
    col_insets = _sample_insets(width // 2, layer_count, rng)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for color, inset_r, inset_c in zip(colors, row_insets, col_insets):
        _paint_centered_rect(grid, color, inset_r, inset_c)

    return grid


def _mask_stats(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
    r0: int,
    c0: int,
) -> tuple[bool, bool]:
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    has_horizontal = False
    has_both = False

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            h_vis = not (r0 <= r <= r1 and c0 <= width - 1 - c <= c1)
            v_vis = not (r0 <= height - 1 - r <= r1 and c0 <= c <= c1)
            p_vis = not (r0 <= height - 1 - r <= r1 and c0 <= width - 1 - c <= c1)
            if not (h_vis or v_vis or p_vis):
                return False, False
            has_horizontal |= h_vis
            has_both |= h_vis and v_vis

    return has_horizontal, has_both


def _valid_mask_positions(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mid_r = height // 2
    mid_c = width // 2
    priority: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []

    for r0 in range(height - rect_h + 1):
        if r0 <= mid_r <= r0 + rect_h - 1:
            continue
        for c0 in range(width - rect_w + 1):
            if not (c0 <= mid_c <= c0 + rect_w - 1):
                continue
            has_horizontal, has_both = _mask_stats(height, width, rect_h, rect_w, r0, c0)
            if not has_horizontal:
                continue
            if has_both:
                priority.append((r0, c0))
            else:
                fallback.append((r0, c0))

    return priority, fallback


def _count_runs(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return 1 + int(np.count_nonzero(values[1:] != values[:-1]))


def _attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(SIZE_OPTIONS))
    width = int(rng.choice(SIZE_OPTIONS))
    grid = _build_pattern(height, width, rng)
    middle = grid[:, width // 2]

    if _count_runs(middle) < 3:
        return None
    non_bg = {int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, MARKER)}
    if not (3 <= len(non_bg) <= 5):
        return None

    rect_h, rect_w = MARKER_SIZES[int(rng.integers(0, len(MARKER_SIZES)))]
    priority, fallback = _valid_mask_positions(height, width, rect_h, rect_w)
    if not priority and not fallback:
        return None

    if priority and (not fallback or rng.random() < ORACLE_PRIORITY_PROB):
        r0, c0 = priority[int(rng.integers(0, len(priority)))]
    else:
        choices = fallback or priority
        r0, c0 = choices[int(rng.integers(0, len(choices)))]

    inp = grid.copy()
    inp[r0 : r0 + rect_h, c0 : c0 + rect_w] = MARKER
    expected = grid[:, width // 2].reshape(height, 1)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(inp == MARKER) != rect_h * rect_w:
        return None
    if np.count_nonzero(inp[:, width // 2] == MARKER) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
