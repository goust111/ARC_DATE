import numpy as np


BLACK = 0
RED = 2
GREY = 5


BG_CHOICES = (1, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (15, 25)
LEFT_WIDTH_RANGE = (8, 15)
RIGHT_WIDTH_RANGE = (4, 9)
COMPONENT_COUNT_RANGE = (1, 5)
MIN_ROW_GAP = 1
FLUSH_PROB = 0.25
GENERATE_RETRIES = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    sep = int(cols[0])
    assert 0 < sep < grid.shape[1] - 1
    return sep


def _background_color(grid: np.ndarray) -> int:
    candidates = grid[(grid != BLACK) & (grid != GREY)]
    assert candidates.size > 0, "expected a visible background color distinct from black and grey"
    values, counts = np.unique(candidates, return_counts=True)
    return int(values[np.argmax(counts)])


def _trigger_rows_from_left(left: np.ndarray, bg: int) -> list[int]:
    h, w = left.shape
    if w < 2:
        return []

    trigger_rows: set[int] = set()
    for comp in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        for r, c in comp:
            if c == w - 2 and left[r, w - 1] == BLACK:
                trigger_rows.add(int(r))
    return sorted(trigger_rows)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a grey separator, then emit red rays from
    enclosed holes that reach the separator-facing inner column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    sep = _find_separator_col(g)
    bg = _background_color(g)

    left = g[:, :sep].copy()
    components = _components4(left == BLACK)

    slid_left = left.copy()
    slid_left[slid_left == BLACK] = bg

    target_right = sep - 1
    for comp in components:
        delta = target_right - max(c for _, c in comp)
        assert delta >= 0, "all black components must start on the left side of the separator"
        for r, c in comp:
            slid_left[r, c + delta] = BLACK

    out = g.copy()
    out[:, :sep] = slid_left

    for r in _trigger_rows_from_left(slid_left, bg):
        out[r, sep + 1 :] = RED

    return out


def _scatter(total: int, slots: int, rng: np.random.Generator) -> list[int]:
    out = [0] * slots
    for _ in range(total):
        out[int(rng.integers(0, slots))] += 1
    return out


def _row_starts(height: int, sprite_heights: list[int], rng: np.random.Generator) -> list[int]:
    needed = sum(sprite_heights) + MIN_ROW_GAP * max(0, len(sprite_heights) - 1)
    slack = height - needed
    assert slack >= 0
    extras = _scatter(slack, len(sprite_heights) + 1, rng)
    starts: list[int] = []
    row = extras[0]
    for i, sprite_h in enumerate(sprite_heights):
        starts.append(int(row))
        row += sprite_h
        if i + 1 < len(sprite_heights):
            row += MIN_ROW_GAP + extras[i + 1]
    return starts


def _bar_rows(height: int, rng: np.random.Generator) -> list[int]:
    candidates = list(range(2, height - 2))
    if not candidates:
        return []
    max_bars = min(2, max(0, (height - 4) // 2))
    if max_bars == 0:
        return []
    want = int(rng.integers(0, max_bars + 1))
    rows: list[int] = []
    for row in rng.permutation(candidates):
        row = int(row)
        if all(abs(row - other) > 1 for other in rows):
            rows.append(row)
            if len(rows) == want:
                break
    rows.sort()
    return rows


def _make_emitter(height: int, width: int, rng: np.random.Generator, *, blind: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if blind:
        mask[:, -2] = True
    for row in _bar_rows(height, rng):
        mask[row, :] = True
    return mask


def _make_solid(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    style = int(rng.integers(0, 3))
    if style == 0:
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool)
    start = max(1, (width + 1) // 2)
    widths = [int(rng.integers(start, width + 1))]
    for _ in range(1, height):
        prev = widths[-1]
        low = max(1, prev - 1)
        high = min(width, prev + 1)
        widths.append(int(rng.integers(low, high + 1)))

    if style == 1:
        for r, row_width in enumerate(widths):
            mask[r, :row_width] = True
    else:
        for r, row_width in enumerate(widths):
            mask[r, width - row_width :] = True
    return mask


def _sample_shape(kind: str, left_width: int, rng: np.random.Generator) -> np.ndarray | None:
    if kind == "solid":
        max_width = min(7, left_width)
        if max_width < 2:
            return None
        height = int(rng.integers(2, 7))
        width = int(rng.integers(2, max_width + 1))
        return _make_solid(height, width, rng)

    min_width = 4 if kind == "emitter" else 5
    max_width = min(8, left_width)
    if max_width < min_width:
        return None
    height = int(rng.integers(4, 9))
    width = int(rng.integers(min_width, max_width + 1))
    return _make_emitter(height, width, rng, blind=(kind == "blind"))


def _component_types(count: int, rng: np.random.Generator) -> list[str]:
    kinds = ["emitter"]
    for _ in range(count - 1):
        kinds.append(str(rng.choice(["solid", "solid", "blind", "emitter"])))
    if count > 1 and all(kind == "emitter" for kind in kinds):
        kinds[1] = str(rng.choice(["solid", "blind"]))
    rng.shuffle(kinds)
    return kinds


def _expected_left(height: int, left_width: int, placements: list[tuple[np.ndarray, int]]) -> np.ndarray:
    bg = -1
    left = np.full((height, left_width), bg, dtype=int)
    for mask, row0 in placements:
        sprite_h, sprite_w = mask.shape
        col0 = left_width - sprite_w
        window = left[row0 : row0 + sprite_h, col0 : col0 + sprite_w]
        window[~mask] = bg
        window[mask] = BLACK
    return left


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1]))
    left_width = int(rng.integers(LEFT_WIDTH_RANGE[0], LEFT_WIDTH_RANGE[1]))
    right_width = int(rng.integers(RIGHT_WIDTH_RANGE[0], RIGHT_WIDTH_RANGE[1]))
    width = left_width + 1 + right_width
    bg = int(rng.choice(BG_CHOICES))

    count = int(rng.integers(COMPONENT_COUNT_RANGE[0], COMPONENT_COUNT_RANGE[1]))
    kinds = _component_types(count, rng)
    masks = [_sample_shape(kind, left_width, rng) for kind in kinds]
    if any(mask is None for mask in masks):
        return None

    sprite_heights = [int(mask.shape[0]) for mask in masks if mask is not None]
    needed = sum(sprite_heights) + MIN_ROW_GAP * (count - 1)
    if needed > height:
        return None

    starts = _row_starts(height, sprite_heights, rng)
    grid = np.full((height, width), bg, dtype=int)
    grid[:, left_width] = GREY

    placements: list[tuple[np.ndarray, int]] = []
    for mask, row0 in zip(masks, starts):
        assert mask is not None
        sprite_h, sprite_w = mask.shape
        max_shift = left_width - sprite_w
        if max_shift < 0:
            return None
        if max_shift == 0 or rng.random() < FLUSH_PROB:
            shift = 0
        else:
            shift = int(rng.integers(1, max_shift + 1))
        col0 = left_width - sprite_w - shift

        window = grid[row0 : row0 + sprite_h, col0 : col0 + sprite_w]
        if np.any(window[mask] != bg):
            return None
        window[mask] = BLACK
        placements.append((mask, int(row0)))

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    expected_left = _expected_left(height, left_width, placements)
    expected_left[expected_left == -1] = bg

    trigger_rows = _trigger_rows_from_left(expected_left, bg)
    if not trigger_rows:
        return None

    expected = grid.copy()
    expected[:, :left_width] = expected_left
    for row in trigger_rows:
        expected[row, left_width + 1 :] = RED

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
