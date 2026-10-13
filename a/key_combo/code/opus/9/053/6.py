import random

import numpy as np


BACKGROUND = 0
LEFT_COLOR = 1
RIGHT_COLOR = 2
SOURCE_COLOR = 4

MAX_ATTEMPTS = 500
GRID_ROWS = (12, 20)
GRID_COLS = (14, 24)
RECT_HEIGHTS = (2, 4)
RECT_WIDTHS = (2, 4)
GAP_WIDTHS = (3, 7)
TOP_MARGIN_MIN = 4
BOTTOM_MARGIN = (2, 6)
SIDE_MARGIN = (1, 4)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _single_solid_rect(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    comps = _component_coords(grid == color)
    assert len(comps) == 1, f"expected exactly one component of color {color}"
    box = _bbox(comps[0])
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert len(comps[0]) == h * w, f"color {color} must form a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    return box


def _structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    left_box = _single_solid_rect(g, LEFT_COLOR)
    right_box = _single_solid_rect(g, RIGHT_COLOR)
    lr0, lc0, lr1, lc1 = left_box
    rr0, rc0, rr1, rc1 = right_box

    assert (lr0, lr1) == (rr0, rr1), "rectangles must share a row span"
    assert lc1 < rc0, "blue rectangle must be left of red rectangle"
    assert (lr1 - lr0) == (rr1 - rr0)
    assert (lc1 - lc0) == (rc1 - rc0)

    gap_left = lc1 + 1
    gap_right = rc0 - 1
    gap_width = gap_right - gap_left + 1
    assert GAP_WIDTHS[0] <= gap_width <= GAP_WIDTHS[1]

    split = gap_width // 2
    left_domain = set(range(gap_left, gap_left + split))
    right_domain = set(range(rc0 - split, rc0))
    mid_col = gap_left + split if gap_width % 2 == 1 else None

    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == SOURCE_COLOR)]
    assert len(sources) >= 3
    source_cols = [c for _, c in sources]
    assert len(set(source_cols)) == len(source_cols), "at most one source per column"
    assert all(gap_left <= c <= gap_right for c in source_cols)
    assert all(r <= lr0 - 3 for r, _ in sources), "sources must be well above the rectangles"
    assert any(c in left_domain for c in source_cols)
    assert any(c in right_domain for c in source_cols)
    if mid_col is not None:
        assert mid_col in source_cols

    allowed = {BACKGROUND, LEFT_COLOR, RIGHT_COLOR, SOURCE_COLOR}
    assert set(map(int, np.unique(g))) <= allowed

    return {
        "left_box": left_box,
        "right_box": right_box,
        "top_row": lr0,
        "gap_left": gap_left,
        "gap_right": gap_right,
        "gap_width": gap_width,
        "left_domain": left_domain,
        "right_domain": right_domain,
        "mid_col": mid_col,
        "sources": sources,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor downward rays above a hidden half-gap partition between two aligned rectangles.

    Input structure:
    - Black background contains exactly two same-sized solid rectangles sharing one row span: a blue rectangle on the left and a red rectangle on the right.
    - The columns strictly between the rectangles form a black horizontal gap of width 3 to 7.
    - Several isolated yellow source pixels sit at least three rows above the rectangles, with at most one source per gap column.

    Rule:
    - Infer the gap columns from the two rectangles and split that gap logically into a left half-domain and a right half-domain.
    - If the gap width is odd, the single middle column belongs to neither domain.
    - For each yellow source pixel, draw a vertical ray from the source row to the bottom: it stays yellow above the rectangles, then from the rectangles' top row downward it becomes blue in left-domain columns, red in right-domain columns, and stays yellow in the odd-width middle column.
    - Do not alter the original rectangle pixels.

    Key insight:
    - The gap split is invisible in the input; only the ray recoloring pattern reveals the hidden left/right influence domains, with the odd-width middle column acting as the disambiguating unchanged case.
    """
    g = np.asarray(grid, dtype=int)
    info = _structure(g)
    top_row = info["top_row"]

    out = g.copy()
    for r, c in info["sources"]:
        if c in info["left_domain"]:
            lower_color = LEFT_COLOR
        elif c in info["right_domain"]:
            lower_color = RIGHT_COLOR
        else:
            assert info["mid_col"] == c
            lower_color = SOURCE_COLOR

        out[r:top_row, c] = SOURCE_COLOR
        out[top_row:, c] = lower_color

    lr0, lc0, lr1, lc1 = info["left_box"]
    rr0, rc0, rr1, rc1 = info["right_box"]
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = LEFT_COLOR
    out[rr0 : rr1 + 1, rc0 : rc1 + 1] = RIGHT_COLOR
    return out


def _choose_source_columns(gap_left: int, gap_right: int) -> list[int]:
    cols = list(range(gap_left, gap_right + 1))
    gap_width = len(cols)
    split = gap_width // 2
    left_cols = cols[:split]
    right_cols = cols[-split:]
    required = [random.choice(left_cols), random.choice(right_cols)]
    if gap_width % 2 == 1:
        required.append(cols[split])

    extras = [c for c in cols if c not in required]
    n_sources = random.randint(len(required), len(cols))
    chosen = required + random.sample(extras, n_sources - len(required))
    random.shuffle(chosen)
    return chosen


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    info = _structure(inp)
    top_row = info["top_row"]
    if np.array_equal(inp, out):
        return False

    for r, c in info["sources"]:
        if not np.all(out[r:top_row, c] == SOURCE_COLOR):
            return False
        expected = SOURCE_COLOR
        if c in info["left_domain"]:
            expected = LEFT_COLOR
        elif c in info["right_domain"]:
            expected = RIGHT_COLOR
        if not np.all(out[top_row:, c] == expected):
            return False

    lr0, lc0, lr1, lc1 = info["left_box"]
    rr0, rc0, rr1, rc1 = info["right_box"]
    if not np.all(out[lr0 : lr1 + 1, lc0 : lc1 + 1] == LEFT_COLOR):
        return False
    if not np.all(out[rr0 : rr1 + 1, rc0 : rc1 + 1] == RIGHT_COLOR):
        return False

    source_cols = {c for _, c in info["sources"]}
    for c in source_cols:
        if c in info["left_domain"]:
            if not np.any(out[top_row:, c] == LEFT_COLOR):
                return False
        elif c in info["right_domain"]:
            if not np.any(out[top_row:, c] == RIGHT_COLOR):
                return False
        else:
            if not np.all(out[top_row:, c] == SOURCE_COLOR):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(*RECT_HEIGHTS)
    rect_w = random.randint(*RECT_WIDTHS)
    gap_w = random.randint(*GAP_WIDTHS)
    top_margin = random.randint(TOP_MARGIN_MIN, 8)
    bottom_margin = random.randint(*BOTTOM_MARGIN)
    left_margin = random.randint(*SIDE_MARGIN)
    right_margin = random.randint(*SIDE_MARGIN)

    rows = top_margin + rect_h + bottom_margin
    cols = left_margin + rect_w + gap_w + rect_w + right_margin
    if not (GRID_ROWS[0] <= rows <= GRID_ROWS[1] and GRID_COLS[0] <= cols <= GRID_COLS[1]):
        return None

    grid = np.zeros((rows, cols), dtype=int)
    rect_top = top_margin
    left_c0 = left_margin
    left_c1 = left_c0 + rect_w - 1
    right_c0 = left_c1 + gap_w + 1
    right_c1 = right_c0 + rect_w - 1

    grid[rect_top : rect_top + rect_h, left_c0 : left_c1 + 1] = LEFT_COLOR
    grid[rect_top : rect_top + rect_h, right_c0 : right_c1 + 1] = RIGHT_COLOR

    source_cols = _choose_source_columns(left_c1 + 1, right_c0 - 1)
    max_source_row = rect_top - 3
    if max_source_row < 0:
        return None

    source_rows = [random.randint(0, max_source_row) for _ in source_cols]
    if len(set(source_rows)) == 1 and max_source_row >= 1:
        idx = random.randrange(len(source_rows))
        source_rows[idx] = (source_rows[idx] + 1) % (max_source_row + 1)
    for r, c in zip(source_rows, source_cols):
        grid[r, c] = SOURCE_COLOR

    try:
        expected = main(grid)
        if not _semantic_checks(grid, expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
