import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

COMPLEMENT = {
    BLUE: CYAN,
    CYAN: BLUE,
    GREEN: YELLOW,
    YELLOW: GREEN,
    MAGENTA: MAROON,
    MAROON: MAGENTA,
}

HLINE = ((1, 1, 1),)
VLINE = ((1,), (1,), (1,))
L_SHAPES = {
    ((1, 1), (1, 0)),
    ((1, 1), (0, 1)),
    ((1, 0), (1, 1)),
    ((0, 1), (1, 1)),
}
CROSS = (
    (0, 1, 0),
    (1, 1, 1),
    (0, 1, 0),
)
OPS = ("hflip", "vflip", "rot90cw", "invert")
PAIRED_COLORS = [BLUE, CYAN, GREEN, YELLOW, MAGENTA, MAROON]
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    mask = grid != BLACK
    r0, c0, r1, c1 = _bbox(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_component(comp: list[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=int)
    for r, c in comp:
        mask[r - r0, c - c0] = 1
    return tuple(tuple(int(v) for v in row) for row in mask)


def _decode_symbol(comp: list[tuple[int, int]]) -> str:
    shape = _normalize_component(comp)
    if shape == HLINE:
        return "hflip"
    if shape == VLINE:
        return "vflip"
    if shape in L_SHAPES:
        return "rot90cw"
    if shape == CROSS:
        return "invert"
    raise AssertionError(f"unrecognized symbol shape: {shape}")


def _apply_op(pattern: np.ndarray, op: str) -> np.ndarray:
    if op == "hflip":
        return np.fliplr(pattern)
    if op == "vflip":
        return np.flipud(pattern)
    if op == "rot90cw":
        return np.rot90(pattern, -1)
    if op == "invert":
        out = pattern.copy()
        for color, comp in COMPLEMENT.items():
            out[pattern == color] = comp
        return out
    raise AssertionError(f"unknown op: {op}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a red-symbol instruction panel and apply the encoded spatial transforms.

    Input structure
    - Black background.
    - One full grey separator row splits the grid into source (top) and instructions (bottom).
    - The source region contains a multicolor pattern.
    - The instruction region contains 1-3 horizontally ordered 3x3 red symbols.

    Rule
    - Horizontal line -> horizontal flip.
    - Vertical line -> vertical flip.
    - L shape -> rotate clockwise 90 degrees.
    - Cross -> replace each foreground color with its complementary partner.
    - Apply instructions from left to right and output the transformed source pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert sep_rows, "missing grey separator row"
    assert sep_rows == list(range(sep_rows[0], sep_rows[-1] + 1)), "separator rows must be contiguous"
    top = g[: sep_rows[0], :]
    bottom = g[sep_rows[-1] + 1 :, :]
    assert top.size > 0 and bottom.size > 0

    source = _crop_nonblack(top)
    red_components = _components_of_color(bottom, RED)
    assert 1 <= len(red_components) <= 3
    ops = [
        _decode_symbol(comp)
        for comp in sorted(red_components, key=lambda comp: min(c for _, c in comp))
    ]

    out = source.copy()
    for op in ops:
        out = _apply_op(out, op)
    return out


def _symbol_mask(op: str) -> np.ndarray:
    if op == "hflip":
        return np.array([[1, 1, 1]], dtype=bool)
    if op == "vflip":
        return np.array([[1], [1], [1]], dtype=bool)
    if op == "rot90cw":
        return np.array([[1, 1], [1, 0]], dtype=bool)
    if op == "invert":
        return np.array(CROSS, dtype=bool)
    raise AssertionError(op)


def _random_connected_mask(h: int, w: int, min_cells: int, max_cells: int) -> np.ndarray | None:
    target = random.randint(min_cells, min(max_cells, h * w))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target:
        if not frontier:
            return None
        choices = sorted(
            frontier,
            key=lambda rc: sum((rr, cc) in cells for rr, cc in _neighbors4(*rc)),
            reverse=True,
        )
        pick = random.choice(choices[: max(1, min(4, len(choices)))])
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(mask.shape, dtype=int)
    count = int(mask.sum())
    palette_size = random.randint(2, min(4, len(PAIRED_COLORS), count))
    palette = random.sample(PAIRED_COLORS, palette_size)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(coords)
    used = coords[:palette_size]
    for color, (r, c) in zip(palette, used):
        grid[r, c] = color
    for r, c in coords[palette_size:]:
        grid[r, c] = random.choice(palette)
    return grid


def _has_effect(source: np.ndarray, ops: list[str]) -> bool:
    out = source.copy()
    for op in ops:
        out = _apply_op(out, op)
    return not np.array_equal(out, source)


def _build_input(source: np.ndarray, ops: list[str]) -> np.ndarray:
    sh, sw = source.shape
    symbol_width = sum(_symbol_mask(op).shape[1] for op in ops)
    symbol_width += random.randint(1, 2) * (len(ops) - 1)
    width = max(sw, symbol_width) + random.randint(2, 6)
    top_margin = random.randint(1, 3)
    left_source = random.randint(1, width - sw - 1)
    bottom_h = 3 + random.randint(1, 3)
    height = top_margin + sh + 1 + bottom_h

    grid = np.zeros((height, width), dtype=int)
    grid[top_margin + sh, :] = GREY
    grid[top_margin : top_margin + sh, left_source : left_source + sw] = source

    panel_top = top_margin + sh + 1 + random.randint(0, bottom_h - 3)
    total_symbols = sum(_symbol_mask(op).shape[1] for op in ops)
    total_gaps = symbol_width - total_symbols
    gap_slots = [0] * (len(ops) - 1)
    for _ in range(total_gaps):
        gap_slots[random.randrange(len(gap_slots))] += 1
    cursor = random.randint(0, width - symbol_width)
    for i, op in enumerate(ops):
        mask = _symbol_mask(op)
        mh, mw = mask.shape
        patch = np.zeros((3, mw), dtype=int)
        row_offset = (3 - mh) // 2
        patch[row_offset : row_offset + mh, :mw][mask] = RED
        grid[panel_top : panel_top + 3, cursor : cursor + mw] = np.maximum(
            grid[panel_top : panel_top + 3, cursor : cursor + mw],
            patch,
        )
        cursor += mw
        if i < len(gap_slots):
            cursor += gap_slots[i]
    return grid


def _sample_ops() -> list[str]:
    count = random.randint(1, 3)
    if count == 1:
        return [random.choice(OPS)]
    if count == 2:
        return random.sample(list(OPS), 2)
    return random.sample(list(OPS), 3)


def _gen_attempt() -> np.ndarray | None:
    src_h = random.randint(3, 7)
    src_w = random.randint(3, 7)
    if random.random() < 0.5:
        src_w = random.choice([w for w in range(3, 8) if w != src_h] or [src_w])

    mask = _random_connected_mask(src_h, src_w, min_cells=4, max_cells=max(5, (src_h * src_w) // 2 + 3))
    if mask is None:
        return None
    source = _colorize_mask(mask)
    ops = _sample_ops()

    if "invert" in ops and len(np.unique(source[source != BLACK])) < 2:
        return None
    if not _has_effect(source, ops):
        return None

    inp = _build_input(source, ops)
    expected = source.copy()
    for op in ops:
        expected = _apply_op(expected, op)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
