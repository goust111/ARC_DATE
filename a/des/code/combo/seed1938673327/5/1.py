import random
from collections import defaultdict, deque

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LEGEND_GAP = 1
MAX_GENERATION_ATTEMPTS = 400

_RAW_TEMPLATES = {
    2: (
        ("10", "01"),
        ("11", "01"),
        ("10", "11"),
        ("11", "10"),
        ("01", "11"),
    ),
    3: (
        ("101", "010", "101"),
        ("111", "010", "010"),
        ("110", "010", "011"),
        ("100", "111", "001"),
        ("101", "111", "000"),
        ("010", "111", "010"),
        ("110", "101", "011"),
    ),
}
TEMPLATE_LIBRARY = {
    size: [np.array([[ch == "1" for ch in row] for row in rows], dtype=bool) for rows in patterns]
    for size, patterns in _RAW_TEMPLATES.items()
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _legend_side(grid: np.ndarray) -> str:
    left = np.any((grid[:, 0] != BG) & (grid[:, 0] != WALL))
    right = np.any((grid[:, -1] != BG) & (grid[:, -1] != WALL))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    if left:
        return "left"
    if right:
        return "right"
    raise AssertionError("legend must touch the left or right edge")


def _parse_legend(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    side = _legend_side(grid)
    mapping: dict[int, np.ndarray] = {}
    edge_col = 0 if side == "left" else w - 1
    r = 0
    while r < h:
        color = int(grid[r, edge_col])
        if color in (BG, WALL):
            r += 1
            continue
        r1 = r
        while r1 < h and int(grid[r1, edge_col]) == color:
            r1 += 1
        size = r1 - r
        assert size in (2, 3), "legend entries must be 2x2 or 3x3"
        if side == "left":
            patch = grid[r:r1, 1 : 1 + size]
        else:
            patch = grid[r:r1, w - 1 - size : w - 1]
        assert patch.shape == (size, size)
        assert np.all((patch == BG) | (patch == color))
        mapping[color] = (patch == color).copy()
        r = r1
    assert mapping, "expected a non-empty legend"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color→template legend on the side edge and tile each enclosed region
    with the template keyed by that region's marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    mapping = _parse_legend(g)
    h, w = g.shape
    out = g.copy()

    for comp in _components(g != WALL):
        if _touches_border(comp, h, w):
            continue
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BG}
        assert len(colors) == 1, "each region must contain one marker color"
        color = colors.pop()
        template = mapping[color]
        th, tw = template.shape
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        for r, c in comp:
            out[r, c] = color if template[(r - r0) % th, (c - c0) % tw] else BG

    return out


def _pick_templates(rng: random.Random, count: int) -> list[np.ndarray]:
    sizes = [rng.choice((2, 3)) for _ in range(count)]
    if count >= 2:
        sizes[0] = 2
        sizes[1] = 3
        rng.shuffle(sizes)

    chosen: list[np.ndarray] = []
    used_by_size = defaultdict(set)
    for size in sizes:
        pool = TEMPLATE_LIBRARY[size]
        choices = [idx for idx in range(len(pool)) if idx not in used_by_size[size]]
        idx = rng.choice(choices)
        used_by_size[size].add(idx)
        chosen.append(pool[idx].copy())
    return chosen


def _spans(start: int, lengths: list[int]) -> list[tuple[int, int]]:
    spans = []
    pos = start + 1
    for length in lengths:
        spans.append((pos, pos + length - 1))
        pos += length + 1
    return spans


def _paint_board(grid: np.ndarray, top: int, left: int, heights: list[int], widths: list[int]):
    row_spans = _spans(top, heights)
    col_spans = _spans(left, widths)
    bottom = row_spans[-1][1] + 1
    right = col_spans[-1][1] + 1

    grid[top : bottom + 1, left : right + 1] = BG
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL

    for r0, r1 in row_spans[:-1]:
        grid[r1 + 1, left : right + 1] = WALL
    for c0, c1 in col_spans[:-1]:
        grid[top : bottom + 1, c1 + 1] = WALL

    return row_spans, col_spans


def _paint_legend(grid: np.ndarray, side: str, top: int, colors: list[int], templates: list[np.ndarray]) -> None:
    h, w = grid.shape
    row = top
    for color, template in zip(colors, templates):
        size = int(template.shape[0])
        if side == "left":
            grid[row : row + size, 0] = color
            patch = grid[row : row + size, 1 : 1 + size]
        else:
            grid[row : row + size, w - 1] = color
            patch = grid[row : row + size, w - 1 - size : w - 1]
        patch[template] = color
        row += size + LEGEND_GAP


def _tile_region(out: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int, template: np.ndarray) -> None:
    th, tw = template.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = color if template[(r - r0) % th, (c - c0) % tw] else BG


def _region_foreground(
    r0: int, r1: int, c0: int, c1: int, template: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fg: list[tuple[int, int]] = []
    bg: list[tuple[int, int]] = []
    th, tw = template.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if template[(r - r0) % th, (c - c0) % tw]:
                fg.append((r, c))
            else:
                bg.append((r, c))
    return fg, bg


def _choose_markers(
    rng: random.Random, r0: int, r1: int, c0: int, c1: int, template: np.ndarray
) -> list[tuple[int, int]]:
    fg, bg = _region_foreground(r0, r1, c0, c1, template)
    cells = fg + bg
    count = rng.randint(1, min(4, len(cells)))
    chosen: list[tuple[int, int]] = []
    used = set()

    if count >= 2 and fg and bg:
        pair = [rng.choice(fg), rng.choice(bg)]
        for cell in pair:
            if cell not in used:
                chosen.append(cell)
                used.add(cell)

    if not chosen:
        pool = bg if bg and rng.random() < 0.5 else fg
        if not pool:
            pool = cells
        first = rng.choice(pool)
        chosen.append(first)
        used.add(first)

    remaining = [cell for cell in cells if cell not in used]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_region_rows = rng.randint(2, 3)
    n_region_cols = rng.randint(2, 4)
    heights = [rng.randint(2, 5) for _ in range(n_region_rows)]
    widths = [rng.randint(2, 5) for _ in range(n_region_cols)]
    n_regions = n_region_rows * n_region_cols

    n_colors = rng.randint(2, min(4, n_regions))
    colors = rng.sample(COLOR_POOL, n_colors)
    templates = _pick_templates(rng, n_colors)
    template_by_color = {color: template for color, template in zip(colors, templates)}

    legend_height = sum(int(template.shape[0]) for template in templates) + LEGEND_GAP * (n_colors - 1)
    board_height = sum(heights) + n_region_rows + 1
    board_width = sum(widths) + n_region_cols + 1

    side = rng.choice(("left", "right"))
    board_top = rng.randint(1, 3)
    board_bottom_margin = rng.randint(1, 3)
    h = max(board_top + board_height + board_bottom_margin, legend_height + rng.randint(0, 3))
    legend_top = rng.randint(0, h - legend_height)

    gap = rng.randint(2, 3)
    if side == "left":
        board_left = 4 + gap + rng.randint(0, 2)
        w = board_left + board_width + rng.randint(1, 3)
    else:
        board_left = rng.randint(1, 3)
        w = board_left + board_width + gap + 4 + rng.randint(0, 2)

    base = np.full((h, w), BG, dtype=int)
    _paint_legend(base, side, legend_top, colors, templates)
    row_spans, col_spans = _paint_board(base, board_top, board_left, heights, widths)

    region_boxes = [(r0, r1, c0, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    rng.shuffle(region_boxes)

    color_assignments: list[int] = []
    for color in colors:
        color_assignments.append(color)
    while len(color_assignments) < len(region_boxes):
        color_assignments.append(rng.choice(colors))
    rng.shuffle(color_assignments)

    expected = base.copy()
    inp = base.copy()
    for (r0, r1, c0, c1), color in zip(region_boxes, color_assignments):
        template = template_by_color[color]
        _tile_region(expected, r0, r1, c0, c1, color, template)
        for mr, mc in _choose_markers(rng, r0, r1, c0, c1, template):
            inp[mr, mc] = color

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
