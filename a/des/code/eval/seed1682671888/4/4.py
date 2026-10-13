import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
PANEL_SIZE_RANGE = (7, 11)
PANEL_COUNT_RANGE = (4, 6)
SYMMETRY_KINDS = ("vertical", "horizontal", "rot180", "none")
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _full_line_indices(grid: np.ndarray, axis: int, color: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _regular_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    points = [-1] + sorted(separators) + [size]
    intervals = []
    for a, b in zip(points, points[1:]):
        if b - a > 1:
            intervals.append((a + 1, b))
    return intervals


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    h, w = mask.shape
    seen = {tuple(map(int, cells[0]))}
    stack = [tuple(map(int, cells[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _classify_mask(mask: np.ndarray) -> str:
    tight = _tight_mask(mask)
    vertical = np.array_equal(tight, np.fliplr(tight))
    horizontal = np.array_equal(tight, np.flipud(tight))
    rot180 = np.array_equal(tight, np.rot90(tight, 2))
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    if rot180:
        return "rot180"
    return "none"


def _single_object_color(panel: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(panel) if int(v) != bg]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    return colors[0]


def _complete_panel(panel: np.ndarray, color: int, kind: str, bg: int) -> np.ndarray:
    if kind == "vertical":
        return np.where((panel == color) | (np.fliplr(panel) == color), color, bg)
    if kind == "horizontal":
        return np.where((panel == color) | (np.flipud(panel) == color), color, bg)
    if kind == "rot180":
        return np.where((panel == color) | (np.rot90(panel, 2) == color), color, bg)
    return panel.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each lower work panel according to the symmetry class of the colored sample above it.

    Input structure:
    - Background is black.
    - Grey separator lines split the canvas into two rows of equal-size panels.
    - Top panels contain small monochrome sample patterns in distinct colors.
    - Bottom panels contain larger monochrome half-finished patterns whose colors
      match the sample panels.

    Rule:
    - Classify each sample pattern by its symmetry:
      vertical mirror, horizontal mirror, 180-degree rotational-only, or none.
    - Match each work panel to its sample by color.
    - Complete the work panel with the corresponding symmetry operation.
    - Leave the sample row and separator lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    line_colors = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        rows = _full_line_indices(g, axis=0, color=color)
        cols = _full_line_indices(g, axis=1, color=color)
        if rows and cols:
            line_colors.append((color, rows, cols))
    assert len(line_colors) == 1, "expected one separator color"
    sep_color, row_seps, col_seps = line_colors[0]

    row_intervals = _regular_intervals(row_seps, g.shape[0])
    col_intervals = _regular_intervals(col_seps, g.shape[1])
    assert len(row_intervals) == 2, "expected two panel rows"
    assert len({r1 - r0 for r0, r1 in row_intervals}) == 1, "panel rows must share one height"
    assert len({c1 - c0 for c0, c1 in col_intervals}) == 1, "panel cols must share one width"

    top_row = row_intervals[0]
    bottom_row = row_intervals[1]

    kind_by_color = {}
    for c0, c1 in col_intervals:
        panel = g[top_row[0] : top_row[1], c0:c1]
        color = _single_object_color(panel, bg)
        kind_by_color[color] = _classify_mask(panel == color)

    out = g.copy()
    for c0, c1 in col_intervals:
        panel = g[bottom_row[0] : bottom_row[1], c0:c1]
        color = _single_object_color(panel, bg)
        kind = kind_by_color[color]
        out[bottom_row[0] : bottom_row[1], c0:c1] = _complete_panel(panel, color, kind, bg)

    assert np.all(out[np.where(g == sep_color)] == sep_color)
    return out


def _random_connected_mask(h: int, w: int, min_cells: int, max_cells: int) -> np.ndarray | None:
    total = h * w
    if total < min_cells:
        return None
    target = random.randint(min_cells, min(max_cells, total))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    filled = {start}
    while len(filled) < target:
        frontier = []
        for r, c in filled:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in filled:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cell = random.choice(frontier)
        filled.add(cell)
        mask[cell] = True
    return mask


def _make_vertical_template(max_size: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, min(5, max_size))
        w = random.randint(3, min(5, max_size))
        half_w = (w + 1) // 2
        base = _random_connected_mask(h, half_w, 2, max(2, h * half_w - 1))
        if base is None:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :half_w] = base
        mask |= np.fliplr(mask)
        if _classify_mask(mask) == "vertical":
            return mask
    return None


def _make_horizontal_template(max_size: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, min(5, max_size))
        w = random.randint(3, min(5, max_size))
        half_h = (h + 1) // 2
        base = _random_connected_mask(half_h, w, 2, max(2, half_h * w - 1))
        if base is None:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:half_h, :] = base
        mask |= np.flipud(mask)
        if _classify_mask(mask) == "horizontal":
            return mask
    return None


def _make_rot180_template(max_size: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, min(5, max_size))
        w = random.randint(3, min(5, max_size))
        qh = max(1, h // 2)
        qw = max(1, w // 2)
        base = _random_connected_mask(qh, qw, 1, max(1, qh * qw))
        if base is None:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:qh, :qw] = base
        mask |= np.rot90(mask, 2)
        if _classify_mask(mask) == "rot180":
            return mask
    return None


def _make_none_template(max_size: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, min(5, max_size))
        w = random.randint(3, min(5, max_size))
        mask = _random_connected_mask(h, w, 3, max(3, h * w - 1))
        if mask is None:
            continue
        if _classify_mask(mask) == "none":
            return mask
    return None


def _template_mask(kind: str, panel_size: int) -> np.ndarray:
    makers = {
        "vertical": _make_vertical_template,
        "horizontal": _make_horizontal_template,
        "rot180": _make_rot180_template,
        "none": _make_none_template,
    }
    mask = makers[kind](panel_size - 2)
    if mask is None:
        raise RuntimeError(f"failed to build template mask for {kind}")
    return mask


def _stamp_centered(panel: np.ndarray, mask: np.ndarray, color: int) -> None:
    h, w = panel.shape
    mh, mw = mask.shape
    r0 = random.randint(1, h - mh - 1)
    c0 = random.randint(1, w - mw - 1)
    panel[r0 : r0 + mh, c0 : c0 + mw][mask] = color


def _panel_symmetry_kind(panel: np.ndarray, color: int, bg: int) -> str:
    mask = panel == color
    vertical = np.array_equal(mask, np.fliplr(mask))
    horizontal = np.array_equal(mask, np.flipud(mask))
    rot180 = np.array_equal(mask, np.rot90(mask, 2))
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    if rot180:
        return "rot180"
    return "none"


def _make_vertical_work(panel_size: int, color: int) -> np.ndarray | None:
    side_w = panel_size // 2
    base = _random_connected_mask(panel_size, side_w, panel_size, panel_size * side_w - 1)
    if base is None:
        return None
    panel = np.full((panel_size, panel_size), BG, dtype=int)
    if random.random() < 0.5:
        panel[:, :side_w][base] = color
    else:
        panel[:, side_w + 1 :][base] = color
    return panel


def _make_horizontal_work(panel_size: int, color: int) -> np.ndarray | None:
    side_h = panel_size // 2
    base = _random_connected_mask(side_h, panel_size, panel_size, side_h * panel_size - 1)
    if base is None:
        return None
    panel = np.full((panel_size, panel_size), BG, dtype=int)
    if random.random() < 0.5:
        panel[:side_h, :][base] = color
    else:
        panel[side_h + 1 :, :][base] = color
    return panel


def _make_rot180_work(panel_size: int, color: int) -> np.ndarray | None:
    q = panel_size // 2
    base = _random_connected_mask(q, q, max(3, q), q * q)
    if base is None:
        return None
    panel = np.full((panel_size, panel_size), BG, dtype=int)
    quadrants = [
        (slice(0, q), slice(0, q)),
        (slice(0, q), slice(q + 1, panel_size)),
        (slice(q + 1, panel_size), slice(0, q)),
        (slice(q + 1, panel_size), slice(q + 1, panel_size)),
    ]
    rs, cs = random.choice(quadrants)
    panel[rs, cs][base] = color
    return panel


def _make_none_work(panel_size: int, color: int) -> np.ndarray | None:
    for _ in range(80):
        mask = _random_connected_mask(panel_size, panel_size, panel_size, panel_size * panel_size // 2)
        if mask is None:
            continue
        panel = np.full((panel_size, panel_size), BG, dtype=int)
        panel[mask] = color
        if _panel_symmetry_kind(panel, color, BG) == "none":
            return panel
    return None


def _work_panel(kind: str, panel_size: int, color: int) -> np.ndarray:
    makers = {
        "vertical": _make_vertical_work,
        "horizontal": _make_horizontal_work,
        "rot180": _make_rot180_work,
        "none": _make_none_work,
    }
    for _ in range(120):
        panel = makers[kind](panel_size, color)
        if panel is not None:
            return panel
    raise RuntimeError(f"failed to build work panel for {kind}")


def _build_instance() -> np.ndarray | None:
    panel_size = random.choice([s for s in range(PANEL_SIZE_RANGE[0], PANEL_SIZE_RANGE[1] + 1) if s % 2 == 1])
    panel_count = random.randint(PANEL_COUNT_RANGE[0], PANEL_COUNT_RANGE[1])
    if panel_count > len(COLOR_POOL):
        return None

    branch_order = list(SYMMETRY_KINDS)
    extras = random.sample(SYMMETRY_KINDS, panel_count - len(SYMMETRY_KINDS))
    all_kinds = branch_order + extras
    random.shuffle(all_kinds)

    colors = random.sample(COLOR_POOL, panel_count)
    kind_by_color = dict(zip(colors, all_kinds, strict=False))

    top_panels = []
    bottom_panels = []
    for color in colors:
        kind = kind_by_color[color]
        template = np.full((panel_size, panel_size), BG, dtype=int)
        _stamp_centered(template, _template_mask(kind, panel_size), color)
        bottom = _work_panel(kind, panel_size, color)
        top_panels.append(template)
        bottom_panels.append(bottom)

    width = panel_count * panel_size + (panel_count - 1)
    height = panel_size * 2 + 1
    grid = np.full((height, width), BG, dtype=int)

    grid[panel_size, :] = SEP
    for idx in range(1, panel_count):
        grid[:, idx * panel_size + (idx - 1)] = SEP

    col_intervals = _regular_intervals(_full_line_indices(grid, axis=1, color=SEP), width)
    assert len(col_intervals) == panel_count
    for (c0, c1), top, bottom in zip(col_intervals, top_panels, bottom_panels, strict=False):
        grid[:panel_size, c0:c1] = top
        grid[panel_size + 1 :, c0:c1] = bottom

    if Counter(_classify_mask(panel == color) for panel, color in zip(top_panels, colors, strict=False))["none"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance()
        if grid is None:
            continue
        out = main(grid)
        if out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid input")
