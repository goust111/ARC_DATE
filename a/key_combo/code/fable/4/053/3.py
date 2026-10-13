import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
NON_PANEL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    return Counter(map(int, border)).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _extract_layout(grid: np.ndarray):
    bg = _border_background(grid)
    comps = _components(grid != bg)

    rectangles = []
    singletons = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        area = len(coords)
        if area == 1:
            singletons.append(coords[0])
        elif area == (r1 - r0 + 1) * (c1 - c0 + 1):
            rectangles.append((r0, c0, r1, c1))

    assert len(rectangles) == 4, "expected three top panels and one verdict panel"
    assert len(singletons) == 3, "expected three legend pixels"

    rectangles.sort(key=lambda b: (b[0], b[1]))
    panel_h = rectangles[0][2] - rectangles[0][0] + 1
    panel_w = rectangles[0][3] - rectangles[0][1] + 1
    assert all((b[2] - b[0] + 1, b[3] - b[1] + 1) == (panel_h, panel_w) for b in rectangles)

    top_row = min(b[0] for b in rectangles)
    top_panels = sorted([b for b in rectangles if b[0] == top_row], key=lambda b: b[1])
    verdicts = [b for b in rectangles if b[0] != top_row]
    assert len(top_panels) == 3 and len(verdicts) == 1
    verdict = verdicts[0]

    legends = sorted(singletons, key=lambda rc: rc[1])
    for (_lr, lc), panel in zip(legends, top_panels):
        assert panel[1] <= lc <= panel[3], "legend must sit above its panel"

    return bg, top_panels, verdict, legends


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unseen third voting panel from two visible panels and a signed verdict panel.

    Input structure:
    - The background canvas contains three same-sized top panel slots laid out left to right as A, B, and a blank X slot.
    - Each top slot has one legend pixel above it that gives that panel's identity color.
    - A and B are black/grey binary panels, X is not drawn, and a same-sized verdict panel C appears lower on the canvas.

    Rule:
    - Read A, B, the three legend colors, and verdict panel C.
    - For each cell: if C is black or grey then all three panels agreed, so X equals that value.
    - If C uses X's legend color, X was the dissenter and must be the opposite of the shared A/B value; if C uses A's or B's legend color, then X matches the non-dissenting visible panel.

    Key insight:
    - The missing panel is never shown directly; verdict colors encode which participant was the unique dissenter at each cell, so the hidden binary state can be inverted cell by cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, top_panels, verdict_bbox, legends = _extract_layout(g)
    a_bbox, b_bbox, x_bbox = top_panels
    verdict_r0, verdict_c0, verdict_r1, verdict_c1 = verdict_bbox

    a_color = int(g[legends[0]])
    b_color = int(g[legends[1]])
    x_color = int(g[legends[2]])
    assert len({a_color, b_color, x_color}) == 3

    a_panel = g[a_bbox[0] : a_bbox[2] + 1, a_bbox[1] : a_bbox[3] + 1]
    b_panel = g[b_bbox[0] : b_bbox[2] + 1, b_bbox[1] : b_bbox[3] + 1]
    x_slot = g[x_bbox[0] : x_bbox[2] + 1, x_bbox[1] : x_bbox[3] + 1]
    verdict = g[verdict_r0 : verdict_r1 + 1, verdict_c0 : verdict_c1 + 1]

    allowed_visible = {BLACK, GREY}
    assert set(map(int, np.unique(a_panel))) <= allowed_visible
    assert set(map(int, np.unique(b_panel))) <= allowed_visible
    assert np.all(x_slot == BLACK), "the missing slot should be blank"
    assert set(map(int, np.unique(verdict))) <= allowed_visible | {a_color, b_color, x_color}

    out = np.empty_like(a_panel)
    for r in range(a_panel.shape[0]):
        for c in range(a_panel.shape[1]):
            cell = int(verdict[r, c])
            aval = int(a_panel[r, c])
            bval = int(b_panel[r, c])
            if cell in (BLACK, GREY):
                out[r, c] = cell
            elif cell == x_color:
                assert aval == bval
                out[r, c] = GREY if aval == BLACK else BLACK
            elif cell == a_color:
                out[r, c] = bval
            else:
                assert cell == b_color
                out[r, c] = aval

    return out


def _opposite(value: int) -> int:
    return GREY if int(value) == BLACK else BLACK


def _verdict_types(size: int) -> np.ndarray:
    seeds = random.sample([(r, c) for r in range(size) for c in range(size)], 5)
    labels = list(range(5))
    random.shuffle(labels)

    verdict_types = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            best_idx = min(
                range(5),
                key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), seeds[i][0], seeds[i][1]),
            )
            verdict_types[r, c] = labels[best_idx]
    return verdict_types


def _panels_from_types(
    verdict_types: np.ndarray, a_color: int, b_color: int, x_color: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = verdict_types.shape[0]
    a_panel = np.full((size, size), BLACK, dtype=int)
    b_panel = np.full((size, size), BLACK, dtype=int)
    x_panel = np.full((size, size), BLACK, dtype=int)
    verdict = np.full((size, size), BLACK, dtype=int)

    for r in range(size):
        for c in range(size):
            kind = int(verdict_types[r, c])
            if kind == 0:
                a_panel[r, c] = b_panel[r, c] = x_panel[r, c] = BLACK
                verdict[r, c] = BLACK
            elif kind == 1:
                a_panel[r, c] = b_panel[r, c] = x_panel[r, c] = GREY
                verdict[r, c] = GREY
            elif kind == 2:
                shared = random.choice((BLACK, GREY))
                a_panel[r, c] = _opposite(shared)
                b_panel[r, c] = shared
                x_panel[r, c] = shared
                verdict[r, c] = a_color
            elif kind == 3:
                shared = random.choice((BLACK, GREY))
                a_panel[r, c] = shared
                b_panel[r, c] = _opposite(shared)
                x_panel[r, c] = shared
                verdict[r, c] = b_color
            else:
                shared = random.choice((BLACK, GREY))
                a_panel[r, c] = shared
                b_panel[r, c] = shared
                x_panel[r, c] = _opposite(shared)
                verdict[r, c] = x_color

    return a_panel, b_panel, x_panel, verdict


def _layout_dimensions(size: int, left: int, gap_x: int, verdict_top: int) -> tuple[int, int]:
    trio_width = left + 3 * size + 2 * gap_x + 1
    width = random.randint(trio_width, 30)

    min_height = verdict_top + size + 1
    height = random.randint(min_height, 30)
    return height, width


def _render_grid(
    a_panel: np.ndarray, b_panel: np.ndarray, verdict: np.ndarray, bg: int, a_color: int, b_color: int, x_color: int
) -> np.ndarray:
    size = a_panel.shape[0]
    top_margin = random.randint(1, 2)
    width_options = [
        (left, gap_x)
        for left in range(1, 4)
        for gap_x in range(1, 4)
        if left + 3 * size + 2 * gap_x + 1 <= 30
    ]
    left, gap_x = random.choice(width_options)
    gap_y = random.randint(1, 3)
    legend_row = top_margin
    panel_top = legend_row + 2
    verdict_top = panel_top + size + gap_y
    height, width = _layout_dimensions(size, left, gap_x, verdict_top)

    a_left = left
    b_left = a_left + size + gap_x
    x_left = b_left + size + gap_x
    verdict_left = random.randint(1, width - size - 1)

    grid = np.full((height, width), bg, dtype=int)
    grid[panel_top : panel_top + size, a_left : a_left + size] = a_panel
    grid[panel_top : panel_top + size, b_left : b_left + size] = b_panel
    grid[panel_top : panel_top + size, x_left : x_left + size] = BLACK
    grid[verdict_top : verdict_top + size, verdict_left : verdict_left + size] = verdict

    legend_cols = [
        random.randint(a_left, a_left + size - 1),
        random.randint(b_left, b_left + size - 1),
        random.randint(x_left, x_left + size - 1),
    ]
    for col, color in zip(legend_cols, (a_color, b_color, x_color)):
        grid[legend_row, col] = color

    return grid


def _valid_hidden_panel(panel: np.ndarray) -> bool:
    vals = set(map(int, np.unique(panel)))
    return vals == {BLACK, GREY}


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(4, 7)
    bg, a_color, b_color, x_color = random.sample(NON_PANEL_COLORS, 4)

    verdict_types = _verdict_types(size)
    a_panel, b_panel, x_panel, verdict = _panels_from_types(verdict_types, a_color, b_color, x_color)
    if not (_valid_hidden_panel(a_panel) and _valid_hidden_panel(b_panel) and _valid_hidden_panel(x_panel)):
        return None

    grid = _render_grid(a_panel, b_panel, verdict, bg, a_color, b_color, x_color)
    if not np.array_equal(main(grid), x_panel):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
