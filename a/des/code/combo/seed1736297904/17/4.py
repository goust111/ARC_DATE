from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
WALL_COLORS = (1, 2, 3)
CELL_COLORS = (4, 5, 6, 7, 8, 9)

SOLID = "solid"
HORIZONTAL = "horizontal"
VERTICAL = "vertical"


GRID_MIN = 11
GRID_MAX = 28
CELL_ROWS_RANGE = (2, 4)
CELL_COLS_RANGE = (2, 4)
CELL_SIZE_RANGE = (3, 5)
MARGIN_RANGE = (1, 2)
GENERATE_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _side_strip(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
) -> np.ndarray | None:
    if side == "top":
        return None if r0 == 0 else grid[r0 - 1, c0 : c1 + 1]
    if side == "bottom":
        return None if r1 == grid.shape[0] - 1 else grid[r1 + 1, c0 : c1 + 1]
    if side == "left":
        return None if c0 == 0 else grid[r0 : r1 + 1, c0 - 1]
    if side == "right":
        return None if c1 == grid.shape[1] - 1 else grid[r0 : r1 + 1, c1 + 1]
    raise ValueError(f"unknown side: {side}")


def _side_wall_color(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
) -> int | None:
    strip = _side_strip(grid, r0, c0, r1, c1, side)
    if strip is None:
        return None

    values = [int(v) for v in strip.ravel().tolist()]
    unique = set(values)
    if len(unique) == 1:
        color = values[0]
        return color if color in WALL_COLORS else None

    if unique.issubset(WALL_COLORS):
        counts = Counter(values)
        color, count = counts.most_common(1)[0]
        if count > len(values) // 2:
            return int(color)
    return None


def _decide_pattern(horizontal_walls: list[int], vertical_walls: list[int]) -> str:
    wall_colors = horizontal_walls + vertical_walls
    if len(set(wall_colors)) == 1:
        return SOLID
    if len(set(wall_colors)) != 2:
        return SOLID

    h_uniform = bool(horizontal_walls) and len(set(horizontal_walls)) == 1
    v_uniform = bool(vertical_walls) and len(set(vertical_walls)) == 1

    if h_uniform and not v_uniform:
        return HORIZONTAL
    if v_uniform and not h_uniform:
        return VERTICAL

    # When both axes are internally consistent but point to different colors,
    # border cells can still be oriented by whichever axis contributes more walls.
    if h_uniform and v_uniform and set(horizontal_walls) != set(vertical_walls):
        if len(horizontal_walls) > len(vertical_walls):
            return HORIZONTAL
        if len(vertical_walls) > len(horizontal_walls):
            return VERTICAL

    return SOLID


def _vote_color(original_color: int, wall_colors: list[int]) -> int:
    counts = Counter(wall_colors + [original_color])
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    return original_color if len(winners) != 1 else int(winners[0])


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    wall_mask = np.isin(g, WALL_COLORS)
    cell_mask = (g != BLACK) & (~wall_mask)

    regions: list[dict[str, object]] = []
    for comp in _components(cell_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        rect = cell_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(rect), "each cell region must be a solid rectangle"

        colors = np.unique(g[r0 : r1 + 1, c0 : c1 + 1])
        assert colors.size == 1, "each cell must start as a single-color fill"
        original_color = int(colors[0])
        assert original_color not in WALL_COLORS and original_color != BLACK

        side_colors: dict[str, int] = {}
        for side in ("top", "bottom", "left", "right"):
            color = _side_wall_color(g, r0, c0, r1, c1, side)
            if color is not None:
                side_colors[side] = color

        assert 2 <= len(side_colors) <= 4, "each cell must have 2 to 4 surrounding walls"
        horizontal_walls = [side_colors[side] for side in ("top", "bottom") if side in side_colors]
        vertical_walls = [side_colors[side] for side in ("left", "right") if side in side_colors]
        pattern = _decide_pattern(horizontal_walls, vertical_walls)
        draw_color = _vote_color(original_color, list(side_colors.values()))

        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "original_color": original_color,
                "wall_colors": side_colors,
                "pattern": pattern,
                "draw_color": draw_color,
            }
        )

    regions.sort(key=lambda region: tuple(region["bbox"]))
    assert regions, "expected at least one rectangular cell"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular region using a pattern determined by its adjacent wall colors.

    Input structure:
    - Black background.
    - Single-pixel red/blue/green walls split the scene into rectangular cells.
    - Each cell interior is a solid non-black, non-wall color and has size at least 3x3.

    Rule:
    - Collect the colors of the actual walls around each cell.
    - If all surrounding walls share one color, the cell becomes a solid fill.
    - If exactly two wall colors occur, use their horizontal/vertical placement to
      choose horizontal stripes, vertical stripes, or fall back to solid.
    - The drawing color is the majority color among the cell's original fill color
      and all of its adjacent wall colors; ties keep the original fill color.
    - Wall pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for region in _extract_regions(g):
        r0, c0, r1, c1 = region["bbox"]
        pattern = str(region["pattern"])
        color = int(region["draw_color"])

        if pattern == SOLID:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        elif pattern == HORIZONTAL:
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            out[r0 : r1 + 1 : 2, c0 : c1 + 1] = color
        else:
            assert pattern == VERTICAL
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            out[r0 : r1 + 1, c0 : c1 + 1 : 2] = color

    return out


def _axis_positions(sizes: list[int], margin_before: int) -> tuple[list[int], int]:
    starts: list[int] = []
    pos = margin_before
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i != len(sizes) - 1:
            pos += 1
    return starts, pos


def _sample_cell_color_grid(n_rows: int, n_cols: int) -> np.ndarray:
    colors = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            forbidden = set()
            if r > 0:
                forbidden.add(int(colors[r - 1, c]))
            if c > 0:
                forbidden.add(int(colors[r, c - 1]))
            choices = [color for color in CELL_COLORS if color not in forbidden]
            colors[r, c] = random.choice(choices or list(CELL_COLORS))
    return colors


def _sample_layout(focus: str) -> tuple[int, int, list[int], list[int], tuple[int, int, int, int]] | None:
    if focus == HORIZONTAL:
        n_rows = random.randint(2, CELL_ROWS_RANGE[1])
        n_cols = random.randint(3, CELL_COLS_RANGE[1])
    elif focus == VERTICAL:
        n_rows = random.randint(3, CELL_ROWS_RANGE[1])
        n_cols = random.randint(2, CELL_COLS_RANGE[1])
    else:
        n_rows = random.randint(*CELL_ROWS_RANGE)
        n_cols = random.randint(*CELL_COLS_RANGE)

    heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_rows)]
    widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_cols)]
    margins = tuple(random.randint(*MARGIN_RANGE) for _ in range(4))
    mt, mb, ml, mr = margins

    grid_h = mt + mb + sum(heights) + (n_rows - 1)
    grid_w = ml + mr + sum(widths) + (n_cols - 1)
    if not (GRID_MIN <= grid_h <= GRID_MAX and GRID_MIN <= grid_w <= GRID_MAX):
        return None

    return n_rows, n_cols, heights, widths, margins


def _force_focus(h_lines: list[int], v_lines: list[int], focus: str, n_rows: int, n_cols: int) -> None:
    if focus == HORIZONTAL:
        color_a, color_b = random.sample(WALL_COLORS, 2)
        target_col = random.randint(1, n_cols - 2)
        edge_row = random.choice((0, n_rows - 1))
        h_lines[0 if edge_row == 0 else n_rows - 2] = color_a
        v_lines[target_col - 1] = color_a
        v_lines[target_col] = color_b
        return

    if focus == VERTICAL:
        color_a, color_b = random.sample(WALL_COLORS, 2)
        target_row = random.randint(1, n_rows - 2)
        edge_col = random.choice((0, n_cols - 1))
        v_lines[0 if edge_col == 0 else n_cols - 2] = color_b
        h_lines[target_row - 1] = color_a
        h_lines[target_row] = color_b
        return

    color = random.choice(WALL_COLORS)
    corner_row = random.choice((0, n_rows - 1))
    corner_col = random.choice((0, n_cols - 1))
    h_lines[0 if corner_row == 0 else n_rows - 2] = color
    v_lines[0 if corner_col == 0 else n_cols - 2] = color


def _render_grid(
    n_rows: int,
    n_cols: int,
    heights: list[int],
    widths: list[int],
    margins: tuple[int, int, int, int],
    h_lines: list[int],
    v_lines: list[int],
    cell_colors: np.ndarray,
) -> np.ndarray:
    mt, mb, ml, mr = margins
    row_starts, after_rows = _axis_positions(heights, mt)
    col_starts, after_cols = _axis_positions(widths, ml)
    grid_h = after_rows + mb
    grid_w = after_cols + mr
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for r in range(n_rows):
        for c in range(n_cols):
            rr = row_starts[r]
            cc = col_starts[c]
            grid[rr : rr + heights[r], cc : cc + widths[c]] = int(cell_colors[r, c])

    content_top = mt
    content_bottom = grid_h - mb - 1
    content_left = ml
    content_right = grid_w - mr - 1

    for idx, color in enumerate(h_lines):
        wall_row = row_starts[idx] + heights[idx]
        grid[wall_row, content_left : content_right + 1] = color

    for idx, color in enumerate(v_lines):
        wall_col = col_starts[idx] + widths[idx]
        grid[content_top : content_bottom + 1, wall_col] = color

    return grid


def _valid_focus(grid: np.ndarray, focus: str) -> bool:
    try:
        regions = _extract_regions(grid)
    except AssertionError:
        return False

    patterns = [str(region["pattern"]) for region in regions]
    if focus not in patterns:
        return False

    return np.array_equal(main(grid), main(grid.copy()))


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        focus = random.choice((SOLID, HORIZONTAL, VERTICAL))
        layout = _sample_layout(focus)
        if layout is None:
            continue
        n_rows, n_cols, heights, widths, margins = layout

        h_lines = [random.choice(WALL_COLORS) for _ in range(n_rows - 1)]
        v_lines = [random.choice(WALL_COLORS) for _ in range(n_cols - 1)]
        _force_focus(h_lines, v_lines, focus, n_rows, n_cols)

        cell_colors = _sample_cell_color_grid(n_rows, n_cols)
        grid = _render_grid(
            n_rows=n_rows,
            n_cols=n_cols,
            heights=heights,
            widths=widths,
            margins=margins,
            h_lines=h_lines,
            v_lines=v_lines,
            cell_colors=cell_colors,
        )

        if _valid_focus(grid, focus):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
