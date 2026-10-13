from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _detect_frame_color(grid: np.ndarray, bg: int) -> int:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[1:-1, 0]
    right = grid[1:-1, -1]
    ring = np.concatenate([top, bottom, left, right])
    ring = ring[ring != bg]
    assert ring.size > 0, "expected a non-background outer frame"
    colors, counts = np.unique(ring, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_holes(grid: np.ndarray, bg: int) -> list[tuple[str, int, int]]:
    height, width = map(int, grid.shape)
    holes: list[tuple[str, int, int]] = []
    for col in range(1, width - 1):
        if int(grid[0, col]) == bg:
            holes.append(("top", 0, col))
        if int(grid[height - 1, col]) == bg:
            holes.append(("bottom", height - 1, col))
    for row in range(1, height - 1):
        if int(grid[row, 0]) == bg:
            holes.append(("left", row, 0))
        if int(grid[row, width - 1]) == bg:
            holes.append(("right", row, width - 1))
    return holes


def _find_components(
    grid: np.ndarray, bg: int, frame_color: int
) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], np.ndarray]:
    height, width = map(int, grid.shape)
    labels = np.full((height, width), -1, dtype=int)
    components: list[dict[str, int | tuple[int, int, int, int]]] = []
    next_id = 0

    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color in (bg, frame_color) or labels[row, col] != -1:
                continue

            stack = [(row, col)]
            labels[row, col] = next_id
            coords: list[tuple[int, int]] = []
            r0 = r1 = row
            c0 = c1 = col

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if labels[nr, nc] != -1 or int(grid[nr, nc]) != color:
                        continue
                    labels[nr, nc] = next_id
                    stack.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )
            next_id += 1

    return components, labels


def _trace_to_component(
    grid: np.ndarray,
    labels: np.ndarray,
    bg: int,
    frame_color: int,
    side: str,
    row: int,
    col: int,
) -> int | None:
    height, width = map(int, grid.shape)

    if side == "top":
        cells = ((rr, col) for rr in range(1, height - 1))
    elif side == "bottom":
        cells = ((rr, col) for rr in range(height - 2, 0, -1))
    elif side == "left":
        cells = ((row, cc) for cc in range(1, width - 1))
    else:
        cells = ((row, cc) for cc in range(width - 2, 0, -1))

    for rr, cc in cells:
        value = int(grid[rr, cc])
        if value in (bg, frame_color):
            continue
        comp_id = int(labels[rr, cc])
        return comp_id if comp_id >= 0 else None
    return None


def _stretch_component(
    out: np.ndarray, component: dict[str, int | tuple[int, int, int, int]], side: str
) -> None:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    height, width = map(int, out.shape)

    if side == "top" and r1 + 1 <= height - 2:
        out[r1 + 1 : height - 1, c0 : c1 + 1] = color
    elif side == "bottom" and 1 <= r0 - 1:
        out[1:r0, c0 : c1 + 1] = color
    elif side == "left" and c1 + 1 <= width - 2:
        out[r0 : r1 + 1, c1 + 1 : width - 1] = color
    elif side == "right" and 1 <= c0 - 1:
        out[r0 : r1 + 1, 1:c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each interior color block from the border hole that points to it until the opposite inner wall.

    Input structure:
    - The background is black and the grid is surrounded by a one-cell-thick gray frame.
    - Some frame cells are replaced by single black holes on the top, bottom, left, or right side.
    - Inside the frame sit several small monochrome rectangles, each uniquely aligned with one hole in the same row or column.

    Rule:
    - Find every hole on the outer frame.
    - For each hole, look inward along the direction implied by its side: top goes down, bottom goes up, left goes right, right goes left.
    - Locate the aligned interior rectangle and extend it further in that same direction, preserving its perpendicular thickness, until the extension reaches the opposite inner wall.
    - Keep the original rectangle and leave all untouched background and frame pixels unchanged.

    Key insight:
    - A hole is both a selector and a direction marker: its border side tells which way the matched rectangle must project.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(grid)
    frame_color = _detect_frame_color(grid, bg)
    components, labels = _find_components(grid, bg, frame_color)
    out = grid.copy()

    for side, row, col in _find_holes(grid, bg):
        comp_id = _trace_to_component(grid, labels, bg, frame_color, side, row, col)
        assert comp_id is not None, "every hole must point to an interior block"
        _stretch_component(out, components[comp_id], side)

    return out


def _sample_side_choices(rng: np.random.Generator, axis: str, count: int) -> list[str]:
    if axis == "vertical":
        sides = ["top", "bottom"]
    else:
        sides = ["left", "right"]

    picks = [sides[int(rng.integers(0, 2))] for _ in range(count)]
    if count >= 2 and rng.random() < 0.65:
        picks[0] = sides[0]
        picks[1] = sides[1]
        rng.shuffle(picks)
    return picks


def _make_framed_grid(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = FRAME_COLOR
    grid[-1, :] = FRAME_COLOR
    grid[:, 0] = FRAME_COLOR
    grid[:, -1] = FRAME_COLOR
    return grid


def _generate_vertical_case(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(1, 6))
    widths = rng.integers(1, 4, size=count)
    heights = rng.integers(1, 4, size=count)
    gaps = rng.integers(1, 3, size=max(0, count - 1))
    left_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    base_width = 2 + left_margin + right_margin + int(widths.sum()) + int(gaps.sum())
    if base_width > 30:
        return None

    extra_width = int(rng.integers(0, 31 - base_width))
    extra_left = int(rng.integers(0, extra_width + 1))
    left_margin += extra_left
    right_margin += extra_width - extra_left
    width = base_width + extra_width

    min_height = max(8, int(heights.max()) + 4)
    height = int(rng.integers(min_height, 31))
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=count, replace=False)]
    sides = _sample_side_choices(rng, "vertical", count)

    grid = _make_framed_grid(height, width)
    col = 1 + left_margin
    for idx in range(count):
        block_w = int(widths[idx])
        block_h = int(heights[idx])
        c0 = col
        c1 = col + block_w - 1
        r0 = int(rng.integers(2, height - block_h - 1))
        r1 = r0 + block_h - 1
        hole_col = int(rng.integers(c0, c1 + 1))

        if sides[idx] == "top":
            grid[0, hole_col] = BG
        else:
            grid[height - 1, hole_col] = BG

        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]
        if idx + 1 < count:
            col = c1 + 1 + int(gaps[idx])

    return grid


def _generate_horizontal_case(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(1, 6))
    heights = rng.integers(1, 4, size=count)
    widths = rng.integers(1, 4, size=count)
    gaps = rng.integers(1, 3, size=max(0, count - 1))
    top_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    base_height = 2 + top_margin + bottom_margin + int(heights.sum()) + int(gaps.sum())
    if base_height > 30:
        return None

    extra_height = int(rng.integers(0, 31 - base_height))
    extra_top = int(rng.integers(0, extra_height + 1))
    top_margin += extra_top
    bottom_margin += extra_height - extra_top
    height = base_height + extra_height

    min_width = max(8, int(widths.max()) + 4)
    width = int(rng.integers(min_width, 31))
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=count, replace=False)]
    sides = _sample_side_choices(rng, "horizontal", count)

    grid = _make_framed_grid(height, width)
    row = 1 + top_margin
    for idx in range(count):
        block_h = int(heights[idx])
        block_w = int(widths[idx])
        r0 = row
        r1 = row + block_h - 1
        c0 = int(rng.integers(2, width - block_w - 1))
        c1 = c0 + block_w - 1
        hole_row = int(rng.integers(r0, r1 + 1))

        if sides[idx] == "left":
            grid[hole_row, 0] = BG
        else:
            grid[hole_row, width - 1] = BG

        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]
        if idx + 1 < count:
            row = r1 + 1 + int(gaps[idx])

    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        bg = _most_common_color(grid)
        frame_color = _detect_frame_color(grid, bg)
        holes = _find_holes(grid, bg)
        if not holes:
            return False

        components, labels = _find_components(grid, bg, frame_color)
        if len(components) != len(holes):
            return False

        matched_ids: list[int] = []
        for side, row, col in holes:
            comp_id = _trace_to_component(grid, labels, bg, frame_color, side, row, col)
            if comp_id is None:
                return False
            matched_ids.append(comp_id)

        if len(set(matched_ids)) != len(components):
            return False

        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return False
    except Exception:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if bool(rng.integers(0, 2)):
            grid = _generate_vertical_case(rng)
        else:
            grid = _generate_horizontal_case(rng)
        if grid is not None and _validate_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
