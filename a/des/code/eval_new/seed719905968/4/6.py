from __future__ import annotations

from collections import deque
import itertools

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = tuple(color for color in range(10) if color not in {BLACK, GREY})
MAX_GRID_SIZE = 30
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = [coord[0] for coord in coords]
            cols = [coord[1] for coord in coords]
            components.append(
                {
                    "color": int(color),
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _is_rectangular_frame(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for component in _find_components_of_color(grid, color):
            if _is_rectangular_frame(component):
                frames.append(component)

    assert len(frames) >= 2, "expected at least two nested frames"
    depths = [
        1 + sum(_contains(other["bbox"], frame["bbox"]) for other in frames if other is not frame)
        for frame in frames
    ]
    assert sorted(depths) == list(range(1, len(frames) + 1)), "frames must form a single nested chain"
    return [frame for _depth, frame in sorted(zip(depths, frames), key=lambda item: item[0])]


def _extract_core_pixels(grid: np.ndarray, inner_frame: dict) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = map(int, inner_frame["bbox"])
    core: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color != BLACK:
                core.append((r - r0, c - c0, color))
    return core


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _render_output_from_specs(frame_specs: list[dict], core_pixels: list[tuple[int, int, int]]) -> np.ndarray:
    outer_h = _bbox_height(frame_specs[0]["bbox"])
    widths = [_bbox_width(frame["bbox"]) for frame in frame_specs]
    out_w = sum(widths) + len(widths) - 1
    out = np.full((outer_h, out_w), BLACK, dtype=int)

    x = 0
    inner_anchor: tuple[int, int] | None = None
    for depth, frame in enumerate(frame_specs, start=1):
        height = _bbox_height(frame["bbox"])
        width = _bbox_width(frame["bbox"])
        top = (outer_h - height) // 2
        if height > 2 and width > 2 and depth % 2 == 1:
            out[top + 1 : top + height - 1, x + 1 : x + width - 1] = GREY
        _draw_frame(out, top, x, height, width, int(frame["color"]))
        if depth == len(frame_specs):
            inner_anchor = (top, x)
        x += width + 1

    assert inner_anchor is not None
    for dr, dc, color in core_pixels:
        out[inner_anchor[0] + int(dr), inner_anchor[1] + int(dc)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unnest a chain of rectangular frames into a left-to-right display whose interiors depend on depth parity.

    Input structure:
    - The background is black.
    - A single group of distinct-colored, one-pixel-thick rectangular frames is nested by strict containment, with at least one black cell between consecutive frames.
    - The innermost frame may contain a small non-black core pattern inside its interior.

    Rule:
    - Identify the frames from the outermost container to the innermost contained frame.
    - Place those frames side by side from left to right, separated by one black column, while preserving each frame's original size and border color.
    - Fill odd-depth frame interiors with grey and leave even-depth frame interiors black.
    - Preserve the innermost core pattern at the same position relative to that innermost frame.

    Key insight:
    - The transformation depends on recovering the nesting order from geometric containment first, then applying a separate odd/even depth rule to each extracted frame.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    core_pixels = _extract_core_pixels(g, frames[-1])
    return _render_output_from_specs(frames, core_pixels)


def _build_axis_sizes(inner_size: int, gaps: tuple[int, ...]) -> list[int]:
    sizes = [int(inner_size)]
    for gap in reversed(gaps):
        sizes.append(sizes[-1] + 2 * (int(gap) + 1))
    return list(reversed(sizes))


def _feasible_structures() -> dict[int, list[dict]]:
    structures: dict[int, list[dict]] = {2: [], 3: []}
    for layer_count in (2, 3):
        for inner_h in range(3, 12):
            for gap_rows in itertools.product((1, 2, 3), repeat=layer_count - 1):
                heights = _build_axis_sizes(inner_h, gap_rows)
                if heights[0] > MAX_GRID_SIZE - 2:
                    continue
                for inner_w in range(3, 12):
                    for gap_cols in itertools.product((1, 2, 3), repeat=layer_count - 1):
                        widths = _build_axis_sizes(inner_w, gap_cols)
                        if widths[0] > MAX_GRID_SIZE - 2:
                            continue
                        if sum(widths) + layer_count - 1 > MAX_GRID_SIZE:
                            continue
                        structures[layer_count].append(
                            {
                                "layer_count": layer_count,
                                "gap_rows": tuple(int(x) for x in gap_rows),
                                "gap_cols": tuple(int(x) for x in gap_cols),
                                "heights": tuple(int(x) for x in heights),
                                "widths": tuple(int(x) for x in widths),
                            }
                        )
    return structures


# Four or more layers cannot fit after the horizontal unnesting step under the ARC 30x30 cap:
# even the minimum-width chain 15/11/7/3 already needs 39 columns with separators.
STRUCTURES_BY_LAYER_COUNT = _feasible_structures()


def _sample_margin_pair(rng: np.random.Generator, used: int) -> tuple[int, int]:
    remaining = MAX_GRID_SIZE - int(used)
    assert remaining >= 2
    total = int(rng.integers(2, remaining + 1))
    before = int(rng.integers(1, total))
    after = total - before
    return before, after


def _sample_connected_shape(rng: np.random.Generator, height: int, width: int, area: int) -> list[tuple[int, int]]:
    start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _render_input(structure: dict, frame_colors: list[int], core_color: int | None, core_cells: list[tuple[int, int]], top: int, left: int) -> tuple[np.ndarray, list[dict], list[tuple[int, int, int]]]:
    heights = structure["heights"]
    widths = structure["widths"]
    gap_rows = structure["gap_rows"]
    gap_cols = structure["gap_cols"]

    grid_h = heights[0] + top
    grid_w = widths[0] + left
    frame_specs: list[dict] = []

    r0 = top
    c0 = left
    for depth, color in enumerate(frame_colors):
        height = int(heights[depth])
        width = int(widths[depth])
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        frame_specs.append({"bbox": bbox, "color": int(color)})
        if depth + 1 < len(frame_colors):
            r0 += int(gap_rows[depth]) + 1
            c0 += int(gap_cols[depth]) + 1

    grid_h += 0
    grid_w += 0
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for frame in frame_specs:
        height = _bbox_height(frame["bbox"])
        width = _bbox_width(frame["bbox"])
        _draw_frame(grid, int(frame["bbox"][0]), int(frame["bbox"][1]), height, width, int(frame["color"]))

    core_pixels: list[tuple[int, int, int]] = []
    if core_color is not None:
        inner = frame_specs[-1]
        ir0, ic0, _ir1, _ic1 = map(int, inner["bbox"])
        for rr, cc in core_cells:
            abs_r = ir0 + 1 + int(rr)
            abs_c = ic0 + 1 + int(cc)
            grid[abs_r, abs_c] = int(core_color)
            core_pixels.append((1 + int(rr), 1 + int(cc), int(core_color)))

    return grid, frame_specs, core_pixels


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layer_count = 2 if bool(rng.integers(0, 2)) else 3
    structure_pool = STRUCTURES_BY_LAYER_COUNT[layer_count]
    structure = structure_pool[int(rng.integers(0, len(structure_pool)))]

    outer_h = int(structure["heights"][0])
    outer_w = int(structure["widths"][0])
    top_margin, bottom_margin = _sample_margin_pair(rng, outer_h)
    left_margin, right_margin = _sample_margin_pair(rng, outer_w)
    top = top_margin
    left = left_margin

    has_core = bool(rng.integers(0, 2))
    color_count = layer_count + (1 if has_core else 0)
    chosen = rng.choice(np.array(VISIBLE_COLORS, dtype=int), size=color_count, replace=False)
    frame_colors = [int(color) for color in chosen[:layer_count]]
    core_color = int(chosen[-1]) if has_core else None

    core_cells: list[tuple[int, int]] = []
    if has_core:
        inner_h = int(structure["heights"][-1]) - 2
        inner_w = int(structure["widths"][-1]) - 2
        area = int(rng.integers(1, min(4, inner_h * inner_w) + 1))
        core_cells = _sample_connected_shape(rng, inner_h, inner_w, area)

    minimal_grid, frame_specs, core_pixels = _render_input(
        structure, frame_colors, core_color, core_cells, top, left
    )
    grid = np.full(
        (outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin),
        BLACK,
        dtype=int,
    )
    grid[: minimal_grid.shape[0], : minimal_grid.shape[1]] = minimal_grid

    expected = _render_output_from_specs(frame_specs, core_pixels)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
