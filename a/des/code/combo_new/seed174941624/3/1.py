from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
LEGEND_GAP = 1
GEN_RETRIES = 400

TEMPLATE_LIBRARY = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
)


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_color(patch: np.ndarray) -> int | None:
    if patch.shape != (3, 3) or int(patch[1, 1]) != BLACK:
        return None
    ring = [int(patch[r, c]) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    color = ring[0]
    if color not in MARKER_COLORS:
        return None
    if all(value == color for value in ring):
        return color
    return None


def _decode_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], set[tuple[int, int]]]:
    h, w = grid.shape
    mapping: dict[int, np.ndarray] = {}
    legend_cells: set[tuple[int, int]] = set()

    for r in range(h - 5):
        for c in range(w - 2):
            color = _ring_color(grid[r : r + 3, c : c + 3])
            if color is None:
                continue
            template = grid[r + 3 : r + 6, c : c + 3]
            if not set(map(int, template.ravel())).issubset({BLACK, MAROON}):
                continue
            mask = template == MAROON
            if not np.any(mask):
                continue
            previous = mapping.get(color)
            if previous is not None:
                assert np.array_equal(previous, mask), "legend color must map to one template"
            mapping[color] = mask.copy()
            for rr in range(r, r + 6):
                for cc in range(c, c + 3):
                    if int(grid[rr, cc]) != BLACK:
                        legend_cells.add((rr, cc))

    assert mapping, "expected at least one color-template legend pair"
    return mapping, legend_cells


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for cells in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(cells)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 7 or width < 7:
            continue
        border_area = 2 * height + 2 * width - 4
        if len(cells) != border_area:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))

    assert frames, "expected at least one grey frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner color legend, then replace each frame's scattered markers with the matching centered template.

    Input structure:
    - The background is black.
    - One corner contains a legend made of 3x3 color rings whose centers are black, each with a 3x3 maroon template directly beneath it.
    - Elsewhere there are one or more hollow grey rectangular frames, each containing 1-3 scattered marker pixels of a single non-grey color.

    Rule:
    - Read each legend pair as a mapping from ring color to the 3x3 template drawn underneath it.
    - For every grey frame, use the interior marker color to choose its template, clear the scattered markers, and redraw that template in the same color centered on the frame interior.
    - Erase the whole legend area to black while leaving the grey frames themselves unchanged.

    Key insight:
    - The marker pixels inside a frame do not encode shape by position or count; only their color matters, and that color must first be decoded through the separate legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    mapping, legend_cells = _decode_legend(g)
    frames = _find_frames(g)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = BLACK

    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere(np.isin(inner, MARKER_COLORS))
        assert 1 <= len(marker_coords) <= inner.size
        marker_colors = {int(inner[r, c]) for r, c in marker_coords}
        assert len(marker_colors) == 1, "each frame should contain one marker color"
        marker_color = marker_colors.pop()
        assert marker_color in mapping, "frame marker color must appear in legend"

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK

        template = mapping[marker_color]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        top = center_r - 1
        left = center_c - 1
        for dr in range(3):
            for dc in range(3):
                if template[dr, dc]:
                    out[top + dr, left + dc] = marker_color

    return out


def _legend_size(pair_count: int, legend_rows: int) -> tuple[int, int, int]:
    legend_cols = math.ceil(pair_count / legend_rows)
    height = legend_rows * 6 + (legend_rows - 1) * LEGEND_GAP
    width = legend_cols * 3 + (legend_cols - 1) * LEGEND_GAP
    return legend_rows, legend_cols, height, width


def _legend_origin(corner: str, grid_h: int, grid_w: int, legend_h: int, legend_w: int) -> tuple[int, int]:
    top = 0 if "T" in corner else grid_h - legend_h
    left = 0 if "L" in corner else grid_w - legend_w
    return top, left


def _paint_template(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for dr in range(3):
        for dc in range(3):
            if mask[dr, dc]:
                grid[top + dr, left + dc] = color


def _draw_legend(
    grid: np.ndarray,
    ordered_pairs: list[tuple[int, np.ndarray]],
    corner: str,
    legend_rows: int,
) -> tuple[int, int, int, int]:
    _, legend_cols, legend_h, legend_w = _legend_size(len(ordered_pairs), legend_rows)
    base_r, base_c = _legend_origin(corner, grid.shape[0], grid.shape[1], legend_h, legend_w)

    for index, (color, mask) in enumerate(ordered_pairs):
        block_row = index // legend_cols
        block_col = index % legend_cols
        top = base_r + block_row * (6 + LEGEND_GAP)
        left = base_c + block_col * (3 + LEGEND_GAP)
        for dr in range(3):
            for dc in range(3):
                if (dr, dc) != (1, 1):
                    grid[top + dr, left + dc] = color
        _paint_template(grid, mask, MAROON, top + 3, left)

    return base_r, base_c, base_r + legend_h - 1, base_c + legend_w - 1


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _place_frames(
    grid_h: int,
    grid_w: int,
    legend_bbox: tuple[int, int, int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    _occupy(occupied, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1)

    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for index in order:
        height, width = sizes[index]
        choices = [
            (top, left)
            for top in range(1, grid_h - height)
            for left in range(1, grid_w - width)
            if _can_place(occupied, top, left, height, width)
        ]
        if not choices:
            return None
        top, left = random.choice(choices)
        placements[index] = (top, left)
        _occupy(occupied, top, left, height, width)

    return [placement for placement in placements if placement is not None]


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    grid[top, left : left + outer_w] = GREY
    grid[top + outer_h - 1, left : left + outer_w] = GREY
    grid[top : top + outer_h, left] = GREY
    grid[top : top + outer_h, left + outer_w - 1] = GREY


def _choose_marker_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(40):
        positions = random.sample(cells, count)
        if count == 1:
            return positions
        rows = {r for r, _ in positions}
        cols = {c for _, c in positions}
        if len(rows) > 1 or len(cols) > 1:
            return positions
    return random.sample(cells, count)


def _render_expected(
    inp: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    frame_specs: list[dict[str, object]],
    mapping: dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    for spec in frame_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        color = int(spec["color"])
        out[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = BLACK

        center_r = top + 1 + (inner_h - 1) // 2
        center_c = left + 1 + (inner_w - 1) // 2
        _paint_template(out, mapping[color], color, center_r - 1, center_c - 1)

    return out


def _generate_case() -> np.ndarray | None:
    color_count = random.randint(2, 4)
    active_colors = random.sample(list(MARKER_COLORS), color_count)
    template_choices = random.sample(list(TEMPLATE_LIBRARY), color_count)
    mapping = {color: template.copy() for color, template in zip(active_colors, template_choices)}

    legend_rows = random.choice([1, 2]) if color_count >= 3 else 1
    if color_count == 2 and random.random() < 0.5:
        legend_rows = 2
    corner = random.choice(["TL", "TR", "BL", "BR"])
    _, _, legend_h, legend_w = _legend_size(color_count, legend_rows)

    frame_count = random.randint(color_count, 4)
    frame_colors = active_colors[:] + random.choices(active_colors, k=frame_count - color_count)
    random.shuffle(frame_colors)

    inner_max = {2: 10, 3: 8, 4: 7}[frame_count]
    frame_specs: list[dict[str, object]] = []
    outer_sizes: list[tuple[int, int]] = []
    for color in frame_colors:
        inner_h = random.randint(5, inner_max)
        inner_w = random.randint(5, inner_max)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        frame_specs.append({"inner_h": inner_h, "inner_w": inner_w, "color": color})
        outer_sizes.append((outer_h, outer_w))

    largest_h = max(height for height, _ in outer_sizes)
    largest_w = max(width for _, width in outer_sizes)
    grid_h_min = max(legend_h + 2, largest_h + 2, 16 + frame_count * 3)
    grid_w_min = max(legend_w + 2, largest_w + 2, 16 + frame_count * 3)
    if grid_h_min > 30 or grid_w_min > 30:
        return None

    grid_h = random.randint(grid_h_min, 30)
    grid_w = random.randint(grid_w_min, 30)
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)

    ordered_pairs = list(mapping.items())
    random.shuffle(ordered_pairs)
    legend_bbox = _draw_legend(inp, ordered_pairs, corner, legend_rows)

    placements = _place_frames(grid_h, grid_w, legend_bbox, outer_sizes)
    if placements is None:
        return None

    for spec, (top, left), (outer_h, outer_w) in zip(frame_specs, placements, outer_sizes):
        spec["top"] = top
        spec["left"] = left
        _draw_frame(inp, top, left, outer_h, outer_w)

        marker_count = random.randint(1, 3)
        positions = _choose_marker_positions(int(spec["inner_h"]), int(spec["inner_w"]), marker_count)
        for rr, cc in positions:
            inp[top + 1 + rr, left + 1 + cc] = int(spec["color"])

    expected = _render_expected(inp, legend_bbox, frame_specs, mapping)
    actual = main(inp)
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
