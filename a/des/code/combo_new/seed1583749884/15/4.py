from __future__ import annotations

import numpy as np


LEGEND_COLORS = (2, 7, 4, 3, 1)  # red, orange, yellow, green, blue
FRAME_COLOR = 5
BACKGROUND_COLOR = 0
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": int(color), "coords": coords, "bbox": _bbox(coords)})

    return comps


def _is_solid_rect(bbox: tuple[int, int, int, int], coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_rect_frame(bbox: tuple[int, int, int, int], coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False

    coord_set = set(coords)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr == r0 or rr == r1 or cc == c0 or cc == c1
            if on_border != ((rr, cc) in coord_set):
                return False
    return True


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_frame_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        comps = _components_of_color(grid, color)
        if len(comps) < 2:
            continue
        if all(_is_rect_frame(comp["bbox"], comp["coords"]) for comp in comps):
            candidates.append((len(comps), color))

    assert candidates, "expected one repeated frame color"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _legend_colors_from_grid(grid: np.ndarray, bg: int, frame_color: int) -> list[int]:
    blocks: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (int(bg), int(frame_color)):
            continue
        for comp in _components_of_color(grid, color):
            if _is_solid_rect(comp["bbox"], comp["coords"]):
                blocks.append(comp)

    blocks.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    colors = [int(comp["color"]) for comp in blocks]
    assert len(colors) >= 5, "expected a five-block legend"
    return colors[:5]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames by reading the left legend from top to bottom.

    Input structure:
    - The background is black.
    - A vertical legend of five filled color blocks sits on the left side.
    - On the right side, one group of disconnected one-pixel-thick rectangular frames shares a single placeholder color.

    Rule:
    - Read the legend colors from top to bottom as the colors for nesting depths 1 through 5.
    - Find every placeholder-colored rectangular frame and determine its depth by counting how many larger frames contain its bounding box.
    - Recolor each frame with the legend color for its depth, leaving the inner hole and all other background pixels unchanged.

    Key insight:
    - The frame color does not encode depth; depth comes purely from geometric containment, and the legend converts that spatial depth into output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color = _find_frame_color(g, bg)
    legend_colors = _legend_colors_from_grid(g, bg, frame_color)

    frame_comps = [
        comp
        for comp in _components_of_color(g, frame_color)
        if _is_rect_frame(comp["bbox"], comp["coords"])
    ]
    bboxes = [comp["bbox"] for comp in frame_comps]

    out = g.copy()
    for comp in frame_comps:
        depth = 1 + sum(_contains(other, comp["bbox"]) for other in bboxes)
        assert 1 <= depth <= len(legend_colors), "frame depth exceeds legend length"
        new_color = int(legend_colors[depth - 1])
        for r, c in comp["coords"]:
            out[r, c] = new_color

    return out


def _draw_filled_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _sample_local_frame_bboxes(rng: np.random.Generator, depth: int) -> list[tuple[int, int, int, int]]:
    min_dim = 3 + 4 * (depth - 1)
    outer_h = int(rng.integers(min_dim, min(24, min_dim + 5) + 1))
    outer_w = int(rng.integers(min_dim, min(20, min_dim + 7) + 1))

    r0 = c0 = 0
    r1 = outer_h - 1
    c1 = outer_w - 1
    bboxes = [(r0, c0, r1, c1)]

    for level in range(depth - 1):
        remaining_inner = depth - level - 2
        min_next_h = 3 + 4 * remaining_inner
        min_next_w = 3 + 4 * remaining_inner

        next_h = int(rng.integers(min_next_h, (r1 - r0 + 1) - 4 + 1))
        next_w = int(rng.integers(min_next_w, (c1 - c0 + 1) - 4 + 1))

        removed_h = (r1 - r0 + 1) - next_h
        removed_w = (c1 - c0 + 1) - next_w

        top_gap = int(rng.integers(2, removed_h - 2 + 1))
        left_gap = int(rng.integers(2, removed_w - 2 + 1))
        bottom_gap = removed_h - top_gap
        right_gap = removed_w - left_gap

        r0 += top_gap
        c0 += left_gap
        r1 -= bottom_gap
        c1 -= right_gap
        bboxes.append((r0, c0, r1, c1))

    return bboxes


def _expected_output(inp: np.ndarray, shifted_bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for depth, bbox in enumerate(shifted_bboxes, start=1):
        _draw_frame(out, bbox, LEGEND_COLORS[depth - 1])
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    depth = int(rng.integers(2, 6))
    local_bboxes = _sample_local_frame_bboxes(rng, depth)
    outer_h = local_bboxes[0][2] + 1
    outer_w = local_bboxes[0][3] + 1

    legend_w = int(rng.integers(1, 4))
    legend_heights = [int(rng.integers(1, 4)) for _ in LEGEND_COLORS]
    legend_gap = int(rng.integers(1, 3))
    legend_h = sum(legend_heights) + legend_gap * (len(LEGEND_COLORS) - 1)

    min_sep = int(rng.integers(2, 6))
    min_w = legend_w + min_sep + outer_w
    min_h = max(legend_h, outer_h)
    if min_w > 30 or min_h > 30:
        return None

    width = int(rng.integers(min_w, min(30, min_w + 6) + 1))
    height = int(rng.integers(min_h, min(30, min_h + 6) + 1))

    legend_left = int(rng.integers(0, width - min_w + 1))
    frame_left_min = legend_left + legend_w + min_sep
    frame_left = int(rng.integers(frame_left_min, width - outer_w + 1))
    legend_top = int(rng.integers(0, height - legend_h + 1))
    frame_top = int(rng.integers(0, height - outer_h + 1))

    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)

    cursor = legend_top
    for color, block_h in zip(LEGEND_COLORS, legend_heights, strict=False):
        _draw_filled_rect(inp, (cursor, legend_left, cursor + block_h - 1, legend_left + legend_w - 1), color)
        cursor += block_h + legend_gap

    shifted_bboxes: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1 in local_bboxes:
        shifted = (frame_top + r0, frame_left + c0, frame_top + r1, frame_left + c1)
        shifted_bboxes.append(shifted)
        _draw_frame(inp, shifted, FRAME_COLOR)

    expected = _expected_output(inp, shifted_bboxes)
    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(out, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
