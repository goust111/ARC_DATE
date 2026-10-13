from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GENERATION_ATTEMPTS = 300

LEVEL_RANGE = (2, 4)
INNER_SIZE_RANGE = (5, 8)
EXPANSION_RANGE = (2, 3)
LEGEND_GAP_RANGE = (1, 3)
LEGEND_SPACING_RANGE = (1, 2)
CANVAS_EXTRA_WIDTH = 4
CANVAS_EXTRA_TOP = 3

USABLE_COLORS = tuple(color for color in range(1, 10) if color != YELLOW)
PATTERN_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_component(
    g: np.ndarray,
    start_r: int,
    start_c: int,
    seen: np.ndarray,
    active_rows: int,
) -> tuple[int, np.ndarray]:
    color = int(g[start_r, start_c])
    stack = [(int(start_r), int(start_c))]
    seen[start_r, start_c] = True
    coords: list[tuple[int, int]] = []
    h, w = g.shape

    while stack:
        r, c = stack.pop()
        coords.append((int(r), int(c)))
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < active_rows and 0 <= nc < w):
                continue
            if seen[nr, nc] or int(g[nr, nc]) != color:
                continue
            seen[nr, nc] = True
            stack.append((int(nr), int(nc)))

    return color, np.array(coords, dtype=int)


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    expected_count = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_count:
        return None

    border = set()
    for cc in range(c0, c1 + 1):
        border.add((r0, cc))
        border.add((r1, cc))
    for rr in range(r0, r1 + 1):
        border.add((rr, c0))
        border.add((rr, c1))

    coord_set = {tuple(map(int, cell)) for cell in coords}
    if coord_set != border:
        return None
    return r0, c0, r1, c1


def _find_frames(g: np.ndarray) -> list[dict]:
    active_rows = int(g.shape[0] - 1)
    seen = np.zeros((active_rows, g.shape[1]), dtype=bool)
    frames: list[dict] = []

    for r in range(active_rows):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) == BLACK:
                continue
            color, coords = _extract_component(g, r, c, seen, active_rows)
            bbox = _frame_bbox(coords)
            if bbox is None:
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": tuple(map(int, bbox)),
                    "area": int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)),
                }
            )

    assert frames, "expected at least one rectangular frame"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(inner[2]) < int(outer[2])
        and int(inner[3]) < int(outer[3])
    )


def _legend_colors(g: np.ndarray) -> list[int]:
    colors = [int(value) for value in g[-1, :].tolist() if int(value) != BLACK]
    assert colors, "expected a bottom legend row"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular containers according to legend colors and depth parity.

    Input structure:
    - Black is the background.
    - The grid contains one-pixel-thick rectangular frames in distinct colors, with frames nested inside other frames.
    - The innermost frame contains a small connected Yellow motif, and the last row is a legend made of isolated non-black color pixels separated by background cells.

    Rule:
    - Detect every rectangular frame and compute its nesting depth from containment, with outermost frames at depth 0.
    - Read the legend on the bottom row from left to right so the kth legend color is the fill color for depth k.
    - For even depths, fill every background cell in that frame's interior except regions occupied by deeper nested frame boxes; for odd depths, color only the four interior corner cells that remain available. Keep all frame borders and the Yellow motif unchanged.

    Key insight:
    The transformation combines three linked cues: rectangle containment defines depth, the bottom legend maps depth to color, and depth parity chooses between full-region filling and corner-only marking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    legend = _legend_colors(g)
    frames = _find_frames(g)

    for frame in frames:
        frame["depth"] = sum(
            1
            for other in frames
            if other is not frame and _contains(other["bbox"], frame["bbox"])
        )

    max_depth = max(int(frame["depth"]) for frame in frames)
    assert max_depth < len(legend), "legend must provide a color for every depth"

    base_bg = g == BLACK
    out = g.copy()
    frames.sort(key=lambda frame: (int(frame["depth"]), -int(frame["area"])))

    for frame in frames:
        depth = int(frame["depth"])
        fill_color = int(legend[depth])
        r0, c0, r1, c1 = map(int, frame["bbox"])

        available = np.zeros_like(base_bg, dtype=bool)
        available[r0 + 1 : r1, c0 + 1 : c1] = True
        for other in frames:
            if other is frame:
                continue
            if _contains(frame["bbox"], other["bbox"]):
                rr0, cc0, rr1, cc1 = map(int, other["bbox"])
                available[rr0 : rr1 + 1, cc0 : cc1 + 1] = False

        target = available & base_bg
        if depth % 2 == 0:
            out[target] = fill_color
            continue

        corners = {
            (r0 + 1, c0 + 1),
            (r0 + 1, c1 - 1),
            (r1 - 1, c0 + 1),
            (r1 - 1, c1 - 1),
        }
        for rr, cc in corners:
            if target[rr, cc]:
                out[rr, cc] = fill_color

    return out


def _normalize_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    min_r = min(int(b[0]) for b in bboxes)
    min_c = min(int(b[1]) for b in bboxes)
    return [
        (int(r0 - min_r), int(c0 - min_c), int(r1 - min_r), int(c1 - min_c))
        for r0, c0, r1, c1 in bboxes
    ]


def _sample_nested_chain(levels: int) -> list[tuple[int, int, int, int]] | None:
    inner_h = random.randint(*INNER_SIZE_RANGE)
    inner_w = random.randint(*INNER_SIZE_RANGE)
    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(int(levels) - 1):
        r0, c0, r1, c1 = bboxes[-1]
        top = random.randint(*EXPANSION_RANGE)
        bottom = random.randint(*EXPANSION_RANGE)
        left = random.randint(*EXPANSION_RANGE)
        right = random.randint(*EXPANSION_RANGE)
        bboxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))

    normalized = _normalize_bboxes(bboxes)
    outer = normalized[-1]
    outer_h = int(outer[2] - outer[0] + 1)
    outer_w = int(outer[3] - outer[1] + 1)
    if outer_h > 28 or outer_w > 30:
        return None
    return list(reversed(normalized))


def _draw_frame(g: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    g[r0, c0 : c1 + 1] = int(color)
    g[r1, c0 : c1 + 1] = int(color)
    g[r0 : r1 + 1, c0] = int(color)
    g[r0 : r1 + 1, c1] = int(color)


def _legend_positions(width: int, n_colors: int) -> list[int] | None:
    gaps = [random.randint(*LEGEND_SPACING_RANGE) for _ in range(int(n_colors) - 1)]
    legend_w = int(n_colors + sum(gaps))
    if legend_w > int(width):
        return None

    start = random.randint(0, int(width) - legend_w)
    positions = [int(start)]
    for gap in gaps:
        positions.append(int(positions[-1] + 1 + gap))
    return positions


def _sample_pattern(interior_h: int, interior_w: int) -> np.ndarray:
    candidates = [
        shape
        for shape in PATTERN_SHAPES
        if int(shape.shape[0]) <= int(interior_h) and int(shape.shape[1]) <= int(interior_w)
    ]
    assert candidates, "innermost interior must fit a yellow motif"
    return random.choice(candidates).copy()


def _pattern_anchor(span: int, size: int) -> int:
    if span - size >= 2:
        return random.randint(1, int(span - size - 1))
    return random.randint(0, int(span - size))


def _place_pattern(g: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    interior_h = int(r1 - r0 - 1)
    interior_w = int(c1 - c0 - 1)
    pattern = _sample_pattern(interior_h, interior_w)
    pr = _pattern_anchor(interior_h, int(pattern.shape[0]))
    pc = _pattern_anchor(interior_w, int(pattern.shape[1]))
    view = g[r0 + 1 + pr : r0 + 1 + pr + pattern.shape[0], c0 + 1 + pc : c0 + 1 + pc + pattern.shape[1]]
    view[pattern] = YELLOW


def _build_grid(levels: int, frame_colors: list[int], legend_colors: list[int]) -> np.ndarray | None:
    bboxes = _sample_nested_chain(levels)
    if bboxes is None:
        return None

    outer_r0, outer_c0, outer_r1, outer_c1 = map(int, bboxes[0])
    outer_h = int(outer_r1 - outer_r0 + 1)
    outer_w = int(outer_c1 - outer_c0 + 1)
    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    max_top = 30 - (outer_h + legend_gap + 1)
    if max_top < 0:
        return None

    extra_top = random.randint(0, min(CANVAS_EXTRA_TOP, int(max_top)))
    height = int(outer_h + legend_gap + 1 + extra_top)

    min_width = max(int(outer_w), levels * 2 - 1)
    width = int(min_width + random.randint(0, min(CANVAS_EXTRA_WIDTH, 30 - min_width)))
    legend_positions = _legend_positions(width, levels)
    if legend_positions is None:
        return None

    g = np.zeros((height, width), dtype=int)
    frame_col = random.randint(0, int(width - outer_w))
    frame_row = int(extra_top)

    shifted = []
    for bbox, color in zip(bboxes, frame_colors, strict=False):
        r0, c0, r1, c1 = map(int, bbox)
        placed = (r0 + frame_row, c0 + frame_col, r1 + frame_row, c1 + frame_col)
        shifted.append(placed)
        _draw_frame(g, placed, int(color))

    _place_pattern(g, shifted[-1])
    for pos, color in zip(legend_positions, legend_colors, strict=False):
        g[-1, int(pos)] = int(color)
    return g


def _valid_input(g: np.ndarray, levels: int) -> bool:
    try:
        frames = _find_frames(g)
        depths = [
            sum(1 for other in frames if other is not frame and _contains(other["bbox"], frame["bbox"]))
            for frame in frames
        ]
        if len(frames) != int(levels):
            return False
        if sorted(depths) != list(range(int(levels))):
            return False
        out = main(g)
    except Exception:
        return False
    return not np.array_equal(out, g)


def _gen_attempt() -> np.ndarray | None:
    levels = random.randint(*LEVEL_RANGE)
    colors = random.sample(list(USABLE_COLORS), 2 * int(levels))
    frame_colors = colors[:levels]
    legend_colors = colors[levels:]
    grid = _build_grid(levels, frame_colors, legend_colors)
    if grid is None:
        return None
    if not _valid_input(grid, levels):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-container ARC input")
