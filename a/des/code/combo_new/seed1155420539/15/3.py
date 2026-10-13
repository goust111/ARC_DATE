import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
EDGES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300


def _rect_outline(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _edge_cells(bbox: tuple[int, int, int, int], edge: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if edge == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if edge == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if edge == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if edge == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown edge: {edge}")


def _panel_spans(width: int, sep_cols: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for col in sep_cols:
        spans.append((start, col))
        start = col + 1
    spans.append((start, width))
    return spans


def _find_separators(grid: np.ndarray) -> tuple[int, list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    del h
    by_color: dict[int, list[int]] = {}
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            by_color.setdefault(int(grid[0, c]), []).append(c)

    for color, cols in by_color.items():
        if color == BLACK or not (1 <= len(cols) <= 3):
            continue
        spans = _panel_spans(w, cols)
        widths = [c1 - c0 for c0, c1 in spans]
        if min(widths) < 5 or len(set(widths)) != 1:
            continue
        if all(c0 < c1 for c0, c1 in spans):
            return color, cols, spans

    raise ValueError("could not identify separator columns")


def _find_frame(panel: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    for color in map(int, np.unique(panel)):
        if color == BLACK:
            continue
        coords = np.argwhere(panel == color)
        if coords.shape[0] < 8:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        outline = set(_rect_outline((r0, c0, r1, c1)))
        points = {tuple(map(int, rc)) for rc in coords}
        if outline != points:
            continue
        if np.all(panel[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
            return color, (r0, c0, r1, c1)
    raise ValueError("could not identify the shared frame")


def _marker_edge(r: int, c: int, bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge panel-local edge color hints into one reconstructed colored rectangle outline.

    Input structure:
    - Black background, with 1-cell-wide full-height separator columns splitting the grid into 2-4 equal-width panels.
    - Each panel contains the same 1-pixel-thick rectangular outline and some colored single-pixel markers placed just outside that outline.
    - Every marker is 4-neighbor-adjacent to exactly one side of the rectangle and indicates a color for that side.

    Rule:
    - Detect the separator columns to recover the individual panels, then locate the shared rectangular outline inside each panel.
    - For every marker, identify which side of the rectangle it touches and recolor that entire side in the marker's color.
    - Merge the panels from left to right into one single-panel output: later panels overwrite earlier colors only where they actively recolor outline pixels, while untouched outline pixels stay grey and all non-outline pixels stay black.

    Key insight:
    - The panels are not independent scenes; they are distributed partial encodings of one rectangle whose edge colors must be combined in panel order.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    sep_color, sep_cols, spans = _find_separators(g)
    del sep_cols
    panels = [g[:, c0:c1] for c0, c1 in spans]

    frame_color, bbox = _find_frame(panels[0])
    out = np.full(panels[0].shape, BLACK, dtype=int)
    for r, c in _rect_outline(bbox):
        out[r, c] = frame_color

    for panel in panels:
        panel_frame_color, panel_bbox = _find_frame(panel)
        if panel_bbox != bbox or panel_frame_color != frame_color:
            raise ValueError("panels do not share the same frame")

        marker_coords = np.argwhere((panel != BLACK) & (panel != sep_color) & (panel != frame_color))
        for r, c in marker_coords:
            rr, cc = map(int, (r, c))
            edge = _marker_edge(rr, cc, bbox)
            if edge is None:
                raise ValueError("found a non-frame pixel that is not a valid marker")
            color = int(panel[rr, cc])
            for pr, pc in _edge_cells(bbox, edge):
                out[pr, pc] = color

    return out


def _opposite_pairs(edges: list[str]) -> list[tuple[str, str]]:
    pairs = []
    edge_set = set(edges)
    if {"top", "bottom"} <= edge_set:
        pairs.append(("top", "bottom"))
    if {"left", "right"} <= edge_set:
        pairs.append(("left", "right"))
    return pairs


def _assign_panel_edges(panel_count: int) -> list[list[str]]:
    # When one panel carries two markers, keep them on opposite sides to avoid
    # an unspecified same-panel corner conflict.
    if panel_count == 4:
        groups = [[edge] for edge in random.sample(list(EDGES), 4)]
        random.shuffle(groups)
        return groups

    if panel_count == 3:
        total_edges = random.choice((3, 4))
        chosen = random.sample(list(EDGES), total_edges)
        if total_edges == 3:
            groups = [[edge] for edge in chosen]
        else:
            pair = list(random.choice(_opposite_pairs(chosen)))
            singles = [[edge] for edge in chosen if edge not in pair]
            groups = [pair] + singles
        random.shuffle(groups)
        return groups

    total_edges = random.choice((2, 3, 4))
    chosen = random.sample(list(EDGES), total_edges)
    if total_edges == 2:
        groups = [[chosen[0]], [chosen[1]]]
    elif total_edges == 3:
        pair = list(random.choice(_opposite_pairs(chosen)))
        single = next(edge for edge in chosen if edge not in pair)
        groups = [pair, [single]]
    else:
        groups = [["top", "bottom"], ["left", "right"]]
        if random.random() < 0.5:
            groups.reverse()
    random.shuffle(groups)
    return groups


def _random_bbox(panel_h: int, panel_w: int) -> tuple[int, int, int, int]:
    rect_h = random.randint(3, panel_h - 2)
    rect_w = random.randint(3, panel_w - 2)
    r0 = random.randint(1, panel_h - rect_h - 1)
    c0 = random.randint(1, panel_w - rect_w - 1)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _marker_position(bbox: tuple[int, int, int, int], edge: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if edge in ("top", "bottom"):
        cols = list(range(c0, c1 + 1))
        if len(cols) > 2 and random.random() < 0.75:
            cols = cols[1:-1]
        c = random.choice(cols)
        return (r0 - 1, c) if edge == "top" else (r1 + 1, c)

    rows = list(range(r0, r1 + 1))
    if len(rows) > 2 and random.random() < 0.75:
        rows = rows[1:-1]
    r = random.choice(rows)
    return (r, c0 - 1) if edge == "left" else (r, c1 + 1)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    panel_count = random.randint(2, 4)
    max_panel_w = (30 - (panel_count - 1)) // panel_count
    panel_w = random.randint(5, max_panel_w)
    panel_h = random.randint(6, 15)
    bbox = _random_bbox(panel_h, panel_w)
    panel_edges = _assign_panel_edges(panel_count)

    all_edges = [edge for group in panel_edges for edge in group]
    edge_colors = dict(zip(all_edges, random.sample(MARKER_COLORS, len(all_edges))))

    total_w = panel_count * panel_w + (panel_count - 1)
    grid = np.full((panel_h, total_w), BLACK, dtype=int)
    expected = np.full((panel_h, panel_w), BLACK, dtype=int)
    for r, c in _rect_outline(bbox):
        expected[r, c] = GREY

    starts = []
    cursor = 0
    for panel_idx in range(panel_count):
        starts.append(cursor)
        cursor += panel_w
        if panel_idx < panel_count - 1:
            grid[:, cursor] = AZURE
            cursor += 1

    outline = _rect_outline(bbox)
    for panel_idx, start in enumerate(starts):
        panel = grid[:, start : start + panel_w]
        for r, c in outline:
            panel[r, c] = GREY

        used_positions = set()
        for edge in panel_edges[panel_idx]:
            marker = _marker_position(bbox, edge)
            if marker in used_positions:
                raise AssertionError("marker placement collided")
            used_positions.add(marker)
            panel[marker] = edge_colors[edge]
            for r, c in _edge_cells(bbox, edge):
                expected[r, c] = edge_colors[edge]

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    grid, expected = _build_case()
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-merge puzzle")
