from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _panel_start(grid: np.ndarray) -> int:
    grey_rows = np.flatnonzero(np.any(grid == GREY, axis=1))
    assert grey_rows.size > 0, "expected a grey lower panel"
    start = int(grey_rows[0])
    assert start in (1, 2), "instruction strip must have height 1 or 2"
    assert np.all(grid[:start] != GREY), "grey should only appear in the lower panel"
    return start


def _markers(top: np.ndarray) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    for row in range(top.shape[0]):
        cols = np.flatnonzero(top[row] != BLACK)
        assert len(cols) <= 1, "expected at most one marker per instruction-strip row"
        if len(cols) == 0:
            continue
        color = int(top[row, int(cols[0])])
        assert color not in seen_colors, "marker colors must be unique"
        seen_colors.add(color)
        markers.append((int(row), color))
    assert markers, "expected at least one marker"
    return markers


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the scene objects named by the top markers, then turn those markers into left-aligned count bars.

    Input structure:
    - Black fills the top background and the instruction strip is 1-2 rows tall.
    - Each instruction-strip row contains at most one isolated non-black marker pixel, and marker colors are unique.
    - The rows below form a full-width rectangular scene panel with Grey background and scattered 4-connected monochrome objects.

    Rule:
    - Read the marker colors from the top strip.
    - In the lower panel, preserve every object whose color matches a marker color and recolor all other non-Grey pixels to Grey.
    - For each marker row, replace the marker pixel by a horizontal bar of that marker color starting at column 0.
    - The bar length equals the number of 4-connected lower-panel objects of that color; when that count is 0, the row stays black.

    Key insight:
    - The top strip is both a color filter and a count display: it selects which lower-panel objects survive, then reports their per-color counts as bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start = _panel_start(g)
    top = g[:start]
    panel = g[start:]
    markers = _markers(top)
    marker_colors = {color for _row, color in markers}

    out = np.full_like(g, BLACK)
    cleaned = panel.copy()
    for color in set(map(int, np.unique(panel))) - marker_colors - {GREY}:
        cleaned[cleaned == color] = GREY
    out[start:] = cleaned

    width = g.shape[1]
    for row, color in markers:
        count = _component_count(panel == color)
        assert count <= width, "count bar must fit the grid width"
        out[row, :count] = color
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _polyomino(area: int) -> np.ndarray:
    canvas = np.zeros((7, 7), dtype=bool)
    cells = {(3, 3)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    patch = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
    for r, c in cells:
        patch[r - min(rows), c - min(cols)] = True
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _shape_mask() -> np.ndarray:
    kind = random.choice(("dot", "line", "rect", "poly"))
    if kind == "dot":
        return np.ones((1, 1), dtype=bool)
    if kind == "line":
        length = random.randint(2, 4)
        patch = np.ones((1, length), dtype=bool)
        return patch if random.random() < 0.5 else patch.T
    if kind == "rect":
        h, w = random.choice(((2, 2), (2, 3), (3, 2)))
        return np.ones((h, w), dtype=bool)
    return _polyomino(random.randint(3, 6))


def _place_objects(panel_shape: tuple[int, int], objects: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    panel = np.full(panel_shape, GREY, dtype=int)
    occupied = np.zeros(panel_shape, dtype=bool)

    order = sorted(range(len(objects)), key=lambda i: int(objects[i][1].sum()), reverse=True)
    for idx in order:
        color, mask = objects[idx]
        mh, mw = mask.shape
        ph, pw = panel_shape
        if mh > ph - 2 or mw > pw - 2:
            return None

        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, ph - mh):
            for c0 in range(1, pw - mw):
                if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        rows, cols = np.nonzero(mask)
        panel[r0 + rows, c0 + cols] = color
        occupied[r0 + rows, c0 + cols] = True

    if not np.all(np.any(panel == GREY, axis=1)):
        return None
    return panel


def _marker_columns(width: int, count: int) -> list[int]:
    cols = []
    for _ in range(count):
        if width == 1:
            cols.append(0)
        elif random.random() < 0.75:
            cols.append(random.randrange(1, width))
        else:
            cols.append(random.randrange(width))
    return cols


def _marker_counts(top_h: int, width: int) -> list[int]:
    zero_branch = random.random() < (0.25 if top_h == 1 else 0.5)
    positive_max = min(width - 2, 4)
    assert positive_max >= 1

    if top_h == 1:
        return [0 if zero_branch else random.randint(1, positive_max)]

    counts = [random.randint(1, positive_max) for _ in range(top_h)]
    if zero_branch:
        counts[random.randrange(top_h)] = 0
    return counts


def _distractor_colors(used: set[int], total: int) -> list[int]:
    pool = [color for color in PALETTE if color not in used]
    color_count = random.randint(1, min(2, total, len(pool)))
    colors = random.sample(pool, color_count)
    out = colors[:]
    while len(out) < total:
        out.append(random.choice(colors))
    random.shuffle(out)
    return out


def _expected_output(
    top_h: int,
    width: int,
    panel: np.ndarray,
    marker_colors: list[int],
    counts: list[int],
) -> np.ndarray:
    out = np.full((top_h + panel.shape[0], width), BLACK, dtype=int)
    cleaned = panel.copy()
    for color in set(map(int, np.unique(panel))) - set(marker_colors) - {GREY}:
        cleaned[cleaned == color] = GREY
    out[top_h:] = cleaned
    for row, (color, count) in enumerate(zip(marker_colors, counts)):
        out[row, :count] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.choice((1, 2, 2))
    width = random.randint(8, 15)
    panel_h = random.randint(7, min(14, 30 - top_h))

    marker_colors = random.sample(PALETTE, top_h)
    counts = _marker_counts(top_h, width)
    used_colors = set(marker_colors)

    distractor_total = random.randint(1, 4 if sum(counts) == 0 else 3)
    matched_objects = [(color, _shape_mask()) for color, count in zip(marker_colors, counts) for _ in range(count)]
    distractor_objects = [(color, _shape_mask()) for color in _distractor_colors(used_colors, distractor_total)]
    objects = matched_objects + distractor_objects
    random.shuffle(objects)

    panel = _place_objects((panel_h, width), objects)
    if panel is None:
        return None

    inp = np.full((top_h + panel_h, width), BLACK, dtype=int)
    inp[top_h:] = panel
    for row, (col, color) in enumerate(zip(_marker_columns(width, top_h), marker_colors)):
        inp[row, col] = color

    expected = _expected_output(top_h, width, panel, marker_colors, counts)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
