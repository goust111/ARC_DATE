from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _transform(mask: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return mask
    if corner == "tr":
        return np.fliplr(mask)
    if corner == "bl":
        return np.flipud(mask)
    if corner == "br":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown corner {corner}")


def _marker_cells(corner: str, h: int, w: int) -> set[tuple[int, int]]:
    if corner == "tl":
        return {(0, 0), (0, 1), (1, 0)}
    if corner == "tr":
        return {(0, w - 1), (0, w - 2), (1, w - 1)}
    if corner == "bl":
        return {(h - 1, 0), (h - 2, 0), (h - 1, 1)}
    if corner == "br":
        return {(h - 1, w - 1), (h - 1, w - 2), (h - 2, w - 1)}
    raise AssertionError(f"unknown corner {corner}")


def _decode_marker(block: np.ndarray, bg: int, grey: int) -> tuple[str, int] | None:
    h, w = block.shape
    mask = (block != int(bg)) & (block != int(grey))
    if not bool(mask.any()):
        return None

    colors = np.unique(block[mask])
    assert len(colors) == 1, "marker block must use exactly one non-background color"
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert len(pts) == 3, "L marker must contain exactly three cells"

    for corner in ("tl", "tr", "bl", "br"):
        if pts == _marker_cells(corner, h, w):
            return corner, int(colors[0])
    raise AssertionError("invalid L marker geometry")


def _extract_segments(lines: list[int], start: int, end: int, size: int) -> list[int]:
    starts: list[int] = []
    current = int(start)
    for line in sorted(int(x) for x in lines) + [int(end)]:
        span = int(line - current)
        if span > 0:
            assert span == int(size), f"expected span {size}, got {span}"
            starts.append(current)
        current = int(line + 1)
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode corner L-markers into template transforms and recolor the transformed
    template into the corresponding output cells.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_mode_color(g))

    full_rows = [r for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != int(bg)]
    assert full_rows, "missing grey separator row"
    divider = int(full_rows[0])
    grey = int(g[divider, 0])

    top = g[:divider, :]
    template_colors = [int(color) for color in np.unique(top) if int(color) not in {bg, grey}]
    assert len(template_colors) == 1, "template region must contain exactly one foreground color"
    template_color = int(template_colors[0])

    template_coords = np.argwhere(top == int(template_color))
    assert template_coords.size > 0, "template must be non-empty"
    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    template = top[tr0 : tr1 + 1, tc0 : tc1 + 1] == int(template_color)
    cell_h, cell_w = template.shape
    assert cell_h >= 2 and cell_w >= 2

    horizontal_lines = [r for r in range(divider + 1, h) if np.all(g[r, :] == int(grey))]
    row_starts = _extract_segments(horizontal_lines, divider + 1, h, cell_h)
    assert row_starts, "instruction grid must contain at least one row of cells"

    vertical_lines = [c for c in range(w) if np.all(g[divider:, c] == int(grey))]
    col_starts = _extract_segments(vertical_lines, 0, w, cell_w)
    assert col_starts, "instruction grid must contain at least one column of cells"

    out = np.full((len(row_starts) * cell_h, len(col_starts) * cell_w), int(bg), dtype=int)

    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            block = g[r0 : r0 + cell_h, c0 : c0 + cell_w]
            decoded = _decode_marker(block, bg, grey)
            if decoded is None:
                continue
            corner, color = decoded
            transformed = _transform(template, corner)
            target = out[rr * cell_h : (rr + 1) * cell_h, cc * cell_w : (cc + 1) * cell_w]
            target[transformed] = int(color)

    return out


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _all_transforms_distinct(mask: np.ndarray) -> bool:
    variants = {
        _transform(mask, corner).astype(np.uint8).tobytes()
        for corner in ("tl", "tr", "bl", "br")
    }
    return len(variants) == 4


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_template_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    area = h * w
    min_cells = max(5, (area + 2) // 3)
    max_cells = max(min_cells, area - 2)

    for _ in range(200):
        target = rng.randint(min_cells, max_cells)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start, h, w))

        while frontier and len(cells) < target:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            for nxt in _neighbors4(cell, h, w):
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _touches_all_sides(mask):
            continue
        if not _all_transforms_distinct(mask):
            continue
        if int(mask.sum()) >= area - 1:
            continue
        return mask

    return None


def _build_expected(
    template: np.ndarray,
    placements: dict[tuple[int, int], tuple[str, int]],
    rows: int,
    cols: int,
) -> np.ndarray:
    h, w = template.shape
    out = np.full((rows * h, cols * w), BLACK, dtype=int)
    for (rr, cc), (corner, color) in placements.items():
        block = out[rr * h : (rr + 1) * h, cc * w : (cc + 1) * w]
        block[_transform(template, corner)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    cell_h = rng.randint(3, 5)
    cell_w = rng.randint(3, 5)
    template = _random_template_mask(cell_h, cell_w, rng)
    if template is None:
        return None

    rows = rng.randint(2, 4)
    cols = rng.randint(3, 5)
    total = rows * cols
    active_count = rng.randint(4, max(4, total - 1))

    positions = [(r, c) for r in range(rows) for c in range(cols)]
    active_positions = rng.sample(positions, active_count)

    corners = ["tl", "tr", "bl", "br"]
    assignments = corners + [rng.choice(corners) for _ in range(active_count - 4)]
    rng.shuffle(assignments)

    colors = list(rng.sample(MARKER_COLORS, min(len(MARKER_COLORS), active_count)))
    while len(colors) < active_count:
        colors.append(rng.choice(MARKER_COLORS))
    rng.shuffle(colors)

    placements: dict[tuple[int, int], tuple[str, int]] = {}
    for pos, corner, color in zip(active_positions, assignments, colors):
        placements[pos] = (corner, int(color))

    width = 1 + cols * (cell_w + 1)
    top_height = rng.randint(cell_h + 2, cell_h + 6)
    divider = top_height
    height = divider + 1 + rows * (cell_h + 1)

    grid = np.full((height, width), BLACK, dtype=int)

    grid[divider, :] = GREY
    for r in range(rows + 1):
        line = divider + r * (cell_h + 1)
        if line < height:
            grid[line, :] = GREY
    for c in range(cols + 1):
        line = c * (cell_w + 1)
        grid[divider:, line] = GREY

    template_top = rng.randint(1, top_height - cell_h - 1)
    template_left = rng.randint(1, width - cell_w - 1)
    grid[template_top : template_top + cell_h, template_left : template_left + cell_w][template] = MAROON

    for (rr, cc), (corner, color) in placements.items():
        r0 = divider + 1 + rr * (cell_h + 1)
        c0 = 1 + cc * (cell_w + 1)
        for dr, dc in _marker_cells(corner, cell_h, cell_w):
            grid[r0 + dr, c0 + dc] = int(color)

    expected = _build_expected(template, placements, rows, cols)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
