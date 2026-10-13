import random

import numpy as np


BLACK = 0
GREY = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_OPTIONS = ("top", "bottom", "left", "right")
DIM_OPTIONS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
COPY_COUNT_RANGE = (2, 5)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _palette_from_bbox(cells: np.ndarray) -> np.ndarray:
    rows, cols = cells.shape
    palette = np.zeros((2 * rows - 1, 2 * cols - 1), dtype=int)
    palette[1::2, :] = GREY
    palette[:, 1::2] = GREY
    palette[::2, ::2] = cells
    return palette


def _find_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    grey_parts = _components(grid == GREY)
    candidates = []
    h, w = grid.shape
    for part in grey_parts:
        r0, c0, r1, c1 = _bbox(part)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[0] < 3 or sub.shape[1] < 3:
            continue
        if sub.shape[0] % 2 == 0 or sub.shape[1] % 2 == 0:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue

        rr, cc = np.indices(sub.shape)
        line_mask = (rr % 2 == 1) | (cc % 2 == 1)
        cell_mask = ~line_mask
        if not np.all(sub[line_mask] == GREY):
            continue
        if np.any(sub[cell_mask] == BLACK) or np.any(sub[cell_mask] == GREY):
            continue
        candidates.append(((r0, c0, r1, c1), sub[::2, ::2].copy()))

    assert len(candidates) == 1, f"expected one palette candidate, got {len(candidates)}"
    return candidates[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    (pr0, pc0, pr1, pc1), palette_cells = _find_palette(g)
    box_h, box_w = palette_cells.shape
    palette_mask = np.zeros(g.shape, dtype=bool)
    palette_mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    copy_mask = (g != BLACK) & (~palette_mask)
    copy_colors = [int(color) for color in np.unique(g[copy_mask]) if int(color) != GREY]
    assert len(copy_colors) >= 2, "expected multiple sprite copies"

    copies = []
    template = np.zeros((box_h, box_w), dtype=bool)
    for color in copy_colors:
        coords = np.argwhere((g == color) & (~palette_mask))
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 == box_h
        assert c1 - c0 + 1 == box_w
        rel = np.zeros((box_h, box_w), dtype=bool)
        rel[coords[:, 0] - r0, coords[:, 1] - c0] = True
        copies.append((r0, c0, rel))
        template |= rel

    assert template.any(), "template must contain foreground"

    out = g.copy()
    for r0, c0, rel in copies:
        missing = np.argwhere(template & (~rel))
        assert len(missing) == 1, f"expected one missing pixel, got {len(missing)}"
        mr, mc = map(int, missing[0])
        new_color = int(palette_cells[mr, mc])
        patch = out[r0 : r0 + box_h, c0 : c0 + box_w]
        patch[template] = new_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor monochrome template copies using a palette lookup keyed by the missing pixel position.

    Input structure:
    - Background is black.
    - One edge contains a palette matrix rendered as colored cells with grey separator lines.
    - The palette matrix has the same row/column count as the copies' shared bounding box.
    - Elsewhere, several same-shape monochrome copies appear; each copy is missing exactly one template pixel.

    Rule:
    - Recover the common template by combining the incomplete copies.
    - For each copy, locate its missing pixel inside the shared bounding box.
    - Use that coordinate to read a color from the palette matrix.
    - Repaint the entire completed copy with the palette color.
    """
    return _solve(grid)


def _make_template(rows: int, cols: int) -> np.ndarray | None:
    area = rows * cols
    min_area = max(3, (area + 1) // 2)
    target = random.randint(min_area, area)
    for _ in range(200):
        mask = np.zeros((rows, cols), dtype=bool)
        seed = (random.randrange(rows), random.randrange(cols))
        mask[seed] = True
        cells = [seed]
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, rows, cols):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
            cells.append((nr, nc))
        if int(mask.sum()) != target:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != 0 or c0 != 0 or r1 != rows - 1 or c1 != cols - 1:
            continue
        return mask
    return None


def _mark_forbidden(forbidden: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = forbidden.shape
    forbidden[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _place_copies(
    grid_shape: tuple[int, int],
    forbidden: np.ndarray,
    rows: int,
    cols: int,
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    candidates = [(r, c) for r in range(h - rows + 1) for c in range(w - cols + 1)]
    random.shuffle(candidates)
    origins: list[tuple[int, int]] = []
    local_forbidden = forbidden.copy()
    for _ in range(count):
        placed = None
        for r0, c0 in candidates:
            if local_forbidden[r0 : r0 + rows, c0 : c0 + cols].any():
                continue
            placed = (r0, c0)
            break
        if placed is None:
            return None
        origins.append(placed)
        _mark_forbidden(local_forbidden, placed[0], placed[1], placed[0] + rows - 1, placed[1] + cols - 1)
    return origins


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(DIM_OPTIONS)
    template = _make_template(rows, cols)
    if template is None:
        return None

    shape_cells = [tuple(map(int, cell)) for cell in np.argwhere(template)]
    max_copies = min(COPY_COUNT_RANGE[1], len(shape_cells), len(USABLE_COLORS))
    min_copies = min(COPY_COUNT_RANGE[0], max_copies)
    if min_copies < 2:
        return None
    copy_count = random.randint(min_copies, max_copies)

    palette_colors = random.sample(list(USABLE_COLORS), rows * cols)
    palette_cells = np.array(palette_colors, dtype=int).reshape(rows, cols)
    palette = _palette_from_bbox(palette_cells)
    pal_h, pal_w = palette.shape

    side = random.choice(SIDE_OPTIONS)
    for _ in range(80):
        h = random.randint(max(GRID_H_RANGE[0], pal_h + 3), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], pal_w + 3), GRID_W_RANGE[1])
        if h < pal_h or w < pal_w:
            continue
        grid = np.zeros((h, w), dtype=int)
        forbidden = np.zeros((h, w), dtype=bool)

        if side == "top":
            pr0 = 0
            pc0 = random.randint(0, w - pal_w)
        elif side == "bottom":
            pr0 = h - pal_h
            pc0 = random.randint(0, w - pal_w)
        elif side == "left":
            pr0 = random.randint(0, h - pal_h)
            pc0 = 0
        else:
            pr0 = random.randint(0, h - pal_h)
            pc0 = w - pal_w

        grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = palette
        _mark_forbidden(forbidden, pr0, pc0, pr0 + pal_h - 1, pc0 + pal_w - 1)

        origins = _place_copies((h, w), forbidden, rows, cols, copy_count)
        if origins is None:
            continue

        copy_colors = random.sample(list(USABLE_COLORS), copy_count)
        missing_positions = random.sample(shape_cells, copy_count)
        for (r0, c0), color, missing in zip(origins, copy_colors, missing_positions):
            patch = grid[r0 : r0 + rows, c0 : c0 + cols]
            patch[template] = color
            patch[missing] = BLACK

        try:
            solved = _solve(grid)
        except AssertionError:
            continue

        if np.array_equal(solved, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
