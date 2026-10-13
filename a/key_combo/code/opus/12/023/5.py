from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_CELLS = 4
SEP_WIDTH = 1
GEN_ATTEMPTS = 400

_GEN_COUNTER = 0


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(separators):
        if idx <= start or idx >= length:
            raise AssertionError("invalid separator placement")
        spans.append((start, idx))
        start = idx + 1
    if start >= length:
        raise AssertionError("missing terminal span")
    spans.append((start, length))
    return spans


def _find_singleton(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    if coords.shape != (1, 2):
        raise AssertionError(f"expected exactly one pixel of color {color}")
    return int(coords[0, 0]), int(coords[0, 1])


def _locate_span(spans: list[tuple[int, int]], idx: int) -> int:
    for pos, (start, end) in enumerate(spans):
        if start <= idx < end:
            return pos
    raise AssertionError("coordinate not inside any span")


def _component_count(mask: np.ndarray) -> int:
    seen: set[tuple[int, int]] = set()
    count = 0
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        count += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            rr, cc = stack.pop()
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                    nxt = (nr, nc)
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
    return count


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == RED)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == RED)]
    if len(sep_rows) != GRID_CELLS - 1 or len(sep_cols) != GRID_CELLS - 1:
        raise AssertionError("expected a 4x4 separator lattice")

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    if len(row_spans) != GRID_CELLS or len(col_spans) != GRID_CELLS:
        raise AssertionError("expected four cell rows and columns")

    cell_heights = {end - start for start, end in row_spans}
    cell_widths = {end - start for start, end in col_spans}
    if len(cell_heights) != 1 or len(cell_widths) != 1:
        raise AssertionError("expected equal-sized cells")
    cell_h = cell_heights.pop()
    cell_w = cell_widths.pop()
    if cell_h != cell_w or cell_h % 2 == 0:
        raise AssertionError("expected odd square cells")
    cell_size = cell_h

    blue_pos = _find_singleton(g, BLUE)
    yellow_pos = _find_singleton(g, YELLOW)
    blue_cell = (_locate_span(row_spans, blue_pos[0]), _locate_span(col_spans, blue_pos[1]))
    yellow_cell = (_locate_span(row_spans, yellow_pos[0]), _locate_span(col_spans, yellow_pos[1]))

    if blue_cell[0] == yellow_cell[0] or blue_cell[1] == yellow_cell[1]:
        raise AssertionError("blue and yellow selectors must not share a cell row or column")

    template_cells: list[tuple[int, int]] = []
    template_patch = None
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            patch = g[r0:r1, c0:c1]
            if np.any(patch == GREEN):
                template_cells.append((rr, cc))
                template_patch = patch.copy()

    if len(template_cells) != 1 or template_patch is None:
        raise AssertionError("expected exactly one template cell")
    template_cell = template_cells[0]

    green_mask = template_patch == GREEN
    if not np.all(np.isin(template_patch, [BLACK, GREEN])):
        raise AssertionError("template cell must contain only black and green")
    if _component_count(green_mask) != 1:
        raise AssertionError("template sprite must be 4-connected")

    area = int(green_mask.sum())
    if area < 3 or area > 20:
        raise AssertionError("template area must be 3-20")

    box = _bbox(green_mask)
    crop = _crop_bbox(green_mask.astype(int), box)
    if min(crop.shape) < 2:
        raise AssertionError("template sprite must be two-dimensional")
    if crop.sum() == crop.size:
        raise AssertionError("template sprite must be irregular")
    if np.array_equal(crop, np.fliplr(crop)):
        raise AssertionError("template sprite cannot be horizontally symmetric")
    if np.array_equal(crop, np.flipud(crop)):
        raise AssertionError("template sprite cannot be vertically symmetric")
    if np.array_equal(crop, np.rot90(crop, 2)):
        raise AssertionError("template sprite cannot have 180-degree symmetry")

    original = template_patch
    hflip = np.fliplr(original)
    vflip = np.flipud(original)
    if np.array_equal(original, hflip) or np.array_equal(original, vflip) or np.array_equal(hflip, vflip):
        raise AssertionError("full-cell mirrors must be visibly distinct")

    intersection = (blue_cell[0], yellow_cell[1])
    protected = {blue_cell, yellow_cell, template_cell}
    if intersection in protected:
        raise AssertionError("intersection cell must differ from all special cells")

    return {
        "grid": g,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cell_size": cell_size,
        "blue_pos": blue_pos,
        "yellow_pos": yellow_pos,
        "blue_cell": blue_cell,
        "yellow_cell": yellow_cell,
        "active_row": blue_cell[0],
        "active_col": yellow_cell[1],
        "template_cell": template_cell,
        "template_patch": template_patch,
        "intersection": intersection,
    }


def _write_cell(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], patch: np.ndarray) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0:r1, c0:c1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """Use the blue cell row and yellow cell column to stamp a green sprite into their crossing and mirrored companion cells.

    Input structure:
    - Red separator lines split the grid into a 4x4 lattice of equal odd square cells.
    - Exactly one cell contains a small irregular green 4-connected sprite on black.
    - One different cell has a single blue center pixel, and another different cell has a single yellow center pixel.

    Rule:
    - Take the blue pixel's cell row as the active row and the yellow pixel's cell column as the active column.
    - Copy the full green template cell into the row-column intersection cell.
    - In the other eligible cells of the active row, place left-right mirrored copies of the template cell.
    - In the other eligible cells of the active column, place top-bottom mirrored copies of the template cell, while leaving the blue cell, yellow cell, template cell, black background, and red separators otherwise unchanged.

    Key insight:
    - The isolated selector pixels do not draw rays directly; they choose one cell row and one cell column, and the lattice coordinates determine where the original template and the two mirror variants must be stamped.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    g = info["grid"]
    out = g.copy()

    active_row = info["active_row"]
    active_col = info["active_col"]
    blue_cell = info["blue_cell"]
    yellow_cell = info["yellow_cell"]
    template_cell = info["template_cell"]
    intersection = info["intersection"]
    template_patch = info["template_patch"]
    hflip = np.fliplr(template_patch)
    vflip = np.flipud(template_patch)

    _write_cell(out, info["row_spans"][intersection[0]], info["col_spans"][intersection[1]], template_patch)

    for cc in range(GRID_CELLS):
        cell = (active_row, cc)
        if cell in {blue_cell, template_cell, intersection}:
            continue
        _write_cell(out, info["row_spans"][active_row], info["col_spans"][cc], hflip)

    for rr in range(GRID_CELLS):
        cell = (rr, active_col)
        if cell in {yellow_cell, template_cell, intersection}:
            continue
        _write_cell(out, info["row_spans"][rr], info["col_spans"][active_col], vflip)

    return out


def _blank_lattice(cell_size: int) -> np.ndarray:
    side = GRID_CELLS * cell_size + (GRID_CELLS - 1) * SEP_WIDTH
    grid = np.full((side, side), BLACK, dtype=int)
    for k in range(1, GRID_CELLS):
        idx = k * cell_size + (k - 1) * SEP_WIDTH
        grid[idx : idx + SEP_WIDTH, :] = RED
        grid[:, idx : idx + SEP_WIDTH] = RED
    return grid


def _cell_slice(index: int, cell_size: int) -> slice:
    start = index * cell_size + index * SEP_WIDTH
    return slice(start, start + cell_size)


def _make_template_patch(cell_size: int) -> np.ndarray:
    for _ in range(500):
        area = random.randint(3, min(20, cell_size * cell_size - 1))
        coords = {(random.randrange(cell_size), random.randrange(cell_size))}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) != area:
            continue
        patch = np.full((cell_size, cell_size), BLACK, dtype=int)
        for r, c in coords:
            patch[r, c] = GREEN
        try:
            # Reuse the same semantic checks used by the solver.
            _ = _analyze(_embed_template_only(patch))
        except AssertionError:
            continue
        return patch
    raise RuntimeError("failed to create a valid template patch")


def _embed_template_only(patch: np.ndarray) -> np.ndarray:
    cell_size = patch.shape[0]
    grid = _blank_lattice(cell_size)
    center = cell_size // 2
    grid[_cell_slice(0, cell_size), _cell_slice(0, cell_size)] = patch
    grid[_cell_slice(1, cell_size).start + center, _cell_slice(2, cell_size).start + center] = BLUE
    grid[_cell_slice(2, cell_size).start + center, _cell_slice(1, cell_size).start + center] = YELLOW
    return grid


def _pick_template_cell(active_row: int, active_col: int, blue_col: int, yellow_row: int, relation: str) -> tuple[int, int]:
    if relation == "row":
        options = [(active_row, col) for col in range(GRID_CELLS) if col not in {blue_col, active_col}]
    elif relation == "col":
        options = [(row, active_col) for row in range(GRID_CELLS) if row not in {yellow_row, active_row}]
    elif relation == "neither":
        options = [
            (row, col)
            for row in range(GRID_CELLS)
            for col in range(GRID_CELLS)
            if row != active_row and col != active_col
        ]
    else:
        raise ValueError(f"unknown relation: {relation}")
    return random.choice(options)


def _gen_attempt(case_index: int) -> np.ndarray | None:
    cell_size = 3 if (case_index // 3) % 2 == 0 else 5
    relation = ("neither", "row", "col")[case_index % 3]

    active_row = random.randrange(GRID_CELLS)
    active_col = random.randrange(GRID_CELLS)
    blue_col = random.choice([c for c in range(GRID_CELLS) if c != active_col])
    yellow_row = random.choice([r for r in range(GRID_CELLS) if r != active_row])
    template_cell = _pick_template_cell(active_row, active_col, blue_col, yellow_row, relation)

    grid = _blank_lattice(cell_size)
    template_patch = _make_template_patch(cell_size)
    center = cell_size // 2

    rs = _cell_slice(template_cell[0], cell_size)
    cs = _cell_slice(template_cell[1], cell_size)
    grid[rs, cs] = template_patch

    blue_r = _cell_slice(active_row, cell_size).start + center
    blue_c = _cell_slice(blue_col, cell_size).start + center
    yellow_r = _cell_slice(yellow_row, cell_size).start + center
    yellow_c = _cell_slice(active_col, cell_size).start + center
    grid[blue_r, blue_c] = BLUE
    grid[yellow_r, yellow_c] = YELLOW

    if not _validate_generated(grid):
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)

        center = info["cell_size"] // 2
        blue_r0, blue_c0 = info["row_spans"][info["blue_cell"][0]][0], info["col_spans"][info["blue_cell"][1]][0]
        yellow_r0, yellow_c0 = info["row_spans"][info["yellow_cell"][0]][0], info["col_spans"][info["yellow_cell"][1]][0]
        if info["blue_pos"] != (blue_r0 + center, blue_c0 + center):
            return False
        if info["yellow_pos"] != (yellow_r0 + center, yellow_c0 + center):
            return False

        if not np.array_equal(out.shape, grid.shape):
            return False

        if not np.array_equal(out[grid == RED], grid[grid == RED]):
            return False

        template = info["template_patch"]
        hflip = np.fliplr(template)
        vflip = np.flipud(template)
        row_copies = 0
        col_copies = 0

        for rr in range(GRID_CELLS):
            for cc in range(GRID_CELLS):
                rspan = info["row_spans"][rr]
                cspan = info["col_spans"][cc]
                cell_in = grid[rspan[0] : rspan[1], cspan[0] : cspan[1]]
                cell_out = out[rspan[0] : rspan[1], cspan[0] : cspan[1]]
                cell = (rr, cc)

                if cell == info["template_cell"] or cell == info["blue_cell"] or cell == info["yellow_cell"]:
                    if not np.array_equal(cell_out, cell_in):
                        return False
                elif cell == info["intersection"]:
                    if not np.array_equal(cell_out, template):
                        return False
                elif rr == info["active_row"]:
                    if not np.array_equal(cell_out, hflip):
                        return False
                    row_copies += 1
                elif cc == info["active_col"]:
                    if not np.array_equal(cell_out, vflip):
                        return False
                    col_copies += 1
                else:
                    if not np.array_equal(cell_out, np.full_like(cell_out, BLACK)):
                        return False

        if row_copies < 1 or col_copies < 1:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for attempt in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_GEN_COUNTER + attempt)
        if grid is not None:
            _GEN_COUNTER += attempt + 1
            return grid
    raise RuntimeError("failed to generate a valid input")
