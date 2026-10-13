from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import bbox_coords, components_mask


BACKGROUND = 0
FRAME = 7
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_ATTEMPTS = 500


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _slot_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frame_coords = np.argwhere(grid == FRAME)
    assert frame_coords.size > 0, "expected an orange table"
    table_box = bbox_coords(frame_coords)
    tr0, tc0, tr1, tc1 = table_box

    boxes: list[tuple[int, int, int, int]] = []
    for coords in components_mask(grid == BACKGROUND, connectivity=4):
        if _touches_edge(coords, grid.shape):
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if tr0 < r0 <= r1 < tr1 and tc0 < c0 <= c1 < tc1:
            boxes.append((r0, c0, r1, c1))
    assert boxes, "expected enclosed black table cells"
    return boxes


def _table_summary(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_coords = np.argwhere(grid == FRAME)
    table_box = bbox_coords(frame_coords)
    boxes = _slot_bboxes(grid)
    row_starts = sorted({b[0] for b in boxes})
    col_starts = sorted({b[1] for b in boxes})
    assert len(boxes) == len(row_starts) * len(col_starts), "expected a complete rectangular cell array"

    box_by_anchor = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in boxes}
    out = np.zeros((len(row_starts), len(col_starts)), dtype=int)
    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            assert (r0, c0) in box_by_anchor, "missing cell in table"
            sr0, sc0, sr1, sc1 = box_by_anchor[(r0, c0)]
            window = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
            colors = sorted(set(map(int, window.flatten())) - {BACKGROUND, FRAME})
            assert len(colors) <= 1, "each cell must contain at most one non-black non-frame color"
            if colors:
                out[rr, cc] = colors[0]
    return out, table_box


def _field_start(grid: np.ndarray, table_box: tuple[int, int, int, int]) -> int:
    _, _, _, tc1 = table_box
    cols = np.argwhere((grid != BACKGROUND) & (grid != FRAME) & (np.indices(grid.shape)[1] > tc1))
    assert cols.size > 0, "expected right-side sprites"
    return int(cols[:, 1].min())


def _sprites(field: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, field.flatten())) - {BACKGROUND}):
        coords_list = components_mask(field == color, connectivity=4)
        assert len(coords_list) == 1, "each sprite color must appear in exactly one component"
        coords = coords_list[0]
        sprites.append({"color": int(color), "coords": coords, "bbox": bbox_coords(coords)})
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the enclosed color summary table to choose which right-side sprite moves, and use the cell coordinates to encode how far it moves.

    Input structure:
    - A black canvas contains one left-side orange enclosed table with outer frame and internal orange separators.
    - Each table cell is a black enclosed region; some cells contain exactly one non-black non-orange pixel.
    - To the right, the black field contains several disjoint single-color 4-connected sprites, one sprite per summary color.

    Rule:
    - Extract the table's row-by-column summary: each cell contributes its unique non-black non-orange color, or black if empty.
    - Keep only the right field for the output.
    - For every non-black summary entry at table position `(row, col)`, find the right-side sprite of the same color and translate that sprite down by `row` and right by `col`.
    - Clear the original sprite positions; the output is the right field after all such moves.

    Key insight:
    - The same table carries two roles at once: the cell color says which sprite to operate on, while the cell's row and column indices say how far that sprite moves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be 2D"

    summary, table_box = _table_summary(g)
    start = _field_start(g, table_box)
    field = g[:, start:]
    sprites = _sprites(field)

    entries = [
        (r, c, int(summary[r, c]))
        for r in range(summary.shape[0])
        for c in range(summary.shape[1])
        if int(summary[r, c]) != BACKGROUND
    ]
    assert len(entries) == len(sprites), "table entries and sprites must match in count"

    sprite_by_color = {item["color"]: item for item in sprites}
    assert len(sprite_by_color) == len(sprites), "sprite colors must be unique"
    assert {color for _, _, color in entries} == set(sprite_by_color), "table colors and sprite colors must match exactly"

    out = np.zeros_like(field)
    occupied = np.zeros(field.shape, dtype=bool)
    h, w = field.shape
    for dr, dc, color in entries:
        coords = sprite_by_color[color]["coords"]
        moved = coords + np.array([dr, dc])
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h) & (0 <= moved[:, 1]) & (moved[:, 1] < w))
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "translated sprites must not overlap"
        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = color
    return out


def _random_shape(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
        cells.add((r + dr, c + dc))

    if random.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    if random.random() < 0.5:
        cells = {(-r, c) for r, c in cells}
    turns = random.randint(0, 3)
    for _ in range(turns):
        cells = {(c, -r) for r, c in cells}

    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    box = bbox_coords(coords)
    return box[2] - box[0] + 1, box[3] - box[1] + 1


def _expand_one(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                expanded.add((rr, cc))
    return expanded


def _can_use(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells:
        return False
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
        return False
    return cells.isdisjoint(occupied)


def _sample_positions(k: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(80):
        chosen = random.sample(all_cells, k)
        if len({r for r, _ in chosen}) < 2:
            continue
        if len({c for _, c in chosen}) < 2:
            continue
        if not any(r > 0 for r, _ in chosen):
            continue
        if not any(c > 0 for _, c in chosen):
            continue
        return chosen
    return None


def _table_layout(k: int) -> dict | None:
    for _ in range(120):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        if rows * cols <= k:
            continue

        row_heights = [random.randint(3, 5) for _ in range(rows)]
        col_widths = [random.randint(3, 5) for _ in range(cols)]
        h_lines = [random.randint(1, 2) for _ in range(rows + 1)]
        v_lines = [random.randint(1, 2) for _ in range(cols + 1)]
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        gap = random.randint(2, 4)
        field_width = random.randint(9, 14)

        table_h = sum(row_heights) + sum(h_lines)
        table_w = sum(col_widths) + sum(v_lines)
        grid_h = top_margin + table_h + bottom_margin
        grid_w = left_margin + table_w + gap + field_width
        if grid_h > 30 or grid_w > 30:
            continue
        positions = _sample_positions(k, rows, cols)
        if positions is None:
            continue
        return {
            "rows": rows,
            "cols": cols,
            "row_heights": row_heights,
            "col_widths": col_widths,
            "h_lines": h_lines,
            "v_lines": v_lines,
            "top": top_margin,
            "left": left_margin,
            "gap": gap,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "field_w": field_width,
            "positions": positions,
        }
    return None


def _cell_boxes(layout: dict) -> list[list[tuple[int, int, int, int]]]:
    boxes: list[list[tuple[int, int, int, int]]] = []
    r = layout["top"] + layout["h_lines"][0]
    for rr, h in enumerate(layout["row_heights"]):
        row_boxes = []
        c = layout["left"] + layout["v_lines"][0]
        for cc, w in enumerate(layout["col_widths"]):
            row_boxes.append((r, c, r + h - 1, c + w - 1))
            c += w + layout["v_lines"][cc + 1]
        boxes.append(row_boxes)
        r += h + layout["h_lines"][rr + 1]
    return boxes


def _draw_table(grid: np.ndarray, layout: dict) -> None:
    top = layout["top"]
    left = layout["left"]
    table_h = sum(layout["row_heights"]) + sum(layout["h_lines"])
    table_w = sum(layout["col_widths"]) + sum(layout["v_lines"])
    local = np.zeros((table_h, table_w), dtype=int)

    r = 0
    for idx, thickness in enumerate(layout["h_lines"]):
        local[r : r + thickness, :] = FRAME
        r += thickness
        if idx < len(layout["row_heights"]):
            r += layout["row_heights"][idx]

    c = 0
    for idx, thickness in enumerate(layout["v_lines"]):
        local[:, c : c + thickness] = FRAME
        c += thickness
        if idx < len(layout["col_widths"]):
            c += layout["col_widths"][idx]

    grid[top : top + table_h, left : left + table_w] = local


def _place_sprites(
    grid_h: int,
    field_w: int,
    assignments: list[tuple[int, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    field_in = np.zeros((grid_h, field_w), dtype=int)
    field_out = np.zeros((grid_h, field_w), dtype=int)
    placed: list[dict] = []
    blocked_in: set[tuple[int, int]] = set()
    blocked_out: set[tuple[int, int]] = set()
    forced_index = random.randrange(len(assignments))

    for idx, (color, (dr, dc)) in enumerate(assignments):
        shape = _random_shape(random.randint(3, 6))
        h, w = _shape_dims(shape)
        max_r0 = grid_h - h - dr
        max_c0 = field_w - w - dc
        if max_r0 < 0 or max_c0 < 0:
            return None

        success = False
        for _ in range(200):
            r0 = random.randint(0, max_r0)
            c0 = 0 if idx == forced_index else random.randint(0, max_c0)
            if idx == forced_index and c0 > max_c0:
                continue
            cells_in = {(r0 + int(r), c0 + int(c)) for r, c in shape}
            cells_out = {(r0 + dr + int(r), c0 + dc + int(c)) for r, c in shape}
            if not _can_use(cells_in, blocked_in, grid_h, field_w):
                continue
            if not _can_use(cells_out, blocked_out, grid_h, field_w):
                continue
            for r, c in cells_in:
                field_in[r, c] = color
            for r, c in cells_out:
                field_out[r, c] = color
            blocked_in |= _expand_one(cells_in)
            blocked_out |= _expand_one(cells_out)
            coords = np.array(sorted(cells_in), dtype=int)
            placed.append({"color": color, "dr": dr, "dc": dc, "coords": coords, "bbox": bbox_coords(coords)})
            success = True
            break
        if not success:
            return None
    return field_in, field_out, placed


def _expected_from_meta(inp: np.ndarray, field_start: int, assignments: list[tuple[int, tuple[int, int]]]) -> np.ndarray:
    field = inp[:, field_start:]
    sprite_by_color = {item["color"]: item for item in _sprites(field)}
    out = np.zeros_like(field)
    for color, (dr, dc) in assignments:
        coords = sprite_by_color[color]["coords"] + np.array([dr, dc])
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 5)
    layout = _table_layout(k)
    if layout is None:
        return None

    colors = random.sample(SPRITE_COLORS, k)
    positions = list(layout["positions"])
    random.shuffle(positions)
    assignments = list(zip(colors, positions))

    if not any(dr > 0 for _, (dr, _) in assignments):
        return None
    if not any(dc > 0 for _, (_, dc) in assignments):
        return None

    placed = _place_sprites(layout["grid_h"], layout["field_w"], assignments)
    if placed is None:
        return None
    field_in, field_out, sprite_meta = placed

    table_color_order = [color for color, _ in sorted(assignments, key=lambda item: item[1])]
    sprite_color_order = [item["color"] for item in sorted(sprite_meta, key=lambda item: (item["bbox"][0], item["bbox"][1]))]
    if table_color_order == sprite_color_order:
        return None

    full = np.zeros((layout["grid_h"], layout["grid_w"]), dtype=int)
    _draw_table(full, layout)
    boxes = _cell_boxes(layout)
    for color, (rr, cc) in assignments:
        r0, c0, r1, c1 = boxes[rr][cc]
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        pr = random.randint(r0 + 1, r1 - 1)
        pc = random.randint(c0 + 1, c1 - 1)
        full[pr, pc] = color

    field_start = layout["left"] + sum(layout["col_widths"]) + sum(layout["v_lines"]) + layout["gap"]
    full[:, field_start:] = field_in

    expected = _expected_from_meta(full, field_start, assignments)
    if not np.array_equal(expected, field_out):
        return None
    solved = main(full)
    if not np.array_equal(solved, field_out):
        return None

    summary, table_box = _table_summary(full)
    if int(np.count_nonzero(summary)) != k:
        return None
    if summary.size <= k:
        return None
    if _field_start(full, table_box) != field_start:
        return None
    if np.array_equal(full[:, field_start:], field_out):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
