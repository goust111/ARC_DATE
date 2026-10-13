from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 9
NON_SPECIAL_COLORS = tuple(range(1, 9))
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300


def _border_majority_color(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep = set(int(x) for x in separators)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in sep:
            if start is not None:
                out.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        out.append((start, int(length) - 1))
    assert out, "expected at least one non-separator band"
    return out


def _separator_color(board: np.ndarray, bg: int) -> int:
    row_colors = {
        int(board[r, 0])
        for r in range(board.shape[0])
        if np.all(board[r, :] == board[r, 0]) and int(board[r, 0]) != int(bg)
    }
    col_colors = {
        int(board[0, c])
        for c in range(board.shape[1])
        if np.all(board[:, c] == board[0, c]) and int(board[0, c]) != int(bg)
    }
    common = row_colors & col_colors
    assert len(common) == 1, f"expected a unique separator color, got {sorted(common)}"
    return common.pop()


def _panel_records(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _border_majority_color(g)
    r0, c0, r1, c1 = _bbox(g != bg)
    board = g[r0 : r1 + 1, c0 : c1 + 1]

    sep = _separator_color(board, bg)
    sep_rows = [r for r in range(board.shape[0]) if np.all(board[r, :] == sep)]
    sep_cols = [c for c in range(board.shape[1]) if np.all(board[:, c] == sep)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separator lines"

    row_bands = _bands(board.shape[0], sep_rows)
    col_bands = _bands(board.shape[1], sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-panel grid"

    row_heights = {r1b - r0b + 1 for r0b, r1b in row_bands}
    col_widths = {c1b - c0b + 1 for c0b, c1b in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "panels must have equal size"

    records: list[dict[str, object]] = []
    for pr0, pr1 in row_bands:
        for pc0, pc1 in col_bands:
            cell = board[pr0 : pr1 + 1, pc0 : pc1 + 1]
            colors, counts = np.unique(cell, return_counts=True)
            pairs = [
                (int(color), int(count))
                for color, count in zip(colors, counts)
                if int(color) not in {int(bg), int(sep)}
            ]
            assert len(pairs) == 2, f"expected two panel colors, got {pairs}"
            pairs.sort(key=lambda item: item[1], reverse=True)
            assert pairs[0][1] > pairs[1][1], "dominant color area must be unique"

            motif_color = int(pairs[1][0])
            records.append(
                {
                    "row_slice": slice(r0 + pr0, r0 + pr1 + 1),
                    "col_slice": slice(c0 + pc0, c0 + pc1 + 1),
                    "motif_mask": cell == motif_color,
                    "dominant_color": int(pairs[0][0]),
                    "dominant_area": int(pairs[0][1]),
                }
            )

    areas = [int(record["dominant_area"]) for record in records]
    assert len(set(areas)) == len(areas), "dominant areas must induce a strict order"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel's minority motif with the next panel's dominant color in dominant-area order.

    Input structure:
    - Black background surrounds a rectangular board split by one-pixel separator
      lines of a single non-background color.
    - The separator rows and columns divide the board into equal rectangular panels.
    - Each panel contains exactly two non-background, non-separator colors: a larger
      dominant fill and a smaller motif color.

    Rule:
    - Detect the separator grid and inspect every panel independently.
    - In each panel, identify the dominant color by larger pixel count and the motif
      color by smaller pixel count.
    - Rank panels by dominant-color area from largest to smallest.
    - Keep every dominant-color pixel unchanged, but recolor each panel's motif pixels
      to the dominant color of the next panel in that ranking, wrapping the last panel
      back to the first.
    - Preserve the background and separator lines.

    Key insight:
    - The motif shape stays in place; only its color changes, and the new color comes
      from a cyclic shift over the hidden ordering induced by dominant areas.
    """
    out = np.asarray(grid, dtype=int).copy()
    records = _panel_records(out)
    order = sorted(range(len(records)), key=lambda idx: int(records[idx]["dominant_area"]), reverse=True)

    for rank, idx in enumerate(order):
        target_idx = order[(rank + 1) % len(order)]
        target_color = int(records[target_idx]["dominant_color"])
        row_slice = records[idx]["row_slice"]
        col_slice = records[idx]["col_slice"]
        motif_mask = np.asarray(records[idx]["motif_mask"], dtype=bool)
        panel = out[row_slice, col_slice]
        panel[motif_mask] = target_color

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _grow_connected_mask(h: int, w: int, size: int) -> np.ndarray | None:
    for _ in range(80):
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < size:
            frontier = {
                nxt
                for r, c in cells
                for nxt in _neighbors4(r, c, h, w)
                if nxt not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != size:
            continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _random_motif_shape(cell_h: int, cell_w: int) -> np.ndarray | None:
    bbox_h = random.randint(2, min(4, cell_h - 2))
    bbox_w = random.randint(2, min(4, cell_w - 2))
    max_area = min(6, bbox_h * bbox_w - 1)
    if max_area < 3:
        return None

    size = random.randint(3, max_area)
    shape = _grow_connected_mask(bbox_h, bbox_w, size)
    if shape is None:
        return None
    return shape


def _embed_motif_shape(shape: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    bbox_h, bbox_w = shape.shape
    top = random.randint(1, cell_h - bbox_h - 1)
    left = random.randint(1, cell_w - bbox_w - 1)
    mask = np.zeros((cell_h, cell_w), dtype=bool)
    mask[top : top + bbox_h, left : left + bbox_w] = shape
    return mask


def _sample_hole_mask(candidates: list[tuple[int, int]], count: int, cell_h: int, cell_w: int) -> np.ndarray | None:
    mask = np.zeros((cell_h, cell_w), dtype=bool)
    if count == 0:
        return mask
    if count > len(candidates):
        return None

    allowed = set(candidates)
    for _ in range(40):
        seed = random.choice(candidates)
        cells = {seed}
        while len(cells) < count:
            frontier = {
                nxt
                for r, c in cells
                for nxt in _neighbors4(r, c, cell_h, cell_w)
                if nxt in allowed and nxt not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) == count:
            for r, c in cells:
                mask[r, c] = True
            return mask

    sampled = random.sample(candidates, count)
    for r, c in sampled:
        mask[r, c] = True
    return mask


def _layout_options() -> list[tuple[int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int]] = []
    for rows in range(2, 5):
        for cols in range(2, 5):
            if not (4 <= rows * cols <= 8):
                continue
            for cell_h in range(6, 9):
                for cell_w in range(6, 9):
                    for border in (1, 2):
                        height = 2 * border + rows * cell_h + (rows - 1)
                        width = 2 * border + cols * cell_w + (cols - 1)
                        if height <= MAX_GRID_SIZE and width <= MAX_GRID_SIZE:
                            options.append((rows, cols, cell_h, cell_w, border))
    return [(r, c, h, w, b) for r, c, h, w, b in options]


LAYOUT_OPTIONS = _layout_options()


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[slice, slice, np.ndarray, int, int]],
) -> np.ndarray:
    out = grid.copy()
    order = sorted(range(len(placements)), key=lambda idx: placements[idx][4], reverse=True)
    for rank, idx in enumerate(order):
        target_idx = order[(rank + 1) % len(order)]
        row_slice, col_slice, motif_mask, _motif_color, _area = placements[idx]
        target_color = int(placements[target_idx][3])
        panel = out[row_slice, col_slice]
        panel[motif_mask] = target_color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w, border = random.choice(LAYOUT_OPTIONS)
    panel_count = rows * cols

    motif_shape = _random_motif_shape(cell_h, cell_w)
    if motif_shape is None:
        return None
    motif_area = int(motif_shape.sum())

    interior_count = (cell_h - 2) * (cell_w - 2)
    max_holes = min(
        interior_count - motif_area,
        cell_h * cell_w - 2 * motif_area - 1,
    )
    if max_holes + 1 < panel_count:
        return None

    hole_counts = random.sample(list(range(max_holes + 1)), panel_count)
    random.shuffle(hole_counts)

    dominants = random.sample(list(NON_SPECIAL_COLORS), panel_count)
    motif_colors = [
        random.choice([color for color in NON_SPECIAL_COLORS if color != dominant])
        for dominant in dominants
    ]

    height = 2 * border + rows * cell_h + (rows - 1)
    width = 2 * border + cols * cell_w + (cols - 1)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    placements: list[tuple[slice, slice, np.ndarray, int, int]] = []

    for panel_idx in range(panel_count):
        rr = panel_idx // cols
        cc = panel_idx % cols
        r0 = border + rr * (cell_h + 1)
        c0 = border + cc * (cell_w + 1)

        motif_mask = _embed_motif_shape(motif_shape, cell_h, cell_w)

        available = [
            (r, c)
            for r in range(1, cell_h - 1)
            for c in range(1, cell_w - 1)
            if not motif_mask[r, c]
        ]
        hole_mask = _sample_hole_mask(available, hole_counts[panel_idx], cell_h, cell_w)
        if hole_mask is None:
            return None

        dominant_mask = ~motif_mask & ~hole_mask
        cell = np.full((cell_h, cell_w), BACKGROUND, dtype=int)
        cell[dominant_mask] = dominants[panel_idx]
        cell[motif_mask] = motif_colors[panel_idx]

        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
        dominant_area = int(np.sum(cell == dominants[panel_idx]))
        placements.append(
            (
                slice(r0, r0 + cell_h),
                slice(c0, c0 + cell_w),
                motif_mask,
                int(dominants[panel_idx]),
                dominant_area,
            )
        )

    for r in range(rows - 1):
        sep_r = border + (r + 1) * cell_h + r
        grid[sep_r, border : width - border] = SEPARATOR
    for c in range(cols - 1):
        sep_c = border + (c + 1) * cell_w + c
        grid[border : height - border, sep_c] = SEPARATOR

    expected = _expected_output(grid, placements)

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
