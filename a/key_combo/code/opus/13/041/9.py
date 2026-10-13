from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

LEFT_ACTIVE_WIDTHS = (3, 4)
RIGHT_ACTIVE_WIDTH_RANGE = (7, 11)
EXEMPLAR_HEIGHT_RANGE = (4, 6)
GRID_HEIGHT_RANGE = (10, 16)
MAX_ATTEMPTS = 400

ROW_TYPES = ("both", "left", "right", "none")


def _single_divider_column(grid: np.ndarray, frame: int) -> int:
    dividers = [c for c in range(1, grid.shape[1] - 1) if np.all(grid[:, c] == frame)]
    assert len(dividers) == 1, "expected exactly one full-height divider"
    return int(dividers[0])


def _active_slices(grid: np.ndarray) -> tuple[int, slice, slice, int]:
    frame = int(grid[0, 0])
    assert np.all(grid[0, :] == frame) and np.all(grid[-1, :] == frame), "bad horizontal frame"
    assert np.all(grid[:, 0] == frame) and np.all(grid[:, -1] == frame), "bad vertical frame"

    divider = _single_divider_column(grid, frame)
    left_active = slice(2, divider - 1)
    right_active = slice(divider + 2, grid.shape[1] - 2)
    assert left_active.stop - left_active.start >= 3, "left active span must contain boundary + interior + boundary"
    assert right_active.stop - right_active.start > left_active.stop - left_active.start, "right active span must be wider"
    return frame, left_active, right_active, divider


def _panel_background(grid: np.ndarray, divider: int) -> int:
    left_panel = grid[1:-1, 1:divider]
    right_panel = grid[1:-1, divider + 1 : -1]

    bg = int(left_panel[0, 0])
    assert np.all(left_panel[:, 0] == bg) and np.all(left_panel[:, -1] == bg), "left panel margins must be uniform"
    assert np.all(right_panel[:, 0] == bg) and np.all(right_panel[:, -1] == bg), "right panel margins must be uniform"
    return bg


def _exemplar_rows(grid: np.ndarray, left_active: slice, bg: int) -> tuple[list[int], list[tuple[int, int, int]]]:
    left = grid[1:-1, left_active]
    nonbg_rows = np.where(np.any(left != bg, axis=1))[0]
    assert len(nonbg_rows) >= 4, "expected a multi-row exemplar"
    assert np.array_equal(nonbg_rows, np.arange(nonbg_rows[0], nonbg_rows[-1] + 1)), "exemplar rows must be contiguous"
    assert not np.any(left[: nonbg_rows[0]] != bg), "rows above exemplar must be blank"
    assert not np.any(left[nonbg_rows[-1] + 1 :] != bg), "rows below exemplar must be blank"

    rows = [int(r + 1) for r in nonbg_rows]
    descs: list[tuple[int, int, int]] = []
    for r in rows:
        row = grid[r, left_active]
        assert np.all(row != bg), "exemplar rows must fill the entire active width"
        left_color = int(row[0])
        inner = row[1:-1]
        assert inner.size >= 1 and np.all(inner == inner[0]), "interior run must be monochrome"
        inner_color = int(inner[0])
        right_color = int(row[-1])
        assert left_color != bg and inner_color != bg and right_color != bg, "exemplar colors must be visible"
        descs.append((left_color, inner_color, right_color))
    return rows, descs


def _row_kind(row: np.ndarray, bg: int) -> str:
    left_on = bool(np.all(row[:2] != bg))
    right_on = bool(np.all(row[-2:] != bg))
    middle = row[2:-2]
    assert not np.any(middle != bg), "right panel input should only contain endpoint markers"
    if left_on and right_on:
        return "both"
    if left_on:
        return "left"
    if right_on:
        return "right"
    return "none"


def _paint_row(width: int, row_kind: str, desc: tuple[int, int, int], bg: int) -> np.ndarray:
    left_color, inner_color, right_color = desc
    row = np.full(width, bg, dtype=int)
    if row_kind == "both":
        row[0] = left_color
        row[1:-1] = inner_color
        row[-1] = right_color
    elif row_kind == "left":
        row[0] = left_color
        row[1] = inner_color
    elif row_kind == "right":
        row[-2] = inner_color
        row[-1] = right_color
    elif row_kind != "none":
        raise AssertionError(f"unknown row kind: {row_kind}")
    return row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each exemplar row into the right panel according to its endpoint markers.

    Input structure:
    - A 1-cell frame surrounds the grid, and one full-height divider splits a narrow left reference panel from a wider right target panel.
    - Both panels have a 1-cell blank margin on each side of their active columns.
    - The left active span contains one contiguous multicolor exemplar block; each exemplar row has a 1-pixel left boundary, a monochrome interior run, and a 1-pixel right boundary.
    - The right active span is blank except for optional 2-pixel endpoint markers at its left edge and/or right edge on rows aligned with the exemplar.

    Rule:
    - Keep the frame, divider, left panel, and all endpoint marker pixels unchanged.
    - For each exemplar row, inspect the aligned right-panel row:
      if both endpoint markers are present, copy the exemplar boundary colors to the two ends and elastically stretch the interior color across the middle;
      if only the left marker is present, copy only the left boundary plus one interior pixel;
      if only the right marker is present, copy only one interior pixel plus the right boundary;
      if no marker is present, leave the row blank.
    - The 2-pixel endpoint markers are preserved on top of the copied content.

    Key insight:
    - The left exemplar defines per-row boundary and interior roles, while the right endpoint markers decide which rows use full elastic stretching and which rows use only a local partial copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _frame, left_active, right_active, divider = _active_slices(g)
    bg = _panel_background(g, divider)
    rows, descs = _exemplar_rows(g, left_active, bg)

    out = g.copy()
    width = right_active.stop - right_active.start
    for r, desc in zip(rows, descs):
        source = g[r, right_active].copy()
        kind = _row_kind(source, bg)
        painted = _paint_row(width, kind, desc, bg)
        if kind in ("both", "left"):
            painted[:2] = source[:2]
        if kind in ("both", "right"):
            painted[-2:] = source[-2:]
        out[r, right_active] = painted

    return out


def _sample_palette(rng: random.Random) -> list[int]:
    size = rng.randint(3, 4)
    return rng.sample(list(COLOR_POOL), size)


def _sample_row_desc(rng: random.Random, palette: list[int]) -> tuple[int, int, int]:
    inner = rng.choice(palette)
    edge_choices = [c for c in palette if c != inner]
    left = rng.choice(edge_choices)
    right = rng.choice(edge_choices)
    return int(left), int(inner), int(right)


def _expected_right_row(width: int, row_kind: str, desc: tuple[int, int, int]) -> tuple[int, ...]:
    row = _paint_row(width, row_kind, desc, BG)
    if row_kind in ("both", "left"):
        row[0] = desc[0]
        row[1] = desc[1]
    if row_kind in ("both", "right"):
        row[-2] = desc[1]
        row[-1] = desc[2]
    return tuple(map(int, row))


def _sample_rows(
    rng: random.Random, height: int, width: int, palette: list[int]
) -> tuple[list[str], list[tuple[int, int, int]]]:
    row_types = list(ROW_TYPES)
    while len(row_types) < height:
        row_types.append(rng.choice(ROW_TYPES))
    rng.shuffle(row_types)

    descs: list[tuple[int, int, int]] = []
    seen_descs: set[tuple[int, int, int]] = set()
    seen_outputs: set[tuple[int, ...]] = set()
    for row_kind in row_types:
        for _ in range(80):
            desc = _sample_row_desc(rng, palette)
            out_row = _expected_right_row(width, row_kind, desc)
            if desc in seen_descs or out_row in seen_outputs:
                continue
            seen_descs.add(desc)
            seen_outputs.add(out_row)
            descs.append(desc)
            break
        else:
            raise RuntimeError("failed to sample distinct exemplar rows")
    return row_types, descs


def _build_grid(
    rng: random.Random,
) -> tuple[np.ndarray, list[int], list[str], list[tuple[int, int, int]], slice, slice]:
    left_w = rng.choice(LEFT_ACTIVE_WIDTHS)
    right_w = rng.randint(max(RIGHT_ACTIVE_WIDTH_RANGE[0], left_w + 3), RIGHT_ACTIVE_WIDTH_RANGE[1])
    exemplar_h = rng.randint(*EXEMPLAR_HEIGHT_RANGE)
    total_h = rng.randint(max(GRID_HEIGHT_RANGE[0], exemplar_h + 4), GRID_HEIGHT_RANGE[1])
    total_w = left_w + right_w + 7

    assert total_h <= 30 and total_w <= 30

    top = rng.randint(2, total_h - exemplar_h - 2)
    rows = list(range(top, top + exemplar_h))
    palette = _sample_palette(rng)
    row_types, descs = _sample_rows(rng, exemplar_h, right_w, palette)

    grid = np.full((total_h, total_w), BG, dtype=int)
    divider = left_w + 3
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    grid[:, divider] = FRAME

    left_active = slice(2, divider - 1)
    right_active = slice(divider + 2, total_w - 2)

    for r, desc, row_kind in zip(rows, descs, row_types):
        left_color, inner_color, right_color = desc
        grid[r, left_active.start] = left_color
        grid[r, left_active.start + 1 : left_active.stop - 1] = inner_color
        grid[r, left_active.stop - 1] = right_color

        if row_kind in ("both", "left"):
            grid[r, right_active.start : right_active.start + 2] = (left_color, inner_color)
        if row_kind in ("both", "right"):
            grid[r, right_active.stop - 2 : right_active.stop] = (inner_color, right_color)

    return grid, rows, row_types, descs, left_active, right_active


def _expected_output(
    grid: np.ndarray,
    rows: list[int],
    row_types: list[str],
    descs: list[tuple[int, int, int]],
    right_active: slice,
) -> np.ndarray:
    out = grid.copy()
    width = right_active.stop - right_active.start
    for r, row_kind, desc in zip(rows, row_types, descs):
        row = _paint_row(width, row_kind, desc, BG)
        if row_kind in ("both", "left"):
            row[:2] = (desc[0], desc[1])
        if row_kind in ("both", "right"):
            row[-2:] = (desc[1], desc[2])
        out[r, right_active] = row
    return out


def _validate_generated(
    grid: np.ndarray,
    rows: list[int],
    row_types: list[str],
    descs: list[tuple[int, int, int]],
    left_active: slice,
    right_active: slice,
) -> bool:
    expected = _expected_output(grid, rows, row_types, descs, right_active)
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    right_width = right_active.stop - right_active.start
    transformed_rows = {
        tuple(expected[r, right_active].tolist())
        for r in rows
    }
    if len(transformed_rows) != len(rows):
        return False

    if set(row_types) != set(ROW_TYPES):
        return False

    full_rows = [r for r, k in zip(rows, row_types) if k == "both"]
    left_rows = [r for r, k in zip(rows, row_types) if k == "left"]
    right_rows = [r for r, k in zip(rows, row_types) if k == "right"]
    none_rows = [r for r, k in zip(rows, row_types) if k == "none"]
    if not (full_rows and left_rows and right_rows and none_rows):
        return False

    if not any(np.count_nonzero(expected[r, right_active] != BG) == right_width for r in full_rows):
        return False
    if not all(np.count_nonzero(expected[r, right_active] != BG) == 2 for r in left_rows + right_rows):
        return False
    if not all(np.count_nonzero(expected[r, right_active] != BG) == 0 for r in none_rows):
        return False

    left_patch = grid[np.array(rows), left_active]
    if len({tuple(row.tolist()) for row in left_patch}) != len(rows):
        return False

    if len({color for desc in descs for color in desc}) < 3:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    try:
        grid, rows, row_types, descs, left_active, right_active = _build_grid(rng)
    except RuntimeError:
        return None

    if _validate_generated(grid, rows, row_types, descs, left_active, right_active):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid elastic-endpoint ARC input")
