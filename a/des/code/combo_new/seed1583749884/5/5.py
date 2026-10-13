from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400
TILE_CHOICES = (2, 3)


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _repeat_tile(tile: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    return np.tile(tile, (int(tile_rows), int(tile_cols)))


def _content_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(np.any(grid != BG, axis=1))[0]
    cols = np.where(np.any(grid != BG, axis=0))[0]
    assert len(rows) > 0 and len(cols) > 0, "expected non-background content"
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _find_panel_layout(grid: np.ndarray) -> dict:
    r0, r1, c0, c1 = _content_bbox(grid)
    core = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    colors = [int(color) for color in np.unique(core) if int(color) != BG]
    layouts = []

    for color in colors:
        full_cols = [col for col in range(core.shape[1]) if np.all(core[:, col] == color)]
        if not full_cols:
            continue
        sep_runs = _group_runs(full_cols)
        panel_count = len(sep_runs) + 1
        if panel_count not in (2, 3):
            continue

        sep_widths = {end - start + 1 for start, end in sep_runs}
        if len(sep_widths) != 1:
            continue

        sep_mask = np.zeros(core.shape[1], dtype=bool)
        for start, end in sep_runs:
            sep_mask[start : end + 1] = True
        if np.any(core[:, ~sep_mask] == color):
            continue

        intervals = []
        left = 0
        valid = True
        for start, end in sep_runs:
            if start <= left:
                valid = False
                break
            intervals.append((left, start))
            left = end + 1
        if not valid or left >= core.shape[1]:
            continue
        intervals.append((left, core.shape[1]))

        widths = {stop - start for start, stop in intervals}
        if len(widths) != 1:
            continue
        panel_width = next(iter(widths))
        if panel_width <= 0:
            continue

        panels = [core[:, start:stop].copy() for start, stop in intervals]
        layouts.append(
            {
                "bbox": (r0, r1, c0, c1),
                "separator_color": int(color),
                "panels": panels,
                "panel_intervals": [(c0 + start, c0 + stop) for start, stop in intervals],
            }
        )

    assert len(layouts) == 1, "expected one separator-defined panel layout"
    return layouts[0]


def _candidate_info(panels: list[np.ndarray], tile_size: int) -> dict | None:
    panel_h, panel_w = map(int, panels[0].shape)
    if panel_h % tile_size != 0 or panel_w % tile_size != 0:
        return None

    tile_rows = panel_h // tile_size
    tile_cols = panel_w // tile_size
    canonical: np.ndarray | None = None
    visible_blocks = 0

    for panel in panels:
        if panel.shape != (panel_h, panel_w):
            return None
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                block = panel[
                    tr * tile_size : (tr + 1) * tile_size,
                    tc * tile_size : (tc + 1) * tile_size,
                ]
                all_bg = bool(np.all(block == BG))
                all_fg = bool(np.all(block != BG))
                if not (all_bg or all_fg):
                    return None
                if all_fg:
                    visible_blocks += 1
                    if canonical is None:
                        canonical = block.copy()
                    elif not np.array_equal(canonical, block):
                        return None

    if canonical is None or visible_blocks == 0:
        return None

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            if not any(
                np.all(
                    panel[
                        tr * tile_size : (tr + 1) * tile_size,
                        tc * tile_size : (tc + 1) * tile_size,
                    ]
                    != BG
                )
                for panel in panels
            ):
                return None

    return {
        "tile_size": int(tile_size),
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "tile": canonical,
    }


def _infer_tile(panels: list[np.ndarray]) -> dict:
    candidates = [info for size in TILE_CHOICES if (info := _candidate_info(panels, size)) is not None]
    assert len(candidates) == 1, "expected exactly one valid tile size"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair complementary panel defects by copying each missing tile from another intact panel.

    Input structure:
    - The background is black.
    - Two or three equal-width rectangular panels are arranged horizontally, separated by one or more full grey columns.
    - Every panel contains the same repeated 2x2 or 3x3 colored tile, but some whole tile positions are replaced by the black background.

    Rule:
    - Locate the grey separator columns and extract the equal-sized panel interiors.
    - Infer whether the repeated unit is 2x2 or 3x3 by requiring every candidate tile block to be either fully visible or fully black, and requiring every tile position to stay visible in at least one panel.
    - Recover the repeated tile from the visible blocks, then repaint every panel as the clean tiling while leaving the grey separator columns and any outer black margin unchanged.

    Key insight:
    - A defect is not repaired from local evidence inside one panel; the missing tile is recovered because another panel keeps that same tile position intact.
    """
    g = np.asarray(grid, dtype=int)
    layout = _find_panel_layout(g)
    tile_info = _infer_tile(layout["panels"])

    out = g.copy()
    r0, r1, _c0, _c1 = layout["bbox"]
    restored_panel = _repeat_tile(tile_info["tile"], tile_info["tile_rows"], tile_info["tile_cols"])
    for start, stop in layout["panel_intervals"]:
        out[r0 : r1 + 1, start:stop] = restored_panel
    return out


def _tile_is_rich(tile: np.ndarray) -> bool:
    size = int(tile.shape[0])
    if len(set(map(int, tile.reshape(-1)))) < 2:
        return False
    if any(len(set(map(int, tile[row, :]))) == 1 for row in range(size)):
        return False
    if any(len(set(map(int, tile[:, col]))) == 1 for col in range(size)):
        return False
    row_patterns = {tuple(map(int, tile[row, :])) for row in range(size)}
    col_patterns = {tuple(map(int, tile[:, col])) for col in range(size)}
    return len(row_patterns) >= 2 and len(col_patterns) >= 2


def _sample_tile(tile_size: int) -> np.ndarray | None:
    for _ in range(200):
        palette_size = random.randint(2, min(4, len(PALETTE)))
        colors = random.sample(PALETTE, palette_size)
        tile = np.array(
            [[random.choice(colors) for _ in range(tile_size)] for _ in range(tile_size)],
            dtype=int,
        )
        if _tile_is_rich(tile):
            return tile
    return None


def _count_options(tile_size: int, panel_count: int, sep_width: int) -> list[tuple[int, int]]:
    options = []
    max_tile_rows = min(5, 30 // tile_size)
    max_tile_cols = min(5, (30 - (panel_count - 1) * sep_width) // (panel_count * tile_size))

    for tile_rows in range(2, max_tile_rows + 1):
        for tile_cols in range(2, max_tile_cols + 1):
            panel_h = tile_rows * tile_size
            panel_w = tile_cols * tile_size
            core_w = panel_count * panel_w + (panel_count - 1) * sep_width
            if panel_h > 30 or core_w > 30:
                continue
            if tile_size == 2 and tile_rows % 3 == 0 and tile_cols % 3 == 0:
                continue
            if tile_size == 3 and tile_rows % 2 == 0 and tile_cols % 2 == 0:
                continue
            options.append((tile_rows, tile_cols))
    return options


def _sample_margins(limit: int) -> tuple[int, int]:
    left = random.randint(0, min(2, limit))
    right = random.randint(0, min(2, limit - left))
    return int(left), int(right)


def _sample_missing_map(panel_count: int, tile_rows: int, tile_cols: int) -> np.ndarray | None:
    total_positions = tile_rows * tile_cols
    for _ in range(200):
        missing = np.zeros((panel_count, tile_rows, tile_cols), dtype=bool)
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                if panel_count == 2:
                    roll = random.random()
                    if roll < 0.25:
                        subset: list[int] = []
                    else:
                        subset = [random.randrange(panel_count)]
                else:
                    roll = random.random()
                    if roll < 0.20:
                        subset = []
                    elif roll < 0.60:
                        subset = [random.randrange(panel_count)]
                    else:
                        subset = random.sample(range(panel_count), 2)
                for panel_idx in subset:
                    missing[panel_idx, tr, tc] = True

        if np.count_nonzero(np.any(missing, axis=0)) < max(1, total_positions // 4):
            continue
        if any(int(np.sum(missing[panel_idx])) == 0 for panel_idx in range(panel_count)):
            continue
        if any(int(np.sum(~missing[panel_idx])) == 0 for panel_idx in range(panel_count)):
            continue
        if any(np.all(missing[:, tr, tc]) for tr in range(tile_rows) for tc in range(tile_cols)):
            continue
        if any(np.all(missing[panel_idx, tr, :]) for panel_idx in range(panel_count) for tr in range(tile_rows)):
            continue
        if any(np.all(missing[panel_idx, :, tc]) for panel_idx in range(panel_count) for tc in range(tile_cols)):
            continue

        if panel_count == 3:
            counts = np.sum(missing, axis=0)
            if not np.any(counts == 1):
                continue
            if not np.any(counts == 2):
                continue

        return missing
    return None


def _apply_missing(
    clean_grid: np.ndarray,
    top: int,
    panel_starts: list[int],
    panel_h: int,
    panel_w: int,
    tile_size: int,
    missing: np.ndarray,
) -> np.ndarray:
    grid = clean_grid.copy()
    tile_rows, tile_cols = map(int, missing.shape[1:])
    for panel_idx, start in enumerate(panel_starts):
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                if not missing[panel_idx, tr, tc]:
                    continue
                rr = top + tr * tile_size
                cc = start + tc * tile_size
                grid[rr : rr + tile_size, cc : cc + tile_size] = BG
        assert np.any(grid[top : top + panel_h, start : start + panel_w] != BG)
    return grid


def _build_clean_example(
    panel_count: int,
    tile_size: int,
    sep_width: int,
    tile_rows: int,
    tile_cols: int,
    tile: np.ndarray,
) -> tuple[np.ndarray, int, list[int], int, int]:
    panel_h = tile_rows * tile_size
    panel_w = tile_cols * tile_size
    core_w = panel_count * panel_w + (panel_count - 1) * sep_width

    top, bottom = _sample_margins(30 - panel_h)
    left, right = _sample_margins(30 - core_w)

    height = top + panel_h + bottom
    width = left + core_w + right
    grid = np.full((height, width), BG, dtype=int)
    panel = _repeat_tile(tile, tile_rows, tile_cols)

    starts = []
    col = left
    for panel_idx in range(panel_count):
        grid[top : top + panel_h, col : col + panel_w] = panel
        starts.append(int(col))
        col += panel_w
        if panel_idx + 1 < panel_count:
            grid[top : top + panel_h, col : col + sep_width] = SEPARATOR
            col += sep_width

    return grid, int(top), starts, int(panel_h), int(panel_w)


def _generate_attempt() -> np.ndarray | None:
    panel_count = random.choice((2, 3))
    tile_size = random.choice(TILE_CHOICES)
    sep_width = random.choice((1, 2))

    options = _count_options(tile_size, panel_count, sep_width)
    if not options:
        return None
    tile_rows, tile_cols = random.choice(options)

    tile = _sample_tile(tile_size)
    if tile is None:
        return None

    clean, top, panel_starts, panel_h, panel_w = _build_clean_example(
        panel_count=panel_count,
        tile_size=tile_size,
        sep_width=sep_width,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile=tile,
    )

    missing = _sample_missing_map(panel_count, tile_rows, tile_cols)
    if missing is None:
        return None

    inp = _apply_missing(
        clean_grid=clean,
        top=top,
        panel_starts=panel_starts,
        panel_h=panel_h,
        panel_w=panel_w,
        tile_size=tile_size,
        missing=missing,
    )

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
