from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PATH_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray, allowed: set[int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) not in allowed:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _same_color_components(grid, {GREY}):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1):
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or np.any(inner == GREY):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one grey output canvas, got {len(candidates)}"
    return max(candidates)[1]


def _block_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top, left),
        (top + 1, left),
        (top, left + 1),
        (top + 1, left + 1),
    }


def _find_endpoint_blocks(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    tops: set[tuple[int, int]] = set()
    for r, c in comp:
        if r + 1 >= h or c + 1 >= w:
            continue
        if (
            int(grid[r, c]) == color
            and int(grid[r + 1, c]) == color
            and int(grid[r, c + 1]) == color
            and int(grid[r + 1, c + 1]) == color
        ):
            tops.add((r, c))
    blocks = sorted(tops)
    assert len(blocks) == 2, f"expected two 2x2 endpoint blocks for color {color}, got {blocks}"
    return blocks


def _extract_paths(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    allowed = set(PATH_COLORS)
    for comp in _same_color_components(grid, allowed):
        color = int(grid[comp[0]])
        assert all(int(grid[r, c]) == color for r, c in comp), "components must be monochrome"
        blocks = _find_endpoint_blocks(grid, comp, color)
        endpoint_cells = set()
        for br, bc in blocks:
            endpoint_cells |= _block_cells(br, bc)
        path_cells = [cell for cell in comp if cell not in endpoint_cells]
        assert path_cells, f"path color {color} must contain non-endpoint cells"
        marker_cells: set[tuple[int, int]] = set()
        for r, c in path_cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == AZURE:
                    marker_cells.add((nr, nc))
        infos.append(
            {
                "color": color,
                "length": len(path_cells),
                "markers": len(marker_cells),
                "sort_key": (min(bc for _br, bc in blocks), min(br for br, _bc in blocks)),
            }
        )
    infos.sort(key=lambda item: item["sort_key"])
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each curved endpoint-to-endpoint path into the grey canvas and recolor its prefix by adjacent azure markers.

    Input structure:
    - The background is black.
    - Several monochrome 1-pixel-wide colored paths connect pairs of same-colored 2x2 endpoint blocks.
    - Azure single-pixel markers sit orthogonally adjacent to path pixels, and one grey hollow rectangle on the grid edge acts as the output canvas.

    Rule:
    - Find the grey edge-touching rectangular canvas and clear everything else to black in the output.
    - For each monochrome path component, identify its two 2x2 endpoint blocks; the remaining same-color pixels form the path, whose count is the path length.
    - Count how many azure markers are orthogonally adjacent to that path, sort the paths by their leftmost endpoint position, and draw one horizontal segment per path from the canvas's top-left interior downward.
    - Each segment has the original path length, its first m pixels are azure where m is the marker count, and its remaining pixels keep the path color.

    Key insight:
    - The curved geometry is discarded except for two retained quantities: path length becomes horizontal extent, and the number of adjacent azure markers becomes the recolored prefix length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvas = _find_canvas(g)
    infos = _extract_paths(g)

    r0, c0, r1, c1 = canvas
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= len(infos), "canvas must have enough rows for all straightened paths"
    assert inner_w >= max(info["length"] for info in infos), "canvas must have enough columns for all straightened paths"

    out = np.full_like(g, BG)
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY

    for idx, info in enumerate(infos):
        row = r0 + 1 + idx
        start = c0 + 1
        length = int(info["length"])
        azure_len = min(int(info["markers"]), length)
        if azure_len:
            out[row, start : start + azure_len] = AZURE
        if azure_len < length:
            out[row, start + azure_len : start + length] = int(info["color"])

    return out


def _build_slot_path(tracks: int, corridor_w: int, start_side: str) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], int, int]:
    assert tracks in (1, 3)
    assert start_side in ("left", "right")
    slot_h = 2 * tracks
    slot_w = corridor_w + 4
    left_col = 2
    right_col = slot_w - 3
    path: list[tuple[int, int]] = []
    for i in range(tracks):
        row = 2 * i
        left_to_right = (start_side == "left" and i % 2 == 0) or (start_side == "right" and i % 2 == 1)
        if left_to_right:
            row_cells = [(row, c) for c in range(left_col, right_col + 1)]
        else:
            row_cells = [(row, c) for c in range(right_col, left_col - 1, -1)]
        if i > 0:
            path.append((row - 1, row_cells[0][1]))
        path.extend(row_cells)

    start_r, start_c = path[0]
    end_r, end_c = path[-1]
    if start_side == "left":
        assert start_c == left_col
        start_block = (start_r, 0)
        assert end_c == right_col
        end_block = (end_r, slot_w - 2)
    else:
        assert start_c == right_col
        start_block = (start_r, slot_w - 2)
        assert end_c == left_col
        end_block = (end_r, 0)
    return path, start_block, end_block, slot_h, slot_w


def _candidate_markers(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    endpoint_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in path_cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) != BG:
                continue
            adj_path = sum((pr, pc) in path_cells for pr, pc in _neighbors4(nr, nc))
            adj_endpoint = sum((pr, pc) in endpoint_cells for pr, pc in _neighbors4(nr, nc))
            if adj_path == 1 and adj_endpoint == 0:
                candidates.add((nr, nc))
    return sorted(candidates)


def _draw_expected(shape: tuple[int, int], canvas: tuple[int, int, int, int], items: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = canvas
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    ordered = sorted(items, key=lambda item: item["sort_key"])
    for idx, item in enumerate(ordered):
        row = r0 + 1 + idx
        start = c0 + 1
        length = int(item["length"])
        azure_len = min(int(item["markers"]), length)
        if azure_len:
            out[row, start : start + azure_len] = AZURE
        if azure_len < length:
            out[row, start + azure_len : start + length] = int(item["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_paths = random.choice((2, 3))
    colors = random.sample(PATH_COLORS, n_paths)
    gap_x = random.randint(2, 3)

    specs: list[dict] = []
    for color in colors:
        tracks = random.choice((1, 3))
        if n_paths == 2:
            corridor_w = random.randint(4, 7)
        else:
            corridor_w = random.randint(3, 4)
        path, start_block, end_block, slot_h, slot_w = _build_slot_path(tracks, corridor_w, random.choice(("left", "right")))
        specs.append(
            {
                "color": color,
                "path_local": path,
                "start_block_local": start_block,
                "end_block_local": end_block,
                "slot_h": slot_h,
                "slot_w": slot_w,
                "length": len(path),
            }
        )

    strip_w = sum(spec["slot_w"] for spec in specs) + gap_x * (n_paths - 1)
    max_slot_h = max(spec["slot_h"] for spec in specs)
    max_len = max(spec["length"] for spec in specs)

    canvas_inner_w = max_len + random.randint(0, 2)
    canvas_inner_h = n_paths + random.randint(0, 2)
    canvas_h = canvas_inner_h + 2
    canvas_w = canvas_inner_w + 2
    gap_y = random.randint(1, 3)

    width = max(strip_w + random.randint(2, 4), canvas_w + random.randint(0, 3))
    height = canvas_h + gap_y + max_slot_h + random.randint(1, 3)
    if width > 30 or height > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    canvas_top = random.choice((True, False))
    if canvas_top:
        canvas_r0 = 0
        free_r0 = canvas_h + gap_y
    else:
        canvas_r0 = height - canvas_h
        free_r0 = 0
    canvas_c0 = random.randint(0, width - canvas_w)
    canvas = (canvas_r0, canvas_c0, canvas_r0 + canvas_h - 1, canvas_c0 + canvas_w - 1)
    r0, c0, r1, c1 = canvas
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    free_h = height - canvas_h - gap_y
    if free_h < max_slot_h:
        return None

    slack = width - strip_w
    left_margin = random.randint(1, max(1, slack - 1))
    if left_margin + strip_w > width - 1:
        left_margin = width - strip_w
    cursor = left_margin

    placed: list[dict] = []
    for spec in specs:
        extra_rows = free_h - spec["slot_h"]
        if extra_rows >= 2:
            slot_top = free_r0 + random.randint(1, extra_rows - 1)
        else:
            slot_top = free_r0 + random.randint(0, extra_rows)
        slot_left = cursor
        cursor += spec["slot_w"] + gap_x

        start_r, start_c = spec["start_block_local"]
        end_r, end_c = spec["end_block_local"]
        start_block = (slot_top + start_r, slot_left + start_c)
        end_block = (slot_top + end_r, slot_left + end_c)
        endpoint_cells = _block_cells(*start_block) | _block_cells(*end_block)
        path_cells = {(slot_top + pr, slot_left + pc) for pr, pc in spec["path_local"]}

        for pr, pc in path_cells:
            grid[pr, pc] = spec["color"]
        for br, bc in endpoint_cells:
            grid[br, bc] = spec["color"]

        leftmost_endpoint_col = min(start_block[1], end_block[1])
        topmost_endpoint_row = min(start_block[0], end_block[0])
        placed.append(
            {
                "color": spec["color"],
                "path_cells": path_cells,
                "endpoint_cells": endpoint_cells,
                "blocks": (start_block, end_block),
                "length": spec["length"],
                "sort_key": (leftmost_endpoint_col, topmost_endpoint_row),
            }
        )

    for item in placed:
        candidates = _candidate_markers(grid, item["path_cells"], item["endpoint_cells"])
        marker_cap = min(len(candidates), item["length"], 5)
        if marker_cap == 0 or random.random() < 0.5:
            marker_count = 0
        else:
            marker_count = random.randint(1, marker_cap)
        chosen = random.sample(candidates, marker_count) if marker_count else []
        for mr, mc in chosen:
            grid[mr, mc] = AZURE
        item["markers"] = marker_count

    expected = _draw_expected(grid.shape, canvas, placed)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid path-straightening input")
