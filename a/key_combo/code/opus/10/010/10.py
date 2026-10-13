import itertools
import random
from functools import lru_cache

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_RETRIES = 400
GEN_ROW_COUNT_CHOICES = (3, 4)
GEN_COL_COUNT = 3
GEN_ROW_HEIGHTS = (3, 4, 5)
GEN_COL_WIDTHS = (4, 5, 6)
GEN_EMPTY_PROB = 0.6
GEN_MAX_EMPTY = 2


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _compress_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if out.shape[0] > 1:
        keep_rows = [0]
        for r in range(1, out.shape[0]):
            if not np.array_equal(out[r], out[r - 1]):
                keep_rows.append(r)
        out = out[keep_rows, :]
    if out.shape[1] > 1:
        keep_cols = [0]
        for c in range(1, out.shape[1]):
            if not np.array_equal(out[:, c], out[:, c - 1]):
                keep_cols.append(c)
        out = out[:, keep_cols]
    return out


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(variant) for variant in _orientations(mask)}


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((int(total),),)
    out = []
    min_head = 1
    max_head = int(total) - int(parts) + 1
    for head in range(min_head, max_head + 1):
        for tail in _compositions(int(total - head), int(parts - 1)):
            out.append((int(head),) + tail)
    return tuple(out)


def _regions_between(lines: list[int], size: int) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    prev = -1
    for line in lines + [int(size)]:
        start = prev + 1
        end = int(line) - 1
        if start <= end:
            regions.append((start, end))
        prev = int(line)
    return regions


def _parse_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREEN)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREEN)]
    assert full_rows and full_cols

    expected = np.zeros(grid.shape, dtype=bool)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    assert np.array_equal(grid == GREEN, expected)

    row_regions = _regions_between(full_rows, grid.shape[0])
    col_regions = _regions_between(full_cols, grid.shape[1])
    assert len(row_regions) in GEN_ROW_COUNT_CHOICES
    assert len(col_regions) == GEN_COL_COUNT
    return row_regions, col_regions


def _cell_view(grid: np.ndarray, row_region: tuple[int, int], col_region: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_region
    c0, c1 = col_region
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _extract_cell_shape(cell: np.ndarray) -> np.ndarray | None:
    assert np.all((cell == BLACK) | (cell == YELLOW))
    yellow = cell == YELLOW
    if not yellow.any():
        return None
    return _crop_mask(yellow)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    row_regions, col_regions = _parse_table(g)
    proto_cell = _cell_view(g, row_regions[0], col_regions[0])
    prototype = _extract_cell_shape(proto_cell)
    assert prototype is not None
    proto_key = _mask_key(prototype)
    proto_signatures = _signature_set(prototype)

    cells = []
    for ri, row_region in enumerate(row_regions):
        for ci, col_region in enumerate(col_regions):
            cell = _cell_view(g, row_region, col_region)
            shape = _extract_cell_shape(cell)
            info = {
                "row": ri,
                "col": ci,
                "row_region": row_region,
                "col_region": col_region,
                "cell": cell.copy(),
                "empty": shape is None,
                "is_prototype": ri == 0 and ci == 0,
            }
            if shape is not None:
                compressed = _compress_mask(shape)
                raw_key = _mask_key(shape)
                compressed_key = _mask_key(compressed)
                info.update(
                    {
                        "shape": shape,
                        "compressed": compressed,
                        "raw_key": raw_key,
                        "compressed_key": compressed_key,
                        "matched": compressed_key in proto_signatures,
                        "needs_compression": raw_key not in proto_signatures,
                        "needs_rigid_transform": compressed_key != proto_key,
                    }
                )
            cells.append(info)

    return {
        "grid": g,
        "row_regions": row_regions,
        "col_regions": col_regions,
        "prototype": prototype,
        "prototype_key": proto_key,
        "prototype_signatures": proto_signatures,
        "cells": cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each non-empty table cell blue or red according to whether its run-compressed yellow shape matches the top-left prototype.

    Input structure:
    - A black background is partitioned by complete green horizontal and vertical separator lines into a 3x3 or 4x3 table of rectangular cells.
    - The top-left cell contains a yellow prototype shape; other cells may contain one yellow shape or be empty.
    - Cell sizes vary by row and column, so matching shapes can appear stretched inside different cells.

    Rule:
    - Preserve every green separator line unchanged.
    - Treat the yellow bbox crop in the top-left cell as the prototype.
    - For each other non-empty cell, crop its yellow bbox, collapse adjacent identical rows and adjacent identical columns, and compare the compressed mask against all rotations and reflections of the prototype.
    - If it matches, fill that cell's black background pixels blue; if it does not, fill that cell's black background pixels red; yellow pixels stay yellow, and empty cells stay black.

    Key insight:
    - The same underlying symbol is stretched differently by different table-cell dimensions, so shape identity is recovered only after collapsing repeated row and column runs.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    out = analysis["grid"].copy()

    for info in analysis["cells"]:
        if info["is_prototype"] or info["empty"]:
            continue
        fill_color = BLUE if info["matched"] else RED
        r0, r1 = info["row_region"]
        c0, c1 = info["col_region"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BLACK] = fill_color

    return out


def _shape_pool() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen = set()
    for h in range(2, 4):
        for w in range(2, 5):
            cells = [(r, c) for r in range(h) for c in range(w)]
            max_area = min(6, h * w - 1)
            for area in range(3, max_area + 1):
                for combo in itertools.combinations(cells, area):
                    mask = np.zeros((h, w), dtype=bool)
                    for r, c in combo:
                        mask[r, c] = True
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if not _is_connected(mask):
                        continue
                    if not np.array_equal(_compress_mask(mask), mask):
                        continue
                    key = _mask_key(mask)
                    if key in seen:
                        continue
                    seen.add(key)
                    shapes.append(mask.copy())
    return shapes


SHAPE_POOL = _shape_pool()
ASYM_PROTOTYPES = [shape for shape in SHAPE_POOL if len(_signature_set(shape)) > 1]


def _choose_band_sizes(count: int, values: tuple[int, ...]) -> list[int]:
    for _ in range(200):
        sizes = [random.choice(values) for _ in range(count)]
        if len(set(sizes)) > 1:
            return sizes
    raise RuntimeError("failed to choose non-uniform band sizes")


def _line_positions(bands: list[int]) -> list[int]:
    positions = []
    cursor = int(bands[0])
    for width in bands[1:]:
        positions.append(cursor)
        cursor += 1 + int(width)
    return positions


def _build_table(row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((height, width), dtype=int)
    grid[_line_positions(row_heights), :] = GREEN
    grid[:, _line_positions(col_widths)] = GREEN
    return grid


def _fit_shapes_for_cell(shape: np.ndarray, cell_h: int, cell_w: int) -> list[dict]:
    options = []
    for variant in _orientations(shape):
        vh, vw = variant.shape
        if vh > cell_h or vw > cell_w:
            continue
        for row_total in range(vh, cell_h + 1):
            for col_total in range(vw, cell_w + 1):
                if (cell_h - row_total) + (cell_w - col_total) < 1:
                    continue
                for row_reps in _compositions(row_total, vh):
                    for col_reps in _compositions(col_total, vw):
                        stretched = np.repeat(np.repeat(variant, row_reps, axis=0), col_reps, axis=1)
                        for top in range(cell_h - stretched.shape[0] + 1):
                            for left in range(cell_w - stretched.shape[1] + 1):
                                options.append(
                                    {
                                        "mask": stretched,
                                        "top": int(top),
                                        "left": int(left),
                                        "row_reps": tuple(int(v) for v in row_reps),
                                        "col_reps": tuple(int(v) for v in col_reps),
                                        "variant_key": _mask_key(variant),
                                    }
                                )
    return options


def _paint_cell(grid: np.ndarray, row_region: tuple[int, int], col_region: tuple[int, int], mask: np.ndarray, top: int, left: int) -> None:
    r0, _r1 = row_region
    c0, _c1 = col_region
    h, w = mask.shape
    patch = grid[r0 + top : r0 + top + h, c0 + left : c0 + left + w]
    patch[mask] = YELLOW


def _cell_coords(row_count: int, col_count: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(row_count) for c in range(col_count) if not (r == 0 and c == 0)]


def _assign_statuses(row_count: int, col_count: int) -> dict[tuple[int, int], str]:
    coords = _cell_coords(row_count, col_count)
    cell_total = len(coords)
    for _ in range(300):
        max_empty = min(GEN_MAX_EMPTY, cell_total - 5)
        if max_empty > 0 and random.random() < GEN_EMPTY_PROB:
            empty_count = random.randint(1, max_empty)
        else:
            empty_count = 0
        remaining = cell_total - empty_count
        mismatch_count = random.randint(2, remaining - 3)
        match_count = remaining - mismatch_count
        if match_count < 3:
            continue
        tags = ["match"] * match_count + ["mismatch"] * mismatch_count + ["empty"] * empty_count
        random.shuffle(tags)
        status = {coord: tag for coord, tag in zip(coords, tags)}

        row_mix = False
        for r in range(row_count):
            row_tags = [status[(r, c)] for c in range(col_count) if (r, c) in status and status[(r, c)] != "empty"]
            if "match" in row_tags and "mismatch" in row_tags:
                row_mix = True
                break

        col_mix = False
        for c in range(col_count):
            col_tags = [status[(r, c)] for r in range(row_count) if (r, c) in status and status[(r, c)] != "empty"]
            if "match" in col_tags and "mismatch" in col_tags:
                col_mix = True
                break

        if row_mix and col_mix:
            return status
    raise RuntimeError("failed to assign mixed statuses")


def _generator_attempt() -> np.ndarray | None:
    row_count = random.choice(GEN_ROW_COUNT_CHOICES)
    col_count = GEN_COL_COUNT
    row_heights = _choose_band_sizes(row_count, GEN_ROW_HEIGHTS)
    col_widths = _choose_band_sizes(col_count, GEN_COL_WIDTHS)

    grid = _build_table(row_heights, col_widths)
    row_regions, col_regions = _parse_table(grid)

    proto_cell_h = row_heights[0]
    proto_cell_w = col_widths[0]
    proto_candidates = [shape for shape in ASYM_PROTOTYPES if shape.shape[0] <= proto_cell_h and shape.shape[1] <= proto_cell_w and proto_cell_h * proto_cell_w > shape.sum()]
    if not proto_candidates:
        return None
    prototype = random.choice(proto_candidates)
    proto_options = _fit_shapes_for_cell(prototype, proto_cell_h, proto_cell_w)
    if not proto_options:
        return None
    proto_choice = random.choice(proto_options)
    _paint_cell(grid, row_regions[0], col_regions[0], proto_choice["mask"], proto_choice["top"], proto_choice["left"])

    proto_signatures = _signature_set(prototype)
    status = _assign_statuses(row_count, col_count)

    need_compression_cell = random.choice([coord for coord, tag in status.items() if tag == "match"])
    rigid_transform_candidates = [coord for coord, tag in status.items() if tag == "match" and coord != need_compression_cell]
    need_transform_cell = random.choice(rigid_transform_candidates) if rigid_transform_candidates else need_compression_cell

    for coord, tag in status.items():
        r, c = coord
        cell_h = row_heights[r]
        cell_w = col_widths[c]
        if tag == "empty":
            continue

        if tag == "match":
            candidates = _fit_shapes_for_cell(prototype, cell_h, cell_w)
            if coord == need_compression_cell:
                candidates = [
                    item
                    for item in candidates
                    if (max(item["row_reps"]) > 1 or max(item["col_reps"]) > 1)
                    and _mask_key(item["mask"]) not in proto_signatures
                ]
            if coord == need_transform_cell:
                candidates = [item for item in candidates if item["variant_key"] != _mask_key(prototype)]
            if not candidates:
                return None
            choice = random.choice(candidates)
        else:
            distractor_shapes = [
                shape
                for shape in SHAPE_POOL
                if _signature_set(shape).isdisjoint(proto_signatures) and shape.shape[0] <= cell_h and shape.shape[1] <= cell_w
            ]
            if not distractor_shapes:
                return None
            random.shuffle(distractor_shapes)
            choice = None
            for distractor in distractor_shapes[:40]:
                options = _fit_shapes_for_cell(distractor, cell_h, cell_w)
                if not options:
                    continue
                picked = random.choice(options)
                if _mask_key(_compress_mask(picked["mask"])) in proto_signatures:
                    continue
                choice = picked
                break
            if choice is None:
                return None

        _paint_cell(grid, row_regions[r], col_regions[c], choice["mask"], choice["top"], choice["left"])

    if not _validate_semantics(grid):
        return None
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except Exception:
        return False

    row_heights = [r1 - r0 + 1 for r0, r1 in analysis["row_regions"]]
    col_widths = [c1 - c0 + 1 for c0, c1 in analysis["col_regions"]]
    if len(set(row_heights)) < 2 or len(set(col_widths)) < 2:
        return False

    nonproto = [cell for cell in analysis["cells"] if not cell["is_prototype"]]
    nonempty = [cell for cell in nonproto if not cell["empty"]]
    matches = [cell for cell in nonempty if cell["matched"]]
    mismatches = [cell for cell in nonempty if not cell["matched"]]
    empties = [cell for cell in nonproto if cell["empty"]]

    if len(matches) < 3 or len(mismatches) < 2:
        return False
    if len(empties) > GEN_MAX_EMPTY:
        return False
    if not any(cell["needs_compression"] for cell in matches):
        return False
    if not any(cell["needs_rigid_transform"] for cell in matches):
        return False

    row_mix = False
    for r in range(len(analysis["row_regions"])):
        row_cells = [cell for cell in nonempty if cell["row"] == r]
        if any(cell["matched"] for cell in row_cells) and any(not cell["matched"] for cell in row_cells):
            row_mix = True
            break
    if not row_mix:
        return False

    col_mix = False
    for c in range(len(analysis["col_regions"])):
        col_cells = [cell for cell in nonempty if cell["col"] == c]
        if any(cell["matched"] for cell in col_cells) and any(not cell["matched"] for cell in col_cells):
            col_mix = True
            break
    if not col_mix:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    added_blue = int((out == BLUE).sum())
    added_red = int((out == RED).sum())
    if added_blue <= 0 or added_red <= 0:
        return False

    for cell in matches:
        r0, r1 = cell["row_region"]
        c0, c1 = cell["col_region"]
        region_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((region_in == BLACK) & (region_out != BLUE)):
            return False
    for cell in mismatches:
        r0, r1 = cell["row_region"]
        c0, c1 = cell["col_region"]
        region_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((region_in == BLACK) & (region_out != RED)):
            return False
    for cell in empties:
        r0, r1 = cell["row_region"]
        c0, c1 = cell["col_region"]
        if not np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
