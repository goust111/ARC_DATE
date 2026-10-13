from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("TL", "TR", "BL", "BR")
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []
    height, width = map(int, grid.shape)

    for row in range(height):
        values = np.unique(grid[row, :])
        if values.size == 1 and int(values[0]) != BG:
            row_candidates.append((row, int(values[0])))

    for col in range(width):
        values = np.unique(grid[:, col])
        if values.size == 1 and int(values[0]) != BG:
            col_candidates.append((col, int(values[0])))

    matches = []
    for row, color_r in row_candidates:
        for col, color_c in col_candidates:
            if color_r == color_c:
                matches.append((int(color_r), int(row), int(col)))

    assert len(matches) == 1, "expected exactly one separator cross"
    return matches[0]


def _quadrant_of(row: int, col: int, cross_row: int, cross_col: int) -> str:
    if row < cross_row and col < cross_col:
        return "TL"
    if row < cross_row and col > cross_col:
        return "TR"
    if row > cross_row and col < cross_col:
        return "BL"
    if row > cross_row and col > cross_col:
        return "BR"
    raise AssertionError("seed must not lie on the separator cross")


def _grow_direction(quadrant: str) -> tuple[int, int]:
    return {
        "TL": (1, 1),
        "TR": (1, -1),
        "BL": (-1, 1),
        "BR": (-1, -1),
    }[quadrant]


def _quadrant_bounds(
    cross_row: int, cross_col: int, height: int, width: int, quadrant: str
) -> tuple[int, int, int, int]:
    if quadrant == "TL":
        return 0, cross_row - 1, 0, cross_col - 1
    if quadrant == "TR":
        return 0, cross_row - 1, cross_col + 1, width - 1
    if quadrant == "BL":
        return cross_row + 1, height - 1, 0, cross_col - 1
    if quadrant == "BR":
        return cross_row + 1, height - 1, cross_col + 1, width - 1
    raise AssertionError("unknown quadrant")


def _extract_groups(
    grid: np.ndarray, sep_color: int, cross_row: int, cross_col: int
) -> list[dict]:
    groups: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (BG, int(sep_color)):
            continue

        coords = np.argwhere(grid == int(color))
        assert coords.size > 0
        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})

        quadrant = _quadrant_of(int(coords[0, 0]), int(coords[0, 1]), cross_row, cross_col)
        assert all(
            _quadrant_of(int(r), int(c), cross_row, cross_col) == quadrant
            for r, c in coords
        ), "same-color seeds must stay in one quadrant"

        dr, dc = _grow_direction(quadrant)
        count = int(coords.shape[0])

        if len(rows) == 1 and len(cols) > 1:
            orient = "horizontal"
        elif len(cols) == 1 and len(rows) > 1:
            orient = "vertical"
        elif count == 1:
            row = rows[0]
            col = cols[0]
            orient = "horizontal" if abs(row - cross_row) <= abs(col - cross_col) else "vertical"
        else:
            raise AssertionError("same-color seeds must lie on one row or one column")

        if orient == "horizontal":
            row = rows[0]
            col0, col1 = cols[0], cols[-1]
            assert count == col1 - col0 + 1, "horizontal seeds must be contiguous"
            anchor_col = col1 if dc == 1 else col0
            anchor = (int(row), int(anchor_col))
            groups.append(
                {
                    "color": color,
                    "quadrant": quadrant,
                    "orient": orient,
                    "dr": dr,
                    "dc": dc,
                    "seed_row": int(row),
                    "col0": int(col0),
                    "col1": int(col1),
                    "count": count,
                    "anchor": anchor,
                }
            )
        else:
            col = cols[0]
            row0, row1 = rows[0], rows[-1]
            assert count == row1 - row0 + 1, "vertical seeds must be contiguous"
            anchor_row = row1 if dr == 1 else row0
            anchor = (int(anchor_row), int(col))
            groups.append(
                {
                    "color": color,
                    "quadrant": quadrant,
                    "orient": orient,
                    "dr": dr,
                    "dc": dc,
                    "seed_col": int(col),
                    "row0": int(row0),
                    "row1": int(row1),
                    "count": count,
                    "anchor": anchor,
                }
            )

    groups.sort(
        key=lambda group: (
            abs(int(group["anchor"][0]) - int(cross_row))
            + abs(int(group["anchor"][1]) - int(cross_col)),
            int(group["anchor"][0]),
            int(group["anchor"][1]),
        )
    )
    return groups


def _apply_groups(
    grid: np.ndarray,
    groups: list[dict],
    sep_color: int,
    *,
    collect_details: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    height, width = map(int, out.shape)
    details: list[dict] = []

    for group in groups:
        color = int(group["color"])
        if group["orient"] == "horizontal":
            seed_row = int(group["seed_row"])
            col0 = int(group["col0"])
            col1 = int(group["col1"])
            length = 1
            while True:
                cand_row = seed_row + int(group["dr"]) * length
                if not (0 <= cand_row < height):
                    stop_reason = "edge"
                    break
                segment = out[cand_row, col0 : col1 + 1]
                if np.any(segment != BG):
                    stop_reason = "separator" if np.any(segment == int(sep_color)) else "occupied"
                    break
                out[cand_row, col0 : col1 + 1] = color
                length += 1

            if collect_details:
                end_row = seed_row + int(group["dr"]) * (length - 1)
                details.append(
                    {
                        "color": color,
                        "quadrant": group["quadrant"],
                        "orient": group["orient"],
                        "anchor": tuple(group["anchor"]),
                        "count": int(group["count"]),
                        "extent": int(length),
                        "stop_reason": stop_reason,
                        "bbox": (min(seed_row, end_row), col0, max(seed_row, end_row), col1),
                    }
                )
        else:
            seed_col = int(group["seed_col"])
            row0 = int(group["row0"])
            row1 = int(group["row1"])
            length = 1
            while True:
                cand_col = seed_col + int(group["dc"]) * length
                if not (0 <= cand_col < width):
                    stop_reason = "edge"
                    break
                segment = out[row0 : row1 + 1, cand_col]
                if np.any(segment != BG):
                    stop_reason = "separator" if np.any(segment == int(sep_color)) else "occupied"
                    break
                out[row0 : row1 + 1, cand_col] = color
                length += 1

            if collect_details:
                end_col = seed_col + int(group["dc"]) * (length - 1)
                details.append(
                    {
                        "color": color,
                        "quadrant": group["quadrant"],
                        "orient": group["orient"],
                        "anchor": tuple(group["anchor"]),
                        "count": int(group["count"]),
                        "extent": int(length),
                        "stop_reason": stop_reason,
                        "bbox": (row0, min(seed_col, end_col), row1, max(seed_col, end_col)),
                    }
                )

    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each color's seed line toward the center cross until it hits the separator or earlier growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_color, cross_row, cross_col = _find_separator_cross(g)
    groups = _extract_groups(g, sep_color, cross_row, cross_col)
    out, _ = _apply_groups(g, groups, sep_color, collect_details=False)
    return out


def _cells_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for row, col in cells:
        if (row, col) in occupied:
            return False
        for rr in range(row - 1, row + 2):
            for cc in range(col - 1, col + 2):
                if (rr, cc) in occupied and (rr, cc) not in cell_set:
                    return False
    return True


def _build_seed_spec(
    quadrant: str,
    orient: str,
    count: int,
    seed_line: int,
    center_edge: int,
    cross_row: int,
    cross_col: int,
    height: int,
    width: int,
) -> dict | None:
    dr, dc = _grow_direction(quadrant)
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(cross_row, cross_col, height, width, quadrant)

    if orient == "horizontal":
        row = int(seed_line)
        if dc == 1:
            col1 = int(center_edge)
            col0 = col1 - int(count) + 1
        else:
            col0 = int(center_edge)
            col1 = col0 + int(count) - 1
        if not (r_lo <= row <= r_hi and c_lo <= col0 <= col1 <= c_hi):
            return None
        cells = [(row, col) for col in range(col0, col1 + 1)]
        return {
            "quadrant": quadrant,
            "orient": orient,
            "count": int(count),
            "seed_line": row,
            "center_edge": int(center_edge),
            "cells": cells,
        }

    col = int(seed_line)
    if dr == 1:
        row1 = int(center_edge)
        row0 = row1 - int(count) + 1
    else:
        row0 = int(center_edge)
        row1 = row0 + int(count) - 1
    if not (c_lo <= col <= c_hi and r_lo <= row0 <= row1 <= r_hi):
        return None
    cells = [(row, col) for row in range(row0, row1 + 1)]
    return {
        "quadrant": quadrant,
        "orient": orient,
        "count": int(count),
        "seed_line": col,
        "center_edge": int(center_edge),
        "cells": cells,
    }


def _random_single_spec(
    quadrant: str,
    orient: str,
    cross_row: int,
    cross_col: int,
    height: int,
    width: int,
) -> dict | None:
    dr, dc = _grow_direction(quadrant)
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(cross_row, cross_col, height, width, quadrant)

    if orient == "horizontal":
        max_count = min(5, c_hi - c_lo + 1)
        if max_count < 2:
            return None
        count = random.randint(2, max_count)
        seed_row = random.randint(r_lo, r_hi)
        if dc == 1:
            center_min, center_max = c_lo + count - 1, c_hi
        else:
            center_min, center_max = c_lo, c_hi - count + 1
        if center_min > center_max:
            return None
        center_edge = random.randint(center_min, center_max)
        return _build_seed_spec(
            quadrant, orient, count, seed_row, center_edge, cross_row, cross_col, height, width
        )

    max_count = min(5, r_hi - r_lo + 1)
    if max_count < 2:
        return None
    count = random.randint(2, max_count)
    seed_col = random.randint(c_lo, c_hi)
    if dr == 1:
        center_min, center_max = r_lo + count - 1, r_hi
    else:
        center_min, center_max = r_lo, r_hi - count + 1
    if center_min > center_max:
        return None
    center_edge = random.randint(center_min, center_max)
    return _build_seed_spec(
        quadrant, orient, count, seed_col, center_edge, cross_row, cross_col, height, width
    )


def _random_blocking_pair(
    quadrant: str,
    orient: str,
    cross_row: int,
    cross_col: int,
    height: int,
    width: int,
) -> list[dict] | None:
    dr, dc = _grow_direction(quadrant)
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(cross_row, cross_col, height, width, quadrant)

    if orient == "horizontal":
        band_width = c_hi - c_lo + 1
        band_height = r_hi - r_lo + 1
        if band_width < 2 or band_height < 4:
            return None

        count_near = random.randint(2, min(5, band_width))
        count_far = random.randint(2, min(5, band_width))
        max_count = max(count_near, count_far)
        if dc == 1:
            center_min, center_max = c_lo + max_count - 1, c_hi
        else:
            center_min, center_max = c_lo, c_hi - max_count + 1
        if center_min > center_max:
            return None
        center_edge = random.randint(center_min, center_max)

        if dr == 1:
            near_min, near_max = r_lo + 2, r_hi - 1
            if near_min > near_max:
                return None
            seed_near = random.randint(near_min, near_max)
            far_min, far_max = r_lo, seed_near - 2
        else:
            near_min, near_max = r_lo + 1, r_hi - 2
            if near_min > near_max:
                return None
            seed_near = random.randint(near_min, near_max)
            far_min, far_max = seed_near + 2, r_hi
        if far_min > far_max:
            return None
        seed_far = random.randint(far_min, far_max)

        near_spec = _build_seed_spec(
            quadrant,
            orient,
            count_near,
            seed_near,
            center_edge,
            cross_row,
            cross_col,
            height,
            width,
        )
        far_spec = _build_seed_spec(
            quadrant,
            orient,
            count_far,
            seed_far,
            center_edge,
            cross_row,
            cross_col,
            height,
            width,
        )
        if near_spec is None or far_spec is None:
            return None
        return [near_spec, far_spec]

    band_width = c_hi - c_lo + 1
    band_height = r_hi - r_lo + 1
    if band_width < 4 or band_height < 2:
        return None

    count_near = random.randint(2, min(5, band_height))
    count_far = random.randint(2, min(5, band_height))
    max_count = max(count_near, count_far)
    if dr == 1:
        center_min, center_max = r_lo + max_count - 1, r_hi
    else:
        center_min, center_max = r_lo, r_hi - max_count + 1
    if center_min > center_max:
        return None
    center_edge = random.randint(center_min, center_max)

    if dc == 1:
        near_min, near_max = c_lo + 2, c_hi - 1
        if near_min > near_max:
            return None
        seed_near = random.randint(near_min, near_max)
        far_min, far_max = c_lo, seed_near - 2
    else:
        near_min, near_max = c_lo + 1, c_hi - 2
        if near_min > near_max:
            return None
        seed_near = random.randint(near_min, near_max)
        far_min, far_max = seed_near + 2, c_hi
    if far_min > far_max:
        return None
    seed_far = random.randint(far_min, far_max)

    near_spec = _build_seed_spec(
        quadrant,
        orient,
        count_near,
        seed_near,
        center_edge,
        cross_row,
        cross_col,
        height,
        width,
    )
    far_spec = _build_seed_spec(
        quadrant,
        orient,
        count_far,
        seed_far,
        center_edge,
        cross_row,
        cross_col,
        height,
        width,
    )
    if near_spec is None or far_spec is None:
        return None
    return [near_spec, far_spec]


def _place_specs(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color: int,
    specs: list[dict],
    placements: list[dict],
) -> bool:
    pending = set(occupied)
    for spec in specs:
        if not _cells_clear(spec["cells"], pending):
            return False
        pending.update(spec["cells"])

    for spec in specs:
        for row, col in spec["cells"]:
            grid[row, col] = int(color)
            occupied.add((int(row), int(col)))
        placements.append(
            {
                "color": int(color),
                "quadrant": spec["quadrant"],
                "orient": spec["orient"],
            }
        )
    return True


def _specs_clear(specs: list[dict], occupied: set[tuple[int, int]]) -> bool:
    pending = set(occupied)
    for spec in specs:
        if not _cells_clear(spec["cells"], pending):
            return False
        pending.update(spec["cells"])
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 22)
    width = random.randint(13, 22)
    cross_row = random.randint(4, height - 5)
    cross_col = random.randint(4, width - 5)
    sep_color = 5

    grid = np.full((height, width), BG, dtype=int)
    grid[cross_row, :] = sep_color
    grid[:, cross_col] = sep_color

    occupied: set[tuple[int, int]] = set()
    placements: list[dict] = []

    want_blocking = random.random() < 0.55
    target_groups = random.randint(3, 6)
    colors = list(PALETTE)
    random.shuffle(colors)
    if target_groups > len(colors):
        return None

    color_index = 0
    if want_blocking:
        block_orient = random.choice(("horizontal", "vertical"))
        block_quads = list(QUADRANTS)
        random.shuffle(block_quads)
        placed_block = False
        for quadrant in block_quads:
            pair = _random_blocking_pair(
                quadrant, block_orient, cross_row, cross_col, height, width
            )
            if pair is None:
                continue
            if _specs_clear(pair, occupied):
                for spec in pair:
                    placed = _place_specs(
                        grid, occupied, colors[color_index], [spec], placements
                    )
                    assert placed
                    color_index += 1
                placed_block = True
                break
        if not placed_block:
            return None

    while len(placements) < target_groups and color_index < len(colors):
        missing_orients = {"horizontal", "vertical"} - {p["orient"] for p in placements}
        orient = (
            random.choice(sorted(missing_orients))
            if missing_orients and target_groups - len(placements) <= len(missing_orients)
            else random.choice(("horizontal", "vertical"))
        )

        preferred_quads = list(QUADRANTS)
        random.shuffle(preferred_quads)
        if len({p["quadrant"] for p in placements}) < 2 and target_groups - len(placements) <= 1:
            preferred_quads = [q for q in preferred_quads if q not in {p["quadrant"] for p in placements}] + [
                q for q in preferred_quads if q in {p["quadrant"] for p in placements}
            ]

        placed = False
        for quadrant in preferred_quads:
            spec = _random_single_spec(quadrant, orient, cross_row, cross_col, height, width)
            if spec is None:
                continue
            if _place_specs(grid, occupied, colors[color_index], [spec], placements):
                color_index += 1
                placed = True
                break
        if not placed:
            return None

    if len(placements) != target_groups:
        return None
    if len({p["quadrant"] for p in placements}) < 2:
        return None
    if len({p["orient"] for p in placements}) < 2:
        return None

    groups = _extract_groups(grid, sep_color, cross_row, cross_col)
    out, details = _apply_groups(grid, groups, sep_color, collect_details=True)
    if len(details) != target_groups:
        return None
    if not any(detail["extent"] > 1 for detail in details):
        return None
    if not any(detail["stop_reason"] == "separator" for detail in details):
        return None
    if want_blocking and not any(detail["stop_reason"] == "occupied" for detail in details):
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != BG) > int(height * width * 0.55):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
