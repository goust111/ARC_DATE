from __future__ import annotations

import random

import numpy as np


BG = 8
SEPARATOR = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7)
CATEGORIES = ("h", "v", "hv", "none")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(cells)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _symmetry_type(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    if horizontal and vertical:
        return "hv"
    if horizontal:
        return "h"
    if vertical:
        return "v"
    return "none"


def _transformed_mask(mask: np.ndarray) -> np.ndarray:
    kind = _symmetry_type(mask)
    if kind == "h":
        return np.flipud(mask)
    if kind == "v":
        return np.fliplr(mask)
    if kind == "hv":
        return np.rot90(mask, 2)
    return mask.copy()


def _absolute_mask_cells(mask: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r_mid = mask.shape[0] // 2
    c_mid = mask.shape[1] // 2
    out: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mask):
        out.add((int(anchor[0] + rr - r_mid), int(anchor[1] + cc - c_mid)))
    return out


def _stamp_mask(out: np.ndarray, mask: np.ndarray, color: int, anchor: tuple[int, int]) -> None:
    for rr, cc in _absolute_mask_cells(mask, anchor):
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
        out[rr, cc] = int(color)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(cols) == 1
    return int(cols[0])


def _extract_templates(grid: np.ndarray, sep_col: int) -> dict[int, np.ndarray]:
    left = grid[:, :sep_col]
    comps = _monochrome_components(left, ignore={BG})
    assert 2 <= len(comps) <= 4
    templates: dict[int, np.ndarray] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color not in (BG, SEPARATOR, MAROON)
        assert color not in templates
        mask = comp["mask"]
        assert 3 <= mask.shape[0] <= 5 and 3 <= mask.shape[1] <= 5
        templates[color] = mask
    return templates


def _extract_markers(grid: np.ndarray, sep_col: int, colors: set[int]) -> dict[int, tuple[int, int]]:
    markers: dict[int, tuple[int, int]] = {}
    right = grid[:, sep_col + 1 :]
    for color in colors:
        pts = np.argwhere(right == int(color))
        assert pts.shape[0] == 1
        r, c = map(int, pts[0])
        markers[color] = (r, c + sep_col + 1)
    assert set(markers) == colors
    return markers


def _adjacent_marker_colors(square_cells: set[tuple[int, int]], markers: dict[int, tuple[int, int]]) -> set[int]:
    adjacent: set[int] = set()
    for color, (mr, mc) in markers.items():
        if any(abs(rr - mr) + abs(cc - mc) == 1 for rr, cc in square_cells):
            adjacent.add(int(color))
    return adjacent


def _validate_maroon_blocks(grid: np.ndarray, sep_col: int, markers: dict[int, tuple[int, int]]) -> None:
    right = grid[:, sep_col + 1 :]
    comps = _monochrome_components(right, ignore={BG})
    maroons = [comp for comp in comps if int(comp["color"]) == MAROON]
    assert maroons
    for comp in maroons:
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        assert len(comp["cells"]) == 4
        cells = {(int(rr), int(cc + sep_col + 1)) for rr, cc in comp["cells"]}
        assert len(_adjacent_marker_colors(cells, markers)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each template object's symmetry, choose the corresponding transform,
    then stamp the transformed object on the matching marker in the canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_col = _find_separator_col(g)
    templates = _extract_templates(g, sep_col)
    markers = _extract_markers(g, sep_col, set(templates))
    _validate_maroon_blocks(g, sep_col, markers)

    out = g.copy()
    for color, mask in templates.items():
        _stamp_mask(out, _transformed_mask(mask), color, markers[color])
    return out


MASK_LIBRARY: dict[str, tuple[np.ndarray, ...]] = {
    "h": (
        _mask_from_rows(
            "111",
            "110",
            "111",
        ),
        _mask_from_rows(
            "11110",
            "00100",
            "00100",
            "00100",
            "11110",
        ),
        _mask_from_rows(
            "11111",
            "00110",
            "00100",
            "00110",
            "11111",
        ),
    ),
    "v": (
        _mask_from_rows(
            "111",
            "111",
            "101",
        ),
        _mask_from_rows(
            "00100",
            "01110",
            "11111",
            "00100",
            "00100",
        ),
        _mask_from_rows(
            "00100",
            "00100",
            "11111",
            "01110",
            "00100",
        ),
    ),
    "hv": (
        _mask_from_rows(
            "010",
            "111",
            "010",
        ),
        _mask_from_rows(
            "00100",
            "00100",
            "11111",
            "00100",
            "00100",
        ),
        _mask_from_rows(
            "01110",
            "11111",
            "11111",
            "11111",
            "01110",
        ),
    ),
    "none": (
        _mask_from_rows(
            "111",
            "110",
            "100",
        ),
        _mask_from_rows(
            "110",
            "111",
            "001",
        ),
        _mask_from_rows(
            "11100",
            "00100",
            "00111",
            "00010",
            "01110",
        ),
        _mask_from_rows(
            "11000",
            "11110",
            "00100",
            "00111",
            "00001",
        ),
    ),
}


def _verify_mask_library() -> None:
    for kind, masks in MASK_LIBRARY.items():
        for mask in masks:
            assert mask.dtype == bool
            assert mask.shape[0] in (3, 5) and mask.shape[1] in (3, 5)
            assert mask[mask.shape[0] // 2, mask.shape[1] // 2]
            assert len(_component_cells(mask)) == 1
            assert _symmetry_type(mask) == kind


_verify_mask_library()


def _bands(length: int) -> list[tuple[int, int]]:
    usable_lo = 1
    usable_hi = length - 2
    mid = (usable_lo + usable_hi) // 2
    bands = [(usable_lo, mid - 1), (mid + 1, usable_hi)]
    assert all(hi - lo + 1 >= 5 for lo, hi in bands)
    return bands


def _slots(height: int, width: int, col_offset: int) -> list[tuple[int, int, int, int]]:
    slots: list[tuple[int, int, int, int]] = []
    for r0, r1 in _bands(height):
        for c0, c1 in _bands(width):
            slots.append((r0, r1, c0 + col_offset, c1 + col_offset))
    return slots


def _top_left_options(slot: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = slot
    h, w = shape
    out: list[tuple[int, int]] = []
    for rr in range(r0, r1 - h + 2):
        for cc in range(c0, c1 - w + 2):
            out.append((rr, cc))
    return out


def _anchor_options(slot: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = slot
    out: list[tuple[int, int]] = []
    for rr in range(r0 + 2, r1 - 1):
        for cc in range(c0 + 2, c1 - 1):
            out.append((rr, cc))
    return out


def _paint_template(grid: np.ndarray, mask: np.ndarray, color: int, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _square_offset_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    footprint = _absolute_mask_cells(mask, (0, 0))
    candidates: list[tuple[int, int]] = []
    for dr0 in range(-2, 2):
        for dc0 in range(-2, 2):
            square = {(dr0 + dr, dc0 + dc) for dr in (0, 1) for dc in (0, 1)}
            if (0, 0) in square:
                continue
            if not any(abs(rr) + abs(cc) == 1 for rr, cc in square):
                continue
            if not (square & footprint):
                continue
            candidates.append((dr0, dc0))
    random.shuffle(candidates)
    return candidates


def _paint_maroon_square(grid: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells = {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}
    for rr, cc in cells:
        grid[rr, cc] = MAROON
    return cells


def _attempt_input() -> np.ndarray | None:
    height = random.randint(13, 17)
    left_width = random.randint(13, 17)
    right_width = random.randint(13, 18)
    sep_col = left_width
    width = left_width + 1 + right_width

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = SEPARATOR

    n_objects = random.randint(2, 4)
    categories = random.sample(CATEGORIES, n_objects)
    colors = random.sample(PALETTE, n_objects)

    objects: list[dict] = []
    for kind, color in zip(categories, colors, strict=False):
        mask = random.choice(MASK_LIBRARY[kind]).copy()
        objects.append(
            {
                "kind": kind,
                "color": int(color),
                "mask": mask,
                "transformed": _transformed_mask(mask),
            }
        )

    left_slots = random.sample(_slots(height, left_width, 0), n_objects)
    right_slots = random.sample(_slots(height, right_width, sep_col + 1), n_objects)

    for obj, slot in zip(objects, left_slots, strict=False):
        options = _top_left_options(slot, obj["mask"].shape)
        if not options:
            return None
        top_left = random.choice(options)
        obj["template_top_left"] = top_left
        _paint_template(grid, obj["mask"], obj["color"], top_left)

    markers: dict[int, tuple[int, int]] = {}
    square_cells_used: set[tuple[int, int]] = set()
    predicted_footprints: dict[int, set[tuple[int, int]]] = {}

    for obj, slot in zip(objects, right_slots, strict=False):
        options = _anchor_options(slot)
        if not options:
            return None
        anchor = random.choice(options)
        markers[obj["color"]] = anchor
        predicted_footprints[obj["color"]] = _absolute_mask_cells(obj["transformed"], anchor)
        grid[anchor] = obj["color"]

    marker_positions = set(markers.values())
    for obj in objects:
        anchor = markers[obj["color"]]
        own_footprint = predicted_footprints[obj["color"]]
        options = []
        for dr0, dc0 in _square_offset_candidates(obj["transformed"]):
            global_top_left = (anchor[0] + dr0, anchor[1] + dc0)
            cells = {(global_top_left[0] + dr, global_top_left[1] + dc) for dr in (0, 1) for dc in (0, 1)}
            if cells & marker_positions:
                continue
            if cells & square_cells_used:
                continue
            if not (cells & own_footprint):
                continue
            if _adjacent_marker_colors(cells, markers) != {obj["color"]}:
                continue
            options.append(global_top_left)
        if not options:
            return None

        square_count = 2 if len(options) >= 2 and random.random() < 0.5 else 1
        random.shuffle(options)
        chosen = options[:square_count]
        for top_left in chosen:
            square_cells_used |= _paint_maroon_square(grid, top_left)

    try:
        out = main(grid)
    except AssertionError:
        return None

    # Every maroon square must be hit by its corresponding stamped object.
    sep = _find_separator_col(grid)
    markers_check = _extract_markers(grid, sep, set(colors))
    right = grid[:, sep + 1 :]
    maroons = [comp for comp in _monochrome_components(right, ignore={BG}) if int(comp["color"]) == MAROON]
    for comp in maroons:
        cells = {(int(rr), int(cc + sep + 1)) for rr, cc in comp["cells"]}
        linked = _adjacent_marker_colors(cells, markers_check)
        assert len(linked) == 1
        linked_color = next(iter(linked))
        if all(int(out[rr, cc]) == MAROON for rr, cc in cells):
            return None
        if not any(int(out[rr, cc]) == linked_color for rr, cc in cells):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
