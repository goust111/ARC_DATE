from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


AZURE = 8
RED = 2
MAROON = 9
YELLOW = 4
SPECIAL_COLORS = {AZURE, RED, MAROON, YELLOW}
FRAGMENT_COLORS = (0, 1, 3, 5, 6, 7)

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) != int(bg):
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            arr = np.array(coords, dtype=int)
            r_min = int(arr[:, 0].min())
            r_max = int(arr[:, 0].max())
            c_min = int(arr[:, 1].min())
            c_max = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r_min, c_min, r_max, c_max),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _tight_mask(coords: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r_min = int(arr[:, 0].min())
    r_max = int(arr[:, 0].max())
    c_min = int(arr[:, 1].min())
    c_max = int(arr[:, 1].max())
    mask = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=bool)
    mask[arr[:, 0] - r_min, arr[:, 1] - c_min] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _component_signature(grid: np.ndarray, comp: dict) -> dict:
    coords = [tuple(map(int, rc)) for rc in comp["coords"]]
    marker_cells = [(r, c) for r, c in coords if int(grid[r, c]) in (RED, MAROON)]
    assert len(marker_cells) == 1
    marker = marker_cells[0]
    other = [(r, c) for r, c in coords if (r, c) != marker]
    assert 2 <= len(other) <= 5
    body_colors = {int(grid[r, c]) for r, c in other}
    assert len(body_colors) == 1
    return {
        "coords": coords,
        "marker": marker,
        "marker_color": int(grid[marker]),
        "body_color": next(iter(body_colors)),
        "shape_key": _mask_key(_tight_mask(other)),
    }


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        assert 0 <= r < h and 0 <= c < w
        assert int(grid[r, c]) == AZURE
        grid[r, c] = int(color)


def _transform_pair(
    out: np.ndarray,
    leftish: dict,
    rightish: dict,
) -> None:
    ra, ca = map(int, leftish["marker"])
    rb, cb = map(int, rightish["marker"])

    if ra == rb:
        assert ca != cb
        if ca > cb:
            leftish, rightish = rightish, leftish
            ra, ca = map(int, leftish["marker"])
            rb, cb = map(int, rightish["marker"])
        gap = int(cb - ca)
        assert gap >= 2 and gap % 2 == 0
        center = (int(ra), int((ca + cb) // 2))
        shift = int((gap - 2) // 2)
        delta_a = (0, shift)
        delta_b = (0, -shift)
    else:
        assert ca == cb
        if ra > rb:
            leftish, rightish = rightish, leftish
            ra, ca = map(int, leftish["marker"])
            rb, cb = map(int, rightish["marker"])
        gap = int(rb - ra)
        assert gap >= 2 and gap % 2 == 0
        center = (int((ra + rb) // 2), int(ca))
        shift = int((gap - 2) // 2)
        delta_a = (shift, 0)
        delta_b = (-shift, 0)

    for item, (dr, dc) in ((leftish, delta_a), (rightish, delta_b)):
        moved = [(int(r + dr), int(c + dc)) for r, c in item["coords"] if (r, c) != item["marker"]]
        _paint_cells(out, moved, int(item["body_color"]))

    cr, cc = center
    assert int(out[cr, cc]) == AZURE
    out[cr, cc] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair red- and maroon-marked fragments by their marker-free shapes, then slide each pair together and fuse the markers.

    Input structure:
    - The background is Azure.
    - Several small 4-connected multicolor fragments are scattered on the canvas, each with 3-6 cells total.
    - Every fragment contains exactly one Red marker cell or one Maroon marker cell, while its other cells all share that fragment's own color.
    - The number of Red-marked and Maroon-marked fragments is equal, and removing each marker leaves a unique cross-polarity shape match.

    Rule:
    - Extract every fragment and remove its single Red or Maroon marker cell.
    - Pair one Red fragment with one Maroon fragment when the remaining tight-cropped binary shapes are exactly identical, with no rotation or reflection.
    - For each matched pair, inspect the two marker positions; they are aligned either horizontally or vertically.
    - Slide the pair along that axis toward each other until the markers would sit one cell to either side of their midpoint, erase those markers, and paint one Yellow cell at the midpoint instead.
    - Keep all non-marker colored cells unchanged apart from the rigid translation, and clear the original fragment locations back to Azure.

    Key insight:
    - The marker colors only give the polarities; the actual pairing comes from comparing the shapes after deleting the markers, and only then do the marker coordinates reveal the motion axis.
    """
    g = np.array(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    assert bg == AZURE

    comps = _find_components(g, bg)
    assert comps and len(comps) % 2 == 0

    grouped: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in comps:
        sig = _component_signature(g, comp)
        grouped.setdefault(sig["shape_key"], []).append(sig)

    out = np.full_like(g, AZURE)
    for members in grouped.values():
        assert len(members) == 2
        colors = Counter(int(m["marker_color"]) for m in members)
        assert colors[RED] == 1 and colors[MAROON] == 1
        a, b = members
        _transform_pair(out, a, b)

    return out


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen[start] = True
    count = 0
    while queue:
        r, c = queue.popleft()
        count += 1
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return count == int(mask.sum())


def _build_template(rows: tuple[str, ...]) -> dict | None:
    raw_mask = _mask_from_rows(rows)
    mask = _tight_mask(np.argwhere(raw_mask))
    area = int(mask.sum())
    if not (2 <= area <= 5) or not _mask_connected(mask):
        return None

    axes: list[str] = []
    if np.array_equal(mask, np.fliplr(mask)):
        rows_lr = [r for r in range(mask.shape[0]) if mask[r, 0] and mask[r, -1]]
        if rows_lr:
            axes.append("h")
    else:
        rows_lr = []

    if np.array_equal(mask, np.flipud(mask)):
        cols_ud = [c for c in range(mask.shape[1]) if mask[0, c] and mask[-1, c]]
        if cols_ud:
            axes.append("v")
    else:
        cols_ud = []

    if not axes:
        return None

    return {
        "mask": mask,
        "cells": [tuple(map(int, rc)) for rc in np.argwhere(mask)],
        "key": _mask_key(mask),
        "axes": tuple(axes),
        "h_attach_rows": rows_lr,
        "v_attach_cols": cols_ud,
    }


RAW_TEMPLATES = (
    ("11",),
    ("111",),
    ("1", "1"),
    ("1", "1", "1"),
    ("11", "11"),
    ("111", "010"),
    ("010", "111"),
    ("010", "111", "010"),
    ("101", "111"),
    ("111", "101"),
    ("10", "11", "10"),
    ("01", "11", "01"),
    ("11", "01", "11"),
    ("11", "10", "11"),
    ("010", "110", "010"),
    ("010", "011", "010"),
)

_templates_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
for _rows in RAW_TEMPLATES:
    _tpl = _build_template(_rows)
    if _tpl is None:
        continue
    _key = _tpl["key"]
    if _key not in _templates_by_key:
        _templates_by_key[_key] = _tpl
        continue
    prev = _templates_by_key[_key]
    prev["axes"] = tuple(sorted(set(prev["axes"]) | set(_tpl["axes"])))
    prev["h_attach_rows"] = sorted(set(prev["h_attach_rows"]) | set(_tpl["h_attach_rows"]))
    prev["v_attach_cols"] = sorted(set(prev["v_attach_cols"]) | set(_tpl["v_attach_cols"]))

TEMPLATES = list(_templates_by_key.values())


def _build_fragment_specs(template: dict, axis: str) -> tuple[dict, dict]:
    mask = template["mask"]
    cells = template["cells"]
    h, w = mask.shape

    if axis == "h":
        attach_row = random.choice(template["h_attach_rows"])
        first = {"cells": cells, "marker": (attach_row, w)}
        second = {"cells": [(r, c + 1) for r, c in cells], "marker": (attach_row, 0)}
        return first, second

    attach_col = random.choice(template["v_attach_cols"])
    first = {"cells": cells, "marker": (h, attach_col)}
    second = {"cells": [(r + 1, c) for r, c in cells], "marker": (0, attach_col)}
    return first, second


def _absolute_cells(spec: dict, marker_pos: tuple[int, int], include_marker: bool) -> list[tuple[int, int]]:
    mr, mc = map(int, spec["marker"])
    gr, gc = map(int, marker_pos)
    out = [(int(gr + r - mr), int(gc + c - mc)) for r, c in spec["cells"]]
    if include_marker:
        out.append((int(gr), int(gc)))
    return out


def _coords_bounds(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _mark(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _place_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    occupied_in: np.ndarray,
    occupied_out: np.ndarray,
    template: dict,
    axis: str,
    colors: tuple[int, int],
) -> bool:
    h, w = input_grid.shape
    for _ in range(PAIR_ATTEMPTS):
        first_spec, second_spec = _build_fragment_specs(template, axis)
        slide = random.randint(1, 5)

        if axis == "h":
            first_in = _absolute_cells(first_spec, (0, -1 - slide), include_marker=True)
            second_in = _absolute_cells(second_spec, (0, 1 + slide), include_marker=True)
            first_out = _absolute_cells(first_spec, (0, -1), include_marker=False)
            second_out = _absolute_cells(second_spec, (0, 1), include_marker=False)
        else:
            first_in = _absolute_cells(first_spec, (-1 - slide, 0), include_marker=True)
            second_in = _absolute_cells(second_spec, (1 + slide, 0), include_marker=True)
            first_out = _absolute_cells(first_spec, (-1, 0), include_marker=False)
            second_out = _absolute_cells(second_spec, (1, 0), include_marker=False)

        input_rel = first_in + second_in
        output_rel = first_out + second_out + [(0, 0)]

        min_r1, max_r1, min_c1, max_c1 = _coords_bounds(input_rel)
        min_r2, max_r2, min_c2, max_c2 = _coords_bounds(output_rel)
        min_r = min(min_r1, min_r2)
        max_r = max(max_r1, max_r2)
        min_c = min(min_c1, min_c2)
        max_c = max(max_c1, max_c2)

        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        center_r = random.randint(row_lo, row_hi)
        center_c = random.randint(col_lo, col_hi)
        placed_in = _translate(input_rel, center_r, center_c)
        placed_out = _translate(output_rel, center_r, center_c)
        if not _can_place(placed_in, occupied_in):
            continue
        if not _can_place(placed_out, occupied_out):
            continue

        first_in_global = _translate(first_in, center_r, center_c)
        second_in_global = _translate(second_in, center_r, center_c)
        first_out_global = _translate(first_out, center_r, center_c)
        second_out_global = _translate(second_out, center_r, center_c)

        marker_shuffle = [RED, MAROON]
        random.shuffle(marker_shuffle)

        _paint_cells(input_grid, first_in_global[:-1], colors[0])
        _paint_cells(input_grid, second_in_global[:-1], colors[1])
        _paint_cells(input_grid, [first_in_global[-1]], marker_shuffle[0])
        _paint_cells(input_grid, [second_in_global[-1]], marker_shuffle[1])

        _paint_cells(output_grid, first_out_global, colors[0])
        _paint_cells(output_grid, second_out_global, colors[1])
        _paint_cells(output_grid, [(center_r, center_c)], YELLOW)

        _mark(occupied_in, placed_in)
        _mark(occupied_out, placed_out)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    n_pairs = random.choice((2, 3))
    axes = [random.choice(("h", "v")) for _ in range(n_pairs)]

    input_grid = np.full((h, w), AZURE, dtype=int)
    output_grid = np.full((h, w), AZURE, dtype=int)
    occupied_in = np.zeros((h, w), dtype=bool)
    occupied_out = np.zeros((h, w), dtype=bool)

    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    available_colors = list(FRAGMENT_COLORS)
    random.shuffle(available_colors)

    for i, axis in enumerate(axes):
        candidates = [tpl for tpl in TEMPLATES if axis in tpl["axes"] and tpl["key"] not in used_keys]
        if not candidates:
            return None
        template = random.choice(candidates)
        used_keys.add(template["key"])
        colors = (available_colors[2 * i], available_colors[2 * i + 1])
        if not _place_pair(input_grid, output_grid, occupied_in, occupied_out, template, axis, colors):
            return None

    comps = _find_components(input_grid, AZURE)
    if len(comps) != 2 * n_pairs:
        return None
    if Counter(comp["area"] for comp in comps) and not all(3 <= int(comp["area"]) <= 6 for comp in comps):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnetic-fragment sample")
