from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste


BG = Color.BLACK
ANCHOR = Color.GREY
FRAGMENT = Color.ORANGE
LEGEND_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON]

ANCHOR_SHAPES = [
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
]

FRAGMENT_LIBRARY: dict[int, list[np.ndarray]] = {
    3: [
        np.array([[1, 0], [1, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    ],
}


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, int(color), BG).astype(int)


def _orient(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    if random.random() < 0.5:
        out = out[::-1, :]
    if random.random() < 0.5:
        out = out[:, ::-1]
    return np.array(out, copy=True)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.array(mask, copy=True)
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != 0)
    return [(int(top + r), int(left + c)) for r, c in coords]


def _is_nonrectangular_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    area = int(coords.shape[0])
    return area != (r1 - r0 + 1) * (c1 - c0 + 1)


def _component_centroid_key(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _find_anchor_component(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid == ANCHOR, connectivity=4)
    assert len(comps) == 1, "expected one grey anchor component"
    return comps[0]


def _anchor_axes(anchor_coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(anchor_coords)
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "anchor bbox must have integer center"
    ar = (r0 + r1) // 2
    ac = (c0 + c1) // 2
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[anchor_coords[:, 0] - r0, anchor_coords[:, 1] - c0] = True
    assert np.array_equal(mask, mask[::-1, :]), "anchor must be horizontally symmetric"
    assert np.array_equal(mask, mask[:, ::-1]), "anchor must be vertically symmetric"
    return int(ar), int(ac)


def _legend_from_bottom_row(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    cols = [c for c, value in enumerate(row) if int(value) != bg]
    colors = [int(row[c]) for c in cols]
    assert len(colors) in (2, 3), "legend must contain 2 or 3 colors"
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    assert all(color not in (ANCHOR, FRAGMENT) for color in colors), "legend colors must differ from anchor/fragments"
    if len(cols) > 1:
        gaps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        assert all(gap == 2 for gap in gaps), "legend points must be separated by one blank cell"
    return colors


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    anchor = _find_anchor_component(g)
    ar, ac = _anchor_axes(anchor)
    legend = _legend_from_bottom_row(g, bg)

    fragments = []
    for coords in components_mask(g == FRAGMENT, connectivity=4):
        rows = coords[:, 0]
        cols = coords[:, 1]
        assert np.all(rows < ar) and np.all(cols < ac), "orange fragments must be confined to the upper-left quadrant"
        fragments.append(coords)

    assert len(fragments) == len(legend), "fragment count must equal legend count"
    areas = [int(coords.shape[0]) for coords in fragments]
    assert len(set(areas)) == len(areas), "fragment areas must be unique"

    ranked = sorted(
        fragments,
        key=lambda coords: (-int(coords.shape[0]), bbox_coords(coords)),
    )

    out = np.array(g, copy=True)
    for rank, coords in enumerate(ranked):
        color = int(legend[rank])
        for r, c in coords:
            points = (
                (int(r), int(c)),
                (int(r), int(2 * ac - c)),
                (int(2 * ar - r), int(c)),
                (int(2 * ar - r), int(2 * ac - c)),
            )
            for rr, cc in points:
                out[rr, cc] = color
    out[anchor[:, 0], anchor[:, 1]] = ANCHOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor upper-left orange fragments by area rank from the bottom legend, then mirror them across the grey anchor's two symmetry axes.

    Input structure:
    - Black background with one grey 4-connected anchor object near the center; the anchor is symmetric across both a horizontal axis and a vertical axis.
    - Only the upper-left quadrant contains 2 or 3 disconnected orange fragments, each 4-connected, non-rectangular, and with a distinct pixel area.
    - The bottom row contains the same number of single-pixel legend colors, ordered left-to-right and separated by one black cell.

    Rule:
    - Use the grey anchor's bbox center as the horizontal and vertical reflection axes.
    - Rank the orange fragments by area from largest to smallest.
    - Read the legend colors from left to right; recolor rank 1 with the first legend color, rank 2 with the second, and so on.
    - Keep those recolored fragments in the upper-left quadrant and copy every recolored pixel to its horizontal, vertical, and diagonal mirror positions across the anchor axes.
    - Preserve the grey anchor and the legend row.

    Key insight:
    - The final mirrored colors are not the original fragment color: area ranking must be resolved first, then the legend maps each rank to a new color, and only that recolored result is symmetrically propagated.
    """
    return _solve(grid)


def _expected_output_from_parts(
    grid: np.ndarray,
    ar: int,
    ac: int,
    fragments: list[dict],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    ranked = sorted(fragments, key=lambda item: -int(item["area"]))
    for color, item in zip(legend_colors, ranked):
        for r, c in item["cells"]:
            for rr, cc in (
                (r, c),
                (r, 2 * ac - c),
                (2 * ar - r, c),
                (2 * ar - r, 2 * ac - c),
            ):
                out[rr, cc] = int(color)
    return out


def _placement_orders(fragments: list[dict], ar: int, ac: int) -> dict[str, list[int]]:
    with_index = list(enumerate(fragments))
    return {
        "area": [i for i, _ in sorted(with_index, key=lambda kv: -int(kv[1]["area"]))],
        "left_right": [i for i, _ in sorted(with_index, key=lambda kv: _component_centroid_key(kv[1]["coords"])[1])],
        "top_bottom": [i for i, _ in sorted(with_index, key=lambda kv: _component_centroid_key(kv[1]["coords"])[0])],
        "distance": [
            i
            for i, _ in sorted(
                with_index,
                key=lambda kv: abs(_component_centroid_key(kv[1]["coords"])[0] - ar)
                + abs(_component_centroid_key(kv[1]["coords"])[1] - ac),
            )
        ],
    }


def _passes_shortcut_checks(fragments: list[dict], ar: int, ac: int) -> bool:
    orders = _placement_orders(fragments, ar, ac)
    area_order = orders["area"]
    mismatches = sum(area_order != orders[name] for name in ("left_right", "top_bottom", "distance"))
    if len(fragments) == 2:
        return mismatches >= 1
    return mismatches >= 2 and area_order != orders["left_right"] and area_order != orders["top_bottom"]


def _place_fragments(
    grid: np.ndarray,
    ar: int,
    ac: int,
    fragment_patches: list[np.ndarray],
) -> list[dict] | None:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    occ[grid != BG] = True
    padded_occ = _dilate8(occ)
    placed: list[dict] = []

    order = list(range(len(fragment_patches)))
    random.shuffle(order)
    for patch_index in order:
        patch = fragment_patches[patch_index]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        row_max = ar - ph - 2
        col_max = ac - pw - 2
        if row_max < 1 or col_max < 1:
            return None

        for top in range(1, row_max + 1):
            for left in range(1, col_max + 1):
                cells = _patch_cells(patch, top, left)
                if any(padded_occ[r, c] for r, c in cells):
                    continue
                mirrored = []
                for r, c in cells:
                    copies = {
                        (r, c),
                        (r, 2 * ac - c),
                        (2 * ar - r, c),
                        (2 * ar - r, 2 * ac - c),
                    }
                    mirrored.extend(copies)
                if len(mirrored) != len(set(mirrored)):
                    continue
                if any(rr < 0 or rr >= h - 1 or cc < 0 or cc >= w for rr, cc in mirrored):
                    continue
                candidates.append((top, left))

        if not candidates:
            return None

        top, left = random.choice(candidates)
        colored = _colorize(patch, FRAGMENT)
        paste(grid, colored, top, left, transparent=BG)
        cells = _patch_cells(patch, top, left)
        coords = np.array(cells, dtype=int)
        placed.append(
            {
                "coords": coords,
                "cells": cells,
                "area": int(coords.shape[0]),
            }
        )
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        occ |= mask
        padded_occ = _dilate8(occ)

    placed.sort(key=lambda item: (bbox_coords(item["coords"])[0], bbox_coords(item["coords"])[1]))
    return placed


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    ar: int,
    ac: int,
    fragments: list[dict],
    legend_colors: list[int],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    anchor = _find_anchor_component(grid)
    if not (6 <= len(anchor) <= 12):
        return False
    if _anchor_axes(anchor) != (ar, ac):
        return False

    if len(fragments) != len(legend_colors):
        return False
    if len(set(legend_colors)) != len(legend_colors):
        return False

    anchor_set = {tuple(map(int, p)) for p in anchor}
    for item in fragments:
        coords = item["coords"]
        area = int(coords.shape[0])
        if area < 3 or area > 6:
            return False
        if not _is_nonrectangular_component(coords):
            return False
        if not np.all(coords[:, 0] < ar) or not np.all(coords[:, 1] < ac):
            return False
        for r, c in coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (int(r) + dr, int(c) + dc) in anchor_set:
                    return False

    if len({int(item["area"]) for item in fragments}) != len(fragments):
        return False
    if not _passes_shortcut_checks(fragments, ar, ac):
        return False

    output_colors = set(int(v) for v in np.unique(expected))
    if not set(legend_colors).issubset(output_colors):
        return False
    if expected[-1].tolist() != grid[-1].tolist():
        return False
    return True


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.choice([21, 23, 25, 27])
    w = random.choice([19, 21, 23, 25, 27])
    ar = h // 2 - 1
    ac = w // 2

    grid = np.full((h, w), BG, dtype=int)

    anchor_mask = random.choice(ANCHOR_SHAPES)
    ah, aw = anchor_mask.shape
    anchor_r0 = ar - ah // 2
    anchor_c0 = ac - aw // 2
    paste(grid, _colorize(anchor_mask, ANCHOR), anchor_r0, anchor_c0, transparent=BG)

    n_fragments = random.choice([2, 3])
    areas = random.sample(sorted(FRAGMENT_LIBRARY), n_fragments)
    fragment_patches = [_orient(random.choice(FRAGMENT_LIBRARY[area])) for area in areas]

    fragments = _place_fragments(grid, ar, ac, fragment_patches)
    if fragments is None:
        return None
    if not _passes_shortcut_checks(fragments, ar, ac):
        return None

    legend_colors = random.sample(LEGEND_COLORS, n_fragments)
    legend_width = 2 * n_fragments - 1
    start_col = random.randint(1, w - legend_width - 2)
    legend_cols = [start_col + 2 * i for i in range(n_fragments)]
    for color, col in zip(legend_colors, legend_cols):
        grid[h - 1, col] = int(color)

    expected = _expected_output_from_parts(grid, ar, ac, fragments, legend_colors)
    if not _validate_instance(grid, expected, ar, ac, fragments, legend_colors):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ranked-recolor symmetry instance")
