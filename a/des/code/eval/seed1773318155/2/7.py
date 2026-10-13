from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
DIR_ORDER = ("top", "bottom", "left", "right")
CORNERS = ("tl", "tr", "bl", "br")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _set_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _border_positions(bbox):
        grid[r, c] = int(color)


def _corner_coord(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1
    if corner == "bl":
        return r1, c0
    if corner == "br":
        return r1, c1
    raise ValueError(corner)


def _midpoints(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    return {
        "top": (r0, cm),
        "bottom": (r1, cm),
        "left": (rm, c0),
        "right": (rm, c1),
    }


def _all_border_equal(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    return all(int(grid[r, c]) == int(color) for r, c in _border_positions(bbox))


def _all_border_non_bg(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> bool:
    return all(int(grid[r, c]) != int(bg) for r, c in _border_positions(bbox))


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border_values = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    bg = Counter(map(int, border_values)).most_common(1)[0][0]

    occupied = g != bg
    coords = np.argwhere(occupied)
    assert coords.size > 0, "expected a non-empty puzzle structure"
    outer_bbox = _bbox(coords)

    rings_outer_to_inner: list[tuple[int, int, int, int]] = []
    bbox = outer_bbox
    while True:
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "ring bbox too small"
        assert _all_border_non_bg(g, bbox, bg), "every ring border cell must be non-background"
        rings_outer_to_inner.append(bbox)

        inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        assert inner_bbox[0] <= inner_bbox[2] and inner_bbox[1] <= inner_bbox[3], "invalid inner bbox"
        if _all_border_non_bg(g, inner_bbox, bg):
            core_bbox = inner_bbox
            break

        assert _all_border_equal(g, inner_bbox, bg), "non-innermost rings must have a one-cell background gap"
        bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)

    rings = list(reversed(rings_outer_to_inner))
    assert 3 <= len(rings) <= 5, "expected 3 to 5 nested rings"

    cr0, cc0, cr1, cc1 = core_bbox
    core_colors = set(map(int, g[cr0 : cr1 + 1, cc0 : cc1 + 1].ravel()))
    assert bg not in core_colors and len(core_colors) == 1, "core rectangle must be solid and monochrome"
    core_color = int(next(iter(core_colors)))

    expected_mask = np.zeros_like(occupied)
    for ring_bbox in rings:
        for r, c in _border_positions(ring_bbox):
            expected_mask[r, c] = True
    expected_mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = True
    assert np.array_equal(occupied, expected_mask), "occupied cells must be exactly the rings plus the core"

    ring_color_counts: Counter[int] = Counter()
    for ring_bbox in rings:
        for r, c in _border_positions(ring_bbox):
            ring_color_counts[int(g[r, c])] += 1
    grey = ring_color_counts.most_common(1)[0][0]
    assert grey != bg and grey != core_color, "ring base color must differ from background and core"

    marker_colors: list[int] = []
    for ring_bbox in rings[:-1]:
        corners = [_corner_coord(ring_bbox, corner) for corner in CORNERS]
        specials = [int(g[r, c]) for r, c in corners if int(g[r, c]) != grey]
        assert len(specials) == 1, "each non-outer ring must have exactly one corner marker"
        marker_colors.append(int(specials[0]))
    assert len(set(marker_colors)) == len(marker_colors), "corner marker colors must be unique"

    edge_pos = _midpoints(outer_bbox)
    edge_map = {direction: int(g[pos]) for direction, pos in edge_pos.items()}
    edge_colors = list(edge_map.values())
    assert len(set(edge_colors)) == 4, "outer edge markers must be four distinct colors"

    return {
        "grid": g,
        "bg": int(bg),
        "grey": int(grey),
        "core_bbox": core_bbox,
        "core_color": int(core_color),
        "rings": rings,
        "marker_colors": marker_colors,
        "edge_map": edge_map,
        "edge_pos": edge_pos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular rings by chaining inner corner markers through the outer edge legend.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    grey = scene["grey"]
    core_color = scene["core_color"]
    rings = scene["rings"]
    marker_colors = scene["marker_colors"]
    edge_map = scene["edge_map"]

    color_to_dir = {color: direction for direction, color in edge_map.items()}
    final_colors = [grey] * len(rings)
    final_colors[0] = core_color

    for k in range(1, len(rings) - 1):
        inner_marker = marker_colors[k - 1]
        assert inner_marker in color_to_dir, "every inner corner marker must appear on the outer edge legend"
        direction = color_to_dir[inner_marker]
        if direction == "top":
            final_colors[k] = marker_colors[k]
        elif direction == "bottom":
            final_colors[k] = core_color
        elif direction == "left":
            final_colors[k] = final_colors[k - 1]
        else:
            final_colors[k] = grey if k == len(rings) - 2 else marker_colors[k + 1]

    out = g.copy()
    for idx, bbox in enumerate(rings[:-1]):
        _set_border(out, bbox, final_colors[idx])

    _set_border(out, rings[-1], grey)
    for direction, pos in scene["edge_pos"].items():
        out[pos] = edge_map[direction]
    return out


def _build_expected(
    grid: np.ndarray,
    rings: list[tuple[int, int, int, int]],
    core_color: int,
    marker_colors: list[int],
    edge_map: dict[str, int],
) -> np.ndarray:
    color_to_dir = {color: direction for direction, color in edge_map.items()}
    final_colors = [GREY] * len(rings)
    final_colors[0] = core_color

    for k in range(1, len(rings) - 1):
        direction = color_to_dir[marker_colors[k - 1]]
        if direction == "top":
            final_colors[k] = marker_colors[k]
        elif direction == "bottom":
            final_colors[k] = core_color
        elif direction == "left":
            final_colors[k] = final_colors[k - 1]
        else:
            final_colors[k] = GREY if k == len(rings) - 2 else marker_colors[k + 1]

    out = grid.copy()
    for idx, bbox in enumerate(rings[:-1]):
        _set_border(out, bbox, final_colors[idx])
    _set_border(out, rings[-1], GREY)
    for direction, pos in _midpoints(rings[-1]).items():
        out[pos] = edge_map[direction]
    return out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray]:
    n_rings = random.randint(3, 5)
    core_h = random.choice((3, 5, 7))
    core_w = random.choice((3, 5, 7))

    outer_h = core_h + 2 + 4 * (n_rings - 1)
    outer_w = core_w + 2 + 4 * (n_rings - 1)
    pad_top = random.randint(1, 4)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_right = random.randint(1, 4)
    height = outer_h + pad_top + pad_bottom
    width = outer_w + pad_left + pad_right

    grid = np.full((height, width), BG, dtype=int)
    outer_r0 = pad_top
    outer_c0 = pad_left
    core_r0 = outer_r0 + 2 * (n_rings - 1) + 1
    core_c0 = outer_c0 + 2 * (n_rings - 1) + 1
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1
    core_bbox = (core_r0, core_c0, core_r1, core_c1)

    need_extras = 4 - (n_rings - 1)
    special_colors = random.sample(PALETTE, 1 + (n_rings - 1) + need_extras)
    core_color = int(special_colors[0])
    marker_colors = [int(color) for color in special_colors[1 : n_rings]]
    extra_edge_colors = [int(color) for color in special_colors[n_rings :]]
    edge_colors = marker_colors + extra_edge_colors
    random.shuffle(edge_colors)
    edge_map = {direction: edge_colors[idx] for idx, direction in enumerate(DIR_ORDER)}

    rings: list[tuple[int, int, int, int]] = []
    for idx in range(n_rings):
        delta = 1 + 2 * idx
        bbox = (core_r0 - delta, core_c0 - delta, core_r1 + delta, core_c1 + delta)
        rings.append(bbox)
        _set_border(grid, bbox, GREY)

    grid[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1] = core_color

    for idx, bbox in enumerate(rings[:-1]):
        grid[_corner_coord(bbox, random.choice(CORNERS))] = marker_colors[idx]

    for direction, pos in _midpoints(rings[-1]).items():
        grid[pos] = edge_map[direction]

    expected = _build_expected(grid, rings, core_color, marker_colors, edge_map)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, expected = _attempt_generate()
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
