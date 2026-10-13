from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
LAYER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    return bool(np.all(on_border))


def _find_canvas_frame(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            if not _is_rect_frame(comp):
                continue
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            if not bool(np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG)):
                continue
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((area, color, box))
    assert len(candidates) == 1, "expected exactly one empty rectangular frame"
    _area, color, box = candidates[0]
    return int(color), box


def _find_sequence_strip(
    g: np.ndarray, frame_color: int
) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for col in (0, w - 1):
        coords = [
            (r, col)
            for r in range(h)
            if int(g[r, col]) not in (BG, int(frame_color))
        ]
        if coords:
            candidates.append((col, coords))
    assert len(candidates) == 1, "expected a single edge color strip"
    col, coords = candidates[0]
    coords.sort()
    colors = [int(g[r, col]) for r, _c in coords]
    assert len(colors) >= 2 and len(colors) == len(set(colors))
    return colors, set(coords)


def _layer_group(
    g: np.ndarray, color: int, ignored: set[tuple[int, int]]
) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(g == int(color))
    kept = [(int(r), int(c)) for r, c in coords if (int(r), int(c)) not in ignored]
    assert kept, "each strip color must have matching fragments"
    arr = np.asarray(kept, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    rel = arr - np.array([r0, c0], dtype=int)
    return rel, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _solve(g: np.ndarray) -> np.ndarray:
    frame_color, box = _find_canvas_frame(g)
    strip_colors, ignored = _find_sequence_strip(g, frame_color)
    fr0, fc0, fr1, fc1 = box
    inner_r0 = fr0 + 1
    inner_c0 = fc0 + 1
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1

    out = np.zeros_like(g)
    out[fr0, fc0 : fc1 + 1] = frame_color
    out[fr1, fc0 : fc1 + 1] = frame_color
    out[fr0 : fr1 + 1, fc0] = frame_color
    out[fr0 : fr1 + 1, fc1] = frame_color

    for color in strip_colors:
        rel, h, w = _layer_group(g, int(color), ignored)
        assert h <= inner_h and w <= inner_w
        assert (inner_h - h) % 2 == 0 and (inner_w - w) % 2 == 0
        top = inner_r0 + (inner_h - h) // 2
        left = inner_c0 + (inner_w - w) // 2
        for rr, cc in rel:
            out[top + int(rr), left + int(cc)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered color layers onto the empty framed canvas in strip order.

    Input structure:
    - The background is black.
    - One large grey rectangular border encloses an empty black canvas area.
    - Several small 4-connected monochrome fragments are scattered elsewhere; fragments sharing a color belong to one layer.
    - A one-cell-wide vertical strip on the left or right edge lists the participating layer colors from top to bottom.

    Rule:
    - Read the edge strip from top to bottom as the bottom-to-top stacking order.
    - For each strip color, gather all non-strip pixels of that color and treat their union as one layer while keeping all relative offsets unchanged.
    - Center each layer's bounding box inside the framed canvas interior and place the layers in strip order.
    - Later layers overwrite earlier ones; everything outside the frame becomes black in the output.

    Key insight:
    - The edge strip is not another object: it is the legend that both selects the layer colors and encodes the occlusion order, while each color's scattered fragments already preserve that layer's internal geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g)


def _expanded_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        out.update(_neighbors8(r, c))
    return out


def _grow_fragment(
    allowed: set[tuple[int, int]],
    anchor: tuple[int, int],
    area: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if anchor not in allowed:
        return None
    blocked = _expanded_cells(occupied)
    if anchor in blocked:
        return None

    shape = {anchor}
    while len(shape) < int(area):
        frontier = set()
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in shape and (nr, nc) not in blocked:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        weights = []
        for r, c in frontier:
            dist = abs(r - anchor[0]) + abs(c - anchor[1])
            weights.append(4 if dist <= 1 else 1)
        choice = random.choices(list(frontier), weights=weights, k=1)[0]
        shape.add(choice)
    return shape


def _window_cells(
    r0: int, r1: int, c0: int, c1: int
) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _make_layer(inner_h: int, inner_w: int) -> dict | None:
    min_h = 5 if inner_h % 2 == 1 else 6
    min_w = 5 if inner_w % 2 == 1 else 6
    h_options = [h for h in range(min_h, inner_h + 1, 2)]
    w_options = [w for w in range(min_w, inner_w + 1, 2)]
    h = random.choice(h_options)
    w = random.choice(w_options)

    center = ((h - 1) // 2, (w - 1) // 2)
    corner_mode = random.choice(["tl_br", "tr_bl"])

    if corner_mode == "tl_br":
        corner_anchors = [(0, 0), (h - 1, w - 1)]
        corner_windows = [
            _window_cells(0, min(2, h - 1), 0, min(2, w - 1)),
            _window_cells(max(0, h - 3), h - 1, max(0, w - 3), w - 1),
        ]
    else:
        corner_anchors = [(0, w - 1), (h - 1, 0)]
        corner_windows = [
            _window_cells(0, min(2, h - 1), max(0, w - 3), w - 1),
            _window_cells(max(0, h - 3), h - 1, 0, min(2, w - 1)),
        ]

    center_window = _window_cells(
        max(0, center[0] - 1),
        min(h - 1, center[0] + 1),
        max(0, center[1] - 1),
        min(w - 1, center[1] + 1),
    )

    occupied: set[tuple[int, int]] = set()
    fragments: list[set[tuple[int, int]]] = []

    center_area = random.randint(2, min(4, len(center_window)))
    center_fragment = _grow_fragment(center_window, center, center_area, occupied)
    if center_fragment is None:
        return None
    fragments.append(center_fragment)
    occupied |= center_fragment

    for window, anchor in zip(corner_windows, corner_anchors):
        max_area = 4 if len(window) >= 4 and min(h, w) >= 7 else 3
        area = random.randint(2, min(max_area, len(window)))
        fragment = _grow_fragment(window, anchor, area, occupied)
        if fragment is None:
            return None
        fragments.append(fragment)
        occupied |= fragment

    coords = np.asarray(sorted(occupied), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    assert r0 == 0 and c0 == 0 and r1 == h - 1 and c1 == w - 1
    return {
        "coords": coords,
        "height": h,
        "width": w,
    }


def _place_rect(
    reserved: np.ndarray,
    rect_h: int,
    rect_w: int,
    *,
    c_start: int = 1,
    c_stop: int | None = None,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if c_stop is None:
        c_stop = w - rect_w - 1
    positions = []
    for r in range(1, h - rect_h):
        for c in range(max(1, c_start), c_stop + 1):
            if c + rect_w >= w:
                continue
            window = reserved[r - 1 : r + rect_h + 1, c - 1 : c + rect_w + 1]
            if not bool(window.any()):
                positions.append((r, c))
    if not positions:
        return None
    return random.choice(positions)


def _mark_reserved(reserved: np.ndarray, r: int, c: int, rect_h: int, rect_w: int) -> None:
    h, w = reserved.shape
    reserved[max(0, r - 1) : min(h, r + rect_h + 1), max(0, c - 1) : min(w, c + rect_w + 1)] = True


def _centered_absolute(
    layer: dict, inner_top: int, inner_left: int, inner_h: int, inner_w: int
) -> set[tuple[int, int]]:
    top = inner_top + (inner_h - int(layer["height"])) // 2
    left = inner_left + (inner_w - int(layer["width"])) // 2
    return {
        (top + int(r), left + int(c))
        for r, c in np.asarray(layer["coords"], dtype=int)
    }


def _compose_expected(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    layers: list[dict],
    order: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    fr0, fc0, fr1, fc1 = frame_box
    out[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    out[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    out[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    out[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    inner_top = fr0 + 1
    inner_left = fc0 + 1
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    for color, layer in zip(order, layers):
        top = inner_top + (inner_h - int(layer["height"])) // 2
        left = inner_left + (inner_w - int(layer["width"])) // 2
        for r, c in np.asarray(layer["coords"], dtype=int):
            out[top + int(r), left + int(c)] = int(color)
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    order: list[int],
    strip_positions: set[tuple[int, int]],
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    for color in order:
        coords = [(r, c) for r, c in np.argwhere(grid == int(color)) if (int(r), int(c)) not in strip_positions]
        if not coords:
            return False
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        comp_sizes = sorted(int(comp.shape[0]) for comp in _components(mask))
        if not comp_sizes or any(size < 2 or size > 8 for size in comp_sizes):
            return False
    visible = set(int(v) for v in np.unique(expected) if int(v) not in (BG, FRAME_COLOR))
    return len(visible) >= 2


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(2, 4)
    order = random.sample(LAYER_COLORS, layer_count)

    max_inner = 12 - max(0, layer_count - 2)
    inner_h = random.randint(7, max_inner)
    inner_w = random.randint(7, max_inner)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    grid_h = random.randint(max(22, frame_h + 6), 30)
    grid_w = random.randint(max(22, frame_w + 6), 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    strip_side = random.choice(["left", "right"])
    strip_col = 0 if strip_side == "left" else grid_w - 1
    strip_start = random.randint(1, grid_h - layer_count - 1)
    strip_positions = {(strip_start + i, strip_col) for i in range(layer_count)}
    for i, color in enumerate(order):
        grid[strip_start + i, strip_col] = int(color)

    if strip_side == "left":
        reserved[:, :2] = True
        frame_c_start = 3
        frame_c_stop = grid_w - frame_w - 2
    else:
        reserved[:, -2:] = True
        frame_c_start = 1
        frame_c_stop = grid_w - frame_w - 4
    if frame_c_stop < frame_c_start:
        return None

    frame_pos = _place_rect(reserved, frame_h, frame_w, c_start=frame_c_start, c_stop=frame_c_stop)
    if frame_pos is None:
        return None
    fr0, fc0 = frame_pos
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    _mark_reserved(reserved, fr0, fc0, frame_h, frame_w)

    layers: list[dict] = []
    for _ in range(layer_count):
        layer = _make_layer(inner_h, inner_w)
        if layer is None:
            return None
        layers.append(layer)

    inner_top = fr0 + 1
    inner_left = fc0 + 1
    abs_layers = [_centered_absolute(layer, inner_top, inner_left, inner_h, inner_w) for layer in layers]
    overlap = False
    for i in range(layer_count):
        for j in range(i + 1, layer_count):
            if abs_layers[i] & abs_layers[j]:
                overlap = True
                break
        if overlap:
            break
    if not overlap:
        return None

    for color, layer in zip(order, layers):
        pos = _place_rect(reserved, int(layer["height"]), int(layer["width"]))
        if pos is None:
            return None
        top, left = pos
        for r, c in np.asarray(layer["coords"], dtype=int):
            grid[top + int(r), left + int(c)] = int(color)
        _mark_reserved(reserved, top, left, int(layer["height"]), int(layer["width"]))

    expected = _compose_expected(grid.shape, (fr0, fc0, fr1, fc1), layers, order)
    if not _validate_generated(grid, expected, order, strip_positions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
