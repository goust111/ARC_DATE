from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

STACK_ORDER = (BLUE, GREEN, YELLOW, ORANGE, RED)
ORDER_INDEX = {color: idx for idx, color in enumerate(STACK_ORDER)}

GEN_RETRIES = 300
LAYER_COUNT_CHOICES = (2, 3)
INNER_HEIGHT_CHOICES = (5, 7, 9, 11)
INNER_WIDTH_CHOICES = (5, 7, 9, 11)
MAX_SHIFT = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    ranges: list[tuple[int, int]] = []
    start = 0
    c = 0
    while c < w:
        if np.all(grid[:, c] == GREY):
            if start < c:
                ranges.append((start, c))
            while c < w and np.all(grid[:, c] == GREY):
                c += 1
            start = c
        else:
            c += 1
    if start < w:
        ranges.append((start, w))
    assert ranges, "expected at least one panel"
    panel_widths = {c1 - c0 for c0, c1 in ranges}
    assert len(panel_widths) == 1, "all panels must have the same width"
    return [grid[:, c0:c1] for c0, c1 in ranges]


def _panel_border_color(panel: np.ndarray) -> int:
    border = np.concatenate([panel[0, :], panel[-1, :], panel[1:-1, 0], panel[1:-1, -1]])
    values = set(map(int, np.unique(border)))
    assert len(values) == 1, "panel border must be uniform"
    color = values.pop()
    assert color in ORDER_INDEX, "unexpected border color"
    return color


def _marker_is_embedded(shape: np.ndarray, anchor_r: int, anchor_c: int) -> bool:
    up = anchor_r - 1 >= 0 and bool(shape[anchor_r - 1, anchor_c])
    down = anchor_r + 1 < shape.shape[0] and bool(shape[anchor_r + 1, anchor_c])
    left = anchor_c - 1 >= 0 and bool(shape[anchor_r, anchor_c - 1])
    right = anchor_c + 1 < shape.shape[1] and bool(shape[anchor_r, anchor_c + 1])
    count = int(up) + int(down) + int(left) + int(right)
    return count >= 3 or (up and down) or (left and right)


def _layer_from_panel(panel: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int]]:
    color = _panel_border_color(panel)
    inner = panel[1:-1, 1:-1]
    markers = np.argwhere(inner == AZURE)
    assert markers.shape[0] == 1, "each panel must contain one azure marker"
    anchor_r, anchor_c = map(int, markers[0])
    shape = inner == MAROON
    if _marker_is_embedded(shape, anchor_r, anchor_c):
        shape = shape.copy()
        shape[anchor_r, anchor_c] = True
    return color, shape, (anchor_r, anchor_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each panel's marked maroon shape onto one shared canvas and merge the layers by border-color order.

    Input structure:
    - The grid is a single horizontal strip of equal-size panels separated by one-pixel grey columns.
    - Every panel has a one-pixel border whose color is one of blue, green, yellow, orange, or red.
    - Inside each border is a black field containing a maroon shape plus one azure marker pixel, with the marker either embedded in the body or touching it from the side.

    Rule:
    - Crop each panel to its interior; the output size equals that interior size.
    - Order the panels from bottom layer to top layer by border color: blue, then green, then yellow, then orange, then red.
    - Translate each panel's maroon shape so that its azure marker lands on the output center pixel; if the marker sits inside the body, that center cell belongs to the aligned maroon footprint even though the azure pixel itself is not preserved.
    - Paint the aligned layers in the fixed order onto a black canvas, with later maroon pixels covering earlier ones.

    Key insight:
    - The marker gives the translation anchor, while the border color gives the layer order; both cues are needed to reconstruct the aligned composite.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)
    layers = [_layer_from_panel(panel) for panel in panels]

    inner_shapes = {(panel.shape[0] - 2, panel.shape[1] - 2) for panel in panels}
    assert len(inner_shapes) == 1, "all panel interiors must have the same size"
    out_h, out_w = inner_shapes.pop()
    center_r, center_c = out_h // 2, out_w // 2

    out = np.full((out_h, out_w), BLACK, dtype=int)
    for color, shape, (anchor_r, anchor_c) in sorted(layers, key=lambda item: ORDER_INDEX[item[0]]):
        for r, c in np.argwhere(shape):
            rr = center_r + int(r) - anchor_r
            cc = center_c + int(c) - anchor_c
            if 0 <= rr < out_h and 0 <= cc < out_w:
                out[rr, cc] = MAROON
    return out


def _width_choices_for_layers(layer_count: int) -> list[int]:
    choices = []
    for inner_w in INNER_WIDTH_CHOICES:
        full_w = layer_count * (inner_w + 2) + (layer_count - 1)
        if full_w <= 30:
            choices.append(inner_w)
    return choices


def _connected(points: set[tuple[int, int]]) -> bool:
    if not points:
        return False
    start = next(iter(points))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in points and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(points)


def _grow_connected_shape(
    rng: random.Random,
    height: int,
    width: int,
    allowed: np.ndarray,
    seed: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    min_size: int,
    max_size: int,
) -> np.ndarray | None:
    if any(not allowed[r, c] for r, c in seed):
        return None
    points = set(seed)
    target = rng.randint(min_size, max_size)

    def frontier_cells() -> list[tuple[int, int]]:
        frontier: set[tuple[int, int]] = set()
        for r, c in points:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and allowed[nr, nc]:
                    cell = (nr, nc)
                    if cell not in points and cell not in forbidden:
                        frontier.add(cell)
        return list(frontier)

    frontier = frontier_cells()
    while frontier and len(points) < target:
        cell = rng.choice(frontier)
        points.add(cell)
        frontier = frontier_cells()
        if len(points) >= min_size and rng.random() < 0.18:
            break

    if len(points) < min_size or not _connected(points):
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _pick_delta(rng: random.Random, height: int, width: int) -> tuple[int, int]:
    center_r, center_c = height // 2, width // 2
    row_span = min(MAX_SHIFT, center_r - 1)
    col_span = min(MAX_SHIFT, center_c - 1)
    candidates = [(dr, dc) for dr in range(-row_span, row_span + 1) for dc in range(-col_span, col_span + 1)]
    nonzero = [delta for delta in candidates if delta != (0, 0)]
    if nonzero and rng.random() < 0.8:
        return rng.choice(nonzero)
    return rng.choice(candidates)


def _allowed_mask(height: int, width: int, dr: int, dc: int) -> np.ndarray:
    row_start = max(0, -dr)
    row_stop = min(height, height - dr)
    col_start = max(0, -dc)
    col_stop = min(width, width - dc)
    mask = np.zeros((height, width), dtype=bool)
    mask[row_start:row_stop, col_start:col_stop] = True
    return mask


def _make_aligned_shape(
    rng: random.Random,
    height: int,
    width: int,
    mode: str,
    allowed: np.ndarray,
) -> np.ndarray | None:
    center = (height // 2, width // 2)
    orth = [(center[0] - 1, center[1]), (center[0] + 1, center[1]), (center[0], center[1] - 1), (center[0], center[1] + 1)]

    if mode == "inside":
        seed = {center, *orth}
        min_size = len(seed)
        max_size = min(int(allowed.sum()), max(8, height * width // 3))
        return _grow_connected_shape(rng, height, width, allowed, seed, set(), min_size, max_size)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rng.shuffle(dirs)
    for dr, dc in dirs:
        first = (center[0] + dr, center[1] + dc)
        if not (0 <= first[0] < height and 0 <= first[1] < width and allowed[first]):
            continue
        forbidden = {center}
        for cell in orth:
            if cell != first:
                forbidden.add(cell)
        seed = {first}
        second = (first[0] + dr, first[1] + dc)
        if 0 <= second[0] < height and 0 <= second[1] < width and allowed[second]:
            seed.add(second)
        else:
            side_options = []
            for sdr, sdc in ((dc, dr), (-dc, -dr)):
                alt = (first[0] + sdr, first[1] + sdc)
                if 0 <= alt[0] < height and 0 <= alt[1] < width and allowed[alt] and alt != center:
                    side_options.append(alt)
            if side_options:
                seed.add(rng.choice(side_options))
        max_size = min(int(allowed.sum()), max(6, height * width // 3))
        mask = _grow_connected_shape(rng, height, width, allowed, seed, forbidden, max(3, len(seed)), max_size)
        if mask is None:
            continue
        if bool(mask[center]):
            continue
        if sum(bool(mask[r, c]) for r, c in orth) != 1:
            continue
        return mask
    return None


def _panel_from_aligned(
    aligned: np.ndarray,
    border_color: int,
    dr: int,
    dc: int,
    embedded: bool,
) -> np.ndarray:
    height, width = aligned.shape
    panel = np.full((height + 2, width + 2), border_color, dtype=int)
    inner = np.full((height, width), BLACK, dtype=int)
    coords = np.argwhere(aligned)
    for r, c in coords:
        inner[int(r) + dr, int(c) + dc] = MAROON
    anchor = (height // 2 + dr, width // 2 + dc)
    inner[anchor] = AZURE
    if not embedded:
        assert not aligned[height // 2, width // 2]
    panel[1:-1, 1:-1] = inner
    return panel


def _compose_input(panels: list[np.ndarray]) -> np.ndarray:
    height = panels[0].shape[0]
    width = sum(panel.shape[1] for panel in panels) + len(panels) - 1
    grid = np.full((height, width), GREY, dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        pw = panel.shape[1]
        grid[:, cursor : cursor + pw] = panel
        cursor += pw
        if idx != len(panels) - 1:
            grid[:, cursor] = GREY
            cursor += 1
    return grid


def _each_layer_contributes(masks: list[np.ndarray]) -> bool:
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    if int(union.sum()) <= max(int(mask.sum()) for mask in masks):
        return False
    for idx, mask in enumerate(masks):
        others = np.zeros_like(mask, dtype=bool)
        for j, other in enumerate(masks):
            if j != idx:
                others |= other
        if not np.any(mask & ~others):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.choice(LAYER_COUNT_CHOICES)
    inner_h = rng.choice(INNER_HEIGHT_CHOICES)
    width_choices = _width_choices_for_layers(layer_count)
    if not width_choices:
        return None
    inner_w = rng.choice(width_choices)

    selected_colors = rng.sample(STACK_ORDER, layer_count)
    placement_colors = selected_colors[:]
    rng.shuffle(placement_colors)

    center = (inner_h // 2, inner_w // 2)
    deltas: list[tuple[int, int]] = []
    modes: list[str] = []
    aligned_masks: list[np.ndarray] = []

    for _ in range(layer_count):
        mode = rng.choice(("inside", "adjacent"))
        dr, dc = _pick_delta(rng, inner_h, inner_w)
        deltas.append((dr, dc))
        modes.append(mode)

        allowed = _allowed_mask(inner_h, inner_w, dr, dc)
        aligned = _make_aligned_shape(rng, inner_h, inner_w, mode, allowed)
        if aligned is None:
            return None
        if mode == "inside":
            if not bool(aligned[center]):
                return None
        else:
            if bool(aligned[center]):
                return None
        aligned_masks.append(aligned)

    if all(delta == (0, 0) for delta in deltas):
        return None
    if not _each_layer_contributes(aligned_masks):
        return None
    for i in range(layer_count):
        for j in range(i + 1, layer_count):
            if np.array_equal(aligned_masks[i], aligned_masks[j]):
                return None

    panels_by_color: dict[int, np.ndarray] = {}
    for color, aligned, (dr, dc), mode in zip(selected_colors, aligned_masks, deltas, modes):
        panels_by_color[color] = _panel_from_aligned(aligned, color, dr, dc, embedded=(mode == "inside"))

    panels = [panels_by_color[color] for color in placement_colors]
    grid = _compose_input(panels)
    if max(grid.shape) > 30:
        return None

    expected = np.full((inner_h, inner_w), BLACK, dtype=int)
    for color in STACK_ORDER:
        if color not in panels_by_color:
            continue
        aligned = aligned_masks[selected_colors.index(color)]
        expected[aligned] = MAROON

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
