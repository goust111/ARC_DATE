from __future__ import annotations

import itertools
import random

import numpy as np


GEN_ATTEMPTS = 300
GRID_LIMIT = 30
KEY_SIDES = ("left", "right")
K_CHOICES = (3, 4)
SIZE_OPTIONS = {
    3: (4, 5, 6, 7, 8),
    4: (5,),
}
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_pair(panel: np.ndarray, r: int, c: int, color: int) -> None:
    mirror_c = panel.shape[1] - 1 - c
    panel[r, c] = int(color)
    panel[r, mirror_c] = int(color)


def _off_axis_positions(size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(size) for c in range(size // 2)]


def _find_panel_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, tuple[int, int, int, int], int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue

        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        panel_size = height - 2
        if panel_size < 3:
            continue
        if (width - 1) % (panel_size + 1) != 0:
            continue

        panel_count = (width - 1) // (panel_size + 1)
        if panel_count < 2:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((height, width), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        for j in range(panel_count + 1):
            expected[:, j * (panel_size + 1)] = True

        if np.array_equal(sub == color, expected):
            candidates.append((color, (r0, c0, r1, c1), panel_size, panel_count))

    assert len(candidates) == 1, "expected exactly one framed row of square panels"
    return candidates[0]


def _extract_key_colors(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    panel_count: int,
) -> list[int]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere((g != bg) & outside)

    assert len(coords) == panel_count, "expected one detached key pixel per panel"
    kr0, kc0, kr1, kc1 = _bbox_coords(coords)
    assert kc0 == kc1, "key strip must be a single column"
    assert kr1 - kr0 + 1 == panel_count, "key strip height must equal panel count"
    assert len({tuple(map(int, p)) for p in coords}) == panel_count

    key_colors = [int(g[r, kc0]) for r in range(kr0, kr1 + 1)]
    assert all(color != bg for color in key_colors)
    assert len(set(key_colors)) == panel_count, "key colors must be distinct"
    return key_colors


def _extract_panels(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    panel_size: int,
    panel_count: int,
) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, _r1, _c1 = frame_bbox
    panels: list[np.ndarray] = []
    for j in range(panel_count):
        pc0 = c0 + 1 + j * (panel_size + 1)
        panel = g[r0 + 1 : r0 + 1 + panel_size, pc0 : pc0 + panel_size].copy()
        assert panel.shape == (panel_size, panel_size)
        panels.append(panel)
    return panels


def _repair_panel(panel: np.ndarray, key_colors: set[int]) -> tuple[np.ndarray, int]:
    p = np.asarray(panel, dtype=int)
    size = p.shape[0]
    assert p.shape == (size, size), "panel must be square"

    mismatches: list[tuple[int, int, int, int, int, bool]] = []
    for r in range(size):
        for c in range(size // 2):
            mc = size - 1 - c
            left = int(p[r, c])
            right = int(p[r, mc])
            if left == right:
                continue
            left_is_key = left in key_colors
            right_is_key = right in key_colors
            assert left_is_key ^ right_is_key, "mismatch must involve exactly one impurity color"
            mismatches.append((r, c, mc, left, right, left_is_key))

    assert len(mismatches) == 1, "expected exactly one symmetry-breaking impurity"
    r, c, mc, left, right, left_is_key = mismatches[0]
    repaired = p.copy()
    if left_is_key:
        repaired[r, c] = right
        impurity_color = left
    else:
        repaired[r, mc] = left
        impurity_color = right

    assert np.array_equal(repaired, np.fliplr(repaired)), "repaired panel must regain vertical symmetry"
    return repaired, int(impurity_color)


def _overlay_panels(panels: list[np.ndarray], order: tuple[int, ...], bg: int) -> np.ndarray:
    size = panels[0].shape[0]
    out = np.full((size, size), int(bg), dtype=int)
    for idx in order:
        panel = panels[int(idx)]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the tagged symmetric panels, then overlay them by the detached key-strip order.

    Input structure:
    - One side of the grid contains a single row of equal square panels inside one
      monochrome frame with vertical separators between panels.
    - Elsewhere there is a detached vertical key strip with exactly one colored pixel
      per panel, read from top to bottom.
    - Every panel is vertically mirror-symmetric except for one impurity pixel whose
      color appears in the key strip and nowhere in the normal panel patterns.

    Rule:
    - For each panel, find the unique left-right mismatch and identify the impurity as
      the mismatching side whose color belongs to the key strip.
    - Repair that impurity by copying the mirrored partner color back into its place.
    - Match repaired panels to key-strip colors, then overlay them on one panel-sized
      canvas: the first key color is the bottom layer, the last key color is the top
      layer, and background pixels stay transparent.

    Key insight:
    - The asymmetric pixel is only a removable identity tag: symmetry reveals how to
      restore the real panel, and the detached key strip uses that tag color to define
      the transparent stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _border_color, frame_bbox, panel_size, panel_count = _find_panel_frame(g, bg)
    key_colors = _extract_key_colors(g, bg, frame_bbox, panel_count)
    key_set = set(key_colors)

    panels_by_key: dict[int, np.ndarray] = {}
    for panel in _extract_panels(g, frame_bbox, panel_size, panel_count):
        repaired, impurity_color = _repair_panel(panel, key_set)
        assert impurity_color not in panels_by_key, "each impurity color must label exactly one panel"
        panels_by_key[impurity_color] = repaired

    assert set(panels_by_key) == key_set, "every key color must map to one repaired panel"

    out = np.full((panel_size, panel_size), int(bg), dtype=int)
    for key_color in key_colors:
        panel = panels_by_key[int(key_color)]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def _choose_orders(panel_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = tuple(range(panel_count))
    orders = list(itertools.permutations(ids))
    spatial_order = random.choice(orders)
    valid_layer_orders = [
        order
        for order in orders
        if order != spatial_order and order != spatial_order[::-1]
    ]
    assert valid_layer_orders
    layer_order = random.choice(valid_layer_orders)
    return spatial_order, layer_order


def _build_repaired_panels(
    panel_count: int,
    panel_size: int,
    bg: int,
    panel_colors: list[int],
) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    pool = _off_axis_positions(panel_size)
    needed = panel_count + panel_count * (panel_count - 1) // 2
    assert len(pool) >= needed, "panel is too small for the required visible relations"

    picked = random.sample(pool, needed)
    private_positions = picked[:panel_count]
    pair_keys = list(itertools.combinations(range(panel_count), 2))
    pair_positions = {pair: pos for pair, pos in zip(pair_keys, picked[panel_count:])}

    panels = [np.full((panel_size, panel_size), int(bg), dtype=int) for _ in range(panel_count)]

    for idx in range(panel_count):
        r, c = private_positions[idx]
        _paint_pair(panels[idx], r, c, panel_colors[idx])

    for (a, b), (r, c) in pair_positions.items():
        _paint_pair(panels[a], r, c, panel_colors[a])
        _paint_pair(panels[b], r, c, panel_colors[b])

    return panels, private_positions, pair_positions


def _damage_panels(
    repaired_panels: list[np.ndarray],
    private_positions: list[tuple[int, int]],
    impurity_colors: list[int],
) -> list[np.ndarray]:
    damaged = [panel.copy() for panel in repaired_panels]
    for idx, (r, c) in enumerate(private_positions):
        mirror_c = damaged[idx].shape[1] - 1 - c
        impurity = int(impurity_colors[idx])
        if random.random() < 0.5:
            damaged[idx][r, c] = impurity
        else:
            damaged[idx][r, mirror_c] = impurity
    return damaged


def _draw_frame(
    grid: np.ndarray,
    frame_r0: int,
    frame_c0: int,
    panel_size: int,
    panel_count: int,
    border_color: int,
) -> None:
    frame_h = panel_size + 2
    frame_w = panel_count * panel_size + panel_count + 1
    grid[frame_r0, frame_c0 : frame_c0 + frame_w] = border_color
    grid[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = border_color
    for j in range(panel_count + 1):
        col = frame_c0 + j * (panel_size + 1)
        grid[frame_r0 : frame_r0 + frame_h, col] = border_color


def _compose_input(
    damaged_panels: list[np.ndarray],
    spatial_order: tuple[int, ...],
    layer_order: tuple[int, ...],
    panel_size: int,
    bg: int,
    border_color: int,
    impurity_colors: list[int],
) -> np.ndarray:
    panel_count = len(damaged_panels)
    frame_h = panel_size + 2
    frame_w = panel_count * panel_size + panel_count + 1
    gap_choices = [gap for gap in (1, 2) if frame_w + gap + 1 <= GRID_LIMIT]
    gap = random.choice(gap_choices)

    base_h = frame_h
    base_w = frame_w + gap + 1

    spare_h = GRID_LIMIT - base_h
    spare_w = GRID_LIMIT - base_w
    top = random.randint(0, min(3, spare_h))
    bottom = random.randint(0, min(3, spare_h - top))
    left = random.randint(0, min(3, spare_w))
    right = random.randint(0, min(3, spare_w - left))

    height = base_h + top + bottom
    width = base_w + left + right
    grid = np.full((height, width), int(bg), dtype=int)

    key_side = random.choice(KEY_SIDES)
    frame_r0 = top
    if key_side == "left":
        key_c = left
        frame_c0 = left + 1 + gap
    else:
        frame_c0 = left
        key_c = left + frame_w + gap

    _draw_frame(grid, frame_r0, frame_c0, panel_size, panel_count, border_color)

    for slot, panel_idx in enumerate(spatial_order):
        pc0 = frame_c0 + 1 + slot * (panel_size + 1)
        grid[frame_r0 + 1 : frame_r0 + 1 + panel_size, pc0 : pc0 + panel_size] = damaged_panels[panel_idx]

    key_r0 = frame_r0 + random.randint(0, frame_h - panel_count)
    for offset, panel_idx in enumerate(layer_order):
        grid[key_r0 + offset, key_c] = int(impurity_colors[panel_idx])

    return grid


def _panel_has_one_impurity(panel: np.ndarray, impurity_color: int) -> bool:
    return int(np.count_nonzero(panel == impurity_color)) == 1


def _visible_for_panel(panels: list[np.ndarray], order: tuple[int, ...], bg: int, panel_idx: int) -> bool:
    size = panels[0].shape[0]
    depth = order.index(panel_idx)
    higher = order[depth + 1 :]
    for r in range(size):
        for c in range(size):
            if int(panels[panel_idx][r, c]) == bg:
                continue
            if all(int(panels[h][r, c]) == bg for h in higher):
                return True
    return False


def _semantic_checks(
    grid: np.ndarray,
    repaired_panels: list[np.ndarray],
    damaged_panels: list[np.ndarray],
    spatial_order: tuple[int, ...],
    layer_order: tuple[int, ...],
    bg: int,
    private_positions: list[tuple[int, int]],
    panel_colors: list[int],
    impurity_colors: list[int],
) -> None:
    panel_count = len(repaired_panels)
    expected = _overlay_panels(repaired_panels, layer_order, bg)

    assert all(np.array_equal(panel, np.fliplr(panel)) for panel in repaired_panels)
    assert len({int(np.count_nonzero(panel != bg)) for panel in repaired_panels}) == 1

    for idx in range(panel_count):
        assert _panel_has_one_impurity(damaged_panels[idx], impurity_colors[idx])
        assert not np.any(repaired_panels[idx] == impurity_colors[idx])
        r, c = private_positions[idx]
        mirror_c = repaired_panels[idx].shape[1] - 1 - c
        assert int(repaired_panels[idx][r, c]) == int(panel_colors[idx])
        assert int(repaired_panels[idx][r, mirror_c]) == int(panel_colors[idx])

    orders = list(itertools.permutations(range(panel_count)))
    outputs = {
        order: _overlay_panels(repaired_panels, order, bg)
        for order in orders
    }
    for order, out in outputs.items():
        if order == layer_order:
            assert np.array_equal(out, expected)
        else:
            assert not np.array_equal(out, expected), "every wrong layer order must change the output"

    assert not np.array_equal(_overlay_panels(repaired_panels, spatial_order, bg), expected)
    assert not np.array_equal(_overlay_panels(damaged_panels, layer_order, bg), expected)
    assert all(_visible_for_panel(repaired_panels, layer_order, bg, idx) for idx in range(panel_count))
    assert not any(np.array_equal(expected, panel) for panel in repaired_panels)
    assert np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray:
    panel_count = random.choice(K_CHOICES)
    panel_size = random.choice(SIZE_OPTIONS[panel_count])
    palette = random.sample(range(10), 2 + 2 * panel_count)

    bg = int(palette[0])
    border_color = int(palette[1])
    panel_colors = [int(c) for c in palette[2 : 2 + panel_count]]
    impurity_colors = [int(c) for c in palette[2 + panel_count :]]

    repaired_panels, private_positions, _pair_positions = _build_repaired_panels(
        panel_count,
        panel_size,
        bg,
        panel_colors,
    )
    damaged_panels = _damage_panels(repaired_panels, private_positions, impurity_colors)
    spatial_order, layer_order = _choose_orders(panel_count)

    grid = _compose_input(
        damaged_panels,
        spatial_order,
        layer_order,
        panel_size,
        bg,
        border_color,
        impurity_colors,
    )
    _semantic_checks(
        grid,
        repaired_panels,
        damaged_panels,
        spatial_order,
        layer_order,
        bg,
        private_positions,
        panel_colors,
        impurity_colors,
    )
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_attempt()
        except AssertionError:
            continue
        if isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= GRID_LIMIT:
            out = main(grid)
            signature = out.tobytes() + str(out.shape).encode()
            if signature in _SEEN_OUTPUT_SIGNATURES:
                continue
            _SEEN_OUTPUT_SIGNATURES.add(signature)
            return grid
    raise RuntimeError("failed to generate a valid label-repair overlay input")
