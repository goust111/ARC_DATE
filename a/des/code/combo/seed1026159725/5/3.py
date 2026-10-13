from __future__ import annotations

from collections import deque

import numpy as np


_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator controls.
_MAX_GENERATION_ATTEMPTS = 400
_PANEL_COUNT_RANGE = (3, 5)
_PANEL_HEIGHT_RANGE = (6, 11)
_PANEL_WIDTH_RANGE = (5, 9)
_GAP_WIDTH_RANGE = (1, 3)
_SEGMENT_LENGTH_MIN = 1
_ANCHOR_MODE_PROB = 0.5


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    reached = 0

    while queue:
        r, c = queue.popleft()
        reached += 1
        for dr, dc in _DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))

    return reached == int(mask.sum())


def _panel_object(panel: np.ndarray, bg: int) -> dict[str, object]:
    colors = [int(color) for color in np.unique(panel) if int(color) != bg]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = colors[0]
    mask = panel == color
    assert _is_connected(mask), "panel foreground must form one 4-connected object"
    coords = np.argwhere(mask)
    return {
        "color": color,
        "coords": coords,
        "area": int(coords.shape[0]),
    }


def _infer_layout(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    candidates: list[tuple[int, int, int, list[dict[str, object]]]] = []
    max_panels = min(6, width)

    for panel_count in range(2, max_panels + 1):
        for gap_width in range(1, width):
            remaining = width - (panel_count - 1) * gap_width
            if remaining <= 0 or remaining % panel_count != 0:
                continue
            panel_width = remaining // panel_count
            if panel_width < 2:
                continue

            panels: list[dict[str, object]] = []
            ok = True
            for idx in range(panel_count):
                c0 = idx * (panel_width + gap_width)
                c1 = c0 + panel_width
                panel = g[:, c0:c1]
                if panel.shape != (height, panel_width):
                    ok = False
                    break
                if idx < panel_count - 1 and not np.all(g[:, c1 : c1 + gap_width] == bg):
                    ok = False
                    break
                try:
                    obj = _panel_object(panel, bg)
                except AssertionError:
                    ok = False
                    break
                panels.append(obj)

            if not ok:
                continue

            colors = [int(panel["color"]) for panel in panels]
            if len(set(colors)) != len(colors):
                continue
            candidates.append((panel_count, panel_width, gap_width, panels))

    assert candidates, "failed to infer an equal-width panel layout"
    best = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    assert sum(item[:3] == best[:3] for item in candidates) == 1, "panel layout must be unique"
    return _get_bg(g), best[1], best[3]


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay one object per panel after sorting by area, with smaller objects on top."""
    bg, panel_width, panels = _infer_layout(np.asarray(grid, dtype=int))
    height = int(grid.shape[0])

    out = np.full((height, panel_width), bg, dtype=int)
    ordered = sorted(enumerate(panels), key=lambda item: (-int(item[1]["area"]), item[0]))
    for _idx, panel in ordered:
        coords = np.asarray(panel["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(panel["color"])
    return out


def _make_ribbon_mask(height: int, width: int, anchor_row: int | None) -> np.ndarray:
    max_len = max(_SEGMENT_LENGTH_MIN, min(height - 1, height // 2 + 1))
    lengths = np.empty(width, dtype=int)
    tops = np.empty(width, dtype=int)

    if anchor_row is None:
        first_len = int(_RNG.integers(_SEGMENT_LENGTH_MIN, max_len + 1))
        first_top = int(_RNG.integers(0, height - first_len + 1))
    else:
        first_len = int(_RNG.integers(_SEGMENT_LENGTH_MIN, max_len + 1))
        low = max(0, anchor_row - first_len + 1)
        high = min(anchor_row, height - first_len)
        if low > high:
            low = max(0, min(anchor_row, height - first_len))
            high = low
        first_top = int(_RNG.integers(low, high + 1))

    lengths[0] = first_len
    tops[0] = first_top

    for col in range(1, width):
        prev_top = int(tops[col - 1])
        prev_bottom = prev_top + int(lengths[col - 1]) - 1
        seg_len = int(_RNG.integers(_SEGMENT_LENGTH_MIN, max_len + 1))

        low = max(0, prev_top - seg_len + 1)
        high = min(height - seg_len, prev_bottom)
        if anchor_row is not None:
            low = max(low, anchor_row - seg_len + 1)
            high = min(high, anchor_row)
        if low > high:
            return np.zeros((height, width), dtype=bool)

        tops[col] = int(_RNG.integers(low, high + 1))
        lengths[col] = seg_len

    mask = np.zeros((height, width), dtype=bool)
    for col in range(width):
        top = int(tops[col])
        seg_len = int(lengths[col])
        mask[top : top + seg_len, col] = True

    assert _is_connected(mask)
    assert np.all(mask.sum(axis=0) >= 1), "mask must reveal the full panel width"
    return mask


def _sample_masks(height: int, width: int, count: int) -> list[np.ndarray] | None:
    anchor_row = None
    if _RNG.random() < _ANCHOR_MODE_PROB:
        anchor_row = int(_RNG.integers(1, height - 1))

    masks: list[np.ndarray] = []
    areas: set[int] = set()
    for _ in range(count):
        mask = _make_ribbon_mask(height, width, anchor_row)
        area = int(mask.sum())
        if area in areas:
            return None
        masks.append(mask)
        areas.add(area)
    return masks


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(_RNG.integers(_PANEL_COUNT_RANGE[0], _PANEL_COUNT_RANGE[1] + 1))
    panel_height = int(_RNG.integers(_PANEL_HEIGHT_RANGE[0], _PANEL_HEIGHT_RANGE[1] + 1))
    panel_width = int(_RNG.integers(_PANEL_WIDTH_RANGE[0], _PANEL_WIDTH_RANGE[1] + 1))
    gap_width = int(_RNG.integers(_GAP_WIDTH_RANGE[0], _GAP_WIDTH_RANGE[1] + 1))

    masks = _sample_masks(panel_height, panel_width, panel_count)
    if masks is None:
        return None

    colors = list(range(10))
    bg = int(_RNG.choice(colors))
    colors.remove(bg)
    object_colors = [int(color) for color in _RNG.choice(colors, size=panel_count, replace=False)]
    _RNG.shuffle(object_colors)

    width = panel_count * panel_width + (panel_count - 1) * gap_width
    grid = np.full((panel_height, width), bg, dtype=int)

    for idx, (mask, color) in enumerate(zip(masks, object_colors)):
        c0 = idx * (panel_width + gap_width)
        panel = grid[:, c0 : c0 + panel_width]
        panel[mask] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (panel_height, panel_width):
        return None

    # Enforce that the generated layout is genuinely size-driven, not panel-order-driven.
    areas_in_order = [int(mask.sum()) for mask in masks]
    if areas_in_order == sorted(areas_in_order, reverse=True):
        return None
    if areas_in_order == sorted(areas_in_order):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_GENERATION_ATTEMPTS} attempts")
