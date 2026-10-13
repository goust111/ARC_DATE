from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PANEL_SIZE_OPTIONS = (5, 6)
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
OVERLAY_ORDER = (0, 3, 2, 1)  # UL, LR, LL, UR
GENERATION_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, mask.astype(int).ravel()))


def _rotation_family(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    panel_h = h // 2
    panel_w = w // 2
    return [
        grid[:panel_h, :panel_w],
        grid[:panel_h, panel_w:],
        grid[panel_h:, :panel_w],
        grid[panel_h:, panel_w:],
    ]


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.full(mask.shape, BACKGROUND, dtype=int)
    panel[mask] = color
    return panel


def _panel_foreground_colors(panel: np.ndarray) -> set[int]:
    return set(map(int, np.unique(panel))) - {BACKGROUND}


def _missing_rotation_mask(source_masks: list[np.ndarray]) -> np.ndarray:
    seen_keys = {_mask_key(mask) for mask in source_masks}
    family_keys = {
        _mask_key(rotated)
        for mask in source_masks
        for rotated in _rotation_family(mask)
    }
    assert len(seen_keys) == 3, "expected three distinct shown rotations"
    assert len(family_keys) == 4, "expected a four-member rotation family"
    missing_keys = family_keys - seen_keys
    assert len(missing_keys) == 1, "expected exactly one missing rotation"
    side = source_masks[0].shape[0]
    return np.array(next(iter(missing_keys)), dtype=bool).reshape(side, side)


def _compose_panels(panels: list[np.ndarray]) -> np.ndarray:
    out = np.full(panels[0].shape, BACKGROUND, dtype=int)
    for index in OVERLAY_ORDER:
        panel = panels[index]
        mask = panel != BACKGROUND
        out[mask] = panel[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the missing rotation panel, then transparently composite the four panels.

    Input structure:
    - The input is a non-empty even square grid split into a 2x2 arrangement of equal square
      panels on a shared black background.
    - Three panels contain the same foreground color painted on black, and their masks are
      three distinct 90-degree rotations of one asymmetric base pattern.
    - The remaining panel is a solid non-black placeholder color that marks the missing member
      of the rotation family.

    Rule:
    - Identify the solid placeholder panel and treat the other three panels as source masks.
    - Enumerate the four 90-degree rotations of that mask family and recover the unique rotation
      that is not already present among the three shown panels.
    - Replace the placeholder with that missing rotated mask rendered in the shared foreground
      color on black.
    - Transparently composite the completed panels into one single-panel output by painting the
      upper-left panel first, then lower-right, then lower-left, and finally upper-right, with
      only non-black cells overwriting earlier layers.

    Key insight:
    - The solid panel is not part of the picture content; it stands for the absent fourth
      rotation, and the final answer is the completed 2x2 panel set collapsed through a fixed
      positional overlay order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even square grid"

    panels = _split_panels(g)
    panel_shape = panels[0].shape
    assert panel_shape[0] == panel_shape[1] and panel_shape[0] > 0, "expected square panels"
    assert all(panel.shape == panel_shape for panel in panels), "expected four equal panels"

    placeholder_indices = []
    for index, panel in enumerate(panels):
        values = np.unique(panel)
        if len(values) == 1 and int(values[0]) != BACKGROUND:
            placeholder_indices.append(index)
    assert len(placeholder_indices) == 1, "expected exactly one solid placeholder panel"
    placeholder_index = placeholder_indices[0]

    source_panels = [panel for i, panel in enumerate(panels) if i != placeholder_index]
    source_colors = set()
    for panel in source_panels:
        colors = _panel_foreground_colors(panel)
        assert len(colors) == 1, "each source panel must use exactly one foreground color"
        source_colors |= colors
    assert len(source_colors) == 1, "expected one shared source color across the three panels"
    source_color = next(iter(source_colors))

    source_masks = []
    for panel in source_panels:
        assert set(map(int, np.unique(panel))) <= {BACKGROUND, source_color}
        source_masks.append(panel == source_color)

    missing_mask = _missing_rotation_mask(source_masks)
    completed_panels = [panel.copy() for panel in panels]
    completed_panels[placeholder_index] = _render_mask(missing_mask, source_color)
    return _compose_panels(completed_panels)


def _connected_shape(side: int, cells: int) -> np.ndarray | None:
    mask = np.zeros((side, side), dtype=bool)
    start = (random.randrange(side), random.randrange(side))
    mask[start] = True

    while int(mask.sum()) < cells:
        frontier = set()
        for r, c in zip(*np.where(mask)):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    return mask


def _sample_base_mask(side: int) -> np.ndarray | None:
    min_cells = max(4, side - 1)
    max_cells = min(side + 2, side * side - 3)
    target_cells = random.randint(min_cells, max_cells)
    mask = _connected_shape(side, target_cells)
    if mask is None:
        return None

    family = _rotation_family(mask)
    if len({_mask_key(rotated) for rotated in family}) != 4:
        return None

    union = np.zeros_like(mask, dtype=bool)
    for rotated in family:
        union |= rotated
    union_count = int(union.sum())
    if not (side + 4 <= union_count <= side * side - 2):
        return None

    return mask


def _build_grid(
    family: list[np.ndarray],
    source_color: int,
    placeholder_color: int,
    placeholder_index: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    side = family[0].shape[0]
    panels: list[np.ndarray] = []
    for index, mask in enumerate(family):
        if index == placeholder_index:
            panels.append(np.full((side, side), placeholder_color, dtype=int))
        else:
            panels.append(_render_mask(mask, source_color))

    grid = np.zeros((2 * side, 2 * side), dtype=int)
    grid[:side, :side] = panels[0]
    grid[:side, side:] = panels[1]
    grid[side:, :side] = panels[2]
    grid[side:, side:] = panels[3]
    return grid, panels


def _expected_output(family: list[np.ndarray], source_color: int) -> np.ndarray:
    completed = [_render_mask(mask, source_color) for mask in family]
    return _compose_panels(completed)


def _generate_candidate() -> np.ndarray | None:
    side = random.choice(PANEL_SIZE_OPTIONS)
    base_mask = _sample_base_mask(side)
    if base_mask is None:
        return None

    source_color, placeholder_color = random.sample(PALETTE, 2)
    family = _rotation_family(base_mask)
    assignment = random.sample(range(4), 4)
    positioned_family = [family[i] for i in assignment]
    placeholder_index = random.randrange(4)

    grid, _ = _build_grid(positioned_family, source_color, placeholder_color, placeholder_index)
    expected = _expected_output(positioned_family, source_color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid rotation-overlay input")
