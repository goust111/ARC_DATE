from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PANEL_COLORS = (1, 2, 3, 4)
SEPARATOR = 5
MARKER = 9

PANEL_SIZE = 5
SPRITE_SIZE = 3
INPUT_SHAPE = (5, 23)
SEPARATOR_COLS = (5, 11, 17)
PANEL_STARTS = (0, 6, 12, 18)
CORNER_NAMES = ("NW", "NE", "SW", "SE")

ROTATION_TO_NW = {
    "NW": 0,
    "NE": 1,
    "SE": 2,
    "SW": 3,
}

CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

GEN_MAX_ATTEMPTS = 400
GEN_AREA_RANGE = (3, 5)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _build_mask_bank() -> list[np.ndarray]:
    bank: list[np.ndarray] = []
    for bit in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array(
            [(bit >> i) & 1 for i in range(SPRITE_SIZE * SPRITE_SIZE)],
            dtype=bool,
        ).reshape((SPRITE_SIZE, SPRITE_SIZE))
        area = int(mask.sum())
        if not (GEN_AREA_RANGE[0] <= area <= GEN_AREA_RANGE[1]):
            continue
        if not _is_connected(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if np.all(mask.sum(axis=0) > 0) or np.all(mask.sum(axis=1) > 0):
            bank.append(mask)
    assert bank, "mask bank must not be empty"
    return bank


MASK_BANK = _build_mask_bank()


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.shape[0] == PANEL_SIZE, "expected five rows"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert sep_cols == list(SEPARATOR_COLS), "expected three full-height separator columns"

    panels = []
    start = 0
    for sep in sep_cols:
        panel = g[:, start:sep]
        assert panel.shape == (PANEL_SIZE, PANEL_SIZE), "each panel must be 5x5"
        panels.append(panel)
        start = sep + 1
    last = g[:, start:]
    assert last.shape == (PANEL_SIZE, PANEL_SIZE), "each panel must be 5x5"
    panels.append(last)
    assert len(panels) == 4
    return panels


def _outside_corners(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    corners = {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SW": (r1, c0),
        "SE": (r1, c1),
    }
    return {
        name: (corners[name][0] + CORNER_OFFSETS[name][0], corners[name][1] + CORNER_OFFSETS[name][1])
        for name in CORNER_NAMES
    }


def _parse_panel(panel: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    marker_coords = np.argwhere(panel == MARKER)
    assert len(marker_coords) == 1, "each panel must contain exactly one marker"
    marker = tuple(map(int, marker_coords[0]))

    bbox = (1, 1, 3, 3)
    r0, c0, r1, c1 = bbox
    patch = panel[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, np.unique(patch))) - {BACKGROUND}
    assert len(colors) == 1, "each sprite must use exactly one foreground color"
    outer = panel.copy()
    outer[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    assert not np.any((outer != BACKGROUND) & (outer != MARKER)), "non-marker foreground must stay inside the centered 3x3"

    outside = _outside_corners(bbox)
    source = [name for name, pos in outside.items() if pos == marker]
    assert len(source) == 1, "marker must identify one sprite corner"
    return patch.copy(), bbox, source[0]


def _rotated_panel(panel: np.ndarray) -> np.ndarray:
    patch, bbox, source = _parse_panel(panel)
    r0, c0, r1, c1 = bbox
    out = np.zeros_like(panel)
    out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(patch, k=ROTATION_TO_NW[source])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel's centered 3x3 sprite so the marker-named corner moves to the upper-left, then overlay the four rotated sprites.

    Input structure:
    - The grid has five rows and is split by three full-height grey separator columns into four 5x5 panels.
    - Each panel contains a centered 3x3 monochrome sprite on black background plus one maroon marker pixel diagonally outside one sprite corner.
    - The marker names the sprite corner that should become the sprite's upper-left corner after rotation.

    Rule:
    - For each panel, locate the 3x3 sprite bbox and the single marker pixel.
    - Rotate the sprite by the unique quarter-turn that maps the marker-indicated source corner to the sprite's upper-left corner.
    - Remove the marker pixels and separator columns.
    - Overlay the four rotated 5x5 panel sprites into one 5x5 output using transparency, with leftmost panel in front and rightmost panel in back.

    Key insight:
    - The marker is not an output object; it encodes a rotation choice, and only the rotated sprite layouts determine which colors remain visible after the transparent overlay.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    rotated = [_rotated_panel(panel) for panel in panels]

    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for layer in rotated:
        mask = (layer != BACKGROUND) & (out == BACKGROUND)
        out[mask] = layer[mask]
    return out


def _place_sprite(mask: np.ndarray, color: int, source_corner: str) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    panel[1:4, 1:4][mask] = color
    marker_r, marker_c = {
        "NW": (0, 0),
        "NE": (0, 4),
        "SW": (4, 0),
        "SE": (4, 4),
    }[source_corner]
    panel[marker_r, marker_c] = MARKER
    return panel


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, SEPARATOR_COLS] = SEPARATOR
    for start, panel in zip(PANEL_STARTS, panels):
        grid[:, start : start + PANEL_SIZE] = panel
    return grid


def _overlay_with_owners(rotated_panels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    owner = np.full((PANEL_SIZE, PANEL_SIZE), -1, dtype=int)
    for index, panel in enumerate(rotated_panels):
        mask = (panel != BACKGROUND) & (out == BACKGROUND)
        out[mask] = panel[mask]
        owner[mask] = index
    return out, owner


def _valid_instance(masks: list[np.ndarray], corners: list[str]) -> bool:
    rotated_panels = []
    colors = random.sample(list(PANEL_COLORS), k=4)
    for mask, color, corner in zip(masks, colors, corners):
        panel = _place_sprite(mask, color, corner)
        rotated_panels.append(_rotated_panel(panel))

    out, owner = _overlay_with_owners(rotated_panels)
    occupancy_sum = sum(int((panel != 0).sum()) for panel in rotated_panels)
    union_area = int((out != 0).sum())
    overlap_area = occupancy_sum - union_area
    visible_layers = len(set(map(int, owner[owner >= 0])))
    visible_colors = len(set(map(int, out[out != 0])))

    if overlap_area < 1:
        return False
    if visible_layers < 2:
        return False
    if visible_colors < 2:
        return False
    if union_area >= 9:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        masks = random.sample(MASK_BANK, k=4)
        corners = random.sample(list(CORNER_NAMES), k=4)
        if not _valid_instance(masks, corners):
            continue

        colors = random.sample(list(PANEL_COLORS), k=4)
        panels = [
            _place_sprite(mask, color, corner)
            for mask, color, corner in zip(masks, colors, corners)
        ]
        grid = _assemble_input(panels)
        assert np.array_equal(main(grid), main(grid.copy()))
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
