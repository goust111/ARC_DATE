from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_w, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
SPRITE_COLOR = Color.YELLOW
SEPARATOR_COLOR = Color.GREY
PANEL_H = 6
PANEL_COLORS = (Color.RED, Color.GREEN, Color.BLUE)
GEN_RETRIES = 300

SPRITE_SHAPES: dict[int, tuple[np.ndarray, ...]] = {
    1: (
        np.array([[1], [1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1], [1], [1], [1]], dtype=bool),
    ),
    2: (
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    ),
}

PANEL_SHAPES: tuple[np.ndarray, ...] = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
)


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r] == SEPARATOR_COLOR))]
    assert len(rows) == 3, "expected exactly three full-width separator rows"
    return rows


def _panel_boxes(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    sep0, sep1, sep2 = _separator_rows(grid)
    height = int(sep1 - sep0 - 1)
    assert height > 0, "panels must have positive height"
    assert int(sep2 - sep1 - 1) == height, "panel heights must match"
    assert int(grid.shape[0] - sep2 - 1) == height, "panel heights must match"
    w = int(grid.shape[1] - 1)
    boxes = [
        (sep0 + 1, 0, sep1 - 1, w),
        (sep1 + 1, 0, sep2 - 1, w),
        (sep2 + 1, 0, grid.shape[0] - 1, w),
    ]
    return boxes, sep0


def _sprite_info(grid: np.ndarray) -> tuple[int, int, int]:
    boxes, sep0 = _panel_boxes(grid)
    del boxes
    top = grid[:sep0, :]
    assert top.size > 0, "expected a sprite region above the first separator"
    bg = get_bg(grid)
    sprites = find_components(
        top,
        bg=bg,
        colors=[SPRITE_COLOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(sprites) == 1, "expected one yellow sprite"
    sprite = sprites[0]
    width = int(bbox_w(sprite["bbox"]))
    _r0, c0, _r1, c1 = map(int, sprite["bbox"])
    return width, c0, c1


def _anchor_index(width: int) -> int:
    if width == 1:
        return 2
    if width == 2:
        return 1
    return 0


def _merge_panels(panels: list[np.ndarray], anchor: int, bg: int) -> np.ndarray:
    out = panels[anchor].copy()
    for idx, panel in enumerate(panels):
        if idx == anchor:
            continue
        paste(out, panel, 0, 0, transparent=bg, overwrite=False)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose one stacked panel by the yellow sprite's width, then overlay the other panels onto it.

    Input structure:
    - The grid has a black background, one yellow 4-connected sprite in the top staging area, and three full-width gray separator rows.
    - Between the separator rows sit three equal-height panels with sparse foreground masks.
    - The sprite's bbox width is 1, 2, or at least 3, and the panels directly below its bbox columns are empty.

    Rule:
    - Measure the yellow sprite's bbox width.
    - Width 1 selects the bottom panel, width 2 selects the middle panel, and width 3 or more selects the top panel.
    - Crop the three panels, erase the sprite and separators, then use the selected panel as the destination canvas.
    - Overlay the other two cropped panels onto that destination by aligned local coordinates, treating black as transparent.

    Key insight:
    - The sprite is not copied into the answer; its width only chooses which of the three aligned panels becomes the merge target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    boxes, _sep0 = _panel_boxes(g)
    bg = get_bg(g)
    width, c0, c1 = _sprite_info(g)
    panels = [crop_bbox(g, box) for box in boxes]

    for panel in panels:
        assert not np.any(panel[:, c0 : c1 + 1] != bg), "sprite columns must be clear inside every panel"

    return _merge_panels(panels, _anchor_index(width), bg)


def _sample_sprite(width: int, top_h: int) -> np.ndarray:
    options = [shape for shape in SPRITE_SHAPES[width] if int(shape.shape[0]) <= top_h]
    assert options, "top staging area is too short for this branch"
    return random.choice(options).copy()


def _place_template(
    panel: np.ndarray,
    occupied: np.ndarray,
    color: int,
    blocked_cols: set[int],
    template: np.ndarray,
) -> bool:
    h, w = map(int, template.shape)
    valid: list[tuple[int, int]] = []
    for r0 in range(PANEL_H - h + 1):
        for c0 in range(panel.shape[1] - w + 1):
            coords = np.argwhere(template) + np.array([[r0, c0]])
            cols = {int(c) for _r, c in coords}
            if cols & blocked_cols:
                continue
            if bool(occupied[coords[:, 0], coords[:, 1]].any()):
                continue
            valid.append((r0, c0))
    if not valid:
        return False

    r0, c0 = random.choice(valid)
    coords = np.argwhere(template) + np.array([[r0, c0]])
    panel[coords[:, 0], coords[:, 1]] = int(color)
    occupied[coords[:, 0], coords[:, 1]] = True
    return True


def _render_input(
    top_h: int,
    panel_w: int,
    sprite_mask: np.ndarray,
    sprite_col: int,
    panels: list[np.ndarray],
) -> np.ndarray:
    height = top_h + 3 + 3 * PANEL_H
    g = np.full((height, panel_w), BG, dtype=int)

    sh, sw = map(int, sprite_mask.shape)
    g[:sh, sprite_col : sprite_col + sw][sprite_mask] = SPRITE_COLOR

    sep_rows = [top_h, top_h + 1 + PANEL_H, top_h + 2 + 2 * PANEL_H]
    for r in sep_rows:
        g[r, :] = SEPARATOR_COLOR

    starts = [sep_rows[0] + 1, sep_rows[1] + 1, sep_rows[2] + 1]
    for panel, start in zip(panels, starts):
        g[start : start + PANEL_H, :] = panel

    return g


def _oracle_output(panels: list[np.ndarray], width: int) -> np.ndarray:
    return _merge_panels(panels, _anchor_index(width), BG)


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice([1, 2, 3])
    top_h = random.choice([2, 3, 4])
    panel_w = random.randint(8, 12)

    sprite_mask = _sample_sprite(branch, top_h)
    sh, sw = map(int, sprite_mask.shape)
    if sw > panel_w:
        return None
    sprite_col = random.randint(0, panel_w - sw)
    blocked_cols = set(range(sprite_col, sprite_col + sw))

    occupied = np.zeros((PANEL_H, panel_w), dtype=bool)
    panels = [np.full((PANEL_H, panel_w), BG, dtype=int) for _ in range(3)]
    templates = [random.choice(PANEL_SHAPES) for _ in range(3)]
    random.shuffle(templates)
    for panel, color, first_template in zip(panels, PANEL_COLORS, templates):
        shape_count = random.choice([1, 1, 2])
        used_any = False
        for shape_idx in range(shape_count):
            template = first_template if shape_idx == 0 else random.choice(PANEL_SHAPES)
            if not _place_template(panel, occupied, color, blocked_cols, template):
                return None
            used_any = True
        if not used_any or int(np.count_nonzero(panel)) < 2:
            return None

    grid = _render_input(top_h, panel_w, sprite_mask, sprite_col, panels)
    expected = _oracle_output(panels, branch)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid width-gated panel overlay input")
