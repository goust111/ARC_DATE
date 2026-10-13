from __future__ import annotations

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox  # noqa: E402


BG = Color.BLACK
SEPARATOR = Color.GREY
TILE_FG = Color.BLUE
CELL_FILL = Color.YELLOW
CELL_MARK = Color.ORANGE
PALETTE_CHOICES = (Color.RED, Color.GREEN, Color.PINK, Color.TEAL, Color.MAROON)
CELL_SIZES = (2, 3)
MAX_ATTEMPTS = 400

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()

_CORNER_TO_ROT_CW = {
    "nw": 0,
    "ne": 1,
    "se": 2,
    "sw": 3,
}
_CORNERS = tuple(_CORNER_TO_ROT_CW)


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    box = bbox(grid != BG)
    assert box is not None, "expected a non-empty panel"
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, k=(-int(quarter_turns)) % 4)


def _full_color_columns(grid: np.ndarray, color: int) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]


def _decode_cell(cell: np.ndarray) -> tuple[bool, str | None]:
    nonzero = cell[cell != BG]
    if nonzero.size == 0:
        return False, None

    counts = Counter(map(int, nonzero))
    assert len(counts) == 2, "foreground layout cells must have one fill color and one marker color"
    items = sorted(counts.items(), key=lambda item: item[1])
    mark_color, mark_count = items[0]
    fill_color, fill_count = items[1]
    assert mark_count == 1, "each foreground layout cell must have exactly one indicator pixel"
    assert fill_count == int(cell.size) - 1, "each foreground layout cell must otherwise be solid"
    assert fill_color != mark_color, "indicator color must differ from fill color"

    rr, cc = np.argwhere(cell == int(mark_color))[0]
    rr = int(rr)
    cc = int(cc)
    h, w = map(int, cell.shape)
    assert h == w and h >= 2, "layout cells must be square blocks"
    corners = {
        (0, 0): "nw",
        (0, w - 1): "ne",
        (h - 1, w - 1): "se",
        (h - 1, 0): "sw",
    }
    assert (rr, cc) in corners, "indicator pixel must sit at a cell corner"
    return True, corners[(rr, cc)]


def _decode_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = _full_color_columns(g, SEPARATOR)
    assert len(sep_cols) == 2, f"expected exactly two separator columns, found {sep_cols}"
    s0, s1 = sep_cols
    assert 0 < s0 < s1 < g.shape[1] - 1, "separator columns must split the grid into three panels"

    left = _crop_nonzero(g[:, :s0])
    middle = _crop_nonzero(g[:, s0 + 1 : s1])
    right = _crop_nonzero(g[:, s1 + 1 :])

    tile_mask = left != BG
    assert tile_mask.shape == (3, 3), f"tile panel must crop to 3x3, got {tile_mask.shape}"
    assert np.all(left[tile_mask] == TILE_FG), "tile foreground must use the tile color"
    assert len({np.rot90(tile_mask, k).tobytes() for k in range(4)}) == 4, "tile must not be rotation-symmetric"

    mh, mw = map(int, middle.shape)
    assert mh == mw and mh % 3 == 0, f"layout panel must crop to a 3x3 cell grid, got {middle.shape}"
    cell_size = mh // 3
    assert cell_size >= 2, "layout cells must be at least 2x2"

    layout_mask = np.zeros((3, 3), dtype=bool)
    corners = np.full((3, 3), "", dtype=object)
    for lr in range(3):
        for lc in range(3):
            cell = middle[
                lr * cell_size : (lr + 1) * cell_size,
                lc * cell_size : (lc + 1) * cell_size,
            ]
            present, corner = _decode_cell(cell)
            layout_mask[lr, lc] = present
            corners[lr, lc] = "" if corner is None else corner

    assert int(layout_mask.sum()) >= 3, "layout must contain at least three foreground cells"
    assert int(layout_mask.sum()) < 9, "layout must contain at least one background cell"
    assert np.all(layout_mask.any(axis=0)), "layout must use every cell column so the panel crops unambiguously"
    assert np.all(layout_mask.any(axis=1)), "layout must use every cell row so the panel crops unambiguously"
    used_corners = {str(corners[r, c]) for r, c in zip(*np.where(layout_mask))}
    assert len(used_corners) >= 2, "layout must use at least two different indicator corners"

    assert right.shape == (2, 1), f"palette panel must crop to 2x1, got {right.shape}"
    palette = (int(right[0, 0]), int(right[1, 0]))
    assert palette[0] != palette[1], "palette colors must differ"
    assert SEPARATOR not in palette, "palette colors must differ from the separator color"

    return tile_mask, layout_mask, corners, palette


def _render_output(
    tile_mask: np.ndarray,
    layout_mask: np.ndarray,
    corners: np.ndarray,
    palette: tuple[int, int],
) -> np.ndarray:
    fg_color, bg_color = map(int, palette)
    out = np.full((9, 9), bg_color, dtype=int)
    for lr in range(3):
        for lc in range(3):
            r0 = lr * 3
            c0 = lc * 3
            if not bool(layout_mask[lr, lc]):
                continue
            rot = _CORNER_TO_ROT_CW[str(corners[lr, lc])]
            rotated = _rotate_cw(tile_mask, rot)
            block = np.where(rotated, fg_color, bg_color)
            out[r0 : r0 + 3, c0 : c0 + 3] = block
    return out


def _uniform_rotation_output(
    tile_mask: np.ndarray,
    layout_mask: np.ndarray,
    palette: tuple[int, int],
    rot: int,
) -> np.ndarray:
    fg_color, bg_color = map(int, palette)
    block = np.where(_rotate_cw(tile_mask, rot), fg_color, bg_color)
    out = np.full((9, 9), bg_color, dtype=int)
    for lr in range(3):
        for lc in range(3):
            if layout_mask[lr, lc]:
                out[lr * 3 : lr * 3 + 3, lc * 3 : lc * 3 + 3] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a 3x3 stamp across a 3x3 layout, rotating each placed copy from its local corner marker.

    Input structure:
    - Two full-height grey separator columns split the grid into a left stamp panel, a middle layout panel, and a right vertical palette panel.
    - The left panel contains a 3x3 binary stamp on black background; its blue foreground shape is chosen so all four rotations differ.
    - The middle panel is a 3x3 array of square cells; background cells are black, while each foreground cell is solid yellow except for one contrasting corner marker pixel.
    - The right panel contains two stacked color cells: the top color is the output foreground and the bottom color is the output background.

    Rule:
    - For each foreground layout cell, read which corner contains the singleton marker: NW means no rotation, NE means 90 degrees clockwise, SE means 180 degrees, and SW means 270 degrees clockwise.
    - Build a 9x9 output made of 3x3 blocks. Each foreground layout cell contributes one rotated copy of the left stamp recolored with the palette panel; each background layout cell becomes a solid background-color block.
    - The output block at layout row r and column c is placed at rows 3r..3r+2 and columns 3c..3c+2.

    Key insight:
    - The middle panel is not just a placement mask: each occupied cell independently encodes the rotation of the same left-panel stamp, while the right palette panel supplies the final two output colors.
    """
    tile_mask, layout_mask, corners, palette = _decode_input(np.asarray(grid, dtype=int))
    return _render_output(tile_mask, layout_mask, corners, palette)


def _random_tile_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((3, 3), dtype=bool)
        k = rng.randint(4, 6)
        picks = rng.sample(range(9), k)
        for idx in picks:
            mask[idx // 3, idx % 3] = True
        if not np.all(mask.any(axis=0)):
            continue
        if not np.all(mask.any(axis=1)):
            continue
        if len({np.rot90(mask, i).tobytes() for i in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric 3x3 tile mask")


def _random_layout(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        mask = np.zeros((3, 3), dtype=bool)
        count = rng.randint(4, 7)
        picks = rng.sample(range(9), count)
        for idx in picks:
            mask[idx // 3, idx % 3] = True
        if not np.all(mask.any(axis=0)):
            continue
        if not np.all(mask.any(axis=1)):
            continue

        corners = np.full((3, 3), "", dtype=object)
        used_count = rng.choices((2, 3, 4), weights=(2, 4, 4), k=1)[0]
        used_corners = rng.sample(_CORNERS, used_count)
        fg_cells = list(zip(*np.where(mask)))
        forced = min(len(fg_cells), used_count)
        rng.shuffle(fg_cells)
        for idx, (r, c) in enumerate(fg_cells):
            if idx < forced:
                corners[r, c] = used_corners[idx]
            else:
                corners[r, c] = rng.choice(used_corners)
        if len({str(corners[r, c]) for r, c in zip(*np.where(mask))}) < 2:
            continue
        return mask, corners
    raise RuntimeError("failed to sample a non-degenerate 3x3 layout")


def _paint_layout_cell(cell_size: int, corner: str) -> np.ndarray:
    cell = np.full((cell_size, cell_size), CELL_FILL, dtype=int)
    corner_pos = {
        "nw": (0, 0),
        "ne": (0, cell_size - 1),
        "se": (cell_size - 1, cell_size - 1),
        "sw": (cell_size - 1, 0),
    }
    rr, cc = corner_pos[corner]
    cell[rr, cc] = CELL_MARK
    return cell


def _build_input(
    tile_mask: np.ndarray,
    layout_mask: np.ndarray,
    corners: np.ndarray,
    palette: tuple[int, int],
    cell_size: int,
    rng: random.Random,
) -> np.ndarray:
    left_h = 3
    left_w = 3 + rng.randint(0, 2)
    middle_h = cell_size * 3
    middle_w = middle_h + rng.randint(0, 2)
    right_h = 2
    right_w = 1 + rng.randint(0, 2)

    left_top = rng.randint(0, 2)
    middle_top = rng.randint(0, 2)
    right_top = rng.randint(0, 2)
    left_extra_bottom = rng.randint(0, 2)
    middle_extra_bottom = rng.randint(0, 2)
    right_extra_bottom = rng.randint(0, 2)
    left_inner_left = rng.randint(0, left_w - 3)
    middle_inner_left = rng.randint(0, middle_w - middle_h)
    right_inner_left = rng.randint(0, right_w - 1)

    height = max(
        left_top + left_h + left_extra_bottom,
        middle_top + middle_h + middle_extra_bottom,
        right_top + right_h + right_extra_bottom,
    )
    right_margin = rng.randint(0, 2)
    width = left_w + 1 + middle_w + 1 + right_w + right_margin

    grid = np.full((height, width), BG, dtype=int)

    sep0 = left_w
    sep1 = left_w + 1 + middle_w
    grid[:, sep0] = SEPARATOR
    grid[:, sep1] = SEPARATOR

    for r in range(3):
        for c in range(3):
            if tile_mask[r, c]:
                grid[left_top + r, left_inner_left + c] = TILE_FG

    middle_col0 = sep0 + 1 + middle_inner_left
    for lr in range(3):
        for lc in range(3):
            if not layout_mask[lr, lc]:
                continue
            cell = _paint_layout_cell(cell_size, str(corners[lr, lc]))
            r0 = middle_top + lr * cell_size
            c0 = middle_col0 + lc * cell_size
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell

    palette_col = sep1 + 1 + right_inner_left
    grid[right_top, palette_col] = int(palette[0])
    grid[right_top + 1, palette_col] = int(palette[1])
    return grid


def _validate_semantics(grid: np.ndarray) -> np.ndarray:
    tile_mask, layout_mask, corners, palette = _decode_input(grid)
    out = _render_output(tile_mask, layout_mask, corners, palette)
    solved = main(grid)
    assert np.array_equal(solved, out), "solver output must match direct rendering"

    for rot in range(4):
        simpler = _uniform_rotation_output(tile_mask, layout_mask, palette, rot)
        assert not np.array_equal(out, simpler), "sample degenerates to a uniform-rotation shortcut"

    used_corners = {str(corners[r, c]) for r, c in zip(*np.where(layout_mask))}
    assert len(used_corners) >= 2, "sample must expose at least two corner encodings"
    blocks = []
    for r, c in zip(*np.where(layout_mask)):
        blocks.append(tuple(out[r * 3 : r * 3 + 3, c * 3 : c * 3 + 3].flatten()))
    assert len(set(blocks)) >= 2, "foreground blocks must not all look the same"
    assert int((out == int(palette[0])).sum()) > 0, "output must contain the palette foreground color"
    assert int((out == int(palette[1])).sum()) > 0, "output must contain the palette background color"
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = random.Random()
        tile_mask = _random_tile_mask(rng)
        layout_mask, corners = _random_layout(rng)
        palette = tuple(rng.sample(PALETTE_CHOICES, 2))
        cell_size = rng.choice(CELL_SIZES)
        grid = _build_input(tile_mask, layout_mask, corners, palette, cell_size, rng)
        out = _validate_semantics(grid)

        in_sig = _sig(grid)
        out_sig = _sig(out)
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
