from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, paste


BLACK = 0
BLUE = 1
MAROON = 9

KEEP = 0
MIRROR = 1
ROT180 = 2

GRID_RETRY_LIMIT = 400


def _full_line_positions(region: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(region.shape[0]) if np.all(region[r, :] == color)]
    return [c for c in range(region.shape[1]) if np.all(region[:, c] == color)]


def _parse_three_by_three(region: np.ndarray, frame_color: int) -> list[list[np.ndarray]]:
    row_lines = _full_line_positions(region, frame_color, axis=0)
    col_lines = _full_line_positions(region, frame_color, axis=1)
    assert len(row_lines) == 4 and len(col_lines) == 4, "expected a framed 3x3 lattice"
    assert all(b - a >= 2 for a, b in zip(row_lines, row_lines[1:])), "invalid row partition"
    assert all(b - a >= 2 for a, b in zip(col_lines, col_lines[1:])), "invalid col partition"

    cells: list[list[np.ndarray]] = []
    for i in range(3):
        row_cells = []
        for j in range(3):
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1]
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1]
            row_cells.append(region[r0:r1, c0:c1].copy())
        cells.append(row_cells)
    return cells


def _parse_mask(region: np.ndarray, frame_color: int) -> np.ndarray:
    cells = _parse_three_by_three(region, frame_color)
    mask = np.zeros((3, 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            cell = cells[i][j]
            colors = set(map(int, np.unique(cell)))
            assert colors <= {BLACK, MAROON}, "mask cells must be black or maroon"
            mask[i, j] = bool(np.any(cell == MAROON))
    return mask


def _render_with_cells(cells: list[list[np.ndarray]], frame_color: int) -> np.ndarray:
    cell_h = cells[0][0].shape[0]
    cell_w = cells[0][0].shape[1]
    out = np.full((3 * cell_h + 4, 3 * cell_w + 4), frame_color, dtype=int)
    for i in range(3):
        for j in range(3):
            r0 = 1 + i * (cell_h + 1)
            c0 = 1 + j * (cell_w + 1)
            paste(out, cells[i][j], r0, c0)
    return out


def _apply_action(tile: np.ndarray, action: int) -> np.ndarray:
    if action == ROT180:
        return np.rot90(tile, 2).copy()
    if action == MIRROR:
        return np.fliplr(tile).copy()
    assert action == KEEP
    return tile.copy()


def _extract_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    blue_box = bbox(grid == BLUE)
    assert blue_box is not None, "expected one blue separator bar"
    br0, bc0, br1, bc1 = blue_box
    assert bc0 == bc1, "blue separator must be one column wide"
    assert np.all(grid[br0 : br1 + 1, bc0 : bc1 + 1] == BLUE), "blue separator must be solid"

    band = grid[br0 : br1 + 1, :]
    frame_colors = [int(c) for c in np.unique(band) if int(c) not in {BLACK, BLUE, MAROON}]
    assert len(frame_colors) == 1, "expected one frame color in the mask band"
    frame_color = frame_colors[0]

    frame_boxes = [bbox_coords(coords) for coords in components_mask(grid == frame_color, connectivity=4)]
    mask_boxes = sorted(
        [box for box in frame_boxes if box[0] == br0 and box[2] == br1],
        key=lambda box: box[1],
    )
    top_boxes = [box for box in frame_boxes if box[2] < br0]
    assert len(mask_boxes) == 2 and len(top_boxes) == 1, "failed to identify lattice and masks"

    lattice = crop_bbox(grid, top_boxes[0])
    left_mask = crop_bbox(grid, mask_boxes[0])
    right_mask = crop_bbox(grid, mask_boxes[1])
    return lattice, left_mask, right_mask, frame_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the pairwise logic states of two 3x3 masks to transform the matching tiles in a 3x3 lattice.

    Input structure:
    - The upper part is a framed 3x3 lattice of tiles separated by one shared frame color.
    - Each tile interior is a small non-black pattern that is neither left-right symmetric nor
      180-degree symmetric.
    - The lower part contains two framed 3x3 black/maroon masks separated by a solid blue
      vertical bar; mask cells correspond to lattice tiles by (row, col).

    Rule:
    - Parse the two lower masks cell by cell.
    - For each lattice tile: if both masks are maroon, rotate that tile interior by 180 degrees;
      if exactly one mask is maroon, mirror that tile interior left-right; if both are black,
      leave the tile unchanged.
    - Keep the lattice frame lines unchanged and delete the masks and blue separator from the output.

    Key insight:
    - The lower diagrams are a tile-scale instruction table rather than another picture: their
      three logical states select three different whole-tile geometric actions.
    """
    g = np.asarray(grid, dtype=int)
    lattice, left_mask_region, right_mask_region, frame_color = _extract_layout(g)
    mask_a = _parse_mask(left_mask_region, frame_color)
    mask_b = _parse_mask(right_mask_region, frame_color)
    tile_cells = _parse_three_by_three(lattice, frame_color)

    out_cells: list[list[np.ndarray]] = []
    for i in range(3):
        row_out = []
        for j in range(3):
            both = bool(mask_a[i, j]) and bool(mask_b[i, j])
            xor = bool(mask_a[i, j]) ^ bool(mask_b[i, j])
            action = ROT180 if both else MIRROR if xor else KEEP
            row_out.append(_apply_action(tile_cells[i][j], action))
        out_cells.append(row_out)

    return _render_with_cells(out_cells, frame_color)


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out.append((rr, cc))
    return out


def _grow_connected_mask(size: int, target: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            frontier.update(_neighbors4(mask.shape, int(r), int(c)))
        frontier = [pos for pos in frontier if not mask[pos]]
        if not frontier:
            return mask
        mask[random.choice(frontier)] = True
    return mask


def _tile_variants(tile: np.ndarray) -> list[np.ndarray]:
    return [tile, np.fliplr(tile), np.rot90(tile, 2)]


def _make_tile(size: int, color: int, seen: set[bytes]) -> np.ndarray:
    min_count = max(3, size)
    max_count = min(size * size - 2, size + 3)
    for _ in range(200):
        count = random.randint(min_count, max_count)
        mask = _grow_connected_mask(size, count)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        box = bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r0 == r1 or c0 == c1:
            continue
        tile = np.zeros((size, size), dtype=int)
        tile[mask] = color
        variants = _tile_variants(tile)
        serials = [variant.tobytes() for variant in variants]
        if len(set(serials)) != 3:
            continue
        if any(serial in seen for serial in serials):
            continue
        seen.update(serials)
        return tile
    raise RuntimeError("failed to generate an asymmetric tile")


def _sample_actions() -> np.ndarray:
    while True:
        actions = [KEEP, MIRROR, ROT180]
        actions.extend(random.choices([KEEP, MIRROR, ROT180], k=6))
        counts = {action: actions.count(action) for action in (KEEP, MIRROR, ROT180)}
        if min(counts.values()) < 2:
            continue
        random.shuffle(actions)
        return np.array(actions, dtype=int).reshape(3, 3)


def _actions_to_masks(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_a = np.zeros((3, 3), dtype=bool)
    mask_b = np.zeros((3, 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            action = int(actions[i, j])
            if action == ROT180:
                mask_a[i, j] = True
                mask_b[i, j] = True
            elif action == MIRROR:
                if random.random() < 0.5:
                    mask_a[i, j] = True
                else:
                    mask_b[i, j] = True
    return mask_a, mask_b


def _render_lattice(tiles: list[list[np.ndarray]], frame_color: int) -> np.ndarray:
    return _render_with_cells(tiles, frame_color)


def _render_mask(mask: np.ndarray, cell_size: int, frame_color: int) -> np.ndarray:
    cells = []
    for i in range(3):
        row = []
        for j in range(3):
            color = MAROON if bool(mask[i, j]) else BLACK
            row.append(np.full((cell_size, cell_size), color, dtype=int))
        cells.append(row)
    return _render_with_cells(cells, frame_color)


def _expected_output(tiles: list[list[np.ndarray]], actions: np.ndarray, frame_color: int) -> np.ndarray:
    out_tiles: list[list[np.ndarray]] = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(_apply_action(tiles[i][j], int(actions[i, j])))
        out_tiles.append(row)
    return _render_lattice(out_tiles, frame_color)


def _choose_canvas_dims(top_shape: tuple[int, int], bottom_shape: tuple[int, int]) -> tuple[int, int, int]:
    top_h, top_w = top_shape
    bottom_h, bottom_w = bottom_shape
    max_gap = min(3, 30 - top_h - bottom_h)
    if max_gap < 1:
        raise RuntimeError("no vertical space for both regions")
    gap = random.randint(1, max_gap)
    min_h = top_h + gap + bottom_h
    max_h = min(30, min_h + 3)
    height = random.randint(min_h, max_h)
    min_w = max(top_w, bottom_w)
    max_w = min(30, min_w + 6)
    width = random.randint(min_w, max_w)
    return height, width, gap


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    frame_color = random.choice([2, 3, 4, 5, 6, 7, 8])
    tile_palette = [c for c in [2, 3, 4, 5, 6, 7, 8] if c != frame_color]
    tile_size = random.randint(3, 5)
    mask_cell_size = random.randint(1, 2)

    actions = _sample_actions()
    mask_a, mask_b = _actions_to_masks(actions)
    if not (0 < int(mask_a.sum()) < 9 and 0 < int(mask_b.sum()) < 9):
        raise RuntimeError("degenerate masks")

    seen: set[bytes] = set()
    colors = random.choices(tile_palette, k=9)
    tiles: list[list[np.ndarray]] = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(_make_tile(tile_size, colors[3 * i + j], seen))
        tiles.append(row)

    lattice = _render_lattice(tiles, frame_color)
    left_mask = _render_mask(mask_a, mask_cell_size, frame_color)
    right_mask = _render_mask(mask_b, mask_cell_size, frame_color)
    bottom_h = left_mask.shape[0]
    bottom_w = left_mask.shape[1] + 1 + right_mask.shape[1]
    expected = _expected_output(tiles, actions, frame_color)

    canvas_h, canvas_w, gap = _choose_canvas_dims(lattice.shape, (bottom_h, bottom_w))
    grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)

    top_r = random.randint(0, canvas_h - (lattice.shape[0] + gap + bottom_h))
    bottom_r = top_r + lattice.shape[0] + gap
    top_c = random.randint(0, canvas_w - lattice.shape[1])
    bottom_c = random.randint(0, canvas_w - bottom_w)

    paste(grid, lattice, top_r, top_c)
    paste(grid, left_mask, bottom_r, bottom_c)
    grid[bottom_r : bottom_r + bottom_h, bottom_c + left_mask.shape[1]] = BLUE
    paste(grid, right_mask, bottom_r, bottom_c + left_mask.shape[1] + 1)

    return grid, expected


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    lattice, left_mask_region, right_mask_region, frame_color = _extract_layout(inp)
    tiles = _parse_three_by_three(lattice, frame_color)
    mask_a = _parse_mask(left_mask_region, frame_color)
    mask_b = _parse_mask(right_mask_region, frame_color)
    actions = np.where(mask_a & mask_b, ROT180, np.where(mask_a ^ mask_b, MIRROR, KEEP))
    if set(map(int, np.unique(actions))) != {KEEP, MIRROR, ROT180}:
        return False
    if not (0 < int(mask_a.sum()) < 9 and 0 < int(mask_b.sum()) < 9):
        return False
    for row in tiles:
        for tile in row:
            if len({variant.tobytes() for variant in _tile_variants(tile)}) != 3:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRY_LIMIT):
        try:
            inp, expected = _build_instance()
        except RuntimeError:
            continue
        if _validate_instance(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid mask-lattice instance")
