from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared_for_042_2", str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py"))
).load_module()
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste
scale_sprite = _ARC_UTILS.scale_sprite


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PATTERN_COLORS = (BLUE, RED, GREEN)
NON_CORNER_CELLS = (1, 3, 4, 5, 7)
NEAR_SQUARE_DIMS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4))
GEN_RETRIES = 400


def _panel_lines(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    grey_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    candidates: list[tuple[tuple[int, int, int, int], list[int], list[int], int]] = []
    for comp in grey_components:
        box = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = box
        rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GREY)]
        cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == GREY)]
        if len(rows) != 4 or len(cols) != 4:
            continue
        row_sizes = [rows[i + 1] - rows[i] - 1 for i in range(3)]
        col_sizes = [cols[i + 1] - cols[i] - 1 for i in range(3)]
        if min(row_sizes) <= 0 or min(col_sizes) <= 0:
            continue
        if len(set(row_sizes)) != 1 or len(set(col_sizes)) != 1 or row_sizes[0] != col_sizes[0]:
            continue
        candidates.append((box, rows, cols, int(comp["area"])))
    assert len(candidates) == 1, "expected one 3x3 grey partition panel"
    box, rows, cols, _area = candidates[0]
    return box, rows, cols


def _extract_cells(grid: np.ndarray, rows: list[int], cols: list[int]) -> list[np.ndarray]:
    cells: list[np.ndarray] = []
    for br in range(3):
        for bc in range(3):
            cell = grid[rows[br] + 1 : rows[br + 1], cols[bc] + 1 : cols[bc + 1]].copy()
            cells.append(cell)
    return cells


def _normal_and_outlier(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    keyed = [cell.tobytes() for cell in cells]
    counts = Counter(keyed)
    assert sorted(counts.values()) == [1, 8], "expected one outlier tile and eight normal tiles"
    normal_key = next(key for key, count in counts.items() if count == 8)
    outlier_key = next(key for key, count in counts.items() if count == 1)
    outlier_index = keyed.index(outlier_key)
    return cells[keyed.index(normal_key)], cells[outlier_index], int(outlier_index)


def _difference_patch(normal_tile: np.ndarray, outlier_tile: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    diff_mask = outlier_tile != normal_tile
    diff_box = bbox(diff_mask)
    assert diff_box is not None, "outlier tile must differ from the normal tile"
    patch = crop_bbox(np.where(diff_mask, outlier_tile, BG), diff_box)
    return patch, tuple(map(int, diff_box))


def _frame_bbox(grid: np.ndarray, min_left: int = 0) -> tuple[int, int, int, int]:
    red_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[RED],
        return_format="dict",
    )
    candidates: list[tuple[int, int, int, int]] = []
    for comp in red_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if c0 <= int(min_left):
            continue
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4 or h != w:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(sub, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(sub[border_mask] == RED):
            continue
        if np.any(sub[~border_mask] == RED):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one hollow red frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the anomalous tile's pixelwise difference from the repeated tile pattern, then scale and stamp that difference into the hollow frame.

    Input structure:
    - Black background with a grey 3x3 partition panel on the left and a hollow red square frame on the right.
    - Each grey-bounded cell contains the same multicolor tile except for exactly one non-corner outlier tile.
    - The outlier differs from the normal tile at a small set of pixel positions, and the frame interior is an integer 2x or 3x multiple of that difference bbox's longer side.

    Rule:
    - Use the grey full-row and full-column separators to recover the 3x3 cell contents.
    - Identify the tile pattern that appears eight times as normal and the singleton tile as the outlier.
    - Form a difference mask by keeping only pixels where the outlier color differs from the normal tile, crop it to its minimal bbox, scale it by the frame-determined integer factor, and stamp it into the frame interior aligned to the inner top-left corner.
    - Preserve the original panel, separators, background, and red frame border.

    Key insight:
    - The answer is not the unique tile itself but only its deviation from the repeated tile, and the red frame independently determines how large that deviation must be rendered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_box, rows, cols = _panel_lines(g)
    cells = _extract_cells(g, rows, cols)
    normal_tile, outlier_tile, _outlier_index = _normal_and_outlier(cells)
    patch, diff_box = _difference_patch(normal_tile, outlier_tile)

    frame_box = _frame_bbox(g, min_left=panel_box[3])
    fr0, fc0, fr1, fc1 = frame_box
    interior_side = fr1 - fr0 - 1
    bbox_long = max(diff_box[2] - diff_box[0] + 1, diff_box[3] - diff_box[1] + 1)
    assert interior_side % bbox_long == 0, "frame scale must be an integer"
    scale = interior_side // bbox_long
    assert scale in (2, 3), "expected frame-driven scale factor 2 or 3"

    stamp = scale_sprite(patch, scale)
    out = g.copy()
    paste(out, stamp, fr0 + 1, fc0 + 1, transparent=BG)
    return out


def _sample_base_tile(cell_size: int, rng: random.Random) -> np.ndarray:
    total = cell_size * cell_size
    for _ in range(200):
        tile = np.zeros((cell_size, cell_size), dtype=int)
        colors = rng.sample(list(PATTERN_COLORS), rng.randint(2, 3))
        colored_count = rng.randint(max(len(colors) + 1, total // 2), total - 1)
        coords = rng.sample([(r, c) for r in range(cell_size) for c in range(cell_size)], colored_count)
        rng.shuffle(coords)
        for color, coord in zip(colors, coords):
            tile[coord] = color
        for coord in coords[len(colors) :]:
            tile[coord] = rng.choice(colors)
        if len(set(map(int, tile[tile != BG].tolist()))) != len(colors):
            continue
        if np.count_nonzero(tile == BG) == 0:
            continue
        return tile
    raise RuntimeError("failed to sample a diverse base tile")


def _sample_diff(base_tile: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    cell_size = int(base_tile.shape[0])
    positions = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    for _ in range(300):
        diff_count = rng.randint(1, 4)
        diff_positions = rng.sample(positions, diff_count)
        rs = [r for r, _c in diff_positions]
        cs = [c for _r, c in diff_positions]
        box = (min(rs), min(cs), max(rs), max(cs))
        dims = (box[2] - box[0] + 1, box[3] - box[1] + 1)
        if dims not in NEAR_SQUARE_DIMS:
            continue

        mask = np.zeros_like(base_tile, dtype=bool)
        for r, c in diff_positions:
            mask[r, c] = True

        sub_base = base_tile[box[0] : box[2] + 1, box[1] : box[3] + 1]
        sub_mask = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        unchanged_visible = np.any((sub_base != BG) & ~sub_mask)
        if not unchanged_visible:
            continue

        outlier = base_tile.copy()
        for r, c in diff_positions:
            choices = [color for color in PATTERN_COLORS if color != int(base_tile[r, c])]
            outlier[r, c] = rng.choice(choices)

        patch = crop_bbox(np.where(mask, outlier, BG), box)
        if np.array_equal(scale_sprite(crop_bbox(outlier, box), 2), scale_sprite(patch, 2)):
            continue
        return outlier, mask, tuple(map(int, box))
    raise RuntimeError("failed to sample a valid outlier tile")


def _expected_output(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    diff_box: tuple[int, int, int, int],
    outlier_tile: np.ndarray,
    normal_tile: np.ndarray,
    scale: int,
) -> np.ndarray:
    diff_mask = outlier_tile != normal_tile
    patch = crop_bbox(np.where(diff_mask, outlier_tile, BG), diff_box)
    out = grid.copy()
    paste(out, scale_sprite(patch, scale), frame_top + 1, frame_left + 1, transparent=BG)
    return out


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    outlier_index: int,
    normal_tile: np.ndarray,
    outlier_tile: np.ndarray,
    diff_box: tuple[int, int, int, int],
    scale: int,
) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert outlier_index in NON_CORNER_CELLS

    _panel_box, rows, cols = _panel_lines(inp)
    cells = _extract_cells(inp, rows, cols)
    normal2, outlier2, outlier_index2 = _normal_and_outlier(cells)
    assert outlier_index2 == outlier_index
    assert np.array_equal(normal2, normal_tile)
    assert np.array_equal(outlier2, outlier_tile)

    diff_mask = outlier_tile != normal_tile
    assert 1 <= int(diff_mask.sum()) <= 4
    assert tuple(map(int, bbox(diff_mask))) == diff_box
    assert scale in (2, 3)

    patch = crop_bbox(np.where(diff_mask, outlier_tile, BG), diff_box)
    naive_crop = crop_bbox(outlier_tile, diff_box)
    assert not np.array_equal(patch, naive_crop), "difference stamp must not equal the outlier crop"

    panel_box, _rows, _cols = _panel_lines(inp)
    frame_box = _frame_bbox(inp, min_left=panel_box[3])
    interior_side = frame_box[2] - frame_box[0] - 1
    bbox_long = max(diff_box[2] - diff_box[0] + 1, diff_box[3] - diff_box[1] + 1)
    assert interior_side == bbox_long * scale

    solved = main(inp)
    assert np.array_equal(solved, expected), "main() must match the constructed target"
    assert not np.array_equal(inp, solved), "output must change the input"


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict] | None:
    cell_size = rng.choice([3, 4])
    panel_size = 3 * cell_size + 4
    normal_tile = _sample_base_tile(cell_size, rng)
    outlier_tile, _mask, diff_box = _sample_diff(normal_tile, rng)
    bbox_long = max(diff_box[2] - diff_box[0] + 1, diff_box[3] - diff_box[1] + 1)

    gap = rng.randint(1, 3)
    left_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 2)
    valid_scales = []
    for scale in (2, 3):
        frame_outer = bbox_long * scale + 2
        width = left_margin + panel_size + gap + frame_outer + right_margin
        if width <= 30:
            valid_scales.append(scale)
    if not valid_scales:
        return None
    scale = rng.choice(valid_scales)
    frame_outer = bbox_long * scale + 2

    panel_top = rng.randint(0, 3)
    frame_top = rng.randint(0, 6)
    bottom_margin = rng.randint(0, 2)
    grid_h = max(panel_top + panel_size, frame_top + frame_outer) + bottom_margin
    grid_w = left_margin + panel_size + gap + frame_outer + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    panel_left = left_margin
    row_lines = [panel_top + i * (cell_size + 1) for i in range(4)]
    col_lines = [panel_left + i * (cell_size + 1) for i in range(4)]
    grid[row_lines[0] : row_lines[-1] + 1, col_lines] = GREY
    grid[row_lines, col_lines[0] : col_lines[-1] + 1] = GREY

    outlier_index = rng.choice(NON_CORNER_CELLS)
    for idx in range(9):
        br, bc = divmod(idx, 3)
        r0 = row_lines[br] + 1
        c0 = col_lines[bc] + 1
        tile = outlier_tile if idx == outlier_index else normal_tile
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = tile

    frame_left = panel_left + panel_size + gap
    grid[frame_top : frame_top + frame_outer, frame_left] = RED
    grid[frame_top : frame_top + frame_outer, frame_left + frame_outer - 1] = RED
    grid[frame_top, frame_left : frame_left + frame_outer] = RED
    grid[frame_top + frame_outer - 1, frame_left : frame_left + frame_outer] = RED

    expected = _expected_output(grid, frame_top, frame_left, diff_box, outlier_tile, normal_tile, scale)
    meta = {
        "outlier_index": outlier_index,
        "normal_tile": normal_tile,
        "outlier_tile": outlier_tile,
        "diff_box": diff_box,
        "scale": scale,
    }
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        inp, expected, meta = attempt
        _validate_instance(inp, expected, **meta)
        return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
