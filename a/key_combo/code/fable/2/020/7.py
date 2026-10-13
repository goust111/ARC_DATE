from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import crop_bbox, find_components, paste


BLACK = 0
GENERATION_RETRIES = 400
KEY_CORNERS = ("tl", "tr", "bl", "br")


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _grid_lattice_component(component: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    coords = component["coords"] - np.array([r0, c0], dtype=int)
    mask[coords[:, 0], coords[:, 1]] = True

    full_rows = [r for r in range(h) if mask[r].all()]
    full_cols = [c for c in range(w) if mask[:, c].all()]
    if len(full_rows) != 2 or len(full_cols) != 2:
        return None

    row_spans = [full_rows[0], full_rows[1] - full_rows[0] - 1, h - full_rows[1] - 1]
    col_spans = [full_cols[0], full_cols[1] - full_cols[0] - 1, w - full_cols[1] - 1]
    if min(row_spans) <= 0 or min(col_spans) <= 0:
        return None
    if len(set(row_spans)) != 1 or len(set(col_spans)) != 1 or row_spans[0] != col_spans[0]:
        return None

    expected = np.zeros_like(mask)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    if not np.array_equal(mask, expected):
        return None

    tile_size = int(row_spans[0])
    tile_bboxes = []
    for tr in range(3):
        for tc in range(3):
            rr0 = r0 + tr * (tile_size + 1)
            cc0 = c0 + tc * (tile_size + 1)
            tile_bboxes.append((rr0, cc0, rr0 + tile_size - 1, cc0 + tile_size - 1))

    return {
        "divider_color": int(component["color"]),
        "board_bbox": (r0, c0, r1, c1),
        "tile_size": tile_size,
        "tile_bboxes": tile_bboxes,
    }


def _find_board(grid: np.ndarray) -> dict:
    candidates = []
    for component in find_components(grid, bg=BLACK, connectivity=4, monochromatic=True):
        parsed = _grid_lattice_component(component)
        if parsed is not None:
            candidates.append(parsed)
    assert len(candidates) == 1, "expected one divider lattice"
    return candidates[0]


def _key_panel_info(panel: np.ndarray) -> tuple[int, int] | None:
    values, counts = np.unique(panel, return_counts=True)
    if BLACK in values or len(values) != 2:
        return None
    marker_idx = int(np.argmin(counts))
    fill_idx = int(np.argmax(counts))
    if counts[marker_idx] >= counts[fill_idx]:
        return None
    marker = int(values[marker_idx])
    fill = int(values[fill_idx])
    if int(panel[0, 0]) != marker:
        return None
    return marker, fill


def _find_key_panel(grid: np.ndarray, board_bbox: tuple[int, int, int, int]) -> dict:
    candidates = []
    for component in find_components(grid, bg=BLACK, connectivity=4, monochromatic=False):
        bbox = tuple(map(int, component["bbox"]))
        if _bbox_intersects(bbox, board_bbox):
            continue
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3) or int(component["area"]) != 9:
            continue
        panel = crop_bbox(grid, bbox)
        info = _key_panel_info(panel)
        if info is None:
            continue
        marker, fill = info
        candidates.append(
            {
                "key_bbox": bbox,
                "key_panel": panel,
                "marker_color": marker,
                "fill_color": fill,
            }
        )
    assert len(candidates) == 1, "expected one 3x3 key panel"
    return candidates[0]


def _parse_grid(grid: np.ndarray) -> dict:
    board = _find_board(grid)
    key = _find_key_panel(grid, board["board_bbox"])
    return {**board, **key}


def _complete_tile(template: np.ndarray, tile: np.ndarray, divider_color: int) -> np.ndarray:
    out = tile.copy()
    repair_mask = (template != BLACK) & (tile != template)
    out[repair_mask] = int(divider_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner 3x3 key to decide which tiles in the divided 3x3 array are completed and which are erased.

    Input structure:
    - A one-cell-thick divider lattice splits the main body into a 3 by 3 array of equal square tiles.
    - The top-left tile is the complete template, while the other tiles are black-missing versions of that template.
    - Outside the main body there is one isolated 3 by 3 key panel with exactly two colors; its top-left cell is the sparse marker color.

    Rule:
    - Match the nine key cells to the nine large tiles by position.
    - For a marker-color key cell, compare the corresponding tile with the top-left template and recolor every missing or mismatching template-foreground cell to the divider color.
    - For a fill-color key cell, erase the whole corresponding tile to black; keep the divider lattice and the key panel unchanged.

    Key insight:
    - The small key is a coarse control map over the large 3 by 3 tile array, so the solver must first recover the shared position reference frame before applying completion versus erasure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    parsed = _parse_grid(g)
    template = crop_bbox(g, parsed["tile_bboxes"][0])
    out = g.copy()

    for index, bbox in enumerate(parsed["tile_bboxes"]):
        r0, c0, r1, c1 = bbox
        kr, kc = divmod(index, 3)
        tile = crop_bbox(g, bbox)
        if int(parsed["key_panel"][kr, kc]) == int(parsed["fill_color"]):
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = _complete_tile(template, tile, parsed["divider_color"])
    return out


def _random_palette(exclude: set[int], count: int) -> list[int]:
    choices = [color for color in range(1, 10) if color not in exclude]
    return random.sample(choices, count)


def _random_template(tile_size: int, palette: list[int]) -> np.ndarray | None:
    cells = [(r, c) for r in range(tile_size) for c in range(tile_size)]
    area = tile_size * tile_size
    min_fg = max(6, area // 3)
    max_fg = min(area - 2, (2 * area) // 3)
    for _ in range(200):
        fg_count = random.randint(min_fg, max_fg)
        chosen = random.sample(cells, fg_count)
        color_count = random.randint(1, min(3, len(palette)))
        active_palette = random.sample(palette, color_count)

        values = [random.choice(active_palette) for _ in range(fg_count)]
        for color in active_palette:
            values[random.randrange(fg_count)] = color

        tile = np.zeros((tile_size, tile_size), dtype=int)
        for (r, c), color in zip(chosen, values):
            tile[r, c] = int(color)

        rows = np.any(tile != BLACK, axis=1)
        cols = np.any(tile != BLACK, axis=0)
        if rows.sum() < 2 or cols.sum() < 2:
            continue
        if np.all(tile != BLACK) or np.all(tile == BLACK):
            continue
        return tile
    return None


def _corrupt_tile(template: np.ndarray, remove_lo: int, remove_hi: int) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template != BLACK)]
    keep_at_least = 1
    max_remove = min(int(remove_hi), len(coords) - keep_at_least)
    min_remove = min(int(remove_lo), max_remove)
    remove_n = random.randint(min_remove, max_remove) if max_remove >= min_remove else max_remove
    corrupted = template.copy()
    if remove_n > 0:
        for r, c in random.sample(coords, remove_n):
            corrupted[r, c] = BLACK
    return corrupted


def _key_panel(marker_color: int, fill_color: int) -> tuple[np.ndarray, set[int]]:
    marker_count = random.randint(2, 4)
    positions = list(range(1, 9))
    marker_indices = {0, *random.sample(positions, marker_count - 1)}
    panel = np.full((3, 3), fill_color, dtype=int)
    for idx in marker_indices:
        r, c = divmod(idx, 3)
        panel[r, c] = marker_color
    return panel, set(marker_indices)


def _layout_for_corner(board_size: int, corner: str) -> dict:
    gap_r = random.randint(1, 2)
    gap_c = random.randint(1, 2)
    outer_top = random.randint(1, 2)
    outer_bottom = random.randint(1, 2)
    outer_left = random.randint(1, 2)
    outer_right = random.randint(1, 2)

    key_top = corner in {"tl", "tr"}
    key_left = corner in {"tl", "bl"}

    if key_top:
        key_r0 = 0
        board_r0 = 3 + gap_r
        height = board_r0 + board_size + outer_bottom
    else:
        board_r0 = outer_top
        key_r0 = board_r0 + board_size + gap_r
        height = key_r0 + 3

    if key_left:
        key_c0 = 0
        board_c0 = 3 + gap_c
        width = board_c0 + board_size + outer_right
    else:
        board_c0 = outer_left
        key_c0 = board_c0 + board_size + gap_c
        width = key_c0 + 3

    assert height <= 30 and width <= 30
    return {
        "shape": (height, width),
        "board_origin": (board_r0, board_c0),
        "key_origin": (key_r0, key_c0),
    }


def _tile_bboxes(board_r0: int, board_c0: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for tr in range(3):
        for tc in range(3):
            r0 = board_r0 + tr * (tile_size + 1)
            c0 = board_c0 + tc * (tile_size + 1)
            bboxes.append((r0, c0, r0 + tile_size - 1, c0 + tile_size - 1))
    return bboxes


def _board_patch(tile_size: int, divider_color: int) -> np.ndarray:
    board_size = 3 * tile_size + 2
    board = np.zeros((board_size, board_size), dtype=int)
    for offset in (tile_size, 2 * tile_size + 1):
        board[offset, :] = divider_color
        board[:, offset] = divider_color
    return board


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    template = crop_bbox(grid, meta["tile_bboxes"][0])
    for idx, bbox in enumerate(meta["tile_bboxes"]):
        r0, c0, r1, c1 = bbox
        tile = crop_bbox(grid, bbox)
        if idx in meta["erase_indices"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = _complete_tile(template, tile, meta["divider_color"])
    return out


def _build_case() -> tuple[np.ndarray, dict] | None:
    tile_size = random.randint(4, 7)
    board_size = 3 * tile_size + 2
    divider_color, key_marker_color, key_fill_color = _random_palette({BLACK}, 3)
    template_palette = _random_palette({BLACK, divider_color, key_marker_color, key_fill_color}, random.randint(1, 3))
    template = _random_template(tile_size, template_palette)
    if template is None:
        return None

    panel, marker_indices = _key_panel(key_marker_color, key_fill_color)
    if len(marker_indices) <= 1 or len(marker_indices) >= 9:
        return None

    layout = _layout_for_corner(board_size, random.choice(KEY_CORNERS))
    grid = np.zeros(layout["shape"], dtype=int)
    board = _board_patch(tile_size, divider_color)
    board_r0, board_c0 = layout["board_origin"]
    paste(grid, board, board_r0, board_c0)

    tile_bboxes = _tile_bboxes(board_r0, board_c0, tile_size)
    for idx, bbox in enumerate(tile_bboxes):
        r0, c0, _r1, _c1 = bbox
        if idx == 0:
            tile = template
        elif idx in marker_indices:
            tile = _corrupt_tile(template, 1, max(1, int(np.sum(template != BLACK)) // 2))
        else:
            tile = _corrupt_tile(template, 0, max(0, int(np.sum(template != BLACK)) // 3))
        paste(grid, tile, r0, c0, transparent=BLACK)

    key_r0, key_c0 = layout["key_origin"]
    paste(grid, panel, key_r0, key_c0)

    meta = {
        "divider_color": divider_color,
        "key_panel": panel,
        "key_origin": layout["key_origin"],
        "board_origin": layout["board_origin"],
        "tile_bboxes": tile_bboxes,
        "erase_indices": set(range(9)) - set(marker_indices),
        "marker_indices": set(marker_indices),
    }
    return grid, meta


def _validate_generated_case(grid: np.ndarray, meta: dict) -> bool:
    out = main(grid)
    expected = _expected_from_meta(grid, meta)
    if not np.array_equal(out, expected):
        return False

    parsed = _parse_grid(grid)
    if parsed["divider_color"] != meta["divider_color"]:
        return False
    if tuple(parsed["key_bbox"][:2]) != tuple(meta["key_origin"]):
        return False
    if not np.array_equal(parsed["key_panel"], meta["key_panel"]):
        return False

    divider_changed = False
    erased_changed = False
    template = crop_bbox(grid, meta["tile_bboxes"][0])
    for idx, bbox in enumerate(meta["tile_bboxes"]):
        in_tile = crop_bbox(grid, bbox)
        out_tile = crop_bbox(out, bbox)
        if idx in meta["erase_indices"]:
            if not np.all(out_tile == BLACK):
                return False
            if np.any(in_tile != BLACK):
                erased_changed = True
        else:
            repaired = _complete_tile(template, in_tile, meta["divider_color"])
            if not np.array_equal(out_tile, repaired):
                return False
            if idx != 0 and np.any(out_tile != in_tile):
                divider_changed = True

    kr0, kc0 = meta["key_origin"]
    if not np.array_equal(out[kr0 : kr0 + 3, kc0 : kc0 + 3], grid[kr0 : kr0 + 3, kc0 : kc0 + 3]):
        return False
    return erased_changed and divider_changed


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, meta = built
        if _validate_generated_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid key-controlled tile puzzle")
