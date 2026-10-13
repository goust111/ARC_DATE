from __future__ import annotations

import itertools
import random

import numpy as np


BOARD_N = 3
TILE = 3
SEP = 1
GRID_SIZE = BOARD_N * TILE + (BOARD_N - 1) * SEP
DIVIDER_ROWS = (TILE, 2 * TILE + 1)
DIVIDER_COLS = (TILE, 2 * TILE + 1)
RETRIES = 400

def _tile_origin(mr: int, mc: int) -> tuple[int, int]:
    return int(mr * (TILE + SEP)), int(mc * (TILE + SEP))


def _extract_tile(grid: np.ndarray, mr: int, mc: int) -> np.ndarray:
    r0, c0 = _tile_origin(mr, mc)
    return grid[r0 : r0 + TILE, c0 : c0 + TILE].copy()


def _find_dividers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if H != GRID_SIZE or W != GRID_SIZE:
        raise AssertionError(f"expected {GRID_SIZE}x{GRID_SIZE} grid, got {g.shape}")

    row_candidates = [r for r in range(H) if np.all(g[r, :] == g[r, 0])]
    col_candidates = [c for c in range(W) if np.all(g[:, c] == g[0, c])]

    for r0, r1 in itertools.combinations(row_candidates, 2):
        if (r0, r1) != DIVIDER_ROWS:
            continue
        divider = int(g[r0, 0])
        if int(g[r1, 0]) != divider:
            continue
        for c0, c1 in itertools.combinations(col_candidates, 2):
            if (c0, c1) != DIVIDER_COLS:
                continue
            if int(g[0, c0]) != divider or int(g[0, c1]) != divider:
                continue
            return (r0, r1), (c0, c1), divider
    raise AssertionError("failed to locate the two separator rows and columns")


def _analyze_board(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    divider_rows, divider_cols, divider = _find_dividers(g)

    tiles: list[dict] = []
    solid_colors = []
    for mr in range(BOARD_N):
        for mc in range(BOARD_N):
            tile = _extract_tile(g, mr, mc)
            if np.all(tile == tile[0, 0]):
                solid_colors.append(int(tile[0, 0]))
            tiles.append({"mr": mr, "mc": mc, "tile": tile})

    if not solid_colors:
        raise AssertionError("at least one empty tile is required")
    bg = solid_colors[0]
    if any(color != bg for color in solid_colors):
        raise AssertionError("all empty tiles must share one background color")

    macro_occ = np.zeros((BOARD_N, BOARD_N), dtype=bool)
    for tile_info in tiles:
        tile = tile_info["tile"]
        if np.any(tile == divider):
            raise AssertionError("tile interiors may not reuse the divider color")
        mask = tile != bg
        tile_info["mask"] = mask
        tile_info["occupied"] = bool(mask.any())
        macro_occ[tile_info["mr"], tile_info["mc"]] = bool(mask.any())

    candidates = []
    for tile_info in tiles:
        if tile_info["occupied"] and np.array_equal(tile_info["mask"], macro_occ):
            candidates.append((tile_info["mr"], tile_info["mc"]))
    if len(candidates) != 1:
        raise AssertionError("expected exactly one self-map tile")

    map_pos = candidates[0]
    return {
        "grid": g,
        "bg": int(bg),
        "divider": int(divider),
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
        "macro_occ": macro_occ,
        "tiles": tiles,
        "map_pos": map_pos,
        "map_tile": _extract_tile(g, *map_pos),
    }


def _apply_rule(analysis: dict) -> np.ndarray:
    out = analysis["grid"].copy()
    bg = int(analysis["bg"])
    map_tile = analysis["map_tile"]

    for tile_info in analysis["tiles"]:
        if not tile_info["occupied"]:
            continue
        mr = int(tile_info["mr"])
        mc = int(tile_info["mc"])
        color = int(map_tile[mr, mc])
        if color == bg:
            raise AssertionError("map tile must color every occupied macro position")
        r0, c0 = _tile_origin(mr, mc)
        patch = out[r0 : r0 + TILE, c0 : c0 + TILE]
        patch[tile_info["mask"]] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the one tile whose 3x3 occupancy pattern equals the board's own occupancy map, then use its pixel colors to recolor every occupied tile.

    Input structure:
    - Two full separator rows and two full separator columns split the canvas into a 3x3 board of 3x3 tiles.
    - Some tiles contain multicolor patterns on a shared background color, while the remaining tiles are completely empty.
    - Exactly one non-empty tile has the same occupied-vs-empty 3x3 pattern as the whole board's non-empty tile layout.

    Rule:
    - Compress the board to a 3x3 occupancy map by marking which tiles are non-empty.
    - Locate the unique tile whose own occupied cells match that occupancy map exactly; this is the map tile.
    - For each non-empty tile at macro position (r,c), recolor all of its non-background cells to the color of the map tile at pixel (r,c). Empty tiles and separator lines stay unchanged.

    Key insight:
    - One tile is a self-portrait of the board at a smaller scale, and its pixel colors act as per-position repaint instructions, including for the map tile itself.
    """
    return _apply_rule(_analyze_board(grid))


def _build_mask_catalog() -> tuple[list[np.ndarray], dict[int, list[np.ndarray]], list[np.ndarray]]:
    all_masks: list[np.ndarray] = []
    by_area: dict[int, list[np.ndarray]] = {}
    macro_masks: list[np.ndarray] = []

    for bits in itertools.product([0, 1], repeat=TILE * TILE):
        mask = np.array(bits, dtype=bool).reshape(TILE, TILE)
        area = int(mask.sum())
        if 2 <= area <= 8:
            all_masks.append(mask)
            by_area.setdefault(area, []).append(mask)
        if 3 <= area <= 6:
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            if len(rows) >= 2 and len(cols) >= 2:
                macro_masks.append(mask)

    return all_masks, by_area, macro_masks


ALL_MASKS, MASKS_BY_AREA, MACRO_MASKS = _build_mask_catalog()


def _paint_mask(mask: np.ndarray, bg: int, palette: list[int]) -> np.ndarray:
    area = int(mask.sum())
    if area < 2:
        raise AssertionError("non-empty tiles must stay visibly multicolor")

    n_colors = random.randint(2, min(4, len(palette), area))
    chosen = random.sample(palette, n_colors)
    colors = list(chosen)
    while len(colors) < area:
        colors.append(random.choice(chosen))
    random.shuffle(colors)

    tile = np.full((TILE, TILE), int(bg), dtype=int)
    for (r, c), color in zip(np.argwhere(mask), colors):
        tile[int(r), int(c)] = int(color)
    return tile


def _assemble_grid(bg: int, divider: int, tile_bank: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    grid[DIVIDER_ROWS[0], :] = int(divider)
    grid[DIVIDER_ROWS[1], :] = int(divider)
    grid[:, DIVIDER_COLS[0]] = int(divider)
    grid[:, DIVIDER_COLS[1]] = int(divider)

    for mr in range(BOARD_N):
        for mc in range(BOARD_N):
            r0, c0 = _tile_origin(mr, mc)
            grid[r0 : r0 + TILE, c0 : c0 + TILE] = tile_bank[(mr, mc)]
    return grid


def _expected_from_blueprint(grid: np.ndarray, bg: int, macro_occ: np.ndarray, map_tile: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for mr in range(BOARD_N):
        for mc in range(BOARD_N):
            if not bool(macro_occ[mr, mc]):
                continue
            color = int(map_tile[mr, mc])
            r0, c0 = _tile_origin(mr, mc)
            tile = out[r0 : r0 + TILE, c0 : c0 + TILE]
            tile[tile != bg] = color
    return out


def _semantic_checks(
    grid: np.ndarray,
    *,
    bg: int,
    divider: int,
    macro_occ: np.ndarray,
    map_pos: tuple[int, int],
    expected: np.ndarray,
    same_area_decoy: tuple[int, int],
) -> bool:
    analysis = _analyze_board(grid)
    if analysis["bg"] != bg or analysis["divider"] != divider:
        return False
    if analysis["map_pos"] != map_pos:
        return False
    if not np.array_equal(analysis["macro_occ"], macro_occ):
        return False

    area = int(macro_occ.sum())
    if area < 3 or area > 6:
        return False

    occupied_tiles = [t for t in analysis["tiles"] if t["occupied"]]
    if len(occupied_tiles) != area:
        return False
    if np.array_equal(analysis["tiles"][same_area_decoy[0] * BOARD_N + same_area_decoy[1]]["mask"], macro_occ):
        return False

    same_area_count = 0
    for tile_info in occupied_tiles:
        tile = tile_info["tile"]
        colors = sorted(set(map(int, tile[tile != bg].flatten())))
        if len(colors) < 2:
            return False
        if int(tile_info["mask"].sum()) == area:
            same_area_count += 1
    if same_area_count < 2:
        return False

    map_tile = analysis["map_tile"]
    map_colors = sorted(set(map(int, map_tile[macro_occ].flatten())))
    if len(map_colors) < 2:
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    mr, mc = map_pos
    r0, c0 = _tile_origin(mr, mc)
    map_in = grid[r0 : r0 + TILE, c0 : c0 + TILE]
    map_out = out[r0 : r0 + TILE, c0 : c0 + TILE]
    self_color = int(map_tile[mr, mc])
    if np.array_equal(map_in, map_out):
        return False
    if not np.all(map_out[macro_occ] == self_color):
        return False

    target_pos = same_area_decoy
    tr, tc = target_pos
    old_color = int(map_tile[tr, tc])
    replacement_choices = [c for c in range(10) if c not in (bg, divider, old_color)]
    if not replacement_choices:
        return False
    replacement = random.choice(replacement_choices)
    changed = grid.copy()
    changed[r0 + tr, c0 + tc] = replacement
    changed_out = main(changed)
    if np.array_equal(changed_out, out):
        return False

    dr0, dc0 = _tile_origin(tr, tc)
    decoy_before = out[dr0 : dr0 + TILE, dc0 : dc0 + TILE]
    decoy_after = changed_out[dr0 : dr0 + TILE, dc0 : dc0 + TILE]
    if np.array_equal(decoy_before, decoy_after):
        return False
    if not np.all(decoy_after[analysis["tiles"][tr * BOARD_N + tc]["mask"]] == replacement):
        return False

    return True


def _make_attempt() -> np.ndarray | None:
    bg, divider = random.sample(range(10), 2)
    palette = [c for c in range(10) if c not in (bg, divider)]
    if len(palette) < 4:
        return None

    macro_occ = random.choice(MACRO_MASKS).copy()
    occupied_positions = [tuple(map(int, pos)) for pos in np.argwhere(macro_occ)]
    map_pos = random.choice(occupied_positions)

    same_area_pool = [
        mask for mask in MASKS_BY_AREA[int(macro_occ.sum())] if not np.array_equal(mask, macro_occ)
    ]
    if not same_area_pool:
        return None

    non_map_positions = [pos for pos in occupied_positions if pos != map_pos]
    if not non_map_positions:
        return None
    same_area_decoy = random.choice(non_map_positions)

    tile_bank: dict[tuple[int, int], np.ndarray] = {
        (mr, mc): np.full((TILE, TILE), int(bg), dtype=int)
        for mr in range(BOARD_N)
        for mc in range(BOARD_N)
    }

    map_tile = _paint_mask(macro_occ, bg, palette)
    tile_bank[map_pos] = map_tile

    for pos in non_map_positions:
        if pos == same_area_decoy:
            mask = random.choice(same_area_pool)
        else:
            mask = random.choice([m for m in ALL_MASKS if not np.array_equal(m, macro_occ) and int(m.sum()) <= 6])
        tile_bank[pos] = _paint_mask(mask, bg, palette)

    grid = _assemble_grid(bg, divider, tile_bank)
    expected = _expected_from_blueprint(grid, bg, macro_occ, map_tile)

    if not _semantic_checks(
        grid,
        bg=bg,
        divider=divider,
        macro_occ=macro_occ,
        map_pos=map_pos,
        expected=expected,
        same_area_decoy=same_area_decoy,
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated semantic validation")
