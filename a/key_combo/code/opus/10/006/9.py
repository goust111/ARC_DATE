from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
SEP = 5
MARK = 8
FG_COLORS = [1, 2, 3, 4, 6, 7, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in N4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_values(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _component_from_mask(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w and mask[sr, sc]):
        return []
    q = deque([start])
    seen = {start}
    out: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _all_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                comp = _component_from_mask(mask, (r, c))
                for rr, cc in comp:
                    seen[rr, cc] = True
                comps.append(comp)
    return comps


def _intervals(length: int, lines: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(lines) + [length]
    parts: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1:
            raise AssertionError("separator lines must leave non-empty cells")
        parts.append((a + 1, b - 1))
    return parts


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if row_lines or col_lines:
            candidates.append((color, row_lines, col_lines))
    assert len(candidates) == 1, "expected exactly one separator color"
    color, row_lines, col_lines = candidates[0]
    row_regions = _intervals(h, row_lines)
    col_regions = _intervals(w, col_lines)
    assert len(col_regions) == 3, "expected 3 column regions"
    assert len(row_regions) in {2, 3}, "expected 2 or 3 row regions"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_regions}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_regions}
    assert len(row_heights) == 1 and len(col_widths) == 1, "cell sizes must be equal"
    return color, row_regions, col_regions


def _cell_index(regions: list[tuple[int, int]], pos: int) -> int:
    for i, (a, b) in enumerate(regions):
        if a <= pos <= b:
            return i
    raise AssertionError("position not inside any cell")


def _extract_target(grid: np.ndarray) -> dict:
    marker_coords = np.argwhere(grid == MARK)
    assert marker_coords.shape[0] == 1, "expected exactly one marker"
    mr, mc = map(int, marker_coords[0])
    sep_color, row_regions, col_regions = _find_layout(grid)
    cell_r = _cell_index(row_regions, mr)
    cell_c = _cell_index(col_regions, mc)
    r0, r1 = row_regions[cell_r]
    c0, c1 = col_regions[cell_c]
    assert sep_color == SEP

    sprite_mask = (grid != BG) & (grid != MARK) & (grid != sep_color)
    seeds = []
    for nr, nc in _neighbors4(mr, mc):
        if r0 <= nr <= r1 and c0 <= nc <= c1 and sprite_mask[nr, nc]:
            seeds.append((nr, nc))
    assert seeds, "marker must touch a sprite"

    components_by_seed = []
    seen = set()
    for seed in seeds:
        comp = tuple(sorted(_component_from_mask(sprite_mask, seed)))
        if comp and comp not in seen:
            seen.add(comp)
            components_by_seed.append(list(comp))
    assert len(components_by_seed) == 1, "marker must select exactly one component"

    target_coords = components_by_seed[0]
    tile, bbox = _crop_values(grid, target_coords)
    tile[tile == MARK] = BG
    tile[tile == sep_color] = BG
    assert np.any(tile != BG), "target tile must be non-empty"
    return {
        "sep_color": sep_color,
        "row_regions": row_regions,
        "col_regions": col_regions,
        "marker": (mr, mc),
        "target_cell": (cell_r, cell_c),
        "target_coords": target_coords,
        "tile": tile,
    }


def _cell_slice(region: tuple[int, int]) -> slice:
    return slice(region[0], region[1] + 1)


def _place_centered(cell_h: int, cell_w: int, tile: np.ndarray) -> tuple[int, int]:
    th, tw = tile.shape
    assert th <= cell_h and tw <= cell_w
    return (cell_h - th) // 2, (cell_w - tw) // 2


def _build_output_from_parsed(grid: np.ndarray, parsed: dict) -> np.ndarray:
    out = np.full(grid.shape, BG, dtype=int)
    sep_color = parsed["sep_color"]
    row_regions = parsed["row_regions"]
    col_regions = parsed["col_regions"]
    tile = parsed["tile"]
    target_cell = parsed["target_cell"]

    row_line_set = set(range(grid.shape[0])) - {
        r for r0, r1 in row_regions for r in range(r0, r1 + 1)
    }
    col_line_set = set(range(grid.shape[1])) - {
        c for c0, c1 in col_regions for c in range(c0, c1 + 1)
    }
    for r in row_line_set:
        out[r, :] = sep_color
    for c in col_line_set:
        out[:, c] = sep_color

    target_color_values = {int(grid[r, c]) for r, c in parsed["target_coords"]}
    target_color_values.discard(BG)
    assert len(target_color_values) == 1, "target sprite must be monochrome"

    for rr, cc in parsed["target_coords"]:
        out[rr, cc] = grid[rr, cc]

    for i, row_region in enumerate(row_regions):
        rs = _cell_slice(row_region)
        cell_h = row_region[1] - row_region[0] + 1
        for j, col_region in enumerate(col_regions):
            if (i, j) == target_cell:
                continue
            cs = _cell_slice(col_region)
            block = grid[rs, cs]
            cell_has_sprite = np.any((block != BG) & (block != MARK))
            if cell_has_sprite:
                continue
            dr, dc = _place_centered(cell_h, col_region[1] - col_region[0] + 1, tile)
            patch = tile != BG
            out[rs.start + dr : rs.start + dr + tile.shape[0], cs.start + dc : cs.start + dc + tile.shape[1]][patch] = tile[patch]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Select the teal-marked sprite, crop it, and tile that crop into every empty separator cell.

    Input structure:
    - A black background contains grey full-width/full-height separator lines that split the board into a 2x3 or 3x3 grid of equal cells.
    - Two or three cells contain distinct single-color 4-connected sprites; exactly one of those cells also contains one teal marker pixel.
    - The teal marker sits just outside the marked sprite's bounding box and is 4-neighbor-adjacent to that sprite.

    Rule:
    - Detect the separator rows and columns to recover the equal cell partition.
    - In the marker's cell, find the unique sprite component touched by the teal marker, then crop that sprite's bounding box to obtain the tile pattern; exclude the marker from the crop.
    - Clear every non-target sprite cell to black, keep the marked sprite in its original position with the marker removed, and copy the cropped tile into every empty cell centered inside the cell with top-left bias when exact centering is impossible.

    Key insight:
    - The teal marker decides which sprite becomes the tile, while the separator grid decides where the tile is repeated; both constraints are required to determine the unique output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    parsed = _extract_target(g)
    return _build_output_from_parsed(g, parsed)


def _random_connected_mask(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(1, max_h + 1))
        w = int(rng.integers(1, max_w + 1))
        if h * w < 3:
            continue
        area = int(rng.integers(3, min(6, h * w) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_to_bbox(mask)
        if mask.sum() < 3 or mask.sum() > 6:
            continue
        return mask
    raise RuntimeError("failed to generate connected mask")


def _mask_signature(mask: np.ndarray) -> tuple:
    return (mask.shape, tuple(map(int, mask.flatten())))


def _marker_candidates(mask: np.ndarray, top: int, left: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = set()
    for c in range(w):
        if mask[0, c] and top > 0:
            candidates.add((top - 1, left + c))
        if mask[h - 1, c] and top + h < cell_h:
            candidates.add((top + h, left + c))
    for r in range(h):
        if mask[r, 0] and left > 0:
            candidates.add((top + r, left - 1))
        if mask[r, w - 1] and left + w < cell_w:
            candidates.add((top + r, left + w))
    return sorted(candidates)


def _possible_target_placements(
    mask: np.ndarray, cell_h: int, cell_w: int
) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = mask.shape
    placements = []
    for top in range(cell_h - h + 1):
        for left in range(cell_w - w + 1):
            for marker in _marker_candidates(mask, top, left, cell_h, cell_w):
                placements.append((top, left, marker))
    return placements


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _build_expected_from_meta(meta: dict) -> np.ndarray:
    out = np.full((meta["H"], meta["W"]), BG, dtype=int)
    for r in meta["row_lines"]:
        out[r, :] = SEP
    for c in meta["col_lines"]:
        out[:, c] = SEP

    target = meta["target"]
    tr0, tc0 = meta["cell_origins"][target["cell"]]
    _paint_mask(out, target["mask"], tr0 + target["top"], tc0 + target["left"], target["color"])

    centered_top, centered_left = _place_centered(meta["cell_h"], meta["cell_w"], target["mask"].astype(int))
    for cell in meta["empty_cells"]:
        r0, c0 = meta["cell_origins"][cell]
        _paint_mask(out, target["mask"], r0 + centered_top, c0 + centered_left, target["color"])
    return out


def _alternative_wrong_tile(meta: dict, distractor: dict) -> np.ndarray:
    out = np.full((meta["H"], meta["W"]), BG, dtype=int)
    for r in meta["row_lines"]:
        out[r, :] = SEP
    for c in meta["col_lines"]:
        out[:, c] = SEP
    target = meta["target"]
    tr0, tc0 = meta["cell_origins"][target["cell"]]
    _paint_mask(out, target["mask"], tr0 + target["top"], tc0 + target["left"], target["color"])
    dtop, dleft = _place_centered(meta["cell_h"], meta["cell_w"], distractor["mask"].astype(int))
    for cell in meta["empty_cells"]:
        r0, c0 = meta["cell_origins"][cell]
        _paint_mask(out, distractor["mask"], r0 + dtop, c0 + dleft, distractor["color"])
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_rows = int(rng.choice([2, 3]))
    cell_cols = 3
    cell_h = int(rng.integers(5, 9))
    cell_w = int(rng.integers(5, 9))
    H = cell_rows * cell_h + (cell_rows - 1)
    W = cell_cols * cell_w + (cell_cols - 1)

    row_lines = [cell_h + i * (cell_h + 1) for i in range(cell_rows - 1)]
    col_lines = [cell_w, 2 * cell_w + 1]

    cell_origins = {}
    all_cells = []
    for i in range(cell_rows):
        for j in range(cell_cols):
            r0 = i * (cell_h + 1)
            c0 = j * (cell_w + 1)
            cell_origins[(i, j)] = (r0, c0)
            all_cells.append((i, j))

    sprite_count = int(rng.choice([2, 3]))
    occupied_idx = rng.choice(len(all_cells), size=sprite_count, replace=False)
    occupied_cells = [all_cells[int(i)] for i in occupied_idx]
    target_cell = occupied_cells[int(rng.integers(sprite_count))]
    empty_cells = [cell for cell in all_cells if cell not in occupied_cells]
    if not empty_cells:
        return None

    colors = list(rng.choice(FG_COLORS, size=sprite_count, replace=False))
    color_by_cell = {cell: colors[i] for i, cell in enumerate(occupied_cells)}

    target_mask = _random_connected_mask(rng, min(4, cell_h - 1), min(4, cell_w - 1))
    target_placements = _possible_target_placements(target_mask, cell_h, cell_w)
    if not target_placements:
        return None
    centered = _place_centered(cell_h, cell_w, target_mask.astype(int))
    off_center = [p for p in target_placements if (p[0], p[1]) != centered]
    if off_center:
        target_top, target_left, marker = off_center[int(rng.integers(len(off_center)))]
    else:
        target_top, target_left, marker = target_placements[int(rng.integers(len(target_placements)))]

    used_signatures = {_mask_signature(target_mask)}
    distractors = []
    for idx, cell in enumerate(occupied_cells):
        if cell == target_cell:
            continue
        for _ in range(200):
            mask = _random_connected_mask(rng, min(4, cell_h), min(4, cell_w))
            sig = _mask_signature(mask)
            if sig in used_signatures:
                continue
            placements = [(r, c) for r in range(cell_h - mask.shape[0] + 1) for c in range(cell_w - mask.shape[1] + 1)]
            if not placements:
                continue
            top, left = placements[int(rng.integers(len(placements)))]
            distractors.append(
                {
                    "cell": cell,
                    "mask": mask,
                    "top": top,
                    "left": left,
                    "color": color_by_cell[cell],
                }
            )
            used_signatures.add(sig)
            break
        else:
            return None

    target_color = color_by_cell[target_cell]
    meta = {
        "H": H,
        "W": W,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "cell_origins": cell_origins,
        "target": {
            "cell": target_cell,
            "mask": target_mask,
            "top": target_top,
            "left": target_left,
            "color": target_color,
            "marker": marker,
        },
        "distractors": distractors,
        "empty_cells": empty_cells,
    }

    grid = np.full((H, W), BG, dtype=int)
    for r in row_lines:
        grid[r, :] = SEP
    for c in col_lines:
        grid[:, c] = SEP

    tr0, tc0 = cell_origins[target_cell]
    _paint_mask(grid, target_mask, tr0 + target_top, tc0 + target_left, target_color)
    grid[tr0 + marker[0], tc0 + marker[1]] = MARK

    for distractor in distractors:
        r0, c0 = cell_origins[distractor["cell"]]
        _paint_mask(grid, distractor["mask"], r0 + distractor["top"], c0 + distractor["left"], distractor["color"])

    expected = _build_expected_from_meta(meta)
    if not _validate_generated(grid, expected, meta):
        return None
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        parsed = _extract_target(grid)
        solved = main(grid)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    if not np.array_equal(parsed["tile"] != BG, meta["target"]["mask"]):
        return False

    if parsed["target_cell"] != meta["target"]["cell"]:
        return False

    marker_r, marker_c = parsed["marker"]
    tr0, tc0 = meta["cell_origins"][meta["target"]["cell"]]
    if (marker_r - tr0, marker_c - tc0) != meta["target"]["marker"]:
        return False

    centered_top, centered_left = _place_centered(meta["cell_h"], meta["cell_w"], meta["target"]["mask"].astype(int))
    target_centered = np.full((meta["cell_h"], meta["cell_w"]), BG, dtype=int)
    _paint_mask(target_centered, meta["target"]["mask"], centered_top, centered_left, meta["target"]["color"])
    target_original = np.full((meta["cell_h"], meta["cell_w"]), BG, dtype=int)
    _paint_mask(
        target_original,
        meta["target"]["mask"],
        meta["target"]["top"],
        meta["target"]["left"],
        meta["target"]["color"],
    )
    if np.array_equal(target_centered, target_original):
        return False

    if not meta["distractors"]:
        return False

    wrong_keep = expected.copy()
    for distractor in meta["distractors"]:
        r0, c0 = meta["cell_origins"][distractor["cell"]]
        _paint_mask(wrong_keep, distractor["mask"], r0 + distractor["top"], c0 + distractor["left"], distractor["color"])
    if np.array_equal(wrong_keep, expected):
        return False

    if np.array_equal(_alternative_wrong_tile(meta, meta["distractors"][0]), expected):
        return False

    comps = _all_components((grid != BG) & (grid != SEP) & (grid != MARK))
    if len(comps) != 1 + len(meta["distractors"]):
        return False

    shape_sigs = {_mask_signature(meta["target"]["mask"])}
    for distractor in meta["distractors"]:
        sig = _mask_signature(distractor["mask"])
        if sig in shape_sigs:
            return False
        shape_sigs.add(sig)

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
