from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
RECT_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
AREA_VALUES = tuple(range(2, 11))
TILE_SHAPES = ((5, 7), (7, 5), (6, 7), (7, 6))
GENERATION_ATTEMPTS = 400
NOISE_COUNTS = (1, 2, 3)
MODE_BAG = (
    "attached_same",
    "attached_same",
    "attached_same",
    "same_color",
    "same_color",
    "same_color",
    "foreign",
    "foreign",
    "foreign",
)


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    start = 0
    out: list[tuple[int, int]] = []
    for sep in separators + [length]:
        out.append((start, sep))
        start = sep + 1
    return out


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(row_seps) == 2
    assert len(col_seps) == 2

    row_segments = _segments(grid.shape[0], row_seps)
    col_segments = _segments(grid.shape[1], col_seps)
    assert len(row_segments) == 3
    assert len(col_segments) == 3

    row_sizes = {r1 - r0 for r0, r1 in row_segments}
    col_sizes = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_sizes) == 1 and min(row_sizes) > 0
    assert len(col_sizes) == 1 and min(col_sizes) > 0
    return row_segments, col_segments


def _largest_solid_rectangle(tile: np.ndarray) -> dict[str, int]:
    h, w = tile.shape
    candidates: list[tuple[int, int, int, int, int, int]] = []

    for color in sorted(int(v) for v in np.unique(tile) if int(v) != BG):
        mask = tile == color
        for top in range(h):
            full_cols = np.ones(w, dtype=bool)
            for bottom in range(top, h):
                full_cols &= mask[bottom]
                col = 0
                while col < w:
                    while col < w and not full_cols[col]:
                        col += 1
                    if col == w:
                        break
                    left = col
                    while col < w and full_cols[col]:
                        col += 1
                    right = col - 1
                    area = (bottom - top + 1) * (right - left + 1)
                    candidates.append((area, color, top, left, bottom, right))

    assert candidates, "each tile must contain at least one non-background rectangle candidate"
    best_area = max(area for area, *_ in candidates)
    winners = [cand for cand in candidates if cand[0] == best_area]
    winners.sort(key=lambda item: (item[2], item[3], item[4], item[5], item[1]))
    area, color, top, left, bottom, right = winners[0]
    return {
        "area": int(area),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
        "winner_count": int(len(winners)),
    }


def _clean_tile(tile: np.ndarray) -> tuple[np.ndarray, int]:
    rect = _largest_solid_rectangle(tile)
    out = np.zeros_like(tile)
    out[rect["top"] : rect["bottom"] + 1, rect["left"] : rect["right"] + 1] = rect["color"]
    return out, rect["area"]


def _assemble_tiles(
    tiles: list[np.ndarray],
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> np.ndarray:
    h = row_segments[-1][1]
    w = col_segments[-1][1]
    out = np.full((h, w), SEP, dtype=int)
    idx = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            out[r0 : r1, c0 : c1] = tiles[idx]
            idx += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise each tile to its largest solid rectangle, then sort the cleaned tiles by rectangle area.

    Input structure:
    - The grid is split by full grey separator rows and columns into a 3x3 tile array.
    - Each tile has a black background, one intended solid monochrome rectangle, and extra non-background noise pixels around it.

    Rule:
    - In every tile, find the largest axis-aligned filled single-color rectangle and erase all other non-background pixels.
    - Use the preserved rectangle's area as that tile's sorting key.
    - Repack the nine cleaned tiles in row-major order from smallest area at the top-left to largest area at the bottom-right, while keeping the grey separators.

    Key insight:
    - The sort key is not the raw foreground count or the largest connected blob; it is the area of the maximal filled rectangle that remains after geometric denoising.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_segments, col_segments = _parse_layout(g)
    cleaned: list[tuple[int, np.ndarray]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            clean, area = _clean_tile(tile)
            cleaned.append((area, clean))

    ordered_tiles = [tile for _area, tile in sorted(cleaned, key=lambda item: item[0])]
    return _assemble_tiles(ordered_tiles, row_segments, col_segments)


def _factor_pairs(area: int, tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, tile_h + 1):
        if area % h != 0:
            continue
        w = area // h
        if 1 <= w <= tile_w:
            pairs.append((h, w))
    return pairs


def _rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _adjacent_to_rect(pos: tuple[int, int], rect_cells: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in rect_cells:
        if abs(rr - r) + abs(cc - c) == 1:
            return True
    return False


def _largest_component_size(tile: np.ndarray) -> int:
    h, w = tile.shape
    seen = np.zeros((h, w), dtype=bool)
    best = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or tile[r, c] == BG:
                continue
            color = int(tile[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and tile[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            best = max(best, size)
    return best


def _nonzero_bbox_area(tile: np.ndarray) -> int:
    coords = np.argwhere(tile != BG)
    if coords.size == 0:
        return 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _rank_order(values: list[int]) -> list[int]:
    return [idx for idx, _value in sorted(enumerate(values), key=lambda item: (item[1], item[0]))]


def _make_clean_tile(tile_h: int, tile_w: int, area: int, color: int) -> tuple[np.ndarray, dict[str, int]]:
    shapes = _factor_pairs(area, tile_h, tile_w)
    random.shuffle(shapes)
    for rect_h, rect_w in shapes:
        positions = [
            (top, left)
            for top in range(tile_h - rect_h + 1)
            for left in range(tile_w - rect_w + 1)
        ]
        random.shuffle(positions)
        for top, left in positions:
            tile = np.zeros((tile_h, tile_w), dtype=int)
            tile[top : top + rect_h, left : left + rect_w] = color
            rect = _largest_solid_rectangle(tile)
            if rect["winner_count"] == 1 and rect["area"] == area:
                return tile, rect
    raise RuntimeError(f"failed to place clean rectangle of area {area}")


def _choose_noise_colors(base_color: int, count: int, mode: str) -> list[int]:
    foreign = [color for color in RECT_PALETTE if color != base_color]
    if mode == "foreign":
        return [random.choice(foreign) for _ in range(count)]
    colors = [base_color]
    while len(colors) < count:
        if random.random() < 0.5:
            colors.append(base_color)
        else:
            colors.append(random.choice(foreign))
    random.shuffle(colors)
    return colors


def _try_add_noise(
    clean_tile: np.ndarray,
    rect: dict[str, int],
    noise_count: int,
    mode: str,
) -> np.ndarray | None:
    tile_h, tile_w = clean_tile.shape
    rect_box = (rect["top"], rect["left"], rect["bottom"], rect["right"])
    rect_cells = _rect_cells(
        rect["top"],
        rect["left"],
        rect["bottom"] - rect["top"] + 1,
        rect["right"] - rect["left"] + 1,
    )
    outside = [(r, c) for r in range(tile_h) for c in range(tile_w) if (r, c) not in rect_cells]
    adjacent = [pos for pos in outside if _adjacent_to_rect(pos, rect_cells)]
    nonadjacent = [pos for pos in outside if pos not in adjacent]
    if not outside:
        return None
    if mode == "attached_same" and not adjacent:
        return None
    if mode == "same_color" and not nonadjacent:
        return None

    for _ in range(300):
        positions: list[tuple[int, int]] = []
        if mode == "attached_same":
            positions.append(random.choice(adjacent))
        elif mode == "same_color":
            positions.append(random.choice(nonadjacent))

        pool = [pos for pos in outside if pos not in positions]
        if len(pool) + len(positions) < noise_count:
            continue
        random.shuffle(pool)
        positions.extend(pool[: noise_count - len(positions)])
        colors = _choose_noise_colors(rect["color"], noise_count, mode)
        if mode == "foreign" and any(color == rect["color"] for color in colors):
            continue
        if mode == "attached_same":
            colors[0] = rect["color"]
        if mode == "same_color":
            colors[0] = rect["color"]

        tile = clean_tile.copy()
        for (r, c), color in zip(positions, colors):
            tile[r, c] = color

        found = _largest_solid_rectangle(tile)
        if (
            found["winner_count"] == 1
            and found["area"] == rect["area"]
            and found["color"] == rect["color"]
            and (found["top"], found["left"], found["bottom"], found["right"]) == rect_box
        ):
            return tile
    return None


def _make_tile_spec(
    tile_h: int,
    tile_w: int,
    area: int,
    color: int,
    noise_count: int,
    mode: str,
) -> dict[str, object] | None:
    clean_tile, rect = _make_clean_tile(tile_h, tile_w, area, color)
    noisy_tile = _try_add_noise(clean_tile, rect, noise_count, mode)
    if noisy_tile is None:
        return None
    return {
        "area": int(area),
        "color": int(color),
        "mode": mode,
        "noise_count": int(noise_count),
        "clean": clean_tile,
        "tile": noisy_tile,
    }


def _assemble_from_shape(tiles: list[np.ndarray], tile_h: int, tile_w: int) -> np.ndarray:
    row_segments = [(0, tile_h), (tile_h + 1, 2 * tile_h + 1), (2 * tile_h + 2, 3 * tile_h + 2)]
    col_segments = [(0, tile_w), (tile_w + 1, 2 * tile_w + 1), (2 * tile_w + 2, 3 * tile_w + 2)]
    return _assemble_tiles(tiles, row_segments, col_segments)


def _validate_case(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    row_segments, col_segments = _parse_layout(grid)
    tile_areas: list[int] = []
    raw_counts: list[int] = []
    component_sizes: list[int] = []
    bbox_areas: list[int] = []

    extracted_tiles: list[np.ndarray] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = grid[r0:r1, c0:c1]
            extracted_tiles.append(tile)
            rect = _largest_solid_rectangle(tile)
            if rect["winner_count"] != 1:
                return False
            tile_areas.append(rect["area"])
            raw_counts.append(int(np.count_nonzero(tile)))
            component_sizes.append(_largest_component_size(tile))
            bbox_areas.append(_nonzero_bbox_area(tile))

    expected_tiles = [spec["tile"] for spec in specs]
    for actual, expected in zip(extracted_tiles, expected_tiles):
        if not np.array_equal(actual, expected):
            return False

    true_order = _rank_order(tile_areas)
    if true_order != _rank_order([int(spec["area"]) for spec in specs]):
        return False
    if _rank_order(raw_counts) == true_order:
        return False
    if _rank_order(component_sizes) == true_order:
        return False
    if _rank_order(bbox_areas) == true_order:
        return False

    attached_same = sum(spec["mode"] == "attached_same" for spec in specs)
    foreign = sum(spec["mode"] == "foreign" for spec in specs)
    if attached_same < 2 or foreign < 2:
        return False

    expected_output = _assemble_from_shape(
        [spec["clean"] for spec in sorted(specs, key=lambda spec: int(spec["area"]))],
        row_segments[0][1] - row_segments[0][0],
        col_segments[0][1] - col_segments[0][0],
    )
    actual_output = main(grid)
    if not np.array_equal(actual_output, expected_output):
        return False
    if np.array_equal(grid, actual_output):
        return False

    out_tiles: list[np.ndarray] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            out_tiles.append(actual_output[r0:r1, c0:c1])

    out_areas = [_largest_solid_rectangle(tile)["area"] for tile in out_tiles]
    if out_areas != sorted(out_areas):
        return False
    for tile in out_tiles:
        if np.count_nonzero(tile) != _largest_solid_rectangle(tile)["area"]:
            return False

    return True


def _generate_one() -> np.ndarray | None:
    tile_h, tile_w = random.choice(TILE_SHAPES)
    colors = list(RECT_PALETTE) + [random.choice(RECT_PALETTE)]
    random.shuffle(colors)

    noise_pattern = [3, 1, 2, 3, 1, 2, 3, 1, 2]
    random.shuffle(noise_pattern)
    modes = list(MODE_BAG)
    random.shuffle(modes)

    specs: list[dict[str, object]] = []
    for area, color, noise_count, mode in zip(AREA_VALUES, colors, noise_pattern, modes):
        spec = _make_tile_spec(tile_h, tile_w, area, color, noise_count, mode)
        if spec is None:
            return None
        specs.append(spec)

    random.shuffle(specs)
    tiles = [spec["tile"] for spec in specs]
    grid = _assemble_from_shape(tiles, tile_h, tile_w)
    return grid if _validate_case(grid, specs) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy-rectangle sorting input")
