from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            out.append(np.array(cells, dtype=int))

    return out


def _count_extra_holes(component: np.ndarray) -> int:
    r0 = int(component[:, 0].min())
    c0 = int(component[:, 1].min())
    r1 = int(component[:, 0].max())
    c1 = int(component[:, 1].max())

    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    crop[component[:, 0] - r0, component[:, 1] - c0] = True
    bg = ~crop

    enclosed = 0
    for region in _components(bg):
        touches_edge = bool(
            np.any(region[:, 0] == 0)
            or np.any(region[:, 0] == crop.shape[0] - 1)
            or np.any(region[:, 1] == 0)
            or np.any(region[:, 1] == crop.shape[1] - 1)
        )
        if not touches_edge:
            enclosed += 1

    assert enclosed >= 1, "each region must have a main enclosed interior"
    return int(enclosed - 1)


def _legend_row_and_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h = int(grid.shape[0])
    candidates: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] != BACKGROUND)
        if len(cols) < 2:
            continue
        if not np.all(np.diff(cols) >= 2):
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        candidates.append(r)

    assert candidates, "legend row not found"
    legend_r = max(candidates)
    palette = [int(grid[legend_r, c]) for c in np.flatnonzero(grid[legend_r] != BACKGROUND)]
    assert palette, "palette must be non-empty"
    return legend_r, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed region by indexing the bottom legend with its extra-hole count.

    The foreground object is a single color shared by all regions. Each region contains one
    main enclosed black interior plus zero or more additional black holes inside it. The
    hole count selects the legend color at the same zero-based index. The legend row itself
    is removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    legend_r, palette = _legend_row_and_palette(g)

    work = g.copy()
    work[legend_r, :] = BACKGROUND

    colors = sorted(set(map(int, np.unique(work))) - {BACKGROUND})
    assert len(colors) == 1, "expected exactly one non-background drawing color"
    object_color = int(colors[0])

    out = work.copy()
    for component in _components(work == object_color):
        hole_count = _count_extra_holes(component)
        assert hole_count < len(palette), "legend is too short for the region hole count"
        out[component[:, 0], component[:, 1]] = int(palette[hole_count])

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (20, 29)
GRID_WIDTH_RANGE = (20, 30)
LEGEND_COLORS = 4
LEGEND_GAP_RANGE = (2, 3)
OBJECT_GAP = 2
OUTER_MARGIN = 1
TOP_ROWS_OPTIONS = (1, 2)

REGION_BUILD_RETRIES = 400
REGION_HOLE_COUNTS = (0, 1, 2, 3)
OUTER_WALL_OPTIONS = (2, 3)
INNER_LOOP_H_RANGE = (3, 5)
INNER_LOOP_W_RANGE = (3, 6)
OUTER_CAVITY_H_BASE = (5, 7)
OUTER_CAVITY_W_BASE = (6, 9)
NOTCH_COUNT_RANGE = (0, 3)
NOTCH_SIZE_RANGE = (2, 4)
SPUR_COUNT_RANGE = (0, 3)
SPUR_MIN_LENGTH = 1

PACK_RETRIES = 200
GENERATE_RETRIES = 400


def _draw_frame(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _carve_notches(mask: np.ndarray, wall: int) -> None:
    h, w = map(int, mask.shape)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)

    count = random.randint(*NOTCH_COUNT_RANGE)
    for side in sides[:count]:
        if side in ("top", "bottom") and w >= 8:
            width = random.randint(*NOTCH_SIZE_RANGE)
            width = min(width, w - 4)
            start = random.randint(1, w - width - 1)
            row = random.randint(0, wall - 2)
            mask[row if side == "top" else h - 1 - row, start : start + width] = False
        elif side in ("left", "right") and h >= 8:
            height = random.randint(*NOTCH_SIZE_RANGE)
            height = min(height, h - 4)
            start = random.randint(1, h - height - 1)
            col = random.randint(0, wall - 2)
            mask[start : start + height, col if side == "left" else w - 1 - col] = False


def _add_spurs(mask: np.ndarray, wall: int) -> None:
    h, w = map(int, mask.shape)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)

    for _ in range(n_spurs):
        placed = False
        for _attempt in range(80):
            side = random.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                if w - 2 * wall < 3:
                    continue
                c = random.randint(wall + 1, w - wall - 2)
                if side == "top":
                    occupied = np.flatnonzero(mask[wall:, c])
                    if len(occupied) == 0:
                        continue
                    gap = int(occupied[0])
                    if gap <= 1:
                        continue
                    length = random.randint(SPUR_MIN_LENGTH, gap - 1)
                    mask[wall : wall + length, c] = True
                else:
                    occupied = np.flatnonzero(mask[: h - wall, c][::-1])
                    if len(occupied) == 0:
                        continue
                    gap = int(occupied[0])
                    if gap <= 1:
                        continue
                    length = random.randint(SPUR_MIN_LENGTH, gap - 1)
                    mask[h - wall - length : h - wall, c] = True
            else:
                if h - 2 * wall < 3:
                    continue
                r = random.randint(wall + 1, h - wall - 2)
                if side == "left":
                    occupied = np.flatnonzero(mask[r, wall:])
                    if len(occupied) == 0:
                        continue
                    gap = int(occupied[0])
                    if gap <= 1:
                        continue
                    length = random.randint(SPUR_MIN_LENGTH, gap - 1)
                    mask[r, wall : wall + length] = True
                else:
                    occupied = np.flatnonzero(mask[r, : w - wall][::-1])
                    if len(occupied) == 0:
                        continue
                    gap = int(occupied[0])
                    if gap <= 1:
                        continue
                    length = random.randint(SPUR_MIN_LENGTH, gap - 1)
                    mask[r, w - wall - length : w - wall] = True

            placed = True
            break
        if not placed:
            continue


def _build_region(extra_holes: int) -> np.ndarray | None:
    for _ in range(REGION_BUILD_RETRIES):
        wall = random.choice(OUTER_WALL_OPTIONS)
        inner_h = random.randint(OUTER_CAVITY_H_BASE[0] + extra_holes, OUTER_CAVITY_H_BASE[1] + 2 * extra_holes)
        inner_w = random.randint(OUTER_CAVITY_W_BASE[0] + 2 * extra_holes, OUTER_CAVITY_W_BASE[1] + 3 * extra_holes)
        h = inner_h + 2 * wall
        w = inner_w + 2 * wall

        mask = np.zeros((h, w), dtype=bool)
        mask[:wall, :] = True
        mask[h - wall :, :] = True
        mask[:, :wall] = True
        mask[:, w - wall :] = True
        _carve_notches(mask, wall)

        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for _hole in range(extra_holes):
            placed = False
            for _attempt in range(120):
                loop_h = random.randint(*INNER_LOOP_H_RANGE)
                loop_w = random.randint(*INNER_LOOP_W_RANGE)
                if h - 2 * wall - loop_h < 3 or w - 2 * wall - loop_w < 3:
                    continue

                r0 = random.randint(wall + 1, h - wall - loop_h - 1)
                c0 = random.randint(wall + 1, w - wall - loop_w - 1)
                r1 = r0 + loop_h - 1
                c1 = c0 + loop_w - 1
                box = (r0, c0, r1, c1)

                if any(not _boxes_separated(box, other, pad=1) for other in boxes):
                    continue

                bridges: list[tuple[str, int]] = []
                for sc in range(c0 + 1, c1):
                    if not np.any(mask[wall:r0, sc]):
                        bridges.append(("top", sc))
                    if not np.any(mask[r1 + 1 : h - wall, sc]):
                        bridges.append(("bottom", sc))
                for sr in range(r0 + 1, r1):
                    if not np.any(mask[sr, wall:c0]):
                        bridges.append(("left", sr))
                    if not np.any(mask[sr, c1 + 1 : w - wall]):
                        bridges.append(("right", sr))

                if not bridges:
                    continue

                _draw_frame(mask, r0, c0, r1, c1)
                side, pos = random.choice(bridges)
                if side == "top":
                    mask[wall - 1 : r0 + 1, pos] = True
                elif side == "bottom":
                    mask[r1 : h - wall + 1, pos] = True
                elif side == "left":
                    mask[pos, wall - 1 : c0 + 1] = True
                else:
                    mask[pos, c1 : w - wall + 1] = True

                boxes.append(box)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        _add_spurs(mask, wall)
        components = _components(mask)
        if len(components) != 1:
            continue
        if _count_extra_holes(components[0]) != extra_holes:
            continue
        return mask.astype(int)

    return None


def _split_total(total: int, parts: int) -> list[int]:
    slots = [0] * parts
    for _ in range(total):
        slots[random.randrange(parts)] += 1
    return slots


def _pack_layout(tiles: list[np.ndarray], legend_gap: int, legend_width: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    sizes = [(int(tile.shape[0]), int(tile.shape[1])) for tile in tiles]

    for _ in range(PACK_RETRIES):
        row_count = random.choice(TOP_ROWS_OPTIONS)
        row_count = min(row_count, len(tiles))
        assignment = [random.randrange(row_count) for _ in tiles]
        if len(set(assignment)) != row_count:
            continue

        rows: list[list[int]] = [[] for _ in range(row_count)]
        for idx, row_id in enumerate(assignment):
            rows[row_id].append(idx)
        for row in rows:
            random.shuffle(row)

        row_widths = []
        row_heights = []
        for row in rows:
            widths = [sizes[idx][1] for idx in row]
            heights = [sizes[idx][0] for idx in row]
            row_widths.append(sum(widths) + OBJECT_GAP * max(0, len(row) - 1))
            row_heights.append(max(heights))

        packed_w = max(row_widths)
        packed_h = sum(row_heights) + OBJECT_GAP * max(0, row_count - 1)

        min_w = max(legend_width + 2 * OUTER_MARGIN, packed_w + 2 * OUTER_MARGIN)
        min_h = packed_h + legend_gap + 1 + 2 * OUTER_MARGIN
        if min_w > GRID_WIDTH_RANGE[1] or min_h > GRID_HEIGHT_RANGE[1]:
            continue

        grid_w = random.randint(min_w, GRID_WIDTH_RANGE[1])
        grid_h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
        top_h = grid_h - legend_gap - 1
        if packed_h > top_h - 2 * OUTER_MARGIN:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        placements: list[tuple[int, int]] = [(-1, -1)] * len(tiles)

        extra_y = top_h - 2 * OUTER_MARGIN - packed_h
        vertical_parts = _split_total(extra_y, row_count + 1)
        y = OUTER_MARGIN + vertical_parts[0]
        for row_idx, row in enumerate(rows):
            row_w = row_widths[row_idx]
            row_h = row_heights[row_idx]
            extra_x = grid_w - 2 * OUTER_MARGIN - row_w
            horizontal_parts = _split_total(extra_x, len(row) + 1)
            x = OUTER_MARGIN + horizontal_parts[0]
            for tile_pos, idx in enumerate(row):
                tile = tiles[idx]
                th, tw = map(int, tile.shape)
                yoff = random.randint(0, row_h - th)
                grid[y + yoff : y + yoff + th, x : x + tw] = tile
                placements[idx] = (y + yoff, x)
                x += tw
                if tile_pos + 1 < len(row):
                    x += OBJECT_GAP + horizontal_parts[tile_pos + 1]
            y += row_h
            if row_idx + 1 < row_count:
                y += OBJECT_GAP + vertical_parts[row_idx + 1]

        return grid, placements

    return None


def _sample_tiles() -> list[np.ndarray] | None:
    count = random.choice([2, 3])
    hole_counts = random.sample(list(REGION_HOLE_COUNTS), count)
    tiles: list[np.ndarray] = []
    for hole_count in hole_counts:
        tile = _build_region(hole_count)
        if tile is None:
            return None
        tiles.append(tile)
    return tiles


def _gen_attempt() -> np.ndarray | None:
    palette_colors = random.sample(list(range(1, 10)), LEGEND_COLORS + 1)
    object_color = int(palette_colors[0])
    legend = [int(c) for c in palette_colors[1:]]

    tiles = _sample_tiles()
    if tiles is None:
        return None
    tiles = [tile * object_color for tile in tiles]

    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    legend_width = 2 * len(legend) - 1
    packed = _pack_layout(tiles, legend_gap, legend_width)
    if packed is None:
        return None
    grid, _placements = packed

    legend_r = grid.shape[0] - 1
    start = random.randint(0, grid.shape[1] - legend_width)
    for i, color in enumerate(legend):
        grid[legend_r, start + 2 * i] = int(color)

    body = grid.copy()
    body[legend_r, :] = BACKGROUND
    colors = sorted(set(map(int, np.unique(body))) - {BACKGROUND})
    if colors != [object_color]:
        return None

    comps = _components(body == object_color)
    if len(comps) != len(tiles):
        return None
    counts = sorted(_count_extra_holes(comp) for comp in comps)
    if any(count >= len(legend) for count in counts):
        return None
    if len(set(counts)) != len(counts):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
