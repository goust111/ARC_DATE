from __future__ import annotations

import random

import numpy as np


WALL_COLOR = 5
ARROW_COLOR = 2
PALETTE = tuple(color for color in range(10) if color not in (WALL_COLOR, ARROW_COLOR))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
    )


def _most_common_color(grid: np.ndarray, *, exclude: set[int]) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best_color = None
    best_count = -1
    for color, count in zip(values, counts):
        color = int(color)
        count = int(count)
        if color in exclude:
            continue
        if count > best_count:
            best_color = color
            best_count = count
    assert best_color is not None
    return best_color


def _decode_arrow(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    local = coords - np.array([r0, c0], dtype=int)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    if h == 2 and w >= 2:
        row_counts = [int(np.count_nonzero(local[:, 0] == row)) for row in range(2)]
        assert sorted(row_counts) == [1, w], "invalid horizontal L arrow"
        short_row = 0 if row_counts[0] == 1 else 1
        short_cols = local[local[:, 0] == short_row][:, 1]
        assert short_cols.size == 1
        tip_col = int(short_cols[0])
        if tip_col == 0:
            return (0, -1)
        assert tip_col == w - 1
        return (0, 1)

    if w == 2 and h >= 2:
        col_counts = [int(np.count_nonzero(local[:, 1] == col)) for col in range(2)]
        assert sorted(col_counts) == [1, h], "invalid vertical L arrow"
        short_col = 0 if col_counts[0] == 1 else 1
        short_rows = local[local[:, 1] == short_col][:, 0]
        assert short_rows.size == 1
        tip_row = int(short_rows[0])
        if tip_row == 0:
            return (-1, 0)
        assert tip_row == h - 1
        return (1, 0)

    raise AssertionError("arrow must be an axis-aligned L shape")


def _stamp_tile(out: np.ndarray, tile: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = tile != bg
    region = out[top : top + tile.shape[0], left : left + tile.shape[1]]
    region[mask] = tile[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the seed tile through each enclosed room in the direction encoded by the red L arrow.

    Input structure:
    - Grey one-cell walls form closed rooms on a background.
    - Each room contains one small seed pattern and one red L-shaped arrow.
    - The seed pattern's tight bounding box is the tile that repeats.

    Rule:
    - Find each enclosed non-wall region.
    - Decode the red L arrow: the little spur marks which end of the long segment
      is the arrow tip, so it determines left/right/up/down.
    - Repeatedly copy the seed tile's bounding box along that direction until the
      next full tile would leave the room.
    - Erase the arrow back to background and keep the walls unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == WALL_COLOR), "expected grey walls"
    assert np.any(g == ARROW_COLOR), "expected red arrows"

    bg = _most_common_color(g, exclude={WALL_COLOR})
    h, w = g.shape
    out = g.copy()

    for region_coords in _components(g != WALL_COLOR):
        if _touches_border(region_coords, h, w):
            continue

        region_mask = np.zeros((h, w), dtype=bool)
        region_mask[region_coords[:, 0], region_coords[:, 1]] = True

        arrow_components = _components(region_mask & (g == ARROW_COLOR))
        assert len(arrow_components) == 1, "each room must contain exactly one arrow"
        arrow_coords = arrow_components[0]
        dr, dc = _decode_arrow(arrow_coords)

        seed_mask = region_mask & (g != bg) & (g != WALL_COLOR) & (g != ARROW_COLOR)
        seed_components = _components(seed_mask)
        assert len(seed_components) == 1, "each room must contain exactly one seed pattern"
        seed_coords = seed_components[0]
        sr0, sc0, sr1, sc1 = _bbox(seed_coords)
        tile = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        tile[(tile == WALL_COLOR) | (tile == ARROW_COLOR)] = bg
        tile_h, tile_w = tile.shape
        step_r = dr * tile_h
        step_c = dc * tile_w

        out[arrow_coords[:, 0], arrow_coords[:, 1]] = bg
        pos_r, pos_c = sr0, sc0
        while True:
            _stamp_tile(out, tile, pos_r, pos_c, bg)
            next_r = pos_r + step_r
            next_c = pos_c + step_c
            if not (0 <= next_r and 0 <= next_c and next_r + tile_h <= h and next_c + tile_w <= w):
                break
            if not np.all(region_mask[next_r : next_r + tile_h, next_c : next_c + tile_w]):
                break
            pos_r, pos_c = next_r, next_c

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _generate_seed_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    min_area = max(2, (height * width + 1) // 2)
    target_area = random.randint(min_area, height * width)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    while int(mask.sum()) > target_area and cells:
        r, c = cells.pop()
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
            continue
        if not _mask_connected(trial):
            continue
        mask = trial
    return mask


def _make_arrow(direction: str, long_len: int) -> np.ndarray:
    assert long_len >= 2
    if direction in ("left", "right"):
        arrow = np.full((2, long_len), -1, dtype=int)
        long_row = random.randint(0, 1)
        short_row = 1 - long_row
        arrow[long_row, :] = ARROW_COLOR
        tip_col = 0 if direction == "left" else long_len - 1
        arrow[short_row, tip_col] = ARROW_COLOR
        arrow[arrow == -1] = 0
        return arrow

    arrow = np.full((long_len, 2), -1, dtype=int)
    long_col = random.randint(0, 1)
    short_col = 1 - long_col
    arrow[:, long_col] = ARROW_COLOR
    tip_row = 0 if direction == "up" else long_len - 1
    arrow[tip_row, short_col] = ARROW_COLOR
    arrow[arrow == -1] = 0
    return arrow


def _seed_origin(direction: str, interior_h: int, interior_w: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    if direction == "right":
        return 0, 0
    if direction == "left":
        return 0, interior_w - tile_w
    if direction == "down":
        return 0, 0
    assert direction == "up"
    return interior_h - tile_h, 0


def _arrow_origin(
    direction: str,
    interior_h: int,
    interior_w: int,
    tile_h: int,
    tile_w: int,
    arrow_h: int,
    arrow_w: int,
) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []

    if direction == "right":
        for top in range(0, interior_h - arrow_h + 1):
            for left in range(tile_w, interior_w - arrow_w + 1):
                options.append((top, left))
    elif direction == "left":
        for top in range(0, interior_h - arrow_h + 1):
            for left in range(0, interior_w - tile_w - arrow_w + 1):
                options.append((top, left))
    elif direction == "down":
        for top in range(tile_h, interior_h - arrow_h + 1):
            for left in range(0, interior_w - arrow_w + 1):
                options.append((top, left))
    else:
        for top in range(0, interior_h - tile_h - arrow_h + 1):
            for left in range(0, interior_w - arrow_w + 1):
                options.append((top, left))

    if not options:
        return None
    return random.choice(options)


def _build_region_patch(bg: int, seed_color: int, direction: str) -> np.ndarray | None:
    tile_h = random.randint(2, 5)
    tile_w = random.randint(2, 5)
    tile_mask = _generate_seed_mask(tile_h, tile_w)

    if direction in ("left", "right"):
        n_tiles = random.randint(2, 5)
        interior_h = tile_h
        interior_w = tile_w * n_tiles
        free_len = interior_w - tile_w
        long_len = random.randint(2, min(4, free_len))
    else:
        n_tiles = random.randint(2, 5)
        interior_h = tile_h * n_tiles
        interior_w = tile_w
        free_len = interior_h - tile_h
        long_len = random.randint(2, min(4, free_len))

    arrow = _make_arrow(direction, long_len)
    arrow_h, arrow_w = arrow.shape
    arrow_pos = _arrow_origin(direction, interior_h, interior_w, tile_h, tile_w, arrow_h, arrow_w)
    if arrow_pos is None:
        return None

    extra_top = random.randint(0, 2)
    extra_bottom = random.randint(0, 2)
    extra_left = random.randint(0, 2)
    extra_right = random.randint(0, 2)

    base_h = interior_h + 2
    base_w = interior_w + 2
    patch_h = base_h + extra_top + extra_bottom
    patch_w = base_w + extra_left + extra_right
    patch = np.full((patch_h, patch_w), bg, dtype=int)

    wr0 = extra_top
    wc0 = extra_left
    wr1 = wr0 + base_h - 1
    wc1 = wc0 + base_w - 1
    patch[wr0, wc0 : wc1 + 1] = WALL_COLOR
    patch[wr1, wc0 : wc1 + 1] = WALL_COLOR
    patch[wr0 : wr1 + 1, wc0] = WALL_COLOR
    patch[wr0 : wr1 + 1, wc1] = WALL_COLOR

    if extra_top > 0 and base_w >= 3 and random.random() < 0.7:
        col = random.randint(wc0 + 1, wc1 - 1)
        length = random.randint(1, extra_top)
        patch[wr0 - length : wr0, col] = WALL_COLOR
    if extra_bottom > 0 and base_w >= 3 and random.random() < 0.7:
        col = random.randint(wc0 + 1, wc1 - 1)
        length = random.randint(1, extra_bottom)
        patch[wr1 + 1 : wr1 + length + 1, col] = WALL_COLOR
    if extra_left > 0 and base_h >= 3 and random.random() < 0.7:
        row = random.randint(wr0 + 1, wr1 - 1)
        length = random.randint(1, extra_left)
        patch[row, wc0 - length : wc0] = WALL_COLOR
    if extra_right > 0 and base_h >= 3 and random.random() < 0.7:
        row = random.randint(wr0 + 1, wr1 - 1)
        length = random.randint(1, extra_right)
        patch[row, wc1 + 1 : wc1 + length + 1] = WALL_COLOR

    ir0 = wr0 + 1
    ic0 = wc0 + 1
    seed_r, seed_c = _seed_origin(direction, interior_h, interior_w, tile_h, tile_w)
    seed_region = patch[ir0 + seed_r : ir0 + seed_r + tile_h, ic0 + seed_c : ic0 + seed_c + tile_w]
    seed_region[tile_mask] = seed_color

    arrow_r, arrow_c = arrow_pos
    arrow_region = patch[ir0 + arrow_r : ir0 + arrow_r + arrow_h, ic0 + arrow_c : ic0 + arrow_c + arrow_w]
    arrow_region[arrow == ARROW_COLOR] = ARROW_COLOR
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, patch: np.ndarray, bg: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph >= h or left + pw >= w:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    candidates = [
        (top, left)
        for top in range(1, gh - ph)
        for left in range(1, gw - pw)
        if _can_place(occupied, top, left, patch, bg)
    ]
    if not candidates:
        return False

    top, left = random.choice(candidates)
    mask = patch != bg
    region = grid[top : top + ph, left : left + pw]
    region[mask] = patch[mask]
    occupied[top : top + ph, left : left + pw] |= mask
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    bg = random.choice(PALETTE)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    region_count = random.choices([1, 2, 3], weights=[2, 4, 2])[0]
    seed_colors = random.sample([c for c in PALETTE if c != bg], region_count)
    directions = [random.choice(("left", "right", "up", "down")) for _ in range(region_count)]
    random.shuffle(directions)

    for seed_color, direction in zip(seed_colors, directions):
        patch = None
        for _ in range(PLACEMENT_RETRIES):
            patch = _build_region_patch(bg, seed_color, direction)
            if patch is not None:
                break
        if patch is None:
            return None
        if not _place_patch(grid, occupied, patch, bg):
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
