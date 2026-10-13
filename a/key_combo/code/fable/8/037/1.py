from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, neighbors4, paste


MAX_GENERATION_ATTEMPTS = 200


def _tile_bfs(rows: int, cols: int, blocked: set[tuple[int, int]], seed: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {seed: 0}
    q: deque[tuple[int, int]] = deque([seed])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            nxt = (nr, nc)
            if nxt in blocked or nxt in dist:
                continue
            dist[nxt] = dist[(r, c)] + 1
            q.append(nxt)
    return dist


def _locate_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    sprite_candidates = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(sprite_candidates) == 1, "expected exactly one non-singleton sprite component"

    sprite_info = sprite_candidates[0]
    r0, c0, r1, c1 = map(int, sprite_info["bbox"])
    sprite = crop_bbox(grid, (r0, c0, r1, c1))
    assert len(colors_present(sprite, exclude={bg})) >= 2, "sprite must be multicolor"
    h, w = sprite.shape
    H, W = grid.shape
    assert H % h == 0 and W % w == 0, "grid must be tiled by the sprite bbox"
    assert r0 % h == 0 and c0 % w == 0, "sprite bbox must align to the tile lattice"
    return sprite, (r0, c0, r1, c1), (r0 // h, c0 // w)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    sprite, sprite_bbox, seed_tile = _locate_sprite(g)
    tile_h, tile_w = sprite.shape
    rows, cols = g.shape[0] // tile_h, g.shape[1] // tile_w

    blocked: set[tuple[int, int]] = set()
    for tr in range(rows):
        for tc in range(cols):
            if (tr, tc) == seed_tile:
                continue
            tile = g[tr * tile_h : (tr + 1) * tile_h, tc * tile_w : (tc + 1) * tile_w]
            if np.any(tile != bg):
                blocked.add((tr, tc))

    reachable = _tile_bfs(rows, cols, blocked, seed_tile)
    out = g.copy()
    for tr, tc in reachable:
        if (tr, tc) == seed_tile:
            continue
        top = tr * tile_h
        left = tc * tile_w
        tile = g[top : top + tile_h, left : left + tile_w]
        assert np.all(tile == bg), "reachable placement tiles must be clean in the input"
        paste(out, sprite, top, left)

    r0, c0, r1, c1 = sprite_bbox
    assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], sprite)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the unique multicolor sprite flood-copy across the hidden tile lattice, but only through blocker-free footprints.

    Input structure:
    - The grid dimensions are exact multiples of the sprite's bounding-box height and width, so the sprite bbox defines an implicit rectangular tile lattice.
    - Exactly one multicolor sprite occupies one full tile position, and its internal background holes are part of the sprite crop.
    - Other occupied cells are isolated single-pixel blockers in different colors from both the background and the sprite.

    Rule:
    - Use the sprite bbox as one lattice tile and treat the sprite tile as the flood-fill seed.
    - On the tile lattice, expand only through 4-neighbor tiles whose entire footprint is clean in the input.
    - Paste an unchanged copy of the sprite into every reached clean tile; any tile containing at least one blocker stays unchanged and also blocks further propagation behind it.

    Key insight:
    - The flood happens in tile space, not pixel space: one blocker pixel invalidates its whole tile footprint, so clean tiles cut off behind blocker tiles must remain empty.
    """
    return _solve(grid)


def _connect_path(
    rng: np.random.Generator,
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> None:
    r, c = start
    tr, tc = target
    cells.add((r, c))
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((1 if tr > r else -1, 0))
        if c != tc:
            moves.append((0, 1 if tc > c else -1))
        dr, dc = moves[int(rng.integers(len(moves)))]
        r += dr
        c += dc
        cells.add((r, c))


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == len(points)


def _random_sprite_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    anchors = [
        (int(rng.integers(h)), int(rng.integers(w))),
        (0, int(rng.integers(w))),
        (h - 1, int(rng.integers(w))),
        (int(rng.integers(h)), 0),
        (int(rng.integers(h)), w - 1),
    ]
    cells = {anchors[0]}
    for target in anchors[1:]:
        start = min(cells, key=lambda rc: abs(rc[0] - target[0]) + abs(rc[1] - target[1]))
        _connect_path(rng, cells, start, target)

    min_area = max(len(cells), (h * w * 2) // 5)
    max_area = max(min_area, (h * w * 3) // 4)
    target_area = int(rng.integers(min_area, max_area + 1))

    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if h >= 4 and w >= 4 and rng.random() < 0.5:
        removable = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not mask[r, c]:
                    continue
                if not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if _mask_connected(trial):
                    removable.append((r, c))
        if removable:
            rr, cc = removable[int(rng.integers(len(removable)))]
            mask[rr, cc] = False

    assert _mask_connected(mask)
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    return mask


def _paint_sprite(
    rng: np.random.Generator,
    mask: np.ndarray,
    bg: int,
    palette: list[int],
) -> np.ndarray:
    coords = np.argwhere(mask)
    max_colors = min(4, len(coords) // 2)
    assert max_colors >= 2
    for _ in range(50):
        color_count = int(rng.integers(2, max_colors + 1))
        colors = palette[:color_count]
        seed_ix = rng.choice(len(coords), size=color_count, replace=False)
        seed_coords = coords[seed_ix]
        sprite = np.full(mask.shape, bg, dtype=int)
        for r, c in coords:
            dists = np.abs(seed_coords - np.array([r, c])).sum(axis=1)
            color_idx = int(np.argmin(dists))
            sprite[r, c] = colors[color_idx]
        counts = [int(np.sum(sprite == color)) for color in colors]
        if min(counts) >= 2:
            return sprite
    raise RuntimeError("failed to paint a multicolor sprite")


def _generate_sprite(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    available_colors: list[int],
) -> tuple[np.ndarray, list[int]]:
    assert len(available_colors) >= 4
    palette = list(rng.permutation(available_colors))
    mask = _random_sprite_mask(rng, h, w)
    sprite = _paint_sprite(rng, mask, bg, palette)
    sprite_colors = colors_present(sprite, exclude={bg})
    return sprite, sprite_colors


def _choose_layout(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    all_tiles = {(r, c) for r in range(rows) for c in range(cols)}
    for _ in range(100):
        if rng.random() < 0.5:
            wall_col = int(rng.integers(1, cols - 1))
            base_blocked = {(r, wall_col) for r in range(rows)}
            side_options = []
            left_count = rows * wall_col
            right_count = rows * (cols - wall_col - 1)
            if left_count >= 3 and right_count >= 1:
                side_options.append("left")
            if right_count >= 3 and left_count >= 1:
                side_options.append("right")
            if not side_options:
                continue
            side = side_options[int(rng.integers(len(side_options)))]
            if side == "left":
                seed_pool = [(r, c) for r in range(rows) for c in range(wall_col)]
            else:
                seed_pool = [(r, c) for r in range(rows) for c in range(wall_col + 1, cols)]
        else:
            wall_row = int(rng.integers(1, rows - 1))
            base_blocked = {(wall_row, c) for c in range(cols)}
            side_options = []
            top_count = wall_row * cols
            bottom_count = (rows - wall_row - 1) * cols
            if top_count >= 3 and bottom_count >= 1:
                side_options.append("top")
            if bottom_count >= 3 and top_count >= 1:
                side_options.append("bottom")
            if not side_options:
                continue
            side = side_options[int(rng.integers(len(side_options)))]
            if side == "top":
                seed_pool = [(r, c) for r in range(wall_row) for c in range(cols)]
            else:
                seed_pool = [(r, c) for r in range(wall_row + 1, rows) for c in range(cols)]

        seed = seed_pool[int(rng.integers(len(seed_pool)))]
        candidates = [tile for tile in all_tiles if tile not in base_blocked and tile != seed]
        extra_cap = min(len(candidates), rows + cols - 3)
        extra_count = int(rng.integers(0, extra_cap + 1))
        extra_tiles = set()
        if extra_count:
            chosen = rng.choice(len(candidates), size=extra_count, replace=False)
            extra_tiles = {candidates[int(idx)] for idx in np.atleast_1d(chosen)}

        blocked = base_blocked | extra_tiles
        if seed in blocked:
            continue

        reachable = set(_tile_bfs(rows, cols, blocked, seed))
        open_tiles = all_tiles - blocked
        unreachable = open_tiles - reachable
        if len(reachable) < 3:
            continue
        if not unreachable:
            continue
        if max(_tile_bfs(rows, cols, blocked, seed).values()) < 2:
            continue
        return seed, blocked, reachable
    raise RuntimeError("failed to choose a non-degenerate lattice layout")


def _choose_blocker_spots(
    rng: np.random.Generator,
    h: int,
    w: int,
    count: int,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(candidates)
    spots: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in spots):
            spots.append((r, c))
        if len(spots) == count:
            break
    return spots


def _build_input(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray | None:
    bg = int(rng.integers(10))
    all_colors = [color for color in range(10) if color != bg]
    rng.shuffle(all_colors)

    sprite, sprite_colors = _generate_sprite(rng, tile_h, tile_w, bg, all_colors)
    blocker_palette = [color for color in all_colors if color not in sprite_colors]
    if not blocker_palette:
        return None

    seed_tile, blocked_tiles, reachable = _choose_layout(rng, rows, cols)
    grid = np.full((rows * tile_h, cols * tile_w), bg, dtype=int)
    paste(grid, sprite, seed_tile[0] * tile_h, seed_tile[1] * tile_w)

    for tr, tc in blocked_tiles:
        top = tr * tile_h
        left = tc * tile_w
        blocker_count = 1
        if tile_h >= 5 and tile_w >= 5 and rng.random() < 0.35:
            blocker_count = 2
        spots = _choose_blocker_spots(rng, tile_h, tile_w, blocker_count)
        if len(spots) != blocker_count:
            return None
        for rr, cc in spots:
            grid[top + rr, left + cc] = blocker_palette[int(rng.integers(len(blocker_palette)))]

    expected = grid.copy()
    for tr, tc in reachable:
        if (tr, tc) == seed_tile:
            continue
        paste(expected, sprite, tr * tile_h, tc * tile_w)

    if np.array_equal(expected, grid):
        return None

    all_clean = {(r, c) for r in range(rows) for c in range(cols)} - blocked_tiles
    if reachable == all_clean:
        return None
    if len(all_clean - reachable) == 0:
        return None

    dist = _tile_bfs(rows, cols, blocked_tiles, seed_tile)
    if max(dist.values()) < 2:
        return None

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    choices = []
    for tile_h in range(3, 7):
        for tile_w in range(3, 7):
            max_rows = min(6, 30 // tile_h)
            max_cols = min(6, 30 // tile_w)
            for rows in range(4, max_rows + 1):
                for cols in range(4, max_cols + 1):
                    choices.append((rows, cols, tile_h, tile_w))
    rows, cols, tile_h, tile_w = choices[int(rng.integers(len(choices)))]
    return _build_input(rng, rows, cols, tile_h, tile_w)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tile-flood input")
