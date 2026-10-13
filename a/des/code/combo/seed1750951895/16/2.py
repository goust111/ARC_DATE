import random
from typing import List, Tuple

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SQUARE_SIZE = 5
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_solid_square(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    if coords.shape[0] != SQUARE_SIZE * SQUARE_SIZE:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != SQUARE_SIZE or c1 - c0 + 1 != SQUARE_SIZE:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _centered_sprite_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = True
    sh, sw = sprite.shape
    assert sh <= SQUARE_SIZE and sw <= SQUARE_SIZE
    mask = np.zeros((SQUARE_SIZE, SQUARE_SIZE), dtype=bool)
    top = (SQUARE_SIZE - sh) // 2
    left = (SQUARE_SIZE - sw) // 2
    mask[top : top + sh, left : left + sw] = sprite
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each monochrome 5x5 solid square with its same-color sprite, carve the
    centered sprite shape out of the square, then output the carved squares in a
    single left-to-right row with a 1-pixel black border and 1-pixel gaps.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors

    carved_blocks: List[Tuple[int, int, np.ndarray]] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) >= 2

        square = None
        sprites = []
        for comp in comps:
            if _is_solid_square(g, color, comp):
                assert square is None
                square = comp
            else:
                sprites.append(comp)

        assert square is not None
        assert len(sprites) == 1

        sr0, sc0, _sr1, _sc1 = _bbox(square)
        carve = np.full((SQUARE_SIZE, SQUARE_SIZE), color, dtype=int)
        carve[_centered_sprite_mask(sprites[0])] = BG
        carved_blocks.append((sr0, sc0, carve))

    carved_blocks.sort(key=lambda item: (item[0], item[1]))
    n = len(carved_blocks)
    out = np.zeros((SQUARE_SIZE + 2, n * SQUARE_SIZE + (n - 1) + 2), dtype=int)
    for i, (_r, _c, block) in enumerate(carved_blocks):
        left = 1 + i * (SQUARE_SIZE + 1)
        out[1 : 1 + SQUARE_SIZE, left : left + SQUARE_SIZE] = block
    return out


def _random_sprite_shape(height: int, width: int, area: int) -> np.ndarray:
    total = height * width
    area = max(1, min(area, total))
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = [start]
    frontier = [start]
    while len(cells) < area:
        if not frontier:
            frontier = cells[:]
        rr, cc = random.choice(frontier)
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier = [p for p in frontier if p != (rr, cc)]
            continue
        nxt = random.choice(nbrs)
        mask[nxt] = True
        cells.append(nxt)
        frontier.append(nxt)

    # Trim empty outer rows/cols while keeping connectedness and non-rectangular variety.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.where(rows)[0][0] : np.where(rows)[0][-1] + 1, np.where(cols)[0][0] : np.where(cols)[0][-1] + 1]


def _dilated(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = True
    return out


def _can_place(occupied: np.ndarray, top: int, left: int, shape: np.ndarray) -> bool:
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > occupied.shape[0] or left + w > occupied.shape[1]:
        return False
    region = occupied[top : top + h, left : left + w]
    return not bool(np.any(region & _dilated(shape)))


def _paint(grid: np.ndarray, top: int, left: int, shape: np.ndarray, color: int) -> None:
    h, w = shape.shape
    region = grid[top : top + h, left : left + w]
    region[shape] = color


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 5)
    colors = random.sample(COLOR_POOL, n_objects)
    grid_h = random.randint(14, 30)
    grid_w = random.randint(max(14, 7 * n_objects), min(36, max(14, 10 * n_objects)))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    placements = []
    for color in colors:
        square_shape = np.ones((SQUARE_SIZE, SQUARE_SIZE), dtype=bool)
        placed = False
        for _ in range(80):
            r = random.randint(1, grid_h - SQUARE_SIZE - 1)
            c = random.randint(1, grid_w - SQUARE_SIZE - 1)
            if _can_place(occupied, r, c, square_shape):
                _paint(grid, r, c, square_shape, color)
                occupied[r : r + SQUARE_SIZE, c : c + SQUARE_SIZE] |= _dilated(square_shape)
                placements.append((r, c, color))
                placed = True
                break
        if not placed:
            return None

    for _square_r, _square_c, color in placements:
        sprite_h = random.randint(1, 4)
        sprite_w = random.randint(1, 4)
        max_area = sprite_h * sprite_w
        min_area = 1 if max_area == 1 else 2
        area = random.randint(min_area, max_area)
        sprite = _random_sprite_shape(sprite_h, sprite_w, area)
        if sprite.shape == (SQUARE_SIZE, SQUARE_SIZE) and sprite.all():
            return None
        if sprite.shape[0] > SQUARE_SIZE or sprite.shape[1] > SQUARE_SIZE:
            return None
        placed = False
        for _ in range(120):
            r = random.randint(1, grid_h - sprite.shape[0] - 1)
            c = random.randint(1, grid_w - sprite.shape[1] - 1)
            if _can_place(occupied, r, c, sprite):
                _paint(grid, r, c, sprite, color)
                occupied[r : r + sprite.shape[0], c : c + sprite.shape[1]] |= _dilated(sprite)
                placed = True
                break
        if not placed:
            return None

    # Reject degenerate layouts where colors accidentally merge or parsing assumptions fail.
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape[0] != 7:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
