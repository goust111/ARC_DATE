from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANT_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _tile_key(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in tile.ravel())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _find_split(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full grey cross"
    split_r = rows[0]
    split_c = cols[0]
    top_h = split_r
    left_w = split_c
    bottom_h = h - split_r - 1
    right_w = w - split_c - 1
    assert top_h == bottom_h == left_w == right_w, "expected four equal square quadrants"
    return split_r, split_c, top_h


def _get_blocks(grid: np.ndarray, n: int) -> list[np.ndarray]:
    return [
        grid[:n, :n].copy(),
        grid[:n, n + 1 :].copy(),
        grid[n + 1 :, :n].copy(),
        grid[n + 1 :, n + 1 :].copy(),
    ]


def _set_block(grid: np.ndarray, idx: int, n: int, block: np.ndarray) -> None:
    r_mul, c_mul = QUADRANT_OFFSETS[idx]
    r0 = r_mul * (n + 1)
    c0 = c_mul * (n + 1)
    grid[r0 : r0 + n, c0 : c0 + n] = block


def _infer_missing_rotation(visible_blocks: list[np.ndarray]) -> np.ndarray:
    visible_keys = {_tile_key(block) for block in visible_blocks}
    assert len(visible_keys) == 3, "expected three distinct visible rotations"
    for block in visible_blocks:
        family = [np.rot90(block, k) for k in range(4)]
        family_keys = [_tile_key(member) for member in family]
        if len(set(family_keys)) != 4:
            continue
        if visible_keys.issubset(set(family_keys)):
            for member, key in zip(family, family_keys):
                if key not in visible_keys:
                    return member
    raise AssertionError("could not infer the missing rotation")


def _region_fill_color(area: int) -> int:
    if area <= 2:
        return RED
    if area <= 4:
        return GREEN
    return YELLOW


def _fill_enclosed_regions(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=int).copy()
    h, w = block.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if block[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if block[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and block[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    enclosed = (block == BLACK) & ~exterior
    for coords in _components(enclosed):
        color = _region_fill_color(len(coords))
        for r, c in coords:
            block[r, c] = color
    return block


def _hole_areas(block: np.ndarray) -> list[int]:
    block = np.asarray(block, dtype=int)
    h, w = block.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if block[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if block[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and block[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return sorted(len(coords) for coords in _components((block == BLACK) & ~exterior))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing rotation quadrant and fill each enclosed black region by its area bucket.

    Input structure:
    - An odd square grid is split by a one-cell grey cross into four equal square quadrants.
    - Three quadrants show blue patterns on a black background; the fourth quadrant is a solid orange placeholder.
    - The three visible blue patterns are distinct 90-degree rotations from one rotation family.

    Rule:
    - Detect the grey cross and extract the four quadrants.
    - Infer the unique missing 90-degree rotation and replace the orange placeholder with that blue-on-black tile.
    - In every quadrant, find 4-connected black regions that do not connect to the quadrant boundary, then fill area 1-2 with red, area 3-4 with green, and area 5+ with yellow.

    Key insight:
    - The orange quadrant hides one member of a four-rotation family, and enclosed-region filling only becomes correct after that family is completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    _split_r, _split_c, n = _find_split(g)
    blocks = _get_blocks(g, n)
    placeholder_ids = [i for i, block in enumerate(blocks) if np.all(block == ORANGE)]
    assert len(placeholder_ids) == 1, "expected exactly one orange placeholder quadrant"
    placeholder_idx = placeholder_ids[0]

    visible = [block for i, block in enumerate(blocks) if i != placeholder_idx]
    for block in visible:
        colors = set(map(int, np.unique(block)))
        assert colors <= {BLACK, BLUE}, "visible quadrants must be blue on black"

    blocks[placeholder_idx] = _infer_missing_rotation(visible)
    filled_blocks = [_fill_enclosed_regions(block) for block in blocks]

    out = g.copy()
    for idx, block in enumerate(filled_blocks):
        _set_block(out, idx, n, block)
    return out


def _arm_candidates(n: int, top: int, left: int, outer_h: int, outer_w: int) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = []

    def emit(anchors: list[tuple[int, int]], outward: tuple[int, int], max_len: int, turns: list[tuple[int, int]]) -> None:
        if max_len <= 0 or not anchors:
            return
        for ar, ac in anchors:
            for length in range(1, max_len + 1):
                arm: list[tuple[int, int]] = []
                ok = True
                for step in range(1, length + 1):
                    rr = ar + outward[0] * step
                    cc = ac + outward[1] * step
                    if not (0 <= rr < n and 0 <= cc < n):
                        ok = False
                        break
                    arm.append((rr, cc))
                if not ok:
                    continue
                candidates.append(tuple(arm))
                end_r, end_c = arm[-1]
                for turn in turns:
                    bent = list(arm)
                    for bend_len in range(1, 3):
                        rr = end_r + turn[0] * bend_len
                        cc = end_c + turn[1] * bend_len
                        if not (0 <= rr < n and 0 <= cc < n):
                            break
                        bent.append((rr, cc))
                        candidates.append(tuple(bent))

    emit(
        [(top, c) for c in range(left + 1, left + outer_w - 1)],
        (-1, 0),
        top,
        [(0, -1), (0, 1)],
    )
    emit(
        [(top + outer_h - 1, c) for c in range(left + 1, left + outer_w - 1)],
        (1, 0),
        n - (top + outer_h),
        [(0, -1), (0, 1)],
    )
    emit(
        [(r, left) for r in range(top + 1, top + outer_h - 1)],
        (0, -1),
        left,
        [(-1, 0), (1, 0)],
    )
    emit(
        [(r, left + outer_w - 1) for r in range(top + 1, top + outer_h - 1)],
        (0, 1),
        n - (left + outer_w),
        [(-1, 0), (1, 0)],
    )
    return list(dict.fromkeys(candidates))


def _make_base_tile(n: int, hole_h: int, hole_w: int, rng: random.Random) -> np.ndarray | None:
    outer_h = hole_h + 2
    outer_w = hole_w + 2
    placements = [(top, left) for top in range(n - outer_h + 1) for left in range(n - outer_w + 1)]
    rng.shuffle(placements)
    for top, left in placements:
        frame = np.full((n, n), BLACK, dtype=int)
        frame[top, left : left + outer_w] = BLUE
        frame[top + outer_h - 1, left : left + outer_w] = BLUE
        frame[top : top + outer_h, left] = BLUE
        frame[top : top + outer_h, left + outer_w - 1] = BLUE

        candidates = _arm_candidates(n, top, left, outer_h, outer_w)
        rng.shuffle(candidates)
        for arm in candidates:
            tile = frame.copy()
            for r, c in arm:
                tile[r, c] = BLUE
            family = [_tile_key(np.rot90(tile, k)) for k in range(4)]
            if len(set(family)) != 4:
                continue
            if _hole_areas(tile) != [hole_h * hole_w]:
                continue
            if len(_components(tile == BLUE)) != 1:
                continue
            return tile
    return None


def _sample_hole_dims(rng: random.Random) -> tuple[int, int, int]:
    bucket = rng.choice(["small", "medium", "large"])
    if bucket == "small":
        dims = rng.choice([(1, 1), (1, 2), (2, 1)])
        n = rng.choice([5, 6, 7])
    elif bucket == "medium":
        dims = rng.choice([(1, 3), (3, 1), (2, 2)])
        n = rng.choice([6, 7])
    else:
        dims = rng.choice([(2, 3), (3, 2)])
        n = rng.choice([6, 7])
    return n, dims[0], dims[1]


def _compose_grid(blocks: list[np.ndarray], n: int) -> np.ndarray:
    grid = np.full((2 * n + 1, 2 * n + 1), BLACK, dtype=int)
    grid[n, :] = GREY
    grid[:, n] = GREY
    for idx, block in enumerate(blocks):
        _set_block(grid, idx, n, block)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n, hole_h, hole_w = _sample_hole_dims(rng)
    base = _make_base_tile(n, hole_h, hole_w, rng)
    if base is None:
        return None

    rotations = [np.rot90(base, k) for k in range(4)]
    permutation = list(range(4))
    rng.shuffle(permutation)
    placed_tiles = [rotations[idx] for idx in permutation]

    missing_idx = rng.randrange(4)
    inp_tiles = [tile.copy() for tile in placed_tiles]
    inp_tiles[missing_idx] = np.full((n, n), ORANGE, dtype=int)

    inp = _compose_grid(inp_tiles, n)
    out_tiles = [_fill_enclosed_regions(tile) for tile in placed_tiles]
    expected = _compose_grid(out_tiles, n)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
