from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 500
SPRITE_SIZE = 3
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT_DIRS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    dirs = CARDINAL_DIRS if connectivity == 4 else EIGHT_DIRS
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            comp.append((cr, cc2))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc2 + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert len(grey) >= 4, "expected a grey rectangular frame"
    r0, c0, r1, c1 = _bbox(grey)
    assert r1 - r0 >= 6 and c1 - c0 >= 6, "frame must have room for at least three inner layers"
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    assert len(grey) == expected, "grey pixels must form a 1-pixel hollow rectangle"
    assert np.all(grid[r0, c0 : c1 + 1] == GREY)
    assert np.all(grid[r1, c0 : c1 + 1] == GREY)
    assert np.all(grid[r0 : r1 + 1, c0] == GREY)
    assert np.all(grid[r0 : r1 + 1, c1] == GREY)
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY)
    return r0, c0, r1, c1


def _extract_source_sprite(grid: np.ndarray, frame_top: int) -> tuple[np.ndarray, np.ndarray]:
    red_comps = _connected_components(grid == RED, connectivity=8)
    assert red_comps, "expected one red source sprite"
    above = [coords for coords in red_comps if int(coords[:, 0].max()) < frame_top]
    assert len(above) == 1, "expected exactly one red sprite above the frame"
    coords = above[0]
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sprite, coords


def _paste_sprite(out: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    for dr in range(h):
        for dc in range(w):
            if not sprite[dr, dc]:
                continue
            rr = top + dr
            cc = left + dc
            if not _in_bounds(out.shape, rr, cc):
                return
            if out[rr, cc] in (GREY, YELLOW):
                continue
            out[rr, cc] = RED


def _layer_depth(frame: tuple[int, int, int, int], r: int, c: int) -> int:
    r0, c0, r1, c1 = frame
    return int(min(r - r0, r1 - r, c - c0, c1 - c))


def _max_depth(frame: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    return int(min((inner_h + 1) // 2, (inner_w + 1) // 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp mirrored copies of the source sprite across the frame's concentric inner layers, alternating the mirror axis by layer depth parity.

    Input structure:
    - A black background contains one grey 1-pixel-thick hollow rectangle.
    - A small red asymmetric sprite sits above that frame, and one yellow marker lies inside the frame near the top-left corner of some concentric layer.
    - The frame interior is otherwise black and large enough to contain several concentric depth layers.

    Rule:
    - Identify the hollow grey rectangle and treat its inside as layered by distance to the inner border: cells adjacent to the frame are depth 1, the next ring is depth 2, and so on.
    - Read the red sprite above the frame as the source pattern, erase that external source in the output, and use the yellow marker as the top-left anchor for the sprite copy on the marker's layer.
    - Copy the sprite to every reachable corresponding top-left corner across shallower and deeper layers; odd depths use a left-right mirror, even depths use an up-down mirror, and deeper layers overwrite shallower red copies while never overwriting the grey frame or yellow marker.

    Key insight:
    - The controlling signal is not spatial side but concentric depth: each layer has its own parity, and that parity alone decides whether the sprite is flipped horizontally or vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_frame(g)
    frame_top, _frame_left, _frame_bottom, _frame_right = frame

    marker_cells = np.argwhere(g == YELLOW)
    assert len(marker_cells) == 1, "expected exactly one yellow marker"
    marker_r, marker_c = map(int, marker_cells[0])
    marker_depth = _layer_depth(frame, marker_r, marker_c)
    assert marker_depth >= 1, "marker must be inside the frame"

    sprite, source_coords = _extract_source_sprite(g, frame_top)
    assert sprite.shape == (SPRITE_SIZE, SPRITE_SIZE), "source sprite must occupy a 3x3 crop"

    out = g.copy()
    out[source_coords[:, 0], source_coords[:, 1]] = BLACK

    deepest = _max_depth(frame)
    for depth in range(1, deepest + 1):
        if depth % 2 == 1:
            transformed = np.fliplr(sprite)
        else:
            transformed = np.flipud(sprite)
        top = marker_r + (depth - marker_depth)
        left = marker_c + (depth - marker_depth)
        if top < frame[0] + 1 or left < frame[1] + 1:
            continue
        if top + transformed.shape[0] - 1 >= frame[2] or left + transformed.shape[1] - 1 >= frame[3]:
            continue
        _paste_sprite(out, transformed, top, left)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_connected_components(mask, connectivity=4)) == 1


def _build_sprite_catalog() -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for bits in range(1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for idx in range(SPRITE_SIZE * SPRITE_SIZE):
            if (bits >> idx) & 1:
                mask[idx // SPRITE_SIZE, idx % SPRITE_SIZE] = True
        cells = int(mask.sum())
        if cells < 4 or cells > 6:
            continue
        if mask[0, 0]:
            continue
        if not _mask_connected(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if not mask[0].any() and not mask[:, 0].any():
            continue
        sprites.append(mask)
    assert sprites, "sprite catalog must not be empty"
    return sprites


SPRITE_CATALOG = _build_sprite_catalog()


def _choose_sprite() -> np.ndarray:
    return random.choice(SPRITE_CATALOG).copy()


def _marker_depth_candidates(max_depth: int) -> list[int]:
    depths = list(range(1, max_depth + 1))
    odd = [d for d in depths if d % 2 == 1]
    even = [d for d in depths if d % 2 == 0]
    if odd and even:
        return odd if random.random() < 0.5 else even
    return depths


def _place_frame_case() -> np.ndarray | None:
    outer_h = random.randint(11, 17)
    outer_w = random.randint(11, 17)
    gap = random.randint(1, 2)
    sprite = _choose_sprite()

    top_margin = random.randint(3 + gap, 8)
    left_margin = random.randint(1, 5)
    bottom_margin = random.randint(1, 6)
    right_margin = random.randint(1, 6)

    height = top_margin + outer_h + bottom_margin
    width = left_margin + outer_w + right_margin
    if height > 30 or width > 30:
        return None

    frame_top = top_margin
    frame_left = left_margin
    frame_bottom = frame_top + outer_h - 1
    frame_right = frame_left + outer_w - 1

    grid = np.zeros((height, width), dtype=int)
    grid[frame_top, frame_left : frame_right + 1] = GREY
    grid[frame_bottom, frame_left : frame_right + 1] = GREY
    grid[frame_top : frame_bottom + 1, frame_left] = GREY
    grid[frame_top : frame_bottom + 1, frame_right] = GREY

    max_depth = _max_depth((frame_top, frame_left, frame_bottom, frame_right))
    if max_depth < 3:
        return None
    marker_depth = random.choice(_marker_depth_candidates(max_depth))
    marker_r = frame_top + marker_depth
    marker_c = frame_left + marker_depth
    grid[marker_r, marker_c] = YELLOW

    sprite_top = frame_top - gap - SPRITE_SIZE
    sprite_left = frame_left + (outer_w - SPRITE_SIZE) // 2
    for dr in range(SPRITE_SIZE):
        for dc in range(SPRITE_SIZE):
            if sprite[dr, dc]:
                grid[sprite_top + dr, sprite_left + dc] = RED

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _place_frame_case()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input grid")
