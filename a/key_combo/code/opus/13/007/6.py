from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _sort_key(coords: np.ndarray) -> tuple[int, int]:
    box = _bbox_from_coords(coords)
    return box[0], box[1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _hole_components(grid: np.ndarray, red_mask: np.ndarray) -> list[np.ndarray]:
    red_box = _bbox(red_mask)
    assert red_box is not None, "expected a red shape"
    r0, c0, r1, c1 = red_box
    local_black = grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK
    holes: list[np.ndarray] = []
    for coords in _components(local_black):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == r1 - r0)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == c1 - c0)
        ):
            continue
        shifted = coords.copy()
        shifted[:, 0] += r0
        shifted[:, 1] += c0
        holes.append(shifted)
    holes.sort(key=_sort_key)
    return holes


def _piece_components(grid: np.ndarray) -> list[np.ndarray]:
    pieces = _components(grid == GREY)
    pieces.sort(key=_sort_key)
    return pieces


def _crop_piece_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox_from_coords(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, box


def _apply_puzzle_fill(grid: np.ndarray) -> np.ndarray:
    red_mask = grid == RED
    holes = _hole_components(grid, red_mask)
    pieces = _piece_components(grid)
    assert len(holes) == len(pieces) and len(holes) >= 1, "piece-hole count mismatch"

    out = grid.copy()
    out[out == GREY] = BLACK

    for hole_coords, piece_coords in zip(holes, pieces):
        hole_mask, hole_box = _crop_piece_mask(hole_coords)
        piece_mask, _piece_box = _crop_piece_mask(piece_coords)
        assert hole_mask.shape == piece_mask.shape and np.array_equal(hole_mask, piece_mask), "piece does not match hole"
        hr0, hc0, _hr1, _hc1 = hole_box
        local = out[hr0 : hr0 + hole_mask.shape[0], hc0 : hc0 + hole_mask.shape[1]]
        assert np.all(local[hole_mask] == BLACK), "hole must be empty before placement"
        local[hole_mask] = ORANGE
    return out


def _scanline_fill(grid: np.ndarray) -> np.ndarray:
    marker_coords = np.argwhere(grid == BLUE)
    assert marker_coords.shape == (1, 2), "expected exactly one blue marker"
    marker_col = int(marker_coords[0, 1])

    out = grid.copy()
    wall_mask = (out == RED) | (out == ORANGE)
    for r in range(out.shape[0]):
        wall_cols = np.flatnonzero(wall_mask[r])
        if wall_cols.size == 0:
            continue
        left = wall_cols[wall_cols < marker_col]
        right = wall_cols[wall_cols >= marker_col]
        if left.size == 0 or right.size == 0:
            continue
        start = int(left.max()) + 1
        stop = int(right.min()) - 1
        if start > stop:
            continue
        for c in range(start, stop + 1):
            if out[r, c] not in (RED, ORANGE, BLUE):
                out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move gray hole-fillers into the red wall shape, recolor them orange, then scanline-fill the passage selected by the blue reference column.

    Input structure:
    - A black background contains one vertically oriented red wall-like shape plus one blue singleton marker.
    - The red shape has 2-3 enclosed black holes, while 2-3 separate grey puzzle pieces lie elsewhere on the canvas.
    - Each grey piece matches one hole exactly without rotation, and the blue pixel indicates the reference column used after the wall is completed.

    Rule:
    - Detect the enclosed black holes inside the red shape and the grey puzzle pieces, crop them to tight bounding boxes, and sort both groups by bounding-box top-left.
    - Translate each sorted grey piece into the correspondingly sorted hole, clear its original grey cells to black, and recolor the placed cells orange.
    - Treat all red and orange cells as wall pixels. For each row, find the nearest wall strictly left of the blue column and the nearest wall at or right of it; if both exist, recolor every non-wall, non-blue cell strictly between them to yellow.
    - Preserve the blue marker and preserve all red/orange wall pixels.

    Key insight:
    - The scanline boundaries are not fully present in the input: the grey pieces must first repair the red structure, and only that completed wall determines which row intervals the blue reference column can fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    repaired = _apply_puzzle_fill(g)
    return _scanline_fill(repaired)


def _make_u_shape(height: int, left_w: int, gap_w: int, right_w: int, bridge_t: int, open_down: bool) -> np.ndarray:
    width = left_w + gap_w + right_w
    mask = np.zeros((height, width), dtype=bool)
    mask[:, :left_w] = True
    mask[:, width - right_w :] = True
    if open_down:
        mask[:bridge_t, :] = True
    else:
        mask[height - bridge_t :, :] = True
    return mask


def _grow_region(mask: np.ndarray, target_area: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return None
    random.shuffle(coords)
    for seed in coords[: min(25, len(coords))]:
        region = {seed}
        frontier = {p for p in _neighbors4(*seed) if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p]}
        while frontier and len(region) < target_area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            region.add(cell)
            for nxt in _neighbors4(*cell):
                if 0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1] and mask[nxt] and nxt not in region:
                    frontier.add(nxt)
        if len(region) == target_area:
            return sorted(region)
    return None


def _enclosed_component(mask_after: np.ndarray, region: list[tuple[int, int]]) -> bool:
    black = ~mask_after
    comp = None
    start = region[0]
    for coords in _components(black):
        s = {(int(r), int(c)) for r, c in coords}
        if start in s:
            comp = s
            break
    if comp is None or len(comp) != len(region):
        return False
    h, w = mask_after.shape
    return all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp)


def _carve_holes(base_mask: np.ndarray, hole_count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    current = base_mask.copy()
    holes: list[np.ndarray] = []
    for _ in range(hole_count):
        success = False
        for _attempt in range(300):
            target_area = random.randint(3, 6 if hole_count == 3 else 8)
            candidate = _grow_region(current, target_area)
            if candidate is None:
                continue
            trial = current.copy()
            for r, c in candidate:
                trial[r, c] = False
            if not _enclosed_component(trial, candidate):
                continue
            coords = np.array(candidate, dtype=int)
            current = trial
            holes.append(coords)
            success = True
            break
        if not success:
            return None
    holes.sort(key=_sort_key)
    return current, holes


def _sample_shape_and_holes() -> tuple[np.ndarray, list[np.ndarray], bool] | None:
    for _ in range(400):
        height = random.randint(8, 10)
        left_w = random.randint(3, 4)
        right_w = random.randint(3, 4)
        gap_w = random.randint(2, 4)
        bridge_t = random.randint(2, 3)
        open_down = random.random() < 0.5
        base_mask = _make_u_shape(height, left_w, gap_w, right_w, bridge_t, open_down)
        if not (48 <= int(base_mask.sum()) <= 68):
            continue
        hole_count = random.randint(2, 3)
        carved = _carve_holes(base_mask, hole_count)
        if carved is None:
            continue
        shape_mask, holes = carved
        if not (40 <= int(shape_mask.sum()) <= 60):
            continue
        cavity_rows = range(bridge_t, height) if open_down else range(0, height - bridge_t)
        if not cavity_rows:
            continue
        return shape_mask, holes, open_down
    return None


def _hole_sprite(coords: np.ndarray) -> np.ndarray:
    mask, _ = _crop_piece_mask(coords)
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = GREY
    return sprite


def _can_place(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> bool:
    h, w = sprite.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    return np.all(patch == BLACK)


def _paste(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    h, w = sprite.shape
    mask = sprite != BLACK
    grid[r0 : r0 + h, c0 : c0 + w][mask] = sprite[mask]


def _build_input() -> np.ndarray | None:
    sampled = _sample_shape_and_holes()
    if sampled is None:
        return None
    shape_mask, holes, open_down = sampled
    shape_h, shape_w = shape_mask.shape

    piece_sprites = [_hole_sprite(hole) for hole in holes]
    piece_heights = [sprite.shape[0] for sprite in piece_sprites]
    piece_widths = [sprite.shape[1] for sprite in piece_sprites]
    strip_gap = random.randint(1, 2)
    strip_h = sum(piece_heights) + strip_gap * (len(piece_sprites) - 1)
    max_piece_w = max(piece_widths)

    piece_side = random.choice(("left", "right"))
    left_margin = random.randint(2, 4) + (max_piece_w + 2 if piece_side == "left" else 0)
    right_margin = random.randint(2, 4) + (max_piece_w + 2 if piece_side == "right" else 0)
    width = shape_w + left_margin + right_margin
    height = max(shape_h + 4, strip_h + 4, random.randint(15, 19))
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    shape_r0 = random.randint(1, height - shape_h - 2)
    shape_c0 = left_margin
    grid[shape_r0 : shape_r0 + shape_h, shape_c0 : shape_c0 + shape_w][shape_mask] = RED

    piece_r = random.randint(1, height - strip_h - 1)
    piece_rows: list[int] = []
    for sprite in piece_sprites:
        piece_rows.append(piece_r)
        piece_r += sprite.shape[0] + strip_gap

    for row, sprite in zip(piece_rows, piece_sprites):
        if piece_side == "left":
            c0 = random.randint(1, left_margin - sprite.shape[1] - 1)
        else:
            c0 = random.randint(shape_c0 + shape_w + 1, width - sprite.shape[1] - 1)
        if not _can_place(grid, sprite, row, c0):
            return None
        _paste(grid, sprite, row, c0)

    cavity_probe_row = shape_h - 1 if open_down else 0
    local_gap_cols = [int(c) for c in np.flatnonzero(~shape_mask[cavity_probe_row])]
    if not local_gap_cols:
        return None
    marker_local_c = random.choice(local_gap_cols)
    cavity_rows = list(range(2, shape_h)) if open_down else list(range(0, shape_h - 2))
    cavity_rows = [r for r in cavity_rows if not shape_mask[r, marker_local_c]]
    if not cavity_rows:
        cavity_rows = [r for r in range(shape_h) if not shape_mask[r, marker_local_c]]
    if not cavity_rows:
        return None
    marker_r = shape_r0 + random.choice(cavity_rows)
    marker_c = shape_c0 + marker_local_c
    if grid[marker_r, marker_c] != BLACK:
        return None
    grid[marker_r, marker_c] = BLUE

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.count_nonzero(solved == ORANGE):
        return None
    if not np.count_nonzero(solved == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
