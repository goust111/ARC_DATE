from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
GEN_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the blue sprite to the wall indicated by the red cross, then mirror-concatenate it across that wall.

    Input structure:
    - The background is black.
    - A single blue 4-connected sprite, a single red orthogonal cross marker with one uniquely longest arm, and a single grey straight wall segment appear on the grid.
    - The wall is horizontal or vertical, and the blue sprite lies between the marker and that wall along the marker's longest-arm direction.

    Rule:
    - Read the longest arm of the red cross to obtain the slide direction.
    - Slide the entire blue sprite rigidly one cell at a time in that direction until its next step would hit the grey wall, so the sprite stops 4-neighbor adjacent to the wall without covering it.
    - Keep that slid blue sprite, keep the grey wall, erase the red marker, and place a green mirror image of the slid blue sprite on the opposite side of the wall using the wall as the reflection axis.

    Key insight:
    - The red cross does not mark a destination cell; it encodes the direction of a mandatory slide, and only after the blue sprite reaches the wall does the wall become the axis that determines the mirrored concatenation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = _parse_wall(g)
    direction = _parse_marker_direction(g)
    blue = _single_component_coords(g == BLUE)
    slid = _slide_until_wall(blue, direction, wall, g.shape)
    mirrored = _mirror_coords(slid, wall)

    out = np.full_like(g, BG)
    out[g == GREY] = GREY
    out[slid[:, 0], slid[:, 1]] = BLUE
    for r, c in mirrored:
        assert out[r, c] == BG
        out[r, c] = GREEN
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")


def _generate_attempt() -> np.ndarray | None:
    direction = DIRS4[int(RNG.integers(len(DIRS4)))]
    long_arm = int(RNG.integers(2, 4))
    slide_gap = int(RNG.integers(2, 5))
    marker_gap = int(RNG.integers(1, 3))
    sprite = _random_sprite_mask(direction)
    h, w = sprite.shape

    if direction in ((0, -1), (0, 1)):
        wall_len = int(RNG.integers(max(5, h), 9))
        extra = wall_len - h
        top_extra = int(RNG.integers(extra + 1))
        top_margin = int(RNG.integers(1, 4))
        bottom_margin = int(RNG.integers(1, 4))
        H = top_margin + wall_len + bottom_margin
        if H > 30:
            return None
        wall_r0 = top_margin
        sprite_r0 = wall_r0 + top_extra
        input_rows = sprite_r0 + np.argwhere(sprite)[:, 0]
        marker_center_r = int(RNG.choice(input_rows))

        left_margin = int(RNG.integers(1, 4))
        right_margin = int(RNG.integers(1, 4))

        if direction == (0, 1):
            marker_left = left_margin
            marker_center_c = marker_left + 1
            marker_right = marker_center_c + long_arm
            input_c0 = marker_right + marker_gap + 1
            final_c0 = input_c0 + slide_gap
            wall_c = final_c0 + w
            W = wall_c + w + right_margin + 1
        else:
            wall_c = left_margin + w
            final_c0 = wall_c + 1
            input_c0 = final_c0 + slide_gap
            marker_left = input_c0 + w + marker_gap
            marker_center_c = marker_left + long_arm
            W = marker_center_c + 2 + right_margin

        if W > 30:
            return None

        wall_cells = {(r, wall_c) for r in range(wall_r0, wall_r0 + wall_len)}
        final_offset = np.array([sprite_r0, final_c0], dtype=int)
        input_offset = np.array([sprite_r0, input_c0], dtype=int)
        marker_center = (marker_center_r, marker_center_c)

    else:
        wall_len = int(RNG.integers(max(5, w), 9))
        extra = wall_len - w
        left_extra = int(RNG.integers(extra + 1))
        left_margin = int(RNG.integers(1, 4))
        right_margin = int(RNG.integers(1, 4))
        W = left_margin + wall_len + right_margin
        if W > 30:
            return None
        wall_c0 = left_margin
        sprite_c0 = wall_c0 + left_extra
        input_cols = sprite_c0 + np.argwhere(sprite)[:, 1]
        marker_center_c = int(RNG.choice(input_cols))

        top_margin = int(RNG.integers(1, 4))
        bottom_margin = int(RNG.integers(1, 4))

        if direction == (1, 0):
            marker_top = top_margin
            marker_center_r = marker_top + 1
            marker_bottom = marker_center_r + long_arm
            input_r0 = marker_bottom + marker_gap + 1
            final_r0 = input_r0 + slide_gap
            wall_r = final_r0 + h
            H = wall_r + h + bottom_margin + 1
        else:
            wall_r = top_margin + h
            final_r0 = wall_r + 1
            input_r0 = final_r0 + slide_gap
            marker_top = input_r0 + h + marker_gap
            marker_center_r = marker_top + long_arm
            H = marker_center_r + 2 + bottom_margin

        if H > 30:
            return None

        wall_cells = {(wall_r, c) for c in range(wall_c0, wall_c0 + wall_len)}
        final_offset = np.array([final_r0, sprite_c0], dtype=int)
        input_offset = np.array([input_r0, sprite_c0], dtype=int)
        marker_center = (marker_center_r, marker_center_c)

    if not (0 <= marker_center[0] < H and 0 <= marker_center[1] < W):
        return None

    input_blue = np.argwhere(sprite) + input_offset
    final_blue = np.argwhere(sprite) + final_offset
    if _coords_overlap(set(map(tuple, input_blue)), wall_cells):
        return None
    if _coords_overlap(set(map(tuple, final_blue)), wall_cells):
        return None

    marker_cells = _marker_cells(marker_center, direction, long_arm)
    if any(not _in_bounds((H, W), r, c) for r, c in marker_cells):
        return None

    grid = np.zeros((H, W), dtype=int)
    for r, c in wall_cells:
        grid[r, c] = GREY
    for r, c in input_blue:
        if grid[r, c] != BG:
            return None
        grid[r, c] = BLUE
    for r, c in marker_cells:
        if grid[r, c] != BG:
            return None
        grid[r, c] = RED

    expected = np.zeros((H, W), dtype=int)
    for r, c in wall_cells:
        expected[r, c] = GREY
    expected[final_blue[:, 0], final_blue[:, 1]] = BLUE
    wall = _parse_wall(grid)
    for r, c in _mirror_coords(final_blue, wall):
        if expected[r, c] != BG:
            return None
        expected[r, c] = GREEN

    if not _generated_input_is_valid(grid, direction, slide_gap, expected):
        return None
    return grid


def _generated_input_is_valid(grid: np.ndarray, direction: tuple[int, int], slide_gap: int, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) != {BG, BLUE, RED, GREY}:
        return False

    blue = _single_component_coords(grid == BLUE)
    if blue.shape[0] < 4 or blue.shape[0] > 6:
        return False
    blue_mask = _tight_mask(blue)
    if np.array_equal(blue_mask, np.flipud(blue_mask)) or np.array_equal(blue_mask, np.fliplr(blue_mask)):
        return False

    wall = _parse_wall(grid)
    red = _parse_marker_direction(grid)
    if red != direction:
        return False

    slid = _slide_until_wall(blue, direction, wall, grid.shape)
    shift = slid[0] - blue[0]
    moved = abs(int(shift[0])) + abs(int(shift[1]))
    if moved != slide_gap:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    no_slide = np.zeros_like(grid)
    no_slide[grid == GREY] = GREY
    no_slide[blue[:, 0], blue[:, 1]] = BLUE
    no_slide_possible = True
    for r, c in _mirror_coords(blue, wall):
        if not _in_bounds(grid.shape, r, c):
            no_slide_possible = False
            break
        if no_slide[r, c] != BG:
            return False
        no_slide[r, c] = GREEN
    if no_slide_possible and np.array_equal(no_slide, expected):
        return False

    viable = 0
    for cand in DIRS4:
        trial = _try_full_transform(blue, cand, wall, grid.shape)
        if trial is not None:
            viable += 1
    if viable != 1:
        return False

    if not _sprite_between_marker_and_wall(grid, direction, wall):
        return False
    return True


def _try_full_transform(
    blue: np.ndarray,
    direction: tuple[int, int],
    wall: dict,
    shape: tuple[int, int],
) -> np.ndarray | None:
    try:
        slid = _slide_until_wall(blue, direction, wall, shape)
        mirrored = _mirror_coords(slid, wall)
    except AssertionError:
        return None
    H, W = shape
    occ = {tuple(x) for x in slid.tolist()}
    for r, c in mirrored:
        if not (0 <= r < H and 0 <= c < W):
            return None
        if (r, c) in occ or (r, c) in wall["cells"]:
            return None
    if np.array_equal(slid, blue):
        return None
    return slid


def _sprite_between_marker_and_wall(grid: np.ndarray, direction: tuple[int, int], wall: dict) -> bool:
    blue = _single_component_coords(grid == BLUE)
    red = np.argwhere(grid == RED)
    rb0, cb0, rb1, cb1 = _bbox(red)
    bb0, bc0, bb1, bc1 = _bbox(blue)
    if direction == (0, 1):
        return cb1 < bc0 and bc1 < wall["axis"]
    if direction == (0, -1):
        return wall["axis"] < bc0 and bc1 < cb0
    if direction == (1, 0):
        return rb1 < bb0 and bb1 < wall["axis"]
    return wall["axis"] < bb0 and bb1 < rb0


def _parse_wall(grid: np.ndarray) -> dict:
    coords = _single_component_coords(grid == GREY)
    n = coords.shape[0]
    assert 5 <= n <= 8
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if rows.size == 1:
        row = int(rows[0])
        c0, c1 = int(cols.min()), int(cols.max())
        assert cols.size == c1 - c0 + 1
        return {"orientation": "h", "axis": row, "span": (c0, c1), "cells": {tuple(x) for x in coords.tolist()}}
    if cols.size == 1:
        col = int(cols[0])
        r0, r1 = int(rows.min()), int(rows.max())
        assert rows.size == r1 - r0 + 1
        return {"orientation": "v", "axis": col, "span": (r0, r1), "cells": {tuple(x) for x in coords.tolist()}}
    raise AssertionError("wall must be a straight segment")


def _parse_marker_direction(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert 6 <= coords.shape[0] <= 7
    cells = {tuple(x) for x in coords.tolist()}
    centers = []
    for r, c in cells:
        if all((r + dr, c + dc) in cells for dr, dc in DIRS4):
            centers.append((r, c))
    assert len(centers) == 1
    center = centers[0]
    lengths = {}
    covered = {center}
    for dr, dc in DIRS4:
        rr, cc = center
        length = 0
        while (rr + dr, cc + dc) in cells:
            rr += dr
            cc += dc
            covered.add((rr, cc))
            length += 1
        lengths[(dr, dc)] = length
        assert length >= 1
    assert covered == cells
    best = max(lengths, key=lengths.get)
    assert list(lengths.values()).count(lengths[best]) == 1
    return best


def _slide_until_wall(
    blue: np.ndarray,
    direction: tuple[int, int],
    wall: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    dr, dc = direction
    wall_cells = wall["cells"]
    cur = blue.copy()
    H, W = shape
    while True:
        nxt = cur + np.array([dr, dc], dtype=int)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= H) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= W):
            break
        if any((int(r), int(c)) in wall_cells for r, c in nxt):
            break
        cur = nxt
    assert any((int(r) + dr, int(c) + dc) in wall_cells for r, c in cur)
    return cur


def _mirror_coords(coords: np.ndarray, wall: dict) -> list[tuple[int, int]]:
    if wall["orientation"] == "v":
        axis = wall["axis"]
        mirrored = [(int(r), int(2 * axis - c)) for r, c in coords]
    else:
        axis = wall["axis"]
        mirrored = [(int(2 * axis - r), int(c)) for r, c in coords]
    return mirrored


def _single_component_coords(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    assert len(comps) == 1
    return comps[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _random_sprite_mask(direction: tuple[int, int]) -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(2, 5))
        w = int(RNG.integers(2, 5))
        area = int(RNG.integers(4, 7))
        cells = {(int(RNG.integers(h)), int(RNG.integers(w)))}
        while len(cells) < area:
            r, c = list(cells)[int(RNG.integers(len(cells)))]
            nbrs = []
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            cells.add(nbrs[int(RNG.integers(len(nbrs)))])
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.sum() != area:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        if direction == (0, 1) and not mask[:, -1].any():
            continue
        if direction == (0, -1) and not mask[:, 0].any():
            continue
        if direction == (1, 0) and not mask[-1].any():
            continue
        if direction == (-1, 0) and not mask[0].any():
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    raise RuntimeError("failed to sample sprite")


def _marker_cells(center: tuple[int, int], direction: tuple[int, int], long_arm: int) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for dr, dc in DIRS4:
        length = long_arm if (dr, dc) == direction else 1
        for step in range(1, length + 1):
            cells.add((r + dr * step, c + dc * step))
    return cells


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _coords_overlap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return not a.isdisjoint(b)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]
