from __future__ import annotations

import random

import numpy as np


BG = 0
AZURE = 8
MARKER_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
WINDOW = 5
MARKER_BOX = 2
GEN_ATTEMPTS = 400

# Missing-corner position in the 2x2 L marker -> np.rot90 k value.
ROTATION_K = {
    (0, 0): 0,  # no rotation
    (0, 1): 3,  # clockwise 90
    (1, 1): 2,  # 180
    (1, 0): 1,  # counterclockwise 90
}


def _scan_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    markers: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            values = [int(v) for v in block.ravel()]
            counts: dict[int, int] = {}
            for value in values:
                if value in (BG, AZURE):
                    continue
                counts[value] = counts.get(value, 0) + 1
            color = next((col for col, count in counts.items() if count == 3), None)
            if color is None:
                continue
            marker_mask = block == color
            if marker_mask.sum() != 3:
                continue
            miss_r, miss_c = np.argwhere(~marker_mask)[0]
            markers.append((r, c, ROTATION_K[(int(miss_r), int(miss_c))]))
    return markers


def _texture_only(grid: np.ndarray) -> np.ndarray:
    return np.where(grid == AZURE, AZURE, BG)


def _marker_patch(texture: np.ndarray, marker_r: int, marker_c: int) -> np.ndarray:
    return texture[marker_r - 1 : marker_r + 4, marker_c - 1 : marker_c + 4]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    markers = _scan_markers(g)
    assert markers, "expected at least one L marker"

    texture = _texture_only(g)
    pieces = [np.rot90(_marker_patch(texture, r, c), k) for r, c, k in sorted(markers)]
    return np.concatenate(pieces, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Crop texture windows around colored L markers, rotate them by marker orientation, and concatenate the results.

    Input structure:
    - A larger black-background grid contains an azure repeated texture pattern.
    - Several colored L markers sit on top of the texture; each marker is a 3-pixel monochrome L inside a 2x2 box.
    - The empty corner of each L encodes one of four orientations, and the marker boxes do not overlap.

    Rule:
    - For each L marker, take the 5x5 texture window anchored around that marker's 2x2 box and ignore all marker pixels as black.
    - Decode the missing corner of the L as a rotation: top-left missing -> 0 degrees, top-right missing -> 90 degrees clockwise, bottom-right missing -> 180 degrees, bottom-left missing -> 90 degrees counterclockwise.
    - Sort the markers by their positions from top to bottom and then left to right, rotate each cropped 5x5 texture block accordingly, and concatenate the rotated blocks horizontally.

    Key insight:
    - Each L marker carries two pieces of information at once: its position selects which texture region to crop, while its missing corner silently specifies how that crop must be rotated before assembly.
    """
    return _solve(grid)


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = tile.shape
    reps = np.tile(tile, (h // th + 1, w // tw + 1))
    return reps[:h, :w]


def _random_tile(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        th = rng.randint(2, 4)
        tw = rng.randint(2, 5)
        tile = np.zeros((th, tw), dtype=int)

        density = rng.uniform(0.28, 0.58)
        for r in range(th):
            for c in range(tw):
                if rng.random() < density:
                    tile[r, c] = AZURE

        if not tile.any() or np.all(tile == AZURE):
            continue
        if len({tuple(row.tolist()) for row in tile}) == 1 and len({tuple(col.tolist()) for col in tile.T}) == 1:
            continue

        if th == tw:
            if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
                continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate texture tile")


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_windows(rng: random.Random, h: int, w: int, count: int) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(500):
        placements.clear()
        rects.clear()
        coords = [(r, c) for r in range(h - WINDOW + 1) for c in range(w - WINDOW + 1)]
        rng.shuffle(coords)
        for r, c in coords:
            rect = (r, c, r + WINDOW - 1, c + WINDOW - 1)
            if any(_rects_overlap(rect, prev, gap=1) for prev in rects):
                continue
            placements.append((r, c))
            rects.append(rect)
            if len(placements) == count:
                return placements
    raise RuntimeError("failed to place crop windows")


def _orientation_sequence(rng: random.Random, count: int) -> list[tuple[int, int]]:
    corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(corners)
    result = corners[: min(count, 4)]
    while len(result) < count:
        result.append(rng.choice(corners))
    rng.shuffle(result)
    return result


def _paint_marker(grid: np.ndarray, marker_r: int, marker_c: int, missing_corner: tuple[int, int], color: int) -> None:
    for dr in range(MARKER_BOX):
        for dc in range(MARKER_BOX):
            if (dr, dc) == missing_corner:
                continue
            grid[marker_r + dr, marker_c + dc] = color


def _valid_patches(texture: np.ndarray, markers: list[tuple[int, int, tuple[int, int]]]) -> bool:
    rotated_signatures = []
    for marker_r, marker_c, missing_corner in markers:
        patch = _marker_patch(texture, marker_r, marker_c)
        if patch.shape != (WINDOW, WINDOW):
            return False
        if AZURE not in patch or np.all(patch == AZURE):
            return False
        if all(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            return False
        k = ROTATION_K[missing_corner]
        rotated = np.rot90(patch, k)
        rotated_signatures.append(rotated.tobytes())
    return len(set(rotated_signatures)) >= max(2, len(markers) // 2)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    marker_count = rng.randint(2, 5)
    out_width = marker_count * WINDOW
    if out_width > 30:
        return None

    h = rng.randint(12, 29)
    w = rng.randint(max(12, out_width), 29)
    if h < WINDOW or w < WINDOW:
        return None

    try:
        windows = _place_windows(rng, h, w, marker_count)
    except RuntimeError:
        return None

    tile = _random_tile(rng)
    texture = _repeat_tile(tile, h, w)
    colors = rng.sample(MARKER_COLORS, marker_count)
    missing_corners = _orientation_sequence(rng, marker_count)

    markers: list[tuple[int, int, tuple[int, int]]] = []
    for (win_r, win_c), missing_corner in zip(windows, missing_corners):
        marker_r = win_r + 1
        marker_c = win_c + 1
        markers.append((marker_r, marker_c, missing_corner))

    if not _valid_patches(texture, markers):
        return None

    grid = texture.copy()
    for (marker_r, marker_c, missing_corner), color in zip(markers, colors):
        _paint_marker(grid, marker_r, marker_c, missing_corner, color)

    found = sorted(_scan_markers(grid))
    expected = sorted((marker_r, marker_c, ROTATION_K[missing_corner]) for marker_r, marker_c, missing_corner in markers)
    if found != expected:
        return None

    out = _solve(grid)
    if out.shape != (WINDOW, out_width) or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
