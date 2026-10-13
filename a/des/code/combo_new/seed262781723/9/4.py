import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
PALETTE = tuple(color for color in range(10) if color not in (BLACK, RED))

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATIONAL = "rotational"
NONE = "none"
BRANCHES = (HORIZONTAL, VERTICAL, ROTATIONAL, NONE)

LAYOUTS = ((1, 4), (2, 2), (4, 1), (2, 3), (3, 2))
SEED_MIN = 2
SEED_MAX_BY_COUNT = {1: 5, 2: 5, 3: 4, 4: 3}
MAX_GENERATION_ATTEMPTS = 400


def _bands(size: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(int(x) for x in separators):
        if start < idx:
            bands.append((start, idx))
        start = idx + 1
    if start < size:
        bands.append((start, size))
    return bands


def _symmetry_flags(seed: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        np.array_equal(seed, np.flipud(seed)),
        np.array_equal(seed, np.fliplr(seed)),
        np.array_equal(seed, np.rot90(seed, 2)),
    )


def _classify_seed(seed: np.ndarray) -> str:
    has_h, has_v, has_rot = _symmetry_flags(seed)
    if has_h and not has_v:
        return HORIZONTAL
    if has_v and not has_h:
        return VERTICAL
    if has_rot and not has_h and not has_v:
        return ROTATIONAL
    return NONE


def _normalize_room(room: np.ndarray) -> tuple[np.ndarray, str]:
    coords = np.argwhere(room != BLACK)
    assert len(coords) > 0, "room must contain a seed"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = room.shape

    if r0 == 0 and c0 == 0:
        return room.copy(), "tl"
    if r0 == 0 and c1 == w - 1:
        return np.fliplr(room), "tr"
    if r1 == h - 1 and c0 == 0:
        return np.flipud(room), "bl"
    if r1 == h - 1 and c1 == w - 1:
        return np.flipud(np.fliplr(room)), "br"
    raise AssertionError("seed must occupy one room corner")


def _restore_orientation(room: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return room
    if corner == "tr":
        return np.fliplr(room)
    if corner == "bl":
        return np.flipud(room)
    if corner == "br":
        return np.fliplr(np.flipud(room))
    raise AssertionError(f"unknown corner {corner}")


def _fill_from_topleft(seed: np.ndarray, symmetry: str) -> np.ndarray:
    sh, sw = seed.shape
    up_down = np.flipud(seed)
    left_right = np.fliplr(seed)
    rotated = np.rot90(seed, 2)

    if symmetry == HORIZONTAL:
        left_half = np.vstack([seed, up_down])
        return np.hstack([left_half, np.fliplr(left_half)])

    if symmetry == VERTICAL:
        top_half = np.hstack([seed, left_right])
        return np.vstack([top_half, np.flipud(top_half)])

    out = np.empty((sh * 2, sw * 2), dtype=int)
    out[:sh, :sw] = seed
    out[:sh, sw:] = left_right
    out[sh:, :sw] = up_down
    out[sh:, sw:] = rotated
    return out


def _expand_room(room: np.ndarray) -> np.ndarray:
    normalized, corner = _normalize_room(room)
    coords = np.argwhere(normalized != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r0 == 0 and c0 == 0, "normalized seed must be in the upper-left corner"

    seed = normalized[: r1 + 1, : c1 + 1]
    h, w = normalized.shape
    sh, sw = seed.shape
    assert h == sh * 2 and w == sw * 2, "seed must occupy exactly one quadrant"

    symmetry = _classify_seed(seed)
    filled = _fill_from_topleft(seed, symmetry)
    return _restore_orientation(filled, corner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each corner seed across its red-lined room using the seed's symmetry class.

    Input structure:
    - The grid has black background and full red separator rows and columns that split it into rectangular rooms.
    - Every room contains exactly one multicolor seed pattern in one corner, and that seed occupies half the room height and half the room width.
    - The rest of each room is black until the seed is expanded.

    Rule:
    - Split the grid into rooms at the red separator rows and columns, then crop the non-black seed from each room corner.
    - Classify the seed by its color-pattern symmetry: horizontal-axis, vertical-axis, 180-degree rotational, or none.
    - Normalize the room so the seed sits in the upper-left quadrant, complete the room with the matching reflections and rotation, then undo the normalization.
    - Horizontal-axis seeds repeat across the top/bottom split before mirroring left-right; vertical-axis seeds repeat across the left/right split before mirroring top-bottom.
    - Rotationally symmetric seeds repeat on the diagonal, while non-symmetric seeds use the original, a left-right mirror, a top-bottom mirror, and a 180-degree rotation in the four quadrants. Red separator lines stay unchanged.

    Key insight:
    - The visible seed is only one quarter of its room, so the correct completion depends on recognizing which symmetries the seed itself already has before deciding which reflected copies should coincide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == RED)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == RED)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            room = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _expand_room(room)
    return out


def _choice_grid(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    return np.array(
        [[random.choice(palette) for _ in range(shape[1])] for _ in range(shape[0])],
        dtype=int,
    )


def _uses_enough_colors(seed: np.ndarray) -> bool:
    return len(set(map(int, np.unique(seed)))) >= 2


def _make_horizontal_seed(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    seed = np.empty((h, w), dtype=int)
    for r in range((h + 1) // 2):
        row = [random.choice(palette) for _ in range(w)]
        seed[r, :] = row
        seed[h - 1 - r, :] = row
    return seed


def _make_vertical_seed(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    seed = np.empty((h, w), dtype=int)
    for c in range((w + 1) // 2):
        col = [random.choice(palette) for _ in range(h)]
        seed[:, c] = col
        seed[:, w - 1 - c] = col
    return seed


def _make_rotational_seed(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    seed = np.empty((h, w), dtype=int)
    done = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if done[r, c]:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            color = random.choice(palette)
            seed[r, c] = color
            seed[rr, cc] = color
            done[r, c] = True
            done[rr, cc] = True
    return seed


def _make_none_seed(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    return _choice_grid(shape, palette)


def _build_seed(shape: tuple[int, int], branch: str) -> np.ndarray:
    makers = {
        HORIZONTAL: _make_horizontal_seed,
        VERTICAL: _make_vertical_seed,
        ROTATIONAL: _make_rotational_seed,
        NONE: _make_none_seed,
    }
    for _ in range(300):
        palette_size = random.randint(2, min(4, len(PALETTE)))
        palette = random.sample(list(PALETTE), palette_size)
        seed = makers[branch](shape, palette)
        if not _uses_enough_colors(seed):
            continue
        if _classify_seed(seed) == branch:
            return seed
    raise RuntimeError(f"failed to generate a {branch} seed of shape {shape}")


def _sample_seed_sizes(count: int) -> list[int]:
    max_seed = SEED_MAX_BY_COUNT[count]
    return [random.randint(SEED_MIN, max_seed) for _ in range(count)]


def _room_branch_list(room_count: int) -> list[str]:
    branches = list(BRANCHES)
    while len(branches) < room_count:
        extras = list(BRANCHES)
        random.shuffle(extras)
        branches.extend(extras)
    random.shuffle(branches)
    return branches[:room_count]


def _place_seed(room: np.ndarray, seed: np.ndarray, corner: str) -> np.ndarray:
    sh, sw = seed.shape
    out = room.copy()
    if corner == "tl":
        out[:sh, :sw] = seed
    elif corner == "tr":
        out[:sh, -sw:] = seed
    elif corner == "bl":
        out[-sh:, :sw] = seed
    elif corner == "br":
        out[-sh:, -sw:] = seed
    else:
        raise AssertionError(f"unknown corner {corner}")
    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(LAYOUTS)
    seed_heights = _sample_seed_sizes(room_rows)
    seed_widths = _sample_seed_sizes(room_cols)

    room_heights = [2 * value for value in seed_heights]
    room_widths = [2 * value for value in seed_widths]
    total_h = sum(room_heights) + (room_rows - 1)
    total_w = sum(room_widths) + (room_cols - 1)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_edges = []
    cursor = 0
    for idx, height in enumerate(room_heights):
        row_edges.append((cursor, cursor + height))
        cursor += height
        if idx != room_rows - 1:
            grid[cursor, :] = RED
            cursor += 1

    col_edges = []
    cursor = 0
    for idx, width in enumerate(room_widths):
        col_edges.append((cursor, cursor + width))
        cursor += width
        if idx != room_cols - 1:
            grid[:, cursor] = RED
            cursor += 1

    branches = _room_branch_list(room_rows * room_cols)
    corners = ["tl", "tr", "bl", "br"]
    index = 0
    for r_idx, (r0, r1) in enumerate(row_edges):
        for c_idx, (c0, c1) in enumerate(col_edges):
            branch = branches[index]
            index += 1
            seed = _build_seed((seed_heights[r_idx], seed_widths[c_idx]), branch)
            room = np.full((r1 - r0, c1 - c0), BLACK, dtype=int)
            room = _place_seed(room, seed, random.choice(corners))
            grid[r0:r1, c0:c1] = room

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid symmetry-room input")
