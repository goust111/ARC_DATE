import random
from typing import List, Optional, Sequence, Tuple

import numpy as np


BG = 0
BAR_COLOR = 5
STAMP_COLORS = (1, 2, 4, 6)
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

MIN_SIZE = 14
MAX_SIZE = 20
MIN_BAR_COUNT = 2
MAX_BAR_COUNT = 5
MIN_BAR_LENGTH = 3
MAX_BAR_LENGTH = 5
GEN_RETRIES = 400


def _find_components(mask: np.ndarray) -> List[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    components: List[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _component_bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _classify_component(grid: np.ndarray, coords: np.ndarray) -> Tuple[str, np.ndarray | str]:
    r0, c0, r1, c1 = _component_bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(len(coords))

    if area == 9 and h == 3 and w == 3 and np.all(patch != BG):
        return "stamp", patch.copy()

    if MIN_BAR_LENGTH <= area <= MAX_BAR_LENGTH and h == w == area:
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if np.array_equal(local, np.eye(h, dtype=bool)):
            return "bar", "backslash"
        if np.array_equal(local, np.fliplr(np.eye(h, dtype=bool))):
            return "bar", "slash"

    raise AssertionError("unexpected component shape")


def _render_output(stamp: np.ndarray, rows: int, cols: int) -> np.ndarray:
    mirrored = np.fliplr(stamp)
    out = np.empty((rows * 3, cols * 3), dtype=int)
    for i in range(rows):
        for j in range(cols):
            tile = stamp if (i + j) % 2 == 0 else mirrored
            out[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count diagonal bars by orientation, then tile the unique 3x3 stamp in a checkerboard
    alternation of original and left-right-mirrored copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    components = _find_components(g != BG)
    assert components

    stamp: Optional[np.ndarray] = None
    backslash_count = 0
    slash_count = 0
    for coords in components:
        kind, value = _classify_component(g, coords)
        if kind == "stamp":
            assert stamp is None
            stamp = value  # type: ignore[assignment]
        elif value == "backslash":
            backslash_count += 1
        else:
            slash_count += 1

    assert stamp is not None
    assert MIN_BAR_COUNT <= backslash_count <= MAX_BAR_COUNT
    assert MIN_BAR_COUNT <= slash_count <= MAX_BAR_COUNT
    return _render_output(stamp, backslash_count, slash_count)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _make_bar(length: int, orientation: str) -> np.ndarray:
    patch = np.full((length, length), BG, dtype=int)
    if orientation == "backslash":
        for i in range(length):
            patch[i, i] = BAR_COLOR
    else:
        for i in range(length):
            patch[i, length - 1 - i] = BAR_COLOR
    return patch


def _ensure_palette_coverage(stamp: np.ndarray, palette: Sequence[int]) -> None:
    cells = list(range(stamp.size))
    random.shuffle(cells)
    for color, index in zip(palette, cells):
        r, c = divmod(index, 3)
        stamp[r, c] = color


def _make_stamp() -> np.ndarray:
    modes = ("noise", "rows", "cols", "checker")
    for _ in range(100):
        color_count = random.randint(2, len(STAMP_COLORS))
        palette = random.sample(STAMP_COLORS, color_count)
        stamp = np.empty((3, 3), dtype=int)
        mode = random.choice(modes)

        if mode == "noise":
            stamp[:] = np.array([random.choice(palette) for _ in range(9)], dtype=int).reshape(3, 3)
        elif mode == "rows":
            for r in range(3):
                stamp[r, :] = random.choice(palette)
        elif mode == "cols":
            for c in range(3):
                stamp[:, c] = random.choice(palette)
        else:
            offset = random.randrange(color_count)
            rr, cc = np.indices((3, 3))
            stamp[:] = np.asarray(palette, dtype=int)[(rr + cc + offset) % color_count]

        _ensure_palette_coverage(stamp, palette)
        if np.unique(stamp).size < 2:
            continue
        if np.array_equal(stamp, np.fliplr(stamp)):
            row = random.randrange(3)
            right = int(stamp[row, 2])
            choices = [color for color in palette if color != right]
            if not choices:
                continue
            stamp[row, 0] = random.choice(choices)
        if not np.array_equal(stamp, np.fliplr(stamp)):
            return stamp
    raise RuntimeError("failed to create asymmetric stamp")


def _candidate_positions(blocked: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = mask.shape
    positions: List[Tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                positions.append((r0, c0))
    return positions


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    mask = patch != BG
    blocked = _dilate8(occupied)
    positions = _candidate_positions(blocked, mask)
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    ph, pw = patch.shape
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = patch[mask]
    occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
    return True


def _pick_grid_shape(backslash_count: int, slash_count: int) -> Tuple[int, int]:
    total_bars = backslash_count + slash_count
    min_side = max(MIN_SIZE, 10 + total_bars)
    return random.randint(min_side, MAX_SIZE), random.randint(min_side, MAX_SIZE)


def _expected_input_output(stamp: np.ndarray, backslash_count: int, slash_count: int) -> np.ndarray:
    return _render_output(stamp, backslash_count, slash_count)


def _build_scene() -> Optional[np.ndarray]:
    backslash_count, slash_count = random.sample(range(MIN_BAR_COUNT, MAX_BAR_COUNT + 1), 2)
    height, width = _pick_grid_shape(backslash_count, slash_count)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    stamp = _make_stamp()
    objects: List[np.ndarray] = [stamp]
    for _ in range(backslash_count):
        objects.append(_make_bar(random.randint(MIN_BAR_LENGTH, MAX_BAR_LENGTH), "backslash"))
    for _ in range(slash_count):
        objects.append(_make_bar(random.randint(MIN_BAR_LENGTH, MAX_BAR_LENGTH), "slash"))

    random.shuffle(objects)
    objects.sort(key=lambda patch: max(patch.shape), reverse=True)
    for patch in objects:
        if not _place_patch(grid, occupied, patch):
            return None

    components = _find_components(grid != BG)
    if len(components) != 1 + backslash_count + slash_count:
        return None

    try:
        expected = _expected_input_output(stamp, backslash_count, slash_count)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
