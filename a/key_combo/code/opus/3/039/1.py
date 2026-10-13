from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    """ARC-style 4-neighborhood using (row, col) coordinates."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _detect_axis_from_walls(grid: np.ndarray) -> str:
    walls = np.asarray(grid == WALL, dtype=bool)
    h, w = walls.shape
    vertical = (w % 2 == 1) and np.array_equal(walls, walls[:, ::-1])
    horizontal = (h % 2 == 1) and np.array_equal(walls, walls[::-1, :])
    if vertical == horizontal:
        raise AssertionError("expected exactly one wall-symmetry axis")
    return "vertical" if vertical else "horizontal"


def _mirror_point(r: int, c: int, axis: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if axis == "vertical":
        return r, w - 1 - c
    return h - 1 - r, c


def _mirror_coords(coords: np.ndarray, axis: str, shape: tuple[int, int]) -> np.ndarray:
    mirrored = np.empty_like(coords)
    if axis == "vertical":
        mirrored[:, 0] = coords[:, 0]
        mirrored[:, 1] = shape[1] - 1 - coords[:, 1]
    else:
        mirrored[:, 0] = shape[0] - 1 - coords[:, 0]
        mirrored[:, 1] = coords[:, 1]
    return mirrored


def _coord_key(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror one-sided chamber markers across the symmetric wall axis, then flood-fill both chambers.

    Input structure:
    - Black background cells are partitioned by grey wall pixels into 4-connected regions.
    - The grey wall layout is exactly symmetric across either the center column or the center row.
    - Some enclosed regions on only one half contain a single colored marker pixel from {1,2,3,4}; their mirrored partner regions are empty.

    Rule:
    - Infer the unique symmetry axis from the grey wall geometry alone.
    - Reflect every non-black, non-grey marker across that axis, placing the same color in the mirrored chamber.
    - Treat grey as barriers; for each enclosed 4-connected non-wall region, fill the whole region if it contains exactly one marker, otherwise leave it unchanged.

    Key insight:
    - The wall symmetry reveals where missing seeds belong, but the visible area change happens only after chamber-wise flood filling from those mirrored seeds.
    """
    g = np.asarray(grid, dtype=int)
    axis = _detect_axis_from_walls(g)

    seeded = g.copy()
    markers = np.argwhere((g != BG) & (g != WALL))
    for r, c in markers:
        color = int(g[r, c])
        mr, mc = _mirror_point(int(r), int(c), axis, g.shape)
        if int(g[mr, mc]) == WALL:
            raise AssertionError("mirrored marker must land inside a chamber")
        if int(seeded[mr, mc]) not in (BG, color):
            raise AssertionError("mirrored markers must not conflict")
        seeded[mr, mc] = color

    out = seeded.copy()
    for comp in _components(seeded != WALL):
        if _touches_border(comp, seeded.shape):
            continue
        marker_cells = [(int(r), int(c)) for r, c in comp if int(seeded[r, c]) not in (BG, WALL)]
        if not marker_cells:
            continue
        if len(marker_cells) != 1:
            raise AssertionError("each enclosed chamber must contain exactly one marker after mirroring")
        color = int(seeded[marker_cells[0]])
        out[comp[:, 0], comp[:, 1]] = color

    return out


def _add_left_room(walls: np.ndarray, mid: int, r0: int, h_in: int, w_in: int) -> None:
    r1 = r0 + h_in + 1
    c0 = mid - w_in - 1
    walls[r0, c0 : mid + 1] = True
    walls[r1, c0 : mid + 1] = True
    walls[r0 : r1 + 1, c0] = True

    base_area = h_in * w_in
    remaining = base_area - 4
    if remaining <= 0 or random.random() >= 0.7:
        return

    choices: list[tuple[str, int]] = []
    if h_in >= 2:
        choices.append(("top", min(h_in - 1, remaining)))
        choices.append(("bottom", min(h_in - 1, remaining)))
    if w_in >= 2:
        choices.append(("outer", min(w_in - 1, remaining)))
    choices = [(kind, limit) for kind, limit in choices if limit >= 1]
    if not choices:
        return

    kind, limit = random.choice(choices)
    length = random.randint(1, limit)
    c0 = mid - w_in - 1
    r1 = r0 + h_in + 1
    if kind == "top":
        col = random.randint(c0 + 1, mid - 1)
        walls[r0 + 1 : r0 + 1 + length, col] = True
    elif kind == "bottom":
        col = random.randint(c0 + 1, mid - 1)
        walls[r1 - length : r1, col] = True
    else:
        row = random.randint(r0 + 1, r1 - 1)
        walls[row, c0 + 1 : c0 + 1 + length] = True


def _paired_chambers(grid: np.ndarray, axis: str) -> list[tuple[np.ndarray, np.ndarray]] | None:
    chambers = [comp for comp in _components(grid == BG) if not _touches_border(comp, grid.shape)]
    if not chambers or len(chambers) % 2 != 0:
        return None

    index = {_coord_key(comp): comp for comp in chambers}
    used: set[frozenset[tuple[int, int]]] = set()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for comp in chambers:
        key = _coord_key(comp)
        if key in used:
            continue
        mirrored = _coord_key(_mirror_coords(comp, axis, grid.shape))
        mate = index.get(mirrored)
        if mate is None or mirrored == key:
            return None
        used.add(key)
        used.add(mirrored)
        if axis == "vertical":
            if int(comp[:, 1].mean()) < int(mate[:, 1].mean()):
                first, second = comp, mate
            else:
                first, second = mate, comp
        else:
            if int(comp[:, 0].mean()) < int(mate[:, 0].mean()):
                first, second = comp, mate
            else:
                first, second = mate, comp
        pairs.append((first, second))

    return pairs if len(used) == len(chambers) else None


def _build_vertical_case() -> tuple[np.ndarray, np.ndarray] | None:
    room_count = random.choice((2, 3, 3, 4))
    heights: list[int] = []
    widths: list[int] = []
    for _ in range(room_count):
        h_in = random.randint(2, 5 if room_count == 2 else 4)
        w_max = min(5, 20 // h_in)
        if w_max < 2:
            return None
        heights.append(h_in)
        widths.append(random.randint(2, w_max))

    min_mid = max(6, max(widths) + 2)
    if min_mid > 10:
        return None
    mid = random.randint(min_mid, 10)
    width = 2 * mid + 1

    room_heights = [h + 2 for h in heights]
    base_height = sum(room_heights) + room_count + 1
    height_choices = [h for h in range(max(13, base_height), 22) if h % 2 == 1]
    if not height_choices:
        return None
    height = random.choice(height_choices)

    gaps = [1] * (room_count + 1)
    for _ in range(height - base_height):
        gaps[random.randrange(len(gaps))] += 1

    left_walls = np.zeros((height, width), dtype=bool)
    left_walls[:, mid] = True

    r = gaps[0]
    for h_in, w_in, room_h in zip(heights, widths, room_heights):
        _add_left_room(left_walls, mid, r, h_in, w_in)
        r += room_h + gaps.pop(1)

    walls = left_walls | np.fliplr(left_walls)
    if len(_components(walls)) != 1:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[walls] = WALL
    pairs = _paired_chambers(grid, "vertical")
    if pairs is None or len(pairs) != room_count:
        return None
    if any(not (4 <= len(left) <= 20 and 4 <= len(right) <= 20) for left, right in pairs):
        return None
    try:
        if _detect_axis_from_walls(grid) != "vertical":
            return None
    except AssertionError:
        return None

    source_left = random.random() < 0.5
    marked_count = random.randint(1, room_count - 1)
    marked_ids = set(random.sample(range(room_count), marked_count))
    expected = grid.copy()
    for i, (left, right) in enumerate(pairs):
        if i not in marked_ids:
            continue
        src, dst = (left, right) if source_left else (right, left)
        marker = tuple(map(int, random.choice(src.tolist())))
        mr, mc = _mirror_point(marker[0], marker[1], "vertical", grid.shape)
        if (mr, mc) not in _coord_key(dst):
            return None
        color = random.choice(MARKER_COLORS)
        grid[marker] = color
        expected[left[:, 0], left[:, 1]] = color
        expected[right[:, 0], right[:, 1]] = color

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    built = _build_vertical_case()
    if built is None:
        return None
    grid, expected = built
    if random.random() < 0.5:
        grid = grid.T.copy()
        expected = expected.T.copy()

    if np.array_equal(grid, expected):
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
