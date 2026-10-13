from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LEGEND_BORDER = 5
FILTER_PALETTE = (2, 3, 4, 6)
CROSS_COLORS = (1, 8, 9)
GRID_SIZES = (13, 15, 17, 19)
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0

    cells = {tuple(map(int, rc)) for rc in coords}
    count = 0
    while cells:
        count += 1
        start = cells.pop()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells:
                    cells.remove((nr, nc))
                    queue.append((nr, nc))
    return count


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        full_rows = np.where(np.all(grid == color, axis=1))[0]
        full_cols = np.where(np.all(grid == color, axis=0))[0]
        for r in full_rows:
            for c in full_cols:
                candidates.append((color, int(r), int(c)))
    assert len(candidates) == 1, "expected exactly one full cross"
    color, r, c = candidates[0]
    assert h % 2 == 1 and w % 2 == 1, "crossed grids must have odd side lengths"
    assert r == h // 2 and c == w // 2, "cross must sit at the grid center"
    return color, r, c


def _legend_colors(grid: np.ndarray, cross_r: int, cross_c: int) -> set[int]:
    assert cross_r >= 3 and cross_c >= 3, "quadrant too small for the legend"
    patch = grid[:3, :3]
    assert np.all(patch[2, :] == LEGEND_BORDER), "legend bottom edge missing"
    assert np.all(patch[:, 2] == LEGEND_BORDER), "legend right edge missing"
    colors = set(map(int, np.unique(patch[:2, :2])))
    colors.discard(BG)
    colors.discard(LEGEND_BORDER)
    assert 1 <= len(colors) <= 2, "legend must encode one or two target colors"
    return colors


def _source_mask(grid: np.ndarray, cross_r: int, cross_c: int, cross_color: int) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    lt = grid[:cross_r, :cross_c]
    src = (lt != BG) & (lt != cross_color) & (lt != LEGEND_BORDER)
    src[:3, :3] = False
    mask[:cross_r, :cross_c] = src
    assert np.any(src), "expected a source pattern in the upper-left quadrant"
    return mask


def _quadrant_slice(name: str, cross_r: int, cross_c: int, shape: tuple[int, int]):
    h, w = shape
    if name == "lt":
        return slice(0, cross_r), slice(0, cross_c)
    if name == "rt":
        return slice(0, cross_r), slice(cross_c + 1, w)
    if name == "lb":
        return slice(cross_r + 1, h), slice(0, cross_c)
    if name == "rb":
        return slice(cross_r + 1, h), slice(cross_c + 1, w)
    raise ValueError(f"unknown quadrant {name}")


def _quadrant_colors(
    grid: np.ndarray, name: str, cross_r: int, cross_c: int, cross_color: int
) -> set[int]:
    rs, cs = _quadrant_slice(name, cross_r, cross_c, grid.shape)
    region = grid[rs, cs]
    colors = set(map(int, np.unique(region)))
    colors.discard(BG)
    colors.discard(cross_color)
    colors.discard(LEGEND_BORDER)
    return colors


def _mirror_coords(
    coords: list[tuple[int, int]], cross_r: int, cross_c: int, flip_v: bool, flip_h: bool
) -> list[tuple[int, int]]:
    mirrored = []
    for r, c in coords:
        rr = 2 * cross_r - r if flip_v else r
        cc = 2 * cross_c - c if flip_h else c
        mirrored.append((rr, cc))
    return mirrored


def _expected_from_recipe(
    shape: tuple[int, int],
    source_coords: list[tuple[int, int]],
    cross_r: int,
    cross_c: int,
    cross_color: int,
    pass_flags: dict[str, bool],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in source_coords:
        out[r, c] = cross_color

    for name, (flip_v, flip_h) in {
        "rt": (False, True),
        "lb": (True, False),
        "rb": (True, True),
    }.items():
        if not pass_flags[name]:
            continue
        for r, c in _mirror_coords(source_coords, cross_r, cross_c, flip_v, flip_h):
            out[r, c] = cross_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the legend colors to choose which quadrants receive a cross-colored mirror of the upper-left pattern.

    Input structure:
    - The background is black, and one full horizontal line plus one full vertical line of the same color form a centered cross.
    - The top-left corner contains a 3x3 legend whose bottom edge and right edge are gray and whose 2x2 interior shows one or two target colors.
    - The upper-left quadrant also contains one non-cross foreground pattern away from the legend, while the other three quadrants contain scattered single-pixel color markers.

    Rule:
    - Read the legend interior colors as the target set.
    - Extract the non-black, non-cross, non-gray pattern mask from the upper-left quadrant, excluding the legend.
    - Erase the cross and legend, repaint the upper-left mask with the cross color, and mirror that mask into each other quadrant whose marker-color set contains every target color.
    - Leave quadrants that fail the color-set test empty.

    Key insight:
    The legend does not select objects directly; it filters whole quadrants, and only the quadrants that contain the full target palette participate in the four-way mirror structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_color, cross_r, cross_c = _find_cross(g)
    targets = _legend_colors(g, cross_r, cross_c)
    source = _source_mask(g, cross_r, cross_c, cross_color)
    source_coords = [tuple(map(int, rc)) for rc in np.argwhere(source)]

    pass_flags = {
        name: targets.issubset(_quadrant_colors(g, name, cross_r, cross_c, cross_color))
        for name in ("rt", "lb", "rb")
    }
    return _expected_from_recipe(g.shape, source_coords, cross_r, cross_c, cross_color, pass_flags)


def _sample_target_colors() -> tuple[int, ...]:
    size = 2 if random.random() < 0.7 else 1
    return tuple(random.sample(FILTER_PALETTE, size))


def _grow_connected_shape(
    allowed: list[tuple[int, int]], q: int, size: int
) -> list[tuple[int, int]] | None:
    allowed_set = set(allowed)
    start = random.choice(allowed)
    cells = {start}
    frontier = {nbr for nbr in _neighbors4(*start) if nbr in allowed_set}

    while len(cells) < size:
        if not frontier:
            return None
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nbr in _neighbors4(*nxt):
            if nbr in allowed_set and nbr not in cells:
                frontier.add(nbr)

    mask = np.zeros((q, q), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.flipud(np.fliplr(mask)))
    ):
        return None
    return sorted(cells)


def _paint_legend(grid: np.ndarray, targets: tuple[int, ...]) -> None:
    grid[:3, :3] = BG
    grid[2, :3] = LEGEND_BORDER
    grid[:3, 2] = LEGEND_BORDER

    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fills = [random.choice(targets) for _ in cells]
    for i, color in enumerate(targets):
        fills[i] = color
    random.shuffle(fills)
    for (r, c), color in zip(cells, fills):
        grid[r, c] = int(color)


def _paint_source_pattern(
    grid: np.ndarray, q: int, targets: tuple[int, ...]
) -> list[tuple[int, int]] | None:
    allowed = [(r, c) for r in range(3, q) for c in range(3, q)]
    if len(allowed) < 6:
        return None

    for _ in range(80):
        size = random.randint(3, 6)
        coords = _grow_connected_shape(allowed, q, size)
        if coords is None:
            continue

        if len(targets) == 1 and random.random() < 0.5:
            colors = [targets[0], random.choice([c for c in FILTER_PALETTE if c != targets[0]])]
        else:
            colors = list(targets)

        assigned = [random.choice(colors) for _ in coords]
        for idx, color in enumerate(targets):
            assigned[idx] = color
        random.shuffle(assigned)

        for (r, c), color in zip(coords, assigned):
            grid[r, c] = int(color)
        return coords
    return None


def _target_positions_for_quadrant(
    source_coords: list[tuple[int, int]], cross_r: int, cross_c: int, name: str
) -> set[tuple[int, int]]:
    flips = {"rt": (False, True), "lb": (True, False), "rb": (True, True)}
    flip_v, flip_h = flips[name]
    return set(_mirror_coords(source_coords, cross_r, cross_c, flip_v, flip_h))


def _marker_color_set(targets: tuple[int, ...], should_pass: bool) -> tuple[int, ...]:
    targets_set = set(targets)
    extras = [c for c in FILTER_PALETTE if c not in targets_set]

    if should_pass:
        chosen = set(targets)
        if extras and random.random() < 0.55 and len(chosen) < 3:
            chosen.add(random.choice(extras))
        return tuple(sorted(chosen))

    if len(targets) == 1:
        return tuple(sorted(random.sample(extras, random.randint(1, min(3, len(extras))))))

    missing = random.choice(targets)
    available = [c for c in FILTER_PALETTE if c != missing]
    count = random.randint(1, min(3, len(available)))
    chosen = set(random.sample(available, count))
    chosen.discard(missing)
    if targets_set.issubset(chosen):
        return tuple(sorted(c for c in chosen if c != missing))
    return tuple(sorted(chosen))


def _spaced_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _paint_quadrant_markers(
    grid: np.ndarray,
    q: int,
    cross_r: int,
    cross_c: int,
    source_coords: list[tuple[int, int]],
    name: str,
    color_set: tuple[int, ...],
) -> bool:
    rs, cs = _quadrant_slice(name, cross_r, cross_c, grid.shape)
    forbidden = _target_positions_for_quadrant(source_coords, cross_r, cross_c, name)
    cells = [
        (r, c)
        for r in range(rs.start, rs.stop)
        for c in range(cs.start, cs.stop)
        if (r, c) not in forbidden
    ]
    marker_count = random.randint(len(color_set), 3)
    positions = _spaced_positions(cells, marker_count)
    if positions is None:
        return False

    colors = list(color_set)
    while len(colors) < marker_count:
        colors.append(random.choice(color_set))
    random.shuffle(colors)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = int(color)
    return True


def _build_expected(
    grid: np.ndarray,
    source_coords: list[tuple[int, int]],
    cross_r: int,
    cross_c: int,
    cross_color: int,
    targets: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, bool]]:
    pass_flags = {
        name: set(targets).issubset(_quadrant_colors(grid, name, cross_r, cross_c, cross_color))
        for name in ("rt", "lb", "rb")
    }
    expected = _expected_from_recipe(grid.shape, source_coords, cross_r, cross_c, cross_color, pass_flags)
    return expected, pass_flags


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    source_coords: list[tuple[int, int]],
    targets: tuple[int, ...],
    pass_flags: dict[str, bool],
    cross_color: int,
) -> bool:
    if grid.shape != expected.shape or max(grid.shape) > 30:
        return False

    cross2, cross_r, cross_c = _find_cross(grid)
    if cross2 != cross_color:
        return False
    if _legend_colors(grid, cross_r, cross_c) != set(targets):
        return False

    src_mask = _source_mask(grid, cross_r, cross_c, cross_color)
    src_coords2 = [tuple(map(int, rc)) for rc in np.argwhere(src_mask)]
    if sorted(source_coords) != sorted(src_coords2):
        return False
    if not (3 <= len(src_coords2) <= 6):
        return False
    if _component_count(src_mask) != 1:
        return False

    source_colors = {int(grid[r, c]) for r, c in src_coords2}
    if not set(targets).issubset(source_colors):
        return False

    if not (any(pass_flags.values()) and not all(pass_flags.values())):
        return False

    for name in ("rt", "lb", "rb"):
        rs, cs = _quadrant_slice(name, cross_r, cross_c, grid.shape)
        coords = [
            tuple(map(int, rc))
            for rc in np.argwhere(
                (grid[rs, cs] != BG) & (grid[rs, cs] != cross_color) & (grid[rs, cs] != LEGEND_BORDER)
            )
        ]
        if not (1 <= len(coords) <= 3):
            return False
        for i, (r0, c0) in enumerate(coords):
            for r1, c1 in coords[i + 1 :]:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    return False
        if set(targets).issubset(_quadrant_colors(grid, name, cross_r, cross_c, cross_color)) != pass_flags[name]:
            return False

    if np.any(expected[cross_r, :] != BG) or np.any(expected[:, cross_c] != BG):
        return False
    if np.any(expected[:3, :3] == LEGEND_BORDER) or np.any(np.isin(expected[:2, :2], targets)):
        return False

    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    cross_color = random.choice(CROSS_COLORS)
    cross_r = cross_c = q
    targets = _sample_target_colors()
    pass_pattern = random.choice(
        [
            {"rt": True, "lb": False, "rb": False},
            {"rt": False, "lb": True, "rb": False},
            {"rt": False, "lb": False, "rb": True},
            {"rt": True, "lb": True, "rb": False},
            {"rt": True, "lb": False, "rb": True},
            {"rt": False, "lb": True, "rb": True},
        ]
    )

    grid = np.zeros((n, n), dtype=int)
    grid[cross_r, :] = cross_color
    grid[:, cross_c] = cross_color
    _paint_legend(grid, targets)

    source_coords_local = _paint_source_pattern(grid, q, targets)
    if source_coords_local is None:
        return None
    source_coords = [(r, c) for r, c in source_coords_local]

    for name in ("rt", "lb", "rb"):
        color_set = _marker_color_set(targets, pass_pattern[name])
        if not color_set:
            return None
        if not _paint_quadrant_markers(grid, q, cross_r, cross_c, source_coords, name, color_set):
            return None

    expected, pass_flags = _build_expected(grid, source_coords, cross_r, cross_c, cross_color, targets)
    if pass_flags != pass_pattern:
        return None
    if not _validate_instance(grid, expected, source_coords, targets, pass_flags, cross_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-filtered quadrant mirror input")
