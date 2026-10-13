from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_SIZE_RANGE = (16, 30)
INNER_HEIGHT_RANGE = (3, 6)
INNER_WIDTH_RANGE = (3, 6)
EXTRA_REGION_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 300


def _outer_background_mask(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == bg and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == bg and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == bg:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _uniform_color(values: np.ndarray, bg: int = BG) -> int | None:
    if values.size == 0:
        return None
    color = int(values.flat[0])
    if color == bg or not bool(np.all(values == color)):
        return None
    return color


def _transform_kind(top: int, bottom: int, left: int, right: int) -> str:
    if top == bottom == left == right:
        return "rot180"
    if left == right and top != bottom:
        return "hflip"
    if top == bottom and left != right:
        return "vflip"
    return "identity"


def _apply_transform(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.fliplr(patch)
    if kind == "vflip":
        return np.flipud(patch)
    if kind == "rot180":
        return np.rot90(patch, 2)
    return patch.copy()


def _frame_info(
    grid: np.ndarray,
    outer_bg: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int = BG,
) -> tuple[int, int, int, int] | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    if bool(np.any(grid[r0, c0 : c1 + 1] == bg)):
        return None
    if bool(np.any(grid[r1, c0 : c1 + 1] == bg)):
        return None
    if bool(np.any(grid[r0 : r1 + 1, c0] == bg)):
        return None
    if bool(np.any(grid[r0 : r1 + 1, c1] == bg)):
        return None

    top = _uniform_color(grid[r0, c0 + 1 : c1], bg)
    bottom = _uniform_color(grid[r1, c0 + 1 : c1], bg)
    left = _uniform_color(grid[r0 + 1 : r1, c0], bg)
    right = _uniform_color(grid[r0 + 1 : r1, c1], bg)
    if None in {top, bottom, left, right}:
        return None

    if len({int(top), int(bottom), int(left), int(right)}) > 2:
        return None

    corners = (
        (int(grid[r0, c0]), {int(top), int(left)}),
        (int(grid[r0, c1]), {int(top), int(right)}),
        (int(grid[r1, c0]), {int(bottom), int(left)}),
        (int(grid[r1, c1]), {int(bottom), int(right)}),
    )
    if any(corner == bg or corner not in allowed for corner, allowed in corners):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not bool(np.any(interior != bg)):
        return None
    if bool(np.isin(interior, [int(top), int(bottom), int(left), int(right)]).any()):
        return None

    side_checks = []
    if r0 > 0:
        side_checks.append((grid[r0 - 1, c0 : c1 + 1], outer_bg[r0 - 1, c0 : c1 + 1]))
    if r1 + 1 < grid.shape[0]:
        side_checks.append((grid[r1 + 1, c0 : c1 + 1], outer_bg[r1 + 1, c0 : c1 + 1]))
    if c0 > 0:
        side_checks.append((grid[r0 : r1 + 1, c0 - 1], outer_bg[r0 : r1 + 1, c0 - 1]))
    if c1 + 1 < grid.shape[1]:
        side_checks.append((grid[r0 : r1 + 1, c1 + 1], outer_bg[r0 : r1 + 1, c1 + 1]))
    for values, mask in side_checks:
        if bool(np.any(values != bg)) or not bool(mask.all()):
            return None

    return int(top), int(bottom), int(left), int(right)


def _find_regions(grid: np.ndarray, bg: int = BG) -> list[tuple[int, int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    outer_bg = _outer_background_mask(g, bg)
    h, w = g.shape
    regions: list[tuple[int, int, int, int, int, int, int, int]] = []

    mask = g != bg
    seen = np.zeros((h, w), dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            queue: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            info = _frame_info(g, outer_bg, r0, c0, r1, c1, bg)
            if info is not None:
                top, bottom, left, right = info
                item = (r0, c0, r1, c1, top, bottom, left, right)
                if item not in regions:
                    regions.append(item)
    return sorted(regions, key=lambda item: (item[0], item[1], item[2], item[3]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each framed interior according to which opposite border sides share a color.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping rectangular regions separated by black gaps.
    - Each region has a one-pixel border whose four straight side segments are individually uniform in color, using at most two non-black colors overall.
    - Inside each border sits a non-black pattern that avoids the border colors; side colors are read from the edge segments between corners.

    Rule:
    - Detect every rectangular framed region and keep its border unchanged.
    - If all four side colors are identical, rotate the interior pattern by 180 degrees.
    - If left and right are the only matched opposite pair, horizontally mirror the interior.
    - If top and bottom are the only matched opposite pair, vertically mirror the interior.
    - Any other side-color arrangement leaves the interior unchanged.

    Key insight:
    - The colors themselves do not encode the action; only the pairing pattern of opposite sides matters, so the solver must read border geometry first and then choose the matching symmetry transform.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1, top, bottom, left, right in _find_regions(g, BG):
        kind = _transform_kind(top, bottom, left, right)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _apply_transform(interior, kind)
    return out


def _random_colors(count: int, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else set(map(int, exclude))
    pool = [color for color in PALETTE if color not in banned]
    return random.sample(pool, count)


def _sample_border(case: str) -> tuple[int, int, int, int]:
    if case == "rot180":
        color = _random_colors(1)[0]
        return color, color, color, color

    a, b = _random_colors(2)
    if case == "hflip":
        if random.random() < 0.5:
            return a, b, a, a
        return b, a, a, a
    if case == "vflip":
        if random.random() < 0.5:
            return a, a, a, b
        return a, a, b, a

    patterns = [
        (a, b, a, b),
        (a, b, b, a),
        (b, a, a, b),
        (b, a, b, a),
    ]
    return random.choice(patterns)


def _sample_interior(height: int, width: int, border_colors: set[int], case: str) -> np.ndarray:
    allowed = [color for color in PALETTE if color not in border_colors]
    area = height * width
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(400):
        patch = np.zeros((height, width), dtype=int)
        color_count = random.randint(1, min(3, len(allowed)))
        colors = random.sample(allowed, color_count)
        min_fill = min(area, max(3, area // 3))
        max_fill = area if area <= 4 else area - 1
        fill = random.randint(min_fill, max_fill)
        for r, c in random.sample(cells, fill):
            patch[r, c] = random.choice(colors)

        if len(set(map(int, patch.flat)) - {BG}) == 0:
            continue

        if case == "hflip" and bool(np.array_equal(patch, np.fliplr(patch))):
            continue
        if case == "vflip" and bool(np.array_equal(patch, np.flipud(patch))):
            continue
        if case == "rot180" and bool(np.array_equal(patch, np.rot90(patch, 2))):
            continue
        if case == "identity":
            if bool(np.array_equal(patch, np.fliplr(patch))):
                continue
            if bool(np.array_equal(patch, np.flipud(patch))):
                continue
            if bool(np.array_equal(patch, np.rot90(patch, 2))):
                continue
        return patch

    raise RuntimeError(f"failed to sample interior for {case}")


def _build_region(case: str) -> tuple[np.ndarray, np.ndarray]:
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)
    top, bottom, left, right = _sample_border(case)
    border_colors = {int(top), int(bottom), int(left), int(right)}
    interior = _sample_interior(inner_h, inner_w, border_colors, case)

    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = int(top)
    patch[-1, :] = int(bottom)
    patch[1:-1, 0] = int(left)
    patch[1:-1, -1] = int(right)
    patch[1:-1, 1:-1] = interior

    expected = patch.copy()
    expected[1:-1, 1:-1] = _apply_transform(interior, case)
    return patch, expected


def _place_regions(
    regions: list[tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = sorted(range(len(regions)), key=lambda idx: regions[idx][0].size, reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch, _expected = regions[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - ph):
            for c in range(1, width - pw):
                if bool(blocked[r - 1 : r + ph + 1, c - 1 : c + pw + 1].any()):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        r, c = placements[idx]
        blocked[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True

    for idx, (patch, expected) in enumerate(regions):
        r, c = placements[idx]
        ph, pw = patch.shape
        inp[r : r + ph, c : c + pw] = patch
        out[r : r + ph, c : c + pw] = expected

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    cases = ["hflip", "vflip", "rot180", "identity"]
    if random.random() < EXTRA_REGION_PROB:
        cases.append(random.choice(cases))
    random.shuffle(cases)

    try:
        regions = [_build_region(case) for case in cases]
    except RuntimeError:
        return None

    max_h = max(patch.shape[0] for patch, _ in regions)
    max_w = max(patch.shape[1] for patch, _ in regions)
    min_h = max(GRID_SIZE_RANGE[0], max_h + 2)
    min_w = max(GRID_SIZE_RANGE[0], max_w + 2)
    if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
        return None

    for _ in range(80):
        height = random.randint(min_h, GRID_SIZE_RANGE[1])
        width = random.randint(min_w, GRID_SIZE_RANGE[1])
        placed = _place_regions(regions, height, width)
        if placed is None:
            continue
        inp, expected = placed
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
