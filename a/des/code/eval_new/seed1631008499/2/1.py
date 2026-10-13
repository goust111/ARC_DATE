from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
PATTERNS = [
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 2), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (1, 2), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (2, 0), (2, 1)),
]

GRID_MIN = 22
GRID_MAX = 30
TEMPLATE_BAND = 6
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_vertices(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i, (r0, c0) in enumerate(vertices):
        r1, c1 = vertices[(i + 1) % len(vertices)]
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                cells.add((r0, c))
        else:
            assert c0 == c1
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                cells.add((r, c0))
    return cells


def _interior_from_boundary(boundary: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = np.zeros((h, w), dtype=bool)
    for r, c in boundary:
        wall[r, c] = True
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    interior = set()
    for r in range(h):
        for c in range(w):
            if not wall[r, c] and not outside[r, c]:
                interior.add((r, c))
    return interior


def _find_templates(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for r in range(h - 2):
        for c in range(w - 2):
            sub = grid[r : r + 3, c : c + 3]
            key = int(sub[0, 0])
            if key in (BLACK, GREY, MAROON):
                continue
            if not np.all(np.isin(sub[1:, :], [BLACK, MAROON])):
                continue
            if not np.all(np.isin(sub[0, 1:], [BLACK, MAROON])):
                continue
            if not np.any(sub == MAROON):
                continue
            offsets = tuple(
                (rr - 1, cc - 1)
                for rr in range(3)
                for cc in range(3)
                if (rr, cc) != (0, 0) and int(sub[rr, cc]) == MAROON
            )
            if key in templates:
                assert templates[key] == offsets
            else:
                templates[key] = offsets
    return templates


def _classify_region(
    grid: np.ndarray, grey_comp: list[tuple[int, int]]
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(grey_comp)
    color_points: dict[int, list[tuple[int, int]]] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            val = int(grid[r, c])
            if val not in (BLACK, GREY, MAROON):
                color_points.setdefault(val, []).append((r, c))
    assert len(color_points) == 1
    color, pts = next(iter(color_points.items()))
    assert len(pts) == 2

    sr0 = max(0, r0 - 1)
    sc0 = max(0, c0 - 1)
    sr1 = min(grid.shape[0] - 1, r1 + 1)
    sc1 = min(grid.shape[1] - 1, c1 + 1)
    sub = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]

    for gap, seed in (pts, pts[::-1]):
        passable = sub == BLACK
        sl = (seed[0] - sr0, seed[1] - sc0)
        passable[sl] = True
        q = deque([sl])
        seen = {sl}
        touches_border = sl[0] in (0, passable.shape[0] - 1) or sl[1] in (0, passable.shape[1] - 1)
        while q and not touches_border:
            r, c = q.popleft()
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]):
                    continue
                if not passable[nr, nc] or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                if nr in (0, passable.shape[0] - 1) or nc in (0, passable.shape[1] - 1):
                    touches_border = True
                    break
                q.append((nr, nc))
        if not touches_border:
            return color, seed, gap
    raise AssertionError("failed to classify seed and gap")


def _ray_path(
    grid: np.ndarray, seed: tuple[int, int], gap: tuple[int, int], color: int
) -> list[tuple[int, int]]:
    h, w = grid.shape
    solutions: list[list[tuple[int, int]]] = []
    for dr, dc in DIRS4:
        r, c = seed
        path = [seed]
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            if (r, c) == gap and int(grid[r, c]) == color:
                path.append((r, c))
                solutions.append(path)
                break
            if int(grid[r, c]) != BLACK:
                break
            path.append((r, c))
    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow template-stamped vines from interior seeds toward the matching colored border gaps.

    Input structure:
    - The background is black.
    - Several grey one-pixel outlines form closed orthogonal regions, and each outline has exactly one colored gap pixel on its border.
    - Each region contains exactly one interior seed pixel whose color matches its gap pixel.
    - Separate 3x3 style templates appear elsewhere in the grid: their top-left pixel is a region color, and the other eight cells use only black and maroon.

    Rule:
    - For each region, find the 3x3 template whose top-left color matches the seed and gap color, and read the maroon cells as brush offsets relative to the template center.
    - From the seed, follow the unique straight cardinal ray that reaches the same-colored border gap; this seed-to-gap line is the vine trunk.
    - Stamp the brush at every trunk position using the seed color, but only onto black cells, leaving grey borders, templates, seeds, and gaps unchanged.

    Key insight:
    - Color links three separate roles at once: the seed matches a unique border gap, and that same color indexes a template whose maroon mask is repeatedly printed along the straight growth path.
    """
    g = np.asarray(grid, dtype=int)
    templates = _find_templates(g)
    assert templates

    out = g.copy()
    for comp in _components(g == GREY):
        color, seed, gap = _classify_region(g, comp)
        assert color in templates
        path = _ray_path(g, seed, gap, color)
        for r, c in path:
            for dr, dc in templates[color]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) == BLACK:
                    out[nr, nc] = color
    return out


def _random_interval(rng: random.Random, length: int) -> tuple[int, int, int]:
    start = rng.randint(1, length - 4)
    end = rng.randint(start + 1, length - 2)
    ext = rng.randint(1, 3)
    return start, end, ext


def _build_vertical_shape(rng: random.Random, direction: str) -> dict[str, object]:
    base_h = rng.randint(7, 10)
    base_w = rng.randint(6, 9)
    left = _random_interval(rng, base_h) if rng.random() < 0.55 else None
    right = _random_interval(rng, base_h) if rng.random() < 0.55 else None
    if left is None and right is None and rng.random() < 0.7:
        if rng.random() < 0.5:
            left = _random_interval(rng, base_h)
        else:
            right = _random_interval(rng, base_h)

    left_ext = 0 if left is None else left[2]
    right_ext = 0 if right is None else right[2]
    vertices = [(0, 0), (0, base_w - 1)]
    if right is None:
        vertices.append((base_h - 1, base_w - 1))
    else:
        a, b, ext = right
        vertices.extend(
            [
                (a, base_w - 1),
                (a, base_w - 1 + ext),
                (b, base_w - 1 + ext),
                (b, base_w - 1),
                (base_h - 1, base_w - 1),
            ]
        )
    vertices.append((base_h - 1, 0))
    if left is None:
        vertices.append((0, 0))
    else:
        a, b, ext = left
        vertices.extend([(b, 0), (b, -ext), (a, -ext), (a, 0), (0, 0)])

    shifted = [(r, c + left_ext) for r, c in vertices]
    boundary = _draw_vertices(shifted)
    h = max(r for r, _ in boundary) + 1
    w = max(c for _, c in boundary) + 1
    interior = _interior_from_boundary(boundary, h, w)

    gap_col = left_ext + rng.randint(2, base_w - 3)
    if direction == "up":
        gap = (0, gap_col)
        seed = (rng.randint(2, base_h - 2), gap_col)
    else:
        gap = (base_h - 1, gap_col)
        seed = (rng.randint(1, base_h - 3), gap_col)

    assert seed in interior
    step = 1 if direction == "up" else -1
    for r in range(seed[0] - step, gap[0], -step):
        assert (r, gap_col) in interior
    return {"boundary": boundary, "interior": interior, "gap": gap, "seed": seed, "shape": (h, w)}


def _build_horizontal_shape(rng: random.Random, direction: str) -> dict[str, object]:
    base_h = rng.randint(6, 9)
    base_w = rng.randint(7, 10)
    top = _random_interval(rng, base_w) if rng.random() < 0.55 else None
    bottom = _random_interval(rng, base_w) if rng.random() < 0.55 else None
    if top is None and bottom is None and rng.random() < 0.7:
        if rng.random() < 0.5:
            top = _random_interval(rng, base_w)
        else:
            bottom = _random_interval(rng, base_w)

    top_ext = 0 if top is None else top[2]
    bottom_ext = 0 if bottom is None else bottom[2]
    vertices = [(0, 0)]
    if top is None:
        vertices.append((0, base_w - 1))
    else:
        a, b, ext = top
        vertices.extend([(0, a), (-ext, a), (-ext, b), (0, b), (0, base_w - 1)])
    vertices.append((base_h - 1, base_w - 1))
    if bottom is None:
        vertices.append((base_h - 1, 0))
    else:
        a, b, ext = bottom
        vertices.extend(
            [
                (base_h - 1, b),
                (base_h - 1 + ext, b),
                (base_h - 1 + ext, a),
                (base_h - 1, a),
                (base_h - 1, 0),
            ]
        )
    vertices.append((0, 0))

    shifted = [(r + top_ext, c) for r, c in vertices]
    boundary = _draw_vertices(shifted)
    h = max(r for r, _ in boundary) + 1
    w = max(c for _, c in boundary) + 1
    interior = _interior_from_boundary(boundary, h, w)

    gap_row = top_ext + rng.randint(2, base_h - 3)
    if direction == "left":
        gap = (gap_row, 0)
        seed = (gap_row, rng.randint(2, base_w - 2))
    else:
        gap = (gap_row, base_w - 1)
        seed = (gap_row, rng.randint(1, base_w - 3))

    assert seed in interior
    step = 1 if direction == "left" else -1
    for c in range(seed[1] - step, gap[1], -step):
        assert (gap_row, c) in interior
    return {"boundary": boundary, "interior": interior, "gap": gap, "seed": seed, "shape": (h, w)}


def _build_shape(rng: random.Random, direction: str) -> dict[str, object]:
    if direction in ("up", "down"):
        return _build_vertical_shape(rng, direction)
    return _build_horizontal_shape(rng, direction)


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _place_templates(
    grid: np.ndarray, blocked: np.ndarray, rng: random.Random, colors: list[int], patterns: list[tuple[tuple[int, int], ...]]
) -> bool:
    order = list(range(len(colors)))
    rng.shuffle(order)
    for idx in order:
        color = colors[idx]
        maroons = set(patterns[idx])
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = rng.randint(1, TEMPLATE_BAND - 3)
            left = rng.randint(1, grid.shape[1] - 4)
            if not _can_place(blocked, top, left, 3, 3, 1):
                continue
            for dr in range(3):
                for dc in range(3):
                    rr, cc = top + dr, left + dc
                    if (dr, dc) == (0, 0):
                        grid[rr, cc] = color
                    elif (dr, dc) in maroons:
                        grid[rr, cc] = MAROON
                    else:
                        grid[rr, cc] = BLACK
            _mark(blocked, top, left, 3, 3, 1)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: random.Random,
    color: int,
    direction: str,
) -> bool:
    shape = _build_shape(rng, direction)
    h, w = shape["shape"]  # type: ignore[index]
    row_min = TEMPLATE_BAND + 2
    row_max = grid.shape[0] - h - 2
    col_min = 2
    col_max = grid.shape[1] - w - 2
    if row_min > row_max or col_min > col_max:
        return False

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(row_min, row_max)
        left = rng.randint(col_min, col_max)
        if not _can_place(blocked, top, left, h, w, 2):
            continue
        boundary = shape["boundary"]  # type: ignore[index]
        seed = shape["seed"]  # type: ignore[index]
        gap = shape["gap"]  # type: ignore[index]
        for r, c in boundary:
            rr, cc = top + r, left + c
            grid[rr, cc] = color if (r, c) == gap else GREY
        sr, sc = seed
        grid[top + sr, left + sc] = color
        _mark(blocked, top, left, h, w, 2)
        return True
    return False


def _validate_generated(grid: np.ndarray, expected_regions: int, colors: set[int]) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    templates = _find_templates(grid)
    assert set(templates) == colors
    comps = _components(grid == GREY)
    assert len(comps) == expected_regions
    parsed_colors = set()
    for comp in comps:
        color, seed, gap = _classify_region(grid, comp)
        parsed_colors.add(color)
        path = _ray_path(grid, seed, gap, color)
        assert len(path) >= 3
    assert parsed_colors == colors
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        grid = np.full((h, w), BLACK, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        blocked[: TEMPLATE_BAND + 1, :] = False

        max_regions = 3 if h >= 24 and w >= 24 else 2
        region_count = rng.randint(1, max_regions)
        colors = rng.sample(SEED_COLORS, region_count)
        patterns = rng.sample(PATTERNS, region_count)

        if not _place_templates(grid, blocked, rng, colors, patterns):
            continue

        directions = [rng.choice(["up", "down", "left", "right"]) for _ in range(region_count)]
        ok = True
        for color, direction in zip(colors, directions):
            if not _place_region(grid, blocked, rng, color, direction):
                ok = False
                break
        if not ok:
            continue

        try:
            _validate_generated(grid, region_count, set(colors))
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
