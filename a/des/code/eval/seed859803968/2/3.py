from __future__ import annotations

import random

import numpy as np


BG = 8
COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
DIRECTIONS = ("right", "left", "up", "down")
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
GRID_H_RANGE = (11, 21)
GRID_W_RANGE = (11, 21)
BODY_LEN_RANGE = (1, 6)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components8_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == int(bg):
                continue

            color = int(grid[r0, c0])
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not seen[nr, nc]
                            and int(grid[nr, nc]) == color
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            components.append((int(color), coords))

    return components


def _parse_arrow(
    coords: list[tuple[int, int]]
) -> tuple[str, int, tuple[int, int], tuple[int, int]]:
    coord_set = set((int(r), int(c)) for r, c in coords)
    area = len(coord_set)
    assert area >= 3, "each arrow must have at least one body pixel and two tip pixels"

    rows = [r for r, _ in coord_set]
    cols = [c for _, c in coord_set]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    body_len = int(area - 2)

    matches: list[tuple[str, int, tuple[int, int], tuple[int, int]]] = []

    if h == 3 and w == body_len + 1:
        mid_r = int(r0 + 1)

        right_cells = {(mid_r, c) for c in range(c0, c1)} | {(r0, c1), (r1, c1)}
        if coord_set == right_cells:
            matches.append(("right", body_len, (mid_r, c1 + 1), (0, 1)))

        left_cells = {(mid_r, c) for c in range(c0 + 1, c1 + 1)} | {(r0, c0), (r1, c0)}
        if coord_set == left_cells:
            matches.append(("left", body_len, (mid_r, c0 - 1), (0, -1)))

    if w == 3 and h == body_len + 1:
        mid_c = int(c0 + 1)

        down_cells = {(r, mid_c) for r in range(r0, r1)} | {(r1, c0), (r1, c1)}
        if coord_set == down_cells:
            matches.append(("down", body_len, (r1 + 1, mid_c), (1, 0)))

        up_cells = {(r, mid_c) for r in range(r0 + 1, r1 + 1)} | {(r0, c0), (r0, c1)}
        if coord_set == up_cells:
            matches.append(("up", body_len, (r0 - 1, mid_c), (-1, 0)))

    assert len(matches) == 1, "every component must be exactly one valid arrow"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect scattered arrows, then emit a same-color ray from each arrow's apex.

    The arrow body length L determines the ray length, while the diagonal tip
    placement determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    out = g.copy()
    for color, coords in _components8_by_color(g, bg):
        _direction, body_len, start, step = _parse_arrow(coords)
        sr, sc = start
        dr, dc = step
        for k in range(body_len):
            rr = int(sr + dr * k)
            cc = int(sc + dc * k)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)
            else:
                break

    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _cells_from_spec(
    direction: str,
    body_len: int,
    apex: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r, c = map(int, apex)

    if direction == "right":
        arrow = [(r, c - 2 - i) for i in range(body_len)] + [(r - 1, c - 1), (r + 1, c - 1)]
        ray = [(r, c + i) for i in range(body_len)]
    elif direction == "left":
        arrow = [(r, c + 2 + i) for i in range(body_len)] + [(r - 1, c + 1), (r + 1, c + 1)]
        ray = [(r, c - i) for i in range(body_len)]
    elif direction == "up":
        arrow = [(r + 2 + i, c) for i in range(body_len)] + [(r + 1, c - 1), (r + 1, c + 1)]
        ray = [(r - i, c) for i in range(body_len)]
    else:
        assert direction == "down"
        arrow = [(r - 2 - i, c) for i in range(body_len)] + [(r - 1, c - 1), (r - 1, c + 1)]
        ray = [(r + i, c) for i in range(body_len)]

    return arrow, ray


def _placement_ranges(
    h: int,
    w: int,
    direction: str,
    body_len: int,
    truncate: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == "right":
        row_range = (1, h - 2)
        if truncate:
            col_range = (max(body_len + 1, w - body_len + 1), w - 1)
        else:
            col_range = (body_len + 1, w - body_len)
    elif direction == "left":
        row_range = (1, h - 2)
        if truncate:
            col_range = (0, min(body_len - 2, w - body_len - 2))
        else:
            col_range = (body_len - 1, w - body_len - 2)
    elif direction == "up":
        col_range = (1, w - 2)
        if truncate:
            row_range = (0, min(body_len - 2, h - body_len - 2))
        else:
            row_range = (body_len - 1, h - body_len - 2)
    else:
        assert direction == "down"
        col_range = (1, w - 2)
        if truncate:
            row_range = (max(body_len + 1, h - body_len + 1), h - 1)
        else:
            row_range = (body_len + 1, h - body_len)

    if row_range[0] > row_range[1] or col_range[0] > col_range[1]:
        return None
    return row_range, col_range


def _candidate_lengths(h: int, w: int, direction: str, truncate: bool) -> list[int]:
    axis_limit = w - 2 if direction in ("right", "left") else h - 2
    max_len = min(BODY_LEN_RANGE[1], axis_limit)
    candidates: list[int] = []
    for body_len in range(BODY_LEN_RANGE[0], max_len + 1):
        if _placement_ranges(h, w, direction, body_len, truncate) is not None:
            candidates.append(int(body_len))
    return candidates


def _dilate8(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sample_directions(n_arrows: int) -> list[str]:
    dirs = [random.choice(["left", "right"]), random.choice(["up", "down"])]
    remaining = [d for d in DIRECTIONS if d not in dirs]
    random.shuffle(remaining)
    dirs.extend(remaining[: max(0, min(4, n_arrows) - 2)])
    while len(dirs) < n_arrows:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _sample_truncations(n_arrows: int) -> list[bool]:
    n_truncated = n_arrows // 2
    if n_arrows % 2 == 1 and random.random() < 0.5:
        n_truncated += 1
    flags = [True] * n_truncated + [False] * (n_arrows - n_truncated)
    random.shuffle(flags)
    return flags


def _paint_expected(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        direction = str(spec["direction"])
        body_len = int(spec["body_len"])
        apex = tuple(map(int, spec["apex"]))
        arrow, ray = _cells_from_spec(direction, body_len, apex)
        for r, c in arrow:
            out[r, c] = color
        for r, c in ray:
            if _in_bounds(shape, r, c):
                out[r, c] = color
            else:
                break
    return out


def _place_one(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    truncate: bool,
) -> dict[str, object] | None:
    h, w = grid.shape
    lengths = _candidate_lengths(h, w, direction, truncate)
    if not lengths:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        body_len = random.choice(lengths)
        ranges = _placement_ranges(h, w, direction, body_len, truncate)
        if ranges is None:
            continue
        row_range, col_range = ranges
        apex = (random.randint(*row_range), random.randint(*col_range))
        arrow, ray = _cells_from_spec(direction, body_len, apex)

        if not all(_in_bounds((h, w), r, c) for r, c in arrow):
            continue

        ray_in_bounds = [(int(r), int(c)) for r, c in ray if _in_bounds((h, w), r, c)]
        full_footprint = set((int(r), int(c)) for r, c in arrow) | set(ray_in_bounds)
        halo = _dilate8((h, w), full_footprint)
        if any(bool(blocked[r, c]) for r, c in halo):
            continue

        for r, c in arrow:
            grid[r, c] = int(color)
        for r, c in halo:
            blocked[r, c] = True

        return {
            "color": int(color),
            "direction": str(direction),
            "body_len": int(body_len),
            "apex": tuple(map(int, apex)),
        }

    return None


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_arrows = random.randint(2, 5)
    directions = _sample_directions(n_arrows)
    truncations = _sample_truncations(n_arrows)
    colors = random.sample(COLORS, n_arrows)

    specs: list[dict[str, object]] = []
    for color, direction, truncate in zip(colors, directions, truncations):
        spec = _place_one(grid, blocked, color, direction, truncate)
        if spec is None:
            return None
        specs.append(spec)

    expected = _paint_expected((h, w), specs)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    used_dirs = {str(spec["direction"]) for spec in specs}
    used_trunc = {
        len([p for p in _cells_from_spec(str(spec["direction"]), int(spec["body_len"]), tuple(map(int, spec["apex"])))[1] if _in_bounds((h, w), *p)])
        < int(spec["body_len"])
        for spec in specs
    }
    if len(used_dirs) < 2 or len(used_trunc) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
