from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CARGO = 9
ARROW_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    coord_set = set(coords)
    degrees = {}
    for cell in coords:
        degrees[cell] = sum((nbr in coord_set) for nbr in _neighbors4(*cell))
    degree_values = sorted(degrees.values())
    if degree_values != [1, 1, 2]:
        return None
    elbow = next(cell for cell, deg in degrees.items() if deg == 2)
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    return elbow, endpoints[0], endpoints[1]


def _scan_first_cargo(
    labels: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = labels.shape
    r = tip[0] + direction[0]
    c = tip[1] + direction[1]
    dist = 1
    while 0 <= r < h and 0 <= c < w:
        comp_id = int(labels[r, c])
        if comp_id >= 0:
            return comp_id, dist
        r += direction[0]
        c += direction[1]
        dist += 1
    return None


def _resolve_arrow(
    cargo_labels: np.ndarray, coords: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    parsed = _parse_arrow(coords)
    assert parsed is not None, "non-cargo colored components must be 3-pixel L arrows"
    elbow, a, b = parsed
    candidates = []
    for start, tip in ((a, b), (b, a)):
        direction = (elbow[0] - start[0], elbow[1] - start[1])
        hit = _scan_first_cargo(cargo_labels, tip, direction)
        if hit is None:
            continue
        cargo_id, distance = hit
        candidates.append((distance, start, tip, direction, cargo_id))
    assert candidates, "every arrow must see a cargo object in one candidate direction"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    _distance, start, tip, direction, cargo_id = candidates[0]
    return start, tip, direction, cargo_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the first maroon cargo seen by each L-arrow to the mirrored position behind the arrow
    and recolor the moved cargo with the arrow's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)

    cargo_components = _components(g == CARGO)
    cargo_labels = np.full(g.shape, -1, dtype=int)
    for comp_id, coords in enumerate(cargo_components):
        assert 3 <= len(coords) <= 6, "cargo objects must be small connected shapes"
        for r, c in coords:
            cargo_labels[r, c] = comp_id

    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {bg, CARGO}):
        for coords in _components(g == color):
            start, tip, _direction, cargo_id = _resolve_arrow(cargo_labels, coords)
            arrows.append(
                {
                    "color": color,
                    "start": start,
                    "tip": tip,
                    "cargo_id": cargo_id,
                }
            )

    out = g.copy()
    for cargo_id in {int(arrow["cargo_id"]) for arrow in arrows}:
        for r, c in cargo_components[cargo_id]:
            out[r, c] = bg

    for arrow in arrows:
        color = int(arrow["color"])
        start = tuple(int(v) for v in arrow["start"])
        tip = tuple(int(v) for v in arrow["tip"])
        cargo_id = int(arrow["cargo_id"])
        for r, c in cargo_components[cargo_id]:
            nr = start[0] - (r - tip[0])
            nc = start[1] - (c - tip[1])
            assert _in_bounds(g.shape, (nr, nc)), "moved cargo must stay in bounds"
            assert out[nr, nc] == bg, "moved cargo must land on background"
            out[nr, nc] = color
    return out


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r += direction[0]
        c += direction[1]
    return cells


def _pad8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    padded.add((rr, cc))
    return padded


def _too_close(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _random_forward_shape(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for fwd, side in cells:
            for nf, ns in ((fwd + 1, side), (fwd - 1, side), (fwd, side + 1), (fwd, side - 1)):
                if nf < 0 or nf > 4 or abs(ns) > 2 or (nf, ns) in cells:
                    continue
                frontier.add((nf, ns))
        if not frontier:
            return []
        weighted = []
        for cell in frontier:
            nf, ns = cell
            weight = 3 + nf - abs(ns)
            weighted.extend([cell] * max(1, weight))
        cells.add(rng.choice(weighted))
    if rng.random() < 0.5:
        cells = {(fwd, -side) for fwd, side in cells}
    return sorted(cells)


def _random_free_shape(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if abs(nr) > 2 or abs(nc) > 2 or (nr, nc) in cells:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return []
        cells.add(rng.choice(sorted(frontier)))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return normalized


def _apply_basis(
    origin: tuple[int, int], direction: tuple[int, int], perp: tuple[int, int], a: int, b: int
) -> tuple[int, int]:
    return (
        origin[0] + a * direction[0] + b * perp[0],
        origin[1] + a * direction[1] + b * perp[1],
    )


def _try_place_arrow_pair(
    rng: random.Random,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        direction = rng.choice(DIRS)
        perp = rng.choice(PERPS[direction])
        cargo_shape = _random_forward_shape(rng, rng.randint(3, 6))
        if not cargo_shape:
            continue
        gap = rng.randint(2, 6)
        start = (rng.randrange(h), rng.randrange(w))
        elbow = (start[0] + direction[0], start[1] + direction[1])
        tip = (elbow[0] + perp[0], elbow[1] + perp[1])
        arrow = {start, elbow, tip}
        source = {
            _apply_basis(tip, direction, perp, gap + fwd, side) for fwd, side in cargo_shape
        }
        dest = {
            (start[0] - (r - tip[0]), start[1] - (c - tip[1])) for r, c in source
        }

        all_cells = arrow | source | dest
        if not all(_in_bounds(shape, cell) for cell in all_cells):
            continue
        if any(cell in blocked for cell in all_cells):
            continue
        if _too_close(arrow, source) or _too_close(arrow, dest) or _too_close(source, dest):
            continue

        alt_direction = (-perp[0], -perp[1])
        intended_ray = set(_ray_cells(shape, (tip[0] + direction[0], tip[1] + direction[1]), direction))
        alt_ray = set(_ray_cells(shape, (start[0] + alt_direction[0], start[1] + alt_direction[1]), alt_direction))
        if any(cell in blocked for cell in (intended_ray - source) | alt_ray):
            continue
        if source & alt_ray:
            continue

        return {
            "direction": direction,
            "perp": perp,
            "start": start,
            "arrow": arrow,
            "source": source,
            "dest": dest,
            "intended_ray": intended_ray - source,
            "alt_ray": alt_ray,
        }
    return None


def _try_place_extra_cargo(
    rng: random.Random,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        cargo_shape = _random_free_shape(rng, rng.randint(3, 6))
        if not cargo_shape:
            continue
        anchor = (rng.randrange(h), rng.randrange(w))
        cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in cargo_shape}
        if not all(_in_bounds(shape, cell) for cell in cells):
            continue
        if any(cell in blocked for cell in cells):
            continue
        return cells
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 25)
    w = rng.randint(15, 25)
    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()

    pair_count = rng.randint(2, 4)
    extra_count = rng.choice([0, 1, 1, 2])
    arrow_colors = rng.sample(ARROW_COLORS, pair_count)

    pairs: list[dict[str, object]] = []
    for color in arrow_colors:
        pair = _try_place_arrow_pair(rng, grid.shape, blocked)
        if pair is None:
            return None
        pair["color"] = color
        pairs.append(pair)

        arrow = set(pair["arrow"])
        source = set(pair["source"])
        dest = set(pair["dest"])
        for r, c in arrow:
            grid[r, c] = color
        for r, c in source:
            grid[r, c] = CARGO

        blocked |= _pad8(arrow, grid.shape)
        blocked |= _pad8(source, grid.shape)
        blocked |= _pad8(dest, grid.shape)
        blocked |= set(pair["intended_ray"])
        blocked |= set(pair["alt_ray"])

    for _ in range(extra_count):
        extra = _try_place_extra_cargo(rng, grid.shape, blocked)
        if extra is None:
            return None
        for r, c in extra:
            grid[r, c] = CARGO
        blocked |= _pad8(extra, grid.shape)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
