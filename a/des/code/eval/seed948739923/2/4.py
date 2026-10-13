from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
SPECIAL = {BG, GREY, MAROON}

ORTH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNER_RAYS = (
    (0, 0, -1, -1),
    (0, 1, -1, 1),
    (1, 0, 1, -1),
    (1, 1, 1, 1),
)

GEN_RETRIES = 500
PLACEMENT_RETRIES = 250
PRIORITY_RETRIES = 80


def _detect_layout(g: np.ndarray) -> tuple[str, int, tuple[int, int, int, int]]:
    h, w = g.shape
    candidates: list[tuple[str, int, tuple[int, int, int, int]]] = []

    if h >= 5 and np.all(g[1, :] == GREY) and not np.all(g[0, :] == GREY):
        candidates.append(("top", 0, (2, h - 2, 1, w - 2)))
    if h >= 5 and np.all(g[h - 2, :] == GREY) and not np.all(g[h - 1, :] == GREY):
        candidates.append(("bottom", h - 1, (1, h - 3, 1, w - 2)))
    if w >= 5 and np.all(g[:, 1] == GREY) and not np.all(g[:, 0] == GREY):
        candidates.append(("left", 0, (1, h - 2, 2, w - 2)))
    if w >= 5 and np.all(g[:, w - 2] == GREY) and not np.all(g[:, w - 1] == GREY):
        candidates.append(("right", w - 1, (1, h - 2, 1, w - 3)))

    assert len(candidates) == 1, "expected exactly one legend side outside the grey frame"
    side, legend_index, bounds = candidates[0]
    r0, r1, c0, c1 = bounds
    assert r0 <= r1 and c0 <= c1
    assert np.all(g[r0 - 1, c0 : c1 + 1] == GREY)
    assert np.all(g[r1 + 1, c0 : c1 + 1] == GREY)
    assert np.all(g[r0 : r1 + 1, c0 - 1] == GREY)
    assert np.all(g[r0 : r1 + 1, c1 + 1] == GREY)
    return side, legend_index, bounds


def _parse_priority(g: np.ndarray, side: str, legend_index: int) -> list[int]:
    line = g[legend_index, :] if side in {"top", "bottom"} else g[:, legend_index]

    colors: list[int] = []
    prev = None
    for value in line:
        color = int(value)
        if color in SPECIAL:
            prev = None
            continue
        if color != prev:
            colors.append(color)
            prev = color

    assert len(colors) >= 2, "legend must contain at least two colors"
    assert len(colors) == len(set(colors)), "legend colors must appear once each in priority order"
    return colors


def _is_tower_block(g: np.ndarray, bounds: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, r1, c0, c1 = bounds
    if r + 1 > r1 or c + 1 > c1:
        return False

    color = int(g[r, c])
    if color in SPECIAL:
        return False

    block = g[r : r + 2, c : c + 2]
    if block.shape != (2, 2) or not np.all(block == color):
        return False

    for rr in range(r, r + 2):
        for cc in range(c, c + 2):
            for dr, dc in ORTH4:
                nr, nc = rr + dr, cc + dc
                if r <= nr <= r + 1 and c <= nc <= c + 1:
                    continue
                if r0 <= nr <= r1 and c0 <= nc <= c1 and int(g[nr, nc]) == color:
                    return False
    return True


def _find_towers(g: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    r0, r1, c0, c1 = bounds
    tower_mask = np.zeros(g.shape, dtype=bool)
    towers: list[tuple[int, int, int]] = []

    for r in range(r0, r1):
        for c in range(c0, c1):
            if not _is_tower_block(g, bounds, r, c):
                continue
            if tower_mask[r : r + 2, c : c + 2].any():
                continue
            color = int(g[r, c])
            towers.append((r, c, color))
            tower_mask[r : r + 2, c : c + 2] = True

    assert towers, "expected at least one 2x2 signal tower"
    colors = [color for _r, _c, color in towers]
    assert len(colors) == len(set(colors)), "tower colors must be unique"
    return towers, tower_mask


def _trace_ray_colors(
    g: np.ndarray,
    bounds: tuple[int, int, int, int],
    towers: list[tuple[int, int, int]],
    tower_mask: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    r0, r1, c0, c1 = bounds
    contributors: dict[tuple[int, int], set[int]] = {}

    for tr, tc, color in towers:
        for corner_dr, corner_dc, dr, dc in CORNER_RAYS:
            r = tr + corner_dr + dr
            c = tc + corner_dc + dc
            while r0 <= r <= r1 and c0 <= c <= c1:
                if tower_mask[r, c]:
                    break
                value = int(g[r, c])
                if value not in {color, MAROON}:
                    break
                contributors.setdefault((r, c), set()).add(color)
                r += dr
                c += dc
    return contributors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each maroon ray intersection by taking the highest-priority color from the outside legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    side, legend_index, bounds = _detect_layout(g)
    priority = _parse_priority(g, side, legend_index)
    rank = {color: idx for idx, color in enumerate(priority)}

    towers, tower_mask = _find_towers(g, bounds)
    contributors = _trace_ray_colors(g, bounds, towers, tower_mask)

    out = g.copy()
    r0, r1, c0, c1 = bounds
    maroons = np.argwhere(g[r0 : r1 + 1, c0 : c1 + 1] == MAROON)
    for rr, cc in maroons:
        r = int(r0 + rr)
        c = int(c0 + cc)
        colors = contributors.get((r, c), set())
        assert len(colors) >= 2, "each maroon pixel must be crossed by at least two ray colors"
        assert all(color in rank for color in colors), "legend must rank every ray color"
        out[r, c] = min(colors, key=lambda color: rank[color])

    return out


def _boxes_too_close(a: tuple[int, int], b: tuple[int, int], pad: int = 1) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + 1 + pad < br - pad
        or br + 1 + pad < ar - pad
        or ac + 1 + pad < bc - pad
        or bc + 1 + pad < ac - pad
    )


def _sample_towers(h: int, w: int, colors: list[int]) -> list[tuple[int, int, int]] | None:
    towers: list[tuple[int, int, int]] = []
    for color in colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(0, h - 2)
            c = random.randint(0, w - 2)
            if any(_boxes_too_close((r, c), (rr, cc)) for rr, cc, _other in towers):
                continue
            towers.append((r, c, color))
            placed = True
            break
        if not placed:
            return None
    return towers


def _simulate_rays(
    h: int,
    w: int,
    towers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], set[int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    tower_mask = np.zeros((h, w), dtype=bool)
    for r, c, _color in towers:
        tower_mask[r : r + 2, c : c + 2] = True

    contributors: dict[tuple[int, int], set[int]] = {}
    ray_meta: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tower_id, (tr, tc, color) in enumerate(towers):
        for ray_id, (corner_dr, corner_dc, dr, dc) in enumerate(CORNER_RAYS):
            slope = 1 if dr == dc else -1
            r = tr + corner_dr + dr
            c = tc + corner_dc + dc
            while 0 <= r < h and 0 <= c < w and not tower_mask[r, c]:
                contributors.setdefault((r, c), set()).add(color)
                ray_meta.setdefault((r, c), []).append((color, slope * (tower_id * 10 + ray_id + 1)))
                r += dr
                c += dc
    return tower_mask, contributors, ray_meta


def _render_interior(
    h: int,
    w: int,
    towers: list[tuple[int, int, int]],
    contributors: dict[tuple[int, int], set[int]],
    priority: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    rank = {color: idx for idx, color in enumerate(priority)}

    for r, c, color in towers:
        inp[r : r + 2, c : c + 2] = color
        out[r : r + 2, c : c + 2] = color

    for (r, c), colors in contributors.items():
        if len(colors) == 1:
            color = next(iter(colors))
            inp[r, c] = color
            out[r, c] = color
        else:
            inp[r, c] = MAROON
            out[r, c] = min(colors, key=lambda color: rank[color])

    return inp, out


def _legend_runs(span: int, count: int) -> list[tuple[int, int]] | None:
    lengths = [random.randint(1, 2) for _ in range(count)]
    needed = sum(lengths) + (count - 1)
    if needed > span:
        return None

    free = span - needed
    extras = [0] * (count + 1)
    for _ in range(free):
        extras[random.randrange(count + 1)] += 1

    runs: list[tuple[int, int]] = []
    pos = extras[0]
    for idx, length in enumerate(lengths):
        runs.append((pos, pos + length - 1))
        pos += length
        if idx + 1 < count:
            pos += 1 + extras[idx + 1]
    return runs


def _compose_grid(
    interior: np.ndarray,
    side: str,
    priority: list[int],
    runs: list[tuple[int, int]],
) -> np.ndarray:
    h, w = interior.shape

    if side == "top":
        grid = np.full((h + 3, w + 2), BG, dtype=int)
        grid[1, :] = GREY
        grid[-1, :] = GREY
        grid[1:, 0] = GREY
        grid[1:, -1] = GREY
        grid[2 : 2 + h, 1 : 1 + w] = interior
        for color, (a, b) in zip(priority, runs):
            grid[0, 1 + a : 1 + b + 1] = color
        return grid

    if side == "bottom":
        grid = np.full((h + 3, w + 2), BG, dtype=int)
        grid[0, :] = GREY
        grid[-2, :] = GREY
        grid[:-1, 0] = GREY
        grid[:-1, -1] = GREY
        grid[1 : 1 + h, 1 : 1 + w] = interior
        for color, (a, b) in zip(priority, runs):
            grid[-1, 1 + a : 1 + b + 1] = color
        return grid

    if side == "left":
        grid = np.full((h + 2, w + 3), BG, dtype=int)
        grid[0, 1:] = GREY
        grid[-1, 1:] = GREY
        grid[:, 1] = GREY
        grid[:, -1] = GREY
        grid[1 : 1 + h, 2 : 2 + w] = interior
        for color, (a, b) in zip(priority, runs):
            grid[1 + a : 1 + b + 1, 0] = color
        return grid

    assert side == "right"
    grid = np.full((h + 2, w + 3), BG, dtype=int)
    grid[0, :-1] = GREY
    grid[-1, :-1] = GREY
    grid[:, 0] = GREY
    grid[:, -2] = GREY
    grid[1 : 1 + h, 1 : 1 + w] = interior
    for color, (a, b) in zip(priority, runs):
        grid[1 + a : 1 + b + 1, -1] = color
    return grid


def _choose_priority(
    colors: list[int],
    maroon_sets: list[set[int]],
) -> list[int] | None:
    need_two_winners = len(maroon_sets) >= 2
    for _ in range(PRIORITY_RETRIES):
        order = random.sample(colors, len(colors))
        rank = {color: idx for idx, color in enumerate(order)}
        winners = [min(color_set, key=lambda color: rank[color]) for color_set in maroon_sets]
        if need_two_winners and len(set(winners)) < 2:
            continue
        return order
    return None


def _structurally_valid(
    h: int,
    w: int,
    towers: list[tuple[int, int, int]],
    contributors: dict[tuple[int, int], set[int]],
    ray_meta: dict[tuple[int, int], list[tuple[int, int]]],
) -> bool:
    coverage = len(contributors)
    if coverage < (h + w):
        return False
    if coverage > (h * w * 3) // 5:
        return False

    maroon_sets = [colors for colors in contributors.values() if len(colors) >= 2]
    if len(maroon_sets) < 2:
        return False
    if len(maroon_sets) > max(8, (h * w) // 5):
        return False

    for entries in ray_meta.values():
        colors = {color for color, _ray_tag in entries}
        if len(colors) < 2:
            continue
        slopes = {1 if ray_tag > 0 else -1 for _color, ray_tag in entries}
        if len(slopes) < 2:
            return False

    if len({frozenset(colors) for colors in maroon_sets}) < 2:
        return False

    ray_only = sum(1 for colors in contributors.values() if len(colors) == 1)
    return ray_only >= len(towers) * 2


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_towers = random.randint(3, 5)
    min_side = max(10, 2 * n_towers + 4)
    h = random.randint(min_side, 18)
    w = random.randint(min_side, 18)

    colors = random.sample(list(PALETTE), n_towers)
    towers = _sample_towers(h, w, colors)
    if towers is None:
        return None

    _tower_mask, contributors, ray_meta = _simulate_rays(h, w, towers)
    if not _structurally_valid(h, w, towers, contributors, ray_meta):
        return None

    maroon_sets = [colors for colors in contributors.values() if len(colors) >= 2]
    priority = _choose_priority(colors, maroon_sets)
    if priority is None:
        return None

    interior_in, interior_out = _render_interior(h, w, towers, contributors, priority)
    side = random.choice(("top", "bottom", "left", "right"))
    span = w if side in {"top", "bottom"} else h
    runs = _legend_runs(span, len(priority))
    if runs is None:
        return None
    inp = _compose_grid(interior_in, side, priority, runs)
    out = _compose_grid(interior_out, side, priority, runs)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_case()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
