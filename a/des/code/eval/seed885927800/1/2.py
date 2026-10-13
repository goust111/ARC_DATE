from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MAX_RETRIES = 300
PALETTE = list(range(1, 10))

AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(2, 6):
    for _w in range(2, 7):
        _area = _h * _w
        if 6 <= _area <= 18:
            AREA_TO_DIMS.setdefault(_area, []).append((_h, _w))
AREAS = sorted(AREA_TO_DIMS)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    body = g[:-1].copy()
    legend_row = g[-1]
    legend_colors = [int(legend_row[c]) for c in np.flatnonzero(legend_row != 0)]
    assert legend_colors
    assert len(legend_colors) == len(set(legend_colors))

    fragment_colors = sorted(set(map(int, np.unique(body))) - {0})
    assert len(fragment_colors) == len(legend_colors)

    # If same-color pieces can be translated to tile one solid rectangle,
    # that rectangle's area is exactly the total number of cells of that color.
    areas = {color: int(np.count_nonzero(body == color)) for color in fragment_colors}
    assert len(set(areas.values())) == len(areas)

    ranked_colors = sorted(fragment_colors, key=lambda color: areas[color])
    out = body.copy()
    for src, dst in zip(ranked_colors, legend_colors):
        out[body == src] = int(dst)
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_quotas(total: int, parts: int) -> list[int]:
    min_piece = 2 if total >= 2 * parts else 1
    quotas = [min_piece] * parts
    for _ in range(total - min_piece * parts):
        quotas[random.randrange(parts)] += 1
    random.shuffle(quotas)
    return quotas


def _partition_rectangle(h: int, w: int, parts: int) -> list[list[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    area = h * w
    for _ in range(200):
        quotas = _random_quotas(area, parts)
        seeds = random.sample(cells, parts)
        regions = [{seed} for seed in seeds]
        owner = {seed: i for i, seed in enumerate(seeds)}

        while len(owner) < area:
            choices: list[tuple[int, list[tuple[int, int]]]] = []
            weights: list[int] = []
            for i, region in enumerate(regions):
                if len(region) >= quotas[i]:
                    continue
                frontier = set()
                for r, c in region:
                    for nr, nc in _neighbors4(r, c, h, w):
                        if (nr, nc) not in owner:
                            frontier.add((nr, nc))
                if frontier:
                    choices.append((i, list(frontier)))
                    weights.append(quotas[i] - len(region))
            if not choices:
                break
            idx = random.choices(range(len(choices)), weights=weights, k=1)[0]
            region_id, frontier = choices[idx]
            pick = random.choice(frontier)
            owner[pick] = region_id
            regions[region_id].add(pick)

        if len(owner) != area:
            continue
        if any(len(region) != quota for region, quota in zip(regions, quotas)):
            continue
        return [sorted(region) for region in regions]
    return None


def _cells_to_patch(cells: list[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _place_patch(body: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = body.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    patch_mask = patch != 0

    for r0, c0 in positions:
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + pw + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        window = body[r0 : r0 + ph, c0 : c0 + pw]
        window[patch_mask] = patch[patch_mask]
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _add_legend_row(grid: np.ndarray, legend_colors: list[int]) -> bool:
    w = grid.shape[1]
    gaps = [random.randint(1, 3) for _ in range(len(legend_colors) - 1)]
    span = len(legend_colors) + sum(gaps)
    if span > w:
        return False
    c = random.randint(0, w - span)
    for i, color in enumerate(legend_colors):
        grid[-1, c] = int(color)
        c += 1
        if i < len(gaps):
            c += gaps[i]
    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    body = grid[:-1].copy()
    legend_colors = [int(grid[-1, c]) for c in np.flatnonzero(grid[-1] != 0)]
    body_colors = sorted(set(map(int, np.unique(body))) - {0})
    counts = {color: int(np.count_nonzero(body == color)) for color in body_colors}
    ranked = sorted(body_colors, key=lambda color: counts[color])
    for src, dst in zip(ranked, legend_colors):
        body[grid[:-1] == src] = int(dst)
    return body


def _validate_generated(
    grid: np.ndarray,
    area_by_color: dict[int, int],
    pieces_by_color: dict[int, int],
    legend_colors: list[int],
) -> bool:
    body = grid[:-1]
    legend = [int(grid[-1, c]) for c in np.flatnonzero(grid[-1] != 0)]
    if legend != [int(c) for c in legend_colors]:
        return False

    body_colors = sorted(set(map(int, np.unique(body))) - {0})
    if set(body_colors) != set(area_by_color):
        return False
    if set(body_colors) & set(legend_colors):
        return False
    if len(set(area_by_color.values())) != len(area_by_color):
        return False

    for color in body_colors:
        if int(np.count_nonzero(body == color)) != int(area_by_color[color]):
            return False
        if len(_components(body == color, connectivity=8)) != int(pieces_by_color[color]):
            return False
        if pieces_by_color[color] < 2:
            return False

    out = main(grid)
    if out.shape != body.shape:
        return False
    return np.array_equal(out, _expected_output(grid))


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(2, 4)
    chosen = random.sample(PALETTE, 2 * color_count)
    fragment_colors = chosen[:color_count]
    legend_colors = chosen[color_count:]
    areas = random.sample(AREAS, color_count)

    pieces: list[np.ndarray] = []
    area_by_color: dict[int, int] = {}
    pieces_by_color: dict[int, int] = {}

    for color, area in zip(fragment_colors, areas):
        h, w = random.choice(AREA_TO_DIMS[area])
        parts = random.randint(2, min(4, area // 2))
        regions = _partition_rectangle(h, w, parts)
        if regions is None:
            return None
        area_by_color[int(color)] = int(area)
        pieces_by_color[int(color)] = int(parts)
        for region in regions:
            pieces.append(_cells_to_patch(region, int(color)))

    random.shuffle(pieces)
    pieces.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    piece_count = len(pieces)
    max_h = max(patch.shape[0] for patch in pieces)
    max_w = max(patch.shape[1] for patch in pieces)
    body_h_min = max(16, piece_count + 4, max_h + 1)
    body_w_min = max(18, piece_count + 6, max_w + 1)
    if body_h_min > 27 or body_w_min > 29:
        return None

    body_h = random.randint(body_h_min, 27)
    body_w = random.randint(body_w_min, 29)
    body = np.zeros((body_h, body_w), dtype=int)
    blocked = np.zeros_like(body, dtype=bool)

    for patch in pieces:
        if not _place_patch(body, blocked, patch):
            return None

    grid = np.zeros((body_h + 1, body_w), dtype=int)
    grid[:-1] = body
    if not _add_legend_row(grid, legend_colors):
        return None
    if not _validate_generated(grid, area_by_color, pieces_by_color, legend_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
