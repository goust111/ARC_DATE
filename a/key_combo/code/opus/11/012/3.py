from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 5
TEMPLATE = 0
INDICATOR_COLORS = (1, 2, 3)
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VALID_SIDE_SETS = (
    ("up", "left"),
    ("up", "right"),
    ("down", "left"),
    ("down", "right"),
    ("up", "down", "left"),
    ("up", "down", "right"),
    ("up", "left", "right"),
    ("down", "left", "right"),
)
GRID_MIN = 13
GRID_MAX = 18
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_template(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == TEMPLATE)
    assert len(comps) == 1, "expected one black template component"
    return comps[0]


def _direction_from_centroid(indicator: tuple[int, int], centroid: tuple[float, float]) -> str:
    dr = indicator[0] - centroid[0]
    dc = indicator[1] - centroid[1]
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _path_overlay(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int, bool, int]]]]:
    template_coords = _extract_template(grid)
    template_set = {tuple(map(int, rc)) for rc in template_coords}
    centroid = (
        float(template_coords[:, 0].mean()),
        float(template_coords[:, 1].mean()),
    )

    indicators = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != BG) & (grid != TEMPLATE))
    ]
    assert 2 <= len(indicators) <= 3, "expected 2 or 3 indicator pixels"

    h, w = grid.shape
    best_k = np.full((h, w), 10**9, dtype=int)
    best_horizontal = np.zeros((h, w), dtype=bool)
    best_color = grid.copy()
    coverage: dict[tuple[int, int], list[tuple[int, int, bool, int]]] = {}

    for path_id, (ir, ic, color) in enumerate(indicators):
        side = _direction_from_centroid((ir, ic), centroid)
        dr, dc = SIDE_TO_DELTA[side]
        horizontal = dc != 0
        for k in range(1, max(h, w) + 1):
            visible = False
            for tr, tc in template_coords:
                nr = int(tr) + dr * k
                nc = int(tc) + dc * k
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                visible = True
                if (nr, nc) in template_set:
                    continue
                cell = (nr, nc)
                coverage.setdefault(cell, []).append((path_id, k, horizontal, color))
                if k < best_k[nr, nc] or (k == best_k[nr, nc] and horizontal and not best_horizontal[nr, nc]):
                    best_k[nr, nc] = k
                    best_horizontal[nr, nc] = horizontal
                    best_color[nr, nc] = color
            if not visible:
                break

    out = grid.copy()
    for (r, c), _covers in coverage.items():
        out[r, c] = best_color[r, c]
    for r, c in template_set:
        out[r, c] = TEMPLATE
    return out, coverage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a black template outward from colored side indicators and resolve overlaps by path distance.

    Input structure:
    - The grid uses gray background cells, one 4-connected black template object, and 2-3 colored single-pixel indicators.
    - Each indicator sits orthogonally adjacent to a different side of the template.
    - Indicator colors are the colors used for the outward copies.

    Rule:
    - Infer each indicator's direction from its position relative to the template centroid, snapping to up/down/left/right.
    - Repaint translated copies of the template outward along that direction, keeping the original template black.
    - When multiple directional paths cover the same pixel, keep the color from the smaller distance layer; if distances tie, horizontal paths win.

    Key insight:
    - The indicator is both a color key and a direction handle, and the overlap is resolved by comparing how far each competing path has traveled from the same source template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    out, _coverage = _path_overlay(g)
    return out


def _random_template_mask(box_h: int, box_w: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((box_h, box_w), dtype=bool)
        start = (random.randrange(box_h), random.randrange(box_w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (box_h, box_w):
            continue
        return mask
    return None


def _place_template(mask: np.ndarray, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    grid = np.full((height, width), BG, dtype=int)
    box_h, box_w = mask.shape
    r0 = random.randint(2, height - box_h - 3)
    c0 = random.randint(2, width - box_w - 3)
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[r0 + int(dr), c0 + int(dc)] = TEMPLATE
    return grid, (r0, c0)


def _indicator_for_side(template_coords: np.ndarray, side: str) -> tuple[int, int]:
    centroid = (
        float(template_coords[:, 0].mean()),
        float(template_coords[:, 1].mean()),
    )
    if side == "up":
        edge = [tuple(map(int, rc)) for rc in template_coords if int(rc[0]) == int(template_coords[:, 0].min())]
        base = min(edge, key=lambda p: abs(p[1] - centroid[1]))
        return base[0] - 1, base[1]
    if side == "down":
        edge = [tuple(map(int, rc)) for rc in template_coords if int(rc[0]) == int(template_coords[:, 0].max())]
        base = min(edge, key=lambda p: abs(p[1] - centroid[1]))
        return base[0] + 1, base[1]
    if side == "left":
        edge = [tuple(map(int, rc)) for rc in template_coords if int(rc[1]) == int(template_coords[:, 1].min())]
        base = min(edge, key=lambda p: abs(p[0] - centroid[0]))
        return base[0], base[1] - 1
    edge = [tuple(map(int, rc)) for rc in template_coords if int(rc[1]) == int(template_coords[:, 1].max())]
    base = min(edge, key=lambda p: abs(p[0] - centroid[0]))
    return base[0], base[1] + 1


def _coverage_stats(
    coverage: dict[tuple[int, int], list[tuple[int, int, bool, int]]]
) -> tuple[int, int, int]:
    overlap = 0
    tie = 0
    strict_distance = 0
    for covers in coverage.values():
        by_path = {}
        for path_id, k, horizontal, color in covers:
            prev = by_path.get(path_id)
            item = (k, horizontal, color)
            if prev is None or k < prev[0]:
                by_path[path_id] = item
        items = list(by_path.values())
        if len(items) < 2:
            continue
        overlap += 1
        ks = [k for k, _horizontal, _color in items]
        min_k = min(ks)
        min_items = [item for item in items if item[0] == min_k]
        if len(min_items) >= 2 and any(horizontal for _k, horizontal, _color in min_items) and any(
            not horizontal for _k, horizontal, _color in min_items
        ):
            tie += 1
        if any(k > min_k for k in ks):
            strict_distance += 1
    return overlap, tie, strict_distance


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    box_h, box_w = random.choice(((2, 3), (3, 2), (3, 3)))
    area_lo = max(4, box_h + box_w - 1)
    area_hi = min(7, box_h * box_w)
    area = random.randint(area_lo, area_hi)
    mask = _random_template_mask(box_h, box_w, area)
    if mask is None:
        return None

    grid, (r0, c0) = _place_template(mask, height, width)
    template_coords = np.argwhere(grid == TEMPLATE)

    sides = random.choice(VALID_SIDE_SETS)
    colors = random.sample(INDICATOR_COLORS, len(sides))
    used_cells = set()
    for side, color in zip(sides, colors):
        ir, ic = _indicator_for_side(template_coords, side)
        if not (0 <= ir < height and 0 <= ic < width):
            return None
        if (ir, ic) in used_cells or grid[ir, ic] != BG:
            return None
        if _direction_from_centroid((ir, ic), (float(template_coords[:, 0].mean()), float(template_coords[:, 1].mean()))) != side:
            return None
        grid[ir, ic] = color
        used_cells.add((ir, ic))

    out, coverage = _path_overlay(grid)
    overlap, tie, strict_distance = _coverage_stats(coverage)
    if overlap == 0 or tie == 0 or strict_distance == 0:
        return None

    painted = int(np.count_nonzero((out != BG) & (grid == BG)))
    if not (10 <= painted <= (height * width) // 2):
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
