from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(counts.argmax())])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _side_colors(g: np.ndarray) -> dict[str, int]:
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid must be large enough to expose four edge stripes"
    colors = {
        "top": _mode(g[0, 1:-1]),
        "bottom": _mode(g[h - 1, 1:-1]),
        "left": _mode(g[1:-1, 0]),
        "right": _mode(g[1:-1, w - 1]),
    }
    assert len(set(colors.values())) == 4, "expected four distinct edge stripe colors"
    return colors


def _nearest_side(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h, w = shape
    dists = {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }
    best = min(dists.values())
    winners = [side for side, dist in dists.items() if dist == best]
    assert len(winners) == 1, "each object must be uniquely closest to one edge stripe"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each interior object by the largest square not exceeding its area,
    anchored at the object's bounding-box top-left corner and colored by the
    nearest edge stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small"

    bg = _mode(g[1:-1, 1:-1])
    side_colors = _side_colors(g)

    mask = np.zeros((h, w), dtype=bool)
    mask[1:-1, 1:-1] = g[1:-1, 1:-1] != bg
    comps = _components4(mask)
    assert comps, "expected at least one interior foreground object"

    out = g.copy()
    for comp in comps:
        for r, c in comp:
            out[r, c] = bg

    for comp in comps:
        area = len(comp)
        side = _nearest_side(_bbox(comp), g.shape)
        k = int(math.isqrt(area))
        assert k >= 1, "object area must be positive"
        r0, c0, _r1, _c1 = _bbox(comp)
        out[r0 : r0 + k, c0 : c0 + k] = int(side_colors[side])

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
OBJECT_COLOR = 5
ALL_COLORS = tuple(range(10))
GRID_MIN = 16
GRID_MAX = 24
OBJECT_COUNT_RANGE = (2, 4)
SIDE_DIST_RANGE = (2, 4)
NEAREST_GAP = 2
OBJECT_GAP = 1
MAX_SIDE = 4
MAX_EXTRA = 6
SHAPE_ATTEMPTS = 80


def _paint_stripes(grid: np.ndarray, side_colors: dict[str, int]) -> None:
    h, w = grid.shape
    grid[0, :] = int(side_colors["top"])
    grid[h - 1, :] = int(side_colors["bottom"])
    grid[:, 0] = int(side_colors["left"])
    grid[:, w - 1] = int(side_colors["right"])


def _sample_palette() -> tuple[int, dict[str, int]]:
    bg_choices = [c for c in ALL_COLORS if c != OBJECT_COLOR]
    bg = random.choice(bg_choices)
    stripe_choices = [c for c in ALL_COLORS if c not in (bg, OBJECT_COLOR)]
    picked = random.sample(stripe_choices, 4)
    side_colors = {side: int(color) for side, color in zip(SIDES, picked, strict=False)}
    return int(bg), side_colors


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                cells.add((int(nr), int(nc)))
    return list(cells)


def _grow_shape(side: int, extra: int) -> np.ndarray | None:
    target = side * side + extra
    for _ in range(SHAPE_ATTEMPTS):
        if extra == 0:
            return np.ones((side, side), dtype=bool)

        add_h = random.randint(0, min(3, extra))
        add_w = random.randint(0, min(3, extra))
        if add_h == 0 and add_w == 0:
            continue

        h = side + add_h
        w = side + add_w
        if h * w - side * side < extra:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[:side, :side] = True
        while int(mask.sum()) < target:
            frontier = _frontier(mask)
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) != target:
            continue

        used_rows = np.where(mask.any(axis=1))[0]
        used_cols = np.where(mask.any(axis=0))[0]
        mask = mask[: int(used_rows[-1]) + 1, : int(used_cols[-1]) + 1]
        return mask
    return None


def _sample_object_mask() -> tuple[np.ndarray, int]:
    side = random.randint(1, MAX_SIDE)
    if random.random() < 0.5:
        extra = 0
    else:
        extra = random.randint(1, min(MAX_EXTRA, 2 * side + 1))
    mask = _grow_shape(side, extra)
    assert mask is not None, "shape growth failed"
    return mask, int(side)


def _side_distances(r0: int, c0: int, h: int, w: int, grid_shape: tuple[int, int]) -> dict[str, int]:
    gh, gw = grid_shape
    return {
        "top": int(r0),
        "bottom": int(gh - r0 - h),
        "left": int(c0),
        "right": int(gw - c0 - w),
    }


def _rect_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    gh, gw = occupied.shape
    rr0 = max(1, r0 - OBJECT_GAP)
    rr1 = min(gh - 1, r0 + h + OBJECT_GAP)
    cc0 = max(1, c0 - OBJECT_GAP)
    cc1 = min(gw - 1, c0 + w + OBJECT_GAP)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    gh, gw = occupied.shape
    rr0 = max(1, r0 - OBJECT_GAP)
    rr1 = min(gh - 1, r0 + h + OBJECT_GAP)
    cc0 = max(1, c0 - OBJECT_GAP)
    cc1 = min(gw - 1, c0 + w + OBJECT_GAP)
    occupied[rr0:rr1, cc0:cc1] = True


def _candidate_positions(
    grid_shape: tuple[int, int],
    box_shape: tuple[int, int],
    side: str,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    h, w = box_shape
    candidates = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            if not _rect_is_free(occupied, int(r0), int(c0), int(h), int(w)):
                continue
            dists = _side_distances(int(r0), int(c0), int(h), int(w), grid_shape)
            own = dists[side]
            other_best = min(dist for name, dist in dists.items() if name != side)
            if not (SIDE_DIST_RANGE[0] <= own <= SIDE_DIST_RANGE[1]):
                continue
            if own + NEAREST_GAP > other_best:
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _render_expected(
    grid: np.ndarray,
    placements: list[dict[str, object]],
    bg: int,
    side_colors: dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    for placed in placements:
        mask = placed["mask"]
        r0 = int(placed["r0"])
        c0 = int(placed["c0"])
        coords = np.argwhere(mask)
        for rr, cc in coords:
            out[r0 + int(rr), c0 + int(cc)] = int(bg)
    for placed in placements:
        k = int(placed["side_len"])
        r0 = int(placed["r0"])
        c0 = int(placed["c0"])
        color = int(side_colors[str(placed["nearest_side"])])
        out[r0 : r0 + k, c0 : c0 + k] = color
    return out


def _place_objects(
    grid: np.ndarray,
    object_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    occupied = np.zeros_like(grid, dtype=bool)
    placements: list[dict[str, object]] = []
    for spec in object_specs:
        mask = spec["mask"]
        nearest_side = str(spec["nearest_side"])
        candidates = _candidate_positions(grid.shape, mask.shape, nearest_side, occupied)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = OBJECT_COLOR
        _mark_occupied(occupied, int(r0), int(c0), int(mask.shape[0]), int(mask.shape[1]))
        placements.append(
            {
                "mask": mask.copy(),
                "r0": int(r0),
                "c0": int(c0),
                "side_len": int(spec["side_len"]),
                "nearest_side": nearest_side,
            }
        )
    return placements


def _sample_sides(count: int) -> list[str]:
    base = random.sample(SIDES, k=min(count, len(SIDES)))
    while len(base) < count:
        base.append(random.choice(SIDES))
    random.shuffle(base)
    return base


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg, side_colors = _sample_palette()

    grid = np.full((h, w), bg, dtype=int)
    _paint_stripes(grid, side_colors)

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    nearest_sides = _sample_sides(n_objects)
    specs: list[dict[str, object]] = []
    for nearest_side in nearest_sides:
        mask, side_len = _sample_object_mask()
        specs.append(
            {
                "mask": mask,
                "side_len": int(side_len),
                "nearest_side": nearest_side,
            }
        )

    specs.sort(key=lambda spec: int(np.prod(spec["mask"].shape)), reverse=True)
    placements = _place_objects(grid, specs)
    if placements is None:
        return None

    if _mode(grid[1:-1, 1:-1]) != bg:
        return None

    expected = _render_expected(grid, placements, bg, side_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
