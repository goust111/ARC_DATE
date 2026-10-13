from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# --------------- solver ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _fill_color(missing_count: int, palette: list[int]) -> int | None:
    if 1 <= missing_count <= 2:
        return int(palette[0])
    if 3 <= missing_count <= 5:
        return int(palette[1])
    if missing_count >= 6:
        return int(palette[2])
    return None


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corner_points = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))

    candidates: list[tuple[list[int], np.ndarray]] = []
    for coords in _components(g != BG):
        if coords.shape[0] != 3:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if (r1 - r0 + 1, c1 - c0 + 1) not in ((1, 3), (3, 1)):
            continue
        colors = [int(g[r, c]) for r, c in coords]
        if len(set(colors)) != 3:
            continue

        pts = {tuple(map(int, rc)) for rc in coords}
        touched = [corner for corner in corner_points if corner in pts]
        if len(touched) != 1:
            continue
        corner_r, corner_c = touched[0]
        ordered = sorted(
            pts,
            key=lambda rc: abs(rc[0] - corner_r) + abs(rc[1] - corner_c),
        )
        if [abs(r - corner_r) + abs(c - corner_c) for r, c in ordered] != [0, 1, 2]:
            continue
        legend_mask = np.zeros_like(g, dtype=bool)
        for r, c in ordered:
            legend_mask[r, c] = True
        candidates.append(([int(g[r, c]) for r, c in ordered], legend_mask))

    assert len(candidates) == 1, "expected a unique 3-cell legend in one corner"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's bbox-minus-object region with a legend color based on its size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    palette, legend_mask = _find_legend(g)
    object_mask = (g != BG) & (~legend_mask)
    out = g.copy()

    for coords in _components(object_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        missing = region == BG
        fill = _fill_color(int(missing.sum()), palette)
        if fill is None:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1][missing] = int(fill)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
OBJECT_COUNT_RANGE = (2, 5)
MAX_GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
PAD = 1

CATEGORY_RECT = 0
CATEGORY_SMALL = 1
CATEGORY_MEDIUM = 2
CATEGORY_LARGE = 3

MISSING_RANGES = {
    CATEGORY_RECT: (0, 0),
    CATEGORY_SMALL: (1, 2),
    CATEGORY_MEDIUM: (3, 5),
    CATEGORY_LARGE: (6, 10),
}


def _sample_categories(count: int) -> list[int]:
    base = [CATEGORY_RECT, CATEGORY_SMALL, CATEGORY_MEDIUM, CATEGORY_LARGE]
    random.shuffle(base)
    cats = base[: min(count, len(base))]
    while len(cats) < count:
        cats.append(random.choice(base))
    random.shuffle(cats)
    return cats


def _grid_positions(h: int, w: int, target_h: int, target_w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, h - target_h)
        for c in range(1, w - target_w)
    ]


def _pick_legend_cells(h: int, w: int) -> list[tuple[int, int]]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    orient = random.choice(("h", "v"))
    if corner == "tl":
        if orient == "h":
            return [(0, 0), (0, 1), (0, 2)]
        return [(0, 0), (1, 0), (2, 0)]
    if corner == "tr":
        if orient == "h":
            return [(0, w - 1), (0, w - 2), (0, w - 3)]
        return [(0, w - 1), (1, w - 1), (2, w - 1)]
    if corner == "bl":
        if orient == "h":
            return [(h - 1, 0), (h - 1, 1), (h - 1, 2)]
        return [(h - 1, 0), (h - 2, 0), (h - 3, 0)]
    if orient == "h":
        return [(h - 1, w - 1), (h - 1, w - 2), (h - 1, w - 3)]
    return [(h - 1, w - 1), (h - 2, w - 1), (h - 3, w - 1)]


def _grow_cutout(h: int, w: int, size: int, mode: str) -> set[tuple[int, int]] | None:
    if mode == "corner":
        seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    elif mode == "edge":
        seeds = []
        for r in range(h):
            for c in range(w):
                on_edge = r in (0, h - 1) or c in (0, w - 1)
                if on_edge and (r, c) not in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                    seeds.append((r, c))
    else:
        seeds = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

    if not seeds:
        return None

    region = {random.choice(seeds)}
    while len(region) < size:
        frontier = []
        for r, c in region:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in region:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        region.add(random.choice(frontier))
    return region


def _make_mask(category: int) -> tuple[np.ndarray, int]:
    if category == CATEGORY_RECT:
        h = random.randint(2, 5)
        w = random.randint(2, 6)
        return np.ones((h, w), dtype=bool), 0

    lo, hi = MISSING_RANGES[category]
    for _ in range(300):
        missing = random.randint(lo, hi)
        min_h = 3 if category != CATEGORY_LARGE else 4
        min_w = 3 if category != CATEGORY_LARGE else 4
        max_h = 6 if category != CATEGORY_LARGE else 7
        max_w = 7 if category != CATEGORY_LARGE else 8

        candidates = [
            (h, w)
            for h in range(min_h, max_h + 1)
            for w in range(min_w, max_w + 1)
            if h * w >= missing + 6
        ]
        h, w = random.choice(candidates)
        mode = random.choice(("corner", "edge", "interior"))
        cutout = _grow_cutout(h, w, missing, mode)
        if cutout is None:
            continue

        mask = np.ones((h, w), dtype=bool)
        for r, c in cutout:
            mask[r, c] = False

        if int(mask.sum()) < 4:
            continue
        if not _spans_bbox(mask):
            continue
        if not _is_connected(mask):
            continue
        return mask, missing

    raise RuntimeError(f"failed to build object mask for category {category}")


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h - 1, r1 + PAD)
    cc1 = min(w - 1, c1 + PAD)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h - 1, r1 + PAD)
    cc1 = min(w - 1, c1 + PAD)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _place_objects(
    grid: np.ndarray,
    occupied: np.ndarray,
    palette: list[int],
    object_colors: list[int],
    categories: list[int],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    h, w = grid.shape
    placements: list[dict[str, object]] = []
    for color, category in zip(object_colors, categories):
        mask, missing = _make_mask(category)
        oh, ow = mask.shape
        candidates = _grid_positions(h, w, oh, ow)
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue

            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            region[mask] = int(color)
            _reserve(occupied, r0, c0, r1, c1)
            placements.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask.copy(),
                    "missing": missing,
                    "fill": _fill_color(missing, palette),
                }
            )
            placed = True
            break

        if not placed:
            return None
    return grid, placements


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for info in placements:
        fill = info["fill"]
        if fill is None:
            continue
        r0, c0, r1, c1 = info["bbox"]
        mask = info["mask"]
        out[r0 : r1 + 1, c0 : c1 + 1][~mask] = int(fill)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_colors = random.sample(list(COLORS), 3)
    legend_cells = _pick_legend_cells(h, w)
    for (r, c), color in zip(legend_cells, legend_colors):
        grid[r, c] = int(color)

    legend_rows = [r for r, _ in legend_cells]
    legend_cols = [c for _, c in legend_cells]
    _reserve(occupied, min(legend_rows), min(legend_cols), max(legend_rows), max(legend_cols))

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    categories = _sample_categories(object_count)
    remaining = [c for c in COLORS if c not in legend_colors]
    object_colors = random.sample(remaining, object_count)

    placed = _place_objects(grid, occupied, legend_colors, object_colors, categories)
    if placed is None:
        return None
    built_grid, placements = placed

    expected = _expected_output(built_grid, placements)
    solved = main(built_grid)
    if not np.array_equal(solved, expected):
        return None
    return built_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
