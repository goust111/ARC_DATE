from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _edge_runs(values: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(values != BG)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for raw in idx[1:]:
        pos = int(raw)
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev + 1))
        start = pos
        prev = pos
    runs.append((start, prev + 1))
    return runs


def _find_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    bottom_runs = _edge_runs(grid[h - 1, :])
    right_runs = _edge_runs(grid[:, w - 1])

    candidates: list[tuple[int, str, int, int]] = []
    if bottom_runs:
        start, end = max(bottom_runs, key=lambda item: item[1] - item[0])
        candidates.append((end - start, "bottom", start, end))
    if right_runs:
        start, end = max(right_runs, key=lambda item: item[1] - item[0])
        candidates.append((end - start, "right", start, end))

    assert candidates, "expected a palette strip on the bottom row or right column"
    candidates.sort(key=lambda item: (item[0], item[1] == "bottom"), reverse=True)
    _, side, start, end = candidates[0]

    mask = np.zeros((h, w), dtype=bool)
    if side == "bottom":
        mask[h - 1, start:end] = True
        colors = [int(v) for v in grid[h - 1, start:end]]
    else:
        mask[start:end, w - 1] = True
        colors = [int(v) for v in grid[start:end, w - 1]]
    assert colors and all(color != BG for color in colors), "palette must be non-empty and non-black"
    return colors, mask


def _region_labels(grid: np.ndarray, palette_mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if palette_mask[r, c] or grid[r, c] == BG or labels[r, c] >= 0:
                continue
            color = int(grid[r, c])
            labels[r, c] = next_id
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not palette_mask[nr, nc]
                        and labels[nr, nc] < 0
                        and int(grid[nr, nc]) == color
                    ):
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            regions.append(comp)
            next_id += 1
    return labels, regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    palette, palette_mask = _find_palette(g)
    labels, _regions = _region_labels(g, palette_mask)

    out = g.copy()
    holes_by_region: dict[int, list[list[tuple[int, int]]]] = {}
    bg_comps = _components((g == BG) & (~palette_mask))
    h, w = g.shape
    for comp in bg_comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacent = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0:
                    adjacent.add(int(labels[nr, nc]))
        assert len(adjacent) == 1, "each internal hole must belong to exactly one region"
        region_id = adjacent.pop()
        holes_by_region.setdefault(region_id, []).append(comp)

    for region_id, hole_comps in holes_by_region.items():
        hole_count = len(hole_comps)
        assert 1 <= hole_count <= len(palette), "hole count must index a palette color"
        fill_color = int(palette[hole_count - 1])
        for comp in hole_comps:
            for r, c in comp:
                out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each region's holes with the palette color indexed by that region's hole count.

    Input structure:
    - Background is black.
    - The grid contains several non-overlapping 4-connected closed colored regions; each region is a thick solid block and may contain single-pixel black holes.
    - A palette strip lies on the bottom row or the rightmost column, with adjacent colored pixels ordered left-to-right or top-to-bottom.

    Rule:
    - Detect the palette strip and read its colors in order so that palette position k defines the fill color for hole-count k.
    - For every closed colored region, count how many enclosed black hole components it contains.
    - Fill every hole of that region with the color at palette position equal to the region's hole count.
    - Keep the region borders, the rest of the background, and the palette strip unchanged.

    Key insight:
    - The relevant attribute is not the region's visible color or size, but its hidden count of enclosed holes, which must be used as an index into the shared edge palette.
    """
    return _solve(grid)


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return out


def _coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _make_rect_mask() -> np.ndarray:
    h = random.randint(7, 12)
    w = random.randint(7, 12)
    return np.ones((h, w), dtype=bool)


def _make_union_mask() -> np.ndarray | None:
    for _ in range(60):
        h1 = random.randint(6, 9)
        w1 = random.randint(6, 9)
        h2 = random.randint(5, 8)
        w2 = random.randint(5, 8)
        side = random.choice(("left", "right", "top", "bottom"))

        cells = {(r, c) for r in range(h1) for c in range(w1)}
        if side in ("left", "right"):
            overlap = random.randint(3, min(h1, h2))
            start1 = random.randint(0, h1 - overlap)
            start2 = random.randint(0, h2 - overlap)
            top2 = start1 - start2
            left2 = -w2 if side == "left" else w1
        else:
            overlap = random.randint(3, min(w1, w2))
            start1 = random.randint(0, w1 - overlap)
            start2 = random.randint(0, w2 - overlap)
            top2 = -h2 if side == "top" else h1
            left2 = start1 - start2

        for r in range(h2):
            for c in range(w2):
                cells.add((top2 + r, left2 + c))

        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        shifted = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _c in shifted) + 1
        w = max(c for _r, c in shifted) + 1
        if h > 12 or w > 12:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in shifted:
            mask[r, c] = True
        return mask
    return None


def _choose_holes(mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    safe = _erode4(_erode4(mask))
    candidates = _coords(safe)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _build_object(hole_count: int) -> dict[str, object] | None:
    modes = ["rect", "union"]
    random.shuffle(modes)
    for mode in modes:
        for _ in range(50):
            mask = _make_rect_mask() if mode == "rect" else _make_union_mask()
            if mask is None:
                continue
            holes = _choose_holes(mask, hole_count)
            if holes is None:
                continue
            color = -1
            return {"mask": mask, "holes": holes, "color": color, "hole_count": int(hole_count)}
    return None


def _mark_blocked(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    orientation: str,
) -> list[dict[str, object]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []
    order = sorted(range(len(objects)), key=lambda idx: -int(np.sum(objects[idx]["mask"])))
    for idx in order:
        obj = objects[idx]
        mask = np.asarray(obj["mask"], dtype=bool)
        hole_set = set(tuple(pos) for pos in obj["holes"])
        oh, ow = mask.shape
        top_max = h - oh - 1
        left_max = w - ow - 1
        if top_max < 1 or left_max < 1:
            return None

        candidates = [(top, left) for top in range(1, top_max + 1) for left in range(1, left_max + 1)]
        random.shuffle(candidates)

        selected = None
        colored_cells_local = [(r, c) for r, c in _coords(mask) if (r, c) not in hole_set]
        for top, left in candidates:
            global_cells = [(top + r, left + c) for r, c in colored_cells_local]
            if any(blocked[r, c] for r, c in global_cells):
                continue
            selected = (top, left, global_cells)
            break
        if selected is None:
            return None

        top, left, global_cells = selected
        for r, c in global_cells:
            grid[r, c] = int(obj["color"])
        _mark_blocked(blocked, global_cells)

        placed.append(
            {
                "index": int(idx),
                "color": int(obj["color"]),
                "hole_count": int(obj["hole_count"]),
                "top": int(top),
                "left": int(left),
                "mask": mask,
                "holes": [(int(top + r), int(left + c)) for r, c in hole_set],
            }
        )

    assert len(placed) == len(objects)
    return placed


def _palette_start(edge_len: int, palette_len: int) -> int:
    assert edge_len >= palette_len + 3
    return random.randint(1, edge_len - palette_len - 1)


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.choice((2, 3))
    max_palette_len = 9 - object_count
    palette_len = random.randint(max(3, object_count), min(6, max_palette_len))
    hole_counts = random.sample(range(1, palette_len + 1), object_count)
    random.shuffle(hole_counts)
    orientation = random.choice(("bottom", "right"))

    palette_and_object_colors = random.sample(range(1, 10), palette_len + object_count)
    palette_colors = palette_and_object_colors[:palette_len]
    object_colors = palette_and_object_colors[palette_len:]

    objects: list[dict[str, object]] = []
    for hole_count, color in zip(hole_counts, object_colors, strict=False):
        obj = _build_object(hole_count)
        if obj is None:
            return None
        obj["color"] = int(color)
        objects.append(obj)

    min_h = max(int(np.asarray(obj["mask"]).shape[0]) for obj in objects) + 3
    min_w = max(int(np.asarray(obj["mask"]).shape[1]) for obj in objects) + 3
    if orientation == "bottom":
        min_w = max(min_w, palette_len + 3)
    else:
        min_h = max(min_h, palette_len + 3)

    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(max(18, min_h), 30)
    w = random.randint(max(18, min_w), 30)
    grid = np.zeros((h, w), dtype=int)

    placed = _place_objects(grid, objects, orientation)
    if placed is None:
        return None

    if orientation == "bottom":
        start = _palette_start(w, palette_len)
        grid[h - 1, start : start + palette_len] = palette_colors
    else:
        start = _palette_start(h, palette_len)
        grid[start : start + palette_len, w - 1] = palette_colors

    expected = grid.copy()
    for obj in placed:
        fill = int(palette_colors[int(obj["hole_count"]) - 1])
        for r, c in obj["holes"]:
            expected[r, c] = fill

    actual = _solve(grid)
    if not np.array_equal(actual, expected):
        return None

    palette, palette_mask = _find_palette(grid)
    if len(palette) != palette_len:
        return None
    labels, regions = _region_labels(grid, palette_mask)
    if len(regions) != object_count:
        return None
    region_ids = {labels[r, c] for r, c in np.argwhere((grid != BG) & (~palette_mask))}
    if len(region_ids) != object_count:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        example = _build_example()
        if example is not None:
            return example[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
