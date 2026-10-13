from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
REGION = 1
MARKER = 2
RANK_COLORS = [3, 4, 6, 7]

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=int)
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    return out


def _transform_shape(coords: np.ndarray) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    k = random.randrange(4)
    for _ in range(k):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    return _normalize(arr)


def _enclosed_black_area(grid: np.ndarray, coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_mask = np.zeros(sub.shape, dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded_grid = np.pad(sub, 1, constant_values=BG)
    padded_mask = np.pad(local_mask, 1, constant_values=False)
    h, w = padded_grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                if padded_grid[nr, nc] == BG and not padded_mask[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

    enclosed = (padded_grid == BG) & (~padded_mask) & (~seen)
    return int(enclosed.sum())


def _priority_key(region: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = region["bbox"]
    return (
        int(region["enclosed"]),
        int(region["bbox_area"]),
        -int(r0),
        -int(c0),
    )


def _find_unique_marker(coords: np.ndarray, markers: list[tuple[int, int]]) -> tuple[tuple[int, int], str]:
    rows = set(map(int, coords[:, 0]))
    cols = set(map(int, coords[:, 1]))
    hits = []
    for marker in markers:
        mr, mc = marker
        share_row = mr in rows
        share_col = mc in cols
        if share_row or share_col:
            hits.append((marker, share_row, share_col))
    assert len(hits) == 1, "each region must match exactly one marker"
    marker, share_row, share_col = hits[0]
    assert share_row ^ share_col, "marker must share exactly one axis"
    return marker, ("row" if share_row else "col")


def _slide_region(coords: np.ndarray, marker: tuple[int, int], axis: str) -> np.ndarray:
    mr, mc = marker
    moved = np.array(coords, dtype=int)
    if axis == "row":
        row_cells = moved[moved[:, 0] == mr]
        cols = row_cells[:, 1]
        if mc > int(cols.max()):
            dc = mc - 1 - int(cols.max())
        else:
            assert mc < int(cols.min())
            dc = mc + 1 - int(cols.min())
        moved[:, 1] += dc
    else:
        col_cells = moved[moved[:, 1] == mc]
        rows = col_cells[:, 0]
        if mr > int(rows.max()):
            dr = mr - 1 - int(rows.max())
        else:
            assert mr < int(rows.min())
            dr = mr + 1 - int(rows.min())
        moved[:, 0] += dr
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue connected region to its unique marker, resolve overlaps by pre-slide enclosed area, then recolor by that ranking.

    Input structure:
    - The grid is mostly black with 3 or 4 blue 4-connected regions and several red singleton markers.
    - Each blue region is separated from the others and shares exactly one row or one column with exactly one red marker.
    - Some regions enclose black pocket cells before moving; others do not.

    Rule:
    - For each blue region, measure how many black cells are fully enclosed by that region in the original input.
    - Slide the entire region horizontally or vertically toward its unique marker until the nearest boundary pixel on the shared row or column becomes adjacent to the marker.
    - If moved regions overlap, each overlapped cell belongs to the region with larger original enclosed area; ties use larger original bounding-box area, then smaller original bounding-box top row.
    - Recolor the surviving moved cells by enclosed-area rank: first green, second yellow, third magenta, fourth orange; regions with enclosed area 0 stay blue. Red markers remain unchanged.

    Key insight:
    - Both overlap ownership and final recoloring depend on the enclosed black area measured before any sliding, not on the post-slide geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = [tuple(map(int, p)) for p in np.argwhere(g == MARKER)]
    regions = []
    for coords in _components(g == REGION):
        bbox = _bbox(coords)
        enclosed = _enclosed_black_area(g, coords)
        marker, axis = _find_unique_marker(coords, markers)
        moved = _slide_region(coords, marker, axis)
        r0, c0, r1, c1 = bbox
        regions.append(
            {
                "coords": coords,
                "bbox": bbox,
                "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "enclosed": enclosed,
                "moved": moved,
            }
        )

    ownership: dict[tuple[int, int], int] = {}
    for idx, region in enumerate(regions):
        for r, c in map(tuple, region["moved"]):
            winner = ownership.get((int(r), int(c)))
            if winner is None or _priority_key(region) > _priority_key(regions[winner]):
                ownership[(int(r), int(c))] = idx

    kept = {i: [] for i in range(len(regions))}
    for cell, idx in ownership.items():
        kept[idx].append(cell)

    positives = [i for i, region in enumerate(regions) if int(region["enclosed"]) > 0]
    positives.sort(key=lambda i: _priority_key(regions[i]), reverse=True)
    recolor = {idx: RANK_COLORS[rank] for rank, idx in enumerate(positives[: len(RANK_COLORS)])}

    out = np.full(g.shape, BG, dtype=int)
    for idx, cells in kept.items():
        color = REGION if int(regions[idx]["enclosed"]) == 0 else recolor[idx]
        for r, c in cells:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
    for r, c in markers:
        out[r, c] = MARKER
    return out


AREA0_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
]

AREA1_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (2, 0), (2, 1)],
]

AREA2_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)],
]

TEMPLATES = {
    0: [np.array(t, dtype=int) for t in AREA0_TEMPLATES],
    1: [np.array(t, dtype=int) for t in AREA1_TEMPLATES],
    2: [np.array(t, dtype=int) for t in AREA2_TEMPLATES],
}

AREA_PROFILES = {
    3: [([2, 1, 0], 4), ([2, 1, 1], 3), ([2, 2, 0], 2), ([1, 1, 0], 2)],
    4: [([2, 1, 1, 0], 4), ([2, 2, 1, 0], 3), ([2, 1, 0, 0], 2), ([1, 1, 1, 0], 2)],
}


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _cells_from_top_left(coords: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(int(r0 + r), int(c0 + c)) for r, c in map(tuple, coords)}


def _adjacent4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _valid_specs(shape: np.ndarray, final_top_left: tuple[int, int], h: int, w: int) -> list[dict]:
    fr, fc = final_top_left
    sh, sw = _shape_dims(shape)
    specs = []

    for local_r in sorted(set(map(int, shape[:, 0]))):
        cols = shape[shape[:, 0] == local_r][:, 1]
        left_edge = int(cols.min())
        right_edge = int(cols.max())

        marker_r = fr + local_r
        marker_c = fc + left_edge - 1
        max_shift = w - (fc + sw)
        if 0 <= marker_r < h and 0 <= marker_c < w and max_shift >= 2:
            specs.append(
                {
                    "axis": "row",
                    "marker": (marker_r, marker_c),
                    "orig_top_left": lambda d, fr=fr, fc=fc: (fr, fc + d),
                    "max_shift": max_shift,
                }
            )

        marker_c = fc + right_edge + 1
        max_shift = fc
        if 0 <= marker_r < h and 0 <= marker_c < w and max_shift >= 2:
            specs.append(
                {
                    "axis": "row",
                    "marker": (marker_r, marker_c),
                    "orig_top_left": lambda d, fr=fr, fc=fc: (fr, fc - d),
                    "max_shift": max_shift,
                }
            )

    for local_c in sorted(set(map(int, shape[:, 1]))):
        rows = shape[shape[:, 1] == local_c][:, 0]
        top_edge = int(rows.min())
        bottom_edge = int(rows.max())

        marker_r = fr + top_edge - 1
        marker_c = fc + local_c
        max_shift = h - (fr + sh)
        if 0 <= marker_r < h and 0 <= marker_c < w and max_shift >= 2:
            specs.append(
                {
                    "axis": "col",
                    "marker": (marker_r, marker_c),
                    "orig_top_left": lambda d, fr=fr, fc=fc: (fr + d, fc),
                    "max_shift": max_shift,
                }
            )

        marker_r = fr + bottom_edge + 1
        max_shift = fr
        if 0 <= marker_r < h and 0 <= marker_c < w and max_shift >= 2:
            specs.append(
                {
                    "axis": "col",
                    "marker": (marker_r, marker_c),
                    "orig_top_left": lambda d, fr=fr, fc=fc: (fr - d, fc),
                    "max_shift": max_shift,
                }
            )

    return specs


def _profile_for_count(n_regions: int) -> list[int]:
    options, weights = zip(*AREA_PROFILES[n_regions])
    profile = list(random.choices(options, weights=weights, k=1)[0])
    random.shuffle(profile)
    return profile


def _sample_shapes(n_regions: int) -> list[dict]:
    profile = _profile_for_count(n_regions)
    shapes = []
    for area in profile:
        shape = _transform_shape(random.choice(TEMPLATES[area]))
        bbox = _bbox(shape)
        shapes.append(
            {
                "shape": shape,
                "enclosed": area,
                "bbox_area": (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1),
            }
        )
    return shapes


def _sample_final_positions(shapes: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    n_regions = len(shapes)
    pair = [0, 1]
    finals: list[tuple[int, int] | None] = [None] * n_regions

    for _ in range(80):
        cr = random.randint(4, h - 5)
        cc = random.randint(4, w - 5)
        ok = True
        for idx in pair:
            shape = shapes[idx]["shape"]
            pr, pc = map(int, random.choice(list(map(tuple, shape))))
            fr = cr - pr
            fc = cc - pc
            sh, sw = _shape_dims(shape)
            if not (0 <= fr and 0 <= fc and fr + sh <= h and fc + sw <= w):
                ok = False
                break
            finals[idx] = (fr, fc)
        if ok:
            break
    else:
        return None

    for idx in range(2, n_regions):
        shape = shapes[idx]["shape"]
        sh, sw = _shape_dims(shape)
        placed = False
        for _ in range(60):
            fr = random.randint(2, max(2, h - sh - 2))
            fc = random.randint(2, max(2, w - sw - 2))
            finals[idx] = (fr, fc)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in finals if pos is not None]


def _construct_input() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    n_regions = random.choice([3, 4])
    shapes = _sample_shapes(n_regions)
    finals = _sample_final_positions(shapes, h, w)
    if finals is None:
        return None

    regions = []
    used_markers: set[tuple[int, int]] = set()
    used_inputs: list[set[tuple[int, int]]] = []

    for idx, info in enumerate(shapes):
        shape = info["shape"]
        specs = _valid_specs(shape, finals[idx], h, w)
        random.shuffle(specs)
        chosen = None
        for spec in specs:
            marker = spec["marker"]
            if marker in used_markers:
                continue
            max_shift = min(7, int(spec["max_shift"]))
            if max_shift < 2:
                continue
            d = random.randint(2, max_shift)
            orig_top_left = spec["orig_top_left"](d)
            cells = _cells_from_top_left(shape, orig_top_left)
            if marker in cells:
                continue
            if any(cells & other for other in used_inputs):
                continue
            if any(_adjacent4(cells, other) for other in used_inputs):
                continue
            chosen = {
                "shape": shape,
                "final_top_left": finals[idx],
                "orig_top_left": orig_top_left,
                "marker": marker,
            }
            break
        if chosen is None:
            return None
        regions.append(chosen)
        used_markers.add(chosen["marker"])
        used_inputs.append(_cells_from_top_left(shape, chosen["orig_top_left"]))

    grid = np.zeros((h, w), dtype=int)
    for region in regions:
        for r, c in _cells_from_top_left(region["shape"], region["orig_top_left"]):
            grid[r, c] = REGION
    for r, c in used_markers:
        if grid[r, c] != BG:
            return None
        grid[r, c] = MARKER

    if len(_components(grid == REGION)) != n_regions:
        return None

    markers = list(used_markers)
    for region in regions:
        orig_coords = np.array(sorted(_cells_from_top_left(region["shape"], region["orig_top_left"])), dtype=int)
        hits = []
        rows = set(map(int, orig_coords[:, 0]))
        cols = set(map(int, orig_coords[:, 1]))
        for marker in markers:
            share_row = marker[0] in rows
            share_col = marker[1] in cols
            if share_row or share_col:
                hits.append((marker, share_row, share_col))
        if len(hits) != 1 or not (hits[0][1] ^ hits[0][2]):
            return None

    out = main(grid)
    moved_total = sum(int((grid == REGION).sum()) for _ in [0])
    overlap_pixels = moved_total - int(np.count_nonzero(out != BG)) + len(used_markers)
    if overlap_pixels < 1:
        return None

    color_counts = {color: int(np.count_nonzero(out == color)) for color in [1, 3, 4, 6, 7]}
    if max(color_counts.values()) > 12:
        return None
    if sum(v > 0 for v in color_counts.values()) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _construct_input()
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
