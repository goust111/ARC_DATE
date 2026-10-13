from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GEN_MARKER_COLOR = 3
GEN_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_HEIGHT_RANGE = (15, 24)
GEN_WIDTH_RANGE = (15, 24)
GEN_PAIR_COUNT_RANGE = (2, 4)
GEN_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _component_records(grid: np.ndarray, skip: set[tuple[int, int]] | None = None) -> list[dict]:
    h, w = grid.shape
    skip = skip or set()
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG or (r, c) in skip:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                if (rr, cc) in skip:
                    continue
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color and (nr, nc) not in skip:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0 = min(rr for rr, _ in coords)
            c0 = min(cc for _, cc in coords)
            r1 = max(rr for rr, _ in coords)
            c1 = max(cc for _, cc in coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local_cells: list[tuple[int, int]] = []
            for rr, cc in coords:
                lr = rr - r0
                lc = cc - c0
                mask[lr, lc] = True
                local_cells.append((lr, lc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "local_cells": local_cells,
                }
            )
    return comps


def _bottom_right_marker(components: list[dict]) -> dict:
    singles = [comp for comp in components if len(comp["coords"]) == 1]
    if not singles:
        raise ValueError("expected a singleton overlap marker")
    return max(singles, key=lambda comp: comp["coords"][0])


def _merge_axis(a: int, b: int) -> tuple[int, int]:
    lo = min(a, b)
    hi = max(a, b)
    mid_lo = (lo + hi) // 2
    mid_hi = (lo + hi + 1) // 2
    if a <= b:
        return mid_lo, mid_hi
    return mid_hi, mid_lo


def _paint_cell(out: np.ndarray, r: int, c: int, value: int, overlap_color: int) -> None:
    current = int(out[r, c])
    if current == BG:
        out[r, c] = value
    elif current == value:
        return
    else:
        out[r, c] = overlap_color


def main(grid: np.ndarray) -> np.ndarray:
    """Merge each same-color pair at the midpoint of their bbox centers.

    The midpoint lives in continuous space. When that midpoint falls between cells,
    each object keeps its relative order along that axis and lands on the adjacent
    floor/ceil integer positions, so odd offsets create partial overlap.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    comps = _component_records(g)
    marker = _bottom_right_marker(comps)
    marker_pos = marker["coords"][0]
    overlap_color = marker["color"]

    objects = _component_records(g, skip={marker_pos})
    groups: dict[int, list[dict]] = defaultdict(list)
    for comp in objects:
        groups[int(comp["color"])].append(comp)

    out = np.zeros_like(g)

    for color in sorted(groups):
        pair = groups[color]
        if len(pair) != 2:
            raise ValueError(f"color {color} should appear in exactly two objects")
        a, b = pair
        if a["mask"].shape != b["mask"].shape or not np.array_equal(a["mask"], b["mask"]):
            raise ValueError(f"color {color} pair is not translation-equivalent")

        ar0, ac0, _, _ = a["bbox"]
        br0, bc0, _, _ = b["bbox"]
        nr0_a, nr0_b = _merge_axis(ar0, br0)
        nc0_a, nc0_b = _merge_axis(ac0, bc0)

        counts: dict[tuple[int, int], int] = defaultdict(int)
        for top_r, top_c in ((nr0_a, nc0_a), (nr0_b, nc0_b)):
            for dr, dc in a["local_cells"]:
                rr = top_r + dr
                cc = top_c + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    counts[(rr, cc)] += 1

        for (rr, cc), count in counts.items():
            value = overlap_color if count > 1 else int(color)
            _paint_cell(out, rr, cc, value, overlap_color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(1, 4)
    w = rng.randint(1, 4)
    target = rng.randint(1, min(h * w, 7))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(sorted(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    mask = _crop_mask(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.35:
        mask = mask.T
    return _crop_mask(mask)


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _halo(cells: list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _merged_counts(mask: np.ndarray, top_a: tuple[int, int], top_b: tuple[int, int], h: int, w: int) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for top_r, top_c in (top_a, top_b):
        for rr, cc in _mask_cells(mask, top_r, top_c):
            if 0 <= rr < h and 0 <= cc < w:
                counts[(rr, cc)] += 1
    return counts


def _sample_delta(rng: random.Random, limit: int, parity: int) -> int | None:
    values = [delta for delta in range(parity, limit + 1, 2)]
    if not values:
        return None
    if parity == 0 and len(values) > 1:
        weights = [1 if delta == 0 else 3 for delta in values]
        return rng.choices(values, weights=weights, k=1)[0]
    return rng.choice(values)


def _place_pair(
    rng: random.Random,
    h: int,
    w: int,
    mask: np.ndarray,
    input_blocked: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], int]] | None:
    mh, mw = mask.shape
    top_r_limit = h - 2 - mh
    top_c_limit = w - 2 - mw
    if top_r_limit < 0 or top_c_limit < 0:
        return None

    parity_row, parity_col = rng.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
    max_dr = top_r_limit
    max_dc = top_c_limit

    for _ in range(200):
        dr = _sample_delta(rng, max_dr, parity_row)
        dc = _sample_delta(rng, max_dc, parity_col)
        if dr is None or dc is None:
            return None
        if dr == 0 and dc == 0:
            continue
        if max(dr, dc) < 2:
            continue

        sign_r = 1 if dr == 0 else rng.choice((-1, 1))
        sign_c = 1 if dc == 0 else rng.choice((-1, 1))

        r1_min = dr if sign_r < 0 else 0
        r1_max = top_r_limit if sign_r < 0 else top_r_limit - dr
        c1_min = dc if sign_c < 0 else 0
        c1_max = top_c_limit if sign_c < 0 else top_c_limit - dc
        if r1_min > r1_max or c1_min > c1_max:
            continue

        r1 = rng.randint(r1_min, r1_max)
        c1 = rng.randint(c1_min, c1_max)
        r2 = r1 + sign_r * dr
        c2 = c1 + sign_c * dc

        cells_a = _mask_cells(mask, r1, c1)
        cells_b = _mask_cells(mask, r2, c2)
        if any(cell in _halo(cells_a, h, w, radius=1) for cell in cells_b):
            continue
        if any(cell in input_blocked for cell in cells_a) or any(cell in input_blocked for cell in cells_b):
            continue

        nr1, nr2 = _merge_axis(r1, r2)
        nc1, nc2 = _merge_axis(c1, c2)
        merged = _merged_counts(mask, (nr1, nc1), (nr2, nc2), h, w)
        if not any(count > 1 for count in merged.values()):
            continue
        if any((rr, cc) in output_blocked for rr, cc in merged):
            continue

        return (r1, c1), (r2, c2), merged
    return None


def _grid_from_pairs(h: int, w: int, marker_color: int, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, w - 1] = marker_color
    for pair in pairs:
        color = int(pair["color"])
        mask = pair["mask"]
        for top_r, top_c in pair["tops"]:
            for rr, cc in _mask_cells(mask, top_r, top_c):
                grid[rr, cc] = color
    return grid


def _expected_from_pairs(h: int, w: int, marker_color: int, pairs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for pair in pairs:
        color = int(pair["color"])
        for (rr, cc), count in pair["merged"].items():
            out[rr, cc] = marker_color if count > 1 else color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        h = rng.randint(*GEN_HEIGHT_RANGE)
        w = rng.randint(*GEN_WIDTH_RANGE)
        pair_count = rng.randint(*GEN_PAIR_COUNT_RANGE)
        colors = rng.sample(GEN_COLORS, pair_count)

        input_blocked: set[tuple[int, int]] = {(h - 1, c) for c in range(w)}
        input_blocked.update((r, w - 1) for r in range(h))
        output_blocked: set[tuple[int, int]] = {(h - 1, c) for c in range(w)}
        output_blocked.update((r, w - 1) for r in range(h))

        pairs: list[dict] = []
        ok = True

        for color in colors:
            placed = None
            for _shape_try in range(80):
                mask = _random_shape(rng)
                candidate = _place_pair(rng, h, w, mask, input_blocked, output_blocked)
                if candidate is None:
                    continue
                top_a, top_b, merged = candidate
                placed = {"color": color, "mask": mask, "tops": (top_a, top_b), "merged": merged}
                break

            if placed is None:
                ok = False
                break

            cells_a = _mask_cells(placed["mask"], *placed["tops"][0])
            cells_b = _mask_cells(placed["mask"], *placed["tops"][1])
            input_blocked.update(_halo(cells_a + cells_b, h, w, radius=1))
            output_blocked.update(_halo(list(placed["merged"].keys()), h, w, radius=1))
            pairs.append(placed)

        if not ok:
            continue

        grid = _grid_from_pairs(h, w, GEN_MARKER_COLOR, pairs)
        expected = _expected_from_pairs(h, w, GEN_MARKER_COLOR, pairs)

        try:
            solved = main(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        if int((solved == GEN_MARKER_COLOR).sum()) == 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
