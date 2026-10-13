from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (3, 5)
REGION_RETRIES = 120
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _sorted_colors_by_area(counts: Counter[int]) -> list[int]:
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    sizes = [count for _, count in ranked]
    if len(set(sizes)) != len(sizes):
        raise AssertionError("color areas must be strictly ordered inside each region")
    return [int(color) for color, _ in ranked]


def _find_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, object]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perimeter = _perimeter_coords(r0, c0, r1, c1)
        perimeter_set = set(perimeter)
        comp_set = set(comp)
        if not comp_set.issubset(perimeter_set):
            continue

        missing = list(perimeter_set - comp_set)
        if len(missing) != 1:
            continue

        gap = missing[0]
        if int(g[gap]) != BLACK:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == GREY):
            continue

        colors = [int(v) for v in np.unique(interior) if int(v) not in (BLACK, GREY)]
        if not 1 <= len(colors) <= 3:
            continue

        counts: Counter[int] = Counter()
        for color in colors:
            counts[color] = int(np.sum(interior == color))

        regions.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap[0]), int(gap[1])),
                "border": list(comp),
                "counts": counts,
            }
        )

    regions.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each grey open rectangle based on the number of interior colors.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _find_regions(g):
        counts = region["counts"]  # type: ignore[assignment]
        ranked = _sorted_colors_by_area(counts)
        gap_r, gap_c = region["gap"]  # type: ignore[misc]
        border = region["border"]  # type: ignore[assignment]
        color_count = len(ranked)

        if color_count == 1:
            out[gap_r, gap_c] = ranked[0]
            continue

        major = ranked[0]
        minor = ranked[-1]
        for r, c in border:
            out[r, c] = major
        out[gap_r, gap_c] = minor

        if color_count == 3:
            middle = ranked[1]
            r0, c0, r1, c1 = region["bbox"]  # type: ignore[misc]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == middle] = BLACK

    return out


def _blob_with_area(inner_h: int, inner_w: int, area: int, blocked: np.ndarray) -> set[tuple[int, int]] | None:
    free_count = int((~blocked).sum())
    if area > free_count:
        return None

    for _ in range(80):
        free = [(r, c) for r in range(inner_h) for c in range(inner_w) if not blocked[r, c]]
        seed = random.choice(free)
        blob = {seed}
        while len(blob) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in blob:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < inner_h and 0 <= nc < inner_w and not blocked[nr, nc] and (nr, nc) not in blob:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            blob.add(random.choice(tuple(frontier)))
        if len(blob) == area:
            return blob
    return None


def _mark_moat(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _sample_total(kind: int, inner_area: int) -> int | None:
    min_total = {1: 2, 2: 4, 3: 6}[kind]
    high_ratio = {1: 0.55, 2: 0.5, 3: 0.42}[kind]
    max_total = int(inner_area * high_ratio)
    if max_total < min_total:
        return None
    return random.randint(min_total, max_total)


def _sample_counts(kind: int, total: int) -> list[int] | None:
    if kind == 1:
        return [int(total)]

    if kind == 2:
        max_small = (total - 1) // 2
        if max_small < 1:
            return None
        small = random.randint(1, max_small)
        large = total - small
        if large <= small:
            return None
        return [int(large), int(small)]

    for _ in range(80):
        small = random.randint(1, max(1, total // 4))
        mid_upper = max(small + 1, (total - small - 1) // 2)
        middle = random.randint(small + 1, mid_upper)
        large = total - middle - small
        if large > middle > small:
            return [int(large), int(middle), int(small)]
    return None


def _kind_size_range(kind: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == 1:
        return (3, 7), (3, 7)
    if kind == 2:
        return (4, 7), (4, 7)
    return (5, 8), (5, 8)


def _generate_region(kind: int) -> np.ndarray | None:
    row_range, col_range = _kind_size_range(kind)
    for _ in range(REGION_RETRIES):
        inner_h = random.randint(*row_range)
        inner_w = random.randint(*col_range)
        total = _sample_total(kind, inner_h * inner_w)
        if total is None:
            continue
        counts = _sample_counts(kind, total)
        if counts is None:
            continue

        inner = np.zeros((inner_h, inner_w), dtype=int)
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        colors = random.sample(COLOR_POOL, kind)
        ok = True
        for color, area in zip(colors, counts):
            blob = _blob_with_area(inner_h, inner_w, area, blocked)
            if blob is None:
                ok = False
                break
            for r, c in blob:
                inner[r, c] = color
            _mark_moat(blocked, blob)
        if not ok:
            continue

        if len([v for v in np.unique(inner) if int(v) != BLACK]) != kind:
            continue

        patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        patch[1:-1, 1:-1] = inner
        gap = random.choice(_perimeter_coords(0, 0, patch.shape[0] - 1, patch.shape[1] - 1))
        patch[gap] = BLACK

        regions = _find_regions(patch)
        if len(regions) != 1:
            continue
        found_counts = regions[0]["counts"]  # type: ignore[index]
        if sorted(found_counts.values(), reverse=True) != counts:
            continue
        if len(found_counts) != kind:
            continue
        return patch
    return None


def _choose_grid_shape(patches: list[np.ndarray]) -> tuple[int, int] | None:
    reserved_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    side = int(np.ceil(np.sqrt(reserved_area)))
    min_h = max(GRID_H_RANGE[0], max(int(p.shape[0]) for p in patches) + 2, side)
    min_w = max(GRID_W_RANGE[0], max(int(p.shape[1]) for p in patches) + 2, side)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    return h, w


def _place_patches(grid: np.ndarray, patches: list[np.ndarray]) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    ordered = sorted(patches, key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in ordered:
        ph, pw = map(int, patch.shape)
        if h - ph - 1 < 1 or w - pw - 1 < 1:
            return False

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(1, w - pw - 1)
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + ph + 1
            cc1 = c0 + pw + 1
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            mask = patch != BLACK
            sub[mask] = patch[mask]
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        object_count = random.randint(*OBJECT_COUNT_RANGE)
        branch_types = [1, 2, 3]
        while len(branch_types) < object_count:
            branch_types.append(random.choice((1, 2, 3)))
        random.shuffle(branch_types)

        patches: list[np.ndarray] = []
        for kind in branch_types:
            patch = _generate_region(kind)
            if patch is None:
                patches = []
                break
            patches.append(patch)
        if not patches:
            continue

        shape = _choose_grid_shape(patches)
        if shape is None:
            continue

        grid = np.zeros(shape, dtype=int)
        if not _place_patches(grid, patches):
            continue

        regions = _find_regions(grid)
        if len(regions) != object_count:
            continue
        branch_signature = sorted(len(region["counts"]) for region in regions)  # type: ignore[arg-type]
        if not {1, 2, 3}.issubset(set(branch_signature)):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
