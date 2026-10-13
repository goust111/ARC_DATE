from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 3
MARKER = 7
FILL_COLORS = (1, 2, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MIN_REGION_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _extract_regions(grid: np.ndarray) -> list[dict]:
    bg = int(Counter(map(int, grid.flatten())).most_common(1)[0][0])
    comps = _components(grid != bg)
    regions: list[dict] = []
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(grid, bbox)
        h, w = patch.shape
        assert int(coords.shape[0]) == h * w, "every non-background component must fill its bbox"
        regions.append({"bbox": bbox, "patch": patch})
    assert 3 <= len(regions) <= 4, "expected 3-4 separated rectangular regions"
    return regions


def _count_area5_markers(patch: np.ndarray) -> int:
    comps = _components(patch == MARKER)
    return sum(int(comp.shape[0]) == 5 for comp in comps)


def _main_diag_symmetric(patch: np.ndarray) -> bool:
    return patch.shape[0] == patch.shape[1] and bool(np.array_equal(patch, patch.T))


def _select_region(regions: list[dict]) -> dict:
    counts = [_count_area5_markers(region["patch"]) for region in regions]
    best = max(counts)
    winners = [region for region, count in zip(regions, counts) if count == best]
    if len(winners) == 1:
        return winners[0]
    asym = [region for region in winners if not _main_diag_symmetric(region["patch"])]
    assert len(asym) == 1, "tied maxima must contain exactly one asymmetric region"
    return asym[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count area-5 orange marker objects in each separated rectangle, then crop the winner with diagonal asymmetry breaking ties.

    Input structure:
    - Green background cells separate the canvas into 3-4 disjoint rectangular regions.
    - Each region is a fully filled rectangle of a non-green base color with some orange marker pixels embedded inside it.
    - Orange pixels form one or more 4-connected marker objects of mixed small areas.

    Rule:
    - Identify every non-green rectangular region and crop it to its own bounding box.
    - Inside each region, count the orange 4-connected objects whose area is exactly 5 pixels.
    - Output the crop of the region with the largest such count.
    - If multiple regions tie for the largest count, choose the unique tied crop whose full pixel pattern is not symmetric across the main diagonal.

    Key insight:
    - The primary selector is a count over only the area-5 orange components; diagonal symmetry matters only as a tie-breaker between equally counted cropped regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _select_region(_extract_regions(g))["patch"]


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in cells) + 1
    return _normalize_shape(tuple((c, h - 1 - r) for r, c in cells))


def _shape_variants(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cur = _normalize_shape(base)
    seen = set()
    out = []
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rotate_shape(cur)
    return out


AREA5_BASES = [
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
]

DISTRACTOR_BASES = {
    1: [((0, 0),)],
    2: [((0, 0), (0, 1))],
    3: [((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (1, 1))],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
    ],
    7: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (4, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)),
    ],
}

AREA5_SHAPES = [variant for base in AREA5_BASES for variant in _shape_variants(base)]
DISTRACTOR_SHAPES = {
    size: [variant for base in bases for variant in _shape_variants(base)]
    for size, bases in DISTRACTOR_BASES.items()
}

CROSS5 = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


def _can_place(cells: set[tuple[int, int]], shape: tuple[tuple[int, int], ...], r0: int, c0: int, h: int, w: int) -> bool:
    trial = {(r0 + r, c0 + c) for r, c in shape}
    for rr, cc in trial:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if (rr, cc) in cells:
            return False
    for rr, cc in trial:
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) in cells and (nr, nc) not in trial:
                return False
    return True


def _place_shape(patch: np.ndarray, cells: set[tuple[int, int]], shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> None:
    for r, c in shape:
        rr = r0 + r
        cc = c0 + c
        patch[rr, cc] = MARKER
        cells.add((rr, cc))


def _try_place_random_shape(
    patch: np.ndarray,
    cells: set[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
) -> bool:
    h, w = patch.shape
    options = list(shapes)
    random.shuffle(options)
    for shape in options:
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        starts = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
        random.shuffle(starts)
        for r0, c0 in starts:
            if _can_place(cells, shape, r0, c0, h, w):
                _place_shape(patch, cells, shape, r0, c0)
                return True
    return False


def _count_nonempty_objects(patch: np.ndarray) -> int:
    return len(_components(patch == MARKER))


def _make_general_patch(h: int, w: int, fill: int, area5_count: int) -> np.ndarray | None:
    for _ in range(120):
        patch = np.full((h, w), int(fill), dtype=int)
        cells: set[tuple[int, int]] = set()
        ok = True
        for _ in range(area5_count):
            if not _try_place_random_shape(patch, cells, AREA5_SHAPES):
                ok = False
                break
        if not ok:
            continue

        extra_objects = random.randint(1, 3)
        if area5_count == 0:
            extra_objects = max(extra_objects, 2)
        for _ in range(extra_objects):
            size = random.choice((1, 2, 3, 4, 6, 7))
            if not _try_place_random_shape(patch, cells, DISTRACTOR_SHAPES[size]):
                ok = False
                break
        if not ok:
            continue

        if _count_area5_markers(patch) != area5_count:
            continue
        if _count_nonempty_objects(patch) < max(2, area5_count + 1):
            continue
        return patch
    return None


def _available_diagonal_centers(size: int, count: int) -> list[int] | None:
    candidates = list(range(1, size - 1))
    picks: list[int] = []
    random.shuffle(candidates)
    for center in sorted(candidates):
        if all(abs(center - other) >= 3 for other in picks):
            picks.append(center)
            if len(picks) == count:
                return sorted(picks)
    return None


def _find_single_cell(
    patch: np.ndarray,
    cells: set[tuple[int, int]],
    *,
    diagonal_only: bool = False,
    off_diagonal_only: bool = False,
) -> tuple[int, int] | None:
    h, w = patch.shape
    choices = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(choices)
    for r, c in choices:
        if diagonal_only and r != c:
            continue
        if off_diagonal_only and r == c:
            continue
        if not _can_place(cells, ((0, 0),), r, c, h, w):
            continue
        return r, c
    return None


def _make_tie_patch(size: int, fill: int, area5_count: int, asymmetric: bool) -> np.ndarray | None:
    if size < 4 or area5_count < 1:
        return None
    for _ in range(120):
        patch = np.full((size, size), int(fill), dtype=int)
        cells: set[tuple[int, int]] = set()
        centers = _available_diagonal_centers(size, area5_count)
        if centers is None:
            continue
        for center in centers:
            _place_shape(patch, cells, CROSS5, center - 1, center - 1)

        if random.random() < 0.6:
            diag = _find_single_cell(patch, cells, diagonal_only=True)
            if diag is not None:
                patch[diag] = MARKER
                cells.add(diag)

        if asymmetric:
            cell = _find_single_cell(patch, cells, off_diagonal_only=True)
            if cell is None:
                continue
            tr = (cell[1], cell[0])
            if patch[tr] == MARKER:
                continue
            patch[cell] = MARKER
            cells.add(cell)
        else:
            if random.random() < 0.5:
                choices = [(r, c) for r in range(size) for c in range(r + 1, size)]
                random.shuffle(choices)
                placed = False
                for r, c in choices:
                    if not _can_place(cells, ((0, 0),), r, c, size, size):
                        continue
                    if not _can_place(cells, ((0, 0),), c, r, size, size):
                        continue
                    patch[r, c] = MARKER
                    patch[c, r] = MARKER
                    cells.add((r, c))
                    cells.add((c, r))
                    placed = True
                    break
                if not placed and random.random() < 0.2:
                    continue

        if _count_area5_markers(patch) != area5_count:
            continue
        if asymmetric == _main_diag_symmetric(patch):
            continue
        return patch
    return None


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_REGION_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    min_h = max(15, max(heights) + 2)
    min_w = max(15, max(widths) + 2)
    max_h = 24
    max_w = 24
    for _ in range(180):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        canvas = np.full((h, w), BACKGROUND, dtype=int)
        boxes: list[tuple[int, int, int, int]] = []
        order = list(range(len(patches)))
        order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            ph, pw = patches[idx].shape
            starts = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
            random.shuffle(starts)
            placed = False
            for r0, c0 in starts:
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_boxes_separated(box, other) for other in boxes):
                    placements[idx] = (r0, c0)
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        for idx, (r0, c0) in placements.items():
            ph, pw = patches[idx].shape
            canvas[r0 : r0 + ph, c0 : c0 + pw] = patches[idx]
        if Counter(map(int, canvas.flatten())).most_common(1)[0][0] != BACKGROUND:
            continue
        return canvas
    return None


def _region_summary(grid: np.ndarray) -> list[dict]:
    regions = _extract_regions(grid)
    out = []
    for region in regions:
        patch = region["patch"]
        out.append(
            {
                "bbox": region["bbox"],
                "shape": patch.shape,
                "count5": _count_area5_markers(patch),
                "symmetric": _main_diag_symmetric(patch),
                "patch": patch,
            }
        )
    return out


def _distinct_shapes(patches: list[np.ndarray]) -> bool:
    shapes = [tuple(map(int, patch.shape)) for patch in patches]
    return len(set(shapes)) == len(shapes)


def _make_tie_case() -> np.ndarray | None:
    region_count = random.choice((3, 4))
    count5 = random.choice((1, 2, 3))
    fills = random.sample(FILL_COLORS, region_count)

    min_size = max(5, 3 * count5)
    sym_size = random.randint(min_size, min(10, min_size + 2))
    asym_size_choices = [s for s in range(min_size, min(11, min_size + 4)) if s != sym_size]
    if not asym_size_choices:
        return None
    asym_size = random.choice(asym_size_choices)

    sym_patch = _make_tie_patch(sym_size, fills[0], count5, asymmetric=False)
    asym_patch = _make_tie_patch(asym_size, fills[1], count5, asymmetric=True)
    if sym_patch is None or asym_patch is None:
        return None

    patches = [sym_patch, asym_patch]
    used_shapes = {tuple(sym_patch.shape), tuple(asym_patch.shape)}
    for i in range(2, region_count):
        for _ in range(80):
            h = random.randint(4, 9)
            w = random.randint(4, 9)
            if (h, w) in used_shapes:
                continue
            low_count = random.randint(0, count5 - 1)
            patch = _make_general_patch(h, w, fills[i], low_count)
            if patch is not None:
                patches.append(patch)
                used_shapes.add((h, w))
                break
        else:
            return None

    random.shuffle(patches)
    grid = _place_patches(patches)
    if grid is None:
        return None

    summary = _region_summary(grid)
    counts = [item["count5"] for item in summary]
    best = max(counts)
    winners = [item for item in summary if item["count5"] == best]
    if len(winners) != 2:
        return None
    if sum(not item["symmetric"] for item in winners) != 1:
        return None
    if not _distinct_shapes([item["patch"] for item in summary]):
        return None
    expected = [item["patch"] for item in winners if not item["symmetric"]][0]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _make_unique_case() -> np.ndarray | None:
    region_count = random.choice((3, 4))
    fills = random.sample(FILL_COLORS, region_count)
    max_count = random.choice((1, 2, 3))
    winner_idx = random.randrange(region_count)
    counts = [random.randint(0, max_count - 1) for _ in range(region_count)]
    counts[winner_idx] = max_count

    patches: list[np.ndarray] = []
    used_shapes: set[tuple[int, int]] = set()
    asym_distractor_assigned = False
    for i in range(region_count):
        for _ in range(100):
            if i == winner_idx:
                h = random.randint(max(5, max_count + 3), 10)
                w = random.randint(max(5, max_count + 3), 10)
                if (h, w) in used_shapes:
                    continue
                patch = _make_general_patch(h, w, fills[i], counts[i])
            else:
                make_asym = counts[i] > 0 and not asym_distractor_assigned and random.random() < 0.7
                if make_asym:
                    size = random.randint(5, 9)
                    if (size, size) in used_shapes:
                        continue
                    patch = _make_tie_patch(size, fills[i], counts[i] if counts[i] > 0 else 1, asymmetric=True)
                    if patch is not None and counts[i] == 0:
                        # keep the region asymmetric but ensure no size-5 objects when requested.
                        if _count_area5_markers(patch) != 0:
                            patch = None
                else:
                    h = random.randint(4, 9)
                    w = random.randint(4, 9)
                    if (h, w) in used_shapes:
                        continue
                    patch = _make_general_patch(h, w, fills[i], counts[i])
            if patch is None:
                continue
            if tuple(patch.shape) in used_shapes:
                continue
            patches.append(patch)
            used_shapes.add(tuple(map(int, patch.shape)))
            if i != winner_idx and patch.shape[0] == patch.shape[1] and not _main_diag_symmetric(patch):
                asym_distractor_assigned = True
            break
        else:
            return None

    grid = _place_patches(patches)
    if grid is None:
        return None
    summary = _region_summary(grid)
    counts_found = [item["count5"] for item in summary]
    if counts_found.count(max(counts_found)) != 1:
        return None
    if not _distinct_shapes([item["patch"] for item in summary]):
        return None
    selected = _select_region([{"patch": item["patch"], "bbox": item["bbox"]} for item in summary])["patch"]
    if not np.array_equal(main(grid), selected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_tie_case() if random.random() < 0.58 else _make_unique_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
