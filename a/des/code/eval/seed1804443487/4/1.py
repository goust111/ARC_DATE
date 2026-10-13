from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
SIDES = ["left", "right", "top", "bottom"]
COLORS = list(range(10))

GEN_RETRIES = 300
SHAPE_RETRIES = 80
MARKER_RETRIES = 80
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _normalize_shape(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    min_r = min(rs)
    min_c = min(cs)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = np.array(shape, dtype=int)
    variants = set()
    for reflect in (False, True):
        pts = arr.copy()
        if reflect:
            pts[:, 1] *= -1
        for _ in range(4):
            norm = _normalize_shape([tuple(p) for p in pts.tolist()])
            variants.add(norm)
            pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return sorted(variants)


def _canonical_shape(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_shape_transforms(_normalize_shape(cells)))


def _partition_hole(
    hole_cells: list[tuple[int, int]] | set[tuple[int, int]],
    markers: list[tuple[tuple[int, int], int]],
) -> dict[tuple[int, int], int]:
    hole_set = set(hole_cells)
    ordered = sorted(markers, key=lambda item: (item[0][0], item[0][1], item[1]))
    owners: dict[tuple[int, int], int] = {}
    q = deque()

    for pos, color in ordered:
        for nb in _neighbors4(*pos):
            if nb in hole_set and nb not in owners:
                owners[nb] = color
                q.append((nb, color))

    while q:
        (r, c), color = q.popleft()
        for nb in _neighbors4(r, c):
            if nb in hole_set and nb not in owners:
                owners[nb] = color
                q.append((nb, color))

    assert len(owners) == len(hole_set)
    return owners


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    fg_components = _components_mask(g != bg)
    border_components = [i for i, comp in enumerate(fg_components) if _touches_border(comp, h, w)]
    assert border_components, "expected a border-touching frame component"

    frame_idx = max(border_components, key=lambda i: len(fg_components[i]))
    frame_comp = fg_components[frame_idx]
    frame_set = set(frame_comp)
    frame_color = Counter(int(g[r, c]) for r, c in frame_comp).most_common(1)[0][0]

    out = g.copy()
    blob_shapes = Counter()
    for i, comp in enumerate(fg_components):
        if i == frame_idx:
            continue
        blob_shapes[_canonical_shape(comp)] += 1
        for r, c in comp:
            out[r, c] = bg

    for hole in _components_mask(g == bg):
        if _touches_border(hole, h, w):
            continue
        hole_set = set(hole)
        boundary = set()
        valid = True
        for r, c in hole:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in hole_set or int(g[nr, nc]) == bg:
                    continue
                boundary.add((nr, nc))
                if (nr, nc) not in frame_set:
                    valid = False
        if not valid or not boundary:
            continue

        shape = _canonical_shape(hole)
        if blob_shapes[shape] <= 0:
            continue
        blob_shapes[shape] -= 1

        markers = [((r, c), int(g[r, c])) for r, c in boundary if int(g[r, c]) != frame_color]
        assert markers, "each matched hole must have at least one marker"
        fill = _partition_hole(hole, markers)
        for (r, c), color in fill.items():
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(np.asarray(grid, dtype=int))


def _random_shape(
    rng: random.Random,
    min_area: int = 4,
    max_area: int = 7,
    max_h: int = 4,
    max_w: int = 4,
) -> tuple[tuple[int, int], ...] | None:
    target = rng.randint(min_area, max_area)
    cells = {(0, 0)}
    while len(cells) < target:
        candidates = []
        for r, c in sorted(cells):
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells:
                    continue
                cand = set(cells)
                cand.add((nr, nc))
                norm = _normalize_shape(cand)
                h, w = _shape_dims(norm)
                if h <= max_h and w <= max_w:
                    candidates.append(norm)
        if not candidates:
            return None
        cells = set(rng.choice(candidates))

    norm = _normalize_shape(cells)
    h, w = _shape_dims(norm)
    if h < 2 or w < 2:
        return None
    return norm


def _translate_shape(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(dr + r, dc + c) for r, c in shape]


def _expand_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int, dirs) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _frame_bbox(side: str, h: int, w: int, rng: random.Random) -> tuple[int, int, int, int]:
    if side in ("left", "right"):
        frame_h = rng.randint(max(10, h // 2), h - 2)
        frame_w = rng.randint(7, min(10, w - 8))
        top = rng.randint(1, h - frame_h - 1)
        if side == "left":
            left = 0
        else:
            left = w - frame_w
        return top, left, top + frame_h - 1, left + frame_w - 1

    frame_h = rng.randint(7, min(10, h - 8))
    frame_w = rng.randint(max(10, w // 2), w - 2)
    left = rng.randint(1, w - frame_w - 1)
    if side == "top":
        top = 0
    else:
        top = h - frame_h
    return top, left, top + frame_h - 1, left + frame_w - 1


def _place_holes(
    rng: random.Random,
    base_shapes: list[tuple[tuple[int, int], ...]],
    frame_bbox: tuple[int, int, int, int],
) -> list[list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h < 3 or inner_w < 3:
        return None

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    placed: list[list[tuple[int, int]]] = []

    for shape in sorted(base_shapes, key=len, reverse=True):
        candidates = []
        variants = _shape_transforms(shape)
        rng.shuffle(variants)
        for variant in variants:
            sh, sw = _shape_dims(variant)
            for rr in range(inner_h - sh + 1):
                for cc in range(inner_w - sw + 1):
                    abs_cells = _translate_shape(variant, r0 + 1 + rr, c0 + 1 + cc)
                    cell_set = set(abs_cells)
                    if cell_set & blocked:
                        continue
                    candidates.append(abs_cells)
        if not candidates:
            return None
        choice = rng.choice(candidates)
        placed.append(choice)
        choice_set = set(choice)
        occupied |= choice_set
        blocked |= _expand_cells(choice_set, r1 + 1, c1 + 1, DIRS4 + [(0, 0)])

    return placed


def _marker_distances(
    hole_cells: list[tuple[int, int]] | set[tuple[int, int]],
    marker_pos: tuple[int, int],
) -> dict[tuple[int, int], int]:
    hole_set = set(hole_cells)
    q = deque()
    dist: dict[tuple[int, int], int] = {}
    for nb in _neighbors4(*marker_pos):
        if nb in hole_set and nb not in dist:
            dist[nb] = 1
            q.append(nb)
    while q:
        r, c = q.popleft()
        for nb in _neighbors4(r, c):
            if nb in hole_set and nb not in dist:
                dist[nb] = dist[(r, c)] + 1
                q.append(nb)
    return dist


def _markers_are_unambiguous(
    hole_cells: list[tuple[int, int]],
    marker_positions: list[tuple[int, int]],
) -> bool:
    dists = [_marker_distances(hole_cells, pos) for pos in marker_positions]
    counts = [0] * len(marker_positions)
    for cell in hole_cells:
        vals = [dist.get(cell, 10**9) for dist in dists]
        best = min(vals)
        winners = [i for i, v in enumerate(vals) if v == best]
        if len(winners) != 1:
            return False
        counts[winners[0]] += 1
    return all(count > 0 for count in counts)


def _choose_markers(
    rng: random.Random,
    hole_cells: list[tuple[int, int]],
    frame_color: int,
    marker_palette: list[int],
) -> list[tuple[tuple[int, int], int]] | None:
    hole_set = set(hole_cells)
    boundary = sorted(
        {
            (nr, nc)
            for r, c in hole_cells
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) not in hole_set
        }
    )
    if not boundary:
        return None

    marker_count = rng.choice([1, 2, 3])
    marker_count = min(marker_count, len(boundary), len(marker_palette))

    for _ in range(MARKER_RETRIES):
        positions = rng.sample(boundary, marker_count)
        if not _markers_are_unambiguous(hole_cells, positions):
            continue
        colors = rng.sample(marker_palette, marker_count)
        return list(zip(positions, colors))
    return None


def _place_blobs(
    rng: random.Random,
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    shapes: list[tuple[tuple[int, int], ...]],
    blob_palette: list[int],
) -> list[tuple[list[tuple[int, int]], int]] | None:
    r0, c0, r1, c1 = frame_bbox
    blocked = _expand_cells(
        [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
        h,
        w,
        DIRS8,
    )
    for c in range(w):
        blocked.add((0, c))
        blocked.add((h - 1, c))
    for r in range(h):
        blocked.add((r, 0))
        blocked.add((r, w - 1))

    placed: list[tuple[list[tuple[int, int]], int]] = []
    for shape in shapes:
        candidates: list[list[tuple[int, int]]] = []
        variants = _shape_transforms(shape)
        rng.shuffle(variants)
        for variant in variants:
            sh, sw = _shape_dims(variant)
            for rr in range(1, h - sh):
                for cc in range(1, w - sw):
                    abs_cells = _translate_shape(variant, rr, cc)
                    cell_set = set(abs_cells)
                    if cell_set & blocked:
                        continue
                    candidates.append(abs_cells)
        if not candidates:
            return None
        cells = rng.choice(candidates)
        color = rng.choice(blob_palette)
        placed.append((cells, color))
        blocked |= _expand_cells(cells, h, w, DIRS8)
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 25)
    w = rng.randint(18, 25)
    side = rng.choice(SIDES)

    color_order = rng.sample(COLORS, len(COLORS))
    bg = color_order[0]
    frame_color = color_order[1]
    blob_palette = color_order[2:4]
    marker_palette = color_order[4:]
    if len(blob_palette) < 2 or len(marker_palette) < 3:
        return None

    frame_bbox = _frame_bbox(side, h, w, rng)
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h < 4 or inner_w < 4:
        return None

    hole_count = rng.choice([1, 2, 3])
    if inner_h * inner_w < hole_count * 7:
        hole_count = max(1, min(2, hole_count))

    base_shapes: list[tuple[tuple[int, int], ...]] = []
    for _ in range(SHAPE_RETRIES):
        shape = _random_shape(rng)
        if shape is None:
            continue
        if _canonical_shape(shape) in {_canonical_shape(s) for s in base_shapes}:
            continue
        base_shapes.append(shape)
        if len(base_shapes) == hole_count:
            break
    if len(base_shapes) != hole_count:
        return None

    hole_cells = _place_holes(rng, base_shapes, frame_bbox)
    if hole_cells is None or len(hole_cells) != hole_count:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = frame_color

    all_markers: list[tuple[tuple[int, int], int]] = []
    hole_to_markers: list[list[tuple[tuple[int, int], int]]] = []
    for hole in hole_cells:
        for r, c in hole:
            grid[r, c] = bg
        markers = _choose_markers(rng, hole, frame_color, marker_palette)
        if markers is None:
            return None
        hole_to_markers.append(markers)
        all_markers.extend(markers)
        for (mr, mc), color in markers:
            grid[mr, mc] = color

    blob_shapes = [_canonical_shape(hole) for hole in hole_cells]
    blobs = _place_blobs(rng, h, w, frame_bbox, blob_shapes, blob_palette)
    if blobs is None or len(blobs) != hole_count:
        return None
    for cells, color in blobs:
        for r, c in cells:
            grid[r, c] = color

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(v for k, v in counts.items() if k != bg):
        return None

    fg_components = _components_mask(grid != bg)
    if len(fg_components) != hole_count + 1:
        return None
    border_components = [comp for comp in fg_components if _touches_border(comp, h, w)]
    if len(border_components) != 1:
        return None

    expected = grid.copy()
    for cells, _color in blobs:
        for r, c in cells:
            expected[r, c] = bg
    for hole, markers in zip(hole_cells, hole_to_markers):
        fill = _partition_hole(hole, markers)
        for (r, c), color in fill.items():
            expected[r, c] = color

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
