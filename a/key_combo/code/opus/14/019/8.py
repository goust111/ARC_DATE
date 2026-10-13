from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
OBJECT_COLORS = (1, 3, 4, 5, 8)

GRID_ROWS_RANGE = (14, 18)
GRID_COLS_RANGE = (14, 18)
MAX_GENERATION_ATTEMPTS = 500
PADDING_RADIUS = 2  # Enforce at least two background cells between objects.

MARKER_PROFILES = (
    (1, 2, 3, 4),
    (2, 3, 4, 5),
    (1, 2, 3, 4, 5),
    (2, 3, 4, 5, 6),
)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    comps = _connected_components(grid != BG)
    assert 4 <= len(comps) <= 5, "expected 4-5 separated foreground objects"

    infos: list[dict] = []
    for idx, coords in enumerate(comps):
        values = grid[coords[:, 0], coords[:, 1]]
        colors = set(map(int, values))
        assert colors - {RED}, "each object needs a non-red body color"
        assert len(colors - {RED}) == 1, "each object must use exactly one non-red color"
        base_color = next(iter(colors - {RED}))
        red_count = int(np.count_nonzero(values == RED))
        assert red_count >= 1, "every generated object should contain red markers"
        r0, c0, r1, c1 = _bbox(coords)
        infos.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(base_color),
                "red_count": red_count,
                "area": int(len(coords)),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return infos


def _layered_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.zeros((side, side), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _sorted_survivors(infos: list[dict]) -> list[dict]:
    survivors = [info for info in infos if info["red_count"] <= 3]
    assert 2 <= len(survivors) <= 4, "survivor count must determine a valid output size"
    survivors.sort(key=lambda info: (info["red_count"], info["bbox"][1], info["bbox"][0]))
    return survivors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter multi-color objects by their red-marker counts, then paint surviving colors into concentric square layers.

    Input structure:
    - The grid is mostly black background with 4-5 separated 4-connected foreground objects.
    - Each object is a mixed-color component made from one non-red body color plus one or more embedded red marker pixels.
    - Different objects use different body colors, and the red markers are counted within their enclosing object.

    Rule:
    - Treat each mixed-color connected component as one object and count how many red pixels it contains.
    - Erase every object whose red-marker count is strictly greater than 3.
    - Sort the surviving objects by red-marker count ascending, breaking ties by the leftmost then topmost bbox corner.
    - Let M be the number of survivors, create a black output of size (2M-1) x (2M-1), and fill concentric square layers from outside to inside using the survivors' non-red body colors in that order.

    Key insight:
    - The same red-marker count controls three linked decisions at once: whether an object survives, where its color sits in the nesting order, and how many nested layers the output contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    infos = _analyze_objects(g)
    survivors = _sorted_survivors(infos)
    return _layered_output([info["color"] for info in survivors])


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    arr = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    return arr


def _shape_is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nb in neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _generate_shape(area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({nb for cell in cells for nb in neighbors4(*cell)} - cells)
            random.shuffle(frontier)
            added = False
            for candidate in frontier:
                trial = cells | {candidate}
                shape = _normalize_shape(trial)
                h = int(shape[:, 0].max()) + 1
                w = int(shape[:, 1].max()) + 1
                if h <= 5 and w <= 5:
                    cells = trial
                    added = True
                    break
            if not added:
                break
        if len(cells) != area:
            continue

        shape = _normalize_shape(cells)
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        if h < 2 or w < 2:
            continue
        if area == h * w and random.random() < 0.8:
            continue
        return shape
    return None


def _choose_markers(shape: np.ndarray, marker_count: int) -> set[tuple[int, int]] | None:
    cells = [tuple(map(int, rc)) for rc in shape]
    cell_set = set(cells)
    if len(cells) <= marker_count:
        return None
    for _ in range(200):
        markers = set(random.sample(cells, marker_count))
        base = cell_set - markers
        if len(base) < 2 or not _shape_is_connected(base):
            continue
        if all(any(nb in base for nb in neighbors4(r, c)) for r, c in markers):
            return markers
    return None


def _build_patch(marker_count: int, color: int, area: int) -> np.ndarray | None:
    shape = _generate_shape(area)
    if shape is None:
        return None
    markers = _choose_markers(shape, marker_count)
    if markers is None:
        return None
    h = int(shape[:, 0].max()) + 1
    w = int(shape[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    for r, c in markers:
        patch[int(r), int(c)] = RED
    return patch


def _mark_blocked(blocked: np.ndarray, cells: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, int(r) - PADDING_RADIUS)
        r1 = min(h, int(r) + PADDING_RADIUS + 1)
        c0 = max(0, int(c) - PADDING_RADIUS)
        c1 = min(w, int(c) + PADDING_RADIUS + 1)
        blocked[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    offsets = np.argwhere(patch != BG)
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        rr = offsets[:, 0] + r0
        for c0 in range(w - pw + 1):
            cc = offsets[:, 1] + c0
            if not blocked[rr, cc].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    rr = offsets[:, 0] + r0
    cc = offsets[:, 1] + c0
    grid[rr, cc] = patch[offsets[:, 0], offsets[:, 1]]
    placed = np.column_stack([rr, cc]).astype(int)
    _mark_blocked(blocked, placed)
    return placed, (r0, c0)


def _choose_areas(marker_counts: tuple[int, ...]) -> list[int] | None:
    for _ in range(80):
        areas = []
        for count in marker_counts:
            min_area = max(4, int(count) + 2 + (1 if count >= 5 else 0))
            if min_area > 10:
                return None
            areas.append(random.randint(min_area, 10))
        survivors = [(count, area) for count, area in zip(marker_counts, areas) if count <= 3]
        survivor_areas = [area for _count, area in survivors]
        if len(set(survivor_areas)) != len(survivor_areas):
            continue
        by_count = [count for count, _area in sorted(survivors)]
        by_area = [count for count, _area in sorted(survivors, key=lambda item: item[1])]
        if by_count == by_area:
            continue
        return areas
    return None


def _chebyshev_gap_ok(infos: list[dict], min_gap: int = 3) -> bool:
    for i, info_a in enumerate(infos):
        a = info_a["coords"]
        for info_b in infos[i + 1 :]:
            b = info_b["coords"]
            d = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2).min()
            if int(d) < min_gap:
                return False
    return True


def _output_from_order(infos: list[dict]) -> np.ndarray:
    return _layered_output([info["color"] for info in infos])


def _semantic_instance_ok(grid: np.ndarray) -> bool:
    try:
        infos = _analyze_objects(grid)
    except AssertionError:
        return False

    counts = [info["red_count"] for info in infos]
    if len(counts) != len(set(counts)):
        return False
    if 3 not in counts or not any(count > 3 for count in counts):
        return False
    if len({info["color"] for info in infos}) != len(infos):
        return False
    if not _chebyshev_gap_ok(infos):
        return False

    survivors = _sorted_survivors(infos)
    if len(survivors) not in (2, 3):
        return False

    true_out = _output_from_order(survivors)
    alternatives = [
        _output_from_order(sorted(survivors, key=lambda info: (-info["red_count"], info["bbox"][1], info["bbox"][0]))),
        _output_from_order(sorted(survivors, key=lambda info: (info["area"], info["bbox"][1], info["bbox"][0]))),
        _output_from_order(sorted(survivors, key=lambda info: (info["bbox"][1], info["bbox"][0]))),
        _output_from_order(sorted([info for info in infos if info["red_count"] < 3], key=lambda info: (info["red_count"], info["bbox"][1], info["bbox"][0]))),
        _output_from_order(sorted([info for info in infos if info["red_count"] <= 4], key=lambda info: (info["red_count"], info["bbox"][1], info["bbox"][0]))),
    ]
    all_sorted = sorted(infos, key=lambda info: (info["red_count"], info["bbox"][1], info["bbox"][0]))
    alternatives.append(_output_from_order(all_sorted))

    for alt in alternatives:
        if np.array_equal(alt, true_out):
            return False

    true_ids = [info["id"] for info in survivors]
    area_ids = [info["id"] for info in sorted(survivors, key=lambda info: info["area"])]
    pos_ids = [info["id"] for info in sorted(survivors, key=lambda info: (info["bbox"][1], info["bbox"][0]))]
    if true_ids == area_ids or true_ids == pos_ids:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    marker_counts = random.choice(MARKER_PROFILES)
    colors = random.sample(list(OBJECT_COLORS), len(marker_counts))
    areas = _choose_areas(marker_counts)
    if areas is None:
        return None

    specs = list(zip(marker_counts, colors, areas))
    random.shuffle(specs)
    patches: list[np.ndarray] = []
    for marker_count, color, area in specs:
        patch = _build_patch(marker_count, color, area)
        if patch is None:
            return None
        patches.append(patch)

    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    grid = np.zeros((rows, cols), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for i in order:
        placed = _place_patch(grid, blocked, patches[i])
        if placed is None:
            return None

    if not _semantic_instance_ok(grid):
        return None

    try:
        infos = _analyze_objects(grid)
        expected = _output_from_order(_sorted_survivors(infos))
    except AssertionError:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
