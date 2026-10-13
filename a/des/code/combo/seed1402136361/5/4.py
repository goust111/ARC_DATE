from __future__ import annotations

from itertools import combinations

import numpy as np


RED = 2
GREEN = 3
BG_COLOR = 0
PALETTE = (1, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RNG = np.random.default_rng()

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    base = _tight_mask(mask)
    variants: list[tuple[tuple[int, int], bytes]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        for transformed in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(transformed)
            variants.append((tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes()))
    return min(variants)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _bresenham(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p0
    r1, c1 = p1
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    cells: list[tuple[int, int]] = []

    while True:
        cells.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return cells


def _draw_axis_or_diag(center_a: tuple[float, float], center_b: tuple[float, float]) -> list[tuple[int, int]]:
    ra, ca = center_a
    rb, cb = center_b
    p0 = (int(round(ra)), int(round(ca)))
    p1 = (int(round(rb)), int(round(cb)))
    dr = rb - ra
    dc = cb - ca

    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return [p0]

    scores = []
    if abs(dc) > 1e-9:
        scores.append(("horizontal", abs(dc)))
    if abs(dr) > 1e-9:
        scores.append(("vertical", abs(dr)))
    if abs(dr) > 1e-9 and abs(dc) > 1e-9:
        scores.append(("diagonal", (abs(dr) + abs(dc)) / np.sqrt(2.0)))
    kind = max(scores, key=lambda item: item[1])[0]

    if kind == "horizontal" and p0[0] == p1[0]:
        row = p0[0]
        c0, c1 = sorted((p0[1], p1[1]))
        return [(row, c) for c in range(c0, c1 + 1)]

    if kind == "vertical" and p0[1] == p1[1]:
        col = p0[1]
        r0, r1 = sorted((p0[0], p1[0]))
        return [(r, col) for r in range(r0, r1 + 1)]

    if kind == "diagonal" and abs(p1[0] - p0[0]) == abs(p1[1] - p0[1]):
        step_r = 1 if p1[0] > p0[0] else -1
        step_c = 1 if p1[1] > p0[1] else -1
        length = abs(p1[0] - p0[0])
        return [(p0[0] + i * step_r, p0[1] + i * step_c) for i in range(length + 1)]

    return _bresenham(p0, p1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect objects that match another object's shape up to rotation/flip, recolor them,
    and connect each matching pair with a red segment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    if not components:
        return g.copy()

    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, component in enumerate(components):
        key = _canonical_key(_component_mask(component))
        groups.setdefault(key, []).append(idx)

    matched: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        matched.update(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.append((indices[i], indices[j]))

    out = g.copy()
    for idx in matched:
        coords = components[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = GREEN

    H, W = out.shape
    for left, right in pairs:
        for r, c in _draw_axis_or_diag(
            _bbox_center(components[left]["bbox"]),
            _bbox_center(components[right]["bbox"]),
        ):
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = RED

    return out


# --------------- generator ---------------

GRID_MIN = 23
GRID_MAX = 29
ANCHOR_STEP = 6
SHAPE_RETRIES = 200
GENERATION_RETRIES = 400
PAIR_COUNT_RANGE = (1, 3)  # sampled with integers(low, high)
UNIQUE_COUNT_RANGE = (1, 3)
ORIENTATIONS = ("horizontal", "vertical", "diag_down", "diag_up")


def _random_connected_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = int(RNG.choice([1, 3, 5]))
        w = int(RNG.choice([1, 3, 5]))
        if h * w < 3:
            continue

        min_area = 3
        area = int(RNG.integers(min_area, h * w + 1))
        center = (h // 2, w // 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True
        frontier = {center}

        while int(mask.sum()) < area:
            border: set[tuple[int, int]] = set()
            for r, c in frontier:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        border.add((nr, nc))
            if not border:
                break
            nr, nc = list(border)[int(RNG.integers(len(border)))]
            mask[nr, nc] = True
            frontier.add((nr, nc))

        tight = _tight_mask(mask)
        if tight.size == 0 or int(tight.sum()) != area:
            continue
        if tight.shape[0] % 2 == 0 or tight.shape[1] % 2 == 0:
            continue
        if not tight[tight.shape[0] // 2, tight.shape[1] // 2]:
            continue
        return tight

    raise RuntimeError("failed to build a connected odd-sized mask")


def _apply_transform(mask: np.ndarray, transform_id: int) -> np.ndarray:
    out = np.rot90(mask, transform_id % 4)
    if transform_id >= 4:
        out = np.fliplr(out)
    return out.copy()


def _mask_cells(mask: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    r0 = cr - mask.shape[0] // 2
    c0 = cc - mask.shape[1] // 2
    coords = np.argwhere(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in coords}


def _fits(mask: np.ndarray, center: tuple[int, int], H: int, W: int) -> bool:
    cr, cc = center
    r0 = cr - mask.shape[0] // 2
    c0 = cc - mask.shape[1] // 2
    return 0 <= r0 and 0 <= c0 and r0 + mask.shape[0] <= H and c0 + mask.shape[1] <= W


def _anchor_points(H: int, W: int) -> list[tuple[int, int]]:
    rows = list(range(2, H - 2, ANCHOR_STEP))
    cols = list(range(2, W - 2, ANCHOR_STEP))
    return [(r, c) for r in rows for c in cols]


def _matches_orientation(a: tuple[int, int], b: tuple[int, int], kind: str) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if kind == "horizontal":
        return dr == 0 and dc != 0
    if kind == "vertical":
        return dc == 0 and dr != 0
    if kind == "diag_down":
        return dr != 0 and dr == dc
    return dr != 0 and dr == -dc


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_unique_masks(pair_count: int, unique_count: int) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    matched_groups: list[list[np.ndarray]] = []
    unique_masks: list[np.ndarray] = []

    for _ in range(pair_count):
        while True:
            proto = _random_connected_mask()
            key = _canonical_key(proto)
            if key in used_keys:
                continue
            used_keys.add(key)
            matched_groups.append(
                [
                    _apply_transform(proto, int(RNG.integers(0, 8))),
                    _apply_transform(proto, int(RNG.integers(0, 8))),
                ]
            )
            break

    for _ in range(unique_count):
        while True:
            mask = _random_connected_mask()
            key = _canonical_key(mask)
            if key in used_keys:
                continue
            used_keys.add(key)
            unique_masks.append(mask)
            break

    return matched_groups, unique_masks


def _place_pair(
    pair_masks: list[np.ndarray],
    orientation: str,
    anchors: list[tuple[int, int]],
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_lines: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = []
    for a, b in combinations(anchors, 2):
        if not _matches_orientation(a, b, orientation):
            continue
        for first_center, second_center in ((a, b), (b, a)):
            if not _fits(pair_masks[0], first_center, H, W) or not _fits(pair_masks[1], second_center, H, W):
                continue
            first_cells = _mask_cells(pair_masks[0], first_center)
            second_cells = _mask_cells(pair_masks[1], second_center)
            pair_cells = first_cells | second_cells
            if any(occupied[r, c] or reserved_lines[r, c] for r, c in pair_cells):
                continue

            line = _draw_axis_or_diag(first_center, second_center)
            blocked = False
            for r, c in line:
                if not (0 <= r < H and 0 <= c < W):
                    blocked = True
                    break
                if (r, c) in pair_cells:
                    continue
                if occupied[r, c] or reserved_lines[r, c]:
                    blocked = True
                    break
            if blocked:
                continue
            candidates.append((first_center, second_center, first_cells, second_cells, line))

    if not candidates:
        return None
    return candidates[int(RNG.integers(len(candidates)))]


def _place_unique(
    mask: np.ndarray,
    anchors: list[tuple[int, int]],
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_lines: np.ndarray,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    candidates = []
    for center in anchors:
        if not _fits(mask, center, H, W):
            continue
        cells = _mask_cells(mask, center)
        if any(occupied[r, c] or reserved_lines[r, c] for r, c in cells):
            continue
        candidates.append((center, cells))

    if not candidates:
        return None
    return candidates[int(RNG.integers(len(candidates)))]


def _gen_attempt() -> np.ndarray | None:
    H = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    W = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    anchors = _anchor_points(H, W)
    if len(anchors) < 9:
        return None

    pair_count = int(RNG.integers(*PAIR_COUNT_RANGE))
    unique_count = int(RNG.integers(*UNIQUE_COUNT_RANGE))
    total_objects = 2 * pair_count + unique_count
    if total_objects > len(PALETTE):
        return None

    matched_groups, unique_masks = _sample_unique_masks(pair_count, unique_count)
    colors = RNG.choice(np.asarray(PALETTE, dtype=int), size=total_objects, replace=False)
    color_index = 0

    grid = np.full((H, W), BG_COLOR, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    reserved_lines = np.zeros((H, W), dtype=bool)

    pair_orientations = list(RNG.choice(np.asarray(ORIENTATIONS), size=pair_count, replace=False))
    placements: list[tuple[set[tuple[int, int]], int]] = []

    for orientation, pair_masks in zip(pair_orientations, matched_groups):
        placed = _place_pair(pair_masks, orientation, anchors, H, W, occupied, reserved_lines)
        if placed is None:
            return None
        first_center, second_center, first_cells, second_cells, line = placed
        del first_center, second_center

        first_color = int(colors[color_index])
        second_color = int(colors[color_index + 1])
        color_index += 2
        placements.append((first_cells, first_color))
        placements.append((second_cells, second_color))
        for r, c in first_cells | second_cells:
            occupied[r, c] = True
        for r, c in line:
            reserved_lines[r, c] = True

    for mask in unique_masks:
        placed = _place_unique(mask, anchors, H, W, occupied, reserved_lines)
        if placed is None:
            return None
        _center, cells = placed
        del _center

        color = int(colors[color_index])
        color_index += 1
        placements.append((cells, color))
        for r, c in cells:
            occupied[r, c] = True

    for cells, color in placements:
        _paint_cells(grid, cells, color)

    components = _find_components(grid, BG_COLOR)
    if len(components) != total_objects:
        return None

    counts = sorted(len(indices) for indices in _group_indices_by_shape(components).values())
    expected_counts = sorted([2] * pair_count + [1] * unique_count)
    if counts != expected_counts:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def _group_indices_by_shape(components: list[dict]) -> dict[tuple[tuple[int, int], bytes], list[int]]:
    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, component in enumerate(components):
        groups.setdefault(_canonical_key(_component_mask(component)), []).append(idx)
    return groups


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
