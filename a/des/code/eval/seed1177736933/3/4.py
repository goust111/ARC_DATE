from __future__ import annotations

from collections import deque
import math

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
_RNG = np.random.default_rng()

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (19, 27)
OBJECT_COUNT_RANGE = (3, 6)
GENERATE_RETRIES = 400
SHAPE_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    seen = set()
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen.add(start)
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _is_threefold(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    center = coords.mean(axis=0)
    base = {tuple(map(int, xy)) for xy in coords}
    for step in (1, 2):
        theta = 2.0 * math.pi * step / 3.0
        ct = math.cos(theta)
        st = math.sin(theta)
        rotated: set[tuple[int, int]] = set()
        for r, c in coords:
            rr = center[0] + ct * (r - center[0]) - st * (c - center[1])
            cc = center[1] + st * (r - center[0]) + ct * (c - center[1])
            rr_i = int(round(rr))
            cc_i = int(round(cc))
            if abs(rr - rr_i) > 1e-6 or abs(cc - cc_i) > 1e-6:
                return False
            rotated.add((rr_i, cc_i))
        if rotated != base:
            return False
    return True


def _symmetry_order(mask: np.ndarray) -> int:
    mask = _crop_mask(np.asarray(mask, dtype=bool))
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask, 1)):
        return 4
    if _is_threefold(mask):
        return 3
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _background_color(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10).argmax())


def _find_strip(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    cols = np.flatnonzero(grid[0] != bg)
    if cols.size < 4:
        raise ValueError("missing top color strip")
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))

    for run_start, run_end in runs:
        if run_end - run_start + 1 < 4:
            continue
        for c0 in range(run_start, run_end - 3 + 1):
            colors = [int(v) for v in grid[0, c0 : c0 + 4]]
            if len(set(colors)) == 4 and all(color != bg for color in colors):
                return c0, colors
    raise ValueError("could not parse top color strip")


def _extract_objects(grid: np.ndarray, bg: int, strip_start: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    mask = g != bg
    mask[0, strip_start : strip_start + 4] = False
    visited = np.zeros_like(mask, dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not mask[r, c] or visited[r, c]:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < g.shape[0]
                        and 0 <= nc < g.shape[1]
                        and mask[nr, nc]
                        and not visited[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            obj_mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            obj_mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            objects.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": obj_mask,
                }
            )
    return objects


def _draw_outer_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    top = max(0, int(r0) - 1)
    left = max(0, int(c0) - 1)
    bottom = min(out.shape[0] - 1, int(r1) + 1)
    right = min(out.shape[1] - 1, int(c1) + 1)
    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Color each object's outer bounding frame using the strip color indexed by its rotation symmetry order."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    strip_start, strip_colors = _find_strip(g, bg)
    objects = _extract_objects(g, bg, strip_start)

    out = g.copy()
    for obj in objects:
        order = _symmetry_order(np.asarray(obj["mask"], dtype=bool))
        if not (1 <= order <= len(strip_colors)):
            raise ValueError(f"unsupported symmetry order: {order}")
        _draw_outer_border(out, obj["bbox"], strip_colors[order - 1])
    return out


def _orbit_touches(mask: np.ndarray, orbit: list[tuple[int, int]]) -> bool:
    if not mask.any():
        return True
    h, w = mask.shape
    for r, c in orbit:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _apply_orbit(mask: np.ndarray, orbit: list[tuple[int, int]]) -> None:
    for r, c in orbit:
        mask[int(r), int(c)] = True


def _orbits_180(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {(r, c), (h - 1 - r, w - 1 - c)}
            seen |= orbit
            orbits.append(sorted(orbit))
    return orbits


def _orbits_90(n: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = set()
            rr, cc = r, c
            for _ in range(4):
                orbit.add((rr, cc))
                rr, cc = cc, n - 1 - rr
            seen |= orbit
            orbits.append(sorted(orbit))
    return orbits


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return _crop_mask(out)


def _mask_ok(mask: np.ndarray, order: int) -> bool:
    mask = _crop_mask(mask)
    h, w = mask.shape
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return False
    if int(mask.sum()) < 4 or int(mask.sum()) >= h * w:
        return False
    if not _mask_connected(mask):
        return False
    return _symmetry_order(mask) == order


def _make_order4_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        n = int(_RNG.choice([3, 5]))
        orbits = _orbits_90(n)
        center_idx = next(i for i, orbit in enumerate(orbits) if len(orbit) == 1)
        mask = np.zeros((n, n), dtype=bool)
        used = {center_idx}
        _apply_orbit(mask, orbits[center_idx])

        min_orbits = 2
        max_orbits = min(len(orbits), 5)
        target_orbits = int(_RNG.integers(min_orbits, max_orbits + 1))
        while len(used) < target_orbits:
            candidates = [i for i, orbit in enumerate(orbits) if i not in used and _orbit_touches(mask, orbit)]
            if not candidates:
                break
            idx = int(_RNG.choice(candidates))
            used.add(idx)
            _apply_orbit(mask, orbits[idx])
            if len(used) >= min_orbits and _RNG.random() < 0.35:
                break

        mask = _crop_mask(mask)
        if _mask_ok(mask, 4):
            return mask
    raise RuntimeError("failed to generate order-4 mask")


def _make_order2_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = int(_RNG.choice([3, 5]))
        w = int(_RNG.choice([3, 5]))
        orbits = _orbits_180(h, w)
        center_idx = next(i for i, orbit in enumerate(orbits) if len(orbit) == 1)
        mask = np.zeros((h, w), dtype=bool)
        used = {center_idx}
        _apply_orbit(mask, orbits[center_idx])

        min_orbits = 3
        max_orbits = min(len(orbits), 7)
        target_orbits = int(_RNG.integers(min_orbits, max_orbits + 1))
        while len(used) < target_orbits:
            candidates = [i for i, orbit in enumerate(orbits) if i not in used and _orbit_touches(mask, orbit)]
            if not candidates:
                break
            idx = int(_RNG.choice(candidates))
            used.add(idx)
            _apply_orbit(mask, orbits[idx])
            if len(used) >= min_orbits and _RNG.random() < 0.30:
                break

        mask = _random_transform(mask)
        if _mask_ok(mask, 2):
            return mask
    raise RuntimeError("failed to generate order-2 mask")


def _make_order1_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = int(_RNG.integers(3, 6))
        w = int(_RNG.integers(3, 6))
        max_area = h * w - 1
        min_area = max(4, (h + w) // 2)
        area = int(_RNG.integers(min_area, max_area + 1))
        mask = np.zeros((h, w), dtype=bool)
        seed = (int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))
        mask[seed] = True

        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(_RNG.integers(0, len(frontier)))]
            mask[nr, nc] = True

        mask = _random_transform(mask)
        if _mask_ok(mask, 1):
            return mask
    raise RuntimeError("failed to generate order-1 mask")


def _make_mask(order: int) -> np.ndarray:
    if order == 4:
        return _make_order4_mask()
    if order == 2:
        return _make_order2_mask()
    if order == 1:
        return _make_order1_mask()
    raise ValueError(f"unsupported target order: {order}")


def _place_object(
    grid: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    row_choices = range(2, grid.shape[0] - h)
    col_choices = range(1, grid.shape[1] - w)
    candidates = [(r, c) for r in row_choices for c in col_choices]
    if not candidates:
        return None
    _RNG.shuffle(candidates)

    for r0, c0 in candidates:
        if forbidden[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
            continue
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = color
        rr0 = max(0, r0 - 2)
        cc0 = max(0, c0 - 2)
        rr1 = min(grid.shape[0], r0 + h + 2)
        cc1 = min(grid.shape[1], c0 + w + 2)
        forbidden[rr0:rr1, cc0:cc1] = True
        return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    return None


def _expected_output(
    inp: np.ndarray,
    objects: list[dict[str, int]],
    strip_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        _draw_outer_border(out, (obj["r0"], obj["c0"], obj["r1"], obj["c1"]), strip_colors[obj["order"] - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))
    n_objects = int(_RNG.integers(*OBJECT_COUNT_RANGE))
    if w < 4 or h < 8:
        return None

    all_colors = [int(c) for c in _RNG.choice(np.arange(1, 10), size=4 + n_objects, replace=False)]
    strip_colors = all_colors[:4]
    object_colors = all_colors[4:]
    strip_start = int(_RNG.integers(0, w - 4 + 1))

    orders = [1, 2, 4]
    extras = [int(_RNG.choice([1, 2, 4])) for _ in range(n_objects - 3)]
    orders.extend(extras)
    _RNG.shuffle(orders)

    grid = np.zeros((h, w), dtype=int)
    grid[0, strip_start : strip_start + 4] = strip_colors
    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[0, :] = True

    items = []
    for order, color in zip(orders, object_colors):
        mask = _make_mask(int(order))
        items.append(
            {
                "order": int(order),
                "color": int(color),
                "mask": mask,
                "area": int(mask.shape[0] * mask.shape[1]),
            }
        )
    items.sort(key=lambda item: int(item["area"]), reverse=True)

    placed: list[dict[str, int]] = []
    for item in items:
        bbox = _place_object(grid, forbidden, np.asarray(item["mask"], dtype=bool), int(item["color"]))
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        placed.append({"order": int(item["order"]), "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    expected = _expected_output(grid, placed, strip_colors)
    extracted = _extract_objects(grid, BG, strip_start)
    if len(extracted) != n_objects:
        return None
    if not all(3 <= np.asarray(obj["mask"]).shape[0] <= 5 and 3 <= np.asarray(obj["mask"]).shape[1] <= 5 for obj in extracted):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
