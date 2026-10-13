import random
from typing import Iterable

import numpy as np


BG = 0
RING_SIZE_MIN = 5
RING_SIZE_MAX = 7
KEY_OFFSET_MIN = 2  # one blank cell between key and ring
KEY_OFFSET_MAX = 4
GEN_RETRIES = 400
PLACE_RETRIES = 120
DIRS = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if not (RING_SIZE_MIN <= h <= RING_SIZE_MAX and RING_SIZE_MIN <= w <= RING_SIZE_MAX):
        return None
    if coords.shape[0] != 2 * (h + w) - 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[rs - r0, cs - c0] = True
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return None
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return None
    if h > 2 and w > 2 and np.any(mask[1:-1, 1:-1]):
        return None
    return r0, c0, r1, c1


def _parse_objects(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], tuple[int, int]]]:
    objects: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        comps = _components(grid == color)
        key: tuple[int, int] | None = None
        bbox: tuple[int, int, int, int] | None = None
        for comp in comps:
            if comp.shape[0] == 1:
                assert key is None, f"color {color}: expected one key"
                key = (int(comp[0, 0]), int(comp[0, 1]))
                continue
            frame_bbox = _rect_frame_bbox(comp)
            assert frame_bbox is not None, f"color {color}: expected a rectangular ring"
            assert bbox is None, f"color {color}: expected one ring"
            bbox = frame_bbox
        assert key is not None and bbox is not None, f"color {color}: missing key or ring"
        objects.append((color, bbox, key))
    assert objects, "expected at least one ring-key pair"
    return objects


def _direction_from_key(key: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    kr, kc = key
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = kr - center_r
    dc = kc - center_c
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"

    candidates: list[tuple[int, str]] = []
    if dr <= 0:
        candidates.append((abs(kr - r0), "up"))
    if dr >= 0:
        candidates.append((abs(kr - r1), "down"))
    if dc <= 0:
        candidates.append((abs(kc - c0), "left"))
    if dc >= 0:
        candidates.append((abs(kc - c1), "right"))
    assert candidates, "key must lie outside or off-center relative to its ring"
    return min(candidates)[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each colored hollow ring on the side indicated by its same-colored key,
    fill the ring interior, and erase the key pixel.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color, bbox, key in _parse_objects(g):
        r0, c0, r1, c1 = bbox
        direction = _direction_from_key(key, bbox)

        out[key] = BG
        out[r0 + 1 : r1, c0 + 1 : c1] = color

        if direction == "up":
            out[r0, c0 : c1 + 1] = BG
        elif direction == "down":
            out[r1, c0 : c1 + 1] = BG
        elif direction == "left":
            out[r0 : r1 + 1, c0] = BG
        else:
            out[r0 : r1 + 1, c1] = BG

    return out


def _ring_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _sample_key(
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    offset = rng.randint(KEY_OFFSET_MIN, KEY_OFFSET_MAX)

    if direction == "up":
        kr = r0 - offset
        if kr < 0:
            return None
        low = max(0, c0 - 1)
        high = min(w - 1, c1 + 1)
        return kr, rng.randint(low, high)
    if direction == "down":
        kr = r1 + offset
        if kr >= h:
            return None
        low = max(0, c0 - 1)
        high = min(w - 1, c1 + 1)
        return kr, rng.randint(low, high)
    if direction == "left":
        kc = c0 - offset
        if kc < 0:
            return None
        low = max(0, r0 - 1)
        high = min(h - 1, r1 + 1)
        return rng.randint(low, high), kc

    kc = c1 + offset
    if kc >= w:
        return None
    low = max(0, r0 - 1)
    high = min(h - 1, r1 + 1)
    return rng.randint(low, high), kc


def _tie_free(key: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    return abs(key[0] - center_r) != abs(key[1] - center_c)


def _can_place(blocked: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    rs = [r for r, _ in occupied]
    cs = [c for _, c in occupied]
    r0 = max(0, min(rs) - 1)
    r1 = min(h - 1, max(rs) + 1)
    c0 = max(0, min(cs) - 1)
    c1 = min(w - 1, max(cs) + 1)
    return not np.any(blocked[r0 : r1 + 1, c0 : c1 + 1])


def _mark_blocked(blocked: np.ndarray, occupied: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    rs = [r for r, _ in occupied]
    cs = [c for _, c in occupied]
    r0 = max(0, min(rs) - 1)
    r1 = min(h - 1, max(rs) + 1)
    c0 = max(0, min(cs) - 1)
    c1 = min(w - 1, max(cs) + 1)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACE_RETRIES):
        ring_h = rng.randint(RING_SIZE_MIN, RING_SIZE_MAX)
        ring_w = rng.randint(RING_SIZE_MIN, RING_SIZE_MAX)
        r0 = rng.randint(0, h - ring_h)
        c0 = rng.randint(0, w - ring_w)
        bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
        key = _sample_key(rng, h, w, bbox, direction)
        if key is None or not _tie_free(key, bbox):
            continue

        occupied = _ring_cells(r0, c0, ring_h, ring_w)
        occupied.add(key)
        if not _can_place(blocked, occupied):
            continue

        for rr, cc in occupied:
            assert grid[rr, cc] == BG
            grid[rr, cc] = color
        _mark_blocked(blocked, occupied)
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 1])[0]
    min_dim = 12 + 3 * n_objects
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    colors = rng.sample(list(range(1, 10)), n_objects)
    directions = [rng.choice(DIRS) for _ in range(n_objects)]

    order = list(range(n_objects))
    rng.shuffle(order)
    for idx in order:
        if not _place_object(grid, blocked, colors[idx], directions[idx], rng):
            return None

    objects = _parse_objects(grid)
    for _color, bbox, key in objects:
        assert _tie_free(key, bbox)
        direction = _direction_from_key(key, bbox)
        assert direction in DIRS

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
