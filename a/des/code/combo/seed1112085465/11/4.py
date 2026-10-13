from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 7       # orange
FG = 5       # grey
BLUE = 1
RED = 2
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        cells: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            cells.append((cr, cc2))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))

    return comps


def _bbox_mask(coords: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return r0, c0, r1, c1, mask


def _has_horizontal_symmetry(mask: np.ndarray) -> bool:
    return mask.shape[0] % 2 == 1 and np.array_equal(mask, np.flipud(mask))


def _has_vertical_symmetry(mask: np.ndarray) -> bool:
    return mask.shape[1] % 2 == 1 and np.array_equal(mask, np.fliplr(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only symmetric grey objects and draw their reflection axes.

    Horizontal symmetry paints the center row of the object's bounding box blue.
    Vertical symmetry paints the center column red, after blue.
    Asymmetric objects are erased to the orange background.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("expected a non-empty 2D grid")

    out = np.full(g.shape, BG, dtype=int)
    for coords in _components(g == FG):
        r0, c0, r1, c1, mask = _bbox_mask(coords)
        has_h = _has_horizontal_symmetry(mask)
        has_v = _has_vertical_symmetry(mask)
        if not (has_h or has_v):
            continue

        out[coords[:, 0], coords[:, 1]] = FG
        if has_h:
            out[r0 + mask.shape[0] // 2, c0 : c1 + 1] = BLUE
        if has_v:
            out[r0 : r1 + 1, c0 + mask.shape[1] // 2] = RED

    return out


def _orbit_cells(r: int, c: int, h: int, w: int, sym_h: bool, sym_v: bool) -> tuple[tuple[int, int], ...]:
    cells = {(r, c)}
    if sym_h:
        cells.add((h - 1 - r, c))
    if sym_v:
        cells.add((r, w - 1 - c))
    if sym_h and sym_v:
        cells.add((h - 1 - r, w - 1 - c))
    return tuple(sorted(cells))


def _build_orbits(h: int, w: int, sym_h: bool, sym_v: bool) -> list[tuple[tuple[int, int], ...]]:
    rows = range((h + 1) // 2) if sym_h else range(h)
    cols = range((w + 1) // 2) if sym_v else range(w)
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in rows:
        for c in cols:
            orbit = _orbit_cells(r, c, h, w, sym_h, sym_v)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _orbit_adjacency(orbits: list[tuple[tuple[int, int], ...]]) -> list[set[int]]:
    neighbors = [set() for _ in orbits]
    for i, orbit_a in enumerate(orbits):
        for j in range(i + 1, len(orbits)):
            orbit_b = orbits[j]
            touching = False
            for r0, c0 in orbit_a:
                for r1, c1 in orbit_b:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        touching = True
                        break
                if touching:
                    break
            if touching:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _kind_of(mask: np.ndarray) -> str:
    has_h = _has_horizontal_symmetry(mask)
    has_v = _has_vertical_symmetry(mask)
    if has_h and has_v:
        return "HV"
    if has_h:
        return "H"
    if has_v:
        return "V"
    return "A"


def _connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _grow_mask(
    h: int,
    w: int,
    sym_h: bool,
    sym_v: bool,
    min_ratio: float,
    max_ratio: float,
    rng: random.Random,
) -> np.ndarray | None:
    orbits = _build_orbits(h, w, sym_h, sym_v)
    neighbors = _orbit_adjacency(orbits)
    if not orbits:
        return None

    if sym_h and sym_v:
        start_candidates = [i for i, orbit in enumerate(orbits) if (h // 2, w // 2) in orbit]
    elif sym_h:
        start_candidates = [i for i, orbit in enumerate(orbits) if any(r == h // 2 for r, _ in orbit)]
    elif sym_v:
        start_candidates = [i for i, orbit in enumerate(orbits) if any(c == w // 2 for _, c in orbit)]
    else:
        start_candidates = list(range(len(orbits)))
    if not start_candidates:
        return None

    min_area = max(5, int(np.ceil(h * w * min_ratio)))
    max_area = max(min_area, int(np.floor(h * w * max_ratio)))

    for _ in range(40):
        selected = {rng.choice(start_candidates)}
        area = sum(len(orbits[i]) for i in selected)
        frontier = set(neighbors[next(iter(selected))]) - selected

        while frontier:
            candidates = [i for i in frontier if area + len(orbits[i]) <= max_area]
            if not candidates:
                break
            if area >= min_area and rng.random() < 0.35:
                break
            nxt = rng.choice(candidates)
            selected.add(nxt)
            area += len(orbits[nxt])
            frontier |= neighbors[nxt]
            frontier -= selected

        if area < min_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for idx in selected:
            for r, c in orbits[idx]:
                mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if mask.sum() < 5 or not _connected(mask):
            continue
        return mask

    return None


def _generate_object_mask(kind: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        if kind == "H":
            h = rng.randrange(3, 10, 2)
            w = rng.randint(3, 8)
            mask = _grow_mask(h, w, True, False, 0.30, 0.68, rng)
        elif kind == "V":
            h = rng.randint(3, 8)
            w = rng.randrange(3, 10, 2)
            mask = _grow_mask(h, w, False, True, 0.30, 0.68, rng)
        elif kind == "HV":
            h = rng.randrange(3, 10, 2)
            w = rng.randrange(3, 10, 2)
            mask = _grow_mask(h, w, True, True, 0.22, 0.60, rng)
        else:
            h = rng.randint(3, 9)
            w = rng.randint(3, 9)
            mask = _grow_mask(h, w, False, False, 0.28, 0.62, rng)

        if mask is None:
            continue
        if _kind_of(mask) != kind:
            continue
        return mask

    return None


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return not blocked[top : top + h, left : left + w].any()


def _place_objects(masks: list[np.ndarray], rng: random.Random, grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        if h > grid_h or w > grid_w:
            return None

        placed = False
        for _ in range(160):
            top = rng.randint(0, grid_h - h)
            left = rng.randint(0, grid_w - w)
            if not _can_place(blocked, top, left, h, w):
                continue

            grid[top : top + h, left : left + w][mask] = FG
            r0 = max(0, top - 1)
            r1 = min(grid_h, top + h + 1)
            c0 = max(0, left - 1)
            c1 = min(grid_w, left + w + 1)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(18, 26)
    grid_w = rng.randint(18, 26)
    kinds = ["H", "V", "HV", "A"]
    if rng.random() < 0.6:
        kinds.append(rng.choice(kinds))

    masks: list[np.ndarray] = []
    for kind in kinds:
        mask = _generate_object_mask(kind, rng)
        if mask is None:
            return None
        masks.append(mask)

    grid = _place_objects(masks, rng, grid_h, grid_w)
    if grid is None:
        return None

    comps = _components(grid == FG)
    if len(comps) != len(masks):
        return None

    found_kinds = [_kind_of(_bbox_mask(coords)[4]) for coords in comps]
    if not {"H", "V", "HV", "A"}.issubset(set(found_kinds)):
        return None

    out = main(grid)
    if np.count_nonzero(out == RED) == 0 or np.count_nonzero(out == BLUE) == 0:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
