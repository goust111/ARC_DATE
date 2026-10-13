from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
COLORS = np.arange(1, 10, dtype=int)
MAX_RETRIES = 400


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _bottom_connected_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    fg = grid != bg
    h, w = grid.shape
    debris = np.zeros_like(fg, dtype=bool)
    q = deque()
    for c in range(w):
        if fg[h - 1, c]:
            debris[h - 1, c] = True
            q.append((h - 1, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(grid.shape, r, c):
            if fg[nr, nc] and not debris[nr, nc]:
                debris[nr, nc] = True
                q.append((nr, nc))
    return debris


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a floating 180-degree-symmetric object by returning bottom debris to its missing cells.

    Input structure:
    - Black background contains one floating multicolor object whose silhouette is missing some cells.
    - Every missing cell belongs to a 180-degree rotationally symmetric silhouette, while its rotated partner cell still exists.
    - Bottom-edge debris piles store the missing colors column by column as tight vertical stacks disconnected from the object.

    Rule:
    - Split the bottom-connected debris from the floating object.
    - Inside the floating object's bounding box, recover the missing silhouette cells by taking the union of the visible cells and their 180-degree rotated partners.
    - For each column, read the debris pile from top to bottom and fill that column's missing cells from top to bottom, then clear the debris piles.

    Key insight:
    - Rotational symmetry recovers where the holes are, but only the bottom piles preserve the missing colors and their within-column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(np.bincount(g.ravel()).argmax())
    debris = _bottom_connected_foreground(g, bg)
    visible_object = (g != bg) & (~debris)
    assert visible_object.any(), "expected a floating object above the debris"

    r0, c0, r1, c1 = _bbox(visible_object)
    obj = visible_object[r0 : r1 + 1, c0 : c1 + 1]
    completed = obj | np.flipud(np.fliplr(obj))
    missing = completed & (~obj)

    out = g.copy()
    out[debris] = bg

    for local_c in range(obj.shape[1]):
        global_c = c0 + local_c
        missing_rows = np.flatnonzero(missing[:, local_c])
        pile_rows = np.flatnonzero(debris[:, global_c])
        assert len(missing_rows) == len(pile_rows), "each pile column must match the number of missing cells"
        if len(missing_rows) == 0:
            continue
        colors = [int(g[row, global_c]) for row in pile_rows.tolist()]
        for local_r, color in zip(missing_rows.tolist(), colors, strict=True):
            out[r0 + local_r, global_c] = color

    return out


def _orbit_cells(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    mate = (h - 1 - r, w - 1 - c)
    if mate == (r, c):
        return ((r, c),)
    return tuple(sorted(((r, c), mate)))


def _center_kernel(h: int, w: int) -> list[tuple[int, int]]:
    rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
    cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
    return [(r, c) for r in rows for c in cols]


def _grow_symmetric_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in _center_kernel(h, w):
        mask[r, c] = True

    while int(mask.sum()) < target:
        candidates: dict[tuple[tuple[int, int], ...], float] = {}
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors4(mask.shape, r, c):
                orbit = _orbit_cells(h, w, nr, nc)
                if all(mask[rr, cc] for rr, cc in orbit):
                    continue
                dist = sum(abs(rr - (h - 1) / 2) + abs(cc - (w - 1) / 2) for rr, cc in orbit) / len(orbit)
                candidates[orbit] = 1.0 / (1.0 + dist)
        if not candidates:
            break
        orbits = list(candidates)
        weights = np.array([candidates[orbit] for orbit in orbits], dtype=float)
        weights /= weights.sum()
        chosen = orbits[int(rng.choice(len(orbits), p=weights))]
        for rr, cc in chosen:
            mask[rr, cc] = True
    return mask


def _assign_colors(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    colored = np.zeros(mask.shape, dtype=int)
    palette_size = int(rng.integers(3, 7))
    palette = rng.choice(COLORS, size=palette_size, replace=False)
    coords = np.argwhere(mask)
    samples = rng.choice(palette, size=len(coords), replace=True)
    for (r, c), color in zip(coords.tolist(), samples.tolist(), strict=True):
        colored[int(r), int(c)] = int(color)

    used = {int(colored[r, c]) for r, c in coords.tolist()}
    if len(used) == 1:
        r, c = map(int, coords[0])
        choices = [int(v) for v in palette.tolist() if int(v) != colored[r, c]]
        colored[r, c] = choices[0]

    symmetric_same = True
    h, w = mask.shape
    for r, c in coords.tolist():
        rr, cc = h - 1 - int(r), w - 1 - int(c)
        if mask[rr, cc] and colored[int(r), int(c)] != colored[rr, cc]:
            symmetric_same = False
            break
    if symmetric_same:
        for r, c in coords.tolist():
            rr, cc = h - 1 - int(r), w - 1 - int(c)
            if (rr, cc) != (int(r), int(c)) and mask[rr, cc]:
                choices = [int(v) for v in palette.tolist() if int(v) != colored[rr, cc]]
                if choices:
                    colored[int(r), int(c)] = choices[0]
                    break
    return colored


def _candidate_orbits(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    out = []
    seen = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        orbit = _orbit_cells(h, w, r, c)
        if len(orbit) != 2 or orbit in seen:
            continue
        seen.add(orbit)
        (r1, c1), (r2, c2) = orbit
        if min(r1, r2, c1, c2) == 0 or max(r1, r2) == h - 1 or max(c1, c2) == w - 1:
            continue
        out.append((orbit[0], orbit[1]))
    return out


def _choose_missing_cells(rng: np.random.Generator, mask: np.ndarray) -> list[tuple[int, int]] | None:
    orbits = _candidate_orbits(mask)
    if len(orbits) < 2:
        return None

    original_bbox = _bbox(mask)
    want_tall_pile = rng.random() < 0.65
    max_missing = min(8, len(orbits))

    for _ in range(250):
        count = int(rng.integers(2, max_missing + 1))
        chosen_ids = rng.choice(len(orbits), size=count, replace=False)
        missing = []
        visible = mask.copy()
        for idx in chosen_ids.tolist():
            a, b = orbits[int(idx)]
            cell = a if rng.random() < 0.5 else b
            missing.append(cell)
            visible[cell] = False

        if not visible.any() or not _mask_connected(visible):
            continue
        if _bbox(visible) != original_bbox:
            continue

        col_counts = Counter(c for _r, c in missing)
        if max(col_counts.values()) > 6:
            continue
        if want_tall_pile and max(col_counts.values()) < 2:
            continue
        return sorted(missing)
    return None


def _compose_grids(
    rng: np.random.Generator,
    full_object: np.ndarray,
    missing: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = full_object.shape
    col_counts = Counter(c for _r, c in missing)
    max_pile = max(col_counts.values())

    top = int(rng.integers(1, 5))
    gap = int(rng.integers(2, 6))
    height = top + h + gap + max_pile
    if height > 30:
        return None

    extra_w = int(rng.integers(2, min(30 - w, 8) + 1))
    width = w + extra_w
    left = int(rng.integers(1, extra_w))
    if width > 30 or left + w >= width:
        return None

    out = np.zeros((height, width), dtype=int)
    out[top : top + h, left : left + w] = full_object
    inp = out.copy()

    by_col: dict[int, list[int]] = {}
    for r, c in missing:
        inp[top + r, left + c] = BG
        by_col.setdefault(c, []).append(r)

    for c, rows in by_col.items():
        rows = sorted(rows)
        colors = [int(full_object[r, c]) for r in rows]
        start = height - len(colors)
        for i, color in enumerate(colors):
            inp[start + i, left + c] = color

    return inp, out


def _generate_complete_object(rng: np.random.Generator) -> np.ndarray | None:
    base_h = int(rng.integers(8, 15))
    base_w = int(rng.integers(8, 15))
    min_target = max(len(_center_kernel(base_h, base_w)) + 10, (base_h * base_w) // 4)
    max_target = min((base_h * base_w * 3) // 5, 44)
    if min_target > max_target:
        return None

    mask = _grow_symmetric_mask(rng, base_h, base_w, int(rng.integers(min_target, max_target + 1)))
    mask = _crop_to_bbox(mask)
    if min(mask.shape) < 5 or int(mask.sum()) < 16:
        return None
    if not _mask_connected(mask):
        return None
    return _assign_colors(rng, mask)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    full_object = _generate_complete_object(rng)
    if full_object is None:
        return None

    mask = full_object != BG
    missing = _choose_missing_cells(rng, mask)
    if missing is None:
        return None

    grids = _compose_grids(rng, full_object, missing)
    if grids is None:
        return None
    inp, out = grids

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
