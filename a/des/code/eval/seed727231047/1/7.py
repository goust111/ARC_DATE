from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _boundary_cell_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if (
                r == 0
                or not mask[r - 1, c]
                or r == h - 1
                or not mask[r + 1, c]
                or c == 0
                or not mask[r, c - 1]
                or c == w - 1
                or not mask[r, c + 1]
            ):
                count += 1
    return count


def _has_hole(mask: np.ndarray) -> bool:
    bg = np.pad(~mask.astype(bool), 1, constant_values=True)
    return len(_connected_components(bg)) > 1


def _has_opposite_exposed_sides(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            top = r == 0 or not mask[r - 1, c]
            bottom = r == h - 1 or not mask[r + 1, c]
            left = c == 0 or not mask[r, c - 1]
            right = c == w - 1 or not mask[r, c + 1]
            if (top and bottom) or (left and right):
                return True
    return False


def _trace_boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trace the outer contour clockwise and return boundary pixels in order."""
    m = np.pad(mask.astype(bool), 1, constant_values=False)
    edges: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for r, c in np.argwhere(m):
        if not m[r - 1, c]:
            edges.append(((r, c), (r, c + 1), (r, c)))
        if not m[r, c + 1]:
            edges.append(((r, c + 1), (r + 1, c + 1), (r, c)))
        if not m[r + 1, c]:
            edges.append(((r + 1, c + 1), (r + 1, c), (r, c)))
        if not m[r, c - 1]:
            edges.append(((r + 1, c), (r, c), (r, c)))

    assert edges, "object must have a boundary"

    edge_map: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for start, end, owner in edges:
        if start in edge_map:
            assert edge_map[start][0] == end, "boundary start vertex must be unique"
        edge_map[start] = (end, owner)

    start = min(edge_map)
    owners: list[tuple[int, int]] = []
    current = start
    for _ in range(len(edges) + 1):
        nxt, owner = edge_map[current]
        owners.append(owner)
        current = nxt
        if current == start:
            break
    else:
        raise AssertionError("boundary trace did not close")

    cells: list[tuple[int, int]] = []
    for owner in owners:
        unpadded = (owner[0] - 1, owner[1] - 1)
        if not cells or cells[-1] != unpadded:
            cells.append(unpadded)
    if cells and cells[0] == cells[-1]:
        cells.pop()

    if len(cells) != len(set(cells)):
        uniq: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for cell in cells:
            if cell not in seen:
                uniq.append(cell)
                seen.add(cell)
        cells = uniq

    return cells


def _rotate_to_start(seq: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    idx = seq.index(start)
    return seq[idx:] + seq[:idx]


def main(grid: np.ndarray) -> np.ndarray:
    """Color half of each object's contour clockwise starting from its seed pixel."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _connected_components(g != 0):
        values = g[coords[:, 0], coords[:, 1]]
        counts = Counter(map(int, values))
        body_color = max(counts.items(), key=lambda kv: kv[1])[0]
        seed_colors = [color for color in counts if color != body_color]
        assert len(seed_colors) == 1, "each object must contain exactly one seed color"
        seed_color = int(seed_colors[0])

        seed_cells = coords[values == seed_color]
        assert seed_cells.shape[0] == 1, "seed must be unique"
        seed_r, seed_c = map(int, seed_cells[0])

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True

        contour = _trace_boundary_cells(local)
        seed_local = (seed_r - r0, seed_c - c0)
        assert seed_local in contour, "seed must lie on the contour"
        contour = _rotate_to_start(contour, seed_local)

        steps = len(contour) // 2
        for lr, lc in contour[:steps]:
            out[r0 + lr, c0 + lc] = seed_color

    return out


# --------------- generator (seed727231047-7) ---------------

_RNG = np.random.default_rng()

BG = 0
BODY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

OBJECT_COUNT_RANGE = (2, 6)
GRID_SIZE_MIN = 20
GRID_SIZE_MAX_EXCLUSIVE = 32
PLACEMENT_ATTEMPTS = 160
SHAPE_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    if _RNG.random() < 0.5:
        out = np.fliplr(out)
    k = int(_RNG.integers(0, 4))
    return np.rot90(out, k).copy()


def _shape_rectangle() -> np.ndarray:
    h = int(_RNG.integers(4, 9))
    w = int(_RNG.integers(4, 9))
    return np.ones((h, w), dtype=bool)


def _shape_l() -> np.ndarray:
    h = int(_RNG.integers(6, 11))
    w = int(_RNG.integers(6, 11))
    tmax = min(4, h - 2, w - 2)
    if tmax < 2:
        return _shape_rectangle()
    t = int(_RNG.integers(2, tmax + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :t] = True
    mask[h - t :, :] = True
    return _random_orientation(mask)


def _shape_t() -> np.ndarray:
    h = int(_RNG.integers(7, 12))
    w = int(_RNG.integers(7, 12))
    cap_h = int(_RNG.integers(2, min(4, h - 2) + 1))
    stem_w = int(_RNG.integers(2, min(4, w - 2) + 1))
    max_left = w - stem_w - 1
    stem_left = 1 if max_left <= 1 else int(_RNG.integers(1, max_left + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:cap_h, :] = True
    mask[:, stem_left : stem_left + stem_w] = True
    return _random_orientation(mask)


def _shape_u() -> np.ndarray:
    h = int(_RNG.integers(7, 12))
    w = int(_RNG.integers(7, 12))
    arm_max = min(3, (w - 2) // 2, h - 2)
    if arm_max < 2:
        return _shape_rectangle()
    arm_w = int(_RNG.integers(2, arm_max + 1))
    base_h = int(_RNG.integers(2, min(4, h - 2) + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :arm_w] = True
    mask[:, w - arm_w :] = True
    mask[h - base_h :, :] = True
    return _random_orientation(mask)


def _shape_plus() -> np.ndarray:
    h = int(_RNG.integers(7, 12))
    w = int(_RNG.integers(7, 12))
    band_h = int(_RNG.integers(2, min(4, h - 2) + 1))
    band_w = int(_RNG.integers(2, min(4, w - 2) + 1))
    row0 = int(_RNG.integers(1, h - band_h))
    col0 = int(_RNG.integers(1, w - band_w))
    mask = np.zeros((h, w), dtype=bool)
    mask[row0 : row0 + band_h, :] = True
    mask[:, col0 : col0 + band_w] = True
    return _random_orientation(mask)


_SHAPE_FACTORIES = [_shape_rectangle, _shape_l, _shape_t, _shape_u, _shape_plus]


def _is_valid_shape(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    if len(_connected_components(mask)) != 1:
        return False
    if _has_hole(mask):
        return False
    if _has_opposite_exposed_sides(mask):
        return False
    try:
        contour = _trace_boundary_cells(mask)
    except AssertionError:
        return False
    return (
        len(contour) >= 8
        and len(contour) == len(set(contour))
        and len(contour) == _boundary_cell_count(mask)
    )


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        factory = _SHAPE_FACTORIES[int(_RNG.integers(len(_SHAPE_FACTORIES)))]
        mask = factory()
        if _is_valid_shape(mask):
            return mask
    raise RuntimeError("failed to generate a valid shape")


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + mh + 1)
    c1 = min(w, left + mw + 1)
    return not occupied[r0:r1, c0:c1].any()


def _gen_attempt() -> np.ndarray | None:
    object_count = int(_RNG.integers(*OBJECT_COUNT_RANGE))
    h = int(_RNG.integers(max(GRID_SIZE_MIN, 16 + 2 * object_count), GRID_SIZE_MAX_EXCLUSIVE))
    w = int(_RNG.integers(max(GRID_SIZE_MIN, 16 + 2 * object_count), GRID_SIZE_MAX_EXCLUSIVE))
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = list(_RNG.choice(SEED_COLORS, size=object_count, replace=False))

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            mask = _random_shape()
            mh, mw = mask.shape
            if mh + 2 > h or mw + 2 > w:
                continue

            top = int(_RNG.integers(1, h - mh))
            left = int(_RNG.integers(1, w - mw))
            if not _can_place(occupied, mask, top, left):
                continue

            contour = _trace_boundary_cells(mask)
            seed_r, seed_c = contour[int(_RNG.integers(len(contour)))]

            view = grid[top : top + mh, left : left + mw]
            view[mask] = BODY
            view[seed_r, seed_c] = int(color)

            occ_view = occupied[top : top + mh, left : left + mw]
            occ_view[mask] = True
            placed = True
            break
        if not placed:
            return None

    comps = _connected_components(grid != BG)
    if len(comps) != object_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid ARC input")
