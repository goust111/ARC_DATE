from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
MAROON = 9

OUTPUT_COLOR = {
    "horizontal": RED,
    "vertical": BLUE,
    "central": GREEN,
}

MAIN_COLORS = (4, 5, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_MIN = 21
GRID_SIDE_MAX = 31
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
GAP = 1
GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 250


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    components = _find_components(mask.astype(bool))
    return len(components) == 1


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    central = np.array_equal(mask, np.rot90(mask, 2))
    return horizontal, vertical, central


def _classify_symmetry(mask: np.ndarray) -> str:
    horizontal, vertical, central = _symmetry_flags(mask)
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    if central:
        return "central"
    raise AssertionError("component does not match any supported symmetry")


def _is_exact_class(mask: np.ndarray, kind: str) -> bool:
    horizontal, vertical, central = _symmetry_flags(mask)
    if kind == "horizontal":
        return horizontal and not vertical
    if kind == "vertical":
        return vertical and not horizontal
    if kind == "central":
        return central and not horizontal and not vertical
    raise ValueError(kind)


def _transform_point(kind: str, r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if kind == "horizontal":
        return h - 1 - r, c
    if kind == "vertical":
        return r, w - 1 - c
    if kind == "central":
        return h - 1 - r, w - 1 - c
    raise ValueError(kind)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each maroon spot to its symmetric partner and recolor by symmetry type."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _find_components(g != BLACK)
    assert components

    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        assert MAROON in colors
        main_colors = colors - {MAROON}
        assert len(main_colors) == 1
        main_color = next(iter(main_colors))

        spot_coords = [(r, c) for r, c in coords if int(g[r, c]) == MAROON]
        assert len(spot_coords) == 1
        spot_r, spot_c = spot_coords[0]

        r0, c0, r1, c1 = _bbox(coords)
        rel_coords = [(r - r0, c - c0) for r, c in coords]
        mask = _mask_from_coords(rel_coords, (r1 - r0 + 1, c1 - c0 + 1))
        kind = _classify_symmetry(mask)

        lr, lc = spot_r - r0, spot_c - c0
        tr, tc = _transform_point(kind, lr, lc, mask.shape[0], mask.shape[1])
        assert mask[tr, tc]

        out[spot_r, spot_c] = main_color
        out[r0 + tr, c0 + tc] = OUTPUT_COLOR[kind]

    return out


def _orbit(kind: str, r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    cells = {(r, c), _transform_point(kind, r, c, h, w)}
    return tuple(sorted(cells))


def _unique_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orb = _orbit(kind, r, c, h, w)
            if orb not in seen:
                seen.add(orb)
                orbits.append(orb)
    return orbits


def _seed_orbit(kind: str, h: int, w: int) -> tuple[tuple[int, int], ...]:
    if kind == "horizontal":
        row = h // 2 if h % 2 == 1 else h // 2 - 1
        return _orbit(kind, row, random.randrange(w), h, w)
    if kind == "vertical":
        col = w // 2 if w % 2 == 1 else w // 2 - 1
        return _orbit(kind, random.randrange(h), col, h, w)
    if kind == "central":
        return ((h // 2, w // 2),)
    raise ValueError(kind)


def _touches_occupied(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    for r, c in orbit:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _spot_pairs(mask: np.ndarray, kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        tr, tc = _transform_point(kind, int(r), int(c), h, w)
        if (int(r), int(c)) == (tr, tc) or not mask[tr, tc]:
            continue
        pair = tuple(sorted(((int(r), int(c)), (tr, tc))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _sample_dims(kind: str) -> tuple[int, int]:
    if kind == "central":
        return random.choice((5, 7, 9)), random.choice((5, 7, 9))
    return random.randint(4, 9), random.randint(4, 9)


def _generate_mask(kind: str) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h, w = _sample_dims(kind)
        occupied = set(_seed_orbit(kind, h, w))
        min_area = 6
        area_cap = random.randint(max(min_area + 1, len(occupied) + 2), max(min_area + 2, int(0.68 * h * w)))
        orbits = _unique_orbits(kind, h, w)

        while True:
            candidates: list[tuple[tuple[int, int], ...]] = []
            for orbit in orbits:
                if all(cell in occupied for cell in orbit):
                    continue
                if any(cell in occupied for cell in orbit):
                    candidates = []
                    break
                if len(occupied) + len(orbit) > area_cap:
                    continue
                if _touches_occupied(orbit, occupied):
                    candidates.append(orbit)
            if not candidates:
                break
            if len(occupied) >= min_area and random.random() < 0.35:
                break
            occupied.update(random.choice(candidates))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in occupied:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if min(mask.shape) < 3:
            continue
        if not _is_connected(mask):
            continue
        if not _is_exact_class(mask, kind):
            continue
        if not (6 <= int(mask.sum()) <= int(0.75 * mask.size)):
            continue
        if not _spot_pairs(mask, kind):
            continue
        return mask

    raise RuntimeError(f"failed to build a {kind} mask")


def _build_object(kind: str, color: int) -> dict:
    mask = _generate_mask(kind)
    pairs = _spot_pairs(mask, kind)
    src, dst = random.choice(pairs)
    if random.random() < 0.5:
        src, dst = dst, src

    h, w = mask.shape
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    inp[mask] = color
    out[mask] = color
    inp[src] = MAROON
    out[dst] = OUTPUT_COLOR[kind]

    assert _classify_symmetry(mask) == kind
    return {
        "kind": kind,
        "input": inp,
        "output": out,
        "shape": (h, w),
    }


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_objects(
    h: int, w: int, objects: list[dict]
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["shape"][0] * objects[idx]["shape"][1],
        reverse=True,
    )
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    boxes: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        oh, ow = objects[idx]["shape"]
        candidates = [(r, c) for r in range(h - oh + 1) for c in range(w - ow + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates[:PLACEMENT_RETRIES]:
            box = (r, c, r + oh - 1, c + ow - 1)
            if all(_separated(box, other) for other in boxes.values()):
                positions[idx] = (r, c)
                boxes[idx] = box
                placed = True
                break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _compose_grid(shape: tuple[int, int], objects: list[dict], positions: list[tuple[int, int]], key: str) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for obj, (r0, c0) in zip(objects, positions):
        patch = obj[key]
        ph, pw = patch.shape
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert not np.any((sub != BLACK) & (patch != BLACK))
        sub[patch != BLACK] = patch[patch != BLACK]
    return grid


def _branch_types() -> list[str]:
    count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    kinds = ["horizontal", "vertical", "central"]
    while len(kinds) < count:
        kinds.append(random.choice(kinds))
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    kinds = _branch_types()
    colors = random.sample(MAIN_COLORS, len(kinds))
    objects = [_build_object(kind, color) for kind, color in zip(kinds, colors)]

    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    positions = _place_objects(h, w, objects)
    if positions is None:
        return None

    inp = _compose_grid((h, w), objects, positions, "input")
    expected = _compose_grid((h, w), objects, positions, "output")

    if np.count_nonzero(inp != BLACK) * 2 >= h * w:
        return None

    components = _find_components(inp != BLACK)
    if len(components) != len(objects):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
