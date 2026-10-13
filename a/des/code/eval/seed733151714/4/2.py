from __future__ import annotations

import itertools
import random
from functools import lru_cache

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_signature(coords: list[tuple[int, int]]) -> tuple[str, int]:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) == 1:
        axis = "H"
        ordered = sorted(c for _r, c in coords)
        assert ordered == list(range(ordered[0], ordered[-1] + 1))
    elif len(cols) == 1:
        axis = "V"
        ordered = sorted(r for r, _c in coords)
        assert ordered == list(range(ordered[0], ordered[-1] + 1))
    else:
        raise AssertionError("every component must be a straight segment")
    length = len(coords)
    assert 2 <= length <= 5
    return axis, int(length)


def _extract_segments(grid: np.ndarray) -> dict[int, tuple[tuple[str, int], ...]]:
    top = np.asarray(grid[:-1], dtype=int)
    colors = [int(c) for c in np.unique(top) if int(c) != BG]
    segments: dict[int, tuple[tuple[str, int], ...]] = {}
    for color in colors:
        comps = _find_components(top == int(color))
        assert 2 <= len(comps) <= 4
        signatures = tuple(_segment_signature(comp) for comp in comps)
        segments[int(color)] = signatures
    return segments


def _segment_cells(
    start: tuple[int, int], axis: str, length: int, sign: int
) -> list[tuple[int, int]]:
    r0, c0 = start
    if axis == "H":
        return [(int(r0), int(c0 + sign * k)) for k in range(length)]
    return [(int(r0 + sign * k), int(c0)) for k in range(length)]


def _normalize_shape(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _is_path_shape(shape: tuple[tuple[int, int], ...]) -> bool:
    cells = set(shape)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = 0
        for dr, dc in DIRS4:
            if (int(r + dr), int(c + dc)) in cells:
                deg += 1
        degrees[(int(r), int(c))] = deg
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return False
    if any(deg not in (1, 2) for deg in degrees.values()):
        return False
    return degrees.get((0, 0), 0) == 1


@lru_cache(maxsize=None)
def _assembled_shapes(
    signature: tuple[tuple[str, int], ...]
) -> tuple[tuple[tuple[int, int], ...], ...]:
    n = len(signature)
    idxs = range(n)
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for perm in itertools.permutations(idxs):
        for signs in itertools.product((-1, 1), repeat=n):
            used: set[tuple[int, int]] = set()
            cur = (0, 0)
            ok = True
            for step, seg_idx in enumerate(perm):
                axis, length = signature[int(seg_idx)]
                cells = _segment_cells(cur, axis, int(length), int(signs[step]))
                if step == 0:
                    if any(pt in used for pt in cells):
                        ok = False
                        break
                    used.update(cells)
                else:
                    if cells[0] != cur or cur not in used:
                        ok = False
                        break
                    if any(pt in used for pt in cells[1:]):
                        ok = False
                        break
                    used.update(cells[1:])
                cur = cells[-1]
            if not ok:
                continue
            shape = _normalize_shape(used)
            if _is_path_shape(shape):
                shapes.add(shape)
    return tuple(sorted(shapes))


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    patch = np.full((h, w), BG, dtype=int)
    rr, cc = np.array(shape, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _build_output(
    shapes_by_color: dict[int, tuple[tuple[int, int], ...]],
    order: list[int],
) -> np.ndarray:
    patches = [_shape_to_patch(shapes_by_color[int(color)], int(color)) for color in order]
    out_h = sum(int(patch.shape[0]) for patch in patches) + len(patches) - 1
    out_w = max(int(patch.shape[1]) for patch in patches)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for patch in patches:
        h, w = patch.shape
        mask = patch != BG
        out[row : row + h, :w][mask] = patch[mask]
        row += int(h) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered colored segments by color, assemble each color into its unique path,
    and stack the solved paths by the bottom-row color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h >= 4

    order = [int(color) for color in g[h - 1] if int(color) != BG]
    assert 3 <= len(order) <= 5
    assert len(order) == len(set(order))

    segments_by_color = _extract_segments(g)
    assert set(order) == set(segments_by_color)

    shapes_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in order:
        shapes = _assembled_shapes(tuple(segments_by_color[int(color)]))
        assert len(shapes) == 1
        shapes_by_color[int(color)] = shapes[0]

    return _build_output(shapes_by_color, order)


def _random_path_spec() -> tuple[tuple[tuple[str, int], ...], tuple[tuple[int, int], ...]]:
    for _ in range(400):
        segment_count = random.randint(2, 4)
        used: set[tuple[int, int]] = {(0, 0)}
        current = (0, 0)
        pieces: list[tuple[str, int]] = []
        success = True
        for idx in range(segment_count):
            placed = False
            for _attempt in range(120):
                axis = random.choice(("H", "V"))
                length = random.randint(2, 5)
                if idx == 0:
                    sign = 1
                else:
                    sign = random.choice((-1, 1))
                cells = _segment_cells(current, axis, length, sign)
                if min(r for r, _c in cells) < 0 or min(c for _r, c in cells) < 0:
                    continue
                if any(pt in used for pt in cells[1:]):
                    continue
                used.update(cells[1:])
                current = cells[-1]
                pieces.append((axis, int(length)))
                placed = True
                break
            if not placed:
                success = False
                break
        if not success:
            continue
        signature = tuple(pieces)
        shape = _normalize_shape(used)
        shapes = _assembled_shapes(signature)
        if len(shapes) == 1 and shapes[0] == shape:
            return signature, shape
    raise RuntimeError("failed to sample a unique path")


def _can_place_segment(
    occupied: np.ndarray, coords: list[tuple[int, int]]
) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if bool(occupied[r, c]):
            return False
        for dr, dc in DIRS8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                return False
    return True


def _place_segments(
    specs_by_color: dict[int, tuple[tuple[str, int], ...]],
    order: list[int],
    h: int,
    w: int,
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    playable_h = h - 2

    items: list[tuple[int, str, int]] = []
    for color, signature in specs_by_color.items():
        for axis, length in signature:
            items.append((int(color), axis, int(length)))
    random.shuffle(items)
    items.sort(key=lambda item: int(item[2]), reverse=True)

    for color, axis, length in items:
        placed = False
        for _ in range(180):
            if axis == "H":
                r0 = random.randint(0, playable_h - 1)
                c0 = random.randint(0, w - length)
                coords = [(int(r0), int(c0 + k)) for k in range(length)]
            else:
                r0 = random.randint(0, playable_h - length)
                c0 = random.randint(0, w - 1)
                coords = [(int(r0 + k), int(c0)) for k in range(length)]
            if not _can_place_segment(occupied, coords):
                continue
            rr, cc = np.array(coords, dtype=int).T
            grid[rr, cc] = int(color)
            occupied[rr, cc] = True
            placed = True
            break
        if not placed:
            return None

    gaps = [random.randint(1, 3) for _ in range(len(order) - 1)]
    strip_width = len(order) + sum(gaps)
    if strip_width > w:
        return None
    start_c = random.randint(0, w - strip_width)
    c = start_c
    strip_r = h - 1
    for idx, color in enumerate(order):
        grid[strip_r, c] = int(color)
        c += 1
        if idx < len(gaps):
            c += gaps[idx]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        color_count = random.randint(3, 5)
        colors = random.sample(PALETTE, color_count)
        order = random.sample(colors, len(colors))

        specs_by_color: dict[int, tuple[tuple[str, int], ...]] = {}
        shapes_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
        try:
            for color in colors:
                signature, shape = _random_path_spec()
                specs_by_color[int(color)] = signature
                shapes_by_color[int(color)] = shape
        except RuntimeError:
            continue

        total_segments = sum(len(signature) for signature in specs_by_color.values())
        min_h = max(16, total_segments + 6)
        h = random.randint(min_h, min_h + 8)
        w = random.randint(20, 30)
        grid = _place_segments(specs_by_color, order, h, w)
        if grid is None:
            continue

        expected = _build_output(shapes_by_color, order)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
