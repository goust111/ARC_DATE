from __future__ import annotations

import random

import numpy as np


BG = 0
IN_COLOR = 5
ONLY_ORIG = 1
ONLY_SHIFT = 2
OVERLAP = 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(np.array(sorted(cells), dtype=int))

    components.sort(key=lambda coords: _bbox_coords(coords))
    return components


def _shifted_positions(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shifted: set[tuple[int, int]] = set()
    for r, c in coords:
        has_support = any(rr > r and cc > c for rr, cc in coords)
        shifted.add((r, c + 1) if has_support else (r, c))
    return shifted


def _overlay_sets_from_coords(coords: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    orig = {tuple(map(int, rc)) for rc in coords}
    shifted = _shifted_positions(orig)
    return orig - shifted, shifted - orig, orig & shifted


def _paint(out: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay each gray object with its one-step eastward support slide.

    Input structure:
    - The grid is black background with 2 or 3 gray 4-connected objects.
    - Each object is irregular and isolated from the others.
    - Within an object, some pixels have a strictly southeast partner and some do not.

    Rule:
    - For each gray object, keep the original pixel set and also build a shifted copy.
    - A pixel shifts one column right iff some other pixel of the same object lies strictly southeast of it.
    - Paint blue where only the original copy exists, red where only the shifted copy exists, and yellow where the two copies overlap.

    Key insight:
    - The output is the union of two aligned versions of the same object, where southeast-supported pixels move and the frontier stays fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BG, dtype=int)
    for coords in _find_components(g, IN_COLOR):
        only_orig, only_shift, overlap = _overlay_sets_from_coords(coords)
        _paint(out, only_orig, ONLY_ORIG)
        _paint(out, only_shift, ONLY_SHIFT)
        _paint(out, overlap, OVERLAP)
    return out


_BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2)),
)


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


_BASE_MASKS = tuple(_coords_to_mask(shape) for shape in _BASE_SHAPES)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=int, copy=True)
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _sample_shape() -> np.ndarray:
    for _ in range(64):
        mask = np.array(random.choice(_BASE_MASKS), copy=True)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        coords = _normalize_coords(np.argwhere(mask))
        only_orig, only_shift, overlap = _overlay_sets_from_coords(coords)
        r0, c0, r1, c1 = _bbox_coords(coords)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(only_orig) and len(only_shift) and len(overlap) and int(coords.shape[0]) < bbox_area:
            return coords
    raise RuntimeError("failed to sample a valid object shape")


def _translate(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _separated(candidate: set[tuple[int, int]], existing: set[tuple[int, int]], gap: int) -> bool:
    for r, c in candidate:
        for rr, cc in existing:
            if abs(int(r) - int(rr)) <= gap and abs(int(c) - int(cc)) <= gap:
                return False
    return True


def _build_object() -> dict[str, object]:
    coords = _sample_shape()
    orig = {tuple(map(int, rc)) for rc in coords}
    only_orig, only_shift, overlap = _overlay_sets_from_coords(coords)
    union = orig | only_shift
    height = max(r for r, _c in union) + 1
    width = max(c for _r, c in union) + 1
    return {
        "orig": orig,
        "only_orig": only_orig,
        "only_shift": only_shift,
        "overlap": overlap,
        "union": union,
        "height": int(height),
        "width": int(width),
    }


def _place_objects(height: int, width: int, objects: list[dict[str, object]]) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    used_input: set[tuple[int, int]] = set()
    used_union: set[tuple[int, int]] = set()

    for obj in sorted(objects, key=lambda item: (-int(item["width"]), -int(item["height"]))):
        obj_h = int(obj["height"])
        obj_w = int(obj["width"])
        candidates = [(r, c) for r in range(height - obj_h + 1) for c in range(width - obj_w + 1)]
        random.shuffle(candidates)

        for top, left in candidates:
            orig = _translate(obj["orig"], top, left)
            union = _translate(obj["union"], top, left)
            if not _separated(orig, used_input, gap=1):
                continue
            if not _separated(union, used_union, gap=1):
                continue

            placed.append(
                {
                    "orig": orig,
                    "only_orig": _translate(obj["only_orig"], top, left),
                    "only_shift": _translate(obj["only_shift"], top, left),
                    "overlap": _translate(obj["overlap"], top, left),
                }
            )
            used_input |= orig
            used_union |= union
            break
        else:
            return None

    return placed


def _render_input(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in placed:
        _paint(grid, obj["orig"], IN_COLOR)
    return grid


def _render_output(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for obj in placed:
        _paint(out, obj["only_orig"], ONLY_ORIG)
        _paint(out, obj["only_shift"], ONLY_SHIFT)
        _paint(out, obj["overlap"], OVERLAP)
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice((2, 3))
    height = random.randint(10 + (object_count == 3), 15)
    width = random.randint(10 + object_count, 16)

    objects = [_build_object() for _ in range(object_count)]
    placed = _place_objects(height, width, objects)
    if placed is None:
        return None

    inp = _render_input(height, width, placed)
    expected = _render_output(height, width, placed)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    colors = set(map(int, np.unique(actual)))
    if not {ONLY_ORIG, ONLY_SHIFT, OVERLAP}.issubset(colors):
        return None
    if len(_find_components(inp, IN_COLOR)) != object_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
