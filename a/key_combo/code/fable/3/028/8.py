from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]

ARC_COLORS = tuple(range(10))
GEN_RETRIES = 500
OBJECT_PADDING = 2


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords) -> Shape:
    arr = np.asarray(list(coords), dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_to_mask(shape: Shape) -> np.ndarray:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_to_shape(mask: np.ndarray) -> Shape:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _normalize(coords)


def _connected_cells(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q: deque[Coord] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _embedding_offsets(prototype: Shape, fragment: Shape) -> list[Coord]:
    pset = set(prototype)
    ph, pw = _shape_size(prototype)
    fh, fw = _shape_size(fragment)
    offsets: list[Coord] = []
    for dr in range(ph - fh + 1):
        for dc in range(pw - fw + 1):
            if all((r + dr, c + dc) in pset for r, c in fragment):
                offsets.append((int(dr), int(dc)))
    return offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each uniquely embedded fragment into the largest prototype shape.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of separated monochromatic 4-connected components.
    - Exactly one component has the largest area; this component is the complete prototype.
    - Every other component is a connected true subset of that prototype shape with one translation-only embedding.

    Rule:
    - Keep the prototype unchanged.
    - For each smaller component, normalize its shape and find the only offset where it fits inside the normalized prototype.
    - Use that offset to recover the full prototype's top-left position at the fragment location.
    - Paint the missing prototype pixels there with the fragment's own color while preserving its original pixels.

    Key insight:
    - The fragment's visible bounding box is not the anchor; the anchor is determined by its unique subshape alignment inside the largest component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    comps = _components(g, bg)
    assert len(comps) >= 2, "expected a prototype and at least one fragment"

    max_area = max(comp["area"] for comp in comps)
    largest = [i for i, comp in enumerate(comps) if comp["area"] == max_area]
    assert len(largest) == 1, "expected a unique largest component"
    proto_i = largest[0]
    proto = comps[proto_i]
    prototype_shape = _normalize(map(tuple, proto["coords"]))
    prototype_area = len(prototype_shape)

    out = g.copy()
    for i, comp in enumerate(comps):
        if i == proto_i:
            continue

        color = int(comp["color"])
        coords = comp["coords"]
        fragment_shape = _normalize(map(tuple, coords))
        assert 0 < len(fragment_shape) < prototype_area, "fragment must be a true prototype subset"

        offsets = _embedding_offsets(prototype_shape, fragment_shape)
        assert len(offsets) == 1, "fragment must have exactly one prototype embedding"
        off_r, off_c = offsets[0]

        fr0, fc0, _fr1, _fc1 = map(int, comp["bbox"])
        top_r = fr0 - off_r
        top_c = fc0 - off_c
        original = {tuple(map(int, rc)) for rc in coords}

        for pr, pc in prototype_shape:
            rr = top_r + pr
            cc = top_c + pc
            assert 0 <= rr < h and 0 <= cc < w, "completed prototype must stay inside the grid"
            if (rr, cc) in original:
                assert int(g[rr, cc]) == color, "fragment pixels must keep their color"
            else:
                assert int(out[rr, cc]) == bg, "completed fragments must not overlap other objects"
            out[rr, cc] = color

    return out


# --------------- generator ---------------


def _random_connected_shape() -> Shape:
    for _ in range(200):
        box_h = random.randint(4, 6)
        box_w = random.randint(4, 6)
        target = random.randint(9, min(15, box_h * box_w - 3))
        cells: set[Coord] = {(random.randrange(box_h), random.randrange(box_w))}

        while len(cells) < target:
            frontier: set[Coord] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            weighted = list(frontier)
            edge_cells = [p for p in weighted if p[0] in (0, box_h - 1) or p[1] in (0, box_w - 1)]
            if edge_cells and random.random() < 0.45:
                cells.add(random.choice(edge_cells))
            else:
                cells.add(random.choice(weighted))

        shape = _normalize(cells)
        h, w = _shape_size(shape)
        if len(shape) != target or h < 4 or w < 4:
            continue
        if len(shape) > h * w - 2:
            continue

        mask = _shape_to_mask(shape)
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        return _transform_shape(shape)

    return _transform_shape(random.choice(_fallback_shapes()))


def _fallback_shapes() -> list[Shape]:
    masks = [
        [
            "01000",
            "11100",
            "01010",
            "01110",
            "00010",
        ],
        [
            "0110",
            "1110",
            "0100",
            "0111",
            "0001",
        ],
        [
            "00100",
            "11100",
            "00110",
            "01110",
            "01000",
        ],
        [
            "0100",
            "1110",
            "0010",
            "0011",
            "0110",
            "0100",
        ],
    ]
    out: list[Shape] = []
    for rows in masks:
        arr = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
        out.append(_mask_to_shape(arr))
    return out


def _transform_shape(shape: Shape) -> Shape:
    mask = _shape_to_mask(shape)
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _mask_to_shape(mask)


def _fragment_candidates(prototype: Shape) -> list[dict]:
    pcoords = list(prototype)
    n = len(pcoords)
    candidates: list[dict] = []
    seen: set[tuple[Shape, Coord]] = set()
    ph, pw = _shape_size(prototype)

    for size in range(3, n - 1):
        for combo in combinations(pcoords, size):
            cells = set(combo)
            if not _connected_cells(cells):
                continue

            fragment = _normalize(cells)
            offsets = _embedding_offsets(prototype, fragment)
            if len(offsets) != 1:
                continue
            offset = offsets[0]
            key = (fragment, offset)
            if key in seen:
                continue
            seen.add(key)

            fh, fw = _shape_size(fragment)
            before = offset[0] > 0 or offset[1] > 0
            after = offset[0] + fh < ph or offset[1] + fw < pw
            if n - size < 2:
                continue
            candidates.append(
                {
                    "cells": tuple(sorted(cells)),
                    "shape": fragment,
                    "offset": offset,
                    "before": before,
                    "after": after,
                    "bidirectional": before and after,
                    "area": int(size),
                }
            )
    return candidates


def _can_place(occupied: np.ndarray, shape: Shape, r0: int, c0: int, padding: int) -> bool:
    h, w = occupied.shape
    for pr, pc in shape:
        rr = r0 + pr
        cc = c0 + pc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r_min = max(0, rr - padding)
        r_max = min(h, rr + padding + 1)
        c_min = max(0, cc - padding)
        c_max = min(w, cc + padding + 1)
        if bool(occupied[r_min:r_max, c_min:c_max].any()):
            return False
    return True


def _random_location(occupied: np.ndarray, shape: Shape, padding: int) -> Coord | None:
    h, w = occupied.shape
    sh, sw = _shape_size(shape)
    options: list[Coord] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if _can_place(occupied, shape, r0, c0, padding):
                options.append((r0, c0))
    if not options:
        return None
    return random.choice(options)


def _mark(occupied: np.ndarray, shape: Shape, r0: int, c0: int) -> None:
    for pr, pc in shape:
        occupied[r0 + pr, c0 + pc] = True


def _paint(grid: np.ndarray, cells, r0: int, c0: int, color: int) -> None:
    for pr, pc in cells:
        grid[r0 + pr, c0 + pc] = int(color)


def _choose_fragments(candidates: list[dict], count: int) -> list[dict] | None:
    shifted = [c for c in candidates if c["before"]]
    bidirectional = [c for c in candidates if c["bidirectional"]]
    after = [c for c in candidates if c["after"]]
    if not shifted or not after:
        return None

    selected: list[dict] = []
    selected.append(random.choice(bidirectional or shifted))
    if count > 1:
        selected.append(random.choice(after))
    while len(selected) < count:
        selected.append(random.choice(candidates))
    random.shuffle(selected)
    return selected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, prototype: Shape) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    bg = _bg_color(inp)
    comps = _components(inp, bg)
    areas = [comp["area"] for comp in comps]
    if areas.count(max(areas)) != 1:
        return False

    proto_comp = comps[areas.index(max(areas))]
    if _normalize(map(tuple, proto_comp["coords"])) != prototype:
        return False

    shifted = False
    after = False
    additions = 0
    ph, pw = _shape_size(prototype)
    for comp in comps:
        if comp is proto_comp:
            continue
        frag = _normalize(map(tuple, comp["coords"]))
        offsets = _embedding_offsets(prototype, frag)
        if len(offsets) != 1 or len(frag) >= len(prototype):
            return False
        off_r, off_c = offsets[0]
        fh, fw = _shape_size(frag)
        shifted |= off_r > 0 or off_c > 0
        after |= off_r + fh < ph or off_c + fw < pw
        additions += len(prototype) - len(frag)
    return shifted and after and additions >= 4


def _gen_attempt() -> np.ndarray | None:
    prototype = _random_connected_shape()
    candidates = _fragment_candidates(prototype)
    if len(candidates) < 8:
        return None

    fragment_count = random.randint(2, 5)
    selected = _choose_fragments(candidates, fragment_count)
    if selected is None:
        return None

    ph, pw = _shape_size(prototype)
    h = random.randint(max(17, 2 * ph + 8), 30)
    w = random.randint(max(17, 2 * pw + 8), 30)

    bg = random.choice(ARC_COLORS)
    palette = [c for c in ARC_COLORS if c != bg]
    colors = random.sample(palette, fragment_count + 1)
    proto_color = colors[0]
    fragment_colors = colors[1:]

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    proto_loc = _random_location(occupied, prototype, OBJECT_PADDING)
    if proto_loc is None:
        return None
    pr0, pc0 = proto_loc
    _paint(inp, prototype, pr0, pc0, proto_color)
    _paint(expected, prototype, pr0, pc0, proto_color)
    _mark(occupied, prototype, pr0, pc0)

    for frag, color in zip(selected, fragment_colors):
        loc = _random_location(occupied, prototype, OBJECT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        _paint(inp, frag["cells"], r0, c0, color)
        _paint(expected, prototype, r0, c0, color)
        _mark(occupied, prototype, r0, c0)

    if not _semantic_checks(inp, expected, prototype):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prototype-fragment completion input")
