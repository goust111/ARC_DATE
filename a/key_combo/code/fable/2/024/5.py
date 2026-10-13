from __future__ import annotations

from collections import Counter

import numpy as np


RED = 2
BACKGROUND = 0
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    """Return 4-neighbors in the shared (row, col) coordinate system."""
    r = int(r)
    c = int(c)
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox (r0, c0, r1, c1) for an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coord_key(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    base = np.asarray(coords, dtype=int).copy()
    base[:, 0] -= int(base[:, 0].min())
    base[:, 1] -= int(base[:, 1].min())
    order = np.lexsort((base[:, 1], base[:, 0]))
    return tuple((int(base[i, 0]), int(base[i, 1])) for i in order)


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg != RED, "background may not be the red shadow color"

    objects: list[dict] = []
    shadows: list[dict] = []
    colors = sorted(set(map(int, np.unique(g))))
    for color in colors:
        if color == bg:
            continue
        for coords in components_mask(g == color):
            comp = {
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
                "sig": _shape_signature(coords),
                "key": _coord_key(coords),
            }
            if color == RED:
                shadows.append(comp)
            else:
                assert comp["area"] >= 2, "every sprite must have at least two cells"
                objects.append(comp)
    assert len(objects) >= 2, "expected multiple non-red sprites"
    assert len(shadows) >= 2, "expected multiple red shadow candidates"
    return int(bg), objects, shadows


def _candidate_vectors(objects: list[dict], shadows: list[dict]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    shadows_by_sig: dict[tuple[tuple[int, int], ...], list[tuple[int, dict]]] = {}
    for j, shadow in enumerate(shadows):
        shadows_by_sig.setdefault(shadow["sig"], []).append((j, shadow))

    pairs_by_vector: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, obj in enumerate(objects):
        for j, shadow in shadows_by_sig.get(obj["sig"], []):
            dr = int(shadow["bbox"][0] - obj["bbox"][0])
            dc = int(shadow["bbox"][1] - obj["bbox"][1])
            shifted = frozenset((int(r) + dr, int(c) + dc) for r, c in obj["coords"])
            if shifted != shadow["key"]:
                continue
            pairs_by_vector.setdefault((dr, dc), []).append((int(i), int(j)))

    clean: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for vector, pairs in pairs_by_vector.items():
        kept: list[tuple[int, int]] = []
        used_objects: set[int] = set()
        used_shadows: set[int] = set()
        for i, j in pairs:
            if i in used_objects or j in used_shadows:
                continue
            used_objects.add(int(i))
            used_shadows.add(int(j))
            kept.append((int(i), int(j)))
        clean[vector] = kept
    return clean


def _infer_vector(objects: list[dict], shadows: list[dict]) -> tuple[tuple[int, int], list[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    vectors = _candidate_vectors(objects, shadows)
    qualifying = {
        vector: pairs
        for vector, pairs in vectors.items()
        if len({i for i, _j in pairs}) >= 2 and len({_j for _i, _j in pairs}) >= 2
    }
    assert len(qualifying) == 1, "expected one unique global translation with at least two pairs"
    (vector, pairs), = qualifying.items()
    return vector, pairs, vectors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the consistent red shadow translation and erase red impostors.

    Input structure:
    - The background is the most common color and is not red.
    - Every non-red foreground component is a monochrome 4-connected sprite.
    - Red 4-connected components are candidate shadows: some are exact translated
      copies of sprites under one shared offset, while others are decoys.

    Rule:
    - Compare each non-red sprite with each red component of the same shape.
    - The correct offset is the unique translation vector that makes at least two
      sprite-shadow pairs valid at once.
    - Remove every red component, then redraw one red translated copy of every
      non-red sprite at that offset, leaving the original sprites unchanged.

    Key insight:
    - A lone red cluster can support a false offset, so the true shadow vector is
      the only one corroborated by multiple independent shape matches.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, shadows = _extract_scene(g)
    (dr, dc), _pairs, _vectors = _infer_vector(objects, shadows)

    out = g.copy()
    out[out == RED] = bg
    for obj in objects:
        moved = obj["coords"] + np.array([[int(dr), int(dc)]], dtype=int)
        rr = moved[:, 0]
        cc = moved[:, 1]
        assert np.all(0 <= rr) and np.all(rr < g.shape[0])
        assert np.all(0 <= cc) and np.all(cc < g.shape[1])
        assert np.all(out[rr, cc] == bg), "supplemented shadow must land on empty background"
        out[rr, cc] = RED
    return out


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return tuple((int(arr[i, 0]), int(arr[i, 1])) for i in order)


def _transform_shape(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    arr = np.array(shape, dtype=int)
    if flip:
        arr[:, 1] *= -1
    for _ in range(int(rot) % 4):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    return _normalize_shape(tuple(map(tuple, arr)))


BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
)


def _build_shape_library() -> tuple[tuple[tuple[int, int], ...], ...]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                variants.add(_transform_shape(base, rot, flip))
    return tuple(sorted(variants, key=lambda shape: (len(shape), shape[-1][0] + 1, shape[-1][1] + 1, shape)))


SHAPE_LIBRARY = _build_shape_library()


def _sample_true_vector(rng: np.random.Generator) -> tuple[int, int]:
    kind = int(rng.choice(3, p=[0.55, 0.225, 0.225]))
    if kind == 0:
        dr = int(rng.choice((-1, 1)) * rng.integers(2, 8))
        dc = int(rng.choice((-1, 1)) * rng.integers(2, 8))
    elif kind == 1:
        dr = 0
        dc = int(rng.choice((-1, 1)) * rng.integers(3, 9))
    else:
        dr = int(rng.choice((-1, 1)) * rng.integers(3, 9))
        dc = 0
    return int(dr), int(dc)


def _shape_overlaps_shift(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> bool:
    cells = set(shape)
    shifted = {(int(r) + int(dr), int(c) + int(dc)) for r, c in shape}
    return bool(cells & shifted)


def _pad8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r) + int(dr)
                cc = int(c) + int(dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _translate_cell_set(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r) + int(dr), int(c) + int(dc)) for r, c in cells}


def _place_pairs(
    rng: np.random.Generator,
    h: int,
    w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    vector: tuple[int, int],
) -> list[dict] | None:
    dr, dc = map(int, vector)
    order = sorted(range(len(shapes)), key=lambda i: (-len(shapes[i]), shapes[i]))
    reserved: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * len(shapes)

    for idx in order:
        shape = shapes[idx]
        rel = np.array(shape, dtype=int)
        union_rel = np.vstack((rel, rel + np.array([[dr, dc]], dtype=int)))
        min_r, min_c = union_rel.min(axis=0)
        max_r, max_c = union_rel.max(axis=0)

        candidates = [
            (int(r0), int(c0))
            for r0 in range(int(-min_r), int(h - max_r))
            for c0 in range(int(-min_c), int(w - max_c))
        ]
        rng.shuffle(candidates)

        chosen = None
        for r0, c0 in candidates:
            object_cells = {(int(r0 + r), int(c0 + c)) for r, c in shape}
            shadow_cells = _translate_cell_set(object_cells, dr, dc)
            union_cells = object_cells | shadow_cells
            if union_cells & reserved:
                continue
            chosen = {
                "shape": shape,
                "object_cells": object_cells,
                "shadow_cells": shadow_cells,
                "object_bbox": (
                    min(r for r, _ in object_cells),
                    min(c for _, c in object_cells),
                    max(r for r, _ in object_cells),
                    max(c for _, c in object_cells),
                ),
            }
            reserved |= _pad8(union_cells, h, w)
            break

        if chosen is None:
            return None
        placed[idx] = chosen

    return [item for item in placed if item is not None]


def _wrong_vector_pool(true_vector: tuple[int, int]) -> list[tuple[int, int]]:
    tv = tuple(map(int, true_vector))
    pool = []
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if (dr, dc) == (0, 0) or (dr, dc) == tv:
                continue
            if dr == 0 and abs(dc) < 2:
                continue
            if dc == 0 and abs(dr) < 2:
                continue
            if dr != 0 and dc != 0 and max(abs(dr), abs(dc)) < 2:
                continue
            pool.append((int(dr), int(dc)))
    return pool


def _place_decoys(
    rng: np.random.Generator,
    h: int,
    w: int,
    placed: list[dict],
    true_vector: tuple[int, int],
    decoy_count: int,
) -> list[set[tuple[int, int]]] | None:
    reserved = _pad8(set().union(*(item["object_cells"] | item["shadow_cells"] for item in placed)), h, w)
    used_vectors: set[tuple[int, int]] = set()
    decoys: list[set[tuple[int, int]]] = []
    candidates = [(i, vector) for i in range(len(placed)) for vector in _wrong_vector_pool(true_vector)]
    rng.shuffle(candidates)

    for obj_idx, wrong_vector in candidates:
        if wrong_vector in used_vectors:
            continue
        object_cells = placed[obj_idx]["object_cells"]
        decoy_cells = _translate_cell_set(object_cells, wrong_vector[0], wrong_vector[1])
        if not all(0 <= r < h and 0 <= c < w for r, c in decoy_cells):
            continue
        if decoy_cells & reserved:
            continue
        decoys.append(decoy_cells)
        reserved |= _pad8(decoy_cells, h, w)
        used_vectors.add(wrong_vector)
        if len(decoys) == int(decoy_count):
            return decoys
    return None


def _render_scene(
    h: int,
    w: int,
    colors: list[int],
    placed: list[dict],
    keep_indices: set[int],
    decoys: list[set[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BACKGROUND, dtype=int)
    out = np.full((h, w), BACKGROUND, dtype=int)
    for color, item in zip(colors, placed):
        for r, c in item["object_cells"]:
            inp[r, c] = int(color)
            out[r, c] = int(color)
        for r, c in item["shadow_cells"]:
            out[r, c] = RED
            if item["index"] in keep_indices:
                inp[r, c] = RED
    for decoy in decoys:
        for r, c in decoy:
            inp[r, c] = RED
    return inp, out


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    object_count = int(rng.integers(3, 6))
    true_vector = _sample_true_vector(rng)

    candidates = [shape for shape in SHAPE_LIBRARY if not _shape_overlaps_shift(shape, true_vector[0], true_vector[1])]
    if len(candidates) < object_count:
        return None

    chosen_indices = rng.choice(len(candidates), size=object_count, replace=False)
    shapes = [candidates[int(i)] for i in chosen_indices]
    placed = _place_pairs(rng, h, w, shapes, true_vector)
    if placed is None:
        return None
    for idx, item in enumerate(placed):
        item["index"] = int(idx)

    colors = list(map(int, rng.choice(OBJECT_COLORS, size=object_count, replace=False)))
    keep_count = int(rng.integers(2, object_count))
    keep_order = list(range(object_count))
    rng.shuffle(keep_order)
    keep_indices = set(keep_order[:keep_count])

    decoy_count = 1 if rng.random() < 0.6 else 2
    decoys = _place_decoys(rng, h, w, placed, true_vector, decoy_count)
    if decoys is None:
        return None

    inp, out = _render_scene(h, w, colors, placed, keep_indices, decoys)
    return inp, out, true_vector


def _semantic_checks(inp: np.ndarray, out: np.ndarray, true_vector: tuple[int, int]) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    if get_bg(inp) != BACKGROUND or get_bg(out) != BACKGROUND:
        return False
    if np.array_equal(inp, out):
        return False

    input_red = inp == RED
    output_red = out == RED
    if not np.any(input_red & ~output_red):
        return False
    if not np.any(output_red & ~input_red):
        return False

    try:
        bg, objects, shadows = _extract_scene(inp)
        inferred, true_pairs, all_vectors = _infer_vector(objects, shadows)
    except AssertionError:
        return False

    if bg != BACKGROUND or inferred != tuple(map(int, true_vector)):
        return False

    paired_objects = {i for i, _j in true_pairs}
    paired_shadows = {_j for _i, _j in true_pairs}
    if len(paired_objects) < 2:
        return False
    if len(paired_objects) >= len(objects):
        return False
    if len(paired_shadows) >= len(shadows):
        return False

    wrong_but_plausible = False
    for vector, pairs in all_vectors.items():
        unique_objects = {i for i, _j in pairs}
        if vector == tuple(map(int, true_vector)):
            continue
        if len(unique_objects) >= 2:
            return False
        if len(unique_objects) >= 1:
            wrong_but_plausible = True
    if not wrong_but_plausible:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt(rng)
        if built is None:
            continue
        inp, out, true_vector = built
        if _semantic_checks(inp, out, true_vector):
            return inp
    raise RuntimeError("failed to generate a valid shadow-forensics input")
