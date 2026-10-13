from __future__ import annotations

import itertools

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        for coords in components_mask(g == color):
            box = bbox_coords(coords)
            patch = crop_bbox(g, box)
            objs.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "area": int(coords.shape[0]),
                    "patch": patch,
                }
            )
    objs.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    return objs


def _assemble_output(objects: list[dict], order: list[int], scales: list[int], patches: list[np.ndarray] | None = None) -> np.ndarray:
    use_patches = [objects[i]["patch"] for i in range(len(objects))] if patches is None else patches
    scaled = [scale_sprite(use_patches[i], scales[i]) for i in order]
    out_h = max(int(arr.shape[0]) for arr in scaled)
    out_w = sum(int(arr.shape[1]) for arr in scaled) + len(scaled) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for arr in scaled:
        paste(out, arr, 0, col)
        col += int(arr.shape[1]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each connected object by its own pixel area, then concatenate the scaled bbox patches from left to right.

    Input structure:
    - The grid has black background and 2 to 4 separated non-black 4-connected objects.
    - Each object is monochrome, has its own color, and fits inside a small bbox.
    - Objects are arranged with distinct horizontal positions so their left-to-right order is readable.

    Rule:
    - Find every non-black 4-connected object and measure its pixel area.
    - Crop that object's minimal bbox, including any black cells inside the bbox, and scale the full bbox patch by the object's area.
    - Sort objects by bbox left column and place the scaled patches left-to-right with one black separator column.
    - Top-align all scaled patches in the output and leave any uncovered cells black.

    Key insight:
    - The counted area is not the final answer by itself; it becomes the scale factor applied back onto that same object's bbox patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    order = list(range(len(objects)))
    scales = [int(obj["area"]) for obj in objects]
    return _assemble_output(objects, order, scales)


_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_COLORS = np.arange(1, 10, dtype=int)

_AREA_MASKS: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ),
    3: (
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    ),
}

_THREE_AREA_ORDERS = [
    perm
    for perm in itertools.permutations((2, 3, 4))
    if perm not in ((2, 3, 4), (4, 3, 2))
]
_TWO_AREA_OPTIONS = ((2, 4), (3, 4), (2, 3))
_MAX_GENERATION_ATTEMPTS = 400


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, int(color), 0).astype(int)


def _choose_area_sequence(mode: int) -> tuple[int, ...]:
    if mode in (0, 2):
        return _THREE_AREA_ORDERS[int(_GEN_RNG.integers(len(_THREE_AREA_ORDERS)))]
    pair = _TWO_AREA_OPTIONS[int(_GEN_RNG.integers(len(_TWO_AREA_OPTIONS)))]
    return tuple(sorted(pair, reverse=True))


def _make_object_patches(areas: tuple[int, ...]) -> list[np.ndarray]:
    colors = _GEN_RNG.choice(_COLORS, size=len(areas), replace=False)
    patches = []
    for area, color in zip(areas, colors):
        mask_bank = _AREA_MASKS[int(area)]
        mask = mask_bank[int(_GEN_RNG.integers(len(mask_bank)))]
        patches.append(_colorize(mask, int(color)))
    return patches


def _layout_grid(patches: list[np.ndarray]) -> np.ndarray:
    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]
    h = max(5, max(heights) + int(_GEN_RNG.integers(1, 4)))
    gaps = [int(_GEN_RNG.integers(2, 5)) for _ in range(len(patches) - 1)]
    lead = int(_GEN_RNG.integers(0, 3))
    tail = int(_GEN_RNG.integers(0, 3))
    w = lead + sum(widths) + sum(gaps) + tail
    grid = np.zeros((h, w), dtype=int)

    c0 = lead
    for idx, patch in enumerate(patches):
        ph, pw = patch.shape
        r0 = int(_GEN_RNG.integers(0, h - ph + 1))
        paste(grid, patch, r0, c0)
        c0 += int(pw)
        if idx < len(gaps):
            c0 += int(gaps[idx])

    return grid


def _objects_have_required_structure(objects: list[dict]) -> bool:
    if len(objects) not in (2, 3):
        return False
    colors = [obj["color"] for obj in objects]
    areas = [obj["area"] for obj in objects]
    if len(set(colors)) != len(colors) or len(set(areas)) != len(areas):
        return False
    if not all(2 <= int(area) <= 4 for area in areas):
        return False
    if not all(obj["patch"].shape[0] <= 3 and obj["patch"].shape[1] <= 3 for obj in objects):
        return False
    if not any(np.any(obj["patch"] == 0) for obj in objects):
        return False
    for left, right in zip(objects, objects[1:]):
        if right["bbox"][1] - left["bbox"][3] - 1 < 2:
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    if not _objects_have_required_structure(objects):
        return False

    actual = main(grid)
    order = list(range(len(objects)))
    scales = [int(obj["area"]) for obj in objects]
    expected = _assemble_output(objects, order, scales)
    if not np.array_equal(actual, expected):
        return False

    area_asc_order = sorted(order, key=lambda i: objects[i]["area"])
    if area_asc_order != order:
        if np.array_equal(actual, _assemble_output(objects, area_asc_order, scales)):
            return False

    count_scales = [len(objects)] * len(objects)
    if np.array_equal(actual, _assemble_output(objects, order, count_scales)):
        return False

    solid_patches = [np.full_like(obj["patch"], obj["color"]) for obj in objects]
    if np.array_equal(actual, _assemble_output(objects, order, scales, patches=solid_patches)):
        return False

    return actual.ndim == 2 and max(actual.shape) <= 30


def _gen_attempt(mode: int) -> np.ndarray | None:
    areas = _choose_area_sequence(mode)
    patches = _make_object_patches(areas)
    grid = _layout_grid(patches)
    if max(grid.shape) > 30:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
