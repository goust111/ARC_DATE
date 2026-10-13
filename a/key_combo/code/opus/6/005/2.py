from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 7]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0:r1 + 1, c0:c1 + 1].copy()


def find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def object_mask(obj: dict) -> np.ndarray:
    r0, c0, r1, c1 = obj["bbox"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in obj["coords"]:
        local[int(r) - r0, int(c) - c0] = True
    return local


def symmetry_axes(mask: np.ndarray) -> set[str]:
    axes: set[str] = set()
    if np.array_equal(mask, mask[::-1, :]):
        axes.add("h")
    if np.array_equal(mask, mask[:, ::-1]):
        axes.add("v")
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            axes.add("d")
        anti = np.fliplr(np.flipud(mask)).T
        if np.array_equal(mask, anti):
            axes.add("a")
    return axes


def endpoint_pixels(obj: dict) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in obj["coords"]}
    out: list[tuple[int, int]] = []
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
        if deg == 1:
            out.append((r, c))
    return out


def growth_pixels(shape: tuple[int, int], obj: dict, grid: np.ndarray) -> set[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in obj["coords"]}
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIAGONALS:
            rr, cc = r + dr, c + dc
            if not in_bounds(shape, rr, cc) or int(grid[rr, cc]) != BG:
                continue
            has_orth = any((rr + odr, cc + odc) in cells for odr, odc in ORTHOGONALS)
            if not has_orth:
                out.add((rr, cc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the unique non-reflection-symmetric object diagonally while symmetric objects lose their 4-leaf endpoints.

    Input structure:
    - The grid has black background and 4 or 5 small monochrome 4-connected objects in distinct colors.
    - Exactly one object has no reflection symmetry inside its own tight bbox across horizontal, vertical, main-diagonal, or anti-diagonal axes.
    - Every other object has at least one such reflection axis, and the objects are well separated.

    Rule:
    - Split the foreground objects by the bbox-local symmetry test.
    - For the unique asymmetric object, paint its color onto every background cell that is diagonally adjacent to it but not orthogonally adjacent to it.
    - For each symmetric object, remove every pixel whose 4-neighborhood degree inside that object is exactly 1, using a single non-iterative pass.
    - If a diagonal growth target is not background in the original input, do not overwrite it.

    Key insight:
    - Symmetry is the global selector: it decides which object grows and which objects shrink, while the same diagonal-vs-orthogonal geometry drives both edits in opposite directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = find_components(g, bg=BG)
    assert len(objects) >= 2, "expected multiple foreground objects"

    asymmetric: list[dict] = []
    symmetric: list[dict] = []
    for obj in objects:
        if symmetry_axes(object_mask(obj)):
            symmetric.append(obj)
        else:
            asymmetric.append(obj)
    assert len(asymmetric) == 1, "expected exactly one asymmetric object"

    out = g.copy()
    for obj in symmetric:
        for r, c in endpoint_pixels(obj):
            out[r, c] = BG

    odd = asymmetric[0]
    odd_color = int(odd["color"])
    for r, c in growth_pixels(g.shape, odd, g):
        if int(g[r, c]) == BG and int(out[r, c]) == BG:
            out[r, c] = odd_color
    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            trimmed = trim_mask(candidate)
            key = (trimmed.shape, trimmed.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(trimmed)
    return variants


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


ORTH_SYM_BASES = [
    _mask_from_rows(["1111"]),
    _mask_from_rows(["11111"]),
    _mask_from_rows(["010", "111"]),
    _mask_from_rows(["111", "010", "010"]),
    _mask_from_rows(["010", "111", "010"]),
    _mask_from_rows(["101", "111"]),
    _mask_from_rows(["101", "111", "101"]),
    _mask_from_rows(["111111"]),
    _mask_from_rows(["1111111"]),
]
DIAG_SYM_BASES = [
    _mask_from_rows(["111", "110", "100"]),
]
ASYM_BASES = [
    _mask_from_rows(["100", "101", "111"]),
    _mask_from_rows(["110", "010", "011"]),
    _mask_from_rows(["100", "111", "011"]),
]


def _build_pool(bases: list[np.ndarray], predicate) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in bases:
        for variant in _unique_variants(base):
            if predicate(variant):
                key = (variant.shape, variant.tobytes())
                if key not in seen:
                    seen.add(key)
                    pool.append(variant)
    return pool


ORTH_SYM_POOL = _build_pool(
    ORTH_SYM_BASES,
    lambda m: 4 <= _mask_area(m) <= 8 and bool(symmetry_axes(m)),
)
DIAG_ONLY_POOL = _build_pool(
    DIAG_SYM_BASES,
    lambda m: 4 <= _mask_area(m) <= 8 and symmetry_axes(m) & {"d", "a"} and not symmetry_axes(m) & {"h", "v"},
)
ASYM_POOL = _build_pool(
    ASYM_BASES,
    lambda m: 4 <= _mask_area(m) <= 8 and not symmetry_axes(m),
)


def _placeable(coords: list[tuple[int, int]], placed: list[dict]) -> bool:
    current = set(coords)
    for other in placed:
        for r0, c0 in current:
            for r1, c1 in other["coords"]:
                if max(abs(r0 - r1), abs(c0 - c1)) < 3:
                    return False
    return True


def _mask_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _choose_symmetric_masks(rng: random.Random, asym_area: int, n_objects: int) -> list[np.ndarray]:
    small_choices = [m for m in ORTH_SYM_POOL if _mask_area(m) < asym_area]
    large_choices = [m for m in ORTH_SYM_POOL if _mask_area(m) > asym_area]
    assert small_choices and large_choices and DIAG_ONLY_POOL

    chosen = [
        rng.choice(DIAG_ONLY_POOL),
        rng.choice(small_choices),
        rng.choice(large_choices),
    ]
    while len(chosen) < n_objects - 1:
        chosen.append(rng.choice(ORTH_SYM_POOL + DIAG_ONLY_POOL))
    rng.shuffle(chosen)
    return chosen


def _render_case(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in spec["coords"]:
            grid[r, c] = int(spec["color"])
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    objects = find_components(grid, bg=BG)
    if len(objects) not in {4, 5}:
        return False

    axes_list = [symmetry_axes(object_mask(obj)) for obj in objects]
    asymmetric_ids = [i for i, axes in enumerate(axes_list) if not axes]
    if len(asymmetric_ids) != 1:
        return False

    asym_idx = asymmetric_ids[0]
    areas = [obj["area"] for obj in objects]
    asym_area = areas[asym_idx]
    if not (min(areas) < asym_area < max(areas)):
        return False

    if not any((axes & {"d", "a"}) and not (axes & {"h", "v"}) for axes in axes_list):
        return False

    odd = objects[asym_idx]
    growth = growth_pixels(grid.shape, odd, grid)
    if not growth:
        return False

    for idx, obj in enumerate(objects):
        if idx == asym_idx:
            continue
        if not endpoint_pixels(obj):
            return False
        obj_cells = {tuple(map(int, rc)) for rc in obj["coords"]}
        for gr, gc in growth:
            if any(max(abs(gr - r), abs(gc - c)) <= 1 for r, c in obj_cells):
                return False

    out = main(grid)
    added = set(map(tuple, np.argwhere((grid == BG) & (out != BG))))
    removed = set(map(tuple, np.argwhere((grid != BG) & (out == BG))))
    expected_removed = set()
    for idx, obj in enumerate(objects):
        if idx != asym_idx:
            expected_removed.update(endpoint_pixels(obj))

    if added != growth or removed != expected_removed:
        return False
    if not added or not removed:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        height = rng.randint(14, 18)
        width = rng.randint(14, 18)
        n_objects = rng.choice([4, 5])
        colors = rng.sample(PALETTE, n_objects)

        asym_mask = rng.choice(ASYM_POOL)
        asym_area = _mask_area(asym_mask)
        sym_masks = _choose_symmetric_masks(rng, asym_area, n_objects)

        specs = [{"role": "asym", "mask": asym_mask, "color": colors[0]}]
        specs.extend({"role": "sym", "mask": mask, "color": color} for mask, color in zip(sym_masks, colors[1:]))
        rng.shuffle(specs)

        placed: list[dict] = []
        order = sorted(specs, key=lambda spec: (max(spec["mask"].shape), _mask_area(spec["mask"])), reverse=True)
        ok = True
        for spec in order:
            mask = spec["mask"]
            mh, mw = mask.shape
            positions = [(r, c) for r in range(height - mh + 1) for c in range(width - mw + 1)]
            rng.shuffle(positions)
            picked = None
            for top, left in positions:
                coords = _mask_coords(mask, top, left)
                if _placeable(coords, placed):
                    picked = coords
                    break
            if picked is None:
                ok = False
                break
            placed.append(
                {
                    "role": spec["role"],
                    "mask": mask,
                    "color": spec["color"],
                    "coords": picked,
                }
            )
        if not ok:
            continue

        grid = _render_case(placed, (height, width))
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
