from __future__ import annotations

from collections import Counter, deque
from itertools import permutations

import numpy as np


BACKGROUND = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
FEATURE_KEYS = ("color_count", "bend_count", "area", "has_hole")
DIMENSION_PAIRS = tuple(permutations(FEATURE_KEYS, 2))
OBJECT_COUNT_OPTIONS = (5, 6)

MIN_GRID_SIDE = 18
MAX_GRID_SIDE = 30
PLACEMENT_GAP = 1

MAX_GENERATION_ATTEMPTS = 500
MAX_TEMPLATE_ATTEMPTS = 1200
MAX_PLACEMENT_ATTEMPTS = 150

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _count_bends(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=int), 1)
    quad_sums = padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
    return int(np.count_nonzero((quad_sums == 1) | (quad_sums == 3)))


def _has_hole(mask: np.ndarray) -> int:
    filled = np.asarray(mask, dtype=bool)
    holespace = ~filled
    if not holespace.any():
        return 0

    h, w = holespace.shape
    seen = np.zeros_like(holespace, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if holespace[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if holespace[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and holespace[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))

    return int(np.any(holespace & ~seen))


def _describe_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict[str, object]:
    bbox = _bbox_coords(coords)
    r0, c0, _r1, _c1 = bbox
    crop = _crop_bbox(grid, bbox)
    local = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros(crop.shape, dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    colors = {int(v) for v in crop[mask] if int(v) != bg}

    return {
        "coords": coords,
        "bbox": bbox,
        "crop": crop,
        "mask": mask,
        "area": int(mask.sum()),
        "color_count": int(len(colors)),
        "bend_count": _count_bends(mask),
        "has_hole": _has_hole(mask),
    }


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = [_describe_object(g, coords, bg) for coords in _components(g != bg)]
    return bg, objects


def _singleton_outlier_index(items: list[dict[str, object]], key: str) -> int | None:
    values = [int(item[key]) for item in items]
    counts = Counter(values)
    if len(counts) != 2:
        return None

    singleton = [value for value, count in counts.items() if count == 1]
    common = [value for value, count in counts.items() if count == len(values) - 1]
    if len(singleton) != 1 or len(common) != 1:
        return None
    return int(values.index(singleton[0]))


def _valid_dimensions(items: list[dict[str, object]]) -> list[tuple[str, int]]:
    valid: list[tuple[str, int]] = []
    for key in FEATURE_KEYS:
        idx = _singleton_outlier_index(items, key)
        if idx is not None:
            valid.append((key, idx))
    return valid


def _discover_layers(items: list[dict[str, object]]) -> tuple[str, int, str, int]:
    first = _valid_dimensions(items)
    assert len(first) == 1, "expected one unique first-layer outlier dimension"
    first_key, first_idx = first[0]

    survivors = [item for i, item in enumerate(items) if i != first_idx]
    survivor_map = [i for i in range(len(items)) if i != first_idx]

    second = _valid_dimensions(survivors)
    assert len(second) == 1, "expected one unique second-layer outlier dimension"
    second_key, second_local_idx = second[0]
    assert second_key != first_key, "second layer must use a different dimension"

    return first_key, int(first_idx), second_key, int(survivor_map[second_local_idx])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Strip the first unique outlier dimension, then crop the second-layer outlier.

    Input structure:
    - The background contains several small, mutually separated foreground objects.
    - Objects may differ by color diversity, bend count, filled area, and hole presence.
    - Exactly one dimension forms an n-1 same / 1 different split on the full object set,
      and after removing that object a different dimension forms the same pattern again.

    Rule:
    - Extract every foreground object as one 4-connected component on the non-background mask.
    - Find the only feature dimension whose values are identical for all but one object, and
      remove that first-layer outlier from consideration.
    - Recompute the same test on the remaining objects, select the unique second-layer outlier,
      and return the tight crop of its original bounding box, preserving its colors and holes.

    Key insight:
    - The salient first anomaly is only a shell to peel away; the real target is the object that
      becomes the sole exception only after the first-layer outlier has been removed.
    """
    g = np.asarray(grid, dtype=int)
    _bg, objects = _extract_objects(g)
    assert len(objects) >= 4, "expected at least four foreground objects"
    _first_key, _first_idx, _second_key, target_idx = _discover_layers(objects)
    return _crop_bbox(g, objects[target_idx]["bbox"])


def _rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _l_shape(h: int, w: int, thickness: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[-thickness:, :] = True
    return mask


def _t_shape(h: int, w: int) -> np.ndarray:
    stem = max(1, w // 3)
    c0 = (w - stem) // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, c0 : c0 + stem] = True
    return mask


def _ring(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _u_shape(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :3] = True
    mask[1:, 0] = True
    mask[-1, 2:] = True
    mask[:-1, -1] = True
    return mask


def _build_shape_library() -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []

    for h in range(2, 5):
        for w in range(2, 6):
            shapes.append({"family": "rect", "name": f"rect_{h}x{w}", "mask": _rect(h, w)})

    for h in range(3, 6):
        for w in range(3, 6):
            for thickness in (1, 2):
                if thickness < min(h, w):
                    shapes.append(
                        {
                            "family": "L",
                            "name": f"L_{h}x{w}_t{thickness}",
                            "mask": _l_shape(h, w, thickness),
                        }
                    )

    for h in range(3, 6):
        for w in range(3, 7, 2):
            shapes.append({"family": "T", "name": f"T_{h}x{w}", "mask": _t_shape(h, w)})

    for h in range(3, 6):
        for w in range(3, 6):
            shapes.append({"family": "ring", "name": f"ring_{h}x{w}", "mask": _ring(h, w)})

    for h, w in ((3, 5), (4, 5), (5, 5)):
        shapes.append({"family": "U", "name": f"U_{h}x{w}", "mask": _u_shape(h, w)})

    for i, shape in enumerate(shapes):
        mask = np.asarray(shape["mask"], dtype=bool)
        shape["base_index"] = i
        shape["area"] = int(mask.sum())
        shape["bend_count"] = _count_bends(mask)
        shape["has_hole"] = _has_hole(mask)
    return shapes


BASE_SHAPES = _build_shape_library()
CANDIDATE_POOL = [
    {
        "base_index": int(base["base_index"]),
        "family": str(base["family"]),
        "name": str(base["name"]),
        "mask": np.asarray(base["mask"], dtype=bool),
        "area": int(base["area"]),
        "bend_count": int(base["bend_count"]),
        "has_hole": int(base["has_hole"]),
        "color_count": color_count,
    }
    for base in BASE_SHAPES
    for color_count in (1, 2, 3)
]

TEMPLATE_SPECS = (
    (5, "has_hole", "color_count"),
    (6, "has_hole", "color_count"),
    (5, "bend_count", "color_count"),
    (5, "color_count", "has_hole"),
    (6, "color_count", "has_hole"),
    (5, "has_hole", "area"),
    (5, "area", "has_hole"),
    (5, "color_count", "bend_count"),
)


def _count_same_value(items: list[dict[str, object]], key: str, idx: int) -> int:
    values = [int(item[key]) for item in items]
    return int(values.count(values[idx]))


def _template_ok(items: list[dict[str, object]], first_key: str, second_key: str) -> bool:
    first = _valid_dimensions(items)
    if len(first) != 1 or first[0][0] != first_key:
        return False
    first_idx = first[0][1]

    survivors = [item for i, item in enumerate(items) if i != first_idx]
    survivor_map = [i for i in range(len(items)) if i != first_idx]
    second = _valid_dimensions(survivors)
    if len(second) != 1 or second[0][0] != second_key or second_key == first_key:
        return False

    target_idx = survivor_map[second[0][1]]
    if target_idx == first_idx:
        return False

    if any(_count_same_value(items, key, target_idx) == 1 for key in FEATURE_KEYS):
        return False

    if sum(item["family"] == items[target_idx]["family"] for item in items) < 2:
        return False

    return True


def _choose_template(rng: np.random.Generator) -> tuple[list[dict[str, object]], str, str]:
    for spec_index in rng.permutation(len(TEMPLATE_SPECS)):
        n_objects, first_key, second_key = TEMPLATE_SPECS[int(spec_index)]
        for _ in range(MAX_TEMPLATE_ATTEMPTS):
            indices = rng.integers(len(CANDIDATE_POOL), size=n_objects)
            items = [{k: v for k, v in CANDIDATE_POOL[int(idx)].items()} for idx in indices]
            if _template_ok(items, first_key, second_key):
                return items, first_key, second_key
    raise RuntimeError("failed to find a non-degenerate double-outlier template")


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if int(rng.integers(2)):
        out = np.fliplr(out)
    return np.rot90(out, int(rng.integers(4))).copy()


def _ordered_coords(mask: np.ndarray, mode: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if mode == 0:
        order = np.lexsort((coords[:, 1], coords[:, 0]))
    elif mode == 1:
        order = np.lexsort((coords[:, 0], coords[:, 1]))
    elif mode == 2:
        order = np.lexsort((coords[:, 1], coords[:, 0] + coords[:, 1]))
    else:
        order = np.lexsort((coords[:, 0], coords[:, 0] - coords[:, 1]))
    return coords[order]


def _paint_mask(mask: np.ndarray, color_count: int, rng: np.random.Generator) -> np.ndarray:
    colors = rng.choice(PALETTE, size=color_count, replace=False).astype(int)
    coords = _ordered_coords(mask, int(rng.integers(4)))
    chunks = np.array_split(coords, color_count)
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    for color, chunk in zip(colors, chunks):
        patch[chunk[:, 0], chunk[:, 1]] = int(color)
    return patch


def _instantiate_objects(items: list[dict[str, object]], rng: np.random.Generator) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for item in items:
        mask = _random_transform(np.asarray(item["mask"], dtype=bool), rng)
        patch = _paint_mask(mask, int(item["color_count"]), rng)
        objects.append(
            {
                "family": str(item["family"]),
                "base_index": int(item["base_index"]),
                "patch": patch,
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
            }
        )
    return objects


def _place_objects(
    objects: list[dict[str, object]], rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int]]] | tuple[None, None]:
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        height = int(rng.integers(MIN_GRID_SIDE, MAX_GRID_SIDE + 1))
        width = int(rng.integers(MIN_GRID_SIDE, MAX_GRID_SIDE + 1))
        grid = np.full((height, width), BACKGROUND, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(objects)

        order = sorted(
            range(len(objects)),
            key=lambda i: objects[i]["height"] * objects[i]["width"],
            reverse=True,
        )

        ok = True
        for i in order:
            patch = np.asarray(objects[i]["patch"], dtype=int)
            h, w = patch.shape
            candidates: list[tuple[int, int]] = []

            for r0 in range(1, height - h):
                for c0 in range(1, width - w):
                    rr0 = max(0, r0 - PLACEMENT_GAP)
                    cc0 = max(0, c0 - PLACEMENT_GAP)
                    rr1 = min(height, r0 + h + PLACEMENT_GAP)
                    cc1 = min(width, c0 + w + PLACEMENT_GAP)
                    if not blocked[rr0:rr1, cc0:cc1].any():
                        candidates.append((r0, c0))

            if not candidates:
                ok = False
                break

            r0, c0 = candidates[int(rng.integers(len(candidates)))]
            mask = patch != BACKGROUND
            grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]

            rr0 = max(0, r0 - PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            rr1 = min(height, r0 + h + PLACEMENT_GAP)
            cc1 = min(width, c0 + w + PLACEMENT_GAP)
            blocked[rr0:rr1, cc0:cc1] = True
            placements[i] = (r0, c0)

        if ok:
            return grid, [(int(r), int(c)) for r, c in placements if r is not None and c is not None]
    return None, None


def _semantic_check(grid: np.ndarray, expected_first: str, expected_second: str, object_count: int) -> bool:
    bg, objects = _extract_objects(grid)
    if bg != BACKGROUND:
        return False
    if len(objects) != object_count:
        return False

    try:
        first_key, first_idx, second_key, target_idx = _discover_layers(objects)
    except AssertionError:
        return False

    if first_key != expected_first or second_key != expected_second:
        return False

    target = objects[target_idx]
    if any(_count_same_value(objects, key, target_idx) == 1 for key in FEATURE_KEYS):
        return False
    if sum(obj["crop"].shape == target["crop"].shape for obj in objects) < 2:
        return False

    expected = _crop_bbox(grid, target["bbox"])
    actual = main(grid)
    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    items, first_key, second_key = _choose_template(rng)
    objects = _instantiate_objects(items, rng)
    grid, _placements = _place_objects(objects, rng)
    if grid is None:
        return None

    if not _semantic_check(grid, first_key, second_key, len(items)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-layer outlier scene")
