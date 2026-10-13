from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.random import default_rng


BLUE = 1
RED = 2
BG = 0
INPUT_PALETTE = (3, 4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
LOCAL_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 240
GRID_MIN = 17
GRID_MAX = 24
TARGET_GROUP_RANGE = (3, 5)
NOISE_COUNT_RANGE = (1, 3)
SKELETON_AREA_RANGE = (4, 7)
NOISE_AREA_RANGE = (3, 8)
SKELETON_MAX_DIM = 5
OBJECT_MAX_DIM = 7
SPARSITY_LIMIT = 5


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(coords: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = list(coords)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
    return norm, (min_r, min_c)


def _mask_dims(mask: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in mask) + 1, max(c for _, c in mask) + 1


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        cur = stack.pop()
        for nxt in _neighbors4(cur):
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            abs_coords = tuple(sorted(coords))
            r0 = min(rr for rr, _ in abs_coords)
            c0 = min(cc for _, cc in abs_coords)
            r1 = max(rr for rr, _ in abs_coords)
            c1 = max(cc for _, cc in abs_coords)
            local = tuple(sorted((rr - r0, cc - c0) for rr, cc in abs_coords))
            objects.append(
                {
                    "coords": np.array(abs_coords, dtype=int),
                    "coord_set": frozenset(abs_coords),
                    "bbox": (r0, c0, r1, c1),
                    "local": local,
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _variants_for_object(local: tuple[tuple[int, int], ...]) -> dict[tuple[tuple[int, int], ...], list[dict]]:
    coords = list(local)
    n = len(coords)
    variants: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    seen: set[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = set()
    max_removed = min(2, n - 1)
    for removed in range(max_removed + 1):
        for remove_idx in combinations(range(n), removed):
            decor = {coords[i] for i in remove_idx}
            skeleton = set(coords) - decor
            if not _is_connected(skeleton):
                continue
            mask, _ = _normalize(skeleton)
            skeleton_norm = tuple(sorted(skeleton))
            decor_norm = tuple(sorted(decor))
            key = (mask, skeleton_norm, decor_norm)
            if key in seen:
                continue
            seen.add(key)
            variants.setdefault(mask, []).append(
                {
                    "skeleton": skeleton_norm,
                    "decor": decor_norm,
                    "removed": removed,
                }
            )
    return variants


def _pick_variant(candidates: list[dict]) -> dict:
    return min(candidates, key=lambda item: (item["removed"], item["decor"], item["skeleton"]))


def _support_summary(objects: list[dict]) -> dict[tuple[tuple[int, int], ...], tuple[int, ...]]:
    support: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for idx, obj in enumerate(objects):
        for mask in obj["variants"]:
            support.setdefault(mask, []).append(idx)
    return {mask: tuple(sorted(indices)) for mask, indices in support.items() if len(indices) >= 2}


def _analyze_group(
    grid: np.ndarray,
) -> tuple[int, list[dict], tuple[tuple[int, int], ...], tuple[int, ...], dict, dict[tuple[tuple[int, int], ...], tuple[int, ...]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _components(g, bg)
    assert objects, "expected at least one foreground object"

    metadata: dict[tuple[tuple[int, int], ...], dict] = {}
    for idx, obj in enumerate(objects):
        obj_variants = _variants_for_object(obj["local"])
        obj["variants"] = obj_variants
        for mask, candidates in obj_variants.items():
            metadata.setdefault(mask, {})[idx] = candidates

    support = _support_summary(objects)
    assert support, "expected at least one repeated skeleton"

    best_count = max(len(indices) for indices in support.values())
    best_masks = [mask for mask, indices in support.items() if len(indices) == best_count]
    best_area = max(len(mask) for mask in best_masks)
    best_masks = [mask for mask in best_masks if len(mask) == best_area]
    best_mask = min(best_masks)
    return bg, objects, best_mask, support[best_mask], metadata, support


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the largest family of objects that share a common skeleton after removing
    at most two pixels from each object, then recolor the shared skeleton blue and
    the per-object decoration pixels red.
    """
    bg, objects, best_mask, indices, metadata, _support = _analyze_group(grid)
    selected = [objects[idx] for idx in indices]

    min_r = min(obj["bbox"][0] for obj in selected)
    min_c = min(obj["bbox"][1] for obj in selected)
    max_r = max(obj["bbox"][2] for obj in selected)
    max_c = max(obj["bbox"][3] for obj in selected)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    for idx in indices:
        obj = objects[idx]
        variant = _pick_variant(metadata[best_mask][idx])
        r0, c0, _, _ = obj["bbox"]
        for dr, dc in variant["skeleton"]:
            out[r0 - min_r + dr, c0 - min_c + dc] = BLUE
        for dr, dc in variant["decor"]:
            out[r0 - min_r + dr, c0 - min_c + dc] = RED

    return out


def _random_connected_shape(
    rng: np.random.Generator,
    area_range: tuple[int, int],
    max_dim: int,
) -> tuple[tuple[int, int], ...] | None:
    low, high = area_range
    for _ in range(LOCAL_ATTEMPTS):
        target_area = int(rng.integers(low, high + 1))
        shape = {(0, 0)}
        while len(shape) < target_area:
            options = []
            for cell in shape:
                for nxt in _neighbors4(cell):
                    if nxt in shape:
                        continue
                    trial = set(shape)
                    trial.add(nxt)
                    norm, _ = _normalize(trial)
                    hh, ww = _mask_dims(norm)
                    if hh <= max_dim and ww <= max_dim:
                        options.append(nxt)
            if not options:
                break
            shape.add(options[int(rng.integers(len(options)))])
        if len(shape) != target_area:
            continue
        norm, _ = _normalize(shape)
        hh, ww = _mask_dims(norm)
        if hh < 2 or ww < 2:
            continue
        if len(_fringe(set(norm))) < 4:
            continue
        return norm
    return None


def _fringe(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for cell in shape:
        for nxt in _neighbors4(cell):
            if nxt not in shape:
                border.add(nxt)
    return border


def _decorate_shape(
    rng: np.random.Generator,
    skeleton_mask: tuple[tuple[int, int], ...],
    decor_count: int,
) -> dict | None:
    skeleton = set(skeleton_mask)
    current = set(skeleton)
    decor_cells: list[tuple[int, int]] = []
    for _ in range(decor_count):
        options = []
        for cell in _fringe(current):
            trial = set(current)
            trial.add(cell)
            norm, _ = _normalize(trial)
            hh, ww = _mask_dims(norm)
            if hh <= OBJECT_MAX_DIM and ww <= OBJECT_MAX_DIM:
                options.append(cell)
        if not options:
            return None
        chosen = options[int(rng.integers(len(options)))]
        current.add(chosen)
        decor_cells.append(chosen)

    full_mask, shift = _normalize(current)
    sr, sc = shift
    skeleton_local = tuple(sorted((r - sr, c - sc) for r, c in skeleton))
    decor_local = tuple(sorted((r - sr, c - sc) for r, c in decor_cells))
    h, w = _mask_dims(full_mask)
    return {
        "shape": full_mask,
        "skeleton": skeleton_local,
        "decor": decor_local,
        "h": h,
        "w": w,
    }


def _decoration_counts(rng: np.random.Generator, size: int) -> list[int]:
    counts = rng.integers(0, 3, size=size).tolist()
    counts[int(rng.integers(size))] = 2
    if all(count == 2 for count in counts):
        counts[int(rng.integers(size))] = int(rng.integers(0, 2))
    if len(set(counts)) == 1:
        counts[0] = (counts[0] + 1) % 3
        counts[-1] = 2
    rng.shuffle(counts)
    return counts


def _place_shapes(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    shapes: list[dict],
    forbidden_rects: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    blocked = np.zeros((h, w), dtype=bool)
    for rect in forbidden_rects or []:
        r0, c0, r1, c1 = rect
        blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx]["h"] * shapes[idx]["w"], reverse=True)
    for idx in order:
        obj = shapes[idx]
        if obj["h"] > h or obj["w"] > w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(rng.integers(0, h - obj["h"] + 1))
            left = int(rng.integers(0, w - obj["w"] + 1))
            rr0 = max(0, top - 1)
            rr1 = min(h, top + obj["h"] + 1)
            cc0 = max(0, left - 1)
            cc1 = min(w, left + obj["w"] + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [(int(top), int(left)) for top, left in placements]  # type: ignore[misc]


def _compact_layout(shapes: list[dict], placements: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    min_r = min(top for top, _ in placements)
    min_c = min(left for _, left in placements)
    shifted = [(top - min_r, left - min_c) for top, left in placements]
    max_r = max(top + shape["h"] - 1 for shape, (top, left) in zip(shapes, shifted))
    max_c = max(left + shape["w"] - 1 for shape, (top, left) in zip(shapes, shifted))
    return shifted, (max_r + 1, max_c + 1)


def _paint_object(grid: np.ndarray, obj: dict, top: int, left: int, color: int) -> frozenset[tuple[int, int]]:
    coords = []
    for dr, dc in obj["shape"]:
        rr = top + dr
        cc = left + dc
        grid[rr, cc] = color
        coords.append((rr, cc))
    return frozenset(coords)


def _assign_colors(rng: np.random.Generator, count: int) -> list[int]:
    palette_size = int(rng.integers(2, 4))
    palette = rng.choice(np.array(INPUT_PALETTE, dtype=int), size=palette_size, replace=False).tolist()
    colors = list(palette)
    while len(colors) < count:
        colors.append(int(palette[int(rng.integers(len(palette)))]))
    rng.shuffle(colors)
    return [int(color) for color in colors]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    skeleton = _random_connected_shape(rng, SKELETON_AREA_RANGE, SKELETON_MAX_DIM)
    if skeleton is None:
        return None

    group_size = int(rng.integers(TARGET_GROUP_RANGE[0], TARGET_GROUP_RANGE[1] + 1))
    target_shapes: list[dict] = []
    for count in _decoration_counts(rng, group_size):
        obj = _decorate_shape(rng, skeleton, count)
        if obj is None:
            return None
        target_shapes.append(obj)

    temp_h = max(9, sum(shape["h"] for shape in target_shapes) // 2 + group_size + 2)
    temp_w = max(9, sum(shape["w"] for shape in target_shapes) // 2 + group_size + 2)
    temp_h = min(temp_h, GRID_MAX - 4)
    temp_w = min(temp_w, GRID_MAX - 4)
    target_layout = _place_shapes(rng, (temp_h, temp_w), target_shapes)
    if target_layout is None:
        return None
    target_layout, target_span = _compact_layout(target_shapes, target_layout)

    occupied = sum(len(shape["shape"]) for shape in target_shapes)
    if target_span[0] * target_span[1] > occupied * SPARSITY_LIMIT:
        return None

    grid_h = int(rng.integers(max(GRID_MIN, target_span[0] + 4), GRID_MAX + 1))
    grid_w = int(rng.integers(max(GRID_MIN, target_span[1] + 4), GRID_MAX + 1))
    if target_span[0] + 2 > grid_h or target_span[1] + 2 > grid_w:
        return None

    target_top = int(rng.integers(1, grid_h - target_span[0]))
    target_left = int(rng.integers(1, grid_w - target_span[1]))
    target_rect = (
        target_top,
        target_left,
        target_top + target_span[0] - 1,
        target_left + target_span[1] - 1,
    )

    noise_shapes: list[dict] = []
    noise_count = int(rng.integers(NOISE_COUNT_RANGE[0], NOISE_COUNT_RANGE[1] + 1))
    for _ in range(noise_count):
        shape = _random_connected_shape(rng, NOISE_AREA_RANGE, OBJECT_MAX_DIM)
        if shape is None:
            return None
        noise_shapes.append({"shape": shape, "h": _mask_dims(shape)[0], "w": _mask_dims(shape)[1]})

    noise_layout = _place_shapes(rng, (grid_h, grid_w), noise_shapes, forbidden_rects=[target_rect])
    if noise_layout is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = _assign_colors(rng, len(target_shapes) + len(noise_shapes))
    target_sets: set[frozenset[tuple[int, int]]] = set()

    for idx, obj in enumerate(target_shapes):
        local_top, local_left = target_layout[idx]
        abs_top = target_top + local_top
        abs_left = target_left + local_left
        target_sets.add(_paint_object(grid, obj, abs_top, abs_left, colors[idx]))

    for offset, obj in enumerate(noise_shapes, start=len(target_shapes)):
        top, left = noise_layout[offset - len(target_shapes)]
        _paint_object(grid, obj, top, left, colors[offset])

    fg_colors = set(map(int, np.unique(grid))) - {BG}
    if len(fg_colors) not in {2, 3}:
        return None

    bg, objects, best_mask, indices, metadata, support = _analyze_group(grid)
    selected_sets = {objects[idx]["coord_set"] for idx in indices}
    max_count = max(len(group) for group in support.values())
    num_max = sum(len(group) == max_count for group in support.values())
    if selected_sets != target_sets:
        return None
    if best_mask != skeleton:
        return None
    if len(indices) != len(target_shapes):
        return None
    if max_count != len(target_shapes) or num_max != 1:
        return None

    out = main(grid)
    if not np.any(out == BLUE) or not np.any(out == RED):
        return None
    if bg != BG or not metadata[best_mask]:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
