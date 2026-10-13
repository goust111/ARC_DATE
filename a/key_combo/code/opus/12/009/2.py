from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BACKGROUND = 0
FOREGROUND = 1
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _crop_from_coords(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BACKGROUND, dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return out


def _symmetry_score(crop: np.ndarray) -> int:
    mask = crop != BACKGROUND
    return int(np.count_nonzero(mask != np.fliplr(mask)))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for coords in _components_mask(np.asarray(grid) == FOREGROUND):
        crop = _crop_from_coords(grid, coords)
        objs.append(
            {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "crop": crop,
                "score": _symmetry_score(crop),
                "area": int(coords.shape[0]),
            }
        )
    return objs


def _assemble_row(crops: list[np.ndarray]) -> np.ndarray:
    height = max(int(crop.shape[0]) for crop in crops)
    width = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.full((height, width), BACKGROUND, dtype=int)
    c0 = 0
    for crop in crops:
        h, w = crop.shape
        out[:h, c0 : c0 + w] = crop
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrange the cropped blue objects from most symmetric to least symmetric by left-right mismatch score.

    Input structure:
    - The grid has a black background with 4 to 6 separated blue 4-connected objects.
    - Each object is identified only by its shape; all objects share the same foreground color.
    - Objects can have different bounding-box sizes, and their bounding boxes are separated by background.

    Rule:
    - Find every blue 4-connected object and crop each one to its tight bounding box.
    - For each crop, compare the foreground mask with its left-right reflection and count mismatching cells.
    - Sort the crops by this asymmetry score in ascending order.
    - Place the sorted crops left to right with one black separator column, top-aligned in a canvas whose height is the maximum crop height.

    Key insight:
    - The output order is determined neither by position nor area, but by a quantitative mirror-symmetry score computed from each object's own tight crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"
    objects.sort(key=lambda obj: (int(obj["score"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    return _assemble_row([obj["crop"] for obj in objects])


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _shape_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize(coords)
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier -= coords
        candidates = []
        for cell in frontier:
            trial = _normalize(coords | {cell})
            rows = [r for r, _ in trial]
            cols = [c for _, c in trial]
            if max(rows) + 1 <= max_h and max(cols) + 1 <= max_w:
                candidates.append(cell)
        if not candidates:
            return None
        coords.add(random.choice(candidates))

    mask = _shape_to_mask(coords)
    for _ in range(random.randint(0, 3)):
        mask = np.rot90(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _random_symmetric_shape(max_h: int, max_w: int) -> np.ndarray | None:
    templates = [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    ]
    random.shuffle(templates)
    for mask in templates:
        for flip_ud in (False, True):
            variant = np.flipud(mask) if flip_ud else mask
            if variant.shape[0] <= max_h and variant.shape[1] <= max_w and 4 <= int(variant.sum()) <= 10:
                if np.array_equal(variant, np.fliplr(variant)):
                    return variant.copy()
    return None


def _sample_shape_pool(
    needed_scores: int,
    *,
    max_h: int,
    max_w: int,
    require_zero_score: bool,
) -> dict[int, list[np.ndarray]] | None:
    pool: dict[int, list[np.ndarray]] = defaultdict(list)
    seen: set[tuple[tuple[int, int], bytes]] = set()

    if require_zero_score:
        symmetric = _random_symmetric_shape(max_h, max_w)
        if symmetric is not None:
            key = (tuple(map(int, symmetric.shape)), symmetric.tobytes())
            seen.add(key)
            pool[0].append(symmetric.astype(int))

    for _ in range(700):
        area = random.randint(4, 10)
        mask = _random_connected_shape(area, max_h, max_w)
        if mask is None:
            continue
        key = (tuple(map(int, mask.shape)), mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        score = int(np.count_nonzero(mask != np.fliplr(mask)))
        pool[score].append(mask.astype(int))
        if len(pool) >= needed_scores and (not require_zero_score or 0 in pool):
            return pool
    return pool if len(pool) >= needed_scores and (not require_zero_score or 0 in pool) else None


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_patches(canvas: np.ndarray, patches: list[np.ndarray]) -> bool:
    h, w = canvas.shape
    placed_rects: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for index in order:
        patch = patches[index]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_rects_separated(rect, other, 1) for other in placed_rects):
                    candidates.append((r0, c0, rect))
        if not candidates:
            return False
        r0, c0, rect = random.choice(candidates)
        canvas[r0 : r0 + ph, c0 : c0 + pw][patch.astype(bool)] = FOREGROUND
        placed_rects.append(rect)
    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    if not 4 <= len(objects) <= 6:
        return False
    if any(obj["area"] < 4 or obj["area"] > 10 for obj in objects):
        return False
    scores = [int(obj["score"]) for obj in objects]
    if len(set(scores)) != len(scores):
        return False
    score_order = sorted(range(len(objects)), key=lambda i: scores[i])
    x_order = sorted(range(len(objects)), key=lambda i: int(objects[i]["bbox"][1]))
    area_order = sorted(range(len(objects)), key=lambda i: int(objects[i]["area"]))
    if score_order == x_order:
        return False
    if score_order == area_order:
        return False
    out = main(grid)
    expected = _assemble_row([objects[i]["crop"] for i in score_order])
    return np.array_equal(out, expected) and max(out.shape) <= 30


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    grid_h = random.randint(18, 24)
    grid_w = random.randint(18, 24)

    max_patch_w = {4: 6, 5: 5, 6: 4}[n_objects]
    max_patch_h = 5
    require_zero_score = random.random() < 0.75

    pool = _sample_shape_pool(
        n_objects,
        max_h=max_patch_h,
        max_w=max_patch_w,
        require_zero_score=require_zero_score,
    )
    if pool is None:
        return None

    scores = list(pool)
    if require_zero_score and 0 not in scores:
        return None
    if require_zero_score:
        chosen_scores = [0]
        remaining = [score for score in scores if score != 0]
        if len(remaining) < n_objects - 1:
            return None
        chosen_scores.extend(random.sample(remaining, n_objects - 1))
    else:
        chosen_scores = random.sample(scores, n_objects)
    chosen_scores.sort()

    patches = []
    total_output_width = n_objects - 1
    for score in chosen_scores:
        patch = random.choice(pool[score]).astype(int)
        total_output_width += int(patch.shape[1])
        patches.append(patch)
    if total_output_width > 30:
        return None

    canvas = np.zeros((grid_h, grid_w), dtype=int)
    if not _place_patches(canvas, patches):
        return None
    return canvas if _semantic_ok(canvas) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid asymmetry-sorting ARC input")
