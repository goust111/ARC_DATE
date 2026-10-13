from __future__ import annotations

import math
from collections import deque

import numpy as np


MAX_GEN_ATTEMPTS = 256
FOREGROUND_COLORS = np.arange(1, 10, dtype=int)
DIR_INDEX = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))


def _find_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        steps = ((-1, 0), (0, 1), (1, 0), (0, -1))

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def _clockwise_fallback(
    pixels: list[tuple[int, int]],
    start_pixel: tuple[int, int],
) -> list[tuple[int, int]]:
    if not pixels:
        return []

    center_r = sum(r for r, _ in pixels) / len(pixels)
    center_c = sum(c for _, c in pixels) / len(pixels)

    def angle_of(pixel: tuple[int, int]) -> float:
        r, c = pixel
        return math.atan2(-(r - center_r), c - center_c)

    start_angle = angle_of(start_pixel)
    tau = 2.0 * math.pi

    def sort_key(pixel: tuple[int, int]) -> tuple[float, float, int, int]:
        angle = angle_of(pixel)
        clockwise_delta = (start_angle - angle) % tau
        radius = (pixel[0] - center_r) ** 2 + (pixel[1] - center_c) ** 2
        return clockwise_delta, -radius, pixel[0], pixel[1]

    return sorted(pixels, key=sort_key)


def _ordered_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    exterior = _exterior_background(padded)

    contour_pixels: list[tuple[int, int]] = []
    edges: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []

    for r in range(1, padded.shape[0] - 1):
        for c in range(1, padded.shape[1] - 1):
            if not padded[r, c]:
                continue
            pixel = (r - 1, c - 1)
            exposed = False
            if exterior[r - 1, c]:
                edges.append(((r, c), (r, c + 1), pixel, 0))
                exposed = True
            if exterior[r, c + 1]:
                edges.append(((r, c + 1), (r + 1, c + 1), pixel, 1))
                exposed = True
            if exterior[r + 1, c]:
                edges.append(((r + 1, c + 1), (r + 1, c), pixel, 2))
                exposed = True
            if exterior[r, c - 1]:
                edges.append(((r + 1, c), (r, c), pixel, 3))
                exposed = True
            if exposed:
                contour_pixels.append(pixel)

    if not contour_pixels:
        return []

    start_pixel = min(contour_pixels)
    outgoing: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], int]]] = {}
    for start, end, pixel, direction in edges:
        outgoing.setdefault(start, []).append((end, pixel, direction))

    current_start = (start_pixel[0] + 1, start_pixel[1] + 1)
    current_end = (current_start[0], current_start[1] + 1)
    current_pixel = start_pixel
    current_direction = 0
    start_state = (current_start, current_end, current_pixel, current_direction)

    traversal: list[tuple[int, int]] = []
    for _ in range(len(edges) + 1):
        traversal.append(current_pixel)
        candidates = outgoing.get(current_end, [])
        if not candidates:
            break

        next_state = None
        for wanted in (
            (current_direction + 1) % 4,
            current_direction,
            (current_direction - 1) % 4,
            (current_direction + 2) % 4,
        ):
            for end, pixel, direction in candidates:
                if direction == wanted:
                    next_state = (current_end, end, pixel, direction)
                    break
            if next_state is not None:
                break

        if next_state is None:
            break

        current_start, current_end, current_pixel, current_direction = next_state
        if (current_start, current_end, current_pixel, current_direction) == start_state:
            break

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pixel in traversal:
        if pixel not in seen:
            seen.add(pixel)
            ordered.append(pixel)

    if len(ordered) < len(contour_pixels):
        missing = [pixel for pixel in contour_pixels if pixel not in seen]
        ordered.extend(_clockwise_fallback(missing, start_pixel))

    return ordered


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _find_components(grid != 0, connectivity=8)
    objects: list[dict[str, object]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for r, c in coords:
            patch[r - r0, c - c0] = int(grid[r, c])
            mask[r - r0, c - c0] = True
        contour = _ordered_contour(mask)
        objects.append(
            {
                "coords": coords,
                "area": len(coords),
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
                "contour": contour,
                "anchor": min(coords),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll each object's exterior contour into a horizontal color band.

    The output is rectangular, so every band is left-aligned and padded with black
    out to the maximum contour length across all objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    if not objects:
        return np.zeros((0, 0), dtype=int)

    objects.sort(key=lambda obj: (-int(obj["area"]), tuple(obj["anchor"])))
    bands: list[list[int]] = []
    for obj in objects:
        patch = np.asarray(obj["patch"], dtype=int)
        contour = list(obj["contour"])
        assert contour, "every object needs an exterior contour"
        bands.append([int(patch[r, c]) for r, c in contour])

    out_h = 2 * len(bands) - 1
    out_w = max(len(band) for band in bands)
    out = np.zeros((out_h, out_w), dtype=int)
    for idx, band in enumerate(bands):
        out[2 * idx, : len(band)] = np.array(band, dtype=int)
    return out


def _make_rectangle(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(4, 9))
    w = int(rng.integers(4, 10))
    return np.ones((h, w), dtype=bool)


def _make_diamond(rng: np.random.Generator) -> np.ndarray:
    radius = int(rng.integers(2, 5))
    size = 2 * radius + 1
    rr, cc = np.indices((size, size))
    return (np.abs(rr - radius) + np.abs(cc - radius)) <= radius


def _make_ellipse(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(5, 10))
    w = int(rng.integers(5, 11))
    rr, cc = np.indices((h, w))
    cr = (h - 1) / 2.0
    cw = (w - 1) / 2.0
    yr = max(1.25, h / 2.0 - 0.35)
    xr = max(1.25, w / 2.0 - 0.35)
    return ((rr - cr) / yr) ** 2 + ((cc - cw) / xr) ** 2 <= 1.0


def _try_carve_hole(mask: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray | None:
    h, w = mask.shape
    if kind == "rectangle":
        if h < 5 or w < 5:
            return None
        hole_h = int(rng.integers(1, h - 2))
        hole_w = int(rng.integers(1, w - 2))
        top_max = h - hole_h - 1
        left_max = w - hole_w - 1
        if top_max <= 0 or left_max <= 0:
            return None
        top = int(rng.integers(1, top_max + 1))
        left = int(rng.integers(1, left_max + 1))
        carved = mask.copy()
        carved[top : top + hole_h, left : left + hole_w] = False
        return carved

    if kind == "diamond":
        radius = (h - 1) // 2
        if radius < 3:
            return None
        inner = int(rng.integers(1, radius - 1))
        rr, cc = np.indices(mask.shape)
        carved = mask.copy()
        carved[(np.abs(rr - radius) + np.abs(cc - radius)) <= inner] = False
        return carved

    if kind == "ellipse":
        if h < 7 or w < 7:
            return None
        rr, cc = np.indices((h, w))
        cr = (h - 1) / 2.0
        cw = (w - 1) / 2.0
        yr = h / 2.0 - rng.uniform(2.0, max(2.1, h / 2.0 - 0.9))
        xr = w / 2.0 - rng.uniform(2.0, max(2.1, w / 2.0 - 0.9))
        if yr <= 0.8 or xr <= 0.8:
            return None
        carved = mask.copy()
        hole = ((rr - cr) / yr) ** 2 + ((cc - cw) / xr) ** 2 <= 1.0
        carved[hole] = False
        return carved

    return None


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _generate_shape(rng: np.random.Generator) -> np.ndarray | None:
    kind = str(rng.choice(("rectangle", "diamond", "ellipse")))
    if kind == "rectangle":
        mask = _make_rectangle(rng)
    elif kind == "diamond":
        mask = _make_diamond(rng)
    else:
        mask = _make_ellipse(rng)

    if rng.random() < 0.5:
        carved = _try_carve_hole(mask, kind, rng)
        if carved is not None:
            mask = carved

    mask = _trim_mask(mask)
    if mask.sum() < 8:
        return None
    if len(_find_components(mask, connectivity=4)) != 1:
        return None
    if len(_ordered_contour(mask)) < 6:
        return None
    return mask


def _paint_object(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    contour = _ordered_contour(mask)
    assert contour, "shape needs a contour"

    pool_size = int(rng.integers(2, 5))
    pool = [int(x) for x in rng.choice(FOREGROUND_COLORS, size=pool_size, replace=False)]
    segment_count = int(rng.integers(2, min(6, len(contour)) + 1))
    if segment_count > 1:
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, len(contour)), size=segment_count - 1, replace=False))
    else:
        cuts = []
    bounds = [0] + cuts + [len(contour)]

    segment_colors: list[int] = []
    for idx in range(segment_count):
        choices = pool if idx == 0 else [col for col in pool if col != segment_colors[-1]]
        segment_colors.append(int(rng.choice(choices)))
    if len(set(segment_colors)) == 1:
        segment_colors[-1] = next(col for col in pool if col != segment_colors[0])

    for seg_idx, color in enumerate(segment_colors):
        lo = bounds[seg_idx]
        hi = bounds[seg_idx + 1]
        for r, c in contour[lo:hi]:
            patch[r, c] = int(color)

    interior = np.argwhere(mask & (patch == 0))
    if interior.size > 0:
        interior_pool = [int(x) for x in rng.choice(FOREGROUND_COLORS, size=int(rng.integers(1, 4)), replace=False)]
        mode = int(rng.integers(0, 3))
        for r, c in interior:
            if mode == 0:
                color = interior_pool[0]
            elif mode == 1:
                color = interior_pool[(int(r) + int(c)) % len(interior_pool)]
            else:
                color = interior_pool[(2 * int(r) + int(c)) % len(interior_pool)]
            patch[int(r), int(c)] = int(color)

    return patch


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    for r, c in np.argwhere(patch != 0):
        if blocked[top + int(r), left + int(c)]:
            return False
    return True


def _place(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w][patch != 0] = patch[patch != 0]
    for r, c in np.argwhere(patch != 0):
        rr = top + int(r)
        cc = left + int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    blocked[nr, nc] = True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(2, 5))
    objects: list[tuple[np.ndarray, int]] = []
    used_areas: set[int] = set()

    while len(objects) < object_count:
        if len(objects) > 12:
            return None
        mask = _generate_shape(rng)
        if mask is None:
            continue
        area = int(mask.sum())
        if area in used_areas:
            continue
        used_areas.add(area)
        objects.append((_paint_object(mask, rng), area))

    max_h = max(obj.shape[0] for obj, _ in objects)
    max_w = max(obj.shape[1] for obj, _ in objects)
    total_area = sum(int((obj != 0).sum()) for obj, _ in objects)

    min_side = max(16, max_h + 4, max_w + 4)
    slack = int(np.ceil(np.sqrt(total_area))) + 6
    height = int(rng.integers(min_side, min_side + slack + 1))
    width = int(rng.integers(min_side, min_side + slack + 1))

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    rng.shuffle(objects)
    for patch, _area in objects:
        h, w = patch.shape
        placed = False
        for _ in range(200):
            top = int(rng.integers(1, height - h))
            left = int(rng.integers(1, width - w))
            if _can_place(blocked, patch, top, left):
                _place(grid, blocked, patch, top, left)
                placed = True
                break
        if not placed:
            return None

    extracted = _extract_objects(grid)
    if len(extracted) != object_count:
        return None
    if len({int(obj["area"]) for obj in extracted}) != object_count:
        return None
    if any(len(set(grid[r0 + r, c0 + c] for r, c in obj["contour"])) < 2 for obj in extracted for r0, c0, _, _ in [obj["bbox"]]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
