from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FRAME = 5
AREA_TO_COLOR = {
    0: 2,  # area 2..3 -> red
    1: 1,  # area 4..5 -> blue
    2: 3,  # area 6..8 -> green
}

_CATALOG: dict[tuple[bool, int], list[np.ndarray]] | None = None


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_mask_from_patch(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    black = patch == BACKGROUND
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return black & ~outside


def _area_bin(area: int) -> int | None:
    if 2 <= area <= 3:
        return 0
    if 4 <= area <= 5:
        return 1
    if 6 <= area <= 8:
        return 2
    return None


def _max_degree(mask: np.ndarray) -> int:
    h, w = mask.shape
    best = 0
    for r, c in np.argwhere(mask):
        degree = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        best = max(best, degree)
    return best


def _has_interior_2x2(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if mask[r : r + 2, c : c + 2].all():
                return True
    return False


def _is_reasonable_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    base_area = 2 * h + 2 * w - 4
    extra_grey = int(mask.sum()) - base_area
    if extra_grey > 4:
        return False
    if _max_degree(mask) > 3:
        return False
    if _has_interior_2x2(mask):
        return False
    return True


def _build_catalog() -> dict[tuple[bool, int], list[np.ndarray]]:
    catalog: dict[tuple[bool, int], list[np.ndarray]] = {
        (True, 0): [],
        (True, 1): [],
        (True, 2): [],
        (False, 0): [],
        (False, 1): [],
        (False, 2): [],
    }

    for h in range(4, 7):
        for w in range(4, 7):
            base = np.zeros((h, w), dtype=bool)
            base[0, :] = True
            base[-1, :] = True
            base[:, 0] = True
            base[:, -1] = True
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

            for bits in range(1 << len(interior)):
                mask = base.copy()
                for i, (r, c) in enumerate(interior):
                    if bits & (1 << i):
                        mask[r, c] = True

                if len(_components_from_mask(mask)) != 1:
                    continue
                if not _is_reasonable_frame(mask):
                    continue

                hole_mask = _hole_mask_from_patch(mask.astype(int))
                if len(_components_from_mask(hole_mask)) != 1:
                    continue

                hole_area = int(hole_mask.sum())
                bucket = _area_bin(hole_area)
                if bucket is None:
                    continue

                symmetric = bool(np.array_equal(mask, np.fliplr(mask)))
                patch = np.full((h, w), BACKGROUND, dtype=int)
                patch[mask] = FRAME
                catalog[(symmetric, bucket)].append(patch)

    for key, patches in catalog.items():
        if not patches:
            raise RuntimeError(f"missing frame patterns for category {key}")
    return catalog


def _get_catalog() -> dict[tuple[bool, int], list[np.ndarray]]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


def _random_variant(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0
    return not separated


def _place_patches(
    height: int, width: int, patches: list[np.ndarray]
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    indexed = list(enumerate(patches))
    indexed.sort(key=lambda item: int(np.sum(item[1] != BACKGROUND)), reverse=True)

    for _idx, patch in indexed:
        ph, pw = patch.shape
        placed = False
        for _ in range(250):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_too_close(box, other) for other in placed_boxes):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch == FRAME] = FRAME
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return grid


def _generate_attempt() -> np.ndarray | None:
    catalog = _get_catalog()

    object_count = random.randint(4, 6)
    symmetric_count = random.randint(2, object_count - 1)
    asymmetric_count = object_count - symmetric_count

    required_bucket_count = random.randint(2, min(3, symmetric_count))
    required_buckets = random.sample([0, 1, 2], k=required_bucket_count)

    selected_patches: list[np.ndarray] = []
    for bucket in required_buckets:
        selected_patches.append(_random_variant(random.choice(catalog[(True, bucket)])))

    for _ in range(symmetric_count - len(required_buckets)):
        bucket = random.choice([0, 1, 2])
        selected_patches.append(_random_variant(random.choice(catalog[(True, bucket)])))

    for _ in range(asymmetric_count):
        bucket = random.choice([0, 1, 2])
        selected_patches.append(_random_variant(random.choice(catalog[(False, bucket)])))

    random.shuffle(selected_patches)

    height = random.randint(15, 19)
    width = random.randint(15, 19)
    grid = _place_patches(height, width, selected_patches)
    if grid is None:
        return None

    components = _components_from_mask(grid == FRAME)
    if len(components) != object_count:
        return None

    symmetric_areas: list[int] = []
    asymmetric_seen = 0
    for coords in components:
        patch = _crop_bbox(grid, _bbox_coords(coords))
        hole_mask = _hole_mask_from_patch(patch)
        if len(_components_from_mask(hole_mask)) != 1:
            return None
        hole_area = int(hole_mask.sum())
        if _area_bin(hole_area) is None:
            return None
        if np.array_equal(patch, np.fliplr(patch)):
            symmetric_areas.append(hole_area)
        else:
            asymmetric_seen += 1

    if len(symmetric_areas) < 2 or asymmetric_seen < 1:
        return None
    if len({_area_bin(area) for area in symmetric_areas}) < 2:
        return None

    output = main(grid)
    if np.array_equal(output, grid):
        return None
    changed_colors = set(map(int, np.unique(output))) - {BACKGROUND, FRAME}
    if len(changed_colors) < 2:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only vertically symmetric gray frames, and choose each fill color from the enclosed hole area.

    Input structure:
    - The grid is a black background with several gray closed frame objects.
    - Each frame is a 4-connected gray component with one enclosed black hole in its tight bounding box.
    - Some frame bbox patterns are mirror-symmetric left-to-right, and others are not.

    Rule:
    - Detect each gray component and crop it to its tight bounding box.
    - If the bbox contents are unchanged by a left-right reflection, treat the frame as active; otherwise leave it untouched.
    - For each active frame, find its enclosed black hole and fill that hole red when its area is 2..3, blue when its area is 4..5, and green when its area is 6..8.

    Key insight:
    - Vertical symmetry is the gate that decides whether a frame participates at all, while the hole's own area decides the replacement color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for coords in _components_from_mask(g == FRAME):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = _crop_bbox(g, (r0, c0, r1, c1))
        if not np.array_equal(patch, np.fliplr(patch)):
            continue

        hole_mask = _hole_mask_from_patch(patch)
        hole_area = int(hole_mask.sum())
        bucket = _area_bin(hole_area)
        if bucket is None:
            continue

        filled = patch.copy()
        filled[hole_mask] = AREA_TO_COLOR[bucket]
        out[r0 : r1 + 1, c0 : c1 + 1] = filled

    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
