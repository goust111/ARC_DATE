from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
OUT_COLOR = 8
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
TRANSFORMS = ("hflip", "vflip", "rot180")
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "hflip":
        return np.fliplr(arr)
    if transform == "vflip":
        return np.flipud(arr)
    if transform == "rot180":
        return np.rot90(arr, 2)
    raise ValueError(f"unknown transform: {transform}")


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    fg = grid != bg
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _match_missing(complete_crop: np.ndarray, damaged_crop: np.ndarray) -> np.ndarray | None:
    if complete_crop.shape != damaged_crop.shape:
        return None
    complete_mask = complete_crop != BG
    damaged_mask = damaged_crop != BG
    if not complete_mask.any() or not damaged_mask.any():
        return None

    missing = complete_mask & ~damaged_mask
    extra = damaged_mask & ~complete_mask
    overlap = complete_mask & damaged_mask

    if not missing.any() or extra.any():
        return None
    if not np.array_equal(complete_crop[overlap], damaged_crop[overlap]):
        return None
    return missing


def _all_candidates(grid: np.ndarray) -> list[np.ndarray]:
    components = _find_components(grid, bg=BG)
    assert len(components) == 2, "expected exactly two foreground objects"

    crops = [_crop(grid, comp["bbox"]) for comp in components]
    candidates = []

    for i, j in ((0, 1), (1, 0)):
        for transform in TRANSFORMS:
            transformed_complete = _apply_transform(crops[i], transform)
            missing = _match_missing(transformed_complete, crops[j])
            if missing is None:
                continue
            restored = _apply_transform(missing, transform)
            candidates.append(restored)

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Find the hidden flip/rotation, extract the deleted cells, and map them back to the complete motif's orientation.

    Input structure:
    - The grid has a black background and exactly two 4-connected multicolor foreground objects.
    - One object is a complete motif; the other is the same motif after a hidden horizontal flip, vertical flip, or 180-degree rotation, with some foreground cells deleted.

    Rule:
    - Enumerate which object is complete and which allowed transform maps it into the damaged copy's pose.
    - After aligning the two object crops by bounding box, keep cells that are foreground in the transformed complete motif and background in the damaged copy.
    - Apply the same transform again to send that missing-cell mask back into the complete motif's original pose, crop it to its own bbox, and recolor it teal on black.

    Key insight:
    - The damage must be measured in the damaged copy's transformed coordinate frame first, then inverted back into the complete motif's coordinate frame before producing the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g != BG), "expected foreground"

    candidates = _all_candidates(g)
    assert len(candidates) == 1, "expected a unique complete/damaged transform match"

    missing_mask = candidates[0]
    box = _bbox(missing_mask)
    assert box is not None, "expected at least one missing cell"
    cropped = _crop(missing_mask, box)

    out = np.zeros(cropped.shape, dtype=int)
    out[cropped] = OUT_COLOR
    return out


def _grow_mask(h: int, w: int, area: int, rng: np.random.Generator) -> np.ndarray:
    start = (int(rng.integers(h)), int(rng.integers(w)))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < area:
        options = list(frontier)
        if not options:
            return np.zeros((h, w), dtype=bool)
        weights = np.array(
            [1 + sum((pr, pc) in cells for pr, pc in _neighbors4(r, c)) for r, c in options],
            dtype=float,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(options), p=weights))
        chosen = options[idx]
        frontier.remove(chosen)
        cells.add(chosen)
        for nr, nc in _neighbors4(*chosen):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _colorize(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask)
    color_count = int(rng.integers(3, 6))
    palette = [int(c) for c in rng.choice(INPUT_COLORS, size=color_count, replace=False)]
    motif = np.zeros(mask.shape, dtype=int)

    for idx, (r, c) in enumerate(coords):
        color_idx = (idx + 2 * int(r) + 3 * int(c)) % color_count
        motif[int(r), int(c)] = palette[color_idx]

    return motif


def _gen_motif(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(5, 10))
    w = int(rng.integers(5, 10))
    max_area = max(14, int(0.72 * h * w))
    min_area = max(10, int(0.38 * h * w))
    area = int(rng.integers(min_area, max_area + 1))
    mask = _grow_mask(h, w, area, rng)
    box = _bbox(mask)
    if box is None:
        return None
    mask = _crop(mask, box)
    if min(mask.shape) < 4:
        return None
    if any(np.array_equal(mask, _apply_transform(mask, transform)) for transform in TRANSFORMS):
        return None

    motif = _colorize(mask, rng)
    fg_colors = np.unique(motif[motif != BG])
    if fg_colors.size < 3:
        return None
    return motif


def _remove_connected_subset(mask: np.ndarray, motif: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    damaged = mask.copy()
    area = int(mask.sum())
    target = int(rng.integers(1, min(7, max(2, area // 4)) + 1))
    removed = 0

    while removed < target:
        candidates = []
        for r, c in np.argwhere(damaged):
            if int(damaged.sum()) <= 2:
                break
            damaged[r, c] = False
            if damaged.any() and _is_connected(damaged):
                remaining_colors = np.unique(motif[damaged])
                if remaining_colors.size >= 2:
                    candidates.append((int(r), int(c)))
            damaged[r, c] = True

        if not candidates:
            break
        rr, cc = candidates[int(rng.integers(len(candidates)))]
        damaged[rr, cc] = False
        removed += 1

    if removed == 0 or not _is_connected(damaged):
        return None
    return damaged


def _expected_from_mask(restored_missing: np.ndarray) -> np.ndarray:
    box = _bbox(restored_missing)
    assert box is not None
    cropped = _crop(restored_missing, box)
    out = np.zeros(cropped.shape, dtype=int)
    out[cropped] = OUT_COLOR
    return out


def _place_pair(complete: np.ndarray, damaged: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = complete.shape
    gap = int(rng.integers(3, 8))
    margin = lambda: int(rng.integers(1, 5))
    horizontal = bool(rng.integers(2))

    if horizontal:
        r0 = margin()
        r1 = margin()
        c0 = margin()
        c1 = c0 + w + gap + int(rng.integers(0, 4))
        H = max(r0 + h, r1 + h) + margin()
        W = c1 + w + margin()
        pos_a = (r0, c0)
        pos_b = (r1, c1)
    else:
        r0 = margin()
        r1 = r0 + h + gap + int(rng.integers(0, 4))
        c0 = margin()
        c1 = margin()
        H = r1 + h + margin()
        W = max(c0 + w, c1 + w) + margin()
        pos_a = (r0, c0)
        pos_b = (r1, c1)

    grid = np.zeros((H, W), dtype=int)
    if bool(rng.integers(2)):
        (ra, ca), (rb, cb) = pos_a, pos_b
        grid[ra : ra + h, ca : ca + w] = complete
        grid[rb : rb + h, cb : cb + w] = damaged
    else:
        (ra, ca), (rb, cb) = pos_a, pos_b
        grid[ra : ra + h, ca : ca + w] = damaged
        grid[rb : rb + h, cb : cb + w] = complete
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    motif = _gen_motif(rng)
    if motif is None:
        return None

    transform = TRANSFORMS[int(rng.integers(len(TRANSFORMS)))]
    transformed_complete = _apply_transform(motif, transform)
    transformed_mask = transformed_complete != BG
    damaged_mask = _remove_connected_subset(transformed_mask, transformed_complete, rng)
    if damaged_mask is None:
        return None

    damaged = transformed_complete.copy()
    damaged[~damaged_mask] = BG
    restored_missing = _apply_transform(transformed_mask & ~damaged_mask, transform)
    expected = _expected_from_mask(restored_missing)

    grid = _place_pair(motif, damaged, rng)
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None

    components = _find_components(grid, bg=BG)
    if len(components) != 2:
        return None

    if len(_all_candidates(grid)) != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
