from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the dominant grid color."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coordinate array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
) -> list[dict]:
    """Find foreground connected components using arc_utils-compatible semantics."""
    if int(connectivity) != 4:
        raise ValueError("this puzzle only uses 4-connectivity")

    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )

    return out


def _signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in patch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the sprite pattern whose frequency is the smallest after excluding both the most frequent and least frequent groups.

    Input structure:
    - The background is the dominant color.
    - Foreground pixels form several separated 4-connected sprites.
    - Sprites belong to exact repeated pattern classes: within a class, the full
      colored bounding-box crop is identical, including black cells inside the bbox.

    Rule:
    - Crop every 4-connected sprite to its bounding box and group sprites by exact crop.
    - Count how many sprites belong to each crop pattern.
    - Discard the group with the largest count and the group with the smallest count.
    - Among the remaining groups, choose the one with the smallest count and return
      any one of its crops.

    Key insight:
    - The answer is not the rarest or the most common pattern; it is the lower
      middle-frequency pattern after removing both extremes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4)
    assert components, "expected at least one foreground sprite"

    groups: dict[tuple[tuple[int, ...], ...], list[np.ndarray]] = defaultdict(list)
    for comp in components:
        patch = crop_bbox(g, comp["bbox"])
        groups[_signature(patch)].append(patch)

    counts = sorted((len(patches), sig) for sig, patches in groups.items())
    assert len(counts) >= 3, "expected at least three pattern groups"

    min_count = counts[0][0]
    max_count = counts[-1][0]
    middle = [(count, sig) for count, sig in counts if min_count < count < max_count]
    assert middle, "expected at least one non-extreme frequency group"

    target_sig = min(middle)[1]
    return groups[target_sig][0].copy()


_BG = 0
_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_COUNT_OPTIONS = [
    counts
    for counts in combinations(range(1, 7), 4)
    if 10 <= sum(counts) <= 15
]
_MAX_GENERATION_ATTEMPTS = 300
_PLACEMENT_PADDING = 1

_BASE_MASKS = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    H, W = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            sig = _signature(candidate.astype(int))
            if sig not in seen:
                seen.add(sig)
                out.append(candidate.astype(bool))
    return out


def _build_mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in _BASE_MASKS:
        assert 4 <= int(base.sum()) <= 7
        assert _is_connected(base)
        for mask in _mask_variants(base):
            sig = _signature(mask.astype(int))
            if sig not in seen:
                seen.add(sig)
                masks.append(mask)
    return masks


_MASK_LIBRARY = _build_mask_library()


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_categories() -> list[dict]:
    masks = random.sample(_MASK_LIBRARY, 4)
    colors = random.sample(_PALETTE, 4)
    counts = random.sample(list(random.choice(_COUNT_OPTIONS)), 4)
    categories = []
    for mask, color, count in zip(masks, colors, counts):
        categories.append({"patch": _paint_mask(mask, color), "count": int(count)})
    return categories


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    H, W = blocked.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(patch != _BG)
    for dr, dc in coords:
        if blocked[int(r0 + dr), int(c0 + dc)]:
            return False
    return True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    H, W = grid.shape
    for dr, dc in np.argwhere(patch != _BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(patch[dr, dc])
    for dr, dc in np.argwhere(patch != _BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - _PLACEMENT_PADDING)
        r2 = min(H, rr + _PLACEMENT_PADDING + 1)
        c1 = max(0, cc - _PLACEMENT_PADDING)
        c2 = min(W, cc + _PLACEMENT_PADDING + 1)
        blocked[r1:r2, c1:c2] = True


def _place_instances(H: int, W: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(
        key=lambda idx: (int(np.count_nonzero(patches[idx])), int(patches[idx].shape[0] * patches[idx].shape[1])),
        reverse=True,
    )

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        starts = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(starts)
        placed = False
        for r0, c0 in starts:
            if _can_place(blocked, patch, r0, c0):
                _place_patch(grid, blocked, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    return grid


def _winner_patch(categories: list[dict]) -> np.ndarray:
    counts = [cat["count"] for cat in categories]
    min_count = min(counts)
    max_count = max(counts)
    remaining = [cat for cat in categories if min_count < cat["count"] < max_count]
    return min(remaining, key=lambda cat: cat["count"])["patch"]


def _generate_attempt() -> np.ndarray | None:
    categories = _sample_categories()
    instances: list[np.ndarray] = []
    for cat in categories:
        for _ in range(cat["count"]):
            instances.append(cat["patch"])

    total_objects = len(instances)
    side_lo = min(30, 20 + total_objects // 2)
    H = random.randint(side_lo, 30)
    W = random.randint(side_lo, 30)
    grid = _place_instances(H, W, instances)
    if grid is None:
        return None

    expected = _winner_patch(categories)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
