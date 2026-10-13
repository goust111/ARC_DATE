from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _canonical_shift(dr: int, dc: int) -> tuple[int, int]:
    if dr > 0 or (dr == 0 and dc > 0):
        return int(dr), int(dc)
    return int(-dr), int(-dc)


def _consistent_support(grid: np.ndarray, dr: int, dc: int, missing_color: int) -> int:
    r0, r1, c0, c1 = _overlap_slices(grid.shape, dr, dc)
    if r0 >= r1 or c0 >= c1:
        return -1

    a = grid[r0:r1, c0:c1]
    b = grid[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    visible = (a != missing_color) & (b != missing_color)
    support = int(np.count_nonzero(visible))
    if support == 0:
        return -1
    if not np.all(a[visible] == b[visible]):
        return -1
    return support


def _best_translation(grid: np.ndarray, missing_color: int = Color.BLACK) -> tuple[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best_support = 0
    best_shifts: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            support = _consistent_support(g, dr, dc, missing_color)
            if support < 0:
                continue
            if support > best_support:
                best_support = support
                best_shifts = [(int(dr), int(dc))]
            elif support == best_support:
                best_shifts.append((int(dr), int(dc)))

    assert best_support > 0, "expected at least one supported nonzero translation"

    canonical = {_canonical_shift(dr, dc) for dr, dc in best_shifts}
    assert len(canonical) == 1, "max-supported translation must be unique up to sign"
    return next(iter(canonical)), int(best_support)


def _complete_by_translation(grid: np.ndarray, shift: tuple[int, int], missing_color: int = Color.BLACK) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dr, dc = map(int, shift)
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for step_r, step_c in ((dr, dc), (-dr, -dc)):
        r0, r1, c0, c1 = _overlap_slices(g.shape, step_r, step_c)
        for r in range(r0, r1):
            rr = r + step_r
            base = r * w
            shifted = rr * w
            for c in range(c0, c1):
                union(base + c, shifted + c + step_c)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == missing_color:
                continue
            root = find(r * w + c)
            colors_by_root.setdefault(root, set()).add(color)

    assert all(len(colors) == 1 for colors in colors_by_root.values()), "translation classes must be monochrome"

    out = g.copy()
    for r, c in np.argwhere(g == missing_color):
        colors = colors_by_root.get(find(int(r) * w + int(c)), set())
        assert len(colors) == 1, "every missing pixel must be inferable from its translation class"
        out[int(r), int(c)] = next(iter(colors))

    return out


def _extract_unique_half(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    options: list[np.ndarray] = []

    if h % 2 == 0 and np.array_equal(g[: h // 2, :], g[h // 2 :, :]):
        options.append(crop_bbox(g, (0, 0, h // 2 - 1, w - 1)))
    if w % 2 == 0 and np.array_equal(g[:, : w // 2], g[:, w // 2 :]):
        options.append(crop_bbox(g, (0, 0, h - 1, w // 2 - 1)))

    assert len(options) == 1, "completed grid must have exactly one repeated half-axis"
    return options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a hidden translation period, then crop the unique repeated half-tile.

    Input structure:
    - The grid is rectangular with one axis exactly twice the other.
    - Non-black colors form a partially occluded periodic pattern; black cells are missing pixels.
    - The fully restored pattern is exactly two identical halves along one axis, but the raw holes hide that duplication.

    Rule:
    - Among all nonzero translations, find the one whose overlap contains the most visible pairs with no color conflicts.
    - Use that translation and its inverse to link periodic-equivalent cells, then fill every black cell from the unique visible color in its class.
    - After completion, find the unique axis that splits the grid into two identical halves.
    - Output the first half on that axis (top half or left half).

    Key insight:
    - The duplication needed for cropping is invisible at first; only translation-based completion reveals the exact repeated tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.any(g == Color.BLACK), "expected missing black pixels"

    shift, _support = _best_translation(g, Color.BLACK)
    completed = _complete_by_translation(g, shift, Color.BLACK)
    assert not np.any(completed == Color.BLACK), "all missing pixels should be recoverable"
    return _extract_unique_half(completed)


def _build_classes(
    shape: tuple[int, int],
    shifts: list[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = shape
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for dr, dc in shifts:
        r0, r1, c0, c1 = _overlap_slices(shape, int(dr), int(dc))
        for r in range(r0, r1):
            rr = r + int(dr)
            base = r * w
            shifted = rr * w
            for c in range(c0, c1):
                union(base + c, shifted + c + int(dc))

    buckets: dict[int, list[tuple[int, int]]] = {}
    class_of = np.empty(shape, dtype=int)
    index_of_root: dict[int, int] = {}

    for r in range(h):
        for c in range(w):
            root = find(r * w + c)
            if root not in index_of_root:
                index_of_root[root] = len(index_of_root)
                buckets[root] = []
            idx = index_of_root[root]
            class_of[r, c] = idx
            buckets[root].append((int(r), int(c)))

    classes = [buckets[root] for root, _idx in sorted(index_of_root.items(), key=lambda item: item[1])]
    return classes, class_of


def _duplicate_shift(tile_size: int, horizontal: bool) -> tuple[int, int]:
    return (0, tile_size) if horizontal else (tile_size, 0)


def _first_half(grid: np.ndarray, horizontal: bool) -> np.ndarray:
    return grid[:, : grid.shape[1] // 2].copy() if horizontal else grid[: grid.shape[0] // 2, :].copy()


def _union_reconstruct(inp: np.ndarray, horizontal: bool) -> np.ndarray:
    if horizontal:
        left = inp[:, : inp.shape[1] // 2]
        right = inp[:, inp.shape[1] // 2 :]
        return np.where(left != Color.BLACK, left, right)
    top = inp[: inp.shape[0] // 2, :]
    bottom = inp[inp.shape[0] // 2 :, :]
    return np.where(top != Color.BLACK, top, bottom)


def _candidate_period_shifts(shape: tuple[int, int], duplicate_support: int) -> list[tuple[int, int]]:
    h, w = shape
    base = [
        (1, 1),
        (1, -1),
        (1, 2),
        (2, 1),
        (1, -2),
        (2, -1),
        (2, 2),
        (2, -2),
    ]
    out = []
    for dr, dc in base:
        if abs(dr) >= h or abs(dc) >= w:
            continue
        support = (h - abs(dr)) * (w - abs(dc))
        if support > duplicate_support:
            out.append((int(dr), int(dc)))
    return out


def _random_colored_grid(classes: list[list[tuple[int, int]]], shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    color_assignments: list[int] = []
    class_ids = list(range(len(classes)))
    random.shuffle(class_ids)

    for i, class_id in enumerate(class_ids):
        if i < len(palette):
            color_assignments.append(int(palette[i]))
        else:
            color_assignments.append(int(random.choice(palette)))

    per_class = [0] * len(classes)
    for class_id, color in zip(class_ids, color_assignments):
        per_class[class_id] = int(color)

    for class_id, coords in enumerate(classes):
        color = int(per_class[class_id])
        for r, c in coords:
            grid[r, c] = color
    return grid


def _build_complete_example() -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool] | None:
    tile_size = random.randint(6, 10)
    horizontal = bool(random.getrandbits(1))
    shape = (tile_size, tile_size * 2) if horizontal else (tile_size * 2, tile_size)
    duplicate_support = tile_size * tile_size
    shifts = _candidate_period_shifts(shape, duplicate_support)
    if not shifts:
        return None

    period_shift = random.choice(shifts)
    duplicate_shift = _duplicate_shift(tile_size, horizontal)
    classes, _class_of = _build_classes(shape, [period_shift, duplicate_shift])
    if len(classes) < 8:
        return None

    palette_size = random.choice((3, 4))
    palette = random.sample(list(range(1, 10)), palette_size)
    full = _random_colored_grid(classes, shape, palette)

    try:
        best_shift, _support = _best_translation(full, Color.BLACK)
        if best_shift != _canonical_shift(*period_shift):
            return None
        half = _extract_unique_half(full)
    except AssertionError:
        return None

    expected_half = _first_half(full, horizontal)
    if not np.array_equal(half, expected_half):
        return None
    return full, expected_half, period_shift, horizontal


def _pair_coordinates(shape: tuple[int, int], tile_size: int, horizontal: bool) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    if horizontal:
        for r in range(shape[0]):
            for c in range(tile_size):
                pairs.append(((int(r), int(c)), (int(r), int(c + tile_size))))
    else:
        for r in range(tile_size):
            for c in range(shape[1]):
                pairs.append(((int(r), int(c)), (int(r + tile_size), int(c))))
    return pairs


def _make_occluded_input(
    full: np.ndarray,
    tile: np.ndarray,
    period_shift: tuple[int, int],
    horizontal: bool,
) -> np.ndarray | None:
    h, w = full.shape
    tile_size = tile.shape[0]
    total = h * w
    missing_lo = int(np.ceil(total * 0.20))
    missing_hi = int(np.floor(total * 0.30))
    fill_classes, class_of = _build_classes(full.shape, [period_shift])
    pairs = _pair_coordinates(full.shape, tile_size, horizontal)

    for _ in range(200):
        missing_target = random.randint(missing_lo, missing_hi)
        max_both = min(len(pairs) // 5, missing_target // 2, 6)
        if max_both < 2:
            continue
        both_count = random.randint(2, max_both)
        single_count = missing_target - 2 * both_count
        if single_count < 4 or both_count + single_count > len(pairs):
            continue

        pair_indices = list(range(len(pairs)))
        both_indices = set(random.sample(pair_indices, both_count))
        remaining = [idx for idx in pair_indices if idx not in both_indices]
        single_indices = random.sample(remaining, single_count)
        random.shuffle(single_indices)

        missing = np.zeros(full.shape, dtype=bool)
        left_single = 0
        right_single = 0

        for idx in both_indices:
            (r1, c1), (r2, c2) = pairs[idx]
            missing[r1, c1] = True
            missing[r2, c2] = True

        for order, idx in enumerate(single_indices):
            (r1, c1), (r2, c2) = pairs[idx]
            if order == 0:
                take_first = True
            elif order == 1:
                take_first = False
            else:
                take_first = bool(random.getrandbits(1))
            if take_first:
                missing[r1, c1] = True
                left_single += 1
            else:
                missing[r2, c2] = True
                right_single += 1

        if left_single == 0 or right_single == 0:
            continue

        visible_counts = np.zeros(len(fill_classes), dtype=int)
        for r in range(h):
            for c in range(w):
                if not missing[r, c]:
                    visible_counts[class_of[r, c]] += 1
        if np.any(visible_counts == 0):
            continue

        inp = full.copy()
        inp[missing] = Color.BLACK

        if horizontal:
            if np.array_equal(inp[:, :tile_size], inp[:, tile_size:]):
                continue
        else:
            if np.array_equal(inp[:tile_size, :], inp[tile_size:, :]):
                continue

        if np.array_equal(_union_reconstruct(inp, horizontal), tile):
            continue

        try:
            best_shift, _support = _best_translation(inp, Color.BLACK)
            if best_shift != _canonical_shift(*period_shift):
                continue
            completed = _complete_by_translation(inp, best_shift, Color.BLACK)
            if not np.array_equal(completed, full):
                continue
            if not np.array_equal(main(inp), tile):
                continue
        except AssertionError:
            continue

        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_complete_example()
        if built is None:
            continue
        full, tile, period_shift, horizontal = built
        inp = _make_occluded_input(full, tile, period_shift, horizontal)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid periodic-completion crop puzzle")
