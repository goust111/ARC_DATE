from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PATTERN_COLORS = (3, 4, 5, 6, 7, 8, 9)
SPECIAL = {BLACK, BLUE, RED}
MISSING = -1
MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _best_supported_translations(masked: np.ndarray) -> list[tuple[int, int]]:
    h, w = masked.shape
    best_support = 0
    best: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(masked.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue

            a = masked[r0:r1, c0:c1]
            b = masked[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = (a != MISSING) & (b != MISSING)
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue

            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))

    assert best_support > 0, "no consistent repeating translation found"
    best_set = set(best)
    assert len(best_set) == 2, f"expected one translation pair, got {sorted(best_set)}"
    assert {(-dr, -dc) for dr, dc in best_set} == best_set
    return sorted(best_set)


def _merge_visible_pattern(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    merged = np.full((h, w), MISSING, dtype=int)
    for r in range(h):
        for c in range(w):
            top_color = int(top[r, c])
            bottom_color = int(bottom[r, c])
            top_visible = top_color not in SPECIAL
            bottom_visible = bottom_color not in SPECIAL
            if top_visible and bottom_visible:
                assert top_color == bottom_color, "the two halves must expose the same base pattern"
                merged[r, c] = top_color
            elif top_visible:
                merged[r, c] = top_color
            elif bottom_visible:
                merged[r, c] = bottom_color
    return merged


def _repair_targets(merged: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    h, w = merged.shape
    shifts = _best_supported_translations(merged)
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
        r0, r1, c0, c1 = _overlap_slices((h, w), dr, dc)
        for r in range(r0, r1):
            rr = r + dr
            for c in range(c0, c1):
                union(r * w + c, rr * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(merged[r, c])
            if color == MISSING:
                continue
            colors_by_root.setdefault(find(r * w + c), set()).add(color)

    assert all(len(colors) == 1 for colors in colors_by_root.values())

    out = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(target_mask):
        colors = colors_by_root.get(find(int(r) * w + int(c)), set())
        assert len(colors) == 1, "every target class must remain visible somewhere else"
        out[int(r), int(c)] = next(iter(colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the cells that are black in both stacked masks and restore only those cells from the shared periodic pattern.

    Input structure:
    - The grid is split into two equal-height halves stacked vertically.
    - Each half contains the same underlying periodic color field, plus black holes and one opaque obstacle color: blue in the top half and red in the bottom half.
    - Non-special colors are genuine pattern pixels; black, blue, and red are not part of the recoverable field.

    Rule:
    - Compare corresponding cells of the two halves and mark the positions that are black in both as the repair targets.
    - Merge the visible non-black, non-blue, non-red pixels from both halves into one partial view of the periodic field.
    - Find the nonzero translation with maximum support whose overlapping visible pixels agree, and treat forward/backward shifts by that vector as one equivalence relation.
    - For each repair target, recover its unique class color from the visible pixels in the same translation class; output that color only at the target positions and black elsewhere.

    Key insight:
    - The stacked-mask comparison does not reveal the answer directly; it isolates a small set of true holes, and the repeating translation of the surviving pattern pixels is what determines their colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0

    h = g.shape[0] // 2
    top = g[:h, :]
    bottom = g[h:, :]
    targets = (top == BLACK) & (bottom == BLACK)
    merged = _merge_visible_pattern(top, bottom)
    assert np.any(merged != MISSING), "need at least one visible pattern pixel"
    return _repair_targets(merged, targets)


def _random_blob(
    shape: tuple[int, int],
    size: int,
    forbidden: np.ndarray,
    preferred_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    if size <= 0:
        return np.zeros(shape, dtype=bool)

    allowed = ~forbidden
    if not np.any(allowed):
        return None

    if preferred_seed is not None and np.any(allowed & preferred_seed):
        seed_choices = np.argwhere(allowed & preferred_seed)
    else:
        seed_choices = np.argwhere(allowed)
    seed_r, seed_c = map(int, seed_choices[random.randrange(len(seed_choices))])

    blob = {(seed_r, seed_c)}
    frontier = set()
    for nr, nc in _neighbors4(seed_r, seed_c):
        if _in_bounds(shape, nr, nc) and allowed[nr, nc]:
            frontier.add((nr, nc))

    while len(blob) < size:
        if not frontier:
            frontier = {
                (nr, nc)
                for r, c in blob
                for nr, nc in _neighbors4(r, c)
                if _in_bounds(shape, nr, nc) and allowed[nr, nc] and (nr, nc) not in blob
            }
            if not frontier:
                return None
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        blob.add(pick)
        for nr, nc in _neighbors4(*pick):
            if _in_bounds(shape, nr, nc) and allowed[nr, nc] and (nr, nc) not in blob:
                frontier.add((nr, nc))

    mask = np.zeros(shape, dtype=bool)
    for r, c in blob:
        mask[r, c] = True
    return mask


def _split_total(total: int, parts: int, min_each: int) -> list[int] | None:
    if parts <= 0:
        return [] if total == 0 else None
    if total < parts * min_each:
        return None
    sizes = [min_each] * parts
    remaining = total - parts * min_each
    while remaining > 0:
        idx = random.randrange(parts)
        sizes[idx] += 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def _blob_union(
    shape: tuple[int, int],
    total_size: int,
    component_count: int,
    forbidden: np.ndarray,
    preferred_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    out = np.zeros(shape, dtype=bool)
    if total_size <= 0:
        return out

    min_each = 2 if total_size >= 2 * component_count else 1
    sizes = _split_total(total_size, component_count, min_each)
    if sizes is None:
        return None

    current_forbidden = forbidden.copy()
    for idx, size in enumerate(sizes):
        seed_pref = preferred_seed if idx == 0 else None
        blob = _random_blob(shape, size, current_forbidden, preferred_seed=seed_pref)
        if blob is None:
            return None
        out |= blob
        current_forbidden |= blob
    return out


def _candidate_vectors(h: int, w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    max_dr = min(3, h - 1)
    max_dc = min(4, w - 1)
    for dr in range(-max_dr, max_dr + 1):
        for dc in range(-max_dc, max_dc + 1):
            if dr == 0 and dc == 0:
                continue
            if math.gcd(abs(dr), abs(dc)) != 1:
                continue
            support = (h - abs(dr)) * (w - abs(dc))
            if support < int(0.58 * h * w):
                continue
            candidates.append((dr, dc))
    random.shuffle(candidates)
    return candidates


def _build_periodic_field(
    shape: tuple[int, int], shift: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    dr, dc = shift
    roots = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not _in_bounds(shape, r - dr, c - dc)
    ]
    if len(roots) < 6:
        return None
    roots.sort(key=lambda rc: (rc[0] * (-dc) + rc[1] * dr, rc[0], rc[1]))

    max_colors = min(4, len(PATTERN_COLORS), len(roots))
    min_colors = min(3, max_colors)
    if max_colors < 3:
        return None
    n_colors = random.randint(min_colors, max_colors)
    palette = random.sample(PATTERN_COLORS, n_colors)

    for _ in range(80):
        root_colors = [random.choice(palette) for _ in roots]
        if len(set(root_colors)) < n_colors:
            continue
        short_period = False
        for p in range(1, min(4, len(root_colors))):
            if len(root_colors) > p and all(root_colors[i] == root_colors[i % p] for i in range(len(root_colors))):
                short_period = True
                break
        if short_period:
            continue

        field = np.zeros(shape, dtype=int)
        class_ids = np.full(shape, -1, dtype=int)
        for idx, ((r, c), color) in enumerate(zip(roots, root_colors)):
            rr, cc = r, c
            while _in_bounds(shape, rr, cc):
                field[rr, cc] = int(color)
                class_ids[rr, cc] = idx
                rr += dr
                cc += dc
        assert np.all(class_ids >= 0)
        return field, class_ids
    return None


def _make_masks(
    shape: tuple[int, int], class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = shape
    cells = h * w

    shared_max = min(max(2, cells // 10), max(3, int(0.14 * cells)))
    shared_size = random.randint(2, shared_max)
    top_extra_size = random.randint(3, min(max(3, cells // 9), 10))
    bottom_extra_size = random.randint(3, min(max(3, cells // 9), 10))
    obstacle_max = min(max(4, cells // 8), 12)
    blue_size = random.randint(4, obstacle_max)
    red_size = random.randint(4, obstacle_max)

    shared = _blob_union(shape, shared_size, 1, np.zeros(shape, dtype=bool))
    if shared is None:
        return None

    top_only = _blob_union(
        shape,
        top_extra_size,
        random.choice((1, 1, 2)),
        shared.copy(),
    )
    if top_only is None:
        return None

    bottom_only = _blob_union(
        shape,
        bottom_extra_size,
        random.choice((1, 1, 2)),
        shared | top_only,
    )
    if bottom_only is None:
        return None

    top_black = shared | top_only
    bottom_black = shared | bottom_only

    blue_pref = bottom_only.copy()
    red_pref = top_only.copy()
    blue = _blob_union(
        shape,
        blue_size,
        random.choice((1, 1, 2)),
        top_black.copy(),
        preferred_seed=blue_pref,
    )
    if blue is None:
        return None

    red = _blob_union(
        shape,
        red_size,
        random.choice((1, 1, 2)),
        bottom_black.copy(),
        preferred_seed=red_pref,
    )
    if red is None:
        return None

    visible_union = ~(top_black | blue) | ~(bottom_black | red)
    visible_classes = set(map(int, class_ids[visible_union]))
    if len(visible_classes) != int(class_ids.max()) + 1:
        return None

    if len(set(map(int, class_ids[shared]))) < 2:
        return None
    if len(set(map(int, np.unique(class_ids[shared])))) < 2:
        return None

    hidden_but_not_target = (top_black | blue) & (bottom_black | red) & ~shared
    if not np.any(hidden_but_not_target):
        return None

    return top_black, bottom_black, blue, red


def _oracle_output(field: np.ndarray, shared: np.ndarray) -> np.ndarray:
    out = np.zeros(field.shape, dtype=int)
    out[shared] = field[shared]
    return out


def _attempt() -> np.ndarray | None:
    half_h = random.randint(7, 12)
    half_w = random.randint(10, 16)
    shape = (half_h, half_w)

    for shift in _candidate_vectors(half_h, half_w):
        built = _build_periodic_field(shape, shift)
        if built is None:
            continue
        field, class_ids = built

        for _ in range(80):
            masks = _make_masks(shape, class_ids)
            if masks is None:
                continue
            top_black, bottom_black, blue, red = masks
            shared = top_black & bottom_black

            top = field.copy()
            top[top_black] = BLACK
            top[blue] = BLUE

            bottom = field.copy()
            bottom[bottom_black] = BLACK
            bottom[red] = RED

            top_visible = ~np.isin(top, list(SPECIAL))
            bottom_visible = ~np.isin(bottom, list(SPECIAL))
            if not np.any((top_black & ~bottom_black) & bottom_visible):
                continue
            if not np.any((bottom_black & ~top_black) & top_visible):
                continue

            merged = _merge_visible_pattern(top, bottom)
            try:
                best = set(_best_supported_translations(merged))
            except AssertionError:
                continue
            if best != {shift, (-shift[0], -shift[1])}:
                continue

            expected = _oracle_output(field, shared)
            if int(np.count_nonzero(expected)) < 2:
                continue
            if len(set(map(int, expected[expected != BLACK].tolist()))) < 2:
                continue

            inp = np.vstack([top, bottom])
            try:
                pred = main(inp)
            except AssertionError:
                continue
            if not np.array_equal(pred, expected):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-mask periodic-repair input")
