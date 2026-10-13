import itertools
import random
from collections import deque

import numpy as np


BG = 0
LEGEND_GREY = 5
LEGEND_AZURE = 8
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    return mask[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == len(pts)


def _symmetry_category(mask: np.ndarray) -> str:
    lr_sym = np.array_equal(mask, np.fliplr(mask))
    ud_sym = np.array_equal(mask, np.flipud(mask))
    if lr_sym and ud_sym:
        return "both"
    if lr_sym:
        return "lr"
    if ud_sym:
        return "ud"
    return "asym"


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    category = _symmetry_category(mask)
    if category == "both":
        return mask.copy()
    if category == "lr":
        return np.flipud(mask)
    if category == "ud":
        return np.fliplr(mask)
    return np.rot90(mask, 2)


def _centered_row(width: int, filled: int) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    mid = width // 2
    half = filled // 2
    row[mid - half : mid + half + 1] = True
    return row


def _centered_pattern(width: int, fills: tuple[int, ...]) -> np.ndarray:
    return _trim_mask(np.vstack([_centered_row(width, filled) for filled in fills]))


def _dedupe_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    deduped: list[np.ndarray] = []
    seen = set()
    for mask in masks:
        key = (mask.shape, mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mask.copy())
    return deduped


def _build_pattern_libraries() -> tuple[list[np.ndarray], list[np.ndarray]]:
    both: list[np.ndarray] = []
    lr: list[np.ndarray] = []

    for width, fills in ((3, (1, 3)), (5, (1, 3, 5))):
        for height in range(2, 6):
            for seq in itertools.product(fills, repeat=height):
                if not (4 <= sum(seq) <= 8):
                    continue
                mask = _centered_pattern(width, seq)
                if not _is_connected(mask):
                    continue
                cat = _symmetry_category(mask)
                if cat == "both":
                    both.append(mask)
                elif cat == "lr":
                    lr.append(mask)

    rectangles = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
    both.extend(np.ones((h, w), dtype=bool) for h, w in rectangles)

    both = _dedupe_masks(both)
    lr = _dedupe_masks(lr)
    assert both and lr
    return both, lr


BOTH_PATTERNS, LR_PATTERNS = _build_pattern_libraries()


def _random_dihedral(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out)


def _make_both(rng: random.Random) -> np.ndarray:
    mask = BOTH_PATTERNS[rng.randrange(len(BOTH_PATTERNS))].copy()
    mask = _random_dihedral(mask, rng)
    assert _symmetry_category(mask) == "both"
    return mask


def _make_lr(rng: random.Random) -> np.ndarray:
    mask = LR_PATTERNS[rng.randrange(len(LR_PATTERNS))].copy()
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    mask = _trim_mask(mask)
    assert _symmetry_category(mask) == "lr"
    return mask


def _make_ud(rng: random.Random) -> np.ndarray:
    mask = LR_PATTERNS[rng.randrange(len(LR_PATTERNS))].T.copy()
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = _trim_mask(mask)
    assert _symmetry_category(mask) == "ud"
    return mask


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _make_asym(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        target = rng.randint(4, 8)
        coords = {(0, 0)}
        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in coords:
                        frontier.add((nr, nc))
            coords.add(rng.choice(tuple(frontier)))

        mask = _random_dihedral(_coords_to_mask(coords), rng)
        if mask.shape[0] > 5 or mask.shape[1] > 5:
            continue
        if _symmetry_category(mask) != "asym":
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric mask")


def _make_mask(category: str, rng: random.Random) -> np.ndarray:
    if category == "both":
        return _make_both(rng)
    if category == "lr":
        return _make_lr(rng)
    if category == "ud":
        return _make_ud(rng)
    if category == "asym":
        return _make_asym(rng)
    raise ValueError(category)


def _legend_block() -> np.ndarray:
    icon_same = np.array(
        [
            [0, LEGEND_GREY, 0],
            [LEGEND_AZURE, LEGEND_AZURE, LEGEND_AZURE],
            [0, LEGEND_GREY, 0],
        ],
        dtype=int,
    )
    icon_flip_ud = np.array(
        [
            [LEGEND_GREY, LEGEND_GREY, LEGEND_GREY],
            [LEGEND_AZURE, LEGEND_AZURE, LEGEND_AZURE],
            [0, LEGEND_GREY, 0],
        ],
        dtype=int,
    )
    icon_flip_lr = np.array(
        [
            [0, LEGEND_GREY, 0],
            [LEGEND_AZURE, LEGEND_AZURE, LEGEND_AZURE],
            [LEGEND_GREY, LEGEND_GREY, LEGEND_GREY],
        ],
        dtype=int,
    )
    icon_rot180 = np.array(
        [
            [LEGEND_GREY, LEGEND_GREY, 0],
            [LEGEND_AZURE, LEGEND_AZURE, LEGEND_AZURE],
            [0, LEGEND_GREY, LEGEND_GREY],
        ],
        dtype=int,
    )

    legend = np.zeros((6, 6), dtype=int)
    legend[0:3, 0:3] = icon_same
    legend[0:3, 3:6] = icon_flip_ud
    legend[3:6, 0:3] = icon_flip_lr
    legend[3:6, 3:6] = icon_rot180
    return legend


def _connected_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(pts)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 6 and g.shape[1] >= 6

    out = g.copy()
    legend_r0 = g.shape[0] - 6
    legend_c0 = g.shape[1] - 6

    work = g.copy()
    work[legend_r0:, legend_c0:] = BG

    for color in np.unique(work):
        color = int(color)
        if color == BG:
            continue
        for comp in _connected_components_of_color(work, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True

            transformed = _transform_mask(mask)
            patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
            patch[mask] = BG
            patch[transformed] = color
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(gh - mh + 1):
        rr0 = max(0, r - 1)
        rr1 = min(gh, r + mh + 1)
        for c in range(gw - mw + 1):
            cc0 = max(0, c - 1)
            cc1 = min(gw, c + mw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r, c))
    if not candidates:
        return False

    r, c = rng.choice(candidates)
    grid[r : r + mh, c : c + mw][mask] = color
    occupied[max(0, r - 1) : min(gh, r + mh + 1), max(0, c - 1) : min(gw, c + mw + 1)] = True
    return True


def _count_objects(grid: np.ndarray) -> int:
    work = grid.copy()
    work[grid.shape[0] - 6 :, grid.shape[1] - 6 :] = BG
    count = 0
    for color in np.unique(work):
        if int(color) == BG:
            continue
        count += len(_connected_components_of_color(work, int(color)))
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(4, 6)
    h = rng.randint(15 + (n_objects > 4), 23)
    w = rng.randint(15 + (n_objects > 4), 23)

    grid = np.zeros((h, w), dtype=int)
    legend = _legend_block()
    legend_r0 = h - 6
    legend_c0 = w - 6
    grid[legend_r0:, legend_c0:] = legend

    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, legend_r0 - 1) :, max(0, legend_c0 - 1) :] = True

    categories = ["both", "lr", "ud", "asym"]
    while len(categories) < n_objects:
        categories.append(rng.choice(["both", "lr", "ud", "asym"]))
    rng.shuffle(categories)

    colors = rng.sample(OBJECT_COLORS, n_objects)
    objects = []
    for category, color in zip(categories, colors):
        mask = _make_mask(category, rng)
        if mask.sum() < 4 or mask.sum() > 8:
            return None
        if _symmetry_category(mask) != category:
            return None
        if category != "both" and np.array_equal(mask, _transform_mask(mask)):
            return None
        objects.append((mask, color))

    objects.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    for mask, color in objects:
        if not _place_object(grid, occupied, mask, color, rng):
            return None

    if _count_objects(grid) != n_objects:
        return None
    if not np.array_equal(grid[legend_r0:, legend_c0:], legend):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out[legend_r0:, legend_c0:], legend):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
