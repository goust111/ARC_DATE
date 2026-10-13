from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
BRANCHES = ("row", "col", "other")
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "mask must be non-empty"
    r0, c0, r1, c1 = box
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_width_constant_non_bg_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            rows.append(int(r))
    return rows


def _find_occurrences(main_grid: np.ndarray, key_mask: np.ndarray, bg: int) -> list[tuple[int, int]]:
    kh, kw = map(int, key_mask.shape)
    mh, mw = map(int, main_grid.shape)
    hits: list[tuple[int, int]] = []
    for r in range(mh - kh + 1):
        for c in range(mw - kw + 1):
            window = main_grid[r : r + kh, c : c + kw]
            if np.array_equal(window != bg, key_mask):
                hits.append((int(r), int(c)))
    return hits


def _classify_positions(positions: list[tuple[int, int]]) -> str:
    rows = {r for r, _ in positions}
    cols = {c for _, c in positions}
    if len(rows) == 1:
        return "row"
    if len(cols) == 1:
        return "col"
    return "other"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find all direct occurrences of the key shape in the main pattern, then transform the whole main pattern from their alignment.

    Input structure:
    - A main pattern occupies the rows above a single full-width separator line.
    - Below the separator sits one small single-color key shape on the background.
    - Copies of the same shape appear inside the main pattern with arbitrary colors, while other nearby objects may use similar colors or transformed silhouettes.

    Rule:
    - Extract the key shape below the separator as a binary mask, ignoring its color.
    - In the main pattern, find every exact occurrence of that mask without rotating or mirroring it, also ignoring colors.
    - If all occurrence positions share one row, output the left-right mirror of the main pattern; if they all share one column, output the up-down mirror; otherwise output the main pattern rotated by 180 degrees.

    Key insight:
    - The key shape itself is only a lookup mask: the real instruction is the collinearity of all its direct occurrences in the main pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4, "expected a 2D grid with a separator and key"
    bg = _get_bg(g)

    sep_rows = _full_width_constant_non_bg_rows(g, bg)
    assert len(sep_rows) == 1, "expected exactly one full-width non-background separator row"
    sep = sep_rows[0]
    assert 0 < sep < g.shape[0] - 1, "separator must split main pattern and key"

    main_grid = g[:sep, :]
    lower = g[sep + 1 :, :]
    key_box = _bbox(lower != bg)
    assert key_box is not None, "expected a key shape below the separator"
    kr0, kc0, kr1, kc1 = key_box
    key_mask = np.asarray(lower[kr0 : kr1 + 1, kc0 : kc1 + 1] != bg, dtype=bool)

    occurrences = _find_occurrences(main_grid, key_mask, bg)
    assert len(occurrences) >= 2, "expected multiple occurrences of the key shape"

    branch = _classify_positions(occurrences)
    if branch == "row":
        return np.fliplr(main_grid)
    if branch == "col":
        return np.flipud(main_grid)
    return np.rot90(main_grid, 2)


def _sample_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _generate_key_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 5)
        area_hi = min(8, h * w - 1)
        if area_hi < 4:
            continue
        area = random.randint(4, area_hi)
        mask = _sample_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if not _is_connected(mask):
            continue
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to sample a suitable key mask")


def _spaced_starts(limit: int, size: int, count: int, gap: int = 1) -> list[int] | None:
    required = count * size + (count - 1) * gap
    if required > limit:
        return None
    slack = limit - required
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1
    start = extras[0]
    starts = []
    for i in range(count):
        starts.append(int(start))
        start += size + gap + extras[i + 1]
    return starts


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    return not bool(np.any(blocked[rr0:rr1, cc0:cc1]))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r: int, c: int, palette: list[int]) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        grid[r + int(rr), c + int(cc)] = int(random.choice(palette))


def _random_blob_mask(max_h: int = 4, max_w: int = 4) -> np.ndarray:
    for _ in range(100):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        area = random.randint(1, min(h * w, 6))
        mask = _sample_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if not _is_connected(mask):
            continue
        return _crop_mask(mask)
    raise RuntimeError("failed to sample a noise blob")


def _place_random_mask(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    palette: list[int],
    *,
    mandatory: bool,
) -> bool:
    h, w = map(int, mask.shape)
    spots = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1) if _can_place(blocked, r, c, h, w)]
    if not spots:
        return not mandatory
    r, c = random.choice(spots)
    _paint_mask(grid, mask, r, c, palette)
    _reserve(blocked, r, c, h, w)
    return True


def _sample_occurrence_positions(branch: str, main_h: int, main_w: int, kh: int, kw: int) -> list[tuple[int, int]] | None:
    if branch == "row":
        max_count = min(4, (main_w + 1) // (kw + 1))
        if max_count < 2:
            return None
        count = random.randint(2, max_count)
        cols = _spaced_starts(main_w, kw, count, gap=1)
        if cols is None:
            return None
        row = random.randint(0, main_h - kh)
        return [(int(row), int(c)) for c in cols]
    if branch == "col":
        max_count = min(4, (main_h + 1) // (kh + 1))
        if max_count < 2:
            return None
        count = random.randint(2, max_count)
        rows = _spaced_starts(main_h, kh, count, gap=1)
        if rows is None:
            return None
        col = random.randint(0, main_w - kw)
        return [(int(r), int(col)) for r in rows]

    count = random.randint(2, 4)
    blocked = np.zeros((main_h, main_w), dtype=bool)
    positions: list[tuple[int, int]] = []
    all_spots = [(r, c) for r in range(main_h - kh + 1) for c in range(main_w - kw + 1)]
    random.shuffle(all_spots)
    for r, c in all_spots:
        if not _can_place(blocked, r, c, kh, kw):
            continue
        positions.append((int(r), int(c)))
        _reserve(blocked, r, c, kh, kw)
        if len(positions) == count:
            break
    if len(positions) < 2:
        return None
    if _classify_positions(positions) != "other":
        return None
    return positions


def _all_transforms(main_grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "row": np.fliplr(main_grid),
        "col": np.flipud(main_grid),
        "other": np.rot90(main_grid, 2),
    }


def _sample_main_size(kh: int, kw: int) -> tuple[int, int]:
    min_h = max(8, kh + 4)
    min_w = max(8, kw + 4)
    max_h = min(18, 30 - 1 - kh)
    max_w = 18
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _build_attempt() -> np.ndarray | None:
    key_mask = _generate_key_mask()
    kh, kw = map(int, key_mask.shape)
    main_h, main_w = _sample_main_size(kh, kw)
    branch = random.choice(BRANCHES)
    positions = _sample_occurrence_positions(branch, main_h, main_w, kh, kw)
    if positions is None:
        return None

    main_grid = np.full((main_h, main_w), BG, dtype=int)
    blocked = np.zeros_like(main_grid, dtype=bool)
    colors = random.sample(PALETTE, 7)
    sep_color, key_color = colors[:2]
    fill_palette = colors[2:]

    for r, c in positions:
        _paint_mask(main_grid, key_mask, r, c, fill_palette)
        _reserve(blocked, r, c, kh, kw)

    decoy_masks = []
    for candidate in (np.fliplr(key_mask), np.flipud(key_mask), np.rot90(key_mask, 2)):
        if not any(np.array_equal(candidate, seen) for seen in decoy_masks):
            decoy_masks.append(candidate.copy())
    random.shuffle(decoy_masks)
    need_decoys = random.randint(1, min(2, len(decoy_masks)))
    for decoy in decoy_masks[:need_decoys]:
        if not _place_random_mask(main_grid, blocked, decoy, fill_palette, mandatory=True):
            return None

    for _ in range(random.randint(1, 3)):
        blob = _random_blob_mask()
        if np.array_equal(blob, key_mask):
            continue
        _place_random_mask(main_grid, blocked, blob, fill_palette, mandatory=False)

    if _full_width_constant_non_bg_rows(main_grid, BG):
        return None

    found = _find_occurrences(main_grid, key_mask, BG)
    if sorted(found) != sorted(positions):
        return None
    if _classify_positions(found) != branch:
        return None

    transforms = _all_transforms(main_grid)
    outputs = list(transforms.values())
    if any(np.array_equal(outputs[i], outputs[j]) for i in range(3) for j in range(i + 1, 3)):
        return None

    lower_h = kh + random.randint(1, 3)
    key_row = random.randint(0, lower_h - kh)
    if main_w <= kw + 1:
        return None
    key_col = random.randint(1, main_w - kw - 1)

    grid = np.full((main_h + 1 + lower_h, main_w), BG, dtype=int)
    grid[:main_h, :] = main_grid
    grid[main_h, :] = int(sep_color)
    key_patch = np.full((kh, kw), BG, dtype=int)
    key_patch[key_mask] = int(key_color)
    grid[main_h + 1 + key_row : main_h + 1 + key_row + kh, key_col : key_col + kw] = key_patch

    if sorted(_full_width_constant_non_bg_rows(grid, BG)) != [main_h]:
        return None
    if np.array_equal(main(grid), transforms[branch]):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
