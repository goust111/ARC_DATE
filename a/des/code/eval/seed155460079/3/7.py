import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
LOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
KEY_CANONICAL = np.array([[1, 0], [1, 1]], dtype=bool)

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
BOOK_GAP = 1
OBJECT_PAD = 1
EDGE_MARGIN = 1


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pt = (nr, nc)
                if pt not in seen:
                    seen.add(pt)
                    q.append(pt)
    return len(seen) == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, -k)) for k in range(1, 4))


def _key_orientation(mask: np.ndarray) -> int | None:
    if mask.shape != (2, 2) or int(mask.sum()) != 3:
        return None
    for k in range(4):
        if np.array_equal(mask, np.rot90(KEY_CANONICAL, -k)):
            return k
    return None


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, tuple[int, int]], dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    locks: dict[int, tuple[int, int]] = {}
    keys: dict[int, dict[str, object]] = {}
    books: dict[int, dict[str, object]] = {}
    singleton_candidates: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color):
            area = int(len(coords))
            r0, c0, r1, c1 = _bbox(coords)
            bh = int(r1 - r0 + 1)
            bw = int(c1 - c0 + 1)
            if area == 25 and bh == 5 and bw == 5 and bool((g[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                if color in locks:
                    raise AssertionError("duplicate lock color")
                locks[color] = (r0, c0)
                continue
            if area == 3 and bh == 2 and bw == 2:
                key_mask = np.zeros((2, 2), dtype=bool)
                for r, c in coords:
                    key_mask[int(r - r0), int(c - c0)] = True
                orientation = _key_orientation(key_mask)
                if orientation is not None:
                    if color in keys:
                        raise AssertionError("duplicate key color")
                    keys[color] = {"coords": coords.copy(), "orientation": orientation}
                    continue
            if area == 1:
                singleton_candidates.append((color, int(coords[0, 0]), int(coords[0, 1])))

    for color, r, c in singleton_candidates:
        if not (1 <= r < h - 1 and c >= 3):
            continue
        touches_edge = r - 1 == 0 or r + 1 == h - 1 or c - 3 == 0 or c == w - 1
        if not touches_edge:
            continue
        block = g[r - 1 : r + 2, c - 3 : c]
        pattern_colors = [int(v) for v in np.unique(block) if int(v) != bg]
        if len(pattern_colors) != 1:
            continue
        paint_color = int(pattern_colors[0])
        if paint_color == color:
            continue
        pattern_mask = block == paint_color
        if not _is_asymmetric(pattern_mask):
            continue
        if color in books:
            raise AssertionError("duplicate book color")
        books[color] = {"mask": pattern_mask.copy(), "paint_color": paint_color}

    return bg, locks, keys, books


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, locks, keys, books = _parse_scene(g)

    out = g.copy()
    for key in keys.values():
        coords = np.asarray(key["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    for color, (r0, c0) in locks.items():
        if color not in keys or color not in books:
            raise AssertionError("every lock must have a matching key and codebook entry")
        orientation = int(keys[color]["orientation"])
        pattern_mask = np.rot90(np.asarray(books[color]["mask"], dtype=bool), -orientation)
        paint_color = int(books[color]["paint_color"])
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = np.where(pattern_mask, paint_color, bg)

    return out


def _random_pattern() -> np.ndarray:
    for _ in range(200):
        cell_count = random.randint(3, 5)
        mask = np.zeros((3, 3), dtype=bool)
        mask.flat[random.sample(range(9), cell_count)] = True
        if _is_connected(mask) and _is_asymmetric(mask):
            return mask
    raise RuntimeError("failed to sample asymmetric pattern")


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    return not bool(reserved[top : top + height, left : left + width].any())


def _reserve(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = OBJECT_PAD) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _place_box(reserved: np.ndarray, height: int, width: int, margin: int = EDGE_MARGIN) -> tuple[int, int] | None:
    h, w = reserved.shape
    min_r = margin
    min_c = margin
    max_r = h - height - margin
    max_c = w - width - margin
    if max_r < min_r or max_c < min_c:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(min_r, max_r)
        left = random.randint(min_c, max_c)
        if _can_place(reserved, top, left, height, width):
            _reserve(reserved, top, left, height, width)
            return top, left
    return None


def _book_positions(edge: str, count: int, height: int, width: int) -> list[tuple[int, int]] | None:
    if edge in ("left", "right"):
        total_h = count * 3 + (count - 1) * BOOK_GAP
        if total_h > height:
            return None
        top = random.randint(0, height - total_h)
        col = 0 if edge == "left" else width - 4
        return [(top + i * (3 + BOOK_GAP), col) for i in range(count)]
    total_w = count * 4 + (count - 1) * BOOK_GAP
    if total_w > width:
        return None
    row = 0 if edge == "top" else height - 3
    left = random.randint(0, width - total_w)
    return [(row, left + i * (4 + BOOK_GAP)) for i in range(count)]


def _paint_book_entry(grid: np.ndarray, top: int, left: int, pattern_mask: np.ndarray, color: int) -> None:
    grid[top : top + 3, left : left + 3][pattern_mask] = GREY
    grid[top + 1, left + 3] = color


def _paint_key(grid: np.ndarray, top: int, left: int, color: int, orientation: int) -> None:
    mask = np.rot90(KEY_CANONICAL, -orientation)
    grid[top : top + 2, left : left + 2][mask] = color


def _build_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    height = random.randint(max(20, 4 * count + 8), 30)
    width = random.randint(max(20, 5 * count + 4), 30)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    colors = random.sample(LOCK_COLORS, count)
    orientations = [random.randrange(4) for _ in range(count)]
    patterns = [_random_pattern() for _ in range(count)]
    edge = random.choice(("top", "bottom", "left", "right"))

    book_pos = _book_positions(edge, count, height, width)
    if book_pos is None:
        return None

    for (top, left), pattern, color in zip(book_pos, patterns, colors):
        _paint_book_entry(grid, top, left, pattern, color)
        _reserve(reserved, top, left, 3, 4)

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        pos = _place_box(reserved, 5, 5)
        if pos is None:
            return None
        top, left = pos
        grid[top : top + 5, left : left + 5] = colors[idx]

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        pos = _place_box(reserved, 2, 2)
        if pos is None:
            return None
        top, left = pos
        _paint_key(grid, top, left, colors[idx], orientations[idx])

    try:
        _bg, locks, keys, books = _parse_scene(grid)
    except AssertionError:
        return None

    color_set = set(colors)
    if set(locks) != color_set or set(keys) != color_set or set(books) != color_set:
        return None

    out = main(grid)
    for color, (r0, c0) in locks.items():
        if int(np.count_nonzero(out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] == GREY)) == 0:
            return None
        key_coords = np.asarray(keys[color]["coords"], dtype=int)
        if int(np.count_nonzero(out[key_coords[:, 0], key_coords[:, 1]])) != 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
