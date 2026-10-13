from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8

OUTPUT_MAP: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (1, BLUE): ((0, 0), (1, 0), (2, 0)),
    (1, RED): ((1, 0), (1, 1), (1, 2)),
    (1, GREEN): ((0, 2), (1, 2), (2, 2)),
    (2, BLUE): ((0, 0), (0, 1), (0, 2)),
    (2, RED): ((0, 0), (1, 1), (2, 2)),
    (2, GREEN): ((2, 0), (2, 1), (2, 2)),
    (3, BLUE): ((0, 2), (1, 1), (2, 0)),
    (3, RED): ((0, 0), (0, 2), (2, 0), (2, 2)),
    (3, GREEN): ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
}

_KEY_BAG: list[tuple[int, int]] = []


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    grey_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(grey_cols) == 1, "expected one grey separator column"
    sep = int(grey_cols[0])
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape
    assert left.shape[0] == h and left.shape[1] > 0
    return left, right


def _reconstruct_walls(left: np.ndarray) -> np.ndarray:
    orange = left == ORANGE
    h, w = orange.shape
    row_fill = orange.sum(axis=1) > (w / 2.0)
    col_fill = orange.sum(axis=0) > (h / 2.0)

    walls = orange.copy()
    for r, fill in enumerate(row_fill):
        if bool(fill):
            walls[r, :] = True
    for c, fill in enumerate(col_fill):
        if bool(fill):
            walls[:, c] = True
    return walls


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = walls.shape
    seen = np.zeros((h, w), dtype=bool)
    rooms = 0
    for r in range(h):
        for c in range(w):
            if walls[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                rooms += 1
    return int(rooms)


def _left_room_count(left: np.ndarray) -> int:
    return _count_enclosed_regions(_reconstruct_walls(left))


def _raw_room_count(left: np.ndarray) -> int:
    return _count_enclosed_regions(left == ORANGE)


def _dominant_color(right: np.ndarray) -> int:
    counts = Counter(int(v) for v in right.ravel() if int(v) != BLACK)
    assert counts, "expected foreground pixels on the right half"
    color, amount = counts.most_common(1)[0]
    assert list(counts.values()).count(amount) == 1, "dominant color must be unique"
    return int(color)


def _template_for_key(key: tuple[int, int]) -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    for r, c in OUTPUT_MAP[key]:
        out[r, c] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use reconstructed room count and dominant right-side color as a two-key lookup for a 3x3 stencil.

    Input structure:
    - One grey separator column splits the grid into equal left and right halves.
    - The left half contains orange rectilinear walls on black, with some wall cells replaced by magenta noise.
    - The right half contains scattered non-black pixels whose unique most frequent color is blue, red, or green.

    Rule:
    - Ignore magenta on the left; if an orange-majority row or column appears in the left half, reconstruct that entire row or column as orange wall.
    - Count the 4-connected black regions fully enclosed by the reconstructed orange walls to get N in {1,2,3}.
    - On the right, count non-black pixels and take the unique most frequent color C.
    - Use the pair (N, C) as a lookup key and return the corresponding 3x3 black grid with teal cells at the encoded positions.

    Key insight:
    - Neither half is sufficient alone: the left half supplies a geometric count after reconstruction, while the right half supplies a color-statistics key.
    """
    left, right = _split_input(np.asarray(grid, dtype=int))
    key = (_left_room_count(left), _dominant_color(right))
    assert key in OUTPUT_MAP, "expected a valid (room_count, dominant_color) key"
    return _template_for_key(key)


def _positive_partition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_wall_mask(side: int, rooms: int, orientation: str) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[side - 1, :] = True
    mask[:, 0] = True
    mask[:, side - 1] = True

    if rooms == 1:
        return mask

    black_total = side - (rooms + 1)
    spans = _positive_partition(black_total, rooms)

    if orientation == "vertical":
        pos = 0
        for width in spans[:-1]:
            pos += width + 1
            mask[:, pos] = True
    else:
        pos = 0
        for height in spans[:-1]:
            pos += height + 1
            mask[pos, :] = True
    return mask


def _critical_break_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    side = mask.shape[0]
    candidates: list[tuple[int, int]] = []
    for c in range(1, side - 1):
        if mask[0, c] and not mask[1, c]:
            candidates.append((0, c))
        if mask[side - 1, c] and not mask[side - 2, c]:
            candidates.append((side - 1, c))
    for r in range(1, side - 1):
        if mask[r, 0] and not mask[r, 1]:
            candidates.append((r, 0))
        if mask[r, side - 1] and not mask[r, side - 2]:
            candidates.append((r, side - 1))
    return candidates


def _damage_left(mask: np.ndarray, rooms: int) -> np.ndarray:
    side = mask.shape[0]
    wall_cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    openers = _critical_break_candidates(mask)
    assert openers

    for _ in range(80):
        left = np.where(mask, ORANGE, BLACK).astype(int)

        opener = random.choice(openers)
        left[opener] = MAGENTA

        extra_wall = random.randint(1, min(3, max(1, len(wall_cells) // 6)))
        random.shuffle(wall_cells)
        placed = {opener}
        for r, c in wall_cells:
            if (r, c) in placed:
                continue
            if random.random() < 0.22:
                left[r, c] = MAGENTA
                placed.add((r, c))
            if len(placed) >= extra_wall + 1:
                break

        off_candidates = [tuple(map(int, coord)) for coord in np.argwhere(~mask)]
        off_count = random.randint(1, min(6, len(off_candidates)))
        for r, c in random.sample(off_candidates, off_count):
            left[r, c] = MAGENTA

        if _raw_room_count(left) == rooms:
            continue
        if not np.array_equal(_reconstruct_walls(left), mask):
            continue
        return left

    raise RuntimeError("failed to damage left half without changing the reconstructed meaning")


def _grow_chunk(free: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    h, w = free.shape
    choices = [tuple(map(int, coord)) for coord in np.argwhere(free)]
    if not choices:
        return None
    for _ in range(60):
        start = random.choice(choices)
        cells = {start}
        frontier = {start}
        while len(cells) < size:
            candidates: set[tuple[int, int]] = set()
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and (nr, nc) not in cells:
                        candidates.add((nr, nc))
            if not candidates:
                break
            nxt = random.choice(list(candidates))
            cells.add(nxt)
            frontier.add(nxt)
        if len(cells) == size:
            return list(cells)
    return None


def _place_color_chunks(
    right: np.ndarray,
    occupied: np.ndarray,
    color: int,
    total: int,
    *,
    split_bias: bool,
) -> bool:
    if total <= 0:
        return True

    if split_bias and total >= 4:
        chunks = random.randint(2, min(3, total))
    else:
        chunks = random.randint(1, min(2, total))
    sizes = _positive_partition(total, chunks)

    for size in sizes:
        free = ~occupied
        cells = _grow_chunk(free, size)
        if cells is None:
            return False
        rows, cols = zip(*cells)
        right[rows, cols] = int(color)
        occupied[rows, cols] = True
    return True


def _build_right_half(side: int, dominant: int) -> np.ndarray:
    right = np.zeros((side, side), dtype=int)
    occupied = np.zeros((side, side), dtype=bool)

    other_candidates = [c for c in (BLUE, RED, GREEN) if c != dominant]
    present_candidates = [random.choice(other_candidates)]
    if random.random() < 0.6:
        present_candidates.append([c for c in other_candidates if c not in present_candidates][0])

    dom_count = random.randint(5, min(10, side + 2))
    counts: dict[int, int] = {dominant: dom_count}

    rival = present_candidates[0]
    counts[rival] = random.randint(max(1, dom_count - 3), dom_count - 1)
    if len(present_candidates) == 2:
        counts[present_candidates[1]] = random.randint(1, counts[rival] - 1)

    extras = [4, MAGENTA, ORANGE, TEAL, 9]
    extra_count = random.randint(1, 3)
    random.shuffle(extras)
    for color in extras[:extra_count]:
        upper = min(dom_count - 1, 4)
        counts[color] = random.randint(1, upper)

    items = list(counts.items())
    random.shuffle(items)

    dominant_blob_color = rival
    for color, total in items:
        split_bias = color == dominant or color != dominant_blob_color
        if not _place_color_chunks(right, occupied, color, total, split_bias=split_bias):
            raise RuntimeError("failed to place right-half color chunks")

    return right


def _next_key() -> tuple[int, int]:
    global _KEY_BAG
    if not _KEY_BAG:
        _KEY_BAG = list(OUTPUT_MAP)
        random.shuffle(_KEY_BAG)
    return _KEY_BAG.pop()


def _validate_instance(grid: np.ndarray, key: tuple[int, int], wall_mask: np.ndarray) -> bool:
    left, right = _split_input(grid)
    room_key, color_key = key

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.array_equal(_reconstruct_walls(left), wall_mask):
        return False
    if _left_room_count(left) != room_key:
        return False
    if _raw_room_count(left) == room_key:
        return False

    counts = Counter(int(v) for v in right.ravel() if int(v) != BLACK)
    if _dominant_color(right) != color_key:
        return False
    if color_key not in (BLUE, RED, GREEN):
        return False
    if len([c for c in counts if c in (BLUE, RED, GREEN)]) < 2:
        return False
    if counts[color_key] <= max(v for c, v in counts.items() if c != color_key):
        return False

    return np.array_equal(main(grid), _template_for_key(key))


def generate_input() -> np.ndarray:
    for _ in range(200):
        key = _next_key()
        rooms, dominant = key
        side_choices = [side for side in (7, 9, 11) if side >= 2 * rooms + 3]
        side = random.choice(side_choices)
        orientation = random.choice(("vertical", "horizontal"))

        wall_mask = _build_wall_mask(side, rooms, orientation)
        left = _damage_left(wall_mask, rooms)

        try:
            right = _build_right_half(side, dominant)
        except RuntimeError:
            _KEY_BAG.append(key)
            continue

        grid = np.zeros((side, side * 2 + 1), dtype=int)
        grid[:, :side] = left
        grid[:, side] = GREY
        grid[:, side + 1 :] = right

        if _validate_instance(grid, key, wall_mask):
            return grid
        _KEY_BAG.append(key)

    raise RuntimeError("failed to generate a valid input")


assert len({tuple(sorted(v)) for v in OUTPUT_MAP.values()}) == 9
