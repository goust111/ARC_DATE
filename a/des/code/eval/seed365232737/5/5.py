from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROTATION_BY_DIRECTION = {
    "right": 0,
    "up": 3,
    "left": 2,
    "down": 1,
}

# Generator palette. The solver infers these roles from structure.
YELLOW = 4
GREY = 5
GEN_BG_POOL = [0, 1, 2, 3, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
PLACE_TRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _extract_locks(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], int]:
    colors = set(map(int, grid.ravel())) - {bg}
    frame_candidates: list[dict[str, object]] = []

    for color in sorted(colors):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(comp) != _rect_border_cells((r0, c0, r1, c1)):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            interior_colors = set(map(int, np.unique(interior)))
            if len(interior_colors) != 1:
                continue
            cover = next(iter(interior_colors))
            if cover == color:
                continue
            frame_candidates.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "cover": int(cover),
                    "inner_size": int(r1 - r0 - 1),
                }
            )

    assert frame_candidates, "expected at least one lock frame"
    cover_counts = Counter(int(lock["cover"]) for lock in frame_candidates)
    cover = int(cover_counts.most_common(1)[0][0])
    locks = [lock for lock in frame_candidates if int(lock["cover"]) == cover]
    assert len(locks) >= 2, "expected multiple locks"
    assert len({int(lock["color"]) for lock in locks}) == len(locks), "lock colors must be unique"

    sizes = {int(lock["inner_size"]) for lock in locks}
    assert len(sizes) == 1, "all locks must share one interior size"
    return locks, cover


def _parse_key(grid: np.ndarray, comp: list[tuple[int, int]], lock_colors: set[int]) -> dict[str, object] | None:
    color_set = {int(grid[r, c]) for r, c in comp}
    if len(color_set) != 2:
        return None
    shaft_candidates = [color for color in color_set if color in lock_colors]
    if len(shaft_candidates) != 1:
        return None
    shaft_color = int(shaft_candidates[0])
    yellow = next(color for color in color_set if color != shaft_color)
    shaft = [(r, c) for r, c in comp if int(grid[r, c]) == shaft_color]
    teeth = [(r, c) for r, c in comp if int(grid[r, c]) == yellow]
    if len(shaft) != 3 or len(teeth) not in (1, 2):
        return None

    rows = sorted({r for r, _c in shaft})
    cols = sorted({c for _r, c in shaft})
    direction: str
    endpoint: tuple[int, int]

    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        row = rows[0]
        left = (row, cols[0])
        right = (row, cols[2])
        left_touch = [cell for cell in teeth if abs(cell[0] - left[0]) + abs(cell[1] - left[1]) == 1]
        right_touch = [cell for cell in teeth if abs(cell[0] - right[0]) + abs(cell[1] - right[1]) == 1]
        if bool(left_touch) == bool(right_touch):
            return None
        if right_touch:
            direction = "right"
            endpoint = right
        else:
            direction = "left"
            endpoint = left
        for tr, tc in teeth:
            if tc != endpoint[1] or abs(tr - endpoint[0]) != 1:
                return None
        if len(teeth) == 2 and {tr - endpoint[0] for tr, _tc in teeth} != {-1, 1}:
            return None
    elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        col = cols[0]
        top = (rows[0], col)
        bottom = (rows[2], col)
        top_touch = [cell for cell in teeth if abs(cell[0] - top[0]) + abs(cell[1] - top[1]) == 1]
        bottom_touch = [cell for cell in teeth if abs(cell[0] - bottom[0]) + abs(cell[1] - bottom[1]) == 1]
        if bool(top_touch) == bool(bottom_touch):
            return None
        if top_touch:
            direction = "up"
            endpoint = top
        else:
            direction = "down"
            endpoint = bottom
        for tr, tc in teeth:
            if tr != endpoint[0] or abs(tc - endpoint[1]) != 1:
                return None
        if len(teeth) == 2 and {tc - endpoint[1] for _tr, tc in teeth} != {-1, 1}:
            return None
    else:
        return None

    return {
        "shaft_color": shaft_color,
        "yellow": int(yellow),
        "direction": direction,
        "coords": [(int(r), int(c)) for r, c in comp],
    }


def _extract_keys(
    grid: np.ndarray,
    bg: int,
    cover: int,
    lock_colors: set[int],
) -> tuple[dict[int, dict[str, object]], int]:
    mask = (grid != bg) & (grid != cover)
    keys: dict[int, dict[str, object]] = {}
    yellow_colors: set[int] = set()
    for comp in _components(mask):
        parsed = _parse_key(grid, comp, lock_colors)
        if parsed is None:
            continue
        shaft_color = int(parsed["shaft_color"])
        assert shaft_color not in keys, "each lock color must have one key"
        keys[shaft_color] = parsed
        yellow_colors.add(int(parsed["yellow"]))

    assert set(keys) == lock_colors, "every lock must have a matching key"
    assert len(yellow_colors) == 1, "all key teeth must share one color"
    return keys, int(next(iter(yellow_colors)))


def _corner_codebook_candidates(shape: tuple[int, int], count: int, size: int) -> list[list[tuple[int, int]]]:
    h, w = shape
    total = count * size + (count - 1)
    candidates: list[list[tuple[int, int]]] = []

    if total <= w:
        candidates.append([(0, i * (size + 1)) for i in range(count)])
        candidates.append([(0, w - total + i * (size + 1)) for i in range(count)])
        candidates.append([(h - size, i * (size + 1)) for i in range(count)])
        candidates.append([(h - size, w - total + i * (size + 1)) for i in range(count)])

    if total <= h:
        candidates.append([(i * (size + 1), 0) for i in range(count)])
        candidates.append([(i * (size + 1), w - size) for i in range(count)])
        candidates.append([(h - total + i * (size + 1), 0) for i in range(count)])
        candidates.append([(h - total + i * (size + 1), w - size) for i in range(count)])

    return candidates


def _read_codebook_candidate(
    grid: np.ndarray,
    bg: int,
    lock_colors: set[int],
    size: int,
    tiles: list[tuple[int, int]],
) -> dict[int, np.ndarray] | None:
    patterns: dict[int, np.ndarray] = {}
    horizontal = len({r for r, _c in tiles}) == 1

    for idx, (r0, c0) in enumerate(tiles):
        patch = grid[r0 : r0 + size, c0 : c0 + size]
        colors = set(map(int, np.unique(patch))) - {bg}
        if len(colors) != 1:
            return None
        color = next(iter(colors))
        if color not in lock_colors or color in patterns:
            return None
        patterns[color] = patch == color

        if idx + 1 < len(tiles):
            nr0, nc0 = tiles[idx + 1]
            if horizontal:
                gap = grid[r0 : r0 + size, c0 + size : nc0]
            else:
                gap = grid[r0 + size : nr0, c0 : c0 + size]
            if gap.size == 0 or not np.all(gap == bg):
                return None

    if set(patterns) != lock_colors:
        return None
    return patterns


def _extract_codebook(
    grid: np.ndarray,
    bg: int,
    lock_colors: set[int],
    size: int,
) -> dict[int, np.ndarray]:
    found: list[dict[int, np.ndarray]] = []
    for tiles in _corner_codebook_candidates(grid.shape, len(lock_colors), size):
        patterns = _read_codebook_candidate(grid, bg, lock_colors, size, tiles)
        if patterns is not None:
            found.append(patterns)
    assert len(found) == 1, "expected one codebook strip in a corner"
    return found[0]


def _rotate_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(mask, ROTATION_BY_DIRECTION[direction]).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock grey-filled rectangular frames using same-color keys and a corner codebook.

    - Lock color identifies which reference sprite to reveal.
    - Key shaft direction selects the sprite rotation.
    - Keys are removed after unlocking.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    locks, cover = _extract_locks(g, bg)
    lock_colors = {int(lock["color"]) for lock in locks}
    size = int(next(iter({int(lock["inner_size"]) for lock in locks})))
    codebook = _extract_codebook(g, bg, lock_colors, size)
    keys, _yellow = _extract_keys(g, bg, cover, lock_colors)

    out = g.copy()
    for key in keys.values():
        for r, c in key["coords"]:
            out[r, c] = bg

    for lock in locks:
        color = int(lock["color"])
        r0, c0, r1, c1 = lock["bbox"]
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1
        patch = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        patch[:] = bg
        rotated = _rotate_mask(codebook[color], str(keys[color]["direction"]))
        assert rotated.shape == patch.shape, "rotated codebook sprite must match the lock interior"
        patch[rotated] = color

    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch >= 0
    region[mask] = patch[mask]


def _draw_lock_patch(size: int, frame_color: int, cover: int) -> np.ndarray:
    patch = np.full((size + 2, size + 2), cover, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _key_patch(frame_color: int, yellow: int, direction: str, tooth_count: int, tooth_side: int) -> np.ndarray:
    shaft: list[tuple[int, int]]
    teeth: list[tuple[int, int]]
    if direction == "right":
        shaft = [(0, 0), (0, 1), (0, 2)]
        teeth = [(-1, 2)] if tooth_count == 1 else [(-1, 2), (1, 2)]
        if tooth_count == 1:
            teeth = [(tooth_side, 2)]
    elif direction == "left":
        shaft = [(0, 0), (0, 1), (0, 2)]
        teeth = [(-1, 0)] if tooth_count == 1 else [(-1, 0), (1, 0)]
        if tooth_count == 1:
            teeth = [(tooth_side, 0)]
    elif direction == "down":
        shaft = [(0, 0), (1, 0), (2, 0)]
        teeth = [(2, -1)] if tooth_count == 1 else [(2, -1), (2, 1)]
        if tooth_count == 1:
            teeth = [(2, tooth_side)]
    else:
        shaft = [(0, 0), (1, 0), (2, 0)]
        teeth = [(0, -1)] if tooth_count == 1 else [(0, -1), (0, 1)]
        if tooth_count == 1:
            teeth = [(0, tooth_side)]

    cells = shaft + teeth
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm_shaft = [(r - min_r, c - min_c) for r, c in shaft]
    norm_teeth = [(r - min_r, c - min_c) for r, c in teeth]
    h = max(r for r, _c in norm_shaft + norm_teeth) + 1
    w = max(c for _r, c in norm_shaft + norm_teeth) + 1
    patch = np.full((h, w), -1, dtype=int)
    for r, c in norm_shaft:
        patch[r, c] = frame_color
    for r, c in norm_teeth:
        patch[r, c] = yellow
    return patch


def _random_pattern(size: int) -> np.ndarray:
    min_area = max(3, size)
    max_area = max(min_area, min(size * size - 1, size + 4))
    for _ in range(200):
        target = random.randint(min_area, max_area)
        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            options = []
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    options.append((nr, nc))
            if options:
                cells.add(random.choice(options))
        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        row_span = int(coords[:, 0].max() - coords[:, 0].min())
        col_span = int(coords[:, 1].max() - coords[:, 1].min())
        if row_span == 0 or col_span == 0:
            continue
        if cells == _rect_border_cells((0, 0, size - 1, size - 1)):
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) < 4:
            continue
        return mask
    raise RuntimeError("failed to sample a distinct asymmetric pattern")


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, gap: int = 1) -> bool:
    rows, cols = blocked.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(rows, top + h + gap)
    c1 = min(cols, left + w + gap)
    return not blocked[r0:r1, c0:c1].any()


def _occupy(blocked: np.ndarray, top: int, left: int, h: int, w: int, gap: int = 1) -> None:
    rows, cols = blocked.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(rows, top + h + gap)
    c1 = min(cols, left + w + gap)
    blocked[r0:r1, c0:c1] = True


def _codebook_tiles(
    rows: int,
    cols: int,
    count: int,
    size: int,
    orientation: str,
    corner: str,
) -> list[tuple[int, int]]:
    total = count * size + (count - 1)
    top = 0 if corner.startswith("top") else rows - (size if orientation == "horizontal" else total)
    left = 0 if corner.endswith("left") else cols - (total if orientation == "horizontal" else size)
    if orientation == "horizontal":
        return [(top, left + i * (size + 1)) for i in range(count)]
    return [(top + i * (size + 1), left) for i in range(count)]


def _sample_patterns(lock_colors: list[int], size: int) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    used: set[bytes] = set()
    for color in lock_colors:
        for _ in range(200):
            mask = _random_pattern(size)
            key = mask.tobytes()
            if key in used:
                continue
            patterns[color] = mask
            used.add(key)
            break
        else:
            raise RuntimeError("failed to sample unique patterns")
    return patterns


def _expected_output(
    inp: np.ndarray,
    bg: int,
    locks: list[dict[str, object]],
    patterns: dict[int, np.ndarray],
    keys: dict[int, dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for key in keys.values():
        for r, c in key["coords"]:
            out[r, c] = bg
    for lock in locks:
        color = int(lock["color"])
        r0, c0, r1, c1 = lock["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[:] = bg
        patch[_rotate_mask(patterns[color], str(keys[color]["direction"]))] = color
    return out


def _generate_case() -> np.ndarray | None:
    lock_count = random.choice((2, 3, 4))
    size = random.randint(3, 5)
    outer = size + 2
    bg = random.choice(GEN_BG_POOL)
    lock_colors = random.sample([c for c in range(10) if c not in {bg, GREY, YELLOW}], lock_count)
    patterns = _sample_patterns(lock_colors, size)

    orientation = random.choice(("horizontal", "vertical"))
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    total = lock_count * size + (lock_count - 1)
    codebook_h = size if orientation == "horizontal" else total
    codebook_w = total if orientation == "horizontal" else size

    rows_min = max(codebook_h + outer + 8, 2 * outer + 10, 20)
    cols_min = max(codebook_w + outer + 8, 2 * outer + 10, 20)
    rows = random.randint(rows_min, rows_min + 8)
    cols = random.randint(cols_min, cols_min + 8)

    grid = np.full((rows, cols), bg, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    codebook_order = lock_colors[:]
    random.shuffle(codebook_order)
    tiles = _codebook_tiles(rows, cols, lock_count, size, orientation, corner)
    for color, (r0, c0) in zip(codebook_order, tiles):
        patch = np.full((size, size), bg, dtype=int)
        patch[patterns[color]] = color
        _paint_patch(grid, patch, r0, c0, bg)
    first_r, first_c = tiles[0]
    last_r, last_c = tiles[-1]
    codebook_bbox_h = (last_r - first_r + size) if orientation == "vertical" else size
    codebook_bbox_w = (last_c - first_c + size) if orientation == "horizontal" else size
    _occupy(blocked, first_r, first_c, codebook_bbox_h, codebook_bbox_w, gap=1)

    locks: list[dict[str, object]] = []
    for color in random.sample(lock_colors, len(lock_colors)):
        placed = False
        for _ in range(PLACE_TRIES):
            r0 = random.randint(0, rows - outer)
            c0 = random.randint(0, cols - outer)
            if not _can_place(blocked, r0, c0, outer, outer, gap=1):
                continue
            _paint_patch(grid, _draw_lock_patch(size, color, GREY), r0, c0, bg)
            _occupy(blocked, r0, c0, outer, outer, gap=1)
            locks.append({"color": color, "bbox": (r0, c0, r0 + outer - 1, c0 + outer - 1)})
            placed = True
            break
        if not placed:
            return None

    directions = ["right", "up", "left", "down"]
    random.shuffle(directions)
    keys: dict[int, dict[str, object]] = {}
    for color, direction in zip(lock_colors, directions[:lock_count]):
        tooth_count = random.choice((1, 2))
        tooth_side = random.choice((-1, 1))
        patch = _key_patch(color, YELLOW, direction, tooth_count, tooth_side)
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACE_TRIES):
            r0 = random.randint(0, rows - ph)
            c0 = random.randint(0, cols - pw)
            if not _can_place(blocked, r0, c0, ph, pw, gap=1):
                continue
            _paint_patch(grid, patch, r0, c0, bg)
            _occupy(blocked, r0, c0, ph, pw, gap=1)
            coords = [(r0 + r, c0 + c) for r, c in zip(*np.where(patch >= 0))]
            keys[color] = {
                "direction": direction,
                "coords": [(int(r), int(c)) for r, c in coords],
            }
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, bg, locks, patterns, keys)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
