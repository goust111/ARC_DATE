import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

SPRITE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE)
MAX_GENERATION_ATTEMPTS = 400
_VALID_CELL_CACHE: dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]] = {}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return sprite


def _sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return sprite.shape, tuple(int(x) for x in sprite.ravel())


def _duplicate_template(grid: np.ndarray) -> np.ndarray:
    sprite_mask = (grid != BLACK) & (grid != GREY)
    components = _components_from_mask(sprite_mask)
    assert 5 <= len(components) <= 7

    counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    sprites: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for coords in components:
        sprite = _crop_component(grid, coords)
        key = _sprite_key(sprite)
        counts[key] = counts.get(key, 0) + 1
        sprites[key] = sprite

    duplicate_keys = [key for key, count in counts.items() if count == 2]
    assert len(duplicate_keys) == 1
    return sprites[duplicate_keys[0]].copy()


def _marker_square(grid: np.ndarray):
    markers = np.argwhere(grid == GREY)
    assert markers.shape == (4, 2)
    r0, c0 = markers.min(axis=0)
    r1, c1 = markers.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 == c1 - c0
    assert r1 - r0 >= 6

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    marker_set = {tuple(map(int, rc)) for rc in markers.tolist()}
    present_corners = corners & marker_set
    assert len(present_corners) == 3
    missing = next(iter(corners - marker_set))

    distractors = marker_set - corners
    assert len(distractors) == 1
    dr, dc = next(iter(distractors))
    assert r0 < dr < r1 and c0 < dc < c1

    adjacent = [corner for corner in corners if corner != missing and (corner[0] == missing[0] or corner[1] == missing[1])]
    assert len(adjacent) == 2
    return (r0, c0, r1, c1), missing, adjacent


def _draw_leg(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        out[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        out[lo : hi + 1, c0] = color


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = out[top : top + h, left : left + w]
    mask = patch != BLACK
    region[mask] = patch[mask]


def _connected_cell_set(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cell_set)


def _line_mask(shape: tuple[int, int], missing: tuple[int, int], adjacent: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for corner in adjacent:
        if corner[0] == missing[0]:
            lo, hi = sorted((missing[1], corner[1]))
            mask[missing[0], lo : hi + 1] = True
        else:
            lo, hi = sorted((missing[0], corner[0]))
            mask[lo : hi + 1, missing[1]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Select the repeated sprite crop, upscale it 2x, and stamp it at the grey square's missing corner while drawing the teal legs.

    Input structure:
    - The grid has a black background with 5-7 separated multicolor 4-connected sprites made of non-black, non-grey cells.
    - Exactly one sprite crop, taken from its tight bounding box and keeping internal black cells, appears exactly twice.
    - There are exactly four grey marker pixels: three corners of an axis-aligned square and one extra grey pixel strictly inside that square.

    Rule:
    - Crop every non-grey sprite to its tight bounding box and compare the full colored crops, including black holes.
    - Find the unique crop that occurs twice, enlarge it by a factor of 2 using pixel replication, and place its top-left corner at the missing square corner.
    - Draw two teal axis-aligned segments from the missing corner to the two adjacent marked corners, let the enlarged sprite overwrite teal where they overlap, and finally keep all grey marker pixels grey.

    Key insight:
    - The output depends on three separate deductions that must be combined: exact crop repetition selects the template, the missing square corner gives the placement anchor, and the teal legs are drawn in the same area where the enlarged stamp may partially cover them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template = _duplicate_template(g)
    scaled = np.kron(template, np.ones((2, 2), dtype=int))

    _square, missing, adjacent = _marker_square(g)
    top, left = missing
    h, w = scaled.shape
    assert top + h <= g.shape[0] and left + w <= g.shape[1]

    out = np.zeros_like(g)
    for corner in adjacent:
        _draw_leg(out, missing, corner, TEAL)
    _stamp(out, scaled, top, left)
    out[g == GREY] = GREY
    return out


def _random_connected_cells(h: int, w: int, k: int) -> set[tuple[int, int]]:
    key = (h, w, k)
    if key not in _VALID_CELL_CACHE:
        all_cells = [(r, c) for r in range(h) for c in range(w)]
        valid = []
        for combo in itertools.combinations(all_cells, k):
            rows = [r for r, _ in combo]
            cols = [c for _, c in combo]
            if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                continue
            if _connected_cell_set(combo):
                valid.append(combo)
        _VALID_CELL_CACHE[key] = valid
    if not _VALID_CELL_CACHE[key]:
        raise RuntimeError("no tight connected sprite exists for requested size")
    return set(random.choice(_VALID_CELL_CACHE[key]))


def _random_sprite(board_palette: tuple[int, ...], used: set[tuple[tuple[int, int], tuple[int, ...]]]) -> np.ndarray:
    for _ in range(400):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        min_cells = max(3, h + w - 2)
        max_cells = min(6, h * w)
        if min_cells > max_cells:
            continue
        k = random.randint(min_cells, max_cells)
        try:
            cells = _random_connected_cells(h, w, k)
        except RuntimeError:
            continue
        sprite = np.zeros((h, w), dtype=int)
        n_colors = random.randint(2, min(4, len(board_palette), k))
        colors = random.sample(board_palette, n_colors)
        coords = list(cells)
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if idx < n_colors:
                sprite[r, c] = colors[idx]
            else:
                sprite[r, c] = random.choice(colors)
        key = _sprite_key(sprite)
        if key not in used:
            return sprite
    raise RuntimeError("failed to generate a unique sprite")


def _candidate_positions(blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = blocked.shape
    candidates = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(H, r + h + 1)
            cc1 = min(W, c + w + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    return candidates


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    H, W = blocked.shape
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(H, r + h + 1)
    cc1 = min(W, c + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    region = grid[top : top + h, left : left + w]
    mask = sprite != BLACK
    region[mask] = sprite[mask]


def _marker_bounds_for_corner(
    H: int, W: int, side: int, stamp_h: int, stamp_w: int, corner_name: str
) -> tuple[range, range] | None:
    if corner_name == "tl":
        r_max = min(H - side - 1, H - stamp_h)
        c_max = min(W - side - 1, W - stamp_w)
    elif corner_name == "tr":
        r_max = min(H - side - 1, H - stamp_h)
        c_max = min(W - side - 1, W - side - stamp_w)
    elif corner_name == "bl":
        r_max = min(H - side - 1, H - side - stamp_h)
        c_max = min(W - side - 1, W - stamp_w)
    else:
        r_max = min(H - side - 1, H - side - stamp_h)
        c_max = min(W - side - 1, W - side - stamp_w)
    if r_max < 0 or c_max < 0:
        return None
    return range(0, r_max + 1), range(0, c_max + 1)


def _place_markers(H: int, W: int, stamp_h: int, stamp_w: int):
    choices = [("tl", 3), ("tr", 2), ("bl", 2), ("br", 3)]
    corner_name = random.choices([name for name, _ in choices], weights=[wt for _, wt in choices], k=1)[0]
    max_side = min(10, H - 2, W - 2)
    min_side = max(6, stamp_h, stamp_w)
    if min_side > max_side:
        return None
    side = random.randint(min_side, max_side)
    bounds = _marker_bounds_for_corner(H, W, side, stamp_h, stamp_w, corner_name)
    if bounds is None:
        return None
    row_range, col_range = bounds
    if len(row_range) == 0 or len(col_range) == 0:
        return None
    r0 = random.choice(list(row_range))
    c0 = random.choice(list(col_range))
    r1, c1 = r0 + side, c0 + side

    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    missing = corners[corner_name]
    present = [pt for name, pt in corners.items() if name != corner_name]

    inner_candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if r != missing[0] and c != missing[1]
    ]
    if not inner_candidates:
        return None
    distractor = random.choice(inner_candidates)
    return (r0, c0, r1, c1), missing, present, distractor


def _build_input() -> np.ndarray | None:
    H = random.randint(18, 24)
    W = random.randint(18, 24)
    grid = np.zeros((H, W), dtype=int)

    board_palette = tuple(random.sample(SPRITE_COLORS, random.randint(4, len(SPRITE_COLORS))))
    used_keys: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    duplicate = _random_sprite(board_palette, used_keys)
    used_keys.add(_sprite_key(duplicate))

    sprite_total = random.randint(5, 7)
    sprites = [duplicate.copy(), duplicate.copy()]
    while len(sprites) < sprite_total:
        sprite = _random_sprite(board_palette, used_keys)
        used_keys.add(_sprite_key(sprite))
        sprites.append(sprite)

    random.shuffle(sprites)
    scaled = np.kron(duplicate, np.ones((2, 2), dtype=int))
    placed_markers = _place_markers(H, W, scaled.shape[0], scaled.shape[1])
    if placed_markers is None:
        return None
    square, _missing, present, distractor = placed_markers
    r0, c0, r1, c1 = square

    blocked = np.zeros((H, W), dtype=bool)
    _reserve(blocked, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    for r, c in set(present + [distractor]):
        grid[r, c] = GREY

    for sprite in sorted(sprites, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        h, w = sprite.shape
        candidates = _candidate_positions(blocked, h, w)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paste_sprite(grid, sprite, top, left)
        _reserve(blocked, top, left, h, w)

    return grid


def _input_invariants(grid: np.ndarray) -> bool:
    try:
        template = _duplicate_template(grid)
        _square, missing, adjacent = _marker_square(grid)
    except AssertionError:
        return False

    scaled = np.kron(template, np.ones((2, 2), dtype=int))
    top, left = missing
    if top + scaled.shape[0] > grid.shape[0] or left + scaled.shape[1] > grid.shape[1]:
        return False

    sprite_mask = (grid != BLACK) & (grid != GREY)
    for coords in _components_from_mask(sprite_mask):
        sprite = _crop_component(grid, coords)
        if np.count_nonzero(sprite) < 3 or np.count_nonzero(sprite) > 6:
            return False

    line_mask = _line_mask(grid.shape, missing, adjacent)
    stamp_mask = np.zeros_like(grid, dtype=bool)
    stamp_mask[top : top + scaled.shape[0], left : left + scaled.shape[1]] = scaled != BLACK
    if not np.any(line_mask & stamp_mask):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out[grid == GREY] != GREY):
        return False
    if not np.any(out == TEAL):
        return False
    if not np.any((out != BLACK) & (out != GREY) & (out != TEAL)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        if _input_invariants(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
