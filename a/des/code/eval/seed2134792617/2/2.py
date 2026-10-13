from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8

FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- shared helpers ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in FOUR_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = np.asarray(mask, dtype=bool)
    return (tuple(map(int, mask.shape)), mask.tobytes())


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_mask_key(np.rot90(mask, k)) for k in range(4)}


def _rotation_canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_keys(mask))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    cells = np.argwhere(mask)
    for dr, dc in cells:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = map(int, occupied.shape)
    ph, pw = map(int, mask.shape)
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    for dr, dc in np.argwhere(mask):
        r = int(r0 + dr)
        c = int(c0 + dc)
        if r < pad or r >= h - pad or c < pad or c >= w - pad:
            return False
        r1 = max(0, r - pad)
        r2 = min(h, r + pad + 1)
        c1 = max(0, c - pad)
        c2 = min(w, c + pad + 1)
        if bool(np.any(occupied[r1:r2, c1:c2])):
            return False
    return True


def _place_masks(canvas_shape: tuple[int, int], masks: list[np.ndarray], pad: int = 1) -> list[tuple[int, int]] | None:
    h, w = map(int, canvas_shape)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(
        range(len(masks)),
        key=lambda i: (-(masks[i].shape[0] * masks[i].shape[1]), -int(masks[i].sum())),
    )

    for idx in order:
        mask = masks[idx]
        ph, pw = map(int, mask.shape)
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            if not _can_place_mask(occupied, mask, int(r0), int(c0), pad=pad):
                continue
            _paint_mask(occupied, mask, int(r0), int(c0), True)
            placements[idx] = (int(r0), int(c0))
            break
        if placements[idx] is None:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored key to the matching lock hole in the board, then fill the hole with the key color.

    Input structure:
    - Background is black.
    - A full grey horizontal row separates the top keys from the lower board.
    - The lower region contains one azure rectangle with black hole components inside it.
    - Each top key is a connected monochrome shape, and each hole matches one key up to rotation.

    Rule:
    - Remove the top keys.
    - Match each key to the unique hole with the same shape after a 0/90/180/270 rotation.
    - Fill that hole with the key color.
    - Leave the separator row and non-hole board cells unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "grid must be a non-empty 2D array"

    sep_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert sep_rows.size == 1, "expected one full grey separator row"
    sep = int(sep_rows[0])
    assert 0 < sep < g.shape[0] - 1, "separator must split the grid into two non-empty regions"

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]

    board_pixels = np.argwhere(bottom == AZURE)
    assert board_pixels.size > 0, "expected an azure board below the separator"
    br0, bc0 = board_pixels.min(axis=0)
    br1, bc1 = board_pixels.max(axis=0)
    br0 = int(br0 + sep + 1)
    br1 = int(br1 + sep + 1)
    bc0 = int(bc0)
    bc1 = int(bc1)

    board = g[br0 : br1 + 1, bc0 : bc1 + 1]
    assert np.all((board == AZURE) | (board == BLACK)), "board may only contain azure cells and black holes"
    assert np.all(board[0, :] == AZURE) and np.all(board[-1, :] == AZURE)
    assert np.all(board[:, 0] == AZURE) and np.all(board[:, -1] == AZURE)

    keys = []
    for coords in _components(top != BLACK):
        colors = np.unique(top[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each key must be monochrome"
        keys.append(
            {
                "coords": coords.copy(),
                "mask": _tight_mask(coords),
                "color": int(colors[0]),
                "rot_keys": _rotation_keys(_tight_mask(coords)),
            }
        )
    assert keys, "expected at least one key above the separator"

    holes = []
    for coords in _components(board == BLACK):
        assert np.all(coords[:, 0] > 0) and np.all(coords[:, 0] < board.shape[0] - 1)
        assert np.all(coords[:, 1] > 0) and np.all(coords[:, 1] < board.shape[1] - 1)
        holes.append(
            {
                "coords": coords + np.array([br0, bc0], dtype=int),
                "mask": _tight_mask(coords),
                "key": _mask_key(_tight_mask(coords)),
            }
        )

    assert len(keys) == len(holes), "key count must match hole count"

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = BLACK

    used_holes: set[int] = set()
    for key in sorted(keys, key=lambda item: (-int(item["mask"].sum()), item["color"])):
        candidates = [i for i, hole in enumerate(holes) if i not in used_holes and hole["key"] in key["rot_keys"]]
        assert len(candidates) == 1, "each key must match exactly one hole"
        hole = holes[candidates[0]]
        used_holes.add(candidates[0])
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(key["color"])

    return out


# --------------- generator ---------------


KEY_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
NUM_KEYS_RANGE = (2, 5)
KEY_AREA_RANGE = (3, 6)

GEN_MAX_ATTEMPTS = 400
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 120

_ROTATION_QUEUE: list[int] = []


def _random_connected_shape(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in FOUR_DIRS:
                    cell = (r + dr, c + dc)
                    if cell not in cells:
                        frontier.append(cell)
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        norm = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[int(r), int(c)] = True

        if len(_rotation_keys(mask)) != 4:
            continue
        return mask

    return None


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for _ in range(SHAPE_RETRIES * count * 2):
        area = random.randint(*KEY_AREA_RANGE)
        mask = _random_connected_shape(area)
        if mask is None:
            continue
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        canon = _rotation_canonical_key(mask)
        if canon in seen:
            continue
        seen.add(canon)
        shapes.append(mask.copy())
        if len(shapes) == count:
            return shapes

    return None


def _balanced_rotations(count: int) -> list[int]:
    global _ROTATION_QUEUE
    out: list[int] = []
    while len(out) < count:
        if not _ROTATION_QUEUE:
            _ROTATION_QUEUE = list(range(4))
            random.shuffle(_ROTATION_QUEUE)
        out.append(int(_ROTATION_QUEUE.pop()))
    random.shuffle(out)
    return out


def _board_shape_range(masks: list[np.ndarray]) -> tuple[range, range]:
    max_h = max(int(mask.shape[0]) for mask in masks)
    max_w = max(int(mask.shape[1]) for mask in masks)
    total_area = sum(int(mask.sum()) for mask in masks)
    density = int(np.ceil(np.sqrt(total_area))) + len(masks)
    h_lo = max(max_h + 2, density)
    w_lo = max(max_w + 2, density + 1)
    h_hi = min(18, h_lo + 5)
    w_hi = min(20, w_lo + 6)
    return range(h_lo, h_hi + 1), range(w_lo, w_hi + 1)


def _place_holes(masks: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    heights, widths = _board_shape_range(masks)
    options = [(h, w) for h in heights for w in widths]
    random.shuffle(options)
    for board_h, board_w in options[:PLACEMENT_RETRIES]:
        placements = _place_masks((int(board_h), int(board_w)), masks, pad=1)
        if placements is not None:
            return int(board_h), int(board_w), placements
    return None


def _grid_width(board_w: int, key_masks: list[np.ndarray]) -> int | None:
    key_width_hint = sum(int(mask.shape[1]) for mask in key_masks) + len(key_masks) + 2
    width_lo = max(board_w + 2, key_width_hint)
    width_hi = min(30, width_lo + 8)
    if width_lo > width_hi:
        return None
    return random.randint(width_lo, width_hi)


def _top_height(count: int, key_masks: list[np.ndarray]) -> int:
    max_h = max(int(mask.shape[0]) for mask in key_masks)
    return random.randint(max(max_h + 2, 6), min(12, max_h + 4 + (1 if count >= 4 else 0)))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*NUM_KEYS_RANGE)
    shapes = _sample_unique_shapes(count)
    if shapes is None:
        return None

    key_colors = random.sample(KEY_COLOR_POOL, count)
    hole_rotations = _balanced_rotations(count)

    key_masks = []
    hole_masks = []
    for shape, hole_rot in zip(shapes, hole_rotations, strict=False):
        key_mask = np.rot90(shape, random.randrange(4))
        key_masks.append(key_mask.copy())
        hole_masks.append(np.rot90(key_mask, int(hole_rot)).copy())

    hole_layout = _place_holes(hole_masks)
    if hole_layout is None:
        return None
    board_h, board_w, hole_positions = hole_layout

    grid_w = _grid_width(board_w, key_masks)
    if grid_w is None:
        return None

    top_h = _top_height(count, key_masks)
    key_positions = _place_masks((top_h, grid_w), key_masks, pad=1)
    if key_positions is None:
        return None

    bottom_h = board_h + random.randint(2, 4)
    total_h = top_h + 1 + bottom_h
    if total_h > 30:
        return None

    board_c0 = random.randint(1, grid_w - board_w - 1)
    board_r0 = top_h + 1 + random.randint(1, bottom_h - board_h - 1)

    inp = np.full((total_h, grid_w), BLACK, dtype=int)
    inp[top_h, :] = GREY
    inp[board_r0 : board_r0 + board_h, board_c0 : board_c0 + board_w] = AZURE

    expected = inp.copy()

    for color, key_mask, key_pos, hole_mask, hole_pos in zip(
        key_colors, key_masks, key_positions, hole_masks, hole_positions, strict=False
    ):
        kr, kc = key_pos
        hr, hc = hole_pos
        _paint_mask(inp, key_mask, int(kr), int(kc), int(color))
        _paint_mask(inp, hole_mask, int(board_r0 + hr), int(board_c0 + hc), BLACK)

        _paint_mask(expected, key_mask, int(kr), int(kc), BLACK)
        _paint_mask(expected, hole_mask, int(board_r0 + hr), int(board_c0 + hc), int(color))

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-and-lock ARC input")
