from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
GRID_MIN = 18
GRID_MAX = 30
BOARD_MIN = 10
BOARD_MAX = 16
NOTCH_COUNT_CHOICES = (2, 3, 4)
MAX_SHAPE_AREA = 6
MAX_SHAPE_SPAN = 4
GEN_RETRIES = 500

_RNG = random.Random()
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_EDGES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _rotate_shape(shape: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    cur = shape
    for _ in range(k % 4):
        h, _w = _shape_dims(cur)
        cur = _normalize_coords((c, h - 1 - r) for r, c in cur)
    return cur


def _canonical_shape_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    variants = []
    cur = _normalize_coords(shape)
    for _ in range(4):
        variants.append(cur)
        cur = _rotate_shape(cur, 1)
    return min(variants)


def _unique_rotation_count(shape: tuple[tuple[int, int], ...]) -> int:
    return len({_rotate_shape(shape, k) for k in range(4)})


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _random_polyomino() -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        area = _RNG.randint(3, MAX_SHAPE_AREA)
        cells = {(0, 0)}
        for _ in range(area * 6):
            if len(cells) >= area:
                break
            br, bc = _RNG.choice(tuple(cells))
            dr, dc = _RNG.choice(_DIR4)
            cells.add((br + dr, bc + dc))
        if len(cells) != area:
            continue
        shape = _normalize_coords(cells)
        h, w = _shape_dims(shape)
        if h > MAX_SHAPE_SPAN or w > MAX_SHAPE_SPAN:
            continue
        if h < 2 or w < 2:
            continue
        if _unique_rotation_count(shape) < 2:
            continue
        return shape
    return None


def _shape_to_array(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    arr = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        arr[r, c] = int(color)
    return arr


def _translate_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in shape]


def _find_piece_positions(occ: np.ndarray, obj_h: int, obj_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - obj_h):
        for left in range(1, w - obj_w):
            if occ[top - 1 : top + obj_h + 1, left - 1 : left + obj_w + 1].any():
                continue
            positions.append((top, left))
    return positions


def _choose_notch_anchor(
    board_h: int,
    board_w: int,
    edge: str,
    shape: tuple[tuple[int, int], ...],
    notch_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    sh, sw = _shape_dims(shape)
    if edge in ("top", "bottom"):
        if board_w - sw <= 1:
            return None
        row = 0 if edge == "top" else board_h - sh
        anchors = [(row, col) for col in range(1, board_w - sw)]
    else:
        if board_h - sh <= 1:
            return None
        col = 0 if edge == "left" else board_w - sw
        anchors = [(row, col) for row in range(1, board_h - sh)]
    _RNG.shuffle(anchors)

    for ar, ac in anchors:
        coords = _translate_shape(shape, ar, ac)
        if any(cell in notch_cells for cell in coords):
            continue
        touching = False
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in notch_cells:
                    touching = True
                    break
            if touching:
                break
        if touching:
            continue
        return coords
    return None


def _place_notches(
    board_h: int,
    board_w: int,
    shape_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    notch_cells: set[tuple[int, int]] = set()
    placements: list[dict[str, object] | None] = [None] * len(shape_specs)
    order = sorted(
        range(len(shape_specs)),
        key=lambda i: (-len(shape_specs[i]["shape"]), -max(_shape_dims(shape_specs[i]["shape"]))),
    )
    for idx in order:
        spec = shape_specs[idx]
        coords = _choose_notch_anchor(
            board_h,
            board_w,
            str(spec["edge"]),
            spec["shape"],
            notch_cells,
        )
        if coords is None:
            return None
        placements[idx] = {
            "shape": spec["shape"],
            "edge": spec["edge"],
            "key": spec["key"],
            "coords": coords,
        }
        notch_cells.update(coords)

    board_mask = np.ones((board_h, board_w), dtype=bool)
    for r, c in notch_cells:
        board_mask[r, c] = False
    if len(_components_mask(board_mask)) != 1:
        return None

    notches = _components_mask(~board_mask)
    if len(notches) != len(shape_specs):
        return None
    keys = [_canonical_shape_key(_normalize_coords(comp)) for comp in notches]
    if len(set(keys)) != len(keys):
        return None

    return [p for p in placements if p is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the scattered fragments into the matching notches of the large grey board.

    Input structure:
    - Black background.
    - One large monochrome grey board whose bounding box is a rectangle, except that several connected black notch regions are carved inward from its border.
    - Several monochrome colored fragments lie outside the board; each fragment shape matches exactly one notch up to 0/90/180/270 degree rotation.

    Rule:
    - Identify the grey board as the largest monochrome foreground component and use its bounding box as the completed outer rectangle.
    - Inside that bounding box, collect every missing connected notch region.
    - Match each outside fragment to the unique notch with the same shape under rotation, remove the outside fragment, and fill the notch with that fragment's color.
    - Leave the rest of the grey board unchanged and keep the background black.

    Key insight:
    The black cutouts inside the board are the negative shapes of the outside fragments, so solving the puzzle is a rotation-invariant shape-matching task between missing cavities and scattered pieces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0

    bg = _border_bg(g)

    color_components: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) != int(bg)):
        for comp in _components_mask(g == int(color)):
            color_components.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "bbox": _bbox(comp),
                }
            )
    assert color_components, "expected a board and fragments"

    board = max(color_components, key=lambda d: len(d["coords"]))
    board_color = int(board["color"])
    board_mask = np.zeros((h, w), dtype=bool)
    for r, c in board["coords"]:
        board_mask[r, c] = True
    r0, c0, r1, c1 = board["bbox"]

    notch_mask = ~board_mask[r0 : r1 + 1, c0 : c1 + 1]
    notch_components = _components_mask(notch_mask)
    assert notch_components, "expected at least one notch"

    pieces_mask = (g != int(bg)) & (~board_mask)
    piece_components = _components_mask(pieces_mask)
    assert len(piece_components) == len(notch_components), "piece count must match notch count"

    notches_by_key: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = {}
    for comp in notch_components:
        norm = _normalize_coords(comp)
        key = _canonical_shape_key(norm)
        global_coords = [(r0 + r, c0 + c) for r, c in comp]
        notches_by_key.setdefault(key, []).append(
            {
                "coords": global_coords,
                "bbox": _bbox(global_coords),
            }
        )

    pieces_by_key: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = {}
    for comp in piece_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = next(iter(colors))
        norm = _normalize_coords(comp)
        key = _canonical_shape_key(norm)
        pieces_by_key.setdefault(key, []).append(
            {
                "coords": comp,
                "bbox": _bbox(comp),
                "color": int(color),
            }
        )

    assert set(notches_by_key) == set(pieces_by_key), "fragment shapes must match notch shapes"

    out = np.full_like(g, int(bg))
    out[board_mask] = int(board_color)

    for key in sorted(notches_by_key):
        notch_group = sorted(notches_by_key[key], key=lambda d: d["bbox"])
        piece_group = sorted(pieces_by_key[key], key=lambda d: d["bbox"])
        assert len(notch_group) == len(piece_group), "shape multiplicities must agree"
        for notch, piece in zip(notch_group, piece_group):
            for r, c in notch["coords"]:
                out[r, c] = int(piece["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(GRID_MIN, GRID_MAX)
    width = _RNG.randint(GRID_MIN, GRID_MAX)

    max_board_h = min(BOARD_MAX, height - 4)
    max_board_w = min(BOARD_MAX, width - 4)
    if max_board_h < BOARD_MIN or max_board_w < BOARD_MIN:
        return None

    board_h = _RNG.randint(BOARD_MIN, max_board_h)
    board_w = _RNG.randint(BOARD_MIN, max_board_w)
    board_top = _RNG.randint(2, height - board_h - 2)
    board_left = _RNG.randint(2, width - board_w - 2)

    notch_count = _RNG.choice(NOTCH_COUNT_CHOICES)
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    while len(shapes) < notch_count:
        shape = _random_polyomino()
        if shape is None:
            return None
        key = _canonical_shape_key(shape)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        shapes.append(shape)

    shape_specs: list[dict[str, object]] = []
    for shape in shapes:
        shape_specs.append(
            {
                "key": _canonical_shape_key(shape),
                "shape": _rotate_shape(shape, _RNG.randrange(4)),
                "edge": _RNG.choice(_EDGES),
            }
        )

    notch_specs = _place_notches(board_h, board_w, shape_specs)
    if notch_specs is None:
        return None

    colors = [c for c in range(1, 10) if c != GREY]
    if len(colors) < notch_count:
        return None
    piece_colors = _RNG.sample(colors, notch_count)

    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    inp[board_top : board_top + board_h, board_left : board_left + board_w] = GREY
    expected[board_top : board_top + board_h, board_left : board_left + board_w] = GREY

    piece_specs: list[dict[str, object]] = []
    for notch_spec, color, base_shape in zip(notch_specs, piece_colors, shapes):
        for r, c in notch_spec["coords"]:
            inp[board_top + r, board_left + c] = BG
            expected[board_top + r, board_left + c] = int(color)
        piece_specs.append(
            {
                "shape": _rotate_shape(base_shape, _RNG.randrange(4)),
                "color": int(color),
            }
        )

    occ = np.zeros((height, width), dtype=bool)
    occ[
        max(0, board_top - 1) : min(height, board_top + board_h + 1),
        max(0, board_left - 1) : min(width, board_left + board_w + 1),
    ] = True

    piece_specs.sort(key=lambda d: -len(d["shape"]))
    for spec in piece_specs:
        arr = _shape_to_array(spec["shape"], int(spec["color"]))
        ph, pw = arr.shape
        positions = _find_piece_positions(occ, ph, pw)
        if not positions:
            return None
        top, left = _RNG.choice(positions)
        mask = arr != BG
        inp[top : top + ph, left : left + pw][mask] = arr[mask]
        occ[
            max(0, top - 1) : min(height, top + ph + 1),
            max(0, left - 1) : min(width, left + pw + 1),
        ] = True

    if int(_border_bg(inp)) != BG:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-matching sample")
