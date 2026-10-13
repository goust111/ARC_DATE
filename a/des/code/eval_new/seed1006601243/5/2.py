import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

CELL_SIZE = 3
STEP = CELL_SIZE + 1
GROUP_AREA = 6
GROUP_STEP = 7
MAX_GENERATION_ATTEMPTS = 300


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_mask(mask: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return mask.copy()
    if rotation == 1:
        return np.rot90(mask, -1)
    if rotation == 2:
        return np.rot90(mask, 2)
    if rotation == 3:
        return np.rot90(mask, 1)
    raise ValueError(f"bad rotation code {rotation}")


def _indicator_mask(rotation: int) -> np.ndarray:
    if rotation == 0:
        return np.array([[True, True]], dtype=bool)
    if rotation == 1:
        return np.array([[True, False], [False, True]], dtype=bool)
    if rotation == 2:
        return np.array([[True], [True]], dtype=bool)
    if rotation == 3:
        return np.array([[False, True], [True, False]], dtype=bool)
    raise ValueError(f"bad indicator rotation {rotation}")


def _decode_indicator(coords: np.ndarray) -> int:
    assert coords.shape[0] == 2, "indicator must contain exactly two pixels"
    r0, c0, r1, c1 = _bbox(coords)
    tight = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        tight[int(r - r0), int(c - c0)] = True
    if tight.shape == (1, 2) and bool(tight[0, 0]) and bool(tight[0, 1]):
        return 0
    if tight.shape == (2, 1) and bool(tight[0, 0]) and bool(tight[1, 0]):
        return 2
    if tight.shape == (2, 2) and int(tight.sum()) == 2:
        if bool(tight[0, 0]) and bool(tight[1, 1]):
            return 1
        if bool(tight[0, 1]) and bool(tight[1, 0]):
            return 3
    raise AssertionError("unrecognized rotation indicator")


def _min_chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            dist = max(abs(int(ar) - int(br)), abs(int(ac) - int(bc)))
            if dist < best:
                best = dist
    return int(best)


def _bbox_chebyshev_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    def gap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
        if a1 < b0:
            return b0 - a1
        if b1 < a0:
            return a0 - b1
        return 0

    row_gap = gap_1d(a[0], a[2], b[0], b[2])
    col_gap = gap_1d(a[1], a[3], b[1], b[3])
    return max(row_gap, col_gap)


def _is_board_prefix(g: np.ndarray, n: int) -> bool:
    side = STEP * n + 1
    if side > g.shape[0] or side > g.shape[1]:
        return False

    marker_rows = {2 + STEP * i for i in range(n)}
    marker_cols = {2 + STEP * j for j in range(n)}

    for r in range(side):
        want_grey = r not in marker_rows
        if want_grey:
            if int(g[r, 0]) != GREY:
                return False
        else:
            if int(g[r, 0]) in {BLACK, GREY, MAROON}:
                return False
    for c in range(side):
        want_grey = c not in marker_cols
        if want_grey:
            if int(g[0, c]) != GREY:
                return False
        else:
            if int(g[0, c]) in {BLACK, GREY, MAROON}:
                return False

    for r in range(STEP, side, STEP):
        if not np.all(g[r, :side] == GREY):
            return False
    for c in range(STEP, side, STEP):
        if not np.all(g[:side, c] == GREY):
            return False

    for i in range(n):
        for j in range(n):
            cell = g[1 + STEP * i : 1 + STEP * i + CELL_SIZE, 1 + STEP * j : 1 + STEP * j + CELL_SIZE]
            if cell.shape != (CELL_SIZE, CELL_SIZE):
                return False
            if np.any(cell == GREY) or np.any(cell == MAROON):
                return False
    return True


def _find_board_size(g: np.ndarray) -> int:
    max_n = (min(g.shape) - 1) // STEP
    for n in range(max_n, 0, -1):
        if _is_board_prefix(g, n):
            return n
    raise AssertionError("failed to locate the coded board")


def _outside_mask(shape: tuple[int, int], board_side: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    mask[:board_side, :board_side] = False
    return mask


def _extract_sprites(g: np.ndarray, board_side: int) -> list[dict]:
    outside = _outside_mask(g.shape, board_side)
    color_order: list[int] = []
    for value in g[outside].tolist():
        color = int(value)
        if color in {BLACK, GREY, MAROON} or color in color_order:
            continue
        color_order.append(color)

    sprites: list[dict] = []
    for color in color_order:
        comps = _components((g == color) & outside, connectivity=4)
        assert len(comps) == 1, "each sprite color must form exactly one component"
        coords = comps[0]
        mask = np.zeros_like(g, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        tight = _crop_mask(mask)
        assert tight.shape[0] <= CELL_SIZE and tight.shape[1] <= CELL_SIZE
        sprites.append(
            {
                "color": color,
                "coords": coords,
                "bbox": _bbox(coords),
                "mask": tight,
            }
        )
    return sprites


def _extract_indicators(g: np.ndarray, board_side: int) -> list[dict]:
    outside = _outside_mask(g.shape, board_side)
    comps = _components((g == MAROON) & outside, connectivity=8)
    indicators: list[dict] = []
    for coords in comps:
        indicators.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "rotation": _decode_indicator(coords),
            }
        )
    return indicators


def _sprite_mapping(g: np.ndarray, board_side: int) -> dict[int, tuple[np.ndarray, int]]:
    sprites = _extract_sprites(g, board_side)
    indicators = _extract_indicators(g, board_side)
    assert len(sprites) == len(indicators) >= 1

    unused = list(range(len(indicators)))
    mapping: dict[int, tuple[np.ndarray, int]] = {}
    for sprite in sprites:
        candidates = [
            idx
            for idx in unused
            if _bbox_chebyshev_gap(sprite["bbox"], indicators[idx]["bbox"]) == 1
        ]
        assert len(candidates) == 1, "each sprite must touch exactly one rotation indicator"
        idx = candidates[0]
        unused.remove(idx)
        mapping[int(sprite["color"])] = (sprite["mask"], int(indicators[idx]["rotation"]))
    assert not unused
    return mapping


def _blank_output_canvas(g: np.ndarray, board_side: int) -> np.ndarray:
    out = np.full_like(g, BLACK)
    out[:board_side, :board_side] = g[:board_side, :board_side]
    n = (board_side - 1) // STEP
    for i in range(n):
        for j in range(n):
            out[1 + STEP * i : 1 + STEP * i + CELL_SIZE, 1 + STEP * j : 1 + STEP * j + CELL_SIZE] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome sprites into a coded 3x3-cell board by decoding border colors and maroon rotation markers.

    Input structure:
    - The background is black, and a grey lattice in the upper-left corner splits a square board into N×N cells whose usable interiors are 3×3.
    - The top border contains one colored code pixel centered above each column, and the left border contains one colored code pixel centered beside each row.
    - Outside the board, each scattered sprite is a single-color 4-connected pattern fitting inside 3×3, and each sprite has one adjacent two-pixel maroon indicator.

    Rule:
    - Read the board size from the grey lattice, then read the top colors as column sprite selectors and the left colors as row repaint colors.
    - Decode each maroon indicator as one of four rotations: horizontal = no rotation, main diagonal = clockwise 90, vertical = 180, anti-diagonal = counterclockwise 90.
    - Clear the scattered pieces, keep the board and its code strips, and for every cell whose column color matches a sprite, rotate that sprite accordingly, repaint it with the row color, and place it into the cell's 3×3 interior. Cells whose column color has no sprite stay black.

    Key insight:
    - The top codes choose sprite identity, the left codes choose final color, and the nearby maroon pair independently encodes orientation, so the output requires combining three separate visible cues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    n = _find_board_size(g)
    board_side = STEP * n + 1
    top_colors = [int(g[0, 2 + STEP * j]) for j in range(n)]
    left_colors = [int(g[2 + STEP * i, 0]) for i in range(n)]
    sprites = _sprite_mapping(g, board_side)

    out = _blank_output_canvas(g, board_side)
    for i, row_color in enumerate(left_colors):
        for j, column_color in enumerate(top_colors):
            sprite = sprites.get(column_color)
            if sprite is None:
                continue
            mask, rotation = sprite
            placed = _rotate_mask(mask, rotation)
            r0 = 1 + STEP * i
            c0 = 1 + STEP * j
            cell = out[r0 : r0 + placed.shape[0], c0 : c0 + placed.shape[1]]
            cell[placed] = row_color
    return out


def _build_shape_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    seen: set[bytes] = set()
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        count = int(mask.sum())
        if count < 3 or count > 6:
            continue
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        rotations = [np.rot90(mask, k) for k in range(4)]
        if any(np.array_equal(mask, rotations[k]) for k in range(1, 4)):
            continue
        key = min(rot.tobytes() for rot in rotations)
        if key in seen:
            continue
        seen.add(key)
        library.append(mask.copy())
    assert len(library) >= 8
    return library


SHAPE_LIBRARY = _build_shape_library()


def _sample_row_colors(n: int, palette: list[int]) -> list[int]:
    colors = palette.copy()
    random.shuffle(colors)
    seq = colors[: min(len(colors), n)]
    while len(seq) < n:
        seq.append(random.choice(palette))
    random.shuffle(seq)
    if n > 1 and len(set(seq)) == 1 and len(palette) > 1:
        seq[0] = palette[0]
        seq[1] = palette[1]
        random.shuffle(seq)
    return [int(x) for x in seq]


def _sample_top_colors(n: int, sprite_colors: list[int], dummy_color: int | None) -> list[int]:
    choices = sprite_colors.copy()
    seq = sprite_colors.copy()
    if dummy_color is not None:
        choices.append(dummy_color)
        seq.append(dummy_color)
    while len(seq) < n:
        seq.append(random.choice(choices))
    random.shuffle(seq)
    return [int(x) for x in seq]


def _indicator_positions(sr: int, sc: int, h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for dc in range(CELL_SIZE - w + 1):
        positions.append((sr - h, sc + dc))
        positions.append((sr + CELL_SIZE, sc + dc))
    for dr in range(CELL_SIZE - h + 1):
        positions.append((sr + dr, sc - w))
        positions.append((sr + dr, sc + CELL_SIZE))
    positions.extend(
        [
            (sr - h, sc - w),
            (sr - h, sc + CELL_SIZE),
            (sr + CELL_SIZE, sc - w),
            (sr + CELL_SIZE, sc + CELL_SIZE),
        ]
    )
    dedup: list[tuple[int, int]] = []
    for item in positions:
        if item not in dedup:
            dedup.append(item)
    return dedup


def _place_group(grid: np.ndarray, base_r: int, base_c: int, mask: np.ndarray, color: int, rotation: int) -> None:
    indicator = _indicator_mask(rotation)
    ih, iw = indicator.shape
    layouts: list[tuple[int, int, int, int]] = []
    for sr in range(GROUP_AREA - CELL_SIZE + 1):
        for sc in range(GROUP_AREA - CELL_SIZE + 1):
            for ir, ic in _indicator_positions(sr, sc, ih, iw):
                if 0 <= ir and 0 <= ic and ir + ih <= GROUP_AREA and ic + iw <= GROUP_AREA:
                    layouts.append((sr, sc, ir, ic))
    assert layouts
    sr, sc, ir, ic = random.choice(layouts)
    sprite_view = grid[base_r + sr : base_r + sr + CELL_SIZE, base_c + sc : base_c + sc + CELL_SIZE]
    sprite_view[mask] = int(color)
    indicator_view = grid[base_r + ir : base_r + ir + ih, base_c + ic : base_c + ic + iw]
    indicator_view[indicator] = MAROON


def _expected_output_from_spec(
    grid: np.ndarray,
    n: int,
    top_colors: list[int],
    left_colors: list[int],
    sprite_specs: list[dict],
) -> np.ndarray:
    board_side = STEP * n + 1
    mapping = {int(spec["color"]): (spec["mask"], int(spec["rotation"])) for spec in sprite_specs}
    out = _blank_output_canvas(grid, board_side)
    for i, row_color in enumerate(left_colors):
        for j, column_color in enumerate(top_colors):
            sprite = mapping.get(int(column_color))
            if sprite is None:
                continue
            mask, rotation = sprite
            placed = _rotate_mask(mask, rotation)
            r0 = 1 + STEP * i
            c0 = 1 + STEP * j
            cell = out[r0 : r0 + placed.shape[0], c0 : c0 + placed.shape[1]]
            cell[placed] = int(row_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, 4)
    use_dummy = n >= 3 and random.random() < 0.5
    max_sprites = n - 1 if use_dummy else n
    sprite_count = random.randint(2, max_sprites)

    sprite_colors = random.sample(list(COLOR_POOL), sprite_count)
    remaining = [color for color in COLOR_POOL if color not in sprite_colors]

    row_palette_size = 2
    if not use_dummy and len(remaining) >= 3 and random.random() < 0.5:
        row_palette_size = 3
    row_palette = random.sample(remaining, row_palette_size)
    leftovers = [color for color in remaining if color not in row_palette]
    dummy_color = random.choice(leftovers) if use_dummy else None

    left_colors = _sample_row_colors(n, row_palette)
    top_colors = _sample_top_colors(n, sprite_colors, dummy_color)

    shape_indices = random.sample(range(len(SHAPE_LIBRARY)), sprite_count)
    rotations = [random.randrange(4) for _ in range(sprite_count)]
    if sprite_count > 1 and len(set(rotations)) == 1:
        rotations[-1] = (rotations[-1] + 1) % 4

    sprite_specs = [
        {
            "color": int(color),
            "mask": SHAPE_LIBRARY[idx].copy(),
            "rotation": int(rotation),
        }
        for color, idx, rotation in zip(sprite_colors, shape_indices, rotations)
    ]

    board_side = STEP * n + 1
    side = random.choice(("right", "bottom"))
    if side == "right":
        height = max(board_side, (sprite_count - 1) * GROUP_STEP + GROUP_AREA)
        width = board_side + 1 + GROUP_AREA
    else:
        height = board_side + 1 + GROUP_AREA
        width = max(board_side, (sprite_count - 1) * GROUP_STEP + GROUP_AREA)

    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:board_side, :board_side] = BLACK
    for r in range(0, board_side, STEP):
        grid[r, :board_side] = GREY
    for c in range(0, board_side, STEP):
        grid[:board_side, c] = GREY
    for j, color in enumerate(top_colors):
        grid[0, 2 + STEP * j] = int(color)
    for i, color in enumerate(left_colors):
        grid[2 + STEP * i, 0] = int(color)

    for idx, spec in enumerate(sprite_specs):
        if side == "right":
            base_r = idx * GROUP_STEP
            base_c = board_side + 1
        else:
            base_r = board_side + 1
            base_c = idx * GROUP_STEP
        _place_group(grid, base_r, base_c, spec["mask"], int(spec["color"]), int(spec["rotation"]))

    expected = _expected_output_from_spec(grid, n, top_colors, left_colors, sprite_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid coded-sprite assembly input")
