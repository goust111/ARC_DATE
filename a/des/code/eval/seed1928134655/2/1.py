import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
TEXTURE_SIZE = 3
PALETTE_BLOCK = 4
PALETTE_WIDTH = 4
GAP_RANGE = (2, 4)
RETRY_LIMIT = 400
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SHAPE_KINDS = ("rectangle", "notch", "poly")

PATTERN_LIBRARY = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    assert len(rows) > 0 and len(cols) > 0
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _find_canvas_start(grid: np.ndarray) -> int:
    seen_non_bg = False
    w = grid.shape[1]
    c = 0
    while c < w:
        has_color = bool(np.any(grid[:, c] != BG))
        if has_color:
            seen_non_bg = True
            c += 1
            continue
        if not seen_non_bg:
            c += 1
            continue
        gap_start = c
        while c < w and not np.any(grid[:, c] != BG):
            c += 1
        if c < w:
            return c
        raise AssertionError("expected a main canvas after the palette gap")
    raise AssertionError("failed to locate the palette/canvas split")


def _read_palette(panel: np.ndarray) -> dict[int, np.ndarray]:
    assert panel.shape[1] >= PALETTE_WIDTH, "palette area is too narrow"
    mapping: dict[int, np.ndarray] = {}
    r = 0
    while r < panel.shape[0]:
        row = panel[r, :PALETTE_WIDTH]
        if not np.any(row != BG):
            r += 1
            continue
        if np.all(row == WALL):
            r += 1
            continue
        if r + PALETTE_BLOCK > panel.shape[0]:
            break
        seed = int(panel[r, 0])
        if seed in (BG, WALL):
            r += 1
            continue
        texture = panel[r + 1 : r + 1 + TEXTURE_SIZE, 1 : 1 + TEXTURE_SIZE]
        assert texture.shape == (TEXTURE_SIZE, TEXTURE_SIZE)
        values = set(map(int, np.unique(texture)))
        assert values <= {BG, seed}, "palette textures must be binary in seed color and black"
        assert seed not in mapping, "palette seed colors must be unique"
        mapping[seed] = texture.copy()
        r += PALETTE_BLOCK
    assert mapping, "expected at least one palette block"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the palette texture keyed by its seed color,
    then remove the palette area and keep only the main canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    canvas_start = _find_canvas_start(g)
    palette_end = canvas_start - 1
    while palette_end >= 0 and not np.any(g[:, palette_end] != BG):
        palette_end -= 1
    assert palette_end + 1 >= PALETTE_WIDTH, "palette width must include a 3x3 texture block"

    palette = _read_palette(g[:, : palette_end + 1])
    canvas = g[:, canvas_start:].copy()
    out = canvas.copy()

    for coords in _components(canvas != WALL, connectivity=4):
        if _touches_border(coords, canvas.shape):
            continue
        region_colors = {
            int(canvas[r, c])
            for r, c in coords
            if int(canvas[r, c]) not in (BG, WALL)
        }
        assert len(region_colors) == 1, "each enclosed region must contain exactly one seed color"
        seed = next(iter(region_colors))
        assert seed in palette, "seed color must appear in the palette"
        texture = palette[seed]
        r0, c0, _, _ = _bbox(coords)
        for r, c in coords:
            out[r, c] = int(texture[(int(r) - r0) % TEXTURE_SIZE, (int(c) - c0) % TEXTURE_SIZE])
    return out


def _transform_pattern(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _sample_textures(colors: list[int]) -> dict[int, np.ndarray] | None:
    textures: dict[int, np.ndarray] = {}
    seen: set[bytes] = set()
    for color in colors:
        for _ in range(80):
            pattern = _transform_pattern(random.choice(PATTERN_LIBRARY))
            if int(pattern.sum()) < 3 or int(pattern.sum()) > 7:
                continue
            sig = pattern.tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            textures[color] = np.where(pattern, color, BG).astype(int)
            break
        else:
            return None
    return textures


def _make_rectangle() -> np.ndarray:
    return np.ones((random.randint(2, 6), random.randint(2, 6)), dtype=bool)


def _make_notch() -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    notch_h = random.randint(1, h - 1)
    notch_w = random.randint(1, w - 1)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    mask = _trim_mask(mask)
    if mask.sum() < 4 or min(mask.shape) < 2:
        raise AssertionError("degenerate notch mask")
    return mask


def _make_poly() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    target = random.randint(max(5, (h * w) // 2), h * w - 1)
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.sum() < 5 or min(mask.shape) < 2 or np.all(mask):
            continue
        return mask
    return None


def _sample_region_mask() -> np.ndarray | None:
    kind = random.choice(SHAPE_KINDS)
    if kind == "rectangle":
        return _make_rectangle()
    if kind == "notch":
        return _make_notch()
    return _make_poly()


def _region_patch(inner_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner = np.pad(inner_mask, 1, constant_values=False)
    wall = _dilate4(inner) & ~inner
    reserve = _dilate4(inner | wall)
    return inner, wall, reserve


def _placement_candidates(occupied: np.ndarray, reserve: np.ndarray) -> list[tuple[int, int]]:
    h, w = occupied.shape
    rh, rw = reserve.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            if not np.any(occupied[r0 : r0 + rh, c0 : c0 + rw] & reserve):
                out.append((r0, c0))
    return out


def _paint_region(
    inp: np.ndarray,
    out: np.ndarray,
    inner: np.ndarray,
    wall: np.ndarray,
    top: int,
    left: int,
    seed: int,
    texture: np.ndarray,
) -> None:
    wall_coords = np.argwhere(wall)
    for wr, wc in wall_coords:
        inp[top + wr, left + wc] = WALL
        out[top + wr, left + wc] = WALL

    inner_coords = np.argwhere(inner)
    assert len(inner_coords) > 0
    r0 = int(inner_coords[:, 0].min())
    c0 = int(inner_coords[:, 1].min())
    for ir, ic in inner_coords:
        value = int(texture[(int(ir) - r0) % TEXTURE_SIZE, (int(ic) - c0) % TEXTURE_SIZE])
        out[top + ir, left + ic] = value
        inp[top + ir, left + ic] = BG

    seed_r, seed_c = map(int, random.choice(inner_coords.tolist()))
    inp[top + seed_r, left + seed_c] = seed


def _shift_left(grid: np.ndarray) -> np.ndarray:
    non_bg_cols = np.where(np.any(grid != BG, axis=0))[0]
    assert len(non_bg_cols) > 0
    shift = int(non_bg_cols[0])
    if shift == 0:
        return grid.copy()
    out = np.full_like(grid, BG)
    out[:, : grid.shape[1] - shift] = grid[:, shift:]
    return out


def _build_palette(colors: list[int], textures: dict[int, np.ndarray]) -> np.ndarray:
    height = len(colors) * PALETTE_BLOCK + (len(colors) - 1)
    panel = np.full((height, PALETTE_WIDTH), BG, dtype=int)
    r = 0
    for index, color in enumerate(colors):
        panel[r, 0] = int(color)
        panel[r + 1 : r + 1 + TEXTURE_SIZE, 1 : 1 + TEXTURE_SIZE] = textures[color]
        r += PALETTE_BLOCK
        if index + 1 < len(colors):
            panel[r, :] = WALL
            r += 1
    return panel


def _embed_canvas(canvas: np.ndarray, full_height: int, top_offset: int) -> np.ndarray:
    out = np.full((full_height, canvas.shape[1]), BG, dtype=int)
    out[top_offset : top_offset + canvas.shape[0], :] = canvas
    return out


def _gen_attempt() -> np.ndarray | None:
    palette_count = random.randint(2, 4)
    palette_colors = random.sample(COLOR_POOL, palette_count)
    textures = _sample_textures(palette_colors)
    if textures is None:
        return None

    region_count = random.randint(1, 4)
    canvas_h = random.randint(9 + 2 * region_count, 13 + 3 * region_count)
    canvas_w = random.randint(10 + 3 * region_count, 16 + 4 * region_count)
    inp_canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    out_canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    if region_count <= palette_count and random.random() < 0.5:
        region_colors = random.sample(palette_colors, region_count)
    else:
        region_colors = [random.choice(palette_colors) for _ in range(region_count)]
    random.shuffle(region_colors)

    for seed in region_colors:
        placed = False
        for _ in range(120):
            mask = _sample_region_mask()
            if mask is None:
                continue
            inner, wall, reserve = _region_patch(mask)
            candidates = _placement_candidates(occupied, reserve)
            if not candidates:
                continue
            top, left = random.choice(candidates)
            _paint_region(inp_canvas, out_canvas, inner, wall, top, left, seed, textures[seed])
            occupied[top : top + reserve.shape[0], left : left + reserve.shape[1]] |= reserve
            placed = True
            break
        if not placed:
            return None

    inp_canvas = _shift_left(inp_canvas)
    out_canvas = _shift_left(out_canvas)

    palette_panel = _build_palette(palette_colors, textures)
    top_offset = random.randint(0, 2)
    full_height = max(palette_panel.shape[0], top_offset + canvas_h)
    inp_main = _embed_canvas(inp_canvas, full_height, top_offset)
    out_main = _embed_canvas(out_canvas, full_height, top_offset)

    gap = random.randint(*GAP_RANGE)
    inp = np.full((full_height, palette_panel.shape[1] + gap + canvas_w), BG, dtype=int)
    inp[: palette_panel.shape[0], : palette_panel.shape[1]] = palette_panel
    inp[:, palette_panel.shape[1] + gap :] = inp_main

    try:
        if np.array_equal(main(inp), out_main):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
