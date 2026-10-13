import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
ROOM_COUNT_CHOICES = (2, 3, 4)
HEIGHT_RANGE = (14, 20)
LEFT_WIDTH_RANGE = (14, 20)
RIGHT_WIDTH_RANGE = (12, 18)
SHAPE_SIZE_RANGE = (4, 7)
SHAPE_AREA_MIN = 10
ROOM_PAD = 1
LEFT_BORDER_GAP = 2
RIGHT_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask, 1, constant_values=False)
    return (
        pad[1:-1, 1:-1]
        | pad[:-2, 1:-1]
        | pad[2:, 1:-1]
        | pad[1:-1, :-2]
        | pad[1:-1, 2:]
    )


def _erode4(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask, 1, constant_values=False)
    return (
        pad[1:-1, 1:-1]
        & pad[:-2, 1:-1]
        & pad[2:, 1:-1]
        & pad[1:-1, :-2]
        & pad[1:-1, 2:]
    )


def _shape_edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~_erode4(mask)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _adjacent_colors(grid: np.ndarray, comp_mask: np.ndarray, bg: int) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for r, c in np.argwhere(comp_mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                color = int(grid[nr, nc])
                if color != bg:
                    colors.add(color)
    return colors


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color) and int(np.sum(grid == color)) == h:
            candidates.append(c)
    assert len(candidates) == 1, "expected a unique separator column"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered fragment to the uniquely colored enclosed region on the
    left, rotate by shape if needed, and fill that region with the fragment's
    interior color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep = _find_separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    room_by_color: dict[int, np.ndarray] = {}
    for coords in _components(left == bg):
        if _touches_border(coords, left.shape):
            continue
        comp_mask = np.zeros_like(left, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        neighbors = _adjacent_colors(left, comp_mask, bg)
        if len(neighbors) != 1:
            continue
        color = int(next(iter(neighbors)))
        room_by_color[color] = comp_mask

    assert room_by_color, "expected enclosed target regions on the left"

    out = left.copy()
    seen_colors: set[int] = set()
    for coords in _components(right != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = right[r0 : r1 + 1, c0 : c1 + 1]
        shape = patch != bg
        edge = _shape_edge(shape)
        edge_colors = {int(v) for v in np.unique(patch[edge]) if int(v) != bg}
        assert len(edge_colors) == 1, "fragment edge color must be unique"
        border_color = int(next(iter(edge_colors)))
        assert border_color in room_by_color, "fragment must map to a region boundary color"

        fill_mask = shape & ~edge
        fill_colors = {int(v) for v in np.unique(patch[fill_mask]) if int(v) != bg}
        assert len(fill_colors) == 1, "fragment fill color must be unique"
        fill_color = int(next(iter(fill_colors)))

        room_mask = room_by_color[border_color]
        room_shape = _crop_mask(room_mask)
        assert any(np.array_equal(np.rot90(shape, k), room_shape) for k in range(4)), "fragment shape must match target region up to rotation"

        out[room_mask] = fill_color
        seen_colors.add(border_color)

    assert seen_colors == set(room_by_color), "every region must receive exactly one fragment"
    return out


def _connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _core_nonempty(mask: np.ndarray) -> bool:
    return bool(_erode4(mask).any())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_cut(mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    h, w = mask.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    trial = mask.copy()
    if side in ("top", "bottom"):
        if h < 4 or w < 3:
            return None
        depth = rng.randint(1, h - 2)
        c0 = rng.randint(0, w - 2)
        c1 = rng.randint(c0 + 1, w - 1)
        if c0 == 0 and c1 == w - 1:
            return None
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        trial[rows, c0 : c1 + 1] = False
    else:
        if w < 4 or h < 3:
            return None
        depth = rng.randint(1, w - 2)
        r0 = rng.randint(0, h - 2)
        r1 = rng.randint(r0 + 1, h - 1)
        if r0 == 0 and r1 == h - 1:
            return None
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        trial[r0 : r1 + 1, cols] = False
    if not trial.any():
        return None
    trial = _trim_mask(trial)
    if int(trial.sum()) < SHAPE_AREA_MIN or not _connected(trial) or not _core_nonempty(trial):
        return None
    return trial


def _generate_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        h = rng.randint(*SHAPE_SIZE_RANGE)
        w = rng.randint(*SHAPE_SIZE_RANGE)
        mask = np.ones((h, w), dtype=bool)
        cut_count = rng.choice([0, 1, 2])
        for _ in range(cut_count):
            for _ in range(30):
                cut = _random_cut(mask, rng)
                if cut is not None:
                    mask = cut
                    break
        if int(mask.sum()) < SHAPE_AREA_MIN:
            continue
        if not _connected(mask) or not _core_nonempty(mask):
            continue
        return mask
    return None


def _room_patch(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region = np.pad(shape, 1, constant_values=False)
    shell = _dilate4(region) & ~region
    return region, shell


def _place_patch_bbox(
    reserved: np.ndarray,
    patch_shape: tuple[int, int],
    rng: random.Random,
    edge_gap: int,
    pad: int,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    ph, pw = patch_shape
    r_lo = edge_gap
    c_lo = edge_gap
    r_hi = h - ph - edge_gap
    c_hi = w - pw - edge_gap
    if r_lo > r_hi or c_lo > c_hi:
        return None

    spots = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    rng.shuffle(spots)
    for r0, c0 in spots:
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h, r0 + ph + pad)
        cc1 = min(w, c0 + pw + pad)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _fragment_from_shape(shape: np.ndarray, border_color: int, fill_color: int, rot: int, bg: int) -> np.ndarray:
    edge = _shape_edge(shape)
    patch = np.full(shape.shape, bg, dtype=int)
    patch[edge] = border_color
    patch[shape & ~edge] = fill_color
    return np.rot90(patch, rot)


def _build_expected(left: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = left.copy()
    for room in rooms:
        out[room["abs_region"]] = int(room["fill_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    height = rng.randint(*HEIGHT_RANGE)
    left_w = rng.randint(*LEFT_WIDTH_RANGE)
    right_w = rng.randint(*RIGHT_WIDTH_RANGE)
    room_count = rng.choice(ROOM_COUNT_CHOICES)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    separator_color = palette.pop()
    frame_color = palette.pop()
    border_colors = [palette.pop() for _ in range(room_count)]
    fill_palette_size = rng.choice([2, 3]) if len(palette) >= 3 else 2
    fill_palette = [palette.pop() for _ in range(fill_palette_size)]

    left = np.full((height, left_w), bg, dtype=int)
    left[0, :] = frame_color
    left[-1, :] = frame_color
    left[:, 0] = frame_color
    left[:, -1] = frame_color

    rooms: list[dict] = []
    reserved_left = np.zeros_like(left, dtype=bool)
    reserved_left[0, :] = True
    reserved_left[-1, :] = True
    reserved_left[:, 0] = True
    reserved_left[:, -1] = True

    for border_color in border_colors:
        shape = _generate_shape(rng)
        if shape is None:
            return None
        region, shell = _room_patch(shape)
        placed = _place_patch_bbox(reserved_left, shell.shape, rng, LEFT_BORDER_GAP, ROOM_PAD)
        if placed is None:
            return None
        r0, c0 = placed
        left[r0 : r0 + shell.shape[0], c0 : c0 + shell.shape[1]][shell] = border_color

        abs_region = np.zeros_like(left, dtype=bool)
        abs_region[r0 : r0 + region.shape[0], c0 : c0 + region.shape[1]] = region
        rooms.append(
            {
                "border_color": border_color,
                "fill_color": rng.choice(fill_palette),
                "shape": shape,
                "abs_region": abs_region,
            }
        )

    expected = _build_expected(left, rooms)

    right = np.full((height, right_w), bg, dtype=int)
    reserved_right = np.zeros_like(right, dtype=bool)
    for room in rooms:
        rot = rng.randrange(4)
        frag = _fragment_from_shape(room["shape"], room["border_color"], room["fill_color"], rot, bg)
        placed = _place_patch_bbox(reserved_right, frag.shape, rng, 0, RIGHT_PAD)
        if placed is None:
            return None
        r0, c0 = placed
        right[r0 : r0 + frag.shape[0], c0 : c0 + frag.shape[1]][frag != bg] = frag[frag != bg]

    grid = np.full((height, left_w + 1 + right_w), bg, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = separator_color
    grid[:, left_w + 1 :] = right

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
