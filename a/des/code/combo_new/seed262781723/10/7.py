import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LAYER_COUNT_CHOICES = (2, 3, 4)
CORE_SIZE_BY_LAYERS = {
    2: (2, 5),
    3: (2, 5),
    4: (2, 4),
}
CORE_COLOR_COUNT_CHOICES = (1, 2, 3)
CORE_KIND_CHOICES = ("blob", "bands", "holes")
PADDING_CHOICES = (2, 3)
SIDE_CHOICES = ("top", "bottom", "left", "right")
CORNER_CHOICES = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(grid != bg)
    assert box is not None, "grid must contain at least one non-background pixel"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    return mask


def _detect_outer_frame(grid: np.ndarray, bg: int) -> dict | None:
    h, w = map(int, grid.shape)
    if h < 5 or w < 5:
        return None

    border = _border_mask((h, w))
    if np.any(grid[border] == bg):
        return None

    border_vals = grid[border]
    colors, counts = np.unique(border_vals, return_counts=True)
    if len(colors) != 2:
        return None

    order = np.argsort(counts)
    door_color = int(colors[order[0]])
    frame_color = int(colors[order[1]])
    if int(counts[order[0]]) != 1:
        return None
    if int(counts[order[1]]) != int(border.sum()) - 1:
        return None

    door_pos = np.argwhere(border & (grid == door_color))
    if door_pos.shape != (1, 2):
        return None
    dr, dc = map(int, door_pos[0])
    if (dr in (0, h - 1)) and (dc in (0, w - 1)):
        return None

    key_candidates = {
        "tl": (1, 1),
        "tr": (1, w - 2),
        "bl": (h - 2, 1),
        "br": (h - 2, w - 2),
    }
    matches = [(name, pos) for name, pos in key_candidates.items() if int(grid[pos]) == door_color]
    if len(matches) != 1:
        return None
    key_name, key_pos = matches[0]

    interior = grid[1:-1, 1:-1].copy()
    kr, kc = key_pos
    interior[kr - 1, kc - 1] = bg
    if not np.any(interior != bg):
        return None

    return {
        "frame_color": frame_color,
        "door_color": door_color,
        "door_pos": (dr, dc),
        "key_name": key_name,
        "key_pos": key_pos,
        "border_mask": border,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Strip nested keyed door-frames until only the hidden core pattern remains.

    Input structure:
    - The background is black, and the meaningful object is a central stack of 2-4 nested one-pixel-thick rectangular frames.
    - Each frame border is mostly one color, except for one single-pixel door on an edge whose color matches that layer's key.
    - Each key is a single pixel placed at one inner corner of its frame, and deeper layers plus the final core pattern sit inside the frame.

    Rule:
    - Crop to the overall non-background structure and inspect its outermost rectangle.
    - If the outer border is a valid frame with exactly one off-color door and exactly one matching inner-corner key, remove that whole border and that key pixel.
    - Repeat the same peeling step from the outside inward until no such keyed frame remains.
    - Return the remaining non-background core pattern, cropped to its minimal bounding box.

    Key insight:
    - The door and key are local clues that identify which rectangular layer to peel next; after repeatedly stripping those carrier layers, the only meaningful content left is the inner pattern itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    if len(set(corners)) == 1:
        bg = corners[0]
    else:
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(counts)])

    current = _crop_non_bg(g, bg)
    while True:
        info = _detect_outer_frame(current, bg)
        if info is None:
            return _crop_non_bg(current, bg)

        current = current.copy()
        current[info["border_mask"]] = bg
        kr, kc = info["key_pos"]
        current[kr, kc] = bg
        current = _crop_non_bg(current, bg)


def _random_blob_mask(rng: np.random.Generator, h: int, w: int, holes: bool) -> np.ndarray:
    target_min = max(3, (h * w) // 3)
    target_max = max(target_min, h * w - (1 if holes else 0))
    target = int(rng.integers(target_min, target_max + 1))

    mask = np.zeros((h, w), dtype=bool)
    seed_count = int(rng.integers(1, min(3, h * w) + 1))
    seeds = rng.choice(h * w, size=seed_count, replace=False)
    frontier: list[tuple[int, int]] = []
    for idx in np.atleast_1d(seeds):
        r = int(idx) // w
        c = int(idx) % w
        mask[r, c] = True
        frontier.append((r, c))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while int(mask.sum()) < target:
        r, c = frontier[int(rng.integers(len(frontier)))]
        rng.shuffle(dirs)
        placed = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                mask[nr, nc] = True
                frontier.append((nr, nc))
                placed = True
                break
        if not placed:
            empties = np.argwhere(~mask)
            if empties.size == 0:
                break
            nr, nc = map(int, empties[int(rng.integers(len(empties)))])
            if any(
                0 <= nr + dr < h and 0 <= nc + dc < w and mask[nr + dr, nc + dc]
                for dr, dc in dirs
            ):
                mask[nr, nc] = True
                frontier.append((nr, nc))

    if holes and h >= 3 and w >= 3 and int(mask.sum()) >= 6:
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if mask[r, c]]
        rng.shuffle(interior)
        for r, c in interior:
            trial = mask.copy()
            trial[r, c] = False
            if _is_4_connected(trial):
                mask = trial
                break

    return mask


def _is_4_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _colorize_mask(
    rng: np.random.Generator, mask: np.ndarray, colors: tuple[int, ...], kind: str
) -> np.ndarray:
    h, w = mask.shape
    grid = np.full((h, w), BG, dtype=int)
    k = len(colors)
    row_shift = int(rng.integers(k))
    col_shift = int(rng.integers(k))

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if kind == "bands":
                idx = (r + row_shift) % k
            elif kind == "holes":
                idx = (r + c + row_shift) % k
            else:
                idx = (2 * r + c + col_shift) % k
            grid[r, c] = int(colors[idx])
    return _crop_non_bg(grid, BG)


def _random_core_pattern(rng: np.random.Generator, max_dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    for _ in range(200):
        h = int(rng.integers(2, max_dim + 1))
        w = int(rng.integers(2, max_dim + 1))
        color_count = int(rng.choice(CORE_COLOR_COUNT_CHOICES))
        colors = tuple(int(c) for c in rng.choice(PALETTE, size=color_count, replace=False))
        kind = str(rng.choice(CORE_KIND_CHOICES))
        mask = _random_blob_mask(rng, h, w, holes=(kind == "holes"))
        if not _is_4_connected(mask):
            continue
        grid = _colorize_mask(rng, mask, colors, kind)
        if min(grid.shape) < 2:
            continue
        if _detect_outer_frame(grid, BG) is not None:
            continue
        used_colors = tuple(int(c) for c in np.unique(grid) if int(c) != BG)
        return grid, used_colors
    raise RuntimeError("failed to build a non-degenerate core pattern")


def _split_margin(rng: np.random.Generator, total: int) -> tuple[int, int]:
    low = max(1, total // 2 - 1)
    high = min(total - 1, total // 2 + 1)
    if low > high:
        low, high = 1, total - 1
    first = int(rng.integers(low, high + 1))
    return first, int(total - first)


def _wrap_with_frame(
    rng: np.random.Generator,
    content: np.ndarray,
    frame_color: int,
    door_color: int,
    pad_y: int,
    pad_x: int,
    side: str,
    corner: str,
) -> np.ndarray:
    ch, cw = map(int, content.shape)
    h = ch + 2 * pad_y + 2
    w = cw + 2 * pad_x + 2
    out = np.full((h, w), BG, dtype=int)

    out[0, :] = frame_color
    out[h - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, w - 1] = frame_color

    top = 1 + pad_y
    left = 1 + pad_x
    out[top : top + ch, left : left + cw] = content

    if side in ("top", "bottom"):
        door_col = int(rng.integers(1, w - 1))
        door_pos = (0, door_col) if side == "top" else (h - 1, door_col)
    else:
        door_row = int(rng.integers(1, h - 1))
        door_pos = (door_row, 0) if side == "left" else (door_row, w - 1)
    out[door_pos] = door_color

    key_pos = {
        "tl": (1, 1),
        "tr": (1, w - 2),
        "bl": (h - 2, 1),
        "br": (h - 2, w - 2),
    }[corner]
    out[key_pos] = door_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layer_count = int(rng.choice(LAYER_COUNT_CHOICES))
    core_max_dim = CORE_SIZE_BY_LAYERS[layer_count][1]
    core, core_colors = _random_core_pattern(rng, core_max_dim)

    needed = layer_count + len(core_colors)
    if needed > len(PALETTE):
        return None

    sampled = tuple(int(c) for c in rng.choice(PALETTE, size=needed, replace=False))
    frame_colors = sampled[:layer_count]
    remap = sampled[layer_count:]
    color_map = {old: new for old, new in zip(core_colors, remap)}
    expected = np.full_like(core, BG)
    for old, new in color_map.items():
        expected[core == old] = new
    core_colors = tuple(color_map[c] for c in core_colors)

    terminal_color = int(rng.choice(core_colors))
    door_colors = list(frame_colors[1:]) + [terminal_color]

    current = expected.copy()
    for idx in range(layer_count - 1, -1, -1):
        pad_y = int(rng.choice(PADDING_CHOICES))
        pad_x = int(rng.choice(PADDING_CHOICES))
        side = str(rng.choice(SIDE_CHOICES))
        corner = str(rng.choice(CORNER_CHOICES))
        current = _wrap_with_frame(
            rng,
            current,
            frame_color=int(frame_colors[idx]),
            door_color=int(door_colors[idx]),
            pad_y=pad_y,
            pad_x=pad_x,
            side=side,
            corner=corner,
        )

    oh, ow = map(int, current.shape)
    if oh > 28 or ow > 28:
        return None

    extra_h = int(rng.integers(2, 30 - oh + 1))
    extra_w = int(rng.integers(2, 30 - ow + 1))
    top, bottom = _split_margin(rng, extra_h)
    left, right = _split_margin(rng, extra_w)

    grid = np.full((oh + extra_h, ow + extra_w), BG, dtype=int)
    grid[top : top + oh, left : left + ow] = current

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid keyed-frame puzzle")
