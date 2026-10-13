import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


ARROW_PATTERNS = {
    "up": np.array([[0, 1, 0], [1, 0, 1]], dtype=bool),
    "down": np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    "left": np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
    "right": np.array([[1, 0], [0, 1], [1, 0]], dtype=bool),
}


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _is_rect_frame(coords: np.ndarray) -> bool:
    if len(coords) < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = _coords_to_mask(coords)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _detect_arrow(coords: np.ndarray) -> str | None:
    if len(coords) != 3:
        return None
    mask = _coords_to_mask(coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if mask.shape == pattern.shape and np.array_equal(mask, pattern):
            return direction
    return None


def _stamp_position(
    frame_bbox: tuple[int, int, int, int], sprite_shape: tuple[int, int], direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0 = r0 + 1, c0 + 1
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    sh, sw = sprite_shape
    assert sh <= ih and sw <= iw, "sprite must fit inside the frame interior"
    if direction == "right":
        return ir0 + (ih - sh) // 2, ic0
    if direction == "left":
        return ir0 + (ih - sh) // 2, ic0 + (iw - sw)
    if direction == "down":
        return ir0, ic0 + (iw - sw) // 2
    if direction == "up":
        return ir0 + (ih - sh), ic0 + (iw - sw) // 2
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _bg_color(g)
    out = g.copy()

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps = _components(g == color)
        if len(comps) < 3:
            continue

        frame = next((comp for comp in comps if _is_rect_frame(comp)), None)
        if frame is None:
            continue

        arrow = None
        direction = None
        for comp in comps:
            d = _detect_arrow(comp)
            if d is not None:
                arrow = comp
                direction = d
                break
        if arrow is None or direction is None:
            continue

        sprite_candidates = [comp for comp in comps if comp is not frame and comp is not arrow]
        if len(sprite_candidates) != 1:
            sprite_candidates = sorted(sprite_candidates, key=len, reverse=True)
        if not sprite_candidates:
            continue
        sprite = sprite_candidates[0]

        sr0, sc0, sr1, sc1 = _bbox(sprite)
        sprite_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        sprite_mask[sprite[:, 0] - sr0, sprite[:, 1] - sc0] = True

        top, left = _stamp_position(_bbox(frame), sprite_mask.shape, direction)
        view = out[top : top + sprite_mask.shape[0], left : left + sprite_mask.shape[1]]
        view[sprite_mask] = color

    return out


def _make_sprite_mask(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    h = rng.randint(2, max_h)
    w = rng.randint(2, max_w)
    area_choices = [n for n in range(3, min(h * w, 7) + 1)]
    if h * w >= 5:
        area_choices = [n for n in area_choices if n < h * w] or area_choices
    area = rng.choice(area_choices)

    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _make_sprite_mask(rng, max_h, max_w)
        cells.add(rng.choice(frontier))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    mask = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    if any(np.array_equal(mask, pattern) for pattern in ARROW_PATTERNS.values()):
        return _make_sprite_mask(rng, max_h, max_w)
    return mask


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w][patch] = color


def _draw_frame(canvas: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    bottom = top + h - 1
    right = left + w - 1
    canvas[top, left : right + 1] = color
    canvas[bottom, left : right + 1] = color
    canvas[top : bottom + 1, left] = color
    canvas[top : bottom + 1, right] = color


def _build_group(
    rng: random.Random, color: int, direction: str, bg: int
) -> tuple[np.ndarray, np.ndarray]:
    frame_h = rng.randint(5, 9)
    frame_w = rng.randint(5, 9)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    sprite = _make_sprite_mask(rng, min(4, inner_h), min(4, inner_w))
    sh, sw = sprite.shape
    arrow = ARROW_PATTERNS[direction]
    ah, aw = arrow.shape

    gap_sprite_frame = rng.randint(1, 2)
    gap_arrow_sprite = rng.randint(1, 2)
    ortho_pad = rng.randint(0, 2)

    if direction in ("left", "right"):
        group_h = max(frame_h, sh, ah) + 2 * ortho_pad
        group_w = frame_w + sw + aw + gap_sprite_frame + gap_arrow_sprite
    else:
        group_h = frame_h + sh + ah + gap_sprite_frame + gap_arrow_sprite
        group_w = max(frame_w, sw, aw) + 2 * ortho_pad

    inp = np.full((group_h, group_w), bg, dtype=int)
    out = np.full((group_h, group_w), bg, dtype=int)

    if direction == "right":
        frame_top = ortho_pad + (group_h - 2 * ortho_pad - frame_h) // 2
        frame_left = group_w - frame_w
        sprite_top = ortho_pad + (group_h - 2 * ortho_pad - sh) // 2
        sprite_left = frame_left - gap_sprite_frame - sw
        arrow_top = ortho_pad + (group_h - 2 * ortho_pad - ah) // 2
        arrow_left = sprite_left - gap_arrow_sprite - aw
    elif direction == "left":
        frame_top = ortho_pad + (group_h - 2 * ortho_pad - frame_h) // 2
        frame_left = 0
        sprite_top = ortho_pad + (group_h - 2 * ortho_pad - sh) // 2
        sprite_left = frame_left + frame_w + gap_sprite_frame
        arrow_top = ortho_pad + (group_h - 2 * ortho_pad - ah) // 2
        arrow_left = sprite_left + sw + gap_arrow_sprite
    elif direction == "down":
        frame_top = 0
        frame_left = ortho_pad + (group_w - 2 * ortho_pad - frame_w) // 2
        sprite_top = frame_top + frame_h + gap_sprite_frame
        sprite_left = ortho_pad + (group_w - 2 * ortho_pad - sw) // 2
        arrow_top = sprite_top + sh + gap_arrow_sprite
        arrow_left = ortho_pad + (group_w - 2 * ortho_pad - aw) // 2
    else:
        frame_top = group_h - frame_h
        frame_left = ortho_pad + (group_w - 2 * ortho_pad - frame_w) // 2
        arrow_top = 0
        arrow_left = ortho_pad + (group_w - 2 * ortho_pad - aw) // 2
        sprite_top = arrow_top + ah + gap_arrow_sprite
        sprite_left = ortho_pad + (group_w - 2 * ortho_pad - sw) // 2

    _draw_frame(inp, frame_top, frame_left, frame_h, frame_w, color)
    _draw_frame(out, frame_top, frame_left, frame_h, frame_w, color)
    _paint_patch(inp, sprite, sprite_top, sprite_left, color)
    _paint_patch(out, sprite, sprite_top, sprite_left, color)
    _paint_patch(inp, arrow, arrow_top, arrow_left, color)
    _paint_patch(out, arrow, arrow_top, arrow_left, color)

    stamp_top, stamp_left = _stamp_position(
        (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1),
        sprite.shape,
        direction,
    )
    _paint_patch(out, sprite, stamp_top, stamp_left, color)
    return inp, out


def _layout_shapes(
    rng: random.Random, shapes: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(shapes)
    if n == 1:
        tile_rows, tile_cols = 1, 1
    elif n == 2:
        tile_rows, tile_cols = rng.choice([(1, 2), (2, 1)])
    else:
        tile_rows, tile_cols = 2, 2

    order = list(range(n))
    rng.shuffle(order)
    slots = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]

    row_groups = [[] for _ in range(tile_rows)]
    col_groups = [[] for _ in range(tile_cols)]
    for idx, slot in zip(order, slots):
        ph, pw = shapes[idx][0].shape
        row_groups[slot[0]].append(ph)
        col_groups[slot[1]].append(pw)

    row_heights = [max(group, default=0) + rng.randint(2, 5) for group in row_groups]
    col_widths = [max(group, default=0) + rng.randint(2, 5) for group in col_groups]
    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    row_gap = rng.randint(1, 3)
    col_gap = rng.randint(1, 3)

    H = top_margin + sum(row_heights) + row_gap * (tile_rows - 1) + bottom_margin
    W = left_margin + sum(col_widths) + col_gap * (tile_cols - 1) + right_margin

    bg = _bg_color(shapes[0][0])
    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    row_starts = []
    cur = top_margin
    for height in row_heights:
        row_starts.append(cur)
        cur += height + row_gap

    col_starts = []
    cur = left_margin
    for width in col_widths:
        col_starts.append(cur)
        cur += width + col_gap

    for idx, slot in zip(order, slots):
        patch_in, patch_out = shapes[idx]
        cell_h = row_heights[slot[0]]
        cell_w = col_widths[slot[1]]
        ph, pw = patch_in.shape
        if ph > cell_h or pw > cell_w:
            return None
        top = row_starts[slot[0]] + rng.randint(0, cell_h - ph)
        left = col_starts[slot[1]] + rng.randint(0, cell_w - pw)
        inp[top : top + ph, left : left + pw] = patch_in
        out[top : top + ph, left : left + pw] = patch_out

    return inp, out


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    return counts[bg] > max((n for c, n in counts.items() if c != bg), default=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    n_groups = rng.randint(2, 4)
    colors = rng.sample([c for c in range(10) if c != bg], n_groups)
    directions = [rng.choice(list(DIRS)) for _ in range(n_groups)]

    groups = []
    expected = []
    for color, direction in zip(colors, directions):
        patch_in, patch_out = _build_group(rng, color, direction, bg)
        groups.append((patch_in, patch_out))
        expected.append(direction)

    packed = _layout_shapes(rng, groups)
    if packed is None:
        return None
    inp, out = packed

    if not _bg_is_mode(inp, bg):
        return None
    if any(direction not in ("up", "down", "left", "right") for direction in expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
