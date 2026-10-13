import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FRAME_COLORS = (RED, BLUE, GREEN, YELLOW)
SPRITE_COLORS = (5, 6, 7, 8, 9)

MAX_GEN_ATTEMPTS = 400
MAX_GRID_SIDE = 30

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _tight_crop(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BLACK)
    assert coords.size > 0, "expected a non-empty sprite"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _apply_transform(arr: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(arr)
    if color == BLUE:
        return np.flipud(arr)
    if color == GREEN:
        return np.rot90(arr, -1)
    if color == YELLOW:
        return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    raise ValueError(f"unsupported frame color: {color}")


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.int16).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _border_is_uniform(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    top = _rect_sum(prefix, r0, r0, c0, c1)
    bottom = _rect_sum(prefix, r1, r1, c0, c1)
    left = _rect_sum(prefix, r0 + 1, r1 - 1, c0, c0)
    right = _rect_sum(prefix, r0 + 1, r1 - 1, c1, c1)
    border_count = top + bottom + left + right
    perimeter = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
    return border_count == perimeter


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []

    for color in FRAME_COLORS:
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 8:
            continue

        rows_to_cols: dict[int, list[int]] = {}
        cols_to_rows: dict[int, list[int]] = {}
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            rows_to_cols.setdefault(rr, []).append(cc)
            cols_to_rows.setdefault(cc, []).append(rr)

        for cols in rows_to_cols.values():
            cols.sort()
        for rows in cols_to_rows.values():
            rows.sort()
        col_row_sets = {col: set(rows) for col, rows in cols_to_rows.items()}
        prefix = _prefix_sum(grid == color)
        seen: set[tuple[int, int, int, int]] = set()

        for r0, cols in rows_to_cols.items():
            for idx, c0 in enumerate(cols):
                possible_bottoms = cols_to_rows[c0]
                for c1 in cols[idx + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    rows_at_c1 = col_row_sets[c1]
                    for r1 in possible_bottoms:
                        if r1 - r0 < 2 or r1 not in rows_at_c1:
                            continue
                        bbox = (int(r0), int(c0), int(r1), int(c1))
                        if bbox in seen:
                            continue
                        if _border_is_uniform(prefix, *bbox):
                            seen.add(bbox)
                            frames.append(
                                {
                                    "color": int(color),
                                    "bbox": bbox,
                                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                                }
                            )

    return frames


def _contains(parent_bbox: tuple[int, int, int, int], child_bbox: tuple[int, int, int, int]) -> bool:
    pr0, pc0, pr1, pc1 = parent_bbox
    cr0, cc0, cr1, cc1 = child_bbox
    return pr0 < cr0 and pc0 < cc0 and cr1 < pr1 and cc1 < pc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse a nested chain of colored rectangular frames and apply frame-coded
    transformations to the innermost sprite from inside to outside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one operation frame"
    frames.sort(key=lambda frame: frame["area"])

    parent: list[int | None] = [None] * len(frames)
    children: list[list[int]] = [[] for _ in frames]
    for i, frame in enumerate(frames):
        best_parent = None
        for j, other in enumerate(frames):
            if other["area"] <= frame["area"]:
                continue
            if _contains(other["bbox"], frame["bbox"]):
                if best_parent is None or other["area"] < frames[best_parent]["area"]:
                    best_parent = j
        parent[i] = best_parent
        if best_parent is not None:
            children[best_parent].append(i)

    depths = [0] * len(frames)

    def depth_of(idx: int) -> int:
        if depths[idx]:
            return depths[idx]
        p = parent[idx]
        depths[idx] = 1 if p is None else depth_of(p) + 1
        return depths[idx]

    best_leaf = None
    best_score = None
    best_sprite = None
    for i, frame in enumerate(frames):
        if children[i]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any(inner != BLACK):
            continue
        sprite = _tight_crop(inner)
        score = (depth_of(i), int(np.count_nonzero(sprite)), int(sprite.shape[0] * sprite.shape[1]))
        if best_score is None or score > best_score:
            best_score = score
            best_leaf = i
            best_sprite = sprite

    assert best_leaf is not None and best_sprite is not None, "expected an innermost framed sprite"

    out = best_sprite
    cur = best_leaf
    while cur is not None:
        out = _apply_transform(out, frames[cur]["color"])
        cur = parent[cur]
    return out


def _random_connected_sprite() -> np.ndarray:
    h = int(_RNG.integers(2, 6))
    w = int(_RNG.integers(2, 6))
    area_hi = max(4, h * w)
    area = int(_RNG.integers(3, area_hi))

    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        pick = frontier[int(_RNG.integers(len(frontier)))]
        cells.add(pick)

    color_count = 1 + int((_GEN_COUNTER + int(_RNG.integers(2))) % 2)
    palette = list(map(int, _RNG.choice(SPRITE_COLORS, size=color_count, replace=False)))
    sprite = np.zeros((h, w), dtype=int)
    coords = list(cells)
    _RNG.shuffle(coords)

    if color_count == 1:
        for r, c in coords:
            sprite[r, c] = palette[0]
    else:
        split = int(_RNG.integers(1, len(coords)))
        for idx, (r, c) in enumerate(coords):
            sprite[r, c] = palette[0] if idx < split else palette[1]

    return _tight_crop(sprite)


def _frame_colors_for_attempt() -> list[int]:
    global _GEN_COUNTER
    call_id = _GEN_COUNTER
    depth = 1 + (call_id % 4)
    shift = (call_id // 4) % 4
    order = list(FRAME_COLORS[shift:] + FRAME_COLORS[:shift])
    return order[:depth]


def _wrap_chain(sprite: np.ndarray, inner_to_outer_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    sprite_h, sprite_w = map(int, sprite.shape)
    inner_pad_t = int(_RNG.integers(0, 2))
    inner_pad_b = int(_RNG.integers(0, 2))
    inner_pad_l = int(_RNG.integers(0, 2))
    inner_pad_r = int(_RNG.integers(0, 2))

    current_h = sprite_h + inner_pad_t + inner_pad_b + 2
    current_w = sprite_w + inner_pad_l + inner_pad_r + 2
    frame_boxes = [(0, 0, current_h - 1, current_w - 1)]
    sprite_pos = (1 + inner_pad_t, 1 + inner_pad_l)

    for _ in range(len(inner_to_outer_colors) - 1):
        pad_t = int(_RNG.integers(1, 3))
        pad_b = int(_RNG.integers(1, 3))
        pad_l = int(_RNG.integers(1, 3))
        pad_r = int(_RNG.integers(1, 3))
        shift_r = 1 + pad_t
        shift_c = 1 + pad_l

        frame_boxes = [
            (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
            for r0, c0, r1, c1 in frame_boxes
        ]
        sprite_pos = (sprite_pos[0] + shift_r, sprite_pos[1] + shift_c)
        current_h += pad_t + pad_b + 2
        current_w += pad_l + pad_r + 2
        frame_boxes.insert(0, (0, 0, current_h - 1, current_w - 1))

    margin_t = int(_RNG.integers(1, 4))
    margin_b = int(_RNG.integers(1, 4))
    margin_l = int(_RNG.integers(1, 4))
    margin_r = int(_RNG.integers(1, 4))
    grid_h = current_h + margin_t + margin_b
    grid_w = current_w + margin_l + margin_r
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        raise ValueError("layout too large")

    grid = np.zeros((grid_h, grid_w), dtype=int)
    row_off = margin_t
    col_off = margin_l

    for color, (r0, c0, r1, c1) in zip(reversed(inner_to_outer_colors), frame_boxes):
        rr0 = r0 + row_off
        cc0 = c0 + col_off
        rr1 = r1 + row_off
        cc1 = c1 + col_off
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color

    sr = sprite_pos[0] + row_off
    sc = sprite_pos[1] + col_off
    sh, sw = sprite.shape
    sprite_mask = sprite != BLACK
    patch = grid[sr : sr + sh, sc : sc + sw]
    patch[sprite_mask] = sprite[sprite_mask]

    out = sprite.copy()
    for color in inner_to_outer_colors:
        out = _apply_transform(out, color)
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    sprite = _random_connected_sprite()
    colors = _frame_colors_for_attempt()

    try:
        grid, expected = _wrap_chain(sprite, colors)
    except ValueError:
        return None

    if np.array_equal(expected, sprite):
        return None

    frames = _find_frames(grid)
    if len(frames) != len(colors):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GEN_ATTEMPTS):
        _GEN_COUNTER += 1
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
