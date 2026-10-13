import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT_CHOICES = (2, 3, 4)
INNER_SIZE_RANGE = (3, 8)
GRID_SIDE_RANGE = (22, 38)
GRID_NEAR_SQUARE_PROB = 0.7
GRID_DELTAS = (-4, -2, 0, 2, 4)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

PATTERN_MODES = ("block", "cross", "ring", "stairs", "stripes", "diag", "blob")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _scale_block(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    sh, sw = src.shape
    th, tw = target_shape
    row_idx = (np.arange(th) * sh) // th
    col_idx = (np.arange(tw) * sw) // tw
    return src[row_idx[:, None], col_idx[None, :]]


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frame_mask = grid == FRAME_COLOR
    frames: list[dict[str, object]] = []
    for coords in _components(frame_mask):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        box = frame_mask[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros_like(box, dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if not np.array_equal(box, expected):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        colors = [int(v) for v in np.unique(inner) if int(v) not in (BACKGROUND, FRAME_COLOR)]
        assert len(colors) == 2, "each frame interior must contain exactly one pattern color and one marker color"
        counts = {color: int(np.sum(inner == color)) for color in colors}
        marker_candidates = [color for color, count in counts.items() if count == 1]
        assert len(marker_candidates) == 1, "marker color must be unique inside its frame"
        marker_color = marker_candidates[0]
        pattern_color = next(color for color in colors if color != marker_color)

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_top": r0 + 1,
                "inner_left": c0 + 1,
                "inner": inner,
                "marker_color": marker_color,
                "pattern_color": pattern_color,
            }
        )

    frames.sort(key=lambda item: tuple(item["bbox"]))  # stable pairing/output order
    assert len(frames) >= 2 and len(frames) % 2 == 0, "expected an even number of frames"
    return frames


def _frame_pairs(frames: list[dict[str, object]]) -> list[tuple[int, int]]:
    pattern_to_index: dict[int, int] = {}
    for idx, frame in enumerate(frames):
        pattern_color = int(frame["pattern_color"])
        assert pattern_color not in pattern_to_index, "pattern colors must be unique across frames"
        pattern_to_index[pattern_color] = idx

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for idx, frame in enumerate(frames):
        if idx in used:
            continue
        partner = pattern_to_index.get(int(frame["marker_color"]))
        assert partner is not None and partner != idx, "every marker color must reference another frame pattern color"
        partner_frame = frames[partner]
        assert int(partner_frame["marker_color"]) == int(frame["pattern_color"]), "pairing must be mutual"
        used.add(idx)
        used.add(partner)
        pairs.append(tuple(sorted((idx, partner))))

    assert len(used) == len(frames), "all frames must be paired"
    pairs = sorted(set(pairs))
    assert len(pairs) * 2 == len(frames)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    pairs = _frame_pairs(frames)
    out = g.copy()
    interiors = [np.array(frame["inner"], copy=True) for frame in frames]

    for left, right in pairs:
        left_frame = frames[left]
        right_frame = frames[right]
        left_inner = interiors[left]
        right_inner = interiors[right]

        left_top = int(left_frame["inner_top"])
        left_left = int(left_frame["inner_left"])
        right_top = int(right_frame["inner_top"])
        right_left = int(right_frame["inner_left"])

        left_shape = left_inner.shape
        right_shape = right_inner.shape

        out[left_top : left_top + left_shape[0], left_left : left_left + left_shape[1]] = _scale_block(
            right_inner, left_shape
        )
        out[
            right_top : right_top + right_shape[0],
            right_left : right_left + right_shape[1],
        ] = _scale_block(left_inner, right_shape)

    return out


def _empty_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _mask_block(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    rh = random.randint(max(2, h // 2), h)
    rw = random.randint(max(2, w // 2), w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _mask_cross(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    rr = random.randrange(h)
    cc = random.randrange(w)
    mask[rr, :] = True
    mask[:, cc] = True
    return mask


def _mask_ring(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    inset_max = min((h - 1) // 2, (w - 1) // 2)
    inset = random.randint(0, max(0, inset_max - 1))
    r0, c0 = inset, inset
    r1, c1 = h - 1 - inset, w - 1 - inset
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _mask_stairs(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    main_axis_rows = random.choice((True, False))
    reverse = random.choice((True, False))
    if main_axis_rows:
        if h == 1:
            cols = [0]
        else:
            cols = np.linspace(0, w - 1, h).round().astype(int).tolist()
        for r, c in enumerate(cols):
            cc = w - 1 - c if reverse else c
            mask[r, cc] = True
            if random.random() < 0.35 and 0 <= cc + 1 < w:
                mask[r, cc + 1] = True
    else:
        if w == 1:
            rows = [0]
        else:
            rows = np.linspace(0, h - 1, w).round().astype(int).tolist()
        for c, r in enumerate(rows):
            rr = h - 1 - r if reverse else r
            mask[rr, c] = True
            if random.random() < 0.35 and 0 <= rr + 1 < h:
                mask[rr + 1, c] = True
    return mask


def _mask_stripes(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    axis_rows = random.choice((True, False))
    offset = random.randint(0, 1)
    if axis_rows:
        mask[offset::2, :] = True
        keep_cols = random.randint(max(2, w // 2), w)
        c0 = random.randint(0, w - keep_cols)
        trimmed = _empty_mask(h, w)
        trimmed[:, c0 : c0 + keep_cols] = True
        mask &= trimmed
    else:
        mask[:, offset::2] = True
        keep_rows = random.randint(max(2, h // 2), h)
        r0 = random.randint(0, h - keep_rows)
        trimmed = _empty_mask(h, w)
        trimmed[r0 : r0 + keep_rows, :] = True
        mask &= trimmed
    return mask


def _mask_diag(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    anti = random.choice((True, False))
    thick = random.choice((1, 2))
    for r in range(h):
        c = int(round(r * (w - 1) / max(1, h - 1)))
        if anti:
            c = w - 1 - c
        for dc in range(thick):
            cc = c + dc
            if 0 <= cc < w:
                mask[r, cc] = True
    return mask


def _mask_blob(h: int, w: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    target = random.randint(max(4, (h * w) // 4), max(4, (2 * h * w) // 3))
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.discard(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                if random.random() < 0.75 or not frontier:
                    mask[nr, nc] = True
                    frontier.add((nr, nc))
                    if int(mask.sum()) >= target:
                        break
    return mask


def _make_pattern_mask(h: int, w: int) -> np.ndarray:
    builders = {
        "block": _mask_block,
        "cross": _mask_cross,
        "ring": _mask_ring,
        "stairs": _mask_stairs,
        "stripes": _mask_stripes,
        "diag": _mask_diag,
        "blob": _mask_blob,
    }
    for _ in range(20):
        mask = builders[random.choice(PATTERN_MODES)](h, w)
        count = int(mask.sum())
        if 4 <= count < h * w:
            return mask
    fallback = _mask_block(h, w)
    if int(fallback.sum()) < 4:
        fallback[:, :] = True
        fallback[0, 0] = False
    return fallback


def _make_inner(h: int, w: int, pattern_color: int, marker_color: int) -> np.ndarray:
    inner = np.full((h, w), BACKGROUND, dtype=int)
    mask = _make_pattern_mask(h, w)
    inner[mask] = pattern_color

    marker_on_pattern = random.choice((True, False))
    if marker_on_pattern and int(mask.sum()) >= 4:
        candidates = list(map(tuple, np.argwhere(mask)))
    else:
        candidates = list(map(tuple, np.argwhere(~mask)))
        if not candidates:
            candidates = list(map(tuple, np.argwhere(mask)))
    mr, mc = random.choice(candidates)
    inner[mr, mc] = marker_color

    pattern_count = int(np.sum(inner == pattern_color))
    assert pattern_count >= 2
    assert int(np.sum(inner == marker_color)) == 1
    return inner


def _frame_patch(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    patch = np.full((h + 2, w + 2), BACKGROUND, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    patch[1:-1, 1:-1] = inner
    return patch


def _sample_pair_shapes() -> tuple[tuple[int, int], tuple[int, int]]:
    h1 = random.randint(*INNER_SIZE_RANGE)
    w1 = random.randint(*INNER_SIZE_RANGE)
    mode = random.choice(("sameish", "expand", "mixed"))
    if mode == "sameish":
        h2 = min(INNER_SIZE_RANGE[1], max(INNER_SIZE_RANGE[0], h1 + random.choice((-1, 0, 1))))
        w2 = min(INNER_SIZE_RANGE[1], max(INNER_SIZE_RANGE[0], w1 + random.choice((-1, 0, 1))))
    elif mode == "expand":
        h2 = random.randint(h1, INNER_SIZE_RANGE[1])
        w2 = random.randint(w1, INNER_SIZE_RANGE[1])
        if h2 == h1 and w2 == w1:
            if h2 < INNER_SIZE_RANGE[1]:
                h2 += 1
            elif w2 < INNER_SIZE_RANGE[1]:
                w2 += 1
    else:
        h2 = random.randint(*INNER_SIZE_RANGE)
        w2 = random.randint(*INNER_SIZE_RANGE)
        if (h2 >= h1 and w2 >= w1) or (h2 <= h1 and w2 <= w1):
            if random.choice((True, False)):
                h2 = random.randint(h1, INNER_SIZE_RANGE[1])
                w2 = random.randint(INNER_SIZE_RANGE[0], w1)
            else:
                h2 = random.randint(INNER_SIZE_RANGE[0], h1)
                w2 = random.randint(w1, INNER_SIZE_RANGE[1])
    return (h1, w1), (h2, w2)


def _sample_canvas_shape(patches: list[np.ndarray]) -> tuple[int, int]:
    padded_area = sum((patch.shape[0] + 2 * PLACEMENT_PADDING) * (patch.shape[1] + 2 * PLACEMENT_PADDING) for patch in patches)
    max_h = max(patch.shape[0] for patch in patches) + 2
    max_w = max(patch.shape[1] for patch in patches) + 2
    base = int(np.ceil(np.sqrt(padded_area * random.uniform(1.35, 1.75))))
    rows = max(max_h, base + random.randint(-2, 4))
    cols = max(max_w, base + random.randint(-2, 4))
    if random.random() < GRID_NEAR_SQUARE_PROB:
        cols = max(max_w, rows + random.choice(GRID_DELTAS))
    rows = min(GRID_SIDE_RANGE[1], max(GRID_SIDE_RANGE[0], rows))
    cols = min(GRID_SIDE_RANGE[1], max(GRID_SIDE_RANGE[0], cols))
    return rows, cols


def _place_patches(rows: int, cols: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        row_max = rows - ph - 1
        col_max = cols - pw - 1
        if row_max < 1 or col_max < 1:
            return None

        positions = [(r, c) for r in range(1, row_max + 1) for c in range(1, col_max + 1)]
        random.shuffle(positions)

        placed = False
        for r, c in positions:
            rr0 = max(0, r - PLACEMENT_PADDING)
            rr1 = min(rows, r + ph + PLACEMENT_PADDING)
            cc0 = max(0, c - PLACEMENT_PADDING)
            cc1 = min(cols, c + pw + PLACEMENT_PADDING)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r : r + ph, c : c + pw] = patch
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, 2 * pair_count)
    random.shuffle(colors)

    patches: list[np.ndarray] = []
    for pair_idx in range(pair_count):
        color_a = colors[2 * pair_idx]
        color_b = colors[2 * pair_idx + 1]
        shape_a, shape_b = _sample_pair_shapes()
        inner_a = _make_inner(shape_a[0], shape_a[1], color_a, color_b)
        inner_b = _make_inner(shape_b[0], shape_b[1], color_b, color_a)
        patches.append(_frame_patch(inner_a))
        patches.append(_frame_patch(inner_b))

    rows, cols = _sample_canvas_shape(patches)
    grid = _place_patches(rows, cols, patches)
    if grid is None:
        return None

    frames = _extract_frames(grid)
    pairs = _frame_pairs(frames)
    if len(frames) != 2 * pair_count or len(pairs) != pair_count:
        return None
    if not np.any(main(grid) != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
