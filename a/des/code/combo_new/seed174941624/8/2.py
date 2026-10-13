from __future__ import annotations

import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _scan_order_colors(grid: np.ndarray, bg: int) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in map(int, grid.reshape(-1)):
        if value == bg or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _mask_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, np.argwhere(mask)))


def _rectangle_options(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    coords_a = _mask_coords(mask_a)
    h1, w1 = map(int, mask_a.shape)
    options: set[tuple[int, int]] = set()

    for rot in range(4):
        rot_b = np.rot90(mask_b, rot)
        coords_b = list(map(tuple, np.argwhere(rot_b)))
        h2, w2 = map(int, rot_b.shape)

        for dr in range(-h2, h1 + 1):
            for dc in range(-w2, w1 + 1):
                shifted = {(r + dr, c + dc) for r, c in coords_b}
                if coords_a & shifted:
                    continue

                union = coords_a | shifted
                rows = [r for r, _ in union]
                cols = [c for _, c in union]
                out_h = max(rows) - min(rows) + 1
                out_w = max(cols) - min(cols) + 1
                if len(union) == out_h * out_w:
                    options.add((out_h, out_w))

    return sorted(options)


def _assemble_rectangle(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    options = _rectangle_options(mask_a, mask_b)
    assert options, "same-color fragments must assemble into a rectangle"
    return options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color L fragments into rectangles and output them in ascending area order.

    Input structure:
    - A dominant background surrounds several monochrome 4-connected L-shaped fragments.
    - Each non-background color appears in exactly two disconnected fragments.
    - The two fragments of one color can be combined, after rotating one fragment, into one solid axis-aligned rectangle.

    Rule:
    - For each non-background color, isolate its two connected components.
    - Keep one fragment fixed, rotate the other by 0/90/180/270 degrees, and find a non-overlapping placement whose union is a filled rectangle.
    - Record the reconstructed rectangle for that color, then sort all such rectangles by area from small to large.
    - Draw the sorted solid rectangles from left to right with one background column between neighbors and align all rectangle bottoms.

    Key insight:
    - The scattered orientations and positions are irrelevant; the task is to detect per-color complementary L shapes, recover the full rectangle they tile, and only then notice that the output order is determined by rectangle area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    rectangles: list[tuple[int, int, int, int]] = []

    for color in _scan_order_colors(g, bg):
        components = _find_components(g == color)
        assert len(components) == 2, "each foreground color must have exactly two pieces"
        components.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))

        mask_a = _coords_to_mask(components[0])
        mask_b = _coords_to_mask(components[1])
        rect_h, rect_w = _assemble_rectangle(mask_a, mask_b)
        rectangles.append((rect_h * rect_w, rect_h, rect_w, color))

    rectangles.sort(key=lambda item: item[0])
    out_h = max(h for _, h, _w, _color in rectangles)
    out_w = sum(w for _area, _h, w, _color in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for _area, rect_h, rect_w, color in rectangles:
        out[out_h - rect_h :, col : col + rect_w] = color
        col += rect_w + 1

    return out


# --------------- generator (o5x) ---------------

COLOR_VALUES = tuple(range(10))
PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 6
PIECE_WIDTH_MAX = 5
CANVAS_SIDE_MIN = 16
CANVAS_SIDE_MAX = 30
MAX_GENERATION_ATTEMPTS = 300


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paint_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _build_l_pair() -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    rect_h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
    if rect_h < 2:
        return None

    for _ in range(80):
        width_a = random.randint(2, PIECE_WIDTH_MAX)
        width_b = random.randint(2, PIECE_WIDTH_MAX)
        overlap = random.randint(1, min(width_a, width_b) - 1)
        rect_w = width_a + width_b - overlap
        if not (3 <= rect_w <= RECT_SIDE_MAX):
            continue

        notch_h = random.randint(1, rect_h - 1)
        mask_a = np.ones((rect_h, width_a), dtype=bool)
        mask_b = np.ones((rect_h, width_b), dtype=bool)
        mask_a[:notch_h, width_a - overlap :] = False
        mask_b[notch_h:, :overlap] = False

        options = _rectangle_options(mask_a, mask_b)
        if options != [(rect_h, rect_w)]:
            continue
        return mask_a, mask_b, (rect_h, rect_w)

    return None


def _sample_pairs(n_pairs: int, bg: int) -> list[tuple[int, list[np.ndarray]]] | None:
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_pairs)
    used_areas: set[int] = set()
    pairs: list[tuple[int, list[np.ndarray]]] = []

    for color in colors:
        for _ in range(80):
            built = _build_l_pair()
            if built is None:
                continue
            mask_a, mask_b, dims = built
            area = dims[0] * dims[1]
            if area not in used_areas:
                break
        else:
            return None
        used_areas.add(area)

        patch_a = _paint_mask(np.rot90(mask_a, random.randrange(4)), color, bg)
        patch_b = _paint_mask(np.rot90(mask_b, random.randrange(4)), color, bg)
        pairs.append((color, [patch_a, patch_b]))

    return pairs


def _place_patches(
    patches: list[np.ndarray],
    bg: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    used_boxes = np.zeros((height, width), dtype=bool)

    for patch in sorted(patches, key=lambda arr: int((arr != bg).sum()), reverse=True):
        ph, pw = map(int, patch.shape)
        blocked = _dilate8(used_boxes)
        candidates = [
            (r0, c0)
            for r0 in range(height - ph + 1)
            for c0 in range(width - pw + 1)
            if not bool(blocked[r0 : r0 + ph, c0 : c0 + pw].any())
        ]
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        mask = patch != bg
        window = grid[r0 : r0 + ph, c0 : c0 + pw]
        window[mask] = patch[mask]
        used_boxes[r0 : r0 + ph, c0 : c0 + pw] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_VALUES)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    sampled = _sample_pairs(n_pairs, bg)
    if sampled is None:
        return None

    patches = [patch for _color, pair in sampled for patch in pair]
    total_fg = sum(int((patch != bg).sum()) for patch in patches)
    max_patch_h = max(int(patch.shape[0]) for patch in patches)
    max_patch_w = max(int(patch.shape[1]) for patch in patches)

    min_h = max(CANVAS_SIDE_MIN, max_patch_h + 2)
    min_w = max(CANVAS_SIDE_MIN, max_patch_w + 2)
    if min_h > CANVAS_SIDE_MAX or min_w > CANVAS_SIDE_MAX:
        return None

    height = random.randint(min_h, CANVAS_SIDE_MAX)
    width = random.randint(min_w, CANVAS_SIDE_MAX)
    if height * width - total_fg <= total_fg:
        return None

    grid = _place_patches(patches, bg, height, width)
    if grid is None:
        return None
    if get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_scan_order_colors(grid, bg)) != n_pairs:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
