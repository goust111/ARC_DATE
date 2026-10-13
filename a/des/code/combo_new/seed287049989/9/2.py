import random
from collections import defaultdict, deque

import numpy as np


BG_COLOR = 0
ARC_LIMIT = 30
TEMP_CANVAS = 30
GENERATION_ATTEMPTS = 300
PAIR_BUILD_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3, 4)
PAD_KEEP_MAX = 4
COLOR_POOL = tuple(range(1, 10))


def _component_masks(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG_COLOR or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append({"color": color, "mask": mask, "bbox": (r0, c0, r1, c1)})

    return comps


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(mask)]


def _find_rectangle_tilings(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    for fixed, moved in ((mask_a, np.rot90(mask_b, 2)), (mask_b, np.rot90(mask_a, 2))):
        fixed_cells = _mask_cells(fixed)
        moved_cells = _mask_cells(moved)
        fixed_set = set(fixed_cells)
        fh, fw = fixed.shape
        mh, mw = moved.shape

        for dr in range(-mh, fh + 1):
            for dc in range(-mw, fw + 1):
                shifted = [(r + dr, c + dc) for r, c in moved_cells]
                if any(cell in fixed_set for cell in shifted):
                    continue

                all_cells = fixed_cells + shifted
                rows = [r for r, _ in all_cells]
                cols = [c for _, c in all_cells]
                out_h = max(rows) - min(rows) + 1
                out_w = max(cols) - min(cols) + 1
                if out_h * out_w == len(all_cells):
                    candidates.append((out_h, out_w))

    return candidates


def _solve_pair_dims(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    dims = sorted(set(_find_rectangle_tilings(mask_a, mask_b)))
    assert len(dims) == 1, "same-color pair must complete one rectangle"
    return dims[0]


def _l_shape_mask(h: int, w: int, notch_h: int, notch_w: int, corner: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if corner == 0:
        mask[:notch_h, :notch_w] = False
    elif corner == 1:
        mask[:notch_h, w - notch_w :] = False
    elif corner == 2:
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored L-shaped fragments, reconstruct their rectangles, and pack them by area.

    Input structure:
    - The background is black.
    - Each non-black connected component is one monochrome L-shape: a solid rectangle with one rectangular corner removed.
    - Every foreground color appears in exactly two separate fragments, and each same-colored pair can form one full rectangle after a 180-degree rotation of one fragment.

    Rule:
    - Find all monochrome connected components and group them by color.
    - For each color pair, rotate one fragment by 180 degrees and search the unique non-overlapping translation where the two masks fill a solid rectangle.
    - Record the completed rectangle's height, width, and area.
    - Output filled rectangles of the original colors sorted by increasing area, laid out left-to-right with one black separator column and bottom alignment.

    Key insight:
    - The fragments are matched by complementary missing corners, not by their original positions: the only reliable invariant is the rectangle they can complete after a 180-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    comps = _component_masks(g)
    by_color: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp["mask"])

    rects: list[tuple[int, int, int, int]] = []
    for color, masks in by_color.items():
        assert len(masks) == 2, "each color must appear on exactly two fragments"
        out_h, out_w = _solve_pair_dims(masks[0], masks[1])
        rects.append((out_h * out_w, out_h, out_w, color))

    rects.sort(key=lambda item: item[0])
    out_h = max(h for _, h, _, _ in rects)
    out_w = sum(w for _, _, w, _ in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG_COLOR, dtype=int)

    c = 0
    for _, h, w, color in rects:
        out[out_h - h :, c : c + w] = color
        c += w + 1

    return out


def _sample_pair_masks(pair_count: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    max_common = {2: 8, 3: 7, 4: 6}[pair_count]
    max_outer = {2: 7, 3: 6, 4: 5}[pair_count]

    for _ in range(PAIR_BUILD_ATTEMPTS):
        family = random.choice(("vertical", "horizontal"))
        corner = random.randrange(4)

        if family == "vertical":
            full_w = random.randint(3, max_common)
            h1 = random.randint(2, max_outer)
            h2 = random.randint(2, max_outer)
            notch_h = random.randint(1, min(h1, h2) - 1)
            notch_w1 = random.randint(1, full_w - 1)
            notch_w2 = full_w - notch_w1
            mask_a = _l_shape_mask(h1, full_w, notch_h, notch_w1, corner)
            mask_b = _l_shape_mask(h2, full_w, notch_h, notch_w2, corner)
        else:
            full_h = random.randint(3, max_common)
            w1 = random.randint(2, max_outer)
            w2 = random.randint(2, max_outer)
            notch_w = random.randint(1, min(w1, w2) - 1)
            notch_h1 = random.randint(1, full_h - 1)
            notch_h2 = full_h - notch_h1
            mask_a = _l_shape_mask(full_h, w1, notch_h1, notch_w, corner)
            mask_b = _l_shape_mask(full_h, w2, notch_h2, notch_w, corner)

        turns = random.randrange(4)
        mask_a = np.rot90(mask_a, turns).copy()
        mask_b = np.rot90(mask_b, turns).copy()
        candidates = _find_rectangle_tilings(mask_a, mask_b)
        dims = sorted(set(candidates))
        if len(dims) != 1 or len(candidates) > 2:
            continue

        out_h, out_w = dims[0]
        if out_h * out_w < 6 or max(out_h, out_w) > 12:
            continue
        return mask_a, mask_b, (out_h, out_w)

    return None


def _can_place(mask: np.ndarray, blocked: np.ndarray, top: int, left: int) -> bool:
    rows, cols = np.where(mask)
    return not np.any(blocked[top + rows, left + cols])


def _place_piece(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = mask.shape
    positions = [(r, c) for r in range(TEMP_CANVAS - h + 1) for c in range(TEMP_CANVAS - w + 1)]
    random.shuffle(positions)

    for top, left in positions[:PLACEMENT_ATTEMPTS]:
        if not _can_place(mask, blocked, top, left):
            continue

        rows, cols = np.where(mask)
        abs_rows = top + rows
        abs_cols = left + cols
        grid[abs_rows, abs_cols] = color
        for rr, cc in zip(abs_rows, abs_cols):
            blocked[max(0, rr - 1) : min(TEMP_CANVAS, rr + 2), max(0, cc - 1) : min(TEMP_CANVAS, cc + 2)] = True
        return True

    return False


def _crop_with_padding(grid: np.ndarray) -> np.ndarray:
    rows, cols = np.where(grid != BG_COLOR)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    keep_top = random.randint(0, min(PAD_KEEP_MAX, r0))
    keep_bottom = random.randint(0, min(PAD_KEEP_MAX, TEMP_CANVAS - 1 - r1))
    keep_left = random.randint(0, min(PAD_KEEP_MAX, c0))
    keep_right = random.randint(0, min(PAD_KEEP_MAX, TEMP_CANVAS - 1 - c1))

    return grid[r0 - keep_top : r1 + keep_bottom + 1, c0 - keep_left : c1 + keep_right + 1]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, pair_count)
    pieces: list[tuple[np.ndarray, int]] = []
    output_rects: list[tuple[int, int]] = []
    used_areas: set[int] = set()

    for color in colors:
        sampled = _sample_pair_masks(pair_count)
        if sampled is None:
            return None
        mask_a, mask_b, out_shape = sampled
        area = out_shape[0] * out_shape[1]
        if area in used_areas:
            return None
        used_areas.add(area)
        output_rects.append(out_shape)
        pieces.append((mask_a, color))
        pieces.append((mask_b, color))

    packed_h = max(h for h, _ in output_rects)
    packed_w = sum(w for _, w in output_rects) + pair_count - 1
    if packed_h > ARC_LIMIT or packed_w > ARC_LIMIT:
        return None

    grid = np.full((TEMP_CANVAS, TEMP_CANVAS), BG_COLOR, dtype=int)
    blocked = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=bool)
    random.shuffle(pieces)
    for mask, color in pieces:
        if not _place_piece(grid, blocked, mask, color):
            return None

    return _crop_with_padding(grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid L-fragment puzzle")
