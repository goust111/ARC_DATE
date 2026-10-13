import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CODE_COLORS = (BLUE, RED, GREEN, YELLOW)
RESERVED = np.array((BLACK, BLUE, RED, GREEN, YELLOW, GREY))


def _transform(crop, code):
    if code == BLUE:
        return np.rot90(crop, -1)
    if code == RED:
        return np.rot90(crop, 1)
    if code == GREEN:
        return np.fliplr(crop)
    if code == YELLOW:
        return np.flipud(crop)
    return crop.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Transform each quadrant object according to the colored center indicator.

    Input structure:
    A black grid is split into four quadrants by a grey center row and a grey
    center column. Around their intersection, the four diagonal cells of the
    3x3 center marker encode the transform for the matching quadrant: blue for
    clockwise rotation, red for counterclockwise rotation, green for horizontal
    flip, and yellow for vertical flip. Each quadrant contains one small
    connected single-color object whose color is not black, grey, or an
    indicator color.

    Rule:
    Read the four diagonal indicator cells. For each quadrant, find the
    non-reserved colored object, crop it to its bounding box, apply the encoded
    rotation or flip around that bounding-box center, and paste the transformed
    pixels back with the same center position. The grey cross and the indicator
    are removed, so all other cells become black.

    Key insight:
    The central 3x3 marker is not part of any object; its four corner colors
    specify which geometric transform to apply independently in the four
    quadrants.
    """
    arr = np.asarray(grid)
    h, w = arr.shape
    mid_r, mid_c = h // 2, w // 2
    out = np.zeros_like(arr)

    quadrants = (
        (slice(0, mid_r), slice(0, mid_c), -1, -1),
        (slice(0, mid_r), slice(mid_c + 1, w), -1, 1),
        (slice(mid_r + 1, h), slice(0, mid_c), 1, -1),
        (slice(mid_r + 1, h), slice(mid_c + 1, w), 1, 1),
    )

    for row_slice, col_slice, indicator_dr, indicator_dc in quadrants:
        code = int(arr[mid_r + indicator_dr, mid_c + indicator_dc])
        block = arr[row_slice, col_slice]
        object_mask = ~np.isin(block, RESERVED)
        if not np.any(object_mask):
            continue

        rows, cols = np.where(object_mask)
        local_top, local_bottom = int(rows.min()), int(rows.max())
        local_left, local_right = int(cols.min()), int(cols.max())
        crop_values = block[local_top : local_bottom + 1, local_left : local_right + 1]
        crop_mask = object_mask[local_top : local_bottom + 1, local_left : local_right + 1]
        crop = np.where(crop_mask, crop_values, BLACK)
        changed = _transform(crop, code)

        abs_top = (row_slice.start or 0) + local_top
        abs_bottom = (row_slice.start or 0) + local_bottom
        abs_left = (col_slice.start or 0) + local_left
        abs_right = (col_slice.start or 0) + local_right
        center2_r = abs_top + abs_bottom
        center2_c = abs_left + abs_right
        new_h, new_w = changed.shape
        paste_top = (center2_r - (new_h - 1)) // 2
        paste_left = (center2_c - (new_w - 1)) // 2

        rr, cc = np.where(changed != BLACK)
        for r, c in zip(rr, cc):
            out_r = paste_top + int(r)
            out_c = paste_left + int(c)
            if 0 <= out_r < h and 0 <= out_c < w:
                out[out_r, out_c] = changed[r, c]

    return out


def _walk_path(cells, start, target):
    r, c = start
    tr, tc = target
    cells.add((r, c))
    while (r, c) != (tr, tc):
        if r != tr and c != tc:
            move_row = bool(random.getrandbits(1))
        else:
            move_row = r != tr
        if move_row:
            r += 1 if tr > r else -1
        else:
            c += 1 if tc > c else -1
        cells.add((r, c))
    return r, c


def _random_connected_shape(height, width, code):
    last_mask = np.ones((height, width), dtype=bool)
    for _ in range(80):
        start = (random.randrange(height), random.randrange(width))
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        random.shuffle(anchors)
        cells = {start}
        current = start
        for target in anchors:
            current = _walk_path(cells, current, target)

        max_cells = min(height * width - 1, max(len(cells), int(height * width * 0.7)))
        if max_cells < len(cells):
            max_cells = len(cells)
        desired = random.randint(len(cells), max_cells)
        while len(cells) < desired:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        last_mask = mask
        transformed = _transform(mask.astype(int), code)
        if not np.array_equal(transformed, mask.astype(int)):
            return mask
    return last_mask


def _placement_options(r0, r1, c0, c1, height, width, code):
    if code in (BLUE, RED):
        new_height, new_width = width, height
    else:
        new_height, new_width = height, width

    tops = []
    for top in range(r0, r1 - height + 2):
        center2 = 2 * top + height - 1
        new_top2 = center2 - (new_height - 1)
        if new_top2 % 2:
            continue
        new_top = new_top2 // 2
        if r0 <= new_top and new_top + new_height - 1 <= r1:
            tops.append(top)

    lefts = []
    for left in range(c0, c1 - width + 2):
        center2 = 2 * left + width - 1
        new_left2 = center2 - (new_width - 1)
        if new_left2 % 2:
            continue
        new_left = new_left2 // 2
        if c0 <= new_left and new_left + new_width - 1 <= c1:
            lefts.append(left)

    return [(top, left) for top in tops for left in lefts]


def _choose_box(r0, r1, c0, c1, code):
    max_height = min(7, r1 - r0 + 1)
    max_width = min(7, c1 - c0 + 1)
    candidates = []
    for height in range(3, max_height + 1):
        for width in range(3, max_width + 1):
            if (height - width) % 2:
                continue
            options = _placement_options(r0, r1, c0, c1, height, width, code)
            if options:
                candidates.append((height, width, options))
    height, width, options = random.choice(candidates)
    top, left = random.choice(options)
    return top, left, height, width


def generate_input() -> np.ndarray:
    height = random.randrange(15, 30, 2)
    width = random.randrange(15, 30, 2)
    mid_r, mid_c = height // 2, width // 2
    grid = np.zeros((height, width), dtype=int)

    grid[mid_r, :] = GREY
    grid[:, mid_c] = GREY

    codes = list(CODE_COLORS)
    random.shuffle(codes)
    quadrants = (
        ((0, mid_r - 2, 0, mid_c - 2), (-1, -1)),
        ((0, mid_r - 2, mid_c + 2, width - 1), (-1, 1)),
        ((mid_r + 2, height - 1, 0, mid_c - 2), (1, -1)),
        ((mid_r + 2, height - 1, mid_c + 2, width - 1), (1, 1)),
    )

    for code, (_, (dr, dc)) in zip(codes, quadrants):
        grid[mid_r + dr, mid_c + dc] = code

    object_colors = random.sample((6, 7, 8, 9), 4)
    for code, ((r0, r1, c0, c1), _), color in zip(codes, quadrants, object_colors):
        top, left, obj_h, obj_w = _choose_box(r0, r1, c0, c1, code)
        shape = _random_connected_shape(obj_h, obj_w, code)
        rr, cc = np.where(shape)
        grid[top + rr, left + cc] = color

    return grid
