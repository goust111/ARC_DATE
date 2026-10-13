import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mirror_bbox_h(box, width: int):
    r0, c0, r1, c1 = box
    return int(r0), int(width - 1 - c1), int(r1), int(width - 1 - c0)


def _paint_symmetry(blue_coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    if blue_coords.size == 0:
        return out
    rs = blue_coords[:, 0]
    cs = blue_coords[:, 1]
    out[rs, cs] = BLUE
    out[rs, w - 1 - cs] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the left-half blue scaffold into a four-way symmetric cropped canvas and copy the red markers into the new right quadrants.

    Input structure:
    - The grid has a black background, blue structure pixels in the upper-left and lower-left quadrants, and red marker pixels inside those quadrants' blue bounding boxes.
    - The upper-left and lower-left blue structures are vertical mirror copies across the grid center row.
    - The right half is empty, and red markers avoid the center axes.

    Rule:
    - Mirror every blue pixel across the grid center column to create the missing right half.
    - Take the minimal bounding box of the completed blue figure; that completed box is the output canvas.
    - For each red marker, identify whether it belongs to the upper-left or lower-left quadrant, measure its distances to that quadrant's blue bounding-box edges, and place a red copy in the mirrored right quadrant with top/bottom distances preserved and left/right distances swapped.
    - Keep the original red markers and render the completed blue figure on the cropped output canvas.

    Key insight:
    - The blue completion defines both the output canvas and the right-side reference boxes, so the red copies must be placed in the completed symmetric crop rather than in the original input frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "expected odd dimensions"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED}

    mid_r = h // 2
    mid_c = w // 2
    assert not np.any(g[:, mid_c + 1 :] == RED) and not np.any(g[:, mid_c + 1 :] == BLUE)
    assert not np.any(g[mid_r, :] == RED)
    assert not np.any(g[:, mid_c] == RED)

    blue = g == BLUE
    blue_coords = np.argwhere(blue)
    assert blue_coords.size > 0, "expected blue structure pixels"

    top_bbox = _bbox(blue[:mid_r, :mid_c])
    bottom_bbox_local = _bbox(blue[mid_r + 1 :, :mid_c])
    assert top_bbox is not None and bottom_bbox_local is not None, "expected blue structure in both left quadrants"
    br0, bc0, br1, bc1 = bottom_bbox_local
    bottom_bbox = (br0 + mid_r + 1, bc0, br1 + mid_r + 1, bc1)

    full_blue = _paint_symmetry(blue_coords, (h, w))
    full_bbox = _bbox(full_blue == BLUE)
    assert full_bbox is not None, "completed blue figure must be non-empty"

    full = full_blue.copy()
    red_coords = np.argwhere(g == RED)
    for r, c in red_coords:
        r = int(r)
        c = int(c)
        assert c < mid_c, "red markers must start in the left half"
        source_bbox = top_bbox if r < mid_r else bottom_bbox
        sr0, sc0, sr1, sc1 = source_bbox
        assert sr0 <= r <= sr1 and sc0 <= c <= sc1, "marker must lie inside its quadrant bbox"

        d_top = r - sr0
        d_bottom = sr1 - r
        d_left = c - sc0
        d_right = sc1 - c

        target_bbox = _mirror_bbox_h(source_bbox, w)
        tr0, tc0, tr1, tc1 = target_bbox
        rr = tr0 + d_top
        cc = tc0 + d_right
        assert rr == tr1 - d_bottom and cc == tc1 - d_left

        full[r, c] = RED
        full[rr, cc] = RED

    return _crop(full, full_bbox)


def _make_top_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if rng.random() < 0.8:
        mask[:, rng.randrange(1, width - 1)] = True
    if rng.random() < 0.8:
        mask[rng.randrange(1, height - 1), :] = True

    if rng.random() < 0.45 and width >= 6:
        row = rng.randrange(1, height - 1)
        end = rng.randrange(2, width - 1)
        mask[row, :end] = True
    if rng.random() < 0.45 and height >= 6:
        col = rng.randrange(1, width - 1)
        end = rng.randrange(2, height - 1)
        mask[:end, col] = True

    return mask


def _pick_marker_split(total: int, top_cap: int, bottom_cap: int, rng: random.Random):
    modes = []
    if top_cap >= total:
        modes.append("top")
    if bottom_cap >= total:
        modes.append("bottom")
    if top_cap >= 1 and bottom_cap >= 1:
        modes.append("both")
    if not modes:
        return None

    mode = rng.choice(modes)
    if mode == "top":
        return total, 0
    if mode == "bottom":
        return 0, total

    lo = max(1, total - bottom_cap)
    hi = min(top_cap, total - 1)
    if lo > hi:
        return None
    top_count = rng.randint(lo, hi)
    return top_count, total - top_count


def _build_example(rng: random.Random):
    side = rng.choice([13, 15, 17, 19, 21, 23])
    mid = side // 2

    max_box = min(8, mid - 1)
    if max_box < 5:
        return None
    height = rng.randint(5, max_box)
    width = rng.randint(5, max_box)

    max_r0 = mid - height - 1
    max_c0 = mid - width - 1
    if max_r0 < 1 or max_c0 < 1:
        return None
    r0 = rng.randint(1, max_r0)
    c0 = rng.randint(1, max_c0)
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    top_mask = _make_top_mask(height, width, rng)
    interior_empty = [
        (rr, cc)
        for rr in range(1, height - 1)
        for cc in range(1, width - 1)
        if not top_mask[rr, cc]
    ]
    if len(interior_empty) < 3:
        return None

    grid = np.full((side, side), BLACK, dtype=int)
    top_blue_abs = []
    for rr in range(height):
        for cc in range(width):
            if not top_mask[rr, cc]:
                continue
            ar = r0 + rr
            ac = c0 + cc
            grid[ar, ac] = BLUE
            top_blue_abs.append((ar, ac))

    for ar, ac in top_blue_abs:
        grid[side - 1 - ar, ac] = BLUE

    top_candidates = [(r0 + rr, c0 + cc) for rr, cc in interior_empty]
    bottom_candidates = [(side - 1 - (r0 + rr), c0 + cc) for rr, cc in interior_empty]

    total_red = rng.randint(2, 5)
    split = _pick_marker_split(total_red, len(top_candidates), len(bottom_candidates), rng)
    if split is None:
        return None
    top_count, bottom_count = split

    reds = []
    if top_count:
        reds.extend(rng.sample(top_candidates, top_count))
    if bottom_count:
        reds.extend(rng.sample(bottom_candidates, bottom_count))

    for r, c in reds:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    spec = {
        "side": side,
        "top_bbox": (r0, c0, r1, c1),
        "bottom_bbox": (side - 1 - r1, c0, side - 1 - r0, c1),
        "full_bbox": (r0, c0, side - 1 - r0, side - 1 - c0),
        "blue_top": tuple(top_blue_abs),
        "reds": tuple(reds),
    }
    return grid, spec


def _expected_from_spec(spec) -> np.ndarray:
    side = spec["side"]
    full = np.full((side, side), BLACK, dtype=int)

    for r, c in spec["blue_top"]:
        full[r, c] = BLUE
        full[side - 1 - r, c] = BLUE
        full[r, side - 1 - c] = BLUE
        full[side - 1 - r, side - 1 - c] = BLUE

    top_bbox = spec["top_bbox"]
    bottom_bbox = spec["bottom_bbox"]
    for r, c in spec["reds"]:
        source_bbox = top_bbox if r < side // 2 else bottom_bbox
        sr0, sc0, sr1, sc1 = source_bbox
        d_top = r - sr0
        d_bottom = sr1 - r
        d_left = c - sc0
        d_right = sc1 - c
        tr0, tc0, tr1, tc1 = _mirror_bbox_h(source_bbox, side)
        rr = tr0 + d_top
        cc = tc0 + d_right
        assert rr == tr1 - d_bottom and cc == tc1 - d_left
        full[r, c] = RED
        full[rr, cc] = RED

    return _crop(full, spec["full_bbox"])


def _validate_example(grid: np.ndarray, spec) -> bool:
    expected = _expected_from_spec(spec)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if actual.shape == grid.shape:
        return False
    if np.array_equal(actual, _crop(grid, spec["full_bbox"])):
        return False

    side = spec["side"]
    mid = side // 2
    if np.any(grid[:, mid + 1 :] != BLACK):
        return False
    if np.any(grid[side // 2, :] != BLACK):
        return False

    blue_out = actual == BLUE
    if not np.array_equal(blue_out, np.flip(blue_out, axis=0)):
        return False
    if not np.array_equal(blue_out, np.flip(blue_out, axis=1)):
        return False

    red_in = int(np.sum(grid == RED))
    red_out = int(np.sum(actual == RED))
    if red_out != 2 * red_in:
        return False

    wrong_full = np.full_like(grid, BLACK)
    for r, c in spec["blue_top"]:
        wrong_full[r, c] = BLUE
        wrong_full[side - 1 - r, c] = BLUE
        wrong_full[r, side - 1 - c] = BLUE
        wrong_full[side - 1 - r, side - 1 - c] = BLUE
    for r, c in spec["reds"]:
        wrong_full[r, c] = RED
        wrong_full[r, side - 1 - c] = RED
    if wrong_full.shape == actual.shape and np.array_equal(wrong_full, actual):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        built = _build_example(rng)
        if built is None:
            continue
        grid, spec = built
        if _validate_example(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid input")
