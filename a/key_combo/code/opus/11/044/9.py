import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREY = 5


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

GENERATION_RETRIES = 400
GRID_SIDE_RANGE = (9, 12)
SPRITE_AREA_RANGE = (3, 5)
SPRITE_BOX_RANGE = (2, 4)
SCALE_VALUES = (2, 3)
ROTATION_VALUES = (1, 2, 3)


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _apply_transform(sprite: np.ndarray, scale_factor: int, rotation_count: int) -> np.ndarray:
    scaled = _scale_sprite(sprite, int(scale_factor))
    rotated = np.rot90(scaled, k=(-int(rotation_count)) % 4).copy()
    coords = np.argwhere(rotated != Color.BLACK)
    box = _bbox_from_coords(coords)
    return _crop_bbox(rotated, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the grey sprite and the two colored control bars, then output the sprite after scaling and clockwise rotation.

    Input structure:
    - The grid uses black background.
    - There is exactly one grey 4-connected sprite, one red horizontal control bar,
      and one blue vertical control bar.
    - The control bars are separated from the sprite and from each other; bar length
      is the only parameter source.

    Rule:
    - Crop the grey sprite to its tight bounding box.
    - The red horizontal bar length is the scaling factor.
    - The blue vertical bar length is the number of clockwise 90-degree rotations.
    - Apply the red instruction first: expand each sprite pixel into a square block
      of that size.
    - Apply the blue instruction second: rotate the scaled sprite clockwise by the
      encoded number of quarter-turns.
    - Return the smallest black rectangle that contains the transformed grey sprite.

    Key insight:
    - The bars are not decorations: their orientation tells which operation they
      encode, and their lengths supply the transformation parameters for the grey sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_comps = _components_of_color(g, Color.GREY)
    red_comps = _components_of_color(g, Color.RED)
    blue_comps = _components_of_color(g, Color.BLUE)
    assert len(grey_comps) == 1, "expected exactly one grey sprite"
    assert len(red_comps) == 1, "expected exactly one red bar"
    assert len(blue_comps) == 1, "expected exactly one blue bar"

    sprite_coords = grey_comps[0]
    red_coords = red_comps[0]
    blue_coords = blue_comps[0]

    sprite_box = _bbox_from_coords(sprite_coords)
    red_box = _bbox_from_coords(red_coords)
    blue_box = _bbox_from_coords(blue_coords)

    red_len = int(red_coords.shape[0])
    blue_len = int(blue_coords.shape[0])

    assert 3 <= int(sprite_coords.shape[0]) <= 5, "sprite must have 3-5 pixels"
    assert red_box[0] == red_box[2], "red control must be horizontal"
    assert blue_box[1] == blue_box[3], "blue control must be vertical"
    assert 2 <= red_len <= 3, "red bar length must encode scale 2 or 3"
    assert 1 <= blue_len <= 3, "blue bar length must encode 1-3 quarter-turns"

    red_cols = sorted(int(c) for c in red_coords[:, 1])
    blue_rows = sorted(int(r) for r in blue_coords[:, 0])
    assert red_cols == list(range(red_cols[0], red_cols[0] + red_len)), "red bar must be contiguous"
    assert blue_rows == list(range(blue_rows[0], blue_rows[0] + blue_len)), "blue bar must be contiguous"

    sprite = _crop_bbox(g, sprite_box)
    return _apply_transform(sprite, red_len, blue_len)


def _touching_or_overlapping(a: np.ndarray, b: np.ndarray) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if max(abs(int(ar) - int(br)), abs(int(ac) - int(bc))) <= 1:
                return True
    return False


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier: set[Coord] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    box = _bbox_from_coords(np.argwhere(mask))
    return _crop_bbox(mask.astype(int), box).astype(bool)


def _is_asymmetric(mask: np.ndarray) -> bool:
    for k in range(1, 4):
        if np.array_equal(mask, np.rot90(mask, k=k)):
            return False
    for k in range(4):
        if np.array_equal(mask, np.fliplr(np.rot90(mask, k=k))):
            return False
        if np.array_equal(mask, np.flipud(np.rot90(mask, k=k))):
            return False
    return True


def _random_sprite() -> np.ndarray:
    for _ in range(200):
        height = random.randint(*SPRITE_BOX_RANGE)
        width = random.randint(*SPRITE_BOX_RANGE)
        area = random.randint(*SPRITE_AREA_RANGE)
        if area > height * width:
            continue
        mask = _random_connected_mask(height, width, area)
        if mask is None or mask.sum() != area:
            continue
        if not _is_asymmetric(mask):
            continue
        sprite = np.zeros(mask.shape, dtype=int)
        sprite[mask] = Color.GREY
        return sprite
    raise RuntimeError("failed to sample an asymmetric sprite")


def _choose_red_start(c0: int, c1: int, width: int, bar_len: int) -> int | None:
    lo = max(0, c0 - bar_len + 1)
    hi = min(width - bar_len, c1)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _choose_blue_start_for_above(r0: int, r1: int, height: int, bar_len: int) -> int | None:
    lo = r0
    hi = min(r1, height - bar_len)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _choose_blue_start_for_below(r0: int, r1: int, height: int, bar_len: int) -> int | None:
    lo = max(0, r0 - bar_len + 1)
    hi = min(r1 - bar_len + 1, height - bar_len)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_candidate_grid() -> np.ndarray | None:
    sprite = _random_sprite()
    sh, sw = sprite.shape
    scale_factor = random.choice(SCALE_VALUES)
    rotation_count = random.choice(ROTATION_VALUES)

    out = _apply_transform(sprite, scale_factor, rotation_count)
    if max(out.shape) > 30:
        return None

    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    if height < sh + 5 or width < sw + 5:
        return None

    sprite_r0 = random.randint(2, height - sh - 3)
    sprite_c0 = random.randint(2, width - sw - 3)
    sprite_r1 = sprite_r0 + sh - 1
    sprite_c1 = sprite_c0 + sw - 1

    red_side = random.choice(("above", "below"))
    blue_side = random.choice(("left", "right"))

    red_row = sprite_r0 - 2 if red_side == "above" else sprite_r1 + 2
    blue_col = sprite_c0 - 2 if blue_side == "left" else sprite_c1 + 2
    if not (0 <= red_row < height and 0 <= blue_col < width):
        return None

    red_col = _choose_red_start(sprite_c0, sprite_c1, width, scale_factor)
    if red_col is None:
        return None
    if red_side == "above":
        blue_row = _choose_blue_start_for_above(sprite_r0, sprite_r1, height, rotation_count)
    else:
        blue_row = _choose_blue_start_for_below(sprite_r0, sprite_r1, height, rotation_count)
    if blue_row is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sw] = sprite
    grid[red_row, red_col : red_col + scale_factor] = Color.RED
    grid[blue_row : blue_row + rotation_count, blue_col] = Color.BLUE
    return grid


def _infer_structure(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    grey_comps = _components_of_color(grid, Color.GREY)
    red_comps = _components_of_color(grid, Color.RED)
    blue_comps = _components_of_color(grid, Color.BLUE)
    if len(grey_comps) != 1 or len(red_comps) != 1 or len(blue_comps) != 1:
        return None
    return grey_comps[0], red_comps[0], blue_comps[0]


def _validate_example(grid: np.ndarray) -> bool:
    parsed = _infer_structure(grid)
    if parsed is None:
        return False
    sprite_coords, red_coords, blue_coords = parsed

    if _touching_or_overlapping(sprite_coords, red_coords):
        return False
    if _touching_or_overlapping(sprite_coords, blue_coords):
        return False
    if _touching_or_overlapping(red_coords, blue_coords):
        return False

    sprite_box = _bbox_from_coords(sprite_coords)
    red_box = _bbox_from_coords(red_coords)
    blue_box = _bbox_from_coords(blue_coords)
    red_len = int(red_coords.shape[0])
    blue_len = int(blue_coords.shape[0])

    if red_box[0] != red_box[2] or blue_box[1] != blue_box[3]:
        return False
    if red_len not in SCALE_VALUES or blue_len not in ROTATION_VALUES:
        return False
    if not (red_box[0] < sprite_box[0] or red_box[0] > sprite_box[2]):
        return False
    if not (blue_box[1] < sprite_box[1] or blue_box[1] > sprite_box[3]):
        return False

    sprite = _crop_bbox(grid, sprite_box)
    sprite_mask = sprite == Color.GREY
    if not _is_asymmetric(sprite_mask):
        return False

    expected = _apply_transform(sprite, red_len, blue_len)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    alt_matches = 0
    for alt_scale in SCALE_VALUES:
        for alt_rot in ROTATION_VALUES:
            candidate = _apply_transform(sprite, alt_scale, alt_rot)
            if np.array_equal(candidate, expected):
                alt_matches += 1
    if alt_matches != 1:
        return False

    if np.array_equal(expected, sprite):
        return False
    if np.array_equal(expected, _apply_transform(sprite, red_len, 0)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_candidate_grid()
        if grid is not None and _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
