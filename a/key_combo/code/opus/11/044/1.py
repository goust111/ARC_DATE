from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

PANEL_W = 5
GRID_H_MIN = 6
GRID_H_MAX = 10
DIVIDER_COUNT = 2
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _has_symmetry(mask: np.ndarray) -> bool:
    variants = (
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 2),
        mask.T,
        np.fliplr(mask).T,
    )
    return any(np.array_equal(mask, variant) for variant in variants)


def _build_sprite_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    for chosen in itertools.combinations(range(9), 5):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in chosen:
            mask[idx // 3, idx % 3] = True
        if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue
        if not _is_connected(mask):
            continue
        if _has_symmetry(mask):
            continue
        sprite = np.zeros((3, 3), dtype=int)
        sprite[mask] = RED
        library.append(sprite)
    assert library, "sprite library must be non-empty"
    return library


SPRITE_LIBRARY = _build_sprite_library()


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    divider_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(divider_cols) == DIVIDER_COUNT, "expected exactly two full-height grey divider columns"
    d1, d2 = divider_cols
    assert 0 < d1 < d2 < w - 1, "divider columns must split three non-empty panels"

    left_w = d1
    middle_w = d2 - d1 - 1
    right_w = w - d2 - 1
    assert left_w == middle_w == right_w, "three panels must have equal width"

    left = g[:, :d1]
    middle = g[:, d1 + 1 : d2]
    right = g[:, d2 + 1 :]
    panel_shape = left.shape
    assert middle.shape == panel_shape == right.shape

    assert set(np.unique(left)).issubset({BLACK, BLUE}), "left panel must be a blue/black mask"
    assert set(np.unique(middle)).issubset({BLACK, BLUE}), "middle panel must be a blue/black mask"
    assert set(np.unique(right)).issubset({BLACK, RED}), "right panel must contain only background and red sprite"

    red_mask = right == RED
    assert red_mask.any(), "right panel must contain a red sprite"
    assert _is_connected(red_mask), "red sprite must be a single 4-connected component"
    red_coords = np.argwhere(red_mask)
    assert 3 <= len(red_coords) <= 5, "sprite must contain 3 to 5 red cells"

    sr0, sc0, sr1, sc1 = _bbox_from_coords(red_coords)
    sprite = right[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    sprite_h, sprite_w = sprite.shape
    assert sprite_h <= 3 and sprite_w <= 3, "sprite bbox must fit in 3x3"
    assert sprite_h % 2 == 1 and sprite_w % 2 == 1, "sprite bbox must have a unique center cell"

    hits = np.argwhere((left == BLUE) & (middle == BLUE))
    return {
        "grid": g,
        "panel_shape": panel_shape,
        "left": left,
        "middle": middle,
        "right": right,
        "hits": hits,
        "sprite": sprite,
        "sprite_bbox": (sr0, sc0, sr1, sc1),
        "sprite_center": (sprite_h // 2, sprite_w // 2),
    }


def _render_output(panel_shape: tuple[int, int], hits: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    out = np.zeros(panel_shape, dtype=int)
    sprite_h, sprite_w = sprite.shape
    center_r = sprite_h // 2
    center_c = sprite_w // 2

    for hit_r, hit_c in hits:
        top = int(hit_r) - center_r
        left = int(hit_c) - center_c
        assert 0 <= top and 0 <= left
        assert top + sprite_h <= panel_shape[0] and left + sprite_w <= panel_shape[1]
        out[top : top + sprite_h, left : left + sprite_w] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the overlap of two masks as stamp centers for the right-panel sprite.

    Input structure:
    - The grid is split by two full-height grey columns into three equal-width panels.
    - The left and middle panels are blue/black masks on the same coordinate system.
    - The right panel contains one red 4-connected sprite on a black background.

    Rule:
    - Find the blue cells that are on in both mask panels at the same coordinates.
    - Extract the red sprite's bounding box from the right panel as a stamp template.
    - Create a black output canvas the size of one panel and stamp that template at every hit,
      aligning the template center to the hit coordinate and processing hits top-to-bottom,
      then left-to-right.

    Key insight:
    - The logical AND does not produce output colors directly; it produces the position set that
      controls where the third-panel sprite is stamped.
    """
    parsed = _parse_grid(grid)
    return _render_output(parsed["panel_shape"], parsed["hits"], parsed["sprite"])


def _valid_hit_row_sets(height: int) -> list[tuple[int, ...]]:
    rows = list(range(1, height - 1))
    options: list[tuple[int, ...]] = []
    for count in (2, 3):
        for combo in itertools.combinations(rows, count):
            if all(combo[i + 1] - combo[i] >= 3 for i in range(len(combo) - 1)):
                options.append(combo)
    return options


def _stamp_boxes(centers: list[tuple[int, int]], sprite_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = sprite_shape
    cr = h // 2
    cc = w // 2
    boxes = []
    for r, c in centers:
        boxes.append((r - cr, c - cc, r - cr + h - 1, c - cc + w - 1))
    return boxes


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _sample_extras(
    height: int,
    forbidden: set[tuple[int, int]],
    count: int,
) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(PANEL_W) if (r, c) not in forbidden]
    border_like = [(r, c) for r, c in cells if r in (0, height - 1) or c in (0, PANEL_W - 1)]
    interior = [cell for cell in cells if cell not in border_like]

    picks: list[tuple[int, int]] = []
    available = set(cells)
    if count >= 2 and border_like and interior:
        first = random.choice(border_like)
        picks.append(first)
        available.remove(first)
        second_choices = [cell for cell in interior if cell in available]
        second = random.choice(second_choices)
        picks.append(second)
        available.remove(second)
    elif count >= 1:
        pool = border_like if border_like and random.random() < 0.6 else cells
        first = random.choice(pool)
        picks.append(first)
        available.remove(first)

    while len(picks) < count:
        remaining = sorted(available)
        choice = random.choice(remaining)
        picks.append(choice)
        available.remove(choice)
    return set(picks)


def _sample_masks(height: int, hit_centers: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((height, PANEL_W), dtype=int)
    middle = np.zeros((height, PANEL_W), dtype=int)
    hit_set = set(hit_centers)

    min_total = max(3, len(hit_set) + 1)
    left_total = random.randint(min_total, 6)
    middle_total = random.randint(min_total, 6)

    left_only = _sample_extras(height, hit_set, left_total - len(hit_set))
    middle_only = _sample_extras(height, hit_set | left_only, middle_total - len(hit_set))

    for r, c in hit_set | left_only:
        left[r, c] = BLUE
    for r, c in hit_set | middle_only:
        middle[r, c] = BLUE

    assert np.count_nonzero(left == BLUE) == left_total
    assert np.count_nonzero(middle == BLUE) == middle_total
    assert left_only and middle_only
    assert not ((left == BLUE) & (middle == BLUE)).sum() > len(hit_set)
    return left, middle


def _build_right_panel(height: int, sprite: np.ndarray) -> np.ndarray:
    panel = np.zeros((height, PANEL_W), dtype=int)
    top = random.randint(0, height - sprite.shape[0])
    left = random.randint(0, PANEL_W - sprite.shape[1])
    panel[top : top + sprite.shape[0], left : left + sprite.shape[1]] = sprite
    return panel


def _assemble_grid(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = left.shape[0]
    grid = np.zeros((h, PANEL_W * 3 + DIVIDER_COUNT), dtype=int)
    grid[:, :PANEL_W] = left
    grid[:, PANEL_W] = GREY
    grid[:, PANEL_W + 1 : 2 * PANEL_W + 1] = middle
    grid[:, 2 * PANEL_W + 1] = GREY
    grid[:, 2 * PANEL_W + 2 :] = right
    return grid


def _accepts(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_grid(grid)
        expected = _render_output(parsed["panel_shape"], parsed["hits"], parsed["sprite"])
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    hits = [tuple(map(int, hit)) for hit in parsed["hits"]]
    if len(hits) < 2:
        return False

    left_only = set(map(tuple, np.argwhere(parsed["left"] == BLUE))) - set(hits)
    middle_only = set(map(tuple, np.argwhere(parsed["middle"] == BLUE))) - set(hits)
    if not left_only or not middle_only:
        return False

    boxes = _stamp_boxes(hits, parsed["sprite"].shape)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                return False

    modified_left = grid.copy()
    first_hit = hits[0]
    modified_left[first_hit[0], first_hit[1]] = BLACK
    if np.array_equal(main(modified_left), expected):
        return False

    modified_middle = grid.copy()
    middle_col = PANEL_W + 1 + first_hit[1]
    modified_middle[first_hit[0], middle_col] = BLACK
    if np.array_equal(main(modified_middle), expected):
        return False

    alt_sprite = random.choice([sprite for sprite in SPRITE_LIBRARY if not np.array_equal(sprite, parsed["sprite"])])
    sr0, sc0, sr1, sc1 = parsed["sprite_bbox"]
    modified_sprite = grid.copy()
    right_c0 = 2 * PANEL_W + 2
    modified_sprite[sr0 : sr1 + 1, right_c0 + sc0 : right_c0 + sc1 + 1] = BLACK
    modified_sprite[sr0 : sr1 + 1, right_c0 + sc0 : right_c0 + sc1 + 1] = alt_sprite
    if np.array_equal(main(modified_sprite), expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_H_MIN, GRID_H_MAX)
    sprite = random.choice(SPRITE_LIBRARY)
    hit_rows = random.choice(_valid_hit_row_sets(height))
    hit_centers = [(row, random.randint(1, PANEL_W - 2)) for row in hit_rows]
    left, middle = _sample_masks(height, hit_centers)
    right = _build_right_panel(height, sprite)
    grid = _assemble_grid(left, middle, right)
    return grid if _accepts(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the stamp-from-and task")
