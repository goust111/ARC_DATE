from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6

MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1:
        return 0
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    steps = max(abs(r1 - r0), abs(c1 - c0))
    if steps == 0:
        mask[r0, c0] = True
        return
    for t in range(steps + 1):
        rr = int(round(r0 + (r1 - r0) * t / steps))
        cc = int(round(c0 + (c1 - c0) * t / steps))
        mask[rr, cc] = True


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not stroke[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not stroke[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not stroke[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return stroke | ~outside


def _stretch_sprite_mask(sprite_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite_mask.shape
    dst_h, dst_w = target_shape
    assert src_h >= 2 and src_w >= 2, "sprite bbox must be non-degenerate"
    assert dst_h > src_h and dst_w > src_w, "frame interior must strictly exceed sprite bbox"

    coords = np.argwhere(sprite_mask)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    stroke = np.zeros((dst_h, dst_w), dtype=bool)

    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True

    coord_set = {tuple(map(int, rc)) for rc in coords}
    for sr, sc in coord_set:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in coord_set:
                _draw_segment(stroke, mapped[(sr, sc)], mapped[(nr, nc)])

    return _fill_enclosed(stroke)


def _xor_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, dtype=int)
    out[a ^ b] = PINK
    return out


def _sprite_local_mask(grid: np.ndarray, sprite_box: tuple[int, int, int, int]) -> np.ndarray:
    sr0, sc0, sr1, sc1 = sprite_box
    return grid[sr0 : sr1 + 1, sc0 : sc1 + 1] == RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the outside red sprite into the frame interior, then xor that hidden mask with the lower blue mask.

    Input structure:
    - The background is black.
    - One green object is a one-pixel-thick hollow rectangular frame in the upper part.
    - One red 4-connected sprite sits outside the frame, and one blue/black mask of the frame-interior size sits below the frame after a black separator band.

    Rule:
    - Find the green frame and take its interior bbox as the destination region.
    - Find the red sprite, map its bbox corners to the frame-interior corners, map each red pixel by rounded linear interpolation, connect mapped 4-neighbor pairs with line segments, and fill any enclosed holes to obtain a red/black intermediate mask.
    - Find the lower blue mask, align it cellwise with the intermediate mask, and output pink where exactly one of the two masks is foreground; output black where they agree.

    Key insight:
    - The stretched red figure is only an invisible intermediate mask; the final answer is the symmetric difference between that hidden mask and the visible lower blue mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_box = _rect_frame_bbox(g == GREEN)
    assert frame_box is not None, "expected one green hollow frame"
    fr0, fc0, fr1, fc1 = frame_box
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1
    interior_shape = (ir1 - ir0 + 1, ic1 - ic0 + 1)

    red_components = _components4(g == RED)
    assert len(red_components) == 1, "expected a single red sprite"
    sprite_box = _bbox(g == RED)
    assert sprite_box is not None
    sr0, sc0, sr1, sc1 = sprite_box
    assert sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1, "sprite must be outside the frame bbox"
    sprite_mask = _sprite_local_mask(g, sprite_box)

    blue_box = _bbox(g == BLUE)
    assert blue_box is not None, "expected a lower blue mask"
    br0, bc0, br1, bc1 = blue_box
    assert (br1 - br0 + 1, bc1 - bc0 + 1) == interior_shape, "blue mask bbox must match the frame interior"
    blue_mask = g[br0 : br1 + 1, bc0 : bc1 + 1] == BLUE

    stretched = _stretch_sprite_mask(sprite_mask, interior_shape)
    return _xor_masks(stretched, blue_mask)


SPRITE_LIBRARY: tuple[np.ndarray, ...] = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
)


def _random_bottom_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    area = h * w
    target = random.randint(max(4, area // 4), max(4, (2 * area) // 3))
    mask = np.zeros((h, w), dtype=bool)

    seeds = {
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    }
    for r, c in seeds:
        mask[r, c] = True

    cells = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    random.shuffle(cells)
    for r, c in cells[: max(0, target - int(mask.sum()))]:
        mask[r, c] = True
    return mask


def _place_top_case(
    sprite: np.ndarray,
    interior_shape: tuple[int, int],
    sep_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ih, iw = interior_shape
    sh, sw = sprite.shape
    fh, fw = ih + 2, iw + 2

    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    width = max(fw, iw, sw) + random.randint(2, 6)
    height = top_margin + sh + 1 + fh + sep_rows + ih + bottom_margin
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    frame_r0 = top_margin + sh + 1
    frame_c0 = random.randint(0, width - fw)
    lower_r0 = frame_r0 + fh + sep_rows
    lower_c0 = random.randint(0, width - iw)
    sprite_r0 = top_margin
    sprite_c0 = random.randint(0, width - sw)

    grid[frame_r0 : frame_r0 + fh, frame_c0] = GREEN
    grid[frame_r0 : frame_r0 + fh, frame_c0 + fw - 1] = GREEN
    grid[frame_r0, frame_c0 : frame_c0 + fw] = GREEN
    grid[frame_r0 + fh - 1, frame_c0 : frame_c0 + fw] = GREEN
    grid[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sw][sprite] = RED
    return grid, np.array([lower_r0, lower_c0]), np.array([frame_r0 + 1, frame_c0 + 1])


def _place_side_case(
    sprite: np.ndarray,
    interior_shape: tuple[int, int],
    sep_rows: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ih, iw = interior_shape
    sh, sw = sprite.shape
    fh, fw = ih + 2, iw + 2

    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    extra_side = random.randint(1, 4)
    width = max(iw + random.randint(0, 3), fw + sw + 1 + extra_side)
    height = top_margin + fh + sep_rows + ih + bottom_margin
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    if side == "left":
        frame_c0 = random.randint(sw + 1, width - fw)
        sprite_c0 = frame_c0 - sw - 1
    else:
        frame_c0 = random.randint(0, width - fw - sw - 1)
        sprite_c0 = frame_c0 + fw + 1
    frame_r0 = top_margin
    sprite_r0 = random.randint(0, frame_r0 + fh - sh)
    lower_r0 = frame_r0 + fh + sep_rows
    lower_c0 = random.randint(0, width - iw)

    grid[frame_r0 : frame_r0 + fh, frame_c0] = GREEN
    grid[frame_r0 : frame_r0 + fh, frame_c0 + fw - 1] = GREEN
    grid[frame_r0, frame_c0 : frame_c0 + fw] = GREEN
    grid[frame_r0 + fh - 1, frame_c0 : frame_c0 + fw] = GREEN
    grid[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sw][sprite] = RED
    return grid, np.array([lower_r0, lower_c0]), np.array([frame_r0 + 1, frame_c0 + 1])


def _assemble_case() -> np.ndarray | None:
    sprite = random.choice(SPRITE_LIBRARY).copy()
    sh, sw = sprite.shape
    ih = random.randint(max(sh + 1, 4), min(sh + 4, 7))
    iw = random.randint(max(sw + 1, 4), min(sw + 4, 7))
    interior_shape = (ih, iw)
    stretched = _stretch_sprite_mask(sprite, interior_shape)
    if stretched.sum() == 0:
        return None

    sep_rows = random.choice((1, 2))
    branch = random.choice(("top", "left", "right"))
    if branch == "top":
        placed = _place_top_case(sprite, interior_shape, sep_rows)
    else:
        placed = _place_side_case(sprite, interior_shape, sep_rows, branch)
    if placed is None:
        return None
    grid, lower_origin, _frame_inner_origin = placed
    lower_r0, lower_c0 = map(int, lower_origin)

    for _ in range(60):
        blue_mask = _random_bottom_mask(interior_shape)
        overlap = int(np.count_nonzero(stretched & blue_mask))
        xor_only = int(np.count_nonzero(stretched ^ blue_mask))
        if overlap == 0 or xor_only == 0:
            continue
        if not np.any(stretched & ~blue_mask) and not np.any(blue_mask & ~stretched):
            continue
        grid[lower_r0 : lower_r0 + ih, lower_c0 : lower_c0 + iw][blue_mask] = BLUE

        expected = _xor_masks(stretched, blue_mask)
        if not np.array_equal(main(grid), expected):
            grid[lower_r0 : lower_r0 + ih, lower_c0 : lower_c0 + iw] = BLACK
            continue

        red_box = _bbox(grid == RED)
        blue_box = _bbox(grid == BLUE)
        frame_box = _rect_frame_bbox(grid == GREEN)
        assert red_box is not None and blue_box is not None and frame_box is not None
        rsz = (red_box[2] - red_box[0] + 1, red_box[3] - red_box[1] + 1)
        assert rsz[0] >= 2 and rsz[1] >= 2
        assert rsz[0] < ih and rsz[1] < iw
        assert len(_components4(grid == RED)) == 1
        assert blue_box[2] - blue_box[0] + 1 == ih
        assert blue_box[3] - blue_box[1] + 1 == iw
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _assemble_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
