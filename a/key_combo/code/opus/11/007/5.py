from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_SIDE_RANGE = (13, 18)
INTERIOR_HEIGHT_RANGE = (7, 9)
INTERIOR_WIDTH_RANGE = (7, 9)
SPRITE_SIZE_RANGE = (4, 7)
OUTLINE_ROWS_RANGE = (3, 5)
GENERATION_ATTEMPTS = 600
OUTLINE_ATTEMPTS = 160
SPRITE_ATTEMPTS = 160


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in offsets:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    return box if np.array_equal(mask, expected) else None


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len <= 0 or dst_len <= 0:
        raise ValueError("axis lengths must be positive")
    if src_len == 1:
        return 0 if dst_len == 1 else dst_len // 2
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _draw_segment(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True
        return
    raise ValueError("adjacent sprite pixels must map to a horizontal or vertical segment")


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    open_space = ~stroke
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1:-1, 1:-1]
    return stroke | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    coords = np.argwhere(sprite)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}

    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True

    sprite_set = {(int(r), int(c)) for r, c in coords}
    for sr, sc in sprite_set:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in sprite_set:
                r0, c0 = mapped[(sr, sc)]
                r1, c1 = mapped[(nr, nc)]
                _draw_segment(stroke, r0, c0, r1, c1)

    return _fill_enclosed(stroke)


def _fill_outline_rows(outline: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(outline.any(axis=1))
    assert 3 <= len(rows) <= 5, "expected a 3-5 row outline"
    filled = np.zeros_like(outline, dtype=bool)
    for r in rows:
        cols = np.flatnonzero(outline[r])
        assert len(cols) == 2, "each occupied row must have exactly two green boundary pixels"
        filled[r, cols[0] : cols[1] + 1] = True
    return filled


def _compose_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    bg: int = BLACK,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0] = RED
    out[fr0 : fr1 + 1, fc1] = RED
    out[fr0, fc0 : fc1 + 1] = RED
    out[fr1, fc0 : fc1 + 1] = RED
    interior = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    interior[a_mask & ~b_mask] = BLUE
    interior[b_mask & ~a_mask] = GREEN
    interior[a_mask & b_mask] = YELLOW
    return out


def _parse_case(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    frame = _frame_bbox(g == RED)
    assert frame is not None, "expected one red hollow frame"
    fr0, fc0, fr1, fc1 = frame
    assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2
    ir0, ic0 = fr0 + 1, fc0 + 1
    ir1, ic1 = fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1, "frame must have a non-empty interior"

    blue_parts = _components(g == BLUE, connectivity=4)
    assert len(blue_parts) == 1, "expected one blue 4-connected sprite"
    sprite_coords = blue_parts[0]
    sprite_box = _bbox(g == BLUE)
    assert sprite_box is not None
    sr0, sc0, sr1, sc1 = sprite_box
    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == BLUE
    assert sprite.shape[0] >= 2 and sprite.shape[1] >= 2, "sprite bbox must be at least 2x2"
    frame_region = np.zeros_like(g, dtype=bool)
    frame_region[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    assert not np.any(frame_region[sprite_coords[:, 0], sprite_coords[:, 1]]), "sprite must lie outside the frame"

    outline_global = g == GREEN
    green_parts = _components(outline_global, connectivity=8)
    assert len(green_parts) == 1, "expected one green outline object"
    outline = outline_global[ir0 : ir1 + 1, ic0 : ic1 + 1]
    assert np.count_nonzero(outline_global) == np.count_nonzero(outline), "green outline must be inside the frame interior"
    rows, cols = np.nonzero(outline)
    assert rows.size > 0
    assert rows.min() >= 1 and rows.max() <= outline.shape[0] - 2, "outline must not touch the frame border"
    assert cols.min() >= 1 and cols.max() <= outline.shape[1] - 2, "outline must not touch the frame border"

    a_mask = _stretch_sprite(sprite, outline.shape)
    b_mask = _fill_outline_rows(outline)
    return bg, frame, a_mask, b_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the outside sprite into the frame, fill the inside row-paired outline, and color their boolean overlap.

    Input structure:
    - A black background contains one red one-pixel-thick hollow rectangle.
    - Outside the frame there is one blue 4-connected sprite; inside the frame there is one sparse green outline.
    - The green outline occupies 3-5 rows, with exactly two green boundary pixels on each occupied row, and neither object touches the red border.

    Rule:
    - Clear the original outside sprite.
    - Stretch the blue sprite into the frame interior by mapping its bbox extremes to the interior extremes, connecting mapped 4-neighbor pairs with axis-aligned segments, and filling enclosed regions to get region A.
    - Fill the green outline row by row between each pair of boundary pixels to get region B, then output blue for A-only, green for B-only, yellow for A∩B, black for neither, while keeping the red frame unchanged.

    Key insight:
    - The answer is not either transformation alone: the final interior requires computing two independent spatial constructions in the same coordinate system and then taking their pixelwise boolean composition.
    """
    bg, frame, a_mask, b_mask = _parse_case(grid)
    return _compose_output(np.asarray(grid).shape, frame, a_mask, b_mask, bg=bg)


def _has_both_adjacent_directions(mask: np.ndarray) -> bool:
    horizontal = bool(np.any(mask[:, :-1] & mask[:, 1:]))
    vertical = bool(np.any(mask[:-1, :] & mask[1:, :]))
    return horizontal and vertical


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_sprite() -> np.ndarray | None:
    dims = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
    for _ in range(SPRITE_ATTEMPTS):
        h, w = random.choice(dims)
        limit = min(SPRITE_SIZE_RANGE[1], h * w - 1)
        if limit < SPRITE_SIZE_RANGE[0]:
            continue
        target = random.randint(SPRITE_SIZE_RANGE[0], limit)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if not _has_both_adjacent_directions(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _good_stretch(a_mask: np.ndarray) -> bool:
    area = int(a_mask.sum())
    h, w = a_mask.shape
    if area < max(10, (h * w) // 5):
        return False
    if area > (4 * h * w) // 5:
        return False
    box = _bbox(a_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area == box_area:
        return False
    if len(np.flatnonzero(a_mask.any(axis=0))) < 4:
        return False
    if len(np.flatnonzero(a_mask.any(axis=1))) < 4:
        return False
    return True


def _frame_positions(grid_h: int, grid_w: int, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    candidates = []
    center_r = (grid_h - frame_h) / 2.0
    center_c = (grid_w - frame_w) / 2.0
    for r0 in range(1, grid_h - frame_h):
        for c0 in range(1, grid_w - frame_w):
            dist = abs(r0 - center_r) + abs(c0 - center_c)
            if dist <= 3.5:
                candidates.append((r0, c0))
    if candidates:
        return candidates
    return [(r0, c0) for r0 in range(grid_h - frame_h + 1) for c0 in range(grid_w - frame_w + 1)]


def _sprite_locations(
    grid_h: int,
    grid_w: int,
    frame: tuple[int, int, int, int],
    sprite_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame
    sh, sw = sprite_shape
    out = []
    for r0 in range(grid_h - sh + 1):
        for c0 in range(grid_w - sw + 1):
            r1 = r0 + sh - 1
            c1 = c0 + sw - 1
            separated = r1 < fr0 - 1 or r0 > fr1 + 1 or c1 < fc0 - 1 or c0 > fc1 + 1
            if separated:
                out.append((r0, c0))
    return out


def _outline_mask_for_rows(
    interior_h: int,
    interior_w: int,
    a_mask: np.ndarray,
) -> np.ndarray | None:
    if interior_h < 5 or interior_w < 5:
        return None
    a_box = _bbox(a_mask)
    assert a_box is not None
    _, ac0, _, ac1 = a_box

    for _ in range(OUTLINE_ATTEMPTS):
        row_count = random.randint(OUTLINE_ROWS_RANGE[0], min(OUTLINE_ROWS_RANGE[1], interior_h - 2))
        starts = [
            s
            for s in range(1, interior_h - row_count)
            if np.any(a_mask[s : s + row_count])
        ]
        if not starts:
            continue
        start = random.choice(starts)
        rows = list(range(start, start + row_count))
        apex_top = random.choice([True, False])
        apex_index = 0 if apex_top else row_count - 1

        apex_lo = max(1, ac0 - 1)
        apex_hi = min(interior_w - 3, ac1)
        if apex_lo > apex_hi:
            apex_lo, apex_hi = 1, interior_w - 3
        left = random.randint(apex_lo, apex_hi)
        right = left + 1

        intervals = [None] * row_count
        intervals[apex_index] = (left, right)
        order = range(apex_index + 1, row_count) if apex_top else range(apex_index - 1, -1, -1)

        ok = True
        current_left, current_right = left, right
        widened = False
        for idx in order:
            candidates = []
            for dl in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    nl = current_left + dl
                    nr = current_right + dr
                    if not (1 <= nl < nr <= interior_w - 2):
                        continue
                    if nr - nl < current_right - current_left:
                        continue
                    candidates.append((nl, nr))
            if not candidates:
                ok = False
                break

            wideners = [pair for pair in candidates if pair[1] - pair[0] > current_right - current_left]
            if wideners and random.random() < 0.8:
                nl, nr = random.choice(wideners)
            else:
                nl, nr = random.choice(candidates)
            widened |= (nr - nl) > (current_right - current_left)
            intervals[idx] = (nl, nr)
            current_left, current_right = nl, nr

        if not ok or not widened:
            continue

        outline = np.zeros((interior_h, interior_w), dtype=bool)
        for row, pair in zip(rows, intervals):
            left, right = pair
            outline[row, left] = True
            outline[row, right] = True

        if len(_components(outline, connectivity=8)) != 1:
            continue
        b_mask = _fill_outline_rows(outline)
        if not np.any(a_mask & b_mask):
            continue
        if not np.any(a_mask & ~b_mask):
            continue
        if not np.any(b_mask & ~a_mask):
            continue
        if int(np.count_nonzero(a_mask & b_mask)) * 2 > int(np.count_nonzero(a_mask | b_mask)):
            continue
        return outline
    return None


def _build_input(
    grid_h: int,
    grid_w: int,
    frame: tuple[int, int, int, int],
    sprite: np.ndarray,
    sprite_pos: tuple[int, int],
    outline: np.ndarray,
) -> np.ndarray:
    fr0, fc0, fr1, fc1 = frame
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[fr0 : fr1 + 1, fc0] = RED
    grid[fr0 : fr1 + 1, fc1] = RED
    grid[fr0, fc0 : fc1 + 1] = RED
    grid[fr1, fc0 : fc1 + 1] = RED
    grid[fr0 + 1 : fr1, fc0 + 1 : fc1][outline] = GREEN

    sr, sc = sprite_pos
    sh, sw = sprite.shape
    grid[sr : sr + sh, sc : sc + sw][sprite] = BLUE
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    try:
        bg, frame, a_mask, b_mask = _parse_case(grid)
    except Exception:
        return False
    if bg != BLACK:
        return False
    if not np.any(a_mask & b_mask):
        return False
    if not np.any(a_mask & ~b_mask):
        return False
    if not np.any(b_mask & ~a_mask):
        return False
    out = main(grid)
    expected = _compose_output(grid.shape, frame, a_mask, b_mask, bg=bg)
    if not np.array_equal(out, expected):
        return False
    palette = set(map(int, out.ravel()))
    return {RED, BLUE, GREEN, YELLOW}.issubset(palette)


def _attempt_input() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    interior_h = random.randint(*INTERIOR_HEIGHT_RANGE)
    interior_w = random.randint(*INTERIOR_WIDTH_RANGE)
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    if frame_h >= grid_h or frame_w >= grid_w:
        return None

    sprite = _random_sprite()
    if sprite is None:
        return None

    a_mask = _stretch_sprite(sprite, (interior_h, interior_w))
    if not _good_stretch(a_mask):
        return None

    frame_positions = _frame_positions(grid_h, grid_w, frame_h, frame_w)
    random.shuffle(frame_positions)
    for fr0, fc0 in frame_positions[:24]:
        frame = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
        outline = _outline_mask_for_rows(interior_h, interior_w, a_mask)
        if outline is None:
            continue
        locations = _sprite_locations(grid_h, grid_w, frame, sprite.shape)
        if not locations:
            continue
        sprite_pos = random.choice(locations)
        grid = _build_input(grid_h, grid_w, frame, sprite, sprite_pos, outline)
        if _validate_case(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
