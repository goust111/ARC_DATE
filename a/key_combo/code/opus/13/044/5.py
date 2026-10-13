from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _border_reachable_background(walls: np.ndarray) -> np.ndarray:
    """Return the background cells that are 4-connected to the grid border."""
    h, w = walls.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not walls[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not walls[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def _hole_components(walls: np.ndarray) -> list[np.ndarray]:
    """Return enclosed background regions as lists of coordinates."""
    h, w = walls.shape
    exterior = _border_reachable_background(walls)
    holes = (~walls) & (~exterior)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not holes[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _compose_output(blue_mask: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
    walls = blue_mask | red_mask
    exterior = _border_reachable_background(walls)
    out = np.zeros(walls.shape, dtype=int)
    out[exterior] = GREY
    out[(~walls) & (~exterior)] = YELLOW
    out[blue_mask] = BLUE
    out[red_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two side-by-side panels, then flood-fill outside and enclosed background in the composite wall map.

    Input structure:
    - The input width is split into two equal side-by-side panels with black background.
    - The left panel contains only blue line pixels and the right panel contains only red line pixels.
    - Blue and red never occupy the same aligned panel cell.

    Rule:
    - Align the left and right panels cell-by-cell to form one output panel.
    - Keep blue-only cells blue and red-only cells red; treat both as walls.
    - Flood-fill black background cells from the output border with 4-connectivity: reachable cells become grey, unreachable enclosed cells become yellow.

    Key insight:
    - Neither source panel closes a region on its own; the enclosure only appears after the two panels are overlaid into one joint wall structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w2 = g.shape
    assert w2 % 2 == 0, "expected two equal-width panels"

    w = w2 // 2
    left = g[:, :w]
    right = g[:, w:]

    assert np.all((left == BLACK) | (left == BLUE)), "left panel must use only black and blue"
    assert np.all((right == BLACK) | (right == RED)), "right panel must use only black and red"

    blue_mask = left == BLUE
    red_mask = right == RED
    assert not np.any(blue_mask & red_mask), "aligned panel cells must not both be occupied"

    return _compose_output(blue_mask, red_mask)


def _rect_perimeter(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _random_lengths(total: int, parts: int, min_each: int) -> list[int]:
    assert total >= parts * min_each
    lengths = [min_each] * parts
    for _ in range(total - parts * min_each):
        lengths[random.randrange(parts)] += 1
    return lengths


def _assign_cycle_colors(
    cycle: list[tuple[int, int]], segment_count: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    rotated = cycle[:]
    offset = random.randrange(len(rotated))
    rotated = rotated[offset:] + rotated[:offset]

    lengths = _random_lengths(len(rotated), segment_count, min_each=2)
    start_blue = bool(random.getrandbits(1))
    blue_cells: set[tuple[int, int]] = set()
    red_cells: set[tuple[int, int]] = set()

    idx = 0
    for seg_idx, seg_len in enumerate(lengths):
        color_is_blue = start_blue if seg_idx % 2 == 0 else not start_blue
        target = blue_cells if color_is_blue else red_cells
        for cell in rotated[idx : idx + seg_len]:
            target.add(cell)
        idx += seg_len
    return blue_cells, red_cells


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_rectangles(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(80):
        rh = random.randint(4, min(7, h - 2))
        rw = random.randint(4, min(7, w - 2))
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if all(_boxes_separated(rect, other, gap=1) for other in rects):
            rects.append(rect)
            if len(rects) == count:
                return rects
    return None


def _tail_candidates(
    rect: tuple[int, int, int, int], color_mask: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    r0, c0, r1, c1 = rect
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for c in range(c0 + 1, c1):
        if color_mask[r0, c]:
            candidates.append(((r0, c), (-1, 0), r0))
        if color_mask[r1, c]:
            candidates.append(((r1, c), (1, 0), color_mask.shape[0] - 1 - r1))
    for r in range(r0 + 1, r1):
        if color_mask[r, c0]:
            candidates.append(((r, c0), (0, -1), c0))
        if color_mask[r, c1]:
            candidates.append(((r, c1), (0, 1), color_mask.shape[1] - 1 - c1))
    return [cand for cand in candidates if cand[2] >= 1]


def _try_add_tail(mask: np.ndarray, occupied: np.ndarray, start: tuple[int, int], step: tuple[int, int], max_len: int) -> bool:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = step
    for _ in range(min(max_len, 3)):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
            break
        cells.append((r, c))
    if not cells:
        return False
    tail_len = random.randint(1, len(cells))
    for r, c in cells[:tail_len]:
        mask[r, c] = True
        occupied[r, c] = True
    return True


def _augment_with_tails(
    rects: list[tuple[int, int, int, int]], blue_mask: np.ndarray, red_mask: np.ndarray
) -> None:
    occupied = blue_mask | red_mask
    for rect in rects:
        if random.random() < 0.55:
            color_mask = blue_mask if random.random() < 0.5 else red_mask
            candidates = _tail_candidates(rect, color_mask)
            random.shuffle(candidates)
            for start, step, max_len in candidates:
                if _try_add_tail(color_mask, occupied, start, step, max_len):
                    break
        if random.random() < 0.30:
            color_mask = red_mask if random.random() < 0.5 else blue_mask
            candidates = _tail_candidates(rect, color_mask)
            random.shuffle(candidates)
            for start, step, max_len in candidates:
                if _try_add_tail(color_mask, occupied, start, step, max_len):
                    break


def _rectangle_color_coverage(
    rect: tuple[int, int, int, int], blue_mask: np.ndarray, red_mask: np.ndarray
) -> tuple[int, int]:
    cycle = _rect_perimeter(rect)
    blue_count = sum(1 for r, c in cycle if blue_mask[r, c])
    red_count = sum(1 for r, c in cycle if red_mask[r, c])
    return blue_count, red_count


def _build_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]] | None:
    chamber_count = random.choice((1, 1, 2))
    rects = _sample_rectangles(h, w, chamber_count)
    if rects is None:
        return None

    blue_mask = np.zeros((h, w), dtype=bool)
    red_mask = np.zeros((h, w), dtype=bool)
    for rect in rects:
        cycle = _rect_perimeter(rect)
        segment_count = random.choice((2, 4))
        blue_cells, red_cells = _assign_cycle_colors(cycle, segment_count)
        for r, c in blue_cells:
            blue_mask[r, c] = True
        for r, c in red_cells:
            red_mask[r, c] = True

    if np.any(blue_mask & red_mask):
        return None

    _augment_with_tails(rects, blue_mask, red_mask)
    return blue_mask, red_mask, rects


def _validate_masks(
    blue_mask: np.ndarray, red_mask: np.ndarray, rects: list[tuple[int, int, int, int]]
) -> bool:
    if not blue_mask.any() or not red_mask.any():
        return False
    if np.any(blue_mask & red_mask):
        return False

    blue_holes = _hole_components(blue_mask)
    red_holes = _hole_components(red_mask)
    combo_holes = _hole_components(blue_mask | red_mask)
    if blue_holes or red_holes:
        return False
    if len(combo_holes) != len(rects):
        return False
    if any(comp.shape[0] < 4 for comp in combo_holes):
        return False

    for rect in rects:
        blue_count, red_count = _rectangle_color_coverage(rect, blue_mask, red_mask)
        if blue_count < 2 or red_count < 2:
            return False

    return True


def _build_input_grid(blue_mask: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
    h, w = blue_mask.shape
    grid = np.zeros((h, 2 * w), dtype=int)
    grid[:, :w][blue_mask] = BLUE
    grid[:, w:][red_mask] = RED
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        h = random.randint(10, 12)
        w = random.randint(10, 12)
        built = _build_masks(h, w)
        if built is None:
            continue
        blue_mask, red_mask, rects = built
        if not _validate_masks(blue_mask, red_mask, rects):
            continue

        grid = _build_input_grid(blue_mask, red_mask)
        expected = _compose_output(blue_mask, red_mask)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid overlay-flood ARC input")
