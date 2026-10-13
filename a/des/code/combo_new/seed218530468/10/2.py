from __future__ import annotations

import random
from collections import deque

import numpy as np


RECT_OPTIONS: tuple[tuple[int, int], ...] = ((2, 3), (2, 4), (3, 3), (2, 5))
PALETTE: tuple[int, ...] = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = _trim_mask(np.rot90(mask, k))
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rotated)
    return out


def _solid_rectangle_dims(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    valid_dims: set[tuple[int, int]] = set()

    for rot_a in _rotations(mask_a):
        coords_a = {(int(r), int(c)) for r, c in np.argwhere(rot_a)}
        ha, wa = rot_a.shape
        for rot_b in _rotations(mask_b):
            coords_b = [(int(r), int(c)) for r, c in np.argwhere(rot_b)]
            hb, wb = rot_b.shape
            for dr in range(-hb, ha + 1):
                for dc in range(-wb, wa + 1):
                    shifted_b = {(r + dr, c + dc) for r, c in coords_b}
                    if coords_a & shifted_b:
                        continue
                    union = coords_a | shifted_b
                    rows = [r for r, _ in union]
                    cols = [c for _, c in union]
                    rect_h = max(rows) - min(rows) + 1
                    rect_w = max(cols) - min(cols) + 1
                    if len(union) == rect_h * rect_w:
                        valid_dims.add((min(rect_h, rect_w), max(rect_h, rect_w)))

    assert len(valid_dims) == 1, "each color pair must assemble into one normalized rectangle"
    return next(iter(valid_dims))


def _render_stack(rectangles: list[dict[str, int]], bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item["area"], item["height"], item["width"]))
    out_h = sum(item["height"] for item in ordered)
    out_w = max(item["width"] for item in ordered)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    bottom = out_h
    for item in ordered:
        h = item["height"]
        w = item["width"]
        bottom -= h
        left = (out_w - w) // 2
        out[bottom : bottom + h, left : left + w] = int(item["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored scattered L fragments into full rectangles, then stack the rectangles by area.

    Input structure:
    - The background is the most common color.
    - Every non-background color forms exactly two separate 4-connected L-shaped fragments.
    - Each fragment is a one-cell-thick right-angle shape with three to five cells.
    - The two fragments of one color can be translated and rotated to make one solid rectangle.

    Rule:
    - For each non-background color, isolate its two fragments and find the unique rectangle they can form.
    - Treat that completed rectangle as a solid block of the same color.
    - Sort the completed rectangles by area from small to large.
    - Draw them on a minimal output canvas, stacked bottom-to-top with no gaps and horizontally centered.

    Key insight:
    - The input objects are incomplete corners rather than final objects; the solver must mentally reassemble each
      color pair into a rectangle first, then use rectangle area as the ordering key for the final stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    rectangles: list[dict[str, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        components = _find_components(g == color)
        assert len(components) == 2, "each foreground color must appear in exactly two fragments"
        mask_a = _coords_to_mask(components[0])
        mask_b = _coords_to_mask(components[1])
        height, width = _solid_rectangle_dims(mask_a, mask_b)
        rectangles.append(
            {
                "color": color,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert rectangles, "expected at least one foreground color"
    return _render_stack(rectangles, bg)


def _rectangle_piece_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if height == 2 and width >= 3:
        first = np.zeros((2, width - 1), dtype=bool)
        first[0, :] = True
        first[1, 0] = True
        second = np.zeros((2, width - 1), dtype=bool)
        second[1, :] = True
        second[0, -1] = True
        return first, second

    if width == 2 and height >= 3:
        first = np.zeros((height - 1, 2), dtype=bool)
        first[:, 0] = True
        first[0, 1] = True
        second = np.zeros((height - 1, 2), dtype=bool)
        second[:, 1] = True
        second[-1, 0] = True
        return first, second

    if height == 3 and width == 3:
        first = np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
        second = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)
        return first, second

    raise ValueError(f"unsupported rectangle size: {(height, width)}")


def _random_l_pair(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    first, second = _rectangle_piece_masks(height, width)
    first = _trim_mask(np.rot90(first, random.randrange(4)))
    second = _trim_mask(np.rot90(second, random.randrange(4)))
    if random.random() < 0.5:
        return first, second
    return second, first


def _dilate(mask: np.ndarray) -> np.ndarray:
    expanded = np.array(mask, dtype=bool, copy=True)
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    return expanded


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = mask.shape
    grid_h, grid_w = grid.shape
    if h > grid_h or w > grid_w:
        return False

    forbidden = _dilate(occupied)
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            if not bool((forbidden[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return True


def _sample_canvas_side(num_colors: int) -> int:
    low = max(12, 6 + 3 * num_colors)
    high = 30
    return random.randint(low, high)


def _generate_one() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    usable_colors = [color for color in PALETTE if color != bg]

    count = random.randint(2, len(RECT_OPTIONS))
    chosen_rects = random.sample(list(RECT_OPTIONS), count)
    colors = random.sample(usable_colors, count)
    rectangles = [
        {"height": int(h), "width": int(w), "area": int(h * w), "color": int(color)}
        for (h, w), color in zip(chosen_rects, colors)
    ]
    expected = _render_stack(rectangles, bg)

    pieces: list[tuple[np.ndarray, int]] = []
    for rect in rectangles:
        piece_a, piece_b = _random_l_pair(rect["height"], rect["width"])
        pieces.append((piece_a, rect["color"]))
        pieces.append((piece_b, rect["color"]))
    random.shuffle(pieces)

    for _ in range(PLACEMENT_ATTEMPTS):
        side_h = _sample_canvas_side(count)
        side_w = _sample_canvas_side(count)
        grid = np.full((side_h, side_w), int(bg), dtype=int)
        occupied = np.zeros((side_h, side_w), dtype=bool)

        ok = True
        for mask, color in sorted(pieces, key=lambda item: int(np.prod(item[0].shape)), reverse=True):
            if not _place_mask(grid, occupied, mask, color):
                ok = False
                break
        if not ok:
            continue

        if _background_color(grid) != bg:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scattered-L puzzle")
