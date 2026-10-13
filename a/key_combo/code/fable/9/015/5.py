from __future__ import annotations

from collections import deque
import random

import numpy as np


TEMPLATE_COLOR = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MARKER_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_RETRIES = 600
TEMPLATE_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, color: int, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 8:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    components: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _parse_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    grey_components = _color_components(grid, TEMPLATE_COLOR, connectivity=8)
    candidates = []
    for coords in grey_components:
        if len(coords) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        if np.any((patch != bg) & (patch != TEMPLATE_COLOR)):
            continue
        hole_coords = np.argwhere(patch == bg)
        if hole_coords.shape[0] != 1:
            continue
        hr, hc = map(int, hole_coords[0])
        if hr in (0, patch.shape[0] - 1) or hc in (0, patch.shape[1] - 1):
            continue
        if not np.all((patch == TEMPLATE_COLOR) | (patch == bg)):
            continue
        candidates.append((patch, (r0, c0, r1, c1), (hr, hc)))
    assert len(candidates) == 1, "expected exactly one grey template with one internal hole"
    return candidates[0]


def _stamp(
    out: np.ndarray,
    patch: np.ndarray,
    hole: tuple[int, int],
    intersection: tuple[int, int],
    color: int,
    bg: int,
) -> None:
    hole_r, hole_c = hole
    h, w = patch.shape
    target_r = int(intersection[0])
    target_c = int(intersection[1])
    r0 = target_r - hole_r
    c0 = target_c - hole_c
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    for dr in range(h):
        for dc in range(w):
            if int(patch[dr, dc]) == TEMPLATE_COLOR:
                out[r0 + dr, c0 + dc] = int(color)
            else:
                out[r0 + dr, c0 + dc] = int(bg)


def _build_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _get_bg(g)
    template_patch, template_bbox, hole = _parse_template(g, bg)

    col_markers = [(int(c), int(g[0, c])) for c in range(W - 1) if int(g[0, c]) != bg]
    row_markers = [(int(r), int(g[r, W - 1])) for r in range(1, H) if int(g[r, W - 1]) != bg]
    assert col_markers and row_markers, "expected top-row and right-column markers"

    out = g.copy()
    template_mask = template_patch == TEMPLATE_COLOR
    r0, c0, r1, c1 = template_bbox
    out[r0 : r1 + 1, c0 : c1 + 1][template_mask] = bg

    for row_r, row_color in row_markers:
        for col_c, col_color in col_markers:
            if row_color != col_color:
                continue
            _stamp(out, template_patch, hole, (row_r, col_c), row_color, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored copies of the grey hole-template onto intersections whose top and right headers share a color.

    Input structure:
    - The background is the dominant color.
    - Some header pixels lie on the top row and pick columns; some lie on the
      rightmost column and pick rows, with colors allowed to repeat.
    - One grey template sprite sits near a corner; inside its bounding box there
      is exactly one background hole.

    Rule:
    - Pair every right-edge row marker with every top-edge column marker.
    - Only same-colored pairs activate; for each active pair, place one copy of
      the template recolored to that pair color, with the template's unique hole
      aligned exactly on the row/column intersection.
    - Keep the edge markers unchanged, keep the hole as background, and erase the
      original grey template from the output.

    Key insight:
    - The edge markers are colored row/column headers, so not every intersection
      is used; the unique background hole inside the template, not the bbox
      center, is the anchoring point that fixes each stamp position.
    """
    return _build_output(np.asarray(grid, dtype=int))


def _pick_spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if high - low + 1 < count:
        return None
    for _ in range(200):
        positions = sorted(random.sample(range(low, high + 1), count))
        if all(b - a >= min_gap for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _choose_template_shape() -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        hole_choices = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if (2 * r != h - 1) or (2 * c != w - 1)
        ]
        if not hole_choices:
            continue
        hr, hc = random.choice(hole_choices)
        patch = np.full((h, w), TEMPLATE_COLOR, dtype=int)
        patch[hr, hc] = -1
        return patch, (hr, hc)
    raise RuntimeError("failed to sample template shape")


def _template_origin(corner: str, H: int, W: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 1, 1
    if corner == "TR":
        return 1, W - w - 2
    if corner == "BL":
        return H - h - 2, 1
    return H - h - 2, W - w - 2


def _choose_marker_colors(row_count: int, col_count: int) -> tuple[list[int], list[int]] | None:
    total = row_count * col_count
    for _ in range(300):
        pool = random.sample(MARKER_PALETTE, random.randint(3, 5))
        row_colors = [random.choice(pool) for _ in range(row_count)]
        col_colors = [random.choice(pool) for _ in range(col_count)]
        if len(set(row_colors)) < 2 or len(set(col_colors)) < 2:
            continue
        common = set(row_colors) & set(col_colors)
        if not common:
            continue
        active = sum(row_colors.count(color) * col_colors.count(color) for color in common)
        if active < 2 or active >= total:
            continue
        if not (any(row_colors.count(color) > 1 for color in common) or any(col_colors.count(color) > 1 for color in common) or len(common) >= 2):
            continue
        return row_colors, col_colors
    return None


def _paint_template(grid: np.ndarray, patch: np.ndarray, origin: tuple[int, int], bg: int) -> None:
    r0, c0 = origin
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[:, :] = np.where(patch == TEMPLATE_COLOR, TEMPLATE_COLOR, bg)


def _stamp_bbox(anchor_r: int, anchor_c: int, hole: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    hr, hc = hole
    h, w = shape
    r0 = anchor_r - hr
    c0 = anchor_c - hc
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _validate_instance(inp: np.ndarray) -> bool:
    out = main(inp)
    bg = _get_bg(inp)
    H, W = inp.shape
    template_patch, template_bbox, hole = _parse_template(inp, bg)
    h, w = template_patch.shape
    hr, hc = hole

    if int(inp[0, W - 1]) != bg:
        return False

    col_markers = [(int(c), int(inp[0, c])) for c in range(W - 1) if int(inp[0, c]) != bg]
    row_markers = [(int(r), int(inp[r, W - 1])) for r in range(1, H) if int(inp[r, W - 1]) != bg]
    if len(col_markers) < 2 or len(row_markers) < 2:
        return False

    active_boxes = []
    active_colors = set()
    inactive = 0
    for row_r, row_color in row_markers:
        for col_c, col_color in col_markers:
            if int(inp[row_r, col_c]) != bg:
                return False
            if row_color == col_color:
                active_colors.add(int(row_color))
                box = _stamp_bbox(row_r, col_c, hole, (h, w))
                if box[0] < 0 or box[1] < 0 or box[2] >= H or box[3] >= W:
                    return False
                active_boxes.append((box, row_r, col_c, int(row_color)))
            else:
                inactive += 1
    if len(active_boxes) < 2 or inactive == 0:
        return False
    if len(active_colors) < 1:
        return False

    for i, (box_a, _, _, _) in enumerate(active_boxes):
        if _boxes_overlap(box_a, template_bbox):
            return False
        for box_b, _, _, _ in active_boxes[i + 1 :]:
            if _boxes_overlap(box_a, box_b):
                return False

    if np.count_nonzero(inp == TEMPLATE_COLOR) != h * w - 1:
        return False
    if np.count_nonzero(template_patch == bg) != 1:
        return False
    if (2 * hr == h - 1) and (2 * hc == w - 1):
        return False

    expected = inp.copy()
    r0, c0, r1, c1 = template_bbox
    expected[r0 : r1 + 1, c0 : c1 + 1][template_patch == TEMPLATE_COLOR] = bg
    for box, row_r, col_c, color in active_boxes:
        br0, bc0, br1, bc1 = box
        patch = np.full((h, w), color, dtype=int)
        patch[hr, hc] = bg
        expected[br0 : br1 + 1, bc0 : bc1 + 1] = patch
        if int(expected[row_r, col_c]) != bg:
            return False
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    patch, hole = _choose_template_shape()
    h, w = patch.shape
    hr, hc = hole

    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)
    row_span_need = 1 + (row_count - 1) * (h + 1)
    col_span_need = 1 + (col_count - 1) * (w + 1)

    H_min = max(16, row_span_need + 10, 2 * h + 6)
    W_min = max(16, col_span_need + 8, w + 8)
    if H_min > 30 or W_min > 30:
        return None
    H = random.randint(H_min, 30)
    W = random.randint(W_min, 30)

    corner = random.choice(["TL", "TR", "BL", "BR"])
    tr, tc = _template_origin(corner, H, W, h, w)

    row_low = hr + 1
    row_high = H - (h - hr)
    if corner in {"TL", "TR"}:
        row_low = max(row_low, tr + h + hr + TEMPLATE_GAP)
    else:
        row_high = min(row_high, tr - h + hr - TEMPLATE_GAP)
    col_low = hc
    col_high = W - 1 - (w - hc)
    if row_low > row_high or col_low > col_high:
        return None

    row_positions = _pick_spaced_positions(row_low, row_high, row_count, h + 1)
    col_positions = _pick_spaced_positions(col_low, col_high, col_count, w + 1)
    if row_positions is None or col_positions is None:
        return None

    colors = _choose_marker_colors(row_count, col_count)
    if colors is None:
        return None
    row_colors, col_colors = colors

    grid = np.full((H, W), bg, dtype=int)
    _paint_template(grid, patch, (tr, tc), bg)
    for r, color in zip(row_positions, row_colors):
        grid[r, W - 1] = color
    for c, color in zip(col_positions, col_colors):
        grid[0, c] = color

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
