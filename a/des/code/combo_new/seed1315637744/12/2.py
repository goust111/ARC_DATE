from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

LEGEND_PATTERN = np.array([[RED, BLUE], [YELLOW, GREEN]], dtype=int)
ROTATION_BY_COLOR = {RED: 0, BLUE: 1, GREEN: 2, YELLOW: 3}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _points_to_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    return len(_components(mask)) == 1


def _find_legend(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    matches: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if np.array_equal(grid[r : r + 2, c : c + 2], LEGEND_PATTERN):
                matches.append((r, c))
    assert matches, "rotation legend not found"
    return max(matches, key=lambda rc: (rc[0], rc[1]))


def _extract_color_objects(
    grid: np.ndarray, color: int, legend_mask: np.ndarray
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], tuple[int, int] | None] | None:
    comps = _components((grid == color) & ~legend_mask)
    if not comps:
        return None
    comps.sort(key=len, reverse=True)
    template = comps[0]
    tr0, tc0, tr1, tc1 = _bbox(template)

    outside: list[list[tuple[int, int]]] = []
    inside: list[list[tuple[int, int]]] = []
    for comp in comps[1:]:
        r0, c0, r1, c1 = _bbox(comp)
        if tr0 <= r0 and r1 <= tr1 and tc0 <= c0 and c1 <= tc1:
            inside.append(comp)
        else:
            outside.append(comp)

    if not outside:
        return None
    piece = max(outside, key=len)

    marker = None
    singletons = [comp[0] for comp in inside if len(comp) == 1]
    if len(singletons) == 1:
        marker = singletons[0]
    elif len(singletons) > 1:
        marker = min(singletons)

    return (tr0, tc0, tr1, tc1), piece, marker


def _hole_for_template(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int], marker: tuple[int, int] | None
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    tr0, tc0, tr1, tc1 = template_bbox
    sub = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(sub == BLACK):
        if any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in comp):
            continue
        holes.append(comp)
    assert holes, "template must contain an enclosed black region"

    chosen = max(holes, key=len)
    if marker is not None:
        mr, mc = marker
        marker_neighbors = {(mr - tr0 + dr, mc - tc0 + dc) for dr, dc in DIR4}
        adjacent = [comp for comp in holes if any(cell in marker_neighbors for cell in comp)]
        if adjacent:
            adjacent.sort(key=len, reverse=True)
            chosen = adjacent[0]

    hole_mask, (hr0, hc0, hr1, hc1) = _points_to_mask(chosen)
    return hole_mask, (tr0 + hr0, tc0 + hc0, tr0 + hr1, tc0 + hc1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered colored fragment by the angle encoded in the legend and fit it into the matching hollow template.

    Input structure:
    - The background is black.
    - The upper part contains one or more colored rectangular templates; each template uses one border color from a 2×2 color legend and contains one enclosed irregular black cavity plus a single same-colored marker pixel inside that cavity.
    - The lower part contains scattered monochrome fragments whose colors match the template colors one-to-one.
    - A 2×2 legend in the lower-right corner shows red, blue, green, and yellow clockwise for 0°, 90°, 180°, and 270° clockwise rotation.

    Rule:
    - Locate the legend and convert each template color into a clockwise quarter-turn count.
    - For each active legend color, pair the largest same-colored upper component with the matching lower fragment.
    - Rotate the fragment by the color-indicated angle until it matches the enclosed black cavity of the template, then fill that cavity with the fragment color.
    - Return only the combined template area, dropping the scattered lower fragments and the legend.

    Key insight:
    - Color performs two jobs at once: it pairs each fragment with its template and, through the fixed red/blue/green/yellow legend, also encodes the rotation needed before shape matching works.
    """
    g = np.asarray(grid, dtype=int)
    legend_r, legend_c = _find_legend(g)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[legend_r : legend_r + 2, legend_c : legend_c + 2] = True

    objects: list[dict[str, object]] = []
    for color in (RED, BLUE, GREEN, YELLOW):
        extracted = _extract_color_objects(g, color, legend_mask)
        if extracted is None:
            continue
        template_bbox, piece_pts, marker = extracted
        hole_mask, hole_bbox = _hole_for_template(g, template_bbox, marker)
        piece_mask, _ = _points_to_mask(piece_pts)
        rot_k = ROTATION_BY_COLOR[color]
        rotated_piece = np.rot90(piece_mask, -rot_k)
        assert rotated_piece.shape == hole_mask.shape, "rotated piece shape does not fit cavity bbox"
        assert np.array_equal(rotated_piece, hole_mask), "rotated piece must exactly match cavity"
        objects.append(
            {
                "color": color,
                "template_bbox": template_bbox,
                "hole_bbox": hole_bbox,
                "hole_mask": hole_mask,
            }
        )

    assert objects, "no active templates found"

    min_r = min(obj["template_bbox"][0] for obj in objects)  # type: ignore[index]
    min_c = min(obj["template_bbox"][1] for obj in objects)  # type: ignore[index]
    max_r = max(obj["template_bbox"][2] for obj in objects)  # type: ignore[index]
    max_c = max(obj["template_bbox"][3] for obj in objects)  # type: ignore[index]

    out = g[min_r : max_r + 1, min_c : max_c + 1].copy()
    for obj in objects:
        color = int(obj["color"])
        hr0, hc0, hr1, hc1 = obj["hole_bbox"]  # type: ignore[misc]
        hole_mask = obj["hole_mask"]  # type: ignore[assignment]
        target = out[hr0 - min_r : hr1 - min_r + 1, hc0 - min_c : hc1 - min_c + 1]
        target[hole_mask] = color
    return out


def _grow_connected_mask(rng: random.Random, h: int, w: int, cells: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier = set()
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while int(mask.sum()) < cells:
        if not frontier:
            return None
        nr, nc = rng.choice(tuple(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for dr, dc in DIR4:
            rr, cc = nr + dr, nc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))
    return _trim_mask(mask)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    candidates: list[tuple[int, int]] = []
    for r, c in corners:
        if not mask[r, c]:
            continue
        test = mask.copy()
        test[r, c] = False
        if not _is_connected(test):
            continue
        if not (np.any(test[0, :]) and np.any(test[-1, :]) and np.any(test[:, 0]) and np.any(test[:, -1])):
            continue
        if not any(
            0 <= r + dr < h and 0 <= c + dc < w and test[r + dr, c + dc]
            for dr, dc in DIR4
        ):
            continue
        candidates.append((r, c))
    return candidates


def _make_shape_bundle(rng: random.Random) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(3, 4)
        w = rng.randint(3, 4)
        max_cells = min(h * w, 8)
        cells = rng.randint(5, max_cells)
        full = _grow_connected_mask(rng, h, w, cells)
        if full is None or full.shape[0] < 2 or full.shape[1] < 2:
            continue
        candidates = _marker_candidates(full)
        if not candidates:
            continue
        marker = rng.choice(candidates)
        hole = full.copy()
        hole[marker] = False
        if int(hole.sum()) < 3 or not _is_connected(hole):
            continue
        return hole, marker
    return None


def _paint_mask(array: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    array[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color


def _build_template(color: int, hole_mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = hole_mask.shape
    top_pad = rng.randint(1, 2)
    bottom_pad = rng.randint(1, 2)
    left_pad = rng.randint(1, 2)
    right_pad = rng.randint(1, 2)
    template = np.full((hh + top_pad + bottom_pad, hw + left_pad + right_pad), color, dtype=int)
    window = template[top_pad : top_pad + hh, left_pad : left_pad + hw]
    window[hole_mask] = BLACK
    filled = template.copy()
    filled[top_pad : top_pad + hh, left_pad : left_pad + hw][hole_mask] = color
    return template, filled


def _row_layout(order: list[int], items: list[dict[str, object]], gap: int) -> int:
    return sum(items[idx]["template"].shape[1] for idx in order) + gap * (len(order) - 1)  # type: ignore[index]


def _compose_top_block(
    items: list[dict[str, object]], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    n = len(items)
    order = list(range(n))
    rng.shuffle(order)

    if n == 1:
        rows = [order]
    elif n == 2:
        rows = [order] if rng.random() < 0.5 else [[order[0]], [order[1]]]
    elif n == 3:
        split = rng.choice([1, 2])
        rows = [order[:split], order[split:]]
    else:
        rows = [order[:2], order[2:]]

    hgap = rng.randint(1, 2)
    vgap = rng.randint(1, 2)
    row_offsets = [0] + [rng.randint(0, 2) for _ in range(len(rows) - 1)]
    row_heights = [max(items[idx]["template"].shape[0] for idx in row) for row in rows]  # type: ignore[index]
    block_w = max(row_offsets[i] + _row_layout(rows[i], items, hgap) for i in range(len(rows)))
    block_h = sum(row_heights) + vgap * (len(rows) - 1)

    top_input = np.zeros((block_h, block_w), dtype=int)
    top_output = np.zeros((block_h, block_w), dtype=int)

    cursor_r = 0
    for row_index, row in enumerate(rows):
        cursor_c = row_offsets[row_index]
        row_h = row_heights[row_index]
        for idx in row:
            template = items[idx]["template"]  # type: ignore[assignment]
            filled = items[idx]["filled"]  # type: ignore[assignment]
            th, tw = template.shape
            vshift = rng.randint(0, row_h - th)
            rr = cursor_r + vshift
            top_input[rr : rr + th, cursor_c : cursor_c + tw] = np.maximum(
                top_input[rr : rr + th, cursor_c : cursor_c + tw], template
            )
            top_output[rr : rr + th, cursor_c : cursor_c + tw] = np.maximum(
                top_output[rr : rr + th, cursor_c : cursor_c + tw], filled
            )
            cursor_c += tw + hgap
        cursor_r += row_h + vgap

    return top_input, top_output


def _place_piece(
    grid: np.ndarray,
    reserved: np.ndarray,
    piece_mask: np.ndarray,
    color: int,
    top_min: int,
    rng: random.Random,
) -> bool:
    ph, pw = piece_mask.shape
    gh, gw = grid.shape
    if top_min + ph > gh:
        return False
    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(top_min, gh - ph)
        left = rng.randint(0, gw - pw)
        rr0 = max(0, top - 1)
        rr1 = min(gh, top + ph + 1)
        cc0 = max(0, left - 1)
        cc1 = min(gw, left + pw + 1)
        if np.any(reserved[rr0:rr1, cc0:cc1]):
            continue
        _paint_mask(grid, piece_mask, color, top, left)
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = [RED, BLUE, GREEN, YELLOW]
    rng.shuffle(colors)
    count = rng.randint(1, 4)
    active_colors = colors[:count]

    items: list[dict[str, object]] = []
    max_piece_h = 0
    for color in active_colors:
        bundle = _make_shape_bundle(rng)
        if bundle is None:
            return None
        hole_mask, _marker = bundle
        template, filled = _build_template(color, hole_mask, rng)
        piece_mask = np.rot90(hole_mask, ROTATION_BY_COLOR[color])
        max_piece_h = max(max_piece_h, piece_mask.shape[0])
        items.append(
            {
                "color": color,
                "template": template,
                "filled": filled,
                "piece_mask": piece_mask,
            }
        )

    top_input, expected = _compose_top_block(items, rng)
    block_h, block_w = top_input.shape

    top_margin = rng.randint(0, 1)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    lower_band_h = max(max_piece_h + 4, block_h // 2 + 3, 7)
    grid_h = top_margin + block_h + lower_band_h
    grid_w = max(left_margin + block_w + right_margin, 14)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top_margin : top_margin + block_h, left_margin : left_margin + block_w] = top_input
    legend_r = grid_h - 2
    legend_c = grid_w - 2
    grid[legend_r : legend_r + 2, legend_c : legend_c + 2] = LEGEND_PATTERN

    reserved = np.zeros_like(grid, dtype=bool)
    reserved[max(0, legend_r - 1) : grid_h, max(0, legend_c - 1) : grid_w] = True
    piece_top = top_margin + block_h + 1
    if piece_top >= legend_r + 1:
        return None

    for item in items:
        piece_mask = item["piece_mask"]  # type: ignore[assignment]
        color = int(item["color"])
        if not _place_piece(grid, reserved, piece_mask, color, piece_top, rng):
            return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
