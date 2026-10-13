import random
from collections import deque

import numpy as np


_BG = 0
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
_GEN_RETRIES = 300
_BODY_H_RANGE = (8, 29)   # input adds one palette row, so total height stays <= 30
_BODY_W_RANGE = (10, 30)
_PALETTE_LEN_RANGE = (2, 5)
_REGION_COUNT_RANGE = (2, 5)
_THICKNESS_OPTIONS = (1, 2)
_OUTER_H_RANGE = (4, 10)
_OUTER_W_RANGE = (4, 10)
_EDGE_MARGIN = 1
_REGION_GAP = 1
_SPEC_RETRIES = 80


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
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_regions(border_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = border_mask.shape
    open_mask = ~border_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = open_mask & ~outside
    return _components(enclosed)


def _extract_palette_and_body(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[int] = []
    for row_idx in (0, g.shape[0] - 1):
        row = g[row_idx]
        nz = np.flatnonzero(row != _BG)
        if nz.size == 0:
            continue
        colors = [int(row[c]) for c in nz]
        if len(colors) >= 2 and len(set(colors)) == len(colors):
            candidates.append(int(row_idx))

    assert len(candidates) == 1, "expected exactly one palette row on the top or bottom edge"
    palette_row = candidates[0]
    nz = np.flatnonzero(g[palette_row] != _BG)
    palette = [int(g[palette_row, c]) for c in nz]
    body = g[1:, :].copy() if palette_row == 0 else g[:-1, :].copy()
    return palette, body


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region by using its internal marker count as an index into the palette strip.

    Input structure:
    - The background is black.
    - The top or bottom edge contains a one-row palette strip made of distinct colored pixels read from left to right.
    - Elsewhere, several closed single-color ring outlines enclose black interiors, and some interiors contain small marker pixels of another color.

    Rule:
    - Remove the palette strip from the output.
    - For each closed outline, count how many non-outline marker pixels lie inside its enclosed interior.
    - If the count is N > 0, fill the interior black pixels with the N-th palette color (1-based); if the count is 0, leave the interior black.
    - Erase the marker pixels themselves back to black.

    Key insight:
    - The marker pixels are not copied directly; they are counted first, and that count is used to look up a fill color in the edge palette.
    """
    palette, body = _extract_palette_and_body(np.asarray(grid, dtype=int))
    out = body.copy()

    for color in [int(v) for v in np.unique(body) if int(v) != _BG]:
        for comp in _components(body == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                border_mask[r - r0, c - c0] = True

            for cavity in _enclosed_regions(border_mask):
                sub = body[r0 : r1 + 1, c0 : c1 + 1]
                marker_count = 0
                black_cells: list[tuple[int, int]] = []
                marker_cells: list[tuple[int, int]] = []

                for rr, cc in cavity:
                    value = int(sub[rr, cc])
                    if value == _BG:
                        black_cells.append((rr, cc))
                    else:
                        marker_count += 1
                        marker_cells.append((rr, cc))

                assert marker_count <= len(palette), "marker count exceeds palette length"
                fill_color = _BG if marker_count == 0 else int(palette[marker_count - 1])

                for rr, cc in black_cells:
                    out[r0 + rr, c0 + cc] = fill_color
                for rr, cc in marker_cells:
                    out[r0 + rr, c0 + cc] = _BG

    return out


def _sample_region_spec(rng: random.Random, count: int, body_h: int, body_w: int) -> dict | None:
    min_inner_area = 1 if count == 0 else count + 1
    for _ in range(_SPEC_RETRIES):
        thickness = rng.choice(_THICKNESS_OPTIONS)
        outer_h = rng.randint(max(_OUTER_H_RANGE[0], 2 * thickness + 1), min(_OUTER_H_RANGE[1], body_h - 2 * _EDGE_MARGIN))
        outer_w = rng.randint(max(_OUTER_W_RANGE[0], 2 * thickness + 1), min(_OUTER_W_RANGE[1], body_w - 2 * _EDGE_MARGIN))
        inner_h = outer_h - 2 * thickness
        inner_w = outer_w - 2 * thickness
        inner_area = inner_h * inner_w
        if inner_h <= 0 or inner_w <= 0 or inner_area < min_inner_area:
            continue
        return {
            "count": int(count),
            "thickness": int(thickness),
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
            "area": int(outer_h * outer_w),
        }
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < _EDGE_MARGIN or left < _EDGE_MARGIN:
        return False
    if top + height > h - _EDGE_MARGIN or left + width > w - _EDGE_MARGIN:
        return False
    r0 = max(0, top - _REGION_GAP)
    c0 = max(0, left - _REGION_GAP)
    r1 = min(h, top + height + _REGION_GAP)
    c1 = min(w, left + width + _REGION_GAP)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_regions(rng: random.Random, body_h: int, body_w: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((body_h, body_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["area"], reverse=True)

    for idx in order:
        spec = specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        tops = range(_EDGE_MARGIN, body_h - _EDGE_MARGIN - outer_h + 1)
        lefts = range(_EDGE_MARGIN, body_w - _EDGE_MARGIN - outer_w + 1)
        candidates = [(r, c) for r in tops for c in lefts]
        rng.shuffle(candidates)

        placed = None
        for top, left in candidates:
            if _can_place(occupied, top, left, outer_h, outer_w):
                placed = (top, left)
                occupied[top : top + outer_h, left : left + outer_w] = True
                break
        if placed is None:
            return None
        placements[idx] = placed

    return [(int(r), int(c)) for r, c in placements if (r, c) is not None]


def _draw_region(
    grid: np.ndarray,
    rng: random.Random,
    top: int,
    left: int,
    spec: dict,
    border_color: int,
    marker_color: int,
) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    thickness = int(spec["thickness"])
    count = int(spec["count"])

    grid[top : top + outer_h, left : left + outer_w] = border_color
    inner_top = top + thickness
    inner_left = left + thickness
    inner_h = outer_h - 2 * thickness
    inner_w = outer_w - 2 * thickness
    grid[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = _BG

    cells = [(r, c) for r in range(inner_top, inner_top + inner_h) for c in range(inner_left, inner_left + inner_w)]
    if count:
        for r, c in rng.sample(cells, count):
            grid[r, c] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    body_h = rng.randint(*_BODY_H_RANGE)
    body_w = rng.randint(*_BODY_W_RANGE)
    max_palette_len = min(_PALETTE_LEN_RANGE[1], body_w)
    if max_palette_len < _PALETTE_LEN_RANGE[0]:
        return None
    palette_len = rng.randint(_PALETTE_LEN_RANGE[0], max_palette_len)

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    border_color = color_pool.pop()
    marker_color = color_pool.pop()
    palette = [int(color_pool.pop()) for _ in range(palette_len)]

    area_capacity = max(_REGION_COUNT_RANGE[0], min(_REGION_COUNT_RANGE[1], body_h * body_w // 70 + 2))
    region_count = rng.randint(_REGION_COUNT_RANGE[0], area_capacity)

    counts = [0, rng.randint(1, palette_len)]
    while len(counts) < region_count:
        counts.append(rng.randint(0, palette_len))
    rng.shuffle(counts)

    specs: list[dict] = []
    for count in counts:
        spec = _sample_region_spec(rng, count, body_h, body_w)
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_regions(rng, body_h, body_w, specs)
    if placements is None or len(placements) != len(specs):
        return None

    body = np.zeros((body_h, body_w), dtype=int)
    for spec, (top, left) in zip(specs, placements):
        _draw_region(body, rng, top, left, spec, border_color, marker_color)

    palette_row = np.zeros(body_w, dtype=int)
    start = rng.randint(0, body_w - palette_len)
    palette_row[start : start + palette_len] = np.array(palette, dtype=int)

    if rng.random() < 0.5:
        full = np.vstack([palette_row[None, :], body])
    else:
        full = np.vstack([body, palette_row[None, :]])

    assert full.ndim == 2 and full.shape[0] <= 30 and full.shape[1] <= 30
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
