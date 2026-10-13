from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CANVAS = 5
STAMP = 9
BAND_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIDES = ("top", "bottom", "left", "right")
BIG_GRID = 30
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _solid_rectangles(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == color):
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1):
            boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _uniform_strip(
    grid: np.ndarray, cells: list[tuple[int, int]]
) -> int | None:
    if not cells:
        return None
    values = [int(grid[r, c]) for r, c in cells]
    color = values[0]
    if color in (BG, CANVAS, STAMP):
        return None
    if all(v == color for v in values):
        return color
    return None


def _band_for_canvas(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    rows, cols = grid.shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[str, int]] = []

    if r0 > 0:
        color = _uniform_strip(grid, [(r0 - 1, c) for c in range(c0, c1 + 1)])
        if color is not None:
            candidates.append(("top", color))
    if r1 + 1 < rows:
        color = _uniform_strip(grid, [(r1 + 1, c) for c in range(c0, c1 + 1)])
        if color is not None:
            candidates.append(("bottom", color))
    if c0 > 0:
        color = _uniform_strip(grid, [(r, c0 - 1) for r in range(r0, r1 + 1)])
        if color is not None:
            candidates.append(("left", color))
    if c1 + 1 < cols:
        color = _uniform_strip(grid, [(r, c1 + 1) for r in range(r0, r1 + 1)])
        if color is not None:
            candidates.append(("right", color))

    assert len(candidates) == 1, "each canvas must have exactly one colored strip"
    return candidates[0]


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _stamp_side_for_canvas(
    coords: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    banned_side: str,
) -> str | None:
    r0, c0, r1, c1 = bbox
    sr0, sc0, sr1, sc1 = _bbox(coords)
    hits: list[str] = []

    if banned_side != "top" and sr1 < r0 and r0 - sr1 - 1 in (0, 1) and _interval_overlap(sc0, sc1, c0, c1):
        hits.append("top")
    if banned_side != "bottom" and sr0 > r1 and sr0 - r1 - 1 in (0, 1) and _interval_overlap(sc0, sc1, c0, c1):
        hits.append("bottom")
    if banned_side != "left" and sc1 < c0 and c0 - sc1 - 1 in (0, 1) and _interval_overlap(sr0, sr1, r0, r1):
        hits.append("left")
    if banned_side != "right" and sc0 > c1 and sc0 - c1 - 1 in (0, 1) and _interval_overlap(sr0, sr1, r0, r1):
        hits.append("right")

    if len(hits) == 1:
        return hits[0]
    return None


def _project_stamp(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    canvas_bbox: tuple[int, int, int, int],
    side: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = canvas_bbox
    if side == "top":
        base = max(r for r, _ in coords)
        for r, c in coords:
            tr = r0 + (base - r)
            if r0 <= tr <= r1 and c0 <= c <= c1:
                out[tr, c] = color
    elif side == "bottom":
        base = min(r for r, _ in coords)
        for r, c in coords:
            tr = r1 - (r - base)
            if r0 <= tr <= r1 and c0 <= c <= c1:
                out[tr, c] = color
    elif side == "left":
        base = max(c for _, c in coords)
        for r, c in coords:
            tc = c0 + (base - c)
            if r0 <= r <= r1 and c0 <= tc <= c1:
                out[r, tc] = color
    else:
        base = min(c for _, c in coords)
        for r, c in coords:
            tc = c1 - (c - base)
            if r0 <= r <= r1 and c0 <= tc <= c1:
                out[r, tc] = color


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    maroon_components = _components(out == STAMP)
    canvases = _solid_rectangles(out, CANVAS)
    used: set[int] = set()

    out[out == STAMP] = BG
    for canvas_bbox in canvases:
        band_side, band_color = _band_for_canvas(out, canvas_bbox)
        matches: list[tuple[int, str]] = []
        for idx, coords in enumerate(maroon_components):
            if idx in used:
                continue
            side = _stamp_side_for_canvas(coords, canvas_bbox, band_side)
            if side is not None:
                matches.append((idx, side))
        assert len(matches) == 1, "each canvas must match exactly one maroon object"
        idx, stamp_side = matches[0]
        used.add(idx)
        _project_stamp(out, maroon_components[idx], canvas_bbox, stamp_side, band_color)

    assert len(used) == len(maroon_components), "every maroon object must belong to one canvas"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each maroon object into its neighboring grey canvas using the adjacent strip color.

    Input structure:
    - The background is black.
    - Several disconnected grey solid rectangles act as canvases.
    - Each canvas has exactly one one-pixel-wide colored strip touching one outer edge.
    - Separate maroon 4-connected objects sit beside canvases on non-strip sides, with
      zero or one black cell of gap.

    Rule:
    - For each grey canvas, locate its unique colored strip and its unique neighboring
      maroon object on a different side.
    - Determine whether that maroon object is above, below, left, or right of the canvas.
    - Mirror the object's depth inward from the canvas edge while keeping edge alignment,
      recoloring the projected cells with the strip color and clipping anything outside
      the canvas.
    - Remove the original maroon objects; keep the strip cells and untouched grey canvas
      cells unchanged.

    Key insight:
    - The shape, the inward projection direction, and the final color come from three
      different visual elements: the maroon object, its side relative to the canvas,
      and the canvas's colored strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _apply_rule(g)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    area_min = min(h * w, 3)
    area_max = min(h * w, max(4, min(8, h * w)))
    target = rng.randint(area_min, area_max)

    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if np.all(mask):
        mask[rng.randrange(h), rng.randrange(w)] = False
    return _tight_crop(mask)


def _edge_shift_range(mask: np.ndarray, canvas_len: int, side: str) -> tuple[int, int]:
    if side == "top":
        edge_positions = np.flatnonzero(mask[-1])
    elif side == "bottom":
        edge_positions = np.flatnonzero(mask[0])
    elif side == "left":
        edge_positions = np.flatnonzero(mask[:, -1])
    else:
        edge_positions = np.flatnonzero(mask[:, 0])

    lo = -int(edge_positions.max())
    hi = canvas_len - 1 - int(edge_positions.min())
    return lo, hi


def _build_cluster(
    rng: random.Random, band_color: int, stamp_side: str, band_side: str
) -> dict | None:
    canvas_h = rng.randint(4, 8)
    canvas_w = rng.randint(4, 8)
    gap = rng.randint(0, 1)

    stamp_h = rng.randint(2, min(6, canvas_h + 2))
    stamp_w = rng.randint(2, min(6, canvas_w + 2))
    stamp_mask = _random_connected_mask(rng, stamp_h, stamp_w)
    stamp_h, stamp_w = stamp_mask.shape

    shift_lo, shift_hi = _edge_shift_range(
        stamp_mask,
        canvas_w if stamp_side in ("top", "bottom") else canvas_h,
        stamp_side,
    )
    if shift_lo > shift_hi:
        return None
    shift = rng.randint(shift_lo, shift_hi)

    cells: list[tuple[int, int, int]] = []
    for r in range(canvas_h):
        for c in range(canvas_w):
            cells.append((r, c, CANVAS))

    if band_side == "top":
        for c in range(canvas_w):
            cells.append((-1, c, band_color))
    elif band_side == "bottom":
        for c in range(canvas_w):
            cells.append((canvas_h, c, band_color))
    elif band_side == "left":
        for r in range(canvas_h):
            cells.append((r, -1, band_color))
    else:
        for r in range(canvas_h):
            cells.append((r, canvas_w, band_color))

    if stamp_side == "top":
        stamp_r0 = -(gap + stamp_h)
        stamp_c0 = shift
    elif stamp_side == "bottom":
        stamp_r0 = canvas_h + gap
        stamp_c0 = shift
    elif stamp_side == "left":
        stamp_r0 = shift
        stamp_c0 = -(gap + stamp_w)
    else:
        stamp_r0 = shift
        stamp_c0 = canvas_w + gap

    stamp_cells_raw: list[tuple[int, int]] = []
    for r, c in np.argwhere(stamp_mask):
        rr = int(stamp_r0 + r)
        cc = int(stamp_c0 + c)
        cells.append((rr, cc, STAMP))
        stamp_cells_raw.append((rr, cc))

    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)

    grid = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for r, c, color in cells:
        grid[r - min_r, c - min_c] = color

    canvas_bbox = (-min_r, -min_c, -min_r + canvas_h - 1, -min_c + canvas_w - 1)
    stamp_cells = [(r - min_r, c - min_c) for r, c in stamp_cells_raw]

    return {
        "grid": grid,
        "canvas_bbox": canvas_bbox,
        "stamp_side": stamp_side,
        "band_side": band_side,
        "band_color": band_color,
        "stamp_cells": stamp_cells,
    }


def _candidate_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(BIG_GRID - h + 1):
        for c0 in range(BIG_GRID - w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(BIG_GRID, r0 + h + 1)
            cc1 = min(BIG_GRID, c0 + w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _translate_bbox(
    bbox: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    expected = grid.copy()
    expected[expected == STAMP] = BG
    for spec in specs:
        _project_stamp(
            expected,
            spec["stamp_cells"],
            spec["canvas_bbox"],
            spec["stamp_side"],
            spec["band_color"],
        )
    return expected


def _crop_result(
    rng: random.Random, grid: np.ndarray, specs: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    filled = np.argwhere(grid != BG)
    r0, c0 = filled.min(axis=0)
    r1, c1 = filled.max(axis=0)

    top = int(rng.randint(0, int(r0)))
    left = int(rng.randint(0, int(c0)))
    bottom = int(rng.randint(0, BIG_GRID - 1 - int(r1)))
    right = int(rng.randint(0, BIG_GRID - 1 - int(c1)))

    cr0 = int(r0) - top
    cc0 = int(c0) - left
    cr1 = int(r1) + bottom
    cc1 = int(c1) + right

    cropped = grid[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
    translated_specs: list[dict] = []
    for spec in specs:
        translated_specs.append(
            {
                "canvas_bbox": _translate_bbox(spec["canvas_bbox"], -cr0, -cc0),
                "stamp_side": spec["stamp_side"],
                "band_color": spec["band_color"],
                "stamp_cells": [(r - cr0, c - cc0) for r, c in spec["stamp_cells"]],
            }
        )
    return cropped, translated_specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas_count = rng.randint(2, 4)
    band_colors = rng.sample(BAND_COLORS, canvas_count)

    cluster_specs: list[dict] = []
    for band_color in band_colors:
        stamp_side = rng.choice(SIDES)
        band_side = rng.choice([side for side in SIDES if side != stamp_side])
        cluster = _build_cluster(rng, band_color, stamp_side, band_side)
        if cluster is None:
            return None
        cluster_specs.append(cluster)

    order = list(range(canvas_count))
    rng.shuffle(order)

    grid = np.full((BIG_GRID, BIG_GRID), BG, dtype=int)
    occupied = np.zeros((BIG_GRID, BIG_GRID), dtype=bool)
    placed_specs: list[dict] = []

    for idx in order:
        cluster = cluster_specs[idx]
        local = cluster["grid"]
        positions = _candidate_positions(occupied, local.shape)
        if not positions:
            return None
        top, left = rng.choice(positions)
        h, w = local.shape
        non_bg = local != BG
        grid[top : top + h, left : left + w][non_bg] = local[non_bg]
        occupied[max(0, top - 1) : min(BIG_GRID, top + h + 1), max(0, left - 1) : min(BIG_GRID, left + w + 1)] = True

        placed_specs.append(
            {
                "canvas_bbox": _translate_bbox(cluster["canvas_bbox"], top, left),
                "stamp_side": cluster["stamp_side"],
                "band_color": cluster["band_color"],
                "stamp_cells": [(r + top, c + left) for r, c in cluster["stamp_cells"]],
            }
        )

    cropped, cropped_specs = _crop_result(rng, grid, placed_specs)
    expected = _render_expected(cropped, cropped_specs)

    if max(cropped.shape) > 30 or max(expected.shape) > 30:
        return None
    if len(_solid_rectangles(cropped, CANVAS)) != canvas_count:
        return None
    if len(_components(cropped == STAMP)) != canvas_count:
        return None

    try:
        actual = main(cropped)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid canvas-stamp ARC input")
