from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
RED_COLOR = 2
YELLOW_COLOR = 4
ORANGE_COLOR = 7
FRAME_COLOR = 8

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

FRAME_STEP = 4
INNER_SIZE_MIN = 5
INNER_SIZE_MAX = 8
EXTRA_MARGIN_MIN = 6
EXTRA_MARGIN_MAX = 10
OUTER_MARGIN_MIN = 3
EXTRA_RED_MAX = 4
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _infer_input_colors(grid: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    comp0 = _color_components(grid, colors[0])
    comp1 = _color_components(grid, colors[1])
    all_singletons0 = comp0 and all(int(comp["area"]) == 1 for comp in comp0)
    all_singletons1 = comp1 and all(int(comp["area"]) == 1 for comp in comp1)

    if all_singletons0 and not all_singletons1:
        return colors[0], colors[1]
    if all_singletons1 and not all_singletons0:
        return colors[1], colors[0]
    raise AssertionError("could not distinguish isolated markers from frame components")


def _is_closed_frame(component: dict[str, object]) -> bool:
    coords = set(component["coords"])
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    expected_area = 2 * (height + width) - 4
    if int(component["area"]) != expected_area:
        return False

    for rr, cc in coords:
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False

    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coords or (r1, cc) not in coords:
            return False
    for rr in range(r0 + 1, r1):
        if (rr, c0) not in coords or (rr, c1) not in coords:
            return False

    return True


def _strictly_contains(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the closed frames around each marker pixel and grow parity-dependent halos.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms axis-aligned hollow rectangular frames; some are fully closed and some have a gap.
    - The other foreground color appears only as isolated single pixels.

    Rule:
    - For each isolated marker pixel, count the closed frames whose interiors strictly contain it.
    - If that positive count is odd, color its four diagonal background neighbors yellow; if it is positive and even, color its four orthogonal background neighbors orange.
    - Ignore broken frames in the count and keep all original frame and marker pixels unchanged.

    Key insight:
    - The emitted neighborhood depends on enclosure depth, so the solver must first distinguish true rectangular cycles from broken lookalikes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    red_color, frame_color = _infer_input_colors(g, bg)
    frame_components = _color_components(g, frame_color)
    closed_bboxes = [
        tuple(map(int, component["bbox"]))
        for component in frame_components
        if _is_closed_frame(component)
    ]

    out = g.copy()
    h, w = map(int, g.shape)

    for r, c in np.argwhere(g == red_color):
        rr = int(r)
        cc = int(c)
        depth = sum(_strictly_contains(bbox, rr, cc) for bbox in closed_bboxes)
        if depth == 0:
            continue

        mark_color = YELLOW_COLOR if depth % 2 == 1 else ORANGE_COLOR
        deltas = DIAGONALS if depth % 2 == 1 else ORTHOGONALS
        for dr, dc in deltas:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == bg:
                out[nr, nc] = mark_color

    return out


def _frame_bboxes(
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for inset in (0, FRAME_STEP, 2 * FRAME_STEP):
        height = inner_h + 4 * FRAME_STEP - 2 * inset
        width = inner_w + 4 * FRAME_STEP - 2 * inset
        r0 = top + inset
        c0 = left + inset
        bboxes.append((r0, c0, r0 + height - 1, c0 + width - 1))
    return bboxes


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], *, closed: bool) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    if closed:
        return

    side = random.choice(("top", "bottom", "left", "right"))
    side_len = (c1 - c0 + 1) if side in ("top", "bottom") else (r1 - r0 + 1)
    gap_len = random.randint(2, max(2, side_len // 2))
    start_offset = random.randint(1, side_len - gap_len - 1)

    if side == "top":
        grid[r0, c0 + start_offset : c0 + start_offset + gap_len] = BG_COLOR
    elif side == "bottom":
        grid[r1, c0 + start_offset : c0 + start_offset + gap_len] = BG_COLOR
    elif side == "left":
        grid[r0 + start_offset : r0 + start_offset + gap_len, c0] = BG_COLOR
    else:
        grid[r0 + start_offset : r0 + start_offset + gap_len, c1] = BG_COLOR


def _all_neighbors_are_bg(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = map(int, grid.shape)
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != BG_COLOR:
        return False
    for dr, dc in NEIGHBORS8:
        if int(grid[r + dr, c + dc]) != BG_COLOR:
            return False
    return True


def _depth_candidates(
    grid: np.ndarray,
    closed_bboxes: list[tuple[int, int, int, int]],
) -> dict[int, list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    by_depth: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _all_neighbors_are_bg(grid, r, c):
                continue
            depth = sum(_strictly_contains(bbox, r, c) for bbox in closed_bboxes)
            by_depth.setdefault(depth, []).append((r, c))
    return by_depth


def _far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 3 for r, c in chosen)


def _pick_spaced(
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
) -> tuple[int, int] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for cell in shuffled:
        if _far_enough(cell, chosen):
            return cell
    return None


def _attempt() -> np.ndarray | None:
    inner_h = random.randint(INNER_SIZE_MIN, INNER_SIZE_MAX)
    inner_w = random.randint(INNER_SIZE_MIN, INNER_SIZE_MAX)
    outer_h = inner_h + 4 * FRAME_STEP
    outer_w = inner_w + 4 * FRAME_STEP

    max_extra_h = min(EXTRA_MARGIN_MAX, 30 - outer_h)
    max_extra_w = min(EXTRA_MARGIN_MAX, 30 - outer_w)
    if max_extra_h < EXTRA_MARGIN_MIN or max_extra_w < EXTRA_MARGIN_MIN:
        return None

    extra_h = random.randint(EXTRA_MARGIN_MIN, max_extra_h)
    extra_w = random.randint(EXTRA_MARGIN_MIN, max_extra_w)
    top = random.randint(OUTER_MARGIN_MIN, extra_h - OUTER_MARGIN_MIN)
    left = random.randint(OUTER_MARGIN_MIN, extra_w - OUTER_MARGIN_MIN)
    height = outer_h + extra_h
    width = outer_w + extra_w

    grid = np.full((height, width), BG_COLOR, dtype=int)
    outer_bbox, broken_bbox, inner_bbox = _frame_bboxes(top, left, inner_h, inner_w)
    _draw_frame(grid, outer_bbox, closed=True)
    _draw_frame(grid, broken_bbox, closed=False)
    _draw_frame(grid, inner_bbox, closed=True)

    closed_bboxes = [outer_bbox, inner_bbox]
    candidates_by_depth = _depth_candidates(grid, closed_bboxes)

    depth0 = candidates_by_depth.get(0, [])
    depth1 = candidates_by_depth.get(1, [])
    depth2 = candidates_by_depth.get(2, [])

    odd_broken = [
        cell
        for cell in depth1
        if _strictly_contains(broken_bbox, cell[0], cell[1])
        and not _strictly_contains(inner_bbox, cell[0], cell[1])
    ]
    even_inner = [
        cell for cell in depth2 if _strictly_contains(inner_bbox, cell[0], cell[1])
    ]

    chosen: list[tuple[int, int]] = []
    for candidates in (depth0, odd_broken, even_inner):
        cell = _pick_spaced(candidates, chosen)
        if cell is None:
            return None
        chosen.append(cell)

    target_count = 3 + random.randint(0, EXTRA_RED_MAX)
    while len(chosen) < target_count:
        depth = random.choice((0, 1, 1, 2, 2))
        cell = _pick_spaced(candidates_by_depth.get(depth, []), chosen)
        if cell is None:
            break
        chosen.append(cell)

    for r, c in chosen:
        grid[r, c] = RED_COLOR

    out = main(grid)
    if not np.any(out == YELLOW_COLOR):
        return None
    if not np.any(out == ORANGE_COLOR):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
