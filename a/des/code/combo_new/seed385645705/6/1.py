import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    pts = set(cells)
    if len(pts) != 2 * h + 2 * w - 4:
        return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _frame_priorities(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GREY))
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        comps = _component_cells(g == color)
        frame_comps = [comp for comp in comps if _is_frame_component(comp)]
        if not frame_comps:
            continue
        best = max(frame_comps, key=lambda comp: (len(comp),) + _bbox(comp))
        frames.append((color, _bbox(best)))

    assert frames, "expected at least one colored rectangular frame"

    ordered = sorted(
        frames,
        key=lambda item: (
            sum(1 for _c, other in frames if _strictly_contains(other, item[1])),
            -(item[1][2] - item[1][0] + 1) * (item[1][3] - item[1][1] + 1),
        ),
    )
    return {color: rank for rank, (color, _bb) in enumerate(ordered)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey marker using the highest-priority adjacent cluster color implied by nested frames.

    Input structure:
    - The background is black.
    - One side of the grid contains several one-cell-thick hollow rectangular frames, each in a distinct color.
    - These frames are nested with visible black gaps, so outer containment defines an ordering from outside to inside.
    - Elsewhere in the grid are small monochrome color clusters plus grey single-pixel markers; every cluster color matches one frame color.

    Rule:
    - Detect the colored hollow rectangular frames and order their colors by nesting depth: outermost frame first, innermost last.
    - For every grey marker, inspect its 4-neighbors and collect the adjacent non-black, non-grey cluster colors.
    - Recolor the grey marker to the adjacent color with the highest frame priority.
    - Leave the frames, the colored clusters, and the background unchanged.

    Key insight:
    - The nested frame geometry is a legend that defines a color ranking, and that ranking must be transferred to distant local marker decisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    priority = _frame_priorities(g)
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != GREY:
                continue
            neighbor_colors = {
                int(g[nr, nc])
                for nr, nc in _neighbors4(r, c, g.shape)
                if int(g[nr, nc]) in priority
            }
            assert neighbor_colors, "every grey marker must touch at least one ranked color cluster"
            out[r, c] = min(neighbor_colors, key=lambda color: priority[color])

    return out


_EAST_TEMPLATES = [
    {(0, 1)},
    {(0, 1), (0, 2)},
    {(0, 1), (0, 2), (0, 3)},
    {(0, 1), (0, 2), (1, 2)},
    {(0, 1), (0, 2), (-1, 2)},
    {(0, 1), (0, 2), (1, 3)},
    {(0, 1), (0, 2), (-1, 3)},
    {(0, 1), (0, 2), (0, 3), (1, 3)},
    {(0, 1), (0, 2), (0, 3), (-1, 3)},
    {(0, 1), (0, 2), (1, 2), (1, 3)},
    {(0, 1), (0, 2), (-1, 2), (-1, 3)},
    {(0, 1), (0, 2), (1, 2), (0, 3)},
    {(0, 1), (0, 2), (-1, 2), (0, 3)},
    {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)},
    {(0, 1), (0, 2), (-1, 2), (0, 3), (-1, 3)},
]


def _transform_template(cells: set[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    if direction == "E":
        return set(cells)
    if direction == "W":
        return {(r, -c) for r, c in cells}
    if direction == "N":
        return {(-c, r) for r, c in cells}
    if direction == "S":
        return {(c, -r) for r, c in cells}
    raise ValueError(direction)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _proposal_is_clear(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cells:
                    continue
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_marker_bundle(
    grid: np.ndarray,
    right_start: int,
    bundle_type: str,
    colors: list[int],
) -> tuple[tuple[int, int], list[int]] | None:
    h, w = grid.shape
    if bundle_type == "pair":
        directions = list(random.choice([("W", "E"), ("N", "S")]))
    else:
        directions = [random.choice(["N", "S", "E", "W"])]

    for _ in range(PLACEMENT_RETRIES):
        marker = (
            random.randint(1, h - 2),
            random.randint(right_start + 1, w - 2),
        )
        templates = [_transform_template(random.choice(_EAST_TEMPLATES), direction) for direction in directions]
        bundles: list[tuple[int, set[tuple[int, int]]]] = []
        proposed = {marker}
        ok = True
        for color, rel_cells in zip(colors, templates):
            abs_cells = {(marker[0] + dr, marker[1] + dc) for dr, dc in rel_cells}
            if any(c < right_start for _r, c in abs_cells):
                ok = False
                break
            if proposed & abs_cells:
                ok = False
                break
            bundles.append((color, abs_cells))
            proposed |= abs_cells
        if not ok or not _proposal_is_clear(grid, proposed):
            continue

        adjacent_colors = []
        for color, abs_cells in bundles:
            if any(abs(r - marker[0]) + abs(c - marker[1]) == 1 for r, c in abs_cells):
                adjacent_colors.append(color)
            else:
                ok = False
                break
        if not ok:
            continue

        grid[marker] = GREY
        for color, abs_cells in bundles:
            for r, c in abs_cells:
                grid[r, c] = color
        return marker, adjacent_colors
    return None


def _place_extra_cluster(grid: np.ndarray, right_start: int, color: int) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        direction = random.choice(["N", "S", "E", "W"])
        rel_cells = _transform_template(random.choice(_EAST_TEMPLATES), direction)
        rows = [r for r, _ in rel_cells]
        cols = [c for _, c in rel_cells]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        anchor = (
            random.randint(1 - r_min, h - 2 - r_max),
            random.randint(max(right_start, 1 - c_min), w - 2 - c_max),
        )
        abs_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in rel_cells}
        if all(c >= right_start for _r, c in abs_cells) and _proposal_is_clear(grid, abs_cells):
            for r, c in abs_cells:
                grid[r, c] = color
            return True
    return False


def _nested_frame_layout(n_frames: int):
    inner_h = random.randint(3, 5)
    inner_w = random.randint(3, 5)
    inner = (0, 0, inner_h - 1, inner_w - 1)
    bboxes = [inner]
    for _ in range(n_frames - 1):
        prev = bboxes[-1]
        top_gap = random.choice([2, 2, 3])
        left_gap = random.choice([2, 2, 3])
        bottom_gap = random.choice([2, 2, 3])
        right_gap = random.choice([2, 2, 3])
        bboxes.append(
            (
                prev[0] - top_gap,
                prev[1] - left_gap,
                prev[2] + bottom_gap,
                prev[3] + right_gap,
            )
        )
    outer = bboxes[-1]
    shift_r = -outer[0]
    shift_c = -outer[1]
    normalized = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in reversed(bboxes)
    ]
    outer_h = normalized[0][2] - normalized[0][0] + 1
    outer_w = normalized[0][3] - normalized[0][1] + 1
    return normalized, outer_h, outer_w


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(3, 5)
    frame_colors = random.sample(FRAME_COLOR_POOL, n_frames)
    bboxes, outer_h, outer_w = _nested_frame_layout(n_frames)

    if outer_h > 24 or outer_w > 22:
        return None

    min_h = max(outer_h + 2, 14)
    max_h = min(30, outer_h + 8)
    if min_h > max_h:
        return None
    h = random.randint(min_h, max_h)

    left_margin = random.randint(1, 2)
    sep = random.randint(2, 4)
    max_right_w = 30 - (left_margin + outer_w + sep + 1)
    if max_right_w < 7:
        return None
    right_w = random.randint(7, min(13, max_right_w))
    w = left_margin + outer_w + sep + right_w + 1

    top_margin = random.randint(1, h - outer_h - 1)
    frame_bboxes = [
        (top_margin + r0, left_margin + c0, top_margin + r1, left_margin + c1)
        for r0, c0, r1, c1 in bboxes
    ]

    grid = np.zeros((h, w), dtype=int)
    for color, bbox in zip(frame_colors, frame_bboxes):
        _draw_frame(grid, bbox, color)

    right_start = frame_bboxes[0][3] + sep
    if right_start >= w - 6:
        return None

    n_markers = random.randint(3, 5)
    n_pairs = random.randint(1, n_markers - 1)
    bundle_types = ["pair"] * n_pairs + ["single"] * (n_markers - n_pairs)
    random.shuffle(bundle_types)

    markers: list[tuple[tuple[int, int], list[int]]] = []
    coverage = frame_colors.copy()
    random.shuffle(coverage)
    for kind in bundle_types:
        if kind == "pair":
            chosen = []
            while len(chosen) < 2:
                if coverage:
                    color = coverage.pop()
                else:
                    color = random.choice(frame_colors)
                if color not in chosen:
                    chosen.append(color)
            placed = _place_marker_bundle(grid, right_start, kind, chosen)
        else:
            chosen = [coverage.pop()] if coverage else [random.choice(frame_colors)]
            placed = _place_marker_bundle(grid, right_start, kind, chosen)
        if placed is None:
            return None
        markers.append(placed)

    for _ in range(random.randint(0, 2)):
        if not _place_extra_cluster(grid, right_start, random.choice(frame_colors)):
            break

    priority = {color: rank for rank, color in enumerate(frame_colors)}
    expected = grid.copy()
    for (r, c), colors in markers:
        expected[r, c] = min(colors, key=lambda color: priority[color])

    try:
        result = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(result, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
