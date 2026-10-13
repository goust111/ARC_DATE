from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

_GEN_MAX_ATTEMPTS = 800
_GEN_BRANCHES = 4
_GEN_COUNTER = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    blue_components = _component_coords(grid == BLUE)
    assert blue_components, "expected a blue rectangle"

    solid_rects: list[tuple[int, int, int, int, int]] = []
    for comp in blue_components:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        area = int(comp.shape[0])
        if area == (r1 - r0 + 1) * (c1 - c0 + 1):
            solid_rects.append((area, r0, c0, r1, c1))

    assert solid_rects, "expected at least one solid blue rectangle"
    solid_rects.sort(reverse=True)
    assert len(solid_rects) == 1 or solid_rects[0][0] > solid_rects[1][0], "expected one largest blue rectangle"
    _area, r0, c0, r1, c1 = solid_rects[0]
    return r0, c0, r1, c1


def _marker_sets(grid: np.ndarray, rect: tuple[int, int, int, int]) -> dict[str, set[int]]:
    r0, c0, r1, c1 = rect
    markers = np.argwhere(grid == ORANGE)
    assert len(markers) > 0, "expected orange markers"

    marked = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    for r, c in markers:
        rr = int(r)
        cc = int(c)
        side_hits = 0
        if rr < r0 and c0 <= cc <= c1:
            marked["top"].add(cc)
            side_hits += 1
        if rr > r1 and c0 <= cc <= c1:
            marked["bottom"].add(cc)
            side_hits += 1
        if cc < c0 and r0 <= rr <= r1:
            marked["left"].add(rr)
            side_hits += 1
        if cc > c1 and r0 <= rr <= r1:
            marked["right"].add(rr)
            side_hits += 1
        assert side_hits == 1, "each marker must align with exactly one rectangle side"
    return marked


def _extend_rays(grid: np.ndarray, rect: tuple[int, int, int, int], marked: dict[str, set[int]]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    r0, c0, r1, c1 = rect

    for c in range(c0, c1 + 1):
        if c not in marked["top"]:
            region = out[:r0, c]
            region[region == BLACK] = BLUE
        if c not in marked["bottom"]:
            region = out[r1 + 1 :, c]
            region[region == BLACK] = BLUE

    for r in range(r0, r1 + 1):
        if r not in marked["left"]:
            region = out[r, :c0]
            region[region == BLACK] = BLUE
        if r not in marked["right"]:
            region = out[r, c1 + 1 :]
            region[region == BLACK] = BLUE

    assert out.shape == (h, w)
    return out


def _closed_black_components(grid: np.ndarray) -> list[np.ndarray]:
    closed: list[np.ndarray] = []
    for comp in _component_coords(grid == BLACK):
        rows = comp[:, 0]
        cols = comp[:, 1]
        touches_border = (
            int(rows.min()) == 0
            or int(cols.min()) == 0
            or int(rows.max()) == grid.shape[0] - 1
            or int(cols.max()) == grid.shape[1] - 1
        )
        if not touches_border:
            closed.append(comp)
    return closed


def _area_color(area: int) -> int:
    if area <= 4:
        return RED
    if area <= 9:
        return GREEN
    return YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue rectangle into four directional rays, use orange side markers to suppress selected rows or columns, then fill the enclosed black pockets by area.

    Input structure:
    - The grid is a black background containing one solid blue rectangle and several orange singleton markers.
    - Every orange marker lies in exactly one side band aligned with the rectangle: above, below, left, or right.
    - The rectangle is separated from the grid border by black space on every side.

    Rule:
    - Keep the blue rectangle and orange markers.
    - Extend blue stripes from every rectangle column upward and downward to the grid edge, except the columns blocked by orange markers on that side.
    - Extend blue stripes from every rectangle row leftward and rightward to the grid edge, except the rows blocked by orange markers on that side.
    - After extension, find each 4-connected black region that does not touch the grid border and fill it by area: 1-4 cells red, 5-9 cells green, 10+ cells yellow.
    - Leave all border-connected black background cells unchanged.

    Key insight:
    - The orange markers do not mark pockets directly; they remove specific rays, and those missing rays change which black background regions stay open and which become enclosed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect = _find_rectangle(g)
    marked = _marker_sets(g, rect)
    extended = _extend_rays(g, rect, marked)
    out = extended.copy()

    for comp in _closed_black_components(extended):
        out[comp[:, 0], comp[:, 1]] = _area_color(int(comp.shape[0]))
    return out


def _side_length(side: str, rect_h: int, rect_w: int) -> int:
    return rect_w if side in ("top", "bottom") else rect_h


def _internal_run_mask(length: int, run_len: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if length <= 2:
        return mask
    start_min = 1
    start_max = length - run_len - 1
    start = int(np.random.randint(start_min, start_max + 1))
    mask[start : start + run_len] = True
    return mask


def _edge_only_mask(length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if np.random.random() < 0.6:
        mask[0] = True
    if np.random.random() < 0.6:
        mask[-1] = True
    if not mask.any():
        mask[int(np.random.choice([0, length - 1]))] = True
    return mask


def _sample_masks(
    rect_h: int,
    rect_w: int,
    branch: int,
) -> dict[str, np.ndarray]:
    lengths = {
        "top": rect_w,
        "bottom": rect_w,
        "left": rect_h,
        "right": rect_h,
    }
    pocket_candidates = [side for side, length in lengths.items() if length >= 3]
    assert len(pocket_candidates) >= 2

    if branch == 0:
        preferred = ["top", "bottom", "left"]
    elif branch == 1:
        preferred = ["left", "right", "top"]
    elif branch == 2:
        preferred = ["top", "right", "bottom"]
    else:
        preferred = ["top", "bottom", "left", "right"]

    pocket_sides = [side for side in preferred if side in pocket_candidates][: int(np.random.choice([2, 3]))]
    if len(pocket_sides) < 2:
        pocket_sides = list(np.random.choice(pocket_candidates, size=2, replace=False))

    masks: dict[str, np.ndarray] = {}
    for side, length in lengths.items():
        if side in pocket_sides:
            max_internal = max(1, length - 2)
            if branch == 3:
                run_len = int(np.random.randint(1, max_internal + 1))
            else:
                hi = min(max_internal, 2 if length <= 4 else 3)
                run_len = int(np.random.randint(1, hi + 1))
            mask = _internal_run_mask(length, run_len)
            if np.random.random() < 0.45:
                extra = _edge_only_mask(length)
                mask |= extra
                if mask.all():
                    mask[np.where(extra)[0][0]] = False
            masks[side] = mask
        else:
            if np.random.random() < 0.75:
                masks[side] = _edge_only_mask(length)
            else:
                masks[side] = np.zeros(length, dtype=bool)
    return masks


def _render_input(
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
    rect_h: int,
    rect_w: int,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    h = top_margin + rect_h + bottom_margin
    w = left_margin + rect_w + right_margin
    grid = np.zeros((h, w), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    for idx, marked in enumerate(masks["top"]):
        if marked:
            grid[0, c0 + idx] = ORANGE
    for idx, marked in enumerate(masks["bottom"]):
        if marked:
            grid[h - 1, c0 + idx] = ORANGE
    for idx, marked in enumerate(masks["left"]):
        if marked:
            grid[r0 + idx, 0] = ORANGE
    for idx, marked in enumerate(masks["right"]):
        if marked:
            grid[r0 + idx, w - 1] = ORANGE

    return grid


def _analyze_semantics(inp: np.ndarray, out: np.ndarray) -> dict[str, object] | None:
    try:
        rect = _find_rectangle(inp)
        marked = _marker_sets(inp, rect)
        extended = _extend_rays(inp, rect, marked)
    except AssertionError:
        return None

    if not np.array_equal(out, main(inp)):
        return None
    if not np.array_equal(extended[out == BLUE], np.full(np.count_nonzero(out == BLUE), BLUE, dtype=int)):
        return None

    closed = _closed_black_components(extended)
    if len(closed) < 2:
        return None

    black_components = _component_coords(extended == BLACK)
    open_components = []
    for comp in black_components:
        rows = comp[:, 0]
        cols = comp[:, 1]
        if (
            int(rows.min()) == 0
            or int(cols.min()) == 0
            or int(rows.max()) == extended.shape[0] - 1
            or int(cols.max()) == extended.shape[1] - 1
        ):
            open_components.append(comp)
    if not open_components:
        return None

    pocket_sizes = sorted(int(comp.shape[0]) for comp in closed)
    fill_bins = {_area_color(area) for area in pocket_sizes}
    if len(fill_bins) < 2:
        return None

    r0, c0, r1, c1 = rect
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    side_pocket_counts = {side: 0 for side in ("top", "bottom", "left", "right")}

    for comp in closed:
        rows = comp[:, 0]
        cols = comp[:, 1]
        rr0, cc0, rr1, cc1 = _bbox(comp)
        if rr1 < r0 and c0 <= cc0 <= cc1 <= c1:
            side_pocket_counts["top"] += 1
        elif rr0 > r1 and c0 <= cc0 <= cc1 <= c1:
            side_pocket_counts["bottom"] += 1
        elif cc1 < c0 and r0 <= rr0 <= rr1 <= r1:
            side_pocket_counts["left"] += 1
        elif cc0 > c1 and r0 <= rr0 <= rr1 <= r1:
            side_pocket_counts["right"] += 1
        else:
            return None

    active_marker_sides = sum(bool(marked[side]) for side in marked)
    pocket_sides = sum(count > 0 for count in side_pocket_counts.values())
    if active_marker_sides < 3 or pocket_sides < 2:
        return None

    # Require visible evidence that suppressed edge positions and enclosed internal positions both exist.
    edge_marked = False
    internal_marked = False
    for side, side_marked in marked.items():
        if side in ("top", "bottom"):
            offset = c0
            length = rect_w
        else:
            offset = r0
            length = rect_h
        for pos in side_marked:
            idx = pos - offset
            if idx in (0, length - 1):
                edge_marked = True
            else:
                internal_marked = True
    if not edge_marked or not internal_marked:
        return None

    return {
        "rect": rect,
        "pocket_sizes": pocket_sizes,
        "fill_bins": fill_bins,
        "side_pocket_counts": side_pocket_counts,
    }


def _pick_margins(branch: int) -> tuple[int, int, int, int]:
    if branch == 0:
        top = int(np.random.randint(4, 7))
        bottom = int(np.random.randint(6, 10))
        left = int(np.random.randint(4, 8))
        right = int(np.random.randint(5, 9))
    elif branch == 1:
        top = int(np.random.randint(5, 9))
        bottom = int(np.random.randint(4, 8))
        left = int(np.random.randint(6, 10))
        right = int(np.random.randint(4, 7))
    elif branch == 2:
        top = int(np.random.randint(4, 6))
        bottom = int(np.random.randint(7, 10))
        left = int(np.random.randint(7, 10))
        right = int(np.random.randint(4, 6))
    else:
        top = int(np.random.randint(4, 9))
        bottom = int(np.random.randint(4, 9))
        left = int(np.random.randint(4, 9))
        right = int(np.random.randint(4, 9))
    return top, bottom, left, right


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _attempt(branch: int) -> np.ndarray | None:
    rect_h = int(np.random.randint(3, 6))
    rect_w = int(np.random.randint(3, 6))
    top, bottom, left, right = _pick_margins(branch)
    if max(top + rect_h + bottom, left + rect_w + right) > 30:
        return None

    masks = _sample_masks(rect_h, rect_w, branch)
    inp = _render_input(top, bottom, left, right, rect_h, rect_w, masks)
    out = main(inp)
    semantics = _analyze_semantics(inp, out)
    if semantics is None:
        return None

    input_sig = _signature(inp)
    output_sig = _signature(out)
    if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(_GEN_MAX_ATTEMPTS):
        branch = _GEN_COUNTER % _GEN_BRANCHES
        _GEN_COUNTER += 1
        grid = _attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
