from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SLIDER_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")

FRAME_SIZE_RANGE = (4, 5)
FRAME_GAP = 2
FRAME_MARGIN = 2
GRID_MIN = 18
GRID_MAX = 30
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_side(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each slider must touch exactly one outer edge"
    return sides[0]


def _exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    return exterior


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    enclosed = (grid == BLACK) & ~_exterior_background(grid)
    return _components(enclosed)


def _detect_sliders(grid: np.ndarray) -> list[dict]:
    sliders: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BLACK, RED)):
        for coords in _components(grid == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            assert int(coords.shape[0]) == hh * ww, "slider must be a solid rectangle"
            assert 1 <= hh <= 3 and 1 <= ww <= 2, "slider dimensions out of range"
            side = _edge_side(box, grid.shape)
            sliders.append(
                {
                    "color": color,
                    "side": side,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                }
            )
    assert 2 <= len(sliders) <= 4, "expected 2 to 4 edge sliders"
    return sliders


def _region_hit_colors(region: np.ndarray, sliders: list[dict]) -> list[int]:
    rows = region[:, 0]
    cols = region[:, 1]
    hits: list[int] = []
    for slider in sliders:
        if slider["side"] in ("top", "bottom"):
            if np.any((cols >= slider["c0"]) & (cols <= slider["c1"])):
                hits.append(int(slider["color"]))
        else:
            if np.any((rows >= slider["r0"]) & (rows <= slider["r1"])):
                hits.append(int(slider["color"]))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge rectangle across the grid and use its swept band to recolor any enclosed region it crosses.

    Input structure:
    - The grid has a black background, red one-cell-thick wall segments, and 2-4 small solid colored rectangles touching exactly one outer edge.
    - Red walls and the grid boundary together create some fully enclosed black regions and some border-reachable black space.
    - Each colored rectangle is a 1-2 by 1-3 block whose color is one of blue, green, yellow, or orange.

    Rule:
    - Flood-fill from the border through black cells only; any black component not reached is an enclosed region.
    - Slide each edge rectangle straight toward the opposite edge, using its full swept row-band or column-band as a query path.
    - If a swept band contains any cell from one enclosed region, recolor that whole region to the slider's color; then erase all slider rectangles while leaving red walls unchanged.

    Key insight:
    - The slide leaves no visible trail; its only purpose is to test which enclosed regions lie inside that moving row-band or column-band and transfer the slider's color to them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30

    sliders = _detect_sliders(g)
    regions = _enclosed_regions(g)
    assert regions, "expected at least one enclosed region"

    out = g.copy()
    out[(out != BLACK) & (out != RED)] = BLACK

    for region in regions:
        hits = _region_hit_colors(region, sliders)
        assert len(hits) <= 1, "each enclosed region must be crossed by at most one slider"
        if hits:
            out[region[:, 0], region[:, 1]] = hits[0]
    return out


def _composition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _pack_intervals(total: int, sizes: list[int], min_gap: int) -> list[tuple[int, int]] | None:
    extra = total - sum(sizes) - min_gap * (len(sizes) - 1)
    if extra < 0:
        return None
    padding = _composition(extra, len(sizes) + 1)
    out: list[tuple[int, int]] = []
    pos = padding[0]
    for i, size in enumerate(sizes):
        start = pos
        end = start + size - 1
        out.append((start, end))
        pos = end + 1
        if i + 1 < len(sizes):
            pos += min_gap + padding[i + 1]
    return out


def _draw_rect_outline(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED


def _interior_coords(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return np.array(cells, dtype=int)


def _slider_cells(slider: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(int(slider["r0"]), int(slider["r1"]) + 1)
        for c in range(int(slider["c0"]), int(slider["c1"]) + 1)
    ]


def _band_hits_box(slider: dict, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if slider["side"] in ("top", "bottom"):
        return not (slider["c1"] < c0 or slider["c0"] > c1)
    return not (slider["r1"] < r0 or slider["r0"] > r1)


def _actual_hits(sliders: list[dict], frames: list[dict]) -> dict[int, int]:
    hits: dict[int, int] = {}
    for frame in frames:
        frame_hits = [slider for slider in sliders if _band_hits_box(slider, frame["interior"])]
        assert len(frame_hits) <= 1
        if frame_hits:
            hits[int(frame["id"])] = int(frame_hits[0]["color"])
    return hits


def _free_for_slider(grid: np.ndarray, occupied: set[tuple[int, int]], slider: dict) -> bool:
    h, w = grid.shape
    cells = _slider_cells(slider)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BLACK or (r, c) in occupied:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _hit_candidates(frame: dict, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    r0, c0, r1, c1 = frame["interior"]
    ih = r1 - r0 + 1
    iw = c1 - c0 + 1
    out: list[dict] = []

    for side in ("top", "bottom"):
        for rect_w in range(1, min(2, iw) + 1):
            for rect_h in range(1, 4):
                for c in range(c0, c1 - rect_w + 2):
                    if side == "top":
                        out.append({"side": side, "r0": 0, "c0": c, "r1": rect_h - 1, "c1": c + rect_w - 1})
                    else:
                        out.append({"side": side, "r0": h - rect_h, "c0": c, "r1": h - 1, "c1": c + rect_w - 1})

    for side in ("left", "right"):
        for rect_h in range(1, min(3, ih) + 1):
            for rect_w in range(1, 3):
                for r in range(r0, r1 - rect_h + 2):
                    if side == "left":
                        out.append({"side": side, "r0": r, "c0": 0, "r1": r + rect_h - 1, "c1": rect_w - 1})
                    else:
                        out.append({"side": side, "r0": r, "c0": w - rect_w, "r1": r + rect_h - 1, "c1": w - 1})
    random.shuffle(out)
    return out


def _miss_candidates(frames: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    unsafe_rows = set()
    unsafe_cols = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["interior"]
        unsafe_rows.update(range(r0, r1 + 1))
        unsafe_cols.update(range(c0, c1 + 1))

    out: list[dict] = []
    for side in ("top", "bottom"):
        for rect_w in (1, 2):
            for rect_h in range(1, 4):
                for c in range(0, w - rect_w + 1):
                    if any(col in unsafe_cols for col in range(c, c + rect_w)):
                        continue
                    if side == "top":
                        out.append({"side": side, "r0": 0, "c0": c, "r1": rect_h - 1, "c1": c + rect_w - 1})
                    else:
                        out.append({"side": side, "r0": h - rect_h, "c0": c, "r1": h - 1, "c1": c + rect_w - 1})

    for side in ("left", "right"):
        for rect_h in range(1, 4):
            for rect_w in (1, 2):
                for r in range(0, h - rect_h + 1):
                    if any(row in unsafe_rows for row in range(r, r + rect_h)):
                        continue
                    if side == "left":
                        out.append({"side": side, "r0": r, "c0": 0, "r1": r + rect_h - 1, "c1": rect_w - 1})
                    else:
                        out.append({"side": side, "r0": r, "c0": w - rect_w, "r1": r + rect_h - 1, "c1": w - 1})
    random.shuffle(out)
    return out


def _place_sliders(grid: np.ndarray, frames: list[dict]) -> tuple[list[dict], dict[int, int]] | None:
    n_frames = len(frames)
    n_sliders = random.randint(2, min(4, len(SLIDER_COLORS)))
    hit_count = random.randint(1, min(n_frames - 1, n_sliders - 1))
    hit_ids = set(random.sample([int(frame["id"]) for frame in frames], hit_count))
    colors = random.sample(SLIDER_COLORS, n_sliders)

    occupied: set[tuple[int, int]] = set()
    sliders: list[dict] = []
    intended_hits: dict[int, int] = {}
    color_idx = 0

    for frame in random.sample(frames, len(frames)):
        if int(frame["id"]) not in hit_ids:
            continue
        placed = None
        for cand in _hit_candidates(frame, grid.shape):
            cand = dict(cand)
            cand["color"] = colors[color_idx]
            if _free_for_slider(grid, occupied, cand):
                placed = cand
                break
        if placed is None:
            return None
        sliders.append(placed)
        intended_hits[int(frame["id"])] = int(placed["color"])
        for cell in _slider_cells(placed):
            occupied.add(cell)
        color_idx += 1

    misses_needed = n_sliders - hit_count
    miss_pool = _miss_candidates(frames, grid.shape)
    for cand in miss_pool:
        if misses_needed == 0:
            break
        cand = dict(cand)
        cand["color"] = colors[color_idx]
        if _free_for_slider(grid, occupied, cand):
            sliders.append(cand)
            for cell in _slider_cells(cand):
                occupied.add(cell)
            color_idx += 1
            misses_needed -= 1
    if misses_needed != 0:
        return None

    for slider in sliders:
        for r, c in _slider_cells(slider):
            grid[r, c] = int(slider["color"])
    return sliders, intended_hits


def _make_layout() -> tuple[np.ndarray, list[dict], tuple[int, int, int, int]] | None:
    n_frames = random.randint(2, 4)
    row_sizes = [random.randint(*FRAME_SIZE_RANGE) for _ in range(n_frames)]
    col_sizes = [random.randint(*FRAME_SIZE_RANGE) for _ in range(n_frames)]
    min_h = sum(row_sizes) + FRAME_GAP * (n_frames - 1) + 2 * FRAME_MARGIN
    min_w = sum(col_sizes) + FRAME_GAP * (n_frames - 1) + 2 * FRAME_MARGIN
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(max(GRID_MIN, min_h), min(GRID_MAX, min_h + 4))
    w = random.randint(max(GRID_MIN, min_w), min(GRID_MAX, min_w + 4))
    row_intervals = _pack_intervals(h - 2 * FRAME_MARGIN, row_sizes, FRAME_GAP)
    col_intervals = _pack_intervals(w - 2 * FRAME_MARGIN, col_sizes, FRAME_GAP)
    if row_intervals is None or col_intervals is None:
        return None

    row_intervals = [(r0 + FRAME_MARGIN, r1 + FRAME_MARGIN) for r0, r1 in row_intervals]
    col_intervals = [(c0 + FRAME_MARGIN, c1 + FRAME_MARGIN) for c0, c1 in col_intervals]
    perm = list(range(n_frames))
    random.shuffle(perm)

    grid = np.zeros((h, w), dtype=int)
    frames: list[dict] = []
    occupied_pairs = set()
    for i, (r0, r1) in enumerate(row_intervals):
        j = perm[i]
        c0, c1 = col_intervals[j]
        occupied_pairs.add((i, j))
        box = (r0, c0, r1, c1)
        _draw_rect_outline(grid, box)
        frames.append({"id": i, "bbox": box, "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1)})

    unused_pairs = [(i, j) for i in range(n_frames) for j in range(n_frames) if (i, j) not in occupied_pairs]
    open_i, open_j = random.choice(unused_pairs)
    open_box = (row_intervals[open_i][0], col_intervals[open_j][0], row_intervals[open_i][1], col_intervals[open_j][1])
    _draw_rect_outline(grid, open_box)
    side = random.choice(SIDES)
    if side in ("top", "bottom"):
        gap_c = random.randint(open_box[1] + 1, open_box[3] - 1)
        gap_r = open_box[0] if side == "top" else open_box[2]
    else:
        gap_r = random.randint(open_box[0] + 1, open_box[2] - 1)
        gap_c = open_box[1] if side == "left" else open_box[3]
    grid[gap_r, gap_c] = BLACK
    return grid, frames, open_box


def _verify_closed_regions(grid: np.ndarray, frames: list[dict], open_box: tuple[int, int, int, int]) -> bool:
    regions = _enclosed_regions(grid)
    if len(regions) != len(frames):
        return False

    expected_sets = {
        frozenset(tuple(cell) for cell in _interior_coords(frame["bbox"]))
        for frame in frames
    }
    actual_sets = {
        frozenset(tuple(cell) for cell in region)
        for region in regions
    }
    if actual_sets != expected_sets:
        return False

    open_interior = frozenset(tuple(cell) for cell in _interior_coords(open_box))
    return open_interior not in actual_sets


def _independent_expected(grid: np.ndarray, frames: list[dict], intended_hits: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    out[(out != BLACK) & (out != RED)] = BLACK
    for frame in frames:
        color = intended_hits.get(int(frame["id"]))
        if color is None:
            continue
        r0, c0, r1, c1 = frame["interior"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _attempt() -> np.ndarray | None:
    made = _make_layout()
    if made is None:
        return None
    grid, frames, open_box = made
    if not _verify_closed_regions(grid, frames, open_box):
        return None

    placed = _place_sliders(grid, frames)
    if placed is None:
        return None
    sliders, intended_hits = placed

    if len(intended_hits) == 0 or len(intended_hits) >= len(frames):
        return None
    actual_hits = _actual_hits(sliders, frames)
    if actual_hits != intended_hits:
        return None

    expected = _independent_expected(grid, frames, intended_hits)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-enclosure ARC input")
