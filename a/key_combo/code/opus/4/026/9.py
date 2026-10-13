import random
from itertools import combinations

import numpy as np


BG = 0
GRAY = 5
BAR = 8
SEGMENT_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _analyze(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    h, w = g.shape
    bg = _get_bg(g)
    assert h >= 8 and w >= 8

    assert np.all(g[-1] == BAR), "expected a full teal bottom bar"
    gray_parts = _components(g == GRAY)
    assert len(gray_parts) == 1, "expected one gray coverage shape"
    gray_coords = gray_parts[0]
    shape_bbox = _bbox_coords(gray_coords)

    segments: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in {bg, GRAY, BAR}:
            continue
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            assert r0 == r1, "segments must be horizontal"
            assert coords.shape[0] == c1 - c0 + 1, "segments must be solid one-cell-thick runs"
            assert 2 <= coords.shape[0] <= 4, "unexpected segment length"
            assert r0 < h - 1, "non-bar object cannot be on the wall row"
            segments.append(
                {
                    "color": color,
                    "row": r0,
                    "c0": c0,
                    "c1": c1,
                    "coords": coords,
                }
            )

    segments.sort(key=lambda seg: seg["row"])
    assert 4 <= len(segments) <= 6, "expected 4-6 horizontal segments"
    rows = [seg["row"] for seg in segments]
    assert len(set(rows)) == len(rows), "segment rows must be unique"

    gray_rows_by_col = np.full(w, h, dtype=int)
    for c in range(w):
        rows_here = np.flatnonzero(g[:, c] == GRAY)
        if rows_here.size:
            gray_rows_by_col[c] = int(rows_here.min())

    return bg, gray_coords, segments, gray_rows_by_col


def _covered_columns(gray_rows_by_col: np.ndarray, row: int, c0: int, c1: int) -> list[bool]:
    return [int(gray_rows_by_col[c]) < int(row) for c in range(c0, c1 + 1)]


def _eligible_segments(gray_rows_by_col: np.ndarray, segments: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = []
    ineligible = []
    for seg in segments:
        covered = _covered_columns(gray_rows_by_col, seg["row"], seg["c0"], seg["c1"])
        tagged = dict(seg)
        tagged["covered"] = covered
        if all(covered):
            eligible.append(tagged)
        else:
            ineligible.append(tagged)
    return eligible, ineligible


def _paint_projections(grid: np.ndarray, segments: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    h, _w = grid.shape
    locked = grid != bg
    for seg in sorted(segments, key=lambda seg: seg["row"]):
        r0 = seg["row"] + 1
        r1 = h - 2
        if r0 > r1:
            continue
        view = out[r0 : r1 + 1, seg["c0"] : seg["c1"] + 1]
        mask = ~locked[r0 : r1 + 1, seg["c0"] : seg["c1"] + 1]
        view[mask] = int(seg["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the gray shape's per-column overhead coverage to gate which horizontal segments cast downward strips to the bottom wall.

    Input structure:
    - The background is black.
    - The last row is a solid teal wall bar.
    - A single large gray 4-connected shape sits in the upper half, and 4-6 colored one-cell-thick horizontal segments sit below or around it on distinct rows.

    Rule:
    - For each segment column, consider that column covered iff the same column contains at least one gray cell above the segment row.
    - A segment is eligible only when all of its columns are covered.
    - Each eligible segment paints its same-color rectangular strip from the next row down to the row just above the teal wall, and strips from higher segments are painted first so lower segments overwrite them in overlaps.
    - Keep the gray shape, wall bar, and all original segments unchanged.

    Key insight:
    - The gray object is not copied or extended; its irregular column-by-column overhead footprint acts as an all-or-nothing gate that decides which segments may project.
    """
    bg, _gray_coords, segments, gray_rows_by_col = _analyze(grid)
    eligible, _ineligible = _eligible_segments(gray_rows_by_col, segments)
    return _paint_projections(np.asarray(grid, dtype=int), eligible, bg)


def _build_shape(h: int, w: int, limit_row: int) -> tuple[np.ndarray, int, int, np.ndarray]:
    width = random.randint(max(10, w - 8), min(w - 3, 15))
    c0 = random.randint(1, w - width - 1)
    c1 = c0 + width - 1

    top = np.zeros(width, dtype=int)
    bottom = np.zeros(width, dtype=int)
    bottom_cap = limit_row - 2
    if bottom_cap < 6:
        raise ValueError("not enough room above segments")
    top[0] = random.randint(1, min(3, bottom_cap - 3))
    bottom[0] = min(bottom_cap, top[0] + random.randint(4, 8))

    for i in range(1, width):
        prev_t = int(top[i - 1])
        prev_b = int(bottom[i - 1])
        candidates = []
        for dt in (-2, -1, 0, 1, 2):
            t = prev_t + dt
            if not 1 <= t <= bottom_cap - 3:
                continue
            for db in (-2, -1, 0, 1, 2):
                b = prev_b + db
                if b - t + 1 < 4 or b - t + 1 > 9:
                    continue
                if b > bottom_cap:
                    continue
                if max(t, prev_t) <= min(b, prev_b):
                    candidates.append((t, b))
        if not candidates:
            raise ValueError("failed to continue shape ribbon")
        top[i], bottom[i] = random.choice(candidates)

    mask = np.zeros((h, w), dtype=bool)
    for i, c in enumerate(range(c0, c1 + 1)):
        mask[top[i] : bottom[i] + 1, c] = True
    area = int(mask.sum())
    frac = area / float(h * w)
    assert 0.18 <= frac <= 0.42
    assert len(_components(mask)) == 1
    return mask, c0, c1, top


def _rows_compatible(chosen: list[dict], cand: dict) -> bool:
    return all(abs(int(seg["row"]) - int(cand["row"])) >= 2 for seg in chosen)


def _overlap_cols(a: dict, b: dict) -> bool:
    return max(int(a["c0"]), int(b["c0"])) <= min(int(a["c1"]), int(b["c1"]))


def _choose_rows(h: int, count: int) -> list[int] | None:
    choices = list(range(8, h - 2, 2))
    if len(choices) < count:
        return None
    rows = sorted(random.sample(choices, count))
    return rows


def _choose_overlap_pair(shape_c0: int, shape_c1: int) -> tuple[dict, dict]:
    len_a = random.randint(3, 4)
    len_b = random.randint(3, 4)
    start_a = random.randint(shape_c0 + 1, shape_c1 - len_a)
    overlap_col = random.randint(start_a + 1, start_a + len_a - 1)
    min_b = max(shape_c0 + 1, overlap_col - len_b + 1)
    max_b = min(shape_c1 - len_b + 1, overlap_col)
    start_b = random.randint(min_b, max_b)
    return (
        {"c0": start_a, "c1": start_a + len_a - 1, "length": len_a, "covered": [True] * len_a},
        {"c0": start_b, "c1": start_b + len_b - 1, "length": len_b, "covered": [True] * len_b},
    )


def _edge_near_miss(shape_c0: int, shape_c1: int, w: int) -> dict | None:
    options = []
    for length in range(2, 5):
        if shape_c0 >= 1 and shape_c0 + length - 2 <= shape_c1:
            options.append(
                {
                    "c0": shape_c0 - 1,
                    "c1": shape_c0 + length - 2,
                    "length": length,
                    "covered": [False] + [True] * (length - 1),
                }
            )
        if shape_c1 + 1 < w and shape_c1 - length + 2 >= shape_c0:
            options.append(
                {
                    "c0": shape_c1 - length + 2,
                    "c1": shape_c1 + 1,
                    "length": length,
                    "covered": [True] * (length - 1) + [False],
                }
            )
    return random.choice(options) if options else None


def _extra_segment(shape_c0: int, shape_c1: int, w: int, want_eligible: bool) -> dict | None:
    if want_eligible:
        length = random.randint(2, 4)
        start = random.randint(shape_c0, shape_c1 - length + 1)
        return {"c0": start, "c1": start + length - 1, "length": length, "covered": [True] * length}
    return _edge_near_miss(shape_c0, shape_c1, w)


def _select_segments(rows: list[int], w: int, shape_c0: int, shape_c1: int) -> list[dict] | None:
    total = len(rows)
    first, second = _choose_overlap_pair(shape_c0, shape_c1)
    chosen = [first, second]

    bad = _edge_near_miss(shape_c0, shape_c1, w)
    if bad is None:
        return None
    chosen.append(bad)

    while len(chosen) < total:
        want_eligible = random.random() < 0.6
        extra = _extra_segment(shape_c0, shape_c1, w, want_eligible)
        if extra is None:
            return None
        chosen.append(extra)

    random.shuffle(chosen)
    for seg, row in zip(chosen, rows):
        seg["row"] = row
    chosen.sort(key=lambda seg: seg["row"])
    return chosen


def _render_input(h: int, w: int, shape_mask: np.ndarray, segments: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[-1, :] = BAR
    grid[shape_mask] = GRAY

    colors = random.sample(SEGMENT_COLORS, len(segments))
    for seg, color in zip(segments, colors):
        seg["color"] = color
        grid[int(seg["row"]), int(seg["c0"]) : int(seg["c1"]) + 1] = color
    return grid


def _all_projected_output(grid: np.ndarray) -> np.ndarray:
    bg, _gray_coords, segments, _gray_rows_by_col = _analyze(grid)
    return _paint_projections(grid, segments, bg)


def _reverse_order_output(grid: np.ndarray) -> np.ndarray:
    bg, _gray_coords, segments, gray_rows_by_col = _analyze(grid)
    eligible, _ineligible = _eligible_segments(gray_rows_by_col, segments)
    out = grid.copy()
    h, _w = grid.shape
    locked = grid != bg
    for seg in sorted(eligible, key=lambda seg: seg["row"], reverse=True):
        r0 = seg["row"] + 1
        r1 = h - 2
        if r0 > r1:
            continue
        view = out[r0 : r1 + 1, seg["c0"] : seg["c1"] + 1]
        mask = ~locked[r0 : r1 + 1, seg["c0"] : seg["c1"] + 1]
        view[mask] = int(seg["color"])
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    bg, _gray_coords, segments, gray_rows_by_col = _analyze(grid)
    eligible, ineligible = _eligible_segments(gray_rows_by_col, segments)
    if len(eligible) < 2 or len(ineligible) < 1:
        return False

    if not any(_overlap_cols(a, b) for a, b in combinations(eligible, 2)):
        return False

    if not any(sum(seg["covered"]) == len(seg["covered"]) - 1 for seg in ineligible):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _all_projected_output(grid)):
        return False
    if np.array_equal(out, _reverse_order_output(grid)):
        return False

    locked = grid != bg
    if not np.array_equal(out[locked], grid[locked]):
        return False

    changed = (out != grid) & (~locked)
    if int(changed.sum()) < 6:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 23)
    w = random.randint(18, 21)
    try:
        segment_count = random.randint(4, 6)
        rows = _choose_rows(h, segment_count)
        if rows is None:
            return None
        shape_limit = min(rows) - 1
        if shape_limit < 8:
            return None
        shape_mask, shape_c0, shape_c1, _top = _build_shape(h, w, shape_limit)
    except Exception:
        return None

    segments = _select_segments(rows, w, shape_c0, shape_c1)
    if segments is None:
        return None

    grid = _render_input(h, w, shape_mask, segments)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
