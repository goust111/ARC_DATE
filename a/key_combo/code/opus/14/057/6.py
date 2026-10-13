import random
from collections import Counter, deque

import numpy as np


BG = 0
LEFT_COLORS = (1, 2, 3)
RIGHT_COLORS = (4, 6, 7)
GRID_H_RANGE = (13, 19)
SIDE_WIDTH_RANGE = (6, 11)
GAP_WIDTHS = (3, 4, 5, 6, 7)
COUNT_RANGE = (2, 4)
HEIGHT_RANGE = (2, 3)
WIDTH_RANGE = (2, 4)
GEN_ATTEMPTS = 600


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "area": int(len(cells)),
                }
            )
    return comps


def _gap_runs(grid: np.ndarray, bg: int = BG) -> list[tuple[int, int]]:
    full_bg = np.all(grid == bg, axis=0)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < grid.shape[1]:
        if not full_bg[c]:
            c += 1
            continue
        start = c
        while c + 1 < grid.shape[1] and full_bg[c + 1]:
            c += 1
        runs.append((start, c))
        c += 1
    return runs


def _select_gap(grid: np.ndarray, bg: int = BG) -> tuple[int, int]:
    w = grid.shape[1]
    mid = (w - 1) / 2.0
    candidates = []
    for start, end in _gap_runs(grid, bg):
        if start == 0 or end == w - 1:
            continue
        if not np.any(grid[:, :start] != bg):
            continue
        if not np.any(grid[:, end + 1 :] != bg):
            continue
        width = end - start + 1
        center = (start + end) / 2.0
        candidates.append((start, end, width, abs(center - mid)))
    assert candidates, "expected a central full-height gap"
    candidates.sort(key=lambda item: (-item[2], item[3], item[0]))
    best = candidates[0]
    return int(best[0]), int(best[1])


def _extract_rectangles(grid: np.ndarray, bg: int = BG) -> tuple[list[dict], tuple[int, int]]:
    gap_start, gap_end = _select_gap(grid, bg)
    comps = _find_components(grid, bg)
    rects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == comp["color"]), "each object must be a solid rectangle"
        side = "left" if c1 < gap_start else "right" if c0 > gap_end else None
        assert side is not None, "objects must stay outside the central gap"
        rect = dict(comp)
        rect["side"] = side
        rects.append(rect)
    return rects, (gap_start, gap_end)


def _project_columns(rects: list[dict]) -> list[int]:
    cols = set()
    for rect in rects:
        _r0, c0, _r1, c1 = rect["bbox"]
        cols.update(range(int(c0), int(c1) + 1))
    return sorted(cols)


def _solve(grid: np.ndarray, *, left_target_mode: str = "half", collisions: bool = True) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = BG
    rects, (gap_start, gap_end) = _extract_rectangles(g, bg)
    gap_width = int(gap_end - gap_start + 1)
    half = gap_width // 2
    if left_target_mode == "half":
        left_boundary = int(gap_start + half - 1)
        right_boundary = int(gap_end - half + 1)
    elif left_target_mode == "outer":
        left_boundary = int(gap_start - 1)
        right_boundary = int(gap_end + 1)
    else:
        raise ValueError(left_target_mode)

    left_rects = [rect for rect in rects if rect["side"] == "left"]
    right_rects = [rect for rect in rects if rect["side"] == "right"]
    assert COUNT_RANGE[0] <= len(left_rects) <= COUNT_RANGE[1]
    assert COUNT_RANGE[0] <= len(right_rects) <= COUNT_RANGE[1]

    left_order = sorted(
        left_rects,
        key=lambda rect: (gap_start - rect["bbox"][3] - 1, rect["bbox"][0], rect["bbox"][1]),
    )
    right_order = sorted(
        right_rects,
        key=lambda rect: (rect["bbox"][1] - gap_end - 1, rect["bbox"][0], rect["bbox"][1]),
    )

    placed_left: list[dict] = []
    placed_right: list[dict] = []
    out = np.full((h, w), bg, dtype=int)
    meta = {
        "gap": (gap_start, gap_end),
        "gap_width": gap_width,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "left_order": [],
        "right_order": [],
        "objects": [],
    }

    for side, order, placed in (
        ("left", left_order, placed_left),
        ("right", right_order, placed_right),
    ):
        for rect in order:
            r0, c0, r1, c1 = rect["bbox"]
            obj_w = rect["w"]
            if side == "left":
                final_c0 = int(left_boundary - obj_w + 1)
                blocker = None
                if collisions:
                    for other in placed:
                        if _intervals_overlap(r0, r1, other["r0"], other["r1"]):
                            candidate = int(other["c0"] - obj_w)
                            if candidate < final_c0:
                                final_c0 = candidate
                                blocker = other["id"]
                final_c1 = int(final_c0 + obj_w - 1)
                assert final_c0 >= c0 and final_c1 <= left_boundary
            else:
                final_c0 = int(right_boundary)
                blocker = None
                if collisions:
                    for other in placed:
                        if _intervals_overlap(r0, r1, other["r0"], other["r1"]):
                            candidate = int(other["c1"] + 1)
                            if candidate > final_c0:
                                final_c0 = candidate
                                blocker = other["id"]
                final_c1 = int(final_c0 + obj_w - 1)
                assert final_c0 >= right_boundary and final_c0 <= c0
            out[r0 : r1 + 1, final_c0 : final_c1 + 1] = int(rect["color"])
            record = {
                "id": len(meta["objects"]),
                "side": side,
                "color": int(rect["color"]),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(final_c0),
                "c1": int(final_c1),
                "src_c0": int(c0),
                "src_c1": int(c1),
                "width": int(obj_w),
                "distance": int(gap_start - c1 - 1 if side == "left" else c0 - gap_end - 1),
                "blocked_by": blocker,
            }
            placed.append(record)
            meta[f"{side}_order"].append(record)
            meta["objects"].append(record)

    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multiple rectangles from both sides into the two halves of a central gap, stacking on same-side collisions.

    Input structure:
    - The grid has black background and one central full-height vertical gap made of consecutive black columns.
    - The left side contains 2-4 solid monochromatic rectangles; the right side contains 2-4 solid monochromatic rectangles.
    - Objects stay outside the gap, keep fixed row spans, and may overlap in rows with other objects on the same side.

    Rule:
    - Split the gap into left and right half-zones. If the gap width is odd, leave the middle column black.
    - Process left objects from nearest to farthest from the gap, breaking equal distances by smaller top row, and slide each rightward until it reaches the left half-zone boundary or the first settled same-side blocker.
    - Process right objects analogously, sliding leftward into the right half-zone boundary or against earlier settled same-side blockers.
    - Redraw only the settled rectangles on a black canvas; shapes, colors, and row positions stay unchanged.

    Key insight:
    - The gap width does not merely mark empty space: its even/odd half split defines the common stopping boundary for every object, while ordered same-side collisions determine how farther objects stack behind nearer ones.
    """
    return _solve(grid)[0]


def _choose_colors(count: int, palette: tuple[int, ...]) -> list[int]:
    colors = list(palette)
    random.shuffle(colors)
    out = colors[: min(len(colors), count)]
    while len(out) < count:
        out.append(random.choice(palette))
    random.shuffle(out)
    return out


def _rows_have_overlap(rows: list[tuple[int, int]]) -> bool:
    for i, (r0a, r1a) in enumerate(rows):
        for r0b, r1b in rows[i + 1 :]:
            if _intervals_overlap(r0a, r1a, r0b, r1b):
                return True
    return False


def _sample_row_layout(count: int, band_top: int, band_bottom: int, need_overlap: bool) -> list[tuple[int, int]] | None:
    for _ in range(300):
        heights = [random.randint(*HEIGHT_RANGE) for _ in range(count)]
        top_idx = random.randrange(count)
        bottom_idx = random.randrange(count - 1)
        if bottom_idx >= top_idx:
            bottom_idx += 1
        rows = [None] * count
        rows[top_idx] = int(band_top)
        rows[bottom_idx] = int(band_bottom - heights[bottom_idx] + 1)
        valid = True
        for i in range(count):
            if rows[i] is not None:
                if rows[i] < band_top or rows[i] + heights[i] - 1 > band_bottom:
                    valid = False
                    break
                continue
            lo = band_top
            hi = band_bottom - heights[i] + 1
            if lo > hi:
                valid = False
                break
            rows[i] = random.randint(lo, hi)
        if not valid:
            continue
        spans = [(int(rows[i]), int(rows[i] + heights[i] - 1)) for i in range(count)]
        if min(r0 for r0, _ in spans) != band_top:
            continue
        if max(r1 for _, r1 in spans) != band_bottom:
            continue
        if need_overlap != _rows_have_overlap(spans):
            continue
        return spans
    return None


def _columns_contiguous(rects: list[dict]) -> bool:
    cols = _project_columns(rects)
    return bool(cols) and cols == list(range(cols[0], cols[-1] + 1))


def _no_overlap(rects: list[dict]) -> bool:
    for i, a in enumerate(rects):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in rects[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            if _intervals_overlap(ar0, ar1, br0, br1) and _intervals_overlap(ac0, ac1, bc0, bc1):
                return False
    return True


def _assign_left_columns(rects: list[dict], gap_start: int, need_tie: bool) -> bool:
    n = len(rects)
    order = list(range(n))
    random.shuffle(order)
    tie_pos = random.randrange(n - 1) if need_tie and n >= 2 else None
    for _ in range(600):
        right_edges = []
        prev = gap_start - 1 - random.randint(0, 1)
        ok = True
        for pos, idx in enumerate(order):
            width = rects[idx]["w"]
            min_edge = width - 1
            if pos == 0:
                edge = prev
            elif tie_pos is not None and pos == tie_pos + 1:
                edge = right_edges[-1]
            else:
                max_drop = min(3, prev - min_edge)
                if max_drop < 1:
                    ok = False
                    break
                edge = prev - random.randint(1, max_drop)
            if edge < min_edge:
                ok = False
                break
            right_edges.append(edge)
            prev = edge
        if not ok:
            continue
        candidate = []
        for idx, edge in zip(order, right_edges):
            r0, r1 = rects[idx]["rows"]
            width = rects[idx]["w"]
            c0 = edge - width + 1
            rect = dict(rects[idx])
            rect["bbox"] = (r0, c0, r1, edge)
            candidate.append(rect)
        original_order = [None] * n
        for idx, rect in zip(order, candidate):
            original_order[idx] = rect
        if not _no_overlap(original_order):
            continue
        if not _columns_contiguous(original_order):
            continue
        for i, rect in enumerate(original_order):
            rects[i]["bbox"] = rect["bbox"]
        return True
    return False


def _assign_right_columns(rects: list[dict], gap_end: int, grid_w: int, need_tie: bool) -> bool:
    n = len(rects)
    order = list(range(n))
    random.shuffle(order)
    tie_pos = random.randrange(n - 1) if need_tie and n >= 2 else None
    for _ in range(600):
        left_edges = []
        prev = gap_end + 1 + random.randint(0, 1)
        ok = True
        for pos, idx in enumerate(order):
            width = rects[idx]["w"]
            max_left = grid_w - width
            if pos == 0:
                edge = prev
            elif tie_pos is not None and pos == tie_pos + 1:
                edge = left_edges[-1]
            else:
                max_rise = min(3, max_left - prev)
                if max_rise < 1:
                    ok = False
                    break
                edge = prev + random.randint(1, max_rise)
            if edge > max_left:
                ok = False
                break
            left_edges.append(edge)
            prev = edge
        if not ok:
            continue
        candidate = []
        for idx, edge in zip(order, left_edges):
            r0, r1 = rects[idx]["rows"]
            width = rects[idx]["w"]
            rect = dict(rects[idx])
            rect["bbox"] = (r0, edge, r1, edge + width - 1)
            candidate.append(rect)
        original_order = [None] * n
        for idx, rect in zip(order, candidate):
            original_order[idx] = rect
        if not _no_overlap(original_order):
            continue
        if not _columns_contiguous(original_order):
            continue
        for i, rect in enumerate(original_order):
            rects[i]["bbox"] = rect["bbox"]
        return True
    return False


def _render_input(height: int, width: int, left_rects: list[dict], right_rects: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for rect in left_rects + right_rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _solve_ignore_collisions(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, collisions=False)[0]


def _solve_outer_gap(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, left_target_mode="outer")[0]


def _semantic_checks(grid: np.ndarray, collision_mode: str, tie_side: str | None) -> bool:
    out, meta = _solve(grid)
    h, w = grid.shape
    if not (h <= 30 and w <= 30):
        return False
    gap_start, gap_end = meta["gap"]
    gap_width = meta["gap_width"]
    runs = [
        (start, end)
        for start, end in _gap_runs(grid)
        if start > 0 and end < w - 1 and np.any(grid[:, :start] != BG) and np.any(grid[:, end + 1 :] != BG)
    ]
    if len([run for run in runs if run == (gap_start, gap_end)]) != 1:
        return False
    if max(end - start + 1 for start, end in runs) != gap_width:
        return False
    if gap_width % 2 == 1:
        center = (gap_start + gap_end) // 2
        if np.any(out[:, center] != BG):
            return False
    if np.array_equal(out, _solve_outer_gap(grid)):
        return False
    if collision_mode == "none":
        if any(obj["blocked_by"] is not None for obj in meta["objects"]):
            return False
    else:
        if collision_mode in ("left", "both") and not any(obj["side"] == "left" and obj["blocked_by"] is not None for obj in meta["objects"]):
            return False
        if collision_mode in ("right", "both") and not any(obj["side"] == "right" and obj["blocked_by"] is not None for obj in meta["objects"]):
            return False
    if collision_mode != "none" and np.array_equal(out, _solve_ignore_collisions(grid)):
        return False
    if tie_side is not None:
        side_order = meta[f"{tie_side}_order"]
        distances = [obj["distance"] for obj in side_order]
        if all(a != b for a, b in zip(distances, distances[1:])):
            return False
    if np.array_equal(grid, out):
        return False
    sig = Counter(int(x) for x in out.flatten() if int(x) != BG)
    if max(sig.values()) == out.size:
        return False
    return True


def _make_side_rects(rows: list[tuple[int, int]], colors: list[int]) -> list[dict]:
    rects = []
    widths = [random.randint(*WIDTH_RANGE) for _ in rows]
    for i, ((r0, r1), color, width) in enumerate(zip(rows, colors, widths)):
        rects.append({"id": i, "rows": (r0, r1), "w": int(width), "color": int(color)})
    return rects


def _gen_attempt() -> np.ndarray | None:
    gap_width = random.choice(GAP_WIDTHS)
    left_count = random.randint(*COUNT_RANGE)
    right_count = random.randint(*COUNT_RANGE)
    collision_mode = random.choice(["left", "right", "both", "left", "right", "both", "none"])
    tie_side = random.choice([None, None, "left", "right"])
    left_overlap = collision_mode in ("left", "both")
    right_overlap = collision_mode in ("right", "both")

    left_space = random.randint(*SIDE_WIDTH_RANGE)
    right_space = random.randint(*SIDE_WIDTH_RANGE)
    width = left_space + gap_width + right_space
    if width > 30:
        return None
    gap_start = left_space
    gap_end = gap_start + gap_width - 1

    min_band = max(6, 2 * max(left_count if not left_overlap else 2, right_count if not right_overlap else 2))
    height = random.randint(max(GRID_H_RANGE[0], min_band + 2), GRID_H_RANGE[1])
    max_band = min(height - 2, 10)
    if min_band > max_band:
        return None
    band_height = random.randint(min_band, max_band)
    band_top = random.randint(1, height - band_height - 1)
    band_bottom = band_top + band_height - 1

    left_rows = _sample_row_layout(left_count, band_top, band_bottom, left_overlap)
    right_rows = _sample_row_layout(right_count, band_top, band_bottom, right_overlap)
    if left_rows is None or right_rows is None:
        return None

    left_rects = _make_side_rects(left_rows, _choose_colors(left_count, LEFT_COLORS))
    right_rects = _make_side_rects(right_rows, _choose_colors(right_count, RIGHT_COLORS))

    if not _assign_left_columns(left_rects, gap_start, tie_side == "left"):
        return None
    if not _assign_right_columns(right_rects, gap_end, width, tie_side == "right"):
        return None

    grid = _render_input(height, width, left_rects, right_rects)
    try:
        if not _semantic_checks(grid, collision_mode, tie_side):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-stack sample")
