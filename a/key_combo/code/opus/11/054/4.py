from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GEN_RETRIES = 600
GEN_MARGIN = 1
GEN_ROWS = (18, 24)
GEN_COLS = (18, 24)
GEN_STEM = (3, 5)
GEN_H_HEIGHT = (2, 4)
GEN_H_WIDTH = (5, 8)
GEN_V_HEIGHT = (5, 8)
GEN_V_WIDTH = (2, 4)
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    comps: list[np.ndarray] = []
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
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best_area = 0
    best: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if not mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        continue
                    if area > best_area:
                        best_area = area
                        best = []
                    best.append((r0, c0, r1, c1))
    assert best_area > 0
    assert len(best) == 1, "each object must contain a unique largest solid rectangle"
    return best[0]


def _body_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _stem_shift(stem: np.ndarray, body_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = body_box
    rows, cols = np.where(stem)
    assert rows.size > 0, "expected a connector stem"

    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))
    if len(unique_rows) == 1:
        row = unique_rows[0]
        cols_sorted = sorted(map(int, cols))
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
        assert r0 <= row <= r1
        if cols_sorted[-1] < c0:
            return 0, -1, c0 - cols_sorted[0]
        if c1 < cols_sorted[0]:
            return 0, 1, cols_sorted[-1] - c1
        raise AssertionError("horizontal stem must lie fully outside the body")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        rows_sorted = sorted(map(int, rows))
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
        assert c0 <= col <= c1
        if rows_sorted[-1] < r0:
            return -1, 0, r0 - rows_sorted[0]
        if r1 < rows_sorted[0]:
            return 1, 0, rows_sorted[-1] - r1
        raise AssertionError("vertical stem must lie fully outside the body")

    raise AssertionError("stem must be a straight one-cell-wide segment")


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int, steps: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _intersect_box(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _parse_object(grid: np.ndarray, color: int, bg: int) -> dict:
    mask = grid == int(color)
    comps = _components_mask(mask, connectivity=4)
    assert len(comps) == 1, "each colored object must be a single 4-connected component"
    box = _largest_solid_rectangle(mask)
    body = _body_mask(mask.shape, box)
    stem = mask & ~body
    dr, dc, dist = _stem_shift(stem, box)
    final_box = _shift_box(box, dr, dc, dist)
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w > h:
        orientation = "horizontal"
    elif h > w:
        orientation = "vertical"
    else:
        raise AssertionError("body aspect ratio must distinguish horizontal from vertical")

    fr0, fc0, fr1, fc1 = final_box
    H, W = grid.shape
    assert 0 <= fr0 <= fr1 < H and 0 <= fc0 <= fc1 < W, "moved body must stay in bounds"
    return {
        "color": int(color),
        "mask": mask,
        "body_box": box,
        "body_height": int(h),
        "body_width": int(w),
        "dr": int(dr),
        "dc": int(dc),
        "distance": int(dist),
        "orientation": orientation,
        "final_box": final_box,
        "bg": int(bg),
    }


def _expected_from_objects(grid_shape: tuple[int, int], horizontal: dict, vertical: dict, bg: int) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    _paint_box(out, horizontal["final_box"], horizontal["color"])
    _paint_box(out, vertical["final_box"], vertical["color"])

    overlap = _intersect_box(horizontal["final_box"], vertical["final_box"])
    assert overlap is not None
    r0, c0, r1, c1 = overlap
    assert r1 - r0 + 1 >= 2 and c1 - c0 + 1 >= 2, "overlap must be at least 2x2"
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = horizontal["color"] if (r + c) % 2 == 0 else vertical["color"]
    return out


def _parse_scene(grid: np.ndarray) -> tuple[dict, dict, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    colors = _colors_present(g, bg)
    assert len(colors) == 2, "expected exactly two foreground colors"
    objs = [_parse_object(g, color, bg) for color in colors]
    horizontals = [obj for obj in objs if obj["orientation"] == "horizontal"]
    verticals = [obj for obj in objs if obj["orientation"] == "vertical"]
    assert len(horizontals) == 1 and len(verticals) == 1, "expected one horizontal and one vertical body"
    return horizontals[0], verticals[0], bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide both rectangle bodies along their stems and weave the rectangular overlap.

    Input structure:
    - A black-background grid contains exactly two differently colored foreground objects.
    - One object has a wide solid horizontal rectangle plus a one-cell-wide horizontal stem attached on its left or right edge.
    - The other object has a tall solid vertical rectangle plus a one-cell-wide vertical stem attached on its top or bottom edge.

    Rule:
    - For each object, identify its unique largest solid rectangle and erase the whole original object, including the stem.
    - Translate each rectangle in the stem direction by exactly the stem length, then redraw only the moved rectangles.
    - Compute the overlap of the moved rectangles; in that overlap, cells with even `(r + c)` take the horizontal rectangle color and odd cells take the vertical rectangle color.
    - Outside the overlap, each moved rectangle keeps its own original color.

    Key insight:
    - The stems are not output shapes; they jointly encode both motion vectors, and only after both slides does the overlap region become visible and need the parity-based weave.
    """
    horizontal, vertical, bg = _parse_scene(np.asarray(grid, dtype=int))
    return _expected_from_objects(np.asarray(grid).shape, horizontal, vertical, bg)


def _render_input(spec: dict) -> np.ndarray:
    H, W = spec["shape"]
    grid = np.full((H, W), BG, dtype=int)

    hr0, hc0, hr1, hc1 = spec["h_input_box"]
    vr0, vc0, vr1, vc1 = spec["v_input_box"]
    _paint_box(grid, (hr0, hc0, hr1, hc1), spec["h_color"])
    _paint_box(grid, (vr0, vc0, vr1, vc1), spec["v_color"])

    if spec["h_side"] == "left":
        grid[spec["h_stem_row"], hc1 + 1 : hc1 + 1 + spec["h_len"]] = spec["h_color"]
    else:
        grid[spec["h_stem_row"], hc0 - spec["h_len"] : hc0] = spec["h_color"]

    if spec["v_side"] == "top":
        grid[vr1 + 1 : vr1 + 1 + spec["v_len"], spec["v_stem_col"]] = spec["v_color"]
    else:
        grid[vr0 - spec["v_len"] : vr0, spec["v_stem_col"]] = spec["v_color"]

    return grid


def _render_expected(spec: dict) -> np.ndarray:
    horizontal = {
        "final_box": spec["h_final_box"],
        "color": spec["h_color"],
    }
    vertical = {
        "final_box": spec["v_final_box"],
        "color": spec["v_color"],
    }
    return _expected_from_objects(spec["shape"], horizontal, vertical, BG)


def _render_shortcut(spec: dict, *, move_h: bool, move_v: bool, weave: bool) -> np.ndarray:
    out = np.full(spec["shape"], BG, dtype=int)
    h_box = spec["h_final_box"] if move_h else spec["h_input_box"]
    v_box = spec["v_final_box"] if move_v else spec["v_input_box"]
    _paint_box(out, h_box, spec["h_color"])
    _paint_box(out, v_box, spec["v_color"])
    overlap = _intersect_box(h_box, v_box)
    if overlap is not None and weave:
        r0, c0, r1, c1 = overlap
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = spec["h_color"] if (r + c) % 2 == 0 else spec["v_color"]
    return out


def _sample_spec() -> dict | None:
    H = random.randint(*GEN_ROWS)
    W = random.randint(*GEN_COLS)
    h_color, v_color = random.sample(GEN_COLORS, 2)

    hh = random.randint(*GEN_H_HEIGHT)
    hw = random.randint(*GEN_H_WIDTH)
    vh = random.randint(*GEN_V_HEIGHT)
    vw = random.randint(*GEN_V_WIDTH)
    h_len = random.randint(*GEN_STEM)
    v_len = random.randint(*GEN_STEM)
    if hh * hw <= hw + h_len:
        return None
    if vh * vw <= vh + v_len:
        return None

    ov_h = random.randint(2, min(hh, v_len))
    ov_w = random.randint(2, min(vw, h_len))
    if ov_h > hh or ov_w > vw:
        return None
    h_side = random.choice(["left", "right"])
    v_side = random.choice(["top", "bottom"])
    row_lo = GEN_MARGIN + (vh + v_len - ov_h if v_side == "top" else 0)
    row_hi = H - GEN_MARGIN - hh - (vh + v_len - ov_h if v_side == "bottom" else 0)
    col_lo = GEN_MARGIN + (hw + h_len - ov_w if h_side == "left" else 0)
    col_hi = W - GEN_MARGIN - vw - (hw + h_len - ov_w if h_side == "right" else 0)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    hr0 = random.randint(row_lo, row_hi)
    vc0 = random.randint(col_lo, col_hi)
    hr1 = hr0 + hh - 1
    vc1 = vc0 + vw - 1

    if v_side == "top":
        vr0 = hr0 + ov_h - vh
        overlap_r0 = hr0
    else:
        vr0 = hr1 - ov_h + 1
        overlap_r0 = hr1 - ov_h + 1
    vr1 = vr0 + vh - 1

    if h_side == "left":
        hc0 = vc0 + ov_w - hw
        overlap_c0 = vc0
    else:
        hc0 = vc1 - ov_w + 1
        overlap_c0 = vc1 - ov_w + 1
    hc1 = hc0 + hw - 1

    overlap_box = (overlap_r0, overlap_c0, overlap_r0 + ov_h - 1, overlap_c0 + ov_w - 1)
    h_final_box = (hr0, hc0, hr1, hc1)
    v_final_box = (vr0, vc0, vr1, vc1)

    if h_side == "left":
        h_input_box = (
            h_final_box[0],
            h_final_box[1] - h_len,
            h_final_box[2],
            h_final_box[3] - h_len,
        )
    else:
        h_input_box = (
            h_final_box[0],
            h_final_box[1] + h_len,
            h_final_box[2],
            h_final_box[3] + h_len,
        )

    if v_side == "top":
        v_input_box = (
            v_final_box[0] - v_len,
            v_final_box[1],
            v_final_box[2] - v_len,
            v_final_box[3],
        )
    else:
        v_input_box = (
            v_final_box[0] + v_len,
            v_final_box[1],
            v_final_box[2] + v_len,
            v_final_box[3],
        )

    h_cols = (
        range(h_input_box[3] + 1, h_input_box[3] + 1 + h_len)
        if h_side == "left"
        else range(h_input_box[1] - h_len, h_input_box[1])
    )
    v_rows = (
        range(v_input_box[2] + 1, v_input_box[2] + 1 + v_len)
        if v_side == "top"
        else range(v_input_box[0] - v_len, v_input_box[0])
    )
    h_rows = list(range(h_input_box[0], h_input_box[2] + 1))
    v_cols = list(range(v_input_box[1], v_input_box[3] + 1))
    safe_pairs = [
        (row, col)
        for row in h_rows
        for col in v_cols
        if row not in v_rows or col not in h_cols
    ]
    if not safe_pairs:
        return None

    overlap_rows = set(range(overlap_box[0], overlap_box[2] + 1))
    overlap_cols = set(range(overlap_box[1], overlap_box[3] + 1))
    scored_pairs = []
    for row, col in safe_pairs:
        score = 0
        if row not in overlap_rows:
            score += 1
        if col not in overlap_cols:
            score += 1
        scored_pairs.append((score, row, col))
    best_score = max(score for score, _row, _col in scored_pairs)
    candidates = [(row, col) for score, row, col in scored_pairs if score == best_score]
    h_stem_row, v_stem_col = random.choice(candidates)

    spec = {
        "shape": (H, W),
        "h_color": int(h_color),
        "v_color": int(v_color),
        "h_side": h_side,
        "v_side": v_side,
        "h_len": int(h_len),
        "v_len": int(v_len),
        "h_input_box": h_input_box,
        "v_input_box": v_input_box,
        "h_final_box": h_final_box,
        "v_final_box": v_final_box,
        "overlap_box": overlap_box,
        "h_stem_row": int(h_stem_row),
        "v_stem_col": int(v_stem_col),
    }
    h_cells = _box_cells(h_input_box)
    v_cells = _box_cells(v_input_box)
    if h_side == "left":
        h_cells |= {(h_stem_row, c) for c in range(h_input_box[3] + 1, h_input_box[3] + 1 + h_len)}
    else:
        h_cells |= {(h_stem_row, c) for c in range(h_input_box[1] - h_len, h_input_box[1])}
    if v_side == "top":
        v_cells |= {(r, v_stem_col) for r in range(v_input_box[2] + 1, v_input_box[2] + 1 + v_len)}
    else:
        v_cells |= {(r, v_stem_col) for r in range(v_input_box[0] - v_len, v_input_box[0])}
    if h_cells & v_cells:
        return None
    return spec


def _validate_generated(spec: dict, grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.shape != spec["shape"] or expected.shape != spec["shape"]:
        return False
    if _get_bg(grid) != BG:
        return False

    try:
        horizontal, vertical, bg = _parse_scene(grid)
    except AssertionError:
        return False

    if bg != BG:
        return False
    if horizontal["color"] != spec["h_color"] or vertical["color"] != spec["v_color"]:
        return False
    if horizontal["distance"] != spec["h_len"] or vertical["distance"] != spec["v_len"]:
        return False
    if horizontal["final_box"] != spec["h_final_box"] or vertical["final_box"] != spec["v_final_box"]:
        return False

    overlap = _intersect_box(horizontal["final_box"], vertical["final_box"])
    if overlap != spec["overlap_box"]:
        return False

    if np.array_equal(_render_shortcut(spec, move_h=True, move_v=True, weave=False), expected):
        return False
    if np.array_equal(_render_shortcut(spec, move_h=False, move_v=True, weave=True), expected):
        return False
    if np.array_equal(_render_shortcut(spec, move_h=True, move_v=False, weave=True), expected):
        return False

    overlap_cells = expected[
        spec["overlap_box"][0] : spec["overlap_box"][2] + 1,
        spec["overlap_box"][1] : spec["overlap_box"][3] + 1,
    ]
    if len(set(map(int, overlap_cells.ravel()))) != 2:
        return False

    h_only = np.zeros(spec["shape"], dtype=bool)
    v_only = np.zeros(spec["shape"], dtype=bool)
    hr0, hc0, hr1, hc1 = spec["h_final_box"]
    vr0, vc0, vr1, vc1 = spec["v_final_box"]
    h_only[hr0 : hr1 + 1, hc0 : hc1 + 1] = True
    v_only[vr0 : vr1 + 1, vc0 : vc1 + 1] = True
    overlap_mask = h_only & v_only
    if not np.any(h_only & ~overlap_mask):
        return False
    if not np.any(v_only & ~overlap_mask):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    spec = _sample_spec()
    if spec is None:
        return None

    grid = _render_input(spec)
    expected = _render_expected(spec)
    try:
        predicted = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(predicted, expected):
        return None
    if not _validate_generated(spec, grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-overlap sliding-rectangles input")
