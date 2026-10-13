from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
SIDE_ORDER = ("top", "right", "bottom", "left")
_RNG = np.random.default_rng()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, bbox)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _row_candidate(grid: np.ndarray, r: int) -> tuple[int, int, int] | None:
    cols = np.flatnonzero(grid[r] != BG)
    if cols.size < 10:
        return None
    if not np.all(np.diff(cols) == 1):
        return None
    c0, c1 = int(cols[0]), int(cols[-1])
    interior = grid[r, c0 + 1 : c1]
    if interior.size < 8:
        return None
    if int(interior[0]) == BG or not np.all(interior == interior[0]):
        return None
    return c0, c1, int(interior[0])


def _col_candidate(grid: np.ndarray, c: int) -> tuple[int, int, int] | None:
    rows = np.flatnonzero(grid[:, c] != BG)
    if rows.size < 10:
        return None
    if not np.all(np.diff(rows) == 1):
        return None
    r0, r1 = int(rows[0]), int(rows[-1])
    interior = grid[r0 + 1 : r1, c]
    if interior.size < 8:
        return None
    if int(interior[0]) == BG or not np.all(interior == interior[0]):
        return None
    return r0, r1, int(interior[0])


def _find_frame(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    row_candidates = []
    col_candidates = []
    for r in range(H):
        cand = _row_candidate(g, r)
        if cand is not None:
            row_candidates.append((int(r), *cand))
    for c in range(W):
        cand = _col_candidate(g, c)
        if cand is not None:
            col_candidates.append((int(c), *cand))

    matches = []
    for top in row_candidates:
        for bottom in row_candidates:
            if top[0] >= bottom[0]:
                continue
            tr, c0, c1, top_color = top
            br, bc0, bc1, bottom_color = bottom
            if (c0, c1) != (bc0, bc1):
                continue
            if top_color == bottom_color:
                continue
            for left in col_candidates:
                for right in col_candidates:
                    if left[0] >= right[0]:
                        continue
                    lc, r0, r1, left_color = left
                    rc, rr0, rr1, right_color = right
                    if (r0, r1) != (rr0, rr1):
                        continue
                    if (r0, r1, lc, rc) != (tr, br, c0, c1):
                        continue
                    side_colors = {
                        "top": int(top_color),
                        "right": int(right_color),
                        "bottom": int(bottom_color),
                        "left": int(left_color),
                    }
                    if len(set(side_colors.values())) != 4:
                        continue
                    tl = int(g[r0, c0])
                    trc = int(g[r0, c1])
                    bl = int(g[r1, c0])
                    brc = int(g[r1, c1])
                    if tl not in {side_colors["top"], side_colors["left"]}:
                        continue
                    if trc not in {side_colors["top"], side_colors["right"]}:
                        continue
                    if bl not in {side_colors["bottom"], side_colors["left"]}:
                        continue
                    if brc not in {side_colors["bottom"], side_colors["right"]}:
                        continue
                    matches.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "side_colors": side_colors,
                        }
                    )
    assert len(matches) == 1, f"expected one frame, found {len(matches)}"
    return matches[0]


def _segment_cells(
    r: int,
    c: int,
    side: str,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    rr = int(r)
    cc = int(c)

    if side == "top":
        assert c0 < cc < c1
        cells = []
        for cur_r in range(min(rr, r0), max(rr, r0) + 1):
            if r0 <= cur_r <= r1:
                if cur_r == r1 and rr > r1:
                    continue
                cells.append((cur_r, cc))
        return cells
    if side == "bottom":
        assert c0 < cc < c1
        cells = []
        for cur_r in range(min(rr, r1), max(rr, r1) + 1):
            if r0 <= cur_r <= r1:
                if cur_r == r0 and rr < r0:
                    continue
                cells.append((cur_r, cc))
        return cells
    if side == "left":
        assert r0 < rr < r1
        cells = []
        for cur_c in range(min(cc, c0), max(cc, c0) + 1):
            if c0 <= cur_c <= c1:
                if cur_c == c1 and cc > c1:
                    continue
                cells.append((rr, cur_c))
        return cells
    assert side == "right"
    assert r0 < rr < r1
    cells = []
    for cur_c in range(min(cc, c1), max(cc, c1) + 1):
        if c0 <= cur_c <= c1:
            if cur_c == c0 and cc < c0:
                continue
            cells.append((rr, cur_c))
    return cells


def _histogram_start(inner_w: int) -> int:
    return 1 + (inner_w - 7) // 2


def _bar_columns(crop_shape: tuple[int, int]) -> dict[str, int]:
    h, w = map(int, crop_shape)
    inner_w = w - 2
    start = _histogram_start(inner_w)
    return {side: start + 2 * idx for idx, side in enumerate(SIDE_ORDER)}


def _render_from_metadata(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
    scatters: list[dict],
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    crop = _crop(grid, bbox)
    r0, c0, r1, c1 = map(int, bbox)
    counts = {side: 0 for side in SIDE_ORDER}
    line_only = crop.copy()
    color_to_side = {color: side for side, color in side_colors.items()}

    for scatter in scatters:
        side = str(scatter["side"])
        color = int(scatter["color"])
        rr = int(scatter["r"])
        cc = int(scatter["c"])
        assert color_to_side[color] == side
        counts[side] += 1
        for gr, gc in _segment_cells(rr, cc, side, bbox):
            line_only[gr - r0, gc - c0] = color

    out = line_only.copy()
    h, w = out.shape
    inner_h = h - 2
    inner_w = w - 2
    assert inner_h >= 6 and inner_w >= 8
    bar_cols = _bar_columns(out.shape)
    base_row = h - 2
    for side in SIDE_ORDER:
        height = counts[side]
        if height == 0:
            continue
        col = bar_cols[side]
        out[base_row - height + 1 : base_row + 1, col] = side_colors[side]
    return out, counts, line_only


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the colored rectangle frame, connect each scattered pixel to its same-colored side, and render side-count bars.

    Input structure:
    - A black-background grid contains one rectangular frame made from four complete sides.
    - The top, right, bottom, and left sides use four distinct colors from blue, red, green, and yellow; corner cells may take either adjacent side color.
    - Extra single pixels of those side colors are scattered off the border, either inside the frame interior or outside the frame.

    Rule:
    - Detect the rectangle and crop exactly to its bounding box.
    - For every scattered pixel, connect it orthogonally to the frame side with the same color: top/bottom colors connect vertically, left/right colors connect horizontally, clipping away any part that stays outside the crop while keeping non-target border sides unchanged.
    - Count how many scattered pixels belong to each side color, then draw a bottom-aligned 4-column histogram inside the frame, ordered left-to-right as top, right, bottom, left side.

    Key insight:
    - A scattered pixel's color simultaneously identifies which side it belongs to, which orthogonal projection to draw, and which histogram bar height to increase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame = _find_frame(g)
    bbox = frame["bbox"]
    side_colors = frame["side_colors"]
    border = _frame_border_mask(g.shape, bbox)
    color_to_side = {color: side for side, color in side_colors.items()}

    scatters = []
    for r, c in np.argwhere((g != BG) & ~border):
        color = int(g[r, c])
        assert color in color_to_side, f"unexpected scatter color {color}"
        side = color_to_side[color]
        scatters.append({"r": int(r), "c": int(c), "color": color, "side": side})

    out, _counts, _line_only = _render_from_metadata(g, bbox, side_colors, scatters)
    return out


def _corner_color(primary_color: int, primary_coord: int, secondary_color: int, secondary_coord: int) -> int:
    return int(primary_color if int(primary_coord) <= int(secondary_coord) else secondary_color)


def _random_counts() -> dict[str, int]:
    for _ in range(400):
        counts = _RNG.integers(1, 6, size=4)
        if 9 <= int(counts.sum()) <= 14 and len(set(map(int, counts))) >= 2:
            return {side: int(count) for side, count in zip(SIDE_ORDER, counts)}
    raise RuntimeError("failed to sample side counts")


def _choose_without_replacement(values: Iterable[int], k: int) -> list[int]:
    vals = list(values)
    choice = _RNG.choice(np.array(vals, dtype=int), size=k, replace=False)
    return [int(x) for x in choice.tolist()]


def _place_scatter(
    occupied: set[tuple[int, int]],
    candidates: list[tuple[int, int]],
) -> tuple[int, int]:
    for idx in _RNG.permutation(len(candidates)):
        cell = tuple(map(int, candidates[int(idx)]))
        if cell not in occupied:
            occupied.add(cell)
            return cell
    raise RuntimeError("failed to place scatter without overlap")


def _build_visible_scatter_plan(
    bbox: tuple[int, int, int, int],
) -> dict[str, dict[str, list[int]]]:
    r0, c0, r1, c1 = map(int, bbox)
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    split_r = len(inner_rows) // 2
    split_c = len(inner_cols) // 2
    upper_rows = inner_rows[:split_r]
    lower_rows = inner_rows[split_r:]
    left_cols = inner_cols[:split_c]
    right_cols = inner_cols[split_c:]
    assert len(upper_rows) >= 5 and len(lower_rows) >= 5
    assert len(left_cols) >= 5 and len(right_cols) >= 5
    return {
        "top": {"proj": right_cols, "depth": upper_rows[1:] if len(upper_rows) > 1 else upper_rows},
        "right": {"proj": lower_rows, "depth": right_cols[:-1] if len(right_cols) > 1 else right_cols},
        "bottom": {"proj": left_cols, "depth": lower_rows[:-1] if len(lower_rows) > 1 else lower_rows},
        "left": {"proj": upper_rows, "depth": left_cols[1:] if len(left_cols) > 1 else left_cols},
        "top_hidden": {"proj": inner_cols},
        "right_hidden": {"proj": inner_rows},
        "bottom_hidden": {"proj": inner_cols},
        "left_hidden": {"proj": inner_rows},
    }


def _construct_instance() -> tuple[np.ndarray, np.ndarray]:
    H = int(_RNG.integers(19, 27))
    W = int(_RNG.integers(19, 27))
    inner_h = int(_RNG.integers(10, 15))
    inner_w = int(_RNG.integers(10, 15))
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if H < frame_h + 2 or W < frame_w + 2:
        raise RuntimeError("grid too small for requested frame")

    r0 = int(_RNG.integers(1, H - frame_h))
    c0 = int(_RNG.integers(1, W - frame_w))
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    bbox = (r0, c0, r1, c1)

    side_colors = {
        side: int(color)
        for side, color in zip(SIDE_ORDER, _RNG.permutation(np.array(FRAME_COLORS, dtype=int)))
    }
    counts = _random_counts()
    layout = _build_visible_scatter_plan(bbox)

    grid = np.zeros((H, W), dtype=int)
    grid[r0, c0 : c1 + 1] = side_colors["top"]
    grid[r1, c0 : c1 + 1] = side_colors["bottom"]
    grid[r0 : r1 + 1, c0] = side_colors["left"]
    grid[r0 : r1 + 1, c1] = side_colors["right"]
    grid[r0, c0] = _corner_color(side_colors["top"], r0, side_colors["left"], c0)
    grid[r0, c1] = _corner_color(side_colors["top"], r0, side_colors["right"], c1)
    grid[r1, c0] = _corner_color(side_colors["bottom"], r1, side_colors["left"], c0)
    grid[r1, c1] = _corner_color(side_colors["bottom"], r1, side_colors["right"], c1)

    occupied: set[tuple[int, int]] = set()
    scatters: list[dict] = []

    for side in SIDE_ORDER:
        visible_proj = _choose_without_replacement(layout[side]["proj"], 1)[0]
        if side == "top":
            candidates = [(row, visible_proj) for row in layout[side]["depth"]]
        elif side == "bottom":
            candidates = [(row, visible_proj) for row in layout[side]["depth"]]
        elif side == "left":
            candidates = [(visible_proj, col) for col in layout[side]["depth"]]
        else:
            candidates = [(visible_proj, col) for col in layout[side]["depth"]]
        rr, cc = _place_scatter(occupied, candidates)
        grid[rr, cc] = side_colors[side]
        scatters.append({"r": rr, "c": cc, "color": side_colors[side], "side": side, "kind": "visible"})

        hidden_needed = counts[side] - 1
        if hidden_needed <= 0:
            continue

        if side == "top":
            hidden_proj = [x for x in layout["top_hidden"]["proj"] if x != visible_proj][:hidden_needed]
            for proj in hidden_proj:
                rr, cc = r0 - 1, int(proj)
                grid[rr, cc] = side_colors[side]
                scatters.append({"r": rr, "c": cc, "color": side_colors[side], "side": side, "kind": "hidden"})
        elif side == "bottom":
            hidden_proj = [x for x in reversed(layout["bottom_hidden"]["proj"]) if x != visible_proj][:hidden_needed]
            for proj in hidden_proj:
                rr, cc = r1 + 1, int(proj)
                grid[rr, cc] = side_colors[side]
                scatters.append({"r": rr, "c": cc, "color": side_colors[side], "side": side, "kind": "hidden"})
        elif side == "left":
            hidden_proj = [x for x in layout["left_hidden"]["proj"] if x != visible_proj][:hidden_needed]
            for proj in hidden_proj:
                rr, cc = int(proj), c0 - 1
                grid[rr, cc] = side_colors[side]
                scatters.append({"r": rr, "c": cc, "color": side_colors[side], "side": side, "kind": "hidden"})
        else:
            hidden_proj = [x for x in reversed(layout["right_hidden"]["proj"]) if x != visible_proj][:hidden_needed]
            for proj in hidden_proj:
                rr, cc = int(proj), c1 + 1
                grid[rr, cc] = side_colors[side]
                scatters.append({"r": rr, "c": cc, "color": side_colors[side], "side": side, "kind": "hidden"})

    expected, parsed_counts, line_only = _render_from_metadata(grid, bbox, side_colors, scatters)

    frame = _find_frame(grid)
    assert frame["bbox"] == bbox
    assert frame["side_colors"] == side_colors
    assert parsed_counts == counts
    assert any(scatter["kind"] == "hidden" for scatter in scatters)
    assert any(scatter["kind"] == "visible" for scatter in scatters)
    assert np.count_nonzero(line_only != _crop(grid, bbox)) >= 4
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            grid, expected = _construct_instance()
        except Exception:
            continue
        signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        return grid
    raise RuntimeError("failed to generate a unique valid input")
