import random
from collections import Counter

import numpy as np


BLACK = 0
MAX_GENERATION_ATTEMPTS = 300
PALETTE = tuple(range(10))


def _mode(values) -> int:
    cnt = Counter(int(v) for v in values)
    return int(max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r] == color) and color != BLACK:
            candidates.append(("h", r, color))
    for c in range(w):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color) and color != BLACK:
            candidates.append(("v", c, color))

    assert len(candidates) == 1, "expected exactly one full separator line"
    return candidates[0]


def _extract_side_info(
    grid: np.ndarray, orient: str, sep_idx: int, sep_color: int, ref_side: str
) -> tuple[dict, dict]:
    h, w = grid.shape

    if orient == "v":
        left = grid[:, :sep_idx]
        right = grid[:, sep_idx + 1 :]
        ref_slice = left if ref_side == "left" else right
        src_slice = right if ref_side == "left" else left
        ref_offset = 0 if ref_side == "left" else sep_idx + 1
        src_offset = sep_idx + 1 if ref_side == "left" else 0
    else:
        top = grid[:sep_idx, :]
        bottom = grid[sep_idx + 1 :, :]
        ref_slice = top if ref_side == "top" else bottom
        src_slice = bottom if ref_side == "top" else top
        ref_offset = 0 if ref_side == "top" else sep_idx + 1
        src_offset = sep_idx + 1 if ref_side == "top" else 0

    def build(side: np.ndarray, offset: int, is_vertical: bool) -> dict:
        mask = (side != BLACK) & (side != sep_color)
        r0, c0, r1, c1 = _bbox(mask)
        region = side[r0 : r1 + 1, c0 : c1 + 1]
        bg = _mode(region[(region != BLACK) & (region != sep_color)])

        if is_vertical:
            active_candidates = []
            for c in range(c0, c1 + 1):
                vals = side[r0 : r1 + 1, c]
                if np.any((vals != BLACK) & (vals != sep_color) & (vals != bg)):
                    active_candidates.append(c)
            assert len(active_candidates) == 1
            active = active_candidates[0]
            abs_bbox = (r0, c0 + offset, r1, c1 + offset)
            abs_active = active + offset
        else:
            active_candidates = []
            for r in range(r0, r1 + 1):
                vals = side[r, c0 : c1 + 1]
                if np.any((vals != BLACK) & (vals != sep_color) & (vals != bg)):
                    active_candidates.append(r)
            assert len(active_candidates) == 1
            active = active_candidates[0]
            abs_bbox = (r0 + offset, c0, r1 + offset, c1)
            abs_active = active + offset

        return {"bbox": abs_bbox, "bg": int(bg), "active": int(abs_active)}

    is_vertical = orient == "v"
    return build(ref_slice, ref_offset, is_vertical), build(src_slice, src_offset, is_vertical)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    orient, sep_idx, sep_color = _find_separator(g)

    if orient == "v":
        left_grey = int(np.count_nonzero(g[:, :sep_idx] == sep_color))
        right_grey = int(np.count_nonzero(g[:, sep_idx + 1 :] == sep_color))
        assert (left_grey == 0) != (right_grey == 0)
        ref_side = "left" if left_grey > 0 else "right"
    else:
        top_grey = int(np.count_nonzero(g[:sep_idx, :] == sep_color))
        bottom_grey = int(np.count_nonzero(g[sep_idx + 1 :, :] == sep_color))
        assert (top_grey == 0) != (bottom_grey == 0)
        ref_side = "top" if top_grey > 0 else "bottom"

    ref_info, src_info = _extract_side_info(g, orient, sep_idx, sep_color, ref_side)
    out = g.copy()

    if orient == "v":
        ref_col = ref_info["active"]
        src_col = src_info["active"]
        ref_bg = ref_info["bg"]
        src_bg = src_info["bg"]
        sr0, sc0, sr1, sc1 = src_info["bbox"]
        ref_rows = [
            r for r in range(h)
            if int(g[r, ref_col]) not in (BLACK, sep_color, ref_bg)
        ]
        assert ref_rows

        for r in ref_rows:
            ref_color = int(g[r, ref_col])
            src_color = int(g[r, src_col])
            assert src_color not in (BLACK, sep_color, src_bg)
            out[r, src_col] = src_bg

            source_on_right = src_col > sep_idx
            near_col = sc0 if source_on_right else sc1
            far_col = sc1 if source_on_right else sc0
            step = 1 if source_on_right else -1

            if src_color == ref_color:
                out[r, near_col] = src_color
                out[r, near_col + step] = src_color
            else:
                out[r, far_col] = src_color
    else:
        ref_row = ref_info["active"]
        src_row = src_info["active"]
        ref_bg = ref_info["bg"]
        src_bg = src_info["bg"]
        sr0, sc0, sr1, sc1 = src_info["bbox"]
        ref_cols = [
            c for c in range(w)
            if int(g[ref_row, c]) not in (BLACK, sep_color, ref_bg)
        ]
        assert ref_cols

        for c in ref_cols:
            ref_color = int(g[ref_row, c])
            src_color = int(g[src_row, c])
            assert src_color not in (BLACK, sep_color, src_bg)
            out[src_row, c] = src_bg

            source_on_bottom = src_row > sep_idx
            near_row = sr0 if source_on_bottom else sr1
            far_row = sr1 if source_on_bottom else sr0
            step = 1 if source_on_bottom else -1

            if src_color == ref_color:
                out[near_row, c] = src_color
                out[near_row + step, c] = src_color
            else:
                out[far_row, c] = src_color

    return out


def _rand_color(exclude: set[int]) -> int:
    choices = [c for c in PALETTE if c not in exclude]
    assert choices
    return int(random.choice(choices))


def _gen_vertical() -> np.ndarray:
    h = random.randint(9, 18)
    left_w = random.randint(3, 7)
    right_w = random.randint(3, 7)
    w = left_w + 1 + right_w
    sep = left_w

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    r0 = top_margin
    r1 = h - 1 - bottom_margin
    assert r1 - r0 + 1 >= 4

    ref_side = random.choice(["left", "right"])
    grey = _rand_color({BLACK})
    ref_bg = _rand_color({BLACK, grey})
    src_bg = _rand_color({BLACK, grey, ref_bg})

    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, sep] = grey
    grid[r0 : r1 + 1, :sep] = ref_bg if ref_side == "left" else src_bg
    grid[r0 : r1 + 1, sep + 1 :] = src_bg if ref_side == "left" else ref_bg

    ref_cols = list(range(0, sep)) if ref_side == "left" else list(range(sep + 1, w))
    src_cols = list(range(sep + 1, w)) if ref_side == "left" else list(range(0, sep))
    ref_col = random.choice(ref_cols)
    src_col = random.choice(src_cols)

    row_choices = list(range(r0 + 1, r1))
    count = random.randint(2, min(5, len(row_choices)))
    rows = sorted(random.sample(row_choices, count))
    match_count = random.randint(1, count - 1)
    matches = set(random.sample(rows, match_count))

    used = {BLACK, grey, ref_bg, src_bg}
    for r in rows:
        ref_color = _rand_color(used)
        if r in matches:
            src_color = ref_color
        else:
            src_color = _rand_color(used | {ref_color})
        grid[r, ref_col] = ref_color
        grid[r, src_col] = src_color

    marker_rows = list(range(0, r0)) + list(range(r1 + 1, h))
    assert marker_rows
    marker_count = random.randint(1, min(3, len(marker_rows)))
    marker_positions = random.sample(
        [(r, c) for r in marker_rows for c in ref_cols if c != ref_col],
        marker_count,
    )
    for r, c in marker_positions:
        grid[r, c] = grey

    return grid


def _gen_horizontal() -> np.ndarray:
    w = random.randint(9, 18)
    top_h = random.randint(3, 7)
    bottom_h = random.randint(3, 7)
    h = top_h + 1 + bottom_h
    sep = top_h

    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    c0 = left_margin
    c1 = w - 1 - right_margin
    assert c1 - c0 + 1 >= 4

    ref_side = random.choice(["top", "bottom"])
    grey = _rand_color({BLACK})
    ref_bg = _rand_color({BLACK, grey})
    src_bg = _rand_color({BLACK, grey, ref_bg})

    grid = np.full((h, w), BLACK, dtype=int)
    grid[sep, :] = grey
    grid[:sep, c0 : c1 + 1] = ref_bg if ref_side == "top" else src_bg
    grid[sep + 1 :, c0 : c1 + 1] = src_bg if ref_side == "top" else ref_bg

    ref_rows = list(range(0, sep)) if ref_side == "top" else list(range(sep + 1, h))
    src_rows = list(range(sep + 1, h)) if ref_side == "top" else list(range(0, sep))
    ref_row = random.choice(ref_rows)
    src_row = random.choice(src_rows)

    col_choices = list(range(c0 + 1, c1))
    count = random.randint(2, min(5, len(col_choices)))
    cols = sorted(random.sample(col_choices, count))
    match_count = random.randint(1, count - 1)
    matches = set(random.sample(cols, match_count))

    used = {BLACK, grey, ref_bg, src_bg}
    for c in cols:
        ref_color = _rand_color(used)
        if c in matches:
            src_color = ref_color
        else:
            src_color = _rand_color(used | {ref_color})
        grid[ref_row, c] = ref_color
        grid[src_row, c] = src_color

    marker_cols = list(range(0, c0)) + list(range(c1 + 1, w))
    assert marker_cols
    marker_count = random.randint(1, min(3, len(marker_cols)))
    marker_positions = random.sample(
        [(r, c) for r in ref_rows for c in marker_cols if r != ref_row],
        marker_count,
    )
    for r, c in marker_positions:
        grid[r, c] = grey

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_vertical() if random.random() < 0.5 else _gen_horizontal()
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid sample")
