import random
from typing import Callable

import numpy as np


GEN_ATTEMPTS = 500
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(int(v) for v in np.unique(np.asarray(grid, dtype=int)))
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contiguous(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return False
    return int(indices[-1] - indices[0] + 1) == int(indices.size)


def _is_full_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _detect_horizontal_slider(crop: np.ndarray):
    h, _w = crop.shape
    col_counts = crop.sum(axis=0)
    body_cols = np.flatnonzero(col_counts == h)
    if body_cols.size < 2 or not _contiguous(body_cols):
        return None

    bc0 = int(body_cols[0])
    bc1 = int(body_cols[-1])
    if not np.all(crop[:, bc0 : bc1 + 1]):
        return None

    tail_cols = np.flatnonzero((col_counts > 0) & (col_counts < h))
    if tail_cols.size < 2 or not _contiguous(tail_cols):
        return None

    if int(tail_cols[-1]) == bc0 - 1:
        motion = (0, 1)
    elif int(tail_cols[0]) == bc1 + 1:
        motion = (0, -1)
    else:
        return None

    rows = [np.flatnonzero(crop[:, int(c)]) for c in tail_cols]
    if any(rr.size != 1 for rr in rows):
        return None
    tail_r = int(rows[0][0])
    if tail_r <= 0 or tail_r >= h - 1:
        return None
    if any(int(rr[0]) != tail_r for rr in rows):
        return None

    body_mask = np.zeros_like(crop, dtype=bool)
    body_mask[:, bc0 : bc1 + 1] = True
    tail_mask = np.zeros_like(crop, dtype=bool)
    tail_mask[tail_r, int(tail_cols[0]) : int(tail_cols[-1]) + 1] = True
    if not np.array_equal(crop, body_mask | tail_mask):
        return None

    return (0, bc0, h - 1, bc1), motion, int(tail_cols.size)


def _detect_vertical_slider(crop: np.ndarray):
    _h, w = crop.shape
    row_counts = crop.sum(axis=1)
    body_rows = np.flatnonzero(row_counts == w)
    if body_rows.size < 2 or not _contiguous(body_rows):
        return None

    br0 = int(body_rows[0])
    br1 = int(body_rows[-1])
    if not np.all(crop[br0 : br1 + 1, :]):
        return None

    tail_rows = np.flatnonzero((row_counts > 0) & (row_counts < w))
    if tail_rows.size < 2 or not _contiguous(tail_rows):
        return None

    if int(tail_rows[-1]) == br0 - 1:
        motion = (1, 0)
    elif int(tail_rows[0]) == br1 + 1:
        motion = (-1, 0)
    else:
        return None

    cols = [np.flatnonzero(crop[int(r), :]) for r in tail_rows]
    if any(cc.size != 1 for cc in cols):
        return None
    tail_c = int(cols[0][0])
    if tail_c <= 0 or tail_c >= w - 1:
        return None
    if any(int(cc[0]) != tail_c for cc in cols):
        return None

    body_mask = np.zeros_like(crop, dtype=bool)
    body_mask[br0 : br1 + 1, :] = True
    tail_mask = np.zeros_like(crop, dtype=bool)
    tail_mask[int(tail_rows[0]) : int(tail_rows[-1]) + 1, tail_c] = True
    if not np.array_equal(crop, body_mask | tail_mask):
        return None

    return (br0, 0, br1, w - 1), motion, int(tail_rows.size)


def _detect_slider(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]

    candidates = []
    for detector in (_detect_horizontal_slider, _detect_vertical_slider):
        found = detector(crop)
        if found is not None:
            body_local, motion, length = found
            br0, bc0, br1, bc1 = body_local
            candidates.append(((r0 + br0, c0 + bc0, r0 + br1, c0 + bc1), motion, length))

    if len(candidates) != 1:
        return None
    return candidates[0]


def _gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return int(b2[1] - b1[3] - 1)
    if (dr, dc) == (0, -1):
        return int(b1[1] - b2[3] - 1)
    if (dr, dc) == (1, 0):
        return int(b2[0] - b1[2] - 1)
    if (dr, dc) == (-1, 0):
        return int(b1[0] - b2[2] - 1)
    raise ValueError("invalid direction")


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    rects = []
    slider_color = None
    direction = None
    budget = None

    for color in colors_present(g, exclude={bg}):
        mask = g == int(color)
        slider = _detect_slider(mask)
        if slider is None:
            assert _is_full_rectangle(mask), "non-slider object must be a solid rectangle"
            rects.append({"color": int(color), "bbox": bbox_coords(np.argwhere(mask))})
        else:
            assert slider_color is None, "expected exactly one slider"
            body_bbox, direction, budget = slider
            slider_color = int(color)
            rects.append({"color": int(color), "bbox": body_bbox})

    assert slider_color is not None and direction is not None and budget is not None
    axis = 1 if direction[0] == 0 else 0
    if axis == 1:
        assert max(rec["bbox"][0] for rec in rects) <= min(rec["bbox"][2] for rec in rects), "rectangles must share a common row"
        if direction[1] == 1:
            rects.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][3]))
        else:
            rects.sort(key=lambda rec: (-rec["bbox"][3], -rec["bbox"][1]))
    else:
        assert max(rec["bbox"][1] for rec in rects) <= min(rec["bbox"][3] for rec in rects), "rectangles must share a common column"
        if direction[0] == 1:
            rects.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][2]))
        else:
            rects.sort(key=lambda rec: (-rec["bbox"][2], -rec["bbox"][0]))

    slider_idx = next(i for i, rec in enumerate(rects) if int(rec["color"]) == slider_color)
    for left, right in zip(rects, rects[1:]):
        assert _gap(left["bbox"], right["bbox"], direction) > 0, "rectangles must be separated by positive gaps"

    return bg, rects, slider_idx, direction, int(budget)


def _simulate_relay(rects: list[dict], slider_idx: int, direction: tuple[int, int], budget: int):
    shifts = [(0, 0) for _ in rects]
    current = int(slider_idx)
    remaining = int(budget)
    relays = 0

    while remaining > 0:
        if current == len(rects) - 1:
            dr, dc = direction
            shifts[current] = (dr * remaining, dc * remaining)
            remaining = 0
            break

        gap = _gap(rects[current]["bbox"], rects[current + 1]["bbox"], direction)
        step = min(remaining, gap)
        dr, dc = direction
        shifts[current] = (dr * step, dc * step)
        remaining -= step
        if remaining == 0:
            break
        current += 1
        relays += 1

    return shifts, relays


def _render(
    shape: tuple[int, int],
    rects: list[dict],
    shifts: list[tuple[int, int]],
    bg: int,
    *,
    allow_overlap: bool = False,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rec, (dr, dc) in zip(rects, shifts):
        r0, c0, r1, c1 = rec["bbox"]
        rr0, cc0 = r0 + dr, c0 + dc
        rr1, cc1 = r1 + dr, c1 + dc
        assert 0 <= rr0 <= rr1 < shape[0]
        assert 0 <= cc0 <= cc1 < shape[1]
        if not allow_overlap:
            assert np.all(out[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg), "final rectangles must not overlap"
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(rec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay one tail-length movement budget down a straight queue of solid rectangles.

    Input structure:
    - The background is black.
    - Several differently colored solid rectangles lie on one common row or one common column, with positive gaps between neighboring rectangles.
    - Exactly one rectangle has an extra one-cell-thick straight tail of length at least two; that tail is collinear with the queue.

    Rule:
    - Erase the tail and treat its length as a shared movement budget.
    - The tailed rectangle moves in the queue direction, consuming one budget unit per cell; if it reaches the next rectangle while budget remains, it stops there and hands all remaining budget to that next rectangle.
    - Repeat the same relay step along the queue until some rectangle spends the remaining budget; draw only the rectangles at their final positions, with no tail.

    Key insight:
    - The tail length is not just the first rectangle's displacement: it is a conserved budget that is spent on gaps and can propagate through multiple collisions.
    """
    bg, rects, slider_idx, direction, budget = _parse_scene(grid)
    shifts, _relays = _simulate_relay(rects, slider_idx, direction, budget)
    return _render(np.asarray(grid).shape, rects, shifts, bg)


def _render_input(shape, rects, slider_idx, tail_len, attach_offset):
    grid = np.full(shape, BG, dtype=int)
    for i, rec in enumerate(rects):
        r0, c0, r1, c1 = rec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rec["color"])
        if i == slider_idx:
            tail_r = r0 + attach_offset
            grid[tail_r, c0 - tail_len : c0] = int(rec["color"])
    return grid


def _transform_grid(grid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return grid.copy()
    if mode == "fliplr":
        return np.fliplr(grid).copy()
    if mode == "transpose":
        return grid.T.copy()
    if mode == "transpose_flipud":
        return np.flipud(grid.T).copy()
    raise ValueError(f"unknown transform mode: {mode}")


def _sample_budget(gaps: list[int]):
    prefixes = []
    total = 0
    for gap in gaps:
        total += int(gap)
        prefixes.append(total)

    cases: list[tuple[int, Callable[[], int]]] = []

    if gaps[0] >= 3:
        cases.append((1, lambda upper=gaps[0] - 1: random.randint(2, upper)))
    if prefixes[0] >= 2:
        cases.append((1, lambda val=prefixes[0]: val))

    for idx in range(1, len(prefixes)):
        lo = max(2, prefixes[idx - 1] + 1)
        hi = prefixes[idx] - 1
        if lo <= hi:
            cases.append((4, lambda lo=lo, hi=hi: random.randint(lo, hi)))
        if prefixes[idx] >= 2:
            cases.append((3, lambda val=prefixes[idx]: val))

    extra = random.randint(1, 4)
    cases.append((4, lambda total=prefixes[-1], extra=extra: total + extra))

    weights = [weight for weight, _builder in cases]
    chosen = random.choices(cases, weights=weights, k=1)[0][1]
    return int(chosen())


def _make_base_instance():
    n_rects = random.randint(3, 5)
    height = random.randint(3, 5)
    lengths = [random.randint(2, 5) for _ in range(n_rects)]
    gaps = [random.randint(1, 4) for _ in range(n_rects - 1)]
    if len(gaps) >= 2 and len(set(gaps)) == 1:
        tweak = random.randrange(len(gaps))
        gaps[tweak] = 1 if gaps[tweak] != 1 else 4

    tail_len = _sample_budget(gaps)
    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    attach_offset = random.randint(1, height - 2)

    total_span = sum(lengths) + sum(gaps)
    max_extra = max(0, tail_len - sum(gaps))
    width = left_margin + tail_len + total_span + max_extra + right_margin
    height_total = top_margin + height + bottom_margin
    if width > 30 or height_total > 30:
        return None

    colors = random.sample(PALETTE, n_rects)
    row0 = top_margin
    c = left_margin + tail_len
    rects = []
    for idx, (color, length) in enumerate(zip(colors, lengths)):
        rects.append({"color": int(color), "bbox": (row0, c, row0 + height - 1, c + length - 1)})
        c += length
        if idx < len(gaps):
            c += gaps[idx]

    shifts, relays = _simulate_relay(rects, 0, (0, 1), tail_len)
    expected = _render((height_total, width), rects, shifts, BG)
    inp = _render_input((height_total, width), rects, 0, tail_len, attach_offset)

    slider_only = [(0, 0) for _ in rects]
    slider_only[0] = (0, tail_len)
    naive = _render((height_total, width), rects, slider_only, BG, allow_overlap=True)

    return {
        "input": inp,
        "expected": expected,
        "naive": naive,
        "relays": relays,
    }


def _gen_attempt() -> np.ndarray | None:
    base = _make_base_instance()
    if base is None:
        return None

    mode = random.choice(["identity", "fliplr", "transpose", "transpose_flipud"])
    inp = _transform_grid(base["input"], mode)
    expected = _transform_grid(base["expected"], mode)
    naive = _transform_grid(base["naive"], mode)

    try:
        parsed_bg, rects, slider_idx, direction, budget = _parse_scene(inp)
        shifts, relays = _simulate_relay(rects, slider_idx, direction, budget)
        rebuilt = _render(inp.shape, rects, shifts, parsed_bg)
    except AssertionError:
        return None

    if not np.array_equal(rebuilt, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if base["relays"] > 0 and np.array_equal(expected, naive):
        return None
    if len(np.unique(inp)) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-relay input")
