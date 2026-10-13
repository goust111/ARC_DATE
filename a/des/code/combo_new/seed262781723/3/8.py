from __future__ import annotations

import random

import numpy as np


BG = 0
FULL_CANVAS = 30
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400
PATTERN_KINDS = ("horizontal", "vertical", "checker")


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    return [int(color) for color in np.unique(grid) if int(color) not in blocked]


def _runs_1d(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(values.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _candidate_edge_rows(mask: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(mask.shape[0]):
        if any(end - start + 1 >= 3 for start, end in _runs_1d(mask[r])):
            rows.append(r)
    return rows


def _candidate_edge_cols(mask: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(mask.shape[1]):
        if any(end - start + 1 >= 3 for start, end in _runs_1d(mask[:, c])):
            cols.append(c)
    return cols


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains_rect(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1


def _overlap_box(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 >= r1 or c0 >= c1:
        return None
    return (r0, c0, r1, c1)


def _inner_box(
    rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = rect
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


def _enumerate_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rows = _candidate_edge_rows(mask)
    cols = _candidate_edge_cols(mask)
    rects: set[tuple[int, int, int, int]] = set()
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 2:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    if not np.all(mask[r0, c0 : c1 + 1]):
                        continue
                    if not np.all(mask[r1, c0 : c1 + 1]):
                        continue
                    if not np.all(mask[r0 : r1 + 1, c0]):
                        continue
                    if not np.all(mask[r0 : r1 + 1, c1]):
                        continue
                    rects.add((r0, c0, r1, c1))
    return sorted(
        rects,
        key=lambda rect: (_rect_area(rect), rect[2] - rect[0], rect[3] - rect[1]),
        reverse=True,
    )


def _pick_frame_pair(
    rects: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    best_pair = None
    best_score = -1
    for i, first in enumerate(rects):
        for second in rects[i + 1 :]:
            overlap = _overlap_box(first, second)
            if overlap is None:
                continue
            if _contains_rect(first, second) or _contains_rect(second, first):
                continue
            inner = _inner_box(overlap)
            if inner is None:
                continue
            score = _rect_area(first) + _rect_area(second)
            if score > best_score:
                best_pair = (first, second)
                best_score = score
    return best_pair


def _seed_info(
    grid: np.ndarray,
    inner: tuple[int, int, int, int],
    frame_color: int,
) -> tuple[int, str] | None:
    r0, c0, r1, c1 = inner
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (region != BG) & (region != frame_color)
    if not np.any(mask):
        return None

    colors = _colors_present(region[mask], exclude=set())
    if len(colors) != 1:
        return None
    seed_color = int(colors[0])

    coords = np.argwhere(region == seed_color)
    if len(coords) == 1:
        return seed_color, "checker"

    rs = coords[:, 0]
    cs = coords[:, 1]
    if int(rs.min()) == int(rs.max()):
        if len(coords) == int(cs.max()) - int(cs.min()) + 1:
            return seed_color, "horizontal"
    if int(cs.min()) == int(cs.max()):
        if len(coords) == int(rs.max()) - int(rs.min()) + 1:
            return seed_color, "vertical"
    return None


def _apply_pattern(
    out: np.ndarray,
    inner: tuple[int, int, int, int],
    seed_color: int,
    kind: str,
) -> None:
    r0, c0, r1, c1 = inner
    if kind == "horizontal":
        for r in range(r0, r1 + 1):
            if (r - r0) % 2 == 0:
                out[r, c0 : c1 + 1] = seed_color
    elif kind == "vertical":
        for c in range(c0, c1 + 1):
            if (c - c0) % 2 == 0:
                out[r0 : r1 + 1, c] = seed_color
    else:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if ((r - r0) + (c - c0)) % 2 == 0:
                    out[r, c] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interior of each same-color frame overlap using the seed color and seed-shaped pattern code.

    Input structure:
    - The grid has a Black background.
    - Several colors each form exactly two one-pixel-thick hollow rectangular frames.
    - The two frames of the same color partially overlap, and a small third-color seed lies inside the overlap interior.

    Rule:
    - For each frame color, recover the two rectangular borders and compute their overlap rectangle.
    - Inspect the non-frame seed inside the overlap interior: a horizontal segment means horizontal stripes, a vertical segment means vertical stripes, and a single pixel means a checkerboard.
    - Repaint only the strict interior of the overlap rectangle with the seed color using that pattern, leaving all frame-border pixels unchanged.

    Key insight:
    - The same-color border union encodes two overlapping rectangles, while the seed inside their shared interior simultaneously specifies the fill color and the fill mode.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame_color in _colors_present(g, exclude={BG}):
        rects = _enumerate_rectangles(g == frame_color)
        if len(rects) < 2:
            continue
        pair = _pick_frame_pair(rects)
        if pair is None:
            continue
        overlap = _overlap_box(pair[0], pair[1])
        if overlap is None:
            continue
        inner = _inner_box(overlap)
        if inner is None:
            continue
        seed = _seed_info(g, inner, frame_color)
        if seed is None:
            continue
        seed_color, kind = seed
        _apply_pattern(out, inner, seed_color, kind)

    return out


def _draw_frame(
    patch: np.ndarray, rect: tuple[int, int, int, int], color: int
) -> None:
    r0, c0, r1, c1 = rect
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color


def _build_pair_patch(
    frame_color: int,
    seed_color: int,
    kind: str,
    size_hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    h1 = random.randint(6, size_hi)
    w1 = random.randint(6, size_hi)
    h2 = random.randint(6, size_hi)
    w2 = random.randint(6, size_hi)

    max_inner_h = min(h1 - 3, h2 - 3, 6)
    max_inner_w = min(w1 - 3, w2 - 3, 6)
    inner_h = random.randint(2, max_inner_h)
    inner_w = random.randint(2, max_inner_w)

    overlap_h = inner_h + 2
    overlap_w = inner_w + 2
    dr = h1 - overlap_h
    dc = w1 - overlap_w
    canvas_h = h1 + h2 - overlap_h
    canvas_w = w1 + w2 - overlap_w

    first = (0, 0, h1 - 1, w1 - 1)
    second = (dr, dc, dr + h2 - 1, dc + w2 - 1)
    inner = (dr + 1, dc + 1, h1 - 2, w1 - 2)

    inp = np.zeros((canvas_h, canvas_w), dtype=int)
    out = np.zeros((canvas_h, canvas_w), dtype=int)
    _draw_frame(inp, first, frame_color)
    _draw_frame(inp, second, frame_color)
    _draw_frame(out, first, frame_color)
    _draw_frame(out, second, frame_color)
    _apply_pattern(out, inner, seed_color, kind)

    ir0, ic0, ir1, ic1 = inner
    if kind == "horizontal":
        row_offset = random.choice([offset for offset in range(inner_h) if offset % 2 == 0])
        length = random.randint(2, inner_w)
        start = random.randint(0, inner_w - length)
        rr = ir0 + row_offset
        inp[rr, ic0 + start : ic0 + start + length] = seed_color
    elif kind == "vertical":
        col_offset = random.choice([offset for offset in range(inner_w) if offset % 2 == 0])
        length = random.randint(2, inner_h)
        start = random.randint(0, inner_h - length)
        cc = ic0 + col_offset
        inp[ir0 + start : ir0 + start + length, cc] = seed_color
    else:
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == 0]
        rr, cc = random.choice(cells)
        inp[ir0 + rr, ic0 + cc] = seed_color

    if random.random() < 0.5:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    return inp, out


def _placement_positions(
    occupied: np.ndarray, patch_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(FULL_CANVAS - ph + 1):
        for c0 in range(FULL_CANVAS - pw + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(FULL_CANVAS, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(FULL_CANVAS, c0 + pw + PLACEMENT_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _crop_to_variant(
    inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != BG) | (out != BG))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, int(r0))
    left = random.randint(0, int(c0))
    bottom = random.randint(int(r1), inp.shape[0] - 1)
    right = random.randint(int(c1), inp.shape[1] - 1)
    return inp[top : bottom + 1, left : right + 1], out[top : bottom + 1, left : right + 1]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    size_hi = {1: 12, 2: 10, 3: 8}[pair_count]
    palette = random.sample(range(1, 10), 2 * pair_count)
    frame_colors = palette[:pair_count]
    seed_colors = palette[pair_count:]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(pair_count):
        kind = random.choice(PATTERN_KINDS)
        patches.append(_build_pair_patch(frame_colors[idx], seed_colors[idx], kind, size_hi))

    order = sorted(range(pair_count), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
    occupied = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        positions = _placement_positions(occupied, patches[idx][0].shape)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placements[idx] = (r0, c0)
        ph, pw = patches[idx][0].shape
        occupied[
            max(0, r0 - PLACEMENT_PADDING) : min(FULL_CANVAS, r0 + ph + PLACEMENT_PADDING),
            max(0, c0 - PLACEMENT_PADDING) : min(FULL_CANVAS, c0 + pw + PLACEMENT_PADDING),
        ] = True

    inp = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=int)
    out = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=int)
    for idx, (inp_patch, out_patch) in enumerate(patches):
        r0, c0 = placements[idx]
        ph, pw = inp_patch.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = inp_patch
        out[r0 : r0 + ph, c0 : c0 + pw] = out_patch

    if np.array_equal(inp, out):
        return None

    cropped_inp, cropped_out = _crop_to_variant(inp, out)
    try:
        solved = main(cropped_inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
