import random
from collections import deque

import numpy as np


BG = 0
DIVIDER_COLOR = 5
MARKER_COLOR = 6
FRAME_COLORS = (1, 2, 3, 4, 7, 8, 9)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        column = grid[:, c]
        if int(column[0]) == bg:
            continue
        if not np.all(column == column[0]):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append((int(c), int(column[0])))
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return candidates[0]


def _components_monochrome(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            rel = arr - np.array([r0, c0], dtype=int)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "rel": rel,
                }
            )
    return comps


def _frame_like(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    if np.any(mask[1:-1, 1:-1]):
        return False
    return True


def _extract_frames(left_grid: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    frames: dict[tuple[int, int], dict] = {}
    for comp in _components_monochrome(left_grid, bg):
        mask = comp["mask"]
        assert _frame_like(mask), "left-side objects must be hollow rectangular frames"
        key = tuple(map(int, mask.shape))
        assert key not in frames, "frame sizes must be unique"
        frames[key] = comp
    assert frames, "expected at least one frame on the left"
    return frames


def _has_color(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color


def _detect_corner_sets(grid: np.ndarray, color: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    tls: set[tuple[int, int]] = set()
    trs: set[tuple[int, int]] = set()
    bls: set[tuple[int, int]] = set()
    brs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color:
                continue
            if (
                _has_color(grid, r, c + 1, color)
                and _has_color(grid, r + 1, c, color)
                and not _has_color(grid, r, c - 1, color)
                and not _has_color(grid, r - 1, c, color)
            ):
                tls.add((int(r), int(c)))
            if (
                _has_color(grid, r, c - 1, color)
                and _has_color(grid, r + 1, c, color)
                and not _has_color(grid, r, c + 1, color)
                and not _has_color(grid, r - 1, c, color)
            ):
                trs.add((int(r), int(c)))
            if (
                _has_color(grid, r, c + 1, color)
                and _has_color(grid, r - 1, c, color)
                and not _has_color(grid, r, c - 1, color)
                and not _has_color(grid, r + 1, c, color)
            ):
                bls.add((int(r), int(c)))
            if (
                _has_color(grid, r, c - 1, color)
                and _has_color(grid, r - 1, c, color)
                and not _has_color(grid, r, c + 1, color)
                and not _has_color(grid, r + 1, c, color)
            ):
                brs.add((int(r), int(c)))
    return tls, trs, bls, brs


def _rectangles_from_markers(right_grid: np.ndarray, marker: int) -> list[tuple[int, int, int, int]]:
    tls, trs, bls, brs = _detect_corner_sets(right_grid, marker)
    assert tls and len(tls) == len(trs) == len(bls) == len(brs), "corner counts must match"

    tr_by_row = {r: c for r, c in trs}
    bl_by_col = {c: r for r, c in bls}
    br_set = set(brs)

    rects: list[tuple[int, int, int, int]] = []
    for r0, c0 in sorted(tls):
        assert r0 in tr_by_row, "top-left must have a matching top-right"
        assert c0 in bl_by_col, "top-left must have a matching bottom-left"
        c1 = int(tr_by_row[r0])
        r1 = int(bl_by_col[c0])
        assert c1 > c0 and r1 > r0, "rectangle corners must be properly ordered"
        assert (r1, c1) in br_set, "matching bottom-right corner missing"
        rects.append((int(r0), int(c0), int(r1), int(c1)))

    seen_sizes = set()
    for r0, c0, r1, c1 in rects:
        size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        assert size not in seen_sizes, "marker rectangles must have unique sizes"
        seen_sizes.add(size)
    return rects


def _paint_component(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(color)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    divider_col, _divider_color = _find_divider(g, bg)

    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    marker_colors = sorted(set(map(int, right.flatten())) - {int(bg)})
    assert len(marker_colors) == 1, "right side must contain only one non-background marker color"
    marker = int(marker_colors[0])

    rects_local = _rectangles_from_markers(right, marker)
    frames = _extract_frames(left, bg)
    assert len(frames) == len(rects_local), "number of frames must equal number of marker rectangles"

    out = np.full_like(g, bg)
    for r0, c0, r1, c1 in rects_local:
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        key = (h, w)
        assert key in frames, "each inferred rectangle must match a left-side frame by size"
        frame = frames[key]
        _paint_component(out, frame["mask"], int(r0), int(divider_col + 1 + c0), int(frame["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered hollow frames from the left area onto the nested marker-defined rectangles on the right.
    """
    return _solve(grid)


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _stamp_corner_markers(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    pts = [
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0, c1),
        (r0, c1 - 1),
        (r0 + 1, c1),
        (r1, c0),
        (r1, c0 + 1),
        (r1 - 1, c0),
        (r1, c1),
        (r1, c1 - 1),
        (r1 - 1, c1),
    ]
    for r, c in pts:
        grid[int(r), int(c)] = int(color)


def _mark_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_frames_left(
    h: int,
    left_w: int,
    frame_specs: list[dict],
    rng: random.Random,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    forbidden = np.zeros((h, left_w), dtype=bool)
    placements: dict[tuple[int, int], tuple[int, int]] = {}

    order = sorted(frame_specs, key=lambda item: (item["mask"].shape[0] * item["mask"].shape[1]), reverse=True)
    for spec in order:
        mask = spec["mask"]
        fh, fw = mask.shape
        candidates = [(r, c) for r in range(h - fh + 1) for c in range(left_w - fw + 1)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if np.any(forbidden[r0 : r0 + fh, c0 : c0 + fw] & mask):
                continue
            cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(mask)]
            placements[tuple(map(int, mask.shape))] = (int(r0), int(c0))
            _mark_forbidden(forbidden, cells)
            placed = True
            break
        if not placed:
            return None
    return placements


def _rects_from_sizes(
    outer_top: int,
    outer_left: int,
    sizes: list[tuple[int, int]],
    gap_rs: list[int],
    gap_cs: list[int],
) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    n = len(sizes)
    for i, (hh, ww) in enumerate(sizes):
        off_r = sum(gap_rs[i:]) if i < n - 1 else 0
        off_c = sum(gap_cs[i:]) if i < n - 1 else 0
        r0 = int(outer_top + off_r)
        c0 = int(outer_left + off_c)
        rects.append((r0, c0, int(r0 + hh - 1), int(c0 + ww - 1)))
    return rects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = 2 if rng.random() < 0.5 else 3

    inner_h = rng.randint(4, 8)
    inner_w = rng.randint(4, 8)
    gap_rs = [rng.randint(2, 3) for _ in range(layer_count - 1)]
    gap_cs = [rng.randint(2, 3) for _ in range(layer_count - 1)]

    sizes = [(int(inner_h), int(inner_w))]
    cur_h, cur_w = inner_h, inner_w
    for gr, gc in zip(gap_rs, gap_cs):
        cur_h += 2 * gr
        cur_w += 2 * gc
        sizes.append((int(cur_h), int(cur_w)))

    outer_h, outer_w = sizes[-1]
    if outer_w > 12 or outer_h > 18:
        return None

    min_w = 2 * outer_w + 4
    max_w = min(30, min_w + 3)
    if min_w > 30:
        return None
    grid_w = rng.randint(min_w, max_w)

    left_w_min = outer_w + 1
    left_w_max = grid_w - 1 - (outer_w + 2)
    if left_w_min > left_w_max:
        return None
    left_w = rng.randint(left_w_min, left_w_max)
    right_w = grid_w - left_w - 1

    min_h = outer_h + 2
    max_h = min(30, outer_h + 6)
    grid_h = rng.randint(min_h, max_h)

    right_top = rng.randint(1, grid_h - outer_h - 1)
    right_left_margin = rng.randint(1, right_w - outer_w - 1)
    right_left = left_w + 1 + right_left_margin

    rects = _rects_from_sizes(right_top, right_left, sizes, gap_rs, gap_cs)
    colors = rng.sample(FRAME_COLORS, layer_count)
    frame_specs = [{"size": size, "mask": _frame_mask(*size), "color": color} for size, color in zip(sizes, colors)]

    placements = _place_frames_left(grid_h, left_w, frame_specs, rng)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, left_w] = DIVIDER_COLOR
    for rect in rects:
        _stamp_corner_markers(grid, rect, MARKER_COLOR)

    for spec in frame_specs:
        h0, w0 = spec["size"]
        r0, c0 = placements[(h0, w0)]
        _paint_component(grid, spec["mask"], int(r0), int(c0), int(spec["color"]))

    expected = np.full_like(grid, BG)
    for spec, rect in zip(frame_specs, rects):
        r0, c0, _r1, _c1 = rect
        _paint_component(expected, spec["mask"], int(r0), int(c0), int(spec["color"]))

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
