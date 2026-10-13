from __future__ import annotations

import numpy as np


TL = 0
TR = 1
BL = 2
BR = 3
QUADRANTS = (TL, TR, BL, BR)

BG = 0
MIN_FRAME_SIZE = 6
MAX_FRAME_SIZE = 11
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
FRAME_COUNT_RANGE = (2, 5)
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        dirs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _nonbg_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color, connectivity=4):
            comps.append({"color": color, "coords": coords})
    return comps


def _frame_info(comp: dict) -> dict | None:
    coords = comp["coords"]
    color = int(comp["color"])
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0 = int(rs.min())
    c0 = int(cs.min())
    r1 = int(rs.max())
    c1 = int(cs.max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[rs - r0, cs - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
    }


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[int, int]]]:
    frames: list[dict] = []
    markers: list[tuple[int, int]] = []
    for comp in _nonbg_components(grid, bg):
        frame = _frame_info(comp)
        if frame is not None:
            frames.append(frame)
            continue
        if int(comp["coords"].shape[0]) == 1:
            r, c = map(int, comp["coords"][0])
            markers.append((r, c))
    return frames, markers


def _marker_relation(marker: tuple[int, int], frame: dict) -> tuple[int, int] | None:
    mr, mc = marker
    r0, c0, r1, c1 = frame["bbox"]
    if mr < r0 and mc < c0:
        return TL, int((r0 - mr) + (c0 - mc))
    if mr < r0 and mc > c1:
        return TR, int((r0 - mr) + (mc - c1))
    if mr > r1 and mc < c0:
        return BL, int((mr - r1) + (c0 - mc))
    if mr > r1 and mc > c1:
        return BR, int((mr - r1) + (mc - c1))
    return None


def _assign_markers(frames: list[dict], markers: list[tuple[int, int]]) -> dict[int, int]:
    assignments: dict[int, tuple[int, int]] = {}
    for marker in markers:
        best: tuple[int, int, int] | None = None
        for idx, frame in enumerate(frames):
            rel = _marker_relation(marker, frame)
            if rel is None:
                continue
            quad, dist = rel
            if best is None or dist < best[2]:
                best = (idx, quad, dist)
        if best is None:
            continue
        idx, quad, dist = best
        prev = assignments.get(idx)
        if prev is None or dist < prev[1]:
            assignments[idx] = (quad, dist)
    return {idx: quad for idx, (quad, _dist) in assignments.items()}


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    if h <= 0 or w <= 0:
        return
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _inner_copy_origin(frame: dict, quad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    h2 = int(frame["height"] // 2)
    w2 = int(frame["width"] // 2)

    top = int(r0 + 1) if quad in (TL, TR) else int(r1 - h2)
    left = int(c0 + 1) if quad in (TL, BL) else int(c1 - w2)
    return top, left, h2, w2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a half-size frame copy inside each frame, using the outside marker's quadrant as the target corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.reshape(-1)).argmax())

    frames, markers = _classify_objects(g, bg)
    assignments = _assign_markers(frames, markers)

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg

    for idx, frame in enumerate(frames):
        if idx not in assignments:
            continue
        top, left, h2, w2 = _inner_copy_origin(frame, assignments[idx])
        _draw_frame(out, top, left, h2, w2, int(frame["color"]))

    return out


def _reserve_ok(used: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    return bool(np.all(~used[r0 : r1 + 1, c0 : c1 + 1]))


def _make_cells(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _quadrants_for_count(rng: np.random.Generator, n: int) -> list[int]:
    if n <= 4:
        return [int(q) for q in rng.permutation(np.array(QUADRANTS))[:n]]
    quads = [int(q) for q in rng.permutation(np.array(QUADRANTS))]
    while len(quads) < n:
        quads.append(int(rng.integers(0, 4)))
    rng.shuffle(quads)
    return quads


def _sample_frame_in_cell(
    rng: np.random.Generator,
    cell: tuple[int, int, int, int],
    quad: int,
) -> tuple[int, int, int, int, int, int] | None:
    r_lo, r_hi, c_lo, c_hi = map(int, cell)
    cell_h = int(r_hi - r_lo + 1)
    cell_w = int(c_hi - c_lo + 1)
    if cell_h < MIN_FRAME_SIZE + 1 or cell_w < MIN_FRAME_SIZE + 1:
        return None

    max_v_gap = min(3, cell_h - MIN_FRAME_SIZE)
    max_h_gap = min(3, cell_w - MIN_FRAME_SIZE)
    if max_v_gap < 1 or max_h_gap < 1:
        return None
    dr = int(rng.integers(1, max_v_gap + 1))
    dc = int(rng.integers(1, max_h_gap + 1))

    h_max = min(MAX_FRAME_SIZE, cell_h - dr)
    w_max = min(MAX_FRAME_SIZE, cell_w - dc)
    if h_max < MIN_FRAME_SIZE or w_max < MIN_FRAME_SIZE:
        return None
    fh = int(rng.integers(MIN_FRAME_SIZE, h_max + 1))
    fw = int(rng.integers(MIN_FRAME_SIZE, w_max + 1))

    top_branch = quad in (TL, TR)
    left_branch = quad in (TL, BL)

    min_r0 = int(r_lo + (dr if top_branch else 0))
    max_r0 = int(r_hi - fh + 1 - (0 if top_branch else dr))
    min_c0 = int(c_lo + (dc if left_branch else 0))
    max_c0 = int(c_hi - fw + 1 - (0 if left_branch else dc))
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None

    r0 = int(rng.integers(min_r0, max_r0 + 1))
    c0 = int(rng.integers(min_c0, max_c0 + 1))
    return r0, c0, fh, fw, dr, dc


def _marker_for_frame(r0: int, c0: int, fh: int, fw: int, dr: int, dc: int, quad: int) -> tuple[int, int]:
    r1 = int(r0 + fh - 1)
    c1 = int(c0 + fw - 1)
    if quad == TL:
        return int(r0 - dr), int(c0 - dc)
    if quad == TR:
        return int(r0 - dr), int(c1 + dc)
    if quad == BL:
        return int(r1 + dr), int(c0 - dc)
    return int(r1 + dr), int(c1 + dc)


def _build_expected(inp: np.ndarray) -> np.ndarray:
    out = main(inp)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    grid = np.full((h, w), BG, dtype=int)

    cells = _make_cells(h, w)
    valid_cells = [
        idx
        for idx, (r0, r1, c0, c1) in enumerate(cells)
        if (r1 - r0 + 1) >= MIN_FRAME_SIZE + 1 and (c1 - c0 + 1) >= MIN_FRAME_SIZE + 1
    ]
    if len(valid_cells) < FRAME_COUNT_RANGE[0]:
        return None

    max_frames = min(FRAME_COUNT_RANGE[1] - 1, len(valid_cells))
    n_frames = int(rng.integers(FRAME_COUNT_RANGE[0], max_frames + 1))
    chosen_cells = [int(x) for x in rng.choice(valid_cells, size=n_frames, replace=False)]
    quadrants = _quadrants_for_count(rng, n_frames)

    color_pool = rng.permutation(np.arange(1, 10))
    frame_colors = [int(x) for x in color_pool[:n_frames]]
    marker_colors = [int(x) for x in color_pool[n_frames : 2 * n_frames]]

    used = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    for slot, quad, frame_color, marker_color in zip(chosen_cells, quadrants, frame_colors, marker_colors):
        placed = None
        for _ in range(50):
            sample = _sample_frame_in_cell(rng, cells[slot], quad)
            if sample is None:
                continue
            r0, c0, fh, fw, dr, dc = sample
            mr, mc = _marker_for_frame(r0, c0, fh, fw, dr, dc, quad)
            r1 = int(r0 + fh - 1)
            c1 = int(c0 + fw - 1)
            rr0 = max(0, min(r0, mr) - 1)
            rr1 = min(h - 1, max(r1, mr) + 1)
            cc0 = max(0, min(c0, mc) - 1)
            cc1 = min(w - 1, max(c1, mc) + 1)
            if _reserve_ok(used, rr0, rr1, cc0, cc1):
                placed = (r0, c0, fh, fw, mr, mc, rr0, rr1, cc0, cc1)
                break
        if placed is None:
            return None

        r0, c0, fh, fw, mr, mc, rr0, rr1, cc0, cc1 = placed
        r1 = int(r0 + fh - 1)
        c1 = int(c0 + fw - 1)
        _draw_frame(grid, r0, c0, fh, fw, frame_color)
        grid[mr, mc] = marker_color
        used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        specs.append(
            {
                "color": frame_color,
                "bbox": (r0, c0, r1, c1),
                "height": fh,
                "width": fw,
                "marker": (mr, mc),
                "quad": quad,
            }
        )

    expected = np.full_like(grid, BG)
    for spec, marker_color in zip(specs, marker_colors):
        r0, c0, r1, c1 = spec["bbox"]
        _draw_frame(expected, r0, c0, spec["height"], spec["width"], spec["color"])
        top, left, h2, w2 = _inner_copy_origin(spec, spec["quad"])
        _draw_frame(expected, top, left, h2, w2, spec["color"])
        mr, mc = spec["marker"]
        expected[mr, mc] = BG
    for spec in specs:
        mr, mc = spec["marker"]
        if grid[mr, mc] == BG:
            return None

    try:
        produced = _build_expected(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
