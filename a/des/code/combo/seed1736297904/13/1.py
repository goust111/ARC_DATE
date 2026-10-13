import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
RECT_COUNT_OPTIONS = (1, 2, 2, 3, 3, 4)
OUTER_H_RANGE = (4, 8)
OUTER_W_RANGE = (4, 8)
MAX_MARKERS_PER_RECT = 5
GEN_RETRIES = 500
PLACEMENT_RETRIES = 120


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _gap_side(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> str:
    gr, gc = gap
    if gr == r0:
        return "top"
    if gr == r1:
        return "bottom"
    if gc == c0:
        return "left"
    return "right"


def _ray_dir(side: str) -> tuple[int, int]:
    return {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for pts in _components4(g == GREY):
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = set(_border_coords(r0, c0, r1, c1))
        grey_pts = {tuple(map(int, p)) for p in pts}
        if not grey_pts.issubset(border):
            continue
        if len(border) - len(grey_pts) != 1:
            continue

        gap = next(iter(border - grey_pts))
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if int(g[gap]) != BLACK:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = {int(v) for v in inner.flat if int(v) not in (BLACK, GREY)}
        if len(colors) != 1:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": _gap_side(r0, c0, r1, c1, gap),
                "color": next(iter(colors)),
            }
        )

    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot a ray of the interior marker color outward from each frame gap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for frame in _extract_frames(g):
        gr, gc = frame["gap"]
        dr, dc = _ray_dir(frame["side"])
        color = int(frame["color"])
        r, c = gr + dr, gc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) == GREY:
                break
            out[r, c] = color
            r += dr
            c += dc

    return out


def _random_size(rng: random.Random, limit: int, lo: int, hi: int) -> int:
    upper = min(hi, limit)
    return rng.randint(lo, upper)


def _sample_markers(
    rng: random.Random, r0: int, c0: int, r1: int, c1: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    count = rng.randint(1, min(MAX_MARKERS_PER_RECT, len(cells)))
    return rng.sample(cells, count)


def _make_frame(
    rng: random.Random,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    color: int,
    forced_gap: tuple[int, int] | None = None,
) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if forced_gap is None:
        if side == "top":
            gap = (r0, rng.randint(c0 + 1, c1 - 1))
        elif side == "bottom":
            gap = (r1, rng.randint(c0 + 1, c1 - 1))
        elif side == "left":
            gap = (rng.randint(r0 + 1, r1 - 1), c0)
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c1)
    else:
        gap = forced_gap
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "color": color,
        "markers": _sample_markers(rng, r0, c0, r1, c1),
    }


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1


def _can_add_frame(frame: dict, frames: list[dict], h: int, w: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    for other in frames:
        if not _boxes_separated(frame["bbox"], other["bbox"]):
            return False
    return True


def _frame_border_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    border = set(_border_coords(r0, c0, r1, c1))
    border.remove(frame["gap"])
    return border


def _frame_bbox_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _trace_ray(
    frame: dict,
    h: int,
    w: int,
    border_cells: set[tuple[int, int]],
    bbox_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], bool] | None:
    gr, gc = frame["gap"]
    dr, dc = _ray_dir(frame["side"])
    r, c = gr + dr, gc + dc
    path: list[tuple[int, int]] = []
    blocked = False
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in border_cells:
            blocked = True
            break
        if (r, c) in bbox_cells:
            return None
        path.append((r, c))
        r += dr
        c += dc
    if not path:
        return None
    return path, blocked


def _build_grid(h: int, w: int, frames: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for frame in frames:
        for r, c in _frame_border_cells(frame):
            grid[r, c] = GREY
        for r, c in frame["markers"]:
            grid[r, c] = int(frame["color"])
    return grid


def _frame_signature(frame: dict) -> tuple[int, int, int, int, int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    return (r0, c0, r1, c1, gr, gc, int(frame["color"]))


def _validate_layout(frames: list[dict], h: int, w: int, require_blocked: bool) -> bool:
    if not frames:
        return False
    if len({int(frame["color"]) for frame in frames}) != len(frames):
        return False
    for i in range(len(frames)):
        if not _can_add_frame(frames[i], frames[:i] + frames[i + 1 :], h, w):
            return False

    grid = _build_grid(h, w, frames)
    detected = _extract_frames(grid)
    if len(detected) != len(frames):
        return False
    if sorted(_frame_signature(f) for f in detected) != sorted(_frame_signature(f) for f in frames):
        return False

    border_cells: set[tuple[int, int]] = set()
    bbox_cells: set[tuple[int, int]] = set()
    for frame in frames:
        border_cells |= _frame_border_cells(frame)
        bbox_cells |= _frame_bbox_cells(frame)

    out_expected = grid.copy()
    used_path_cells: set[tuple[int, int]] = set()
    blocked_count = 0
    for frame in frames:
        traced = _trace_ray(frame, h, w, border_cells, bbox_cells)
        if traced is None:
            return False
        path, blocked = traced
        if used_path_cells.intersection(path):
            return False
        used_path_cells.update(path)
        if blocked:
            blocked_count += 1
        for r, c in path:
            out_expected[r, c] = int(frame["color"])

    if require_blocked:
        if blocked_count < 1:
            return False
    elif blocked_count != 0:
        return False

    out_actual = main(grid)
    if np.array_equal(out_actual, grid):
        return False
    return np.array_equal(out_actual, out_expected)


def _place_random_open_frame(
    rng: random.Random, h: int, w: int, frames: list[dict], color: int
) -> dict | None:
    max_h = min(OUTER_H_RANGE[1], h - 2)
    max_w = min(OUTER_W_RANGE[1], w - 2)
    if max_h < OUTER_H_RANGE[0] or max_w < OUTER_W_RANGE[0]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        fh = _random_size(rng, max_h, OUTER_H_RANGE[0], OUTER_H_RANGE[1])
        fw = _random_size(rng, max_w, OUTER_W_RANGE[0], OUTER_W_RANGE[1])
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        frame = _make_frame(rng, r0, c0, fh, fw, rng.choice(SIDES), color)
        if _can_add_frame(frame, frames, h, w):
            return frame
    return None


def _place_blocked_pair(
    rng: random.Random, h: int, w: int, color_a: int, color_b: int
) -> tuple[dict, dict] | None:
    avoid_side = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    max_h = min(OUTER_H_RANGE[1], h - 2)
    max_w = min(OUTER_W_RANGE[1], w - 2)
    if max_h < OUTER_H_RANGE[0] or max_w < OUTER_W_RANGE[0]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        side = rng.choice(SIDES)
        sh = _random_size(rng, max_h, OUTER_H_RANGE[0], OUTER_H_RANGE[1])
        sw = _random_size(rng, max_w, OUTER_W_RANGE[0], OUTER_W_RANGE[1])
        bh = _random_size(rng, max_h, OUTER_H_RANGE[0], OUTER_H_RANGE[1])
        bw = _random_size(rng, max_w, OUTER_W_RANGE[0], OUTER_W_RANGE[1])

        if side == "top":
            if h < sh + bh + 4:
                continue
            c0s = rng.randint(1, w - sw - 1)
            r0s = rng.randint(bh + 2, h - sh - 1)
            gap_c = rng.randint(c0s + 1, c0s + sw - 2)
            c0b_lo = max(1, gap_c - bw + 1)
            c0b_hi = min(w - bw - 1, gap_c)
            if c0b_lo > c0b_hi:
                continue
            c0b = rng.randint(c0b_lo, c0b_hi)
            r0b_hi = r0s - bh - 1
            if r0b_hi < 1:
                continue
            r0b = rng.randint(1, r0b_hi)
            source = _make_frame(rng, r0s, c0s, sh, sw, side, color_a, forced_gap=(r0s, gap_c))
            blocker = _make_frame(
                rng, r0b, c0b, bh, bw, rng.choice([s for s in SIDES if s != avoid_side[side]]), color_b
            )
        elif side == "bottom":
            if h < sh + bh + 4:
                continue
            c0s = rng.randint(1, w - sw - 1)
            r0s_hi = h - sh - bh - 1
            if r0s_hi < 1:
                continue
            r0s = rng.randint(1, r0s_hi)
            gap_c = rng.randint(c0s + 1, c0s + sw - 2)
            c0b_lo = max(1, gap_c - bw + 1)
            c0b_hi = min(w - bw - 1, gap_c)
            if c0b_lo > c0b_hi:
                continue
            c0b = rng.randint(c0b_lo, c0b_hi)
            r0b_lo = r0s + sh + 1
            r0b_hi = h - bh - 1
            if r0b_lo > r0b_hi:
                continue
            r0b = rng.randint(r0b_lo, r0b_hi)
            source = _make_frame(
                rng, r0s, c0s, sh, sw, side, color_a, forced_gap=(r0s + sh - 1, gap_c)
            )
            blocker = _make_frame(
                rng, r0b, c0b, bh, bw, rng.choice([s for s in SIDES if s != avoid_side[side]]), color_b
            )
        elif side == "left":
            if w < sw + bw + 4:
                continue
            r0s = rng.randint(1, h - sh - 1)
            c0s = rng.randint(bw + 2, w - sw - 1)
            gap_r = rng.randint(r0s + 1, r0s + sh - 2)
            r0b_lo = max(1, gap_r - bh + 1)
            r0b_hi = min(h - bh - 1, gap_r)
            if r0b_lo > r0b_hi:
                continue
            r0b = rng.randint(r0b_lo, r0b_hi)
            c0b_hi = c0s - bw - 1
            if c0b_hi < 1:
                continue
            c0b = rng.randint(1, c0b_hi)
            source = _make_frame(rng, r0s, c0s, sh, sw, side, color_a, forced_gap=(gap_r, c0s))
            blocker = _make_frame(
                rng, r0b, c0b, bh, bw, rng.choice([s for s in SIDES if s != avoid_side[side]]), color_b
            )
        else:
            if w < sw + bw + 4:
                continue
            r0s = rng.randint(1, h - sh - 1)
            c0s_hi = w - sw - bw - 1
            if c0s_hi < 1:
                continue
            c0s = rng.randint(1, c0s_hi)
            gap_r = rng.randint(r0s + 1, r0s + sh - 2)
            r0b_lo = max(1, gap_r - bh + 1)
            r0b_hi = min(h - bh - 1, gap_r)
            if r0b_lo > r0b_hi:
                continue
            r0b = rng.randint(r0b_lo, r0b_hi)
            c0b_lo = c0s + sw + 1
            c0b_hi = w - bw - 1
            if c0b_lo > c0b_hi:
                continue
            c0b = rng.randint(c0b_lo, c0b_hi)
            source = _make_frame(
                rng, r0s, c0s, sh, sw, side, color_a, forced_gap=(gap_r, c0s + sw - 1)
            )
            blocker = _make_frame(
                rng, r0b, c0b, bh, bw, rng.choice([s for s in SIDES if s != avoid_side[side]]), color_b
            )

        if not _boxes_separated(source["bbox"], blocker["bbox"]):
            continue
        return source, blocker

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_rects = rng.choice(RECT_COUNT_OPTIONS)
    require_blocked = rng.random() < 0.5
    if require_blocked:
        n_rects = max(2, n_rects)

    if n_rects > len(MARKER_COLORS):
        return None
    colors = list(rng.sample(MARKER_COLORS, n_rects))
    frames: list[dict] = []

    if require_blocked:
        pair = _place_blocked_pair(rng, h, w, colors[0], colors[1])
        if pair is None:
            return None
        frames.extend(pair)
        start_idx = 2
    else:
        start_idx = 0

    for color in colors[start_idx:]:
        frame = _place_random_open_frame(rng, h, w, frames, color)
        if frame is None:
            return None
        frames.append(frame)

    if not _validate_layout(frames, h, w, require_blocked=require_blocked):
        return None
    return _build_grid(h, w, frames)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
