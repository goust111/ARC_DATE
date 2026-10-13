from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DOT_COLORS = (RED, BLUE, GREEN, YELLOW)
STAMP_COLORS = (MAGENTA, ORANGE, AZURE, MAROON)
DIRECTIONS = ("up", "down", "left", "right")

WORK_H = 30
WORK_W = 30
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BLACK) -> None:
    mask = patch != transparent
    if not np.any(mask):
        return
    h, w = patch.shape
    dst_slice = dst[r0 : r0 + h, c0 : c0 + w]
    dst_slice[mask] = patch[mask]


def _paint_stamp(dst: np.ndarray, stamp: np.ndarray, r0: int, c0: int) -> None:
    h, w = dst.shape
    sh, sw = stamp.shape
    for sr in range(sh):
        rr = r0 + sr
        if not (0 <= rr < h):
            continue
        for sc in range(sw):
            cc = c0 + sc
            if 0 <= cc < w and int(stamp[sr, sc]) != BLACK:
                dst[rr, cc] = int(stamp[sr, sc])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        if len(comp) != 2 * h + 2 * w - 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == GREY):
            continue
        if not np.all(patch[-1, :] == GREY):
            continue
        if not np.all(patch[:, 0] == GREY):
            continue
        if not np.all(patch[:, -1] == GREY):
            continue
        if np.any(patch[1:-1, 1:-1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _dot_direction(
    pr: int, pc: int, stamp_bbox: tuple[int, int, int, int]
) -> str | None:
    sr0, sc0, sr1, sc1 = stamp_bbox
    aligned: list[str] = []
    if pr < sr0 and sc0 <= pc <= sc1:
        aligned.append("up")
    if pr > sr1 and sc0 <= pc <= sc1:
        aligned.append("down")
    if pc < sc0 and sr0 <= pr <= sr1:
        aligned.append("left")
    if pc > sc1 and sr0 <= pr <= sr1:
        aligned.append("right")
    if len(aligned) == 1:
        return aligned[0]

    distances: list[tuple[int, str]] = []
    if pr < sr0:
        distances.append((sr0 - pr, "up"))
    if pr > sr1:
        distances.append((pr - sr1, "down"))
    if pc < sc0:
        distances.append((sc0 - pc, "left"))
    if pc > sc1:
        distances.append((pc - sc1, "right"))
    if not distances:
        return None
    best = min(distances, key=lambda item: item[0])[0]
    winners = [direction for dist, direction in distances if dist == best]
    return winners[0] if len(winners) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read side-indicator dots inside each frame and tile the enclosed stamp outward in those directions.

    Input structure:
    - Black background with 2-3 non-overlapping grey one-pixel-thick rectangular frames.
    - Each frame encloses one small multicolor stamp occupying a 2x2 to 4x4 area.
    - The same frame interior also contains 1-4 isolated single-pixel colored dots.
    - Every dot sits strictly above, below, left, or right of the stamp's bounding box.

    Rule:
    - For each frame, detect the stamp and count the total number N of interior dots.
    - Each dot encodes one direction by its position relative to the stamp bbox: up, down, left, or right.
    - Starting just outside the corresponding frame wall, tile unchanged copies of the stamp N times in that direction, with each copy touching the previous one.
    - Keep the grey frame, but clear the original stamp and all dots from the frame interior to black.

    Key insight:
    - Dot position supplies the outward direction, while the number of copies comes from the aggregate count of dots in the same frame rather than from any single dot color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for fr0, fc0, fr1, fc1 in _frame_bboxes(g):
        ir0, ic0 = fr0 + 1, fc0 + 1
        ir1, ic1 = fr1 - 1, fc1 - 1
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        comps = _components(interior != BLACK)
        if not comps:
            continue

        sizes = [len(comp) for comp in comps]
        stamp_idx = int(np.argmax(sizes))
        if sizes[stamp_idx] <= 1:
            continue

        stamp_coords = comps[stamp_idx]
        sr0, sc0, sr1, sc1 = _bbox(stamp_coords)
        stamp = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        sh, sw = stamp.shape
        stamp_grid_r0 = ir0 + sr0
        stamp_grid_c0 = ic0 + sc0

        dots: list[tuple[int, int, str]] = []
        for idx, comp in enumerate(comps):
            if idx == stamp_idx or len(comp) != 1:
                continue
            pr, pc = map(int, comp[0])
            direction = _dot_direction(pr, pc, (sr0, sc0, sr1, sc1))
            if direction is not None:
                dots.append((pr, pc, direction))

        count = len(dots)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
        if count == 0:
            continue

        for _pr, _pc, direction in dots:
            for step in range(count):
                if direction == "up":
                    rr = fr0 - (step + 1) * sh
                    cc = stamp_grid_c0
                elif direction == "down":
                    rr = fr1 + 1 + step * sh
                    cc = stamp_grid_c0
                elif direction == "left":
                    rr = stamp_grid_r0
                    cc = fc0 - (step + 1) * sw
                else:
                    rr = stamp_grid_r0
                    cc = fc1 + 1 + step * sw
                _paint_stamp(out, stamp, rr, cc)

    return out


def _make_stamp(rng: np.random.Generator, n_dots: int) -> np.ndarray:
    max_dim = {1: 4, 2: 4, 3: 3, 4: 2}[int(n_dots)]
    sh = int(rng.integers(2, max_dim + 1))
    sw = int(rng.integers(2, max_dim + 1))
    max_colors = min(len(STAMP_COLORS), sh * sw)
    n_colors = int(rng.integers(2, max_colors + 1))
    colors = [int(v) for v in rng.choice(STAMP_COLORS, size=n_colors, replace=False)]

    stamp = np.full((sh, sw), colors[0], dtype=int)
    mode = int(rng.integers(0, 5))
    if mode == 0:
        for r in range(sh):
            for c in range(sw):
                stamp[r, c] = colors[(r + c) % n_colors]
    elif mode == 1:
        for r in range(sh):
            stamp[r, :] = colors[r % n_colors]
    elif mode == 2:
        for c in range(sw):
            stamp[:, c] = colors[c % n_colors]
    elif mode == 3:
        split_r = sh // 2
        split_c = sw // 2
        for r in range(sh):
            for c in range(sw):
                idx = (r >= split_r) + 2 * (c >= split_c)
                stamp[r, c] = colors[idx % n_colors]
    else:
        flat = stamp.reshape(-1)
        flat[:] = rng.choice(colors, size=flat.size, replace=True)
        for i, color in enumerate(colors):
            flat[i] = int(color)
        rng.shuffle(flat)

    if len(np.unique(stamp)) < 2:
        stamp[0, 0] = colors[0]
        stamp[-1, -1] = colors[1]
    return stamp


def _build_frame_case(rng: np.random.Generator) -> dict[str, np.ndarray] | None:
    n_dots = int(rng.integers(1, 5))
    directions = [str(v) for v in rng.choice(np.array(DIRECTIONS), size=n_dots, replace=False)]
    stamp = _make_stamp(rng, n_dots)
    sh, sw = stamp.shape

    def side_gap(active: bool) -> int:
        if active:
            return int(rng.integers(2, 4)) if n_dots <= 2 else 2
        return int(rng.integers(1, 3))

    top_gap = side_gap("up" in directions)
    bottom_gap = side_gap("down" in directions)
    left_gap = side_gap("left" in directions)
    right_gap = side_gap("right" in directions)

    interior_h = top_gap + sh + bottom_gap
    interior_w = left_gap + sw + right_gap
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    top_extent = n_dots * sh if "up" in directions else 0
    bottom_extent = n_dots * sh if "down" in directions else 0
    left_extent = n_dots * sw if "left" in directions else 0
    right_extent = n_dots * sw if "right" in directions else 0

    patch_h = top_extent + frame_h + bottom_extent
    patch_w = left_extent + frame_w + right_extent
    if patch_h > WORK_H or patch_w > WORK_W:
        return None

    inp = np.full((patch_h, patch_w), BLACK, dtype=int)
    out = np.full((patch_h, patch_w), BLACK, dtype=int)

    fr0 = top_extent
    fc0 = left_extent
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    inp[fr0, fc0 : fc1 + 1] = GREY
    inp[fr1, fc0 : fc1 + 1] = GREY
    inp[fr0 : fr1 + 1, fc0] = GREY
    inp[fr0 : fr1 + 1, fc1] = GREY
    out[fr0, fc0 : fc1 + 1] = GREY
    out[fr1, fc0 : fc1 + 1] = GREY
    out[fr0 : fr1 + 1, fc0] = GREY
    out[fr0 : fr1 + 1, fc1] = GREY

    sr0 = fr0 + 1 + top_gap
    sc0 = fc0 + 1 + left_gap
    sr1 = sr0 + sh - 1
    sc1 = sc0 + sw - 1
    _paste(inp, stamp, sr0, sc0)

    dot_colors = [int(v) for v in rng.choice(DOT_COLORS, size=n_dots, replace=False)]
    for direction, color in zip(directions, dot_colors):
        if direction == "up":
            rr = int(rng.integers(fr0 + 1, sr0 - 1))
            cc = int(rng.integers(sc0, sc1 + 1))
        elif direction == "down":
            rr = int(rng.integers(sr1 + 2, fr1))
            cc = int(rng.integers(sc0, sc1 + 1))
        elif direction == "left":
            rr = int(rng.integers(sr0, sr1 + 1))
            cc = int(rng.integers(fc0 + 1, sc0 - 1))
        else:
            rr = int(rng.integers(sr0, sr1 + 1))
            cc = int(rng.integers(sc1 + 2, fc1))
        inp[rr, cc] = color

    for direction in directions:
        for step in range(n_dots):
            if direction == "up":
                rr = fr0 - (step + 1) * sh
                cc = sc0
            elif direction == "down":
                rr = fr1 + 1 + step * sh
                cc = sc0
            elif direction == "left":
                rr = sr0
                cc = fc0 - (step + 1) * sw
            else:
                rr = sr0
                cc = fc1 + 1 + step * sw
            _paste(out, stamp, rr, cc)

    frame_box = np.zeros_like(inp, dtype=bool)
    frame_box[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    guard = _dilate8(frame_box | (out != BLACK))

    return {
        "input": inp,
        "output": out,
        "guard": guard,
    }


def _place_case(
    rng: np.random.Generator,
    inp_canvas: np.ndarray,
    out_canvas: np.ndarray,
    blocked: np.ndarray,
    case: dict[str, np.ndarray],
) -> bool:
    patch_in = case["input"]
    patch_out = case["output"]
    guard = case["guard"]
    ph, pw = patch_in.shape
    positions = [(r, c) for r in range(WORK_H - ph + 1) for c in range(WORK_W - pw + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & guard):
            continue
        _paste(inp_canvas, patch_in, r0, c0)
        _paste(out_canvas, patch_out, r0, c0)
        blocked[r0 : r0 + ph, c0 : c0 + pw] |= guard
        return True
    return False


def _crop_pair(
    rng: np.random.Generator, inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(out != BLACK)
    r0, c0, r1, c1 = _bbox(coords)

    top_room = r0
    bottom_room = WORK_H - 1 - r1
    left_room = c0
    right_room = WORK_W - 1 - c1

    top_pad = int(rng.integers(0, top_room + 1))
    bottom_pad = int(rng.integers(0, bottom_room + 1))
    left_pad = int(rng.integers(0, left_room + 1))
    right_pad = int(rng.integers(0, right_room + 1))

    rr0 = r0 - top_pad
    rr1 = r1 + bottom_pad
    cc0 = c0 - left_pad
    cc1 = c1 + right_pad
    return (
        inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(),
        out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(),
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(2, 4))
    cases: list[dict[str, np.ndarray]] = []
    for _ in range(frame_count):
        case = _build_frame_case(rng)
        if case is None:
            return None
        cases.append(case)

    cases.sort(key=lambda case: int(np.count_nonzero(case["guard"])), reverse=True)
    inp_canvas = np.full((WORK_H, WORK_W), BLACK, dtype=int)
    out_canvas = np.full((WORK_H, WORK_W), BLACK, dtype=int)
    blocked = np.zeros((WORK_H, WORK_W), dtype=bool)

    for case in cases:
        if not _place_case(rng, inp_canvas, out_canvas, blocked, case):
            return None

    inp, out = _crop_pair(rng, inp_canvas, out_canvas)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    try:
        got = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(got, out) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
