from __future__ import annotations

import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _largest_enclosed_bg(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    h, w = grid.shape
    best_pts: list[tuple[int, int]] | None = None
    for pts in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        if best_pts is None or len(pts) > len(best_pts):
            best_pts = pts
    assert best_pts is not None, "expected a background region enclosed by the frame"
    rs = [r for r, _ in best_pts]
    cs = [c for _, c in best_pts]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in best_pts:
        mask[r, c] = True
    return bbox, mask


def _frame_info(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray, dict[str, int]]:
    (ir0, ic0, ir1, ic1), gap_mask = _largest_enclosed_bg(grid, bg)
    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert 0 <= fr0 < fr1 < grid.shape[0]
    assert 0 <= fc0 < fc1 < grid.shape[1]
    assert int(grid[fr0, fc0]) == bg and int(grid[fr0, fc1]) == bg
    assert int(grid[fr1, fc0]) == bg and int(grid[fr1, fc1]) == bg

    top = grid[fr0, ic0 : ic1 + 1]
    bottom = grid[fr1, ic0 : ic1 + 1]
    left = grid[ir0 : ir1 + 1, fc0]
    right = grid[ir0 : ir1 + 1, fc1]

    top_color = int(top[0])
    bottom_color = int(bottom[0])
    left_color = int(left[0])
    right_color = int(right[0])
    assert top_color != bg and bottom_color != bg and left_color != bg and right_color != bg
    assert np.all(top == top_color) and np.all(bottom == bottom_color)
    assert np.all(left == left_color) and np.all(right == right_color)
    assert len({top_color, bottom_color, left_color, right_color}) == 4

    return (fr0, fc0, fr1, fc1), gap_mask, {
        "up": top_color,
        "down": bottom_color,
        "left": left_color,
        "right": right_color,
    }


def _detect_notches(
    grid: np.ndarray, gap_mask: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int
) -> list[tuple[list[tuple[int, int]], str]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    found: list[tuple[list[tuple[int, int]], str]] = []
    used: set[tuple[int, int]] = set()

    for r in range(ir0 + 2, ir1 - 1):
        for c in range(ic0 + 2, ic1 - 1):
            plus = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if any((pr, pc) in used for pr, pc in plus):
                continue
            if not all(gap_mask[pr, pc] for pr, pc in plus):
                continue

            openings: list[str] = []
            closed_colors: list[int] = []
            valid = True
            for name, (dr, dc) in DIRS.items():
                rr, cc = r + 2 * dr, c + 2 * dc
                if not (ir0 <= rr <= ir1 and ic0 <= cc <= ic1):
                    valid = False
                    break
                if gap_mask[rr, cc]:
                    openings.append(name)
                else:
                    v = int(grid[rr, cc])
                    if v == bg:
                        valid = False
                        break
                    closed_colors.append(v)
            if not valid or len(openings) != 1 or len(set(closed_colors)) != 1:
                continue

            direction = openings[0]
            found.append((plus, direction))
            used.update(plus)

    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background plus-shaped notch with the color of the frame side it opens toward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame_bbox, gap_mask, side_colors = _frame_info(g, bg)
    notches = _detect_notches(g, gap_mask, frame_bbox, bg)

    out = g.copy()
    for plus, direction in notches:
        color = side_colors[direction]
        for r, c in plus:
            out[r, c] = color
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask.astype(bool))
    return len(comps) == 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rand_span(length: int, min_len: int) -> tuple[int, int]:
    span = random.randint(min_len, length)
    start = random.randint(0, length - span)
    return start, start + span


def _make_object_mask(max_h: int, max_w: int) -> np.ndarray | None:
    obj_h = random.randint(7, max_h)
    obj_w = random.randint(7, max_w)
    mask = np.zeros((obj_h, obj_w), dtype=bool)

    core_h = random.randint(6, obj_h)
    core_w = random.randint(6, obj_w)
    core_r0 = random.randint(0, obj_h - core_h)
    core_c0 = random.randint(0, obj_w - core_w)
    core_r1 = core_r0 + core_h
    core_c1 = core_c0 + core_w
    mask[core_r0:core_r1, core_c0:core_c1] = True

    top_avail = core_r0
    bottom_avail = obj_h - core_r1
    left_avail = core_c0
    right_avail = obj_w - core_c1

    if top_avail > 0 and random.random() < 0.8:
        depth = random.randint(1, top_avail)
        c0, c1 = _rand_span(core_w, 3)
        mask[core_r0 - depth : core_r0, core_c0 + c0 : core_c0 + c1] = True
    if bottom_avail > 0 and random.random() < 0.8:
        depth = random.randint(1, bottom_avail)
        c0, c1 = _rand_span(core_w, 3)
        mask[core_r1 : core_r1 + depth, core_c0 + c0 : core_c0 + c1] = True
    if left_avail > 0 and random.random() < 0.8:
        depth = random.randint(1, left_avail)
        r0, r1 = _rand_span(core_h, 3)
        mask[core_r0 + r0 : core_r0 + r1, core_c0 - depth : core_c0] = True
    if right_avail > 0 and random.random() < 0.8:
        depth = random.randint(1, right_avail)
        r0, r1 = _rand_span(core_h, 3)
        mask[core_r0 + r0 : core_r0 + r1, core_c1 : core_c1 + depth] = True

    mask = _trim_mask(mask)
    if mask.shape[0] < 6 or mask.shape[1] < 6:
        return None
    return mask


def _plus_cells(center_r: int, center_c: int) -> list[tuple[int, int]]:
    return [
        (center_r, center_c),
        (center_r - 1, center_c),
        (center_r + 1, center_c),
        (center_r, center_c - 1),
        (center_r, center_c + 1),
    ]


def _notch_candidates(
    mask: np.ndarray, direction: str, chosen_centers: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = mask.shape
    dr, dc = DIRS[direction]
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if any(max(abs(r - rr), abs(c - cc)) <= 3 for rr, cc in chosen_centers):
                continue
            plus = _plus_cells(r, c)
            if not all(0 <= pr < h and 0 <= pc < w and mask[pr, pc] for pr, pc in plus):
                continue

            rr, cc = r + 2 * dr, c + 2 * dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                continue

            ok = True
            for other, (er, ec) in DIRS.items():
                if other == direction:
                    continue
                br, bc = r + 2 * er, c + 2 * ec
                if not (0 <= br < h and 0 <= bc < w and mask[br, bc]):
                    ok = False
                    break
            if ok:
                out.append((int(r), int(c)))
    random.shuffle(out)
    return out


def _carve_plus(mask: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    out = mask.copy()
    for r, c in _plus_cells(center_r, center_c):
        out[r, c] = False
    return out


def _place_notches(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]]] | None:
    working = mask.copy()
    chosen: list[tuple[int, int, str]] = []
    centers: list[tuple[int, int]] = []

    selected_dirs = [name for name in DIRS if random.random() < 0.55]
    if not selected_dirs:
        selected_dirs = [random.choice(list(DIRS))]

    targets: list[str] = []
    for direction in selected_dirs:
        targets.append(direction)
        if random.random() < 0.4:
            targets.append(direction)
    random.shuffle(targets)
    targets = targets[:6]

    for direction in targets:
        placed = False
        for r, c in _notch_candidates(working, direction, centers):
            trial = _carve_plus(working, r, c)
            if _mask_connected(trial):
                working = trial
                centers.append((r, c))
                chosen.append((r, c, direction))
                placed = True
                break
        if not placed and not chosen:
            return None

    if not chosen:
        return None
    return working, chosen


def _draw_frame(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
    bg: int,
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0] = bg
    grid[fr0, fc1] = bg
    grid[fr1, fc0] = bg
    grid[fr1, fc1] = bg
    grid[fr0, fc0 + 1 : fc1] = side_colors["up"]
    grid[fr1, fc0 + 1 : fc1] = side_colors["down"]
    grid[fr0 + 1 : fr1, fc0] = side_colors["left"]
    grid[fr0 + 1 : fr1, fc1] = side_colors["right"]


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    obj_color = palette[0]
    side_colors = {
        "up": palette[1],
        "down": palette[2],
        "left": palette[3],
        "right": palette[4],
    }

    for _ in range(80):
        frame_h = random.randint(13, h - 2)
        frame_w = random.randint(13, w - 2)
        fr0 = random.randint(1, h - frame_h - 1)
        fc0 = random.randint(1, w - frame_w - 1)
        fr1 = fr0 + frame_h - 1
        fc1 = fc0 + frame_w - 1

        inner_h = frame_h - 2
        inner_w = frame_w - 2
        max_obj_h = inner_h - 4
        max_obj_w = inner_w - 4
        if max_obj_h < 7 or max_obj_w < 7:
            continue

        base_mask = _make_object_mask(max_obj_h, max_obj_w)
        if base_mask is None:
            continue
        placed = _place_notches(base_mask)
        if placed is None:
            continue
        object_mask, notches = placed
        obj_h, obj_w = object_mask.shape

        min_r = fr0 + 3
        max_r = fr1 - 3 - obj_h + 1
        min_c = fc0 + 3
        max_c = fc1 - 3 - obj_w + 1
        if min_r > max_r or min_c > max_c:
            continue

        obj_r0 = random.randint(min_r, max_r)
        obj_c0 = random.randint(min_c, max_c)

        inp = np.full((h, w), bg, dtype=int)
        out = np.full((h, w), bg, dtype=int)
        frame_bbox = (fr0, fc0, fr1, fc1)
        _draw_frame(inp, frame_bbox, side_colors, bg)
        _draw_frame(out, frame_bbox, side_colors, bg)

        abs_notches: list[tuple[list[tuple[int, int]], str]] = []
        for r in range(obj_h):
            for c in range(obj_w):
                if object_mask[r, c]:
                    inp[obj_r0 + r, obj_c0 + c] = obj_color
                    out[obj_r0 + r, obj_c0 + c] = obj_color

        for cr, cc, direction in notches:
            plus = [(obj_r0 + r, obj_c0 + c) for r, c in _plus_cells(cr, cc)]
            abs_notches.append((plus, direction))
            for r, c in plus:
                inp[r, c] = bg
                out[r, c] = side_colors[direction]

        if _get_bg(inp) != bg:
            continue
        if np.array_equal(inp, out):
            continue
        try:
            if not np.array_equal(main(inp), out):
                continue
        except Exception:
            continue
        return inp, out

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        sample = _generate_case()
        if sample is not None:
            inp, _ = sample
            return inp
    raise RuntimeError("failed to generate a valid sample")
