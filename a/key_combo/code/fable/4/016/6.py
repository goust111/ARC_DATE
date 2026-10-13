from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from arc_utils import bbox_coords, colors_present, find_components, get_bg
except ModuleNotFoundError:  # pragma: no cover
    _here = Path(__file__).resolve()
    _loaded = False
    for parent in _here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            from arc_utils import bbox_coords, colors_present, find_components, get_bg

            _loaded = True
            break
    if not _loaded:
        raise


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]


def _is_full_rect(mask: np.ndarray, bbox: BBox) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _detect_slider(mask: np.ndarray) -> tuple[BBox, Direction, int] | None:
    coords = np.argwhere(mask)
    assert coords.size, "expected non-empty component for slider detection"
    r0, c0, r1, c1 = bbox_coords(coords)

    candidates: list[tuple[BBox, Direction, int]] = []

    def add_if_slider(body: BBox, direction: Direction, length: int, outside: np.ndarray) -> None:
        if length < 2:
            return
        br0, bc0, br1, bc1 = body
        if not _is_full_rect(mask, body):
            return

        dr, dc = direction
        rows = set(map(int, outside[:, 0]))
        cols = set(map(int, outside[:, 1]))
        if dr == 0:
            expected_cols = set(range(bc1 + 1, c1 + 1)) if dc > 0 else set(range(c0, bc0))
            if len(rows) == 1 and cols == expected_cols:
                candidates.append((body, direction, length))
        else:
            expected_rows = set(range(br1 + 1, r1 + 1)) if dr > 0 else set(range(r0, br0))
            if len(cols) == 1 and rows == expected_rows:
                candidates.append((body, direction, length))

    for bc1 in range(c0, c1):
        body = (r0, c0, r1, bc1)
        add_if_slider(body, (0, 1), c1 - bc1, coords[coords[:, 1] > bc1])

    for bc0 in range(c0 + 1, c1 + 1):
        body = (r0, bc0, r1, c1)
        add_if_slider(body, (0, -1), bc0 - c0, coords[coords[:, 1] < bc0])

    for br1 in range(r0, r1):
        body = (r0, c0, br1, c1)
        add_if_slider(body, (1, 0), r1 - br1, coords[coords[:, 0] > br1])

    for br0 in range(r0 + 1, r1 + 1):
        body = (br0, c0, r1, c1)
        add_if_slider(body, (-1, 0), br0 - r0, coords[coords[:, 0] < br0])

    assert len(candidates) <= 1, "slider body/tail decomposition must be unique"
    return candidates[0] if candidates else None


def _first_collision(
    mover_idx: int,
    bboxes: dict[int, BBox],
    direction: Direction,
) -> tuple[int | None, int]:
    r0, c0, r1, c1 = bboxes[mover_idx]
    dr, dc = direction
    best_dist = 10**9
    best_idx: int | None = None

    for j, (rr0, cc0, rr1, cc1) in bboxes.items():
        if j == mover_idx:
            continue
        if dr == 0 and dc > 0:
            if cc0 > c1 and not (rr1 < r0 or rr0 > r1):
                dist = int(cc0 - c1 - 1)
                if dist < best_dist:
                    best_dist, best_idx = dist, j
        elif dr == 0 and dc < 0:
            if cc1 < c0 and not (rr1 < r0 or rr0 > r1):
                dist = int(c0 - cc1 - 1)
                if dist < best_dist:
                    best_dist, best_idx = dist, j
        elif dr > 0 and dc == 0:
            if rr0 > r1 and not (cc1 < c0 or cc0 > c1):
                dist = int(rr0 - r1 - 1)
                if dist < best_dist:
                    best_dist, best_idx = dist, j
        else:
            if rr1 < r0 and not (cc1 < c0 or cc0 > c1):
                dist = int(r0 - rr1 - 1)
                if dist < best_dist:
                    best_dist, best_idx = dist, j

    if best_idx is None:
        return None, best_dist
    return best_idx, int(best_dist)


def _move_chain(
    mover_idx: int,
    budget: int,
    direction: Direction,
    bboxes: dict[int, BBox],
    H: int,
    W: int,
    in_flight: set[int] | None = None,
) -> None:
    if budget <= 0:
        return
    if in_flight is None:
        in_flight = set()
    if mover_idx in in_flight:
        return
    in_flight.add(mover_idx)

    r0, c0, r1, c1 = bboxes[mover_idx]
    dr, dc = direction
    if dr > 0:
        wall = H - 1 - r1
    elif dr < 0:
        wall = r0
    elif dc > 0:
        wall = W - 1 - c1
    else:
        wall = c0

    hit_idx, hit_dist = _first_collision(mover_idx, bboxes, direction)
    if hit_idx is None:
        move = min(budget, wall)
        if move:
            bboxes[mover_idx] = (r0 + dr * move, c0 + dc * move, r1 + dr * move, c1 + dc * move)
        in_flight.remove(mover_idx)
        return

    move = min(budget, hit_dist)
    if move > wall:
        move = wall
    if move:
        bboxes[mover_idx] = (r0 + dr * move, c0 + dc * move, r1 + dr * move, c1 + dc * move)

    remain = budget - move
    if hit_dist <= move and remain > 0:
        _move_chain(hit_idx, remain, direction, bboxes, H, W, in_flight)
    in_flight.remove(mover_idx)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer tail budget along a chain of rectangular collisions.

    Input structure:
    - One dominant background color and several non-overlapping solid colored rectangles.
    - Exactly one rectangle has a one-cell-thick straight tail of length at least two.
    - All rectangles share a single color and the tail direction is cardinal.

    Rule:
    - Remove the tail and let the slider rectangle move in the tail direction by its tail length.
    - If the moving rectangle reaches another rectangle, it stops touching it and transfers all remaining budget to that object.
    - The recipient rectangle then moves with the same direction and remaining budget, producing a collision chain.
    - The first rectangle that does not hit another object, or hits the boundary, stops and no longer transfers budget.

    Key insight:
    - The tail length is the global displacement budget shared by the whole collision chain, not a private budget for only the original slider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = int(get_bg(g))

    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground color"

    objects: list[tuple[int, BBox]] = []
    slider = None

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected at least one foreground component"

    for comp in components:
        color = int(comp["color"])
        mask = g == color
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        area = int(coords.shape[0])
        body_area = (r1 - r0 + 1) * (c1 - c0 + 1)

        if area == body_area:
            objects.append((color, (r0, c0, r1, c1)))
            continue

        slider_info = _detect_slider(mask)
        assert slider_info is not None, "non-rectangular component must be a slider"
        assert slider is None, "expected exactly one slider object"
        body, direction, tail_len = slider_info
        slider = (color, body, direction, tail_len)

    assert slider is not None, "expected exactly one slider object"

    color, slider_body, direction, tail_len = slider
    assert tail_len >= 2

    bboxes = {i: bbox for i, (_, bbox) in enumerate(objects)}
    slider_idx: int | None = None
    for idx, (col, _) in enumerate(objects):
        if col == color:
            slider_idx = idx
            break
    if slider_idx is None:
        slider_idx = len(objects)
        objects.append((color, slider_body))
        bboxes[slider_idx] = slider_body
    else:
        bboxes[slider_idx] = slider_body

    _move_chain(slider_idx, tail_len, direction, bboxes, H, W)

    out = np.full_like(g, bg)
    for i, (col, _) in enumerate(objects):
        rr0, cc0, rr1, cc1 = bboxes[i]
        assert 0 <= rr0 <= rr1 < H and 0 <= cc0 <= cc1 < W
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = col

    assert not np.array_equal(out, g), "expected a visible transformation"
    return out


def _rects_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _draw_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _blocked_axis_intervals(
    objects: list[BBox], direction: Direction, H: int, W: int
) -> tuple[np.ndarray, np.ndarray]:
    blocked_r = np.zeros(H, dtype=bool)
    blocked_c = np.zeros(W, dtype=bool)
    for r0, c0, r1, c1 in objects:
        blocked_r[r0 : r1 + 1] = True
        blocked_c[c0 : c1 + 1] = True
    if direction[0] == 0:
        return blocked_r, blocked_c
    return blocked_c, blocked_r


def _paint_slider_body_and_tail(
    grid: np.ndarray,
    body: BBox,
    color: int,
    direction: Direction,
    tail_len: int,
    tail_anchor: int,
) -> None:
    r0, c0, r1, c1 = map(int, body)
    bg = int(grid[0, 0])
    _draw_rect(grid, body, color)
    if direction == (0, 1):
        _draw_rect(grid, (tail_anchor, c1 + 1, tail_anchor, c1 + tail_len), color)
    elif direction == (0, -1):
        _draw_rect(grid, (tail_anchor, c0 - tail_len, tail_anchor, c0 - 1), color)
    elif direction == (1, 0):
        _draw_rect(grid, (r1 + 1, tail_anchor, r1 + tail_len, tail_anchor), color)
    else:
        _draw_rect(grid, (r0 - tail_len, tail_anchor, r0 - 1, tail_anchor), color)


def _build_chain_boxes_linear(
    first_box: BBox,
    direction: Direction,
    chain_count: int,
    initial_shift: int,
    H: int,
    W: int,
    forbid_tail: tuple[int, int] | None = None,
) -> list[BBox] | None:
    boxes: list[BBox] = []
    remain = int(initial_shift)
    prev = first_box
    for i in range(chain_count):
        if remain <= 0:
            return None
        max_gap = max(1, remain - 1) if remain > 1 else 0
        gap = random.randint(0, max_gap)
        h = random.randint(1, 3)
        w = random.randint(2, 4)
        pr0, pc0, pr1, pc1 = prev

        if direction == (0, 1):
            c0 = pc1 + gap + 1
            c1 = c0 + w - 1
            if c1 >= W:
                return None
            if i == chain_count - 1 and c1 + remain >= W:
                return None
            lo = max(0, pr0 - h + 1)
            hi = min(pr1, H - h)
            if lo > hi:
                return None
            r0 = random.randint(lo, hi)
            r1 = r0 + h - 1
            if gap == 0 and i == 0 and forbid_tail is not None:
                tail_row = forbid_tail[0]
                if r0 <= tail_row <= r1:
                    return None
        elif direction == (0, -1):
            c1 = pc0 - gap - 1
            c0 = c1 - w + 1
            if c0 < 0:
                return None
            if i == chain_count - 1 and c0 - remain < 0:
                return None
            lo = max(0, pr0 - h + 1)
            hi = min(pr1, H - h)
            if lo > hi:
                return None
            r0 = random.randint(lo, hi)
            r1 = r0 + h - 1
            if gap == 0 and i == 0 and forbid_tail is not None:
                tail_row = forbid_tail[0]
                if r0 <= tail_row <= r1:
                    return None
        elif direction == (1, 0):
            r0 = pr1 + gap + 1
            r1 = r0 + h - 1
            if r1 >= H:
                return None
            if i == chain_count - 1 and r1 + remain >= H:
                return None
            lo = max(0, pc0 - w + 1)
            hi = min(pc1, W - w)
            if lo > hi:
                return None
            c0 = random.randint(lo, hi)
            c1 = c0 + w - 1
            if gap == 0 and i == 0 and forbid_tail is not None:
                tail_col = forbid_tail[1]
                if c0 <= tail_col <= c1:
                    return None
        else:
            r1 = pr0 - gap - 1
            r0 = r1 - h + 1
            if r0 < 0:
                return None
            if i == chain_count - 1 and r0 - remain < 0:
                return None
            lo = max(0, pc0 - w + 1)
            hi = min(pc1, W - w)
            if lo > hi:
                return None
            c0 = random.randint(lo, hi)
            c1 = c0 + w - 1
            if gap == 0 and i == 0 and forbid_tail is not None:
                tail_col = forbid_tail[1]
                if c0 <= tail_col <= c1:
                    return None

        boxes.append((r0, c0, r1, c1))
        prev = (r0, c0, r1, c1)
        remain = max(0, remain - gap)
    return boxes


def _build_candidate(direction: Direction, H: int, W: int, chain_count: int, bg: int) -> tuple[np.ndarray, list[BBox]] | None:
    colors = list(range(1, 10))
    colors = [c for c in colors if c != bg]

    body_h = random.randint(2, 4)
    body_w = random.randint(2, 4)
    tail_len = random.randint(2, 5)
    front_room = random.randint(7, 11)

    if direction == (0, 1):
        max_sc = W - (body_w + tail_len + front_room + 1)
        if max_sc < 2:
            return None
        sr = random.randint(0, max(0, H - body_h - 1))
        sc = random.randint(0, max_sc)
        tail_anchor = random.randint(sr, sr + body_h - 1)
    elif direction == (0, -1):
        min_sc = tail_len + front_room
        max_sc = W - body_w
        if min_sc >= max_sc:
            return None
        sr = random.randint(0, max(0, H - body_h - 1))
        sc = random.randint(min_sc, max_sc)
        tail_anchor = random.randint(sr, sr + body_h - 1)
    elif direction == (1, 0):
        max_sr = H - (body_h + tail_len + front_room + 1)
        if max_sr < 2:
            return None
        sr = random.randint(0, max_sr)
        sc = random.randint(0, max(0, W - body_w - 1))
        tail_anchor = random.randint(sc, sc + body_w - 1)
    else:
        min_sr = tail_len + front_room
        max_sr = H - body_h
        if min_sr >= max_sr:
            return None
        sr = random.randint(min_sr, max_sr)
        sc = random.randint(0, max(0, W - body_w - 1))
        tail_anchor = random.randint(sc, sc + body_w - 1)

    body_box = (sr, sc, sr + body_h - 1, sc + body_w - 1)

    slider_color = random.choice(colors)
    grid = np.full((H, W), bg, dtype=int)
    _paint_slider_body_and_tail(grid, body_box, slider_color, direction, tail_len, tail_anchor)

    movers: list[BBox] = [body_box]
    forbid_tail = None
    if direction in ((0, 1), (0, -1)):
        forbid_tail = (tail_anchor, sc)
    elif direction in ((1, 0), (-1, 0)):
        forbid_tail = (sr, tail_anchor)

    chain_boxes = _build_chain_boxes_linear(body_box, direction, chain_count, tail_len, H, W, forbid_tail)
    if chain_boxes is None:
        return None
    for box in chain_boxes:
        if any(_rects_overlap(box, m) for m in movers):
            return None
        movers.append(box)

    used = colors[:1]
    for i, box in enumerate(chain_boxes):
        if i + 1 >= len(colors):
            return None
        col = colors[i + 1]
        _draw_rect(grid, box, col)
        used.append(col)
    if chain_count > 0 and len(used) != 1 + chain_count:
        return None

    return grid, movers


def _place_extras(
    grid: np.ndarray,
    movers: list[BBox],
    bg: int,
    used_colors: list[int],
    direction: Direction,
    extra_count: int,
) -> bool:
    if extra_count <= 0:
        return True

    H, W = grid.shape
    occupied = grid != bg
    move_blocked_rows, move_blocked_cols = _blocked_axis_intervals(movers, direction, H, W)
    colors = [c for c in range(1, 10) if c not in used_colors and c != bg]
    if not colors:
        return False

    for i in range(extra_count):
        if not colors:
            return False
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        col = colors.pop(random.randrange(len(colors)))

        if direction == (0, 1):
            row_candidates = [r for r in range(H - h + 1) if not move_blocked_rows[r : r + h].any()]
            col_regions: list[tuple[int, int]] = []
            min_c = min(b[1] for b in movers)
            if min_c - w - 1 >= 0:
                col_regions.append((0, min_c - w - 1))
            max_c = max(b[3] for b in movers)
            if max_c + 1 <= W - w:
                col_regions.append((max_c + 1, W - w))
        elif direction == (0, -1):
            row_candidates = [r for r in range(H - h + 1) if not move_blocked_rows[r : r + h].any()]
            col_regions: list[tuple[int, int]] = []
            min_c = min(b[1] for b in movers)
            if min_c - 1 >= 0:
                col_regions.append((0, min_c - 1))
            max_c = max(b[3] for b in movers)
            if max_c + w + 1 <= W - 1:
                col_regions.append((max_c + w + 1 - w, W - w))
        elif direction == (1, 0):
            col_candidates = [c for c in range(W - w + 1) if not move_blocked_cols[c : c + w].any()]
            row_regions: list[tuple[int, int]] = []
            min_r = min(b[0] for b in movers)
            if min_r - h - 1 >= 0:
                row_regions.append((0, min_r - h - 1))
            max_r = max(b[2] for b in movers)
            if max_r + 1 <= H - h:
                row_regions.append((max_r + 1, H - h))
        else:
            col_candidates = [c for c in range(W - w + 1) if not move_blocked_cols[c : c + w].any()]
            row_regions: list[tuple[int, int]] = []
            min_r = min(b[0] for b in movers)
            if min_r - 1 >= 0:
                row_regions.append((0, min_r - 1))
            max_r = max(b[2] for b in movers)
            if max_r + h + 1 <= H - 1:
                row_regions.append((max_r + h + 1 - h, H - h))

        candidates: list[tuple[int, int]] = []
        if direction in ((0, 1), (0, -1)):
            regions = [("col", col_regions), ("row", row_candidates)]
            if not regions[0][1] or not regions[1][1]:
                return False
            for _ in range(120):
                r0 = random.choice(row_candidates)
                seg = random.choice(regions[0][1])
                c0 = random.randint(seg[0], seg[1])
                candidates.append((r0, c0))
        else:
            regions = [("row", row_regions), ("col", col_candidates)]
            if not regions[0][1] or not regions[1][1]:
                return False
            for _ in range(120):
                c0 = random.choice(col_candidates)
                seg = random.choice(regions[0][1])
                r0 = random.randint(seg[0], seg[1])
                candidates.append((r0, c0))

        placed = False
        for r0, c0 in candidates:
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
                continue
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            _draw_rect(grid, (r0, c0, r1, c1), col)
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            used_colors.append(col)
            placed = True
            break
        if not placed:
            return False
    return True


def _changed_colors(inp: np.ndarray, out: np.ndarray, bg: int) -> list[int]:
    colors = [c for c in set(map(int, inp.ravel())) | set(map(int, out.ravel())) if c != bg]
    changed: list[int] = []
    for c in sorted(colors):
        ia = {(int(r), int(c0)) for r, c0 in np.argwhere(inp == c)}
        oa = {(int(r), int(c0)) for r, c0 in np.argwhere(out == c)}
        if ia != oa:
            changed.append(c)
    return changed


def _validate_candidate(branch: int, inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    changed = _changed_colors(inp, out, bg)
    if branch == 0:
        return len(changed) == 1
    if branch == 1:
        return len(changed) == 2
    return len(changed) >= 3


def generate_input() -> np.ndarray:
    bg = 0
    for _ in range(900):
        H = random.randint(16, 30)
        W = random.randint(16, 30)
        branch = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        chain_count = 0 if branch == 0 else (1 if branch == 1 else random.randint(2, 3))

        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

        payload = _build_candidate(direction, H, W, chain_count, bg)
        if payload is None:
            continue
        grid, movers = payload

        extra_count = 1 if branch == 0 else random.randint(0, 1)
        used_colors = [int(v) for v in set(grid[grid != bg])]
        if not _place_extras(grid, movers, bg, used_colors, direction, extra_count):
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if _validate_candidate(branch, grid, out, bg):
            if grid.shape[0] <= 30 and grid.shape[1] <= 30 and np.max(grid) <= 9 and np.max(out) <= 9:
                return grid
    raise RuntimeError("failed to generate a valid input for task 6")
