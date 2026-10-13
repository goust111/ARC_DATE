from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


ARROW_COLOR = 9
MASTER_SIZE = 30
INNER_MIN = 1
INNER_MAX = MASTER_SIZE - 2
GENERATE_RETRIES = 400
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background_from_corners(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _components_with_labels(mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            labels[r, c] = idx
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(cells),
                }
            )
    return comps, labels


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    mask = (grid != int(bg)) & (grid != ARROW_COLOR)
    comps, _ = _components_with_labels(mask)
    rects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if int(comp["area"]) != int((r1 - r0 + 1) * (c1 - c0 + 1)):
            continue
        rects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int),
            }
        )
    rects.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return rects


def _movement_distance(height: int, width: int, direction: str, arrow_height: int) -> int:
    span = int(height if direction in ("up", "down") else width)
    return int(span * arrow_height)


def _match_arrow(rect: dict, arrow_mask: np.ndarray, arrow_labels: np.ndarray, arrow_comps: list[dict]) -> tuple[str, int] | None:
    r0, c0, r1, c1 = rect["bbox"]
    candidates: list[tuple[str, int]] = []

    def add_candidate(direction: str, base_cells: list[tuple[int, int]]):
        ids = {int(arrow_labels[rr, cc]) for rr, cc in base_cells}
        if len(ids) != 1 or -1 in ids:
            return
        aid = ids.pop()
        comp = arrow_comps[aid]
        ar0, ac0, ar1, ac1 = comp["bbox"]
        if direction == "up":
            if not (ar1 == r0 - 1 and ac0 >= c0 and ac1 <= c1 and all(rr < r0 for rr, _ in comp["cells"])):
                return
        elif direction == "down":
            if not (ar0 == r1 + 1 and ac0 >= c0 and ac1 <= c1 and all(rr > r1 for rr, _ in comp["cells"])):
                return
        elif direction == "left":
            if not (ac1 == c0 - 1 and ar0 >= r0 and ar1 <= r1 and all(cc < c0 for _, cc in comp["cells"])):
                return
        else:
            if not (ac0 == c1 + 1 and ar0 >= r0 and ar1 <= r1 and all(cc > c1 for _, cc in comp["cells"])):
                return
        candidates.append((direction, aid))

    if r0 > 0 and bool(arrow_mask[r0 - 1, c0 : c1 + 1].all()):
        add_candidate("up", [(r0 - 1, c) for c in range(c0, c1 + 1)])
    if r1 + 1 < arrow_mask.shape[0] and bool(arrow_mask[r1 + 1, c0 : c1 + 1].all()):
        add_candidate("down", [(r1 + 1, c) for c in range(c0, c1 + 1)])
    if c0 > 0 and bool(arrow_mask[r0 : r1 + 1, c0 - 1].all()):
        add_candidate("left", [(r, c0 - 1) for r in range(r0, r1 + 1)])
    if c1 + 1 < arrow_mask.shape[1] and bool(arrow_mask[r0 : r1 + 1, c1 + 1].all()):
        add_candidate("right", [(r, c1 + 1) for r in range(r0, r1 + 1)])

    if len(candidates) != 1:
        return None
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each marked rectangle along its adjacent triangle arrow, with larger arrows winning overlaps.

    Input structure:
    - The background fills the empty canvas.
    - Non-arrow cells form fully filled axis-aligned rectangles whose interior colors may contain patterns.
    - Some rectangles have an adjacent triangle of a dedicated arrow color (here color 9) attached flush to exactly one side.

    Rule:
    - For each attached arrow, treat its neighboring rectangle as the source block to be translated in the arrow's pointing direction.
    - The translation distance equals the arrow height times the rectangle size measured along that movement axis.
    - Erase the arrow markers and clear every moved rectangle's original position to background.
    - Stamp all translated rectangles onto their targets; if target cells overlap, the move with the larger arrow area overwrites the smaller one.

    Key insight:
    - The arrow height is a multiplier, not the offset itself: you must recover both the arrow direction and its height, then multiply by the source rectangle's own span.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_from_corners(g)

    arrow_mask = g == ARROW_COLOR
    arrow_comps, arrow_labels = _components_with_labels(arrow_mask)
    rects = _find_rectangles(g, bg)

    moves: list[dict] = []
    used_arrow_ids: set[int] = set()
    for rect in rects:
        match = _match_arrow(rect, arrow_mask, arrow_labels, arrow_comps)
        if match is None:
            continue
        direction, aid = match
        if int(aid) in used_arrow_ids:
            continue
        used_arrow_ids.add(int(aid))

        r0, c0, r1, c1 = rect["bbox"]
        rh = int(r1 - r0 + 1)
        rw = int(c1 - c0 + 1)
        ar0, ac0, ar1, ac1 = arrow_comps[aid]["bbox"]
        arrow_height = int(ar1 - ar0 + 1 if direction in ("up", "down") else ac1 - ac0 + 1)
        dist = _movement_distance(rh, rw, direction, arrow_height)
        dr, dc = DIRS[direction]
        tr = int(r0 + dr * dist)
        tc = int(c0 + dc * dist)
        moves.append(
            {
                "source": (int(r0), int(c0), int(r1), int(c1)),
                "target": (int(tr), int(tc)),
                "patch": rect["patch"],
                "arrow_area": int(arrow_comps[aid]["area"]),
            }
        )

    out = g.copy()
    out[arrow_mask] = int(bg)
    for move in moves:
        r0, c0, r1, c1 = move["source"]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    moves.sort(key=lambda d: (int(d["arrow_area"]), int(d["source"][0]), int(d["source"][1])))
    for move in moves:
        patch = move["patch"]
        ph, pw = patch.shape
        tr, tc = move["target"]
        rr0 = max(0, int(tr))
        cc0 = max(0, int(tc))
        rr1 = min(h, int(tr + ph))
        cc1 = min(w, int(tc + pw))
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        out[rr0:rr1, cc0:cc1] = patch[rr0 - tr : rr1 - tr, cc0 - tc : cc1 - tc]

    return out


def _segment_lengths(base_len: int, height: int) -> list[int]:
    if height <= 1:
        return [int(base_len)]
    out: list[int] = []
    prev = int(base_len)
    for d in range(height):
        seg = int(round(base_len - (base_len - 1) * d / (height - 1)))
        if d == height - 1:
            seg = 1
        seg = max(1, min(prev, seg))
        out.append(int(seg))
        prev = int(seg)
    return out


def _arrow_cells(r0: int, c0: int, height: int, width: int, direction: str, arrow_height: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    base_len = int(width if direction in ("up", "down") else height)
    lengths = _segment_lengths(base_len, arrow_height)
    for d, seg in enumerate(lengths):
        if direction == "up":
            rr = int(r0 - 1 - d)
            start = int(c0 + (width - seg) // 2)
            cells.extend((rr, int(cc)) for cc in range(start, start + seg))
        elif direction == "down":
            rr = int(r0 + height + d)
            start = int(c0 + (width - seg) // 2)
            cells.extend((rr, int(cc)) for cc in range(start, start + seg))
        elif direction == "left":
            cc = int(c0 - 1 - d)
            start = int(r0 + (height - seg) // 2)
            cells.extend((int(rr), cc) for rr in range(start, start + seg))
        else:
            cc = int(c0 + width + d)
            start = int(r0 + (height - seg) // 2)
            cells.extend((int(rr), cc) for rr in range(start, start + seg))
    return cells


def _rect_cells(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r in range(r0, r0 + height) for c in range(c0, c0 + width)}


def _dilate_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < MASTER_SIZE and 0 <= cc < MASTER_SIZE:
                    out.add((rr, cc))
    return out


def _all_inside_inner(cells: set[tuple[int, int]]) -> bool:
    return all(INNER_MIN <= r <= INNER_MAX and INNER_MIN <= c <= INNER_MAX for r, c in cells)


def _random_palette(bg: int, k: int) -> list[int]:
    colors = [c for c in range(10) if c not in (bg, ARROW_COLOR)]
    return random.sample(colors, k)


def _make_patch(height: int, width: int, bg: int) -> np.ndarray:
    colors = _random_palette(bg, random.choice([1, 2, 2, 3]))
    kind = random.choice(["solid", "hstripe", "vstripe", "checker", "frame", "quadrants"])
    patch = np.zeros((height, width), dtype=int)
    if kind == "solid" or len(colors) == 1:
        patch[:, :] = int(colors[0])
        return patch
    if kind == "hstripe":
        for r in range(height):
            patch[r, :] = int(colors[r % len(colors)])
        return patch
    if kind == "vstripe":
        for c in range(width):
            patch[:, c] = int(colors[c % len(colors)])
        return patch
    if kind == "checker":
        for r in range(height):
            for c in range(width):
                patch[r, c] = int(colors[(r + c) % len(colors)])
        return patch
    if kind == "frame" and height >= 3 and width >= 3:
        patch[:, :] = int(colors[0])
        inner = colors[1:] if len(colors) > 1 else colors
        patch[1:-1, 1:-1] = int(inner[0])
        return patch
    patch[:, :] = int(colors[0])
    mids_r = (height + 1) // 2
    mids_c = (width + 1) // 2
    palette = colors + colors[:1] * max(0, 4 - len(colors))
    patch[:mids_r, :mids_c] = int(palette[0])
    patch[:mids_r, mids_c:] = int(palette[1 % len(palette)])
    patch[mids_r:, :mids_c] = int(palette[2 % len(palette)])
    patch[mids_r:, mids_c:] = int(palette[3 % len(palette)])
    return patch


def _build_move(r0: int, c0: int, height: int, width: int, direction: str, arrow_height: int, patch: np.ndarray) -> dict:
    rect = _rect_cells(r0, c0, height, width)
    arrow = set(_arrow_cells(r0, c0, height, width, direction, arrow_height))
    dist = _movement_distance(height, width, direction, arrow_height)
    dr, dc = DIRS[direction]
    tr = int(r0 + dr * dist)
    tc = int(c0 + dc * dist)
    target = _rect_cells(tr, tc, height, width)
    return {
        "src": (int(r0), int(c0)),
        "shape": (int(height), int(width)),
        "direction": direction,
        "arrow_height": int(arrow_height),
        "patch": patch,
        "rect_cells": rect,
        "arrow_cells": arrow,
        "footprint": rect | arrow,
        "target_cells": target,
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch


def _sample_non_overlap_move(bg: int, blocked: set[tuple[int, int]]) -> dict | None:
    for _ in range(250):
        direction = random.choice(list(DIRS))
        height = random.randint(2, 6)
        width = random.randint(2, 6)
        arrow_height = random.randint(2, 4)
        dist = _movement_distance(height, width, direction, arrow_height)
        if direction == "up":
            r_lo, r_hi = INNER_MIN + dist, INNER_MAX - height + 1
            c_lo, c_hi = INNER_MIN, INNER_MAX - width + 1
        elif direction == "down":
            r_lo, r_hi = INNER_MIN, INNER_MAX - height - dist + 1
            c_lo, c_hi = INNER_MIN, INNER_MAX - width + 1
        elif direction == "left":
            r_lo, r_hi = INNER_MIN, INNER_MAX - height + 1
            c_lo, c_hi = INNER_MIN + dist, INNER_MAX - width + 1
        else:
            r_lo, r_hi = INNER_MIN, INNER_MAX - height + 1
            c_lo, c_hi = INNER_MIN, INNER_MAX - width - dist + 1
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        patch = _make_patch(height, width, bg)
        move = _build_move(r0, c0, height, width, direction, arrow_height, patch)
        all_cells = move["footprint"] | move["target_cells"]
        if not _all_inside_inner(all_cells):
            continue
        if any(cell in blocked for cell in all_cells):
            continue
        return move
    return None


def _sample_overlap_pair(bg: int, blocked: set[tuple[int, int]]) -> tuple[dict, dict] | None:
    for _ in range(250):
        axis = random.choice(["horizontal", "vertical"])
        if axis == "horizontal":
            height = random.randint(3, 6)
            width = random.randint(2, 4)
            dir1, dir2 = "right", "left"
        else:
            height = random.randint(2, 4)
            width = random.randint(3, 6)
            dir1, dir2 = "down", "up"
        arrow_height1, arrow_height2 = random.sample([2, 3, 4], 2)
        dist1 = _movement_distance(height, width, dir1, arrow_height1)
        dist2 = _movement_distance(height, width, dir2, arrow_height2)
        if axis == "horizontal":
            r_lo, r_hi = INNER_MIN, INNER_MAX - height + 1
            c_lo = INNER_MIN + dist1
            c_hi = INNER_MAX - width - dist2 + 1
        else:
            r_lo = INNER_MIN + dist1
            r_hi = INNER_MAX - height - dist2 + 1
            c_lo, c_hi = INNER_MIN, INNER_MAX - width + 1
        if r_lo > r_hi or c_lo > c_hi:
            continue
        tr = random.randint(r_lo, r_hi)
        tc = random.randint(c_lo, c_hi)
        if axis == "horizontal":
            src1 = (tr, tc - dist1)
            src2 = (tr, tc + dist2)
        else:
            src1 = (tr - dist1, tc)
            src2 = (tr + dist2, tc)
        patch1 = _make_patch(height, width, bg)
        patch2 = _make_patch(height, width, bg)
        if bool((patch1 == patch2).all()):
            continue
        move1 = _build_move(src1[0], src1[1], height, width, dir1, arrow_height1, patch1)
        move2 = _build_move(src2[0], src2[1], height, width, dir2, arrow_height2, patch2)
        if move1["target_cells"] != move2["target_cells"]:
            continue
        if len(move1["arrow_cells"]) == len(move2["arrow_cells"]):
            continue
        cells1 = move1["footprint"]
        cells2 = move2["footprint"]
        shared_target = move1["target_cells"]
        all_cells = cells1 | cells2 | shared_target
        if not _all_inside_inner(all_cells):
            continue
        if any(cell in blocked for cell in all_cells):
            continue
        if cells1 & cells2:
            continue
        return move1, move2
    return None


def _sample_static_rect(bg: int, blocked: set[tuple[int, int]]) -> dict | None:
    for _ in range(120):
        height = random.randint(2, 6)
        width = random.randint(2, 6)
        r0 = random.randint(INNER_MIN, INNER_MAX - height + 1)
        c0 = random.randint(INNER_MIN, INNER_MAX - width + 1)
        cells = _rect_cells(r0, c0, height, width)
        if any(cell in blocked for cell in cells):
            continue
        return {"src": (int(r0), int(c0)), "patch": _make_patch(height, width, bg), "cells": cells}
    return None


def _render_case(bg: int, moves: list[dict], statics: list[dict]) -> np.ndarray:
    canvas = np.full((MASTER_SIZE, MASTER_SIZE), int(bg), dtype=int)
    for item in statics:
        r0, c0 = item["src"]
        _place_patch(canvas, item["patch"], r0, c0)
    for move in moves:
        r0, c0 = move["src"]
        _place_patch(canvas, move["patch"], r0, c0)
        for rr, cc in move["arrow_cells"]:
            canvas[int(rr), int(cc)] = ARROW_COLOR
    crop_cells: set[tuple[int, int]] = set()
    for item in statics:
        crop_cells |= item["cells"]
    for move in moves:
        crop_cells |= move["footprint"]
        crop_cells |= move["target_cells"]
    rows = [r for r, _ in crop_cells]
    cols = [c for _, c in crop_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    top_pad = random.randint(1, min(3, r0))
    bottom_pad = random.randint(1, min(3, MASTER_SIZE - 1 - r1))
    left_pad = random.randint(1, min(3, c0))
    right_pad = random.randint(1, min(3, MASTER_SIZE - 1 - c1))
    return np.array(
        canvas[r0 - top_pad : r1 + bottom_pad + 1, c0 - left_pad : c1 + right_pad + 1],
        dtype=int,
    )


def _build_case(overlap: bool) -> np.ndarray | None:
    bg = random.choice([c for c in range(9) if c != ARROW_COLOR])
    blocked: set[tuple[int, int]] = set()
    moves: list[dict] = []
    statics: list[dict] = []

    if overlap:
        pair = _sample_overlap_pair(bg, blocked)
        if pair is None:
            return None
        move1, move2 = pair
        moves.extend([move1, move2])
        blocked |= _dilate_cells(move1["footprint"] | move2["footprint"] | move1["target_cells"])
        remaining_moves = random.choice([0, 1])
    else:
        remaining_moves = random.randint(2, 4)

    while (len(moves) < remaining_moves) if not overlap else (remaining_moves > 0):
        move = _sample_non_overlap_move(bg, blocked)
        if move is None:
            return None
        moves.append(move)
        blocked |= _dilate_cells(move["footprint"] | move["target_cells"])
        if overlap:
            remaining_moves -= 1

    for _ in range(random.choice([0, 1, 1, 2])):
        item = _sample_static_rect(bg, blocked)
        if item is None:
            break
        statics.append(item)
        blocked |= _dilate_cells(item["cells"])

    inp = _render_case(bg, moves, statics)
    out = main(inp)
    if bool((inp == ARROW_COLOR).any()) and not bool(np.array_equal(inp, out)):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        overlap = bool(random.getrandbits(1))
        grid = _build_case(overlap=overlap)
        if grid is None:
            continue
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid push-by-arrow sample")
