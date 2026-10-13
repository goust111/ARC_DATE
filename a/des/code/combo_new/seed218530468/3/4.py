import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAME_SIZE = 5
FRAME_BORDER_AREA = 16
MAX_GRID_SIZE = 30
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 160
PAIR_GAP_RANGE = (1, 3)
PAIR_COUNT_RANGE = (2, 4)
PAIR_PAD = 1
MARGIN_CAP = 4

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

COMPATIBLE_MISSING = {
    "up": ((0, 0), (0, 1)),
    "down": ((1, 0), (1, 1)),
    "left": ((0, 0), (1, 0)),
    "right": ((0, 1), (1, 1)),
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border_set(r0: int, c0: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r0 + FRAME_SIZE):
        for c in range(c0, c0 + FRAME_SIZE):
            if r in (r0, r0 + FRAME_SIZE - 1) or c in (c0, c0 + FRAME_SIZE - 1):
                cells.add((r, c))
    return cells


def _is_frame_component(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != FRAME_BORDER_AREA:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != FRAME_SIZE or c1 - c0 + 1 != FRAME_SIZE:
        return False
    return set(coords) == _frame_border_set(r0, c0)


def _decode_l(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
        return None

    local = {(r - r0, c - c0) for r, c in coords}
    missing = next((((r, c) for r in range(2) for c in range(2) if (r, c) not in local)), None)
    if missing is None:
        return None

    pivot = (1 - missing[0], 1 - missing[1])
    if pivot not in local:
        return None

    arms: list[tuple[str, tuple[int, int]]] = []
    for name, (dr, dc) in DIRS.items():
        nr = pivot[0] + dr
        nc = pivot[1] + dc
        if 0 <= nr < 2 and 0 <= nc < 2 and (nr, nc) in local:
            arms.append((name, (r0 + nr, c0 + nc)))

    if len(arms) != 2:
        return None
    return {"bbox": (r0, c0, r1, c1), "missing": missing, "arms": arms}


def _first_frame_on_ray(frame_ids: np.ndarray, endpoint: tuple[int, int], direction: str) -> int | None:
    h, w = frame_ids.shape
    dr, dc = DIRS[direction]
    r = endpoint[0] + dr
    c = endpoint[1] + dc
    while 0 <= r < h and 0 <= c < w:
        frame_id = int(frame_ids[r, c])
        if frame_id >= 0:
            return frame_id
        r += dr
        c += dc
    return None


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= MAX_GRID_SIZE

    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    comps = _components4(g != bg)

    frames: list[dict[str, object]] = []
    l_shapes: list[dict[str, object]] = []
    for comp in comps:
        color = int(g[comp[0][0], comp[0][1]])
        if _is_frame_component(comp):
            r0, c0, r1, c1 = _bbox(comp)
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "cells": set(comp),
                    "color": color,
                }
            )
            continue

        l_info = _decode_l(comp)
        if l_info is not None:
            l_info["color"] = color
            l_info["cells"] = set(comp)
            l_shapes.append(l_info)
            continue

        raise AssertionError("unexpected foreground component")

    assert frames, "expected at least one frame"
    assert l_shapes, "expected at least one L shape"

    frame_ids = np.full(g.shape, -1, dtype=int)
    for idx, frame in enumerate(frames):
        for r, c in frame["cells"]:
            frame_ids[r, c] = idx

    out = g.copy()
    hits_per_frame = [0] * len(frames)
    for l_info in l_shapes:
        candidates: list[int] = []
        for direction, endpoint in l_info["arms"]:
            frame_id = _first_frame_on_ray(frame_ids, endpoint, direction)
            if frame_id is not None:
                candidates.append(frame_id)
        assert len(candidates) == 1, "each L must point to exactly one frame"

        target = frames[candidates[0]]
        hits_per_frame[candidates[0]] += 1
        r0, c0, _, _ = target["bbox"]
        miss_r, miss_c = l_info["missing"]
        fill_r = r0 + 1 + (1 - miss_r)
        fill_c = c0 + 1 + (1 - miss_c)
        out[fill_r : fill_r + 2, fill_c : fill_c + 2] = int(l_info["color"])

    return out, {"bg": bg, "frames": frames, "l_shapes": l_shapes, "hits_per_frame": hits_per_frame}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each pointed hollow square with a mirrored 2x2 completion of its nearby colored L marker.

    Input structure:
    - The background is black.
    - The grid contains one or more grey hollow 5x5 square frames with 3x3 black interiors.
    - Outside the frames are colored 3-cell L shapes, each occupying a 2x2 box with one missing corner.

    Rule:
    - For each L shape, inspect its two orthogonal arms and extend each arm forward in a straight line.
    - Exactly one arm first reaches a grey frame; that frame is the L shape's target.
    - Determine which corner of the L shape's 2x2 box is missing, then paint a solid 2x2 block of the L color inside the target frame's 3x3 interior at the diagonally opposite corner.
    - Leave the original L shapes and all grey frame borders unchanged.

    Key insight:
    - The L shape encodes two pieces of information at once: the arm direction selects the destination frame, and the missing corner must be mirrored to the opposite corner when placing the interior 2x2 block.
    """
    return _solve(grid)[0]


def _l_cells_from_missing(missing: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2) for c in range(2) if (r, c) != missing]


def _build_pair(
    direction: str,
    missing: tuple[int, int],
    gap: int,
    align: int,
    color: int,
) -> dict[str, object]:
    frame_cells = list(_frame_border_set(0, 0))
    pivot = (1 - missing[0], 1 - missing[1])
    dr, dc = DIRS[direction]
    endpoint = (pivot[0] + dr, pivot[1] + dc)
    assert 0 <= endpoint[0] < 2 and 0 <= endpoint[1] < 2
    assert endpoint in _l_cells_from_missing(missing)

    if direction == "right":
        l_r0 = align - endpoint[0]
        l_c0 = -(gap + 2)
    elif direction == "left":
        l_r0 = align - endpoint[0]
        l_c0 = FRAME_SIZE + gap
    elif direction == "down":
        l_r0 = -(gap + 2)
        l_c0 = align - endpoint[1]
    else:
        l_r0 = FRAME_SIZE + gap
        l_c0 = align - endpoint[1]

    l_cells = [(l_r0 + r, l_c0 + c) for r, c in _l_cells_from_missing(missing)]
    all_cells = frame_cells + l_cells
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shift_r = -min_r
    shift_c = -min_c

    frame_cells = [(r + shift_r, c + shift_c) for r, c in frame_cells]
    l_cells = [(r + shift_r, c + shift_c) for r, c in l_cells]

    max_r = max(r for r, _ in frame_cells + l_cells)
    max_c = max(c for _, c in frame_cells + l_cells)

    return {
        "direction": direction,
        "missing": missing,
        "color": color,
        "frame_cells": frame_cells,
        "l_cells": l_cells,
        "height": max_r + 1,
        "width": max_c + 1,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _crop_with_margins(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)

    max_top = min(int(r0), MARGIN_CAP)
    max_left = min(int(c0), MARGIN_CAP)
    max_bottom = min(int(grid.shape[0] - 1 - r1), MARGIN_CAP)
    max_right = min(int(grid.shape[1] - 1 - c1), MARGIN_CAP)

    top = random.randint(0, max_top)
    left = random.randint(0, max_left)
    bottom = random.randint(0, max_bottom)
    right = random.randint(0, max_right)
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, k=pair_count)
    specs: list[dict[str, object]] = []
    for color in colors:
        direction = random.choice(tuple(DIRS))
        missing = random.choice(COMPATIBLE_MISSING[direction])
        gap = random.randint(*PAIR_GAP_RANGE)
        align = random.randint(0, FRAME_SIZE - 1)
        specs.append(_build_pair(direction, missing, gap, align, color))

    canvas = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), BG, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: int(item["height"]) * int(item["width"]), reverse=True):
        h = int(spec["height"])
        w = int(spec["width"])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(0, MAX_GRID_SIZE - h)
            left = random.randint(0, MAX_GRID_SIZE - w)
            box = (top, left, top + h - 1, left + w - 1)
            if any(_boxes_overlap(box, other, PAIR_PAD) for other in placed_boxes):
                continue

            for r, c in spec["frame_cells"]:
                canvas[top + r, left + c] = FRAME_COLOR
            for r, c in spec["l_cells"]:
                canvas[top + r, left + c] = int(spec["color"])

            placed_boxes.append(box)
            placed = True
            break

        if not placed:
            return None

    cropped = _crop_with_margins(canvas)
    try:
        out, meta = _solve(cropped)
    except AssertionError:
        return None

    if len(meta["frames"]) != pair_count or len(meta["l_shapes"]) != pair_count:
        return None
    if any(hit != 1 for hit in meta["hits_per_frame"]):
        return None
    if np.array_equal(cropped, out):
        return None
    if cropped.shape != out.shape or max(cropped.shape) > MAX_GRID_SIZE:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
