from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 5)
GRID_LIMIT = 30
ODD_SIZES = [3, 5, 7]
OTHER_SIZES = [2, 3, 4, 5, 6]
INPUT_CLEARANCE = 1
OUTPUT_CLEARANCE = 1
PLACEMENT_ATTEMPTS = 180


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _direction_from_marker(mr: int, mc: int, h: int, w: int) -> tuple[str, tuple[int, int]]:
    candidates: list[tuple[str, tuple[int, int]]] = []
    if h % 2 == 1 and mr == h // 2:
        if mc == 0:
            candidates.append(("left", (0, -h)))
        if mc == w - 1:
            candidates.append(("right", (0, h)))
    if w % 2 == 1 and mc == w // 2:
        if mr == 0:
            candidates.append(("up", (-w, 0)))
        if mr == h - 1:
            candidates.append(("down", (w, 0)))
    assert len(candidates) == 1, "marker must sit at exactly one edge midpoint"
    return candidates[0]


def _parse_object(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> dict:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, patch.shape)

    assert bool(np.all(patch != bg)), "each component bbox must be a dense rectangle"
    vals, counts = np.unique(patch, return_counts=True)
    assert len(vals) == 2, "each rectangle must have one marker color and one body color"
    marker_idx = int(np.argmin(counts))
    marker_color = int(vals[marker_idx])
    body_color = int(vals[1 - marker_idx])
    assert int(counts[marker_idx]) == 1, "marker must be a single pixel"
    assert int(counts[1 - marker_idx]) == patch.size - 1, "body must fill the rest of the rectangle"

    marker_pos = np.argwhere(patch == marker_color)
    assert len(marker_pos) == 1, "marker position must be unique"
    mr, mc = map(int, marker_pos[0])
    direction, shift = _direction_from_marker(mr, mc, h, w)

    return {
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "patch": patch.copy(),
        "direction": direction,
        "shift": shift,
        "body_color": body_color,
        "marker_color": marker_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each marked rectangle in the direction encoded by its internal edge marker.

    Input structure:
    - The background is black.
    - Several separated non-background components each form a dense axis-aligned rectangle.
    - Inside each rectangle, exactly one pixel has a different color and sits on the midpoint cell of one edge.

    Rule:
    - Treat each non-background component as one rectangle object including its marker pixel.
    - The marker edge gives the movement direction: left edge -> move left, right edge -> move right, top edge -> move up, bottom edge -> move down.
    - Move the whole rectangle by a distance equal to its size in the perpendicular direction: height for left/right moves, width for up/down moves.
    - Clear the original position to background and clip any translated pixels that leave the grid.

    Key insight:
    The only explicit cue is the marker location; the shift length must be inferred from the rectangle's own geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    objects = [_parse_object(g, comp, bg) for comp in _components(g != bg)]
    assert objects, "expected at least one marked rectangle"
    objects.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))

    out = np.full(g.shape, int(bg), dtype=int)
    gh, gw = map(int, g.shape)
    for obj in objects:
        patch = obj["patch"]
        dr, dc = map(int, obj["shift"])
        nr0 = int(obj["r0"] + dr)
        nc0 = int(obj["c0"] + dc)
        for rr in range(int(obj["h"])):
            tr = int(nr0 + rr)
            if not (0 <= tr < gh):
                continue
            for cc in range(int(obj["w"])):
                tc = int(nc0 + cc)
                if not (0 <= tc < gw):
                    continue
                assert int(out[tr, tc]) == int(bg), "translated rectangles must not overlap"
                out[tr, tc] = int(patch[rr, cc])
    return out


def _build_patch(body_color: int, marker_color: int, h: int, w: int, direction: str) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(body_color), dtype=int)
    if direction == "left":
        patch[h // 2, 0] = int(marker_color)
    elif direction == "right":
        patch[h // 2, w - 1] = int(marker_color)
    elif direction == "up":
        patch[0, w // 2] = int(marker_color)
    elif direction == "down":
        patch[h - 1, w // 2] = int(marker_color)
    else:
        raise AssertionError("unknown direction")
    return patch


def _shift_for(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "left":
        return 0, -int(h)
    if direction == "right":
        return 0, int(h)
    if direction == "up":
        return -int(w), 0
    if direction == "down":
        return int(w), 0
    raise AssertionError("unknown direction")


def _visible_bbox(r0: int, c0: int, h: int, w: int, dr: int, dc: int, gh: int, gw: int) -> tuple[int, int, int, int] | None:
    nr0, nc0 = int(r0 + dr), int(c0 + dc)
    nr1, nc1 = int(nr0 + h - 1), int(nc0 + w - 1)
    vr0, vc0 = max(0, nr0), max(0, nc0)
    vr1, vc1 = min(gh - 1, nr1), min(gw - 1, nc1)
    if vr0 > vr1 or vc0 > vc1:
        return None
    return int(vr0), int(vc0), int(vr1), int(vc1)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _axis_interval(limit: int, span: int, shift: int, side: str, clipped: bool) -> tuple[int, int] | None:
    if side in ("left", "up"):
        if clipped:
            lo = max(0, int(shift - span + 1))
            hi = min(int(limit - span), int(shift - 1))
        else:
            lo = int(shift)
            hi = int(limit - span)
    else:
        if clipped:
            lo = max(0, int(limit - span - shift + 1))
            hi = min(int(limit - span), int(limit - shift - 1))
        else:
            lo = 0
            hi = int(limit - span - shift)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _sample_shape(direction: str, gh: int, gw: int) -> tuple[int, int] | None:
    odd_h = [v for v in ODD_SIZES if v <= gh]
    odd_w = [v for v in ODD_SIZES if v <= gw]
    any_h = [v for v in OTHER_SIZES if v <= gh]
    any_w = [v for v in OTHER_SIZES if v <= gw]
    if direction in ("left", "right"):
        if not odd_h or not any_w:
            return None
        h = int(random.choice(odd_h))
        if random.random() < 0.35 and h <= gw:
            return int(h), int(h)
        w = int(random.choice(any_w))
        return int(h), int(w)
    if not odd_w or not any_h:
        return None
    w = int(random.choice(odd_w))
    if random.random() < 0.35 and w <= gh:
        return int(w), int(w)
    h = int(random.choice(any_h))
    return int(h), int(w)


def _pick_directions(nobj: int) -> list[str]:
    dirs = ["left", "right", "up", "down"]
    random.shuffle(dirs)
    out = dirs[: min(nobj, 4)]
    while len(out) < nobj:
        out.append(random.choice(dirs))
    random.shuffle(out)
    return out


def _clip_flags(nobj: int) -> list[bool]:
    flags = [True, False]
    while len(flags) < nobj:
        flags.append(bool(random.getrandbits(1)))
    random.shuffle(flags)
    return flags[:nobj]


def _gen_attempt() -> np.ndarray | None:
    nobj = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(12, 8 + 2 * nobj)
    gh = random.randint(min_side, GRID_LIMIT)
    gw = random.randint(min_side, GRID_LIMIT)

    directions = _pick_directions(nobj)
    clip_flags = _clip_flags(nobj)
    body_colors = random.sample(PALETTE, nobj)

    objects: list[dict] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for idx in range(nobj):
        direction = directions[idx]
        clipped = clip_flags[idx]
        body_color = int(body_colors[idx])
        marker_choices = [c for c in PALETTE if c != body_color]
        marker_color = int(random.choice(marker_choices))

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _sample_shape(direction, gh, gw)
            if shape is None:
                return None
            h, w = map(int, shape)
            dr, dc = _shift_for(direction, h, w)

            if direction in ("left", "right"):
                row_interval = (0, gh - h)
                col_interval = _axis_interval(gw, w, h, direction, clipped)
            else:
                row_interval = _axis_interval(gh, h, w, direction, clipped)
                col_interval = (0, gw - w)
            if row_interval is None or col_interval is None:
                continue

            r0 = random.randint(int(row_interval[0]), int(row_interval[1]))
            c0 = random.randint(int(col_interval[0]), int(col_interval[1]))
            in_box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            out_box = _visible_bbox(r0, c0, h, w, dr, dc, gh, gw)
            if out_box is None:
                continue
            if any(_boxes_too_close(in_box, prev, INPUT_CLEARANCE) for prev in input_boxes):
                continue
            if any(_boxes_too_close(out_box, prev, OUTPUT_CLEARANCE) for prev in output_boxes):
                continue

            objects.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(h),
                    "w": int(w),
                    "direction": direction,
                    "body_color": body_color,
                    "marker_color": marker_color,
                    "patch": _build_patch(body_color, marker_color, h, w, direction),
                }
            )
            input_boxes.append(in_box)
            output_boxes.append(out_box)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((gh, gw), dtype=int)
    for obj in objects:
        r0, c0 = int(obj["r0"]), int(obj["c0"])
        h, w = int(obj["h"]), int(obj["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = obj["patch"]

    # Re-run the solver to guarantee the generator satisfies its assumptions.
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
