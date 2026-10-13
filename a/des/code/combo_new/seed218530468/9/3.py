from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
OBJECT_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_RETRIES = 400

DIRS = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}


def _find_source(g: np.ndarray) -> tuple[int, int, str]:
    pts = np.argwhere(g == YELLOW)
    assert pts.shape[0] == 1, "expected one yellow source marker"
    r, c = map(int, pts[0])
    h, w = g.shape
    assert (r in (0, h - 1)) or (c in (0, w - 1)), "source must be on the border"
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "source cannot be in a corner"
    if r == 0:
        return r, c, "down"
    if r == h - 1:
        return r, c, "up"
    if c == 0:
        return r, c, "right"
    return r, c, "left"


def _objects_from_grid(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    objects: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, YELLOW)):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "object color must exist"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "objects must be solid rectangles"
        objects.append((color, r0, c0, r1, c1))
    assert objects, "expected at least one rectangle object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a grey shadow from each solid rectangle using the direction implied by a single border light marker.

    Input structure:
    - The background is black.
    - Exactly one yellow pixel sits on a non-corner cell of one grid edge and acts as the light marker.
    - The interior contains several non-overlapping solid monochrome rectangles, each in a distinct non-yellow color.

    Rule:
    - The yellow marker's edge determines the global shadow direction: top -> down, bottom -> up, left -> right, right -> left.
    - Remove the yellow marker in the output, keep every colored rectangle unchanged, and draw a grey shadow for each rectangle starting immediately beyond its far edge in the projection direction.
    - For vertical projection, each shadow has the same width as its rectangle and a length equal to that width; for horizontal projection, each shadow has the same height as its rectangle and a length equal to that height.
    - Paint shadow cells only where the current output is still black; any rectangle cells block those pixels, but the shadow continues through later cells in its fixed extent.

    Key insight:
    - The edge marker supplies a single global direction, while the shadow length is controlled by the rectangle's size on the perpendicular axis rather than by its depth along the casting axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    _sr, _sc, direction = _find_source(g)
    dr, dc = DIRS[direction]
    objects = _objects_from_grid(g)

    out = g.copy()
    out[out == YELLOW] = BG

    for _color, r0, c0, r1, c1 in objects:
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        shadow_len = obj_w if dr != 0 else obj_h

        if dr == 1:
            base_r = r1 + 1
            for step in range(shadow_len):
                rr = base_r + step
                if not (0 <= rr < h):
                    continue
                for cc in range(c0, c1 + 1):
                    if out[rr, cc] == BG:
                        out[rr, cc] = GREY
        elif dr == -1:
            base_r = r0 - 1
            for step in range(shadow_len):
                rr = base_r - step
                if not (0 <= rr < h):
                    continue
                for cc in range(c0, c1 + 1):
                    if out[rr, cc] == BG:
                        out[rr, cc] = GREY
        elif dc == 1:
            base_c = c1 + 1
            for step in range(shadow_len):
                cc = base_c + step
                if not (0 <= cc < w):
                    continue
                for rr in range(r0, r1 + 1):
                    if out[rr, cc] == BG:
                        out[rr, cc] = GREY
        else:
            base_c = c0 - 1
            for step in range(shadow_len):
                cc = base_c - step
                if not (0 <= cc < w):
                    continue
                for rr in range(r0, r1 + 1):
                    if out[rr, cc] == BG:
                        out[rr, cc] = GREY

    return out


def _distribute_extra(total: int, slots: int) -> list[int]:
    out = [0] * slots
    for _ in range(total):
        out[random.randrange(slots)] += 1
    return out


def _pack_intervals(inner_len: int, lengths: list[int]) -> list[int] | None:
    if not lengths:
        return []
    min_required = sum(lengths) + (len(lengths) - 1)
    if min_required > inner_len:
        return None
    extras = _distribute_extra(inner_len - min_required, len(lengths) + 1)
    starts: list[int] = []
    pos = 1 + extras[0]
    for i, length in enumerate(lengths):
        starts.append(pos)
        pos += length
        if i < len(lengths) - 1:
            pos += 1 + extras[i + 1]
    return starts


def _middle_choices(length: int) -> list[int]:
    if length <= 4:
        return list(range(1, length - 1))
    lo = max(1, length // 3)
    hi = min(length - 2, length - 1 - length // 3)
    if lo > hi:
        lo, hi = 1, length - 2
    return list(range(lo, hi + 1))


def _sample_specs(direction: str, count: int, h: int, w: int, blocked: bool) -> list[tuple[int, int]] | None:
    vertical = direction in ("down", "up")
    inner_h = h - 2
    inner_w = w - 2
    lane_budget = inner_w if vertical else inner_h
    max_size = min(5, inner_w if vertical else inner_h, h - 2, w - 2)
    if max_size < 1:
        return None

    for _ in range(300):
        specs: list[tuple[int, int]] = []
        if blocked:
            if vertical:
                primary_w = random.randint(2, min(5, inner_w))
                primary_h = random.randint(1, min(4, h - primary_w - 1))
                blocker_w = random.randint(1, primary_w)
                blocker_h = random.randint(1, min(4, h - blocker_w - 1))
                specs.extend([(primary_h, primary_w), (blocker_h, blocker_w)])
            else:
                primary_h = random.randint(2, min(5, inner_h))
                primary_w = random.randint(1, min(4, w - primary_h - 1))
                blocker_h = random.randint(1, primary_h)
                blocker_w = random.randint(1, min(4, w - blocker_h - 1))
                specs.extend([(primary_h, primary_w), (blocker_h, blocker_w)])
            remaining = count - 2
        else:
            remaining = count

        for _ in range(remaining):
            if vertical:
                obj_w = random.randint(1, min(5, inner_w))
                obj_h = random.randint(1, min(4, h - obj_w - 1))
                specs.append((obj_h, obj_w))
            else:
                obj_h = random.randint(1, min(5, inner_h))
                obj_w = random.randint(1, min(4, w - obj_h - 1))
                specs.append((obj_h, obj_w))

        if any(min(sh, sw) < 1 for sh, sw in specs):
            continue

        if vertical:
            if any(sh + sw > h - 1 for sh, sw in specs):
                continue
            rel_sizes = [sw for sh, sw in specs]
            lane_widths = [specs[0][1]] + [sw for _sh, sw in specs[2:]] if blocked else rel_sizes
        else:
            if any(sh + sw > w - 1 for sh, sw in specs):
                continue
            rel_sizes = [sh for sh, _sw in specs]
            lane_widths = [specs[0][0]] + [sh for sh, _sw in specs[2:]] if blocked else rel_sizes

        if len(set(rel_sizes)) < 2:
            continue
        if sum(lane_widths) + max(0, len(lane_widths) - 1) > lane_budget:
            continue
        if all(sh == sw for sh, sw in specs):
            continue
        return specs
    return None


def _place_vertical_clear(h: int, w: int, direction: str, specs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    starts = _pack_intervals(w - 2, [specs[i][1] for i in order])
    if starts is None:
        return None

    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    for lane_start, idx in zip(starts, order):
        sh, sw = specs[idx]
        if direction == "down":
            lo, hi = 1, h - sh - sw
        else:
            lo, hi = sw, h - sh - 1
        if lo > hi:
            return None
        r0 = random.randint(lo, hi)
        placements[idx] = (r0, lane_start, sh, sw)

    return [p for p in placements if p is not None]


def _place_horizontal_clear(h: int, w: int, direction: str, specs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    starts = _pack_intervals(h - 2, [specs[i][0] for i in order])
    if starts is None:
        return None

    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    for lane_start, idx in zip(starts, order):
        sh, sw = specs[idx]
        if direction == "right":
            lo, hi = 1, w - sw - sh
        else:
            lo, hi = sh, w - sw - 1
        if lo > hi:
            return None
        c0 = random.randint(lo, hi)
        placements[idx] = (lane_start, c0, sh, sw)

    return [p for p in placements if p is not None]


def _place_vertical_blocked(h: int, w: int, direction: str, specs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    assert len(specs) >= 2
    (ph, pw), (bh, bw) = specs[:2]
    others = specs[2:]

    lanes = [("pair", pw, None)]
    for idx, (_sh, sw) in enumerate(others, start=2):
        lanes.append(("single", sw, idx))
    random.shuffle(lanes)

    starts = _pack_intervals(w - 2, [lane[1] for lane in lanes])
    if starts is None:
        return None

    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    pair_c = None
    for lane_start, lane in zip(starts, lanes):
        kind, _width, idx = lane
        if kind == "pair":
            pair_c = lane_start
            continue
        sh, sw = specs[idx]
        if direction == "down":
            lo, hi = 1, h - sh - sw
        else:
            lo, hi = sw, h - sh - 1
        if lo > hi:
            return None
        r0 = random.randint(lo, hi)
        placements[idx] = (r0, lane_start, sh, sw)

    assert pair_c is not None
    blocker_c = pair_c + random.randint(0, pw - bw)

    if direction == "down":
        gap = random.randint(1, pw - 1)
        top_lo = 1
        top_hi = min(h - ph - pw, h - ph - gap - bh - bw)
        if top_lo > top_hi:
            return None
        primary_r = random.randint(top_lo, top_hi)
        blocker_r = primary_r + ph + gap
    else:
        gap = random.randint(1, pw - 1)
        top_lo = max(pw, bw + bh + gap)
        top_hi = h - ph - 1
        if top_lo > top_hi:
            return None
        primary_r = random.randint(top_lo, top_hi)
        blocker_r = primary_r - bh - gap

    placements[0] = (primary_r, pair_c, ph, pw)
    placements[1] = (blocker_r, blocker_c, bh, bw)
    return [p for p in placements if p is not None]


def _place_horizontal_blocked(h: int, w: int, direction: str, specs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    assert len(specs) >= 2
    (ph, pw), (bh, bw) = specs[:2]
    others = specs[2:]

    lanes = [("pair", ph, None)]
    for idx, (sh, _sw) in enumerate(others, start=2):
        lanes.append(("single", sh, idx))
    random.shuffle(lanes)

    starts = _pack_intervals(h - 2, [lane[1] for lane in lanes])
    if starts is None:
        return None

    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    pair_r = None
    for lane_start, lane in zip(starts, lanes):
        kind, _height, idx = lane
        if kind == "pair":
            pair_r = lane_start
            continue
        sh, sw = specs[idx]
        if direction == "right":
            lo, hi = 1, w - sw - sh
        else:
            lo, hi = sh, w - sw - 1
        if lo > hi:
            return None
        c0 = random.randint(lo, hi)
        placements[idx] = (lane_start, c0, sh, sw)

    assert pair_r is not None
    blocker_r = pair_r + random.randint(0, ph - bh)

    if direction == "right":
        gap = random.randint(1, ph - 1)
        left_lo = 1
        left_hi = min(w - pw - ph, w - pw - gap - bw - bh)
        if left_lo > left_hi:
            return None
        primary_c = random.randint(left_lo, left_hi)
        blocker_c = primary_c + pw + gap
    else:
        gap = random.randint(1, ph - 1)
        left_lo = max(ph, bh + bw + gap)
        left_hi = w - pw - 1
        if left_lo > left_hi:
            return None
        primary_c = random.randint(left_lo, left_hi)
        blocker_c = primary_c - bw - gap

    placements[0] = (pair_r, primary_c, ph, pw)
    placements[1] = (blocker_r, blocker_c, bh, bw)
    return [p for p in placements if p is not None]


def _place_specs(h: int, w: int, direction: str, specs: list[tuple[int, int]], blocked: bool) -> list[tuple[int, int, int, int]] | None:
    if direction in ("down", "up"):
        if blocked:
            return _place_vertical_blocked(h, w, direction, specs)
        return _place_vertical_clear(h, w, direction, specs)
    if blocked:
        return _place_horizontal_blocked(h, w, direction, specs)
    return _place_horizontal_clear(h, w, direction, specs)


def _paint_objects(h: int, w: int, direction: str, placements: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)

    if direction == "down":
        grid[0, random.choice(_middle_choices(w))] = YELLOW
    elif direction == "up":
        grid[h - 1, random.choice(_middle_choices(w))] = YELLOW
    elif direction == "right":
        grid[random.choice(_middle_choices(h)), 0] = YELLOW
    else:
        grid[random.choice(_middle_choices(h)), w - 1] = YELLOW

    colors = random.sample(OBJECT_COLORS, len(placements))
    for color, (r0, c0, sh, sw) in zip(colors, placements):
        grid[r0 : r0 + sh, c0 : c0 + sw] = color
    return grid


def _input_is_informative(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if inp.shape != out.shape:
        return False
    if not np.any(out == GREY):
        return False
    if np.any((inp == YELLOW) & (out == YELLOW)):
        return False

    _sr, _sc, direction = _find_source(inp)
    objects = _objects_from_grid(inp)
    if len(objects) < 2:
        return False

    if direction in ("down", "up"):
        widths = {c1 - c0 + 1 for _color, _r0, c0, _r1, c1 in objects}
        if len(widths) < 2:
            return False
    else:
        heights = {r1 - r0 + 1 for _color, r0, _c0, r1, _c1 in objects}
        if len(heights) < 2:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    direction = random.choice(tuple(DIRS))
    blocked = random.choice([False, True])
    h = random.randint(11, 20)
    w = random.randint(11, 20)
    count = random.randint(2, 4) if blocked else random.randint(2, 5)
    if blocked:
        count = max(2, count)

    specs = _sample_specs(direction, count, h, w, blocked)
    if specs is None:
        return None

    placements = _place_specs(h, w, direction, specs, blocked)
    if placements is None:
        return None

    grid = _paint_objects(h, w, direction, placements)
    if not _input_is_informative(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
