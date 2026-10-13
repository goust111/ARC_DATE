from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np


RED = 2
GREEN = 3
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_offsets(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _extract_lane(
    grid: np.ndarray,
    bg: int,
    marker_mask: np.ndarray,
    marker_coords: np.ndarray,
) -> dict[str, object] | None:
    h, w = grid.shape
    _r0, c0, _r1, c1 = _bbox(marker_coords)
    lane_mask = np.zeros((h, w), dtype=bool)
    lane_mask[:, c0 : c1 + 1] = True
    sprite_mask = lane_mask & (grid != bg) & (~marker_mask)
    comps = _components(sprite_mask)
    if not comps:
        return None

    groups: dict[tuple[tuple[tuple[int, int], ...], int], list[np.ndarray]] = defaultdict(list)
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        groups[(_shape_offsets(comp), color)].append(comp)
    if not groups:
        return None

    best_key = max(groups, key=lambda key: (len(groups[key]), len(key[0])))
    shape, color = best_key
    sprites = groups[best_key]
    sprites = sorted(sprites, key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1]))

    top_rows = [_bbox(pts)[0] for pts in sprites]
    left_cols = [_bbox(pts)[1] for pts in sprites]
    diffs = [b - a for a, b in zip(top_rows, top_rows[1:]) if b > a]
    period = None
    for diff in diffs:
        period = diff if period is None else math.gcd(period, diff)

    sprite_h = max(dr for dr, _dc in shape) + 1
    return {
        "c0": c0,
        "c1": c1,
        "center": (c0 + c1) / 2.0,
        "shape": shape,
        "color": int(color),
        "sprites": sprites,
        "occupied": set(int(x) for x in top_rows),
        "period": period,
        "bottom_top": int(max(top_rows)),
        "ref_left": int(Counter(left_cols).most_common(1)[0][0]),
        "sprite_h": int(sprite_h),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify red- and green-marked vertical channels, then move counts from red channels into the nearest green channel.

    - Sprites inside each red channel turn grey in place.
    - The same number of new sprites are added to the nearest green channel.
    - New sprites use the source sprite shape and the destination channel color.
    - Marker bars disappear.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    red_markers = _components(g == RED)
    green_markers = _components(g == GREEN)
    assert red_markers and green_markers, "expected both red and green marker bars"

    marker_mask = (g == RED) | (g == GREEN)
    lanes: list[dict[str, object]] = []
    for kind, markers in (("red", red_markers), ("green", green_markers)):
        for marker in markers:
            lane = _extract_lane(g, bg, marker_mask, marker)
            if lane is None:
                continue
            lane["kind"] = kind
            lane["marker"] = marker
            lanes.append(lane)

    red_lanes = [lane for lane in lanes if lane["kind"] == "red"]
    green_lanes = [lane for lane in lanes if lane["kind"] == "green"]
    assert red_lanes and green_lanes, "expected usable red and green channels"

    global_period = None
    for lane in lanes:
        period = lane["period"]
        if period is None:
            continue
        global_period = int(period) if global_period is None else math.gcd(global_period, int(period))
    if global_period is None:
        global_period = max(int(lane["sprite_h"]) for lane in lanes) + 1
    for lane in lanes:
        if lane["period"] is None:
            lane["period"] = int(global_period)

    out = g.copy()
    red_lanes = sorted(red_lanes, key=lambda lane: (float(lane["center"]), int(lane["c0"])))
    for red_lane in red_lanes:
        sprites = red_lane["sprites"]
        for sprite in sprites:
            out[sprite[:, 0], sprite[:, 1]] = GREY

        dst = min(
            green_lanes,
            key=lambda lane: (
                abs(float(lane["center"]) - float(red_lane["center"])),
                float(lane["center"]),
                int(lane["c0"]),
            ),
        )

        placed = 0
        slot_index = 0
        shape = red_lane["shape"]
        while placed < len(sprites):
            new_top = int(dst["bottom_top"]) - slot_index * int(dst["period"])
            slot_index += 1
            if new_top < 0:
                break
            if new_top in dst["occupied"]:
                continue

            cells = [(new_top + dr, int(dst["ref_left"]) + dc) for dr, dc in shape]
            ok = True
            for rr, cc in cells:
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for rr, cc in cells:
                out[rr, cc] = int(dst["color"])
            dst["occupied"].add(new_top)
            placed += 1

    for lane in lanes:
        marker = lane["marker"]
        out[marker[:, 0], marker[:, 1]] = bg

    return out


GEN_ATTEMPTS = 400
BG_CHOICES = tuple(c for c in range(10) if c not in (RED, GREEN, GREY))


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_sprite_mask() -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 4)
        width = random.randint(2, 5)
        lo = max(3, (height * width + 1) // 2)
        hi = height * width
        area = random.randint(lo, hi)
        mask = _grow_connected_mask(height, width, area)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if random.choice([False, True]):
            mask = np.flipud(mask)
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        return mask.copy()
    return np.ones((2, 2), dtype=bool)


def _paint_stack(
    grid: np.ndarray,
    lane_left: int,
    sprite_left: int,
    slot_tops: list[int],
    count: int,
    mask: np.ndarray,
    color: int,
) -> None:
    del lane_left  # lane-left is part of the scene description, but painting uses sprite-left directly.
    offsets = np.argwhere(mask)
    for top in slot_tops[-count:]:
        rr = top + offsets[:, 0]
        cc = sprite_left + offsets[:, 1]
        grid[rr, cc] = int(color)


def _nearest_green(center2: dict[int, int], red_idx: int, green_indices: list[int]) -> int:
    return min(green_indices, key=lambda idx: (abs(center2[red_idx] - center2[idx]), center2[idx], idx))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    sprite_mask = _sample_sprite_mask()
    sprite_h, sprite_w = sprite_mask.shape
    gap_y = random.randint(1, 3)
    period = sprite_h + gap_y
    slot_count = random.randint(4, 7)

    marker_side = random.choice(["top", "bottom"])
    marker_h = random.randint(1, 2)
    marker_gap = random.randint(1, 2)
    bottom_margin = random.randint(0, 2)
    top_margin = random.randint(0, 2)

    if marker_side == "top":
        marker_r0 = top_margin
        top_start = marker_r0 + marker_h + marker_gap
        height = top_start + (slot_count - 1) * period + sprite_h + bottom_margin
    else:
        top_start = top_margin
        bottom_top = top_start + (slot_count - 1) * period
        marker_r0 = bottom_top + sprite_h + marker_gap
        height = marker_r0 + marker_h + bottom_margin
    if not (14 <= height <= 30):
        return None

    lane_count = random.randint(3, 5)
    pads = [(random.randint(0, 2), random.randint(0, 2)) for _ in range(lane_count)]
    lane_widths = [sprite_w + left + right for left, right in pads]
    gaps_x = [random.randint(1, 4) for _ in range(lane_count - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    width = left_margin + sum(lane_widths) + sum(gaps_x) + right_margin
    if not (14 <= width <= 30):
        return None

    lane_lefts: list[int] = []
    sprite_lefts: list[int] = []
    center2: dict[int, int] = {}
    cursor = left_margin
    for idx, (lane_w, (pad_l, _pad_r)) in enumerate(zip(lane_widths, pads)):
        lane_lefts.append(cursor)
        sprite_lefts.append(cursor + pad_l)
        center2[idx] = 2 * cursor + lane_w - 1
        cursor += lane_w
        if idx + 1 < lane_count:
            cursor += gaps_x[idx]

    red_count = random.randint(1, min(2, lane_count - 1))
    indices = list(range(lane_count))
    random.shuffle(indices)
    red_indices = sorted(indices[:red_count])
    green_indices = sorted(indices[red_count:])

    for red_idx in red_indices:
        dists = sorted(abs(center2[red_idx] - center2[green_idx]) for green_idx in green_indices)
        if len(dists) >= 2 and dists[0] == dists[1]:
            return None

    palette = [c for c in range(10) if c not in (bg, RED, GREEN, GREY)]
    if len(palette) < lane_count:
        return None
    lane_colors = random.sample(palette, lane_count)

    red_counts = {
        idx: random.randint(1, min(3, slot_count - 2))
        for idx in red_indices
    }
    mapping = {idx: _nearest_green(center2, idx, green_indices) for idx in red_indices}
    gains = {idx: 0 for idx in green_indices}
    for red_idx, green_idx in mapping.items():
        gains[green_idx] += red_counts[red_idx]

    green_counts: dict[int, int] = {}
    for green_idx in green_indices:
        max_base = slot_count - gains[green_idx]
        if max_base < 2:
            return None
        green_counts[green_idx] = random.randint(2, max_base)

    slot_tops = [top_start + i * period for i in range(slot_count)]
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for lane_idx in range(lane_count):
        lane_left = lane_lefts[lane_idx]
        sprite_left = sprite_lefts[lane_idx]
        color = lane_colors[lane_idx]
        if lane_idx in red_indices:
            count = red_counts[lane_idx]
            _paint_stack(inp, lane_left, sprite_left, slot_tops, count, sprite_mask, color)
            _paint_stack(out, lane_left, sprite_left, slot_tops, count, sprite_mask, GREY)
            inp[marker_r0 : marker_r0 + marker_h, lane_left : lane_left + lane_widths[lane_idx]] = RED
        else:
            base = green_counts[lane_idx]
            total = base + gains[lane_idx]
            _paint_stack(inp, lane_left, sprite_left, slot_tops, base, sprite_mask, color)
            _paint_stack(out, lane_left, sprite_left, slot_tops, total, sprite_mask, color)
            inp[marker_r0 : marker_r0 + marker_h, lane_left : lane_left + lane_widths[lane_idx]] = GREEN

    counts = Counter(map(int, inp.ravel()))
    ranked = counts.most_common(2)
    if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
        return None
    if ranked[0][0] != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
