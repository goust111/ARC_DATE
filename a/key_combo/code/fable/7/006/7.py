import random
from collections import Counter, deque

import numpy as np


MAX_GEN_ATTEMPTS = 500
BG = 0
PALETTE = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _rectangles_from_grid(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    rectangles = []
    non_rectangles = []
    for comp in _find_monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        rect_h = int(r1 - r0 + 1)
        rect_w = int(c1 - c0 + 1)
        is_rect = rect_h >= 2 and rect_w >= 2 and int(comp["area"]) == rect_h * rect_w
        target = rectangles if is_rect else non_rectangles
        target.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": rect_h,
                "w": rect_w,
                "area": int(comp["area"]),
                "coords": comp["coords"],
            }
        )
    return rectangles, non_rectangles


def _draw_rect(out: np.ndarray, rect: dict) -> None:
    out[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])


def _can_place_rect(out: np.ndarray, rect: dict, r0: int, c0: int, bg: int) -> bool:
    h, w = int(rect["h"]), int(rect["w"])
    hh, ww = map(int, out.shape)
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    return bool(np.all(out[r0 : r0 + h, c0 : c0 + w] == bg))


def _classify_mover(rect: dict, magnet: dict) -> dict:
    row_band = _intervals_overlap(rect["r0"], rect["r1"], magnet["r0"], magnet["r1"])
    col_band = _intervals_overlap(rect["c0"], rect["c1"], magnet["c0"], magnet["c1"])
    assert row_band ^ col_band, "each smaller rectangle must align with exactly one magnet band"

    same = int(rect["color"]) == int(magnet["color"])
    relation = "same" if same else "diff"
    mover = dict(rect)
    mover["relation"] = relation

    if row_band:
        mover["axis"] = "h"
        if rect["c1"] < magnet["c0"]:
            mover["side"] = "left"
            mover["distance"] = int(magnet["c0"] - rect["c1"] - 1)
            mover["dr"], mover["dc"] = (0, 1) if same else (0, -1)
        else:
            assert rect["c0"] > magnet["c1"]
            mover["side"] = "right"
            mover["distance"] = int(rect["c0"] - magnet["c1"] - 1)
            mover["dr"], mover["dc"] = (0, -1) if same else (0, 1)
    else:
        mover["axis"] = "v"
        if rect["r1"] < magnet["r0"]:
            mover["side"] = "top"
            mover["distance"] = int(magnet["r0"] - rect["r1"] - 1)
            mover["dr"], mover["dc"] = (1, 0) if same else (-1, 0)
        else:
            assert rect["r0"] > magnet["r1"]
            mover["side"] = "bottom"
            mover["distance"] = int(rect["r0"] - magnet["r1"] - 1)
            mover["dr"], mover["dc"] = (-1, 0) if same else (1, 0)

    return mover


def _settle_mover(out: np.ndarray, mover: dict, bg: int) -> dict:
    r0 = int(mover["r0"])
    c0 = int(mover["c0"])
    dr = int(mover["dr"])
    dc = int(mover["dc"])
    while _can_place_rect(out, mover, r0 + dr, c0 + dc, bg):
        r0 += dr
        c0 += dc
    placed = dict(mover)
    placed["r0"] = r0
    placed["c0"] = c0
    placed["r1"] = int(r0 + mover["h"] - 1)
    placed["c1"] = int(c0 + mover["w"] - 1)
    _draw_rect(out, placed)
    return placed


def _solve_from_rectangles(shape: tuple[int, int], bg: int, rectangles: list[dict]) -> np.ndarray:
    areas = [int(rect["area"]) for rect in rectangles]
    max_area = max(areas)
    magnets = [rect for rect in rectangles if int(rect["area"]) == max_area]
    assert len(magnets) == 1, "expected a unique largest rectangle"
    magnet = dict(magnets[0])

    movers = [_classify_mover(rect, magnet) for rect in rectangles if rect is not magnets[0]]
    movers.sort(
        key=lambda rect: (
            rect["axis"],
            rect["side"],
            rect["dr"],
            rect["dc"],
            rect["distance"],
            rect["r0"],
            rect["c0"],
            rect["h"],
            rect["w"],
            rect["color"],
        )
    )

    out = np.full(shape, bg, dtype=int)
    _draw_rect(out, magnet)
    for mover in movers:
        _settle_mover(out, mover, bg)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Delete non-rectangular noise and slide band-aligned small rectangles toward or away from the unique largest magnet rectangle.

    Input structure:
    - The background is black with scattered colored noise pixels, short 1xn or nx1 noise lines,
      and several solid monochromatic rectangles whose side lengths are at least 2.
    - Exactly one rectangle has unique maximum area and stays fixed as the magnet.
    - Every smaller rectangle overlaps exactly one of the magnet's row band or column band.
    - Some smaller rectangles share the magnet color and some use different colors.

    Rule:
    - Ignore every foreground component that is not a solid axis-aligned rectangle with both sides at least 2.
    - Keep the unique largest rectangle fixed.
    - A same-color smaller rectangle slides along its aligned band toward the magnet until the next step would collide.
    - A different-color smaller rectangle slides along the same axis away from the magnet until the next step would collide with the edge or an earlier-settled rectangle.
    - Rectangles moving in the same direction settle from nearest to farthest, so they stack without crossing.

    Key insight:
    - The magnet is identified by unique maximum rectangle area, while deceptive 1xn lines fail the rectangle test; movement axis comes from row-band versus column-band overlap, and movement direction comes from color equality with the magnet.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    rectangles, _non_rectangles = _rectangles_from_grid(grid, bg)
    assert rectangles, "expected at least one valid rectangle"
    return _solve_from_rectangles(grid.shape, bg, rectangles)


def _pick_disjoint_intervals(length: int, count: int) -> list[tuple[int, int]]:
    candidates = []
    for seg_len in (2, 3):
        if seg_len > length:
            continue
        for start in range(length - seg_len + 1):
            candidates.append((start, start + seg_len - 1))
    random.shuffle(candidates)
    chosen = []
    for start, end in candidates:
        if all(end + 1 < other_start or other_end + 1 < start for other_start, other_end in chosen):
            chosen.append((start, end))
            if len(chosen) == count:
                return sorted(chosen)
    raise ValueError("could not pick enough disjoint intervals")


def _margin_for_side(shape: tuple[int, int], magnet: dict, axis: str, side: str) -> int:
    h, w = map(int, shape)
    if axis == "h":
        return int(magnet["c0"]) if side == "left" else int(w - magnet["c1"] - 1)
    return int(magnet["r0"]) if side == "top" else int(h - magnet["r1"] - 1)


def _spawn_group(group: dict, magnet: dict, shape: tuple[int, int]) -> list[dict]:
    count = int(group["count"])
    axis = str(group["axis"])
    side = str(group["side"])
    relation = str(group["relation"])
    margin = _margin_for_side(shape, magnet, axis, side)
    colors = [int(magnet["color"])] * count
    if relation == "diff":
        diff_colors = [c for c in PALETTE if c != int(magnet["color"])]
        random.shuffle(diff_colors)
        colors = diff_colors[:count]

    for _ in range(200):
        if axis == "h":
            lane_r0, lane_r1 = group["lane"]
            lane_h = int(lane_r1 - lane_r0 + 1)
            widths = [random.randint(2, 3 if count >= 3 else 4) for _ in range(count)]
            gap0 = random.randint(1, min(3, margin))
            gaps = [random.randint(1, 2) for _ in range(count - 1)]
            total = gap0 + sum(widths) + sum(gaps)
            if total > margin:
                continue
            rects = []
            if side == "left":
                right = int(magnet["c0"] - gap0 - 1)
                for idx, width in enumerate(widths):
                    c1 = right
                    c0 = int(c1 - width + 1)
                    rects.append(
                        {
                            "group_id": group["group_id"],
                            "order": idx,
                            "axis": axis,
                            "side": side,
                            "relation": relation,
                            "color": int(colors[idx]),
                            "r0": int(lane_r0),
                            "r1": int(lane_r1),
                            "c0": c0,
                            "c1": c1,
                            "h": lane_h,
                            "w": width,
                            "area": lane_h * width,
                        }
                    )
                    if idx + 1 < count:
                        right = int(c0 - gaps[idx] - 1)
            else:
                left = int(magnet["c1"] + gap0 + 1)
                for idx, width in enumerate(widths):
                    c0 = left
                    c1 = int(c0 + width - 1)
                    rects.append(
                        {
                            "group_id": group["group_id"],
                            "order": idx,
                            "axis": axis,
                            "side": side,
                            "relation": relation,
                            "color": int(colors[idx]),
                            "r0": int(lane_r0),
                            "r1": int(lane_r1),
                            "c0": c0,
                            "c1": c1,
                            "h": lane_h,
                            "w": width,
                            "area": lane_h * width,
                        }
                    )
                    if idx + 1 < count:
                        left = int(c1 + gaps[idx] + 1)
            return rects

        lane_c0, lane_c1 = group["lane"]
        lane_w = int(lane_c1 - lane_c0 + 1)
        heights = [random.randint(2, 3 if count >= 3 else 4) for _ in range(count)]
        gap0 = random.randint(1, min(3, margin))
        gaps = [random.randint(1, 2) for _ in range(count - 1)]
        total = gap0 + sum(heights) + sum(gaps)
        if total > margin:
            continue
        rects = []
        if side == "top":
            bottom = int(magnet["r0"] - gap0 - 1)
            for idx, height in enumerate(heights):
                r1 = bottom
                r0 = int(r1 - height + 1)
                rects.append(
                    {
                        "group_id": group["group_id"],
                        "order": idx,
                        "axis": axis,
                        "side": side,
                        "relation": relation,
                        "color": int(colors[idx]),
                        "r0": r0,
                        "r1": r1,
                        "c0": int(lane_c0),
                        "c1": int(lane_c1),
                        "h": height,
                        "w": lane_w,
                        "area": height * lane_w,
                    }
                )
                if idx + 1 < count:
                    bottom = int(r0 - gaps[idx] - 1)
        else:
            top = int(magnet["r1"] + gap0 + 1)
            for idx, height in enumerate(heights):
                r0 = top
                r1 = int(r0 + height - 1)
                rects.append(
                    {
                        "group_id": group["group_id"],
                        "order": idx,
                        "axis": axis,
                        "side": side,
                        "relation": relation,
                        "color": int(colors[idx]),
                        "r0": r0,
                        "r1": r1,
                        "c0": int(lane_c0),
                        "c1": int(lane_c1),
                        "h": height,
                        "w": lane_w,
                        "area": height * lane_w,
                    }
                )
                if idx + 1 < count:
                    top = int(r1 + gaps[idx] + 1)
        return rects
    raise ValueError("could not place group")


def _oracle_output(shape: tuple[int, int], magnet: dict, movers: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    _draw_rect(out, magnet)

    by_group: dict[str, list[dict]] = {}
    for mover in movers:
        by_group.setdefault(str(mover["group_id"]), []).append(mover)

    h, w = map(int, shape)
    for group_id in sorted(by_group):
        items = sorted(by_group[group_id], key=lambda item: int(item["order"]))
        sample = items[0]
        axis = str(sample["axis"])
        side = str(sample["side"])
        relation = str(sample["relation"])

        if axis == "h":
            if relation == "same":
                if side == "left":
                    cursor = int(magnet["c0"] - 1)
                    for item in items:
                        c1 = cursor
                        c0 = int(c1 - item["w"] + 1)
                        placed = dict(item)
                        placed["c0"], placed["c1"] = c0, c1
                        _draw_rect(out, placed)
                        cursor = int(c0 - 1)
                else:
                    cursor = int(magnet["c1"] + 1)
                    for item in items:
                        c0 = cursor
                        c1 = int(c0 + item["w"] - 1)
                        placed = dict(item)
                        placed["c0"], placed["c1"] = c0, c1
                        _draw_rect(out, placed)
                        cursor = int(c1 + 1)
            else:
                if side == "left":
                    cursor = 0
                    for item in items:
                        c0 = cursor
                        c1 = int(c0 + item["w"] - 1)
                        placed = dict(item)
                        placed["c0"], placed["c1"] = c0, c1
                        _draw_rect(out, placed)
                        cursor = int(c1 + 1)
                else:
                    cursor = int(w - 1)
                    for item in items:
                        c1 = cursor
                        c0 = int(c1 - item["w"] + 1)
                        placed = dict(item)
                        placed["c0"], placed["c1"] = c0, c1
                        _draw_rect(out, placed)
                        cursor = int(c0 - 1)
            continue

        if relation == "same":
            if side == "top":
                cursor = int(magnet["r0"] - 1)
                for item in items:
                    r1 = cursor
                    r0 = int(r1 - item["h"] + 1)
                    placed = dict(item)
                    placed["r0"], placed["r1"] = r0, r1
                    _draw_rect(out, placed)
                    cursor = int(r0 - 1)
            else:
                cursor = int(magnet["r1"] + 1)
                for item in items:
                    r0 = cursor
                    r1 = int(r0 + item["h"] - 1)
                    placed = dict(item)
                    placed["r0"], placed["r1"] = r0, r1
                    _draw_rect(out, placed)
                    cursor = int(r1 + 1)
        else:
            if side == "top":
                cursor = 0
                for item in items:
                    r0 = cursor
                    r1 = int(r0 + item["h"] - 1)
                    placed = dict(item)
                    placed["r0"], placed["r1"] = r0, r1
                    _draw_rect(out, placed)
                    cursor = int(r1 + 1)
            else:
                cursor = int(h - 1)
                for item in items:
                    r1 = cursor
                    r0 = int(r1 - item["h"] + 1)
                    placed = dict(item)
                    placed["r0"], placed["r1"] = r0, r1
                    _draw_rect(out, placed)
                    cursor = int(r0 - 1)
    return out


def _clear_with_padding(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_noise(grid: np.ndarray, occupied: np.ndarray, count: int, *, line: bool) -> int:
    placed = 0
    h, w = map(int, grid.shape)
    for _ in range(count * 30):
        if placed >= count:
            break
        if line:
            length = random.randint(2, 5)
            if random.random() < 0.5:
                rr_h, rr_w = 1, length
            else:
                rr_h, rr_w = length, 1
        else:
            rr_h, rr_w = 1, 1
        candidates = []
        for r0 in range(h - rr_h + 1):
            for c0 in range(w - rr_w + 1):
                if _clear_with_padding(occupied, r0, c0, rr_h, rr_w, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            break
        r0, c0 = random.choice(candidates)
        color = random.choice(PALETTE)
        grid[r0 : r0 + rr_h, c0 : c0 + rr_w] = color
        _mark_occupied(occupied, r0, c0, rr_h, rr_w)
        placed += 1
    return placed


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            magnet_h = random.randint(6, 8)
            magnet_w = random.randint(6, 8)
            top_margin = random.randint(5, 10)
            bottom_margin = random.randint(5, 10)
            left_margin = random.randint(5, 10)
            right_margin = random.randint(5, 10)
            height = top_margin + magnet_h + bottom_margin
            width = left_margin + magnet_w + right_margin
            if height > 30 or width > 30:
                continue

            magnet_color = random.choice(PALETTE)
            magnet = {
                "color": magnet_color,
                "r0": top_margin,
                "c0": left_margin,
                "r1": top_margin + magnet_h - 1,
                "c1": left_margin + magnet_w - 1,
                "h": magnet_h,
                "w": magnet_w,
                "area": magnet_h * magnet_w,
            }

            h_lanes_rel = _pick_disjoint_intervals(magnet_h, 2)
            v_lanes_rel = _pick_disjoint_intervals(magnet_w, 2)
            h_lanes = [(magnet["r0"] + a, magnet["r0"] + b) for a, b in h_lanes_rel]
            v_lanes = [(magnet["c0"] + a, magnet["c0"] + b) for a, b in v_lanes_rel]

            h_side = random.choice(["left", "right"])
            v_side = random.choice(["top", "bottom"])
            stack_axis = random.choice(["h", "v"])

            if stack_axis == "h":
                stack_margin = _margin_for_side((height, width), magnet, "h", h_side)
                stack_count = 3 if stack_margin >= 9 and random.random() < 0.5 else 2
                groups = [
                    {"group_id": "stack_h", "axis": "h", "lane": h_lanes[0], "side": h_side, "relation": "diff", "count": stack_count},
                    {"group_id": "same_h", "axis": "h", "lane": h_lanes[1], "side": "right" if h_side == "left" else "left", "relation": "same", "count": 1},
                    {"group_id": "same_v", "axis": "v", "lane": v_lanes[0], "side": v_side, "relation": "same", "count": 1},
                    {"group_id": "diff_v", "axis": "v", "lane": v_lanes[1], "side": "bottom" if v_side == "top" else "top", "relation": "diff", "count": 1},
                ]
            else:
                stack_margin = _margin_for_side((height, width), magnet, "v", v_side)
                stack_count = 3 if stack_margin >= 9 and random.random() < 0.5 else 2
                groups = [
                    {"group_id": "stack_v", "axis": "v", "lane": v_lanes[0], "side": v_side, "relation": "diff", "count": stack_count},
                    {"group_id": "same_v", "axis": "v", "lane": v_lanes[1], "side": "bottom" if v_side == "top" else "top", "relation": "same", "count": 1},
                    {"group_id": "same_h", "axis": "h", "lane": h_lanes[0], "side": h_side, "relation": "same", "count": 1},
                    {"group_id": "diff_h", "axis": "h", "lane": h_lanes[1], "side": "right" if h_side == "left" else "left", "relation": "diff", "count": 1},
                ]

            if random.random() < 0.45:
                if stack_axis == "h":
                    groups.append(
                        {
                            "group_id": "extra_v",
                            "axis": "v",
                            "lane": random.choice(v_lanes),
                            "side": random.choice(["top", "bottom"]),
                            "relation": random.choice(["same", "diff"]),
                            "count": 1,
                        }
                    )
                else:
                    groups.append(
                        {
                            "group_id": "extra_h",
                            "axis": "h",
                            "lane": random.choice(h_lanes),
                            "side": random.choice(["left", "right"]),
                            "relation": random.choice(["same", "diff"]),
                            "count": 1,
                        }
                    )

            movers = []
            for group in groups:
                movers.extend(_spawn_group(group, magnet, (height, width)))

            grid = np.full((height, width), BG, dtype=int)
            _draw_rect(grid, magnet)
            for mover in movers:
                _draw_rect(grid, mover)

            occupied = grid != BG
            line_count = _place_noise(grid, occupied, random.randint(1, 3), line=True)
            pixel_count = _place_noise(grid, occupied, random.randint(2, 5), line=False)
            if line_count < 1 or pixel_count < 2:
                continue

            expected = _oracle_output((height, width), magnet, movers, BG)
            meta = {
                "magnet": magnet,
                "movers": movers,
                "groups": groups,
                "line_count": line_count,
                "pixel_count": pixel_count,
            }
            _validate_case(grid, expected, meta)
            return grid, expected, meta
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to build a valid case")


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert isinstance(grid, np.ndarray) and isinstance(expected, np.ndarray)
    assert grid.ndim == 2 and expected.ndim == 2
    assert max(grid.shape) <= 30 and max(expected.shape) <= 30
    assert not np.array_equal(grid, expected), "degenerate case: nothing changes"

    bg = get_bg(grid)
    rectangles, non_rectangles = _rectangles_from_grid(grid, bg)
    assert len(non_rectangles) >= 3, "need visible non-rectangular noise"

    areas = [int(rect["area"]) for rect in rectangles]
    assert areas.count(max(areas)) == 1, "largest rectangle must be unique"
    assert len(rectangles) == 1 + len(meta["movers"]), "valid rectangles should match planted rectangles"

    magnet = max(rectangles, key=lambda rect: int(rect["area"]))
    axes = set()
    relations = set()
    distances = []
    for rect in rectangles:
        if rect["area"] == magnet["area"] and rect["bbox"] == magnet["bbox"]:
            continue
        mover = _classify_mover(rect, magnet)
        axes.add(str(mover["axis"]))
        relations.add(str(mover["relation"]))
        distances.append(int(mover["distance"]))
    assert axes == {"h", "v"}, "need both row-band and column-band movers"
    assert relations == {"same", "diff"}, "need both attraction and repulsion"
    assert max(distances) >= 2, "need at least one clearly moving rectangle"
    assert any(int(group["count"]) >= 2 for group in meta["groups"]), "need a stacking example"

    solved = main(grid)
    assert np.array_equal(solved, expected), "solver must match planted semantics"


def generate_input() -> np.ndarray:
    return _build_case()[0]
