import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
RECT_COLORS = (1, 2, 3, 4, 6, 7)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_filled_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    return comp["area"] == block.size and bool(np.all(block == color))


def _analyze_scene(grid: np.ndarray) -> tuple[dict, list[dict], list[int]]:
    comps = _extract_components(grid)
    source_candidates = [comp for comp in comps if int(comp["color"]) == GREY and _is_filled_rectangle(grid, comp)]
    assert len(source_candidates) == 1, "expected one solid grey source rectangle"
    source = source_candidates[0]

    rectangles = [
        comp
        for comp in comps
        if int(comp["color"]) in RECT_COLORS and _is_filled_rectangle(grid, comp)
    ]
    assert 4 <= len(rectangles) <= 6, "expected 4-6 filled colored rectangles"

    coord_to_rect: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rectangles):
        for coord in rect["coords"]:
            coord_to_rect[coord] = idx

    r0, c0, r1, c1 = source["bbox"]
    reachable: set[int] = set()
    for r in range(r0, r1 + 1):
        for start_c, step in ((c0 - 1, -1), (c1 + 1, 1)):
            c = start_c
            while 0 <= c < grid.shape[1]:
                value = int(grid[r, c])
                if value in (BG, MAROON):
                    c += step
                    continue
                rect_id = coord_to_rect.get((r, c))
                if rect_id is not None:
                    reachable.add(rect_id)
                break

    ranked = sorted(
        reachable,
        key=lambda idx: (
            -(rectangles[idx]["bbox"][2] - rectangles[idx]["bbox"][0] + 1)
            * (rectangles[idx]["bbox"][3] - rectangles[idx]["bbox"][1] + 1),
            rectangles[idx]["bbox"][0],
            rectangles[idx]["bbox"][1],
        ),
    )
    return source, rectangles, ranked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep, recolor, or erase only the rectangles that are horizontally visible from the grey source block.

    Input structure:
    - The grid has a black background, one solid grey source rectangle, 4-6 solid colored rectangles, and a few maroon single-pixel distractors.
    - The colored rectangles are axis-aligned filled rectangles with distinct non-grey colors and do not overlap each other or the source.

    Rule:
    - Ignore all maroon pixels: they neither block rays nor survive in the output.
    - Cast horizontal rays left and right from each row of the grey source rectangle; along each row, the first non-background, non-maroon object hit is reachable.
    - Sort the reachable colored rectangles by decreasing area, breaking ties by smaller top-row coordinate.
    - Leave the first reachable rectangle unchanged, recolor the second reachable rectangle grey, erase every later reachable rectangle to black, and leave all unreachable colored rectangles and the source in place.

    Key insight:
    - Only rectangles that are actually seen by the source rays enter the area ranking, so a larger but occluded rectangle can remain untouched while a smaller visible one is modified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    _source, rectangles, ranked = _analyze_scene(g)

    out = g.copy()
    out[out == MAROON] = BG
    for order, rect_id in enumerate(ranked):
        r0, c0, r1, c1 = rectangles[rect_id]["bbox"]
        if order == 0:
            continue
        fill = GREY if order == 1 else BG
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _rect_from_bbox(name: str, color: int, r0: int, c0: int, h: int, w: int) -> dict:
    return {
        "name": name,
        "color": color,
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "area": h * w,
    }


def _interval_overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _can_place(rect: dict, occupied: list[tuple[int, int, int, int]], h: int, w: int, pad: int = 1) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    for or0, oc0, or1, oc1 in occupied:
        if _interval_overlaps(r0 - pad, r1 + pad, or0, or1) and _interval_overlaps(c0 - pad, c1 + pad, oc0, oc1):
            return False
    return True


def _paint_rect(grid: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])


def _visible_rect_ids_from_specs(source: dict, rects: list[dict]) -> list[int]:
    source_rows = range(source["r0"], source["r0"] + source["h"])
    visible: set[int] = set()
    for r in source_rows:
        candidates = [idx for idx, rect in enumerate(rects) if rect["r0"] <= r <= rect["r0"] + rect["h"] - 1]
        if not candidates:
            continue
        visible.add(min(candidates, key=lambda idx: rects[idx]["c0"]))
    return sorted(
        visible,
        key=lambda idx: (-rects[idx]["area"], rects[idx]["r0"], rects[idx]["c0"]),
    )


def _expected_output(grid: np.ndarray, rects: list[dict], ranked_visible: list[int]) -> np.ndarray:
    out = grid.copy()
    out[out == MAROON] = BG
    for order, idx in enumerate(ranked_visible):
        rect = rects[idx]
        r0, c0, r1, c1 = rect["bbox"]
        if order == 0:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY if order == 1 else BG
    return out


def _try_place_extra(
    name: str,
    color: int,
    source: dict,
    occupied: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> dict | None:
    zones = []
    if source["r0"] >= 4:
        zones.append(("top", 0, source["r0"] - 2))
    lower_start = source["r0"] + source["h"] + 1
    if lower_start <= grid_h - 2:
        zones.append(("bottom", lower_start, grid_h - 1))
    random.shuffle(zones)

    for _ in range(80):
        if not zones:
            return None
        _zone_name, row_lo, row_hi = random.choice(zones)
        rect_h = random.randint(2, min(5, row_hi - row_lo + 1))
        rect_w = random.randint(2, 4)
        r0 = random.randint(row_lo, row_hi - rect_h + 1)
        c0 = random.randint(2, grid_w - rect_w - 2)
        rect = _rect_from_bbox(name, color, r0, c0, rect_h, rect_w)
        if _can_place(rect, occupied, grid_h, grid_w):
            return rect
    return None


def _build_primary_rects(grid_h: int, grid_w: int, source: dict, colors: list[int]) -> list[dict] | None:
    hs = source["h"]
    s = source["r0"]
    primary_names = ["top", "bottom", "blocked"] if hs == 2 else ["top", "middle", "bottom", "blocked"]
    tie_branch = random.random() < 0.35
    tie_pair = None
    if tie_branch:
        pair_choices = [("top", "bottom")]
        if hs == 3:
            pair_choices.extend([("top", "middle"), ("middle", "bottom")])
        tie_pair = random.choice(pair_choices)

    dims: dict[str, tuple[int, int]] = {}
    dims["top"] = (random.randint(2, 3), random.randint(2, 4))
    dims["bottom"] = (random.randint(2, 3), random.randint(2, 4))
    if hs == 3:
        dims["middle"] = (random.randint(2, 3), random.randint(2, 4))

    if tie_pair is not None:
        area_options: dict[str, set[int]] = {}
        for name in tie_pair:
            h_options = [2, 3]
            areas = {hh * ww for hh in h_options for ww in range(2, 5)}
            area_options[name] = areas
        common_areas = sorted(set.intersection(*(area_options[name] for name in tie_pair)))
        tie_area = random.choice(common_areas)
        for name in tie_pair:
            choices = [(hh, ww) for hh in (2, 3) for ww in range(2, 5) if hh * ww == tie_area]
            dims[name] = random.choice(choices)

    if hs == 2:
        blocked_h = random.randint(2, 5)
    else:
        blocked_h = random.randint(3, 5)
    dims["blocked"] = (blocked_h, random.randint(2, 4))

    reachable_names = [name for name in primary_names if name != "blocked"]
    if random.random() < 0.7:
        winner = random.choice(reachable_names)
        winner_h = 3 if winner != "blocked" else dims[winner][0]
        if winner in dims:
            dims[winner] = (winner_h, 4)
        blocked_area_target = max(h * w for name, (h, w) in dims.items() if name != "blocked") + 1
        blocked_choices = [(hh, ww) for hh in range(dims["blocked"][0], 6) for ww in range(2, 5) if hh * ww >= blocked_area_target]
        if blocked_choices:
            dims["blocked"] = random.choice(blocked_choices)

    rect_specs: list[tuple[str, int, int]] = []
    top_h, top_w = dims["top"]
    rect_specs.append(("top", s - top_h + 1, top_h, top_w))
    if hs == 2:
        bottom_h, bottom_w = dims["bottom"]
        rect_specs.append(("bottom", s + 1, bottom_h, bottom_w))
    else:
        middle_h, middle_w = dims["middle"]
        rect_specs.append(("middle", s + 2 - middle_h, middle_h, middle_w))
        bottom_h, bottom_w = dims["bottom"]
        rect_specs.append(("bottom", s + 2, bottom_h, bottom_w))

    blocked_h, blocked_w = dims["blocked"]
    blocked_lo = max(0, s + hs - blocked_h)
    blocked_hi = min(s, grid_h - blocked_h)
    if blocked_lo > blocked_hi:
        return None
    rect_specs.append(("blocked", random.randint(blocked_lo, blocked_hi), blocked_h, blocked_w))

    widths = [spec[3] for spec in rect_specs]
    gaps = [random.randint(2, 4) for _ in range(len(rect_specs))]
    cur_c = source["c0"] + source["w"] + gaps[0]
    rects: list[dict] = []
    occupied = [(source["r0"], source["c0"], source["r0"] + source["h"] - 1, source["c0"] + source["w"] - 1)]
    for idx, (name, r0, h0, w0) in enumerate(rect_specs):
        rect = _rect_from_bbox(name, colors[idx], r0, cur_c, h0, w0)
        if not _can_place(rect, occupied, grid_h, grid_w):
            return None
        rects.append(rect)
        occupied.append(rect["bbox"])
        cur_c += w0 + (gaps[idx + 1] if idx + 1 < len(gaps) else random.randint(2, 4))

    return rects


def _semantic_checks(source: dict, rects: list[dict], expected_visible: list[int]) -> bool:
    overlap_rows = range(source["r0"], source["r0"] + source["h"])
    if source["h"] == 2:
        planned = {"top", "bottom"}
    else:
        planned = {"top", "middle", "bottom"}
    visible_names = {rects[idx]["name"] for idx in expected_visible}
    if visible_names != planned:
        return False

    blocked = next(rect for rect in rects if rect["name"] == "blocked")
    if not any(blocked["r0"] <= r <= blocked["r0"] + blocked["h"] - 1 for r in overlap_rows):
        return False
    if any(rects[idx]["name"] == "blocked" for idx in expected_visible):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 26)
    grid_w = random.randint(20, 29)
    source_h, source_w = random.choice(((2, 3), (3, 2), (3, 3)))
    source_h = 3 if random.random() < 0.65 else 2
    source_w = 3 if source_h == 2 else random.choice((2, 3))
    if grid_h < source_h + 5:
        return None

    source_r0 = random.randint(2, grid_h - source_h - 3)
    source = {"r0": source_r0, "c0": 1, "h": source_h, "w": source_w}

    rect_count = random.randint(4, 6)
    colors = random.sample(list(RECT_COLORS), rect_count)
    primary = _build_primary_rects(grid_h, grid_w, source, colors)
    if primary is None:
        return None

    extra_needed = rect_count - len(primary)
    rects = primary[:]
    occupied = [(source["r0"], source["c0"], source["r0"] + source["h"] - 1, source["c0"] + source["w"] - 1)]
    occupied.extend(rect["bbox"] for rect in rects)
    for idx in range(extra_needed):
        extra = _try_place_extra(f"extra{idx}", colors[len(rects)], source, occupied, grid_h, grid_w)
        if extra is None:
            return None
        rects.append(extra)
        occupied.append(extra["bbox"])

    expected_visible = _visible_rect_ids_from_specs(source, rects)
    if len(expected_visible) < 2:
        return None
    if not _semantic_checks(source, rects, expected_visible):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[source["r0"] : source["r0"] + source["h"], source["c0"] : source["c0"] + source["w"]] = GREY
    for rect in rects:
        _paint_rect(grid, rect)

    maroon_count = random.randint(2, 4)
    maroons: set[tuple[int, int]] = set()
    source_end = source["c0"] + source["w"] - 1
    path_rows = [source["r0"] + i for i in range(source["h"])]
    if random.random() < 0.8:
        for r in path_rows:
            hits = [rect for rect in rects if rect["r0"] <= r <= rect["r0"] + rect["h"] - 1]
            if not hits:
                continue
            first = min(hits, key=lambda rect: rect["c0"])
            if first["c0"] - source_end >= 3:
                c = random.randint(source_end + 1, first["c0"] - 1)
                maroons.add((r, c))
                break
    for _ in range(80):
        if len(maroons) >= maroon_count:
            break
        r = random.randint(0, grid_h - 1)
        c = random.randint(0, grid_w - 1)
        if grid[r, c] != BG or (r, c) in maroons:
            continue
        maroons.add((r, c))
    if len(maroons) != maroon_count:
        return None
    for r, c in maroons:
        grid[r, c] = MAROON

    expected = _expected_output(grid, rects, expected_visible)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
        expected = np.fliplr(expected)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    source_comp, out_rects, ranked = _analyze_scene(grid)
    if not (2 <= len(ranked) <= 3):
        return None
    if source_comp["bbox"][1] not in (1, grid.shape[1] - source_w - 1):
        return None
    if Counter(map(int, grid.flat))[MAROON] != maroon_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
