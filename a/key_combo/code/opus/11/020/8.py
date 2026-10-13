from __future__ import annotations

import numpy as np


GREY = 5
BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

AREA_TO_SHAPES = {
    6: ((2, 3), (3, 2)),
    8: ((2, 4), (4, 2)),
    9: ((3, 3),),
    12: ((3, 4), (4, 3)),
    16: ((4, 4),),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_offset(height: int, width: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, (width - 1) // 2
    if direction == "down":
        return height - 1, (width - 1) // 2
    if direction == "left":
        return (height - 1) // 2, 0
    if direction == "right":
        return (height - 1) // 2, width - 1
    raise ValueError(direction)


def _shape_options(area: int, direction: str) -> list[tuple[int, int]]:
    out = []
    for height, width in AREA_TO_SHAPES[area]:
        if direction in ("up", "down") and width <= 2:
            continue
        if direction in ("left", "right") and height <= 2:
            continue
        out.append((height, width))
    return out


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for idx, coords in enumerate(_components(grid != BACKGROUND)):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        height, width = sub.shape
        assert np.all(sub != BACKGROUND), "every object must be a solid rectangle with one grey marker"

        marker_coords = np.argwhere(sub == GREY)
        assert marker_coords.shape[0] == 1, "each rectangle must contain exactly one grey marker"
        mr, mc = map(int, marker_coords[0])

        colors = {int(v) for v in np.unique(sub)}
        colors.discard(GREY)
        assert len(colors) == 1, "each rectangle must use one foreground color plus grey"
        color = next(iter(colors))

        direction_candidates = []
        if mr == 0 and 0 < mc < width - 1:
            direction_candidates.append("up")
        if mr == height - 1 and 0 < mc < width - 1:
            direction_candidates.append("down")
        if mc == 0 and 0 < mr < height - 1:
            direction_candidates.append("left")
        if mc == width - 1 and 0 < mr < height - 1:
            direction_candidates.append("right")
        assert len(direction_candidates) == 1, "grey marker must lie on a non-corner edge midpoint"

        rects.append(
            {
                "id": idx,
                "color": int(color),
                "direction": direction_candidates[0],
                "marker": (mr, mc),
                "r0": r0,
                "c0": c0,
                "h": height,
                "w": width,
                "area": height * width,
            }
        )
    return rects


def _slide_rectangles(rects: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    placed = []
    order = sorted(rects, key=lambda rect: (-rect["area"], rect["r0"], rect["c0"], rect["id"]))
    for rect in order:
        rr = rect["r0"]
        cc = rect["c0"]
        dr, dc = DIRS[rect["direction"]]
        while True:
            nr = rr + dr
            nc = cc + dc
            if nr < 0 or nc < 0 or nr + rect["h"] > h or nc + rect["w"] > w:
                break
            if occupied[nr : nr + rect["h"], nc : nc + rect["w"]].any():
                break
            rr, cc = nr, nc
        moved = dict(rect)
        moved["r0"] = rr
        moved["c0"] = cc
        occupied[rr : rr + rect["h"], cc : cc + rect["w"]] = True
        placed.append(moved)
    return sorted(placed, key=lambda rect: rect["id"])


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _pair_records(rects: list[dict]) -> list[dict]:
    pairs = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            a_r0, a_c0 = a["r0"], a["c0"]
            a_r1, a_c1 = a_r0 + a["h"] - 1, a_c0 + a["w"] - 1
            b_r0, b_c0 = b["r0"], b["c0"]
            b_r1, b_c1 = b_r0 + b["h"] - 1, b_c0 + b["w"] - 1

            if a_c1 < b_c0:
                left, right = a, b
                gap0, gap1 = a_c1 + 1, b_c0 - 1
            elif b_c1 < a_c0:
                left, right = b, a
                gap0, gap1 = b_c1 + 1, a_c0 - 1
            else:
                continue
            if gap0 > gap1:
                continue

            row0 = max(left["r0"], right["r0"])
            row1 = min(left["r0"] + left["h"] - 1, right["r0"] + right["h"] - 1)
            if row0 > row1:
                continue

            blocked = False
            for k, blocker in enumerate(rects):
                if k in (i, j):
                    continue
                br0, bc0 = blocker["r0"], blocker["c0"]
                br1 = br0 + blocker["h"] - 1
                bc1 = bc0 + blocker["w"] - 1
                if _ranges_overlap(row0, row1, br0, br1) and _ranges_overlap(gap0, gap1, bc0, bc1):
                    blocked = True
                    break

            if left["area"] > right["area"]:
                fill_color = left["color"]
            elif right["area"] > left["area"]:
                fill_color = right["color"]
            else:
                fill_color = left["color"]

            pairs.append(
                {
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "row0": row0,
                    "row1": row1,
                    "gap0": gap0,
                    "gap1": gap1,
                    "blocked": blocked,
                    "fill_color": fill_color,
                }
            )
    return pairs


def _render_rectangles(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0 = rect["r0"], rect["c0"]
        grid[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]
        mr, mc = rect["marker"]
        grid[r0 + mr, c0 + mc] = GREY
    return grid


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    rects = _extract_rectangles(grid)
    final_rects = _slide_rectangles(rects, grid.shape)
    out = _render_rectangles(final_rects, grid.shape)
    pairs = _pair_records(final_rects)
    for pair in pairs:
        if pair["blocked"]:
            continue
        out[pair["row0"] : pair["row1"] + 1, pair["gap0"] : pair["gap1"] + 1] = pair["fill_color"]
    return out, final_rects, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the marked rectangles to their stopping positions, then fill unobstructed horizontal gaps by the larger neighbor's color.

    Input structure:
    - The grid is black background with 4-5 non-overlapping solid rectangles in distinct colors, each containing one grey marker pixel.
    - Every grey marker sits on a non-corner edge midpoint of its rectangle and indicates the rectangle's sliding direction.
    - Rectangles are initially separated so they can slide rigidly without changing shape or losing the grey marker.

    Rule:
    - Sort rectangles by area from large to small, and slide each one stepwise in its marker direction until the next step would leave the grid or overlap a previously stopped rectangle.
    - After all sliding finishes, inspect every horizontally separated rectangle pair whose row spans overlap; the shared-row strip between their bounding boxes is that pair's gap region.
    - If any third rectangle's bounding box overlaps both that row strip and the gap columns, leave the pair unfilled; otherwise fill the whole gap with the color of the larger of the two rectangles, keeping all rectangles and grey markers unchanged.

    Key insight:
    - The gap colors and even which gaps exist depend on the post-slide layout, so the solver must first simulate the ordered motion and only then apply the horizontal-gap blocking test.
    """
    arr = np.asarray(grid, dtype=int)
    assert arr.ndim == 2
    out, _final_rects, _pairs = _solve(arr)
    return out


def _pair_row_candidates(h1: int, h2: int, band_start: int, band_end: int) -> list[tuple[int, int]]:
    candidates = []
    for r0a in range(band_start, band_end - h1 + 2):
        for r0b in range(band_start, band_end - h2 + 2):
            overlap = min(r0a + h1 - 1, r0b + h2 - 1) - max(r0a, r0b) + 1
            if 1 <= overlap <= min(3, h1, h2):
                candidates.append((r0a, r0b))
    return candidates


def _pick_pair_rows(
    h1: int,
    h2: int,
    band_start: int,
    band_end: int,
    blocker_rows: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    candidates = _pair_row_candidates(h1, h2, band_start, band_end)
    if blocker_rows is not None:
        br0, br1 = blocker_rows
        filtered = []
        for r0a, r0b in candidates:
            row0 = max(r0a, r0b)
            row1 = min(r0a + h1 - 1, r0b + h2 - 1)
            if _ranges_overlap(row0, row1, br0, br1):
                filtered.append((r0a, r0b))
        candidates = filtered
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _choose_shape(area: int, direction: str) -> tuple[int, int] | None:
    options = _shape_options(area, direction)
    if not options:
        return None
    return options[int(np.random.randint(len(options)))]


def _final_signature(rects: list[dict]) -> tuple[tuple[int, int, int, int, int, str], ...]:
    return tuple(
        sorted(
            (
                rect["color"],
                rect["r0"],
                rect["c0"],
                rect["h"],
                rect["w"],
                rect["direction"],
            )
            for rect in rects
        )
    )


def _build_input_from_final(final_rects: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    for _ in range(80):
        shifted = []
        ok = True
        for rect in final_rects:
            moved = dict(rect)
            if rect["direction"] == "left":
                max_shift = min(4, w - rect["w"])
                if max_shift < 1:
                    ok = False
                    break
                moved["c0"] = int(np.random.randint(1, max_shift + 1))
            elif rect["direction"] == "right":
                max_shift = min(4, w - rect["w"])
                if max_shift < 1:
                    ok = False
                    break
                moved["c0"] = w - rect["w"] - int(np.random.randint(1, max_shift + 1))
            elif rect["direction"] == "up":
                max_shift = min(4, h - rect["h"])
                if max_shift < 1:
                    ok = False
                    break
                moved["r0"] = int(np.random.randint(1, max_shift + 1))
            else:
                max_shift = min(4, h - rect["h"])
                if max_shift < 1:
                    ok = False
                    break
                moved["r0"] = h - rect["h"] - int(np.random.randint(1, max_shift + 1))
            shifted.append(moved)
        if not ok:
            continue

        canvas = np.zeros(shape, dtype=bool)
        for rect in shifted:
            r0, c0 = rect["r0"], rect["c0"]
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + rect["h"] + 1)
            cc1 = min(w, c0 + rect["w"] + 1)
            footprint = canvas[rr0:rr1, cc0:cc1]
            if footprint.any():
                ok = False
                break
            canvas[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = True
        if not ok:
            continue

        inp = _render_rectangles(shifted, shape)
        if len(_components(inp != BACKGROUND)) != len(final_rects):
            continue
        _out, solved_final, pairs = _solve(inp)
        if _final_signature(solved_final) != _final_signature(final_rects):
            continue
        if not any(not pair["blocked"] for pair in pairs):
            continue
        return inp
    return None


def _make_fillable_layout() -> list[dict] | None:
    h = int(np.random.randint(12, 16))
    w = int(np.random.randint(14, 19))

    areas = [int(a) for a in np.random.choice(np.array(list(AREA_TO_SHAPES), dtype=int), 4, replace=False)]
    colors = [int(c) for c in np.random.permutation(np.array(PALETTE, dtype=int))[:4]]
    np.random.shuffle(areas)

    roles = ("left", "right", "left", "right")
    rect_specs = []
    for area, color, direction in zip(areas, colors, roles):
        shape = _choose_shape(area, direction)
        if shape is None:
            return None
        rect_specs.append({"area": area, "color": color, "direction": direction, "h": shape[0], "w": shape[1]})

    top_rows = _pick_pair_rows(rect_specs[0]["h"], rect_specs[1]["h"], 0, 4)
    bottom_rows = _pick_pair_rows(rect_specs[2]["h"], rect_specs[3]["h"], h - 5, h - 1)
    if top_rows is None or bottom_rows is None:
        return None

    final_rects = []
    for idx, spec in enumerate(rect_specs):
        rect = dict(spec)
        rect["id"] = idx
        rect["marker"] = _marker_offset(rect["h"], rect["w"], rect["direction"])
        if idx == 0:
            rect["r0"], rect["c0"] = top_rows[0], 0
        elif idx == 1:
            rect["r0"], rect["c0"] = top_rows[1], w - rect["w"]
        elif idx == 2:
            rect["r0"], rect["c0"] = bottom_rows[0], 0
        else:
            rect["r0"], rect["c0"] = bottom_rows[1], w - rect["w"]
        final_rects.append(rect)

    pairs = _pair_records(sorted(final_rects, key=lambda rect: rect["id"]))
    unblocked = sum(not pair["blocked"] for pair in pairs)
    if unblocked < 2 or any(pair["blocked"] for pair in pairs):
        return None

    inp = _build_input_from_final(final_rects, (h, w))
    if inp is None:
        return None
    return [dict(rect) for rect in _slide_rectangles(_extract_rectangles(inp), inp.shape)]


def _make_mixed_layout() -> list[dict] | None:
    h = int(np.random.randint(12, 16))
    w = int(np.random.randint(14, 19))
    blocked_at_top = bool(np.random.randint(0, 2))
    blocker_direction = "up" if blocked_at_top else "down"

    areas = list(np.random.permutation(np.array(list(AREA_TO_SHAPES), dtype=int)))
    colors = [int(c) for c in np.random.permutation(np.array(PALETTE, dtype=int))]
    roles = ("left", "right", "left", "right", blocker_direction)

    rect_specs = []
    for area, color, direction in zip(areas, colors, roles):
        shape = _choose_shape(int(area), direction)
        if shape is None:
            return None
        rect_specs.append(
            {"area": int(area), "color": color, "direction": direction, "h": shape[0], "w": shape[1]}
        )

    blocker_rows = (0, rect_specs[4]["h"] - 1) if blocked_at_top else (h - rect_specs[4]["h"], h - 1)
    blocked_band = (0, 4) if blocked_at_top else (h - 5, h - 1)
    fill_band = (h - 5, h - 1) if blocked_at_top else (0, 4)

    blocked_rows = _pick_pair_rows(
        rect_specs[0]["h"], rect_specs[1]["h"], blocked_band[0], blocked_band[1], blocker_rows=blocker_rows
    )
    fill_rows = _pick_pair_rows(rect_specs[2]["h"], rect_specs[3]["h"], fill_band[0], fill_band[1])
    if blocked_rows is None or fill_rows is None:
        return None

    final_rects = []
    for idx, spec in enumerate(rect_specs):
        rect = dict(spec)
        rect["id"] = idx
        rect["marker"] = _marker_offset(rect["h"], rect["w"], rect["direction"])
        if idx == 0:
            rect["r0"], rect["c0"] = blocked_rows[0], 0
        elif idx == 1:
            rect["r0"], rect["c0"] = blocked_rows[1], w - rect["w"]
        elif idx == 2:
            rect["r0"], rect["c0"] = fill_rows[0], 0
        elif idx == 3:
            rect["r0"], rect["c0"] = fill_rows[1], w - rect["w"]
        elif blocked_at_top:
            rect["r0"] = 0
            gap0 = final_rects[0]["w"]
            gap1 = w - final_rects[1]["w"] - 1
            if gap1 - gap0 + 1 < rect["w"]:
                return None
            rect["c0"] = int(np.random.randint(gap0, gap1 - rect["w"] + 2))
        else:
            rect["r0"] = h - rect["h"]
            gap0 = final_rects[0]["w"]
            gap1 = w - final_rects[1]["w"] - 1
            if gap1 - gap0 + 1 < rect["w"]:
                return None
            rect["c0"] = int(np.random.randint(gap0, gap1 - rect["w"] + 2))
        final_rects.append(rect)

    ordered = sorted(final_rects, key=lambda rect: rect["id"])
    canvas = np.zeros((h, w), dtype=bool)
    for rect in ordered:
        view = canvas[rect["r0"] : rect["r0"] + rect["h"], rect["c0"] : rect["c0"] + rect["w"]]
        if view.any():
            return None
        canvas[rect["r0"] : rect["r0"] + rect["h"], rect["c0"] : rect["c0"] + rect["w"]] = True

    pairs = _pair_records(ordered)
    blocked = sum(pair["blocked"] for pair in pairs)
    unblocked = sum(not pair["blocked"] for pair in pairs)
    if blocked < 1 or unblocked < 1 or len(pairs) > 5:
        return None

    inp = _build_input_from_final(ordered, (h, w))
    if inp is None:
        return None
    return [dict(rect) for rect in _slide_rectangles(_extract_rectangles(inp), inp.shape)]


def generate_input() -> np.ndarray:
    for _ in range(400):
        if bool(np.random.randint(0, 2)):
            final_rects = _make_fillable_layout()
        else:
            final_rects = _make_mixed_layout()
        if final_rects is None:
            continue

        shape = (max(rect["r0"] + rect["h"] for rect in final_rects), max(rect["c0"] + rect["w"] for rect in final_rects))
        h = max(shape[0], 12)
        w = max(shape[1], 14)
        if h > 30 or w > 30:
            continue

        final_rects = [dict(rect) for rect in final_rects]
        inp = _build_input_from_final(final_rects, (h, w))
        if inp is None:
            continue

        out, solved_final, pairs = _solve(inp)
        if _final_signature(solved_final) != _final_signature(final_rects):
            continue
        if len({int(v) for v in np.unique(inp)} - {BACKGROUND}) < 4:
            continue
        if not any(not pair["blocked"] for pair in pairs):
            continue
        if np.array_equal(inp, out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle input")
