from __future__ import annotations

import random

import numpy as np


Grid = np.ndarray
Coord = tuple[int, int]
Box = tuple[int, int, int, int]

BG = 0
OVERLAP_COLOR = 2
SOURCE_COLORS = (1, 3, 4, 7)
RECT_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
GRID_SIDE_MIN = 13
GRID_SIDE_MAX = 20
GEN_ATTEMPTS = 500

_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox_from_coords(coords: list[Coord]) -> Box:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _background_color(grid: Grid) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_h(box: Box) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: Box) -> int:
    return int(box[3] - box[1] + 1)


def _bbox_area(box: Box) -> int:
    return _bbox_h(box) * _bbox_w(box)


def _touches_border(box: Box, height: int, width: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1


def _boxes_overlap(box_a: Box, box_b: Box) -> bool:
    return not (
        box_a[2] < box_b[0]
        or box_b[2] < box_a[0]
        or box_a[3] < box_b[1]
        or box_b[3] < box_a[1]
    )


def _boxes_have_gap(box_a: Box, box_b: Box, gap: int = 1) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _intersection_area(box_a: Box, box_b: Box) -> int:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _filled_rectangle(component: dict) -> bool:
    return int(component["area"]) == _bbox_area(component["bbox"])


def _find_monochrome_components(grid: Grid, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _quadrant_direction(marker: Coord, box: Box) -> tuple[int, int]:
    mr, mc = marker
    r0, c0, r1, c1 = box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    assert mr != center_r and mc != center_c, "marker must lie in a diagonal quadrant"
    dr = -1 if mr < center_r else 1
    dc = -1 if mc < center_c else 1
    return int(dr), int(dc)


def _parse_objects(grid: Grid) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    bg = _background_color(g)

    components = _find_monochrome_components(g, bg=bg)
    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    objects: list[dict] = []
    for color, parts in sorted(by_color.items()):
        rectangles = [part for part in parts if int(part["area"]) > 1]
        markers = [part for part in parts if int(part["area"]) == 1]
        assert len(rectangles) == 1, "each source color must have one rectangle component"
        assert len(markers) == 1, "each source color must have one marker pixel"

        rect = rectangles[0]
        box = tuple(map(int, rect["bbox"]))
        h = _bbox_h(box)
        w = _bbox_w(box)
        assert h >= 2 and w >= 2, "rectangle must be at least 2x2"
        assert _filled_rectangle(rect), "source object must be a filled rectangle"
        assert _touches_border(box, height, width), "rectangle must touch the grid border"

        marker = tuple(map(int, markers[0]["coords"][0]))
        assert 0 < marker[0] < height - 1 and 0 < marker[1] < width - 1, "marker may not sit on the border"
        dr, dc = _quadrant_direction(marker, box)

        target_box = (
            box[0] + dr * h,
            box[1] + dc * w,
            box[2] + dr * h,
            box[3] + dc * w,
        )
        assert 0 <= target_box[0] <= target_box[2] < height, "vertical translation must stay in bounds"
        assert 0 <= target_box[1] <= target_box[3] < width, "horizontal translation must stay in bounds"

        objects.append(
            {
                "color": int(color),
                "box": box,
                "marker": marker,
                "dr": int(dr),
                "dc": int(dc),
                "target_box": target_box,
            }
        )

    assert 2 <= len(objects) <= 4, "expected two to four rectangles"
    assert len({obj["color"] for obj in objects}) == len(objects), "rectangle colors must be distinct"
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            assert not _boxes_overlap(obj_a["box"], obj_b["box"]), "source rectangles may not overlap"
    return bg, objects


def _render_targets(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    count = np.zeros(shape, dtype=int)
    color_grid = np.full(shape, bg, dtype=int)

    for obj in objects:
        r0, c0, r1, c1 = obj["target_box"]
        count_view = count[r0 : r1 + 1, c0 : c1 + 1]
        color_view = color_grid[r0 : r1 + 1, c0 : c1 + 1]
        empty = count_view == 0
        color_view[empty] = int(obj["color"])
        color_view[~empty] = OVERLAP_COLOR
        count_view += 1

    out = np.full(shape, bg, dtype=int)
    out[count == 1] = color_grid[count == 1]
    out[count >= 2] = OVERLAP_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each border-aligned rectangle diagonally by its own size according to a same-color marker, then color overlaps red.

    Input structure:
    - The background is black.
    - Two to four distinct monochrome filled rectangles sit against the grid border; each rectangle color appears nowhere else except once.
    - For every rectangle, a same-color isolated marker pixel lies off the border in one diagonal quadrant relative to that rectangle's bbox center.

    Rule:
    - Pair each border rectangle with its unique same-color marker pixel.
    - Read the marker quadrant relative to the rectangle center: top/bottom chooses vertical direction and left/right chooses horizontal direction.
    - Move the whole rectangle by exactly one rectangle-height vertically and one rectangle-width horizontally in that diagonal direction, and erase both the original rectangle and all markers.
    - After all rectangles move, any pixel claimed by two or more translated rectangles becomes red; non-overlap translated pixels keep their rectangle color.

    Key insight:
    - The marker does not label a ray or a destination cell; it encodes a diagonal sign pair that controls a size-dependent rectangle translation, so overlap must be resolved only after all rectangles move.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _parse_objects(g)
    return _render_targets(g.shape, objects, bg=bg)


def _make_candidate(height: int, width: int, h: int, w: int, r0: int, c0: int, dr: int, dc: int, kind: str) -> dict | None:
    target_box = (r0 + dr * h, c0 + dc * w, r0 + dr * h + h - 1, c0 + dc * w + w - 1)
    if not (0 <= target_box[0] <= target_box[2] < height and 0 <= target_box[1] <= target_box[3] < width):
        return None
    box = (r0, c0, r0 + h - 1, c0 + w - 1)
    return {
        "box": box,
        "target_box": target_box,
        "dr": int(dr),
        "dc": int(dc),
        "kind": kind,
        "h": int(h),
        "w": int(w),
        "area": int(h * w),
    }


def _placement_candidates(height: int, width: int) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[Box, int, int]] = set()

    def add(candidate: dict | None) -> None:
        if candidate is None:
            return
        key = (candidate["box"], candidate["dr"], candidate["dc"])
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    for h, w in RECT_SHAPES:
        if height >= 2 * h and width >= 2 * w:
            add(_make_candidate(height, width, h, w, 0, 0, 1, 1, "corner"))
            add(_make_candidate(height, width, h, w, 0, width - w, 1, -1, "corner"))
            add(_make_candidate(height, width, h, w, height - h, 0, -1, 1, "corner"))
            add(_make_candidate(height, width, h, w, height - h, width - w, -1, -1, "corner"))

        if height >= 2 * h:
            for dc in (-1, 1):
                c_min = max(1, w if dc == -1 else 1)
                c_max = min(width - w - 2, width - 2 * w if dc == 1 else width - w - 2)
                for c0 in range(c_min, c_max + 1):
                    add(_make_candidate(height, width, h, w, 0, c0, 1, dc, "side"))
                    add(_make_candidate(height, width, h, w, height - h, c0, -1, dc, "side"))

        if width >= 2 * w:
            for dr in (-1, 1):
                r_min = max(1, h if dr == -1 else 1)
                r_max = min(height - h - 2, height - 2 * h if dr == 1 else height - h - 2)
                for r0 in range(r_min, r_max + 1):
                    add(_make_candidate(height, width, h, w, r0, 0, dr, 1, "side"))
                    add(_make_candidate(height, width, h, w, r0, width - w, dr, -1, "side"))
    return candidates


def _coverage_map(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["target_box"]
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _target_layout_clear(shape: tuple[int, int], specs: list[dict]) -> bool:
    cover = _coverage_map(shape, specs)
    overlap_cells = int(np.count_nonzero(cover >= 2))
    triple_cells = int(np.count_nonzero(cover >= 3))
    if overlap_cells < 1 or overlap_cells > 6 or triple_cells > 2:
        return False

    for spec in specs:
        r0, c0, r1, c1 = spec["target_box"]
        unique_cells = int(np.count_nonzero(cover[r0 : r1 + 1, c0 : c1 + 1] == 1))
        if unique_cells < 2:
            return False

    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            if _intersection_area(spec_a["target_box"], spec_b["target_box"]) >= min(spec_a["area"], spec_b["area"]):
                return False
    return True


def _overlap_pair_count(specs: list[dict]) -> int:
    total = 0
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            if _intersection_area(spec_a["target_box"], spec_b["target_box"]) > 0:
                total += 1
    return total


def _plan_for_call(call_idx: int) -> dict:
    n_objects = 2 + (call_idx % 3)
    primary_pairs = [((1, 1), (-1, -1)), ((1, -1), (-1, 1))]
    return {
        "n_objects": n_objects,
        "primary_dirs": primary_pairs[call_idx % 2],
        "min_side": 1 if n_objects == 2 else 2,
        "require_corner": n_objects >= 3 and call_idx % 4 in (1, 2),
        "min_overlap_pairs": 2 if n_objects == 4 and call_idx % 2 == 0 else 1,
    }


def _choose_specs(height: int, width: int, plan: dict) -> list[dict] | None:
    candidates = _placement_candidates(height, width)
    random.shuffle(candidates)
    preferred_dirs = {plan["primary_dirs"][0], plan["primary_dirs"][1]}
    pair_candidates: list[tuple[dict, dict]] = []

    for i, cand_a in enumerate(candidates):
        for cand_b in candidates[i + 1 :]:
            if {(cand_a["dr"], cand_a["dc"]), (cand_b["dr"], cand_b["dc"])} != preferred_dirs:
                continue
            if not _boxes_have_gap(cand_a["box"], cand_b["box"], gap=1):
                continue
            overlap_area = _intersection_area(cand_a["target_box"], cand_b["target_box"])
            if overlap_area < 1 or overlap_area >= min(cand_a["area"], cand_b["area"]):
                continue
            side_count = int(cand_a["kind"] == "side") + int(cand_b["kind"] == "side")
            if side_count + (plan["n_objects"] - 2) < plan["min_side"]:
                continue
            if plan["require_corner"]:
                corner_count = int(cand_a["kind"] == "corner") + int(cand_b["kind"] == "corner")
                if corner_count + (plan["n_objects"] - 2) < 1:
                    continue
            if not _target_layout_clear((height, width), [cand_a, cand_b]):
                continue
            pair_candidates.append((cand_a, cand_b))

    random.shuffle(pair_candidates)
    for cand_a, cand_b in pair_candidates:
        base = [cand_a, cand_b]
        pool = [cand for cand in candidates if cand not in base]
        for _ in range(60):
            chosen = list(base)
            random.shuffle(pool)
            for cand in pool:
                if len(chosen) == plan["n_objects"]:
                    break
                if not all(_boxes_have_gap(cand["box"], prev["box"], gap=1) for prev in chosen):
                    continue
                tentative = chosen + [cand]
                if not _target_layout_clear((height, width), tentative):
                    continue
                chosen.append(cand)

            if len(chosen) != plan["n_objects"]:
                continue
            side_count = sum(spec["kind"] == "side" for spec in chosen)
            corner_count = sum(spec["kind"] == "corner" for spec in chosen)
            if side_count < plan["min_side"]:
                continue
            if plan["require_corner"] and corner_count < 1:
                continue
            if _overlap_pair_count(chosen) < plan["min_overlap_pairs"]:
                continue
            if not _target_layout_clear((height, width), chosen):
                continue
            return chosen
    return None


def _marker_candidates(spec: dict, height: int, width: int, rect_cells: set[Coord], marker_cells: set[Coord]) -> list[Coord]:
    r0, c0, r1, c1 = spec["box"]
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    candidates: list[tuple[float, float, int, int]] = []

    for r in range(1, height - 1):
        if spec["dr"] == -1 and not (r < center_r):
            continue
        if spec["dr"] == 1 and not (r > center_r):
            continue
        for c in range(1, width - 1):
            if spec["dc"] == -1 and not (c < center_c):
                continue
            if spec["dc"] == 1 and not (c > center_c):
                continue
            if (r, c) in rect_cells or (r, c) in marker_cells:
                continue
            if any((nr, nc) in rect_cells or (nr, nc) in marker_cells for nr, nc in _neighbors4(r, c)):
                continue
            chebyshev = max(abs(r - center_r), abs(c - center_c))
            if chebyshev < 1.1:
                continue
            manhattan = abs(r - center_r) + abs(c - center_c)
            candidates.append((abs(chebyshev - 4.0), manhattan, r, c))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [(r, c) for _score, _manhattan, r, c in candidates]


def _place_markers(specs: list[dict], height: int, width: int) -> list[dict] | None:
    rect_cells = {
        (r, c)
        for spec in specs
        for r in range(spec["box"][0], spec["box"][2] + 1)
        for c in range(spec["box"][1], spec["box"][3] + 1)
    }
    order = list(range(len(specs)))
    order.sort(key=lambda idx: len(_marker_candidates(specs[idx], height, width, rect_cells, set())))
    placed: dict[int, Coord] = {}

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        current_markers = set(placed.values())
        candidates = _marker_candidates(specs[idx], height, width, rect_cells, current_markers)
        if not candidates:
            return False
        head = candidates[: min(len(candidates), 12)]
        random.shuffle(head)
        tail = candidates[min(len(candidates), 12) :]
        for marker in head + tail:
            placed[idx] = marker
            if search(pos + 1):
                return True
            del placed[idx]
        return False

    if not search(0):
        return None

    enriched = []
    for idx, spec in enumerate(specs):
        enriched.append({**spec, "marker": placed[idx]})
    return enriched


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    height, width = shape
    out = np.full(shape, BG, dtype=int)
    claims = np.zeros(shape, dtype=int)
    last_color = np.full(shape, BG, dtype=int)

    for spec in specs:
        tr0, tc0, tr1, tc1 = spec["target_box"]
        color = int(spec["color"])
        for r in range(tr0, tr1 + 1):
            for c in range(tc0, tc1 + 1):
                claims[r, c] += 1
                if claims[r, c] == 1:
                    last_color[r, c] = color
                else:
                    last_color[r, c] = OVERLAP_COLOR

    for r in range(height):
        for c in range(width):
            if claims[r, c] == 1:
                out[r, c] = last_color[r, c]
            elif claims[r, c] >= 2:
                out[r, c] = OVERLAP_COLOR
    return out


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    for spec in specs:
        mr, mc = spec["marker"]
        assert int(grid[mr, mc]) == BG
        grid[mr, mc] = int(spec["color"])
    return grid


def _validate_specs(shape: tuple[int, int], specs: list[dict], plan: dict) -> None:
    height, width = shape
    assert 2 <= len(specs) <= 4
    colors = [int(spec["color"]) for spec in specs]
    assert len(set(colors)) == len(colors)
    assert all(color in SOURCE_COLORS for color in colors)

    all_rect_cells: set[Coord] = {
        (r, c)
        for spec in specs
        for r in range(spec["box"][0], spec["box"][2] + 1)
        for c in range(spec["box"][1], spec["box"][3] + 1)
    }
    marker_cells: set[Coord] = set()
    for i, spec in enumerate(specs):
        box = spec["box"]
        target_box = spec["target_box"]
        marker = spec["marker"]
        h = _bbox_h(box)
        w = _bbox_w(box)
        assert (h, w) in RECT_SHAPES
        assert _touches_border(box, height, width)
        assert 0 <= target_box[0] <= target_box[2] < height
        assert 0 <= target_box[1] <= target_box[3] < width
        assert _quadrant_direction(marker, box) == (spec["dr"], spec["dc"])
        assert 0 < marker[0] < height - 1 and 0 < marker[1] < width - 1
        assert marker not in marker_cells
        assert marker not in all_rect_cells
        assert all((nr, nc) not in all_rect_cells and (nr, nc) not in marker_cells for nr, nc in _neighbors4(*marker))
        marker_cells.add(marker)
        for other in specs[i + 1 :]:
            assert _boxes_have_gap(box, other["box"], gap=1)

    assert sum(spec["kind"] == "side" for spec in specs) >= plan["min_side"]
    if plan["require_corner"]:
        assert any(spec["kind"] == "corner" for spec in specs)
    assert _overlap_pair_count(specs) >= plan["min_overlap_pairs"]
    assert _target_layout_clear(shape, specs)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    plan = _plan_for_call(call_idx)

    for _ in range(GEN_ATTEMPTS):
        height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
        width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
        specs = _choose_specs(height, width, plan)
        if specs is None:
            continue

        colors = random.sample(list(SOURCE_COLORS), k=plan["n_objects"])
        random.shuffle(specs)
        colored_specs = [{**spec, "color": color} for spec, color in zip(specs, colors)]
        placed_specs = _place_markers(colored_specs, height, width)
        if placed_specs is None:
            continue

        _validate_specs((height, width), placed_specs, plan)
        grid = _render_input((height, width), placed_specs)
        expected = _expected_from_specs((height, width), placed_specs)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
