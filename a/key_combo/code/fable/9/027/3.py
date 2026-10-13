from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")
DIR_BY_CORNER = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
OPPOSITE_CORNER = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_dicts_for_colors(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _background_components(grid: np.ndarray, bg: int = BLACK) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    ids = np.full((h, w), -1, dtype=int)
    comps: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    for r in range(h):
        for c in range(w):
            if ids[r, c] != -1 or int(grid[r, c]) != bg:
                continue
            q = deque([(r, c)])
            ids[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(grid.shape, nr, nc) and ids[nr, nc] == -1 and int(grid[nr, nc]) == bg:
                        ids[nr, nc] = next_id
                        q.append((nr, nc))
            comps[next_id] = coords
            next_id += 1
    return ids, comps


def _corner_point(box: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1
    if corner == "bl":
        return r1, c0
    return r1, c1


def _bbox_from_corner(anchor: tuple[int, int], corner: str, height: int, width: int) -> tuple[int, int, int, int]:
    r, c = anchor
    if corner == "tl":
        return r, c, r + height - 1, c + width - 1
    if corner == "tr":
        return r, c - width + 1, r + height - 1, c
    if corner == "bl":
        return r - height + 1, c, r, c + width - 1
    return r - height + 1, c - width + 1, r, c


def _cells_in_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _box_fits(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _box_is_solid_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_divider_corner(divider_mask: np.ndarray, r: int, c: int) -> bool:
    if not divider_mask[r, c]:
        return False
    up = r > 0 and divider_mask[r - 1, c]
    down = r + 1 < divider_mask.shape[0] and divider_mask[r + 1, c]
    left = c > 0 and divider_mask[r, c - 1]
    right = c + 1 < divider_mask.shape[1] and divider_mask[r, c + 1]
    return (int(up) + int(down) == 1) and (int(left) + int(right) == 1)


def _find_rectangles(grid: np.ndarray, bg: int, divider: int) -> list[dict]:
    colors = set(map(int, grid.ravel())) - {int(bg), int(divider)}
    comps = _component_dicts_for_colors(grid, colors)
    assert comps, "expected at least one colored rectangle"
    for comp in comps:
        assert _box_is_solid_rectangle(comp), "every non-divider foreground component must be a solid rectangle"
    return comps


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "expected a black canvas"
    divider_mask = g == GREY
    assert np.any(divider_mask), "expected grey divider lines"

    rectangles = _find_rectangles(g, bg, GREY)
    bg_ids, bg_components = _background_components(g, bg)
    rays: list[dict] = []

    for rect_index, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        for corner_name, start in {
            "tl": (r0, c0),
            "tr": (r0, c1),
            "bl": (r1, c0),
            "br": (r1, c1),
        }.items():
            dr, dc = DIR_BY_CORNER[corner_name]
            rr, cc = start[0] + dr, start[1] + dc
            while _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == bg:
                rr += dr
                cc += dc

            if not _in_bounds(g.shape, rr, cc):
                rays.append(
                    {
                        "rect_index": rect_index,
                        "src_color": int(rect["color"]),
                        "corner": corner_name,
                        "status": "boundary",
                    }
                )
                continue

            hit = int(g[rr, cc])
            if hit != GREY:
                rays.append(
                    {
                        "rect_index": rect_index,
                        "src_color": int(rect["color"]),
                        "corner": corner_name,
                        "status": "rectangle",
                        "hit": (rr, cc),
                    }
                )
                continue

            if not _is_divider_corner(divider_mask, rr, cc):
                rays.append(
                    {
                        "rect_index": rect_index,
                        "src_color": int(rect["color"]),
                        "corner": corner_name,
                        "status": "divider_edge",
                        "hit": (rr, cc),
                    }
                )
                continue

            nr, nc = rr + dr, cc + dc
            if not _in_bounds(g.shape, nr, nc) or int(g[nr, nc]) != bg:
                rays.append(
                    {
                        "rect_index": rect_index,
                        "src_color": int(rect["color"]),
                        "corner": corner_name,
                        "status": "blocked_after_corner",
                        "hit": (rr, cc),
                    }
                )
                continue

            component = int(bg_ids[nr, nc])
            assert component >= 0
            rays.append(
                {
                    "rect_index": rect_index,
                    "src_color": int(rect["color"]),
                    "corner": corner_name,
                    "status": "valid",
                    "hit": (rr, cc),
                    "entry": (nr, nc),
                    "component": component,
                }
            )

    return {
        "grid": g,
        "bg": bg,
        "rectangles": rectangles,
        "bg_ids": bg_ids,
        "bg_components": bg_components,
        "rays": rays,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the black region entered after a rectangle corner's diagonal passes exactly through a divider corner.

    Input structure:
    - A black canvas is partitioned by one-cell-thick grey divider lines, with multiple closed black regions.
    - One or more solid colored rectangles sit inside different regions.
    - Divider pixels and rectangle pixels are already drawn in the input.

    Rule:
    - From each rectangle corner, follow the outward diagonal without drawing it.
    - If the first non-black pixel on that diagonal is a grey convex corner, move one more step in the same direction and flood-fill every black cell in that entered region with the rectangle's color.
    - If the diagonal first reaches a grey straight edge, another rectangle, or the canvas boundary, nothing happens for that ray.
    - Divider lines and rectangles stay unchanged.

    Key insight:
    - The diagonal is only a selector: color transfers across a wall exactly when a rectangle corner is corner-to-corner aligned with a divider elbow, and the filled target is the region beyond that elbow, not the ray itself.
    """
    analysis = _analyze_grid(grid)
    out = analysis["grid"].copy()
    assignments: dict[int, int] = {}

    for ray in analysis["rays"]:
        if ray["status"] != "valid":
            continue
        component = int(ray["component"])
        color = int(ray["src_color"])
        prev = assignments.get(component)
        assert prev is None or prev == color, "a target region must not be claimed ambiguously"
        assignments[component] = color

    assert assignments, "expected at least one active corner-gated ray"

    for component, color in assignments.items():
        for r, c in analysis["bg_components"][component]:
            out[r, c] = color
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int = GREY) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_outer_border(grid: np.ndarray) -> None:
    _draw_frame(grid, (0, 0, grid.shape[0] - 1, grid.shape[1] - 1))


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_sep = ar1 + gap < br0 or br1 + gap < ar0
    col_sep = ac1 + gap < bc0 or bc1 + gap < ac0
    return row_sep or col_sep


def _sample_inner_frames(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(80):
        boxes: list[tuple[int, int, int, int]] = []
        order = list(range(count))
        random.shuffle(order)
        for _slot in order:
            placed = False
            for _attempt in range(80):
                fh = random.randint(5, 8)
                fw = random.randint(5, 8)
                if fh >= height - 4 or fw >= width - 4:
                    continue
                r0 = random.randint(2, height - fh - 2)
                c0 = random.randint(2, width - fw - 2)
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if all(_boxes_separated(box, other, gap=2) for other in boxes):
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                break
        if len(boxes) == count:
            boxes.sort()
            return boxes
    return None


def _build_divider_grid(height: int, width: int, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    _draw_outer_border(grid)
    for box in frames:
        _draw_frame(grid, box)
    return grid


def _box_cells_are_component(
    divider_grid: np.ndarray,
    box: tuple[int, int, int, int],
    component_id: int,
    bg_ids: np.ndarray,
    require_margin: bool = True,
) -> bool:
    if not _box_fits(divider_grid.shape, box):
        return False
    for r, c in _cells_in_box(box):
        if int(divider_grid[r, c]) != BLACK or int(bg_ids[r, c]) != component_id:
            return False
    if require_margin:
        r0, c0, r1, c1 = box
        exp = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        if not _box_fits(divider_grid.shape, exp):
            return False
        for r, c in _cells_in_box(exp):
            if int(divider_grid[r, c]) != BLACK or int(bg_ids[r, c]) != component_id:
                return False
    return True


def _clear_diagonal_to_corner(
    divider_grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    hit: tuple[int, int],
) -> bool:
    dr, dc = direction
    r, c = start
    while (r, c) != hit:
        r += dr
        c += dc
        if (r, c) == hit:
            return True
        if not _in_bounds(divider_grid.shape, r, c) or int(divider_grid[r, c]) != BLACK:
            return False
    return True


def _enumerate_outside_candidates(
    divider_grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    annulus_id: int,
    interior_id: int,
    bg_ids: np.ndarray,
) -> list[dict]:
    candidates: list[dict] = []
    for target_corner in CORNERS:
        src_corner = OPPOSITE_CORNER[target_corner]
        dr, dc = DIR_BY_CORNER[src_corner]
        hit = _corner_point(frame_box, target_corner)
        entry = (hit[0] + dr, hit[1] + dc)
        if not _in_bounds(divider_grid.shape, entry[0], entry[1]) or int(bg_ids[entry]) != interior_id:
            continue
        for distance in range(2, 7):
            anchor = (hit[0] - distance * dr, hit[1] - distance * dc)
            for rect_h in range(1, 4):
                for rect_w in range(1, 5):
                    box = _bbox_from_corner(anchor, src_corner, rect_h, rect_w)
                    if not _box_cells_are_component(divider_grid, box, annulus_id, bg_ids):
                        continue
                    if not _clear_diagonal_to_corner(divider_grid, anchor, (dr, dc), hit):
                        continue
                    valid = _solo_valid_rays(divider_grid, box)
                    if len(valid) != 1:
                        continue
                    only = valid[0]
                    if only["corner"] != src_corner or int(only["component"]) != interior_id:
                        continue
                    candidates.append(
                        {
                            "box": box,
                            "src_corner": src_corner,
                            "target_corner": target_corner,
                            "target_component": interior_id,
                            "source_component": annulus_id,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _enumerate_inside_candidates(
    divider_grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    annulus_id: int,
    interior_id: int,
    bg_ids: np.ndarray,
) -> list[dict]:
    candidates: list[dict] = []
    for target_corner in CORNERS:
        src_corner = target_corner
        dr, dc = DIR_BY_CORNER[src_corner]
        hit = _corner_point(frame_box, target_corner)
        entry = (hit[0] + dr, hit[1] + dc)
        if not _in_bounds(divider_grid.shape, entry[0], entry[1]) or int(bg_ids[entry]) != annulus_id:
            continue
        for distance in range(2, 7):
            anchor = (hit[0] - distance * dr, hit[1] - distance * dc)
            for rect_h in range(1, 4):
                for rect_w in range(1, 5):
                    box = _bbox_from_corner(anchor, src_corner, rect_h, rect_w)
                    if not _box_cells_are_component(divider_grid, box, interior_id, bg_ids, require_margin=False):
                        continue
                    if not _clear_diagonal_to_corner(divider_grid, anchor, (dr, dc), hit):
                        continue
                    valid = _solo_valid_rays(divider_grid, box)
                    if len(valid) != 1:
                        continue
                    only = valid[0]
                    if only["corner"] != src_corner or int(only["component"]) != annulus_id:
                        continue
                    candidates.append(
                        {
                            "box": box,
                            "src_corner": src_corner,
                            "target_corner": target_corner,
                            "target_component": annulus_id,
                            "source_component": interior_id,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _enumerate_region_boxes(
    divider_grid: np.ndarray,
    component_id: int,
    bg_ids: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for rect_h in range(1, 4):
        for rect_w in range(1, 5):
            for r in range(divider_grid.shape[0] - rect_h + 1):
                for c in range(divider_grid.shape[1] - rect_w + 1):
                    box = (r, c, r + rect_h - 1, c + rect_w - 1)
                    if _box_cells_are_component(divider_grid, box, component_id, bg_ids):
                        boxes.append(box)
    random.shuffle(boxes)
    return boxes


def _place_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _solo_valid_rays(divider_grid: np.ndarray, box: tuple[int, int, int, int]) -> list[dict]:
    temp = divider_grid.copy()
    _place_rectangle(temp, box, RECT_COLORS[0])
    return [ray for ray in _analyze_grid(temp)["rays"] if ray["status"] == "valid"]


def _ray_summary(analysis: dict) -> dict:
    per_rect_valid: dict[int, int] = {}
    per_rect_reasons: dict[int, set[str]] = {}
    valid = []
    for ray in analysis["rays"]:
        rect_index = int(ray["rect_index"])
        per_rect_reasons.setdefault(rect_index, set()).add(str(ray["status"]))
        if ray["status"] == "valid":
            per_rect_valid[rect_index] = per_rect_valid.get(rect_index, 0) + 1
            valid.append(ray)
    return {
        "valid": valid,
        "per_rect_valid": per_rect_valid,
        "per_rect_reasons": per_rect_reasons,
    }


def _validate_generated_instance(
    grid: np.ndarray,
    planned: dict,
) -> bool:
    analysis = _analyze_grid(grid)
    summary = _ray_summary(analysis)
    valid = summary["valid"]
    if len(analysis["rectangles"]) != planned["rect_count"]:
        return False
    if len(valid) != planned["valid_ray_count"]:
        return False

    per_rect_valid = summary["per_rect_valid"]
    if sorted(per_rect_valid.values()) != sorted(planned["per_rect_valid"]):
        return False

    seen_targets = {}
    actual_targets = set()
    for ray in valid:
        component = int(ray["component"])
        color = int(ray["src_color"])
        if component in seen_targets and seen_targets[component] != color:
            return False
        seen_targets[component] = color
        actual_targets.add(component)
    expected_targets = set()
    for anchor in planned["target_anchors"]:
        r, c = anchor
        if int(analysis["grid"][r, c]) != BLACK:
            return False
        expected_targets.add(int(analysis["bg_ids"][r, c]))
    if actual_targets != expected_targets:
        return False

    if "require_reasons" in planned:
        reasons = {str(ray["status"]) for ray in analysis["rays"] if ray["status"] != "valid"}
        if not planned["require_reasons"].issubset(reasons):
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for component in expected_targets:
        coords = analysis["bg_components"][component]
        colors = {int(out[r, c]) for r, c in coords}
        if len(colors) != 1:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    frame_count = random.choices([2, 3], weights=[4, 1])[0]
    frames = _sample_inner_frames(height, width, frame_count)
    if frames is None:
        return None

    divider_grid = _build_divider_grid(height, width, frames)
    bg_ids, _bg_components = _background_components(divider_grid, BLACK)
    annulus_id = int(bg_ids[1, 1])
    interior_ids = [int(bg_ids[r0 + 1, c0 + 1]) for r0, c0, _r1, _c1 in frames]

    target_index = random.randrange(frame_count)
    source_choices = [i for i in range(frame_count) if i != target_index]
    source_index = random.choice(source_choices)
    outside_candidates = _enumerate_outside_candidates(
        divider_grid, frames[target_index], annulus_id, interior_ids[target_index], bg_ids
    )
    inside_candidates = _enumerate_inside_candidates(
        divider_grid, frames[source_index], annulus_id, interior_ids[source_index], bg_ids
    )
    if not outside_candidates or not inside_candidates:
        return None

    colors = random.sample(list(RECT_COLORS), 2)
    pair_tries = 0
    for outside in outside_candidates[:40]:
        for inside in inside_candidates[:40]:
            pair_tries += 1
            if pair_tries > 220:
                return None
            grid = divider_grid.copy()
            _place_rectangle(grid, outside["box"], colors[0])
            _place_rectangle(grid, inside["box"], colors[1])

            planned = {
                "rect_count": 2,
                "valid_ray_count": 2,
                "per_rect_valid": [1, 1],
                "target_anchors": {(1, 1), (frames[target_index][0] + 1, frames[target_index][1] + 1)},
                "require_reasons": {"divider_edge"},
            }
            if _validate_generated_instance(grid, planned):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-gated ray puzzle instance")
