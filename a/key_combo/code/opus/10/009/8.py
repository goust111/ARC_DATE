from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BORDER_COLORS = [1, 5, 6, 8, 9]
SOURCE_COLORS = [2, 3, 4, 7]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for coords in _components_of_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "norm": _normalize(coords),
                    "area": len(coords),
                }
            )
    return comps


def _touches_bbox_boundary(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r == r0 or r == r1 or c == c0 or c == c1


def _is_template_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if not (5 <= h <= 7 and 5 <= w <= 7):
        return False
    if comp["area"] < 2 * (h + w) - 8:
        return False
    perimeter = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    coords = set(comp["coords"])
    if not coords <= perimeter:
        return False
    missing = perimeter - coords
    if not missing:
        return False
    return True


def _black_components_in_box(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == BG
    comps = _components_of_mask(patch)
    return [[(r0 + r, c0 + c) for r, c in comp] for comp in comps]


def _hole_regions_in_box(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    holes = []
    for comp in _black_components_in_box(grid, box):
        if any(_touches_bbox_boundary(cell, box) for cell in comp):
            continue
        holes.append(comp)
    return holes


def _classify_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    templates = []
    sources = []
    for comp in _monochrome_components(grid):
        if _is_template_component(comp):
            templates.append(comp)
        else:
            sources.append(comp)
    templates.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    sources.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return templates, sources


def _candidate_placement(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    source_norm: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> dict | None:
    r0, c0, r1, c1 = box
    placed = [(top + dr, left + dc) for dr, dc in source_norm]
    if any(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in placed):
        return None
    if any(grid[r, c] != BG for r, c in placed):
        return None
    if not any(_touches_bbox_boundary(cell, box) for cell in placed):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    for r, c in placed:
        patch[r - r0, c - c0] = 1

    probe = np.full_like(grid, BG)
    probe[r0 : r1 + 1, c0 : c1 + 1] = patch
    holes = _hole_regions_in_box(probe, box)
    area = sum(len(hole) for hole in holes)
    if not holes or not (1 <= area <= 4):
        return None
    return {"box": box, "placed": tuple(sorted(placed)), "holes": holes, "hole_area": area}


def _unique_assignment(candidates: list[list[dict]], template_count: int) -> list[dict]:
    order = sorted(range(len(candidates)), key=lambda i: len(candidates[i]))
    used_templates: set[int] = set()
    current: list[dict | None] = [None] * len(candidates)
    solutions: list[list[dict]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append([item for item in current if item is not None])
            return
        src_idx = order[pos]
        for cand in candidates[src_idx]:
            temp_idx = cand["template_idx"]
            if temp_idx in used_templates:
                continue
            used_templates.add(temp_idx)
            current[src_idx] = cand
            dfs(pos + 1)
            current[src_idx] = None
            used_templates.remove(temp_idx)

    dfs(0)
    assert len(solutions) == 1, "source-to-template matching must be unique"
    assignment = solutions[0]
    assert len({cand["template_idx"] for cand in assignment}) == template_count
    return assignment


def _match_sources_to_templates(grid: np.ndarray, templates: list[dict], sources: list[dict]) -> list[dict]:
    assert len(templates) == len(sources) >= 2
    for template in templates:
        assert not _hole_regions_in_box(grid, template["bbox"]), "input templates must not contain holes yet"

    candidates: list[list[dict]] = [[] for _ in sources]
    for src_idx, source in enumerate(sources):
        shape = source["norm"]
        max_dr = max(dr for dr, _ in shape)
        max_dc = max(dc for _, dc in shape)
        for temp_idx, template in enumerate(templates):
            r0, c0, r1, c1 = template["bbox"]
            for top in range(r0, r1 - max_dr + 1):
                for left in range(c0, c1 - max_dc + 1):
                    cand = _candidate_placement(grid, template["bbox"], shape, top, left)
                    if cand is None:
                        continue
                    cand["template_idx"] = temp_idx
                    cand["source_idx"] = src_idx
                    candidates[src_idx].append(cand)
        assert candidates[src_idx], "each source must fit at least one template gap"

    return _unique_assignment(candidates, len(templates))


def _apply_rays(grid: np.ndarray, templates: list[dict]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for template in templates:
        color = template["color"]
        base = out.copy()
        holes = _hole_regions_in_box(base, template["bbox"] if "bbox" in template else template["box"])
        for hole in holes:
            for r, c in sorted(hole):
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w and base[nr, nc] == BG:
                        out[nr, nc] = color
                        nr += dr
                        nc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each source polyomino to the unique border gap that seals a template pocket, then emit border-colored rays from the newly enclosed holes.

    Input structure:
    - The grid uses black as background.
    - Several distinct-color rectangular template borders occupy the left side; each border is 5x5 to 7x7 and has a single connected boundary gap plus interior black space.
    - Separate distinct-color 4-connected source shapes appear on the right side, and their colors differ from every border color.

    Rule:
    - Detect every template border and every source shape.
    - For each source, try every translation inside every template gap and keep the unique placement that turns a previously open black region into one or more enclosed black holes inside that template.
    - Move the matched source into that placement, erase its original right-side copy to black, and then, for each enclosed hole pixel, cast rays up, down, left, and right in the template border color until a non-black pixel blocks the ray.
    - Process templates from top-left to bottom-right; later templates may overwrite earlier rays on shared black cells.

    Key insight:
    - The source shape is identified not by color or proximity but by the only translation whose fill changes template topology: it seals an opening and makes a new hole appear, and that emergent hole alone determines the ray origins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates, sources = _classify_objects(g)
    assignment = _match_sources_to_templates(g, templates, sources)

    out = g.copy()
    for source, match in zip(sources, assignment):
        for r, c in source["coords"]:
            out[r, c] = BG
        for r, c in match["placed"]:
            out[r, c] = source["color"]

    return _apply_rays(out, templates)


def _build_template_patch(
    height: int,
    width: int,
    corner: str,
    pocket_h: int,
    pocket_w: int,
    border_color: int,
    source_color: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    assert 5 <= height <= 7 and 5 <= width <= 7
    assert 1 <= pocket_h <= 2 and 1 <= pocket_w <= 2
    assert width >= pocket_w + 4

    out = np.zeros((height, width), dtype=int)
    out[0, :] = border_color
    out[-1, :] = border_color
    out[:, 0] = border_color
    out[:, -1] = border_color

    plug: set[tuple[int, int]] = set()
    if corner == "tl":
        anchor_c = pocket_w + 1
        open_c = anchor_c + 1
        out[0, anchor_c] = BG
        out[0, open_c] = BG
        plug.add((0, anchor_c))
        for r in range(1, pocket_h + 1):
            plug.add((r, anchor_c))
        for c in range(1, anchor_c + 1):
            plug.add((pocket_h + 1, c))
    elif corner == "tr":
        anchor_c = width - pocket_w - 2
        open_c = anchor_c - 1
        out[0, anchor_c] = BG
        out[0, open_c] = BG
        plug.add((0, anchor_c))
        for r in range(1, pocket_h + 1):
            plug.add((r, anchor_c))
        for c in range(anchor_c, width - 1):
            plug.add((pocket_h + 1, c))
    elif corner == "bl":
        anchor_c = pocket_w + 1
        open_c = anchor_c + 1
        out[-1, anchor_c] = BG
        out[-1, open_c] = BG
        plug.add((height - 1, anchor_c))
        for r in range(height - pocket_h - 1, height - 1):
            plug.add((r, anchor_c))
        for c in range(1, anchor_c + 1):
            plug.add((height - pocket_h - 2, c))
    else:
        anchor_c = width - pocket_w - 2
        open_c = anchor_c - 1
        out[-1, anchor_c] = BG
        out[-1, open_c] = BG
        plug.add((height - 1, anchor_c))
        for r in range(height - pocket_h - 1, height - 1):
            plug.add((r, anchor_c))
        for c in range(anchor_c, width - 1):
            plug.add((height - pocket_h - 2, c))

    for r, c in plug:
        out[r, c] = source_color

    inp = out.copy()
    for r, c in plug:
        inp[r, c] = BG

    return inp, out, _normalize(sorted(plug)), (0, 0, height - 1, width - 1)


def _template_specs() -> list[dict]:
    specs = []
    for height in range(5, 8):
        for width in range(5, 8):
            for corner in ("tl", "tr", "bl", "br"):
                for pocket_h in (1, 2):
                    for pocket_w in (1, 2):
                        if width < pocket_w + 4:
                            continue
                        size = pocket_h * pocket_w
                        if not (1 <= size <= 4):
                            continue
                        dummy_inp, dummy_out, norm, _ = _build_template_patch(
                            height,
                            width,
                            corner,
                            pocket_h,
                            pocket_w,
                            border_color=1,
                            source_color=2,
                        )
                        holes = _hole_regions_in_box(dummy_out, (0, 0, height - 1, width - 1))
                        if sum(len(h) for h in holes) != size:
                            continue
                        if _hole_regions_in_box(dummy_inp, (0, 0, height - 1, width - 1)):
                            continue
                        specs.append(
                            {
                                "height": height,
                                "width": width,
                                "corner": corner,
                                "pocket_h": pocket_h,
                                "pocket_w": pocket_w,
                                "shape": norm,
                            }
                        )
    by_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for spec in specs:
        by_shape.setdefault(spec["shape"], spec)
    return list(by_shape.values())


SHAPE_LIBRARY = _template_specs()


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _choose_template_instances(count: int) -> list[dict]:
    chosen = random.sample(SHAPE_LIBRARY, count)
    random.shuffle(chosen)
    return chosen


def _place_templates(height: int, left_width: int, specs: list[dict]) -> list[dict] | None:
    for _ in range(120):
        placements: list[dict] = []
        for spec in specs:
            h = spec["height"]
            w = spec["width"]
            candidates = [
                (r, c)
                for r in range(0, height - h + 1)
                for c in range(0, left_width - w + 1)
            ]
            random.shuffle(candidates)
            placed = False
            for r, c in candidates:
                box = (r, c, r + h - 1, c + w - 1)
                if any(_boxes_overlap(box, other["box"]) for other in placements):
                    continue
                placements.append({"spec": spec, "box": box})
                placed = True
                break
            if not placed:
                break
        if len(placements) != len(specs):
            continue

        row_overlap = any(
            _ranges_overlap(a["box"][0], a["box"][2], b["box"][0], b["box"][2])
            for i, a in enumerate(placements)
            for b in placements[i + 1 :]
        )
        col_overlap = any(
            _ranges_overlap(a["box"][1], a["box"][3], b["box"][1], b["box"][3])
            for i, a in enumerate(placements)
            for b in placements[i + 1 :]
        )
        if row_overlap and col_overlap:
            placements.sort(key=lambda item: (item["box"][0], item["box"][1]))
            return placements
    return None


def _place_sources(
    height: int,
    width: int,
    right_start: int,
    shapes: list[tuple[tuple[int, int], ...]],
) -> list[tuple[int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    for shape in shapes:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        candidates = [
            (r, c)
            for r in range(0, height - max_r)
            for c in range(right_start, width - max_c)
        ]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            box = (r, c, r + max_r, c + max_c)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            boxes.append(box)
            positions.append((r, c))
            placed = True
            break
        if not placed:
            return None
    return positions


def _ray_segments(pre_rays: np.ndarray, post_rays: np.ndarray, templates: list[dict]) -> list[dict]:
    segs = []
    for template in templates:
        color = template["color"]
        before = pre_rays.copy()
        after = _apply_rays(before, [template])
        diff = np.argwhere((before != after) & (after == color))
        for r, c in diff:
            segs.append({"color": color, "cell": (int(r), int(c))})
        pre_rays = _apply_rays(pre_rays, [template])
    return segs


def _collision_count(before_rays: np.ndarray, after_rays: np.ndarray, templates: list[dict]) -> int:
    h, w = before_rays.shape
    collisions = 0
    current = before_rays.copy()
    for template in templates:
        base = current.copy()
        holes = _hole_regions_in_box(base, template["bbox"])
        for hole in holes:
            for r, c in hole:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    progressed = False
                    while 0 <= nr < h and 0 <= nc < w and base[nr, nc] == BG:
                        progressed = True
                        nr += dr
                        nc += dc
                    if progressed and 0 <= nr < h and 0 <= nc < w and base[nr, nc] != BG:
                        collisions += 1
        current = _apply_rays(current, [template])
    assert np.array_equal(current, after_rays)
    return collisions


def _generate_case() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    count = random.randint(2, 4)
    specs = _choose_template_instances(count)
    left_width = random.randint(11, 15)
    right_width = random.randint(6, 10)
    width = left_width + 1 + right_width
    height = random.randint(16, 22)

    placements = _place_templates(height, left_width, specs)
    if placements is None:
        return None

    border_colors = random.sample(BORDER_COLORS, count)
    source_colors = random.sample(SOURCE_COLORS, count)

    inp = np.zeros((height, width), dtype=int)
    solved = np.zeros((height, width), dtype=int)
    template_records: list[dict] = []
    shapes: list[tuple[tuple[int, int], ...]] = []

    for placement, border_color, source_color in zip(placements, border_colors, source_colors):
        spec = placement["spec"]
        r0, c0, r1, c1 = placement["box"]
        in_patch, out_patch, shape, _ = _build_template_patch(
            spec["height"],
            spec["width"],
            spec["corner"],
            spec["pocket_h"],
            spec["pocket_w"],
            border_color,
            source_color,
        )
        if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return None
        inp[r0 : r1 + 1, c0 : c1 + 1] = in_patch
        solved[r0 : r1 + 1, c0 : c1 + 1] = out_patch
        shapes.append(shape)
        template_records.append(
            {
                "box": (r0, c0, r1, c1),
                "color": border_color,
                "source_color": source_color,
                "shape": shape,
            }
        )

    src_positions = _place_sources(height, width, left_width + 1, shapes)
    if src_positions is None:
        return None

    for record, shape, (r0, c0) in zip(template_records, shapes, src_positions):
        for dr, dc in shape:
            inp[r0 + dr, c0 + dc] = record["source_color"]

    out = _apply_rays(solved, sorted(template_records, key=lambda item: (item["box"][0], item["box"][1])))
    return inp, out, template_records


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    templates, sources = _classify_objects(inp)
    if len(templates) != len(records) or len(sources) != len(records):
        return False
    try:
        assignment = _match_sources_to_templates(inp, templates, sources)
    except AssertionError:
        return False

    expected_boxes = {record["box"] for record in records}
    matched_boxes = {templates[item["template_idx"]]["bbox"] for item in assignment}
    if expected_boxes != matched_boxes:
        return False

    filled = inp.copy()
    for source, match in zip(sources, assignment):
        for r, c in source["coords"]:
            filled[r, c] = BG
        for r, c in match["placed"]:
            filled[r, c] = source["color"]

    if any(_hole_regions_in_box(inp, template["bbox"]) for template in templates):
        return False
    total_hole_area = 0
    for template in templates:
        holes = _hole_regions_in_box(filled, template["bbox"])
        area = sum(len(hole) for hole in holes)
        if not (1 <= area <= 4):
            return False
        total_hole_area += area
    if total_hole_area < len(templates):
        return False

    if _collision_count(filled, expected, templates) == 0:
        return False
    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected, records = case
        if _semantic_ok(inp, expected, records):
            return inp
    raise RuntimeError("failed to generate a valid complementary-hole-ray case")
