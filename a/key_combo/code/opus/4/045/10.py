from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
MOVING_COLOR_CHOICES = (1, 2, 3)
SINGLETON_COLOR_CHOICES = (4, 5, 6, 7, 8, 9)
SIZE_OPTIONS = ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4))
GRID_MIN = 13
GRID_MAX = 18
GENERATION_RETRIES = 300
CORE_ATTEMPTS = 900
ADD_ATTEMPTS = 400


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _bbox_in_bounds(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _expand_bbox(
    bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _bbox_intersects_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _touch_kind(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    touches = {
        "T": r0 == 0,
        "B": r1 == h - 1,
        "L": c0 == 0,
        "R": c1 == w - 1,
    }
    if touches["T"] and touches["B"]:
        return None
    if touches["L"] and touches["R"]:
        return None
    count = sum(bool(v) for v in touches.values())
    if count == 1:
        if touches["T"]:
            return "top"
        if touches["B"]:
            return "bottom"
        if touches["L"]:
            return "left"
        return "right"
    if count == 2:
        if touches["T"] and touches["L"]:
            return "TL"
        if touches["T"] and touches["R"]:
            return "TR"
        if touches["B"] and touches["L"]:
            return "BL"
        if touches["B"] and touches["R"]:
            return "BR"
    return None


def _move_unit(kind: str) -> tuple[int, int]:
    if kind == "top":
        return 1, 0
    if kind == "bottom":
        return -1, 0
    if kind == "left":
        return 0, 1
    if kind == "right":
        return 0, -1
    if kind == "TL":
        return 1, 1
    if kind == "TR":
        return 1, -1
    if kind == "BL":
        return -1, 1
    if kind == "BR":
        return -1, -1
    raise ValueError(f"unknown kind: {kind}")


def _base_target_bbox(bbox: tuple[int, int, int, int], kind: str) -> tuple[int, int, int, int]:
    dr_unit, dc_unit = _move_unit(kind)
    dr = dr_unit * _bbox_height(bbox) if dr_unit else 0
    dc = dc_unit * _bbox_width(bbox) if dc_unit else 0
    return _shift_bbox(bbox, dr, dc)


def _step_bbox(bbox: tuple[int, int, int, int], kind: str, steps: int = 1) -> tuple[int, int, int, int]:
    dr_unit, dc_unit = _move_unit(kind)
    return _shift_bbox(bbox, dr_unit * steps, dc_unit * steps)


def _candidate_bboxes(h: int, w: int, kinds: tuple[str, ...] | list[str] | None = None) -> list[tuple[str, tuple[int, int, int, int]]]:
    kinds = tuple(kinds or ("top", "bottom", "left", "right", "TL", "TR", "BL", "BR"))
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for height, width in SIZE_OPTIONS:
        if height >= h or width >= w:
            continue
        if "top" in kinds:
            for c0 in range(1, w - width):
                out.append(("top", (0, c0, height - 1, c0 + width - 1)))
        if "bottom" in kinds:
            r0 = h - height
            for c0 in range(1, w - width):
                out.append(("bottom", (r0, c0, h - 1, c0 + width - 1)))
        if "left" in kinds:
            for r0 in range(1, h - height):
                out.append(("left", (r0, 0, r0 + height - 1, width - 1)))
        if "right" in kinds:
            c0 = w - width
            for r0 in range(1, h - height):
                out.append(("right", (r0, c0, r0 + height - 1, w - 1)))
        if "TL" in kinds:
            out.append(("TL", (0, 0, height - 1, width - 1)))
        if "TR" in kinds:
            out.append(("TR", (0, w - width, height - 1, w - 1)))
        if "BL" in kinds:
            out.append(("BL", (h - height, 0, h - 1, width - 1)))
        if "BR" in kinds:
            out.append(("BR", (h - height, w - width, h - 1, w - 1)))
    return out


def _separated_from_all(
    bbox: tuple[int, int, int, int], existing: list[dict], h: int, w: int
) -> bool:
    grown = _expand_bbox(bbox, h, w, 1)
    for obj in existing:
        if _bboxes_overlap(grown, _expand_bbox(obj["bbox"], h, w, 1)):
            return False
    return True


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == BG:
                continue
            color = int(g[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            rr0, cc0, rr1, cc1 = bbox
            if not np.all(g[rr0 : rr1 + 1, cc0 : cc1 + 1] == color):
                raise AssertionError("every object must be a solid rectangle")
            objects.append({"color": color, "bbox": bbox})
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _render_input(objects: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        _paint_rect(grid, obj["bbox"], obj["color"])
    return grid


def _analyze_objects(
    objects: list[dict], h: int, w: int
) -> dict | None:
    if any(not _bbox_in_bounds(obj["bbox"], h, w) for obj in objects):
        return None
    for idx, obj in enumerate(objects):
        if _touch_kind(obj["bbox"], h, w) is None:
            return None
        if any(
            _bboxes_overlap(_expand_bbox(obj["bbox"], h, w, 1), _expand_bbox(other["bbox"], h, w, 1))
            for j, other in enumerate(objects)
            if j < idx
        ):
            return None

    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_color[int(obj["color"])].append(obj)
    moving_colors = [color for color, comps in by_color.items() if len(comps) > 1]
    if len(moving_colors) != 1:
        return None
    moving_color = int(moving_colors[0])
    if len(by_color[moving_color]) not in (2, 3):
        return None
    if any(len(comps) != 1 for color, comps in by_color.items() if color != moving_color):
        return None

    singletons = [obj for obj in objects if int(obj["color"]) != moving_color]
    repulsion = np.zeros((h, w), dtype=bool)
    for obj in singletons:
        repulsion[_expand_bbox(obj["bbox"], h, w, 1)[0] : _expand_bbox(obj["bbox"], h, w, 1)[2] + 1,
                  _expand_bbox(obj["bbox"], h, w, 1)[1] : _expand_bbox(obj["bbox"], h, w, 1)[3] + 1] = True

    mover_infos = []
    singleton_bboxes = [obj["bbox"] for obj in singletons]
    for obj in by_color[moving_color]:
        bbox = obj["bbox"]
        kind = _touch_kind(bbox, h, w)
        if kind is None:
            return None
        base_bbox = _base_target_bbox(bbox, kind)
        if not _bbox_in_bounds(base_bbox, h, w):
            return None
        if any(_bboxes_overlap(base_bbox, static_bbox) for static_bbox in singleton_bboxes):
            return None
        step5_bbox = base_bbox
        repel_steps = 0
        while _bbox_intersects_mask(repulsion, step5_bbox):
            repel_steps += 1
            if repel_steps > h + w:
                return None
            step5_bbox = _step_bbox(step5_bbox, kind, 1)
            if not _bbox_in_bounds(step5_bbox, h, w):
                return None
        mover_infos.append(
            {
                "color": moving_color,
                "bbox": bbox,
                "kind": kind,
                "base_bbox": base_bbox,
                "step5_bbox": step5_bbox,
                "repel_steps": repel_steps,
                "distance": abs(step5_bbox[0] - bbox[0]) + abs(step5_bbox[1] - bbox[1]),
            }
        )

    step5_conflicts = []
    ambiguous_tie = False
    tie_conflict_used = False
    for i in range(len(mover_infos)):
        for j in range(i + 1, len(mover_infos)):
            a = mover_infos[i]
            b = mover_infos[j]
            if not _bboxes_overlap(a["step5_bbox"], b["step5_bbox"]):
                continue
            step5_conflicts.append((i, j))
            if a["distance"] == b["distance"]:
                if a["bbox"][0] == b["bbox"][0]:
                    ambiguous_tie = True
                else:
                    tie_conflict_used = True
    if ambiguous_tie:
        return None

    placement_order = sorted(
        range(len(mover_infos)),
        key=lambda idx: (-mover_infos[idx]["distance"], mover_infos[idx]["bbox"][0], mover_infos[idx]["bbox"][1]),
    )
    placed_infos: list[dict] = []
    for idx in placement_order:
        info = mover_infos[idx]
        final_bbox = info["step5_bbox"]
        conflict_steps = 0
        while _bbox_intersects_mask(repulsion, final_bbox) or any(
            _bboxes_overlap(final_bbox, placed["final_bbox"]) for placed in placed_infos
        ):
            conflict_steps += 1
            if conflict_steps > h + w:
                return None
            final_bbox = _step_bbox(final_bbox, info["kind"], 1)
            if not _bbox_in_bounds(final_bbox, h, w):
                return None
        info = dict(info)
        info["final_bbox"] = final_bbox
        info["conflict_steps"] = conflict_steps
        placed_infos.append(info)

    final_grid = np.zeros((h, w), dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)
    for obj in singletons:
        if _bbox_intersects_mask(occupancy, obj["bbox"]):
            return None
        _paint_rect(final_grid, obj["bbox"], obj["color"])
        occupancy[obj["bbox"][0] : obj["bbox"][2] + 1, obj["bbox"][1] : obj["bbox"][3] + 1] = True
    for info in placed_infos:
        if _bbox_intersects_mask(occupancy, info["final_bbox"]):
            return None
        _paint_rect(final_grid, info["final_bbox"], moving_color)
        occupancy[
            info["final_bbox"][0] : info["final_bbox"][2] + 1,
            info["final_bbox"][1] : info["final_bbox"][3] + 1,
        ] = True

    return {
        "moving_color": moving_color,
        "singletons": singletons,
        "mover_infos": mover_infos,
        "placed_infos": placed_infos,
        "step5_conflicts": step5_conflicts,
        "repelled_count": sum(int(info["repel_steps"] > 0) for info in mover_infos),
        "conflict_shifted_count": sum(int(info["conflict_steps"] > 0) for info in placed_infos),
        "tie_conflict_used": tie_conflict_used,
        "output_grid": final_grid,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the repeated border rectangles inward while avoiding singleton dilation zones and cascading mover collisions.

    Input structure:
    - The grid has black background and 4 to 6 solid monochrome rectangles.
    - Exactly one foreground color appears as 2 or 3 separate rectangles; every other foreground color appears once.
    - Every rectangle touches exactly one border or one corner of the grid.

    Rule:
    - Identify the repeated-color rectangles as movers and all singleton-color rectangles as static obstacles.
    - Dilate every static rectangle by one cell in the 8-neighborhood to form an invisible repulsion field.
    - Move each repeated-color rectangle away from its touched border(s) by its own height and/or width, then keep stepping one cell farther along the same direction until it no longer overlaps the repulsion field.
    - If adjusted mover targets still overlap each other, keep the larger-distance mover in place first; lower-priority movers continue stepping one cell farther along their own directions until they avoid both the repulsion field and already placed movers.
    - Erase the movers from their original positions, redraw them at their resolved targets, and keep every singleton rectangle unchanged.

    Key insight:
    - The singleton rectangles matter only through their 8-neighborhood safety halos, and those halos can push a mover into another mover, so dilation, translation, and priority-based re-placement must be solved as one chained process.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _parse_objects(g)
    analysis = _analyze_objects(objects, *g.shape)
    assert analysis is not None, "invalid puzzle instance"
    return analysis["output_grid"]


def _search_core(h: int, w: int, require_tie: bool) -> tuple[list[dict], dict] | None:
    if require_tie:
        pool_a = _candidate_bboxes(h, w, ("left",))
        pool_b = _candidate_bboxes(h, w, ("TR",))
        pool_s = _candidate_bboxes(h, w, ("top",))
    else:
        pool_a = _candidate_bboxes(h, w, ("top",))
        pool_b = _candidate_bboxes(h, w, ("left",))
        pool_s = _candidate_bboxes(h, w, ("left", "right"))

    for _ in range(CORE_ATTEMPTS):
        kind_a, bbox_a = random.choice(pool_a)
        kind_b, bbox_b = random.choice(pool_b)
        kind_s, bbox_s = random.choice(pool_s)
        objects = [
            {"color": 1, "bbox": bbox_a, "kind": kind_a},
            {"color": 1, "bbox": bbox_b, "kind": kind_b},
            {"color": 4, "bbox": bbox_s, "kind": kind_s},
        ]
        analysis = _analyze_objects(objects, h, w)
        if analysis is None:
            continue
        if analysis["repelled_count"] < 1:
            continue
        if analysis["conflict_shifted_count"] < 1:
            continue
        if not analysis["step5_conflicts"]:
            continue
        if require_tie != bool(analysis["tie_conflict_used"]):
            continue
        return objects, analysis
    return None


def _with_real_colors(objects: list[dict], moving_color: int, singleton_colors: list[int]) -> list[dict]:
    out = []
    singleton_iter = iter(singleton_colors)
    for obj in objects:
        color = moving_color if int(obj["color"]) == 1 else next(singleton_iter)
        out.append({"color": color, "bbox": obj["bbox"]})
    return out


def _try_add_object(
    objects: list[dict],
    h: int,
    w: int,
    color: int,
    kinds: tuple[str, ...],
    require_tie: bool,
) -> tuple[list[dict], dict] | None:
    pool = _candidate_bboxes(h, w, kinds)
    random.shuffle(pool)
    for kind, bbox in pool[:ADD_ATTEMPTS]:
        candidate = {"color": color, "bbox": bbox}
        if not _separated_from_all(bbox, objects, h, w):
            continue
        new_objects = objects + [candidate]
        analysis = _analyze_objects(new_objects, h, w)
        if analysis is None:
            continue
        if analysis["repelled_count"] < 1 or analysis["conflict_shifted_count"] < 1:
            continue
        if require_tie != bool(analysis["tie_conflict_used"]):
            continue
        return new_objects, analysis
    return None


def _validate_scene(
    objects: list[dict], h: int, w: int, require_tie: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    if not (4 <= len(objects) <= 6):
        return None
    colors = Counter(int(obj["color"]) for obj in objects)
    repeated = [color for color, count in colors.items() if count > 1]
    if len(repeated) != 1 or colors[repeated[0]] not in (2, 3):
        return None
    analysis = _analyze_objects(objects, h, w)
    if analysis is None:
        return None
    if analysis["repelled_count"] < 1:
        return None
    if analysis["conflict_shifted_count"] < 1:
        return None
    if require_tie != bool(analysis["tie_conflict_used"]):
        return None

    inp = _render_input(objects, h, w)
    out = analysis["output_grid"]
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    require_tie = random.random() < 0.45
    moving_count = 3 if random.random() < 0.45 else 2
    target_objects = random.randint(5 if moving_count == 3 else 4, 6 if moving_count == 3 else 5)

    core = _search_core(h, w, require_tie=require_tie)
    if core is None:
        return None
    base_objects, _base_analysis = core

    moving_color = random.choice(MOVING_COLOR_CHOICES)
    singleton_needed = target_objects - moving_count
    if singleton_needed < 1:
        return None
    singleton_palette = random.sample(SINGLETON_COLOR_CHOICES, singleton_needed)
    objects = _with_real_colors(base_objects, moving_color, singleton_palette[:1])
    used_singleton_colors = {objects[-1]["color"]}

    if moving_count == 3:
        added = _try_add_object(objects, h, w, moving_color, ("bottom", "BL", "BR", "right"), require_tie)
        if added is None:
            return None
        objects, _analysis = added

    remaining_singletons = singleton_needed - len(used_singleton_colors)
    available_singletons = [c for c in singleton_palette if c not in used_singleton_colors]
    for color in available_singletons[:remaining_singletons]:
        added = _try_add_object(
            objects,
            h,
            w,
            color,
            ("top", "bottom", "left", "right", "TL", "TR", "BL", "BR"),
            require_tie,
        )
        if added is None:
            return None
        objects, _analysis = added

    validated = _validate_scene(objects, h, w, require_tie)
    if validated is None:
        return None
    inp, _out = validated
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
