from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 200
GRID_RANGE = (18, 29)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _bbox_from_points(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return (min(rows), min(cols), max(rows), max(cols))


def _rect_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _translate_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1, c1 - c0 + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _chebyshev_clear(cell: tuple[int, int], occupied: set[tuple[int, int]], radius: int = 1) -> bool:
    r, c = cell
    for rr in range(r - radius, r + radius + 1):
        for cc in range(c - radius, c + radius + 1):
            if (rr, cc) in occupied:
                return False
    return True


def _dilate8(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr, dc in DIR8:
            out.add((r + dr, c + dc))
    return out


def _adjacent_portals(
    footprint: set[tuple[int, int]],
    portals: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    hits: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, (p0, p1) in portals.items():
        for src, dst in ((p0, p1), (p1, p0)):
            if any(nei in footprint for nei in _neighbors4(src[0], src[1])):
                hits.append((color, src, dst))
    return hits


def _grow_object(
    cells: dict[tuple[int, int], int],
    border_color: int,
    h: int,
    w: int,
) -> dict[tuple[int, int], int] | None:
    footprint = set(cells)
    grown = _dilate8(footprint)
    for r, c in grown:
        if not (0 <= r < h and 0 <= c < w):
            return None
    out = dict(cells)
    for pos in grown - footprint:
        out[pos] = border_color
    return out


def _simulate(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps = _components(g, bg)

    portal_candidates: dict[int, list[tuple[int, int]]] = {}
    objects: list[dict] = []

    for comp in comps:
        if len(comp["coords"]) == 1:
            portal_candidates.setdefault(int(comp["color"]), []).append(comp["coords"][0])
            continue
        cells = {pos: int(comp["color"]) for pos in comp["coords"]}
        objects.append(
            {
                "cells": cells,
                "initial_bbox": comp["bbox"],
                "steps": 0,
                "stop_reason": "none",
            }
        )

    portals = {
        int(color): (tuple(points[0]), tuple(points[1]))
        for color, points in portal_candidates.items()
        if len(points) == 2
    }
    objects.sort(key=lambda obj: obj["initial_bbox"])

    for idx, obj in enumerate(objects):
        while True:
            footprint = set(obj["cells"])
            matches = _adjacent_portals(footprint, portals)
            if len(matches) != 1:
                obj["stop_reason"] = "no_portal" if len(matches) == 0 else "ambiguous"
                break

            portal_color, src, dst = matches[0]
            dr = dst[0] - src[0]
            dc = dst[1] - src[1]

            shifted: dict[tuple[int, int], int] = {}
            failed = False
            for (r, c), color in obj["cells"].items():
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    failed = True
                    break
                shifted[(rr, cc)] = color

            if not failed:
                shifted_points = set(shifted)
                for other_idx, other in enumerate(objects):
                    if other_idx == idx:
                        continue
                    if shifted_points & set(other["cells"]):
                        failed = True
                        break

            if not failed:
                grown = _grow_object(shifted, portal_color, h, w)
                if grown is None:
                    failed = True
                else:
                    grown_points = set(grown)
                    for other_idx, other in enumerate(objects):
                        if other_idx == idx:
                            continue
                        if grown_points & set(other["cells"]):
                            failed = True
                            break

            if failed:
                obj["stop_reason"] = "blocked"
                break

            del portals[portal_color]
            obj["cells"] = grown
            obj["steps"] += 1

        obj["final_bbox"] = _bbox_from_points(set(obj["cells"]))

    out = np.full((h, w), bg, dtype=int)
    for _color, (p0, p1) in portals.items():
        out[p0] = _color
        out[p1] = _color
    for obj in objects:
        for (r, c), color in obj["cells"].items():
            out[r, c] = color

    return {"output": out, "objects": objects, "remaining_portals": portals, "bg": bg}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each adjacent object along matched portal pairs and add a portal-colored border after every successful jump.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _side_point(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return (r0 - 1, random.randint(c0, c1))
    if side == "bottom":
        return (r1 + 1, random.randint(c0, c1))
    if side == "left":
        return (random.randint(r0, r1), c0 - 1)
    return (random.randint(r0, r1), c1 + 1)


def _valid_side(box: tuple[int, int, int, int], side: str, h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0 > 0
    if side == "bottom":
        return r1 < h - 1
    if side == "left":
        return c0 > 0
    return c1 < w - 1


def _sample_success_target(
    current: tuple[int, int, int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    local_boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int, int, int], int, int] | None:
    height, width = _box_size(current)
    min_r0 = 1
    max_r0 = h - height - 1
    min_c0 = 1
    max_c0 = w - width - 1
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None

    for _ in range(80):
        nr0 = random.randint(min_r0, max_r0)
        nc0 = random.randint(min_c0, max_c0)
        dr = nr0 - current[0]
        dc = nc0 - current[1]
        if abs(dr) + abs(dc) < 3:
            continue
        translated = _translate_box(current, dr, dc)
        grown = _expand_box(translated, 1)
        if any(_boxes_overlap(grown, other, pad=2) for other in existing_boxes):
            continue
        if any(_boxes_overlap(grown, other, pad=1) for other in local_boxes):
            continue
        return translated, dr, dc
    return None


def _sample_blocked_target(
    current: tuple[int, int, int, int],
    side: str,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    height, width = _box_size(current)
    if side == "left":
        nc0 = random.randint(w - width + 1, w)
        return (random.randint(0, h - height), nc0 - current[1])
    if side == "right":
        nc0 = -random.randint(1, width)
        return (random.randint(0, h - height), nc0 - current[1])
    if side == "top":
        nr0 = random.randint(h - height + 1, h)
        return (nr0 - current[0], random.randint(0, w - width) - current[1])
    nr0 = -random.randint(1, height)
    return (nr0 - current[0], random.randint(0, w - width) - current[1])


def _pick_branch_configs(n_objects: int) -> list[dict]:
    branch_names = ["short", "chain", "blocked0", "blocked1"]
    for _ in range(100):
        chosen = random.sample(branch_names, n_objects)
        configs: list[dict] = []
        for name in chosen:
            if name == "short":
                configs.append({"name": name, "successes": 1, "blocked": False})
            elif name == "chain":
                configs.append({"name": name, "successes": random.choice([2, 2, 3]), "blocked": False})
            elif name == "blocked0":
                configs.append({"name": name, "successes": 0, "blocked": True})
            else:
                configs.append({"name": name, "successes": random.choice([1, 2]), "blocked": True})
        portal_count = sum(cfg["successes"] + int(cfg["blocked"]) for cfg in configs)
        if portal_count <= 9 - n_objects:
            return configs
    raise RuntimeError("failed to sample branch configs")


def _build_object_spec(
    h: int,
    w: int,
    object_color: int,
    portal_colors: list[int],
    branch: dict,
    existing_boxes: list[tuple[int, int, int, int]],
    occupied_input: set[tuple[int, int]],
) -> dict | None:
    total_portals = branch["successes"] + int(branch["blocked"])
    assert total_portals == len(portal_colors)

    for _ in range(OBJECT_ATTEMPTS):
        base_h, base_w = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        if h - base_h - 2 < 1 or w - base_w - 2 < 1:
            return None

        r0 = random.randint(1, h - base_h - 2)
        c0 = random.randint(1, w - base_w - 2)
        initial_box = (r0, c0, r0 + base_h - 1, c0 + base_w - 1)
        if any(_boxes_overlap(initial_box, other, pad=2) for other in existing_boxes):
            continue
        initial_cells = _rect_cells(initial_box)
        if any(not _chebyshev_clear(cell, occupied_input, 1) for cell in initial_cells):
            continue

        local_boxes = [initial_box]
        local_input = set(initial_cells)
        current = initial_box
        portals: list[dict] = []
        ok = True

        for step_idx in range(total_portals):
            needs_success = step_idx < branch["successes"]
            side_choices = [side for side in ("top", "bottom", "left", "right") if _valid_side(current, side, h, w)]
            random.shuffle(side_choices)
            placed = False

            for side in side_choices:
                src = _side_point(current, side)
                src_blockers = (occupied_input | local_input) - (initial_cells if current == initial_box else set())
                if src in local_input or not _chebyshev_clear(src, src_blockers, 1):
                    continue

                if needs_success:
                    target = _sample_success_target(current, existing_boxes, local_boxes, h, w)
                    if target is None:
                        continue
                    translated_box, dr, dc = target
                else:
                    blocked_delta = _sample_blocked_target(current, side, h, w)
                    if blocked_delta is None:
                        continue
                    dr, dc = blocked_delta
                    translated_box = _translate_box(current, dr, dc)

                dst = (src[0] + dr, src[1] + dc)
                if not (0 <= dst[0] < h and 0 <= dst[1] < w):
                    continue
                if dst in local_input or not _chebyshev_clear(dst, occupied_input | local_input, 1):
                    continue

                portals.append(
                    {
                        "color": portal_colors[step_idx],
                        "src": src,
                        "dst": dst,
                        "success": needs_success,
                    }
                )
                local_input.add(src)
                local_input.add(dst)
                if needs_success:
                    current = _expand_box(translated_box, 1)
                    local_boxes.append(current)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        return {
            "object_color": object_color,
            "initial_box": initial_box,
            "portals": portals,
            "expected_steps": branch["successes"],
            "expected_stop": "blocked" if branch["blocked"] else "no_portal",
            "stage_boxes": local_boxes,
            "input_cells": local_input,
        }
    return None


def _assemble_grid(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        for cell in _rect_cells(spec["initial_box"]):
            grid[cell] = spec["object_color"]
        for portal in spec["portals"]:
            grid[portal["src"]] = portal["color"]
            grid[portal["dst"]] = portal["color"]
    return grid


def _matches_expectation(specs: list[dict], analysis: dict) -> bool:
    expected = sorted(
        [(spec["initial_box"], spec["expected_steps"], spec["expected_stop"]) for spec in specs],
        key=lambda item: item[0],
    )
    actual = sorted(
        [(obj["initial_bbox"], obj["steps"], obj["stop_reason"]) for obj in analysis["objects"]],
        key=lambda item: item[0],
    )
    return expected == actual


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    n_objects = random.choice([2, 3])
    branches = _pick_branch_configs(n_objects)
    total_portals = sum(cfg["successes"] + int(cfg["blocked"]) for cfg in branches)
    colors = random.sample(PALETTE, n_objects + total_portals)

    specs: list[dict] = []
    existing_boxes: list[tuple[int, int, int, int]] = []
    occupied_input: set[tuple[int, int]] = set()
    color_idx = 0

    for branch in branches:
        object_color = colors[color_idx]
        color_idx += 1
        portal_count = branch["successes"] + int(branch["blocked"])
        portal_colors = colors[color_idx : color_idx + portal_count]
        color_idx += portal_count
        spec = _build_object_spec(h, w, object_color, portal_colors, branch, existing_boxes, occupied_input)
        if spec is None:
            return None
        specs.append(spec)
        existing_boxes.extend(spec["stage_boxes"])
        occupied_input |= spec["input_cells"]

    grid = _assemble_grid(h, w, specs)
    analysis = _simulate(grid)
    if not _matches_expectation(specs, analysis):
        return None
    if len({obj["steps"] for obj in analysis["objects"]}) < 2:
        return None
    if not any(obj["stop_reason"] == "blocked" for obj in analysis["objects"]) and not any(
        obj["steps"] >= 2 for obj in analysis["objects"]
    ):
        return None
    if np.array_equal(grid, analysis["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-chain puzzle")
