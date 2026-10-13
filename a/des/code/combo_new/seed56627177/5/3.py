import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GRAY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SCENARIO_COLLISIONS = ("tie", "unequal")
SCENARIO_STOPS = ("edge", "wall", "body")
GEN_RETRIES = 400

ARROW_TEMPLATES = {
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}): {"dir": (-1, 0), "tip": (0, 1)},
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}): {"dir": (1, 0), "tip": (2, 1)},
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}): {"dir": (0, -1), "tip": (1, 0)},
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}): {"dir": (0, 1), "tip": (1, 2)},
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_monochrome_components(grid: np.ndarray, blocked: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in blocked:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    dr, dc = direction
    pr, pc = dc, -dr
    tip = (r + dr, c + dc)
    tail = (r - dr, c - dc)
    return (
        (r, c),
        tip,
        tail,
        (tail[0] + pr, tail[1] + pc),
        (tail[0] - pr, tail[1] - pc),
    )


def _make_arrow(color: int, center: tuple[int, int], direction: tuple[int, int]) -> dict:
    cells = tuple(sorted(_arrow_cells(center, direction)))
    tip = (center[0] + direction[0], center[1] + direction[1])
    return {"color": color, "center": center, "dir": direction, "tip": tip, "cells": cells}


def _expand_cells(shape: tuple[int, int], cells: tuple[tuple[int, int], ...], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(shape[0], r + radius + 1)):
            for cc in range(max(0, c - radius), min(shape[1], c + radius + 1)):
                out.add((rr, cc))
    return out


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _find_monochrome_components(grid, blocked={BLACK, GRAY}):
        coords = comp["coords"]
        if len(coords) != 5:
            raise AssertionError("every non-black non-gray component must be a 5-pixel arrow")
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 != 2 or c1 - c0 != 2:
            raise AssertionError("every arrow must fit in a 3x3 bounding box")
        normalized = frozenset((r - r0, c - c0) for r, c in coords)
        template = ARROW_TEMPLATES.get(normalized)
        if template is None:
            raise AssertionError("unrecognized arrow shape")
        tip = (r0 + template["tip"][0], c0 + template["tip"][1])
        arrows.append(
            {
                "color": int(comp["color"]),
                "dir": template["dir"],
                "tip": tip,
                "cells": tuple(sorted(coords)),
            }
        )
    if not arrows:
        raise AssertionError("expected at least one arrow")
    return arrows


def _trace_rays(grid: np.ndarray, arrows: list[dict]):
    body = {cell for arrow in arrows for cell in arrow["cells"]}
    hits: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    traces = []
    for idx, arrow in enumerate(arrows):
        dr, dc = arrow["dir"]
        rr = arrow["tip"][0] + dr
        cc = arrow["tip"][1] + dc
        dist = 1
        path = []
        reason = "edge"
        while _in_bounds(grid.shape, rr, cc):
            if (rr, cc) in body:
                reason = "body"
                break
            if int(grid[rr, cc]) == GRAY:
                reason = "wall"
                break
            if int(grid[rr, cc]) != BLACK:
                raise AssertionError("ray path may only encounter black, gray, or arrow cells")
            path.append((rr, cc))
            hits[(rr, cc)].append((dist, idx))
            rr += dr
            cc += dc
            dist += 1
        traces.append({"path": path, "reason": reason, "arrow_index": idx})
    return hits, traces


def _layout_metrics(grid: np.ndarray) -> dict:
    arrows = _detect_arrows(grid)
    hits, traces = _trace_rays(grid, arrows)
    tie = False
    collision = False
    for entries in hits.values():
        if len(entries) < 2:
            continue
        best = min(dist for dist, _idx in entries)
        winners = sum(dist == best for dist, _idx in entries)
        if winners >= 2:
            tie = True
        else:
            collision = True
    reasons = {
        "edge": any(trace["reason"] == "edge" and trace["path"] for trace in traces),
        "wall": any(trace["reason"] == "wall" and trace["path"] for trace in traces),
        "body": any(trace["reason"] == "body" and trace["path"] for trace in traces),
    }
    return {
        "tie": tie,
        "unequal": collision,
        "edge": reasons["edge"],
        "wall": reasons["wall"],
        "body": reasons["body"],
        "painted": sum(len(trace["path"]) for trace in traces),
        "arrow_count": len(arrows),
        "wave_cells": len(hits),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate straight color ripples from each arrow tip, resolving overlaps by source distance.

    Input structure:
    - The background is black.
    - Several monochrome 5-pixel arrows appear in the grid; each arrow is a 3x3 cardinal arrow made of one center pixel, one tip pixel, and a 3-pixel tail bar.
    - Optional gray wall pixels may appear on the grid edge, and every other non-black non-gray pixel belongs to an arrow body.

    Rule:
    - Keep every arrow body and every gray wall pixel unchanged.
    - For each arrow, start from the cell just beyond its tip and move straight along the arrow direction, collecting consecutive black cells until the ray would leave the grid or hit a gray wall or any arrow-body pixel.
    - A black cell reached by exactly one nearest source is recolored to that arrow's color; if multiple arrows reach the same cell at the same minimum distance from their own tips, that cell stays black.

    Key insight:
    - The arrow geometry fully determines each source tip and firing direction, and overlap resolution depends only on comparing per-ray distances to the same target cell rather than on any color priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    arrows = _detect_arrows(g)
    hits, _traces = _trace_rays(g, arrows)

    out = g.copy()
    for cell, entries in hits.items():
        best = min(dist for dist, _idx in entries)
        winners = [idx for dist, idx in entries if dist == best]
        if len(winners) == 1:
            out[cell] = arrows[winners[0]]["color"]
    return out


def _can_place_specs(
    shape: tuple[int, int],
    specs: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
) -> bool:
    new_cells = []
    for spec in specs:
        for cell in spec["cells"]:
            if not _in_bounds(shape, *cell):
                return False
            if cell in body_cells or cell in walls or cell in reserved_black or cell in halo_cells:
                return False
            new_cells.append(cell)
    if len(set(new_cells)) != len(new_cells):
        return False
    for i, spec_a in enumerate(specs):
        halo_a = _expand_cells(shape, spec_a["cells"], radius=1)
        for spec_b in specs[i + 1 :]:
            if halo_a & set(spec_b["cells"]):
                return False
    return True


def _add_specs(
    shape: tuple[int, int],
    specs: list[dict],
    arrows: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
) -> None:
    for spec in specs:
        arrows.append(spec)
        body_cells.update(spec["cells"])
        halo_cells.update(_expand_cells(shape, spec["cells"], radius=1))


def _render_grid(shape: tuple[int, int], arrows: list[dict], walls: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in walls:
        grid[r, c] = GRAY
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = arrow["color"]
    return grid


def _reserve_paths(
    shape: tuple[int, int],
    arrows: list[dict],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
    start_index: int,
) -> list[dict]:
    temp = _render_grid(shape, arrows, walls)
    _hits, traces = _trace_rays(temp, arrows)
    for trace in traces[start_index:]:
        reserved_black.update(trace["path"])
        arrow = arrows[trace["arrow_index"]]
        if trace["reason"] == "edge":
            dr, dc = arrow["dir"]
            rr = arrow["tip"][0] + dr
            cc = arrow["tip"][1] + dc
            while _in_bounds(shape, rr, cc) and int(temp[rr, cc]) == BLACK:
                rr += dr
                cc += dc
            if not _in_bounds(shape, rr, cc):
                reserved_no_wall.add((rr - dr, cc - dc))
    return traces


def _choose_scenarios() -> set[str]:
    selected = {random.choice(SCENARIO_COLLISIONS)}
    stop_count = random.randint(1, 2)
    selected.update(random.sample(SCENARIO_STOPS, k=stop_count))
    if random.random() < 0.35:
        selected.update(SCENARIO_COLLISIONS)
    return selected


def _place_opposed_pair(
    shape: tuple[int, int],
    parity: str,
    colors: list[int],
    arrows: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(120):
        axis = random.choice(("h", "v"))
        if axis == "h":
            valid_d = [d for d in range(4, w - 4) if (d % 2 == 0) == (parity == "tie")]
            if not valid_d:
                continue
            dist = random.choice(valid_d)
            max_tip = w - 3 - dist
            if max_tip < 2:
                continue
            row = random.randint(2, h - 3)
            tip_a = (row, random.randint(2, max_tip))
            tip_b = (row, tip_a[1] + dist)
            specs = [
                _make_arrow(colors.pop(), (tip_a[0], tip_a[1] - 1), (0, 1)),
                _make_arrow(colors.pop(), (tip_b[0], tip_b[1] + 1), (0, -1)),
            ]
        else:
            valid_d = [d for d in range(4, h - 4) if (d % 2 == 0) == (parity == "tie")]
            if not valid_d:
                continue
            dist = random.choice(valid_d)
            max_tip = h - 3 - dist
            if max_tip < 2:
                continue
            col = random.randint(2, w - 3)
            tip_a = (random.randint(2, max_tip), col)
            tip_b = (tip_a[0] + dist, col)
            specs = [
                _make_arrow(colors.pop(), (tip_a[0] - 1, tip_a[1]), (1, 0)),
                _make_arrow(colors.pop(), (tip_b[0] + 1, tip_b[1]), (-1, 0)),
            ]
        colors.extend([])
        if not _can_place_specs(shape, specs, body_cells, halo_cells, walls, reserved_black):
            colors.extend([spec["color"] for spec in specs])
            continue
        start = len(arrows)
        _add_specs(shape, specs, arrows, body_cells, halo_cells)
        traces = _reserve_paths(shape, arrows, walls, reserved_black, reserved_no_wall, start)
        paths = [set(trace["path"]) for trace in traces[start:]]
        if not all(path for path in paths):
            del arrows[start:]
            for spec in specs:
                for cell in spec["cells"]:
                    body_cells.remove(cell)
            halo_cells.clear()
            for arrow in arrows:
                halo_cells.update(_expand_cells(shape, arrow["cells"], radius=1))
            reserved_black.clear()
            reserved_no_wall.clear()
            _rebuild_reservations(shape, arrows, walls, reserved_black, reserved_no_wall)
            colors.extend([spec["color"] for spec in specs])
            continue
        overlap = paths[0] & paths[1]
        if not overlap:
            del arrows[start:]
            for spec in specs:
                for cell in spec["cells"]:
                    body_cells.remove(cell)
            halo_cells.clear()
            for arrow in arrows:
                halo_cells.update(_expand_cells(shape, arrow["cells"], radius=1))
            reserved_black.clear()
            reserved_no_wall.clear()
            _rebuild_reservations(shape, arrows, walls, reserved_black, reserved_no_wall)
            colors.extend([spec["color"] for spec in specs])
            continue
        return True
    return False


def _place_same_direction_pair(
    shape: tuple[int, int],
    colors: list[int],
    arrows: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
) -> bool:
    h, w = shape
    choices = [
        ("right", (0, 1)),
        ("left", (0, -1)),
        ("down", (1, 0)),
        ("up", (-1, 0)),
    ]
    for _ in range(120):
        name, direction = random.choice(choices)
        if name == "right":
            dist = random.randint(5, max(5, w - 6))
            max_tip = w - 2 - dist
            if max_tip < 2:
                continue
            row = random.randint(2, h - 3)
            rear_tip = (row, random.randint(2, max_tip))
            front_tip = (row, rear_tip[1] + dist)
            specs = [
                _make_arrow(colors.pop(), (rear_tip[0], rear_tip[1] - 1), direction),
                _make_arrow(colors.pop(), (front_tip[0], front_tip[1] - 1), direction),
            ]
        elif name == "left":
            dist = random.randint(5, max(5, w - 6))
            max_front = w - 3 - dist
            if max_front < 2:
                continue
            row = random.randint(2, h - 3)
            front_tip = (row, random.randint(2, max_front))
            rear_tip = (row, front_tip[1] + dist)
            specs = [
                _make_arrow(colors.pop(), (rear_tip[0], rear_tip[1] + 1), direction),
                _make_arrow(colors.pop(), (front_tip[0], front_tip[1] + 1), direction),
            ]
        elif name == "down":
            dist = random.randint(5, max(5, h - 6))
            max_tip = h - 2 - dist
            if max_tip < 2:
                continue
            col = random.randint(2, w - 3)
            rear_tip = (random.randint(2, max_tip), col)
            front_tip = (rear_tip[0] + dist, col)
            specs = [
                _make_arrow(colors.pop(), (rear_tip[0] - 1, rear_tip[1]), direction),
                _make_arrow(colors.pop(), (front_tip[0] - 1, front_tip[1]), direction),
            ]
        else:
            dist = random.randint(5, max(5, h - 6))
            max_front = h - 3 - dist
            if max_front < 2:
                continue
            col = random.randint(2, w - 3)
            front_tip = (random.randint(2, max_front), col)
            rear_tip = (front_tip[0] + dist, col)
            specs = [
                _make_arrow(colors.pop(), (rear_tip[0] + 1, rear_tip[1]), direction),
                _make_arrow(colors.pop(), (front_tip[0] + 1, front_tip[1]), direction),
            ]
        if not _can_place_specs(shape, specs, body_cells, halo_cells, walls, reserved_black):
            colors.extend([spec["color"] for spec in specs])
            continue
        start = len(arrows)
        _add_specs(shape, specs, arrows, body_cells, halo_cells)
        traces = _reserve_paths(shape, arrows, walls, reserved_black, reserved_no_wall, start)
        if not any(trace["reason"] == "body" and trace["path"] for trace in traces[start:]):
            del arrows[start:]
            for spec in specs:
                for cell in spec["cells"]:
                    body_cells.remove(cell)
            halo_cells.clear()
            for arrow in arrows:
                halo_cells.update(_expand_cells(shape, arrow["cells"], radius=1))
            reserved_black.clear()
            reserved_no_wall.clear()
            _rebuild_reservations(shape, arrows, walls, reserved_black, reserved_no_wall)
            colors.extend([spec["color"] for spec in specs])
            continue
        return True
    return False


def _edge_segment(side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if side in {"top", "bottom"}:
        length = random.randint(1, max(1, min(5, w - 4)))
        start = random.randint(2, w - 2 - length)
        row = 0 if side == "top" else h - 1
        return [(row, c) for c in range(start, start + length)]
    length = random.randint(1, max(1, min(5, h - 4)))
    start = random.randint(2, h - 2 - length)
    col = 0 if side == "left" else w - 1
    return [(r, col) for r in range(start, start + length)]


def _place_wall_or_edge_arrow(
    kind: str,
    shape: tuple[int, int],
    colors: list[int],
    arrows: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(120):
        side = random.choice(("top", "bottom", "left", "right"))
        new_walls = []
        target = None
        if kind == "wall":
            segment = _edge_segment(side, shape)
            if any(cell in body_cells or cell in reserved_black or cell in reserved_no_wall for cell in segment):
                continue
            target = random.choice(segment)
            new_walls = segment
        else:
            if side == "top":
                target = (0, random.randint(2, w - 3))
            elif side == "bottom":
                target = (h - 1, random.randint(2, w - 3))
            elif side == "left":
                target = (random.randint(2, h - 3), 0)
            else:
                target = (random.randint(2, h - 3), w - 1)
            if target in walls or target in reserved_no_wall:
                continue
        if side == "top":
            direction = (-1, 0)
            tip = (random.randint(2, h - 3), target[1])
            center = (tip[0] + 1, tip[1])
        elif side == "bottom":
            direction = (1, 0)
            tip = (random.randint(2, h - 3), target[1])
            center = (tip[0] - 1, tip[1])
        elif side == "left":
            direction = (0, -1)
            tip = (target[0], random.randint(2, w - 3))
            center = (tip[0], tip[1] + 1)
        else:
            direction = (0, 1)
            tip = (target[0], random.randint(2, w - 3))
            center = (tip[0], tip[1] - 1)
        spec = _make_arrow(colors.pop(), center, direction)
        if not _can_place_specs(shape, [spec], body_cells, halo_cells, walls | set(new_walls), reserved_black):
            colors.append(spec["color"])
            continue
        if any(cell in reserved_no_wall for cell in new_walls):
            colors.append(spec["color"])
            continue
        start = len(arrows)
        prior_wall_count = len(walls)
        walls.update(new_walls)
        _add_specs(shape, [spec], arrows, body_cells, halo_cells)
        traces = _reserve_paths(shape, arrows, walls, reserved_black, reserved_no_wall, start)
        good = any(trace["reason"] == kind and trace["path"] for trace in traces[start:])
        if good:
            return True
        del arrows[start:]
        for cell in spec["cells"]:
            body_cells.remove(cell)
        while len(walls) > prior_wall_count:
            walls.pop()
        halo_cells.clear()
        for arrow in arrows:
            halo_cells.update(_expand_cells(shape, arrow["cells"], radius=1))
        reserved_black.clear()
        reserved_no_wall.clear()
        _rebuild_reservations(shape, arrows, walls, reserved_black, reserved_no_wall)
        colors.append(spec["color"])
    return False


def _place_random_arrow(
    shape: tuple[int, int],
    colors: list[int],
    arrows: list[dict],
    body_cells: set[tuple[int, int]],
    halo_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(80):
        direction = random.choice(DIRS4)
        center = (random.randint(2, h - 3), random.randint(2, w - 3))
        spec = _make_arrow(colors.pop(), center, direction)
        if _can_place_specs(shape, [spec], body_cells, halo_cells, walls, reserved_black):
            _add_specs(shape, [spec], arrows, body_cells, halo_cells)
            return True
        colors.append(spec["color"])
    return False


def _add_extra_walls(
    shape: tuple[int, int],
    walls: set[tuple[int, int]],
    body_cells: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
) -> None:
    for _ in range(random.randint(0, 2)):
        for _ in range(40):
            segment = _edge_segment(random.choice(("top", "bottom", "left", "right")), shape)
            if any(cell in body_cells or cell in reserved_black or cell in reserved_no_wall for cell in segment):
                continue
            walls.update(segment)
            break


def _rebuild_reservations(
    shape: tuple[int, int],
    arrows: list[dict],
    walls: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    reserved_no_wall: set[tuple[int, int]],
) -> None:
    temp = _render_grid(shape, arrows, walls)
    _hits, traces = _trace_rays(temp, arrows)
    for trace in traces:
        reserved_black.update(trace["path"])
        arrow = arrows[trace["arrow_index"]]
        if trace["reason"] == "edge":
            dr, dc = arrow["dir"]
            rr = arrow["tip"][0] + dr
            cc = arrow["tip"][1] + dc
            while _in_bounds(shape, rr, cc) and int(temp[rr, cc]) == BLACK:
                rr += dr
                cc += dc
            if not _in_bounds(shape, rr, cc):
                reserved_no_wall.add((rr - dr, cc - dc))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    shape = (h, w)
    colors = list(ARROW_COLORS)
    random.shuffle(colors)
    arrows: list[dict] = []
    body_cells: set[tuple[int, int]] = set()
    halo_cells: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    reserved_black: set[tuple[int, int]] = set()
    reserved_no_wall: set[tuple[int, int]] = set()

    selected = _choose_scenarios()
    if "tie" in selected and not _place_opposed_pair(
        shape, "tie", colors, arrows, body_cells, halo_cells, walls, reserved_black, reserved_no_wall
    ):
        return None
    if "unequal" in selected and not _place_opposed_pair(
        shape, "unequal", colors, arrows, body_cells, halo_cells, walls, reserved_black, reserved_no_wall
    ):
        return None
    if "body" in selected and not _place_same_direction_pair(
        shape, colors, arrows, body_cells, halo_cells, walls, reserved_black, reserved_no_wall
    ):
        return None
    if "wall" in selected and not _place_wall_or_edge_arrow(
        "wall", shape, colors, arrows, body_cells, halo_cells, walls, reserved_black, reserved_no_wall
    ):
        return None
    if "edge" in selected and not _place_wall_or_edge_arrow(
        "edge", shape, colors, arrows, body_cells, halo_cells, walls, reserved_black, reserved_no_wall
    ):
        return None

    _add_extra_walls(shape, walls, body_cells, reserved_black, reserved_no_wall)

    extra_arrows = random.randint(0, min(2, len(colors)))
    for _ in range(extra_arrows):
        if not _place_random_arrow(shape, colors, arrows, body_cells, halo_cells, walls, reserved_black):
            break

    grid = _render_grid(shape, arrows, walls)
    metrics = _layout_metrics(grid)
    if metrics["arrow_count"] < 2 or metrics["painted"] < 5 or metrics["wave_cells"] < 4:
        return None
    if any(not metrics[name] for name in selected):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
