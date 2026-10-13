import random
from collections import deque

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPENDICULARS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}
ARROW_REL = {
    (0, 1): [(-1, 1), (0, -1), (0, 0), (0, 1), (1, 1)],
    (0, -1): [(-1, -1), (0, -1), (0, 0), (0, 1), (1, -1)],
    (-1, 0): [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    (1, 0): [(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)],
}
ARROW_MASK_TO_DIR = {
    frozenset((r + 1, c + 1) for r, c in rels): direction
    for direction, rels in ARROW_REL.items()
}

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
GAP_RANGE = (1, 4)
GEN_RETRIES = 400
PATH_RETRIES = 80


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _scale(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _in_bounds(pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _distance_to_edge(pos: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    h, w = shape
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return steps
        steps += 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in ARROW_REL[direction]]


def _arrow_exit(center: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return _add(center, direction)


def _prism_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _place_arrow_from(anchor: tuple[int, int], travel: tuple[int, int], gap: int) -> tuple[int, int]:
    return _add(anchor, _scale(travel, gap + 1))


def _place_prism_from(anchor: tuple[int, int], travel: tuple[int, int], gap: int) -> tuple[int, int]:
    hit = _add(anchor, _scale(travel, gap + 1))
    if travel == (0, 1):
        return hit
    if travel == (0, -1):
        return hit[0], hit[1] - 1
    if travel == (1, 0):
        return hit
    assert travel == (-1, 0)
    return hit[0] - 1, hit[1]


def _prism_far_anchor(top_left: tuple[int, int], travel: tuple[int, int]) -> tuple[int, int]:
    r, c = top_left
    if travel == (0, 1):
        return r, c + 1
    if travel == (0, -1):
        return r, c
    if travel == (1, 0):
        return r + 1, c
    assert travel == (-1, 0)
    return r, c


def _prism_far_cell(top_left: tuple[int, int], travel: tuple[int, int], hit: tuple[int, int]) -> tuple[int, int]:
    r, c = top_left
    hr, hc = hit
    if travel == (0, 1):
        return hr, c + 1
    if travel == (0, -1):
        return hr, c
    if travel == (1, 0):
        return r + 1, hc
    assert travel == (-1, 0)
    return r, hc


def _reserve(cells: list[tuple[int, int]], forbidden: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden.add((nr, nc))


def _can_place(cells: list[tuple[int, int]], forbidden: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for cell in cells:
        if not _in_bounds(cell, shape):
            return False
        if cell in forbidden:
            return False
    if any(r in (0, h - 1) and c in (0, w - 1) for r, c in cells):
        return False
    return True


def _draw_arrow(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], color: int) -> None:
    for r, c in _arrow_cells(center, direction):
        grid[r, c] = color


def _draw_prism(grid: np.ndarray, top_left: tuple[int, int], top: int, bottom: int) -> None:
    r, c = top_left
    grid[r, c : c + 2] = top
    grid[r + 1, c : c + 2] = bottom


def _source_for_direction(shape: tuple[int, int], travel: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if travel == (0, 1):
        return random.randint(2, h - 3), random.choice([0, 1])
    if travel == (0, -1):
        return random.randint(2, h - 3), random.choice([w - 2, w - 1])
    if travel == (1, 0):
        return random.choice([0, 1]), random.randint(2, w - 3)
    assert travel == (-1, 0)
    return random.choice([h - 2, h - 1]), random.randint(2, w - 3)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cell_to_obj: dict[tuple[int, int], tuple[str, int]] = {}
    arrows = []
    prisms = []
    sources = []
    unknown = []

    for coords in _components(g != BG):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = len(coords)
        colors = {int(g[r, c]) for r, c in coords}

        if area == 1:
            color = int(g[coords[0]])
            index = len(sources)
            obj = {"pos": coords[0], "color": color}
            sources.append(obj)
            cell_to_obj[coords[0]] = ("source", index)
            continue

        if (r1 - r0, c1 - c0, area, len(colors)) == (2, 2, 5, 1):
            rel = frozenset((r - r0, c - c0) for r, c in coords)
            direction = ARROW_MASK_TO_DIR.get(rel)
            if direction is not None:
                index = len(arrows)
                obj = {
                    "center": (r0 + 1, c0 + 1),
                    "direction": direction,
                    "color": int(next(iter(colors))),
                    "bbox": (r0, c0, r1, c1),
                }
                arrows.append(obj)
                for cell in coords:
                    cell_to_obj[cell] = ("arrow", index)
                continue

        if (r1 - r0, c1 - c0, area) == (1, 1, 4):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            top = int(patch[0, 0])
            bottom = int(patch[1, 0])
            if np.all(patch[0] == top) and np.all(patch[1] == bottom) and top != bottom:
                index = len(prisms)
                obj = {"top_left": (r0, c0), "top": top, "bottom": bottom}
                prisms.append(obj)
                for cell in coords:
                    cell_to_obj[cell] = ("prism", index)
                continue

        unknown.append(coords)

    assert not unknown, "unexpected object shape in input"
    assert arrows and prisms and sources, "scene must contain arrows, prisms, and sources"
    return {
        "grid": g,
        "shape": (h, w),
        "cell_to_obj": cell_to_obj,
        "arrows": arrows,
        "prisms": prisms,
        "sources": sorted(sources, key=lambda item: (item["pos"][0], item["pos"][1], item["color"])),
    }


def _initial_direction(scene: dict, source: dict) -> tuple[int, int]:
    g = scene["grid"]
    h, w = scene["shape"]
    r0, c0 = source["pos"]
    candidates: list[tuple[int, tuple[int, int]]] = []

    for direction in DIRS:
        dr, dc = direction
        rr, cc = r0, c0
        dist = 0
        while True:
            rr += dr
            cc += dc
            dist += 1
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(g[rr, cc]) == BG:
                continue
            obj = scene["cell_to_obj"].get((rr, cc))
            if obj is not None and obj[0] == "arrow":
                candidates.append((dist, direction))
            break

    assert candidates, "each source must see at least one arrow"
    candidates.sort(key=lambda item: item[0])
    assert len(candidates) == 1 or candidates[0][0] != candidates[1][0], "nearest arrow must be unique"
    return candidates[0][1]


def _trace_source(scene: dict, out: np.ndarray, source: dict) -> tuple[set[tuple[int, int]], int, int]:
    g = scene["grid"]
    h, w = scene["shape"]
    cell_to_obj = scene["cell_to_obj"]
    color = int(source["color"])
    direction = _initial_direction(scene, source)
    pos = source["pos"]
    painted: set[tuple[int, int]] = set()
    converted = 0
    blocked = 0
    seen: set[tuple[int, int, int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction[0], direction[1], color)
        assert state not in seen, "ray loop detected"
        seen.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not (0 <= nr < h and 0 <= nc < w):
            break

        if int(g[nr, nc]) == BG:
            out[nr, nc] = color
            painted.add((nr, nc))
            pos = (nr, nc)
            continue

        kind, index = cell_to_obj[(nr, nc)]
        if kind == "arrow":
            arrow = scene["arrows"][index]
            direction = arrow["direction"]
            pos = _arrow_exit(arrow["center"], direction)
            continue

        if kind == "prism":
            prism = scene["prisms"][index]
            if color == prism["top"]:
                color = prism["bottom"]
                converted += 1
                pos = _prism_far_cell(prism["top_left"], direction, (nr, nc))
                continue
            blocked += 1
            break

        break

    return painted, converted, blocked


def _simulate(grid: np.ndarray, collect_stats: bool = False):
    scene = _parse_scene(grid)
    out = np.asarray(grid, dtype=int).copy()

    if not collect_stats:
        for source in scene["sources"]:
            _trace_source(scene, out, source)
        return out

    total_painted: set[tuple[int, int]] = set()
    converted = 0
    blocked = 0
    overlap = False
    per_source_counts = []
    for source in scene["sources"]:
        painted, conv, stop = _trace_source(scene, out, source)
        overlap = overlap or bool(total_painted & painted)
        total_painted |= painted
        converted += conv
        blocked += stop
        per_source_counts.append(len(painted))

    return out, {
        "sources": len(scene["sources"]),
        "arrows": len(scene["arrows"]),
        "prisms": len(scene["prisms"]),
        "converted": converted,
        "blocked": blocked,
        "painted": len(total_painted),
        "per_source_counts": per_source_counts,
        "overlap": overlap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors toward aligned arrows; arrows redirect the ray, prisms map top-row color to bottom-row color.
    """
    return _simulate(np.asarray(grid, dtype=int), collect_stats=False)


def _draw_objects(grid: np.ndarray, objects: list[dict]) -> None:
    for obj in objects:
        if obj["kind"] == "source":
            r, c = obj["pos"]
            grid[r, c] = obj["color"]
        elif obj["kind"] == "arrow":
            _draw_arrow(grid, obj["center"], obj["direction"], obj["color"])
        else:
            _draw_prism(grid, obj["top_left"], obj["top"], obj["bottom"])


def _template_single(shape: tuple[int, int]) -> tuple[np.ndarray, dict] | None:
    source_color, mapped_color, arrow1_color, arrow2_color, stop_top, stop_bottom = random.sample(range(1, 10), 6)
    forbidden: set[tuple[int, int]] = set()
    for _ in range(PATH_RETRIES):
        first_dir = random.choice(DIRS)
        turn1 = random.choice(PERPENDICULARS[first_dir])
        turn2 = random.choice(PERPENDICULARS[turn1])
        steps = [
            {"kind": "arrow", "direction": turn1, "color": arrow1_color},
            {"kind": "prism", "top": source_color, "bottom": mapped_color},
            {"kind": "arrow", "direction": turn2, "color": arrow2_color},
            {"kind": "prism", "top": stop_top, "bottom": stop_bottom},
        ]
        built = _build_path_with_first_dir(shape, forbidden, source_color, first_dir, steps)
        if built is None:
            continue
        objects, _, _current, _travel = built
        grid = np.zeros(shape, dtype=int)
        _draw_objects(grid, objects)
        return grid, {"sources": 1, "arrows": 2, "prisms": 2}
    return None


def _build_path_with_first_dir(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    source_color: int,
    first_dir: tuple[int, int],
    steps: list[dict],
):
    local_forbidden = set(forbidden)
    source = _source_for_direction(shape, first_dir)
    if not _can_place([source], local_forbidden, shape):
        return None

    objects = [{"kind": "source", "pos": source, "color": source_color}]
    _reserve([source], local_forbidden, shape)
    current = source
    travel = first_dir

    for step in steps:
        gap = random.randint(*GAP_RANGE)
        if step["kind"] == "arrow":
            center = _place_arrow_from(current, travel, gap)
            cells = _arrow_cells(center, step["direction"])
            if not _can_place(cells, local_forbidden, shape):
                return None
            objects.append(
                {
                    "kind": "arrow",
                    "center": center,
                    "direction": step["direction"],
                    "color": step["color"],
                }
            )
            _reserve(cells, local_forbidden, shape)
            current = _arrow_exit(center, step["direction"])
            travel = step["direction"]
        else:
            top_left = _place_prism_from(current, travel, gap)
            cells = _prism_cells(top_left)
            if not _can_place(cells, local_forbidden, shape):
                return None
            objects.append(
                {
                    "kind": "prism",
                    "top_left": top_left,
                    "top": step["top"],
                    "bottom": step["bottom"],
                }
            )
            _reserve(cells, local_forbidden, shape)
            current = _prism_far_anchor(top_left, travel)

    return objects, local_forbidden, current, travel


def _template_double(shape: tuple[int, int]) -> tuple[np.ndarray, dict] | None:
    palette = random.sample(range(1, 10), 8)
    source1_color, mapped_color, arrow1_color, arrow2_color, source2_color, arrow3_color, stop_top, stop_bottom = palette

    for _ in range(PATH_RETRIES):
        forbidden: set[tuple[int, int]] = set()
        first_dir1 = random.choice(DIRS)
        turn1 = random.choice(PERPENDICULARS[first_dir1])
        turn2 = random.choice(PERPENDICULARS[turn1])
        steps1 = [
            {"kind": "arrow", "direction": turn1, "color": arrow1_color},
            {"kind": "prism", "top": source1_color, "bottom": mapped_color},
            {"kind": "arrow", "direction": turn2, "color": arrow2_color},
        ]
        built1 = _build_path_with_first_dir(shape, forbidden, source1_color, first_dir1, steps1)
        if built1 is None:
            continue
        objects1, forbidden1, current1, travel1 = built1
        if _distance_to_edge(current1, travel1, shape) < 2:
            continue

        for _ in range(PATH_RETRIES):
            start_dir2 = random.choice(DIRS)
            turn3 = random.choice(PERPENDICULARS[start_dir2])
            steps2 = [
                {"kind": "arrow", "direction": turn3, "color": arrow3_color},
                {"kind": "prism", "top": stop_top, "bottom": stop_bottom},
            ]
            built2 = _build_path_with_first_dir(shape, forbidden1, source2_color, start_dir2, steps2)
            if built2 is None:
                continue
            objects2, _, _current2, _travel2 = built2
            grid = np.zeros(shape, dtype=int)
            _draw_objects(grid, objects1 + objects2)
            return grid, {"sources": 2, "arrows": 3, "prisms": 2}

    return None


def _validate_generated(grid: np.ndarray, expected: dict) -> bool:
    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return False

    if stats["sources"] != expected["sources"]:
        return False
    if stats["arrows"] != expected["arrows"]:
        return False
    if stats["prisms"] != expected["prisms"]:
        return False
    if stats["converted"] < 1 or stats["blocked"] < 1:
        return False
    if stats["painted"] < 5:
        return False
    if stats["overlap"]:
        return False
    if any(count == 0 for count in stats["per_source_counts"]):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shape = (
            random.randint(*GRID_H_RANGE),
            random.randint(*GRID_W_RANGE),
        )
        if random.random() < 0.5:
            built = _template_single(shape)
        else:
            built = _template_double(shape)
        if built is None:
            continue
        grid, expected = built
        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid signal-arrow-prism sample")
