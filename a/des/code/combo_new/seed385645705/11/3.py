from collections import Counter, deque

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")

GEN_RETRIES = 400
PLACEMENT_RETRIES = 500


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
            pts: list[tuple[int, int]] = []
            while q:
                pr, pc = q.popleft()
                pts.append((pr, pc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _middle_positions(start: int, length: int) -> list[int]:
    if length % 2 == 1:
        return [start + length // 2]
    return [start + length // 2 - 1, start + length // 2]


def _marker_position(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if direction == "up":
        return r0 - 1, int(rng.choice(_middle_positions(c0, w)))
    if direction == "down":
        return r0 + h, int(rng.choice(_middle_positions(c0, w)))
    if direction == "left":
        return int(rng.choice(_middle_positions(r0, h))), c0 - 1
    if direction == "right":
        return int(rng.choice(_middle_positions(r0, h))), c0 + w
    raise AssertionError(f"unknown direction: {direction}")


def _placement_ranges(
    grid_h: int,
    grid_w: int,
    obj_h: int,
    obj_w: int,
    direction: str,
    distance: int,
) -> tuple[int, int, int, int] | None:
    r_lo, r_hi = 0, grid_h - obj_h
    c_lo, c_hi = 0, grid_w - obj_w

    if direction == "up":
        r_lo = max(r_lo, 1, distance)
    elif direction == "down":
        r_hi = min(r_hi, grid_h - obj_h - 1, grid_h - obj_h - distance)
    elif direction == "left":
        c_lo = max(c_lo, 1, distance)
    elif direction == "right":
        c_hi = min(c_hi, grid_w - obj_w - 1, grid_w - obj_w - distance)
    else:
        raise AssertionError(f"unknown direction: {direction}")

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    objects: list[dict[str, object]] = []
    used_markers: set[tuple[int, int]] = set()

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, MAROON)):
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            obj_h = r1 - r0 + 1
            obj_w = c1 - c0 + 1
            if obj_h not in (2, 3) or obj_w not in (2, 3):
                raise AssertionError("objects must be small 2x2/2x3/3x2/3x3 rectangles")
            if len(pts) != obj_h * obj_w:
                raise AssertionError("objects must be solid rectangles")
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                raise AssertionError("object bbox must be uniformly colored")

            marker_hits: list[tuple[tuple[int, int], str]] = []
            if r0 - 1 >= 0:
                for cc in range(c0, c1 + 1):
                    if int(g[r0 - 1, cc]) == MAROON:
                        marker_hits.append(((r0 - 1, cc), "up"))
            if r1 + 1 < h:
                for cc in range(c0, c1 + 1):
                    if int(g[r1 + 1, cc]) == MAROON:
                        marker_hits.append(((r1 + 1, cc), "down"))
            if c0 - 1 >= 0:
                for rr in range(r0, r1 + 1):
                    if int(g[rr, c0 - 1]) == MAROON:
                        marker_hits.append(((rr, c0 - 1), "left"))
            if c1 + 1 < w:
                for rr in range(r0, r1 + 1):
                    if int(g[rr, c1 + 1]) == MAROON:
                        marker_hits.append(((rr, c1 + 1), "right"))

            if len(marker_hits) != 1:
                raise AssertionError("each object must have exactly one adjacent maroon marker")

            marker, direction = marker_hits[0]
            used_markers.add(marker)
            objects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "h": obj_h,
                    "w": obj_w,
                    "direction": direction,
                    "marker": marker,
                }
            )

    all_markers = {tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)}
    if used_markers != all_markers:
        raise AssertionError("every maroon pixel must belong to exactly one object")

    return objects


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    objects = _extract_objects(g)
    color_counts = Counter(int(obj["color"]) for obj in objects)

    for obj in objects:
        color = int(obj["color"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        obj_h = int(obj["h"])
        obj_w = int(obj["w"])
        distance = int(color_counts[color] - 1)
        dr, dc = DIR_VECTORS[str(obj["direction"])]
        rr = r0 + distance * dr
        cc = c0 + distance * dc
        out[rr : rr + obj_h, cc : cc + obj_w] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked rectangle in the direction of its maroon arrow by the number of same-colored companions.

    Input structure:
    - The background is black.
    - The foreground contains several small solid colored rectangles, each 2x2 or 3x3.
    - Multiple rectangles may share the same color.
    - Every rectangle has exactly one adjacent maroon marker pixel on one outside side, indicating up, down, left, or right movement.

    Rule:
    - Detect every solid colored rectangle and its unique adjacent maroon marker.
    - For each rectangle, count how many other rectangles in the grid have the same color; call this number K.
    - Translate the entire rectangle by K cells in the marker-indicated direction.
    - Remove every maroon marker from the output; rectangles with K = 0 stay in place.

    Key insight:
    - The direction is local and visible from the maroon marker, but the travel distance is a global count of same-colored partner objects elsewhere in the grid.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sample_group_counts(rng: np.random.Generator) -> list[int]:
    choices = np.array([1, 2, 3, 4], dtype=int)
    probs = np.array([0.34, 0.31, 0.22, 0.13], dtype=float)
    for _ in range(200):
        n_colors = int(rng.integers(2, 5))
        counts = rng.choice(choices, size=n_colors, replace=True, p=probs)
        total = int(np.sum(counts))
        if 4 <= total <= 8 and np.any(counts == 1) and np.any(counts > 1):
            return [int(x) for x in counts]
    raise RuntimeError("failed to sample valid color-group counts")


def _build_specs(rng: np.random.Generator) -> list[dict[str, object]]:
    counts = _sample_group_counts(rng)
    colors = [int(c) for c in rng.choice(np.array(COLOR_POOL, dtype=int), size=len(counts), replace=False)]

    specs: list[dict[str, object]] = []
    for color, count in zip(colors, counts):
        for _ in range(count):
            size = int(rng.choice(np.array([2, 3], dtype=int)))
            direction = str(rng.choice(np.array(DIR_ORDER)))
            specs.append(
                {
                    "color": color,
                    "count": count,
                    "distance": count - 1,
                    "h": size,
                    "w": size,
                    "direction": direction,
                }
            )

    if len(specs) >= 2 and len({int(spec["h"]) for spec in specs}) == 1:
        idx = int(rng.integers(len(specs)))
        specs[idx]["h"] = 5 - int(specs[idx]["h"])
        specs[idx]["w"] = int(specs[idx]["h"])

    if len(specs) >= 2 and len({str(spec["direction"]) for spec in specs}) == 1:
        idx = int(rng.integers(len(specs)))
        current = str(specs[idx]["direction"])
        others = [d for d in DIR_ORDER if d != current]
        specs[idx]["direction"] = str(rng.choice(np.array(others)))

    order = list(range(len(specs)))
    rng.shuffle(order)
    specs = [specs[i] for i in order]
    specs.sort(key=lambda spec: (-int(spec["distance"]), -int(spec["h"])))
    return specs


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _build_specs(rng)
    total_objects = len(specs)
    max_distance = max(int(spec["distance"]) for spec in specs)

    min_h = max(12, 6 + total_objects * 2 + max_distance)
    min_w = max(12, 7 + total_objects * 2 + max_distance)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(rng.integers(min_h, 31))
    grid_w = int(rng.integers(min_w, 31))

    placements: list[dict[str, object]] = []
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()

    for spec in specs:
        obj_h = int(spec["h"])
        obj_w = int(spec["w"])
        direction = str(spec["direction"])
        distance = int(spec["distance"])
        ranges = _placement_ranges(grid_h, grid_w, obj_h, obj_w, direction, distance)
        if ranges is None:
            return None
        r_lo, r_hi, c_lo, c_hi = ranges
        dr, dc = DIR_VECTORS[direction]

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(r_lo, r_hi + 1))
            c0 = int(rng.integers(c_lo, c_hi + 1))
            marker = _marker_position(r0, c0, obj_h, obj_w, direction, rng)

            input_cells = _rect_cells(r0, c0, obj_h, obj_w)
            footprint = set(input_cells)
            footprint.add(marker)
            if footprint & blocked_input:
                continue

            out_r0 = r0 + distance * dr
            out_c0 = c0 + distance * dc
            output_cells = _rect_cells(out_r0, out_c0, obj_h, obj_w)
            if output_cells & blocked_output:
                continue

            placements.append(
                {
                    "color": int(spec["color"]),
                    "r0": r0,
                    "c0": c0,
                    "h": obj_h,
                    "w": obj_w,
                    "direction": direction,
                    "marker": marker,
                    "distance": distance,
                    "out_r0": out_r0,
                    "out_c0": out_c0,
                }
            )
            blocked_input |= _halo(footprint, grid_h, grid_w)
            blocked_output |= _halo(output_cells, grid_h, grid_w)
            placed = True
            break

        if not placed:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for item in placements:
        r0 = int(item["r0"])
        c0 = int(item["c0"])
        obj_h = int(item["h"])
        obj_w = int(item["w"])
        color = int(item["color"])
        grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = color
        mr, mc = item["marker"]
        grid[mr, mc] = MAROON

    try:
        parsed = _extract_objects(grid)
    except AssertionError:
        return None

    if len(parsed) != len(placements):
        return None

    counts = Counter(int(obj["color"]) for obj in parsed)
    if not any(count == 1 for count in counts.values()):
        return None
    if not any(count > 1 for count in counts.values()):
        return None

    out = _solve(grid)
    if np.any(out == MAROON):
        return None

    # Make sure the intended displacement is visible and the color counts are preserved.
    in_colors = np.sort(grid[(grid != BLACK) & (grid != MAROON)])
    out_colors = np.sort(out[out != BLACK])
    if not np.array_equal(in_colors, out_colors):
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input after repeated retries")
