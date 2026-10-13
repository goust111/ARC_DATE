"""ARC task: enclosed holes launch bbox-corner diagonal rays."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _holes(grid: np.ndarray, coords: np.ndarray, bg: int) -> list[np.ndarray]:
    """Find bbox-local background components which do not reach its border."""
    r0, c0, r1, c1 = _bbox(coords)
    local_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes = []
    lh, lw = local_bg.shape
    for comp in _components(local_bg):
        if any(r in (0, lh - 1) or c in (0, lw - 1) for r, c in comp):
            continue
        holes.append(comp + np.array([r0, c0]))
    return holes


def _quadrant(hole: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Classify a whole hole using the component bbox's exact midlines."""
    r0, c0, r1, c1 = box
    row_sides = {"T" if 2 * int(r) < r0 + r1 else "B" if 2 * int(r) > r0 + r1 else "M" for r in hole[:, 0]}
    col_sides = {"L" if 2 * int(c) < c0 + c1 else "R" if 2 * int(c) > c0 + c1 else "M" for c in hole[:, 1]}
    assert len(row_sides) == len(col_sides) == 1, "each hole must lie wholly in one bbox quadrant"
    row_side, col_side = next(iter(row_sides)), next(iter(col_sides))
    assert row_side != "M" and col_side != "M", "holes may not cross or touch a bbox midline"
    return row_side + col_side


def _sprites(grid: np.ndarray, bg: int):
    sprites = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        for coords in comps:
            box = _bbox(coords)
            holes = _holes(grid, coords, bg)
            assert holes, "every colored component must enclose at least one hole"
            lengths = Counter()
            for hole in holes:
                lengths[_quadrant(hole, box)] += len(hole)
            sprites.append((color, coords, box, holes, lengths))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Launch each sprite's enclosed-hole areas as bbox-oriented diagonal rays.

    Input structure:
    - A dominant background contains several mutually separated monochrome sprites of distinct colors.
    - Every sprite encloses one or more 4-connected background holes inside its own bounding box;
      each complete hole lies strictly within one quadrant made by that box's two midlines.

    Rule:
    - For each sprite, assign every hole to the like-named top-left, top-right, bottom-left,
      or bottom-right bbox corner.  Add the areas of holes sharing a quadrant, then paint that
      many sprite-colored cells diagonally outward, beginning one cell beyond the corner and
      clipping the ray at the grid boundary.  Keep the input sprite and its holes unchanged.

    Key insight:
    - A hole is an instruction rather than damage: its bbox-local quadrant selects a launch
      direction, while its area selects the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    out = g.copy()
    for color, _coords, (r0, c0, r1, c1), _holes_found, lengths in _sprites(g, bg):
        corners = {"TL": (r0, c0), "TR": (r0, c1), "BL": (r1, c0), "BR": (r1, c1)}
        for quadrant, length in lengths.items():
            dr, dc = DIRS[quadrant]
            cr, cc = corners[quadrant]
            for step in range(1, int(length) + 1):
                r, c = cr + step * dr, cc + step * dc
                if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                    break
                out[r, c] = color
    return out


# ------------------------------ generator ------------------------------

PALETTE = tuple(range(1, 10))
QUADRANTS = tuple(DIRS)


def _shape_templates(area: int) -> list[set[tuple[int, int]]]:
    if area == 1:
        bases = [{(0, 0)}]
    elif area == 2:
        bases = [{(0, 0), (0, 1)}, {(0, 0), (1, 0)}]
    elif area == 3:
        bases = [
            {(0, 0), (0, 1), (1, 0)},
            {(0, 0), (0, 1), (0, 2)},
            {(0, 0), (1, 0), (2, 0)},
        ]
    else:
        bases = [{(0, 0), (0, 1), (1, 0), (1, 1)}]
    return bases


def _local_quadrant(cell: tuple[int, int], h: int, w: int) -> str | None:
    r, c = cell
    rs = "T" if 2 * r < h - 1 else "B" if 2 * r > h - 1 else "M"
    cs = "L" if 2 * c < w - 1 else "R" if 2 * c > w - 1 else "M"
    return None if "M" in (rs, cs) else rs + cs


def _make_sprite(h: int, w: int, plan: list[tuple[str, int]]):
    mask = np.ones((h, w), dtype=bool)
    occupied: set[tuple[int, int]] = set()
    holes: list[tuple[str, set[tuple[int, int]]]] = []
    for quadrant, area in plan:
        choices = []
        for template in _shape_templates(area):
            th = max(r for r, _ in template) + 1
            tw = max(c for _, c in template) + 1
            for r0 in range(1, h - th):
                for c0 in range(1, w - tw):
                    cells = {(r0 + r, c0 + c) for r, c in template}
                    if any(_local_quadrant(p, h, w) != quadrant for p in cells):
                        continue
                    if any(abs(r - rr) + abs(c - cc) <= 1 for r, c in cells for rr, cc in occupied):
                        continue
                    choices.append(cells)
        if not choices:
            return None
        cells = random.choice(choices)
        occupied |= cells
        holes.append((quadrant, cells))
        for r, c in cells:
            mask[r, c] = False
    if len(_components(mask)) != 1:
        return None
    return mask, holes


def _ray_capacity(box, quadrant: str, shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    h, w = shape
    return {
        "TL": min(r0, c0),
        "TR": min(r0, w - 1 - c1),
        "BL": min(h - 1 - r1, c0),
        "BR": min(h - 1 - r1, w - 1 - c1),
    }[quadrant]


def _box_at_corner(quadrant: str, h: int, w: int, H: int, W: int):
    r0 = 1 if quadrant[0] == "T" else H - h - 1
    c0 = 1 if quadrant[1] == "L" else W - w - 1
    return r0, c0


def _separated(box, boxes) -> bool:
    r0, c0, r1, c1 = box
    for a0, b0, a1, b1 in boxes:
        if not (r1 + 1 < a0 or a1 + 1 < r0 or c1 + 1 < b0 or b1 + 1 < c0):
            return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict], bool] | None:
    H, W = random.randint(25, 30), random.randint(25, 30)
    # Three-object layouts need more rejection attempts, so oversample them here;
    # accepted outputs then contain both object-count branches at useful rates.
    count = random.choices((2, 3), weights=(1, 4))[0]
    colors = random.sample(PALETTE, count)
    objects = []

    aggregate_q = random.choice(QUADRANTS)
    other_q = random.choice([q for q in QUADRANTS if q != aggregate_q])
    plans = [[(aggregate_q, 1), (aggregate_q, 2), (other_q, random.randint(2, 4))]]

    remaining = [q for q in QUADRANTS if q not in {aggregate_q, other_q}]
    q1 = random.choice(remaining)
    q2 = random.choice([q for q in QUADRANTS if q != q1])
    plans.append([(q1, 2), (q2, random.randint(3, 4))])
    if count == 3:
        qs = random.sample(QUADRANTS, random.choice((1, 2)))
        plans.append([(q, random.randint(1, 4)) for q in qs])

    for color, plan in zip(colors, plans):
        made = None
        for _ in range(60):
            h, w = random.randint(9, 11), random.randint(9, 11)
            made = _make_sprite(h, w, plan)
            if made is not None:
                break
        if made is None:
            return None
        mask, holes = made
        lengths = Counter(q for q, cells in holes for _unused in range(len(cells)))
        objects.append({"color": color, "mask": mask, "holes": holes, "lengths": lengths})

    want_clip = random.random() < 0.4
    clip_q = random.choice([q for q, length in objects[1]["lengths"].items() if length >= 2]) if want_clip else None

    # Place the intentionally clipped object first, then rejection-sample the others.
    order = [1, 0] + list(range(2, count))
    boxes = []
    for index in order:
        obj = objects[index]
        h, w = obj["mask"].shape
        placed = None
        for attempt in range(500):
            if index == 1 and want_clip:
                r0, c0 = _box_at_corner(clip_q, h, w, H, W)
            else:
                r0 = random.randint(1, H - h - 1)
                c0 = random.randint(1, W - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not _separated(box, boxes):
                continue
            if index == 0 and _ray_capacity(box, aggregate_q, (H, W)) < objects[0]["lengths"][aggregate_q]:
                continue
            placed = box
            break
        if placed is None:
            return None
        obj["box"] = placed
        boxes.append(placed)

    grid = np.zeros((H, W), dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["box"]
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        view[obj["mask"]] = obj["color"]

    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != 0:
        return None

    expected = grid.copy()
    used_rays = set()
    clipped = False
    for obj in objects:
        r0, c0, r1, c1 = obj["box"]
        corners = {"TL": (r0, c0), "TR": (r0, c1), "BL": (r1, c0), "BR": (r1, c1)}
        for q, length in obj["lengths"].items():
            cap = _ray_capacity(obj["box"], q, (H, W))
            clipped |= cap < length
            visible = min(cap, length)
            if visible < 1:
                return None
            dr, dc = DIRS[q]
            cr, cc = corners[q]
            cells = [(cr + t * dr, cc + t * dc) for t in range(1, visible + 1)]
            if any(grid[r, c] != 0 or (r, c) in used_rays for r, c in cells):
                return None
            # A ray may touch its own launch corner diagonally, but should not visually
            # merge with another sprite or with another ray.
            for r, c in cells:
                for rr in range(max(0, r - 1), min(H, r + 2)):
                    for cc in range(max(0, c - 1), min(W, c + 2)):
                        if grid[rr, cc] not in (0, obj["color"]):
                            return None
                        if (rr, cc) in used_rays:
                            return None
            for r, c in cells:
                expected[r, c] = obj["color"]
                used_rays.add((r, c))

    if want_clip != clipped:
        return None
    return grid, expected, objects, want_clip


def _validate_case(grid: np.ndarray, expected: np.ndarray, objects: list[dict], want_clip: bool) -> bool:
    """Validate intended semantics and several tempting but incorrect shortcuts."""
    if grid.shape != expected.shape or max(grid.shape) > 30 or np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    found = _sprites(grid, 0)
    if len(found) != len(objects) or len({x[0] for x in found}) != len(objects):
        return False
    by_color = {x[0]: x for x in found}
    all_q = set()
    local_vs_global = False
    wrong_count = grid.copy()
    for obj in objects:
        color = obj["color"]
        if color not in by_color:
            return False
        _col, coords, box, holes, lengths = by_color[color]
        if box != obj["box"] or lengths != obj["lengths"] or len(coords) != int(obj["mask"].sum()):
            return False
        intended_holes = {frozenset((obj["box"][0] + r, obj["box"][1] + c) for r, c in cells) for _, cells in obj["holes"]}
        if {frozenset(map(tuple, h)) for h in holes} != intended_holes:
            return False
        all_q.update(lengths)
        for hole in holes:
            q = _quadrant(hole, box)
            gr = "T" if 2 * float(hole[:, 0].mean()) < grid.shape[0] - 1 else "B"
            gc = "L" if 2 * float(hole[:, 1].mean()) < grid.shape[1] - 1 else "R"
            local_vs_global |= q != gr + gc

        # Deliberately wrong oracle: ray length is hole count, not summed hole area.
        counts = Counter(_quadrant(hole, box) for hole in holes)
        r0, c0, r1, c1 = box
        corners = {"TL": (r0, c0), "TR": (r0, c1), "BL": (r1, c0), "BR": (r1, c1)}
        for q, length in counts.items():
            dr, dc = DIRS[q]
            cr, cc = corners[q]
            for t in range(1, length + 1):
                r, c = cr + t * dr, cc + t * dc
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    break
                wrong_count[r, c] = color

    if len(all_q) < 3 or not local_vs_global or np.array_equal(wrong_count, expected):
        return False
    # The aggregate ray is deliberately fully visible, so summing is observable.
    first = objects[0]
    repeated = [q for q in first["lengths"] if sum(1 for qq, _ in first["holes"] if qq == q) > 1]
    if len(repeated) != 1 or _ray_capacity(first["box"], repeated[0], grid.shape) < first["lengths"][repeated[0]]:
        return False
    actual_clip = any(_ray_capacity(obj["box"], q, grid.shape) < n for obj in objects for q, n in obj["lengths"].items())
    return actual_clip == want_clip


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance with two or three sprites."""
    for _ in range(600):
        case = _build_case()
        if case is not None and _validate_case(*case):
            return case[0]
    raise RuntimeError("failed to generate a valid hole-directed ray instance")
