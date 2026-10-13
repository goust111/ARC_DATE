import random

import numpy as np


DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

DIRECTION_VECTORS = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}

GEN_BG = 0
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_CORE_SIZE_RANGE = (4, 6)
GEN_FRAGMENT_COUNT_RANGE = (3, 5)
GEN_TRAIL_COUNT_RANGE = (1, 3)
GEN_MAX_ATTEMPTS = 300


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != bg)
    records: list[dict] = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        records.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "area": len(coords),
                "colors": colors,
                "center": (float(np.mean(rs)), float(np.mean(cs))),
            }
        )
    return records


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _touch_count(coords: list[tuple[int, int]], other: set[tuple[int, int]]) -> int:
    count = 0
    for cell in coords:
        if any(nei in other for nei in _neighbors8(cell)):
            count += 1
    return count


def _min_chebyshev(coords: list[tuple[int, int]], other: set[tuple[int, int]]) -> int:
    best: int | None = None
    for r, c in coords:
        for rr, cc in other:
            dist = max(abs(r - rr), abs(c - cc))
            if best is None or dist < best:
                best = dist
                if best == 0:
                    return 0
    assert best is not None
    return best


def _translate(coords: list[tuple[int, int]], dr: int, dc: int, dist: int) -> list[tuple[int, int]]:
    return [(r - dr * dist, c - dc * dist) for r, c in coords]


def _infer_direction(
    core_center: tuple[float, float],
    fragment_coords: list[tuple[int, int]],
    trail_points: list[tuple[int, int]],
) -> tuple[int, int]:
    frag_r = float(np.mean([r for r, _ in fragment_coords]))
    frag_c = float(np.mean([c for _, c in fragment_coords]))
    base = (_sign(frag_r - core_center[0]), _sign(frag_c - core_center[1]))

    if trail_points:
        rows = [r for r, _ in trail_points]
        cols = [c for _, c in trail_points]
        if len(set(rows)) == 1:
            return (0, base[1] if base[1] != 0 else 1)
        if len(set(cols)) == 1:
            return (base[0] if base[0] != 0 else 1, 0)
        if len({r - c for r, c in trail_points}) == 1:
            s = base[0] if base[0] != 0 else base[1]
            return (s if s != 0 else 1, s if s != 0 else 1)
        if len({r + c for r, c in trail_points}) == 1:
            dr = base[0] if base[0] != 0 else (-base[1] if base[1] != 0 else 1)
            return (dr, -dr)

    if base != (0, 0):
        return base
    return (1, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble exploded monochrome fragments back onto a compact multicolor core.

    Input structure:
    - Background is black.
    - One central compact multicolor component is the fixed core wreckage.
    - Each fragment color also appears as a same-color sparse track of singleton
      pixels along one of the eight compass directions.

    Rule:
    - For each fragment, infer its outward direction from its track and move it
      back toward the core by a multiple of 3.
    - Clear the track pixels.
    - Keep the core in place.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    records = _component_records(g, bg)
    assert records, "expected non-background objects"

    grid_center = ((h - 1) / 2.0, (w - 1) / 2.0)
    core_candidates = [rec for rec in records if len(rec["colors"]) >= 2]
    assert core_candidates, "expected one multicolor core component"
    core = max(
        core_candidates,
        key=lambda rec: (
            len(rec["colors"]),
            rec["area"],
            -(
                abs(rec["center"][0] - grid_center[0])
                + abs(rec["center"][1] - grid_center[1])
            ),
        ),
    )

    core_set = core["coord_set"]
    out = np.full_like(g, bg)
    for r, c in core["coords"]:
        out[r, c] = int(g[r, c])

    occupied = set(core_set)
    all_colors = [int(color) for color in np.unique(g) if int(color) != bg]

    for color in all_colors:
        color_mask = (g == color)
        if not np.any(color_mask):
            continue
        for r, c in core_set:
            color_mask[r, c] = False
        if not np.any(color_mask):
            continue

        color_comps = _components(color_mask)
        fragments = [comp for comp in color_comps if len(comp) >= 2]
        trails = [comp[0] for comp in color_comps if len(comp) == 1]
        assert len(fragments) <= 1, "expected at most one fragment per color"
        if not fragments:
            continue
        assert trails, "each fragment must have a visible track"

        fragment = fragments[0]
        direction = _infer_direction(core["center"], fragment, trails)
        trail_count = len(trails)

        candidate_steps = {
            step
            for step in (
                trail_count - 1,
                trail_count,
                trail_count + 1,
                trail_count + 2,
                trail_count + 3,
            )
            if step >= 1
        }
        candidate_steps.update(range(1, trail_count + 5))

        best_coords: list[tuple[int, int]] | None = None
        best_score: tuple[int, int, int, int] | None = None

        for steps in sorted(candidate_steps):
            dist = 3 * steps
            shifted = _translate(fragment, direction[0], direction[1], dist)

            if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
                continue
            if any(cell in occupied for cell in shifted):
                continue

            touch_core = _touch_count(shifted, core_set)
            touch_any = _touch_count(shifted, occupied)
            min_gap = _min_chebyshev(shifted, core_set)
            score = (
                1 if touch_core > 0 else 0,
                touch_core,
                1 if touch_any > 0 else 0,
                -min_gap,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_coords = shifted

        assert best_coords is not None and best_score is not None
        for r, c in best_coords:
            if out[r, c] == bg:
                out[r, c] = color
        occupied.update(best_coords)

    return out


def _random_core(core_h: int, core_w: int, colors: list[int]) -> np.ndarray:
    core = np.array(random.choices(colors, k=core_h * core_w), dtype=int).reshape(core_h, core_w)
    all_positions = [(r, c) for r in range(core_h) for c in range(core_w)]
    random.shuffle(all_positions)
    for color, (r, c) in zip(colors, all_positions):
        core[r, c] = color
    return core


def _grow_connected(candidates: list[tuple[int, int]], area: int) -> list[tuple[int, int]] | None:
    candidate_set = set(candidates)
    if (0, 0) not in candidate_set:
        return None
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS8:
                nxt = (r + dr, c + dc)
                if nxt in candidate_set and nxt not in cells:
                    frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return list(cells)


def _fragment_offsets(name: str) -> list[tuple[int, int]] | None:
    if name in ("E", "W"):
        primary = random.randint(2, 4)
        side = random.randint(0, 1)
        rows = range(-side, side + 1)
        cols = range(primary) if name == "E" else range(-(primary - 1), 1)
        area = random.randint(2, min(5, len(rows) * len(cols)))
    elif name in ("N", "S"):
        primary = random.randint(2, 4)
        side = random.randint(0, 1)
        rows = range(primary) if name == "S" else range(-(primary - 1), 1)
        cols = range(-side, side + 1)
        area = random.randint(2, min(5, len(rows) * len(cols)))
    else:
        span_r = random.randint(2, 3)
        span_c = random.randint(2, 3)
        rows = range(span_r) if "S" in name else range(-(span_r - 1), 1)
        cols = range(span_c) if "E" in name else range(-(span_c - 1), 1)
        area = random.randint(2, min(4, len(rows) * len(cols)))

    candidates = [(r, c) for r in rows for c in cols]
    return _grow_connected(candidates, area)


def _choose_anchor(
    name: str,
    offsets: list[tuple[int, int]],
    selected: set[str],
    core_h: int,
    core_w: int,
) -> tuple[int, int] | None:
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)

    if name == "E":
        lo = -min_r + (1 if "NE" in selected else 0)
        hi = core_h - 1 - max_r - (1 if "SE" in selected else 0)
        if lo > hi:
            return None
        return (random.randint(lo, hi), core_w)
    if name == "W":
        lo = -min_r + (1 if "NW" in selected else 0)
        hi = core_h - 1 - max_r - (1 if "SW" in selected else 0)
        if lo > hi:
            return None
        return (random.randint(lo, hi), -1)
    if name == "N":
        lo = -min_c + (1 if "NW" in selected else 0)
        hi = core_w - 1 - max_c - (1 if "NE" in selected else 0)
        if lo > hi:
            return None
        return (-1, random.randint(lo, hi))
    if name == "S":
        lo = -min_c + (1 if "SW" in selected else 0)
        hi = core_w - 1 - max_c - (1 if "SE" in selected else 0)
        if lo > hi:
            return None
        return (core_h, random.randint(lo, hi))
    if name == "NE":
        return (-1, core_w)
    if name == "NW":
        return (-1, -1)
    if name == "SE":
        return (core_h, core_w)
    if name == "SW":
        return (core_h, -1)
    return None


def _shift_cells(
    cells: list[tuple[int, int]],
    delta_r: int,
    delta_c: int,
) -> list[tuple[int, int]]:
    return [(r + delta_r, c + delta_c) for r, c in cells]


def _min_pairwise_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(max(abs(r - rr), abs(c - cc)) for r, c in a for rr, cc in b)


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    for r, c in cells:
        if grid[r, c] != GEN_BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    fragment_count = random.randint(*GEN_FRAGMENT_COUNT_RANGE)
    directions = random.sample(list(DIRECTION_VECTORS.keys()), fragment_count)
    selected = set(directions)

    core_h = random.randint(*GEN_CORE_SIZE_RANGE)
    core_w = random.randint(*GEN_CORE_SIZE_RANGE)
    core_color_count = 2 if fragment_count >= 5 else random.randint(2, 3)

    palette = list(range(1, 10))
    random.shuffle(palette)
    core_colors = palette[:core_color_count]
    fragment_colors = palette[core_color_count : core_color_count + fragment_count]

    core = _random_core(core_h, core_w, core_colors)
    core_cells = [(r, c) for r in range(core_h) for c in range(core_w)]

    solved_fragments: list[dict] = []
    used_output_sets: list[set[tuple[int, int]]] = [set(core_cells)]

    for name, color in zip(directions, fragment_colors):
        offsets = _fragment_offsets(name)
        if offsets is None:
            return None
        anchor = _choose_anchor(name, offsets, selected, core_h, core_w)
        if anchor is None:
            return None

        solved_cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets]
        solved_set = set(solved_cells)
        if solved_set & used_output_sets[0]:
            return None

        if _min_pairwise_distance(solved_set, used_output_sets[0]) != 1:
            return None
        if any(_min_pairwise_distance(solved_set, other) <= 1 for other in used_output_sets[1:]):
            return None

        trail_count = random.randint(*GEN_TRAIL_COUNT_RANGE)
        dr, dc = DIRECTION_VECTORS[name]
        shift = 3 * (trail_count + 1)
        exploded_cells = _shift_cells(solved_cells, dr * shift, dc * shift)
        anchor_point = anchor
        trail_cells = [
            (anchor_point[0] + dr * (3 * step), anchor_point[1] + dc * (3 * step))
            for step in range(1, trail_count + 1)
        ]

        solved_fragments.append(
            {
                "color": color,
                "solved": solved_cells,
                "exploded": exploded_cells,
                "trail": trail_cells,
            }
        )
        used_output_sets.append(solved_set)

    input_objects = [set(core_cells)]
    for item in solved_fragments:
        exploded_set = set(item["exploded"])
        if any(_min_pairwise_distance(exploded_set, other) <= 1 for other in input_objects):
            return None
        input_objects.append(exploded_set)
        for trail_cell in item["trail"]:
            singleton = {trail_cell}
            if any(_min_pairwise_distance(singleton, other) <= 1 for other in input_objects):
                return None
            input_objects.append(singleton)

    all_input_cells = set(core_cells)
    for item in solved_fragments:
        all_input_cells.update(item["exploded"])
        all_input_cells.update(item["trail"])

    min_r = min(r for r, _ in all_input_cells)
    max_r = max(r for r, _ in all_input_cells)
    min_c = min(c for _, c in all_input_cells)
    max_c = max(c for _, c in all_input_cells)
    obj_h = max_r - min_r + 1
    obj_w = max_c - min_c + 1

    h_lo = max(GEN_GRID_MIN, obj_h + 2)
    h_hi = min(GEN_GRID_MAX, obj_h + 6)
    w_lo = max(GEN_GRID_MIN, obj_w + 2)
    w_hi = min(GEN_GRID_MAX, obj_w + 6)
    if h_lo > h_hi or w_lo > w_hi:
        return None

    grid_h = random.randint(h_lo, h_hi)
    grid_w = random.randint(w_lo, w_hi)
    top = random.randint(1, grid_h - obj_h - 1)
    left = random.randint(1, grid_w - obj_w - 1)
    delta_r = top - min_r
    delta_c = left - min_c

    inp = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    out = np.full((grid_h, grid_w), GEN_BG, dtype=int)

    shifted_core = _shift_cells(core_cells, delta_r, delta_c)
    for (rr, cc), value in zip(shifted_core, core.ravel().tolist()):
        inp[rr, cc] = int(value)
        out[rr, cc] = int(value)

    for item in solved_fragments:
        solved = _shift_cells(item["solved"], delta_r, delta_c)
        exploded = _shift_cells(item["exploded"], delta_r, delta_c)
        trail = _shift_cells(item["trail"], delta_r, delta_c)
        if not _paint(inp, exploded, int(item["color"])):
            return None
        if not _paint(out, solved, int(item["color"])):
            return None
        if not _paint(inp, trail, int(item["color"])):
            return None

    if np.count_nonzero(inp != GEN_BG) >= (inp.size // 2):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC sample")
