# Combo variant of 80a900e0 (index 13)
# combo_spec: D9/1+D17/1
# Name: 密集交叉标记
#
# Design analysis for the fused variant:
# a) Concrete behavior:
#    - Original: 1-3 diamonds, each outer segment emits two rays, and the first ray to reach a
#      background cell keeps it because later rays stop on painted pixels.
#    - Fused: 4-6 diamonds are packed into one grid, so many endpoint rays meet each other.
#      Rays are still blocked by the original static objects (diamonds / short segments), but
#      ray-vs-ray encounters are merged after tracing all paths. If one cell is claimed by
#      exactly one ray color it keeps that color; if claimed by different ray colors it becomes 9.
#      Example: a `sum`-family segment (color 7) and a `diff`-family segment (color 8) cross on
#      a checkerboard cell. Original logic would keep whichever ray arrived first; fused logic
#      paints that crossing cell as 9 and lets both rays continue.
# b) Complexity calibration:
#    - The generator raises the diamond count to 4-6 and biases most diamonds to 3-4 occupied
#      edges, so the visual difference from the seed task is obvious.
#    - Segment colors are restricted to two repeated slope-linked colors (7 for `sum`, 8 for
#      `diff`) so the collision rule stays inferable instead of becoming arbitrary recoloring.
# c) Hidden assumptions reviewed:
#    - Seed solver assumed `out` could be painted in-place and rays should stop on earlier rays.
#      That is invalid now, so `main()` first traces against the input's static non-bg objects and
#      only then merges all ray claims.
#    - Seed solver also relied on outer segments being isolated 8-components adjacent only to
#      their own diamond. The generator now enforces a one-cell halo around every diamond+segment
#      scaffold so that assumption remains true even at higher density.
# d) Placement legality:
#    - The generator searches only placements whose diamonds and short outer segments stay in
#      bounds, have at least 2 visible pixels per segment, and whose static halos do not overlap.
#    - Rays may cross each other, but they are still traced to stop before any original static
#      object, matching the solver's assumption.
# e) Visual observability:
#    - The collision color 9 is reserved for output-only marks, so crossings are visually salient.
#      Inputs show many diamonds plus two repeated segment colors, and outputs add many 9 marks
#      exactly where opposite-slope ray families intersect.
# f) Visual cleanliness:
#    - Only one overlap layer is allowed on dynamic cells: normal ray color or collision color 9.
#      Static objects never overlap, and halo spacing prevents adjacent diamonds from blending.
# g) Inferability of the change:
#    - The important variable is structural, not numeric: same short segment shape => same ray
#      behavior, and opposite ray colors meeting => 9. The generator requires a minimum number of
#      collision markers so the fused behavior cannot collapse back to the seed task.

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _trace_bg_ray(
    g: np.ndarray,
    bg_colors: set[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = g.shape
    dr, dc = direction
    rr, cc = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        if int(g[rr, cc]) not in bg_colors:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow dense diagonal ray patterns from many diamond-border segments and mark ray crossings.

    Input structure:
    - A checkerboard background made from the two most frequent colors.
    - Multiple hollow 8-pixel diamonds in distinct non-background colors.
    - Several short diagonal segments beside the diamond edges; each segment is a separate
      monochrome outer border and acts as the source scaffold for two rays.

    Rule:
    - For each outer segment, infer which diagonal direction points outward by checking which
      candidate normal points back into the adjacent diamond.
    - Emit one ray from each endpoint of the segment, moving diagonally across background cells
      until the grid edge or an original non-background object is reached.
    - Merge all rays simultaneously: a cell claimed by one visible ray color keeps that color,
      while a cell claimed by different ray colors becomes the special crossing color 9.

    Key insight:
    - In this dense variant, rays must be traced against the input's static objects first and
      only merged afterward. Crossings are not blockers anymore; they are explicit output marks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()
    ray_claims: dict[tuple[int, int], set[int]] = defaultdict(set)

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj: set[tuple[int, int]] = set()
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.add((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            colors = {int(g[r, c]) for r, c in oc}
            if len(colors) != 1:
                continue
            ocol = next(iter(colors))
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info

            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue

            outward = (-best_n[0], -best_n[1])
            for start in (endpoints[0], endpoints[-1]):
                for cell in _trace_bg_ray(g, bg_colors, start, outward):
                    ray_claims[cell].add(ocol)

    for (r, c), colors in ray_claims.items():
        out[r, c] = 9 if len(colors) >= 2 else next(iter(colors))

    return out


# --------------- generator ---------------

GRID_H_RANGE = (23, 24)
GRID_W_RANGE = (26, 30)
DIAMOND_COUNT_RANGE = (4, 6)
COLOR_RANGE = tuple(range(2, 9))
COLLISION_COLOR = 9
MAX_GEN_RETRIES = 220
SEARCH_ROUNDS = 28
CANDIDATE_LIMIT = 180

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_SLOPE_TO_SEGMENT_COLOR = {"sum": 7, "diff": 8}
_SINGLE_FAMILY_SPECS = (
    ("TL", "BR"),
    ("TR", "BL"),
)
_MIXED_SPECS = (
    ("TL", "BR", "TR"),
    ("TL", "BR", "BL"),
    ("TR", "BL", "TL"),
    ("TR", "BL", "BR"),
    ("TL", "TR", "BL", "BR"),
)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _quadrant_of(r0: int, c0: int, h: int, w: int) -> str:
    cr = r0 + 2
    cc = c0 + 2
    return ("N" if 2 * cr < h else "S") + ("W" if 2 * cc < w else "E")


def _pick_specs(diamond_count: int) -> list[tuple[str, ...]]:
    min_single = max(1, diamond_count - 4)
    max_single = min(2, diamond_count - 1)
    single_count = random.randint(min_single, max_single)
    specs = [random.choice(_SINGLE_FAMILY_SPECS) for _ in range(single_count)]
    specs.extend(random.choice(_MIXED_SPECS) for _ in range(diamond_count - single_count))
    random.shuffle(specs)
    return specs


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    edges: tuple[str, ...],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    static = set(diamond)
    edge_parts = []
    forbidden_colors: set[int] = set()

    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < 2:
            return None
        segment = _sort_segment(segment, info["slope"])
        if any(cell in static for cell in segment):
            return None
        static.update(segment)
        color = _SLOPE_TO_SEGMENT_COLOR[info["slope"]]
        forbidden_colors.add(color)
        edge_parts.append(
            {
                "edge": edge,
                "segment": segment,
                "outward": info["outward"],
                "slope": info["slope"],
                "color": color,
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "static": frozenset(static),
        "halo": frozenset(_halo(static, (h, w), pad=1)),
        "quad": _quadrant_of(r0, c0, h, w),
        "forbidden_colors": frozenset(forbidden_colors),
        "anchor": (r0, c0),
    }


def _trace_ray_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    static_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in static_cells:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _collect_ray_claims(
    shape: tuple[int, int],
    placements: list[dict[str, object]],
) -> tuple[dict[tuple[int, int], set[int]], set[tuple[int, int]]]:
    static_cells = set().union(*(set(pl["static"]) for pl in placements))
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    ray_cells: set[tuple[int, int]] = set()

    for placement in placements:
        for edge_part in placement["edges"]:
            segment = edge_part["segment"]
            color = edge_part["color"]
            direction = edge_part["outward"]
            for start in (segment[0], segment[-1]):
                for cell in _trace_ray_cells(shape, start, direction, static_cells):
                    claims[cell].add(color)
                    ray_cells.add(cell)
    return claims, ray_cells


def _assign_diamond_colors(placements: list[dict[str, object]]) -> list[int] | None:
    options = []
    for placement in placements:
        forbidden = set(placement["forbidden_colors"])
        allowed = [c for c in COLOR_RANGE if c not in forbidden]
        options.append(allowed)

    order = sorted(range(len(placements)), key=lambda i: (len(options[i]), random.random()))
    assigned = [-1] * len(placements)
    used: set[int] = set()

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        pi = order[i]
        candidates = [c for c in options[pi] if c not in used]
        random.shuffle(candidates)
        for color in candidates:
            assigned[pi] = color
            used.add(color)
            if backtrack(i + 1):
                return True
            used.remove(color)
            assigned[pi] = -1
        return False

    if not backtrack(0):
        return None
    return assigned


def _layout_score(
    shape: tuple[int, int],
    placements: list[dict[str, object]],
) -> tuple[int, int, int]:
    claims, ray_cells = _collect_ray_claims(shape, placements)
    collision_count = sum(1 for colors in claims.values() if len(colors) >= 2)
    return collision_count, len(ray_cells), len(claims)


def _quadrant_cover(placements: list[dict[str, object]]) -> int:
    return len({placement["quad"] for placement in placements})


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    positions = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]
    random.shuffle(positions)

    cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for spec in set(specs):
        candidates = []
        for r0, c0 in positions:
            placement = _candidate_placement(h, w, r0, c0, spec)
            if placement is not None:
                candidates.append(placement)
        random.shuffle(candidates)
        cache[spec] = candidates[:CANDIDATE_LIMIT]
        if not cache[spec]:
            return None

    best: tuple[int, int, int] | None = None
    best_layout: list[dict[str, object]] | None = None

    for _ in range(SEARCH_ROUNDS):
        order = sorted(range(len(specs)), key=lambda i: (len(specs[i]), random.random()), reverse=True)
        occupied_halo: set[tuple[int, int]] = set()
        quad_counts = Counter()
        placed: list[dict[str, object] | None] = [None] * len(specs)
        ok = True

        for spec_i in order:
            candidates = []
            for placement in cache[specs[spec_i]]:
                if not occupied_halo.isdisjoint(placement["halo"]):
                    continue
                quad_penalty = quad_counts[placement["quad"]]
                center_r = placement["anchor"][0] + 2
                center_c = placement["anchor"][1] + 2
                center_dist = abs(2 * center_r - h) + abs(2 * center_c - w)
                candidates.append((quad_penalty, center_dist, random.random(), placement))
            if not candidates:
                ok = False
                break
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            shortlist = [item[3] for item in candidates[: min(12, len(candidates))]]
            chosen = random.choice(shortlist)
            placed[spec_i] = chosen
            occupied_halo.update(chosen["halo"])
            quad_counts[chosen["quad"]] += 1

        if not ok:
            continue

        layout = [placement for placement in placed if placement is not None]
        if _quadrant_cover(layout) < 4:
            continue
        if _assign_diamond_colors(layout) is None:
            continue

        score = _layout_score((h, w), layout)
        if best is None or score > best:
            best = score
            best_layout = layout

    return best_layout


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in range(2, 10)), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _build_output(
    h: int,
    w: int,
    placements: list[dict[str, object]],
    diamond_colors: list[int],
) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    out = _checkerboard(h, w)
    for placement, dcolor in zip(placements, diamond_colors):
        _paint_cells(out, placement["diamond"], dcolor)
        for edge_part in placement["edges"]:
            _paint_cells(out, edge_part["segment"], edge_part["color"])

    claims, ray_cells = _collect_ray_claims((h, w), placements)
    collision_count = 0
    for (r, c), colors in claims.items():
        if len(colors) >= 2:
            out[r, c] = COLLISION_COLOR
            collision_count += 1
        else:
            out[r, c] = next(iter(colors))
    return out, ray_cells, collision_count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _pick_specs(diamond_count)
    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = _assign_diamond_colors(placements)
    if diamond_colors is None:
        return None

    out, ray_cells, collision_count = _build_output(h, w, placements, diamond_colors)
    min_collisions = max(6, diamond_count + 2)
    if collision_count < min_collisions:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_cells:
        rs, cs = zip(*ray_cells)
        rs = np.asarray(rs)
        cs = np.asarray(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not _bg_counts_safe(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
