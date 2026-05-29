# Combo variant of 13e47133 (index 18)
# combo_spec: D7/1+D15/2
# Name: 多色墙壁+墙壁指示点

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the combo variant
# a) Concrete behavior and behavior trace:
#    - Original task: one wall color splits the grid into regions. Inside each region,
#      marker depths define an ordered palette, and the region is filled by stepping
#      forward through that palette with increasing Chebyshev depth.
#    - D15 part: put one colored dot on a wall cell next to a region. That dot repeats
#      one marker color from the region and says "start the boundary band with this
#      palette entry" instead of the original first entry.
#    - D7 part for this combo: the true separator walls now use two base colors. The
#      wall color underneath that dot decides whether the palette is traversed forward
#      or backward as depth increases.
#    - Small example: suppose a region has base 0 and markers [(d0, red), (d1, green)],
#      so its source palette is [red, green]. A green wall dot rotates the start to
#      green. If that dotted wall segment uses the majority wall color, the fill order
#      is [green, red, green, ...]. If the same dot sat on the minority wall color,
#      the boundary is still green but the inward order is [green, red, green, ...] for
#      length 2, so the direction is not visible. Therefore the generator forces at
#      least one reverse-controlled region whose palette length is >= 3, e.g.
#      [base, red, green] with green as the dot, yielding forward
#      [green, base, red, ...] versus reverse [green, red, base, ...].
#    - Net difference: the wall is no longer a passive separator. Its colored dot picks
#      the first band, and the wall segment's own base color picks the cyclic direction.
#
# b) Complexity calibration:
#    - The generator uses exactly two wall base colors and makes one clearly more common
#      than the other. The more common wall color means "forward"; the less common wall
#      color means "reverse". This keeps the rule binary and visually learnable.
#    - Every region gets exactly one wall dot and at least one internal marker, so both
#      symbol-reading and wall-color-reading are always present.
#    - At least one region must show a visible clue rotation, and at least one region on
#      the minority wall color must have palette length >= 3 so the reverse rule changes
#      the output compared with a clue-only variant.
#
# c) Hidden-assumption audit against the original main():
#    - "wall is one global color" breaks: walls now use two base colors plus overwritten
#      clue dots, so `free = g != wall` is invalid.
#    - "second-most-common non-bg color is the wall" breaks for the same reason.
#    - "all non-base colored cells inside components are region markers" breaks because
#      clue dots live on the wall. The solver must reconstruct the wall structurally.
#    - "sorted marker depths fully determine the palette traversal" breaks because the
#      clue dot changes the start and the wall color changes the traversal direction.
#    - New local assumption: a clue dot is stamped only on a straight interior wall cell
#      whose same-axis neighbors keep the underlying wall color visible. The generator
#      enforces this so the solver can recover the hidden wall color under the clue.
#
# d) Placement legality:
#    - Regions are still axis-aligned rectangles produced by recursive 1-cell splits, so
#      internal markers and wall clues can be placed without overlap or ambiguity.
#    - Internal markers stay on one corner diagonal within their region and never leave
#      bounds. Wall dots overwrite only true wall cells and are kept non-adjacent.
#    - Candidate clue cells adjacent to a region are filtered to avoid junctions,
#      endpoints, and cells next to internal markers.
#
# e) Visual observability:
#    - Human solvers can match a wall dot's color to a same-colored marker inside the
#      adjacent region, which explains the boundary band's first color.
#    - Across samples, regions whose dot sits on the dominant wall color progress inward
#      in one cyclic order, while regions whose dot sits on the rarer wall color use the
#      opposite order. The difference is visible because some reverse regions have 3-step
#      or 4-step palettes.
#
# f) Visual cleanliness:
#    - The scene uses four clear layers only: base region cells, sparse internal markers,
#      thin two-colored walls, and one-pixel wall dots. No object overlaps another kind
#      of object except that a dot intentionally overwrites one wall cell.
#    - Clue dots are placed on straight wall interiors, so they read as localized symbols
#      rather than noisy wall damage at junctions.
#
# g) Change inferability:
#    - No rule depends on raw color ids. The two wall roles are inferred from visible
#      wall-color frequency: more-common wall color => forward; less-common => reverse.
#    - Clue colors never introduce a new abstract code. Each dot repeats a real marker
#      color from the neighboring region, so the palette-start link is grounded locally.
#
# Assumption audit for the new main():
#    - Every true wall or wall-dot cell belongs to a horizontal/vertical non-base chain.
#      Generator guarantee: walls are 1-cell axis-aligned split lines; clue dots only
#      overwrite wall cells.
#    - Every internal marker is not itself part of such a chain.
#      Generator guarantee: markers are isolated diagonal points inside rectangles.
#    - Each region has exactly one uniquely matchable wall dot.
#      Generator guarantee: every region receives one dot; candidate cells are filtered
#      so the dot color matches exactly one adjacent region's marker set.
#    - The wall-dot cell reveals its hidden wall color via same-axis wall neighbors.
#      Generator guarantee: dots are placed only on straight interior wall cells whose
#      aligned neighbors share the same base wall color.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance to the component boundary; -1 outside the component."""
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _wall_mask_from_structure(g: np.ndarray, base: int) -> np.ndarray:
    """
    Recover the wall mask after clue dots recolor some wall cells.

    A true wall cell lies on a 4-neighbor non-base chain, while an internal marker is
    only a local anomaly inside one region.
    """
    h, w = g.shape
    non_base = g != base
    wall_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not non_base[r, c]:
                continue
            dirs = []
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and non_base[rr, cc]:
                    dirs.append((dr, dc))

            if len(dirs) >= 3:
                wall_mask[r, c] = True
                continue

            if len(dirs) == 2:
                (dr1, dc1), (dr2, dc2) = dirs
                if dr1 == -dr2 and dc1 == -dc2:
                    wall_mask[r, c] = True
                continue

            if len(dirs) == 1:
                dr, dc = dirs[0]
                rr, cc = r + dr, c + dc
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and non_base[nr, nc]:
                    wall_mask[r, c] = True
    return wall_mask


def _wall_base_colors(g: np.ndarray, wall_mask: np.ndarray) -> list[int]:
    vals, counts = np.unique(g[wall_mask], return_counts=True)
    order = np.argsort(-counts)
    return [int(vals[i]) for i in order[:2]]


def _build_region_infos(g: np.ndarray):
    h, w = g.shape
    base = _mode(g)
    wall_mask = _wall_mask_from_structure(g, base)
    wall_colors = _wall_base_colors(g, wall_mask)

    comp_index = np.full((h, w), -1, dtype=int)
    infos = []
    for idx, pts in enumerate(_cc4(~wall_mask)):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            r = int(r)
            c = int(c)
            comp_mask[r, c] = True
            comp_index[r, c] = idx

        vals = g[comp_mask]
        region_base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        marker_colors: set[int] = set()
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == region_base:
                continue
            marker_colors.add(v)
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            palette = []
            ds = sorted(dist_to_color)
            if ds[0] > 0:
                palette.append(int(region_base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(region_base)]

        infos.append(
            {
                "pts": pts,
                "dist": dist,
                "palette": palette,
                "marker_colors": marker_colors,
            }
        )

    return base, wall_mask, wall_colors, comp_index, infos


def _underlying_wall_color(
    g: np.ndarray,
    wall_mask: np.ndarray,
    wall_colors: list[int],
    r: int,
    c: int,
) -> int | None:
    wall_set = set(wall_colors)
    left = c - 1 >= 0 and wall_mask[r, c - 1]
    right = c + 1 < g.shape[1] and wall_mask[r, c + 1]
    up = r - 1 >= 0 and wall_mask[r - 1, c]
    down = r + 1 < g.shape[0] and wall_mask[r + 1, c]

    if left and right:
        colors = [int(g[r, c - 1]), int(g[r, c + 1])]
        if colors[0] in wall_set and colors[0] == colors[1]:
            return colors[0]
    if up and down:
        colors = [int(g[r - 1, c]), int(g[r + 1, c])]
        if colors[0] in wall_set and colors[0] == colors[1]:
            return colors[0]
    return None


def _region_clues(
    g: np.ndarray,
    wall_mask: np.ndarray,
    wall_colors: list[int],
    comp_index: np.ndarray,
    infos: list[dict],
) -> dict[int, tuple[int, int]]:
    clues: dict[int, tuple[int, int]] = {}
    wall_set = set(wall_colors)
    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)
        clue_color = int(g[r, c])
        if clue_color in wall_set:
            continue

        neighbors: list[int] = []
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                idx = int(comp_index[rr, cc])
                if idx >= 0 and idx not in neighbors:
                    neighbors.append(idx)

        matches = [idx for idx in neighbors if clue_color in infos[idx]["marker_colors"]]
        if len(matches) != 1:
            continue

        wall_color = _underlying_wall_color(g, wall_mask, wall_colors, r, c)
        if wall_color is None:
            continue
        idx = matches[0]
        if idx not in clues:
            clues[idx] = (clue_color, wall_color)
    return clues


def _palette_value(palette: list[int], start_color: int | None, reverse: bool, d: int) -> int:
    m = len(palette)
    if m == 0:
        raise ValueError("empty palette")
    if start_color is None or start_color not in palette:
        k = 0
    else:
        k = palette.index(start_color)
    step = -1 if reverse else 1
    return int(palette[(k + step * d) % m])


def _solve(grid: np.ndarray, use_clues: bool, use_direction: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _base, wall_mask, wall_colors, comp_index, infos = _build_region_infos(g)
    clues = _region_clues(g, wall_mask, wall_colors, comp_index, infos) if use_clues else {}
    reverse_color = wall_colors[1] if len(wall_colors) >= 2 else None

    for idx, info in enumerate(infos):
        start_color = None
        reverse = False
        if idx in clues:
            start_color, wall_color = clues[idx]
            reverse = use_direction and reverse_color is not None and wall_color == reverse_color
        palette = info["palette"]
        for r, c in info["pts"]:
            r = int(r)
            c = int(c)
            d = int(info["dist"][r, c])
            out[r, c] = _palette_value(palette, start_color, reverse, d)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill walled regions with cyclic bands, using wall dots for the first color and wall colors for the traversal direction.

    Input structure:
    - Thin axis-aligned separator walls divide the board into rectangular regions, but the walls now use two dominant colors instead of one.
    - Each region still contains a few isolated internal marker pixels whose Chebyshev depths define the region's palette entries.
    - One wall cell next to each region is recolored as a clue dot; that dot repeats one marker color from the adjacent region.

    Rule:
    - Recover the true wall mask structurally, then for each region sort its internal marker colors by boundary distance and prepend the base color if the first marker starts deeper than distance 0.
    - The wall dot tells which palette color should appear on the boundary band of that region, and the underlying wall color at that dotted cell tells whether to step forward or backward through the palette as depth increases.
    - The more common wall color means forward traversal, the less common wall color means reverse traversal; all wall pixels, including clue dots, stay unchanged.

    Key insight: Internal markers still say which colors belong to a region and in what depth order, but the wall has become an instruction carrier: its dot chooses the starting color, and its base color chooses the cyclic direction.
    """
    return _solve(grid, use_clues=True, use_direction=True)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [2, 4, 4]  # for 1/2/3 markers
START_DIST_MAX = 1

# ---- Distinct marker colors across the whole grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

# ---- Visibility constraints ----
ROTATION_REGIONS_MIN = 1
REVERSE_VISIBLE_REGIONS_MIN = 1

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 600


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall_major = rng.choice(colors)
    colors.remove(wall_major)
    wall_minor = rng.choice(colors)
    colors.remove(wall_minor)
    return bg, wall_major, wall_minor, colors


def _place_colored_walls(H: int, W: int, wall_major: int, wall_minor: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_base = np.full((H, W), -1, dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)
        wall_color = wall_major if rng.random() < 0.75 else wall_minor

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0 : c1 + 1] = True
            wall_base[rr, c0 : c1 + 1] = wall_color
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = True
            wall_base[r0 : r1 + 1, cc] = wall_color
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, wall_base, rects


def _rect_contains(rect, r: int, c: int) -> bool:
    return rect[0] <= r <= rect[2] and rect[1] <= c <= rect[3]


def _straight_wall_axis(wall_mask: np.ndarray, r: int, c: int) -> str | None:
    left = c - 1 >= 0 and wall_mask[r, c - 1]
    right = c + 1 < wall_mask.shape[1] and wall_mask[r, c + 1]
    up = r - 1 >= 0 and wall_mask[r - 1, c]
    down = r + 1 < wall_mask.shape[0] and wall_mask[r + 1, c]
    if left and right and not up and not down:
        return "h"
    if up and down and not left and not right:
        return "v"
    return None


def _region_wall_candidates(rect, wall_mask: np.ndarray, wall_base: np.ndarray) -> list[tuple[int, int, int]]:
    H, W = wall_mask.shape
    r0, c0, r1, c1 = rect
    raw: list[tuple[int, int]] = []

    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0 and wall_mask[r, c0 - 1]:
            raw.append((r, c0 - 1))
        if c1 + 1 < W and wall_mask[r, c1 + 1]:
            raw.append((r, c1 + 1))
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0 and wall_mask[r0 - 1, c]:
            raw.append((r0 - 1, c))
        if r1 + 1 < H and wall_mask[r1 + 1, c]:
            raw.append((r1 + 1, c))

    out: list[tuple[int, int, int]] = []
    seen = set()
    for r, c in raw:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        axis = _straight_wall_axis(wall_mask, r, c)
        if axis is None:
            continue
        base_color = int(wall_base[r, c])
        if base_color < 0:
            continue
        if axis == "h":
            if int(wall_base[r, c - 1]) != base_color or int(wall_base[r, c + 1]) != base_color:
                continue
        else:
            if int(wall_base[r - 1, c]) != base_color or int(wall_base[r + 1, c]) != base_color:
                continue
        out.append((r, c, base_color))
    return out


def _plan_region(rect, marker_pool: list[int], start_color: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return None

    options = list(range(1, upper + 1))
    weights = MARKER_COUNT_WEIGHTS[:upper]
    n = rng.choices(options, weights=weights)[0]
    start_d_hi = min(START_DIST_MAX, max_d - (n - 1))
    if start_d_hi < 0:
        return None
    start_d = rng.randint(0, start_d_hi)

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    colors = [None] * n
    if n == 1:
        clue_index = 0
    elif start_d > 0:
        clue_index = rng.randrange(n)
    else:
        clue_index = rng.randrange(1, n)
    colors[clue_index] = int(start_color)

    others = rng.sample([col for col in marker_pool if col != start_color], n - 1)
    oi = 0
    for i in range(n):
        if colors[i] is None:
            colors[i] = int(others[oi])
            oi += 1

    markers = []
    for k, col in enumerate(colors):
        d = start_d + k
        rr = br + dr * d
        cc = bc + dc * d
        markers.append((d, int(col), rr, cc))

    palette_len = n + (1 if start_d > 0 else 0)
    rotation_visible = start_d > 0 or clue_index != 0
    return {
        "markers": markers,
        "start_color": int(start_color),
        "marker_colors": set(int(col) for col in colors),
        "palette_len": palette_len,
        "rotation_visible": rotation_visible,
    }


def _stamp_region_markers(grid: np.ndarray, rects, plans, bg: int):
    used = set()
    marker_cells = set()
    for rect, plan in zip(rects, plans):
        for _d, col, rr, cc in plan["markers"]:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(int(col))
            marker_cells.add((int(rr), int(cc)))
    return used, marker_cells


def _adjacent_rect_ids(rects: list[tuple[int, int, int, int]], r: int, c: int) -> list[int]:
    ids: list[int] = []
    for idx, rect in enumerate(rects):
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if _rect_contains(rect, rr, cc):
                ids.append(idx)
                break
    return ids


def _build_candidate_map(rects, plans, wall_mask, wall_base, marker_cells):
    candidate_map: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for idx, rect in enumerate(rects):
        start_color = plans[idx]["start_color"]
        groups: dict[int, list[tuple[int, int]]] = {}
        for r, c, wall_color in _region_wall_candidates(rect, wall_mask, wall_base):
            if any((r + dr, c + dc) in marker_cells for dr, dc in DIR4):
                continue
            adj = _adjacent_rect_ids(rects, r, c)
            if idx not in adj:
                continue
            conflict = False
            for other in adj:
                if other != idx and start_color in plans[other]["marker_colors"]:
                    conflict = True
                    break
            if conflict:
                continue
            groups.setdefault(int(wall_color), []).append((int(r), int(c)))
        candidate_map[idx] = groups
    return candidate_map


def _choose_wall_roles(candidate_map, plans, wall_major: int, wall_minor: int, rng: random.Random):
    desired: dict[int, int] = {}
    reverse_eligible = [
        idx
        for idx, groups in candidate_map.items()
        if wall_minor in groups and plans[idx]["palette_len"] >= 3
    ]
    if not reverse_eligible:
        return None
    reverse_idx = rng.choice(reverse_eligible)

    for idx, groups in candidate_map.items():
        if idx == reverse_idx:
            desired[idx] = wall_minor
            continue
        if wall_major in groups:
            desired[idx] = wall_major
        elif wall_minor in groups:
            desired[idx] = wall_minor
        else:
            return None

    if not any(color == wall_major for color in desired.values()):
        major_only = [idx for idx, groups in candidate_map.items() if idx != reverse_idx and wall_major in groups]
        if not major_only:
            return None
        desired[rng.choice(major_only)] = wall_major

    return desired


def _assign_clue_cells(candidate_map, desired_wall_color, rng: random.Random):
    order = sorted(desired_wall_color, key=lambda idx: len(candidate_map[idx][desired_wall_color[idx]]))
    used = set()
    blocked = set()
    assigned: dict[int, tuple[int, int]] = {}

    for idx in order:
        choices = candidate_map[idx][desired_wall_color[idx]][:]
        rng.shuffle(choices)
        pick = next((cell for cell in choices if cell not in used and cell not in blocked), None)
        if pick is None:
            return None
        assigned[idx] = pick
        used.add(pick)
        r, c = pick
        blocked.update({pick, (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
    return assigned


def _stamp_wall_clues(grid: np.ndarray, wall_base: np.ndarray, clue_cells: dict[int, tuple[int, int]], plans):
    for idx, (r, c) in clue_cells.items():
        if wall_base[r, c] < 0:
            return False
        grid[r, c] = plans[idx]["start_color"]
    return True


def _validate_wall_counts(wall_base: np.ndarray, wall_major: int, wall_minor: int) -> bool:
    major_count = int(np.sum(wall_base == wall_major))
    minor_count = int(np.sum(wall_base == wall_minor))
    return major_count > minor_count > 0


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    _base, wall_mask, wall_colors, comp_index, infos = _build_region_infos(g)
    clues = _region_clues(g, wall_mask, wall_colors, comp_index, infos)
    reverse_color = wall_colors[1] if len(wall_colors) >= 2 else None

    clue_regions = len(clues)
    rotation_regions = 0
    reverse_regions = 0
    reverse_visible_regions = 0
    for idx, info in enumerate(infos):
        if idx not in clues:
            continue
        start_color, wall_color = clues[idx]
        palette = info["palette"]
        if start_color in palette and palette[0] != start_color:
            rotation_regions += 1
        if reverse_color is not None and wall_color == reverse_color:
            reverse_regions += 1
            if len(palette) >= 3:
                reverse_visible_regions += 1

    no_clue = _solve(g, use_clues=False, use_direction=False)
    clue_only = _solve(g, use_clues=True, use_direction=False)
    full = _solve(g, use_clues=True, use_direction=True)
    return {
        "clue_regions": clue_regions,
        "rotation_regions": rotation_regions,
        "reverse_regions": reverse_regions,
        "reverse_visible_regions": reverse_visible_regions,
        "clue_effect_cells": int(np.sum(no_clue != full)),
        "direction_effect_cells": int(np.sum(clue_only != full)),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    bg, wall_major, wall_minor, marker_pool = _pick_palette(rng)

    wall_mask, wall_base, rects = _place_colored_walls(H, W, wall_major, wall_minor, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None
    if not _validate_wall_counts(wall_base, wall_major, wall_minor):
        return None

    grid = np.full((H, W), bg, dtype=int)
    for color in (wall_major, wall_minor):
        grid[wall_base == color] = color

    start_colors = rng.sample(marker_pool, len(rects))
    plans = []
    for rect, start_color in zip(rects, start_colors):
        plan = _plan_region(rect, marker_pool, start_color, rng)
        if plan is None:
            return None
        plans.append(plan)

    if sum(1 for plan in plans if plan["rotation_visible"]) < ROTATION_REGIONS_MIN:
        return None

    used, marker_cells = _stamp_region_markers(grid, rects, plans, bg)
    if used is None or marker_cells is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None

    candidate_map = _build_candidate_map(rects, plans, wall_mask, wall_base, marker_cells)
    desired_wall_color = _choose_wall_roles(candidate_map, plans, wall_major, wall_minor, rng)
    if desired_wall_color is None:
        return None
    if sum(
        1
        for idx, color in desired_wall_color.items()
        if color == wall_minor and plans[idx]["palette_len"] >= 3
    ) < REVERSE_VISIBLE_REGIONS_MIN:
        return None

    clue_cells = _assign_clue_cells(candidate_map, desired_wall_color, rng)
    if clue_cells is None:
        return None
    if not _stamp_wall_clues(grid, wall_base, clue_cells, plans):
        return None

    try:
        metrics = _combo_metrics(grid)
        no_clue = _solve(grid, use_clues=False, use_direction=False)
        clue_only = _solve(grid, use_clues=True, use_direction=False)
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(no_clue, out):
        return None
    if np.array_equal(clue_only, out):
        return None
    if metrics["clue_regions"] != len(rects):
        return None
    if metrics["rotation_regions"] < ROTATION_REGIONS_MIN:
        return None
    if metrics["reverse_visible_regions"] < REVERSE_VISIBLE_REGIONS_MIN:
        return None
    if metrics["clue_effect_cells"] == 0 or metrics["direction_effect_cells"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
