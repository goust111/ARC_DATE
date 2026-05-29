# Variant of d35bdbdc
# Dimension: D2 — 路径连通性判定 (match_cond)
# Change: 灰色路径端点必须连接到精灵的中心位置，以此判定哪些精灵是目标精灵(d)... → 路径颜色编码不同组

import random

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


# Design analysis for this D2 variant
# -----------------------------------
# a) Concrete behavior:
#    The original task had exactly one gray trail, so the whole matching stage only had
#    one global pair of endpoints and therefore only two destination sprites. Here there
#    are two thin trails with different fixed colors: gray=5 and teal=8. Each colored
#    trail is its own 8-connected component and has its own two endpoints. The solver
#    must first separate path cells by color, then find the two sprites attached to each
#    color-specific trail, and keep all of those destinations. Example trace:
#    - old logic: a single gray trail touches only two corners -> keep those 2 sprites.
#    - new logic: a gray trail touches the top pair, a teal trail touches the bottom pair
#      -> keep 4 sprites total, recoloring each by the same border->center lookup rule.
#    The changed stage is therefore destination selection, not color transfer itself.
#
# b) Complexity calibration:
#    The grouping signal is explicit but not excessive: there are exactly two path colors
#    and both appear in every generated sample. Each color supports two destinations, so
#    the solver cannot collapse the task back to the original "pick one pair" behavior.
#    The transfer rule remains one lookup hop, which keeps the task readable.
#
# c) Old implicit assumptions that must change:
#    - Old `_detect_bordered_sprites(..., grey)` assumed exactly one forbidden path color.
#      Now every sprite detector must exclude both path colors.
#    - Old endpoint detection ran once on all gray cells. That silently depends on there
#      being only one trail component and only one relevant color. Now endpoints must be
#      found per colored component.
#    - Old `dest_ids` contained exactly 2 sprites, so deleting all others was safe. Now
#      deletion must preserve the union of destinations from both color groups.
#    - Old generator only needed one source-destination lookup chain. The new generator
#      must guarantee two distinct groups so the second path color changes the output.
#
# d) Placement legality:
#    Four destination sprites occupy the four corners as 3x3 plus-shaped patches. Two
#    colored trails live in non-overlapping center corridors (horizontal or vertical
#    layout), so they never overwrite each other or the sprites. Source sprites are then
#    searched from a small set of legal interior positions; this shape restriction is the
#    minimal adaptation needed to fit 4 destinations + 2 paths + 2 sources on a 10x10
#    board without accidental 4-connected sprite merges.
#
# e) Visual observability:
#    In every sample both gray and teal trails are present. The output keeps the two
#    destination sprites touched by gray and also the two touched by teal; sources vanish.
#    A human can see that path color partitions the kept corner sprites into two groups.
#
# f) Visual cleanliness:
#    The layout is sparse and layered only once: corner sprites, two mid-board trails, and
#    two side sources. No path overlaps any sprite, and the two trail colors use disjoint
#    corridors, so the grouping is easy to read.
#
# g) Color-change inferability:
#    Only two transfer groups exist. Within each group, both destination centers in the
#    input share the same lookup color (the matching source border), and both become the
#    same source-center color in the output. This gives repeated evidence for the lookup
#    rule while the path colors explain which four sprites survive.
#
# New solver assumptions checked against generator:
# - Path colors are fixed to 5 and 8 and are never used as sprite border/center colors.
# - Every colored trail component is a thin one-stroke path with two usable endpoints.
# - Each destination center color matches exactly one source border color.
# - No source sprite is itself connected to a trail endpoint.


GRID_H, GRID_W = 10, 10
BG = 0
PATH_COLORS = (5, 8)
SPRITE_PALETTE = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 200

DEST_LAYOUTS = {
    "horizontal": {
        "dest_slots": {
            "tl": (0, 0),
            "tr": (0, 7),
            "bl": (7, 0),
            "br": (7, 7),
        },
        "groups": [
            {"members": ("tl", "tr"), "ports": ((1, 3), (1, 6))},
            {"members": ("bl", "br"), "ports": ((8, 3), (8, 6))},
        ],
        "source_slots": [(3, 1), (3, 2), (3, 5), (3, 6), (4, 1), (4, 2), (4, 5), (4, 6)],
    },
    "vertical": {
        "dest_slots": {
            "tl": (0, 0),
            "tr": (0, 7),
            "bl": (7, 0),
            "br": (7, 7),
        },
        "groups": [
            {"members": ("tl", "bl"), "ports": ((3, 1), (6, 1))},
            {"members": ("tr", "br"), "ports": ((3, 8), (6, 8))},
        ],
        "source_slots": [(1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4), (6, 3), (6, 4)],
    },
}


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite with the center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _detect_bordered_sprites(g: np.ndarray, bg: int, path_colors: set[int]) -> list[dict]:
    """
    A bordered sprite is a 4-connected non-path component with exactly two colors:
    one unique center pixel and one border color repeated around it.
    """
    mask = (g != bg)
    for pc in path_colors:
        mask &= (g != pc)
    comps = components_mask(mask, connectivity=4)

    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color == bg or border_color in path_colors or center_color in path_colors:
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A path endpoint is a port for this sprite if it is 2 Manhattan steps from the center
    and the intermediate cell in that direction uses the sprite border color.
    """
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _find_component_endpoints(
    cells: set[tuple[int, int]], g: np.ndarray, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoints of one colored trail under 8-connectivity."""
    h, w = g.shape
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == total:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path is not None, "could not trace colored path endpoints"
    return path[0], path[-1]


def _destinations_for_path_component(
    g: np.ndarray, sprites: list[dict], cells: set[tuple[int, int]],
) -> list[dict]:
    """Resolve the 2 destination sprites attached to one colored path component."""
    ep1, ep2 = _find_component_endpoints(cells, g, sprites)
    endpoints = [ep1, ep2]
    endpoint_set = set(endpoints)
    opts = {
        ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, endpoint_set)]
        for ep in endpoints
    }

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[ep1]) <= len(opts[ep2]) else [1, 0]
    used: set[int] = set()
    for idx in order:
        ep = endpoints[idx]
        other = endpoints[1 - idx]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        assert cands, "path endpoint did not resolve to a destination sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[idx] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 destinations per colored path"
    return dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer lookup colors into bordered sprites, grouped by multiple colored path trails.

    Input structure:
    - The board contains several 3x3 bordered sprites plus multiple thin path components.
    - Paths are not all the same color: gray (5) and teal (8) each mark a separate trail,
      and each trail touches exactly two destination sprites through endpoint ports.

    Rule:
    - Process each path color component independently and find the two sprites attached to
      that component's endpoints.
    - Keep every destination sprite reached by any colored trail; remove all other sprites.
    - For each kept destination, use its original center color as a lookup key: find the
      source sprite whose border color matches that key, then copy the source center color
      into the destination center.

    Key insight:
    - The path colors first partition the board into independent destination groups, and
      only after those groups are found does the old border->center color transfer apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    path_color_set = set(PATH_COLORS)

    sprites = _detect_bordered_sprites(g, bg, path_color_set)
    assert sprites, "expected bordered sprites in the grid"

    destinations: list[dict] = []
    dest_ids: set[int] = set()
    for path_color in PATH_COLORS:
        comps = components_mask(g == path_color, connectivity=8)
        assert comps, f"missing expected path color {path_color}"
        for comp in comps:
            cells = set((int(r), int(c)) for r, c in comp)
            pair = _destinations_for_path_component(g, sprites, cells)
            for sp in pair:
                sid = id(sp)
                assert sid not in dest_ids, "one sprite was attached to multiple path groups"
                destinations.append(sp)
                dest_ids.add(sid)

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in destinations:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] not in path_color_set and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc]:
                    continue
                if out[nr, nc] == bg or out[nr, nc] in path_color_set:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg)
    for pc in path_color_set:
        drop &= (out != pc)
    drop &= (~keep)
    out[drop] = bg

    for d in destinations:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination lookup color should match one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=False)


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap or direct 4-touching between two non-path sprite components."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return True
            if grid[rr, cc] != BG:
                return True
            patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, *PATH_COLORS):
                return True
    return False


def _draw_path_if_clear(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    for r, c in cells:
        if not (0 <= r < GRID_H and 0 <= c < GRID_W):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _path_variants(start: tuple[int, int], end: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """A few deterministic thin path shapes between the fixed ports used by the generator."""
    sr, sc = start
    er, ec = end
    if sr == er:
        lo, hi = sorted((sc, ec))
        row = sr
        inner = row + 1 if row < GRID_H // 2 else row - 1
        outer = row + 2 if row < GRID_H // 2 else row - 2
        return [
            [(row, c) for c in range(lo, hi + 1)],
            [start] + [(inner, c) for c in range(lo + 1, hi)] + [end],
            [
                start,
                (inner, lo),
                *[(outer, c) for c in range(lo + 1, hi)],
                (inner, hi),
                end,
            ],
        ]

    top, bot = sorted((sr, er))
    col = sc
    inner = col + 1 if col < GRID_W // 2 else col - 1
    outer = col + 2 if col < GRID_W // 2 else col - 2
    return [
        [(r, col) for r in range(top, bot + 1)],
        [start] + [(r, inner) for r in range(top + 1, bot)] + [end],
        [
            start,
            (top, inner),
            *[(r, outer) for r in range(top + 1, bot)],
            (bot, inner),
            end,
        ],
    ]


def _choose_paths(layout_name: str, color_order: tuple[int, int]) -> dict[int, list[tuple[int, int]]] | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    chosen: dict[int, list[tuple[int, int]]] = {}
    groups = list(DEST_LAYOUTS[layout_name]["groups"])
    random.shuffle(groups)
    for group, color in zip(groups, color_order, strict=True):
        variants = _path_variants(group["ports"][0], group["ports"][1])
        random.shuffle(variants)
        placed = False
        for cells in variants:
            if _draw_path_if_clear(grid, cells, color):
                chosen[color] = cells
                placed = True
                break
        if not placed:
            return None
    return chosen


def _gen_attempt() -> np.ndarray | None:
    shape_type = "plus"
    layout_name = random.choice(list(DEST_LAYOUTS))
    layout = DEST_LAYOUTS[layout_name]

    border_colors = random.sample(SPRITE_PALETTE, 6)
    dest_border = {
        "tl": border_colors[0],
        "tr": border_colors[1],
        "bl": border_colors[2],
        "br": border_colors[3],
    }
    source_border = {
        PATH_COLORS[0]: border_colors[4],
        PATH_COLORS[1]: border_colors[5],
    }

    source_center_choices = [c for c in SPRITE_PALETTE if c not in source_border.values()]
    source_center_values = random.sample(source_center_choices, 2)
    source_center = {
        PATH_COLORS[0]: source_center_values[0],
        PATH_COLORS[1]: source_center_values[1],
    }

    path_cells = _choose_paths(layout_name, PATH_COLORS)
    if path_cells is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)

    group_for_color: dict[int, dict] = {}
    for color, cells in path_cells.items():
        endpoints = {cells[0], cells[-1]}
        matched = None
        for group in layout["groups"]:
            if endpoints == set(group["ports"]):
                matched = group
                break
        if matched is None:
            return None
        group_for_color[color] = matched

    for slot_name, (r0, c0) in layout["dest_slots"].items():
        group_color = None
        for color, group in group_for_color.items():
            if slot_name in group["members"]:
                group_color = color
                break
        assert group_color is not None
        patch = _make_sprite_patch(shape_type, dest_border[slot_name], source_center[group_color])
        _place_sprite(output_grid, patch, r0, c0)

    for color, cells in path_cells.items():
        if not _draw_path_if_clear(output_grid, cells, color):
            return None

    input_grid = output_grid.copy()
    for slot_name, (r0, c0) in layout["dest_slots"].items():
        group_color = None
        for color, group in group_for_color.items():
            if slot_name in group["members"]:
                group_color = color
                break
        assert group_color is not None
        input_grid[r0 + 1, c0 + 1] = source_border[group_color]

    source_items = list(PATH_COLORS)
    random.shuffle(source_items)
    for color in source_items:
        patch = _make_sprite_patch(shape_type, source_border[color], source_center[color])
        slots = list(layout["source_slots"])
        random.shuffle(slots)
        placed = False
        for r0, c0 in slots:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    if np.array_equal(out, output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
