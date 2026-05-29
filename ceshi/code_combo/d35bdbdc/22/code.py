# Combo variant of d35bdbdc (index 22)
# combo_spec: D2/3+D7/3
# Name: 多色路径分组+多条独立路径

"""
Design analysis:
  a_concrete_behavior_and_trace:
    original:
      - one grey trail selects exactly two destination sprites from its two endpoints
      - every destination then performs a global border->center lookup against all sprites
      - all non-destination sprites are deleted
    combo:
      - there are multiple thin path components, and path cells use two colors: 5 and 8
      - each 8-connected component still attaches to exactly two sprites, so connectivity
        determines a local sprite pair for that component
      - one path color appears on two separate components, so color alone is not enough;
        the solver must also split same-colored cells by component
      - all sprites touched by components survive, but recoloring is now restricted to
        the sprites that share the same path color as the source
    trace_example:
      - suppose columns A and B use grey, column C uses teal
      - among the four grey-linked sprites, exactly one sprite has border color k and
        center color sg; the other three grey-linked sprites have center color k
      - among the two teal-linked sprites, exactly one has border color k and center
        color st; the other teal-linked sprite has center color k
      - output keeps all six linked sprites; every grey destination center becomes sg,
        while the teal destination center becomes st
    difference:
      - original endpoint logic ran once globally; combo logic runs per component
      - original lookup pool was global; combo lookup pool is grouped by path color
      - because the two color groups deliberately reuse the same lookup key k, color is
        required to break ambiguity, while repeated same-color components require
        connectivity to collect the full group

  b_complexity_calibration:
    - exactly three components appear in every sample; component colors are distributed
      as 2+1 across the two path colors, so both dimensions are always visible
    - each color group has repeated evidence: the repeated-color group has three
      destinations sharing one source, and the single-color group has one destination
      sharing the same lookup key but a different copied center color
    - the copied colors come from only two source centers, keeping the pattern readable
    - the layout stays rigid and sparse so the extra grouping burden does not become noise

  c_hidden_assumptions_review:
    original_assumptions_that_break:
      - one call to find trail endpoints was enough because all path cells formed one trail
      - banning only grey from sprite detection was enough because there was one path color
      - keeping exactly two sprites was safe because only one endpoint pair mattered
      - a global border lookup was unambiguous because source borders were unique
    combo_fixes:
      - detect bordered sprites while excluding both path colors
      - enumerate connected components separately inside each path color
      - collect the two sprites attached to every component, then union them by color
      - resolve border->center lookups only inside each color group, because generator
        intentionally duplicates the source border across the two different colors

  d_placement_legality:
    - generation uses fixed top/bottom or left/right edge slots, so six linked sprites
      and three paths fit on 10x10 without overlap
    - every slot has an explicit port direction, so each path endpoint is guaranteed to
      be two Manhattan steps from the corresponding sprite center with a border cell in
      between
    - paths are straight disjoint segments, so component detection is stable
    - optional distractors are placed only after the linked structure is built and only in
      cells that avoid overlap and 4-touching

  e_visual_observability:
    - a human can see three separate path segments immediately
    - one color repeats across two disconnected segments, so "same color but different
      component" is visibly present in every sample
    - the same center lookup color appears under both path colors, yet outputs differ by
      path color, making the color grouping legible from examples

  f_visual_cleanliness:
    - sprites only occupy edge slots and paths only occupy three clean lanes
    - no component crossings, no layered overlaps, and at most one unconnected distractor
      keep the board easy to parse
    - all kept sprites remain in place; only centers change, so the transformation is
      visually stable

  g_change_inferability:
    - the repeated lookup key across the two path colors is the main structural clue:
      input centers alone do not determine the output color
    - same-colored disconnected components all resolve to the same copied center color,
      providing repeated evidence for the color-grouped lookup rule
    - connectivity is also inferable because the repeated path color is split across two
      physically separate lanes, yet both lanes contribute linked sprites to the same
      color group

  solver_assumptions_checked_against_generator:
    - background is fixed to 0; this is safer than boundary-majority inference because
      many sprites and paths intentionally touch the outer frame
    - only path colors 5 and 8 are used for trails and are never valid sprite centers
    - each path component is a straight one-stroke segment with exactly two usable ports
    - within each color group, any center color that matches a border does so for exactly
      one sprite in that same group
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import components_mask, neighbors4, neighbors8, paste


GRID_H, GRID_W = 10, 10
BG = 0
PATH_COLORS = (5, 8)
SPRITE_PALETTE = [1, 2, 3, 4, 6, 7, 9]
DISTRACTOR_COUNTS = [0, 1]
GEN_RETRIES = 200

LAYOUTS = {
    "vertical": {
        "components": [
            ({"center": (0, 0), "dir": (1, 0)}, {"center": (9, 0), "dir": (-1, 0)}),
            ({"center": (0, 4), "dir": (1, 0)}, {"center": (9, 4), "dir": (-1, 0)}),
            ({"center": (0, 8), "dir": (1, 0)}, {"center": (9, 8), "dir": (-1, 0)}),
        ],
    },
    "horizontal": {
        "components": [
            ({"center": (0, 0), "dir": (0, 1)}, {"center": (0, 9), "dir": (0, -1)}),
            ({"center": (4, 0), "dir": (0, 1)}, {"center": (4, 9), "dir": (0, -1)}),
            ({"center": (8, 0), "dir": (0, 1)}, {"center": (8, 9), "dir": (0, -1)}),
        ],
    },
}


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    paste(grid, patch, center[0] - 1, center[1] - 1, transparent=BG, clip=True)


def _detect_bordered_sprites(g: np.ndarray, bg: int, path_colors: set[int]) -> list[dict]:
    """Find 4-connected two-color components with a unique center pixel."""
    mask = g != bg
    for path_color in path_colors:
        mask &= g != path_color
    comps = components_mask(mask, connectivity=4)

    sprites: list[dict] = []
    for pts in comps:
        coords = [(int(r), int(c)) for r, c in pts]
        cols = [int(g[r, c]) for r, c in coords]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {color: cols.count(color) for color in uniq}
        center_color = min(counts, key=lambda color: counts[color])
        if counts[center_color] != 1:
            continue
        border_color = next(color for color in uniq if color != center_color)
        if border_color == bg or border_color in path_colors or center_color in path_colors:
            continue
        center_pos = next((rc for rc in coords if int(g[rc]) == center_color), None)
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": coords,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": (int(center_pos[0]), int(center_pos[1])),
            }
        )
    return sprites


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, candidate_endpoints: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return endpoint cells that form a valid 2-step port for the sprite."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in candidate_endpoints:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _find_component_endpoints(
    g: np.ndarray, cells: set[tuple[int, int]], sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two path endpoints inside one 8-connected component."""
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
    assert path is not None, "could not trace a path component to two ports"
    return path[0], path[-1]


def _pick_endpoint_sprite(
    endpoint: tuple[int, int],
    other_endpoint: tuple[int, int],
    sprites: list[dict],
    g: np.ndarray,
    endpoints: set[tuple[int, int]],
    used_ids: set[int],
) -> dict:
    cands = [sp for sp in sprites if id(sp) not in used_ids and endpoint in _ports_for_sprite(sp, g, endpoints)]
    assert cands, "path endpoint did not resolve to a sprite"
    return max(
        cands,
        key=lambda sp: abs(sp["center_pos"][0] - other_endpoint[0]) + abs(sp["center_pos"][1] - other_endpoint[1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor path-linked bordered sprites by combining path color groups with independent path components.

    Input structure:
    - The board contains small bordered sprites together with thin path segments of color 5 and/or 8.
    - Path cells are split into multiple 8-connected components, and each component's two endpoints touch
      two sprites through valid 2-step ports.
    - The same path color may appear on more than one disconnected component, so both color and connectivity
      matter when grouping sprites.

    Rule:
    - Resolve every path component separately to find its two attached sprites, and keep the union of all
      such sprites while deleting unconnected sprites.
    - Group the kept sprites by the color of the path component(s) that touch them.
    - Inside each color group only, if a sprite's center color matches another sprite's border color, replace
      that center with the matched sprite's center color. Sprites with no same-color match stay unchanged.

    Key insight:
    - Path connectivity tells which sprite pairs participate, but path color tells which lookup pool they belong
      to; the repeated path color creates multiple disconnected components that still share one color-specific
      source, while the two different colors can reuse the same lookup key without conflicting.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    path_color_set = set(PATH_COLORS)

    sprites = _detect_bordered_sprites(g, bg, path_color_set)
    assert sprites, "expected bordered sprites"

    group_members: dict[int, list[dict]] = defaultdict(list)
    group_ids: dict[int, set[int]] = defaultdict(set)
    keep_ids: set[int] = set()

    for path_color in PATH_COLORS:
        comps = components_mask(g == path_color, connectivity=8)
        for comp in comps:
            cells = {(int(r), int(c)) for r, c in comp}
            ep1, ep2 = _find_component_endpoints(g, cells, sprites)
            eps = {ep1, ep2}
            used_here: set[int] = set()
            first = _pick_endpoint_sprite(ep1, ep2, sprites, g, eps, used_here)
            used_here.add(id(first))
            second = _pick_endpoint_sprite(ep2, ep1, sprites, g, eps, used_here)
            assert id(first) != id(second), "one component must touch two different sprites"
            for sp in (first, second):
                sid = id(sp)
                if sid not in group_ids[path_color]:
                    group_members[path_color].append(sp)
                    group_ids[path_color].add(sid)
                keep_ids.add(sid)

    assert keep_ids, "expected at least one path-linked sprite"

    out = g.copy()
    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for sp in sprites:
        if id(sp) not in keep_ids:
            continue
        for r, c in sp["pts"]:
            if out[r, c] == bg or out[r, c] in path_color_set or keep[r, c]:
                continue
            keep[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] == bg or out[nr, nc] in path_color_set:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg)
    for path_color in path_color_set:
        drop &= out != path_color
    drop &= ~keep
    out[drop] = bg

    for path_color, members in group_members.items():
        by_border: dict[int, list[dict]] = defaultdict(list)
        for sp in members:
            by_border[int(sp["border"])].append(sp)
        for sp in members:
            matches = by_border.get(int(sp["center"]), [])
            if not matches:
                continue
            assert len(matches) == 1, f"ambiguous same-color lookup for path color {path_color}"
            src = matches[0]
            rr, cc = sp["center_pos"]
            out[rr, cc] = int(src["center"])

    return out


def _overlaps(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> bool:
    """Reject overlap or 4-touching against existing non-path sprites and paths."""
    h, w = grid.shape
    r0 = center[0] - 1
    c0 = center[1] - 1
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
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


def _port_cell(slot: dict) -> tuple[int, int]:
    cr, cc = slot["center"]
    dr, dc = slot["dir"]
    return cr + 2 * dr, cc + 2 * dc


def _straight_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if start[0] == end[0]:
        row = start[0]
        c0, c1 = sorted((start[1], end[1]))
        return [(row, c) for c in range(c0, c1 + 1)]
    if start[1] == end[1]:
        col = start[1]
        r0, r1 = sorted((start[0], end[0]))
        return [(r, col) for r in range(r0, r1 + 1)]
    raise ValueError("ports must align horizontally or vertically")


def _draw_path_if_clear(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    for r, c in cells:
        if not (0 <= r < GRID_H and 0 <= c < GRID_W):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _iter_all_centers() -> list[tuple[int, int]]:
    return [(r, c) for r in range(GRID_H) for c in range(GRID_W)]


def _place_optional_distractor(
    grid: np.ndarray,
    shape_type: str,
    forbidden_center: int,
) -> bool:
    border_choices = [c for c in SPRITE_PALETTE if c != forbidden_center]
    random.shuffle(border_choices)
    for border in border_choices:
        center_choices = [c for c in SPRITE_PALETTE if c != border]
        random.shuffle(center_choices)
        for center_color in center_choices:
            patch = _make_sprite_patch(shape_type, border, center_color)
            centers = _iter_all_centers()
            random.shuffle(centers)
            for center in centers:
                if _overlaps(grid, patch, center):
                    continue
                _place_sprite(grid, patch, center)
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    layout_name = random.choice(list(LAYOUTS))
    layout = LAYOUTS[layout_name]

    repeated_color = random.choice(PATH_COLORS)
    singleton_color = PATH_COLORS[1] if repeated_color == PATH_COLORS[0] else PATH_COLORS[0]
    singleton_index = random.randrange(3)
    component_colors = [repeated_color, repeated_color, repeated_color]
    component_colors[singleton_index] = singleton_color

    lookup_key = random.choice(SPRITE_PALETTE)
    remainder = [c for c in SPRITE_PALETTE if c != lookup_key]
    random.shuffle(remainder)
    dest_borders = remainder[:4]
    source_centers = remainder[4:6]

    source_center_by_color = {
        repeated_color: source_centers[0],
        singleton_color: source_centers[1],
    }

    slots: list[dict] = []
    for comp_index, (slot_a, slot_b) in enumerate(layout["components"]):
        for side_index, slot in enumerate((slot_a, slot_b)):
            slots.append(
                {
                    "component_index": comp_index,
                    "side_index": side_index,
                    "path_color": component_colors[comp_index],
                    "center": slot["center"],
                    "dir": slot["dir"],
                }
            )

    repeated_slots = [slot for slot in slots if slot["path_color"] == repeated_color]
    singleton_slots = [slot for slot in slots if slot["path_color"] == singleton_color]
    source_repeated = random.choice(repeated_slots)
    source_singleton = random.choice(singleton_slots)
    source_keys = {
        (source_repeated["component_index"], source_repeated["side_index"]),
        (source_singleton["component_index"], source_singleton["side_index"]),
    }

    border_iter = iter(dest_borders)
    slot_meta: list[dict] = []
    for slot in slots:
        slot_key = (slot["component_index"], slot["side_index"])
        path_color = slot["path_color"]
        if slot_key in source_keys:
            border_color = lookup_key
            center_color = source_center_by_color[path_color]
            is_source = True
        else:
            border_color = next(border_iter)
            center_color = source_center_by_color[path_color]
            is_source = False
        slot_meta.append(
            {
                **slot,
                "border": border_color,
                "output_center": center_color,
                "input_center": center_color if is_source else lookup_key,
                "is_source": is_source,
            }
        )

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for meta in slot_meta:
        patch = _make_sprite_patch(shape_type, int(meta["border"]), int(meta["output_center"]))
        if _overlaps(output_grid, patch, meta["center"]):
            return None
        _place_sprite(output_grid, patch, meta["center"])

    for comp_index, pair in enumerate(layout["components"]):
        color = component_colors[comp_index]
        start = _port_cell(pair[0])
        end = _port_cell(pair[1])
        if not _draw_path_if_clear(output_grid, _straight_path(start, end), color):
            return None

    input_grid = output_grid.copy()
    for meta in slot_meta:
        cr, cc = meta["center"]
        input_grid[cr, cc] = int(meta["input_center"])

    if random.choice(DISTRACTOR_COUNTS):
        if not _place_optional_distractor(input_grid, shape_type, forbidden_center=lookup_key):
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
