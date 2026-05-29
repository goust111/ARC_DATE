# Variant of d35bdbdc
# Dimension: D7 — 灰色路径结构 (meta)
# Change: 灰色像素构成连通路径，端点连接到目标精灵的中心位置... → 多条独立路径分别连接不同精灵对

"""
design_analysis:
  variant_goal:
    keep:
      - 10x10 canvas, bg=0, grey=5
      - same bordered-sprite vocabulary as the source task
      - color-chain rule: destination.center is a lookup key for a source border,
        and the source center is copied into the destination center
    change:
      - replace the single global grey trail with two separate grey path components
      - each grey component links exactly one ordered sprite pair
      - the linked pair is now local: one sprite on that path is the destination,
        the other sprite on the same path is the source

  concrete_behavior:
    input:
      - 4 connected sprites arranged as 2 grey-linked pairs, plus 0-1 unconnected distractor sprites
      - each grey component is a thin straight segment whose two endpoints are valid sprite ports
      - within each pair, exactly one sprite has center_color == partner.border_color;
        that sprite is the destination
      - the source sprite's center uses a non-border color, so direction is visually asymmetric
    output:
      - keep both sprites from every grey-linked pair and keep all grey path components
      - for each pair, overwrite the destination center with the source center
      - delete any unconnected distractor sprites

  behavior_trace_example:
    original:
      - one grey component has two endpoints, so solver extracts exactly two destination sprites
      - source lookup is global by border color, and every non-destination sprite is removed
    variant:
      - grey splits into two components, so solver must process components independently
      - component A might connect sprite X(border 2, center 7) with sprite Y(border 4, center 2)
      - Y.center == X.border, so Y is destination and changes from 2 to 7
      - component B does the same with another pair; both connected sources remain visible
    difference:
      - "which sprites participate" is no longer a single global set of two endpoints
      - deletion now depends on path-component membership, not on belonging to one special pair
      - transfer is resolved inside each path pair, not by a single whole-grid endpoint stage

  complexity_calibration:
    - exactly two grey components are used in every generated sample so the D7 change is always visible
    - each sample still has only 4-5 sprites total, avoiding a crowded grid
    - border colors stay distinct, while source centers are chosen from colors not used as borders;
      this prevents accidental bidirectional matches and keeps each pair legible
    - destination side is randomized per pair, so the rule cannot be reduced to "copy left to right"

  hidden_assumptions_review:
    original_single_path_dependencies:
      - _find_trail_endpoints assumes all grey cells belong to one traceable component
      - trail_eps = {te1, te2} assumes exactly two relevant ports in the whole grid
      - dest is a fixed list of length 2, so the output can only preserve two sprites
      - by_border is used globally, which ignores any per-path pairing information
    variant_fixes:
      - detect 8-connected grey components first, then find endpoints inside each component
      - choose the two sprites attached to that specific component's endpoints
      - collect all path-linked sprites as survivors
      - perform color transfer only between the two sprites attached to the same grey component

  placement_legality:
    - generation uses a small set of clean layouts with two non-touching lanes
    - every sprite slot has an explicit port direction, so each path endpoint is guaranteed valid
    - path cells are drawn only after sprite placement and never overlap sprites
    - optional distractors are placed in reserved center slots away from both grey lanes

  visual_observability:
    - two disconnected grey segments are the main visual cue; this is immediately distinct from the source task
    - each connected pair stays in the output, while the isolated distractor disappears
    - within each pair, only one center changes, and it changes to the partner's center color
    - because destination side varies across samples, the human-visible invariant is the local color-chain,
      not a fixed spatial direction

  cleanliness:
    - components occupy separate top/bottom or left/right lanes to avoid tangled overlaps
    - only one grey color and one sprite shape family appear per example
    - the center distractor, when present, does not touch grey paths, so the "connected vs unconnected"
      distinction is visually crisp

  color_inference:
    - border colors remain few and distinct
    - source centers come from non-border colors, making the copied color easy to track
    - only destination centers encode border-color lookups, so examples expose a clear structural clue
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        coords = [(int(r), int(c)) for r, c in pts]
        cols = [int(g[r, c]) for r, c in coords]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color == grey:
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
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it is exactly 2 steps away from the sprite center in a cardinal direction
      - the intermediate cell is the sprite border color
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in candidate_endpoints:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _grey_components(g: np.ndarray, grey: int) -> list[set[tuple[int, int]]]:
    comps = components_mask(g == grey, connectivity=8)
    return [{(int(r), int(c)) for r, c in comp} for comp in comps]


def _find_component_endpoints(
    g: np.ndarray, grey_cells: set[tuple[int, int]], sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the 2 endpoints of one grey path component under 8-connectivity."""
    h, w = g.shape
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in grey_cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in grey_cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (
                (er, ec) in grey_cells
                and 0 <= ir < h
                and 0 <= ic < w
                and int(g[ir, ic]) == b
            ):
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(grey_cells))
    total = len(grey_cells)

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
    assert path, "could not trace grey component as one-stroke trail"
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
    assert cands, "expected grey endpoint to touch a bordered sprite"
    return max(
        cands,
        key=lambda sp: abs(sp["center_pos"][0] - other_endpoint[0]) + abs(sp["center_pos"][1] - other_endpoint[1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors inside each grey-linked sprite pair, keeping only path-connected sprites.

    Input structure:
    - The grid contains bordered sprites plus grey (5) path cells.
    - Grey cells are split into multiple independent 8-connected path components; each component's two endpoints
      touch two different sprites via valid ports.

    Rule:
    - Process each grey component separately and identify the two sprites attached to its endpoints.
    - Within that local pair, if one sprite's center color equals the other sprite's border color, that sprite is
      the destination and its center is replaced by the partner's center color.
    - Keep all sprites that participate in any grey-linked pair, preserve the grey paths, and delete unconnected sprites.

    Key insight:
    - The grey structure no longer only says "these are the targets"; each disconnected grey segment defines a local
      source/destination pairing, so color transfer must be resolved component-by-component rather than globally.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    grey_comps = _grey_components(g, grey)
    assert grey_comps, "expected at least one grey path component"

    connected_pairs: list[tuple[dict, dict]] = []
    keep_ids: set[int] = set()
    for comp in grey_comps:
        ep1, ep2 = _find_component_endpoints(g, comp, sprites)
        eps = {ep1, ep2}
        used_here: set[int] = set()
        first = _pick_endpoint_sprite(ep1, ep2, sprites, g, eps, used_here)
        used_here.add(id(first))
        second = _pick_endpoint_sprite(ep2, ep1, sprites, g, eps, used_here)
        assert id(first) != id(second), "each grey component should link two different sprites"
        connected_pairs.append((first, second))
        keep_ids.add(id(first))
        keep_ids.add(id(second))

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
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for left, right in connected_pairs:
        if int(left["center"]) == int(right["border"]):
            lr, lc = left["center_pos"]
            out[lr, lc] = int(right["center"])
        if int(right["center"]) == int(left["border"]):
            rr, rc = right["center_pos"]
            out[rr, rc] = int(left["center"])

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
PAIR_COUNT = 2
DISTRACTOR_COUNTS = [0, 1]
GEN_RETRIES = 200

LAYOUTS = [
    {
        "pairs": [
            ({"center": (1, 1), "dir": (0, 1)}, {"center": (1, 8), "dir": (0, -1)}),
            ({"center": (8, 1), "dir": (0, 1)}, {"center": (8, 8), "dir": (0, -1)}),
        ],
        "distractors": [(4, 4), (5, 5)],
    },
    {
        "pairs": [
            ({"center": (1, 1), "dir": (1, 0)}, {"center": (8, 1), "dir": (-1, 0)}),
            ({"center": (1, 8), "dir": (1, 0)}, {"center": (8, 8), "dir": (-1, 0)}),
        ],
        "distractors": [(4, 4), (5, 5)],
    },
    {
        "pairs": [
            ({"center": (2, 1), "dir": (0, 1)}, {"center": (2, 8), "dir": (0, -1)}),
            ({"center": (7, 1), "dir": (0, 1)}, {"center": (7, 8), "dir": (0, -1)}),
        ],
        "distractors": [(4, 4), (4, 5), (5, 4), (5, 5)],
    },
    {
        "pairs": [
            ({"center": (1, 2), "dir": (1, 0)}, {"center": (8, 2), "dir": (-1, 0)}),
            ({"center": (1, 7), "dir": (1, 0)}, {"center": (8, 7), "dir": (-1, 0)}),
        ],
        "distractors": [(4, 4), (4, 5), (5, 4), (5, 5)],
    },
]


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
    r0 = center[0] - 1
    c0 = center[1] - 1
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _overlaps(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> bool:
    h, w = grid.shape
    r0 = center[0] - 1
    c0 = center[1] - 1
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _port_cell(slot: dict) -> tuple[int, int]:
    cr, cc = slot["center"]
    dr, dc = slot["dir"]
    return cr + 2 * dr, cc + 2 * dc


def _straight_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if start[0] == end[0]:
        r = start[0]
        c0, c1 = sorted([start[1], end[1]])
        return [(r, c) for c in range(c0, c1 + 1)]
    if start[1] == end[1]:
        c = start[1]
        r0, r1 = sorted([start[0], end[0]])
        return [(r, c) for r in range(r0, r1 + 1)]
    raise ValueError("layout ports must align horizontally or vertically")


def _place_pair_output(
    grid: np.ndarray,
    shape_type: str,
    slot_a: dict,
    slot_b: dict,
    border_a: int,
    border_b: int,
    source_center: int,
    dest_is_a: bool,
) -> tuple[list[tuple[int, int]], dict]:
    if dest_is_a:
        center_a = source_center
        center_b = source_center
        input_center_a = border_b
        input_center_b = source_center
    else:
        center_a = source_center
        center_b = source_center
        input_center_a = source_center
        input_center_b = border_a

    patch_a = _make_sprite_patch(shape_type, border_a, center_a)
    patch_b = _make_sprite_patch(shape_type, border_b, center_b)
    assert not _overlaps(grid, patch_a, slot_a["center"])
    _place_sprite(grid, patch_a, slot_a["center"])
    assert not _overlaps(grid, patch_b, slot_b["center"])
    _place_sprite(grid, patch_b, slot_b["center"])

    path_cells = _straight_path(_port_cell(slot_a), _port_cell(slot_b))
    for r, c in path_cells:
        assert grid[r, c] == BG
        grid[r, c] = GREY

    return path_cells, {
        "slot_a": slot_a,
        "slot_b": slot_b,
        "border_a": border_a,
        "border_b": border_b,
        "source_center": source_center,
        "dest_is_a": dest_is_a,
        "input_center_a": input_center_a,
        "input_center_b": input_center_b,
    }


def _build_input_from_output(
    output_grid: np.ndarray,
    shape_type: str,
    pair_meta: list[dict],
    distractor: tuple[int, int, int] | None,
) -> np.ndarray:
    inp = output_grid.copy()
    for meta in pair_meta:
        cr_a, cc_a = meta["slot_a"]["center"]
        cr_b, cc_b = meta["slot_b"]["center"]
        inp[cr_a, cc_a] = meta["input_center_a"]
        inp[cr_b, cc_b] = meta["input_center_b"]
    if distractor is not None:
        border, center, pos = distractor
        patch = _make_sprite_patch(shape_type, border, center)
        assert not _overlaps(inp, patch, pos)
        _place_sprite(inp, patch, pos)
    return inp


def _count_sprite_like_components(grid: np.ndarray) -> int:
    bg = _bg_boundary(grid)
    grey = GREY
    return len(_detect_bordered_sprites(grid, bg, grey))


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    layout = random.choice(LAYOUTS)
    distractor_count = random.choice(DISTRACTOR_COUNTS)

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, 4 + distractor_count)
    non_border_colors = [c for c in palette if c not in border_colors]
    random.shuffle(non_border_colors)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    pair_meta: list[dict] = []

    for i, (slot_a, slot_b) in enumerate(layout["pairs"][:PAIR_COUNT]):
        border_a = border_colors[2 * i]
        border_b = border_colors[2 * i + 1]
        source_center = non_border_colors[i]
        dest_is_a = bool(random.getrandbits(1))
        _, meta = _place_pair_output(
            output_grid,
            shape_type,
            slot_a,
            slot_b,
            border_a,
            border_b,
            source_center,
            dest_is_a,
        )
        pair_meta.append(meta)

    distractor = None
    if distractor_count:
        border = border_colors[-1]
        center = non_border_colors[PAIR_COUNT]
        patch = _make_sprite_patch(shape_type, border, center)
        slots = list(layout["distractors"])
        random.shuffle(slots)
        pos = next((slot for slot in slots if not _overlaps(output_grid, patch, slot)), None)
        if pos is None:
            return None
        distractor = (border, center, pos)

    input_grid = _build_input_from_output(output_grid, shape_type, pair_meta, distractor)

    grey_comps = _grey_components(input_grid, GREY)
    if len(grey_comps) != PAIR_COUNT:
        return None
    if _count_sprite_like_components(input_grid) != 4 + distractor_count:
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
