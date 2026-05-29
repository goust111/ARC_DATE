# Combo variant of d35bdbdc (index 8)
# combo_spec: D7/3+D10/2
# Name: 多条独立路径+链式传递

"""
Design analysis:
  fused_goal:
    keep:
      - 10x10 canvas, bg=0, grey=5
      - bordered-sprite vocabulary with one unique center cell per sprite
      - color-link idea: visible equality between one sprite center and another sprite border
      - deletion of sprites not attached to grey structure
    change:
      - replace the single global grey trail with multiple disconnected grey components
      - each grey component now links an ordered chain of three sprites, not a single pair
      - color propagation is one-step local along each chain, so middle sprites are both receivers and senders

  concrete_behavior:
    input:
      - exactly two grey components are present in every generated sample
      - each component is a straight thin segment whose cells include three valid sprite ports
      - each component touches three sprites in trail order A,B,C
      - input centers along that order are [seed, A.border, B.border], where seed is not any border color
      - optional distractor sprites may appear away from both grey components
    output:
      - keep all sprites attached to either grey component and keep both grey components
      - delete unconnected sprites
      - inside each component, perform one-step propagation: A stays, B gets A.center, C gets B.center

  behavior_trace_example:
    original:
      - the solver expects one grey component and extracts only two endpoint sprites
      - each endpoint looks up a separate source somewhere else in the grid
      - every non-endpoint sprite is deleted
    fused_variant:
      - component 1 might have trail order [X,Y,Z] with borders [2,8,4] and input centers [6,2,8]
      - its output centers become [6,6,2]
      - component 2 independently does the same on another disconnected grey segment
      - both components survive together, so the solver must segment by grey component first and then orient each chain
    difference:
      - participation is no longer "the two global endpoints"; it is "all sprites attached to each local grey chain"
      - transfer is no longer a one-hop lookup into detached sources; it is chained local propagation on each component

  complexity_calibration:
    - both grey changes are forced in every sample: there are always 2 components and each has length 3
    - chain length 3 is the smallest size that visibly exhibits a middle relay sprite without overcrowding 10x10
    - connected border colors stay distinct, and each chain start color avoids all border colors, making orientation unique
    - geometry varies across horizontal and vertical templates, and color direction can align with or oppose the obvious reading direction
    - generation uses plus-shaped sprites only, because six connected sprites plus two grey components would make box sprites
      too cramped and visually noisy on a 10x10 canvas

  hidden_assumptions_review:
    original_assumptions_broken:
      - treating all grey cells as one component is invalid
      - choosing only two kept sprites is invalid
      - updating exactly two centers is invalid
      - global border lookup is no longer the right mechanism
    solver_replacements:
      - first split grey cells into 8-connected components
      - trace each component independently and collect every sprite with a valid port on that component
      - order those sprites along the traced path in both directions and pick the direction with the stronger visible color-link chain
      - propagate original centers one step forward inside each ordered component
    new_solver_assumptions:
      - each grey component is a simple path with two endpoints
      - each generated component touches exactly three sprites through valid ports
      - only one trail direction attains the full adjacent-link score
    generator_guarantees:
      - templates draw straight non-touching grey segments with three ports each
      - connected sprite centers are assigned as [seed, prev.border, prev.border] style chain inputs, giving one unique direction
      - distractors are accepted only if they do not form any valid port to any grey component

  placement_legality:
    - both grey components live in disjoint rows or disjoint columns, so they cannot merge under 8-connectivity
    - connected sprite slots are template-controlled and checked with non-overlap / non-touch rules before placement
    - grey cells are painted only on reserved straight lanes between ports
    - distractor placement is brute-force checked against overlap, clipping, and accidental path-port creation

  visual_observability:
    - disconnected grey groups are immediately visible, so the D7 change is explicit
    - each grey group visibly keeps three sprites, so the D10 chain change is explicit
    - only the first sprite in a chain keeps its center; the next two shift by one step, exposing directionality
    - direction comes from equality cues between input centers and neighboring borders, not from hidden color ordering

  cleanliness:
    - the two chains occupy separate lanes, avoiding tangled grey geometry
    - using one sprite family per example keeps the picture legible
    - the optional distractor is only kept when it can be placed without visually competing with the chain lanes
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
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
    Clipped edge cases remain valid.
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
        border_color = next(c for c in uniq if c != center_color)
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


def _grey_components(g: np.ndarray, grey: int) -> list[set[tuple[int, int]]]:
    comps = components_mask(g == grey, connectivity=8)
    return [{(int(r), int(c)) for r, c in comp} for comp in comps]


def _trace_component(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Trace one thin grey component from one endpoint to the other."""
    assert cells, "expected non-empty grey component"
    adj = {cell: [nb for nb in neighbors8(*cell) if nb in cells] for cell in cells}
    endpoints = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "each grey component should be a simple path"
    start, goal = endpoints
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if cur == goal and len(visited) == total:
            return [cur]
        for nb in adj[cur]:
            if nb in visited:
                continue
            result = dfs(nb, visited | {nb})
            if result is not None:
                return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path is not None, "could not trace grey component"
    return path


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, path_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return every grey path cell that forms a valid cardinal port for this sprite."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in path_cells:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _sequence_score(seq: list[dict]) -> tuple[int, int]:
    """Prefer more adjacent border/center links, then a cleaner unmatched start."""
    matches = sum(int(seq[i]["border"]) == int(seq[i + 1]["center"]) for i in range(len(seq) - 1))
    start_unmatched = int(all(int(seq[0]["center"]) != int(sp["border"]) for sp in seq[1:]))
    return matches, start_unmatched


def _ordered_chain_for_component(g: np.ndarray, sprites: list[dict], comp: set[tuple[int, int]]) -> list[dict]:
    trail = _trace_component(comp)
    trail_idx = {cell: i for i, cell in enumerate(trail)}

    connected: list[dict] = []
    for sp in sprites:
        ports = _ports_for_sprite(sp, g, comp)
        if not ports:
            continue
        connected.append(
            {
                **sp,
                "sprite_id": id(sp),
                "ports": ports,
                "port_lo": min(trail_idx[p] for p in ports),
                "port_hi": max(trail_idx[p] for p in ports),
            }
        )

    assert len(connected) >= 3, "expected a chain of at least three sprites on each grey component"
    forward = sorted(
        connected,
        key=lambda sp: (int(sp["port_lo"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )
    backward = sorted(
        connected,
        key=lambda sp: (-int(sp["port_hi"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )
    return forward if _sequence_score(forward) >= _sequence_score(backward) else backward


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate center colors along each disconnected grey-linked sprite chain, keeping only chain-connected sprites.

    Input structure:
    - The grid contains several bordered sprites plus grey (5) path cells.
    - Grey cells form multiple disconnected thin path components, and each component touches an ordered chain of
      bordered sprites through valid cardinal ports.

    Rule:
    - Process each grey component independently, keep every sprite that has a valid port on that component, and
      delete sprites that are not attached to any grey component.
    - For each component, order the attached sprites along the path and choose the direction whose adjacent
      center-to-border equalities reveal the chain orientation.
    - The first sprite in that order keeps its original center, and every later sprite receives the original
      center color of the previous sprite in the same component.

    Key insight:
    - The grey geometry now carries two kinds of information at once: disconnected components define independent
      groups, and the visible center/border links inside each group determine the direction of one-step propagation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite"

    grey_comps = _grey_components(g, grey)
    assert len(grey_comps) >= 2, "expected multiple grey components"

    chains = [_ordered_chain_for_component(g, sprites, comp) for comp in grey_comps]
    keep_ids = {int(sp["sprite_id"]) for chain in chains for sp in chain}

    out = g.copy()
    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for chain in chains:
        for sp in chain:
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

    for chain in chains:
        original_centers = [int(sp["center"]) for sp in chain]
        for i in range(1, len(chain)):
            r, c = chain[i]["center_pos"]
            out[r, c] = original_centers[i - 1]

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
GEN_RETRIES = 300
DISTRACTOR_COUNTS = [0, 1]

TEMPLATES = [
    {
        "name": "h4",
        "components": [
            [
                {"center": (3, 1), "port": (3, 3)},
                {"center": (1, 4), "port": (3, 4)},
                {"center": (3, 8), "port": (3, 6)},
            ],
            [
                {"center": (7, 1), "port": (7, 3)},
                {"center": (9, 4), "port": (7, 4)},
                {"center": (7, 8), "port": (7, 6)},
            ],
        ],
    },
    {
        "name": "h5",
        "components": [
            [
                {"center": (3, 1), "port": (3, 3)},
                {"center": (1, 5), "port": (3, 5)},
                {"center": (3, 8), "port": (3, 6)},
            ],
            [
                {"center": (7, 1), "port": (7, 3)},
                {"center": (9, 5), "port": (7, 5)},
                {"center": (7, 8), "port": (7, 6)},
            ],
        ],
    },
    {
        "name": "v4",
        "components": [
            [
                {"center": (1, 2), "port": (3, 2)},
                {"center": (4, 0), "port": (4, 2)},
                {"center": (8, 2), "port": (6, 2)},
            ],
            [
                {"center": (1, 7), "port": (3, 7)},
                {"center": (4, 9), "port": (4, 7)},
                {"center": (8, 7), "port": (6, 7)},
            ],
        ],
    },
    {
        "name": "v5",
        "components": [
            [
                {"center": (1, 2), "port": (3, 2)},
                {"center": (5, 0), "port": (5, 2)},
                {"center": (8, 2), "port": (6, 2)},
            ],
            [
                {"center": (1, 7), "port": (3, 7)},
                {"center": (5, 9), "port": (5, 7)},
                {"center": (8, 7), "port": (6, 7)},
            ],
        ],
    },
]


def _make_sprite_patch(border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 plus-shaped bordered sprite patch."""
    patch = np.zeros((3, 3), dtype=int)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    paste(grid, patch, center[0] - 1, center[1] - 1, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    r0 = center[0] - 1
    c0 = center[1] - 1
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> bool:
    """Reject overlap and 4-touching with existing non-bg, non-grey cells."""
    r0 = center[0] - 1
    c0 = center[1] - 1
    h, w = grid.shape
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
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _straight_path(ports: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(ports) >= 2
    rows = {r for r, _ in ports}
    cols = {c for _, c in ports}
    if len(rows) == 1:
        row = next(iter(rows))
        c0 = min(c for _, c in ports)
        c1 = max(c for _, c in ports)
        return [(row, c) for c in range(c0, c1 + 1)]
    if len(cols) == 1:
        col = next(iter(cols))
        r0 = min(r for r, _ in ports)
        r1 = max(r for r, _ in ports)
        return [(r, col) for r in range(r0, r1 + 1)]
    raise ValueError("template ports must be collinear")


def _chain_input_centers(chain_borders: list[int], seed: int) -> list[int]:
    return [seed] + chain_borders[:-1]


def _chain_output_centers(input_centers: list[int]) -> list[int]:
    return [input_centers[0]] + input_centers[:-1]


def _place_connected_component(
    output_grid: np.ndarray,
    slots: list[dict],
    chain_borders: list[int],
    chain_seed: int,
) -> list[dict]:
    chain_slots = list(slots if random.getrandbits(1) else list(reversed(slots)))
    input_centers = _chain_input_centers(chain_borders, chain_seed)
    output_centers = _chain_output_centers(input_centers)
    placed: list[dict] = []
    for i, slot in enumerate(chain_slots):
        patch = _make_sprite_patch(chain_borders[i], output_centers[i])
        if _overlaps(output_grid, patch, slot["center"]):
            raise AssertionError("template placement unexpectedly overlaps")
        _place_sprite(output_grid, patch, slot["center"])
        placed.append(
            {
                "center_pos": slot["center"],
                "border": int(chain_borders[i]),
                "input_center": int(input_centers[i]),
                "output_center": int(output_centers[i]),
                "port": slot["port"],
            }
        )
    for r, c in _straight_path([slot["port"] for slot in slots]):
        assert output_grid[r, c] == BG, "grey path should not overlap sprites"
        output_grid[r, c] = GREY
    return placed


def _candidate_distractor_centers() -> list[tuple[int, int]]:
    cands = [(r, c) for r in range(GRID_H) for c in range(GRID_W)]
    random.shuffle(cands)
    return cands


def _place_distractor(
    input_grid: np.ndarray,
    border_color: int,
    center_color: int,
    path_cells: set[tuple[int, int]],
) -> bool:
    patch = _make_sprite_patch(border_color, center_color)
    for center in _candidate_distractor_centers():
        if _overlaps(input_grid, patch, center):
            continue
        _place_sprite(input_grid, patch, center)
        sprites = _detect_bordered_sprites(input_grid, BG, GREY)
        placed_sprite = next(
            (
                sp
                for sp in sprites
                if sp["center_pos"] == center
                and int(sp["border"]) == int(border_color)
                and int(sp["center"]) == int(center_color)
            ),
            None,
        )
        if placed_sprite is not None and not _ports_for_sprite(placed_sprite, input_grid, path_cells):
            return True
        _erase_sprite(input_grid, patch, center)
    return False


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(TEMPLATES)
    distractor_count = random.choice(DISTRACTOR_COUNTS)

    palette = [c for c in range(1, 10) if c != GREY]
    total_borders = 6 + distractor_count
    border_colors = random.sample(palette, total_borders)
    connected_borders = border_colors[:6]
    distractor_borders = border_colors[6:]
    spare_colors = [c for c in palette if c not in border_colors]
    if not spare_colors:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    placed_components: list[list[dict]] = []
    for i, slots in enumerate(template["components"]):
        chain_borders = connected_borders[3 * i : 3 * (i + 1)]
        seed = random.choice(spare_colors)
        placed = _place_connected_component(output_grid, slots, chain_borders, seed)
        placed_components.append(placed)

    input_grid = output_grid.copy()
    for placed in placed_components:
        for sp in placed:
            r, c = sp["center_pos"]
            input_grid[r, c] = sp["input_center"]

    path_cells = {(int(r), int(c)) for r, c in zip(*np.where(output_grid == GREY), strict=False)}
    for border_color in distractor_borders:
        center_choices = [c for c in palette if c != border_color]
        center_color = random.choice(center_choices)
        if not _place_distractor(input_grid, border_color, center_color, path_cells):
            return None

    grey_comps = _grey_components(input_grid, GREY)
    if len(grey_comps) != 2:
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(out, output_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
