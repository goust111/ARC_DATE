# Combo variant of d35bdbdc (index 28)
# combo_spec: D7/3+D8/1+D10/2
# Name: 多条路径+路径着色+链式传递

"""
Design analysis:
  concretization:
    keep:
      - 10x10 canvas, bg=0, grey=5, bordered-sprite vocabulary from the source task
      - sprite semantics still rely on border-color / center-color equality, never on numeric color order
      - non-connected foreground junk still disappears in the output
    change:
      - the single global trail is replaced by exactly two disconnected trail components
      - each component links a local chain of three sprites instead of a single destination pair
      - center colors propagate one hop along the chain: A -> B -> C
      - the grey trail is not preserved as grey; each path segment is recolored to the source color that moves across it
    pixel_level_example:
      - suppose one component has physical trail order P0, P1, P2
      - if the visible color links say P0.border == P1.center and P1.border == P2.center in reverse,
        then the real chain is P2 -> P1 -> P0
      - output centers become [old(P1.center), old(P2.center), old(P2.center)] in physical order
      - the segment between P2 and P1 takes old(P2.center), while the segment between P1 and P0 takes old(P1.center)
      - this differs from the source task twice: more than two sprites survive, and the trail itself now shows directed transfer

  complexity_calibration:
    decisions:
      - use exactly two components, each with exactly three connected sprites
      - use plus-shaped bordered sprites only; this is a deliberate simplification to fit two clean 3-sprite chains
        inside 10x10 without accidental 4-touching that full 3x3 boxes would cause
      - keep each component short and isolated, but force two different segment colors on every output path
    why_not_simpler:
      - two components avoid collapse back to the original "one special path" reading
      - three sprites per component make the chain aspect visible; two sprites would reduce to a single-hop transfer
      - mandatory path recoloring makes the D8 contribution visually explicit instead of latent

  implicit_assumptions_audit:
    original_solver_assumptions_that_break:
      - one call to `_find_trail_endpoints(...)` is insufficient because grey cells are no longer one component
      - keeping only two endpoint sprites is wrong because middle chain sprites must survive
      - global border lookup over arbitrary sprites is no longer the main mechanism; adjacency along the path matters
      - leaving trails grey is wrong because recolored path cells now encode transfer information
    fused_solver_replacements:
      - split grey into 8-connected components and solve each component independently
      - trace each component as an ordered path, gather all sprites with valid ports on that path,
        and order them by port position
      - choose the chain direction by visible equality links `next.center == prev.border`
      - recolor path intervals after cleanup so recolored trail cells are not accidentally dropped
    new_assumptions_and_generator_guarantees:
      - each connected sprite has exactly one valid port on its local path
      - each path component is a simple one-stroke trace with three attached sprites
      - only one of the two possible path directions yields two consecutive visible color matches
      - every recolored interval uses a non-grey color distinct from the neighboring interval

  placement_legality:
    generator_strategy:
      - output-first construction on a fixed clean geometric template, then optional horizontal / vertical flips
      - sprites are placed before paths; path cells are hardcoded to avoid all sprite pixels
      - optional distractors are isolated single pixels added only in the input, never touching sprites or paths
    legality_checks:
      - transformed layouts preserve in-bounds ports and pairwise sprite separation
      - path components remain 8-disconnected from each other after every flip
      - oracle validation runs `main(input)` and only accepts exact matches to the constructed output

  visual_observability:
    cues_humans_can_use:
      - there are two separate path systems, not one
      - each surviving group has three sprites, so the middle sprite clearly participates as both receiver and sender
      - each output path shows two color blocks, which line up with the one-hop transfers along that chain
      - chain direction is inferable from border/center equalities, not from hidden color ranking

  cleanliness:
    layout:
      - the two chains live in upper and lower lanes with a blank band between them
      - path recoloring uses only the colors already visible as sprite centers / border keys
      - extra input-only noise is limited to isolated pixels so the main structure stays readable
"""

from __future__ import annotations

import random

import numpy as np

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
    Clipped variants near the boundary remain valid.
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


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, path_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return every path cell that is a valid port for this sprite."""
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


def _trace_component_path(path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Recover one simple ordered path that visits all cells of this grey component."""
    assert path_cells, "expected a non-empty grey component"
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in path_cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in path_cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    start = deg1[0] if deg1 else next(iter(path_cells))
    fixed_end = deg1[1] if len(deg1) >= 2 else None
    total = len(path_cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == total:
            if fixed_end is None:
                return [cur]
            return [cur] if cur == fixed_end else None
        for nb in adj[cur]:
            if nb in visited:
                continue
            result = dfs(nb, visited | {nb})
            if result is not None:
                return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path is not None, "could not trace grey component as a one-stroke path"
    return path


def _sequence_score(seq: list[dict]) -> tuple[int, int]:
    """Higher is better: more visible color-link matches, then a cleaner unmatched start."""
    matches = sum(int(seq[i + 1]["center"]) == int(seq[i]["border"]) for i in range(len(seq) - 1))
    later_borders = {int(sp["border"]) for sp in seq[1:]}
    start_clean = int(int(seq[0]["center"]) not in later_borders)
    return matches, start_clean


def _resolve_component_chain(g: np.ndarray, comp: set[tuple[int, int]], sprites: list[dict]) -> dict:
    """Resolve one component into an ordered chain plus the path cells that connect it."""
    path = _trace_component_path(comp)
    idx_of = {cell: i for i, cell in enumerate(path)}

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
                "port_idx": min(idx_of[p] for p in ports),
            }
        )

    assert len(connected) >= 2, "expected at least two sprites attached to each grey component"
    physical = sorted(
        connected,
        key=lambda sp: (int(sp["port_idx"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )
    reversed_physical = list(reversed(physical))
    chain = physical if _sequence_score(physical) >= _sequence_score(reversed_physical) else reversed_physical

    return {"path": path, "chain": chain}


def _cleanup_foreground(
    out: np.ndarray, sprites: list[dict], keep_ids: set[int], bg: int, grey: int
) -> None:
    """Delete every non-kept sprite and any other leftover non-grey foreground components."""
    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for sp in sprites:
        if id(sp) not in keep_ids:
            continue
        for r, c in sp["pts"]:
            if out[r, c] in (bg, grey) or keep[r, c]:
                continue
            keep[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                continue
            if keep[nr, nc] or out[nr, nc] in (bg, grey):
                continue
            keep[nr, nc] = True
            q.append((nr, nc))

    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg


def _recolor_component_path(out: np.ndarray, path: list[tuple[int, int]], chain: list[dict]) -> None:
    """Paint each path interval with the original center color that travels across that link."""
    idx_of = {cell: i for i, cell in enumerate(path)}
    for i in range(len(chain) - 1):
        source = chain[i]
        target = chain[i + 1]
        source_idx = min(idx_of[p] for p in source["ports"])
        target_idx = min(idx_of[p] for p in target["ports"])
        lo, hi = sorted((source_idx, target_idx))
        for k in range(lo, hi + 1):
            r, c = path[k]
            out[r, c] = int(source["center"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate center colors along multiple independent path-linked sprite chains and recolor the paths accordingly.

    Input structure:
    - The grid contains several bordered sprites plus multiple disconnected grey (5) path components.
    - Each grey component is a simple local trail whose visited port cells attach to a small ordered sprite chain,
      while unrelated sprites or isolated non-sprite pixels may also appear elsewhere.

    Rule:
    - Solve each grey component independently: trace its path, collect the sprites attached to that path, and orient
      the chain using the visible equalities `next.center == previous.border`.
    - Keep every sprite that belongs to any chain and delete all other foreground objects.
    - Within each oriented chain, the first sprite keeps its center and every later sprite receives the original
      center color of the previous sprite; after that, repaint each path interval with the color that was sent across it.

    Key insight:
    - The grey structure now carries two jobs at once: disconnected components define separate groups, and path order
      inside each component reveals a directed multi-step transfer whose intermediate steps become visible on the path itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite"

    grey_comps = _grey_components(g, grey)
    assert grey_comps, "expected at least one grey path component"

    component_infos = [_resolve_component_chain(g, comp, sprites) for comp in grey_comps]
    keep_ids = {int(sp["sprite_id"]) for info in component_infos for sp in info["chain"]}

    out = g.copy()
    _cleanup_foreground(out, sprites, keep_ids, bg, grey)

    for info in component_infos:
        chain = info["chain"]
        original_centers = [int(sp["center"]) for sp in chain]
        for i in range(1, len(chain)):
            r, c = chain[i]["center_pos"]
            out[r, c] = original_centers[i - 1]

    for info in component_infos:
        _recolor_component_path(out, info["path"], info["chain"])

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
GEN_RETRIES = 400
COMPONENT_COUNT = 2
CHAIN_LENGTH = 3
EXTRA_PIXEL_PROB = 0.7

BASE_LAYOUT = [
    {
        "slots": [
            {"center": (2, 1), "dir": (0, 1)},
            {"center": (1, 4), "dir": (1, 0)},
            {"center": (2, 8), "dir": (0, -1)},
        ],
        "path": [(2, 3), (3, 4), (3, 5), (2, 6)],
    },
    {
        "slots": [
            {"center": (7, 1), "dir": (0, 1)},
            {"center": (8, 4), "dir": (-1, 0)},
            {"center": (7, 8), "dir": (0, -1)},
        ],
        "path": [(7, 3), (6, 4), (6, 5), (7, 6)],
    },
]


def _make_sprite_patch(border_color: int, center_color: int) -> np.ndarray:
    """Return the plus-shaped bordered sprite patch used by this combo generator."""
    patch = np.zeros((3, 3), dtype=int)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    paste(grid, patch, center[0] - 1, center[1] - 1, transparent=BG, clip=True)


def _transform_cell(cell: tuple[int, int], flip_h: bool, flip_v: bool) -> tuple[int, int]:
    r, c = cell
    if flip_v:
        r = GRID_H - 1 - r
    if flip_h:
        c = GRID_W - 1 - c
    return int(r), int(c)


def _transform_dir(direction: tuple[int, int], flip_h: bool, flip_v: bool) -> tuple[int, int]:
    dr, dc = direction
    if flip_v:
        dr = -dr
    if flip_h:
        dc = -dc
    return int(dr), int(dc)


def _transform_layout(flip_h: bool, flip_v: bool) -> list[dict]:
    layout: list[dict] = []
    for comp in BASE_LAYOUT:
        slots = []
        for slot in comp["slots"]:
            slots.append(
                {
                    "center": _transform_cell(slot["center"], flip_h, flip_v),
                    "dir": _transform_dir(slot["dir"], flip_h, flip_v),
                }
            )
        path = [_transform_cell(cell, flip_h, flip_v) for cell in comp["path"]]
        layout.append({"slots": slots, "path": path})
    return layout


def _port_of_slot(slot: dict) -> tuple[int, int]:
    cr, cc = slot["center"]
    dr, dc = slot["dir"]
    return cr + 2 * dr, cc + 2 * dc


def _add_input_noise(input_grid: np.ndarray, palette: list[int]) -> None:
    """Add one isolated non-sprite pixel to the input to keep deletion behavior visible."""
    if random.random() >= EXTRA_PIXEL_PROB:
        return
    candidates: list[tuple[int, int]] = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if input_grid[r, c] != BG:
                continue
            if any(
                0 <= nr < GRID_H and 0 <= nc < GRID_W and input_grid[nr, nc] != BG
                for nr, nc in neighbors4(r, c)
            ):
                continue
            candidates.append((r, c))
    if not candidates:
        return
    r, c = random.choice(candidates)
    input_grid[r, c] = random.choice(palette)


def _build_component(
    output_grid: np.ndarray,
    component_layout: dict,
    chain_borders: list[int],
    start_color: int,
    reverse_chain: bool,
) -> dict:
    """Place one 3-sprite component and paint its final colored path."""
    physical_slots = list(component_layout["slots"])
    chain_slots = list(reversed(physical_slots)) if reverse_chain else list(physical_slots)
    input_centers = [start_color, chain_borders[0], chain_borders[1]]
    output_centers = [start_color, start_color, chain_borders[0]]

    slot_meta: list[dict] = []
    for i in range(CHAIN_LENGTH):
        slot = chain_slots[i]
        meta = {
            "center": slot["center"],
            "dir": slot["dir"],
            "border": int(chain_borders[i]),
            "input_center": int(input_centers[i]),
            "output_center": int(output_centers[i]),
        }
        slot_meta.append(meta)
        patch = _make_sprite_patch(meta["border"], meta["output_center"])
        _place_sprite(output_grid, patch, meta["center"])

    path = component_layout["path"]
    path_idx = {cell: i for i, cell in enumerate(path)}
    for i in range(CHAIN_LENGTH - 1):
        source = slot_meta[i]
        target = slot_meta[i + 1]
        source_port = (source["center"][0] + 2 * source["dir"][0], source["center"][1] + 2 * source["dir"][1])
        target_port = (target["center"][0] + 2 * target["dir"][0], target["center"][1] + 2 * target["dir"][1])
        lo, hi = sorted((path_idx[source_port], path_idx[target_port]))
        for k in range(lo, hi + 1):
            r, c = path[k]
            output_grid[r, c] = int(source["input_center"])

    return {"sprites": slot_meta, "path": path}


def _gen_attempt() -> np.ndarray | None:
    flip_h = bool(random.getrandbits(1))
    flip_v = bool(random.getrandbits(1))
    layout = _transform_layout(flip_h, flip_v)

    palette = [c for c in range(1, 10) if c != GREY]
    random.shuffle(palette)
    all_borders = palette[: COMPONENT_COUNT * CHAIN_LENGTH]
    start_colors = palette[COMPONENT_COUNT * CHAIN_LENGTH : COMPONENT_COUNT * CHAIN_LENGTH + COMPONENT_COUNT]
    if len(start_colors) != COMPONENT_COUNT:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    component_meta: list[dict] = []
    for i, comp in enumerate(layout):
        chain_borders = all_borders[i * CHAIN_LENGTH : (i + 1) * CHAIN_LENGTH]
        reverse_chain = bool(random.getrandbits(1))
        meta = _build_component(output_grid, comp, chain_borders, start_colors[i], reverse_chain)
        component_meta.append(meta)

    input_grid = output_grid.copy()
    for meta in component_meta:
        for r, c in meta["path"]:
            input_grid[r, c] = GREY
        for sp in meta["sprites"]:
            r, c = sp["center"]
            input_grid[r, c] = int(sp["input_center"])

    _add_input_noise(input_grid, palette)

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
