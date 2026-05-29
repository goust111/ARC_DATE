# Variant of d35bdbdc
# Dimension: D10 — 精灵数量与配对基数 (multi_step)
# Change: 多个源和多个目标精灵，多对多匹配（每个目标找到一个匹配的源）... → 链式传递（A->B->C）

"""
Design analysis for the D10 chain variant
=========================================

variant_spec:
  idea:
    keep the original visual vocabulary:
      - 10x10 grid
      - grey one-stroke trail
      - 3x3 bordered sprites with one center pixel
      - non-connected sprites disappear in the output
    change the role structure:
      - original: only the 2 endpoint sprites survive, and each endpoint looks up some separate source
      - variant: every sprite attached to the grey trail survives, and their center colors propagate one step
        forward along the trail order: A->B->C->D

  concrete_pixels:
    connected_sprites:
      count: 3 or 4
      placement: one per edge side, with the grey trail touching exactly one inward-facing port cell
      order cue: the trail passes through the connected sprites' port cells in a unique sequence
    distractors:
      count: 1 or 2
      placement: clipped corner sprites, never touching the grey trail
    path:
      shape: contiguous subpath of a thin snake in the 4x4 central area (rows 3..6, cols 3..6)
      reason: this keeps the trail visible while avoiding accidental extra ports on the edge sprites

  behavior_trace:
    original_local_step:
      - the solver identifies exactly 2 destination sprites at the two grey endpoints
      - if dest.center == source.border, the source.center is copied into that destination
      - every non-destination sprite is deleted
    variant_local_step:
      - the solver identifies every sprite that has a valid port on the grey trail
      - these sprites are ordered along the trail
      - the valid direction is the one whose adjacent color links satisfy next.center == prev.border
      - sprite i keeps its border but receives the original center of sprite i-1
      - sprites not attached to the trail are deleted
    concrete_example:
      - borders along trail order: [2, 8, 4, 7]
      - input centers: [6, 2, 8, 4]
      - output centers: [6, 6, 2, 8]
      - difference from the original task:
        the middle sprites now matter; they are both targets and sources in the same example

  complexity_calibration:
    why_not_two_sprites:
      - a 2-sprite case collapses back to the original single-hop intuition
      - forcing 3 or 4 connected sprites makes the chain visible in 2-3 examples
    why_keep_color_links:
      - the center->border lookup motif is preserved, so the change stays focused on D10
      - the new difficulty comes from ordered multi-step participation, not from unrelated mechanics

  hidden_assumptions_audit:
    old_assumptions_that_break:
      - "_find_trail_endpoints(...)" only returns two special sprites; invalid because middle trail sprites must survive
      - "dest_ids" as the keep set is too small; now the keep set is all trail-connected sprites
      - "for d in dest" updates only two centers; invalid because a whole ordered chain must update
      - plain border lookup over all sprites is unsafe if distractors share relevant colors; the generator now keeps
        connected borders unique and the solver orients the chain by adjacent trail order
    new_assumptions_and_generator_guarantees:
      - every connected sprite has exactly one inward port on the grey path
      - connected sprites appear in one simple color-link chain, with exactly one valid direction
      - the first chain sprite has a center color that matches no connected border
      - distractors sit in corners so the central snake cannot accidentally connect to them

  placement_legality:
    connected_layout:
      - connected sprites are edge-anchored; the trail lives only in rows/cols 3..6
      - this blocks accidental side-ports because non-inward candidate ports lie on rows/cols 0..2 or 7..9
    distractor_layout:
      - distractors use clipped corner placements, which remain valid bordered-sprite components
      - their possible ports lie outside the central snake, so they are guaranteed to be deleted

  observability:
    human_cue:
      - the grey line visibly passes by 3 or 4 survivors, not just 2
      - the centers visibly shift one step along that line
      - the chain direction is recoverable because only one trail direction makes the center colors line up
    color_discipline:
      - no sorting or mapping depends on numeric color id order
      - colors are only compared for equality in visible border/center relationships

  neatness:
    path_area: compact 4x4 central snake keeps the output readable
    sprite_area: edge sprites and corner distractors stay separated, so components remain visually distinct
"""

import random
from collections import deque
from itertools import permutations

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _trace_trail(g: np.ndarray, grey: int, sprites: list[dict]) -> list[tuple[int, int]]:
    """Trace the full one-stroke grey trail under 8-connectivity."""
    h, w = g.shape
    cells = {(int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False)}
    assert cells, "expected a grey trail"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    goal_cells = set(deg1[1:]) if len(deg1) >= 2 else set(port_cells)
    if not goal_cells:
        goal_cells = {start}
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur in goal_cells else None
        for nb in adj[cur]:
            if nb in visited:
                continue
            result = dfs(nb, visited | {nb})
            if result is not None:
                return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    Clipped edge/corner versions remain valid.
    """
    mask = (g != bg) & (g != grey)
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
        if border_color == bg or border_color == grey or center_color == grey:
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


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, path_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return every grey path cell that forms a valid port for this sprite."""
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in path_cells:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _sequence_score(seq: list[dict]) -> tuple[int, int]:
    """Higher is better: more valid color links, then a cleaner unmatched start."""
    matches = sum(int(seq[i]["border"]) == int(seq[i + 1]["center"]) for i in range(len(seq) - 1))
    start_unmatched = int(all(int(seq[0]["center"]) != int(sp["border"]) for sp in seq[1:]))
    return matches, start_unmatched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate center colors one step along the ordered chain of grey-connected bordered sprites.

    Input structure:
    - Background plus grey (5) one-stroke trail, and several small bordered sprites with one distinct center pixel.
    - Multiple sprites attach to the same grey trail by valid ports; unrelated sprites may also appear elsewhere.

    Rule:
    - Keep every sprite whose border touches the grey trail through a valid port, and delete the rest.
    - Order the kept sprites along the grey trail; choose the trail direction whose adjacent sprites satisfy the
      visible link next.center == previous.border.
    - The first sprite keeps its center; each later sprite receives the original center color of the previous sprite.

    Key insight: The grey trail no longer selects just two endpoints. It defines a whole ordered chain, and the
    center-to-border color links reveal the direction of one-step color propagation along that chain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    trail = _trace_trail(g, grey, sprites)
    trail_set = set(trail)
    trail_idx = {cell: i for i, cell in enumerate(trail)}

    connected: list[dict] = []
    for sp in sprites:
        ports = _ports_for_sprite(sp, g, trail_set)
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

    assert len(connected) >= 2, "expected at least two trail-connected sprites"

    forward = sorted(connected, key=lambda sp: (int(sp["port_lo"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])))
    backward = sorted(
        connected, key=lambda sp: (-int(sp["port_hi"]), int(sp["center_pos"][0]), int(sp["center_pos"][1]))
    )
    chain = forward if _sequence_score(forward) >= _sequence_score(backward) else backward
    keep_ids = {int(sp["sprite_id"]) for sp in chain}

    out = g.copy()

    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for sp in chain:
        for r, c in sp["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    original_centers = [int(sp["center"]) for sp in chain]
    for i in range(1, len(chain)):
        r, c = chain[i]["center_pos"]
        out[r, c] = original_centers[i - 1]

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
CHAIN_LENGTHS = [3, 4]
GEN_RETRIES = 500
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MAX_THICKNESS = 2
SAFE_ROWS = range(2, 8)
SAFE_COLS = range(2, 8)
SIDES = ("top", "right", "bottom", "left")


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        p = np.full((3, 3), border_color, dtype=int)
    else:
        p = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
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
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _side_candidates(side: str) -> list[dict]:
    cands: list[dict] = []
    if side == "top":
        for c in range(3, 7):
            cands.append({"side": side, "r0": 0, "c0": c - 1, "port": (3, c), "center_pos": (1, c)})
    elif side == "bottom":
        for c in range(3, 7):
            cands.append({"side": side, "r0": 7, "c0": c - 1, "port": (6, c), "center_pos": (8, c)})
    elif side == "left":
        for r in range(3, 7):
            cands.append({"side": side, "r0": r - 1, "c0": 0, "port": (r, 3), "center_pos": (r, 1)})
    else:
        for r in range(3, 7):
            cands.append({"side": side, "r0": r - 1, "c0": 7, "port": (r, 6), "center_pos": (r, 8)})
    return cands


SIDE_CANDIDATES = {side: _side_candidates(side) for side in SIDES}
CORNER_POSITIONS = [(-1, -1), (-1, 8), (8, -1), (8, 8)]


def _choose_connected_layout(shape_type: str, chain_len: int) -> list[dict] | None:
    dummy_patch = _make_sprite_patch(shape_type, 1, 2)
    for _ in range(300):
        grid = np.zeros((GRID_H, GRID_W), dtype=int)
        picks: list[dict] = []
        sides = random.sample(SIDES, chain_len)
        ok = True
        for side in sides:
            cands = list(SIDE_CANDIDATES[side])
            random.shuffle(cands)
            placed = False
            for cand in cands:
                if cand["port"] in {pick["port"] for pick in picks}:
                    continue
                if _overlaps(grid, dummy_patch, cand["r0"], cand["c0"]):
                    continue
                _place_sprite(grid, dummy_patch, cand["r0"], cand["c0"])
                picks.append(cand)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        return picks
    return None


def _chain_centers(border_colors: list[int]) -> tuple[list[int], list[int]]:
    palette = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    assert palette, "expected at least one spare color for chain start"
    start_center = random.choice(palette)
    input_centers = [start_center] + border_colors[:-1]
    output_centers = [start_center] + input_centers[:-1]
    return input_centers, output_centers


def _place_corner_distractors(
    input_grid: np.ndarray, shape_type: str, distractor_specs: list[tuple[int, int]], path_cells: set[tuple[int, int]]
) -> bool:
    slots = list(CORNER_POSITIONS)
    random.shuffle(slots)
    for border_color, center_color in distractor_specs:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        placed = False
        for r0, c0 in slots:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            sprites = _detect_bordered_sprites(input_grid, BG, GREY)
            new_sprite = None
            for sp in sprites:
                if int(sp["border"]) == int(border_color) and int(sp["center"]) == int(center_color):
                    new_sprite = sp
                    break
            if new_sprite is not None and not _ports_for_sprite(new_sprite, input_grid, path_cells):
                placed = True
                slots.remove((r0, c0))
                break
            _erase_sprite(input_grid, patch, r0, c0)
        if not placed:
            return False
    return True


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Reject additions that would make the 8-neighbor trail branch/thicken beyond a simple chain."""
    nbs = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in occupied]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(
            1
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (nr + dr, nc + dc) in occupied
        )
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _creates_2x2(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Avoid solid 2x2 grey blocks, which obscure the one-stroke structure."""
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            block = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
            if (r, c) not in block:
                continue
            if sum((cell in occupied) or (cell == (r, c)) for cell in block) == 4:
                return True
    return False


def _search_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    steps: int,
    allowed: set[tuple[int, int]],
    base_occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Exact-length self-avoiding 8-connected segment from start to end."""

    def rec(
        cur: tuple[int, int], steps_left: int, local_visited: set[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        if steps_left == 0:
            return [cur] if cur == end else None

        min_needed = max(abs(cur[0] - end[0]), abs(cur[1] - end[1]))
        if min_needed > steps_left:
            return None

        path_occupied = base_occupied | local_visited
        moves: list[tuple[int, int, int]] = []
        for nr, nc in neighbors8(*cur):
            nxt = (nr, nc)
            if nxt == end:
                if steps_left != 1:
                    continue
            else:
                if nxt not in allowed or nxt in forbidden or nxt in base_occupied or nxt in local_visited:
                    continue
            if _too_thick(nr, nc, path_occupied):
                continue
            if _creates_2x2(nr, nc, path_occupied):
                continue
            slack = steps_left - 1 - max(abs(nr - end[0]), abs(nc - end[1]))
            moves.append((nr, nc, slack))

        random.shuffle(moves)
        moves.sort(key=lambda item: (abs(item[2]), abs(item[0] - end[0]) + abs(item[1] - end[1])))

        for nr, nc, _slack in moves:
            nxt = (nr, nc)
            result = rec(nxt, steps_left - 1, local_visited | {nxt})
            if result is not None:
                return [cur] + result
        return None

    return rec(start, steps, {start})


def _build_path_through_ports(
    occupied_cells: set[tuple[int, int]], ports: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    allowed = {(r, c) for r in SAFE_ROWS for c in SAFE_COLS if (r, c) not in occupied_cells}
    min_total = 1 + sum(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ports, ports[1:], strict=False))
    if min_total > PATH_LEN_MAX:
        return None

    for _ in range(150):
        total_target = random.randint(max(PATH_LEN_MIN, min_total), PATH_LEN_MAX)
        path = [ports[0]]
        used = {ports[0]}
        remaining_add = total_target - 1
        ok = True
        for i, (start, end) in enumerate(zip(ports, ports[1:], strict=False)):
            future_min = sum(
                max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ports[i + 1 :], ports[i + 2 :], strict=False)
            )
            min_add = max(abs(start[0] - end[0]), abs(start[1] - end[1]))
            max_add = remaining_add - future_min
            if max_add < min_add:
                ok = False
                break

            choices = list(range(min_add, max_add + 1))
            random.shuffle(choices)
            segment = None
            future_ports = set(ports[i + 2 :])
            for add_steps in choices:
                segment = _search_segment(start, end, add_steps, allowed, used - {start}, future_ports)
                if segment is not None:
                    break
            if segment is None:
                ok = False
                break

            path.extend(segment[1:])
            used.update(segment[1:])
            remaining_add -= len(segment) - 1

        if ok and remaining_add == 0:
            return path
    return None


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    chain_len = random.choice(CHAIN_LENGTHS)
    num_distractors = 1 if chain_len == 4 else random.choice([1, 2])
    total_sprites = chain_len + num_distractors

    layout = _choose_connected_layout(shape_type, chain_len)
    if layout is None:
        return None

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, total_sprites)
    chain_borders = border_colors[:chain_len]
    distractor_borders = border_colors[chain_len:]

    chain_input_centers, chain_output_centers = _chain_centers(chain_borders)
    distractor_centers = [
        random.choice([c for c in palette if c != border_color]) for border_color in distractor_borders
    ]

    dummy_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dummy_patch = _make_sprite_patch(shape_type, 1, 2)
    occupied_cells: set[tuple[int, int]] = set()
    for pos in layout:
        _place_sprite(dummy_grid, dummy_patch, pos["r0"], pos["c0"])
    for r, c in zip(*np.where(dummy_grid != BG), strict=False):
        occupied_cells.add((int(r), int(c)))

    layout_orders = list(permutations(layout))
    random.shuffle(layout_orders)
    chosen_layout = None
    path_cells = None
    for order in layout_orders:
        ports = [item["port"] for item in order]
        candidate_path = _build_path_through_ports(occupied_cells, ports)
        if candidate_path is not None:
            chosen_layout = list(order)
            path_cells = candidate_path
            break
    if chosen_layout is None or path_cells is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for i, pos in enumerate(chosen_layout):
        patch = _make_sprite_patch(shape_type, chain_borders[i], chain_output_centers[i])
        if _overlaps(output_grid, patch, pos["r0"], pos["c0"]):
            return None
        _place_sprite(output_grid, patch, pos["r0"], pos["c0"])
    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    for i, pos in enumerate(chosen_layout):
        cr, cc = pos["center_pos"]
        input_grid[cr, cc] = chain_input_centers[i]

    distractor_specs = list(zip(distractor_borders, distractor_centers, strict=False))
    if not _place_corner_distractors(input_grid, shape_type, distractor_specs, set(path_cells)):
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
