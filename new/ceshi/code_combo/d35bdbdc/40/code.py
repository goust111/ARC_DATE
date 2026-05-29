# Combo variant of d35bdbdc (index 40)
# combo_spec: D10/2+D11/0
# Name: 链式传递+按路径长度排序

"""
Design analysis for the D10/2 + D11/0 fusion
============================================

variant_spec:
  idea:
    keep the original visual language:
      - 10x10 grid
      - grey one-stroke trail
      - 3x3 bordered sprites with one center pixel
      - non-connected sprites disappear
    fuse the two changes into one mechanism:
      - every trail-connected sprite survives as part of one chain
      - adjacent sprites along the trail define grey path segments
      - those segments are processed from shortest to longest
      - each processed segment copies the current source center into the neighbor
      - because later segments use the already-updated middle sprite, one endpoint color diffuses through the chain

  concrete_pixels:
    connected_sprites:
      count: 3 or 4
      placement: edge-anchored sprites with one inward-facing port each
      ordering: the grey snake passes through their ports in a unique trail order
    segment_lengths:
      count: connected_sprites - 1
      property: all distinct, and strictly monotone from one trail end toward the other
      effect: the shortest segment identifies the propagation start side; later longer segments continue the wave
    distractors:
      count: 1 or 2
      placement: clipped corner sprites outside the central snake area

  behavior_trace:
    original_local_step:
      - only the 2 endpoint sprites are kept
      - each endpoint uses its center color as a key into some other sprite's border color
      - transfer happens independently at the two ends
    fused_local_step:
      - all sprites connected to the grey trail are kept
      - consecutive connected sprites define segments with visible grey lengths
      - the shortest segment fires first; its color-link decides which endpoint is the source
      - every next longer segment must touch exactly one already-activated sprite, so the propagated color keeps moving
    concrete_example:
      - trail order borders: [2, 8, 4, 7]
      - segment lengths along trail: [8, 5, 2]
      - input centers: [8, 4, 7, 6]
      - processing order by length: (4,7) then (8,4) then (2,8)
      - output centers: [6, 6, 6, 6]
      - difference from the original task:
        the middle sprites are not passive lookup objects; they are updated and then immediately become sources

  complexity_calibration:
    why_not_two_sprites:
      - with only 2 connected sprites, there is no multi-step dependency and D11 becomes cosmetic
      - forcing 3 or 4 connected sprites makes the chained diffusion visible
    why_monotone_lengths:
      - arbitrary segment-length permutations could break the wave into disconnected local rewrites
      - a monotone ladder keeps the rule human-readable while still making length order essential
    why_unique_lengths:
      - ties would add a hidden tiebreaker
      - distinct lengths keep the order visually recoverable

  hidden_assumptions_audit:
    old_assumptions_that_break:
      - "only the two trail endpoints matter" is false; now every trail-connected sprite matters
      - "all transfers are independent" is false; later steps depend on earlier updates
      - "original input centers are enough for every transfer value" is false; current centers must be mutated in sequence
    new_assumptions_and_generator_guarantees:
      - every connected sprite has exactly one valid port on the central snake
      - consecutive port distances are all distinct
      - after sorting segments by length, each later segment touches exactly one already-activated sprite
      - the shortest segment has an unambiguous source/target direction from the visible color link

  placement_legality:
    connected_layout:
      - connected sprites sit on distinct sides so their inward ports land in rows/cols 3..6
      - the path is searched only inside a central safe region, avoiding accidental extra contacts
    distractor_layout:
      - distractors are clipped into corners and validated to have no trail port
      - overlap and 4-touch checks prevent merged components

  observability:
    human_cue:
      - the same border-center vocabulary from the base task remains visible
      - multiple surviving sprites and uneven grey gaps make the new control flow visible
      - one endpoint color floods through the whole kept chain, which is much more conspicuous than two independent endpoint rewrites
    color_discipline:
      - no step depends on numeric color order
      - all matching uses equality between visible border and center colors

  neatness:
    path_area: compact central snake keeps segment lengths legible
    sprite_area: edge anchors and corner distractors keep components separated
    inference:
      - from 2-3 examples, a solver can notice that shorter grey gaps are resolved first and the propagated color keeps spreading
"""

from __future__ import annotations

import random
from itertools import combinations, product

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


def _extract_connected_chain(
    g: np.ndarray, bg: int, grey: int
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Return all sprites, the trail-ordered connected chain, and the traced trail."""
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
        port_ids = sorted(trail_idx[p] for p in ports)
        connected.append(
            {
                **sp,
                "sprite_id": id(sp),
                "ports": ports,
                "port_idx": int(port_ids[0]),
            }
        )

    assert len(connected) >= 3, "expected at least three trail-connected sprites"
    chain = sorted(connected, key=lambda sp: (int(sp["port_idx"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])))
    return sprites, chain, trail


def _segment_specs(chain: list[dict]) -> list[dict]:
    """Adjacent trail segments between connected sprites."""
    segments = []
    for i in range(len(chain) - 1):
        segments.append(
            {
                "a": i,
                "b": i + 1,
                "length": int(chain[i + 1]["port_idx"]) - int(chain[i]["port_idx"]),
            }
        )
    assert all(seg["length"] > 0 for seg in segments), "ports should appear in trail order"
    return segments


def _infer_first_direction(chain: list[dict], centers: list[int], seg: dict) -> tuple[int, int]:
    """Use the visible border-center link on the shortest segment to choose source -> target."""
    a, b = int(seg["a"]), int(seg["b"])
    a_is_target = int(centers[a]) == int(chain[b]["border"])
    b_is_target = int(centers[b]) == int(chain[a]["border"])
    assert a_is_target ^ b_is_target, "shortest segment should have a unique source-target direction"
    return (b, a) if a_is_target else (a, b)


def _prune_to_chain(out: np.ndarray, sprites: list[dict], keep_ids: set[int], bg: int, grey: int) -> None:
    """Delete all non-kept sprites and any extra non-grey component fragments."""
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
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))

    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse one endpoint's center color through the whole chain of grey-connected bordered sprites.

    Input structure:
    - Background plus a grey (5) one-stroke trail, and several small bordered sprites with one distinct center pixel.
    - Three or four sprites attach to the same trail by valid ports; their consecutive port gaps have different visible lengths.
    - Unrelated bordered sprites may appear in corners away from the trail.

    Rule:
    - Keep every sprite that has a valid port on the grey trail, and delete the rest.
    - Between consecutive kept sprites, treat the grey cells as path segments and sort those segments from shortest to longest.
    - The shortest segment reveals the first source-target direction via the border/center color link; after that, each longer segment
      must extend the already-updated side of the chain, so the current propagated center color keeps moving outward.

    Key insight: The grey trail no longer just picks survivors or a static order. Its segment lengths define the update schedule, and
    because each later segment uses an already-modified middle sprite as the next source, one endpoint color spreads through the chain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites, chain, _trail = _extract_connected_chain(g, bg, grey)
    segments = _segment_specs(chain)
    sorted_segments = sorted(segments, key=lambda seg: (int(seg["length"]), int(seg["a"])))
    lengths = [int(seg["length"]) for seg in sorted_segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be distinct"

    current_centers = [int(sp["center"]) for sp in chain]
    first_source, first_target = _infer_first_direction(chain, current_centers, sorted_segments[0])
    current_centers[first_target] = current_centers[first_source]
    activated = {int(first_source), int(first_target)}

    for seg in sorted_segments[1:]:
        a, b = int(seg["a"]), int(seg["b"])
        a_on = a in activated
        b_on = b in activated
        assert a_on ^ b_on, "each later segment should touch exactly one activated sprite"
        source = a if a_on else b
        target = b if a_on else a
        assert int(current_centers[target]) == int(chain[source]["border"]), "later segment should continue the color chain"
        current_centers[target] = current_centers[source]
        activated.add(target)

    out = g.copy()
    keep_ids = {int(sp["sprite_id"]) for sp in chain}
    _prune_to_chain(out, sprites, keep_ids, bg, grey)

    for i, sp in enumerate(chain):
        r, c = sp["center_pos"]
        out[r, c] = int(current_centers[i])

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
CORNER_POSITIONS = [(-1, -1), (-1, 8), (8, -1), (8, 8)]
CENTRAL_SNAKE = [
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6),
    (5, 6),
    (6, 6),
    (6, 5),
    (6, 4),
    (6, 3),
    (5, 3),
    (4, 3),
]
SNAKE_INDEX = {cell: i for i, cell in enumerate(CENTRAL_SNAKE)}


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
        if ok:
            return picks
    return None


def _enumerate_connected_layouts(shape_type: str, chain_len: int) -> list[list[dict]]:
    """Enumerate non-overlapping edge layouts for connected sprites."""
    side_sets = [SIDES] if chain_len == 4 else list(combinations(SIDES, chain_len))
    dummy_patch = _make_sprite_patch(shape_type, 1, 2)
    layouts: list[list[dict]] = []
    for side_set in side_sets:
        for picks in product(*(SIDE_CANDIDATES[side] for side in side_set)):
            ports = [pick["port"] for pick in picks]
            if len(set(ports)) != len(ports):
                continue
            grid = np.zeros((GRID_H, GRID_W), dtype=int)
            ok = True
            placed: list[dict] = []
            for cand in picks:
                if _overlaps(grid, dummy_patch, cand["r0"], cand["c0"]):
                    ok = False
                    break
                _place_sprite(grid, dummy_patch, cand["r0"], cand["c0"])
                placed.append(cand)
            if ok:
                layouts.append(placed)
    random.shuffle(layouts)
    return layouts


def _propagation_centers(border_colors: list[int], source_on_left: bool) -> tuple[list[int], list[int], int]:
    """Build input/output centers for a stateful diffusion chain."""
    palette = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    assert palette, "expected at least one spare color for propagation source"
    seed = random.choice(palette)
    n = len(border_colors)
    output_centers = [seed] * n
    if source_on_left:
        input_centers = [seed] + border_colors[:-1]
    else:
        input_centers = border_colors[1:] + [seed]
    return input_centers, output_centers, seed


def _place_corner_distractors(
    input_grid: np.ndarray, shape_type: str, distractor_specs: list[tuple[int, int]], path_cells: set[tuple[int, int]]
) -> bool:
    slots = list(CORNER_POSITIONS)
    random.shuffle(slots)
    for border_color, center_color in distractor_specs:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        placed = False
        for r0, c0 in list(slots):
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


def _min_monotone_suffix(min_steps: list[int], start: int, prev: int) -> int:
    """Lower bound on the remaining strictly increasing sequence sum."""
    total = 0
    cur_prev = prev
    for i in range(start, len(min_steps)):
        cur = max(int(min_steps[i]), int(cur_prev) + 1)
        total += cur
        cur_prev = cur
    return total


def _candidate_length_sequences(min_steps: list[int], source_on_left: bool) -> list[list[int]]:
    """All distinct monotone segment-length choices within the allowed total budget."""
    oriented_min = list(min_steps if source_on_left else reversed(min_steps))
    results: list[list[int]] = []
    sum_min = sum(oriented_min)
    max_sum = PATH_LEN_MAX - 1
    if sum_min > max_sum:
        return results

    def rec(i: int, prev: int, acc: list[int], acc_sum: int) -> None:
        if i == len(oriented_min):
            if PATH_LEN_MIN - 1 <= acc_sum <= PATH_LEN_MAX - 1:
                seq = list(acc if source_on_left else reversed(acc))
                results.append(seq)
            return

        lower = max(int(oriented_min[i]), int(prev) + 1)
        for value in range(lower, PATH_LEN_MAX):
            min_rest = _min_monotone_suffix(oriented_min, i + 1, value)
            if acc_sum + value + min_rest > PATH_LEN_MAX - 1:
                break
            acc.append(value)
            rec(i + 1, value, acc, acc_sum + value)
            acc.pop()

    rec(0, 0, [], 0)
    return results


def _build_path_with_lengths(
    occupied_cells: set[tuple[int, int]], ports: list[tuple[int, int]], lengths: list[int]
) -> list[tuple[int, int]] | None:
    """Build a grey snake that visits the ports in order using exact consecutive lengths."""
    assert len(ports) == len(lengths) + 1
    allowed = {(r, c) for r in SAFE_ROWS for c in SAFE_COLS if (r, c) not in occupied_cells}

    for _ in range(120):
        path = [ports[0]]
        used = {ports[0]}
        ok = True
        for i, (start, end, steps) in enumerate(zip(ports, ports[1:], lengths, strict=False)):
            future_ports = set(ports[i + 2 :])
            segment = _search_segment(start, end, int(steps), allowed, used - {start}, future_ports)
            if segment is None:
                ok = False
                break
            path.extend(segment[1:])
            used.update(segment[1:])
        if ok:
            return path
    return None


def _choose_path_plan(layout: list[dict]) -> tuple[list[dict], list[tuple[int, int]], list[int], bool] | None:
    """Pick a central-snake subpath whose consecutive connected gaps are distinct and monotone."""
    try:
        ordered = sorted(layout, key=lambda item: SNAKE_INDEX[item["port"]])
    except KeyError:
        return None

    port_ids = [int(SNAKE_INDEX[item["port"]]) for item in ordered]
    if len(set(port_ids)) != len(port_ids):
        return None

    seg_lengths = [b - a for a, b in zip(port_ids, port_ids[1:], strict=False)]
    if not seg_lengths or len(set(seg_lengths)) != len(seg_lengths):
        return None
    increasing = seg_lengths == sorted(seg_lengths)
    decreasing = seg_lengths == sorted(seg_lengths, reverse=True)
    if not (increasing or decreasing):
        return None

    path = CENTRAL_SNAKE[port_ids[0] : port_ids[-1] + 1]
    if not (PATH_LEN_MIN <= len(path) <= PATH_LEN_MAX):
        return None
    return ordered, path, seg_lengths, increasing


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    chain_len = random.choice(CHAIN_LENGTHS)
    num_distractors = 1 if chain_len == 4 else random.choice([1, 2])
    total_sprites = chain_len + num_distractors

    layouts = _enumerate_connected_layouts(shape_type, chain_len)
    chosen_layout = None
    path_cells = None
    source_on_left = True
    for layout in layouts:
        plan = _choose_path_plan(layout)
        if plan is None:
            continue
        chosen_layout, path_cells, _lengths, source_on_left = plan
        break
    if chosen_layout is None or path_cells is None:
        return None

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, total_sprites)
    chain_borders = border_colors[:chain_len]
    distractor_borders = border_colors[chain_len:]
    input_centers, output_centers, _seed = _propagation_centers(chain_borders, source_on_left)
    distractor_centers = [
        random.choice([c for c in palette if c != border_color]) for border_color in distractor_borders
    ]

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for i, pos in enumerate(chosen_layout):
        patch = _make_sprite_patch(shape_type, chain_borders[i], output_centers[i])
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
        input_grid[cr, cc] = input_centers[i]

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
