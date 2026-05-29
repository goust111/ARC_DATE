# Combo variant of d35bdbdc (index 33)
# combo_spec: D5/2+D11/0
# Name: 按路径长度排序+跨层转移

import sys
from pathlib import Path
from collections import deque
from itertools import permutations, product
import random

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, neighbors4, neighbors8, paste


# Design analysis for combo D5/2 + D11/0
# 1) Concrete behavior and trace
#    The original task only had two grey-endpoint destinations, so there was nothing meaningful to sort.
#    To make "sort by path length" visible, the combo keeps 3-4 sprites attached to one grey trail. The
#    attached sprites are ordered along the trail by their unique port cells. Each adjacent trail segment is
#    one transfer candidate, and those candidates are processed from the shortest grey segment to the longest.
#    The D5 change redirects the write target from d.center to d.border: if src.current_border equals
#    dst.center, repaint dst's whole visible border with src.center while leaving dst.center untouched.
#    A minimal trace is:
#      A: border=2 center=7, B: border=4 center=2, C: border=8 center=7
#      grey segment lengths: A-B = 3, B-C = 6
#      step 1 (shorter segment): A->B matches because 2 == B.center, so B.border becomes 7
#      step 2 (longer segment): B->C now matches because B's updated border is 7, so C.border becomes 2
#    If the solver ignored ordering or used a static border lookup, only the first repaint would happen.
#
# 2) Complexity calibration
#    The cascade is intentionally narrow: one ordered chain, one visible comparison key per step, and
#    only 3-4 connected sprites. That is enough to make the new behavior observable without introducing a
#    large search space. The generator enforces strictly increasing segment lengths, so the human solver can
#    actually see why one pair should happen before the next. The color system is also disciplined:
#    one seed border color and one seed center color alternate through the chain, which gives repeated
#    visual support for the rule instead of many unrelated recolors.
#
# 3) Hidden assumptions audit
#    Several original assumptions break here:
#    - `_find_trail_endpoints()` is no longer enough because more than two sprites survive.
#    - A static `border -> source` dictionary is invalid because destination borders can become later sources.
#    - Writing only `center_pos` would silently implement the old rule; the solver must repaint the whole border.
#    - A direction-free interpretation of the trail is unsafe, because only one orientation yields the full
#      cascade. The solver therefore evaluates both trail directions and keeps the one with more successful
#      sorted transfers. The generator guarantees that this score is unique.
#
# 4) Placement legality
#    Connected sprites are edge-anchored and use exactly one inward-facing port each. The grey trail is
#    searched only inside the central safe area, so it cannot accidentally touch corner distractors. Corner
#    distractors are clipped but still valid 2-color bordered sprites, and they are explicitly checked to have
#    no valid grey ports.
#
# 5) Observability and cleanliness
#    The output keeps every trail-connected sprite and deletes the rest, so the preserved chain is easy to
#    isolate visually. Segment lengths differ by at least two grey steps, which makes the short-to-long order
#    legible in 2-3 examples. Because only borders change, centers act as stable visible keys across the chain.
#    The central snake plus edge/corner sprite layout avoids heavy overlap and keeps each object readable.
#
# 6) New solver assumptions and generator guarantees
#    - Each connected sprite has exactly one visible trail port.
#    - Adjacent port distances along the intended trail order are strictly increasing.
#    - Only the intended orientation yields the maximal number of successful transfers.
#    - Every intended repaint changes color, and every cascade beyond the first depends on an updated border.
#    `_build_path_through_ports()`, `_choose_chain_colors()`, and the oracle/metric checks enforce these.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
CHAIN_LENGTHS = [3, 4]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
GEN_RETRIES = 600
MAX_THICKNESS = 2
SEGMENT_STEP_GAP = 2
SAFE_ROWS = range(2, 8)
SAFE_COLS = range(2, 8)
SIDES = ("top", "right", "bottom", "left")
CORNER_POSITIONS = [(-1, -1), (-1, 8), (8, -1), (8, 8)]


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
        if border_color == bg or border_color == grey or center_color in (bg, grey):
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


def _ports_for_sprite(sprite: dict, g: np.ndarray, path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
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


def _repaint_border(out: np.ndarray, sprite: dict, color: int) -> None:
    """Recolor every visible non-center cell of a detected bordered sprite."""
    center_pos = sprite["center_pos"]
    for r, c in sprite["pts"]:
        if (int(r), int(c)) != center_pos:
            out[r, c] = int(color)


def _extract_connected_chain(g: np.ndarray, bg: int, grey: int) -> tuple[list[dict], list[tuple[int, int]], list[dict]]:
    """Detect all bordered sprites and keep the ones attached to the grey trail."""
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
        port = min(ports, key=lambda cell: (trail_idx[cell], cell[0], cell[1]))
        connected.append(
            {
                **sp,
                "sprite_id": id(sp),
                "ports": ports,
                "port": port,
                "port_idx": int(trail_idx[port]),
            }
        )

    assert len(connected) >= 2, "expected at least two trail-connected sprites"
    return sprites, trail, connected


def _segment_lengths(chain: list[dict]) -> list[int]:
    """Grey-step lengths between adjacent sprites in the chosen trail order."""
    return [abs(int(chain[i + 1]["port_idx"]) - int(chain[i]["port_idx"])) for i in range(len(chain) - 1)]


def _simulate_pairs(chain: list[dict], segment_lengths: list[int], *, cascade: bool) -> dict:
    """Simulate sorted transfers on a chain, optionally allowing updated borders to propagate."""
    current_borders = [int(sp["border"]) for sp in chain]
    original_borders = list(current_borders)
    step_order = sorted(range(len(segment_lengths)), key=lambda i: (int(segment_lengths[i]), i))
    repaints: list[tuple[int, int]] = []
    success = 0

    for i in step_order:
        src_border = current_borders[i] if cascade else original_borders[i]
        if src_border != int(chain[i + 1]["center"]):
            continue
        new_color = int(chain[i]["center"])
        repaints.append((i + 1, new_color))
        success += 1
        if cascade:
            current_borders[i + 1] = new_color

    return {
        "success": int(success),
        "repaints": repaints,
        "step_order": step_order,
        "final_borders": current_borders,
    }


def _choose_processing_chain(connected: list[dict]) -> tuple[list[dict], list[int], dict]:
    """Choose the trail orientation that supports the strongest sorted cascade."""
    forward = sorted(connected, key=lambda sp: (int(sp["port_idx"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])))
    reverse = list(reversed(forward))

    f_lengths = _segment_lengths(forward)
    r_lengths = _segment_lengths(reverse)
    f_sim = _simulate_pairs(forward, f_lengths, cascade=True)
    r_sim = _simulate_pairs(reverse, r_lengths, cascade=True)

    f_score = (int(f_sim["success"]), -len(forward))
    r_score = (int(r_sim["success"]), -len(reverse))
    if r_score > f_score:
        return reverse, r_lengths, r_sim
    return forward, f_lengths, f_sim


def _cleanup_unconnected(out: np.ndarray, sprites: list[dict], keep_ids: set[int], bg: int, grey: int) -> None:
    """Delete every sprite not selected into the surviving trail-connected chain."""
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
    Repaint borders along a grey-connected sprite chain, processing shorter path gaps first.

    Input structure:
    - The grid contains background, one thin grey (5) trail, and several small bordered sprites.
    - Each sprite has one repeated border color and one single-pixel center color, and 3-4 of those
      sprites touch the grey trail through visible port cells while other sprites may sit elsewhere.

    Rule:
    - Keep every sprite that is attached to the grey trail by a valid port, and remove the rest.
    - Order the kept sprites along the trail, consider each adjacent trail segment as one candidate transfer,
      and process those candidates from the shortest segment to the longest.
    - For a candidate src -> dst, if src's current border color equals dst's center color, repaint the whole
      visible border of dst with src's center color; this updated border can then enable later transfers.

    Key insight:
    - The grey trail is now a multi-step schedule rather than a two-endpoint selector: segment length decides
      when each neighbor pair is evaluated, and because transfers write onto borders, earlier repaints can
      create the color matches required by later, longer segments.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites, _trail, connected = _extract_connected_chain(g, bg, grey)
    chain, segment_lengths, sim = _choose_processing_chain(connected)
    keep_ids = {int(sp["sprite_id"]) for sp in chain}

    out = g.copy()
    _cleanup_unconnected(out, sprites, keep_ids, bg, grey)

    for dst_idx, color in sim["repaints"]:
        _repaint_border(out, chain[dst_idx], int(color))

    return out


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on the grid, skipping transparent background cells."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a sprite placement by clearing every non-background patch cell."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlaps and 4-touching between sprite cells."""
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
    """Enumerate edge placements whose unique inward port sits in the safe central area."""
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
    """Pick non-overlapping edge slots for the trail-connected sprites."""
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


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Reject additions that would make the grey trail branch or blob too much."""
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
    """Avoid solid 2x2 grey blocks."""
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


def _candidate_segment_steps(min_steps: list[int]) -> list[tuple[int, ...]]:
    """Enumerate strictly increasing per-segment grey lengths within the global path budget."""
    ranges = [range(int(m), PATH_LEN_MAX) for m in min_steps]
    cands: list[tuple[int, ...]] = []
    for steps in product(*ranges):
        if any(steps[i + 1] < steps[i] + SEGMENT_STEP_GAP for i in range(len(steps) - 1)):
            continue
        total_cells = 1 + sum(int(s) for s in steps)
        if PATH_LEN_MIN <= total_cells <= PATH_LEN_MAX:
            cands.append(tuple(int(s) for s in steps))
    random.shuffle(cands)
    return cands


def _build_path_through_ports(
    occupied_cells: set[tuple[int, int]], ports: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]] | None, list[int] | None]:
    """Build a one-stroke trail whose consecutive port gaps are strictly increasing."""
    allowed = {(r, c) for r in SAFE_ROWS for c in SAFE_COLS if (r, c) not in occupied_cells}
    min_steps = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ports, ports[1:], strict=False)]
    for step_tuple in _candidate_segment_steps(min_steps):
        path = [ports[0]]
        used = {ports[0]}
        ok = True
        for i, (start, end) in enumerate(zip(ports, ports[1:], strict=False)):
            future_ports = set(ports[i + 2 :])
            segment = _search_segment(start, end, int(step_tuple[i]), allowed, used - {start}, future_ports)
            if segment is None:
                ok = False
                break
            path.extend(segment[1:])
            used.update(segment[1:])
        if ok:
            return path, [int(v) for v in step_tuple]
    return None, None


def _choose_chain_colors(chain_borders: list[int], total_borders: list[int]) -> tuple[list[int], list[int]] | None:
    """Choose centers and output borders that force the sorted cascade."""
    spare = [c for c in range(1, 10) if c not in set(total_borders) and c != GREY]
    if not spare:
        return None
    seed_border = int(chain_borders[0])
    seed_center = random.choice(spare)
    chain_centers = [seed_center if i % 2 == 0 else seed_border for i in range(len(chain_borders))]
    output_borders = [seed_border] + [chain_centers[i - 1] for i in range(1, len(chain_borders))]
    return chain_centers, output_borders


def _place_corner_distractors(
    input_grid: np.ndarray, shape_type: str, distractor_specs: list[tuple[int, int]], path_cells: set[tuple[int, int]]
) -> bool:
    """Place clipped corner distractors and confirm they do not attach to the grey trail."""
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


def _place_connected_output(
    output_grid: np.ndarray, shape_type: str, ordered_layout: list[dict], output_borders: list[int], centers: list[int]
) -> None:
    """Place the trail-connected sprites in their output-state colors."""
    for i, pos in enumerate(ordered_layout):
        patch = _make_sprite_patch(shape_type, int(output_borders[i]), int(centers[i]))
        _place_sprite(output_grid, patch, int(pos["r0"]), int(pos["c0"]))


def _restore_connected_input(
    input_grid: np.ndarray, shape_type: str, ordered_layout: list[dict], input_borders: list[int], centers: list[int]
) -> None:
    """Overwrite the output-state connected sprites back to their input border colors."""
    for i, pos in enumerate(ordered_layout):
        patch = _make_sprite_patch(shape_type, int(input_borders[i]), int(centers[i]))
        _place_sprite(input_grid, patch, int(pos["r0"]), int(pos["c0"]))


def _instance_metrics(grid: np.ndarray) -> dict:
    """Quantify whether the generated instance really uses cross-layer cascade behavior."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sprites, _trail, connected = _extract_connected_chain(g, bg, GREY)
    chain, segment_lengths, dynamic = _choose_processing_chain(connected)
    static = _simulate_pairs(chain, segment_lengths, cascade=False)
    out = main(g)

    center_changes = 0
    border_changes = 0
    for sp in chain:
        r, c = sp["center_pos"]
        center_changes += int(out[r, c] != g[r, c])
        for rr, cc in sp["pts"]:
            if (int(rr), int(cc)) == sp["center_pos"]:
                continue
            border_changes += int(out[rr, cc] != g[rr, cc])

    return {
        "connected_count": len(chain),
        "segment_lengths": tuple(int(v) for v in segment_lengths),
        "dynamic_steps": int(dynamic["success"]),
        "static_steps": int(static["success"]),
        "cascade_gap": int(dynamic["success"] - static["success"]),
        "center_changes": int(center_changes),
        "border_changes": int(border_changes),
        "kept_sprites": len(chain),
        "total_sprites": len(sprites),
    }


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

    colors = _choose_chain_colors(chain_borders, border_colors)
    if colors is None:
        return None
    chain_centers, output_borders = colors

    dummy_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dummy_patch = _make_sprite_patch(shape_type, 1, 2)
    occupied_cells: set[tuple[int, int]] = set()
    for pos in layout:
        _place_sprite(dummy_grid, dummy_patch, int(pos["r0"]), int(pos["c0"]))
    for r, c in zip(*np.where(dummy_grid != BG), strict=False):
        occupied_cells.add((int(r), int(c)))

    layout_orders = list(permutations(layout))
    random.shuffle(layout_orders)
    chosen_layout = None
    path_cells = None
    segment_steps = None
    for order in layout_orders:
        ports = [item["port"] for item in order]
        candidate_path, candidate_steps = _build_path_through_ports(occupied_cells, ports)
        if candidate_path is not None:
            chosen_layout = list(order)
            path_cells = candidate_path
            segment_steps = candidate_steps
            break
    if chosen_layout is None or path_cells is None or segment_steps is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    _place_connected_output(output_grid, shape_type, chosen_layout, output_borders, chain_centers)

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    _restore_connected_input(input_grid, shape_type, chosen_layout, chain_borders, chain_centers)

    distractor_centers = []
    for border_color in distractor_borders:
        candidates = [c for c in palette if c != border_color]
        if not candidates:
            return None
        distractor_centers.append(random.choice(candidates))
    distractor_specs = list(zip(distractor_borders, distractor_centers, strict=False))
    if not _place_corner_distractors(input_grid, shape_type, distractor_specs, set(path_cells)):
        return None

    try:
        out = main(input_grid)
        metrics = _instance_metrics(input_grid)
        if (
            np.array_equal(out, output_grid)
            and tuple(segment_steps) == metrics["segment_lengths"]
            and metrics["connected_count"] == chain_len
            and metrics["dynamic_steps"] == chain_len - 1
            and metrics["static_steps"] == 1
            and metrics["cascade_gap"] == chain_len - 2
            and metrics["center_changes"] == 0
            and metrics["border_changes"] > 0
        ):
            return input_grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
