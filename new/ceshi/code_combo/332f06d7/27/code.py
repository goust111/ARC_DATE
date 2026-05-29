# Combo variant of 332f06d7 (index 27)
# combo_spec: D2/2+D16/0
# Name: 多级门控+显式障碍

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for the fused variant
=====================================

a) Concretization and behavior trace
------------------------------------
Original task:
- one black n×n square and one red n×n square lie on a blue winding river
- a stuck case is created by literally carving the river thinner than n
- the solver picks one size class, builds the n×n feasible-placement grid, and BFSes on it

Fused task here:
- we keep one mover / one target of the same visible size n
- the river now contains several explicit obstacle bars of a fifth color
- each obstacle bar forms a gate together with a one-sided side-pocket of blue pixels
- different gates have different visible side-pocket clearances, e.g. for n=2 the sequence may be
  widths [4, 1, 3]:
  1. gate 1: width 4 >= 2, so the square can drop into the pocket and bypass the obstacle
  2. gate 2: width 1 < 2, so the square cannot shift far enough to clear the obstacle column
  3. gate 3 exists visually farther ahead, but the square never reaches it
- output: the black square moves to the red target if every gate on the route has width >= n;
  otherwise it stops at the farthest reachable n×n placement before the first too-narrow gate

The crucial difference from the original is that "blocked vs passable" is no longer encoded by
"where blue river was erased into background". It is encoded by an explicit obstacle color inside
the river, with several gates of different visible widths along the same route.

b) Complexity calibration
-------------------------
- The layout uses exactly one mover, one destination, one river color, one obstacle color, and one
  background color. This keeps color semantics legible.
- Gates are limited to 2-3 per sample, with distinct widths drawn from a tiny set {1,2,3,4}. That
  is enough repetition to expose the rule without letting combinatorics explode.
- The obstacle shape is always a thin rectangular bar. Visual variety comes from gate width, side
  pocket side (up/down), and global rotation/flip, not from many obstacle types.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original assumption: sorting candidate squares by raw color id is safe.
  Broken here: the fused task should remain human-visible, so mover/dest are identified by geometry:
  the destination touches an outer border, while the mover is near a border but does not touch it.
- Original assumption: all non-background non-square pixels belong to one traversable river color.
  Broken here: a second non-background non-square color is now an obstacle and must be excluded from
  traversability.
- Original assumption: a stuck case only comes from river self-narrowing.
  Broken here: the river can stay pixel-connected while explicit obstacles eliminate every feasible
  n×n window through a gate.
- Original assumption: one BFS over one size class is enough.
  Still valid here because this fused design keeps exactly one mover/target size pair.

d) Placement legality
---------------------
- The generator paints a rectangular base corridor and then adds side pockets for each gate.
- Every obstacle bar is strictly rectangular and width 1, so it cannot be mistaken for a solid
  square target object.
- Grid size is square and bounded by 19-20, so all objects remain in-bounds after optional symmetry
  transforms.
- The destination is always flush with one outer edge; the mover is always one cell away from an
  edge before transform. The solver relies on this and the generator guarantees it.

e) Visual observability
-----------------------
- A human can literally compare black-square size against the pocket width beside each obstacle bar.
- Multiple obstacle gates appear in one sample, so the task is visibly "check each gate in order",
  not just "there is one odd obstacle somewhere".
- No hidden ordering by color number or component enumeration is used.

f) Visual cleanliness
---------------------
- The main corridor stays horizontal before the final global symmetry transform; pockets alternate
  above/below so the path looks winding without cluttering the whole board.
- Obstacles occupy the center lane only; the blue side pockets are visually separate, making the
  bypass mechanism clear.

g) Change inferability
----------------------
- Only one new color is introduced, dedicated to obstacles.
- Gate width is accompanied by structure: the obstacle bar, the side pocket, and the black square's
  stopping depth all co-vary. That gives enough evidence from few samples to infer the rule.

New main() assumptions and generator guarantees
-----------------------------------------------
- Assumption: there is exactly one repeated square size with one edge-touching square and one
  non-edge-touching square. Guarantee: generator creates exactly one mover/destination pair.
- Assumption: among colors other than background/mover/destination, the most common is the river and
  the rest are explicit obstacle colors. Guarantee: obstacle area is tiny compared with the river.
- Assumption: the mover's start is a feasible n×n placement. Guarantee: generator validates this
  with the same feasibility/BFS logic before returning.
"""


_BG, _RIVER, _MOVER, _DEST, _OBST = 3, 1, 0, 2, 8
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEXT_REACHED = False
_LAST_META: dict | None = None


def _background_color(g: np.ndarray) -> int:
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _solid_squares(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        bbox = comp["bbox"]
        if bbox_h(bbox) != bbox_w(bbox):
            continue
        patch = g[int(bbox[0]) : int(bbox[2]) + 1, int(bbox[1]) : int(bbox[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(bbox[0]), "c0": int(bbox[1]), "n": int(bbox_h(bbox))})
    return squares


def _touches_edge(square: dict, H: int, W: int) -> bool:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_gap(square: dict, H: int, W: int) -> int:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _pick_roles(squares: list[dict], H: int, W: int) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for square in squares:
        by_n[int(square["n"])].append(square)

    candidate_sizes = [n for n, group in by_n.items() if len(group) >= 2]
    assert candidate_sizes
    nside = max(candidate_sizes, key=lambda n: (len(by_n[n]), n))
    candidates = list(by_n[nside])

    edge_touching = [s for s in candidates if _touches_edge(s, H, W)]
    interior = [s for s in candidates if not _touches_edge(s, H, W)]
    assert edge_touching and interior

    dest = min(edge_touching, key=lambda s: _edge_gap(s, H, W))
    mover = min(interior, key=lambda s: _edge_gap(s, H, W))
    return mover, dest


def _remaining_colors(g: np.ndarray, excluded: set[int]) -> list[tuple[int, int]]:
    counts = Counter(map(int, g.flatten()))
    colors = [(int(col), int(ct)) for col, ct in counts.items() if int(col) not in excluded]
    colors.sort(key=lambda kv: (-kv[1], kv[0]))
    return colors


def _feasible_mask(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def _pick_terminal(dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> tuple[int, int]:
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and feasible[dr, dc] and dist[dr, dc] >= 0:
        return dr, dc

    best = -1
    tr, tc = sr, sc
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def _solve_state(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _background_color(g)
    squares = _solid_squares(g, bg)
    assert len(squares) >= 2

    mover, dest = _pick_roles(squares, H, W)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    remaining = _remaining_colors(g, {bg, int(mover["col"]), int(dest["col"])})
    assert remaining
    river = int(remaining[0][0])
    obstacle_colors = {int(col) for col, _ in remaining[1:]}

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    for obs in obstacle_colors:
        trav &= g != obs

    feasible = _feasible_mask(trav, n)
    assert feasible[sr, sc]
    dist = _bfs_placement(feasible, sr, sc)
    tr, tc = _pick_terminal(dist, feasible, sr, sc, dr, dc)
    return {
        "grid": g,
        "bg": bg,
        "river": river,
        "mover": mover,
        "dest": dest,
        "n": n,
        "sr": sr,
        "sc": sc,
        "dr": dr,
        "dc": dc,
        "feasible": feasible,
        "dist": dist,
        "tr": tr,
        "tc": tc,
        "obstacle_colors": obstacle_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    state = _solve_state(grid)
    g = state["grid"]
    sr, sc = int(state["sr"]), int(state["sc"])
    tr, tc = int(state["tr"]), int(state["tc"])
    n = int(state["n"])
    river = int(state["river"])
    mover_col = int(state["mover"]["col"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = mover_col
    return out


def _apply_transform(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, 1)
    if code == 2:
        return np.rot90(arr, 2)
    if code == 3:
        return np.rot90(arr, 3)
    if code == 4:
        return np.fliplr(arr)
    if code == 5:
        return np.flipud(arr)
    if code == 6:
        return np.rot90(np.fliplr(arr), 1)
    if code == 7:
        return np.rot90(np.flipud(arr), 1)
    raise AssertionError("bad transform code")


def _count_components(mask: np.ndarray) -> int:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    count = 0
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _gate_width_templates(n: int, reached: bool) -> list[list[int]]:
    if n == 2:
        if reached:
            return [[4, 3, 2], [3, 4, 2], [2, 4, 3]]
        return [[4, 2, 1], [4, 1, 3], [3, 1, 4]]
    if n == 3:
        if reached:
            return [[4, 3], [3, 4]]
        return [[4, 2], [3, 2]]
    raise AssertionError("unsupported n")


def _paint_gate(grid: np.ndarray, base_r: int, x0: int, n: int, clearance: int, side: str) -> None:
    chamber_len = 2 * n + 1
    x1 = x0 + chamber_len
    if side == "up":
        grid[base_r - clearance : base_r, x0:x1] = _RIVER
    else:
        grid[base_r + n : base_r + n + clearance, x0:x1] = _RIVER
    grid[base_r : base_r + n, x0 + n : x0 + n + 1] = _OBST


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    state = _solve_state(grid)
    out = state["grid"].copy()
    sr, sc = int(state["sr"]), int(state["sc"])
    tr, tc = int(state["tr"]), int(state["tc"])
    n = int(state["n"])
    out[sr : sr + n, sc : sc + n] = int(state["river"])
    out[tr : tr + n, tc : tc + n] = int(state["mover"]["col"])
    return out, state


def _variant_stats(grid: np.ndarray, gate_widths: list[int]) -> dict:
    state = _solve_state(grid)
    g = state["grid"]
    n = int(state["n"])
    sr, sc = int(state["sr"]), int(state["sc"])
    dr, dc = int(state["dr"]), int(state["dc"])
    river = int(state["river"])

    trav_with = (g == river) | (g == int(state["mover"]["col"])) | (g == int(state["dest"]["col"]))
    trav_without = trav_with | (g == _OBST)
    feasible_with = _feasible_mask(trav_with, n)
    feasible_without = _feasible_mask(trav_without, n)
    dist_with = _bfs_placement(feasible_with, sr, sc)
    dist_without = _bfs_placement(feasible_without, sr, sc)

    first_block = None
    for idx, width in enumerate(gate_widths):
        if width < n:
            first_block = idx
            break

    return {
        "n": n,
        "gate_widths": list(gate_widths),
        "gate_count": len(gate_widths),
        "distinct_widths": len(set(gate_widths)),
        "obstacle_components": _count_components(g == _OBST),
        "first_block_gate": first_block,
        "feasible_with": int(feasible_with.sum()),
        "feasible_without": int(feasible_without.sum()),
        "feasible_loss": int(feasible_without.sum() - feasible_with.sum()),
        "dest_with_obstacles": int(dist_with[dr, dc]) if feasible_with[dr, dc] else -1,
        "dest_without_obstacles": int(dist_without[dr, dc]) if feasible_without[dr, dc] else -1,
        "farthest_with_obstacles": int(dist_with.max()),
        "farthest_without_obstacles": int(dist_without.max()),
        "obstacle_pixels": int((g == _OBST).sum()),
    }


def _build_case(reached: bool) -> tuple[np.ndarray, dict] | None:
    n = random.choice([2, 3])
    gate_widths = list(random.choice(_gate_width_templates(n, reached)))
    gate_count = len(gate_widths)
    chamber_len = 2 * n + 1
    tail_len = n + 1
    layout_width = 1 + gate_count * chamber_len + tail_len
    if layout_width > 20:
        return None

    first_side = random.choice(["up", "down"])
    sides = [first_side if idx % 2 == 0 else ("down" if first_side == "up" else "up") for idx in range(gate_count)]
    max_up = max([w for w, side in zip(gate_widths, sides) if side == "up"] + [0])
    max_down = max([w for w, side in zip(gate_widths, sides) if side == "down"] + [0])

    inner_h = max(10, max_up + n + max_down + 3)
    G = random.choice([layout_width, min(20, layout_width + 1)])
    if inner_h > G:
        G = inner_h
    if G > 20:
        return None

    grid = np.full((G, G), _BG, dtype=int)
    x_shift = G - layout_width
    top_slack = G - inner_h
    y_shift = random.randint(0, top_slack) if top_slack > 0 else 0
    base_r = y_shift + max_up + 1

    corridor_start = x_shift + 1
    grid[base_r : base_r + n, corridor_start:G] = _RIVER

    for idx, (width, side) in enumerate(zip(gate_widths, sides)):
        x0 = corridor_start + idx * chamber_len
        _paint_gate(grid, base_r, x0, n, width, side)

    mover_r, mover_c = base_r, corridor_start
    dest_r, dest_c = base_r, G - n
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    expected, state = _oracle_output(grid)
    feasible = state["feasible"]
    dist = state["dist"]
    if not feasible[mover_r, mover_c]:
        return None
    dest_ok = bool(feasible[dest_r, dest_c] and dist[dest_r, dest_c] >= 0)
    if dest_ok != reached:
        return None

    stats = _variant_stats(grid, gate_widths)
    if stats["gate_count"] < 2 or stats["distinct_widths"] != stats["gate_count"]:
        return None
    if stats["obstacle_components"] != gate_count:
        return None
    if stats["obstacle_pixels"] != gate_count * n:
        return None
    if stats["dest_without_obstacles"] < 0:
        return None
    if stats["feasible_loss"] <= 0:
        return None
    if reached:
        if stats["dest_with_obstacles"] < 0:
            return None
    else:
        if stats["dest_with_obstacles"] >= 0:
            return None
        if stats["farthest_without_obstacles"] <= stats["farthest_with_obstacles"]:
            return None

    code = random.randint(0, 7)
    transformed_in = _apply_transform(grid, code)
    transformed_out = _apply_transform(expected, code)
    if not np.array_equal(main(transformed_in), transformed_out):
        return None

    transformed_stats = _variant_stats(transformed_in, gate_widths)
    if transformed_stats["gate_count"] < 2 or transformed_stats["distinct_widths"] != transformed_stats["gate_count"]:
        return None
    if transformed_stats["obstacle_components"] != gate_count:
        return None

    meta = {
        "reached": reached,
        "n": n,
        "gate_widths": list(gate_widths),
        "sides": list(sides),
        "stats": transformed_stats,
    }
    return transformed_in, meta


def generate_input() -> np.ndarray:
    global _NEXT_REACHED, _LAST_META

    target = _NEXT_REACHED
    _NEXT_REACHED = not _NEXT_REACHED

    for _ in range(300):
        built = _build_case(target)
        if built is None:
            continue
        grid, meta = built
        _LAST_META = meta
        return grid

    for _ in range(150):
        built = _build_case(random.random() < 0.5)
        if built is None:
            continue
        grid, meta = built
        _LAST_META = meta
        return grid

    raise RuntimeError("Failed to generate valid input after 450 attempts")
