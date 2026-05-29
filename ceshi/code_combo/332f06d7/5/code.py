# Combo variant of 332f06d7 (index 5)
# combo_spec: D2/3+D16/1
# Name: 动态障碍+可推动障碍

from __future__ import annotations

from collections import Counter, deque
import random
import numpy as np


# Design analysis
# ===============
# a) Concrete behavior and trace
#    - I intentionally collapse the mover/destination size to 1x1. This is the same
#      compatibility move used by the standalone D16 push variant: once pushing is
#      stateful, keeping a larger NxN mover makes the task dominated by window-geometry
#      details instead of the visible obstacle logic. Here the human-visible change is:
#        blue channel + yellow fixed blockers + magenta pushable blockers.
#    - The generator builds a rectangular fork:
#        start ---- A ===== upper lane ===== B ---- dest
#                    \==== lower lane =====/
#      The upper lane is sealed by fixed yellow blocker(s). The lower lane contains a
#      corner occupied by one or more magenta pushable cells, plus a short blue pocket
#      behind them. To proceed, the black mover must push that chain into the pocket.
#      In stuck cases there is a second lower-lane push gate with no spare pocket cell.
#    - Small trace:
#        original task: the black square stops because the blue river narrows.
#        combo task: the river does not pinch. Instead:
#          1. the obvious upper route is blocked by yellow fixed cells,
#          2. the mover is forced onto the lower route,
#          3. at the first lower corner it pushes magenta cells out of the way,
#          4. if a later lower corner also has enough pocket, it reaches red; otherwise
#             it stops at the farthest reachable point before that blocked push gate.
#    - Difference from simply "adding both obstacle types":
#      the fixed obstacle is not decorative. It forces the mover onto the route where
#      pushing happens. If the fixed blocker were removed, the solver would usually take
#      the upper bypass and leave the pushable cells untouched.
#
# b) Complexity calibration
#    - Only six stable colors are used:
#        green=background, blue=channel, black=mover, red=destination,
#        yellow=fixed obstacle, magenta=pushable obstacle.
#    - Each sample contains:
#        at least one fixed obstacle,
#        at least one successful push,
#        and in stuck cases one later blocked push gate.
#      This keeps the fusion visible in 2-4 examples without introducing many object
#      categories or hidden numerical thresholds.
#
# c) Assumption audit against the original solver
#    - Original assumption: roles are recovered by "same-sized solid squares" and color
#      sorting. This is no longer valid because pushable/fixed obstacles add more colored
#      components and the mover is now 1x1. New solver assumption: exactly one black
#      cell and one red cell exist. The generator guarantees that.
#    - Original assumption: traversability is static and depends only on river width.
#      This is invalid because magenta obstacles move. New solver performs BFS on states
#      (mover position, pushable-obstacle positions).
#    - Original assumption: output only moves the black square. This is invalid because
#      pushed obstacles must be redrawn at their new locations.
#    - Original assumption: "river = most common non-background color" is unnecessary.
#      Stable visible color roles are clearer and more ARC-friendly here.
#
# d) Placement legality
#    - The generator samples one clean fork layout entirely inside the grid, then places
#      blockers only on already carved channel cells.
#    - Push pockets extend into empty space that is guaranteed not to overlap the main
#      route or the destination.
#    - The lower blocked gate, when present, is always downstream from the first
#      successful push so the output still exhibits an actual push before stopping.
#
# e) Visual observability
#    - Humans can see that yellow cells never move, while magenta cells shift into the
#      short blue pocket in the output. Comparing multiple examples shows the mover is
#      forced away from the yellow-blocked branch and can only progress when pushing the
#      magenta chain is physically possible.
#
# f) Visual cleanliness
#    - The scene is a sparse orthogonal fork rather than a dense maze. Pockets are only
#      0-2 cells long, and blockers live on separate branches/corners, so the diagram
#      stays readable.
#
# g) New main() assumptions and generator contract
#    - main() assumes:
#        one black mover, one red destination,
#        at least one yellow fixed obstacle,
#        at least one magenta pushable obstacle,
#        all non-background cells belong to the channel/fork structure.
#    - generate_input() enforces all of those assumptions and additionally rejects any
#      sample where removing fixed blockers or disabling pushing would yield the same
#      final output, preventing degeneration back to a simpler task.


_BG = 3
_RIVER = 1
_MOVER = 0
_DEST = 2
_FIXED = 4
_PUSH = 8

# Up-first ordering is deliberate: if the fixed blockers are removed, the solver prefers
# the upper bypass instead of gratuitously taking the lower push route.
_DIR4 = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def _find_single(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == color)
    assert coords.shape[0] == 1
    return int(coords[0, 0]), int(coords[0, 1])


def _sorted_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _trace_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    assert ac == bc
    step = 1 if br >= ar else -1
    return [(r, ac) for r in range(ar, br + step, step)]


def _trace_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _trace_segment(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _simulate_states(grid: np.ndarray, *, allow_push: bool = True) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(
        Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0]
    )
    mover = _find_single(g, _MOVER)
    dest = _find_single(g, _DEST)
    fixed = _sorted_coords(g == _FIXED)
    pushes0 = _sorted_coords(g == _PUSH)
    assert len(pushes0) >= 1

    floor = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    fixed_set = set(fixed)

    start_state = (mover, pushes0)
    q = deque([start_state])
    dist = {start_state: 0}
    best_state = start_state
    best_pos_dist = 0
    seen_pos = {mover}
    dest_state = None

    while q:
        state = q.popleft()
        (mr, mc), pushes = state
        cur_dist = dist[state]
        if (mr, mc) == dest:
            dest_state = state
            break

        push_set = set(pushes)
        for dr, dc in _DIR4:
            nr, nc = mr + dr, mc + dc
            nxt = (nr, nc)
            if nxt not in floor or nxt in fixed_set:
                continue

            if nxt not in push_set:
                new_state = (nxt, pushes)
            else:
                if not allow_push:
                    continue
                chain = []
                cr, cc = nr, nc
                while (cr, cc) in push_set:
                    chain.append((cr, cc))
                    cr += dr
                    cc += dc
                landing = (cr, cc)
                if landing not in floor or landing in fixed_set or landing == dest:
                    continue

                new_pushes = set(push_set)
                for cell in chain:
                    new_pushes.remove(cell)
                for cell in chain:
                    new_pushes.add((cell[0] + dr, cell[1] + dc))
                if dest in new_pushes:
                    continue
                new_state = (nxt, tuple(sorted(new_pushes)))

            if new_state in dist:
                continue
            dist[new_state] = cur_dist + 1
            if new_state[0] not in seen_pos:
                seen_pos.add(new_state[0])
                if cur_dist + 1 > best_pos_dist:
                    best_pos_dist = cur_dist + 1
                    best_state = new_state
            q.append(new_state)

    return {
        "start": start_state,
        "best": best_state,
        "dest_state": dest_state,
        "dest": dest,
        "bg": bg,
    }


def _render_state(
    grid: np.ndarray,
    mover: tuple[int, int],
    pushes: tuple[tuple[int, int], ...],
    *,
    reached: bool,
) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    out[out == _MOVER] = _RIVER
    out[out == _PUSH] = _RIVER
    if reached:
        out[out == _DEST] = _RIVER
    for r, c in pushes:
        out[r, c] = _PUSH
    out[mover] = _MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the black cell through the blue fork toward the red destination.

    Yellow cells are fixed walls inside the channel. Magenta cells are pushable: when
    the mover walks into a contiguous magenta chain, the whole chain shifts forward by
    one step if there is free blue channel behind it; otherwise the move is illegal.
    The mover keeps moving through all reachable states until it either reaches red or
    stops at the farthest reachable position. Any pushed magenta cells remain moved in
    the output.
    """
    sim = _simulate_states(grid, allow_push=True)
    if sim["dest_state"] is not None:
        mover, pushes = sim["dest_state"]
        return _render_state(grid, mover, pushes, reached=True)
    mover, pushes = sim["best"]
    return _render_state(grid, mover, pushes, reached=False)


def _sample_layout(size: int):
    for _ in range(200):
        dest_c = size - 2
        c1_hi = min(size // 3, dest_c - 5)
        if c1_hi < 3:
            return None
        c1 = random.randint(3, c1_hi)
        c2_lo = c1 + 4
        c2_hi = dest_c - 3
        if c2_hi < c2_lo:
            continue
        c2 = random.randint(c2_lo, c2_hi)

        r_mid = random.randint(4, size - 5)
        top_gap_hi = min(3, r_mid - 1)
        bot_gap_hi = min(4, size - r_mid - 3)
        if top_gap_hi < 2 or bot_gap_hi < 3:
            continue
        top_gap = random.randint(2, top_gap_hi)
        bot_gap = random.randint(3, bot_gap_hi)

        r_top = r_mid - top_gap
        r_bot = r_mid + bot_gap

        start = (r_mid, 1)
        a = (r_mid, c1)
        u1 = (r_top, c1)
        u2 = (r_top, c2)
        b = (r_mid, c2)
        l1 = (r_bot, c1)
        l2 = (r_bot, c2)
        dest = (r_mid, dest_c)

        river = set(_trace_polyline([start, a]))
        river.update(_trace_polyline([a, u1, u2, b]))
        river.update(_trace_polyline([a, l1, l2, b]))
        river.update(_trace_polyline([b, dest]))

        return {
            "size": size,
            "start": start,
            "dest": dest,
            "a": a,
            "b": b,
            "u1": u1,
            "u2": u2,
            "l1": l1,
            "l2": l2,
            "river": river,
        }
    return None


def _available_pocket_len(size: int, gate: tuple[int, int], step: tuple[int, int], river: set[tuple[int, int]]) -> int:
    count = 0
    cur = gate
    while count < 2:
        cur = (cur[0] + step[0], cur[1] + step[1])
        if not (0 <= cur[0] < size and 0 <= cur[1] < size):
            break
        if cur in river:
            break
        count += 1
    return count


def _build_grid(layout: dict, *, reachable: bool) -> np.ndarray | None:
    size = int(layout["size"])
    start = tuple(layout["start"])
    dest = tuple(layout["dest"])
    river = set(layout["river"])
    l1 = tuple(layout["l1"])
    l2 = tuple(layout["l2"])
    u1 = tuple(layout["u1"])
    u2 = tuple(layout["u2"])

    # Fixed obstacle blocks the upper bypass. Removing it reopens a route that avoids
    # pushing, which is the intended D2+D16 coupling.
    upper_len = u2[1] - u1[1] - 1
    if upper_len < 1:
        return None
    fixed_len = 1 if upper_len == 1 else random.randint(1, min(2, upper_len))
    fixed_start_c = random.randint(u1[1] + 1, u2[1] - fixed_len)
    fixed = {(u1[0], c) for c in range(fixed_start_c, fixed_start_c + fixed_len)}

    push = set()
    extras = set()

    def add_gate(gate: tuple[int, int], step: tuple[int, int], chain_len: int, pocket_len: int) -> bool:
        for k in range(1, pocket_len + 1):
            cell = (gate[0] + step[0] * k, gate[1] + step[1] * k)
            if not (0 <= cell[0] < size and 0 <= cell[1] < size):
                return False
            if cell in river:
                return False
            extras.add(cell)
        for k in range(chain_len):
            cell = (gate[0] + step[0] * k, gate[1] + step[1] * k)
            if not (0 <= cell[0] < size and 0 <= cell[1] < size):
                return False
            if k > 0 and cell in river:
                return False
            push.add(cell)
        return True

    avail1 = _available_pocket_len(size, l1, (1, 0), river)
    if avail1 < 1:
        return None
    chain1 = random.randint(1, min(2, avail1))
    if not add_gate(l1, (1, 0), chain1, chain1):
        return None

    use_gate2 = (not reachable) or (random.random() < 0.5)
    if use_gate2:
        avail2 = _available_pocket_len(size, l2, (0, 1), river)
        if avail2 < 1:
            return None
        if reachable:
            chain2 = random.randint(1, min(2, avail2))
            pocket2 = chain2
        else:
            chain2 = random.randint(1, min(2, avail2 + 1))
            pocket2 = chain2 - 1
        if not add_gate(l2, (0, 1), chain2, pocket2):
            return None

    if start in fixed or start in push or dest in fixed or dest in push:
        return None

    all_river = river | extras
    grid = np.full((size, size), _BG, dtype=int)
    for cell in all_river:
        grid[cell] = _RIVER
    for cell in fixed:
        grid[cell] = _FIXED
    for cell in push:
        grid[cell] = _PUSH
    grid[start] = _MOVER
    grid[dest] = _DEST
    return grid


def _baseline_no_fixed(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    g[g == _FIXED] = _RIVER
    return main(g)


def _baseline_no_push(grid: np.ndarray) -> np.ndarray:
    sim = _simulate_states(grid, allow_push=False)
    if sim["dest_state"] is not None:
        mover, pushes = sim["dest_state"]
        return _render_state(grid, mover, pushes, reached=True)
    mover, pushes = sim["best"]
    return _render_state(grid, mover, pushes, reached=False)


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict:
    in_push = set(_sorted_coords(inp == _PUSH))
    out_push = set(_sorted_coords(out == _PUSH))
    return {
        "push_count": len(in_push),
        "moved_push_cells": len(in_push.symmetric_difference(out_push)) // 2,
        "reached": bool(_find_single(out, _MOVER) == _find_single(inp, _DEST)),
        "fixed_count": int(np.count_nonzero(inp == _FIXED)),
        "no_fixed_same": bool(np.array_equal(out, _baseline_no_fixed(inp))),
        "no_push_same": bool(np.array_equal(out, _baseline_no_push(inp))),
    }


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(13, 20)
    reachable = random.random() < 0.5
    layout = _sample_layout(size)
    if layout is None:
        return None
    grid = _build_grid(layout, reachable=reachable)
    if grid is None:
        return None

    out = main(grid)
    metrics = _variant_metrics(grid, out)
    if metrics["moved_push_cells"] < 1:
        return None
    if metrics["no_fixed_same"]:
        return None
    if metrics["no_push_same"]:
        return None
    if metrics["reached"] != reachable:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 300 attempts")
