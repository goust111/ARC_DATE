from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
BLUE = 1
GREY = 5
SPECIAL_COLORS = (2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield int(r + dr), int(c + dc)


def _component_from(grid: np.ndarray, start_cells: list[tuple[int, int]]) -> np.ndarray:
    mask = grid != BG
    out = np.zeros_like(mask, dtype=bool)
    q = deque()
    for r, c in start_cells:
        out[r, c] = True
        q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(grid.shape, rr, cc) and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


def _find_blue_square(grid: np.ndarray) -> tuple[int, int]:
    blue = np.argwhere(grid == BLUE)
    assert len(blue) == 4, "expected exactly one 2x2 blue mover"
    r0, c0 = blue.min(axis=0)
    r1, c1 = blue.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (2, 2), "blue mover must be 2x2"
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == BLUE), "blue mover must be solid"
    return int(r0), int(c0)


def _feasible_positions(traversable: np.ndarray) -> set[tuple[int, int]]:
    h, w = traversable.shape
    pos: set[tuple[int, int]] = set()
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(traversable[r : r + 2, c : c + 2].all()):
                pos.add((int(r), int(c)))
    return pos


def _coverage_map(positions: set[tuple[int, int]], shape: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cover: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            hits = []
            for pr, pc in positions:
                if pr <= r <= pr + 1 and pc <= c <= pc + 1:
                    hits.append((int(pr), int(pc)))
            if hits:
                cover[(int(r), int(c))] = hits
    return cover


def _placement_neighbors(pos: tuple[int, int], positions: set[tuple[int, int]]):
    r, c = pos
    for dr, dc in DIR4:
        nxt = (int(r + dr), int(c + dc))
        if nxt in positions:
            yield nxt


def _reachable_positions(positions: set[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _placement_neighbors(cur, positions):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _path_count(positions: set[tuple[int, int]], start: tuple[int, int], target: tuple[int, int], limit: int = 2) -> int:
    count = 0
    seen: set[tuple[int, int]] = set()

    def dfs(cur: tuple[int, int]) -> None:
        nonlocal count
        if count >= limit:
            return
        if cur == target:
            count += 1
            return
        seen.add(cur)
        for nxt in _placement_neighbors(cur, positions):
            if nxt not in seen:
                dfs(nxt)
        seen.remove(cur)

    dfs(start)
    return int(count)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    start = _find_blue_square(g)
    start_cells = [(start[0] + dr, start[1] + dc) for dr in (0, 1) for dc in (0, 1)]
    corridor = _component_from(g, start_cells)

    counts = Counter(map(int, g.flat))
    blue_count = int(counts[BLUE])
    all_open_positions = _feasible_positions(corridor)
    assert start in all_open_positions, "start position must be feasible"
    cover = _coverage_map(all_open_positions, g.shape)

    special_cells = []
    for r, c in np.argwhere(corridor):
        color = int(g[r, c])
        if color not in (BLUE, GREY):
            key = (int(r), int(c))
            assert key in cover, "corridor special cell must lie inside some 2x2 placement"
            special_cells.append(key)

    candidates = []
    gates = []
    for cell in special_cells:
        freq = len(cover[cell])
        if freq == 1:
            candidates.append(cell)
        else:
            gates.append(cell)

    assert 3 <= len(candidates) <= 4, "expected 3-4 candidate markers"
    assert 2 <= len(gates) <= 4, "expected 2-4 gate markers"

    target_candidates = [cell for cell in candidates if int(counts[int(g[cell])]) == 1]
    assert len(target_candidates) == 1, "expected exactly one frequency-unique candidate"
    target_cell = target_candidates[0]

    traversable = corridor.copy()
    open_gates = []
    closed_gates = []
    for cell in gates:
        if int(counts[int(g[cell])]) > blue_count:
            open_gates.append(cell)
        else:
            closed_gates.append(cell)
            traversable[cell] = False

    actual_positions = _feasible_positions(traversable)
    assert start in actual_positions, "start must remain feasible after gate closure"
    target_positions = [pos for pos in actual_positions if pos[0] <= target_cell[0] <= pos[0] + 1 and pos[1] <= target_cell[1] <= pos[1] + 1]
    assert len(target_positions) == 1, "target cell must determine a unique final placement"
    target_pos = target_positions[0]
    reachable = _reachable_positions(actual_positions, start)
    assert target_pos in reachable, "target must be reachable"

    return {
        "grid": g,
        "counts": counts,
        "start": start,
        "corridor": corridor,
        "all_open_positions": all_open_positions,
        "cover": cover,
        "candidates": candidates,
        "gates": gates,
        "target_cell": target_cell,
        "target_pos": target_pos,
        "actual_positions": actual_positions,
        "open_gates": open_gates,
        "closed_gates": closed_gates,
        "reachable": reachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the 2x2 blue block through the grey channel toward the uniquely frequent candidate while gate colors open or close by global frequency comparison.

    Input structure:
    - Black is background. One connected non-black component contains a grey winding corridor and a solid 2x2 blue mover.
    - Inside that corridor, several singleton non-grey colors act as corridor specials; some are leaf-end candidate markers and some sit on chokepoints as gates.
    - Extra isolated pixels elsewhere repeat some of those special colors so global color frequencies differ.

    Rule:
    - Count every non-background color in the whole grid.
    - Among corridor special cells, the markers covered by exactly one possible 2x2 placement are the target candidates; the unique one whose color appears exactly once globally is the destination.
    - Corridor special cells covered by multiple possible 2x2 placements are gates. A gate is passable iff its global color frequency is strictly greater than the blue frequency; otherwise that cell becomes blocked.
    - Slide the blue 2x2 block through the feasible 2x2 placements to the unique reachable placement that covers the destination marker, overwrite that 2x2 area with blue, and recolor the original blue 2x2 area to grey.

    Key insight:
    - The same global frequency table drives two different inferences at once: singleton-frequency among leaf markers chooses where to go, while greater-than-blue frequency on chokepoints chooses which corridor placements remain usable.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    out = g.copy()
    sr, sc = info["start"]
    tr, tc = info["target_pos"]
    out[sr : sr + 2, sc : sc + 2] = GREY
    out[tr : tr + 2, tc : tc + 2] = BLUE
    return out


def _normalize_points(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _c in points)
    min_c = min(c for _r, c in points)
    norm = [(int(r - min_r), int(c - min_c)) for r, c in points]
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    return norm, int(h), int(w)


def _transform_points(points: list[tuple[int, int]], rot_k: int, flip_lr: bool) -> list[tuple[int, int]]:
    pts, h, w = _normalize_points(points)
    cur = pts
    hh, ww = h, w
    for _ in range(rot_k % 4):
        cur = [(int(ww - 1 - c), int(r)) for r, c in cur]
        hh, ww = ww, hh
    if flip_lr:
        cur = [(int(r), int(ww - 1 - c)) for r, c in cur]
    cur, _hh2, _ww2 = _normalize_points(cur)
    return cur


def _build_layout() -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    perimeter = {(0, c) for c in range(w + 1)}
    perimeter |= {(h, c) for c in range(w + 1)}
    perimeter |= {(r, 0) for r in range(h + 1)}
    perimeter |= {(r, w) for r in range(h + 1)}

    start_len = random.randint(1, 2)
    target_len = random.randint(1, 2)
    right_len = random.randint(1, 2)
    left_len = random.randint(1, 2)
    top_len = random.randint(1, 2)

    start_anchor = (0, 0)
    target_anchor = (h, w)
    right_anchor = (random.randint(1, h - 1), w)
    left_anchor = (random.randint(1, h - 1), 0)
    top_anchor = (0, random.randint(1, w - 1))

    def branch(anchor: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
        ar, ac = anchor
        dr, dc = direction
        return [(int(ar + dr * k), int(ac + dc * k)) for k in range(1, int(length) + 1)]

    nodes = set(perimeter)
    nodes.update(branch(start_anchor, (-1, 0), start_len))
    nodes.update(branch(target_anchor, (1, 0), target_len))

    decoys = [branch(right_anchor, (0, 1), right_len)[-1], branch(left_anchor, (0, -1), left_len)[-1]]
    nodes.update(branch(right_anchor, (0, 1), right_len))
    nodes.update(branch(left_anchor, (0, -1), left_len))
    if random.random() < 0.5:
        nodes.update(branch(top_anchor, (-1, 0), top_len))
        decoys.append(branch(top_anchor, (-1, 0), top_len)[-1])

    start = branch(start_anchor, (-1, 0), start_len)[-1]
    target = branch(target_anchor, (1, 0), target_len)[-1]

    pts = list(nodes)
    tagged = pts + [start, target, *decoys]
    rot_k = random.randrange(4)
    flip_lr = bool(random.getrandbits(1))
    tagged = _transform_points(tagged, rot_k, flip_lr)
    transformed = set(tagged[: len(pts)])
    idx = len(pts)
    start = tuple(tagged[idx])
    target = tuple(tagged[idx + 1])
    decoys = [tuple(p) for p in tagged[idx + 2 :]]
    return transformed, start, target, decoys


def _render_corridor(nodes: set[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _c in nodes) + 2
    w = max(c for _r, c in nodes) + 2
    mask = np.zeros((h, w), dtype=bool)
    for r, c in nodes:
        mask[r : r + 2, c : c + 2] = True
    return mask


def _degree(pos: tuple[int, int], positions: set[tuple[int, int]]) -> int:
    return sum(1 for _ in _placement_neighbors(pos, positions))


def _sample_leaf_cell(
    leaf: tuple[int, int],
    cover: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int] | None:
    cells = []
    lr, lc = leaf
    for r in range(lr, lr + 2):
        for c in range(lc, lc + 2):
            if (r, c) in cover and len(cover[(r, c)]) == 1:
                cells.append((int(r), int(c)))
    if not cells:
        return None
    return random.choice(cells)


def _find_gate_pair(
    positions: set[tuple[int, int]],
    cover: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    target: tuple[int, int],
    decoys: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if _path_count(positions, start, target, limit=3) < 2:
        return None

    gate_cells = []
    for cell, hits in cover.items():
        if len(hits) < 2:
            continue
        if start in hits or target in hits:
            continue
        gate_cells.append(cell)
    random.shuffle(gate_cells)

    for closed_cell in gate_cells:
        closed_hits = set(cover[closed_cell])
        pos_after_closed = positions - closed_hits
        if start not in pos_after_closed or target not in pos_after_closed:
            continue
        if _path_count(pos_after_closed, start, target, limit=2) != 1:
            continue
        reachable_after_closed = _reachable_positions(pos_after_closed, start)
        if not any(decoy in reachable_after_closed for decoy in decoys):
            continue

        open_options = gate_cells[:]
        random.shuffle(open_options)
        for open_cell in open_options:
            if open_cell == closed_cell:
                continue
            open_hits = set(cover[open_cell])
            if start in open_hits:
                continue
            pos_after_both = pos_after_closed - open_hits
            if start not in pos_after_both:
                continue
            if target not in _reachable_positions(pos_after_both, start):
                return closed_cell, open_cell
    return None


def _place_isolated_extras(grid: np.ndarray, color_plan: dict[int, int]) -> bool:
    slots = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            ok = True
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
                    ok = False
                    break
            if ok:
                slots.append((int(r), int(c)))
    random.shuffle(slots)

    needed = sum(color_plan.values())
    if len(slots) < needed:
        return False

    chosen = []
    while slots and len(chosen) < needed:
        cell = slots.pop()
        chosen.append(cell)
        slots = [
            other
            for other in slots
            if abs(other[0] - cell[0]) + abs(other[1] - cell[1]) > 1
        ]
    if len(chosen) < needed:
        return False

    idx = 0
    for color, extra_count in color_plan.items():
        for _ in range(extra_count):
            r, c = chosen[idx]
            idx += 1
            grid[r, c] = int(color)
    return True


def _generate_one() -> np.ndarray | None:
    nodes, start, target, decoys = _build_layout()
    corridor_mask = _render_corridor(nodes)
    if _feasible_positions(corridor_mask) != nodes:
        return None
    if _degree(start, nodes) != 1 or _degree(target, nodes) != 1:
        return None
    if any(_degree(decoy, nodes) != 1 for decoy in decoys):
        return None

    cover = _coverage_map(nodes, corridor_mask.shape)
    target_cell = _sample_leaf_cell(target, cover)
    decoy_cells = [_sample_leaf_cell(decoy, cover) for decoy in decoys]
    if target_cell is None or any(cell is None for cell in decoy_cells):
        return None
    all_candidate_cells = [target_cell, *decoy_cells]
    if len(set(all_candidate_cells)) != len(all_candidate_cells):
        return None

    gate_pair = _find_gate_pair(nodes, cover, start, target, decoys)
    if gate_pair is None:
        return None
    closed_gate_cell, open_gate_cell = gate_pair
    if closed_gate_cell in all_candidate_cells or open_gate_cell in all_candidate_cells:
        return None

    candidate_count = 1 + len(decoys)
    gate_count = 2
    palette = random.sample(SPECIAL_COLORS, candidate_count + gate_count)
    candidate_colors = palette[:candidate_count]
    gate_colors = palette[candidate_count:]
    color_by_cell = {
        target_cell: candidate_colors[0],
        open_gate_cell: gate_colors[0],
        closed_gate_cell: gate_colors[1],
    }
    for cell, color in zip(decoy_cells, candidate_colors[1:]):
        color_by_cell[cell] = int(color)

    inner_h, inner_w = corridor_mask.shape
    canvas_h = random.randint(max(15, inner_h + 6), min(24, inner_h + 12))
    canvas_w = random.randint(max(15, inner_w + 6), min(24, inner_w + 12))
    if canvas_h > 30 or canvas_w > 30:
        return None
    row_off = random.randint(2, canvas_h - inner_h - 2)
    col_off = random.randint(2, canvas_w - inner_w - 2)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    grid[row_off : row_off + inner_h, col_off : col_off + inner_w][corridor_mask] = GREY
    sr, sc = start
    grid[row_off + sr : row_off + sr + 2, col_off + sc : col_off + sc + 2] = BLUE

    for (r, c), color in color_by_cell.items():
        grid[row_off + r, col_off + c] = int(color)

    extras: dict[int, int] = {}
    extras[int(candidate_colors[0])] = 0
    for color in candidate_colors[1:]:
        extras[int(color)] = random.randint(1, 2)
    extras[int(gate_colors[0])] = random.randint(4, 6)
    extras[int(gate_colors[1])] = random.randint(1, 3)
    if not _place_isolated_extras(grid, extras):
        return None

    info = _analyze_grid(grid)
    if tuple(info["target_cell"]) != (row_off + target_cell[0], col_off + target_cell[1]):
        return None
    if len(info["open_gates"]) < 1 or len(info["closed_gates"]) < 1:
        return None
    if len(info["candidates"]) != candidate_count or len(info["gates"]) != gate_count:
        return None
    if _path_count(info["all_open_positions"], info["start"], info["target_pos"], limit=3) < 2:
        return None
    if _path_count(info["actual_positions"], info["start"], info["target_pos"], limit=2) != 1:
        return None
    if not any(decoy in info["reachable"] for decoy in [(row_off + r, col_off + c) for r, c in decoys]):
        return None

    critical_open = False
    for gate in info["open_gates"]:
        reduced = info["actual_positions"] - set(info["cover"][gate])
        if info["start"] in reduced and info["target_pos"] not in _reachable_positions(reduced, info["start"]):
            critical_open = True
            break
    if not critical_open:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-gated channel instance")
