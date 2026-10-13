from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAGENTA = 6
ORANGE = 7

DIRS = (
    (0, -1),  # left
    (0, 1),   # right
    (-1, 0),  # up
    (1, 0),   # down
)
DIR_ORDER = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    if not mask[start]:
        return set()
    out: set[tuple[int, int]] = set()
    q = deque([start])
    out.add(start)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in out:
                out.add((nr, nc))
                q.append((nr, nc))
    return out


def _touches_border(cells: set[tuple[int, int]] | list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _pick_slide(
    grid: np.ndarray,
    pos: tuple[int, int],
    *,
    strict: bool,
    require_move: bool,
):
    r, c = pos
    hits: list[tuple[int, int, int, int, int, tuple[int, int]]] = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        dist = 1
        while _in_bounds(grid.shape, rr, cc):
            value = int(grid[rr, cc])
            if value != BG:
                if value == BLUE:
                    axis_rank = 0 if dr == 0 else 1
                    hits.append((dist, axis_rank, DIR_ORDER[(dr, dc)], dr, dc, (rr, cc)))
                break
            rr += dr
            cc += dc
            dist += 1

    if not hits:
        return None

    hits.sort()
    best_dist, _best_axis, _best_dir, dr, dc, (wr, wc) = hits[0]
    if strict and sum(hit[0] == best_dist for hit in hits) > 1:
        return None

    stop = (wr - dr, wc - dc)
    if require_move and stop == pos:
        return None
    return {
        "source": pos,
        "landing": stop,
        "direction": (dr, dc),
        "wall": (wr, wc),
        "distance": best_dist,
    }


def _flood_reachable(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    reach = np.zeros_like(grid, dtype=bool)
    if int(grid[start]) != BG:
        return reach
    q = deque([start])
    reach[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and not reach[nr, nc] and int(grid[nr, nc]) == BG:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _solve_details(grid: np.ndarray, *, strict_reds: bool) -> dict | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    marker_coords = [tuple(map(int, p)) for p in np.argwhere(g == MAGENTA)]
    if len(marker_coords) != 1:
        return None
    marker = marker_coords[0]
    if marker not in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        return None

    red_sources = [tuple(map(int, p)) for p in np.argwhere(g == RED)]
    moves = []
    landings = set()
    for src in red_sources:
        move = _pick_slide(g, src, strict=strict_reds, require_move=False)
        if move is None:
            return None
        if move["landing"] in landings:
            return None
        landings.add(move["landing"])
        moves.append(move)

    after_slide = g.copy()
    for src in red_sources:
        after_slide[src] = BG
    for move in moves:
        after_slide[move["landing"]] = BLUE

    blue_mask = after_slide == BLUE
    keep_mask = np.zeros_like(blue_mask)
    for comp in _components(blue_mask):
        if _touches_border(comp, g.shape):
            for r, c in comp:
                keep_mask[r, c] = True

    after_erase = after_slide.copy()
    after_erase[(after_slide == BLUE) & (~keep_mask)] = BG

    flood_grid = after_erase.copy()
    flood_grid[marker] = BG
    reachable = _flood_reachable(flood_grid, marker)

    out = flood_grid.copy()
    out[(flood_grid == BG) & (~reachable)] = ORANGE

    return {
        "red_sources": red_sources,
        "moves": {move["source"]: move for move in moves},
        "after_slide": after_slide,
        "after_erase": after_erase,
        "reachable": reachable,
        "output": out,
    }


def _transform_grid(grid: np.ndarray, rotations: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _fragment_shape() -> list[tuple[int, int]]:
    size = random.randint(2, 5)
    kind = random.choice(("h", "v", "l"))
    if kind == "h":
        return [(0, c) for c in range(size)]
    if kind == "v":
        return [(r, 0) for r in range(size)]
    if size == 2:
        return [(0, 0), (1, 0)]
    arm = random.randint(2, size - 1)
    tail = size - arm + 1
    coords = {(r, 0) for r in range(arm)}
    coords |= {(arm - 1, c) for c in range(tail)}
    return sorted(coords)


def _can_place_with_padding(grid: np.ndarray, coords: list[tuple[int, int]], padding: int = 1) -> bool:
    coord_set = set(coords)
    h, w = grid.shape
    for r, c in coords:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in coords:
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if not _in_bounds(grid.shape, rr, cc):
                    continue
                if (rr, cc) in coord_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _place_doomed_fragments(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
) -> list[set[tuple[int, int]]] | None:
    doomed: list[set[tuple[int, int]]] = []
    want = 1 + (1 if random.random() < 0.45 else 0)
    h, w = grid.shape
    for _ in range(want):
        placed = False
        for _try in range(120):
            shape = _fragment_shape()
            height = max(r for r, _ in shape) + 1
            width = max(c for _, c in shape) + 1
            r0 = random.randint(1, h - height - 2)
            c0 = random.randint(1, w - width - 2)
            coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(cell in reserved for cell in coords):
                continue
            if not _can_place_with_padding(grid, coords, padding=1):
                continue
            for r, c in coords:
                grid[r, c] = BLUE
            doomed.append(set(coords))
            placed = True
            break
        if not placed:
            return None
    return doomed


def _all_valid_red_sources(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BG:
                continue
            move = _pick_slide(grid, (r, c), strict=True, require_move=True)
            if move is None:
                continue
            out.append(((r, c), move["landing"]))
    random.shuffle(out)
    return out


def _validate_generated_case(
    grid: np.ndarray,
    *,
    rescue_fragment: set[tuple[int, int]],
    doomed_fragments: list[set[tuple[int, int]]],
    room_cells: set[tuple[int, int]],
    bridge_source: tuple[int, int],
) -> dict | None:
    details = _solve_details(grid, strict_reds=True)
    if details is None:
        return None

    rescue_seed = next(iter(rescue_fragment))
    after_slide_comp = _component_from(details["after_slide"] == BLUE, rescue_seed)
    after_erase_comp = _component_from(details["after_erase"] == BLUE, rescue_seed)
    if not after_slide_comp or not _touches_border(after_slide_comp, grid.shape):
        return None
    if not after_erase_comp or not _touches_border(after_erase_comp, grid.shape):
        return None
    if not all(int(details["after_erase"][r, c]) == BLUE for r, c in rescue_fragment):
        return None

    for doomed in doomed_fragments:
        if any(int(details["after_erase"][r, c]) == BLUE for r, c in doomed):
            return None

    room_orange = {cell for cell in room_cells if int(details["output"][cell]) == ORANGE}
    if len(room_orange) < max(2, len(room_cells) // 6):
        return None

    if int(grid[bridge_source]) != RED:
        return None
    bridge_move = details["moves"].get(bridge_source)
    if bridge_move is None:
        return None

    without_bridge = np.asarray(grid, dtype=int).copy()
    without_bridge[bridge_source] = BG
    without_details = _solve_details(without_bridge, strict_reds=True)
    if without_details is None:
        return None

    if any(int(without_details["after_erase"][r, c]) == BLUE for r, c in rescue_fragment):
        return None
    if any(int(without_details["output"][cell]) == ORANGE for cell in room_cells):
        return None
    if np.array_equal(details["output"], without_details["output"]):
        return None

    return details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rescue doorway fragments by sliding red bridge markers before border-connected flood partitioning.

    Input structure:
    - The grid is black except for blue wall pixels, 2-4 red singleton markers, and one magenta corner marker.
    - Some blue components touch the grid border and act as structural walls; other 4-connected blue fragments sit fully inside the grid.
    - Each red marker has a clear horizontal or vertical sightline to at least one blue wall pixel.

    Rule:
    - For every red marker, find the nearest blue wall along a cardinal sightline; choose the shortest such sightline, breaking horizontal-vs-vertical ties in favor of horizontal.
    - Slide the red marker along that direction until it stops on the last background cell before the wall, then recolor the stopped marker blue so it becomes part of the wall set.
    - Recompute blue 4-connected components and erase every blue component that does not touch the grid border.
    - Starting from the magenta corner, flood-fill through black background cells only, with the surviving blue walls blocking the flood.
    - Recolor every black cell not reached by that flood to orange, and erase the magenta marker back to black.

    Key insight:
    - The red markers matter only because their sliding can rescue an otherwise decorative interior wall fragment, and that rescued fragment can survive the border-connectivity pruning long enough to change which background region the corner flood cannot reach.
    """
    details = _solve_details(np.asarray(grid, dtype=int), strict_reds=False)
    if details is None:
        raise ValueError("invalid puzzle instance")
    return details["output"]


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        grid = np.zeros((h, w), dtype=int)

        top = random.randint(2, 3)
        bottom = random.randint(top + 7, min(h - 4, top + 10))
        left = random.randint(2, 4)
        right = random.randint(max(left + 7, w - 7), w - 3)

        frag_len = random.randint(2, min(5, bottom - top - 3))
        door_top = random.randint(top + 2, bottom - frag_len - 1)
        door_bottom = door_top + frag_len - 1
        frag_col = right - 1
        bridge_target = (door_bottom + 1, frag_col)

        # Border-connected room with a door gap on the right side.
        grid[: bottom + 1, left] = BLUE
        grid[top, left : right + 1] = BLUE
        grid[bottom, left : right + 1] = BLUE
        grid[top : door_top, right] = BLUE
        grid[door_bottom + 1 : bottom + 1, right] = BLUE

        rescue_fragment = {(r, frag_col) for r in range(door_top, door_bottom + 1)}
        for r, c in rescue_fragment:
            grid[r, c] = BLUE

        marker = (h - 1, w - 1)
        if int(grid[marker]) != BG:
            continue

        reserved = set()
        for r in range(top - 1, bottom + 2):
            for c in range(left - 1, right + 2):
                if _in_bounds(grid.shape, r, c):
                    reserved.add((r, c))
        reserved.add(marker)

        doomed_fragments = _place_doomed_fragments(grid, reserved)
        if doomed_fragments is None:
            continue

        grid[marker] = MAGENTA

        room_cells = {
            (r, c)
            for r in range(top + 1, bottom)
            for c in range(left + 1, frag_col)
            if int(grid[r, c]) == BG
        }
        if len(room_cells) < 6:
            continue

        candidates = _all_valid_red_sources(grid)
        bridge_sources = [src for src, landing in candidates if landing == bridge_target]
        if not bridge_sources:
            continue
        random.shuffle(bridge_sources)

        for bridge_source in bridge_sources:
            trial = grid.copy()
            trial[bridge_source] = RED
            if _validate_generated_case(
                trial,
                rescue_fragment=rescue_fragment,
                doomed_fragments=doomed_fragments,
                room_cells=room_cells,
                bridge_source=bridge_source,
            ) is None:
                continue

            desired_reds = random.randint(2, 4)
            placed = [bridge_source]
            current = trial

            while len(placed) < desired_reds:
                current_details = _solve_details(current, strict_reds=True)
                if current_details is None:
                    break

                viable: list[tuple[int, int]] = []
                for src, _landing in _all_valid_red_sources(current):
                    candidate = current.copy()
                    candidate[src] = RED
                    cand_details = _validate_generated_case(
                        candidate,
                        rescue_fragment=rescue_fragment,
                        doomed_fragments=doomed_fragments,
                        room_cells=room_cells,
                        bridge_source=bridge_source,
                    )
                    if cand_details is None:
                        continue
                    landing = cand_details["moves"][src]["landing"]
                    if int(cand_details["after_erase"][landing]) != BLUE:
                        continue
                    if int(current_details["after_erase"][landing]) == BLUE:
                        continue
                    viable.append(src)
                if not viable:
                    break
                src = random.choice(viable)
                current[src] = RED
                placed.append(src)

            if len(placed) < 2:
                continue

            if _validate_generated_case(
                current,
                rescue_fragment=rescue_fragment,
                doomed_fragments=doomed_fragments,
                room_cells=room_cells,
                bridge_source=bridge_source,
            ) is None:
                continue

            rotations = random.randrange(4)
            flip_lr = random.random() < 0.5
            final = _transform_grid(current, rotations, flip_lr)

            # Extra final check on the transformed instance.
            final_details = _solve_details(final, strict_reds=True)
            if final_details is None:
                continue
            return final

    raise RuntimeError("failed to generate a valid puzzle-10 input")
