from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


WALL = 5
BG = 0
ROOM_H = 3
ROOM_W = 4
SOURCE_W = 2
MIN_OBJECTS = 5
MAX_OBJECTS = 8
MAX_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_dict(mask: np.ndarray, color: int) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(cells)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                comp_mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": cells,
                    "bbox": (r0, c0, r1, c1),
                    "mask": comp_mask,
                    "area": len(cells),
                }
            )
    return comps


def _all_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, WALL}):
        comps.extend(_component_dict(grid == color, color))
    return comps


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(np.all(comp["mask"]))


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1 + 1, c0 + c1 + 1


def _center_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rows = [int((r0 + r1) // 2)] if (r1 - r0 + 1) % 2 else [int((r0 + r1) // 2), int((r0 + r1) // 2 + 1)]
    cols = [int((c0 + c1) // 2)] if (c1 - c0 + 1) % 2 else [int((c0 + c1) // 2), int((c0 + c1) // 2 + 1)]
    return [(r, c) for r in rows for c in cols]


def _shortest_path_info(
    open_mask: np.ndarray, start_cells: list[tuple[int, int]], goal_cells: list[tuple[int, int]]
) -> tuple[int | None, int]:
    h, w = open_mask.shape
    q = deque(start_cells)
    dist = np.full((h, w), -1, dtype=int)
    count = np.zeros((h, w), dtype=np.uint8)
    for r, c in start_cells:
        dist[r, c] = 0
        count[r, c] = 1

    while q:
        cr, cc = q.popleft()
        nd = dist[cr, cc] + 1
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not open_mask[nr, nc]:
                continue
            if dist[nr, nc] < 0:
                dist[nr, nc] = nd
                count[nr, nc] = count[cr, cc]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                count[nr, nc] = min(2, int(count[nr, nc]) + int(count[cr, cc]))

    best_dist = None
    best_count = 0
    for gr, gc in goal_cells:
        cur_dist = int(dist[gr, gc])
        if cur_dist < 0:
            continue
        if best_dist is None or cur_dist < best_dist:
            best_dist = cur_dist
            best_count = int(count[gr, gc])
        elif cur_dist == best_dist:
            best_count = min(2, best_count + int(count[gr, gc]))
    return best_dist, best_count


def _draw_walls(open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    grid = np.full((h, w), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            if open_mask[r, c]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc]:
                    grid[r, c] = WALL
                    break
    return grid


def _stamp(out: np.ndarray, source: dict, target: dict) -> None:
    src_h = source["bbox"][2] - source["bbox"][0] + 1
    src_w = source["bbox"][3] - source["bbox"][1] + 1
    trg_r2, trg_c2 = _center2(target["bbox"])
    dst_r0 = (trg_r2 - src_h) // 2
    dst_c0 = (trg_c2 - src_w) // 2
    for rr in range(source["mask"].shape[0]):
        for cc in range(source["mask"].shape[1]):
            if source["mask"][rr, cc]:
                out[dst_r0 + rr, dst_c0 + cc] = int(source["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each repeated-color rectangle through the corridor maze to its nearest singleton rectangle and stamp it there.

    Input structure:
    - Black cells are the background floor and grey cells form one-cell-thick maze walls around narrow orthogonal corridors.
    - Several colored 4-connected solid rectangles sit in corridor rooms; exactly one non-black, non-grey color appears on multiple rectangles.
    - Every repeated-color rectangle has a unique shortest orthogonal path through the walkable corridor floor to a different singleton rectangle.

    Rule:
    - Count 4-connected colored rectangles by color; the color with more than one rectangle is the movable repeated color and all other colored rectangles are singleton targets.
    - For each repeated-color rectangle, measure corridor distance by the shortest orthogonal path between rectangle bbox centers while staying inside non-grey cells; choose the nearest singleton rectangle.
    - Process repeated rectangles from shorter target distance to longer: erase the source rectangle to black, then stamp the source shape in the target rectangle with centers aligned, overwriting only the stamped cells and leaving the rest of the target rectangle unchanged.

    Key insight:
    - The deciding signal is not geometric proximity in open space alone: repeated-color counting selects the movers, while the maze corridor topology determines which singleton is truly nearest and where the source shape must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    comps = _all_components(g)
    assert comps, "expected colored rectangles"
    assert all(_is_solid_rect(comp) for comp in comps), "expected solid rectangular objects"

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    repeated_colors = [color for color, items in by_color.items() if len(items) > 1]
    assert len(repeated_colors) == 1, "expected exactly one repeated color"
    repeated_color = repeated_colors[0]

    repeated = by_color[repeated_color]
    singles = [items[0] for color, items in by_color.items() if color != repeated_color]
    assert singles, "expected singleton targets"

    open_mask = g != WALL
    assignments: list[tuple[int, tuple[int, int, int, int], dict, dict]] = []
    used_targets: set[tuple[int, int, int, int]] = set()
    for src in repeated:
        best_dist = None
        best_target = None
        tie = False
        for trg in singles:
            dist, count = _shortest_path_info(open_mask, _center_cells(src["bbox"]), _center_cells(trg["bbox"]))
            assert dist is not None and count >= 1, "all source/target pairs must be connected"
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_target = trg
                tie = False
            elif dist == best_dist:
                tie = True
        assert not tie and best_target is not None
        key = tuple(best_target["bbox"])
        assert key not in used_targets, "repeated sources must map to distinct targets"
        used_targets.add(key)
        assignments.append((int(best_dist), src["bbox"], src, best_target))

    assignments.sort(key=lambda item: (item[0], item[1]))
    out = g.copy()
    for _dist, _bbox_key, src, trg in assignments:
        for r, c in src["coords"]:
            out[r, c] = BG
        _stamp(out, src, trg)
    return out


def _paint_repeated(grid: np.ndarray, room: dict, color: int) -> None:
    r0, c0, _r1, _c1 = room["bbox"]
    grid[r0 : r0 + ROOM_H, c0 + 1 : c0 + 3] = int(color)


def _paint_single(grid: np.ndarray, room: dict, color: int) -> None:
    r0, c0, _r1, _c1 = room["bbox"]
    grid[r0 : r0 + ROOM_H, c0 : c0 + ROOM_W] = int(color)


def _room_candidates(h: int, w: int) -> list[int]:
    return list(range(4, h - 4, 4))


def _build_structure() -> tuple[np.ndarray, list[dict]] | None:
    h = random.randint(22, 28)
    w = random.randint(24, 29)
    rows = _room_candidates(h, w)
    if len(rows) < MIN_OBJECTS:
        return None

    trunk_c = random.randint(10, w - 11)
    use_side_spine = random.random() < 0.8
    side_spine = None
    if use_side_spine:
        side = random.choice((-1, 1))
        pivot = random.choice(rows[1:-1])
        side_c = trunk_c + side * random.randint(5, 7)
        if not (5 <= side_c <= w - 6):
            return None
        lo_choices = [r for r in rows if r < pivot]
        hi_choices = [r for r in rows if r > pivot]
        if not lo_choices or not hi_choices:
            return None
        lo = random.choice(lo_choices)
        hi = random.choice(hi_choices)
        side_spine = {"side": side, "pivot": pivot, "lo": lo, "hi": hi, "col": side_c}

    open_mask = np.zeros((h, w), dtype=bool)
    open_mask[min(rows) : max(rows) + 1, trunk_c] = True
    if side_spine is not None:
        c0, c1 = sorted((trunk_c, side_spine["col"]))
        open_mask[side_spine["pivot"], c0 : c1 + 1] = True
        open_mask[side_spine["lo"] : side_spine["hi"] + 1, side_spine["col"]] = True

    specs: list[tuple[str, int, int, int, int]] = []
    for r in rows:
        for side in (-1, 1):
            if side_spine is not None and side == side_spine["side"] and side_spine["lo"] <= r <= side_spine["hi"]:
                continue
            specs.append(("trunk", r, side, trunk_c, random.randint(2, 4)))
    if side_spine is not None:
        for r in rows:
            if side_spine["lo"] <= r <= side_spine["hi"] and r != side_spine["pivot"]:
                specs.append(("side", r, side_spine["side"], side_spine["col"], random.randint(2, 4)))
    random.shuffle(specs)

    room_count = random.randint(MIN_OBJECTS, min(MAX_OBJECTS, len(specs)))
    rooms: list[dict] = []
    used_rows: set[int] = set()
    for branch_type, r, side, spine_c, branch_len in specs:
        if r in used_rows:
            continue
        if side == -1:
            c0 = spine_c - branch_len - ROOM_W
            corridor = slice(c0 + ROOM_W, spine_c)
        else:
            c0 = spine_c + branch_len + 1
            corridor = slice(spine_c + 1, c0)
        r0 = r - 1
        if not (1 <= r0 and r0 + ROOM_H < h - 1 and 1 <= c0 and c0 + ROOM_W < w - 1):
            continue

        room_mask = np.zeros_like(open_mask)
        room_mask[r0 : r0 + ROOM_H, c0 : c0 + ROOM_W] = True
        corridor_mask = np.zeros_like(open_mask)
        corridor_mask[r, corridor] = True
        add_mask = room_mask | corridor_mask
        overlap = add_mask & open_mask
        if np.any(overlap & ~(np.arange(w)[None, :] == spine_c)):
            continue

        open_mask |= add_mask
        used_rows.add(r)
        rooms.append(
            {
                "bbox": (r0, c0, r0 + ROOM_H - 1, c0 + ROOM_W - 1),
                "branch_type": branch_type,
                "side": side,
                "spine_c": spine_c,
                "row": r,
            }
        )
        if len(rooms) == room_count:
            break

    return (open_mask, rooms) if len(rooms) >= MIN_OBJECTS else None


def _choose_roles(open_mask: np.ndarray, rooms: list[dict]) -> tuple[list[int], list[int]] | None:
    centers = [_center_cells(room["bbox"]) for room in rooms]
    pair_info: dict[tuple[int, int], tuple[int | None, int]] = {}
    for i, src in enumerate(centers):
        for j, trg in enumerate(centers):
            if i == j:
                continue
            pair_info[(i, j)] = _shortest_path_info(open_mask, src, trg)

    room_ids = list(range(len(rooms)))
    rep_counts = [2]
    if len(rooms) >= 6:
        rep_counts.append(3)
    random.shuffle(rep_counts)
    for rep_count in rep_counts:
        combos = list(itertools.combinations(room_ids, rep_count))
        random.shuffle(combos)
        for repeated in combos:
            repeated = list(repeated)
            singles = [i for i in room_ids if i not in repeated]
            targets: list[int] = []
            ok = True
            for src_idx in repeated:
                best_dist = None
                best_target = None
                tie = False
                for trg_idx in singles:
                    dist, count = pair_info[(src_idx, trg_idx)]
                    if dist is None or count != 1:
                        ok = False
                        break
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_target = trg_idx
                        tie = False
                    elif dist == best_dist:
                        tie = True
                if not ok or tie or best_target is None:
                    ok = False
                    break
                targets.append(best_target)
            if ok and len(set(targets)) == len(targets):
                return repeated, targets
    return None


def _semantic_ok(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    comps_in = _all_components(inp)
    comps_out = _all_components(out)
    if not comps_in or not comps_out:
        return False

    colors_in = {int(comp["color"]) for comp in comps_in}
    colors_out = {int(comp["color"]) for comp in comps_out}
    if colors_in != colors_out:
        return False

    repeated_color = next(color for color in colors_in if sum(comp["color"] == color for comp in comps_in) > 1)
    in_rep = [comp for comp in comps_in if comp["color"] == repeated_color]
    out_rep = [comp for comp in comps_out if comp["color"] == repeated_color]
    if len(out_rep) != len(in_rep):
        return False

    return bool(np.any(inp != out))


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_structure()
        if built is None:
            continue
        open_mask, rooms = built
        role_choice = _choose_roles(open_mask, rooms)
        if role_choice is None:
            continue
        repeated_idx, target_idx = role_choice

        palette = [1, 2, 3, 4, 6, 7, 8, 9]
        repeated_color = random.choice(palette)
        singles_palette = [color for color in palette if color != repeated_color]
        if len(singles_palette) < len(rooms) - len(repeated_idx):
            continue
        single_colors = random.sample(singles_palette, len(rooms) - len(repeated_idx))

        grid = _draw_walls(open_mask)
        repeated_set = set(repeated_idx)
        single_iter = iter(single_colors)
        for idx, room in enumerate(rooms):
            if idx in repeated_set:
                _paint_repeated(grid, room, repeated_color)
            else:
                _paint_single(grid, room, next(single_iter))

        if max(grid.shape) > 30:
            continue
        if not _semantic_ok(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
