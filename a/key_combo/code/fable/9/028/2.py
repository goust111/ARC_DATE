from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
INF = 10**9


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> BBox:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _ring_mask(h: int, w: int) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def _frame_from_component(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> dict | None:
    if int(coords.shape[0]) < 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = True

    ring = _ring_mask(h, w)
    if np.any(patch & ~ring):
        return None
    if np.any(sub[1:-1, 1:-1] != bg):
        return None

    missing = np.argwhere(ring & ~patch)
    if len(missing) > 1:
        return None
    if int(patch.sum()) != int(ring.sum()) - len(missing):
        return None

    opening: Coord | None = None
    inner_start: Coord | None = None
    if len(missing) == 1:
        mr, mc = map(int, missing[0])
        is_corner = (mr in (0, h - 1)) and (mc in (0, w - 1))
        if is_corner:
            return None
        opening = (r0 + mr, c0 + mc)
        if mr == 0:
            inner_start = (r0 + 1, c0 + mc)
        elif mr == h - 1:
            inner_start = (r1 - 1, c0 + mc)
        elif mc == 0:
            inner_start = (r0 + mr, c0 + 1)
        else:
            inner_start = (r0 + mr, c1 - 1)

    interior = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)]
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "inner_start": inner_start,
        "capacity": max(0, (h - 2) * (w - 2)),
        "interior": interior,
        "border_coords": {tuple(map(int, xy)) for xy in coords.tolist()},
    }


def _bfs_dist(passable: np.ndarray, start: Coord, restrict_to: set[Coord] | None = None) -> np.ndarray:
    dist = np.full(passable.shape, INF, dtype=int)
    if not passable[start]:
        return dist
    allowed = None if restrict_to is None else set(restrict_to)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(passable.shape, nr, nc) or not passable[nr, nc]:
                continue
            if allowed is not None and (nr, nc) not in allowed:
                continue
            if dist[nr, nc] != INF:
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def _fill_order(frame: dict) -> list[Coord]:
    if frame["opening"] is None or frame["inner_start"] is None:
        return []
    interior = set(frame["interior"])
    passable = np.zeros((frame["bbox"][2] + 1, frame["bbox"][3] + 1), dtype=bool)
    for r, c in interior:
        passable[r, c] = True
    dist = _bfs_dist(passable, frame["inner_start"], restrict_to=interior)
    ordered = [(int(dist[r, c]), r, c) for r, c in interior]
    assert all(d < INF for d, _, _ in ordered), "frame interior must be reachable from the opening"
    ordered.sort()
    return [(r, c) for _, r, c in ordered]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    frames: list[dict] = []
    token_colors: list[int] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components(g == color)
        frame_candidates = [
            frame
            for coords in comps
            if (frame := _frame_from_component(g, color, coords, bg)) is not None
        ]
        if frame_candidates:
            assert len(frame_candidates) == 1, "each frame color must define exactly one hollow rectangle"
            assert len(comps) == 1, "frame colors may not contain extra detached pixels"
            frames.append(frame_candidates[0])
        else:
            token_colors.append(color)

    assert len(frames) >= 2, "expected at least two frames"
    assert len(token_colors) == 1, "expected exactly one scattered-token color"
    token_color = int(token_colors[0])
    token_coords = [tuple(map(int, xy)) for xy in np.argwhere(g == token_color).tolist()]

    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["color"]))
    for frame in frames:
        frame["fill_order"] = _fill_order(frame)

    wall_colors = {int(frame["color"]) for frame in frames}
    passable = ~np.isin(g, list(wall_colors))
    return {
        "grid": g,
        "bg": bg,
        "frames": frames,
        "token_color": token_color,
        "tokens": token_coords,
        "passable": passable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign scattered grains to the nearest reachable opening and pour them into frame interiors up to capacity.

    Input structure:
    - The background contains several one-cell-thick hollow rectangular frames of different colors.
    - Most frames have exactly one missing border cell as an opening; some frames are fully closed.
    - A single extra color forms scattered grain pixels outside the frames, and frame borders act as walls.

    Rule:
    - For each grain, compare 4-connected path distances through passable non-wall cells to every reachable opening; assign the grain to the uniquely nearest opening's frame.
    - Each open frame fills its interior from the cell just inside its opening, ordered by interior path distance from that entry cell, breaking ties by row then column.
    - If a frame receives more grains than interior capacity, only the closest assigned grains move; the rest stay in place, and every moved grain is erased from its source cell.

    Key insight:
    - Ownership depends on shortest paths around walls rather than straight-line proximity, and capacity truncation makes the grain ranking matter twice: once for frame assignment and again for deciding which sources disappear.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = scene["bg"]
    out = g.copy()

    open_frames = [frame for frame in scene["frames"] if frame["opening"] is not None]
    if not open_frames or not scene["tokens"]:
        return out

    openings = [frame["opening"] for frame in open_frames]
    assert all(opening is not None for opening in openings)
    dists = [_bfs_dist(scene["passable"], opening) for opening in openings]  # type: ignore[arg-type]

    assigned: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(open_frames))}
    for r, c in scene["tokens"]:
        finite = [(int(dist[r, c]), idx) for idx, dist in enumerate(dists) if int(dist[r, c]) < INF]
        if not finite:
            continue
        best = min(d for d, _ in finite)
        winners = [idx for d, idx in finite if d == best]
        assert len(winners) == 1, "each grain must have a unique nearest reachable opening"
        assigned[winners[0]].append((best, r, c))

    for frame_idx, frame in enumerate(open_frames):
        grains = sorted(assigned[frame_idx])
        fill_order = frame["fill_order"]
        count = min(len(grains), len(fill_order))
        for i in range(count):
            _, sr, sc = grains[i]
            dr, dc = fill_order[i]
            out[sr, sc] = bg
            out[dr, dc] = scene["token_color"]

    return out


GRID_ROWS = (18, 26)
GRID_COLS = (18, 26)
FRAME_COUNT_CHOICES = (2, 3, 3, 4)
FRAME_HW = (5, 8)
FRAME_GAP = 2
MAX_PLACEMENT_TRIES = 120
MAX_GENERATION_TRIES = 400


def _frame_patch(h: int, w: int, color: int, opening_side: str | None, opening_pos: int | None) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if opening_side is not None and opening_pos is not None:
        if opening_side == "top":
            patch[0, opening_pos] = 0
        elif opening_side == "bottom":
            patch[h - 1, opening_pos] = 0
        elif opening_side == "left":
            patch[opening_pos, 0] = 0
        else:
            patch[opening_pos, w - 1] = 0
    return patch


def _interior_cells(frame: dict) -> set[Coord]:
    r0, c0, r1, c1 = frame["bbox"]
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _boxes_far_enough(a: BBox, b: BBox, gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(
    h: int,
    w: int,
    bg: int,
    colors: list[int],
    frame_count: int,
    closed_count: int,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), bg, dtype=int)
    specs: list[dict] = []
    open_mask = [True] * frame_count
    if closed_count:
        for idx in random.sample(range(frame_count), closed_count):
            open_mask[idx] = False

    for idx in range(frame_count):
        fh = random.randint(*FRAME_HW)
        fw = random.randint(*FRAME_HW)
        opening_side = None
        opening_pos = None
        if open_mask[idx]:
            opening_side = random.choice(("top", "bottom", "left", "right"))
            opening_pos = random.randint(1, (fw - 2) if opening_side in ("top", "bottom") else (fh - 2))
        specs.append(
            {
                "color": int(colors[idx]),
                "h": fh,
                "w": fw,
                "opening_side": opening_side,
                "opening_pos": opening_pos,
            }
        )

    placed: list[dict] = []
    for spec in sorted(specs, key=lambda s: -(s["h"] * s["w"])):
        ok = False
        for _ in range(MAX_PLACEMENT_TRIES):
            r0 = random.randint(1, h - spec["h"] - 1)
            c0 = random.randint(1, w - spec["w"] - 1)
            bbox = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
            if all(_boxes_far_enough(bbox, other["bbox"], FRAME_GAP) for other in placed):
                patch = _frame_patch(spec["h"], spec["w"], spec["color"], spec["opening_side"], spec["opening_pos"])
                grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = patch
                spec = dict(spec)
                spec["bbox"] = bbox
                if spec["opening_side"] is None:
                    spec["opening"] = None
                    spec["inner_start"] = None
                elif spec["opening_side"] == "top":
                    spec["opening"] = (r0, c0 + spec["opening_pos"])
                    spec["inner_start"] = (r0 + 1, c0 + spec["opening_pos"])
                elif spec["opening_side"] == "bottom":
                    spec["opening"] = (r0 + spec["h"] - 1, c0 + spec["opening_pos"])
                    spec["inner_start"] = (r0 + spec["h"] - 2, c0 + spec["opening_pos"])
                elif spec["opening_side"] == "left":
                    spec["opening"] = (r0 + spec["opening_pos"], c0)
                    spec["inner_start"] = (r0 + spec["opening_pos"], c0 + 1)
                else:
                    spec["opening"] = (r0 + spec["opening_pos"], c0 + spec["w"] - 1)
                    spec["inner_start"] = (r0 + spec["opening_pos"], c0 + spec["w"] - 2)
                spec["capacity"] = (spec["h"] - 2) * (spec["w"] - 2)
                placed.append(spec)
                ok = True
                break
        if not ok:
            return None
    return grid, placed


def _distance_maps(shape: tuple[int, int], frames: list[dict]) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    wall_colors = [int(frame["color"]) for frame in frames]
    canvas = np.zeros(shape, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = _frame_patch(r1 - r0 + 1, c1 - c0 + 1, frame["color"], frame["opening_side"], frame["opening_pos"])
        canvas[r0 : r1 + 1, c0 : c1 + 1] = patch
    passable = ~np.isin(canvas, wall_colors)
    open_frames = [frame for frame in frames if frame["opening"] is not None]
    dists = [_bfs_dist(passable, frame["opening"]) for frame in open_frames]  # type: ignore[arg-type]
    return passable, dists, open_frames


def _exterior_cells(shape: tuple[int, int], frames: list[dict]) -> set[Coord]:
    exterior = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    for frame in frames:
        exterior -= _interior_cells(frame)
        if frame["opening"] is not None:
            exterior.discard(frame["opening"])
    return exterior


def _territories(shape: tuple[int, int], frames: list[dict]) -> tuple[dict[int, list[tuple[int, int, int]]], list[tuple[int, int, int]]]:
    passable, dists, open_frames = _distance_maps(shape, frames)
    exterior = _exterior_cells(shape, frames)

    territories: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(open_frames))}
    trap_candidates: list[tuple[int, int, int]] = []
    openings = [frame["opening"] for frame in open_frames]

    for r, c in sorted(exterior):
        if not passable[r, c]:
            continue
        finite = [(int(dist[r, c]), idx) for idx, dist in enumerate(dists) if int(dist[r, c]) < INF]
        if not finite:
            continue
        best = min(d for d, _ in finite)
        winners = [idx for d, idx in finite if d == best]
        if len(winners) != 1:
            continue
        win = winners[0]
        territories[win].append((best, r, c))

        naive = [(abs(r - opening[0]) + abs(c - opening[1]), idx) for idx, opening in enumerate(openings)]  # type: ignore[index]
        naive_best = min(d for d, _ in naive)
        naive_winners = [idx for d, idx in naive if d == naive_best]
        if len(naive_winners) == 1 and naive_winners[0] != win:
            trap_candidates.append((win, r, c))

    for cells in territories.values():
        cells.sort()
    return territories, trap_candidates


def _adjacent_to_closed(frames: list[dict], cell: Coord) -> bool:
    r, c = cell
    closed_border: set[Coord] = set()
    for frame in frames:
        if frame["opening"] is None:
            closed_border |= {
                (rr, cc)
                for rr in range(frame["bbox"][0], frame["bbox"][2] + 1)
                for cc in range(frame["bbox"][1], frame["bbox"][3] + 1)
                if rr in (frame["bbox"][0], frame["bbox"][2]) or cc in (frame["bbox"][1], frame["bbox"][3])
            }
    return any((nr, nc) in closed_border for nr, nc in _neighbors4(r, c))


def _pick_with_spacing(candidates: list[tuple[int, int, int]], need: int, occupied: set[Coord]) -> list[Coord] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[Coord] = []
    banned = set(occupied)
    for _dist, r, c in pool:
        if (r, c) in banned:
            continue
        if any((nr, nc) in banned for nr, nc in _neighbors4(r, c)):
            continue
        chosen.append((r, c))
        banned.add((r, c))
        if len(chosen) == need:
            return chosen
    return None


def _semantic_signature(grid: np.ndarray) -> dict:
    scene = _parse_scene(grid)
    open_frames = [frame for frame in scene["frames"] if frame["opening"] is not None]
    openings = [frame["opening"] for frame in open_frames]
    dists = [_bfs_dist(scene["passable"], opening) for opening in openings]  # type: ignore[arg-type]

    path_trap = False
    assignments: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(open_frames))}
    for r, c in scene["tokens"]:
        finite = [(int(dist[r, c]), idx) for idx, dist in enumerate(dists) if int(dist[r, c]) < INF]
        if not finite:
            continue
        best = min(d for d, _ in finite)
        winners = [idx for d, idx in finite if d == best]
        assert len(winners) == 1, "generator must avoid tie assignments"
        win = winners[0]
        assignments[win].append((best, r, c))

        naive = [(abs(r - opening[0]) + abs(c - opening[1]), idx) for idx, opening in enumerate(openings)]  # type: ignore[index]
        naive_best = min(d for d, _ in naive)
        naive_winners = [idx for d, idx in naive if d == naive_best]
        if len(naive_winners) == 1 and naive_winners[0] != win:
            path_trap = True

    overfull = any(len(assignments[i]) > open_frames[i]["capacity"] for i in range(len(open_frames)))
    out = main(grid)
    token_color = scene["token_color"]
    leftovers = sum(int(out[r, c] == token_color) for r, c in scene["tokens"])
    moved = len(scene["tokens"]) - leftovers
    return {
        "open_count": len(open_frames),
        "closed_count": len(scene["frames"]) - len(open_frames),
        "path_trap": path_trap,
        "overfull": overfull,
        "moved": moved,
        "leftovers": leftovers,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    closed_count = 0 if frame_count == 2 else random.choice((0, 1))

    palette = random.sample(range(1, 10), frame_count + 1)
    bg = 0
    token_color = int(palette[-1])

    placed = _place_frames(h, w, bg, palette[:-1], frame_count, closed_count)
    if placed is None:
        return None
    grid, frames = placed

    open_frames = [frame for frame in frames if frame["opening"] is not None]
    if len(open_frames) < 2:
        return None

    territories, trap_candidates = _territories((h, w), frames)
    if not trap_candidates:
        return None

    viable_overfull = [
        idx
        for idx, frame in enumerate(open_frames)
        if len(territories[idx]) >= frame["capacity"] + 2
    ]
    if not viable_overfull:
        return None
    over_idx = random.choice(viable_overfull)

    targets = [0] * len(open_frames)
    for idx, frame in enumerate(open_frames):
        territory_size = len(territories[idx])
        if territory_size == 0:
            return None
        if idx == over_idx:
            hi = min(territory_size, frame["capacity"] + 4)
            lo = frame["capacity"] + 1
            targets[idx] = random.randint(lo, hi)
        else:
            upper = min(territory_size, max(1, frame["capacity"] + random.choice((-1, 0, 1))))
            lower = 1 if idx < 2 else random.choice((0, 1))
            targets[idx] = random.randint(lower, upper)

    chosen: dict[int, list[Coord]] = {idx: [] for idx in range(len(open_frames))}
    occupied: set[Coord] = set()

    trap_pool = trap_candidates[:]
    if closed_count:
        random.shuffle(trap_pool)
        trap_pool.sort(key=lambda item: not _adjacent_to_closed(frames, (item[1], item[2])))
    trap_frame_idx, tr, tc = random.choice(trap_pool)
    chosen[trap_frame_idx].append((tr, tc))
    occupied.add((tr, tc))
    targets[trap_frame_idx] -= 1

    if closed_count:
        nearby_closed = [
            item for item in territories[trap_frame_idx]
            if _adjacent_to_closed(frames, (item[1], item[2])) and (item[1], item[2]) not in occupied
        ]
        if nearby_closed and targets[trap_frame_idx] > 0:
            picked_closed = _pick_with_spacing(nearby_closed, 1, occupied)
            if picked_closed is not None:
                rr, cc = picked_closed[0]
                chosen[trap_frame_idx].append((rr, cc))
                occupied.add((rr, cc))
                targets[trap_frame_idx] -= 1

    for idx in range(len(open_frames)):
        need = targets[idx]
        if need <= 0:
            continue
        picked = _pick_with_spacing(territories[idx], need, occupied)
        if picked is None:
            return None
        chosen[idx].extend(picked)
        occupied.update(picked)

    for cells in chosen.values():
        for r, c in cells:
            grid[r, c] = token_color

    sig = _semantic_signature(grid)
    if not sig["path_trap"] or not sig["overfull"]:
        return None
    if sig["moved"] <= 0 or sig["leftovers"] <= 0:
        return None
    if sig["open_count"] <= sig["closed_count"]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for open-capacity gathering")
