from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    out: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            if int(g[r, c]) == int(bg):
                seen[r, c] = True
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc]:
                        continue
                    if monochromatic:
                        good = int(g[nr, nc]) == color
                    else:
                        good = int(g[nr, nc]) != int(bg)
                    if good:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color if monochromatic else None,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _window_all_true(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = mask.shape
    if H < h or W < w:
        return np.zeros((0, 0), dtype=bool)
    ii = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    sums = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    return sums == int(h * w)


def _bfs_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    H, W = mask.shape
    sr, sc = start
    seen = np.zeros_like(mask, dtype=bool)
    if not (0 <= sr < H and 0 <= sc < W) or not bool(mask[sr, sc]):
        return seen
    q = deque([(int(sr), int(sc))])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return seen


def _degree_grid(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    deg = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue
            count = 0
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]):
                    count += 1
            deg[r, c] = int(count)
    return deg


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    solid_rects = []
    for comp in comps:
        box = comp["bbox"]
        h = bbox_h(box)
        w = bbox_w(box)
        if h == w:
            continue
        r0, c0, r1, c1 = box
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        if int(comp["area"]) == int(h * w) and bool((patch == color).all()):
            solid_rects.append(
                {
                    "color": color,
                    "bbox": box,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(h),
                    "w": int(w),
                }
            )

    groups: dict[tuple[int, int], list[dict]] = {}
    for rect in solid_rects:
        groups.setdefault((int(rect["h"]), int(rect["w"])), []).append(rect)
    rect_groups = [items for items in groups.values() if len(items) == 3]
    assert len(rect_groups) == 1
    rects = rect_groups[0]
    h = int(rects[0]["h"])
    w = int(rects[0]["w"])

    obj_colors = {int(r["color"]) for r in rects}
    counts = Counter(map(int, g.reshape(-1)))
    river_candidates = [col for col, _ in counts.most_common() if int(col) not in obj_colors and int(col) != bg]
    assert len(river_candidates) == 1
    river = int(river_candidates[0])

    traversable = (g == river)
    for rect in rects:
        traversable[rect["r0"] : rect["r0"] + h, rect["c0"] : rect["c0"] + w] = True

    feasible = _window_all_true(traversable, h, w)
    assert feasible.size > 0
    deg = _degree_grid(feasible)

    for rect in rects:
        assert bool(feasible[rect["r0"], rect["c0"]])

    obj_pos = [(int(r["r0"]), int(r["c0"])) for r in rects]
    obj_deg = [int(deg[r, c]) for r, c in obj_pos]
    mover_ids = [i for i, d in enumerate(obj_deg) if d > 1]
    assert len(mover_ids) == 1
    mover_id = int(mover_ids[0])

    reachable = _bfs_mask(feasible, obj_pos[mover_id])
    reachable_targets = [i for i in range(3) if i != mover_id and bool(reachable[obj_pos[i]])]
    assert len(reachable_targets) == 1
    target_id = int(reachable_targets[0])
    blocked_id = int([i for i in range(3) if i not in (mover_id, target_id)][0])
    assert not bool(reachable[obj_pos[blocked_id]])

    return {
        "bg": bg,
        "river": river,
        "rects": rects,
        "h": h,
        "w": w,
        "feasible": feasible,
        "degrees": deg,
        "mover_id": mover_id,
        "target_id": target_id,
        "blocked_id": blocked_id,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the solid rectangle through the winding channel and replace the only fully reachable branch-end target.

    Input structure:
    - The most common color is background, and one other color forms a connected channel made of alternating vertical and horizontal segments.
    - Three distinct non-channel colors each form the same solid MxN rectangle, with M != N.
    - One rectangle sits on the channel's start-side track, while the other two sit at the ends of the two branch arms; rectangle cells count as channel for movement.

    Rule:
    - Treat channel-color cells plus the three rectangle footprints as traversable.
    - A rectangle placement is legal only when the full MxN footprint lies inside that traversable region, so vertical motion tests the rectangle width and horizontal motion tests the rectangle height.
    - Exactly one of the two branch-end rectangles is reachable from the start-side rectangle through legal placements; move the start-side rectangle there, recolor its old location to the channel color, and leave the blocked branch target unchanged.

    Key insight:
    - The correct object is identified by the placement graph, not by color: it is the only same-sized rectangle that is not a dead-end placement, and branch reachability must be checked along the full sequence of alternating-axis corridor segments.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    rects = info["rects"]
    mover = rects[info["mover_id"]]
    target = rects[info["target_id"]]
    h = int(info["h"])
    w = int(info["w"])
    river = int(info["river"])

    out = g.copy()
    out[mover["r0"] : mover["r0"] + h, mover["c0"] : mover["c0"] + w] = river
    out[target["r0"] : target["r0"] + h, target["c0"] : target["c0"] + w] = int(mover["color"])
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(int(ar), int(c)) for c in range(int(ac), int(bc) + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(int(r), int(ac)) for r in range(int(ar), int(br) + step, step)]
    raise ValueError("segment must be axis-aligned")


def _polyline(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[str, list[tuple[int, int]]]]]:
    cells = [points[0]]
    segments: list[tuple[str, list[tuple[int, int]]]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment_cells(a, b)
        axis = "h" if a[0] == b[0] else "v"
        segments.append((axis, seg))
        cells.extend(seg[1:])
    return cells, segments


def _render_track(
    cells: set[tuple[int, int]],
    *,
    h: int,
    w: int,
    bg: int,
    river: int,
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    top, bottom, left, right = margins
    minr = min(rows)
    maxr = max(rows)
    minc = min(cols)
    maxc = max(cols)
    shift = (int(top - minr), int(left - minc))
    H = int(maxr - minr + h + top + bottom)
    W = int(maxc - minc + w + left + right)
    grid = np.full((H, W), int(bg), dtype=int)
    sr, sc = shift
    for r, c in cells:
        rr = int(r + sr)
        cc = int(c + sc)
        grid[rr : rr + h, cc : cc + w] = int(river)
    return grid, (int(sr), int(sc))


def _carve_blocker(
    grid: np.ndarray,
    seg: tuple[str, list[tuple[int, int]]],
    *,
    h: int,
    w: int,
    bg: int,
) -> bool:
    axis, cells = seg
    if len(cells) < 5:
        return False
    k0 = random.randint(1, len(cells) - 3)
    k1 = random.randint(k0, len(cells) - 2)
    if axis == "h":
        base_r = int(cells[0][0])
        row = int(base_r + (0 if random.random() < 0.5 else h - 1))
        x0 = int(cells[k0][1] + w // 2)
        x1 = int(cells[k1][1] + w // 2)
        grid[row, x0 : x1 + 1] = int(bg)
    else:
        base_c = int(cells[0][1])
        col = int(base_c + (0 if random.random() < 0.5 else w - 1))
        y0 = int(cells[k0][0] + h // 2)
        y1 = int(cells[k1][0] + h // 2)
        grid[y0 : y1 + 1, col] = int(bg)
    return True


def _apply_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    k = random.randint(0, 3)
    out = np.rot90(out, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _make_base_case() -> np.ndarray | None:
    h, w = random.choice([(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)])
    bg, river, mover_col, t1_col, t2_col = random.sample(list(range(10)), 5)

    trunk_h = random.randint(5, 8)
    trunk_v = random.randint(2, 4)
    start_dir = random.choice([-1, 1])
    mover_step = random.randint(1, trunk_h - 1)

    up1 = random.randint(3, 5)
    up2 = random.randint(4, 7)
    up3 = 0 if random.random() < 0.35 else random.randint(2, 4)

    dn1 = random.randint(3, 5)
    dn2 = random.randint(4, 7)
    dn3 = 0 if random.random() < 0.35 else random.randint(2, 4)

    junction = (0, 0)
    start_leaf = (int(start_dir * trunk_v), int(-trunk_h))
    start_turn = (0, int(-trunk_h))
    mover_pos = (0, int(-mover_step))

    up_points = [junction, (-up1, 0), (-up1, up2)]
    if up3 > 0:
        up_points.append((-(up1 + up3), up2))

    dn_points = [junction, (dn1, 0), (dn1, dn2)]
    if dn3 > 0:
        dn_points.append((dn1 + dn3, dn2))

    start_cells, start_segments = _polyline([start_leaf, start_turn, junction])
    up_cells, up_segments = _polyline(up_points)
    dn_cells, dn_segments = _polyline(dn_points)

    up_target = up_cells[-1]
    dn_target = dn_cells[-1]

    all_cells = set(start_cells) | set(up_cells) | set(dn_cells)
    margins = (
        random.randint(1, 3),
        random.randint(1, 3),
        random.randint(1, 3),
        random.randint(1, 3),
    )
    grid, shift = _render_track(all_cells, h=h, w=w, bg=bg, river=river, margins=margins)
    if max(grid.shape) > 30:
        return None

    sr, sc = shift

    def shift_pt(pt: tuple[int, int]) -> tuple[int, int]:
        return int(pt[0] + sr), int(pt[1] + sc)

    branches = [
        {"name": "up", "target": shift_pt(up_target), "segs": [(axis, [shift_pt(p) for p in seg]) for axis, seg in up_segments]},
        {"name": "dn", "target": shift_pt(dn_target), "segs": [(axis, [shift_pt(p) for p in seg]) for axis, seg in dn_segments]},
    ]
    blocked_branch = random.choice(branches)

    candidate_ids = [1]
    if len(blocked_branch["segs"]) >= 3:
        candidate_ids.append(2)
    chosen_seg = blocked_branch["segs"][random.choice(candidate_ids)]
    if not _carve_blocker(grid, chosen_seg, h=h, w=w, bg=bg):
        return None

    mover_r, mover_c = shift_pt(mover_pos)
    t1_r, t1_c = branches[0]["target"]
    t2_r, t2_c = branches[1]["target"]
    grid[mover_r : mover_r + h, mover_c : mover_c + w] = int(mover_col)
    grid[t1_r : t1_r + h, t1_c : t1_c + w] = int(t1_col)
    grid[t2_r : t2_r + h, t2_c : t2_c + w] = int(t2_col)

    grid = _apply_symmetry(grid)
    if max(grid.shape) > 30:
        return None

    try:
        info = _analyze_grid(grid)
        rects = info["rects"]
        mover = rects[info["mover_id"]]
        target = rects[info["target_id"]]
        blocked = rects[info["blocked_id"]]
    except AssertionError:
        return None

    if int(mover["color"]) != int(mover_col):
        return None
    if int(target["color"]) not in (int(t1_col), int(t2_col)):
        return None
    if int(blocked["color"]) == int(target["color"]):
        return None
    mover_deg = int(info["degrees"][mover["r0"], mover["c0"]])
    target_deg = int(info["degrees"][target["r0"], target["c0"]])
    blocked_deg = int(info["degrees"][blocked["r0"], blocked["c0"]])
    if mover_deg <= 1 or target_deg != 1 or blocked_deg != 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if int(out[mover["r0"], mover["c0"]]) != int(info["river"]):
        return None
    if int(out[target["r0"], target["c0"]]) != int(mover_col):
        return None
    if int(out[blocked["r0"], blocked["c0"]]) != int(blocked["color"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_base_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arc input")
