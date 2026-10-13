from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 8
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PAIR_COUNT_OPTIONS = (2, 3)
GRID_RANGES = {
    2: ((18, 26), (18, 28)),
    3: ((22, 30), (22, 30)),
}
RECT_H_RANGE = (4, 7)
RECT_W_RANGE = (4, 7)
RECT_PADDING = 2
SCATTER_PADDING = 1
GEN_RETRIES = 400
PARTITION_RETRIES = 160
PLACEMENT_RETRIES = 220


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(counts.most_common(1)[0][0])


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _component_cells(mask: np.ndarray, top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]


def _dilate_points(shape: tuple[int, int], points: list[tuple[int, int]], pad: int) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=bool)
    for r, c in points:
        rr0 = max(0, int(r) - pad)
        rr1 = min(h, int(r) + pad + 1)
        cc0 = max(0, int(c) - pad)
        cc1 = min(w, int(c) + pad + 1)
        out[rr0:rr1, cc0:cc1] = True
    return out


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "idx": int(len(comps)),
                    "color": color,
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                    "coord_set": set(coords),
                    "mask": mask,
                    "top_key": (int(r0), int(c0)),
                }
            )
    return comps


def _anchor_and_mover(a: dict, b: dict) -> tuple[dict, dict]:
    if int(a["area"]) > int(b["area"]):
        return a, b
    if int(b["area"]) > int(a["area"]):
        return b, a
    if tuple(a["top_key"]) <= tuple(b["top_key"]):
        return a, b
    return b, a


def _rectangle_placements(anchor: dict, mover: dict, grid_shape: tuple[int, int]) -> list[dict]:
    h, w = map(int, grid_shape)
    ar0, ac0, ar1, ac1 = map(int, anchor["bbox"])
    total_area = int(anchor["area"]) + int(mover["area"])
    mover_rot = np.rot90(np.asarray(mover["mask"], dtype=bool), 2)
    mh, mw = map(int, mover_rot.shape)
    rel = [tuple(map(int, rc)) for rc in np.argwhere(mover_rot)]
    anchor_cells = anchor["coord_set"]
    placements = []

    row_lo = max(0, ar0 - mh)
    row_hi = min(h - mh, ar1 + 1)
    col_lo = max(0, ac0 - mw)
    col_hi = min(w - mw, ac1 + 1)

    for mr0 in range(row_lo, row_hi + 1):
        for mc0 in range(col_lo, col_hi + 1):
            moved_cells = [(mr0 + rr, mc0 + cc) for rr, cc in rel]
            if any(cell in anchor_cells for cell in moved_cells):
                continue
            rect_r0 = min(ar0, mr0)
            rect_c0 = min(ac0, mc0)
            rect_r1 = max(ar1, mr0 + mh - 1)
            rect_c1 = max(ac1, mc0 + mw - 1)
            if (rect_r1 - rect_r0 + 1) * (rect_c1 - rect_c0 + 1) != total_area:
                continue
            placements.append(
                {
                    "moved_top_left": (int(mr0), int(mc0)),
                    "moved_cells": moved_cells,
                    "rect_bbox": (int(rect_r0), int(rect_c0), int(rect_r1), int(rect_c1)),
                }
            )

    placements.sort(key=lambda p: (p["rect_bbox"], p["moved_top_left"]))
    return placements


def _pair_options(
    components: list[dict],
    grid_shape: tuple[int, int],
    reject_multi_placements: bool,
) -> dict[int, list[dict]] | None:
    n = len(components)
    options = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            anchor, mover = _anchor_and_mover(components[i], components[j])
            placements = _rectangle_placements(anchor, mover, grid_shape)
            if reject_multi_placements and len(placements) != 1:
                if placements:
                    return None
                continue
            if not placements:
                continue
            choice = placements[0]
            rec = {
                "members": (i, j),
                "anchor_idx": int(anchor["idx"]),
                "mover_idx": int(mover["idx"]),
                "rect_bbox": choice["rect_bbox"],
                "moved_cells": choice["moved_cells"],
            }
            options[i].append(rec)
            options[j].append(rec)
    return options


def _solve_pairing(
    components: list[dict],
    grid_shape: tuple[int, int],
    limit: int = 1,
    reject_multi_placements: bool = False,
) -> list[list[dict]]:
    options = _pair_options(components, grid_shape, reject_multi_placements=reject_multi_placements)
    if options is None:
        return []

    n = len(components)
    used: set[int] = set()
    current: list[dict] = []
    solutions: list[list[dict]] = []

    def bt() -> None:
        if len(solutions) >= limit:
            return
        if len(used) == n:
            solutions.append([dict(item) for item in current])
            return
        idx = min(i for i in range(n) if i not in used)
        for opt in options[idx]:
            a, b = opt["members"]
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            current.append(opt)
            bt()
            current.pop()
            used.remove(a)
            used.remove(b)

    bt()
    return solutions


def _render_solution(grid_shape: tuple[int, int], bg: int, components: list[dict], solution: list[dict]) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    for pair in solution:
        r0, c0, r1, c1 = map(int, pair["rect_bbox"])
        color = int(components[int(pair["anchor_idx"])]["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments by 180-degree complementary shape and assemble each pair into a solid rectangle.

    Input structure:
    - The background is azure.
    - Several 4-connected monochrome fragments of distinct colors are scattered on the grid.
    - Every fragment belongs to exactly one pair, and in each pair one fragment can be rotated 180 degrees and translated so that the two fragments exactly fill a solid axis-aligned rectangle.

    Rule:
    - Find the complementary pairs by testing which two fragments can form a rectangle after rotating one member by 180 degrees.
    - In each pair, compare fragment areas: the larger fragment keeps its color and the smaller fragment is recolored to that larger fragment's color.
    - Move the smaller fragment, after the 180-degree rotation, into the larger fragment's missing rectangle area so the pair becomes one solid rectangle.
    - If the two areas are equal, use the color of the fragment whose row is smaller, and move the lower fragment to the upper fragment.

    Key insight:
    - The core step is geometric rather than chromatic: two irregular pieces are matched by rectangle-complementarity under a 180-degree turn, and only after that does the visible area comparison decide the final color owner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components(g, bg)
    assert comps and len(comps) % 2 == 0

    solutions = _solve_pairing(comps, g.shape, limit=1, reject_multi_placements=False)
    assert solutions, "expected a valid complementary pairing"
    return _render_solution(g.shape, bg, comps, solutions[0])


def _irregular(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 4 and area < h * w and h >= 2 and w >= 2


def _grow_connected_region(h: int, w: int, target: int) -> np.ndarray | None:
    for _ in range(PARTITION_RETRIES):
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr, cc = int(r), int(c)
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            frontier_list = list(frontier)
            random.shuffle(frontier_list)
            candidates = []
            for cell in frontier_list:
                mask[cell] = True
                if _connected(~mask):
                    candidates.append(cell)
                mask[cell] = False
            if not candidates:
                break
            mask[random.choice(candidates)] = True
        if int(mask.sum()) == target and _connected(mask) and _connected(~mask):
            return mask
    return None


def _random_partition(equal_area: bool) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PARTITION_RETRIES):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        total = rect_h * rect_w
        if equal_area and total % 2 == 1:
            continue
        if total < 12:
            continue

        if equal_area:
            area_a = total // 2
        else:
            candidates = [a for a in range(4, total - 3) if a != total - a]
            if not candidates:
                continue
            area_a = random.choice(candidates)

        small_area = min(area_a, total - area_a)
        small = _grow_connected_region(rect_h, rect_w, small_area)
        if small is None:
            continue

        if area_a == small_area:
            a_full = small
            b_full = ~small
        else:
            b_full = small
            a_full = ~small

        a_crop, _ = _crop_mask(a_full)
        b_crop, _ = _crop_mask(b_full)
        if not (_irregular(a_crop) and _irregular(b_crop)):
            continue
        return a_full, b_full
    return None


def _make_pair_spec(equal_area: bool, anchor_color: int, mover_color: int) -> dict | None:
    part = _random_partition(equal_area)
    if part is None:
        return None
    a_full, b_full = part
    a_area = int(a_full.sum())
    b_area = int(b_full.sum())

    if equal_area:
        anchor_full, mover_full = random.choice([(a_full, b_full), (b_full, a_full)])
    elif a_area > b_area:
        anchor_full, mover_full = a_full, b_full
    else:
        anchor_full, mover_full = b_full, a_full

    mover_crop, mover_offset = _crop_mask(mover_full)
    if np.array_equal(mover_crop, np.rot90(mover_crop, 2)):
        return None
    anchor_crop, anchor_offset = _crop_mask(anchor_full)

    return {
        "equal_area": bool(equal_area),
        "anchor_color": int(anchor_color),
        "mover_color": int(mover_color),
        "rect_shape": tuple(map(int, anchor_full.shape)),
        "anchor_full": anchor_full,
        "anchor_crop": anchor_crop,
        "anchor_offset": tuple(map(int, anchor_offset)),
        "anchor_area": int(anchor_full.sum()),
        "mover_full": mover_full,
        "mover_crop": mover_crop,
        "mover_offset": tuple(map(int, mover_offset)),
        "mover_area": int(mover_full.sum()),
    }


def _can_place_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    gh, gw = occupied.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    r0 = max(0, top - pad)
    r1 = min(gh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(gw, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rectangles(pairs: list[dict], grid_shape: tuple[int, int]) -> bool:
    h, w = map(int, grid_shape)
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(pairs)), key=lambda i: -pairs[i]["anchor_full"].size)

    for idx in order:
        rect_h, rect_w = pairs[idx]["rect_shape"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - rect_h)
            left = random.randint(0, w - rect_w)
            if not _can_place_rect(occupied, top, left, rect_h, rect_w, RECT_PADDING):
                continue
            occupied[top : top + rect_h, left : left + rect_w] = True
            pairs[idx]["rect_top_left"] = (int(top), int(left))
            placed = True
            break
        if not placed:
            return False
    return True


def _scatter_piece(
    blocked: np.ndarray,
    pair: dict,
    anchor_top: int,
) -> tuple[int, int] | None:
    gh, gw = blocked.shape
    mover_input = np.rot90(np.asarray(pair["mover_crop"], dtype=bool), 2)
    mh, mw = mover_input.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mover_input)]
    candidates = []
    for top in range(0, gh - mh + 1):
        if pair["equal_area"] and top <= anchor_top:
            continue
        for left in range(0, gw - mw + 1):
            ok = True
            for rr, cc in coords:
                if bool(blocked[top + rr, left + cc]):
                    ok = False
                    break
            if ok:
                candidates.append((int(top), int(left)))
    if not candidates:
        return None
    top, left = random.choice(candidates)
    for rr, cc in coords:
        blocked[max(0, top + rr - SCATTER_PADDING) : min(gh, top + rr + SCATTER_PADDING + 1),
                max(0, left + cc - SCATTER_PADDING) : min(gw, left + cc + SCATTER_PADDING + 1)] = True
    return int(top), int(left)


def _pair_components_for_uniqueness(grid: np.ndarray) -> bool:
    bg = _bg_color(grid)
    comps = _components(grid, bg)
    if not comps or len(comps) % 2 == 1:
        return False
    sols = _solve_pairing(comps, grid.shape, limit=2, reject_multi_placements=True)
    return len(sols) == 1


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample(COLOR_POOL, 2 * pair_count)
    pairs = []

    for i in range(pair_count):
        equal_area = bool(random.getrandbits(1))
        spec = _make_pair_spec(equal_area, colors[2 * i], colors[2 * i + 1])
        if spec is None:
            return None
        pairs.append(spec)

    (h_range, w_range) = GRID_RANGES[pair_count]
    grid_h = random.randint(*h_range)
    grid_w = random.randint(*w_range)
    if not _place_rectangles(pairs, (grid_h, grid_w)):
        return None

    expected = np.full((grid_h, grid_w), BG, dtype=int)
    input_grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for pair in pairs:
        rect_h, rect_w = pair["rect_shape"]
        rect_r0, rect_c0 = pair["rect_top_left"]
        expected[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = int(pair["anchor_color"])
        reserved[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = True

        for rr, cc in np.argwhere(pair["anchor_full"]):
            input_grid[rect_r0 + int(rr), rect_c0 + int(cc)] = int(pair["anchor_color"])

    blocked = _dilate_points((grid_h, grid_w), [tuple(map(int, rc)) for rc in np.argwhere(reserved)], 1)
    for pair in pairs:
        rect_r0, rect_c0 = pair["rect_top_left"]
        anchor_cells = _component_cells(pair["anchor_crop"], (rect_r0 + pair["anchor_offset"][0], rect_c0 + pair["anchor_offset"][1]))
        anchor_top = min(r for r, _ in anchor_cells)
        loc = _scatter_piece(blocked, pair, anchor_top)
        if loc is None:
            return None
        top, left = loc
        pair["mover_input_top_left"] = (top, left)
        mover_input = np.rot90(pair["mover_crop"], 2)
        for rr, cc in np.argwhere(mover_input):
            input_grid[top + int(rr), left + int(cc)] = int(pair["mover_color"])

    if Counter(map(int, input_grid.ravel())).most_common(1)[0][0] != BG:
        return None
    if not _pair_components_for_uniqueness(input_grid):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle sample")
