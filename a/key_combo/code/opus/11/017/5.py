import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
BRIDGE = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FG_COLORS = [1, 2, 3, 4, 6, 7, 9]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def clip_rect_to_grid(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    h, w = shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    clipped = clip_rect_to_grid(box, grid.shape)
    if clipped is None:
        return
    r0, c0, r1, c1 = clipped
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def rects_from_grid(grid: np.ndarray) -> dict[int, list[tuple[int, int, int, int]]]:
    bg = get_bg(grid)
    by_color: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, BRIDGE)):
        for comp in find_components(grid == color):
            box = component_bbox(comp)
            r0, c0, r1, c1 = box
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                raise AssertionError("foreground components must be solid rectangles")
            by_color[color].append(box)
    return by_color


def _multiple_of_step(delta: tuple[int, int], step: tuple[int, int]) -> int | None:
    dr, dc = delta
    sr, sc = step
    if sr == 0 and sc == 0:
        return None
    if sr == 0:
        if dr != 0 or sc == 0 or dc % sc != 0:
            return None
        k = dc // sc
        return k if k >= 0 else None
    if sc == 0:
        if dc != 0 or dr % sr != 0:
            return None
        k = dr // sr
        return k if k >= 0 else None
    if dr % sr != 0 or dc % sc != 0:
        return None
    kr = dr // sr
    kc = dc // sc
    if kr != kc or kr < 0:
        return None
    return kr


def _infer_1d_positions(
    positions: list[int], rect_len: int, limit: int
) -> list[int]:
    positions = sorted(int(p) for p in positions)
    pos_set = set(positions)
    candidates: list[tuple[tuple[int, ...], list[int]]] = []
    diffs = sorted({positions[j] - positions[i] for i in range(len(positions)) for j in range(i + 1, len(positions)) if positions[j] > positions[i]})
    for diff in diffs:
        for div in range(1, 6):
            if diff % div != 0:
                continue
            step = diff // div
            if step <= 0:
                continue
            base = positions[0]
            if any((p - base) % step != 0 for p in positions):
                continue
            indices = [(p - base) // step for p in positions]
            min_idx = min(indices)
            max_idx = max(indices)
            span = max_idx - min_idx + 1
            internal_missing = span - len(positions)
            if internal_missing > 2:
                continue
            for missing_total in (1, 2):
                extra = missing_total - internal_missing
                if extra < 0:
                    continue
                total_count = span + extra
                if not (3 <= total_count <= 6):
                    continue
                for before in range(extra + 1):
                    start = base + (min_idx - before) * step
                    full = [start + k * step for k in range(total_count)]
                    if any((p + rect_len - 1) < 0 or p >= limit for p in full):
                        continue
                    if not pos_set.issubset(full):
                        continue
                    missing = [p for p in full if p not in pos_set]
                    internal_missing_now = sum(full[0] < p < full[-1] for p in missing)
                    clipped_missing = sum(p < 0 or p + rect_len - 1 >= limit for p in missing)
                    end_inside_missing = len(missing) - internal_missing_now - clipped_missing
                    score = (
                        step,
                        end_inside_missing,
                        total_count,
                        -internal_missing_now,
                        -clipped_missing,
                        abs(start),
                    )
                    candidates.append((score, full))
    if not candidates:
        raise AssertionError("could not infer arithmetic sequence")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def infer_full_rects_for_color(
    rects: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    rects = sorted(rects)
    heights = {r1 - r0 + 1 for r0, _, r1, _ in rects}
    widths = {c1 - c0 + 1 for _, c0, _, c1 in rects}
    if len(heights) != 1 or len(widths) != 1:
        raise AssertionError("same-color rectangles must share one size")
    h = next(iter(heights))
    w = next(iter(widths))
    same_rows = len({(r0, r1) for r0, _, r1, _ in rects}) == 1
    same_cols = len({(c0, c1) for _, c0, _, c1 in rects}) == 1
    if same_rows and not same_cols:
        r0, _, r1, _ = rects[0]
        positions = [c0 for _, c0, _, _ in rects]
        full_positions = _infer_1d_positions(positions, w, shape[1])
        return [(r0, c0, r1, c0 + w - 1) for c0 in full_positions]
    if same_cols and not same_rows:
        _, c0, _, c1 = rects[0]
        positions = [r0 for r0, _, _, _ in rects]
        full_positions = _infer_1d_positions(positions, h, shape[0])
        return [(r0, c0, r0 + h - 1, c1) for r0 in full_positions]

    centers = sorted((r0 + r1, c0 + c1) for r0, c0, r1, c1 in rects)
    center_set = set(centers)
    candidates: list[tuple[tuple[int, ...], list[tuple[int, int, int, int]]]] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dr = centers[j][0] - centers[i][0]
            dc = centers[j][1] - centers[i][1]
            for div in range(1, 6):
                if dr % div != 0 or dc % div != 0:
                    continue
                step = (dr // div, dc // div)
                if step == (0, 0):
                    continue
                base = centers[0]
                ks = []
                for center in centers:
                    k = _multiple_of_step((center[0] - base[0], center[1] - base[1]), step)
                    if k is None:
                        break
                    ks.append(k)
                else:
                    min_k = min(ks)
                    max_k = max(ks)
                    span = max_k - min_k + 1
                    internal_missing = span - len(centers)
                    if internal_missing > 2:
                        continue
                    for missing_total in (1, 2):
                        extra = missing_total - internal_missing
                        if extra < 0:
                            continue
                        total_count = span + extra
                        if not (3 <= total_count <= 6):
                            continue
                        for before in range(extra + 1):
                            start_center = (
                                base[0] + (min_k - before) * step[0],
                                base[1] + (min_k - before) * step[1],
                            )
                            full_centers = [
                                (start_center[0] + n * step[0], start_center[1] + n * step[1])
                                for n in range(total_count)
                            ]
                            if not center_set.issubset(full_centers):
                                continue
                            full_rects = []
                            valid = True
                            for cr2, cc2 in full_centers:
                                if (cr2 - (h - 1)) % 2 != 0 or (cc2 - (w - 1)) % 2 != 0:
                                    valid = False
                                    break
                                r0 = (cr2 - (h - 1)) // 2
                                c0 = (cc2 - (w - 1)) // 2
                                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                                if clip_rect_to_grid(box, shape) is None:
                                    valid = False
                                    break
                                full_rects.append(box)
                            if not valid:
                                continue
                            candidates.append(
                                (
                                    (
                                        abs(step[0]) + abs(step[1]),
                                        total_count,
                                        abs(start_center[0]) + abs(start_center[1]),
                                    ),
                                    full_rects,
                                )
                            )
    if not candidates:
        raise AssertionError("could not infer fallback sequence")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def bridge_candidates(
    rects: list[tuple[int, int, int, int]], rect_grid: np.ndarray, bg: int
) -> list[tuple[int, int, int, tuple[str, int, int, int]]]:
    candidates: list[tuple[int, int, int, tuple[str, int, int, int]]] = []
    for i in range(len(rects)):
        ar0, ac0, ar1, ac1 = rects[i]
        for j in range(i + 1, len(rects)):
            br0, bc0, br1, bc1 = rects[j]
            shared_r0 = max(ar0, br0)
            shared_r1 = min(ar1, br1)
            if shared_r1 - shared_r0 + 1 >= 3:
                if ac1 < bc0:
                    gap_c0, gap_c1 = ac1 + 1, bc0 - 1
                elif bc1 < ac0:
                    gap_c0, gap_c1 = bc1 + 1, ac0 - 1
                else:
                    gap_c0 = gap_c1 = 1
                if gap_c0 <= gap_c1:
                    rr0 = max(0, shared_r0)
                    rr1 = min(rect_grid.shape[0] - 1, shared_r1)
                    cc0 = max(0, gap_c0)
                    cc1 = min(rect_grid.shape[1] - 1, gap_c1)
                    if rr0 <= rr1 and cc0 <= cc1 and np.all(rect_grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg):
                        row = (shared_r0 + shared_r1) // 2
                        candidates.append((gap_c1 - gap_c0 + 1, i, j, ("H", row, gap_c0, gap_c1)))
            shared_c0 = max(ac0, bc0)
            shared_c1 = min(ac1, bc1)
            if shared_c1 - shared_c0 + 1 >= 3:
                if ar1 < br0:
                    gap_r0, gap_r1 = ar1 + 1, br0 - 1
                elif br1 < ar0:
                    gap_r0, gap_r1 = br1 + 1, ar0 - 1
                else:
                    gap_r0 = gap_r1 = 1
                if gap_r0 <= gap_r1:
                    rr0 = max(0, gap_r0)
                    rr1 = min(rect_grid.shape[0] - 1, gap_r1)
                    cc0 = max(0, shared_c0)
                    cc1 = min(rect_grid.shape[1] - 1, shared_c1)
                    if rr0 <= rr1 and cc0 <= cc1 and np.all(rect_grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg):
                        col = (shared_c0 + shared_c1) // 2
                        candidates.append((gap_r1 - gap_r0 + 1, i, j, ("V", col, gap_r0, gap_r1)))
    return candidates


def paint_bridge(
    out: np.ndarray, geom: tuple[str, int, int, int], color: int = BRIDGE
) -> None:
    axis, fixed, start, end = geom
    if axis == "H":
        if 0 <= fixed < out.shape[0]:
            c0 = max(0, start)
            c1 = min(out.shape[1] - 1, end)
            if c0 <= c1:
                out[fixed, c0 : c1 + 1] = color
        return
    if 0 <= fixed < out.shape[1]:
        r0 = max(0, start)
        r1 = min(out.shape[0] - 1, end)
        if r0 <= r1:
            out[r0 : r1 + 1, fixed] = color


def kruskal_bridges(
    rects: list[tuple[int, int, int, int]], rect_grid: np.ndarray, bg: int
) -> list[tuple[str, int, int, int]]:
    edges = sorted(bridge_candidates(rects, rect_grid, bg), key=lambda item: (item[0], item[1], item[2], item[3]))
    parent = list(range(len(rects)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen: list[tuple[str, int, int, int]] = []
    for _dist, a, b, geom in edges:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        chosen.append(geom)
    return chosen


def reconstruct_all_rects(
    grid: np.ndarray,
) -> dict[int, list[tuple[int, int, int, int]]]:
    visible = rects_from_grid(grid)
    return {
        color: infer_full_rects_for_color(rects, grid.shape)
        for color, rects in visible.items()
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color's arithmetic rectangle sequence and add teal bridges across the shortest clear gaps.

    Input structure:
    - The grid has a black background and two foreground colors.
    - Each foreground component is a solid axis-aligned rectangle of size 2x3 or 3x2.
    - Same-color rectangles share one row band or one column band, and the visible rectangles are members of a longer arithmetic sequence with 1-2 missing positions.

    Rule:
    - For each color independently, infer the constant step of the rectangle sequence and restore the missing same-size rectangles, clipping any restored rectangle to the visible canvas.
    - After restoration, consider all rectangles of that color and find facing pairs whose overlapping row span or column span is at least three cells and whose full gap region is still background.
    - Add teal bridges only for the shortest acyclic set of candidate gaps, drawing each bridge as a one-pixel-wide line through the middle of the shared span.

    Key insight:
    - The legal bridge targets are determined only after the hidden arithmetic-sequence members have been restored, so sequence completion must happen before shortest-gap bridge selection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    all_rects = reconstruct_all_rects(g)

    rect_grid = np.full_like(g, bg)
    for color, rects in all_rects.items():
        for box in rects:
            paint_rect(rect_grid, box, color)

    out = rect_grid.copy()
    for color, rects in all_rects.items():
        for geom in kruskal_bridges(rects, rect_grid, bg):
            paint_bridge(out, geom, BRIDGE)

    return out


def _sequence_boxes(
    orientation: str,
    fixed: int,
    positions: list[int],
    rect_h: int,
    rect_w: int,
) -> list[tuple[int, int, int, int]]:
    if orientation == "H":
        return [(fixed, c0, fixed + rect_h - 1, c0 + rect_w - 1) for c0 in positions]
    return [(r0, fixed, r0 + rect_h - 1, fixed + rect_w - 1) for r0 in positions]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _visible_full_inside(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return 1 <= r0 and r1 <= h - 2 and 1 <= c0 and c1 <= w - 2


def _partial_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    clipped = clip_rect_to_grid(box, shape)
    return clipped is not None and clipped != box


def _sequence_gap(rect_h: int, rect_w: int, step: int, orientation: str) -> int:
    if orientation == "H":
        return step - rect_w
    return step - rect_h


def _place_group(
    shape: tuple[int, int],
    orientation: str,
    total_count: int,
    missing_mode: str,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]] | None:
    h, w = shape
    color = -1
    rect_h, rect_w = (3, 2) if orientation == "H" else (2, 3)
    step = (rect_w if orientation == "H" else rect_h) + random.randint(1, 4)
    gap = _sequence_gap(rect_h, rect_w, step, orientation)
    if gap <= 0:
        return None

    indices = list(range(total_count))
    if missing_mode == "internal1":
        missing = {random.randint(1, total_count - 2)}
    elif missing_mode == "internal2":
        missing = set(random.sample(indices[1:-1], 2))
    elif missing_mode == "clip1":
        missing = {0 if random.random() < 0.5 else total_count - 1}
    else:
        end_idx = 0 if random.random() < 0.5 else total_count - 1
        internal_choices = [i for i in indices[1:-1]]
        missing = {end_idx, random.choice(internal_choices)}

    if orientation == "H":
        fixed = random.randint(1, h - rect_h - 2)
        if 0 in missing and total_count - 1 in missing:
            return None
        if 0 in missing:
            pos0 = -(rect_w - 1) - random.randint(0, step - 1)
        else:
            pos0 = random.randint(1, 4)
        max_c1 = pos0 + (total_count - 1) * step + rect_w - 1
        if max_c1 > w - 2:
            shift = max_c1 - (w - 2)
            pos0 -= shift
        if total_count - 1 in missing:
            pos0 = min(pos0, w - (total_count - 1) * step - 1)
            end_box = (fixed, pos0 + (total_count - 1) * step, fixed + rect_h - 1, pos0 + (total_count - 1) * step + rect_w - 1)
            if end_box[1] > w - 1:
                pos0 -= end_box[1] - (w - 1)
        positions = [pos0 + i * step for i in range(total_count)]
    else:
        fixed = random.randint(1, w - rect_w - 2)
        if 0 in missing and total_count - 1 in missing:
            return None
        if 0 in missing:
            pos0 = -(rect_h - 1) - random.randint(0, step - 1)
        else:
            pos0 = random.randint(1, 4)
        max_r1 = pos0 + (total_count - 1) * step + rect_h - 1
        if max_r1 > h - 2:
            shift = max_r1 - (h - 2)
            pos0 -= shift
        if total_count - 1 in missing:
            pos0 = min(pos0, h - (total_count - 1) * step - 1)
            end_box = (pos0 + (total_count - 1) * step, fixed, pos0 + (total_count - 1) * step + rect_h - 1, fixed + rect_w - 1)
            if end_box[0] > h - 1:
                pos0 -= end_box[0] - (h - 1)
        positions = [pos0 + i * step for i in range(total_count)]

    full_boxes = _sequence_boxes(orientation, fixed, positions, rect_h, rect_w)
    visible_boxes = [box for i, box in enumerate(full_boxes) if i not in missing]
    missing_boxes = [box for i, box in enumerate(full_boxes) if i in missing]

    if len(visible_boxes) < 2:
        return None
    if any(not _visible_full_inside(box, shape) for box in visible_boxes):
        return None
    if any(clip_rect_to_grid(box, shape) is None for box in full_boxes):
        return None
    if missing_mode.startswith("clip") and not any(_partial_box(box, shape) for box in missing_boxes):
        return None
    if any(_boxes_overlap(a, b) for i, a in enumerate(full_boxes) for b in full_boxes[i + 1 :]):
        return None
    for box in full_boxes:
        for other in occupied:
            if _boxes_overlap(box, other):
                return None
    if not bridge_candidates(full_boxes, np.full(shape, BLACK, dtype=int), BLACK):
        return None

    grid = np.zeros(shape, dtype=int)
    return grid, fixed, visible_boxes, full_boxes


def _group_params_variants() -> list[tuple[int, str]]:
    return [
        (4, "internal1"),
        (5, "internal1"),
        (5, "internal2"),
        (4, "clip1"),
        (5, "clip2"),
    ]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    shape = (h, w)
    colors = random.sample(FG_COLORS, 2)
    orientations = random.sample(["H", "V"], 2) if random.random() < 0.7 else [random.choice(["H", "V"]) for _ in range(2)]
    group_specs = random.sample(_group_params_variants(), 2)

    groups = []
    occupied: list[tuple[int, int, int, int]] = []
    for orientation, spec, color in zip(orientations, group_specs, colors):
        total_count, missing_mode = spec
        placed = None
        for _ in range(80):
            placed = _place_group(shape, orientation, total_count, missing_mode, occupied)
            if placed is not None:
                break
        if placed is None:
            return None
        _blank, _fixed, visible_boxes, full_boxes = placed
        occupied.extend(full_boxes)
        groups.append((color, visible_boxes, full_boxes))

    grid = np.zeros(shape, dtype=int)
    truth: dict[int, list[tuple[int, int, int, int]]] = {}
    for color, visible_boxes, full_boxes in groups:
        truth[color] = sorted(full_boxes)
        for box in visible_boxes:
            paint_rect(grid, box, color)

    detected = reconstruct_all_rects(grid)
    if {k: sorted(v) for k, v in detected.items()} != truth:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if BRIDGE not in out:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
