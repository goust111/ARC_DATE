import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
ORANGE = 7
PALETTE = list(range(10))
ARROW_COLORS = [c for c in PALETTE if c not in (BLACK, GRAY, ORANGE)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ["up", "down", "left", "right"]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _frame_mask(shape: tuple[int, int], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _find_components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub, border):
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _arrow_template(direction: str, shaft_len: int) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    if direction == "right":
        w = shaft_len + 2
        coords = [(1, c) for c in range(w)] + [(0, shaft_len), (2, shaft_len)]
        return coords, (1, w - 1), (3, w)
    if direction == "left":
        w = shaft_len + 2
        coords = [(1, c) for c in range(w)] + [(0, 1), (2, 1)]
        return coords, (1, 0), (3, w)
    if direction == "down":
        h = shaft_len + 2
        coords = [(r, 1) for r in range(h)] + [(shaft_len, 0), (shaft_len, 2)]
        return coords, (h - 1, 1), (h, 3)
    assert direction == "up"
    h = shaft_len + 2
    coords = [(r, 1) for r in range(h)] + [(1, 0), (1, 2)]
    return coords, (0, 1), (h, 3)


def _detect_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True

    h, w = sub.shape
    if h == 3 and w >= 3 and sub[1, :].all() and int(sub[0].sum()) == 1 and int(sub[2].sum()) == 1:
        tc = int(np.flatnonzero(sub[0])[0])
        bc = int(np.flatnonzero(sub[2])[0])
        assert tc == bc
        if tc == 1:
            return (0, -1), (r0 + 1, c0)
        if tc == w - 2:
            return (0, 1), (r0 + 1, c1)

    if w == 3 and h >= 3 and sub[:, 1].all() and int(sub[:, 0].sum()) == 1 and int(sub[:, 2].sum()) == 1:
        lr = int(np.flatnonzero(sub[:, 0])[0])
        rr = int(np.flatnonzero(sub[:, 2])[0])
        assert lr == rr
        if lr == 1:
            return (-1, 0), (r0, c0 + 1)
        if lr == h - 2:
            return (1, 0), (r1, c0 + 1)

    raise AssertionError("failed to detect arrow orientation")


def _frame_arrow_info(
    grid: np.ndarray, frame: tuple[int, int, int, int]
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in np.unique(interior) if int(v) != BLACK]
    assert len(colors) == 1, "frame interior must contain exactly one non-black color"
    arrow_color = colors[0]
    total = int(np.count_nonzero(interior == arrow_color))

    color_mask = np.zeros_like(grid, dtype=bool)
    color_mask[r0 + 1 : r1, c0 + 1 : c1] = interior == arrow_color
    comps = _find_components(color_mask)
    assert comps, "expected arrow component"
    arrow_comp = max(comps, key=len)
    for comp in comps:
        if comp is arrow_comp:
            continue
        assert len(comp) == 1, "non-arrow same-color pixels must be isolated dots"

    direction, tip = _detect_arrow(arrow_comp)
    return arrow_color, total, direction, tip


def _ray_start(
    frame: tuple[int, int, int, int], tip: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    tr, tc = tip
    dr, dc = direction
    if dr == -1:
        return r0 - 1, tc
    if dr == 1:
        return r1 + 1, tc
    if dc == -1:
        return tr, c0 - 1
    return tr, c1 + 1


def _find_outer_rectangles(
    grid: np.ndarray, frame_cover: np.ndarray
) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], np.ndarray]:
    mask = (~frame_cover) & (grid != BLACK) & (grid != GRAY)
    rects: list[dict[str, int | tuple[int, int, int, int]]] = []
    rect_map = np.full(grid.shape, -1, dtype=int)
    for comp in _find_components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = int(grid[comp[0]])
        assert all(int(grid[r, c]) == color for r, c in comp)
        assert len(comp) == h * w, "outer object must be a solid rectangle"
        rect_idx = len(rects)
        rects.append({"bbox": (r0, c0, r1, c1), "color": color})
        rect_map[r0 : r1 + 1, c0 : c1 + 1] = rect_idx
    return rects, rect_map


def _split_rectangle(
    out: np.ndarray,
    rect: dict[str, int | tuple[int, int, int, int]],
    rr: int,
    cc: int,
    ray_color: int,
    direction: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
    base = int(rect["color"])  # type: ignore[arg-type]
    dr, dc = direction
    if dr == 0:
        out[r0:rr, c0 : c1 + 1] = base
        out[rr, c0 : c1 + 1] = ray_color
        out[rr + 1 : r1 + 1, c0 : c1 + 1] = ORANGE
    else:
        out[r0 : r1 + 1, c0:cc] = base
        out[r0 : r1 + 1, cc] = ray_color
        out[r0 : r1 + 1, cc + 1 : c1 + 1] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one gray frame"
    frame_cover = _frame_mask(g.shape, frames)
    rects, rect_map = _find_outer_rectangles(g, frame_cover)

    out = g.copy()
    cut_rects: set[int] = set()
    for frame in frames:
        arrow_color, total, direction, tip = _frame_arrow_info(g, frame)
        sr, sc = _ray_start(frame, tip, direction)
        dr, dc = direction
        for step in range(total):
            rr = sr + dr * step
            cc = sc + dc * step
            if not _in_bounds(g.shape, rr, cc):
                break
            rect_idx = int(rect_map[rr, cc])
            if rect_idx >= 0:
                if rect_idx not in cut_rects:
                    _split_rectangle(out, rects[rect_idx], rr, cc, arrow_color, direction)
                    cut_rects.add(rect_idx)
            else:
                out[rr, cc] = arrow_color
    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_COUNT_CHOICES = [1, 2, 3]
FRAME_COUNT_WEIGHTS = [2, 3, 2]
FRAME_H_RANGE = (8, 12)
FRAME_W_RANGE = (8, 12)
FRAME_EDGE_MARGIN = 3
FRAME_GAP = 2
FRAME_PLACE_RETRIES = 120
CONTENT_RETRIES = 120
SHAFT_MAX = 4
DOT_COUNT_RANGE = (0, 4)
RAY_HIT_WEIGHTS = [2, 4, 1]  # 0,1,2 hits
RECT_SIZE_RANGE = (3, 5)
EXTRA_RECT_RANGE = (0, 3)
RECT_PLACE_RETRIES = 120
GENERATE_RETRIES = 400


def _bbox_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _trace_ray(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
    total: int,
) -> list[tuple[int, int]]:
    sr, sc = _ray_start(frame, tip, direction)
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(total):
        rr = sr + dr * step
        cc = sc + dc * step
        if not _in_bounds(shape, rr, cc):
            break
        cells.append((rr, cc))
    return cells


def _is_isolated_dot(
    cell: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    dots: set[tuple[int, int]],
) -> bool:
    r, c = cell
    for rr, cc in arrow_cells | dots:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _sample_dots(
    frame: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    dot_count: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = frame
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in arrow_cells
    ]
    random.shuffle(candidates)
    dots: set[tuple[int, int]] = set()
    for cell in candidates:
        if _is_isolated_dot(cell, arrow_cells, dots):
            dots.add(cell)
            if len(dots) == dot_count:
                return sorted(dots)
    return None if dot_count else []


def _sample_region_contents(
    frame: tuple[int, int, int, int]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    for _ in range(CONTENT_RETRIES):
        direction = random.choice(DIRECTIONS)
        if direction in ("left", "right"):
            max_shaft = min(SHAFT_MAX, inner_w - 2)
        else:
            max_shaft = min(SHAFT_MAX, inner_h - 2)
        if max_shaft < 1:
            continue
        shaft_len = random.randint(1, max_shaft)
        rel_coords, rel_tip, (ah, aw) = _arrow_template(direction, shaft_len)
        top = random.randint(r0 + 1, r1 - ah)
        left = random.randint(c0 + 1, c1 - aw)
        arrow_cells = {(top + rr, left + cc) for rr, cc in rel_coords}
        tip = (top + rel_tip[0], left + rel_tip[1])
        max_dots = min(DOT_COUNT_RANGE[1], inner_h * inner_w - len(arrow_cells))
        min_dots = min(DOT_COUNT_RANGE[0], max_dots)
        dot_count = random.randint(min_dots, max_dots)
        dots = _sample_dots(frame, arrow_cells, dot_count)
        if dots is None:
            continue
        return {
            "frame": frame,
            "direction_name": direction,
            "direction": {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }[direction],
            "shaft_len": shaft_len,
            "arrow_cells": sorted(arrow_cells),
            "dots": dots,
            "tip": tip,
            "total": len(arrow_cells) + len(dots),
        }
    return None


def _place_frames(h: int, w: int, n_frames: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_frames):
        placed = False
        for _ in range(FRAME_PLACE_RETRIES):
            fh = random.randint(*FRAME_H_RANGE)
            fw = random.randint(*FRAME_W_RANGE)
            if fh + 2 * FRAME_EDGE_MARGIN > h or fw + 2 * FRAME_EDGE_MARGIN > w:
                continue
            r0 = random.randint(FRAME_EDGE_MARGIN, h - fh - FRAME_EDGE_MARGIN)
            c0 = random.randint(FRAME_EDGE_MARGIN, w - fw - FRAME_EDGE_MARGIN)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(box, other, pad=FRAME_GAP) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    boxes.sort()
    return boxes


def _build_ray_masks(rays: list[list[tuple[int, int]]], shape: tuple[int, int]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for ray in rays:
        mask = np.zeros(shape, dtype=bool)
        for r, c in ray:
            mask[r, c] = True
        masks.append(mask)
    return masks


def _choose_rect_color(arrow_colors: set[int]) -> int:
    pool = [c for c in PALETTE if c not in arrow_colors and c not in (BLACK, GRAY, ORANGE)]
    return random.choice(pool)


def _candidate_hit_bbox(
    ray: list[tuple[int, int]],
    direction: tuple[int, int],
    occupied: np.ndarray,
    other_ray_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    horizontal = direction[0] == 0
    sizes = [(rh, rw) for rh in range(RECT_SIZE_RANGE[0], RECT_SIZE_RANGE[1] + 1) for rw in range(RECT_SIZE_RANGE[0], RECT_SIZE_RANGE[1] + 1)]
    random.shuffle(sizes)
    for rh, rw in sizes:
        axis_len = rw if horizontal else rh
        if len(ray) < axis_len:
            continue
        cuts = list(range(1, (rh if horizontal else rw) - 1))
        random.shuffle(cuts)
        starts = list(range(len(ray) - axis_len + 1))
        random.shuffle(starts)
        for cut in cuts:
            for start in starts:
                seg = ray[start : start + axis_len]
                if horizontal:
                    row = seg[0][0]
                    cols = [c for _, c in seg]
                    bbox = (row - cut, min(cols), row - cut + rh - 1, max(cols))
                else:
                    col = seg[0][1]
                    rows = [r for r, _ in seg]
                    bbox = (min(rows), col - cut, max(rows), col - cut + rw - 1)
                r0, c0, r1, c1 = bbox
                if r0 < 0 or c0 < 0 or r1 >= occupied.shape[0] or c1 >= occupied.shape[1]:
                    continue
                if other_ray_mask[r0 : r1 + 1, c0 : c1 + 1].any():
                    continue
                if not _bbox_free(occupied, bbox, pad=1):
                    continue
                return bbox
    return None


def _place_hit_rectangles(
    regions: list[dict[str, object]],
    rays: list[list[tuple[int, int]]],
    ray_masks: list[np.ndarray],
    occupied: np.ndarray,
    arrow_colors: set[int],
) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    total_hits = 0
    for i, region in enumerate(regions):
        choices = [0, 1, 2] if len(rays[i]) >= 5 else [0, 1]
        weights = RAY_HIT_WEIGHTS[: len(choices)]
        desired = random.choices(choices, weights=weights)[0]
        other_mask = np.zeros_like(occupied)
        for j, mask in enumerate(ray_masks):
            if i != j:
                other_mask |= mask
        placed = 0
        for _ in range(desired):
            bbox = _candidate_hit_bbox(rays[i], region["direction"], occupied, other_mask)  # type: ignore[arg-type]
            if bbox is None:
                break
            rect = {"bbox": bbox, "color": _choose_rect_color(arrow_colors), "ray": i}
            rects.append(rect)
            _mark_bbox(occupied, bbox)
            placed += 1
        total_hits += placed

    if total_hits:
        return rects

    order = list(range(len(regions)))
    random.shuffle(order)
    for i in order:
        other_mask = np.zeros_like(occupied)
        for j, mask in enumerate(ray_masks):
            if i != j:
                other_mask |= mask
        bbox = _candidate_hit_bbox(rays[i], regions[i]["direction"], occupied, other_mask)  # type: ignore[arg-type]
        if bbox is None:
            continue
        rect = {"bbox": bbox, "color": _choose_rect_color(arrow_colors), "ray": i}
        rects.append(rect)
        _mark_bbox(occupied, bbox)
        return rects
    return []


def _place_extra_rectangles(
    shape: tuple[int, int],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    arrow_colors: set[int],
    min_count: int,
) -> list[dict[str, object]] | None:
    rects: list[dict[str, object]] = []
    desired = random.randint(max(min_count, EXTRA_RECT_RANGE[0]), EXTRA_RECT_RANGE[1])
    for _ in range(desired):
        placed = False
        for _ in range(RECT_PLACE_RETRIES):
            rh = random.randint(*RECT_SIZE_RANGE)
            rw = random.randint(*RECT_SIZE_RANGE)
            r0 = random.randint(0, shape[0] - rh)
            c0 = random.randint(0, shape[1] - rw)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if ray_mask[r0 : r0 + rh, c0 : c0 + rw].any():
                continue
            if not _bbox_free(occupied, bbox, pad=1):
                continue
            rect = {"bbox": bbox, "color": _choose_rect_color(arrow_colors), "ray": None}
            rects.append(rect)
            _mark_bbox(occupied, bbox)
            placed = True
            break
        if not placed:
            return None
    return rects


def _render_input(
    shape: tuple[int, int],
    regions: list[dict[str, object]],
    arrow_colors: list[int],
    rects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for region, color in zip(regions, arrow_colors):
        r0, c0, r1, c1 = region["frame"]  # type: ignore[index]
        grid[r0 : r1 + 1, c0] = GRAY
        grid[r0 : r1 + 1, c1] = GRAY
        grid[r0, c0 : c1 + 1] = GRAY
        grid[r1, c0 : c1 + 1] = GRAY
        for r, c in region["arrow_cells"]:  # type: ignore[index]
            grid[r, c] = color
        for r, c in region["dots"]:  # type: ignore[index]
            grid[r, c] = color
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _oracle(
    inp: np.ndarray,
    regions: list[dict[str, object]],
    arrow_colors: list[int],
    rects: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    rect_map = np.full(inp.shape, -1, dtype=int)
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
        rect_map[r0 : r1 + 1, c0 : c1 + 1] = idx
    cut: set[int] = set()
    for region, color in zip(regions, arrow_colors):
        direction = region["direction"]  # type: ignore[assignment]
        frame = region["frame"]  # type: ignore[assignment]
        tip = region["tip"]  # type: ignore[assignment]
        ray = _trace_ray(inp.shape, frame, tip, direction, int(region["total"]))  # type: ignore[arg-type]
        for rr, cc in ray:
            rect_idx = int(rect_map[rr, cc])
            if rect_idx >= 0:
                if rect_idx not in cut:
                    _split_rectangle(out, rects[rect_idx], rr, cc, color, direction)  # type: ignore[arg-type]
                    cut.add(rect_idx)
            else:
                out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_frames = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS)[0]

    frames = _place_frames(h, w, n_frames)
    if frames is None:
        return None

    regions: list[dict[str, object]] = []
    for frame in frames:
        region = _sample_region_contents(frame)
        if region is None:
            return None
        regions.append(region)

    frame_cover = _frame_mask((h, w), frames)
    rays: list[list[tuple[int, int]]] = []
    seen_ray_cells: set[tuple[int, int]] = set()
    for region in regions:
        ray = _trace_ray((h, w), region["frame"], region["tip"], region["direction"], int(region["total"]))  # type: ignore[arg-type]
        if not ray:
            return None
        if any(frame_cover[r, c] for r, c in ray):
            return None
        if any(cell in seen_ray_cells for cell in ray):
            return None
        rays.append(ray)
        seen_ray_cells.update(ray)

    ray_masks = _build_ray_masks(rays, (h, w))
    all_ray_mask = np.zeros((h, w), dtype=bool)
    for mask in ray_masks:
        all_ray_mask |= mask

    occupied = frame_cover.copy()
    arrow_colors = random.sample(ARROW_COLORS, len(regions))
    rects = _place_hit_rectangles(regions, rays, ray_masks, occupied, set(arrow_colors))
    min_extra = 1 if not rects else 0
    extras = _place_extra_rectangles((h, w), occupied, all_ray_mask, set(arrow_colors), min_extra)
    if extras is None:
        return None
    rects.extend(extras)
    if not rects:
        return None

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
        for i, ray in enumerate(rays):
            inside = [(r, c) for r, c in ray if r0 <= r <= r1 and c0 <= c <= c1]
            if not inside:
                continue
            if rect["ray"] != i:  # type: ignore[index]
                return None

    inp = _render_input((h, w), regions, arrow_colors, rects)
    expected = _oracle(inp, regions, arrow_colors, rects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
