import random
from collections import Counter, deque

import numpy as np


FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_NEIGHBORS = FOUR_NEIGHBORS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ARC_COLORS = list(range(10))
GREY = 5
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_mask(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    deltas = FOUR_NEIGHBORS if connectivity == 4 else EIGHT_NEIGHBORS
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
    return labels, next_id


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h = grid.shape[0]
    candidates = []
    for r in range(h):
        row = grid[r]
        if row[0] == bg or not np.all(row == row[0]):
            continue
        if r == 0 or r == h - 1:
            continue
        if np.any(grid[:r] != bg) and np.any(grid[r + 1 :] != bg):
            candidates.append(r)
    if not candidates:
        raise AssertionError("expected a full-width separator row")
    return candidates[0]


def _extract_legend_colors(top: np.ndarray, bg: int) -> list[int]:
    colors = []
    for color in np.unique(top):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(top == color)
        if len(coords) != 2:
            raise AssertionError("legend swatches must use exactly two cells")
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1, c1 - c0 + 1) not in {(1, 2), (2, 1)}:
            raise AssertionError("legend swatches must be 2x1 or 1x2")
        colors.append((int(c0), int(r0), color))
    colors.sort()
    if not colors:
        raise AssertionError("legend must be non-empty")
    return [color for _c0, _r0, color in colors]


def _analyze_bottom(bottom: np.ndarray, contour_color: int) -> dict:
    contour_mask = bottom == contour_color
    if not np.any(contour_mask):
        raise AssertionError("expected contour pixels")

    region_labels, n_regions = _label_mask(~contour_mask, connectivity=4)
    contour_labels, n_contours = _label_mask(contour_mask, connectivity=8)
    if n_regions == 0 or n_contours == 0:
        raise AssertionError("expected both background regions and contour components")

    border_ids = set(int(v) for v in region_labels[0, :] if v > 0)
    border_ids |= set(int(v) for v in region_labels[-1, :] if v > 0)
    border_ids |= set(int(v) for v in region_labels[:, 0] if v > 0)
    border_ids |= set(int(v) for v in region_labels[:, -1] if v > 0)
    if not border_ids:
        raise AssertionError("expected an exterior background region")

    region_to_contours = {rid: set() for rid in range(1, n_regions + 1)}
    contour_to_regions = {}
    h, w = bottom.shape

    for cid in range(1, n_contours + 1):
        reg_ids = set()
        coords = np.argwhere(contour_labels == cid)
        for r, c in coords:
            for dr, dc in EIGHT_NEIGHBORS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    rid = int(region_labels[nr, nc])
                    if rid > 0:
                        reg_ids.add(rid)
        if len(reg_ids) < 2:
            raise AssertionError("each contour must separate inside and outside regions")
        contour_to_regions[cid] = reg_ids
        for rid in reg_ids:
            region_to_contours[rid].add(cid)

    region_depth = {rid: 0 for rid in border_ids}
    contour_depth = {}
    q = deque(("region", rid) for rid in border_ids)
    while q:
        kind, idx = q.popleft()
        if kind == "region":
            depth = region_depth[idx]
            for cid in region_to_contours[idx]:
                if cid not in contour_depth:
                    contour_depth[cid] = depth + 1
                    q.append(("contour", cid))
        else:
            depth = contour_depth[idx]
            for rid in contour_to_regions[idx]:
                if rid not in region_depth:
                    region_depth[rid] = depth
                    q.append(("region", rid))

    if len(region_depth) != n_regions or len(contour_depth) != n_contours:
        raise AssertionError("nesting graph must reach every region and contour")

    return {
        "region_labels": region_labels,
        "contour_labels": contour_labels,
        "region_depth": region_depth,
        "contour_depth": contour_depth,
        "max_depth": max(contour_depth.values()),
    }


def _render_bottom(bottom: np.ndarray, bg: int, contour_color: int, legend: list[int]) -> np.ndarray:
    analysis = _analyze_bottom(bottom, contour_color)
    if analysis["max_depth"] > len(legend):
        raise AssertionError("legend is too short for contour depth")

    out = np.full(bottom.shape, bg, dtype=int)
    region_labels = analysis["region_labels"]
    contour_labels = analysis["contour_labels"]

    for rid, depth in analysis["region_depth"].items():
        if depth == 0:
            continue
        out[region_labels == rid] = legend[depth - 1]

    for cid, depth in analysis["contour_depth"].items():
        out[contour_labels == cid] = legend[depth - 1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_row = _find_separator_row(g, bg)
    legend = _extract_legend_colors(g[:sep_row], bg)
    bottom = g[sep_row + 1 :]
    colors = [int(c) for c in np.unique(bottom) if int(c) != bg]
    if not colors:
        raise AssertionError("bottom area must contain contours")
    contour_color = max(colors, key=lambda c: int((bottom == c).sum()))
    out = np.full(g.shape, bg, dtype=int)
    out[sep_row + 1 :] = _render_bottom(bottom, bg, contour_color, legend)
    return out


def _draw_rect_contour(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _draw_diamond_contour(canvas: np.ndarray, center_r: int, center_c: int, radius: int, color: int) -> None:
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        canvas[center_r + dr, center_c + dc] = color
        canvas[center_r + dr, center_c - dc] = color


def _shape_bbox(shape: dict) -> tuple[int, int, int, int]:
    if shape["style"] == "rect":
        return shape["r0"], shape["c0"], shape["r1"], shape["c1"]
    radius = shape["radius"]
    return (
        shape["center_r"] - radius,
        shape["center_c"] - radius,
        shape["center_r"] + radius,
        shape["center_c"] + radius,
    )


def _draw_chain(canvas: np.ndarray, shape: dict, depth: int, contour_color: int) -> None:
    if shape["style"] == "rect":
        for level in range(depth):
            inset = 2 * level
            _draw_rect_contour(
                canvas,
                shape["r0"] + inset,
                shape["c0"] + inset,
                shape["r1"] - inset,
                shape["c1"] - inset,
                contour_color,
            )
    else:
        for level in range(depth):
            _draw_diamond_contour(
                canvas,
                shape["center_r"],
                shape["center_c"],
                shape["radius"] - 2 * level,
                contour_color,
            )


def _sample_chain_plan(bottom_h: int, bottom_w: int, depth: int) -> dict | None:
    style = random.choice(["rect", "diamond"])
    if style == "rect":
        min_h = 4 * depth - 1
        min_w = 4 * depth - 1
        if min_h + 2 > bottom_h or min_w + 2 > bottom_w:
            return None
        max_h = min(bottom_h - 2, min_h + 8)
        max_w = min(bottom_w - 2, min_w + 10)
        if max_h < min_h or max_w < min_w:
            return None
        height = random.randint(min_h, max_h)
        width = random.randint(min_w, max_w)
        return {"style": "rect", "height": height, "width": width}

    min_radius = 2 * depth
    max_radius = min((bottom_h - 3) // 2, (bottom_w - 3) // 2, min_radius + 3)
    if max_radius < min_radius:
        return None
    radius = random.randint(min_radius, max_radius)
    return {"style": "diamond", "radius": radius}


def _place_chain(bottom_h: int, bottom_w: int, plan: dict, occupied_boxes: list[tuple[int, int, int, int]]) -> dict | None:
    for _ in range(120):
        if plan["style"] == "rect":
            height = plan["height"]
            width = plan["width"]
            r0 = random.randint(1, bottom_h - height - 1)
            c0 = random.randint(1, bottom_w - width - 1)
            shape = {"style": "rect", "r0": r0, "c0": c0, "r1": r0 + height - 1, "c1": c0 + width - 1}
        else:
            radius = plan["radius"]
            center_r = random.randint(radius + 1, bottom_h - radius - 2)
            center_c = random.randint(radius + 1, bottom_w - radius - 2)
            shape = {"style": "diamond", "center_r": center_r, "center_c": center_c, "radius": radius}

        r0, c0, r1, c1 = _shape_bbox(shape)
        ok = True
        for rr0, cc0, rr1, cc1 in occupied_boxes:
            if not (r1 + 2 < rr0 or rr1 + 2 < r0 or c1 + 2 < cc0 or cc1 + 2 < c0):
                ok = False
                break
        if ok:
            occupied_boxes.append((r0, c0, r1, c1))
            return shape
    return None


def _sample_depths(max_depth: int, group_count: int) -> list[int]:
    depths = [random.randint(1, max_depth) for _ in range(group_count)]
    depths[random.randrange(group_count)] = max_depth
    return depths


def _choose_palette(max_depth: int) -> tuple[int, int, list[int]]:
    usable = [c for c in ARC_COLORS if c != GREY]
    bg = random.choice(usable)
    remaining = [c for c in usable if c != bg]
    contour_color = random.choice(remaining)
    remaining = [c for c in remaining if c != contour_color]
    legend = random.sample(remaining, max_depth)
    return bg, contour_color, legend


def _assemble_grid(bottom: np.ndarray, bg: int, legend: list[int]) -> np.ndarray:
    bottom_h, bottom_w = bottom.shape
    vertical_swatches = random.random() < 0.5
    top_h = random.randint(2, 4)
    grid = np.full((top_h + 1 + bottom_h, bottom_w), bg, dtype=int)
    sep_row = top_h
    grid[sep_row, :] = GREY

    if vertical_swatches:
        row0 = random.randint(0, top_h - 2)
        needed = 2 * len(legend) + (len(legend) - 1)
        start_c = random.randint(1, bottom_w - needed - 1)
        for i, color in enumerate(legend):
            c = start_c + 3 * i
            grid[row0 : row0 + 2, c] = color
    else:
        row0 = random.randint(0, top_h - 1)
        needed = 3 * len(legend) - 1
        start_c = random.randint(1, bottom_w - needed - 1)
        for i, color in enumerate(legend):
            c = start_c + 3 * i
            grid[row0, c : c + 2] = color

    grid[sep_row + 1 :] = bottom
    return grid


def _gen_attempt() -> np.ndarray | None:
    max_depth = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
    bottom_h = random.randint(max(12, 4 * max_depth + 4), 24)
    bottom_w = random.randint(max(16, 4 * max_depth + 8), 30)
    bg, contour_color, legend = _choose_palette(max_depth)

    group_limit = min(3, 1 + bottom_w // 10)
    group_count = random.randint(1, group_limit)
    depths = _sample_depths(max_depth, group_count)

    plans = []
    for depth in depths:
        plan = _sample_chain_plan(bottom_h, bottom_w, depth)
        if plan is None:
            return None
        plans.append((depth, plan))
    plans.sort(key=lambda item: (item[1]["style"] == "diamond", -item[0]))

    occupied_boxes = []
    placed = []
    for depth, plan in plans:
        shape = _place_chain(bottom_h, bottom_w, plan, occupied_boxes)
        if shape is None:
            return None
        placed.append((depth, shape))

    bottom = np.full((bottom_h, bottom_w), bg, dtype=int)
    for depth, shape in placed:
        _draw_chain(bottom, shape, depth, contour_color)

    try:
        analysis = _analyze_bottom(bottom, contour_color)
    except AssertionError:
        return None
    if analysis["max_depth"] != max_depth:
        return None
    depth_counts = Counter(analysis["contour_depth"].values())
    if any(depth_counts[d] == 0 for d in range(1, max_depth + 1)):
        return None

    grid = _assemble_grid(bottom, bg, legend)
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out[: np.where(np.all(grid == GREY, axis=1))[0][0] + 1] != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
