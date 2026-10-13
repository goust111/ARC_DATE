import numpy as np


BG = 0
YELLOW = 4
BODY_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_MASKS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}
OVERLAP_DIMS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)
MOVER_SIDES = (7, 9)
STATIC_SIDES = (5, 7, 9)
COVER_SIDES = (5, 7)
MAX_GEN_ATTEMPTS = 400
MAX_GRID = 30
MIN_GAP = 2


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _render_patch(height: int, width: int, color: int, direction: str) -> np.ndarray:
    patch = np.full((height, width), int(color), dtype=int)
    mask = ARROW_MASKS[direction]
    cr = height // 2
    cc = width // 2
    patch[cr - 1 : cr + 2, cc - 1 : cc + 2][mask] = YELLOW
    return patch


def _stamp(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(grid.shape[0], r0 + h)
    cc1 = min(grid.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    grid[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def _bbox_intersection(a, b):
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _corner_bbox(height: int, width: int, corner: str, oh: int, ow: int):
    if corner == "tl":
        return (0, 0, oh - 1, ow - 1)
    if corner == "tr":
        return (0, width - ow, oh - 1, width - 1)
    if corner == "bl":
        return (height - oh, 0, height - 1, ow - 1)
    return (height - oh, width - ow, height - 1, width - 1)


def _overlap_hits_arrow(height: int, width: int, corner: str, oh: int, ow: int) -> bool:
    cr = height // 2
    cc = width // 2
    arrow_bbox = (cr - 1, cc - 1, cr + 1, cc + 1)
    overlap_bbox = _corner_bbox(height, width, corner, oh, ow)
    return _bbox_intersection(arrow_bbox, overlap_bbox) is not None


def _compatible_corners(direction: str):
    if direction == "up":
        return ("bl", "br")
    if direction == "down":
        return ("tl", "tr")
    if direction == "left":
        return ("tr", "br")
    return ("tl", "bl")


def _cover_position(
    mover_h: int,
    mover_w: int,
    cover_h: int,
    cover_w: int,
    corner: str,
    oh: int,
    ow: int,
):
    if corner == "tl":
        return (-(cover_h - oh), -(cover_w - ow))
    if corner == "tr":
        return (-(cover_h - oh), mover_w - ow)
    if corner == "bl":
        return (mover_h - oh, -(cover_w - ow))
    return (mover_h - oh, mover_w - ow)


def _detect_arrows(grid: np.ndarray):
    h, w = grid.shape
    arrows = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = grid[r - 1 : r + 2, c - 1 : c + 2] == YELLOW
            if int(patch.sum()) != 5:
                continue
            matched = [name for name, mask in ARROW_MASKS.items() if np.array_equal(patch, mask)]
            if not matched:
                continue
            direction = matched[0]
            mask = ARROW_MASKS[direction]
            cells = {
                (r - 1 + rr, c - 1 + cc)
                for rr in range(3)
                for cc in range(3)
                if mask[rr, cc]
            }
            arrows.append({"center": (r, c), "direction": direction, "arrow_cells": cells})
    return arrows


def _dominant_body_color(grid: np.ndarray, center):
    r, c = center
    h, w = grid.shape
    r0 = max(0, r - 2)
    r1 = min(h, r + 3)
    c0 = max(0, c - 2)
    c1 = min(w, c + 3)
    colors = []
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if abs(rr - r) <= 1 and abs(cc - c) <= 1:
                continue
            value = int(grid[rr, cc])
            if value not in (BG, YELLOW):
                colors.append(value)
    assert colors
    values, counts = np.unique(np.array(colors, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_objects(grid: np.ndarray):
    arrows = _detect_arrows(grid)
    assert arrows
    objects = []
    used_colors = set()
    for arrow in arrows:
        color = _dominant_body_color(grid, arrow["center"])
        assert color not in used_colors
        used_colors.add(color)
        body_coords = [tuple(map(int, pos)) for pos in np.argwhere(grid == color)]
        assert body_coords
        visible = set(body_coords) | set(arrow["arrow_cells"])
        rows = [r for r, _ in visible]
        cols = [c for _, c in visible]
        cr, cc = arrow["center"]
        half_h = max(cr - min(rows), max(rows) - cr)
        half_w = max(cc - min(cols), max(cols) - cc)
        r0 = cr - half_h
        c0 = cc - half_w
        r1 = cr + half_h
        c1 = cc + half_w
        visible_count = 0
        for rr in range(r0, r1 + 1):
            for cc2 in range(c0, c1 + 1):
                if (rr, cc2) in visible:
                    visible_count += 1
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        objects.append(
            {
                "color": color,
                "direction": arrow["direction"],
                "center": (cr, cc),
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "visible": visible,
                "hidden": area - visible_count,
            }
        )
    return objects


def _stack_order(objects):
    n = len(objects)
    edges = [set() for _ in range(n)]
    indeg = [0] * n
    visible_sets = [obj["visible"] for obj in objects]

    for i in range(n):
        for j in range(i + 1, n):
            inter = _bbox_intersection(objects[i]["bbox"], objects[j]["bbox"])
            if inter is None:
                continue
            i_seen = False
            j_seen = False
            for rr in range(inter[0], inter[2] + 1):
                for cc in range(inter[1], inter[3] + 1):
                    if (rr, cc) in visible_sets[i]:
                        i_seen = True
                    if (rr, cc) in visible_sets[j]:
                        j_seen = True
            if i_seen and not j_seen:
                if i not in edges[j]:
                    edges[j].add(i)
                    indeg[i] += 1
            elif j_seen and not i_seen:
                if j not in edges[i]:
                    edges[i].add(j)
                    indeg[j] += 1

    def sort_key(idx: int):
        r0, c0, _r1, _c1 = objects[idx]["bbox"]
        return (r0, c0, objects[idx]["color"])

    ready = [idx for idx in range(n) if indeg[idx] == 0]
    order = []
    while ready:
        ready.sort(key=sort_key)
        idx = ready.pop(0)
        order.append(idx)
        for nxt in sorted(edges[idx], key=sort_key):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    if len(order) < n:
        missing = [idx for idx in range(n) if idx not in order]
        order.extend(sorted(missing, key=sort_key))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    order = _stack_order(objects)
    out = np.full_like(g, BG)

    for idx in order:
        obj = objects[idx]
        r0, c0, _r1, _c1 = obj["bbox"]
        dr, dc = DIR_VECS[obj["direction"]]
        distance = int(obj["hidden"])
        patch = _render_patch(int(obj["height"]), int(obj["width"]), int(obj["color"]), obj["direction"])
        _stamp(out, patch, int(r0 + dr * distance), int(c0 + dc * distance))

    return out


def _build_pair_assembly(rng: np.random.Generator, mover_color: int, cover_color: int):
    mover_h = int(_choice(rng, MOVER_SIDES))
    mover_w = int(_choice(rng, MOVER_SIDES))
    cover_h = int(_choice(rng, COVER_SIDES))
    cover_w = int(_choice(rng, COVER_SIDES))
    mover_dir = _choice(rng, DIRS)
    cover_dir = _choice(rng, DIRS)

    corners = list(_compatible_corners(mover_dir))
    rng.shuffle(corners)
    overlap_dims = list(OVERLAP_DIMS)
    rng.shuffle(overlap_dims)

    for corner in corners:
        for oh, ow in overlap_dims:
            if oh > cover_h or ow > cover_w:
                continue
            if _overlap_hits_arrow(mover_h, mover_w, corner, oh, ow):
                continue
            move = int(oh * ow)
            dr, dc = DIR_VECS[mover_dir]
            mover_in = (0, 0)
            mover_out = (dr * move, dc * move)
            cover_in = _cover_position(mover_h, mover_w, cover_h, cover_w, corner, oh, ow)
            cover_out = cover_in

            coords = [
                (mover_in[0], mover_in[1], mover_h, mover_w),
                (mover_out[0], mover_out[1], mover_h, mover_w),
                (cover_in[0], cover_in[1], cover_h, cover_w),
            ]
            min_r = min(r for r, _c, _h, _w in coords)
            min_c = min(c for _r, c, _h, _w in coords)
            max_r = max(r + h - 1 for r, _c, h, _w in coords)
            max_c = max(c + w - 1 for _r, c, _h, w in coords)
            shift_r = -min_r
            shift_c = -min_c

            mover_in = (mover_in[0] + shift_r, mover_in[1] + shift_c)
            mover_out = (mover_out[0] + shift_r, mover_out[1] + shift_c)
            cover_in = (cover_in[0] + shift_r, cover_in[1] + shift_c)
            cover_out = (cover_out[0] + shift_r, cover_out[1] + shift_c)

            return {
                "height": max_r - min_r + 1,
                "width": max_c - min_c + 1,
                "objects": [
                    {
                        "color": mover_color,
                        "direction": mover_dir,
                        "height": mover_h,
                        "width": mover_w,
                        "input": mover_in,
                        "output": mover_out,
                        "z": 0,
                    },
                    {
                        "color": cover_color,
                        "direction": cover_dir,
                        "height": cover_h,
                        "width": cover_w,
                        "input": cover_in,
                        "output": cover_out,
                        "z": 1,
                    },
                ],
            }
    return None


def _build_single_assembly(rng: np.random.Generator, color: int):
    height = int(_choice(rng, STATIC_SIDES))
    width = int(_choice(rng, STATIC_SIDES))
    direction = _choice(rng, DIRS)
    return {
        "height": height,
        "width": width,
        "objects": [
            {
                "color": color,
                "direction": direction,
                "height": height,
                "width": width,
                "input": (0, 0),
                "output": (0, 0),
                "z": 0,
            }
        ],
    }


def _pack_assemblies(rng: np.random.Generator, assemblies):
    assemblies = list(assemblies)
    rng.shuffle(assemblies)
    count = len(assemblies)

    assignments = []
    if count == 1:
        assignments = [[assemblies]]
    else:
        for mask in range(1 << count):
            rows = [[], []]
            for idx, asm in enumerate(assemblies):
                rows[(mask >> idx) & 1].append(asm)
            rows = [row for row in rows if row]
            if not rows or len(rows) > 2:
                continue
            assignments.append(rows)

    rng.shuffle(assignments)
    for rows in assignments:
        row_widths = []
        row_heights = []
        for row in rows:
            width = sum(asm["width"] for asm in row) + MIN_GAP * (len(row) - 1)
            height = max(asm["height"] for asm in row)
            row_widths.append(width)
            row_heights.append(height)

        min_w = max(row_widths)
        min_h = sum(row_heights) + MIN_GAP * (len(rows) - 1)
        if min_w > MAX_GRID or min_h > MAX_GRID:
            continue

        extra_w = int(rng.integers(0, MAX_GRID - min_w + 1))
        extra_h = int(rng.integers(0, MAX_GRID - min_h + 1))
        width = min_w + extra_w
        height = min_h + extra_h

        if len(rows) == 1:
            top_margin = int(rng.integers(0, extra_h + 1))
            row_gap = []
        else:
            top_margin = int(rng.integers(0, extra_h + 1))
            row_gap = [MIN_GAP + (extra_h - top_margin)]

        placements = []
        y = top_margin
        ok = True
        for row_idx, row in enumerate(rows):
            row_h = row_heights[row_idx]
            row_w = row_widths[row_idx]
            x = int(rng.integers(0, width - row_w + 1))
            cursor = x
            for asm in row:
                dy = int(rng.integers(0, row_h - asm["height"] + 1))
                placements.append((asm, y + dy, cursor))
                cursor += asm["width"] + MIN_GAP
            y += row_h
            if row_idx < len(rows) - 1:
                y += row_gap[row_idx]
        if not ok:
            continue
        return height, width, placements
    return None


def _render_scene(shape, objects, key):
    grid = np.full(shape, BG, dtype=int)
    order = sorted(range(len(objects)), key=lambda idx: (objects[idx]["z"], idx))
    for idx in order:
        obj = objects[idx]
        r0, c0 = obj[key]
        patch = _render_patch(int(obj["height"]), int(obj["width"]), int(obj["color"]), obj["direction"])
        _stamp(grid, patch, int(r0), int(c0))
    return grid


def _gen_attempt(rng: np.random.Generator):
    total_objects = int(rng.integers(2, 5))
    colors = [int(c) for c in rng.choice(BODY_COLORS, size=total_objects, replace=False)]

    pair_assembly = _build_pair_assembly(rng, colors[0], colors[1])
    if pair_assembly is None:
        return None

    assemblies = [pair_assembly]
    for color in colors[2:]:
        assemblies.append(_build_single_assembly(rng, color))

    packed = _pack_assemblies(rng, assemblies)
    if packed is None:
        return None
    grid_h, grid_w, placements = packed

    objects = []
    for asm, top, left in placements:
        for obj in asm["objects"]:
            objects.append(
                {
                    "color": obj["color"],
                    "direction": obj["direction"],
                    "height": obj["height"],
                    "width": obj["width"],
                    "input": (obj["input"][0] + top, obj["input"][1] + left),
                    "output": (obj["output"][0] + top, obj["output"][1] + left),
                    "z": obj["z"],
                }
            )

    inp = _render_scene((grid_h, grid_w), objects, "input")
    out = _render_scene((grid_h, grid_w), objects, "output")

    if np.array_equal(inp, out):
        return None
    if len(_detect_arrows(inp)) != len(objects):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
