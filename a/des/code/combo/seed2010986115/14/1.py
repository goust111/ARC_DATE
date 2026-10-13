import random
from collections import deque

import numpy as np


Shape = tuple[tuple[int, int], ...]


SHAPES: tuple[Shape, ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)

BG = 0
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 30


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "id": cid,
                    "color": color,
                    "coords": coords,
                    "coord_set": set((int(rr), int(cc)) for rr, cc in coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
            cid += 1
    return comps


def _component_is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _hole_signature(grid: np.ndarray, bg: int, comp: dict) -> tuple[tuple[int, int], ...] | None:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(comp["color"])

    if np.any((sub != color) & (sub != bg)):
        return None
    if not np.all(sub[0, :] == color):
        return None
    if not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:, 0] == color):
        return None
    if not np.all(sub[:, -1] == color):
        return None

    hole_mask = sub == bg
    if not hole_mask.any():
        return None

    hole_comps = _components_mask(hole_mask)
    if len(hole_comps) != 1:
        return None

    hole = hole_comps[0]
    if np.any(hole[:, 0] == 0) or np.any(hole[:, 0] == sub.shape[0] - 1):
        return None
    if np.any(hole[:, 1] == 0) or np.any(hole[:, 1] == sub.shape[1] - 1):
        return None
    return _normalize(hole)


def _components_touch(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    if ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1:
        return False

    aset = a["coord_set"]
    bset = b["coord_set"]
    if len(aset) > len(bset):
        aset, bset = bset, aset
    for r, c in aset:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _legend_mapping(components: list[dict], object_ids: set[int]) -> dict[tuple[tuple[int, int], ...], int]:
    small = [comp for comp in components if comp["id"] not in object_ids]
    graph = {comp["id"]: set() for comp in small}
    by_id = {comp["id"]: comp for comp in small}

    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            if _components_touch(small[i], small[j]):
                graph[small[i]["id"]].add(small[j]["id"])
                graph[small[j]["id"]].add(small[i]["id"])

    seen: set[int] = set()
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in small:
        cid = comp["id"]
        if cid in seen:
            continue
        stack = [cid]
        cluster: list[int] = []
        seen.add(cid)
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        if len(cluster) != 2:
            continue

        c0 = by_id[cluster[0]]
        c1 = by_id[cluster[1]]
        rect0 = _component_is_solid_rect(c0)
        rect1 = _component_is_solid_rect(c1)
        if rect0 and not rect1:
            block, shape = c0, c1
        elif rect1 and not rect0:
            block, shape = c1, c0
        else:
            block, shape = (c0, c1) if c0["area"] >= c1["area"] else (c1, c0)

        signature = _normalize(shape["coords"])
        mapping[signature] = int(block["color"])

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each object's internal background hole against the edge legend's small shapes,
    then recolor that object's outer rectangular border using the legend's paired color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _color_components(g, bg)

    objects: list[dict] = []
    object_ids: set[int] = set()
    for comp in components:
        signature = _hole_signature(g, bg, comp)
        if signature is None:
            continue
        item = dict(comp)
        item["hole_signature"] = signature
        objects.append(item)
        object_ids.add(int(comp["id"]))

    mapping = _legend_mapping(components, object_ids)
    assert objects, "expected at least one rectangular object with an internal hole"
    assert mapping, "expected at least one legend entry"

    out = g.copy()
    for obj in objects:
        signature = obj["hole_signature"]
        assert signature in mapping, "hole shape must appear in legend"
        color = int(mapping[signature])
        r0, c0, r1, c1 = obj["bbox"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _draw_shape(grid: np.ndarray, top: int, left: int, shape: Shape, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _draw_object(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    frame_color: int,
    shape: Shape,
    hole_top: int,
    hole_left: int,
) -> None:
    grid[top : top + height, left : left + width] = frame_color
    for dr, dc in shape:
        grid[top + hole_top + dr, left + hole_left + dc] = BG


def _draw_left_legend(
    grid: np.ndarray,
    top: int,
    entries: list[tuple[Shape, int]],
    shape_color: int,
    gaps: list[int],
) -> int:
    r = top
    max_width = 0
    for idx, (shape, block_color) in enumerate(entries):
        sh, sw = _shape_dims(shape)
        _draw_shape(grid, r, 0, shape, shape_color)
        grid[r : r + sh, sw : sw + 2] = block_color
        max_width = max(max_width, sw + 2)
        r += sh
        if idx < len(gaps):
            r += gaps[idx]
    return max_width


def _draw_bottom_legend(
    grid: np.ndarray,
    left: int,
    entries: list[tuple[Shape, int]],
    shape_color: int,
    gaps: list[int],
) -> tuple[int, int]:
    shape_heights = [_shape_dims(shape)[0] for shape, _ in entries]
    band_h = max(shape_heights) + 2
    top = grid.shape[0] - band_h
    c = left
    for idx, (shape, block_color) in enumerate(entries):
        sh, sw = _shape_dims(shape)
        block_w = max(sw, 2)
        shape_top = top + band_h - 2 - sh
        _draw_shape(grid, shape_top, c, shape, shape_color)
        grid[top + band_h - 2 : top + band_h, c : c + block_w] = block_color
        c += block_w
        if idx < len(gaps):
            c += gaps[idx]
    return top, c - left


def _expected_output(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        color = int(obj["target_color"])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _partition_rows(count: int, rng: random.Random) -> list[list[int]]:
    order = list(range(count))
    rng.shuffle(order)
    if count == 2:
        return [order]
    if count == 3:
        return [order] if rng.random() < 0.5 else [order[:1], order[1:]]
    if rng.random() < 0.5:
        return [order]
    cut = rng.randint(1, count - 1)
    return [order[:cut], order[cut:]]


def _rand_with_margin(rng: random.Random, lo: int, hi: int, margin: int = 1) -> int:
    if hi - lo >= 2 * margin:
        return rng.randint(lo + margin, hi - margin)
    return rng.randint(lo, hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(1, 10))
    frame_color, shape_color = rng.sample(palette, 2)
    remaining = [c for c in palette if c not in {frame_color, shape_color}]

    object_count = rng.randint(2, 4)
    max_legend = min(len(SHAPES), object_count + 2, len(remaining))
    legend_count = rng.randint(object_count, max_legend)
    legend_shapes = list(rng.sample(list(SHAPES), legend_count))
    object_shapes = rng.sample(legend_shapes, object_count)
    legend_colors = rng.sample(remaining, legend_count)
    color_by_shape = {shape: color for shape, color in zip(legend_shapes, legend_colors)}
    entries = list(zip(legend_shapes, legend_colors))
    rng.shuffle(entries)

    orientation = rng.choice(["left", "bottom"])
    legend_gaps = [rng.randint(1, 2) for _ in range(legend_count - 1)]

    object_specs: list[dict] = []
    for shape in object_shapes:
        sh, sw = _shape_dims(shape)
        height = rng.randint(sh + 3, sh + 6)
        width = rng.randint(sw + 3, sw + 6)
        hole_top = rng.randint(1, height - sh - 1)
        hole_left = rng.randint(1, width - sw - 1)
        object_specs.append(
            {
                "shape": shape,
                "height": height,
                "width": width,
                "hole_top": hole_top,
                "hole_left": hole_left,
                "target_color": color_by_shape[shape],
            }
        )

    row_groups = _partition_rows(len(object_specs), rng)
    row_heights: list[int] = []
    row_widths: list[int] = []
    row_gaps: list[list[int]] = []
    for group in row_groups:
        gaps = [rng.randint(2, 4) for _ in range(len(group) - 1)]
        row_gaps.append(gaps)
        row_heights.append(max(object_specs[idx]["height"] for idx in group))
        row_widths.append(
            sum(object_specs[idx]["width"] for idx in group) + sum(gaps)
        )

    pack_h = sum(row_heights)
    if len(row_groups) > 1:
        vgap = rng.randint(2, 4)
        pack_h += vgap
    else:
        vgap = 0
    pack_w = max(row_widths)

    if orientation == "left":
        legend_h = sum(_shape_dims(shape)[0] for shape, _ in entries) + sum(legend_gaps)
        legend_w = max(_shape_dims(shape)[1] + 2 for shape, _ in entries)
        sep = rng.randint(2, 4)
        min_h = max(legend_h, pack_h) + rng.randint(0, 4)
        min_w = legend_w + sep + pack_w + rng.randint(2, 5)
    else:
        legend_w = sum(max(_shape_dims(shape)[1], 2) for shape, _ in entries) + sum(legend_gaps)
        legend_h = max(_shape_dims(shape)[0] for shape, _ in entries) + 2
        sep = rng.randint(2, 4)
        min_h = pack_h + sep + legend_h + rng.randint(2, 5)
        min_w = max(legend_w, pack_w) + rng.randint(0, 4)

    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h_lo = max(GRID_MIN, min_h)
    grid_h_hi = min(GRID_MAX, max(grid_h_lo, min_h + 4))
    grid_w_lo = max(GRID_MIN, min_w)
    grid_w_hi = min(GRID_MAX, max(grid_w_lo, min_w + 4))
    grid_h = rng.randint(grid_h_lo, grid_h_hi)
    grid_w = rng.randint(grid_w_lo, grid_w_hi)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    if orientation == "left":
        legend_top = _rand_with_margin(rng, 0, grid_h - legend_h, margin=1)
        _draw_left_legend(grid, legend_top, entries, shape_color, legend_gaps)
        pack_r0 = _rand_with_margin(rng, 0, grid_h - pack_h, margin=1)
        pack_c0 = _rand_with_margin(rng, legend_w + sep, grid_w - pack_w, margin=1)
    else:
        legend_left = _rand_with_margin(rng, 0, grid_w - legend_w, margin=1)
        legend_top, _ = _draw_bottom_legend(grid, legend_left, entries, shape_color, legend_gaps)
        usable_h = legend_top - sep
        if usable_h < pack_h:
            return None
        pack_r0 = _rand_with_margin(rng, 0, usable_h - pack_h, margin=1)
        pack_c0 = _rand_with_margin(rng, 0, grid_w - pack_w, margin=1)

    cur_r = pack_r0
    for row_idx, group in enumerate(row_groups):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        row_start_c = pack_c0 + rng.randint(0, pack_w - row_w)
        cur_c = row_start_c
        for pos, obj_idx in enumerate(group):
            spec = object_specs[obj_idx]
            obj_r = cur_r + rng.randint(0, row_h - spec["height"])
            obj_c = cur_c
            _draw_object(
                grid,
                obj_r,
                obj_c,
                spec["height"],
                spec["width"],
                frame_color,
                spec["shape"],
                spec["hole_top"],
                spec["hole_left"],
            )
            spec["bbox"] = (obj_r, obj_c, obj_r + spec["height"] - 1, obj_c + spec["width"] - 1)
            cur_c += spec["width"]
            if pos < len(row_gaps[row_idx]):
                cur_c += row_gaps[row_idx][pos]
        cur_r += row_h
        if row_idx + 1 < len(row_groups):
            cur_r += vgap

    if np.count_nonzero(grid != BG) >= grid.size * 0.45:
        return None

    expected = _expected_output(grid, object_specs)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(produced, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
