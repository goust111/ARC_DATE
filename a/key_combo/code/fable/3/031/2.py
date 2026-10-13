from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]

BACKGROUND = 0
MAX_GRID_SIZE = 30
GEN_RETRY_LIMIT = 400
PAIR_RETRY_LIMIT = 200
LOCAL_SHAPE_RETRY_LIMIT = 200
MASTER_CANVAS_SIZE = 30
PLACEMENT_PADDING = 2


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: set[Coord] | list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: set[Coord] | list[Coord]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_to_coords(mask: np.ndarray, offset: Coord = (0, 0)) -> set[Coord]:
    r0, c0 = offset
    coords = set()
    for r, c in np.argwhere(mask):
        coords.add((int(r) + r0, int(c) + c0))
    return coords


def _is_connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nbr in _neighbors4(r, c):
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return len(seen) == len(coords)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            coords: set[Coord] = set()
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                coords.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _enclosed_regions_from_blocked(blocked_coords: set[Coord] | list[Coord]) -> list[set[Coord]]:
    if not blocked_coords:
        return []
    blocked = set(blocked_coords)
    r0, c0, r1, c1 = _bbox(blocked)
    h = r1 - r0 + 3
    w = c1 - c0 + 3
    local = np.zeros((h, w), dtype=bool)
    for r, c in blocked:
        local[r - r0 + 1, c - c0 + 1] = True

    outside = np.zeros_like(local, dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not local[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not local[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not local[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    hidden = (~local) & (~outside)
    regions: list[set[Coord]] = []
    seen = np.zeros_like(hidden, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not hidden[r, c] or seen[r, c]:
                continue
            region_local = {(r, c)}
            seen[r, c] = True
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and hidden[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        region_local.add((nr, nc))
                        queue.append((nr, nc))
            region_global = {(rr + r0 - 1, cc + c0 - 1) for rr, cc in region_local}
            regions.append(region_global)
    return regions


def _wall_completion_options(comp_coords: set[Coord], shape: tuple[int, int], allowed_mask: np.ndarray) -> list[dict]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(comp_coords)
    candidate_cells: set[Coord] = set()
    for r, c in comp_coords:
        for nr, nc in _neighbors4(r, c):
            if max(r0 - 1, 0) <= nr <= min(r1 + 1, h - 1) and max(c0 - 1, 0) <= nc <= min(c1 + 1, w - 1):
                if (nr, nc) not in comp_coords:
                    candidate_cells.add((nr, nc))

    options: list[dict] = []
    for gap in candidate_cells:
        gr, gc = gap
        if not allowed_mask[gr, gc]:
            continue
        neighbor_count = sum((nr, nc) in comp_coords for nr, nc in _neighbors4(gr, gc))
        if neighbor_count < 2:
            continue
        completed = set(comp_coords)
        completed.add(gap)
        if not _is_connected(completed):
            continue
        interior_regions = _enclosed_regions_from_blocked(completed)
        if len(interior_regions) != 1:
            continue
        interior = interior_regions[0]
        if not interior:
            continue
        options.append({"gap": gap, "completed": completed, "interior": interior})
    return options


def _extract_walls(grid: np.ndarray, bg: int) -> list[dict]:
    allowed_mask = grid == bg
    walls: list[dict] = []
    for comp in _find_components(grid, bg):
        options = _wall_completion_options(comp["coords"], grid.shape, allowed_mask)
        if len(options) != 1:
            continue
        option = options[0]
        completed_mask, bbox = _coords_to_mask(option["completed"])
        content: list[tuple[int, int, int]] = []
        for r, c in sorted(option["interior"]):
            color = int(grid[r, c])
            if color != bg:
                content.append((r, c, color))
        walls.append(
            {
                "wall_color": int(comp["color"]),
                "open_coords": set(comp["coords"]),
                "completed_coords": option["completed"],
                "gap": option["gap"],
                "interior": option["interior"],
                "content": content,
                "complete_mask": completed_mask,
                "bbox": bbox,
            }
        )
    walls.sort(key=lambda info: (info["bbox"][0], info["bbox"][1], info["bbox"][2], info["bbox"][3], info["wall_color"]))
    return walls


def _mirror_axis(src_mask: np.ndarray, dst_mask: np.ndarray) -> str | None:
    if src_mask.shape != dst_mask.shape:
        return None
    hits: list[str] = []
    if np.array_equal(np.fliplr(src_mask), dst_mask):
        hits.append("h")
    if np.array_equal(np.flipud(src_mask), dst_mask):
        hits.append("v")
    if len(hits) == 1:
        return hits[0]
    return None


def _map_point(point: Coord, src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int], axis: str) -> Coord:
    sr0, sc0, sr1, sc1 = src_bbox
    dr0, dc0, dr1, dc1 = dst_bbox
    rr = point[0] - sr0
    cc = point[1] - sc0
    if axis == "h":
        return dr0 + rr, dc0 + (dc1 - dc0) - cc
    if axis == "v":
        return dr0 + (dr1 - dr0) - rr, dc0 + cc
    raise ValueError(f"unknown axis: {axis}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mirrored broken ring-walls, repair each gap, and merge the split interior pattern into both containers.

    Input structure:
    - A black background contains an even number of separate single-color ring-wall containers, each missing exactly one wall pixel.
    - Each wall shape is itself asymmetric, but every container has exactly one partner whose completed wall is its left-right or up-down mirror after translation.
    - Inside each paired container there are some non-wall pixels that represent complementary halves of one pattern, mirrored across the same wall-defined transform.

    Rule:
    - Identify each wall component by finding the unique background pixel whose addition closes it into one enclosed container.
    - Pair containers whose completed wall masks become identical under one mirror direction, and use the wall correspondence to determine the unique mirror transform between their interiors.
    - Repair every wall gap, map each container's interior pixels through that transform into its partner, and union the mapped pixels with the partner's existing contents in both directions.

    Key insight:
    - The walls, not the interior colors, define both the pairing and the exact mirror alignment: once the missing wall pixel is restored, the same mirror transform simultaneously explains gap repair and content completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = BACKGROUND
    walls = _extract_walls(g, bg)
    assert walls and len(walls) % 2 == 0, "expected an even number of mirror-paired wall containers"

    matches: dict[int, tuple[int, str]] = {}
    for i, src in enumerate(walls):
        options: list[tuple[int, str]] = []
        for j, dst in enumerate(walls):
            if i == j:
                continue
            axis = _mirror_axis(src["complete_mask"], dst["complete_mask"])
            if axis is not None:
                options.append((j, axis))
        assert len(options) == 1, "each container must have exactly one mirrored partner"
        matches[i] = options[0]

    out = g.copy()
    for wall in walls:
        for r, c in wall["completed_coords"]:
            out[r, c] = wall["wall_color"]

    paired: set[int] = set()
    for i, (j, axis) in matches.items():
        if i in paired:
            continue
        reverse = matches[j]
        assert reverse[0] == i and reverse[1] == axis, "pairing must be symmetric"
        a = walls[i]
        b = walls[j]
        for r, c, color in a["content"]:
            rr, cc = _map_point((r, c), a["bbox"], b["bbox"], axis)
            out[rr, cc] = color
        for r, c, color in b["content"]:
            rr, cc = _map_point((r, c), b["bbox"], a["bbox"], axis)
            out[rr, cc] = color
        paired.add(i)
        paired.add(j)

    assert len(paired) == len(walls)
    return out


def _connected_mask(mask: np.ndarray) -> bool:
    return _is_connected(_mask_to_coords(mask))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _boundary_mask(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    wall = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not solid[nr, nc]:
                    wall[r, c] = True
                    break
    return wall


def _has_hole(mask: np.ndarray) -> bool:
    return bool(_enclosed_regions_from_blocked(_mask_to_coords(mask)))


def _mirror_local_coords(coords: set[Coord], shape: tuple[int, int], axis: str) -> set[Coord]:
    h, w = shape
    if axis == "h":
        return {(r, w - 1 - c) for r, c in coords}
    if axis == "v":
        return {(h - 1 - r, c) for r, c in coords}
    raise ValueError(f"unknown axis: {axis}")


def _deep_interior(interior: set[Coord]) -> set[Coord]:
    deep: set[Coord] = set()
    for r, c in interior:
        if all(nbr in interior for nbr in _neighbors4(r, c)):
            deep.add((r, c))
    return deep


def _grow_connected_subset(pool: set[Coord], target_size: int) -> set[Coord] | None:
    cells = list(pool)
    random.shuffle(cells)
    for seed in cells:
        chosen = {seed}
        while len(chosen) < target_size:
            frontier = [
                cell
                for cell in pool
                if cell not in chosen and any(nbr in chosen for nbr in _neighbors4(*cell))
            ]
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) == target_size:
            return chosen
    return None


def _choose_gap(wall_mask: np.ndarray) -> Coord | None:
    full_coords = _mask_to_coords(wall_mask)
    gap_choices: list[Coord] = []
    for gap in full_coords:
        open_coords = set(full_coords)
        open_coords.remove(gap)
        if not _is_connected(open_coords):
            continue
        allowed = np.ones(wall_mask.shape, dtype=bool)
        for r, c in open_coords:
            allowed[r, c] = False
        options = _wall_completion_options(open_coords, wall_mask.shape, allowed)
        if len(options) == 1 and options[0]["gap"] == gap:
            gap_choices.append(gap)
    if not gap_choices:
        return None
    return random.choice(gap_choices)


def _canonical_shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [mask, np.fliplr(mask), np.flipud(mask)]
    payload = min(np.asarray(v, dtype=np.uint8).tobytes() for v in variants)
    return mask.shape, payload


def _random_notched_solid() -> np.ndarray | None:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    solid = np.ones((h, w), dtype=bool)
    notch_count = random.randint(1, 3)

    for _ in range(notch_count):
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            max_depth = min(3, h - 4)
            max_length = w - 2
            if max_depth < 1 or max_length < 1:
                return None
            depth = random.randint(1, max_depth)
            length = random.randint(1, max_length)
            start = random.randint(1, w - length - 1)
            if side == "top":
                solid[:depth, start : start + length] = False
            else:
                solid[h - depth :, start : start + length] = False
        else:
            max_depth = min(3, w - 4)
            max_length = h - 2
            if max_depth < 1 or max_length < 1:
                return None
            depth = random.randint(1, max_depth)
            length = random.randint(1, max_length)
            start = random.randint(1, h - length - 1)
            if side == "left":
                solid[start : start + length, :depth] = False
            else:
                solid[start : start + length, w - depth :] = False
    if not solid.any():
        return None
    return _trim_mask(solid)


def _make_pair(axis: str, wall_color: int, pattern_color: int, seen_signatures: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(PAIR_RETRY_LIMIT):
        solid = _random_notched_solid()
        if solid is None or not _connected_mask(solid) or _has_hole(solid):
            continue
        wall = _boundary_mask(solid)
        if not _connected_mask(wall):
            continue
        if np.array_equal(wall, np.fliplr(wall)) or np.array_equal(wall, np.flipud(wall)):
            continue
        signature = _canonical_shape_signature(wall)
        if signature in seen_signatures:
            continue

        interior_regions = _enclosed_regions_from_blocked(_mask_to_coords(wall))
        if len(interior_regions) != 1:
            continue
        interior = interior_regions[0]
        if len(interior) < 6:
            continue
        deep = _deep_interior(interior)
        pool = deep if len(deep) >= 4 else interior
        if len(pool) < 3:
            continue

        target_size = random.randint(3, min(8, len(pool)))
        pattern_full = _grow_connected_subset(pool, target_size)
        if pattern_full is None:
            continue

        gap_a = _choose_gap(wall)
        if gap_a is None:
            continue

        wall_b = np.fliplr(wall) if axis == "h" else np.flipud(wall)
        gap_b = next(iter(_mirror_local_coords({gap_a}, wall.shape, axis)))
        wall_a_open = wall.copy()
        wall_b_open = wall_b.copy()
        wall_a_open[gap_a] = False
        wall_b_open[gap_b] = False

        cells = list(pattern_full)
        random.shuffle(cells)
        split = random.randint(1, len(cells) - 1)
        a_half = set(cells[:split])
        a_missing = set(cells[split:])
        b_full = _mirror_local_coords(pattern_full, wall.shape, axis)
        b_half = _mirror_local_coords(a_missing, wall.shape, axis)

        seen_signatures.add(signature)
        return {
            "axis": axis,
            "wall_color": wall_color,
            "pattern_color": pattern_color,
            "shape": wall.shape,
            "wall_full_a": wall,
            "wall_open_a": wall_a_open,
            "wall_full_b": wall_b,
            "wall_open_b": wall_b_open,
            "pattern_full_a": pattern_full,
            "pattern_half_a": a_half,
            "pattern_full_b": b_full,
            "pattern_half_b": b_half,
        }
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_rectangles(rect_sizes: list[tuple[int, int]]) -> list[Coord] | None:
    placements: list[Coord] = []
    padded_boxes: list[tuple[int, int, int, int]] = []
    for h, w in rect_sizes:
        options: list[Coord] = []
        for r0 in range(MASTER_CANVAS_SIZE - h + 1):
            for c0 in range(MASTER_CANVAS_SIZE - w + 1):
                box = (
                    max(0, r0 - PLACEMENT_PADDING),
                    max(0, c0 - PLACEMENT_PADDING),
                    min(MASTER_CANVAS_SIZE - 1, r0 + h - 1 + PLACEMENT_PADDING),
                    min(MASTER_CANVAS_SIZE - 1, c0 + w - 1 + PLACEMENT_PADDING),
                )
                if all(not _boxes_overlap(box, other) for other in padded_boxes):
                    options.append((r0, c0))
        if not options:
            return None
        choice = random.choice(options)
        placements.append(choice)
        r0, c0 = choice
        padded_boxes.append(
            (
                max(0, r0 - PLACEMENT_PADDING),
                max(0, c0 - PLACEMENT_PADDING),
                min(MASTER_CANVAS_SIZE - 1, r0 + h - 1 + PLACEMENT_PADDING),
                min(MASTER_CANVAS_SIZE - 1, c0 + w - 1 + PLACEMENT_PADDING),
            )
        )
    return placements


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top_left: Coord, color: int) -> None:
    r0, c0 = top_left
    rows, cols = np.where(mask)
    grid[r0 + rows, c0 + cols] = color


def _draw_cells(grid: np.ndarray, cells: set[Coord], top_left: Coord, color: int) -> None:
    r0, c0 = top_left
    for r, c in cells:
        grid[r0 + r, c0 + c] = color


def _crop_with_margin(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BACKGROUND)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    top = random.randint(0, min(3, r0))
    left = random.randint(0, min(3, c0))
    bottom = random.randint(0, min(3, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(3, inp.shape[1] - 1 - c1))

    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _render_pair_scene(pair_specs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    rects: list[tuple[int, int]] = []
    container_refs: list[tuple[int, str]] = []
    for idx, spec in enumerate(pair_specs):
        rects.append(spec["shape"])
        container_refs.append((idx, "a"))
        rects.append(spec["shape"])
        container_refs.append((idx, "b"))

    placements = _place_rectangles(rects)
    if placements is None:
        return None

    inp = np.full((MASTER_CANVAS_SIZE, MASTER_CANVAS_SIZE), BACKGROUND, dtype=int)
    out = np.full((MASTER_CANVAS_SIZE, MASTER_CANVAS_SIZE), BACKGROUND, dtype=int)

    for (pair_idx, side), pos in zip(container_refs, placements):
        spec = pair_specs[pair_idx]
        wall_color = spec["wall_color"]
        pattern_color = spec["pattern_color"]
        if side == "a":
            _draw_mask(inp, spec["wall_open_a"], pos, wall_color)
            _draw_mask(out, spec["wall_full_a"], pos, wall_color)
            _draw_cells(inp, spec["pattern_half_a"], pos, pattern_color)
            _draw_cells(out, spec["pattern_full_a"], pos, pattern_color)
        else:
            _draw_mask(inp, spec["wall_open_b"], pos, wall_color)
            _draw_mask(out, spec["wall_full_b"], pos, wall_color)
            _draw_cells(inp, spec["pattern_half_b"], pos, pattern_color)
            _draw_cells(out, spec["pattern_full_b"], pos, pattern_color)

    return _crop_with_margin(inp, out)


def generate_input() -> np.ndarray:
    seen_signatures: set[tuple[tuple[int, int], bytes]]
    for _ in range(GEN_RETRY_LIMIT):
        pair_count = 2 if random.random() < 0.45 else 1
        axes = [random.choice(("h", "v")) for _ in range(pair_count)]
        if pair_count == 2 and random.random() < 0.7:
            axes = ["h", "v"]
            random.shuffle(axes)

        palette = random.sample(range(1, 10), 1 + pair_count)
        wall_color = palette[0]
        pattern_colors = palette[1:]

        seen_signatures = set()
        pair_specs: list[dict] = []
        for axis, pattern_color in zip(axes, pattern_colors):
            spec = _make_pair(axis, wall_color, pattern_color, seen_signatures)
            if spec is None:
                break
            pair_specs.append(spec)
        if len(pair_specs) != pair_count:
            continue

        rendered = _render_pair_scene(pair_specs)
        if rendered is None:
            continue
        inp, expected = rendered
        if inp.ndim != 2 or expected.ndim != 2:
            continue
        if max(inp.shape) > MAX_GRID_SIZE or max(expected.shape) > MAX_GRID_SIZE:
            continue

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
