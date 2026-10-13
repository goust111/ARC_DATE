import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

DEPTH_COLORS = (RED, BLUE, GREEN, MAROON)
MAX_GENERATION_ATTEMPTS = 400
MAX_GROUP_ATTEMPTS = 120
MAX_PATTERN_ATTEMPTS = 100
MAX_GRID_SIDE = 30

_RNG = random.Random()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def bbox_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    h, w = bbox_shape(bbox)
    return h * w


def bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def ring_points(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def ring_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def transform_mask(mask: np.ndarray, depth: int) -> np.ndarray:
    if depth == 1:
        return mask.copy()
    if depth == 2:
        return np.fliplr(mask)
    if depth == 3:
        return np.rot90(mask, 2)
    if depth == 4:
        return np.flipud(mask)
    raise ValueError(f"unsupported depth: {depth}")


def crop_mask_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def render_layers(groups: list[list[tuple[int, np.ndarray]]], bg: int) -> np.ndarray:
    rendered_groups: list[np.ndarray] = []
    total_width = 0
    max_height = 0
    for layers in groups:
        gh = max(mask.shape[0] for _, mask in layers)
        gw = sum(mask.shape[1] for _, mask in layers) + max(0, len(layers) - 1)
        block = np.full((gh, gw), bg, dtype=int)
        col = 0
        for color, mask in layers:
            h, w = mask.shape
            block[:h, col : col + w][mask] = color
            col += w + 1
        rendered_groups.append(block)
        total_width += gw
        max_height = max(max_height, gh)
    total_width += max(0, len(rendered_groups) - 1) * 2
    out = np.full((max_height, total_width), bg, dtype=int)
    col = 0
    for block in rendered_groups:
        h, w = block.shape
        out[:h, col : col + w] = block
        col += w + 2
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for depth, color in enumerate(DEPTH_COLORS, start=1):
        for pts in components_mask(grid == color):
            bbox = bbox_coords(pts)
            h, w = bbox_shape(bbox)
            if h < 3 or w < 3:
                continue
            if set(pts) != ring_points(bbox):
                continue
            frames.append(
                {
                    "id": len(frames),
                    "depth": depth,
                    "color": color,
                    "bbox": bbox,
                    "area": len(pts),
                }
            )
    return frames


def _build_groups(frames: list[dict]) -> list[list[dict]]:
    by_id = {frame["id"]: frame for frame in frames}
    for frame in frames:
        parents = [other for other in frames if bbox_contains(other["bbox"], frame["bbox"])]
        if not parents:
            frame["parent"] = None
            continue
        parent = min(parents, key=lambda other: bbox_area(other["bbox"]))
        frame["parent"] = parent["id"]
    roots = [frame for frame in frames if frame["parent"] is None]
    groups: list[list[dict]] = []
    for root in roots:
        chain = [root]
        current = root
        while True:
            children = [frame for frame in frames if frame["parent"] == current["id"]]
            assert len(children) <= 1, "each frame should contain at most one direct child"
            if not children:
                break
            current = children[0]
            chain.append(current)
        expected = DEPTH_COLORS[: len(chain)]
        got = tuple(frame["color"] for frame in chain)
        assert got == expected, f"invalid depth-color chain: {got}"
        groups.append(chain)
    groups.sort(key=lambda chain: (chain[0]["bbox"][1], chain[0]["bbox"][0]))
    by_id.clear()
    return groups


def _layer_zone(shape: tuple[int, int], outer: tuple[int, int, int, int], child: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = shape
    zone = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = outer
    zone[r0 + 1 : r1, c0 + 1 : c1] = True
    if child is not None:
        cr0, cc0, cr1, cc1 = child
        zone[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
    return zone


def _extract_layer_masks(grid: np.ndarray) -> list[list[tuple[int, np.ndarray]]]:
    frames = _find_frames(grid)
    assert frames, "no nested frames found"
    groups = _build_groups(frames)
    h, w = grid.shape
    extracted: list[list[tuple[int, np.ndarray]]] = []
    for chain in groups:
        layers: list[tuple[int, np.ndarray]] = []
        for index, frame in enumerate(chain):
            child = chain[index + 1]["bbox"] if index + 1 < len(chain) else None
            zone = _layer_zone((h, w), frame["bbox"], child)
            yellow = np.argwhere(zone & (grid == YELLOW))
            assert len(yellow) > 0, "every layer must contain a yellow pattern"
            pts = [(int(r), int(c)) for r, c in yellow]
            sprite = crop_mask_from_points(pts)
            sprite = transform_mask(sprite, frame["depth"])
            layers.append((frame["color"], sprite))
        extracted.append(layers)
    return extracted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold nested colored frames by depth, transform each layer's yellow pattern,
    and lay the recolored results out horizontally.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    groups = _extract_layer_masks(g)
    return render_layers(groups, bg)


def _choose_depth_count(rng: random.Random) -> int:
    return rng.randint(1, 4)


def _grow_boxes(depth_count: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    inner_h = rng.randint(4, 7)
    inner_w = rng.randint(4, 7)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth_count - 1):
        child = boxes[-1]
        top = rng.randint(2, 4)
        bottom = rng.randint(2, 4)
        left = rng.randint(2, 4)
        right = rng.randint(2, 4)
        boxes.append((child[0] - top, child[1] - left, child[2] + bottom, child[3] + right))
    boxes = list(reversed(boxes))
    r_shift = -boxes[0][0]
    c_shift = -boxes[0][1]
    norm = []
    for r0, c0, r1, c1 in boxes:
        norm.append((r0 + r_shift, c0 + c_shift, r1 + r_shift, c1 + c_shift))
    oh, ow = bbox_shape(norm[0])
    if oh > 22 or ow > 22:
        return None
    return norm


def _random_connected_shape(allowed: np.ndarray, target: int, rng: random.Random) -> list[tuple[int, int]] | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    if len(coords) < target:
        return None
    start = rng.choice(coords)
    shape = {start}
    frontier = [start]
    while len(shape) < target and frontier:
        candidates: list[tuple[int, int]] = []
        for r, c in frontier:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc] and (nr, nc) not in shape:
                    candidates.append((nr, nc))
        if not candidates:
            break
        cell = rng.choice(candidates)
        shape.add(cell)
        frontier = list(shape)
    if len(shape) != target:
        return None
    return sorted(shape)


def _shape_ok(mask: np.ndarray, depth: int) -> bool:
    filled = int(mask.sum())
    if filled < 3:
        return False
    h, w = mask.shape
    if h * w > filled * 3:
        return False
    if depth == 1:
        return h >= 2 or w >= 2
    if depth == 2:
        return w >= 2 and not np.array_equal(mask, np.fliplr(mask))
    if depth == 3:
        return (h >= 2 or w >= 2) and not np.array_equal(mask, np.rot90(mask, 2))
    if depth == 4:
        return h >= 2 and not np.array_equal(mask, np.flipud(mask))
    return False


def _place_layer_pattern(
    zone: np.ndarray,
    depth: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    allowed_count = int(zone.sum())
    if allowed_count < 3:
        return None
    for _ in range(MAX_PATTERN_ATTEMPTS):
        target = rng.randint(3, min(7, allowed_count))
        pts = _random_connected_shape(zone, target, rng)
        if pts is None:
            continue
        sprite = crop_mask_from_points(pts)
        if not _shape_ok(sprite, depth):
            continue
        return pts, sprite
    return None


def _make_group(rng: random.Random) -> dict | None:
    depth_count = _choose_depth_count(rng)
    for _ in range(MAX_GROUP_ATTEMPTS):
        boxes = _grow_boxes(depth_count, rng)
        if boxes is None:
            continue
        h, w = bbox_shape(boxes[0])
        grid = np.full((h, w), BLACK, dtype=int)
        layers: list[dict] = []
        for depth, bbox in enumerate(boxes, start=1):
            r0, c0, r1, c1 = bbox
            grid[r0 : r1 + 1, c0 : c1 + 1][ring_mask((r1 - r0 + 1, c1 - c0 + 1))] = DEPTH_COLORS[depth - 1]
        ok = True
        for depth, bbox in enumerate(boxes, start=1):
            child = boxes[depth] if depth < depth_count else None
            zone = _layer_zone(grid.shape, bbox, child)
            placed = _place_layer_pattern(zone, depth, rng)
            if placed is None:
                ok = False
                break
            pts, sprite = placed
            for r, c in pts:
                grid[r, c] = YELLOW
            layers.append({"color": DEPTH_COLORS[depth - 1], "mask": sprite})
        if not ok:
            continue
        return {"grid": grid, "layers": layers}
    return None


def _partition_groups(count: int, row_count: int, rng: random.Random) -> list[list[int]]:
    order = list(range(count))
    rng.shuffle(order)
    rows = [[] for _ in range(row_count)]
    for i, idx in enumerate(order):
        rows[i % row_count].append(idx)
    for row in rows:
        rng.shuffle(row)
    return rows


def _pack_groups(groups: list[dict], rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    row_count = 1 if len(groups) == 1 else rng.randint(1, min(2, len(groups)))
    rows = _partition_groups(len(groups), row_count, rng)
    y = rng.randint(1, 3)
    max_right = 0
    placements: list[dict] = [None] * len(groups)  # type: ignore[list-item]
    for row in rows:
        if not row:
            continue
        x = rng.randint(1, 4)
        row_height = 0
        row_right = x
        for idx in row:
            gh, gw = groups[idx]["grid"].shape
            placements[idx] = {"top": y, "left": x}
            row_right = x + gw
            x += gw + rng.randint(2, 4)
            row_height = max(row_height, gh)
        max_right = max(max_right, row_right)
        y += row_height + rng.randint(2, 5)
    bottom = y + rng.randint(0, 2)
    width = max_right + rng.randint(1, 4)
    height = bottom + rng.randint(1, 3)
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None
    canvas = np.full((height, width), BLACK, dtype=int)
    specs: list[dict] = []
    for idx, group in enumerate(groups):
        top = placements[idx]["top"]
        left = placements[idx]["left"]
        gh, gw = group["grid"].shape
        canvas[top : top + gh, left : left + gw] = group["grid"]
        specs.append({"top": top, "left": left, "layers": group["layers"]})
    return canvas, specs


def _expected_from_specs(specs: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: (spec["left"], spec["top"]))
    groups: list[list[tuple[int, np.ndarray]]] = []
    for spec in ordered:
        layers: list[tuple[int, np.ndarray]] = []
        for depth, layer in enumerate(spec["layers"], start=1):
            layers.append((layer["color"], transform_mask(layer["mask"], depth)))
        groups.append(layers)
    return render_layers(groups, bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.randint(1, 3)
    groups = []
    for _ in range(group_count):
        group = _make_group(rng)
        if group is None:
            return None
        groups.append(group)
    packed = _pack_groups(groups, rng)
    if packed is None:
        return None
    grid, specs = packed
    colors = [BLACK, YELLOW, *DEPTH_COLORS]
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    if counts[BLACK] <= max(count for color, count in counts.items() if color != BLACK):
        return None
    expected = _expected_from_specs(specs, BLACK)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame task input")
