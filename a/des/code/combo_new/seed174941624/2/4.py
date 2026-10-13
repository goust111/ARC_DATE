from __future__ import annotations

import random

import numpy as np


CORNER_TO_DIRECTION = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
CORNER_ORDER = ("tl", "tr", "bl", "br")
LAYOUT_PAIR_COUNTS = (1, 2, 3)
GRID_MAX = 30
OBJECT_SIZE_RANGE = (2, 5)
PAIR_GAP_RANGE = (3, 8)
PAIR_PLACEMENT_TRIES = 200
GENERATE_RETRIES = 500
PALETTE = tuple(range(10))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))

    return parts


def _missing_offset(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 1
    if corner == "bl":
        return height - 1, 0
    if corner == "br":
        return height - 1, width - 1
    raise ValueError(f"unknown corner: {corner}")


def _opposite_offset(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return height - 1, width - 1
    if corner == "tr":
        return height - 1, 0
    if corner == "bl":
        return 0, width - 1
    if corner == "br":
        return 0, 0
    raise ValueError(f"unknown corner: {corner}")


def _top_left_from_missing(
    missing: tuple[int, int],
    height: int,
    width: int,
    corner: str,
) -> tuple[int, int]:
    mr, mc = map(int, missing)
    if corner == "tl":
        return mr, mc
    if corner == "tr":
        return mr, mc - width + 1
    if corner == "bl":
        return mr - height + 1, mc
    if corner == "br":
        return mr - height + 1, mc - width + 1
    raise ValueError(f"unknown corner: {corner}")


def _top_left_from_opposite(
    hit: tuple[int, int],
    height: int,
    width: int,
    corner: str,
) -> tuple[int, int]:
    hr, hc = map(int, hit)
    if corner == "tl":
        return hr - height + 1, hc - width + 1
    if corner == "tr":
        return hr - height + 1, hc
    if corner == "bl":
        return hr, hc - width + 1
    if corner == "br":
        return hr, hc
    raise ValueError(f"unknown corner: {corner}")


def _build_spec(
    top: int,
    left: int,
    height: int,
    width: int,
    corner: str,
    color: int,
    expected_target: int | None,
) -> dict:
    return {
        "top": int(top),
        "left": int(left),
        "height": int(height),
        "width": int(width),
        "corner": str(corner),
        "color": int(color),
        "expected_target": None if expected_target is None else int(expected_target),
    }


def _object_cells(spec: dict) -> list[tuple[int, int]]:
    r0 = int(spec["top"])
    c0 = int(spec["left"])
    height = int(spec["height"])
    width = int(spec["width"])
    miss_r, miss_c = _missing_offset(height, width, str(spec["corner"]))
    cells: list[tuple[int, int]] = []
    for dr in range(height):
        for dc in range(width):
            if dr == miss_r and dc == miss_c:
                continue
            cells.append((r0 + dr, c0 + dc))
    return cells


def _missing_cell(spec: dict) -> tuple[int, int]:
    r0 = int(spec["top"])
    c0 = int(spec["left"])
    miss_r, miss_c = _missing_offset(int(spec["height"]), int(spec["width"]), str(spec["corner"]))
    return r0 + miss_r, c0 + miss_c


def _opposite_cell(spec: dict) -> tuple[int, int]:
    r0 = int(spec["top"])
    c0 = int(spec["left"])
    off_r, off_c = _opposite_offset(int(spec["height"]), int(spec["width"]), str(spec["corner"]))
    return r0 + off_r, c0 + off_c


def _in_bounds(spec: dict, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    r0 = int(spec["top"])
    c0 = int(spec["left"])
    r1 = r0 + int(spec["height"]) - 1
    c1 = c0 + int(spec["width"]) - 1
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _touch_box(spec: dict, shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    height, width = map(int, shape)
    r0 = max(0, int(spec["top"]) - pad)
    c0 = max(0, int(spec["left"]) - pad)
    r1 = min(height - 1, int(spec["top"]) + int(spec["height"]) - 1 + pad)
    c1 = min(width - 1, int(spec["left"]) + int(spec["width"]) - 1 + pad)
    return r0, c0, r1, c1


def _blocks_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _ray_cells_from(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    dr, dc = map(int, direction)
    r = int(start[0] + dr)
    c = int(start[1] + dc)
    cells: list[tuple[int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _ray_prefix_to_hit(
    start: tuple[int, int],
    hit: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    r = int(start[0] + dr)
    c = int(start[1] + dc)
    cells: list[tuple[int, int]] = []
    while (r, c) != (int(hit[0]), int(hit[1])):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _place_spec(grid: np.ndarray, spec: dict) -> None:
    for r, c in _object_cells(spec):
        grid[r, c] = int(spec["color"])


def _analyze_component(g: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 2 and width >= 2, "objects must be at least 2x2 before corner removal"

    window = g[r0 : r1 + 1, c0 : c1 + 1]
    color_mask = window == int(color)
    assert int(color_mask.sum()) == height * width - 1, "object bbox must be a rectangle missing one cell"
    missing = np.argwhere(~color_mask)
    assert missing.shape[0] == 1, "object must have exactly one missing cell"

    miss_r = int(missing[0, 0])
    miss_c = int(missing[0, 1])
    assert int(window[miss_r, miss_c]) == int(bg), "missing corner must be background"
    corner_by_offset = {
        (0, 0): "tl",
        (0, width - 1): "tr",
        (height - 1, 0): "bl",
        (height - 1, width - 1): "br",
    }
    assert (miss_r, miss_c) in corner_by_offset, "missing cell must be a rectangle corner"
    corner = corner_by_offset[(miss_r, miss_c)]

    return {
        "color": int(color),
        "coords": coords.astype(int, copy=False),
        "top": r0,
        "left": c0,
        "height": height,
        "width": width,
        "corner": corner,
        "missing": (r0 + miss_r, c0 + miss_c),
        "direction": CORNER_TO_DIRECTION[corner],
    }


def _find_objects(g: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(g == color):
            objects.append(_analyze_component(g, coords, color, bg))
    objects.sort(key=lambda obj: (obj["top"], obj["left"], obj["color"]))
    return objects


def _compute_hits(g: np.ndarray, bg: int) -> tuple[list[dict], list[int | None]]:
    objects = _find_objects(g, bg)
    owner = np.full(g.shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx

    hits: list[int | None] = []
    height, width = g.shape
    for idx, obj in enumerate(objects):
        dr, dc = map(int, obj["direction"])
        r, c = map(int, obj["missing"])
        r += dr
        c += dc
        target = None
        while 0 <= r < height and 0 <= c < width:
            other = int(owner[r, c])
            if other != -1:
                assert other != idx, "ray should not hit its own source object"
                target = other
                break
            assert int(g[r, c]) == int(bg), "ray path should only cross background before the first hit"
            r += dr
            c += dc
        hits.append(target)
    return objects, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each corner-missing rectangle that lies on another object's diagonal ray.

    Input structure:
    - The background is the most common color.
    - The foreground consists of several non-touching monochromatic rectangles, each at least 2x2 before one corner pixel is erased.
    - Every object therefore appears as a solid axis-aligned rectangle with exactly one background-colored missing corner.

    Rule:
    - For each object, identify which corner is missing and map that defect to one diagonal direction: upper-left, upper-right, lower-left, or lower-right.
    - Starting from the missing corner cell, trace a one-pixel-wide 45-degree ray outward along that direction until either the grid boundary is reached or a foreground pixel is encountered.
    - If the ray hits another object, recolor all pixels of that target object to the source object's color while preserving the target's original shape, including its own missing corner.
    - Rays themselves are invisible in the output, source objects keep their own shape, and rays that reach the boundary without a hit do nothing.

    Key insight:
    - The erased corner is a directional cue rather than damage: once the missing-corner orientation is decoded into a diagonal ray, the task reduces to ray tracing plus whole-object color propagation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = _mode_color(g)
    objects, hits = _compute_hits(g, bg)

    incoming: dict[int, list[int]] = {}
    for src_idx, target_idx in enumerate(hits):
        if target_idx is None:
            continue
        incoming.setdefault(int(target_idx), []).append(int(src_idx))

    out = g.copy()
    for target_idx, src_indices in incoming.items():
        assert len(src_indices) == 1, "a target object should not be hit by multiple sources"
        src = objects[src_indices[0]]
        target = objects[target_idx]
        coords = target["coords"]
        out[coords[:, 0], coords[:, 1]] = int(src["color"])

    return out


def _candidate_pair(
    shape: tuple[int, int],
    blocked: list[tuple[int, int, int, int]],
    forbidden: np.ndarray,
    occupied: np.ndarray,
    source_color: int,
    target_color: int,
) -> tuple[dict, dict] | None:
    height, width = map(int, shape)
    for _ in range(PAIR_PLACEMENT_TRIES):
        corner = random.choice(CORNER_ORDER)
        direction = CORNER_TO_DIRECTION[corner]
        src_h = random.randint(*OBJECT_SIZE_RANGE)
        src_w = random.randint(*OBJECT_SIZE_RANGE)
        tgt_h = random.randint(*OBJECT_SIZE_RANGE)
        tgt_w = random.randint(*OBJECT_SIZE_RANGE)
        gap = random.randint(*PAIR_GAP_RANGE)
        hit = (random.randint(0, height - 1), random.randint(0, width - 1))

        tgt_top, tgt_left = _top_left_from_opposite(hit, tgt_h, tgt_w, corner)
        source_missing = (hit[0] - gap * direction[0], hit[1] - gap * direction[1])
        src_top, src_left = _top_left_from_missing(source_missing, src_h, src_w, corner)

        source = _build_spec(src_top, src_left, src_h, src_w, corner, source_color, target_color)
        target = _build_spec(tgt_top, tgt_left, tgt_h, tgt_w, corner, target_color, None)
        if not _in_bounds(source, shape) or not _in_bounds(target, shape):
            continue

        source_box = _touch_box(source, shape)
        target_box = _touch_box(target, shape)
        if _blocks_overlap(source_box, target_box):
            continue
        if any(_blocks_overlap(source_box, box) or _blocks_overlap(target_box, box) for box in blocked):
            continue

        source_cells = _object_cells(source)
        target_cells = _object_cells(target)
        if any(forbidden[r, c] for r, c in source_cells + target_cells):
            continue

        prefix = _ray_prefix_to_hit(_missing_cell(source), _opposite_cell(target), direction)
        if not prefix:
            continue
        if any(bool(occupied[r, c]) for r, c in prefix):
            continue

        target_continuation = _ray_cells_from(_missing_cell(target), direction, shape)
        if not target_continuation:
            continue
        if any(bool(occupied[r, c]) for r, c in target_continuation):
            continue

        return source, target
    return None


def _attempt_layout(pair_count: int) -> tuple[np.ndarray, list[dict]] | None:
    min_side = 11 + 4 * pair_count
    height = random.randint(min_side, GRID_MAX)
    width = random.randint(min_side, GRID_MAX)

    bg = random.choice(PALETTE)
    available = [color for color in PALETTE if color != bg]
    colors = random.sample(available, 2 * pair_count)

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    forbidden = np.zeros((height, width), dtype=bool)
    blocked: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for pair_idx in range(pair_count):
        source_color = colors[2 * pair_idx]
        target_color = colors[2 * pair_idx + 1]
        candidate = _candidate_pair(
            grid.shape,
            blocked,
            forbidden,
            occupied,
            source_color,
            target_color,
        )
        if candidate is None:
            return None

        source, target = candidate
        _place_spec(grid, target)
        _place_spec(grid, source)

        for spec in (target, source):
            for r, c in _object_cells(spec):
                occupied[r, c] = True
            blocked.append(_touch_box(spec, grid.shape))
            placed.append(spec)

        for r, c in _ray_prefix_to_hit(_missing_cell(source), _opposite_cell(target), CORNER_TO_DIRECTION[source["corner"]]):
            forbidden[r, c] = True
        for r, c in _ray_cells_from(_missing_cell(target), CORNER_TO_DIRECTION[target["corner"]], grid.shape):
            forbidden[r, c] = True

    return grid, placed


def _valid_generated_grid(grid: np.ndarray, expected_specs: list[dict]) -> bool:
    bg = _mode_color(grid)
    objects, hits = _compute_hits(grid, bg)
    if len(objects) != len(expected_specs):
        return False

    by_color = {int(obj["color"]): obj for obj in objects}
    if len(by_color) != len(objects):
        return False
    if {int(spec["color"]) for spec in expected_specs} != set(by_color):
        return False

    hit_by_color: dict[int, int | None] = {}
    for obj, target_idx in zip(objects, hits):
        hit_by_color[int(obj["color"])] = None if target_idx is None else int(objects[target_idx]["color"])

    incoming_counts: dict[int, int] = {}
    for target_color in hit_by_color.values():
        if target_color is not None:
            incoming_counts[target_color] = incoming_counts.get(target_color, 0) + 1
    if any(count > 1 for count in incoming_counts.values()):
        return False

    for spec in expected_specs:
        if hit_by_color[int(spec["color"])] != spec["expected_target"]:
            return False

    return not np.array_equal(main(grid), grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        pair_count = random.choice(LAYOUT_PAIR_COUNTS)
        attempt = _attempt_layout(pair_count)
        if attempt is None:
            continue
        grid, specs = attempt
        if _valid_generated_grid(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid corner-ray recoloring puzzle")
