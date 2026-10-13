import random
from collections import deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
ROTATE_CW = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox):
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_record(comp):
    coords = comp["coords"]
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    cell_set = set(coords)
    perimeter = _perimeter_cells(bbox)
    if not cell_set <= perimeter:
        return None

    missing = perimeter - cell_set
    if len(missing) != 1:
        return None

    gap = next(iter(missing))
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        gap_side = "top"
    elif gc == c1 and r0 < gr < r1:
        gap_side = "right"
    elif gr == r1 and c0 < gc < c1:
        gap_side = "bottom"
    elif gc == c0 and r0 < gr < r1:
        gap_side = "left"
    else:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": bbox,
        "gap": gap,
        "gap_side": gap_side,
    }


def _source_record(comp, h: int, w: int):
    coords = comp["coords"]
    if len(coords) != 1:
        return None
    r, c = coords[0]
    side = None
    if r == 0 and 0 < c < w - 1:
        side = "top"
    elif r == h - 1 and 0 < c < w - 1:
        side = "bottom"
    elif c == 0 and 0 < r < h - 1:
        side = "left"
    elif c == w - 1 and 0 < r < h - 1:
        side = "right"
    if side is None:
        return None
    return {"color": int(comp["color"]), "pos": (int(r), int(c)), "side": side}


def _bbox_area(bbox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_inside(inner, outer) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _structure_groups(frames):
    groups = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        key = (r0 + r1, c0 + c1)
        groups.setdefault(key, []).append(frame)

    structures = []
    for group in groups.values():
        ordered = sorted(group, key=lambda frame: _bbox_area(frame["bbox"]), reverse=True)
        if len(ordered) < 2:
            continue
        nested = [ordered[0]]
        for frame in ordered[1:]:
            if _strictly_inside(frame["bbox"], nested[-1]["bbox"]):
                nested.append(frame)
        if len(nested) >= 2:
            structures.append(nested)
    return structures


def _path_clear(grid: np.ndarray, bg: int, source, gap) -> bool:
    h, w = grid.shape
    sr, sc = source["pos"]
    gr, gc = gap
    side = source["side"]
    if side == "top":
        return sc == gc and sr == 0 and gr > 0 and np.all(grid[1:gr, sc] == bg)
    if side == "bottom":
        return sc == gc and sr == h - 1 and gr < h - 1 and np.all(grid[gr + 1 : h - 1, sc] == bg)
    if side == "left":
        return sr == gr and sc == 0 and gc > 0 and np.all(grid[sr, 1:gc] == bg)
    return sr == gr and sc == w - 1 and gc < w - 1 and np.all(grid[sr, gc + 1 : w - 1] == bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow edge-aligned light sources through concentric rectangular frames.

    A source can activate one nested structure when it lines up with the gap on
    the outer layer. Each successful penetration rotates the expected gap side
    clockwise for the next inner layer, and the light color is replaced by the
    layer color it just crossed. If every layer in the chain matches this side
    sequence, fill the innermost core with the last crossed frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    h, w = g.shape
    frames = []
    sources = []
    for comp in _components(g, bg):
        frame = _frame_record(comp)
        if frame is not None:
            frames.append(frame)
            continue
        source = _source_record(comp, h, w)
        if source is not None:
            sources.append(source)

    out = g.copy()
    used_sources = set()
    for structure in _structure_groups(frames):
        outer = structure[0]
        candidates = []
        for idx, source in enumerate(sources):
            if idx in used_sources or source["side"] != outer["gap_side"]:
                continue
            if _path_clear(g, bg, source, outer["gap"]):
                candidates.append(idx)
        if len(candidates) != 1:
            continue

        used_sources.add(candidates[0])
        expected_side = outer["gap_side"]
        last_color = bg
        success = True
        for frame in structure:
            if frame["gap_side"] != expected_side:
                success = False
                break
            last_color = int(frame["color"])
            expected_side = ROTATE_CW[expected_side]

        if not success:
            continue

        r0, c0, r1, c1 = structure[-1]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = last_color

    return out


# --------------- generator ---------------

BG = 0
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
LAYERS_RANGE = (2, 4)
CORE_RANGE = (1, 4)
STRUCTURE_COUNT_CHOICES = (1, 2, 3)
PLACE_RETRIES = 180
GEN_RETRIES = 400


def _expected_sides(start_side: str, layers: int):
    sides = [start_side]
    for _ in range(1, layers):
        sides.append(ROTATE_CW[sides[-1]])
    return sides


def _sample_structure_spec(rng: random.Random, h: int, w: int, start_side: str, success: bool):
    layers = rng.randint(*LAYERS_RANGE)
    max_core_h = min(CORE_RANGE[1], h - 4 * layers - 2)
    max_core_w = min(CORE_RANGE[1], w - 4 * layers - 2)
    if max_core_h < CORE_RANGE[0] or max_core_w < CORE_RANGE[0]:
        return None

    core_h = rng.randint(CORE_RANGE[0], max_core_h)
    core_w = rng.randint(CORE_RANGE[0], max_core_w)
    outer_h = core_h + 4 * layers - 2
    outer_w = core_w + 4 * layers - 2

    colors = rng.sample(list(range(1, 10)), layers + 1)
    layer_colors = colors[:layers]
    source_color = colors[-1]

    gap_sides = _expected_sides(start_side, layers)
    if not success:
        fail_layer = rng.randint(1, layers - 1)
        wrong_choices = [side for side in SIDES if side != gap_sides[fail_layer]]
        gap_sides[fail_layer] = rng.choice(wrong_choices)
        for idx in range(fail_layer + 1, layers):
            gap_sides[idx] = rng.choice(SIDES)

    return {
        "layers": layers,
        "core_h": core_h,
        "core_w": core_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "layer_colors": layer_colors,
        "source_color": source_color,
        "gap_sides": gap_sides,
        "success": success,
    }


def _gap_cell_for_side(bbox, side: str, rng: random.Random, fixed: int | None = None):
    r0, c0, r1, c1 = bbox
    if side == "top":
        col = fixed if fixed is not None else rng.randint(c0 + 1, c1 - 1)
        return r0, col
    if side == "bottom":
        col = fixed if fixed is not None else rng.randint(c0 + 1, c1 - 1)
        return r1, col
    if side == "left":
        row = fixed if fixed is not None else rng.randint(r0 + 1, r1 - 1)
        return row, c0
    row = fixed if fixed is not None else rng.randint(r0 + 1, r1 - 1)
    return row, c1


def _frame_bboxes(r0: int, c0: int, spec):
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    bboxes = []
    for depth in range(spec["layers"]):
        top = r0 + 2 * depth
        left = c0 + 2 * depth
        bottom = r0 + outer_h - 1 - 2 * depth
        right = c0 + outer_w - 1 - 2 * depth
        bboxes.append((top, left, bottom, right))
    return bboxes


def _paint_frame(grid: np.ndarray, bbox, color: int, gap):
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        if (r0, c) != gap:
            grid[r0, c] = color
        if (r1, c) != gap:
            grid[r1, c] = color
    for r in range(r0 + 1, r1):
        if (r, c0) != gap:
            grid[r, c0] = color
        if (r, c1) != gap:
            grid[r, c1] = color


def _line_cells(h: int, w: int, side: str, bbox, axis_coord: int):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r, axis_coord) for r in range(0, r0)]
    if side == "bottom":
        return [(r, axis_coord) for r in range(r1 + 1, h)]
    if side == "left":
        return [(axis_coord, c) for c in range(0, c0)]
    return [(axis_coord, c) for c in range(c1 + 1, w)]


def _reserve_box(reserved: np.ndarray, bbox):
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    reserved[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _balanced_statuses(rng: random.Random, n: int):
    if n == 1:
        return [bool(rng.randrange(2))]
    success_count = n // 2
    if n % 2 == 1 and rng.randrange(2):
        success_count += 1
    statuses = [True] * success_count + [False] * (n - success_count)
    rng.shuffle(statuses)
    return statuses


def _place_structure(
    grid: np.ndarray,
    reserved: np.ndarray,
    used_axes: list[int],
    side: str,
    spec,
    rng: random.Random,
):
    h, w = grid.shape
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(2, h - outer_h - 2)
        c0 = rng.randint(1, w - outer_w - 1)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

        if reserved[max(0, r0 - 1) : min(h, r0 + outer_h + 1), max(0, c0 - 1) : min(w, c0 + outer_w + 1)].any():
            continue

        if side in ("top", "bottom"):
            axis_coord = rng.randint(c0 + 1, c0 + outer_w - 2)
        else:
            axis_coord = rng.randint(r0 + 1, r0 + outer_h - 2)

        if any(abs(axis_coord - old) <= 1 for old in used_axes):
            continue

        line_cells = _line_cells(h, w, side, bbox, axis_coord)
        if any(reserved[r, c] for r, c in line_cells):
            continue

        bboxes = _frame_bboxes(r0, c0, spec)
        gaps = []
        for depth, layer_bbox in enumerate(bboxes):
            fixed = axis_coord if depth == 0 else None
            gaps.append(_gap_cell_for_side(layer_bbox, spec["gap_sides"][depth], rng, fixed=fixed))

        for layer_bbox, color, gap in zip(bboxes, spec["layer_colors"], gaps):
            _paint_frame(grid, layer_bbox, int(color), gap)

        if side == "top":
            source_pos = (0, axis_coord)
        elif side == "bottom":
            source_pos = (h - 1, axis_coord)
        elif side == "left":
            source_pos = (axis_coord, 0)
        else:
            source_pos = (axis_coord, w - 1)
        grid[source_pos] = int(spec["source_color"])

        _reserve_box(reserved, bbox)
        for r, c in line_cells:
            reserved[r, c] = True
        used_axes.append(int(axis_coord))

        return {
            "bboxes": bboxes,
            "success": bool(spec["success"]),
            "fill_color": int(spec["layer_colors"][-1]),
        }

    return None


def _expected_output(inp: np.ndarray, placed):
    out = inp.copy()
    for item in placed:
        if not item["success"]:
            continue
        r0, c0, r1, c1 = item["bboxes"][-1]
        out[r0 + 1 : r1, c0 + 1 : c1] = item["fill_color"]
    return out


def _gen_attempt():
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    side = rng.choice(SIDES)
    count = rng.choice(STRUCTURE_COUNT_CHOICES)
    statuses = _balanced_statuses(rng, count)

    specs = []
    for success in statuses:
        spec = _sample_structure_spec(rng, h, w, side, success)
        if spec is None:
            return None
        specs.append(spec)
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    used_axes = []
    placed = []

    for spec in specs:
        item = _place_structure(grid, reserved, used_axes, side, spec, rng)
        if item is None:
            return None
        placed.append(item)

    if _bg_color(grid) != BG:
        return None

    expected = _expected_output(grid, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid nested-light ARC sample")
