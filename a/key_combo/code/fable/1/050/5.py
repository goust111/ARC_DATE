import random
from collections import Counter

import numpy as np


ODD_SIDES = (1, 3, 5, 7, 9)
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(ODD_SIDES)}

PALETTE = tuple(range(10))
FULL_CANVAS = 30

GEN_OBJECT_COUNT = (2, 5)
GEN_MARKER_STEPS = (1, 4)
GEN_WRAP_SCENE_PROB = 0.65
GEN_EXTRA_WRAP_PROB = 0.4
GEN_CLEARANCE = 1
GEN_MARGIN_CAP = 7
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if visited[start_r, start_c]:
            continue
        visited[start_r, start_c] = True
        stack = [(start_r, start_c)]
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in steps:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g) if bg is None else int(bg)

    if monochromatic:
        out: list[dict] = []
        for color in sorted(set(map(int, g.ravel())) - {bg}):
            for coords in components_mask(g == color, connectivity=connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    out = []
    for coords in components_mask(g != bg, connectivity=connectivity):
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _outer_ring_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bbox_from_center(center_r: int, center_c: int, side: int) -> tuple[int, int, int, int]:
    radius = side // 2
    return center_r - radius, center_c - radius, center_r + radius, center_c + radius


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _side_after(side: int, direction: int, marker_count: int) -> int:
    idx = SIDE_TO_INDEX[int(side)]
    return int(ODD_SIDES[(idx + int(direction) * int(marker_count)) % len(ODD_SIDES)])


def _is_wrap(side: int, direction: int, marker_count: int) -> bool:
    idx = SIDE_TO_INDEX[int(side)]
    raw = idx + int(direction) * int(marker_count)
    return raw < 0 or raw >= len(ODD_SIDES)


def _marker_cells(side: int, direction: int) -> list[tuple[int, int]]:
    ring = _outer_ring_mask(side)
    if direction == 1:
        return [(r, c) for r in range(side) for c in range(side) if ring[r, c]]
    return [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]


def _sample_spread(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert 1 <= count <= len(cells)
    chosen = [random.choice(cells)]
    while len(chosen) < count:
        remaining = [cell for cell in cells if cell not in chosen]
        best_score = -1
        best_cells: list[tuple[int, int]] = []
        for cell in remaining:
            score = min(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) for other in chosen)
            if score > best_score:
                best_score = score
                best_cells = [cell]
            elif score == best_score:
                best_cells.append(cell)
        chosen.append(random.choice(best_cells))
    random.shuffle(chosen)
    return chosen


def _analyze_component(grid: np.ndarray, obj: dict, bg: int) -> dict:
    r0, c0, r1, c1 = obj["bbox"]
    side = int(r1 - r0 + 1)
    assert side == int(c1 - c0 + 1), "object bbox must be square"
    assert side in ODD_SIDES, f"unexpected square side {side}"
    assert int(obj["area"]) == side * side, "square must occupy its full bbox"

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any(region == bg), "square bbox must not contain background holes"

    counts = Counter(map(int, region.ravel()))
    fill_count = max(counts.values())
    fill_colors = [color for color, count in counts.items() if count == fill_count]
    assert len(fill_colors) == 1, "square fill color must be uniquely dominant"
    fill_color = int(fill_colors[0])

    marker_mask = region != fill_color
    marker_count = int(marker_mask.sum())
    marker_values = [int(v) for v in region[marker_mask].tolist()]
    assert len(marker_values) == len(set(marker_values)), "marker colors must be distinct within a square"

    center = (int((r0 + r1) // 2), int((c0 + c1) // 2))
    if marker_count == 0:
        direction = 0
        output_side = side
    else:
        ring_mask = _outer_ring_mask(side)
        ring_count = int(np.count_nonzero(marker_mask & ring_mask))
        inner_count = marker_count - ring_count
        assert ring_count in (0, marker_count), "markers must all lie on one layer class"
        direction = 1 if ring_count == marker_count else -1
        output_side = _side_after(side, direction, marker_count)

    return {
        "fill_color": fill_color,
        "input_side": side,
        "output_side": output_side,
        "marker_count": marker_count,
        "direction": direction,
        "center": center,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize each marked odd-sided square by a layer-count direction rule, wrapping around the 1-3-5-7-9 cycle.

    Input structure:
    - A single background color surrounds several separated solid odd-sided squares.
    - Each square's bbox is fully occupied by its fill color plus a few marker pixels of distinct colors.
    - For any one square, all markers lie either on the outer ring or entirely in the strict interior.

    Rule:
    - Treat each connected non-background object as one square and ignore marker colors except for their count and layer class.
    - Outer-ring markers mean expansion, strict-interior markers mean contraction, and the number of markers is the number of size steps.
    - Keep each square's center and fill color, change the side length by two per step on the cyclic ladder 1-3-5-7-9, and remove all markers in the output.

    Key insight:
    - Marker color is a decoy; only marker layer decides direction and marker count decides the wrapped magnitude.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert objects, "expected at least one marked square"

    out = np.full_like(g, bg)
    painted = np.zeros(g.shape, dtype=bool)
    h, w = g.shape

    for obj in objects:
        info = _analyze_component(g, obj, bg)
        fill = info["fill_color"]
        new_side = info["output_side"]
        center_r, center_c = info["center"]
        nr0, nc0, nr1, nc1 = _bbox_from_center(center_r, center_c, new_side)
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "resized square must remain in-bounds"
        assert not painted[nr0 : nr1 + 1, nc0 : nc1 + 1].any(), "resized squares must stay disjoint"
        patch = np.full((new_side, new_side), fill, dtype=int)
        paste(out, patch, nr0, nc0)
        painted[nr0 : nr1 + 1, nc0 : nc1 + 1] = True

    return out


def _candidate_specs() -> list[dict]:
    specs: list[dict] = []
    for input_side in ODD_SIDES:
        if input_side == 1:
            continue
        for direction in (1, -1):
            capacity = len(_marker_cells(input_side, direction))
            for marker_count in range(GEN_MARKER_STEPS[0], min(GEN_MARKER_STEPS[1], capacity) + 1):
                specs.append(
                    {
                        "input_side": int(input_side),
                        "direction": int(direction),
                        "marker_count": int(marker_count),
                        "output_side": _side_after(input_side, direction, marker_count),
                        "wrap": _is_wrap(input_side, direction, marker_count),
                    }
                )
    return specs


GEN_SPECS = _candidate_specs()


def _choose_scene_specs() -> tuple[int, list[dict]]:
    count = random.randint(*GEN_OBJECT_COUNT)
    bg = random.choice(PALETTE)
    fill_colors = random.sample([color for color in PALETTE if color != bg], count)
    marker_palette = [color for color in PALETTE if color not in {bg, *fill_colors}]

    assert len(marker_palette) >= GEN_MARKER_STEPS[1]

    specs: list[dict] = []
    want_wrap = random.random() < GEN_WRAP_SCENE_PROB
    if want_wrap:
        wrap_dir = random.choice((1, -1))
        wrap_options = [spec for spec in GEN_SPECS if spec["direction"] == wrap_dir and spec["wrap"]]
        specs.append(dict(random.choice(wrap_options)))

    if not any(spec["direction"] == 1 for spec in specs):
        specs.append(dict(random.choice([spec for spec in GEN_SPECS if spec["direction"] == 1])))
    if not any(spec["direction"] == -1 for spec in specs):
        specs.append(dict(random.choice([spec for spec in GEN_SPECS if spec["direction"] == -1])))

    while len(specs) < count:
        pool = [spec for spec in GEN_SPECS if spec["wrap"]] if random.random() < GEN_EXTRA_WRAP_PROB else GEN_SPECS
        specs.append(dict(random.choice(pool)))

    random.shuffle(specs)
    for spec, fill in zip(specs, fill_colors, strict=False):
        spec["fill_color"] = int(fill)
        spec["marker_colors"] = random.sample(marker_palette, spec["marker_count"])
        spec["reserve_side"] = max(spec["input_side"], spec["output_side"])
        spec["local_markers"] = _sample_spread(
            _marker_cells(spec["input_side"], spec["direction"]),
            spec["marker_count"],
        )
    return bg, specs


def _place_objects(specs: list[dict]) -> list[dict] | None:
    placements: list[dict | None] = [None] * len(specs)
    reserved_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["reserve_side"], reverse=True)

    for idx in order:
        spec = specs[idx]
        margin = spec["reserve_side"] // 2
        centers = [
            (r, c)
            for r in range(margin, FULL_CANVAS - margin)
            for c in range(margin, FULL_CANVAS - margin)
        ]
        random.shuffle(centers)

        for center_r, center_c in centers:
            reserve_box = _bbox_from_center(center_r, center_c, spec["reserve_side"])
            if all(_boxes_clear(reserve_box, other, gap=GEN_CLEARANCE) for other in reserved_boxes):
                placements[idx] = {
                    "center": (center_r, center_c),
                    "reserve_box": reserve_box,
                    "input_box": _bbox_from_center(center_r, center_c, spec["input_side"]),
                    "output_box": _bbox_from_center(center_r, center_c, spec["output_side"]),
                }
                reserved_boxes.append(reserve_box)
                break
        else:
            return None

    return [placement for placement in placements if placement is not None]


def _choose_crop_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox_union(boxes)
    top_room = min(r0, GEN_MARGIN_CAP)
    left_room = min(c0, GEN_MARGIN_CAP)
    bottom_room = min(FULL_CANVAS - 1 - r1, GEN_MARGIN_CAP)
    right_room = min(FULL_CANVAS - 1 - c1, GEN_MARGIN_CAP)

    extra_top = random.randint(0, top_room)
    extra_left = random.randint(0, left_room)
    extra_bottom = random.randint(0, bottom_room)
    extra_right = random.randint(0, right_room)
    return r0 - extra_top, c0 - extra_left, r1 + extra_bottom, c1 + extra_right


def _render_input(bg: int, specs: list[dict], placements: list[dict]) -> np.ndarray:
    grid = np.full((FULL_CANVAS, FULL_CANVAS), bg, dtype=int)
    for spec, placement in zip(specs, placements, strict=False):
        r0, c0, r1, c1 = placement["input_box"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), spec["fill_color"], dtype=int)
        paste(grid, patch, r0, c0)
        for (lr, lc), color in zip(spec["local_markers"], spec["marker_colors"], strict=False):
            grid[r0 + lr, c0 + lc] = int(color)
    return grid


def _render_output(bg: int, specs: list[dict], placements: list[dict]) -> np.ndarray:
    grid = np.full((FULL_CANVAS, FULL_CANVAS), bg, dtype=int)
    for spec, placement in zip(specs, placements, strict=False):
        r0, c0, r1, c1 = placement["output_box"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), spec["fill_color"], dtype=int)
        paste(grid, patch, r0, c0)
    return grid


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _validate_scene(inp: np.ndarray, expected: np.ndarray, bg: int, specs: list[dict], placements: list[dict], crop_box: tuple[int, int, int, int]) -> bool:
    if get_bg(inp) != bg:
        return False
    if not any(spec["direction"] == 1 for spec in specs):
        return False
    if not any(spec["direction"] == -1 for spec in specs):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    observed = Counter()
    for obj in find_components(inp, bg=bg, connectivity=4, monochromatic=False):
        info = _analyze_component(inp, obj, bg)
        observed[
            (
                info["fill_color"],
                info["input_side"],
                info["marker_count"],
                info["direction"],
                info["output_side"],
                info["center"],
            )
        ] += 1

    cr0, cc0, _cr1, _cc1 = crop_box
    expected_signature = Counter()
    for spec, placement in zip(specs, placements, strict=False):
        center_r, center_c = placement["center"]
        expected_signature[
            (
                spec["fill_color"],
                spec["input_side"],
                spec["marker_count"],
                spec["direction"],
                spec["output_side"],
                (center_r - cr0, center_c - cc0),
            )
        ] += 1

    return observed == expected_signature


def _generate_candidate() -> np.ndarray | None:
    bg, specs = _choose_scene_specs()
    placements = _place_objects(specs)
    if placements is None:
        return None

    crop_box = _choose_crop_box([placement["reserve_box"] for placement in placements])
    inp = _crop(_render_input(bg, specs, placements), crop_box)
    expected = _crop(_render_output(bg, specs, placements), crop_box)

    if not _validate_scene(inp, expected, bg, specs, placements, crop_box):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid marked-pressure-scaling input")
