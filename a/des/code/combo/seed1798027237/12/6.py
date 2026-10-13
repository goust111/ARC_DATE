import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = np.arange(10, dtype=int)
GRID_MAX = 30
GEN_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    mask = _component_mask(component)
    expected_area = 2 * (h + w) - 4
    if int(mask.sum()) != expected_area:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    return not bool(mask[1:-1, 1:-1].any())


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _select_object_and_frame(components: list[dict]) -> tuple[dict, dict]:
    frames = [component for component in components if _is_frame(component)]
    solids = [component for component in components if _is_solid_rectangle(component)]
    assert frames, "expected at least one frame"
    assert solids, "expected a solid object"

    ranked: list[tuple[int, int, dict, list[dict]]] = []
    for solid in solids:
        enclosing = [frame for frame in frames if _strictly_contains(frame["bbox"], solid["bbox"])]
        if enclosing:
            ranked.append((len(enclosing), _bbox_area(solid["bbox"]), solid, enclosing))

    assert ranked, "expected a solid rectangle enclosed by frames"
    max_depth = max(depth for depth, _area, _solid, _frames in ranked)
    deepest = [item for item in ranked if item[0] == max_depth]
    _depth, _area, obj, enclosing_frames = min(deepest, key=lambda item: item[1])

    ordered = sorted(enclosing_frames, key=lambda frame: _bbox_area(frame["bbox"]), reverse=True)
    for outer, inner in zip(ordered, ordered[1:]):
        assert _strictly_contains(outer["bbox"], inner["bbox"]), "frames must form a nested chain"

    innermost = min(enclosing_frames, key=lambda frame: _bbox_area(frame["bbox"]))
    return obj, innermost


def _eject_direction(
    obj_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    or0, oc0, or1, oc1 = obj_bbox
    fr0, fc0, fr1, fc1 = frame_bbox

    top_gap = or0 - (fr0 + 1)
    bottom_gap = (fr1 - 1) - or1
    left_gap = oc0 - (fc0 + 1)
    right_gap = (fc1 - 1) - oc1
    assert min(top_gap, bottom_gap, left_gap, right_gap) >= 0

    if top_gap != bottom_gap:
        dr = -1 if top_gap < bottom_gap else 1
    else:
        obj_center_r = (or0 + or1) / 2.0
        frame_center_r = (fr0 + fr1) / 2.0
        dr = -1 if obj_center_r < frame_center_r else 1

    if left_gap != right_gap:
        dc = -1 if left_gap < right_gap else 1
    else:
        obj_center_c = (oc0 + oc1) / 2.0
        frame_center_c = (fc0 + fc1) / 2.0
        dc = -1 if obj_center_c < frame_center_c else 1

    return int(dr), int(dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch through nested frames by ejecting the inner rectangle toward the corner it favors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    components = _find_components(g, bg)
    obj, inner_frame = _select_object_and_frame(components)
    dr, dc = _eject_direction(obj["bbox"], inner_frame["bbox"])

    r0, c0, r1, c1 = obj["bbox"]
    h, w = g.shape
    row_room = r0 if dr < 0 else h - 1 - r1
    col_room = c0 if dc < 0 else w - 1 - c1
    steps = int(min(row_room, col_room))
    assert steps > 0, "object must have room to reach an edge"

    out = g.copy()
    direction = np.array([dr, dc], dtype=int)
    coords = obj["coords"]
    for step in range(steps + 1):
        shifted = coords + direction * step
        out[shifted[:, 0], shifted[:, 1]] = bg

    final_coords = coords + direction * steps
    out[final_coords[:, 0], final_coords[:, 1]] = int(obj["color"])
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _biased_pair(rng: np.random.Generator, toward_negative: bool) -> tuple[int, int]:
    near = int(rng.integers(0, 2))
    far = int(rng.integers(near + 1, near + 4))
    return (near, far) if toward_negative else (far, near)


def _sample_spacings(rng: np.random.Generator, frame_count: int) -> list[int]:
    if frame_count <= 1:
        return []
    if frame_count >= 5:
        return [2] * (frame_count - 1)
    return [2 + int(rng.random() < 0.35) for _ in range(frame_count - 1)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(2, 6))
    obj_h = int(rng.integers(1, 4))
    obj_w = int(rng.integers(1, 4))
    dr, dc = [(-1, -1), (-1, 1), (1, -1), (1, 1)][int(rng.integers(0, 4))]

    top_gap, bottom_gap = _biased_pair(rng, toward_negative=(dr < 0))
    left_gap, right_gap = _biased_pair(rng, toward_negative=(dc < 0))

    inner_h = obj_h + top_gap + bottom_gap + 2
    inner_w = obj_w + left_gap + right_gap + 2
    spacings = _sample_spacings(rng, frame_count)
    total_offset = sum(spacings)
    outer_h = inner_h + 2 * total_offset
    outer_w = inner_w + 2 * total_offset

    if outer_h > GRID_MAX - 2 or outer_w > GRID_MAX - 2:
        return None

    max_extra_h = min(10, GRID_MAX - outer_h)
    max_extra_w = min(10, GRID_MAX - outer_w)
    if max_extra_h < 2 or max_extra_w < 2:
        return None

    extra_h = int(rng.integers(2, max_extra_h + 1))
    extra_w = int(rng.integers(2, max_extra_w + 1))
    top_margin = int(rng.integers(1, extra_h))
    bottom_margin = extra_h - top_margin
    left_margin = int(rng.integers(1, extra_w))
    right_margin = extra_w - left_margin

    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    colors = list(map(int, rng.choice(PALETTE, size=frame_count + 2, replace=False)))
    bg = colors[0]
    frame_colors = colors[1 : 1 + frame_count]
    obj_color = colors[-1]

    grid = np.full((h, w), bg, dtype=int)
    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_h - 1,
        left_margin + outer_w - 1,
    )

    offsets = [0]
    for spacing in spacings:
        offsets.append(offsets[-1] + spacing)

    for offset, color in zip(offsets, frame_colors):
        bbox = (
            outer_bbox[0] + offset,
            outer_bbox[1] + offset,
            outer_bbox[2] - offset,
            outer_bbox[3] - offset,
        )
        _draw_frame(grid, bbox, color)

    inner_bbox = (
        outer_bbox[0] + offsets[-1],
        outer_bbox[1] + offsets[-1],
        outer_bbox[2] - offsets[-1],
        outer_bbox[3] - offsets[-1],
    )
    ir0, ic0, _ir1, _ic1 = inner_bbox
    obj_r0 = ir0 + 1 + top_gap
    obj_c0 = ic0 + 1 + left_gap
    grid[obj_r0 : obj_r0 + obj_h, obj_c0 : obj_c0 + obj_w] = obj_color

    if _bg_color(grid) != bg:
        return None

    try:
        components = _find_components(grid, bg)
        obj, innermost = _select_object_and_frame(components)
        if int(obj["color"]) != obj_color:
            return None
        if innermost["bbox"] != inner_bbox:
            return None
        if _eject_direction(obj["bbox"], innermost["bbox"]) != (dr, dc):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
