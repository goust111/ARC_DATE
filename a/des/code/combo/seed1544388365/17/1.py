import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("up", "down", "left", "right")
ROT_CCW = {"up": "left", "left": "down", "down": "right", "right": "up"}
FLIP_LR = {"up": "up", "down": "down", "left": "right", "right": "left"}
ORIENTATIONS = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _points_to_array(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        arr[r - r0, c - c0] = True
    return arr


def _signature(arr: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(arr)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _apply_orientation(arr: np.ndarray, rotations: int, flip_lr: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr
    return np.rot90(out, rotations).copy()


def _map_side(side: str, rotations: int, flip_lr: bool) -> str:
    out = FLIP_LR[side] if flip_lr else side
    for _ in range(rotations):
        out = ROT_CCW[out]
    return out


def _canonical_shape(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    base = _points_to_array(list(points))
    return min(_signature(_apply_orientation(base, rot, flip)) for rot, flip in ORIENTATIONS)


def _is_asymmetric(points: tuple[tuple[int, int], ...]) -> bool:
    base = _points_to_array(list(points))
    return len({_signature(_apply_orientation(base, rot, flip)) for rot, flip in ORIENTATIONS}) == 8


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(points) != 2 * (h + w) - 4:
        return False
    point_set = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False
    return True


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_mask = (interior != bg) & (interior != color)
            inner_comps = _components(inner_mask)
            if len(inner_comps) != 1:
                continue
            inner_comp = inner_comps[0]
            inner_colors = {int(interior[r, c]) for r, c in inner_comp}
            if len(inner_colors) != 1:
                continue
            inner_color = next(iter(inner_colors))
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "border_color": color,
                    "border_points": tuple(sorted(comp)),
                    "shape_color": inner_color,
                    "shape_arr": _points_to_array(inner_comp),
                }
            )
    return frames


def _marker_side(marker: tuple[int, int], shape_bbox: tuple[int, int, int, int]) -> str | None:
    mr, mc = marker
    r0, c0, r1, c1 = shape_bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "up"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "down"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "right"
    return None


def _extract_templates(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    h, w = grid.shape
    interior_mask = np.zeros((h, w), dtype=bool)
    frame_border_sets = {frozenset(frame["border_points"]) for frame in frames}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    comps = []
    next_id = 0
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            comp_key = frozenset(comp)
            if comp_key in frame_border_sets:
                continue
            if all(interior_mask[r, c] for r, c in comp):
                continue
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "points": tuple(sorted(comp)),
                    "bbox": _bbox(comp),
                }
            )
            next_id += 1

    by_id = {comp["id"]: comp for comp in comps}
    singleton_by_cell = {
        comp["points"][0]: comp for comp in comps if len(comp["points"]) == 1
    }

    templates = []
    used_markers = set()
    for comp in comps:
        if len(comp["points"]) <= 1:
            continue
        neighbor_marker_ids = set()
        for r, c in comp["points"]:
            for dr, dc in DIR4:
                marker = singleton_by_cell.get((r + dr, c + dc))
                if marker is None or marker["color"] == comp["color"]:
                    continue
                neighbor_marker_ids.add(marker["id"])
        if len(neighbor_marker_ids) != 1:
            continue
        marker = by_id[next(iter(neighbor_marker_ids))]
        if marker["id"] in used_markers:
            continue
        side = _marker_side(marker["points"][0], comp["bbox"])
        if side is None:
            continue
        used_markers.add(marker["id"])
        templates.append(
            {
                "shape_color": comp["color"],
                "shape_arr": _points_to_array(list(comp["points"])),
                "marker_color": marker["color"],
                "marker_side": side,
            }
        )
    return templates


def _matched_side(frame_shape: np.ndarray, template_shape: np.ndarray, marker_side: str) -> str | None:
    hits = []
    for rotations, flip_lr in ORIENTATIONS:
        oriented = _apply_orientation(template_shape, rotations, flip_lr)
        if oriented.shape == frame_shape.shape and np.array_equal(oriented, frame_shape):
            hits.append(_map_side(marker_side, rotations, flip_lr))
    if not hits:
        return None
    if len(set(hits)) != 1:
        raise AssertionError("symmetric shape causes ambiguous side mapping")
    return hits[0]


def _choose_template(frame: dict, templates: list[dict]) -> tuple[dict, str]:
    matches = []
    for template in templates:
        side = _matched_side(frame["shape_arr"], template["shape_arr"], template["marker_side"])
        if side is not None:
            matches.append((template, side))
    if not matches:
        raise AssertionError("frame has no matching template")

    same_color = [(template, side) for template, side in matches if template["shape_color"] == frame["shape_color"]]
    candidates = same_color if same_color else matches
    if len(candidates) != 1:
        unique = {
            (template["shape_color"], template["marker_color"], side): (template, side)
            for template, side in candidates
        }
        if len(unique) == 1:
            return next(iter(unique.values()))
        raise AssertionError("frame does not map to a unique template")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's internal shape to an external template, then recolor the
    corresponding frame side with the template marker color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _extract_frames(g, bg)
    templates = _extract_templates(g, bg, frames)

    out = g.copy()
    for frame in frames:
        template, mapped_side = _choose_template(frame, templates)
        r0, c0, r1, c1 = frame["bbox"]
        color = int(template["marker_color"])
        if mapped_side == "up":
            out[r0, c0 : c1 + 1] = color
        elif mapped_side == "down":
            out[r1, c0 : c1 + 1] = color
        elif mapped_side == "left":
            out[r0 : r1 + 1, c0] = color
        else:
            out[r0 : r1 + 1, c1] = color
    return out


RAW_SHAPES = [
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (3, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
]


def _build_shape_library() -> list[tuple[tuple[int, int], ...]]:
    library = []
    seen = set()
    for shape in RAW_SHAPES:
        if not _is_asymmetric(shape):
            continue
        canon = _canonical_shape(shape)
        if canon in seen:
            continue
        seen.add(canon)
        library.append(tuple(shape))
    return library


SHAPE_LIBRARY = _build_shape_library()

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_COUNT_CHOICES = (2, 3)
FRAME_PLACEMENT_TRIES = 200
TEMPLATE_PLACEMENT_TRIES = 120
GEN_RETRIES = 400


def _frame_patch(shape_arr: np.ndarray, border_color: int, shape_color: int, bg: int, rng: random.Random) -> np.ndarray:
    sh, sw = shape_arr.shape
    gap_top = rng.randint(1, 2)
    gap_bottom = rng.randint(1, 2)
    gap_left = rng.randint(1, 2)
    gap_right = rng.randint(1, 2)
    fh = sh + gap_top + gap_bottom + 2
    fw = sw + gap_left + gap_right + 2
    patch = np.full((fh, fw), bg, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    top = 1 + gap_top
    left = 1 + gap_left
    patch[top : top + sh, left : left + sw][shape_arr] = shape_color
    return patch


def _template_patch(shape_arr: np.ndarray, marker_side: str, shape_color: int, marker_color: int, bg: int, rng: random.Random) -> np.ndarray:
    sh, sw = shape_arr.shape
    if marker_side == "up":
        patch = np.full((sh + 1, sw), bg, dtype=int)
        patch[1 : 1 + sh, :sw][shape_arr] = shape_color
        cols = [int(c) for c in np.flatnonzero(shape_arr[0])]
        patch[0, rng.choice(cols)] = marker_color
        return patch
    if marker_side == "down":
        patch = np.full((sh + 1, sw), bg, dtype=int)
        patch[:sh, :sw][shape_arr] = shape_color
        cols = [int(c) for c in np.flatnonzero(shape_arr[-1])]
        patch[sh, rng.choice(cols)] = marker_color
        return patch
    if marker_side == "left":
        patch = np.full((sh, sw + 1), bg, dtype=int)
        patch[:sh, 1 : 1 + sw][shape_arr] = shape_color
        rows = [int(r) for r in np.flatnonzero(shape_arr[:, 0])]
        patch[rng.choice(rows), 0] = marker_color
        return patch
    patch = np.full((sh, sw + 1), bg, dtype=int)
    patch[:sh, :sw][shape_arr] = shape_color
    rows = [int(r) for r in np.flatnonzero(shape_arr[:, -1])]
    patch[rng.choice(rows), sw] = marker_color
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _reserve(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != bg
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _random_oriented_shape(shape: tuple[tuple[int, int], ...], rng: random.Random) -> np.ndarray:
    base = _points_to_array(list(shape))
    rotations, flip_lr = rng.choice(ORIENTATIONS)
    return _apply_orientation(base, rotations, flip_lr)


def _place_frame_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, bg: int, rng: random.Random) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if gh - ph < 8 or gw - pw < 8:
        return False
    for _ in range(FRAME_PLACEMENT_TRIES):
        top = rng.randint(3, gh - ph - 3)
        left = rng.randint(3, gw - pw - 3)
        if not _can_place(blocked, top, left, ph, pw, pad=1):
            continue
        _paste(grid, patch, top, left, bg)
        _reserve(blocked, top, left, ph, pw, pad=1)
        return True
    return False


def _place_template_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, bg: int, rng: random.Random) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    edges = ["top", "bottom", "left", "right"]
    rng.shuffle(edges)
    for edge in edges:
        for _ in range(TEMPLATE_PLACEMENT_TRIES):
            if edge == "top":
                top = 0
                left = rng.randint(0, gw - pw)
            elif edge == "bottom":
                top = gh - ph
                left = rng.randint(0, gw - pw)
            elif edge == "left":
                top = rng.randint(0, gh - ph)
                left = 0
            else:
                top = rng.randint(0, gh - ph)
                left = gw - pw
            if not _can_place(blocked, top, left, ph, pw, pad=1):
                continue
            _paste(grid, patch, top, left, bg)
            _reserve(blocked, top, left, ph, pw, pad=1)
            return True
    return False


def _is_valid_instance(grid: np.ndarray, expected_frames: int) -> bool:
    try:
        bg = _background_color(grid)
        frames = _extract_frames(grid, bg)
        templates = _extract_templates(grid, bg, frames)
        if len(frames) != expected_frames or len(templates) != expected_frames:
            return False
        matched_sides = []
        for frame in frames:
            template, side = _choose_template(frame, templates)
            if template["marker_color"] == frame["border_color"]:
                return False
            matched_sides.append(side)
        if not matched_sides:
            return False
        return not np.array_equal(main(grid), grid)
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice(FRAME_COUNT_CHOICES)
    gh = rng.randint(*GRID_H_RANGE)
    gw = rng.randint(*GRID_W_RANGE)
    bg = rng.randint(0, 9)
    grid = np.full((gh, gw), bg, dtype=int)
    blocked = np.zeros((gh, gw), dtype=bool)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    needed = 3 * n_frames
    if len(palette) < needed:
        return None
    border_colors = palette[:n_frames]
    shape_colors = palette[n_frames : 2 * n_frames]
    marker_colors = palette[2 * n_frames : 3 * n_frames]

    shape_choices = rng.sample(SHAPE_LIBRARY, n_frames)
    rng.shuffle(shape_choices)

    template_specs = []
    for i in range(n_frames):
        frame_shape = _random_oriented_shape(shape_choices[i], rng)
        frame_patch = _frame_patch(frame_shape, border_colors[i], shape_colors[i], bg, rng)
        if not _place_frame_patch(grid, blocked, frame_patch, bg, rng):
            return None

        template_shape = _random_oriented_shape(shape_choices[i], rng)
        marker_side = rng.choice(SIDES)
        template_patch = _template_patch(template_shape, marker_side, shape_colors[i], marker_colors[i], bg, rng)
        template_specs.append(template_patch)

    rng.shuffle(template_specs)
    for patch in template_specs:
        if not _place_template_patch(grid, blocked, patch, bg, rng):
            return None

    if not _is_valid_instance(grid, n_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
