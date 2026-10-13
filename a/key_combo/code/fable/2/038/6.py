from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = tuple(range(10))
MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 600
FRAME_COUNT_CHOICES: tuple[int, ...] = (2, 3)
FRAME_SCALE_CHOICES: tuple[int, ...] = (2, 3, 4, 5)
OBJECT_PADDING = 1
EDGE_MARGIN = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            queue: deque[Coord] = deque([(r, c)])
            cells: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue: deque[Coord] = deque([start])
    seen = {start}

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _is_hollow_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _compact_sparse_template(comp: dict) -> np.ndarray | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)

    if np.any(rel[:, 0] % 2 != 0) or np.any(rel[:, 1] % 2 != 0):
        return None

    sparse = np.zeros((h, w), dtype=bool)
    sparse[rel[:, 0], rel[:, 1]] = True
    nonempty_rows = np.any(sparse, axis=1)
    nonempty_cols = np.any(sparse, axis=0)
    compact = sparse[nonempty_rows][:, nonempty_cols]
    if compact.size == 0:
        return None
    if compact.shape == sparse.shape:
        return None
    if compact.shape[0] < 2 or compact.shape[1] < 2:
        return None
    if not _mask_is_connected(compact):
        return None
    return compact


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    bg = _get_bg(grid)
    frames: list[dict] = []
    templates: list[dict] = []
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert colors, "expected non-background objects"

    for color in colors:
        coords = np.argwhere(grid == int(color))
        obj = {
            "color": int(color),
            "coords": coords,
            "bbox": _bbox_coords(coords),
            "area": int(coords.shape[0]),
        }
        if _is_hollow_frame(obj):
            frames.append(obj)
            continue
        compact = _compact_sparse_template(obj)
        if compact is None:
            raise AssertionError("unexpected foreground object")
        template = dict(obj)
        template["compact"] = compact
        templates.append(template)

    assert len(templates) == 1, "expected one sparse template"
    assert frames, "expected at least one hollow frame"
    return bg, templates[0], frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact the sparse template and stamp its integer-scaled copies into every hollow frame.

    Input structure:
    - A dominant background surrounds exactly one monochrome sparse template and one or more rectangular hollow frames.
    - The template's pixels lie only on even-row/even-column intersections of its own bounding box, so blank rows and columns separate the true shape.
    - Each frame has a different border color and an empty interior whose height and width are equal integer multiples of the compacted template.

    Rule:
    - Identify the sparse template and delete every empty row and empty column inside its bounding box to recover the compact shape.
    - For each hollow frame, compute the shared integer scale factor from frame-interior size to compact-template size.
    - Fill the frame interior with a pixel-block-scaled copy of the compact template, using the frame's border color, while leaving the original template and frame borders unchanged.

    Key insight:
    - The visible template bbox is misleadingly sparse; only after compacting away its empty spacer rows and columns do the frame interior dimensions reveal clean integer scale factors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, template, frames = _parse_scene(g)
    mask = template["compact"]
    th, tw = map(int, mask.shape)

    out = g.copy()
    seen_scales: set[int] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        ih, iw = map(int, interior.shape)
        assert ih > 0 and iw > 0
        assert ih % th == 0 and iw % tw == 0, "frame interior must be an integer multiple of the template"
        scale_h = ih // th
        scale_w = iw // tw
        assert scale_h == scale_w, "frame interior must use a single integer scale"
        scale = int(scale_h)
        assert scale >= 1
        seen_scales.add(scale)

        scaled = _scale_sprite(mask.astype(int), scale).astype(bool)
        interior[:, :] = bg
        interior[scaled] = int(frame["color"])

    assert len(seen_scales) == len(frames), "frame scales must be distinct"
    return out


def _random_connected_template(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(2, 4)
    width = rng.randint(2, 4)
    max_area = height * width - 1
    min_area = max(3, height + width - 2)
    if min_area > max_area:
        return None

    target_area = rng.randint(min_area, max_area)
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = {
        (start[0] + dr, start[1] + dc)
        for dr, dc in NEIGHBORS4
        if 0 <= start[0] + dr < height and 0 <= start[1] + dc < width
    }

    while frontier and len(cells) < target_area:
        choice = rng.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for dr, dc in NEIGHBORS4:
            nr = choice[0] + dr
            nc = choice[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target_area:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
        return None
    if not _mask_is_connected(mask):
        return None
    if bool(mask.all()):
        return None

    if height >= 3 and width >= 3:
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(mask, border):
            return None

    row_patterns = {tuple(map(int, row)) for row in mask}
    col_patterns = {tuple(map(int, col)) for col in mask.T}
    if len(row_patterns) == 1 and len(col_patterns) == 1:
        return None

    return mask


def _transform_mask(rng: random.Random, mask: np.ndarray) -> np.ndarray:
    transformed = mask.copy()
    if rng.random() < 0.5:
        transformed = np.flipud(transformed)
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    if rng.random() < 0.5:
        transformed = transformed.T
    return transformed.copy()


def _sparse_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    patch = np.full((2 * h - 1, 2 * w - 1), bg, dtype=int)
    patch[::2, ::2][mask] = int(color)
    return patch


def _frame_patch(inner_h: int, inner_w: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _expected_output(
    inp: np.ndarray,
    compact_mask: np.ndarray,
    frame_specs: list[dict],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]
        scale = int(spec["scale"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = bg
        scaled = _scale_sprite(compact_mask.astype(int), scale).astype(bool)
        interior[scaled] = int(spec["color"])
    return out


def _sparse_bbox_has_integer_ratio(mask: np.ndarray, scale: int) -> bool:
    th, tw = map(int, mask.shape)
    sparse_h = 2 * th - 1
    sparse_w = 2 * tw - 1
    inner_h = th * int(scale)
    inner_w = tw * int(scale)
    if inner_h % sparse_h != 0 or inner_w % sparse_w != 0:
        return False
    return inner_h // sparse_h == inner_w // sparse_w


def _valid_scales(mask: np.ndarray) -> list[int]:
    th, tw = map(int, mask.shape)
    scales: list[int] = []
    for scale in FRAME_SCALE_CHOICES:
        inner_h = th * scale
        inner_w = tw * scale
        if max(inner_h + 2, inner_w + 2) > 18:
            continue
        if _sparse_bbox_has_integer_ratio(mask, scale):
            continue
        scales.append(int(scale))
    return scales


def _find_placement(
    rng: random.Random,
    occupied: np.ndarray,
    patch_shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = map(int, patch_shape)
    positions: list[tuple[int, int]] = []

    for r0 in range(EDGE_MARGIN, h - ph - EDGE_MARGIN + 1):
        for c0 in range(EDGE_MARGIN, w - pw - EDGE_MARGIN + 1):
            rr0 = max(0, r0 - OBJECT_PADDING)
            cc0 = max(0, c0 - OBJECT_PADDING)
            rr1 = min(h, r0 + ph + OBJECT_PADDING)
            cc1 = min(w, c0 + pw + OBJECT_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))

    if not positions:
        return None
    return rng.choice(positions)


def _place_objects(
    rng: random.Random,
    objects: list[dict],
    grid_shape: tuple[int, int],
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full(grid_shape, bg, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    placed_specs: list[dict] = []

    order = sorted(range(len(objects)), key=lambda i: objects[i]["patch"].shape[0] * objects[i]["patch"].shape[1], reverse=True)
    for idx in order:
        obj = objects[idx]
        pos = _find_placement(rng, occupied, obj["patch"].shape)
        if pos is None:
            return None
        r0, c0 = pos
        ph, pw = obj["patch"].shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = obj["patch"]
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True

        spec = dict(obj["meta"])
        spec["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        placed_specs.append(spec)

    return grid, placed_specs


def _validate_generated_scene(
    grid: np.ndarray,
    compact_mask: np.ndarray,
    template_color: int,
    frame_specs: list[dict],
    bg: int,
) -> bool:
    try:
        parsed_bg, template, frames = _parse_scene(grid)
    except AssertionError:
        return False

    if parsed_bg != bg:
        return False
    if int(template["color"]) != int(template_color):
        return False
    if not np.array_equal(template["compact"], compact_mask):
        return False

    if len(frames) != len(frame_specs):
        return False
    expected_by_color = {int(spec["color"]): spec for spec in frame_specs}
    if set(expected_by_color) != {int(frame["color"]) for frame in frames}:
        return False

    used_scales: set[int] = set()
    for frame in frames:
        color = int(frame["color"])
        spec = expected_by_color[color]
        if tuple(map(int, frame["bbox"])) != tuple(map(int, spec["bbox"])):
            return False
        r0, c0, r1, c1 = frame["bbox"]
        ih = int(r1 - r0 - 1)
        iw = int(c1 - c0 - 1)
        th, tw = map(int, compact_mask.shape)
        if ih % th != 0 or iw % tw != 0 or ih // th != iw // tw:
            return False
        scale = ih // th
        if scale != int(spec["scale"]):
            return False
        if _sparse_bbox_has_integer_ratio(compact_mask, scale):
            return False
        used_scales.add(int(scale))

    if len(used_scales) != len(frame_specs):
        return False

    expected = _expected_output(grid, compact_mask, frame_specs, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    rng = random.Random()

    compact_mask = _random_connected_template(rng)
    if compact_mask is None:
        return None
    compact_mask = _transform_mask(rng, compact_mask)

    valid_scales = _valid_scales(compact_mask)
    if len(valid_scales) < 2:
        return None

    max_frames = min(3, len(valid_scales))
    frame_count = rng.choice([count for count in FRAME_COUNT_CHOICES if count <= max_frames])
    chosen_scales = sorted(rng.sample(valid_scales, frame_count))

    colors = rng.sample(PALETTE, frame_count + 2)
    bg = int(colors[0])
    template_color = int(colors[1])
    frame_colors = [int(c) for c in colors[2:]]

    template_patch = _sparse_patch(compact_mask, template_color, bg)
    objects: list[dict] = [
        {
            "patch": template_patch,
            "meta": {"kind": "template", "color": template_color},
        }
    ]
    frame_specs: list[dict] = []
    th, tw = map(int, compact_mask.shape)
    for scale, color in zip(chosen_scales, frame_colors):
        inner_h = th * scale
        inner_w = tw * scale
        patch = _frame_patch(inner_h, inner_w, color, bg)
        objects.append(
            {
                "patch": patch,
                "meta": {"kind": "frame", "color": color, "scale": int(scale)},
            }
        )

    box_area = sum((obj["patch"].shape[0] + 2) * (obj["patch"].shape[1] + 2) for obj in objects)
    longest_h = max(obj["patch"].shape[0] for obj in objects) + 2 * EDGE_MARGIN
    longest_w = max(obj["patch"].shape[1] for obj in objects) + 2 * EDGE_MARGIN
    side_floor = int(np.ceil(np.sqrt(box_area * 1.35)))
    min_h = max(16, longest_h, side_floor - 2)
    min_w = max(16, longest_w, side_floor - 2)
    if min_h > MAX_GRID_SIDE or min_w > MAX_GRID_SIDE:
        return None

    grid_h = rng.randint(min_h, MAX_GRID_SIDE)
    grid_w = rng.randint(min_w, MAX_GRID_SIDE)
    placed = _place_objects(rng, objects, (grid_h, grid_w), bg)
    if placed is None:
        return None
    grid, placed_specs = placed

    actual_frame_specs: list[dict] = []
    saw_template = False
    for spec in placed_specs:
        if spec["kind"] == "template":
            saw_template = True
        else:
            actual_frame_specs.append(spec)
    if not saw_template:
        return None

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None
    if not _validate_generated_scene(grid, compact_mask, template_color, actual_frame_specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-template calibration task")
