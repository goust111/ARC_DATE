import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
KEY_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Canonical lockhole shapes.  Each shape is defined in its "right-side" form,
# where the rectangle would attach along the shape's left edge.
LOCK_SHAPES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)

SIDES = ("right", "bottom", "left", "top")
SIDE_TO_ROTATION = {"right": 0, "bottom": 1, "left": 2, "top": 3}

MATCHED_COUNT_RANGE = (1, 3)
UNMATCHED_COUNT_RANGE = (1, 2)
UNMATCHED_PROBABILITY = 0.5
CANVAS_SIDE_MIN = 18
CANVAS_SIDE_MAX = 30
MAX_GENERATION_ATTEMPTS = 300


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [tuple(map(int, coord)) for coord in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate_coords(coords, turns: int) -> tuple[tuple[int, int], ...]:
    pts = [tuple(map(int, coord)) for coord in coords]
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_coords(pts)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    foreground = grid[grid != bg]
    assert foreground.size > 0, "expected at least one non-background color"
    values, counts = np.unique(foreground, return_counts=True)
    return int(values[np.argmax(counts)])


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _valid_protrusion(mask: np.ndarray, protrusion: np.ndarray, side: str, depth: int) -> bool:
    if not protrusion.any():
        return False
    if len(_component_coords(protrusion)) != 1:
        return False

    h, w = mask.shape
    area = int(protrusion.sum())
    if area > 6:
        return False

    if side == "top":
        if not protrusion[0, :].any() or area >= depth * w:
            return False
        contact = np.where(protrusion[depth - 1, :])[0]
    elif side == "bottom":
        if not protrusion[h - 1, :].any() or area >= depth * w:
            return False
        contact = np.where(protrusion[h - depth, :])[0]
    elif side == "left":
        if not protrusion[:, 0].any() or area >= depth * h:
            return False
        contact = np.where(protrusion[:, depth - 1])[0]
    else:
        if not protrusion[:, w - 1].any() or area >= depth * h:
            return False
        contact = np.where(protrusion[:, w - depth])[0]

    if contact.size == 0 or contact.size > 3:
        return False
    return bool(contact.size == 1 or np.all(np.diff(contact) == 1))


def _extract_lock_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    h, w = mask.shape

    candidates: list[tuple[tuple[int, int], ...]] = []
    for depth in (1, 2):
        if h - depth >= 2:
            if np.all(mask[depth:, :]):
                protrusion = mask.copy()
                protrusion[depth:, :] = False
                if _valid_protrusion(mask, protrusion, "top", depth):
                    candidates.append(_normalize_coords(np.argwhere(protrusion)))
            if np.all(mask[: h - depth, :]):
                protrusion = mask.copy()
                protrusion[: h - depth, :] = False
                if _valid_protrusion(mask, protrusion, "bottom", depth):
                    candidates.append(_normalize_coords(np.argwhere(protrusion)))
        if w - depth >= 2:
            if np.all(mask[:, depth:]):
                protrusion = mask.copy()
                protrusion[:, depth:] = False
                if _valid_protrusion(mask, protrusion, "left", depth):
                    candidates.append(_normalize_coords(np.argwhere(protrusion)))
            if np.all(mask[:, : w - depth]):
                protrusion = mask.copy()
                protrusion[:, : w - depth] = False
                if _valid_protrusion(mask, protrusion, "right", depth):
                    candidates.append(_normalize_coords(np.argwhere(protrusion)))

    unique_candidates: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates[0] if len(unique_candidates) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """Match each colored key to the lockhole-shaped protrusion on a grey container and paint that container's interior.

    Input structure:
    - Background is black.
    - Grey one-pixel-thick outlines enclose black container interiors; each interior is a rectangle with one small protruding lockhole on exactly one side.
    - Separate non-grey, non-black 4-connected colored sprites are keys, and a key matches a lockhole shape up to 90-degree rotations.

    Rule:
    - Find every enclosed black region whose shape is a full rectangle plus one small side protrusion.
    - For each colored key, locate the unique enclosed region whose protruding part has the same shape after rotation.
    - Remove the matched key from its original location and fill that entire enclosed region, including the protrusion, with the key's color.
    - Any container whose protrusion has no matching key remains unchanged.

    Key insight:
    - The key identifies a container by the shape of the small protruding lockhole only; once that small shape is matched, the whole enclosed region receives the key's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    wall = _wall_color(g, bg)
    out = g.copy()

    holes: list[dict[str, object]] = []
    for comp in _component_coords(g == bg):
        if _touches_boundary(comp, g.shape):
            continue
        lock_shape = _extract_lock_shape(comp)
        if lock_shape is not None:
            holes.append({"coords": comp, "shape": lock_shape})

    used_holes: set[int] = set()
    for color in np.unique(g):
        color = int(color)
        if color in {bg, wall}:
            continue
        for comp in _component_coords(g == color):
            base_shape = _normalize_coords(comp)
            rotations = {_rotate_coords(base_shape, turns) for turns in range(4)}
            matches = [
                idx
                for idx, hole in enumerate(holes)
                if idx not in used_holes and hole["shape"] in rotations
            ]
            if len(matches) != 1:
                continue
            hole_idx = matches[0]
            used_holes.add(hole_idx)
            out[comp[:, 0], comp[:, 1]] = bg
            hole_coords = holes[hole_idx]["coords"]
            out[hole_coords[:, 0], hole_coords[:, 1]] = color

    return out


def _build_container(
    shape: tuple[tuple[int, int], ...],
    side: str,
    core_h: int,
    core_w: int,
) -> dict[str, np.ndarray] | None:
    oriented = _rotate_coords(shape, SIDE_TO_ROTATION[side])
    protrusion_h = max(r for r, _ in oriented) + 1
    protrusion_w = max(c for _, c in oriented) + 1

    if side in {"left", "right"}:
        if core_h < protrusion_h + 2:
            return None
        anchor = random.randint(1, core_h - protrusion_h - 1)
        total_h = core_h
        total_w = core_w + protrusion_w
        region = np.zeros((total_h, total_w), dtype=bool)
        if side == "right":
            region[:, :core_w] = True
            for r, c in oriented:
                region[anchor + r, core_w + c] = True
        else:
            region[:, protrusion_w:] = True
            for r, c in oriented:
                region[anchor + r, c] = True
    else:
        if core_w < protrusion_w + 2:
            return None
        anchor = random.randint(1, core_w - protrusion_w - 1)
        total_h = core_h + protrusion_h
        total_w = core_w
        region = np.zeros((total_h, total_w), dtype=bool)
        if side == "bottom":
            region[:core_h, :] = True
            for r, c in oriented:
                region[core_h + r, anchor + c] = True
        else:
            region[protrusion_h:, :] = True
            for r, c in oriented:
                region[r, anchor + c] = True

    region = np.pad(region, 1)
    wall = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and not region[nr, nc]:
                wall[nr, nc] = True
    wall &= ~region
    return {"region": region, "wall": wall}


def _place_bbox(
    reserved: np.ndarray,
    box_h: int,
    box_w: int,
    attempts: int = 500,
) -> tuple[int, int] | None:
    canvas_h, canvas_w = reserved.shape
    for _ in range(attempts):
        top = random.randint(0, canvas_h - box_h)
        left = random.randint(0, canvas_w - box_w)
        r0 = max(0, top - 1)
        r1 = min(canvas_h, top + box_h + 1)
        c0 = max(0, left - 1)
        c1 = min(canvas_w, left + box_w + 1)
        if reserved[r0:r1, c0:c1].any():
            continue
        reserved[r0:r1, c0:c1] = True
        return top, left
    return None


def _make_container_spec(shape: tuple[tuple[int, int], ...]) -> dict[str, object] | None:
    side = random.choice(SIDES)
    oriented = _rotate_coords(shape, SIDE_TO_ROTATION[side])
    protrusion_h = max(r for r, _ in oriented) + 1
    protrusion_w = max(c for _, c in oriented) + 1

    min_core_h = max(4, protrusion_h + 2 if side in {"left", "right"} else 4)
    min_core_w = max(4, protrusion_w + 2 if side in {"top", "bottom"} else 4)
    max_core_h = min(8, min_core_h + 3)
    max_core_w = min(8, min_core_w + 4)
    core_h = random.randint(min_core_h, max_core_h)
    core_w = random.randint(min_core_w, max_core_w)

    container = _build_container(shape, side, core_h, core_w)
    if container is None:
        return None
    return {"shape": shape, "side": side, **container}


def _expected_output(
    grid: np.ndarray,
    matched_specs: list[dict[str, object]],
    matched_positions: list[tuple[int, int]],
    key_specs: list[dict[str, object]],
    key_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for key_spec, (top, left) in zip(key_specs, key_positions):
        for dr, dc in key_spec["coords"]:
            out[top + dr, left + dc] = BG_COLOR
    for spec, (top, left), key_spec in zip(matched_specs, matched_positions, key_specs):
        region = spec["region"]
        out[top : top + region.shape[0], left : left + region.shape[1]][region] = key_spec["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    matched_count = random.randint(*MATCHED_COUNT_RANGE)
    unmatched_count = 0
    if random.random() < UNMATCHED_PROBABILITY:
        unmatched_count = random.randint(*UNMATCHED_COUNT_RANGE)

    total_containers = matched_count + unmatched_count
    if total_containers > len(LOCK_SHAPES):
        return None

    chosen_shapes = random.sample(list(LOCK_SHAPES), total_containers)
    container_specs: list[dict[str, object]] = []
    for shape in chosen_shapes:
        spec = _make_container_spec(shape)
        if spec is None:
            return None
        container_specs.append(spec)

    key_colors = random.sample(list(KEY_COLOR_POOL), matched_count)
    key_specs: list[dict[str, object]] = []
    for shape, color in zip(chosen_shapes[:matched_count], key_colors):
        rotated = _rotate_coords(shape, random.randint(0, 3))
        key_h = max(r for r, _ in rotated) + 1
        key_w = max(c for _, c in rotated) + 1
        key_specs.append({"coords": rotated, "color": color, "h": key_h, "w": key_w})

    max_h = max([spec["wall"].shape[0] for spec in container_specs] + [key["h"] for key in key_specs])
    max_w = max([spec["wall"].shape[1] for spec in container_specs] + [key["w"] for key in key_specs])
    footprint_area = sum(spec["wall"].size for spec in container_specs) + sum(key["h"] * key["w"] for key in key_specs)
    base_side = max(CANVAS_SIDE_MIN, int(np.ceil(np.sqrt(2.3 * footprint_area))))
    min_h = max(max_h + 2, min(base_side, CANVAS_SIDE_MAX))
    min_w = max(max_w + 2, min(base_side, CANVAS_SIDE_MAX))
    if min_h > CANVAS_SIDE_MAX or min_w > CANVAS_SIDE_MAX:
        return None

    canvas_h = random.randint(min_h, CANVAS_SIDE_MAX)
    canvas_w = random.randint(min_w, CANVAS_SIDE_MAX)
    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    reserved = np.zeros((canvas_h, canvas_w), dtype=bool)

    order = sorted(range(len(container_specs)), key=lambda idx: -container_specs[idx]["wall"].size)
    container_positions: list[tuple[int, int] | None] = [None] * len(container_specs)
    for idx in order:
        spec = container_specs[idx]
        wall = spec["wall"]
        top_left = _place_bbox(reserved, wall.shape[0], wall.shape[1])
        if top_left is None:
            return None
        top, left = top_left
        grid[top : top + wall.shape[0], left : left + wall.shape[1]][wall] = WALL_COLOR
        container_positions[idx] = top_left

    key_positions: list[tuple[int, int]] = []
    for key_spec in key_specs:
        top_left = _place_bbox(reserved, key_spec["h"], key_spec["w"])
        if top_left is None:
            return None
        top, left = top_left
        for dr, dc in key_spec["coords"]:
            grid[top + dr, left + dc] = key_spec["color"]
        key_positions.append(top_left)

    if _most_common_color(grid) != BG_COLOR:
        return None
    if _wall_color(grid, BG_COLOR) != WALL_COLOR:
        return None

    matched_specs = container_specs[:matched_count]
    matched_positions = [container_positions[idx] for idx in range(matched_count)]
    assert all(pos is not None for pos in matched_positions)
    expected = _expected_output(
        grid,
        matched_specs,
        matched_positions,  # type: ignore[arg-type]
        key_specs,
        key_positions,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key puzzle")
