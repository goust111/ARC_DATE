from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
ROTATION_K = {"up": 0, "left": 1, "down": 2, "right": 3}
SHAPE_FAMILIES = ("rect", "side_arm", "back_rib")
GRID_HEIGHT_RANGE = (15, 30)
GRID_WIDTH_RANGE = (15, 30)
OBJECT_COUNT_RANGE = (1, 4)
CONTAINER_HEIGHT_RANGE = (4, 8)
CONTAINER_WIDTH_RANGE = (4, 9)
EDGE_BIAS_PROBABILITY = 0.35
MIN_VISIBLE_STAMP_CELLS = 3
PLACEMENT_RETRIES = 160
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _infer_bg_and_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _majority_color(grid)
    non_bg = grid[grid != bg]
    assert non_bg.size > 0, "expected non-background objects"
    colors, counts = np.unique(non_bg, return_counts=True)
    wall = int(colors[int(np.argmax(counts))])
    return int(bg), wall


def _opening_direction(wall_mask: np.ndarray) -> str:
    edge_full = {
        "up": bool(np.all(wall_mask[0, :])),
        "down": bool(np.all(wall_mask[-1, :])),
        "left": bool(np.all(wall_mask[:, 0])),
        "right": bool(np.all(wall_mask[:, -1])),
    }
    openings = [direction for direction, full in edge_full.items() if not full]
    assert len(openings) == 1, "container must have exactly one open bbox edge"
    return openings[0]


def _mirror_mask_and_seed(
    mask: np.ndarray, seed_rc: tuple[int, int], direction: str
) -> tuple[np.ndarray, tuple[int, int]]:
    sr, sc = seed_rc
    h, w = mask.shape
    if direction in {"up", "down"}:
        return np.flipud(mask), (h - 1 - sr, sc)
    return np.fliplr(mask), (sr, w - 1 - sc)


def _stamp_origin(
    bbox: tuple[int, int, int, int],
    seed_rc: tuple[int, int],
    mirrored_seed_rc: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    sr, sc = seed_rc
    msr, msc = mirrored_seed_rc
    if direction == "up":
        anchor_r, anchor_c = r0 - 1, c0 + sc
    elif direction == "down":
        anchor_r, anchor_c = r1 + 1, c0 + sc
    elif direction == "left":
        anchor_r, anchor_c = r0 + sr, c0 - 1
    else:
        anchor_r, anchor_c = r0 + sr, c1 + 1
    return anchor_r - msr, anchor_c - msc


def _stamp_cells(
    cavity_mask: np.ndarray,
    seed_rc: tuple[int, int],
    direction: str,
    bbox_r0: int,
    bbox_c0: int,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = cavity_mask.shape
    mirrored_mask, mirrored_seed = _mirror_mask_and_seed(cavity_mask, seed_rc, direction)
    stamp_r0, stamp_c0 = _stamp_origin(
        (bbox_r0, bbox_c0, bbox_r0 + h - 1, bbox_c0 + w - 1),
        seed_rc,
        mirrored_seed,
        direction,
    )
    gh, gw = grid_shape
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mirrored_mask):
        gr = stamp_r0 + int(rr)
        gc = stamp_c0 + int(cc)
        if 0 <= gr < gh and 0 <= gc < gw:
            cells.add((gr, gc))
    return cells


def _paint_stamp(
    out: np.ndarray,
    cavity_mask: np.ndarray,
    seed_rc: tuple[int, int],
    direction: str,
    bbox_r0: int,
    bbox_c0: int,
    seed_color: int,
    bg: int,
    wall: int,
) -> None:
    h, w = cavity_mask.shape
    mirrored_mask, mirrored_seed = _mirror_mask_and_seed(cavity_mask, seed_rc, direction)
    stamp_r0, stamp_c0 = _stamp_origin(
        (bbox_r0, bbox_c0, bbox_r0 + h - 1, bbox_c0 + w - 1),
        seed_rc,
        mirrored_seed,
        direction,
    )
    gh, gw = out.shape
    for rr, cc in np.argwhere(mirrored_mask):
        gr = stamp_r0 + int(rr)
        gc = stamp_c0 + int(cc)
        if not (0 <= gr < gh and 0 <= gc < gw):
            continue
        current = int(out[gr, gc])
        assert current != wall, "stamp must not overwrite a wall"
        assert current in {bg, seed_color}, "stamps must not conflict with other colors"
        out[gr, gc] = seed_color


def _render_from_specs(
    grid_shape: tuple[int, int], placements: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(grid_shape, BG_COLOR, dtype=int)
    for placement in placements:
        wall_mask = placement["wall_mask"]
        seed_r, seed_c = placement["seed_rc"]
        r0, c0 = placement["origin"]
        color = placement["seed_color"]
        h, w = wall_mask.shape
        region = inp[r0 : r0 + h, c0 : c0 + w]
        region[wall_mask] = WALL_COLOR
        assert region[seed_r, seed_c] == BG_COLOR
        region[seed_r, seed_c] = color

    out = inp.copy()
    for placement in placements:
        _paint_stamp(
            out,
            placement["cavity_mask"],
            placement["seed_rc"],
            placement["direction"],
            placement["origin"][0],
            placement["origin"][1],
            placement["seed_color"],
            BG_COLOR,
            WALL_COLOR,
        )
    return inp, out


def _expanded_bbox(cells: set[tuple[int, int]], grid_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h, w = grid_shape
    return (
        max(0, min(rows) - 1),
        max(0, min(cols) - 1),
        min(h - 1, max(rows) + 1),
        min(w - 1, max(cols) + 1),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _canonical_open_up() -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    h = random.randint(*CONTAINER_HEIGHT_RANGE)
    w = random.randint(*CONTAINER_WIDTH_RANGE)
    wall_mask = np.zeros((h, w), dtype=bool)
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True
    wall_mask[-1, :] = True

    family = random.choice(SHAPE_FAMILIES)
    if family == "side_arm":
        arm_r = random.randint(1, h - 2)
        arm_len = random.randint(1, w - 3)
        if random.random() < 0.5:
            wall_mask[arm_r, : arm_len + 1] = True
        else:
            wall_mask[arm_r, w - 1 - arm_len :] = True
    elif family == "back_rib":
        arm_c = random.randint(1, w - 2)
        arm_len = random.randint(1, h - 3)
        wall_mask[h - 1 - arm_len :, arm_c] = True

    cavity_mask = ~wall_mask
    assert len(_components(wall_mask)) == 1
    assert len(_components(cavity_mask)) == 1
    assert _opening_direction(wall_mask) == "up"

    seed_candidates = [(0, c) for c in range(w) if cavity_mask[0, c]]
    assert seed_candidates, "canonical container must expose a top opening"
    seed_rc = random.choice(seed_candidates)
    return wall_mask, cavity_mask, seed_rc, family


def _rotated_container() -> dict:
    wall_mask, cavity_mask, seed_rc, family = _canonical_open_up()
    direction = random.choice(DIRECTIONS)
    k = ROTATION_K[direction]

    seed_mask = np.zeros_like(wall_mask, dtype=bool)
    seed_mask[seed_rc] = True
    wall_mask = np.rot90(wall_mask, k)
    cavity_mask = np.rot90(cavity_mask, k)
    seed_mask = np.rot90(seed_mask, k)
    seed_r, seed_c = map(int, np.argwhere(seed_mask)[0])

    assert len(_components(wall_mask)) == 1
    assert len(_components(cavity_mask)) == 1
    assert _opening_direction(wall_mask) == direction

    return {
        "wall_mask": wall_mask,
        "cavity_mask": cavity_mask,
        "seed_rc": (seed_r, seed_c),
        "direction": direction,
        "family": family,
    }


def _placement_cells(
    spec: dict, origin: tuple[int, int], grid_shape: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    wall_mask = spec["wall_mask"]
    seed_r, seed_c = spec["seed_rc"]
    r0, c0 = origin
    input_cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(wall_mask)}
    input_cells.add((r0 + seed_r, c0 + seed_c))
    stamp_cells = _stamp_cells(
        spec["cavity_mask"],
        spec["seed_rc"],
        spec["direction"],
        r0,
        c0,
        grid_shape,
    )
    return input_cells, stamp_cells


def _biased_origin(spec: dict, grid_shape: tuple[int, int]) -> tuple[int, int]:
    gh, gw = grid_shape
    h, w = spec["wall_mask"].shape
    max_r = gh - h
    max_c = gw - w
    direction = spec["direction"]

    if random.random() < EDGE_BIAS_PROBABILITY:
        if direction == "up":
            r0 = random.randint(0, min(2, max_r))
            c0 = random.randint(0, max_c)
        elif direction == "down":
            r0 = random.randint(max(0, max_r - 2), max_r)
            c0 = random.randint(0, max_c)
        elif direction == "left":
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, min(2, max_c))
        else:
            r0 = random.randint(0, max_r)
            c0 = random.randint(max(0, max_c - 2), max_c)
    else:
        r0 = random.randint(0, max_r)
        c0 = random.randint(0, max_c)
    return r0, c0


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    grid_shape = (grid_h, grid_w)

    max_objects = OBJECT_COUNT_RANGE[1]
    if grid_h * grid_w < 380:
        max_objects = min(max_objects, 3)
    target_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    seed_colors = random.sample(SEED_COLORS, target_count)

    protected_boxes: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []

    for color in seed_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _rotated_container()
            origin = _biased_origin(spec, grid_shape)
            input_cells, stamp_cells = _placement_cells(spec, origin, grid_shape)
            if len(stamp_cells) < MIN_VISIBLE_STAMP_CELLS:
                continue
            if input_cells & stamp_cells:
                continue
            all_cells = input_cells | stamp_cells
            bbox = _expanded_bbox(all_cells, grid_shape)
            if any(_boxes_overlap(bbox, other) for other in protected_boxes):
                continue

            protected_boxes.append(bbox)
            placements.append(
                {
                    **spec,
                    "origin": origin,
                    "seed_color": int(color),
                }
            )
            placed = True
            break
        if not placed:
            return None

    inp, expected = _render_from_specs(grid_shape, placements)
    if _majority_color(inp) != BG_COLOR:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed outward from its grey container by stamping a mirrored copy of the container cavity.

    Input structure:
    - The background is black.
    - One or more grey 4-connected container components sit on the grid; each component has wall thickness one cell and exactly one open side.
    - Each container's bounding box contains one connected non-grey cavity region, and exactly one colored seed pixel lies inside that cavity.

    Rule:
    - For each grey container, take its tight bounding box and identify the unique bbox edge that is not completely covered by grey cells; that edge is the opening direction.
    - Treat every non-grey cell inside the bbox as the cavity template, including the seed cell's location inside that template.
    - Mirror the cavity template across the axis perpendicular to the opening direction, then place the mirrored copy outside the opening so that the mirrored seed position sits one cell beyond the opening edge and stays aligned with the original seed along the opening edge.
    - Paint every in-bounds cell of that mirrored copy with the seed's color, while keeping the original container and seed unchanged.

    Key insight:
    - The grey shape is only a mold: first recover the cavity inside each container, then use the opening direction to decide which way to mirror and where to stamp the seed-colored copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, wall = _infer_bg_and_wall(g)
    out = g.copy()

    containers = _components(g == wall)
    assert containers, "expected at least one grey container"

    for comp in containers:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        wall_mask = sub == wall
        direction = _opening_direction(wall_mask)

        cavity_mask = sub != wall
        cavity_components = _components(cavity_mask)
        assert len(cavity_components) == 1, "container cavity must be a single connected region"

        seed_positions = np.argwhere((sub != bg) & (sub != wall))
        assert len(seed_positions) == 1, "each container must contain exactly one seed"
        seed_r, seed_c = map(int, seed_positions[0])
        seed_color = int(sub[seed_r, seed_c])
        assert cavity_mask[seed_r, seed_c]

        _paint_stamp(out, cavity_mask, (seed_r, seed_c), direction, r0, c0, seed_color, bg, wall)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
