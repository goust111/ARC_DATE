from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8

SPRITE_COLORS = (BLUE, RED, GREEN, ORANGE, CYAN)
DIRECTION_VALUES = (-1, 0, 1)
ALL_DIRECTIONS = [(dr, dc) for dr in DIRECTION_VALUES for dc in DIRECTION_VALUES]
CORE_DIRECTIONS = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
BOX_AREAS = (3, 4, 6, 8, 9, 12, 16)
MAX_GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> np.ndarray:
    h, w = patch.shape
    r0 = int(r0)
    c0 = int(c0)
    rr0 = max(0, r0) if clip else r0
    cc0 = max(0, c0) if clip else c0
    rr1 = min(dst.shape[0], r0 + h) if clip else r0 + h
    cc1 = min(dst.shape[1], c0 + w) if clip else c0 + w
    if rr0 >= rr1 or cc0 >= cc1:
        return dst
    if not clip and (rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]):
        raise ValueError("paste out of bounds")
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    view = patch[pr0:pr1, pc0:pc1]
    if transparent is None:
        dst[rr0:rr1, cc0:cc1] = view
    else:
        mask = view != int(transparent)
        dst_view = dst[rr0:rr1, cc0:cc1]
        dst_view[mask] = view[mask]
    return dst


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def find_components_monochrome(
    grid: np.ndarray,
    *,
    exclude: set[int],
    connectivity: int = 4,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in exclude or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                iterator = neighbors4(rr, cc) if connectivity == 4 else neighbors8(rr, cc)
                for nr, nc in iterator:
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0, r1, c1 = bbox_coords(arr)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = color
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "area": int(arr.shape[0]),
                    "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    return components


def point_bbox_manhattan(box: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = map(int, box)
    pr, pc = map(int, point)
    dr = 0 if r0 <= pr <= r1 else min(abs(pr - r0), abs(pr - r1))
    dc = 0 if c0 <= pc <= c1 else min(abs(pc - c0), abs(pc - c1))
    return int(dr + dc)


def canvas_info(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GRAY)
    assert coords.size > 0, "missing gray canvas"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (9, 9), "canvas must be 9x9"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY), "canvas must be filled"
    assert coords.shape[0] == 81, "expected exactly one filled gray canvas"
    return r0, c0, r0 + 4, c0 + 4


def direction_from_control(pr: int, pc: int, cr: int, cc: int) -> tuple[int, int]:
    dr = -1 if pr < cr else (1 if pr > cr else 0)
    dc = -1 if pc < cc else (1 if pc > cc else 0)
    return int(dr), int(dc)


def slot_top_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return 3 * (int(dr) + 1), 3 * (int(dc) + 1)


def parse_instance(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    canvas_r0, canvas_c0, cr, cc = canvas_info(g)

    controls = [tuple(map(int, rc)) for rc in np.argwhere(g == MAGENTA)]
    assert 3 <= len(controls) <= 5, "expected 3-5 control pixels"

    sprites = find_components_monochrome(g, exclude={BLACK, GRAY, MAGENTA}, connectivity=4)
    assert len(sprites) == len(controls), "expected one control per sprite"

    bbox_areas = [sprite["bbox_area"] for sprite in sprites]
    assert len(set(bbox_areas)) == len(bbox_areas), "bbox areas must be unique"

    used_controls: set[tuple[int, int]] = set()
    used_directions: set[tuple[int, int]] = set()
    for sprite in sprites:
        distances = sorted(
            (point_bbox_manhattan(sprite["bbox"], control), control)
            for control in controls
        )
        assert len(distances) >= 1
        best_dist, best_control = distances[0]
        assert best_dist in (1, 2), "control must be 1-2 cells outside its sprite bbox"
        if len(distances) > 1:
            assert best_dist < distances[1][0], "nearest control must be unique"
        assert best_control not in used_controls, "control reuse is ambiguous"
        used_controls.add(best_control)
        sprite["control"] = best_control
        direction = direction_from_control(best_control[0], best_control[1], cr, cc)
        assert direction not in used_directions, "control directions must be unique"
        used_directions.add(direction)
        sprite["direction"] = direction

        r0, c0, r1, c1 = sprite["bbox"]
        assert r1 < g.shape[0] // 2 + 1, "sprites must live in the upper half region"
        assert r1 < canvas_r0 or c1 < canvas_c0 or c0 > canvas_c0 + 8, "sprites must stay outside the canvas"

    return (canvas_r0, canvas_c0, cr, cc), sprites


def compose_output(sprites: list[dict], *, descending: bool = True) -> np.ndarray:
    ordered = sorted(sprites, key=lambda sprite: int(sprite["bbox_area"]), reverse=descending)
    out = np.full((9, 9), BLACK, dtype=int)
    for sprite in ordered:
        sr, sc = slot_top_left(sprite["direction"])
        paste(out, sprite["patch"], sr, sc, transparent=BLACK, clip=False)
    return out


def overlap_count(sprites: list[dict]) -> int:
    occupied_masks = []
    for sprite in sprites:
        mask = sprite["patch"] != BLACK
        sr, sc = slot_top_left(sprite["direction"])
        placed = np.zeros((9, 9), dtype=bool)
        paste(placed, mask.astype(int), sr, sc, transparent=0, clip=False)
        occupied_masks.append(placed)
    total = 0
    for i in range(len(occupied_masks)):
        for j in range(i + 1, len(occupied_masks)):
            total += int(np.count_nonzero(occupied_masks[i] & occupied_masks[j]))
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each sprite into the canvas slot chosen by its control pixel direction, with smaller bbox-areas covering larger ones.

    Input structure:
    - A black background contains 3-5 small monochromatic 4-connected sprites in the upper half, each paired with one nearby magenta control pixel.
    - A filled gray 9x9 square marks the destination canvas; its center defines the reference frame for relative directions.
    - Each sprite has a unique bbox area and each control pixel occupies a unique relative direction around the canvas center.

    Rule:
    - Find the gray 9x9 canvas and its center.
    - Associate every sprite with the unique nearest magenta control pixel by Manhattan distance to the sprite bbox.
    - Convert the control's row/column relation to the canvas center into one of the 3x3 direction slots, then align the sprite bbox upper-left to that slot's upper-left corner inside the 9x9 output.
    - Render sprites from largest bbox area to smallest bbox area so later smaller sprites overwrite earlier larger ones, and return only the 9x9 canvas contents on a black background.

    Key insight:
    - The control pixel decides where a sprite lands, while bbox-area ordering decides which color survives wherever two stamped sprites overlap.
    """
    _canvas, sprites = parse_instance(grid)
    return compose_output(sprites, descending=True)


SHAPE_LIBRARY: dict[int, tuple[np.ndarray, ...]] = {
    3: (
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=int),
    ),
    6: (
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1], [1, 1]], dtype=int),
    ),
    8: (
        np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=int),
    ),
    9: (
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    ),
    12: (
        np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=int),
    ),
    16: (
        np.array([[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=int),
    ),
}


def randomize_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def fits_canvas_slot(direction: tuple[int, int], mask: np.ndarray) -> bool:
    r0, c0 = slot_top_left(direction)
    h, w = mask.shape
    return r0 + h <= 9 and c0 + w <= 9


def choose_mask(area: int, direction: tuple[int, int]) -> np.ndarray | None:
    options = [mask for mask in SHAPE_LIBRARY[area] if fits_canvas_slot(direction, mask)]
    if not options:
        return None
    return randomize_mask(random.choice(options))


def adjacent_directions(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr0, dc0 = direction
    out = []
    for dr, dc in ALL_DIRECTIONS:
        if (dr, dc) == direction:
            continue
        if abs(dr - dr0) <= 1 and abs(dc - dc0) <= 1:
            out.append((dr, dc))
    return out


def sample_directions(n: int) -> list[tuple[int, int]]:
    large_dir = random.choice(CORE_DIRECTIONS)
    neighbor = random.choice(adjacent_directions(large_dir))
    remaining = [d for d in ALL_DIRECTIONS if d not in {large_dir, neighbor}]
    weights = []
    for dr, dc in remaining:
        weight = 1.0
        if dr == 0:
            weight += 0.7
        if dr == 1:
            weight += 0.7
        if dc == 0:
            weight += 0.2
        weights.append(weight)
    chosen = [large_dir, neighbor]
    while len(chosen) < n:
        pick = random.choices(remaining, weights=weights, k=1)[0]
        idx = remaining.index(pick)
        chosen.append(pick)
        remaining.pop(idx)
        weights.pop(idx)
    random.shuffle(chosen)
    return chosen


def build_output_specs() -> list[dict] | None:
    n = random.randint(3, 5)
    for _ in range(400):
        directions = sample_directions(n)
        colors = random.sample(list(SPRITE_COLORS), n)

        large_dir = next((direction for direction in directions if direction in CORE_DIRECTIONS), None)
        if large_dir is None:
            continue

        specs: list[dict] = []
        used_areas = {16}
        mask16 = choose_mask(16, large_dir)
        if mask16 is None:
            continue
        specs.append(
            {
                "color": colors[len(specs)],
                "direction": large_dir,
                "bbox_area": 16,
                "patch": mask16 * colors[len(specs)],
            }
        )

        other_dirs = [d for d in directions if d != large_dir]
        random.shuffle(other_dirs)
        available_areas = [area for area in BOX_AREAS if area not in used_areas]
        for direction in other_dirs:
            viable_areas = [area for area in available_areas if choose_mask(area, direction) is not None]
            if not viable_areas:
                break
            area = random.choice(viable_areas)
            mask = choose_mask(area, direction)
            assert mask is not None
            color = colors[len(specs)]
            specs.append(
                {
                    "color": color,
                    "direction": direction,
                    "bbox_area": area,
                    "patch": mask * color,
                }
            )
            available_areas.remove(area)
        if len(specs) != n:
            continue
        if overlap_count(specs) == 0:
            continue
        desc = compose_output(specs, descending=True)
        asc = compose_output(specs, descending=False)
        if np.array_equal(desc, asc):
            continue
        return specs
    return None


def bbox_ring_positions(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    out: list[tuple[int, int]] = []
    for r in range(max(0, r0 - 2), min(shape[0], r1 + 3)):
        for c in range(max(0, c0 - 2), min(shape[1], c1 + 3)):
            d = point_bbox_manhattan(box, (r, c))
            if d in (1, 2):
                out.append((r, c))
    return out


def group_cells_for_candidate(
    patch: np.ndarray,
    top: int,
    left: int,
    control: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = {(top + r, left + c) for r, c in np.argwhere(patch != BLACK)}
    cells.add((int(control[0]), int(control[1])))
    return cells


def touches_existing(group: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in group:
        if (r, c) in occupied:
            return True
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def control_matches_direction(control: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> bool:
    return direction_from_control(control[0], control[1], center[0], center[1]) == direction


def placement_candidates(
    grid_shape: tuple[int, int],
    canvas_box: tuple[int, int, int, int],
    center: tuple[int, int],
    spec: dict,
) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = spec["patch"].shape
    grid_h, grid_w = grid_shape
    upper_limit = grid_h // 2
    r_max = upper_limit - h
    if r_max < 0:
        return []
    r_canvas0, c_canvas0, r_canvas1, c_canvas1 = canvas_box
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for top in range(0, r_max + 1):
        bottom = top + h - 1
        for left in range(0, grid_w - w + 1):
            right = left + w - 1
            if not (right < c_canvas0 or left > c_canvas1 or bottom < r_canvas0 or top > r_canvas1):
                continue
            box = (top, left, bottom, right)
            controls = bbox_ring_positions(box, grid_shape)
            random.shuffle(controls)
            for control in controls:
                if r_canvas0 <= control[0] <= r_canvas1 and c_canvas0 <= control[1] <= c_canvas1:
                    continue
                if not control_matches_direction(control, center, spec["direction"]):
                    continue
                candidates.append((top, left, control))
    random.shuffle(candidates)
    return candidates


def place_specs_into_input(
    specs: list[dict],
    grid_shape: tuple[int, int],
    canvas_box: tuple[int, int, int, int],
    center: tuple[int, int],
) -> list[dict] | None:
    candidates = {idx: placement_candidates(grid_shape, canvas_box, center, spec) for idx, spec in enumerate(specs)}
    if any(not options for options in candidates.values()):
        return None

    order = sorted(candidates, key=lambda idx: len(candidates[idx]))
    occupied: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * len(specs)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = specs[idx]
        for top, left, control in candidates[idx]:
            group = group_cells_for_candidate(spec["patch"], top, left, control)
            if touches_existing(group, occupied):
                continue
            placed[idx] = dict(spec)
            placed[idx]["input_top_left"] = (int(top), int(left))
            placed[idx]["control"] = (int(control[0]), int(control[1]))
            occupied.update(group)
            if backtrack(pos + 1):
                return True
            occupied.difference_update(group)
            placed[idx] = None
        return False

    if not backtrack(0):
        return None
    return [spec for spec in placed if spec is not None]


def draw_input(
    specs: list[dict],
    grid_shape: tuple[int, int],
    canvas_top_left: tuple[int, int],
) -> np.ndarray:
    grid = np.full(grid_shape, BLACK, dtype=int)
    r0, c0 = canvas_top_left
    grid[r0 : r0 + 9, c0 : c0 + 9] = GRAY
    for spec in specs:
        top, left = spec["input_top_left"]
        paste(grid, spec["patch"], top, left, transparent=BLACK, clip=False)
        pr, pc = spec["control"]
        grid[pr, pc] = MAGENTA
    return grid


def validate_generated_instance(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        canvas, sprites = parse_instance(inp)
    except Exception:
        return False
    if overlap_count(sprites) == 0:
        return False
    if np.array_equal(compose_output(sprites, descending=True), compose_output(sprites, descending=False)):
        return False
    actual = main(inp)
    if not np.array_equal(actual, expected_out):
        return False
    return True


def gen_attempt() -> np.ndarray | None:
    specs = build_output_specs()
    if specs is None:
        return None

    grid_h = random.randint(19, 24)
    grid_w = random.randint(18, 24)
    canvas_r0 = random.randint(max(6, grid_h // 2 - 4), min(grid_h - 9, grid_h // 2 - 1))
    canvas_c0 = (grid_w - 9) // 2
    canvas_box = (canvas_r0, canvas_c0, canvas_r0 + 8, canvas_c0 + 8)
    center = (canvas_r0 + 4, canvas_c0 + 4)

    placed = place_specs_into_input(specs, (grid_h, grid_w), canvas_box, center)
    if placed is None:
        return None

    out = compose_output(placed, descending=True)
    inp = draw_input(placed, (grid_h, grid_w), (canvas_r0, canvas_c0))

    if not validate_generated_instance(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid task instance")
