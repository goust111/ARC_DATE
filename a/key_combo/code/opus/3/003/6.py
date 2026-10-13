import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors8 = _ARC_UTILS.neighbors8


BLACK = Color.BLACK
RED = Color.RED
PINK = Color.PINK
TEAL = Color.TEAL
INPUT_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.MAROON)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250

_PSEUDO_SQUARE_TEMPLATES = (
    np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
        ],
        dtype=bool,
    ),
)

_NON_SQUARE_TEMPLATES = (
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
)


def _random_symmetry(mask: np.ndarray, rng: random.Random, *extra_masks: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = [mask.copy(), *(arr.copy() for arr in extra_masks)]
    if rng.random() < 0.5:
        arrays = [np.fliplr(arr) for arr in arrays]
    k = rng.randrange(4)
    arrays = [np.rot90(arr, k=k).copy() for arr in arrays]
    return tuple(arrays)


def _mask_has_complete_square_border(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w:
        return False
    return bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all())


def _boundary_reachable_background(grid: np.ndarray, bg: int) -> np.ndarray:
    reachable = np.zeros(grid.shape, dtype=bool)
    for comp in components_mask(grid == int(bg), connectivity=4):
        if any(
            int(r) == 0
            or int(c) == 0
            or int(r) == grid.shape[0] - 1
            or int(c) == grid.shape[1] - 1
            for r, c in comp
        ):
            reachable[comp[:, 0], comp[:, 1]] = True
    return reachable


def _component_info(grid: np.ndarray, comp: dict, boundary_bg: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    local = np.zeros((h, w), dtype=bool)
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    for r, c in coords:
        local[r - r0, c - c0] = True

    complete_square = _mask_has_complete_square_border(local)
    hole_coords: set[tuple[int, int]] = set()
    if complete_square and h >= 3:
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == int(bg) and not bool(boundary_bg[r, c]):
                    hole_coords.add((r, c))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "bbox_square": h == w,
        "color": color,
        "coords": coords,
        "local_mask": local,
        "complete_square": complete_square,
        "hole_coords": hole_coords,
    }


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    boundary_bg = _boundary_reachable_background(g, bg)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    infos = [_component_info(g, comp, boundary_bg, bg) for comp in components]
    return g, bg, infos


def _corner_mark_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _outline_positions(shape: set[tuple[int, int]], grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    outline: set[tuple[int, int]] = set()
    for r, c in shape:
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid_shape, nr, nc) and (nr, nc) not in shape:
                outline.add((int(nr), int(nc)))
    return outline


def _solve_with_mode(grid: np.ndarray, *, square_mode: str = "complete", hollow_mode: str = "true") -> np.ndarray:
    g, bg, infos = _analyze_grid(grid)
    out = g.copy()

    annotated = []
    for info in infos:
        if square_mode == "bbox":
            is_square = bool(info["bbox_square"])
        else:
            is_square = bool(info["complete_square"])

        if hollow_mode == "ignore":
            hole_coords: set[tuple[int, int]] = set()
        else:
            hole_coords = set(info["hole_coords"])

        if is_square and hole_coords:
            for r, c in info["coords"]:
                out[r, c] = TEAL
            for r, c in hole_coords:
                out[r, c] = PINK
            branch = "hollow"
            shape = set(info["coords"]) | hole_coords
        elif is_square:
            branch = "solid"
            shape = set(info["coords"])
        else:
            branch = "irregular"
            shape = set(info["coords"])

        annotated.append(
            {
                "bbox": info["bbox"],
                "branch": branch,
                "shape": shape,
            }
        )

    red_positions: set[tuple[int, int]] = set()
    for info in annotated:
        if info["branch"] == "solid":
            for r, c in _corner_mark_positions(info["bbox"]):
                if in_bounds(out.shape, r, c) and int(out[r, c]) == int(bg):
                    red_positions.add((int(r), int(c)))
            continue

        for r, c in _outline_positions(info["shape"], out.shape):
            if int(out[r, c]) == int(bg):
                red_positions.add((int(r), int(c)))

    for r, c in red_positions:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split foreground objects into solid complete squares, hollow complete squares, and non-squares, then mark each branch differently.

    Input structure:
    - Black background with several 4-connected monochrome foreground objects in non-red colors.
    - Some objects are complete squares: their bounding boxes are square and every bbox-border cell belongs to the object.
    - Complete squares may be solid or may surround one or more enclosed black holes; other objects are irregular distractors.

    Rule:
    - Find each 4-connected foreground component and test whether it is a complete square by the component's own bbox and border pixels.
    - For complete squares only, test whether any black background region inside the bbox interior is enclosed and cannot reach the grid boundary.
    - Solid complete squares keep their color and receive red orthogonal corner-outside marks; hollow complete squares are recolored teal, their enclosed holes are filled pink, and then a red 8-neighbor outline is drawn around the filled shape; non-square objects keep their color and only receive that red outline.
    - Apply all recolor/fill changes first, then paint every outline and corner mark onto still-black cells.

    Key insight:
    - The first decision is not color or size but whether the component really forms a complete square border, and the second decision is whether that square traps interior background; those two tests jointly select three visibly different output behaviors.
    """
    return _solve_with_mode(grid, square_mode="complete", hollow_mode="true")


def _connected_subset(height: int, width: int, size: int, rng: random.Random) -> np.ndarray:
    size = max(1, min(int(size), height * width))
    mask = np.zeros((height, width), dtype=bool)
    seed = (rng.randrange(height), rng.randrange(width))
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < size:
        candidates = []
        for r, c in frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            break
        choice = rng.choice(candidates)
        mask[choice] = True
        frontier.add(choice)
    return mask


def _make_solid_square(rng: random.Random) -> dict:
    size = rng.randint(3, 6)
    mask = np.ones((size, size), dtype=bool)
    return {
        "kind": "solid",
        "pseudo_square": False,
        "mask": mask,
        "hole_mask": np.zeros_like(mask),
    }


def _make_hollow_square(rng: random.Random) -> dict:
    size = rng.randint(4, 6)
    inner = size - 2
    hole_inner = np.zeros((inner, inner), dtype=bool)
    if rng.random() < 0.45 and inner * inner >= 2:
        choices = [(r, c) for r in range(inner) for c in range(inner)]
        rng.shuffle(choices)
        target = rng.randint(2, min(3, len(choices)))
        picked: list[tuple[int, int]] = []
        for r, c in choices:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in picked):
                picked.append((r, c))
            if len(picked) == target:
                break
        if not picked:
            picked.append(rng.choice(choices))
        for r, c in picked:
            hole_inner[r, c] = True
    else:
        max_hole = inner * inner
        if rng.random() < 0.25:
            target = max_hole
        else:
            target = rng.randint(1, max_hole)
        hole_inner = _connected_subset(inner, inner, target, rng)

    mask = np.ones((size, size), dtype=bool)
    full_hole = np.zeros((size, size), dtype=bool)
    full_hole[1:-1, 1:-1] = hole_inner
    mask[full_hole] = False
    mask, full_hole = _random_symmetry(mask, rng, full_hole)
    return {
        "kind": "hollow",
        "pseudo_square": False,
        "mask": mask,
        "hole_mask": full_hole,
    }


def _make_irregular(rng: random.Random, *, pseudo_square: bool) -> dict:
    if pseudo_square:
        mask = rng.choice(_PSEUDO_SQUARE_TEMPLATES).copy()
    else:
        mask = rng.choice(_NON_SQUARE_TEMPLATES).copy()
    (mask,) = _random_symmetry(mask, rng)
    return {
        "kind": "irregular",
        "pseudo_square": pseudo_square,
        "mask": mask,
        "hole_mask": np.zeros_like(mask),
    }


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_rect(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - 1),
        max(0, c0 - 1),
        min(shape[0] - 1, r0 + h),
        min(shape[1] - 1, c0 + w),
    )


def _place_objects(specs: list[dict], grid_shape: tuple[int, int], rng: random.Random) -> list[dict] | None:
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []
    order = sorted(specs, key=lambda spec: spec["mask"].shape[0] * spec["mask"].shape[1], reverse=True)
    for spec in order:
        mask = spec["mask"]
        h, w = mask.shape
        done = None
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(0, grid_shape[0] - h)
            c0 = rng.randint(0, grid_shape[1] - w)
            rect = _expanded_rect(r0, c0, h, w, grid_shape)
            if any(_rects_overlap(rect, other) for other in reserved):
                continue
            placed_spec = {
                **spec,
                "r0": r0,
                "c0": c0,
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "reserved": rect,
            }
            done = placed_spec
            break
        if done is None:
            return None
        reserved.append(done["reserved"])
        placed.append(done)
    return placed


def _render_case(placed: list[dict], grid_shape: tuple[int, int]) -> dict:
    grid = np.full(grid_shape, BLACK, dtype=int)
    rendered: list[dict] = []
    for spec in placed:
        mask = spec["mask"]
        hole_mask = spec["hole_mask"]
        color = int(spec["color"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        coords = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
        hole_coords = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(hole_mask)}
        for r, c in coords:
            grid[r, c] = color
        rendered.append(
            {
                **spec,
                "coords": coords,
                "hole_coords": hole_coords,
            }
        )
    return {"grid": grid, "objects": rendered}


def _expected_output(case: dict) -> np.ndarray:
    grid = case["grid"]
    out = grid.copy()
    red_positions: set[tuple[int, int]] = set()
    for obj in case["objects"]:
        if obj["kind"] != "hollow":
            continue
        for r, c in obj["coords"]:
            out[r, c] = TEAL
        for r, c in obj["hole_coords"]:
            out[r, c] = PINK

    for obj in case["objects"]:
        if obj["kind"] == "solid":
            for r, c in _corner_mark_positions(obj["bbox"]):
                if in_bounds(out.shape, r, c) and int(out[r, c]) == BLACK:
                    red_positions.add((int(r), int(c)))
            continue
        if obj["kind"] == "hollow":
            shape = set(obj["coords"]) | set(obj["hole_coords"])
        else:
            shape = set(obj["coords"])
        for r, c in _outline_positions(shape, out.shape):
            if int(out[r, c]) == BLACK:
                red_positions.add((int(r), int(c)))

    for r, c in red_positions:
        out[r, c] = RED
    return out


def _shortcut_bbox_square_only(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, square_mode="bbox", hollow_mode="true")


def _shortcut_ignore_holes(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, square_mode="complete", hollow_mode="ignore")


def _visible_outline_pixels(grid: np.ndarray, shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {coord for coord in _outline_positions(shape, grid.shape) if int(grid[coord]) == BLACK}


def _validate_case(case: dict) -> bool:
    grid = case["grid"]
    objects = case["objects"]
    if not (4 <= len(objects) <= 6):
        return False
    if not {"solid", "hollow", "irregular"} <= {obj["kind"] for obj in objects}:
        return False
    if not any(obj["pseudo_square"] for obj in objects if obj["kind"] == "irregular"):
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    actual_components = find_components(grid, bg=BLACK, connectivity=4, monochromatic=True, return_format="dict")
    if len(actual_components) != len(objects):
        return False

    expected_signature = sorted((obj["bbox"], len(obj["coords"]), int(obj["color"])) for obj in objects)
    actual_signature = sorted((tuple(map(int, comp["bbox"])), int(comp["area"]), int(comp["color"])) for comp in actual_components)
    if expected_signature != actual_signature:
        return False

    owner = {coord: idx for idx, obj in enumerate(objects) for coord in obj["coords"]}
    for idx, obj in enumerate(objects):
        for r, c in obj["coords"]:
            for nr, nc in neighbors8(r, c):
                other = owner.get((int(nr), int(nc)))
                if other is not None and other != idx:
                    return False

    for obj in objects:
        if obj["kind"] == "solid":
            visible_marks = [coord for coord in _corner_mark_positions(obj["bbox"]) if in_bounds(grid.shape, *coord) and int(grid[coord]) == BLACK]
            if not visible_marks:
                return False
        elif obj["kind"] == "hollow":
            if not obj["hole_coords"]:
                return False
            outline = _visible_outline_pixels(grid, set(obj["coords"]) | set(obj["hole_coords"]))
            if not outline:
                return False
        else:
            outline = _visible_outline_pixels(grid, set(obj["coords"]))
            if not outline:
                return False
            h, w = obj["mask"].shape
            if obj["pseudo_square"]:
                if h != w or _mask_has_complete_square_border(obj["mask"]):
                    return False

    expected = _expected_output(case)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(_shortcut_bbox_square_only(grid), expected):
        return False
    if np.array_equal(_shortcut_ignore_holes(grid), expected):
        return False

    hollow_objs = [obj for obj in objects if obj["kind"] == "hollow"]
    solid_objs = [obj for obj in objects if obj["kind"] == "solid"]
    irregular_objs = [obj for obj in objects if obj["kind"] == "irregular"]
    if not hollow_objs or not solid_objs or not irregular_objs:
        return False
    if not any(int(expected[r, c]) == PINK for obj in hollow_objs for r, c in obj["hole_coords"]):
        return False
    if not any(int(expected[r, c]) == TEAL for obj in hollow_objs for r, c in obj["coords"]):
        return False
    if not any(int(expected[r, c]) == RED for obj in solid_objs for r, c in _corner_mark_positions(obj["bbox"]) if in_bounds(expected.shape, r, c)):
        return False
    if not any(_visible_outline_pixels(grid, set(obj["coords"])) for obj in irregular_objs):
        return False
    return True


def _build_case(rng: random.Random) -> dict | None:
    grid_shape = (rng.randint(*GRID_H_RANGE), rng.randint(*GRID_W_RANGE))
    total = rng.randint(4, 6)

    specs = [_make_solid_square(rng), _make_hollow_square(rng), _make_irregular(rng, pseudo_square=True)]
    while len(specs) < total:
        choice = rng.choices(
            ["solid", "hollow", "pseudo", "irregular"],
            weights=[2, 2, 2, 3],
            k=1,
        )[0]
        if choice == "solid":
            specs.append(_make_solid_square(rng))
        elif choice == "hollow":
            specs.append(_make_hollow_square(rng))
        elif choice == "pseudo":
            specs.append(_make_irregular(rng, pseudo_square=True))
        else:
            specs.append(_make_irregular(rng, pseudo_square=False))

    active_colors = rng.sample(INPUT_COLORS, rng.randint(2, min(5, len(INPUT_COLORS))))
    for spec in specs:
        spec["color"] = int(rng.choice(active_colors))

    placed = _place_objects(specs, grid_shape, rng)
    if placed is None:
        return None
    return _render_case(placed, grid_shape)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        case = _build_case(rng)
        if case is None:
            continue
        if _validate_case(case):
            return case["grid"]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
