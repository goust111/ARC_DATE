from __future__ import annotations

import itertools
import random
from collections import Counter
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import (
        Color,
        crop_bbox,
        find_components,
        get_bg,
        paste,
        random_free_location,
        random_sprite,
    )
except ImportError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    paste = _ARC_UTILS.paste
    random_free_location = _ARC_UTILS.random_free_location
    random_sprite = _ARC_UTILS.random_sprite


BG = Color.BLACK
TRANSFORMS = range(8)
NEAR_BEST_BAND = 2
NEAR_BEST_CAP = 14
GEN_RETRIES = 400
PROTO_SIDE_RANGE = (5, 8)
GRID_SIDE_RANGE = (18, 30)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MIN_PROTO_AREA = 14
MIN_PAIR_OVERLAP = 5
MIN_VISIBLE_AREA = 10
MAX_HOLE_SIDE = 4
MAX_HOLE_AREA = 6
MAX_EXTRA_SIDE = 2


def _transform_shape(shape: tuple[int, int], transform: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return (h, w) if transform in {0, 2, 4, 6} else (w, h)


def _transform_coord(coord: tuple[int, int], shape: tuple[int, int], transform: int) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    if transform == 0:
        return r, c
    if transform == 1:
        return w - 1 - c, r
    if transform == 2:
        return h - 1 - r, w - 1 - c
    if transform == 3:
        return c, h - 1 - r
    if transform == 4:
        return r, w - 1 - c
    if transform == 5:
        return c, r
    if transform == 6:
        return h - 1 - r, c
    if transform == 7:
        return w - 1 - c, h - 1 - r
    raise ValueError(f"unknown transform {transform}")


def _inverse_transform_coord(
    coord: tuple[int, int],
    shape: tuple[int, int],
    transform: int,
) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    if transform == 0:
        return r, c
    if transform == 1:
        return c, w - 1 - r
    if transform == 2:
        return h - 1 - r, w - 1 - c
    if transform == 3:
        return h - 1 - c, r
    if transform == 4:
        return r, w - 1 - c
    if transform == 5:
        return c, r
    if transform == 6:
        return h - 1 - r, c
    if transform == 7:
        return h - 1 - c, w - 1 - r
    raise ValueError(f"unknown transform {transform}")


def _transform_coords(
    coords: frozenset[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    transform: int,
) -> frozenset[tuple[int, int]]:
    return frozenset(_transform_coord(coord, shape, transform) for coord in coords)


def _bbox_from_coords(coords: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(
    coords: frozenset[tuple[int, int]] | set[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    box = _bbox_from_coords(coords)
    assert box is not None
    r0, c0, r1, c1 = box
    normalized = frozenset((r - r0, c - c0) for r, c in coords)
    return normalized, (r1 - r0 + 1, c1 - c0 + 1), (r0, c0)


def _coords_patch(
    coords: frozenset[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    color: int,
) -> np.ndarray:
    patch = np.zeros(shape, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _is_connected(coords: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    coords = set(coords)
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(coords)


def _is_filled_rectangle(coords: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    box = _bbox_from_coords(coords)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if len(coords) != expected_area:
        return False
    return all((r, c) in coords for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _candidate_alignment_options(
    base_coords: frozenset[tuple[int, int]],
    other_transforms: list[frozenset[tuple[int, int]]],
    *,
    near_best: bool,
) -> list[dict]:
    options: list[dict] = []
    for transform, coords_t in enumerate(other_transforms):
        if not coords_t:
            continue
        counter: Counter[tuple[int, int]] = Counter()
        for ar, ac in base_coords:
            for br, bc in coords_t:
                counter[(ar - br, ac - bc)] += 1
        for shift, overlap in counter.items():
            options.append(
                {
                    "transform": transform,
                    "shift": (int(shift[0]), int(shift[1])),
                    "overlap": int(overlap),
                }
            )
    if not options:
        return []

    best_overlap = max(option["overlap"] for option in options)
    if not near_best:
        return [option for option in options if option["overlap"] == best_overlap]

    filtered = [option for option in options if option["overlap"] >= best_overlap - NEAR_BEST_BAND]
    filtered.sort(
        key=lambda option: (
            -option["overlap"],
            option["transform"],
            option["shift"][0],
            option["shift"][1],
        )
    )
    return filtered[:NEAR_BEST_CAP]


def _best_pair_alignment(base: dict, other: dict) -> dict | None:
    candidates = _candidate_alignment_options(base["coords"], other["transforms"], near_best=False)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _try_solution(objects: list[dict], members: list[dict]) -> dict | None:
    placed: list[frozenset[tuple[int, int]]] = []
    for member in members:
        coords_t = objects[member["index"]]["transforms"][member["transform"]]
        dr, dc = member["shift"]
        placed.append(frozenset((r + dr, c + dc) for r, c in coords_t))

    if any(len(placed[i] & placed[j]) == 0 for i in range(3) for j in range(i + 1, 3)):
        return None

    proto = frozenset().union(*placed)
    if not _is_connected(proto):
        return None

    holes = [frozenset(proto - coords) for coords in placed]
    if any(not hole or not _is_filled_rectangle(hole) for hole in holes):
        return None
    if len({tuple(sorted(hole)) for hole in holes}) != 3:
        return None

    return {
        "members": tuple(
            {
                "index": int(member["index"]),
                "transform": int(member["transform"]),
                "shift": (int(member["shift"][0]), int(member["shift"][1])),
            }
            for member in members
        ),
        "proto": proto,
    }


def _enumerate_solutions(objects: list[dict], *, exhaustive: bool) -> list[dict]:
    solutions: list[dict] = []
    for triple in itertools.combinations(range(len(objects)), 3):
        for base_index in triple:
            others = [idx for idx in triple if idx != base_index]
            member_lists: list[list[dict]] = [[{"index": base_index, "transform": 0, "shift": (0, 0)}]]
            feasible = True
            for other_index in others:
                if exhaustive:
                    options = _candidate_alignment_options(
                        objects[base_index]["coords"],
                        objects[other_index]["transforms"],
                        near_best=True,
                    )
                    options = [
                        {
                            "index": other_index,
                            "transform": option["transform"],
                            "shift": option["shift"],
                        }
                        for option in options
                    ]
                else:
                    best = _best_pair_alignment(objects[base_index], objects[other_index])
                    options = (
                        [
                            {
                                "index": other_index,
                                "transform": best["transform"],
                                "shift": best["shift"],
                            }
                        ]
                        if best is not None
                        else []
                    )
                if not options:
                    feasible = False
                    break
                member_lists.append(options)
            if not feasible:
                continue

            for combo in itertools.product(*member_lists):
                solution = _try_solution(objects, list(combo))
                if solution is not None:
                    solutions.append(solution)
    return solutions


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    assert bg == BG
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) == 4, "expected exactly four monochromatic components"

    objects = []
    for component in components:
        color = int(component["color"])
        assert color != bg
        bbox = tuple(map(int, component["bbox"]))
        patch = crop_bbox(grid, bbox)
        local_coords = frozenset(map(tuple, np.argwhere(patch == color)))
        assert local_coords, "empty component"
        shape = tuple(map(int, patch.shape))
        objects.append(
            {
                "color": color,
                "bbox": bbox,
                "shape": shape,
                "coords": local_coords,
                "abs_coords": frozenset(map(tuple, np.asarray(component["coords"], dtype=int))),
                "transforms": [_transform_coords(local_coords, shape, transform) for transform in TRANSFORMS],
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _solution_output(grid: np.ndarray, objects: list[dict], solution: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    members = {member["index"]: member for member in solution["members"]}

    for index, obj in enumerate(objects):
        if index in members:
            for r, c in obj["abs_coords"]:
                out[int(r), int(c)] = BG
            member = members[index]
            for cr, cc in solution["proto"]:
                local_t = (int(cr) - member["shift"][0], int(cc) - member["shift"][1])
                local = _inverse_transform_coord(local_t, obj["shape"], member["transform"])
                abs_r = obj["bbox"][0] + local[0]
                abs_c = obj["bbox"][1] + local[1]
                assert 0 <= abs_r < out.shape[0] and 0 <= abs_c < out.shape[1]
                out[abs_r, abs_c] = obj["color"]
        else:
            for r, c in obj["abs_coords"]:
                out[int(r), int(c)] = BG
    return out


def _solve_outputs(grid: np.ndarray, *, exhaustive: bool) -> tuple[list[dict], list[np.ndarray]]:
    objects = _extract_objects(grid)
    solutions = _enumerate_solutions(objects, exhaustive=exhaustive)
    outputs_by_key: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for solution in solutions:
        out = _solution_output(grid, objects, solution)
        outputs_by_key[(tuple(map(int, out.shape)), out.tobytes())] = out
    return solutions, list(outputs_by_key.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden prototype from three rotated-or-flipped damaged copies, complete them, and erase the impostor.

    Input structure:
    - A black background contains exactly four separated monochromatic objects, all in different colors.
    - Three objects are transformed copies of one prototype with one rectangular chunk removed from each copy.
    - The fourth object is an impostor that cannot be made consistent with the damaged-copy family under any dihedral alignment.

    Rule:
    - Compare objects up to rotation and reflection and find the three whose visible cells can align into one common prototype.
    - In a valid family, the union of the aligned visible cells is the prototype, and each member differs from that union by one non-empty filled rectangle.
    - Repaint that prototype back at each damaged copy's original orientation and position using the copy's own color.
    - Erase the impostor to the black background.

    Key insight:
    - The complete source shape never appears directly; it must be inferred as the common union of three dihedrally aligned fragments, which simultaneously identifies the impostor and the exact missing rectangle in each genuine copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    solutions, outputs = _solve_outputs(g, exhaustive=False)
    if len(outputs) == 1:
        return outputs[0]

    solutions, outputs = _solve_outputs(g, exhaustive=True)
    assert outputs, "no valid damaged-copy trio found"
    assert len(outputs) == 1, "ambiguous reconstruction"
    return outputs[0]


def _generate_prototype() -> tuple[frozenset[tuple[int, int]], tuple[int, int]] | None:
    h = random.randint(*PROTO_SIDE_RANGE)
    w = random.randint(*PROTO_SIDE_RANGE)
    density = random.uniform(0.48, 0.68)
    sprite = random_sprite(
        [h],
        [w],
        density=density,
        symmetry="not_symmetric",
        color_palette=[Color.BLUE],
        connectivity=4,
        background=BG,
    )
    proto = frozenset(map(tuple, np.argwhere(sprite != BG)))
    if len(proto) < MIN_PROTO_AREA:
        return None
    for transform in range(1, 8):
        if _transform_shape(sprite.shape, transform) == sprite.shape:
            if _transform_coords(proto, sprite.shape, transform) == proto:
                return None
    return proto, tuple(map(int, sprite.shape))


def _enumerate_holes(
    proto: frozenset[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = map(int, shape)
    holes: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0, min(h, r0 + MAX_HOLE_SIDE)):
                for c1 in range(c0, min(w, c0 + MAX_HOLE_SIDE)):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < 2 or area > MAX_HOLE_AREA:
                        continue
                    rect = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
                    if not rect <= proto:
                        continue
                    fragment = frozenset(proto - rect)
                    if len(fragment) < MIN_VISIBLE_AREA or not _is_connected(fragment):
                        continue
                    holes.append(
                        {
                            "coords": rect,
                            "bbox": (r0, c0, r1, c1),
                            "touches_border": r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
                            "shrinks_bbox": _bbox_from_coords(fragment) != (0, 0, h - 1, w - 1),
                        }
                    )
    return holes


def _choose_hole_triple(
    proto: frozenset[tuple[int, int]],
    shape: tuple[int, int],
    hole_candidates: list[dict],
) -> tuple[list[dict], list[frozenset[tuple[int, int]]]] | None:
    require_shrink = random.random() < 0.6
    combos = list(itertools.combinations(hole_candidates, 3))
    random.shuffle(combos)
    for holes in combos:
        if holes[0]["coords"] & holes[1]["coords"] & holes[2]["coords"]:
            continue
        if not (
            (holes[0]["coords"] & holes[1]["coords"])
            or (holes[0]["coords"] & holes[2]["coords"])
            or (holes[1]["coords"] & holes[2]["coords"])
        ):
            continue
        fragments = [frozenset(proto - hole["coords"]) for hole in holes]
        if any(len(fragments[i] & fragments[j]) < MIN_PAIR_OVERLAP for i in range(3) for j in range(i + 1, 3)):
            continue
        if require_shrink and not any(hole["shrinks_bbox"] for hole in holes):
            continue
        return list(holes), fragments
    return None


def _extra_rectangles(
    base: frozenset[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    box = _bbox_from_coords(base)
    assert box is not None
    r0, c0, r1, c1 = box
    rects: list[frozenset[tuple[int, int]]] = []
    for top in range(r0 - 1, r1 + 2):
        for left in range(c0 - 1, c1 + 2):
            for height in range(1, MAX_EXTRA_SIDE + 1):
                for width in range(1, MAX_EXTRA_SIDE + 1):
                    rect = frozenset(
                        (r, c)
                        for r in range(top, top + height)
                        for c in range(left, left + width)
                    )
                    if rect & base:
                        continue
                    if not any(
                        nbr in base
                        for r, c in rect
                        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    ):
                        continue
                    rects.append(rect)
    return rects


def _generate_fake(
    proto: frozenset[tuple[int, int]],
    holes: list[dict],
) -> frozenset[tuple[int, int]] | None:
    extra_options = _extra_rectangles(proto)
    if not extra_options:
        return None
    for _ in range(120):
        remove = random.choice(holes)["coords"]
        extra = random.choice(extra_options)
        fake = frozenset((proto - remove) | extra)
        if len(fake) < MIN_VISIBLE_AREA or not _is_connected(fake):
            continue
        box = _bbox_from_coords(fake)
        assert box is not None
        if box[2] - box[0] + 1 > 10 or box[3] - box[1] + 1 > 10:
            continue
        return fake
    return None


def _sample_true_transforms() -> list[int]:
    for _ in range(80):
        transforms = [random.randrange(8) for _ in range(3)]
        if len(set(transforms)) < 2:
            continue
        if all(transform == 0 for transform in transforms):
            continue
        if random.random() < 0.5 and not any(transform in {1, 3, 5, 7} for transform in transforms):
            continue
        if random.random() < 0.5 and not any(transform >= 4 for transform in transforms):
            continue
        return transforms
    return [0, 1, 4]


def _place_items(
    items: list[dict],
    colors: list[int],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    reservation = np.zeros(grid_shape, dtype=int)
    inp = np.zeros(grid_shape, dtype=int)
    out = np.zeros(grid_shape, dtype=int)

    shuffled = list(zip(items, colors))
    random.shuffle(shuffled)
    for item, color in shuffled:
        try:
            location = random_free_location(
                reservation,
                item["reserve"],
                transparent=BG,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reservation, item["reserve"], location[0], location[1], transparent=BG)
        paste(inp, item["input_patch"] * color, location[0], location[1], transparent=BG)
        if item["is_true"]:
            paste(out, item["output_patch"] * color, location[0], location[1], transparent=BG)
            item["true_color"] = color
    return inp, out


def _build_true_item(
    proto: frozenset[tuple[int, int]],
    proto_shape: tuple[int, int],
    fragment: frozenset[tuple[int, int]],
    transform: int,
) -> dict:
    oriented_shape = _transform_shape(proto_shape, transform)
    proto_oriented = _transform_coords(proto, proto_shape, transform)
    fragment_oriented = _transform_coords(fragment, proto_shape, transform)
    return {
        "is_true": True,
        "reserve": np.ones(oriented_shape, dtype=int),
        "input_patch": _coords_patch(fragment_oriented, oriented_shape, 1),
        "output_patch": _coords_patch(proto_oriented, oriented_shape, 1),
    }


def _build_fake_item(fake: frozenset[tuple[int, int]], transform: int) -> dict:
    normalized, shape, _offset = _normalize_coords(fake)
    oriented_shape = _transform_shape(shape, transform)
    oriented = _transform_coords(normalized, shape, transform)
    return {
        "is_true": False,
        "reserve": np.ones(oriented_shape, dtype=int),
        "input_patch": _coords_patch(oriented, oriented_shape, 1),
        "output_patch": np.zeros(oriented_shape, dtype=int),
    }


def _validate_case(inp: np.ndarray, expected: np.ndarray, true_colors: set[int]) -> bool:
    try:
        objects = _extract_objects(inp)
    except AssertionError:
        return False

    color_to_index = {obj["color"]: idx for idx, obj in enumerate(objects)}
    if true_colors - set(color_to_index):
        return False
    true_members = tuple(sorted(color_to_index[color] for color in true_colors))

    fast_solutions = _enumerate_solutions(objects, exhaustive=False)
    fast_outputs = {
        _solution_output(inp, objects, solution).tobytes(): _solution_output(inp, objects, solution)
        for solution in fast_solutions
    }
    if not fast_outputs:
        return False
    if len(fast_outputs) != 1:
        return False
    if not np.array_equal(next(iter(fast_outputs.values())), expected):
        return False

    exhaustive_solutions = _enumerate_solutions(objects, exhaustive=True)
    if not exhaustive_solutions:
        return False
    member_sets = {tuple(sorted(member["index"] for member in solution["members"])) for solution in exhaustive_solutions}
    if member_sets != {true_members}:
        return False
    exhaustive_outputs = {
        _solution_output(inp, objects, solution).tobytes(): _solution_output(inp, objects, solution)
        for solution in exhaustive_solutions
    }
    if len(exhaustive_outputs) != 1:
        return False
    return np.array_equal(next(iter(exhaustive_outputs.values())), expected)


def _attempt_generate() -> np.ndarray | None:
    generated = _generate_prototype()
    if generated is None:
        return None
    proto, proto_shape = generated
    holes = _enumerate_holes(proto, proto_shape)
    if len(holes) < 3:
        return None
    hole_choice = _choose_hole_triple(proto, proto_shape, holes)
    if hole_choice is None:
        return None
    hole_triple, fragments = hole_choice
    fake = _generate_fake(proto, holes)
    if fake is None:
        return None

    true_transforms = _sample_true_transforms()
    fake_transform = random.randrange(8)
    items = [
        _build_true_item(proto, proto_shape, fragment, transform)
        for fragment, transform in zip(fragments, true_transforms)
    ]
    items.append(_build_fake_item(fake, fake_transform))

    colors = random.sample(Color.NOT_BLACK, 4)
    reserve_cells = sum(int(np.prod(item["reserve"].shape)) for item in items)
    min_side = max(max(item["reserve"].shape) + 4 for item in items)
    target_side = max(min_side, int(np.ceil(np.sqrt(reserve_cells * 2.6))))
    low = max(GRID_SIDE_RANGE[0], target_side)
    if low > GRID_SIDE_RANGE[1]:
        return None
    grid_shape = (
        random.randint(low, GRID_SIDE_RANGE[1]),
        random.randint(low, GRID_SIDE_RANGE[1]),
    )
    placed = _place_items(items, colors, grid_shape)
    if placed is None:
        return None
    inp, expected = placed
    true_colors = {item["true_color"] for item in items if item["is_true"]}
    if not _validate_case(inp, expected, true_colors):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
