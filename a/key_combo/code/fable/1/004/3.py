from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


VISIBLE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            stack = [(int(r), int(c))]
            visited[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(mask.shape, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(visited[nr, nc]):
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def enclosed_regions(open_mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    enclosed_mask = np.zeros_like(open_mask, dtype=bool)
    enclosed_components: list[np.ndarray] = []
    for coords in components_mask(open_mask):
        touches_edge = bool(
            (coords[:, 0] == 0).any()
            or (coords[:, 0] == open_mask.shape[0] - 1).any()
            or (coords[:, 1] == 0).any()
            or (coords[:, 1] == open_mask.shape[1] - 1).any()
        )
        if touches_edge:
            continue
        enclosed_mask[coords[:, 0], coords[:, 1]] = True
        enclosed_components.append(coords)
    return enclosed_mask, enclosed_components


def isolated_dot_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    dots = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == int(bg):
                continue
            if all(not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == int(bg) for nr, nc in neighbors4(r, c)):
                dots[r, c] = True
    return dots


def analyze_component(subgrid: np.ndarray, component_mask: np.ndarray) -> dict:
    colors, counts = np.unique(subgrid[component_mask], return_counts=True)
    shell_color = int(colors[int(np.argmax(counts))])
    shell_mask = component_mask & (subgrid == int(shell_color))

    plug_mask = component_mask & ~shell_mask
    for coords in components_mask(plug_mask):
        if coords.shape[0] != 1:
            raise AssertionError("plug pixels must be isolated singletons")

    visible_mask, visible_regions = enclosed_regions(~component_mask)
    all_holes_mask, all_hole_regions = enclosed_regions(~shell_mask)
    return {
        "shell_color": int(shell_color),
        "shell_mask": shell_mask,
        "plug_mask": plug_mask,
        "visible_mask": visible_mask,
        "visible_count": int(len(visible_regions)),
        "hole_mask": all_holes_mask,
        "total_count": int(len(all_hole_regions)),
        "plug_count": int(len(all_hole_regions) - len(visible_regions)),
    }


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    dot_mask = isolated_dot_mask(g, bg)
    if not dot_mask.any():
        raise AssertionError("expected at least one isolated dot")
    dot_colors = np.unique(g[dot_mask])
    if len(dot_colors) != 1:
        raise AssertionError("all isolated dots must share one color")
    dot_color = int(dot_colors[0])

    object_mask = (g != int(bg)) & ~dot_mask
    objects = []
    for coords in components_mask(object_mask):
        box = bbox_coords(coords)
        subgrid = crop_bbox(g, box)
        local_mask = np.zeros(subgrid.shape, dtype=bool)
        r0, c0, _r1, _c1 = box
        local_mask[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
        info = analyze_component(subgrid, local_mask)
        info.update(
            {
                "bbox": tuple(map(int, box)),
                "coords": coords,
                "subgrid": subgrid,
            }
        )
        objects.append(info)

    dot_count = int(dot_mask.sum())
    target_indices = [i for i, obj in enumerate(objects) if int(obj["total_count"]) == int(dot_count)]
    visible_match_indices = [i for i, obj in enumerate(objects) if int(obj["visible_count"]) == int(dot_count)]
    return {
        "bg": int(bg),
        "dot_color": int(dot_color),
        "dot_count": int(dot_count),
        "dot_mask": dot_mask,
        "objects": objects,
        "target_indices": target_indices,
        "visible_match_indices": visible_match_indices,
    }


def render_object(info: dict, bg: int, fill_color: int) -> np.ndarray:
    out = np.full(info["subgrid"].shape, int(bg), dtype=int)
    out[info["shell_mask"]] = int(info["shell_color"])
    out[info["hole_mask"]] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count isolated dots, pick the object whose total hole count matches that count, and fill all of its holes with the dot color.

    Input structure:
    - The background is dominant, with several separated foreground objects and some far-away single-pixel dots.
    - Each object is mostly one solid color, but may contain visible background holes and single-cell plug pixels of another color.
    - All dots share one color that is different from the shell color and the plug color.

    Rule:
    - Count the isolated dots to get m.
    - For each foreground object, count total holes = visible holes + plugged holes, where each plug pixel stands for one filled-in hole.
    - Select the unique object whose total hole count equals m, crop it to its tight bounding box, and recolor every hole cell and every plug position with the dot color.

    Key insight:
    - The colored singletons provide both the selection count and the fill color, and plug pixels must be treated as hidden holes rather than ordinary foreground.
    """
    analysis = analyze_grid(np.asarray(grid, dtype=int))
    if len(analysis["target_indices"]) != 1:
        raise AssertionError("expected a unique object with total hole count equal to the dot count")
    target = analysis["objects"][analysis["target_indices"][0]]
    return render_object(target, analysis["bg"], analysis["dot_color"])


def choose_shape(prefer_large: bool) -> np.ndarray:
    pool = [shape for shape in VISIBLE_SHAPES if (shape.sum() > 1) or not prefer_large]
    shape = np.array(random.choice(pool), dtype=bool)
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    shape = np.rot90(shape, k=random.randrange(4))
    return shape.copy()


def build_patch(
    visible_count: int,
    plug_count: int,
    *,
    bg: int,
    shell: int,
    plug: int,
    dot: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(visible_count + plug_count)
    candidates: list[tuple[int, int]] = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            capacity = rows * cols
            if capacity < total or capacity > total + 2:
                continue
            candidates.append((rows, cols))
    rows, cols = random.choice(candidates)
    patch = np.full((3 * rows + 1, 3 * cols + 1), int(shell), dtype=int)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slots)
    hole_slots = slots[:total]
    visible_slots = hole_slots[:visible_count]
    plug_slots = hole_slots[visible_count:]

    for idx, (sr, sc) in enumerate(visible_slots):
        shape = choose_shape(prefer_large=(role == "target" and idx == 0))
        sh, sw = shape.shape
        base_r = 1 + 3 * sr
        base_c = 1 + 3 * sc
        off_r = random.randint(0, 2 - sh)
        off_c = random.randint(0, 2 - sw)
        view = patch[base_r + off_r : base_r + off_r + sh, base_c + off_c : base_c + off_c + sw]
        view[shape] = int(bg)

    for sr, sc in plug_slots:
        pr = 1 + 3 * sr + random.randint(0, 1)
        pc = 1 + 3 * sc + random.randint(0, 1)
        patch[pr, pc] = int(plug)

    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, k=random.randrange(4))

    expected = patch.copy()
    expected[expected != int(shell)] = int(dot)
    return patch, expected


def sample_specs(dot_count: int, n_objects: int) -> list[dict]:
    target_visible = random.randint(1, dot_count - 1)
    target_plugs = dot_count - target_visible
    decoy_plugs = random.randint(1, min(2, 7 - dot_count))

    specs = [
        {"role": "target", "visible": target_visible, "plug": target_plugs, "total": dot_count},
        {"role": "decoy", "visible": dot_count, "plug": decoy_plugs, "total": dot_count + decoy_plugs},
    ]

    used_totals = {int(spec["total"]) for spec in specs}
    candidate_totals = [t for t in range(1, 8) if t not in used_totals]
    remaining = n_objects - 2
    chosen_totals: list[int] = []

    near_totals = [t for t in candidate_totals if abs(t - dot_count) <= 2]
    if remaining > 0 and near_totals:
        t = random.choice(near_totals)
        chosen_totals.append(int(t))
        candidate_totals.remove(int(t))

    while len(chosen_totals) < remaining:
        t = random.choice(candidate_totals)
        chosen_totals.append(int(t))
        candidate_totals.remove(int(t))

    for total in chosen_totals:
        options = [(v, total - v) for v in range(total + 1) if v != dot_count]
        if total > 1 and random.random() < 0.6:
            richer = [pair for pair in options if pair[0] > 0 and pair[1] > 0]
            if richer:
                options = richer
        visible, plug = random.choice(options)
        specs.append({"role": "other", "visible": int(visible), "plug": int(plug), "total": int(total)})

    random.shuffle(specs)
    return specs


def sector_boxes(height: int, width: int) -> list[BBox]:
    mr = height // 2
    mc = width // 2
    return [
        (1, 1, mr - 2, mc - 2),
        (1, mc + 1, mr - 2, width - 2),
        (mr + 1, 1, height - 2, mc - 2),
        (mr + 1, mc + 1, height - 2, width - 2),
    ]


def place_patch(grid: np.ndarray, patch: np.ndarray, box: BBox) -> BBox | None:
    r0, c0, r1, c1 = map(int, box)
    h, w = patch.shape
    if h > (r1 - r0 + 1) or w > (c1 - c0 + 1):
        return None
    top = random.randint(r0, r1 - h + 1)
    left = random.randint(c0, c1 - w + 1)
    grid[top : top + h, left : left + w] = patch
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def can_place_dot(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    if int(grid[r, c]) != int(bg):
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != int(bg):
                return False
    return True


def validate_scene(grid: np.ndarray, expected_output: np.ndarray, expected_roles: dict, dot_count: int) -> bool:
    try:
        analysis = analyze_grid(grid)
    except Exception:
        return False

    if int(analysis["dot_count"]) != int(dot_count):
        return False
    if len(analysis["target_indices"]) != 1:
        return False
    if len(analysis["visible_match_indices"]) != 1:
        return False

    bbox_to_role = {tuple(map(int, box)): role for box, role in expected_roles.items()}
    roles = []
    for obj in analysis["objects"]:
        box = tuple(map(int, obj["bbox"]))
        if box not in bbox_to_role:
            return False
        roles.append(bbox_to_role[box])

    role_to_obj = {
        bbox_to_role[tuple(map(int, obj["bbox"]))]: obj
        for obj in analysis["objects"]
    }

    target = role_to_obj.get("target")
    decoy = role_to_obj.get("decoy")
    if target is None or decoy is None:
        return False
    if int(target["total_count"]) != int(dot_count):
        return False
    if int(target["visible_count"]) >= int(dot_count):
        return False
    if int(target["plug_count"]) <= 0 or int(target["visible_count"]) <= 0:
        return False
    if int(decoy["visible_count"]) != int(dot_count):
        return False
    if int(decoy["total_count"]) <= int(dot_count):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected_output):
        return False

    visible_match_role = bbox_to_role[tuple(map(int, analysis["objects"][analysis["visible_match_indices"][0]]["bbox"]))]
    target_role = bbox_to_role[tuple(map(int, analysis["objects"][analysis["target_indices"][0]]["bbox"]))]
    if visible_match_role != "decoy" or target_role != "target":
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg, shell, plug, dot = random.sample(range(10), 4)
    dot_count = random.randint(2, 5)
    n_objects = random.choice((3, 4))

    specs = sample_specs(dot_count, n_objects)
    patches = []
    for spec in specs:
        patch, expected = build_patch(
            spec["visible"],
            spec["plug"],
            bg=bg,
            shell=shell,
            plug=plug,
            dot=dot,
            role=spec["role"],
        )
        patches.append((spec, patch, expected))

    height = random.randint(26, 30)
    width = random.randint(26, 30)
    grid = np.full((height, width), int(bg), dtype=int)

    sectors = sector_boxes(height, width)
    chosen_sectors = random.sample(range(4), len(patches))
    expected_roles: dict[BBox, str] = {}
    expected_output = None

    for (spec, patch, expected), sector_id in zip(patches, chosen_sectors):
        placed = place_patch(grid, patch, sectors[sector_id])
        if placed is None:
            return None
        expected_roles[placed] = str(spec["role"])
        if spec["role"] == "target":
            expected_output = expected

    if expected_output is None:
        return None

    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if can_place_dot(grid, r, c, bg)]
    random.shuffle(candidates)
    if len(candidates) < dot_count:
        return None
    for r, c in candidates[:dot_count]:
        grid[r, c] = int(dot)

    if not validate_scene(grid, expected_output, expected_roles, dot_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
