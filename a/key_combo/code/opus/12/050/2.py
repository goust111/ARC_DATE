from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 500
GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (15, 18)


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            cur = np.rot90(base, k)
            key = cur.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
    return out


LARGE_SHAPES = [
    _mask(
        [
            "1001",
            "1001",
            "1001",
            "0110",
        ]
    ),
    np.fliplr(
        _mask(
            [
                "1001",
                "1000",
                "1001",
                "0111",
            ]
        )
    ),
    _mask(
        [
            "1001",
            "1000",
            "1001",
            "0111",
        ]
    ),
]

MEDIUM_SHAPES = [
    _mask(
        [
            "101",
            "100",
            "011",
        ]
    ),
    np.fliplr(
        _mask(
            [
                "101",
                "100",
                "011",
            ]
        )
    ),
]

SMALL_SHAPES = []
for base in (
    _mask(["11", "10"]),
    _mask(["11", "01"]),
    _mask(["10", "11", "01"]),
    _mask(["11", "10", "11"]),
    _mask(["111", "010"]),
):
    SMALL_SHAPES.extend(_unique_transforms(base))

INNER_SMALL_SHAPES = [
    shape.copy()
    for shape in SMALL_SHAPES
    if shape.shape[1] <= 2
]


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, box)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _extract_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        comp["color"] = int(comp["color"])
        comp["area"] = int(comp["area"])
        comp["bbox"] = tuple(map(int, comp["bbox"]))
    return comps


def _drop_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    comps = _extract_components(g)
    comps.sort(key=lambda comp: (comp["bbox"][2], comp["bbox"][0], comp["bbox"][1]), reverse=True)

    settled = np.zeros_like(g)
    for comp in comps:
        coords = comp["coords"].copy()
        color = int(comp["color"])
        shift = 0
        while True:
            moved = coords.copy()
            moved[:, 0] += shift + 1
            if int(moved[:, 0].max()) >= h:
                break
            if np.any(settled[moved[:, 0], moved[:, 1]] != BG):
                break
            shift += 1
        coords[:, 0] += shift
        settled[coords[:, 0], coords[:, 1]] = color
    return settled


def _containment_graph(comps: list[dict]) -> tuple[list[list[int]], list[set[int]]]:
    outers_by_inner: list[list[int]] = [[] for _ in comps]
    children_by_outer: list[set[int]] = [set() for _ in comps]
    for i, inner in enumerate(comps):
        for j, outer in enumerate(comps):
            if i == j:
                continue
            if int(inner["color"]) == int(outer["color"]):
                continue
            if int(outer["area"]) <= int(inner["area"]):
                continue
            if _bbox_area(outer["bbox"]) <= _bbox_area(inner["bbox"]):
                continue
            if _bbox_contains(outer["bbox"], inner["bbox"]):
                outers_by_inner[i].append(j)
                children_by_outer[j].add(i)
    return outers_by_inner, children_by_outer


def _erase_one_layer(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    out = np.asarray(grid, dtype=int).copy()
    comps = _extract_components(out)
    outers_by_inner, children_by_outer = _containment_graph(comps)
    erasable = [
        idx
        for idx in range(len(comps))
        if outers_by_inner[idx] and not children_by_outer[idx]
    ]
    erased_colors: set[int] = set()
    for idx in erasable:
        comp = comps[idx]
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        erased_colors.add(int(comp["color"]))
    return out, erased_colors


def _erase_layers(grid: np.ndarray) -> tuple[np.ndarray, list[set[int]]]:
    out = np.asarray(grid, dtype=int).copy()
    layers: list[set[int]] = []
    while True:
        nxt, erased = _erase_one_layer(out)
        if not erased:
            return out, layers
        layers.append(erased)
        out = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """Drop every floating object downward, then erase smaller foreign-color objects whose post-drop bounding boxes become enclosed.

    Input structure:
    - The grid is black background with 4 to 6 separated 8-connected monochrome foreground objects.
    - At least one large sparse object has a bounding box of at least 4x4, and several small objects have bounding boxes of at most 3x3.
    - Before motion, no object's bounding box is contained in another object's bounding box.

    Rule:
    - Identify every 8-connected non-black monochrome object.
    - Apply gravity to each object as a rigid body, processing lower objects first so each object falls until it hits the bottom edge or an already settled object.
    - Recompute objects after the fall; whenever an object's bounding box is contained in the larger bounding box of a different-color object, erase that inner object.
    - If containment is nested across multiple levels, erase the innermost layer first and repeat the containment test on the new grid until nothing more can be removed.

    Key insight:
    - The initial grid is arranged so containment does not exist yet; gravity changes the vertical relationships, and only that new post-drop geometry reveals which objects must be erased.
    """
    settled = _drop_grid(np.asarray(grid, dtype=int))
    erased, _layers = _erase_layers(settled)
    return erased


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    mask = patch != BG
    target = grid[top : top + h, left : left + w]
    return bool(np.all(target[mask] == BG))


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = patch[mask]


def _placements_within_bbox(
    grid: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    ph, pw = patch.shape
    places = []
    for top in range(r0, r1 - ph + 2):
        for left in range(c0, c1 - pw + 2):
            if _can_place(grid, patch, top, left):
                places.append((int(top), int(left)))
    return places


def _patch_bbox(top: int, left: int, patch: np.ndarray) -> tuple[int, int, int, int]:
    return int(top), int(left), int(top + patch.shape[0] - 1), int(left + patch.shape[1] - 1)


def _shape_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _cells_from_patch(top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    pts = set()
    for r, c in np.argwhere(patch != BG):
        pts.add((int(top + r), int(left + c)))
    return pts


def _objects_not_adjacent(objs: list[dict]) -> bool:
    cells_by_obj = [_cells_from_patch(obj["top"], obj["left"], obj["patch"]) for obj in objs]
    for i in range(len(cells_by_obj)):
        for j in range(i + 1, len(cells_by_obj)):
            for r, c in cells_by_obj[i]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in cells_by_obj[j]:
                            return False
    return True


def _grid_has_no_input_containment(grid: np.ndarray) -> bool:
    comps = _extract_components(grid)
    for i, inner in enumerate(comps):
        for j, outer in enumerate(comps):
            if i == j:
                continue
            if _bbox_contains(outer["bbox"], inner["bbox"]):
                return False
    return True


def _sample_large_shape() -> np.ndarray:
    return random.choice(LARGE_SHAPES).copy()


def _sample_medium_shape() -> np.ndarray:
    return random.choice(MEDIUM_SHAPES).copy()


def _sample_small_shape(max_area: int | None = None) -> np.ndarray:
    choices = SMALL_SHAPES
    if max_area is not None:
        choices = [shape for shape in SMALL_SHAPES if _shape_area(shape) <= int(max_area)]
    return random.choice(choices).copy()


def _sample_inner_small_shape(max_area: int | None = None) -> np.ndarray:
    choices = INNER_SMALL_SHAPES
    if max_area is not None:
        choices = [shape for shape in INNER_SMALL_SHAPES if _shape_area(shape) <= int(max_area)]
    return random.choice(choices).copy()


def _place_bottom_pair(
    grid_shape: tuple[int, int],
    width_a: int,
    width_b: int,
) -> tuple[int, int] | None:
    _h, w = grid_shape
    pairs = []
    for a_left in range(0, w - width_a + 1):
        for b_left in range(0, w - width_b + 1):
            if a_left == b_left:
                continue
            a_right = a_left + width_a - 1
            b_right = b_left + width_b - 1
            if a_right + 3 < b_left or b_right + 3 < a_left:
                pairs.append((int(a_left), int(b_left)))
    return random.choice(pairs) if pairs else None


def _find_outside_bottom_position(
    grid: np.ndarray,
    patch: np.ndarray,
    forbidden_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    top = int(grid.shape[0] - patch.shape[0])
    candidates = []
    for left in range(0, grid.shape[1] - patch.shape[1] + 1):
        if not _can_place(grid, patch, top, left):
            continue
        box = _patch_bbox(top, left, patch)
        if any(_bbox_contains(outer, box) for outer in forbidden_boxes):
            continue
        candidates.append((int(top), int(left)))
    return random.choice(candidates) if candidates else None


def _build_input_from_settled(shape: tuple[int, int], settled_objects: list[dict]) -> np.ndarray | None:
    by_lane: dict[str, list[dict]] = {}
    for obj in settled_objects:
        by_lane.setdefault(str(obj["lane"]), []).append(obj)

    inp = np.zeros(shape, dtype=int)
    placed: list[dict] = []
    outer_tops: dict[str, int] = {}

    for lane in ("A", "B"):
        outer = next(obj for obj in by_lane[lane] if obj["role"] == "outer")
        max_lift = min(4, int(outer["settled_top"]))
        if max_lift < 2:
            return None
        outer_top = int(outer["settled_top"]) - random.randint(2, max_lift)
        outer_tops[lane] = outer_top

        for obj in by_lane[lane]:
            if obj["role"] == "outer":
                continue
            if obj["role"] == "tiny":
                priority = 0
            else:
                priority = 1
            obj["priority"] = priority

    lane_positions: list[tuple[dict, int, int]] = []
    for lane in ("A", "B"):
        outer = next(obj for obj in by_lane[lane] if obj["role"] == "outer")
        inner_objs = sorted(
            [obj for obj in by_lane[lane] if obj["role"] != "outer"],
            key=lambda obj: int(obj["priority"]),
            reverse=True,
        )
        next_bottom = int(outer_tops[lane]) - 2
        for obj in inner_objs:
            gap = random.randint(1, 2)
            top = next_bottom - obj["patch"].shape[0] + 1
            if top < 0:
                return None
            lane_positions.append((obj, int(top), int(obj["settled_left"])))
            next_bottom = int(top) - gap - 1
        lane_positions.append((outer, int(outer_tops[lane]), int(outer["settled_left"])))

    highest_outer = min(outer_tops.values())
    for obj in by_lane.get("S", []):
        top_hi = highest_outer - obj["patch"].shape[0] - 2
        if top_hi < 1:
            return None
        top = random.randint(1, top_hi)
        lane_positions.append((obj, int(top), int(obj["settled_left"])))

    for obj, top, left in lane_positions:
        if not _can_place(inp, obj["patch"], top, left):
            return None
        _paste(inp, obj["patch"], top, left)
        placed.append(
            {
                "top": int(top),
                "left": int(left),
                "patch": obj["patch"],
                "role": obj["role"],
                "color": int(obj["color"]),
            }
        )

    if not _objects_not_adjacent(placed):
        return None
    if not _grid_has_no_input_containment(inp):
        return None
    return inp


def _validate_candidate(inp: np.ndarray, meta: dict) -> bool:
    comps = _extract_components(inp)
    if not (4 <= len(comps) <= 6):
        return False
    if any(not (3 <= int(comp["area"]) <= 8) for comp in comps):
        return False
    if any(_bbox_area(comp["bbox"]) > 9 and _bbox_area(comp["bbox"]) < 16 for comp in comps):
        return False
    if not any(_bbox_area(comp["bbox"]) >= 16 for comp in comps):
        return False
    if not any(_bbox_area(comp["bbox"]) <= 9 for comp in comps):
        return False
    if not _grid_has_no_input_containment(inp):
        return False

    settled = _drop_grid(inp)
    settled_comps = _extract_components(settled)
    if len(settled_comps) != len(comps):
        return False

    outers_by_inner, _children = _containment_graph(settled_comps)
    contained = [idx for idx, outers in enumerate(outers_by_inner) if outers]
    if not contained:
        return False

    out, layers = _erase_layers(settled)
    if not layers:
        return False
    if np.array_equal(out, settled):
        return False

    remaining_colors = set(map(int, np.unique(out))) - {BG}
    if int(meta["survivor_color"]) not in remaining_colors:
        return False
    for color in meta["erased_colors"]:
        if int(color) in remaining_colors:
            return False
    if meta["nested"] and len(layers) < 2:
        return False
    if not meta["nested"] and len(layers) < 1:
        return False
    return bool(np.array_equal(main(inp), out))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nested = random.random() < 0.4

    needed_colors = 6 if nested else 5
    colors = random.sample(PALETTE, needed_colors)
    color_iter = iter(colors)

    outer_a = _colorize(_sample_large_shape(), next(color_iter))
    outer_b = _colorize(_sample_large_shape(), next(color_iter))

    pair = _place_bottom_pair((h, w), outer_a.shape[1], outer_b.shape[1])
    if pair is None:
        return None
    a_left, b_left = pair
    if random.random() < 0.5:
        a_left, b_left = b_left, a_left

    grid = np.zeros((h, w), dtype=int)
    a_top = h - outer_a.shape[0]
    b_top = h - outer_b.shape[0]
    if not _can_place(grid, outer_a, a_top, a_left):
        return None
    if not _can_place(grid, outer_b, b_top, b_left):
        return None
    _paste(grid, outer_a, a_top, a_left)
    _paste(grid, outer_b, b_top, b_left)

    objects = [
        {
            "patch": outer_a,
            "settled_top": int(a_top),
            "settled_left": int(a_left),
            "role": "outer",
            "color": int(np.max(outer_a)),
            "lane": "A",
        },
        {
            "patch": outer_b,
            "settled_top": int(b_top),
            "settled_left": int(b_left),
            "role": "outer",
            "color": int(np.max(outer_b)),
            "lane": "B",
        },
    ]
    erased_colors: list[int] = []

    if nested:
        medium = _colorize(_sample_medium_shape(), next(color_iter))
        a_box = _patch_bbox(a_top, a_left, outer_a)
        medium_places = _placements_within_bbox(grid, medium, a_box)
        if not medium_places:
            return None
        m_top, m_left = random.choice(medium_places)
        _paste(grid, medium, m_top, m_left)
        medium_color = int(np.max(medium))
        objects.append(
            {
                "patch": medium,
                "settled_top": int(m_top),
                "settled_left": int(m_left),
                "role": "medium",
                "color": medium_color,
                "lane": "A",
            }
        )
        erased_colors.append(medium_color)

        tiny = _colorize(_sample_inner_small_shape(max_area=4), next(color_iter))
        medium_box = _patch_bbox(m_top, m_left, medium)
        tiny_places = _placements_within_bbox(grid, tiny, medium_box)
        if not tiny_places:
            return None
        t_top, t_left = random.choice(tiny_places)
        _paste(grid, tiny, t_top, t_left)
        tiny_color = int(np.max(tiny))
        objects.append(
            {
                "patch": tiny,
                "settled_top": int(t_top),
                "settled_left": int(t_left),
                "role": "tiny",
                "color": tiny_color,
                "lane": "A",
            }
        )
        erased_colors.append(tiny_color)
    else:
        direct = _colorize(_sample_inner_small_shape(max_area=5), next(color_iter))
        a_box = _patch_bbox(a_top, a_left, outer_a)
        direct_places = _placements_within_bbox(grid, direct, a_box)
        if not direct_places:
            return None
        d_top, d_left = random.choice(direct_places)
        _paste(grid, direct, d_top, d_left)
        direct_color = int(np.max(direct))
        objects.append(
            {
                "patch": direct,
                "settled_top": int(d_top),
                "settled_left": int(d_left),
                "role": "target",
                "color": direct_color,
                "lane": "A",
            }
        )
        erased_colors.append(direct_color)

    secondary = _colorize(_sample_inner_small_shape(max_area=5), next(color_iter))
    b_box = _patch_bbox(b_top, b_left, outer_b)
    secondary_places = _placements_within_bbox(grid, secondary, b_box)
    if not secondary_places:
        return None
    s_top, s_left = random.choice(secondary_places)
    _paste(grid, secondary, s_top, s_left)
    secondary_color = int(np.max(secondary))
    objects.append(
        {
            "patch": secondary,
            "settled_top": int(s_top),
            "settled_left": int(s_left),
            "role": "target",
            "color": secondary_color,
            "lane": "B",
        }
    )
    erased_colors.append(secondary_color)

    survivor = _colorize(_sample_small_shape(max_area=5), next(color_iter))
    survivor_place = _find_outside_bottom_position(grid, survivor, [a_box, b_box])
    if survivor_place is None:
        return None
    y_top, y_left = survivor_place
    _paste(grid, survivor, y_top, y_left)
    survivor_color = int(np.max(survivor))
    objects.append(
        {
            "patch": survivor,
            "settled_top": int(y_top),
            "settled_left": int(y_left),
            "role": "survivor",
            "color": survivor_color,
            "lane": "S",
        }
    )

    inp = _build_input_from_settled((h, w), objects)
    if inp is None:
        return None

    meta = {
        "nested": nested,
        "erased_colors": erased_colors,
        "survivor_color": survivor_color,
    }
    if not _validate_candidate(inp, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid gravity-bbox-erase sample")
