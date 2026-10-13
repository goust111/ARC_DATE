from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste


BOX_SIZE = 5
BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COUNT_CHOICES = (4, 5, 6)
AXES = ("horizontal", "vertical")
GEN_RETRIES = 400


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


def _all_symmetries(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    cur = mask.copy()
    for _ in range(4):
        out.append(cur.copy())
        out.append(np.fliplr(cur).copy())
        cur = np.rot90(cur)
    return out


def _hole_count_from_crop(crop: np.ndarray, bg: int) -> int:
    bg_components = components_mask(crop == bg, connectivity=4)
    holes = 0
    h, w = crop.shape
    for coords in bg_components:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        holes += 1
    return holes


def _touching_edge(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _boxes_with_halo_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprites = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert 4 <= len(sprites) <= 6

    items = []
    shapes = set()
    centers = []
    colors = {int(sprite["color"]) for sprite in sprites}
    assert len(colors) == 1

    for sprite in sprites:
        bbox = tuple(map(int, sprite["bbox"]))
        crop = crop_bbox(g, bbox)
        holes = _hole_count_from_crop(crop, bg)
        area = int(sprite["area"])
        r0, c0, r1, c1 = bbox
        items.append(
            {
                "bbox": bbox,
                "crop": crop,
                "holes": holes,
                "area": area,
                "top_left": (r0, c0),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
        )
        shapes.add(tuple(crop.shape))
        centers.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))

    assert len(shapes) == 1
    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    axis = "horizontal" if col_span > row_span else "vertical"
    return items, axis, bg


def _pack_items(items: list[dict], axis: str, bg: int) -> np.ndarray:
    item_h, item_w = items[0]["crop"].shape
    if axis == "horizontal":
        out = np.full((item_h, item_w * len(items)), bg, dtype=int)
        for i, item in enumerate(items):
            paste(out, item["crop"], 0, i * item_w)
    else:
        out = np.full((item_h * len(items), item_w), bg, dtype=int)
        for i, item in enumerate(items):
            paste(out, item["crop"], i * item_h, 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count each sprite's enclosed holes, sort the sprites by that topology, and tightly concatenate their bbox crops.

    Input structure:
    - A black background contains 4-6 separated monochrome 8-connected sprites.
    - All sprites use the same foreground color and share one bbox size.
    - Each sprite may contain 0-3 enclosed black regions inside its own bbox crop.

    Rule:
    - Identify every foreground sprite and crop it to its minimal bbox.
    - Count, inside each crop, the number of black 4-connected regions that do not touch the crop border.
    - Sort sprites by ascending hole count, then by ascending foreground area, then by row-major order of the original bbox top-left corner.
    - If the sprite centers span more columns than rows, pack the sorted crops horizontally; otherwise pack them vertically.

    Key insight:
    - The ordering key is a hidden topological property of each sprite: the number of enclosed holes, not its raw position or pixel count.
    """
    items, axis, bg = _analyze_grid(np.asarray(grid, dtype=int))
    ordered = sorted(items, key=lambda item: (item["holes"], item["area"], item["top_left"]))
    return _pack_items(ordered, axis, bg)


TEMPLATE_ROWS = {
    "h0_a12": (
        "#..#.",
        ".###.",
        "..#..",
        ".###.",
        ".##.#",
    ),
    "h0_a18_a": (
        "##.##",
        "##..#",
        "###.#",
        "###.#",
        ".#.##",
    ),
    "h0_a18_b": (
        "#.###",
        "#.#..",
        "#####",
        "###..",
        "##.##",
    ),
    "h0_a14": (
        "##..#",
        "#.##.",
        "#....",
        "###.#",
        ".###.",
    ),
    "h1_a15": (
        "##...",
        "###.#",
        "#.##.",
        "###.#",
        "#.#..",
    ),
    "h1_a16": (
        "##.#.",
        "#####",
        ".#.##",
        ".##..",
        "##.#.",
    ),
    "h2_a17": (
        "#.#..",
        "##.##",
        ".#.##",
        "###.#",
        "#.###",
    ),
    "h2_a19": (
        "##.##",
        "###.#",
        "#..##",
        "#####",
        "#.#.#",
    ),
    "h3_a19": (
        "#.###",
        "##.##",
        "###.#",
        "#.###",
        "##..#",
    ),
    "h3_a20": (
        ".####",
        ".#.##",
        "#####",
        "#.#.#",
        "#####",
    ),
}


def _template_props(mask: np.ndarray) -> tuple[int, int]:
    area = int(mask.sum())
    crop = mask.astype(int)
    holes = _hole_count_from_crop(crop, 0)
    return holes, area


TEMPLATES = {}
for name, rows in TEMPLATE_ROWS.items():
    mask = _mask_from_rows(rows)
    holes, area = _template_props(mask)
    TEMPLATES[name] = {
        "name": name,
        "mask": mask,
        "holes": holes,
        "area": area,
    }
    assert mask.shape == (BOX_SIZE, BOX_SIZE)
    assert holes == int(name.split("_")[0][1:])
    assert area == int(name.split("_")[1][1:])
    assert len(find_components(mask, bg=0, connectivity=8, monochromatic=True, return_format="dict")) == 1

assert not any(np.array_equal(sym, TEMPLATES["h0_a18_b"]["mask"]) for sym in _all_symmetries(TEMPLATES["h0_a18_a"]["mask"]))


def _random_template_variant(template: dict) -> dict:
    mask = random.choice(_all_symmetries(template["mask"]))
    return {
        "name": template["name"],
        "mask": mask,
        "holes": template["holes"],
        "area": template["area"],
    }


def _choose_scene_templates(count: int) -> list[dict]:
    base = [
        _random_template_variant(TEMPLATES["h0_a18_a"]),
        _random_template_variant(TEMPLATES["h0_a18_b"]),
        _random_template_variant(TEMPLATES["h0_a12"]),
        _random_template_variant(random.choice([TEMPLATES["h1_a15"], TEMPLATES["h1_a16"]])),
    ]
    extras = []
    if count >= 5:
        extras.append(_random_template_variant(random.choice([TEMPLATES["h2_a17"], TEMPLATES["h2_a19"]])))
    if count >= 6:
        extras.append(_random_template_variant(random.choice([TEMPLATES["h3_a19"], TEMPLATES["h3_a20"], TEMPLATES["h0_a14"]])))
    scene = base + extras
    random.shuffle(scene)
    return scene


def _sample_positions(count: int, axis: str, shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    row_choices = list(range(1, h - BOX_SIZE))
    col_choices = list(range(1, w - BOX_SIZE))
    assert row_choices and col_choices

    if axis == "horizontal":
        band_center = random.randint(3, h - BOX_SIZE - 2)
        preferred_rows = [r for r in row_choices if abs(r - band_center) <= 3]
        if not preferred_rows:
            preferred_rows = row_choices
        candidate_positions = [(r, c) for c in col_choices for r in preferred_rows]
    else:
        band_center = random.randint(3, w - BOX_SIZE - 2)
        preferred_cols = [c for c in col_choices if abs(c - band_center) <= 3]
        if not preferred_cols:
            preferred_cols = col_choices
        candidate_positions = [(r, c) for r in row_choices for c in preferred_cols]

    for _ in range(250):
        positions = []
        boxes = []
        for _idx in range(count):
            random.shuffle(candidate_positions)
            chosen = None
            for r0, c0 in candidate_positions:
                box = (r0, c0, r0 + BOX_SIZE - 1, c0 + BOX_SIZE - 1)
                if any(_boxes_with_halo_overlap(box, prev) for prev in boxes):
                    continue
                chosen = (r0, c0)
                positions.append(chosen)
                boxes.append(box)
                break
            if chosen is None:
                break
        if len(positions) != count:
            continue

        centers = [(r0 + (BOX_SIZE - 1) / 2.0, c0 + (BOX_SIZE - 1) / 2.0) for r0, c0 in positions]
        row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
        col_span = max(c for _, c in centers) - min(c for _, c in centers)
        if axis == "horizontal" and not (col_span > row_span):
            continue
        if axis == "vertical" and not (row_span >= col_span):
            continue
        return positions
    return None


def _sorted_feature_signature(templates: list[dict]) -> list[tuple[int, int]]:
    return [(item["holes"], item["area"]) for item in sorted(templates, key=lambda item: (item["holes"], item["area"], item["name"]))]


def _placement_permutation(templates: list[dict], positions: list[tuple[int, int]]) -> list[dict] | None:
    row_major_positions = sorted(positions)
    sorted_templates = sorted(templates, key=lambda item: (item["holes"], item["area"], item["name"]))
    signatures = _sorted_feature_signature(templates)
    for _ in range(200):
        permuted = templates[:]
        random.shuffle(permuted)
        if all(permuted[i]["name"] == sorted_templates[i]["name"] for i in range(len(permuted))):
            continue

        row_major_features = [
            (item["holes"], item["area"])
            for _, item in sorted(zip(row_major_positions, permuted), key=lambda pair: pair[0])
        ]
        if row_major_features == signatures:
            continue
        return permuted
    return None


def _render_input(templates: list[dict], positions: list[tuple[int, int]], color: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for template, (r0, c0) in zip(templates, positions):
        patch = np.where(template["mask"] == 1, color, BACKGROUND)
        paste(grid, patch, r0, c0, transparent=BACKGROUND)
    return grid


def _manual_expected(grid: np.ndarray) -> np.ndarray:
    items, axis, bg = _analyze_grid(grid)
    ordered = sorted(items, key=lambda item: (item["holes"], item["area"], item["top_left"]))
    return _pack_items(ordered, axis, bg)


def _semantic_checks(grid: np.ndarray) -> None:
    items, axis, bg = _analyze_grid(grid)
    assert bg == BACKGROUND
    assert axis in AXES
    assert len({tuple(item["crop"].shape) for item in items}) == 1

    hole_counts = [item["holes"] for item in items]
    areas = [item["area"] for item in items]
    top_lefts = [item["top_left"] for item in items]
    assert len(set(hole_counts)) >= 2

    boxes = [item["bbox"] for item in items]
    for box in boxes:
        assert not _touching_edge(box, grid.shape)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _boxes_with_halo_overlap(boxes[i], boxes[j])

    sorted_items = sorted(items, key=lambda item: (item["holes"], item["area"], item["top_left"]))
    sorted_features = [(item["holes"], item["area"], item["top_left"]) for item in sorted_items]
    area_only_features = [(item["holes"], item["area"], item["top_left"]) for item in sorted(items, key=lambda item: (item["area"], item["top_left"]))]
    row_major_features = [(item["holes"], item["area"], item["top_left"]) for item in sorted(items, key=lambda item: item["top_left"])]
    assert sorted_features != area_only_features
    assert sorted_features != row_major_features

    feature_groups = {}
    for item in items:
        feature_groups.setdefault((item["holes"], item["area"]), []).append(item)
    assert any(len(group) >= 2 and not np.array_equal(group[0]["crop"], group[1]["crop"]) for group in feature_groups.values())

    same_hole_diff_area = False
    for holes in set(hole_counts):
        hole_areas = {item["area"] for item in items if item["holes"] == holes}
        if len(hole_areas) >= 2:
            same_hole_diff_area = True
            break
    assert same_hole_diff_area

    expected = _manual_expected(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    if axis == "horizontal":
        assert actual.shape[0] == BOX_SIZE and actual.shape[1] == BOX_SIZE * len(items)
    else:
        assert actual.shape[0] == BOX_SIZE * len(items) and actual.shape[1] == BOX_SIZE


def _generate_attempt() -> np.ndarray | None:
    count = random.choice(COUNT_CHOICES)
    axis = random.choice(AXES)
    color = random.choice(COLOR_POOL)
    if axis == "horizontal":
        shape = (random.randint(16, 24), random.randint(22, 30))
    else:
        shape = (random.randint(22, 30), random.randint(16, 24))

    templates = _choose_scene_templates(count)
    positions = _sample_positions(count, axis, shape)
    if positions is None:
        return None

    assigned_templates = _placement_permutation(templates, positions)
    if assigned_templates is None:
        return None

    grid = _render_input(assigned_templates, positions, color, shape)
    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 2")
