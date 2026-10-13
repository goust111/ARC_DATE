from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import (  # noqa: E402
    Color,
    bbox_coords,
    components_mask,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


BG = Color.BLACK
BLOCK_COLOR = Color.GREY
KEY_COLOR = Color.YELLOW
TEMPLATE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE)

GRID_MIN_SIDE = 20
GRID_MAX_SIDE = 30
BLOCK_COUNT_CHOICES = (2, 3)
BLOCK_SIDE_RANGE = (4, 5)
GENERATION_RETRIES = 400
TEMPLATE_LAYOUT_RETRIES = 80


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for transformed in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            transforms.append(transformed.copy())
    return transforms


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, transformed.shape)), transformed.tobytes()) for transformed in _unique_transforms(mask))


def _shape_library() -> dict[int, list[np.ndarray]]:
    raw_shapes = [
        [[1, 1]],
        [[1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
    ]
    by_area: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for raw in raw_shapes:
        mask = np.array(raw, dtype=bool)
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        by_area.setdefault(int(mask.sum()), []).append(mask)
    return by_area


SHAPES_BY_AREA = _shape_library()
AREA_PATTERNS_2 = ((2, 3), (2, 4), (3, 4))
AREA_PATTERNS_3 = ((2, 3, 3), (2, 4, 4), (3, 4, 4), (2, 3, 4))


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _extract_key_mask(region: np.ndarray) -> np.ndarray:
    key_components = components_mask(region == KEY_COLOR, connectivity=4)
    assert len(key_components) == 1, "each block must contain exactly one yellow key component"
    coords = key_components[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    mask = (grid == BLOCK_COLOR) | (grid == KEY_COLOR)
    blocks: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(coords.shape[0]) != int(region.size):
            continue
        colors = set(map(int, np.unique(region)))
        if colors - {BLOCK_COLOR, KEY_COLOR}:
            continue
        if BLOCK_COLOR not in colors or KEY_COLOR not in colors:
            continue
        key_mask = _extract_key_mask(region)
        blocks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": region.copy(),
                "key": _canonical_key(key_mask),
            }
        )
    assert blocks, "expected at least one gray rectangle block"
    return sorted(blocks, key=lambda block: block["bbox"])


def _extract_templates(grid: np.ndarray, bg: int) -> list[dict]:
    templates: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {int(bg), BLOCK_COLOR, KEY_COLOR}:
            continue
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        rel = coords - np.array([r0, c0])
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "area": int(coords.shape[0]),
                "key": _canonical_key(mask),
            }
        )
    assert templates, "expected external template sprites"
    return sorted(templates, key=lambda template: template["bbox"])


def _match_blocks_to_templates(blocks: list[dict], templates: list[dict]) -> list[tuple[dict, dict]]:
    by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for template in templates:
        key = template["key"]
        assert key not in by_key, "template shapes must be unique up to rigid transforms"
        by_key[key] = template

    matched: list[tuple[dict, dict]] = []
    for block in blocks:
        assert block["key"] in by_key, "every block key must match a template"
        matched.append((block, by_key[block["key"]]))
    return matched


def _render_output(shape: tuple[int, int], matches: list[tuple[dict, dict]], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    painted = np.zeros(shape, dtype=bool)

    for block, template in matches:
        r0, c0, r1, c1 = block["bbox"]
        pad = int(template["area"])
        er0, ec0 = int(r0 - pad), int(c0 - pad)
        er1, ec1 = int(r1 + pad), int(c1 + pad)
        assert 0 <= er0 <= er1 < shape[0]
        assert 0 <= ec0 <= ec1 < shape[1]
        assert not bool(np.any(painted[er0 : er1 + 1, ec0 : ec1 + 1])), "expanded rings must not overlap"
        out[er0 : er1 + 1, ec0 : ec1 + 1] = int(template["color"])
        painted[er0 : er1 + 1, ec0 : ec1 + 1] = True

    for block, _template in matches:
        r0, c0, _r1, _c1 = block["bbox"]
        paste(out, block["patch"], int(r0), int(c0))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each rectangle's yellow key shape to an outside template, then draw a color ring whose thickness equals the matched template's area.

    Input structure:
    - Black background with 2 or 3 gray filled rectangles, each containing one yellow 4-connected key pattern.
    - Separate outside template sprites are single-color 4-connected components in blue/red/green/orange.
    - Each gray rectangle is a full rectangle when gray and yellow pixels are taken together, and each key matches exactly one template up to rotation/reflection.

    Rule:
    - Detect each gray rectangle block by taking connected gray-or-yellow regions whose bounding box is completely filled by gray/yellow.
    - Extract the yellow key mask inside each block and the mask of every outside template sprite.
    - Match block keys to templates with 8 rigid transforms; for the matched template, let P be its pixel count and C be its color.
    - Expand the block bounding box by P cells on all four sides and fill that expanded rectangle with color C.
    - Restore the original gray block and yellow key on top, and erase the outside templates back to background.

    Key insight:
    - The template contributes two independent outputs at once: shape decides which rectangle it controls, while template area and color decide the ring thickness and ring color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    blocks = _extract_blocks(g)
    templates = _extract_templates(g, bg)
    matches = _match_blocks_to_templates(blocks, templates)
    return _render_output(g.shape, matches, bg)


def _sample_base_shapes(count: int) -> list[np.ndarray]:
    if count == 2:
        areas = random.choice(AREA_PATTERNS_2)
    else:
        areas = random.choice(AREA_PATTERNS_3)

    selected: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], bytes]] = set()
    for area in areas:
        candidates = [shape for shape in SHAPES_BY_AREA[area] if _canonical_key(shape) not in used]
        choice = random.choice(candidates).copy()
        used.add(_canonical_key(choice))
        selected.append(choice)
    random.shuffle(selected)
    return selected


def _make_block_spec(index: int, base_shape: np.ndarray, color: int) -> dict:
    key_mask = random.choice(_unique_transforms(base_shape))
    template_mask = random.choice(_unique_transforms(base_shape))
    kh, kw = map(int, key_mask.shape)
    block_h = random.randint(max(BLOCK_SIDE_RANGE[0], kh), BLOCK_SIDE_RANGE[1])
    block_w = random.randint(max(BLOCK_SIDE_RANGE[0], kw), BLOCK_SIDE_RANGE[1])
    key_r0 = random.randint(0, block_h - kh)
    key_c0 = random.randint(0, block_w - kw)

    patch = np.full((block_h, block_w), BLOCK_COLOR, dtype=int)
    yellow_patch = np.full(key_mask.shape, KEY_COLOR, dtype=int)
    paste(patch, yellow_patch, key_r0, key_c0, mask=key_mask)

    area = int(base_shape.sum())
    return {
        "id": index,
        "key": _canonical_key(base_shape),
        "color": int(color),
        "area": area,
        "block_patch": patch,
        "template_mask": template_mask,
        "expanded_h": block_h + 2 * area,
        "expanded_w": block_w + 2 * area,
    }


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int]:
    max_h = max(spec["expanded_h"] for spec in specs)
    max_w = max(spec["expanded_w"] for spec in specs)
    total_area = sum(spec["expanded_h"] * spec["expanded_w"] for spec in specs)
    min_side = max(GRID_MIN_SIDE, max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area))) + 3)
    if min_side > GRID_MAX_SIDE:
        raise ValueError("expanded boxes cannot fit within ARC limits")
    height = random.randint(min_side, GRID_MAX_SIDE)
    width = random.randint(min_side, GRID_MAX_SIDE)
    return height, width


def _place_expanded_boxes(shape: tuple[int, int], specs: list[dict]) -> bool:
    canvas = np.zeros(shape, dtype=int)
    order = sorted(specs, key=lambda spec: spec["expanded_h"] * spec["expanded_w"], reverse=True)

    for spec in order:
        patch = np.full((spec["expanded_h"], spec["expanded_w"]), 9, dtype=int)
        try:
            er0, ec0 = random_free_location(canvas, patch, transparent=BG, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            return False
        paste(canvas, patch, er0, ec0)

        pad = spec["area"]
        block_h, block_w = spec["block_patch"].shape
        spec["expanded_bbox"] = (int(er0), int(ec0), int(er0 + spec["expanded_h"] - 1), int(ec0 + spec["expanded_w"] - 1))
        spec["bbox"] = (int(er0 + pad), int(ec0 + pad), int(er0 + pad + block_h - 1), int(ec0 + pad + block_w - 1))

    return True


def _render_blocks(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, _r1, _c1 = spec["bbox"]
        paste(grid, spec["block_patch"], int(r0), int(c0))
    return grid


def _layout_is_nontrivial(specs: list[dict], placed_templates: list[dict]) -> bool:
    block_order = [spec["key"] for spec in sorted(specs, key=lambda spec: spec["bbox"])]
    template_order = [template["key"] for template in sorted(placed_templates, key=lambda template: template["bbox"])]
    if template_order == block_order or template_order == list(reversed(block_order)):
        return False

    nearest_all_match = True
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        block_center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        distances = []
        for template in placed_templates:
            tr0, tc0, tr1, tc1 = template["bbox"]
            template_center = ((tr0 + tr1) / 2.0, (tc0 + tc1) / 2.0)
            dist = abs(block_center[0] - template_center[0]) + abs(block_center[1] - template_center[1])
            distances.append((dist, template["id"]))
        distances.sort()
        min_dist = distances[0][0]
        nearest_ids = [idx for dist, idx in distances if dist == min_dist]
        if len(nearest_ids) != 1 or nearest_ids[0] != spec["id"]:
            nearest_all_match = False
            break

    if nearest_all_match:
        return False

    area_values = {spec["area"] for spec in specs}
    return len(area_values) >= 2


def _place_templates(block_grid: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(TEMPLATE_LAYOUT_RETRIES):
        grid = block_grid.copy()
        placed: list[dict] = []
        order = list(specs)
        random.shuffle(order)

        ok = True
        for spec in order:
            mask = random.choice(_unique_transforms(spec["template_mask"]))
            patch = _mask_to_patch(mask, spec["color"])
            try:
                r0, c0 = random_free_location(grid, patch, transparent=BG, border=1, padding=1, padding_connectivity=8)
            except ValueError:
                ok = False
                break
            paste(grid, patch, r0, c0, transparent=BG)
            placed.append(
                {
                    "id": spec["id"],
                    "key": spec["key"],
                    "color": spec["color"],
                    "area": spec["area"],
                    "bbox": (int(r0), int(c0), int(r0 + mask.shape[0] - 1), int(c0 + mask.shape[1] - 1)),
                }
            )

        if ok and _layout_is_nontrivial(specs, placed):
            return grid, placed
    return None


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    matches = []
    for spec in specs:
        block = {"bbox": spec["bbox"], "patch": spec["block_patch"], "key": spec["key"]}
        template = {"color": spec["color"], "area": spec["area"], "key": spec["key"]}
        matches.append((block, template))
    return _render_output(shape, matches, BG)


def _validate_instance(grid: np.ndarray, specs: list[dict], placed_templates: list[dict]) -> bool:
    try:
        blocks = _extract_blocks(grid)
        templates = _extract_templates(grid, BG)
        matches = _match_blocks_to_templates(blocks, templates)
        out = main(grid)
    except Exception:
        return False

    if len(blocks) != len(specs) or len(templates) != len(specs):
        return False

    intended_by_key = {spec["key"]: spec for spec in specs}
    for block in blocks:
        spec = intended_by_key.get(block["key"])
        if spec is None or tuple(block["bbox"]) != tuple(spec["bbox"]):
            return False

    for template in templates:
        spec = intended_by_key.get(template["key"])
        if spec is None:
            return False
        if int(template["color"]) != int(spec["color"]) or int(template["area"]) != int(spec["area"]):
            return False

    expected = _expected_from_specs(grid.shape, specs)
    if not np.array_equal(out, expected):
        return False

    if not _layout_is_nontrivial(specs, placed_templates):
        return False

    seen_keys = {template["key"] for template in templates}
    return seen_keys == {spec["key"] for spec in specs}


def _generate_attempt() -> np.ndarray | None:
    count = random.choice(BLOCK_COUNT_CHOICES)
    base_shapes = _sample_base_shapes(count)
    colors = random.sample(TEMPLATE_COLORS, count)
    specs = [_make_block_spec(index, shape, color) for index, (shape, color) in enumerate(zip(base_shapes, colors))]

    try:
        shape = _choose_grid_shape(specs)
    except ValueError:
        return None

    if not _place_expanded_boxes(shape, specs):
        return None

    block_grid = _render_blocks(shape, specs)
    placed = _place_templates(block_grid, specs)
    if placed is None:
        return None
    grid, placed_templates = placed

    if not _validate_instance(grid, specs, placed_templates):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid expansion-ring ARC input")
