from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
TEMPLATE_COLOR = Color.TEAL
OBJECT_COLORS = (Color.BLUE, Color.RED)
PALETTE_COLORS = (Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)
PALETTE_BASE = Color.GREY

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
TOTAL_OBJECT_RANGE = (5, 8)  # includes the teal template, excludes the palette
MAX_GENERATION_ATTEMPTS = 600


def _shape_signature_from_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _shape_signature_from_mask(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    return _shape_signature_from_coords(np.argwhere(mask))


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = (int(coords[0, 0]), int(coords[0, 1]))
    stack = [start]
    seen = {start}
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _build_shape_catalog() -> tuple[dict[tuple[int, int, int], list[dict[str, object]]], list[dict[str, object]]]:
    catalog: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    all_records: list[dict[str, object]] = []
    for height in range(2, 5):
        for width in range(2, 5):
            total = height * width
            for bits in range(1 << total):
                area = bits.bit_count()
                if area < 4 or area > 8 or area < height + width - 1:
                    continue

                mask = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool).reshape(height, width)
                if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not _is_connected_mask(mask):
                    continue

                key = (height, width, area)
                record = {
                    "key": key,
                    "mask": mask.copy(),
                    "sig": _shape_signature_from_mask(mask),
                }
                catalog.setdefault(key, []).append(record)
                all_records.append(record)
    return catalog, all_records


SHAPE_CATALOG, ALL_SHAPES = _build_shape_catalog()
TEMPLATE_KEYS = [
    key
    for key, records in SHAPE_CATALOG.items()
    if 4 <= key[2] <= 6 and len(records) >= 2 and 2 <= key[1] <= 4
]
assert TEMPLATE_KEYS, "template catalog must not be empty"


def _pick_record(
    keys: list[tuple[int, int, int]] | None = None,
    *,
    exclude_sig: frozenset[tuple[int, int]] | None = None,
    used_sigs: set[frozenset[tuple[int, int]]] | None = None,
) -> dict[str, object]:
    if keys is None:
        pools = ALL_SHAPES
    else:
        pools = [record for key in keys for record in SHAPE_CATALOG[key]]

    filtered = [
        record
        for record in pools
        if (exclude_sig is None or record["sig"] != exclude_sig)
        and (used_sigs is None or record["sig"] not in used_sigs)
    ]
    if filtered:
        return random.choice(filtered)

    relaxed = [record for record in pools if exclude_sig is None or record["sig"] != exclude_sig]
    if not relaxed:
        raise RuntimeError("no suitable shape record found")
    return random.choice(relaxed)


def _pick_nonmatching_record(
    template_key: tuple[int, int, int],
    template_sig: frozenset[tuple[int, int]],
    used_sigs: set[frozenset[tuple[int, int]]],
) -> dict[str, object]:
    template_h, template_w, template_area = template_key
    strategies = [
        [template_key],
        [key for key in SHAPE_CATALOG if key[1] == template_w and key[2] == template_area],
        [key for key in SHAPE_CATALOG if key[1] == template_w],
        [key for key in SHAPE_CATALOG if key[2] == template_area],
        list(SHAPE_CATALOG),
    ]
    for keys in strategies:
        try:
            return _pick_record(keys, exclude_sig=template_sig, used_sigs=used_sigs)
        except RuntimeError:
            continue
    raise RuntimeError("failed to sample a nonmatching shape")


def _sample_palette_columns(width: int) -> np.ndarray:
    while True:
        cols = np.array([random.choice(PALETTE_COLORS) for _ in range(width)], dtype=int)
        if len(set(map(int, cols))) >= 2:
            return cols


def _make_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _parse_palette(g: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height = int(g.shape[0])
    bottom = g[height - 3 : height]
    palette_box = bbox(bottom != int(bg))
    assert palette_box is not None, "expected a palette strip in the bottom three rows"
    pr0, pc0, pr1, pc1 = palette_box
    assert (int(pr0), int(pr1)) == (0, 2), "palette must occupy all bottom-strip rows"

    palette = crop_bbox(bottom, palette_box)
    assert palette.shape[0] == 3
    assert np.array_equal(palette[0], palette[1]), "top two palette rows must match"
    assert len(set(map(int, palette[2]))) == 1 and int(palette[2, 0]) == PALETTE_BASE
    assert set(map(int, palette[0])) <= set(PALETTE_COLORS)
    return palette[0].copy(), (height - 3 + int(pr0), int(pc0), height - 3 + int(pr1), int(pc1))


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    palette_cols, palette_bbox = _parse_palette(g, bg)
    templates = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[TEMPLATE_COLOR],
        return_format="dict",
    )
    assert len(templates) == 1, "expected exactly one cyan template"
    template = templates[0]
    template["sig"] = _shape_signature_from_coords(template["coords"])
    template["height"] = int(template["bbox"][2] - template["bbox"][0] + 1)
    template["width"] = int(template["bbox"][3] - template["bbox"][1] + 1)
    assert int(template["width"]) == int(len(palette_cols))

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=list(OBJECT_COLORS),
        return_format="dict",
    )
    for obj in objects:
        obj["sig"] = _shape_signature_from_coords(obj["coords"])
        obj["height"] = int(obj["bbox"][2] - obj["bbox"][0] + 1)
        obj["width"] = int(obj["bbox"][3] - obj["bbox"][1] + 1)
    return {
        "grid": g,
        "bg": int(bg),
        "palette_cols": palette_cols,
        "palette_bbox": palette_bbox,
        "template": template,
        "objects": objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select every object whose normalized shape matches the cyan template, then recolor each matched object column-by-column from the palette strip.

    Input structure:
    - The grid has a black background with several red or blue 4-connected objects, one cyan template object, and a separate 3-row palette strip at the bottom.
    - The palette strip is a solid rectangle whose width equals the template bbox width; its first two rows are identical column colors and its third row is uniform grey.
    - Only the red and blue objects are candidate targets; the cyan template and the palette are reference objects.

    Rule:
    - Normalize the cyan template shape by translating its bbox upper-left corner to the origin.
    - Normalize each red or blue connected component in the same way and compare it to the template shape.
    - For every matching object, recolor only its foreground pixels so that bbox column k takes the palette's column-k color; nonmatching objects stay unchanged.
    - Remove the cyan template and the entire palette strip to black background in the output.

    Key insight:
    - Shape matching decides which objects are targets, while the palette independently decides the replacement color for each matched bbox column, so both references are required.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    palette_cols = scene["palette_cols"]
    palette_bbox = scene["palette_bbox"]
    template_sig = scene["template"]["sig"]

    out = g.copy()
    for obj in scene["objects"]:
        if obj["sig"] != template_sig:
            continue
        c0 = int(obj["bbox"][1])
        for r, c in obj["coords"]:
            out[int(r), int(c)] = int(palette_cols[int(c) - c0])

    for r, c in scene["template"]["coords"]:
        out[int(r), int(c)] = bg
    r0, c0, r1, c1 = map(int, palette_bbox)
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _build_object_specs() -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    template_key = random.choice(TEMPLATE_KEYS)
    template_record = random.choice(SHAPE_CATALOG[template_key])
    template_mask = np.asarray(template_record["mask"], dtype=bool).copy()
    template_sig = template_record["sig"]
    template_width = int(template_key[1])

    total_objects = random.randint(*TOTAL_OBJECT_RANGE)
    candidate_count = total_objects - 1
    matched_color = random.choice(OBJECT_COLORS)
    other_color = OBJECT_COLORS[0] if matched_color == OBJECT_COLORS[1] else OBJECT_COLORS[1]
    matched_color_count = random.randint(2, candidate_count - 1)
    other_color_count = candidate_count - matched_color_count
    match_count = random.randint(1, matched_color_count - 1)

    palette_cols = _sample_palette_columns(template_width)

    used_sigs: set[frozenset[tuple[int, int]]] = {template_sig}
    objects: list[dict[str, object]] = []

    same_color_decoy = _pick_record([template_key], exclude_sig=template_sig)
    objects.append(
        {
            "mask": np.asarray(same_color_decoy["mask"], dtype=bool).copy(),
            "color": int(matched_color),
            "match": False,
        }
    )
    used_sigs.add(same_color_decoy["sig"])

    for _ in range(match_count):
        objects.append({"mask": template_mask.copy(), "color": int(matched_color), "match": True})

    for _ in range(matched_color_count - match_count - 1):
        record = _pick_nonmatching_record(template_key, template_sig, used_sigs)
        objects.append(
            {
                "mask": np.asarray(record["mask"], dtype=bool).copy(),
                "color": int(matched_color),
                "match": False,
            }
        )
        used_sigs.add(record["sig"])

    other_color_decoy = _pick_record([template_key], exclude_sig=template_sig)
    objects.append(
        {
            "mask": np.asarray(other_color_decoy["mask"], dtype=bool).copy(),
            "color": int(other_color),
            "match": False,
        }
    )
    used_sigs.add(other_color_decoy["sig"])

    for _ in range(other_color_count - 1):
        record = _pick_nonmatching_record(template_key, template_sig, used_sigs)
        objects.append(
            {
                "mask": np.asarray(record["mask"], dtype=bool).copy(),
                "color": int(other_color),
                "match": False,
            }
        )
        used_sigs.add(record["sig"])

    random.shuffle(objects)
    template_spec = {"mask": template_mask, "color": TEMPLATE_COLOR, "match": False, "template": True}
    return template_spec, objects, palette_cols


def _place_specs(
    height: int,
    width: int,
    template_spec: dict[str, object],
    objects: list[dict[str, object]],
    palette_cols: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], int] | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    max_object_row = height - 4

    placed_specs = [dict(template_spec)] + [dict(obj) for obj in objects]
    random.shuffle(placed_specs)
    placed_specs.sort(key=lambda spec: int(np.asarray(spec["mask"]).size), reverse=True)

    for spec in placed_specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        mask_h, mask_w = map(int, mask.shape)
        choices: list[tuple[int, int]] = []
        for r0 in range(0, max_object_row - mask_h + 1):
            for c0 in range(0, width - mask_w + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + mask_h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + mask_w + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        top, left = random.choice(choices)
        spec["top"] = int(top)
        spec["left"] = int(left)
        paste(grid, _make_sprite(mask, int(spec["color"])), int(top), int(left), transparent=BG)

        rr0 = max(0, int(top) - 1)
        rr1 = min(height, int(top) + mask_h + 1)
        cc0 = max(0, int(left) - 1)
        cc1 = min(width, int(left) + mask_w + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    palette_left = random.randint(0, width - int(len(palette_cols)))
    palette_patch = np.vstack(
        [
            palette_cols,
            palette_cols,
            np.full(len(palette_cols), PALETTE_BASE, dtype=int),
        ]
    )
    paste(grid, palette_patch, height - 3, palette_left)
    return grid, placed_specs, palette_left


def _build_expected_output(
    grid: np.ndarray,
    placed_specs: list[dict[str, object]],
    palette_cols: np.ndarray,
    palette_left: int,
) -> np.ndarray:
    out = grid.copy()
    out[out == TEMPLATE_COLOR] = BG
    out[-3:, palette_left : palette_left + len(palette_cols)] = BG

    for spec in placed_specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        top = int(spec["top"])
        left = int(spec["left"])
        if spec.get("template"):
            continue
        if not bool(spec["match"]):
            continue
        for dr, dc in np.argwhere(mask):
            out[top + int(dr), left + int(dc)] = int(palette_cols[int(dc)])
    return out


def _validate_generated_scene(grid: np.ndarray, expected: np.ndarray) -> None:
    scene = _parse_scene(grid)
    template = scene["template"]
    objects = scene["objects"]
    palette_cols = scene["palette_cols"]
    out = main(grid)

    assert 5 <= len(objects) + 1 <= 8
    assert 4 <= int(template["area"]) <= 6
    assert 2 <= int(template["width"]) <= 4
    assert len(set(map(int, palette_cols))) >= 2

    by_color = {color: [] for color in OBJECT_COLORS}
    for obj in objects:
        assert 4 <= int(obj["area"]) <= 8
        assert 2 <= int(obj["width"]) <= 4
        by_color[int(obj["color"])].append(obj)

    assert all(by_color[color] for color in OBJECT_COLORS), "both red and blue must be present"

    template_sig = template["sig"]
    matched_colors = []
    for color, comps in by_color.items():
        flags = [comp["sig"] == template_sig for comp in comps]
        if 0 < sum(flags) < len(flags):
            matched_colors.append(color)
    assert len(matched_colors) == 1, "exactly one object color should be partially matched"

    matched_color = int(matched_colors[0])
    other_color = OBJECT_COLORS[0] if matched_color == OBJECT_COLORS[1] else OBJECT_COLORS[1]
    matched_group = by_color[matched_color]
    other_group = by_color[int(other_color)]
    assert any(comp["sig"] == template_sig for comp in matched_group)
    assert any(comp["sig"] != template_sig for comp in matched_group)
    assert all(comp["sig"] != template_sig for comp in other_group)

    template_key = (int(template["height"]), int(template["width"]), int(template["area"]))
    assert any(
        comp["sig"] != template_sig
        and int(comp["color"]) == matched_color
        and (int(comp["height"]), int(comp["width"]), int(comp["area"])) == template_key
        for comp in matched_group
    ), "need a same-color decoy with the same bbox and area as the template"

    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert np.array_equal(solved, out)
    assert not np.array_equal(grid, solved)

    for r, c in template["coords"]:
        assert int(solved[int(r), int(c)]) == BG

    pr0, pc0, pr1, pc1 = map(int, scene["palette_bbox"])
    assert np.all(solved[pr0 : pr1 + 1, pc0 : pc1 + 1] == BG)

    for obj in objects:
        c0 = int(obj["bbox"][1])
        colors_seen: set[int] = set()
        for r, c in obj["coords"]:
            rr = int(r)
            cc = int(c)
            if obj["sig"] == template_sig:
                target = int(palette_cols[cc - c0])
                assert int(solved[rr, cc]) == target
                colors_seen.add(target)
            else:
                assert int(solved[rr, cc]) == int(grid[rr, cc])
        if obj["sig"] == template_sig:
            assert len(colors_seen) >= 2, "matched objects must visibly use multiple palette columns"


def _generate_attempt() -> np.ndarray | None:
    template_spec, objects, palette_cols = _build_object_specs()
    min_width = max(int(len(palette_cols)) + 2, 10)

    for _ in range(12):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(max(min_width, GRID_W_RANGE[0]), GRID_W_RANGE[1])
        placed = _place_specs(height, width, template_spec, objects, palette_cols)
        if placed is None:
            continue
        grid, placed_specs, palette_left = placed
        expected = _build_expected_output(grid, placed_specs, palette_cols, palette_left)
        try:
            _validate_generated_scene(grid, expected)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
