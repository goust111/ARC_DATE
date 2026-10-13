from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    try:
        import arc_utils as arc  # type: ignore

        return arc
    except Exception:
        arc_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        return SourceFileLoader("_arc_utils_shared", str(arc_path)).load_module()


_arc = _load_arc_utils()
Color = _arc.Color
crop_bbox = _arc.crop_bbox
find_components = _arc.find_components
get_bg = _arc.get_bg
paste = _arc.paste
scale_sprite = _arc.scale_sprite

BG = Color.BLACK
GREY = Color.GREY
FRAME_COLOR = Color.MAROON
NON_GREY_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
)
CANVAS_SIZE = 30
GEN_RETRY_LIMIT = 500


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = np.asarray(comp["coords"], dtype=int) - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _outline_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_outline_rectangle(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    return h >= 3 and w >= 3 and np.array_equal(mask, _outline_mask(mask.shape))


def _inside_open_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, bool]:
    r0, c0, r1, c1 = map(int, bbox)
    inside = [(r0 < int(r) < r1) and (c0 < int(c) < c1) for r, c in coords]
    return all(inside), any(inside)


def _divisors_desc(n: int) -> list[int]:
    return [d for d in range(int(n), 1, -1) if int(n) % d == 0]


def _primitive_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    for k in _divisors_desc(int(np.gcd(h, w))):
        blocks = mask.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3)
        if np.all(blocks == blocks[:, :, :1, :1]):
            return blocks[:, :, 0, 0].copy()
    return mask.copy()


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(mask, dtype=bool)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        variant = np.fliplr(base) if flipped else base
        for k in range(4):
            transformed = np.rot90(variant, k)
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(transformed.copy())
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    primitive = _primitive_mask(mask)
    keys = [
        (tuple(map(int, variant.shape)), variant.tobytes())
        for variant in _dihedral_variants(primitive)
    ]
    return min(keys)


def _shape_key_for_component(comp: dict) -> tuple[tuple[int, int], bytes]:
    return _canonical_shape_key(_component_mask(comp))


def _frame_partition(components: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    candidates = []
    for frame in components:
        if not _is_outline_rectangle(_component_mask(frame)):
            continue
        samples: list[dict] = []
        objects: list[dict] = []
        valid = True
        for comp in components:
            if comp is frame:
                continue
            all_inside, any_inside = _inside_open_bbox(np.asarray(comp["coords"], dtype=int), frame["bbox"])
            if any_inside and not all_inside:
                valid = False
                break
            if all_inside:
                samples.append(comp)
            else:
                objects.append(comp)
        if valid and samples and len(samples) == len(objects):
            candidates.append((frame, samples, objects))
    assert len(candidates) == 1, "expected one valid legend frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match legend samples to scaled outer objects and fill in the two missing colors.

    Input structure:
    - A single hollow rectangular frame encloses several small monochrome sample shapes.
    - Outside that frame sit the same number of larger monochrome objects on a plain background.
    - Each outer object is one sample after an integer k×k block scaling and a rotation or reflection.
    - Exactly one sample and exactly one outer object are grey, and those two grey shapes are different.

    Rule:
    - Detect the frame, then split the remaining 8-connected monochrome components into inner samples and outer objects by whether they lie strictly inside the frame.
    - Normalize every component by collapsing any repeated k×k block scaling and canonicalizing its mask over all rotations and reflections.
    - Use those normalized shape keys to match each outer object with its unique sample.
    - Recolor the grey outer object with its matched sample color, and recolor the grey sample with the color of its matched outer object.

    Key insight:
    - The frame is a bidirectional shape dictionary: colors transfer only along unique shape matches, and those matches become visible only after undoing both block scaling and orientation changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert components, "expected a framed legend and outer objects"

    frame, samples, objects = _frame_partition(components)
    assert len(samples) == len(objects) >= 2, "expected matching sample/object counts"

    sample_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    object_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}

    for sample in samples:
        key = _shape_key_for_component(sample)
        assert key not in sample_by_key, "sample shapes must be unique up to scaling and dihedral symmetry"
        sample_by_key[key] = sample

    for obj in objects:
        key = _shape_key_for_component(obj)
        assert key not in object_by_key, "outer object shapes must be unique up to scaling and dihedral symmetry"
        object_by_key[key] = obj

    assert set(sample_by_key) == set(object_by_key), "samples and outer objects must use the same shape set"

    grey_samples = [comp for comp in samples if int(comp["color"]) == GREY]
    grey_objects = [comp for comp in objects if int(comp["color"]) == GREY]
    assert len(grey_samples) == 1 and len(grey_objects) == 1, "expected one grey sample and one grey outer object"

    grey_sample = grey_samples[0]
    grey_object = grey_objects[0]
    grey_sample_key = _shape_key_for_component(grey_sample)
    grey_object_key = _shape_key_for_component(grey_object)
    assert grey_sample_key != grey_object_key, "the two grey members must not correspond"

    sample_color_for_grey_object = int(sample_by_key[grey_object_key]["color"])
    object_color_for_grey_sample = int(object_by_key[grey_sample_key]["color"])
    assert sample_color_for_grey_object != GREY
    assert object_color_for_grey_sample != GREY

    out = g.copy()
    grey_object_coords = np.asarray(grey_object["coords"], dtype=int)
    grey_sample_coords = np.asarray(grey_sample["coords"], dtype=int)
    out[grey_object_coords[:, 0], grey_object_coords[:, 1]] = sample_color_for_grey_object
    out[grey_sample_coords[:, 0], grey_sample_coords[:, 1]] = object_color_for_grey_sample
    return out


_RAW_SHAPES = (
    _mask_from_rows("11", "10"),
    _mask_from_rows("111", "010"),
    _mask_from_rows("100", "111"),
    _mask_from_rows("110", "011"),
    _mask_from_rows("11", "11", "10"),
    _mask_from_rows("101", "111"),
    _mask_from_rows("011", "110", "010"),
    _mask_from_rows("100", "110", "011"),
    _mask_from_rows("111", "010", "010"),
    _mask_from_rows("100", "100", "111"),
    _mask_from_rows("010", "111", "010"),
    _mask_from_rows("1111", "0010"),
)


def _build_shape_catalog() -> tuple[np.ndarray, ...]:
    catalog: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mask in _RAW_SHAPES:
        key = _canonical_shape_key(mask)
        if key not in seen:
            seen.add(key)
            catalog.append(mask.copy())
    return tuple(catalog)


SHAPE_CATALOG = _build_shape_catalog()


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(np.asarray(mask, dtype=bool), random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    r0, c0, r1, c1 = map(int, box)
    h, w = map(int, patch.shape)
    mask = patch != BG
    if h > r1 - r0 + 1 or w > c1 - c0 + 1:
        return None

    blocked = _dilate8(occupied)
    positions = [
        (rr, cc)
        for rr in range(r0, r1 - h + 2)
        for cc in range(c0, c1 - w + 2)
    ]
    random.shuffle(positions)

    for rr, cc in positions:
        if np.any(blocked[rr : rr + h, cc : cc + w] & mask):
            continue
        paste(grid, patch, rr, cc, transparent=BG)
        occupied[rr : rr + h, cc : cc + w] |= mask
        return np.argwhere(mask) + np.array([rr, cc])
    return None


def _sample_layout() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    side = random.choice(("left", "right", "top", "bottom"))
    if side in ("left", "right"):
        frame_h = random.randint(14, 18)
        frame_w = random.randint(9, 12)
        frame_r0 = random.randint(1, CANVAS_SIZE - frame_h - 1)
        if side == "left":
            frame_c0 = random.randint(1, 3)
            field_box = (1, frame_c0 + frame_w + 3, CANVAS_SIZE - 2, CANVAS_SIZE - 2)
        else:
            frame_c0 = random.randint(CANVAS_SIZE - frame_w - 4, CANVAS_SIZE - frame_w - 2)
            field_box = (1, 1, CANVAS_SIZE - 2, frame_c0 - 3)
    else:
        frame_h = random.randint(9, 12)
        frame_w = random.randint(14, 18)
        frame_c0 = random.randint(1, CANVAS_SIZE - frame_w - 1)
        if side == "top":
            frame_r0 = random.randint(1, 3)
            field_box = (frame_r0 + frame_h + 3, 1, CANVAS_SIZE - 2, CANVAS_SIZE - 2)
        else:
            frame_r0 = random.randint(CANVAS_SIZE - frame_h - 4, CANVAS_SIZE - frame_h - 2)
            field_box = (1, 1, frame_r0 - 3, CANVAS_SIZE - 2)

    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    sample_box = (frame_r0 + 1, frame_c0 + 1, frame_r0 + frame_h - 2, frame_c0 + frame_w - 2)

    if field_box[0] > field_box[2] or field_box[1] > field_box[3]:
        return None
    return frame_bbox, sample_box, field_box


def _assign_colors(count: int, grey_index: int) -> list[int]:
    chosen = random.sample(NON_GREY_COLORS, count - 1)
    colors: list[int] = []
    cursor = 0
    for idx in range(count):
        if idx == grey_index:
            colors.append(GREY)
        else:
            colors.append(int(chosen[cursor]))
            cursor += 1
    return colors


def _crop_with_margin(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - random.randint(1, 3))
    c0 = max(0, int(c0) - random.randint(1, 3))
    r1 = min(inp.shape[0] - 1, int(r1) + random.randint(1, 3))
    c1 = min(inp.shape[1] - 1, int(c1) + random.randint(1, 3))
    bbox = (r0, c0, r1, c1)
    return crop_bbox(inp, bbox), crop_bbox(out, bbox)


def _attempt_generate() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    frame_bbox, sample_box, field_box = layout
    field_h = field_box[2] - field_box[0] + 1
    field_w = field_box[3] - field_box[1] + 1

    count = random.randint(3, 5)
    sample_grey_index = random.randrange(count)
    object_grey_index = random.choice([idx for idx in range(count) if idx != sample_grey_index])
    sample_colors = _assign_colors(count, sample_grey_index)
    object_colors = _assign_colors(count, object_grey_index)

    base_shapes = random.sample(list(SHAPE_CATALOG), count)
    dim_cap = {3: 12, 4: 10, 5: 8}[count]

    entries: list[dict] = []
    for idx, base in enumerate(base_shapes):
        sample_mask = _random_dihedral(base)
        object_seed = _random_dihedral(base)
        feasible_scales = [
            k
            for k in (2, 3, 4)
            if object_seed.shape[0] * k <= field_h
            and object_seed.shape[1] * k <= field_w
            and max(object_seed.shape) * k <= dim_cap
        ]
        if not feasible_scales:
            return None
        scale = random.choice(feasible_scales)
        object_mask = scale_sprite(object_seed.astype(int), scale).astype(bool)
        entries.append(
            {
                "idx": idx,
                "sample_patch": _paint_patch(sample_mask, sample_colors[idx]),
                "object_patch": _paint_patch(object_mask, object_colors[idx]),
                "sample_color": int(sample_colors[idx]),
                "object_color": int(object_colors[idx]),
            }
        )

    scene = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    fr0, fc0, fr1, fc1 = frame_bbox
    frame_patch = _paint_patch(_outline_mask((fr1 - fr0 + 1, fc1 - fc0 + 1)), FRAME_COLOR)
    paste(scene, frame_patch, fr0, fc0, transparent=BG)
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] |= frame_patch != BG

    for entry in sorted(entries, key=lambda item: item["sample_patch"].size, reverse=True):
        coords = _place_patch(scene, occupied, entry["sample_patch"], sample_box)
        if coords is None:
            return None
        entry["sample_coords"] = coords

    for entry in sorted(entries, key=lambda item: item["object_patch"].size, reverse=True):
        coords = _place_patch(scene, occupied, entry["object_patch"], field_box)
        if coords is None:
            return None
        entry["object_coords"] = coords

    expected = scene.copy()
    grey_object_entry = entries[object_grey_index]
    grey_sample_entry = entries[sample_grey_index]
    expected[
        grey_object_entry["object_coords"][:, 0],
        grey_object_entry["object_coords"][:, 1],
    ] = grey_object_entry["sample_color"]
    expected[
        grey_sample_entry["sample_coords"][:, 0],
        grey_sample_entry["sample_coords"][:, 1],
    ] = grey_sample_entry["object_color"]

    cropped_inp, cropped_out = _crop_with_margin(scene, expected)
    if int(get_bg(cropped_inp)) != BG:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-shape-dictionary input")
