from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    base = mask.astype(bool)
    for k in range(4):
        rot = np.rot90(base, k)
        for variant in (rot, np.fliplr(rot)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(variant) for variant in _dihedral_variants(mask))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, min_area=1)
    objects: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        mask, box = _tight_mask_from_coords(coords)
        r0, c0, r1, c1 = box
        objects.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": box,
                "mask": mask,
                "signature": _shape_signature(mask),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _group_objects(objects: list[dict]) -> list[list[int]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for idx, obj in enumerate(objects):
        groups.setdefault(obj["signature"], []).append(idx)
    return [groups[key] for key in sorted(groups, key=lambda sig: (len(groups[sig]), sig))]


def _segment_for_pair(a: dict, b: dict) -> tuple[str, list[tuple[int, int]]] | None:
    ar, ac = a["center"]
    br, bc = b["center"]
    if ar == br:
        left, right = (a, b) if a["bbox"][1] <= b["bbox"][1] else (b, a)
        start = int(left["bbox"][3])
        end = int(right["bbox"][1])
        return "horizontal", [(ar, c) for c in range(start, end + 1)]
    if ac == bc:
        top, bottom = (a, b) if a["bbox"][0] <= b["bbox"][0] else (b, a)
        start = int(top["bbox"][2])
        end = int(bottom["bbox"][0])
        return "vertical", [(r, ac) for r in range(start, end + 1)]
    return None


def _bbox_contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _line_specs(objects: list[dict]) -> list[dict]:
    groups = _group_objects(objects)
    specs: list[dict] = []
    for group in groups:
        if len(group) < 2:
            continue
        group_set = set(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                spec = _segment_for_pair(objects[group[i]], objects[group[j]])
                if spec is None:
                    continue
                orientation, coords = spec
                blocked = any(
                    any(_bbox_contains(objects[other]["bbox"], r, c) for r, c in coords)
                    for other in range(len(objects))
                    if other not in group_set
                )
                specs.append(
                    {
                        "group": tuple(group),
                        "orientation": orientation,
                        "coords": coords,
                        "blocked": blocked,
                    }
                )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group dihedrally equivalent sprites, connect the aligned pairs, and recolor repeated groups.

    Input structure:
    - The grid has a black background with several small 4-connected monochrome sprites separated from each other.
    - Each sprite is considered through its tight bounding-box mask, ignoring its absolute position and original color.
    - Some sprites share the same mask up to rotation or reflection, while others are unique distractors.

    Rule:
    - Partition sprites into equivalence groups by exact mask equality under the 8 dihedral transforms.
    - Within each group, connect every pair whose bounding-box centers share a row with a horizontal blue segment, or share a column with a vertical red segment; if both happen, only draw the horizontal segment.
    - If such a segment passes through any sprite bounding box from a different group, recolor the whole segment yellow; sprite pixels themselves are never overwritten.
    - Recolor every sprite in groups of size at least 2 to green, and leave singleton-group sprites in their original colors.

    Key insight:
    - Shape equivalence is the grouping key, but lines appear only when equivalent sprites are also row- or column-aligned, and blocking depends on other groups' bounding boxes rather than on occupied pixels alone.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    groups = _group_objects(objects)
    bg = int(get_bg(g))

    out = g.copy()
    sprite_mask = g != bg

    for group in groups:
        if len(group) < 2:
            continue
        for idx in group:
            rr, cc = objects[idx]["coords"].T
            out[rr, cc] = Color.GREEN

    for group in groups:
        if len(group) < 2:
            continue
        group_set = set(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                spec = _segment_for_pair(objects[group[i]], objects[group[j]])
                if spec is None:
                    continue
                orientation, coords = spec
                color = Color.BLUE if orientation == "horizontal" else Color.RED
                blocked = any(
                    any(_bbox_contains(objects[other]["bbox"], r, c) for r, c in coords)
                    for other in range(len(objects))
                    if other not in group_set
                )
                if blocked:
                    color = Color.YELLOW
                for r, c in coords:
                    if not sprite_mask[r, c]:
                        out[r, c] = color

    return out


_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
]

_SHAPES_BY_SIGNATURE = {_shape_signature(shape): shape for shape in _BASE_SHAPES}
assert len(_SHAPES_BY_SIGNATURE) == len(_BASE_SHAPES)
_SHAPE_CATALOG = list(_SHAPES_BY_SIGNATURE.values())
_BLOCKER_CATALOG = [shape for shape in _SHAPE_CATALOG if shape.shape != (1, 3) and shape.shape != (1, 4)]

_INPUT_COLORS = [Color.BLUE, Color.RED, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE]
_GRID_RANGE = (17, 23)
_GEN_ATTEMPTS = 400


def _random_variant(mask: np.ndarray, *, need_row_gap: bool = False, need_col_gap: bool = False) -> np.ndarray | None:
    variants = _dihedral_variants(mask)
    random.shuffle(variants)
    for variant in variants:
        rr = (variant.shape[0] - 1) // 2
        cc = (variant.shape[1] - 1) // 2
        if need_row_gap and not np.any(~variant[rr]):
            continue
        if need_col_gap and not np.any(~variant[:, cc]):
            continue
        return variant.copy()
    return None


def _center_interval(length: int, size: int, border: int = 1) -> tuple[int, int]:
    left = (size - 1) // 2
    right = size - 1 - left
    return border + left, length - 1 - border - right


def _bbox_from_center(center: tuple[int, int], mask: np.ndarray) -> tuple[int, int, int, int]:
    cr, cc = center
    h, w = mask.shape
    r0 = int(cr - (h - 1) // 2)
    c0 = int(cc - (w - 1) // 2)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _bboxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_center(
    grid_shape: tuple[int, int],
    mask: np.ndarray,
    row_window: tuple[int, int],
    col_window: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = mask.shape
    row_lo, row_hi = _center_interval(grid_shape[0], h)
    col_lo, col_hi = _center_interval(grid_shape[1], w)
    lo_r = max(row_lo, row_window[0])
    hi_r = min(row_hi, row_window[1])
    lo_c = max(col_lo, col_window[0])
    hi_c = min(col_hi, col_window[1])
    if lo_r > hi_r or lo_c > hi_c:
        return None
    return random.randint(lo_r, hi_r), random.randint(lo_c, hi_c)


def _render(grid_shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][item["mask"]] = int(item["color"])
    return out


def _count_nonaligned_pairs(group_centers: list[tuple[int, int]]) -> int:
    total = 0
    for i in range(len(group_centers)):
        for j in range(i + 1, len(group_centers)):
            if group_centers[i][0] != group_centers[j][0] and group_centers[i][1] != group_centers[j][1]:
                total += 1
    return total


def _sample_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_RANGE)
    width = random.randint(*_GRID_RANGE)
    grid_shape = (height, width)

    group_shapes = random.sample(_SHAPE_CATALOG, 2)
    used_signatures = {_shape_signature(shape) for shape in group_shapes}
    singleton_shape = random.choice(
        [shape for shape in _BLOCKER_CATALOG if _shape_signature(shape) not in used_signatures]
    )
    block_mode = random.choice(["none", "horizontal", "vertical"])

    a_masks = [_random_variant(group_shapes[0]) for _ in range(3)]
    b_masks = [_random_variant(group_shapes[1]) for _ in range(2)]
    if any(mask is None for mask in a_masks + b_masks):
        return None
    if block_mode == "horizontal":
        s_mask = _random_variant(singleton_shape, need_row_gap=True)
    elif block_mode == "vertical":
        s_mask = _random_variant(singleton_shape, need_col_gap=True)
    else:
        s_mask = _random_variant(singleton_shape)
    if s_mask is None:
        return None

    placed: list[dict] = []

    def try_place(name: str, mask: np.ndarray, center: tuple[int, int]) -> bool:
        box = _bbox_from_center(center, mask)
        if box[0] < 0 or box[1] < 0 or box[2] >= height or box[3] >= width:
            return False
        if any(not _bboxes_separate(box, item["bbox"]) for item in placed):
            return False
        placed.append({"name": name, "mask": mask, "center": center, "bbox": box})
        return True

    for _ in range(200):
        placed.clear()

        row_a = random.randint(3, height // 2)
        left_center = _sample_center(grid_shape, a_masks[0], (row_a, row_a), (2, max(2, width // 2 - 3)))
        right_center = _sample_center(grid_shape, a_masks[1], (row_a, row_a), (width // 2 + 1, width - 3))
        if left_center is None or right_center is None:
            continue
        if left_center[1] >= right_center[1]:
            continue
        if not try_place("a0", a_masks[0], left_center):
            continue
        if not try_place("a1", a_masks[1], right_center):
            continue
        gap_h = placed[1]["bbox"][1] - placed[0]["bbox"][3] - 1
        if gap_h < 3:
            continue

        a2_center = _sample_center(
            grid_shape,
            a_masks[2],
            (height // 2, height - 3),
            (2, max(2, width // 2)),
        )
        if a2_center is None:
            continue
        if a2_center[0] == row_a or a2_center[1] in {left_center[1], right_center[1]}:
            continue
        if not try_place("a2", a_masks[2], a2_center):
            continue

        col_b = random.randint(max(width // 2 + 1, 4), width - 4)
        top_center = _sample_center(grid_shape, b_masks[0], (2, max(2, height // 2 - 1)), (col_b, col_b))
        bottom_center = _sample_center(grid_shape, b_masks[1], (height // 2 + 1, height - 3), (col_b, col_b))
        if top_center is None or bottom_center is None:
            continue
        if top_center[0] >= bottom_center[0]:
            continue
        if not try_place("b0", b_masks[0], top_center):
            continue
        if not try_place("b1", b_masks[1], bottom_center):
            continue
        gap_v = placed[4]["bbox"][0] - placed[3]["bbox"][2] - 1
        if gap_v < 3:
            continue

        if block_mode == "horizontal":
            blocker_center = _sample_center(
                grid_shape,
                s_mask,
                (row_a, row_a),
                (placed[0]["bbox"][3] + 2, placed[1]["bbox"][1] - 2),
            )
        elif block_mode == "vertical":
            blocker_center = _sample_center(
                grid_shape,
                s_mask,
                (placed[3]["bbox"][2] + 2, placed[4]["bbox"][0] - 2),
                (col_b, col_b),
            )
        else:
            blocker_center = _sample_center(
                grid_shape,
                s_mask,
                (height // 2, height - 3),
                (width // 2 + 1, width - 3),
            )
        if blocker_center is None:
            continue
        if not try_place("s0", s_mask, blocker_center):
            continue

        colors = _INPUT_COLORS.copy()
        random.shuffle(colors)
        for item, color in zip(placed, colors):
            item["color"] = int(color)

        grid = _render(grid_shape, placed)
        objects = _extract_objects(grid)
        sizes = sorted(len(group) for group in _group_objects(objects))
        if sizes != [1, 2, 3]:
            continue

        target_group = max(_group_objects(objects), key=len)
        target_centers = [objects[idx]["center"] for idx in target_group]
        if _count_nonaligned_pairs(target_centers) == 0:
            continue

        specs = _line_specs(objects)
        if len(specs) != 2:
            continue
        horizontal = [spec for spec in specs if spec["orientation"] == "horizontal"]
        vertical = [spec for spec in specs if spec["orientation"] == "vertical"]
        if len(horizontal) != 1 or len(vertical) != 1:
            continue
        if block_mode == "none" and (horizontal[0]["blocked"] or vertical[0]["blocked"]):
            continue
        if block_mode == "horizontal" and (not horizontal[0]["blocked"] or vertical[0]["blocked"]):
            continue
        if block_mode == "vertical" and (horizontal[0]["blocked"] or not vertical[0]["blocked"]):
            continue

        out = main(grid)
        if out.shape != grid.shape or max(out.shape) > 30:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _sample_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
