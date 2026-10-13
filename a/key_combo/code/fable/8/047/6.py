from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


BG = 0
INPUT_COLOR = 5
FAMILY_TO_COLOR = {
    1: 1,
    2: 3,
    3: 2,
    4: 4,
}
_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _rotation_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _same_up_to_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return any(a.shape == np.rot90(b, k).shape and np.array_equal(a, np.rot90(b, k)) for k in range(4))


def _hole_count(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    bg_comps = _components(~padded)
    holes = 0
    h, w = padded.shape
    for coords in bg_comps:
        rr = coords[:, 0]
        cc = coords[:, 1]
        touches_edge = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_edge:
            holes += 1
    return holes


def _family_analysis(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int, bytes], list[int]]]:
    comps = _components(grid == INPUT_COLOR)
    data = []
    families: dict[tuple[int, int, bytes], list[int]] = defaultdict(list)
    for idx, coords in enumerate(comps):
        mask = _coords_to_mask(coords)
        key = _rotation_key(mask)
        item = {
            "coords": coords,
            "mask": mask,
            "holes": _hole_count(mask),
            "key": key,
            "mirror_key": _rotation_key(np.fliplr(mask)),
        }
        data.append(item)
        families[key].append(idx)
    return data, families


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object by the size of its rotation family, not by its hole count.

    Input structure:
    - The grid has a black background with several 4-connected gray objects.
    - Every gray object contains one to four enclosed black holes inside its own shape.
    - Objects may appear in different orientations, and mirrored shapes are possible distractors.

    Rule:
    - Group all gray objects by rotational equivalence: objects belong to the same family iff one can be rotated by 90, 180, or 270 degrees to match the other.
    - Recolor each whole family by its number of members: 1 -> blue, 2 -> green, 3 -> red, 4 -> yellow.
    - Keep the black background and all internal black holes unchanged.

    Key insight:
    - Hole count is only a per-object distraction; the true class key is the cross-object rotation family size, and mirror images do not count as the same family.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    fg_colors = sorted(set(map(int, np.unique(g))) - {BG})
    assert fg_colors == [INPUT_COLOR]

    data, families = _family_analysis(g)
    assert data

    out = g.copy()
    for family_key, member_indices in families.items():
        family_size = len(member_indices)
        assert family_size in FAMILY_TO_COLOR
        color = FAMILY_TO_COLOR[family_size]
        for idx in member_indices:
            coords = data[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = color
    return out


_HOLE_PATTERNS = {
    1: [((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),)],
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [((0, 0), (0, 1), (1, 0), (1, 1))],
}

_ARM_RECIPES = [
    (("top", "start"), ("right", "end"), ("bottom", "end")),
    (("left", "start"), ("top", "end"), ("right", "middle")),
    (("bottom", "start"), ("left", "end"), ("top", "middle")),
    (("right", "start"), ("bottom", "middle"), ("left", "end")),
]


def _biased_start(span: int, length: int, bias: str) -> int:
    max_start = span - length
    if max_start <= 0:
        return 0
    if max_start >= 2:
        low = 1
        high = max_start - 1
    else:
        low = 0
        high = max_start
    if low > high:
        low = 0
        high = max_start
    if bias == "start":
        return random.randint(low, min(high, low + 1))
    if bias == "end":
        return random.randint(max(low, high - 1), high)
    mid = (low + high) // 2
    return random.randint(max(low, mid - 1), min(high, mid + 1))


def _build_core_mask(hole_count: int) -> np.ndarray:
    cell_heights = [random.randint(1, 2), random.randint(1, 2)]
    cell_widths = [random.randint(1, 2), random.randint(1, 2)]
    h = cell_heights[0] + cell_heights[1] + 3
    w = cell_widths[0] + cell_widths[1] + 3
    mask = np.ones((h, w), dtype=bool)

    row_starts = [1, 2 + cell_heights[0]]
    col_starts = [1, 2 + cell_widths[0]]
    pattern = random.choice(_HOLE_PATTERNS[hole_count])
    for rr, cc in pattern:
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        mask[r0 : r0 + cell_heights[rr], c0 : c0 + cell_widths[cc]] = False
    return mask


def _add_arm(canvas: np.ndarray, base_h: int, base_w: int, edge: str, bias: str) -> None:
    base_r0 = 2
    base_c0 = 2
    if edge in {"top", "bottom"}:
        span = base_w
        length = random.randint(1, min(2, span))
        depth = random.randint(1, 2)
        start = _biased_start(span, length, bias)
        c0 = base_c0 + start
        c1 = c0 + length
        if edge == "top":
            canvas[base_r0 - depth : base_r0, c0:c1] = True
        else:
            canvas[base_r0 + base_h : base_r0 + base_h + depth, c0:c1] = True
    else:
        span = base_h
        length = random.randint(1, min(2, span))
        depth = random.randint(1, 2)
        start = _biased_start(span, length, bias)
        r0 = base_r0 + start
        r1 = r0 + length
        if edge == "left":
            canvas[r0:r1, base_c0 - depth : base_c0] = True
        else:
            canvas[r0:r1, base_c0 + base_w : base_c0 + base_w + depth] = True


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _generate_chiral_shape(hole_count: int) -> np.ndarray:
    for _ in range(200):
        core = _build_core_mask(hole_count)
        h, w = core.shape
        canvas = np.zeros((h + 4, w + 4), dtype=bool)
        canvas[2 : 2 + h, 2 : 2 + w] = core
        for edge, bias in random.choice(_ARM_RECIPES):
            _add_arm(canvas, h, w, edge, bias)
        mask = _crop_mask(canvas)
        if len(_components(mask)) != 1:
            continue
        if _hole_count(mask) != hole_count:
            continue
        if len({_mask_signature(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        if _same_up_to_rotation(mask, np.fliplr(mask)):
            continue
        if max(mask.shape) > 9 or int(mask.sum()) < 12:
            continue
        return mask
    raise RuntimeError(f"failed to generate chiral {hole_count}-hole shape")


def _shape_to_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = INPUT_COLOR
    return patch


def _make_family_patches(base_mask: np.ndarray, family_size: int) -> list[np.ndarray]:
    rotations = [np.rot90(base_mask, k) for k in range(4)]
    assert len({_mask_signature(rot) for rot in rotations}) == 4
    chosen = random.sample(range(4), family_size)
    return [_shape_to_patch(rotations[k]) for k in chosen]


def _family_specs_for_mode(mode: int) -> list[dict]:
    if mode == 0:
        return [
            {"size": 1, "holes": 1, "mirror_group": None},
            {"size": 2, "holes": 2, "mirror_group": None},
            {"size": 3, "holes": 3, "mirror_group": None},
        ]
    if mode == 1:
        shared_holes = random.choice([2, 4])
        extra_holes = random.choice([1, 3]) if shared_holes in {2, 4} else 4
        return [
            {"size": 3, "holes": shared_holes, "mirror_group": "A"},
            {"size": 1, "holes": shared_holes, "mirror_group": "A_mirror"},
            {"size": 2, "holes": extra_holes, "mirror_group": None},
        ]
    return [
        {"size": 4, "holes": 3, "mirror_group": "B"},
        {"size": 2, "holes": 3, "mirror_group": "B_mirror"},
        {"size": 1, "holes": 4, "mirror_group": None},
    ]


def _build_family_masks(mode: int) -> list[dict]:
    specs = _family_specs_for_mode(mode)
    used_keys: set[tuple[int, int, bytes]] = set()
    built: dict[str, np.ndarray] = {}
    families: list[dict] = []

    for spec in specs:
        group = spec["mirror_group"]
        if group and group.endswith("_mirror"):
            base = built[group[:-7]]
            mask = np.fliplr(base)
            if _rotation_key(mask) in used_keys:
                raise RuntimeError("mirror family collided with an existing rotation family")
        else:
            for _ in range(200):
                mask = _generate_chiral_shape(spec["holes"])
                key = _rotation_key(mask)
                if key in used_keys:
                    continue
                if group and _rotation_key(np.fliplr(mask)) == key:
                    continue
                break
            else:
                raise RuntimeError("failed to build distinct family mask")
            if group:
                built[group] = mask
        key = _rotation_key(mask)
        if key in used_keys:
            raise RuntimeError("duplicate rotation family")
        used_keys.add(key)
        families.append({"mask": mask, **spec})
    return families


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    gap = 2
    border = 1
    areas = [int(np.count_nonzero(patch)) for patch in patches]
    order = list(range(len(patches)))
    order.sort(key=lambda idx: (-areas[idx], -patches[idx].shape[0], -patches[idx].shape[1], random.random()))

    for _ in range(200):
        num_rows = 3 if len(patches) >= 5 else 2
        rows: list[list[int]] = [[] for _ in range(num_rows)]

        order = list(range(len(patches)))
        order.sort(key=lambda idx: (-areas[idx], random.random()))
        row_order = list(range(num_rows))
        random.shuffle(row_order)
        for pos, idx in enumerate(order):
            row_idx = row_order[pos % num_rows]
            rows[row_idx].append(idx)

        packed_rows = []
        for row in rows:
            if not row:
                continue
            row_width = sum(patches[idx].shape[1] for idx in row) + gap * (len(row) - 1)
            row_height = max(patches[idx].shape[0] for idx in row)
            packed_rows.append((row, row_width, row_height))
        if len(packed_rows) < 2:
            continue

        inner_w = max(width for _, width, _ in packed_rows)
        inner_h = sum(height for _, _, height in packed_rows) + gap * (len(packed_rows) - 1)
        if inner_w + 2 * border > 30 or inner_h + 2 * border > 30:
            continue

        h = random.randint(max(inner_h + 2 * border, 18), 30)
        w = random.randint(max(inner_w + 2 * border, 18), 30)
        grid = np.zeros((h, w), dtype=int)
        y = random.randint(border, h - inner_h - border)

        for row, row_width, row_height in packed_rows:
            row_patches = row[:]
            random.shuffle(row_patches)
            row_width = sum(patches[idx].shape[1] for idx in row_patches) + gap * (len(row_patches) - 1)
            x = random.randint(border, w - row_width - border)
            for idx in row_patches:
                patch = patches[idx]
                ph, pw = patch.shape
                mask = patch != BG
                grid[y : y + ph, x : x + pw][mask] = patch[mask]
                x += pw + gap
            y += row_height + gap
        return grid
    return None


def _render_scene(mode: int) -> np.ndarray | None:
    families = _build_family_masks(mode)
    patches: list[np.ndarray] = []
    for family in families:
        patches.extend(_make_family_patches(family["mask"], family["size"]))
    random.shuffle(patches)
    return _place_patches(patches)


def _validate_scene(grid: np.ndarray, mode: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, INPUT_COLOR}:
        return False

    data, families = _family_analysis(grid)
    if not data:
        return False

    sizes = sorted(len(indices) for indices in families.values())
    expected_sizes = {0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 4]}[mode]
    if sizes != expected_sizes:
        return False

    family_size_by_key = {key: len(indices) for key, indices in families.items()}
    family_holes = {}
    reps = {}
    for item in data:
        if not 1 <= item["holes"] <= 4:
            return False
        reps.setdefault(item["key"], item["mask"])
        hole_value = family_holes.setdefault(item["key"], item["holes"])
        if hole_value != item["holes"]:
            return False

    output = main(grid)
    if output.shape != grid.shape:
        return False
    if np.any(output[grid == BG] != BG):
        return False

    object_colors = set()
    for item in data:
        coords = item["coords"]
        family_size = family_size_by_key[item["key"]]
        expected_color = FAMILY_TO_COLOR[family_size]
        vals = set(map(int, output[coords[:, 0], coords[:, 1]]))
        if vals != {expected_color}:
            return False
        object_colors.add(expected_color)
    if len(object_colors) != len(expected_sizes):
        return False

    if mode == 0:
        if any(item["holes"] != family_size_by_key[item["key"]] for item in data):
            return False
    else:
        if all(item["holes"] == family_size_by_key[item["key"]] for item in data):
            return False
        families_by_holes: dict[int, set[int]] = defaultdict(set)
        for key, holes in family_holes.items():
            families_by_holes[holes].add(family_size_by_key[key])
        if not any(len(size_set) >= 2 for size_set in families_by_holes.values()):
            return False

        family_keys = list(families.keys())
        mirror_pair_found = False
        for key_a in family_keys:
            for key_b in family_keys:
                if key_a == key_b:
                    continue
                if _rotation_key(np.fliplr(reps[key_a])) == key_b and family_size_by_key[key_a] != family_size_by_key[key_b]:
                    mirror_pair_found = True
                    break
            if mirror_pair_found:
                break
        if not mirror_pair_found:
            return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(300):
        grid = _render_scene(mode)
        if grid is not None and _validate_scene(grid, mode):
            return grid
    raise RuntimeError(f"failed to generate a valid scene for mode {mode}")
