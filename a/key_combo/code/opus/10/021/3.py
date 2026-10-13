from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LARGE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    origin = coords.min(axis=0)
    rel = coords - origin
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def _component_crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != BLACK
    seen = np.zeros_like(mask, dtype=bool)
    H, W = g.shape
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            bbox = _bbox_from_coords(coords)
            crop = _component_crop(g, bbox)
            colors = {int(v) for v in crop[crop != BLACK]}
            comps.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "crop": crop,
                    "colors": colors,
                    "area": int(coords.shape[0]),
                    "shape_key": _shape_key(coords),
                }
            )
    return comps


def _partition_components(components: list[dict]) -> tuple[list[dict], list[dict]]:
    smalls = [comp for comp in components if comp["colors"] == {GREY}]
    larges = [comp for comp in components if comp["colors"] != {GREY}]
    assert smalls and larges, "expected both grey targets and multicolor source shapes"
    assert len(smalls) == len(larges), "expected one small target per large source"
    for comp in larges:
        assert len(comp["colors"]) >= 2, "large source shapes must be multicolor"
    return larges, smalls


def _pair_components(components: list[dict]) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
    larges, smalls = _partition_components(components)
    small_by_key: dict[tuple[tuple[int, int], ...], dict] = {}
    for comp in smalls:
        key = comp["shape_key"]
        assert key not in small_by_key, "small target shapes must be unique"
        small_by_key[key] = comp

    pairs: list[tuple[dict, dict]] = []
    seen_large_keys: set[tuple[tuple[int, int], ...]] = set()
    for comp in larges:
        key = comp["shape_key"]
        assert key not in seen_large_keys, "large source shapes must be unique"
        seen_large_keys.add(key)
        assert key in small_by_key, "each large source must match one grey target by normalized shape"
        pairs.append((comp, small_by_key[key]))

    assert len(pairs) == len(smalls)
    return larges, smalls, pairs


def _should_mirror(src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]) -> bool:
    return (src_bbox[1] + src_bbox[3]) < (dst_bbox[1] + dst_bbox[3])


def _paste_nonblack(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _render_pairs(
    shape_order: list[dict],
    target_order: list[dict],
    mapping: list[int],
    shape_order_indices: list[int] | None = None,
    mirror_override: bool | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    assert shape is not None
    out = np.zeros(shape, dtype=int)
    if shape_order_indices is None:
        shape_order_indices = list(range(len(shape_order)))
    for large_idx in sorted(shape_order_indices, key=lambda idx: (shape_order[idx]["bbox"][0], shape_order[idx]["bbox"][1])):
        large = shape_order[large_idx]
        small = target_order[mapping[large_idx]]
        patch = large["crop"]
        mirrored = _should_mirror(large["bbox"], small["bbox"]) if mirror_override is None else bool(mirror_override)
        if mirrored:
            patch = np.fliplr(patch)
        if small["bbox"][0] + patch.shape[0] > shape[0] or small["bbox"][1] + patch.shape[1] > shape[1]:
            return None
        _paste_nonblack(out, patch, small["bbox"][0], small["bbox"][1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each multicolor source shape with the grey target of the same normalized mask, then stamp the source onto the target position with mirror determined by their horizontal relation.

    Input structure:
    - The grid has a black background with several 4-connected multicolor source shapes and the same number of 4-connected grey target shapes.
    - Every source shape and its matching grey target share the same normalized occupancy mask inside their tight bounding boxes.
    - Source and target objects are separated from one another, and source shapes use 2-3 non-black colors while targets are grey.

    Rule:
    - Split the non-black pixels into 4-connected components, treating grey monochrome components as targets and the multicolor components as sources.
    - Match each source to the unique grey target whose normalized occupancy mask is identical.
    - If the source bbox center column lies left of its matched target bbox center column, mirror the source left-right; otherwise keep it unchanged.
    - Clear the original objects and stamp each resulting source crop with its top-left corner anchored at the matched target's top-left corner.
    - When stamped non-black pixels overlap, draw sources in ascending order of source bbox top-left, letting later stamps overwrite earlier ones.

    Key insight:
    - The grey objects simultaneously encode which source shape to move and where to place it; only after shape matching can the source-target left/right relation decide whether the moved sprite must be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    components = _extract_components(g)
    _larges, _smalls, pairs = _pair_components(components)

    out = np.zeros_like(g)
    for large, small in sorted(pairs, key=lambda pair: (pair[0]["bbox"][0], pair[0]["bbox"][1])):
        patch = large["crop"]
        if _should_mirror(large["bbox"], small["bbox"]):
            patch = np.fliplr(patch)
        _paste_nonblack(out, patch, small["bbox"][0], small["bbox"][1])
    return out


def _normalize_shape(shape: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(sorted(shape), dtype=int)
    arr -= arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _build_shape_library() -> list[tuple[tuple[int, int], ...]]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        shapes_by_size[size] = set()
        for shape in shapes_by_size[size - 1]:
            cells = set(shape)
            for r, c in list(cells):
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    shapes_by_size[size].add(_normalize_shape(cells | {(nr, nc)}))

    lib: list[tuple[tuple[int, int], ...]] = []
    for size in range(4, 7):
        for shape in shapes_by_size[size]:
            rows = [r for r, _ in shape]
            cols = [c for _, c in shape]
            h = max(rows) + 1
            w = max(cols) + 1
            if 2 <= h <= 6 and 2 <= w <= 6 and max(h, w) >= 3:
                lib.append(shape)
    lib.sort(key=lambda shape: (len(shape), shape))
    return lib


SHAPE_LIBRARY = _build_shape_library()


def _patch_from_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _make_large_patch(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = list(shape)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    for _ in range(200):
        n_colors = random.choice([2, 3])
        palette = random.sample(LARGE_COLORS, n_colors)
        repeated = palette[:]
        while len(repeated) < len(coords):
            repeated.append(random.choice(palette))
        random.shuffle(repeated)
        patch = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(coords, repeated):
            patch[r, c] = int(color)
        if len({int(v) for v in patch[patch != BLACK]}) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to make an asymmetric multicolor source patch")


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _can_place_box(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    return all(_boxes_clear(box, other) for other in boxes)


def _all_positions(H: int, W: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]


def _bbox_center_key(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[0] + box[2], box[1] + box[3]


def _sorted_order_mapping(larges: list[dict], smalls: list[dict]) -> list[int]:
    large_order = sorted(range(len(larges)), key=lambda i: (larges[i]["bbox"][0], larges[i]["bbox"][1]))
    small_order = sorted(range(len(smalls)), key=lambda i: (smalls[i]["bbox"][0], smalls[i]["bbox"][1]))
    mapping = [0] * len(larges)
    for li, si in zip(large_order, small_order):
        mapping[li] = si
    return mapping


def _nearest_mapping(larges: list[dict], smalls: list[dict]) -> list[int]:
    remaining = set(range(len(smalls)))
    mapping = [0] * len(larges)
    for li in sorted(range(len(larges)), key=lambda i: (larges[i]["bbox"][0], larges[i]["bbox"][1])):
        src = _bbox_center_key(larges[li]["bbox"])
        choices = []
        for si in remaining:
            dst = _bbox_center_key(smalls[si]["bbox"])
            dist = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
            choices.append((dist, smalls[si]["bbox"][0], smalls[si]["bbox"][1], si))
        choices.sort()
        pick = choices[0][3]
        mapping[li] = pick
        remaining.remove(pick)
    return mapping


def _correct_mapping(larges: list[dict], smalls: list[dict]) -> list[int]:
    small_index_by_key = {small["shape_key"]: idx for idx, small in enumerate(smalls)}
    return [small_index_by_key[large["shape_key"]] for large in larges]


def _semantic_validate(grid: np.ndarray) -> bool:
    try:
        components = _extract_components(grid)
        larges, smalls, pairs = _pair_components(components)
        if len(pairs) < 3:
            return False

        mirror_flags = [_should_mirror(large["bbox"], small["bbox"]) for large, small in pairs]
        if not any(mirror_flags) or all(mirror_flags):
            return False
        if any(np.array_equal(large["crop"], np.fliplr(large["crop"])) for large, _small in pairs):
            return False

        correct = main(grid)
        mapping = _correct_mapping(larges, smalls)

        wrong = _render_pairs(larges, smalls, mapping, shape=grid.shape, mirror_override=False)
        if wrong is not None and np.array_equal(wrong, correct):
            return False
        wrong = _render_pairs(larges, smalls, mapping, shape=grid.shape, mirror_override=True)
        if wrong is not None and np.array_equal(wrong, correct):
            return False
        wrong = _render_pairs(larges, smalls, _sorted_order_mapping(larges, smalls), shape=grid.shape)
        if wrong is not None and np.array_equal(wrong, correct):
            return False
        wrong = _render_pairs(larges, smalls, _nearest_mapping(larges, smalls), shape=grid.shape)
        if wrong is not None and np.array_equal(wrong, correct):
            return False

        return int(correct.sum()) > 0
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 26)
    W = random.randint(18, 26)
    pair_count = random.randint(3, 4)
    shapes = random.sample(SHAPE_LIBRARY, pair_count)
    branches = [True, False] + [random.choice([True, False]) for _ in range(pair_count - 2)]
    random.shuffle(branches)

    records = []
    for shape, mirror_when_moved in zip(shapes, branches):
        large_patch = _make_large_patch(shape)
        small_patch = _patch_from_shape(shape, GREY)
        records.append(
            {
                "shape": shape,
                "large_patch": large_patch,
                "small_patch": small_patch,
                "mirror_branch": bool(mirror_when_moved),
            }
        )

    placed_boxes: list[tuple[int, int, int, int]] = []
    random.shuffle(records)
    for rec in records:
        h, w = rec["large_patch"].shape
        positions = _all_positions(H, W, h, w)
        random.shuffle(positions)
        success = False
        for sr, sc in positions:
            small_box = (sr, sc, sr + h - 1, sc + w - 1)
            if not _can_place_box(small_box, placed_boxes):
                continue
            large_positions = []
            for lr, lc in positions:
                if rec["mirror_branch"] and not (lc < sc):
                    continue
                if not rec["mirror_branch"] and not (lc >= sc):
                    continue
                large_box = (lr, lc, lr + h - 1, lc + w - 1)
                if _can_place_box(large_box, placed_boxes + [small_box]):
                    large_positions.append((lr, lc))
            if not large_positions:
                continue
            lr, lc = random.choice(large_positions)
            rec["small_pos"] = (sr, sc)
            rec["large_pos"] = (lr, lc)
            placed_boxes.extend(
                [
                    (sr, sc, sr + h - 1, sc + w - 1),
                    (lr, lc, lr + h - 1, lc + w - 1),
                ]
            )
            success = True
            break
        if not success:
            return None

    grid = np.zeros((H, W), dtype=int)
    for rec in records:
        lr, lc = rec["large_pos"]
        sr, sc = rec["small_pos"]
        _paste_nonblack(grid, rec["large_patch"], lr, lc)
        _paste_nonblack(grid, rec["small_patch"], sr, sc)

    if not _semantic_validate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
