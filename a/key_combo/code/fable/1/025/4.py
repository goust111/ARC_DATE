from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import itertools
import random

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    from arc_utils import bbox, components_mask, crop_bbox, get_bg
except ModuleNotFoundError:
    _arc_utils = SourceFileLoader("_arc_utils_fable_025_4", str(_ARC_UTILS_PATH)).load_module()
    bbox = _arc_utils.bbox
    components_mask = _arc_utils.components_mask
    crop_bbox = _arc_utils.crop_bbox
    get_bg = _arc_utils.get_bg


PALETTE = tuple(range(10))
GRID_LIMIT = 30
GENERATION_RETRY_LIMIT = 400

TEMPLATE_HEIGHT_RANGE = (4, 6)
TEMPLATE_WIDTH_RANGE = (4, 6)
TEMPLATE_FILL_RANGE = (0.42, 0.72)
FRAGMENT_COUNT_RANGE = (2, 4)
LOCK_COUNT_RANGE = (3, 4)

OUTER_MARGIN = 1
PANEL_GAP_RANGE = (2, 4)
OBJECT_GAP = 1


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _coords_to_cropped_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _coords_bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _shape_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _has_interior_hole(mask: np.ndarray) -> bool:
    for coords in components_mask(~mask, connectivity=4):
        if not _touches_edge(coords, mask.shape):
            return True
    return False


def _component_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(grid))
    records: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        color_mask = grid == color
        for coords in components_mask(color_mask, connectivity=4):
            box = _coords_bbox(coords)
            local = crop_bbox(color_mask, box)
            hole_parts = [
                part
                for part in components_mask(~local, connectivity=4)
                if not _touches_edge(part, local.shape)
            ]
            if hole_parts:
                assert len(hole_parts) == 1, "each lock must contain exactly one enclosed hole"
                hole_local = _coords_to_cropped_mask(hole_parts[0])
                hole_global = hole_parts[0] + np.array([box[0], box[1]], dtype=int)
                records.append(
                    {
                        "kind": "lock",
                        "color": int(color),
                        "coords": coords.astype(int),
                        "bbox": box,
                        "hole_mask": hole_local,
                        "hole_coords": hole_global.astype(int),
                    }
                )
            else:
                records.append(
                    {
                        "kind": "fragment",
                        "color": int(color),
                        "coords": coords.astype(int),
                        "bbox": box,
                        "mask": local.astype(bool),
                    }
                )
    return bg, records


def _aligned_union(masks: list[np.ndarray]) -> np.ndarray:
    h = max(int(mask.shape[0]) for mask in masks)
    w = max(int(mask.shape[1]) for mask in masks)
    out = np.zeros((h, w), dtype=bool)
    for mask in masks:
        mh, mw = map(int, mask.shape)
        out[:mh, :mw] |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full key by aligned union of all partial copies, then fill the unique matching lock hole and erase the fragments.

    Input structure:
    - One side of the grid contains several separated monochrome fragments, each a damaged copy of one common sprite mask.
    - Another side contains several monochrome closed lock frames, each enclosing one background hole shape.
    - Exactly one lock hole matches the aligned union of all fragments, and one single fragment matches a different decoy lock hole.

    Rule:
    - Crop every fragment to its bounding box and align the crops by their top-left corners.
    - Take the union of all aligned fragment masks to recover the full key template.
    - Find the unique lock whose enclosed hole exactly matches that template without rotation or reflection, fill that hole with the lock's own color, and erase every fragment.

    Key insight:
    - Any single fragment is a decoy; only the complementary union of all fragments reconstructs the true key shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, records = _component_records(g)
    fragments = [record for record in records if record["kind"] == "fragment"]
    locks = [record for record in records if record["kind"] == "lock"]
    assert len(fragments) >= 2, "expected multiple key fragments"
    assert locks, "expected at least one lock"

    template = _aligned_union([record["mask"] for record in fragments])
    matches = [record for record in locks if _shape_equal(record["hole_mask"], template)]
    assert len(matches) == 1, "expected a unique lock matched by the fragment union"
    target = matches[0]

    out = g.copy()
    for record in fragments:
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    hole_coords = target["hole_coords"]
    out[hole_coords[:, 0], hole_coords[:, 1]] = int(target["color"])
    return out


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*TEMPLATE_HEIGHT_RANGE)
        w = random.randint(*TEMPLATE_WIDTH_RANGE)
        total = h * w
        min_area = max(7, int(np.ceil(TEMPLATE_FILL_RANGE[0] * total)))
        max_area = min(total - 1, int(np.floor(TEMPLATE_FILL_RANGE[1] * total)))
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (nr, nc)
            for nr, nc in (
                (start[0] - 1, start[1]),
                (start[0] + 1, start[1]),
                (start[0], start[1] - 1),
                (start[0], start[1] + 1),
            )
            if 0 <= nr < h and 0 <= nc < w
        }

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue
        box = bbox(mask)
        if box is None:
            continue
        cropped = crop_bbox(mask, box).astype(bool)
        if cropped.shape[0] < TEMPLATE_HEIGHT_RANGE[0] or cropped.shape[1] < TEMPLATE_WIDTH_RANGE[0]:
            continue
        if np.all(cropped) or _has_interior_hole(cropped):
            continue
        return cropped
    return None


def _removable_cells(mask: np.ndarray, protect: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        if (r, c) in protect:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if not trial[0].any() or not trial[:, 0].any():
            continue
        if len(components_mask(trial, connectivity=4)) != 1:
            continue
        if _has_interior_hole(trial):
            continue
        cells.append((int(r), int(c)))
    return cells


def _damaged_fragments(template: np.ndarray, count: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    area = len(coords)
    if area < count + 4:
        return None

    top_anchor = random.choice([cell for cell in coords if cell[0] == 0])
    left_anchor = random.choice([cell for cell in coords if cell[1] == 0])
    bottom_anchor = random.choice([cell for cell in coords if cell[0] == template.shape[0] - 1])
    right_anchor = random.choice([cell for cell in coords if cell[1] == template.shape[1] - 1])

    max_remove = max(1, min(area - 3, area // 3))
    for _ in range(160):
        missing_counts = {cell: 0 for cell in coords}
        masks: list[np.ndarray] = []
        ok = True
        for i in range(count):
            current = template.copy()
            protect = {top_anchor, left_anchor}
            if i == 0:
                protect |= {bottom_anchor, right_anchor}

            target_remove = random.randint(1, max_remove)
            removed = 0
            while removed < target_remove:
                candidates = [
                    cell
                    for cell in _removable_cells(current, protect)
                    if missing_counts[cell] < count - 1
                ]
                if not candidates:
                    break
                chosen = random.choice(candidates)
                current[chosen] = False
                missing_counts[chosen] += 1
                removed += 1

            if removed == 0:
                ok = False
                break
            if i == 0 and (not current[-1].any() or not current[:, -1].any()):
                ok = False
                break
            if _shape_equal(current, template):
                ok = False
                break
            masks.append(current)

        if not ok:
            continue
        if len({_mask_signature(mask) for mask in masks}) != len(masks):
            continue
        if not _shape_equal(_aligned_union(masks), template):
            continue
        return masks
    return None


def _subset_match_ok(
    fragments: list[np.ndarray],
    lock_holes: list[np.ndarray],
    target_idx: int,
    decoy_lock_idx: int,
    decoy_fragment_idx: int,
) -> bool:
    n = len(fragments)
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            union_mask = _aligned_union([fragments[i] for i in subset])
            matches = [i for i, hole in enumerate(lock_holes) if _shape_equal(union_mask, hole)]
            if size == n:
                if matches != [target_idx]:
                    return False
            elif size == 1 and subset[0] == decoy_fragment_idx:
                if matches != [decoy_lock_idx]:
                    return False
            elif matches:
                return False
    return True


def _render_fragment_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_lock_patch(hole_mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = map(int, hole_mask.shape)
    patch = np.full((h + 2, w + 2), int(color), dtype=int)
    patch[1 : h + 1, 1 : w + 1][hole_mask] = int(bg)
    return patch


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = OBJECT_GAP
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patches(
    region_top: int,
    region_left: int,
    region_h: int,
    region_w: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    boxes: list[tuple[int, int, int, int]] = []
    placed_pairs: list[tuple[int, tuple[int, int, int, int]]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph > region_h or pw > region_w:
            return None
        candidates: list[tuple[int, int]] = []
        for r in range(region_top, region_top + region_h - ph + 1):
            for c in range(region_left, region_left + region_w - pw + 1):
                box = (r, c, r + ph - 1, c + pw - 1)
                if all(_boxes_separated(box, other) for other in boxes):
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[idx] = pos
        box = (pos[0], pos[1], pos[0] + ph - 1, pos[1] + pw - 1)
        boxes.append(box)
        placed_pairs.append((idx, box))

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _panel_height_need(panel_w: int, patches: list[np.ndarray]) -> int:
    usable_w = max(1, panel_w - 1)
    area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    tallest = max(int(patch.shape[0]) for patch in patches)
    return max(tallest + 2, int(np.ceil(area / usable_w)) + len(patches) + 2)


def _build_lock_holes(fragments: list[np.ndarray], target: np.ndarray) -> tuple[list[np.ndarray], int, int] | None:
    lock_count = random.randint(*LOCK_COUNT_RANGE)
    decoy_fragment_idx = random.randrange(len(fragments))
    lock_holes = [target.copy(), fragments[decoy_fragment_idx].copy()]
    forbidden = {_mask_signature(mask) for mask in lock_holes}

    while len(lock_holes) < lock_count:
        candidate = _random_connected_mask()
        if candidate is None:
            return None
        sig = _mask_signature(candidate)
        if sig in forbidden:
            continue
        forbidden.add(sig)
        lock_holes.append(candidate)

    indexed = list(enumerate(lock_holes))
    random.shuffle(indexed)
    shuffled_holes = [mask for _, mask in indexed]
    target_idx = next(i for i, (original, _) in enumerate(indexed) if original == 0)
    decoy_lock_idx = next(i for i, (original, _) in enumerate(indexed) if original == 1)
    if not _subset_match_ok(
        fragments, shuffled_holes, target_idx=target_idx, decoy_lock_idx=decoy_lock_idx, decoy_fragment_idx=decoy_fragment_idx
    ):
        return None
    return shuffled_holes, target_idx, decoy_lock_idx


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = map(int, patch.shape)
    region = grid[top : top + h, left : left + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    template = _random_connected_mask()
    if template is None:
        return None

    fragment_count = random.randint(*FRAGMENT_COUNT_RANGE)
    fragments = _damaged_fragments(template, fragment_count)
    if fragments is None:
        return None
    if any(_shape_equal(_aligned_union([fragments[i] for i in range(len(fragments)) if i != j]), template) for j in range(len(fragments))):
        return None

    built = _build_lock_holes(fragments, template)
    if built is None:
        return None
    lock_holes, target_lock_idx, _decoy_lock_idx = built

    n_frag = len(fragments)
    n_locks = len(lock_holes)
    if n_frag + n_locks + 1 > len(PALETTE):
        return None

    bg = random.choice(PALETTE)
    object_colors = random.sample([color for color in PALETTE if color != bg], n_frag + n_locks)
    fragment_colors = object_colors[:n_frag]
    lock_colors = object_colors[n_frag:]

    fragment_patches = [
        _render_fragment_patch(mask, color, bg) for mask, color in zip(fragments, fragment_colors)
    ]
    lock_patches = [_render_lock_patch(mask, color, bg) for mask, color in zip(lock_holes, lock_colors)]

    max_frag_w = max(int(patch.shape[1]) for patch in fragment_patches)
    max_lock_w = max(int(patch.shape[1]) for patch in lock_patches)
    gap = random.randint(*PANEL_GAP_RANGE)
    base_frag_w = max_frag_w + 2
    base_lock_w = max_lock_w + 2
    base_width = base_frag_w + base_lock_w + gap + 2 * OUTER_MARGIN
    if base_width > GRID_LIMIT:
        return None

    width = random.randint(base_width, GRID_LIMIT)
    extra = width - base_width
    extra_frag = random.randint(0, extra)
    extra_lock = extra - extra_frag
    frag_panel_w = base_frag_w + extra_frag
    lock_panel_w = base_lock_w + extra_lock

    min_height = max(
        16,
        _panel_height_need(frag_panel_w, fragment_patches),
        _panel_height_need(lock_panel_w, lock_patches),
    )
    if min_height > GRID_LIMIT:
        return None
    height = random.randint(min_height, GRID_LIMIT)

    left_is_fragments = bool(random.getrandbits(1))
    if left_is_fragments:
        frag_left = OUTER_MARGIN
        lock_left = OUTER_MARGIN + frag_panel_w + gap
    else:
        lock_left = OUTER_MARGIN
        frag_left = OUTER_MARGIN + lock_panel_w + gap

    panel_top = OUTER_MARGIN
    panel_h = height - 2 * OUTER_MARGIN
    frag_positions = _place_patches(panel_top, frag_left, panel_h, frag_panel_w, fragment_patches)
    if frag_positions is None:
        return None
    lock_positions = _place_patches(panel_top, lock_left, panel_h, lock_panel_w, lock_patches)
    if lock_positions is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for patch, (top, left) in zip(fragment_patches, frag_positions):
        _paint_patch(grid, patch, top, left, bg)
    for patch, (top, left) in zip(lock_patches, lock_positions):
        _paint_patch(grid, patch, top, left, bg)

    counts = np.bincount(grid.ravel(), minlength=max(PALETTE) + 1)
    if int(np.argmax(counts)) != bg:
        return None

    expected = grid.copy()
    for mask, (top, left) in zip(fragments, frag_positions):
        coords = np.argwhere(mask)
        expected[coords[:, 0] + top, coords[:, 1] + left] = bg

    target_mask = lock_holes[target_lock_idx]
    target_top, target_left = lock_positions[target_lock_idx]
    target_coords = np.argwhere(target_mask)
    expected[target_coords[:, 0] + target_top + 1, target_coords[:, 1] + target_left + 1] = lock_colors[target_lock_idx]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/1/025/4")
