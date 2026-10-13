from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _dihedral_arrays(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _unique_sprite_variants(sprite: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    by_mask: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for idx, patch in enumerate(_dihedral_arrays(sprite)):
        mask = patch != BG
        key = _mask_key(mask)
        if key in by_mask:
            assert np.array_equal(by_mask[key], patch), "ambiguous color mapping for symmetric source shape"
            continue
        by_mask[key] = patch.copy()
        variants.append({"id": idx, "patch": patch.copy(), "mask_key": key})
    return variants


def _find_marker_side(grid: np.ndarray, box: tuple[int, int, int, int]) -> str:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    hits: list[str] = []
    if r0 > 0 and np.any(grid[r0 - 1, c0 : c1 + 1] == MARKER):
        hits.append("top")
    if r1 + 1 < h and np.any(grid[r1 + 1, c0 : c1 + 1] == MARKER):
        hits.append("bottom")
    if c0 > 0 and np.any(grid[r0 : r1 + 1, c0 - 1] == MARKER):
        hits.append("left")
    if c1 + 1 < w and np.any(grid[r0 : r1 + 1, c1 + 1] == MARKER):
        hits.append("right")
    assert len(hits) == 1, "each placeholder must have exactly one side marker"
    return hits[0]


def _flip_patch(patch: np.ndarray, side: str) -> np.ndarray:
    if side in {"top", "bottom"}:
        return np.flipud(patch)
    return np.fliplr(patch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the colored source onto each grey copy, flip the stamped copy by the side marker, then erase guides.

    Input structure:
    - A black background contains one irregular multicolor 4-connected source sprite using 2-3 colors that are neither black, grey, nor orange.
    - Elsewhere there are 2-3 grey 4-connected placeholder objects whose masks match the source sprite under rotation and/or reflection.
    - Each grey placeholder has exactly one orange marker pixel directly above, below, left, or right of its tight bounding box.

    Rule:
    - Find the unique multicolor source sprite and compute all of its dihedral transforms.
    - For each grey placeholder, match its tight-bbox mask to exactly one transformed source mask and stamp that transformed colored sprite into the placeholder bbox.
    - If the orange marker is above or below the placeholder, flip the whole stamped bbox vertically; if the marker is left or right, flip the whole stamped bbox horizontally.
    - Erase all orange markers and erase the original source sprite to black.

    Key insight:
    - The grey mask tells which transformed source sprite to stamp, while the external orange marker independently tells which axis to use for the local bbox flip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source_mask = ~np.isin(g, (BG, GREY, MARKER))
    source_components = _components(source_mask)
    assert len(source_components) == 1, "expected exactly one colored source sprite"
    source_box = _bbox(source_components[0])
    sprite = _crop(g, source_box)
    assert len(np.unique(sprite[sprite != BG])) >= 2, "source sprite must be multicolor"
    variants = _unique_sprite_variants(sprite)

    placeholder_components = _components(g == GREY)
    assert 2 <= len(placeholder_components) <= 3, "expected 2-3 grey placeholders"

    out = g.copy()
    out[source_mask] = BG
    out[g == MARKER] = BG

    for coords in placeholder_components:
        box = _bbox(coords)
        placeholder = _crop(g, box)
        key = _mask_key(placeholder == GREY)
        matches = [variant for variant in variants if variant["mask_key"] == key]
        assert len(matches) == 1, "placeholder mask must match exactly one source orientation"

        stamped = np.array(matches[0]["patch"], dtype=int)
        flipped = _flip_patch(stamped, _find_marker_side(g, box))
        r0, c0, r1, c1 = box
        assert flipped.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = flipped

    return out


GRID_SIDE_RANGE = (15, 20)
PLACEHOLDER_COUNT_CHOICES = (2, 3, 3)
MASK_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4))
MASK_PIXELS = (6, 7, 8, 9, 10)
PADDING = 1
MAX_ATTEMPTS = 500


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < target:
        edge: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    edge.add((nr, nc))
        if not edge:
            return mask
        cell = random.choice(tuple(edge))
        mask[cell] = True
        frontier.add(cell)
    return mask


def _full_bbox(mask: np.ndarray) -> bool:
    return np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))


def _mask_variant_count(mask: np.ndarray) -> int:
    return len({_mask_key(variant) for variant in _dihedral_arrays(mask.astype(int))})


def _random_sprite() -> np.ndarray | None:
    h, w = random.choice(MASK_DIMS)
    area = h * w
    low = max(6, int(np.ceil(0.45 * area)))
    high = min(10, area - 1)
    if low > high:
        return None

    for _ in range(120):
        target = random.randint(low, high)
        mask = _random_connected_mask(h, w, target)
        if int(mask.sum()) != target:
            continue
        if not _full_bbox(mask):
            continue
        if _mask_variant_count(mask) != 8:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        color_count = random.choice((2, 3))
        if len(coords) < color_count:
            continue
        palette = random.sample(SOURCE_COLORS, color_count)
        sprite = np.zeros((h, w), dtype=int)
        random.shuffle(coords)
        for i, (r, c) in enumerate(coords):
            sprite[r, c] = palette[i] if i < color_count else random.choice(palette)
        return sprite
    return None


def _edge_slots(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    return [(r, w - 1) for r in range(h) if mask[r, w - 1]]


def _marker_cell(r0: int, c0: int, patch: np.ndarray, side: str) -> tuple[int, int] | None:
    slots = _edge_slots(patch != BG, side)
    if not slots:
        return None
    rr, cc = random.choice(slots)
    if side == "top":
        return r0 - 1, c0 + cc
    if side == "bottom":
        return r0 + patch.shape[0], c0 + cc
    if side == "left":
        return r0 + rr, c0 - 1
    return r0 + rr, c0 + patch.shape[1]


def _reserve_cells(
    occupied: np.ndarray,
    box: tuple[int, int, int, int],
    marker: tuple[int, int] | None,
) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(h - 1, r1 + PADDING)
    cc1 = min(w - 1, c1 + PADDING)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    if marker is not None:
        mr, mc = marker
        occupied[mr, mc] = True


def _placeable(
    occupied: np.ndarray,
    box: tuple[int, int, int, int],
    marker: tuple[int, int] | None,
) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(h - 1, r1 + PADDING)
    cc1 = min(w - 1, c1 + PADDING)
    if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    if marker is None:
        return True
    mr, mc = marker
    if not (0 <= mr < h and 0 <= mc < w):
        return False
    return not occupied[mr, mc]


def _expected_from_metadata(inp: np.ndarray, source_box: tuple[int, int, int, int], placeholders: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = source_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    out[inp == MARKER] = BG
    out[inp == GREY] = BG
    for item in placeholders:
        pr0, pc0, pr1, pc1 = item["box"]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = item["final_patch"]
    return out


def _semantic_checks(inp: np.ndarray, sprite: np.ndarray, source_box: tuple[int, int, int, int], placeholders: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not (2 <= len(placeholders) <= 3):
        return False

    variants = _unique_sprite_variants(sprite)
    axes = []
    final_signatures = set()
    transform_ids = []
    axis_by_transform: dict[int, set[str]] = {}

    for item in placeholders:
        side = str(item["side"])
        axis = "h" if side in {"top", "bottom"} else "v"
        axes.append(axis)
        transform_ids.append(int(item["transform_id"]))
        axis_by_transform.setdefault(int(item["transform_id"]), set()).add(axis)

        stamped = np.array(item["stamped_patch"], dtype=int)
        final_patch = np.array(item["final_patch"], dtype=int)
        opposite = np.fliplr(stamped) if axis == "h" else np.flipud(stamped)
        if np.array_equal(stamped, final_patch):
            return False
        if np.array_equal(final_patch, opposite):
            return False
        final_signatures.add(final_patch.tobytes())

    if "h" not in axes or "v" not in axes:
        return False
    if len(set(transform_ids)) < 2:
        return False
    if len(final_signatures) < 2:
        return False

    repeated = [tid for tid, seen_axes in axis_by_transform.items() if len(seen_axes) >= 2]
    if len(placeholders) == 3 and not repeated:
        return False

    source_colors = set(int(v) for v in sprite[sprite != BG])
    if not (2 <= len(source_colors) <= 3):
        return False
    if source_colors & {GREY, MARKER}:
        return False

    grey_components = _components(inp == GREY)
    if len(grey_components) != len(placeholders):
        return False
    source_components = _components(~np.isin(inp, (BG, GREY, MARKER)))
    if len(source_components) != 1:
        return False

    try:
        expected = _expected_from_metadata(inp, source_box, placeholders)
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(expected, solved)


def _generate_case() -> np.ndarray | None:
    sprite = _random_sprite()
    if sprite is None:
        return None

    h, w = random.randint(*GRID_SIDE_RANGE), random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    sh, sw = sprite.shape
    source_positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(source_positions)
    if not source_positions:
        return None
    sr, sc = source_positions[0]
    source_box = (sr, sc, sr + sh - 1, sc + sw - 1)
    _reserve_cells(occupied, source_box, None)
    grid[sr : sr + sh, sc : sc + sw] = sprite

    variants = _unique_sprite_variants(sprite)
    variant_map = {int(item["id"]): np.array(item["patch"], dtype=int) for item in variants}
    variant_ids = list(variant_map)
    count = random.choice(PLACEHOLDER_COUNT_CHOICES)
    if len(variant_ids) < 2:
        return None

    if count == 2:
        chosen_ids = random.sample(variant_ids, 2)
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    else:
        repeated = random.choice(variant_ids)
        other = random.choice([vid for vid in variant_ids if vid != repeated])
        chosen_ids = [repeated, repeated, other]
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right")), random.choice(("top", "bottom", "left", "right"))]
        random.shuffle(chosen_ids)
        random.shuffle(sides)
        if not any(s in {"top", "bottom"} for s in sides):
            sides[0] = "top"
        if not any(s in {"left", "right"} for s in sides):
            sides[1] = "left"

    placeholders: list[dict[str, object]] = []
    for transform_id, side in zip(chosen_ids, sides):
        patch = variant_map[transform_id]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            marker = _marker_cell(r0, c0, patch, side)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if marker is None or not _placeable(occupied, box, marker):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = GREY
            grid[marker] = MARKER
            _reserve_cells(occupied, box, marker)
            stamped = patch.copy()
            final_patch = _flip_patch(stamped, side)
            placeholders.append(
                {
                    "box": box,
                    "side": side,
                    "transform_id": int(transform_id),
                    "stamped_patch": stamped,
                    "final_patch": final_patch,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not _semantic_checks(grid, sprite, source_box, placeholders):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid ARC input")
