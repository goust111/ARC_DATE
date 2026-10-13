from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_w, crop_bbox, find_components, get_bg, paste


BG = 0
OVERLAP_COLOR = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
GRID_H_RANGE = (16, 23)
GRID_W_RANGE = (16, 24)
SPRITE_COUNT_RANGE = (5, 7)
GENERATE_ATTEMPTS = 600
LAYOUT_ATTEMPTS = 250
TEMPLATE_PLACEMENT_ATTEMPTS = 80


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    pts = {tuple(map(int, rc)) for rc in coords}
    while stack:
        r, c = stack.pop()
        for dr, dc in FOUR_DIRS:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _enumerate_spanning_masks(size: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    cell_count = size * size
    for bits in range(1, 1 << cell_count):
        flat = [(bits >> idx) & 1 for idx in range(cell_count)]
        mask = np.array(flat, dtype=bool).reshape(size, size)
        area = int(mask.sum())
        if area < 3:
            continue
        if not _is_connected(mask):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        masks.append(mask)
    return masks


MASK_LIBRARY = {2: _enumerate_spanning_masks(2), 3: _enumerate_spanning_masks(3)}


def _crop_key(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return patch.shape, tuple(int(x) for x in patch.ravel())


def _random_sprite_patch(size: int, used_keys: set[tuple[tuple[int, int], tuple[int, ...]]]) -> np.ndarray:
    for _ in range(200):
        mask = random.choice(MASK_LIBRARY[size])
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        color_count = random.randint(2, min(4, len(coords)))
        colors = random.sample(INPUT_COLORS, color_count)
        random.shuffle(coords)
        patch = np.zeros((size, size), dtype=int)
        for (r, c), color in zip(coords, colors):
            patch[r, c] = int(color)
        for r, c in coords[color_count:]:
            patch[r, c] = int(random.choice(colors))
        key = _crop_key(patch)
        if key not in used_keys:
            return patch
    raise RuntimeError("failed to generate a unique sprite patch")


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    occ = grid != BG
    for dr, dc in np.argwhere(patch != BG):
        r = int(r0 + dr)
        c = int(c0 + dc)
        if occ[r, c]:
            return False
        rr0 = max(0, r - padding)
        rr1 = min(H, r + padding + 1)
        cc0 = max(0, c - padding)
        cc1 = min(W, c + padding + 1)
        if np.any(occ[rr0:rr1, cc0:cc1]):
            return False
    return True


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _candidate_positions(H: int, W: int, template_size: int, sprite_size: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    max_row = H - max(template_size, sprite_size)
    for r in range(max_row + 1):
        for stamp_c in range(W - template_size + 1):
            orig_c = W - sprite_size - stamp_c
            if not (0 <= orig_c <= W - sprite_size):
                continue
            if stamp_c + template_size + 1 > orig_c:
                continue
            out.append((int(r), int(stamp_c), int(orig_c)))
    return out


def _stamp_overlaps_original(original_grid: np.ndarray, template_patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = template_patch.shape
    target = original_grid[r0 : r0 + h, c0 : c0 + w]
    return bool(np.any((template_patch != BG) & (target != BG)))


def _stamp_coverage(shape: tuple[int, int], template_patch: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    cov = np.zeros(shape, dtype=int)
    fg_coords = np.argwhere(template_patch != BG)
    for r0, c0 in positions:
        for dr, dc in fg_coords:
            cov[int(r0 + dr), int(c0 + dc)] += 1
    return cov


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) >= 3, "expected several foreground sprites"

    patches: list[np.ndarray] = []
    counts: Counter[tuple[tuple[int, int], tuple[int, ...]]] = Counter()
    members: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = {}
    for idx, comp in enumerate(components):
        patch = crop_bbox(g, comp["bbox"])
        key = _crop_key(patch)
        patches.append(patch)
        counts[key] += 1
        members.setdefault(key, []).append(idx)

    duplicate_keys = [key for key, count in counts.items() if count == 2]
    assert len(duplicate_keys) == 1, "expected exactly one repeated sprite crop"
    template_key = duplicate_keys[0]
    assert all(
        count == 1 or (key == template_key and count == 2)
        for key, count in counts.items()
    ), "expected all non-template crops to be unique"

    template_indices = members[template_key]
    template_patch = patches[template_indices[0]].copy()
    remaining_indices = [idx for idx in range(len(components)) if idx not in template_indices]

    placements: list[tuple[int, int, int]] = []
    for idx in remaining_indices:
        r0, c0, _r1, _c1 = map(int, components[idx]["bbox"])
        sprite_w = int(bbox_w(components[idx]["bbox"]))
        new_c = int(g.shape[1] - sprite_w - c0)
        placements.append((int(r0), new_c, idx))

    return {
        "bg": bg,
        "components": components,
        "patches": patches,
        "template_patch": template_patch,
        "template_indices": template_indices,
        "remaining_indices": remaining_indices,
        "placements": placements,
    }


def _render_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(analysis["bg"])
    out = g.copy()
    template_patch = np.asarray(analysis["template_patch"], dtype=int)
    th, tw = template_patch.shape
    H, W = g.shape

    for idx in analysis["template_indices"]:
        coords = analysis["components"][idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    coverage = np.zeros((H, W), dtype=int)
    for r0, c0, _idx in sorted(analysis["placements"], key=lambda item: (item[0], item[1])):
        assert 0 <= r0 <= H - th and 0 <= c0 <= W - tw, "mirrored stamp must fit"
        for dr, dc in np.argwhere(template_patch != bg):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            color = int(template_patch[dr, dc])
            if coverage[rr, cc] > 0:
                coverage[rr, cc] += 1
                out[rr, cc] = OVERLAP_COLOR
                continue
            if int(out[rr, cc]) == bg:
                out[rr, cc] = color
                coverage[rr, cc] = 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the only repeated sprite and use it as a stamp at the horizontal mirror positions of the other sprites.

    Input structure:
    - The background is black.
    - The foreground consists of several separated 4-connected multicolor sprites, each inside a tight 2x2 or 3x3 bbox.
    - Exactly one full colored bbox crop, including any internal background cells, appears twice; every other sprite crop is unique.

    Rule:
    - Crop every sprite to its tight bbox and compare the full crops, keeping internal background cells.
    - Identify the unique crop that appears exactly twice and treat that crop as the stamp template.
    - Erase both copies of the repeated sprite, then for each remaining sprite compute its horizontal mirror bbox position with `new_c = grid_width - sprite_width - old_c`.
    - Place the template with its own orientation unchanged at each mirrored top-left position, painting only onto background cells; if a later stamp hits a cell painted by an earlier stamp, recolor that cell gray.
    - Keep the original non-repeated sprites fixed in place.

    Key insight:
    - The duplicated sprite is not an output object to keep; it is the hidden template, while the other sprites contribute only mirrored placement coordinates.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), analysis)


def _gen_nonrepeat_sizes(template_size: int, count: int) -> list[int]:
    other_size = 3 if template_size == 2 else 2
    sizes = [template_size, other_size]
    while len(sizes) < count:
        sizes.append(random.choice([2, 3]))
    random.shuffle(sizes)
    return sizes


def _select_nonrepeat_layout(
    H: int,
    W: int,
    template_patch: np.ndarray,
    nonrepeat_patches: list[np.ndarray],
    want_overlap: bool,
) -> tuple[list[tuple[int, int, int]], np.ndarray, int] | None:
    template_size = int(template_patch.shape[0])
    candidate_lists = [
        _candidate_positions(H, W, template_size, int(patch.shape[0]))
        for patch in nonrepeat_patches
    ]
    if any(not cands for cands in candidate_lists):
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        order = sorted(range(len(nonrepeat_patches)), key=lambda idx: len(candidate_lists[idx]))
        positions: list[tuple[int, int, int] | None] = [None] * len(nonrepeat_patches)
        original_grid = np.zeros((H, W), dtype=int)
        used_stamp_positions: set[tuple[int, int]] = set()
        ok = True

        for idx in order:
            options = candidate_lists[idx][:]
            random.shuffle(options)
            chosen = None
            for r0, stamp_c, orig_c in options:
                if (r0, stamp_c) in used_stamp_positions:
                    continue
                if not _can_place_patch(original_grid, nonrepeat_patches[idx], r0, orig_c, padding=1):
                    continue
                if _stamp_overlaps_original(original_grid, template_patch, r0, stamp_c):
                    continue
                chosen = (int(r0), int(stamp_c), int(orig_c))
                break
            if chosen is None:
                ok = False
                break
            positions[idx] = chosen
            used_stamp_positions.add((chosen[0], chosen[1]))
            paste(original_grid, nonrepeat_patches[idx], chosen[0], chosen[2], transparent=BG)

        if not ok:
            continue

        final_positions = [pos for pos in positions if pos is not None]
        if any(_stamp_overlaps_original(original_grid, template_patch, r0, stamp_c) for r0, stamp_c, _orig_c in final_positions):
            continue

        stamp_positions = [(r0, stamp_c) for r0, stamp_c, _orig_c in final_positions]
        coverage = _stamp_coverage((H, W), template_patch, stamp_positions)
        overlap_pixels = int((coverage >= 2).sum())
        if want_overlap:
            if overlap_pixels == 0:
                continue
            if overlap_pixels > max(4, int((template_patch != BG).sum()) // 2 + 1):
                continue
        elif overlap_pixels != 0:
            continue

        return final_positions, original_grid, overlap_pixels
    return None


def _place_template_copies(
    grid: np.ndarray,
    template_patch: np.ndarray,
    forbidden_stamp_positions: list[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    th, tw = template_patch.shape
    forbidden_boxes = [
        (r0, c0, r0 + th - 1, c0 + tw - 1)
        for r0, c0 in forbidden_stamp_positions
    ]
    placed: list[tuple[int, int]] = []
    for _ in range(TEMPLATE_PLACEMENT_ATTEMPTS):
        candidates = []
        for r0 in range(H - th + 1):
            for c0 in range(W - tw + 1):
                box = (r0, c0, r0 + th - 1, c0 + tw - 1)
                if any(_bboxes_intersect(box, bad) for bad in forbidden_boxes):
                    continue
                if _can_place_patch(grid, template_patch, r0, c0, padding=1):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        chosen = random.choice(candidates)
        placed.append(chosen)
        paste(grid, template_patch, chosen[0], chosen[1], transparent=BG)
        if len(placed) == 2:
            return True
    return False


def _validate_generated_instance(grid: np.ndarray, expect_overlap: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if int(get_bg(grid)) != BG:
        return False
    if OVERLAP_COLOR in set(map(int, grid.ravel())):
        return False

    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return False

    if not (5 <= len(analysis["components"]) <= 7):
        return False

    template_patch = np.asarray(analysis["template_patch"], dtype=int)
    if template_patch.shape[0] not in (2, 3) or template_patch.shape[1] not in (2, 3):
        return False

    erased = grid.copy()
    for idx in analysis["template_indices"]:
        coords = analysis["components"][idx]["coords"]
        erased[coords[:, 0], coords[:, 1]] = BG

    stamp_positions = []
    for r0, c0, _idx in analysis["placements"]:
        th, tw = template_patch.shape
        if not (0 <= r0 <= grid.shape[0] - th and 0 <= c0 <= grid.shape[1] - tw):
            return False
        if _stamp_overlaps_original(erased, template_patch, r0, c0):
            return False
        stamp_positions.append((r0, c0))

    coverage = _stamp_coverage(grid.shape, template_patch, stamp_positions)
    overlap_pixels = int((coverage >= 2).sum())
    if expect_overlap and overlap_pixels == 0:
        return False
    if not expect_overlap and overlap_pixels != 0:
        return False

    out = main(grid)
    expected = _render_output(grid, analysis)
    if not np.array_equal(out, expected):
        return False
    if expect_overlap != bool(np.any(out == OVERLAP_COLOR)):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    total_sprites = random.randint(*SPRITE_COUNT_RANGE)
    nonrepeat_count = total_sprites - 2
    template_size = random.choice([2, 3])
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    used_keys: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    template_patch = _random_sprite_patch(template_size, used_keys)
    used_keys.add(_crop_key(template_patch))

    sizes = _gen_nonrepeat_sizes(template_size, nonrepeat_count)
    nonrepeat_patches = []
    for size in sizes:
        patch = _random_sprite_patch(size, used_keys)
        used_keys.add(_crop_key(patch))
        nonrepeat_patches.append(patch)

    want_overlap = random.random() < 0.65
    layout = _select_nonrepeat_layout(H, W, template_patch, nonrepeat_patches, want_overlap)
    if layout is None:
        return None
    positions, original_grid, _overlap_pixels = layout

    grid = original_grid.copy()
    stamp_positions = [(r0, stamp_c) for r0, stamp_c, _orig_c in positions]
    if not _place_template_copies(grid, template_patch, stamp_positions):
        return None

    if not _validate_generated_instance(grid, want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
