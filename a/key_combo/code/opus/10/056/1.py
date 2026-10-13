from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500
SPRITE_COUNTS = (5, 7)
PATTERN_SHAPES = ((2, 3), (3, 2))
UNIQUE_BBOXES = ((3, 3), (3, 5), (5, 3))
PAIR_BBOXES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
)
PATTERN_MODES = ("random", "rows", "cols", "checker")
PLACEMENT_PAD = 1
TOP_GAP_RANGE = (2, 4)
TOP_HEIGHT_BY_COUNT = {5: (10, 13), 7: (12, 16)}
GRID_WIDTH_BY_COUNT = {5: (16, 22), 7: (19, 25)}


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    seen = set()
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    stack = [coords[0]]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], mask.shape[0], mask.shape[1]):
            if mask[nxt] and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(coords)


def _is_irregular(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if int(mask.sum()) == h * w:
        return False
    row_mixed = any(0 < int(mask[r].sum()) < w for r in range(h))
    col_mixed = any(0 < int(mask[:, c].sum()) < h for c in range(w))
    return row_mixed or col_mixed


def _random_mask(
    h: int,
    w: int,
    *,
    min_area: int = 4,
    max_area: int = 8,
    require_odd: bool = False,
) -> np.ndarray | None:
    if require_odd and (h % 2 == 0 or w % 2 == 0):
        return None

    lower = max(min_area, h + w - 1)
    upper = min(max_area, h * w - 1)
    if upper < lower:
        return None

    for _ in range(300):
        area = random.randint(lower, upper)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), h, w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
            continue
        if not _is_connected(mask):
            continue
        if not _is_irregular(mask):
            continue
        return mask
    return None


def _build_distinct_mask(
    seen: set[tuple[tuple[int, ...], ...]],
    *,
    bbox_choices: tuple[tuple[int, int], ...],
    force_bbox: tuple[int, int] | None = None,
    force_area: int | None = None,
    require_odd: bool = False,
) -> np.ndarray | None:
    choices = [force_bbox] if force_bbox is not None else list(bbox_choices)
    if force_bbox is None:
        random.shuffle(choices)

    for dims in choices:
        h, w = dims
        for _ in range(120):
            mask = _random_mask(
                h,
                w,
                min_area=force_area if force_area is not None else 4,
                max_area=force_area if force_area is not None else 8,
                require_odd=require_odd,
            )
            if mask is None:
                continue
            sig = _mask_signature(mask)
            if sig not in seen:
                seen.add(sig)
                return mask
    return None


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _candidate_locations(reserved: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    H, W = reserved.shape
    locs = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - PLACEMENT_PAD)
            c0 = max(0, c - PLACEMENT_PAD)
            r1 = min(H, r + h + PLACEMENT_PAD)
            c1 = min(W, c + w + PLACEMENT_PAD)
            if not reserved[r0:r1, c0:c1].any():
                locs.append((r, c))
    random.shuffle(locs)
    return locs


def _place_patches(top_h: int, width: int, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    order = list(range(len(patches)))
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for _ in range(250):
        reserved = np.zeros((top_h, width), dtype=bool)
        locs: list[tuple[int, int] | None] = [None] * len(patches)
        placed = True
        for idx in order:
            patch = patches[idx]
            candidates = _candidate_locations(reserved, patch.shape)
            if not candidates:
                placed = False
                break
            r, c = random.choice(candidates)
            h, w = patch.shape
            reserved[r : r + h, c : c + w] = True
            locs[idx] = (r, c)
        if placed:
            return [loc for loc in locs if loc is not None]
    return None


def _make_pattern(colors: list[int], shape: tuple[int, int], unique_color: int) -> np.ndarray:
    rows, cols = shape
    cell_count = rows * cols
    usable = list(colors)
    want = random.randint(2, min(len(usable), cell_count))

    if random.random() < 0.5 and unique_color in usable and want > 1:
        others = [c for c in usable if c != unique_color]
        chosen = [unique_color] + random.sample(others, want - 1)
    else:
        chosen = random.sample(usable, want)
        if len(usable) > want and unique_color in chosen and random.random() < 0.5:
            replacement = random.choice([c for c in usable if c not in chosen])
            chosen[chosen.index(unique_color)] = replacement

    mode = random.choice(PATTERN_MODES)
    if mode == "rows":
        row_colors = [random.choice(chosen) for _ in range(rows)]
        pattern = np.array([[row_colors[r]] * cols for r in range(rows)], dtype=int)
    elif mode == "cols":
        col_colors = [random.choice(chosen) for _ in range(cols)]
        pattern = np.array([col_colors for _ in range(rows)], dtype=int)
    elif mode == "checker":
        order = random.sample(chosen, len(chosen))
        offset = random.randrange(len(order))
        pattern = np.array(
            [[order[(r + c + offset) % len(order)] for c in range(cols)] for r in range(rows)],
            dtype=int,
        )
    else:
        pattern = np.array(
            [[random.choice(chosen) for _ in range(cols)] for _ in range(rows)],
            dtype=int,
        )

    if len(set(map(int, pattern.ravel()))) == 1:
        flat_index = random.randrange(cell_count)
        replacement_choices = [c for c in chosen if c != int(pattern.ravel()[0])]
        if replacement_choices:
            pattern.flat[flat_index] = random.choice(replacement_choices)

    missing = [color for color in chosen if color not in set(map(int, pattern.ravel()))]
    if missing:
        cells = random.sample(range(cell_count), len(missing))
        for idx, color in zip(cells, missing):
            pattern.flat[idx] = color
    return pattern


def _expand_pattern(pattern: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    th, tw = template_mask.shape
    rows, cols = pattern.shape
    out = np.zeros((rows * th, cols * tw), dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = int(pattern[r, c])
            block = np.zeros((th, tw), dtype=int)
            block[template_mask] = color
            paste(out, block, r * th, c * tw)
    return out


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(comps) in (6, 8), "expected 5 or 7 sprites plus one index grid"

    indexed = list(enumerate(comps))
    index_idx, index_comp = max(indexed, key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][2], item[1]["area"]))
    pattern = crop_bbox(g, index_comp["bbox"])
    assert pattern.shape in PATTERN_SHAPES, "index grid must be 2x3 or 3x2"
    assert np.all(pattern != bg), "index grid must be dense"

    sprite_comps = [comp for i, comp in indexed if i != index_idx]
    assert all(comp["bbox"][2] < index_comp["bbox"][0] for comp in sprite_comps), "sprites must be above the index grid"

    signatures = []
    sprite_colors = []
    for comp in sprite_comps:
        crop = crop_bbox(g, comp["bbox"])
        colors = sorted(set(map(int, crop[crop != bg].ravel())))
        assert len(colors) == 1, "every sprite must be monochrome"
        sprite_colors.append(colors[0])
        signatures.append(_mask_signature(crop != bg))

    counts = Counter(signatures)
    unique = [sig for sig, count in counts.items() if count == 1]
    paired = [count for count in counts.values() if count != 1]
    assert len(unique) == 1, "need exactly one outlier sprite shape"
    assert paired and all(count == 2 for count in paired), "all non-outlier shapes must appear twice"

    template_mask = np.array(unique[0], dtype=bool)
    assert template_mask.shape[0] % 2 == 1 and template_mask.shape[1] % 2 == 1
    assert 4 <= int(template_mask.sum()) <= 8
    assert set(map(int, pattern.ravel())).issubset(set(sprite_colors))
    return pattern, template_mask, sprite_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique upper sprite shape and use it as the stencil for expanding the lower color index grid.

    Input structure:
    - Black background with an upper region containing 5 or 7 separated monochrome 4-connected sprites.
    - Exactly one sprite crop shape appears once; every other crop shape appears exactly twice.
    - A dense 2x3 or 3x2 non-black index grid sits below all sprites and uses only sprite colors.

    Rule:
    - Crop each upper sprite to its bounding box and compare crop shapes by foreground/transparent layout.
    - Select the unique crop shape, keeping only its solid-vs-transparent stencil.
    - Replace each cell of the lower index grid with one copy of that stencil recolored to the cell color, leaving stencil holes black.

    Key insight:
    - The lower matrix does not contain the template explicitly; the only way to recover the output block shape is to detect the outlier sprite by repeated crop-shape counting.
    """
    pattern, template_mask, _sprite_colors = _parse_scene(np.asarray(grid, dtype=int))
    return _expand_pattern(pattern, template_mask)


def _validate_scene(grid: np.ndarray, pattern: np.ndarray, template_mask: np.ndarray) -> bool:
    try:
        parsed_pattern, parsed_template, _sprite_colors = _parse_scene(grid)
    except Exception:
        return False
    return np.array_equal(parsed_pattern, pattern) and np.array_equal(parsed_template, template_mask)


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice(SPRITE_COUNTS)
    pair_count = (sprite_count - 1) // 2
    seen: set[tuple[tuple[int, ...], ...]] = set()

    unique_mask = _build_distinct_mask(seen, bbox_choices=UNIQUE_BBOXES, require_odd=True)
    if unique_mask is None:
        return None

    unique_area = int(unique_mask.sum())
    unique_dims = tuple(map(int, unique_mask.shape))
    pair_masks = []
    for idx in range(pair_count):
        force_bbox = unique_dims if idx == 0 and random.random() < 0.7 else None
        force_area = unique_area if idx == 1 and random.random() < 0.7 else None
        mask = _build_distinct_mask(
            seen,
            bbox_choices=PAIR_BBOXES,
            force_bbox=force_bbox,
            force_area=force_area,
            require_odd=False,
        )
        if mask is None:
            return None
        pair_masks.append(mask)

    colors = random.sample(PALETTE, sprite_count)
    unique_color = colors[0]
    patches = [_mask_to_patch(unique_mask, unique_color)]
    color_idx = 1
    for mask in pair_masks:
        for _ in range(2):
            patches.append(_mask_to_patch(mask, colors[color_idx]))
            color_idx += 1
    random.shuffle(patches)

    pattern_shape = random.choice(PATTERN_SHAPES)
    pattern = _make_pattern(colors, pattern_shape, unique_color)
    expected = _expand_pattern(pattern, unique_mask)
    if max(expected.shape) > 30:
        return None

    top_h = random.randint(*TOP_HEIGHT_BY_COUNT[sprite_count])
    gap = random.randint(*TOP_GAP_RANGE)
    bottom_pad = random.randint(0, 2)
    H = top_h + gap + pattern_shape[0] + bottom_pad
    W = random.randint(*GRID_WIDTH_BY_COUNT[sprite_count])
    if max(H, W) > 30:
        return None

    locations = _place_patches(top_h, W, patches)
    if locations is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    for patch, (r, c) in zip(patches, locations):
        paste(grid, patch, r, c, transparent=BACKGROUND)

    pattern_col = random.randint(0, W - pattern_shape[1])
    pattern_row = H - pattern_shape[0] - bottom_pad
    if pattern_row <= max(r + patch.shape[0] for patch, (r, _c) in zip(patches, locations)):
        return None
    paste(grid, pattern, pattern_row, pattern_col)

    if not _validate_scene(grid, pattern, unique_mask):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
