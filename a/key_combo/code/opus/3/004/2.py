from __future__ import annotations

from itertools import permutations
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
SEP = 5
SPRITE_SIZE = 5
LEGEND_BLOCK = 3
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

# A literal 5-pair legend would need width 6*5 + 4 = 34, which exceeds ARC's 30-cell cap.
N_CHOICES = (3, 4)
WIDTH_CHOICES = {3: (22, 23, 24), 4: (28, 29, 30)}
BASE_SLOTS = {
    3: [(1, 1), (1, 9), (7, 5), (7, 13)],
    4: [(1, 1), (1, 9), (1, 17), (7, 5), (7, 13), (7, 21)],
}
UPPER_BASE_H = 13
MAX_GEN_ATTEMPTS = 400


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_rows = [int(r) for r in range(grid.shape[0]) if bool((grid[r, :] == SEP).all())]
    assert len(sep_rows) == 1, "expected exactly one full gray separator row"
    r_sep = sep_rows[0]
    return grid[:r_sep, :].copy(), grid[r_sep + 1 :, :].copy()


def _extract_sprites(region: np.ndarray) -> list[dict]:
    masked = np.where((region != BG) & (region != SEP), region, BG)
    comps = find_components(masked, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    sprites: list[dict] = []
    for comp in comps:
        patch = crop_bbox(masked, comp["bbox"])
        left_rows = tuple(int(r) for r in np.where(patch[:, 0] != BG)[0])
        right_rows = tuple(int(r) for r in np.where(patch[:, -1] != BG)[0])
        sprites.append(
            {
                "bbox": tuple(int(x) for x in comp["bbox"]),
                "patch": patch,
                "area": int(comp["area"]),
                "left_rows": left_rows,
                "right_rows": right_rows,
            }
        )
    return sprites


def _extract_legend_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    masked = np.where((region != BG) & (region != SEP), region, BG)
    comps = find_components(masked, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    pairs = []
    for comp in comps:
        bbox = tuple(int(x) for x in comp["bbox"])
        patch = crop_bbox(masked, bbox)
        assert patch.shape == (LEGEND_BLOCK, LEGEND_BLOCK * 2), "legend pair must be 3x6"
        left_colors = sorted(set(int(v) for v in patch[:, :LEGEND_BLOCK].flatten()) - {BG})
        right_colors = sorted(set(int(v) for v in patch[:, LEGEND_BLOCK:].flatten()) - {BG})
        assert len(left_colors) == 1 and len(right_colors) == 1, "legend halves must be monochrome"
        pairs.append((bbox[1], left_colors[0], right_colors[0]))
    pairs.sort(key=lambda item: item[0])
    return [(src, dst) for _c0, src, dst in pairs]


def _edge_match(a: dict, b: dict) -> bool:
    return bool(set(a["right_rows"]) & set(b["left_rows"]))


def _all_chain_orders(sprites: list[dict]) -> list[tuple[int, ...]]:
    idx = list(range(len(sprites)))
    orders = []
    for perm in permutations(idx):
        if all(_edge_match(sprites[perm[i]], sprites[perm[i + 1]]) for i in range(len(perm) - 1)):
            orders.append(tuple(int(v) for v in perm))
    return orders


def _render_strip(patches: list[np.ndarray]) -> np.ndarray:
    assert patches
    heights = {int(p.shape[0]) for p in patches}
    assert len(heights) == 1, "all sprite bboxes must share the same height"
    h = next(iter(heights))
    w = sum(int(p.shape[1]) for p in patches) + len(patches) - 1
    out = np.full((h, w), BG, dtype=int)
    c = 0
    for i, patch in enumerate(patches):
        paste(out, patch, 0, c, transparent=BG)
        c += int(patch.shape[1])
        if i < len(patches) - 1:
            out[:, c] = SEP
            c += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a chain of scattered sprites using position-indexed legend mappings.

    Input structure:
    - The background is black, and a full gray row splits the grid into an upper sprite region and a lower legend region.
    - The upper region contains several separated multicolor sprite components.
    - The lower region contains the same number of 3x6 legend pairs, each made from two adjacent 3x3 monochrome blocks.

    Rule:
    - Find the upper sprite components as 4-connected non-black, non-gray regions.
    - Order those sprites by the unique edge-contact chain: sprite A can precede sprite B when A's right-edge foreground row offsets intersect B's left-edge foreground row offsets.
    - Read the legend pairs from left to right; each pair defines one color replacement from its left block color to its right block color.
    - Apply the k-th legend mapping to the k-th sprite in the chain, then pack the recolored sprites tightly from left to right with one gray separator column between neighbors.

    Key insight:
    - Edge-contact geometry determines the sprite order, and that order becomes the index used to choose which legend mapping recolors each sprite.
    """
    g = np.asarray(grid, dtype=int)
    _ = get_bg(g)
    upper, lower = _split_regions(g)
    sprites = _extract_sprites(upper)
    mappings = _extract_legend_pairs(lower)
    assert sprites and len(sprites) == len(mappings), "sprite count must match legend count"

    orders = _all_chain_orders(sprites)
    assert len(orders) == 1, "edge contacts must define a unique full chain"
    order = orders[0]

    recolored = []
    for k, sprite_idx in enumerate(order):
        patch = sprites[sprite_idx]["patch"].copy()
        src, dst = mappings[k]
        patch[patch == src] = dst
        recolored.append(patch)
    return _render_strip(recolored)


def _border_hole_free(mask: np.ndarray) -> bool:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                stack.append((r, c))
                seen[r, c] = True
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                stack.append((r, c))
                seen[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return not bool((~mask & ~seen).any())


def _build_mask(left_row: int, right_row: int) -> np.ndarray:
    top_options = {1, 2, 3}
    if left_row == 0:
        top_options &= {1, 2}
    if right_row == 0:
        top_options &= {2, 3}

    bottom_options = {1, 2, 3}
    if left_row == SPRITE_SIZE - 1:
        bottom_options &= {1, 2}
    if right_row == SPRITE_SIZE - 1:
        bottom_options &= {2, 3}

    top_col = random.choice(sorted(top_options))
    bottom_col = random.choice(sorted(bottom_options))
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)

    mask[left_row, 0] = True
    mask[left_row, 1] = True
    r0, r1 = sorted((left_row, 2))
    mask[r0 : r1 + 1, 1] = True

    mask[right_row, 4] = True
    mask[right_row, 3] = True
    r0, r1 = sorted((right_row, 2))
    mask[r0 : r1 + 1, 3] = True

    mask[0, top_col] = True
    mask[1, top_col] = True
    c0, c1 = sorted((top_col, 2))
    mask[1, c0 : c1 + 1] = True

    mask[4, bottom_col] = True
    mask[3, bottom_col] = True
    c0, c1 = sorted((bottom_col, 2))
    mask[3, c0 : c1 + 1] = True

    mask[2, 2] = True

    extra_budget = random.randint(0, 2)
    for _ in range(extra_budget):
        frontier = []
        for r in range(1, SPRITE_SIZE - 1):
            for c in range(1, SPRITE_SIZE - 1):
                if mask[r, c]:
                    continue
                if any(
                    0 <= r + dr < SPRITE_SIZE
                    and 0 <= c + dc < SPRITE_SIZE
                    and mask[r + dr, c + dc]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ):
                    frontier.append((r, c))
        random.shuffle(frontier)
        for r, c in frontier:
            trial = mask.copy()
            trial[r, c] = True
            if (
                tuple(int(v) for v in np.where(trial[:, 0])[0]) == (left_row,)
                and tuple(int(v) for v in np.where(trial[:, -1])[0]) == (right_row,)
                and int(trial.sum()) <= 15
                and _border_hole_free(trial)
            ):
                mask = trial
                break

    assert 8 <= int(mask.sum()) <= 15
    assert tuple(int(v) for v in np.where(mask[:, 0])[0]) == (left_row,)
    assert tuple(int(v) for v in np.where(mask[:, -1])[0]) == (right_row,)
    assert int(mask[0, :].sum()) <= 3 and int(mask[-1, :].sum()) <= 3
    assert int(mask[:, 0].sum()) <= 3 and int(mask[:, -1].sum()) <= 3
    assert _border_hole_free(mask)
    return mask


def _colorize_mask(mask: np.ndarray, source_color: int, target_color: int) -> np.ndarray:
    coords = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    area = len(coords)
    n_colors = random.choice((2, 3))
    others = random.sample([c for c in PALETTE if c not in {source_color, target_color}], n_colors - 1)
    max_source = area - (n_colors - 1)
    source_count = random.randint(2, min(max_source, max(2, area - 1)))

    source_cells = {(2, 2)}
    remaining_choices = [coord for coord in coords if coord != (2, 2)]
    random.shuffle(remaining_choices)
    for coord in remaining_choices:
        if len(source_cells) >= source_count:
            break
        source_cells.add(coord)

    patch = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    for r, c in source_cells:
        patch[r, c] = source_color

    leftover = [coord for coord in coords if coord not in source_cells]
    random.shuffle(leftover)
    for color, coord in zip(others, leftover):
        patch[coord] = color
    for coord in leftover[len(others) :]:
        patch[coord] = random.choice(others)

    assert source_color in set(int(v) for v in patch.flatten())
    assert target_color not in set(int(v) for v in patch.flatten())
    return patch


def _expected_output(sprites: list[np.ndarray], mappings: list[tuple[int, int]], order: list[int]) -> np.ndarray:
    recolored = []
    for k, sprite_idx in enumerate(order):
        patch = sprites[sprite_idx].copy()
        src, dst = mappings[k]
        patch[patch == src] = dst
        recolored.append(patch)
    return _render_strip(recolored)


def _legend_region(width: int, mappings: list[tuple[int, int]]) -> np.ndarray:
    region = np.full((LEGEND_BLOCK, width), BG, dtype=int)
    content_w = 7 * len(mappings) - 1
    c = random.randint(0, width - content_w)
    for i, (src, dst) in enumerate(mappings):
        region[:, c : c + 3] = src
        region[:, c + 3 : c + 6] = dst
        c += 6
        if i < len(mappings) - 1:
            region[:, c] = SEP
            c += 1
    return region


def _pick_positions(n: int, width: int, y_shift: int, x_shift: int) -> list[tuple[int, int]]:
    slots = [(r + y_shift, c + x_shift) for r, c in BASE_SLOTS[n]]
    for _ in range(200):
        positions = random.sample(slots, n)
        if len({r for r, _c in positions}) < 2:
            continue
        x_order = sorted(range(n), key=lambda i: (positions[i][1], positions[i][0]))
        row_major = sorted(range(n), key=lambda i: positions[i])
        if x_order != list(range(n)) and row_major != list(range(n)):
            return positions
    raise RuntimeError("failed to scramble sprite placement")


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    sprites: list[np.ndarray],
    positions: list[tuple[int, int]],
    mappings: list[tuple[int, int]],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    upper, lower = _split_regions(inp)
    parsed_sprites = _extract_sprites(upper)
    if len(parsed_sprites) != len(sprites):
        return False
    if len(_extract_legend_pairs(lower)) != len(mappings):
        return False

    orders = _all_chain_orders(parsed_sprites)
    if len(orders) != 1:
        return False

    areas = [int(np.count_nonzero(sprite)) for sprite in sprites]
    if len(set(areas)) != len(areas):
        return False

    x_order = sorted(range(len(sprites)), key=lambda i: (positions[i][1], positions[i][0]))
    area_order = sorted(range(len(sprites)), key=lambda i: (areas[i], positions[i][1], positions[i][0]))
    row_major = sorted(range(len(sprites)), key=lambda i: positions[i])
    if x_order == list(range(len(sprites))) or area_order == list(range(len(sprites))) or row_major == list(range(len(sprites))):
        return False

    if np.array_equal(_expected_output(sprites, mappings, x_order), expected):
        return False
    if np.array_equal(_expected_output(sprites, mappings, area_order), expected):
        return False
    if np.array_equal(_expected_output(sprites, mappings, row_major), expected):
        return False

    for sprite, (src, _dst) in zip(sprites, mappings):
        if not bool((sprite == src).any()):
            return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[int, int]], list[tuple[int, int]]]:
    n = random.choice(N_CHOICES)
    width = random.choice(WIDTH_CHOICES[n])
    y_shift = random.randint(0, 1)
    upper_h = UPPER_BASE_H + y_shift
    x_shift = random.randint(0, width - max(c + SPRITE_SIZE for _r, c in BASE_SLOTS[n]))
    positions = _pick_positions(n, width, y_shift, x_shift)

    shared_source = random.choice(PALETTE)
    targets = random.sample([c for c in PALETTE if c != shared_source], n)
    mappings = [(shared_source, target) for target in targets]

    edge_labels = random.sample(range(SPRITE_SIZE), n + 1)
    sprites: list[np.ndarray] = []
    for i in range(n):
        mask = _build_mask(edge_labels[i], edge_labels[i + 1])
        sprite = _colorize_mask(mask, shared_source, targets[i])
        sprites.append(sprite)

    grid = np.full((upper_h + 1 + LEGEND_BLOCK, width), BG, dtype=int)
    for sprite, (r, c) in zip(sprites, positions):
        paste(grid, sprite, r, c, transparent=BG)
    grid[upper_h, :] = SEP
    grid[upper_h + 1 :, :] = _legend_region(width, mappings)

    expected = _expected_output(sprites, mappings, list(range(n)))
    return grid, expected, sprites, positions, mappings


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid, expected, sprites, positions, mappings = _generate_case()
        if _semantic_checks(grid, expected, sprites, positions, mappings):
            return grid

    raise RuntimeError("generate_input failed after repeated semantic checks")
