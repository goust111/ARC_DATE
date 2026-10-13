from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 5
MOVE_BY_COLOR = {
    2: (0, 2),   # red -> right 2
    1: (2, 0),   # blue -> down 2
    3: (0, -2),  # green -> left 2
    7: (-2, 0),  # orange -> up 2
}
OUTPUT_BY_KIND = {
    "H": 4,  # horizontal only
    "V": 6,  # vertical only
    "B": 8,  # both horizontal and vertical
    "N": 9,  # no reflection symmetry
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(sorted(coords), dtype=int))

    components.sort(key=lambda coords: tuple(coords.min(axis=0)))
    return components


def _monochrome_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_from_mask(grid == color):
            sprites.append({"color": int(color), "coords": coords})
    sprites.sort(key=lambda item: (item["coords"][:, 0].min(), item["coords"][:, 1].min(), item["color"]))
    return sprites


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and vertical:
        return "B"
    if horizontal:
        return "H"
    if vertical:
        return "V"
    return "N"


def _translate_coords(coords: np.ndarray, color: int) -> np.ndarray:
    dr, dc = MOVE_BY_COLOR[int(color)]
    return coords + np.array([dr, dc], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored sprite, merge the translated collisions, and recolor each merged region by its reflection symmetry.

    Input structure:
    - The grid is a grey-background canvas containing 4-connected single-color foreground sprites.
    - Foreground colors are motion codes: red means shift right by 2, blue means shift down by 2,
      green means shift left by 2, and orange means shift up by 2.
    - Each sprite has 2-4 cells, and sprites are interpreted independently before any movement.

    Rule:
    - Clear the output to the background color.
    - Translate every monochromatic sprite by the offset encoded by its color.
    - Ignore original sprite boundaries after translation: take the union of all translated foreground
      cells and split that union into 4-connected regions.
    - Recolor each translated region uniformly by its reflection symmetry: horizontal-only -> yellow,
      vertical-only -> magenta, both horizontal and vertical -> teal, none -> maroon.

    Key insight:
    - The output color is determined by an emergent post-translation shape, not by any input color:
      movement creates collision-based merged regions whose symmetry class decides the final recolor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    translated = np.zeros((h, w), dtype=bool)
    for sprite in _monochrome_sprites(g, bg):
        color = int(sprite["color"])
        assert color in MOVE_BY_COLOR, "unexpected foreground color"
        moved = _translate_coords(sprite["coords"], color)
        assert moved.size > 0
        assert np.all(0 <= moved[:, 0]) and np.all(moved[:, 0] < h)
        assert np.all(0 <= moved[:, 1]) and np.all(moved[:, 1] < w)
        translated[moved[:, 0], moved[:, 1]] = True

    out = np.full((h, w), int(bg), dtype=int)
    for coords in _components_from_mask(translated):
        out[coords[:, 0], coords[:, 1]] = OUTPUT_BY_KIND[_symmetry_kind(_tight_mask(coords))]
    return out


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _apply_transform(mask: np.ndarray, code: str) -> np.ndarray:
    if code == "id":
        return mask.copy()
    if code == "rot90":
        return np.rot90(mask, 1)
    if code == "rot180":
        return np.rot90(mask, 2)
    if code == "rot270":
        return np.rot90(mask, 3)
    if code == "flip_lr":
        return np.fliplr(mask)
    if code == "flip_ud":
        return np.flipud(mask)
    if code == "diag":
        return np.fliplr(np.rot90(mask, 1))
    if code == "anti":
        return np.flipud(np.rot90(mask, 1))
    raise ValueError(f"unknown transform {code}")


def _variant_catalog() -> dict[str, list[dict]]:
    transforms = ("id", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "diag", "anti")
    prototypes = [
        [
            _mask_from_coords((3, 2), [(0, 0), (0, 1), (1, 1)]),
            _mask_from_coords((3, 2), [(1, 1), (2, 0), (2, 1)]),
        ],
        [
            _mask_from_coords((2, 2), [(0, 0), (0, 1), (1, 0)]),
            _mask_from_coords((2, 2), [(0, 1), (1, 0), (1, 1)]),
        ],
        [
            _mask_from_coords((3, 3), [(0, 0), (0, 1), (1, 1)]),
            _mask_from_coords((3, 3), [(1, 1), (2, 1), (2, 2)]),
        ],
    ]

    catalog = {key: [] for key in OUTPUT_BY_KIND}
    seen = {key: set() for key in OUTPUT_BY_KIND}
    for prototype in prototypes:
        for code in transforms:
            pieces = [_apply_transform(piece, code) for piece in prototype]
            union = np.zeros_like(pieces[0], dtype=bool)
            for piece in pieces:
                union |= piece
            kind = _symmetry_kind(union)
            piece_coords = [np.argwhere(piece).astype(int) for piece in pieces]
            signature = tuple(tuple(map(tuple, coords.tolist())) for coords in piece_coords)
            if signature in seen[kind]:
                continue
            seen[kind].add(signature)
            catalog[kind].append(
                {
                    "shape": union.shape,
                    "pieces": piece_coords,
                    "union": np.argwhere(union).astype(int),
                }
            )
    return catalog


REGION_VARIANTS = _variant_catalog()


def _ordered_color_pairs(region_count: int) -> list[tuple[int, int]]:
    colors = list(MOVE_BY_COLOR)
    if region_count == 2:
        first = tuple(random.sample(colors, 2))
        remaining = [color for color in colors if color not in first]
        second = tuple(random.sample(remaining, 2))
        pairs = [first, second]
    else:
        pairs = []
        while len(pairs) < 3:
            pair = tuple(random.sample(colors, 2))
            if pair[0] != pair[1]:
                pairs.append(pair)
        if set(color for pair in pairs for color in pair) != set(colors):
            return _ordered_color_pairs(region_count)
    random.shuffle(pairs)
    return pairs


def _min_l1(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2).min())


def _disjoint_with_gap(coords: np.ndarray, others: list[np.ndarray], gap: int = 1) -> bool:
    return all(_min_l1(coords, other) > gap for other in others)


def _in_bounds(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.all(0 <= coords[:, 0])
        and np.all(coords[:, 0] < height)
        and np.all(0 <= coords[:, 1])
        and np.all(coords[:, 1] < width)
    )


def _try_place_region(
    height: int,
    width: int,
    kind: str,
    base_pair: tuple[int, int],
    source_groups: list[np.ndarray],
    final_regions: list[np.ndarray],
) -> list[tuple[int, np.ndarray]] | None:
    variants = REGION_VARIANTS[kind][:]
    random.shuffle(variants)

    for variant in variants:
        vh, vw = variant["shape"]
        anchors = [(r0, c0) for r0 in range(height - vh + 1) for c0 in range(width - vw + 1)]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            union = variant["union"] + np.array([r0, c0], dtype=int)
            if not _disjoint_with_gap(union, final_regions, gap=1):
                continue

            translated_pieces = [piece + np.array([r0, c0], dtype=int) for piece in variant["pieces"]]
            for colors in (base_pair, base_pair[::-1]):
                source_pieces = []
                for piece, color in zip(translated_pieces, colors):
                    dr, dc = MOVE_BY_COLOR[color]
                    source = piece - np.array([dr, dc], dtype=int)
                    if not _in_bounds(source, height, width):
                        break
                    source_pieces.append((int(color), source))
                else:
                    candidate_sources = [coords for _color, coords in source_pieces]
                    if not _disjoint_with_gap(candidate_sources[0], candidate_sources[1:], gap=1):
                        continue
                    if not all(_disjoint_with_gap(coords, source_groups, gap=1) for coords in candidate_sources):
                        continue
                    return source_pieces
    return None


def _render_input(height: int, width: int, source_pieces: list[tuple[int, np.ndarray]]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for color, coords in source_pieces:
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _recolor_without_translation(grid: np.ndarray) -> np.ndarray:
    bg = _background_color(grid)
    out = np.full_like(grid, bg)
    for sprite in _monochrome_sprites(grid, bg):
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = OUTPUT_BY_KIND[_symmetry_kind(_tight_mask(sprite["coords"]))]
    return out


def _semantic_ok(grid: np.ndarray, expected_region_count: int) -> bool:
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False
    if _background_color(grid) != BACKGROUND:
        return False

    sprites = _monochrome_sprites(grid, BACKGROUND)
    if not (4 <= len(sprites) <= 6):
        return False
    if set(int(sprite["color"]) for sprite in sprites) != set(MOVE_BY_COLOR):
        return False
    for sprite in sprites:
        area = int(len(sprite["coords"]))
        if not (2 <= area <= 4):
            return False

    translated_groups = [_translate_coords(sprite["coords"], sprite["color"]) for sprite in sprites]
    translated_mask = np.zeros_like(grid, dtype=bool)
    for coords in translated_groups:
        if not _in_bounds(coords, *grid.shape):
            return False
        translated_mask[coords[:, 0], coords[:, 1]] = True

    regions = _components_from_mask(translated_mask)
    if len(regions) != expected_region_count:
        return False

    region_types = [_symmetry_kind(_tight_mask(region)) for region in regions]
    contributor_counts = []
    for region in regions:
        region_set = {tuple(cell) for cell in region.tolist()}
        count = 0
        for piece in translated_groups:
            if any((int(r), int(c)) in region_set for r, c in piece):
                count += 1
        contributor_counts.append(count)

    merged_types = [kind for kind, count in zip(region_types, contributor_counts) if count >= 2]
    if len(merged_types) < 2 or len(set(merged_types)) < 2:
        return False

    solved = main(grid)
    if np.array_equal(solved, _recolor_without_translation(grid)):
        return False
    if len(set(map(int, np.unique(solved))) - {BACKGROUND}) < 2:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    region_count = random.choice((2, 3))
    kinds = random.sample(list(OUTPUT_BY_KIND), region_count)
    color_pairs = _ordered_color_pairs(region_count)

    source_pieces: list[tuple[int, np.ndarray]] = []
    source_groups: list[np.ndarray] = []
    final_regions: list[np.ndarray] = []

    specs = list(zip(kinds, color_pairs))
    specs.sort(key=lambda item: {"N": 0, "H": 1, "V": 1, "B": 2}[item[0]])
    for kind, pair in specs:
        placed = _try_place_region(height, width, kind, pair, source_groups, final_regions)
        if placed is None:
            return None
        for color, coords in placed:
            source_pieces.append((color, coords))
            source_groups.append(coords)
            final_regions.append(_translate_coords(coords, color))

    grid = _render_input(height, width, source_pieces)
    if not _semantic_ok(grid, expected_region_count=region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
