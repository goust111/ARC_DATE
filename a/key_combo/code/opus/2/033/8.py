from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
SPRITE_SHAPE = (3, 3)
GEN_GRID_MIN = 12
GEN_GRID_MAX = 18
GEN_RETRY_LIMIT = 400
GEN_COUNT_OPTIONS = (
    (3, 2),
    (4, 2),
    (3, 2, 1),
    (4, 3),
    (4, 2, 1),
    (5, 2),
    (5, 3),
    (4, 3, 1),
    (5, 2, 1),
    (4, 2, 1, 1),
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield int(r + dr), int(c + dc)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components_from_mask(grid == int(color)):
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return components


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _sprite_from_mask(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    sprite = np.full(SPRITE_SHAPE, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _reflect_four_way(sprite: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [np.flipud(np.fliplr(sprite)), np.flipud(sprite)],
            [np.fliplr(sprite), sprite],
        ]
    )


def _enclosed_background_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for coords in _components_from_mask(grid == int(bg)):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            int(rows.min()) == 0
            or int(rows.max()) == grid.shape[0] - 1
            or int(cols.min()) == 0
            or int(cols.max()) == grid.shape[1] - 1
        ):
            continue
        holes.append(coords)
    return holes


def _fill_holes(grid: np.ndarray, fill_color: int, bg: int) -> np.ndarray:
    out = grid.copy()
    for coords in _enclosed_background_components(out, bg):
        out[coords[:, 0], coords[:, 1]] = int(fill_color)
    return out


def _analyze_sprite_types(grid: np.ndarray) -> tuple[int, Counter, dict]:
    bg = _get_bg(grid)
    counts: Counter = Counter()
    sprites: dict = {}
    for component in _find_monochrome_components(grid, bg):
        sprite = _crop_bbox(grid, component["bbox"])
        assert sprite.shape == SPRITE_SHAPE, "each sprite must crop to 3x3"
        mask = sprite != int(bg)
        key = (int(component["color"]), _mask_key(mask))
        counts[key] += 1
        sprites[key] = sprite
    assert counts, "expected at least one foreground sprite"
    return bg, counts, sprites


def _winning_types(counts: Counter) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], tuple[int, tuple[tuple[int, ...], ...]]]:
    top_count = max(counts.values())
    top_keys = [key for key, count in counts.items() if int(count) == int(top_count)]
    assert len(top_keys) == 1, "expected a unique most frequent sprite type"

    lower_counts = [int(count) for count in counts.values() if int(count) < int(top_count)]
    assert lower_counts, "expected at least two sprite types"
    second_count = max(lower_counts)
    second_keys = [key for key, count in counts.items() if int(count) == int(second_count)]
    assert len(second_keys) == 1, "expected a unique second-most frequent sprite type"
    return top_keys[0], second_keys[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the most frequent 3x3 sprite into four quadrants, then fill the new enclosed holes with the second-most frequent sprite color.

    Input structure:
    - A black background contains 5-8 separated monochrome sprites.
    - Each sprite is a 4-connected shape whose bounding box is exactly 3x3.
    - Sprite types are defined by the 3x3 occupancy pattern together with the sprite color.

    Rule:
    - Count every sprite type and select the unique most frequent type as the seed.
    - Build a 6x6 output by placing that seed in the lower-right quadrant and its horizontal, vertical, and double reflections in the other three quadrants.
    - Find every 4-connected background region in the 6x6 result that does not touch the border, and fill those hole cells with the color of the unique second-most frequent sprite type.

    Key insight:
    - Frequency determines two different roles at once: the winner chooses the mirrored shape, while the runner-up supplies the hole-fill color created only after the four-way expansion.
    """
    g = np.asarray(grid, dtype=int)
    bg, counts, sprites = _analyze_sprite_types(g)
    seed_key, fill_key = _winning_types(counts)
    seed_sprite = sprites[seed_key]
    fill_color = int(fill_key[0])

    mirrored = _reflect_four_way(seed_sprite)
    return _fill_holes(mirrored, fill_color, bg)


def _is_valid_bbox_mask(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _enumerate_masks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_masks: list[np.ndarray] = []
    seed_masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> k) & 1 for k in range(9)], dtype=bool).reshape(SPRITE_SHAPE)
        if not _is_valid_bbox_mask(mask):
            continue
        if len(_components_from_mask(mask)) != 1:
            continue
        all_masks.append(mask.copy())

        if not (5 <= int(mask.sum()) <= 7):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue

        mirrored = _reflect_four_way(mask.astype(int))
        holes = _enclosed_background_components(mirrored, 0)
        hole_area = sum(int(len(coords)) for coords in holes)
        if holes and len(holes) <= 2 and hole_area <= 8:
            seed_masks.append(mask.copy())

    return all_masks, seed_masks


ALL_MASKS, SEED_MASKS = _enumerate_masks()
assert SEED_MASKS, "need at least one valid seed mask"


def _grid_positions(height: int, width: int) -> list[Coord]:
    rows = list(range(1, int(height) - 2, 4))
    cols = list(range(1, int(width) - 2, 4))
    return [(int(r), int(c)) for r in rows for c in cols]


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    patch = grid[int(r0) : int(r0) + 3, int(c0) : int(c0) + 3]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _sample_type_specs() -> list[tuple[np.ndarray, int, int]]:
    counts = list(random.choice(GEN_COUNT_OPTIONS))
    seed_color, fill_color = random.sample(range(1, 10), 2)

    seed_mask = random.choice(SEED_MASKS).copy()
    remaining_masks = [mask.copy() for mask in ALL_MASKS if _mask_key(mask) != _mask_key(seed_mask)]
    random.shuffle(remaining_masks)

    masks = [seed_mask]
    while len(masks) < len(counts):
        masks.append(remaining_masks.pop())

    colors = [seed_color, fill_color]
    colors.extend(random.choice((seed_color, fill_color)) for _ in range(len(counts) - 2))
    return [(masks[i], int(colors[i]), int(counts[i])) for i in range(len(counts))]


def _expected_output(seed_mask: np.ndarray, seed_color: int, fill_color: int) -> np.ndarray:
    seed_sprite = _sprite_from_mask(seed_mask, seed_color, BG)
    return _fill_holes(_reflect_four_way(seed_sprite), fill_color, BG)


def _generation_valid(grid: np.ndarray, specs: list[tuple[np.ndarray, int, int]]) -> bool:
    try:
        if grid.ndim != 2 or max(grid.shape) > 30:
            return False
        if len(set(map(int, grid.ravel())) - {BG}) != 2:
            return False

        bg, counts, _sprites = _analyze_sprite_types(grid)
        if int(bg) != BG:
            return False
        if sum(counts.values()) < 5 or sum(counts.values()) > 8:
            return False

        seed_key, fill_key = _winning_types(counts)
        seed_mask, seed_color, _seed_count = specs[0]
        _fill_mask, fill_color, _fill_count = specs[1]
        if seed_key != (int(seed_color), _mask_key(seed_mask)):
            return False
        if int(fill_key[0]) != int(fill_color):
            return False

        expected = _expected_output(seed_mask, seed_color, fill_color)
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_type_specs()
    total_objects = sum(int(count) for _mask, _color, count in specs)

    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    positions = _grid_positions(height, width)
    if len(positions) < total_objects:
        return None

    grid = np.zeros((height, width), dtype=int)
    instance_types = [idx for idx, (_mask, _color, count) in enumerate(specs) for _ in range(int(count))]
    random.shuffle(instance_types)
    chosen_positions = random.sample(positions, total_objects)

    for type_idx, (r0, c0) in zip(instance_types, chosen_positions):
        mask, color, _count = specs[type_idx]
        _paste_sprite(grid, _sprite_from_mask(mask, color, BG), r0, c0)

    if _generation_valid(grid, specs):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
