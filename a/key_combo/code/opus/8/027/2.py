import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

DIVIDER = GREY
FILL_COLOR = ORANGE
SPRITE_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, MAROON]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SYMMETRIC_TEMPLATES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
]

ASYMMETRIC_TEMPLATES = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=int),
]


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def region_boxes(grid: np.ndarray, divider: int = DIVIDER) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    divider_rows = [r for r in range(h) if np.all(grid[r, :] == divider)]
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == divider)]
    row_edges = [-1, *divider_rows, h]
    col_edges = [-1, *divider_cols, w]
    boxes = []
    for i in range(len(row_edges) - 1):
        for j in range(len(col_edges) - 1):
            r0 = row_edges[i] + 1
            r1 = row_edges[i + 1] - 1
            c0 = col_edges[j] + 1
            c1 = col_edges[j + 1] - 1
            if r0 <= r1 and c0 <= c1:
                boxes.append((r0, c0, r1, c1))
    return boxes


def extract_region_sprite(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    fg_mask = (patch != BLACK) & (patch != DIVIDER)
    if not fg_mask.any():
        return None

    colors = np.unique(patch[fg_mask])
    if len(colors) != 1:
        raise AssertionError("expected one monochromatic sprite per occupied region")
    color = int(colors[0])

    parts = components_mask(patch == color)
    if len(parts) != 1:
        raise AssertionError("expected one 4-connected sprite per region")

    coords_rel = parts[0]
    br0, bc0, br1, bc1 = bbox_coords(coords_rel)
    sprite_patch = crop_bbox(patch, (br0, bc0, br1, bc1))
    coords_global = coords_rel + np.array([r0, c0], dtype=int)
    bbox_global = (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1)
    return {
        "color": color,
        "coords": coords_global,
        "bbox": bbox_global,
        "patch": sprite_patch,
    }


def has_required_symmetry(sprite_patch: np.ndarray) -> bool:
    mask = sprite_patch != BLACK
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def center_position(region_box: tuple[int, int, int, int], sprite_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = region_box
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    sprite_h, sprite_w = sprite_shape
    rr = r0 + (region_h - sprite_h) // 2
    cc = c0 + (region_w - sprite_w) // 2
    return int(rr), int(cc)


def flood_fill_region(out: np.ndarray, region_box: tuple[int, int, int, int], fill_color: int) -> None:
    r0, c0, r1, c1 = region_box
    if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
        return
    q = deque([(r0, c0)])
    out[r0, c0] = int(fill_color)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if r0 <= nr <= r1 and c0 <= nc <= c1 and out[nr, nc] == BLACK:
                out[nr, nc] = int(fill_color)
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter symmetric sprites, stamp them into empty divider-bounded regions, then flood-fill the regions left empty.

    Input structure:
    - A black background is partitioned by full gray divider rows and/or columns into 4 or 6 rectangular regions.
    - Some regions contain exactly one monochromatic 4-connected sprite, while the others are empty.
    - Sprite colors come from a non-gray palette and each occupied region uses a different color.

    Rule:
    - Crop each sprite to its bbox and keep only sprites with vertical-reflection, horizontal-reflection, or 180-degree rotational symmetry.
    - Erase every original sprite, sort surviving sprites by bbox top-left, and sort initially empty regions by region top-left.
    - Stamp the k-th surviving sprite into the center of the k-th empty region, using floor when exact centering is impossible.
    - Leave gray dividers unchanged, then flood-fill every remaining fully empty bounded region with orange.

    Key insight:
    - Symmetry controls how many sprites survive, and that count globally fixes the row-major pairing between surviving sprites and destination regions.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    boxes = region_boxes(g, divider=DIVIDER)

    sprites = []
    empty_regions = []
    for box in boxes:
        sprite = extract_region_sprite(g, box)
        if sprite is None:
            empty_regions.append(box)
        else:
            sprites.append(sprite)

    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    survivors = [sprite for sprite in sprites if has_required_symmetry(sprite["patch"])]
    survivors.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    empty_regions.sort(key=lambda box: (box[0], box[1]))

    for sprite, box in zip(survivors, empty_regions):
        rr, cc = center_position(box, sprite["patch"].shape)
        paste(out, sprite["patch"], rr, cc, transparent=BLACK)

    for box in boxes:
        flood_fill_region(out, box, FILL_COLOR)

    return out


def transformed_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    unique = []
    seen = set()
    for variant in variants:
        key = (variant.shape, variant.tobytes())
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def choose_template(templates: list[np.ndarray], max_h: int, max_w: int, *, require_symmetric: bool) -> np.ndarray | None:
    choices = []
    for template in templates:
        for variant in transformed_variants(template):
            h, w = variant.shape
            if h >= max_h or w >= max_w:
                continue
            if require_symmetric and not has_required_symmetry(variant):
                continue
            if not require_symmetric and has_required_symmetry(variant):
                continue
            area = int(variant.sum())
            if 3 <= area <= 8:
                choices.append(variant)
    if not choices:
        return None
    return random.choice(choices).copy()


def random_partition_sizes(n_segments: int) -> list[int]:
    if n_segments == 2:
        base = [random.randint(7, 11), random.randint(7, 11)]
    else:
        base = [random.randint(5, 8) for _ in range(3)]
    return base


def draw_dividers(grid: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    if grid.shape != (h, w):
        raise AssertionError("grid shape mismatch")

    acc = row_sizes[0]
    for size in row_sizes[1:]:
        grid[acc, :] = DIVIDER
        acc += 1 + size

    acc = col_sizes[0]
    for size in col_sizes[1:]:
        grid[:, acc] = DIVIDER
        acc += 1 + size

    return region_boxes(grid, divider=DIVIDER)


def paint_sprite(grid: np.ndarray, region_box: tuple[int, int, int, int], shape: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = region_box
    h, w = shape.shape
    max_r = r1 - h + 1
    max_c = c1 - w + 1
    rr = random.randint(r0, max_r)
    cc = random.randint(c0, max_c)
    patch = np.where(shape == 1, int(color), BLACK)
    paste(grid, patch, rr, cc, transparent=BLACK)


def expected_region_after_stamp(region_box: tuple[int, int, int, int], sprite_patch: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = region_box
    region = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rr, cc = center_position((0, 0, r1 - r0, c1 - c0), sprite_patch.shape)
    paste(region, sprite_patch, rr, cc, transparent=BLACK)
    return region


def validate_generated_case(inp: np.ndarray) -> bool:
    boxes = region_boxes(inp, divider=DIVIDER)
    if len(boxes) not in {4, 6}:
        return False

    sprites = []
    empty_boxes = []
    colors = set()
    min_empty_h = min(box[2] - box[0] + 1 for box in boxes)
    min_empty_w = min(box[3] - box[1] + 1 for box in boxes)
    for box in boxes:
        sprite = extract_region_sprite(inp, box)
        if sprite is None:
            empty_boxes.append(box)
            continue
        if sprite["color"] in colors or sprite["color"] not in SPRITE_COLORS:
            return False
        colors.add(sprite["color"])
        sh, sw = sprite["patch"].shape
        region_h = box[2] - box[0] + 1
        region_w = box[3] - box[1] + 1
        if not (sh < region_h and sw < region_w and sh < min_empty_h and sw < min_empty_w):
            return False
        sprites.append(sprite)

    if not empty_boxes:
        return False

    survivors = [sprite for sprite in sprites if has_required_symmetry(sprite["patch"])]
    if len(survivors) != len(empty_boxes):
        return False
    if len(sprites) <= len(empty_boxes):
        return False
    if len(survivors) == 0 or len(survivors) == len(sprites):
        return False

    out = main(inp)
    survivors.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    empty_boxes.sort(key=lambda box: (box[0], box[1]))
    target_boxes = {box for box in empty_boxes[: len(survivors)]}
    expected_targets = {
        box: expected_region_after_stamp(box, sprite["patch"])
        for sprite, box in zip(survivors, empty_boxes)
    }

    for box in boxes:
        r0, c0, r1, c1 = box
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if box in target_boxes:
            if not np.array_equal(patch, expected_targets[box]):
                return False
        else:
            if not np.all(patch == FILL_COLOR):
                return False

    return True


def gen_attempt() -> np.ndarray | None:
    row_segments, col_segments = random.choice([(2, 2), (2, 3), (3, 2)])
    if row_segments * col_segments == 6:
        empty_count = 2 if random.random() < 0.8 else 1
    else:
        empty_count = 1

    row_sizes = random_partition_sizes(row_segments)
    col_sizes = random_partition_sizes(col_segments)
    h = sum(row_sizes) + row_segments - 1
    w = sum(col_sizes) + col_segments - 1
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    boxes = draw_dividers(grid, row_sizes, col_sizes)
    if len(boxes) != row_segments * col_segments:
        return None

    occupied_count = len(boxes) - empty_count
    if occupied_count <= empty_count or occupied_count > len(SPRITE_COLORS):
        return None

    min_h = min(box[2] - box[0] + 1 for box in boxes)
    min_w = min(box[3] - box[1] + 1 for box in boxes)
    symmetric_shape = choose_template(SYMMETRIC_TEMPLATES, min_h, min_w, require_symmetric=True)
    asymmetric_shape = choose_template(ASYMMETRIC_TEMPLATES, min_h, min_w, require_symmetric=False)
    if symmetric_shape is None or asymmetric_shape is None:
        return None

    empty_indices = set(random.sample(range(len(boxes)), empty_count))
    occupied_indices = [idx for idx in range(len(boxes)) if idx not in empty_indices]
    symmetric_indices = set(random.sample(occupied_indices, empty_count))

    colors = random.sample(SPRITE_COLORS, occupied_count)
    color_iter = iter(colors)
    for idx, box in enumerate(boxes):
        if idx in empty_indices:
            continue
        shape = symmetric_shape if idx in symmetric_indices else asymmetric_shape
        paint_sprite(grid, box, shape, next(color_iter))

    if validate_generated_case(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
