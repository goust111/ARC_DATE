from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
ANCHOR_COLORS = (RED, GREEN, YELLOW)
ANCHOR_PLACEHOLDER = 9

GEN_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch)
    for rr, cc in zip(rows, cols):
        dst[top + int(rr), left + int(cc)] = int(patch[int(rr), int(cc)])


def _extract_sprite(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    values = patch[patch != 0]
    counts = Counter(map(int, values))
    anchor_color = min(counts, key=lambda color: (counts[color], color))
    anchor_pos = tuple(map(int, np.argwhere(patch == anchor_color)[0]))
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "anchor_color": int(anchor_color),
        "anchor_pos": anchor_pos,
    }


def _background_components(canvas: np.ndarray) -> list[np.ndarray]:
    mask = (canvas == 0) | (canvas == ORANGE)
    return [comp for comp in _components(mask) if not _touches_border(comp, canvas.shape)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp color-matched source sprites into the right half, then fill the enclosed background pockets with orange.

    Input structure:
    - A black grid is split by one full grey separator column into a left source area and a right target area.
    - The left area contains several 4-connected blue sprites, each with exactly one rarer anchor pixel in red, green, or yellow.
    - The right area contains one singleton marker of each anchor color plus one extra orange singleton on the black background.

    Rule:
    - Extract each left-side sprite as a template and identify its anchor by taking the least frequent non-black color inside that sprite.
    - In source reading order (top-to-bottom, then left-to-right), translate each sprite so its anchor lands on the same-colored singleton marker in the right half; later stamps overwrite earlier ones where they overlap.
    - Clear the source side conceptually by returning only the right half; after stamping, find every 4-connected enclosed background region made of black cells (allowing the pre-existing orange marker to sit inside such a region) and fill each enclosed region with orange.

    Key insight:
    - The target fill regions do not exist in the input by themselves; they emerge only after the color-keyed sprite stamps jointly close off background pockets in the destination area.
    """
    g = np.asarray(grid, dtype=int)
    sep_cols = np.where(np.all(g == GREY, axis=0))[0]
    assert len(sep_cols) >= 1, "expected a full grey separator column"
    sep = int(sep_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]

    sprite_coords = _components(left != 0)
    sprites = [_extract_sprite(left, coords) for coords in sprite_coords]
    sprites.sort(key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1]))

    marker_positions: dict[int, tuple[int, int]] = {}
    for color in ANCHOR_COLORS:
        pts = np.argwhere(right == color)
        if len(pts) == 1:
            marker_positions[int(color)] = tuple(map(int, pts[0]))
    orange_pts = np.argwhere(right == ORANGE)
    orange_pos = tuple(map(int, orange_pts[0])) if len(orange_pts) == 1 else None

    out = np.zeros_like(right)
    for sprite in sprites:
        color = int(sprite["anchor_color"])
        if color not in marker_positions:
            continue
        mr, mc = marker_positions[color]
        ar, ac = sprite["anchor_pos"]
        top = int(mr - ar)
        left_col = int(mc - ac)
        _paste(out, sprite["patch"], top, left_col)

    if orange_pos is not None and out[orange_pos] == 0:
        out[orange_pos] = ORANGE

    enclosed_regions = _background_components(out)
    enclosed_mask = np.zeros(out.shape, dtype=bool)
    for comp in enclosed_regions:
        enclosed_mask[comp[:, 0], comp[:, 1]] = True
    out[enclosed_mask] = ORANGE
    out[(out == ORANGE) & ~enclosed_mask] = 0
    return out


def _transform(arr: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr.copy()
    return np.rot90(out, rot_k)


def _embedded_patch(local_patch: np.ndarray, shape: tuple[int, int], top: int, left: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    h, w = local_patch.shape
    out[top : top + h, left : left + w] = local_patch
    return out


def _base_layers(kind: str) -> list[np.ndarray]:
    shape = (7, 9)
    patch_a = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 9, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    patch_b = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 9, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    )
    patch_c_one = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 9, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    patch_c_two = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 9, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    patch_c = patch_c_one if kind == "one" else patch_c_two
    return [
        _embedded_patch(patch_a, shape, 0, 0),
        _embedded_patch(patch_b, shape, 0, 4),
        _embedded_patch(patch_c, shape, 2, 2),
    ]


def _layer_to_sprite(layer: np.ndarray, anchor_color: int) -> dict:
    mask = layer != 0
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    patch = layer[r0 : r1 + 1, c0 : c1 + 1].copy()
    anchor = tuple(map(int, np.argwhere(patch == ANCHOR_PLACEHOLDER)[0]))
    patch[patch == 1] = BLUE
    patch[patch == ANCHOR_PLACEHOLDER] = int(anchor_color)
    return {"patch": patch, "anchor_pos": anchor, "offset": (r0, c0)}


def _place_source_sprites(
    rng: random.Random,
    height: int,
    left_width: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    taken_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed_here = None
        for _ in range(120):
            top = rng.randint(1, height - ph - 1)
            left = rng.randint(1, left_width - pw - 1)
            box = (top, left, top + ph - 1, left + pw - 1)
            ok = True
            for r0, c0, r1, c1 in taken_boxes:
                if not (box[2] + 1 < r0 or r1 + 1 < box[0] or box[3] + 1 < c0 or c1 + 1 < box[1]):
                    ok = False
                    break
            if ok:
                taken_boxes.append(box)
                placed_here = (top, left)
                break
        if placed_here is None:
            raise ValueError("source placement failed")
        placed[idx] = placed_here
    for idx in range(len(patches)):
        placements.append(placed[idx])
    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    kind = rng.choice(["one", "two"])
    rot_k = rng.randrange(4)
    flip_lr = bool(rng.randrange(2))

    anchor_colors = list(ANCHOR_COLORS)
    rng.shuffle(anchor_colors)

    transformed_layers = [_transform(layer, rot_k, flip_lr) for layer in _base_layers(kind)]
    sprites = [_layer_to_sprite(layer, color) for layer, color in zip(transformed_layers, anchor_colors)]

    arrangement_h, arrangement_w = transformed_layers[0].shape
    min_height = max(18, arrangement_h + 4)
    height = rng.randint(min_height, 24)
    left_width = rng.randint(10, 13)
    right_width_min = max(11, arrangement_w + 2)
    right_width_max = min(15, 29 - left_width)
    if right_width_min > right_width_max:
        return None
    right_width = rng.randint(right_width_min, right_width_max)

    if height - arrangement_h - 1 < 1 or right_width - arrangement_w - 1 < 1:
        return None
    right_top = rng.randint(1, height - arrangement_h - 1)
    right_left = rng.randint(1, right_width - arrangement_w - 1)

    stamped_right = np.zeros((height, right_width), dtype=int)
    marker_positions: list[tuple[int, int]] = []
    for sprite in sprites:
        patch = sprite["patch"]
        off_r, off_c = sprite["offset"]
        top = right_top + off_r
        left = right_left + off_c
        _paste(stamped_right, patch, top, left)
        ar, ac = sprite["anchor_pos"]
        marker_positions.append((top + ar, left + ac))

    holes = _background_components(stamped_right)
    expected_holes = 1 if kind == "one" else 2
    if len(holes) != expected_holes:
        return None
    orange_region = rng.choice(holes)
    orange_r, orange_c = map(int, orange_region[rng.randrange(len(orange_region))])

    patches = [sprite["patch"] for sprite in sprites]
    try:
        source_positions = _place_source_sprites(rng, height, left_width, patches)
    except ValueError:
        return None

    grid = np.zeros((height, left_width + 1 + right_width), dtype=int)
    grid[:, left_width] = GREY
    for patch, (top, left) in zip(patches, source_positions):
        _paste(grid[:, :left_width], patch, top, left)

    right = grid[:, left_width + 1 :]
    for color, (mr, mc) in zip(anchor_colors, marker_positions):
        right[mr, mc] = int(color)
    right[orange_r, orange_c] = ORANGE

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != (height, right_width):
        return None
    if np.count_nonzero(right == ORANGE) != 1:
        return None
    if np.count_nonzero(out == ORANGE) < sum(len(comp) for comp in holes):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")
