from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 8
FRAME_COLOR = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRECTIONS = ("up", "right", "down", "left")
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    assert len(points) > 0
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.tobytes()


def _canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_trim_mask(np.rot90(mask, k))) for k in range(4))


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = _trim_mask(np.rot90(mask, k))
        key = _mask_key(rotated)
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated)
    return variants


def _rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return _trim_mask(np.rot90(mask, -turns))


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    block = np.ones((scale, scale), dtype=int)
    return np.kron(mask.astype(int), block).astype(bool)


def _direction_to_turns(direction: str) -> int:
    return DIRECTIONS.index(direction)


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    grey_mask = grid == FRAME_COLOR
    frames: list[dict] = []
    interior_union = np.zeros_like(grid, dtype=bool)

    for comp in _components(grey_mask):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        sub = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 4 or w < 4:
            continue

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.any(sub & ~border):
            continue

        missing = np.argwhere(border & ~sub)
        if len(missing) != 1:
            continue
        mr, mc = (int(x) for x in missing[0])
        side_hits = int(mr in (0, h - 1)) + int(mc in (0, w - 1))
        if side_hits != 1:
            continue

        if mr == 0:
            direction = "up"
        elif mc == w - 1:
            direction = "right"
        elif mr == h - 1:
            direction = "down"
        else:
            direction = "left"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_colors = set(int(x) for x in np.unique(interior)) - {bg, FRAME_COLOR}
        assert len(seed_colors) == 1, "frame interior must contain exactly one seed color"
        seed_color = seed_colors.pop()

        local_interior = np.zeros_like(grid, dtype=bool)
        local_interior[r0 + 1 : r1, c0 + 1 : c1] = True
        assert not np.any(interior_union & local_interior), "frame interiors must not overlap"
        interior_union |= local_interior

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "seed_color": seed_color,
                "interior_shape": interior.shape,
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert frames, "expected at least one valid grey frame"
    return frames, interior_union


def _extract_templates(grid: np.ndarray, bg: int, interior_union: np.ndarray) -> dict[int, np.ndarray]:
    outside = ~interior_union
    templates: dict[int, np.ndarray] = {}
    colors = [int(x) for x in np.unique(grid[outside]) if int(x) not in (bg, FRAME_COLOR)]

    for color in colors:
        mask = (grid == color) & outside
        comps = _components(mask)
        assert len(comps) == 1, "each template color must define exactly one external template"
        r0, c0, r1, c1 = _bbox_from_coords(comps[0])
        templates[color] = mask[r0 : r1 + 1, c0 : c1 + 1].copy()

    assert templates, "expected at least one external template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey frame with a rotated, scaled copy of its matching external template.

    Input structure:
    - The background is a dominant azure color, with several small external monochrome templates in distinct colors.
    - The grid also contains several grey rectangular frames; each frame has exactly one missing border pixel on one side.
    - Inside every frame there are 1-3 seed pixels of a single color, and that color matches exactly one external template.

    Rule:
    - Find the external template that has the same color as the frame's seed pixels.
    - Read the frame's missing grey border pixel as a direction code: up = 0 degrees, right = 90 degrees, down = 180 degrees, left = 270 degrees clockwise.
    - Rotate the matched template by that amount, scale it uniformly so its bbox fits the frame interior exactly, and repaint the whole frame interior with the seed color on template cells and background on blank cells.

    Key insight:
    - The seed color selects which template to use, while the side of the grey border gap separately selects the template rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames, interior_union = _find_frames(g, bg)
    templates = _extract_templates(g, bg, interior_union)

    out = g.copy()
    for frame in frames:
        seed_color = int(frame["seed_color"])
        assert seed_color in templates, "every frame seed color must match an external template"

        base_mask = templates[seed_color]
        rotated = _rotate_mask_cw(base_mask, _direction_to_turns(frame["direction"]))
        inner_h, inner_w = (int(x) for x in frame["interior_shape"])
        rot_h, rot_w = rotated.shape
        assert inner_h % rot_h == 0 and inner_w % rot_w == 0, "frame interior must be an integer scale of the rotated template"
        scale_h = inner_h // rot_h
        scale_w = inner_w // rot_w
        assert scale_h == scale_w and scale_h >= 1, "template scaling must be uniform"

        scaled = _scale_mask(rotated, int(scale_h))
        r0, c0, r1, c1 = frame["bbox"]
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[:, :] = bg
        region[scaled] = seed_color

    return out


def _random_template_mask(existing_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(mask.all()):
            continue
        if len(_rotation_variants(mask)) != 4:
            continue
        key = _canonical_mask_key(mask)
        if key in existing_keys:
            continue
        return mask
    return None


def _build_template_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    patch[mask] = color
    return patch


def _gap_position(direction: str, inner_h: int, inner_w: int) -> tuple[int, int]:
    outer_h, outer_w = inner_h + 2, inner_w + 2
    if direction == "up":
        return 0, random.randint(1, outer_w - 2)
    if direction == "right":
        return random.randint(1, outer_h - 2), outer_w - 1
    if direction == "down":
        return outer_h - 1, random.randint(1, outer_w - 2)
    return random.randint(1, outer_h - 2), 0


def _build_frame_patches(mask: np.ndarray, color: int, direction: str, scale: int, seed_count: int) -> tuple[np.ndarray, np.ndarray]:
    rotated = _rotate_mask_cw(mask, _direction_to_turns(direction))
    scaled = _scale_mask(rotated, scale)
    inner_h, inner_w = scaled.shape
    patch = np.full((inner_h + 2, inner_w + 2), BG_COLOR, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    gap_r, gap_c = _gap_position(direction, inner_h, inner_w)
    patch[gap_r, gap_c] = BG_COLOR

    filled_cells = [tuple(int(x) for x in pos) for pos in np.argwhere(scaled)]
    assert len(filled_cells) >= seed_count
    for sr, sc in random.sample(filled_cells, seed_count):
        patch[sr + 1, sc + 1] = color

    solved = patch.copy()
    solved[1:-1, 1:-1] = BG_COLOR
    solved[1:-1, 1:-1][scaled] = color
    return patch, solved


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch != BG_COLOR] = patch[patch != BG_COLOR]


def _choose_grid_size(objects: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(obj["input_patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["input_patch"].shape[1]) for obj in objects)
    footprint = sum((int(obj["input_patch"].shape[0]) + 2) * (int(obj["input_patch"].shape[1]) + 2) for obj in objects)

    min_h = max(16, max_h + 2)
    min_w = max(16, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(50):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        if h * w >= footprint + 40:
            return h, w
    return None


def _place_objects(objects: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    grid_in = np.full((h, w), BG_COLOR, dtype=int)
    grid_out = np.full((h, w), BG_COLOR, dtype=int)

    order = sorted(range(len(objects)), key=lambda idx: objects[idx]["input_patch"].size, reverse=True)
    for idx in order:
        obj = objects[idx]
        patch_in = obj["input_patch"]
        patch_out = obj["output_patch"]
        ph, pw = patch_in.shape
        if ph + 2 > h or pw + 2 > w:
            return None

        placed = False
        for _ in range(300):
            top = random.randint(1, h - ph - 1)
            left = random.randint(1, w - pw - 1)
            if np.any(blocked[top : top + ph, left : left + pw]):
                continue

            _draw_patch(grid_in, patch_in, top, left)
            _draw_patch(grid_out, patch_out, top, left)
            blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True
            placed = True
            break
        if not placed:
            return None

    return grid_in, grid_out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice((2, 3, 4))
    n_templates = min(len(TEMPLATE_COLORS), n_frames + random.choice((0, 1)))
    colors = random.sample(TEMPLATE_COLORS, n_templates)
    frame_colors = random.sample(colors, n_frames)

    templates: dict[int, np.ndarray] = {}
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    for color in colors:
        mask = _random_template_mask(seen_keys)
        if mask is None:
            return None
        templates[color] = mask
        seen_keys.add(_canonical_mask_key(mask))

    size_cap = {2: 14, 3: 12, 4: 10}[n_frames]
    directions = random.sample(DIRECTIONS, n_frames)
    objects: list[dict] = []

    for color, mask in templates.items():
        patch = _build_template_patch(mask, color)
        objects.append({"input_patch": patch, "output_patch": patch})

    for color, direction in zip(frame_colors, directions):
        mask = templates[color]
        rotated = _rotate_mask_cw(mask, _direction_to_turns(direction))
        rot_h, rot_w = rotated.shape
        valid_scales = [scale for scale in (2, 3, 4) if max(rot_h * scale + 2, rot_w * scale + 2) <= size_cap]
        if not valid_scales:
            return None
        scale = random.choice(valid_scales)
        seed_count = random.randint(1, 3)
        patch_in, patch_out = _build_frame_patches(mask, color, direction, scale, seed_count)
        objects.append({"input_patch": patch_in, "output_patch": patch_out})

    grid_shape = _choose_grid_size(objects)
    if grid_shape is None:
        return None
    placed = _place_objects(objects, grid_shape)
    if placed is None:
        return None
    grid_in, grid_out = placed

    try:
        solved = main(grid_in)
    except AssertionError:
        return None
    if not np.array_equal(solved, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
