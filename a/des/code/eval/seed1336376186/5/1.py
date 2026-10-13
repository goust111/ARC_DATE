from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
ARROW_TEMPLATES = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
GEN_ATTEMPTS = 400
SPRITE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(len(coords)),
                }
            )
    return comps


def _coord_set(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _arrow_direction(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "arrow bbox must be 3x3"
    local = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    for direction, template in ARROW_TEMPLATES.items():
        if local == template:
            return direction
    raise AssertionError("expected one of the four 5-pixel orthogonal arrow templates")


def _frame_perimeter(r0: int, c0: int, r1: int, c1: int) -> frozenset[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return frozenset(cells)


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frame_infos: list[dict] = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    for color in colors:
        for comp in _components_of_color(g, color):
            r0, c0, r1, c1 = comp["bbox"]
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            if comp["area"] != 2 * h + 2 * w - 4:
                continue
            if _coord_set(comp["coords"]) != _frame_perimeter(r0, c0, r1, c1):
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior == color):
                continue

            inner_colors = [int(v) for v in np.unique(interior) if int(v) != bg]
            if len(inner_colors) != 1:
                continue

            arrow_color = inner_colors[0]
            arrow_local = np.argwhere(interior == arrow_color)
            if arrow_local.shape[0] != 5:
                continue

            arrow_abs = arrow_local + np.array([r0 + 1, c0 + 1], dtype=int)
            if len(_components_of_color(interior, arrow_color)) != 1:
                continue

            try:
                direction = _arrow_direction(arrow_abs)
            except AssertionError:
                continue

            frame_infos.append(
                {
                    "frame_color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "arrow_color": int(arrow_color),
                    "arrow_coords": arrow_abs,
                    "direction": direction,
                }
            )

    assert frame_infos, "expected at least one valid frame"
    frame_colors = {info["frame_color"] for info in frame_infos}
    assert len(frame_colors) == 1, "all frames must share one border color"
    arrow_colors = [info["arrow_color"] for info in frame_infos]
    assert len(set(arrow_colors)) == len(arrow_colors), "arrow colors must be unique"
    return frame_infos


def _mirror_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction in ("left", "right"):
        return np.fliplr(mask)
    return np.flipud(mask)


def _destination_top_left(
    frame_bbox: tuple[int, int, int, int], shape: tuple[int, int], direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    sh, sw = shape

    if direction in ("left", "right"):
        numerator = r0 + r1 - (sh - 1)
        assert numerator % 2 == 0, "frame/sprite row-center parity must match"
        dest_r = numerator // 2
        dest_c = c0 - sw if direction == "left" else c1 + 1
        return int(dest_r), int(dest_c)

    numerator = c0 + c1 - (sw - 1)
    assert numerator % 2 == 0, "frame/sprite col-center parity must match"
    dest_r = r0 - sh if direction == "up" else r1 + 1
    dest_c = numerator // 2
    return int(dest_r), int(dest_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome sprite to the outside of the frame whose internal arrow shares its color.

    Input structure:
    - The background is the most common color.
    - Several same-colored hollow rectangular frames each contain one 5-pixel arrow.
    - Every arrow color appears exactly once again as one disconnected monochrome sprite.

    Rule:
    - Match each sprite to the frame whose arrow has the same color.
    - Mirror the sprite across the axis perpendicular to the arrow direction.
    - Place it flush against the indicated exterior side, with its bbox center aligned to the frame-side midpoint.
    - Erase the original sprite; keep the frames and arrows unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _detect_frames(g, bg)

    components_by_color = {
        color: _components_of_color(g, color)
        for color in [int(c) for c in np.unique(g) if int(c) != bg]
    }

    sprites: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["arrow_color"])
        arrow_cells = _coord_set(frame["arrow_coords"])
        candidates = [
            comp
            for comp in components_by_color[color]
            if _coord_set(comp["coords"]) != arrow_cells
        ]
        assert len(candidates) == 1, "each arrow color must match exactly one sprite"
        sprites[color] = candidates[0]

    out = g.copy()
    for sprite in sprites.values():
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    h, w = g.shape
    for frame in frames:
        color = int(frame["arrow_color"])
        sprite = sprites[color]
        mask = _component_mask(sprite)
        moved = _mirror_mask(mask, frame["direction"])
        dest_r, dest_c = _destination_top_left(frame["bbox"], moved.shape, frame["direction"])
        placed = np.argwhere(moved)
        placed[:, 0] += dest_r
        placed[:, 1] += dest_c

        assert placed.size > 0
        assert placed[:, 0].min() >= 0 and placed[:, 0].max() < h
        assert placed[:, 1].min() >= 0 and placed[:, 1].max() < w
        assert np.all(out[placed[:, 0], placed[:, 1]] == bg), "destination must be empty after clearing sprites"
        out[placed[:, 0], placed[:, 1]] = color

    return out


def _grow_mask(height: int, width: int, cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < cells and frontier:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        if mask[nr, nc]:
            continue
        mask[nr, nc] = True
        add_frontier(nr, nc)
    return mask


def _random_sprite_mask(direction: str) -> np.ndarray:
    for _ in range(SPRITE_ATTEMPTS):
        raw_h = random.randint(2, 5)
        raw_w = random.randint(2, 5)
        max_cells = min(raw_h * raw_w - 1, 8)
        if max_cells < 3:
            continue
        cells = random.randint(3, max_cells)
        mask = _crop_mask(_grow_mask(raw_h, raw_w, cells))
        sh, sw = mask.shape
        if sh < 2 or sw < 2:
            continue
        if direction in ("left", "right") and np.array_equal(mask, np.fliplr(mask)):
            continue
        if direction in ("up", "down") and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate sprite")


def _sample_frame_dims(sprite_shape: tuple[int, int], direction: str) -> tuple[int, int]:
    sh, sw = sprite_shape
    if direction in ("left", "right"):
        frame_h_choices = [v for v in range(5, 9) if v % 2 == sh % 2]
        frame_w_choices = list(range(5, 9))
    else:
        frame_h_choices = list(range(5, 9))
        frame_w_choices = [v for v in range(5, 9) if v % 2 == sw % 2]
    return random.choice(frame_h_choices), random.choice(frame_w_choices)


def _paint_mask(patch: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    patch[top + rows, left + cols] = color


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _build_bundle(
    bg: int,
    frame_color: int,
    color: int,
    direction: str,
    sprite_mask: np.ndarray,
) -> dict:
    sh, sw = sprite_mask.shape
    frame_h, frame_w = _sample_frame_dims((sh, sw), direction)
    moved = _mirror_mask(sprite_mask, direction)

    if direction in ("left", "right"):
        bundle_h = max(frame_h, sh)
        bundle_w = frame_w + sw
        frame_r = (bundle_h - frame_h) // 2
        sprite_r = (bundle_h - sh) // 2
        if direction == "left":
            sprite_c, frame_c = 0, sw
        else:
            frame_c, sprite_c = 0, frame_w
    else:
        bundle_h = frame_h + sh
        bundle_w = max(frame_w, sw)
        frame_c = (bundle_w - frame_w) // 2
        sprite_c = (bundle_w - sw) // 2
        if direction == "up":
            sprite_r, frame_r = 0, sh
        else:
            frame_r, sprite_r = 0, frame_h

    patch_out = np.full((bundle_h, bundle_w), bg, dtype=int)
    patch_in = np.full((bundle_h, bundle_w), bg, dtype=int)

    frame_top = frame_r
    frame_left = frame_c
    _paint_mask(patch_out, _frame_mask(frame_h, frame_w), frame_top, frame_left, frame_color)
    _paint_mask(patch_in, _frame_mask(frame_h, frame_w), frame_top, frame_left, frame_color)

    arrow_top = frame_top + 1 + random.randint(0, frame_h - 5)
    arrow_left = frame_left + 1 + random.randint(0, frame_w - 5)
    arrow_mask = np.zeros((3, 3), dtype=bool)
    for r, c in ARROW_TEMPLATES[direction]:
        arrow_mask[r, c] = True
    _paint_mask(patch_out, arrow_mask, arrow_top, arrow_left, color)
    _paint_mask(patch_in, arrow_mask, arrow_top, arrow_left, color)

    _paint_mask(patch_out, moved, sprite_r, sprite_c, color)

    return {
        "input_patch": patch_in,
        "output_patch": patch_out,
        "input_sprite_mask": sprite_mask,
        "color": int(color),
    }


def _can_place_bbox(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark_bbox(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    rows, cols = np.argwhere(mask).T
    grid[top + rows, left + cols] = patch[rows, cols]


def _sample_directions(count: int) -> list[str]:
    dirs = list(DIRECTIONS)
    random.shuffle(dirs)
    return dirs[:count]


def _sample_canvas_size(bundle_specs: list[dict]) -> tuple[int, int]:
    total_h = sum(spec["input_patch"].shape[0] + 2 for spec in bundle_specs)
    total_w = sum(spec["input_patch"].shape[1] + 2 for spec in bundle_specs)
    max_h = max(spec["input_patch"].shape[0] for spec in bundle_specs)
    max_w = max(spec["input_patch"].shape[1] for spec in bundle_specs)

    min_h = max(18, max_h + 4, int(total_h * 0.6))
    min_w = max(18, max_w + 4, int(total_w * 0.6))
    max_h_choice = max(min_h, min(28, min_h + 6))
    max_w_choice = max(min_w, min(28, min_w + 6))
    return random.randint(min_h, max_h_choice), random.randint(min_w, max_w_choice)


def _place_bundles(inp: np.ndarray, out: np.ndarray, bundle_specs: list[dict], bg: int) -> tuple[bool, np.ndarray]:
    h, w = inp.shape
    blocked = np.zeros((h, w), dtype=bool)
    sprite_blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(bundle_specs)), key=lambda i: bundle_specs[i]["input_patch"].size, reverse=True)

    for idx in order:
        spec = bundle_specs[idx]
        ph, pw = spec["input_patch"].shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            if not _can_place_bbox(blocked, top, left, ph, pw, pad=1):
                continue
            _paste_non_bg(inp, spec["input_patch"], top, left, bg)
            _paste_non_bg(out, spec["output_patch"], top, left, bg)
            _mark_bbox(blocked, top, left, ph, pw, pad=1)
            _mark_bbox(sprite_blocked, top, left, ph, pw, pad=1)
            placed = True
            break
        if not placed:
            return False, sprite_blocked
    return True, sprite_blocked


def _place_original_sprites(inp: np.ndarray, bundle_specs: list[dict], blocked: np.ndarray, bg: int) -> bool:
    h, w = inp.shape
    order = sorted(
        bundle_specs,
        key=lambda spec: spec["input_sprite_mask"].shape[0] * spec["input_sprite_mask"].shape[1],
        reverse=True,
    )

    for spec in order:
        mask = spec["input_sprite_mask"]
        sh, sw = mask.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - sh)
            left = random.randint(0, w - sw)
            if not _can_place_bbox(blocked, top, left, sh, sw, pad=1):
                continue
            view = inp[top : top + sh, left : left + sw]
            if np.any(view[mask] != bg):
                continue
            view[mask] = spec["color"]
            _mark_bbox(blocked, top, left, sh, sw, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _valid_input_properties(grid: np.ndarray, bg: int, frame_color: int, n_frames: int) -> bool:
    if _bg_color(grid) != bg:
        return False
    frame_components = _components_of_color(grid, frame_color)
    if len(frame_components) != n_frames:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    palette = random.sample(list(range(10)), n_frames + 2)
    bg = int(palette[0])
    frame_color = int(palette[1])
    sprite_colors = [int(v) for v in palette[2:]]
    directions = _sample_directions(n_frames)

    bundle_specs = []
    for color, direction in zip(sprite_colors, directions):
        sprite_mask = _random_sprite_mask(direction)
        bundle_specs.append(_build_bundle(bg, frame_color, color, direction, sprite_mask))

    h, w = _sample_canvas_size(bundle_specs)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    ok, sprite_blocked = _place_bundles(inp, out, bundle_specs, bg)
    if not ok:
        return None
    if not _place_original_sprites(inp, bundle_specs, sprite_blocked, bg):
        return None
    if not _valid_input_properties(inp, bg, frame_color, n_frames):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
