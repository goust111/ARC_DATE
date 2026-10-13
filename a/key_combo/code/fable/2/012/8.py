from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
FRAME_COUNT_RANGE = (3, 5)
FRAME_H_RANGE = (5, 8)
FRAME_W_RANGE = (5, 8)
SPRITE_SIDE_RANGE = (2, 4)
SCALE_RANGE = (2, 5)
EDGE_MARGIN = 1
FRAME_GAP = 1
FRAME_PLACEMENT_ATTEMPTS = 120
SPRITE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.repeat(np.repeat(sprite, k, axis=0), k, axis=1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    if coords.shape[0] < 8:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    present = {tuple(map(int, rc)) for rc in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in present) != on_border:
                return False
    return True


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, box


def _inside_frame(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)))


def _count_markers_inside(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(np.sum(grid[r0 + 1 : r1, c0 + 1 : c1] == MARKER_COLOR))


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames: list[dict] = []
    sprites: list[dict] = []

    colors = sorted(set(map(int, np.unique(g))) - {bg, MARKER_COLOR})
    for color in colors:
        for coords in components_mask(g == color):
            info = {"color": int(color), "coords": coords, "bbox": bbox_coords(coords)}
            if _is_rectangular_frame(coords):
                frames.append(info)
            else:
                sprites.append(info)

    frame_by_color = {frame["color"]: frame for frame in frames}
    candidates = []
    for sprite in sprites:
        hosts = [frame for frame in frames if _inside_frame(sprite["coords"], frame["bbox"])]
        pair = frame_by_color.get(sprite["color"])
        if len(hosts) != 1 or pair is None:
            continue
        host = hosts[0]
        if host["color"] == sprite["color"]:
            continue
        candidates.append((sprite, host, pair))

    assert len(candidates) == 1, "expected a unique sprite resolved by color pairing and containment"
    sprite, host, pair = candidates[0]
    scale = _count_markers_inside(g, pair["bbox"])
    assert scale >= 1, "expected at least one grey marker in the paired frame"

    sprite_mask, sprite_box = _coords_to_mask(sprite["coords"])
    return {
        "bg": bg,
        "sprite": sprite,
        "host": host,
        "pair": pair,
        "scale": scale,
        "sprite_mask": sprite_mask,
        "sprite_box": sprite_box,
    }


def _render_output(sprite_mask: np.ndarray, fg_color: int, scale: int, bg: int) -> np.ndarray:
    patch = np.full(sprite_mask.shape, int(bg), dtype=int)
    patch[sprite_mask] = int(fg_color)
    return scale_sprite(patch, int(scale))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the sprite by the grey-count from its same-colored frame and recolor it with its containing frame.

    Input structure:
    - The background contains several one-cell-thick rectangular frames with distinct colors.
    - Grey marker pixels may lie inside frame interiors or outside every frame.
    - One small monochrome 4-connected sprite sits strictly inside one host frame, but the sprite's color matches a different frame.

    Rule:
    - Identify the unique non-frame sprite component.
    - Find the distinct frame with the same color as that sprite and count the grey markers strictly inside it; call that count N.
    - Find the host frame that spatially contains the sprite and take its color C.
    - Crop the sprite to its tight bbox, recolor its foreground to C, and scale every cell of that cropped bitmap by N in both directions.

    Key insight:
    - Color matching and spatial containment point to different frames: the same-colored frame gives the scale, while the containing frame gives the output color.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render_output(scene["sprite_mask"], scene["host"]["color"], scene["scale"], scene["bg"])


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = FRAME_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _interior_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _random_frame_box(
    rng: random.Random,
    shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = shape
    max_h = min(FRAME_H_RANGE[1], h - 2 * EDGE_MARGIN)
    max_w = min(FRAME_W_RANGE[1], w - 2 * EDGE_MARGIN)
    if max_h < FRAME_H_RANGE[0] or max_w < FRAME_W_RANGE[0]:
        return None
    for _ in range(FRAME_PLACEMENT_ATTEMPTS):
        fh = rng.randint(FRAME_H_RANGE[0], max_h)
        fw = rng.randint(FRAME_W_RANGE[0], max_w)
        r0 = rng.randint(EDGE_MARGIN, h - fh - EDGE_MARGIN)
        c0 = rng.randint(EDGE_MARGIN, w - fw - EDGE_MARGIN)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if any(_boxes_too_close(box, other) for other in existing):
            continue
        return box
    return None


def _random_connected_sprite(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SPRITE_ATTEMPTS):
        h = rng.randint(SPRITE_SIDE_RANGE[0], max_h)
        w = rng.randint(SPRITE_SIDE_RANGE[0], max_w)
        max_area = h * w - 1
        if max_area < 3:
            continue
        target = rng.randint(3, max_area)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        coords = np.array(sorted(cells), dtype=int)
        if _is_rectangular_frame(coords):
            continue
        mask, _ = _coords_to_mask(coords)
        if mask.all():
            continue
        return mask
    return None


def _pick_isolated_cells(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    count: int,
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | tuple[None, set[tuple[int, int]]]:
    if count == 0:
        return [], set(blocked)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    forbidden = set(blocked)
    for cell in shuffled:
        if cell in forbidden:
            continue
        chosen.append(cell)
        forbidden.add(cell)
        forbidden.update(neighbors4(*cell))
        if len(chosen) == count:
            return chosen, forbidden
    return None, set(blocked)


def _max_isolated_capacity(box: tuple[int, int, int, int]) -> int:
    inner_h = box[2] - box[0] - 1
    inner_w = box[3] - box[1] - 1
    return (inner_h * inner_w + 1) // 2


def _expected_from_meta(meta: dict) -> np.ndarray:
    return _render_output(meta["sprite_mask"], meta["host_color"], meta["scale"], meta["bg"])


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    frame_boxes: list[tuple[int, int, int, int]] = []
    for _ in range(frame_count):
        box = _random_frame_box(rng, (h, w), frame_boxes)
        if box is None:
            return None
        frame_boxes.append(box)

    host_candidates = [idx for idx, box in enumerate(frame_boxes) if (box[2] - box[0] - 1) >= 2 and (box[3] - box[1] - 1) >= 2]
    pair_candidates = [idx for idx, box in enumerate(frame_boxes) if _max_isolated_capacity(box) >= SCALE_RANGE[0]]
    if len(host_candidates) < 1 or len(pair_candidates) < 2:
        return None

    host_idx = rng.choice(host_candidates)
    pair_options = [idx for idx in pair_candidates if idx != host_idx]
    if not pair_options:
        return None
    pair_idx = rng.choice(pair_options)

    colors = rng.sample(list(FRAME_COLORS), frame_count)
    host_color = colors[host_idx]
    sprite_color = colors[pair_idx]

    host_box = frame_boxes[host_idx]
    host_inner_h = host_box[2] - host_box[0] - 1
    host_inner_w = host_box[3] - host_box[1] - 1
    sprite_mask = _random_connected_sprite(
        rng,
        min(SPRITE_SIDE_RANGE[1], host_inner_h),
        min(SPRITE_SIDE_RANGE[1], host_inner_w),
    )
    if sprite_mask is None:
        return None

    sprite_h, sprite_w = sprite_mask.shape
    sprite_positions = []
    for r0 in range(host_box[0] + 1, host_box[2] - sprite_h + 1):
        for c0 in range(host_box[1] + 1, host_box[3] - sprite_w + 1):
            sprite_box = (r0, c0, r0 + sprite_h - 1, c0 + sprite_w - 1)
            free_host_cells = [
                cell
                for cell in _interior_cells(host_box)
                if not (sprite_box[0] <= cell[0] <= sprite_box[2] and sprite_box[1] <= cell[1] <= sprite_box[3])
            ]
            if free_host_cells:
                sprite_positions.append((sprite_box, free_host_cells))
    if not sprite_positions:
        return None
    sprite_box, host_marker_candidates = rng.choice(sprite_positions)

    max_scale_by_output = min(30 // sprite_h, 30 // sprite_w)
    pair_cap = _max_isolated_capacity(frame_boxes[pair_idx])
    scale_choices = [k for k in range(SCALE_RANGE[0], min(SCALE_RANGE[1], max_scale_by_output, pair_cap) + 1)]
    if not scale_choices:
        return None
    scale = rng.choice(scale_choices)

    host_cap = min(3, len(host_marker_candidates))
    host_choices = [k for k in range(1, host_cap + 1) if k != scale]
    if not host_choices:
        return None
    host_marker_count = rng.choice(host_choices)

    grid = np.full((h, w), BG, dtype=int)
    for color, box in zip(colors, frame_boxes):
        _draw_frame(grid, box, color)

    for r in range(sprite_h):
        for c in range(sprite_w):
            if sprite_mask[r, c]:
                grid[sprite_box[0] + r, sprite_box[1] + c] = sprite_color

    marker_forbidden: set[tuple[int, int]] = set()

    pair_markers, marker_forbidden = _pick_isolated_cells(
        rng,
        _interior_cells(frame_boxes[pair_idx]),
        scale,
        marker_forbidden,
    )
    if pair_markers is None:
        return None

    host_markers, marker_forbidden = _pick_isolated_cells(
        rng,
        host_marker_candidates,
        host_marker_count,
        marker_forbidden,
    )
    if host_markers is None:
        return None

    extra_indices = [idx for idx in range(frame_count) if idx not in (host_idx, pair_idx)]
    extra_marker_counts: dict[int, int] = {}
    for idx in extra_indices:
        cap = min(2, len(_interior_cells(frame_boxes[idx])))
        extra_marker_counts[idx] = rng.randint(0, cap)

    for idx in extra_indices:
        count = extra_marker_counts[idx]
        markers, marker_forbidden = _pick_isolated_cells(
            rng,
            _interior_cells(frame_boxes[idx]),
            count,
            marker_forbidden,
        )
        if markers is None:
            return None
        for r, c in markers:
            grid[r, c] = MARKER_COLOR

    outside_candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG:
                continue
            if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for box in frame_boxes):
                continue
            outside_candidates.append((r, c))
    outside_count = rng.randint(1, min(3, len(outside_candidates)))
    outside_markers, marker_forbidden = _pick_isolated_cells(
        rng,
        outside_candidates,
        outside_count,
        marker_forbidden,
    )
    if outside_markers is None:
        return None

    for r, c in pair_markers + host_markers + outside_markers:
        grid[r, c] = MARKER_COLOR

    if get_bg(grid) != BG:
        return None

    meta = {
        "bg": BG,
        "host_color": host_color,
        "sprite_color": sprite_color,
        "sprite_mask": sprite_mask.copy(),
        "scale": scale,
        "host_marker_count": host_marker_count,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _gen_attempt(rng)
        if built is None:
            continue
        grid, meta = built
        expected = _expected_from_meta(meta)
        wrong_scale = _render_output(meta["sprite_mask"], meta["host_color"], meta["host_marker_count"], meta["bg"])
        wrong_color = _render_output(meta["sprite_mask"], meta["sprite_color"], meta["scale"], meta["bg"])
        if np.array_equal(expected, wrong_scale) or np.array_equal(expected, wrong_color):
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
