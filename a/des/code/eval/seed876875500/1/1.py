from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


TEMPLATE_LIBRARY = [
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
]


GRID_MIN = 28
GRID_MAX = 36
MARGIN_CHOICES = (6, 7)
SLOT_GAP_CHOICES = (2, 3)
FRAME_COUNT_CHOICES = (2, 3, 4)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in coord_set) != on_perimeter:
                return False
    return True


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        if not _is_rect_frame(comp):
            continue
        box = _bbox(comp)
        frames.append({"bbox": box, "color": int(next(iter(colors)))})
    assert frames, "expected at least one rectangular frame"
    frame_colors = {int(frame["color"]) for frame in frames}
    assert len(frame_colors) == 1, "all frames must share one border color"
    frames.sort(key=lambda item: tuple(item["bbox"]))  # stable region order
    return frames


def _extract_templates(
    grid: np.ndarray, bg: int, frames: list[dict[str, object]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mask = grid != bg
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = _components(mask)
    assert len(comps) == 4, "expected exactly four outside templates"

    union_r0 = min(int(frame["bbox"][0]) for frame in frames)
    union_c0 = min(int(frame["bbox"][1]) for frame in frames)
    union_r1 = max(int(frame["bbox"][2]) for frame in frames)
    union_c1 = max(int(frame["bbox"][3]) for frame in frames)

    templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        side_hits = []
        if r1 < union_r0:
            side_hits.append("up")
        if r0 > union_r1:
            side_hits.append("down")
        if c1 < union_c0:
            side_hits.append("left")
        if c0 > union_c1:
            side_hits.append("right")
        assert len(side_hits) == 1, "template side must be unambiguous"
        side = side_hits[0]
        assert side not in templates, "each side needs one template"
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprite_mask = sprite != bg
        templates[side] = (sprite, sprite_mask)

    assert set(templates) == set(DIRECTION_VECTORS), "missing directional template"
    return templates


def _region_seed_direction(
    grid: np.ndarray, bg: int, frame: dict[str, object]
) -> tuple[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame["bbox"]
    border_color = int(frame["color"])
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    cells = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in np.argwhere(sub != bg)]
    assert len(cells) == 2, "each region must contain exactly one two-cell seed"
    assert abs(cells[0][0] - cells[1][0]) + abs(cells[0][1] - cells[1][1]) == 1, "seed must be 4-connected"

    tail = [cell for cell in cells if int(grid[cell]) == border_color]
    head = [cell for cell in cells if int(grid[cell]) != border_color]
    assert len(tail) == 1 and len(head) == 1, "seed must use border-color tail and one head cell"

    dr = head[0][0] - tail[0][0]
    dc = head[0][1] - tail[0][1]
    for name, vec in DIRECTION_VECTORS.items():
        if (dr, dc) == vec:
            return name, cells
    raise AssertionError("seed direction must be cardinal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the direction encoded by each framed region's two-cell seed to choose
    the template sprite on the matching outside side, then paste that template
    centered into the region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _most_common_color(g)
    frames = _extract_frames(g, bg)
    templates = _extract_templates(g, bg, frames)

    out = g.copy()
    for frame in frames:
        direction, seed_cells = _region_seed_direction(g, bg, frame)
        r0, c0, r1, c1 = frame["bbox"]
        inner_r0, inner_c0 = r0 + 1, c0 + 1
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1

        for sr, sc in seed_cells:
            out[sr, sc] = bg

        sprite, sprite_mask = templates[direction]
        sh, sw = sprite.shape
        assert sh <= inner_h and sw <= inner_w, "template must fit inside its region"
        pr = inner_r0 + (inner_h - sh) // 2
        pc = inner_c0 + (inner_w - sw) // 2
        patch = out[pr : pr + sh, pc : pc + sw]
        patch[sprite_mask] = sprite[sprite_mask]

    return out


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    arr = np.full(mask.shape, bg, dtype=int)
    arr[mask] = color
    return arr


def _apply_random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    if rng.random() < 0.5:
        arr = np.flipud(arr)
    return _trim_mask(arr)


def _choose_template_masks(rng: random.Random) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen = set()
    pool = list(TEMPLATE_LIBRARY)
    rng.shuffle(pool)
    for base in pool:
        for _ in range(12):
            candidate = _apply_random_transform(base, rng)
            key = _mask_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            masks.append(candidate)
            break
        if len(masks) == 4:
            return masks
    raise RuntimeError("failed to sample four distinct template masks")


def _center_with_jitter(start: int, end: int, size: int, jitter: int, rng: random.Random) -> int:
    base = start + (end - start + 1 - size) // 2
    lo = max(start, base - jitter)
    hi = min(end - size + 1, base + jitter)
    assert lo <= hi
    return rng.randint(lo, hi)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_slot_subset(count: int, rng: random.Random) -> list[int]:
    if count == 2:
        return rng.choice(([0, 3], [1, 2]))
    if count == 3:
        missing = rng.randrange(4)
        return [idx for idx in range(4) if idx != missing]
    return [0, 1, 2, 3]


def _place_seed(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    direction: str,
    head_color: int,
    border_color: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = box
    safe_r0, safe_c0 = r0 + 2, c0 + 2
    safe_r1, safe_c1 = r1 - 2, c1 - 2
    dr, dc = DIRECTION_VECTORS[direction]
    tails = []
    for tr in range(safe_r0, safe_r1 + 1):
        for tc in range(safe_c0, safe_c1 + 1):
            hr = tr + dr
            hc = tc + dc
            if safe_r0 <= hr <= safe_r1 and safe_c0 <= hc <= safe_c1:
                tails.append((tr, tc))
    assert tails, "region must have room for a non-touching seed"
    tr, tc = rng.choice(tails)
    hr, hc = tr + dr, tc + dc
    grid[tr, tc] = border_color
    grid[hr, hc] = head_color


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    top_margin = rng.choice(MARGIN_CHOICES)
    bottom_margin = rng.choice(MARGIN_CHOICES)
    left_margin = rng.choice(MARGIN_CHOICES)
    right_margin = rng.choice(MARGIN_CHOICES)
    arena_top = top_margin
    arena_left = left_margin
    arena_bottom = height - bottom_margin - 1
    arena_right = width - right_margin - 1
    arena_h = arena_bottom - arena_top + 1
    arena_w = arena_right - arena_left + 1

    template_masks = _choose_template_masks(rng)
    max_template_h = max(mask.shape[0] for mask in template_masks)
    max_template_w = max(mask.shape[1] for mask in template_masks)
    min_frame_h = max(max_template_h + 2, 6)
    min_frame_w = max(max_template_w + 2, 6)

    gap_r = rng.choice(SLOT_GAP_CHOICES)
    gap_c = rng.choice(SLOT_GAP_CHOICES)
    if arena_h < 2 * min_frame_h + gap_r or arena_w < 2 * min_frame_w + gap_c:
        return None

    split_r = rng.randint(min_frame_h, arena_h - gap_r - min_frame_h)
    split_c = rng.randint(min_frame_w, arena_w - gap_c - min_frame_w)
    row_slots = [
        (arena_top, arena_top + split_r - 1),
        (arena_top + split_r + gap_r, arena_bottom),
    ]
    col_slots = [
        (arena_left, arena_left + split_c - 1),
        (arena_left + split_c + gap_c, arena_right),
    ]

    count = rng.choice(FRAME_COUNT_CHOICES)
    chosen_slots = _sample_slot_subset(count, rng)

    palette = list(range(10))
    bg, border_color, top_color, bottom_color, left_color, right_color = rng.sample(palette, 6)
    seed_palette = [color for color in palette if color not in {bg, border_color, top_color, bottom_color, left_color, right_color}]
    assert seed_palette

    grid = np.full((height, width), bg, dtype=int)

    template_by_side = {
        "up": _render_mask(template_masks[0], top_color, bg),
        "down": _render_mask(template_masks[1], bottom_color, bg),
        "left": _render_mask(template_masks[2], left_color, bg),
        "right": _render_mask(template_masks[3], right_color, bg),
    }

    top_sprite = template_by_side["up"]
    bottom_sprite = template_by_side["down"]
    left_sprite = template_by_side["left"]
    right_sprite = template_by_side["right"]

    top_band_bottom = top_margin - 2
    left_band_right = left_margin - 2
    bottom_band_top = height - bottom_margin + 1
    right_band_left = width - right_margin + 1

    if top_band_bottom < 0 or left_band_right < 0:
        return None
    if bottom_band_top > height - 1 or right_band_left > width - 1:
        return None

    placements = {
        "up": (
            _center_with_jitter(0, top_band_bottom, top_sprite.shape[0], 1, rng),
            _center_with_jitter(arena_left, arena_right, top_sprite.shape[1], 2, rng),
        ),
        "down": (
            _center_with_jitter(bottom_band_top, height - 1, bottom_sprite.shape[0], 1, rng),
            _center_with_jitter(arena_left, arena_right, bottom_sprite.shape[1], 2, rng),
        ),
        "left": (
            _center_with_jitter(arena_top, arena_bottom, left_sprite.shape[0], 2, rng),
            _center_with_jitter(0, left_band_right, left_sprite.shape[1], 1, rng),
        ),
        "right": (
            _center_with_jitter(arena_top, arena_bottom, right_sprite.shape[0], 2, rng),
            _center_with_jitter(right_band_left, width - 1, right_sprite.shape[1], 1, rng),
        ),
    }

    for side, sprite in template_by_side.items():
        r0, c0 = placements[side]
        h, w = sprite.shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[sprite != bg] = sprite[sprite != bg]

    frame_boxes = []
    for slot_index in chosen_slots:
        slot_r = slot_index // 2
        slot_c = slot_index % 2
        sr0, sr1 = row_slots[slot_r]
        sc0, sc1 = col_slots[slot_c]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        if slot_h < min_frame_h or slot_w < min_frame_w:
            return None
        frame_h = rng.randint(min_frame_h, slot_h)
        frame_w = rng.randint(min_frame_w, slot_w)
        r0 = rng.randint(sr0, sr1 - frame_h + 1)
        c0 = rng.randint(sc0, sc1 - frame_w + 1)
        box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        _draw_frame(grid, box, border_color)
        frame_boxes.append(box)

    direction_cycle = list(DIRECTION_VECTORS)
    rng.shuffle(direction_cycle)
    directions = []
    while len(directions) < len(frame_boxes):
        cycle = direction_cycle[:]
        rng.shuffle(cycle)
        directions.extend(cycle)
    directions = directions[: len(frame_boxes)]
    rng.shuffle(directions)

    for box, direction in zip(frame_boxes, directions, strict=False):
        head_color = rng.choice(seed_palette)
        _place_seed(grid, box, direction, head_color, border_color, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if _most_common_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
