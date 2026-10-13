import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
GOOD = 3
BAD = 2
COLOR_POOL = [1, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            part: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                part.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(part)
    return out


def build_composite(sprite: np.ndarray, frame_mask: np.ndarray, bg: int = BG) -> np.ndarray:
    sh, sw = sprite.shape
    fh, fw = frame_mask.shape
    out = np.full((fh * sh, fw * sw), bg, dtype=int)
    for r in range(fh):
        for c in range(fw):
            if not bool(frame_mask[r, c]):
                continue
            out[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw] = sprite
    return out


def recolor_non_bg(patch: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = patch.copy()
    out[patch != bg] = int(color)
    return out


def _separator_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = [c for c in range(w) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg]
    assert candidates, "expected a full-height separator column"
    if len(candidates) > 1:
        sep_candidates = [c for c in candidates if int(grid[0, c]) == SEP]
        if sep_candidates:
            candidates = sep_candidates
    assert len(candidates) == 1, "expected a unique legend separator column"
    return int(candidates[0])


def _extract_legend_parts(legend: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    objs = components((legend != bg) & (legend != SEP), connectivity=8)
    assert len(objs) >= 2, "legend must contain sprite and frame"
    boxes = sorted((bbox_coords(obj) for obj in objs), key=lambda b: (b[0], b[1]))
    sprite = crop_bbox(legend, boxes[0])
    frame = crop_bbox(legend, boxes[-1])
    frame_colors = sorted(set(map(int, frame.ravel())) - {bg})
    assert len(frame_colors) == 1, "frame template must be binary"
    return sprite, frame != bg


def _rotation_match(candidate: np.ndarray, standard: np.ndarray) -> bool:
    return any(candidate.shape == np.rot90(standard, k).shape and np.array_equal(candidate, np.rot90(standard, k)) for k in range(4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the legend composite, then mark rotation-equivalent candidates green and all other candidates red.

    Input structure:
    - A full-height grey separator column splits the grid into a left legend area and a right canvas.
    - The legend area contains one connected multicolor unit sprite above one connected binary frame template.
    - The canvas contains several separated candidate composites on a uniform background.

    Rule:
    - Tile the unit sprite into every active frame cell to construct the standard composite.
    - For each canvas candidate, compare its cropped pattern against the standard composite under 0, 90, 180, and 270 degree rotations.
    - Recolor all non-background cells of a matching candidate green and recolor all non-background cells of a non-matching candidate red, then drop the legend and separator.

    Key insight:
    - The target pattern is never read from the canvas; it must be synthesized from the legend, and mirror lookalikes fail because only rotations count.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_col = _separator_column(g, bg)
    legend = g[:, :sep_col]
    canvas = g[:, sep_col + 1 :].copy()

    sprite, frame_mask = _extract_legend_parts(legend, bg)
    standard = build_composite(sprite, frame_mask, bg=bg)

    parts = components(canvas != bg, connectivity=4)
    assert parts, "canvas must contain candidates"

    out = canvas.copy()
    for part in parts:
        box = bbox_coords(part)
        patch = crop_bbox(canvas, box)
        color = GOOD if _rotation_match(patch, standard) else BAD
        recolored = recolor_non_bg(patch, color, bg)
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = recolored
    return out


def _random_sprite(rng: random.Random, size: int) -> np.ndarray:
    a, b = rng.sample(COLOR_POOL, 2)
    while True:
        sprite = np.array([[a if rng.random() < 0.5 else b for _ in range(size)] for _ in range(size)], dtype=int)
        if len(set(map(int, sprite.ravel()))) != 2:
            continue
        if any(np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3)):
            continue
        return sprite


def _grow_connected_mask(rng: random.Random, size: int) -> np.ndarray | None:
    total = size * size
    target = rng.randint(max(4, total // 2), total - 1)
    seed = (rng.randrange(size), rng.randrange(size))
    cells = {seed}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not np.any(mask[0]) and not np.any(mask[-1]) and not np.any(mask[:, 0]) and not np.any(mask[:, -1]):
        return None
    return mask


def _valid_standard(comp: np.ndarray) -> bool:
    rots = [np.rot90(comp, k) for k in range(4)]
    if len({rot.tobytes() for rot in rots}) != 4:
        return False
    mirror = np.fliplr(comp)
    if any(np.array_equal(mirror, rot) for rot in rots):
        return False
    return True


def _random_legend(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    while True:
        sprite_size, frame_size = rng.choice([(2, 3), (2, 4), (3, 3)])
        sprite = _random_sprite(rng, sprite_size)
        frame_mask = _grow_connected_mask(rng, frame_size)
        if frame_mask is None:
            continue
        comp = build_composite(sprite, frame_mask, bg=BG)
        if _valid_standard(comp):
            return sprite, frame_mask


def _perturb_one_tile(rng: random.Random, standard: np.ndarray, sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    sh, sw = sprite.shape
    active = [(r, c) for r in range(frame_mask.shape[0]) for c in range(frame_mask.shape[1]) if bool(frame_mask[r, c])]
    tile_r, tile_c = rng.choice(active)
    variants = []
    for k in (1, 2, 3):
        rot = np.rot90(sprite, k)
        if not np.array_equal(rot, sprite):
            variants.append(rot)
    mirror = np.fliplr(sprite)
    if not np.array_equal(mirror, sprite):
        variants.append(mirror)
    replacement = rng.choice(variants)
    out = standard.copy()
    out[tile_r * sh : (tile_r + 1) * sh, tile_c * sw : (tile_c + 1) * sw] = replacement
    return out


def _perturb_one_pixel(rng: random.Random, standard: np.ndarray) -> np.ndarray:
    out = standard.copy()
    fg = list(zip(*np.where(out != BG)))
    r, c = rng.choice(fg)
    existing = int(out[r, c])
    replacements = [col for col in COLOR_POOL if col not in (existing, GOOD, BAD, SEP)]
    out[r, c] = rng.choice(replacements)
    return out


def _candidate_arrays(rng: random.Random, sprite: np.ndarray, frame_mask: np.ndarray) -> list[np.ndarray]:
    standard = build_composite(sprite, frame_mask, bg=BG)
    positives = [np.rot90(standard, k) for k in (1, 2, 3)]
    rng.shuffle(positives)
    keep_pos = 1 if rng.random() < 0.45 else 2
    chosen = positives[:keep_pos]

    negatives = [np.fliplr(standard), _perturb_one_tile(rng, standard, sprite, frame_mask), _perturb_one_pixel(rng, standard)]
    rng.shuffle(negatives)
    keep_neg = rng.choice([2, 3])

    arrays = chosen + negatives[:keep_neg]
    dedup: list[np.ndarray] = []
    seen: set[bytes] = set()
    for arr in arrays:
        sig = arr.tobytes()
        if sig in seen:
            continue
        assert not _rotation_match(arr, standard) or any(np.array_equal(arr, np.rot90(standard, k)) for k in (1, 2, 3))
        dedup.append(arr)
        seen.add(sig)
    assert any(_rotation_match(arr, standard) for arr in dedup)
    assert any(not _rotation_match(arr, standard) for arr in dedup)
    return dedup


def _layout_slots(comp_size: int, legend_w: int, count: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]]:
    choices = []
    for gap in (1, 2):
        for rows in range(2, 4):
            for cols in range(2, 4):
                if rows * cols < count:
                    continue
                canvas_h = rows * comp_size + (rows + 1) * gap
                canvas_w = cols * comp_size + (cols + 1) * gap
                if canvas_h <= 30 and legend_w + 1 + canvas_w <= 30:
                    choices.append((gap, rows, cols, canvas_h, canvas_w))
    assert choices, "no valid layout for candidate set"
    gap, rows, cols, canvas_h, canvas_w = rng.choice(choices)
    positions = [(gap + r * (comp_size + gap), gap + c * (comp_size + gap)) for r in range(rows) for c in range(cols)]
    rng.shuffle(positions)
    return canvas_h, canvas_w, positions[:count]


def _place_legend(grid: np.ndarray, legend_w: int, sprite: np.ndarray, frame_mask: np.ndarray, rng: random.Random) -> None:
    sh = sprite.shape[0]
    fh = frame_mask.shape[0]
    sprite_r0 = 1
    frame_r0 = sprite_r0 + sh + rng.randint(2, 4)
    sprite_c0 = (legend_w - sprite.shape[1]) // 2
    frame_c0 = (legend_w - frame_mask.shape[1]) // 2
    grid[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sprite.shape[1]] = sprite
    frame_color = rng.choice([col for col in COLOR_POOL if col not in set(map(int, sprite.ravel()))])
    frame = np.where(frame_mask, frame_color, BG)
    grid[frame_r0 : frame_r0 + fh, frame_c0 : frame_c0 + frame_mask.shape[1]] = frame


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    sprite, frame_mask = _random_legend(rng)
    standard = build_composite(sprite, frame_mask, bg=BG)
    comp_size = standard.shape[0]
    legend_w = max(sprite.shape[1], frame_mask.shape[1]) + 2

    candidates = _candidate_arrays(rng, sprite, frame_mask)
    if len(candidates) < 3:
        return None

    try:
        canvas_h, canvas_w, positions = _layout_slots(comp_size, legend_w, len(candidates), rng)
    except AssertionError:
        return None

    legend_h = 1 + sprite.shape[0] + 4 + frame_mask.shape[0]
    grid_h = max(canvas_h, legend_h + rng.randint(0, 3))
    grid_w = legend_w + 1 + canvas_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, legend_w] = SEP
    _place_legend(grid, legend_w, sprite, frame_mask, rng)
    canvas_top = rng.randint(0, grid_h - canvas_h)

    for arr, (r0, c0) in zip(candidates, positions):
        rr0 = canvas_top + r0
        cc0 = c0
        target = grid[rr0 : rr0 + comp_size, legend_w + 1 + cc0 : legend_w + 1 + cc0 + comp_size]
        if np.any((target != BG) & (arr != BG)):
            return None
        target[arr != BG] = arr[arr != BG]

    canvas = grid[:, legend_w + 1 :]
    parts = components(canvas != BG, 4)
    if len(parts) != len(candidates):
        return None
    if any(crop_bbox(canvas, bbox_coords(part)).shape != standard.shape for part in parts):
        return None
    if any(np.array_equal(crop_bbox(canvas, bbox_coords(part)), standard) for part in parts):
        return None
    if int(np.sum(grid == BG)) <= grid.size // 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (grid_h, canvas_w):
        return None
    if np.sum(out == GOOD) == 0 or np.sum(out == BAD) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-composite recolor input")
