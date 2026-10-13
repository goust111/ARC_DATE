from __future__ import annotations

import numpy as np


BG = 0
FRAME = 7
MAX_GRID_SIZE = 30
MIN_CANVAS_SIZE = 12
MAX_EXTRA_PADDING = 8
GEN_ATTEMPTS = 300
SEED_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
SIDE_TO_ROT_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _parse_frame(coords: np.ndarray) -> dict[str, object]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    r0, c0 = map(int, lo)
    r1, c1 = map(int, hi)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 4 and w >= 4

    rel = coords - np.array([r0, c0])
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    assert np.all(mask <= perimeter)
    assert int(mask.sum()) == int(perimeter.sum()) - 1

    missing = np.argwhere(perimeter & ~mask)
    assert len(missing) == 1
    gr, gc = map(int, missing[0])
    if gr == 0:
        assert 0 < gc < w - 1
        side = "top"
    elif gr == h - 1:
        assert 0 < gc < w - 1
        side = "bottom"
    elif gc == 0:
        assert 0 < gr < h - 1
        side = "left"
    else:
        assert gc == w - 1 and 0 < gr < h - 1
        side = "right"

    return {"bbox": (r0, c0, r1, c1), "side": side}


def _extract_seed(interior: np.ndarray) -> np.ndarray:
    seed_mask = (interior != BG) & (interior != FRAME)
    coords = np.argwhere(seed_mask)
    assert len(coords) > 0

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    r0, c0 = map(int, lo)
    r1, c1 = map(int, hi)
    seed = interior[r0 : r1 + 1, c0 : c1 + 1].copy()
    sh, sw = seed.shape

    assert sh == sw and sh in (2, 3)
    assert int(seed_mask.sum()) == sh * sw
    assert np.all(seed != BG)
    assert np.all(seed != FRAME)
    return seed


def _tile_patch(shape: tuple[int, int], patch: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    ph, pw = map(int, patch.shape)
    reps_h = (h + ph - 1) // ph
    reps_w = (w + pw - 1) // pw
    tiled = np.tile(patch, (reps_h, reps_w))
    return tiled[:h, :w]


def _describe_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for coords in _components(grid == FRAME):
        frame = _parse_frame(coords)
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        frame["seed"] = _extract_seed(interior)
        frames.append(frame)
    assert frames
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the seed inside each orange open frame, rotate it by the frame's gap direction, and tile-fill that frame interior.

    Input structure:
    - The background is black.
    - One or more orange 1-pixel-thick rectangular borders appear on the grid, and each border has exactly one 1-pixel gap on one side.
    - Each frame interior contains a single 2x2 or 3x3 non-black seed block in the upper-left area; the seed colors are different from orange.

    Rule:
    - For each orange frame, determine which side contains the border gap.
    - Use the gap direction to choose the seed rotation: top -> 0 degrees, right -> 90 degrees clockwise, bottom -> 180 degrees, left -> 270 degrees clockwise.
    - Tile the rotated seed from the interior's top-left corner across the whole frame interior, cropping on the right and bottom when the interior size is not a multiple of the seed size; keep the orange border and its gap unchanged.

    Key insight:
    - The seed alone is not enough and the gap alone is not enough: the gap direction encodes which rotation of the visible seed must be repeated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for frame in _describe_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        side = frame["side"]
        seed = np.asarray(frame["seed"], dtype=int)
        rotated = np.rot90(seed, SIDE_TO_ROT_K[side])
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_patch((r1 - r0 - 1, c1 - c0 - 1), rotated)
    return out


def _rotations_are_distinct(patch: np.ndarray) -> bool:
    sigs = {np.rot90(patch, k).tobytes() for k in range(4)}
    return len(sigs) == 4


def _random_seed_patch(rng: np.random.Generator, size: int) -> np.ndarray:
    for _ in range(100):
        palette_size = int(rng.integers(2, min(5, len(SEED_COLORS)) + 1))
        palette = rng.choice(SEED_COLORS, size=palette_size, replace=False)
        patch = rng.choice(palette, size=(size, size), replace=True).astype(int)
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        if _rotations_are_distinct(patch):
            return patch
    raise RuntimeError("failed to sample a rotation-distinct seed patch")


def _build_object(rng: np.random.Generator, max_inner: int) -> tuple[np.ndarray, np.ndarray]:
    inner_h = int(rng.integers(2, max_inner + 1))
    inner_w = int(rng.integers(2, max_inner + 1))
    seed_size = int(rng.choice([s for s in (2, 3) if s <= min(inner_h, inner_w)]))

    seed = _random_seed_patch(rng, seed_size)
    side = str(rng.choice(["top", "right", "bottom", "left"]))
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    patch_in = np.zeros((outer_h, outer_w), dtype=int)
    patch_in[0, :] = FRAME
    patch_in[-1, :] = FRAME
    patch_in[:, 0] = FRAME
    patch_in[:, -1] = FRAME

    if side in ("top", "bottom"):
        gap_col = int(rng.integers(1, outer_w - 1))
        patch_in[0 if side == "top" else -1, gap_col] = BG
    else:
        gap_row = int(rng.integers(1, outer_h - 1))
        patch_in[gap_row, 0 if side == "left" else -1] = BG

    max_seed_r = min(1, inner_h - seed_size)
    max_seed_c = min(1, inner_w - seed_size)
    seed_r = int(rng.integers(0, max_seed_r + 1))
    seed_c = int(rng.integers(0, max_seed_c + 1))
    patch_in[1 + seed_r : 1 + seed_r + seed_size, 1 + seed_c : 1 + seed_c + seed_size] = seed

    patch_out = patch_in.copy()
    rotated = np.rot90(seed, SIDE_TO_ROT_K[side])
    patch_out[1:-1, 1:-1] = _tile_patch((inner_h, inner_w), rotated)
    return patch_in, patch_out


def _layout_horizontal(
    rng: np.random.Generator, shapes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    gaps = [int(rng.integers(2, 5)) for _ in range(len(shapes) - 1)]
    total_h = max(h for h, _ in shapes) + int(rng.integers(0, 3))
    total_w = sum(w for _, w in shapes) + sum(gaps)
    positions: list[tuple[int, int]] = []
    c = 0
    for i, (h, w) in enumerate(shapes):
        r = int(rng.integers(0, total_h - h + 1))
        positions.append((r, c))
        c += w
        if i < len(gaps):
            c += gaps[i]
    return positions, total_h, total_w


def _layout_vertical(
    rng: np.random.Generator, shapes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    gaps = [int(rng.integers(2, 5)) for _ in range(len(shapes) - 1)]
    total_h = sum(h for h, _ in shapes) + sum(gaps)
    total_w = max(w for _, w in shapes) + int(rng.integers(0, 3))
    positions: list[tuple[int, int]] = []
    r = 0
    for i, (h, w) in enumerate(shapes):
        c = int(rng.integers(0, total_w - w + 1))
        positions.append((r, c))
        r += h
        if i < len(gaps):
            r += gaps[i]
    return positions, total_h, total_w


def _layout_grid(
    rng: np.random.Generator, shapes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    split = int(rng.integers(1, len(shapes)))
    top_shapes = shapes[:split]
    bot_shapes = shapes[split:]
    top_pos, top_h, top_w = _layout_horizontal(rng, top_shapes)
    bot_pos, bot_h, bot_w = _layout_horizontal(rng, bot_shapes)
    row_gap = int(rng.integers(2, 5))
    total_h = top_h + row_gap + bot_h
    total_w = max(top_w, bot_w) + int(rng.integers(0, 3))
    top_shift = int(rng.integers(0, total_w - top_w + 1))
    bot_shift = int(rng.integers(0, total_w - bot_w + 1))

    positions = [(r, c + top_shift) for r, c in top_pos]
    positions.extend((r + top_h + row_gap, c + bot_shift) for r, c in bot_pos)
    return positions, total_h, total_w


def _layout_patches(
    rng: np.random.Generator, objects: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]], int, int] | None:
    order = list(range(len(objects)))
    rng.shuffle(order)
    ordered = [objects[i] for i in order]
    shapes = [tuple(map(int, obj[0].shape)) for obj in ordered]

    candidates: list[tuple[list[tuple[int, int]], int, int]] = []
    for layout_fn in (_layout_horizontal, _layout_vertical):
        pos, draw_h, draw_w = layout_fn(rng, shapes)
        if draw_h <= MAX_GRID_SIZE and draw_w <= MAX_GRID_SIZE:
            candidates.append((pos, draw_h, draw_w))
    if len(shapes) >= 2:
        pos, draw_h, draw_w = _layout_grid(rng, shapes)
        if draw_h <= MAX_GRID_SIZE and draw_w <= MAX_GRID_SIZE:
            candidates.append((pos, draw_h, draw_w))
    if not candidates:
        return None

    positions, draw_h, draw_w = candidates[int(rng.integers(0, len(candidates)))]
    return ordered, positions, draw_h, draw_w


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _compose_canvas(
    rng: np.random.Generator,
    ordered: list[tuple[np.ndarray, np.ndarray]],
    positions: list[tuple[int, int]],
    draw_h: int,
    draw_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    extra_h_min = max(0, MIN_CANVAS_SIZE - draw_h)
    extra_w_min = max(0, MIN_CANVAS_SIZE - draw_w)
    extra_h_max = min(MAX_EXTRA_PADDING, MAX_GRID_SIZE - draw_h)
    extra_w_max = min(MAX_EXTRA_PADDING, MAX_GRID_SIZE - draw_w)
    if extra_h_min > extra_h_max or extra_w_min > extra_w_max:
        return None

    extra_h = int(rng.integers(extra_h_min, extra_h_max + 1))
    extra_w = int(rng.integers(extra_w_min, extra_w_max + 1))
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    h = draw_h + extra_h
    w = draw_w + extra_w

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for (patch_in, patch_out), (r, c) in zip(ordered, positions):
        _paste(inp, patch_in, top + r, left + c)
        _paste(out, patch_out, top + r, left + c)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(1, 5))
    max_inner = {1: 9, 2: 8, 3: 6, 4: 5}[n_objects]

    objects = [_build_object(rng, max_inner) for _ in range(n_objects)]
    laid_out = _layout_patches(rng, objects)
    if laid_out is None:
        return None
    ordered, positions, draw_h, draw_w = laid_out

    composed = _compose_canvas(rng, ordered, positions, draw_h, draw_w)
    if composed is None:
        return None
    inp, expected = composed

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
