import itertools
import random

import numpy as np


BG = 0
FRAME = 5
ERASED = 6
PATTERN_COLORS = (1, 2, 3, 4, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _rot180(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(grid))


def _transform(grid: np.ndarray, kind: str) -> np.ndarray:
    if kind == "id":
        return np.array(grid, dtype=int, copy=True)
    if kind == "lr":
        return np.fliplr(grid)
    if kind == "ud":
        return np.flipud(grid)
    if kind == "r2":
        return _rot180(grid)
    raise ValueError(f"unknown transform kind: {kind}")


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    frame_coords = np.argwhere(grid == FRAME)
    assert frame_coords.size > 0, "expected one grey frame"
    r0, c0, r1, c1 = _bbox(frame_coords)
    border = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(border[0, :] == FRAME)
    assert np.all(border[-1, :] == FRAME)
    assert np.all(border[:, 0] == FRAME)
    assert np.all(border[:, -1] == FRAME)
    inner = border[1:-1, 1:-1]
    assert inner.shape[0] % 2 == 0 and inner.shape[1] % 2 == 0, "target area must be 2h x 2w"
    return (r0, c0, r1, c1), np.array(inner, dtype=int, copy=True)


def _extract_tiles(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], h: int, w: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    outside = grid != BG
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(outside)
    tiles: list[np.ndarray] = []
    for coords in comps:
        br0, bc0, br1, bc1 = _bbox(coords)
        tile = grid[br0 : br1 + 1, bc0 : bc1 + 1]
        if tile.shape != (h, w):
            continue
        if coords.shape[0] != h * w:
            continue
        if np.any(tile == BG):
            continue
        tiles.append(np.array(tile, dtype=int, copy=True))
    assert len(tiles) == 3, f"expected 3 outside tiles, found {len(tiles)}"
    return tiles


def _reconstruct_base(seed: np.ndarray, tiles: list[np.ndarray]) -> np.ndarray:
    valid_bases: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        oriented = [
            seed,
            _transform(tiles[perm[0]], "lr"),
            _transform(tiles[perm[1]], "ud"),
            _transform(tiles[perm[2]], "r2"),
        ]
        base = np.empty_like(seed)
        ok = True
        for r in range(seed.shape[0]):
            for c in range(seed.shape[1]):
                vals = [int(src[r, c]) for src in oriented if int(src[r, c]) != ERASED]
                if not vals or len(set(vals)) != 1:
                    ok = False
                    break
                base[r, c] = vals[0]
            if not ok:
                break
        if ok:
            valid_bases.append(base)
    assert len(valid_bases) == 1, f"expected a unique tile-orientation assignment, found {len(valid_bases)}"
    return valid_bases[0]


def _expected_output(base: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [base, np.fliplr(base)],
            [np.flipud(base), _rot180(base)],
        ]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the three scattered mirror-tiles into the framed four-quadrant mosaic and repair every magenta erasure.

    Input structure:
    - A grey 1-cell frame encloses a 2h x 2w target area whose top-left h x w quadrant contains a colored tile fragment with magenta erased cells, while the other three quadrants are black.
    - Outside the frame lie exactly three separate h x w rectangular tiles, also using magenta as an erasure color.
    - Ignoring magenta cells, the outside tiles are the left-right mirror, the up-down mirror, and the 180-degree rotation of the true top-left pattern.

    Rule:
    - Infer the framed region size from the grey border and read the top-left quadrant fragment.
    - Extract the three outside h x w tiles and test which one must be the horizontal mirror, vertical mirror, or 180-degree rotation by undoing each transform and checking pixelwise agreement on every non-erased observation.
    - Reconstruct the unique underlying h x w pattern from the four partial observations, then output the repaired 2h x 2w mirror mosaic with quadrants [original, left-right mirror; up-down mirror, 180-degree rotation].

    Key insight:
    - Each outside tile becomes comparable only after reversing its candidate symmetry transform; the correct assignment is the unique one whose four partial views agree cellwise and cover every orbit at least once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox, target = _extract_frame(g)
    h = target.shape[0] // 2
    w = target.shape[1] // 2
    seed = np.array(target[:h, :w], dtype=int, copy=True)
    tiles = _extract_tiles(g, frame_bbox, h, w)
    base = _reconstruct_base(seed, tiles)
    return _expected_output(base)


def _has_forbidden_symmetry(base: np.ndarray) -> bool:
    return (
        np.array_equal(base, np.fliplr(base))
        or np.array_equal(base, np.flipud(base))
        or np.array_equal(base, _rot180(base))
    )


def _paint_shape(base: np.ndarray, color: int) -> None:
    h, w = base.shape
    kind = random.choice(("rect", "hline", "vline", "l"))
    if kind == "rect":
        rh = random.randint(1, min(3, h))
        rw = random.randint(1, min(3, w))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        base[r0 : r0 + rh, c0 : c0 + rw] = color
        return
    if kind == "hline":
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, min(w - 1, c0 + 2))
        base[r, c0 : c1 + 1] = color
        return
    if kind == "vline":
        c = random.randint(0, w - 1)
        r0 = random.randint(0, h - 1)
        r1 = random.randint(r0, min(h - 1, r0 + 2))
        base[r0 : r1 + 1, c] = color
        return
    arm_h = random.randint(1, min(3, h))
    arm_w = random.randint(1, min(3, w))
    r0 = random.randint(0, h - arm_h)
    c0 = random.randint(0, w - arm_w)
    base[r0 : r0 + arm_h, c0] = color
    base[r0, c0 : c0 + arm_w] = color


def _make_base_pattern(h: int, w: int) -> np.ndarray | None:
    palette_size = random.randint(3, min(5, len(PATTERN_COLORS)))
    palette = random.sample(PATTERN_COLORS, palette_size)
    base = np.full((h, w), palette[0], dtype=int)

    accent_pool = list(palette[1:])
    random.shuffle(accent_pool)
    base[0, 0] = accent_pool[0]
    base[0, -1] = accent_pool[-1]
    base[-1, 0] = accent_pool[len(accent_pool) // 2]
    base[-1, -1] = random.choice(palette)

    shape_count = random.randint(3, 6)
    for _ in range(shape_count):
        _paint_shape(base, random.choice(accent_pool))

    if len(np.unique(base)) < 3:
        return None
    non_base = int(np.count_nonzero(base != palette[0]))
    if non_base < max(4, (h * w) // 4):
        return None
    if _has_forbidden_symmetry(base):
        return None
    return base


def _sample_visibility(h: int, w: int) -> np.ndarray | None:
    positions = [(r, c) for r in range(h) for c in range(w)]
    if len(positions) < 10:
        return None
    random.shuffle(positions)

    vis = np.zeros((4, h, w), dtype=bool)
    idx = 0

    for src in range(4):
        r, c = positions[idx]
        idx += 1
        vis[src, r, c] = True

    pairs = list(itertools.combinations(range(4), 2))
    random.shuffle(pairs)
    for src_a, src_b in pairs:
        r, c = positions[idx]
        idx += 1
        vis[src_a, r, c] = True
        vis[src_b, r, c] = True

    for r, c in positions[idx:]:
        count = random.choices((1, 2, 3, 4), weights=(2, 4, 3, 1), k=1)[0]
        chosen = random.sample(range(4), count)
        vis[chosen, r, c] = True

    counts = vis.reshape(4, -1).sum(axis=1)
    area = h * w
    if np.min(counts) < max(4, area // 3):
        return None
    if np.max(counts) >= area:
        return None
    return vis


def _make_partials(base: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    partials = [np.where(vis[i], base, ERASED).astype(int) for i in range(4)]
    tiles = [
        np.fliplr(partials[1]),
        np.flipud(partials[2]),
        _rot180(partials[3]),
    ]
    return partials[0], tiles


def _can_place(
    rect: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    pad: int,
) -> bool:
    r0, c0, h, w = rect
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, ph, pw in placed:
        pr1 = pr0 + ph - 1
        pc1 = pc0 + pw - 1
        rows_disjoint = r1 + pad < pr0 or pr1 + pad < r0
        cols_disjoint = c1 + pad < pc0 or pc1 + pad < c0
        if not (rows_disjoint or cols_disjoint):
            return False
    return True


def _sample_layout(h: int, w: int) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]] | None:
    frame_h = 2 * h + 2
    frame_w = 2 * w + 2
    min_h = max(frame_h + 2, h + 10)
    min_w = max(frame_w + 2, w + 10)

    for _ in range(200):
        canvas_h = random.randint(min_h, 30)
        canvas_w = random.randint(min_w, 30)
        placed: list[tuple[int, int, int, int]] = []

        fr = random.randint(1, canvas_h - frame_h - 1)
        fc = random.randint(1, canvas_w - frame_w - 1)
        placed.append((fr, fc, frame_h, frame_w))

        tile_positions: list[tuple[int, int]] = []
        ok = True
        for _tile in range(3):
            found = False
            for _ in range(200):
                tr = random.randint(0, canvas_h - h)
                tc = random.randint(0, canvas_w - w)
                if _can_place((tr, tc, h, w), placed, pad=1):
                    placed.append((tr, tc, h, w))
                    tile_positions.append((tr, tc))
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return canvas_h, canvas_w, (fr, fc), tile_positions
    return None


def _render_input(
    seed: np.ndarray,
    tiles: list[np.ndarray],
    layout: tuple[int, int, tuple[int, int], list[tuple[int, int]]],
) -> np.ndarray:
    h, w = seed.shape
    canvas_h, canvas_w, (fr, fc), tile_positions = layout
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    frame_h = 2 * h + 2
    frame_w = 2 * w + 2
    grid[fr, fc : fc + frame_w] = FRAME
    grid[fr + frame_h - 1, fc : fc + frame_w] = FRAME
    grid[fr : fr + frame_h, fc] = FRAME
    grid[fr : fr + frame_h, fc + frame_w - 1] = FRAME
    grid[fr + 1 : fr + 1 + h, fc + 1 : fc + 1 + w] = seed

    shuffled_tiles = list(tiles)
    random.shuffle(shuffled_tiles)
    for tile, (tr, tc) in zip(shuffled_tiles, tile_positions):
        grid[tr : tr + h, tc : tc + w] = tile
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(4, 7)
    w = random.randint(4, 7)

    base = _make_base_pattern(h, w)
    if base is None:
        return None

    vis = _sample_visibility(h, w)
    if vis is None:
        return None

    seed, tiles = _make_partials(base, vis)
    layout = _sample_layout(h, w)
    if layout is None:
        return None

    grid = _render_input(seed, tiles, layout)
    expected = _expected_output(base)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
