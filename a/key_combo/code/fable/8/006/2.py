import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _inverse_aligned_quadrants(motif: np.ndarray) -> list[np.ndarray]:
    side = motif.shape[0]
    half = side // 2
    tl = motif[:half, :half]
    tr = motif[:half, half:]
    bl = motif[half:, :half]
    br = motif[half:, half:]
    return [
        tl.copy(),
        np.rot90(tr, 1),
        np.rot90(br, 2),
        np.rot90(bl, -1),
    ]


def _majority_tile(aligned: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(aligned, axis=0)
    n = stack.shape[1]
    base = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            vals = stack[:, r, c].tolist()
            counts = {}
            for value in vals:
                counts[value] = counts.get(value, 0) + 1
            best_color, best_count = max(counts.items(), key=lambda item: item[1])
            assert best_count >= 3, "each aligned position needs a unique majority color"
            base[r, c] = int(best_color)
    return base


def _hidden_sprite(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None, "expected one foreground collage"
    motif = crop_bbox(g, box)

    height, width = motif.shape
    assert height == width and height % 2 == 0, "motif must be an even square"

    aligned = _inverse_aligned_quadrants(motif)
    base = _majority_tile(aligned)

    sprite = np.full(base.shape, bg, dtype=int)
    for tile in aligned:
        diff = tile != base
        sprite[diff] = tile[diff]

    mask = sprite != bg
    assert mask.any(), "expected at least one deviation pixel"
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1, "deviation pixels must assemble into one connected sprite"
    return crop_bbox(sprite, bbox(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the hidden sprite by unrotating four noisy quadrant copies and keeping only their odd pixels.

    Input structure:
    - A mostly background grid contains one even-sided square collage.
    - The collage is split into four equal square quadrants.
    - The quadrants are the same tile shown as 0°, clockwise-90°, 180°, and counterclockwise-90° rotations, with a few substituted pixels.

    Rule:
    - Crop the square collage and inverse-rotate the four quadrants into one shared coordinate system.
    - At each aligned cell, take the majority color across the four candidates as the reference tile color.
    - Any non-majority candidate is a deviation pixel; place that deviation color at the aligned cell.
    - The deviation pixels form one connected hidden sprite; output its bounding-box crop with background elsewhere.

    Key insight:
    - The visible quadrants are only camouflage: the answer is the union of the minority pixels after undoing the quadrant rotations.
    """
    return _hidden_sprite(grid)


MAX_RETRIES = 400
COLOR_POOL = list(range(10))


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    out = []
    prev = 0
    for cut in cuts + [total]:
        out.append(int(cut - prev))
        prev = cut
    return out


def _expand_blocks(blocks: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    tile = np.zeros((sum(row_sizes), sum(col_sizes)), dtype=int)
    r0 = 0
    for rr, rh in enumerate(row_sizes):
        c0 = 0
        for cc, cw in enumerate(col_sizes):
            tile[r0 : r0 + rh, c0 : c0 + cw] = int(blocks[rr, cc])
            c0 += cw
        r0 += rh
    return tile


def _transition_score(tile: np.ndarray) -> int:
    return int(np.count_nonzero(tile[:, 1:] != tile[:, :-1]) + np.count_nonzero(tile[1:, :] != tile[:-1, :]))


def _make_base_tile(rng: np.random.Generator, n: int, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        row_parts = _randint(rng, 2, min(4, n))
        col_parts = _randint(rng, 2, min(4, n))
        row_sizes = _random_partition(n, row_parts, rng)
        col_sizes = _random_partition(n, col_parts, rng)
        blocks = rng.choice(palette, size=(row_parts, col_parts), replace=True).astype(int)
        tile = _expand_blocks(blocks, row_sizes, col_sizes)

        accent_count = _randint(rng, 2, 4)
        for _accent in range(accent_count):
            color = int(rng.choice(palette))
            mode = rng.choice(["row", "col", "rect", "corner"])
            if mode == "row":
                tile[_randint(rng, 0, n - 1), :] = color
            elif mode == "col":
                tile[:, _randint(rng, 0, n - 1)] = color
            elif mode == "rect":
                r0 = _randint(rng, 0, n - 2)
                c0 = _randint(rng, 0, n - 2)
                r1 = _randint(rng, r0, min(n - 1, r0 + 2))
                c1 = _randint(rng, c0, min(n - 1, c0 + 2))
                tile[r0 : r1 + 1, c0 : c1 + 1] = color
            else:
                h = _randint(rng, 1, min(3, n))
                w = _randint(rng, 1, min(3, n))
                if rng.random() < 0.5:
                    tile[:h, :w] = color
                else:
                    tile[n - h :, n - w :] = color

        if len(np.unique(tile)) != len(palette):
            continue
        if min(np.count_nonzero(tile == color) for color in palette) < 2:
            continue
        if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
            continue
        if _transition_score(tile) < n + 4:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to create an asymmetric base tile")


def _grow_polyomino(h: int, w: int, area: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    start = (_randint(rng, 0, h - 1), _randint(rng, 0, w - 1))
    cells = {start}
    frontier = set()
    while len(cells) < area:
        frontier.clear()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(tuple(frontier.pop()) if len(frontier) == 1 else tuple(list(frontier)[_randint(rng, 0, len(frontier) - 1)]))
    return cells


def _make_hidden_pattern(
    rng: np.random.Generator,
    n: int,
    palette: list[int],
    base_tile: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    for _ in range(200):
        h = _randint(rng, 2, min(5, n))
        w = _randint(rng, 2, min(5, n))
        max_area = min(h * w - 1, max(4, min(10, n * n // 2)))
        if max_area < 4:
            continue
        min_area = max(4, min(h * w - 1, h + w - 2))
        area = _randint(rng, min_area, max_area)
        cells = _grow_polyomino(h, w, area, rng)
        if len(cells) == h * w:
            continue

        top = _randint(rng, 0, n - h)
        left = _randint(rng, 0, n - w)
        coords = [(top + r, left + c) for r, c in sorted(cells)]
        if len(coords) < 4:
            continue

        sprite = np.full((n, n), bg, dtype=int)
        perm = rng.permutation(len(coords))
        owners = np.zeros(len(coords), dtype=int)
        owners[perm[:4]] = np.arange(4)
        if len(coords) > 4:
            owners[perm[4:]] = rng.integers(0, 4, size=len(coords) - 4)

        used_colors = set()
        aligned_quads = [base_tile.copy() for _ in range(4)]
        for idx, (r, c) in enumerate(coords):
            options = [color for color in palette if color != int(base_tile[r, c])]
            color = int(rng.choice(options))
            sprite[r, c] = color
            used_colors.add(color)
            aligned_quads[int(owners[idx])][r, c] = color

        if len(used_colors) < min(2, len(palette) - 1):
            continue
        if any(np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3)):
            continue
        if len(components_mask(sprite != bg, connectivity=4)) != 1:
            continue
        return sprite, aligned_quads
    raise RuntimeError("failed to create a hidden deviation sprite")


def _compose_canvas(rng: np.random.Generator, motif: np.ndarray, bg: int) -> np.ndarray:
    side = motif.shape[0]
    min_h = max(side + 2, int(np.ceil(np.sqrt(2 * side * side + 1))))
    height = _randint(rng, min_h, 30)
    min_w = max(side + 2, int(np.ceil((2 * side * side + 1) / height)))
    width = _randint(rng, min_w, 30)
    canvas = np.full((height, width), bg, dtype=int)
    top = _randint(rng, 0, height - side)
    left = _randint(rng, 0, width - side)
    canvas[top : top + side, left : left + side] = motif
    return canvas


def _build_example(rng: np.random.Generator) -> np.ndarray:
    n = _randint(rng, 4, 8)
    bg = int(rng.choice(COLOR_POOL))
    palette_size = _randint(rng, 3, 5)
    palette = rng.choice([c for c in COLOR_POOL if c != bg], size=palette_size, replace=False).astype(int).tolist()

    base_tile = _make_base_tile(rng, n, palette)
    sprite, aligned_quads = _make_hidden_pattern(rng, n, palette, base_tile, bg)

    tl = aligned_quads[0]
    tr = np.rot90(aligned_quads[1], -1)
    br = np.rot90(aligned_quads[2], 2)
    bl = np.rot90(aligned_quads[3], 1)

    motif = np.zeros((2 * n, 2 * n), dtype=int)
    motif[:n, :n] = tl
    motif[:n, n:] = tr
    motif[n:, :n] = bl
    motif[n:, n:] = br

    expected = crop_bbox(sprite, bbox(sprite != bg))
    grid = _compose_canvas(rng, motif, bg)

    assert np.array_equal(main(grid), expected)
    assert len(components_mask(sprite != bg, connectivity=4)) == 1
    assert all(np.any(aligned_quads[i] != base_tile) for i in range(4))
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        try:
            return _build_example(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
