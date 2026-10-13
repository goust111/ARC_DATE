from __future__ import annotations

import math
import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE_DIMS = ((2, 2), (2, 3), (3, 2))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_palette(region: np.ndarray, bg: int) -> np.ndarray | None:
    comps = _components(region != bg)
    if len(comps) != 1:
        return None
    r0, c0, r1, c1 = _bbox(comps[0])
    palette = region[r0 : r1 + 1, c0 : c1 + 1]
    if palette.shape[0] < 2 or palette.shape[1] < 2:
        return None
    if np.any(palette == bg):
        return None
    if len(np.unique(palette)) < 2:
        return None
    return palette.copy()


def _extract_pieces(region: np.ndarray, bg: int) -> list[dict] | None:
    comps = _components(region != bg)
    if not comps:
        return None
    pieces: list[dict] = []
    seen_colors: set[int] = set()
    for comp in comps:
        colors = {int(region[r, c]) for r, c in comp}
        if len(colors) != 1:
            return None
        color = next(iter(colors))
        if color in seen_colors:
            return None
        r0, c0, r1, c1 = _bbox(comp)
        rel = tuple(sorted((r - r0, c - c0) for r, c in comp))
        pieces.append(
            {
                "color": color,
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "shape": rel,
                "area": len(comp),
            }
        )
        seen_colors.add(color)
    return pieces


def _classify_sides(
    first: np.ndarray, second: np.ndarray, bg: int
) -> tuple[np.ndarray, list[dict]] | None:
    palette_first = _extract_palette(first, bg)
    pieces_second = _extract_pieces(second, bg)
    if palette_first is not None and pieces_second is not None:
        return palette_first, pieces_second

    palette_second = _extract_palette(second, bg)
    pieces_first = _extract_pieces(first, bg)
    if palette_second is not None and pieces_first is not None:
        return palette_second, pieces_first
    return None


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    h, w = g.shape

    candidates: list[tuple[np.ndarray, list[dict]]] = []

    for r in range(h):
        row = g[r]
        if int(row[0]) == bg or not np.all(row == row[0]):
            continue
        if not np.any(g[:r] != bg) or not np.any(g[r + 1 :] != bg):
            continue
        parsed = _classify_sides(g[:r, :], g[r + 1 :, :], bg)
        if parsed is not None:
            candidates.append(parsed)

    for c in range(w):
        col = g[:, c]
        if int(col[0]) == bg or not np.all(col == col[0]):
            continue
        if not np.any(g[:, :c] != bg) or not np.any(g[:, c + 1 :] != bg):
            continue
        parsed = _classify_sides(g[:, :c], g[:, c + 1 :], bg)
        if parsed is not None:
            candidates.append(parsed)

    assert len(candidates) == 1, "expected a unique separator and region split"
    return candidates[0]


def _upscale_palette(palette: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(palette, scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered monochrome pieces into a full rectangle and recolor it by the palette.

    The assembled rectangle has the same aspect ratio as the palette rectangle, so its scale is
    determined by area alone. Once that scale is known, the final output is just the palette
    enlarged by that factor.
    """
    palette, pieces = _parse_input(np.asarray(grid, dtype=int))
    total_area = sum(int(piece["area"]) for piece in pieces)
    palette_area = int(palette.shape[0] * palette.shape[1])
    assert total_area % palette_area == 0, "piece area must be an integer multiple of the palette area"
    scale_sq = total_area // palette_area
    scale = int(math.isqrt(scale_sq))
    assert scale > 0 and scale * scale == scale_sq, "assembled rectangle must be an integer upscaling"
    return _upscale_palette(palette.astype(int), scale)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _count_packings(
    shapes: list[tuple[tuple[int, int], ...]], out_h: int, out_w: int, limit: int = 2
) -> int:
    indexed = sorted(enumerate(shapes), key=lambda item: (-len(item[1]), item[0]))
    sorted_shapes = [shape for _, shape in indexed]
    cell_count = out_h * out_w
    full_mask = (1 << cell_count) - 1
    placements_by_piece: list[list[int]] = []
    cover_lookup: list[list[list[int]]] = []

    for shape in sorted_shapes:
        mask = _shape_mask(shape)
        sh, sw = mask.shape
        placements: list[int] = []
        cover = [[] for _ in range(cell_count)]
        for top in range(out_h - sh + 1):
            for left in range(out_w - sw + 1):
                bitmask = 0
                for r, c in shape:
                    idx = (top + r) * out_w + (left + c)
                    bitmask |= 1 << idx
                placements.append(bitmask)
                bits = bitmask
                while bits:
                    lsb = bits & -bits
                    cover[lsb.bit_length() - 1].append(bitmask)
                    bits ^= lsb
        placements_by_piece.append(placements)
        cover_lookup.append(cover)

    solution_count = 0

    def search(occupied: int, used: int) -> None:
        nonlocal solution_count
        if solution_count >= limit:
            return
        if occupied == full_mask:
            solution_count += 1
            return
        empty_bits = full_mask ^ occupied
        lsb = empty_bits & -empty_bits
        cell = lsb.bit_length() - 1
        for piece_idx in range(len(sorted_shapes)):
            if (used >> piece_idx) & 1:
                continue
            for placement in cover_lookup[piece_idx][cell]:
                if placement & occupied:
                    continue
                search(occupied | placement, used | (1 << piece_idx))
                if solution_count >= limit:
                    return

    search(0, 0)
    return solution_count


def _random_partition(out_h: int, out_w: int, piece_count: int) -> list[tuple[tuple[int, int], ...]] | None:
    area = out_h * out_w
    min_area = 3
    if area < piece_count * min_area:
        return None

    cells = [(r, c) for r in range(out_h) for c in range(out_w)]
    random.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - sr) + abs(cell[1] - sc) >= 2 for sr, sc in seeds):
            seeds.append(cell)
            if len(seeds) == piece_count:
                break
    if len(seeds) < piece_count:
        seeds = random.sample(cells, piece_count)

    labels = np.full((out_h, out_w), -1, dtype=int)
    frontiers = [set() for _ in range(piece_count)]
    sizes = [1] * piece_count
    targets = [min_area] * piece_count
    for _ in range(area - piece_count * min_area):
        targets[random.randrange(piece_count)] += 1

    def add_frontier(idx: int, r: int, c: int) -> None:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < out_h and 0 <= nc < out_w and labels[nr, nc] < 0:
                frontiers[idx].add((nr, nc))

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
        add_frontier(idx, r, c)

    assigned = piece_count
    while assigned < area:
        candidates = []
        weights = []
        for idx in range(piece_count):
            if sizes[idx] >= targets[idx]:
                continue
            options = [cell for cell in frontiers[idx] if labels[cell] < 0]
            if not options:
                continue
            candidates.append((idx, options))
            weights.append((targets[idx] - sizes[idx]) * len(options))
        if not candidates:
            return None
        choice_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        idx, options = candidates[choice_idx]
        r, c = random.choice(options)
        labels[r, c] = idx
        sizes[idx] += 1
        assigned += 1
        add_frontier(idx, r, c)

    shapes: list[tuple[tuple[int, int], ...]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    has_irregular = False
    for idx in range(piece_count):
        coords = [(int(r), int(c)) for r, c in np.argwhere(labels == idx)]
        shape = _normalize(coords)
        if shape in seen_shapes:
            return None
        seen_shapes.add(shape)
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            has_irregular = True
        shapes.append(shape)

    if not has_irregular:
        return None
    if _count_packings(shapes, out_h, out_w, limit=2) != 1:
        return None
    return shapes


def _scatter_shapes(
    shapes: list[tuple[tuple[int, int], ...]], colors: list[int], out_h: int, out_w: int
) -> np.ndarray | None:
    masks = [_shape_mask(shape) for shape in shapes]
    board_h = out_h + 2 * len(shapes) + random.randint(3, 5)
    board_w = out_w + 2 * len(shapes) + random.randint(3, 5)
    board = np.full((board_h, board_w), BG, dtype=int)
    blocked = np.zeros((board_h, board_w), dtype=bool)

    order = sorted(range(len(shapes)), key=lambda idx: -int(masks[idx].sum()))
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        positions: list[tuple[int, int]] = []
        for top in range(1, board_h - mh):
            for left in range(1, board_w - mw):
                cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
                if any(blocked[r, c] for r, c in cells):
                    continue
                positions.append((top, left))
        if not positions:
            return None
        top, left = random.choice(positions)
        for r, c in np.argwhere(mask):
            rr = int(top + r)
            cc = int(left + c)
            board[rr, cc] = colors[idx]
            blocked[max(0, rr - 1) : min(board_h, rr + 2), max(0, cc - 1) : min(board_w, cc + 2)] = True
    return board


def _make_palette() -> tuple[np.ndarray, int]:
    ph, pw = random.choice(PALETTE_DIMS)
    palette = np.array(random.sample(COLOR_POOL, ph * pw), dtype=int).reshape(ph, pw)
    scale = random.choice((2, 3)) if (ph, pw) == (2, 2) else 2
    return palette, scale


def _embed_palette(palette: np.ndarray) -> np.ndarray:
    ph, pw = palette.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    region = np.full((top + ph + bottom, left + pw + right), BG, dtype=int)
    region[top : top + ph, left : left + pw] = palette
    return region


def _compose_scene(palette_region: np.ndarray, piece_region: np.ndarray) -> np.ndarray:
    if random.choice((True, False)):
        palette_left = random.choice((True, False))
        left = palette_region if palette_left else piece_region
        right = piece_region if palette_left else palette_region
        core_h = max(left.shape[0], right.shape[0])
        top_pad = random.randint(0, 2)
        bottom_pad = random.randint(0, 2)
        left_pad = random.randint(0, 2)
        right_pad = random.randint(0, 2)
        grid = np.full(
            (top_pad + core_h + bottom_pad, left_pad + left.shape[1] + 1 + right.shape[1] + right_pad),
            BG,
            dtype=int,
        )
        sep_c = left_pad + left.shape[1]
        grid[:, sep_c] = SEPARATOR
        left_top = top_pad + random.randint(0, core_h - left.shape[0])
        right_top = top_pad + random.randint(0, core_h - right.shape[0])
        grid[left_top : left_top + left.shape[0], left_pad : left_pad + left.shape[1]] = left
        grid[right_top : right_top + right.shape[0], sep_c + 1 : sep_c + 1 + right.shape[1]] = right
        return grid

    palette_top = random.choice((True, False))
    top = palette_region if palette_top else piece_region
    bottom = piece_region if palette_top else palette_region
    core_w = max(top.shape[1], bottom.shape[1])
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    grid = np.full(
        (top_pad + top.shape[0] + 1 + bottom.shape[0] + bottom_pad, left_pad + core_w + right_pad),
        BG,
        dtype=int,
    )
    sep_r = top_pad + top.shape[0]
    grid[sep_r, :] = SEPARATOR
    top_left = left_pad + random.randint(0, core_w - top.shape[1])
    bottom_left = left_pad + random.randint(0, core_w - bottom.shape[1])
    grid[top_pad : top_pad + top.shape[0], top_left : top_left + top.shape[1]] = top
    grid[sep_r + 1 : sep_r + 1 + bottom.shape[0], bottom_left : bottom_left + bottom.shape[1]] = bottom
    return grid


def _gen_attempt() -> np.ndarray | None:
    palette, scale = _make_palette()
    out_h = int(palette.shape[0] * scale)
    out_w = int(palette.shape[1] * scale)
    area = out_h * out_w
    piece_count = random.randint(3, min(6, area // 3))
    shapes = _random_partition(out_h, out_w, piece_count)
    if shapes is None:
        return None

    piece_colors = random.sample(COLOR_POOL, piece_count)
    piece_region = _scatter_shapes(shapes, piece_colors, out_h, out_w)
    if piece_region is None:
        return None
    palette_region = _embed_palette(palette)
    grid = _compose_scene(palette_region, piece_region)
    expected = _upscale_palette(palette, scale)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
