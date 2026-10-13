from __future__ import annotations

import random

import numpy as np


BG = 0
FENCE = 5
PALETTE_COLORS = tuple(color for color in range(10) if color not in {BG, FENCE})

GRID_SIZE_BY_FRAME_COUNT = {
    1: (14, 30),
    2: (18, 30),
    3: (22, 30),
    4: (24, 30),
}
FRAME_SIZE_CAP = {
    1: 12,
    2: 10,
    3: 8,
    4: 7,
}
FRAME_COUNT_RANGE = (1, 4)
FRAME_MARGIN = 2
FRAME_PLACE_RETRIES = 200
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            points: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                points.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(points)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _read_palette_mapping(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert h >= 3 and w >= 4, "expected room for four 1x2 corner palette entries"

    corner_pairs = [
        ((0, 0), (0, 1)),
        ((0, w - 2), (0, w - 1)),
        ((h - 1, 0), (h - 1, 1)),
        ((h - 1, w - 2), (h - 1, w - 1)),
    ]

    mapping: dict[int, int] = {}
    palette_mask = np.zeros_like(grid, dtype=bool)
    for (r0, c0), (r1, c1) in corner_pairs:
        a = int(grid[r0, c0])
        b = int(grid[r1, c1])
        assert a != b, "each palette entry must contain two distinct colors"
        assert a not in {BG, FENCE} and b not in {BG, FENCE}, "palette colors must be visible non-grey colors"
        palette_mask[r0, c0] = True
        palette_mask[r1, c1] = True
        if a in mapping:
            assert mapping[a] == b, "palette color cannot map to multiple complements"
        if b in mapping:
            assert mapping[b] == a, "palette color cannot map to multiple complements"
        mapping[a] = b
        mapping[b] = a
    return mapping, palette_mask


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    r0, c0 = points.min(axis=0).tolist()
    r1, c1 = points.max(axis=0).tolist()
    if int(r1 - r0) < 2 or int(c1 - c0) < 2:
        return None

    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    assert colors.size > 0, "each fence interior must contain vote pixels"
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "vote winner must be unique"
    return int(winners[0])


def _draw_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1
    assert total >= parts
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    result: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        result.append(int(cut - prev))
        prev = cut
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey fence by interior majority vote, then recolor the fence with the winner's corner-palette complement.

    Input structure:
    - The background is black.
    - Each grid corner contains a horizontal 1x2 palette entry whose two colors are complementary to each other.
    - The central area contains one or more disjoint one-pixel-thick grey rectangular fences.
    - Inside each fence are scattered colored vote pixels on black background.

    Rule:
    - Read the four corner palette entries and build a two-way complement mapping from each adjacent color pair.
    - Find every grey connected component that forms an exact rectangular outline.
    - For each fence, count the non-black vote pixels inside it and choose the unique majority color as the winner.
    - Repaint the whole fence interior with the winner color, and repaint the grey fence outline with the winner's complementary color from the corner palette.
    - Leave the corner palette entries unchanged.

    Key insight:
    The corner decorations are a lookup table: they define the complement used to recolor each fence after its interior majority color is identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    complement, palette_mask = _read_palette_mapping(g)
    out = g.copy()

    fence_components = _components(g == FENCE)
    assert fence_components, "expected at least one grey fence"
    for points in fence_components:
        mask = np.zeros_like(g, dtype=bool)
        for r, c in points:
            mask[r, c] = True
        box = _rect_frame_bbox(mask)
        assert box is not None, "each grey component must be a one-pixel-thick rectangular fence"
        r0, c0, r1, c1 = box
        assert r0 > 0 and c0 > 0 and r1 < g.shape[0] - 1 and c1 < g.shape[1] - 1, "fences must lie away from the outer border"
        assert not np.any(palette_mask[r0 : r1 + 1, c0 : c1 + 1]), "fences must not overlap the corner palette"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "rectangular fence must enclose a non-empty interior"
        assert not np.any(interior == FENCE), "fence interiors must not contain nested grey pixels"

        votes = interior[(interior != BG) & (interior != FENCE)]
        winner = _majority_color(votes)
        assert winner in complement, "winner color must appear in the corner palette"

        out[r0 + 1 : r1, c0 + 1 : c1] = winner
        _draw_outline(out, r0, c0, r1, c1, complement[winner])

    out[palette_mask] = g[palette_mask]
    return out


def _palette_pairs() -> list[tuple[int, int]]:
    colors = list(PALETTE_COLORS)
    random.shuffle(colors)
    return [(int(colors[i]), int(colors[i + 1])) for i in range(0, len(colors), 2)]


def _paint_corner_palette(grid: np.ndarray, pairs: list[tuple[int, int]]) -> dict[int, int]:
    h, w = grid.shape
    slots = [
        ((0, 0), (0, 1)),
        ((0, w - 2), (0, w - 1)),
        ((h - 1, 0), (h - 1, 1)),
        ((h - 1, w - 2), (h - 1, w - 1)),
    ]
    mapping: dict[int, int] = {}
    for pair, ((r0, c0), (r1, c1)) in zip(pairs, slots, strict=False):
        a, b = pair
        grid[r0, c0] = a
        grid[r1, c1] = b
        mapping[a] = b
        mapping[b] = a
    return mapping


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    if r0 < FRAME_MARGIN or c0 < FRAME_MARGIN or r1 > h - 1 - FRAME_MARGIN or c1 > w - 1 - FRAME_MARGIN:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_frames(rows: int, cols: int, n_frames: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []
    max_side = FRAME_SIZE_CAP[n_frames]

    sizes: list[tuple[int, int]] = []
    for _ in range(n_frames):
        fh = random.randint(4, max_side)
        fw = random.randint(4, max_side)
        sizes.append((fh, fw))
    sizes.sort(key=lambda item: item[0] * item[1], reverse=True)

    for fh, fw in sizes:
        placed = False
        max_r0 = rows - fh - FRAME_MARGIN
        max_c0 = cols - fw - FRAME_MARGIN
        if max_r0 < FRAME_MARGIN or max_c0 < FRAME_MARGIN:
            return None
        for _ in range(FRAME_PLACE_RETRIES):
            r0 = random.randint(FRAME_MARGIN, max_r0)
            c0 = random.randint(FRAME_MARGIN, max_c0)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            frames.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return frames


def _paint_votes(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    winner: int,
) -> None:
    interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    area = len(interior_cells)
    assert area >= 4, "generator uses fence interiors of at least 2x2"

    vote_count = random.randint(1, area - 1)
    max_colors = min(4, (vote_count + 1) // 2)
    n_colors = random.randint(1, max_colors)

    colors = [winner]
    if n_colors > 1:
        others = random.sample([color for color in PALETTE_COLORS if color != winner], n_colors - 1)
        colors.extend(int(color) for color in others)

    if n_colors == 1:
        counts = [vote_count]
    else:
        min_winner = vote_count // 2 + 1
        max_winner = vote_count - (n_colors - 1)
        winner_count = random.randint(min_winner, max_winner)
        counts = [winner_count] + _positive_composition(vote_count - winner_count, n_colors - 1)

    chosen_cells = random.sample(interior_cells, vote_count)
    offset = 0
    for color, count in zip(colors, counts, strict=False):
        for r, c in chosen_cells[offset : offset + count]:
            grid[r, c] = int(color)
        offset += count


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    side_min, side_max = GRID_SIZE_BY_FRAME_COUNT[n_frames]
    rows = random.randint(side_min, side_max)
    cols = random.randint(side_min, side_max)

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)

    pairs = _palette_pairs()
    complement = _paint_corner_palette(inp, pairs)
    _paint_corner_palette(out, pairs)

    frames = _place_frames(rows, cols, n_frames)
    if frames is None:
        return None

    for r0, c0, r1, c1 in frames:
        winner = random.choice(PALETTE_COLORS)
        _draw_outline(inp, r0, c0, r1, c1, FENCE)
        _draw_outline(out, r0, c0, r1, c1, complement[winner])
        out[r0 + 1 : r1, c0 + 1 : c1] = winner
        _paint_votes(inp, r0, c0, r1, c1, winner)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
