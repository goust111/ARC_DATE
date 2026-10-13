import random
from collections import Counter

import numpy as np


GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GEN_ATTEMPTS = 400
ARC_COLORS = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(vals, minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            part: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                part.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(part)
    return parts


def _detect_border_color(grid: np.ndarray, bg: int) -> int:
    if np.any(grid == GREY):
        return GREY
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected at least one foreground color"
    return max(counts, key=counts.get)


def _find_palette(grid: np.ndarray, bg: int, border: int) -> tuple[dict[int, int], list[int]]:
    h, w = grid.shape
    colored = (grid != bg) & (grid != border)
    candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []

    for r in range(h - 1):
        pair_cols = {c for c in range(w) if colored[r, c] and colored[r + 1, c]}
        if len(pair_cols) < 2:
            continue

        for start in sorted(pair_cols):
            if start - 2 in pair_cols:
                continue

            run = [start]
            cur = start
            while (
                cur + 2 in pair_cols
                and cur + 1 < w
                and int(grid[r, cur + 1]) == bg
                and int(grid[r + 1, cur + 1]) == bg
            ):
                cur += 2
                run.append(cur)

            if len(run) < 2:
                continue

            c0 = run[0]
            c1 = run[-1]
            valid = True
            for c in range(c0, c1 + 1):
                if (c - c0) % 2 == 0:
                    if c not in pair_cols:
                        valid = False
                        break
                elif int(grid[r, c]) != bg or int(grid[r + 1, c]) != bg:
                    valid = False
                    break
            if not valid:
                continue

            if c0 > 0 and (int(grid[r, c0 - 1]) != bg or int(grid[r + 1, c0 - 1]) != bg):
                continue
            if c1 + 1 < w and (int(grid[r, c1 + 1]) != bg or int(grid[r + 1, c1 + 1]) != bg):
                continue
            if not (r == 0 or r + 1 == h - 1 or c0 == 0 or c1 == w - 1):
                continue

            pairs = [(int(grid[r, c]), int(grid[r + 1, c])) for c in run]
            if any(src in (bg, border) or dst in (bg, border) for src, dst in pairs):
                continue

            edge_score = int(r == 0 or r + 1 == h - 1) + int(c0 == 0) + int(c1 == w - 1)
            candidates.append((len(run), edge_score, r, pairs))

    assert candidates, "failed to detect palette legend"
    _, _, _, pairs = max(candidates, key=lambda item: (item[0], item[1], -item[2]))

    mapping = {src: dst for src, dst in pairs}
    tie_order: list[int] = []
    for _, dst in pairs:
        if dst not in tie_order:
            tie_order.append(dst)
    return mapping, tie_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed region's vote pixels through the edge legend, then fill the region with the mapped majority color.

    Input structure:
    - The dominant color is the background.
    - Several one-pixel-thick grey borders form separate closed regions whose interiors contain background plus a few colored vote pixels.
    - A legend on the grid edge is made of left-to-right vertical color pairs: the top cell of each pair is an input vote color and the bottom cell is its mapped color.

    Rule:
    - Read the legend as a color mapping from each pair's top color to its bottom color.
    - Treat every non-grey connected component that does not touch the outer grid boundary as one region interior.
    - For each region, map every vote pixel through the legend, count the mapped colors, break ties by the leftmost winning legend pair, and fill the whole region interior with that chosen color.

    Key insight:
    - The visible votes are not counted directly; they must be translated by the edge legend first, so the winning fill color can differ from the raw majority color inside the region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    border = _detect_border_color(g, bg)
    mapping, tie_order = _find_palette(g, bg, border)
    tie_index = {color: idx for idx, color in enumerate(tie_order)}

    out = g.copy()
    for part in _components(g != border):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in part):
            continue

        mapped_counts: Counter[int] = Counter()
        for r, c in part:
            color = int(g[r, c])
            if color == bg or color == border or color not in mapping:
                continue
            mapped_counts[mapping[color]] += 1

        if not mapped_counts:
            continue

        fill = min(mapped_counts, key=lambda color: (-mapped_counts[color], tie_index.get(color, 10**9)))
        for r, c in part:
            out[r, c] = fill

    return out


def _grow_blob(rng: random.Random, h: int, w: int, target: int) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        frontier_list = sorted(frontier)
        weights = []
        for r, c in frontier_list:
            degree = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
            weights.append(degree * degree + 1)
        cells.add(rng.choices(frontier_list, weights=weights, k=1)[0])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _bbox_dims(mask: np.ndarray) -> tuple[int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _make_region_patch(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 7)
        area = inner_h * inner_w
        target = rng.randint(6, min(area, max(6, (2 * area) // 3)))
        inner = _grow_blob(rng, inner_h, inner_w, target)
        bbox_h, bbox_w = _bbox_dims(inner)
        if bbox_h < 2 or bbox_w < 2:
            continue

        interior = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        interior[1:-1, 1:-1] = inner
        footprint = _dilate8(interior)
        border = footprint & ~interior
        if border.sum() == 0 or interior.sum() < 6:
            continue
        return border, interior
    raise RuntimeError("failed to build a valid region patch")


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _valid_positions(blocked: np.ndarray, ph: int, pw: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if not blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                positions.append((r0, c0))
    return positions


def _sample_unique_votes(rng: random.Random, source_colors: list[int], max_votes: int) -> list[int]:
    assert max_votes >= 3
    while True:
        votes = [rng.choice(source_colors) for _ in range(rng.randint(3, min(5, max_votes)))]
        counts = Counter(votes)
        if len(counts) < 2:
            continue
        best = counts.most_common()
        if len(best) >= 2 and best[0][1] == best[1][1]:
            continue
        return votes


def _sample_tie_votes(rng: random.Random, source_colors: list[int], max_votes: int) -> list[int]:
    assert max_votes >= 2
    color_count = len(source_colors)

    if color_count >= 3 and max_votes >= 3 and rng.random() < 0.5:
        idxs = sorted(rng.sample(range(color_count), 3))
        votes = [source_colors[i] for i in idxs]
        rng.shuffle(votes)
        return votes

    idx0, idx1 = sorted(rng.sample(range(color_count), 2))
    reps = 2 if max_votes >= 4 and rng.random() < 0.6 else 1
    votes = [source_colors[idx0]] * reps + [source_colors[idx1]] * reps
    rng.shuffle(votes)
    return votes


def _winner_color(votes: list[int], mapping: dict[int, int], tie_order: list[int]) -> int:
    order_idx = {color: idx for idx, color in enumerate(tie_order)}
    mapped_counts = Counter(mapping[color] for color in votes)
    return min(mapped_counts, key=lambda color: (-mapped_counts[color], order_idx[color]))


def _place_legend(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    source_colors: list[int],
    mapped_colors: list[int],
) -> None:
    h, w = grid.shape
    legend_w = 2 * len(source_colors) - 1
    edge = rng.choice(["top", "bottom", "left", "right"])

    if edge == "top":
        r0 = 0
        c0 = rng.randint(0, w - legend_w)
    elif edge == "bottom":
        r0 = h - 2
        c0 = rng.randint(0, w - legend_w)
    elif edge == "left":
        r0 = rng.randint(0, h - 2)
        c0 = 0
    else:
        r0 = rng.randint(0, h - 2)
        c0 = w - legend_w

    for idx, (src, dst) in enumerate(zip(source_colors, mapped_colors)):
        cc = c0 + 2 * idx
        grid[r0, cc] = src
        grid[r0 + 1, cc] = dst

    _mark_blocked(blocked, r0, c0, 2, legend_w)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.choice([3, 4])
    grid_h = rng.randint(16, 28)
    grid_w = rng.randint(16, 30)
    region_count = rng.choice([2, 3])
    tie_region_idx = rng.randrange(region_count) if rng.choice([False, True]) else None

    bg_choices = [color for color in ARC_COLORS if color != GREY]
    bg = rng.choice(bg_choices)
    palette_pool = [color for color in ARC_COLORS if color not in (bg, GREY)]
    source_colors = rng.sample(palette_pool, color_count)
    shift = rng.randint(1, color_count - 1)
    mapped_colors = source_colors[shift:] + source_colors[:shift]
    mapping = dict(zip(source_colors, mapped_colors))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _place_legend(rng, grid, blocked, source_colors, mapped_colors)

    fill_specs: list[tuple[list[tuple[int, int]], int]] = []
    tie_order = mapped_colors[:]

    for region_idx in range(region_count):
        placed = False
        for _ in range(80):
            border_mask, interior_mask = _make_region_patch(rng)
            ph, pw = border_mask.shape
            positions = _valid_positions(blocked, ph, pw)
            if not positions:
                return None

            r0, c0 = rng.choice(positions)
            interior_cells = [
                (r0 + int(r), c0 + int(c))
                for r, c in np.argwhere(interior_mask)
            ]
            if len(interior_cells) < 6:
                continue

            if region_idx == tie_region_idx:
                votes = _sample_tie_votes(rng, source_colors, len(interior_cells))
            else:
                votes = _sample_unique_votes(rng, source_colors, len(interior_cells))

            vote_cells = rng.sample(interior_cells, len(votes))
            fill_color = _winner_color(votes, mapping, tie_order)

            for rr, cc in np.argwhere(border_mask):
                grid[r0 + int(rr), c0 + int(cc)] = GREY
            for (rr, cc), color in zip(vote_cells, votes):
                grid[rr, cc] = color

            fill_specs.append((interior_cells, fill_color))
            _mark_blocked(blocked, r0, c0, ph, pw)
            placed = True
            break

        if not placed:
            return None

    if _dominant_color(grid) != bg:
        return None

    expected = grid.copy()
    for cells, fill_color in fill_specs:
        for r, c in cells:
            expected[r, c] = fill_color

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
