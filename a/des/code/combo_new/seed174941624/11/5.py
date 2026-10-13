import math
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
MUTATION_STEPS = (6, 14)
PLACEMENT_RETRIES = 250

HOLE_PATTERNS = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)]],
    3: [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    r0, c0 = map(int, pts[0])
    seen[r0, c0] = True
    q = deque([(r0, c0)])
    seen_count = 1
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
                seen_count += 1
    return seen_count == int(mask.sum())


def _external_region(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    if h == 0 or w == 0:
        return seen
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    outside = np.zeros_like(padded, dtype=bool)
    outside[0, 0] = True
    q = deque([(0, 0)])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not padded[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if padded[r, c] or outside[r, c]:
                continue
            holes += 1
            outside[r, c] = True
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not padded[nr, nc] and not outside[nr, nc]:
                        outside[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _parse_legend(legend_zone: np.ndarray) -> list[tuple[int, int]]:
    entries = []
    for comp in _components(legend_zone != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        block = legend_zone[r0 : r1 + 1, c0 : c1 + 1]
        assert block.shape[0] == block.shape[1] and block.shape[0] >= 3
        border_mask = np.zeros_like(block, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        inner_mask = ~border_mask
        border_colors = set(map(int, np.unique(block[border_mask])))
        fill_colors = set(map(int, np.unique(block[inner_mask])))
        assert len(border_colors) == 1 and BLACK not in border_colors
        assert len(fill_colors) == 1 and BLACK not in fill_colors
        entries.append((r0, int(next(iter(fill_colors))), int(next(iter(border_colors)))))
    entries.sort(key=lambda item: item[0])
    return [(fill, border) for _, fill, border in entries]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object using the legend entry indexed by its enclosed hole count.

    Input structure:
    - A black main area is separated from a right-side legend area by one full-height maroon column.
    - The legend area contains a top-to-bottom stack of square swatches; each swatch has a one-pixel border color and a different interior fill color.
    - The main area contains several 4-connected grey objects, some of which enclose black holes.

    Rule:
    - Count the number of enclosed black holes inside each grey object.
    - Use that hole count as the legend index: the first swatch corresponds to 0 holes, the second to 1 hole, and so on.
    - Recolor the entire grey object to the swatch's interior color.
    - Then recolor only the object's outer contour pixels, meaning object pixels 4-neighbor-adjacent to the exterior black background, to the swatch's border color.
    - Leave the maroon separator column and the legend area unchanged.

    Key insight: The legend is a topology-driven lookup table: an object's hole count selects one fill color and one outer-border color.
    """
    g = np.asarray(grid, dtype=int)
    sep_cols = np.where(np.all(g == MAROON, axis=0))[0]
    assert len(sep_cols) >= 1
    sep = int(sep_cols[-1])
    main_area = g[:, :sep]
    legend_zone = g[:, sep + 1 :]
    legend = _parse_legend(legend_zone)
    assert legend

    exterior_black = _external_region(main_area == BLACK)
    out = g.copy()

    for comp in _components(main_area == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            obj_mask[r - r0, c - c0] = True
        holes = _count_holes(obj_mask)
        assert holes < len(legend)
        fill_color, border_color = legend[holes]
        for r, c in comp:
            out[r, c] = fill_color
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < main_area.shape[0] and 0 <= nc < main_area.shape[1] and exterior_black[nr, nc]:
                    out[r, c] = border_color
                    break

    return out


def _base_mask(target_holes: int, rng: random.Random) -> np.ndarray:
    if target_holes == 0:
        return np.ones((rng.randint(4, 6), rng.randint(4, 7)), dtype=bool)

    pattern = rng.choice(HOLE_PATTERNS[target_holes])
    row_count = max(r for r, _ in pattern) + 1
    col_count = max(c for _, c in pattern) + 1
    row_sizes = [rng.choice([1, 1, 2]) for _ in range(row_count)]
    col_sizes = [rng.choice([1, 1, 2]) for _ in range(col_count)]
    top = rng.choice([1, 1, 2])
    bottom = rng.choice([1, 1, 2])
    left = rng.choice([1, 1, 2])
    right = rng.choice([1, 1, 2])

    height = top + bottom + sum(row_sizes) + (row_count - 1)
    width = left + right + sum(col_sizes) + (col_count - 1)
    mask = np.ones((height, width), dtype=bool)

    row_starts = []
    cursor = top
    for size in row_sizes:
        row_starts.append(cursor)
        cursor += size + 1

    col_starts = []
    cursor = left
    for size in col_sizes:
        col_starts.append(cursor)
        cursor += size + 1

    for rr, cc in pattern:
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        mask[r0 : r0 + row_sizes[rr], c0 : c0 + col_sizes[cc]] = False

    return mask


def _boundary_add_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    ext = _external_region(~mask)
    h, w = mask.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or not ext[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    candidates.append((r, c))
                    break
    return candidates


def _boundary_remove_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    ext = _external_region(~mask)
    h, w = mask.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and ext[nr, nc]:
                    candidates.append((r, c))
                    break
    return candidates


def _accepted_shape(mask: np.ndarray, target_holes: int, min_area: int) -> bool:
    if int(mask.sum()) < min_area:
        return False
    if not _is_connected(mask):
        return False
    if _count_holes(mask) != target_holes:
        return False
    cropped = _crop(mask)
    return cropped.shape[0] <= 12 and cropped.shape[1] <= 12


def _random_object_mask(target_holes: int, rng: random.Random) -> np.ndarray:
    base = _base_mask(target_holes, rng)
    pad = 3
    canvas = np.zeros((base.shape[0] + 2 * pad, base.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + base.shape[0], pad : pad + base.shape[1]] = base
    min_area = max(6 + 4 * target_holes, int(base.sum() * 0.7))

    for _ in range(rng.randint(*MUTATION_STEPS)):
        if rng.random() < 0.62:
            candidates = _boundary_add_candidates(canvas)
            if not candidates:
                continue
            proposal = canvas.copy()
            r, c = rng.choice(candidates)
            proposal[r, c] = True
            if rng.random() < 0.35:
                dirs = list(DIR4)
                rng.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < proposal.shape[0] and 0 <= nc < proposal.shape[1] and not proposal[nr, nc]:
                        has_neighbor = False
                        for dr2, dc2 in DIR4:
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < proposal.shape[0] and 0 <= ac < proposal.shape[1] and proposal[ar, ac]:
                                has_neighbor = True
                                break
                        if has_neighbor:
                            proposal[nr, nc] = True
                            break
        else:
            candidates = _boundary_remove_candidates(canvas)
            if not candidates:
                continue
            proposal = canvas.copy()
            r, c = rng.choice(candidates)
            proposal[r, c] = False
            if rng.random() < 0.28:
                dirs = list(DIR4)
                rng.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < proposal.shape[0] and 0 <= nc < proposal.shape[1] and proposal[nr, nc]:
                        proposal[nr, nc] = False
                        break

        if _accepted_shape(proposal, target_holes, min_area):
            canvas = proposal

    mask = _crop(canvas)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.4:
        mask = mask.T
    return np.array(mask, dtype=bool)


def _place_objects(
    grid: np.ndarray,
    main_w: int,
    masks: list[np.ndarray],
    rng: random.Random,
) -> bool:
    h = grid.shape[0]
    occupied = np.zeros((h, main_w), dtype=bool)
    paint_order = sorted(enumerate(masks), key=lambda item: int(item[1].sum()), reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx, mask in paint_order:
        mh, mw = mask.shape
        if mh + 2 > h or mw + 2 > main_w:
            return False
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = rng.randint(1, h - mh - 1)
            c = rng.randint(1, main_w - mw - 1)
            rs = max(0, r - 1)
            re = min(h, r + mh + 1)
            cs = max(0, c - 1)
            ce = min(main_w, c + mw + 1)
            if occupied[rs:re, cs:ce].any():
                continue
            placements[idx] = (r, c)
            occupied[r : r + mh, c : c + mw] |= mask
            placed = True
            break
        if not placed:
            return False

    for idx, mask in enumerate(masks):
        r, c = placements[idx]
        grid[r : r + mask.shape[0], c : c + mask.shape[1]][mask] = GREY
    return True


def _pick_legend_pairs(entry_count: int, rng: random.Random) -> list[tuple[int, int]]:
    ordered_pairs = [(border, fill) for border in PALETTE_COLORS for fill in PALETTE_COLORS if border != fill]
    rng.shuffle(ordered_pairs)
    return ordered_pairs[:entry_count]


def _paint_legend(
    grid: np.ndarray,
    main_w: int,
    block_size: int,
    pairs: list[tuple[int, int]],
    rng: random.Random,
) -> None:
    top_margin = rng.randint(0, 1)
    legend_x = main_w + 1
    for i, (border, fill) in enumerate(pairs):
        r0 = top_margin + i * (block_size + 1)
        c0 = legend_x
        grid[r0 : r0 + block_size, c0 : c0 + block_size] = border
        grid[r0 + 1 : r0 + block_size - 1, c0 + 1 : c0 + block_size - 1] = fill
    grid[:, main_w] = MAROON


def _generated_hole_counts(grid: np.ndarray) -> list[int]:
    sep = int(np.where(np.all(grid == MAROON, axis=0))[0][-1])
    counts = []
    for comp in _components(grid[:, :sep] == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        counts.append(_count_holes(mask))
    counts.sort()
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    entry_count = 4
    block_size = rng.choice([3, 4])
    legend_height = rng.randint(0, 1) + entry_count * block_size + (entry_count - 1)

    hole_counts = [0, 1, 2, 3]
    if rng.random() < 0.5:
        hole_counts.append(rng.choice(hole_counts))
    rng.shuffle(hole_counts)

    masks = [_random_object_mask(hc, rng) for hc in hole_counts]
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    total_area = sum(int(mask.sum()) for mask in masks)

    max_main_w = 30 - 1 - block_size
    min_main_w = max(16, max_w + 2, int(math.ceil(math.sqrt(total_area))) + 5)
    if min_main_w > max_main_w:
        return None
    main_w = rng.randint(min_main_w, max_main_w)

    min_h = max(legend_height, max_h + 2, int(math.ceil(total_area / max(8, main_w - 4))) + 7)
    if min_h > 30:
        return None
    height = rng.randint(min_h, 30)

    grid = np.full((height, main_w + 1 + block_size), BLACK, dtype=int)
    pairs = _pick_legend_pairs(entry_count, rng)
    _paint_legend(grid, main_w, block_size, pairs, rng)

    if not _place_objects(grid, main_w, masks, rng):
        return None

    if _generated_hole_counts(grid) != sorted(hole_counts):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out[:, :main_w] == GREY):
        return None
    if not np.array_equal(out[:, main_w:], grid[:, main_w:]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC grid")
