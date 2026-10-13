from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MAROON = 9

SEED_COLOR_POOL = [2, 3, 4, 6, 7, 8]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PATTERN_LIBRARY = [
    [(-1, 0), (1, 0), (0, -1), (0, 1)],
    [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    [(-1, 0), (1, 0)],
    [(0, -1), (0, 1)],
    [(-1, 0), (0, 1), (-1, 1)],
    [(1, 0), (0, -1), (1, -1)],
    [(-1, 1), (0, 1), (1, 1)],
    [(-1, -1), (0, -1), (1, -1)],
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert rows, "expected a full grey separator row"
    return int(rows[0])


def _parse_legend(legend: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    mapping: dict[int, list[tuple[int, int]]] = {}
    h, w = legend.shape
    for r in range(h):
        for c in range(w):
            v = int(legend[r, c])
            if v in (BLACK, GREY, MAROON):
                continue
            assert 0 <= r - 1 and r + 1 < h and c + 4 < w
            window = legend[r - 1 : r + 2, c + 2 : c + 5]
            offsets = []
            for wr in range(3):
                for wc in range(3):
                    if int(window[wr, wc]) == MAROON:
                        dr, dc = wr - 1, wc - 1
                        if (dr, dc) != (0, 0):
                            offsets.append((dr, dc))
            assert offsets, f"legend entry for color {v} must expose at least one growth offset"
            mapping[v] = offsets
    assert mapping, "legend parsing found no entries"
    return mapping


def _grow_region(
    grid: np.ndarray,
    region: set[tuple[int, int]],
    seed: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
) -> None:
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region:
                continue
            if int(grid[nr, nc]) != BLACK:
                continue
            grid[nr, nc] = color
            q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each seed inside its fenced region by the legend-defined neighborhood mask.

    Input structure:
    - Background is black.
    - A full grey separator row splits a top legend zone from the main play area.
    - Each legend entry contains one seed-colored pixel on the left and a 3x3 maroon/black mask on the right; maroon cells mark allowed one-step growth directions around the current filled pixel.
    - The main area contains several closed blue fence loops, each enclosing black interior cells and exactly one non-black, non-blue single-pixel seed whose color matches a legend entry.

    Rule:
    - Read the legend as a mapping from seed color to allowed neighbor offsets.
    - In the main area, treat blue pixels as blocking fence walls and find every enclosed non-blue region that does not touch the main-area border.
    - For each enclosed region, locate its unique colored seed and repeatedly expand from every filled seed-colored pixel into black neighbors reachable by that seed's legend mask.
    - Stop when no new black cells can be filled; keep the legend and blue fences unchanged.

    Key insight:
    - The 3x3 legend patch is not copied as a static stamp: it is a directional diffusion rule. Fence geometry defines the available space, while the legend mask determines which interior cells are actually reachable from the seed.
    """
    g = np.asarray(grid, dtype=int).copy()
    sep = _find_separator_row(g)
    legend = g[:sep, :]
    main_area = g[sep + 1 :, :].copy()
    mapping = _parse_legend(legend)

    regions = _components(main_area != BLUE)
    for comp in regions:
        if any(r == 0 or c == 0 or r == main_area.shape[0] - 1 or c == main_area.shape[1] - 1 for r, c in comp):
            continue
        seeds = [(r, c) for r, c in comp if int(main_area[r, c]) not in (BLACK, BLUE)]
        assert len(seeds) == 1, "each fenced interior must contain exactly one colored seed"
        sr, sc = seeds[0]
        color = int(main_area[sr, sc])
        assert color in mapping, f"missing legend entry for seed color {color}"
        _grow_region(main_area, set(comp), (sr, sc), color, mapping[color])

    g[sep + 1 :, :] = main_area
    return g


def _simulate_fill(interior: np.ndarray, seed: tuple[int, int], offsets: list[tuple[int, int]]) -> np.ndarray:
    filled = np.zeros_like(interior, dtype=bool)
    q = deque([seed])
    filled[seed] = True
    while q:
        r, c = q.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1]):
                continue
            if not interior[nr, nc] or filled[nr, nc]:
                continue
            filled[nr, nc] = True
            q.append((nr, nc))
    return filled


def _has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    q = deque()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(empty & ~seen))


def _grow_blob(h: int, w: int, target: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier
        cells.add(rng.choice(list(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_blob(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 7)
        min_area = max(4, (h * w + 2) // 3)
        max_area = max(min_area, h * w - 1)
        area = rng.randint(min_area, max_area)
        blob = _grow_blob(h, w, area, rng)
        if _has_hole(blob):
            continue
        rows, cols = np.where(blob)
        bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        if bbox_area == int(blob.sum()) and rng.random() < 0.8:
            continue
        return blob
    raise RuntimeError("failed to generate a hole-free blob")


def _boundary_from_interior(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in zip(*np.where(interior), strict=False):
        r = int(r)
        c = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not interior[nr, nc]:
                boundary[nr, nc] = True
    return boundary


def _build_patch(color: int, offsets: list[tuple[int, int]], rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        core = _make_blob(rng)
        interior = np.pad(core, 1)
        total_area = int(interior.sum())
        candidates = []
        partial_candidates = []
        for sr, sc in zip(*np.where(interior), strict=False):
            seed = (int(sr), int(sc))
            filled = _simulate_fill(interior, seed, offsets)
            fill_area = int(filled.sum())
            if fill_area < 2:
                continue
            item = (seed, fill_area, filled)
            candidates.append(item)
            if fill_area < total_area:
                partial_candidates.append(item)
        if not candidates:
            continue

        pool = partial_candidates if partial_candidates and rng.random() < 0.75 else candidates
        seed, fill_area, filled = rng.choice(pool)
        boundary = _boundary_from_interior(interior)
        patch = np.full(interior.shape, BLACK, dtype=int)
        patch[boundary] = BLUE
        patch[seed] = color
        return {
            "patch": patch,
            "interior": interior,
            "fill_area": fill_area,
            "interior_area": total_area,
        }
    raise RuntimeError("failed to build a valid fence patch")


def _place_patches(
    main_h: int, width: int, patches: list[np.ndarray], rng: random.Random
) -> list[tuple[int, int]] | None:
    reserved = np.zeros((main_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph + 2 > main_h or pw + 2 > width:
            return None
        spots = []
        for r0 in range(1, main_h - ph):
            for c0 in range(1, width - pw):
                rr0 = r0 - 1
                rr1 = r0 + ph + 1
                cc0 = c0 - 1
                cc1 = c0 + pw + 1
                if not reserved[rr0:rr1, cc0:cc1].any():
                    spots.append((r0, c0))
        if not spots:
            return None
        r0, c0 = rng.choice(spots)
        placements[idx] = (r0, c0)
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True

    return [p for p in placements if p is not None]


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random()
        entry_count = rng.randint(2, 4)
        colors = rng.sample(SEED_COLOR_POOL, entry_count)
        patterns = rng.sample(PATTERN_LIBRARY, entry_count)

        objects = [_build_patch(color, offsets, rng) for color, offsets in zip(colors, patterns, strict=False)]
        if not any(int(obj["fill_area"]) < int(obj["interior_area"]) for obj in objects):
            continue

        legend_gap = rng.randint(1, 2)
        legend_width = entry_count * 5 + (entry_count - 1) * legend_gap
        min_width = max(legend_width + 2, max(obj["patch"].shape[1] for obj in objects) + 4, 18)
        width = rng.randint(min_width, 30)
        main_min_h = max(max(obj["patch"].shape[0] for obj in objects) + 3, 11)
        main_h = rng.randint(main_min_h, 24)

        placements = _place_patches(main_h, width, [obj["patch"] for obj in objects], rng)
        if placements is None:
            continue

        grid = np.full((main_h + 6, width), BLACK, dtype=int)
        grid[5, :] = GREY

        legend_left = rng.randint(0, width - legend_width)
        cursor = legend_left
        for color, offsets in zip(colors, patterns, strict=False):
            seed_r, seed_c = 2, cursor
            grid[seed_r, seed_c] = color
            center_c = seed_c + 3
            for dr, dc in offsets:
                grid[seed_r + dr, center_c + dc] = MAROON
            cursor += 5 + legend_gap

        for (r0, c0), obj in zip(placements, objects, strict=False):
            patch = obj["patch"]
            ph, pw = patch.shape
            target = grid[6 + r0 : 6 + r0 + ph, c0 : c0 + pw]
            mask = patch != BLACK
            target[mask] = patch[mask]

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if grid.shape != out.shape or max(grid.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
