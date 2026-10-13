import random
from collections import Counter

import numpy as np


RED = 2
PALETTE = tuple(range(10))
SIZE_OPTIONS = tuple(range(15, 30, 2))
COUNT_PATTERNS = (
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 2),
    (1, 2, 3),
    (1, 2, 4),
    (2, 2, 3),
)
QUADRANTS = ("nw", "ne", "sw", "se")
VECTORS = {
    "nw": (-1, -1),
    "ne": (-1, 1),
    "sw": (1, -1),
    "se": (1, 1),
}
GEN_ATTEMPTS = 400
ASSIGN_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rows = np.flatnonzero(np.all(grid == color, axis=1))
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((color, int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, "expected exactly one full-row/full-column cross"
    return candidates[0]


def _quadrant_of(r: int, c: int, axis_r: int, axis_c: int) -> str:
    assert r != axis_r and c != axis_c, "seed cannot lie on the cross"
    if r < axis_r and c < axis_c:
        return "nw"
    if r < axis_r and c > axis_c:
        return "ne"
    if r > axis_r and c < axis_c:
        return "sw"
    return "se"


def _branch_cells(
    shape: tuple[int, int], r: int, c: int, dr: int, dc: int, length: int
) -> set[tuple[int, int]]:
    h, w = shape
    end_c = c + dc * (length - 1)
    cells: set[tuple[int, int]] = set()
    for step in range(length):
        cc = c + dc * step
        if 0 <= cc < w:
            cells.add((r, cc))
    for step in range(length):
        rr = r + dr * step
        if 0 <= rr < h and 0 <= end_c < w:
            cells.add((rr, end_c))
    return cells


def _branch_is_clipped(
    shape: tuple[int, int], r: int, c: int, dr: int, dc: int, length: int
) -> bool:
    return len(_branch_cells(shape, r, c, dr, dc, length)) < 2 * length - 1


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = shape
    result: set[tuple[int, int]] = set()
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < h and 0 <= cc < w:
                result.add((rr, cc))
    return result


def _expanded(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        halo |= _neighbors8(shape, r, c)
    return halo


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each colored seed into an outward L branch whose length equals the global count of that color.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms a full central cross: a complete row and a
      complete column intersecting at the grid center.
    - Every other non-background pixel is a single-cell colored seed placed in
      one of the four quadrants defined by that cross.

    Rule:
    - Keep the cross color unchanged.
    - For each seed color c, count how many seed pixels of color c exist in the
      whole grid; call that count n.
    - Each seed grows away from the center according to its quadrant: top-left
      seeds grow left then up, top-right seeds right then up, bottom-left seeds
      left then down, and bottom-right seeds right then down.
    - From each seed, draw an L made of n horizontal pixels starting at the
      seed, then n vertical pixels starting from the horizontal segment's far
      endpoint; crop any out-of-bounds part at the grid border.

    Key insight:
    - The branch direction is a local spatial rule from the seed's quadrant,
      while the branch length is a global rule set by the total count of that
      same color across all quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    cross_color, axis_r, axis_c = _find_cross(g, bg)

    seed_mask = (g != bg) & (g != cross_color)
    seed_coords = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    assert seed_coords, "expected at least one seed"

    color_counts = Counter(int(g[r, c]) for r, c in seed_coords)
    out = g.copy()
    for r, c in sorted(seed_coords):
        color = int(g[r, c])
        length = int(color_counts[color])
        quadrant = _quadrant_of(r, c, axis_r, axis_c)
        dr, dc = VECTORS[quadrant]
        for rr, cc in _branch_cells(g.shape, r, c, dr, dc, length):
            out[rr, cc] = color

    out[axis_r, :] = cross_color
    out[:, axis_c] = cross_color
    return out


def _choose_counts() -> tuple[int, ...]:
    return random.choice(COUNT_PATTERNS)


def _assign_quadrants(counts: tuple[int, ...]) -> list[dict] | None:
    need = set(QUADRANTS)
    for _ in range(ASSIGN_ATTEMPTS):
        items: list[dict] = []
        used: set[str] = set()
        per_quadrant = {quadrant: 0 for quadrant in QUADRANTS}
        for color_index, count in enumerate(counts):
            if count == 1:
                chosen = [random.choice(QUADRANTS)]
            else:
                distinct = random.randint(2, min(4, count))
                base = random.sample(QUADRANTS, distinct)
                chosen = base[:]
                while len(chosen) < count:
                    chosen.append(random.choice(base))
                random.shuffle(chosen)
            for quadrant in chosen:
                items.append(
                    {
                        "color_index": color_index,
                        "length": count,
                        "quadrant": quadrant,
                    }
                )
                used.add(quadrant)
                per_quadrant[quadrant] += 1
        if used == need and max(per_quadrant.values()) <= 3:
            return items
    return None


def _quadrant_cells(axis_r: int, axis_c: int, h: int, w: int, quadrant: str) -> list[tuple[int, int]]:
    if quadrant == "nw":
        return [(r, c) for r in range(axis_r) for c in range(axis_c)]
    if quadrant == "ne":
        return [(r, c) for r in range(axis_r) for c in range(axis_c + 1, w)]
    if quadrant == "sw":
        return [(r, c) for r in range(axis_r + 1, h) for c in range(axis_c)]
    return [(r, c) for r in range(axis_r + 1, h) for c in range(axis_c + 1, w)]


def _place_items(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    colors: list[int],
    items: list[dict],
) -> list[dict] | None:
    growable_ids = [i for i, item in enumerate(items) if item["length"] > 1]
    require_clip = random.choice([False, True])
    clip_id = random.choice(growable_ids) if require_clip and growable_ids else None
    full_id = random.choice(growable_ids) if growable_ids else None
    if clip_id is not None and full_id == clip_id and len(growable_ids) > 1:
        full_id = random.choice([i for i in growable_ids if i != clip_id])

    for index, item in enumerate(items):
        if index == clip_id:
            item["clip_pref"] = True
        elif index == full_id:
            item["clip_pref"] = False
        else:
            item["clip_pref"] = None

    def priority(entry: dict) -> tuple[int, int]:
        pref = entry["clip_pref"]
        if pref is True:
            rank = 0
        elif pref is False:
            rank = 1
        else:
            rank = 2
        return rank, -int(entry["length"])

    ordered = sorted(items, key=priority)
    reserved_halo: set[tuple[int, int]] = set()
    seed_halo: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for item in ordered:
        quadrant = item["quadrant"]
        length = int(item["length"])
        color = int(colors[item["color_index"]])
        dr, dc = VECTORS[quadrant]
        candidates = _quadrant_cells(axis_r, axis_c, h, w, quadrant)
        random.shuffle(candidates)

        pref = item["clip_pref"]
        if pref is not None:
            filtered = [
                (r, c)
                for r, c in candidates
                if _branch_is_clipped((h, w), r, c, dr, dc, length) == pref
            ]
            if filtered:
                candidates = filtered

        placed_here = False
        for r, c in candidates:
            if (r, c) in seed_halo:
                continue
            branch = _branch_cells((h, w), r, c, dr, dc, length)
            if branch & reserved_halo:
                continue
            clipped = len(branch) < 2 * length - 1
            placed.append(
                {
                    "r": r,
                    "c": c,
                    "color": color,
                    "length": length,
                    "quadrant": quadrant,
                    "clipped": clipped,
                }
            )
            reserved_halo |= _expanded((h, w), branch)
            seed_halo |= _neighbors8((h, w), r, c)
            placed_here = True
            break

        if not placed_here:
            return None

    if clip_id is not None and not any(seed["length"] > 1 and seed["clipped"] for seed in placed):
        return None
    if not any(seed["length"] > 1 and not seed["clipped"] for seed in placed):
        return None
    return placed


def _build_grid() -> np.ndarray | None:
    h = random.choice(SIZE_OPTIONS)
    w = random.choice(SIZE_OPTIONS)
    axis_r = h // 2
    axis_c = w // 2

    bg = random.choice([color for color in PALETTE if color != RED])
    counts = _choose_counts()
    available = [color for color in PALETTE if color not in (bg, RED)]
    colors = random.sample(available, len(counts))

    items = _assign_quadrants(counts)
    if items is None:
        return None

    placed = _place_items(h, w, axis_r, axis_c, colors, items)
    if placed is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[axis_r, :] = RED
    grid[:, axis_c] = RED
    for seed in placed:
        grid[int(seed["r"]), int(seed["c"])] = int(seed["color"])

    # Reject layouts where the input no longer looks like isolated single-pixel seeds.
    seed_mask = (grid != bg) & (grid != RED)
    for r, c in np.argwhere(seed_mask):
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) != (r, c) and seed_mask[rr, cc]:
                    return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    counts_seen = Counter(int(grid[r, c]) for r, c in np.argwhere(seed_mask))
    if len(counts_seen) < 2 or len(set(counts_seen.values())) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
