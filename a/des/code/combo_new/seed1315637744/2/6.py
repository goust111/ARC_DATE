import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 200


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _choose_majority_color(grid: np.ndarray, region: list[tuple[int, int]]) -> int:
    counts: dict[int, int] = {}
    topmost: dict[int, tuple[int, int]] = {}
    for r, c in region:
        color = int(grid[r, c])
        if color in (BLACK, GRAY):
            continue
        counts[color] = counts.get(color, 0) + 1
        pos = (int(r), int(c))
        if color not in topmost or pos < topmost[color]:
            topmost[color] = pos

    assert counts, "each enclosed region must contain at least one vote pixel"
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    return min(candidates, key=lambda color: topmost[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed outline using the majority vote color from its enclosed interior.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping one-pixel-thick grey closed outlines,
      each forming either a rectangle or an L-shaped orthogonal polygon.
    - Inside every closed outline, a sparse set of colored single-pixel vote markers
      is scattered on black background.
    - Outside the grey outlines there are no non-background pixels.

    Rule:
    - Treat the grey outline pixels as walls, and find every non-grey connected region
      that is fully enclosed and does not touch the outer grid edge.
    - For each enclosed region, count the colors of its vote pixels and choose the
      most frequent one as the winner.
    - If multiple colors tie for the largest count, choose the tied color whose
      topmost vote pixel is higher in the grid.
    - Recolor the entire surrounding grey outline to that winner color, and erase all
      interior vote pixels back to black.

    Key insight:
    - The final boundary colors are not encoded on the boundaries themselves; they are
      determined only after identifying each enclosed region and aggregating its
      interior vote pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    gray_components = _components_mask(g == GRAY, connectivity=4)
    assert gray_components, "expected at least one grey outline"

    gray_index = np.full(g.shape, -1, dtype=int)
    for idx, comp in enumerate(gray_components):
        for r, c in comp:
            gray_index[r, c] = idx

    out = g.copy()
    used_boundaries: set[int] = set()
    enclosed_regions = 0
    h, w = map(int, g.shape)

    for region in _components_mask(g != GRAY, connectivity=4):
        if _touches_edge(region, g.shape):
            continue

        enclosed_regions += 1
        winner = _choose_majority_color(g, region)
        boundary_ids: set[int] = set()

        for r, c in region:
            out[r, c] = BLACK
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    idx = int(gray_index[nr, nc])
                    if idx != -1:
                        boundary_ids.add(idx)

        assert len(boundary_ids) == 1, "each enclosed region must belong to exactly one grey outline"
        boundary_id = next(iter(boundary_ids))
        assert boundary_id not in used_boundaries, "one grey outline cannot surround multiple disjoint interiors"
        used_boundaries.add(boundary_id)

        for r, c in gray_components[boundary_id]:
            out[r, c] = winner

    assert enclosed_regions > 0, "expected at least one enclosed region"
    assert len(used_boundaries) == len(gray_components), "every grey outline must enclose exactly one region"
    return out


def _outline_and_interior(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, mask.shape)
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if (
                r == 0
                or c == 0
                or r == h - 1
                or c == w - 1
                or not mask[r - 1, c]
                or not mask[r + 1, c]
                or not mask[r, c - 1]
                or not mask[r, c + 1]
            ):
                boundary[r, c] = True
    interior = mask & ~boundary
    return boundary, interior


def _make_l_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        h = int(rng.integers(7, 12))
        w = int(rng.integers(7, 12))
        arm_w = int(rng.integers(4, w - 1))
        arm_h = int(rng.integers(4, h - 1))
        orient = int(rng.integers(4))

        mask = np.zeros((h, w), dtype=bool)
        if orient == 0:
            mask[:, :arm_w] = True
            mask[h - arm_h :, :] = True
        elif orient == 1:
            mask[:, w - arm_w :] = True
            mask[h - arm_h :, :] = True
        elif orient == 2:
            mask[:, :arm_w] = True
            mask[:arm_h, :] = True
        else:
            mask[:, w - arm_w :] = True
            mask[:arm_h, :] = True

        _boundary, interior = _outline_and_interior(mask)
        if int(interior.sum()) >= 6 and len(_components_mask(interior, connectivity=4)) == 1:
            return mask
    raise RuntimeError("failed to build a valid L-shaped mask")


def _sample_coords(items: list[tuple[int, int]], count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if count == 0:
        return []
    idx = rng.choice(len(items), size=count, replace=False)
    idx = np.atleast_1d(idx)
    return [items[int(i)] for i in idx]


def _make_tie_votes(
    coords: list[tuple[int, int]], rng: np.random.Generator
) -> dict[tuple[int, int], int] | None:
    patterns = []
    if len(coords) >= 4:
        patterns.append((2, 2, 0))
    if len(coords) >= 5:
        patterns.append((2, 2, 1))
    if len(coords) >= 6:
        patterns.append((3, 3, 0))
    if len(coords) >= 7:
        patterns.append((3, 3, 1))
    if not patterns:
        return None

    winner_count, loser_count, third_count = patterns[int(rng.integers(len(patterns)))]
    ordered = sorted(coords)
    eligible_top = [
        cell
        for cell in ordered
        if sum(1 for r, _c in coords if r > cell[0]) >= loser_count + third_count
    ]
    if not eligible_top:
        return None

    top = eligible_top[int(rng.integers(min(3, len(eligible_top))))]
    below = [cell for cell in coords if cell[0] > top[0]]
    loser_cells = _sample_coords(below, loser_count, rng)
    used = {top, *loser_cells}

    minority_cells: list[tuple[int, int]] = []
    if third_count:
        minority_pool = [cell for cell in below if cell not in used]
        if len(minority_pool) < third_count:
            return None
        minority_cells = _sample_coords(minority_pool, third_count, rng)
        used.update(minority_cells)

    winner_pool = [cell for cell in coords if cell not in used]
    if len(winner_pool) < winner_count - 1:
        return None
    winner_cells = [top, *_sample_coords(winner_pool, winner_count - 1, rng)]

    color_count = 2 + int(third_count > 0)
    colors = [int(c) for c in rng.choice(PALETTE, size=color_count, replace=False)]
    votes = {cell: colors[0] for cell in winner_cells}
    votes.update({cell: colors[1] for cell in loser_cells})
    for cell in minority_cells:
        votes[cell] = colors[2]
    return votes


def _make_majority_votes(
    coords: list[tuple[int, int]], rng: np.random.Generator
) -> dict[tuple[int, int], int] | None:
    two_color_patterns = [(3, 1), (3, 2), (4, 1), (4, 2), (5, 2)]
    three_color_patterns = [(3, 1, 1), (3, 2, 1), (4, 1, 1), (4, 2, 1)]

    patterns = [p for p in two_color_patterns if sum(p) <= len(coords)]
    if len(coords) >= 5:
        patterns.extend(p for p in three_color_patterns if sum(p) <= len(coords))
    if not patterns:
        return None

    counts = patterns[int(rng.integers(len(patterns)))]
    colors = [int(c) for c in rng.choice(PALETTE, size=len(counts), replace=False)]
    chosen_cells = _sample_coords(coords, sum(counts), rng)
    rng.shuffle(chosen_cells)

    votes: dict[tuple[int, int], int] = {}
    start = 0
    for color, count in zip(colors, counts):
        for cell in chosen_cells[start : start + count]:
            votes[cell] = color
        start += count
    return votes


def _make_votes(coords: list[tuple[int, int]], rng: np.random.Generator) -> dict[tuple[int, int], int]:
    want_tie = bool(rng.integers(2))
    votes = _make_tie_votes(coords, rng) if want_tie else _make_majority_votes(coords, rng)
    if votes is None:
        votes = _make_majority_votes(coords, rng) if want_tie else _make_tie_votes(coords, rng)
    assert votes is not None, "failed to build vote pattern"
    return votes


def _make_object(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        if bool(rng.integers(2)):
            filled = np.ones((int(rng.integers(6, 11)), int(rng.integers(6, 11))), dtype=bool)
        else:
            filled = _make_l_mask(rng)

        boundary, interior = _outline_and_interior(filled)
        interior_coords = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        if len(interior_coords) < 6:
            continue

        votes = _make_votes(interior_coords, rng)
        if len(votes) >= len(interior_coords):
            continue

        obj = np.zeros(filled.shape, dtype=int)
        obj[boundary] = GRAY
        for (r, c), color in votes.items():
            obj[r, c] = color
        return obj
    raise RuntimeError("failed to build a valid object")


def _place_object(grid: np.ndarray, blocked: np.ndarray, obj: np.ndarray, rng: np.random.Generator) -> bool:
    gh, gw = map(int, grid.shape)
    oh, ow = map(int, obj.shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, gh - oh):
        for c0 in range(1, gw - ow):
            if blocked[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1].any():
                continue
            candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = candidates[int(rng.integers(len(candidates)))]
    window = grid[r0 : r0 + oh, c0 : c0 + ow]
    mask = obj != BLACK
    window[mask] = obj[mask]
    blocked[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_objects = int(rng.integers(1, 4))
    min_side = 12 + 5 * (num_objects - 1)
    height = int(rng.integers(min_side, 31))
    width = int(rng.integers(min_side, 31))
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    for _ in range(num_objects):
        obj = _make_object(rng)
        if not _place_object(grid, blocked, obj, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.any(out == GRAY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
