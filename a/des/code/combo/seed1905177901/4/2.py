import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SHIFT_DELTAS = (("right", (0, 1)), ("down", (1, 0)))


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    return np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(grid)]
    assert len(colors) >= 2, "expected at least two colors for the checkerboard"

    rr, cc = np.indices(grid.shape)
    even_mask = ((rr + cc) & 1) == 0
    even_vals = grid[even_mask]
    odd_vals = grid[~even_mask]

    scores: dict[tuple[int, int], tuple[int, int]] = {}
    for even_color in colors:
        for odd_color in colors:
            if even_color == odd_color:
                continue
            aligned = int(np.count_nonzero(even_vals == even_color) + np.count_nonzero(odd_vals == odd_color))
            misaligned = int(np.count_nonzero(odd_vals == even_color) + np.count_nonzero(even_vals == odd_color))
            scores[(even_color, odd_color)] = (aligned - misaligned, aligned)

    best_score = max(scores.values())
    best_pairs = [pair for pair, score in scores.items() if score == best_score]
    assert len(best_pairs) == 1 and best_score[0] > 0, "checkerboard phase must be identifiable"
    return best_pairs[0]


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _find_components(grid: np.ndarray, fg_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros_like(fg_mask, dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if not fg_mask[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            queue = [(r, c)]
            visited[r, c] = True
            coords = []

            while queue:
                cr, cc = queue.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and fg_mask[nr, nc] and not visited[nr, nc]:
                        assert int(grid[nr, nc]) == color, "each foreground component must be monochromatic"
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            coords.sort()
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (
                        min(r0 for r0, _ in coords),
                        min(c0 for _, c0 in coords),
                    ),
                }
            )

    components.sort(key=lambda comp: comp["bbox"])
    return components


def _rotate_coords(coords: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    return [(h - 1 - r, w - 1 - c) for r, c in coords]


def _shift_coords(coords: list[tuple[int, int]], delta: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = delta
    return [(r + dr, c + dc) for r, c in coords]


def _in_bounds(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _neighbors_with_self(coords: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in ((0, 0),) + DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _placement_candidates(
    comp_coords: list[tuple[int, int]],
    grid: np.ndarray,
    bg: np.ndarray,
) -> list[tuple[str, list[tuple[int, int]]]]:
    h, w = grid.shape
    rotated = _rotate_coords(comp_coords, h, w)
    valid = []

    for name, delta in SHIFT_DELTAS:
        target = _shift_coords(rotated, delta)
        if not _in_bounds(target, h, w):
            continue
        if any(int(grid[r1, c1]) != int(bg[r1, c1]) for r1, c1 in target):
            continue
        if any(int(bg[r0, c0]) == int(bg[r1, c1]) for (r0, c0), (r1, c1) in zip(comp_coords, target)):
            continue
        valid.append((name, target))

    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Duplicate each checkerboard-overlaid foreground component to its point-reflected location
    with a one-cell phase-flipping offset.

    Input structure
    - The background is a global Azure/Black checkerboard.
    - Foreground objects are 4-connected monochromatic components whose colors differ from the two background colors.

    Rule
    - Preserve the input.
    - For each foreground component, rotate it 180 degrees around the grid center.
    - Test the two allowed one-cell offsets, right and down.
    - Keep the unique offset whose copy lands entirely on empty checkerboard cells and swaps the covered checkerboard parity.

    Key insight
    - The destination is not just the raw 180-degree rotation: the copied object must land on the opposite checkerboard phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    bg_even, bg_odd = _infer_checkerboard(g)
    bg = _checkerboard((h, w), bg_even, bg_odd)

    fg_mask = g != bg
    assert np.any(fg_mask), "expected at least one foreground object"
    assert not np.any(np.isin(g[fg_mask], [bg_even, bg_odd])), "foreground colors must differ from checkerboard colors"

    components = _find_components(g, fg_mask)
    out = g.copy()

    for comp in components:
        coords = comp["coords"]
        assert 1 <= len(coords) <= 6, "foreground components must have area at most 6"
        candidates = _placement_candidates(coords, g, bg)
        assert len(candidates) == 1, "each component must have exactly one valid phase-flipping offset"
        _name, target = candidates[0]

        for r, c in target:
            assert int(out[r, c]) == int(bg[r, c]), "copied components must not overlap"
            out[r, c] = comp["color"]

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
AZURE = 8
BLACK = 0
FG_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
GRID_SIZES = {
    0: np.array([8, 10, 12, 14], dtype=int),
    1: np.array([9, 11, 13, 15], dtype=int),
}
AREA_VALUES = np.array([1, 2, 3, 4, 5, 6], dtype=int)
AREA_WEIGHTS = np.array([1, 2, 4, 4, 3, 2], dtype=float)
AREA_WEIGHTS /= AREA_WEIGHTS.sum()
OBJECT_COUNT_VALUES = np.array([2, 3, 4, 5], dtype=int)
OBJECT_COUNT_WEIGHTS = np.array([1, 3, 3, 2], dtype=float)
OBJECT_COUNT_WEIGHTS /= OBJECT_COUNT_WEIGHTS.sum()
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 200


def _random_shape() -> list[tuple[int, int]]:
    area = int(_RNG.choice(AREA_VALUES, p=AREA_WEIGHTS))
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nbr = (r + dr, c + dc)
                if nbr not in cells:
                    frontier.append(nbr)
        choice = frontier[int(_RNG.integers(len(frontier)))]
        cells.add(choice)
    return _normalize(cells)


def _shape_extent(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _place_shape(shape: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + r, ac + c) for r, c in shape]


def _sample_sizes() -> tuple[int, int]:
    parity = int(_RNG.integers(2))
    choices = GRID_SIZES[parity]
    return int(_RNG.choice(choices)), int(_RNG.choice(choices))


def _make_base_grid(h: int, w: int) -> np.ndarray:
    if int(_RNG.integers(2)) == 0:
        return _checkerboard((h, w), AZURE, BLACK)
    return _checkerboard((h, w), BLACK, AZURE)


def _try_place_blocker(
    direction: str,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    occupied_source: set[tuple[int, int]],
    reserved_targets: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        shape = _random_shape()
        shape_h, shape_w = _shape_extent(shape)

        if direction == "right":
            if shape_h > h or shape_w > w - 1:
                continue
            anchor = (0, int(_RNG.integers(1, w - shape_w + 1)))
        else:
            if shape_h > h - 1 or shape_w > w:
                continue
            anchor = (int(_RNG.integers(1, h - shape_h + 1)), 0)

        source = _place_shape(shape, anchor)
        rotated = _rotate_coords(source, h, w)
        delta = dict(SHIFT_DELTAS)[direction]
        target = _shift_coords(rotated, delta)

        if not _in_bounds(target, h, w):
            continue
        if any(cell in forbidden for cell in source):
            continue
        target_forbidden = forbidden | _neighbors_with_self(source, h, w)
        if any(cell in target_forbidden for cell in target):
            continue
        if set(source) & reserved_targets:
            continue
        if set(target) & occupied_source:
            continue

        alt_name = "down" if direction == "right" else "right"
        alt_target = _shift_coords(rotated, dict(SHIFT_DELTAS)[alt_name])
        assert not _in_bounds(alt_target, h, w), "blockers should invalidate the alternate shift by boundary"

        return {"direction": direction, "source": source, "target": target}
    return None


def _try_place_interior(
    direction: str,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    occupied_source: set[tuple[int, int]],
    reserved_targets: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        shape = _random_shape()
        shape_h, shape_w = _shape_extent(shape)
        if shape_h > h - 2 or shape_w > w - 2:
            continue

        anchor = (
            int(_RNG.integers(1, h - shape_h + 1)),
            int(_RNG.integers(1, w - shape_w + 1)),
        )
        source = _place_shape(shape, anchor)
        rotated = _rotate_coords(source, h, w)
        delta = dict(SHIFT_DELTAS)[direction]
        target = _shift_coords(rotated, delta)
        alt_name = "down" if direction == "right" else "right"
        alt_target = _shift_coords(rotated, dict(SHIFT_DELTAS)[alt_name])

        if not _in_bounds(target, h, w):
            continue
        if not _in_bounds(alt_target, h, w):
            continue
        if any(cell in forbidden for cell in source):
            continue
        target_forbidden = forbidden | _neighbors_with_self(source, h, w)
        if any(cell in target_forbidden for cell in target):
            continue
        if set(source) & reserved_targets:
            continue
        if set(target) & occupied_source:
            continue
        if not (set(alt_target) & occupied_source):
            continue

        return {"direction": direction, "source": source, "target": target}
    return None


def _assemble_grid(h: int, w: int, placements: list[dict]) -> np.ndarray:
    grid = _make_base_grid(h, w)
    colors = _RNG.choice(FG_COLORS, size=len(placements), replace=False)
    for color, placement in zip(colors, placements):
        for r, c in placement["source"]:
            grid[r, c] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg_even, bg_odd = _infer_checkerboard(g)
    if {bg_even, bg_odd} != {AZURE, BLACK}:
        return False
    bg = _checkerboard(g.shape, bg_even, bg_odd)
    fg_mask = g != bg
    if not np.any(fg_mask):
        return False
    if np.any(np.isin(g[fg_mask], [AZURE, BLACK])):
        return False

    components = _find_components(g, fg_mask)
    if len(components) < 2:
        return False

    seen_dirs = set()
    occupied_targets: set[tuple[int, int]] = set()
    for comp in components:
        if len(comp["coords"]) > 6:
            return False
        candidates = _placement_candidates(comp["coords"], g, bg)
        if len(candidates) != 1:
            return False
        seen_dirs.add(candidates[0][0])
        target = set(candidates[0][1])
        if target & occupied_targets:
            return False
        occupied_targets |= target

    return seen_dirs == {"right", "down"}


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_sizes()
    n_objects = int(_RNG.choice(OBJECT_COUNT_VALUES, p=OBJECT_COUNT_WEIGHTS))

    if n_objects > len(FG_COLORS):
        return None

    placements: list[dict] = []
    occupied_source: set[tuple[int, int]] = set()
    reserved_targets: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    blocker_dirs = ["right", "down"]
    _RNG.shuffle(blocker_dirs)
    for direction in blocker_dirs:
        placed = _try_place_blocker(direction, h, w, forbidden, occupied_source, reserved_targets)
        if placed is None:
            return None
        placements.append(placed)
        occupied_source.update(placed["source"])
        reserved_targets.update(placed["target"])
        forbidden |= _neighbors_with_self(placed["source"], h, w)
        forbidden |= _neighbors_with_self(placed["target"], h, w)

    remaining = n_objects - len(placements)
    extra_dirs = []
    if remaining > 0:
        base = ["right", "down"] * ((remaining + 1) // 2)
        _RNG.shuffle(base)
        extra_dirs = base[:remaining]

    for direction in extra_dirs:
        placed = _try_place_interior(direction, h, w, forbidden, occupied_source, reserved_targets)
        if placed is None:
            return None
        placements.append(placed)
        occupied_source.update(placed["source"])
        reserved_targets.update(placed["target"])
        forbidden |= _neighbors_with_self(placed["source"], h, w)
        forbidden |= _neighbors_with_self(placed["target"], h, w)

    grid = _assemble_grid(h, w, placements)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-reflection input")
