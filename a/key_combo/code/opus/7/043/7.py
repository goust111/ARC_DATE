import random

import numpy as np


BG = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4)
DISTRACTOR_COLORS = (6, 7, 8, 9)
RING_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_SIDE_RANGE = (17, 22)
SOURCE_COUNT_OPTIONS = (3, 4)
SOURCE_AREA_RANGE = range(2, 11)
DISTRACTOR_COUNT_OPTIONS = (1, 2)
MAX_PATCH_SIDE = 5
GENERATION_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _foreground_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _stamp_ring(out: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = out.shape
    assert 0 < r < h - 1 and 0 < c < w - 1, "ring centers must have a full 3x3 neighborhood"
    out[r, c] = int(color)
    for dr, dc in RING_OFFSETS:
        out[r + dr, c + dc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the source blobs and keep only 3x3 grey rings centered on markers whose anchor color comes from a median-sized source or larger.

    Input structure:
    - The grid has a black background with several 4-connected non-black components that are mostly grey and each contain exactly one colored anchor pixel.
    - Separate isolated single-pixel markers are scattered elsewhere; some match source anchor colors and some are distractors.
    - Matching marker centers have a clear 3x3 neighborhood so a fixed ring can be stamped around them.

    Rule:
    - Find every 4-connected non-black component of size greater than one, treat it as a source, and record its total area plus its unique non-grey anchor color.
    - Let T be the median of those source areas; only source colors with area >= T are active.
    - Clear the whole grid, then for each isolated single-pixel marker whose color is active, stamp a 3x3 ring of grey cells around it while keeping the center marker color.
    - Remove every original source blob, every marker tied to an inactive source color, and every unrelated distractor marker.

    Key insight:
    - The grey blob shape itself is never copied: it only supplies an anchor color and an area for the hidden median gate, while the visible output is always the same fixed ring stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources: list[dict[str, int]] = []
    markers: list[tuple[int, int, int]] = []
    for comp in _foreground_components(g):
        if len(comp) == 1:
            r, c = comp[0]
            markers.append((r, c, int(g[r, c])))
            continue

        non_grey = [(r, c) for r, c in comp if int(g[r, c]) != GREY]
        assert len(non_grey) == 1, "each source component must contain exactly one non-grey anchor"
        ar, ac = non_grey[0]
        sources.append({"area": len(comp), "color": int(g[ar, ac])})

    assert sources, "expected at least one source component"
    threshold = float(np.median([src["area"] for src in sources]))
    active_colors = {src["color"] for src in sources if src["area"] >= threshold}

    out = np.zeros_like(g)
    for r, c, color in markers:
        if color in active_colors:
            _stamp_ring(out, r, c, color)
    return out


def _grow_shape(area: int) -> list[tuple[int, int]]:
    while True:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c)
                if nbr not in cells
            }
            cells.add(random.choice(tuple(frontier)))

        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        coords = sorted((r - r0, c - c0) for r, c in cells)
        h = max(r for r, _ in coords) + 1
        w = max(c for _, c in coords) + 1
        if max(h, w) > MAX_PATCH_SIDE:
            continue
        if area * 2 < h * w:
            continue
        return coords


def _make_source_patch(area: int, anchor_color: int) -> np.ndarray:
    coords = _grow_shape(area)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r, c] = GREY
    anchor = random.choice(coords)
    patch[anchor] = int(anchor_color)
    return patch


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]], pad: int) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                blocked[rr, cc] = True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    mask = patch != BG
    positions = [(r0, c0) for r0 in range(grid.shape[0] - h + 1) for c0 in range(grid.shape[1] - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        coords = [(r0 + r, c0 + c) for r, c in np.argwhere(mask)]
        _mark_blocked(blocked, coords, pad=1)
        return True
    return False


def _sorted_source_areas(n_sources: int) -> list[int]:
    while True:
        areas = sorted(random.sample(tuple(SOURCE_AREA_RANGE), n_sources))
        if n_sources == 4 and areas[1] == areas[2]:
            continue
        return areas


def _marker_specs(active_colors: list[int], inactive_colors: list[int], anchor_colors: list[int]) -> list[int]:
    specs: list[int] = []
    duplicate_color = random.choice(active_colors)
    for color in active_colors:
        specs.append(color)
        if color == duplicate_color:
            specs.append(color)
    for color in inactive_colors:
        specs.append(color)

    distractor_palette = [color for color in ANCHOR_COLORS if color not in anchor_colors]
    distractor_palette.extend(DISTRACTOR_COLORS)
    for _ in range(random.choice(DISTRACTOR_COUNT_OPTIONS)):
        specs.append(random.choice(distractor_palette))
    random.shuffle(specs)
    return specs


def _choose_marker_centers(shape: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)]
    random.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == count:
                return centers
    return None


def _build_output(shape: tuple[int, int], markers: list[tuple[int, int, int]], active_colors: set[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        if color in active_colors:
            _stamp_ring(out, r, c, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_sources = random.choice(SOURCE_COUNT_OPTIONS)
    anchor_colors = random.sample(ANCHOR_COLORS, n_sources)
    areas = _sorted_source_areas(n_sources)

    threshold = float(np.median(areas))
    active_colors = [color for color, area in zip(anchor_colors, areas) if area >= threshold]
    inactive_colors = [color for color, area in zip(anchor_colors, areas) if area < threshold]
    if not active_colors or not inactive_colors:
        return None

    marker_colors = _marker_specs(active_colors, inactive_colors, anchor_colors)
    centers = _choose_marker_centers((h, w), len(marker_colors))
    if centers is None:
        return None
    markers = [(r, c, color) for (r, c), color in zip(centers, marker_colors)]

    expected = _build_output((h, w), markers, set(active_colors))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for r, c, color in markers:
        grid[r, c] = int(color)
        _mark_blocked(blocked, [(r, c)], pad=1)

    sources = list(zip(areas, anchor_colors))
    random.shuffle(sources)
    for area, color in sources:
        patch = _make_source_patch(area, color)
        if not _place_patch(grid, blocked, patch):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
