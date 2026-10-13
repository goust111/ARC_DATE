import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
FILL_COLORS = tuple(c for c in range(10) if c not in {BG, GREY})
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                touches_border |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            comps.append({"coords": arr, "bbox": bbox, "touches_border": touches_border})
    return comps


def _count_sides(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(np.uint8), 1)
    windows = (
        padded[:-1, :-1]
        + padded[1:, :-1]
        + padded[:-1, 1:]
        + padded[1:, 1:]
    )
    return int(np.count_nonzero((windows == 1) | (windows == 3)))


def _read_legend(grid: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    legend_mask = (grid != bg) & (grid != grey)
    mapping: dict[int, int] = {}
    for comp in _components(legend_mask):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "legend rows must be monochrome"
        assert r0 == r1, "legend entries must be horizontal runs"
        length = int(coords.shape[0])
        assert c1 - c0 + 1 == length, "legend runs must be contiguous"
        assert 3 <= length <= 6, "legend run lengths encode side counts"
        mapping[length] = next(iter(colors))
    assert mapping, "legend must contain at least one row"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey-walled region with the color whose legend bar length matches the region's side count.

    Input structure:
    - The background is the dominant black color.
    - Grey pixels form horizontal and vertical walls that create enclosed rectangular or L-shaped background regions.
    - The bottom-right corner contains a legend area separated from the main drawing by grey lines; each legend row is a horizontal monochrome bar of length 3, 4, 5, or 6.

    Rule:
    - Read each legend row as a mapping from its bar length N to that row's color.
    - Find every background component that is fully enclosed by grey walls, ignoring background that still reaches the grid boundary.
    - Count the enclosed region's sides by counting 90-degree turns along its orthogonal outline.
    - Fill the region's background pixels with the color mapped from that side count, leaving grey walls and the legend unchanged.

    Key insight:
    - The legend does not name colors directly; the visible length of each colored bar encodes the geometric side count that must be computed from each enclosed region's outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    assert non_bg, "expected at least one foreground color"
    grey = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))
    mapping = _read_legend(g, bg, grey)

    out = g.copy()
    for comp in _components(g == bg):
        if comp["touches_border"]:
            continue
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        sides = _count_sides(local)
        fill = mapping.get(sides)
        if fill is not None:
            out[coords[:, 0], coords[:, 1]] = fill
    return out


def _rectangle_region(rng: random.Random) -> tuple[np.ndarray, int]:
    h = rng.randint(2, 6)
    w = rng.randint(2, 6)
    return np.ones((h, w), dtype=bool), 4


def _l_region(rng: random.Random) -> tuple[np.ndarray, int]:
    for _ in range(50):
        h = rng.randint(3, 7)
        w = rng.randint(3, 7)
        cut_h = rng.randint(1, h - 1)
        cut_w = rng.randint(1, w - 1)
        region = np.ones((h, w), dtype=bool)
        corner = rng.choice(("tl", "tr", "bl", "br"))
        rows = slice(0, cut_h) if "t" in corner else slice(h - cut_h, h)
        cols = slice(0, cut_w) if "l" in corner else slice(w - cut_w, w)
        region[rows, cols] = False
        if int(region.sum()) >= 3 and _count_sides(region) == 6:
            return region, 6
    raise RuntimeError("failed to build an L-shaped region")


def _region_to_patch(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    padded_region = np.zeros((h + 2, w + 2), dtype=bool)
    padded_region[1 : h + 1, 1 : w + 1] = region

    border = np.zeros_like(padded_region)
    for r, c in np.argwhere(region):
        border[r : r + 3, c : c + 3] = True
    border &= ~padded_region

    patch = np.full(border.shape, BG, dtype=int)
    patch[border] = GREY
    return patch


def _open_rectangle_patch(rng: random.Random) -> np.ndarray:
    region, _ = _rectangle_region(rng)
    patch = _region_to_patch(region)
    for _ in range(20):
        side = rng.choice(("top", "bottom", "left", "right"))
        trial = patch.copy()
        if side in {"top", "bottom"} and trial.shape[1] >= 3:
            row = 0 if side == "top" else trial.shape[0] - 1
            start = rng.randint(1, trial.shape[1] - 2)
            gap_len = rng.randint(1, max(1, trial.shape[1] - 1 - start))
            trial[row, start : start + gap_len] = BG
        elif side in {"left", "right"} and trial.shape[0] >= 3:
            col = 0 if side == "left" else trial.shape[1] - 1
            start = rng.randint(1, trial.shape[0] - 2)
            gap_len = rng.randint(1, max(1, trial.shape[0] - 1 - start))
            trial[start : start + gap_len, col] = BG
        else:
            continue
        return trial
    return patch


def _place_patch(
    canvas: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    limit_h, limit_w = occupied.shape
    ph, pw = patch.shape
    spots: list[tuple[int, int]] = []
    for r in range(limit_h - ph + 1):
        for c in range(limit_w - pw + 1):
            rr0 = max(0, r - 1)
            rr1 = min(limit_h, r + ph + 1)
            cc0 = max(0, c - 1)
            cc1 = min(limit_w, c + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            spots.append((r, c))
    if not spots:
        return None

    r0, c0 = rng.choice(spots)
    view = canvas[r0 : r0 + ph, c0 : c0 + pw]
    wall_mask = patch == GREY
    view[wall_mask] = GREY

    rr0 = max(0, r0 - 1)
    rr1 = min(limit_h, r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(limit_w, c0 + pw + 1)
    occupied[rr0:rr1, cc0:cc1] = True
    return r0, c0


def _legend_rows(rng: random.Random) -> dict[int, int]:
    lengths = [4, 6]
    optional = [n for n in (3, 5) if rng.random() < 0.6]
    lengths.extend(optional)
    lengths = sorted(set(lengths))
    colors = rng.sample(FILL_COLORS, len(lengths))
    mapping = dict(zip(lengths, colors))
    assert 4 in mapping and 6 in mapping
    return mapping


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mapping = _legend_rows(rng)
    legend_lengths = list(mapping)
    legend_h = 2 * len(legend_lengths) - 1
    legend_w = max(legend_lengths) + rng.randint(1, 3)

    max_main_h = 30 - 1 - legend_h
    max_main_w = 30 - 1 - legend_w
    if max_main_h < 10 or max_main_w < 10:
        return None

    main_h = rng.randint(10, max_main_h)
    main_w = rng.randint(10, max_main_w)
    h = main_h + 1 + legend_h
    w = main_w + 1 + legend_w
    grid = np.full((h, w), BG, dtype=int)

    grid[main_h, main_w:] = GREY
    grid[main_h:, main_w] = GREY

    occupied = np.zeros((main_h, main_w), dtype=bool)
    if main_h >= 2:
        occupied[main_h - 1, :] = True
    if main_w >= 2:
        occupied[:, main_w - 1] = True

    closed_specs: list[dict[str, object]] = []
    closed_count = rng.randint(2, 4)
    if closed_count >= 2 and rng.random() < 0.75:
        kinds = [4, 6] + [rng.choice((4, 6)) for _ in range(closed_count - 2)]
    else:
        kinds = [rng.choice((4, 6)) for _ in range(closed_count)]
    rng.shuffle(kinds)

    for kind in kinds:
        region, sides = _rectangle_region(rng) if kind == 4 else _l_region(rng)
        patch = _region_to_patch(region)
        pos = _place_patch(grid[:main_h, :main_w], occupied, patch, rng)
        if pos is None:
            return None
        pr, pc = pos
        rr, cc = np.nonzero(region)
        coords = np.column_stack((rr + pr + 1, cc + pc + 1)).astype(int)
        closed_specs.append({"coords": coords, "sides": sides})

    if rng.random() < 0.6:
        for _ in range(rng.randint(1, 2)):
            patch = _open_rectangle_patch(rng)
            if _place_patch(grid[:main_h, :main_w], occupied, patch, rng) is None:
                return None

    legend_items = list(mapping.items())
    rng.shuffle(legend_items)
    for i, (length, color) in enumerate(legend_items):
        row = main_h + 1 + 2 * i
        offset = rng.randint(0, legend_w - length)
        start = main_w + 1 + offset
        grid[row, start : start + length] = color

    expected = grid.copy()
    for spec in closed_specs:
        coords = spec["coords"]
        expected[coords[:, 0], coords[:, 1]] = mapping[int(spec["sides"])]

    grey_count = int(np.count_nonzero(grid == GREY))
    if any(int(np.count_nonzero(grid == color)) >= grey_count for color in mapping.values()):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
