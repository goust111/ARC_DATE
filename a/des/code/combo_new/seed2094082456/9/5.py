import random

import numpy as np


MARKER_COLOR = 2
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIZE_RANGE = (16, 30)
PAIR_COUNT_RANGE = (2, 4)
RAW_SHAPE_SIDE_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 7)
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _component_list(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox, mask = _coords_to_mask(arr)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each same-color object pair to locate the extra pixel and mark that column with red nearby.

    Input structure:
    - A dominant background surrounds several monochrome 4-connected objects.
    - For every non-red foreground color, exactly two objects of that color are present.
    - One isolated red pixel sits in a grid corner as a global marker, and no other red object appears.

    Rule:
    - Remove the corner red marker from consideration, then group the remaining objects by color.
    - Within each color pair, align the two object bounding boxes by their top-left corners; one object is the other plus exactly one extra pixel.
    - Keep the original objects unchanged, and place a red pixel one row above the larger object's extra pixel; if that cell is occupied or outside the grid, place the red pixel one row below instead.
    - Delete the original corner red marker in the output.

    Key insight:
    - The missing location is not readable from a single object: it is revealed only by pairing equal-color objects and comparing their aligned shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    assert bg != MARKER_COLOR, "red is reserved for the global/output marker"

    components = _component_list(g, bg)
    assert components, "expected foreground components"
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}

    red_components = [comp for comp in components if int(comp["color"]) == MARKER_COLOR]
    marker_components = [
        comp
        for comp in red_components
        if int(comp["area"]) == 1 and tuple(map(int, comp["coords"][0])) in corners
    ]
    assert len(marker_components) == 1, "expected exactly one isolated corner red marker"
    assert len(red_components) == 1, "red must be reserved for the singleton marker"
    marker = marker_components[0]
    marker_rc = tuple(map(int, marker["coords"][0]))

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        if comp is marker:
            continue
        by_color.setdefault(int(comp["color"]), []).append(comp)

    assert by_color, "expected at least one color pair"
    assert all(len(comps) == 2 for comps in by_color.values()), "every foreground color must appear exactly twice"

    out = g.copy()
    out[marker_rc] = bg

    for comps in by_color.values():
        small, large = sorted(comps, key=lambda comp: int(comp["area"]))
        assert int(large["area"]) == int(small["area"]) + 1, "pair areas must differ by exactly one"

        small_mask = np.asarray(small["mask"], dtype=bool)
        large_mask = np.asarray(large["mask"], dtype=bool)
        hs, ws = small_mask.shape
        hl, wl = large_mask.shape
        assert hs <= hl and ws <= wl, "aligned smaller mask must fit within the larger bbox"

        small_aligned = np.zeros((hl, wl), dtype=bool)
        small_aligned[:hs, :ws] = small_mask
        assert not np.any(small_aligned & ~large_mask), "smaller object must be contained in the larger one after alignment"

        diff = large_mask & ~small_aligned
        assert np.count_nonzero(diff) == 1, "expected exactly one extra pixel in the larger object"
        extra_r, extra_c = map(int, np.argwhere(diff)[0])

        large_r0, large_c0, _large_r1, _large_c1 = map(int, large["bbox"])
        abs_r = large_r0 + extra_r
        abs_c = large_c0 + extra_c

        up = (abs_r - 1, abs_c)
        if 0 <= up[0] < h and int(g[up]) == bg:
            out[up] = MARKER_COLOR
            continue

        down = (abs_r + 1, abs_c)
        assert 0 <= down[0] < h, "fallback marker position must stay in bounds"
        assert int(g[down]) == bg, "fallback marker position must remain background"
        out[down] = MARKER_COLOR

    return out


def _random_connected_mask() -> np.ndarray | None:
    raw_h = random.randint(*RAW_SHAPE_SIDE_RANGE)
    raw_w = random.randint(*RAW_SHAPE_SIDE_RANGE)
    max_area = min(raw_h * raw_w - 1, SHAPE_AREA_RANGE[1])
    min_area = min(SHAPE_AREA_RANGE[0], max_area)
    if max_area < SHAPE_AREA_RANGE[0]:
        return None
    target_area = random.randint(min_area, max_area)

    start = (random.randrange(raw_h), random.randrange(raw_w))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while frontier and len(cells) < target_area:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        add_frontier(nxt)

    if len(cells) != target_area:
        return None

    coords = np.array(sorted(cells), dtype=int)
    _bbox_unused, mask = _coords_to_mask(coords)
    if min(mask.shape) < 2:
        return None
    return mask


def _extra_candidates(mask: np.ndarray, branch: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    candidates: list[tuple[int, int]] = []

    for er in range(h + 1):
        for ec in range(w + 1):
            if er < h and ec < w and bool(mask[er, ec]):
                continue

            adjacent = False
            for dr, dc in NEIGHBORS4:
                nr, nc = er + dr, ec + dc
                if (nr, nc) in coords:
                    adjacent = True
                    break
            if not adjacent:
                continue

            above_occupied = er > 0 and ec < w and bool(mask[er - 1, ec])
            if branch == "up" and above_occupied:
                continue
            if branch == "down" and not above_occupied:
                continue

            below_occupied = er + 1 < h and ec < w and bool(mask[er + 1, ec])
            if branch == "down" and below_occupied:
                continue

            candidates.append((er, ec))

    return candidates


def _build_pair(branch: str) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        small = _random_connected_mask()
        if small is None:
            continue
        candidates = _extra_candidates(small, branch)
        if not candidates:
            continue
        extra_r, extra_c = random.choice(candidates)
        h, w = small.shape
        large = np.zeros((max(h, extra_r + 1), max(w, extra_c + 1)), dtype=bool)
        large[:h, :w] = small
        large[extra_r, extra_c] = True
        marker_rel = (extra_r - 1, extra_c) if branch == "up" else (extra_r + 1, extra_c)
        return {
            "branch": branch,
            "small": small,
            "large": large,
            "extra": (extra_r, extra_c),
            "marker_rel": marker_rel,
        }
    return None


def _dilate(coords: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _place_item(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    mask: np.ndarray,
    marker_rel: tuple[int, int] | None,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int] | None] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(0, h - mh)
        left = random.randint(0, w - mw)
        abs_cells = [(top + r, left + c) for r, c in coords]

        target = None
        relevant = list(abs_cells)
        if marker_rel is not None:
            tr = top + int(marker_rel[0])
            tc = left + int(marker_rel[1])
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            target = (tr, tc)
            if target in abs_cells:
                continue
            relevant.append(target)

        if any(bool(reserved[r, c]) for r, c in relevant):
            continue

        for r, c in abs_cells:
            grid[r, c] = int(color)
        for r, c in _dilate(relevant, h, w):
            reserved[r, c] = True
        return (top, left), abs_cells, target

    return None


def _branch_plan(pair_count: int) -> list[str]:
    up_count = pair_count // 2
    down_count = pair_count // 2
    if pair_count % 2:
        if random.choice([False, True]):
            up_count += 1
        else:
            down_count += 1
    branches = ["up"] * up_count + ["down"] * down_count
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    min_side = max(GRID_SIZE_RANGE[0], 10 + pair_count * 3)
    if min_side > GRID_SIZE_RANGE[1]:
        return None

    h = random.randint(min_side, GRID_SIZE_RANGE[1])
    w = random.randint(min_side, GRID_SIZE_RANGE[1])
    bg_choices = [color for color in range(10) if color != MARKER_COLOR]
    bg = random.choice(bg_choices)
    palette = [color for color in range(10) if color not in {bg, MARKER_COLOR}]
    if len(palette) < pair_count:
        return None
    colors = random.sample(palette, pair_count)

    branches = _branch_plan(pair_count)
    pair_specs = []
    for branch, color in zip(branches, colors):
        pair = _build_pair(branch)
        if pair is None:
            return None
        pair["color"] = int(color)
        pair_specs.append(pair)

    grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker_corner = random.choice(corners)
    grid[marker_corner] = MARKER_COLOR
    for r, c in corners:
        reserved[r, c] = True

    marker_targets: list[tuple[int, int]] = []
    items = []
    for pair in pair_specs:
        items.append((int(pair["color"]), np.asarray(pair["large"], dtype=bool), pair["marker_rel"]))
        items.append((int(pair["color"]), np.asarray(pair["small"], dtype=bool), None))
    random.shuffle(items)

    for color, mask, marker_rel in items:
        placed = _place_item(grid, reserved, color, mask, marker_rel)
        if placed is None:
            return None
        _anchor, _cells, target = placed
        if target is not None:
            marker_targets.append(target)

    expected = grid.copy()
    expected[marker_corner] = bg
    for r, c in marker_targets:
        expected[r, c] = MARKER_COLOR

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
