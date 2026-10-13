import random
from collections import defaultdict, deque

import numpy as np


DIVIDER_COLOR = 9
GEN_COLORS = tuple(color for color in range(1, 10) if color != DIVIDER_COLOR)
GEN_RETRIES = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _trim_coords(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    trimmed = {(r - r0, c - c0) for r, c in coords}
    return trimmed, (r1 - r0 + 1, c1 - c0 + 1)


def _rotate180(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    return {(h - 1 - r, w - 1 - c) for r, c in coords}


def _extract_shape(region: np.ndarray, bg: int) -> dict:
    mask = region != bg
    comps = _components(mask)
    assert len(comps) == 1, "each quadrant must contain exactly one foreground component"
    comp = set(comps[0])
    colors = {int(region[r, c]) for r, c in comp}
    assert len(colors) == 1, "each quadrant shape must be monochrome"
    color = next(iter(colors))
    h, w = region.shape
    for r, c in comp:
        assert 0 < r < h - 1 and 0 < c < w - 1, "shapes must not touch quadrant borders"
    trimmed, shape = _trim_coords(comp)
    return {"color": int(color), "coords": trimmed, "shape": shape, "area": len(trimmed)}


def _completion_dims(shape_a: dict, shape_b: dict) -> tuple[int, int]:
    coords_a = shape_a["coords"]
    coords_b = _rotate180(shape_b["coords"], shape_b["shape"])
    ha, wa = shape_a["shape"]
    hb, wb = shape_b["shape"]
    total_area = len(coords_a) + len(coords_b)
    candidate_dims: set[tuple[int, int]] = set()

    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted_b = {(r + dr, c + dc) for r, c in coords_b}
            if coords_a & shifted_b:
                continue
            union = coords_a | shifted_b
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            full_h = max(rows) - min(rows) + 1
            full_w = max(cols) - min(cols) + 1
            if full_h * full_w == total_area:
                candidate_dims.add((full_h, full_w))

    assert len(candidate_dims) == 1, "same-color shapes must define a unique completed rectangle"
    return next(iter(candidate_dims))


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    row_candidates = [
        r for r in range(h) if int(grid[r, 0]) != bg and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    col_candidates = [
        c for c in range(w) if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    for r in row_candidates:
        if 2 * r != h - 1:
            continue
        for c in col_candidates:
            if 2 * c != w - 1:
                continue
            color = int(grid[r, 0])
            if color == int(grid[0, c]) and bool(np.all(grid[:, c] == color)):
                candidates.append((r, c, color))
    assert len(candidates) == 1, "expected one centered cross divider"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-color shape pair into a centered solid rectangle inside the four cross-divided quadrants.

    Input structure:
    - The background is black, and one monochrome horizontal line plus one monochrome vertical line form a centered cross divider.
    - The divider splits the canvas into four equal rectangular quadrants.
    - Each quadrant contains exactly one monochrome 4-connected shape, and no shape touches its quadrant boundary.
    - The four shapes form two same-color pairs; within each pair, one shape and the other shape rotated 180 degrees are complementary parts of a rectangle.

    Rule:
    - Detect the centered divider row and column, then isolate the four quadrants.
    - Pair quadrants by identical foreground color.
    - For each pair, rotate one shape by 180 degrees and align it against the other shape so their non-overlapping union becomes a full rectangle.
    - Replace both original shapes of that color with the completed solid rectangle, centered inside each of the pair's quadrants, while keeping the divider unchanged.

    Key insight:
    - Color identifies which quadrants belong together, but the actual completion requires geometric reasoning: the partner shape must be rotated 180 degrees before the two pieces tile a rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    div_r, div_c, divider_color = _find_dividers(g, bg)
    quad_h, quad_w = div_r, div_c

    quadrants = {
        "tl": g[:div_r, :div_c],
        "tr": g[:div_r, div_c + 1 :],
        "bl": g[div_r + 1 :, :div_c],
        "br": g[div_r + 1 :, div_c + 1 :],
    }
    offsets = {
        "tl": (0, 0),
        "tr": (0, div_c + 1),
        "bl": (div_r + 1, 0),
        "br": (div_r + 1, div_c + 1),
    }

    by_color: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for name, region in quadrants.items():
        assert region.shape == (quad_h, quad_w)
        shape = _extract_shape(region, bg)
        by_color[int(shape["color"])].append((name, shape))

    assert len(by_color) == 2, "expected exactly two shape colors"
    assert all(len(items) == 2 for items in by_color.values()), "each color must appear in exactly two quadrants"

    out = np.full_like(g, bg)
    out[div_r, :] = divider_color
    out[:, div_c] = divider_color

    for color, items in by_color.items():
        rect_h, rect_w = _completion_dims(items[0][1], items[1][1])
        assert rect_h <= quad_h and rect_w <= quad_w
        local_r = (quad_h - rect_h) // 2
        local_c = (quad_w - rect_w) // 2
        for name, _shape in items:
            off_r, off_c = offsets[name]
            out[off_r + local_r : off_r + local_r + rect_h, off_c + local_c : off_c + local_c + rect_w] = int(color)

    return out


def _shape_connected(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return len(_components(mask)) == 1


def _build_pair(full_h: int, full_w: int, rng: random.Random) -> tuple[dict, dict] | None:
    if full_h < 3 or full_w < 3:
        return None

    mode = rng.choice(["rows", "cols"])
    if mode == "rows":
        cuts = [rng.randint(1, full_w - 1) for _ in range(full_h)]
        if len(set(cuts)) == 1:
            idx = rng.randrange(full_h)
            cuts[idx] = 1 if cuts[idx] != 1 else full_w - 1
        a_full = {(r, c) for r in range(full_h) for c in range(cuts[r])}
    else:
        cuts = [rng.randint(1, full_h - 1) for _ in range(full_w)]
        if len(set(cuts)) == 1:
            idx = rng.randrange(full_w)
            cuts[idx] = 1 if cuts[idx] != 1 else full_h - 1
        a_full = {(r, c) for c in range(full_w) for r in range(cuts[c])}

    all_cells = {(r, c) for r in range(full_h) for c in range(full_w)}
    b_partner = all_cells - a_full
    if min(len(a_full), len(b_partner)) < 3:
        return None

    b_full = _rotate180(b_partner, (full_h, full_w))
    if rng.random() < 0.5:
        a_full = _rotate180(a_full, (full_h, full_w))
        b_full = _rotate180(b_full, (full_h, full_w))
    if rng.random() < 0.5:
        a_full, b_full = b_full, a_full

    a_coords, a_shape = _trim_coords(a_full)
    b_coords, b_shape = _trim_coords(b_full)
    if len(a_coords) == a_shape[0] * a_shape[1] or len(b_coords) == b_shape[0] * b_shape[1]:
        return None
    if not _shape_connected(a_coords, a_shape) or not _shape_connected(b_coords, b_shape):
        return None

    pair_a = {"coords": a_coords, "shape": a_shape, "area": len(a_coords)}
    pair_b = {"coords": b_coords, "shape": b_shape, "area": len(b_coords)}
    try:
        completion = _completion_dims(pair_a, pair_b)
    except AssertionError:
        return None
    if completion != (full_h, full_w):
        return None
    return pair_a, pair_b


def _place_shape(grid: np.ndarray, quadrant: str, shape: dict, color: int, div_r: int, div_c: int, rng: random.Random) -> None:
    quad_h, quad_w = div_r, div_c
    shape_h, shape_w = shape["shape"]
    max_r = quad_h - shape_h - 1
    max_c = quad_w - shape_w - 1
    assert max_r >= 1 and max_c >= 1
    local_r = rng.randint(1, max_r)
    local_c = rng.randint(1, max_c)
    base_r = 0 if quadrant in ("tl", "tr") else div_r + 1
    base_c = 0 if quadrant in ("tl", "bl") else div_c + 1
    for r, c in shape["coords"]:
        grid[base_r + local_r + r, base_c + local_c + c] = int(color)


def _expected_output(pair_specs: list[dict], div_r: int, div_c: int, colors: list[int], assignments: dict[str, int]) -> np.ndarray:
    out = np.zeros((2 * div_r + 1, 2 * div_c + 1), dtype=int)
    out[div_r, :] = DIVIDER_COLOR
    out[:, div_c] = DIVIDER_COLOR
    for quadrant, pair_idx in assignments.items():
        rect_h, rect_w = pair_specs[pair_idx]["full_shape"]
        color = colors[pair_idx]
        local_r = (div_r - rect_h) // 2
        local_c = (div_c - rect_w) // 2
        base_r = 0 if quadrant in ("tl", "tr") else div_r + 1
        base_c = 0 if quadrant in ("tl", "bl") else div_c + 1
        out[base_r + local_r : base_r + local_r + rect_h, base_c + local_c : base_c + local_c + rect_w] = int(color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    quad_h = rng.randint(7, 14)
    quad_w = rng.randint(7, 14)
    grid = np.zeros((2 * quad_h + 1, 2 * quad_w + 1), dtype=int)
    div_r, div_c = quad_h, quad_w
    grid[div_r, :] = DIVIDER_COLOR
    grid[:, div_c] = DIVIDER_COLOR

    colors = rng.sample(list(GEN_COLORS), 2)
    pair_specs: list[dict] = []
    for _ in range(2):
        full_h = rng.randint(3, quad_h - 2)
        full_w = rng.randint(3, quad_w - 2)
        pair = _build_pair(full_h, full_w, rng)
        if pair is None:
            return None
        pair_specs.append(
            {
                "full_shape": (full_h, full_w),
                "shapes": [pair[0], pair[1]],
            }
        )

    quadrants = ["tl", "tr", "bl", "br"]
    rng.shuffle(quadrants)
    assignments = {
        quadrants[0]: 0,
        quadrants[1]: 0,
        quadrants[2]: 1,
        quadrants[3]: 1,
    }

    pair_counters = [0, 0]
    for quadrant in ["tl", "tr", "bl", "br"]:
        pair_idx = assignments[quadrant]
        shape_idx = pair_counters[pair_idx]
        pair_counters[pair_idx] += 1
        shape = pair_specs[pair_idx]["shapes"][shape_idx]
        _place_shape(grid, quadrant, shape, colors[pair_idx], div_r, div_c, rng)

    expected = _expected_output(pair_specs, div_r, div_c, colors, assignments)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after many retries")
