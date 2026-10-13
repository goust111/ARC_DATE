import random
from collections import deque

import numpy as np


BG = 0

# Generator configuration.
GEN_RETRIES = 500
OBJECT_COUNT_RANGE = (3, 4)
MAJOR_LEN_RANGE = (5, 12)
MINOR_LEN_RANGE = (2, 5)
CANVAS_MIN_SIDE = 14
CANVAS_MAX_SIDE = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(pts, dtype=int)})
    return comps


def _component_spec(color: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    horizontal = mask.shape[1] > mask.shape[0]
    active = mask.any(axis=0 if horizontal else 1)
    band = np.where(active, int(color), BG).astype(int)
    return {
        "color": int(color),
        "coords": coords,
        "area": int(coords.shape[0]),
        "band": band,
        "length": int(band.size),
        "anchor": (int(r0), int(c0)),
        "horizontal": bool(horizontal),
    }


def _grid_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    return [_component_spec(comp["color"], comp["coords"]) for comp in _find_components(g)]


def _stack_specs(specs: list[dict], order_key: str = "area") -> np.ndarray:
    assert specs, "expected at least one component"
    if order_key == "area":
        ordered = sorted(specs, key=lambda spec: (-spec["area"], spec["anchor"][0], spec["anchor"][1]))
    elif order_key == "length":
        ordered = sorted(specs, key=lambda spec: (-spec["length"], spec["anchor"][0], spec["anchor"][1]))
    else:
        raise ValueError(order_key)

    out_h = 2 * len(ordered) - 1
    out_w = max(spec["length"] for spec in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    for idx, spec in enumerate(ordered):
        out[2 * idx, : spec["length"]] = spec["band"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each elongated monochrome region into a one-pixel projection band and stack the bands by decreasing area.

    Input structure:
    - The background is black.
    - There are 3-4 disconnected monochrome 4-connected regions, each in a different color.
    - Every region is visibly stretched more along one bbox axis than the other, so its width and height are not equal.

    Rule:
    - For each colored region, compare its bounding-box width and height to decide the main ridge direction.
    - If width > height, project the region column by column; otherwise project it row by row. A projected position is colored iff that row or column contains at least one pixel of the region.
    - Treat the resulting one-pixel-wide projection as a horizontal color band, sort all bands by region area from largest to smallest, and stack them top to bottom with one black row between neighboring bands.
    - Left-align all bands in the output grid and fill any unused cells with black.

    Key insight:
    - The shape first has to be summarized along its longer bbox axis, and only then can the resulting 1D signatures be ordered by the original 2D areas.
    """
    specs = _grid_specs(np.asarray(grid, dtype=int))
    return _stack_specs(specs, order_key="area")


def _random_interval(height: int, prev_top: int | None = None, prev_bottom: int | None = None) -> tuple[int, int]:
    thickness = random.randint(1, height)
    tops = []
    for top in range(height - thickness + 1):
        bottom = top + thickness - 1
        if prev_top is not None and prev_bottom is not None and (top > prev_bottom or bottom < prev_top):
            continue
        tops.append(top)
    assert tops
    if prev_top is not None:
        near = [top for top in tops if abs(top - prev_top) <= 1]
        if near:
            tops = near
    top = random.choice(tops)
    return int(top), int(top + thickness - 1)


def _horizontal_mask(height: int, width: int) -> np.ndarray:
    for _ in range(100):
        intervals = []
        top, bottom = _random_interval(height)
        intervals.append((top, bottom))
        for _col in range(1, width):
            prev_top, prev_bottom = intervals[-1]
            intervals.append(_random_interval(height, prev_top, prev_bottom))

        if len(set(intervals)) == 1:
            continue
        if max(bottom - top + 1 for top, bottom in intervals) == 1:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for c, (top, bottom) in enumerate(intervals):
            mask[top : bottom + 1, c] = True
        return _crop_mask(mask)
    raise RuntimeError("failed to generate a non-rectangular horizontal ribbon")


def _make_patch(color: int, orientation: str) -> np.ndarray:
    for _ in range(120):
        major = random.randint(*MAJOR_LEN_RANGE)
        minor_hi = min(MINOR_LEN_RANGE[1], major - 1)
        minor = random.randint(MINOR_LEN_RANGE[0], minor_hi)
        base = _horizontal_mask(minor, major)
        mask = base if orientation == "h" else base.T
        mask = _crop_mask(mask)
        h, w = mask.shape
        if orientation == "h" and not (w > h):
            continue
        if orientation == "v" and not (h > w):
            continue
        patch = np.where(mask, int(color), BG).astype(int)
        return patch
    raise RuntimeError("failed to generate a valid patch")


def _patch_spec(patch: np.ndarray, color: int) -> dict:
    coords = np.argwhere(patch == int(color))
    return _component_spec(int(color), coords)


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    total_area = sum(int(np.count_nonzero(patch)) for patch in patches)
    side_hint = int(np.ceil(np.sqrt(4 * total_area)))
    min_h = max(CANVAS_MIN_SIDE, max_h + 2, side_hint)
    min_w = max(CANVAS_MIN_SIDE, max_w + 2, side_hint)
    if min_h > CANVAS_MAX_SIDE or min_w > CANVAS_MAX_SIDE:
        return None

    h = random.randint(min_h, CANVAS_MAX_SIDE)
    w = random.randint(min_w, CANVAS_MAX_SIDE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(patches)), key=lambda idx: -int(np.count_nonzero(patches[idx])))
    for idx in order:
        patch = patches[idx]
        mask = patch != BG
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)

        placed = False
        for top, left in positions:
            if bool((blocked[top : top + ph, left : left + pw] & mask).any()):
                continue
            grid[top : top + ph, left : left + pw][mask] = patch[mask]
            for rr, cc in np.argwhere(mask):
                ar = int(top + rr)
                ac = int(left + cc)
                for nr in range(ar - 1, ar + 2):
                    for nc in range(ac - 1, ac + 2):
                        if 0 <= nr < h and 0 <= nc < w:
                            blocked[nr, nc] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _generated_grid_is_valid(grid: np.ndarray, n_objects: int) -> bool:
    specs = _grid_specs(grid)
    if len(specs) != n_objects:
        return False
    if len({spec["color"] for spec in specs}) != n_objects:
        return False
    if len({spec["area"] for spec in specs}) != n_objects:
        return False
    if not any(spec["horizontal"] for spec in specs):
        return False
    if not any(not spec["horizontal"] for spec in specs):
        return False

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(counts[BG]) <= max(int(counts[c]) for c in range(1, 10)):
        return False

    expected = _stack_specs(specs, order_key="area")
    if np.array_equal(expected, _stack_specs(specs, order_key="length")):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    n_horizontal = random.randint(1, n_objects - 1)
    orientations = ["h"] * n_horizontal + ["v"] * (n_objects - n_horizontal)
    random.shuffle(orientations)
    colors = random.sample(list(range(1, 10)), n_objects)

    patches = []
    patch_specs = []
    for color, orientation in zip(colors, orientations):
        patch = _make_patch(int(color), orientation)
        patches.append(patch)
        patch_specs.append(_patch_spec(patch, int(color)))

    if len({spec["area"] for spec in patch_specs}) != n_objects:
        return None
    area_sorted = sorted(patch_specs, key=lambda spec: -spec["area"])
    length_sorted = sorted(patch_specs, key=lambda spec: -spec["length"])
    if [spec["color"] for spec in area_sorted] == [spec["color"] for spec in length_sorted]:
        return None

    grid = _place_patches(patches)
    if grid is None:
        return None
    if not _generated_grid_is_valid(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
