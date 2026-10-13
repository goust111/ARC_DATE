import random
import numpy as np


BG = 7
ARROW = 5
FG_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_RETRIES = 400
MAX_GRID_SIZE = 30

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                coords.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _detect_arrow_direction(coords: set[tuple[int, int]]) -> tuple[int, int]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 == 2 and c1 - c0 == 2
    assert len(coords) == 5
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    for name, pattern in ARROW_PATTERNS.items():
        if np.array_equal(mask, pattern):
            return DIRS[name]
    raise AssertionError("unrecognized arrow shape")


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows = []
    for idx, comp in enumerate(_find_components(grid == ARROW)):
        arrows.append(
            {
                "id": idx,
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "direction": _detect_arrow_direction(comp["coords"]),
            }
        )
    return arrows


def _pair_solution(
    comps: list[dict[str, object]],
    arrows: list[dict[str, object]],
    used_arrow_ids: set[int],
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    assert len(comps) == 2
    a, b = comps
    abox = a["bbox"]
    bbox = b["bbox"]
    assert isinstance(abox, tuple) and isinstance(bbox, tuple)

    horizontal = _interval_overlap(abox[0], abox[2], bbox[0], bbox[2]) and (
        abox[3] < bbox[1] or bbox[3] < abox[1]
    )
    vertical = _interval_overlap(abox[1], abox[3], bbox[1], bbox[3]) and (
        abox[2] < bbox[0] or bbox[2] < abox[0]
    )
    assert horizontal ^ vertical

    if horizontal:
        left, right = (a, b) if abox[1] < bbox[1] else (b, a)
        lbox = left["bbox"]
        rbox = right["bbox"]
        assert isinstance(lbox, tuple) and isinstance(rbox, tuple)
        shared_r0 = max(lbox[0], rbox[0])
        shared_r1 = min(lbox[2], rbox[2])
        gap = rbox[1] - lbox[3] - 1
        assert gap > 0
        candidates = []
        for arrow in arrows:
            if arrow["id"] in used_arrow_ids:
                continue
            arbox = arrow["bbox"]
            direction = arrow["direction"]
            assert isinstance(arbox, tuple) and isinstance(direction, tuple)
            if direction not in {(0, -1), (0, 1)}:
                continue
            if lbox[3] < arbox[1] and arbox[3] < rbox[1] and shared_r0 <= arbox[0] and arbox[2] <= shared_r1:
                candidates.append(arrow)
        assert len(candidates) == 1
        arrow = candidates[0]
        used_arrow_ids.add(int(arrow["id"]))
        if arrow["direction"] == (0, 1):
            fixed = left
            moving = right
            shift = (0, -gap)
        else:
            fixed = right
            moving = left
            shift = (0, gap)
    else:
        top, bottom = (a, b) if abox[0] < bbox[0] else (b, a)
        tbox = top["bbox"]
        bbox2 = bottom["bbox"]
        assert isinstance(tbox, tuple) and isinstance(bbox2, tuple)
        shared_c0 = max(tbox[1], bbox2[1])
        shared_c1 = min(tbox[3], bbox2[3])
        gap = bbox2[0] - tbox[2] - 1
        assert gap > 0
        candidates = []
        for arrow in arrows:
            if arrow["id"] in used_arrow_ids:
                continue
            arbox = arrow["bbox"]
            direction = arrow["direction"]
            assert isinstance(arbox, tuple) and isinstance(direction, tuple)
            if direction not in {(-1, 0), (1, 0)}:
                continue
            if tbox[2] < arbox[0] and arbox[2] < bbox2[0] and shared_c0 <= arbox[1] and arbox[3] <= shared_c1:
                candidates.append(arrow)
        assert len(candidates) == 1
        arrow = candidates[0]
        used_arrow_ids.add(int(arrow["id"]))
        if arrow["direction"] == (1, 0):
            fixed = top
            moving = bottom
            shift = (-gap, 0)
        else:
            fixed = bottom
            moving = top
            shift = (gap, 0)

    fixed_coords = set(fixed["coords"])
    moving_coords = {(r + shift[0], c + shift[1]) for r, c in moving["coords"]}
    assert fixed_coords.isdisjoint(moving_coords)
    merged = fixed_coords | moving_coords
    out_bbox = _bbox_from_coords(merged)
    r0, c0, r1, c1 = out_bbox
    assert len(merged) == (r1 - r0 + 1) * (c1 - c0 + 1)
    return merged, out_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-colored fragment pair into a filled rectangle.

    The grey arrow between the two fragments selects the side that gets pushed:
    the fragment on the pointed side moves opposite the arrow until the two
    fragment bounding boxes become adjacent. The merged result must be a solid
    rectangle, and all grey arrow cells are erased to orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fg_colors = [int(c) for c in np.unique(g) if int(c) not in (BG, ARROW)]
    arrows = _extract_arrows(g)
    assert len(arrows) == len(fg_colors)

    used_arrow_ids: set[int] = set()
    placements: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in fg_colors:
        comps = _find_components(g == color)
        assert len(comps) == 2
        _, bbox = _pair_solution(comps, arrows, used_arrow_ids)
        placements.append((color, bbox))
    assert len(used_arrow_ids) == len(arrows)

    out = g.copy()
    out[out == ARROW] = BG
    for color in fg_colors:
        out[g == color] = BG

    for color, bbox in placements:
        r0, c0, r1, c1 = bbox
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(region == BG)
        region[:] = color
    return out


def _build_case(direction: str, color: int, rng: random.Random) -> np.ndarray:
    if direction in ("left", "right"):
        rect_h = rng.randint(3, 6)
        rect_w = rng.randint(4, 8)
        split = rng.randint(1, rect_w - 1)
        gap = rng.randint(5, 8)
        patch = np.full((rect_h, rect_w + gap), BG, dtype=int)
        patch[:, :split] = color
        patch[:, split + gap :] = color
        arrow_top = rng.randint(0, rect_h - 3)
        arrow_left = split + rng.randint(1, gap - 4)
    else:
        rect_h = rng.randint(4, 8)
        rect_w = rng.randint(3, 6)
        split = rng.randint(1, rect_h - 1)
        gap = rng.randint(5, 8)
        patch = np.full((rect_h + gap, rect_w), BG, dtype=int)
        patch[:split, :] = color
        patch[split + gap :, :] = color
        arrow_top = split + rng.randint(1, gap - 4)
        arrow_left = rng.randint(0, rect_w - 3)

    arrow_mask = ARROW_PATTERNS[direction]
    arrow_region = patch[arrow_top : arrow_top + 3, arrow_left : arrow_left + 3]
    arrow_region[arrow_mask] = ARROW
    return patch


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _assemble_grid(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    shuffled = patches[:]
    rng.shuffle(shuffled)
    min_rows = 1 if len(shuffled) <= 2 else 2
    n_rows = rng.randint(min_rows, min(3, len(shuffled)))
    row_sizes = _random_partition(len(shuffled), n_rows, rng)
    h_gap = rng.randint(2, 4)
    v_gap = rng.randint(2, 4)

    placements: list[tuple[np.ndarray, int, int]] = []
    y = 0
    arrangement_w = 0
    idx = 0
    for row_size in row_sizes:
        row_patches = shuffled[idx : idx + row_size]
        idx += row_size
        row_h = max(p.shape[0] for p in row_patches)
        x = 0
        for patch in row_patches:
            jitter = rng.randint(0, row_h - patch.shape[0])
            placements.append((patch, y + jitter, x))
            x += patch.shape[1] + h_gap
        arrangement_w = max(arrangement_w, x - h_gap)
        y += row_h + v_gap
    arrangement_h = y - v_gap

    if arrangement_h > MAX_GRID_SIZE - 2 or arrangement_w > MAX_GRID_SIZE - 2:
        return None

    extra_h = rng.randint(2, min(8, MAX_GRID_SIZE - arrangement_h))
    extra_w = rng.randint(2, min(8, MAX_GRID_SIZE - arrangement_w))
    grid_h = max(12, arrangement_h + extra_h)
    grid_w = max(12, arrangement_w + extra_w)
    if grid_h > MAX_GRID_SIZE or grid_w > MAX_GRID_SIZE:
        return None

    top0 = rng.randint(0, grid_h - arrangement_h)
    left0 = rng.randint(0, grid_w - arrangement_w)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, pr, pc in placements:
        target = grid[top0 + pr : top0 + pr + patch.shape[0], left0 + pc : left0 + pc + patch.shape[1]]
        assert target.shape == patch.shape
        mask = patch != BG
        assert not np.any(mask & (target != BG))
        target[mask] = patch[mask]
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    fg_colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, ARROW)]
    assert 1 <= len(fg_colors) <= 4
    assert len(_extract_arrows(grid)) == len(fg_colors)
    for color in fg_colors:
        assert len(_find_components(grid == color)) == 2

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == ARROW)
    for color in fg_colors:
        comps = _find_components(out == color)
        assert len(comps) == 1
        bbox = comps[0]["bbox"]
        coords = comps[0]["coords"]
        assert isinstance(bbox, tuple) and isinstance(coords, set)
        r0, c0, r1, c1 = bbox
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choice([1, 2, 2, 3, 3, 4])
    colors = rng.sample(FG_COLORS, n_pairs)
    directions = [rng.choice(list(DIRS.keys())) for _ in range(n_pairs)]
    patches = [_build_case(direction, color, rng) for direction, color in zip(directions, colors)]
    grid = _assemble_grid(patches, rng)
    if grid is None:
        return None
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
