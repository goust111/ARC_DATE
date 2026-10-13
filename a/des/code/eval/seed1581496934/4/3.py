import numpy as np

from collections import Counter, defaultdict, deque


AZURE = 8
GREY = 5
MAROON = 9
CONTAINER_COLORS = (1, 2, 3, 4, 6, 7)
DIRECTIONS = ("up", "down", "left", "right")
GRID_SIZES = (15, 17, 19, 21)
MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 200


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    assert w % 2 == 1, "grid must split into two equal panels plus a divider"
    divider = w // 2
    divider_color = int(grid[0, divider])
    assert divider_color != bg, "divider must be non-background"
    assert np.all(grid[:, divider] == divider_color), "divider must be a full-height solid line"
    return divider, divider_color


def _find_crosshair(grid: np.ndarray, divider: int, bg: int, divider_color: int) -> tuple[int, int, int]:
    h, w = grid.shape
    panel_w = w - divider - 1
    center_r = h // 2
    center_c = divider + 1 + panel_w // 2
    cross_color = int(grid[center_r, center_c])
    assert cross_color not in (bg, divider_color), "crosshair center must be visible"
    for d in range(-2, 3):
        assert int(grid[center_r + d, center_c]) == cross_color, "crosshair vertical arm must have length 5"
        assert int(grid[center_r, center_c + d]) == cross_color, "crosshair horizontal arm must have length 5"
    return center_r, center_c, cross_color


def _classify_border_component(coords: list[tuple[int, int]]) -> tuple[int, int, int, int, str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    pts = set(coords)
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
        return None

    top_missing = all((r0, c) not in pts for c in range(c0 + 1, c1))
    bottom_missing = all((r1, c) not in pts for c in range(c0 + 1, c1))
    left_missing = all((r, c0) not in pts for r in range(r0 + 1, r1))
    right_missing = all((r, c1) not in pts for r in range(r0 + 1, r1))

    top_present = all((r0, c) in pts for c in range(c0 + 1, c1))
    bottom_present = all((r1, c) in pts for c in range(c0 + 1, c1))
    left_present = all((r, c0) in pts for r in range(r0 + 1, r1))
    right_present = all((r, c1) in pts for r in range(r0 + 1, r1))

    candidates: list[str] = []
    if top_missing and bottom_present and left_present and right_present:
        candidates.append("up")
    if bottom_missing and top_present and left_present and right_present:
        candidates.append("down")
    if left_missing and top_present and bottom_present and right_present:
        candidates.append("left")
    if right_missing and top_present and bottom_present and left_present:
        candidates.append("right")
    if len(candidates) != 1:
        return None
    return r0, c0, r1, c1, candidates[0]


def _detect_containers(grid: np.ndarray, bg: int, divider: int) -> list[dict]:
    left = grid[:, :divider]
    left_fg = left != bg
    used = np.zeros_like(left, dtype=bool)
    containers: list[dict] = []

    for color in sorted(int(v) for v in np.unique(left) if int(v) != bg):
        for comp in _components(left == color):
            info = _classify_border_component(comp)
            if info is None:
                continue

            r0, c0, r1, c1, direction = info
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            pts = np.argwhere(interior == color)
            assert pts.size > 0, "each container must contain an interior pattern"

            pr0, pc0 = pts.min(axis=0)
            pr1, pc1 = pts.max(axis=0)
            pattern_coords = [(int(r - pr0), int(c - pc0)) for r, c in pts]

            containers.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "direction": direction,
                    "color": int(color),
                    "pattern_h": int(pr1 - pr0 + 1),
                    "pattern_w": int(pc1 - pc0 + 1),
                    "pattern_coords": pattern_coords,
                }
            )

            for r, c in comp:
                used[r, c] = True
            for r, c in pts:
                used[r0 + 1 + int(r), c0 + 1 + int(c)] = True

    assert 2 <= len(containers) <= 4, "left panel must contain 2 to 4 containers"
    assert np.array_equal(used, left_fg), "all left-panel foreground pixels must belong to containers"
    return containers


def _paint_overlay_cell(overlay: np.ndarray, r: int, c: int, color: int) -> None:
    current = int(overlay[r, c])
    if current == -1 or current == color:
        overlay[r, c] = color
    elif current != MAROON:
        overlay[r, c] = MAROON


def _render_output(grid: np.ndarray, containers: list[dict], divider: int, center_r: int, center_c: int) -> np.ndarray:
    h, w = grid.shape
    right_start = divider + 1
    overlay = np.full((h, w), -1, dtype=int)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in containers:
        grouped[item["direction"]].append(item)
    for items in grouped.values():
        items.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    offsets = {direction: 0 for direction in DIRECTIONS}
    for direction in ("up", "down", "left", "right"):
        for item in grouped[direction]:
            ph = int(item["pattern_h"])
            pw = int(item["pattern_w"])

            if direction == "up":
                base_r = center_r - offsets[direction] - ph
                base_c = center_c - pw // 2
                offsets[direction] += ph
            elif direction == "down":
                base_r = center_r + 1 + offsets[direction]
                base_c = center_c - pw // 2
                offsets[direction] += ph
            elif direction == "left":
                base_r = center_r - ph // 2
                base_c = center_c - offsets[direction] - pw
                offsets[direction] += pw
            else:
                base_r = center_r - ph // 2
                base_c = center_c + 1 + offsets[direction]
                offsets[direction] += pw

            for dr, dc in item["pattern_coords"]:
                rr = base_r + int(dr)
                cc = base_c + int(dc)
                assert 0 <= rr < h and right_start <= cc < w, "stamped pattern must stay inside the right panel"
                _paint_overlay_cell(overlay, rr, cc, int(item["color"]))

    out = grid.copy()
    mask = overlay != -1
    out[mask] = overlay[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect open-sided containers in the left panel and stamp their interior patterns
    outward from the right-panel crosshair. Patterns stack along shared directions,
    and overlaps of different colors turn maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    divider, divider_color = _find_divider(g, bg)
    center_r, center_c, _crosshair_color = _find_crosshair(g, divider, bg, divider_color)
    containers = _detect_containers(g, bg, divider)
    return _render_output(g, containers, divider, center_r, center_c)


def _sample_axis_lengths(rng: np.random.Generator, count: int, budget: int) -> list[int]:
    if count == 0:
        return []
    for _ in range(200):
        lengths = [int(rng.integers(1, 4)) for _ in range(count)]
        if sum(lengths) <= budget:
            return lengths
    raise AssertionError("failed to sample stacking lengths within the panel budget")


def _random_pattern_coords(
    rng: np.random.Generator,
    height: int,
    width: int,
    required: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    mask = rng.random((height, width)) < rng.uniform(0.28, 0.62)
    mask[0, int(rng.integers(width))] = True
    mask[height - 1, int(rng.integers(width))] = True
    mask[int(rng.integers(height)), 0] = True
    mask[int(rng.integers(height)), width - 1] = True
    if required is not None:
        mask[required] = True
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return coords


def _choose_directions(rng: np.random.Generator, count: int, want_overlap: bool) -> tuple[list[str], tuple[int, int] | None]:
    if not want_overlap:
        return [str(rng.choice(DIRECTIONS)) for _ in range(count)], None

    horizontal = str(rng.choice(("left", "right")))
    vertical = str(rng.choice(("up", "down")))
    directions = [horizontal, vertical]
    remaining = [d for d in DIRECTIONS if d not in (horizontal, vertical)]
    while len(directions) < count:
        directions.append(str(rng.choice(remaining)))
    return directions, (0, 1)


def _place_boxes(
    rng: np.random.Generator,
    sizes: list[tuple[int, int]],
    height: int,
    panel_w: int,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((height, panel_w), dtype=bool)
    placed: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for idx in order:
        box_h, box_w = sizes[idx]
        if height - box_h <= 1 or panel_w - box_w <= 1:
            return None

        success = False
        for _ in range(MAX_LAYOUT_ATTEMPTS):
            r0 = int(rng.integers(1, height - box_h))
            c0 = int(rng.integers(1, panel_w - box_w))
            r1 = r0 + box_h
            c1 = c0 + box_w
            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(height, r1 + 1)
            pad_c1 = min(panel_w, c1 + 1)
            if blocked[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue
            blocked[r0:r1, c0:c1] = True
            placed[idx] = (r0, c0)
            success = True
            break
        if not success:
            return None

    return [item for item in placed if item is not None]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    want_overlap = bool(rng.integers(0, 2))
    height = int(rng.choice(GRID_SIZES))
    panel_w = int(rng.choice(GRID_SIZES))
    width = panel_w * 2 + 1
    divider = panel_w
    center_r = height // 2
    center_c = divider + 1 + panel_w // 2

    count = int(rng.integers(2, 5))
    directions, overlap_pair = _choose_directions(rng, count, want_overlap)

    by_direction: dict[str, list[int]] = defaultdict(list)
    for idx, direction in enumerate(directions):
        by_direction[direction].append(idx)

    axis_lengths = [0] * count
    for direction, indices in by_direction.items():
        budget = panel_w // 2 if direction in ("left", "right") else height // 2
        lengths = _sample_axis_lengths(rng, len(indices), budget)
        for idx, length in zip(indices, lengths):
            axis_lengths[idx] = int(length)

    colors = [int(v) for v in rng.choice(CONTAINER_COLORS, size=count, replace=False)]
    specs: list[dict] = []
    sizes: list[tuple[int, int]] = []

    for idx, direction in enumerate(directions):
        is_horizontal = direction in ("left", "right")
        axis_len = axis_lengths[idx]
        orth_len = 1
        required = None

        if want_overlap and overlap_pair is not None and idx == overlap_pair[0]:
            orth_len = int(rng.integers(3, 5))
        elif want_overlap and overlap_pair is not None and idx == overlap_pair[1]:
            orth_len = int(rng.integers(3, 5))
        elif want_overlap:
            orth_len = int(rng.integers(1, 5))

        if is_horizontal:
            pattern_h = orth_len
            pattern_w = axis_len
        else:
            pattern_h = axis_len
            pattern_w = orth_len

        if want_overlap and overlap_pair is not None:
            h_idx, v_idx = overlap_pair
            if idx == h_idx:
                vertical_direction = directions[v_idx]
                target_row = pattern_h // 2 + (1 if vertical_direction == "down" else -1)
                target_col = 0 if direction == "right" else pattern_w - 1
                required = (int(target_row), int(target_col))
            elif idx == v_idx:
                horizontal_direction = directions[h_idx]
                target_row = 0 if direction == "down" else pattern_h - 1
                target_col = pattern_w // 2 + (1 if horizontal_direction == "right" else -1)
                required = (int(target_row), int(target_col))

        pattern_coords = _random_pattern_coords(rng, pattern_h, pattern_w, required=required)

        outer_h = pattern_h + int(rng.integers(4, 7))
        outer_w = pattern_w + int(rng.integers(4, 7))
        sizes.append((outer_h, outer_w))
        specs.append(
            {
                "direction": direction,
                "color": colors[idx],
                "pattern_h": int(pattern_h),
                "pattern_w": int(pattern_w),
                "pattern_coords": pattern_coords,
            }
        )

    starts = _place_boxes(rng, sizes, height, panel_w)
    if starts is None:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    grid[:, divider] = MAROON
    for d in range(-2, 3):
        grid[center_r + d, center_c] = GREY
        grid[center_r, center_c + d] = GREY

    for spec, (r0, c0), (outer_h, outer_w) in zip(specs, starts, sizes):
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        color = int(spec["color"])

        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        if spec["direction"] == "up":
            grid[r0, c0 + 1 : c1] = AZURE
        elif spec["direction"] == "down":
            grid[r1, c0 + 1 : c1] = AZURE
        elif spec["direction"] == "left":
            grid[r0 + 1 : r1, c0] = AZURE
        else:
            grid[r0 + 1 : r1, c1] = AZURE

        interior_h = outer_h - 2
        interior_w = outer_w - 2
        pad_top = int(rng.integers(1, interior_h - spec["pattern_h"]))
        pad_left = int(rng.integers(1, interior_w - spec["pattern_w"]))
        for pr, pc in spec["pattern_coords"]:
            grid[r0 + 1 + pad_top + pr, c0 + 1 + pad_left + pc] = color

        spec["bbox"] = (int(r0), int(c0), int(r1), int(c1))

    expected = _render_output(grid, specs, divider, center_r, center_c)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    overlap_on_right = bool(np.any(actual[:, divider + 1 :] == MAROON))
    if want_overlap != overlap_on_right:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
