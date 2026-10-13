import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
BUILDING_COLORS = (1, 2, 3, 5, 6, 7, 8)
GEN_RETRIES = 400
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _extract_buildings(grid: np.ndarray) -> list[dict]:
    buildings = []
    colors = sorted(set(map(int, grid.ravel())) - {BG, YELLOW, MAROON})
    for color in colors:
        comps = _components(grid == color)
        assert len(comps) == 1, "each building color must form one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        window_coords = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == BG:
                    window_coords.append((r, c))
        assert window_coords, "building must contain windows"
        buildings.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "windows": window_coords,
            }
        )
    return buildings


def _extract_decorations(grid: np.ndarray) -> list[dict]:
    maroon_mask = grid == MAROON
    yellow_comps = []
    for coords in _components(grid == YELLOW):
        r0, c0, r1, c1 = _bbox(coords)
        yellow_comps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "count": int(coords.shape[0]),
                "used": False,
            }
        )

    decorations = []
    for coords in _components(maroon_mask):
        r0, c0, r1, c1 = _bbox(coords)
        mask = maroon_mask[r0 : r1 + 1, c0 : c1 + 1]
        candidates = []
        for y in yellow_comps:
            yr0, yc0, yr1, yc1 = y["bbox"]
            if yr0 == yr1 == r1 + 1 and not (yc1 < c0 or yc0 > c1):
                candidates.append(y)
        assert len(candidates) == 1, "each decoration needs one adjacent yellow marker"
        marker = candidates[0]
        assert not marker["used"], "yellow markers must be uniquely assigned"
        marker["used"] = True
        decorations.append(
            {
                "count": marker["count"],
                "mask": mask.copy(),
            }
        )

    assert all(y["used"] for y in yellow_comps), "all yellow markers must belong to decorations"
    return decorations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair decorations to buildings by window count, then place them on roofs.

    Buildings are colored rectangles with background-pixel windows. Decorations
    are maroon connected shapes with a yellow row beneath them whose length
    equals the target building's window count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    buildings = _extract_buildings(g)
    decorations = _extract_decorations(g)

    building_by_count = {}
    for building in buildings:
        count = len(building["windows"])
        assert count not in building_by_count, "window counts must uniquely identify buildings"
        building_by_count[count] = building

    decoration_by_count = {}
    for decoration in decorations:
        count = decoration["count"]
        assert count not in decoration_by_count, "yellow counts must uniquely identify decorations"
        decoration_by_count[count] = decoration

    assert set(building_by_count) == set(decoration_by_count), "pair counts must match"

    out = g.copy()
    out[out == MAROON] = BG
    out[out == YELLOW] = BG

    for count, building in building_by_count.items():
        decoration = decoration_by_count[count]
        r0, c0, r1, c1 = building["bbox"]
        top_window_row = min(r for r, _ in building["windows"])
        top_window_cols = [c for r, c in building["windows"] if r == top_window_row]
        window_center = float(sum(top_window_cols)) / len(top_window_cols)
        building_center = (c0 + c1) / 2.0

        mask = decoration["mask"]
        if window_center > building_center:
            mask = mask[:, ::-1]
            start_c = c1 - mask.shape[1] + 1
        elif window_center < building_center:
            start_c = c0
        else:
            start_c = c0 + ((c1 - c0 + 1) - mask.shape[1]) // 2
        start_r = r0 - mask.shape[0]

        patch = out[start_r : start_r + mask.shape[0], start_c : start_c + mask.shape[1]]
        patch[mask] = MAROON

    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    row = int(RNG.integers(0, height))
    mask[row, 0] = True
    for col in range(1, width):
        next_row = int(np.clip(row + int(RNG.integers(-1, 2)), 0, height - 1))
        lo, hi = sorted((row, next_row))
        mask[lo : hi + 1, col] = True
        row = next_row
    extra = int(RNG.integers(0, max(1, height * width // 3) + 1))
    for _ in range(extra):
        filled = np.argwhere(mask)
        rr, cc = filled[int(RNG.integers(0, len(filled)))]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = neighbors[int(RNG.integers(0, len(neighbors)))]
            mask[nr, nc] = True
    return _trim_mask(mask)


def _make_building(branch: str, window_count: int) -> dict:
    if branch == "center":
        width = int(RNG.choice([w for w in (5, 7, 9) if w >= window_count]))
    else:
        width = int(RNG.integers(max(5, window_count), 9))
    height = int(RNG.integers(5, 9))

    center = (width - 1) / 2.0
    interior_cols = list(range(1, width - 1))
    if branch == "left":
        top_choices = [c for c in interior_cols if c < center]
    elif branch == "right":
        top_choices = [c for c in interior_cols if c > center]
    else:
        top_choices = [width // 2]
    assert top_choices
    top_col = int(RNG.choice(top_choices))

    available = [(r, c) for r in range(2, height - 1) for c in interior_cols]
    assert len(available) >= window_count - 1
    chosen = []
    if window_count > 1:
        picks = RNG.choice(len(available), size=window_count - 1, replace=False)
        chosen = [available[int(idx)] for idx in np.atleast_1d(picks)]
    windows = [(1, top_col)] + chosen
    return {"height": height, "width": width, "windows": windows, "branch": branch}


def _make_decoration(window_count: int, max_width: int) -> dict:
    width = int(RNG.integers(window_count, max_width + 1))
    height = int(RNG.integers(2, 5))
    mask = _random_connected_mask(height, width)
    if mask.shape[1] < window_count:
        pad = window_count - mask.shape[1]
        mask = np.pad(mask, ((0, 0), (0, pad)))
        mask[:, -1] = True
        mask = _trim_mask(mask)
    return {"mask": mask, "count": window_count}


def _derangement(n: int) -> np.ndarray:
    while True:
        perm = RNG.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def _place_building(grid: np.ndarray, top: int, left: int, color: int, spec: dict) -> None:
    h = spec["height"]
    w = spec["width"]
    grid[top : top + h, left : left + w] = color
    for rr, cc in spec["windows"]:
        grid[top + rr, left + cc] = BG


def _place_decoration(grid: np.ndarray, top: int, left: int, decoration: dict) -> None:
    mask = decoration["mask"]
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = MAROON
    marker_left = left + int(RNG.integers(0, w - decoration["count"] + 1))
    grid[top + h, marker_left : marker_left + decoration["count"]] = YELLOW


def _gen_attempt() -> np.ndarray | None:
    n_buildings = int(RNG.integers(3, 5))
    branches = ["left", "right", "center"]
    if n_buildings == 4:
        branches.append(str(RNG.choice(["left", "right", "center"])))
    RNG.shuffle(branches)

    counts = list(RNG.choice(np.arange(2, 6), size=n_buildings, replace=False))
    specs = []
    for branch, count in zip(branches, counts):
        building = _make_building(branch, int(count))
        decoration = _make_decoration(int(count), building["width"])
        color_choices = [c for c in BUILDING_COLORS if c not in [s["color"] for s in specs]]
        specs.append(
            {
                "count": int(count),
                "branch": branch,
                "building": building,
                "decoration": decoration,
                "color": int(RNG.choice(color_choices)),
            }
        )

    max_dec_total_h = max(spec["decoration"]["mask"].shape[0] + 1 for spec in specs)
    deco_band_h = max_dec_total_h + int(RNG.integers(2, 5))
    building_top_base = deco_band_h + int(RNG.integers(2, 5))

    perm = _derangement(n_buildings)
    slot_for_decoration = {int(perm[i]): i for i in range(n_buildings)}

    slot_widths = []
    for slot in range(n_buildings):
        building_width = specs[slot]["building"]["width"]
        decoration_width = specs[slot_for_decoration[slot]]["decoration"]["mask"].shape[1]
        slot_widths.append(max(building_width, decoration_width) + int(RNG.integers(2, 4)))

    gap = int(RNG.integers(2, 4))
    left_margin = int(RNG.integers(1, 4))
    right_margin = int(RNG.integers(1, 4))
    width = left_margin + right_margin + sum(slot_widths) + gap * (n_buildings - 1)

    building_tops = [building_top_base + int(RNG.integers(0, 3)) for _ in range(n_buildings)]
    height = max(
        building_tops[i] + specs[i]["building"]["height"] for i in range(n_buildings)
    ) + int(RNG.integers(1, 4))
    grid = np.zeros((height, width), dtype=int)

    slot_lefts = []
    cursor = left_margin
    for slot_w in slot_widths:
        slot_lefts.append(cursor)
        cursor += slot_w + gap

    for i, spec in enumerate(specs):
        bw = spec["building"]["width"]
        slot_w = slot_widths[i]
        b_left = slot_lefts[i] + int(RNG.integers(0, slot_w - bw + 1))
        _place_building(grid, building_tops[i], b_left, spec["color"], spec["building"])
        spec["building_left"] = b_left
        spec["building_top"] = building_tops[i]

    for spec_idx, spec in enumerate(specs):
        slot = int(perm[spec_idx])
        mask = spec["decoration"]["mask"]
        slot_w = slot_widths[slot]
        d_left = slot_lefts[slot] + int(RNG.integers(0, slot_w - mask.shape[1] + 1))
        max_top = deco_band_h - (mask.shape[0] + 1)
        if max_top < 1:
            return None
        d_top = int(RNG.integers(1, max_top + 1))
        region = grid[d_top : d_top + mask.shape[0] + 1, d_left : d_left + mask.shape[1]]
        if np.any(region != BG):
            return None
        _place_decoration(grid, d_top, d_left, spec["decoration"])

    # Validate that roof placements fit and remain disjoint.
    for spec in specs:
        b = spec["building"]
        dmask = spec["decoration"]["mask"]
        roof_top = spec["building_top"] - dmask.shape[0]
        if roof_top < 0:
            return None
        bw = b["width"]
        bl = spec["building_left"]
        if spec["branch"] == "left":
            roof_left = bl
        elif spec["branch"] == "right":
            roof_left = bl + bw - dmask.shape[1]
        else:
            roof_left = bl + (bw - dmask.shape[1]) // 2
        roof_right = roof_left + dmask.shape[1]
        if roof_left < 0 or roof_right > width:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    # Keep every branch visible in the output and input.
    if np.count_nonzero(grid == MAROON) == 0 or np.count_nonzero(out == MAROON) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
