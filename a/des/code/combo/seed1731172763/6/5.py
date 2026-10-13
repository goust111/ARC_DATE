import numpy as np


BG = 8
GREY = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]
RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 300

CANONICAL_HOLES = {
    "single": np.array([[1]], dtype=bool),
    "domino": np.array([[1, 1]], dtype=bool),
    "ell": np.array([[1, 1], [1, 0]], dtype=bool),
}

LEGEND_MASKS = {
    "single": np.array([[1, 0], [0, 0]], dtype=bool),
    "domino": np.array([[1, 1], [0, 0]], dtype=bool),
    "ell": np.array([[1, 1], [1, 0]], dtype=bool),
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return _coords_to_mask(coords)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    base = _normalize_mask(mask)
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for k in range(4):
            ori = _normalize_mask(np.rot90(cur, k))
            key = (ori.shape, ori.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(ori)
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for ori in _all_orientations(mask):
        keys.append((ori.shape, ori.tobytes()))
    return min(keys)


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(cols == 0)
        or np.any(rows == h - 1)
        or np.any(cols == w - 1)
    )


def _internal_holes(object_mask: np.ndarray) -> list[np.ndarray]:
    bg_comps = _components(~object_mask, connectivity=4)
    return [comp for comp in bg_comps if not _touches_boundary(comp, object_mask.shape)]


def _find_legend_entries(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "legend grey mask not found"
    legend_r0 = int(grey_coords[:, 0].min())
    assert legend_r0 + 1 < grid.shape[0], "legend must occupy two rows"

    band = grid[legend_r0 : legend_r0 + 2]
    occupied = np.any(band != bg, axis=0)

    runs: list[tuple[int, int]] = []
    c = 0
    while c < grid.shape[1]:
        if not occupied[c]:
            c += 1
            continue
        start = c
        while c + 1 < grid.shape[1] and occupied[c + 1]:
            c += 1
        runs.append((start, c))
        c += 1

    entries: list[dict] = []
    for c0, c1 in runs:
        assert c1 - c0 + 1 == 2, "each legend entry must be exactly 2 columns wide"
        block = grid[legend_r0 : legend_r0 + 2, c0 : c1 + 1]
        fill_colors = {
            int(v) for v in block.ravel() if int(v) not in (bg, GREY)
        }
        assert len(fill_colors) == 1, "each legend entry must contain exactly one fill color"
        mask = block == GREY
        assert mask.any(), "each legend entry must contain a grey mask"
        entries.append(
            {
                "sig": _shape_signature(mask),
                "fill": fill_colors.pop(),
                "row": legend_r0,
                "col": c0,
            }
        )

    assert entries, "legend must be non-empty"
    return legend_r0, entries


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    legend_r0, entries = _find_legend_entries(g, bg)
    by_shape: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for entry in entries:
        by_shape.setdefault(entry["sig"], []).append(entry)

    out = g.copy()
    work = g.copy()
    work[legend_r0 : legend_r0 + 2, :] = bg

    objects = _components(work != bg, connectivity=4)
    assert objects, "expected at least one foreground object"

    for comp in objects:
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = comp - np.array([r0, c0], dtype=int)
        patch[rel[:, 0], rel[:, 1]] = True

        holes = _internal_holes(patch)
        assert len(holes) == 1, "each object must contain exactly one enclosed missing region"
        hole = holes[0]
        sig = _shape_signature(_coords_to_mask(hole))
        candidates = by_shape.get(sig)
        assert candidates, "missing-region shape must have a legend match"
        entry = candidates.pop(0)

        fill = int(entry["fill"])
        hole_abs = hole + np.array([r0, c0], dtype=int)
        out[hole_abs[:, 0], hole_abs[:, 1]] = fill
        out[entry["row"] : entry["row"] + 2, entry["col"] : entry["col"] + 2] = bg

    return out


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    options = _all_orientations(mask)
    idx = int(RNG.integers(len(options)))
    return options[idx]


def _make_base_object(hole_mask: np.ndarray, color: int) -> np.ndarray:
    hh, hw = hole_mask.shape
    h = int(RNG.integers(max(5, hh + 3), max(6, hh + 7)))
    w = int(RNG.integers(max(5, hw + 3), max(6, hw + 7)))
    patch = np.full((h, w), color, dtype=int)

    hole_r = int(RNG.integers(1, h - hh))
    hole_c = int(RNG.integers(1, w - hw))
    patch[hole_r : hole_r + hh, hole_c : hole_c + hw][hole_mask] = BG
    return patch


def _maybe_add_arm(patch: np.ndarray, color: int) -> np.ndarray:
    if RNG.random() < 0.4:
        return patch

    h, w = patch.shape
    side = ["top", "bottom", "left", "right"][int(RNG.integers(4))]
    length = int(RNG.integers(1, 3))

    if side in ("top", "bottom"):
        span = int(RNG.integers(1, min(4, w) + 1))
        c0 = int(RNG.integers(0, w - span + 1))
        new = np.full((h + length, w), BG, dtype=int)
        if side == "top":
            new[length:, :] = patch
            new[:length, c0 : c0 + span] = color
        else:
            new[:h, :] = patch
            new[h:, c0 : c0 + span] = color
        return new

    span = int(RNG.integers(1, min(4, h) + 1))
    r0 = int(RNG.integers(0, h - span + 1))
    new = np.full((h, w + length), BG, dtype=int)
    if side == "left":
        new[:, length:] = patch
        new[r0 : r0 + span, :length] = color
    else:
        new[:, :w] = patch
        new[r0 : r0 + span, w:] = color
    return new


def _make_object_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    for _ in range(60):
        patch = _make_base_object(hole_mask, color)
        patch = _maybe_add_arm(patch, color)
        object_mask = patch != BG
        holes = _internal_holes(object_mask)
        if len(holes) == 1 and _shape_signature(_coords_to_mask(holes[0])) == _shape_signature(hole_mask):
            return patch
    raise RuntimeError("failed to build a valid object patch")


def _legend_block(shape_name: str, fill_color: int) -> np.ndarray:
    block = np.full((2, 2), BG, dtype=int)
    block[LEGEND_MASKS[shape_name]] = GREY
    block[1, 1] = fill_color
    return block


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    blocked: np.ndarray,
    min_row: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    if min_row > h - ph:
        return None
    fg = np.argwhere(patch != BG)
    if fg.size == 0:
        return None

    for _ in range(150):
        r0 = int(RNG.integers(min_row, h - ph + 1))
        c0 = int(RNG.integers(0, w - pw + 1))
        rows = r0 + fg[:, 0]
        cols = c0 + fg[:, 1]
        if np.any(blocked[rows, cols]):
            continue

        grid_slice = grid[r0 : r0 + ph, c0 : c0 + pw]
        grid_slice[patch != BG] = patch[patch != BG]

        for rr, cc in zip(rows.tolist(), cols.tolist()):
            blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True
        return r0, c0

    return None


def _build_expected(
    grid: np.ndarray,
    placements: list[dict],
    legend_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        rows = item["hole_abs"][:, 0]
        cols = item["hole_abs"][:, 1]
        out[rows, cols] = int(item["fill"])
    for r0, c0 in legend_positions:
        out[r0 : r0 + 2, c0 : c0 + 2] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = int(RNG.integers(1, 4))
    shape_names = list(CANONICAL_HOLES)
    chosen_shapes = list(RNG.choice(shape_names, size=object_count, replace=False))
    RNG.shuffle(chosen_shapes)

    colors = list(RNG.choice(COLOR_POOL, size=2 * object_count, replace=False))
    object_colors = colors[:object_count]
    fill_colors = colors[object_count:]

    object_specs = []
    for shape_name, object_color, fill_color in zip(chosen_shapes, object_colors, fill_colors):
        hole_mask = _random_orientation(CANONICAL_HOLES[shape_name])
        patch = _make_object_patch(hole_mask, int(object_color))
        object_specs.append(
            {
                "shape": shape_name,
                "patch": patch,
                "fill": int(fill_color),
            }
        )

    legend_w = 3 * object_count - 1
    max_patch_h = max(spec["patch"].shape[0] for spec in object_specs)
    max_patch_w = max(spec["patch"].shape[1] for spec in object_specs)
    min_h = max(12, max_patch_h + 7, 7 + sum(spec["patch"].shape[0] for spec in object_specs) // 2)
    min_w = max(legend_w + 2, max_patch_w + 2)
    h = int(RNG.integers(min_h, max(min_h + 1, 23)))
    w = int(RNG.integers(min_w, max(min_w + 1, 27)))

    grid = np.full((h, w), BG, dtype=int)

    legend_c0 = int(RNG.integers(0, w - legend_w + 1))
    legend_positions: list[tuple[int, int]] = []
    legend_order = list(range(object_count))
    RNG.shuffle(legend_order)
    for slot, spec_idx in enumerate(legend_order):
        c0 = legend_c0 + 3 * slot
        grid[0:2, c0 : c0 + 2] = _legend_block(object_specs[spec_idx]["shape"], object_specs[spec_idx]["fill"])
        legend_positions.append((0, c0))

    blocked = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []
    min_row = int(RNG.integers(4, 7))
    for spec in sorted(object_specs, key=lambda item: -np.count_nonzero(item["patch"] != BG)):
        pos = _place_patch(grid, spec["patch"], blocked, min_row=min_row)
        if pos is None:
            return None
        r0, c0 = pos
        holes = _internal_holes(spec["patch"] != BG)
        if len(holes) != 1:
            return None
        hole_abs = holes[0] + np.array([r0, c0], dtype=int)
        placements.append({"hole_abs": hole_abs, "fill": spec["fill"]})

    expected = _build_expected(grid, placements, legend_positions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
