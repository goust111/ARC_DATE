import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = int(g[rr, cc])
            objects.append({"bbox": (r0, c0, r1, c1), "coords": coords, "patch": patch})

    return objects


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    p = np.asarray(patch, dtype=int)
    return p.shape, tuple(int(x) for x in p.ravel())


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for rr in range(max(0, r0 - 1), min(h, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(w, c1 + 2)):
            if rr in (r0 - 1, r1 + 1) or cc in (c0 - 1, c1 + 1):
                grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the duplicated master pattern, compare each variant against it, and use
    the variant-only color to draw a one-cell border around that variant.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _find_objects(g, bg)
    assert len(objects) >= 4, "expected two master copies and at least two variants"

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = {}
    for i, obj in enumerate(objects):
        groups.setdefault(_patch_key(obj["patch"]), []).append(i)

    master_groups = [indices for indices in groups.values() if len(indices) == 2]
    assert len(master_groups) == 1, "expected exactly one duplicated master pattern"
    master_ids = sorted(master_groups[0], key=lambda idx: objects[idx]["bbox"])
    keep_id, clear_id = master_ids
    master_patch = objects[keep_id]["patch"]

    out = g.copy()
    r0, c0, r1, c1 = objects[clear_id]["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for i, obj in enumerate(objects):
        if i in master_ids:
            continue

        patch = obj["patch"]
        assert patch.shape == master_patch.shape, "all pattern boxes should match the master size"
        diff = patch != master_patch
        diff_count = int(np.count_nonzero(diff))
        assert 1 <= diff_count <= 2, "each variant should differ at one or two cells"

        signature_colors = {int(x) for x in patch[diff]}
        assert len(signature_colors) == 1, "all differing cells should share one signature color"
        signature = next(iter(signature_colors))
        assert int(np.count_nonzero(patch == signature)) == diff_count, "signature color must be unique to the differing cells"

        _draw_border(out, obj["bbox"], signature)

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 31
OBJ_H_MIN = 3
OBJ_H_MAX = 7
OBJ_W_MIN = 3
OBJ_W_MAX = 7
VARIANT_MIN = 2
VARIANT_MAX = 5
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
MASK_REMOVAL_ATTEMPTS = 300


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    count = 0

    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return count == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_support_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    min_cells = max(6, (h * w + 1) // 2)
    target = int(rng.integers(min_cells, h * w + 1))
    stale = 0

    while int(mask.sum()) > target and stale < MASK_REMOVAL_ATTEMPTS:
        cells = np.argwhere(mask)
        rr, cc = map(int, cells[int(rng.integers(len(cells)))])
        trial = mask.copy()
        trial[rr, cc] = False
        if not _touches_all_sides(trial) or not _mask_connected(trial):
            stale += 1
            continue
        mask = trial
        stale = 0

    return mask


def _colorize_master(
    rng: np.random.Generator,
    mask: np.ndarray,
    bg: int,
    variant_count: int,
) -> tuple[np.ndarray, list[int]]:
    available = [color for color in range(10) if color != bg]
    max_master_colors = min(4, len(available) - variant_count)
    min_master_colors = min(2, max_master_colors)
    n_master_colors = int(rng.integers(min_master_colors, max_master_colors + 1))
    master_colors = [int(x) for x in rng.choice(available, size=n_master_colors, replace=False)]

    patch = np.full(mask.shape, bg, dtype=int)
    coords = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    order = rng.permutation(len(coords))

    for idx, color in zip(order[:n_master_colors], master_colors):
        rr, cc = coords[int(idx)]
        patch[rr, cc] = color
    for idx in order[n_master_colors:]:
        rr, cc = coords[int(idx)]
        patch[rr, cc] = int(master_colors[int(rng.integers(len(master_colors)))])

    return patch, master_colors


def _make_variants(
    rng: np.random.Generator,
    master_patch: np.ndarray,
    bg: int,
    variant_count: int,
    master_colors: list[int],
) -> list[np.ndarray] | None:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(master_patch != bg)]
    signature_pool = [color for color in range(10) if color not in {bg, *master_colors}]
    if len(signature_pool) < variant_count:
        return None

    diff_counts = [1, 2] * (variant_count // 2)
    if variant_count % 2 == 1:
        diff_counts.append(int(rng.choice([1, 2])))
    rng.shuffle(diff_counts)

    used = {_patch_key(master_patch)}
    variants: list[np.ndarray] = []
    sig_colors = [int(x) for x in rng.choice(signature_pool, size=variant_count, replace=False)]

    for diff_count, signature in zip(diff_counts, sig_colors):
        built = None
        for _ in range(100):
            picked = rng.choice(len(coords), size=diff_count, replace=False)
            variant = master_patch.copy()
            for idx in np.atleast_1d(picked):
                rr, cc = coords[int(idx)]
                variant[rr, cc] = signature
            key = _patch_key(variant)
            if key in used:
                continue
            built = variant
            used.add(key)
            break
        if built is None:
            return None
        variants.append(built)

    return variants


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _place_patches(
    rng: np.random.Generator,
    patches: list[np.ndarray],
    bg: int,
) -> np.ndarray | None:
    cell_area = sum(int(np.count_nonzero(patch != bg)) for patch in patches)
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = int(rng.integers(max(GRID_MIN, max_h + 4), GRID_MAX + 1))
        grid_w = int(rng.integers(max(GRID_MIN, max_w + 4), GRID_MAX + 1))
        if grid_h * grid_w < cell_area * 2:
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        placements: list[tuple[int, int, int]] = []
        order = list(range(len(patches)))
        rng.shuffle(order)

        ok = True
        for patch_idx in order:
            patch = patches[patch_idx]
            ph, pw = patch.shape
            if grid_h < ph + 2 or grid_w < pw + 2:
                ok = False
                break

            placed = False
            for _ in range(200):
                top = int(rng.integers(1, grid_h - ph))
                left = int(rng.integers(1, grid_w - pw))
                zone = blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1]
                if zone.any():
                    continue
                blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True
                placements.append((patch_idx, top, left))
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        for patch_idx, top, left in placements:
            _paint_patch(grid, patches[patch_idx], top, left, bg)

        if _get_bg(grid) == bg:
            return grid

    return None


def _valid_scene(grid: np.ndarray, bg: int) -> bool:
    if _get_bg(grid) != bg:
        return False

    objects = _find_objects(grid, bg)
    if len(objects) < 4:
        return False

    shape = objects[0]["patch"].shape
    if any(obj["patch"].shape != shape for obj in objects):
        return False

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = {}
    for i, obj in enumerate(objects):
        groups.setdefault(_patch_key(obj["patch"]), []).append(i)
    master_groups = [indices for indices in groups.values() if len(indices) == 2]
    if len(master_groups) != 1:
        return False

    master_patch = objects[master_groups[0][0]]["patch"]
    master_ids = set(master_groups[0])
    h, w = grid.shape
    for i, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            return False
        frame = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
        ring = np.ones(frame.shape, dtype=bool)
        ring[1:-1, 1:-1] = False
        if np.any(frame[ring] != bg):
            return False
        if i in master_ids:
            continue
        diff = obj["patch"] != master_patch
        diff_count = int(np.count_nonzero(diff))
        if diff_count not in (1, 2):
            return False
        signature_colors = {int(x) for x in obj["patch"][diff]}
        if len(signature_colors) != 1:
            return False
        signature = next(iter(signature_colors))
        if int(np.count_nonzero(obj["patch"] == signature)) != diff_count:
            return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    variant_count = int(rng.integers(VARIANT_MIN, VARIANT_MAX + 1))
    obj_h = int(rng.integers(OBJ_H_MIN, OBJ_H_MAX + 1))
    obj_w = int(rng.integers(OBJ_W_MIN, OBJ_W_MAX + 1))
    bg = int(rng.integers(10))

    support = _random_support_mask(rng, obj_h, obj_w)
    master_patch, master_colors = _colorize_master(rng, support, bg, variant_count)
    variants = _make_variants(rng, master_patch, bg, variant_count, master_colors)
    if variants is None:
        return None

    patches = [master_patch, master_patch.copy(), *variants]
    grid = _place_patches(rng, patches, bg)
    if grid is None or not _valid_scene(grid, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
