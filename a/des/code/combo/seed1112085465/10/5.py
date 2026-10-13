import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ROTATION_BY_EDGES = {
    frozenset(): 0,
    frozenset({"top"}): 2,
    frozenset({"right"}): 1,
    frozenset({"bottom"}): 0,
    frozenset({"left"}): 3,
    frozenset({"top", "left"}): 1,
    frozenset({"top", "right"}): 2,
    frozenset({"bottom", "left"}): 0,
    frozenset({"bottom", "right"}): 3,
}
BRANCHES = ("none", "top", "right", "bottom", "left", "corner")
CORNER_PATTERNS = (
    frozenset({"top", "left"}),
    frozenset({"top", "right"}),
    frozenset({"bottom", "left"}),
    frozenset({"bottom", "right"}),
)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _edge_contacts_from_bbox(
    r0: int,
    c0: int,
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
) -> frozenset[str]:
    edges: set[str] = set()
    if r0 == 0:
        edges.add("top")
    if c0 == 0:
        edges.add("left")
    if r0 + h == grid_h:
        edges.add("bottom")
    if c0 + w == grid_w:
        edges.add("right")
    return frozenset(edges)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            coords: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
            patch[coords_arr[:, 0] - int(r0), coords_arr[:, 1] - int(c0)] = color

            edges: set[str] = set()
            if bool((coords_arr[:, 0] == 0).any()):
                edges.add("top")
            if bool((coords_arr[:, 0] == h - 1).any()):
                edges.add("bottom")
            if bool((coords_arr[:, 1] == 0).any()):
                edges.add("left")
            if bool((coords_arr[:, 1] == w - 1).any()):
                edges.add("right")

            objects.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "edges": frozenset(edges),
                }
            )

    return objects


def _rotate_by_edges(patch: np.ndarray, edges: frozenset[str]) -> np.ndarray:
    assert edges in ROTATION_BY_EDGES, f"unsupported edge-contact pattern: {sorted(edges)}"
    rotated = np.rot90(np.asarray(patch, dtype=int), int(ROTATION_BY_EDGES[edges]))
    return _crop_nonzero(rotated)


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray:
    assert patches
    out_h = max(int(patch.shape[0]) for patch in patches)
    out_w = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for patch in patches:
        hh, ww = map(int, patch.shape)
        out[out_h - hh : out_h, col : col + ww] = patch
        col += ww + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each object according to which grid edges it touches, then pack the
    rotated objects left-to-right with bottom alignment.
    """
    g = np.asarray(grid, dtype=int)
    objects = _find_components(g)
    assert objects

    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    rotated = [_rotate_by_edges(obj["patch"], obj["edges"]) for obj in ordered]
    return _pack_patches(rotated)


def _random_mask() -> np.ndarray:
    for _ in range(120):
        h = int(RNG.integers(3, 6))
        w = int(RNG.integers(3, 6))
        max_area = min(h * w - 1, 8)
        if max_area < 4:
            continue
        target_area = int(RNG.integers(4, max_area + 1))

        mask = np.zeros((h, w), dtype=bool)
        start_r = int(RNG.integers(h))
        start_c = int(RNG.integers(w))
        mask[start_r, start_c] = True

        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            filled = np.argwhere(mask)
            for rr, cc in filled:
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = list(frontier)[int(RNG.integers(len(frontier)))]
            mask[choice] = True

        if int(mask.sum()) != target_area:
            continue

        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue

        mask = np.rot90(mask, int(RNG.integers(4)))
        if bool(RNG.integers(2)):
            mask = np.fliplr(mask)
        mask = _crop_mask(mask)

        variants = {_mask_key(_crop_mask(np.rot90(mask, k))) for k in range(4)}
        if len(variants) != 4:
            continue
        return mask.astype(bool, copy=True)

    raise RuntimeError("failed to sample a sufficiently asymmetric mask")


def _branch_edges(branch: str) -> frozenset[str]:
    if branch == "none":
        return frozenset()
    if branch == "corner":
        return CORNER_PATTERNS[int(RNG.integers(len(CORNER_PATTERNS)))]
    return frozenset({branch})


def _placement_candidates(
    reserved: np.ndarray,
    patch: np.ndarray,
    edges: frozenset[str],
) -> list[tuple[int, int]]:
    grid_h, grid_w = reserved.shape
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            if _edge_contacts_from_bbox(r0, c0, h, w, grid_h, grid_w) != edges:
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue
            candidates.append((r0, c0))
    return candidates


def _place_objects(
    grid_h: int,
    grid_w: int,
    objects: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(
        range(len(objects)),
        key=lambda idx: (len(objects[idx]["edges"]), max(objects[idx]["patch"].shape), objects[idx]["patch"].size),
        reverse=True,
    )

    for idx in order:
        obj = objects[idx]
        patch = np.asarray(obj["patch"], dtype=int)
        edges = obj["edges"]
        candidates = _placement_candidates(reserved, patch, edges)
        if not candidates:
            return None

        chosen = candidates[int(RNG.integers(len(candidates)))]
        r0, c0 = map(int, chosen)
        h, w = map(int, patch.shape)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid_h, r0 + h + 1)
        cc1 = min(grid_w, c0 + w + 1)
        reserved[rr0:rr1, cc0:cc1] = True
        grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]

        placed.append(
            {
                "patch": patch.copy(),
                "edges": edges,
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
            }
        )

    return grid, placed


def _expected_from_metadata(placed: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(placed, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    patches = [_rotate_by_edges(obj["patch"], obj["edges"]) for obj in ordered]
    return _pack_patches(patches)


def _gen_attempt() -> np.ndarray | None:
    obj_count = int(RNG.integers(3, 6))
    branch_indices = RNG.choice(len(BRANCHES), size=obj_count, replace=False)
    branches = [BRANCHES[int(idx)] for idx in branch_indices]
    colors = RNG.choice(np.arange(1, 10, dtype=int), size=obj_count, replace=False)

    objects: list[dict[str, object]] = []
    total_output_width = 0
    max_output_height = 0
    for branch, color in zip(branches, colors):
        mask = _random_mask()
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        edges = _branch_edges(branch)
        rotated = _rotate_by_edges(patch, edges)
        objects.append({"patch": patch, "edges": edges})
        total_output_width += int(rotated.shape[1])
        max_output_height = max(max_output_height, int(rotated.shape[0]))

    total_output_width += obj_count - 1
    if total_output_width > 30 or max_output_height > 30:
        return None

    min_h = max(14, max(int(obj["patch"].shape[0]) for obj in objects) + 6)
    min_w = max(14, max(int(obj["patch"].shape[1]) for obj in objects) + 6)
    if min_h > 24 or min_w > 24:
        return None

    grid_h = int(RNG.integers(min_h, 25))
    grid_w = int(RNG.integers(min_w, 25))
    placed_result = _place_objects(grid_h, grid_w, objects)
    if placed_result is None:
        return None
    grid, placed = placed_result

    if int(np.count_nonzero(grid)) == 0:
        return None
    if int((grid == BG).sum()) <= int((grid != BG).sum()):
        return None

    components = _find_components(grid)
    if len(components) != obj_count:
        return None
    for comp in components:
        if comp["edges"] not in ROTATION_BY_EDGES:
            return None

    expected = _expected_from_metadata(placed)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
