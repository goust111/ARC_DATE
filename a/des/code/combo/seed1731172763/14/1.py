from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 8
RNG = np.random.default_rng()


PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 6
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2).copy()


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]))
    return objects


def _composition_key(grid: np.ndarray) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        int(grid.shape[0] * grid.shape[1]),
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(v) for v in grid.ravel()),
    )


def _compositions_one_fixed(
    base_mask: np.ndarray,
    move_mask: np.ndarray,
    base_label: int,
    move_label: int,
) -> list[np.ndarray]:
    base_coords = _mask_coords(base_mask)
    move_coords = _mask_coords(move_mask)
    base_set = set(base_coords)
    h1, w1 = map(int, base_mask.shape)
    h2, w2 = map(int, move_mask.shape)
    out: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}

    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            shifted = [(r + dr, c + dc) for r, c in move_coords]
            if any(pt in base_set for pt in shifted):
                continue
            union = base_coords + shifted
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if len(union) != hh * ww:
                continue

            grid = np.zeros((hh, ww), dtype=np.uint8)
            for r, c in base_coords:
                grid[r - r0, c - c0] = base_label
            for r, c in shifted:
                grid[r - r0, c - c0] = move_label
            if np.any(grid == 0):
                continue

            key = (int(hh), int(ww), tuple(int(v) for v in grid.ravel()))
            out[key] = grid

    return list(out.values())


def _compose_masks(mask_a: np.ndarray, mask_b: np.ndarray, allow_rotation: bool) -> list[np.ndarray]:
    candidates: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}
    b_variants = [mask_b]
    a_variants = [mask_a]
    if allow_rotation:
        b_rot = _rot180(mask_b)
        a_rot = _rot180(mask_a)
        if not np.array_equal(b_rot, mask_b):
            b_variants.append(b_rot)
        if not np.array_equal(a_rot, mask_a):
            a_variants.append(a_rot)

    for b_mask in b_variants:
        for grid in _compositions_one_fixed(mask_a, b_mask, 1, 2):
            candidates[(int(grid.shape[0]), int(grid.shape[1]), tuple(int(v) for v in grid.ravel()))] = grid
    for a_mask in a_variants:
        for grid in _compositions_one_fixed(mask_b, a_mask, 2, 1):
            candidates[(int(grid.shape[0]), int(grid.shape[1]), tuple(int(v) for v in grid.ravel()))] = grid

    return [candidates[key] for key in sorted(candidates)]


def _unique_matching(pair_exists: list[list[bool]]) -> list[tuple[int, int]] | None:
    n = len(pair_exists)
    solutions: list[list[tuple[int, int]]] = []

    def search(remaining: set[int], current: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(sorted((min(a, b), max(a, b)) for a, b in current))
            return

        i = min(
            remaining,
            key=lambda idx: sum(1 for j in remaining if j != idx and pair_exists[idx][j]),
        )
        options = [j for j in sorted(remaining) if j != i and pair_exists[i][j]]
        if not options:
            return

        for j in options:
            current.append((i, j))
            search(remaining - {i, j}, current)
            current.pop()

    search(set(range(n)), [])
    if len(solutions) != 1:
        return None
    return solutions[0]


def _solve_objects(objects: list[dict[str, object]]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    n = len(objects)
    assert n % 2 == 0 and n > 0

    pair_exists = [[False] * n for _ in range(n)]
    pair_grids: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            masks = _compose_masks(objects[i]["mask"], objects[j]["mask"], allow_rotation=True)
            if not masks:
                continue
            pair_exists[i][j] = True
            pair_exists[j][i] = True
            pair_grids[(i, j)] = min(masks, key=_composition_key)

    matching = _unique_matching(pair_exists)
    assert matching is not None, "pairing must be unique"
    return matching, pair_grids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered puzzle pieces by geometric complement, rebuild full rectangles,
    then pack the rectangles left-to-right by increasing area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    matching, pair_grids = _solve_objects(objects)

    paired_patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i, j in matching:
        comp = pair_grids[(i, j)]
        patch = np.full(comp.shape, bg, dtype=int)
        patch[comp == 1] = int(objects[i]["color"])
        patch[comp == 2] = int(objects[j]["color"])
        paired_patches.append((comp, patch))

    paired_patches.sort(key=lambda item: _composition_key(item[0]))
    patches = [patch for _, patch in paired_patches]

    out_h = max(int(p.shape[0]) for p in patches)
    out_w = sum(int(p.shape[1]) for p in patches) + len(patches) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for idx, patch in enumerate(patches):
        h, w = map(int, patch.shape)
        out[:h, c0 : c0 + w] = patch
        c0 += w
        if idx + 1 < len(patches):
            c0 += 1

    return out


def _area_preserving_dims(n_pairs: int) -> list[tuple[int, int]]:
    area_to_dims: dict[int, list[tuple[int, int]]] = {}
    for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
        for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
            area_to_dims.setdefault(h * w, []).append((h, w))

    areas = sorted(area_to_dims)
    chosen_areas = RNG.choice(areas, size=n_pairs, replace=False)
    dims = [area_to_dims[int(area)][int(RNG.integers(len(area_to_dims[int(area)])))] for area in chosen_areas]
    return dims


def _make_base_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        if bool(RNG.integers(2)):
            split = RNG.integers(1, w, size=h)
            if len(set(int(v) for v in split)) == 1:
                continue
            mask_a = np.zeros((h, w), dtype=bool)
            for r, width in enumerate(split):
                mask_a[int(r), : int(width)] = True
        else:
            split = RNG.integers(1, h, size=w)
            if len(set(int(v) for v in split)) == 1:
                continue
            mask_a = np.zeros((h, w), dtype=bool)
            for c, height in enumerate(split):
                mask_a[: int(height), int(c)] = True
        mask_b = ~mask_a
        if np.all(mask_a) or not np.any(mask_a):
            continue
        return mask_a, mask_b
    return None


def _orient_pair(mask_a: np.ndarray, mask_b: np.ndarray, needs_rotation: bool) -> tuple[np.ndarray, np.ndarray]:
    global_flip = bool(RNG.integers(2))
    extra_target = int(RNG.integers(2)) if needs_rotation else -1

    def orient(mask: np.ndarray, index: int) -> np.ndarray:
        do_flip = global_flip ^ (needs_rotation and extra_target == index)
        return _rot180(mask) if do_flip else mask.copy()

    return orient(mask_a, 0), orient(mask_b, 1)


def _place_masks(masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    sizes = [mask.shape for mask in masks]
    max_h = max(int(h) for h, _ in sizes)
    max_w = max(int(w) for _, w in sizes)
    h_low = max(GRID_SIDE_MIN, max_h + 4)
    w_low = max(GRID_SIDE_MIN, max_w + 4)
    h = int(RNG.integers(h_low, GRID_SIDE_MAX + 1))
    w = int(RNG.integers(w_low, GRID_SIDE_MAX + 1))

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    coords_list = [_mask_coords(mask) for mask in masks]
    order = list(int(v) for v in RNG.permutation(len(masks)))

    for idx in order:
        mask = masks[idx]
        color = int(colors[idx])
        hh, ww = map(int, mask.shape)
        coords = coords_list[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - hh + 1):
            for c0 in range(w - ww + 1):
                ok = True
                for rr, cc in coords:
                    if reserved[r0 + rr, c0 + cc]:
                        ok = False
                        break
                if ok:
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None

        r0, c0 = candidates[int(RNG.integers(len(candidates)))]
        for rr, cc in coords:
            grid[r0 + rr, c0 + cc] = color
        for rr, cc in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        reserved[nr, nc] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(RNG.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    dims = _area_preserving_dims(n_pairs)
    rotation_flags = [False] * n_pairs
    rotated_count = int(RNG.integers(1, n_pairs))
    for idx in RNG.choice(n_pairs, size=rotated_count, replace=False):
        rotation_flags[int(idx)] = True

    masks: list[np.ndarray] = []
    for (h, w), needs_rotation in zip(dims, rotation_flags):
        base = _make_base_pair(int(h), int(w))
        if base is None:
            return None
        mask_a, mask_b = base
        inp_a, inp_b = _orient_pair(mask_a, mask_b, needs_rotation)

        full_candidates = _compose_masks(inp_a, inp_b, allow_rotation=True)
        raw_candidates = _compose_masks(inp_a, inp_b, allow_rotation=False)
        if not full_candidates:
            return None
        if needs_rotation and raw_candidates:
            return None
        if (not needs_rotation) and not raw_candidates:
            return None

        masks.extend([inp_a, inp_b])

    matching = _unique_matching(
        [
            [
                i != j and bool(_compose_masks(masks[i], masks[j], allow_rotation=True))
                for j in range(len(masks))
            ]
            for i in range(len(masks))
        ]
    )
    if matching is None:
        return None

    colors = list(int(c) for c in RNG.choice([0, 1, 2, 3, 4, 5, 6, 7, 9], size=len(masks), replace=False))
    grid = _place_masks(masks, colors)
    if grid is None:
        return None

    objects = _extract_objects(grid, BG)
    if len(objects) != len(masks):
        return None
    try:
        _solve_objects(objects)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
