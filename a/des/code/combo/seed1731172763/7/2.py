import random
from collections import defaultdict, deque
from functools import lru_cache

import numpy as np


BG = 8
AXIS_COLOR = 9
OBJECT_COLORS = tuple(c for c in range(10) if c not in (BG, AXIS_COLOR))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r_min = int(arr[:, 0].min())
            c_min = int(arr[:, 1].min())
            r_max = int(arr[:, 0].max())
            c_max = int(arr[:, 1].max())
            mask = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=bool)
            mask[arr[:, 0] - r_min, arr[:, 1] - c_min] = True
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r_min, c_min, r_max, c_max),
                    "mask": mask,
                }
            )

    return components


def _candidate_edges(components: list[dict]) -> list[tuple[int, int, str, int]]:
    groups: dict[tuple[int, tuple[int, int]], list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        groups[(int(comp["color"]), tuple(map(int, comp["mask"].shape)))].append(idx)

    edges: list[tuple[int, int, str, int]] = []
    for indices in groups.values():
        for pos, i in enumerate(indices):
            a = components[i]
            ar0, ac0, ar1, ac1 = map(int, a["bbox"])
            amask = a["mask"]
            for j in indices[pos + 1 :]:
                b = components[j]
                br0, bc0, br1, bc1 = map(int, b["bbox"])
                bmask = b["mask"]

                if ac0 == bc0 and ac1 == bc1 and np.array_equal(amask, np.flipud(bmask)):
                    axis2 = ar0 + br1
                    if axis2 % 2 == 0:
                        edges.append((i, j, "row", axis2 // 2))

                if ar0 == br0 and ar1 == br1 and np.array_equal(amask, np.fliplr(bmask)):
                    axis2 = ac0 + bc1
                    if axis2 % 2 == 0:
                        edges.append((i, j, "col", axis2 // 2))

    return edges


def _select_pairs(components: list[dict]) -> list[tuple[int, int, str, int]]:
    edges = _candidate_edges(components)
    if not edges:
        return []

    neighbours: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    for i, j, axis_kind, axis_pos in edges:
        neighbours[i].append((j, axis_kind, axis_pos, j))
        neighbours[j].append((i, axis_kind, axis_pos, i))

    n = len(components)

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[tuple[int, int, str, int], ...]]:
        if mask == (1 << n) - 1:
            return 0, ()

        i = 0
        while mask & (1 << i):
            i += 1

        best_count, best_pairs = solve(mask | (1 << i))
        for j, axis_kind, axis_pos, _ in neighbours.get(i, []):
            if j <= i or (mask & (1 << j)):
                continue
            count, pairs = solve(mask | (1 << i) | (1 << j))
            pair = (i, j, axis_kind, axis_pos)
            cand = tuple(sorted((pair,) + pairs))
            count += 1
            if count > best_count or (count == best_count and cand < best_pairs):
                best_count = count
                best_pairs = cand
        return best_count, best_pairs

    return list(solve(0)[1])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)

    components = _find_components(g)
    pairs = _select_pairs(components)

    row_axes = {axis_pos for _, _, axis_kind, axis_pos in pairs if axis_kind == "row"}
    col_axes = {axis_pos for _, _, axis_kind, axis_pos in pairs if axis_kind == "col"}

    for r in row_axes:
        out[r, :] = AXIS_COLOR
    for c in col_axes:
        out[:, c] = AXIS_COLOR

    kept = set()
    for i, j, _axis_kind, _axis_pos in pairs:
        kept.add(i)
        kept.add(j)

    for idx in kept:
        comp = components[idx]
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(comp["color"])

    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(1, 5)
    w = rng.randint(1, 5)
    min_area = 2 if h * w > 1 else 1
    max_area = min(h * w, 8)
    if min_area > max_area:
        return None

    target = rng.randint(min_area, max_area)
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        mask[rng.choice(frontier)] = True

    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not cropped.any(axis=0).all() or not cropped.any(axis=1).all():
        return None
    return cropped


def _mask_signature(mask: np.ndarray) -> tuple:
    variants = []
    for variant in (mask, np.flipud(mask), np.fliplr(mask), np.flipud(np.fliplr(mask))):
        variants.append((variant.shape, tuple(int(x) for x in variant.astype(int).ravel())))
    return min(variants)


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask).astype(int, copy=False)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _can_place(coords: np.ndarray, blocked: np.ndarray, reserved_rows: set[int], reserved_cols: set[int]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if rr in reserved_rows or cc in reserved_cols:
            return False
        if blocked[rr, cc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS8 + [(0, 0)]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _sample_axis_positions(limit: int, count: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(2, limit - 2))
    rng.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= 3 for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _place_horizontal_pair(
    mask: np.ndarray,
    axis_row: int,
    blocked: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    grid_h, grid_w = blocked.shape
    top_min = max(0, 2 * axis_row - (grid_h - 1))
    top_max = axis_row - h
    if top_min > top_max or w > grid_w:
        return None

    candidates = [(r0, c0) for r0 in range(top_min, top_max + 1) for c0 in range(grid_w - w + 1)]
    rng.shuffle(candidates)
    mirrored = np.flipud(mask)
    for r0, c0 in candidates:
        bottom_r0 = 2 * axis_row - (r0 + h - 1)
        top_coords = _coords_from_mask(mask, r0, c0)
        bottom_coords = _coords_from_mask(mirrored, bottom_r0, c0)
        if not _can_place(top_coords, blocked, reserved_rows, reserved_cols):
            continue
        if not _can_place(bottom_coords, blocked, reserved_rows, reserved_cols):
            continue
        return top_coords, bottom_coords
    return None


def _place_vertical_pair(
    mask: np.ndarray,
    axis_col: int,
    blocked: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    grid_h, grid_w = blocked.shape
    left_min = max(0, 2 * axis_col - (grid_w - 1))
    left_max = axis_col - w
    if left_min > left_max or h > grid_h:
        return None

    candidates = [(r0, c0) for r0 in range(grid_h - h + 1) for c0 in range(left_min, left_max + 1)]
    rng.shuffle(candidates)
    mirrored = np.fliplr(mask)
    for r0, c0 in candidates:
        right_c0 = 2 * axis_col - (c0 + w - 1)
        left_coords = _coords_from_mask(mask, r0, c0)
        right_coords = _coords_from_mask(mirrored, r0, right_c0)
        if not _can_place(left_coords, blocked, reserved_rows, reserved_cols):
            continue
        if not _can_place(right_coords, blocked, reserved_rows, reserved_cols):
            continue
        return left_coords, right_coords
    return None


def _place_distractor(
    mask: np.ndarray,
    blocked: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = mask.shape
    grid_h, grid_w = blocked.shape
    if h > grid_h or w > grid_w:
        return None

    candidates = [(r0, c0) for r0 in range(grid_h - h + 1) for c0 in range(grid_w - w + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        coords = _coords_from_mask(mask, r0, c0)
        if _can_place(coords, blocked, reserved_rows, reserved_cols):
            return coords
    return None


def _sample_orientations(rng: random.Random) -> list[str]:
    mode = rng.choice(["horizontal_only", "vertical_only", "mixed"])
    if mode == "horizontal_only":
        return ["row"] * rng.randint(1, 3)
    if mode == "vertical_only":
        return ["col"] * rng.randint(1, 3)
    count = rng.randint(2, 4)
    rest = [rng.choice(["row", "col"]) for _ in range(count - 2)]
    out = ["row", "col"] + rest
    rng.shuffle(out)
    return out


def _assign_group_colors(
    signatures: list[tuple],
    rng: random.Random,
) -> list[int] | None:
    total = len(signatures)
    max_palette = min(4, total, len(OBJECT_COLORS))
    min_palette = 1 if total == 1 else 2
    palette_size = rng.randint(min_palette, max_palette)
    palette = rng.sample(list(OBJECT_COLORS), palette_size)

    for _ in range(100):
        colors = [rng.choice(palette) for _ in range(total)]
        if total > 1 and len(set(colors)) == total:
            continue
        ok = True
        for i in range(total):
            for j in range(i + 1, total):
                if colors[i] == colors[j] and signatures[i] == signatures[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return colors
    return None


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    orientations = _sample_orientations(rng)
    n_pairs = len(orientations)
    n_rows = sum(axis == "row" for axis in orientations)
    n_cols = sum(axis == "col" for axis in orientations)
    n_distractors = rng.randint(1, 3)

    grid_h = rng.randint(max(16, 8 + 4 * n_rows), 28)
    grid_w = rng.randint(max(16, 8 + 4 * n_cols), 28)

    row_axes = _sample_axis_positions(grid_h, n_rows, rng)
    col_axes = _sample_axis_positions(grid_w, n_cols, rng)
    if row_axes is None or col_axes is None:
        return None

    row_iter = iter(row_axes)
    col_iter = iter(col_axes)
    reserved_rows = set(row_axes)
    reserved_cols = set(col_axes)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    pair_groups: list[dict] = []
    group_signatures: list[tuple] = []

    for axis_kind in orientations:
        mask = None
        for _ in range(40):
            candidate = _random_connected_mask(rng)
            if candidate is None:
                continue
            if axis_kind == "row" and candidate.shape[0] > 5:
                continue
            if axis_kind == "col" and candidate.shape[1] > 5:
                continue
            mask = candidate
            break
        if mask is None:
            return None

        if axis_kind == "row":
            axis_pos = next(row_iter)
            placed = _place_horizontal_pair(mask, axis_pos, blocked, reserved_rows, reserved_cols, rng)
        else:
            axis_pos = next(col_iter)
            placed = _place_vertical_pair(mask, axis_pos, blocked, reserved_rows, reserved_cols, rng)
        if placed is None:
            return None

        a_coords, b_coords = placed
        _mark_blocked(blocked, a_coords)
        _mark_blocked(blocked, b_coords)
        pair_groups.append(
            {
                "axis_kind": axis_kind,
                "axis_pos": axis_pos,
                "coords_a": a_coords,
                "coords_b": b_coords,
                "mask": mask,
            }
        )
        group_signatures.append(_mask_signature(mask))

    distractors: list[dict] = []
    for _ in range(n_distractors):
        mask = None
        coords = None
        for _ in range(40):
            candidate = _random_connected_mask(rng)
            if candidate is None:
                continue
            placed = _place_distractor(candidate, blocked, reserved_rows, reserved_cols, rng)
            if placed is None:
                continue
            mask = candidate
            coords = placed
            break
        if mask is None or coords is None:
            return None
        _mark_blocked(blocked, coords)
        distractors.append({"coords": coords, "mask": mask})
        group_signatures.append(_mask_signature(mask))

    group_colors = _assign_group_colors(group_signatures, rng)
    if group_colors is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for r in row_axes:
        expected[r, :] = AXIS_COLOR
    for c in col_axes:
        expected[:, c] = AXIS_COLOR

    color_idx = 0
    for group in pair_groups:
        color = group_colors[color_idx]
        color_idx += 1
        _paint_coords(inp, group["coords_a"], color)
        _paint_coords(inp, group["coords_b"], color)
        _paint_coords(expected, group["coords_a"], color)
        _paint_coords(expected, group["coords_b"], color)

    for distractor in distractors:
        color = group_colors[color_idx]
        color_idx += 1
        _paint_coords(inp, distractor["coords"], color)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    found_pairs = _select_pairs(_find_components(inp))
    if len(found_pairs) != n_pairs:
        return None
    found_rows = {pos for _, _, kind, pos in found_pairs if kind == "row"}
    found_cols = {pos for _, _, kind, pos in found_pairs if kind == "col"}
    if found_rows != set(row_axes) or found_cols != set(col_axes):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
