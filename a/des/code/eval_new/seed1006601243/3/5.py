from __future__ import annotations

import numpy as np


BG = 0
GROUND = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400


def _normalize_shape(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _rotate90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _ in shape) + 1
    return _normalize_shape([(int(c), int(height - 1 - r)) for r, c in shape])


def _reflect(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    width = max(c for _, c in shape) + 1
    return _normalize_shape([(int(r), int(width - 1 - c)) for r, c in shape])


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _bottom_front(shape: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    max_r = max(r for r, _ in shape)
    frontier = [(r, c) for r, c in shape if r == max_r]
    if len(frontier) != 1:
        return None
    return frontier[0]


def _build_bottom_shapes() -> list[dict[str, object]]:
    base_shapes = [
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ]

    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[dict[str, object]] = []
    for base in base_shapes:
        seed = _normalize_shape(base)
        for reflected in (seed, _reflect(seed)):
            cur = reflected
            for _ in range(4):
                front = _bottom_front(cur)
                if front is not None and cur not in seen:
                    seen.add(cur)
                    height, width = _shape_dims(cur)
                    shapes.append(
                        {
                            "coords": cur,
                            "height": int(height),
                            "width": int(width),
                            "front": (int(front[0]), int(front[1])),
                        }
                    )
                cur = _rotate90(cur)

    shapes.sort(
        key=lambda item: (
            len(item["coords"]),
            int(item["height"]),
            int(item["width"]),
            tuple(item["coords"]),
        )
    )
    return shapes


BOTTOM_SHAPES = _build_bottom_shapes()


def _detect_ground_side(grid: np.ndarray) -> tuple[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, int]] = []
    if int(g[0, 0]) != BG and np.all(g[0, :] == g[0, 0]):
        candidates.append(("top", int(g[0, 0])))
    if int(g[h - 1, 0]) != BG and np.all(g[h - 1, :] == g[h - 1, 0]):
        candidates.append(("bottom", int(g[h - 1, 0])))
    if int(g[0, 0]) != BG and np.all(g[:, 0] == g[0, 0]):
        candidates.append(("left", int(g[0, 0])))
    if int(g[0, w - 1]) != BG and np.all(g[:, w - 1] == g[0, w - 1]):
        candidates.append(("right", int(g[0, w - 1])))
    assert len(candidates) == 1, "expected exactly one full-edge ground line"
    return candidates[0]


def _to_bottom(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "bottom":
        return grid.copy()
    if side == "top":
        return np.flipud(grid)
    if side == "left":
        return np.rot90(grid, 1)
    assert side == "right"
    return np.rot90(grid, -1)


def _from_bottom(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "bottom":
        return grid.copy()
    if side == "top":
        return np.flipud(grid)
    if side == "left":
        return np.rot90(grid, -1)
    assert side == "right"
    return np.rot90(grid, 1)


def _is_single_component(coords: list[tuple[int, int]]) -> bool:
    cells = set(coords)
    start = coords[0]
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _choose_front(coords: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in coords)
    frontier = sorted((r, c) for r, c in coords if r == max_r)
    if len(frontier) == 1:
        return frontier[0]
    cols = [c for _, c in frontier]
    center = sum(cols) / len(cols)
    return min(frontier, key=lambda rc: (abs(rc[1] - center), rc[1], rc[0]))


def _extract_seeds(bottom_grid: np.ndarray, ground_color: int) -> list[dict[str, object]]:
    colors = [int(c) for c in np.unique(bottom_grid) if int(c) not in {BG, int(ground_color)}]
    seeds: list[dict[str, object]] = []
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(bottom_grid == color)]
        assert 2 <= len(coords) <= 5, "each seed must contain 2 to 5 pixels"
        assert _is_single_component(coords), "each seed color must form one 4-connected component"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        front = _choose_front(coords)
        seeds.append(
            {
                "color": int(color),
                "coords": coords,
                "front": front,
                "height": int(max(rows) - min(rows) + 1),
                "width": int(max(cols) - min(cols) + 1),
                "distance": int(bottom_grid.shape[0] - 1 - front[0]),
            }
        )
    assert seeds, "expected at least one seed"
    seeds.sort(key=lambda seed: (int(seed["distance"]), int(seed["front"][1]), int(seed["color"])))
    return seeds


def _solve_bottom(bottom_grid: np.ndarray, ground_color: int, return_meta: bool = False):
    g = np.asarray(bottom_grid, dtype=int)
    h, _w = g.shape
    seeds = _extract_seeds(g, ground_color)
    out = g.copy()

    truncated_count = 0
    total_branch_pairs = 0
    blocked_branch_count = 0
    new_pixels = 0

    for seed in seeds:
        trunk: list[tuple[int, int]] = []
        fr, fc = seed["front"]
        max_len = max(0, h - 2 - int(fr))
        for step in range(1, max_len + 1):
            rr = int(fr + step)
            if int(out[rr, int(fc)]) != BG:
                break
            out[rr, int(fc)] = int(seed["color"])
            trunk.append((rr, int(fc)))
        seed["trunk"] = trunk
        if len(trunk) < max_len:
            truncated_count += 1
        new_pixels += len(trunk)

    for seed in seeds:
        width = int(seed["width"])
        length = int(seed["height"])
        trunk = seed["trunk"]
        branch_origins = [trunk[k - 1] for k in range(width, len(trunk) + 1, width)]
        seed["branch_origins"] = branch_origins
        total_branch_pairs += len(branch_origins)

        for rr, cc in branch_origins:
            for dc in (-1, 1):
                drawn = 0
                for step in range(1, length + 1):
                    nc = int(cc + dc * step)
                    if not (0 <= nc < g.shape[1]):
                        break
                    if int(out[int(rr), nc]) != BG:
                        break
                    out[int(rr), nc] = int(seed["color"])
                    drawn += 1
                new_pixels += drawn
                if drawn < length:
                    blocked_branch_count += 1

    if not return_meta:
        return out

    meta = {
        "truncated_count": int(truncated_count),
        "blocked_branch_count": int(blocked_branch_count),
        "total_branch_pairs": int(total_branch_pairs),
        "new_pixels": int(new_pixels),
        "seed_count": int(len(seeds)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each small seed into a tree whose trunk points toward the grey edge and whose branches are encoded by the seed's spans.

    Input structure:
    - The background is black.
    - Exactly one full grid edge is a solid 1-cell-wide grey ground line.
    - Away from that edge sit several small single-color connected seeds, each using a distinct non-grey foreground color.
    - Each seed occupies 2 to 5 cells, and its geometry relative to the ground line defines a span parallel to the ground and a span perpendicular to it.

    Rule:
    - Detect the grey edge and treat it as the growth target; each seed grows from the seed pixel closest to that edge along the perpendicular direction.
    - Paint the 1-cell-wide trunk through background cells until the next step would hit the grey line or an already occupied cell; seeds themselves stay unchanged.
    - The seed's span parallel to the ground line is the branch interval, and its span perpendicular to the ground line is the branch length; after every interval along the realized trunk, grow one branch in each ground-parallel direction until the branch reaches its full length or hits an occupied cell.
    - When two seeds share the same trunk line, the seed nearer the ground is painted first, so its trunk truncates the farther seed's trunk before branches are added.

    Key insight:
    - The grey edge provides the global orientation, but each seed contributes two local parameters: its parallel span determines branch spacing and its perpendicular span determines branch length.
    """
    g = np.asarray(grid, dtype=int)
    side, ground_color = _detect_ground_side(g)
    bottom = _to_bottom(g, side)
    solved = _solve_bottom(bottom, ground_color, return_meta=False)
    return _from_bottom(solved, side)


def _bbox_with_padding(r0: int, c0: int, height: int, width: int, pad: int = 1) -> tuple[int, int, int, int]:
    return (int(r0 - pad), int(c0 - pad), int(r0 + height - 1 + pad), int(c0 + width - 1 + pad))


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _can_place(placed: list[dict[str, object]], r0: int, c0: int, shape: dict[str, object], h: int, w: int) -> bool:
    box = _bbox_with_padding(r0, c0, int(shape["height"]), int(shape["width"]))
    if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
        return False
    for item in placed:
        if _boxes_intersect(box, item["box"]):
            return False
    return True


def _place_shape(grid: np.ndarray, shape: dict[str, object], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape["coords"]:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _sample_shapes(rng: np.random.Generator, count: int, need_blocked: bool) -> list[dict[str, object]] | None:
    if count > len(BOTTOM_SHAPES):
        return None
    for _ in range(80):
        indices = rng.choice(len(BOTTOM_SHAPES), size=count, replace=False)
        shapes = [BOTTOM_SHAPES[int(i)] for i in indices]
        if not any(int(shape["width"]) > 1 for shape in shapes):
            continue
        if not any(int(shape["height"]) > 1 for shape in shapes):
            continue
        if need_blocked and not any(int(shape["height"]) > 1 for shape in shapes):
            continue
        return shapes
    return None


def _random_r0(rng: np.random.Generator, shape: dict[str, object], h: int) -> int | None:
    max_r0 = int(h - int(shape["height"]) - int(shape["width"]) - 1)
    if max_r0 < 1:
        return None
    return int(rng.integers(1, max_r0 + 1))


def _front_col_range(shape: dict[str, object], w: int) -> tuple[int, int]:
    front_c = int(shape["front"][1])
    low = int(1 + front_c)
    high = int(w - int(shape["width"]) - 1 + front_c)
    return low, high


def _place_collinear_pair(
    rng: np.random.Generator,
    placed: list[dict[str, object]],
    shape_far: dict[str, object],
    shape_near: dict[str, object],
    h: int,
    w: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    low_far, high_far = _front_col_range(shape_far, w)
    low_near, high_near = _front_col_range(shape_near, w)
    low = max(low_far, low_near)
    high = min(high_far, high_near)
    if low > high:
        return None

    near_min_r0 = int(1 + int(shape_far["height"]) + 2)
    near_max_r0 = int(h - int(shape_near["height"]) - int(shape_near["width"]) - 1)
    if near_min_r0 > near_max_r0:
        return None

    for _ in range(80):
        target_front_col = int(rng.integers(low, high + 1))
        near_r0 = int(rng.integers(near_min_r0, near_max_r0 + 1))
        far_max_r0 = min(
            int(h - int(shape_far["height"]) - int(shape_far["width"]) - 1),
            int(near_r0 - int(shape_far["height"]) - 2),
        )
        if far_max_r0 < 1:
            continue
        far_r0 = int(rng.integers(1, far_max_r0 + 1))
        far_c0 = int(target_front_col - int(shape_far["front"][1]))
        near_c0 = int(target_front_col - int(shape_near["front"][1]))
        if not _can_place(placed, far_r0, far_c0, shape_far, h, w):
            continue
        far_info = {
            "shape": shape_far,
            "r0": int(far_r0),
            "c0": int(far_c0),
            "box": _bbox_with_padding(int(far_r0), int(far_c0), int(shape_far["height"]), int(shape_far["width"])),
        }
        if not _can_place(placed + [far_info], near_r0, near_c0, shape_near, h, w):
            continue
        near_info = {
            "shape": shape_near,
            "r0": int(near_r0),
            "c0": int(near_c0),
            "box": _bbox_with_padding(int(near_r0), int(near_c0), int(shape_near["height"]), int(shape_near["width"])),
        }
        return far_info, near_info
    return None


def _place_boundary_seed(
    rng: np.random.Generator,
    placed: list[dict[str, object]],
    shape: dict[str, object],
    h: int,
    w: int,
) -> dict[str, object] | None:
    if int(shape["height"]) < 2:
        return None
    low, high = _front_col_range(shape, w)
    for _ in range(80):
        side = "left" if int(rng.integers(2)) == 0 else "right"
        if side == "left":
            t_low = low
            t_high = min(high, int(shape["height"]) - 1)
        else:
            t_low = max(low, w - int(shape["height"]))
            t_high = high
        if t_low > t_high:
            continue
        target_front_col = int(rng.integers(t_low, t_high + 1))
        r0 = _random_r0(rng, shape, h)
        if r0 is None:
            continue
        c0 = int(target_front_col - int(shape["front"][1]))
        if not _can_place(placed, r0, c0, shape, h, w):
            continue
        return {
            "shape": shape,
            "r0": int(r0),
            "c0": int(c0),
            "box": _bbox_with_padding(int(r0), int(c0), int(shape["height"]), int(shape["width"])),
        }
    return None


def _place_random_seed(
    rng: np.random.Generator,
    placed: list[dict[str, object]],
    shape: dict[str, object],
    h: int,
    w: int,
) -> dict[str, object] | None:
    for _ in range(100):
        r0 = _random_r0(rng, shape, h)
        if r0 is None:
            return None
        c0 = int(rng.integers(1, w - int(shape["width"])))
        if not _can_place(placed, r0, c0, shape, h, w):
            continue
        return {
            "shape": shape,
            "r0": int(r0),
            "c0": int(c0),
            "box": _bbox_with_padding(int(r0), int(c0), int(shape["height"]), int(shape["width"])),
        }
    return None


def _generate_bottom_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    need_collinear = bool(int(rng.integers(2)))
    need_blocked = bool(int(rng.integers(2)))
    min_seeds = 2 + int(need_blocked and need_collinear)
    count = int(rng.integers(min_seeds, 6))
    h = int(rng.integers(17, 25))
    w = int(rng.integers(17, 25))

    shapes = _sample_shapes(rng, count, need_blocked)
    if shapes is None:
        return None

    placed: list[dict[str, object]] = []
    remaining_shapes = list(shapes)

    if need_collinear:
        pair = _place_collinear_pair(rng, placed, remaining_shapes[0], remaining_shapes[1], h, w)
        if pair is None:
            return None
        placed.extend(pair)
        remaining_shapes = remaining_shapes[2:]

    if need_blocked:
        blocked_index = None
        for idx, shape in enumerate(remaining_shapes):
            if int(shape["height"]) > 1:
                blocked_index = idx
                break
        if blocked_index is None:
            return None
        blocked_shape = remaining_shapes.pop(blocked_index)
        blocked_seed = _place_boundary_seed(rng, placed, blocked_shape, h, w)
        if blocked_seed is None:
            return None
        placed.append(blocked_seed)

    for shape in remaining_shapes:
        info = _place_random_seed(rng, placed, shape, h, w)
        if info is None:
            return None
        placed.append(info)

    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = GROUND
    colors = [int(c) for c in rng.choice(PALETTE, size=count, replace=False)]
    for color, item in zip(colors, placed):
        _place_shape(grid, item["shape"], int(item["r0"]), int(item["c0"]), color)

    bottom_out, meta = _solve_bottom(grid, GROUND, return_meta=True)
    if meta["total_branch_pairs"] < 1:
        return None
    if meta["new_pixels"] < count + 2:
        return None
    if need_collinear and meta["truncated_count"] < 1:
        return None
    if need_blocked and meta["blocked_branch_count"] < 1:
        return None

    side = ("bottom", "top", "left", "right")[int(rng.integers(4))]
    inp = _from_bottom(grid, side)
    out = _from_bottom(bottom_out, side)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _generate_bottom_case(rng)
        if built is None:
            continue
        inp, _out, _meta = built
        return inp
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
