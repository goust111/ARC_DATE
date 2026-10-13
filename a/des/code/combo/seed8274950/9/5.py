from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


OVERLAP_COLOR = 9
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(9))
SHAPE_DIMS = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 6)
    if h % 2 == w % 2
)
PAIR_COUNT_CHOICES = (1, 2, 3)
UNPAIRED_COUNT_CHOICES = (0, 1, 2)
SCATTER_OFFSETS = tuple(
    (dr, dc)
    for dr in range(-6, 7)
    for dc in range(-6, 7)
    if (dr, dc) != (0, 0) and 2 <= max(abs(dr), abs(dc)) <= 6
)
MAX_GENERATION_RETRIES = 400
MAX_SHAPE_RETRIES = 200
MAX_PLACEMENT_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            mask, bbox = _component_mask(coords)
            r0, c0, r1, c1 = bbox
            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": bbox,
                    "mask": mask,
                    "center2": (r0 + r1, c0 + c1),
                }
            )
    return components


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int).tolist())


def _rotation_key(mask: np.ndarray, k: int) -> tuple[tuple[int, ...], ...]:
    return _mask_key(np.rot90(mask, k))


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_rotation_key(mask, k) for k in range(4))


def _is_quarter_turn_match(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.array_equal(np.rot90(mask_a, 1), mask_b) or np.array_equal(np.rot90(mask_a, 3), mask_b))


def _best_matching(components: list[dict]) -> tuple[tuple[int, int], ...]:
    n = len(components)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _is_quarter_turn_match(components[i]["mask"], components[j]["mask"]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[tuple[int, int], ...]:
        if mask == 0:
            return ()
        i_bit = mask & -mask
        i = i_bit.bit_length() - 1
        best = solve(mask ^ i_bit)
        candidates = adj[i] & (mask ^ i_bit)
        while candidates:
            j_bit = candidates & -candidates
            j = j_bit.bit_length() - 1
            candidate = ((i, j),) + solve(mask ^ i_bit ^ j_bit)
            if len(candidate) > len(best) or (len(candidate) == len(best) and candidate < best):
                best = candidate
            candidates ^= j_bit
        return best

    return solve((1 << n) - 1)


def _check_in_bounds(coords: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in coords:
        assert 0 <= r < h and 0 <= c < w


def main(grid: np.ndarray) -> np.ndarray:
    """Merge every 90-degree-rotation pair at the midpoint of their bbox centers."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)
    pairs = _best_matching(components)
    paired = {idx for pair in pairs for idx in pair}

    out = np.full_like(g, bg)
    for idx, comp in enumerate(components):
        if idx in paired:
            continue
        for r, c in comp["coords"]:
            out[r, c] = int(comp["color"])

    height, width = g.shape
    for i, j in pairs:
        first = components[i]
        second = components[j]
        dr_num = int(second["center2"][0] - first["center2"][0])
        dc_num = int(second["center2"][1] - first["center2"][1])
        assert dr_num % 4 == 0 and dc_num % 4 == 0
        dr = dr_num // 4
        dc = dc_num // 4

        moved_first = {(r + dr, c + dc) for r, c in first["coords"]}
        moved_second = {(r - dr, c - dc) for r, c in second["coords"]}
        _check_in_bounds(moved_first | moved_second, (height, width))

        overlap = moved_first & moved_second
        for r, c in moved_first - overlap:
            out[r, c] = int(first["color"])
        for r, c in moved_second - overlap:
            out[r, c] = int(second["color"])
        for r, c in overlap:
            out[r, c] = OVERLAP_COLOR

    return out


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    path = [(r, c)]
    while (r, c) != end:
        steps: list[tuple[int, int]] = []
        if r < end[0]:
            steps.append((1, 0))
        elif r > end[0]:
            steps.append((-1, 0))
        if c < end[1]:
            steps.append((0, 1))
        elif c > end[1]:
            steps.append((0, -1))
        dr, dc = rng.choice(steps)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _grow_connected_shape(h: int, w: int, rng: random.Random) -> np.ndarray:
    anchors = [
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    ]
    order = anchors[:]
    rng.shuffle(order)
    cells = {order[0]}
    for anchor in order[1:]:
        target = rng.choice(tuple(cells))
        cells.update(_random_path(anchor, target, rng))

    max_area = min(h * w, max(len(cells) + 1, int(np.ceil(h * w * 0.8))))
    target_area = rng.randint(len(cells), max_area)
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _centered_offsets(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    return {
        (2 * r - (h - 1), 2 * c - (w - 1))
        for r, c in map(tuple, np.argwhere(mask))
    }


def _centered_overlap_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return len(_centered_offsets(mask_a) & _centered_offsets(mask_b))


def _make_shape(rng: random.Random, used_keys: set[tuple[tuple[int, ...], ...]], require_pair_shape: bool) -> np.ndarray | None:
    dims = list(SHAPE_DIMS)
    rng.shuffle(dims)
    for _ in range(MAX_SHAPE_RETRIES):
        h, w = rng.choice(dims)
        mask = _grow_connected_shape(h, w, rng)
        key = _canonical_rotation_key(mask)
        if key in used_keys:
            continue
        if require_pair_shape:
            rotated = np.rot90(mask, 1)
            if np.array_equal(mask, rotated):
                continue
            overlap = _centered_overlap_count(mask, rotated)
            area = int(mask.sum())
            if overlap <= 0 or overlap >= area:
                continue
        used_keys.add(key)
        return mask
    return None


def _coords_at(mask: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top
    return {
        (r0 + r, c0 + c)
        for r, c in map(tuple, np.argwhere(mask))
    }


def _expanded(coords: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    around: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    around.add((nr, nc))
    return around


def _all_in_bounds(coords: set[tuple[int, int]], height: int, width: int) -> bool:
    return all(0 <= r < height and 0 <= c < width for r, c in coords)


def _sample_center2(low: int, high: int, parity: int, rng: random.Random) -> int | None:
    start = low if low % 2 == parity else low + 1
    if start > high:
        return None
    return start + 2 * rng.randrange((high - start) // 2 + 1)


def _top_left_from_center2(center2: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    return (
        (center2[0] - (shape[0] - 1)) // 2,
        (center2[1] - (shape[1] - 1)) // 2,
    )


def _place_pair(
    mask: np.ndarray,
    height: int,
    width: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    rotated = np.rot90(mask, 1)
    h, w = mask.shape
    center_low_r = max(h - 1, w - 1)
    center_high_r = min(2 * height - h - 1, 2 * height - w - 1)
    center_low_c = max(w - 1, h - 1)
    center_high_c = min(2 * width - w - 1, 2 * width - h - 1)
    parity = (h - 1) % 2

    for _ in range(MAX_PLACEMENT_RETRIES):
        center2_r = _sample_center2(center_low_r, center_high_r, parity, rng)
        center2_c = _sample_center2(center_low_c, center_high_c, parity, rng)
        if center2_r is None or center2_c is None:
            return None
        center2 = (center2_r, center2_c)
        delta = rng.choice(SCATTER_OFFSETS)

        out_a_top = _top_left_from_center2(center2, mask.shape)
        out_b_top = _top_left_from_center2(center2, rotated.shape)
        in_a_top = (out_a_top[0] - delta[0], out_a_top[1] - delta[1])
        in_b_top = (out_b_top[0] + delta[0], out_b_top[1] + delta[1])

        in_a = _coords_at(mask, in_a_top)
        in_b = _coords_at(rotated, in_b_top)
        out_a = _coords_at(mask, out_a_top)
        out_b = _coords_at(rotated, out_b_top)
        if not (_all_in_bounds(in_a, height, width) and _all_in_bounds(in_b, height, width)):
            continue
        if any(cell in input_forbidden for cell in in_a | in_b):
            continue
        if in_a & _expanded(in_b, height, width):
            continue
        out_union = out_a | out_b
        if any(cell in output_forbidden for cell in out_union):
            continue

        return {
            "input_a": in_a,
            "input_b": in_b,
            "output_a": out_a,
            "output_b": out_b,
        }
    return None


def _place_unpaired(
    mask: np.ndarray,
    height: int,
    width: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    h, w = mask.shape
    for _ in range(MAX_PLACEMENT_RETRIES):
        top = (rng.randrange(height - h + 1), rng.randrange(width - w + 1))
        coords = _coords_at(mask, top)
        if any(cell in input_forbidden for cell in coords):
            continue
        if any(cell in output_forbidden for cell in coords):
            continue
        return {"coords": coords}
    return None


def _paint(coords: set[tuple[int, int]], color: int, grid: np.ndarray) -> None:
    for r, c in coords:
        grid[r, c] = color


def _build_expected(height: int, width: int, bg: int, pairs: list[dict], unpaired: list[dict]) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for obj in unpaired:
        _paint(obj["coords"], int(obj["color"]), out)
    for pair in pairs:
        overlap = pair["output_a"] & pair["output_b"]
        _paint(pair["output_a"] - overlap, int(pair["color_a"]), out)
        _paint(pair["output_b"] - overlap, int(pair["color_b"]), out)
        _paint(overlap, OVERLAP_COLOR, out)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    unpaired_count = rng.choice(UNPAIRED_COUNT_CHOICES)
    object_count = 2 * pair_count + unpaired_count
    base_side = min(32, 14 + 2 * object_count)
    height = rng.randint(base_side, min(34, base_side + 4))
    width = rng.randint(base_side, min(34, base_side + 4))
    bg = rng.choice(COLOR_POOL)
    colors = [color for color in COLOR_POOL if color != bg]
    rng.shuffle(colors)
    if len(colors) < object_count:
        return None

    used_keys: set[tuple[tuple[int, ...], ...]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    pair_records: list[dict] = []
    unpaired_records: list[dict] = []
    color_index = 0

    for _ in range(pair_count):
        mask = _make_shape(rng, used_keys, require_pair_shape=True)
        if mask is None:
            return None
        placed = _place_pair(mask, height, width, input_forbidden, output_forbidden, rng)
        if placed is None:
            return None
        color_a = colors[color_index]
        color_b = colors[color_index + 1]
        color_index += 2
        placed["color_a"] = color_a
        placed["color_b"] = color_b
        pair_records.append(placed)
        input_forbidden |= _expanded(placed["input_a"] | placed["input_b"], height, width)
        output_forbidden |= _expanded(placed["output_a"] | placed["output_b"], height, width)

    for _ in range(unpaired_count):
        mask = _make_shape(rng, used_keys, require_pair_shape=False)
        if mask is None:
            return None
        placed = _place_unpaired(mask, height, width, input_forbidden, output_forbidden, rng)
        if placed is None:
            return None
        placed["color"] = colors[color_index]
        color_index += 1
        unpaired_records.append(placed)
        input_forbidden |= _expanded(placed["coords"], height, width)
        output_forbidden |= _expanded(placed["coords"], height, width)

    grid = np.full((height, width), bg, dtype=int)
    for pair in pair_records:
        _paint(pair["input_a"], int(pair["color_a"]), grid)
        _paint(pair["input_b"], int(pair["color_b"]), grid)
    for obj in unpaired_records:
        _paint(obj["coords"], int(obj["color"]), grid)

    expected = _build_expected(height, width, bg, pair_records, unpaired_records)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if get_bg(grid) != bg:
        return None
    if len(_find_components(grid, bg)) != object_count:
        return None
    if np.count_nonzero(actual == OVERLAP_COLOR) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
