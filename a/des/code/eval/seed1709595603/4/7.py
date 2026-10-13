from __future__ import annotations

from collections import defaultdict, deque
import math
import random

import numpy as np


BG = 0
SHADOW = 5
OBJECT_COLORS = tuple(color for color in range(10) if color not in {BG, SHADOW})
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_monochrome_components(grid: np.ndarray, colors: list[int]) -> list[dict]:
    h, w = grid.shape
    components: list[dict] = []

    for color in colors:
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        starts = np.argwhere(mask)
        for sr, sc in starts:
            sr = int(sr)
            sc = int(sc)
            if seen[sr, sc]:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": int(local.sum()),
                }
            )

    return components


def _shape_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple(int(v) for v in mask.astype(np.uint8).ravel()),
    )


def _shadow_score(grid: np.ndarray, obj: dict) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    mask = obj["mask"]
    h, w = mask.shape
    sr0 = r1 + 2
    sr1 = sr0 + h - 1
    assert sr1 < grid.shape[0], "each object must have room for its shadow directly below"
    shadow = grid[sr0 : sr1 + 1, c0 : c1 + 1]
    assert shadow.shape == mask.shape
    return int(np.count_nonzero((shadow == SHADOW) & mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the color copy whose grey shadow is most complete within each same-shape group.

    Objects are monochrome connected components on a black background. Each object has a grey
    shadow of the same bbox directly below it with one empty row gap; missing shadow pixels are
    black. Objects are grouped by exact normalized shape, then the group member with the largest
    number of preserved grey shadow pixels wins. Ties are broken by the leftmost object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    object_colors = [int(color) for color in np.unique(g) if int(color) not in {BG, SHADOW}]
    objects = _find_monochrome_components(g, object_colors)
    assert objects, "expected at least one colored object"

    groups: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[_shape_key(obj["mask"])].append(obj)

    out = np.full_like(g, BG)
    for members in groups.values():
        assert 3 <= len(members) <= 4, "each group must contain 3 or 4 same-shape copies"
        scored = [(int(_shadow_score(g, obj)), obj) for obj in members]
        best_score = max(score for score, _obj in scored)
        candidates = [obj for score, obj in scored if score == best_score]
        winner = min(candidates, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))

        r0, c0, r1, c1 = winner["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][winner["mask"]] = int(winner["color"])

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
GROUP_COUNT_MIN = 1
GROUP_COUNT_MAX = 3
GROUP_SIZE_CHOICES = (3, 4)
MASK_AREA_MIN = 4
MASK_AREA_MAX = 8
MASK_HEIGHT_MIN = 2
MASK_HEIGHT_MAX = 4
MASK_WIDTH_MIN = 2
MASK_WIDTH_MAX = 5
PLACEMENT_PAD = 1
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 400
MODE_UNIQUE = "unique"
MODE_TIE = "tie"


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_transform(mask: np.ndarray) -> np.ndarray:
    transformed = np.array(mask, copy=True)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    k = random.randint(0, 3)
    transformed = np.rot90(transformed, k)
    return transformed.copy()


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(MASK_HEIGHT_MIN, MASK_HEIGHT_MAX)
        box_w = random.randint(MASK_WIDTH_MIN, MASK_WIDTH_MAX)
        max_area = min(MASK_AREA_MAX, box_h * box_w)
        if max_area < MASK_AREA_MIN:
            continue
        area = random.randint(MASK_AREA_MIN, max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = _cells_to_mask(_normalize_cells(cells))
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        return _random_transform(mask)
    return None


def _sample_modes(group_count: int) -> list[str]:
    if group_count == 1:
        return [random.choice((MODE_UNIQUE, MODE_TIE))]
    modes = [MODE_UNIQUE, MODE_TIE]
    while len(modes) < group_count:
        modes.append(random.choice((MODE_UNIQUE, MODE_TIE)))
    random.shuffle(modes)
    return modes


def _reserve_bounds(
    top: int, left: int, height: int, width: int, grid_shape: tuple[int, int], pad: int
) -> tuple[int, int, int, int]:
    rows, cols = grid_shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + height + pad)
    c1 = min(cols, left + width + pad)
    return r0, c0, r1, c1


def _can_place(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    r0, c0, r1, c1 = _reserve_bounds(top, left, height, width, used.shape, pad)
    return not bool(used[r0:r1, c0:c1].any())


def _mark_used(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    r0, c0, r1, c1 = _reserve_bounds(top, left, height, width, used.shape, pad)
    used[r0:r1, c0:c1] = True


def _place_group(mask: np.ndarray, copies: int, used: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    pair_h = 2 * h + 1
    positions: list[tuple[int, int]] = []
    used_cols: set[int] = set()

    for _ in range(copies):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, used.shape[0] - pair_h)
            left = random.randint(0, used.shape[1] - w)
            if left in used_cols:
                continue
            if not _can_place(used, top, left, pair_h, w, PLACEMENT_PAD):
                continue
            _mark_used(used, top, left, pair_h, w, PLACEMENT_PAD)
            positions.append((top, left))
            used_cols.add(left)
            placed = True
            break
        if not placed:
            return None

    return positions


def _pick_shadow_counts(area: int, placements: list[tuple[int, int]], mode: str) -> tuple[list[int], int]:
    size = len(placements)
    sorted_ids = sorted(range(size), key=lambda idx: (placements[idx][1], placements[idx][0]))
    top_count = random.randint(max(2, area // 2), area - 1)

    if mode == MODE_UNIQUE:
        winner = random.randrange(size)
        counts = [random.randint(1, top_count - 1) for _ in range(size)]
        counts[winner] = top_count
        return counts, winner

    tie_size = random.randint(2, size)
    winners = {sorted_ids[0]}
    if tie_size > 1:
        winners.update(random.sample(sorted_ids[1:], tie_size - 1))
    counts = [random.randint(1, top_count - 1) for _ in range(size)]
    for idx in winners:
        counts[idx] = top_count
    return counts, sorted_ids[0]


def _distinct_shadow_subsets(mask: np.ndarray, counts: list[int]) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    seen: set[tuple[tuple[int, int], ...]] = set()
    subsets: list[np.ndarray] = []

    for count in counts:
        chosen_tuple: tuple[tuple[int, int], ...] | None = None
        for _ in range(200):
            sample = tuple(sorted(random.sample(coords, count)))
            if sample not in seen:
                chosen_tuple = sample
                seen.add(sample)
                break
        if chosen_tuple is None:
            return None

        shadow_mask = np.zeros_like(mask, dtype=bool)
        for r, c in chosen_tuple:
            shadow_mask[r, c] = True
        subsets.append(shadow_mask)

    return subsets


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = int(color)


def _sample_group_specs() -> list[dict] | None:
    group_count = random.randint(GROUP_COUNT_MIN, GROUP_COUNT_MAX)
    modes = _sample_modes(group_count)
    seen_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    groups: list[dict] = []

    for mode in modes:
        mask = _random_connected_mask()
        if mask is None:
            return None
        key = _shape_key(mask)
        tries = 0
        while key in seen_keys and tries < SHAPE_ATTEMPTS:
            mask = _random_connected_mask()
            if mask is None:
                return None
            key = _shape_key(mask)
            tries += 1
        if key in seen_keys:
            return None

        groups.append(
            {
                "mask": mask,
                "mode": mode,
                "copies": random.choice(GROUP_SIZE_CHOICES),
            }
        )
        seen_keys.add(key)

    return groups


def _choose_grid_shape(groups: list[dict]) -> tuple[int, int] | None:
    min_rows = max(2 * group["mask"].shape[0] + 1 for group in groups)
    min_cols = max(group["mask"].shape[1] for group in groups)
    padded_area = sum(
        group["copies"] * (2 * group["mask"].shape[0] + 1 + 2 * PLACEMENT_PAD) * (group["mask"].shape[1] + 2 * PLACEMENT_PAD)
        for group in groups
    )

    for _ in range(40):
        rows = random.randint(max(GRID_MIN, min_rows), GRID_MAX)
        needed_cols = max(min_cols, math.ceil(padded_area * 1.25 / rows))
        if needed_cols > GRID_MAX:
            continue
        cols = random.randint(max(GRID_MIN, needed_cols), GRID_MAX)
        return rows, cols
    return None


def _build_case(groups: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = shape
    used = np.zeros((rows, cols), dtype=bool)
    placements: list[list[tuple[int, int]] | None] = [None] * len(groups)

    order = sorted(
        range(len(groups)),
        key=lambda idx: groups[idx]["copies"] * (2 * groups[idx]["mask"].shape[0] + 1) * groups[idx]["mask"].shape[1],
        reverse=True,
    )
    for idx in order:
        pos = _place_group(groups[idx]["mask"], groups[idx]["copies"], used)
        if pos is None:
            return None
        placements[idx] = pos

    inp = np.full((rows, cols), BG, dtype=int)
    expected = np.full((rows, cols), BG, dtype=int)

    for group, pos in zip(groups, placements, strict=False):
        assert pos is not None
        mask = group["mask"]
        area = int(mask.sum())
        counts, winner_idx = _pick_shadow_counts(area, pos, group["mode"])
        shadow_masks = _distinct_shadow_subsets(mask, counts)
        if shadow_masks is None:
            return None

        colors = random.sample(list(OBJECT_COLORS), len(pos))
        h, _w = mask.shape
        for idx, ((top, left), color, shadow_mask) in enumerate(zip(pos, colors, shadow_masks, strict=False)):
            _draw_mask(inp, top, left, mask, color)
            _draw_mask(inp, top + h + 1, left, shadow_mask, SHADOW)
            if idx == winner_idx:
                _draw_mask(expected, top, left, mask, color)

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    groups = _sample_group_specs()
    if groups is None:
        return None

    grid_shape = _choose_grid_shape(groups)
    if grid_shape is None:
        return None

    case = _build_case(groups, grid_shape)
    if case is None:
        return None
    inp, expected = case

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
