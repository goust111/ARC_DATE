from __future__ import annotations

import random

import numpy as np


BG = 0
NOISE = 6
OBJECT_COLORS = (1, 2, 3, 4, 7, 8, 9)
GEN_RETRIES = 1000


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects: list[dict] = []
    magenta_points: list[tuple[int, int]] = []

    for color in sorted(set(int(v) for v in np.unique(g)) - {BG}):
        comps = _color_components(g, color)
        if color == NOISE:
            for coords in comps:
                assert coords.shape[0] == 1
                r, c = map(int, coords[0])
                assert all(
                    not (0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in OBJECT_COLORS)
                    for nr, nc in neighbors8(r, c)
                )
                magenta_points.append((r, c))
            continue

        assert color in OBJECT_COLORS
        assert len(comps) == 1
        coords = comps[0]
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "area": int(coords.shape[0]),
                "sprite": sprite,
            }
        )

    assert 5 <= len(objects) <= 7 and len(objects) % 2 == 1
    left_cols = [obj["bbox"][1] for obj in objects]
    assert len(left_cols) == len(set(left_cols))
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert 3 <= obj["area"] <= 5
        assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3
    for i, a in enumerate(objects):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in objects[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            assert ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0
    assert 2 <= len(magenta_points) <= 3
    return objects, magenta_points


def main(grid: np.ndarray) -> np.ndarray:
    """Keep odd horizontal ranks and scale each survivor by that same rank value.

    Input structure:
    - Black background with 5 or 7 colored 4-connected objects, each in its own color from {1,2,3,4,7,8,9}.
    - Every real object has 3-5 pixels and fits inside a non-overlapping bbox of size at most 3x3; its left bbox column is unique.
    - A few isolated magenta single pixels are distractors and are not adjacent to any real object.

    Rule:
    - Ignore the magenta single-pixel distractors, then sort the remaining colored objects from left to right by bbox left column and assign ranks 1,2,3,...
    - Erase every even-ranked object, keep every odd-ranked object, and crop each survivor to its own bbox.
    - For a surviving rank-k object, scale that cropped sprite by factor k with pixel replication, then place all scaled survivors left to right with a one-column black gap and top alignment.

    Key insight:
    - The horizontal rank has two jobs at once: its parity decides which objects survive, and its numeric value is also the survivor's scaling factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects, _ = _extract_objects(g)
    ranked = sorted(objects, key=lambda obj: obj["bbox"][1])

    scaled_parts: list[np.ndarray] = []
    for rank, obj in enumerate(ranked, start=1):
        if rank % 2 == 0:
            continue
        scaled_parts.append(scale_sprite(obj["sprite"], rank))

    assert len(scaled_parts) >= 3
    out_h = max(part.shape[0] for part in scaled_parts)
    out_w = sum(part.shape[1] for part in scaled_parts) + len(scaled_parts) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c0 = 0
    for part in scaled_parts:
        h, w = part.shape
        out[:h, c0 : c0 + w] = part
        c0 += w + 1
    return out


BASE_SHAPES = (
    ((1,), (1,), (1,)),
    ((1, 1, 1),),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 1)),
    ((1, 1, 1), (0, 1, 0)),
    ((1, 1, 0), (0, 1, 1)),
    ((1, 0), (1, 0), (1, 1)),
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 0, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
)


def _trim_binary(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_shape_variants() -> list[np.ndarray]:
    variants: dict[bytes, np.ndarray] = {}
    for shape in BASE_SHAPES:
        arr = np.array(shape, dtype=int)
        for k in range(4):
            rot = np.rot90(arr, k)
            for candidate in (rot, np.fliplr(rot)):
                trimmed = _trim_binary(candidate != 0).astype(int)
                if trimmed.shape[0] <= 3 and trimmed.shape[1] <= 3 and 3 <= int(trimmed.sum()) <= 5:
                    variants[trimmed.tobytes() + str(trimmed.shape).encode()] = trimmed
    return list(variants.values())


SHAPE_VARIANTS = _build_shape_variants()


def _choose_ranked_shapes(n_objects: int, rng: random.Random) -> list[np.ndarray] | None:
    odd_ranks = [rank for rank in range(1, n_objects + 1) if rank % 2 == 1]
    width_budget = 30 - (len(odd_ranks) - 1)
    candidates = SHAPE_VARIANTS[:]

    chosen: list[np.ndarray] = []

    def rec(rank: int, used_width: int) -> bool:
        if rank > n_objects:
            return True
        order = candidates[:]
        rng.shuffle(order)
        for shape in order:
            h, w = shape.shape
            next_used = used_width + (rank * w if rank % 2 == 1 else 0)
            if next_used > width_budget:
                continue
            remaining_min = sum(future_rank for future_rank in range(rank + 1, n_objects + 1, 2))
            if next_used + remaining_min > width_budget:
                continue
            chosen.append(shape)
            if rec(rank + 1, next_used):
                return True
            chosen.pop()
        return False

    if not rec(1, 0):
        return None

    survivor_shapes = [shape for rank, shape in enumerate(chosen, start=1) if rank % 2 == 1]
    survivor_widths = [shape.shape[1] for shape in survivor_shapes]
    survivor_heights = [shape.shape[0] for shape in survivor_shapes]
    if max(survivor_widths) == 1:
        return None
    if len({(h, w) for h, w in zip(survivor_heights, survivor_widths)}) < 2:
        return None
    return [shape.copy() for shape in chosen]


def _render_expected(specs: list[dict]) -> np.ndarray:
    survivors: list[np.ndarray] = []
    for rank, spec in enumerate(specs, start=1):
        if rank % 2 == 0:
            continue
        survivors.append(scale_sprite(spec["patch"], rank))

    out_h = max(sprite.shape[0] for sprite in survivors)
    out_w = sum(sprite.shape[1] for sprite in survivors) + len(survivors) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for sprite in survivors:
        h, w = sprite.shape
        out[:h, c0 : c0 + w] = sprite
        c0 += w + 1
    return out


def _sample_marker_cells(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    forbidden = set(occupied)
    for r, c in list(occupied):
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))

    count = rng.choice((2, 3))
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    rng.shuffle(available)
    picked: list[tuple[int, int]] = []
    for cell in available:
        if any(max(abs(cell[0] - pr), abs(cell[1] - pc)) <= 1 for pr, pc in picked):
            continue
        picked.append(cell)
        if len(picked) == count:
            return picked
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice((5, 7))
    patches = _choose_ranked_shapes(n_objects, rng)
    if patches is None:
        return None

    colors = rng.sample(list(OBJECT_COLORS), n_objects)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    col_gaps = [rng.randint(1, 2) for _ in range(n_objects - 1)]

    left_cols: list[int] = []
    cur_c = left_margin
    for i, patch in enumerate(patches):
        left_cols.append(cur_c)
        cur_c += patch.shape[1]
        if i < n_objects - 1:
            cur_c += col_gaps[i]
    width = cur_c + right_margin
    if not (17 <= width <= 28):
        return None

    height = rng.randint(18, 22)
    specs: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for color, patch, left_col in zip(colors, patches, left_cols):
        ph, pw = patch.shape
        top_row = rng.randint(0, height - ph)
        coords = np.argwhere(patch != 0)
        cells = {(top_row + int(r), left_col + int(c)) for r, c in coords}
        assert not (cells & occupied)
        occupied |= cells
        colored_patch = patch * color
        specs.append(
            {
                "color": color,
                "patch": colored_patch,
                "top": top_row,
                "left": left_col,
                "bbox": (top_row, left_col, top_row + ph - 1, left_col + pw - 1),
            }
        )

    markers = _sample_marker_cells((height, width), occupied, rng)
    if markers is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        ph, pw = spec["patch"].shape
        grid[top : top + ph, left : left + pw] = spec["patch"]
    for r, c in markers:
        grid[r, c] = NOISE

    expected = _render_expected(specs)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
        parsed, parsed_markers = _extract_objects(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if len(parsed) != n_objects or len(parsed_markers) != len(markers):
        return None

    parsed_left_cols = sorted(obj["bbox"][1] for obj in parsed)
    if parsed_left_cols != left_cols:
        return None

    survivor_ranks = [rank for rank in range(1, n_objects + 1) if rank % 2 == 1]
    if survivor_ranks != [1, 3, 5] and survivor_ranks != [1, 3, 5, 7]:
        return None
    if expected.shape[1] < 10 or expected.shape[0] < 3:
        return None
    if len(set(int(v) for v in np.unique(actual) if int(v) != 0)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
