from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 6
COLLISION = 9
SPRITE_COLORS = (1, 2, 3, 4, 7)
DIRS = ("up", "down", "left", "right")

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (16, 24)
SPRITE_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 250
EXTRA_ATTEMPTS = 250


BASE_SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=int),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, (r0, c0, r1, c1)


def _shape_mask(rel: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    mask = np.zeros((h, w), dtype=int)
    for r, c in rel:
        mask[r, c] = 1
    return mask


def _transformed_shape(base: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = base.copy()
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, random.randrange(4))
    coords = [tuple(map(int, rc)) for rc in np.argwhere(arr > 0)]
    rel, _ = _normalize_coords(coords)
    return rel


def _sample_shape(used: set[tuple[tuple[int, int], ...]], rich: bool | None = None) -> tuple[tuple[int, int], ...]:
    choices = BASE_SHAPES[:]
    random.shuffle(choices)
    for _ in range(200):
        rel = _transformed_shape(random.choice(choices))
        if rel in used:
            continue
        rows = {r for r, _ in rel}
        cols = {c for _, c in rel}
        if rich is True and (len(rows) < 2 or len(cols) < 2):
            continue
        if rich is False and (len(rows) < 2 and len(cols) < 2):
            continue
        return rel
    raise RuntimeError("failed to sample a unique shape")


def _bbox_dims(rel: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in rel) + 1, max(c for _, c in rel) + 1


def _marker_position(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return r0 - 1, random.randint(c0, c1)
    if direction == "down":
        return r1 + 1, random.randint(c0, c1)
    if direction == "left":
        return random.randint(r0, r1), c0 - 1
    return random.randint(r0, r1), c1 + 1


def _sprite_cells(spec: dict) -> set[tuple[int, int]]:
    r0, c0 = spec["origin"]
    return {(r0 + r, c0 + c) for r, c in spec["shape"]}


def _sprite_bbox(spec: dict) -> tuple[int, int, int, int]:
    r0, c0 = spec["origin"]
    h, w = spec["size"]
    return r0, c0, r0 + h - 1, c0 + w - 1


def _trajectory_cells(spec: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0 = spec["origin"]
    step = spec["size"][0] if spec["direction"] in ("up", "down") else spec["size"][1]
    dr, dc = {
        "up": (-step, 0),
        "down": (step, 0),
        "left": (0, -step),
        "right": (0, step),
    }[spec["direction"]]

    out: set[tuple[int, int]] = set()
    k = 0
    while True:
        placed = [(r0 + r + dr * k, c0 + c + dc * k) for r, c in spec["shape"]]
        if any(r < 1 or r >= h or c < 0 or c >= w for r, c in placed):
            break
        out.update(placed)
        k += 1
    return out


def _coverage_map(sprites: list[dict], shape: tuple[int, int]) -> tuple[dict[tuple[int, int], set[int]], list[set[tuple[int, int]]]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    trajectories: list[set[tuple[int, int]]] = []
    for idx, spec in enumerate(sprites):
        cells = _trajectory_cells(spec, shape)
        trajectories.append(cells)
        for cell in cells:
            coverage.setdefault(cell, set()).add(idx)
    return coverage, trajectories


def _render_from_specs(shape: tuple[int, int], sprites: list[dict]) -> tuple[np.ndarray, list[int], dict[tuple[int, int], set[int]], list[set[tuple[int, int]]]]:
    h, w = shape
    coverage, trajectories = _coverage_map(sprites, shape)
    out = np.zeros((h, w), dtype=int)
    for (r, c), ids in coverage.items():
        if len(ids) >= 2:
            out[r, c] = COLLISION
        else:
            only = next(iter(ids))
            out[r, c] = sprites[only]["color"]

    counts: list[int] = []
    for idx, cells in enumerate(trajectories):
        counts.append(sum(1 for cell in cells if len(coverage[cell]) >= 2))

    cursor = 0
    first = True
    order = sorted(range(len(sprites)), key=lambda i: (sprites[i]["origin"][1], sprites[i]["origin"][0]))
    for idx in order:
        count = counts[idx]
        if count <= 0:
            continue
        if not first:
            cursor += 1
        if cursor + count > w:
            raise AssertionError("indicator bars do not fit")
        out[0, cursor : cursor + count] = sprites[idx]["color"]
        cursor += count
        first = False
    return out, counts, coverage, trajectories


def _find_marker_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], str]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], str]] = []
    if r0 - 1 >= 1:
        for c in range(c0, c1 + 1):
            if grid[r0 - 1, c] == MARKER:
                candidates.append(((r0 - 1, c), "up"))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if grid[r1 + 1, c] == MARKER:
                candidates.append(((r1 + 1, c), "down"))
    if c0 - 1 >= 0:
        for r in range(r0, r1 + 1):
            if grid[r, c0 - 1] == MARKER:
                candidates.append(((r, c0 - 1), "left"))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if grid[r, c1 + 1] == MARKER:
                candidates.append(((r, c1 + 1), "right"))
    assert len(candidates) == 1, "each sprite must have exactly one side marker"
    return candidates[0]


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    body = (grid != BG) & (grid != MARKER)
    comps = _components(body)
    sprites: list[dict] = []
    used_markers: set[tuple[int, int]] = set()
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "sprite component must be monochromatic"
        color = next(iter(colors))
        rel, bbox = _normalize_coords(comp)
        marker, direction = _find_marker_direction(grid, bbox)
        assert marker not in used_markers, "marker cannot belong to two sprites"
        used_markers.add(marker)
        sprites.append(
            {
                "color": int(color),
                "shape": rel,
                "origin": (bbox[0], bbox[1]),
                "size": _bbox_dims(rel),
                "direction": direction,
                "marker": marker,
            }
        )
    assert np.count_nonzero(grid == MARKER) == len(used_markers), "unexpected extra markers"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each directed sprite repeatedly, recolor trajectory collisions maroon, and encode per-sprite collision totals as top-row bars.

    Input structure:
    - The grid has a black background with row 0 reserved as an initially empty indicator strip.
    - Rows 1..end contain 3-5 small monochrome 4-connected sprites in distinct colors plus one magenta side marker per sprite.
    - A marker sits exactly one cell outside one bbox side of its sprite and that side indicates the stamping direction.

    Rule:
    - For each sprite, erase the magenta marker and treat the sprite as a template anchored at its original bbox.
    - Repeat the full template along the marker direction, shifting by one bbox height for vertical motion or one bbox width for horizontal motion, and keep only fully in-bounds copies inside rows 1..end.
    - Any cell covered by trajectories from at least two different sprites becomes maroon; singly covered cells keep the covering sprite color.
    - Sort sprites by original left column and draw top-row bars in that order, with one black gap between consecutive drawn bars and bar length equal to that sprite's number of collision cells.

    Key insight:
    - The side marker determines a whole stamped trajectory, and the useful quantity is not how many objects collide but how many trajectory pixels of each sprite land inside shared overlap cells.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _parse_sprites(g)
    out, _counts, _coverage, _trajectories = _render_from_specs(g.shape, sprites)
    return out


def _build_input(shape: tuple[int, int], sprites: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for spec in sprites:
        for r, c in _sprite_cells(spec):
            grid[r, c] = spec["color"]
        mr, mc = spec["marker"]
        grid[mr, mc] = MARKER
    return grid


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _spec_valid_in_bounds(spec: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = _sprite_bbox(spec)
    mr, mc = spec["marker"]
    return 1 <= r0 <= r1 < h and 0 <= c0 <= c1 < w and 1 <= mr < h and 0 <= mc < w


def _validate_specs(
    shape: tuple[int, int],
    sprites: list[dict],
    *,
    require_zero: bool | None = None,
    min_sprites: int = SPRITE_COUNT_RANGE[0],
) -> tuple[np.ndarray, list[int]] | None:
    if not (min_sprites <= len(sprites) <= SPRITE_COUNT_RANGE[1]):
        return None
    if len({spec["color"] for spec in sprites}) != len(sprites):
        return None
    if len({spec["origin"][1] for spec in sprites}) != len(sprites):
        return None

    original_sets = [_sprite_cells(spec) for spec in sprites]
    markers = [spec["marker"] for spec in sprites]
    if len(set(markers)) != len(markers):
        return None

    for spec, cells in zip(sprites, original_sets):
        if not _spec_valid_in_bounds(spec, shape):
            return None
        if spec["marker"] in cells:
            return None

    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            if _touching(original_sets[i], original_sets[j]):
                return None
            if sprites[i]["marker"] in original_sets[j] or sprites[j]["marker"] in original_sets[i]:
                return None
            if max(abs(sprites[i]["marker"][0] - sprites[j]["marker"][0]), abs(sprites[i]["marker"][1] - sprites[j]["marker"][1])) <= 1:
                return None

    try:
        out, counts, coverage, trajectories = _render_from_specs(shape, sprites)
    except AssertionError:
        return None

    for i, cells in enumerate(original_sets):
        for j, traj in enumerate(trajectories):
            if i != j and cells & traj:
                return None

    collision_cells = {cell for cell, ids in coverage.items() if len(ids) >= 2}
    positive = [count for count in counts if count > 0]
    if len(positive) < 2:
        return None
    if len(collision_cells) < 2:
        return None
    if max(positive) < 2:
        return None
    if len(collision_cells) > max(9, shape[0] // 2):
        return None

    if require_zero is True and not any(count == 0 for count in counts):
        return None
    if require_zero is False and all(count == 0 for count in counts):
        return None

    inp = _build_input(shape, sprites)
    if np.any(inp[0] != 0):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, counts


def _pair_spec(
    shape: tuple[int, int],
    color_a: int,
    color_b: int,
    used_shapes: set[tuple[tuple[int, int], ...]],
) -> list[dict] | None:
    h, w = shape
    for _ in range(PAIR_ATTEMPTS):
        rel_a = _sample_shape(used_shapes, rich=True)
        used_shapes_a = used_shapes | {rel_a}
        rel_b = _sample_shape(used_shapes_a, rich=True)
        dir_a = random.choice(("up", "down"))
        dir_b = random.choice(("left", "right"))
        ha, wa = _bbox_dims(rel_a)
        hb, wb = _bbox_dims(rel_b)
        a_cell = random.choice(rel_a)
        b_cell = random.choice(rel_b)
        k_a = random.randint(1, 3)
        k_b = random.randint(1, 3)

        target_r = random.randint(4, h - 4)
        target_c = random.randint(3, w - 4)
        s_a = 1 if dir_a == "down" else -1
        s_b = 1 if dir_b == "right" else -1

        r0_a = target_r - a_cell[0] - s_a * k_a * ha
        c0_a = target_c - a_cell[1]
        r0_b = target_r - b_cell[0]
        c0_b = target_c - b_cell[1] - s_b * k_b * wb

        spec_a = {"color": color_a, "shape": rel_a, "origin": (r0_a, c0_a), "size": (ha, wa), "direction": dir_a}
        spec_a["marker"] = _marker_position(_sprite_bbox(spec_a), dir_a)
        spec_b = {"color": color_b, "shape": rel_b, "origin": (r0_b, c0_b), "size": (hb, wb), "direction": dir_b}
        spec_b["marker"] = _marker_position(_sprite_bbox(spec_b), dir_b)

        checked = _validate_specs(shape, [spec_a, spec_b], require_zero=False, min_sprites=2)
        if checked is None:
            continue

        if spec_a["origin"][1] == spec_b["origin"][1]:
            continue

        _, counts = checked
        if counts[0] <= 0 or counts[1] <= 0:
            continue
        return [spec_a, spec_b]
    return None


def _random_spec(shape: tuple[int, int], color: int, used_shapes: set[tuple[tuple[int, int], ...]]) -> dict:
    rel = _sample_shape(used_shapes, rich=None)
    hh, ww = _bbox_dims(rel)
    direction = random.choice(DIRS)
    r_min = 1
    r_max = shape[0] - hh
    c_min = 0
    c_max = shape[1] - ww
    if direction == "up":
        r_min = 2
    elif direction == "down":
        r_max = shape[0] - hh - 1
    elif direction == "left":
        c_min = 1
    else:
        c_max = shape[1] - ww - 1
    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    spec = {"color": color, "shape": rel, "origin": (r0, c0), "size": (hh, ww), "direction": direction}
    spec["marker"] = _marker_position(_sprite_bbox(spec), direction)
    return spec


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (
            random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1]),
            random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1]),
        )
        sprite_count = random.randint(SPRITE_COUNT_RANGE[0], SPRITE_COUNT_RANGE[1])
        colors = random.sample(SPRITE_COLORS, sprite_count)
        require_zero = random.random() < 0.65

        used_shapes: set[tuple[tuple[int, int], ...]] = set()
        pair = _pair_spec(shape, colors[0], colors[1], used_shapes)
        if pair is None:
            continue
        sprites = pair[:]
        used_shapes.update(spec["shape"] for spec in sprites)

        ok = True
        for color in colors[2:]:
            placed = False
            for _ in range(EXTRA_ATTEMPTS):
                spec = _random_spec(shape, color, used_shapes)
                trial = sprites + [spec]
                if _validate_specs(shape, trial, require_zero=None) is None:
                    continue
                sprites.append(spec)
                used_shapes.add(spec["shape"])
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        checked = _validate_specs(shape, sprites, require_zero=require_zero)
        if checked is None:
            continue
        inp, _counts = checked
        return inp
    raise RuntimeError("failed to generate a valid input")
