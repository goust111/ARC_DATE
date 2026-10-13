from __future__ import annotations

import importlib.util
import random

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load ARC utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location
scale_sprite = _ARC_UTILS.scale_sprite


def _find_marker_and_target(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 4:
            continue
        if len(components_mask(grid == color, connectivity=4)) != 4:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        if np.sum(rows == rows.min()) != 1 or np.sum(rows == rows.max()) != 1:
            continue
        if np.sum(cols == cols.min()) != 1 or np.sum(cols == cols.max()) != 1:
            continue
        r0 = int(rows.min()) + 1
        r1 = int(rows.max()) - 1
        c0 = int(cols.min()) + 1
        c1 = int(cols.max()) - 1
        if r0 > r1 or c0 > c1:
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != bg):
            continue
        candidates.append((int(color), (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected one marker color defining one empty target region"
    return candidates[0]


def _component_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords.tolist():
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _downscale_blocks(sprite: np.ndarray, factor: int, out_shape: tuple[int, int]) -> np.ndarray | None:
    out_h, out_w = map(int, out_shape)
    k = int(factor)
    if k <= 0 or sprite.shape != (out_h * k, out_w * k):
        return None
    blocks = sprite.reshape(out_h, k, out_w, k)
    merged = blocks[:, 0, :, 0].copy()
    if np.all(blocks == merged[:, None, :, None]):
        return merged
    return None


def _fit_results(sprite: np.ndarray, target_shape: tuple[int, int]) -> list[np.ndarray]:
    target_h, target_w = map(int, target_shape)
    results: list[np.ndarray] = []
    for rot_k in (0, 1, 3):
        oriented = np.rot90(sprite, rot_k)
        src_h, src_w = map(int, oriented.shape)

        if target_h % src_h == 0 and target_w % src_w == 0:
            k_row = target_h // src_h
            k_col = target_w // src_w
            if k_row == k_col:
                results.append(scale_sprite(oriented, k_row))

        if src_h % target_h == 0 and src_w % target_w == 0:
            k_row = src_h // target_h
            k_col = src_w // target_w
            if k_row == k_col:
                reduced = _downscale_blocks(oriented, k_row, (target_h, target_w))
                if reduced is not None:
                    results.append(reduced)

    unique: list[np.ndarray] = []
    for candidate in results:
        if not any(candidate.shape == seen.shape and np.array_equal(candidate, seen) for seen in unique):
            unique.append(candidate)
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the only equal-factor sprite for the marker-defined empty box and stamp its resampled image into it.

    Input structure:
    - The most common color is the background.
    - Four isolated marker pixels of one color give unique top, bottom, left, and right extrema.
    - Elsewhere, several separated multicolor sprites sit outside the empty rectangle enclosed by those markers.

    Rule:
    - Use the marker extrema to recover the empty target rectangle one cell inside them.
    - Among the outside sprites, find the unique one whose bounding-box size matches the target size by the same integer factor on both axes, allowing one 90-degree rotation.
    - If the sprite is smaller, enlarge it by per-pixel integer replication; if it is larger, shrink it by merging uniform k-by-k color blocks; then stamp the resampled result exactly into the target rectangle.

    Key insight:
    - The target is implicit, and the correct source is identified by a shared two-axis integer ratio, not by whether it currently looks larger or smaller.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker, target_bbox = _find_marker_and_target(g, bg)
    r0, c0, r1, c1 = target_bbox
    target_shape = (r1 - r0 + 1, c1 - c0 + 1)

    sprite_mask = (g != bg) & (g != marker)
    components = components_mask(sprite_mask, connectivity=4)
    assert components, "expected at least one sprite outside the marker region"

    matches: list[np.ndarray] = []
    for coords in components:
        patch = _component_patch(g, coords, bg)
        fitted = _fit_results(patch, target_shape)
        if fitted:
            assert len(fitted) == 1, "matching sprite must have a unique resampling"
            matches.append(fitted[0])

    assert len(matches) == 1, "expected exactly one sprite with a shared integer ratio on both axes"

    out = g.copy()
    paste(out, matches[0], r0, c0)
    return out


GEN_RETRIES = 400
PLACEMENT_PADDING = 1
TARGET_PADDING = 2
GRID_MIN = 22
GRID_MAX = 30
DISTRACTOR_COUNT_RANGE = (2, 4)
DISTRACTOR_DIM_MIN = 2
DISTRACTOR_DIM_MAX = 12
PALETTE = list(range(10))


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    out = [(r, c)]
    vertical_first = random.random() < 0.5
    if vertical_first:
        while r != tr:
            r += 1 if tr > r else -1
            out.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            out.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            out.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            out.append((r, c))
    return out


def _grow_mask(mask: np.ndarray, target_area: int) -> np.ndarray:
    filled = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    while len(filled) < target_area:
        frontier = set()
        for r, c in filled:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and (nr, nc) not in filled:
                    frontier.add((nr, nc))
        if not frontier:
            break
        filled.add(random.choice(list(frontier)))
    out = np.zeros_like(mask, dtype=bool)
    for r, c in filled:
        out[r, c] = True
    return out


def _make_logical_sprite(h: int, w: int, bg: int, sprite_colors: list[int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = (random.randrange(h), random.randrange(w))
    edge_points = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    for point in [center, *edge_points]:
        mask[point] = True
    for point in edge_points:
        for r, c in _path_between(center, point):
            mask[r, c] = True

    min_fill = max(int(np.ceil(0.55 * h * w)), int(mask.sum()))
    max_fill = max(min_fill, int(np.floor(0.82 * h * w)))
    target_fill = random.randint(min_fill, max_fill)
    mask = _grow_mask(mask, target_fill)

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = min(len(coords), random.randint(2, min(3, len(sprite_colors))))
    seeds = random.sample(coords, color_count)
    colors = random.sample(sprite_colors, color_count)

    sprite = np.full((h, w), bg, dtype=int)
    for color, (r, c) in zip(colors, seeds, strict=False):
        sprite[r, c] = int(color)
    for r, c in coords:
        best = min(
            range(color_count),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        sprite[r, c] = int(colors[best])
    return sprite


def _is_connected_mask(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _make_rot180_sprite(h: int, w: int, bg: int, sprite_colors: list[int]) -> np.ndarray:
    base = _make_logical_sprite(h, w, bg, sprite_colors)
    mask = (base != bg) | (np.rot90(base, 2) != bg)

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        raise AssertionError("rot180 sprite must be nonempty")
    p = random.choice(coords)
    q = (h - 1 - p[0], w - 1 - p[1])
    for r, c in _path_between(p, q):
        mask[r, c] = True
    mask |= np.rot90(mask, 2)

    assert _is_connected_mask(mask), "rot180 sprite must stay connected"
    color = random.choice(sprite_colors)
    sprite = np.full((h, w), bg, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _candidate_spec(bg: int, sprite_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    branch = random.choice(["enlarge", "shrink"])
    rot_k = 0 if random.random() < 0.5 else random.choice([1, 3])
    scale = random.randint(2, 3)

    if branch == "enlarge":
        while True:
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            if h != w:
                break
        logical = (
            _make_rot180_sprite(h, w, bg, sprite_colors)
            if rot_k
            else _make_logical_sprite(h, w, bg, sprite_colors)
        )
        source = np.rot90(logical, rot_k)
        target = scale_sprite(logical, scale)
    else:
        while True:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
            if h != w and max(h, w) * scale <= 12:
                break
        logical = (
            _make_rot180_sprite(h, w, bg, sprite_colors)
            if rot_k
            else _make_logical_sprite(h, w, bg, sprite_colors)
        )
        source = np.rot90(scale_sprite(logical, scale), rot_k)
        target = logical

    fitted = _fit_results(source, target.shape)
    assert len(fitted) == 1 and np.array_equal(fitted[0], target)
    return source, target


def _dimension_profile(shape: tuple[int, int], target_shape: tuple[int, int]) -> tuple[bool, bool, bool]:
    h, w = map(int, shape)
    target_h, target_w = map(int, target_shape)
    valid = False
    unequal = False
    one_axis = False

    orientations = {(h, w), (w, h)}
    for oh, ow in orientations:
        up_r = target_h % oh == 0
        up_c = target_w % ow == 0
        if up_r and up_c:
            valid |= target_h // oh == target_w // ow
            unequal |= target_h // oh != target_w // ow
        elif up_r ^ up_c:
            one_axis = True

        down_r = oh % target_h == 0
        down_c = ow % target_w == 0
        if down_r and down_c:
            valid |= oh // target_h == ow // target_w
            unequal |= oh // target_h != ow // target_w
        elif down_r ^ down_c:
            one_axis = True

    return valid, unequal, one_axis


def _sample_distractor_dims(target_shape: tuple[int, int], mode: str) -> tuple[int, int] | None:
    for _ in range(300):
        h = random.randint(DISTRACTOR_DIM_MIN, DISTRACTOR_DIM_MAX)
        w = random.randint(DISTRACTOR_DIM_MIN, DISTRACTOR_DIM_MAX)
        valid, unequal, one_axis = _dimension_profile((h, w), target_shape)
        if valid:
            continue
        if mode == "unequal" and unequal:
            return h, w
        if mode == "one_axis" and one_axis:
            return h, w
    return None


def _make_distractor_patch(h: int, w: int, bg: int, sprite_colors: list[int]) -> np.ndarray:
    factors = [
        k
        for k in (2, 3)
        if h % k == 0 and w % k == 0 and h // k >= 2 and w // k >= 2
    ]
    if factors and random.random() < 0.45:
        k = random.choice(factors)
        logical = _make_logical_sprite(h // k, w // k, bg, sprite_colors)
        return scale_sprite(logical, k)
    return _make_logical_sprite(h, w, bg, sprite_colors)


def _reserve_box(reserve: np.ndarray, h: int, w: int, padding: int) -> tuple[int, int] | None:
    blocker = np.ones((h, w), dtype=int)
    try:
        r0, c0 = random_free_location(reserve, blocker, transparent=0, padding=padding)
    except ValueError:
        return None
    paste(reserve, blocker, r0, c0)
    return int(r0), int(c0)


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    padding: int,
) -> tuple[int, int] | None:
    placed = _reserve_box(reserve, patch.shape[0], patch.shape[1], padding)
    if placed is None:
        return None
    paste(grid, patch, placed[0], placed[1], transparent=bg)
    return placed


def _mark_target(grid: np.ndarray, marker: int, target_r0: int, target_c0: int, target_h: int, target_w: int) -> None:
    top = (target_r0 - 1, target_c0 + random.randrange(target_w))
    bottom = (target_r0 + target_h, target_c0 + random.randrange(target_w))
    left = (target_r0 + random.randrange(target_h), target_c0 - 1)
    right = (target_r0 + random.randrange(target_h), target_c0 + target_w)
    for r, c in (top, bottom, left, right):
        grid[r, c] = marker


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(PALETTE)
    marker = random.choice([c for c in PALETTE if c != bg])
    sprite_colors = [c for c in PALETTE if c not in (bg, marker)]

    candidate_source, target_patch = _candidate_spec(bg, sprite_colors)
    target_h, target_w = map(int, target_patch.shape)

    distractor_modes = ["unequal", "one_axis"]
    extra = random.randint(*DISTRACTOR_COUNT_RANGE) - len(distractor_modes)
    for _ in range(extra):
        distractor_modes.append(random.choice(["unequal", "one_axis"]))

    distractors: list[np.ndarray] = []
    for mode in distractor_modes:
        dims = _sample_distractor_dims((target_h, target_w), mode)
        if dims is None:
            return None
        distractors.append(_make_distractor_patch(dims[0], dims[1], bg, sprite_colors))

    for patch in [candidate_source, *distractors]:
        fitted = _fit_results(patch, (target_h, target_w))
        if patch is candidate_source:
            if len(fitted) != 1:
                return None
        elif fitted:
            return None

    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), bg, dtype=int)
    reserve = np.zeros((height, width), dtype=int)

    target_footprint = _reserve_box(reserve, target_h + 2, target_w + 2, TARGET_PADDING)
    if target_footprint is None:
        return None
    target_r0 = target_footprint[0] + 1
    target_c0 = target_footprint[1] + 1
    _mark_target(grid, marker, target_r0, target_c0, target_h, target_w)

    if _place_patch(grid, reserve, candidate_source, bg, PLACEMENT_PADDING) is None:
        return None

    random.shuffle(distractors)
    for patch in distractors:
        if _place_patch(grid, reserve, patch, bg, PLACEMENT_PADDING) is None:
            return None

    inp = grid
    out = inp.copy()
    paste(out, target_patch, target_r0, target_c0)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None

    found_marker, bbox = _find_marker_and_target(inp, bg)
    if found_marker != marker or bbox != (
        target_r0,
        target_c0,
        target_r0 + target_h - 1,
        target_c0 + target_w - 1,
    ):
        return None

    sprite_mask = (inp != bg) & (inp != marker)
    components = components_mask(sprite_mask, connectivity=4)
    matched = 0
    for coords in components:
        patch = _component_patch(inp, coords, bg)
        if _fit_results(patch, target_patch.shape):
            matched += 1
    if matched != 1:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid integer-scale marker puzzle input")
