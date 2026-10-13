from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
if _ARC_SPEC is None or _ARC_SPEC.loader is None:
    raise ImportError(f"cannot load ARC helpers from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste

BG = Color.BLACK
SEPARATOR = Color.GREY
SHADOW = Color.RED
PALETTE = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAX_GENERATION_ATTEMPTS = 600
_GEN_COUNTER = 0
_OFFSETS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _ri(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == SEPARATOR, axis=1))[0]
    assert rows.size == 1, "expected exactly one full-width grey separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into top and bottom regions"
    assert not np.any(grid[np.arange(grid.shape[0]) != sep] == SEPARATOR), "grey must appear only in the separator row"
    return sep


def _extract_sprite(top: np.ndarray) -> np.ndarray:
    comps = find_components(top, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) == 1, "expected one multicolor sprite above the separator"
    box = comps[0]["bbox"]
    sprite = crop_bbox(top, box)
    colors = {int(v) for v in np.unique(sprite)} - {BG}
    assert len(colors) == 2, "expected exactly two sprite colors"
    assert 3 <= int((sprite != BG).sum()) <= 7, "expected a small sprite"
    return sprite


def _extract_anchors(bottom: np.ndarray) -> list[tuple[int, int, int]]:
    pts = np.argwhere(bottom != BG)
    anchors = [(int(r), int(c), int(bottom[r, c])) for r, c in pts]
    assert 2 <= len(anchors) <= 3, "expected 2 or 3 anchor pixels"
    assert all(color not in {SEPARATOR, SHADOW} for _r, _c, color in anchors), "anchors must use sprite colors"
    return sorted(anchors)


def _enumerate_matches(
    sprite: np.ndarray, canvas_shape: tuple[int, int], anchors: list[tuple[int, int, int]]
) -> list[tuple[int, int, int, np.ndarray]]:
    rows, cols = map(int, canvas_shape)
    matches: list[tuple[int, int, int, np.ndarray]] = []
    for rotation in range(4):
        rotated = np.rot90(sprite, rotation)
        height, width = map(int, rotated.shape)
        for top_r in range(rows - height + 1):
            for top_c in range(cols - width + 1):
                ok = True
                for anchor_r, anchor_c, color in anchors:
                    local_r = anchor_r - top_r
                    local_c = anchor_c - top_c
                    if not (0 <= local_r < height and 0 <= local_c < width):
                        ok = False
                        break
                    if int(rotated[local_r, local_c]) != int(color):
                        ok = False
                        break
                if ok:
                    matches.append((rotation, top_r, top_c, rotated))
    return matches


def _render_tile(rotated: np.ndarray, top_r: int, top_c: int, shape: tuple[int, int]) -> np.ndarray:
    tile = np.full(shape, BG, dtype=int)
    paste(tile, rotated, top_r, top_c, transparent=BG)
    return tile


def _add_shadow(tile: np.ndarray) -> np.ndarray:
    base = np.asarray(tile, dtype=int)
    out = base.copy()
    rows, cols = base.shape
    for r, c in np.argwhere(base != BG):
        rr = int(r)
        cc = int(c)
        if rr == 0:
            continue
        sr = rr - 1
        sc = (cc - 1 + cols) % cols
        if int(base[sr, sc]) == BG:
            out[sr, sc] = SHADOW
    return out


def _shadow_features(tile: np.ndarray) -> dict[str, int | bool]:
    rows, cols = tile.shape
    touch_top = False
    wrap_sources = 0
    wrap_added = 0
    collisions = 0
    added_targets: set[tuple[int, int]] = set()
    for r, c in np.argwhere(tile != BG):
        rr = int(r)
        cc = int(c)
        if rr == 0:
            touch_top = True
            continue
        sr = rr - 1
        sc = (cc - 1 + cols) % cols
        if cc == 0:
            wrap_sources += 1
        if int(tile[sr, sc]) != BG:
            collisions += 1
            continue
        if cc == 0:
            wrap_added += 1
        added_targets.add((sr, sc))
    return {
        "touch_top": touch_top,
        "wrap_sources": wrap_sources,
        "wrap_added": wrap_added,
        "collisions": collisions,
        "shadow_pixels": len(added_targets),
    }


def _expected_from_match(rotated: np.ndarray, top_r: int, top_c: int, shape: tuple[int, int]) -> np.ndarray:
    tile = _render_tile(rotated, top_r, top_c, shape)
    shadowed = _add_shadow(tile)
    return np.tile(shadowed, (2, 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the upper sprite to the unique anchor-consistent placement, add a wrapped up-left shadow, and tile the result 2x2.

    Input structure:
    - A black background grid is split by one full-width grey separator row into equal-height top and bottom regions.
    - The top region contains one small 4-connected multicolor sprite with exactly two foreground colors.
    - The bottom region is a same-sized black canvas containing 2 or 3 colored anchor pixels, and red does not appear in the input.

    Rule:
    - Crop the upper sprite, ignore the grey separator, and treat the lower region as the tile canvas.
    - Try all 0/90/180/270 rotations and all in-bounds translations of the sprite; choose the unique placement whose colored pixels cover every anchor with the same colors.
    - Stamp that rotated sprite onto the lower canvas over black, then for each non-black tile pixel add a red pixel one row up and one column left with column wrapping, skipping any target cell already occupied by the stamped sprite.
    - Repeat the shadowed tile twice horizontally and twice vertically, and return that 2x2 tiling.

    Key insight:
    - The sparse lower anchors only determine the sprite's rotation and translation; the full aligned sprite must be reconstructed first before its shadow pattern and tiled output can be computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert not np.any(g == SHADOW), "red must be absent from the input"
    assert int(get_bg(g)) == BG, "expected black to be the dominant background"

    sep = _separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape, "top and bottom regions must have the same size"

    sprite = _extract_sprite(top)
    anchors = _extract_anchors(bottom)
    matches = _enumerate_matches(sprite, bottom.shape, anchors)
    assert len(matches) == 1, "expected a unique rotation and translation"

    _rotation, top_r, top_c, rotated = matches[0]
    return _expected_from_match(rotated, top_r, top_c, bottom.shape)


def _generate_shape_coords(cell_count: int, rng: np.random.Generator) -> set[tuple[int, int]] | None:
    for _ in range(100):
        coords = {(0, 0)}
        while len(coords) < cell_count:
            frontier = []
            seen = set()
            for r, c in coords:
                for dr, dc in _OFFSETS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coords or (nr, nc) in seen:
                        continue
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
            if not frontier:
                break
            coords.add(frontier[_ri(rng, 0, len(frontier) - 1)])
        min_r = min(r for r, _c in coords)
        min_c = min(c for _r, c in coords)
        norm = {(r - min_r, c - min_c) for r, c in coords}
        max_r = max(r for r, _c in norm)
        max_c = max(c for _r, c in norm)
        if max_r < 1 or max_c < 1:
            continue
        return norm
    return None


def _rotations_are_distinct(sprite: np.ndarray) -> bool:
    views = {np.rot90(sprite, k).tobytes() + str(np.rot90(sprite, k).shape).encode() for k in range(4)}
    return len(views) == 4


def _generate_sprite(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(120):
        cell_count = _ri(rng, 3, 7)
        coords = _generate_shape_coords(cell_count, rng)
        if coords is None:
            continue
        colors = rng.choice(np.array(PALETTE, dtype=int), size=2, replace=False)
        patch = np.full(
            (
                max(r for r, _c in coords) + 1,
                max(c for _r, c in coords) + 1,
            ),
            BG,
            dtype=int,
        )
        coord_list = sorted(coords)
        second_count = _ri(rng, 1, len(coord_list) - 1)
        second_mask = set(int(i) for i in rng.choice(len(coord_list), size=second_count, replace=False))
        for idx, (r, c) in enumerate(coord_list):
            patch[r, c] = int(colors[1] if idx in second_mask else colors[0])
        if len({int(v) for v in np.unique(patch)} - {BG}) != 2:
            continue
        if not _rotations_are_distinct(patch):
            continue
        return patch
    return None


def _choose_canvas_shape(
    sprite: np.ndarray, rotated: np.ndarray, placement_mode: str, rng: np.random.Generator
) -> tuple[int, int] | None:
    height = max(int(sprite.shape[0]), int(rotated.shape[0]))
    width = max(int(sprite.shape[1]), int(rotated.shape[1]))
    min_extra_r = 1 if placement_mode in {"left", "interior"} else 0
    min_extra_c = 1 if placement_mode in {"top", "interior"} else 0
    max_extra_r = min(14 - height, 5)
    max_extra_c = min(15 - width, 6)
    if max_extra_r < min_extra_r or max_extra_c < min_extra_c:
        return None
    rows = height + _ri(rng, min_extra_r, max_extra_r)
    cols = width + _ri(rng, min_extra_c, max_extra_c)
    return rows, cols


def _candidate_positions(
    canvas_shape: tuple[int, int], rotated: np.ndarray, placement_mode: str
) -> list[tuple[int, int]]:
    rows, cols = map(int, canvas_shape)
    height, width = map(int, rotated.shape)
    positions = [(r, c) for r in range(rows - height + 1) for c in range(cols - width + 1)]
    if placement_mode == "corner":
        filtered = [(r, c) for r, c in positions if r == 0 and c == 0]
    elif placement_mode == "top":
        filtered = [(r, c) for r, c in positions if r == 0 and c > 0]
    elif placement_mode == "left":
        filtered = [(r, c) for r, c in positions if c == 0 and r > 0]
    else:
        filtered = [(r, c) for r, c in positions if r > 0 and c > 0]
    return filtered or positions


def _tile_positions(tile: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted((int(r), int(c), int(tile[r, c])) for r, c in np.argwhere(tile != BG))


def _target_subset_options(
    sprite: np.ndarray,
    anchors: list[tuple[int, int, int]],
    canvas_shape: tuple[int, int],
    target: tuple[int, int, int],
    require_both_colors: bool,
) -> bool:
    if require_both_colors and len({color for _r, _c, color in anchors}) < 2:
        return False
    matches = _enumerate_matches(sprite, canvas_shape, anchors)
    if len(matches) != 1:
        return False
    rotation, top_r, top_c, _rotated = matches[0]
    return (rotation, top_r, top_c) == target


def _choose_anchors(
    sprite: np.ndarray,
    positions: list[tuple[int, int, int]],
    canvas_shape: tuple[int, int],
    target: tuple[int, int, int],
    preferred_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]] | None:
    for count in (preferred_count, 5 - preferred_count):
        if count <= 0 or count > len(positions):
            continue
        informative: list[tuple[tuple[int, int], tuple[tuple[int, int, int], ...]]] = []
        fallback: list[tuple[tuple[int, int], tuple[tuple[int, int, int], ...]]] = []
        for subset in itertools.combinations(positions, count):
            colors = {color for _r, _c, color in subset}
            if not _target_subset_options(sprite, list(subset), canvas_shape, target, require_both_colors=False):
                continue
            spread = 0
            for (r0, c0, _color0), (r1, c1, _color1) in itertools.combinations(subset, 2):
                spread += abs(r0 - r1) + abs(c0 - c1)
            key = (spread, len(colors))
            if len(colors) == 2:
                informative.append((key, subset))
            else:
                fallback.append((key, subset))
        pool = informative or fallback
        if pool:
            best_key = max(key for key, _subset in pool)
            best_subsets = [list(subset) for key, subset in pool if key == best_key]
            return best_subsets[_ri(rng, 0, len(best_subsets) - 1)]
    return None


def _top_left_for_sprite(
    canvas_shape: tuple[int, int], sprite_shape: tuple[int, int], rng: np.random.Generator
) -> tuple[int, int]:
    rows, cols = map(int, canvas_shape)
    height, width = map(int, sprite_shape)
    return _ri(rng, 0, rows - height), _ri(rng, 0, cols - width)


def _build_input(
    sprite: np.ndarray,
    top_left: tuple[int, int],
    anchors: list[tuple[int, int, int]],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    rows, cols = map(int, canvas_shape)
    out = np.full((rows * 2 + 1, cols), BG, dtype=int)
    out[rows, :] = SEPARATOR
    paste(out[:rows, :], sprite, top_left[0], top_left[1], transparent=BG)
    for r, c, color in anchors:
        out[rows + 1 + r, c] = int(color)
    return out


def _validate_instance(
    grid: np.ndarray,
    target: tuple[int, int, int],
    expected: np.ndarray,
    tile_shape: tuple[int, int],
) -> bool:
    sep = _separator_row(grid)
    sprite = _extract_sprite(grid[:sep])
    anchors = _extract_anchors(grid[sep + 1 :])
    matches = _enumerate_matches(sprite, tile_shape, anchors)
    if len(matches) != 1:
        return False
    if matches[0][:3] != target:
        return False
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_attempt(
    rng: np.random.Generator, target_rotation: int, anchor_count: int, placement_mode: str
) -> np.ndarray | None:
    sprite = _generate_sprite(rng)
    if sprite is None:
        return None

    rotated = np.rot90(sprite, target_rotation)
    canvas_shape = _choose_canvas_shape(sprite, rotated, placement_mode, rng)
    if canvas_shape is None:
        return None

    positions = _candidate_positions(canvas_shape, rotated, placement_mode)
    if not positions:
        return None
    target_r, target_c = positions[_ri(rng, 0, len(positions) - 1)]
    tile = _render_tile(rotated, target_r, target_c, canvas_shape)
    features = _shadow_features(tile)
    if int(features["shadow_pixels"]) == 0:
        return None
    if placement_mode in {"top", "corner"} and not bool(features["touch_top"]):
        return None
    if placement_mode in {"left", "corner"} and int(features["wrap_added"]) == 0:
        return None

    anchors = _choose_anchors(
        sprite,
        _tile_positions(tile),
        canvas_shape,
        (target_rotation, target_r, target_c),
        anchor_count,
        rng,
    )
    if anchors is None:
        return None

    top_left = _top_left_for_sprite(canvas_shape, sprite.shape, rng)
    grid = _build_input(sprite, top_left, anchors, canvas_shape)
    expected = _expected_from_match(rotated, target_r, target_c, canvas_shape)
    if not _validate_instance(grid, (target_rotation, target_r, target_c), expected, canvas_shape):
        return None
    if grid.shape[0] > 30 or grid.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    branch_index = _GEN_COUNTER
    _GEN_COUNTER += 1

    target_rotation = branch_index % 4
    anchor_count = 2 if (branch_index // 4) % 2 == 0 else 3
    placement_mode = ("interior", "top", "left", "corner")[(branch_index // 8) % 4]

    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng, target_rotation, anchor_count, placement_mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
