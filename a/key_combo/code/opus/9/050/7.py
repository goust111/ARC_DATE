from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, paste, scale_sprite


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_RNG = np.random.default_rng()
_SPRITE_COLORS = (BLUE, GREEN, YELLOW)
_RECT_PATTERNS = (
    ("wide", "tall", "square"),
    ("square", "wide", "tall"),
    ("tall", "square", "wide"),
    ("wide", "wide", "tall"),
    ("tall", "wide", "wide"),
    ("wide", "square", "square"),
    ("tall", "square", "square"),
)
_BASE_SPRITES = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
)


def _sprite_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SPRITES:
        for rot in range(4):
            cur = np.rot90(base, rot)
            for variant in (cur, np.fliplr(cur)):
                mask = variant.astype(bool)
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                trimmed = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].astype(int)
                if not 3 <= int(trimmed.sum()) <= 5:
                    continue
                sig = trimmed.astype(np.uint8).tobytes() + str(trimmed.shape).encode()
                if sig in seen:
                    continue
                seen.add(sig)
                library.append(trimmed)
    return library


_SPRITE_LIBRARY = _sprite_library()


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    mask = (grid == GREY) | (grid == RED)
    comps = find_components(
        mask.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[1],
        return_format="dict",
    )
    rects: list[dict] = []
    for comp in comps:
        box = tuple(map(int, bbox_coords(comp["coords"])))
        patch = crop_bbox(grid, box)
        h, w = patch.shape
        if not (3 <= h <= 5 and 4 <= w <= 6):
            continue
        if not np.all((patch == GREY) | (patch == RED)):
            continue
        if np.count_nonzero((patch == GREY) | (patch == RED)) != h * w:
            continue
        markers = np.argwhere(patch == RED)
        if any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in markers):
            continue
        if h > w:
            orientation = "tall"
        elif w > h:
            orientation = "wide"
        else:
            orientation = "square"
        rects.append(
            {
                "bbox": box,
                "patch": patch,
                "orientation": orientation,
                "markers": [(int(r), int(c)) for r, c in markers],
            }
        )
    rects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return rects


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=_SPRITE_COLORS,
        min_area=3,
        return_format="dict",
    ):
        area = int(comp["area"])
        if area > 5:
            continue
        crop = crop_bbox(grid, tuple(map(int, comp["bbox"])))
        color = int(comp["color"])
        if np.count_nonzero(crop == color) != area:
            continue
        sprites.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "crop": crop,
                "color": color,
                "area": area,
            }
        )
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return sprites


def _expand_rectangle(patch: np.ndarray) -> tuple[np.ndarray, int]:
    out = patch.copy()
    markers = np.argwhere(patch == RED)
    h, w = patch.shape
    if h > w:
        rows = sorted({int(r) for r, _ in markers})
        for r in rows:
            out[r, :] = RED
        return out, len(rows)
    cols = sorted({int(c) for _, c in markers})
    for c in cols:
        out[:, c] = RED
    return out, len(cols)


def _stack_scaled(crops: list[np.ndarray], factors: list[int]) -> np.ndarray:
    scaled = [scale_sprite(crop, factor) for crop, factor in zip(crops, factors)]
    out_h = sum(tile.shape[0] for tile in scaled) + len(scaled) - 1
    out_w = max(tile.shape[1] for tile in scaled)
    out = np.zeros((out_h, out_w), dtype=int)
    r0 = 0
    for tile in scaled:
        paste(out, tile, r0, 0)
        r0 += tile.shape[0] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode divider counts from grey rectangles and enlarge the paired sprites by the resulting scale factors.

    Input structure:
    - The background is black.
    - The left side contains three solid grey rectangles, each optionally containing 0, 1, or 2 interior red marker pixels.
    - The right side contains three small monochrome 4-connected sprites in blue, green, and yellow, ordered vertically.

    Rule:
    - Inside each rectangle, every red marker expands along the rectangle's short axis: tall rectangles use full rows, while wide or square rectangles use full columns.
    - The number of resulting divider lines is d, so the paired sprite uses scale factor d+1.
    - Pair the k-th rectangle and k-th sprite by top-to-bottom order, crop each sprite to its tight bounding box, scale it by its factor, then stack the scaled sprites vertically with one black row between them.

    Key insight:
    - The red marks inside the grey rectangles are not output directly; after aspect-ratio-based divider expansion, their line count becomes a numeric code that controls the size of separate sprites elsewhere in the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _parse_rectangles(g)
    sprites = _parse_sprites(g)
    assert len(rects) == 3
    assert len(sprites) == 3
    assert len({sprite["color"] for sprite in sprites}) == 3
    assert max(rect["bbox"][3] for rect in rects) < min(sprite["bbox"][1] for sprite in sprites)

    factors = []
    crops = []
    for rect, sprite in zip(rects, sprites):
        _expanded, divider_count = _expand_rectangle(rect["patch"])
        factors.append(divider_count + 1)
        crops.append(sprite["crop"])

    assert sorted(factors) == [1, 2, 3]
    return _stack_scaled(crops, factors)


def _rect_dims(kind: str) -> tuple[int, int]:
    if kind == "wide":
        options = [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    elif kind == "tall":
        options = [(4, 3), (5, 3), (5, 4)]
    else:
        options = [(4, 4), (5, 5)]
    return options[int(_RNG.integers(len(options)))]


def _sample_markers(h: int, w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    rows = list(range(1, h - 1))
    cols = list(range(1, w - 1))
    if h > w:
        chosen_rows = _RNG.choice(rows, size=count, replace=False)
        chosen_cols = _RNG.choice(cols, size=count, replace=True)
    else:
        chosen_rows = _RNG.choice(rows, size=count, replace=True)
        chosen_cols = _RNG.choice(cols, size=count, replace=False)
    return [(int(r), int(c)) for r, c in zip(chosen_rows, chosen_cols)]


def _choose_sprite_masks() -> list[np.ndarray]:
    picks = _RNG.choice(len(_SPRITE_LIBRARY), size=3, replace=False)
    masks = [_SPRITE_LIBRARY[int(i)].copy() for i in picks]
    masks.sort(key=lambda arr: (arr.shape[0] * arr.shape[1], arr.shape[0], arr.shape[1], arr.sum()))
    return masks


def _build_expected(sprite_specs: list[dict], factors: list[int]) -> np.ndarray:
    crops = [np.where(spec["mask"] == 1, spec["color"], BG) for spec in sprite_specs]
    return _stack_scaled(crops, factors)


def _line_segments_ok(expanded: np.ndarray, divider_count: int, orientation: str) -> bool:
    if orientation == "tall":
        divider_rows = [r for r in range(expanded.shape[0]) if np.all(expanded[r, :] == RED)]
        return len(divider_rows) == divider_count
    divider_cols = [c for c in range(expanded.shape[1]) if np.all(expanded[:, c] == RED)]
    return len(divider_cols) == divider_count


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    rects = _parse_rectangles(inp)
    sprites = _parse_sprites(inp)
    if len(rects) != 3 or len(sprites) != 3:
        return False
    if len({sprite["color"] for sprite in sprites}) != 3:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if max(rect["bbox"][3] for rect in rects) >= min(sprite["bbox"][1] for sprite in sprites):
        return False

    marker_counts = []
    factors = []
    orientations = set()
    for rect in rects:
        expanded, divider_count = _expand_rectangle(rect["patch"])
        marker_count = len(rect["markers"])
        marker_counts.append(marker_count)
        factors.append(divider_count + 1)
        orientations.add(rect["orientation"])
        if divider_count != marker_count:
            return False
        if not _line_segments_ok(expanded, divider_count, rect["orientation"]):
            return False

    if sorted(marker_counts) != [0, 1, 2]:
        return False
    if sorted(factors) != [1, 2, 3]:
        return False
    if len(orientations) < 2:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    sprite_crops = [sprite["crop"] for sprite in sprites]
    intended = _stack_scaled(sprite_crops, factors)

    for perm in itertools.permutations(range(3)):
        if perm != (0, 1, 2):
            alt_factors = [factors[i] for i in perm]
            if np.array_equal(_stack_scaled(sprite_crops, alt_factors), intended):
                return False
            alt_crops = [sprite_crops[i] for i in perm]
            if np.array_equal(_stack_scaled(alt_crops, factors), intended):
                return False

    return True


def _sample_count_order() -> list[int]:
    patterns = (
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
        [2, 1, 0],
    )
    return list(patterns[int(_RNG.integers(len(patterns)))])


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    counts = _sample_count_order()
    kinds = list(_RECT_PATTERNS[int(_RNG.integers(len(_RECT_PATTERNS)))])
    colors = list(_RNG.permutation(np.array(_SPRITE_COLORS, dtype=int)))
    masks = _choose_sprite_masks()

    rect_specs = []
    sprite_specs = []

    rect_r = int(_RNG.integers(1, 3))
    sprite_r = int(_RNG.integers(1, 4))
    for i in range(3):
        rh, rw = _rect_dims(kinds[i])
        rect_specs.append({"r0": rect_r, "h": rh, "w": rw, "count": counts[i]})
        rect_r += rh + int(_RNG.integers(1, 4))

        mask = masks[i]
        sh, sw = mask.shape
        sprite_specs.append({"r0": sprite_r, "h": sh, "w": sw, "mask": mask, "color": int(colors[i])})
        sprite_r += sh + int(_RNG.integers(2, 5))

    h = max(rect_r, sprite_r) + int(_RNG.integers(0, 2))
    if h > 28:
        return None

    w = int(_RNG.integers(16, 23))
    split = w // 2
    grid = np.zeros((h, w), dtype=int)

    for spec in rect_specs:
        left_max = split - spec["w"] - 1
        if left_max < 1:
            return None
        c0 = int(_RNG.integers(1, left_max + 1))
        spec["c0"] = c0
        grid[spec["r0"] : spec["r0"] + spec["h"], c0 : c0 + spec["w"]] = GREY
        for rr, cc in _sample_markers(spec["h"], spec["w"], spec["count"]):
            grid[spec["r0"] + rr, c0 + cc] = RED

    for spec in sprite_specs:
        right_min = split + 1
        right_max = w - spec["w"] - 1
        if right_min > right_max:
            return None
        c0 = int(_RNG.integers(right_min, right_max + 1))
        spec["c0"] = c0
        patch = np.where(spec["mask"] == 1, spec["color"], BG)
        region = grid[spec["r0"] : spec["r0"] + spec["h"], c0 : c0 + spec["w"]]
        if region.shape != patch.shape or np.any(region != BG):
            return None
        paste(grid, patch, spec["r0"], c0, transparent=BG)

    factors = [count + 1 for count in counts]
    expected = _build_expected(sprite_specs, factors)
    if not _validate_semantics(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input for puzzle 7")
