from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg, label_mask, paste


BG = Color.BLACK
BAR = Color.GREY
SPRITE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
RINGS = (1, 2, 3)

GRID_ROWS = (18, 25)
GRID_COLS = (16, 25)
MAX_ATTEMPTS = 400


def _hole_count(mask: np.ndarray) -> int:
    padded_bg = np.pad(~mask.astype(bool), 1, constant_values=True)
    labels, n_labels = label_mask(padded_bg, connectivity=4)
    outside = int(labels[0, 0])
    return sum(1 for k in range(1, n_labels + 1) if k != outside)


def _component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(comp["color"])
    return patch


def _sprite_info(comp: dict, bg: int) -> dict:
    patch = _component_patch(np.empty((1, 1), dtype=int), comp, bg)
    mask = patch != int(bg)
    holes = _hole_count(mask)
    return {
        "color": int(comp["color"]),
        "bbox": tuple(map(int, comp["bbox"])),
        "coords": comp["coords"],
        "patch": patch,
        "mask": mask,
        "holes": int(holes),
    }


def _bar_info(comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == 1, "gray objects must be horizontal 1-cell bars"
    assert int(comp["area"]) == w, "bars must be solid"
    return {
        "bbox": (r0, c0, r1, c1),
        "row": r0,
        "length": w,
        "rings": (w - 1) // 2,
    }


def _aligned_left_column(bar: dict, sprite_width: int) -> int:
    _r0, c0, _r1, c1 = map(int, bar["bbox"])
    return (c0 + c1 - (int(sprite_width) - 1)) // 2


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    bars: list[dict] = []
    sprites: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        if color == int(BAR):
            bars.append(_bar_info(comp))
        else:
            sprites.append(_sprite_info(comp, BG))
    return bars, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each holed sprite to the bar whose diamond-envelope ring count matches its hole count.

    Input structure:
    - Background is black. Three gray horizontal one-cell-thick bars sit in lower rows, each with a different length.
    - Three other foreground objects are monochrome 4-connected sprites in distinct colors.
    - Each sprite's bounding box contains a different number of enclosed 4-connected black holes.

    Rule:
    - For every gray bar of length L, compute the invisible diamond-envelope ring count R = floor((L - 1) / 2).
    - For every colored sprite, count its enclosed black holes H inside its own minimal bounding box.
    - Match sprites to bars by H = R, erase the sprites from their original positions, and redraw each sprite directly above its matched bar.
    - The sprite keeps its exact shape and color; its bottom edge touches the row immediately above the bar, and its bbox is centered on the bar midpoint with left bias when exact centering is impossible.

    Key insight:
    - A one-dimensional bar length and a two-dimensional hole count become comparable only after converting the bar through the diamond-envelope formula, so the relocation key is an intermediate numeric invariant rather than a direct visual attribute.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bars, sprites = _parse_scene(g)
    assert len(bars) == 3, "expected exactly three gray bars"
    assert len(sprites) == 3, "expected exactly three colored sprites"

    bar_by_ring: dict[int, dict] = {}
    for bar in bars:
        rings = int(bar["rings"])
        assert rings in RINGS, "bar ring count must be one of 1,2,3"
        assert rings not in bar_by_ring, "bar ring counts must be unique"
        bar_by_ring[rings] = bar

    sprite_by_holes: dict[int, dict] = {}
    sprite_colors = set()
    for sprite in sprites:
        holes = int(sprite["holes"])
        assert holes in RINGS, "sprite hole count must be one of 1,2,3"
        assert holes not in sprite_by_holes, "sprite hole counts must be unique"
        assert int(sprite["color"]) not in sprite_colors, "sprite colors must be distinct"
        sprite_colors.add(int(sprite["color"]))
        sprite_by_holes[holes] = sprite

    assert set(bar_by_ring) == set(sprite_by_holes) == set(RINGS), "bars and sprites must cover 1,2,3 exactly"

    out = g.copy()
    for sprite in sprites:
        rr, cc = sprite["coords"].T
        out[rr, cc] = int(BG)

    for key in sorted(RINGS):
        bar = bar_by_ring[key]
        sprite = sprite_by_holes[key]
        patch = sprite["patch"]
        h, w = map(int, patch.shape)
        dest_r0 = int(bar["row"]) - h
        dest_c0 = _aligned_left_column(bar, w)
        assert dest_r0 >= 0, "sprite must fit above its matched bar"
        assert 0 <= dest_c0 and dest_c0 + w <= out.shape[1], "sprite must fit horizontally after centering"
        paste(out, patch, dest_r0, dest_c0, transparent=int(BG))

    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    coords = _normalize_coords(coords)
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _base_sprite_mask(holes: int) -> np.ndarray:
    width = 2 * int(holes) + 1
    mask = np.zeros((3, width), dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    mask[1, ::2] = True
    return mask


def _grow_sprite_mask(rng: np.random.Generator, holes: int) -> np.ndarray:
    coords = {tuple(map(int, cell)) for cell in np.argwhere(_base_sprite_mask(holes))}
    base = _coords_to_mask(coords)
    h, w = map(int, base.shape)

    candidates = set()
    for r, c in coords:
        if r == 0:
            candidates.add((r - 1, c))
        if r == h - 1:
            candidates.add((r + 1, c))
        if c == 0:
            candidates.add((r, c - 1))
        if c == w - 1:
            candidates.add((r, c + 1))

    candidate_list = list(candidates)
    rng.shuffle(candidate_list)
    extra_count = int(rng.choice([0, 1, 1, 2]))
    for cell in candidate_list[:extra_count]:
        coords.add(cell)

    mask = _coords_to_mask(coords)
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(0, 2)):
        mask = np.flipud(mask)

    assert len(components_mask(mask, connectivity=4)) == 1
    assert _hole_count(mask) == int(holes)
    return mask


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, int(BG), dtype=int)
    patch[mask] = int(color)
    return patch


def _paint_bar(grid: np.ndarray, row: int, col: int, length: int) -> None:
    grid[int(row), int(col) : int(col) + int(length)] = int(BAR)


def _sample_bar_lengths(rng: np.random.Generator) -> dict[int, int]:
    even_ring = int(rng.choice(np.array(RINGS)))
    lengths = {}
    for ring in RINGS:
        delta = 1 if ring == even_ring else int(rng.integers(0, 2))
        lengths[int(ring)] = 2 * int(ring) + 1 + int(delta)
    return lengths


def _sample_bar_rows(rng: np.random.Generator, height: int, sprite_heights: dict[int, int]) -> dict[int, int] | None:
    for _ in range(120):
        order = list(RINGS)
        rng.shuffle(order)

        top_ring, mid_ring, low_ring = order
        gap_top_mid = max(3, int(sprite_heights[mid_ring]) + 1)
        gap_mid_low = max(3, int(sprite_heights[low_ring]) + 1)

        min_top = max(height // 2, int(sprite_heights[top_ring]))
        max_top = height - 2 - gap_top_mid - gap_mid_low
        if min_top > max_top:
            continue

        top_row = int(rng.integers(min_top, max_top + 1))
        min_mid = top_row + gap_top_mid
        max_mid = height - 2 - gap_mid_low
        if min_mid > max_mid:
            continue

        mid_row = int(rng.integers(min_mid, max_mid + 1))
        min_low = mid_row + gap_mid_low
        max_low = height - 2
        if min_low > max_low:
            continue

        low_row = int(rng.integers(min_low, max_low + 1))
        return {top_ring: top_row, mid_ring: mid_row, low_ring: low_row}
    return None


def _sample_bar_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    sprite_patches: dict[int, np.ndarray],
    lengths: dict[int, int],
    rows: dict[int, int],
) -> dict[int, tuple[int, int]] | None:
    placements: dict[int, tuple[int, int]] = {}
    for ring in rng.permutation(np.array(RINGS)):
        ring = int(ring)
        patch = sprite_patches[ring]
        row = int(rows[ring])
        length = int(lengths[ring])
        sprite_h, sprite_w = map(int, patch.shape)
        shift = (length - sprite_w) // 2
        min_c = max(0, -shift)
        max_c = min(width - length, width - sprite_w - shift)
        if min_c > max_c:
            return None
        col = int(rng.integers(min_c, max_c + 1))
        placements[ring] = (row, col)
        if row - sprite_h < 0:
            return None
    return placements


def _free_rectangles(top_limit: int, width: int, patch_shape: tuple[int, int], occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, patch_shape)
    slots: list[tuple[int, int]] = []
    for r0 in range(0, top_limit - h + 1):
        for c0 in range(0, width - w + 1):
            if occupied[r0 : r0 + h, c0 : c0 + w].any():
                continue
            slots.append((r0, c0))
    return slots


def _place_input_sprites(
    rng: np.random.Generator,
    grid: np.ndarray,
    sprite_patches: dict[int, np.ndarray],
    final_positions: dict[int, tuple[int, int]],
    top_limit: int,
) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros(grid.shape, dtype=bool)
    order = sorted(RINGS, key=lambda ring: sprite_patches[ring].shape[1], reverse=True)
    starts: dict[int, tuple[int, int]] = {}

    for ring in order:
        patch = sprite_patches[ring]
        h, w = map(int, patch.shape)
        if top_limit < h:
            return None
        slots = _free_rectangles(top_limit, grid.shape[1], patch.shape, occupied)
        rng.shuffle(slots)
        chosen = None
        for r0, c0 in slots:
            if (r0, c0) == final_positions[ring]:
                continue
            chosen = (r0, c0)
            break
        if chosen is None:
            return None
        r0, c0 = chosen
        paste(grid, patch, r0, c0, transparent=int(BG))
        occupied[max(0, r0 - 1) : min(grid.shape[0], r0 + h + 1), max(0, c0 - 1) : min(grid.shape[1], c0 + w + 1)] = True
        starts[ring] = chosen
    return starts


def _expected_output_from_specs(
    shape: tuple[int, int],
    sprite_patches: dict[int, np.ndarray],
    lengths: dict[int, int],
    placements: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    out = np.full(shape, int(BG), dtype=int)
    final_positions: dict[int, tuple[int, int]] = {}
    for ring in RINGS:
        row, col = placements[ring]
        length = lengths[ring]
        _paint_bar(out, row, col, length)
    for ring in RINGS:
        row, col = placements[ring]
        patch = sprite_patches[ring]
        h, w = map(int, patch.shape)
        dest_r0 = int(row) - h
        dest_c0 = int(col + (lengths[ring] - w) // 2)
        final_positions[ring] = (dest_r0, dest_c0)
        paste(out, patch, dest_r0, dest_c0, transparent=int(BG))
    return out, final_positions


def _verify_generated_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False

    bars_in, sprites_in = _parse_scene(inp)
    if len(bars_in) != 3 or len(sprites_in) != 3:
        return False
    if sorted(bar["rings"] for bar in bars_in) != [1, 2, 3]:
        return False
    if sorted(sprite["holes"] for sprite in sprites_in) != [1, 2, 3]:
        return False
    if len({sprite["color"] for sprite in sprites_in}) != 3:
        return False

    bars_out, sprites_out = _parse_scene(expected)
    if len(bars_out) != 3 or len(sprites_out) != 3:
        return False
    if any(bar["length"] != next(b2["length"] for b2 in bars_out if b2["rings"] == bar["rings"]) for bar in bars_in):
        return False

    bar_by_ring = {bar["rings"]: bar for bar in bars_in}
    in_by_holes = {sprite["holes"]: sprite for sprite in sprites_in}
    out_by_holes = {sprite["holes"]: sprite for sprite in sprites_out}
    if set(out_by_holes) != set(RINGS):
        return False

    for ring in RINGS:
        sprite_in = in_by_holes[ring]
        sprite_out = out_by_holes[ring]
        if not np.array_equal(sprite_in["patch"], sprite_out["patch"]):
            return False
        bar = bar_by_ring[ring]
        expected_r0 = int(bar["row"]) - sprite_in["patch"].shape[0]
        expected_c0 = _aligned_left_column(bar, sprite_in["patch"].shape[1])
        if tuple(map(int, sprite_out["bbox"][:2])) != (expected_r0, expected_c0):
            return False
        if tuple(map(int, sprite_in["bbox"][:2])) == (expected_r0, expected_c0):
            return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*GRID_ROWS))
    width = int(rng.integers(*GRID_COLS))

    sprite_colors = rng.choice(np.array(SPRITE_COLORS, dtype=int), size=3, replace=False).tolist()
    rng.shuffle(sprite_colors)
    sprite_masks = {ring: _grow_sprite_mask(rng, ring) for ring in RINGS}
    sprite_patches = {ring: _colorize(sprite_masks[ring], int(sprite_colors[ring - 1])) for ring in RINGS}
    sprite_heights = {ring: int(sprite_patches[ring].shape[0]) for ring in RINGS}

    lengths = _sample_bar_lengths(rng)
    rows = _sample_bar_rows(rng, height, sprite_heights)
    if rows is None:
        return None

    placements = _sample_bar_positions(rng, height, width, sprite_patches, lengths, rows)
    if placements is None:
        return None

    expected, final_positions = _expected_output_from_specs((height, width), sprite_patches, lengths, placements)
    top_limit = min(height // 2 + 1, min(row for row, _col in placements.values()) - 1)
    if top_limit < max(patch.shape[0] for patch in sprite_patches.values()):
        return None

    inp = np.full((height, width), int(BG), dtype=int)
    for ring in RINGS:
        row, col = placements[ring]
        _paint_bar(inp, row, col, lengths[ring])

    starts = _place_input_sprites(rng, inp, sprite_patches, final_positions, top_limit)
    if starts is None:
        return None

    if not _verify_generated_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for puzzle 7")
