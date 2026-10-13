from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox, crop_bbox, get_bg, paste
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_hole_template",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    bbox = _ARC_UTILS.bbox
    crop_bbox = _ARC_UTILS.crop_bbox
    get_bg = _ARC_UTILS.get_bg
    paste = _ARC_UTILS.paste


MAX_GRID = 30
MAX_RETRIES = 400

LAYOUTS = [
    (tile_rows, tile_cols, tile_h, tile_w)
    for tile_rows in range(2, 5)
    for tile_cols in range(2, 5)
    for tile_h in range(4, 7)
    for tile_w in range(4, 7)
    if tile_rows * tile_cols >= 4
    and tile_rows * tile_h + tile_rows + 1 <= MAX_GRID
    and tile_cols * tile_w + tile_cols + 1 <= MAX_GRID
]


def _tile_boxes(sep_rows: list[int], sep_cols: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for rr in range(len(sep_rows) - 1):
        for cc in range(len(sep_cols) - 1):
            boxes.append((sep_rows[rr] + 1, sep_cols[cc] + 1, sep_rows[rr + 1] - 1, sep_cols[cc + 1] - 1))
    return boxes


def _regular_tile_span(lines: np.ndarray, limit: int) -> int | None:
    if len(lines) < 3 or int(lines[0]) != 0 or int(lines[-1]) != int(limit - 1):
        return None
    diffs = np.diff(lines)
    if np.any(diffs < 2):
        return None
    if len(set(map(int, diffs))) != 1:
        return None
    return int(diffs[0] - 1)


def _separator_spec(grid: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        sep_rows = np.flatnonzero(np.all(g == color, axis=1))
        sep_cols = np.flatnonzero(np.all(g == color, axis=0))
        tile_h = _regular_tile_span(sep_rows, h)
        tile_w = _regular_tile_span(sep_cols, w)
        if tile_h is None or tile_w is None:
            continue
        candidates.append((int(color), sep_rows.tolist(), sep_cols.tolist(), int(tile_h), int(tile_w)))
    assert len(candidates) == 1, "expected one separator-grid color"
    return candidates[0]


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    bb = bbox(mask)
    assert bb is not None, "pattern must be non-empty"
    return crop_bbox(mask.astype(np.uint8), bb).astype(bool)


def _scene_roles(
    grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]
) -> tuple[int, int]:
    inner_mask = np.ones(grid.shape, dtype=bool)
    inner_mask[sep_rows, :] = False
    inner_mask[:, sep_cols] = False
    inner_values = np.asarray(grid[inner_mask], dtype=int)
    assert inner_values.size > 0, "expected non-separator tile interiors"
    bg = int(get_bg(inner_values.reshape(1, -1)))
    fg_colors = sorted(set(map(int, np.unique(inner_values))) - {bg})
    assert len(fg_colors) == 1, "expected exactly one foreground drawing color inside the tiles"
    return int(bg), int(fg_colors[0])


def _negative_tile_index(tiles: list[np.ndarray], fg: int, bg: int) -> int:
    counts = [int(np.sum(tile == fg)) for tile in tiles]
    max_count = max(counts)
    indices = [idx for idx, count in enumerate(counts) if count == max_count]
    assert len(indices) == 1, "expected a unique dense negative tile"
    neg_idx = int(indices[0])
    neg_tile = tiles[neg_idx]
    assert np.any(neg_tile == bg) and np.any(neg_tile == fg), "negative tile must contain colored cells and holes"
    return neg_idx


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= MAX_GRID, "expected an ARC-sized 2D grid"

    sep, sep_rows, sep_cols, tile_h, tile_w = _separator_spec(g)
    bg, fg = _scene_roles(g, sep_rows, sep_cols)
    boxes = _tile_boxes(sep_rows, sep_cols)
    tiles = [crop_bbox(g, box) for box in boxes]
    assert all(tile.shape == (tile_h, tile_w) for tile in tiles), "all tiles must share one size"
    assert all(set(map(int, np.unique(tile))) <= {bg, fg} for tile in tiles), "tiles may only contain bg and drawing color"

    neg_idx = _negative_tile_index(tiles, fg, bg)
    hole_mask = tiles[neg_idx] == bg
    assert hole_mask.any() and not np.all(hole_mask), "hole template must be non-empty and proper"

    return {
        "grid": g,
        "sep": int(sep),
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "tile_h": int(tile_h),
        "tile_w": int(tile_w),
        "bg": int(bg),
        "fg": int(fg),
        "boxes": boxes,
        "tiles": tiles,
        "negative_index": int(neg_idx),
        "hole_mask": hole_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the holes of the dense tile as the template, invert only exact in-tile matches, and clear the rest.

    Input structure:
    - One separator color forms a full row-and-column lattice, partitioning the grid into equal rectangular tiles.
    - Exactly one tile is a negative tile: it is mostly filled with one drawing color, with some background-colored holes.
    - Every other tile uses the same two tile-interior colors, but on a background base with only a smaller positive pattern.

    Rule:
    - Read the negative tile's background holes as the template, keeping their tile-local coordinates.
    - For every other tile, compare its drawing-color cells against that hole template cell-by-cell over the entire tile.
    - Exact matches invert the whole tile: the tile becomes fully colored except that the matched pattern turns back into background.
    - Non-matches are erased to background, while the separator lattice and the original negative tile stay unchanged.

    Key insight:
    - The template is the negative tile's empty space, not its colored mass, so matching requires figure-ground reversal without recropping or shifting coordinates.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = int(scene["bg"])
    fg = int(scene["fg"])
    boxes = scene["boxes"]
    tiles = scene["tiles"]
    neg_idx = int(scene["negative_index"])
    hole_mask = np.asarray(scene["hole_mask"], dtype=bool)

    out = g.copy()
    for idx, box in enumerate(boxes):
        if idx == neg_idx:
            continue
        tile = tiles[idx]
        pattern = tile == fg
        if np.array_equal(pattern, hole_mask):
            repl = np.full(tile.shape, fg, dtype=int)
            repl[hole_mask] = bg
        else:
            repl = np.full(tile.shape, bg, dtype=int)
        paste(out, repl, box[0], box[1])
    return out


def _random_sprite(local_h: int, local_w: int, area: int) -> np.ndarray:
    mask = np.zeros((local_h, local_w), dtype=bool)
    start = (random.randrange(local_h), random.randrange(local_w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < int(area):
        options = set()
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < local_h and 0 <= nc < local_w and not mask[nr, nc]:
                    options.add((nr, nc))
        if not options:
            empties = np.argwhere(~mask)
            pick = tuple(map(int, random.choice(empties.tolist())))
        else:
            pick = random.choice(sorted(options))
        mask[pick] = True
        frontier.add(pick)
        if random.random() < 0.3:
            empties = np.argwhere(~mask)
            if len(empties):
                rr, cc = map(int, random.choice(empties.tolist()))
                if any(abs(rr - r) + abs(cc - c) <= 1 for r, c in np.argwhere(mask)):
                    mask[rr, cc] = True
    return _tight_crop(mask)


def _embed_sprite(tile_shape: tuple[int, int], sprite: np.ndarray, top: int, left: int) -> np.ndarray:
    tile = np.zeros(tile_shape, dtype=bool)
    h, w = sprite.shape
    tile[int(top) : int(top) + h, int(left) : int(left) + w] = sprite
    return tile


def _toggle_variant(mask: np.ndarray) -> np.ndarray | None:
    true_cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    false_cells = [tuple(map(int, coord)) for coord in np.argwhere(~mask)]
    if not true_cells or not false_cells:
        return None
    for _ in range(80):
        remove = random.choice(true_cells)
        add = random.choice(false_cells)
        cand = mask.copy()
        cand[remove] = False
        cand[add] = True
        if cand.any() and not np.array_equal(cand, mask):
            return cand
    return None


def _distractor_library(template: np.ndarray) -> list[np.ndarray]:
    tile_h, tile_w = template.shape
    template_crop = _tight_crop(template)
    b = bbox(template)
    assert b is not None
    r0, c0, _r1, _c1 = map(int, b)
    sh, sw = template_crop.shape

    library: list[np.ndarray] = []

    def add(candidate: np.ndarray) -> None:
        if not candidate.any() or np.array_equal(candidate, template):
            return
        for existing in library:
            if np.array_equal(existing, candidate):
                return
        library.append(candidate.copy())

    for rr in range(tile_h - sh + 1):
        for cc in range(tile_w - sw + 1):
            if rr == r0 and cc == c0:
                continue
            add(_embed_sprite((tile_h, tile_w), template_crop, rr, cc))

    toggled = _toggle_variant(template)
    if toggled is not None:
        add(toggled)

    for transformed in (
        np.rot90(template_crop, 1),
        np.rot90(template_crop, 2),
        np.rot90(template_crop, 3),
        np.fliplr(template_crop),
        np.flipud(template_crop),
    ):
        th, tw = transformed.shape
        if th <= tile_h and tw <= tile_w:
            rr = min(r0, tile_h - th)
            cc = min(c0, tile_w - tw)
            add(_embed_sprite((tile_h, tile_w), transformed.astype(bool), rr, cc))

    assert library, "expected at least one distractor candidate"
    return library


def _sample_template(tile_h: int, tile_w: int) -> np.ndarray:
    area = tile_h * tile_w
    local_h = random.randint(2, tile_h - 1)
    local_w = random.randint(2, tile_w - 1)
    max_holes = min(local_h * local_w - 1, max(3, area // 3))
    hole_area = random.randint(3, max_holes)
    sprite = _random_sprite(local_h, local_w, hole_area)
    sh, sw = sprite.shape
    top = random.randint(0, tile_h - sh)
    left = random.randint(0, tile_w - sw)
    template = _embed_sprite((tile_h, tile_w), sprite, top, left)
    assert int(template.sum()) * 2 < area, "holes must stay smaller than half the tile"
    return template


def _expected_output(grid: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(meta["bg"])
    fg = int(meta["fg"])
    negative_tile = np.asarray(meta["negative_tile"], dtype=int)
    boxes = meta["boxes"]
    neg_idx = int(meta["negative_index"])
    match_indices = set(meta["match_indices"])
    for idx, box in enumerate(boxes):
        if idx == neg_idx or idx in match_indices:
            paste(out, negative_tile, box[0], box[1])
        else:
            paste(out, np.full(negative_tile.shape, bg, dtype=int), box[0], box[1])
    return out


def _validate_generated(grid: np.ndarray, meta: dict[str, object]) -> bool:
    scene = _parse_scene(grid)
    if int(scene["negative_index"]) != int(meta["negative_index"]):
        return False
    if not np.array_equal(scene["hole_mask"], np.asarray(meta["hole_mask"], dtype=bool)):
        return False

    fg = int(scene["fg"])
    recovered_matches = {
        idx
        for idx, tile in enumerate(scene["tiles"])
        if idx != int(scene["negative_index"]) and np.array_equal(tile == fg, scene["hole_mask"])
    }
    if recovered_matches != set(meta["match_indices"]):
        return False
    if not recovered_matches:
        return False

    shifted_idx = int(meta["shifted_distractor"])
    shifted_mask = scene["tiles"][shifted_idx] == fg
    if np.array_equal(shifted_mask, scene["hole_mask"]):
        return False
    if int(shifted_mask.sum()) != int(np.sum(scene["hole_mask"])):
        return False
    if not np.array_equal(_tight_crop(shifted_mask), _tight_crop(scene["hole_mask"])):
        return False

    out = main(grid)
    if not np.array_equal(out, _expected_output(grid, meta)):
        return False

    bg = int(scene["bg"])
    negative_tile = scene["tiles"][int(scene["negative_index"])]
    for idx, box in enumerate(scene["boxes"]):
        out_tile = crop_bbox(out, box)
        if idx == int(scene["negative_index"]) or idx in recovered_matches:
            if not np.array_equal(out_tile, negative_tile):
                return False
        else:
            if not np.all(out_tile == bg):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w = random.choice(LAYOUTS)
    h = tile_rows * tile_h + tile_rows + 1
    w = tile_cols * tile_w + tile_cols + 1

    bg, sep, fg = random.sample(range(10), 3)
    sep_rows = list(range(0, h, tile_h + 1))
    sep_cols = list(range(0, w, tile_w + 1))
    boxes = _tile_boxes(sep_rows, sep_cols)
    n_tiles = len(boxes)

    negative_index = random.randrange(n_tiles)
    other_indices = [idx for idx in range(n_tiles) if idx != negative_index]
    max_matches = min(3, len(other_indices) - 1)
    if max_matches < 1:
        return None
    match_count = random.randint(1, max_matches)
    match_indices = set(random.sample(other_indices, match_count))
    distractor_indices = [idx for idx in other_indices if idx not in match_indices]
    shifted_distractor = random.choice(distractor_indices)

    hole_mask = _sample_template(tile_h, tile_w)
    distractors = _distractor_library(hole_mask)
    shifted_options = [
        cand
        for cand in distractors
        if int(cand.sum()) == int(hole_mask.sum()) and np.array_equal(_tight_crop(cand), _tight_crop(hole_mask))
    ]
    if not shifted_options:
        return None

    grid = np.full((h, w), sep, dtype=int)
    tile_masks: dict[int, np.ndarray] = {}
    for idx, box in enumerate(boxes):
        if idx == negative_index:
            tile = np.full((tile_h, tile_w), fg, dtype=int)
            tile[hole_mask] = bg
            paste(grid, tile, box[0], box[1])
            continue

        if idx in match_indices:
            mask = hole_mask.copy()
        elif idx == shifted_distractor:
            mask = random.choice(shifted_options).copy()
        else:
            mask = random.choice(distractors).copy()
        tile_masks[idx] = mask
        tile = np.full((tile_h, tile_w), bg, dtype=int)
        tile[mask] = fg
        paste(grid, tile, box[0], box[1])

    negative_tile = np.full((tile_h, tile_w), fg, dtype=int)
    negative_tile[hole_mask] = bg

    meta = {
        "bg": int(bg),
        "sep": int(sep),
        "fg": int(fg),
        "boxes": boxes,
        "negative_index": int(negative_index),
        "match_indices": set(match_indices),
        "shifted_distractor": int(shifted_distractor),
        "hole_mask": hole_mask.copy(),
        "negative_tile": negative_tile,
        "tile_masks": tile_masks,
    }
    if not _validate_generated(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid hole-template ARC input")
