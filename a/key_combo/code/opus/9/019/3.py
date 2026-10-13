from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BG = 0
TILE_SIZE = 3
RNG = np.random.default_rng()
CORNERS = ("tl", "tr", "bl", "br")
OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
CORNER_SETS = (("tl", "br"), ("tr", "bl"), ("tl", "tr", "bl", "br"))


def _corner_from_bbox(bbox: tuple[int, int, int, int], height: int, width: int) -> str | None:
    r0, c0, r1, c1 = map(int, bbox)
    if r0 == 0 and c0 == 0:
        return "tl"
    if r0 == 0 and c1 == width - 1:
        return "tr"
    if r1 == height - 1 and c0 == 0:
        return "bl"
    if r1 == height - 1 and c1 == width - 1:
        return "br"
    return None


def _target_origin(corner: str, patch_shape: tuple[int, int], height: int, width: int) -> tuple[int, int]:
    ph, pw = map(int, patch_shape)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - pw
    if corner == "bl":
        return height - ph, 0
    return height - ph, width - pw


def _find_tile(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = int(get_bg(g))
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"):
        bbox = tuple(map(int, comp["bbox"]))
        if bbox_h(bbox) != TILE_SIZE or bbox_w(bbox) != TILE_SIZE:
            continue
        r0, c0, r1, c1 = bbox
        if min(r0, c0, height - 1 - r1, width - 1 - c1) < 3:
            continue
        patch = crop_bbox(g, bbox)
        colors = set(map(int, np.unique(patch))) - {bg}
        if len(colors) < 2:
            continue
        if np.any(patch == bg):
            continue
        candidates.append((bbox, patch))
    assert len(candidates) == 1, "expected exactly one central 3x3 multicolor tile"
    return candidates[0]


def _corner_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = int(get_bg(g))
    out: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        bbox = tuple(map(int, comp["bbox"]))
        corner = _corner_from_bbox(bbox, height, width)
        if corner is None:
            continue
        patch = crop_bbox(g, bbox)
        mask = patch != bg
        out.append(
            {
                "corner": corner,
                "bbox": bbox,
                "mask": mask,
                "patch": patch,
                "color": int(comp["color"]),
            }
        )
    return out


def _expand_mask(mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    mh, mw = map(int, mask.shape)
    th, tw = map(int, tile.shape)
    out = np.full((mh * th, mw * tw), int(bg), dtype=int)
    for r in range(mh):
        for c in range(mw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each corner-anchored object as a tile mask and place the result at the diagonally opposite corner.

    Input structure:
    - A black background grid contains one central 3x3 multicolor tile, at least one diagonal pair of corner-anchored monochrome objects, and isolated distractor pixels.
    - Each active corner object is a 4-connected component whose bounding box touches exactly one corner's two adjacent edges.
    - The tile is the unique multicolor 8-connected component with a full 3x3 bounding box and distance at least 3 from every grid edge.

    Rule:
    - Treat every corner object's bounding-box contents as a binary mask: non-black cells are 1 and black cells are 0.
    - Replace each 1-cell of that mask by a full copy of the central 3x3 tile, and each 0-cell by a 3x3 black block, yielding a 3x enlarged patch.
    - Move that enlarged patch to the diagonally opposite corner with corner-to-corner alignment, overwrite whatever is there, and keep all non-corner distractors unchanged.
    - Remove the original central tile by painting its 3x3 area black.

    Key insight:
    - The same property, being anchored to a specific corner, simultaneously selects the mask object and determines the destination corner for the enlarged tile image.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape
    tile_bbox, tile = _find_tile(g)

    out = g.copy()
    r0, c0, r1, c1 = tile_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for info in _corner_objects(g):
        mask = np.asarray(info["mask"], dtype=bool)
        expanded = _expand_mask(mask, tile, bg)
        dest_corner = OPPOSITE[str(info["corner"])]
        dest_r0, dest_c0 = _target_origin(dest_corner, expanded.shape, height, width)
        paste(out, expanded, dest_r0, dest_c0)

    return out


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not bool(mask[nr, nc]):
                continue
            nxt = (int(nr), int(nc))
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen) == int(mask.sum())


def _spans_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_mask_library() -> dict[tuple[int, int], list[np.ndarray]]:
    lib: dict[tuple[int, int], list[np.ndarray]] = {}
    for height in range(1, 5):
        for width in range(1, 5):
            if height == width == 1:
                continue
            masks: list[np.ndarray] = []
            area = height * width
            for bits in range(1, 1 << area):
                filled = bits.bit_count()
                if not (3 <= filled <= 6):
                    continue
                mask = np.zeros((height, width), dtype=bool)
                for idx in range(area):
                    if (bits >> idx) & 1:
                        mask[idx // width, idx % width] = True
                if not _spans_full_bbox(mask):
                    continue
                if not _connected4(mask):
                    continue
                masks.append(mask)
            if masks:
                lib[(height, width)] = masks
    return lib


MASK_LIBRARY = _build_mask_library()


def _sample_mask() -> np.ndarray:
    dims = list(MASK_LIBRARY)
    height, width = dims[int(RNG.integers(0, len(dims)))]
    options = MASK_LIBRARY[(height, width)]
    irregular = [mask for mask in options if int(mask.sum()) < int(mask.size)]
    if irregular and RNG.random() < 0.75:
        choice = irregular[int(RNG.integers(0, len(irregular)))]
    else:
        choice = options[int(RNG.integers(0, len(options)))]
    return choice.copy()


def _make_tile(tile_colors: list[int]) -> np.ndarray:
    tile = RNG.choice(np.array(tile_colors, dtype=int), size=(TILE_SIZE, TILE_SIZE), replace=True)
    used = sorted(set(map(int, tile.flatten())))
    while len(used) < 2:
        tile = RNG.choice(np.array(tile_colors, dtype=int), size=(TILE_SIZE, TILE_SIZE), replace=True)
        used = sorted(set(map(int, tile.flatten())))
    for idx, color in enumerate(tile_colors):
        rr = idx // TILE_SIZE
        cc = idx % TILE_SIZE
        tile[rr, cc] = int(color)
    return tile.astype(int)


def _place_corner_patch(grid: np.ndarray, corner: str, mask: np.ndarray, color: int) -> tuple[int, int, int, int]:
    height, width = grid.shape
    mh, mw = map(int, mask.shape)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, width - mw
    elif corner == "bl":
        r0, c0 = height - mh, 0
    else:
        r0, c0 = height - mh, width - mw
    patch = np.full((mh, mw), BG, dtype=int)
    patch[mask] = int(color)
    paste(grid, patch, r0, c0, transparent=BG)
    return r0, c0, r0 + mh - 1, c0 + mw - 1


def _target_sizes(active: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    sizes = {corner: (0, 0) for corner in CORNERS}
    for source_corner, mask in active.items():
        dest_corner = OPPOSITE[source_corner]
        sizes[dest_corner] = (int(mask.shape[0]) * TILE_SIZE, int(mask.shape[1]) * TILE_SIZE)
    return sizes


def _non_overlapping_targets(sizes: dict[str, tuple[int, int]], height: int, width: int) -> bool:
    tl_h, tl_w = sizes["tl"]
    tr_h, tr_w = sizes["tr"]
    bl_h, bl_w = sizes["bl"]
    br_h, br_w = sizes["br"]
    return (
        tl_w + tr_w <= width
        and bl_w + br_w <= width
        and tl_h + bl_h <= height
        and tr_h + br_h <= height
    )


def _pick_tile_origin(height: int, width: int) -> tuple[int, int]:
    row_hi = min(height // 2, height - TILE_SIZE - 3)
    row_lo = 3
    if row_hi < row_lo:
        row_hi = row_lo
    center = (width - TILE_SIZE) // 2
    col_lo = max(3, center - 2)
    col_hi = min(width - TILE_SIZE - 3, center + 2)
    if col_hi < col_lo:
        col_lo, col_hi = 3, width - TILE_SIZE - 3
    return int(RNG.integers(row_lo, row_hi + 1)), int(RNG.integers(col_lo, col_hi + 1))


def _padded_occupancy(grid: np.ndarray) -> np.ndarray:
    occ = grid != BG
    padded = occ.copy()
    padded[:-1, :] |= occ[1:, :]
    padded[1:, :] |= occ[:-1, :]
    padded[:, :-1] |= occ[:, 1:]
    padded[:, 1:] |= occ[:, :-1]
    padded[:-1, :-1] |= occ[1:, 1:]
    padded[:-1, 1:] |= occ[1:, :-1]
    padded[1:, :-1] |= occ[:-1, 1:]
    padded[1:, 1:] |= occ[:-1, :-1]
    return padded


def _place_distractors(grid: np.ndarray, colors: list[int]) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    count = int(RNG.integers(1, 5))
    kinds = ["interior"]
    if RNG.random() < 0.65:
        kinds.append("edge")
    while len(kinds) < count:
        kinds.append("edge" if RNG.random() < 0.45 else "interior")
    RNG.shuffle(kinds)

    placed: list[tuple[int, int, int]] = []
    for kind in kinds:
        padded = _padded_occupancy(grid)
        candidates: list[tuple[int, int]] = []
        if kind == "edge":
            for c in range(1, width - 1):
                candidates.append((0, c))
                candidates.append((height - 1, c))
            for r in range(1, height - 1):
                candidates.append((r, 0))
                candidates.append((r, width - 1))
        else:
            for r in range(1, height - 1):
                for c in range(1, width - 1):
                    candidates.append((r, c))
        RNG.shuffle(candidates)
        found = False
        for r, c in candidates:
            if padded[r, c]:
                continue
            color = int(colors[int(RNG.integers(0, len(colors)))])
            grid[r, c] = color
            placed.append((r, c, color))
            found = True
            break
        if not found:
            return []
    return placed


def _expected_from_spec(
    grid: np.ndarray,
    tile_origin: tuple[int, int],
    tile: np.ndarray,
    active: dict[str, np.ndarray],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    tr, tc = tile_origin
    out[tr : tr + TILE_SIZE, tc : tc + TILE_SIZE] = BG
    height, width = out.shape
    for source_corner, mask in active.items():
        expanded = _expand_mask(mask, tile, BG)
        dest_r0, dest_c0 = _target_origin(OPPOSITE[source_corner], expanded.shape, height, width)
        paste(out, expanded, dest_r0, dest_c0)
    return out


def _scan_semantics(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[str]]:
    tile_bbox, _tile = _find_tile(grid)
    corners = {str(info["corner"]) for info in _corner_objects(grid)}
    return tile_bbox, corners


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    active_corners = CORNER_SETS[int(RNG.integers(0, len(CORNER_SETS)))]
    corner_masks = {corner: _sample_mask() for corner in active_corners}
    if not any(int(mask.sum()) < int(mask.size) for mask in corner_masks.values()):
        return None

    sizes = _target_sizes(corner_masks)
    min_height = max(18, sizes["tl"][0] + sizes["bl"][0], sizes["tr"][0] + sizes["br"][0])
    min_width = max(18, sizes["tl"][1] + sizes["tr"][1], sizes["bl"][1] + sizes["br"][1])
    if min_height > 30 or min_width > 30:
        return None

    height = int(RNG.integers(min_height, 31))
    width = int(RNG.integers(min_width, 31))
    if not _non_overlapping_targets(sizes, height, width):
        return None

    tile_color_count = int(RNG.integers(2, 4))
    unique_needed = len(active_corners) + tile_color_count
    palette = list(map(int, RNG.choice(np.arange(1, 10, dtype=int), size=unique_needed, replace=False)))
    tile_colors = palette[:tile_color_count]
    corner_colors = palette[tile_color_count:]
    tile = _make_tile(tile_colors)
    tile_origin = _pick_tile_origin(height, width)

    grid = np.full((height, width), BG, dtype=int)
    corner_bboxes: dict[str, tuple[int, int, int, int]] = {}
    for corner, color in zip(active_corners, corner_colors):
        corner_bboxes[corner] = _place_corner_patch(grid, corner, corner_masks[corner], color)

    paste(grid, tile, tile_origin[0], tile_origin[1])
    distractors = _place_distractors(grid, tile_colors + corner_colors)
    if not distractors:
        return None

    expected = _expected_from_spec(grid, tile_origin, tile, corner_masks)

    tile_bbox, scanned_corners = _scan_semantics(grid)
    if set(active_corners) != scanned_corners:
        return None
    if tile_bbox != (
        tile_origin[0],
        tile_origin[1],
        tile_origin[0] + TILE_SIZE - 1,
        tile_origin[1] + TILE_SIZE - 1,
    ):
        return None
    if not all(
        _corner_from_bbox((r, c, r, c), height, width) is None
        for r, c, _color in distractors
    ):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _generate_case()
        if case is None:
            continue
        grid, _expected = case
        return grid
    raise RuntimeError("failed to generate a valid input")
