from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste

BG = 0
BLOCK_COLORS = [1, 2, 3]
BLOCK_HEIGHTS = [4, 6]
BLOCK_WIDTH_RANGE = (4, 6)
FULL_GAP_RANGE = (2, 4)
LEFT_COL_RANGE = (1, 3)
MID_GAP_RANGE = (3, 5)
RIGHT_REGION_WIDTH_RANGE = (4, 6)
GRID_MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_motif_library() -> dict[tuple[int, int], list[np.ndarray]]:
    library: dict[tuple[int, int], dict[tuple[str, ...], np.ndarray]] = {}
    for area in range(2, 5):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            if not _is_connected(mask):
                continue
            tight = _tight_mask(mask)
            key = _mask_key(tight)
            library.setdefault(tight.shape, {})
            library[tight.shape][key] = tight
    return {shape: list(items.values()) for shape, items in library.items()}


MOTIF_LIBRARY = _build_motif_library()
MOTIF_BBOXES = sorted(MOTIF_LIBRARY)


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _classify_components(grid: np.ndarray) -> tuple[int, list[dict], dict[int, np.ndarray]]:
    bg = int(get_bg(grid))
    assert bg == BG, "background must be black"

    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(components) == 6, "expected three half-blocks and three motifs"

    blocks_by_color: dict[int, dict] = {}
    motifs_by_color: dict[int, np.ndarray] = {}

    for comp in components:
        color = int(comp["color"])
        box = comp["bbox"]
        h = int(bbox_h(box))
        w = int(bbox_w(box))
        area = int(comp["area"])
        patch = crop_bbox(grid, box)

        if 4 <= w <= 6 and h in (2, 3) and area == h * w and np.all(patch == color):
            assert color not in blocks_by_color, "each color must have one left half-block"
            blocks_by_color[color] = comp
            continue

        assert 1 <= h <= 3 and 1 <= w <= 3, "motif bbox must be at most 3x3"
        assert 2 <= area <= 4, "motif area must be 2..4"
        mask = _component_mask(comp)
        assert mask.shape == (h, w)
        assert color not in motifs_by_color, "each color must have one motif"
        motifs_by_color[color] = mask

    assert len(blocks_by_color) == 3 and len(motifs_by_color) == 3
    assert set(blocks_by_color) == set(motifs_by_color), "motif colors must match block colors"

    blocks = sorted(blocks_by_color.values(), key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    return bg, blocks, motifs_by_color


def _complete_half_block(grid: np.ndarray, component: dict) -> np.ndarray:
    top_half = crop_bbox(grid, component["bbox"])
    return np.concatenate([top_half, top_half[::-1, :]], axis=0)


def _carve_centered(tile: np.ndarray, motif_mask: np.ndarray, bg: int) -> np.ndarray:
    carved = tile.copy()
    mh, mw = motif_mask.shape
    th, tw = carved.shape
    assert mh < th and mw < tw, "motif bbox must be strictly smaller than its completed block"
    off_r = (th - mh) // 2
    off_c = (tw - mw) // 2
    view = carved[off_r : off_r + mh, off_c : off_c + mw]
    view[motif_mask] = int(bg)
    return carved


def _pack_tiles(tiles: list[np.ndarray], bg: int) -> np.ndarray:
    max_w = max(int(tile.shape[1]) for tile in tiles)
    out_h = 2 + sum(int(tile.shape[0]) for tile in tiles) + (len(tiles) - 1)
    out_w = max_w + 2
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r = 1
    for tile in tiles:
        c = 1 + (max_w - int(tile.shape[1])) // 2
        paste(out, tile, r, c)
        r += int(tile.shape[0]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete each half-filled rectangle, carve a same-color centered motif, and pack the results vertically.

    Input structure:
    - The background is black, with exactly three monochrome rectangular half-blocks on the left and one same-colored motif per block on the right.
    - Each half-block is a solid rectangle whose visible height is the upper half of an even-height completed block.
    - Each motif is a small 4-connected same-color component with tight bbox at most 3x3 and strictly smaller than its matched completed block.

    Rule:
    - For each left half-block, reflect its colored upper half downward across the horizontal midline to reconstruct the full rectangle.
    - Match each right motif to the block of the same color, align the motif bbox to the center of the completed block with left/up bias on odd differences, and turn those motif cells into background holes.
    - Output the three carved blocks in their original top-to-bottom order, stacked vertically with one black gap row and a one-cell black outer border; narrower blocks are horizontally centered.

    Key insight:
    - The motif placement depends on the size of the mirrored full block, so symmetry completion must happen before carving and packing.
    """
    g = np.asarray(grid, dtype=int)
    bg, blocks, motifs_by_color = _classify_components(g)

    carved_tiles: list[np.ndarray] = []
    for block in blocks:
        color = int(block["color"])
        full_tile = _complete_half_block(g, block)
        carved_tiles.append(_carve_centered(full_tile, motifs_by_color[color], bg))
    return _pack_tiles(carved_tiles, bg)


def _expected_output(specs: list[dict]) -> np.ndarray:
    tiles = []
    for spec in specs:
        top_half = np.full((spec["full_h"] // 2, spec["w"]), spec["color"], dtype=int)
        full_tile = np.concatenate([top_half, top_half[::-1, :]], axis=0)
        tiles.append(_carve_centered(full_tile, spec["motif_mask"], BG))
    return _pack_tiles(tiles, BG)


def _valid_motif_boxes() -> list[tuple[int, int]]:
    return [shape for shape in MOTIF_BBOXES if shape != (1, 1)]


VALID_MOTIF_BOXES = _valid_motif_boxes()


def _choose_motif_boxes() -> list[tuple[int, int]]:
    for _ in range(200):
        boxes = [random.choice(VALID_MOTIF_BOXES) for _ in range(3)]
        heights = [h for h, _w in boxes]
        widths = [w for _h, w in boxes]
        if 2 in heights and any(h != 2 for h in heights) and 2 in widths and any(w != 2 for w in widths):
            return boxes
    raise RuntimeError("failed to sample motif boxes covering centering branches")


def _choose_specs() -> list[dict]:
    colors = random.sample(BLOCK_COLORS, 3)
    boxes = _choose_motif_boxes()
    specs = []
    for color, motif_box in zip(colors, boxes):
        full_h = random.choice(BLOCK_HEIGHTS)
        w = random.randint(*BLOCK_WIDTH_RANGE)
        mh, mw = motif_box
        assert mh < full_h and mw < w
        motif_mask = random.choice(MOTIF_LIBRARY[motif_box]).copy()
        specs.append(
            {
                "color": color,
                "full_h": full_h,
                "w": w,
                "motif_mask": motif_mask,
            }
        )
    return specs


def _assign_block_positions(specs: list[dict]) -> tuple[int, int, int]:
    top_margin = random.randint(*GRID_MARGIN_RANGE)
    bottom_margin = random.randint(*GRID_MARGIN_RANGE)

    r = top_margin
    for idx, spec in enumerate(specs):
        spec["block_r0"] = r
        spec["block_c0"] = random.randint(*LEFT_COL_RANGE)
        r += spec["full_h"]
        if idx < len(specs) - 1:
            r += random.randint(*FULL_GAP_RANGE)
    height = r + bottom_margin

    max_block_right = max(int(spec["block_c0"]) + int(spec["w"]) for spec in specs)
    right_region_c0 = max_block_right + random.randint(*MID_GAP_RANGE)
    right_region_w = random.randint(*RIGHT_REGION_WIDTH_RANGE)
    right_margin = random.randint(*GRID_MARGIN_RANGE)
    width = right_region_c0 + right_region_w + right_margin
    return height, width, right_region_c0


def _can_place(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    return not bool(blocked[r : r + h, c : c + w].any())


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(blocked.shape[0], r + h + 1)
    c1 = min(blocked.shape[1], c + w + 1)
    blocked[r0:r1, c0:c1] = True


def _place_motifs(specs: list[dict], height: int, width: int, right_region_c0: int) -> bool:
    blocked = np.zeros((height, width), dtype=bool)
    order = specs[:]
    random.shuffle(order)
    for spec in order:
        mh, mw = spec["motif_mask"].shape
        row_min = 1 if height - mh >= 2 else 0
        row_max = height - mh - 1 if height - mh >= 2 else height - mh
        col_min = right_region_c0
        col_max = width - mw - 1 if width - mw >= 2 else width - mw
        candidates = []
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                if _can_place(blocked, r, c, mh, mw):
                    candidates.append((r, c))
        if not candidates:
            return False
        spec["motif_r0"], spec["motif_c0"] = random.choice(candidates)
        _mark_blocked(blocked, spec["motif_r0"], spec["motif_c0"], mh, mw)
    return True


def _render_input(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        r0 = int(spec["block_r0"])
        c0 = int(spec["block_c0"])
        half_h = int(spec["full_h"]) // 2
        grid[r0 : r0 + half_h, c0 : c0 + int(spec["w"])] = int(spec["color"])

    for spec in specs:
        patch = np.where(spec["motif_mask"], int(spec["color"]), BG)
        paste(grid, patch, int(spec["motif_r0"]), int(spec["motif_c0"]), transparent=BG)
    return grid


def _attempt_input() -> np.ndarray | None:
    specs = _choose_specs()
    height, width, right_region_c0 = _assign_block_positions(specs)
    if max(height, width) > 30:
        return None
    if not _place_motifs(specs, height, width, right_region_c0):
        return None

    inp = _render_input(specs, height, width)
    expected = _expected_output(specs)
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for mirrored punch-and-pack ARC task")
