from __future__ import annotations

import numpy as np


BG = 0
FG = 1
MARK = 2

ATLAS_SIZE = 9
SLOT_SIZE = 4
SLOT_STRIDE = 5
SLOT_POSITIONS = ((0, 0), (0, SLOT_STRIDE), (SLOT_STRIDE, 0), (SLOT_STRIDE, SLOT_STRIDE))
BOX_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))

GEN_INPUT_MIN = 17
GEN_INPUT_MAX = 22
GEN_MAX_ATTEMPTS = 300
GEN_ROW_GAP = 4
GEN_COL_GAP = 4

_RNG = np.random.default_rng()
_BANK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return components


def _corner_markers(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    return (
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    )


def _ordered_components(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    fg_colors = sorted(set(map(int, grid.flatten())) - {bg})
    assert fg_colors == [FG], "expected exactly one blue foreground color"

    components = _find_components(grid, bg=bg)
    assert len(components) == 4, "expected exactly four 4-connected sprites"

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert (h, w) in BOX_SHAPES, "each sprite bbox must be 3x3, 3x4, 4x3, or 4x4"

    by_row = sorted(components, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    assert by_row[1]["bbox"][0] < by_row[2]["bbox"][0], "sprites must form upper and lower pairs"

    upper = sorted(by_row[:2], key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    lower = sorted(by_row[2:], key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    assert upper[0]["bbox"][1] < upper[1]["bbox"][1], "upper sprites must be left/right separated"
    assert lower[0]["bbox"][1] < lower[1]["bbox"][1], "lower sprites must be left/right separated"
    return upper + lower


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack four scattered blue sprites into a 2x2 atlas and add red outer-corner marks only for atlas entries whose original short bbox side is odd.

    Input structure:
    - The background is black and the only foreground color is blue.
    - Exactly four separated 4-connected blue sprites are present.
    - Each sprite fits in a 3x3, 3x4, 4x3, or 4x4 bounding box, and the four sprites form an upper pair and a lower pair.

    Rule:
    - Crop each sprite to its original bounding box and keep its internal black holes.
    - Order the crops by input position: upper-left, upper-right, lower-left, lower-right.
    - Paste them into a 9x9 atlas as four 4x4 slots with one black separator row and column, aligning every crop to its slot's top-left corner.
    - For each crop whose original short bbox side is 3, paint red on the orthogonally adjacent outside cells of the crop bbox corners in the atlas whenever those cells are in-bounds and still background.

    Key insight:
    - The marker locations depend on the packed atlas positions, but marker eligibility depends on the pre-pack bbox parity recorded from the original scattered sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ordered = _ordered_components(g)
    out = np.full((ATLAS_SIZE, ATLAS_SIZE), BG, dtype=int)

    placements: list[dict] = []
    for comp, (slot_r, slot_c) in zip(ordered, SLOT_POSITIONS):
        bbox = comp["bbox"]
        crop = _crop_bbox(g, bbox)
        h, w = crop.shape
        _paste(out, crop, slot_r, slot_c)
        placements.append(
            {
                "atlas_bbox": (slot_r, slot_c, slot_r + h - 1, slot_c + w - 1),
                "eligible": min(h, w) % 2 == 1,
            }
        )

    for placed in placements:
        if not placed["eligible"]:
            continue
        for rr, cc in _corner_markers(placed["atlas_bbox"]):
            if 0 <= rr < ATLAS_SIZE and 0 <= cc < ATLAS_SIZE and int(out[rr, cc]) == BG:
                out[rr, cc] = MARK

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    visited = 0
    while stack:
        r, c = stack.pop()
        visited += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return visited == int(mask.sum())


def _component_bank(shape: tuple[int, int]) -> list[np.ndarray]:
    if shape in _BANK_CACHE:
        return _BANK_CACHE[shape]

    h, w = shape
    bank: list[np.ndarray] = []
    min_area = max(4, h + w - 2)
    for bits in range(1, 1 << (h * w)):
        flat = np.array([(bits >> idx) & 1 for idx in range(h * w)], dtype=int)
        mask = flat.reshape(h, w).astype(bool)
        area = int(mask.sum())
        if area < min_area:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _mask_connected(mask):
            continue
        bank.append(np.where(mask, FG, BG).astype(int))

    assert bank, f"no sprites generated for shape {shape}"
    _BANK_CACHE[shape] = bank
    return bank


def _sample_sprite(shape: tuple[int, int]) -> np.ndarray:
    bank = _component_bank(shape)
    return bank[int(_RNG.integers(0, len(bank)))].copy()


def _sample_bbox_shapes() -> list[tuple[int, int]]:
    eligible_count = int(_RNG.choice([0, 1, 2, 3, 4], p=[0.08, 0.27, 0.30, 0.25, 0.10]))
    shapes = []
    for _ in range(eligible_count):
        shapes.append(((3, 3), (3, 4), (4, 3))[int(_RNG.integers(0, 3))])
    for _ in range(4 - eligible_count):
        shapes.append((4, 4))
    _RNG.shuffle(shapes)
    return shapes


def _sample_layout(shape_specs: list[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    top_h = max(shape_specs[0][0], shape_specs[1][0])
    bottom_h = max(shape_specs[2][0], shape_specs[3][0])
    left_w = max(shape_specs[0][1], shape_specs[2][1])
    right_w = max(shape_specs[1][1], shape_specs[3][1])

    row_room = height - (top_h + bottom_h + GEN_ROW_GAP)
    col_room = width - (left_w + right_w + GEN_COL_GAP)
    assert row_room >= 1 and col_room >= 1

    top_band_max = min(6, row_room - 1)
    left_band_max = min(6, col_room - 1)

    for _ in range(80):
        r_tl = int(_RNG.integers(0, top_band_max + 1))
        r_tr = int(_RNG.integers(0, top_band_max + 1))
        top_end = max(r_tl + shape_specs[0][0] - 1, r_tr + shape_specs[1][0] - 1)
        bottom_min = top_end + GEN_ROW_GAP + 1
        if bottom_min > height - shape_specs[2][0] or bottom_min > height - shape_specs[3][0]:
            continue
        r_bl = int(_RNG.integers(bottom_min, height - shape_specs[2][0] + 1))
        r_br = int(_RNG.integers(bottom_min, height - shape_specs[3][0] + 1))

        c_tl = int(_RNG.integers(0, left_band_max + 1))
        c_bl = int(_RNG.integers(0, left_band_max + 1))
        left_end = max(c_tl + shape_specs[0][1] - 1, c_bl + shape_specs[2][1] - 1)
        right_min = left_end + GEN_COL_GAP + 1
        if right_min > width - shape_specs[1][1] or right_min > width - shape_specs[3][1]:
            continue
        c_tr = int(_RNG.integers(right_min, width - shape_specs[1][1] + 1))
        c_br = int(_RNG.integers(right_min, width - shape_specs[3][1] + 1))

        return [(r_tl, c_tl), (r_tr, c_tr), (r_bl, c_bl), (r_br, c_br)]

    raise RuntimeError("failed to sample a valid 2x2 sprite layout")


def _render_input(height: int, width: int, sprites: list[np.ndarray], placements: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for sprite, (r0, c0) in zip(sprites, placements):
        h, w = sprite.shape
        region = grid[r0 : r0 + h, c0 : c0 + w]
        assert np.all(region == BG), "sprites must not overlap"
        _paste(grid, sprite, r0, c0)
    return grid


def _input_is_valid(grid: np.ndarray) -> bool:
    try:
        ordered = _ordered_components(grid)
        main(grid)
    except Exception:
        return False

    upper = ordered[:2]
    lower = ordered[2:]
    top_end = max(comp["bbox"][2] for comp in upper)
    bottom_start = min(comp["bbox"][0] for comp in lower)
    left_end = max(ordered[0]["bbox"][3], ordered[2]["bbox"][3])
    right_start = min(ordered[1]["bbox"][1], ordered[3]["bbox"][1])
    return bottom_start - top_end - 1 >= GEN_ROW_GAP and right_start - left_end - 1 >= GEN_COL_GAP


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(GEN_INPUT_MIN, GEN_INPUT_MAX + 1))
    width = int(_RNG.integers(GEN_INPUT_MIN, GEN_INPUT_MAX + 1))
    shapes = _sample_bbox_shapes()
    sprites = [_sample_sprite(shape) for shape in shapes]
    placements = _sample_layout([sprite.shape for sprite in sprites], height, width)
    grid = _render_input(height, width, sprites, placements)
    if _input_is_valid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
