from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


try:
    from new.solvers.arc_utils import bbox_coords, components_mask, crop_bbox, paste
except Exception:
    _arc_utils = None
    for _parent in Path(__file__).resolve().parents:
        _candidate = _parent / "new" / "solvers" / "arc_utils.py"
        if _candidate.exists():
            _arc_utils = SourceFileLoader("_arc_utils_task_10", str(_candidate)).load_module()
            break
    if _arc_utils is None:
        raise
    bbox_coords = _arc_utils.bbox_coords
    components_mask = _arc_utils.components_mask
    crop_bbox = _arc_utils.crop_bbox
    paste = _arc_utils.paste


BBox = tuple[int, int, int, int]

BG = 0
FRAME = 9
SOURCE_POOL = (1, 2, 3)
TARGET_POOL = (4, 5, 6, 7, 8)
GRID_SIDE_RANGE = (18, 22)
PERIOD_RANGE = (3, 5)
OCCLUSION_RANGE = (4, 6)
CORNERS = ("tl", "tr", "bl", "br")
OUTSIDE_AXES = ("h", "v")
GEN_RETRIES = 500


def _bbox_dims(bbox: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _fill_bbox(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _iter_bbox(bbox: BBox):
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _bbox_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _in_bbox(r: int, c: int, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _apply_tile(canvas: np.ndarray, bbox: BBox, tile: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    pr, pc = tile.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            canvas[r, c] = int(tile[(r - r0) % pr, (c - c0) % pc])


def _find_legend(grid: np.ndarray) -> tuple[BBox, dict[int, int]]:
    candidates: list[tuple[BBox, dict[int, int]]] = []
    for coords in components_mask(grid == FRAME, connectivity=4):
        bbox = bbox_coords(coords)
        h, w = _bbox_dims(bbox)
        if w != 4 or h not in (4, 5):
            continue
        patch = crop_bbox(grid, bbox)
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME):
            continue
        inner = patch[1:-1, 1:-1]
        if inner.shape[1] != 2 or inner.shape[0] not in (2, 3):
            continue
        if np.any(inner == BG) or np.any(inner == FRAME):
            continue
        mapping: dict[int, int] = {}
        ok = True
        for src, dst in inner.tolist():
            src = int(src)
            dst = int(dst)
            if src == dst or src in mapping:
                ok = False
                break
            mapping[src] = dst
        if not ok:
            continue
        if len(set(mapping.values())) != len(mapping):
            continue
        if set(mapping) & set(mapping.values()):
            continue
        candidates.append((bbox, mapping))
    assert len(candidates) == 1, "expected one framed color legend"
    return candidates[0]


def _find_occlusion(grid: np.ndarray) -> BBox:
    H, W = grid.shape
    candidates: list[BBox] = []
    for coords in components_mask(grid == BG, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h, w = _bbox_dims(bbox)
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        if int(coords.shape[0]) != h * w:
            continue
        candidates.append(bbox)
    assert len(candidates) == 1, "expected one interior rectangular black occlusion"
    return candidates[0]


def _pattern_bbox(grid: np.ndarray, source_colors: set[int], legend_bbox: BBox) -> BBox:
    mask = np.isin(grid, list(source_colors))
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "pattern colors must remain visible outside the legend"
    return bbox_coords(coords)


def _candidate_tile(
    grid: np.ndarray,
    bbox: BBox,
    hidden: np.ndarray,
    source_colors: set[int],
    pr: int,
    pc: int,
) -> np.ndarray | None:
    r0, c0, r1, c1 = bbox
    tile = np.full((pr, pc), -1, dtype=int)
    seen = np.zeros((pr, pc), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if hidden[r, c]:
                continue
            color = int(grid[r, c])
            if color not in source_colors:
                return None
            rr = (r - r0) % pr
            cc = (c - c0) % pc
            if tile[rr, cc] == -1:
                tile[rr, cc] = color
            elif tile[rr, cc] != color:
                return None
            seen[rr, cc] = True
    if not np.all(seen):
        return None
    return tile


def _infer_tile(
    grid: np.ndarray,
    bbox: BBox,
    hidden: np.ndarray,
    source_colors: set[int],
) -> tuple[int, int, np.ndarray]:
    h, w = _bbox_dims(bbox)
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, min(6, h) + 1):
        for pc in range(1, min(6, w) + 1):
            tile = _candidate_tile(grid, bbox, hidden, source_colors, pr, pc)
            if tile is not None:
                candidates.append((pr * pc, pr, pc, tile.copy()))
    assert candidates, "no consistent translation period found"
    best_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "translation period must be uniquely determined"
    _area, pr, pc, tile = best[0]
    return pr, pc, tile


def _clean_pattern(grid: np.ndarray) -> tuple[np.ndarray, BBox, BBox, BBox, dict[int, int]]:
    legend_bbox, mapping = _find_legend(grid)
    source_colors = set(mapping)
    occlusion_bbox = _find_occlusion(grid)
    pattern_bbox = _pattern_bbox(grid, source_colors, legend_bbox)

    hidden = np.zeros(grid.shape, dtype=bool)
    _fill_bbox(hidden, legend_bbox, True)
    _fill_bbox(hidden, occlusion_bbox, True)

    _pr, _pc, tile = _infer_tile(grid, pattern_bbox, hidden, source_colors)
    base = np.zeros_like(grid)
    _apply_tile(base, pattern_bbox, tile)
    return base, pattern_bbox, legend_bbox, occlusion_bbox, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the occluded translated pattern, recolor only the restored cells via the framed legend, and erase the legend.

    Input structure:
    - A rectangular region of source colors is filled by a small 2D translated tile.
    - One interior black rectangle occludes part of that pattern.
    - A maroon rectangular frame encloses a 2-column legend whose rows map source colors to new colors.

    Rule:
    - Find the framed legend and read each row as source-color -> target-color.
    - Ignore the legend area and detect the unique small translation period on the visible pattern.
    - Use that period to restore the hidden colors inside the black rectangle.
    - Recolor only those restored occlusion cells with the legend mapping.
    - Remove the legend itself: restore underlying pattern colors where it lies on the pattern, otherwise set it to black.

    Key insight:
    - The pattern restoration and recoloring share the same spatial scope: the legend changes only the cells whose original colors must first be recovered from translation symmetry.
    """
    g = np.asarray(grid, dtype=int)
    base, pattern_bbox, legend_bbox, occlusion_bbox, mapping = _clean_pattern(g)

    out = g.copy()
    for r, c in _iter_bbox(occlusion_bbox):
        out[r, c] = int(mapping[int(base[r, c])])

    for r, c in _iter_bbox(legend_bbox):
        out[r, c] = int(base[r, c]) if _in_bbox(r, c, pattern_bbox) else BG

    return out


def _tile_has_full_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            probe = np.zeros_like(tile)
            _apply_tile(probe, (0, 0, h - 1, w - 1), tile[:pr, :pc])
            if np.array_equal(probe, tile):
                if pr * pc < h * w:
                    return False
    return True


def _random_tile(source_colors: tuple[int, ...], pr: int, pc: int) -> np.ndarray | None:
    for _ in range(300):
        tile = np.array(
            [[random.choice(source_colors) for _c in range(pc)] for _r in range(pr)],
            dtype=int,
        )
        if set(map(int, tile.flatten())) != set(source_colors):
            continue
        if len({tuple(row.tolist()) for row in tile}) == 1:
            continue
        if len({tuple(col.tolist()) for col in tile.T}) == 1:
            continue
        if not _tile_has_full_period(tile):
            continue
        return tile
    return None


def _legend_patch(mapping: dict[int, int]) -> np.ndarray:
    rows = len(mapping)
    patch = np.full((rows + 2, 4), FRAME, dtype=int)
    for idx, src in enumerate(mapping):
        patch[idx + 1, 1] = int(src)
        patch[idx + 1, 2] = int(mapping[src])
    return patch


def _corner_bbox(corner: str, h: int, w: int, H: int, W: int) -> BBox:
    if corner == "tl":
        return 0, 0, h - 1, w - 1
    if corner == "tr":
        return 0, W - w, h - 1, W - 1
    if corner == "bl":
        return H - h, 0, H - 1, w - 1
    if corner == "br":
        return H - h, W - w, H - 1, W - 1
    raise ValueError("invalid corner")


def _layout(H: int, W: int, legend_h: int, legend_w: int, corner: str, legend_on_pattern: bool) -> tuple[BBox, BBox]:
    legend_bbox = _corner_bbox(corner, legend_h, legend_w, H, W)
    if legend_on_pattern:
        return (0, 0, H - 1, W - 1), legend_bbox

    axis = random.choice(OUTSIDE_AXES)
    if axis == "h":
        if corner in ("tl", "tr"):
            pattern_bbox = (legend_h + 1, 0, H - 1, W - 1)
        else:
            pattern_bbox = (0, 0, H - legend_h - 2, W - 1)
    else:
        if corner in ("tl", "bl"):
            pattern_bbox = (0, legend_w + 1, H - 1, W - 1)
        else:
            pattern_bbox = (0, 0, H - 1, W - legend_w - 2)
    return pattern_bbox, legend_bbox


def _choose_occlusion(pattern_bbox: BBox, legend_bbox: BBox, legend_on_pattern: bool) -> BBox | None:
    r0, c0, r1, c1 = pattern_bbox
    pat_h, pat_w = _bbox_dims(pattern_bbox)
    for _ in range(200):
        h = random.randint(OCCLUSION_RANGE[0], min(OCCLUSION_RANGE[1], pat_h - 2))
        w = random.randint(OCCLUSION_RANGE[0], min(OCCLUSION_RANGE[1], pat_w - 2))
        top = random.randint(r0 + 1, r1 - h)
        left = random.randint(c0 + 1, c1 - w)
        occ = (top, left, top + h - 1, left + w - 1)
        if legend_on_pattern and _bbox_overlap(occ, legend_bbox):
            continue
        return occ
    return None


def _semantic_checks(
    inp: np.ndarray,
    clean: np.ndarray,
    expected: np.ndarray,
    pattern_bbox: BBox,
    legend_bbox: BBox,
    occlusion_bbox: BBox,
    mapping: dict[int, int],
    legend_on_pattern: bool,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        solved_base, solved_pattern, solved_legend, solved_occ, solved_mapping = _clean_pattern(inp)
    except Exception:
        return False
    if not np.array_equal(solved_base, clean):
        return False
    if solved_pattern != pattern_bbox or solved_legend != legend_bbox or solved_occ != occlusion_bbox:
        return False
    if solved_mapping != mapping:
        return False

    occ_colors = {int(clean[r, c]) for r, c in _iter_bbox(occlusion_bbox)}
    if len(occ_colors) < 2:
        return False
    if any(np.count_nonzero(clean == color) == np.count_nonzero(crop_bbox(clean, occlusion_bbox) == color) for color in occ_colors):
        return False

    restored_only = clean.copy()
    if np.array_equal(restored_only, expected):
        return False

    recolor_all = clean.copy()
    source_colors = set(mapping)
    for r, c in _iter_bbox(pattern_bbox):
        if int(recolor_all[r, c]) in source_colors:
            recolor_all[r, c] = int(mapping[int(recolor_all[r, c])])
    if np.array_equal(recolor_all, expected):
        return False

    if legend_on_pattern:
        if not any(clean[r, c] != BG for r, c in _iter_bbox(legend_bbox)):
            return False
    else:
        if any(expected[r, c] != BG for r, c in _iter_bbox(legend_bbox)):
            return False

    hidden = np.zeros(inp.shape, dtype=bool)
    _fill_bbox(hidden, legend_bbox, True)
    _fill_bbox(hidden, occlusion_bbox, True)
    candidates = []
    for pr in range(1, 7):
        for pc in range(1, 7):
            tile = _candidate_tile(inp, pattern_bbox, hidden, source_colors, pr, pc)
            if tile is not None:
                candidates.append((pr * pc, pr, pc))
    if not candidates:
        return False
    best_area = min(area for area, _pr, _pc in candidates)
    if sum(area == best_area for area, _pr, _pc in candidates) != 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)

    color_count = random.choice((2, 3))
    source_colors = tuple(random.sample(SOURCE_POOL, color_count))
    target_colors = tuple(random.sample(TARGET_POOL, color_count))
    order = list(source_colors)
    random.shuffle(order)
    mapping = {src: target_colors[idx] for idx, src in enumerate(order)}

    pr = random.randint(*PERIOD_RANGE)
    pc = random.randint(*PERIOD_RANGE)
    tile = _random_tile(source_colors, pr, pc)
    if tile is None:
        return None

    legend_patch = _legend_patch(mapping)
    legend_h, legend_w = legend_patch.shape
    corner = random.choice(CORNERS)
    legend_on_pattern = random.random() < 0.4
    pattern_bbox, legend_bbox = _layout(H, W, legend_h, legend_w, corner, legend_on_pattern)
    pat_h, pat_w = _bbox_dims(pattern_bbox)
    if pat_h < 10 or pat_w < 10:
        return None

    clean = np.zeros((H, W), dtype=int)
    _apply_tile(clean, pattern_bbox, tile)

    occlusion_bbox = _choose_occlusion(pattern_bbox, legend_bbox, legend_on_pattern)
    if occlusion_bbox is None:
        return None

    occ_patch = crop_bbox(clean, occlusion_bbox)
    occ_colors = set(map(int, occ_patch.flatten()))
    if len(occ_colors) < 2:
        return None
    if any(np.count_nonzero(clean == color) == np.count_nonzero(occ_patch == color) for color in occ_colors):
        return None

    inp = clean.copy()
    paste(inp, legend_patch, legend_bbox[0], legend_bbox[1])
    _fill_bbox(inp, occlusion_bbox, BG)

    expected = clean.copy()
    for r, c in _iter_bbox(occlusion_bbox):
        expected[r, c] = int(mapping[int(clean[r, c])])

    if not _semantic_checks(inp, clean, expected, pattern_bbox, legend_bbox, occlusion_bbox, mapping, legend_on_pattern):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid translated-occlusion recolor puzzle")
