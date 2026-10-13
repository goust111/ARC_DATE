from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "new" / "solvers" / "arc_utils.py").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("unable to locate new/solvers/arc_utils.py from this file")


_ensure_arc_utils_on_path()

from new.solvers.arc_utils import Color, bbox_coords, components_mask, crop_bbox  # noqa: E402


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)
PINK = int(Color.PINK)
ORANGE = int(Color.ORANGE)
TEAL = int(Color.TEAL)
MAROON = int(Color.MAROON)

PALETTE_POOL = (BLUE, RED, GREEN, YELLOW, TEAL, MAROON)
SOURCE_MIN = 4
SOURCE_MAX = 5
GRID_H_RANGE = (19, 24)
GRID_W_RANGE = (19, 24)
MATCH_COUNT_RANGE = (3, 6)
MAX_ATTEMPTS = 500


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _all_mask_transforms(mask: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    base = np.asarray(mask, dtype=bool)
    for flipped in (False, True):
        seed = np.fliplr(base) if flipped else base
        for rot_k in range(4):
            transformed = np.rot90(seed, rot_k)
            key = _mask_key(transformed)
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "mask": transformed.copy(),
                    "rot_k": int(rot_k),
                    "flipped": bool(flipped),
                }
            )
    return variants


def _filled_orange_magenta_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    candidates = []
    for coords in components_mask(grid != BG, connectivity=4):
        bbox = tuple(map(int, bbox_coords(coords)))
        crop = crop_bbox(grid, bbox)
        if np.all(crop != BG) and np.all(np.isin(crop, (ORANGE, PINK))) and np.any(crop == ORANGE):
            candidates.append(bbox)
    return candidates


def _find_source_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = _filled_orange_magenta_boxes(grid)
    assert len(candidates) == 1, f"expected exactly one source sprite, got {len(candidates)}"
    return candidates[0]


def _find_palette_bbox_and_colors(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    palette_mask = np.isin(grid, PALETTE_POOL)
    candidates = []
    for coords in components_mask(palette_mask, connectivity=4):
        bbox = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = bbox
        crop = crop_bbox(grid, bbox)
        if c0 != c1:
            continue
        if c0 not in (0, grid.shape[1] - 1):
            continue
        if not (2 <= crop.shape[0] <= 3):
            continue
        if not np.all(crop != BG):
            continue
        candidates.append((bbox, [int(v) for v in crop[:, 0].tolist()]))
    assert len(candidates) == 1, f"expected exactly one palette strip, got {len(candidates)}"
    return candidates[0]


def _find_grey_segment_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for coords in components_mask(grid == GREY, connectivity=4):
        bbox = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = bbox
        if c0 != c1:
            continue
        if not (1 <= (r1 - r0 + 1) <= 2):
            continue
        if not ((r0 == 0 or r1 == grid.shape[0] - 1) and (c0 == 0 or c0 == grid.shape[1] - 1)):
            continue
        candidates.append(bbox)
    assert len(candidates) == 1, f"expected exactly one grey height marker, got {len(candidates)}"
    return candidates[0]


def _find_match_boxes(grid: np.ndarray, mask_variants: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    height, width = grid.shape
    found: dict[tuple[int, int, int, int], None] = {}
    for mask in mask_variants:
        h, w = map(int, mask.shape)
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                region = grid[r0 : r0 + h, c0 : c0 + w]
                if np.all(region[mask] == PINK) and np.all(region[~mask] == BG):
                    found[(int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))] = None
    return sorted(found, key=lambda b: (b[0], b[1], b[2], b[3]))


def _row_major(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))


def _assignment_map(
    boxes: list[tuple[int, int, int, int]],
    palette: list[int],
    repeat_len: int,
) -> dict[tuple[int, int, int, int], int]:
    return {box: int(palette[(idx // repeat_len) % len(palette)]) for idx, box in enumerate(boxes)}


def _render_output(
    grid: np.ndarray,
    matches: list[tuple[int, int, int, int]],
    palette: list[int],
    repeat_len: int,
    source_bbox: tuple[int, int, int, int],
    palette_bbox: tuple[int, int, int, int],
    grey_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    ordered = _row_major(matches)
    for idx, (r0, c0, r1, c1) in enumerate(ordered):
        fill_color = int(palette[(idx // repeat_len) % len(palette)])
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BG] = fill_color

    for r0, c0, r1, c1 in (source_bbox, palette_bbox, grey_bbox):
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match transformed stamp masks, fill each hit rectangle by row-major palette groups, then erase the cues.

    Input structure:
    - The grid has black background, one solid rectangular source sprite, several magenta-only stamp candidates, a vertical palette strip on a side edge, and a short grey vertical segment in a corner.
    - The source sprite is filled with orange except for a few magenta marker pixels, and its full bounding box contains no black cells.
    - Some other magenta groups are exact rotated or mirrored copies of the source marker mask inside otherwise empty rectangles.

    Rule:
    - Find the unique orange-and-magenta filled rectangle and extract its magenta mask over the full source bounding box.
    - Generate every rotation and mirror of that mask, then locate every rectangle where the transformed mask is magenta and every other cell in that rectangle is black.
    - Sort those matches by bounding-box top-left row-major order, read the palette colors from top to bottom, read the grey segment height as the repeat length, fill each matched rectangle's black cells with the grouped palette color, and keep the magenta cells.
    - Finally erase the source sprite, the palette strip, and the grey segment to black.

    Key insight:
    - The visible magenta dots are not filled directly: they define transformed stamp rectangles whose row-major rank controls which palette color group gets applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source_bbox = _find_source_bbox(g)
    source_crop = crop_bbox(g, source_bbox)
    source_mask = source_crop == PINK
    mask_variants = [variant["mask"] for variant in _all_mask_transforms(source_mask)]
    matches = _find_match_boxes(g, mask_variants)

    palette_bbox, palette = _find_palette_bbox_and_colors(g)
    grey_bbox = _find_grey_segment_bbox(g)
    repeat_len = int(grey_bbox[2] - grey_bbox[0] + 1)

    return _render_output(g, matches, palette, repeat_len, source_bbox, palette_bbox, grey_bbox)


def _sample_source_shape(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < 0.65:
        return (SOURCE_MIN, SOURCE_MAX) if rng.random() < 0.5 else (SOURCE_MAX, SOURCE_MIN)
    side = int(rng.integers(SOURCE_MIN, SOURCE_MAX + 1))
    return side, side


def _sample_source_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    weights = np.array([1.0, 2.0, 2.0], dtype=float)
    weights /= weights.sum()
    for _ in range(800):
        magenta_count = int(rng.choice(np.array([2, 3, 4]), p=weights))
        picks = rng.choice(h * w, size=magenta_count, replace=False)
        mask = np.zeros((h, w), dtype=bool)
        mask.flat[picks] = True

        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
            continue
        if len(np.unique(rows)) < 2 or len(np.unique(cols)) < 2:
            continue
        variants = _all_mask_transforms(mask)
        if magenta_count == 2:
            if len(variants) < 4:
                continue
        elif len(variants) < 8:
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate source mask")


def _choose_palette_and_repeat(rng: np.random.Generator) -> tuple[list[int], int]:
    palette_len = int(rng.integers(2, 4))
    palette = [int(v) for v in rng.choice(np.array(PALETTE_POOL), size=palette_len, replace=False).tolist()]
    repeat_len = 2 if rng.random() < 0.65 else 1
    return palette, repeat_len


def _choose_match_count(rng: np.random.Generator, palette_len: int, repeat_len: int) -> int:
    min_matches = max(MATCH_COUNT_RANGE[0], repeat_len + 2)
    choices = list(range(min_matches, MATCH_COUNT_RANGE[1] + 1))
    wrap_needed = repeat_len * palette_len + 1
    if wrap_needed <= MATCH_COUNT_RANGE[1] and rng.random() < 0.55:
        choices = [n for n in choices if n >= wrap_needed]
    return int(rng.choice(np.array(choices)))


def _choose_transform_ids(
    rng: np.random.Generator,
    variants: list[dict[str, object]],
    match_count: int,
) -> list[int]:
    ids = list(range(len(variants)))
    rng.shuffle(ids)

    chosen = [ids[0]]
    non_identity = [idx for idx in ids if idx != ids[0]]
    if non_identity:
        chosen.append(non_identity[0])

    while len(chosen) < min(match_count, 3) and len(chosen) < len(ids):
        chosen.append(ids[len(chosen)])
    while len(chosen) < match_count:
        chosen.append(int(rng.choice(np.array(ids))))

    rng.shuffle(chosen)
    return chosen


def _rect_is_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(reserved.shape[0], int(r0) + h + pad)
    cc1 = min(reserved.shape[1], int(c0) + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    reserved[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = True


def _choose_rect_position(
    rng: np.random.Generator,
    reserved: np.ndarray,
    h: int,
    w: int,
    *,
    margin: int = 1,
    side: int | None = None,
    corner: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    height, width = reserved.shape
    candidates: list[tuple[int, int]] = []

    if corner is not None:
        row_side, col_side = corner
        r_values = [0] if row_side == 0 else [height - h]
        c_values = [0] if col_side == 0 else [width - w]
    elif side is not None:
        c_values = [0] if side == 0 else [width - w]
        row_start = 1
        row_stop = height - h - 1
        if row_stop < row_start:
            return None
        r_values = list(range(row_start, row_stop + 1))
    else:
        row_start = margin
        row_stop = height - h - margin
        col_start = margin
        col_stop = width - w - margin
        if row_stop < row_start or col_stop < col_start:
            return None
        r_values = list(range(row_start, row_stop + 1))
        c_values = list(range(col_start, col_stop + 1))

    for r0 in r_values:
        for c0 in c_values:
            if _rect_is_free(reserved, int(r0), int(c0), h, w):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _render_source(mask: np.ndarray) -> np.ndarray:
    sprite = np.full(mask.shape, ORANGE, dtype=int)
    sprite[mask] = PINK
    return sprite


def _render_magenta_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = PINK
    return patch


def _make_distractor(rng: np.random.Generator, variants: list[dict[str, object]]) -> np.ndarray:
    canonical = { _mask_key(np.asarray(variant["mask"], dtype=bool)) for variant in variants }
    for _ in range(400):
        base = np.array(variants[int(rng.integers(len(variants)))]["mask"], copy=True)
        true_cells = np.argwhere(base)
        false_cells = np.argwhere(~base)
        if rng.random() < 0.5 and len(true_cells) > 1:
            rr, cc = true_cells[int(rng.integers(len(true_cells)))]
            base[int(rr), int(cc)] = False
        else:
            rr, cc = false_cells[int(rng.integers(len(false_cells)))]
            base[int(rr), int(cc)] = True
        if _mask_key(base) not in canonical and base.any():
            return base
    raise RuntimeError("failed to create a distractor mask")


def _apply_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch


def _build_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    source_h, source_w = _sample_source_shape(rng)
    source_mask = _sample_source_mask(rng, source_h, source_w)
    variants = _all_mask_transforms(source_mask)

    palette, repeat_len = _choose_palette_and_repeat(rng)
    match_count = _choose_match_count(rng, len(palette), repeat_len)
    transform_ids = _choose_transform_ids(rng, variants, match_count)

    palette_side = 0 if rng.random() < 0.5 else 1
    palette_pos = _choose_rect_position(rng, reserved, len(palette), 1, side=palette_side)
    if palette_pos is None:
        raise RuntimeError("failed to place palette strip")
    palette_r, palette_c = palette_pos
    palette_patch = np.array(palette, dtype=int).reshape(-1, 1)
    _apply_patch(grid, palette_patch, palette_r, palette_c)
    _reserve_rect(reserved, palette_r, palette_c, len(palette), 1)
    palette_bbox = (palette_r, palette_c, palette_r + len(palette) - 1, palette_c)

    grey_col_side = 1 - palette_side
    grey_row_side = 0 if rng.random() < 0.5 else 1
    grey_height = repeat_len
    grey_pos = _choose_rect_position(rng, reserved, grey_height, 1, corner=(grey_row_side, grey_col_side))
    if grey_pos is None:
        raise RuntimeError("failed to place grey height marker")
    grey_r, grey_c = grey_pos
    grey_patch = np.full((grey_height, 1), GREY, dtype=int)
    _apply_patch(grid, grey_patch, grey_r, grey_c)
    _reserve_rect(reserved, grey_r, grey_c, grey_height, 1)
    grey_bbox = (grey_r, grey_c, grey_r + grey_height - 1, grey_c)

    source_pos = _choose_rect_position(rng, reserved, source_h, source_w, margin=1)
    if source_pos is None:
        raise RuntimeError("failed to place source sprite")
    source_r, source_c = source_pos
    source_patch = _render_source(source_mask)
    _apply_patch(grid, source_patch, source_r, source_c)
    _reserve_rect(reserved, source_r, source_c, source_h, source_w)
    source_bbox = (source_r, source_c, source_r + source_h - 1, source_c + source_w - 1)

    matches: list[dict[str, object]] = []
    for transform_id in transform_ids:
        mask = np.asarray(variants[int(transform_id)]["mask"], dtype=bool)
        patch = _render_magenta_mask(mask)
        pos = _choose_rect_position(rng, reserved, patch.shape[0], patch.shape[1], margin=1)
        if pos is None:
            raise RuntimeError("failed to place a match rectangle")
        r0, c0 = pos
        _apply_patch(grid, patch, r0, c0)
        _reserve_rect(reserved, r0, c0, patch.shape[0], patch.shape[1])
        matches.append(
            {
                "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1),
                "transform_id": int(transform_id),
                "mask": mask.copy(),
            }
        )

    distractor_count = int(rng.integers(1, 3))
    distractors: list[tuple[int, int, int, int]] = []
    for _ in range(distractor_count):
        mask = _make_distractor(rng, variants)
        patch = _render_magenta_mask(mask)
        pos = _choose_rect_position(rng, reserved, patch.shape[0], patch.shape[1], margin=1)
        if pos is None:
            raise RuntimeError("failed to place a distractor")
        r0, c0 = pos
        _apply_patch(grid, patch, r0, c0)
        _reserve_rect(reserved, r0, c0, patch.shape[0], patch.shape[1])
        distractors.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))

    match_boxes = [tuple(map(int, item["bbox"])) for item in matches]
    expected = _render_output(grid, match_boxes, palette, repeat_len, source_bbox, palette_bbox, grey_bbox)
    metadata = {
        "source_bbox": source_bbox,
        "source_mask": source_mask.copy(),
        "variants": [np.asarray(variant["mask"], dtype=bool).copy() for variant in variants],
        "palette": list(palette),
        "repeat_len": int(repeat_len),
        "palette_bbox": palette_bbox,
        "grey_bbox": grey_bbox,
        "matches": matches,
        "distractors": list(distractors),
        "expected": expected.copy(),
    }
    return grid, expected, metadata


def _validate_example(grid: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    source_bbox = tuple(map(int, meta["source_bbox"]))
    palette = [int(v) for v in meta["palette"]]
    repeat_len = int(meta["repeat_len"])
    palette_bbox = tuple(map(int, meta["palette_bbox"]))
    grey_bbox = tuple(map(int, meta["grey_bbox"]))
    variants = [np.asarray(mask, dtype=bool) for mask in meta["variants"]]
    match_boxes = [tuple(map(int, item["bbox"])) for item in meta["matches"]]
    transform_ids = [int(item["transform_id"]) for item in meta["matches"]]
    distractors = [tuple(map(int, box)) for box in meta["distractors"]]

    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(_filled_orange_magenta_boxes(grid)) != 1:
        return False
    if _find_source_bbox(grid) != source_bbox:
        return False
    palette_found_bbox, palette_found = _find_palette_bbox_and_colors(grid)
    if palette_found_bbox != palette_bbox or palette_found != palette:
        return False
    if _find_grey_segment_bbox(grid) != grey_bbox:
        return False

    found_matches = _find_match_boxes(grid, variants)
    if found_matches != _row_major(match_boxes):
        return False
    if any(box in found_matches for box in distractors):
        return False

    if len(set(transform_ids)) < 2:
        return False
    if not any(transform_id != 0 for transform_id in transform_ids):
        return False

    ordered = _row_major(match_boxes)
    rows = {box[0] for box in ordered}
    cols = {box[1] for box in ordered}
    if len(rows) < 2 or len(cols) < 2:
        return False

    row_assignment = _assignment_map(ordered, palette, repeat_len)
    if len(set(row_assignment.values())) < 2:
        return False

    if _assignment_map(ordered, palette, 1 if repeat_len == 2 else 2) == row_assignment:
        return False
    if _assignment_map(ordered, list(reversed(palette)), repeat_len) == row_assignment:
        return False

    col_major = sorted(match_boxes, key=lambda b: (b[1], b[0], b[2], b[3]))
    if _assignment_map(col_major, palette, repeat_len) == row_assignment:
        return False

    src_r0, src_c0, _src_r1, _src_c1 = source_bbox
    dist_order = sorted(
        match_boxes,
        key=lambda b: (abs(b[0] - src_r0) + abs(b[1] - src_c0), b[0], b[1]),
    )
    if _assignment_map(dist_order, palette, repeat_len) == row_assignment:
        return False

    rebuilt = _render_output(grid, match_boxes, palette, repeat_len, source_bbox, palette_bbox, grey_bbox)
    if not np.array_equal(rebuilt, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected, meta = _build_example(rng)
        except RuntimeError:
            continue
        if _validate_example(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid transformed-mask palette example")
