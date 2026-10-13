from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite


BG = 0
PALETTE = tuple(range(1, 10))
GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
TOTAL_OBJECTS_RANGE = (5, 7)
AREA_RANGE = (3, 6)
PLACEMENT_GAP = 2
GEN_ATTEMPTS = 500


def _signature(crop: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in crop)


def _foreground_colors(crop: np.ndarray, bg: int) -> frozenset[int]:
    return frozenset(int(v) for v in np.unique(crop) if int(v) != int(bg))


def _patch_fg_coords(patch: np.ndarray, bg: int = BG) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]


def _shape_of(patch: np.ndarray) -> tuple[int, int]:
    return int(patch.shape[0]), int(patch.shape[1])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(objects) >= 3, "expected several sprites"

    records: list[dict] = []
    for obj in objects:
        crop = crop_bbox(g, obj["bbox"])
        records.append(
            {
                "bbox": tuple(int(x) for x in obj["bbox"]),
                "area": int(obj["area"]),
                "crop": crop,
                "signature": _signature(crop),
                "colors": _foreground_colors(crop, bg),
            }
        )

    counts = Counter(record["signature"] for record in records)
    repeated_groups = [sig for sig, count in counts.items() if count == max(counts.values()) and count > 1]
    assert len(repeated_groups) == 1, "expected exactly one repeated pattern group"
    repeated_signature = repeated_groups[0]
    k = int(counts[repeated_signature])
    repeated_objects = [record for record in records if record["signature"] == repeated_signature]
    repeated_colors = repeated_objects[0]["colors"]

    singletons = [record for record in records if counts[record["signature"]] == 1]
    assert singletons, "expected singleton sprites besides the repeated group"
    target_candidates = [record for record in singletons if record["colors"].isdisjoint(repeated_colors)]
    assert len(target_candidates) == 1, "expected exactly one color-disjoint singleton"
    target = target_candidates[0]

    for record in singletons:
        if record["signature"] == target["signature"]:
            continue
        assert not record["colors"].isdisjoint(repeated_colors), "non-target singleton must share repeated-group colors"

    return {
        "bg": int(bg),
        "objects": records,
        "counts": counts,
        "singletons": singletons,
        "repeated_signature": repeated_signature,
        "repeated_colors": repeated_colors,
        "K": k,
        "target": target,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the singleton sprite whose colors avoid the duplicated pattern's palette, and enlarge it by the duplicate frequency.

    Input structure:
    - The grid has a black background and 5 to 7 separated 4-connected multicolor sprites.
    - Exactly one cropped colored sprite pattern repeats K times, where K is 2 or 3.
    - Every other sprite pattern appears once, and exactly one singleton sprite uses no foreground color from the repeated group.

    Rule:
    - Crop every sprite to its colored bounding box, keeping any background cells inside that box.
    - Group identical crops to find the repeated pattern and its occurrence count K.
    - Among the singleton crops, select the unique sprite whose foreground-color set is disjoint from the repeated group's color set.
    - Replace each pixel of that singleton crop with a K x K block of the same color, paste it at the singleton's original top-left corner, and erase all other sprites to background.

    Key insight:
    - The enlargement factor is not encoded directly; it is hidden in the repeated crop frequency, so the solver must do exact crop matching before the color-based outlier test.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    target = info["target"]
    r0, c0, _r1, _c1 = target["bbox"]
    scaled = scale_sprite(target["crop"], info["K"])

    out = np.full(g.shape, info["bg"], dtype=int)
    paste(out, scaled, r0, c0)
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_mask(rng: random.Random, area: int) -> np.ndarray:
    h = w = 4
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = sorted({nb for r, c in cells for nb in _neighbors4(r, c, h, w) if nb not in cells})
        cells.add(rng.choice(frontier))

    coords = np.array(sorted(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _paint_mask(mask: np.ndarray, colors: tuple[int, ...], rng: random.Random) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(coords)
    crop = np.full(mask.shape, BG, dtype=int)
    color_list = [int(c) for c in colors]
    assert len(color_list) <= len(coords)

    for color, (r, c) in zip(color_list, coords):
        crop[r, c] = color
    for r, c in coords[len(color_list) :]:
        crop[r, c] = rng.choice(color_list)
    return crop


def _make_unique_crop(
    rng: random.Random,
    colors: tuple[int, ...],
    used_signatures: set[tuple[tuple[int, ...], ...]],
    *,
    match_area_or_shape: tuple[int, tuple[int, int]] | None = None,
) -> np.ndarray:
    for _ in range(300):
        area = rng.randint(*AREA_RANGE)
        mask = _random_connected_mask(rng, area)
        if match_area_or_shape is not None:
            target_area, target_shape = match_area_or_shape
            if int(mask.sum()) != int(target_area) and _shape_of(mask) != target_shape:
                continue
        crop = _paint_mask(mask, colors, rng)
        sig = _signature(crop)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        return crop
    raise RuntimeError("failed to build a unique crop")


def _block_object(blocked: set[tuple[int, int]], fg_coords: list[tuple[int, int]], shape: tuple[int, int], padding: int) -> None:
    h, w = shape
    for r, c in fg_coords:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))


def _block_rectangle(blocked: set[tuple[int, int]], shape: tuple[int, int], top: int, left: int, height: int, width: int) -> None:
    h, w = shape
    for r in range(top, min(top + height, h)):
        for c in range(left, min(left + width, w)):
            blocked.add((r, c))


def _candidate_positions(
    blocked: set[tuple[int, int]],
    patch: np.ndarray,
    shape: tuple[int, int],
    *,
    max_row: int | None = None,
    max_col: int | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch.shape
    fg = _patch_fg_coords(patch)
    row_limit = h - ph if max_row is None else min(h - ph, max_row)
    col_limit = w - pw if max_col is None else min(w - pw, max_col)
    candidates: list[tuple[int, int]] = []
    for r0 in range(row_limit + 1):
        for c0 in range(col_limit + 1):
            if all((r0 + dr, c0 + dc) not in blocked for dr, dc in fg):
                candidates.append((r0, c0))
    return candidates


def _place_patch(
    canvas: np.ndarray,
    blocked: set[tuple[int, int]],
    patch: np.ndarray,
    rng: random.Random,
    *,
    max_row: int | None = None,
    max_col: int | None = None,
) -> tuple[int, int]:
    candidates = _candidate_positions(blocked, patch, canvas.shape, max_row=max_row, max_col=max_col)
    if not candidates:
        raise ValueError("no free placement")
    r0, c0 = rng.choice(candidates)
    paste(canvas, patch, r0, c0, transparent=BG)
    abs_fg = [(r0 + dr, c0 + dc) for dr, dc in _patch_fg_coords(patch)]
    _block_object(blocked, abs_fg, canvas.shape, PLACEMENT_GAP)
    return int(r0), int(c0)


def _sample_color_sets(rng: random.Random, singleton_count: int) -> tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, ...]]]:
    repeated_colors = tuple(rng.sample(PALETTE, 2))
    remaining = [color for color in PALETTE if color not in repeated_colors]
    anomaly_size = rng.choice((1, 2))
    anomaly_colors = tuple(rng.sample(remaining, anomaly_size))

    other_sets: list[tuple[int, ...]] = []
    first_shared = tuple(sorted({rng.choice(repeated_colors), rng.choice(anomaly_colors)}))
    other_sets.append(first_shared)
    for _ in range(singleton_count - 2):
        share = rng.choice(repeated_colors)
        options = [color for color in PALETTE if color != share]
        if rng.random() < 0.35:
            colors = (share,)
        else:
            colors = tuple(sorted({share, rng.choice(options)}))
        other_sets.append(colors)
    rng.shuffle(other_sets)
    return repeated_colors, anomaly_colors, other_sets


def _build_scene(rng: random.Random) -> tuple[np.ndarray, dict]:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    k = rng.choice((2, 3))
    total_objects = rng.randint(max(TOTAL_OBJECTS_RANGE[0], k + 2), TOTAL_OBJECTS_RANGE[1])
    singleton_count = total_objects - k

    repeated_colors, anomaly_colors, other_singleton_colors = _sample_color_sets(rng, singleton_count)
    used_signatures: set[tuple[tuple[int, ...], ...]] = set()

    repeated_crop = _make_unique_crop(rng, repeated_colors, used_signatures)
    anomaly_crop = _make_unique_crop(rng, anomaly_colors, used_signatures)
    match_anchor = (int(np.count_nonzero(anomaly_crop)), _shape_of(anomaly_crop))

    singleton_crops = [anomaly_crop]
    for idx, colors in enumerate(other_singleton_colors):
        crop = _make_unique_crop(
            rng,
            colors,
            used_signatures,
            match_area_or_shape=match_anchor if idx == 0 else None,
        )
        singleton_crops.append(crop)

    canvas = np.zeros((h, w), dtype=int)
    blocked: set[tuple[int, int]] = set()

    scaled_h, scaled_w = anomaly_crop.shape[0] * k, anomaly_crop.shape[1] * k
    anomaly_pos = _place_patch(
        canvas,
        blocked,
        anomaly_crop,
        rng,
        max_row=h - scaled_h,
        max_col=w - scaled_w,
    )
    _block_rectangle(blocked, canvas.shape, anomaly_pos[0], anomaly_pos[1], scaled_h, scaled_w)

    for _ in range(k):
        _place_patch(canvas, blocked, repeated_crop, rng)
    for crop in singleton_crops[1:]:
        _place_patch(canvas, blocked, crop, rng)

    meta = {
        "K": k,
        "total_objects": total_objects,
        "repeated_colors": frozenset(repeated_colors),
        "target_signature": _signature(anomaly_crop),
        "target_shape": _shape_of(anomaly_crop),
        "target_area": int(np.count_nonzero(anomaly_crop)),
        "target_colors": frozenset(anomaly_colors),
    }
    return canvas, meta


def _validate_scene(grid: np.ndarray, meta: dict) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    if len(info["objects"]) != meta["total_objects"]:
        return False
    if info["K"] != meta["K"]:
        return False
    if info["repeated_colors"] != meta["repeated_colors"]:
        return False
    if info["target"]["signature"] != meta["target_signature"]:
        return False

    target = info["target"]
    peers = [record for record in info["singletons"] if record["signature"] != target["signature"]]
    if not peers:
        return False
    if not any(record["area"] == target["area"] or _shape_of(record["crop"]) == _shape_of(target["crop"]) for record in peers):
        return False
    if not any(record["colors"] & target["colors"] for record in peers):
        return False

    expected = np.zeros_like(grid)
    r0, c0, _r1, _c1 = target["bbox"]
    paste(expected, scale_sprite(target["crop"], info["K"]), r0, c0)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid, meta = _build_scene(rng)
        if _validate_scene(grid, meta):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
