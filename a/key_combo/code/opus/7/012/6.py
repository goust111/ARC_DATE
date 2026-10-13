from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    return [
        mask.copy(),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.flipud(mask),
        np.fliplr(mask),
        mask.T.copy(),
        np.flipud(np.fliplr(mask)).T.copy(),
    ]


def _is_asymmetric(mask: np.ndarray) -> bool:
    seen = {
        (oriented.shape, oriented.astype(np.uint8).tobytes())
        for oriented in _all_orientations(mask)
    }
    return len(seen) == 8


def _build_sprite_catalog() -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for bits in itertools.product([False, True], repeat=9):
        mask = np.array(bits, dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        cropped = _tight_crop(mask)
        if bbox_h((0, 0, cropped.shape[0] - 1, cropped.shape[1] - 1)) > 3:
            continue
        if bbox_w((0, 0, cropped.shape[0] - 1, cropped.shape[1] - 1)) > 3:
            continue
        if not _is_asymmetric(cropped):
            continue
        key = (cropped.shape, cropped.astype(np.uint8).tobytes())
        if all(key != (s.shape, s.astype(np.uint8).tobytes()) for s in sprites):
            sprites.append(cropped)
    assert sprites
    return sprites


SPRITE_CATALOG = _build_sprite_catalog()

LAYOUTS: tuple[tuple[int, int], ...] = (
    (2, 3),
    (3, 2),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
    (1, 7),
    (7, 1),
)
MAX_GENERATION_ATTEMPTS = 500


def _region_records(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in components_mask(grid != Color.RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(coords.shape[0]) == h * w, "regions must be full rectangles"
        regions.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(h * w),
                "h": int(h),
                "w": int(w),
            }
        )
    regions.sort(key=lambda reg: (reg["bbox"][0], reg["bbox"][1]))
    return regions


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    regions = _region_records(g)
    assert 5 <= len(regions) <= 7, "expected 5-7 regions"

    areas = [reg["area"] for reg in regions]
    min_area = min(areas)
    template_candidates = [i for i, area in enumerate(areas) if area == min_area]
    assert len(template_candidates) == 1, "smallest region must be unique"
    template_idx = template_candidates[0]
    template = regions[template_idx]

    tr0, tc0, tr1, tc1 = template["bbox"]
    template_view = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    green_mask = template_view == Color.GREEN
    green_components = components_mask(green_mask, connectivity=4)
    assert len(green_components) == 1, "template region must contain one green component"
    green_coords = green_components[0]
    area = int(green_coords.shape[0])
    assert 3 <= area <= 6, "template sprite area must be 3-6"
    gb = bbox_coords(green_coords)
    gh = bbox_h(gb)
    gw = bbox_w(gb)
    assert gh <= 3 and gw <= 3, "template bbox must be <= 3x3"

    sprite_mask = np.zeros((gh, gw), dtype=bool)
    for rr, cc in green_coords:
        sprite_mask[int(rr) - gb[0], int(cc) - gb[1]] = True
    assert _is_asymmetric(sprite_mask), "template sprite must be asymmetric"

    template["sprite_mask"] = sprite_mask
    template["sprite_offset"] = (int(gb[0]), int(gb[1]))

    non_template_counts = []
    for idx, reg in enumerate(regions):
        r0, c0, r1, c1 = reg["bbox"]
        view = g[r0 : r1 + 1, c0 : c1 + 1]
        allowed = {Color.BLACK, Color.BLUE}
        if idx == template_idx:
            assert set(map(int, np.unique(view))) <= {Color.BLACK, Color.GREEN}
            continue
        assert set(map(int, np.unique(view))) <= allowed
        reg["marker_count"] = int(np.sum(view == Color.BLUE))
        non_template_counts.append(reg["marker_count"])

    assert any(count == 0 for count in non_template_counts), "need a zero-marker branch"
    assert any(count == 1 for count in non_template_counts), "need a single-marker branch"
    assert any(count >= 2 for count in non_template_counts), "need a multi-marker branch"
    return {"grid": g, "regions": regions, "template_idx": template_idx}


def _tiled_mask(region_h: int, region_w: int, sprite_mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = sprite_mask.shape
    off_r, off_c = map(int, offset)
    mask = np.zeros((region_h, region_w), dtype=bool)
    for base_r in range(off_r, region_h, tile_h):
        for base_c in range(off_c, region_w, tile_w):
            for sr, sc in np.argwhere(sprite_mask):
                rr = int(base_r + sr)
                cc = int(base_c + sc)
                if 0 <= rr < region_h and 0 <= cc < region_w:
                    mask[rr, cc] = True
    return mask


def _expected_region(view: np.ndarray, marker_count: int, sprite_mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    out = view.copy()
    if marker_count == 0:
        return out
    if marker_count >= 2:
        return np.full(view.shape, Color.GREY, dtype=int)
    tiled = _tiled_mask(view.shape[0], view.shape[1], sprite_mask, offset)
    out[tiled] = Color.BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count blue markers per divider region, then keep, tile from the smallest green template region, or repaint grey.

    Input structure:
    - A black background is partitioned into 5-7 rectangular regions by full-span red divider lines.
    - Exactly one uniquely smallest region contains one asymmetric green 4-connected sprite whose bbox is at most 3x3.
    - Every other region contains only black cells plus 0-3 isolated blue marker pixels.

    Rule:
    - Identify the uniquely smallest region as the template region and keep it unchanged.
    - For each other region, count its blue markers: 0 keeps the region unchanged; 1 repeats the template sprite in blue using the template sprite bbox size as the row/column stride and the template sprite's offset inside its own region as the repeated phase; 2 or more repaints the whole region grey.
    - Tiled copies are clipped at the region boundary, original uncovered blue marker pixels stay blue, and all red divider lines stay red.

    Key insight:
    - The blue-marker count is a three-way branch selector, but the single-marker branch cannot be solved locally because the tiled blue pattern must be copied from the uniquely smallest green template region.
    """
    spec = _analyze(grid)
    g = spec["grid"]
    regions = spec["regions"]
    template = regions[spec["template_idx"]]
    sprite_mask = template["sprite_mask"]
    offset = template["sprite_offset"]

    out = g.copy()
    for idx, reg in enumerate(regions):
        if idx == spec["template_idx"]:
            continue
        r0, c0, r1, c1 = reg["bbox"]
        view = g[r0 : r1 + 1, c0 : c1 + 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = _expected_region(view, reg["marker_count"], sprite_mask, offset)
    return out


def _layout_allowed(sprite_h: int, sprite_w: int, layout: tuple[int, int]) -> bool:
    if layout in {(1, 6), (1, 7)}:
        return sprite_w <= 2
    if layout in {(6, 1), (7, 1)}:
        return sprite_h <= 2
    return True


def _pick_layout(sprite_h: int, sprite_w: int) -> tuple[int, int]:
    weighted = [
        (2, 3),
        (2, 3),
        (3, 2),
        (3, 2),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (1, 7),
        (7, 1),
    ]
    choices = [layout for layout in weighted if _layout_allowed(sprite_h, sprite_w, layout)]
    return random.choice(choices)


def _sample_sizes(count: int, small_index: int, small_lo: int, small_hi: int, other_lo: int, other_hi: int) -> list[int]:
    sizes = []
    small = random.randint(small_lo, small_hi)
    for idx in range(count):
        if idx == small_index:
            sizes.append(small)
        else:
            lo = max(other_lo, small + 1)
            hi = max(lo, other_hi)
            sizes.append(random.randint(lo, hi))
    return sizes


def _make_partition(sprite_h: int, sprite_w: int) -> tuple[list[int], list[int], int, int] | None:
    for _ in range(80):
        rows, cols = _pick_layout(sprite_h, sprite_w)
        tr = random.randrange(rows)
        tc = random.randrange(cols)

        if (rows, cols) == (2, 3):
            row_sizes = _sample_sizes(rows, tr, max(sprite_h + 2, 5), max(sprite_h + 3, 6), 7, 10)
            col_sizes = _sample_sizes(cols, tc, max(sprite_w + 2, 5), max(sprite_w + 3, 6), 6, 8)
        elif (rows, cols) == (3, 2):
            row_sizes = _sample_sizes(rows, tr, max(sprite_h + 2, 5), max(sprite_h + 3, 6), 6, 8)
            col_sizes = _sample_sizes(cols, tc, max(sprite_w + 2, 5), max(sprite_w + 3, 6), 7, 10)
        elif rows == 1 and cols == 5:
            row_sizes = [random.randint(max(sprite_h + 2, 16), 20)]
            col_sizes = _sample_sizes(cols, tc, max(sprite_w + 1, 2), max(sprite_w + 1, 3), 4, 5)
        elif rows == 5 and cols == 1:
            row_sizes = _sample_sizes(rows, tr, max(sprite_h + 1, 2), max(sprite_h + 1, 3), 4, 5)
            col_sizes = [random.randint(max(sprite_w + 2, 16), 20)]
        elif rows == 1 and cols == 6:
            row_sizes = [random.randint(max(sprite_h + 2, 16), 20)]
            col_sizes = _sample_sizes(cols, tc, max(sprite_w, 2), max(sprite_w, 2), 3, 4)
        elif rows == 6 and cols == 1:
            row_sizes = _sample_sizes(rows, tr, max(sprite_h, 2), max(sprite_h, 2), 3, 4)
            col_sizes = [random.randint(max(sprite_w + 2, 16), 20)]
        elif rows == 1 and cols == 7:
            row_sizes = [random.randint(max(sprite_h + 2, 16), 20)]
            col_sizes = _sample_sizes(cols, tc, max(sprite_w, 2), max(sprite_w, 2), 3, 4)
        elif rows == 7 and cols == 1:
            row_sizes = _sample_sizes(rows, tr, max(sprite_h, 2), max(sprite_h, 2), 3, 4)
            col_sizes = [random.randint(max(sprite_w + 2, 16), 20)]
        else:
            continue

        height = sum(row_sizes) + rows - 1
        width = sum(col_sizes) + cols - 1
        if 16 <= height <= 30 and 16 <= width <= 30:
            return row_sizes, col_sizes, tr, tc
    return None


def _band_starts(sizes: list[int]) -> list[int]:
    starts = []
    cursor = 0
    for idx, size in enumerate(sizes):
        starts.append(cursor)
        cursor += size
        if idx + 1 < len(sizes):
            cursor += 1
    return starts


def _region_bboxes(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    regions = []
    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)
    for r0, h in zip(row_starts, row_sizes):
        for c0, w in zip(col_starts, col_sizes):
            regions.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
    return regions


def _choose_offset(region_h: int, region_w: int, sprite_h: int, sprite_w: int) -> tuple[int, int]:
    row_slack = region_h - sprite_h
    col_slack = region_w - sprite_w
    options = [(r, c) for r in range(row_slack + 1) for c in range(col_slack + 1)]
    assert options
    nonzero = [(r, c) for r, c in options if r != 0 or c != 0]
    return random.choice(nonzero or options)


def _can_show_clipping(region_h: int, region_w: int, sprite_h: int, sprite_w: int, offset: tuple[int, int]) -> bool:
    off_r, off_c = offset
    enough = region_h >= off_r + sprite_h and region_w >= off_c + sprite_w
    if not enough:
        return False
    return (region_h - off_r) % sprite_h != 0 or (region_w - off_c) % sprite_w != 0


def _uncovered_positions(region_h: int, region_w: int, sprite_mask: np.ndarray, offset: tuple[int, int]) -> list[tuple[int, int]]:
    tiled = _tiled_mask(region_h, region_w, sprite_mask, offset)
    return [tuple(map(int, rc)) for rc in np.argwhere(~tiled)]


def _assign_counts(
    bboxes: list[tuple[int, int, int, int]],
    template_idx: int,
    sprite_mask: np.ndarray,
    offset: tuple[int, int],
) -> dict[int, int] | None:
    sprite_h, sprite_w = sprite_mask.shape
    non_template = [idx for idx in range(len(bboxes)) if idx != template_idx]
    single_candidates = []
    zero_candidates = []
    multi_candidates = []
    for idx in non_template:
        r0, c0, r1, c1 = bboxes[idx]
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        if region_h >= 2 * sprite_h and region_w >= 2 * sprite_w:
            uncovered = _uncovered_positions(region_h, region_w, sprite_mask, offset)
            if uncovered and _can_show_clipping(region_h, region_w, sprite_h, sprite_w, offset):
                single_candidates.append(idx)
        if region_h * region_w >= 4:
            zero_candidates.append(idx)
            multi_candidates.append(idx)
    if not single_candidates:
        return None

    single_idx = random.choice(single_candidates)
    remaining = [idx for idx in non_template if idx != single_idx]
    if len(remaining) < 2:
        return None
    zero_idx = random.choice(remaining)
    remaining = [idx for idx in remaining if idx != zero_idx]
    multi_idx = random.choice(remaining)

    counts = {idx: random.choice([0, 1, 2, 3]) for idx in non_template}
    counts[single_idx] = 1
    counts[zero_idx] = 0
    counts[multi_idx] = random.choice([2, 3])
    if not any(v == 0 for v in counts.values()):
        return None
    if not any(v == 1 for v in counts.values()):
        return None
    if not any(v >= 2 for v in counts.values()):
        return None
    return counts


def _place_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    count: int,
    sprite_mask: np.ndarray,
    offset: tuple[int, int],
    force_uncovered: bool,
) -> bool:
    r0, c0, r1, c1 = bbox
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    positions = [(r, c) for r in range(region_h) for c in range(region_w)]
    if count == 0:
        return True
    if force_uncovered:
        preferred = _uncovered_positions(region_h, region_w, sprite_mask, offset)
        if not preferred:
            return False
        first = random.choice(preferred)
        chosen = [first]
        available = [pos for pos in positions if pos != first]
    else:
        available = positions[:]
        chosen = []
    if count - len(chosen) > len(available):
        return False
    chosen.extend(random.sample(available, count - len(chosen)))
    for rr, cc in chosen:
        grid[r0 + rr, c0 + cc] = Color.BLUE
    return True


def _generate_attempt() -> np.ndarray | None:
    sprite_mask = random.choice(SPRITE_CATALOG).copy()
    sprite_h, sprite_w = sprite_mask.shape
    partition = _make_partition(sprite_h, sprite_w)
    if partition is None:
        return None
    row_sizes, col_sizes, tr, tc = partition
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((height, width), Color.BLACK, dtype=int)

    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)
    for idx in range(1, len(row_sizes)):
        divider_r = row_starts[idx] - 1
        grid[divider_r, :] = Color.RED
    for idx in range(1, len(col_sizes)):
        divider_c = col_starts[idx] - 1
        grid[:, divider_c] = Color.RED

    bboxes = _region_bboxes(row_sizes, col_sizes)
    template_idx = tr * len(col_sizes) + tc
    template_bbox = bboxes[template_idx]
    th = template_bbox[2] - template_bbox[0] + 1
    tw = template_bbox[3] - template_bbox[1] + 1
    offset = _choose_offset(th, tw, sprite_h, sprite_w)

    counts = _assign_counts(bboxes, template_idx, sprite_mask, offset)
    if counts is None:
        return None

    for rr, cc in np.argwhere(sprite_mask):
        grid[template_bbox[0] + offset[0] + rr, template_bbox[1] + offset[1] + cc] = Color.GREEN

    forced_single_regions = [idx for idx, count in counts.items() if count == 1]
    assert forced_single_regions
    forced_idx = random.choice(forced_single_regions)
    for idx, count in counts.items():
        ok = _place_markers(
            grid,
            bboxes[idx],
            count,
            sprite_mask,
            offset,
            force_uncovered=(idx == forced_idx),
        )
        if not ok:
            return None

    if not _semantic_check(grid):
        return None
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        spec = _analyze(grid)
    except AssertionError:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    g = np.asarray(grid, dtype=int)
    regions = spec["regions"]
    template = regions[spec["template_idx"]]
    sprite_mask = template["sprite_mask"]
    offset = template["sprite_offset"]

    if not np.array_equal(out[g == Color.RED], g[g == Color.RED]):
        return False

    saw_single_residual = False
    saw_single_clip = False
    for idx, reg in enumerate(regions):
        r0, c0, r1, c1 = reg["bbox"]
        in_view = g[r0 : r1 + 1, c0 : c1 + 1]
        out_view = out[r0 : r1 + 1, c0 : c1 + 1]
        if idx == spec["template_idx"]:
            if not np.array_equal(in_view, out_view):
                return False
            continue
        expected = _expected_region(in_view, reg["marker_count"], sprite_mask, offset)
        if not np.array_equal(out_view, expected):
            return False
        if reg["marker_count"] == 1:
            tiled = _tiled_mask(reg["h"], reg["w"], sprite_mask, offset)
            blue_in = in_view == Color.BLUE
            if np.any(blue_in & ~tiled):
                saw_single_residual = True
            if _can_show_clipping(reg["h"], reg["w"], sprite_mask.shape[0], sprite_mask.shape[1], offset):
                saw_single_clip = True
    return saw_single_residual and saw_single_clip


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
