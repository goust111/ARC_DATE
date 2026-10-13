from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
SEP = 4
MARKER = 9
SPRITE_COLORS = (1, 2, 3, 5, 6, 7, 8)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = [-1, *sorted(int(v) for v in separators), size]
    return [(cuts[i] + 1, cuts[i + 1]) for i in range(len(cuts) - 1)]


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 partition"

    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3

    regions: list[dict] = []
    idx = 0
    for rb, (r0, r1) in enumerate(row_bands):
        for cb, (c0, c1) in enumerate(col_bands):
            sub = g[r0:r1, c0:c1]
            assert SEP not in sub, "separators must stay outside regions"

            marker_count = int(np.sum(sub == MARKER))
            sprite_mask = (sub != BG) & (sub != MARKER)
            colors = sorted(set(int(v) for v in sub[sprite_mask]))
            assert len(colors) == 1, "each region must contain exactly one sprite color"
            comps = _components(sprite_mask)
            assert len(comps) == 1, "sprite must be one 4-connected component"
            assert 3 <= len(comps[0]) <= 6, "sprite area must be 3..6"

            box = _bbox(sprite_mask)
            assert box is not None
            patch = _crop_bbox(sub, box)

            regions.append(
                {
                    "index": idx,
                    "grid_pos": (rb, cb),
                    "region_box": (r0, c0, r1 - 1, c1 - 1),
                    "count": marker_count,
                    "color": colors[0],
                    "patch": patch,
                    "sprite_box_local": box,
                }
            )
            idx += 1
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the board by yellow separators, count maroon markers, sort the sprites, and mirror only the unique maximum-count one.

    Input structure:
    - The background is black.
    - Two full yellow rows and two full yellow columns split the grid into a 3x3 array of rectangular regions.
    - Each region contains one small monochrome 4-connected sprite plus zero to three maroon marker pixels on background cells.

    Rule:
    - Count the maroon markers inside each of the nine regions.
    - Discard every region whose marker count is zero.
    - Crop each remaining region to the sprite's bounding box, sort the cropped sprites by increasing marker count with reading-order tie-breaks, and horizontally mirror only the sprite from the unique strict maximum-count region.
    - Stack the resulting sprites from top to bottom with one blank row between consecutive sprites, left aligned in an output whose width is the maximum cropped width.

    Key insight:
    - The marker count has two simultaneous roles: it orders the participating sprites and also identifies the single sprite that must be mirrored.
    """
    regions = _extract_regions(np.asarray(grid, dtype=int))
    counts = [region["count"] for region in regions]
    max_count = max(counts)
    assert counts.count(max_count) == 1, "expected a unique maximum marker count"

    kept = [region for region in regions if region["count"] > 0]
    assert kept, "expected at least one non-zero region"
    kept.sort(key=lambda region: (region["count"], region["index"]))

    patches: list[np.ndarray] = []
    max_index = counts.index(max_count)
    for region in kept:
        patch = region["patch"].copy()
        if region["index"] == max_index:
            patch = np.fliplr(patch)
        patches.append(patch)

    out_h = sum(p.shape[0] for p in patches) + (len(patches) - 1)
    out_w = max(p.shape[1] for p in patches)
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = 0
    for i, patch in enumerate(patches):
        h, w = patch.shape
        out[cursor : cursor + h, :w] = patch
        cursor += h
        if i != len(patches) - 1:
            cursor += 1
    return out


def _valid_sprite_masks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    asym_masks: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1, 1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for k in range(h * w):
                    if bits & (1 << k):
                        mask[k // w, k % w] = True
                area = int(mask.sum())
                if not (3 <= area <= 6):
                    continue
                if area == h * w:
                    continue
                if len(_components(mask)) != 1:
                    continue
                box = _bbox(mask)
                assert box is not None
                if box != (0, 0, h - 1, w - 1):
                    continue
                masks.append(mask)
                if not np.array_equal(mask, np.fliplr(mask)):
                    asym_masks.append(mask)
    return masks, asym_masks


SPRITE_MASKS, H_ASYM_MASKS = _valid_sprite_masks()
COUNT_PATTERNS = (
    (0, 0, 1, 1, 1, 2, 2, 2, 3),
    (0, 0, 0, 1, 1, 1, 2, 2, 3),
    (0, 0, 0, 0, 1, 1, 2, 2, 3),
    (0, 0, 0, 1, 1, 1, 1, 1, 2),
    (0, 0, 0, 0, 1, 1, 1, 1, 2),
)


def _sample_counts() -> list[int]:
    for _ in range(100):
        counts = list(random.choice(COUNT_PATTERNS))
        random.shuffle(counts)
        positive = [v for v in counts if v > 0]
        if len(positive) < 4:
            continue
        if not any(positive[i] > positive[j] for i in range(len(positive)) for j in range(i + 1, len(positive))):
            continue
        return counts
    raise RuntimeError("failed to sample a non-monotone count assignment")


def _sample_patch(color: int, *, require_h_asym: bool) -> np.ndarray:
    pool = H_ASYM_MASKS if require_h_asym else SPRITE_MASKS
    for _ in range(100):
        mask = random.choice(pool).copy()
        mask = np.rot90(mask, random.randrange(4))
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        if require_h_asym and np.array_equal(mask, np.fliplr(mask)):
            continue
        return np.where(mask, color, BG).astype(int)
    raise RuntimeError("failed to sample a valid sprite patch")


def _safe_marker_cells(region: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = region.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if region[r, c] != BG:
                continue
            ok = True
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    if region[nr, nc] == color:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _place_region(region_h: int, region_w: int, count: int, color: int, *, require_h_asym: bool) -> dict | None:
    for _ in range(150):
        patch = _sample_patch(color, require_h_asym=require_h_asym)
        ph, pw = patch.shape
        if ph > region_h or pw > region_w:
            continue
        top = random.randint(0, region_h - ph)
        left = random.randint(0, region_w - pw)

        region = np.full((region_h, region_w), BG, dtype=int)
        region[top : top + ph, left : left + pw] = patch

        marker_cells = _safe_marker_cells(region, color)
        if len(marker_cells) < count:
            continue
        markers = random.sample(marker_cells, count)
        for r, c in markers:
            region[r, c] = MARKER

        return {
            "region": region,
            "patch": patch,
            "count": count,
            "color": color,
            "offset": (top, left),
            "markers": tuple(markers),
        }
    return None


def _expected_from_records(records: list[dict]) -> np.ndarray:
    positive = [record for record in records if record["count"] > 0]
    positive.sort(key=lambda record: (record["count"], record["index"]))
    max_count = max(record["count"] for record in records)
    max_index = next(record["index"] for record in records if record["count"] == max_count)

    patches: list[np.ndarray] = []
    for record in positive:
        patch = record["patch"]
        if record["index"] == max_index:
            patch = np.fliplr(patch)
        patches.append(patch)

    out_h = sum(p.shape[0] for p in patches) + len(patches) - 1
    out_w = max(p.shape[1] for p in patches)
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = 0
    for i, patch in enumerate(patches):
        h, w = patch.shape
        out[cursor : cursor + h, :w] = patch
        cursor += h
        if i != len(patches) - 1:
            cursor += 1
    return out


def _validate_semantics(grid: np.ndarray, records: list[dict]) -> bool:
    try:
        parsed = _extract_regions(grid)
    except AssertionError:
        return False

    if len(parsed) != 9 or len(records) != 9:
        return False

    counts = [record["count"] for record in records]
    if counts.count(max(counts)) != 1 or counts.count(0) < 2:
        return False

    if not any(record["count"] > 0 and record["offset"] != (0, 0) for record in records):
        return False

    positive_counts = [record["count"] for record in records if record["count"] > 0]
    if not any(
        positive_counts[i] > positive_counts[j]
        for i in range(len(positive_counts))
        for j in range(i + 1, len(positive_counts))
    ):
        return False

    tie_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        if record["count"] > 0:
            tie_groups[record["count"]].append(record["patch"])
    if not any(
        len(group) >= 2 and any(not np.array_equal(group[0], other) for other in group[1:])
        for group in tie_groups.values()
    ):
        return False

    max_record = next(record for record in records if record["count"] == max(counts))
    if np.array_equal(max_record["patch"], np.fliplr(max_record["patch"])):
        return False

    for parsed_region, record in zip(parsed, records):
        if parsed_region["count"] != record["count"]:
            return False
        if not np.array_equal(parsed_region["patch"], record["patch"]):
            return False

    expected = _expected_from_records(records)
    actual = main(grid)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        return False
    if max(expected.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_heights = [random.randint(5, 7) for _ in range(3)]
    col_widths = [random.randint(5, 7) for _ in range(3)]
    counts = _sample_counts()
    max_index = counts.index(max(counts))

    colors = list(SPRITE_COLORS) + random.sample(SPRITE_COLORS, 2)
    random.shuffle(colors)

    h = sum(row_heights) + 2
    w = sum(col_widths) + 2
    grid = np.full((h, w), BG, dtype=int)

    sep_rows = [row_heights[0], row_heights[0] + 1 + row_heights[1]]
    sep_cols = [col_widths[0], col_widths[0] + 1 + col_widths[1]]
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    records: list[dict] = []
    idx = 0
    for rb, (r0, r1) in enumerate(row_bands):
        for cb, (c0, c1) in enumerate(col_bands):
            placed = _place_region(
                r1 - r0,
                c1 - c0,
                counts[idx],
                colors[idx],
                require_h_asym=idx == max_index,
            )
            if placed is None:
                return None
            grid[r0:r1, c0:c1] = placed["region"]
            records.append(
                {
                    "index": idx,
                    "grid_pos": (rb, cb),
                    "count": counts[idx],
                    "patch": placed["patch"],
                    "offset": placed["offset"],
                }
            )
            idx += 1

    if not _validate_semantics(grid, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
