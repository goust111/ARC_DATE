from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
OBJECT_DIMS = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3))
OBJECT_AREAS = tuple(sorted({h * w for h, w in OBJECT_DIMS}))
OUTLINE_AREAS = tuple(range(5, 13))
GRID_ROWS_RANGE = (17, 24)
GRID_COLS_RANGE = (17, 24)
GEN_ATTEMPTS = 500


def _row_span_interior_area(grid: np.ndarray, coords: np.ndarray, bg: int = BG) -> int:
    occupied = {(int(r), int(c)) for r, c in coords}
    area = 0
    for r in sorted({int(r) for r, _c in occupied}):
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if (r, c) in occupied:
                continue
            assert int(grid[r, c]) == int(bg), "outline interiors must remain background"
            area += 1
    return int(area)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(get_bg(g)) == BG, "expected black background"

    outlines = find_components(g, bg=BG, connectivity=8, monochromatic=True, colors=[GREY])
    objects = find_components(g, bg=BG, connectivity=4, monochromatic=False, colors=PALETTE)
    assert outlines, "expected at least one grey outline"
    assert objects, "expected at least one non-grey object"

    outline_areas = []
    for outline in outlines:
        area = _row_span_interior_area(g, outline["coords"], BG)
        assert area > 0, "every outline must enclose a positive row-span interior"
        outline_areas.append(area)
    assert len(set(outline_areas)) == len(outline_areas), "outline interior areas must be distinct"

    for obj in objects:
        box = obj["bbox"]
        area = int(obj["area"])
        assert area == bbox_h(box) * bbox_w(box), "foreground objects must be solid rectangles"
        values = g[obj["coords"][:, 0], obj["coords"][:, 1]]
        colors = sorted(set(map(int, values)))
        assert len(colors) == 2, "each foreground rectangle must use exactly two colors"
        assert GREY not in colors and BG not in colors

    return outlines, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two colors inside exactly those solid rectangles whose area matches a grey outline interior area.

    Input structure:
    - The background is black.
    - Several separated non-grey solid rectangular foreground objects appear on the canvas; each rectangle is 4-connected and uses exactly two colors.
    - Several grey 8-connected outline objects also appear, with empty black row-span interiors.

    Rule:
    - For each grey outline object, compute its row-span interior area: on every occupied row, count the black cells strictly between the leftmost and rightmost grey pixels.
    - For each non-grey rectangle, compute its total foreground area.
    - If a rectangle's area matches any grey outline interior area, swap its two local colors in place; otherwise keep it unchanged.
    - Keep all grey outline pixels unchanged.

    Key insight:
    - The grey outlines are numeric gates, not painting targets: they selectively activate the local two-color swap only for rectangles whose size matches a cavity area.
    """
    g = np.asarray(grid, dtype=int)
    outlines, objects = _parse_scene(g)
    matched_areas = {
        _row_span_interior_area(g, outline["coords"], BG)
        for outline in outlines
    }

    out = g.copy()
    for obj in objects:
        if int(obj["area"]) not in matched_areas:
            continue
        coords = obj["coords"]
        values = g[coords[:, 0], coords[:, 1]]
        a, b = sorted(set(map(int, values)))
        a_mask = values == a
        b_mask = values == b
        out[coords[a_mask, 0], coords[a_mask, 1]] = b
        out[coords[b_mask, 0], coords[b_mask, 1]] = a
    return out


def _expand8(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    return expanded


def _random_composition(total: int, parts: int, rng: random.Random, max_part: int = 5) -> list[int] | None:
    vals: list[int] = []
    remain = int(total)
    for idx in range(parts):
        left = parts - idx - 1
        lo = max(1, remain - max_part * left)
        hi = min(max_part, remain - left)
        if lo > hi:
            return None
        val = rng.randint(lo, hi)
        vals.append(val)
        remain -= val
    return vals if remain == 0 else None


def _build_outline_patch(target_area: int, rng: random.Random) -> np.ndarray:
    valid_rows = [k for k in (1, 2, 3) if k <= target_area <= 5 * k]
    for _ in range(80):
        interior_rows = rng.choice(valid_rows)
        gaps = _random_composition(target_area, interior_rows, rng)
        if gaps is None:
            continue

        lefts = [rng.randint(0, 1)]
        for _ in range(1, interior_rows):
            lefts.append(max(0, lefts[-1] + rng.choice((-1, 0, 1))))
        rights = [l + gap + 1 for l, gap in zip(lefts, gaps)]
        offset = min(lefts)
        lefts = [l - offset for l in lefts]
        rights = [r - offset for r in rights]

        height = interior_rows + 2
        width = max(rights) + 1
        patch = np.zeros((height, width), dtype=int)
        patch[0, lefts[0] : rights[0] + 1] = GREY
        patch[-1, lefts[-1] : rights[-1] + 1] = GREY
        for r, (left, right) in enumerate(zip(lefts, rights), start=1):
            patch[r, left] = GREY
            patch[r, right] = GREY

        if rng.random() < 0.5:
            patch = np.fliplr(patch)
        if rng.random() < 0.5:
            patch = np.flipud(patch)

        comps = find_components(patch, bg=BG, connectivity=8, monochromatic=True, colors=[GREY])
        if len(comps) != 1:
            continue
        if _row_span_interior_area(patch, comps[0]["coords"], BG) != target_area:
            continue
        return patch
    raise RuntimeError(f"failed to build outline patch for area {target_area}")


def _build_object_patch(area: int, rng: random.Random) -> np.ndarray:
    dims = [(h, w) for h, w in OBJECT_DIMS if h * w == area]
    h, w = rng.choice(dims)
    a, b = rng.sample(PALETTE, 2)
    patch = np.full((h, w), b, dtype=int)
    coords = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(coords)
    count_a = rng.randint(1, area - 1)
    for r, c in coords[:count_a]:
        patch[r, c] = a
    return patch


def _all_positions(height: int, width: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(height - patch_h + 1) for c in range(width - patch_w + 1)]


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = patch.shape
    positions = _all_positions(grid.shape[0], grid.shape[1], h, w)
    rng.shuffle(positions)
    blocked = _expand8(reserved)
    for r0, c0 in positions:
        if blocked[r0 : r0 + h, c0 : c0 + w].any():
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        return r0, c0
    return None


def _expected_output_from_metadata(grid: np.ndarray, object_meta: list[dict]) -> np.ndarray:
    out = grid.copy()
    for meta in object_meta:
        if not meta["matched"]:
            continue
        r0, c0 = meta["top_left"]
        patch = meta["patch"]
        h, w = patch.shape
        region = out[r0 : r0 + h, c0 : c0 + w]
        a, b = sorted(set(map(int, patch.ravel())))
        region[patch == a] = b
        region[patch == b] = a
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, dict]:
    n_objects = rng.randint(3, 4)
    n_outlines = rng.randint(2, 3)
    matched_count = rng.randint(1, n_objects - 1)
    matched_area_kinds = rng.randint(1, min(2, n_outlines, matched_count))

    matched_areas = rng.sample(list(OBJECT_AREAS), matched_area_kinds)
    unmatched_area_pool = [area for area in OBJECT_AREAS if area not in matched_areas]
    object_areas = [rng.choice(matched_areas) for _ in range(matched_count)]
    object_areas.extend(rng.choice(unmatched_area_pool) for _ in range(n_objects - matched_count))
    rng.shuffle(object_areas)

    extra_outline_pool = [area for area in OUTLINE_AREAS if area not in set(object_areas) and area not in matched_areas]
    extra_needed = n_outlines - matched_area_kinds
    outline_areas = matched_areas + rng.sample(extra_outline_pool, extra_needed)
    rng.shuffle(outline_areas)

    height = rng.randint(*GRID_ROWS_RANGE)
    width = rng.randint(*GRID_COLS_RANGE)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    components: list[tuple[int, np.ndarray, dict]] = []
    object_meta: list[dict] = []
    for idx, area in enumerate(object_areas):
        patch = _build_object_patch(area, rng)
        meta = {
            "kind": "object",
            "index": idx,
            "area": area,
            "matched": area in matched_areas,
            "patch": patch.copy(),
        }
        object_meta.append(meta)
        components.append((patch.shape[0] * patch.shape[1], patch, meta))

    for idx, area in enumerate(outline_areas):
        patch = _build_outline_patch(area, rng)
        meta = {"kind": "outline", "index": idx, "area": area}
        components.append((patch.shape[0] * patch.shape[1], patch, meta))

    components.sort(key=lambda item: item[0], reverse=True)
    for _footprint, patch, meta in components:
        placed = _place_patch(grid, reserved, patch, rng)
        if placed is None:
            raise RuntimeError("failed to place all components")
        if meta["kind"] == "object":
            object_meta[meta["index"]]["top_left"] = placed

    metadata = {
        "outline_areas": outline_areas,
        "objects": object_meta,
        "matched_count": matched_count,
    }
    return grid, metadata


def _validate_case(grid: np.ndarray, metadata: dict) -> None:
    outlines, objects = _parse_scene(grid)
    parsed_outline_areas = sorted(_row_span_interior_area(grid, comp["coords"], BG) for comp in outlines)
    parsed_object_areas = sorted(int(obj["area"]) for obj in objects)
    expected_outline_areas = sorted(int(a) for a in metadata["outline_areas"])
    expected_object_areas = sorted(int(obj["area"]) for obj in metadata["objects"])

    assert parsed_outline_areas == expected_outline_areas, "outline areas must match metadata"
    assert parsed_object_areas == expected_object_areas, "object areas must match metadata"

    matched_areas = set(parsed_outline_areas)
    matched = sum(int(obj["area"]) in matched_areas for obj in objects)
    assert matched == int(metadata["matched_count"]), "matched-object count drifted"
    assert 1 <= matched < len(objects), "need both matching and non-matching objects"

    expected = _expected_output_from_metadata(grid, metadata["objects"])
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match metadata oracle"

    changed = 0
    unchanged = 0
    for meta in metadata["objects"]:
        r0, c0 = meta["top_left"]
        patch = meta["patch"]
        h, w = patch.shape
        before = grid[r0 : r0 + h, c0 : c0 + w]
        after = actual[r0 : r0 + h, c0 : c0 + w]
        if meta["matched"]:
            a, b = sorted(set(map(int, patch.ravel())))
            swapped = patch.copy()
            swapped[patch == a] = b
            swapped[patch == b] = a
            assert np.array_equal(after, swapped), "matched object did not swap correctly"
            changed += 1
        else:
            assert np.array_equal(before, after), "unmatched object should stay unchanged"
            unchanged += 1
    assert changed == int(metadata["matched_count"])
    assert unchanged == len(metadata["objects"]) - int(metadata["matched_count"])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        try:
            grid, metadata = _generate_case(rng)
            _validate_case(grid, metadata)
        except (AssertionError, RuntimeError, ValueError):
            continue
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid area-triggered color-swap input")
