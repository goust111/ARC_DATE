from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

BG = 0
BLUE = 1
GREY = 5

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

AREA_POOL = (1, 2, 3, 4, 5, 6)
GEN_ATTEMPTS = 400


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray, connectivity: int) -> list[list[Coord]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _single_component(coords: tuple[Coord, ...], connectivity: int = 8) -> bool:
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return len(_find_components(mask, connectivity)) == 1


def _row_span_cavity_cells(grid: np.ndarray, coords: list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    rows: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))

    cavity: list[Coord] = []
    for r in sorted(rows):
        cols = rows[r]
        left = min(cols)
        right = max(cols)
        for c in range(left + 1, right):
            if int(grid[r, c]) == BG:
                cavity.append((r, c))
    return tuple(cavity)


def _extract_contours(grid: np.ndarray) -> list[dict]:
    contours = []
    for comp in _find_components(grid == GREY, connectivity=8):
        cavity = _row_span_cavity_cells(grid, comp)
        contours.append(
            {
                "coords": tuple(sorted(comp)),
                "bbox": _bbox(comp),
                "cavity": cavity,
                "area": len(cavity),
            }
        )
    contours.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return contours


def _extract_dashed_segments(grid: np.ndarray) -> list[dict]:
    segments = []
    h, _w = grid.shape
    for r in range(h):
        blue_cols = np.where(grid[r] == BLUE)[0].tolist()
        if not blue_cols:
            continue
        i = 0
        while i < len(blue_cols):
            cols = [int(blue_cols[i])]
            j = i
            while j + 1 < len(blue_cols):
                c0 = int(blue_cols[j])
                c1 = int(blue_cols[j + 1])
                if c1 - c0 == 2 and int(grid[r, c0 + 1]) == BG:
                    cols.append(c1)
                    j += 1
                else:
                    break
            if len(cols) >= 2:
                gap_cells = tuple((r, c + 1) for c in cols[:-1])
                segments.append(
                    {
                        "row": r,
                        "c0": cols[0],
                        "c1": cols[-1],
                        "blue_cells": tuple((r, c) for c in cols),
                        "gap_cells": gap_cells,
                        "gaps": len(gap_cells),
                    }
                )
            i = j + 1
    segments.sort(key=lambda item: (item["row"], item["c0"]))
    return segments


def _match_pairs(contours: list[dict], segments: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    used_segments: set[int] = set()
    for contour in sorted(contours, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        for idx, segment in enumerate(sorted(segments, key=lambda item: (item["row"], item["c0"]))):
            if idx in used_segments:
                continue
            if int(segment["gaps"]) == int(contour["area"]):
                used_segments.add(idx)
                pairs.append((contour, segment))
                break
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate matching grey contour cavities and blue dash gaps when their counts agree.

    Input structure:
    - Black is the background.
    - Grey pixels form separate 8-connected closed outline objects whose occupied rows may
      contain black cavity cells between the leftmost and rightmost grey pixels.
    - Blue pixels form separate horizontal dashed segments made of single blue cells with
      one black cell between consecutive dashes.

    Rule:
    - For each grey outline, sum the black pixels strictly between the leftmost and rightmost
      grey pixels on every row occupied by that outline; this is its cavity area.
    - For each blue dashed segment, count the black cells whose left and right neighbors are
      blue within that segment; this is its gap count.
    - Process grey outlines from top to bottom, then left to right. Each outline matches the
      first still-unmatched dashed segment, in top-to-bottom order, whose gap count equals the
      outline's cavity area.
    - Fill every matched outline cavity with blue and fill every matched dashed gap with grey.
      Leave unmatched outlines, unmatched dashed segments, and all original grey/blue pixels unchanged.

    Key insight:
    - The only bridge between the two object types is a shared count: row-span cavity area for
      contours and internal one-cell gap count for dashed segments must agree before either side changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    contours = _extract_contours(g)
    segments = _extract_dashed_segments(g)

    out = g.copy()
    for contour, segment in _match_pairs(contours, segments):
        for r, c in contour["cavity"]:
            out[r, c] = BLUE
        for r, c in segment["gap_cells"]:
            out[r, c] = GREY
    return out


def _positive_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for tail in _positive_compositions(total - first, parts - 1):
            out.append((first,) + tail)
    return out


def _catalog_shape_from_profile(widths: tuple[int, ...], lefts: tuple[int, ...]) -> dict:
    min_left = min(lefts)
    norm_lefts = tuple(left - min_left for left in lefts)
    rights = tuple(norm_lefts[i] + widths[i] + 1 for i in range(len(widths)))
    middle_rows = len(widths)
    height = middle_rows + 2
    width = max(rights) + 1

    pixels: list[Coord] = []
    cavity: list[Coord] = []

    for c in range(norm_lefts[0], rights[0] + 1):
        pixels.append((0, c))
    for idx, (left, right) in enumerate(zip(norm_lefts, rights), start=1):
        pixels.append((idx, left))
        pixels.append((idx, right))
        for c in range(left + 1, right):
            cavity.append((idx, c))
    for c in range(norm_lefts[-1], rights[-1] + 1):
        pixels.append((height - 1, c))

    pixel_tuple = tuple(sorted(set(pixels)))
    cavity_tuple = tuple(sorted(cavity))
    return {"pixels": pixel_tuple, "cavity": cavity_tuple, "height": height, "width": width}


def _mirror_shape(shape: dict, flip_v: bool, flip_h: bool) -> dict:
    height = int(shape["height"])
    width = int(shape["width"])

    def transform(cell: Coord) -> Coord:
        r, c = cell
        if flip_v:
            r = height - 1 - r
        if flip_h:
            c = width - 1 - c
        return r, c

    return {
        "pixels": tuple(sorted(transform(cell) for cell in shape["pixels"])),
        "cavity": tuple(sorted(transform(cell) for cell in shape["cavity"])),
        "height": height,
        "width": width,
    }


def _build_shape_library() -> dict[int, list[dict]]:
    library: dict[int, list[dict]] = {}
    for area in AREA_POOL:
        variants: dict[tuple[Coord, ...], dict] = {}
        for middle_rows in range(1, min(4, area) + 1):
            for widths in _positive_compositions(area, middle_rows):
                if any(abs(widths[i] - widths[i - 1]) > 2 for i in range(1, middle_rows)):
                    continue

                def backtrack(idx: int, lefts: list[int]) -> None:
                    if idx == middle_rows:
                        base = _catalog_shape_from_profile(widths, tuple(lefts))
                        if base["height"] > 6 or base["width"] > 8:
                            return
                        if not _single_component(base["pixels"], connectivity=8):
                            return
                        if len(base["cavity"]) != area:
                            return
                        for flip_v in (False, True):
                            for flip_h in (False, True):
                                shaped = _mirror_shape(base, flip_v=flip_v, flip_h=flip_h)
                                variants.setdefault(shaped["pixels"], shaped)
                        return

                    diff = widths[idx] - widths[idx - 1]
                    for delta in (-1, 0, 1):
                        if abs(delta + diff) <= 1:
                            lefts.append(lefts[-1] + delta)
                            backtrack(idx + 1, lefts)
                            lefts.pop()

                backtrack(1, [0])
        library[area] = list(variants.values())
        assert library[area], f"no contour variants for area {area}"
    return library


SHAPE_LIBRARY = _build_shape_library()


def _make_contour_spec(area: int, rng: random.Random) -> dict:
    shape = rng.choice(SHAPE_LIBRARY[area])
    return {
        "kind": "contour",
        "value": area,
        "pixels": shape["pixels"],
        "fill": shape["cavity"],
        "height": int(shape["height"]),
        "width": int(shape["width"]),
    }


def _make_segment_spec(gaps: int) -> dict:
    blue_cells = tuple((0, 2 * i) for i in range(gaps + 1))
    gap_cells = tuple((0, 2 * i + 1) for i in range(gaps))
    return {
        "kind": "segment",
        "value": gaps,
        "pixels": blue_cells,
        "fill": gap_cells,
        "height": 1,
        "width": 2 * gaps + 1,
    }


def _choose_value_sets(n_contours: int, n_segments: int, rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    max_match = min(n_contours - 1, n_segments - 1)
    match_count = rng.randint(2, max_match)

    pool = list(AREA_POOL)
    rng.shuffle(pool)
    matched = sorted(pool[:match_count])
    rest = pool[match_count:]

    contour_only = rest[: n_contours - match_count]
    segment_only = rest[n_contours - match_count : n_contours - match_count + (n_segments - match_count)]

    contour_values = matched + contour_only
    segment_values = matched + segment_only
    rng.shuffle(contour_values)
    rng.shuffle(segment_values)
    return contour_values, segment_values, matched


def _pick_canvas_size(n_items: int, rng: random.Random) -> tuple[int, int]:
    low = 17 + max(0, n_items - 6) * 2
    high = min(24, low + 3)
    return rng.randint(low, high), rng.randint(low, high)


def _place_specs(specs: list[dict], height: int, width: int, rng: random.Random) -> bool:
    reserved = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: (specs[idx]["width"] * specs[idx]["height"], specs[idx]["width"]), reverse=True)
    for idx in order:
        spec = specs[idx]
        h = int(spec["height"])
        w = int(spec["width"])
        candidates: list[Coord] = []
        for r0 in range(1, height - h):
            for c0 in range(1, width - w):
                if reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        spec["origin"] = (r0, c0)
        reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
    return True


def _instantiate_specs(specs: list[dict]) -> tuple[list[dict], list[dict]]:
    contours: list[dict] = []
    segments: list[dict] = []
    for spec in specs:
        r0, c0 = spec["origin"]
        pixels = tuple(sorted((r0 + r, c0 + c) for r, c in spec["pixels"]))
        fill = tuple(sorted((r0 + r, c0 + c) for r, c in spec["fill"]))
        bbox = (r0, c0, r0 + spec["height"] - 1, c0 + spec["width"] - 1)
        if spec["kind"] == "contour":
            contours.append({"coords": pixels, "cavity": fill, "bbox": bbox, "area": spec["value"]})
        else:
            segments.append(
                {
                    "blue_cells": pixels,
                    "gap_cells": fill,
                    "row": r0,
                    "c0": c0,
                    "c1": c0 + spec["width"] - 1,
                    "gaps": spec["value"],
                }
            )
    contours.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    segments.sort(key=lambda item: (item["row"], item["c0"]))
    return contours, segments


def _render_input(height: int, width: int, contours: list[dict], segments: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for contour in contours:
        for r, c in contour["coords"]:
            grid[r, c] = GREY
    for segment in segments:
        for r, c in segment["blue_cells"]:
            grid[r, c] = BLUE
    return grid


def _expected_output(grid: np.ndarray, contours: list[dict], segments: list[dict]) -> np.ndarray:
    out = grid.copy()
    for contour, segment in _match_pairs(contours, segments):
        for r, c in contour["cavity"]:
            out[r, c] = BLUE
        for r, c in segment["gap_cells"]:
            out[r, c] = GREY
    return out


def _validate_scene(grid: np.ndarray, expected_contours: list[dict], expected_segments: list[dict], matched_values: list[int]) -> bool:
    contours = _extract_contours(grid)
    segments = _extract_dashed_segments(grid)

    if len(contours) != len(expected_contours) or len(segments) != len(expected_segments):
        return False
    if sorted(item["area"] for item in contours) != sorted(item["area"] for item in expected_contours):
        return False
    if sorted(item["gaps"] for item in segments) != sorted(item["gaps"] for item in expected_segments):
        return False

    pairs = _match_pairs(contours, segments)
    if len(pairs) != len(matched_values):
        return False
    if sorted(contour["area"] for contour, _segment in pairs) != sorted(matched_values):
        return False
    if sorted(segment["gaps"] for _contour, segment in pairs) != sorted(matched_values):
        return False

    expected = _expected_output(grid, expected_contours, expected_segments)
    if np.array_equal(grid, expected):
        return False
    return np.array_equal(main(grid), expected)


def _generate_once(rng: random.Random) -> np.ndarray | None:
    n_contours = rng.randint(3, 4)
    n_segments = rng.randint(3, 4)
    contour_values, segment_values, matched_values = _choose_value_sets(n_contours, n_segments, rng)

    specs = [_make_contour_spec(value, rng) for value in contour_values]
    specs.extend(_make_segment_spec(value) for value in segment_values)

    height, width = _pick_canvas_size(len(specs), rng)
    if not _place_specs(specs, height, width, rng):
        return None

    contours, segments = _instantiate_specs(specs)
    grid = _render_input(height, width, contours, segments)
    if not _validate_scene(grid, contours, segments, matched_values):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle after {GEN_ATTEMPTS} attempts")
