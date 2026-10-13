from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 400

SEEDS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_regions(grid: np.ndarray) -> list[dict]:
    comps = _mask_components(grid != WALL)
    regions: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert len(comp) == hh * ww, "expected every wall-bounded region to be rectangular"
        regions.append({"bbox": (r0, c0, r1, c1), "h": hh, "w": ww})
    regions.sort(key=lambda item: item["bbox"])
    return regions


def _region_kind(h: int, w: int) -> str:
    if h == w:
        return "rot"
    if w == 2 * h:
        return "hmirror"
    if h == 2 * w:
        return "vmirror"
    raise AssertionError(f"unsupported region ratio {(h, w)}")


def _transform_coords(coords: set[tuple[int, int]], h: int, w: int, kind: str) -> set[tuple[int, int]]:
    out = set(coords)
    if kind == "rot":
        for r, c in list(coords):
            out.add((c, h - 1 - r))
            out.add((h - 1 - r, w - 1 - c))
            out.add((w - 1 - c, r))
    elif kind == "hmirror":
        for r, c in list(coords):
            out.add((r, w - 1 - c))
    elif kind == "vmirror":
        for r, c in list(coords):
            out.add((h - 1 - r, c))
    elif kind == "hflip":
        for r, c in list(coords):
            out.add((r, w - 1 - c))
    elif kind == "vflip":
        for r, c in list(coords):
            out.add((h - 1 - r, c))
    elif kind == "rot180":
        for r, c in list(coords):
            out.add((h - 1 - r, w - 1 - c))
    else:
        raise AssertionError(f"unknown transform {kind}")
    return {(r, c) for r, c in out if 0 <= r < h and 0 <= c < w}


def _local_object_coords(patch: np.ndarray) -> tuple[int, set[tuple[int, int]]]:
    colors = sorted({int(v) for v in np.unique(patch)} - {BG, WALL})
    assert len(colors) == 1, "expected exactly one foreground color per region"
    color = colors[0]
    coords = {(int(r), int(c)) for r, c in np.argwhere(patch == color)}
    assert 3 <= len(coords) <= 8, "foreground object size must stay in the intended range"
    assert len(_mask_components(patch == color)) == 1, "foreground object must be 4-connected"
    return color, coords


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each region's partial object using the symmetry implied by that region's wall-bounded rectangle.

    Input structure:
    - Black cells (0) form the background inside rectangular regions bounded by gray walls (5).
    - Every wall-bounded region is a solid rectangle of non-wall cells.
    - Each region contains exactly one 4-connected monochrome object in a non-black, non-gray color.

    Rule:
    - Find each rectangular region separated by the gray walls and inspect its width/height ratio.
    - Square regions use 90-degree rotational completion about the region center.
    - Regions with width = 2 * height use horizontal mirror completion across the region's vertical midline.
    - Regions with height = 2 * width use vertical mirror completion across the region's horizontal midline.
    - Add the completed pixels in the object's own color, keep black background elsewhere, and leave gray walls unchanged.

    Key insight:
    - The object itself does not tell you which symmetry to apply; the enclosing wall-defined container does.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for region in _extract_regions(g):
        r0, c0, r1, c1 = region["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        kind = _region_kind(region["h"], region["w"])
        color, coords = _local_object_coords(patch)
        completed = _transform_coords(coords, region["h"], region["w"], kind)
        for rr, cc in completed:
            out[r0 + rr, c0 + cc] = color

    return out


def _region_layout() -> tuple[list[int], list[int]]:
    row_count, col_count = random.choice(((2, 2), (2, 3), (3, 2)))

    if row_count == 2:
        row_units = [1, 2]
        random.shuffle(row_units)
    else:
        row_units = [1, 2, random.choice((1, 2))]
        random.shuffle(row_units)

    if col_count == 2:
        col_units = [1, 2]
        random.shuffle(col_units)
    else:
        col_units = [1, 2, random.choice((1, 2))]
        random.shuffle(col_units)

    max_unit_h = (30 - (row_count + 1)) // sum(row_units)
    max_unit_w = (30 - (col_count + 1)) // sum(col_units)
    max_unit = min(max_unit_h, max_unit_w, 6)
    assert max_unit >= 3
    unit = random.randint(3, max_unit)

    return [unit * v for v in row_units], [unit * v for v in col_units]


def _build_partition() -> tuple[np.ndarray, list[dict]]:
    row_heights, col_widths = _region_layout()
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    assert h <= 30 and w <= 30

    grid = np.full((h, w), WALL, dtype=int)
    regions: list[dict] = []
    r = 1
    for rh in row_heights:
        c = 1
        for cw in col_widths:
            grid[r : r + rh, c : c + cw] = BG
            kind = _region_kind(rh, cw)
            regions.append({"bbox": (r, c, r + rh - 1, c + cw - 1), "h": rh, "w": cw, "kind": kind})
            c += cw + 1
        r += rh + 1
    return grid, regions


def _seed_dims(seed: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in seed) + 1, max(c for _, c in seed) + 1


def _anchor_limits(region: dict, seed: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    sh, sw = _seed_dims(seed)
    h = region["h"]
    w = region["w"]
    if region["kind"] == "rot":
        max_r = min(1, h - sh)
        max_c = min(1, w - sw)
    elif region["kind"] == "hmirror":
        max_r = min(1, h - sh)
        max_c = min(1, h - sw)
    else:
        max_r = min(1, w - sh)
        max_c = min(1, w - sw)
    assert max_r >= 0 and max_c >= 0
    return max_r, max_c


def _paint_seed(
    grid: np.ndarray,
    region: dict,
    seed: tuple[tuple[int, int], ...],
    color: int,
    anchor: tuple[int, int] | None = None,
) -> dict:
    sh, sw = _seed_dims(seed)
    max_r, max_c = _anchor_limits(region, seed)
    if anchor is None:
        ar = random.randint(0, max_r)
        ac = random.randint(0, max_c)
    else:
        ar, ac = anchor
        assert 0 <= ar <= max_r and 0 <= ac <= max_c

    r0, c0, _r1, _c1 = region["bbox"]
    coords = []
    for dr, dc in seed:
        rr = r0 + ar + dr
        cc = c0 + ac + dc
        grid[rr, cc] = color
        coords.append((ar + dr, ac + dc))
    return {"kind": region["kind"], "color": color, "local_coords": set(coords), "seed": seed, "anchor": (ar, ac)}


def _validate_semantics(inp: np.ndarray, records: list[dict]) -> None:
    regions = _extract_regions(inp)
    assert 4 <= len(regions) <= 6
    kinds = [_region_kind(region["h"], region["w"]) for region in regions]
    assert {"rot", "hmirror", "vmirror"} <= set(kinds)

    out = main(inp)
    shared = [record for record in records if record.get("shared")]
    assert {record["kind"] for record in shared} == {"rot", "hmirror", "vmirror"}
    shared_shapes = {_normalize(record["local_coords"]) for record in shared}
    shared_positions = {tuple(sorted(record["local_coords"])) for record in shared}
    assert len(shared_shapes) == 1 and len(shared_positions) == 1

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        patch_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        kind = _region_kind(region["h"], region["w"])
        color, coords = _local_object_coords(patch_in)
        expected = _transform_coords(coords, region["h"], region["w"], kind)
        actual = {(int(r), int(c)) for r, c in np.argwhere(patch_out == color)}
        assert actual == expected
        assert len(actual) > len(coords), "completion must add pixels"
        if kind == "rot":
            assert actual != _transform_coords(coords, region["h"], region["w"], "hflip")
            assert actual != _transform_coords(coords, region["h"], region["w"], "vflip")
        elif kind == "hmirror":
            assert actual != _transform_coords(coords, region["h"], region["w"], "vflip")
            assert actual != _transform_coords(coords, region["h"], region["w"], "rot180")
        else:
            assert actual != _transform_coords(coords, region["h"], region["w"], "hflip")
            assert actual != _transform_coords(coords, region["h"], region["w"], "rot180")


def _generate_one() -> np.ndarray | None:
    grid, regions = _build_partition()
    regions_by_kind: dict[str, list[dict]] = {"rot": [], "hmirror": [], "vmirror": []}
    for region in regions:
        regions_by_kind[region["kind"]].append(region)

    shared_seed = random.choice(SEEDS)
    colors = random.sample(FG_COLORS, len(regions))
    random.shuffle(regions)
    records: list[dict] = []

    color_iter = iter(colors)
    used_shared: set[tuple[int, int, int, int]] = set()
    for kind in ("rot", "hmirror", "vmirror"):
        region = random.choice(regions_by_kind[kind])
        used_shared.add(region["bbox"])
        record = _paint_seed(grid, region, shared_seed, next(color_iter), anchor=(0, 0))
        record["shared"] = True
        records.append(record)

    for region in regions:
        if region["bbox"] in used_shared:
            continue
        seed = random.choice(SEEDS)
        record = _paint_seed(grid, region, seed, next(color_iter))
        record["shared"] = False
        records.append(record)

    try:
        _validate_semantics(grid, records)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        candidate = _generate_one()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
