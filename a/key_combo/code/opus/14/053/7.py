from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
AREA_TO_REGION = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
REGION_TO_AREA = {region: area for area, region in AREA_TO_REGION.items()}
GLYPH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    assert len(separators) == 2
    separators = sorted(int(x) for x in separators)
    out = []
    start = 0
    for cut in separators:
        assert start < cut
        out.append((int(start), int(cut - 1)))
        start = int(cut + 1)
    assert start < size
    out.append((int(start), int(size - 1)))
    return out


def _partition_regions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(sep_rows) == 2
    assert len(sep_cols) == 2
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    regions = []
    for (r0, r1), (c0, c1) in itertools.product(row_bands, col_bands):
        regions.append((int(r0), int(c0), int(r1), int(c1)))
    return regions


def _extract_glyph(region: np.ndarray) -> dict | None:
    colors = sorted(set(map(int, region.flatten())) - {BLACK, GREY})
    if not colors:
        return None
    assert len(colors) == 1, "a region may contain at most one monochrome glyph"
    color = int(colors[0])
    mask = region == color
    comps = _components4(mask)
    assert len(comps) == 1, "glyph must be 4-connected"
    area = len(comps[0])
    assert area in AREA_TO_REGION, "glyph area must decode to a target region"
    return {"color": color, "area": int(area), "mask": mask}


def _parse_grid(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    regions = _partition_regions(grid)
    glyphs: list[dict] = []
    for region_id, bounds in enumerate(regions, start=1):
        r0, c0, r1, c1 = bounds
        glyph = _extract_glyph(grid[r0 : r1 + 1, c0 : c1 + 1])
        if glyph is None:
            continue
        glyphs.append(
            {
                "source_id": int(region_id),
                "target_id": int(AREA_TO_REGION[glyph["area"]]),
                "color": int(glyph["color"]),
                "area": int(glyph["area"]),
                "mask": glyph["mask"],
            }
        )
    return regions, glyphs


def _fill_region(out: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bounds
    tile = out[r0 : r1 + 1, c0 : c1 + 1]
    tile[tile != GREY] = int(color)


def _oracle_output(grid: np.ndarray, glyphs: list[dict], regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for glyph in sorted(glyphs, key=lambda g: g["source_id"]):
        _fill_region(out, regions[glyph["target_id"] - 1], glyph["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode self-referential glyph instructions inside a 3x3 partitioned grid and fill the addressed regions.

    Input structure:
    - A black background is split by exactly two full grey rows and two full grey columns into nine rectangular regions.
    - Some regions contain one small monochrome 4-connected glyph; other regions are empty black.
    - Each glyph's color is its fill color, and its pixel area is its encoded target-region id.

    Rule:
    - Find the 3x3 partition from the grey separator lines and number the nine regions in reading order.
    - Scan regions in reading order; every non-empty region contributes one glyph instruction.
    - Decode glyph area with 3->1, 4->2, 5->3, 6->4, 7->5, 8->6, then fill that target region with the glyph's color.
    - Filling replaces every non-grey pixel in the target region, including black background and any glyph already inside it; grey separators stay unchanged.

    Key insight:
    - The same glyph carries both pieces of instruction data: visible color chooses what to paint, while visible area chooses where to paint inside the partitioned grid.
    """
    g = np.asarray(grid, dtype=int)
    regions, glyphs = _parse_grid(g)
    return _oracle_output(g, glyphs, regions)


def _sample_band_lengths() -> tuple[int, int, int]:
    options = [(a, b, c) for a in (4, 5) for b in (4, 5) for c in (4, 5)]
    weights = np.array([1 if len({a, b, c}) == 1 else 3 for a, b, c in options], dtype=float)
    weights /= weights.sum()
    return tuple(int(x) for x in options[int(RNG.choice(len(options), p=weights))])


def _regions_from_bands(row_bands: tuple[int, int, int], col_bands: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    row_starts = [0, row_bands[0] + 1, row_bands[0] + row_bands[1] + 2]
    col_starts = [0, col_bands[0] + 1, col_bands[0] + col_bands[1] + 2]
    regions = []
    for rr in range(3):
        for cc in range(3):
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            regions.append((int(r0), int(c0), int(r0 + row_bands[rr] - 1), int(c0 + col_bands[cc] - 1)))
    return regions


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    pos = (r + dr, c + dc)
                    if pos not in cells:
                        frontier.append(pos)
            RNG.shuffle(frontier)
            added = None
            for nr, nc in frontier:
                trial = cells | {(int(nr), int(nc))}
                rs = [p[0] for p in trial]
                cs = [p[1] for p in trial]
                if max(rs) - min(rs) + 1 <= max_h and max(cs) - min(cs) + 1 <= max_w:
                    added = (int(nr), int(nc))
                    break
            if added is None:
                break
            cells.add(added)
        if len(cells) != area:
            continue
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_h or w > max_w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to sample a connected glyph shape")


def _place_glyph(grid: np.ndarray, bounds: tuple[int, int, int, int], area: int, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    shape = _random_connected_shape(area, region_h, region_w)
    sh, sw = shape.shape
    off_r = int(RNG.integers(0, region_h - sh + 1))
    off_c = int(RNG.integers(0, region_w - sw + 1))
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region[off_r : off_r + sh, off_c : off_c + sw][shape] = int(color)
    mask = np.zeros((region_h, region_w), dtype=bool)
    mask[off_r : off_r + sh, off_c : off_c + sw] = shape
    return mask


def _choose_sources_and_targets(k: int) -> dict[int, int] | None:
    upper = list(range(1, 7))
    all_regions = list(range(1, 10))
    for _ in range(400):
        covered_region = int(RNG.choice(upper))
        covering_src_choices = [rid for rid in all_regions if rid != covered_region]
        if RNG.random() < 0.7:
            bottom_choices = [rid for rid in covering_src_choices if rid >= 7]
            if bottom_choices:
                covering_src = int(RNG.choice(bottom_choices))
            else:
                covering_src = int(RNG.choice(covering_src_choices))
        else:
            covering_src = int(RNG.choice(covering_src_choices))

        sources = {covered_region, covering_src}
        if RNG.random() < 0.75 and not any(rid >= 7 for rid in sources):
            sources.add(int(RNG.choice([7, 8, 9])))
        while len(sources) < k:
            sources.add(int(RNG.choice([rid for rid in all_regions if rid not in sources])))
        source_ids = sorted(sources)

        target_ids = {covered_region}
        pool = [rid for rid in upper if rid not in target_ids]
        RNG.shuffle(pool)
        for rid in pool:
            if len(target_ids) == k:
                break
            target_ids.add(int(rid))
        if len(target_ids) != k:
            continue

        source_to_target = {covering_src: covered_region}
        remaining_sources = [rid for rid in source_ids if rid != covering_src]
        remaining_targets = [rid for rid in sorted(target_ids) if rid != covered_region]

        if RNG.random() < 0.45:
            self_candidates = [rid for rid in remaining_sources if rid in remaining_targets]
            if self_candidates:
                self_src = int(RNG.choice(self_candidates))
                source_to_target[self_src] = self_src
                remaining_sources.remove(self_src)
                remaining_targets.remove(self_src)

        RNG.shuffle(remaining_targets)
        for src, tgt in zip(remaining_sources, remaining_targets):
            source_to_target[int(src)] = int(tgt)

        ordered_targets = [source_to_target[rid] for rid in source_ids]
        areas = [REGION_TO_AREA[tgt] for tgt in ordered_targets]
        area_set = set(areas)
        if not any(src in set(ordered_targets) for src in source_ids):
            continue
        if not any(tgt not in source_ids for tgt in ordered_targets):
            continue
        if not any(src != tgt for src, tgt in zip(source_ids, ordered_targets)):
            continue
        if not any(area < 8 and area + 1 not in area_set for area in areas):
            continue
        if ordered_targets == sorted(ordered_targets):
            continue
        return source_to_target
    return None


def _color_probe(inp: np.ndarray, glyphs: list[dict], regions: list[tuple[int, int, int, int]]) -> bool:
    if len(glyphs) < 2:
        return False
    g1, g2 = glyphs[0], glyphs[-1]
    variant = np.array(inp, copy=True)
    for glyph, new_color in ((g1, g2["color"]), (g2, g1["color"])):
        r0, c0, r1, c1 = regions[glyph["source_id"] - 1]
        tile = variant[r0 : r1 + 1, c0 : c1 + 1]
        tile[glyph["mask"]] = int(new_color)
    out = main(variant)
    for glyph, new_color in ((g1, g2["color"]), (g2, g1["color"])):
        r0, c0, r1, c1 = regions[glyph["target_id"] - 1]
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == new_color):
            return False
    return True


def _area_probe(inp: np.ndarray, glyphs: list[dict], regions: list[tuple[int, int, int, int]]) -> bool:
    base_out = main(inp)
    used_areas = {glyph["area"] for glyph in glyphs}
    for glyph in glyphs:
        area = glyph["area"]
        if area >= 8 or area + 1 in used_areas:
            continue
        r0, c0, r1, c1 = regions[glyph["source_id"] - 1]
        tile = inp[r0 : r1 + 1, c0 : c1 + 1]
        mask = glyph["mask"]
        candidates = []
        for rr, cc in np.argwhere(tile == BLACK):
            rr = int(rr)
            cc = int(cc)
            if any(
                0 <= rr + dr < mask.shape[0]
                and 0 <= cc + dc < mask.shape[1]
                and mask[rr + dr, cc + dc]
                for dr, dc in DIR4
            ):
                candidates.append((rr, cc))
        RNG.shuffle(candidates)
        for rr, cc in candidates:
            variant = np.array(inp, copy=True)
            variant[r0 + rr, c0 + cc] = glyph["color"]
            out = main(variant)
            new_target = AREA_TO_REGION[area + 1]
            nr0, nc0, nr1, nc1 = regions[new_target - 1]
            or0, oc0, or1, oc1 = regions[glyph["target_id"] - 1]
            if not np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == glyph["color"]):
                continue
            if np.all(out[or0 : or1 + 1, oc0 : oc1 + 1] == glyph["color"]):
                continue
            if np.array_equal(out, base_out):
                continue
            return True
    return False


def _semantic_check(inp: np.ndarray, metadata: list[dict]) -> bool:
    try:
        regions, glyphs = _parse_grid(inp)
        if not (3 <= len(glyphs) <= 5):
            return False
        parsed = sorted((g["source_id"], g["target_id"], g["area"], g["color"]) for g in glyphs)
        expected = sorted((m["source_id"], m["target_id"], m["area"], m["color"]) for m in metadata)
        if parsed != expected:
            return False

        source_ids = [g["source_id"] for g in glyphs]
        target_ids = [g["target_id"] for g in glyphs]
        target_to_color = {g["target_id"]: g["color"] for g in glyphs}
        if len(set(g["color"] for g in glyphs)) != len(glyphs):
            return False
        if len(set(g["area"] for g in glyphs)) != len(glyphs):
            return False
        if not any(src in set(target_ids) for src in source_ids):
            return False
        if not any(tgt not in source_ids for tgt in target_ids):
            return False
        if not any(src != tgt for src, tgt in zip(source_ids, target_ids)):
            return False

        out = main(inp)
        oracle = _oracle_output(inp, glyphs, regions)
        if not np.array_equal(out, oracle):
            return False
        if not np.all(out[inp == GREY] == GREY):
            return False

        for region_id, (r0, c0, r1, c1) in enumerate(regions, start=1):
            in_tile = inp[r0 : r1 + 1, c0 : c1 + 1]
            out_tile = out[r0 : r1 + 1, c0 : c1 + 1]
            if region_id in target_to_color:
                if set(map(int, np.unique(out_tile))) != {target_to_color[region_id]}:
                    return False
            elif not np.array_equal(in_tile, out_tile):
                return False

        return _color_probe(inp, glyphs, regions) and _area_probe(inp, glyphs, regions)
    except Exception:
        return False


def _build_example() -> np.ndarray | None:
    row_bands = _sample_band_lengths()
    col_bands = _sample_band_lengths()
    h = sum(row_bands) + 2
    w = sum(col_bands) + 2
    if max(h, w) > 30:
        return None

    regions = _regions_from_bands(row_bands, col_bands)
    k = int(RNG.integers(3, 6))
    source_to_target = _choose_sources_and_targets(k)
    if source_to_target is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    sep_rows = [row_bands[0], row_bands[0] + row_bands[1] + 1]
    sep_cols = [col_bands[0], col_bands[0] + col_bands[1] + 1]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    source_ids = sorted(source_to_target)
    colors = [int(x) for x in RNG.choice(np.array(GLYPH_COLORS, dtype=int), size=k, replace=False)]
    metadata: list[dict] = []
    for source_id, color in zip(source_ids, colors):
        target_id = int(source_to_target[source_id])
        area = int(REGION_TO_AREA[target_id])
        mask = _place_glyph(grid, regions[source_id - 1], area, color)
        metadata.append(
            {
                "source_id": int(source_id),
                "target_id": int(target_id),
                "area": int(area),
                "color": int(color),
                "mask": mask,
            }
        )

    if not _semantic_check(grid, metadata):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 7")
