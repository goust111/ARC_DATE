from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
MARKER = 8
INNER_H = 2
SIDES = ("top", "bottom", "left", "right")
USABLE_COLORS = tuple(c for c in range(10) if c not in (BG, FRAME, MARKER))
GEN_ATTEMPTS = 400


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
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _frame_perimeter_size(h: int, w: int) -> int:
    return int(2 * (h + w) - 4)


def _detect_filters(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    filters: list[dict] = []

    for comp in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        oh = int(r1 - r0 + 1)
        ow = int(c1 - c0 + 1)
        if oh != INNER_H + 2 or ow < 4:
            continue
        if len(comp) != _frame_perimeter_size(oh, ow):
            continue
        if not np.all(g[r0, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == FRAME):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == FRAME):
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] != INNER_H or inner.shape[1] < 2:
            continue

        marker_cells: list[tuple[str, tuple[int, int]]] = []
        if r0 - 1 >= 0:
            for cc in range(c0, c1 + 1):
                if int(g[r0 - 1, cc]) == MARKER:
                    marker_cells.append(("top", (int(r0 - 1), int(cc))))
        if r1 + 1 < h:
            for cc in range(c0, c1 + 1):
                if int(g[r1 + 1, cc]) == MARKER:
                    marker_cells.append(("bottom", (int(r1 + 1), int(cc))))
        if c0 - 1 >= 0:
            for rr in range(r0, r1 + 1):
                if int(g[rr, c0 - 1]) == MARKER:
                    marker_cells.append(("left", (int(rr), int(c0 - 1))))
        if c1 + 1 < w:
            for rr in range(r0, r1 + 1):
                if int(g[rr, c1 + 1]) == MARKER:
                    marker_cells.append(("right", (int(rr), int(c1 + 1))))

        if len(marker_cells) != 1:
            continue

        side, marker_pos = marker_cells[0]
        top_row = [int(v) for v in inner[0]]
        bottom_row = [int(v) for v in inner[1]]
        mapping: dict[int, int] = {}
        consistent = True
        for src, dst in zip(top_row, bottom_row):
            if src in mapping and mapping[src] != dst:
                consistent = False
                break
            mapping[int(src)] = int(dst)
        if not consistent:
            continue

        filters.append(
            {
                "side": side,
                "marker": marker_pos,
                "outer_bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                "width": int(inner.shape[1]),
                "mapping": mapping,
            }
        )

    assert filters, "expected at least one filter chip"
    return filters


def _main_bbox(g: np.ndarray, filters: list[dict]) -> tuple[int, int, int, int]:
    keep = g != BG
    for filt in filters:
        r0, c0, r1, c1 = filt["outer_bbox"]
        keep[r0 : r1 + 1, c0 : c1 + 1] = False
        mr, mc = filt["marker"]
        keep[mr, mc] = False

    coords = np.argwhere(keep)
    assert coords.size > 0, "main pattern must remain after removing filters"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    sub = keep[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(sub)), "main pattern must be one dense rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _mapping_region(
    filt: dict,
    main_bbox: tuple[int, int, int, int],
) -> tuple[int, slice, slice]:
    mr0, mc0, mr1, mc1 = main_bbox
    mh = int(mr1 - mr0 + 1)
    mw = int(mc1 - mc0 + 1)
    or0, oc0, or1, oc1 = filt["outer_bbox"]
    ir0, ic0, ir1, ic1 = filt["inner_bbox"]
    width = int(filt["width"])
    side = filt["side"]

    if side == "top":
        dist = int(mr0 - or1 - 1)
        col0 = int(ic0 - mc0)
        assert 0 <= col0 and col0 + width <= mw
        return dist, slice(0, INNER_H), slice(col0, col0 + width)
    if side == "bottom":
        dist = int(or0 - mr1 - 1)
        col0 = int(ic0 - mc0)
        assert 0 <= col0 and col0 + width <= mw
        return dist, slice(mh - INNER_H, mh), slice(col0, col0 + width)
    if side == "left":
        dist = int(mc0 - oc1 - 1)
        row0 = int(ir0 - mr0)
        assert 0 <= row0 and row0 + INNER_H <= mh
        return dist, slice(row0, row0 + INNER_H), slice(0, width)

    assert side == "right"
    dist = int(oc0 - mc1 - 1)
    row0 = int(ir0 - mr0)
    assert 0 <= row0 and row0 + INNER_H <= mh
    return dist, slice(row0, row0 + INNER_H), slice(mw - width, mw)


def _apply_mapping(region: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    base = np.array(region, copy=True)
    out = np.array(region, copy=True)
    for src, dst in mapping.items():
        out[base == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply color-replacement filter chips to the large central rectangle.

    Each grey-framed chip has a 2 x k interior. The top interior row maps to the bottom row.
    The azure marker gives the entry side: top/bottom filters affect the first/last 2 rows,
    left/right filters affect the first/last k columns, while the chip's orthogonal span selects
    the corresponding columns or rows. Filters are applied from the farthest chip to the nearest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    filters = _detect_filters(g)
    main_bbox = _main_bbox(g, filters)
    r0, c0, r1, c1 = main_bbox
    main_rect = np.array(g[r0 : r1 + 1, c0 : c1 + 1], copy=True)

    ordered: list[tuple[int, slice, slice, dict[int, int]]] = []
    for filt in filters:
        dist, rs, cs = _mapping_region(filt, main_bbox)
        assert dist >= 1, "filters must start outside the main pattern"
        ordered.append((int(dist), rs, cs, filt["mapping"]))

    dists = [dist for dist, _rs, _cs, _map in ordered]
    assert len(dists) == len(set(dists)), "filter order must be unambiguous"
    ordered.sort(key=lambda item: int(item[0]), reverse=True)

    for _dist, rs, cs, mapping in ordered:
        main_rect[rs, cs] = _apply_mapping(main_rect[rs, cs], mapping)

    return main_rect


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _make_main_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(("tile22", "tile23", "tile32", "tile33"))
    cols = random.sample(palette, 4)

    if mode == "tile22":
        tile = np.array([[cols[0], cols[1]], [cols[2], cols[3]]], dtype=int)
    elif mode == "tile23":
        tile = np.array([[cols[0], cols[1], cols[2]], [cols[2], cols[3], cols[0]]], dtype=int)
    elif mode == "tile32":
        tile = np.array([[cols[0], cols[1]], [cols[2], cols[3]], [cols[1], cols[0]]], dtype=int)
    else:
        tile = np.array(
            [[cols[0], cols[1], cols[2]], [cols[2], cols[3], cols[0]], [cols[1], cols[0], cols[3]]],
            dtype=int,
        )

    pattern = _repeat_tile(tile, h, w)
    if random.random() < 0.5:
        stripe_color = random.choice(palette)
        if random.random() < 0.5:
            pattern[:, w // 2] = stripe_color
        else:
            pattern[h // 2, :] = stripe_color
    return pattern


def _random_derangement(values: list[int], palette: list[int]) -> list[int] | None:
    pool = list(palette)
    for _ in range(60):
        targets = random.sample(pool, len(values))
        if all(int(a) != int(b) for a, b in zip(values, targets)):
            return [int(v) for v in targets]
    return None


def _choose_specs(main_h: int, main_w: int, directions: list[str]) -> dict[str, dict] | None:
    if main_h < INNER_H or main_w < 2:
        return None
    max_width = min(4, main_w)
    if max_width < 2:
        return None

    specs: dict[str, dict] = {}
    for side in directions:
        specs[side] = {
            "side": side,
            "distance": None,
            "width": random.randint(2, max_width),
        }

    distances = random.sample(range(1, 7), len(directions))
    random.shuffle(distances)
    for side, dist in zip(directions, distances):
        specs[side]["distance"] = int(dist)

    available_corners: list[tuple[str, str, str]] = []
    if "top" in specs and "left" in specs:
        available_corners.append(("top", "left", "tl"))
    if "top" in specs and "right" in specs:
        available_corners.append(("top", "right", "tr"))
    if "bottom" in specs and "left" in specs:
        available_corners.append(("bottom", "left", "bl"))
    if "bottom" in specs and "right" in specs:
        available_corners.append(("bottom", "right", "br"))

    chosen_corners: list[str] = []
    used_sides: set[str] = set()
    random.shuffle(available_corners)
    for side_a, side_b, corner in available_corners:
        if side_a in used_sides or side_b in used_sides:
            continue
        if random.random() < 0.6:
            used_sides.add(side_a)
            used_sides.add(side_b)
            chosen_corners.append(corner)

    if "top" in specs:
        k = int(specs["top"]["width"])
        if "tl" in chosen_corners:
            specs["top"]["col0"] = 0
        elif "tr" in chosen_corners:
            specs["top"]["col0"] = int(main_w - k)
        else:
            specs["top"]["col0"] = random.randint(0, main_w - k)

    if "bottom" in specs:
        k = int(specs["bottom"]["width"])
        if "bl" in chosen_corners:
            specs["bottom"]["col0"] = 0
        elif "br" in chosen_corners:
            specs["bottom"]["col0"] = int(main_w - k)
        else:
            specs["bottom"]["col0"] = random.randint(0, main_w - k)

    if "left" in specs:
        if "tl" in chosen_corners:
            specs["left"]["row0"] = 0
        elif "bl" in chosen_corners:
            specs["left"]["row0"] = int(main_h - INNER_H)
        else:
            specs["left"]["row0"] = random.randint(0, main_h - INNER_H)

    if "right" in specs:
        if "tr" in chosen_corners:
            specs["right"]["row0"] = 0
        elif "br" in chosen_corners:
            specs["right"]["row0"] = int(main_h - INNER_H)
        else:
            specs["right"]["row0"] = random.randint(0, main_h - INNER_H)

    return specs


def _spec_region(spec: dict, shape: tuple[int, int]) -> tuple[slice, slice]:
    h, w = shape
    width = int(spec["width"])
    side = spec["side"]
    if side == "top":
        col0 = int(spec["col0"])
        return slice(0, INNER_H), slice(col0, col0 + width)
    if side == "bottom":
        col0 = int(spec["col0"])
        return slice(h - INNER_H, h), slice(col0, col0 + width)
    if side == "left":
        row0 = int(spec["row0"])
        return slice(row0, row0 + INNER_H), slice(0, width)
    assert side == "right"
    row0 = int(spec["row0"])
    return slice(row0, row0 + INNER_H), slice(w - width, w)


def _finalize_specs(main_rect: np.ndarray, specs: dict[str, dict], palette: list[int]) -> tuple[dict[str, dict], np.ndarray] | None:
    current = np.array(main_rect, copy=True)
    ordered_sides = sorted(specs, key=lambda side: int(specs[side]["distance"]), reverse=True)

    for side in ordered_sides:
        spec = specs[side]
        rs, cs = _spec_region(spec, current.shape)
        region = current[rs, cs]
        unique_colors = sorted({int(v) for v in region.reshape(-1)})
        if len(unique_colors) < 2:
            return None

        width = int(spec["width"])
        basis_count = random.randint(2, min(3, len(unique_colors), width))
        basis = random.sample(unique_colors, basis_count)
        targets = _random_derangement(basis, palette)
        if targets is None:
            return None
        mapping = {int(src): int(dst) for src, dst in zip(basis, targets)}

        top_row = basis[:]
        while len(top_row) < width:
            top_row.append(random.choice(basis))
        random.shuffle(top_row)
        top_row = [int(v) for v in top_row[:width]]
        bottom_row = [int(mapping[v]) for v in top_row]

        updated = _apply_mapping(region, mapping)
        if np.array_equal(updated, region):
            return None

        spec["top_row"] = top_row
        spec["bottom_row"] = bottom_row
        spec["mapping"] = mapping
        current[rs, cs] = updated

    return specs, current


def _margin_requirements(specs: dict[str, dict]) -> tuple[int, int, int, int]:
    top = 1
    bottom = 1
    left = 1
    right = 1
    for spec in specs.values():
        side = spec["side"]
        dist = int(spec["distance"])
        width = int(spec["width"])
        if side == "top":
            top = max(top, dist + 5)
        elif side == "bottom":
            bottom = max(bottom, dist + 5)
        elif side == "left":
            left = max(left, dist + width + 3)
        else:
            right = max(right, dist + width + 3)
    return int(top), int(bottom), int(left), int(right)


def _paint_filter(
    grid: np.ndarray,
    occupied: np.ndarray,
    spec: dict,
    main_origin: tuple[int, int],
    main_shape: tuple[int, int],
) -> bool:
    main_r0, main_c0 = main_origin
    main_h, main_w = main_shape
    main_r1 = int(main_r0 + main_h - 1)
    main_c1 = int(main_c0 + main_w - 1)
    width = int(spec["width"])
    dist = int(spec["distance"])
    side = spec["side"]

    if side == "top":
        outer_r0 = int(main_r0 - dist - 4)
        outer_c0 = int(main_c0 + spec["col0"] - 1)
        marker = (int(outer_r0 - 1), int(outer_c0 + 1 + width // 2))
    elif side == "bottom":
        outer_r0 = int(main_r1 + dist + 1)
        outer_c0 = int(main_c0 + spec["col0"] - 1)
        marker = (int(outer_r0 + 4), int(outer_c0 + 1 + width // 2))
    elif side == "left":
        outer_r0 = int(main_r0 + spec["row0"] - 1)
        outer_c0 = int(main_c0 - dist - width - 2)
        marker = (int(outer_r0 + 1), int(outer_c0 - 1))
    else:
        outer_r0 = int(main_r0 + spec["row0"] - 1)
        outer_c0 = int(main_c1 + dist + 1)
        marker = (int(outer_r0 + 1), int(outer_c0 + width + 2))

    patch = np.full((4, width + 2), FRAME, dtype=int)
    patch[1, 1 : width + 1] = np.array(spec["top_row"], dtype=int)
    patch[2, 1 : width + 1] = np.array(spec["bottom_row"], dtype=int)

    ph, pw = patch.shape
    if outer_r0 < 0 or outer_c0 < 0 or outer_r0 + ph > grid.shape[0] or outer_c0 + pw > grid.shape[1]:
        return False
    mr, mc = marker
    if not (0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1]):
        return False

    mask = np.zeros_like(grid, dtype=bool)
    mask[outer_r0 : outer_r0 + ph, outer_c0 : outer_c0 + pw] = True
    mask[mr, mc] = True
    if bool(np.any(occupied & mask)):
        return False

    grid[outer_r0 : outer_r0 + ph, outer_c0 : outer_c0 + pw] = patch
    grid[mr, mc] = MARKER
    occupied |= mask
    return True


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    main_h = random.randint(8, 14)
    main_w = random.randint(8, 14)
    n_filters = random.randint(1, 4)
    directions = random.sample(list(SIDES), n_filters)

    palette = random.sample(list(USABLE_COLORS), random.randint(4, len(USABLE_COLORS)))
    main_rect = _make_main_pattern(main_h, main_w, palette)

    specs = _choose_specs(main_h, main_w, directions)
    if specs is None:
        raise ValueError("could not choose filter specs")

    finalized = _finalize_specs(main_rect, specs, palette)
    if finalized is None:
        raise ValueError("could not finalize mappings")
    specs, out_rect = finalized

    top, bottom, left, right = _margin_requirements(specs)
    top += random.randint(0, 2)
    bottom += random.randint(0, 2)
    left += random.randint(0, 2)
    right += random.randint(0, 2)

    h = int(top + main_h + bottom)
    w = int(left + main_w + right)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    main_origin = (int(top), int(left))
    grid[top : top + main_h, left : left + main_w] = main_rect
    occupied[top : top + main_h, left : left + main_w] = True

    for side in specs:
        if not _paint_filter(grid, occupied, specs[side], main_origin, (main_h, main_w)):
            raise ValueError("filter placement failed")

    return grid, out_rect


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, out = _build_example()
        except ValueError:
            continue
        if np.array_equal(main(grid), out):
            return grid
    raise RuntimeError("failed to generate a valid example")
