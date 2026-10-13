from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_mold(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> tuple[int, int, int, int] | None:
    if int(coords.shape[0]) <= 1:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not set(map(int, np.unique(sub))) <= {int(bg), int(color)}:
        return None
    if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
        return None
    if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
        return None
    if np.count_nonzero(sub == color) != int(coords.shape[0]):
        return None
    interior = sub[1:-1, 1:-1]
    if interior.size == 0 or not np.any(interior == bg):
        return None
    return r0, c0, r1, c1


def _extract_molds_and_markers(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    result: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        mold_bbox = None
        markers: list[tuple[int, int]] = []
        for comp in comps:
            if int(comp.shape[0]) == 1:
                r, c = map(int, comp[0])
                markers.append((r, c))
                continue
            candidate = _is_mold(g, comp, color, bg)
            assert candidate is not None, f"unexpected non-rectangular object for color {color}"
            assert mold_bbox is None, f"multiple molds for color {color}"
            mold_bbox = candidate

        assert mold_bbox is not None, f"missing mold for color {color}"
        assert markers, f"missing marker for color {color}"
        r0, c0, r1, c1 = mold_bbox
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        holes: list[tuple[int, int]] = []
        for rr in range(r0 + 1, r1):
            for cc2 in range(c0 + 1, c1):
                if int(g[rr, cc2]) == bg:
                    holes.append((rr - cr, cc2 - cc))
        assert holes, f"mold for color {color} must contain holes"
        result[color] = {"bbox": mold_bbox, "holes": tuple(sorted(holes)), "markers": sorted(markers)}
    return bg, result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each mold's hole pattern as a positive image centered on same-colored anchor markers.

    Input structure:
    - The background contains several odd-sized monochrome rectangular molds.
    - Each mold is solid on its border and has one or more background-colored hole cells strictly inside.
    - Elsewhere there are isolated one-pixel anchor markers; each marker shares its color with exactly one mold.

    Rule:
    - For each mold, compute every hole cell's offset from the mold's center cell.
    - For every same-colored anchor marker, translate that offset set so the marker becomes the new center.
    - Paint those translated cells with the mold color, but only where the current cell is background; preserve all original molds and markers.

    Key insight:
    - The content to copy is the mold's negative space, not its visible solid area, and the copied geometry must stay centered on the marker rather than aligned to a corner.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _extract_molds_and_markers(g)
    out = g.copy()
    h, w = g.shape

    for color, info in objects.items():
        holes = info["holes"]
        for mr, mc in info["markers"]:
            for dr, dc in holes:
                rr = mr + dr
                cc = mc + dc
                if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                    out[rr, cc] = color

    return out


_MAX_GEN_ATTEMPTS = 400
_GRID_MIN = 20
_GRID_MAX = 30
_ODD_SIZES = (5, 7)
_MARKER_COUNT_WEIGHTS = np.array([0.55, 0.45])
_PLACEMENT_GAP = 1


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    occupied[r0:r1, c0:c1] = True


def _free_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    out: list[tuple[int, int]] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if not occupied[top : top + height, left : left + width].any():
                out.append((top, left))
    return out


def _generate_holes(height: int, width: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    drs = list(range(-(height // 2) + 1, height // 2))
    dcs = list(range(-(width // 2) + 1, width // 2))
    candidates = [(dr, dc) for dr in drs for dc in dcs]
    max_holes = min(len(candidates) - 1, max(4, len(candidates) // 2))
    for _ in range(200):
        count = int(rng.integers(3, max_holes + 1))
        idx = rng.choice(len(candidates), size=count, replace=False)
        holes = tuple(sorted(candidates[int(i)] for i in idx))
        hole_drs = {dr for dr, _ in holes}
        hole_dcs = {dc for _, dc in holes}
        if len(hole_drs) < 2 or len(hole_dcs) < 2:
            continue
        if not (min(hole_drs) < 0 and max(hole_drs) > 0):
            continue
        if not (min(hole_dcs) < 0 and max(hole_dcs) > 0):
            continue
        return holes
    raise RuntimeError("failed to generate non-degenerate hole pattern")


def _build_expected(
    inp: np.ndarray,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    bg = _get_bg(inp)
    h, w = inp.shape
    for spec in specs:
        color = int(spec["color"])
        holes = spec["holes"]
        for mr, mc in spec["markers"]:
            for dr, dc in holes:
                rr = mr + dr
                cc = mc + dc
                if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                    out[rr, cc] = color
    return out


def _structure_checks(inp: np.ndarray, specs: list[dict[str, object]], expected: np.ndarray) -> bool:
    try:
        bg, parsed = _extract_molds_and_markers(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    added = (inp == bg) & (expected != inp)
    if not np.any(added):
        return False

    if set(parsed) != {int(spec["color"]) for spec in specs}:
        return False

    seen_hole_signatures = set()
    for spec in specs:
        color = int(spec["color"])
        info = parsed[color]
        if tuple(info["holes"]) != tuple(spec["holes"]):
            return False
        if list(info["markers"]) != sorted(spec["markers"]):
            return False
        bbox = tuple(info["bbox"])
        if bbox != tuple(spec["bbox"]):
            return False
        signature = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, tuple(info["holes"]))
        if signature in seen_hole_signatures:
            return False
        seen_hole_signatures.add(signature)

        for mr, mc in info["markers"]:
            rr0 = max(0, mr - 1)
            cc0 = max(0, mc - 1)
            rr1 = min(inp.shape[0], mr + 2)
            cc1 = min(inp.shape[1], mc + 2)
            patch = inp[rr0:rr1, cc0:cc1].copy()
            patch[mr - rr0, mc - cc0] = bg
            if np.any(patch != bg):
                return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    mold_count = int(rng.choice([2, 3], p=[0.55, 0.45]))
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    colors = palette[:mold_count]

    specs: list[dict[str, object]] = []
    total_reserved_area = 0
    for color in colors:
        height = int(rng.choice(_ODD_SIZES))
        width = int(rng.choice(_ODD_SIZES))
        holes = _generate_holes(height, width, rng)
        marker_count = int(rng.choice([1, 2], p=_MARKER_COUNT_WEIGHTS))
        specs.append(
            {
                "color": color,
                "height": height,
                "width": width,
                "holes": holes,
                "marker_count": marker_count,
            }
        )
        total_reserved_area += (1 + marker_count) * height * width

    min_side = max(_GRID_MIN, int(np.ceil(np.sqrt(total_reserved_area))) + 6)
    if min_side > _GRID_MAX:
        return None
    grid_h = int(rng.integers(min_side, _GRID_MAX + 1))
    grid_w = int(rng.integers(min_side, _GRID_MAX + 1))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[tuple[int, str, int], tuple[int, int]] = {}
    items: list[tuple[int, int, int, str, int]] = []
    for idx, spec in enumerate(specs):
        height = int(spec["height"])
        width = int(spec["width"])
        area = height * width
        items.append((area, idx, -1, "mold", 0))
        for marker_idx in range(int(spec["marker_count"])):
            items.append((area, idx, marker_idx, "stamp_zone", marker_idx))
    rng.shuffle(items)
    items.sort(key=lambda item: item[0], reverse=True)

    for _area, spec_idx, marker_idx, kind, key_idx in items:
        spec = specs[spec_idx]
        height = int(spec["height"])
        width = int(spec["width"])
        positions = _free_positions(occupied, height, width)
        if not positions:
            return None
        top, left = positions[int(rng.integers(0, len(positions)))]
        _occupy(occupied, top, left, height, width, _PLACEMENT_GAP)
        placements[(spec_idx, kind, key_idx)] = (top, left)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    for spec_idx, spec in enumerate(specs):
        color = int(spec["color"])
        height = int(spec["height"])
        width = int(spec["width"])
        top, left = placements[(spec_idx, "mold", 0)]
        inp[top : top + height, left : left + width] = color
        center_r = top + height // 2
        center_c = left + width // 2
        for dr, dc in spec["holes"]:
            inp[center_r + dr, center_c + dc] = bg
        spec["bbox"] = (top, left, top + height - 1, left + width - 1)

        markers: list[tuple[int, int]] = []
        for marker_idx in range(int(spec["marker_count"])):
            zone_top, zone_left = placements[(spec_idx, "stamp_zone", marker_idx)]
            mr = zone_top + height // 2
            mc = zone_left + width // 2
            inp[mr, mc] = color
            markers.append((mr, mc))
        spec["markers"] = sorted(markers)

    expected = _build_expected(inp, specs)
    if not _structure_checks(inp, specs, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mold-stamp input")
