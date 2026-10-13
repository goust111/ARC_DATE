from __future__ import annotations

import math

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_QUADRANTS = ("tl", "tr", "br", "bl")
_SIDE_CHOICES = (2, 3, 4, 5)
_MAX_GENERATION_ATTEMPTS = 300


def _get_background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _local_quadrant(r: int, c: int, h: int, w: int) -> str:
    top = 2 * int(r) < int(h)
    left = 2 * int(c) < int(w)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and not left:
        return "br"
    return "bl"


def _marker_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    counts: dict[int, int] = {}
    for r, c in coords:
        color = int(grid[r, c])
        counts[color] = counts.get(color, 0) + 1

    marker_colors = [color for color, count in counts.items() if count == 1]
    if len(marker_colors) == 1:
        marker_color = int(marker_colors[0])
    else:
        marker_color = int(min(counts.items(), key=lambda item: (item[1], item[0]))[0])

    marker_coords = [(r, c) for r, c in coords if int(grid[r, c]) == marker_color]
    assert len(marker_coords) == 1
    return marker_coords[0], marker_color


def _spiral_coords_tl(size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top = left = 0
    bottom = right = int(size - 1)

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, int(c)))
        for r in range(top + 1, bottom + 1):
            coords.append((int(r), right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, int(c)))
        if left < right:
            for r in range(bottom - 1, top, -1):
                coords.append((int(r), left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1

    return coords


def _rotate_coord_clockwise(r: int, c: int, size: int, turns: int) -> tuple[int, int]:
    rr = int(r)
    cc = int(c)
    for _ in range(int(turns) % 4):
        rr, cc = cc, int(size - 1 - rr)
    return rr, cc


def _spiral_coords(size: int, quadrant: str) -> list[tuple[int, int]]:
    turns = {"tl": 0, "tr": 1, "br": 2, "bl": 3}[quadrant]
    base = _spiral_coords_tl(size)
    return [_rotate_coord_clockwise(r, c, size, turns) for r, c in base]


def main(grid: np.ndarray) -> np.ndarray:
    """Pack each marked 4-connected object into a centered square spiral."""
    g = np.asarray(grid, dtype=int)
    bg = _get_background(g)
    out = np.full_like(g, bg)

    for coords in _find_components(g, bg):
        ordered = sorted((int(r), int(c)) for r, c in coords)
        area = len(ordered)
        side = int(math.isqrt(area))
        assert side * side == area

        (mr, mc), _marker_color = _marker_info(g, ordered)
        r0, c0, r1, c1 = _bbox(ordered)
        bbox_h = int(r1 - r0 + 1)
        bbox_w = int(c1 - c0 + 1)
        quadrant = _local_quadrant(int(mr - r0), int(mc - c0), bbox_h, bbox_w)
        colors = [int(g[r, c]) for r, c in ordered]
        path = _spiral_coords(side, quadrant)

        # Center the new SxS canvas on the original bbox center.
        dest_r0 = int((r0 + r1 - (side - 1)) // 2)
        dest_c0 = int((c0 + c1 - (side - 1)) // 2)

        for color, (dr, dc) in zip(colors, path):
            out[int(dest_r0 + dr), int(dest_c0 + dc)] = int(color)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    start_cells = np.argwhere(mask)
    if len(start_cells) == 0:
        return False

    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, start_cells[0])
    seen[start_r, start_c] = True
    stack = [(start_r, start_c)]
    seen_count = 0

    while stack:
        r, c = stack.pop()
        seen_count += 1
        for dr, dc in _NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return seen_count == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _marker_candidates(mask: np.ndarray, quadrant: str, avoid_center_lines: bool) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if _local_quadrant(rr, cc, h, w) != quadrant:
            continue
        if avoid_center_lines and (2 * rr + 1 == h or 2 * cc + 1 == w):
            continue
        candidates.append((rr, cc))
    return candidates


def _generate_shape(area: int, h: int, w: int, quadrant: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    slack = int(h * w - area)
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        ok = True

        for _ in range(slack):
            cells = np.argwhere(mask)
            _RNG.shuffle(cells)
            removable: list[tuple[int, int]] = []

            for r, c in cells:
                rr = int(r)
                cc = int(c)
                trial = mask.copy()
                trial[rr, cc] = False
                if not _touches_all_sides(trial):
                    continue
                if not _mask_connected(trial):
                    continue
                removable.append((rr, cc))

            if not removable:
                ok = False
                break

            cut_r, cut_c = removable[int(_RNG.integers(0, len(removable)))]
            mask[cut_r, cut_c] = False

        if not ok or int(mask.sum()) != area or not _touches_all_sides(mask) or not _mask_connected(mask):
            continue

        marker_cells = _marker_candidates(mask, quadrant, avoid_center_lines=True)
        if not marker_cells:
            marker_cells = _marker_candidates(mask, quadrant, avoid_center_lines=False)
        if not marker_cells:
            continue

        marker = marker_cells[int(_RNG.integers(0, len(marker_cells)))]
        return mask, marker

    return None


def _bbox_choices(side: int) -> list[tuple[int, int]]:
    if side == 2:
        return [(2, 2), (2, 3), (3, 2)]
    return [(side, side), (side, side + 1), (side + 1, side), (side + 1, side + 1)]


def _choose_bbox(side: int) -> tuple[int, int]:
    choices = _bbox_choices(side)
    if len(choices) > 1 and _RNG.random() < 0.75:
        return choices[1 + int(_RNG.integers(0, len(choices) - 1))]
    return choices[0]


def _rect_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad <= br0 or br1 + pad <= ar0 or ac1 + pad <= bc0 or bc1 + pad <= ac0)


def _start_positions(total: int, size: int) -> list[int]:
    max_start = int(total - size)
    if max_start < 0:
        return []
    if max_start >= 2:
        return list(range(1, max_start))
    return list(range(max_start + 1))


def _place_specs(specs: list[dict], h: int, w: int) -> bool:
    bbox_rects: list[tuple[int, int, int, int]] = []
    out_rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(specs)), key=lambda i: specs[i]["bbox_h"] * specs[i]["bbox_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        row_choices = _start_positions(h, spec["bbox_h"])
        col_choices = _start_positions(w, spec["bbox_w"])
        if not row_choices or not col_choices:
            return False

        positions = [(r0, c0) for r0 in row_choices for c0 in col_choices]
        _RNG.shuffle(positions)
        placed = False

        for r0, c0 in positions:
            out_r0 = int(r0 + (spec["bbox_h"] - spec["side"]) // 2)
            out_c0 = int(c0 + (spec["bbox_w"] - spec["side"]) // 2)
            bbox_rect = (int(r0), int(c0), int(r0 + spec["bbox_h"]), int(c0 + spec["bbox_w"]))
            out_rect = (out_r0, out_c0, int(out_r0 + spec["side"]), int(out_c0 + spec["side"]))

            if any(_rect_conflict(bbox_rect, other, pad=1) for other in bbox_rects):
                continue
            if any(_rect_conflict(out_rect, other, pad=1) for other in out_rects):
                continue

            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            spec["out_r0"] = out_r0
            spec["out_c0"] = out_c0
            bbox_rects.append(bbox_rect)
            out_rects.append(out_rect)
            placed = True
            break

        if not placed:
            return False

    return True


def _build_specs(bg: int, count: int) -> list[dict]:
    colors = [color for color in range(10) if color != bg]
    _RNG.shuffle(colors)

    sides = _RNG.choice(np.array(_SIDE_CHOICES, dtype=int), size=count, replace=True)
    if count >= 2 and len(set(map(int, sides.tolist()))) == 1:
        idx = int(_RNG.integers(0, count))
        alt = [side for side in _SIDE_CHOICES if side != int(sides[idx])]
        sides[idx] = int(alt[int(_RNG.integers(0, len(alt)))])

    quadrants = list(_QUADRANTS)
    _RNG.shuffle(quadrants)

    specs: list[dict] = []
    for i in range(count):
        side = int(sides[i])
        bbox_h, bbox_w = _choose_bbox(side)
        specs.append(
            {
                "side": side,
                "area": side * side,
                "bbox_h": int(bbox_h),
                "bbox_w": int(bbox_w),
                "quadrant": quadrants[i],
                "body": int(colors[2 * i]),
                "marker": int(colors[2 * i + 1]),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    object_count = int(_RNG.integers(2, 5))
    bg = int(_RNG.integers(0, 10))
    grid_h = int(_RNG.integers(18, 31))
    grid_w = int(_RNG.integers(18, 31))

    specs = _build_specs(bg, object_count)
    if not _place_specs(specs, grid_h, grid_w):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for spec in specs:
        built = _generate_shape(spec["area"], spec["bbox_h"], spec["bbox_w"], spec["quadrant"])
        if built is None:
            return None
        mask, (mr, mc) = built
        spec["mask"] = mask
        spec["marker_local"] = (int(mr), int(mc))

        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        region = grid[r0 : r0 + spec["bbox_h"], c0 : c0 + spec["bbox_w"]]
        region[mask] = int(spec["body"])
        grid[int(r0 + mr), int(c0 + mc)] = int(spec["marker"])

    if _get_background(grid) != bg:
        return None

    components = _find_components(grid, bg)
    if len(components) != object_count:
        return None
    if any(len(component) not in {4, 9, 16, 25} for component in components):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(_find_components(out, bg)) != object_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
