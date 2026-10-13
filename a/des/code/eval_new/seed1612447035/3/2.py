from __future__ import annotations

from collections import defaultdict
import numpy as np


ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Random fragment templates keyed by area. All are 4-connected and avoid
# portal-like length>=3 straight segments.
SHAPE_LIBRARY: dict[int, list[np.ndarray]] = {
    2: [
        np.array([[1, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
}

MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 120
PACK_GAP = 2


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ORTHO4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_portal(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    area = int(comp["area"])
    coords = set(comp["coords"])
    if r0 == r1 and area >= 3 and area == c1 - c0 + 1:
        return all((r0, cc) in coords for cc in range(c0, c1 + 1))
    if c0 == c1 and area >= 3 and area == r1 - r0 + 1:
        return all((rr, c0) in coords for rr in range(r0, r1 + 1))
    return False


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _normalized_shape_key(comp: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in comp["coords"]))


def _extract_groups(grid: np.ndarray) -> list[dict]:
    bg = _most_common_color(grid)
    comps = _find_components(grid, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    groups: list[dict] = []
    for color, same_color in by_color.items():
        portals = [comp for comp in same_color if _is_portal(comp)]
        assert len(portals) == 1, "each color must define exactly one portal"
        portal = portals[0]
        fragments = [comp for comp in same_color if comp is not portal]
        assert 2 <= len(fragments) <= 4, "each portal must have 2-4 fragments"

        r0, c0, r1, c1 = map(int, portal["bbox"])
        if r0 == r1:
            orientation = "horizontal"
            portal_row = r0
            above = all(int(f["bbox"][2]) < portal_row for f in fragments)
            below = all(int(f["bbox"][0]) > portal_row for f in fragments)
            assert above ^ below, "all horizontal fragments must lie on one side"
            side = "above" if above else "below"
        else:
            orientation = "vertical"
            portal_col = c0
            left = all(int(f["bbox"][3]) < portal_col for f in fragments)
            right = all(int(f["bbox"][1]) > portal_col for f in fragments)
            assert left ^ right, "all vertical fragments must lie on one side"
            side = "left" if left else "right"

        groups.append(
            {
                "color": int(color),
                "portal": portal,
                "orientation": orientation,
                "source_side": side,
                "fragments": fragments,
            }
        )
    return groups


def _sort_fragments(fragments: list[dict]) -> list[dict]:
    return sorted(
        fragments,
        key=lambda comp: (
            int(comp["area"]),
            _component_mask(comp).shape[0],
            _component_mask(comp).shape[1],
            _normalized_shape_key(comp),
            tuple(comp["coords"]),
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport same-colored fragments through their matching line segments, ordering them by area and mirroring them to the opposite side.

    Input structure:
    - bg = most common color.
    - Each foreground color forms one monochrome horizontal or vertical line segment of length >= 3, the portal.
    - The same color also appears in 2-4 separate 4-connected monochrome fragments of area 2-6.
    - All fragments matching a portal lie strictly on the same side of that portal.

    Rule:
    - Group components by color and identify the unique straight line segment of that color as the portal.
    - Sort that color's remaining fragments by area from small to large.
    - Remove the original fragments, then place mirrored copies on the opposite side of the portal: horizontal portals flip fragments top-to-bottom, vertical portals flip them left-to-right.
    - The copies start from the portal's left/top endpoint, stay 1 cell away from the portal, and keep 1-cell gaps between consecutive copies along the portal.

    Key insight: Color picks the destination portal, fragment area determines order along it, and portal orientation determines which mirror reflection is used.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()
    groups = _extract_groups(g)

    # Remove all source fragments before placing mirrored copies.
    for group in groups:
        for frag in group["fragments"]:
            for r, c in frag["coords"]:
                out[int(r), int(c)] = bg

    for group in groups:
        portal = group["portal"]
        orientation = group["orientation"]
        source_side = group["source_side"]
        fragments = _sort_fragments(group["fragments"])
        r0, c0, r1, c1 = map(int, portal["bbox"])
        color = int(group["color"])

        if orientation == "horizontal":
            cursor = c0
            portal_row = r0
            for frag in fragments:
                mask = np.flipud(_component_mask(frag))
                fh, fw = mask.shape
                if source_side == "above":
                    dest_r0 = portal_row + 2
                else:
                    dest_r0 = portal_row - fh - 1
                assert 0 <= dest_r0 <= g.shape[0] - fh
                assert cursor + fw - 1 <= c1, "mirrored fragments must fit along the portal"
                for dr, dc in np.argwhere(mask):
                    rr = dest_r0 + int(dr)
                    cc = cursor + int(dc)
                    assert out[rr, cc] == bg, "mirrored fragments may not overwrite existing objects"
                    out[rr, cc] = color
                cursor += fw + 1
        else:
            cursor = r0
            portal_col = c0
            for frag in fragments:
                mask = np.fliplr(_component_mask(frag))
                fh, fw = mask.shape
                if source_side == "left":
                    dest_c0 = portal_col + 2
                else:
                    dest_c0 = portal_col - fw - 1
                assert 0 <= dest_c0 <= g.shape[1] - fw
                assert cursor + fh - 1 <= r1, "mirrored fragments must fit along the portal"
                for dr, dc in np.argwhere(mask):
                    rr = cursor + int(dr)
                    cc = dest_c0 + int(dc)
                    assert out[rr, cc] == bg, "mirrored fragments may not overwrite existing objects"
                    out[rr, cc] = color
                cursor += fh + 1

    return out


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, int(rng.integers(0, 4)))
    if int(rng.integers(0, 2)):
        out = np.fliplr(out)
    return out


def _random_fragment_mask(area: int, rng: np.random.Generator) -> np.ndarray:
    template = SHAPE_LIBRARY[area][int(rng.integers(0, len(SHAPE_LIBRARY[area])))]
    return _random_transform(template, rng)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _can_place(occupied: np.ndarray, frag: np.ndarray, r0: int, c0: int) -> bool:
    h, w = frag.shape
    blocked = _dilate8(occupied)
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & frag).any())


def _paint_mask(canvas: np.ndarray, occupied: np.ndarray, frag: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = frag.shape
    sub = canvas[r0 : r0 + h, c0 : c0 + w]
    sub[frag] = int(color)
    occupied[r0 : r0 + h, c0 : c0 + w] |= frag


def _place_fragment_random(
    canvas: np.ndarray,
    occupied: np.ndarray,
    frag: np.ndarray,
    color: int,
    row_lo: int,
    row_hi_exclusive: int,
    col_lo: int,
    col_hi_exclusive: int,
    rng: np.random.Generator,
) -> bool:
    h, w = frag.shape
    max_r0 = row_hi_exclusive - h
    max_c0 = col_hi_exclusive - w
    if max_r0 < row_lo or max_c0 < col_lo:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(rng.integers(row_lo, max_r0 + 1))
        c0 = int(rng.integers(col_lo, max_c0 + 1))
        if _can_place(occupied, frag, r0, c0):
            _paint_mask(canvas, occupied, frag, r0, c0, color)
            return True
    return False


def _build_group(bg: int, color: int, rng: np.random.Generator) -> np.ndarray | None:
    orientation = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
    if orientation == "horizontal":
        source_side = "above" if int(rng.integers(0, 2)) == 0 else "below"
    else:
        source_side = "left" if int(rng.integers(0, 2)) == 0 else "right"

    frag_count = int(rng.integers(2, 5))
    areas = [int(x) for x in rng.choice(np.arange(2, 7), size=frag_count, replace=False)]
    rng.shuffle(areas)
    fragments = [{"area": area, "mask": _random_fragment_mask(area, rng)} for area in areas]

    if orientation == "horizontal":
        total_span = sum(int(f["mask"].shape[1]) for f in fragments) + (frag_count - 1)
        portal_len = total_span + int(rng.integers(0, 4))
        max_h = max(int(f["mask"].shape[0]) for f in fragments)
        source_depth = max(max_h + 2, 5) + int(rng.integers(0, 3))
        dest_depth = max_h + 1
        top_space = source_depth if source_side == "above" else dest_depth
        bottom_space = source_depth if source_side == "below" else dest_depth
        zone_h = top_space + 1 + bottom_space
        zone_w = portal_len
        portal_row = top_space
        zone = np.full((zone_h, zone_w), bg, dtype=int)
        occupied = np.zeros((zone_h, zone_w), dtype=bool)
        zone[portal_row, :] = color
        occupied[portal_row, :] = True

        source_order = sorted(fragments, key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)
        for frag in source_order:
            if source_side == "above":
                ok = _place_fragment_random(zone, occupied, frag["mask"], color, 0, portal_row - 1, 0, zone_w, rng)
            else:
                ok = _place_fragment_random(zone, occupied, frag["mask"], color, portal_row + 2, zone_h, 0, zone_w, rng)
            if not ok:
                return None
        return zone

    total_span = sum(int(f["mask"].shape[0]) for f in fragments) + (frag_count - 1)
    portal_len = total_span + int(rng.integers(0, 4))
    max_w = max(int(f["mask"].shape[1]) for f in fragments)
    source_depth = max(max_w + 2, 5) + int(rng.integers(0, 3))
    dest_depth = max_w + 1
    left_space = source_depth if source_side == "left" else dest_depth
    right_space = source_depth if source_side == "right" else dest_depth
    zone_h = portal_len
    zone_w = left_space + 1 + right_space
    portal_col = left_space
    zone = np.full((zone_h, zone_w), bg, dtype=int)
    occupied = np.zeros((zone_h, zone_w), dtype=bool)
    zone[:, portal_col] = color
    occupied[:, portal_col] = True

    source_order = sorted(fragments, key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)
    for frag in source_order:
        if source_side == "left":
            ok = _place_fragment_random(zone, occupied, frag["mask"], color, 0, zone_h, 0, portal_col - 1, rng)
        else:
            ok = _place_fragment_random(zone, occupied, frag["mask"], color, 0, zone_h, portal_col + 2, zone_w, rng)
        if not ok:
            return None
    return zone


def _pack_groups(groups: list[np.ndarray], bg: int, rng: np.random.Generator) -> np.ndarray | None:
    shuffled = list(groups)
    rng.shuffle(shuffled)
    widths = [int(g.shape[1]) for g in shuffled]
    min_limit = max(widths)
    max_limit = min(28, sum(widths) + PACK_GAP * (len(widths) - 1))
    width_limits = list(range(min_limit, max_limit + 1))
    rng.shuffle(width_limits)

    for width_limit in width_limits:
        positions: list[tuple[int, int]] = []
        x = 0
        y = 0
        row_h = 0
        used_w = 0
        for group in shuffled:
            gh, gw = map(int, group.shape)
            if x > 0 and x + gw > width_limit:
                y += row_h + PACK_GAP
                x = 0
                row_h = 0
            positions.append((y, x))
            x += gw + PACK_GAP
            row_h = max(row_h, gh)
            used_w = max(used_w, x - PACK_GAP)
        used_h = y + row_h
        if used_h > 28 or used_w > 28:
            continue

        extra_h = int(rng.integers(0, min(30 - used_h, 6) + 1))
        extra_w = int(rng.integers(0, min(30 - used_w, 6) + 1))
        top = int(rng.integers(0, extra_h + 1))
        left = int(rng.integers(0, extra_w + 1))
        bottom = extra_h - top
        right = extra_w - left
        canvas = np.full((used_h + top + bottom, used_w + left + right), bg, dtype=int)
        for group, (r0, c0) in zip(shuffled, positions):
            rr, cc = np.nonzero(group != bg)
            canvas[top + r0 + rr, left + c0 + cc] = group[rr, cc]
        return canvas
    return None


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    group_count = int(rng.integers(2, 4))
    colors = [int(x) for x in rng.choice([c for c in range(10) if c != bg], size=group_count, replace=False)]
    groups: list[np.ndarray] = []
    for color in colors:
        group = _build_group(bg, color, rng)
        if group is None:
            return None
        groups.append(group)

    grid = _pack_groups(groups, bg, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
