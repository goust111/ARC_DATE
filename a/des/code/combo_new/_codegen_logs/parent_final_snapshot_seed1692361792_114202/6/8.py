from __future__ import annotations

import random

import numpy as np


YELLOW = 4
ORANGE = 7
GREY = 5
OBJECT_COLORS = (1, 2, 3, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _infer_stripes(grid: np.ndarray) -> tuple[int, int]:
    score_yellow_even = 0
    score_orange_even = 0
    for r in range(grid.shape[0]):
        expected_a = YELLOW if r % 2 == 0 else ORANGE
        expected_b = ORANGE if r % 2 == 0 else YELLOW
        score_yellow_even += int(np.count_nonzero(grid[r] == expected_a))
        score_orange_even += int(np.count_nonzero(grid[r] == expected_b))
    if score_yellow_even >= score_orange_even:
        return YELLOW, ORANGE
    return ORANGE, YELLOW


def _bg_for_row(r: int, even_color: int, odd_color: int) -> int:
    return int(even_color if r % 2 == 0 else odd_color)


def _same_color_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_objects(grid: np.ndarray, stripe_colors: set[int]) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in stripe_colors or color == GREY:
            continue
        comps = _same_color_components(grid == color)
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            objects.append(
                {
                    "color": color,
                    "coords": tuple(sorted(comp)),
                    "min_r": min(rows),
                    "max_r": max(rows),
                    "max_c": max(cols),
                    "row_count": len(set(rows)),
                }
            )
    assert objects, "expected at least one foreground object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each foreground object into a grey shadow whose vertical offset equals the number of striped rows it covers.

    Input structure:
    - The background is a two-color horizontal stripe pattern that alternates every row.
    - Several 4-connected monochrome foreground objects use colors different from both stripe colors.
    - Each object has one grey marker exactly one column to the right of its bounding box, aligned with either the object's top row or bottom row.
    - A top-aligned marker means the shadow points downward, while a bottom-aligned marker means the shadow points upward.

    Rule:
    - For each object, count the distinct grid rows occupied by that object; this count is the shadow offset N.
    - Erase the grey marker and restore the stripe background color underneath it.
    - Translate the whole object shape by N rows in the marker-indicated vertical direction and paint that translated copy grey.
    - Keep the original colored objects unchanged, and let existing colored objects stay on top if a shadow would overlap them.

    Key insight:
    - The grey marker encodes only the direction; the striped background secretly encodes the distance because the number of stripe rows covered by the object determines the shadow translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    even_color, odd_color = _infer_stripes(g)
    stripe_colors = {int(even_color), int(odd_color)}

    objects = _find_objects(g, stripe_colors)
    marker_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    assert len(marker_coords) == len(objects), "expected one grey marker per object"

    object_mask = np.zeros((h, w), dtype=bool)
    for obj in objects:
        for r, c in obj["coords"]:
            object_mask[r, c] = True

    out = g.copy()
    for r, c in marker_coords:
        out[r, c] = _bg_for_row(r, even_color, odd_color)

    matched_markers: set[tuple[int, int]] = set()
    for obj in objects:
        min_r = int(obj["min_r"])
        max_r = int(obj["max_r"])
        max_c = int(obj["max_c"])
        row_count = int(obj["row_count"])
        assert row_count >= 2, "marker direction is ambiguous for one-row objects"
        assert max_c + 1 < w, "marker must fit to the right of the object"

        top_marker = (min_r, max_c + 1)
        bottom_marker = (max_r, max_c + 1)
        top_ok = top_marker in marker_coords
        bottom_ok = bottom_marker in marker_coords
        assert top_ok ^ bottom_ok, "each object must have exactly one top/bottom right marker"

        direction = 1 if top_ok else -1
        matched_markers.add(top_marker if top_ok else bottom_marker)

        for r, c in obj["coords"]:
            rr = int(r + direction * row_count)
            assert 0 <= rr < h, "shadow must stay in bounds"
            if not object_mask[rr, c]:
                out[rr, c] = GREY

    assert matched_markers == marker_coords, "every grey pixel must be an object marker"
    return out


def _make_stripes(h: int, w: int) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        grid[r, :] = YELLOW if r % 2 == 0 else ORANGE
    return grid


def _crop_cells(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    cropped = sorted((r - min_r, c - min_c) for r, c in cells)
    height = max(r for r, _ in cropped) + 1
    width = max(c for _, c in cropped) + 1
    return cropped, height, width


def _random_shape(direction: int) -> tuple[list[tuple[int, int]], int, int] | None:
    for _ in range(80):
        box_h = random.randint(2, 5)
        box_w = random.randint(1, 5)
        lower = max(3, box_h)
        upper = min(box_h * box_w, 8)
        if upper < lower:
            continue
        target = random.randint(lower, upper)
        forced = (0, box_w - 1) if direction == 1 else (box_h - 1, box_w - 1)
        cells = {forced}
        frontier = [forced]
        while len(cells) < target and frontier:
            base = random.choice(frontier)
            nbrs = [(base[0] + dr, base[1] + dc) for dr, dc in DIRS4]
            random.shuffle(nbrs)
            grew = False
            for nr, nc in nbrs:
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    frontier.append((nr, nc))
                    grew = True
                    break
            if not grew:
                frontier.remove(base)
        if len(cells) < target:
            continue

        cropped, height, width = _crop_cells(cells)
        if height < 2:
            continue
        if direction == 1 and (0, width - 1) not in cropped:
            continue
        if direction == -1 and (height - 1, width - 1) not in cropped:
            continue
        return cropped, height, width
    return None


def _expanded_neighbors(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _shadow_coords(coords: list[tuple[int, int]], direction: int, row_count: int) -> list[tuple[int, int]]:
    return [(r + direction * row_count, c) for r, c in coords]


def _build_expected(h: int, w: int, placements: list[dict[str, object]]) -> np.ndarray:
    out = _make_stripes(h, w)
    object_cells: set[tuple[int, int]] = set()
    for item in placements:
        for r, c in item["coords"]:
            out[r, c] = int(item["color"])
            object_cells.add((r, c))

    for item in placements:
        for r, c in _shadow_coords(item["coords"], int(item["direction"]), int(item["row_count"])):
            if (r, c) not in object_cells:
                out[r, c] = GREY
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(10, 20)
    w = random.randint(12, 20)
    object_count = random.randint(2, 4)
    colors = random.sample(OBJECT_COLORS, object_count)

    down_count = random.randint(1, object_count - 1)
    directions = [1] * down_count + [-1] * (object_count - down_count)
    random.shuffle(directions)

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    order = list(range(object_count))
    random.shuffle(order)
    for idx in order:
        direction = directions[idx]
        color = colors[idx]
        placed = None
        shape = _random_shape(direction)
        if shape is None:
            return None
        local_coords, height, width = shape
        row_count = height

        if direction == 1:
            r_min, r_max = 0, h - 2 * height
        else:
            r_min, r_max = height, h - height
        c_min, c_max = 0, w - width - 1
        if r_min > r_max or c_min > c_max:
            return None

        for _ in range(120):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            coords = [(r0 + r, c0 + c) for r, c in local_coords]
            marker = (r0 if direction == 1 else r0 + height - 1, c0 + width)
            footprint = set(coords) | {marker}
            if footprint & blocked:
                continue
            if any(cell in occupied for cell in footprint):
                continue
            placed = {
                "color": color,
                "coords": coords,
                "direction": direction,
                "row_count": row_count,
                "marker": marker,
            }
            occupied.update(footprint)
            blocked.update(_expanded_neighbors(footprint, h, w))
            placements.append(placed)
            break

        if placed is None:
            return None

    for item in placements:
        shadow = _shadow_coords(item["coords"], int(item["direction"]), int(item["row_count"]))
        if not all(0 <= r < h and 0 <= c < w for r, c in shadow):
            return None
        if not any((r, c) not in occupied for r, c in shadow):
            return None

    inp = _make_stripes(h, w)
    for item in placements:
        for r, c in item["coords"]:
            inp[r, c] = int(item["color"])
        mr, mc = item["marker"]
        inp[mr, mc] = GREY

    expected = _build_expected(h, w, placements)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-shadow ARC input")
