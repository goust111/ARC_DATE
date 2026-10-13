"""ARC task: touching indicators encode stamp direction, color, and rotation."""

from __future__ import annotations

from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _bbox(coords: set[Coord] | list[Coord]) -> BBox:
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return the 4-connected components of a boolean mask in (row, col) coordinates."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[set[Coord]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def _direction(source_bbox: BBox, target_bbox: BBox) -> Coord:
    """Eight-way direction between bbox centers, using doubled integer centers."""
    sr0, sc0, sr1, sc1 = source_bbox
    tr0, tc0, tr1, tc1 = target_bbox
    return _sign((tr0 + tr1) - (sr0 + sr1)), _sign((tc0 + tc1) - (sc0 + sc1))


def _paint_mask_clipped(out: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    """Paste a boolean sprite with clipping, treating False cells as transparent."""
    height, width = out.shape
    side_r, side_c = mask.shape
    rr0, cc0 = max(0, r0), max(0, c0)
    rr1, cc1 = min(height, r0 + side_r), min(width, c0 + side_c)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    local = mask[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
    view = out[rr0:rr1, cc0:cc1]
    view[local] = color


def _stamp_ray(
    out: np.ndarray,
    template_mask: np.ndarray,
    template_bbox: BBox,
    direction: Coord,
    color: int,
    area: int,
) -> set[Coord]:
    """Paint one cumulative-rotation ray and return its in-bounds painted coordinates."""
    height, width = out.shape
    r0, c0, r1, c1 = template_bbox
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1
    dr, dc = direction
    step_turns = area % 4
    painted: set[Coord] = set()

    k = 1
    while True:
        nr0, nc0 = r0 + k * side * dr, c0 + k * side * dc
        if nr0 >= height or nc0 >= width or nr0 + side <= 0 or nc0 + side <= 0:
            break
        rotated = np.rot90(template_mask, -((k * step_turns) % 4))
        for lr, lc in np.argwhere(rotated):
            rr, cc = nr0 + int(lr), nc0 + int(lc)
            if 0 <= rr < height and 0 <= cc < width:
                painted.add((rr, cc))
        _paint_mask_clipped(out, rotated, nr0, nc0, color)
        k += 1
    return painted


def _parse(grid: np.ndarray) -> tuple[int, BBox, np.ndarray, list[dict[str, object]]]:
    """Parse and validate the background, black template, and colored indicators."""
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "background must be uniquely most frequent"
    bg = int(values[int(np.argmax(counts))])
    assert bg != 0, "background must be non-black"

    black_components = _components(grid == 0)
    assert len(black_components) == 1, "expected exactly one 4-connected black template"
    template_coords = black_components[0]
    template_bbox = _bbox(template_coords)
    r0, c0, r1, c1 = template_bbox
    assert r1 - r0 == c1 - c0, "template bounding box must be square"
    template_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == 0
    assert not any(np.array_equal(template_mask, np.rot90(template_mask, turns)) for turns in (1, 2, 3)), (
        "template must be rotationally asymmetric"
    )

    indicators: list[dict[str, object]] = []
    indicator_colors = [int(v) for v in values if int(v) not in (bg, 0)]
    assert indicator_colors, "expected at least one colored indicator"
    for color in indicator_colors:
        comps = _components(grid == color)
        assert len(comps) == 1, "each indicator color must form one component"
        coords = comps[0]
        assert 2 <= len(coords) <= 4, "indicator area must be from two to four pixels"
        assert any(
            (r + dr, c + dc) in template_coords
            for r, c in coords
            for dr, dc in DIR4
        ), "every indicator must touch the template orthogonally"
        indicator_bbox = _bbox(coords)
        direction = _direction(template_bbox, indicator_bbox)
        assert direction != (0, 0), "indicator direction must be well-defined"
        indicators.append(
            {"color": color, "coords": coords, "area": len(coords), "direction": direction}
        )

    directions = [item["direction"] for item in indicators]
    assert len(set(directions)) == len(directions), "at most one indicator may point in each direction"
    return bg, template_bbox, template_mask, indicators


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat a black template in cumulative-rotation stamp rays encoded by touching indicators.

    Input structure:
    - A uniquely most-common non-black background contains one 4-connected black,
      rotationally asymmetric template whose bounding box is square.
    - Each uniquely colored 2-to-4-pixel indicator is 4-connected and touches the
      template orthogonally; its bbox center has a unique eight-way direction from
      the template bbox center.

    Rule:
    - Erase every indicator.  For each one, translate copies by one template-bbox
      side length along its direction until a copy is wholly outside the grid.
    - Recolor that ray with the indicator color.  Copy k is rotated clockwise by
      k times (indicator area modulo four) quarter-turns, with partial edge copies
      clipped to the canvas.  Preserve the original black template.

    Key insight:
    - One visible indicator simultaneously encodes direction by position, stamp
      color by its color, and the cumulative per-copy rotation step by its area.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional grid"
    bg, template_bbox, template_mask, indicators = _parse(g)

    out = g.copy()
    for indicator in indicators:
        for r, c in indicator["coords"]:  # type: ignore[union-attr]
            out[r, c] = bg
    for indicator in indicators:
        _stamp_ray(
            out,
            template_mask,
            template_bbox,
            indicator["direction"],  # type: ignore[arg-type]
            int(indicator["color"]),
            int(indicator["area"]),
        )
    return out


# ---------------------------------------------------------------------------
# Generator


def _random_template(side: int) -> np.ndarray | None:
    """Build a connected, full-bbox, visibly rotation-asymmetric random mask."""
    mask = np.zeros((side, side), dtype=bool)
    center = (side // 2, side // 2)
    if side % 2 == 0:
        centers = (side // 2 - 1, side // 2)
        targets = (
            *((0, c) for c in centers),
            *((side - 1, c) for c in centers),
            *((r, 0) for r in centers),
            *((r, side - 1) for r in centers),
        )
    else:
        mid = side // 2
        targets = ((0, mid), (side - 1, mid), (mid, 0), (mid, side - 1))

    mask[center] = True
    for target in targets:
        r, c = target
        mask[r, c] = True
        while (r, c) != center:
            moves: list[Coord] = []
            if r != center[0]:
                moves.append((_sign(center[0] - r), 0))
            if c != center[1]:
                moves.append((0, _sign(center[1] - c)))
            dr, dc = moves[int(np.random.randint(len(moves)))]
            r, c = r + dr, c + dc
            mask[r, c] = True

    for _ in range(int(np.random.randint(1, side + 1))):
        frontier: list[Coord] = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < side and 0 <= cc < side and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if frontier:
            mask[frontier[int(np.random.randint(len(frontier)))]] = True

    if int(mask.sum()) >= side * side - 2:
        return None
    if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3)):
        return None
    return mask


def _indicator_candidates(
    template_mask: np.ndarray,
    template_bbox: BBox,
    shape: tuple[int, int],
    area: int,
) -> dict[Coord, list[set[Coord]]]:
    """Enumerate small connected outside sprites that touch the template."""
    height, width = shape
    r0, c0, r1, c1 = template_bbox
    template_coords = {
        (r0 + int(r), c0 + int(c)) for r, c in np.argwhere(template_mask)
    }
    allowed = {
        (r, c)
        for r in range(max(0, r0 - area), min(height, r1 + area + 1))
        for c in range(max(0, c0 - area), min(width, c1 + area + 1))
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    }
    seeds = {
        (r + dr, c + dc)
        for r, c in template_coords
        for dr, dc in DIR4
        if (r + dr, c + dc) in allowed
    }

    shapes = {frozenset((seed,)) for seed in seeds}
    for _ in range(1, area):
        expanded: set[frozenset[Coord]] = set()
        for cells in shapes:
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt in allowed and nxt not in cells:
                        expanded.add(cells | {nxt})
        shapes = expanded

    by_direction: dict[Coord, list[set[Coord]]] = {}
    for frozen in shapes:
        cells = set(frozen)
        if not any(
            (r + dr, c + dc) in template_coords
            for r, c in cells
            for dr, dc in DIR4
        ):
            continue
        direction = _direction(template_bbox, _bbox(cells))
        if direction != (0, 0):
            by_direction.setdefault(direction, []).append(cells)
    return by_direction


def _full_copy_count(template_bbox: BBox, direction: Coord, shape: tuple[int, int]) -> int:
    height, width = shape
    r0, c0, r1, c1 = template_bbox
    side = r1 - r0 + 1
    dr, dc = direction
    count = 0
    for k in range(1, max(shape) + 1):
        rr, cc = r0 + k * side * dr, c0 + k * side * dc
        if rr >= height or cc >= width or rr + side <= 0 or cc + side <= 0:
            break
        if 0 <= rr and rr + side <= height and 0 <= cc and cc + side <= width:
            count += 1
    return count


def _compatible(cells: set[Coord], chosen: list[dict[str, object]]) -> bool:
    occupied = set().union(*(item["coords"] for item in chosen)) if chosen else set()
    return all(
        max(abs(r - rr), abs(c - cc)) > 1
        for r, c in cells
        for rr, cc in occupied
    )


def _choose_indicators(
    candidate_maps: dict[int, dict[Coord, list[set[Coord]]]],
    areas: list[int],
    template_bbox: BBox,
    shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    """Backtrack so each sample includes both cardinal and diagonal directions."""
    categories = ["cardinal", "diagonal"] + ["any"] * (len(areas) - 2)
    np.random.shuffle(categories)
    area_order = areas.copy()
    np.random.shuffle(area_order)

    def search(index: int, chosen: list[dict[str, object]]) -> list[dict[str, object]] | None:
        if index == len(area_order):
            return chosen
        area = area_order[index]
        category = categories[index]
        mapping = candidate_maps[area]
        directions = list(mapping)
        np.random.shuffle(directions)
        used = {item["direction"] for item in chosen}
        for direction in directions:
            if direction in used or _full_copy_count(template_bbox, direction, shape) < 2:
                continue
            is_cardinal = direction in CARDINAL
            if category == "cardinal" and not is_cardinal:
                continue
            if category == "diagonal" and direction not in DIAGONAL:
                continue
            candidates = mapping[direction].copy()
            np.random.shuffle(candidates)
            for cells in candidates[:80]:
                if not _compatible(cells, chosen):
                    continue
                found = search(
                    index + 1,
                    chosen + [{"area": area, "direction": direction, "coords": cells}],
                )
                if found is not None:
                    return found
        return None

    return search(0, [])


def _oracle(
    inp: np.ndarray,
    bg: int,
    template_bbox: BBox,
    template_mask: np.ndarray,
    indicators: list[dict[str, object]],
) -> tuple[np.ndarray, set[Coord]]:
    """Construct the expected grid from generation metadata, without parsing the input."""
    out = inp.copy()
    for item in indicators:
        for r, c in item["coords"]:  # type: ignore[union-attr]
            out[r, c] = bg
    painted: set[Coord] = set()
    for item in indicators:
        painted |= _stamp_ray(
            out,
            template_mask,
            template_bbox,
            item["direction"],  # type: ignore[arg-type]
            int(item["color"]),
            int(item["area"]),
        )
    return out, painted


def _generate_attempt() -> np.ndarray | None:
    height = int(np.random.randint(28, 31))
    width = int(np.random.randint(28, 31))
    side = int(np.random.choice((4, 5)))
    template_mask = _random_template(side)
    if template_mask is None:
        return None

    r0 = (height - side) // 2 + int(np.random.randint(-1, 2))
    c0 = (width - side) // 2 + int(np.random.randint(-1, 2))
    template_bbox = (r0, c0, r0 + side - 1, c0 + side - 1)

    indicator_count = int(np.random.choice((3, 4)))
    areas = [2, 3, 4]
    if indicator_count == 4:
        areas.append(int(np.random.randint(2, 5)))
    candidate_maps = {
        area: _indicator_candidates(template_mask, template_bbox, (height, width), area)
        for area in set(areas)
    }
    indicators = _choose_indicators(candidate_maps, areas, template_bbox, (height, width))
    if indicators is None:
        return None

    palette = list(range(1, 10))
    bg = int(palette.pop(int(np.random.randint(len(palette)))))
    chosen_colors = list(np.random.choice(palette, size=indicator_count, replace=False).astype(int))
    inp = np.full((height, width), bg, dtype=int)
    inp[r0 : r0 + side, c0 : c0 + side][template_mask] = 0
    for item, color in zip(indicators, chosen_colors):
        item["color"] = int(color)
        for r, c in item["coords"]:  # type: ignore[union-attr]
            inp[r, c] = int(color)

    expected, painted = _oracle(inp, bg, template_bbox, template_mask, indicators)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected) or np.array_equal(inp, expected):
        return None

    # The erase step must remain visibly necessary for every indicator, rather
    # than all indicator cells being coincidentally repainted by its stamp ray.
    for item in indicators:
        color = int(item["color"])
        coords = item["coords"]
        if not any(int(expected[r, c]) != color for r, c in coords):  # type: ignore[union-attr]
            return None
        if not any(int(expected[r, c]) == color for r, c in painted):
            return None

    # Areas 2 and 3 must visibly require cumulative, not column-uniform, rotation.
    for area in (2, 3):
        first = np.rot90(template_mask, -(area % 4))
        second = np.rot90(template_mask, -((2 * area) % 4))
        if np.array_equal(first, second):
            return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate input satisfying all puzzle invariants."""
    for _ in range(500):
        generated = _generate_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid rotation-stamp input")
