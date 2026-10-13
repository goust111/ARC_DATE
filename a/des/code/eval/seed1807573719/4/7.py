from collections import deque

import numpy as np


BG_COLOR = 0
BRIDGE_COLOR = 5
ISLAND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


_RNG = np.random.default_rng()

# Generator controls.
GEN_ATTEMPTS = 400
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
NUM_GADGETS_CHOICES = (3, 4)
SLOT_MARGIN = 1
MIN_SHARED_AXIS = 4
MAX_SHARED_AXIS = 6
MIN_BOX_SIDE = 3
MAX_BOX_SIDE = 5
MIN_BRIDGE_LEN = 2
MAX_BRIDGE_LEN = 5
MIN_ISLAND_AREA = 4
SHAPE_ATTEMPTS = 100


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    current = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue

            current += 1
            labels[r, c] = current
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        queue.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return labels, components


def _bridge_order_and_contacts(
    coords: np.ndarray, island_labels: np.ndarray
) -> tuple[list[tuple[int, int]], int, int]:
    h, w = island_labels.shape
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if rows.size == 1:
        row = int(rows[0])
        c0 = int(np.min(coords[:, 1]))
        c1 = int(np.max(coords[:, 1]))
        assert len(coords) == c1 - c0 + 1, "horizontal bridge must be contiguous"
        ordered = [(row, c) for c in range(c0, c1 + 1)]
        before = (row, c0 - 1)
        after = (row, c1 + 1)
    elif cols.size == 1:
        col = int(cols[0])
        r0 = int(np.min(coords[:, 0]))
        r1 = int(np.max(coords[:, 0]))
        assert len(coords) == r1 - r0 + 1, "vertical bridge must be contiguous"
        ordered = [(r, col) for r in range(r0, r1 + 1)]
        before = (r0 - 1, col)
        after = (r1 + 1, col)
    else:
        raise AssertionError("bridge must be a one-pixel straight segment")

    br, bc = before
    ar, ac = after
    assert 0 <= br < h and 0 <= bc < w, "bridge must start next to an island"
    assert 0 <= ar < h and 0 <= ac < w, "bridge must end next to an island"

    first_id = int(island_labels[before])
    second_id = int(island_labels[after])
    assert first_id > 0 and second_id > 0, "bridge endpoints must touch islands"
    assert first_id != second_id, "bridge must connect two different islands"

    for idx, (r, c) in enumerate(ordered):
        allowed: set[tuple[int, int]] = set()
        if idx == 0:
            allowed.add(before)
        if idx == len(ordered) - 1:
            allowed.add(after)
        for dr, dc in ORTHO_DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and island_labels[nr, nc] > 0:
                assert (nr, nc) in allowed, "only bridge endpoints may touch islands"

    return ordered, first_id, second_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each grey bridge from the smaller island side toward the larger island.

    Rule:
    - Find every monochrome island (4-connected, non-background, non-grey).
    - For each straight grey bridge, compare the areas of the two endpoint islands.
    - If the areas differ, recolor that many bridge cells from the smaller side
      using the smaller island's color, clipped by the bridge length.
    - Equal-area pairs leave the bridge unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    island_mask = (g != bg) & (g != BRIDGE_COLOR)
    island_labels, islands = _label_components(island_mask)
    assert islands, "expected at least one island"

    island_areas = [0]
    island_colors = [bg]
    for coords in islands:
        vals = np.unique(g[coords[:, 0], coords[:, 1]])
        assert len(vals) == 1, "each island must be monochrome"
        island_areas.append(int(len(coords)))
        island_colors.append(int(vals[0]))

    _, bridges = _label_components(g == BRIDGE_COLOR)
    assert bridges, "expected at least one bridge"

    out = g.copy()
    for coords in bridges:
        ordered, first_id, second_id = _bridge_order_and_contacts(coords, island_labels)
        first_area = island_areas[first_id]
        second_area = island_areas[second_id]
        if first_area == second_area:
            continue

        if first_area < second_area:
            paint_order = ordered
            paint_color = island_colors[first_id]
            paint_count = second_area - first_area
        else:
            paint_order = list(reversed(ordered))
            paint_color = island_colors[second_id]
            paint_count = first_area - second_area

        for r, c in paint_order[: min(paint_count, len(ordered))]:
            out[r, c] = paint_color

    return out


def _is_full_bbox(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return len(cells) == bbox_area


def _grow_shape(
    height: int,
    width: int,
    area: int,
    connector: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {connector}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ORTHO_DIRS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            pick = tuple(frontier)[int(rng.integers(len(frontier)))]
            cells.add(pick)

        if len(cells) != area:
            continue
        if area >= 5 and _is_full_bbox(cells):
            continue
        return np.array(sorted(cells), dtype=int)

    return None


def _pick_area_pair(
    cap1: int,
    cap2: int,
    bridge_len: int,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if scenario == "equal":
        upper = min(cap1, cap2)
        if upper < MIN_ISLAND_AREA:
            return None
        area = _randint(rng, MIN_ISLAND_AREA, upper)
        return area, area

    first_smaller = scenario.startswith("first")
    need_partial = scenario.endswith("partial")
    small_cap = cap1 if first_smaller else cap2
    large_cap = cap2 if first_smaller else cap1

    feasible_diffs = []
    for diff in range(1, large_cap - MIN_ISLAND_AREA + 1):
        if need_partial and diff >= bridge_len:
            continue
        if not need_partial and diff < bridge_len:
            continue
        small_max = min(small_cap, large_cap - diff)
        if small_max >= MIN_ISLAND_AREA:
            feasible_diffs.append(diff)

    if not feasible_diffs:
        return None

    diff = int(rng.choice(feasible_diffs))
    small_max = min(small_cap, large_cap - diff)
    small_area = _randint(rng, MIN_ISLAND_AREA, small_max)
    large_area = small_area + diff

    if first_smaller:
        return small_area, large_area
    return large_area, small_area


def _sample_colors(rng: np.random.Generator) -> tuple[int, int]:
    color1 = int(rng.choice(ISLAND_COLORS))
    if rng.random() < 0.25:
        return color1, color1
    choices = [c for c in ISLAND_COLORS if c != color1]
    return color1, int(rng.choice(choices))


def _sample_scenarios(count: int, rng: np.random.Generator) -> list[str]:
    scenarios = [
        "equal",
        str(rng.choice(["first_partial", "first_full"])),
        str(rng.choice(["second_partial", "second_full"])),
    ]
    while len(scenarios) < count:
        scenarios.append(str(rng.choice(
            ["equal", "first_partial", "first_full", "second_partial", "second_full"]
        )))
    rng.shuffle(scenarios)
    return scenarios


def _sample_orientations(count: int, rng: np.random.Generator) -> list[str]:
    orientations = ["h", "v"]
    while len(orientations) < count:
        orientations.append(str(rng.choice(["h", "v"])))
    rng.shuffle(orientations)
    return orientations


def _make_slots(h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    split_r = _randint(rng, 10, h - 11)
    split_c = _randint(rng, 10, w - 11)
    return [
        (1, split_r - 1, 1, split_c - 1),
        (1, split_r - 1, split_c + 1, w - 2),
        (split_r + 1, h - 2, 1, split_c - 1),
        (split_r + 1, h - 2, split_c + 1, w - 2),
    ]


def _build_horizontal_gadget(
    slot: tuple[int, int, int, int], scenario: str, rng: np.random.Generator
) -> dict | None:
    sr0, sr1, sc0, sc1 = slot
    slot_h = sr1 - sr0 + 1
    slot_w = sc1 - sc0 + 1
    usable_h = slot_h - 2 * SLOT_MARGIN
    usable_w = slot_w - 2 * SLOT_MARGIN

    if usable_h < MIN_SHARED_AXIS or usable_w < 2 * MIN_BOX_SIDE + MIN_BRIDGE_LEN:
        return None

    for _ in range(120):
        box_h = _randint(rng, MIN_SHARED_AXIS, min(MAX_SHARED_AXIS, usable_h))
        w1 = _randint(rng, MIN_BOX_SIDE, min(MAX_BOX_SIDE, usable_w - MIN_BOX_SIDE - MIN_BRIDGE_LEN))
        w2_max = min(MAX_BOX_SIDE, usable_w - w1 - MIN_BRIDGE_LEN)
        if w2_max < MIN_BOX_SIDE:
            continue
        w2 = _randint(rng, MIN_BOX_SIDE, w2_max)
        bridge_max = min(MAX_BRIDGE_LEN, usable_w - w1 - w2)
        if bridge_max < MIN_BRIDGE_LEN:
            continue
        bridge_len = _randint(rng, MIN_BRIDGE_LEN, bridge_max)

        areas = _pick_area_pair(box_h * w1 - 1, box_h * w2 - 1, bridge_len, scenario, rng)
        if areas is None:
            continue

        connector_row = _randint(rng, 0, box_h - 1)
        shape1 = _grow_shape(box_h, w1, areas[0], (connector_row, w1 - 1), rng)
        shape2 = _grow_shape(box_h, w2, areas[1], (connector_row, 0), rng)
        if shape1 is None or shape2 is None:
            continue

        total_w = w1 + bridge_len + w2
        top_min = sr0 + SLOT_MARGIN
        top_max = sr1 - SLOT_MARGIN - box_h + 1
        left_min = sc0 + SLOT_MARGIN
        left_max = sc1 - SLOT_MARGIN - total_w + 1
        if top_max < top_min or left_max < left_min:
            continue

        top = _randint(rng, top_min, top_max)
        left = _randint(rng, left_min, left_max)
        color1, color2 = _sample_colors(rng)

        island1 = [(top + int(r), left + int(c)) for r, c in shape1]
        island2 = [(top + int(r), left + w1 + bridge_len + int(c)) for r, c in shape2]
        bridge = [(top + connector_row, left + w1 + i) for i in range(bridge_len)]
        return {
            "islands": [(island1, color1), (island2, color2)],
            "bridge": bridge,
            "areas": (areas[0], areas[1]),
            "colors": (color1, color2),
        }

    return None


def _build_vertical_gadget(
    slot: tuple[int, int, int, int], scenario: str, rng: np.random.Generator
) -> dict | None:
    sr0, sr1, sc0, sc1 = slot
    slot_h = sr1 - sr0 + 1
    slot_w = sc1 - sc0 + 1
    usable_h = slot_h - 2 * SLOT_MARGIN
    usable_w = slot_w - 2 * SLOT_MARGIN

    if usable_w < MIN_SHARED_AXIS or usable_h < 2 * MIN_BOX_SIDE + MIN_BRIDGE_LEN:
        return None

    for _ in range(120):
        box_w = _randint(rng, MIN_SHARED_AXIS, min(MAX_SHARED_AXIS, usable_w))
        h1 = _randint(rng, MIN_BOX_SIDE, min(MAX_BOX_SIDE, usable_h - MIN_BOX_SIDE - MIN_BRIDGE_LEN))
        h2_max = min(MAX_BOX_SIDE, usable_h - h1 - MIN_BRIDGE_LEN)
        if h2_max < MIN_BOX_SIDE:
            continue
        h2 = _randint(rng, MIN_BOX_SIDE, h2_max)
        bridge_max = min(MAX_BRIDGE_LEN, usable_h - h1 - h2)
        if bridge_max < MIN_BRIDGE_LEN:
            continue
        bridge_len = _randint(rng, MIN_BRIDGE_LEN, bridge_max)

        areas = _pick_area_pair(h1 * box_w - 1, h2 * box_w - 1, bridge_len, scenario, rng)
        if areas is None:
            continue

        connector_col = _randint(rng, 0, box_w - 1)
        shape1 = _grow_shape(h1, box_w, areas[0], (h1 - 1, connector_col), rng)
        shape2 = _grow_shape(h2, box_w, areas[1], (0, connector_col), rng)
        if shape1 is None or shape2 is None:
            continue

        total_h = h1 + bridge_len + h2
        top_min = sr0 + SLOT_MARGIN
        top_max = sr1 - SLOT_MARGIN - total_h + 1
        left_min = sc0 + SLOT_MARGIN
        left_max = sc1 - SLOT_MARGIN - box_w + 1
        if top_max < top_min or left_max < left_min:
            continue

        top = _randint(rng, top_min, top_max)
        left = _randint(rng, left_min, left_max)
        color1, color2 = _sample_colors(rng)

        island1 = [(top + int(r), left + int(c)) for r, c in shape1]
        island2 = [(top + h1 + bridge_len + int(r), left + int(c)) for r, c in shape2]
        bridge = [(top + h1 + i, left + connector_col) for i in range(bridge_len)]
        return {
            "islands": [(island1, color1), (island2, color2)],
            "bridge": bridge,
            "areas": (areas[0], areas[1]),
            "colors": (color1, color2),
        }

    return None


def _draw_gadget(grid: np.ndarray, gadget: dict) -> None:
    for cells, color in gadget["islands"]:
        for r, c in cells:
            grid[r, c] = color
    for r, c in gadget["bridge"]:
        grid[r, c] = BRIDGE_COLOR


def _expected_output(grid: np.ndarray, gadgets: list[dict]) -> np.ndarray:
    out = grid.copy()
    for gadget in gadgets:
        area1, area2 = gadget["areas"]
        color1, color2 = gadget["colors"]
        bridge = gadget["bridge"]
        if area1 == area2:
            continue
        if area1 < area2:
            paint_cells = bridge
            paint_color = color1
            diff = area2 - area1
        else:
            paint_cells = list(reversed(bridge))
            paint_color = color2
            diff = area1 - area2
        for r, c in paint_cells[: min(diff, len(bridge))]:
            out[r, c] = paint_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = _randint(_RNG, *GRID_H_RANGE)
    w = _randint(_RNG, *GRID_W_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    slots = _make_slots(h, w, _RNG)

    num_gadgets = int(_RNG.choice(NUM_GADGETS_CHOICES))
    slot_indices = list(_RNG.choice(len(slots), size=num_gadgets, replace=False))
    scenarios = _sample_scenarios(num_gadgets, _RNG)
    orientations = _sample_orientations(num_gadgets, _RNG)

    gadgets: list[dict] = []
    for slot_idx, scenario, orientation in zip(slot_indices, scenarios, orientations):
        slot = slots[int(slot_idx)]
        if orientation == "h":
            gadget = _build_horizontal_gadget(slot, scenario, _RNG)
        else:
            gadget = _build_vertical_gadget(slot, scenario, _RNG)
        if gadget is None:
            return None
        _draw_gadget(grid, gadget)
        gadgets.append(gadget)

    if _most_common_color(grid) != BG_COLOR:
        return None

    expected = _expected_output(grid, gadgets)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
