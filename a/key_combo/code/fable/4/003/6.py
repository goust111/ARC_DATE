from __future__ import annotations

from collections import deque
import random

import numpy as np


ARC_COLORS = tuple(range(10))
BG = 0
BARRIER = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 700


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = map(int, mask.shape)
    labels = np.full((height, width), -1, dtype=int)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or labels[r, c] != -1:
                continue

            label = len(comps)
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return labels, comps


def _local_holes(patch: np.ndarray, bg: int) -> list[np.ndarray] | None:
    height, width = map(int, patch.shape)
    _labels, comps = _components(patch == int(bg))
    holes: list[np.ndarray] = []

    for coords in comps:
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == width - 1)
        )
        if touches_edge:
            return None
        holes.append(coords)
    return holes


def _block_from_component(
    g: np.ndarray,
    coords: np.ndarray,
    color: int,
    bg: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    allowed = (patch == int(bg)) | (patch == int(color))
    if not np.all(allowed):
        return None

    if (
        not np.all(patch[0, :] == int(color))
        or not np.all(patch[-1, :] == int(color))
        or not np.all(patch[:, 0] == int(color))
        or not np.all(patch[:, -1] == int(color))
    ):
        return None

    if int(np.count_nonzero(patch == int(color))) != int(coords.shape[0]):
        return None

    _color_labels, color_comps = _components(patch == int(color))
    if len(color_comps) != 1:
        return None

    holes = _local_holes(patch, bg)
    if holes is None or len(holes) == 0:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "holes": int(len(holes)),
        "coords": coords.copy(),
    }


def _find_holed_blocks(g: np.ndarray, bg: int) -> list[dict]:
    blocks: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        _labels, comps = _components(g == int(color))
        for coords in comps:
            block = _block_from_component(g, coords, color, bg)
            if block is not None:
                blocks.append(block)

    blocks.sort(key=lambda block: tuple(block["bbox"]))
    return blocks


def _erase_blocks(g: np.ndarray, bg: int, blocks: list[dict]) -> np.ndarray:
    erased = g.copy()
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        erased[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    return erased


def _reading_order_regions(g: np.ndarray, bg: int) -> list[np.ndarray]:
    _labels, regions = _components(g == int(bg))
    regions.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each holed block's hole count as the reading-order address of the region to fill.

    Input structure:
    - One barrier color splits the background into several 4-connected regions.
    - Several monochrome rectangular blocks sit on the background; each block has one or more enclosed background holes.
    - The blocks' hole counts are distinct and no count exceeds the number of barrier-separated background regions.

    Rule:
    - First erase every holed block, including its enclosed holes, to the background color.
    - Number the resulting background regions by the reading order of each region's top-left pixel.
    - For a block with k holes, fill all background pixels of region k with that block's color; untouched regions stay background and the barrier stays fixed.

    Key insight:
    - Hole count is an address into the erased wall regions, not a fixed color code and not the block's own location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"

    bg = _background_color(g)
    blocks = _find_holed_blocks(g, bg)
    assert blocks, "expected at least one holed block"

    addresses = [int(block["holes"]) for block in blocks]
    assert len(set(addresses)) == len(addresses), "hole-count addresses must be distinct"

    erased = _erase_blocks(g, bg, blocks)
    barrier_colors = {int(c) for c in np.unique(erased) if int(c) != int(bg)}
    assert barrier_colors, "expected a remaining foreground barrier"

    regions = _reading_order_regions(erased, bg)
    assert len(regions) >= 2, "barrier must split the background"
    assert max(addresses) <= len(regions), "hole count cannot exceed region count"

    out = erased.copy()
    for block in blocks:
        region = regions[int(block["holes"]) - 1]
        out[region[:, 0], region[:, 1]] = int(block["color"])
    return out


# --------------- generator ---------------


def _bands(sizes: list[int]) -> tuple[int, list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        bands.append((pos, pos + int(size) - 1))
        pos += int(size)
        if idx != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return pos, bands, separators


def _sample_room_sizes(count: int) -> list[int]:
    if count == 3:
        return [random.choice((8, 9)) for _ in range(count)]
    return [random.randint(8, 11) for _ in range(count)]


def _paint_stub(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell: tuple[int, int],
) -> bool:
    cell_r, cell_c = cell
    r0, r1 = row_bands[cell_r]
    c0, c1 = col_bands[cell_c]
    sides: list[str] = []
    if cell_r > 0:
        sides.append("top")
    if cell_r + 1 < len(row_bands):
        sides.append("bottom")
    if cell_c > 0:
        sides.append("left")
    if cell_c + 1 < len(col_bands):
        sides.append("right")
    if not sides:
        return False

    side = random.choice(sides)
    length = random.randint(1, 2)
    if side in ("top", "bottom"):
        if c1 - c0 < 4:
            return False
        c = random.randint(c0 + 2, c1 - 2)
        if side == "top":
            sep = r0 - 1
            for dr in range(1, length + 1):
                grid[sep + dr, c] = BARRIER
        else:
            sep = r1 + 1
            for dr in range(1, length + 1):
                grid[sep - dr, c] = BARRIER
    else:
        if r1 - r0 < 4:
            return False
        r = random.randint(r0 + 2, r1 - 2)
        if side == "left":
            sep = c0 - 1
            for dc in range(1, length + 1):
                grid[r, sep + dc] = BARRIER
        else:
            sep = c1 + 1
            for dc in range(1, length + 1):
                grid[r, sep - dc] = BARRIER
    return True


def _make_barrier_layout() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    room_rows = random.choice((2, 2, 3))
    room_cols = random.choice((2, 3, 3))
    row_sizes = _sample_room_sizes(room_rows)
    col_sizes = _sample_room_sizes(room_cols)
    height, row_bands, row_seps = _bands(row_sizes)
    width, col_bands, col_seps = _bands(col_sizes)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r in row_seps:
        grid[r, :] = BARRIER
    for c in col_seps:
        grid[:, c] = BARRIER

    cells = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    random.shuffle(cells)
    stub_budget = random.randint(2, min(6, len(cells) + 2))
    painted = 0
    for cell in cells:
        if painted >= stub_budget:
            break
        if random.random() < 0.75 and _paint_stub(grid, row_bands, col_bands, cell):
            painted += 1
    if painted == 0 and not _paint_stub(grid, row_bands, col_bands, random.choice(cells)):
        return None

    labels, regions = _components(grid == BG)
    if len(regions) != room_rows * room_cols:
        return None
    regions.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return grid, labels, regions


def _hole_capacity(height: int, width: int) -> int:
    interior_h = max(0, int(height) - 2)
    interior_w = max(0, int(width) - 2)
    return (interior_h * interior_w + 1) // 2


def _region_capacity(labels: np.ndarray, region_id: int) -> int:
    coords = np.argwhere(labels == int(region_id))
    r0, c0, r1, c1 = _bbox(coords)
    max_h = max(0, r1 - r0 - 1)
    max_w = max(0, c1 - c0 - 1)
    best = 0
    for h in range(4, max_h + 1):
        for w in range(4, max_w + 1):
            if abs(h - w) <= 2:
                best = max(best, _hole_capacity(h, w))
    return int(best)


def _sample_addresses(home_regions: list[int], capacities: list[int], region_count: int) -> list[int] | None:
    order = list(range(len(home_regions)))
    random.shuffle(order)
    chosen = [0] * len(home_regions)

    def search(pos: int, used: set[int]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        home = int(home_regions[idx])
        options = [
            address
            for address in range(1, region_count + 1)
            if address not in used
            and address != home + 1
            and address <= int(capacities[home])
        ]
        random.shuffle(options)
        for address in options:
            used.add(address)
            chosen[idx] = int(address)
            if search(pos + 1, used):
                return True
            used.remove(address)
        return False

    if search(0, set()):
        return chosen
    return None


def _candidate_positions(
    grid: np.ndarray,
    labels: np.ndarray,
    region_id: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    coords = np.argwhere(labels == int(region_id))
    r0, c0, r1, c1 = _bbox(coords)
    positions: list[tuple[int, int]] = []

    for r in range(r0 + 1, r1 - height + 2):
        for c in range(c0 + 1, c1 - width + 2):
            rr1 = r + int(height) - 1
            cc1 = c + int(width) - 1
            if rr1 + 1 >= grid.shape[0] or cc1 + 1 >= grid.shape[1]:
                continue
            label_patch = labels[r - 1 : rr1 + 2, c - 1 : cc1 + 2]
            grid_patch = grid[r - 1 : rr1 + 2, c - 1 : cc1 + 2]
            if np.all(label_patch == int(region_id)) and np.all(grid_patch == BG):
                positions.append((int(r), int(c)))
    return positions


def _sample_block_geometry(
    grid: np.ndarray,
    labels: np.ndarray,
    region_id: int,
    hole_count: int,
) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(labels == int(region_id))
    r0, c0, r1, c1 = _bbox(coords)
    max_h = max(0, r1 - r0 - 1)
    max_w = max(0, c1 - c0 - 1)
    dims = [
        (h, w)
        for h in range(4, max_h + 1)
        for w in range(4, max_w + 1)
        if abs(h - w) <= 2 and _hole_capacity(h, w) >= int(hole_count)
    ]
    random.shuffle(dims)

    for height, width in dims:
        positions = _candidate_positions(grid, labels, region_id, height, width)
        if positions:
            r, c = random.choice(positions)
            return int(r), int(c), int(height), int(width)
    return None


def _sample_holes(height: int, width: int, count: int) -> list[tuple[int, int]]:
    parity_options = []
    for parity in (0, 1):
        cells = [
            (r, c)
            for r in range(1, int(height) - 1)
            for c in range(1, int(width) - 1)
            if (r + c) % 2 == parity
        ]
        if len(cells) >= int(count):
            parity_options.append(cells)
    cells = random.choice(parity_options)
    return random.sample(cells, int(count))


def _paint_block(
    grid: np.ndarray,
    r: int,
    c: int,
    height: int,
    width: int,
    color: int,
    hole_count: int,
) -> None:
    grid[r : r + int(height), c : c + int(width)] = int(color)
    for dr, dc in _sample_holes(height, width, hole_count):
        grid[r + dr, c + dc] = BG


def _expected_from_scene(base: np.ndarray, regions: list[np.ndarray], assignments: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for hole_count, color in assignments:
        region = regions[int(hole_count) - 1]
        out[region[:, 0], region[:, 1]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    made = _make_barrier_layout()
    if made is None:
        return None
    base, labels, regions = made
    region_count = len(regions)
    capacities = [_region_capacity(labels, region_id) for region_id in range(region_count)]

    max_blocks = min(4, region_count - 1, len(BLOCK_COLORS))
    if max_blocks < 2:
        return None
    block_count = random.randint(2, max_blocks)

    capable_regions = [idx for idx, cap in enumerate(capacities) if cap >= 1]
    if len(capable_regions) < block_count:
        return None
    home_regions = random.sample(capable_regions, block_count)
    addresses = _sample_addresses(home_regions, capacities, region_count)
    if addresses is None:
        return None

    grid = base.copy()
    colors = random.sample(BLOCK_COLORS, block_count)
    assignments: list[tuple[int, int]] = []
    homes_and_addresses: list[tuple[int, int]] = []

    for home, address, color in zip(home_regions, addresses, colors):
        geom = _sample_block_geometry(grid, labels, home, address)
        if geom is None:
            return None
        r, c, height, width = geom
        _paint_block(grid, r, c, height, width, color, address)
        assignments.append((int(address), int(color)))
        homes_and_addresses.append((int(home), int(address)))

    expected = _expected_from_scene(base, regions, assignments)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    detected = _find_holed_blocks(grid, BG)
    found_pairs = sorted((int(block["holes"]), int(block["color"])) for block in detected)
    if found_pairs != sorted(assignments):
        return None
    if any(home + 1 == address for home, address in homes_and_addresses):
        return None
    if len({address for _home, address in homes_and_addresses}) != block_count:
        return None
    if block_count >= region_count:
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid hole-count address input")
