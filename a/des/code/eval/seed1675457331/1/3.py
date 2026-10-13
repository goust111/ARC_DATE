import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PALETTE = [1, 2, 3, 4, 6, 8, 9]
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append({"coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _route_component_is_valid(grid: np.ndarray, color: int) -> bool:
    comps = _components_of_color(grid, color)
    if len(comps) != 1:
        return False
    r0, c0, r1, c1 = comps[0]["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    h, w = patch.shape
    if h == 2 and w >= 5:
        if not np.all(patch[:, :2]) or not np.all(patch[:, -2:]):
            return False
        middle = patch[:, 2:-2]
        row_sums = middle.sum(axis=1)
        return sorted(map(int, row_sums)) == [0, middle.shape[1]]
    if w == 2 and h >= 5:
        if not np.all(patch[:2, :]) or not np.all(patch[-2:, :]):
            return False
        middle = patch[2:-2, :]
        col_sums = middle.sum(axis=0)
        return sorted(map(int, col_sums)) == [0, middle.shape[0]]
    return False


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected a unique grey separator row"
    row = int(rows[0])
    assert 0 < row < grid.shape[0] - 1
    return row


def _find_frames(grid: np.ndarray, top_offset: int) -> list[dict]:
    orange_comps = _components_of_color(grid[top_offset:, :], ORANGE)
    frames: list[dict] = []
    for comp in orange_comps:
        r0, c0, r1, c1 = comp["bbox"]
        size = int(r1 - r0 + 1)
        assert size == int(c1 - c0 + 1), "orange components must be square frames"
        assert 5 <= size <= 7, "unexpected frame size"
        patch = grid[top_offset + r0 : top_offset + r1 + 1, c0 : c1 + 1]
        border = np.zeros((size, size), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == ORANGE), "frame border must be orange"
        assert np.all(patch[~border] != ORANGE), "frame interior cannot contain orange"
        frames.append(
            {
                "bbox": (top_offset + r0, c0, top_offset + r1, c1),
                "size": size,
            }
        )
    assert len(frames) >= 2, "expected multiple workspace frames"
    sizes = {frame["size"] for frame in frames}
    assert len(sizes) == 1, "all frames must have equal size"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _region_records(grid: np.ndarray, frames: list[dict]) -> tuple[list[dict], dict[int, dict]]:
    sources: list[dict] = []
    targets: dict[int, dict] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colored = np.argwhere(inner != BLACK)
        assert len(colored) >= 1, "each frame must contain a pattern or an anchor"
        record = {
            "bbox": frame["bbox"],
            "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            "pattern": inner.copy(),
        }
        if len(colored) == 1:
            pos = tuple(map(int, colored[0]))
            color = int(inner[pos])
            assert color != ORANGE
            assert color not in targets, "target anchor colors must be unique"
            record["anchor_pos"] = pos
            record["anchor_color"] = color
            targets[color] = record
        else:
            sources.append(record)
    assert sources, "expected at least one source frame"
    assert targets, "expected at least one target frame"
    return sources, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse the route colors from the top panel, then copy each source pattern into
    the matching target frame after the unique 90-degree rotation that aligns the
    source's route-colored pixel with the target anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    top = g[:sep, :]
    route_colors = sorted(int(c) for c in np.unique(top) if c not in {BLACK, GREY})
    assert route_colors, "expected at least one route color"
    for color in route_colors:
        assert _route_component_is_valid(top, color), "invalid route component"

    frames = _find_frames(g, sep + 1)
    sources, targets = _region_records(g, frames)

    source_by_color: dict[int, tuple[dict, tuple[int, int]]] = {}
    for source in sources:
        pattern = source["pattern"]
        for color in route_colors:
            coords = np.argwhere(pattern == int(color))
            if len(coords) == 0:
                continue
            assert len(coords) == 1, "route color must appear exactly once inside a source"
            assert color not in source_by_color, "route color must identify a unique source"
            source_by_color[color] = (source, tuple(map(int, coords[0])))

    assert set(targets) == set(route_colors), "every route color should anchor one target"
    assert set(source_by_color) == set(route_colors), "every route color should appear in one source"

    out = g.copy()
    for color in route_colors:
        source, _src_pos = source_by_color[color]
        pattern = source["pattern"]
        target = targets[color]
        anchor = target["anchor_pos"]
        matches = []
        for k in range(4):
            rotated = np.rot90(pattern, k)
            coords = np.argwhere(rotated == int(color))
            if len(coords) == 1 and tuple(map(int, coords[0])) == anchor:
                matches.append(rotated)
        assert len(matches) == 1, "rotation must be uniquely determined by the anchor"
        ir0, ic0, ir1, ic1 = target["inner_bbox"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = matches[0]
    return out


def _rotate_coord(coord: tuple[int, int], size: int, k: int) -> tuple[int, int]:
    r, c = coord
    if k % 4 == 0:
        return r, c
    if k % 4 == 1:
        return size - 1 - c, r
    if k % 4 == 2:
        return size - 1 - r, size - 1 - c
    return c, size - 1 - r


def _random_connected_cells(size: int, count: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < count:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return cells


def _make_source_pattern(size: int, route_colors: list[int], filler_colors: list[int]) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    center = (size // 2, size // 2)
    min_cells = max(2, len(route_colors) + 1)
    max_cells = min(size * size, len(route_colors) + 5)
    for _ in range(200):
        cell_count = random.randint(min_cells, max_cells)
        cells = _random_connected_cells(size, cell_count)
        non_center = [cell for cell in cells if cell != center]
        if len(non_center) < len(route_colors):
            continue
        route_positions = random.sample(non_center, len(route_colors))
        pattern = np.full((size, size), BLACK, dtype=int)
        pos_map: dict[int, tuple[int, int]] = {}
        taken = set(route_positions)
        for color, pos in zip(route_colors, route_positions):
            pattern[pos] = int(color)
            pos_map[int(color)] = pos
        filler_cells = [cell for cell in cells if cell not in taken]
        if not filler_cells:
            continue
        for cell in filler_cells:
            pattern[cell] = int(random.choice(filler_colors))
        if len(set(map(int, pattern[pattern != BLACK]))) < 2:
            continue
        return pattern, pos_map
    return None


def _partition_count(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    segments = []
    start = 0
    for cut in cuts + [total]:
        segments.append(cut - start)
        start = cut
    random.shuffle(segments)
    return segments


def _layout_frames(region_count: int, frame_size: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    row_count = random.choice([1, 2])
    if row_count == 2 and region_count < 3:
        row_count = 1
    row_counts = _partition_count(region_count, row_count)

    row_widths = []
    row_gaps = []
    for count in row_counts:
        gaps = [random.randint(2, 4) for _ in range(max(0, count - 1))]
        row_widths.append(count * frame_size + sum(gaps))
        row_gaps.append(gaps)

    max_row_width = max(row_widths)
    extra_width = random.randint(2, 6)
    width = max_row_width + extra_width
    if width > MAX_GRID_SIZE:
        return None

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    vertical_gaps = [random.randint(2, 4) for _ in range(max(0, row_count - 1))]
    height = top_pad + row_count * frame_size + sum(vertical_gaps) + bottom_pad
    if height >= MAX_GRID_SIZE:
        return None

    positions: list[tuple[int, int]] = []
    row_top = top_pad
    for idx, count in enumerate(row_counts):
        row_width = row_widths[idx]
        left = random.randint(0, width - row_width)
        col = left
        for j in range(count):
            positions.append((row_top, col))
            col += frame_size
            if j < count - 1:
                col += row_gaps[idx][j]
        if idx < row_count - 1:
            row_top += frame_size + vertical_gaps[idx]
    return width, height, positions


def _make_route_patch(color: int, top_height: int, width: int) -> np.ndarray | None:
    orientations = []
    if width >= 5:
        orientations.append("H")
    if top_height >= 5:
        orientations.append("V")
    if not orientations:
        return None
    orientation = random.choice(orientations)
    if orientation == "H":
        max_len = min(4, width - 4)
        if max_len < 1:
            return None
        line_len = random.randint(1, max_len)
        patch = np.zeros((2, line_len + 4), dtype=int)
        patch[:, :2] = color
        patch[:, -2:] = color
        patch[random.randint(0, 1), 2:-2] = color
        return patch
    max_len = min(4, top_height - 4)
    if max_len < 1:
        return None
    line_len = random.randint(1, max_len)
    patch = np.zeros((line_len + 4, 2), dtype=int)
    patch[:2, :] = color
    patch[-2:, :] = color
    patch[2:-2, random.randint(0, 1)] = color
    return patch


def _place_routes(route_colors: list[int], top_height: int, width: int) -> np.ndarray | None:
    top = np.full((top_height, width), BLACK, dtype=int)
    reserved = np.zeros((top_height, width), dtype=bool)
    for color in random.sample(route_colors, len(route_colors)):
        patch = _make_route_patch(color, top_height, width)
        if patch is None:
            return None
        ph, pw = patch.shape
        candidates = []
        for r in range(top_height - ph + 1):
            for c in range(width - pw + 1):
                if np.any(reserved[r : r + ph, c : c + pw]):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        mask = patch != BLACK
        top[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(top_height, r0 + ph + 1)
        cc1 = min(width, c0 + pw + 1)
        reserved[rr0:rr1, cc0:cc1] = True
    return top


def _build_expected_output(inp: np.ndarray, source_patterns: dict[int, np.ndarray], target_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in target_specs:
        r0, c0 = spec["frame_pos"]
        inner = np.rot90(source_patterns[spec["source_id"]], spec["rotation"])
        out[r0 + 1 : r0 + 1 + inner.shape[0], c0 + 1 : c0 + 1 + inner.shape[1]] = inner
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.randint(5, 7)
    inner_size = frame_size - 2

    n_sources = random.randint(1, 3)
    n_targets = random.randint(max(2, n_sources), 5)
    if n_sources + n_targets > 6:
        return None

    route_colors = random.sample(PALETTE, n_targets)
    filler_colors = [color for color in PALETTE if color not in route_colors]
    if not filler_colors:
        return None

    source_ids = list(range(n_sources))
    owners = source_ids + [random.choice(source_ids) for _ in range(n_targets - n_sources)]
    random.shuffle(owners)

    source_patterns: dict[int, np.ndarray] = {}
    target_specs: list[dict] = []
    source_routes: dict[int, list[tuple[int, int]]] = {sid: [] for sid in source_ids}
    for color, owner in zip(route_colors, owners):
        source_routes[owner].append((int(color), random.randrange(4)))

    for source_id, items in source_routes.items():
        pattern_pack = _make_source_pattern(inner_size, [color for color, _rot in items], filler_colors)
        if pattern_pack is None:
            return None
        pattern, pos_map = pattern_pack
        source_patterns[source_id] = pattern
        for color, rotation in items:
            target_specs.append(
                {
                    "color": int(color),
                    "source_id": source_id,
                    "rotation": int(rotation),
                    "anchor_pos": _rotate_coord(pos_map[color], inner_size, rotation),
                }
            )

    region_count = n_sources + n_targets
    layout = _layout_frames(region_count, frame_size)
    if layout is None:
        return None
    width, workspace_height, positions = layout

    max_top_height = MAX_GRID_SIZE - workspace_height - 1
    if max_top_height < 6:
        return None
    top_height = random.randint(6, min(10, max_top_height))
    top = _place_routes(route_colors, top_height, width)
    if top is None:
        return None

    entries = []
    for source_id in source_ids:
        entries.append({"kind": "source", "source_id": source_id})
    for spec in target_specs:
        entries.append({"kind": "target", "spec": spec})
    random.shuffle(entries)

    inp = np.full((top_height + 1 + workspace_height, width), BLACK, dtype=int)
    inp[:top_height, :] = top
    inp[top_height, :] = GREY

    workspace_top = top_height + 1
    for entry, (wr, wc) in zip(entries, positions):
        gr = workspace_top + wr
        gc = wc
        inp[gr : gr + frame_size, gc] = ORANGE
        inp[gr : gr + frame_size, gc + frame_size - 1] = ORANGE
        inp[gr, gc : gc + frame_size] = ORANGE
        inp[gr + frame_size - 1, gc : gc + frame_size] = ORANGE
        if entry["kind"] == "source":
            pattern = source_patterns[entry["source_id"]]
            inp[gr + 1 : gr + 1 + inner_size, gc + 1 : gc + 1 + inner_size] = pattern
        else:
            spec = entry["spec"]
            ar, ac = spec["anchor_pos"]
            inp[gr + 1 + ar, gc + 1 + ac] = int(spec["color"])
            spec["frame_pos"] = (gr, gc)

    expected = _build_expected_output(inp, source_patterns, target_specs)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
