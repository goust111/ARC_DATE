import numpy as np


BG = 0
GREY = 5
AZURE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 9]

BAND_HEIGHT = 9
BAND_GAP = 1
PAIR_COUNTS = (2, 3)
INNER_SIZES = ((5, 5), (5, 6), (6, 5), (6, 6))
GRID_WIDTH_RANGE = (22, 31)
SOURCE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
MAX_GENERATION_ATTEMPTS = 300
PAIR_ATTEMPTS = 200

RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0) < 2 or (c1 - c0) < 2:
        return False
    return set(coords) == _frame_cells((r0, c0, r1, c1))


def _touches_mapping(
    cell: tuple[int, int], mapping: dict[tuple[int, int], int], shape: tuple[int, int]
) -> bool:
    h, w = shape
    r, c = cell
    for nr, nc in _neighbors4(r, c, h, w):
        if (nr, nc) in mapping:
            return True
    return False


def _ordered_path(
    coords: list[tuple[int, int]],
    source_map: dict[tuple[int, int], int],
    frame_map: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if len(coords) == 1:
        return list(coords)

    path_set = set(coords)
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        r, c = cell
        neighbor_map[cell] = [
            (nr, nc) for nr, nc in _neighbors4(r, c, shape[0], shape[1]) if (nr, nc) in path_set
        ]
    endpoints = [cell for cell, nbs in neighbor_map.items() if len(nbs) == 1]
    if len(endpoints) != 2:
        raise AssertionError("track must be a simple chain")

    source_end = next(
        (ep for ep in endpoints if _touches_mapping(ep, source_map, shape)),
        None,
    )
    frame_end = next(
        (ep for ep in endpoints if _touches_mapping(ep, frame_map, shape)),
        None,
    )
    if source_end is None and frame_end is not None:
        source_end = endpoints[0] if endpoints[1] == frame_end else endpoints[1]
    if source_end is None:
        source_end = endpoints[0]

    ordered = [source_end]
    visited = {source_end}
    prev = None
    cur = source_end
    while True:
        nxts = [cell for cell in neighbor_map[cur] if cell != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt in visited:
            raise AssertionError("track must be a simple chain")
        prev, cur = cur, nxt
        ordered.append(cur)
        visited.add(cur)
    if len(ordered) != len(path_set):
        raise AssertionError("track must be a simple chain")
    return ordered


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((int(np.sign(r1 - r0)), int(np.sign(c1 - c0))))
    turns = 0
    for prev_dir, next_dir in zip(dirs, dirs[1:]):
        if prev_dir != next_dir:
            turns += 1
    return turns


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(patch, dtype=int), -int(turns) % 4)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    source_mask = ~np.isin(g, [BG, GREY, AZURE])
    source_comps = _components(source_mask)
    sources = []
    source_map: dict[tuple[int, int], int] = {}
    for coords in source_comps:
        idx = len(sources)
        for cell in coords:
            source_map[cell] = idx
        sources.append({"coords": coords, "bbox": _bbox(coords), "patch": _crop_patch(g, coords)})

    frame_comps = [coords for coords in _components(g == GREY) if _is_rect_frame(coords)]
    frames = []
    frame_map: dict[tuple[int, int], int] = {}
    for coords in frame_comps:
        idx = len(frames)
        bbox = _bbox(coords)
        for cell in coords:
            frame_map[cell] = idx
        frames.append({"coords": coords, "bbox": bbox})

    if not sources or len(sources) != len(frames):
        raise AssertionError("expected matching non-empty source and frame counts")

    path_comps = _components(g == AZURE)
    if len(path_comps) != len(sources):
        raise AssertionError("expected one azure track per source")

    links = []
    used_sources: set[int] = set()
    used_frames: set[int] = set()
    for coords in path_comps:
        source_ids = set()
        frame_ids = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in source_map:
                    source_ids.add(source_map[(nr, nc)])
                if (nr, nc) in frame_map:
                    frame_ids.add(frame_map[(nr, nc)])
        if len(source_ids) != 1 or len(frame_ids) != 1:
            raise AssertionError("each track must touch exactly one source and one frame")
        source_idx = next(iter(source_ids))
        frame_idx = next(iter(frame_ids))
        if source_idx in used_sources or frame_idx in used_frames:
            raise AssertionError("pairing must be one-to-one")
        ordered = _ordered_path(coords, source_map, frame_map, g.shape)
        links.append(
            {
                "coords": coords,
                "ordered": ordered,
                "turns": _count_turns(ordered),
                "source": source_idx,
                "frame": frame_idx,
            }
        )
        used_sources.add(source_idx)
        used_frames.add(frame_idx)

    return {"sources": sources, "frames": frames, "links": links}


def _apply_transform(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for source in scene["sources"]:
        for r, c in source["coords"]:
            out[r, c] = BG
    for link in scene["links"]:
        for r, c in link["coords"]:
            out[r, c] = BG

    for link in scene["links"]:
        source = scene["sources"][link["source"]]
        frame = scene["frames"][link["frame"]]
        patch = _rotate_cw(source["patch"], link["turns"])
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ph, pw = patch.shape
        if ph > inner_h or pw > inner_w:
            raise AssertionError("rotated source patch does not fit inside target frame")
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2
        target = out[top : top + ph, left : left + pw]
        mask = patch != BG
        target[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each source sprite through its azure rail into the linked grey frame, rotating it by the rail's turn count.

    Input structure:
    - Black background with several small multi-color connected source sprites.
    - The same number of hollow grey rectangular target frames with equal black interiors.
    - One-pixel-wide azure orthogonal rails connect each source sprite to exactly one frame border without branching.

    Rule:
    - Trace every azure rail to determine which source sprite is paired with which target frame.
    - Count the number of 90-degree turns on that rail and rotate the source sprite clockwise by that many quarter-turns modulo 4.
    - Remove the original source sprites and erase all azure rail pixels.
    - Place each rotated sprite centered inside its matched target frame while keeping the grey frames unchanged.

    Key insight:
    - The rail carries two independent signals: connectivity chooses the destination frame, and the number of rail bends encodes the sprite rotation.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    return _apply_transform(g, scene)


def _rotation_signature(patch: np.ndarray) -> tuple:
    arr = np.asarray(patch, dtype=int)
    return tuple((rot.shape, rot.tobytes()) for rot in (_rotate_cw(arr, k) for k in range(4)))


def _random_connected_mask(h: int, w: int, cells: int) -> np.ndarray | None:
    chosen = {(int(RNG.integers(h)), int(RNG.integers(w)))}
    while len(chosen) < cells:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        chosen.add(frontier[int(RNG.integers(len(frontier)))])

    rows = [r for r, _ in chosen]
    cols = [c for _, c in chosen]
    if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _make_source_patch(orientation: str) -> tuple[np.ndarray, list[int]] | None:
    for _ in range(PAIR_ATTEMPTS):
        h, w = SOURCE_SIZES[int(RNG.integers(len(SOURCE_SIZES)))]
        max_cells = min(h * w, 8)
        cell_count = int(RNG.integers(max(5, h + w - 2), max_cells + 1))
        mask = _random_connected_mask(h, w, cell_count)
        if mask is None:
            continue

        color_count = 2 + int(RNG.integers(2))
        if cell_count < color_count:
            continue
        colors = list(RNG.choice(SOURCE_COLORS, size=color_count, replace=False))
        patch = np.full((h, w), BG, dtype=int)
        coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
        RNG.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if idx < color_count:
                patch[r, c] = colors[idx]
            else:
                patch[r, c] = colors[int(RNG.integers(color_count))]

        if len(set(_rotation_signature(patch))) != 4:
            continue

        if orientation == "ltr":
            boundary_rows = [r for r in range(h) if patch[r, w - 1] != BG]
        else:
            boundary_rows = [r for r in range(h) if patch[r, 0] != BG]
        if boundary_rows:
            return patch, boundary_rows
    return None


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("polyline segments must be orthogonal")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(zip(points, points[1:])):
        segment = _line_points(a, b)
        if idx:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _frame_border_bbox(top: int, left: int, h: int, w: int) -> tuple[int, int, int, int]:
    return top, left, top + h - 1, left + w - 1


def _source_global_coords(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = []
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            if patch[r, c] != BG:
                coords.append((top + r, left + c))
    return coords


def _make_pair(
    band_top: int,
    grid_w: int,
    inner_h: int,
    inner_w: int,
    turn_count: int,
) -> dict | None:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    for _ in range(PAIR_ATTEMPTS):
        orientation = "ltr" if int(RNG.integers(2)) == 0 else "rtl"
        made = _make_source_patch(orientation)
        if made is None:
            continue
        patch, boundary_rows = made
        sh, sw = patch.shape

        if orientation == "ltr":
            source_left = int(RNG.integers(1, 4))
            frame_left = grid_w - frame_w - int(RNG.integers(1, 3))
            if frame_left - (source_left + sw) < 5:
                continue
        else:
            frame_left = int(RNG.integers(1, 3))
            source_left = grid_w - sw - int(RNG.integers(1, 4))
            if source_left - (frame_left + frame_w) < 5:
                continue

        frame_top = int(RNG.integers(0, BAND_HEIGHT - frame_h + 1))
        frame_rows = list(range(frame_top + 1, frame_top + frame_h - 1))
        if not frame_rows:
            continue

        if turn_count == 0:
            placements = []
            for target_row in frame_rows:
                for anchor_row in boundary_rows:
                    source_top = target_row - anchor_row
                    if 0 <= source_top <= BAND_HEIGHT - sh:
                        placements.append((source_top, anchor_row, target_row))
            if not placements:
                continue
            source_top, anchor_row, target_row = placements[int(RNG.integers(len(placements)))]
        else:
            source_top = int(RNG.integers(0, BAND_HEIGHT - sh + 1))
            anchor_row = int(boundary_rows[int(RNG.integers(len(boundary_rows)))])
            start_row = source_top + anchor_row
            target_choices = [row for row in frame_rows if row != start_row]
            if not target_choices:
                continue
            target_row = int(target_choices[int(RNG.integers(len(target_choices)))])

        start_row = source_top + anchor_row

        if orientation == "ltr":
            start = (band_top + start_row, source_left + sw)
            end = (band_top + target_row, frame_left - 1)
            min_mid = start[1] + 1
            max_mid = end[1] - 1
        else:
            start = (band_top + start_row, source_left - 1)
            end = (band_top + target_row, frame_left + frame_w)
            min_mid = end[1] + 1
            max_mid = start[1] - 1

        if turn_count in (2, 3) and min_mid > max_mid:
            continue

        if turn_count == 0:
            points = [start, end]
        elif turn_count == 1:
            points = [start, (band_top + target_row, start[1]), end]
        elif turn_count == 2:
            mid_col = int(RNG.integers(min_mid, max_mid + 1))
            points = [start, (start[0], mid_col), (band_top + target_row, mid_col), end]
        else:
            lane_choices = [
                row
                for row in (0, BAND_HEIGHT - 1)
                if not (frame_top <= row <= frame_top + frame_h - 1)
                and row != start_row
                and row != target_row
            ]
            if not lane_choices:
                continue
            lane_row = int(lane_choices[int(RNG.integers(len(lane_choices)))])
            mid_col = int(RNG.integers(min_mid, max_mid + 1))
            points = [
                start,
                (band_top + lane_row, start[1]),
                (band_top + lane_row, mid_col),
                (band_top + target_row, mid_col),
                end,
            ]

        path = _polyline_cells(points)
        if len(path) != len(set(path)):
            continue
        if _count_turns(path) != turn_count:
            continue

        source_coords = _source_global_coords(patch, band_top + source_top, source_left)
        source_set = set(source_coords)
        frame_bbox = _frame_border_bbox(band_top + frame_top, frame_left, frame_h, frame_w)
        frame_set = _frame_cells(frame_bbox)
        if any(cell in source_set or cell in frame_set for cell in path):
            continue

        return {
            "patch": patch,
            "turns": turn_count,
            "source_top": band_top + source_top,
            "source_left": source_left,
            "source_coords": source_coords,
            "frame_bbox": frame_bbox,
            "path": path,
        }
    return None


def _draw_pair(grid: np.ndarray, pair: dict) -> None:
    patch = pair["patch"]
    top = pair["source_top"]
    left = pair["source_left"]
    ph, pw = patch.shape
    target = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    target[mask] = patch[mask]

    r0, c0, r1, c1 = pair["frame_bbox"]
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for r, c in pair["path"]:
        grid[r, c] = AZURE


def _expected_from_pairs(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in pairs:
        for r, c in pair["source_coords"]:
            out[r, c] = BG
        for r, c in pair["path"]:
            out[r, c] = BG

    for pair in pairs:
        patch = _rotate_cw(pair["patch"], pair["turns"])
        r0, c0, r1, c1 = pair["frame_bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ph, pw = patch.shape
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2
        target = out[top : top + ph, left : left + pw]
        mask = patch != BG
        target[mask] = patch[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(PAIR_COUNTS[int(RNG.integers(len(PAIR_COUNTS)))])
    inner_h, inner_w = INNER_SIZES[int(RNG.integers(len(INNER_SIZES)))]
    grid_h = pair_count * BAND_HEIGHT + (pair_count - 1) * BAND_GAP
    min_width = max(22, inner_w + 14)
    grid_w = int(RNG.integers(min_width, GRID_WIDTH_RANGE[1]))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    turn_counts = list(map(int, RNG.choice(4, size=pair_count, replace=False)))
    pairs = []
    for idx in range(pair_count):
        band_top = idx * (BAND_HEIGHT + BAND_GAP)
        pair = _make_pair(band_top, grid_w, inner_h, inner_w, turn_counts[idx])
        if pair is None:
            return None
        pairs.append(pair)
        _draw_pair(grid, pair)

    expected = _expected_from_pairs(grid, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
