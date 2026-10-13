from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PATH_COLORS = (BLUE, RED, GREEN, MAGENTA, ORANGE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

COUNT_OPTIONS = tuple(range(2, 8))
SIZE_OPTIONS = (
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
)
PATCH_SIZE_OPTIONS = (
    (5, 6),
    (5, 7),
    (6, 6),
    (6, 7),
    (6, 8),
    (7, 7),
    (7, 8),
    (8, 8),
    (8, 9),
)
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(sub: np.ndarray, color: int) -> bool:
    h, w = sub.shape
    if h < 2 or w < 2:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(sub == int(color), border)


def _draw_border(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = int(color)
    grid[top + h - 1, left : left + w] = int(color)
    grid[top : top + h, left] = int(color)
    grid[top : top + h, left + w - 1] = int(color)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if _is_rect_frame(sub, GREY):
            candidates.append((int(comp.shape[0]), r0, c0, r1, c1))
    assert candidates, "expected one grey legend frame"
    candidates.sort(key=lambda item: (item[0], item[4], -item[1]), reverse=True)
    _area, r0, c0, r1, c1 = candidates[0]
    return r0, c0, r1, c1


def _read_legend(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]

    azure_frames: list[tuple[int, int, int, int]] = []
    for comp in _components(inner == AZURE):
        ar0, ac0, ar1, ac1 = _bbox(comp)
        sub = inner[ar0 : ar1 + 1, ac0 : ac1 + 1]
        if _is_rect_frame(sub, AZURE):
            azure_frames.append((ar0, ac0, ar1, ac1))

    mapping: dict[int, tuple[int, int]] = {}
    for comp in _components(inner == MAROON):
        mr0, mc0, mr1, mc1 = _bbox(comp)
        if mr0 != mr1:
            continue
        count = int(comp.shape[0])
        matches: list[tuple[int, tuple[int, int, int, int]]] = []
        for frame in azure_frames:
            ar0, ac0, ar1, ac1 = frame
            if ac0 > mc1 and ar0 <= mr0 <= ar1:
                matches.append((ac0 - mc1, frame))
        assert matches, f"missing legend frame for count {count}"
        _dist, frame = min(matches, key=lambda item: item[0])
        ar0, ac0, ar1, ac1 = frame
        mapping[count] = (ar1 - ar0 + 1, ac1 - ac0 + 1)

    assert mapping, "legend mapping must be non-empty"
    return mapping


def _path_degrees(coords: np.ndarray) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    pts = {tuple(map(int, rc)) for rc in coords}
    degrees: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in pts:
                deg += 1
        degrees[(r, c)] = deg
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    return degrees, endpoints


def _order_path(coords: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in coords}
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        nxts = []
        for dr, dc in DIR4:
            cand = (cur[0] + dr, cur[1] + dc)
            if cand in pts and cand != prev:
                nxts.append(cand)
        if not nxts:
            break
        assert len(nxts) == 1, "path must be branch-free"
        prev, cur = cur, nxts[0]
    assert len(order) == len(pts), "path traversal must visit every pixel once"
    return order


def _collect_paths(grid: np.ndarray) -> list[dict[str, object]]:
    legend_map = _read_legend(grid)

    yellow_label = np.zeros(grid.shape, dtype=int)
    yellow_heights: dict[int, int] = {}
    yellow_id = 0
    for comp in _components(grid == YELLOW):
        yellow_id += 1
        yellow_label[comp[:, 0], comp[:, 1]] = yellow_id
        yr0, yc0, yr1, yc1 = _bbox(comp)
        assert yc0 == yc1, "instruction block must be one pixel wide"
        yellow_heights[yellow_id] = yr1 - yr0 + 1

    reserved = {BLACK, YELLOW, GREY, AZURE, MAROON}
    infos: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color in reserved:
            continue
        for comp in _components(grid == color):
            degrees, endpoints = _path_degrees(comp)
            if len(endpoints) != 2:
                continue
            if any(deg not in (1, 2) for deg in degrees.values()):
                continue

            start: tuple[int, int] | None = None
            step = None
            for endpoint in endpoints:
                touching = set()
                er, ec = endpoint
                for dr, dc in DIR4:
                    nr = er + dr
                    nc = ec + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        label = int(yellow_label[nr, nc])
                        if label != 0:
                            touching.add(label)
                if touching:
                    assert len(touching) == 1
                    label = touching.pop()
                    assert start is None, "each path must have a unique instructed endpoint"
                    start = endpoint
                    step = yellow_heights[label]

            if start is None or step is None:
                continue

            ordered = _order_path(comp, start)
            sample_count = len(ordered[::step])
            assert sample_count in legend_map, f"legend missing count {sample_count}"
            bbox = _bbox(comp)
            infos.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "size": legend_map[sample_count],
                }
            )

    assert 2 <= len(infos) <= 3, "expected 2-3 instructed paths"
    infos.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["color"]))  # type: ignore[index]
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sample each colored path using the height of its adjacent yellow instruction block,
    then remap the sample count through the top-right legend into a rectangular border.
    """
    g = np.asarray(grid, dtype=int)
    infos = _collect_paths(g)

    out_h = max(int(info["size"][0]) for info in infos)  # type: ignore[index]
    out_w = sum(int(info["size"][1]) for info in infos) + len(infos) - 1  # type: ignore[index]
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for info in infos:
        color = int(info["color"])
        h, w = info["size"]  # type: ignore[misc]
        _draw_border(out, 0, col, int(h), int(w), color)
        col += int(w) + 1
    return out


def _compose_expected(path_specs: list[dict[str, object]]) -> np.ndarray:
    out_h = max(int(spec["size"][0]) for spec in path_specs)  # type: ignore[index]
    out_w = sum(int(spec["size"][1]) for spec in path_specs) + len(path_specs) - 1  # type: ignore[index]
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for spec in path_specs:
        h, w = spec["size"]  # type: ignore[misc]
        _draw_border(out, 0, col, int(h), int(w), int(spec["color"]))
        col += int(w) + 1
    return out


def _free_counts(used: set[int]) -> list[int]:
    return [count for count in COUNT_OPTIONS if count not in used]


def _render_legend(entries: list[dict[str, object]], rng: np.random.Generator) -> np.ndarray:
    entries = list(entries)
    rng.shuffle(entries)
    inner_h = sum(int(entry["size"][0]) for entry in entries) + max(0, len(entries) - 1)  # type: ignore[index]
    inner_w = 2 + max(int(entry["count"]) + int(entry["size"][1]) for entry in entries)  # type: ignore[index]
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    _draw_border(patch, 0, 0, patch.shape[0], patch.shape[1], GREY)

    top = 1
    for entry in entries:
        count = int(entry["count"])
        h, w = entry["size"]  # type: ignore[misc]
        marker_row = top + int(rng.integers(0, h))
        patch[marker_row, 1 : 1 + count] = MAROON
        frame_left = 1 + count
        _draw_border(patch, top, frame_left, int(h), int(w), AZURE)
        top += int(h) + 1
    return patch


def _induced_path_candidates(
    h: int,
    w: int,
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = cur[0] + dr
        nc = cur[1] + dc
        nxt = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if nxt in used or nxt in blocked:
            continue
        touching = 0
        valid = True
        for sdr, sdc in DIR4:
            rr = nr + sdr
            cc = nc + sdc
            other = (rr, cc)
            if other in used:
                if other != cur:
                    valid = False
                    break
                touching += 1
        if valid and touching == 1:
            out.append(nxt)
    return out


def _search_induced_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    start: tuple[int, int],
    length: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    used = {start}
    path = [start]

    def onward_score(cell: tuple[int, int]) -> int:
        return len(_induced_path_candidates(h, w, cell, used | {cell}, blocked))

    def dfs(cur: tuple[int, int]) -> bool:
        if len(path) == length:
            return True
        cands = _induced_path_candidates(h, w, cur, used, blocked)
        if not cands:
            return False
        order = list(range(len(cands)))
        rng.shuffle(order)
        ordered = [cands[i] for i in order]
        ordered.sort(key=onward_score, reverse=True)
        for nxt in ordered:
            used.add(nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            path.pop()
            used.remove(nxt)
        return False

    if dfs(start):
        return path
    return None


def _make_path_patch(
    rng: np.random.Generator,
    color: int,
    step: int,
    sample_count: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    min_len = (sample_count - 1) * step + 1
    max_len = sample_count * step
    for _ in range(200):
        length = int(rng.integers(min_len, max_len + 1))
        sizes = [
            (h, w)
            for h, w in PATCH_SIZE_OPTIONS
            if h >= step + 1 and h * w - step >= length + 2
        ]
        if not sizes:
            break
        ph, pw = sizes[int(rng.integers(0, len(sizes)))]
        side_left = bool(rng.integers(0, 2))
        block_col = 0 if side_left else pw - 1
        start_col = 1 if side_left else pw - 2
        start_row = int(rng.integers(0, ph))
        top_min = max(0, start_row - step + 1)
        top_max = min(start_row, ph - step)
        if top_min > top_max:
            continue
        block_top = int(rng.integers(top_min, top_max + 1))

        blocked = {(r, block_col) for r in range(block_top, block_top + step)}
        for r in range(block_top, block_top + step):
            if r != start_row:
                blocked.add((r, start_col))

        start = (start_row, start_col)
        path = _search_induced_path(rng, ph, pw, start, length, blocked)
        if path is None:
            continue

        patch = np.zeros((ph, pw), dtype=int)
        for r, c in path:
            patch[r, c] = int(color)
        patch[block_top : block_top + step, block_col] = YELLOW
        return patch, path
    raise RuntimeError("failed to build path patch")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_paths = int(rng.integers(2, 4))
    colors = [int(x) for x in rng.choice(np.asarray(PATH_COLORS), size=num_paths, replace=False)]
    steps = [int(x) for x in rng.integers(1, 5, size=num_paths)]
    counts = [int(x) for x in rng.choice(np.asarray(COUNT_OPTIONS), size=num_paths, replace=False)]

    size_pool = list(SIZE_OPTIONS)
    rng.shuffle(size_pool)
    used_entries: list[dict[str, object]] = []
    count_to_size: dict[int, tuple[int, int]] = {}
    for idx, count in enumerate(counts):
        size = size_pool[idx]
        count_to_size[count] = size
        used_entries.append({"count": count, "size": size})

    extras = int(rng.integers(0, 2))
    remaining_counts = _free_counts(set(counts))
    extra_entries: list[dict[str, object]] = []
    size_index = num_paths
    for _ in range(extras):
        if not remaining_counts or size_index >= len(size_pool):
            break
        pick = int(rng.integers(0, len(remaining_counts)))
        count = remaining_counts.pop(pick)
        extra_entries.append({"count": count, "size": size_pool[size_index]})
        size_index += 1

    legend_patch = _render_legend(used_entries + extra_entries, rng)

    path_specs: list[dict[str, object]] = []
    patches: list[np.ndarray] = []
    for color, step, count in zip(colors, steps, counts):
        patch, ordered = _make_path_patch(rng, color, step, count)
        patches.append(patch)
        path_specs.append({"color": color, "size": count_to_size[count], "ordered": ordered})

    gap_y = int(rng.integers(2, 4))
    top_margin = int(rng.integers(0, 2))
    bottom_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(0, 2))
    left_margin = int(rng.integers(1, 3))
    between_paths = [int(rng.integers(2, 4)) for _ in range(num_paths - 1)]

    path_area_top = top_margin + legend_patch.shape[0] + gap_y
    path_area_height = max(patch.shape[0] for patch in patches) + int(rng.integers(1, 3))
    grid_h = path_area_top + path_area_height + bottom_margin

    path_total_w = left_margin + sum(patch.shape[1] for patch in patches) + sum(between_paths) + right_margin
    grid_w = max(path_total_w, legend_patch.shape[1] + 2 + right_margin)

    grid = np.zeros((grid_h, grid_w), dtype=int)

    legend_right = grid_w - 1 - right_margin
    legend_left = legend_right - legend_patch.shape[1] + 1
    if legend_left < 0:
        return None
    grid[top_margin : top_margin + legend_patch.shape[0], legend_left : legend_left + legend_patch.shape[1]] = legend_patch

    cur_left = left_margin
    for idx, patch in enumerate(patches):
        y_slack = path_area_height - patch.shape[0]
        top = path_area_top + int(rng.integers(0, y_slack + 1))
        left = cur_left
        sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        if np.any((sub != BLACK) & (patch != BLACK)):
            return None
        mask = patch != BLACK
        sub[mask] = patch[mask]
        cur_left += patch.shape[1]
        if idx < len(between_paths):
            cur_left += between_paths[idx]

    expected = _compose_expected(path_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
