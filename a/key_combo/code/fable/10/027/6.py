from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, skip: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color = int(grid[r0, c0])
            if color in skip:
                continue
            seen[r0, c0] = True
            stack = [(r0, c0)]
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in CARDINALS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_hollow_frame(comp: dict, grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 7:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    if set(comp["coords"]) != expected:
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _path_components(grid: np.ndarray, path_color: int) -> list[list[tuple[int, int]]]:
    nodes = {tuple(map(int, rc)) for rc in np.argwhere(grid == path_color)}
    comps: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in nodes:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in STEP2:
                nb = (r + dr, c + dc)
                if nb in nodes and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        comps.append(sorted(comp))
    return comps


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components_by_color(g, {bg, GREY})

    frames = [comp for comp in comps if _is_hollow_frame(comp, g, bg)]
    frames.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    assert len(frames) >= 2, "expected multiple hollow frames"

    patterns = [comp for comp in comps if comp not in frames]
    assert patterns, "expected at least one outside pattern sprite"

    frame_map: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for rc in frame["coords"]:
            frame_map[rc] = idx

    pattern_map: dict[tuple[int, int], int] = {}
    for idx, pat in enumerate(patterns):
        for rc in pat["coords"]:
            pattern_map[rc] = idx

    path_infos: list[dict] = []
    for comp in _path_components(g, GREY):
        touched_frames: set[int] = set()
        touched_patterns: set[int] = set()
        for r, c in comp:
            for dr, dc in CARDINALS:
                nb = (r + dr, c + dc)
                if nb in frame_map:
                    touched_frames.add(frame_map[nb])
                if nb in pattern_map:
                    touched_patterns.add(pattern_map[nb])
        assert len(touched_frames) == 1, "every dotted path must touch exactly one frame"
        assert len(touched_patterns) == 1, "every dotted path must touch exactly one pattern"
        path_infos.append(
            {
                "nodes": comp,
                "frame_idx": next(iter(touched_frames)),
                "pattern_idx": next(iter(touched_patterns)),
            }
        )
    assert path_infos, "expected at least one grey dotted path"
    return bg, frames, patterns, path_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each outside sprite into the frame reached by its grey dotted path, then record how many arrived as interior dots.

    Input structure:
    - The background is the dominant color.
    - Several different-colored hollow rectangular frames sit apart with empty interiors.
    - Outside the frames are small monochrome pattern sprites, each connected to one frame wall by a grey dotted path whose dots are spaced every other cell.
    - A sprite color may match its destination frame or may belong to some other frame color.

    Rule:
    - Identify the hollow frames and the grey dotted-path components using the path's two-cell horizontal/vertical connectivity.
    - Erase every outside sprite and every grey path.
    - For each frame, count how many path components touch that frame wall.
    - On that frame's center row, draw that many frame-colored dots from right to left with one blank cell between consecutive dots.

    Key insight:
    - Ownership is determined by which frame wall the grey path reaches; visible color matches are only a lure and must not override path connectivity.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, patterns, path_infos = _parse_scene(g)

    counts = [0] * len(frames)
    for info in path_infos:
        counts[int(info["frame_idx"])] += 1

    out = g.copy()
    out[g == GREY] = bg
    for pat in patterns:
        for r, c in pat["coords"]:
            out[r, c] = bg

    for frame, count in zip(frames, counts):
        r0, c0, r1, c1 = frame["bbox"]
        row = (r0 + r1) // 2
        color = int(frame["color"])
        for i in range(int(count)):
            col = c1 - 2 - 2 * i
            if col > c0:
                out[row, col] = color
    return out


FRAME_H_CHOICES = (5, 7)
FRAME_W_CHOICES = (7, 9, 11)
FRAME_COUNT_RANGE = (2, 4)
FRAME_GAP_RANGE = (2, 4)
GRID_H_RANGE = (19, 30)
GRID_W_RANGE = (23, 30)
TOTAL_PATH_RANGE = (3, 7)
EXTRA_COLOR_CHOICES = (1, 2)
GEN_ATTEMPTS = 400

SPRITE_BASES = [
    {(0, 0), (1, 0)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (1, 1)},
]


def _rotate(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in coords}


def _reflect(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in coords}


def _sprite_templates() -> list[list[tuple[int, int]]]:
    templates: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in SPRITE_BASES:
        variants = [set(base)]
        for _ in range(3):
            variants.append(_rotate(variants[-1]))
        all_forms = variants + [_reflect(v) for v in variants]
        for form in all_forms:
            if (0, 0) not in form:
                continue
            key = tuple(sorted(form))
            if key not in seen:
                seen.add(key)
                templates.append(list(sorted(form)))
    return templates


SPRITE_TEMPLATES = _sprite_templates()


def _frame_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _dot_capacity(frame: dict) -> int:
    _, c0, _, c1 = frame["bbox"]
    return max(1, (c1 - c0 - 2) // 2)


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_output(shape: tuple[int, int], bg: int, frames: list[dict], counts: list[int]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for frame in frames:
        _draw_frame(out, frame)
    for frame, count in zip(frames, counts):
        r0, c0, r1, c1 = frame["bbox"]
        row = (r0 + r1) // 2
        for i in range(int(count)):
            out[row, c1 - 2 - 2 * i] = int(frame["color"])
    return out


def _clean_frame_input(shape: tuple[int, int], bg: int, frames: list[dict]) -> np.ndarray:
    inp = np.full(shape, bg, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame)
    return inp


def _sample_layout() -> tuple[np.ndarray, int, list[dict], list[int], int]:
    for _ in range(200):
        n_frames = random.choices([2, 3, 4], weights=[3, 3, 2])[0]
        if n_frames == 4:
            heights = [5 if random.random() < 0.85 else 7 for _ in range(n_frames)]
            widths = [random.choice((7, 9)) for _ in range(n_frames)]
            gaps = [random.randint(2, 3) for _ in range(n_frames - 1)]
        else:
            heights = [random.choice(FRAME_H_CHOICES) for _ in range(n_frames)]
            widths = [random.choice(FRAME_W_CHOICES) for _ in range(n_frames)]
            gaps = [random.randint(*FRAME_GAP_RANGE) for _ in range(n_frames - 1)]

        total_h = sum(heights) + sum(gaps)
        h_min = max(GRID_H_RANGE[0], total_h + 2)
        h_max = GRID_H_RANGE[1]
        if h_min > h_max:
            continue
        H = random.randint(h_min, h_max)

        right_margin_need = 10
        w_min = max(GRID_W_RANGE[0], max(widths) + right_margin_need + 4)
        if w_min > GRID_W_RANGE[1]:
            continue
        W = random.randint(w_min, GRID_W_RANGE[1])

        available_top = H - total_h - 1
        if available_top < 1:
            continue
        top = random.randint(1, available_top)
        frames: list[dict] = []
        row = top

        palette = [c for c in range(10) if c != GREY]
        bg = random.choice(palette)
        palette.remove(bg)
        random.shuffle(palette)
        frame_colors = palette[:n_frames]

        ok = True
        for i in range(n_frames):
            max_c0 = min(8, W - widths[i] - right_margin_need)
            if max_c0 < 1:
                ok = False
                break
            c0 = random.randint(1, max_c0)
            frame = {
                "color": int(frame_colors[i]),
                "bbox": (row, c0, row + heights[i] - 1, c0 + widths[i] - 1),
            }
            frame["coords"] = sorted(_frame_cells(frame))
            frames.append(frame)
            if i < n_frames - 1:
                row += heights[i] + gaps[i]
        if not ok:
            continue

        force_zero = n_frames >= 3 and random.random() < 0.65
        while True:
            counts = [random.randint(0, _dot_capacity(frame)) for frame in frames]
            total = sum(counts)
            if TOTAL_PATH_RANGE[0] <= total <= TOTAL_PATH_RANGE[1] and max(counts) >= 2:
                if (not force_zero) or any(count == 0 for count in counts):
                    break

        route_c0 = max(frame["bbox"][3] for frame in frames) + 4
        out = _make_output((H, W), bg, frames, counts)
        return out, bg, frames, counts, route_c0
    raise RuntimeError("failed to sample a valid frame layout")


def _start_candidates(frame: dict, shape: tuple[int, int], grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = shape
    r0, c0, r1, c1 = frame["bbox"]
    candidates: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        candidates.append((r, c0 - 1))
        candidates.append((r, c1 + 1))
    for c in range(c0 + 1, c1):
        candidates.append((r0 - 1, c))
        candidates.append((r1 + 1, c))
    valid: list[tuple[int, int]] = []
    wall = _frame_cells(frame)
    for r, c in candidates:
        if not (0 <= r < H and 0 <= c < W):
            continue
        if int(grid[r, c]) != bg:
            continue
        touching = 0
        blocked = False
        for dr, dc in CARDINALS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in wall:
                touching += 1
            elif int(grid[rr, cc]) != bg:
                blocked = True
        if touching == 1 and not blocked:
            valid.append((r, c))
    return valid


def _node_ok(
    grid: np.ndarray,
    bg: int,
    gray_nodes: set[tuple[int, int]],
    node: tuple[int, int],
    *,
    exempt: set[tuple[int, int]] | None = None,
) -> bool:
    H, W = grid.shape
    r, c = node
    if not (0 <= r < H and 0 <= c < W):
        return False
    if int(grid[r, c]) != bg:
        return False
    exempt = exempt or set()
    for dr, dc in CARDINALS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != bg and (rr, cc) not in exempt:
            return False
    for dr, dc in STEP2:
        rr, cc = r + dr, c + dc
        if (rr, cc) in gray_nodes:
            return False
    return True


def _bfs_path(
    grid: np.ndarray,
    bg: int,
    gray_nodes: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    frame_exempt: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    prev = {start: None}
    H, W = grid.shape
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        steps = list(STEP2)
        random.shuffle(steps)
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            mid = (r + dr // 2, c + dc // 2)
            nb = (nr, nc)
            if nb in prev:
                continue
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if int(grid[mid]) != bg:
                continue
            exempt = frame_exempt if (r, c) == start and nb != start else set()
            if nb != end and not _node_ok(grid, bg, gray_nodes, nb, exempt=exempt):
                continue
            if nb == end and not _node_ok(grid, bg, gray_nodes, nb, exempt=exempt):
                continue
            prev[nb] = (r, c)
            queue.append(nb)
    if end not in prev:
        return None
    path: list[tuple[int, int]] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _choose_path(
    grid: np.ndarray,
    bg: int,
    gray_nodes: set[tuple[int, int]],
    frame: dict,
    route_c0: int,
) -> list[tuple[int, int]] | None:
    starts = _start_candidates(frame, grid.shape, grid, bg)
    random.shuffle(starts)
    wall = _frame_cells(frame)
    H, W = grid.shape
    for start in starts:
        needs_turn = random.random() < 0.75
        candidates: list[tuple[int, int]] = []
        for r in range(1, H - 1):
            for c in range(route_c0, W - 1):
                node = (r, c)
                if node == start:
                    continue
                if abs(r - start[0]) + abs(c - start[1]) < 6:
                    continue
                if needs_turn and (r == start[0] or c == start[1]):
                    continue
                if not _node_ok(grid, bg, gray_nodes, node):
                    continue
                candidates.append(node)
        if not candidates:
            continue
        candidates.sort(key=lambda rc: abs(rc[0] - start[0]) + abs(rc[1] - start[1]), reverse=True)
        for end in random.sample(candidates[: min(30, len(candidates))], min(12, len(candidates))):
            path = _bfs_path(grid, bg, gray_nodes, start, end, wall)
            if path is not None and len(path) >= 3:
                return path
    return None


def _place_sprite(
    grid: np.ndarray,
    bg: int,
    gray_nodes: set[tuple[int, int]],
    end: tuple[int, int],
    color: int,
) -> bool:
    H, W = grid.shape
    anchors = list(CARDINALS)
    random.shuffle(anchors)
    for dr, dc in anchors:
        ar, ac = end[0] + dr, end[1] + dc
        if not (0 <= ar < H and 0 <= ac < W):
            continue
        if int(grid[ar, ac]) != bg:
            continue
        sprite_exempt = {end}
        if not _node_ok(grid, bg, gray_nodes - {end}, (ar, ac), exempt=sprite_exempt):
            continue
        templates = SPRITE_TEMPLATES[:]
        random.shuffle(templates)
        for template in templates:
            cells = [(ar + rr, ac + cc) for rr, cc in template]
            if not all(0 <= r < H and 0 <= c < W for r, c in cells):
                continue
            if any(int(grid[r, c]) != bg for r, c in cells):
                continue
            ok = True
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in cells:
                        continue
                    if (r, c) == (ar, ac) and (nr, nc) == end:
                        continue
                    if int(grid[nr, nc]) != bg:
                        ok = False
                    if (nr, nc) in gray_nodes:
                        ok = False
                if not ok:
                    break
            if not ok:
                continue
            for r, c in cells:
                grid[r, c] = int(color)
            return True
    return False


def _assign_pattern_colors(frame_colors: list[int], path_targets: list[int]) -> list[int]:
    remaining = [c for c in range(10) if c not in set(frame_colors) | {GREY}]
    random.shuffle(remaining)
    pool = list(frame_colors) + remaining[: random.choice(EXTRA_COLOR_CHOICES)]
    colors = [random.choice(pool) for _ in path_targets]

    if not any(colors[i] != frame_colors[path_targets[i]] for i in range(len(colors))):
        idx = random.randrange(len(colors))
        options = [c for c in pool if c != frame_colors[path_targets[idx]]]
        colors[idx] = random.choice(options)

    if len(frame_colors) >= 2:
        idx = random.randrange(len(colors))
        other = [c for c in frame_colors if c != frame_colors[path_targets[idx]]]
        colors[idx] = random.choice(other)

    actual = [0] * len(frame_colors)
    for target in path_targets:
        actual[target] += 1
    color_match = [sum(1 for col in colors if col == frame_colors[i]) for i in range(len(frame_colors))]
    if color_match == actual:
        idx = random.randrange(len(colors))
        options = [c for c in pool if c != frame_colors[path_targets[idx]]]
        colors[idx] = random.choice(options)
    return colors


def _validate_generated(input_grid: np.ndarray, output_grid: np.ndarray, expected_counts: list[int]) -> bool:
    try:
        bg, frames, patterns, path_infos = _parse_scene(input_grid)
    except AssertionError:
        return False
    if len(frames) != len(expected_counts):
        return False
    counts = [0] * len(frames)
    for info in path_infos:
        counts[int(info["frame_idx"])] += 1
    if counts != expected_counts:
        return False
    if not np.array_equal(main(input_grid), output_grid):
        return False
    frame_colors = [int(frame["color"]) for frame in frames]
    pattern_colors = [int(patterns[info["pattern_idx"]]["color"]) for info in path_infos]
    color_match = [sum(1 for col in pattern_colors if col == frame_colors[i]) for i in range(len(frames))]
    if color_match == counts:
        return False
    if not any(pattern_colors[i] != frame_colors[path_infos[i]["frame_idx"]] for i in range(len(path_infos))):
        return False
    return int(bg) != GREY


def _gen_attempt() -> np.ndarray | None:
    out, bg, frames, counts, route_c0 = _sample_layout()
    inp = _clean_frame_input(out.shape, bg, frames)
    frame_colors = [int(frame["color"]) for frame in frames]

    path_targets = [idx for idx, count in enumerate(counts) for _ in range(count)]
    random.shuffle(path_targets)
    pattern_colors = _assign_pattern_colors(frame_colors, path_targets)

    gray_nodes: set[tuple[int, int]] = set()
    order = list(range(len(path_targets)))
    random.shuffle(order)
    for i in order:
        frame = frames[path_targets[i]]
        path = _choose_path(inp, bg, gray_nodes, frame, route_c0)
        if path is None:
            return None
        for r, c in path:
            inp[r, c] = GREY
        new_nodes = set(path)
        gray_nodes |= new_nodes
        if not _place_sprite(inp, bg, gray_nodes, path[-1], pattern_colors[i]):
            return None

    if _validate_generated(inp, out, counts):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
