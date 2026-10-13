import random
from collections import defaultdict

import numpy as np


BG = 0
FRAME = 5
ARROW_COLOR_ORDER = [1, 2, 3, 4, 7, 9]  # blue, red, green, yellow, orange, maroon
PATTERN_COLORS = [6, 8]
SIDES_BY_AXIS = {
    "horizontal": ("left", "right"),
    "vertical": ("up", "down"),
}
ARROW_SHAPES = {
    frozenset({(0, 0), (1, 1), (2, 0)}): {"dir": (0, 1), "axis": "horizontal", "tip": (1, 1)},
    frozenset({(0, 1), (1, 0), (2, 1)}): {"dir": (0, -1), "axis": "horizontal", "tip": (1, 0)},
    frozenset({(0, 1), (1, 0), (1, 2)}): {"dir": (-1, 0), "axis": "vertical", "tip": (0, 1)},
    frozenset({(0, 0), (0, 2), (1, 1)}): {"dir": (1, 0), "axis": "vertical", "tip": (1, 1)},
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_RETRIES = 300


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _hflip(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    _, _, _, c1 = _bbox(coords)
    return _normalize({(r, c1 - c) for r, c in coords})


def _vflip(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    _, _, r1, _ = _bbox(coords)
    return _normalize({(r1 - r, c) for r, c in coords})


def _rotate90(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    _, _, r1, _ = _bbox(coords)
    return _normalize({(c, r1 - r) for r, c in coords})


def _connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(coords)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME, connectivity=4):
        coords = set(comp)
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        if coords != expected:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "coords": coords,
            }
        )
    assert len(frames) >= 2
    inner_shapes = {(f["inner"][2] - f["inner"][0] + 1, f["inner"][3] - f["inner"][1] + 1) for f in frames}
    assert len(inner_shapes) == 1
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _container_of_border_pixel(frames: list[dict], pixel: tuple[int, int]) -> int:
    r, c = pixel
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1 and (r == r0 or r == r1 or c == c0 or c == c1):
            return idx
    raise AssertionError("border pixel does not belong to a frame")


def _point_in_any_box(pixel: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r, c = pixel
    for r0, c0, r1, c1 in boxes:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _find_arrows(grid: np.ndarray, frames: list[dict]) -> dict[int, list[dict]]:
    h, w = grid.shape
    boxes = [frame["bbox"] for frame in frames]
    arrow_map: dict[int, list[dict]] = defaultdict(list)
    colors = sorted(set(int(v) for v in np.unique(grid)) - {BG, FRAME})
    for color in colors:
        mask = grid == color
        for comp in _components(mask, connectivity=8):
            if len(comp) != 3:
                continue
            if any(_point_in_any_box(pt, boxes) for pt in comp):
                continue
            r0, c0, _, _ = _bbox(comp)
            norm = frozenset((r - r0, c - c0) for r, c in comp)
            if norm not in ARROW_SHAPES:
                continue
            info = ARROW_SHAPES[norm]
            tip = (r0 + info["tip"][0], c0 + info["tip"][1])
            dr, dc = info["dir"]
            cur = (tip[0] + dr, tip[1] + dc)
            hit = None
            while 0 <= cur[0] < h and 0 <= cur[1] < w:
                if grid[cur] == FRAME:
                    hit = cur
                    break
                assert grid[cur] == BG
                cur = (cur[0] + dr, cur[1] + dc)
            assert hit is not None
            arrow_map[color].append(
                {
                    "color": color,
                    "axis": info["axis"],
                    "dir": info["dir"],
                    "tip": tip,
                    "container": _container_of_border_pixel(frames, hit),
                }
            )
    assert arrow_map
    for color, arrows in arrow_map.items():
        assert len(arrows) == 2, f"arrow color {color} must appear exactly twice"
    return arrow_map


def _extract_pattern(grid: np.ndarray, frame: dict) -> dict | None:
    r0, c0, r1, c1 = frame["inner"]
    inner = grid[r0 : r1 + 1, c0 : c1 + 1]
    coords = np.argwhere((inner != BG) & (inner != FRAME))
    if len(coords) == 0:
        return None
    colors = {int(inner[r, c]) for r, c in coords}
    assert len(colors) == 1
    rel = {(int(r), int(c)) for r, c in coords}
    assert _connected(rel)
    return {"color": next(iter(colors)), "coords": rel}


def _flip_pattern(pattern: dict, inner_shape: tuple[int, int], axis: str) -> dict:
    h, w = inner_shape
    if axis == "horizontal":
        rel = {(r, w - 1 - c) for r, c in pattern["coords"]}
    else:
        rel = {(h - 1 - r, c) for r, c in pattern["coords"]}
    return {"color": pattern["color"], "coords": rel}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy patterns through paired arrows, flipping horizontally or vertically on each transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    arrows_by_color = _find_arrows(g, frames)
    inner_shape = (
        frames[0]["inner"][2] - frames[0]["inner"][0] + 1,
        frames[0]["inner"][3] - frames[0]["inner"][1] + 1,
    )

    patterns = {idx: _extract_pattern(g, frame) for idx, frame in enumerate(frames)}

    process_order = [color for color in ARROW_COLOR_ORDER if color in arrows_by_color]
    for color in process_order:
        a0, a1 = arrows_by_color[color]
        assert a0["axis"] == a1["axis"]
        c0 = a0["container"]
        c1 = a1["container"]
        p0 = patterns[c0]
        p1 = patterns[c1]
        assert (p0 is None) != (p1 is None), "one endpoint must currently contain a pattern"
        source = c0 if p0 is not None else c1
        target = c1 if source == c0 else c0
        patterns[target] = _flip_pattern(patterns[source], inner_shape, a0["axis"])

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["inner"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for idx, pattern in patterns.items():
        if pattern is None:
            continue
        r0, c0, _, _ = frames[idx]["inner"]
        for rr, cc in pattern["coords"]:
            out[r0 + rr, c0 + cc] = pattern["color"]
    return out


def _random_shape(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(100):
        target_size = rng.choice([3, 4])
        cells = {(0, 0)}
        while len(cells) < target_size:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(DIRS4)
            cells.add((r + dr, c + dc))
        cells = _normalize(cells)
        transforms = []
        cur = cells
        for _ in range(4):
            transforms.append(cur)
            transforms.append(_hflip(cur))
            transforms.append(_vflip(cur))
            cur = _rotate90(cur)
        shape = rng.choice(list({frozenset(t) for t in transforms}))
        coords = set(shape)
        if coords != _hflip(coords) and coords != _vflip(coords):
            return coords
    raise RuntimeError("failed to sample a non-symmetric shape")


def _place_shape(rng: random.Random, inner_size: int) -> set[tuple[int, int]]:
    shape = _random_shape(rng)
    _, _, r1, c1 = _bbox(shape)
    sh = r1 + 1
    sw = c1 + 1
    top = rng.randint(0, inner_size - sh)
    left = rng.randint(0, inner_size - sw)
    return {(r + top, c + left) for r, c in shape}


def _choose_layout(rng: random.Random, n_containers: int, outer: int) -> tuple[int, int, list[tuple[int, int]]]:
    gap_r = rng.randint(5, 7)
    gap_c = rng.randint(5, 7)
    top = rng.randint(3, 5)
    left = rng.randint(3, 5)
    bottom = rng.randint(3, 5)
    right = rng.randint(3, 5)
    h = top + 2 * outer + gap_r + bottom
    w = left + 3 * outer + 2 * gap_c + right
    slots = [
        (top + rr * (outer + gap_r), left + cc * (outer + gap_c))
        for rr in range(2)
        for cc in range(3)
    ]
    picks = rng.sample(range(len(slots)), n_containers)
    return h, w, [slots[idx] for idx in picks]


def _build_edges(rng: random.Random, n_containers: int) -> tuple[list[int], list[tuple[int, int]]]:
    order = list(range(n_containers))
    rng.shuffle(order)
    if n_containers <= 3:
        root_count = 1 if rng.random() < 0.7 else 2
    else:
        root_count = 1 if rng.random() < 0.65 else 2
    root_count = min(root_count, 2, n_containers - 1)
    roots = order[:root_count]
    edges: list[tuple[int, int]] = []
    tips = roots[:]
    last_target = None
    for target in order[root_count:]:
        if last_target in tips and rng.random() < 0.65:
            source = last_target
        else:
            source = rng.choice(tips)
        tips.remove(source)
        tips.append(target)
        edges.append((source, target))
        last_target = target
    return roots, edges


def _axis_sequence(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return [rng.choice(["horizontal", "vertical"])]
    if count == 2:
        axes = ["horizontal", "vertical"]
        rng.shuffle(axes)
        return axes
    axes = [rng.choice(["horizontal", "vertical"]) for _ in range(count)]
    if len(set(axes)) == 1:
        flip_idx = rng.randrange(count)
        axes[flip_idx] = "vertical" if axes[flip_idx] == "horizontal" else "horizontal"
    return axes


def _pick_side(rng: random.Random, used: set[str], axis: str) -> str | None:
    choices = [side for side in SIDES_BY_AXIS[axis] if side not in used]
    if not choices:
        return None
    return rng.choice(choices)


def _arrow_cells(bbox: tuple[int, int, int, int], side: str) -> tuple[list[tuple[int, int]], str]:
    r0, c0, r1, c1 = bbox
    mid_r = r0 + (r1 - r0) // 2
    mid_c = c0 + (c1 - c0) // 2
    if side == "left":
        return [(mid_r - 1, c0 - 2), (mid_r, c0 - 1), (mid_r + 1, c0 - 2)], "horizontal"
    if side == "right":
        return [(mid_r - 1, c1 + 2), (mid_r, c1 + 1), (mid_r + 1, c1 + 2)], "horizontal"
    if side == "up":
        return [(r0 - 2, mid_c - 1), (r0 - 1, mid_c), (r0 - 2, mid_c + 1)], "vertical"
    return [(r1 + 2, mid_c - 1), (r1 + 1, mid_c), (r1 + 2, mid_c + 1)], "vertical"


def _simulate_patterns(
    initial_patterns: dict[int, dict | None],
    edges: list[dict],
    inner_size: int,
) -> dict[int, dict | None]:
    state = {
        idx: None if pattern is None else {"color": pattern["color"], "coords": set(pattern["coords"])}
        for idx, pattern in initial_patterns.items()
    }
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        assert state[source] is not None and state[target] is None
        state[target] = _flip_pattern(state[source], (inner_size, inner_size), edge["axis"])
    return state


def _paint_pattern(grid: np.ndarray, inner_bbox: tuple[int, int, int, int], pattern: dict) -> None:
    r0, c0, _, _ = inner_bbox
    for rr, cc in pattern["coords"]:
        grid[r0 + rr, c0 + cc] = pattern["color"]


def _build_expected(grid: np.ndarray, frames: list[dict], final_patterns: dict[int, dict | None]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["inner"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for idx, pattern in final_patterns.items():
        if pattern is not None:
            _paint_pattern(out, frames[idx]["inner"], pattern)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_size = rng.randint(4, 5)
    outer = inner_size + 2
    n_containers = rng.randint(3, 6)
    h, w, positions = _choose_layout(rng, n_containers, outer)
    frames = []
    for r0, c0 in positions:
        frame = {
            "bbox": (r0, c0, r0 + outer - 1, c0 + outer - 1),
            "inner": (r0 + 1, c0 + 1, r0 + outer - 2, c0 + outer - 2),
        }
        frames.append(frame)

    roots, raw_edges = _build_edges(rng, n_containers)
    edge_colors = sorted(rng.sample(ARROW_COLOR_ORDER, len(raw_edges)), key=ARROW_COLOR_ORDER.index)
    target_axes = _axis_sequence(rng, len(raw_edges))
    used_sides = [set() for _ in range(n_containers)]
    edges: list[dict] = []
    for (source, target), color, preferred_axis in zip(raw_edges, edge_colors, target_axes):
        for axis in (preferred_axis, "vertical" if preferred_axis == "horizontal" else "horizontal"):
            src_side = _pick_side(rng, used_sides[source], axis)
            dst_side = _pick_side(rng, used_sides[target], axis)
            if src_side is None or dst_side is None:
                continue
            used_sides[source].add(src_side)
            used_sides[target].add(dst_side)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "color": color,
                    "axis": axis,
                    "source_side": src_side,
                    "target_side": dst_side,
                }
            )
            break
        else:
            return None

    initial_patterns = {idx: None for idx in range(n_containers)}
    pattern_palette = rng.sample(PATTERN_COLORS, len(roots))
    for root, color in zip(roots, pattern_palette):
        initial_patterns[root] = {"color": color, "coords": _place_shape(rng, inner_size)}
    final_patterns = _simulate_patterns(initial_patterns, edges, inner_size)

    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0 : r1 + 1, [c0, c1]] = FRAME
        grid[[r0, r1], c0 : c1 + 1] = FRAME
    for idx, pattern in initial_patterns.items():
        if pattern is not None:
            _paint_pattern(grid, frames[idx]["inner"], pattern)

    for edge in edges:
        for which in ("source_side", "target_side"):
            container = edge["source"] if which == "source_side" else edge["target"]
            cells, axis = _arrow_cells(frames[container]["bbox"], edge[which])
            assert axis == edge["axis"]
            for r, c in cells:
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if grid[r, c] != BG:
                    return None
            for r, c in cells:
                grid[r, c] = edge["color"]

    expected = _build_expected(grid, frames, final_patterns)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
