import random
from collections import defaultdict

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3)
MARKER_COLORS = (4, 7, 9)
PATH_COLOR = 5
SOURCE_STRATEGY = "col"
SOURCE_MODE = "hv"
TARGET_STRATEGIES = ("row", "sum")
ROUTE_MODES = ("hv", "vh")
ALL_STRATEGIES = ("col", "row", "sum")
ALL_ALGOS = tuple((strategy, mode) for strategy in ALL_STRATEGIES for mode in ROUTE_MODES)
GEN_ATTEMPTS = 400


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in FRAME_COLORS:
        for comp in components(grid == color):
            r0, c0, r1, c1 = bbox_coords(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 5 or ww < 5:
                continue
            expected_area = 2 * (hh + ww) - 4
            if len(comp) != expected_area:
                continue
            border_ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border and int(grid[rr, cc]) != color:
                        border_ok = False
                    if (not on_border) and int(grid[rr, cc]) == color:
                        border_ok = False
                    if not border_ok:
                        break
                if not border_ok:
                    break
            if border_ok:
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def marker_signature(markers: dict[int, tuple[int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple((color, pos[0], pos[1]) for color, pos in sorted(markers.items()))


def strategy_key(strategy: str, pos: tuple[int, int]) -> tuple[int, int, int]:
    r, c = pos
    if strategy == "col":
        return c, r, 0
    if strategy == "row":
        return r, c, 0
    if strategy == "sum":
        return r + c, r, c
    raise ValueError(strategy)


def ordered_colors(markers: dict[int, tuple[int, int]], strategy: str) -> list[int]:
    keys = {color: strategy_key(strategy, pos) for color, pos in markers.items()}
    if len(set(keys.values())) != len(keys):
        raise ValueError("ambiguous strategy ordering")
    return sorted(markers, key=lambda color: keys[color] + (color,))


def edge_set_for_strategy(markers: dict[int, tuple[int, int]], strategy: str) -> frozenset[frozenset[int]]:
    order = ordered_colors(markers, strategy)
    return frozenset((frozenset((order[0], order[1])), frozenset((order[1], order[2]))))


def route_cells(a: tuple[int, int], b: tuple[int, int], mode: str) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2 or c1 == c2:
        raise ValueError("aligned markers make route mode ambiguous")
    cells: list[tuple[int, int]] = []
    if mode == "hv":
        step_c = 1 if c2 > c1 else -1
        for cc in range(c1 + step_c, c2 + step_c, step_c):
            cells.append((r1, cc))
        step_r = 1 if r2 > r1 else -1
        for rr in range(r1 + step_r, r2, step_r):
            cells.append((rr, c2))
        if cells and cells[-1] == b:
            cells.pop()
    elif mode == "vh":
        step_r = 1 if r2 > r1 else -1
        for rr in range(r1 + step_r, r2 + step_r, step_r):
            cells.append((rr, c1))
        step_c = 1 if c2 > c1 else -1
        for cc in range(c1 + step_c, c2, step_c):
            cells.append((r2, cc))
        if cells and cells[-1] == b:
            cells.pop()
    else:
        raise ValueError(mode)
    return cells


def build_paths(
    markers: dict[int, tuple[int, int]],
    strategy: str,
    mode: str,
) -> tuple[list[tuple[int, int]], list[set[tuple[int, int]]]]:
    order = ordered_colors(markers, strategy)
    marker_cells = set(markers.values())
    all_cells: list[tuple[int, int]] = []
    segments: list[set[tuple[int, int]]] = []
    for idx in range(2):
        start = markers[order[idx]]
        end = markers[order[idx + 1]]
        cells = route_cells(start, end, mode)
        cell_set = set(cells)
        if not cell_set:
            raise ValueError("empty route")
        if cell_set & marker_cells:
            raise ValueError("route hits marker")
        for other_color, other_pos in markers.items():
            if other_pos in (start, end):
                continue
            for rr, cc in cells:
                if abs(rr - other_pos[0]) + abs(cc - other_pos[1]) == 1:
                    raise ValueError("route touches unrelated marker")
        for existing in segments:
            if cell_set & existing:
                raise ValueError("route overlap")
            for rr, cc in cell_set:
                if (
                    (rr - 1, cc) in existing
                    or (rr + 1, cc) in existing
                    or (rr, cc - 1) in existing
                    or (rr, cc + 1) in existing
                ):
                    raise ValueError("route adjacency")
        segments.append(cell_set)
        all_cells.extend(cells)
    return all_cells, segments


def extract_frame_info(grid: np.ndarray, frame: dict) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    markers: dict[int, tuple[int, int]] = {}
    for color in MARKER_COLORS:
        coords = np.argwhere(inner == color)
        if len(coords) != 1:
            raise ValueError("invalid marker count")
        rr, cc = coords[0]
        markers[color] = (int(rr), int(cc))
    return {"bbox": frame["bbox"], "frame_color": frame["color"], "inner": inner, "markers": markers}


def infer_frame_algorithm(info: dict) -> tuple[str, str]:
    inner = info["inner"]
    markers = info["markers"]
    grey_mask = inner == PATH_COLOR
    grey_comps = components(grey_mask)
    if len(grey_comps) != 2:
        raise ValueError("expected two path components")
    edge_to_comp: dict[frozenset[int], set[tuple[int, int]]] = {}
    for comp in grey_comps:
        comp_set = set(comp)
        touched: set[int] = set()
        for rr, cc in comp:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1]:
                    cell = int(inner[nr, nc])
                    if cell in MARKER_COLORS:
                        touched.add(cell)
        if len(touched) != 2:
            raise ValueError("path must touch exactly two markers")
        edge = frozenset(touched)
        if edge in edge_to_comp:
            raise ValueError("duplicate edge")
        edge_to_comp[edge] = comp_set
    observed_edges = frozenset(edge_to_comp)
    matched_strategies = []
    for strategy in ALL_STRATEGIES:
        if edge_set_for_strategy(markers, strategy) == observed_edges:
            matched_strategies.append(strategy)
    if len(matched_strategies) != 1:
        raise ValueError("ambiguous ordering strategy")
    strategy = matched_strategies[0]
    order = ordered_colors(markers, strategy)
    edge_pairs = [
        (frozenset((order[0], order[1])), markers[order[0]], markers[order[1]]),
        (frozenset((order[1], order[2])), markers[order[1]], markers[order[2]]),
    ]
    matched_modes = []
    for mode in ROUTE_MODES:
        ok = True
        for edge, start, end in edge_pairs:
            if edge_to_comp[edge] != set(route_cells(start, end, mode)):
                ok = False
                break
        if ok:
            matched_modes.append(mode)
    if len(matched_modes) != 1:
        raise ValueError("ambiguous route mode")
    return strategy, matched_modes[0]


def apply_algorithm(grid: np.ndarray, info: dict, strategy: str, mode: str) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = info["bbox"]
    markers = info["markers"]
    cells, _segments = build_paths(markers, strategy, mode)
    for rr, cc in cells:
        gr = r0 + 1 + rr
        gc = c0 + 1 + cc
        if int(out[gr, gc]) == BG:
            out[gr, gc] = PATH_COLOR
    return out


def manual_expected(
    grid: np.ndarray,
    frames_info: list[dict],
    test_idx: int,
    strategy: str,
    mode: str,
) -> np.ndarray:
    return apply_algorithm(grid, frames_info[test_idx], strategy, mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a routing-rule change from one framed training pair and draw the transformed route in the test frame.

    Input structure:
    - Background is black.
    - The grid contains exactly three hollow rectangular frames in distinct frame colors.
    - Each frame interior contains the same three colored marker pixels; two frames also contain grey orthogonal routes.
    - The training pair shares the same marker layout, while the test frame uses a different layout with no grey route.

    Rule:
    - Find the three frames and read the marker coordinates inside each interior.
    - Identify the training pair as the two frames with identical marker placement, then analyze their grey route components.
    - Infer which ordering strategy the route uses to choose the middle marker and whether each segment is horizontal-then-vertical or vertical-then-horizontal.
    - Treat the frame using the canonical source rule (column order + horizontal-then-vertical) as the source example, take the other matched frame as the transformed target rule, and draw that transformed rule in the remaining test frame.

    Key insight:
    - The analogy is between routing algorithms, not pixel shapes: the path teaches a change in "which marker becomes the middle" and "which L-turn orientation is used", and that abstract rule must be reapplied to new marker coordinates.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_frames(g)
    if len(frames) != 3:
        raise ValueError("expected exactly three frames")
    infos = [extract_frame_info(g, frame) for frame in frames]

    sig_to_indices: dict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
    for idx, info in enumerate(infos):
        sig_to_indices[marker_signature(info["markers"])].append(idx)
    pair = next((indices for indices in sig_to_indices.values() if len(indices) == 2), None)
    if pair is None:
        raise ValueError("failed to find training pair")
    train_a, train_b = pair
    test_idx = ({0, 1, 2} - set(pair)).pop()

    algo_a = infer_frame_algorithm(infos[train_a])
    algo_b = infer_frame_algorithm(infos[train_b])
    if algo_a == (SOURCE_STRATEGY, SOURCE_MODE) and algo_b != (SOURCE_STRATEGY, SOURCE_MODE):
        target_strategy, target_mode = algo_b
    elif algo_b == (SOURCE_STRATEGY, SOURCE_MODE) and algo_a != (SOURCE_STRATEGY, SOURCE_MODE):
        target_strategy, target_mode = algo_a
    else:
        raise ValueError("failed to orient source and target frames")

    return apply_algorithm(g, infos[test_idx], target_strategy, target_mode)


def choose_grid_dims(side: int) -> tuple[int, int, list[tuple[int, int]]]:
    for _ in range(200):
        h = random.randint(22, 29)
        w = random.randint(22, 29)
        r1 = random.randint(1, 3)
        c1 = random.randint(1, 3)
        r2_min = r1 + side + 2
        c2_min = c1 + side + 2
        if r2_min > h - side - 1 or c2_min > w - side - 1:
            continue
        r2 = random.randint(r2_min, h - side - 1)
        c2 = random.randint(c2_min, w - side - 1)
        slots = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
        return h, w, slots
    raise RuntimeError("failed to choose frame slots")


def sample_markers(
    inner_size: int,
    target_strategy: str,
    target_mode: str,
    required_sig: tuple[tuple[int, int, int], ...] | None = None,
) -> dict[int, tuple[int, int]] | None:
    for _ in range(3000):
        rows = random.sample(range(inner_size), 3)
        cols = random.sample(range(inner_size), 3)
        col_perm = random.sample((0, 1, 2), 3)
        chosen = [(rows[i], cols[col_perm[i]]) for i in range(3)]
        random.shuffle(chosen)
        markers = {color: pos for color, pos in zip(MARKER_COLORS, chosen)}
        if required_sig is not None and marker_signature(markers) == required_sig:
            continue
        try:
            middle_colors = {strategy: ordered_colors(markers, strategy)[1] for strategy in ALL_STRATEGIES}
            if len(set(middle_colors.values())) != len(ALL_STRATEGIES):
                continue
            source_cells, _ = build_paths(markers, SOURCE_STRATEGY, SOURCE_MODE)
            target_cells, _ = build_paths(markers, target_strategy, target_mode)
            alt_strategy = next(strategy for strategy in TARGET_STRATEGIES if strategy != target_strategy)
            alt_cells, _ = build_paths(markers, alt_strategy, target_mode)
        except ValueError:
            continue
        if set(source_cells) == set(target_cells):
            continue
        if set(alt_cells) == set(target_cells):
            continue
        if ordered_colors(markers, target_strategy)[1] == ordered_colors(markers, SOURCE_STRATEGY)[1]:
            continue
        return markers
    return None


def draw_frame(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def stamp_markers(grid: np.ndarray, top: int, left: int, markers: dict[int, tuple[int, int]]) -> None:
    for color, (rr, cc) in markers.items():
        grid[top + 1 + rr, left + 1 + cc] = color


def stamp_paths(
    grid: np.ndarray,
    top: int,
    left: int,
    markers: dict[int, tuple[int, int]],
    strategy: str,
    mode: str,
) -> None:
    cells, _segments = build_paths(markers, strategy, mode)
    for rr, cc in cells:
        grid[top + 1 + rr, left + 1 + cc] = PATH_COLOR


def build_scene(
    h: int,
    w: int,
    side: int,
    roles: dict[str, tuple[int, int]],
    frame_colors: dict[str, int],
    train_markers: dict[int, tuple[int, int]],
    test_markers: dict[int, tuple[int, int]],
    target_strategy: str,
    target_mode: str,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for role in ("source", "target", "test"):
        top, left = roles[role]
        draw_frame(grid, top, left, side, frame_colors[role])
        if role == "test":
            stamp_markers(grid, top, left, test_markers)
        else:
            stamp_markers(grid, top, left, train_markers)
    source_top, source_left = roles["source"]
    target_top, target_left = roles["target"]
    stamp_paths(grid, source_top, source_left, train_markers, SOURCE_STRATEGY, SOURCE_MODE)
    stamp_paths(grid, target_top, target_left, train_markers, target_strategy, target_mode)
    return grid


def validate_instance(
    inp: np.ndarray,
    target_strategy: str,
    target_mode: str,
) -> bool:
    frames = find_frames(inp)
    if len(frames) != 3:
        return False
    infos = [extract_frame_info(inp, frame) for frame in frames]
    sig_counts = defaultdict(int)
    for info in infos:
        sig_counts[marker_signature(info["markers"])] += 1
    if sorted(sig_counts.values()) != [1, 2]:
        return False
    algos = []
    path_frame_indices = []
    for idx, info in enumerate(infos):
        if np.any(info["inner"] == PATH_COLOR):
            path_frame_indices.append(idx)
            try:
                algos.append(infer_frame_algorithm(info))
            except Exception:
                return False
    if len(path_frame_indices) != 2:
        return False
    if (SOURCE_STRATEGY, SOURCE_MODE) not in algos:
        return False
    if (target_strategy, target_mode) not in algos:
        return False
    expected = main(inp)
    for strategy, mode in ALL_ALGOS:
        try:
            alt = None
            infos2 = [extract_frame_info(inp, frame) for frame in find_frames(inp)]
            sig_to_indices: dict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
            for idx, info in enumerate(infos2):
                sig_to_indices[marker_signature(info["markers"])].append(idx)
            pair = next(indices for indices in sig_to_indices.values() if len(indices) == 2)
            test_idx = ({0, 1, 2} - set(pair)).pop()
            alt = manual_expected(inp, infos2, test_idx, strategy, mode)
        except Exception:
            continue
        if (strategy, mode) != (target_strategy, target_mode) and np.array_equal(alt, expected):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        side = random.choice((8, 9))
        inner_size = side - 2
        target_strategy = random.choice(TARGET_STRATEGIES)
        target_mode = random.choice(ROUTE_MODES)
        train_markers = sample_markers(inner_size, target_strategy, target_mode)
        if train_markers is None:
            continue
        test_markers = sample_markers(
            inner_size,
            target_strategy,
            target_mode,
            required_sig=marker_signature(train_markers),
        )
        if test_markers is None:
            continue
        h, w, slots = choose_grid_dims(side)
        chosen_slots = random.sample(slots, 3)
        random.shuffle(chosen_slots)
        roles = {"source": chosen_slots[0], "target": chosen_slots[1], "test": chosen_slots[2]}
        permuted_colors = random.sample(FRAME_COLORS, 3)
        frame_colors = {role: permuted_colors[i] for i, role in enumerate(("source", "target", "test"))}
        inp = build_scene(h, w, side, roles, frame_colors, train_markers, test_markers, target_strategy, target_mode)
        if inp.shape[0] > 30 or inp.shape[1] > 30:
            continue
        if not validate_instance(inp, target_strategy, target_mode):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
