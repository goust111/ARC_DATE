from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils_6", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

crop_bbox = _ARC_UTILS.crop_bbox
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EVEN_MASK = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
ODD_MASK = ~EVEN_MASK

GRID_RANGE = (18, 24)
PATH_LEN_RANGE = (4, 8)
RING_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the checkerboard-colored guide path to one ring, fill that ring's cavity with the checkerboard's other color, and crop the ring.

    Input structure:
    - A mostly black grid contains one 3x3 two-color checkerboard and three or four 3x3 monochrome rings.
    - Exactly one colored guide pixel sits 4-neighbor adjacent to the checkerboard, and more guide pixels of that same color continue as a sparse path with one black gap between consecutive guide pixels.
    - The path ends beside exactly one ring; the other rings are separate distractors.

    Rule:
    - Detect the unique 3x3 checkerboard and read the color of its unique adjacent guide pixel.
    - Trace the guide path by moving between same-colored guide pixels that are two cells apart in a cardinal direction with a black midpoint.
    - Identify the unique ring adjacent to the path terminal.
    - Take the checkerboard color that is not the guide color and write it into the target ring's black center.
    - Return only the target ring's 3x3 crop after filling.

    Key insight:
    - The path color chooses which ring to crop, but the fill color is the checkerboard's other color rather than the path color or the ring color.
    """
    scene = _solve_instance(np.asarray(grid, dtype=int))
    out = crop_bbox(scene["grid"], scene["target_bbox"])
    out[1, 1] = scene["fill_color"]
    return out


def _checkerboard_patch(even_color: int, odd_color: int) -> np.ndarray:
    patch = np.empty((3, 3), dtype=int)
    patch[EVEN_MASK] = int(even_color)
    patch[ODD_MASK] = int(odd_color)
    return patch


def _ring_patch(color: int) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    patch[1, 1] = BG
    return patch


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_bbox_cells(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells = set()
    for r in range(r0 - pad, r1 + pad + 1):
        for c in range(c0 - pad, c1 + pad + 1):
            if _in_bounds(shape, r, c):
                cells.add((r, c))
    return cells


def _adjacent_colored_pixels(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> set[tuple[int, int]]:
    box = _bbox_cells(bbox)
    out: set[tuple[int, int]] = set()
    for r, c in box:
        for nr, nc in neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in box and int(grid[nr, nc]) != int(bg):
                out.add((int(nr), int(nc)))
    return out


def _jump_neighbors(grid: np.ndarray, coord: tuple[int, int], color: int, bg: int) -> list[tuple[int, int]]:
    r, c = map(int, coord)
    out = []
    for dr, dc in DIRS4:
        mr, mc = r + dr, c + dc
        nr, nc = r + 2 * dr, c + 2 * dc
        if not _in_bounds(grid.shape, nr, nc):
            continue
        if int(grid[mr, mc]) == int(bg) and int(grid[nr, nc]) == int(color):
            out.append((int(nr), int(nc)))
    return out


def _trace_path(grid: np.ndarray, start: tuple[int, int], color: int, bg: int) -> list[tuple[int, int]]:
    path = [tuple(map(int, start))]
    visited = {path[0]}
    prev = None
    cur = path[0]
    while True:
        neighbors = _jump_neighbors(grid, cur, color, bg)
        if prev is None:
            nxt = [p for p in neighbors if p not in visited]
            if not nxt:
                return path
            if len(nxt) != 1:
                raise AssertionError("guide path branches at the start")
        else:
            non_prev = [p for p in neighbors if p != prev]
            if len(non_prev) > 1:
                raise AssertionError("guide path branches")
            nxt = [p for p in non_prev if p not in visited]
            if not nxt:
                return path
            if len(nxt) != 1:
                raise AssertionError("guide path revisits a node")
        prev, cur = cur, nxt[0]
        visited.add(cur)
        path.append(cur)


def _scan_checkerboards(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    found = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(patch == int(bg)):
                continue
            even_vals = patch[EVEN_MASK]
            odd_vals = patch[ODD_MASK]
            if len(set(map(int, even_vals))) != 1 or len(set(map(int, odd_vals))) != 1:
                continue
            even_color = int(even_vals[0])
            odd_color = int(odd_vals[0])
            if even_color == odd_color:
                continue
            found.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
                    "patch": patch.copy(),
                    "even_color": even_color,
                    "odd_color": odd_color,
                }
            )
    return found


def _scan_rings(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    found = []
    border_mask = np.ones((3, 3), dtype=bool)
    border_mask[1, 1] = False
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != int(bg):
                continue
            border_vals = patch[border_mask]
            if len(set(map(int, border_vals))) != 1:
                continue
            color = int(border_vals[0])
            if color == int(bg):
                continue
            found.append({"bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)), "color": color})
    return found


def _coord_adjacent_to_ring(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    rr, cc = map(int, coord)
    r0, c0, _r1, _c1 = map(int, bbox)
    for lr in range(3):
        for lc in range(3):
            if (lr, lc) == (1, 1):
                continue
            if abs(rr - (r0 + lr)) + abs(cc - (c0 + lc)) == 1:
                return True
    return False


def _solve_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    checkers = _scan_checkerboards(g, bg)
    assert len(checkers) == 1, f"expected one checkerboard, found {len(checkers)}"
    checker = checkers[0]

    rings = _scan_rings(g, bg)
    assert 3 <= len(rings) <= 4, f"expected 3 or 4 rings, found {len(rings)}"

    guide_starts = _adjacent_colored_pixels(g, checker["bbox"], bg)
    assert len(guide_starts) == 1, f"expected one adjacent guide pixel, found {len(guide_starts)}"
    start = next(iter(guide_starts))
    guide_color = int(g[start])

    checker_colors = {checker["even_color"], checker["odd_color"]}
    assert guide_color in checker_colors, "guide color must come from the checkerboard"
    fill_color = next(color for color in checker_colors if int(color) != int(guide_color))

    path = _trace_path(g, start, guide_color, bg)
    terminal = path[-1]
    target_rings = [ring for ring in rings if _coord_adjacent_to_ring(terminal, ring["bbox"])]
    assert len(target_rings) == 1, f"expected one target ring, found {len(target_rings)}"

    return {
        "grid": g,
        "bg": bg,
        "checker": checker,
        "rings": rings,
        "start": start,
        "guide_color": guide_color,
        "fill_color": fill_color,
        "path": path,
        "terminal": terminal,
        "target_bbox": target_rings[0]["bbox"],
        "target_color": int(target_rings[0]["color"]),
    }


def _path_degrees(path: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {node: 0 for node in path}
    for i, a in enumerate(path):
        for b in path[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2 and (a[0] == b[0] or a[1] == b[1]):
                deg[a] += 1
                deg[b] += 1
    return deg


def _path_corridor(path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    corridor = set(path)
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        corridor.add(((r0 + r1) // 2, (c0 + c1) // 2))
    return corridor


def _placement_options(anchor: tuple[int, int]) -> list[dict]:
    r, c = map(int, anchor)
    out = []
    for offset in range(3):
        out.append({"bbox": (r + 1, c - offset, r + 3, c - offset + 2), "contact": (0, offset)})
        out.append({"bbox": (r - 3, c - offset, r - 1, c - offset + 2), "contact": (2, offset)})
        out.append({"bbox": (r - offset, c + 1, r - offset + 2, c + 3), "contact": (offset, 0)})
        out.append({"bbox": (r - offset, c - 3, r - offset + 2, c - 1), "contact": (offset, 2)})
    return out


def _bbox_fits(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return 0 <= r0 and 0 <= c0 and r1 < int(shape[0]) and c1 < int(shape[1])


def _adjacent_path_nodes(path: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    box = _bbox_cells(bbox)
    out = set()
    for node in path:
        for nr, nc in neighbors4(*node):
            if (nr, nc) in box:
                out.add(node)
                break
    return out


def _choose_endpoint_bbox(
    path: list[tuple[int, int]],
    anchor: tuple[int, int],
    shape: tuple[int, int],
    corridor: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    options = _placement_options(anchor)
    rng.shuffle(options)
    for option in options:
        bbox = option["bbox"]
        if not _bbox_fits(bbox, shape):
            continue
        if _bbox_cells(bbox) & corridor:
            continue
        if _adjacent_path_nodes(path, bbox) != {anchor}:
            continue
        return option
    return None


def _random_path(shape: tuple[int, int], rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    for _ in range(200):
        target_len = rng.randint(*PATH_LEN_RANGE)
        start = (rng.randint(3, h - 4), rng.randint(3, w - 4))
        path = [start]
        visited = {start}
        dirs = []
        while len(path) < target_len:
            cur = path[-1]
            options = []
            for dr, dc in DIRS4:
                nxt = (cur[0] + 2 * dr, cur[1] + 2 * dc)
                if not (3 <= nxt[0] <= h - 4 and 3 <= nxt[1] <= w - 4):
                    continue
                if nxt in visited:
                    continue
                if any(
                    old != cur
                    and abs(old[0] - nxt[0]) + abs(old[1] - nxt[1]) == 2
                    and (old[0] == nxt[0] or old[1] == nxt[1])
                    for old in visited
                ):
                    continue
                weight = 2
                if dirs and (dr, dc) != dirs[-1]:
                    weight += 2
                margin = min(nxt[0], nxt[1], h - 1 - nxt[0], w - 1 - nxt[1])
                if margin >= 5:
                    weight += 1
                options.extend([((nxt[0], nxt[1]), (dr, dc))] * weight)
            if not options:
                break
            nxt, step_dir = rng.choice(options)
            path.append(nxt)
            visited.add(nxt)
            dirs.append(step_dir)
        if len(path) != target_len:
            continue
        degrees = sorted(_path_degrees(path).values())
        if degrees != [1, 1] + [2] * (len(path) - 2):
            continue
        if len(set(dirs)) == 1 and rng.random() < 0.75:
            continue
        return path
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _place_distractor_rings(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    corridor: set[tuple[int, int]],
    object_bboxes: list[tuple[int, int, int, int]],
    checker_center: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    n_rings: int,
    colors: list[int],
    rank_mode: str,
    rng: random.Random,
) -> list[tuple[tuple[int, int, int, int], int]] | None:
    target_center = (target_bbox[0] + 1, target_bbox[1] + 1)
    target_dist = _manhattan(checker_center, target_center)
    placed: list[tuple[tuple[int, int, int, int], int]] = [(target_bbox, int(colors[0]))]
    guard = set()
    for bbox in object_bboxes:
        guard |= _expanded_bbox_cells(bbox, shape, pad=1)

    for color in colors[1:]:
        choices = []
        for r0 in range(shape[0] - 2):
            for c0 in range(shape[1] - 2):
                bbox = (r0, c0, r0 + 2, c0 + 2)
                cells = _bbox_cells(bbox)
                if cells & corridor:
                    continue
                if _adjacent_path_nodes(path, bbox):
                    continue
                halo = _expanded_bbox_cells(bbox, shape, pad=1)
                if halo & guard:
                    continue
                dist = _manhattan(checker_center, (r0 + 1, c0 + 1))
                choices.append((bbox, dist))
        rng.shuffle(choices)
        picked = None
        used_dists = {target_dist}
        used_dists.update(_manhattan(checker_center, (bbox[0] + 1, bbox[1] + 1)) for bbox, _ in placed[1:])
        for bbox, dist in choices:
            if dist in used_dists:
                continue
            if rank_mode == "closest" and dist <= target_dist:
                continue
            if rank_mode == "farthest" and dist >= target_dist:
                continue
            picked = (bbox, int(color))
            break
        if picked is None and rank_mode == "middle":
            closer_needed = all(_manhattan(checker_center, (bbox[0] + 1, bbox[1] + 1)) > target_dist for bbox, _ in placed[1:])
            for bbox, dist in choices:
                if dist in used_dists:
                    continue
                if closer_needed and dist < target_dist:
                    picked = (bbox, int(color))
                    break
                if not closer_needed and dist > target_dist:
                    picked = (bbox, int(color))
                    break
        if picked is None:
            return None
        placed.append(picked)
        guard |= _expanded_bbox_cells(picked[0], shape, pad=1)

    dist_values = [_manhattan(checker_center, (bbox[0] + 1, bbox[1] + 1)) for bbox, _ in placed]
    if len(set(dist_values)) != len(dist_values):
        return None
    target_dist = dist_values[0]
    if rank_mode == "closest" and not all(dist > target_dist for dist in dist_values[1:]):
        return None
    if rank_mode == "farthest" and not all(dist < target_dist for dist in dist_values[1:]):
        return None
    if rank_mode == "middle":
        if not any(dist < target_dist for dist in dist_values[1:]):
            return None
        if not any(dist > target_dist for dist in dist_values[1:]):
            return None
    return placed


def _build_scene(shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, dict] | None:
    path = _random_path(shape, rng)
    if path is None:
        return None
    corridor = _path_corridor(path)

    checker_place = _choose_endpoint_bbox(path, path[0], shape, corridor, rng)
    if checker_place is None:
        return None
    target_place = _choose_endpoint_bbox(path, path[-1], shape, corridor, rng)
    if target_place is None:
        return None

    n_rings = rng.choice(RING_COUNT_CHOICES)
    checker_colors = list(rng.sample(PALETTE, 2))
    guide_color = int(rng.choice(checker_colors))
    fill_color = next(color for color in checker_colors if int(color) != int(guide_color))
    ring_colors = list(rng.sample([color for color in PALETTE if color not in checker_colors], n_rings))

    contact = checker_place["contact"]
    if (contact[0] + contact[1]) % 2 == 0:
        checker_patch = _checkerboard_patch(guide_color, fill_color)
    else:
        checker_patch = _checkerboard_patch(fill_color, guide_color)

    checker_bbox = checker_place["bbox"]
    target_bbox = target_place["bbox"]
    checker_center = (checker_bbox[0] + 1, checker_bbox[1] + 1)
    rank_mode = rng.choice(["closest", "middle", "farthest"])
    placed_rings = _place_distractor_rings(
        shape,
        path,
        corridor,
        [checker_bbox, target_bbox],
        checker_center,
        target_bbox,
        n_rings,
        ring_colors,
        rank_mode,
        rng,
    )
    if placed_rings is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = checker_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = checker_patch
    for r, c in path:
        grid[r, c] = int(guide_color)
    for bbox, color in placed_rings:
        rr0, cc0, rr1, cc1 = bbox
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = _ring_patch(color)

    info = {
        "path": path,
        "checker_bbox": checker_bbox,
        "target_bbox": target_bbox,
        "guide_color": int(guide_color),
        "fill_color": int(fill_color),
        "target_color": int(ring_colors[0]),
        "ring_bboxes": [bbox for bbox, _ in placed_rings],
        "ring_count": int(n_rings),
        "rank_mode": rank_mode,
        "guide_uses_center_color": bool(checker_patch[1, 1] == guide_color),
    }
    return grid, info


def _validate_scene(grid: np.ndarray, info: dict) -> bool:
    try:
        scene = _solve_instance(grid)
    except Exception:
        return False

    if tuple(scene["start"]) != tuple(info["path"][0]):
        return False
    if list(scene["path"]) != list(info["path"]):
        return False
    if tuple(scene["target_bbox"]) != tuple(info["target_bbox"]):
        return False
    if int(scene["guide_color"]) != int(info["guide_color"]):
        return False
    if int(scene["fill_color"]) != int(info["fill_color"]):
        return False
    if len(scene["rings"]) != int(info["ring_count"]):
        return False

    degrees = sorted(_path_degrees(scene["path"]).values())
    if degrees != [1, 1] + [2] * (len(scene["path"]) - 2):
        return False

    out = main(grid)
    expected = _ring_patch(info["target_color"])
    expected[1, 1] = int(info["fill_color"])
    if not np.array_equal(out, expected):
        return False

    checker_adj = _adjacent_colored_pixels(grid, info["checker_bbox"], BG)
    if checker_adj != {tuple(info["path"][0])}:
        return False

    for bbox in info["ring_bboxes"]:
        touching = {node for node in info["path"] if _coord_adjacent_to_ring(node, bbox)}
        if bbox == info["target_bbox"]:
            if touching != {tuple(info["path"][-1])}:
                return False
        elif touching:
            return False

    checker_center = (info["checker_bbox"][0] + 1, info["checker_bbox"][1] + 1)
    distances = [_manhattan(checker_center, (bbox[0] + 1, bbox[1] + 1)) for bbox in info["ring_bboxes"]]
    if len(set(distances)) != len(distances):
        return False
    target_dist = distances[0]
    if info["rank_mode"] == "closest" and not all(dist > target_dist for dist in distances[1:]):
        return False
    if info["rank_mode"] == "farthest" and not all(dist < target_dist for dist in distances[1:]):
        return False
    if info["rank_mode"] == "middle":
        if not any(dist < target_dist for dist in distances[1:]):
            return False
        if not any(dist > target_dist for dist in distances[1:]):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        shape = (rng.randint(*GRID_RANGE), rng.randint(*GRID_RANGE))
        built = _build_scene(shape, rng)
        if built is None:
            continue
        grid, info = built
        if _validate_scene(grid, info):
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-path-ring scene")
