from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import crop_bbox, find_components, get_bg, in_bounds, paste
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    in_bounds = _ARC_UTILS.in_bounds
    paste = _ARC_UTILS.paste


BG = 0
OBJ_SIZE = 4
CHECKER_SIZE = 3
MAX_RETRIES = 600


def _make_checker(color_a: int, color_b: int) -> np.ndarray:
    patch = np.empty((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            patch[r, c] = color_a if (r + c) % 2 == 0 else color_b
    return patch


def _make_frame(color: int, bg: int) -> np.ndarray:
    patch = np.full((4, 4), color, dtype=int)
    patch[1:3, 1:3] = bg
    return patch


def _make_solid(color: int) -> np.ndarray:
    return np.full((4, 4), color, dtype=int)


def _is_checker(window: np.ndarray, bg: int) -> bool:
    if window.shape != (3, 3) or np.any(window == bg):
        return False
    colors = sorted(set(map(int, window.flat)))
    if len(colors) != 2:
        return False
    a = int(window[0, 0])
    b = int(window[0, 1])
    return all(int(window[r, c]) == (a if (r + c) % 2 == 0 else b) for r in range(3) for c in range(3))


def _touches_bbox(node: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = node
    r0, c0, r1, c1 = bbox
    return ((r == r0 - 1 or r == r1 + 1) and c0 <= c <= c1) or (
        (c == c0 - 1 or c == c1 + 1) and r0 <= r <= r1
    )


def _bbox_with_size(r0: int, c0: int, size: int) -> tuple[int, int, int, int]:
    return (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1))


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _checker_touch_nodes(r0: int, c0: int, path_nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bbox = _bbox_with_size(r0, c0, CHECKER_SIZE)
    return [node for node in path_nodes if _touches_bbox(node, bbox)]


def _checker_candidates(start: tuple[int, int], path_nodes: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    candidates = [
        (r + 1, c - 1),
        (r - 3, c - 1),
        (r - 1, c + 1),
        (r - 1, c - 3),
    ]
    out: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + 3 > h or c0 + 3 > w:
            continue
        bbox = _bbox_with_size(r0, c0, CHECKER_SIZE)
        if any(r0 <= pr <= r0 + 2 and c0 <= pc <= c0 + 2 for pr, pc in path_nodes):
            continue
        if _checker_touch_nodes(r0, c0, path_nodes) != [start]:
            continue
        out.append((r0, c0))
    random.shuffle(out)
    return out


def _object_candidates(
    node: tuple[int, int],
    path_nodes: list[tuple[int, int]],
    shape: tuple[int, int],
    blocked: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    r, c = node
    raw: list[tuple[int, int]] = []
    for offset in (1, 2):
        raw.extend(
            [
                (r + 1, c - offset),
                (r - 4, c - offset),
                (r - offset, c + 1),
                (r - offset, c - 4),
            ]
        )

    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r0, c0 in raw:
        if (r0, c0) in seen:
            continue
        seen.add((r0, c0))
        if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
            continue
        bbox = _bbox_with_size(r0, c0, OBJ_SIZE)
        if any(r0 <= pr <= r0 + 3 and c0 <= pc <= c0 + 3 for pr, pc in path_nodes):
            continue
        if any(not _boxes_separated(bbox, other, pad=1) for other in blocked):
            continue
        touching = [p for p in path_nodes if _touches_bbox(p, bbox)]
        if touching != [node]:
            continue
        out.append(bbox)
    random.shuffle(out)
    return out


def _path_neighbors(node: tuple[int, int], node_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = node
    neighbors = []
    for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        nxt = (r + dr, c + dc)
        if nxt in node_set:
            neighbors.append(nxt)
    return neighbors


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _generate_path(shape: tuple[int, int], length: int) -> list[tuple[int, int]] | None:
    h, w = shape
    rows = list(range(5, h - 4, 2))
    cols = list(range(5, w - 4, 2))
    if len(rows) * len(cols) < length:
        return None

    min_turns = random.randint(2, max(2, min(4, length - 2)))
    starts = [(r, c) for r in rows for c in cols]
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]], seen: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            used_axes = {0 if path[i][0] != path[i - 1][0] else 1 for i in range(1, len(path))}
            if len(used_axes) == 2 and _turn_count(path) >= min_turns:
                return list(path)
            return None

        r, c = path[-1]
        candidates = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in seen or nxt[0] not in rows or nxt[1] not in cols:
                continue
            candidates.append(nxt)
        random.shuffle(candidates)
        candidates.sort(
            key=lambda node: sum(
                (
                    (node[0] + dr, node[1] + dc) in {(rr, cc) for rr in rows for cc in cols}
                    and (node[0] + dr, node[1] + dc) not in seen
                )
                for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
            ),
            reverse=True,
        )
        for nxt in candidates:
            extra_neighbors = [
                node
                for node in seen
                if node != path[-1]
                and ((node[0] == nxt[0] and abs(node[1] - nxt[1]) == 2) or (node[1] == nxt[1] and abs(node[0] - nxt[0]) == 2))
            ]
            if extra_neighbors:
                continue
            path.append(nxt)
            seen.add(nxt)
            res = dfs(path, seen)
            if res is not None:
                return res
            seen.remove(nxt)
            path.pop()
        return None

    for start in starts:
        res = dfs([start], {start})
        if res is not None:
            return res
    return None


def _follow_path(start: tuple[int, int], nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    degrees = {node: len(_path_neighbors(node, nodes)) for node in nodes}
    assert all(deg <= 2 for deg in degrees.values()), "guide path must be unbranched"
    assert degrees[start] == 1, "start guide pixel must be a path endpoint"
    endpoints = [node for node, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "guide path must have exactly two endpoints"

    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [node for node in _path_neighbors(cur, nodes) if node != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "guide path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(nodes), "path traversal must cover all guide pixels"
    return order


def _detect_checker(grid: np.ndarray, bg: int) -> tuple[int, int]:
    found: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if _is_checker(grid[r : r + 3, c : c + 3], bg):
                found.append((r, c))
    assert len(found) == 1, "expected exactly one 3x3 checkerboard"
    return found[0]


def _detect_objects(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    objects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue
        color = int(comp["color"])
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
        if not np.all(border == color):
            continue
        if np.all(patch[1:3, 1:3] == bg):
            kind = "frame"
        elif np.all(patch[1:3, 1:3] == color):
            kind = "solid"
        else:
            continue
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "kind": kind,
                "color": color,
                "top_left": (r0, c0),
            }
        )
    assert objects, "expected at least one 4x4 object"
    return objects


def _checker_adjacent_pixels(
    grid: np.ndarray, checker: tuple[int, int], object_mask: np.ndarray, bg: int
) -> list[tuple[int, int, int]]:
    r0, c0 = checker
    hits = set()
    for r in range(r0, r0 + 3):
        for c in range(c0, c0 + 3):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not in_bounds(grid.shape, nr, nc):
                    continue
                if r0 <= nr <= r0 + 2 and c0 <= nc <= c0 + 2:
                    continue
                if object_mask[nr, nc] or int(grid[nr, nc]) == bg:
                    continue
                hits.add((int(nr), int(nc), int(grid[nr, nc])))
    return sorted(hits)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    checker = _detect_checker(g, bg)
    objects = _detect_objects(g, bg)

    object_mask = np.zeros(g.shape, dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        object_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    start_hits = _checker_adjacent_pixels(g, checker, object_mask, bg)
    assert len(start_hits) == 1, "checkerboard must have exactly one adjacent guide pixel"
    start_r, start_c, path_color = start_hits[0]

    checker_mask = np.zeros(g.shape, dtype=bool)
    checker_mask[checker[0] : checker[0] + 3, checker[1] : checker[1] + 3] = True
    path_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere((g == path_color) & ~object_mask & ~checker_mask)
    }
    assert (start_r, start_c) in path_nodes, "adjacent guide pixel must belong to the path"
    path_order = _follow_path((start_r, start_c), path_nodes)

    ordered = []
    for obj in objects:
        hits = [idx for idx, node in enumerate(path_order) if _touches_bbox(node, obj["bbox"])]
        assert hits, "each object must be touched by the path"
        item = dict(obj)
        item["touch_index"] = int(min(hits))
        ordered.append(item)
    ordered.sort(key=lambda item: (item["touch_index"], item["top_left"][0], item["top_left"][1]))

    return {
        "bg": bg,
        "checker": checker,
        "path_color": path_color,
        "path_order": path_order,
        "objects": objects,
        "ordered": ordered,
    }


def _build_output(ordered: list[dict], bg: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    solids: list[np.ndarray] = []
    n = len(ordered)
    for i, obj in enumerate(ordered):
        patch = obj["patch"].copy()
        if obj["kind"] == "frame":
            patch[1:3, 1:3] = int(ordered[(i + 1) % n]["color"])
            frames.append(patch)
        else:
            solids.append(patch)
    out = np.full((4 * max(len(frames), len(solids)), 8), int(bg), dtype=int)
    for i, patch in enumerate(frames):
        paste(out, patch, 4 * i, 0)
    for i, patch in enumerate(solids):
        paste(out, patch, 4 * i, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the guide path from the checkerboard, then stack frames and solids by visit order while recoloring each frame with its successor's color.

    Input structure:
    - The background is the dominant color. Several isolated 4×4 single-color objects appear: hollow frames with a 2×2 background interior, and solid 4×4 blocks.
    - Exactly one 3×3 two-color checkerboard marks the path start, and exactly one guide pixel touches that checkerboard.
    - All guide pixels of that guide color lie on a two-cell orthogonal lattice and form one unbranched path that touches every 4×4 object.

    Rule:
    - Detect the checkerboard, find its unique adjacent guide pixel, and traverse the same-color guide path by horizontal or vertical steps of size 2.
    - For each 4×4 object, record the first path position that is 4-neighbor adjacent to the object's bounding box; this first-touch order is the visit order.
    - Output width is 8. Stack visited frames top-to-bottom in columns 0-3 and visited solids top-to-bottom in columns 4-7. Each output frame keeps its border color but fills its 2×2 interior with the color of the next visited object, wrapping from the last visited object back to the first.

    Key insight:
    - The same traversal order drives both the stacking order and the frame interior recoloring, so row position alone cannot explain the output.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _build_output(scene["ordered"], scene["bg"])


def _choose_type_order(n_frames: int, n_solids: int) -> list[str]:
    want_last_frame = random.random() < 0.5
    for _ in range(200):
        types = ["frame"] * n_frames + ["solid"] * n_solids
        random.shuffle(types)
        if want_last_frame and types[-1] != "frame":
            continue
        if not want_last_frame and types[-1] != "solid":
            continue
        if not any(types[i] != types[i + 1] for i in range(len(types) - 1)):
            continue
        if not any(types[i] == "frame" and types[(i + 1) % len(types)] == "solid" for i in range(len(types))):
            continue
        return types
    raise RuntimeError("failed to build a non-degenerate type order")


def _place_boxes(
    path: list[tuple[int, int]], shape: tuple[int, int], count: int
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    checker_candidates = _checker_candidates(path[0], path, shape)
    for checker_r0, checker_c0 in checker_candidates:
        checker_bbox = _bbox_with_size(checker_r0, checker_c0, CHECKER_SIZE)
        node_infos: list[tuple[int, tuple[int, int], list[tuple[int, int, int, int]]]] = []
        for index, node in enumerate(path[1:], start=1):
            candidates = _object_candidates(node, path, shape, [checker_bbox])
            if candidates:
                node_infos.append((index, node, candidates))
        if len(node_infos) < count:
            continue

        def dfs(
            pos: int,
            blocked: list[tuple[int, int, int, int]],
            chosen_nodes: list[tuple[int, int]],
            placed: list[tuple[int, int, int, int]],
        ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
            if len(chosen_nodes) == count:
                return list(chosen_nodes), list(placed)
            need = count - len(chosen_nodes)
            if len(node_infos) - pos < need:
                return None
            for j in range(pos, len(node_infos)):
                _index, node, candidates = node_infos[j]
                candidate_boxes = list(candidates)
                random.shuffle(candidate_boxes)
                for bbox in candidate_boxes:
                    if any(not _boxes_separated(bbox, other, pad=1) for other in blocked):
                        continue
                    blocked.append(bbox)
                    chosen_nodes.append(node)
                    placed.append(bbox)
                    res = dfs(j + 1, blocked, chosen_nodes, placed)
                    if res is not None:
                        return res
                    placed.pop()
                    chosen_nodes.pop()
                    blocked.pop()
            return None

        selection = dfs(0, [checker_bbox], [], [])
        if selection is not None:
            nodes, object_bboxes = selection
            return (checker_r0, checker_c0), nodes, object_bboxes
    return None


def _semantic_ok(types: list[str], bboxes: list[tuple[int, int, int, int]]) -> bool:
    order_by_row = sorted(range(len(bboxes)), key=lambda i: (bboxes[i][0], bboxes[i][1]))
    frame_path = [i for i, kind in enumerate(types) if kind == "frame"]
    solid_path = [i for i, kind in enumerate(types) if kind == "solid"]
    frame_rows = [i for i in order_by_row if types[i] == "frame"]
    solid_rows = [i for i in order_by_row if types[i] == "solid"]
    path_matters = (len(frame_path) >= 2 and frame_path != frame_rows) or (
        len(solid_path) >= 2 and solid_path != solid_rows
    )
    fill_matters = any(types[i] == "frame" and types[(i + 1) % len(types)] == "solid" for i in range(len(types)))
    return path_matters and fill_matters


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 29)
    w = random.randint(22, 29)
    n_frames = random.randint(2, 3)
    n_solids = random.randint(2, 3)
    types = _choose_type_order(n_frames, n_solids)
    count = len(types)

    max_len = len(range(5, h - 4, 2)) * len(range(5, w - 4, 2))
    path_len = random.randint(max(10, count + 4), min(16, max_len))
    path = _generate_path((h, w), path_len)
    if path is None:
        return None

    placement = _place_boxes(path, (h, w), count)
    if placement is None:
        return None
    checker_pos, object_nodes, bboxes = placement
    if not _semantic_ok(types, bboxes):
        return None

    palette = random.sample(range(1, 10), count + 3)
    random.shuffle(palette)
    object_colors = palette[:count]
    path_color = palette[count]
    checker_colors = palette[count + 1 : count + 3]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in path:
        grid[r, c] = path_color

    paste(grid, _make_checker(checker_colors[0], checker_colors[1]), checker_pos[0], checker_pos[1])

    for bbox, kind, color in zip(bboxes, types, object_colors, strict=False):
        patch = _make_frame(color, BG) if kind == "frame" else _make_solid(color)
        paste(grid, patch, bbox[0], bbox[1])

    scene = _parse_scene(grid)
    parsed_types = [obj["kind"] for obj in scene["ordered"]]
    parsed_colors = [obj["color"] for obj in scene["ordered"]]
    if parsed_types != types or parsed_colors != object_colors:
        return None
    if len(scene["path_order"]) != len(path):
        return None
    parsed_touch_nodes = [scene["path_order"][obj["touch_index"]] for obj in scene["ordered"]]
    if parsed_touch_nodes != object_nodes:
        return None
    if not _semantic_ok(parsed_types, [obj["bbox"] for obj in scene["ordered"]]):
        return None
    if not np.array_equal(main(grid), _build_output(scene["ordered"], scene["bg"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid巡径点名 input")
