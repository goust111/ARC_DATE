from __future__ import annotations

from collections import deque
import random

import numpy as np


_N8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_CONTENT_COUNT_OPTIONS = (1, 2, 3)
_MAX_GENERATION_ATTEMPTS = 400


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _N8:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            objects.append(
                {
                    "id": len(objects),
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )

    return objects


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _analyze_layout(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, list[dict]]]:
    objects = _find_objects(grid)
    assert objects, "expected at least one foreground object"

    by_id = {obj["id"]: obj for obj in objects}
    parent_candidates: dict[int, list[int]] = {obj["id"]: [] for obj in objects}
    children: dict[int, list[int]] = {obj["id"]: [] for obj in objects}

    for outer in objects:
        for inner in objects:
            if outer["id"] == inner["id"]:
                continue
            if _contains_bbox(outer["bbox"], inner["bbox"]):
                parent_candidates[inner["id"]].append(int(outer["id"]))
                children[outer["id"]].append(int(inner["id"]))

    resolved_parent: dict[int, int] = {}
    for child_id, parents in parent_candidates.items():
        if not parents:
            continue
        ordered = sorted(parents, key=lambda pid: int(by_id[pid]["area"]))
        chosen = int(ordered[0])
        for other in ordered[1:]:
            assert _contains_bbox(by_id[other]["bbox"], by_id[chosen]["bbox"]), (
                "a content object cannot belong to multiple unrelated containers"
            )
        resolved_parent[child_id] = chosen

    container_ids = sorted(
        [obj["id"] for obj in objects if children[obj["id"]]],
        key=lambda oid: (by_id[oid]["bbox"][0], by_id[oid]["bbox"][1], by_id[oid]["area"]),
    )
    assert container_ids, "expected at least one container"

    contents_by_container: dict[int, list[dict]] = {cid: [] for cid in container_ids}
    for obj in objects:
        oid = int(obj["id"])
        if oid not in resolved_parent:
            continue
        assert oid not in contents_by_container, "nested containers are not supported by this task"
        parent_id = int(resolved_parent[oid])
        assert parent_id in contents_by_container, "every contained object must have a container"
        contents_by_container[parent_id].append(obj)

    containers = [by_id[cid] for cid in container_ids]
    for container in containers:
        cid = int(container["id"])
        contents = contents_by_container[cid]
        assert 1 <= len(contents) <= 3, "each container must have 1-3 contents"
        areas = [int(item["area"]) for item in contents]
        assert len(areas) == len(set(areas)), "content areas must define a unique order"

    return objects, containers, contents_by_container


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack each container's contents at the bottom of its bbox after sorting by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _objects, containers, contents_by_container = _analyze_layout(g)
    out = g.copy()

    for contents in contents_by_container.values():
        for item in contents:
            coords = item["coords"]
            out[coords[:, 0], coords[:, 1]] = 0

    for container in containers:
        r0, c0, r1, c1 = map(int, container["bbox"])
        box_width = int(c1 - c0 + 1)
        contents = sorted(
            contents_by_container[int(container["id"])],
            key=lambda item: (-int(item["area"]), item["bbox"][0], item["bbox"][1]),
        )
        assert len(contents) <= int(container["height"]) - 1, "stack must fit below the container roof"

        for level, item in enumerate(contents):
            bar_row = int(r1 - level)
            assert bar_row > r0, "content bars must stay below the top boundary"
            bar_width = int(item["width"])
            assert bar_width <= box_width - 2, "bar must fit between the container walls"
            bar_col = int(c0 + (box_width - bar_width) // 2)
            segment = out[bar_row, bar_col : bar_col + bar_width]
            assert bool(np.all(segment == 0)), "stacking must not overwrite the container itself"
            out[bar_row, bar_col : bar_col + bar_width] = int(item["color"])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    norm = sorted((int(r - r0), int(c - c0)) for r, c in cells)
    return np.array(norm, dtype=int)


def _grow_shape(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for dr, dc in _N8:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    return _normalize_cells(cells)


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    kinds = ["rect", "grow"]
    if max_h >= 2 and max_w >= 2:
        kinds.append("l")
    kind = random.choice(kinds)

    if kind == "rect":
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        return np.array([(r, c) for r in range(h) for c in range(w)], dtype=int)

    if kind == "l":
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        orientation = random.randrange(4)
        cells: set[tuple[int, int]] = set()
        if orientation == 0:
            cells |= {(r, 0) for r in range(h)}
            cells |= {(h - 1, c) for c in range(w)}
        elif orientation == 1:
            cells |= {(r, w - 1) for r in range(h)}
            cells |= {(h - 1, c) for c in range(w)}
        elif orientation == 2:
            cells |= {(r, 0) for r in range(h)}
            cells |= {(0, c) for c in range(w)}
        else:
            cells |= {(r, w - 1) for r in range(h)}
            cells |= {(0, c) for c in range(w)}
        return _normalize_cells(cells)

    feasible = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= 2]
    if not feasible:
        return None
    height, width = random.choice(feasible)
    area = random.randint(2, height * width)
    return _grow_shape(height, width, area)


def _bbox_of_cells(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max() + 1), int(coords[:, 1].max() + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _can_place_cells(patch: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, patch.shape)
    occupied = patch != 0
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _build_container_patch() -> np.ndarray | None:
    n_contents = random.choice(_CONTENT_COUNT_OPTIONS)

    for _ in range(120):
        top_thickness = random.choice((1, 2))
        min_h = max(6, top_thickness + n_contents + 2)
        h = random.randint(min_h, 10)
        w = random.randint(8, 13)
        max_shape_h = min(4, h - top_thickness - 2)
        max_shape_w = min(5, w - 4)
        if max_shape_h < 1 or max_shape_w < 1:
            continue

        patch = np.zeros((h, w), dtype=int)
        container_color = random.randint(1, 9)
        patch[:top_thickness, :] = container_color
        patch[:, 0] = container_color
        patch[:, -1] = container_color

        colors = random.sample([c for c in range(1, 10) if c != container_color], n_contents)
        shapes: list[np.ndarray] = []
        used_areas: set[int] = set()

        for _shape_attempt in range(200):
            if len(shapes) == n_contents:
                break
            shape = _random_shape(max_shape_h, max_shape_w)
            if shape is None:
                continue
            area = int(shape.shape[0])
            if area in used_areas:
                continue
            shapes.append(shape)
            used_areas.add(area)

        if len(shapes) != n_contents:
            continue

        placed_boxes: list[tuple[int, int, int, int]] = []
        placed = True
        for color, shape in zip(colors, sorted(shapes, key=lambda arr: -int(arr.shape[0])), strict=False):
            sh, sw = _bbox_of_cells(shape)
            row_positions = list(range(top_thickness + 1, h - sh))
            col_positions = list(range(1, w - sw))
            if not row_positions or not col_positions:
                placed = False
                break

            candidates = [(r, c) for r in row_positions for c in col_positions]
            random.shuffle(candidates)
            wrote = False
            for r, c in candidates:
                box = (int(r), int(c), int(r + sh - 1), int(c + sw - 1))
                if any(_boxes_overlap(box, other, margin=0) for other in placed_boxes):
                    continue
                cells = [(int(r + rr), int(c + cc)) for rr, cc in shape.tolist()]
                if not _can_place_cells(patch, cells):
                    continue
                for rr, cc in cells:
                    patch[rr, cc] = int(color)
                placed_boxes.append(box)
                wrote = True
                break

            if not wrote:
                placed = False
                break

        if placed:
            return patch

    return None


def _build_distractor_patch() -> np.ndarray | None:
    for _ in range(60):
        shape = _random_shape(random.randint(1, 4), random.randint(1, 5))
        if shape is None:
            continue
        h, w = _bbox_of_cells(shape)
        patch = np.zeros((h, w), dtype=int)
        color = random.randint(1, 9)
        for r, c in shape.tolist():
            patch[int(r), int(c)] = color
        return patch
    return None


def _place_top_level_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)

    min_side = max(16, max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 2.5))))
    max_side = max(min_side, 30)
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    grid = np.zeros((h, w), dtype=int)

    placements: list[tuple[int, int, int, int]] = []
    ordered = sorted(patches, key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True)
    for patch in ordered:
        ph, pw = map(int, patch.shape)
        if ph + 2 > h or pw + 2 > w:
            return None
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
        ]
        random.shuffle(candidates)

        placed = False
        for r, c in candidates:
            box = (int(r), int(c), int(r + ph - 1), int(c + pw - 1))
            if any(_boxes_overlap(box, other, margin=1) for other in placements):
                continue
            placements.append(box)
            mask = patch != 0
            grid[r : r + ph, c : c + pw][mask] = patch[mask]
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(1, 3)
    n_distractors = random.randint(0, 2)

    patches: list[np.ndarray] = []
    for _ in range(n_containers):
        patch = _build_container_patch()
        if patch is None:
            return None
        patches.append(patch)

    for _ in range(n_distractors):
        patch = _build_distractor_patch()
        if patch is None:
            return None
        patches.append(patch)

    grid = _place_top_level_patches(patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
