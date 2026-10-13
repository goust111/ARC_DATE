from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ODD_SIZES = (1, 3, 5)
COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 600


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_cells(r0: int, c0: int, h: int, w: int, direction: str) -> list[tuple[int, int]]:
    cr = r0 + h // 2
    cc = c0 + w // 2
    if direction == "up":
        return [(r0 - 1, cc - 1), (r0 - 1, cc), (r0 - 1, cc + 1)]
    if direction == "down":
        return [(r0 + h, cc - 1), (r0 + h, cc), (r0 + h, cc + 1)]
    if direction == "left":
        return [(cr - 1, c0 - 1), (cr, c0 - 1), (cr + 1, c0 - 1)]
    assert direction == "right"
    return [(cr - 1, c0 + w), (cr, c0 + w), (cr + 1, c0 + w)]


def _arrow_center(obj: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = obj["bbox"]
    h, w, direction = obj["h"], obj["w"], obj["dir"]
    cr = r0 + h // 2
    cc = c0 + w // 2
    if direction == "up":
        return r0 - 1, cc
    if direction == "down":
        return r0 + h, cc
    if direction == "left":
        return cr, c0 - 1
    return cr, c0 + w


def _shape_cells(obj: dict) -> set[tuple[int, int]]:
    cells = {
        (r, c)
        for r in range(obj["r0"], obj["r0"] + obj["h"])
        for c in range(obj["c0"], obj["c0"] + obj["w"])
    }
    cells.update(_arrow_cells(obj["r0"], obj["c0"], obj["h"], obj["w"], obj["dir"]))
    return cells


def _padded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _match_arrow(obj: dict, arrow: list[tuple[int, int]]) -> str | None:
    rows = sorted(r for r, _ in arrow)
    cols = sorted(c for _, c in arrow)
    r0, c0, r1, c1 = obj["bbox"]
    cr = obj["center_r"]
    cc = obj["center_c"]
    if rows[0] == rows[-1] and cols == [cc - 1, cc, cc + 1]:
        if rows[0] == r0 - 1:
            return "up"
        if rows[0] == r1 + 1:
            return "down"
    if cols[0] == cols[-1] and rows == [cr - 1, cr, cr + 1]:
        if cols[0] == c0 - 1:
            return "left"
        if cols[0] == c1 + 1:
            return "right"
    return None


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, int | None]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = {int(v) for v in np.unique(g)}
    assert BG in colors, "expected black background"
    assert RED in colors, "expected red arrows"

    objects: list[dict] = []
    object_map = np.full(g.shape, -1, dtype=int)
    object_colors = sorted(c for c in colors if c not in (BG, RED))
    assert object_colors, "expected colored rectangles"
    for color in object_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % 2 == 1 and w % 2 == 1, "projection alignment needs odd rectangle sizes"
        assert len(coords) == h * w, "each object must be a solid rectangle"
        obj = {
            "id": len(objects),
            "color": color,
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "h": h,
            "w": w,
            "center_r": r0 + h // 2,
            "center_c": c0 + w // 2,
        }
        for r, c in coords:
            assert object_map[r, c] == -1, "rectangles must not overlap"
            object_map[r, c] = obj["id"]
        objects.append(obj)

    red_comps = _components(g == RED)
    assert len(red_comps) == len(objects), "expected exactly one red arrow per rectangle"
    used_arrows: set[int] = set()
    for obj in objects:
        matches: list[tuple[int, str]] = []
        for i, comp in enumerate(red_comps):
            if len(comp) != 3:
                continue
            direction = _match_arrow(obj, comp)
            if direction is not None:
                matches.append((i, direction))
        assert len(matches) == 1, "each rectangle must have exactly one matching arrow"
        arrow_i, direction = matches[0]
        assert arrow_i not in used_arrows, "each arrow must belong to only one rectangle"
        used_arrows.add(arrow_i)
        obj["dir"] = direction
        obj["arrow"] = red_comps[arrow_i]

    targets: dict[int, int | None] = {}
    h, w = g.shape
    for obj in objects:
        ar, ac = _arrow_center(obj)
        dr, dc = DIRS[obj["dir"]]
        nr, nc = ar + dr, ac + dc
        hit: int | None = None
        while 0 <= nr < h and 0 <= nc < w:
            oid = int(object_map[nr, nc])
            if oid >= 0:
                assert oid != obj["id"], "ray unexpectedly hit source object"
                hit = oid
                break
            nr += dr
            nc += dc
        targets[obj["id"]] = hit

    indegree = {obj["id"]: 0 for obj in objects}
    for dst in targets.values():
        if dst is not None:
            indegree[dst] += 1
    assert max(indegree.values()) <= 1, "expected disjoint directed chains"

    seen: set[int] = set()
    for root in sorted(obj["id"] for obj in objects if indegree[obj["id"]] == 0):
        cur = root
        while cur is not None:
            assert cur not in seen, "chains must be acyclic"
            seen.add(cur)
            cur = targets[cur]
    assert len(seen) == len(objects), "every object must belong to a chain"
    return objects, targets


def _projection_bbox(src: dict, dst: dict) -> tuple[int, int, int, int]:
    direction = src["dir"]
    if direction == "right":
        anchor_r = dst["center_r"]
        anchor_c = dst["bbox"][1]
    elif direction == "left":
        anchor_r = dst["center_r"]
        anchor_c = dst["bbox"][3]
    elif direction == "down":
        anchor_r = dst["bbox"][0]
        anchor_c = dst["center_c"]
    else:
        anchor_r = dst["bbox"][2]
        anchor_c = dst["center_c"]
    hh = src["h"] // 2
    hw = src["w"] // 2
    return anchor_r - hh, anchor_c - hw, anchor_r + hh, anchor_c + hw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build projection chains from arrow directions and stamp each rectangle onto the next rectangle.

    Input structure
    - Black background with several solid, uniquely colored rectangles.
    - Each rectangle has one adjacent red 3-pixel arrow on one side.
    - The arrow points outward and defines the rectangle's ray-cast direction.

    Rule
    - For each rectangle, follow the arrow direction until the ray first hits another rectangle.
    - These hits define disjoint directed chains.
    - Process each chain from root to tail. When A projects to B, stamp A's full rectangle shape
      in A's color onto B, centered at the center pixel of B's facing edge.
    - Remove all red arrow pixels at the end.
    """
    g = np.asarray(grid, dtype=int)
    objects, targets = _extract_scene(g)

    out = g.copy()
    h, w = g.shape
    object_by_id = {obj["id"]: obj for obj in objects}
    indegree = {obj["id"]: 0 for obj in objects}
    for dst in targets.values():
        if dst is not None:
            indegree[dst] += 1

    order: list[int] = []
    for root in sorted(obj["id"] for obj in objects if indegree[obj["id"]] == 0):
        cur = root
        while cur is not None:
            order.append(cur)
            cur = targets[cur]

    for oid in order:
        dst_id = targets[oid]
        if dst_id is None:
            continue
        src = object_by_id[oid]
        dst = object_by_id[dst_id]
        pr0, pc0, pr1, pc1 = _projection_bbox(src, dst)
        assert 0 <= pr0 <= pr1 < h and 0 <= pc0 <= pc1 < w, "projection must stay in bounds"
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = src["color"]

    out[g == RED] = BG
    return out


def _sample_chain_lengths() -> list[int]:
    for _ in range(200):
        chain_count = random.randint(1, 3)
        lengths = [random.randint(1, 4) for _ in range(chain_count)]
        total = sum(lengths)
        if 3 <= total <= 7 and any(length > 1 for length in lengths):
            return lengths
    raise RuntimeError("failed to sample chain lengths")


def _sample_rect() -> tuple[int, int]:
    return random.choice(ODD_SIZES), random.choice(ODD_SIZES)


def _place_scene(h: int, w: int, lengths: list[int]) -> list[dict] | None:
    colors = random.sample(COLOR_POOL, sum(lengths))
    objects: list[dict] = []
    blocked: set[tuple[int, int]] = set()

    def can_place(candidate: dict) -> bool:
        cells = _shape_cells(candidate)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            return False
        if candidate["r0"] < 1 or candidate["c0"] < 1:
            return False
        if candidate["r0"] + candidate["h"] > h - 1 or candidate["c0"] + candidate["w"] > w - 1:
            return False
        return cells.isdisjoint(blocked)

    color_i = 0
    for length in lengths:
        for step in range(length):
            h0, w0 = _sample_rect()
            if step == 0:
                direction_choices = list(DIRS)
                direction = random.choice(direction_choices)
                for _ in range(80):
                    r0 = random.randint(1, h - h0 - 1)
                    c0 = random.randint(1, w - w0 - 1)
                    candidate = {
                        "color": colors[color_i],
                        "r0": r0,
                        "c0": c0,
                        "h": h0,
                        "w": w0,
                        "dir": direction,
                    }
                    if can_place(candidate):
                        break
                else:
                    return None
            else:
                prev = objects[-1]
                incoming = prev["dir"]
                choices = [d for d in DIRS if d != OPPOSITE[incoming]]
                direction = random.choice(choices)
                gap = random.randint(1, 4)
                pr0, pc0, ph, pw = prev["r0"], prev["c0"], prev["h"], prev["w"]
                pcr = pr0 + ph // 2
                pcc = pc0 + pw // 2
                if incoming == "right":
                    r0 = pcr - h0 // 2
                    c0 = pc0 + pw + 1 + gap
                elif incoming == "left":
                    r0 = pcr - h0 // 2
                    c0 = pc0 - w0 - 1 - gap
                elif incoming == "down":
                    r0 = pr0 + ph + 1 + gap
                    c0 = pcc - w0 // 2
                else:
                    r0 = pr0 - h0 - 1 - gap
                    c0 = pcc - w0 // 2
                candidate = {
                    "color": colors[color_i],
                    "r0": r0,
                    "c0": c0,
                    "h": h0,
                    "w": w0,
                    "dir": direction,
                }
                if not can_place(candidate):
                    return None

            objects.append(candidate)
            blocked.update(_padded(_shape_cells(candidate), h, w))
            color_i += 1

    chain_start = 0
    for length in lengths:
        chain_end = chain_start + length
        for i in range(chain_start, chain_end - 1):
            objects[i]["next_color"] = objects[i + 1]["color"]
        objects[chain_end - 1]["next_color"] = None
        chain_start = chain_end
    return objects


def _render(objects: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        grid[obj["r0"] : obj["r0"] + obj["h"], obj["c0"] : obj["c0"] + obj["w"]] = obj["color"]
        for r, c in _arrow_cells(obj["r0"], obj["c0"], obj["h"], obj["w"], obj["dir"]):
            grid[r, c] = RED
    return grid


def _validate_generated(grid: np.ndarray, objects: list[dict]) -> bool:
    try:
        parsed, targets = _extract_scene(grid)
    except AssertionError:
        return False
    by_color = {obj["color"]: obj for obj in parsed}
    color_targets = {
        obj["color"]: (None if targets[obj["id"]] is None else parsed[targets[obj["id"]]]["color"])
        for obj in parsed
    }
    for planned in objects:
        if color_targets.get(planned["color"]) != planned["next_color"]:
            return False

    arrow_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    occupied_by_object: dict[tuple[int, int], int] = {}
    for obj in parsed:
        r0, c0, r1, c1 = obj["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                occupied_by_object[(r, c)] = obj["id"]

    proj_claims: dict[tuple[int, int], int] = {}
    for src in parsed:
        dst_id = targets[src["id"]]
        if dst_id is None:
            continue
        dst = parsed[dst_id]
        pr0, pc0, pr1, pc1 = _projection_bbox(src, dst)
        if not (0 <= pr0 <= pr1 < grid.shape[0] and 0 <= pc0 <= pc1 < grid.shape[1]):
            return False
        for r in range(pr0, pr1 + 1):
            for c in range(pc0, pc1 + 1):
                if (r, c) in arrow_cells:
                    return False
                owner = occupied_by_object.get((r, c))
                if owner is not None and owner != dst_id:
                    return False
                if (r, c) in proj_claims:
                    return False
                proj_claims[(r, c)] = src["id"]

    try:
        out = main(grid)
    except AssertionError:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        lengths = _sample_chain_lengths()
        objects = _place_scene(h, w, lengths)
        if objects is None:
            continue
        grid = _render(objects, h, w)
        if _validate_generated(grid, objects):
            return grid
    raise RuntimeError("generate_input failed")
