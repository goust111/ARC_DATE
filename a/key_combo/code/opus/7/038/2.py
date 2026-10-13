from __future__ import annotations

import random

import numpy as np


RAY_COLORS_BY_RANK = (2, 3, 4, 5)
INPUT_COLORS = (1, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _direction_from_defect(
    bbox: tuple[int, int, int, int],
    defect: tuple[int, int],
) -> str:
    r0, c0, r1, c1 = bbox
    dr, dc = defect
    on_top = dr == r0
    on_bottom = dr == r1
    on_left = dc == c0
    on_right = dc == c1
    edge_count = sum((on_top, on_bottom, on_left, on_right))
    assert edge_count >= 1, "defect must lie on the rectangle boundary"

    if edge_count == 1:
        if on_top:
            return "up"
        if on_bottom:
            return "down"
        if on_left:
            return "left"
        return "right"

    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height <= width:
        return "up" if on_top else "down"
    return "left" if on_left else "right"


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects: list[dict] = []
    for comp in _find_components(g):
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = np.argwhere(patch == 0)
        assert holes.shape == (1, 2), "each rectangle bbox must contain exactly one black defect"
        hh, ww = patch.shape
        assert len(coords) == hh * ww - 1, "each object must be a solid rectangle minus one edge pixel"
        defect = (r0 + int(holes[0, 0]), c0 + int(holes[0, 1]))
        rects.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "defect": defect,
                "direction": _direction_from_defect((r0, c0, r1, c1), defect),
                "area": int(hh * ww),
                "height": int(hh),
                "width": int(ww),
            }
        )

    assert 3 <= len(rects) <= 4, "expected 3 or 4 defect rectangles"
    areas = [rect["area"] for rect in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """Paint area-ranked rays that start from each rectangle's unique edge defect.

    Input structure:
    - The grid is a black background containing 3-4 separated single-color rectangles.
    - Each rectangle is solid except for exactly one black defect pixel on its bbox edge.
    - Rectangle colors are distinct, and each bbox contains exactly one black cell.

    Rule:
    - Locate every rectangle and its unique defect pixel inside the bbox.
    - The defect edge determines the ray direction; corner defects use the shorter bbox dimension to break the tie.
    - Rank rectangles by bbox area from small to large and color their rays red, green, yellow, and grey.
    - Start each ray from the defect cell, leave that defect black, and extend until the grid edge or another rectangle blocks the path.

    Key insight:
    - The missing edge cell is not repaired; it simultaneously encodes the ray's direction and its launch point, while the ray color comes from a global area ranking across all rectangles.
    """
    g = np.asarray(grid, dtype=int)
    rects = _detect_rectangles(g)
    out = g.copy()

    for rank, rect in enumerate(sorted(rects, key=lambda item: item["area"])):
        color = RAY_COLORS_BY_RANK[rank]
        dr, dc = DIRS[rect["direction"]]
        r, c = rect["defect"]
        r += dr
        c += dc
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            if int(g[r, c]) != 0:
                break
            assert int(out[r, c]) in (0, color), "rays should not overlap with conflicting colors"
            out[r, c] = color
            r += dr
            c += dc

    return out


def _sample_dims(direction: str, use_corner: bool) -> tuple[int, int]:
    if use_corner and direction in ("left", "right"):
        h = random.randint(5, 6)
        w = random.randint(4, min(5, h - 1))
        return h, w
    if use_corner and direction in ("up", "down"):
        w = random.randint(5, 7)
        h = random.randint(3, min(5, w - 1))
        return h, w
    return random.randint(3, 6), random.randint(4, 7)


def _gap_ok(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    min_gap: int = 3,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + min_gap < br0
        or br1 + min_gap < ar0
        or ac1 + min_gap < bc0
        or bc1 + min_gap < ac0
    )


def _choose_defect(
    obj: dict,
    blocker: dict | None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    direction = obj["direction"]
    use_corner = obj["use_corner"]

    if direction == "right":
        if use_corner:
            assert obj["width"] < obj["height"]
            if blocker is not None:
                assert blocker["bbox"][0] <= r0 <= blocker["bbox"][2]
            return r0, c1
        lo = r0 + 1
        hi = r1 - 1
        if blocker is not None:
            lo = max(lo, blocker["bbox"][0])
            hi = min(hi, blocker["bbox"][2])
        assert lo <= hi
        return random.randint(lo, hi), c1

    if direction == "left":
        if use_corner:
            assert obj["width"] < obj["height"]
            if blocker is not None:
                assert blocker["bbox"][0] <= r0 <= blocker["bbox"][2]
            return r0, c0
        lo = r0 + 1
        hi = r1 - 1
        if blocker is not None:
            lo = max(lo, blocker["bbox"][0])
            hi = min(hi, blocker["bbox"][2])
        assert lo <= hi
        return random.randint(lo, hi), c0

    if direction == "down":
        if use_corner:
            assert obj["height"] < obj["width"]
            if blocker is not None:
                assert blocker["bbox"][1] <= c0 <= blocker["bbox"][3]
            return r1, c0
        lo = c0 + 1
        hi = c1 - 1
        if blocker is not None:
            lo = max(lo, blocker["bbox"][1])
            hi = min(hi, blocker["bbox"][3])
        assert lo <= hi
        return r1, random.randint(lo, hi)

    if use_corner:
        assert obj["height"] < obj["width"]
        if blocker is not None:
            assert blocker["bbox"][1] <= c0 <= blocker["bbox"][3]
        return r0, c0
    lo = c0 + 1
    hi = c1 - 1
    if blocker is not None:
        lo = max(lo, blocker["bbox"][1])
        hi = min(hi, blocker["bbox"][3])
    assert lo <= hi
    return r0, random.randint(lo, hi)


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
        dr, dc = obj["defect"]
        grid[dr, dc] = 0
    return grid


def _sample_objects(template: str) -> list[dict] | None:
    if template == "chain3":
        dirs = ("right", "down", "left")
    elif template == "cycle4":
        dirs = ("right", "down", "left", "up")
    else:
        dirs = ("right", "down", "left", "down")

    forced_corner = random.randrange(len(dirs))
    objects: list[dict] = []
    used_areas: set[int] = set()
    for idx, direction in enumerate(dirs):
        for _ in range(40):
            use_corner = idx == forced_corner or (random.random() < 0.35)
            h, w = _sample_dims(direction, use_corner)
            area = h * w
            if area in used_areas:
                continue
            objects.append(
                {
                    "direction": direction,
                    "height": h,
                    "width": w,
                    "use_corner": use_corner,
                }
            )
            used_areas.add(area)
            break
        else:
            return None

    random.shuffle(objects)
    for obj, color in zip(objects, random.sample(INPUT_COLORS, len(objects))):
        obj["color"] = color
    return objects


def _place_chain3(objects: list[dict]) -> tuple[tuple[int, int], list[dict], list[int | None]] | None:
    by_dir = {obj["direction"]: obj for obj in objects}
    a = by_dir["right"]
    b = by_dir["down"]
    c = by_dir["left"]

    top_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    gap_h = random.randint(3, 6)
    gap_v = random.randint(3, 5)

    a["bbox"] = (top_margin, left_margin, top_margin + a["height"] - 1, left_margin + a["width"] - 1)
    right_c = left_margin + a["width"] + gap_h
    b["bbox"] = (top_margin, right_c, top_margin + b["height"] - 1, right_c + b["width"] - 1)
    bottom_r = top_margin + max(a["height"], b["height"]) + gap_v
    c["bbox"] = (bottom_r, right_c, bottom_r + c["height"] - 1, right_c + c["width"] - 1)

    height = c["bbox"][2] + 1 + bottom_margin
    width = max(b["bbox"][3], c["bbox"][3]) + 1 + right_margin
    placed = [a, b, c]
    blockers = [1, 2, None]
    return (height, width), placed, blockers


def _place_cycle4(objects: list[dict]) -> tuple[tuple[int, int], list[dict], list[int | None]] | None:
    by_dir = {obj["direction"]: obj for obj in objects}
    a = by_dir["right"]
    b = by_dir["down"]
    c = by_dir["left"]
    d = by_dir["up"]

    top_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    gap_h = random.randint(3, 6)
    gap_v = random.randint(3, 5)

    left_col_w = max(a["width"], d["width"])
    a["bbox"] = (top_margin, left_margin, top_margin + a["height"] - 1, left_margin + a["width"] - 1)
    d["bbox"] = (
        top_margin + max(a["height"], b["height"]) + gap_v,
        left_margin,
        top_margin + max(a["height"], b["height"]) + gap_v + d["height"] - 1,
        left_margin + d["width"] - 1,
    )
    right_c = left_margin + left_col_w + gap_h
    b["bbox"] = (top_margin, right_c, top_margin + b["height"] - 1, right_c + b["width"] - 1)
    c["bbox"] = (
        d["bbox"][0],
        right_c,
        d["bbox"][0] + c["height"] - 1,
        right_c + c["width"] - 1,
    )

    height = max(c["bbox"][2], d["bbox"][2]) + 1 + bottom_margin
    width = max(b["bbox"][3], c["bbox"][3]) + 1 + right_margin
    placed = [a, b, c, d]
    blockers = [1, 2, 3, 0]
    return (height, width), placed, blockers


def _place_mixed4(objects: list[dict]) -> tuple[tuple[int, int], list[dict], list[int | None]] | None:
    rights = [obj for obj in objects if obj["direction"] == "right"]
    lefts = [obj for obj in objects if obj["direction"] == "left"]
    downs = [obj for obj in objects if obj["direction"] == "down"]
    if len(rights) != 1 or len(lefts) != 1 or len(downs) != 2:
        return None
    a = rights[0]
    b, d = sorted(downs, key=lambda obj: obj["height"], reverse=True)
    c = lefts[0]

    top_margin = random.randint(1, 3)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    bottom_margin = random.randint(1, 3)
    gap_h = random.randint(3, 5)
    gap_v1 = random.randint(3, 4)
    gap_v2 = random.randint(3, 4)

    a["bbox"] = (top_margin, left_margin, top_margin + a["height"] - 1, left_margin + a["width"] - 1)
    right_c = left_margin + a["width"] + gap_h
    b["bbox"] = (top_margin, right_c, top_margin + b["height"] - 1, right_c + b["width"] - 1)

    c_r0 = top_margin + max(a["height"], b["height"]) + gap_v1
    c["bbox"] = (c_r0, right_c, c_r0 + c["height"] - 1, right_c + c["width"] - 1)

    d_r0 = c["bbox"][2] + 1 + gap_v2
    d["bbox"] = (d_r0, left_margin, d_r0 + d["height"] - 1, left_margin + d["width"] - 1)

    height = d["bbox"][2] + 1 + bottom_margin
    width = max(b["bbox"][3], c["bbox"][3]) + 1 + right_margin
    placed = [a, b, c, d]
    blockers = [1, 2, None, None]
    return (height, width), placed, blockers


def _place_objects(template: str, objects: list[dict]) -> tuple[tuple[int, int], list[dict], list[int | None]] | None:
    if template == "chain3":
        return _place_chain3(objects)
    if template == "cycle4":
        return _place_cycle4(objects)
    return _place_mixed4(objects)


def _build_base_input(template: str) -> np.ndarray | None:
    objects = _sample_objects(template)
    if objects is None:
        return None
    placed = _place_objects(template, objects)
    if placed is None:
        return None
    shape, objects, blocker_ids = placed
    if max(shape) > 30:
        return None

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if not _gap_ok(objects[i]["bbox"], objects[j]["bbox"]):
                return None

    if template in ("chain3", "cycle4"):
        assert len({obj["direction"] for obj in objects}) >= 3
    else:
        assert len({obj["direction"] for obj in objects}) >= 2

    for idx, blocker_idx in enumerate(blocker_ids):
        blocker = None if blocker_idx is None else objects[blocker_idx]
        objects[idx]["defect"] = _choose_defect(objects[idx], blocker)

    return _render_input(shape, objects)


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _is_valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        rects = _detect_rectangles(grid)
        if len({rect["direction"] for rect in rects}) < 2:
            return False
        out = main(grid)
    except Exception:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    template = random.choices(
        ("chain3", "cycle4", "mixed4"),
        weights=(3, 2, 2),
        k=1,
    )[0]
    grid = _build_base_input(template)
    if grid is None:
        return None
    grid = _apply_random_symmetry(grid)
    if not _is_valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
