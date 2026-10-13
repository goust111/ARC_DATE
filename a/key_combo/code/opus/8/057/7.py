import random
from collections import Counter, defaultdict

import numpy as np


BG = 4
BLUE = 1
RED = 2
GREEN = 3

SIDES = ("top", "bottom", "left", "right")
TOP_TEMPLATES = (
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=bool),
)
GEN_RETRIES = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _flanked_positions(mask: np.ndarray, side: str) -> list[int]:
    h, w = mask.shape
    hits: list[int] = []
    if side == "top":
        for c in range(1, w - 1):
            if (not mask[0, c]) and mask[0, c - 1] and mask[0, c + 1]:
                hits.append(c)
    elif side == "bottom":
        for c in range(1, w - 1):
            if (not mask[h - 1, c]) and mask[h - 1, c - 1] and mask[h - 1, c + 1]:
                hits.append(c)
    elif side == "left":
        for r in range(1, h - 1):
            if (not mask[r, 0]) and mask[r - 1, 0] and mask[r + 1, 0]:
                hits.append(r)
    elif side == "right":
        for r in range(1, h - 1):
            if (not mask[r, w - 1]) and mask[r - 1, w - 1] and mask[r + 1, w - 1]:
                hits.append(r)
    else:
        raise ValueError(side)
    return hits


def _notch_side(mask: np.ndarray) -> str:
    hits = {side: _flanked_positions(mask, side) for side in SIDES}
    active = [side for side, xs in hits.items() if xs]
    assert len(active) == 1, f"expected exactly one notch side, got {hits}"
    return active[0]


def _rotate_top_template(mask: np.ndarray, side: str) -> np.ndarray:
    k = {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]
    return np.rot90(mask, k=k)


def _random_mask(side: str) -> np.ndarray:
    choices = list(TOP_TEMPLATES)
    random.shuffle(choices)
    for base in choices:
        mask = _rotate_top_template(base, side)
        if 4 <= int(mask.sum()) <= 8 and _connected_mask(mask) and _notch_side(mask) == side:
            return mask.copy()
    raise RuntimeError("no valid mask")


def _boundary_positions(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    if side == "right":
        return [(r, w - 1) for r in range(h) if mask[r, w - 1]]
    raise ValueError(side)


def _cells(mask: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = anchor
    out = set()
    for r, c in np.argwhere(mask):
        out.add((r0 + int(r), c0 + int(c)))
    return out


def _occupied_coords(anchor: tuple[int, int], mask: np.ndarray) -> np.ndarray:
    r0, c0 = anchor
    rel = np.argwhere(mask)
    rel = rel + np.array([[int(r0), int(c0)]], dtype=int)
    return rel


def _slide_anchor(side: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    H = r1 + 1
    W = c1 + 1
    if side == "top":
        return 0, c0
    if side == "bottom":
        return H - h, c0
    if side == "left":
        return r0, 0
    if side == "right":
        return r0, W - w
    raise ValueError(side)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (not seen[nr, nc]) and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append({"coords": arr, "bbox": _bbox_coords(arr)})
    return out


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for idx, comp in enumerate(_find_components(grid)):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        colors = Counter(map(int, patch[mask]))
        assert set(colors).issubset({BLUE, RED, GREEN})
        markers = [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere((patch == RED) | (patch == GREEN))]
        assert len(markers) == 1, "each object must have exactly one marker"
        mr, mc, marker_color = markers[0]
        assert colors[BLUE] >= 3, "object must have blue body cells"
        side = _notch_side(mask)
        objects.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "marker_rel": (mr, mc),
                "marker_color": marker_color,
                "side": side,
            }
        )
    counts = Counter(obj["marker_color"] for obj in objects)
    assert counts == Counter({RED: 2, GREEN: 2}), f"expected two red and two green markers, got {counts}"
    return objects


def _marker_abs(obj: dict, anchor: tuple[int, int]) -> tuple[int, int]:
    return anchor[0] + obj["marker_rel"][0], anchor[1] + obj["marker_rel"][1]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0 + 1, c1)]
    if a[1] == b[1]:
        col = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, col) for r in range(r0 + 1, r1)]
    return []


def _render_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    groups: dict[int, list[dict]] = defaultdict(list)
    linked_ids: set[int] = set()
    lines: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for obj in objects:
        groups[obj["marker_color"]].append(obj)
    for color, pair in groups.items():
        assert len(pair) == 2
        p1 = _marker_abs(pair[0], pair[0]["final_anchor"])
        p2 = _marker_abs(pair[1], pair[1]["final_anchor"])
        if p1[0] == p2[0] or p1[1] == p2[1]:
            linked_ids.add(pair[0]["id"])
            linked_ids.add(pair[1]["id"])
            lines.append((color, p1, p2))
    for obj in objects:
        body_color = GREEN if obj["id"] in linked_ids else RED
        r0, c0 = obj["final_anchor"]
        for rr, cc in np.argwhere(obj["mask"]):
            out[r0 + int(rr), c0 + int(cc)] = body_color
        mr, mc = obj["marker_rel"]
        out[r0 + mr, c0 + mc] = obj["marker_color"]
    for color, p1, p2 in lines:
        for rr, cc in _segment_cells(p1, p2):
            if out[rr, cc] == BG:
                out[rr, cc] = color
    return out


def _analyze_alignment(objects: list[dict], key: str) -> dict[int, bool]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["marker_color"]].append(obj)
    status: dict[int, bool] = {}
    for color, pair in groups.items():
        p1 = _marker_abs(pair[0], pair[0][key])
        p2 = _marker_abs(pair[1], pair[1][key])
        status[color] = (p1[0] == p2[0]) or (p1[1] == p2[1])
    return status


def _swept_cells(obj: dict) -> set[tuple[int, int]]:
    start_r, start_c = obj["input_anchor"]
    end_r, end_c = obj["final_anchor"]
    dr = 0 if end_r == start_r else (1 if end_r > start_r else -1)
    dc = 0 if end_c == start_c else (1 if end_c > start_c else -1)
    steps = max(abs(end_r - start_r), abs(end_c - start_c))
    swept: set[tuple[int, int]] = set()
    for t in range(steps + 1):
        anchor = (start_r + dr * t, start_c + dc * t)
        swept |= _cells(obj["mask"], anchor)
    return swept


def _pairwise_disjoint(cell_sets: list[set[tuple[int, int]]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for pts in cell_sets:
        if seen & pts:
            return False
        seen |= pts
    return True


def _pairwise_non_touching(cell_sets: list[set[tuple[int, int]]]) -> bool:
    for i, pts in enumerate(cell_sets):
        others = set()
        for j, other in enumerate(cell_sets):
            if i != j:
                others |= other
        for r, c in pts:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in others:
                    return False
    return True


def _line_clear(objects: list[dict], color: int) -> bool:
    pair = [obj for obj in objects if obj["marker_color"] == color]
    assert len(pair) == 2
    occ = set()
    for obj in objects:
        occ |= _cells(obj["mask"], obj["final_anchor"])
    p1 = _marker_abs(pair[0], pair[0]["final_anchor"])
    p2 = _marker_abs(pair[1], pair[1]["final_anchor"])
    segment = _segment_cells(p1, p2)
    return bool(segment) and all(cell not in occ for cell in segment)


def _validate_objects(objects: list[dict], grid_shape: tuple[int, int], aligned_color: int) -> bool:
    H, W = grid_shape
    for obj in objects:
        h, w = obj["mask"].shape
        for key in ("input_anchor", "final_anchor"):
            r0, c0 = obj[key]
            if not (0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W):
                return False
        if obj["side"] == "top" and obj["final_anchor"][0] != 0:
            return False
        if obj["side"] == "bottom" and obj["final_anchor"][0] != H - h:
            return False
        if obj["side"] == "left" and obj["final_anchor"][1] != 0:
            return False
        if obj["side"] == "right" and obj["final_anchor"][1] != W - w:
            return False
        if tuple(obj["input_anchor"]) == tuple(obj["final_anchor"]):
            return False
    if not _pairwise_disjoint([_cells(obj["mask"], obj["input_anchor"]) for obj in objects]):
        return False
    if not _pairwise_non_touching([_cells(obj["mask"], obj["input_anchor"]) for obj in objects]):
        return False
    if not _pairwise_disjoint([_cells(obj["mask"], obj["final_anchor"]) for obj in objects]):
        return False
    if not _pairwise_disjoint([_swept_cells(obj) for obj in objects]):
        return False
    before = _analyze_alignment(objects, "input_anchor")
    after = _analyze_alignment(objects, "final_anchor")
    other_color = GREEN if aligned_color == RED else RED
    if before[aligned_color] or (not after[aligned_color]):
        return False
    if (not before[other_color]) or after[other_color]:
        return False
    if not _line_clear(objects, aligned_color):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked blue object to the edge indicated by its border notch, then connect same-color markers that align after sliding and recolor objects by whether they joined a line.

    Input structure:
    - Yellow cells form the background.
    - The foreground is split into separate 4-connected objects whose non-marker cells are blue.
    - Each object contains exactly one red or green marker cell and has exactly one flanked notch on one side of its bbox border.

    Rule:
    - Read the notch side of each object: top, bottom, left, or right; slide the whole object in that direction until its bbox touches that grid edge, keeping the marker's local offset unchanged.
    - After all objects move, compare the two red markers and the two green markers separately; if a same-color pair ends up in one row or one column, draw that color on the background cells strictly between them.
    - Recolor every blue body cell of a participating object green; recolor every blue body cell of a non-participating object red; keep marker colors and the yellow background unchanged.

    Key insight:
    - The visible same-color alignments must be evaluated only after the notch-driven slide, so the slide outcome is the indispensable reference frame for both the line drawing and the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = map(int, g.shape)
    objects = _parse_objects(g)
    for obj in objects:
        h, w = obj["mask"].shape
        side = obj["side"]
        r0, c0, _, _ = obj["bbox"]
        if side == "top":
            obj["final_anchor"] = (0, c0)
        elif side == "bottom":
            obj["final_anchor"] = (H - h, c0)
        elif side == "left":
            obj["final_anchor"] = (r0, 0)
        elif side == "right":
            obj["final_anchor"] = (r0, W - w)
        else:
            raise ValueError(side)
    return _render_output(g.shape, objects)


def _row_scenario(H: int, W: int, aligned_color: int) -> list[dict] | None:
    other_color = GREEN if aligned_color == RED else RED
    masks = [_random_mask("top"), _random_mask("top"), _random_mask("top"), _random_mask("bottom")]

    right_rows = {r for r, _ in _boundary_positions(masks[0], "right")}
    left_rows = {r for r, _ in _boundary_positions(masks[1], "left")}
    common_rows = sorted(right_rows & left_rows)
    if not common_rows:
        return None
    rr = random.choice(common_rows)
    marker_rels = [
        (rr, masks[0].shape[1] - 1),
        (rr, 0),
    ]

    b1_side = random.choice(("right", "left"))
    b2_side = random.choice(("right", "left"))
    marker_rels.append(random.choice(_boundary_positions(masks[2], b1_side)))
    marker_rels.append(random.choice(_boundary_positions(masks[3], b2_side)))

    colors = [aligned_color, aligned_color, other_color, other_color]
    objects = []
    for idx, (mask, side, marker_rel, color) in enumerate(
        zip(masks, ("top", "top", "top", "bottom"), marker_rels, colors)
    ):
        objects.append(
            {
                "id": idx,
                "mask": mask,
                "marker_rel": tuple(map(int, marker_rel)),
                "marker_color": color,
                "side": side,
            }
        )

    col_candidates = []
    for _ in range(120):
        cols = []
        for obj in objects:
            h, w = obj["mask"].shape
            if obj["id"] == 0:
                c = random.randint(1, max(1, W // 4))
            elif obj["id"] == 1:
                c = random.randint(max(0, W * 3 // 5 - w), W - w - 1)
            elif obj["id"] == 2:
                c = random.randint(0, max(0, W // 5))
            else:
                c = random.randint(max(0, W * 4 // 5 - w), W - w)
            cols.append(c)
        temp = []
        for obj, c in zip(objects, cols):
            h, w = obj["mask"].shape
            fr = 0 if obj["side"] == "top" else H - h
            temp.append({**obj, "final_anchor": (fr, c)})
        if _pairwise_disjoint([_cells(obj["mask"], obj["final_anchor"]) for obj in temp]) and _line_clear(
            temp, aligned_color
        ):
            col_candidates.append(temp)
    if not col_candidates:
        return None
    final_objs = random.choice(col_candidates)

    for _ in range(200):
        objs = [dict(obj) for obj in final_objs]
        for obj in objs[:2]:
            h, _ = obj["mask"].shape
            obj["input_anchor"] = (random.randint(1, H - h - 2), obj["final_anchor"][1])
        row_target = random.randint(H // 3, H * 2 // 3)
        ok = True
        for obj in objs[2:]:
            h, _ = obj["mask"].shape
            mr = obj["marker_rel"][0]
            r0 = row_target - mr
            if obj["side"] == "top":
                ok &= 1 <= r0 <= H - h - 2
            else:
                ok &= 1 <= r0 <= H - h - 1
            obj["input_anchor"] = (r0, obj["final_anchor"][1])
        if ok and _validate_objects(objs, (H, W), aligned_color):
            return objs
    return None


def _col_scenario(H: int, W: int, aligned_color: int) -> list[dict] | None:
    other_color = GREEN if aligned_color == RED else RED
    masks = [_random_mask("left"), _random_mask("left"), _random_mask("left"), _random_mask("right")]

    bottom_cols = {c for _, c in _boundary_positions(masks[0], "bottom")}
    top_cols = {c for _, c in _boundary_positions(masks[1], "top")}
    common_cols = sorted(bottom_cols & top_cols)
    if not common_cols:
        return None
    cc = random.choice(common_cols)
    marker_rels = [
        (masks[0].shape[0] - 1, cc),
        (0, cc),
    ]

    b1_side = random.choice(("bottom", "top"))
    b2_side = random.choice(("bottom", "top"))
    marker_rels.append(random.choice(_boundary_positions(masks[2], b1_side)))
    marker_rels.append(random.choice(_boundary_positions(masks[3], b2_side)))

    colors = [aligned_color, aligned_color, other_color, other_color]
    objects = []
    for idx, (mask, side, marker_rel, color) in enumerate(
        zip(masks, ("left", "left", "left", "right"), marker_rels, colors)
    ):
        objects.append(
            {
                "id": idx,
                "mask": mask,
                "marker_rel": tuple(map(int, marker_rel)),
                "marker_color": color,
                "side": side,
            }
        )

    row_candidates = []
    for _ in range(120):
        rows = []
        for obj in objects:
            h, w = obj["mask"].shape
            if obj["id"] == 0:
                r = random.randint(1, max(1, H // 4))
            elif obj["id"] == 1:
                r = random.randint(max(0, H * 3 // 5 - h), H - h - 1)
            elif obj["id"] == 2:
                r = random.randint(0, max(0, H // 5))
            else:
                r = random.randint(max(0, H * 4 // 5 - h), H - h)
            rows.append(r)
        temp = []
        for obj, r in zip(objects, rows):
            h, w = obj["mask"].shape
            fc = 0 if obj["side"] == "left" else W - w
            temp.append({**obj, "final_anchor": (r, fc)})
        if _pairwise_disjoint([_cells(obj["mask"], obj["final_anchor"]) for obj in temp]) and _line_clear(
            temp, aligned_color
        ):
            row_candidates.append(temp)
    if not row_candidates:
        return None
    final_objs = random.choice(row_candidates)

    for _ in range(200):
        objs = [dict(obj) for obj in final_objs]
        for obj in objs[:2]:
            _, w = obj["mask"].shape
            obj["input_anchor"] = (obj["final_anchor"][0], random.randint(1, W - w - 2))
        col_target = random.randint(W // 3, W * 2 // 3)
        ok = True
        for obj in objs[2:]:
            _, w = obj["mask"].shape
            mc = obj["marker_rel"][1]
            c0 = col_target - mc
            if obj["side"] == "left":
                ok &= 1 <= c0 <= W - w - 2
            else:
                ok &= 1 <= c0 <= W - w - 1
            obj["input_anchor"] = (obj["final_anchor"][0], c0)
        if ok and _validate_objects(objs, (H, W), aligned_color):
            return objs
    return None


def _objects_to_input(H: int, W: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for obj in objects:
        r0, c0 = obj["input_anchor"]
        for rr, cc in np.argwhere(obj["mask"]):
            grid[r0 + int(rr), c0 + int(cc)] = BLUE
        mr, mc = obj["marker_rel"]
        grid[r0 + mr, c0 + mc] = obj["marker_color"]
    return grid


def _independent_expected(H: int, W: int, objects: list[dict]) -> np.ndarray:
    expected = np.full((H, W), BG, dtype=int)
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["marker_color"]].append(obj)
    participating: set[int] = set()
    for color, pair in groups.items():
        p1 = _marker_abs(pair[0], pair[0]["final_anchor"])
        p2 = _marker_abs(pair[1], pair[1]["final_anchor"])
        if p1[0] == p2[0] or p1[1] == p2[1]:
            participating |= {pair[0]["id"], pair[1]["id"]}
    for obj in objects:
        body_color = GREEN if obj["id"] in participating else RED
        coords = _occupied_coords(obj["final_anchor"], obj["mask"])
        expected[coords[:, 0], coords[:, 1]] = body_color
        mr, mc = obj["marker_rel"]
        expected[obj["final_anchor"][0] + mr, obj["final_anchor"][1] + mc] = obj["marker_color"]
    for color, pair in groups.items():
        p1 = _marker_abs(pair[0], pair[0]["final_anchor"])
        p2 = _marker_abs(pair[1], pair[1]["final_anchor"])
        for rr, cc in _segment_cells(p1, p2):
            if expected[rr, cc] == BG:
                expected[rr, cc] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(13, 16)
    W = random.randint(13, 16)
    aligned_color = random.choice((RED, GREEN))
    axis = random.choice(("row", "col"))
    if axis == "row":
        objects = _row_scenario(H, W, aligned_color)
    else:
        objects = _col_scenario(H, W, aligned_color)
    if objects is None:
        return None
    inp = _objects_to_input(H, W, objects)
    expected = _independent_expected(H, W, objects)
    if not np.array_equal(main(inp), expected):
        return None
    parsed = _parse_objects(inp)
    if len(parsed) != 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
