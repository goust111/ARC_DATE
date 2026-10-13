import random
from collections import deque

import numpy as np


BG = 5
LEFT = 1
RIGHT = 2
OUTLINE = 4
HOLE = 6
RECOLOR = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (16, 20)

DIRECT_TEMPLATES = [
    {(0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 1), (1, 2)},
    {(0, 1), (0, 2), (1, 2), (2, 2)},
    {(0, 1), (1, 1), (2, 1), (2, 2)},
    {(0, 1), (0, 2), (1, 2), (1, 3)},
    {(0, 1), (1, 1), (1, 2), (2, 2)},
    {(0, 1), (0, 2), (0, 3), (1, 2)},
    {(0, 1), (1, 1), (1, 2), (1, 3), (2, 2)},
]


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 8) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _sort_key(coords: np.ndarray, side: str, W: int) -> tuple[float, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    center_c = (c0 + c1) / 2.0
    distance = center_c if side == "left" else (W - 1 - center_c)
    return (float(distance), int(r0 + r1), int(c0), int(c1))


def _extract_objects(grid: np.ndarray, color: int, side: str) -> list[np.ndarray]:
    H, W = grid.shape
    interior = np.full_like(grid, BG)
    interior[:, 1:W - 1] = grid[:, 1:W - 1]
    comps = _components_of_color(interior, color, connectivity=8)
    for comp in comps:
        assert not np.any(comp[:, 1] == 0) and not np.any(comp[:, 1] == W - 1), "internal objects cannot include walls"
    return sorted(comps, key=lambda comp: _sort_key(comp, side, W))


def _slide_objects(grid: np.ndarray, color: int, side: str) -> tuple[list[np.ndarray], np.ndarray]:
    H, W = grid.shape
    objects = _extract_objects(grid, color, side)
    occupied = np.zeros((H, W), dtype=bool)
    settled = []
    for coords in objects:
        cur = coords.copy()
        step = -1 if side == "left" else 1
        while True:
            nxt = cur.copy()
            nxt[:, 1] += step
            if side == "left" and int(nxt[:, 1].min()) < 1:
                break
            if side == "right" and int(nxt[:, 1].max()) > W - 2:
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            cur = nxt
        occupied[cur[:, 0], cur[:, 1]] = True
        settled.append(cur)
    return settled, occupied


def _settled_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 5 and W >= 6, "grid too small"
    assert np.all(g[:, 0] == LEFT), "left wall must be solid blue"
    assert np.all(g[:, W - 1] == RIGHT), "right wall must be solid red"
    assert set(map(int, np.unique(g))) <= {BG, LEFT, RIGHT}, "input may only contain grey background, blue, and red"

    out = np.full_like(g, BG)
    out[:, 0] = LEFT
    out[:, W - 1] = RIGHT

    left_objs, _ = _slide_objects(g, LEFT, "left")
    right_objs, _ = _slide_objects(g, RIGHT, "right")

    for coords in left_objs:
        out[coords[:, 0], coords[:, 1]] = LEFT
    for coords in right_objs:
        assert not np.any(out[coords[:, 0], coords[:, 1]] != BG), "different colors cannot overlap after settling"
        out[coords[:, 0], coords[:, 1]] = RIGHT

    return out, {"left": left_objs, "right": right_objs}


def _components_of_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _closed_holes(settled: np.ndarray) -> list[np.ndarray]:
    H, W = settled.shape
    holes = []
    for comp in _components_of_mask(settled == BG, connectivity=4):
        rows = comp[:, 0]
        touches_open_boundary = bool(np.any(rows == 0) or np.any(rows == H - 1))
        if not touches_open_boundary:
            holes.append(comp)
    return holes


def _finalize_settled(settled: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = settled.shape
    out = settled.copy()
    holes = _closed_holes(settled)
    hole_mask = np.zeros((H, W), dtype=bool)
    for hole in holes:
        hole_mask[hole[:, 0], hole[:, 1]] = True
        out[hole[:, 0], hole[:, 1]] = HOLE

    for r in range(H):
        for c in range(1, W - 1):
            if int(settled[r, c]) not in (LEFT, RIGHT):
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and bool(hole_mask[nr, nc]):
                    out[r, c] = RECOLOR
                    break

    outlined = out.copy()
    for r in range(H):
        for c in range(W):
            if int(out[r, c]) != BG:
                continue
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and int(out[nr, nc]) != BG:
                    outlined[r, c] = OUTLINE
                    break
    return outlined, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide same-colored objects into their matching side walls, fill emergent enclosed holes, recolor the hole-touching object pixels, and outline the final structure.

    Input structure:
    - The background is grey (5).
    - The full left column is a solid blue wall (1), and the full right column is a solid red wall (2).
    - The interior contains several blue 8-connected objects and several red 8-connected objects, all separated from the walls.

    Rule:
    - Clear all interior foreground pixels from their original positions.
    - Slide blue objects leftward, processing them from nearest to farthest from the left wall; each stops when one more step would hit the wall column or an already settled blue object.
    - Slide red objects rightward the same way toward the right wall.
    - After all objects settle, find every 4-connected grey region that does not touch the non-wall outer boundary; these are enclosed holes.
    - Fill each enclosed hole with magenta (6), recolor every non-wall object pixel 4-adjacent to a hole to cyan (8), then draw a one-pixel yellow (4) outline on grey cells in the 8-neighborhood of the final non-background structure.

    Key insight:
    - The holes are not given directly in the input: they appear only after the ordered wallward sliding compacts the objects, so the slide simulation must happen before hole detection, recoloring, and outlining.
    """
    settled, _ = _settled_grid(grid)
    out, _ = _finalize_settled(settled)
    return out


def _transform_side_cells(side: str, W: int, top: int, template: set[tuple[int, int]]) -> np.ndarray:
    coords = []
    for dr, d in template:
        r = top + dr
        c = d if side == "left" else W - 1 - d
        coords.append((r, c))
    return np.array(sorted(coords), dtype=int)


def _hole_cluster(side: str, H: int, W: int) -> list[dict]:
    mid = random.randint(3, H - 4)
    top_extra = random.choice([1, 2])
    bottom_extra = random.choice([1, 2])
    blocker_extra_r = random.choice([-1, 0, 1])
    top = {(-2, top_extra), (-1, 1), (-1, 2)}
    bottom = {(1, 1), (1, 2), (2, bottom_extra)}
    blocker = {(-1, 3), (0, 3), (1, 3), (blocker_extra_r, 4)}
    return [
        {"side": side, "role": "anchor", "coords": _transform_side_cells(side, W, mid, top)},
        {"side": side, "role": "anchor", "coords": _transform_side_cells(side, W, mid, bottom)},
        {"side": side, "role": "blocker", "coords": _transform_side_cells(side, W, mid, blocker)},
    ]


def _template_bbox(template: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in template]
    return max(rows) - min(rows) + 1, max(d for _, d in template)


def _direct_object(side: str, H: int, W: int, used_spans: list[tuple[int, int]]) -> np.ndarray | None:
    templates = DIRECT_TEMPLATES[:]
    random.shuffle(templates)
    for template in templates:
        height, _ = _template_bbox(template)
        rows = list(range(1, H - height))
        random.shuffle(rows)
        for top in rows:
            span = (top - 1, top + height)
            if any(not (span[1] < lo or hi < span[0]) for lo, hi in used_spans):
                continue
            return _transform_side_cells(side, W, top, template)
    return None


def _row_span(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].min()), int(coords[:, 0].max())


def _build_settled_layout() -> tuple[np.ndarray, list[dict]] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    settled = np.full((H, W), BG, dtype=int)
    settled[:, 0] = LEFT
    settled[:, W - 1] = RIGHT

    hole_choice = random.choice(["left", "right", "both"])
    hole_sides = {"left", "right"} if hole_choice == "both" else {hole_choice}
    records: list[dict] = []

    for side in ("left", "right"):
        used_spans: list[tuple[int, int]] = []
        if side in hole_sides:
            cluster = _hole_cluster(side, H, W)
            for rec in cluster:
                lo, hi = _row_span(rec["coords"])
                used_spans.append((lo - 1, hi + 1))
                records.append(rec)
        extra_count = random.randint(1, 2) if side in hole_sides else random.randint(2, 3)
        for _ in range(extra_count):
            coords = _direct_object(side, H, W, used_spans)
            if coords is None:
                return None
            lo, hi = _row_span(coords)
            used_spans.append((lo - 1, hi + 1))
            records.append({"side": side, "role": "extra", "coords": coords})

    for rec in records:
        color = LEFT if rec["side"] == "left" else RIGHT
        coords = rec["coords"]
        if np.any(settled[coords[:, 0], coords[:, 1]] != BG):
            return None
        settled[coords[:, 0], coords[:, 1]] = color

    holes = _closed_holes(settled)
    if not holes:
        return None
    return settled, records


def _shifted_coords(coords: np.ndarray, side: str, offset: int) -> np.ndarray:
    out = coords.copy()
    out[:, 1] += offset if side == "left" else -offset
    return out


def _touches_existing(coords: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr, dc in DIR8 + [(0, 0)]:
            if (int(r + dr), int(c + dc)) in occupied:
                return True
    return False


def _distance_score(coords: np.ndarray, side: str, W: int) -> int:
    _r0, c0, _r1, c1 = _bbox(coords)
    if side == "left":
        return int(c0 + c1)
    return int((2 * (W - 1)) - (c0 + c1))


def _scatter_input_from_settled(settled: np.ndarray, records: list[dict]) -> np.ndarray | None:
    H, W = settled.shape
    inp = np.full_like(settled, BG)
    inp[:, 0] = LEFT
    inp[:, W - 1] = RIGHT
    occupied = {(r, 0) for r in range(H)} | {(r, W - 1) for r in range(H)}

    side_scores: dict[str, list[tuple[str, int]]] = {"left": [], "right": []}
    shuffled = list(range(len(records)))
    random.shuffle(shuffled)
    shuffled.sort(key=lambda i: 0 if records[i]["role"] == "anchor" else (1 if records[i]["role"] == "blocker" else 2))

    anchor_scores: dict[str, list[int]] = {"left": [], "right": []}
    blocker_scores: dict[str, list[int]] = {"left": [], "right": []}

    for idx in shuffled:
        rec = records[idx]
        side = rec["side"]
        coords = rec["coords"]
        color = LEFT if side == "left" else RIGHT
        c_min = int(coords[:, 1].min())
        c_max = int(coords[:, 1].max())
        max_offset = (W - 2 - c_max) if side == "left" else (c_min - 1)
        if max_offset < 1:
            return None

        candidates = list(range(1, max_offset + 1))
        if rec["role"] == "anchor":
            candidates = [d for d in candidates if d <= min(3, max_offset)]
        elif rec["role"] == "blocker":
            candidates = [d for d in candidates if d >= 2]
        random.shuffle(candidates)

        placed = False
        for offset in candidates:
            shifted = _shifted_coords(coords, side, offset)
            if int(shifted[:, 1].min()) <= 1 or int(shifted[:, 1].max()) >= W - 2:
                continue
            if _touches_existing(shifted, occupied):
                continue
            score = _distance_score(shifted, side, W)
            existing_scores = {s for _role, s in side_scores[side]}
            if score in existing_scores:
                continue
            if rec["role"] == "blocker" and anchor_scores[side] and score <= max(anchor_scores[side]):
                continue
            if rec["role"] == "anchor" and blocker_scores[side] and score >= min(blocker_scores[side]):
                continue
            inp[shifted[:, 0], shifted[:, 1]] = color
            occupied.update((int(r), int(c)) for r, c in shifted)
            side_scores[side].append((rec["role"], score))
            if rec["role"] == "anchor":
                anchor_scores[side].append(score)
            elif rec["role"] == "blocker":
                blocker_scores[side].append(score)
            placed = True
            break
        if not placed:
            return None
    return inp


def _cyan_targets(settled: np.ndarray, holes: list[np.ndarray]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    H, W = settled.shape
    hole_mask = np.zeros((H, W), dtype=bool)
    for hole in holes:
        hole_mask[hole[:, 0], hole[:, 1]] = True
    cyan = set()
    plain = set()
    for r in range(H):
        for c in range(1, W - 1):
            if int(settled[r, c]) not in (LEFT, RIGHT):
                continue
            target = plain
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and bool(hole_mask[nr, nc]):
                    target = cyan
                    break
            target.add((r, c))
    plain -= cyan
    return cyan, plain


def _has_blocked_object(records: list[dict]) -> bool:
    return any(rec["role"] == "blocker" for rec in records)


def _semantic_checks(inp: np.ndarray, settled: np.ndarray, records: list[dict]) -> bool:
    if _closed_holes(inp):
        return False
    holes = _closed_holes(settled)
    if not holes:
        return False
    cyan, plain = _cyan_targets(settled, holes)
    if not cyan or not plain:
        return False
    if not _has_blocked_object(records):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _build_settled_layout()
    if layout is None:
        return None
    settled, records = layout
    inp = _scatter_input_from_settled(settled, records)
    if inp is None:
        return None
    if not _semantic_checks(inp, settled, records):
        return None
    expected, _holes = _finalize_settled(settled)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
