from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
TILE_SIZE = 5
INNER_SIZE = 3
LEGEND_GAP = 1
MAIN_LEFT = 7
MAX_GEN_ATTEMPTS = 500

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    assert bool(m.any())
    rs, cs = np.where(m)
    return m[int(rs.min()) : int(rs.max()) + 1, int(cs.min()) : int(cs.max()) + 1]


def _mask_from_points(pts: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(pts)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        m[int(r) - r0, int(c) - c0] = True
    return m


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    m = np.asarray(mask, dtype=bool)
    return int(m.shape[0]), int(m.shape[1]), m.tobytes()


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim_mask(mask)
    seen: set[tuple[int, int, bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        m0 = np.fliplr(base) if flip else base
        for k in range(4):
            m = _trim_mask(np.rot90(m0, k))
            key = _mask_key(m)
            if key not in seen:
                seen.add(key)
                out.append(m)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_key(m) for m in _orientations(mask))


def _extract_legend(g: np.ndarray) -> tuple[dict[tuple[int, int, bytes], int], np.ndarray]:
    h, w = g.shape
    assert w >= TILE_SIZE
    templates: dict[tuple[int, int, bytes], int] = {}
    legend_mask = np.zeros((h, w), dtype=bool)

    r = 0
    while r <= h - TILE_SIZE:
        tile = g[r : r + TILE_SIZE, 0:TILE_SIZE]
        ring_vals = []
        for rr in range(TILE_SIZE):
            for cc in range(TILE_SIZE):
                if rr in (0, TILE_SIZE - 1) or cc in (0, TILE_SIZE - 1):
                    ring_vals.append(int(tile[rr, cc]))
        color = ring_vals[0]
        if color != BLACK and color != GREY and all(v == color for v in ring_vals):
            inner = tile[1:4, 1:4]
            if bool(np.all((inner == BLACK) | (inner == color))) and bool(np.any(inner == color)):
                key = _canonical_key(inner == color)
                assert key not in templates, "legend templates must be unique up to rotation/reflection"
                templates[key] = int(color)
                legend_mask[r : r + TILE_SIZE, 0:TILE_SIZE] = True
                r += TILE_SIZE
                continue
        r += 1

    assert templates, "expected at least one left-edge legend block"
    return templates, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey hollow objects by matching their hole shapes to a left-edge template legend.

    Input structure:
    - Black is the background and Grey (5) forms the foreground objects in the main area.
    - The left edge contains a vertical legend: each 5x5 tile has a colored outer frame and a 3x3 black interior
      where same-colored pixels draw a small template shape.
    - Each Grey object is 4-connected and contains one or more closed Black holes.

    Rule:
    - Read each legend tile as a mapping from its 3x3 internal template mask to the tile frame color.
    - Find every closed Black hole inside each Grey object and match its normalized shape to a legend template,
      allowing rotations and reflections.
    - Fill each hole with the matched template color, recolor the whole Grey object with the color matched by
      its largest hole, and erase the left-edge legend tiles to Black.

    Key insight: The missing interior shape, not the outer Grey silhouette, is the visible key that selects the object's color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    templates, legend_mask = _extract_legend(g)

    grey_components = _components(g == GREY)
    grey_labels = np.zeros((h, w), dtype=int)
    for idx, pts in enumerate(grey_components, start=1):
        for r, c in pts:
            grey_labels[r, c] = int(idx)

    holes_by_object: dict[int, list[list[tuple[int, int]]]] = {i: [] for i in range(1, len(grey_components) + 1)}
    for pts in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        adjacent_grey: set[int] = set()
        touches_foreign_color = False
        for r, c in pts:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(g[rr, cc]) == BLACK:
                    continue
                lab = int(grey_labels[rr, cc])
                if lab:
                    adjacent_grey.add(lab)
                else:
                    touches_foreign_color = True
        if len(adjacent_grey) == 1 and not touches_foreign_color:
            obj_id = next(iter(adjacent_grey))
            holes_by_object[obj_id].append(pts)

    out = g.copy()
    out[legend_mask] = BLACK

    for obj_id, obj_pts in enumerate(grey_components, start=1):
        hole_infos: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
        for hole_pts in holes_by_object.get(obj_id, []):
            hole_mask = _mask_from_points(hole_pts)
            key = _canonical_key(hole_mask)
            assert key in templates, "every object hole must match a legend template"
            color = int(templates[key])
            r0, c0, _r1, _c1 = _bbox(hole_pts)
            hole_infos.append((len(hole_pts), -r0, -c0, color, hole_pts))
            for r, c in hole_pts:
                out[r, c] = color

        assert hole_infos, "every grey object should contain at least one matched hole"
        object_color = max(hole_infos, key=lambda item: (item[0], item[1], item[2]))[3]
        for r, c in obj_pts:
            out[r, c] = int(object_color)

    return out


def _shape_library() -> list[np.ndarray]:
    raw = [
        [[1]],
        [[1, 1]],
        [[1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1], [1, 0], [1, 1]],
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
    ]
    seen: set[tuple[int, int, bytes]] = set()
    shapes: list[np.ndarray] = []
    for pat in raw:
        m = np.array(pat, dtype=bool)
        if len(_components(m)) != 1:
            continue
        key = _canonical_key(m)
        if key not in seen:
            seen.add(key)
            shapes.append(m)
    return shapes


SHAPES = _shape_library()


def _legend_tile(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    tile[0, :] = color
    tile[-1, :] = color
    tile[:, 0] = color
    tile[:, -1] = color

    m = _trim_mask(mask)
    ih, iw = m.shape
    off_r = random.randint(0, INNER_SIZE - ih)
    off_c = random.randint(0, INNER_SIZE - iw)
    inner = tile[1:4, 1:4]
    inner[off_r : off_r + ih, off_c : off_c + iw][m] = color
    return tile


def _closed_hole_keys(mask: np.ndarray) -> Counter:
    h, w = mask.shape
    result: Counter = Counter()
    for pts in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        result[_canonical_key(_mask_from_points(pts))] += 1
    return result


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _can_remove_edge_cell(mask: np.ndarray, r: int, c: int, wanted: Counter) -> bool:
    test = mask.copy()
    test[r, c] = False
    return bool(test.any()) and _is_connected(test) and _closed_hole_keys(test) == wanted


def _trim_object_edges(mask: np.ndarray, wanted: Counter) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for _ in range(random.randint(0, 6)):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if bool(out[r, c]) and (r in (0, h - 1) or c in (0, w - 1))
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            if _can_remove_edge_cell(out, r, c, wanted):
                out[r, c] = False
                break
    return out


def _place_holes_in_object(hole_masks: list[np.ndarray]) -> np.ndarray | None:
    hole_keys = Counter(_canonical_key(m) for m in hole_masks)
    for _ in range(80):
        n = len(hole_masks)
        obj_h = random.randint(5 if n == 1 else 7, 10 if n == 1 else 12)
        obj_w = random.randint(5 if n == 1 else 8, 11 if n == 1 else 15)

        holes = np.zeros((obj_h, obj_w), dtype=bool)
        blocked = np.zeros((obj_h, obj_w), dtype=bool)
        placed = True
        ordered = sorted(hole_masks, key=lambda m: -int(np.sum(m)))

        for m0 in ordered:
            m = random.choice(_orientations(m0))
            mh, mw = m.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, obj_h - mh):
                for c0 in range(1, obj_w - mw):
                    cells = [(r0 + r, c0 + c) for r, c in zip(*np.where(m), strict=False)]
                    if any(bool(blocked[r, c]) for r, c in cells):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                placed = False
                break
            r0, c0 = random.choice(candidates)
            rr, cc = np.where(m)
            holes[r0 + rr, c0 + cc] = True
            for r, c in zip(r0 + rr, c0 + cc, strict=False):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = int(r) + dr, int(c) + dc
                        if 0 <= nr < obj_h and 0 <= nc < obj_w:
                            blocked[nr, nc] = True

        if not placed:
            continue

        obj = np.ones((obj_h, obj_w), dtype=bool)
        obj[holes] = False
        if not _is_connected(obj) or _closed_hole_keys(obj) != hole_keys:
            continue
        obj = _trim_object_edges(obj, hole_keys)
        if _is_connected(obj) and _closed_hole_keys(obj) == hole_keys:
            return obj
    return None


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < MAIN_LEFT or r0 + h > H or c0 + w > W:
        return False
    region = occupied[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)]
    return not bool(region.any())


def _choose_templates() -> tuple[list[np.ndarray], list[int]]:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(100):
        n = random.randint(2, 4)
        shapes = random.sample(SHAPES, n)
        if len({int(np.sum(s)) for s in shapes}) < 2:
            continue
        random.shuffle(colors)
        return shapes, colors[:n]
    raise RuntimeError("failed to choose distinct template shapes")


def _make_plans(num_templates: int, shapes: list[np.ndarray]) -> list[list[int]]:
    areas = [int(np.sum(s)) for s in shapes]
    diff_pairs = [(i, j) for i in range(num_templates) for j in range(i + 1, num_templates) if areas[i] != areas[j]]
    assert diff_pairs
    first_pair = list(random.choice(diff_pairs))
    plans: list[list[int]] = [first_pair]

    used = set(first_pair)
    remaining = [i for i in range(num_templates) if i not in used]
    for idx in remaining:
        plans.append([idx])

    max_objects = min(5, num_templates + 2)
    while len(plans) < max_objects and random.random() < 0.55:
        if random.random() < 0.45 and diff_pairs:
            plans.append(list(random.choice(diff_pairs)))
        else:
            plans.append([random.randrange(num_templates)])

    random.shuffle(plans)
    return plans


def _place_objects(grid: np.ndarray, occupied: np.ndarray, patches: list[np.ndarray]) -> bool:
    H, W = grid.shape
    random.shuffle(patches)
    patches.sort(key=lambda p: -int(p.shape[0] * p.shape[1]))
    for patch in patches:
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, H - ph):
            for c0 in range(MAIN_LEFT, W - pw):
                if _can_place(occupied, patch, r0, c0):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw][patch] = GREY
        occupied[max(0, r0 - 1) : min(H, r0 + ph + 1), max(0, c0 - 1) : min(W, c0 + pw + 1)] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    shapes, colors = _choose_templates()
    n_templates = len(shapes)
    legend_h = n_templates * TILE_SIZE + (n_templates - 1) * LEGEND_GAP
    H = random.randint(max(16, legend_h + 1), 30)
    W = random.randint(22, 30)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    top = random.randint(0, H - legend_h)
    r = top
    for shape, color in zip(shapes, colors, strict=False):
        tile = _legend_tile(shape, int(color))
        grid[r : r + TILE_SIZE, 0:TILE_SIZE] = tile
        occupied[max(0, r - 1) : min(H, r + TILE_SIZE + 1), 0 : MAIN_LEFT] = True
        r += TILE_SIZE + LEGEND_GAP

    occupied[:, 0:MAIN_LEFT] = True

    plans = _make_plans(n_templates, shapes)
    patches: list[np.ndarray] = []
    for plan in plans:
        hole_masks = [shapes[i] for i in plan]
        patch = _place_holes_in_object(hole_masks)
        if patch is None:
            return None
        patches.append(patch)

    if not _place_objects(grid, occupied, patches):
        return None

    if int(np.sum(grid == BLACK)) <= int(np.sum(grid == GREY)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed2047415201/1 input")
