from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

BIG_COLORS = (BLUE, GREEN, YELLOW)
SMALL_COLORS = (PINK, ORANGE, TEAL)
GENERATION_RETRIES = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            coords: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use contained 2x2 inner objects as local cross-erasers inside their enclosing large blocks.

    Input structure:
    - The grid has a black background plus several 8-connected monochromatic foreground objects.
    - Large non-red objects (area > 20) are irregular block-like canvases, and tiny non-red objects (area <= 4) may sit inside some large-object bounding boxes.
    - Red pixels may be scattered inside a large canvas bbox as protected specks.

    Rule:
    - Find all 8-connected monochromatic non-black components.
    - Any tiny non-red component whose bbox is fully contained in a large component's bbox becomes an eraser for that large component.
    - For each large component, erase only inside that component's bbox: on every row and column occupied by one of its erasers, recolor that large component's own color to black, keep red pixels unchanged, and finally clear the eraser pixels themselves to black.

    Key insight:
    - Bounding-box containment identifies the small object's role as a tool, not a deletion target, and the cross erase is scoped to the containing canvas rather than the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components(g)
    bigs = [obj for obj in objects if obj["area"] > 20]
    smalls = [obj for obj in objects if obj["area"] <= 4 and obj["color"] != RED]

    out = g.copy()
    for big in bigs:
        erasers = [obj for obj in smalls if _bbox_contains(big["bbox"], obj["bbox"])]
        if not erasers:
            continue

        r0, c0, r1, c1 = big["bbox"]
        rows = sorted({int(r) for er in erasers for r in er["coords"][:, 0]})
        cols = sorted({int(c) for er in erasers for c in er["coords"][:, 1]})

        for rr in rows:
            for cc in range(c0, c1 + 1):
                if int(out[rr, cc]) == int(big["color"]):
                    out[rr, cc] = BLACK
        for cc in cols:
            for rr in range(r0, r1 + 1):
                if int(out[rr, cc]) == int(big["color"]):
                    out[rr, cc] = BLACK

        for er in erasers:
            coords = er["coords"]
            out[coords[:, 0], coords[:, 1]] = BLACK

    return out


def _choose_hole_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(candidates)
    holes: list[tuple[int, int]] = []
    for r, c in candidates:
        rows = {r, r + 1}
        cols = {c, c + 1}
        if any(rows & {hr, hr + 1} or cols & {hc, hc + 1} for hr, hc in holes):
            continue
        holes.append((r, c))
        if len(holes) == count:
            return holes
    return None


def _carve_irregular(mask: np.ndarray, max_remove: int) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    remove_budget = random.randint(1, max_remove)
    candidates = [(r, c) for r in range(h) for c in range(w) if out[r, c]]
    random.shuffle(candidates)
    removed = 0
    for r, c in candidates:
        if removed >= remove_budget:
            break
        if not out[r, c]:
            continue
        if 0 < r < h - 1 and 0 < c < w - 1 and random.random() < 0.7:
            continue
        trial = out.copy()
        trial[r, c] = False
        if np.min(trial.sum(axis=0)) < 4 or np.min(trial.sum(axis=1)) < 4:
            continue
        if not _mask_connected(trial):
            continue
        out = trial
        removed += 1
    return out


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(int(x) for x in p) for p in np.argwhere(mask)}


def _build_big_object(h: int, w: int, eraser_count: int) -> dict | None:
    holes = _choose_hole_positions(h, w, eraser_count)
    if holes is None:
        return None

    body = np.ones((h, w), dtype=bool)
    eraser_coords_local: list[np.ndarray] = []
    for r, c in holes:
        body[r : r + 2, c : c + 2] = False
        eraser_coords_local.append(np.array([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)], dtype=int))

    body = _carve_irregular(body, max_remove=min(6, max(2, h + w - 8)))
    if not _mask_connected(body):
        return None

    selected_rows = sorted({int(r) for arr in eraser_coords_local for r in arr[:, 0]})
    selected_cols = sorted({int(c) for arr in eraser_coords_local for c in arr[:, 1]})
    if any(int(body[r].sum()) < 4 for r in selected_rows):
        return None
    if any(int(body[:, c].sum()) < 4 for c in selected_cols):
        return None

    big_cells = [tuple(int(x) for x in p) for p in np.argwhere(body)]
    cross_cells = [p for p in big_cells if p[0] in selected_rows or p[1] in selected_cols]
    if not cross_cells:
        return None

    red_count = random.randint(3, 5)
    reds: list[tuple[int, int]] = []
    candidates = cross_cells[:]
    random.shuffle(candidates)
    for cell in candidates:
        reds = [cell]
        break
    if not reds:
        return None

    all_candidates = big_cells[:]
    random.shuffle(all_candidates)
    for cell in all_candidates:
        if len(reds) >= red_count:
            break
        if cell in reds:
            continue
        trial_reds = reds + [cell]
        trial_body = body.copy()
        for rr, cc in trial_reds:
            trial_body[rr, cc] = False
        if not _mask_connected(trial_body):
            continue
        if any(int(trial_body[r].sum()) < 3 for r in selected_rows):
            continue
        if any(int(trial_body[:, c].sum()) < 3 for c in selected_cols):
            continue
        reds = trial_reds

    if len(reds) < 3:
        return None

    final_body = body.copy()
    for rr, cc in reds:
        final_body[rr, cc] = False
    area = int(final_body.sum())
    if not (30 <= area <= 60):
        return None
    if not _mask_connected(final_body):
        return None
    if any(int(final_body[r].sum()) < 3 for r in selected_rows):
        return None
    if any(int(final_body[:, c].sum()) < 3 for c in selected_cols):
        return None

    return {
        "body_mask": final_body,
        "reds_local": [tuple(map(int, p)) for p in reds],
        "erasers_local": eraser_coords_local,
        "selected_rows": selected_rows,
        "selected_cols": selected_cols,
        "area": area,
    }


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_big_bboxes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(200):
        boxes.clear()
        sizes = []
        for _idx in range(count):
            bh = random.randint(6, 8)
            bw = random.randint(6, 9)
            sizes.append((bh, bw))
        random.shuffle(sizes)
        ok = True
        for bh, bw in sizes:
            placed = False
            spots = [(r, c) for r in range(0, h - bh + 1) for c in range(0, w - bw + 1)]
            random.shuffle(spots)
            for r, c in spots:
                bbox = (r, c, r + bh - 1, c + bw - 1)
                if any(_expanded_overlap(bbox, other, pad=2) for other in boxes):
                    continue
                boxes.append(bbox)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return boxes[:]
    return None


def _apply_metadata_rule(grid: np.ndarray, bigs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    for big in bigs:
        r0, c0, r1, c1 = big["bbox"]
        for rr in big["rows"]:
            for cc in range(c0, c1 + 1):
                if int(out[rr, cc]) == int(big["color"]):
                    out[rr, cc] = BLACK
        for cc in big["cols"]:
            for rr in range(r0, r1 + 1):
                if int(out[rr, cc]) == int(big["color"]):
                    out[rr, cc] = BLACK
        for eraser in big["erasers"]:
            out[eraser[:, 0], eraser[:, 1]] = BLACK
    return out


def _wrong_delete_only(grid: np.ndarray, bigs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    for big in bigs:
        for eraser in big["erasers"]:
            out[eraser[:, 0], eraser[:, 1]] = BLACK
    return out


def _wrong_global_cross(grid: np.ndarray, bigs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    rows = sorted({rr for big in bigs for rr in big["rows"]})
    cols = sorted({cc for big in bigs for cc in big["cols"]})
    for rr in rows:
        for cc in range(out.shape[1]):
            if int(out[rr, cc]) != RED:
                out[rr, cc] = BLACK
    for cc in cols:
        for rr in range(out.shape[0]):
            if int(out[rr, cc]) != RED:
                out[rr, cc] = BLACK
    return out


def _wrong_no_red_protection(grid: np.ndarray, bigs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    for big in bigs:
        r0, c0, r1, c1 = big["bbox"]
        for rr in big["rows"]:
            out[rr, c0 : c1 + 1] = np.where(out[rr, c0 : c1 + 1] == BLACK, BLACK, BLACK)
        for cc in big["cols"]:
            out[r0 : r1 + 1, cc] = np.where(out[r0 : r1 + 1, cc] == BLACK, BLACK, BLACK)
        for eraser in big["erasers"]:
            out[eraser[:, 0], eraser[:, 1]] = BLACK
    return out


def _cells_clear(grid: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid.shape[0], r0 + h + pad)
    cc1 = min(grid.shape[1], c0 + w + pad)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BLACK))


def _place_aligned_distractor(grid: np.ndarray, bigs: list[dict], preferred_colors: list[int]) -> bool:
    rows = sorted({rr for big in bigs for rr in big["rows"]})
    cols = sorted({cc for big in bigs for cc in big["cols"]})
    row_first = random.random() < 0.5
    color = random.choice(preferred_colors) if preferred_colors else random.choice(SMALL_COLORS)

    positions: list[tuple[int, int]] = []
    if row_first:
        positions.extend((r, c) for r in rows for c in range(grid.shape[1] - 1))
        positions.extend((r, c) for r in range(grid.shape[0] - 1) for c in cols)
    else:
        positions.extend((r, c) for r in range(grid.shape[0] - 1) for c in cols)
        positions.extend((r, c) for r in rows for c in range(grid.shape[1] - 1))
    random.shuffle(positions)

    for r, c in positions:
        if r + 1 >= grid.shape[0] or c + 1 >= grid.shape[1]:
            continue
        if not _cells_clear(grid, r, c, 2, 2, pad=1):
            continue
        bbox = (r, c, r + 1, c + 1)
        if any(_bbox_contains(big["bbox"], bbox) for big in bigs):
            continue
        grid[r : r + 2, c : c + 2] = color
        return True
    return False


def _place_extra_distractors(grid: np.ndarray, bigs: list[dict]) -> None:
    shapes = (
        np.array([(0, 0)], dtype=int),
        np.array([(0, 0), (0, 1)], dtype=int),
        np.array([(0, 0), (1, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    )
    for _ in range(random.randint(0, 2)):
        order = list(shapes)
        random.shuffle(order)
        placed = False
        for shape in order:
            sh = int(shape[:, 0].max()) + 1
            sw = int(shape[:, 1].max()) + 1
            spots = [(r, c) for r in range(grid.shape[0] - sh + 1) for c in range(grid.shape[1] - sw + 1)]
            random.shuffle(spots)
            for r, c in spots:
                if not _cells_clear(grid, r, c, sh, sw, pad=1):
                    continue
                coords = shape + np.array([r, c], dtype=int)
                bbox = _bbox_coords(coords)
                if any(_bbox_contains(big["bbox"], bbox) for big in bigs):
                    continue
                color = random.choice(SMALL_COLORS)
                grid[coords[:, 0], coords[:, 1]] = color
                placed = True
                break
            if placed:
                break


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(22, 29)
    w = random.randint(22, 29)
    grid = np.zeros((h, w), dtype=int)

    big_count = random.choice((2, 2, 3))
    boxes = _place_big_bboxes(h, w, big_count)
    if boxes is None:
        return None

    random.shuffle(boxes)
    big_colors = random.sample(list(BIG_COLORS), big_count)
    bigs: list[dict] = []
    eraser_palette_used: list[int] = []

    for bbox, color in zip(boxes, big_colors):
        r0, c0, r1, c1 = bbox
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        eraser_count = random.choice((1, 2))
        built = _build_big_object(bh, bw, eraser_count)
        if built is None:
            return None

        body_mask = built["body_mask"]
        grid[r0 : r1 + 1, c0 : c1 + 1][body_mask] = color
        for rr, cc in built["reds_local"]:
            grid[r0 + rr, c0 + cc] = RED

        erasers_global: list[np.ndarray] = []
        for er in built["erasers_local"]:
            er_global = er + np.array([r0, c0], dtype=int)
            er_color = random.choice([c for c in SMALL_COLORS if c != color])
            grid[er_global[:, 0], er_global[:, 1]] = er_color
            erasers_global.append(er_global)
            eraser_palette_used.append(er_color)

        bigs.append(
            {
                "bbox": bbox,
                "color": color,
                "erasers": erasers_global,
                "rows": sorted({int(r) for er in erasers_global for r in er[:, 0]}),
                "cols": sorted({int(c) for er in erasers_global for c in er[:, 1]}),
            }
        )

    if not _place_aligned_distractor(grid, bigs, eraser_palette_used):
        return None
    _place_extra_distractors(grid, bigs)

    expected = _apply_metadata_rule(grid, bigs)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(_wrong_delete_only(grid, bigs), expected):
        return None
    if np.array_equal(_wrong_global_cross(grid, bigs), expected):
        return None
    if np.array_equal(_wrong_no_red_protection(grid, bigs), expected):
        return None

    components = _find_components(grid)
    big_components = sorted([obj for obj in components if obj["area"] > 20], key=lambda obj: obj["bbox"])
    if len(big_components) != big_count:
        return None
    nonred_smalls = [obj for obj in components if obj["area"] <= 4 and obj["color"] != RED]
    if len(nonred_smalls) < sum(len(big["erasers"]) for big in bigs) + 1:
        return None

    for big in bigs:
        hit_red = False
        for rr in big["rows"]:
            row = grid[rr, big["bbox"][1] : big["bbox"][3] + 1]
            if int(np.count_nonzero(row == big["color"])) < 3:
                return None
            hit_red = hit_red or bool(np.any(row == RED))
        for cc in big["cols"]:
            col = grid[big["bbox"][0] : big["bbox"][2] + 1, cc]
            if int(np.count_nonzero(col == big["color"])) < 3:
                return None
            hit_red = hit_red or bool(np.any(col == RED))
        if not hit_red:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
