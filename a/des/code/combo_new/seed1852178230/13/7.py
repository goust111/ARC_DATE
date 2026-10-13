from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
HOLE_COUNTS = (0, 1, 2, 3)

GRID_MIN = 18
GRID_MAX = 30
GEN_ATTEMPTS = 500
OBJECT_ATTEMPTS = 240
PLACEMENT_ATTEMPTS = 300
HOLE_PLACEMENT_ATTEMPTS = 200
MODIFICATION_ATTEMPTS = 80


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
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _colored_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append((color, comp))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _holes_in_local_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return len(_holes_in_local_mask(_crop_mask(mask)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw as many outer dilation rings around each foreground object as the number of holes inside it.

    Input structure:
    - The background is black.
    - The grid contains several separated single-color 4-connected foreground objects.
    - Each object may contain 0-3 enclosed black holes, while open exterior concavities can also appear on its boundary.

    Rule:
    - For each foreground object, count the 4-connected black regions inside its own bounding box that do not touch that box boundary; these are the object's enclosed holes.
    - If an object has k holes, keep the object unchanged and paint every exterior black cell at 4-neighborhood distance 1..k from the object with the object's color.
    - Treat the object's internal holes as forbidden black cells: they stay black and do not seed or receive the outer expansion.

    Key insight:
    - The controlling feature is not object size or color but the number of enclosed black cavities, and that internal count determines how thick the same-color outer halo becomes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for color, comp in _colored_components(g):
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True

        holes = _holes_in_local_mask(local)
        k = len(holes)
        if k == 0:
            continue

        forbidden = np.zeros((h, w), dtype=bool)
        for hole in holes:
            for rr, cc in hole:
                forbidden[r0 + rr, c0 + cc] = True

        dist = np.full((h, w), -1, dtype=int)
        q: deque[tuple[int, int]] = deque()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != BLACK or bool(forbidden[nr, nc]) or int(dist[nr, nc]) != -1:
                    continue
                dist[nr, nc] = 1
                q.append((nr, nc))

        while q:
            r, c = q.popleft()
            if int(dist[r, c]) >= k:
                continue
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != BLACK or bool(forbidden[nr, nc]) or int(dist[nr, nc]) != -1:
                    continue
                dist[nr, nc] = int(dist[r, c]) + 1
                q.append((nr, nc))

        out[(dist >= 1) & (dist <= k)] = color

    return out


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _base_dim_range(hole_count: int, crowded: bool) -> tuple[int, int]:
    if crowded:
        ranges = {
            0: (5, 7),
            1: (6, 8),
            2: (7, 9),
            3: (8, 10),
        }
    else:
        ranges = {
            0: (5, 8),
            1: (6, 9),
            2: (7, 10),
            3: (8, 11),
        }
    return ranges[int(hole_count)]


def _place_holes(mask: np.ndarray, pad: int, base_h: int, base_w: int, hole_count: int) -> bool:
    padded_rects: list[tuple[int, int, int, int]] = []
    for _ in range(hole_count):
        placed = False
        max_hole_h = max(1, min(3, base_h - 4))
        max_hole_w = max(1, min(3, base_w - 4))
        for _attempt in range(HOLE_PLACEMENT_ATTEMPTS):
            hole_h = random.randint(1, max_hole_h)
            hole_w = random.randint(1, max_hole_w)
            r_lo = pad + 2
            c_lo = pad + 2
            r_hi = pad + base_h - hole_h - 2
            c_hi = pad + base_w - hole_w - 2
            if r_lo > r_hi or c_lo > c_hi:
                continue
            rr = random.randint(r_lo, r_hi)
            cc = random.randint(c_lo, c_hi)
            padded = (rr - 1, rr + hole_h + 1, cc - 1, cc + hole_w + 1)
            if any(_rect_overlap(padded, prev) for prev in padded_rects):
                continue
            mask[rr : rr + hole_h, cc : cc + hole_w] = False
            padded_rects.append(padded)
            placed = True
            break
        if not placed:
            return False
    return True


def _valid_object_mask(mask: np.ndarray, hole_count: int) -> bool:
    cropped = _crop_mask(mask)
    if len(_components(cropped)) != 1:
        return False
    return _hole_count(cropped) == hole_count


def _try_add_protrusion(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    side = random.choice(("top", "bottom", "left", "right"))

    if side in {"top", "bottom"}:
        row = r0 - 1 if side == "top" else r1 + 1
        if not (0 <= row < mask.shape[0]):
            return None
        width = c1 - c0 + 1
        seg_len = random.randint(1, max(1, min(4, width)))
        start = random.randint(c0, c1 - seg_len + 1)
        out = mask.copy()
        for c in range(start, start + seg_len):
            anchor = r0 if side == "top" else r1
            if not bool(mask[anchor, c]) or bool(mask[row, c]):
                return None
            out[row, c] = True
        return out

    col = c0 - 1 if side == "left" else c1 + 1
    if not (0 <= col < mask.shape[1]):
        return None
    height = r1 - r0 + 1
    seg_len = random.randint(1, max(1, min(4, height)))
    start = random.randint(r0, r1 - seg_len + 1)
    out = mask.copy()
    for r in range(start, start + seg_len):
        anchor = c0 if side == "left" else c1
        if not bool(mask[r, anchor]) or bool(mask[r, col]):
            return None
        out[r, col] = True
    return out


def _try_carve_notch(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    side = random.choice(("top", "bottom", "left", "right"))

    if side in {"top", "bottom"}:
        row = r0 if side == "top" else r1
        width = c1 - c0 + 1
        if width <= 2:
            return None
        seg_len = random.randint(1, max(1, min(3, width - 1)))
        start = random.randint(c0, c1 - seg_len + 1)
        out = mask.copy()
        out[row, start : start + seg_len] = False
        if not np.any(out):
            return None
        return out

    col = c0 if side == "left" else c1
    height = r1 - r0 + 1
    if height <= 2:
        return None
    seg_len = random.randint(1, max(1, min(3, height - 1)))
    start = random.randint(r0, r1 - seg_len + 1)
    out = mask.copy()
    out[start : start + seg_len, col] = False
    if not np.any(out):
        return None
    return out


def _make_object_mask(hole_count: int, crowded: bool) -> np.ndarray | None:
    lo, hi = _base_dim_range(hole_count, crowded)
    for _ in range(OBJECT_ATTEMPTS):
        base_h = random.randint(lo, hi)
        base_w = random.randint(lo, hi)
        pad = 3
        mask = np.zeros((base_h + 2 * pad, base_w + 2 * pad), dtype=bool)
        mask[pad : pad + base_h, pad : pad + base_w] = True
        if not _place_holes(mask, pad, base_h, base_w, hole_count):
            continue

        accepted = 0
        target = random.randint(1, 4)
        for _attempt in range(MODIFICATION_ATTEMPTS):
            proposal = _try_add_protrusion(mask) if random.random() < 0.55 else _try_carve_notch(mask)
            if proposal is None or not np.any(proposal):
                continue
            if not _valid_object_mask(proposal, hole_count):
                continue
            mask = proposal
            accepted += 1
            if accepted >= target:
                break
        if accepted == 0:
            continue

        cropped = _crop_mask(mask)
        for _ in range(random.randint(0, 3)):
            cropped = np.rot90(cropped)
        if random.random() < 0.5:
            cropped = np.fliplr(cropped)
        if random.random() < 0.5:
            cropped = np.flipud(cropped)
        cropped = np.ascontiguousarray(cropped)

        if not _valid_object_mask(cropped, hole_count):
            continue
        if max(cropped.shape) > 12:
            continue
        if int(cropped.sum()) < 8:
            continue
        return cropped
    return None


def _sample_colors(count: int) -> list[int]:
    if random.random() < 0.35:
        return [random.choice(COLOR_POOL) for _ in range(count)]
    return random.sample(list(COLOR_POOL), count)


def _reserve_rect(top: int, left: int, mask: np.ndarray, hole_count: int) -> tuple[int, int, int, int]:
    margin = hole_count + 1
    h, w = mask.shape
    return top - margin, top + h + margin, left - margin, left + w + margin


def _sample_canvas(objects: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(obj["mask"].shape[0]) + 2 * (int(obj["holes"]) + 1) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) + 2 * (int(obj["holes"]) + 1) for obj in objects)
    total = sum(
        (int(obj["mask"].shape[0]) + 2 * (int(obj["holes"]) + 1))
        * (int(obj["mask"].shape[1]) + 2 * (int(obj["holes"]) + 1))
        for obj in objects
    )
    min_side = max(GRID_MIN, max_h, max_w, int(np.ceil(np.sqrt(total * 1.7))))
    if min_side > GRID_MAX:
        return None
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)
    return h, w


def _place_objects(objects: list[dict], grid_h: int, grid_w: int) -> list[tuple[dict, int, int]] | None:
    placements: list[tuple[dict, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []
    ordered = sorted(
        objects,
        key=lambda obj: (
            (int(obj["mask"].shape[0]) + 2 * (int(obj["holes"]) + 1))
            * (int(obj["mask"].shape[1]) + 2 * (int(obj["holes"]) + 1))
        ),
        reverse=True,
    )
    for obj in ordered:
        mask = obj["mask"]
        h, w = mask.shape
        margin = int(obj["holes"]) + 1
        if h + 2 * margin > grid_h or w + 2 * margin > grid_w:
            return None

        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            top = random.randint(margin, grid_h - h - margin)
            left = random.randint(margin, grid_w - w - margin)
            rect = _reserve_rect(top, left, mask, int(obj["holes"]))
            if any(_rect_overlap(rect, prev) for prev in reserved):
                continue
            placements.append((obj, top, left))
            reserved.append(rect)
            placed = True
            break
        if not placed:
            return None
    return placements


def _hole_counts_for_grid(n_objects: int) -> list[int]:
    counts = list(HOLE_COUNTS)
    random.shuffle(counts)
    return counts[:n_objects]


def _validate_generated_grid(grid: np.ndarray, expected_holes: list[int]) -> bool:
    comps = _colored_components(grid)
    if len(comps) != len(expected_holes):
        return False
    actual_holes: list[int] = []
    for _color, comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        actual_holes.append(_hole_count(local))
    if sorted(actual_holes) != sorted(expected_holes):
        return False
    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_objects = random.choice((2, 3))
        crowded = n_objects == 3
        hole_counts = _hole_counts_for_grid(n_objects)

        objects: list[dict] = []
        failed = False
        for holes in hole_counts:
            mask = _make_object_mask(holes, crowded)
            if mask is None:
                failed = True
                break
            objects.append({"mask": mask, "holes": holes})
        if failed:
            continue

        canvas = _sample_canvas(objects)
        if canvas is None:
            continue
        grid_h, grid_w = canvas
        placements = _place_objects(objects, grid_h, grid_w)
        if placements is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        colors = _sample_colors(n_objects)
        for (obj, top, left), color in zip(placements, colors, strict=False):
            mask = obj["mask"]
            h, w = mask.shape
            window = grid[top : top + h, left : left + w]
            window[mask] = int(color)

        if _validate_generated_grid(grid, hole_counts):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
