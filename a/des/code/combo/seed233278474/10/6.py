from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
OBJECT_RETRIES = 120
PLACEMENT_RETRIES = 180
GRID_MIN = 16
GRID_MAX = 30
MAX_OUTPUT_SIDE = 30


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool, copy=True)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components_mask(~mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components_mask(mask)) == 1


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=bool, copy=True)


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def _add_tabs(mask: np.ndarray, max_tabs: int) -> np.ndarray:
    work = np.pad(np.array(mask, dtype=bool, copy=True), 3)
    for _ in range(random.randint(0, max_tabs)):
        exterior = _exterior_background(work)
        anchors = [tuple(p) for p in np.argwhere(work)]
        random.shuffle(anchors)
        placed = False
        for ar, ac in anchors:
            dirs = list(DIR4)
            random.shuffle(dirs)
            for dr, dc in dirs:
                length = random.randint(1, 3)
                pts: list[tuple[int, int]] = []
                ok = True
                for step in range(1, length + 1):
                    rr = ar + dr * step
                    cc = ac + dc * step
                    if not (0 <= rr < work.shape[0] and 0 <= cc < work.shape[1]):
                        ok = False
                        break
                    if work[rr, cc] or not exterior[rr, cc]:
                        ok = False
                        break
                    neighbors = []
                    for sdr, sdc in DIR4:
                        nr, nc = rr + sdr, cc + sdc
                        if 0 <= nr < work.shape[0] and 0 <= nc < work.shape[1] and work[nr, nc]:
                            neighbors.append((nr, nc))
                    allowed = {(ar, ac)} if step == 1 else {pts[-1]}
                    if any(p not in allowed for p in neighbors):
                        ok = False
                        break
                    pts.append((rr, cc))
                if not ok:
                    continue
                for rr, cc in pts:
                    work[rr, cc] = True
                placed = True
                break
            if placed:
                break
    return _crop_mask(work)


def _make_zero_hole_mask() -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        target = random.randint(5, 16)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < target:
            candidates: dict[tuple[int, int], float] = {}
            for r, c in cells:
                for dr, dc in DIR4:
                    pt = (r + dr, c + dc)
                    if pt in cells:
                        continue
                    adj = sum((pt[0] + sdr, pt[1] + sdc) in cells for sdr, sdc in DIR4)
                    score = 3.0 * adj - 0.35 * (abs(pt[0]) + abs(pt[1])) + random.random()
                    if pt not in candidates or score > candidates[pt]:
                        candidates[pt] = score
            ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            pick_pool = ranked[: min(6, len(ranked))]
            cells.add(random.choice(pick_pool)[0])

        for _ in range(random.randint(0, 2)):
            extras = []
            for r, c in list(cells):
                for dr, dc in DIR4:
                    pt = (r + dr, c + dc)
                    if pt in cells:
                        continue
                    adj = sum((pt[0] + sdr, pt[1] + sdc) in cells for sdr, sdc in DIR4)
                    if adj >= 2:
                        extras.append(pt)
            if extras:
                cells.add(random.choice(extras))

        mask = _mask_from_cells(cells)
        if random.random() < 0.45:
            mask = _add_tabs(mask, max_tabs=1)
        mask = _random_transform(mask)
        mask = _crop_mask(mask)
        if not _is_connected(mask) or _count_holes(mask) != 0:
            continue
        if min(mask.shape) < 2 or max(mask.shape) > 8 or mask.shape[0] * mask.shape[1] > 28:
            continue
        return mask
    raise RuntimeError("failed to generate a zero-hole object")


def _make_ring_module() -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        inner_h = random.randint(1, 3)
        inner_w = random.randint(1, 3)
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        h = inner_h + top + bottom
        w = inner_w + left + right
        if h > 7 or w > 7:
            continue
        mask = np.ones((h, w), dtype=bool)
        mask[top : top + inner_h, left : left + inner_w] = False
        return mask
    raise RuntimeError("failed to generate a ring module")


def _connect_modules(modules: list[np.ndarray], horizontal: bool) -> np.ndarray:
    pad = 3
    if horizontal:
        tops = [0]
        lefts = [pad]
        for i in range(1, len(modules)):
            prev_top = tops[-1]
            prev_h = modules[i - 1].shape[0]
            cur_h = modules[i].shape[0]
            tops.append(random.randint(prev_top - cur_h + 1, prev_top + prev_h - 1))
            lefts.append(lefts[-1] + modules[i - 1].shape[1] + random.randint(1, 3))
        shift = pad - min(tops)
        tops = [t + shift for t in tops]
        height = max(t + m.shape[0] for t, m in zip(tops, modules)) + pad
        width = lefts[-1] + modules[-1].shape[1] + pad
        mask = np.zeros((height, width), dtype=bool)
        for top, left, mod in zip(tops, lefts, modules):
            h, w = mod.shape
            mask[top : top + h, left : left + w] |= mod
        for i in range(len(modules) - 1):
            top_a, top_b = tops[i], tops[i + 1]
            h_a, h_b = modules[i].shape[0], modules[i + 1].shape[0]
            row = random.randint(max(top_a, top_b), min(top_a + h_a - 1, top_b + h_b - 1))
            col_a = lefts[i] + modules[i].shape[1] - 1
            col_b = lefts[i + 1]
            mask[row, col_a : col_b + 1] = True
        return mask

    lefts = [0]
    tops = [pad]
    for i in range(1, len(modules)):
        prev_left = lefts[-1]
        prev_w = modules[i - 1].shape[1]
        cur_w = modules[i].shape[1]
        lefts.append(random.randint(prev_left - cur_w + 1, prev_left + prev_w - 1))
        tops.append(tops[-1] + modules[i - 1].shape[0] + random.randint(1, 3))
    shift = pad - min(lefts)
    lefts = [c + shift for c in lefts]
    height = tops[-1] + modules[-1].shape[0] + pad
    width = max(c + m.shape[1] for c, m in zip(lefts, modules)) + pad
    mask = np.zeros((height, width), dtype=bool)
    for top, left, mod in zip(tops, lefts, modules):
        h, w = mod.shape
        mask[top : top + h, left : left + w] |= mod
    for i in range(len(modules) - 1):
        left_a, left_b = lefts[i], lefts[i + 1]
        w_a, w_b = modules[i].shape[1], modules[i + 1].shape[1]
        col = random.randint(max(left_a, left_b), min(left_a + w_a - 1, left_b + w_b - 1))
        row_a = tops[i] + modules[i].shape[0] - 1
        row_b = tops[i + 1]
        mask[row_a : row_b + 1, col] = True
    return mask


def _make_holey_mask(holes: int) -> np.ndarray:
    if holes == 0:
        return _make_zero_hole_mask()
    for _ in range(OBJECT_RETRIES):
        modules = [_make_ring_module() for _ in range(holes)]
        mask = _connect_modules(modules, horizontal=(random.random() < 0.5))
        mask = _add_tabs(mask, max_tabs=3)
        mask = _random_transform(mask)
        mask = _crop_mask(mask)
        if not _is_connected(mask) or _count_holes(mask) != holes:
            continue
        if mask.shape[0] > 14 or mask.shape[1] > 16:
            continue
        return mask
    raise RuntimeError(f"failed to generate a {holes}-hole object")


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_cells(cells)
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for cr, cc in cells:
                sprite[cr - r0, cc - c0] = color
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                    "holes": _count_holes(sprite != bg),
                }
            )
    return objects


def _build_output(objects: list[dict], bg: int) -> np.ndarray:
    if not objects:
        return np.zeros((0, 0), dtype=int)
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[int(obj["holes"])].append(obj)

    rows: list[np.ndarray] = []
    for hole_count in sorted(groups):
        items = sorted(groups[hole_count], key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
        row_h = max(obj["sprite"].shape[0] for obj in items)
        row_w = sum(obj["sprite"].shape[1] for obj in items) + len(items) - 1
        row = np.full((row_h, row_w), bg, dtype=int)
        c = 0
        for obj in items:
            sprite = obj["sprite"]
            sh, sw = sprite.shape
            row[:sh, c : c + sw] = sprite
            c += sw + 1
        rows.append(row)

    out_h = sum(row.shape[0] for row in rows) + len(rows) - 1
    out_w = max(row.shape[1] for row in rows)
    out = np.full((out_h, out_w), bg, dtype=int)
    top = 0
    for row in rows:
        h, w = row.shape
        out[top : top + h, :w] = row
        top += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _extract_objects(g, bg)
    return _build_output(objects, bg)


def _sample_hole_multiset() -> list[int]:
    n_groups = random.choices([2, 3, 4], weights=[2, 3, 2])[0]
    groups = sorted(random.sample([0, 1, 2, 3], n_groups))
    hole_counts = list(groups)
    dup_candidates = [g for g in groups if g <= 2] or groups
    hole_counts.append(random.choice(dup_candidates))
    if n_groups <= 3 and random.random() < 0.45:
        extra_candidates = [g for g in groups if g != 3] or groups
        hole_counts.append(random.choice(extra_candidates))
    random.shuffle(hole_counts)
    return hole_counts


def _can_place(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    for dr, dc in np.argwhere(mask):
        if blocked[top + int(dr), left + int(dc)]:
            return False
    return True


def _stamp_blocked(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(mask):
        rr = top + int(dr)
        cc = left + int(dc)
        for adr in (-1, 0, 1):
            for adc in (-1, 0, 1):
                nr, nc = rr + adr, cc + adc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["mask"].shape[0] * specs[idx]["mask"].shape[1],
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        mask = spec["mask"]
        sh, sw = mask.shape
        if sh > height - 2 or sw > width - 2:
            return None
        row_lo = 1
        row_hi = height - sh - 1
        col_lo = 1
        col_hi = width - sw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            if not _can_place(blocked, mask, top, left):
                continue
            patch = grid[top : top + sh, left : left + sw]
            patch[mask] = int(spec["color"])
            _stamp_blocked(blocked, mask, top, left)
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_attempt() -> np.ndarray | None:
    hole_counts = _sample_hole_multiset()
    random.shuffle(hole_counts)
    colors = random.sample(range(1, 10), len(hole_counts))
    specs = []
    for holes, color in zip(hole_counts, colors):
        specs.append({"holes": holes, "color": color, "mask": _make_holey_mask(holes)})

    max_h = max(spec["mask"].shape[0] for spec in specs)
    max_w = max(spec["mask"].shape[1] for spec in specs)
    height_lo = max(GRID_MIN, max_h + 2)
    width_lo = max(GRID_MIN, max_w + 2)
    if height_lo > GRID_MAX or width_lo > GRID_MAX:
        return None

    height = random.randint(height_lo, GRID_MAX)
    width = random.randint(width_lo, GRID_MAX)
    grid = _place_objects(specs, height, width)
    if grid is None:
        return None

    objects = _extract_objects(grid, BG)
    if len(objects) != len(specs):
        return None

    detected_holes = sorted(obj["holes"] for obj in objects)
    if detected_holes != sorted(hole_counts):
        return None
    if len(set(detected_holes)) < 2:
        return None

    by_hole: dict[int, set[int]] = defaultdict(set)
    for obj in objects:
        c0 = int(obj["bbox"][1])
        holes = int(obj["holes"])
        if c0 in by_hole[holes]:
            return None
        by_hole[holes].add(c0)

    out = main(grid)
    if out.shape[0] > MAX_OUTPUT_SIDE or out.shape[1] > MAX_OUTPUT_SIDE:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
