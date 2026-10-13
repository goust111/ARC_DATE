from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

MAX_ATTEMPTS = 500
GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
CHAIN_ARM_RANGE = (4, 7)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    neigh = neighbors4 if connectivity == 4 else neighbors8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _color_components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[np.ndarray]:
    return _components_from_mask(grid == color, connectivity=connectivity)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _partition_labels_without_walls(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    walkable = grid != GREY
    parts = _components_from_mask(walkable, connectivity=4)
    labels = np.full(grid.shape, -1, dtype=int)
    for pid, coords in enumerate(parts):
        labels[coords[:, 0], coords[:, 1]] = pid
    return labels, parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group blue fragments by the background region enclosed by diagonal grey wall-chains, fill each group's joint box, and recolor the walls teal.

    Input structure:
    - The background is black.
    - Grey pixels form one or two unbranched diagonal chains whose consecutive cells touch only diagonally.
    - Blue pixels form several detached 4-connected rectangular fragments scattered among the wall-defined regions.

    Rule:
    - Treat grey pixels as walls and partition all non-grey cells by 4-connectivity through cells that are not grey.
    - Assign each blue fragment to the unique wall-defined region containing it.
    - For every region, take the union bbox of all of its blue fragments and recolor black cells inside that bbox to red, leaving blue cells unchanged.
    - Recolor every grey wall pixel to teal.

    Key insight:
    - The diagonal chains are not the objects to complete; they are separators that decide which disconnected blue fragments belong to the same completion group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    labels, _ = _partition_labels_without_walls(g)
    fragments = _color_components(g, BLUE, connectivity=4)

    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for frag in fragments:
        pids = {int(labels[r, c]) for r, c in frag}
        assert len(pids) == 1, "each blue fragment must lie in one wall-defined region"
        pid = next(iter(pids))
        groups[pid].append(frag)

    out = g.copy()
    for frags in groups.values():
        merged = np.concatenate(frags, axis=0)
        r0, c0, r1, c1 = _bbox_coords(merged)
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[window == BG] = RED

    out[out == GREY] = TEAL
    return out


def _make_corner_chain(h: int, w: int, corner: str, arm: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(arm - i, i) for i in range(arm + 1)]
    if corner == "tr":
        return [(i, w - 1 - arm + i) for i in range(arm + 1)]
    if corner == "br":
        return [(h - 1 - arm + i, w - 1 - i) for i in range(arm + 1)]
    if corner == "bl":
        return [(h - 1 - arm + i, i) for i in range(arm + 1)]
    raise ValueError(f"unknown corner {corner}")


def _build_wall_grid(h: int, w: int, specs: list[tuple[str, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for corner, arm in specs:
        for r, c in _make_corner_chain(h, w, corner, arm):
            grid[r, c] = GREY
    return grid


def _rect_candidates(region_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = region_mask.shape
    candidates = []
    sizes = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    for rh, rw in sizes:
        for r0 in range(h - rh + 1):
            for c0 in range(w - rw + 1):
                block = region_mask[r0 : r0 + rh, c0 : c0 + rw]
                if block.all():
                    candidates.append((r0, c0, r0 + rh - 1, c0 + rw - 1))
    return candidates


def _bbox_cells(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
    return np.column_stack([rr.ravel(), cc.ravel()])


def _candidate_fits(forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not forbidden[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve(forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = forbidden.shape
    r0, c0, r1, c1 = bbox
    forbidden[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _pick_region_boxes(
    region_mask: np.ndarray,
    forbidden: np.ndarray,
    count: int,
    *,
    require_disjoint_pair: bool,
) -> list[tuple[int, int, int, int]] | None:
    candidates = _rect_candidates(region_mask)
    if len(candidates) < count:
        return None

    for _ in range(120):
        local_forbidden = forbidden.copy()
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int, int, int]] = []

        if require_disjoint_pair:
            pairs = []
            for i, a in enumerate(order):
                if not _candidate_fits(local_forbidden, a):
                    continue
                pair_forbidden = local_forbidden.copy()
                _reserve(pair_forbidden, a)
                for b in order[i + 1 :]:
                    if not _candidate_fits(pair_forbidden, b):
                        continue
                    if not (
                        _spans_overlap(a[0], a[2], b[0], b[2])
                        or _spans_overlap(a[1], a[3], b[1], b[3])
                    ):
                        pairs.append((a, b))
            if not pairs:
                return None
            a, b = random.choice(pairs)
            chosen.extend([a, b])
            _reserve(local_forbidden, a)
            _reserve(local_forbidden, b)

        while len(chosen) < count:
            valid = [cand for cand in order if _candidate_fits(local_forbidden, cand)]
            if not valid:
                break
            weights = []
            for cand in valid:
                area = (cand[2] - cand[0] + 1) * (cand[3] - cand[1] + 1)
                weights.append(1 if area == 1 else 3 if area <= 4 else 2)
            cand = random.choices(valid, weights=weights, k=1)[0]
            chosen.append(cand)
            _reserve(local_forbidden, cand)

        if len(chosen) != count:
            continue

        merged_coords = np.concatenate([_bbox_cells(b) for b in chosen], axis=0)
        r0, c0, r1, c1 = _bbox_coords(merged_coords)
        total_blue = sum((b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in chosen)
        if (r1 - r0 + 1) * (c1 - c0 + 1) <= total_blue:
            continue
        return chosen

    return None


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _grey_chains_are_valid(grid: np.ndarray) -> bool:
    chains = _color_components(grid, GREY, connectivity=8)
    if not (1 <= len(chains) <= 2):
        return False
    for chain in chains:
        chain_set = {tuple(map(int, cell)) for cell in chain}
        if len(chain_set) < 4 or len(chain_set) > 8:
            return False
        for r, c in chain_set:
            orth = sum((nr, nc) in chain_set for nr, nc in neighbors4(r, c))
            diag = sum((nr, nc) in chain_set for nr, nc in neighbors8(r, c))
            if orth != 0 or diag > 2:
                return False
        ends = 0
        for r, c in chain_set:
            deg = sum((nr, nc) in chain_set for nr, nc in neighbors8(r, c))
            if deg == 1:
                ends += 1
            elif deg != 2:
                return False
        if ends != 2:
            return False
    return True


def _validate_input(grid: np.ndarray, expected_regions: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, BLUE, GREY}:
        return False
    if not _grey_chains_are_valid(grid):
        return False

    labels, parts = _partition_labels_without_walls(grid)
    if len(parts) != expected_regions:
        return False

    fragments = _color_components(grid, BLUE, connectivity=4)
    if len(fragments) < 2 * expected_regions:
        return False

    counts: dict[int, int] = defaultdict(int)
    bboxes_by_region: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for frag in fragments:
        bbox = _bbox_coords(frag)
        area = frag.shape[0]
        if area != (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1):
            return False
        pids = {int(labels[r, c]) for r, c in frag}
        if len(pids) != 1:
            return False
        pid = next(iter(pids))
        counts[pid] += 1
        bboxes_by_region[pid].append(bbox)

    if set(counts) != set(range(expected_regions)):
        return False
    if any(counts[pid] < 2 for pid in range(expected_regions)):
        return False

    same_region_disjoint = False
    for pid, boxes in bboxes_by_region.items():
        merged = np.concatenate([_bbox_cells(b) for b in boxes], axis=0)
        r0, c0, r1, c1 = _bbox_coords(merged)
        blue_area = sum((b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in boxes)
        if (r1 - r0 + 1) * (c1 - c0 + 1) <= blue_area:
            return False
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                if not _spans_overlap(a[0], a[2], b[0], b[2]) and not _spans_overlap(a[1], a[3], b[1], b[3]):
                    same_region_disjoint = True

    cross_region_overlap = False
    flat = [(pid, box) for pid, boxes in bboxes_by_region.items() for box in boxes]
    for i, (pid_a, a) in enumerate(flat):
        for pid_b, b in flat[i + 1 :]:
            if pid_a == pid_b:
                continue
            if _spans_overlap(a[0], a[2], b[0], b[2]) or _spans_overlap(a[1], a[3], b[1], b[3]):
                cross_region_overlap = True

    if not same_region_disjoint or not cross_region_overlap:
        return False

    out = main(grid)
    if not np.any(out == RED):
        return False
    if np.any(out[grid == GREY] != TEAL):
        return False
    if np.any(out[grid == BLUE] != BLUE):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    max_arm = min(CHAIN_ARM_RANGE[1], h - 6, w - 6)
    if max_arm < CHAIN_ARM_RANGE[0]:
        return None

    if random.random() < 0.5:
        specs = [(random.choice(["tl", "tr", "br", "bl"]), random.randint(CHAIN_ARM_RANGE[0], max_arm))]
        expected_regions = 2
    else:
        corner_pair = random.choice([("tl", "br"), ("tr", "bl")])
        specs = [(corner_pair[0], random.randint(CHAIN_ARM_RANGE[0], max_arm))]
        specs.append((corner_pair[1], random.randint(CHAIN_ARM_RANGE[0], max_arm)))
        expected_regions = 3

    grid = _build_wall_grid(h, w, specs)
    labels, parts = _partition_labels_without_walls(grid)
    if len(parts) != expected_regions:
        return None

    forbidden = np.zeros((h, w), dtype=bool)
    boxes_by_region: dict[int, list[tuple[int, int, int, int]]] = {}
    special_region = random.randrange(expected_regions)

    region_order = list(range(expected_regions))
    random.shuffle(region_order)
    for pid in region_order:
        region_mask = labels == pid
        target_count = random.choices([2, 3], weights=[3, 2], k=1)[0]
        chosen = _pick_region_boxes(
            region_mask,
            forbidden,
            target_count,
            require_disjoint_pair=(pid == special_region),
        )
        if chosen is None:
            return None
        boxes_by_region[pid] = chosen
        for bbox in chosen:
            _reserve(forbidden, bbox)

    for boxes in boxes_by_region.values():
        for bbox in boxes:
            _paint_rect(grid, bbox, BLUE)

    if not _validate_input(grid, expected_regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
