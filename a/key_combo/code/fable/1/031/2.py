from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, in_bounds, label_mask, neighbors4, neighbors8, paste


BLACK = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
INPUT_COLORS = (Color.BLUE, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_ids(labels: np.ndarray) -> set[int]:
    ids = set(map(int, labels[0, :]))
    ids.update(map(int, labels[-1, :]))
    ids.update(map(int, labels[:, 0]))
    ids.update(map(int, labels[:, -1]))
    ids.discard(0)
    return ids


def _hole_mask_local(component_mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    box = _bbox_from_mask(component_mask)
    local = crop_bbox(component_mask, box)
    bg_labels, n_bg = label_mask(~local, connectivity=4)
    if n_bg == 0:
        return None

    hole_ids = {k for k in range(1, n_bg + 1)} - _edge_ids(bg_labels)
    if not hole_ids:
        return None

    hole_mask = np.isin(bg_labels, list(hole_ids))
    return box, hole_mask


def _ring_infos(grid: np.ndarray) -> list[dict]:
    fg_labels, n_fg = label_mask(grid != BLACK, connectivity=4)
    infos: list[dict] = []
    for comp_id in range(1, n_fg + 1):
        mask = fg_labels == comp_id
        hole_info = _hole_mask_local(mask)
        if hole_info is None:
            continue
        box, hole_mask = hole_info
        infos.append({"mask": mask, "bbox": box, "hole_mask": hole_mask})
    return infos


def _region_parity(shape: tuple[int, int], ring_infos: list[dict]) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    ring_union = np.zeros(shape, dtype=bool)
    for info in ring_infos:
        ring_union |= info["mask"]

    region_labels, n_regions = label_mask(~ring_union, connectivity=4)
    graph = {region_id: set() for region_id in range(1, n_regions + 1)}

    for info in ring_infos:
        r0, c0, r1, c1 = info["bbox"]
        local_labels = region_labels[r0 : r1 + 1, c0 : c1 + 1]
        inside_all_ids = set(map(int, local_labels[info["hole_mask"]]))
        inside_all_ids.discard(0)

        touch_ids: set[int] = set()
        for r, c in np.argwhere(info["mask"]):
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(shape, nr, nc) and not ring_union[nr, nc]:
                    region_id = int(region_labels[nr, nc])
                    if region_id != 0:
                        touch_ids.add(region_id)

        inside_ids = touch_ids & inside_all_ids
        outside_ids = touch_ids - inside_ids
        for region_a in inside_ids:
            for region_b in outside_ids:
                graph[region_a].add(region_b)
                graph[region_b].add(region_a)

    parity: dict[int, int] = {}
    queue = deque()
    for region_id in _edge_ids(region_labels):
        parity[region_id] = 0
        queue.append(region_id)

    while queue:
        current = queue.popleft()
        next_parity = parity[current] ^ 1
        for neighbor in graph[current]:
            if neighbor not in parity:
                parity[neighbor] = next_parity
                queue.append(neighbor)
            else:
                assert parity[neighbor] == next_parity, "ring nesting parity must be consistent"

    for region_id in range(1, n_regions + 1):
        assert region_id in parity, "every non-ring region must connect to the edge through nested rings"

    return region_labels, parity, ring_union


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the black cells touching closed rims by the even/odd nesting depth of their surrounding region.

    Input structure:
    - The canvas has a black background with several colored 4-connected foreground components.
    - Some components are closed rims that enclose at least one local black region; other components are open curve distractors.
    - Closed rims may be nested inside one another, while all foreground colors themselves stay unchanged.

    Rule:
    - Detect which foreground components are closed rims by checking whether their own bounding boxes contain enclosed non-edge holes.
    - Ignore open curves when computing depth, and treat only the union of closed rims as walls that split the canvas into regions.
    - Give every non-rim region a depth parity: edge-connected regions are even, and crossing one closed rim flips parity.
    - Recolor each black cell that is 8-neighbor-adjacent to any closed rim: even-depth cells become red, odd-depth cells become green.

    Key insight:
    - The colors depend on the global parity of the black region, not on a per-ring inside/outside convention, so nested rims must share the same alternating region depth model.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ring_infos = _ring_infos(g)
    out = g.copy()
    if not ring_infos:
        return out

    region_labels, parity, ring_union = _region_parity(g.shape, ring_infos)

    for r, c in np.argwhere(g == BLACK):
        if not any(in_bounds(g.shape, nr, nc) and ring_union[nr, nc] for nr, nc in neighbors8(int(r), int(c))):
            continue
        region_id = int(region_labels[r, c])
        out[r, c] = RED if parity[region_id] == 0 else GREEN

    return out


def _draw_rect_ring(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _make_ring_family(ring_count: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(120):
        colors = random.sample(list(INPUT_COLORS), ring_count)
        patch = _draw_rect_ring(random.randint(3, 6), random.randint(3, 7), colors[0])
        ok = True
        for layer in range(1, ring_count):
            gap = random.randint(1, 3)
            new_h = patch.shape[0] + 2 * gap + 2
            new_w = patch.shape[1] + 2 * gap + 2
            if new_h > max_h or new_w > max_w:
                ok = False
                break
            grown = _draw_rect_ring(new_h, new_w, colors[layer])
            paste(grown, patch, 1 + gap, 1 + gap, transparent=BLACK)
            patch = grown
        if ok:
            return patch
    return None


def _make_broken_box(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    for _ in range(120):
        h = random.randint(5, min(10, max_h))
        w = random.randint(5, min(10, max_w))
        color = random.choice(INPUT_COLORS)
        patch = _draw_rect_ring(h, w, color)
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            gap_len = random.randint(1, min(3, w - 2))
            start = random.randint(1, w - gap_len - 1)
            row = 0 if side == "top" else h - 1
            patch[row, start : start + gap_len] = BLACK
        else:
            gap_len = random.randint(1, min(3, h - 2))
            start = random.randint(1, h - gap_len - 1)
            col = 0 if side == "left" else w - 1
            patch[start : start + gap_len, col] = BLACK

        labels, n = label_mask(patch != BLACK, connectivity=4)
        if n == 1 and _hole_mask_local(patch != BLACK) is None:
            return patch
    return None


def _make_polyline(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    local_h = min(max_h, 12)
    local_w = min(max_w, 12)
    for _ in range(160):
        canvas = np.zeros((local_h, local_w), dtype=int)
        color = random.choice(INPUT_COLORS)
        r = random.randint(1, local_h - 2)
        c = random.randint(1, local_w - 2)
        cells = [(r, c)]
        used = {(r, c)}
        direction = random.choice(DIRS4)
        segments = random.randint(2, 5)
        ok = True
        for seg in range(segments):
            length = random.randint(2, 4)
            dr, dc = direction
            for _step in range(length):
                nr = r + dr
                nc = c + dc
                if not (1 <= nr < local_h - 1 and 1 <= nc < local_w - 1):
                    ok = False
                    break
                if (nr, nc) in used:
                    ok = False
                    break
                r, c = nr, nc
                used.add((r, c))
                cells.append((r, c))
            if not ok:
                break
            if seg != segments - 1:
                turn_choices = [d for d in DIRS4 if d != direction and d != (-direction[0], -direction[1])]
                direction = random.choice(turn_choices)
        if not ok or len(cells) < 6:
            continue

        coords = np.array(cells, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
        for rr, cc in cells:
            patch[rr - r0, cc - c0] = color

        if _hole_mask_local(patch != BLACK) is not None:
            continue
        return patch
    return None


def _make_open_curve(max_h: int, max_w: int) -> np.ndarray | None:
    makers = [_make_broken_box, _make_polyline]
    random.shuffle(makers)
    for maker in makers:
        patch = maker(max_h, max_w)
        if patch is not None:
            return patch
    return None


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    return not occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if h + 2 > H or w + 2 > W:
        return False

    positions = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(occupied, patch, r0, c0):
            continue
        paste(grid, patch, r0, c0, transparent=BLACK)
        occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        return True
    return False


def _component_ring_count(grid: np.ndarray) -> int:
    return len(_ring_infos(grid))


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    family_count = random.randint(1, 3)
    ring_counts = [random.choices([1, 2, 3, 4], weights=[1, 4, 4, 2])[0] for _ in range(family_count)]
    if max(ring_counts) == 1:
        ring_counts[random.randrange(family_count)] = random.choice([2, 3])
    curve_count = random.randint(1, 3)

    patches: list[tuple[int, np.ndarray]] = []
    for ring_count in ring_counts:
        patch = _make_ring_family(ring_count, max_h=min(18, height - 2), max_w=min(18, width - 2))
        if patch is None:
            return None
        patches.append((1, patch))

    for _ in range(curve_count):
        patch = _make_open_curve(max_h=min(12, height - 2), max_w=min(12, width - 2))
        if patch is None:
            return None
        patches.append((0, patch))

    patches.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for _kind, patch in patches:
        if not _place_patch(grid, occupied, patch):
            return None

    if _component_ring_count(grid) != sum(ring_counts):
        return None
    fg_labels, n_fg = label_mask(grid != BLACK, connectivity=4)
    if n_fg != family_count + curve_count + sum(ring_counts) - family_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out == RED) or not np.any(out == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-parity ARC input")
