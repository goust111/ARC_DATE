from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
GRAY = 5
DOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _scan_docks(grid: np.ndarray) -> list[dict]:
    bottom = np.asarray(grid[-1], dtype=int)
    docks: list[dict] = []
    c = 0
    while c < bottom.size:
        if int(bottom[c]) == BG:
            c += 1
            continue
        color = int(bottom[c])
        c0 = c
        while c < bottom.size and int(bottom[c]) == color:
            c += 1
        docks.append({"left": c0, "right": c - 1, "length": c - c0, "color": color})
        if c < bottom.size:
            assert int(bottom[c]) == BG, "docks must be separated by background"
    return docks


def _outside_bg_mask(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    empty = crop == BG
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _enclosed_bg_area(grid: np.ndarray, comp: dict) -> int:
    crop = _crop_bbox(grid, comp["bbox"])
    assert np.all((crop == BG) | (crop == int(comp["color"])))
    outside = _outside_bg_mask(crop)
    return int(np.count_nonzero((crop == BG) & ~outside))


def _hole_component_count(grid: np.ndarray, comp: dict) -> int:
    crop = _crop_bbox(grid, comp["bbox"])
    outside = _outside_bg_mask(crop)
    holes = (crop == BG) & ~outside
    h, w = holes.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not holes[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each gray hollow object above the berth whose length equals its enclosed background area.

    Input structure:
    - The background is black.
    - The bottom row contains several separated horizontal colored dock segments with pairwise distinct lengths and colors.
    - Above them are separated gray 4-connected objects whose bounding boxes may be wider or narrower than their matching dock.
    - Each gray object encloses one or more black holes inside its bounding box.

    Rule:
    - For each gray object, count the total number of black cells inside its bounding box that are not 4-connected to the bounding-box border through black cells.
    - Find the unique dock whose segment length equals that enclosed-hole area.
    - Translate the whole object so its bounding-box left edge aligns with that dock's left end, and place it immediately above the bottom-row dock.
    - Recolor the object's gray pixels to the dock color, keep enclosed holes black, erase the original object, and leave all docks unchanged.

    Key insight:
    - The destination is selected by an invisible key: enclosed background area, not visible width or silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.any(g == BG)

    docks = _scan_docks(g)
    assert len(docks) >= 2
    dock_lengths = [dock["length"] for dock in docks]
    dock_colors = [dock["color"] for dock in docks]
    assert len(set(dock_lengths)) == len(docks)
    assert len(set(dock_colors)) == len(docks)
    assert all(color not in (BG, GRAY) for color in dock_colors)

    objects = _components_of_color(g[:-1], GRAY)
    assert objects

    length_to_dock = {dock["length"]: dock for dock in docks}
    assert len(length_to_dock) == len(docks)

    out = g.copy()
    for comp in objects:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = BG

    for comp in objects:
        area = _enclosed_bg_area(g, comp)
        assert area in length_to_dock
        dock = length_to_dock[area]
        r0, c0, r1, c1 = comp["bbox"]
        patch = _crop_bbox(g, comp["bbox"])
        recolored = np.where(patch == GRAY, int(dock["color"]), BG)
        dest_r = g.shape[0] - 1 - patch.shape[0]
        dest_c = int(dock["left"])
        assert dest_r >= 0
        _paste(out, recolored, dest_r, dest_c, transparent=BG)

    return out


def _partition_total(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _grow_component(
    owners: np.ndarray,
    comp_id: int,
    target_area: int,
) -> bool:
    ih, iw = owners.shape

    def allowed_seed(rr: int, cc: int) -> bool:
        if owners[rr, cc] != -1:
            return False
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < ih and 0 <= nc < iw and owners[nr, nc] not in (-1, comp_id):
                return False
        return True

    seeds = [(r, c) for r in range(ih) for c in range(iw) if allowed_seed(r, c)]
    if not seeds:
        return False
    cell = random.choice(seeds)
    owners[cell] = comp_id
    cells = [cell]

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        frontier_seen: set[tuple[int, int]] = set()
        for rr, cc in cells:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < ih and 0 <= nc < iw):
                    continue
                if owners[nr, nc] != -1 or (nr, nc) in frontier_seen:
                    continue
                ok = True
                for dr2, dc2 in DIRS4:
                    ar, ac = nr + dr2, nc + dc2
                    if 0 <= ar < ih and 0 <= ac < iw and owners[ar, ac] not in (-1, comp_id):
                        ok = False
                        break
                if ok:
                    frontier.append((nr, nc))
                    frontier_seen.add((nr, nc))
        if not frontier:
            return False
        new_cell = random.choice(frontier)
        owners[new_cell] = comp_id
        cells.append(new_cell)
    return True


def _make_hole_layout(interior_h: int, interior_w: int, area: int, hole_components: int) -> np.ndarray | None:
    sizes = _partition_total(area, hole_components)
    if area > hole_components and all(size == 1 for size in sizes):
        return None
    for _ in range(80):
        owners = np.full((interior_h, interior_w), -1, dtype=int)
        order = list(enumerate(sizes))
        random.shuffle(order)
        ok = True
        for comp_id, size in order:
            if not _grow_component(owners, comp_id, size):
                ok = False
                break
        if ok and np.count_nonzero(owners >= 0) == area:
            return owners >= 0
    return None


def _maybe_add_tab(patch: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        w = patch.shape[1]
        tab = np.zeros((1, w), dtype=int)
        c0 = random.randint(0, w - 2)
        c1 = random.randint(c0 + 1, w - 1)
        tab[0, c0 : c1 + 1] = GRAY
        patch = np.vstack([tab, patch])
    if random.random() < 0.5:
        w = patch.shape[1]
        tab = np.zeros((1, w), dtype=int)
        c0 = random.randint(0, w - 2)
        c1 = random.randint(c0 + 1, w - 1)
        tab[0, c0 : c1 + 1] = GRAY
        patch = np.vstack([patch, tab])
    return patch


def _build_object_patch(width: int, area: int, hole_components: int) -> np.ndarray | None:
    assert width >= 3 and area >= 1 and hole_components >= 1
    interior_w = width - 2
    min_h = 1
    if interior_w == 1:
        min_h = area + hole_components - 1
    else:
        min_h = max((area + interior_w - 1) // interior_w, hole_components)

    for interior_h in range(min_h, min_h + 7):
        if interior_h + 2 > 12:
            break
        holes = _make_hole_layout(interior_h, interior_w, area, hole_components)
        if holes is None:
            continue
        patch = np.full((interior_h + 2, width), GRAY, dtype=int)
        patch[1:-1, 1:-1][holes] = BG
        patch = _maybe_add_tab(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch).copy()

        comps = _components_of_color(patch, GRAY)
        if len(comps) != 1:
            continue
        comp = comps[0]
        if _enclosed_bg_area(patch, comp) != area:
            continue
        if _hole_component_count(patch, comp) != hole_components:
            continue
        if patch.shape[1] != width:
            continue
        return patch
    return None


def _place_non_overlapping(
    shape: tuple[int, int],
    patches: list[np.ndarray],
    *,
    max_bottom: int,
    border: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        ph, pw = patches[idx].shape
        candidates = []
        for r0 in range(0, max_bottom - ph + 1):
            for c0 in range(0, w - pw + 1):
                rr0 = max(0, r0 - border)
                cc0 = max(0, c0 - border)
                rr1 = min(h, r0 + ph + border)
                cc1 = min(w, c0 + pw + border)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        r0, c0 = placements[idx]
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True
    return [(int(r), int(c)) for r, c in placements]


def _dock_layout(widths: list[int], lengths: list[int]) -> tuple[int, list[int], list[int]] | None:
    order = list(range(len(widths)))
    random.shuffle(order)
    footprint = [max(widths[i], lengths[i]) for i in order]
    gaps = [random.randint(2, 4) for _ in range(max(0, len(order) - 1))]
    union = sum(footprint) + sum(gaps)
    min_total = union + 2
    if min_total > 30:
        return None
    total_w = random.randint(min_total, 30)
    slack = total_w - union
    left_margin = random.randint(1, max(1, slack - 1))
    lefts: list[int] = []
    x = left_margin
    for pos, idx in enumerate(order):
        lefts.append(x)
        x += footprint[pos]
        if pos < len(order) - 1:
            x += gaps[pos]
    dock_lefts = [0] * len(widths)
    for idx, left in zip(order, lefts):
        dock_lefts[idx] = left
    return total_w, dock_lefts, order


def _expected_output(
    inp: np.ndarray,
    patches: list[np.ndarray],
    source_pos: list[tuple[int, int]],
    dock_lefts: list[int],
    dock_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for patch, (r0, c0) in zip(patches, source_pos):
        ph, pw = patch.shape
        out[r0 : r0 + ph, c0 : c0 + pw][patch == GRAY] = BG
    for patch, left, color in zip(patches, dock_lefts, dock_colors):
        recolored = np.where(patch == GRAY, color, BG)
        _paste(out, recolored, inp.shape[0] - 1 - patch.shape[0], left, transparent=BG)
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    lengths: list[int],
    widths: list[int],
    hole_component_counts: list[int],
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    docks = _scan_docks(grid)
    if sorted(dock["length"] for dock in docks) != sorted(lengths):
        return False
    objects = _components_of_color(grid[:-1], GRAY)
    if len(objects) != len(lengths):
        return False
    areas = sorted(_enclosed_bg_area(grid, comp) for comp in objects)
    if areas != sorted(lengths):
        return False
    comps = sorted(_hole_component_count(grid, comp) for comp in objects)
    if comps != sorted(hole_component_counts):
        return False
    if not any(area > holes for area, holes in zip(sorted(lengths), sorted(hole_component_counts))):
        return False
    wrong_width_match = 0
    for width, length in zip(widths, lengths):
        if width != length and width in lengths:
            wrong_width_match += 1
    if wrong_width_match == 0:
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_obj = random.randint(2, 3)
    lengths = sorted(random.sample(list(range(3, 8)), n_obj))
    shift = 1 if n_obj == 2 else random.choice([1, 2])
    widths = [lengths[(i + shift) % n_obj] for i in range(n_obj)]

    hole_component_counts = list(range(1, n_obj + 1))
    random.shuffle(hole_component_counts)

    dock_colors = random.sample(DOCK_COLORS, n_obj)
    patches: list[np.ndarray] = []
    for width, area, hole_components in zip(widths, lengths, hole_component_counts):
        patch = _build_object_patch(width, area, hole_components)
        if patch is None:
            return None
        patches.append(patch)

    layout = _dock_layout([patch.shape[1] for patch in patches], lengths)
    if layout is None:
        return None
    total_w, dock_lefts, _order = layout

    max_patch_h = max(patch.shape[0] for patch in patches)
    min_h = max(12, max_patch_h + 7)
    max_h = min(24, max_patch_h + 12)
    if min_h > max_h:
        return None
    total_h = random.randint(min_h, max_h)

    max_source_bottom = total_h - max_patch_h - 4
    if max_source_bottom < 0:
        return None
    source_pos = _place_non_overlapping((total_h - 1, total_w), patches, max_bottom=max_source_bottom, border=1)
    if source_pos is None:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    for left, length, color in zip(dock_lefts, lengths, dock_colors):
        grid[-1, left : left + length] = color
    for patch, (r0, c0) in zip(patches, source_pos):
        _paste(grid, patch, r0, c0, transparent=BG)

    expected = _expected_output(grid, patches, source_pos, dock_lefts, dock_colors)
    if not _validate_case(grid, expected, lengths, [patch.shape[1] for patch in patches], hole_component_counts):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid berth-by-hole-area ARC input")
