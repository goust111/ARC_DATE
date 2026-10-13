from __future__ import annotations

from collections import deque
import math

import numpy as np


BLACK = 0
GREY = 5
LIQUID_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_for_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for coords in _components_for_mask(grid == GREY):
        bbox = _bbox_from_coords(coords)
        patch = _crop(grid, bbox)
        if np.any((patch != BLACK) & (patch != GREY)):
            continue
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        if not (np.all(patch[:, 0] == GREY) and np.all(patch[:, -1] == GREY) and np.all(patch[-1, :] == GREY)):
            continue
        if not (np.any(patch[0, :] == BLACK) and np.any(patch[0, :] == GREY)):
            continue
        cavity_mask = patch == BLACK
        if not cavity_mask.any():
            continue
        cups.append(
            {
                "bbox": bbox,
                "coords": coords,
                "patch": patch,
                "cavity_mask": cavity_mask,
                "capacity": int(cavity_mask.sum()),
            }
        )
    return cups


def _parse_blobs(grid: np.ndarray) -> list[dict]:
    blobs: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BLACK, GREY)):
        comps = _components_for_mask(grid == color)
        for coords in comps:
            bbox = _bbox_from_coords(coords)
            patch = _crop(grid, bbox)
            mask = patch == color
            blobs.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "coords": coords,
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return blobs


def _match_by_area(cups: list[dict], blobs: list[dict]) -> dict[int, dict]:
    areas = [cup["capacity"] for cup in cups]
    blob_areas = [blob["area"] for blob in blobs]
    if len(areas) != len(set(areas)) or len(blob_areas) != len(set(blob_areas)):
        raise ValueError("capacities and blob areas must be unique")
    by_area = {blob["area"]: blob for blob in blobs}
    if set(by_area) != set(areas):
        raise ValueError("cup capacities and blob areas must match exactly")
    return by_area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each colored liquid blob into the gray cup whose black cavity has exactly the same area.

    Input structure:
    - Black background with several gray cup outlines: nearly closed hollow rectangles whose top edge has a black opening.
    - Separate single-color connected liquid blobs in distinct colors, with no blob touching any cup.
    - Each cup bbox contains only gray wall pixels and black cavity pixels.

    Rule:
    - Find every gray cup component and count the black cavity pixels inside its bbox.
    - Find every colored liquid blob and count its pixels.
    - Pair blobs to cups by exact area equality, erase the original blob locations to black, and fill the entire matched cup cavity with that blob's color while leaving gray walls unchanged.

    Key insight:
    - The blob outline is a decoy: pouring destroys the original shape, so the only invariant that matters is pixel count matching cup capacity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cups = _parse_cups(g)
    blobs = _parse_blobs(g)
    by_area = _match_by_area(cups, blobs)

    for blob in blobs:
        rows = blob["coords"][:, 0]
        cols = blob["coords"][:, 1]
        out[rows, cols] = BLACK

    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        color = by_area[cup["capacity"]]["color"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cup["cavity_mask"]] = color

    return out


def _make_cup_patch(h: int, w: int, gap_start: int, gap_w: int) -> np.ndarray:
    patch = np.full((h, w), GREY, dtype=int)
    patch[1:-1, 1:-1] = BLACK
    patch[0, gap_start : gap_start + gap_w] = BLACK
    return patch


def _random_cup_spec() -> dict:
    h = int(np.random.randint(5, 8))
    w = int(np.random.randint(5, 9))
    gap_w = int(np.random.randint(1, w - 1))
    gap_start = int(np.random.randint(1, w - gap_w))
    patch = _make_cup_patch(h, w, gap_start, gap_w)
    cavity_mask = patch == BLACK
    return {
        "h": h,
        "w": w,
        "gap_start": gap_start,
        "gap_w": gap_w,
        "patch": patch,
        "cavity_mask": cavity_mask,
        "capacity": int(cavity_mask.sum()),
    }


def _sample_cups(n_cups: int) -> list[dict] | None:
    cups: list[dict] = []
    used_caps: set[int] = set()

    if n_cups >= 2 and np.random.rand() < 0.65:
        for _ in range(100):
            h = int(np.random.randint(5, 8))
            w = int(np.random.randint(5, 9))
            if w < 5:
                continue
            widths = list(range(1, w - 1))
            if len(widths) < 2:
                continue
            gap_w1, gap_w2 = map(int, np.random.choice(widths, size=2, replace=False))
            gap_specs = []
            for gap_w in (gap_w1, gap_w2):
                gap_start = int(np.random.randint(1, w - gap_w))
                patch = _make_cup_patch(h, w, gap_start, gap_w)
                gap_specs.append(
                    {
                        "h": h,
                        "w": w,
                        "gap_start": gap_start,
                        "gap_w": gap_w,
                        "patch": patch,
                        "cavity_mask": patch == BLACK,
                        "capacity": int((patch == BLACK).sum()),
                    }
                )
            if gap_specs[0]["capacity"] != gap_specs[1]["capacity"]:
                cups.extend(gap_specs)
                used_caps.update(spec["capacity"] for spec in gap_specs)
                break

    while len(cups) < n_cups:
        spec = _random_cup_spec()
        if spec["capacity"] in used_caps:
            continue
        cups.append(spec)
        used_caps.add(spec["capacity"])

    if len(cups) != n_cups:
        return None
    return cups


def _mask_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask.copy()
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grow_connected_subset(allowed: np.ndarray, area: int) -> np.ndarray | None:
    allowed_coords = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(allowed_coords) < area:
        return None
    for _ in range(200):
        start = allowed_coords[int(np.random.randint(len(allowed_coords)))]
        chosen = {start}
        frontier = {nbr for nbr in _neighbors4(*start) if 0 <= nbr[0] < allowed.shape[0] and 0 <= nbr[1] < allowed.shape[1] and allowed[nbr]}
        while len(chosen) < area:
            if not frontier:
                break
            pick = tuple(frontier.pop()) if len(frontier) == 1 else list(frontier)[int(np.random.randint(len(frontier)))]
            frontier.discard(pick)
            if pick in chosen:
                continue
            chosen.add(pick)
            for nbr in _neighbors4(*pick):
                if 0 <= nbr[0] < allowed.shape[0] and 0 <= nbr[1] < allowed.shape[1] and allowed[nbr] and nbr not in chosen:
                    frontier.add(nbr)
        if len(chosen) != area:
            continue
        mask = _crop_mask(_mask_from_cells(chosen, allowed.shape))
        h, w = mask.shape
        if h >= 2 and w >= 2:
            return mask
    return None


def _choose_blob_box(area: int) -> tuple[int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    max_side = min(10, area)
    for h in range(2, max_side + 1):
        w = int(math.ceil(area / h))
        if w > 10:
            continue
        slack = h * w - area
        ratio_gap = abs(h - w)
        candidates.append((slack, ratio_gap, h, w))
    if not candidates:
        return 1, area
    candidates.sort()
    slack, ratio_gap, h, w = candidates[int(np.random.randint(min(4, len(candidates))))]
    return int(h), int(w)


def _make_blob_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _rectangle_mask(area: int) -> np.ndarray | None:
    factors: list[tuple[int, int]] = []
    for h in range(2, min(10, area) + 1):
        if area % h != 0:
            continue
        w = area // h
        if 2 <= w <= 10:
            factors.append((h, w))
    if not factors:
        return None
    h, w = factors[int(np.random.randint(len(factors)))]
    return np.ones((h, w), dtype=bool)


def _build_blobs(cups: list[dict]) -> list[dict] | None:
    n = len(cups)
    colors = [int(c) for c in np.random.choice(LIQUID_COLORS, size=n, replace=False)]
    caps = [cup["capacity"] for cup in cups]
    order = sorted(range(n), key=lambda i: caps[i])
    decoy_targets = order[:-1]
    np.random.shuffle(decoy_targets)
    n_decoys = int(np.random.randint(1, len(decoy_targets) + 1)) if decoy_targets else 0
    decoy_targets = set(decoy_targets[:n_decoys])

    blobs: list[dict] = []
    for i, cup in enumerate(cups):
        area = cup["capacity"]
        mask = None
        if i in decoy_targets:
            hosts = [j for j in order if cups[j]["capacity"] > area]
            np.random.shuffle(hosts)
            for host in hosts:
                mask = _grow_connected_subset(cups[host]["cavity_mask"], area)
                if mask is not None:
                    break
        if mask is None:
            if np.random.rand() < 0.25:
                mask = _rectangle_mask(area)
            if mask is None:
                h, w = _choose_blob_box(area)
                full = np.ones((h, w), dtype=bool)
                mask = _grow_connected_subset(full, area)
        if mask is None or int(mask.sum()) != area:
            return None
        blobs.append(
            {
                "color": colors[i],
                "area": area,
                "mask": mask,
                "patch": _make_blob_patch(mask, colors[i]),
            }
        )
    return blobs


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _find_free_location(occupied: np.ndarray, patch: np.ndarray, pad: int = 1, border: int = 1) -> tuple[int, int] | None:
    h_grid, w_grid = occupied.shape
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h_grid - border - h + 1):
        for c0 in range(border, w_grid - border - w + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h_grid, r0 + h + pad)
            cc1 = min(w_grid, c0 + w + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _mark_occupied(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int = 1) -> None:
    h, w = shape
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(occupied.shape[0], top + h + pad)
    cc1 = min(occupied.shape[1], left + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _fit_mask_in_cavity(blob_mask: np.ndarray, cavity_mask: np.ndarray) -> bool:
    h, w = blob_mask.shape
    H, W = cavity_mask.shape
    if h > H or w > W:
        return False
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if np.all(cavity_mask[r0 : r0 + h, c0 : c0 + w][blob_mask]):
                return True
    return False


def _expected_output(grid: np.ndarray, cups: list[dict], blobs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    by_area = _match_by_area(cups, blobs)
    for blob in blobs:
        rows = blob["coords"][:, 0]
        cols = blob["coords"][:, 1]
        out[rows, cols] = BLACK
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[cup["cavity_mask"]] = by_area[cup["capacity"]]["color"]
    return out


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        cups = _parse_cups(grid)
        blobs = _parse_blobs(grid)
    except Exception:
        return False
    if len(cups) < 2 or len(cups) != len(blobs):
        return False
    capacities = [cup["capacity"] for cup in cups]
    colors = [blob["color"] for blob in blobs]
    areas = [blob["area"] for blob in blobs]
    if len(capacities) != len(set(capacities)):
        return False
    if len(colors) != len(set(colors)):
        return False
    if sorted(capacities) != sorted(areas):
        return False

    decoy_found = False
    for blob in blobs:
        for cup in cups:
            if blob["area"] == cup["capacity"]:
                continue
            if _fit_mask_in_cavity(blob["mask"], cup["cavity_mask"]):
                decoy_found = True
                break
        if decoy_found:
            break
    if not decoy_found:
        return False

    try:
        expected = _expected_output(grid, cups, blobs)
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n_cups = int(np.random.randint(2, 5))
    cups = _sample_cups(n_cups)
    if cups is None:
        return None
    blobs = _build_blobs(cups)
    if blobs is None:
        return None

    footprint = sum(cup["patch"].shape[0] * cup["patch"].shape[1] for cup in cups)
    footprint += sum(blob["patch"].shape[0] * blob["patch"].shape[1] for blob in blobs)
    side_min = max(18, int(math.ceil(math.sqrt(footprint * 2.8))))
    if side_min > 30:
        return None
    h = int(np.random.randint(side_min, min(30, side_min + 5) + 1))
    w = int(np.random.randint(side_min, min(30, side_min + 5) + 1))
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    items = [{"kind": "cup", "patch": cup["patch"]} for cup in cups] + [{"kind": "blob", "patch": blob["patch"]} for blob in blobs]
    np.random.shuffle(items)
    items.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    for item in items:
        loc = _find_free_location(occupied, item["patch"], pad=2 if item["kind"] == "cup" else 1, border=1)
        if loc is None:
            return None
        r0, c0 = loc
        _place_patch(grid, item["patch"], r0, c0)
        _mark_occupied(occupied, r0, c0, item["patch"].shape, pad=2 if item["kind"] == "cup" else 1)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-filling input")
