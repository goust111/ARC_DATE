from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("tl", "tr", "bl", "br")
BACKGROUND = 0


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _hole_info(mask: np.ndarray) -> tuple[str, int, int]:
    h, w = mask.shape
    hole = ~mask
    coords = np.argwhere(hole)
    assert coords.size > 0, "fragment must have a missing corner"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert bool(np.all(hole[r0 : r1 + 1, c0 : c1 + 1])), "hole must be rectangular"
    hole_h = r1 - r0 + 1
    hole_w = c1 - c0 + 1
    assert 0 < hole_h < h and 0 < hole_w < w, "hole must remove a proper corner block"

    if r0 == 0 and c0 == 0:
        return "tl", hole_h, hole_w
    if r0 == 0 and c1 == w - 1:
        return "tr", hole_h, hole_w
    if r1 == h - 1 and c0 == 0:
        return "bl", hole_h, hole_w
    if r1 == h - 1 and c1 == w - 1:
        return "br", hole_h, hole_w
    raise AssertionError("hole must touch a bbox corner")


def _corner_distance(
    base_shape: tuple[int, int],
    move_shape: tuple[int, int],
    corner: str,
    top: int,
    left: int,
) -> int:
    base_h, base_w = base_shape
    move_h, move_w = move_shape
    if corner == "tl":
        return top + left
    if corner == "tr":
        return top + (base_w - (left + move_w))
    if corner == "bl":
        return (base_h - (top + move_h)) + left
    return (base_h - (top + move_h)) + (base_w - (left + move_w))


def _find_embed_position(base_mask: np.ndarray, move_mask: np.ndarray, base_corner: str) -> tuple[int, int]:
    base_h, base_w = base_mask.shape
    move_h, move_w = move_mask.shape
    assert move_h <= base_h and move_w <= base_w

    hole = ~base_mask
    best: tuple[int, int, int, int] | None = None
    for top in range(base_h - move_h + 1):
        for left in range(base_w - move_w + 1):
            placed = np.zeros_like(base_mask)
            placed[top : top + move_h, left : left + move_w] = move_mask
            if not bool(np.all(base_mask | placed)):
                continue
            if not bool(np.any(placed & hole)):
                continue
            dist = _corner_distance(base_mask.shape, move_mask.shape, base_corner, top, left)
            overlap = int(np.count_nonzero(placed & base_mask))
            key = (dist, -overlap, top, left)
            if best is None or key < best:
                best = key
    assert best is not None, "pair must admit a unique embedding into the base bbox"
    return int(best[2]), int(best[3])


def _component_record(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict:
    bbox = _bbox_from_coords(coords)
    mask = _mask_from_coords(coords, bbox)
    hole_corner, hole_h, hole_w = _hole_info(mask)
    r0, c0, r1, c1 = bbox
    return {
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "mask": mask,
        "bbox_area": int(mask.shape[0] * mask.shape[1]),
        "top": int(r0),
        "left": int(c0),
        "hole_corner": hole_corner,
        "hole_h": int(hole_h),
        "hole_w": int(hole_w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored L fragments, keep the larger bbox fixed, and translate the smaller
    fragment so the two combine into a solid rectangle inside the fixed fragment's bbox.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        comps = _component_coords(g == color)
        assert len(comps) == 2, "each color must appear in exactly two disconnected fragments"
        records = [_component_record(g, color, comp) for comp in comps]

        base = max(records, key=lambda rec: (rec["bbox_area"], -rec["top"], -rec["left"]))
        move = records[0] if records[1] is base else records[1]

        top, left = _find_embed_position(base["mask"], move["mask"], base["hole_corner"])
        for r, c in move["coords"]:
            out[r, c] = BACKGROUND

        base_r0, base_c0, _base_r1, _base_c1 = base["bbox"]
        move_h, move_w = move["mask"].shape
        target = out[base_r0 + top : base_r0 + top + move_h, base_c0 + left : base_c0 + left + move_w]
        target[move["mask"]] = color
        out[base_r0 + top : base_r0 + top + move_h, base_c0 + left : base_c0 + left + move_w] = target

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
PAIR_COUNT_OPTIONS = (2, 3, 4)
PLACEMENT_PADDING = 3
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 300
PLACE_ATTEMPTS = 300


def _make_l_mask(height: int, width: int, corner: str, hole_h: int, hole_w: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if corner == "tl":
        mask[:hole_h, :hole_w] = False
    elif corner == "tr":
        mask[:hole_h, width - hole_w :] = False
    elif corner == "bl":
        mask[height - hole_h :, :hole_w] = False
    else:
        mask[height - hole_h :, width - hole_w :] = False
    return mask


def _anchor_position(base_h: int, base_w: int, move_h: int, move_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, base_w - move_w
    if corner == "bl":
        return base_h - move_h, 0
    return base_h - move_h, base_w - move_w


def _build_pair(equal_area: bool, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_ATTEMPTS):
        base_h = random.randint(4, max_side)
        base_w = random.randint(4, max_side)
        base_corner = random.choice(CORNERS)
        base_hole_h = random.randint(1, base_h - 1)
        base_hole_w = random.randint(1, base_w - 1)
        base_mask = _make_l_mask(base_h, base_w, base_corner, base_hole_h, base_hole_w)

        dims: list[tuple[int, int]] = []
        for move_h in range(2, base_h + 1):
            for move_w in range(2, base_w + 1):
                area = move_h * move_w
                if equal_area and area != base_h * base_w:
                    continue
                if not equal_area and area >= base_h * base_w:
                    continue
                dims.append((move_h, move_w))
        random.shuffle(dims)

        for move_h, move_w in dims:
            move_corners = list(CORNERS)
            random.shuffle(move_corners)
            for move_corner in move_corners:
                for move_hole_h in random.sample(list(range(1, move_h)), k=move_h - 1):
                    for move_hole_w in random.sample(list(range(1, move_w)), k=move_w - 1):
                        move_mask = _make_l_mask(move_h, move_w, move_corner, move_hole_h, move_hole_w)
                        top, left = _anchor_position(base_h, base_w, move_h, move_w, base_corner)
                        placed = np.zeros_like(base_mask)
                        placed[top : top + move_h, left : left + move_w] = move_mask
                        if not bool(np.all(base_mask | placed)):
                            continue
                        if not bool(np.any(placed & ~base_mask)):
                            continue
                        if _find_embed_position(base_mask, move_mask, base_corner) != (top, left):
                            continue
                        return base_mask, move_mask
    raise RuntimeError("failed to build a valid complementary pair")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _grid_candidates(patches: list[np.ndarray]) -> list[tuple[int, int]]:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    padded_area = sum(
        int((p.shape[0] + 2 * PLACEMENT_PADDING) * (p.shape[1] + 2 * PLACEMENT_PADDING))
        for p in patches
    )
    candidates: list[tuple[int, int]] = []
    for h in range(max(GRID_H_RANGE[0], max_h + 2), GRID_H_RANGE[1] + 1):
        for w in range(max(GRID_W_RANGE[0], max_w + 2), GRID_W_RANGE[1] + 1):
            if h * w >= padded_area:
                candidates.append((h, w))
    return candidates


def _place_patches(patches: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            r0 = max(0, top - PLACEMENT_PADDING)
            c0 = max(0, left - PLACEMENT_PADDING)
            r1 = min(grid_h, top + h + PLACEMENT_PADDING)
            c1 = min(grid_w, left + w + PLACEMENT_PADDING)
            if bool(reserved[r0:r1, c0:c1].any()):
                continue
            window = grid[top : top + h, left : left + w]
            assert not bool((window != BACKGROUND).any())
            window[patch != BACKGROUND] = patch[patch != BACKGROUND]
            grid[top : top + h, left : left + w] = window
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    if len(comps) != 1:
        return False
    bbox = _bbox_from_coords(comps[0])
    patch = _mask_from_coords(comps[0], bbox)
    return bool(np.all(patch))


def _validate_input(grid: np.ndarray) -> None:
    colors = [int(v) for v in np.unique(grid) if int(v) != BACKGROUND]
    assert 2 <= len(colors) <= 4
    for color in colors:
        comps = _component_coords(grid == color)
        assert len(comps) == 2
        for comp in comps:
            _hole_info(_mask_from_coords(comp, _bbox_from_coords(comp)))


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    assert not np.array_equal(inp, out)
    colors = [int(v) for v in np.unique(inp) if int(v) != BACKGROUND]
    for color in colors:
        mask = out == color
        assert _is_solid_rectangle(mask), "each color should end as one filled rectangle"


def _branch_flags(pair_count: int) -> list[bool]:
    equal_count = pair_count // 2
    if pair_count % 2 == 1 and random.random() < 0.5:
        equal_count += 1
    flags = [True] * equal_count + [False] * (pair_count - equal_count)
    random.shuffle(flags)
    return flags


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair_count = random.choice(PAIR_COUNT_OPTIONS)
        colors = random.sample(list(range(1, 10)), pair_count)
        max_side = {2: 8, 3: 7, 4: 6}[pair_count]

        patches: list[np.ndarray] = []
        for color, equal_area in zip(colors, _branch_flags(pair_count), strict=True):
            base_mask, move_mask = _build_pair(equal_area=equal_area, max_side=max_side)
            patches.append(_mask_to_patch(base_mask, color))
            patches.append(_mask_to_patch(move_mask, color))

        grid_sizes = _grid_candidates(patches)
        if not grid_sizes:
            continue
        grid_h, grid_w = random.choice(grid_sizes)
        grid = _place_patches(patches, grid_h, grid_w)
        if grid is None:
            continue

        try:
            _validate_input(grid)
            out = main(grid)
            _validate_output(grid, out)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
