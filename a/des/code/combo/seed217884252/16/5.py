from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            comp.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        out.append(comp)

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _component_mask(comp: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_mask_key(np.rot90(mask, k)) for k in range(4)}


def _classify_mask(mask: np.ndarray, ref_mask: np.ndarray) -> int:
    if mask.shape == ref_mask.shape and np.array_equal(mask, ref_mask):
        return GREEN
    for k in (1, 2, 3):
        rotated = np.rot90(ref_mask, k)
        if mask.shape == rotated.shape and np.array_equal(mask, rotated):
            return YELLOW
    return RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each grey object by its shape similarity to the unique azure reference object.

    Rule:
    - Exact same normalized mask -> green.
    - Matches after a 90/180/270 degree rotation -> yellow.
    - No rotational match -> red.
    - The reference object remains azure.
    """
    g = np.asarray(grid, dtype=int)

    ref_comps = _components(g == AZURE)
    grey_comps = _components(g == GREY)

    assert len(ref_comps) == 1, "expected exactly one azure reference object"
    assert grey_comps, "expected at least one grey object"

    ref_mask = _component_mask(ref_comps[0])
    out = g.copy()

    for comp in grey_comps:
        color = _classify_mask(_component_mask(comp), ref_mask)
        for r, c in comp:
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

REF_AREA_RANGE = (5, 8)
REF_MAX_DIM = 5
MISMATCH_MAX_DIM = 6
MIN_EXTRA_OBJECTS = 0
MAX_EXTRA_OBJECTS = 4
GRID_MAX_SIDE = 30
GENERATION_RETRIES = 400
SHAPE_RETRIES = 200


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            total += 1
    return total


def _random_orient(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool).copy()


def _random_connected_mask(
    area_lo: int,
    area_hi: int,
    *,
    min_height: int,
    min_width: int,
    max_dim: int,
) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        area = random.randint(int(area_lo), int(area_hi))
        h = random.randint(int(min_height), int(max_dim))
        w = random.randint(int(min_width), int(max_dim))
        if area > h * w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            dense = [cell for cell in frontier if _neighbor_count(mask, *cell) >= 2]
            pool = dense if dense and random.random() < 0.6 else list(frontier)
            nr, nc = random.choice(pool)
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue

        mask = _trim_mask(mask)
        if mask.shape[0] < min_height or mask.shape[1] < min_width:
            continue

        mask = _random_orient(mask)
        if mask.shape[0] > max_dim or mask.shape[1] > max_dim:
            continue
        return mask

    return None


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _make_reference_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        mask = _random_connected_mask(
            REF_AREA_RANGE[0],
            REF_AREA_RANGE[1],
            min_height=2,
            min_width=2,
            max_dim=REF_MAX_DIM,
        )
        if mask is None:
            continue
        if _is_rotation_asymmetric(mask):
            return mask
    raise RuntimeError("failed to generate an asymmetric reference shape")


def _make_mismatch_mask(ref_mask: np.ndarray) -> np.ndarray:
    ref_keys = _rotation_keys(ref_mask)

    reflection_candidates: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in (np.fliplr(ref_mask), np.flipud(ref_mask)):
        for k in range(4):
            variant = np.asarray(np.rot90(base, k), dtype=bool)
            key = _mask_key(variant)
            if key in ref_keys or key in seen:
                continue
            seen.add(key)
            reflection_candidates.append(variant.copy())

    if reflection_candidates and random.random() < 0.35:
        return random.choice(reflection_candidates).copy()

    ref_area = int(ref_mask.sum())
    for _ in range(SHAPE_RETRIES):
        if random.random() < 0.6:
            area = ref_area
        else:
            area = max(4, ref_area + random.choice([-2, -1, 1, 2]))
        mask = _random_connected_mask(
            area,
            area,
            min_height=1,
            min_width=1,
            max_dim=MISMATCH_MAX_DIM,
        )
        if mask is None:
            continue
        if _mask_key(mask) not in ref_keys:
            return mask

    raise RuntimeError("failed to generate a mismatch shape")


def _sample_labels() -> list[str]:
    labels = ["exact", "rotated", "mismatch"]
    for _ in range(random.randint(MIN_EXTRA_OBJECTS, MAX_EXTRA_OBJECTS)):
        labels.append(random.choice(labels[:3]))
    random.shuffle(labels)
    return labels


def _corner_position(mask: np.ndarray, rows: int, cols: int, corner: str) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, cols - w
    if corner == "bl":
        return rows - h, 0
    if corner == "br":
        return rows - h, cols - w
    raise ValueError(f"unknown corner {corner}")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = map(int, mask.shape)
    grid[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, occupied.shape)
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(h, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(w, cc + 2)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _place_object(occupied: np.ndarray, grid: np.ndarray, mask: np.ndarray, color: int) -> tuple[int, int] | None:
    h, w = map(int, mask.shape)
    rows, cols = map(int, grid.shape)
    candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place(occupied, mask, int(r0), int(c0)):
            continue
        _paint_mask(grid, mask, int(r0), int(c0), int(color))
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
        return int(r0), int(c0)
    return None


def _build_expected_output(
    inp: np.ndarray,
    placements: list[tuple[np.ndarray, int, int, str]],
) -> np.ndarray:
    out = inp.copy()
    color_map = {"exact": GREEN, "rotated": YELLOW, "mismatch": RED}
    for mask, r0, c0, label in placements:
        _paint_mask(out, mask, int(r0), int(c0), int(color_map[label]))
    return out


def _grey_mask_for_label(ref_mask: np.ndarray, label: str) -> np.ndarray:
    if label == "exact":
        return ref_mask.copy()
    if label == "rotated":
        return np.asarray(np.rot90(ref_mask, random.choice((1, 2, 3))), dtype=bool).copy()
    if label == "mismatch":
        return _make_mismatch_mask(ref_mask)
    raise ValueError(f"unknown label {label}")


def _gen_attempt() -> np.ndarray | None:
    ref_mask = _make_reference_mask()
    labels = _sample_labels()
    grey_specs = [{"label": label, "mask": _grey_mask_for_label(ref_mask, label)} for label in labels]
    random.shuffle(grey_specs)
    grey_specs.sort(
        key=lambda item: (int(item["mask"].shape[0]) * int(item["mask"].shape[1]), int(np.sum(item["mask"]))),
        reverse=True,
    )

    all_masks = [ref_mask] + [spec["mask"] for spec in grey_specs]
    max_h = max(int(mask.shape[0]) for mask in all_masks)
    max_w = max(int(mask.shape[1]) for mask in all_masks)
    min_side = max(14, max_h + 4, max_w + 4, 8 + 2 * len(all_masks))
    if min_side > GRID_MAX_SIDE:
        return None
    rows = random.randint(min_side, GRID_MAX_SIDE)
    cols = random.randint(min_side, GRID_MAX_SIDE)

    inp = np.full((rows, cols), BLACK, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    ref_r0, ref_c0 = _corner_position(ref_mask, rows, cols, corner)
    _paint_mask(inp, ref_mask, ref_r0, ref_c0, AZURE)
    occupied[ref_r0 : ref_r0 + ref_mask.shape[0], ref_c0 : ref_c0 + ref_mask.shape[1]] |= ref_mask

    placements: list[tuple[np.ndarray, int, int, str]] = []
    for spec in grey_specs:
        placed = _place_object(occupied, inp, spec["mask"], GREY)
        if placed is None:
            return None
        r0, c0 = placed
        placements.append((spec["mask"], int(r0), int(c0), str(spec["label"])))

    if len(_components(inp == GREY)) != len(grey_specs):
        return None

    expected = _build_expected_output(inp, placements)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    colors = set(map(int, np.unique(actual)))
    if not {AZURE, GREEN, YELLOW, RED}.issubset(colors):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
