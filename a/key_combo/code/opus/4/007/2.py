from __future__ import annotations

import random

import numpy as np


BG = 0
PATH = 2
SPRITE = 5
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BBox = tuple[int, int, int, int]


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return crop_bbox(mask.astype(int), bbox_coords(coords)).astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _is_axis_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask))


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        arr = np.array(coords, dtype=int)
        box = bbox_coords(arr)
        mask = crop_bbox((grid == color).astype(int), box).astype(bool)
        comps.append(
            {
                "coords": arr,
                "bbox": box,
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    return comps


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(candidate)
            key = (norm.shape, norm.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(norm)
    return variants


SYMMETRIC_LIBRARY = [
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
]

ASYMMETRIC_LIBRARY = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
]


def _sample_library_mask(want_symmetric: bool) -> np.ndarray:
    library = SYMMETRIC_LIBRARY if want_symmetric else ASYMMETRIC_LIBRARY
    for _ in range(100):
        base = _mask_from_coords(random.choice(library))
        candidates = [m for m in _mask_variants(base) if _is_axis_symmetric(m) == want_symmetric]
        if not candidates:
            continue
        mask = random.choice(candidates).copy()
        if 3 <= int(mask.sum()) <= 7 and _is_connected(mask):
            return mask
    raise RuntimeError("failed to sample a valid sprite mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _extract_gate_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    marker_rows = np.where(g[:, 0] == PATH)[0]
    if len(marker_rows) != 1 or int(np.count_nonzero(g == PATH)) != 1:
        raise AssertionError("expected exactly one red marker on the left border")
    path_row = int(marker_rows[0])

    sprites = _components_of_color(g, SPRITE)
    upper = [comp for comp in sprites if comp["bbox"][2] < path_row]
    lower = [comp for comp in sprites if comp["bbox"][0] > path_row]
    if len(upper) != len(lower):
        raise AssertionError("expected the same number of upper and lower gate sprites")

    upper.sort(key=lambda comp: (comp["bbox"][1] + comp["bbox"][3], comp["bbox"][1], comp["bbox"][0]))
    lower.sort(key=lambda comp: (comp["bbox"][1] + comp["bbox"][3], comp["bbox"][1], comp["bbox"][0]))

    gates: list[dict] = []
    for up, down in zip(upper, lower, strict=True):
        up_sym = _is_axis_symmetric(up["mask"])
        down_sym = _is_axis_symmetric(down["mask"])
        if up_sym == down_sym:
            raise AssertionError("each gate must contain exactly one axis-symmetric sprite")
        left = min(up["bbox"][1], down["bbox"][1])
        right = max(up["bbox"][3], down["bbox"][3])
        gates.append(
            {
                "left": int(left),
                "right": int(right),
                "side": -1 if up_sym else 1,
                "upper_area": up["area"],
                "lower_area": down["area"],
            }
        )

    gates.sort(key=lambda gate: gate["left"])
    return path_row, gates


def _paint_if_open(src: np.ndarray, out: np.ndarray, r: int, c: int) -> None:
    if 0 <= r < src.shape[0] and 0 <= c < src.shape[1] and int(src[r, c]) in (BG, PATH):
        out[r, c] = PATH


def _paint_horizontal(src: np.ndarray, out: np.ndarray, row: int, c0: int, c1: int) -> None:
    if c1 < c0:
        return
    for c in range(c0, c1 + 1):
        _paint_if_open(src, out, row, c)


def _expected_path_coords(path_row: int, gates: list[dict], width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = [(path_row, 0)]
    current = 0
    for gate in gates:
        entry = gate["left"] - 1
        for c in range(current + 1, entry + 1):
            coords.append((path_row, c))
        detour_row = path_row - 1 if gate["side"] == -1 else path_row + 1
        coords.append((detour_row, entry))
        for c in range(gate["left"], gate["right"] + 1):
            coords.append((detour_row, c))
        coords.append((path_row, gate["right"]))
        current = gate["right"]
    for c in range(current + 1, width):
        coords.append((path_row, c))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the red border path through each gate on the side whose sprite is axis-symmetric.

    Input structure:
    - A black background contains one red marker on the left border and 2 to 4 grey gate pairs inside the grid.
    - Each gate pair has one monochrome 4-connected sprite above the travel row and one below it, with the red path's channel row running horizontally between gates.
    - In every gate exactly one of the two sprites is symmetric under a vertical or horizontal axis.

    Rule:
    - Start at the red left-border marker and draw a red path straight to the right on background cells.
    - For each gate, detect whether the upper or lower sprite is axis-symmetric after cropping it to its bbox.
    - If the upper sprite is symmetric, move the path up by one row just before the gate, cross that gate above the channel, then return to the channel row at the gate's far side; if the lower sprite is symmetric, do the analogous one-row detour downward.
    - After the last gate, continue straight to the right border while preserving all grey sprites.

    Key insight:
    - Symmetry is not the final output filter; it is the per-gate routing signal that must be recomputed at each decision point to choose the detour side.
    """
    g = np.asarray(grid, dtype=int)
    path_row, gates = _extract_gate_records(g)
    out = g.copy()
    _paint_if_open(g, out, path_row, 0)

    current = 0
    for gate in gates:
        entry = gate["left"] - 1
        _paint_horizontal(g, out, path_row, current + 1, entry)
        detour_row = path_row - 1 if gate["side"] == -1 else path_row + 1
        _paint_if_open(g, out, detour_row, entry)
        _paint_horizontal(g, out, detour_row, gate["left"], gate["right"])
        _paint_if_open(g, out, path_row, gate["right"])
        current = gate["right"]

    _paint_horizontal(g, out, path_row, current + 1, g.shape[1] - 1)
    return out


def _build_gate_specs(n_gates: int) -> list[dict] | None:
    for _ in range(300):
        sides = [-1, 1]
        while len(sides) < n_gates:
            sides.append(random.choice([-1, 1]))
        random.shuffle(sides)

        specs: list[dict] = []
        relations: list[int] = []
        for side in sides:
            sym_mask = _sample_library_mask(True)
            asym_mask = _sample_library_mask(False)
            relations.append(int(np.sign(int(sym_mask.sum()) - int(asym_mask.sum()))))

            if side == -1:
                upper_mask, lower_mask = sym_mask, asym_mask
            else:
                upper_mask, lower_mask = asym_mask, sym_mask

            band_w = max(upper_mask.shape[1], lower_mask.shape[1]) + random.randint(0, 1)
            band_w = min(4, max(int(band_w), upper_mask.shape[1], lower_mask.shape[1]))
            specs.append(
                {
                    "side": side,
                    "upper_mask": upper_mask,
                    "lower_mask": lower_mask,
                    "upper_gap": random.choice((2, 3)),
                    "lower_gap": random.choice((2, 3)),
                    "band_w": int(band_w),
                }
            )

        if n_gates >= 2 and len(set(relations)) == 1:
            continue
        return specs
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, PATH, SPRITE}:
        return False

    try:
        path_row, gates = _extract_gate_records(grid)
    except AssertionError:
        return False

    if not (2 <= len(gates) <= 4):
        return False
    if {gate["side"] for gate in gates} != {-1, 1}:
        return False
    if not np.all(grid[path_row, 1:] == BG):
        return False

    for prev, cur in zip(gates, gates[1:]):
        if cur["left"] - prev["right"] - 1 < 3:
            return False

    for gate in gates:
        entry = gate["left"] - 1
        if entry < 0:
            return False
        detour_row = path_row - 1 if gate["side"] == -1 else path_row + 1
        if not (0 <= detour_row < grid.shape[0]):
            return False
        if np.any(grid[detour_row, entry : gate["right"] + 1] != BG):
            return False

    expected = grid.copy()
    for r, c in _expected_path_coords(path_row, gates, grid.shape[1]):
        if int(expected[r, c]) in (BG, PATH):
            expected[r, c] = PATH
    if not np.array_equal(main(grid), expected):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n_gates = random.randint(2, 4)
    specs = _build_gate_specs(n_gates)
    if specs is None:
        return None

    left_margin = random.randint(3, 5)
    right_margin = random.randint(2, 5)
    gaps = [random.randint(3, 5) for _ in range(n_gates - 1)]
    width = left_margin + right_margin + sum(spec["band_w"] for spec in specs) + sum(gaps)
    if width > 30:
        return None

    upper_margin = max(spec["upper_gap"] + spec["upper_mask"].shape[0] - 1 for spec in specs)
    lower_margin = max(spec["lower_gap"] + spec["lower_mask"].shape[0] - 1 for spec in specs)
    min_height = upper_margin + lower_margin + 1
    if min_height > 24:
        return None

    height = random.randint(max(14, min_height), min(24, 30))
    path_row = random.randint(upper_margin, height - 1 - lower_margin)

    grid = np.zeros((height, width), dtype=int)
    grid[path_row, 0] = PATH

    cursor = left_margin
    for i, spec in enumerate(specs):
        band_w = spec["band_w"]
        upper_mask = spec["upper_mask"]
        lower_mask = spec["lower_mask"]

        upper_left = cursor + random.randint(0, band_w - upper_mask.shape[1])
        lower_left = cursor + random.randint(0, band_w - lower_mask.shape[1])
        upper_top = path_row - spec["upper_gap"] - upper_mask.shape[0] + 1
        lower_top = path_row + spec["lower_gap"]

        _paint_mask(grid, upper_mask, upper_top, upper_left, SPRITE)
        _paint_mask(grid, lower_mask, lower_top, lower_left, SPRITE)

        cursor += band_w
        if i < len(gaps):
            cursor += gaps[i]

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-gated routing grid")
