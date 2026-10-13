from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
PALETTE = (BLUE, RED, GREEN)
SLOT_BY_KIND = {"H": 0, "V": 1, "D": 2}
KIND_ORDER = ("H", "V", "D")
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("empty mask has no tight crop")
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    coords_list: list[np.ndarray] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_list.append(np.array(comp, dtype=int))
    return coords_list


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _anti_diag_reflect(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2).T


def _analyze_mask(mask: np.ndarray) -> tuple[str, str]:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    diag_main = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T))
    diag_anti = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, _anti_diag_reflect(mask)))
    diagonal = diag_main or diag_anti

    if sum((horizontal, vertical, diagonal)) != 1:
        raise AssertionError("object must have exactly one symmetry family")
    if diagonal and diag_main == diag_anti:
        raise AssertionError("diagonal object must have exactly one diagonal axis")

    if horizontal:
        return "H", "horizontal"
    if vertical:
        return "V", "vertical"
    return "D", "main" if diag_main else "anti"


def _reflect_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "horizontal":
        return np.flipud(mask)
    if axis == "vertical":
        return np.fliplr(mask)
    if axis == "main":
        return mask.T
    if axis == "anti":
        return _anti_diag_reflect(mask)
    raise ValueError(f"unknown axis: {axis}")


def _stamp_centered(dst: np.ndarray, sprite: np.ndarray, row_bounds: tuple[int, int], col_bounds: tuple[int, int]) -> None:
    r0, r1 = row_bounds
    c0, c1 = col_bounds
    sh, sw = sprite.shape
    th = r1 - r0 + 1
    tw = c1 - c0 + 1
    if sh > th or sw > tw:
        raise AssertionError("sprite does not fit inside target slot")
    top = r0 + (th - sh) // 2
    left = c0 + (tw - sw) // 2
    view = dst[top : top + sh, left : left + sw]
    view[sprite != BACKGROUND] = sprite[sprite != BACKGROUND]


def _parse_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], tuple[int, int]]:
    h, w = grid.shape
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == GRAY)]
    if len(divider_cols) != 1:
        raise AssertionError("expected one full-height gray divider column")
    divider_col = divider_cols[0]
    if divider_col <= 0 or divider_col >= w - 1:
        raise AssertionError("divider must split left and right panels")

    right = grid[:, divider_col + 1 :]
    divider_rows = [r for r in range(h) if right.shape[1] > 0 and np.all(right[r, :] == GRAY)]
    if len(divider_rows) != 2:
        raise AssertionError("expected two gray horizontal slot dividers on the right panel")

    bounds = [-1, divider_rows[0], divider_rows[1], h]
    row_intervals = [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(3)]
    heights = [r1 - r0 + 1 for r0, r1 in row_intervals]
    if min(heights) <= 0 or len(set(heights)) != 1:
        raise AssertionError("right slots must have equal positive heights")
    return divider_col, row_intervals, (divider_col + 1, w - 1)


def _extract_left_objects(grid: np.ndarray, divider_col: int) -> list[dict]:
    left = grid[:, :divider_col]
    colors = sorted(set(map(int, np.unique(left))) - {BACKGROUND})
    if colors != sorted(PALETTE):
        raise AssertionError("left panel must contain exactly blue, red, and green objects")

    objects: list[dict] = []
    for color in colors:
        comps = _components_of_color(left, color)
        if len(comps) != 1:
            raise AssertionError("each left-panel color must form exactly one component")
        coords = comps[0]
        if not 3 <= int(coords.shape[0]) <= 8:
            raise AssertionError("sprite area must be between 3 and 8")
        bbox = _bbox(coords)
        mask = _crop_mask(coords, bbox)
        kind, axis = _analyze_mask(mask)
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "kind": kind,
                "axis": axis,
            }
        )

    kinds = [obj["kind"] for obj in objects]
    if sorted(kinds) != sorted(KIND_ORDER):
        raise AssertionError("expected exactly one H, one V, and one D sprite")
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each left-panel sprite to the slot indicated by its reflection-symmetry type, then stamp its self-axis reflection there.

    Input structure:
    - A black grid is split by one full-height gray divider column into a left sprite panel and a right slot panel.
    - The left panel contains exactly three monochrome 4-connected sprites in blue, red, and green; each sprite has area 3-8.
    - The right panel is divided by two gray horizontal rows into three equal-height empty slots.
    - The three left sprites have three different reflection-symmetry families: horizontal-axis, vertical-axis, and diagonal-axis exactly once each.

    Rule:
    - Classify each left sprite by its unique reflection symmetry family, using its tight bounding-box mask.
    - Route the horizontally symmetric sprite to the top slot, the vertically symmetric sprite to the middle slot, and the diagonally symmetric sprite to the bottom slot.
    - Reflect the sprite across its own symmetry axis, keep its original color, and stamp the reflected result centered in the target slot.
    - Preserve the gray divider column, the gray slot-divider rows, and the original left-panel sprites.

    Key insight:
    - The same symmetry analysis has two jobs at once: it selects the destination slot and also identifies which axis the copied sprite must reflect across before stamping.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")
    if max(g.shape) > 30:
        raise AssertionError("ARC grids must fit within 30x30")

    divider_col, slot_rows, slot_cols = _parse_structure(g)
    objects = _extract_left_objects(g, divider_col)

    out = g.copy()
    by_kind = {obj["kind"]: obj for obj in objects}
    for kind in KIND_ORDER:
        obj = by_kind[kind]
        reflected = _reflect_mask(obj["mask"], obj["axis"])
        sprite = np.where(reflected, int(obj["color"]), BACKGROUND).astype(int)
        _stamp_centered(out, sprite, slot_rows[SLOT_BY_KIND[kind]], slot_cols)

    return out


def _orbits_horizontal(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range((h + 1) // 2):
        rr = h - 1 - r
        for c in range(w):
            if r == rr:
                orbit = ((r, c),)
            else:
                orbit = ((r, c), (rr, c))
            orbits.append(orbit)
    return orbits


def _orbits_main_diag(n: int) -> list[tuple[tuple[int, int], ...]]:
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(r, n):
            if r == c:
                orbit = ((r, c),)
            else:
                orbit = ((r, c), (c, r))
            orbits.append(orbit)
    return orbits


def _enumerate_masks(orbits: list[tuple[tuple[int, int], ...]], shape: tuple[int, int], wanted_kind: str) -> list[np.ndarray]:
    h, w = shape
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    masks: list[np.ndarray] = []
    for subset in range(1, 1 << len(orbits)):
        mask = np.zeros(shape, dtype=bool)
        area = 0
        for bit, orbit in enumerate(orbits):
            if not (subset & (1 << bit)):
                continue
            area += len(orbit)
            if area > 8:
                break
            for r, c in orbit:
                mask[r, c] = True
        if area < 3 or area > 8 or not _connected(mask):
            continue
        mask = _tight_mask(mask)
        try:
            kind, _axis = _analyze_mask(mask)
        except AssertionError:
            continue
        if kind != wanted_kind:
            continue
        key = (mask.tobytes(), mask.shape)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    return masks


def _build_template_library() -> dict[str, list[np.ndarray]]:
    horizontal: list[np.ndarray] = []
    for h in range(3, 6):
        for w in range(2, 6):
            horizontal.extend(_enumerate_masks(_orbits_horizontal(h, w), (h, w), "H"))

    vertical: list[np.ndarray] = []
    seen_v: set[tuple[bytes, tuple[int, int]]] = set()
    for mask in horizontal:
        transposed = mask.T.copy()
        try:
            kind, _axis = _analyze_mask(transposed)
        except AssertionError:
            continue
        if kind != "V":
            continue
        key = (transposed.tobytes(), transposed.shape)
        if key in seen_v:
            continue
        seen_v.add(key)
        vertical.append(transposed)

    diagonal: list[np.ndarray] = []
    seen_d: set[tuple[bytes, tuple[int, int]]] = set()
    for n in range(3, 6):
        for mask in _enumerate_masks(_orbits_main_diag(n), (n, n), "D"):
            for variant in (mask, np.fliplr(mask)):
                try:
                    kind, _axis = _analyze_mask(variant)
                except AssertionError:
                    continue
                if kind != "D":
                    continue
                key = (variant.tobytes(), variant.shape)
                if key in seen_d:
                    continue
                seen_d.add(key)
                diagonal.append(variant.copy())

    if not horizontal or not vertical or not diagonal:
        raise RuntimeError("failed to build symmetry template library")
    return {"H": horizontal, "V": vertical, "D": diagonal}


TEMPLATES = _build_template_library()


def _sample_mask(kind: str, max_h: int, max_w: int) -> np.ndarray:
    candidates = [mask for mask in TEMPLATES[kind] if mask.shape[0] <= max_h and mask.shape[1] <= max_w]
    if not candidates:
        raise ValueError("no template fits requested slot size")
    return random.choice(candidates).copy()


def _slot_row_intervals(slot_h: int) -> list[tuple[int, int]]:
    return [
        (0, slot_h - 1),
        (slot_h + 1, 2 * slot_h),
        (2 * slot_h + 2, 3 * slot_h + 1),
    ]


def _place_left_objects(
    grid: np.ndarray,
    left_w: int,
    slot_h: int,
    objects: dict[str, dict],
) -> list[tuple[str, int]]:
    row_bands = _slot_row_intervals(slot_h)
    band_order = list(KIND_ORDER)
    while True:
        random.shuffle(band_order)
        if tuple(band_order) != KIND_ORDER:
            break

    placements: list[tuple[str, int]] = []
    for band_idx, kind in enumerate(band_order):
        mask = objects[kind]["mask"]
        color = int(objects[kind]["color"])
        band_r0, band_r1 = row_bands[band_idx]
        mh, mw = mask.shape
        r = random.randint(band_r0, band_r1 - mh + 1)
        c = random.randint(0, left_w - mw)
        sprite = np.where(mask, color, BACKGROUND).astype(int)
        window = grid[r : r + mh, c : c + mw]
        if np.any(window != BACKGROUND):
            raise AssertionError("left object placement overlapped unexpectedly")
        window[sprite != BACKGROUND] = sprite[sprite != BACKGROUND]
        placements.append((kind, r))
        objects[kind]["placed_row"] = r
        objects[kind]["placed_col"] = c
    return placements


def _expected_from_metadata(inp: np.ndarray, divider_col: int, slot_rows: list[tuple[int, int]], slot_cols: tuple[int, int], objects: dict[str, dict]) -> np.ndarray:
    out = inp.copy()
    for kind in KIND_ORDER:
        meta = objects[kind]
        reflected = _reflect_mask(meta["mask"], meta["axis"])
        sprite = np.where(reflected, int(meta["color"]), BACKGROUND).astype(int)
        _stamp_centered(out, sprite, slot_rows[SLOT_BY_KIND[kind]], slot_cols)
    return out


def _build_candidate() -> tuple[np.ndarray, dict[str, dict]]:
    slot_h = random.randint(4, 7)
    slot_w = random.randint(6, 10)
    h = 3 * slot_h + 2

    objects: dict[str, dict] = {}
    for kind in KIND_ORDER:
        mask = _sample_mask(kind, slot_h, slot_w)
        objects[kind] = {"mask": mask}

    color_assignment = list(PALETTE)
    random.shuffle(color_assignment)
    for kind, color in zip(KIND_ORDER, color_assignment):
        objects[kind]["color"] = int(color)
        _kind, axis = _analyze_mask(objects[kind]["mask"])
        objects[kind]["axis"] = axis

    max_left_width = max(int(meta["mask"].shape[1]) for meta in objects.values())
    left_w = random.randint(max(6, max_left_width), 10)
    w = left_w + 1 + slot_w

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[:, left_w] = GRAY
    grid[slot_h, left_w + 1 :] = GRAY
    grid[2 * slot_h + 1, left_w + 1 :] = GRAY

    _place_left_objects(grid, left_w, slot_h, objects)
    return grid, objects


def _validate_candidate(inp: np.ndarray, objects: dict[str, dict]) -> bool:
    divider_col, slot_rows, slot_cols = _parse_structure(inp)
    parsed = _extract_left_objects(inp, divider_col)
    parsed_by_kind = {obj["kind"]: obj for obj in parsed}
    if set(parsed_by_kind) != set(KIND_ORDER):
        return False

    row_order = [kind for kind, _r in sorted(((k, v["placed_row"]) for k, v in objects.items()), key=lambda x: x[1])]
    if tuple(row_order) == KIND_ORDER:
        return False

    expected = _expected_from_metadata(inp, divider_col, slot_rows, slot_cols, objects)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(actual[:, :divider_col], inp[:, :divider_col]):
        return False

    for kind in KIND_ORDER:
        slot = actual[slot_rows[SLOT_BY_KIND[kind]][0] : slot_rows[SLOT_BY_KIND[kind]][1] + 1, slot_cols[0] : slot_cols[1] + 1]
        if np.count_nonzero(slot == int(objects[kind]["color"])) != int(objects[kind]["mask"].sum()):
            return False
    if np.count_nonzero(actual[:, divider_col + 1 :] != inp[:, divider_col + 1 :]) < 9:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, objects = _build_candidate()
        if _validate_candidate(inp, objects):
            return inp
    raise RuntimeError("failed to generate a valid input")
