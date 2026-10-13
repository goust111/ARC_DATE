from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Canvas and object budgets.
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (1, 3)
OBJECT_HEIGHT_RANGE = (5, 10)
OBJECT_WIDTH_RANGE = (5, 10)
PLACEMENT_MARGIN = 2
GENERATION_ATTEMPTS = 400
OBJECT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
QUARTER_ATTEMPTS = 120
DELETE_ATTEMPTS = 120

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bool_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_bool_components(mask)) == 1


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _grow_connected_subset(allowed: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    comps = [comp for comp in _bool_components(allowed) if len(comp) >= size]
    if not comps:
        return None

    comp = random.choice(comps)
    comp_set = set(comp)
    seed = random.choice(comp)
    chosen = {seed}
    frontier = {cell for cell in _neighbors4(seed[0], seed[1], *allowed.shape) if cell in comp_set}

    while len(chosen) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nb in _neighbors4(cell[0], cell[1], *allowed.shape):
            if nb in comp_set and nb not in chosen:
                frontier.add(nb)
    return chosen


def _make_quarter_mask(qh: int, qw: int) -> np.ndarray | None:
    area = qh * qw
    lo = max(max(qh, qw), (area + 2) // 3)
    hi = min(area - 1, max(lo, (3 * area) // 4))
    if lo > hi:
        return None

    allowed = np.ones((qh, qw), dtype=bool)
    for _ in range(QUARTER_ATTEMPTS):
        target = random.randint(lo, hi)
        cells = _grow_connected_subset(allowed, target)
        if cells is None:
            continue
        mask = np.zeros((qh, qw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != qh - 1:
            continue
        if cols.min() != 0 or cols.max() != qw - 1:
            continue
        return mask
    return None


def _reflect_biaxial(quarter: np.ndarray, h: int, w: int) -> np.ndarray:
    full = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(quarter):
        for rr in {int(r), h - 1 - int(r)}:
            for cc in {int(c), w - 1 - int(c)}:
                full[rr, cc] = True
    return full


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    assert horizontal ^ vertical, "pattern must have exactly one reflection axis"
    return "horizontal" if horizontal else "vertical"


def _make_source_mask(full: np.ndarray, keep_symmetry: str) -> np.ndarray | None:
    h, w = full.shape
    for _ in range(DELETE_ATTEMPTS):
        allowed = np.zeros_like(full, dtype=bool)
        if keep_symmetry == "vertical":
            side = random.choice(("top", "bottom"))
            row_range = range(1, h // 2) if side == "top" else range((h + 1) // 2, h - 1)
            col_range = range(1, (w + 1) // 2)
            for r in row_range:
                for c in col_range:
                    allowed[r, c] = full[r, c]
        else:
            side = random.choice(("left", "right"))
            row_range = range(1, (h + 1) // 2)
            col_range = range(1, w // 2) if side == "left" else range((w + 1) // 2, w - 1)
            for r in row_range:
                for c in col_range:
                    allowed[r, c] = full[r, c]

        available = int(allowed.sum())
        if available == 0:
            continue

        size = random.randint(1, min(4, available))
        blob = _grow_connected_subset(allowed, size)
        if blob is None:
            continue

        deleted: set[tuple[int, int]] = set()
        if keep_symmetry == "vertical":
            for r, c in blob:
                deleted.add((r, c))
                deleted.add((r, w - 1 - c))
        else:
            for r, c in blob:
                deleted.add((r, c))
                deleted.add((h - 1 - r, c))

        src = full.copy()
        for r, c in deleted:
            src[r, c] = False

        if not src.any() or not _is_connected(src):
            continue

        if _bbox_from_mask(src) != (0, h - 1, 0, w - 1):
            continue

        try:
            if _symmetry_kind(src) != keep_symmetry:
                continue
        except AssertionError:
            continue

        return src
    return None


def _build_object() -> dict | None:
    for _ in range(OBJECT_ATTEMPTS):
        h = random.randint(*OBJECT_HEIGHT_RANGE)
        w = random.randint(*OBJECT_WIDTH_RANGE)
        quarter = _make_quarter_mask((h + 1) // 2, (w + 1) // 2)
        if quarter is None:
            continue

        full = _reflect_biaxial(quarter, h, w)
        if not _is_connected(full):
            continue
        if _bbox_from_mask(full) != (0, h - 1, 0, w - 1):
            continue

        keep_symmetry = random.choice(("horizontal", "vertical"))
        src = _make_source_mask(full, keep_symmetry)
        if src is None:
            continue

        missing = full & ~src
        if not missing.any():
            continue

        return {
            "h": h,
            "w": w,
            "src": src,
            "full": full,
            "missing": missing,
            "keep_symmetry": keep_symmetry,
        }
    return None


def _component_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if int(g[nr, nc]) == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": len(coords),
                    "bbox": (r0, r1, c0, c1),
                    "mask": mask,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][2], comp["color"]))
    return comps


def _min_component_distance(a: dict, b: dict) -> int:
    best = 10**9
    for ar, ac in a["coords"]:
        for br, bc in b["coords"]:
            dist = abs(ar - br) + abs(ac - bc)
            if dist < best:
                best = dist
    return best


def _nearest_marker(pattern: dict, components: list[dict]) -> dict:
    ranked = []
    for comp in components:
        if comp is pattern or comp["color"] == pattern["color"]:
            continue
        ranked.append((_min_component_distance(pattern, comp), comp))
    assert ranked, "each pattern needs a nearby marker"
    ranked.sort(key=lambda item: item[0])
    if len(ranked) > 1:
        assert ranked[0][0] < ranked[1][0], "nearest non-matching component must be unique"
    marker = ranked[0][1]
    assert marker["size"] == 1, "nearest non-matching component must be the marker pixel"
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each monochrome pattern across its missing symmetry axis using the nearby marker color.

    Input structure
    - Black (0) is background.
    - Each non-singleton monochrome component is symmetric in exactly one of
      the two orthogonal directions.
    - The nearest differently colored component to each pattern is a single
      marker pixel that supplies the completion color.

    Rule
    - Keep all original pixels unchanged.
    - Detect whether the pattern already has horizontal or vertical symmetry.
    - Mirror the pattern across the orthogonal center line of its tight bbox.
    - Paint only the missing mirrored cells, using the nearby marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    components = _component_records(g)
    patterns = [comp for comp in components if comp["size"] > 1]
    assert patterns, "expected at least one pattern component"

    out = g.copy()
    for pattern in patterns:
        kind = _symmetry_kind(pattern["mask"])
        marker = _nearest_marker(pattern, components)
        fill_color = int(marker["color"])
        r0, r1, c0, c1 = pattern["bbox"]

        for r, c in pattern["coords"]:
            if kind == "horizontal":
                rr, cc = r, c0 + c1 - c
            else:
                rr, cc = r0 + r1 - r, c

            current = int(out[rr, cc])
            assert current in (BG, pattern["color"]), "completion must not overwrite another object"
            if current == BG:
                out[rr, cc] = fill_color

    return out


def _marker_candidates(r0: int, c0: int, h: int, w: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    top = r0 - 2
    bottom = r0 + h + 1
    left = c0 - 2
    right = c0 + w + 1

    if top >= 0:
        cells.extend((top, c) for c in range(c0, c0 + w))
    if bottom < H:
        cells.extend((bottom, c) for c in range(c0, c0 + w))
    if left >= 0:
        cells.extend((r, left) for r in range(r0, r0 + h))
    if right < W:
        cells.extend((r, right) for r in range(r0, r0 + h))

    seen = set()
    unique: list[tuple[int, int]] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def _paint_object(
    inp: np.ndarray,
    out: np.ndarray,
    obj: dict,
    r0: int,
    c0: int,
    pattern_color: int,
    marker_color: int,
    marker_pos: tuple[int, int],
) -> None:
    for r, c in np.argwhere(obj["src"]):
        rr = r0 + int(r)
        cc = c0 + int(c)
        inp[rr, cc] = pattern_color
        out[rr, cc] = pattern_color
    for r, c in np.argwhere(obj["missing"]):
        out[r0 + int(r), c0 + int(c)] = marker_color
    mr, mc = marker_pos
    inp[mr, mc] = marker_color
    out[mr, mc] = marker_color


def _orientation_list(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("horizontal", "vertical"))]
    kinds = ["horizontal", "vertical"]
    while len(kinds) < count:
        kinds.append(random.choice(("horizontal", "vertical")))
    random.shuffle(kinds)
    return kinds


def _build_object_for_kind(kind: str) -> dict | None:
    for _ in range(OBJECT_ATTEMPTS):
        obj = _build_object()
        if obj is not None and obj["keep_symmetry"] == kind:
            return obj
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * count)
    kinds = _orientation_list(count)

    objects: list[tuple[dict, int, int]] = []
    for idx in range(count):
        obj = _build_object_for_kind(kinds[idx])
        if obj is None:
            return None
        objects.append((obj, colors[2 * idx], colors[2 * idx + 1]))

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    for obj, pattern_color, marker_color in objects:
        h = int(obj["h"])
        w = int(obj["w"])
        if h + 2 * PLACEMENT_MARGIN > H or w + 2 * PLACEMENT_MARGIN > W:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(PLACEMENT_MARGIN, H - h - PLACEMENT_MARGIN)
            c0 = random.randint(PLACEMENT_MARGIN, W - w - PLACEMENT_MARGIN)

            rr0 = r0 - PLACEMENT_MARGIN
            rr1 = r0 + h + PLACEMENT_MARGIN
            cc0 = c0 - PLACEMENT_MARGIN
            cc1 = c0 + w + PLACEMENT_MARGIN
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            candidates = [cell for cell in _marker_candidates(r0, c0, h, w, H, W) if not reserved[cell]]
            if not candidates:
                continue
            marker_pos = random.choice(candidates)

            _paint_object(inp, out, obj, r0, c0, pattern_color, marker_color, marker_pos)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
