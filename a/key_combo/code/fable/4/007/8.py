from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GRAY = 5
ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 500
PLACE_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_component(g: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    coord_set = {(int(r), int(c)) for r, c in coords}
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coord_set:
        mask[r - r0, c - c0] = True
    colors = {int(g[r, c]) for r, c in coord_set}
    assert len(colors) == 1, "each sprite instance must be monochrome"
    return mask, box


def _block_reduce(mask: np.ndarray, k: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % k != 0 or w % k != 0:
        return None
    reduced = np.zeros((h // k, w // k), dtype=bool)
    for rr in range(0, h, k):
        for cc in range(0, w, k):
            block = mask[rr : rr + k, cc : cc + k]
            value = bool(block[0, 0])
            if not bool(np.all(block == value)):
                return None
            reduced[rr // k, cc // k] = value
    return reduced if bool(reduced.any()) else None


def _primitive(mask: np.ndarray) -> tuple[np.ndarray, int]:
    limit = min(mask.shape)
    for k in range(limit, 0, -1):
        reduced = _block_reduce(mask, k)
        if reduced is not None:
            return reduced, k
    raise AssertionError("sprite mask must be nonempty")


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return int(mask.shape[0]), int(mask.shape[1]), tuple(int(v) for v in mask.ravel())


def _scaled_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask.copy()
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def _paint_scaled_mask(
    out: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    scale: int,
    color: int,
) -> None:
    patch = _scaled_mask(mask, scale)
    r0, c0 = origin
    h, w = out.shape
    ph, pw = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + ph <= h and c0 + pw <= w, "replacement must stay in bounds"
    region = out[r0 : r0 + ph, c0 : c0 + pw]
    region[patch] = int(color)


def _sprite_records(g: np.ndarray, bg: int) -> list[dict]:
    comps = _components((g != bg) & (g != GRAY))
    assert len(comps) >= 3, "expected several sprite instances plus one gray marker"
    records: list[dict] = []
    for coords in comps:
        mask, box = _mask_from_component(g, coords)
        primitive, scale = _primitive(mask)
        key = _mask_key(primitive)
        color = int(g[int(coords[0, 0]), int(coords[0, 1])])
        records.append(
            {
                "coords": coords,
                "bbox": box,
                "mask": mask,
                "primitive": primitive,
                "key": key,
                "scale": int(scale),
                "color": color,
            }
        )
    return records


def _marker_touched_record(g: np.ndarray, records: list[dict]) -> int:
    markers = np.argwhere(g == GRAY)
    assert markers.shape == (1, 2), "expected exactly one gray marker pixel"
    mr, mc = map(int, markers[0])
    cell_to_record: dict[tuple[int, int], int] = {}
    for idx, rec in enumerate(records):
        for r, c in rec["coords"]:
            cell_to_record[(int(r), int(c))] = idx

    touched = {
        cell_to_record[(rr, cc)]
        for rr, cc in _neighbors4(mr, mc, *g.shape)
        if (rr, cc) in cell_to_record
    }
    assert len(touched) == 1, "gray marker must touch exactly one sprite"
    return touched.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the entire sprite family selected by the touching gray marker with the opposite prototype.

    Input structure:
    - The background is the most common color, and one gray pixel touches exactly one sprite by 4-neighborhood.
    - All other foreground objects are separated monochrome scaled copies of exactly two small prototype masks.
    - Each family has visible smallest copies, while larger copies use integer cell-block scaling.

    Rule:
    - Ignore color and reduce every sprite to its scale-independent prototype mask.
    - Find the prototype family of the unique sprite touched by the gray marker.
    - Every sprite in that same family is cleared and redrawn at its original bbox top-left using the other prototype, its own scale, and its own color.
    - Sprites from the other family, the gray marker, and the background remain unchanged.

    Key insight:
    - The marker selects a shape family rather than a single object, so the transformation must classify sprites by normalized shape before preserving each instance's color, scale, and anchor during replacement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg != GRAY, "gray is reserved for the selector marker"

    records = _sprite_records(g, bg)
    prototypes: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}
    for rec in records:
        prototypes.setdefault(rec["key"], rec["primitive"])
    assert len(prototypes) == 2, "expected exactly two scale-independent sprite families"

    selected_idx = _marker_touched_record(g, records)
    selected_key = records[selected_idx]["key"]
    other_keys = [key for key in prototypes if key != selected_key]
    assert len(other_keys) == 1, "selected family must have a unique opposite prototype"
    replacement = prototypes[other_keys[0]]

    out = g.copy()
    for rec in records:
        if rec["key"] != selected_key:
            continue
        for r, c in rec["coords"]:
            out[int(r), int(c)] = bg
        r0, c0, _r1, _c1 = rec["bbox"]
        _paint_scaled_mask(out, replacement, (r0, c0), rec["scale"], rec["color"])
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _irreducible(mask: np.ndarray) -> bool:
    _base, scale = _primitive(mask)
    return scale == 1


def _random_polyomino() -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(3, min(7, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, h, w):
                    if (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop(mask)
        if mask.shape[0] >= 2 and mask.shape[1] >= 2 and _connected(mask) and _irreducible(mask):
            return mask
    raise RuntimeError("failed to make a prototype")


def _make_prototypes() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        a = _random_polyomino()
        b = _random_polyomino()
        if _mask_key(a) == _mask_key(b):
            continue
        if _mask_key(_primitive(a)[0]) == _mask_key(_primitive(b)[0]):
            continue
        return a, b
    raise RuntimeError("failed to make distinct prototypes")


def _dilated_rect_cells(r0: int, c0: int, h: int, w: int, grid_h: int, grid_w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(max(0, r0 - PLACE_PADDING), min(grid_h, r0 + h + PLACE_PADDING))
        for c in range(max(0, c0 - PLACE_PADDING), min(grid_w, c0 + w + PLACE_PADDING))
    }


def _mask_cells(mask: np.ndarray, origin: tuple[int, int], scale: int) -> set[tuple[int, int]]:
    r0, c0 = origin
    patch = _scaled_mask(mask, scale)
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch)}


def _place_instances(
    prototypes: tuple[np.ndarray, np.ndarray],
    instances: list[dict],
    h: int,
    w: int,
) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []
    ordered = sorted(instances, key=lambda x: x["reserve_area"], reverse=True)
    for inst in ordered:
        fam = int(inst["family"])
        scale = int(inst["scale"])
        cur = prototypes[fam]
        alt = prototypes[1 - fam]
        reserve_h = max(cur.shape[0], alt.shape[0]) * scale
        reserve_w = max(cur.shape[1], alt.shape[1]) * scale
        if reserve_h + 2 > h or reserve_w + 2 > w:
            return None
        anchors = [
            (r, c)
            for r in range(1, h - reserve_h)
            for c in range(1, w - reserve_w)
        ]
        random.shuffle(anchors)
        chosen = None
        for r0, c0 in anchors:
            reserve = _dilated_rect_cells(r0, c0, reserve_h, reserve_w, h, w)
            if reserve & occupied:
                continue
            chosen = (r0, c0, reserve)
            break
        if chosen is None:
            return None
        r0, c0, reserve = chosen
        occupied |= reserve
        inst = dict(inst)
        inst["origin"] = (r0, c0)
        inst["cells"] = _mask_cells(cur, (r0, c0), scale)
        inst["replacement_cells"] = _mask_cells(alt, (r0, c0), scale)
        placed.append(inst)
    return placed


def _draw_instances(
    h: int,
    w: int,
    bg: int,
    prototypes: tuple[np.ndarray, np.ndarray],
    instances: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for inst in instances:
        fam = int(inst["family"])
        _paint_scaled_mask(grid, prototypes[fam], inst["origin"], int(inst["scale"]), int(inst["color"]))
    return grid


def _touching_marker_options(grid: np.ndarray, inst: dict, all_instances: list[dict]) -> list[tuple[int, int]]:
    h, w = grid.shape
    other_cells = set().union(*(set(x["cells"]) for x in all_instances if x is not inst))
    blocked = set().union(*(set(x["cells"]) | set(x["replacement_cells"]) for x in all_instances))
    options: set[tuple[int, int]] = set()
    for r, c in inst["cells"]:
        for rr, cc in _neighbors4(r, c, h, w):
            if int(grid[rr, cc]) != int(grid[0, 0]):
                continue
            if (rr, cc) in blocked:
                continue
            touches_other = any((nr, nc) in other_cells for nr, nc in _neighbors4(rr, cc, h, w))
            if not touches_other:
                options.add((rr, cc))
    return list(options)


def _render_expected(
    grid: np.ndarray,
    bg: int,
    prototypes: tuple[np.ndarray, np.ndarray],
    instances: list[dict],
    selected_family: int,
) -> np.ndarray:
    out = grid.copy()
    replacement = prototypes[1 - selected_family]
    for inst in instances:
        if int(inst["family"]) != selected_family:
            continue
        for r, c in inst["cells"]:
            out[r, c] = bg
        _paint_scaled_mask(out, replacement, inst["origin"], int(inst["scale"]), int(inst["color"]))
    return out


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    instances: list[dict],
    selected_family: int,
) -> bool:
    if np.array_equal(grid, expected):
        return False
    selected = [x for x in instances if int(x["family"]) == selected_family]
    other = [x for x in instances if int(x["family"]) != selected_family]
    if len(selected) < 2 or len(other) < 2:
        return False
    if not any(int(x["scale"]) > 1 for x in selected):
        return False
    for inst in other:
        for r, c in inst["cells"]:
            if int(expected[r, c]) != int(grid[r, c]):
                return False
    for inst in selected:
        changed = any(int(expected[r, c]) != int(grid[r, c]) for r, c in inst["cells"])
        changed = changed or any(int(expected[r, c]) != int(grid[r, c]) for r, c in inst["replacement_cells"])
        if not changed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    colors = [c for c in ARC_COLORS if c not in (bg, GRAY)]
    prototypes = _make_prototypes()
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    selected_family = random.randrange(2)

    instances: list[dict] = []
    counts = [random.randint(2, 3), random.randint(2, 3)]
    for fam in (0, 1):
        scales = [1]
        scales.append(random.choice([2, 2, 3]))
        while len(scales) < counts[fam]:
            scales.append(random.choice([1, 2, 3]))
        for scale in scales:
            color = random.choice(colors)
            if random.random() < 0.35 and instances:
                color = random.choice(instances)["color"]
            reserve_h = max(prototypes[fam].shape[0], prototypes[1 - fam].shape[0]) * scale
            reserve_w = max(prototypes[fam].shape[1], prototypes[1 - fam].shape[1]) * scale
            instances.append(
                {
                    "family": fam,
                    "scale": scale,
                    "color": color,
                    "reserve_area": reserve_h * reserve_w,
                }
            )

    placed = _place_instances(prototypes, instances, h, w)
    if placed is None:
        return None
    grid = _draw_instances(h, w, bg, prototypes, placed)

    selected_instances = [inst for inst in placed if int(inst["family"]) == selected_family]
    random.shuffle(selected_instances)
    marker = None
    for inst in selected_instances:
        options = _touching_marker_options(grid, inst, placed)
        if options:
            marker = random.choice(options)
            break
    if marker is None:
        return None
    grid[marker] = GRAY

    expected = _render_expected(grid, bg, prototypes, placed, selected_family)
    if not _semantic_ok(grid, expected, placed, selected_family):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
