from __future__ import annotations

import math
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SYMMETRY_TYPES = ("lr", "ud", "quad")
PALETTE = tuple(range(10))

MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 220
BROKEN_ATTEMPTS = 260
PLACEMENT_ATTEMPTS = 220
GRID_SIDE_RANGE = (20, 30)


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)
    assert len(order) == 1 or counts[order[-1]] > counts[order[-2]]
    return int(colors[order[-1]])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _is_connected(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    return _cells_connected(coords)


def _symmetry_type(mask: np.ndarray) -> str | None:
    lr = np.array_equal(mask, np.fliplr(mask))
    ud = np.array_equal(mask, np.flipud(mask))
    if lr and ud:
        return "quad"
    if lr:
        return "lr"
    if ud:
        return "ud"
    return None


def _complete_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    out = mask.copy()
    if kind in {"lr", "quad"}:
        out |= np.fliplr(out)
    if kind in {"ud", "quad"}:
        out |= np.flipud(out)
    return out


def _orbit(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c))}
    changed = True
    while changed:
        changed = False
        for rr, cc in tuple(cells):
            if kind in {"lr", "quad"}:
                mirrored = (rr, w - 1 - cc)
                if mirrored not in cells:
                    cells.add(mirrored)
                    changed = True
            if kind in {"ud", "quad"}:
                mirrored = (h - 1 - rr, cc)
                if mirrored not in cells:
                    cells.add(mirrored)
                    changed = True
    return tuple(sorted(cells))


def _all_orbits(kind: str, h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orb = _orbit(kind, h, w, r, c)
            if orb not in seen:
                seen.add(orb)
                orbits.append(orb)
    return orbits


def _orbit_touches(cells: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...]) -> bool:
    for r, c in orbit:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in cells:
                return True
    return False


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _find_legend(components: list[dict[str, object]], shape: tuple[int, int]) -> tuple[int, int, set[tuple[int, int]]]:
    h, w = shape
    assert h >= 2
    candidate_pairs = [
        ((0, 0), (1, 0)),
        ((0, w - 1), (1, w - 1)),
        ((h - 2, 0), (h - 1, 0)),
        ((h - 2, w - 1), (h - 1, w - 1)),
    ]
    singletons = {
        tuple(comp["coords"][0]): int(comp["color"])
        for comp in components
        if len(comp["coords"]) == 1
    }
    matches = [pair for pair in candidate_pairs if pair[0] in singletons and pair[1] in singletons]
    assert len(matches) == 1, "expected exactly one corner legend pair"
    top, bottom = sorted(matches[0])
    return int(singletons[top]), int(singletons[bottom]), {top, bottom}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the dominant symmetry type among complete sample shapes, then use that
    type to complete near-symmetric damaged shapes with the legend marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    components = _find_components(g, bg)
    sample_color, fill_color, legend_cells = _find_legend(components, g.shape)
    assert sample_color != fill_color != bg

    complete_by_type: dict[str, list[dict[str, object]]] = {kind: [] for kind in SYMMETRY_TYPES}
    broken: list[dict[str, object]] = []

    for comp in components:
        coords = comp["coords"]
        if len(coords) == 1 and coords[0] in legend_cells:
            continue

        assert len(coords) > 1, "non-legend objects must have area > 1"
        mask, box = _mask_from_coords(coords)
        kind = _symmetry_type(mask)
        item = {"color": int(comp["color"]), "bbox": box, "mask": mask}
        if kind is None:
            broken.append(item)
        else:
            complete_by_type[kind].append(item)

    counts = {kind: len(items) for kind, items in complete_by_type.items() if items}
    assert counts, "expected at least one complete sample shape"
    max_count = max(counts.values())
    major_types = [kind for kind, count in counts.items() if count == max_count]
    assert len(major_types) == 1, "dominant symmetry type must be unique"
    major_type = major_types[0]

    assert any(int(item["color"]) == sample_color for item in complete_by_type[major_type])
    assert broken, "expected at least one damaged shape"

    out = g.copy()
    for item in broken:
        mask = np.asarray(item["mask"], dtype=bool)
        completed = _complete_mask(mask, major_type)
        assert _symmetry_type(completed) == major_type
        missing = completed & ~mask
        missing_count = int(missing.sum())
        assert 1 <= missing_count <= 3, "damaged shapes should miss only a few cells"

        r0, c0, _r1, _c1 = item["bbox"]
        for rr, cc in np.argwhere(missing):
            ar = int(r0 + rr)
            ac = int(c0 + cc)
            assert int(out[ar, ac]) == bg, "completion cannot overwrite another object"
            out[ar, ac] = fill_color

    return out


def _random_full_mask(kind: str) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        min_area = max(6, (h * w) // 4)
        max_area = min(h * w - 1, 18 if kind != "quad" else 22)
        if min_area > max_area:
            continue

        orbits = _all_orbits(kind, h, w)
        connected_orbits = [orb for orb in orbits if _cells_connected(set(orb))]
        seed = random.choice(connected_orbits or orbits)
        selected = {seed}
        cells = set(seed)
        target = random.randint(min_area, max_area)

        while True:
            frontier = [orb for orb in orbits if orb not in selected and _orbit_touches(cells, orb)]
            if not frontier:
                break
            if len(cells) >= target and random.random() < 0.65:
                break
            room = [orb for orb in frontier if len(cells) + len(orb) <= max_area]
            if not room:
                break
            orb = random.choice(room)
            selected.add(orb)
            cells.update(orb)
            if len(cells) >= min_area and random.random() < 0.15:
                target = min(target, len(cells))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        if min(mask.shape) < 3:
            continue
        if np.all(mask):
            continue
        if not _is_connected(mask):
            continue
        if _symmetry_type(mask) != kind:
            continue
        return mask
    return None


def _random_broken_mask(kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(BROKEN_ATTEMPTS):
        full = _random_full_mask(kind)
        if full is None:
            continue

        h, w = full.shape
        coords = [tuple(map(int, rc)) for rc in np.argwhere(full)]
        deletable = [cell for cell in coords if len(_orbit(kind, h, w, *cell)) > 1]
        if not deletable:
            continue

        for _ in range(BROKEN_ATTEMPTS):
            remove_count = random.randint(1, min(3, len(deletable)))
            removed = random.sample(deletable, remove_count)
            broken = full.copy()
            for r, c in removed:
                broken[r, c] = False

            if not (broken[0].any() and broken[-1].any() and broken[:, 0].any() and broken[:, -1].any()):
                continue
            if not _is_connected(broken):
                continue
            if _symmetry_type(broken) is not None:
                continue
            if not np.array_equal(_complete_mask(broken, kind), full):
                continue
            return broken, full
    return None


def _corner_cells(shape: tuple[int, int], corner: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    if corner == "tl":
        return (0, 0), (1, 0)
    if corner == "tr":
        return (0, w - 1), (1, w - 1)
    if corner == "bl":
        return (h - 2, 0), (h - 1, 0)
    return (h - 2, w - 1), (h - 1, w - 1)


def _reserve_corner(blocked: np.ndarray, corner: str) -> None:
    h, w = blocked.shape
    if corner == "tl":
        blocked[:3, :3] = True
    elif corner == "tr":
        blocked[:3, max(0, w - 3) : w] = True
    elif corner == "bl":
        blocked[max(0, h - 3) : h, :3] = True
    else:
        blocked[max(0, h - 3) : h, max(0, w - 3) : w] = True


def _placement_candidates(height: int, width: int, mask: np.ndarray) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    if height - mh < 2 or width - mw < 2:
        return []
    return [
        (r0, c0)
        for r0 in range(1, height - mh)
        for c0 in range(1, width - mw)
    ]


def _find_placement(blocked: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    candidates = _placement_candidates(*blocked.shape, mask)
    random.shuffle(candidates)
    mh, mw = mask.shape
    for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(blocked.shape[0], r0 + mh + 1)
        cc1 = min(blocked.shape[1], c0 + mw + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        return r0, c0
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    window = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    window[mask] = color


def _build_specs(major_kind: str) -> tuple[list[dict[str, object]], list[int]] | None:
    major_complete = random.randint(2, 3)
    broken_count = random.randint(1, 2)
    other_kinds = [kind for kind in SYMMETRY_TYPES if kind != major_kind]

    if major_complete == 2:
        minor_kinds = [random.choice(other_kinds)]
    else:
        if random.random() < 0.5:
            minor_kinds = other_kinds.copy()
        else:
            pick = random.choice(other_kinds)
            minor_kinds = [pick, pick]

    specs: list[dict[str, object]] = []
    major_complete_indices: list[int] = []
    used_signatures: set[tuple[str, bytes]] = set()

    for kind in [major_kind] * major_complete + minor_kinds:
        mask = _random_full_mask(kind)
        if mask is None:
            return None
        sig = (kind, mask.tobytes())
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        specs.append({"kind": kind, "broken": False, "mask": mask, "full": mask})
        if kind == major_kind:
            major_complete_indices.append(len(specs) - 1)

    for _ in range(broken_count):
        pair = _random_broken_mask(major_kind)
        if pair is None:
            return None
        broken, full = pair
        sig = ("broken-" + major_kind, broken.tobytes())
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        specs.append({"kind": major_kind, "broken": True, "mask": broken, "full": full})

    return specs, major_complete_indices


def _try_generate_case() -> np.ndarray | None:
    major_kind = random.choice(SYMMETRY_TYPES)
    built = _build_specs(major_kind)
    if built is None:
        return None
    specs, major_complete_indices = built

    color_count = len(specs) + 2
    if color_count > len(PALETTE):
        return None
    colors = random.sample(PALETTE, color_count)
    bg = colors[0]
    fill_color = colors[1]
    object_colors = colors[2:]
    random.shuffle(object_colors)

    for spec, color in zip(specs, object_colors):
        spec["color"] = int(color)

    legend_top = int(specs[random.choice(major_complete_indices)]["color"])

    masks = [np.asarray(spec["mask"], dtype=bool) for spec in specs]
    packed_area = 9 + sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in masks)
    min_side = max(GRID_SIDE_RANGE[0], int(math.ceil(math.sqrt(packed_area * 1.25))))
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    height = random.randint(min_side, GRID_SIDE_RANGE[1])
    width = random.randint(min_side, GRID_SIDE_RANGE[1])

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    blocked = np.zeros((height, width), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    _reserve_corner(blocked, corner)
    top_cell, bottom_cell = _corner_cells(inp.shape, corner)
    inp[top_cell] = expected[top_cell] = legend_top
    inp[bottom_cell] = expected[bottom_cell] = fill_color

    placement_order = sorted(
        range(len(specs)),
        key=lambda idx: int(np.asarray(specs[idx]["mask"], dtype=bool).sum()),
        reverse=True,
    )

    for idx in placement_order:
        spec = specs[idx]
        mask = np.asarray(spec["mask"], dtype=bool)
        pos = _find_placement(blocked, mask)
        if pos is None:
            return None

        r0, c0 = pos
        spec["pos"] = pos
        _paint_mask(inp, mask, r0, c0, int(spec["color"]))
        _paint_mask(expected, mask, r0, c0, int(spec["color"]))

        if bool(spec["broken"]):
            full = np.asarray(spec["full"], dtype=bool)
            missing = full & ~mask
            _paint_mask(expected, missing, r0, c0, fill_color)

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height, r0 + mask.shape[0] + 1)
        cc1 = min(width, c0 + mask.shape[1] + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    if int(np.count_nonzero(inp != bg)) >= inp.size // 2:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _try_generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-completion task")
