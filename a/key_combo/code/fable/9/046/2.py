from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


KINDS = ("H", "V", "D", "A")
COLOR_POOL = tuple(range(1, 10))
GRID_ROWS = (18, 28)
GRID_COLS = (18, 28)
EXTRA_OBJECTS = (0, 1, 2)
NOISE_COUNT = (2, 6)
PLACEMENT_PADDING = 1
GEN_RETRIES = 400
MASK_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int = 8):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _reflect_anti(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2).T


def _symmetry_flags(mask: np.ndarray) -> dict[str, bool]:
    flags = {
        "H": bool(np.array_equal(mask, np.flipud(mask))),
        "V": bool(np.array_equal(mask, np.fliplr(mask))),
        "D": False,
        "A": False,
    }
    if mask.shape[0] == mask.shape[1]:
        flags["D"] = bool(np.array_equal(mask, mask.T))
        flags["A"] = bool(np.array_equal(mask, _reflect_anti(mask)))
    return flags


def _classify_single_axis(mask: np.ndarray) -> str | None:
    active = [kind for kind, ok in _symmetry_flags(mask).items() if ok]
    if len(active) != 1:
        return None
    return active[0]


def _missing_axis_reflection(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "H":
        return np.fliplr(mask)
    if kind == "V":
        return np.flipud(mask)
    if kind == "D":
        return _reflect_anti(mask)
    if kind == "A":
        return mask.T
    raise ValueError(f"unknown symmetry kind: {kind}")


def _completed_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    return mask | _missing_axis_reflection(mask, kind)


def _is_valid_completed(mask: np.ndarray, kind: str) -> bool:
    out = _completed_mask(mask, kind)
    flags = _symmetry_flags(out)
    if kind in {"H", "V"}:
        return flags["H"] and flags["V"] and not flags["D"] and not flags["A"]
    return flags["D"] and flags["A"] and not flags["H"] and not flags["V"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each object by mirroring it across the orthogonal axis that its existing symmetry implies.

    Input structure:
    - A black-background grid contains several separated monochrome foreground objects.
    - Every non-singleton object has exactly one reflection symmetry axis within its own
      bounding box: horizontal, vertical, main-diagonal, or anti-diagonal.
    - Isolated single pixels are noise.

    Rule:
    - Remove every singleton noise pixel.
    - For a horizontally symmetric object, union it with its mirror across the bbox's
      vertical midline; for a vertically symmetric object, mirror across the horizontal midline.
    - For a main-diagonal symmetric object, union it with its anti-diagonal mirror; for an
      anti-diagonal symmetric object, union it with its main-diagonal mirror.
    - Keep each object's own color for the completed pixels and leave unrelated objects untouched.

    Key insight:
    - The visible symmetry axis is a selector, not the mirror operation itself: you must infer
      which orthogonal partner axis is missing and add only that counterpart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    bg = _get_bg(g)
    out = g.copy()
    components = _find_components(g, bg=bg, connectivity=8)
    assert components, "expected at least one foreground component"

    for comp in components:
        coords = comp["coords"]
        if comp["area"] == 1:
            out[int(coords[0, 0]), int(coords[0, 1])] = bg
            continue

        color = int(comp["color"])
        mask = _component_mask(comp)
        kind = _classify_single_axis(mask)
        assert kind is not None, "each non-singleton object must have exactly one symmetry axis"

        completed = _completed_mask(mask, kind)
        r0, c0, r1, c1 = comp["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        local = np.zeros_like(mask, dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True

        sub[local] = bg
        fillable = completed & ((sub == bg) | local)
        sub[fillable] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = sub

    return out


def _orbit(kind: str, shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    if kind == "H":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "V":
        cells = {(r, c), (r, w - 1 - c)}
    elif kind == "D":
        cells = {(r, c), (c, r)}
    elif kind == "A":
        cells = {(r, c), (h - 1 - c, w - 1 - r)}
    else:
        raise ValueError(f"unknown kind: {kind}")
    return tuple(sorted((int(rr), int(cc)) for rr, cc in cells))


def _touches(mask: np.ndarray, cell: tuple[int, int], connectivity: int = 8) -> bool:
    r, c = cell
    h, w = mask.shape
    for nr, nc in _neighbors(r, c, connectivity):
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            return True
    return False


def _frontier_orbits(mask: np.ndarray, kind: str, max_new: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    options: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            orbit = _orbit(kind, mask.shape, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            new_cells = [cell for cell in orbit if not mask[cell]]
            if not new_cells or len(new_cells) > max_new:
                continue
            if any(_touches(mask, cell, connectivity=8) for cell in new_cells):
                options.append(orbit)
    return options


def _add_orbit(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> None:
    for r, c in orbit:
        mask[r, c] = True


def _base_mask(kind: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if kind == "H":
        axis_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
        arm_col = random.choice([c for c in range(w) if c != w - 1 - c])
        mask[axis_rows, :] = True
        mask[:, arm_col] = True
        return mask
    if kind == "V":
        axis_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
        arm_row = random.choice([r for r in range(h) if r != h - 1 - r])
        mask[:, axis_cols] = True
        mask[arm_row, :] = True
        return mask
    if kind == "D":
        np.fill_diagonal(mask, True)
        return mask
    if kind == "A":
        for i in range(h):
            mask[i, w - 1 - i] = True
        return mask
    raise ValueError(f"unknown kind: {kind}")


def _shape_options(kind: str) -> list[tuple[int, int]]:
    if kind in {"H", "V"}:
        options = []
        for h in range(3, 9):
            for w in range(3, 10):
                options.append((h, w))
        return options
    return [(n, n) for n in range(4, 9)]


def _area_target(kind: str, h: int, w: int, base_area: int) -> tuple[int, int]:
    density_hi = 0.68 if kind in {"H", "V"} else 0.40
    lower = max(base_area, 6 if kind in {"H", "V"} else 4)
    upper = min(h * w - 1, int(np.floor(h * w * density_hi)))
    return lower, max(lower, upper)


def _generate_mask(kind: str) -> np.ndarray | None:
    options = _shape_options(kind)
    for _ in range(MASK_RETRIES):
        h, w = random.choice(options)
        mask = _base_mask(kind, h, w)
        lower, upper = _area_target(kind, h, w, int(mask.sum()))
        target = random.randint(lower, upper)
        while int(mask.sum()) < target:
            choices = _frontier_orbits(mask, kind, target - int(mask.sum()))
            if not choices:
                break
            _add_orbit(mask, random.choice(choices))

        kind_check = _classify_single_axis(mask)
        if kind_check != kind:
            continue
        if np.array_equal(mask, _completed_mask(mask, kind)):
            continue
        if not _is_valid_completed(mask, kind):
            continue
        if int(_completed_mask(mask, kind).sum()) >= h * w:
            continue
        return mask
    return None


def _expand_bounds(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int], pad: int) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _place_objects(shape: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["mask"].size), reverse=True)
    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        oh, ow = mask.shape
        positions = [(r, c) for r in range(h - oh + 1) for c in range(w - ow + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            rr0, cc0, rr1, cc1 = _expand_bounds(r0, c0, r1, c1, shape, PLACEMENT_PADDING)
            if bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            inp[r0 : r0 + oh, c0 : c0 + ow][mask] = int(obj["color"])
            expected[r0 : r0 + oh, c0 : c0 + ow][obj["completed"]] = int(obj["color"])
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            obj["placement"] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    noise_count = random.randint(*NOISE_COUNT)
    palette = [int(obj["color"]) for obj in objects]
    placed_noise = 0
    for r, c in random.sample([(rr, cc) for rr in range(h) for cc in range(w)], h * w):
        if reserved[r, c]:
            continue
        inp[r, c] = int(random.choice(palette))
        rr0, cc0, rr1, cc1 = _expand_bounds(r, c, r, c, shape, PLACEMENT_PADDING)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        placed_noise += 1
        if placed_noise == noise_count:
            break
    if placed_noise != noise_count:
        return None
    return inp, expected


def _build_objects() -> list[dict] | None:
    kinds = list(KINDS)
    for _ in range(random.choice(EXTRA_OBJECTS)):
        kinds.append(random.choice(KINDS))
    random.shuffle(kinds)

    colors = random.sample(COLOR_POOL, len(kinds))
    seen_in: set[tuple[bytes, tuple[int, int], str]] = set()
    seen_out: set[tuple[bytes, tuple[int, int], str]] = set()
    objects: list[dict] = []
    for kind, color in zip(kinds, colors):
        for _ in range(MASK_RETRIES):
            mask = _generate_mask(kind)
            if mask is None:
                return None
            completed = _completed_mask(mask, kind)
            in_sig = (mask.tobytes(), mask.shape, kind)
            out_sig = (completed.tobytes(), completed.shape, kind)
            if in_sig in seen_in or out_sig in seen_out:
                continue
            seen_in.add(in_sig)
            seen_out.add(out_sig)
            objects.append({"kind": kind, "color": int(color), "mask": mask, "completed": completed})
            break
        else:
            return None
    return objects


def _validate_sample(inp: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    comps = _find_components(inp, bg=0, connectivity=8)
    multi = [comp for comp in comps if comp["area"] > 1]
    singletons = [comp for comp in comps if comp["area"] == 1]
    if not singletons or len(multi) < 4:
        return False

    type_counts = Counter()
    used_colors: set[int] = set()
    for comp in multi:
        color = int(comp["color"])
        if color in used_colors:
            return False
        used_colors.add(color)
        mask = _component_mask(comp)
        kind = _classify_single_axis(mask)
        if kind is None:
            return False
        type_counts[kind] += 1
        if not _is_valid_completed(mask, kind):
            return False
        r0, c0, r1, c1 = comp["bbox"]
        out_mask = expected[r0 : r1 + 1, c0 : c1 + 1] == color
        if not np.array_equal(out_mask, _completed_mask(mask, kind)):
            return False

    if any(type_counts[kind] == 0 for kind in KINDS):
        return False
    for comp in singletons:
        r, c = map(int, comp["coords"][0])
        if int(expected[r, c]) != 0:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    objects = _build_objects()
    if objects is None:
        return None
    shape = (
        random.randint(*GRID_ROWS),
        random.randint(*GRID_COLS),
    )
    placed = _place_objects(shape, objects)
    if placed is None:
        return None
    inp, expected = placed
    if not _validate_sample(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
