from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

MAX_GENERATE_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.asarray([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = _crop_mask(mask)
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        work = np.fliplr(mask) if flip else mask
        for turns in range(4):
            transformed = _crop_mask(np.rot90(work, k=turns))
            sig = _mask_signature(transformed)
            if sig not in seen:
                seen.add(sig)
                variants.append(transformed)
    return variants


def _exposed_edge_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    total = 0
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds((h, w), nr, nc) or not mask[nr, nc]:
                total += 1
    return int(total)


def _component_exposed_edges(coords: list[tuple[int, int]]) -> int:
    cells = set(coords)
    total = 0
    for r, c in coords:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in cells:
                total += 1
    return total


def _shape_entry(name: str, kind: str, mask: np.ndarray) -> dict[str, object] | None:
    normalized = _crop_mask(mask)
    area = int(normalized.sum())
    edges = _exposed_edge_count(normalized)
    if not 4 <= area <= 12 or edges == 2 * area:
        return None
    return {
        "name": name,
        "kind": kind,
        "mask": normalized,
        "area": area,
        "edges": edges,
        "cls": "slender" if edges > 2 * area else "compact",
    }


RAW_SHAPES = [
    ("line4", "line", ["1111"]),
    ("line5", "line", ["11111"]),
    ("line6", "line", ["111111"]),
    ("line7", "line", ["1111111"]),
    ("line8", "line", ["11111111"]),
    ("line9", "line", ["111111111"]),
    ("ell4", "bend", ["100", "111"]),
    ("ell5", "bend", ["100", "100", "111"]),
    ("ell6", "bend", ["1000", "1111", "0001"]),
    ("zig4", "zigzag", ["110", "011"]),
    ("zig5", "zigzag", ["1100", "0111"]),
    ("tee5", "tee", ["111", "010", "010"]),
    ("rect23", "rect", ["111", "111"]),
    ("rect24", "rect", ["1111", "1111"]),
    ("rect25", "rect", ["11111", "11111"]),
    ("rect26", "rect", ["111111", "111111"]),
    ("rect33", "rect", ["111", "111", "111"]),
    ("rect34", "rect", ["1111", "1111", "1111"]),
    ("notch7a", "notch", ["011", "111", "110"]),
    ("notch7b", "notch", ["111", "111", "010"]),
    ("notch8a", "notch", ["011", "111", "111"]),
    ("notch8b", "notch", ["111", "111", "011"]),
    ("notch10", "notch", ["1111", "1111", "0011"]),
    ("notch11", "notch", ["0111", "1111", "1111"]),
]


def _build_shape_banks():
    slender: list[dict[str, object]] = []
    compact: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for base_name, kind, rows in RAW_SHAPES:
        for idx, transformed in enumerate(_unique_transforms(_parse_mask(rows))):
            sig = _mask_signature(transformed)
            if sig in seen:
                continue
            seen.add(sig)
            entry = _shape_entry(f"{base_name}_{idx}", kind, transformed)
            if entry is None:
                continue
            if entry["cls"] == "slender":
                slender.append(entry)
            else:
                compact.append(entry)

    slender_by_area: dict[int, list[dict[str, object]]] = {}
    compact_by_area: dict[int, list[dict[str, object]]] = {}
    slender_by_kind: dict[str, list[dict[str, object]]] = {}
    compact_by_kind: dict[str, list[dict[str, object]]] = {}

    for entry in slender:
        slender_by_area.setdefault(int(entry["area"]), []).append(entry)
        slender_by_kind.setdefault(str(entry["kind"]), []).append(entry)
    for entry in compact:
        compact_by_area.setdefault(int(entry["area"]), []).append(entry)
        compact_by_kind.setdefault(str(entry["kind"]), []).append(entry)

    line_rect_common_areas = sorted(
        set(int(entry["area"]) for entry in slender_by_kind["line"])
        & set(int(entry["area"]) for entry in compact_by_kind["rect"])
    )
    assert slender and compact and line_rect_common_areas
    assert any(entry["kind"] != "line" for entry in slender)
    assert any(entry["kind"] != "rect" for entry in compact)
    return slender, compact, slender_by_area, compact_by_area, slender_by_kind, compact_by_kind, line_rect_common_areas


(
    SLENDER_BANK,
    COMPACT_BANK,
    SLENDER_BY_AREA,
    COMPACT_BY_AREA,
    SLENDER_BY_KIND,
    COMPACT_BY_KIND,
    LINE_RECT_COMMON_AREAS,
) = _build_shape_banks()


def _pick_entry(
    bank: list[dict[str, object]],
    used: dict[str, int],
    *,
    area: int | None = None,
    kind: str | None = None,
    forbid_kind: str | None = None,
) -> dict[str, object]:
    candidates = []
    for entry in bank:
        if area is not None and int(entry["area"]) != int(area):
            continue
        if kind is not None and str(entry["kind"]) != kind:
            continue
        if forbid_kind is not None and str(entry["kind"]) == forbid_kind:
            continue
        if used.get(str(entry["name"]), 0) >= 2:
            continue
        candidates.append(entry)

    if not candidates:
        for entry in bank:
            if area is not None and int(entry["area"]) != int(area):
                continue
            if kind is not None and str(entry["kind"]) != kind:
                continue
            if forbid_kind is not None and str(entry["kind"]) == forbid_kind:
                continue
            candidates.append(entry)

    assert candidates
    entry = random.choice(candidates)
    used[str(entry["name"])] = used.get(str(entry["name"]), 0) + 1
    return entry


def _sample_object_specs() -> list[dict[str, object]]:
    count = random.randint(4, 6)
    if count == 4:
        red_count, blue_count = 2, 2
    elif count == 5:
        red_count, blue_count = random.choice([(2, 3), (3, 2)])
    else:
        red_count, blue_count = random.choice([(2, 4), (3, 3), (4, 2)])

    used: dict[str, int] = {}
    specs: list[dict[str, object]] = []

    common_area = random.choice(LINE_RECT_COMMON_AREAS)
    specs.append(
        {
            "cls": "slender",
            "entry": _pick_entry(SLENDER_BANK, used, area=common_area, kind="line"),
        }
    )
    specs.append(
        {
            "cls": "compact",
            "entry": _pick_entry(COMPACT_BANK, used, area=common_area, kind="rect"),
        }
    )
    red_count -= 1
    blue_count -= 1

    specs.append(
        {
            "cls": "slender",
            "entry": _pick_entry(SLENDER_BANK, used, forbid_kind="line"),
        }
    )
    specs.append(
        {
            "cls": "compact",
            "entry": _pick_entry(COMPACT_BANK, used, forbid_kind="rect"),
        }
    )
    red_count -= 1
    blue_count -= 1

    while red_count > 0:
        specs.append({"cls": "slender", "entry": _pick_entry(SLENDER_BANK, used)})
        red_count -= 1
    while blue_count > 0:
        specs.append({"cls": "compact", "entry": _pick_entry(COMPACT_BANK, used)})
        blue_count -= 1

    random.shuffle(specs)
    return specs


def _place_specs(
    shape: tuple[int, int], specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    h, w = map(int, shape)
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    order = sorted(
        specs,
        key=lambda spec: (
            -int(spec["entry"]["area"]),
            -max(spec["entry"]["mask"].shape),
            -min(spec["entry"]["mask"].shape),
        ),
    )

    for spec in order:
        mask = np.asarray(spec["entry"]["mask"], dtype=bool)
        mh, mw = map(int, mask.shape)
        if mh >= h - 1 or mw >= w - 1:
            return None

        positions = [(r0, c0) for r0 in range(1, h - mh) for c0 in range(1, w - mw)]
        random.shuffle(positions)

        chosen_coords: list[tuple[int, int]] | None = None
        chosen_origin: tuple[int, int] | None = None
        local_coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]

        for r0, c0 in positions:
            coords = [(r0 + r, c0 + c) for r, c in local_coords]
            if any((rr, cc) in blocked for rr, cc in coords):
                continue
            chosen_coords = coords
            chosen_origin = (r0, c0)
            break

        if chosen_coords is None or chosen_origin is None:
            return None

        for rr, cc in chosen_coords:
            for dr, dc in DIR8:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(shape, nr, nc):
                    blocked.add((nr, nc))
            blocked.add((rr, cc))

        placed.append(
            {
                "cls": spec["cls"],
                "entry": spec["entry"],
                "origin": chosen_origin,
                "coords": chosen_coords,
            }
        )

    return placed


def _render_input(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for item in placed:
        for r, c in item["coords"]:
            grid[int(r), int(c)] = GRAY
    return grid


def _same_area_opposite_classes(classes: list[str], areas: list[int]) -> bool:
    seen: dict[int, set[str]] = {}
    for cls, area in zip(classes, areas):
        seen.setdefault(int(area), set()).add(str(cls))
    return any({"compact", "slender"}.issubset(kinds) for kinds in seen.values())


def _scene_is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.any(grid == GRAY):
        return False

    components = _components_of_color(grid, GRAY)
    if not 4 <= len(components) <= 6:
        return False

    classes: list[str] = []
    areas: list[int] = []
    owners = np.full(grid.shape, -1, dtype=int)

    for idx, coords in enumerate(components):
        area = len(coords)
        if not 4 <= area <= 12:
            return False
        edges = _component_exposed_edges(coords)
        if edges == 2 * area:
            return False
        classes.append("slender" if edges > 2 * area else "compact")
        areas.append(area)
        for r, c in coords:
            owners[r, c] = idx

    if classes.count("slender") < 2 or classes.count("compact") < 2:
        return False
    if not _same_area_opposite_classes(classes, areas):
        return False

    for r, c in np.argwhere(owners >= 0):
        rr, cc = int(r), int(c)
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(grid.shape, nr, nc) and owners[nr, nc] >= 0 and owners[nr, nc] != owners[rr, cc]:
                return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    colors = set(map(int, out[grid == GRAY]))
    if not colors.issubset({BLUE, RED}) or len(colors) != 2:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray object by comparing its exposed-side rate to its area.

    Input structure:
    - The grid is a black background containing several separated gray 4-connected objects.
    - Each gray object is a small polyomino with area between 4 and 12 cells.
    - Objects are spaced apart so their neighborhoods do not touch.

    Rule:
    - Find every gray 4-connected component.
    - For each component, compute A as its area and E as the number of 4-neighbor sides that are exposed to background.
    - Recolor the component red when E > 2A and blue when E < 2A, leaving the black background unchanged.

    Key insight:
    - Area alone is insufficient: a thin path and a compact block can have the same size, so the discriminator is exposed boundary per pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components_of_color(g, GRAY):
        area = len(coords)
        edges = _component_exposed_edges(coords)
        assert edges != 2 * area, "tie cases are outside the puzzle distribution"
        color = RED if edges > 2 * area else BLUE
        for r, c in coords:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        specs = _sample_object_specs()
        count = len(specs)
        side_base = {4: 14, 5: 16, 6: 18}[count]
        shape = (
            random.randint(side_base, side_base + 2),
            random.randint(side_base, side_base + 2),
        )
        placed = _place_specs(shape, specs)
        if placed is None:
            continue
        grid = _render_input(shape, placed)
        if _scene_is_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
