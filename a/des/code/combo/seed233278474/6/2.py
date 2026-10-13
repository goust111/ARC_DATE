from __future__ import annotations

from collections import deque

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_RNG = np.random.default_rng()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    h, w = binary.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not binary[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coord_array = np.asarray(coords, dtype=int)
            bbox, mask = _coords_to_mask(coord_array)
            r0, c0, r1, c1 = bbox
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": bbox,
                    "mask": mask,
                    "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )

    return components


def _find_enclosed_holes(component: dict[str, object], bg: int) -> list[dict[str, object]]:
    patch = np.asarray(component["patch"], dtype=int)
    r0, c0, _r1, _c1 = component["bbox"]
    holes: list[dict[str, object]] = []

    for coords in _component_masks(patch == int(bg)):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if np.any(rows == 0) or np.any(cols == 0) or np.any(rows == patch.shape[0] - 1) or np.any(cols == patch.shape[1] - 1):
            continue
        hole_bbox, hole_mask = _coords_to_mask(coords)
        hr0, hc0, hr1, hc1 = hole_bbox
        holes.append(
            {
                "bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                "mask": hole_mask,
                "key": _shape_key(hole_mask),
            }
        )

    return holes


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    variants: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    base = np.asarray(mask, dtype=bool)
    for k in range(4):
        rotated = np.rot90(base, k)
        key = _shape_key(rotated)
        if key in seen:
            continue
        seen.add(key)
        variants.append((k, rotated))
    return variants


def _match_fragments_to_holes(
    fragments: list[dict[str, object]], holes: list[dict[str, object]]
) -> dict[int, tuple[int, int]]:
    options: list[dict[int, int]] = []
    hole_keys = [_shape_key(np.asarray(hole["mask"], dtype=bool)) for hole in holes]

    for fragment in fragments:
        candidates: dict[int, int] = {}
        for rot_k, rotated_mask in _rotation_variants(np.asarray(fragment["mask"], dtype=bool)):
            rotated_key = _shape_key(rotated_mask)
            for hole_index, hole_key in enumerate(hole_keys):
                if rotated_key == hole_key and hole_index not in candidates:
                    candidates[hole_index] = int(rot_k)
        assert candidates, "every fragment must match at least one hole"
        options.append(candidates)

    order = sorted(range(len(fragments)), key=lambda idx: len(options[idx]))
    solutions: list[dict[int, tuple[int, int]]] = []
    used_holes: set[int] = set()
    current: dict[int, tuple[int, int]] = {}

    def backtrack(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(current.copy())
            return
        frag_index = order[pos]
        for hole_index in sorted(options[frag_index]):
            if hole_index in used_holes:
                continue
            used_holes.add(hole_index)
            current[frag_index] = (hole_index, options[frag_index][hole_index])
            backtrack(pos + 1)
            used_holes.remove(hole_index)
            del current[frag_index]

    backtrack(0)
    assert len(solutions) == 1, "fragment-hole assignment must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    components = _find_monochrome_components(g, bg)
    assert components, "expected templates and fragments"

    templates: list[dict[str, object]] = []
    fragments: list[dict[str, object]] = []

    for component in components:
        holes = _find_enclosed_holes(component, bg)
        if len(holes) == 1:
            templates.append({"component": component, "hole": holes[0]})
        else:
            assert len(holes) == 0, "components should either be solid fragments or single-hole templates"
            fragments.append(component)

    assert templates and fragments and len(templates) == len(fragments)
    hole_infos = [template["hole"] for template in templates]
    assignment = _match_fragments_to_holes(fragments, hole_infos)

    out = g.copy()
    for fragment in fragments:
        coords = np.asarray(fragment["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for frag_index, (hole_index, _rot_k) in assignment.items():
        hole = hole_infos[hole_index]
        hr0, hc0, hr1, hc1 = hole["bbox"]
        target = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        target[np.asarray(hole["mask"], dtype=bool)] = int(fragments[frag_index]["color"])

    return out


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    row_min = min(r for r, _c in cells)
    col_min = min(c for _r, c in cells)
    return tuple(sorted((int(r - row_min), int(c - col_min)) for r, c in cells))


def _rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_cells({(int(c), int(-r)) for r, c in cells})


def _canonical_rotation_cells(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    current = _normalize_cells(cells)
    variants: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        variants.append(current)
        current = _rotate_cells(current)
    return min(variants)


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _build_shape_library(min_area: int = 3, max_area: int = 6) -> tuple[np.ndarray, ...]:
    shapes_by_area: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}

    for area in range(2, max_area + 1):
        current: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes_by_area[area - 1]:
            cell_set = set(shape)
            for r, c in shape:
                for dr, dc in _NEIGHBORS4:
                    new_cell = (int(r + dr), int(c + dc))
                    if new_cell in cell_set:
                        continue
                    current.add(_canonical_rotation_cells(cell_set | {new_cell}))
        shapes_by_area[area] = current

    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for area in range(min_area, max_area + 1):
        for shape in sorted(shapes_by_area[area]):
            mask = _cells_to_mask(shape)
            if len(_rotation_variants(mask)) != 4:
                continue
            key = _shape_key(mask)
            if key in seen:
                continue
            seen.add(key)
            library.append(mask)

    assert library, "shape library should not be empty"
    return tuple(library)


_SHAPE_LIBRARY = _build_shape_library()
_BACKGROUND_CHOICES = (0, 1, 8)
_TEMPLATE_COLOR = 5
_COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 8, 9)
_OBJECT_COUNT_RANGE = (2, 4)
_PADDING = 1
_GENERATION_ATTEMPTS = 400


def _random_rotation(mask: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), int(_RNG.integers(0, 4)))


def _make_template_patch(hole_mask: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    hh, hw = hole_mask.shape
    top = int(_RNG.integers(1, 4))
    bottom = int(_RNG.integers(1, 4))
    left = int(_RNG.integers(1, 4))
    right = int(_RNG.integers(1, 4))
    patch = np.full((top + hh + bottom, left + hw + right), _TEMPLATE_COLOR, dtype=int)
    patch[top : top + hh, left : left + hw][hole_mask] = int(bg)
    return patch, (top, left), hole_mask.copy()


def _make_fragment_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    window = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    grid_h, grid_w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
        return False
    r1 = max(0, r0 - padding)
    c1 = max(0, c0 - padding)
    r2 = min(grid_h, r0 + h + padding)
    c2 = min(grid_w, c0 + w + padding)
    return not bool(occupied[r1:r2, c1:c2].any())


def _sample_canvas_shape(patches: list[np.ndarray]) -> tuple[int, int] | None:
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(total_area * 2.3))))
    low = max(18, base)
    high = 31
    if low >= high:
        return None
    return int(_RNG.integers(low, high)), int(_RNG.integers(low, high))


def _place_all_patches(patches: list[np.ndarray], bg: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    shape = _sample_canvas_shape(patches)
    if shape is None:
        return None
    grid_h, grid_w = shape
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    order = list(range(len(patches)))
    _RNG.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    for index in order:
        patch = patches[index]
        h, w = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                if _can_place(occupied, r0, c0, h, w, _PADDING):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        positions[index] = (int(r0), int(c0))
        _paste(grid, patch, int(r0), int(c0), bg)
        occupied[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = True

    return grid, [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_expected(
    grid: np.ndarray,
    template_records: list[dict[str, object]],
    fragment_records: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for fragment in fragment_records:
        r0, c0 = fragment["pos"]
        mask = np.asarray(fragment["mask"], dtype=bool)
        h, w = mask.shape
        target = out[r0 : r0 + h, c0 : c0 + w]
        target[mask] = int(bg)

    for template, fragment in zip(template_records, fragment_records):
        tr0, tc0 = template["pos"]
        hole_r, hole_c = template["hole_offset"]
        hole_mask = np.asarray(template["hole_mask"], dtype=bool)
        hh, hw = hole_mask.shape
        target = out[tr0 + hole_r : tr0 + hole_r + hh, tc0 + hole_c : tc0 + hole_c + hw]
        target[hole_mask] = int(fragment["color"])

    return out


def _generation_attempt() -> np.ndarray | None:
    bg = int(_RNG.choice(_BACKGROUND_CHOICES))
    usable_colors = [color for color in _COLOR_POOL if int(color) not in {bg, _TEMPLATE_COLOR}]
    obj_count = int(_RNG.integers(_OBJECT_COUNT_RANGE[0], _OBJECT_COUNT_RANGE[1] + 1))
    color_choices = _RNG.choice(np.asarray(usable_colors, dtype=int), size=obj_count, replace=False)
    shape_indices = _RNG.choice(len(_SHAPE_LIBRARY), size=obj_count, replace=False)

    template_records: list[dict[str, object]] = []
    fragment_records: list[dict[str, object]] = []
    patches: list[np.ndarray] = []

    for i in range(obj_count):
        base_shape = _SHAPE_LIBRARY[int(shape_indices[i])]
        hole_mask = _random_rotation(base_shape)
        fragment_mask = _random_rotation(base_shape)
        color = int(color_choices[i])

        template_patch, hole_offset, hole_mask_copy = _make_template_patch(hole_mask, bg)
        fragment_patch = _make_fragment_patch(fragment_mask, color, bg)

        template_records.append(
            {
                "patch": template_patch,
                "hole_offset": hole_offset,
                "hole_mask": hole_mask_copy,
            }
        )
        fragment_records.append(
            {
                "patch": fragment_patch,
                "mask": fragment_mask.copy(),
                "color": color,
            }
        )

    patches.extend(template["patch"] for template in template_records)
    patches.extend(fragment["patch"] for fragment in fragment_records)

    placement = _place_all_patches(patches, bg)
    if placement is None:
        return None
    grid, positions = placement

    if len(positions) != len(patches):
        return None

    for idx, template in enumerate(template_records):
        template["pos"] = positions[idx]
    for idx, fragment in enumerate(fragment_records):
        fragment["pos"] = positions[obj_count + idx]

    values, counts = np.unique(grid, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(values, counts)}
    if color_counts.get(bg, 0) != max(color_counts.values()):
        return None
    if list(color_counts.values()).count(color_counts.get(bg, 0)) != 1:
        return None

    expected = _build_expected(grid, template_records, fragment_records, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
