from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
SPRITE_COLORS = (BLUE, RED, GREEN)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
            next_id += 1
    return labels, components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _normalize_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _all_transforms(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = list(coords)
    variants = []
    for kind in range(8):
        transformed = []
        for r, c in pts:
            if kind == 0:
                rr, cc = r, c
            elif kind == 1:
                rr, cc = r, -c
            elif kind == 2:
                rr, cc = -r, c
            elif kind == 3:
                rr, cc = -r, -c
            elif kind == 4:
                rr, cc = c, r
            elif kind == 5:
                rr, cc = c, -r
            elif kind == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            transformed.append((rr, cc))
        variants.append(_normalize_cells(tuple(transformed)))
    return sorted(set(variants))


def _canonical_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_all_transforms(coords))


def _flip_mask(mask: np.ndarray, boundary_orientation: str) -> np.ndarray:
    if boundary_orientation == "horizontal":
        return np.flipud(mask)
    assert boundary_orientation == "vertical"
    return np.fliplr(mask)


def _flip_patch(patch: np.ndarray, boundary_orientation: str) -> np.ndarray:
    if boundary_orientation == "horizontal":
        return np.flipud(patch)
    assert boundary_orientation == "vertical"
    return np.fliplr(patch)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    target[mask] = patch[mask]


def _extract_object(grid: np.ndarray, region_mask: np.ndarray, colors: set[int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    mask = np.isin(grid, list(colors)) & region_mask
    _, comps = _label_mask(mask)
    assert len(comps) == 1, "expected exactly one object in the region"
    bbox = _bbox_from_coords(comps[0])
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    obj_mask = patch != BLACK
    return patch, obj_mask, bbox


def _shared_boundary_orientation(region_labels: np.ndarray, a: int, b: int) -> str | None:
    h, w = region_labels.shape
    horizontal = 0
    vertical = 0
    for r in range(h):
        for c in range(w):
            if region_labels[r, c] != 0:
                continue
            if 0 < r < h - 1:
                up = int(region_labels[r - 1, c])
                down = int(region_labels[r + 1, c])
                if {up, down} == {a, b}:
                    horizontal += 1
            if 0 < c < w - 1:
                left = int(region_labels[r, c - 1])
                right = int(region_labels[r, c + 1])
                if {left, right} == {a, b}:
                    vertical += 1
    if horizontal and vertical:
        raise AssertionError("shared boundary must be purely horizontal or purely vertical")
    if horizontal >= 2:
        return "horizontal"
    if vertical >= 2:
        return "vertical"
    return None


def _find_match(source_patch: np.ndarray, target_mask: np.ndarray, boundary_orientation: str) -> np.ndarray:
    mirrored = _flip_patch(source_patch, boundary_orientation)
    matches: list[np.ndarray] = []
    for k in range(4):
        candidate = np.rot90(mirrored, k)
        if candidate.shape == target_mask.shape and np.array_equal(candidate != BLACK, target_mask):
            matches.append(candidate)
    assert len(matches) == 1, "mirror+rotation match must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp mirrored sprite colors from each source chamber onto its neighboring placeholder chamber.

    Input structure:
    - Black background and orange one-pixel contour lines partition the scene into enclosed 4-connected regions.
    - Some regions contain a small multicolor sprite using blue/red/green; some regions contain a gray placeholder object.
    - Every placeholder region shares one straight orange boundary segment with exactly one source region, and that shared segment is either purely horizontal or purely vertical.

    Rule:
    - Treat orange as a barrier and split all non-orange cells into 4-connected regions.
    - For each gray placeholder region, find the unique adjacent source region that shares the same orange contour segment.
    - If the shared segment is horizontal, flip the source sprite top-to-bottom; if the shared segment is vertical, flip it left-to-right.
    - Then try the four right-angle rotations and keep the unique transform whose mask matches the gray placeholder mask exactly; stamp the source colors with that transform onto the placeholder cells.
    - Leave the source sprite, contour lines, and all remaining black background unchanged.

    Key insight:
    The contour drawing simultaneously encodes region membership, source-target pairing, and which mirror direction must be used before the final shape-alignment rotation.
    """
    g = np.asarray(grid, dtype=int)
    region_labels, region_coords = _label_mask(g != ORANGE)

    source_ids: list[int] = []
    target_ids: list[int] = []
    for idx, coords in enumerate(region_coords, start=1):
        vals = set(int(g[r, c]) for r, c in coords)
        if GREY in vals:
            target_ids.append(idx)
        elif vals & set(SPRITE_COLORS):
            source_ids.append(idx)

    assert 2 <= len(source_ids) == len(target_ids) <= 3, "expected 2-3 source/target pairs"

    out = g.copy()
    used_sources: set[int] = set()
    for target_id in target_ids:
        candidates: list[tuple[int, str]] = []
        for source_id in source_ids:
            orientation = _shared_boundary_orientation(region_labels, source_id, target_id)
            if orientation is not None:
                candidates.append((source_id, orientation))
        assert len(candidates) == 1, "each target must have one unique adjacent source"
        source_id, orientation = candidates[0]
        assert source_id not in used_sources, "sources must pair one-to-one with targets"
        used_sources.add(source_id)

        source_patch, _source_mask, _source_bbox = _extract_object(g, region_labels == source_id, set(SPRITE_COLORS))
        _placeholder_patch, target_mask, target_bbox = _extract_object(g, region_labels == target_id, {GREY})
        matched = _find_match(source_patch, target_mask, orientation)
        r0, c0, r1, c1 = target_bbox
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[target_mask] = matched[target_mask]

    return out


def _build_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        current: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        current.add(_normalize_cells(cells | {(nr, nc)}))
        by_size[size] = current

    catalog: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for size in range(3, 7):
        for shape in by_size[size]:
            canonical = _canonical_shape(shape)
            if canonical in seen:
                continue
            seen.add(canonical)
            mask = _coords_to_mask(canonical)
            if mask.shape[0] <= 4 and mask.shape[1] <= 4:
                rotations = {_mask_to_coords(np.rot90(mask, k)) for k in range(4)}
                if len(rotations) == 4:
                    catalog.append(canonical)
    return catalog


SHAPE_CATALOG = _build_shape_catalog()


def _choose_shape(
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
    boundary_orientation: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    for _ in range(400):
        base = random.choice(SHAPE_CATALOG)
        source_coords = random.choice(_all_transforms(base))
        source_mask = _coords_to_mask(source_coords)
        if source_mask.shape[0] > source_h or source_mask.shape[1] > source_w:
            continue
        unflipped_rotations = {(_coords_to_mask(_mask_to_coords(np.rot90(source_mask, k))).shape, _mask_to_coords(np.rot90(source_mask, k))) for k in range(4)}
        mirrored = _flip_mask(source_mask, boundary_orientation)
        for k in random.sample([1, 2, 3], 3):
            target_mask = np.rot90(mirrored, k)
            if target_mask.shape[0] > target_h or target_mask.shape[1] > target_w:
                continue
            signature = (target_mask.shape, _mask_to_coords(target_mask))
            if signature in unflipped_rotations:
                continue
            matches = 0
            for kk in range(4):
                cand = np.rot90(mirrored, kk)
                if cand.shape == target_mask.shape and np.array_equal(cand, target_mask):
                    matches += 1
            if matches == 1:
                return source_mask, target_mask, k
    raise RuntimeError("failed to choose a non-degenerate sprite shape")


def _colorize(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(2, min(3, len(coords)))
    palette = random.sample(list(SPRITE_COLORS), color_count)
    colors = palette[:]
    while len(colors) < len(coords):
        colors.append(random.choice(palette))
    random.shuffle(colors)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _place_in_chamber(
    chamber: tuple[int, int, int, int],
    patch: np.ndarray,
) -> tuple[int, int]:
    r0, c0, h, w = chamber
    ph, pw = patch.shape
    spare_r = h - ph
    spare_c = w - pw
    r_min = 1 if spare_r >= 2 else 0
    r_max = spare_r - 1 if spare_r >= 2 else spare_r
    c_min = 1 if spare_c >= 2 else 0
    c_max = spare_c - 1 if spare_c >= 2 else spare_c
    off_r = random.randint(r_min, r_max) if r_max >= r_min else 0
    off_c = random.randint(c_min, c_max) if c_max >= c_min else 0
    return r0 + off_r, c0 + off_c


def _make_module(orientation: str) -> dict:
    if orientation == "vertical":
        chamber_h = random.randint(4, 5)
        left_w = random.randint(3, 4)
        right_w = random.randint(3, 4)
        return {
            "orientation": orientation,
            "outer_h": chamber_h + 2,
            "outer_w": left_w + right_w + 3,
            "chambers": (("left", chamber_h, left_w), ("right", chamber_h, right_w)),
        }
    chamber_w = random.randint(4, 5)
    top_h = random.randint(3, 4)
    bottom_h = random.randint(3, 4)
    return {
        "orientation": orientation,
        "outer_h": top_h + bottom_h + 3,
        "outer_w": chamber_w + 2,
        "chambers": (("top", top_h, chamber_w), ("bottom", bottom_h, chamber_w)),
    }


def _position_modules(orientations: list[str]) -> tuple[list[dict], int, int] | None:
    modules = [_make_module(orientation) for orientation in orientations]
    r = random.randint(0, 1)
    c = random.randint(0, 1)
    gap_choices = [1, 2]
    placed: list[dict] = []
    for idx, module in enumerate(modules):
        module = dict(module)
        module["r0"] = r
        module["c0"] = c
        placed.append(module)
        if idx + 1 < len(modules):
            gap = random.choice(gap_choices)
            r += module["outer_h"] + gap + 1
            c += module["outer_w"] + gap + 1
    h = placed[-1]["r0"] + placed[-1]["outer_h"] + random.randint(1, 2)
    w = placed[-1]["c0"] + placed[-1]["outer_w"] + random.randint(1, 2)
    if h > 30 or w > 30:
        return None
    return placed, h, w


def _draw_module(grid: np.ndarray, module: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0 = int(module["r0"])
    c0 = int(module["c0"])
    h = int(module["outer_h"])
    w = int(module["outer_w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = ORANGE
    grid[r0 : r1 + 1, c1] = ORANGE
    grid[r0, c0 : c1 + 1] = ORANGE
    grid[r1, c0 : c1 + 1] = ORANGE

    if module["orientation"] == "vertical":
        left_h, left_w = module["chambers"][0][1], module["chambers"][0][2]
        right_h, right_w = module["chambers"][1][1], module["chambers"][1][2]
        assert left_h == right_h == h - 2
        divider_c = c0 + 1 + left_w
        grid[r0 : r1 + 1, divider_c] = ORANGE
        return (r0 + 1, c0 + 1, left_h, left_w), (r0 + 1, divider_c + 1, right_h, right_w)

    top_h, top_w = module["chambers"][0][1], module["chambers"][0][2]
    bottom_h, bottom_w = module["chambers"][1][1], module["chambers"][1][2]
    assert top_w == bottom_w == w - 2
    divider_r = r0 + 1 + top_h
    grid[divider_r, c0 : c1 + 1] = ORANGE
    return (r0 + 1, c0 + 1, top_h, top_w), (divider_r + 1, c0 + 1, bottom_h, bottom_w)


def _connect_modules_diagonally(grid: np.ndarray, modules: list[dict]) -> None:
    for left, right in zip(modules, modules[1:]):
        r = int(left["r0"] + left["outer_h"])
        c = int(left["c0"] + left["outer_w"])
        end_r = int(right["r0"] - 1)
        end_c = int(right["c0"] - 1)
        assert end_r - r == end_c - c and end_r >= r
        for step in range(end_r - r + 1):
            grid[r + step, c + step] = ORANGE


def _derive_output_from_metadata(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        _paint_patch(out, pair["target_patch"], pair["target_pos"][0], pair["target_pos"][1])
    return out


def _check_instance(inp: np.ndarray, expected: np.ndarray, pair_count: int) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver output mismatch"

    region_labels, region_coords = _label_mask(inp != ORANGE)
    source_count = 0
    target_count = 0
    for coords in region_coords:
        vals = set(int(inp[r, c]) for r, c in coords)
        if GREY in vals:
            target_count += 1
        elif vals & set(SPRITE_COLORS):
            source_count += 1
    assert source_count == target_count == pair_count


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    module_count = random.choice([2, 3])
    orientations = ["vertical", "horizontal"]
    if module_count == 3:
        orientations.append(random.choice(["vertical", "horizontal"]))
    random.shuffle(orientations)

    placement = _position_modules(orientations)
    if placement is None:
        raise RuntimeError("module layout exceeds the ARC size limit")
    modules, h, w = placement
    inp = np.zeros((h, w), dtype=int)
    pair_meta: list[dict] = []
    chamber_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]] = []

    for module in modules:
        chamber_a, chamber_b = _draw_module(inp, module)
        if random.random() < 0.5:
            source_chamber, target_chamber = chamber_a, chamber_b
        else:
            source_chamber, target_chamber = chamber_b, chamber_a
        chamber_pairs.append((source_chamber, target_chamber, module["orientation"]))

    _connect_modules_diagonally(inp, modules)

    for source_chamber, target_chamber, orientation in chamber_pairs:
        _, _, source_h, source_w = source_chamber
        _, _, target_h, target_w = target_chamber
        source_mask, target_mask, rotation_k = _choose_shape(source_h, source_w, target_h, target_w, orientation)
        source_patch = _colorize(source_mask)
        target_patch = np.rot90(_flip_patch(source_patch, orientation), rotation_k)
        source_pos = _place_in_chamber(source_chamber, source_patch)
        target_pos = _place_in_chamber(target_chamber, target_patch)
        placeholder_patch = np.where(target_patch != BLACK, GREY, BLACK)

        _paint_patch(inp, source_patch, source_pos[0], source_pos[1])
        _paint_patch(inp, placeholder_patch, target_pos[0], target_pos[1])
        pair_meta.append(
            {
                "source_patch": source_patch,
                "target_patch": target_patch,
                "source_pos": source_pos,
                "target_pos": target_pos,
                "orientation": orientation,
            }
        )

    out = _derive_output_from_metadata(inp, pair_meta)
    _check_instance(inp, out, len(pair_meta))
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            inp, _out = _generate_instance()
        except (AssertionError, RuntimeError):
            continue
        return inp
    raise RuntimeError("failed to generate a valid contour-mirror-stamp instance")
