from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_TRIES = 400

SHAPE_SPECS = {
    "L5": (
        "11100",
        "11100",
        "11100",
        "11111",
        "11111",
    ),
    "T5": (
        "11111",
        "11111",
        "01110",
        "01110",
        "01110",
    ),
    "F5": (
        "11111",
        "11111",
        "11100",
        "11110",
        "11110",
    ),
    "P5": (
        "11110",
        "11110",
        "11111",
        "11100",
        "11100",
    ),
    "J5": (
        "00111",
        "00111",
        "00111",
        "11111",
        "11111",
    ),
    "Y5": (
        "11111",
        "01111",
        "01110",
        "01110",
        "01110",
    ),
    "S5": (
        "01111",
        "01111",
        "11110",
        "11110",
        "11110",
    ),
    "R5": (
        "11110",
        "11110",
        "11111",
        "11110",
        "11110",
    ),
}


def _parse_mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "bbox requires a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1]).copy()


def _rotate(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, int(k))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_rotate(mask, k)) for k in range(4))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _binary_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif int(connectivity) == 8:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        raise AssertionError("connectivity must be 4 or 8")
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr = int(rr) + int(dr)
                    nc = int(cc) + int(dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            components.append(np.array(coords, dtype=int))
    return components


def _boundary_of_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w and region[nr, nc]):
                    boundary[r, c] = True
                    break
    return boundary


def _interior_of_region(region: np.ndarray) -> np.ndarray:
    return np.asarray(region & ~_boundary_of_region(region), dtype=bool)


def _interior_from_outline(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if outline[r, c] or reachable[r, c]:
                continue
            reachable[r, c] = True
            q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(int(rr), int(cc)):
            if 0 <= nr < h and 0 <= nc < w and not outline[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((int(nr), int(nc)))
    return (~outline) & (~reachable)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1]).copy(), (r0, c0, r1, c1)


def _rotation_to_match(src_outline: np.ndarray, dst_outline: np.ndarray) -> int:
    for k in range(4):
        if np.array_equal(_rotate(src_outline, k), dst_outline):
            return int(k)
    raise AssertionError("no valid rotation between source and template outlines")


def _shape_library() -> list[dict]:
    library: list[dict] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    for name, spec in SHAPE_SPECS.items():
        region = _parse_mask(spec)
        outline = _boundary_of_region(region)
        interior = _interior_of_region(region)
        key = _canonical_rotation_key(outline)
        assert interior.any(), f"{name} must have a real interior"
        assert key not in seen_keys, f"{name} duplicates another outline"
        for k in range(1, 4):
            assert not np.array_equal(outline, _rotate(outline, k)), f"{name} is rotation-symmetric"
        seen_keys.add(key)
        library.append(
            {
                "name": name,
                "region": region,
                "outline": outline,
                "interior": interior,
                "key": key,
            }
        )
    return library


SHAPES = _shape_library()


def _is_connected(mask: np.ndarray) -> bool:
    comps = _binary_components(mask, connectivity=8)
    return len(comps) == 1 if mask.any() else False


def _pattern_candidates(interior: np.ndarray) -> list[np.ndarray]:
    h, w = interior.shape
    rr, cc = np.indices((h, w))
    candidates = [
        interior.copy(),
        interior & (cc % 2 == 0),
        interior & (cc % 2 == 1),
        interior & (rr % 2 == 0),
        interior & (rr % 2 == 1),
        interior & ((rr + cc) % 2 == 0),
        interior & ((rr + cc) % 2 == 1),
        interior & (rr <= cc),
        interior & (rr >= cc),
        interior & (rr + cc <= (h + w - 2) // 2),
        interior & (rr + cc >= (h + w - 2) // 2),
        interior & ((rr == h // 2) | (cc == w // 2)),
        interior & ((rr <= h // 2) | (cc <= w // 2)),
        interior & ((rr >= h // 2) | (cc >= w // 2)),
    ]

    unique: dict[bytes, np.ndarray] = {}
    for cand in candidates:
        if not cand.any():
            continue
        key = cand.tobytes()
        if key not in unique:
            unique[key] = cand
    return list(unique.values())


def _pick_fill_mask(shape: dict, used_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    options = _pattern_candidates(shape["interior"])
    random.shuffle(options)

    valid: list[np.ndarray] = []
    for fill in options:
        if _is_connected(shape["outline"] | fill):
            valid.append(fill)
    assert valid, "at least one pattern must keep the source object connected"

    for fill in valid:
        key = _canonical_rotation_key(fill)
        if key not in used_keys:
            used_keys.add(key)
            return fill.copy()

    fill = random.choice(valid)
    used_keys.add(_canonical_rotation_key(fill))
    return fill.copy()


def _source_object_from_shape(shape: dict, outline_color: int, fill_color: int, rot_k: int, fill_mask: np.ndarray) -> np.ndarray:
    outline = _rotate(shape["outline"], rot_k)
    fill = _rotate(fill_mask, rot_k)
    patch = np.zeros(outline.shape, dtype=int)
    patch[outline] = int(outline_color)
    patch[fill] = int(fill_color)
    return patch


def _template_patch_from_shape(shape: dict, rot_k: int) -> np.ndarray:
    outline = _rotate(shape["outline"], rot_k)
    patch = np.zeros(outline.shape, dtype=int)
    patch[outline] = GREY
    return patch


def _reserve_ok(rects: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, h, w = candidate
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for or0, oc0, oh, ow in rects:
        or1 = or0 + oh - 1
        oc1 = oc0 + ow - 1
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_patches(zone_h: int, zone_w: int, patches: list[np.ndarray], gap: int = 1) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        cells = [(r, c) for r in range(zone_h - h + 1) for c in range(zone_w - w + 1)]
        random.shuffle(cells)
        placed = False
        for r0, c0 in cells:
            cand = (int(r0), int(c0), h, w)
            if _reserve_ok(rects, cand, gap):
                rects.append(cand)
                positions[idx] = (int(r0), int(c0))
                placed = True
                break
        if not placed:
            return None
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions]  # type: ignore[misc]


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not np.any(sub[mask] != BG), "patches must not overlap"
    sub[mask] = patch[mask]


def _separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != int(bg)
    ]
    assert len(rows) == 1, "expected exactly one full separator row"
    return int(rows[0])


def _extract_source_objects(source: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _binary_components(source != bg, connectivity=8):
        patch, _bbox_info = _extract_patch(source, coords)
        colors = [int(v) for v in np.unique(patch) if int(v) != bg]
        assert len(colors) == 2, "source objects must contain exactly outline and fill colors"

        outline_colors: list[int] = []
        for color in colors:
            outline_mask = patch == color
            enclosed = _interior_from_outline(outline_mask)
            other_pixels = (patch != bg) & (~outline_mask)
            if enclosed.any() and np.all(enclosed[other_pixels]):
                outline_colors.append(int(color))
        assert len(outline_colors) == 1, "outline color must be uniquely exposed"
        outline_color = int(outline_colors[0])
        fill_color = int([color for color in colors if color != outline_color][0])
        outline_mask = patch == outline_color
        fill_mask = patch == fill_color
        assert fill_mask.any(), "source object must contain a fill pattern"

        objects.append(
            {
                "outline_color": outline_color,
                "fill_color": fill_color,
                "outline_mask": outline_mask,
                "fill_mask": fill_mask,
                "key": _canonical_rotation_key(outline_mask),
            }
        )
    return objects


def _extract_template_slots(template: np.ndarray) -> list[dict]:
    slots: list[dict] = []
    for coords in _binary_components(template == GREY, connectivity=8):
        patch, bbox_info = _extract_patch(template, coords)
        outline_mask = patch == GREY
        slots.append(
            {
                "bbox": bbox_info,
                "outline_mask": outline_mask,
                "key": _canonical_rotation_key(outline_mask),
            }
        )
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each source outline to its rotated template slot and transfer the rotated fill pattern.

    Input structure:
    - A full-width horizontal separator row splits the grid into an upper source zone and a lower template zone.
    - The source zone has 2-4 non-overlapping multicolor objects on a black background; each object has a 1-pixel outline in one color and an interior fill pattern in another color.
    - The template zone has the same number of grey outline slots on a black background; each slot matches one source outline up to rotation.

    Rule:
    - Extract each source object's outline shape, ignoring its colors for shape matching.
    - Find the grey template slot with the same outline shape, allowing 0/90/180/270-degree rotation.
    - Rotate the source object's fill pattern by the same amount needed to align its outline with the template slot.
    - Recolor the template slot's grey outline to the source outline color, place the rotated fill pattern inside it, and output only the lower template zone.

    Key insight:
    - The mapping depends on three linked steps: shape-based matching, rotation recovery from the matched outlines, and rotating the interior pattern together with the outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    sep = _separator_row(g, bg)

    source = g[:sep, :]
    template = g[sep + 1 :, :]
    assert source.size and template.size, "both source and template zones must exist"

    sources = _extract_source_objects(source, bg)
    slots = _extract_template_slots(template)
    assert len(sources) == len(slots) >= 2

    source_by_key = {obj["key"]: obj for obj in sources}
    slot_by_key = {slot["key"]: slot for slot in slots}
    assert len(source_by_key) == len(sources), "source outline shapes must be unique up to rotation"
    assert len(slot_by_key) == len(slots), "template slots must be unique up to rotation"
    assert set(source_by_key) == set(slot_by_key), "source and template shape sets must match"

    out = np.zeros_like(template)
    for key, obj in source_by_key.items():
        slot = slot_by_key[key]
        rot_k = _rotation_to_match(obj["outline_mask"], slot["outline_mask"])
        rotated_fill = _rotate(obj["fill_mask"], rot_k)
        assert rotated_fill.shape == slot["outline_mask"].shape

        r0, c0, r1, c1 = slot["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[slot["outline_mask"]] = int(obj["outline_color"])
        patch[rotated_fill] = int(obj["fill_color"])
    return out


def _build_expected_output(template_h: int, width: int, objects: list[dict], template_positions: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((template_h, width), dtype=int)
    for obj, (r0, c0) in zip(objects, template_positions, strict=False):
        outline = _rotate(obj["shape"]["outline"], obj["template_rot"])
        fill = _rotate(obj["fill_mask"], obj["template_rot"])
        patch = np.zeros(outline.shape, dtype=int)
        patch[outline] = int(obj["outline_color"])
        patch[fill] = int(obj["fill_color"])
        _paste_nonzero(out, patch, int(r0), int(c0))
    return out


def _random_zone_size(num_objects: int) -> tuple[int, int]:
    min_width = 16 if num_objects <= 2 else 18 if num_objects == 3 else 22
    width = random.randint(min_width, 28)
    height = random.randint(9, 13)
    return height, width


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        num_objects = random.randint(2, 4)
        top_h, width = _random_zone_size(num_objects)
        bottom_h, width2 = _random_zone_size(num_objects)
        width = max(width, width2)
        if width > 28:
            continue
        if top_h + 1 + bottom_h > 30:
            continue

        chosen_shapes = random.sample(SHAPES, num_objects)
        used_fill_keys: set[tuple[tuple[int, int], bytes]] = set()
        objects: list[dict] = []
        source_patches: list[np.ndarray] = []
        template_patches: list[np.ndarray] = []

        for shape in chosen_shapes:
            outline_color = random.choice(PALETTE)
            fill_color = random.choice([c for c in PALETTE if c != outline_color])
            source_rot = random.randrange(4)
            template_rot = random.randrange(4)
            fill_mask = _pick_fill_mask(shape, used_fill_keys)

            source_patch = _source_object_from_shape(shape, outline_color, fill_color, source_rot, fill_mask)
            template_patch = _template_patch_from_shape(shape, template_rot)
            source_patches.append(source_patch)
            template_patches.append(template_patch)
            objects.append(
                {
                    "shape": shape,
                    "outline_color": outline_color,
                    "fill_color": fill_color,
                    "source_rot": source_rot,
                    "template_rot": template_rot,
                    "fill_mask": fill_mask,
                }
            )

        source_positions = _place_patches(top_h, width, source_patches, gap=1)
        template_positions = _place_patches(bottom_h, width, template_patches, gap=1)
        if source_positions is None or template_positions is None:
            continue

        grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
        grid[top_h, :] = AZURE
        for patch, (r0, c0) in zip(source_patches, source_positions, strict=False):
            _paste_nonzero(grid, patch, int(r0), int(c0))
        for patch, (r0, c0) in zip(template_patches, template_positions, strict=False):
            _paste_nonzero(grid, patch, int(top_h + 1 + r0), int(c0))

        expected = _build_expected_output(bottom_h, width, objects, template_positions)
        actual = main(grid)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
