import random
from collections import Counter, deque

import numpy as np


TRANSFORM_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)
ROTATION_NAMES = {"rot90", "rot180", "rot270"}
REFLECTION_NAMES = {"flip_lr", "flip_ud", "transpose", "anti_transpose"}
QUADRANT_NAMES = ("tl", "tr", "bl", "br")
NAIVE_BY_QUADRANT = {"tr": "flip_lr", "bl": "flip_ud", "br": "rot180"}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude=()) -> list[int]:
    blocked = {int(x) for x in exclude}
    return [int(v) for v in np.unique(grid) if int(v) not in blocked]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_mask(mask: np.ndarray) -> np.ndarray:
    box = mask_bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def transform_array(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_lr":
        return np.fliplr(arr)
    if name == "flip_ud":
        return np.flipud(arr)
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.rot90(arr.T, 2)
    raise ValueError(f"unknown transform {name}")


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(pts, dtype=int))
    return out


def orbit_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {
        (transform_array(mask, name).shape, transform_array(mask, name).astype(np.uint8).tobytes())
        for name in TRANSFORM_NAMES
    }


def quadrant_views(grid: np.ndarray, q: int) -> dict[str, np.ndarray]:
    return {
        "tl": grid[:q, :q],
        "tr": grid[:q, q + 1 :],
        "bl": grid[q + 1 :, :q],
        "br": grid[q + 1 :, q + 1 :],
    }


def outer_corner_position(q: int, h: int, w: int, quadrant: str) -> tuple[int, int]:
    if quadrant == "tl":
        return 0, 0
    if quadrant == "tr":
        return 0, q - w
    if quadrant == "bl":
        return q - h, 0
    if quadrant == "br":
        return q - h, q - w
    raise ValueError(quadrant)


def inner_corner_position(q: int, h: int, w: int, quadrant: str, gap_r: int, gap_c: int) -> tuple[int, int]:
    if quadrant == "tl":
        return q - gap_r - h, q - gap_c - w
    if quadrant == "tr":
        return q - gap_r - h, gap_c
    if quadrant == "bl":
        return gap_r, q - gap_c - w
    if quadrant == "br":
        return gap_r, gap_c
    raise ValueError(quadrant)


def find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    assert h == w and h % 2 == 1
    hits = []
    for color in colors_present(grid, exclude={bg}):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        for r in rows:
            for c in cols:
                hits.append((int(color), int(r), int(c)))
    assert len(hits) == 1
    color, r, c = hits[0]
    assert r == c
    return color, r, c


def parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)
    cross_color, center_r, center_c = find_cross(g, bg)
    q = center_r
    assert center_r == center_c and g.shape == (2 * q + 1, 2 * q + 1)

    quads = quadrant_views(g, q)
    remaining = colors_present(g, exclude={bg, cross_color})
    presence = {
        color: tuple(name for name in QUADRANT_NAMES if np.any(quads[name] == color))
        for color in remaining
    }
    glyph_colors = [color for color, qs in presence.items() if len(qs) == 4]
    pattern_colors = [color for color, qs in presence.items() if qs == ("tl",)]
    assert len(glyph_colors) == 1 and len(pattern_colors) == 1
    glyph_color = glyph_colors[0]
    pattern_color = pattern_colors[0]

    glyph_tiles: dict[str, np.ndarray] = {}
    for name, quad in quads.items():
        glyph_mask = quad == glyph_color
        comps = components(glyph_mask)
        assert len(comps) == 1
        r0, c0, r1, c1 = bbox_coords(comps[0])
        tile = glyph_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert tile.shape[0] >= 3 and tile.shape[1] >= 3
        assert np.any(tile[0, :]) and np.any(tile[-1, :]) and np.any(tile[:, 0]) and np.any(tile[:, -1])
        glyph_tiles[name] = tile

    base_tile = glyph_tiles["tl"]
    glyph_transforms = {}
    for name, tile in glyph_tiles.items():
        matches = [t for t in TRANSFORM_NAMES if np.array_equal(transform_array(base_tile, t), tile)]
        assert len(matches) == 1
        glyph_transforms[name] = matches[0]

    pattern_mask = quads["tl"] == pattern_color
    pattern_comps = components(pattern_mask)
    assert len(pattern_comps) == 1
    pr0, pc0, pr1, pc1 = bbox_coords(pattern_comps[0])
    source_pattern = pattern_mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
    gap_r = q - 1 - pr1
    gap_c = q - 1 - pc1
    assert gap_r >= 0 and gap_c >= 0

    return {
        "grid": g,
        "bg": bg,
        "cross_color": cross_color,
        "q": q,
        "quads": quads,
        "glyph_color": glyph_color,
        "pattern_color": pattern_color,
        "glyph_tiles": glyph_tiles,
        "glyph_transforms": glyph_transforms,
        "source_pattern": source_pattern,
        "source_bbox": (pr0, pc0, pr1, pc1),
        "inner_gap": (gap_r, gap_c),
    }


def build_output(parsed: dict) -> np.ndarray:
    g = parsed["grid"]
    out = np.full_like(g, parsed["bg"])
    q = parsed["q"]
    cross_color = parsed["cross_color"]
    pattern_color = parsed["pattern_color"]
    gap_r, gap_c = parsed["inner_gap"]

    out[q, :] = cross_color
    out[:, q] = cross_color

    source_mask = parsed["quads"]["tl"] == pattern_color
    out[:q, :q][source_mask] = pattern_color

    for quadrant in ("tr", "bl", "br"):
        transformed = transform_array(parsed["source_pattern"], parsed["glyph_transforms"][quadrant])
        r0, c0 = inner_corner_position(q, transformed.shape[0], transformed.shape[1], quadrant, gap_r, gap_c)
        view = out[:q, q + 1 :] if quadrant == "tr" else out[q + 1 :, :q] if quadrant == "bl" else out[q + 1 :, q + 1 :]
        view[r0 : r0 + transformed.shape[0], c0 : c0 + transformed.shape[1]][transformed] = pattern_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each quadrant glyph pose as a geometric transform and copy the upper-left pattern accordingly.

    Input structure:
    - A square grid is split into four equal quadrants by one full horizontal and one full vertical line of the same color.
    - The upper-left quadrant contains one foreground pattern near the inner corner and one small glyph in the outer corner.
    - The other three quadrants contain only the same glyph in their own outer corners, but each glyph may be rotated or mirrored.

    Rule:
    - Treat the upper-left glyph as the base pose, which means identity.
    - For each of the other three quadrants, determine which rotation or mirror turns the base glyph into that quadrant's glyph.
    - Apply that same transform to the upper-left foreground pattern, place the transformed copy at the corresponding inner corner of the target quadrant, keep the cross and original upper-left pattern, and erase all four glyphs.

    Key insight:
    - The quadrant position does not determine the copy transform; the glyph's pose is the instruction, so examples where glyph pose disagrees with the usual quadrant mirror layout rule out simpler fixed-symmetry shortcuts.
    """
    parsed = parse_grid(grid)
    return build_output(parsed)


def random_connected_mask(height: int, width: int, cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < cells:
        rr, cc = random.choice(tuple(frontier))
        candidates = []
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                candidates.append((nr, nc))
        if not candidates:
            frontier.discard((rr, cc))
            if not frontier:
                frontier = set(map(tuple, np.argwhere(mask)))
            continue
        chosen = random.choice(candidates)
        mask[chosen] = True
        frontier.add(chosen)
    return mask


def connect_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    while (r, c) != (tr, tc):
        if r != tr and c != tc:
            if random.random() < 0.5:
                r += 1 if tr > r else -1
            else:
                c += 1 if tc > c else -1
        elif r != tr:
            r += 1 if tr > r else -1
        else:
            c += 1 if tc > c else -1
        mask[r, c] = True


def generate_glyph(size: int) -> np.ndarray:
    for _ in range(500):
        mask = np.zeros((size, size), dtype=bool)
        seeds = [
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
            (random.randrange(size), 0),
            (random.randrange(size), size - 1),
        ]
        for seed in seeds:
            mask[seed] = True
        order = seeds[:]
        random.shuffle(order)
        for idx in range(1, len(order)):
            connect_points(mask, order[idx - 1], order[idx])

        target_area = random.randint(int(mask.sum()), min(size * size - 2, int(mask.sum()) + size + 2))
        frontier = set(map(tuple, np.argwhere(mask)))
        while int(mask.sum()) < target_area and frontier:
            rr, cc = random.choice(tuple(frontier))
            candidates = []
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                frontier.discard((rr, cc))
                continue
            chosen = random.choice(candidates)
            mask[chosen] = True
            frontier.add(chosen)
        if len(orbit_keys(mask)) != 8:
            continue
        return mask
    raise RuntimeError("failed to generate glyph")


def generate_pattern(max_side: int) -> np.ndarray:
    for _ in range(500):
        canvas_h = random.randint(3, max_side)
        canvas_w = random.randint(3, max_side)
        area = random.randint(4, min(canvas_h * canvas_w - 1, canvas_h + canvas_w + 1))
        mask = random_connected_mask(canvas_h, canvas_w, area)
        cropped = crop_mask(mask)
        if min(cropped.shape) < 3 or max(cropped.shape) > max_side:
            continue
        if len(orbit_keys(cropped)) != 8:
            continue
        return cropped
    raise RuntimeError("failed to generate pattern")


def render_quadrant_pattern(mask: np.ndarray, q: int, quadrant: str, gap_r: int, gap_c: int) -> np.ndarray:
    out = np.zeros((q, q), dtype=bool)
    r0, c0 = inner_corner_position(q, mask.shape[0], mask.shape[1], quadrant, gap_r, gap_c)
    out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = mask
    return out


def validate_generated_grid(grid: np.ndarray) -> None:
    parsed = parse_grid(grid)
    assert len(orbit_keys(parsed["glyph_tiles"]["tl"])) == 8
    assert len(orbit_keys(parsed["source_pattern"])) == 8

    transforms = parsed["glyph_transforms"]
    assert transforms["tl"] == "id"
    target_names = {transforms[q] for q in ("tr", "bl", "br")}
    assert len(target_names) == 3
    assert any(name in ROTATION_NAMES for name in target_names)
    assert any(name in REFLECTION_NAMES for name in target_names)
    assert any(transforms[q] != NAIVE_BY_QUADRANT[q] for q in ("tr", "bl", "br"))

    source_key = (parsed["source_pattern"].shape, parsed["source_pattern"].astype(np.uint8).tobytes())
    image_keys = {source_key}
    for qname in ("tr", "bl", "br"):
        transformed = transform_array(parsed["source_pattern"], transforms[qname])
        image_keys.add((transformed.shape, transformed.astype(np.uint8).tobytes()))
    assert len(image_keys) == 4

    out = main(grid)
    assert out.shape == grid.shape
    bg = parsed["bg"]
    q = parsed["q"]
    out_quads = quadrant_views(out, q)
    assert not np.any(out == parsed["glyph_color"])
    assert np.array_equal(out_quads["tl"] == parsed["pattern_color"], parsed["quads"]["tl"] == parsed["pattern_color"])
    for qname in ("tr", "bl", "br"):
        expected_mask = render_quadrant_pattern(
            transform_array(parsed["source_pattern"], transforms[qname]),
            q,
            qname,
            *parsed["inner_gap"],
        )
        assert np.array_equal(out_quads[qname] == parsed["pattern_color"], expected_mask)
        assert np.all((out_quads[qname] == parsed["pattern_color"]) | (out_quads[qname] == bg))


def generate_input() -> np.ndarray:
    for _ in range(500):
        q = random.randint(8, 12)
        glyph_size = random.randint(4, min(5, q - 3))
        glyph = generate_glyph(glyph_size)
        pattern = generate_pattern(min(5, q - 2))
        gap_r = random.randint(0, 2)
        gap_c = random.randint(0, 2)

        tl_r0, tl_c0 = inner_corner_position(q, pattern.shape[0], pattern.shape[1], "tl", gap_r, gap_c)
        if tl_r0 <= glyph_size or tl_c0 <= glyph_size:
            continue

        other_names = [name for name in TRANSFORM_NAMES if name != "id"]
        rot_choice = random.choice(sorted(ROTATION_NAMES))
        ref_choice = random.choice(sorted(REFLECTION_NAMES))
        third_pool = [name for name in other_names if name not in {rot_choice, ref_choice}]
        third_choice = random.choice(third_pool)
        assigned = [rot_choice, ref_choice, third_choice]
        random.shuffle(assigned)
        transform_by_quadrant = dict(zip(("tr", "bl", "br"), assigned))
        if all(transform_by_quadrant[qname] == NAIVE_BY_QUADRANT[qname] for qname in ("tr", "bl", "br")):
            continue

        colors = random.sample(range(1, 10), 3)
        cross_color, pattern_color, glyph_color = colors

        grid = np.zeros((2 * q + 1, 2 * q + 1), dtype=int)
        grid[q, :] = cross_color
        grid[:, q] = cross_color

        quad_offsets = {"tl": (0, 0), "tr": (0, q + 1), "bl": (q + 1, 0), "br": (q + 1, q + 1)}
        for quadrant in QUADRANT_NAMES:
            gr0, gc0 = quad_offsets[quadrant]
            gmask = glyph if quadrant == "tl" else transform_array(glyph, transform_by_quadrant[quadrant])
            r0, c0 = outer_corner_position(q, gmask.shape[0], gmask.shape[1], quadrant)
            grid[gr0 + r0 : gr0 + r0 + gmask.shape[0], gc0 + c0 : gc0 + c0 + gmask.shape[1]][gmask] = glyph_color

        grid[tl_r0 : tl_r0 + pattern.shape[0], tl_c0 : tl_c0 + pattern.shape[1]][pattern] = pattern_color

        try:
            validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
