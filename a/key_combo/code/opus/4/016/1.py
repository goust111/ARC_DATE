from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER_OFFSETS = {
    7: (-3, 0),  # orange -> up
    1: (0, 3),   # blue -> right
    9: (3, 0),   # maroon -> down
    6: (0, -3),  # magenta -> left
}
MARKER_COLORS = tuple(MARKER_OFFSETS)
SOURCE_COLORS = (2, 3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500

VALID_RING = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=int,
)

BROKEN_RING_VARIANTS = (
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=int),
)

INVALID_SHAPES = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=int),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in components_from_mask(g == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                }
            )
    return out


def coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([[r0, c0]])
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def has_hole(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~exterior)))


def paste_sprite(out: np.ndarray, sprite: np.ndarray, top: tuple[int, int], bg: int) -> None:
    r0, c0 = top
    h, w = sprite.shape
    patch = out[r0 : r0 + h, c0 : c0 + w]
    fg = sprite != bg
    patch[fg] = sprite[fg]


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    fg_components = components_from_mask(g != bg)
    source_candidates = []
    for coords in fg_components:
        colors = sorted(set(map(int, g[coords[:, 0], coords[:, 1]])))
        if len(colors) > 1:
            source_candidates.append((coords, colors))
    assert len(source_candidates) == 1, "expected exactly one multicolor source sprite"

    source_coords, source_colors = source_candidates[0]
    source_bbox = bbox_coords(source_coords)
    sr0, sc0, sr1, sc1 = source_bbox
    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert len(source_colors) >= 2, "source sprite must be multicolor"

    source_set = {tuple(map(int, p)) for p in source_coords}
    markers = []
    for comp in monochrome_components(g, bg):
        coords = comp["coords"]
        if any(tuple(map(int, p)) in source_set for p in coords):
            continue
        color = int(comp["color"])
        assert color in MARKER_OFFSETS, "unexpected marker color"
        mask = coords_to_mask(coords, comp["bbox"])
        markers.append(
            {
                "color": color,
                "coords": coords,
                "bbox": comp["bbox"],
                "mask": mask,
                "has_hole": has_hole(mask),
            }
        )
    assert markers, "expected marker objects"
    return {
        "bg": bg,
        "sprite": sprite,
        "sprite_bbox": source_bbox,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the multicolor source sprite at color-decoded offsets of hole-containing monochrome markers.

    Input structure:
    - Black is the background. One 4-connected multicolor source sprite sits in the upper-left region.
    - Elsewhere, 4-connected monochrome marker objects use only orange, blue, maroon, or magenta.
    - Some marker objects enclose a 4-connected black hole inside their bbox; others are hole-free distractors.

    Rule:
    - Detect the unique multicolor source sprite and every other monochrome foreground object.
    - Keep only the marker objects whose mask contains an enclosed black hole; erase all hole-free markers.
    - Decode each surviving marker color as an offset: orange=up 3, blue=right 3, maroon=down 3, magenta=left 3.
    - For each surviving marker, add that offset to the marker bbox's upper-left corner and stamp the full source sprite there.
    - Erase the original source sprite and all marker objects; the output contains only the stamped source copies.

    Key insight:
    - Topology gates which objects are active, while marker color independently encodes where the source copy lands; both cues are required before stamping.
    """
    info = analyze_grid(grid)
    g = np.asarray(grid, dtype=int)
    bg = int(info["bg"])
    sprite = info["sprite"]
    h, w = g.shape
    out = np.full_like(g, bg)

    targets = []
    for marker in info["markers"]:
        if not marker["has_hole"]:
            continue
        r0, c0, _, _ = marker["bbox"]
        dr, dc = MARKER_OFFSETS[int(marker["color"])]
        tr, tc = r0 + dr, c0 + dc
        assert 0 <= tr and 0 <= tc
        assert tr + sprite.shape[0] <= h and tc + sprite.shape[1] <= w
        targets.append((int(tr), int(tc)))

    for top in sorted(targets):
        paste_sprite(out, sprite, top, bg)
    return out


def transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=int, copy=True)
    for _ in range(random.randint(0, 3)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    rows, cols = np.where(out)
    return out[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def random_connected_mask(height_range: tuple[int, int], width_range: tuple[int, int], area_range: tuple[int, int]) -> np.ndarray:
    for _ in range(200):
        h = random.randint(*height_range)
        w = random.randint(*width_range)
        min_area = max(area_range[0], 1)
        max_area = min(area_range[1], h * w)
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target:
            candidates = set()
            for r, c in frontier:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        candidates.add((nr, nc))
            if not candidates:
                break
            cell = random.choice(tuple(candidates))
            mask[cell] = True
            frontier.add(cell)
        if int(mask.sum()) != target:
            continue
        coords = np.argwhere(mask)
        rows = {int(r) for r, _ in coords}
        cols = {int(c) for _, c in coords}
        if len(rows) < 2 or len(cols) < 2:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if len(components_from_mask(cropped)) != 1:
            continue
        if has_hole(cropped):
            continue
        return cropped.astype(int)
    raise RuntimeError("failed to generate connected mask")


def random_partition(total: int, parts: int, minimum: int) -> list[int]:
    counts = [minimum] * parts
    remaining = total - minimum * parts
    for _ in range(remaining):
        counts[random.randrange(parts)] += 1
    random.shuffle(counts)
    return counts


def generate_sprite() -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(2, 3)
        mask = random_connected_mask((3, 4), (3, 5), (2 * n_colors, 9))
        n_cells = int(mask.sum())
        if n_cells < 2 * n_colors:
            continue
        colors = random.sample(SOURCE_COLORS, n_colors)
        counts = random_partition(n_cells, n_colors, 2)
        coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
        random.shuffle(coords)
        sprite = np.zeros_like(mask, dtype=int)
        idx = 0
        for color, count in zip(colors, counts):
            for r, c in coords[idx : idx + count]:
                sprite[r, c] = int(color)
            idx += count
        return sprite
    raise RuntimeError("failed to generate sprite")


def rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def rect_in_bounds(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def choose_invalid_mask(prefer_ring_like: bool = False) -> np.ndarray:
    if prefer_ring_like or random.random() < 0.35:
        return transform_mask(random.choice(BROKEN_RING_VARIANTS))
    return transform_mask(random.choice(INVALID_SHAPES))


def marker_specs() -> list[dict]:
    total = random.randint(5, 6)
    base_valid = set(random.sample(MARKER_COLORS, random.randint(2, 3)))
    specs = [{"color": c, "has_hole": c in base_valid, "prefer_ring_like": False} for c in MARKER_COLORS]
    extras = total - 4
    if extras >= 1:
        color = random.choice(tuple(base_valid))
        specs.append({"color": color, "has_hole": False, "prefer_ring_like": True})
    if extras >= 2:
        invalid_colors = [c for c in MARKER_COLORS if c not in base_valid]
        color = random.choice(invalid_colors)
        specs.append({"color": color, "has_hole": True, "prefer_ring_like": False})
    random.shuffle(specs)
    return specs


def place_object(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    blocked: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    for _ in range(300):
        r0 = random.randint(0, h - mh)
        c0 = random.randint(0, w - mw)
        rect = (r0, c0, r0 + mh - 1, c0 + mw - 1)
        if any(rect_overlap(rect, other, pad=1) for other in blocked):
            continue
        patch = grid[r0 : r0 + mh, c0 : c0 + mw]
        if np.any((mask != 0) & (patch != BG)):
            continue
        patch[mask != 0] = int(color)
        return rect
    return None


def build_expected(sprite: np.ndarray, targets: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for top in sorted(targets):
        paste_sprite(out, sprite, top, BG)
    return out


def validate_semantics(grid: np.ndarray, expected: np.ndarray) -> None:
    info = analyze_grid(grid)
    sprite = info["sprite"]
    markers = info["markers"]
    assert len(markers) >= 5
    assert {m["color"] for m in markers} == set(MARKER_COLORS)
    assert any(m["has_hole"] for m in markers)
    assert any(not m["has_hole"] for m in markers)

    status_by_color = {color: set() for color in MARKER_COLORS}
    target_rects = []
    valid_targets = []
    sh, sw = sprite.shape
    sr0, sc0, sr1, sc1 = info["sprite_bbox"]
    source_rect = (sr0, sc0, sr1, sc1)
    for marker in markers:
        status_by_color[marker["color"]].add(bool(marker["has_hole"]))
        if not marker["has_hole"]:
            continue
        r0, c0, _, _ = marker["bbox"]
        dr, dc = MARKER_OFFSETS[marker["color"]]
        tr, tc = r0 + dr, c0 + dc
        rect = (tr, tc, tr + sh - 1, tc + sw - 1)
        assert rect_in_bounds(rect, grid.shape)
        assert not rect_overlap(rect, source_rect, pad=0)
        assert all(not rect_overlap(rect, other, pad=0) for other in target_rects)
        target_rects.append(rect)
        valid_targets.append((tr, tc))
    assert any(len(v) == 2 for v in status_by_color.values()), "need same-color hole and no-hole counterexample"
    assert np.array_equal(main(grid), expected)

    derived = build_expected(sprite, valid_targets, grid.shape)
    assert np.array_equal(expected, derived)
    assert int(np.count_nonzero(expected)) == int(np.count_nonzero(sprite)) * len(valid_targets)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(16, 19)
        w = random.randint(16, 19)
        grid = np.zeros((h, w), dtype=int)

        sprite = generate_sprite()
        sh, sw = sprite.shape
        sr = random.randint(1, 2)
        sc = random.randint(1, 2)
        grid[sr : sr + sh, sc : sc + sw] = sprite
        source_rect = (sr, sc, sr + sh - 1, sc + sw - 1)
        input_reserved: list[tuple[int, int, int, int]] = [source_rect]
        target_reserved: list[tuple[int, int, int, int]] = [source_rect]

        specs = marker_specs()
        valid_specs = [spec for spec in specs if spec["has_hole"]]
        invalid_specs = [spec for spec in specs if not spec["has_hole"]]
        targets: list[tuple[int, int]] = []
        ok = True

        for spec in valid_specs:
            color = int(spec["color"])
            dr, dc = MARKER_OFFSETS[color]
            mask = VALID_RING.copy()
            for _ in range(300):
                r0 = random.randint(0, h - 3)
                c0 = random.randint(0, w - 3)
                target_rect = (r0 + dr, c0 + dc, r0 + dr + sh - 1, c0 + dc + sw - 1)
                rect = (r0, c0, r0 + 2, c0 + 2)
                if not rect_in_bounds(target_rect, grid.shape):
                    continue
                if any(rect_overlap(rect, other, pad=1) for other in input_reserved):
                    continue
                if any(rect_overlap(target_rect, other, pad=1) for other in target_reserved):
                    continue
                patch = grid[r0 : r0 + 3, c0 : c0 + 3]
                if np.any((mask != 0) & (patch != BG)):
                    continue
                patch[mask != 0] = color
                input_reserved.append(rect)
                target_reserved.append(target_rect)
                targets.append((r0 + dr, c0 + dc))
                break
            else:
                ok = False
                break
        if not ok:
            continue

        for spec in invalid_specs:
            color = int(spec["color"])
            mask = choose_invalid_mask(prefer_ring_like=bool(spec["prefer_ring_like"]))
            rect = place_object(grid, mask, color, input_reserved + target_reserved)
            if rect is None:
                ok = False
                break
            input_reserved.append(rect)
        if not ok:
            continue

        expected = build_expected(sprite, targets, grid.shape)
        try:
            validate_semantics(grid, expected)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
