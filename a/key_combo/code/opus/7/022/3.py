from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLOR = PINK
BOUNDARY_COLOR = GREEN
GLYPH_H = 3
GLYPH_W = 2
MAX_GENERATION_ATTEMPTS = 400

_GLYPH_TO_COLOR = {
    "A": RED,
    "B": BLUE,
    "C": YELLOW,
    "D": ORANGE,
}

_GLYPH_PATTERNS = {
    "A": np.array([[0, 0], [0, 5], [5, 5]], dtype=int),
    "B": np.array([[5, 0], [5, 0], [0, 5]], dtype=int),
    "C": np.array([[5, 5], [0, 0], [5, 0]], dtype=int),
    "D": np.array([[0, 5], [5, 5], [0, 0]], dtype=int),
}

_LEFT_PALETTE = [BLACK, BLUE, RED, YELLOW, GREY, ORANGE, TEAL, MAROON]
_LEFT_WEIGHTS = np.array([0.36, 0.10, 0.08, 0.10, 0.08, 0.10, 0.09, 0.09], dtype=float)
_LEFT_WEIGHTS /= _LEFT_WEIGHTS.sum()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _solid_rect_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
        return box
    return None


def _mirror_boxes(h: int, w: int, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = box
    return [
        (r0, w - 1 - c1, r1, w - 1 - c0),
        (h - 1 - r1, c0, h - 1 - r0, c1),
        (h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0),
    ]


def _recover_marker_patch(left: np.ndarray, box: tuple[int, int, int, int], marker: int) -> np.ndarray:
    h, w = left.shape
    r0, c0, r1, c1 = box
    patch = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            vals = []
            for rr, cc in ((h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)):
                if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    v = int(left[rr, cc])
                    if v != marker:
                        vals.append(v)
            if not vals:
                raise AssertionError("marker cell has no visible mirror source")
            if len(set(vals)) != 1:
                raise AssertionError("mirror sources disagree")
            patch[r - r0, c - c0] = vals[0]
    return patch


def _glyph_key(block: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in block.flatten())


_PATTERN_TO_COLOR = {
    _glyph_key(pattern): _GLYPH_TO_COLOR[name]
    for name, pattern in _GLYPH_PATTERNS.items()
}


def _decode_glyph_patch(patch: np.ndarray) -> list[int]:
    if patch.shape[0] != GLYPH_H or patch.shape[1] % GLYPH_W != 0:
        raise AssertionError("unexpected glyph patch shape")
    colors = []
    for i in range(patch.shape[1] // GLYPH_W):
        block = patch[:, i * GLYPH_W : (i + 1) * GLYPH_W]
        key = _glyph_key(block)
        if key not in _PATTERN_TO_COLOR:
            raise AssertionError("unknown glyph pattern")
        colors.append(_PATTERN_TO_COLOR[key])
    return colors


def _find_enclosed_regions(right: np.ndarray) -> list[np.ndarray]:
    h, w = right.shape
    blocked = right == BOUNDARY_COLOR
    exterior = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    enclosed = (right == BLACK) & ~exterior
    regions = _components(enclosed)
    regions.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return regions


def _filled_shape_to_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                boundary[r, c] = True
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if not mask[r + dr, c + dc]:
                    boundary[r, c] = True
                    break
    return boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden glyphs from the mirrored left pattern, decode them into colors, and fill the matching closed regions on the right.

    Input structure:
    - The grid has black background and is logically split into left and right halves of equal width.
    - The left half is horizontally and vertically symmetric except for one solid magenta rectangle hiding a strip of gray-and-black glyph blocks.
    - The right half contains the same number of green closed boundaries as hidden glyph blocks, ordered from top to bottom.

    Rule:
    - Find the unique magenta rectangle in the left half and recover every hidden pixel from its horizontal, vertical, or 180-degree mirror counterpart.
    - Split the restored rectangle into adjacent 3x2 gray glyph blocks and decode each block with the fixed glyph-to-color table.
    - Detect the enclosed black regions inside the green boundaries on the right half, sort them from top to bottom, and flood-fill each interior with the corresponding decoded color.
    - Return the full grid with the left half restored and the right-half interiors filled, while preserving the green boundaries.

    Key insight:
    - The magenta rectangle is only an occlusion mask: symmetry is needed to reconstruct the glyph pixels, and only the reconstructed glyph orientation yields the color sequence that drives the right-side region filling.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[1] % 2 != 0:
        raise AssertionError("expected a 2D grid with even width")

    h, w = g.shape
    half = w // 2
    left = g[:, :half]
    right = g[:, half:]

    marker_mask = left == MARKER_COLOR
    marker_box = _solid_rect_bbox(marker_mask)
    if marker_box is None:
        raise AssertionError("expected one solid marker rectangle")

    restored_patch = _recover_marker_patch(left, marker_box, MARKER_COLOR)
    colors = _decode_glyph_patch(restored_patch)
    regions = _find_enclosed_regions(right)
    if len(regions) != len(colors):
        raise AssertionError("glyph count and enclosed-region count must match")

    out = g.copy()
    r0, c0, r1, c1 = marker_box
    out[r0 : r1 + 1, c0 : c1 + 1] = restored_patch
    for color, coords in zip(colors, regions):
        out[coords[:, 0], half + coords[:, 1]] = color
    return out


def _patch_decodable(patch: np.ndarray) -> bool:
    try:
        _decode_glyph_patch(patch)
    except AssertionError:
        return False
    return True


def _choose_marker_box(h: int, w: int, n_glyphs: int) -> tuple[int, int, int, int]:
    patch_h = GLYPH_H
    patch_w = GLYPH_W * n_glyphs
    options = []
    for r0 in range(0, h - patch_h + 1):
        for c0 in range(0, w - patch_w + 1):
            box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            mirrors = _mirror_boxes(h, w, box)
            all_boxes = [box, *mirrors]
            if len(set(all_boxes)) != 4:
                continue
            overlap = False
            for i in range(4):
                ar0, ac0, ar1, ac1 = all_boxes[i]
                for j in range(i + 1, 4):
                    br0, bc0, br1, bc1 = all_boxes[j]
                    if max(ar0, br0) <= min(ar1, br1) and max(ac0, bc0) <= min(ac1, bc1):
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                options.append(box)
    if not options:
        raise RuntimeError("no valid marker position")
    return random.choice(options)


def _random_left_symmetric_grid(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (h - 1 - r, c),
                (r, w - 1 - c),
                (h - 1 - r, w - 1 - c),
            }
            color = int(np.random.choice(_LEFT_PALETTE, p=_LEFT_WEIGHTS))
            for rr, cc in orbit:
                grid[rr, cc] = color
                seen.add((rr, cc))
    return grid


def _glyph_patch_from_names(names: list[str]) -> np.ndarray:
    return np.hstack([_GLYPH_PATTERNS[name] for name in names]).astype(int)


def _embed_patch_by_orbit(left: np.ndarray, patch: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = left.shape
    r0, c0, _r1, _c1 = box
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            color = int(patch[dr, dc])
            r = r0 + dr
            c = c0 + dc
            for rr, cc in {
                (r, c),
                (h - 1 - r, c),
                (r, w - 1 - c),
                (h - 1 - r, w - 1 - c),
            }:
                left[rr, cc] = color


def _counterpart_patches(left: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    h, w = left.shape
    r0, c0, r1, c1 = box
    patches = []
    for mr0, mc0, mr1, mc1 in _mirror_boxes(h, w, box):
        patches.append(left[mr0 : mr1 + 1, mc0 : mc1 + 1].copy())
    return patches


def _random_region_shape(h: int, w: int) -> np.ndarray:
    filled = np.ones((h, w), dtype=bool)
    variant = random.choice(["rect", "rect", "notch_top", "notch_bottom"])
    if variant == "notch_top" and h >= 5 and w >= 5:
        depth = random.randint(1, h - 3)
        span = random.randint(1, w - 3)
        start = random.randint(1, w - span - 1)
        filled[:depth, start : start + span] = False
    elif variant == "notch_bottom" and h >= 5 and w >= 5:
        depth = random.randint(1, h - 3)
        span = random.randint(1, w - 3)
        start = random.randint(1, w - span - 1)
        filled[h - depth :, start : start + span] = False
    boundary = _filled_shape_to_boundary(filled)
    interior = filled & ~boundary
    if interior.sum() == 0:
        return _random_region_shape(h, w)
    return boundary


def _build_right_half(h: int, w: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    count = len(colors)
    for _ in range(300):
        if count == 2:
            heights = [random.randint(4, 6), random.randint(4, 6)]
        else:
            heights = [random.randint(4, 5) for _ in range(3)]
        min_total = sum(heights) + (count - 1)
        if min_total > h - 2:
            continue

        slack = h - 2 - min_total
        gaps = [0] * (count + 1)
        for _ in range(slack):
            gaps[random.randrange(count + 1)] += 1

        rows = []
        cur = 1 + gaps[0]
        for i, height in enumerate(heights):
            rows.append(cur)
            cur += height + 1 + gaps[i + 1]

        widths = []
        boundary_patches = []
        interior_areas = []
        ok = True
        for i, height in enumerate(heights):
            width = random.randint(4, max(4, w - 2))
            width = min(width, w - 2)
            if width < 4:
                ok = False
                break
            boundary = _random_region_shape(height, width)
            area = int(((~_find_exterior(boundary)) & (boundary == 0)).sum())
            widths.append(width)
            boundary_patches.append(boundary)
            interior_areas.append(area)
        if not ok or len(set(interior_areas)) != count:
            continue

        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        for idx, (r0, width, boundary) in enumerate(zip(rows, widths, boundary_patches)):
            c0 = random.randint(1, w - width - 1)
            patch = np.zeros((boundary.shape[0], boundary.shape[1]), dtype=int)
            patch[boundary] = BOUNDARY_COLOR
            inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
            out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

            sub = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
            regions = _find_enclosed_regions(sub)
            if len(regions) != 1:
                ok = False
                break
            coords = regions[0]
            sub[coords[:, 0], coords[:, 1]] = colors[idx]
        if ok and len(_find_enclosed_regions(inp)) == count:
            return inp, out
    raise RuntimeError("failed to build right half")


def _find_exterior(boundary: np.ndarray) -> np.ndarray:
    h, w = boundary.shape
    blocked = boundary.astype(bool)
    exterior = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _validate_sample(inp: np.ndarray, expected: np.ndarray, box: tuple[int, int, int, int], names: list[str]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    h, w = inp.shape
    half = w // 2
    left = expected[:, :half]
    right_inp = inp[:, half:]
    patch = left[box[0] : box[2] + 1, box[1] : box[3] + 1]
    if not np.array_equal(patch, _glyph_patch_from_names(names)):
        return False
    if _decode_glyph_patch(patch) != [_GLYPH_TO_COLOR[name] for name in names]:
        return False

    marker_mask = inp[:, :half] == MARKER_COLOR
    if _solid_rect_bbox(marker_mask) != box:
        return False

    regions = _find_enclosed_regions(right_inp)
    if len(regions) != len(names):
        return False
    areas = [int(coords.shape[0]) for coords in regions]
    if len(set(areas)) != len(areas):
        return False

    for counterpart in _counterpart_patches(left, box):
        if _patch_decodable(counterpart):
            return False

    if np.array_equal(inp, expected):
        return False
    return True


def _generate_one() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], list[str]]:
    n_glyphs = random.choice([2, 3])
    h = random.choice([14, 16, 18] if n_glyphs == 2 else [16, 18])
    half = random.choice([8, 10] if n_glyphs == 2 else [12, 14])

    glyph_names = random.sample(list(_GLYPH_PATTERNS), k=n_glyphs)
    glyph_patch = _glyph_patch_from_names(glyph_names)
    box = _choose_marker_box(h, half, n_glyphs)

    left = _random_left_symmetric_grid(h, half)
    _embed_patch_by_orbit(left, glyph_patch, box)
    for counterpart in _counterpart_patches(left, box):
        if _patch_decodable(counterpart):
            raise RuntimeError("decodable visible counterpart")

    colors = [_GLYPH_TO_COLOR[name] for name in glyph_names]
    right_inp, right_out = _build_right_half(h, half, colors)

    expected = np.hstack([left, right_out])
    inp = expected.copy()
    r0, c0, r1, c1 = box
    inp[r0 : r1 + 1, c0 : c1 + 1] = MARKER_COLOR
    inp[:, half:] = right_inp
    return inp, expected, box, glyph_names


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected, box, glyph_names = _generate_one()
        except RuntimeError:
            continue
        if _validate_sample(inp, expected, box, glyph_names):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
