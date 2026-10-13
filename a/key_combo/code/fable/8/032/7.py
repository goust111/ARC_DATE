import math
from collections import Counter, deque

import numpy as np


SCALES = (1, 2, 3, 4)
BACKGROUND = 0
GEN_MAX_ATTEMPTS = 500
GRID_MIN_SIDE = 26
GRID_MAX_SIDE = 30


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = int(quarter_turns) % 4
    return np.rot90(np.asarray(grid, dtype=int), -turns)


def find_components(mask: np.ndarray) -> list[dict]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[dict] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "coords": coords_arr,
                    "area": int(len(coords)),
                    "bbox": _component_bbox(coords_arr),
                }
            )
    return components


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _downscale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    h, w = map(int, patch.shape)
    assert h % factor == 0 and w % factor == 0
    out = np.empty((h // factor, w // factor), dtype=int)
    for r in range(0, h, factor):
        for c in range(0, w, factor):
            block = patch[r : r + factor, c : c + factor]
            color = int(block[0, 0])
            assert np.all(block == color), "scaled sprite blocks must be uniform"
            out[r // factor, c // factor] = color
    return out


def _has_rotational_symmetry(patch: np.ndarray) -> bool:
    return any(np.array_equal(patch, rotate_cw(patch, turns)) for turns in (1, 2, 3))


def _decode(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    components = []
    for component in find_components(g == fg):
        box = _component_bbox(component["coords"])
        patch = crop_bbox(g, box)
        h, w = patch.shape
        assert h == w, "each sprite must have a square tight bbox"
        assert np.any(patch == bg), "each sprite must expose the shared motif, not be a solid block"
        components.append({"bbox": box, "patch": patch, "side": int(h)})

    assert len(components) == 3, "expected exactly three sprites"

    motif_side = math.gcd(*(component["side"] for component in components))
    assert motif_side >= 2, "motif must be at least 2x2"

    by_scale: dict[int, np.ndarray] = {}
    for component in components:
        side = component["side"]
        assert side % motif_side == 0, "sprite side must be an integer multiple of motif size"
        scale = side // motif_side
        assert scale in SCALES, "scale must come from {1,2,3,4}"
        assert scale not in by_scale, "scales must be distinct"
        norm = _downscale_patch(component["patch"], scale)
        assert norm.shape == (motif_side, motif_side), "descaled sprite must recover one motif"
        assert bbox(norm != bg) == (0, 0, motif_side - 1, motif_side - 1), "motif bbox must stay tight"
        by_scale[scale] = norm

    assert len(by_scale) == 3
    missing_scales = sorted(set(SCALES) - set(by_scale))
    assert len(missing_scales) == 1, "there must be exactly one missing scale"
    missing_scale = int(missing_scales[0])

    ref_scale = min(by_scale)
    ref_patch = by_scale[ref_scale]
    assert not _has_rotational_symmetry(ref_patch), "motif must not be rotation-symmetric"

    for scale, norm in by_scale.items():
        expected = rotate_cw(ref_patch, scale - ref_scale)
        assert np.array_equal(norm, expected), "scale and orientation must advance together"

    missing_patch = rotate_cw(ref_patch, missing_scale - ref_scale)
    output = scale_sprite(missing_patch, missing_scale)
    assert bbox(output != bg) == (0, 0, output.shape[0] - 1, output.shape[1] - 1)

    return {
        "bg": bg,
        "fg": fg,
        "motif_side": motif_side,
        "by_scale": by_scale,
        "missing_scale": missing_scale,
        "output": output.astype(int),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared motif after descaling the sprites, then output the missing scale-orientation member.

    Input structure:
    - The background is black and the foreground uses exactly one other color.
    - There are exactly three disconnected monochrome sprites.
    - Each sprite is a tight square crop of the same non-rotation-symmetric n x n motif,
      first quarter-rotated, then pixel-scaled by one of the factors 1, 2, 3, 4.
    - Exactly one factor from 1..4 is absent, and every +1 in scale means one extra clockwise quarter-turn.

    Rule:
    - Detect the three sprite components and read each sprite's tight square side length.
    - Use the common divisor of those side lengths to recover the original motif size n, then
      downscale every sprite by its own integer factor back to n x n.
    - Verify that the descaled motifs form the unique quarter-turn progression tied to the scale order.
    - Find the missing factor in 1..4, rotate the motif to that sequence position, scale it by the
      missing factor, and return only that tight sprite.

    Key insight:
    - The sprites only become comparable after dividing out their pixel scales; once descaled, the
      missing output is the sole absent term of a coupled size-and-rotation sequence.
    """
    return _decode(np.asarray(grid, dtype=int))["output"]


def _random_motif(rng: np.random.Generator, side: int) -> np.ndarray:
    if side == 2:
        mask = np.ones((2, 2), dtype=bool)
        drop = int(rng.integers(4))
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        mask[corners[drop]] = False
        return mask

    for _ in range(200):
        mask = np.zeros((side, side), dtype=bool)
        anchors = {
            (0, int(rng.integers(side))),
            (side - 1, int(rng.integers(side))),
            (int(rng.integers(side)), 0),
            (int(rng.integers(side)), side - 1),
        }
        anchor_list = list(anchors)
        rng.shuffle(anchor_list)
        start = anchor_list[0]
        mask[start] = True
        filled = {start}

        for target in anchor_list[1:]:
            current = list(filled)[int(rng.integers(len(filled)))]
            rr, cc = current
            tr, tc = target
            while (rr, cc) != (tr, tc):
                options = []
                if rr < tr:
                    options.append((rr + 1, cc))
                elif rr > tr:
                    options.append((rr - 1, cc))
                if cc < tc:
                    options.append((rr, cc + 1))
                elif cc > tc:
                    options.append((rr, cc - 1))
                rr, cc = options[int(rng.integers(len(options)))]
                mask[rr, cc] = True
                filled.add((rr, cc))

        max_cells = 9 if side == 4 else 6
        if int(mask.sum()) > max_cells:
            continue

        target = int(rng.integers(int(mask.sum()), max_cells + 1))
        while int(mask.sum()) < target:
            frontier = []
            for rr, cc in np.argwhere(mask):
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        if bbox(mask) != (0, 0, side - 1, side - 1):
            continue
        if _has_rotational_symmetry(mask.astype(int)):
            continue
        return mask

    raise RuntimeError("failed to build a valid motif")


def _rectangles_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_patches(
    rng: np.random.Generator, grid_shape: tuple[int, int], patches: list[np.ndarray]
) -> np.ndarray | None:
    height, width = map(int, grid_shape)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    order = list(range(len(patches)))
    rng.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        candidates = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                candidate = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_rectangles_separate(candidate, other, gap=1) for other in boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        region = patch != BACKGROUND
        grid[r0 : r0 + ph, c0 : c0 + pw][region] = patch[region]
        boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return grid


def _make_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    motif_side = int(rng.integers(2, 5))
    motif_mask = _random_motif(rng, motif_side)
    fg = int(rng.integers(1, 10))
    motif = np.where(motif_mask, fg, BACKGROUND).astype(int)

    phase = int(rng.integers(4))
    missing_scale = int(rng.integers(1, 5))
    present_scales = [scale for scale in SCALES if scale != missing_scale]

    patches = []
    for scale in present_scales:
        oriented = rotate_cw(motif, phase + scale - 1)
        patches.append(scale_sprite(oriented, scale))

    for _ in range(80):
        shape = (
            int(rng.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1)),
            int(rng.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1)),
        )
        grid = _place_patches(rng, shape, patches)
        if grid is None:
            continue
        if np.count_nonzero(grid == BACKGROUND) <= np.count_nonzero(grid != BACKGROUND):
            continue
        expected = scale_sprite(rotate_cw(motif, phase + missing_scale - 1), missing_scale)
        return grid, expected.astype(int)

    raise RuntimeError("failed to place sprites")


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    decoded = _decode(grid)
    actual = decoded["output"]
    assert np.array_equal(actual, expected), "main() must match the spec-built output"

    ref_scale = min(decoded["by_scale"])
    ref_patch = decoded["by_scale"][ref_scale]
    missing_scale = decoded["missing_scale"]
    wrong_no_rotation = scale_sprite(ref_patch, missing_scale)
    assert not np.array_equal(wrong_no_rotation, expected), "rotation step must matter"

    for scale, norm in decoded["by_scale"].items():
        assert np.array_equal(norm, rotate_cw(ref_patch, scale - ref_scale))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            grid, expected = _make_example(rng)
            _validate_generated(grid, expected)
            return grid.astype(int)
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid input")
