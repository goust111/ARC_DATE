from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, scale_sprite


GRID_MIN = 14
GRID_MAX_EXCLUSIVE = 23
SMALL_SIDE_CHOICES = (3, 4, 5)
DISTRACTOR_SIDE_CHOICES = (3, 4, 5, 6)
TARGET_RARE_COUNT_BOUNDS = (1, 5)
GEN_RETRY_LIMIT = 600


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(coords.shape[0])


def _touches_all_borders(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_connected_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    area = height * width
    min_target = max(4, int(np.ceil(area * 0.45)))
    max_target = max(min_target, min(area - 1, int(np.floor(area * 0.8))))

    for _ in range(80):
        target = int(rng.integers(min_target, max_target + 1))
        mask = np.ones((height, width), dtype=bool)
        cells = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target:
                break
            test = mask.copy()
            test[r, c] = False
            if not _touches_all_borders(test):
                continue
            if not _mask_connected(test):
                continue
            mask = test

        if (
            int(mask.sum()) == target
            and _touches_all_borders(mask)
            and _mask_connected(mask)
            and not np.all(mask)
        ):
            return mask

    raise RuntimeError("failed to sample a connected mask")


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), Color.BLACK).astype(int)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = find_components(
        g,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
    )
    objects: list[dict] = []
    for component in components:
        color = int(component["color"])
        if color == Color.BLACK:
            continue
        mask = crop_bbox(g == color, component["bbox"]).astype(bool)
        objects.append(
            {
                "color": color,
                "bbox": tuple(map(int, component["bbox"])),
                "mask": mask,
                "area": int(component["area"]),
            }
        )
    return sorted(objects, key=lambda obj: obj["color"])


def _match_offsets(small_mask: np.ndarray, big_mask: np.ndarray) -> list[tuple[int, int]]:
    scaled = scale_sprite(small_mask.astype(int), 2).astype(bool)
    big_h, big_w = map(int, big_mask.shape)
    scaled_h, scaled_w = map(int, scaled.shape)
    if big_h > scaled_h or big_w > scaled_w:
        return []

    matches: list[tuple[int, int]] = []
    for dr in range(scaled_h - big_h + 1):
        for dc in range(scaled_w - big_w + 1):
            if np.array_equal(scaled[dr : dr + big_h, dc : dc + big_w], big_mask):
                matches.append((dr, dc))
    return matches


def _find_scaled_pair(objects: list[dict]) -> tuple[dict, dict, tuple[int, int]]:
    candidates: list[tuple[dict, dict, list[tuple[int, int]]]] = []
    for i, small in enumerate(objects):
        for j, big in enumerate(objects):
            if i == j:
                continue
            small_h, small_w = map(int, small["mask"].shape)
            big_h, big_w = map(int, big["mask"].shape)
            if big_h < small_h or big_w < small_w:
                continue
            if int(big["area"]) <= int(small["area"]):
                continue
            offsets = _match_offsets(small["mask"], big["mask"])
            if offsets:
                candidates.append((small, big, offsets))

    assert len(candidates) == 1, "expected exactly one crop-matching 2x pair"
    small, big, offsets = candidates[0]
    assert len(offsets) == 1, "expected a unique crop offset for the matched pair"
    return small, big, offsets[0]


def _xor_semantics(
    small: dict,
    big: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    scaled = scale_sprite(small["mask"].astype(int), 2).astype(bool)
    big_mask = big["mask"]
    big_h, big_w = map(int, big_mask.shape)
    assert big_h <= scaled.shape[0] and big_w <= scaled.shape[1]

    aligned_small = scaled[:big_h, :big_w]
    big_only = big_mask & ~aligned_small
    small_only = aligned_small & ~big_mask

    big_count = int(big_only.sum())
    small_count = int(small_only.sum())
    assert big_count > 0 and small_count > 0, "xor must contain both colors"
    assert big_count != small_count, "xor colors must have different frequencies"

    if big_count < small_count:
        return aligned_small, big_only, small_only, int(big["color"])
    return aligned_small, big_only, small_only, int(small["color"])


def _render_framed_output(
    shape: tuple[int, int],
    centers: np.ndarray,
    center_color: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    height, width = map(int, shape)
    for r, c in centers:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width:
                    out[nr, nc] = Color.RED
    for r, c in centers:
        out[int(r), int(c)] = int(center_color)
    return out


def _expected_output_from_scene(
    shape: tuple[int, int],
    big_row: int,
    big_col: int,
    small_mask: np.ndarray,
    big_mask: np.ndarray,
    big_color: int,
    small_color: int,
) -> tuple[np.ndarray, int, int]:
    scaled = scale_sprite(small_mask.astype(int), 2).astype(bool)
    big_h, big_w = map(int, big_mask.shape)
    aligned_small = scaled[:big_h, :big_w]
    big_only = big_mask & ~aligned_small
    small_only = aligned_small & ~big_mask
    big_count = int(big_only.sum())
    small_count = int(small_only.sum())

    assert big_count > 0 and small_count > 0 and big_count != small_count
    if big_count < small_count:
        rare_mask = big_only
        rare_color = int(big_color)
    else:
        rare_mask = small_only
        rare_color = int(small_color)

    centers = np.argwhere(rare_mask) + np.array([[int(big_row), int(big_col)]], dtype=int)
    return _render_framed_output(shape, centers, rare_color), int(rare_mask.sum()), rare_color


def _candidate_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    rng: np.random.Generator,
    forced_row: int | None = None,
    forced_col: int | None = None,
    padding: int = 1,
) -> list[tuple[int, int]]:
    height, width = map(int, canvas.shape)
    patch_h, patch_w = map(int, patch.shape)
    rows = [forced_row] if forced_row is not None else list(range(height - patch_h + 1))
    cols = [forced_col] if forced_col is not None else list(range(width - patch_w + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)

    foreground = np.argwhere(patch != Color.BLACK)
    occupied = canvas != Color.BLACK
    positions: list[tuple[int, int]] = []
    for row in rows:
        if row is None or row < 0 or row + patch_h > height:
            continue
        for col in cols:
            if col is None or col < 0 or col + patch_w > width:
                continue
            ok = True
            for pr, pc in foreground:
                rr = int(row + pr)
                cc = int(col + pc)
                for dr in range(-padding, padding + 1):
                    for dc in range(-padding, padding + 1):
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < height and 0 <= nc < width and bool(occupied[nr, nc]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                positions.append((int(row), int(col)))
    return positions


def _scene_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    width = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    grid = np.zeros((height, width), dtype=int)

    small_h = int(rng.choice(SMALL_SIDE_CHOICES))
    small_w = int(rng.choice(SMALL_SIDE_CHOICES))
    small_mask = _random_connected_mask(small_h, small_w, rng)
    scaled_full = scale_sprite(small_mask.astype(int), 2).astype(bool)

    crop_mode = int(rng.integers(3))
    crop_r = 1 if crop_mode in (0, 2) else 0
    crop_c = 1 if crop_mode in (1, 2) else 0
    big_mask = scaled_full[crop_r:, crop_c:]
    big_h, big_w = map(int, big_mask.shape)
    if big_h < 5 or big_w < 5 or not _mask_connected(big_mask):
        return None

    big_patch = _make_patch(big_mask, Color.BLUE)
    big_row = 0 if crop_r else int(rng.integers(0, height - big_h + 1))
    big_col = 0 if crop_c else int(rng.integers(0, width - big_w + 1))
    paste(grid, big_patch, big_row, big_col, transparent=Color.BLACK)

    small_patch = _make_patch(small_mask, Color.GREEN)
    small_positions = _candidate_positions(grid, small_patch, rng=rng, padding=1)
    if not small_positions:
        return None
    small_row, small_col = small_positions[int(rng.integers(len(small_positions)))]
    paste(grid, small_patch, small_row, small_col, transparent=Color.BLACK)

    distractor_h = int(rng.choice(DISTRACTOR_SIDE_CHOICES))
    distractor_w = int(rng.choice(DISTRACTOR_SIDE_CHOICES))
    distractor_mask = _random_connected_mask(distractor_h, distractor_w, rng)
    distractor_patch = _make_patch(distractor_mask, Color.ORANGE)
    distractor_positions = _candidate_positions(grid, distractor_patch, rng=rng, padding=1)
    if not distractor_positions:
        return None
    distractor_row, distractor_col = distractor_positions[int(rng.integers(len(distractor_positions)))]
    paste(grid, distractor_patch, distractor_row, distractor_col, transparent=Color.BLACK)

    try:
        expected, rare_count, rare_color = _expected_output_from_scene(
            grid.shape,
            big_row,
            big_col,
            small_mask,
            big_mask,
            Color.BLUE,
            Color.GREEN,
        )
    except AssertionError:
        return None
    if not (TARGET_RARE_COUNT_BOUNDS[0] <= rare_count <= TARGET_RARE_COUNT_BOUNDS[1]):
        return None
    if int(np.count_nonzero(expected == Color.RED)) > 36:
        return None
    if rare_color not in (Color.BLUE, Color.GREEN):
        return None

    objects = _extract_objects(grid)
    if len(objects) != 3:
        return None
    try:
        small_obj, big_obj, offset = _find_scaled_pair(objects)
        aligned_small, big_only, small_only, computed_rare_color = _xor_semantics(small_obj, big_obj)
    except AssertionError:
        return None

    if small_obj["color"] != Color.GREEN or big_obj["color"] != Color.BLUE:
        return None
    if offset != (crop_r, crop_c):
        return None
    if computed_rare_color != rare_color:
        return None
    if int((big_only if rare_color == Color.BLUE else small_only).sum()) != rare_count:
        return None
    if np.array_equal(distractor_mask.astype(int), small_mask.astype(int)):
        return None
    if np.array_equal(distractor_mask.astype(int), big_mask.astype(int)):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the crop-matched scale pair, XOR it under fixed top-left alignment, and frame the rarer XOR color.

    Input structure:
    - The background is black.
    - Exactly three 4-connected monochrome foreground objects are present.
    - One larger object is a visible crop of another object's 2x enlargement, while the third object is a distractor of a different shape.

    Rule:
    - Separate the three objects and find the unique ordered pair where the larger mask matches a crop inside the smaller mask enlarged by factor 2.
    - Enlarge the smaller object by 2, align that enlarged mask to the larger object's bbox top-left, and within the larger bbox emit the larger color where only the larger mask is filled and the smaller color where only the enlarged mask is filled.
    - Count those two XOR colors, keep only the lower-frequency color's cells, and on a black output grid draw a red 3x3 frame around each kept cell while restoring each frame center to its XOR color.

    Key insight:
    - Scale matching identifies which two objects belong together, but the output comes from comparing the visible large crop against the enlarged small shape under a fixed alignment and then using XOR frequency to decide which difference pixels to mark.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    objects = _extract_objects(g)
    assert len(objects) == 3, "expected exactly three foreground objects"
    small, big, _offset = _find_scaled_pair(objects)
    _aligned_small, big_only, small_only, rare_color = _xor_semantics(small, big)

    if rare_color == int(big["color"]):
        rare_mask = big_only
    else:
        rare_mask = small_only

    r0, c0, _r1, _c1 = big["bbox"]
    centers = np.argwhere(rare_mask) + np.array([[int(r0), int(c0)]], dtype=int)
    return _render_framed_output(g.shape, centers, rare_color)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _scene_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
