from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "arc_utils_057_5",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste

TRANSFORMS: tuple[tuple[int, bool], ...] = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)
IDENTITY_TRANSFORM = (0, False)
MAX_GENERATION_ATTEMPTS = 500
MISSING_RETRIES = 80
MIN_MOTIF_SIDE = 7
MAX_MOTIF_SIDE = 14
PALETTE_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
EDGE_NAMES = ("top", "bottom", "left", "right")


def apply_transform(arr: np.ndarray, transform: tuple[int, bool]) -> np.ndarray:
    rot, mirror = transform
    out = np.asarray(arr)
    if mirror:
        out = np.fliplr(out)
    return np.rot90(out, int(rot))


def inverse_transform(arr: np.ndarray, transform: tuple[int, bool]) -> np.ndarray:
    rot, mirror = transform
    out = np.rot90(np.asarray(arr), -int(rot))
    if mirror:
        out = np.fliplr(out)
    return out


def crop_bool(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None, "expected a non-empty mask"
    return crop_bbox(mask.astype(bool), box).astype(bool)


def count_components(mask: np.ndarray) -> int:
    return len(find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=False))


def ordered_alignment_matches(
    complete_crop: np.ndarray,
    damaged_crop: np.ndarray,
    bg: int,
    transforms: tuple[tuple[int, bool], ...] = TRANSFORMS,
) -> list[dict]:
    damaged_fg = damaged_crop != bg
    if not damaged_fg.any():
        return []

    matches: list[dict] = []
    for transform in transforms:
        oriented_complete = apply_transform(complete_crop, transform)
        complete_fg = oriented_complete != bg
        h_full, w_full = oriented_complete.shape
        h_dam, w_dam = damaged_crop.shape
        if h_dam > h_full or w_dam > w_full:
            continue

        for r0 in range(h_full - h_dam + 1):
            for c0 in range(w_full - w_dam + 1):
                window = oriented_complete[r0 : r0 + h_dam, c0 : c0 + w_dam]
                window_fg = complete_fg[r0 : r0 + h_dam, c0 : c0 + w_dam]
                if np.any(damaged_fg & ~window_fg):
                    continue
                if np.any(window[damaged_fg] != damaged_crop[damaged_fg]):
                    continue

                missing = complete_fg.copy()
                missing[r0 : r0 + h_dam, c0 : c0 + w_dam][damaged_fg] = False
                if not missing.any():
                    continue

                matches.append(
                    {
                        "transform": transform,
                        "offset": (r0, c0),
                        "oriented_complete": oriented_complete,
                        "missing": missing,
                    }
                )
    return matches


def all_input_matches(grid: np.ndarray, bg: int) -> list[dict]:
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    crops = [crop_bbox(grid, component["bbox"]) for component in components]

    matches: list[dict] = []
    for complete_idx in range(len(crops)):
        for damaged_idx in range(len(crops)):
            if complete_idx == damaged_idx:
                continue
            for match in ordered_alignment_matches(crops[complete_idx], crops[damaged_idx], bg):
                enriched = dict(match)
                enriched["complete_index"] = complete_idx
                enriched["damaged_index"] = damaged_idx
                matches.append(enriched)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Align the damaged rotated or mirrored copy back to the full motif, then output the missing cells in the copy's own pose.

    Input structure:
    - The black background contains exactly two multicolor 4-connected foreground objects.
    - One object is a complete rectangular motif; the other is a damaged copy obtained from one of the eight dihedral poses of that motif.
    - The damaged copy is cropped to its own surviving foreground, so it may be smaller than the aligned full motif even after the correct pose is found.

    Rule:
    - Consider both object orderings as candidate (complete, damaged) pairs.
    - For each ordering, try the four rotations and optional mirror on the complete motif, and slide the damaged crop inside that transformed motif.
    - Find the unique pose and offset where every damaged foreground cell matches the transformed complete motif exactly and the damaged foreground is a strict subset.
    - Mark every foreground cell that exists in the aligned complete motif but not in the damaged copy, keep that mask in the damaged copy's orientation, crop it to its own bbox, and recolor it teal on black.

    Key insight:
    - The comparison must happen in the copy's coordinate frame, not the master's: first solve a hidden pose-and-offset search, then take the difference in that aligned frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected a black background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(components) == 2, "expected exactly two foreground objects"

    matches = all_input_matches(g, bg)
    assert len(matches) == 1, "expected a unique complete/damaged alignment"

    missing = matches[0]["missing"]
    out_mask = crop_bool(missing)
    out = np.zeros(out_mask.shape, dtype=int)
    out[out_mask] = Color.TEAL
    return out


def grow_connected(
    height: int,
    width: int,
    target_count: int,
    start_cells: set[tuple[int, int]] | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    cells = set() if start_cells is None else set(start_cells)
    blocked = set() if forbidden is None else set(forbidden)
    all_cells = {(r, c) for r in range(height) for c in range(width)} - blocked
    if not all_cells:
        return None
    if not cells:
        cells.add(random.choice(tuple(all_cells)))
    if not cells <= all_cells or len(cells) > target_count:
        return None

    while len(cells) < target_count:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in all_cells and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def split_two_component_mask(height: int, width: int, target_count: int) -> np.ndarray | None:
    use_vertical = width >= 5 and (height < 5 or random.random() < 0.5)
    if use_vertical:
        split = random.randint(2, width - 3)
        left_w = split
        right_w = width - split - 1
        max_left = height * left_w
        max_right = height * right_w
        if max_left <= 0 or max_right <= 0:
            return None
        left_target = random.randint(1, target_count - 1)
        right_target = target_count - left_target
        if left_target > max_left or right_target > max_right:
            return None
        left = grow_connected(height, left_w, left_target)
        right = grow_connected(height, right_w, right_target)
        if left is None or right is None:
            return None
        mask = np.zeros((height, width), dtype=bool)
        for r, c in left:
            mask[r, c] = True
        for r, c in right:
            mask[r, split + 1 + c] = True
        return mask

    split = random.randint(2, height - 3)
    top_h = split
    bottom_h = height - split - 1
    max_top = top_h * width
    max_bottom = bottom_h * width
    if max_top <= 0 or max_bottom <= 0:
        return None
    top_target = random.randint(1, target_count - 1)
    bottom_target = target_count - top_target
    if top_target > max_top or bottom_target > max_bottom:
        return None
    top = grow_connected(top_h, width, top_target)
    bottom = grow_connected(bottom_h, width, bottom_target)
    if top is None or bottom is None:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for r, c in top:
        mask[r, c] = True
    for r, c in bottom:
        mask[split + 1 + r, c] = True
    return mask


def full_edge_sides(mask: np.ndarray) -> set[str]:
    sides = set()
    if np.all(mask[0]):
        sides.add("top")
    if np.all(mask[-1]):
        sides.add("bottom")
    if np.all(mask[:, 0]):
        sides.add("left")
    if np.all(mask[:, -1]):
        sides.add("right")
    return sides


def generate_missing_mask() -> np.ndarray | None:
    for _ in range(MISSING_RETRIES):
        height = random.randint(2, 6)
        width = random.randint(2, 8)
        max_cells = min(height * width - 1, 14)
        if max_cells < 4:
            continue

        if height >= 5 and width >= 5 and random.random() < 0.35:
            target_count = random.randint(4, max_cells)
            mask = split_two_component_mask(height, width, target_count)
            if mask is None:
                continue
            mask = crop_bool(mask)
            if 1 <= count_components(mask) <= 2:
                return mask
            continue

        base = set()
        if random.random() < 0.55:
            candidate_sides = random.sample(EDGE_NAMES, random.randint(0, 2))
            for side in candidate_sides:
                if side == "top":
                    base.update((0, c) for c in range(width))
                elif side == "bottom":
                    base.update((height - 1, c) for c in range(width))
                elif side == "left":
                    base.update((r, 0) for r in range(height))
                else:
                    base.update((r, width - 1) for r in range(height))

        min_target = max(4, len(base))
        if min_target > max_cells:
            continue
        target_count = random.randint(min_target, max_cells)
        cells = grow_connected(height, width, target_count, start_cells=base if base else None)
        if cells is None:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = crop_bool(mask)
        if 1 <= count_components(mask) <= 2:
            return mask
    return None


def make_motif(height: int, width: int, palette: list[int]) -> np.ndarray:
    n_colors = len(palette)
    row_step = random.randint(1, max(1, min(3, height - 1)))
    col_step = random.randint(1, max(1, min(3, width - 1)))
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            idx = (
                r
                + 2 * c
                + r // row_step
                + 2 * (c // col_step)
                + ((r + c) % 2)
            ) % n_colors
            grid[r, c] = palette[idx]

    row_mark = random.randint(1, height - 2)
    col_mark = random.randint(1, width - 2)
    grid[row_mark, :] = palette[0]
    grid[:, col_mark] = palette[1 % n_colors]

    block_h = 2 if height >= 4 else 1
    block_w = 2 if width >= 4 else 1
    grid[0:block_h, 0:block_w] = palette[2 % n_colors]
    grid[height - block_h : height, width - block_w : width] = palette[3 % n_colors]

    diag_color = palette[4 % n_colors]
    for r in range(height):
        c = (2 * r + 1) % width
        grid[r, c] = diag_color

    if n_colors >= 6:
        anti_color = palette[5]
        for r in range(height):
            c = (width - 1 - (3 * r % width))
            grid[r, c] = anti_color
    return grid


def choose_margins(missing_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    missing_count = int(missing_mask.sum())
    edge_full = list(full_edge_sides(missing_mask))
    random.shuffle(edge_full)
    shrink_count = random.randint(0, min(2, len(edge_full)))
    shrink_sides = set(edge_full[:shrink_count])

    margins = {}
    for side in EDGE_NAMES:
        margins[side] = 0 if side in shrink_sides else random.randint(1, 3)

    height = margins["top"] + missing_mask.shape[0] + margins["bottom"]
    width = margins["left"] + missing_mask.shape[1] + margins["right"]
    if not (MIN_MOTIF_SIDE <= height <= MAX_MOTIF_SIDE and MIN_MOTIF_SIDE <= width <= MAX_MOTIF_SIDE):
        return None
    if height * width - missing_count < 10:
        return None
    return margins["top"], margins["bottom"], margins["left"], margins["right"]


def embed_missing(height: int, width: int, missing_mask: np.ndarray, margins: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = margins
    assert top + missing_mask.shape[0] + bottom == height
    assert left + missing_mask.shape[1] + right == width
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + missing_mask.shape[0], left : left + missing_mask.shape[1]] = missing_mask
    return mask


def wrong_master_frame_output(mask_copy_frame: np.ndarray, transform: tuple[int, bool]) -> np.ndarray:
    mask_master = inverse_transform(mask_copy_frame.astype(int), transform).astype(bool)
    return crop_bool(mask_master)


def place_objects(complete_crop: np.ndarray, damaged_crop: np.ndarray) -> np.ndarray | None:
    h1, w1 = complete_crop.shape
    h2, w2 = damaged_crop.shape
    layout = random.choice(("horizontal", "vertical"))
    gap = random.randint(2, 6)

    if layout == "horizontal":
        extra_top = random.randint(2, 6)
        extra_bottom = random.randint(2, 6)
        extra_left = random.randint(2, 5)
        extra_right = random.randint(2, 5)
        canvas_h = max(h1, h2) + extra_top + extra_bottom
        canvas_w = w1 + w2 + gap + extra_left + extra_right
        if canvas_h > 30 or canvas_w > 30:
            return None
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        r1 = random.randint(extra_top, canvas_h - extra_bottom - h1)
        r2 = random.randint(extra_top, canvas_h - extra_bottom - h2)
        c1 = extra_left
        c2 = canvas_w - extra_right - w2
    else:
        extra_top = random.randint(2, 5)
        extra_bottom = random.randint(2, 5)
        extra_left = random.randint(2, 6)
        extra_right = random.randint(2, 6)
        canvas_h = h1 + h2 + gap + extra_top + extra_bottom
        canvas_w = max(w1, w2) + extra_left + extra_right
        if canvas_h > 30 or canvas_w > 30:
            return None
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        r1 = extra_top
        r2 = canvas_h - extra_bottom - h2
        c1 = random.randint(extra_left, canvas_w - extra_right - w1)
        c2 = random.randint(extra_left, canvas_w - extra_right - w2)

    paste(grid, complete_crop, r1, c1)
    paste(grid, damaged_crop, r2, c2, transparent=Color.BLACK)
    return grid


def make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    missing_crop = generate_missing_mask()
    if missing_crop is None:
        return None

    margins = choose_margins(missing_crop)
    if margins is None:
        return None

    top, bottom, left, right = margins
    motif_h = top + missing_crop.shape[0] + bottom
    motif_w = left + missing_crop.shape[1] + right

    color_count = random.randint(4, 6)
    palette = random.sample(list(PALETTE_POOL), color_count)
    copy_frame_complete = make_motif(motif_h, motif_w, palette)
    full_missing = embed_missing(motif_h, motif_w, missing_crop, margins)

    damaged_full = copy_frame_complete.copy()
    damaged_full[full_missing] = Color.BLACK
    damaged_fg = damaged_full != Color.BLACK
    if not damaged_fg.any() or count_components(damaged_fg) != 1:
        return None

    damaged_crop = crop_bbox(damaged_full, bbox(damaged_fg))
    if len(set(map(int, damaged_crop[damaged_crop != Color.BLACK].reshape(-1)))) < 2:
        return None

    transform = random.choice(TRANSFORMS)
    complete_crop = inverse_transform(copy_frame_complete, transform)

    expected_mask = crop_bool(full_missing)
    expected = np.zeros(expected_mask.shape, dtype=int)
    expected[expected_mask] = Color.TEAL

    if transform != IDENTITY_TRANSFORM:
        wrong = wrong_master_frame_output(full_missing, transform)
        if np.array_equal(wrong.astype(int), expected_mask.astype(int)):
            return None

    grid = place_objects(complete_crop, damaged_crop)
    if grid is None:
        return None

    if count_components(grid != Color.BLACK) != 2:
        return None

    matches = all_input_matches(grid, Color.BLACK)
    if len(matches) != 1:
        return None
    match = matches[0]
    if match["transform"] != transform:
        return None
    if not np.array_equal(match["missing"], full_missing):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = make_attempt()
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid rotating-damage ARC input")
