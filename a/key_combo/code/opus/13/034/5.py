import random
from collections import deque

import numpy as np


BG = 0
FILL = 1
MARK = 2

GRID_MIN = 17
GRID_MAX = 23
SOURCE_SIZES = (5, 6)
MATCH_COUNTS = (1, 2)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
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
                if connectivity == 4:
                    nbrs = _neighbors4(rr, cc)
                else:
                    nbrs = (
                        (rr + dr, cc + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if dr != 0 or dc != 0
                    )
                for nr, nc in nbrs:
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components_mask(mask, connectivity=4)) == 1


def _swap_colors(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    out[patch == FILL] = MARK
    out[patch == MARK] = FILL
    return out


def _apply_transform(arr: np.ndarray, mirror: bool, rot: int) -> np.ndarray:
    transformed = np.fliplr(arr) if mirror else arr.copy()
    return np.rot90(transformed, rot)


def _unique_transforms(arr: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    for mirror in (False, True):
        for rot in range(4):
            transformed = _apply_transform(arr, mirror, rot)
            if any(np.array_equal(transformed, item["grid"]) for item in variants):
                continue
            variants.append(
                {
                    "grid": transformed,
                    "mirror": mirror,
                    "rot": rot,
                }
            )
    return variants


def _find_source(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    blue_mask = grid == FILL
    candidates = []
    for coords in _components_mask(blue_mask, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape[0] not in SOURCE_SIZES or patch.shape[1] != patch.shape[0]:
            continue
        if not np.all((patch == FILL) | (patch == MARK)):
            continue
        if np.count_nonzero(patch == FILL) != len(coords):
            continue
        candidates.append(((r0, c0, r1, c1), patch.copy()))
    assert len(candidates) == 1, "expected a unique filled source rectangle"
    return candidates[0]


def _all_exact_matches(
    grid: np.ndarray,
    variants: list[dict],
    source_bbox: tuple[int, int, int, int],
) -> list[dict]:
    h_grid, w_grid = grid.shape
    matches = []
    for idx, variant in enumerate(variants):
        mask = variant["grid"]
        h, w = mask.shape
        for r0 in range(h_grid - h + 1):
            for c0 in range(w_grid - w + 1):
                r1, c1 = r0 + h - 1, c0 + w - 1
                if (r0, c0, r1, c1) == source_bbox:
                    continue
                patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                if np.array_equal(patch == MARK, mask) and np.all(patch[~mask] == BG):
                    matches.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "variant_idx": idx,
                            "mirror": variant["mirror"],
                            "rot": variant["rot"],
                        }
                    )
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find transformed copies of the source marker mask, then stamp the source rectangle there with fill and marker colors swapped.

    Input structure:
    - The background is black.
    - One blue rectangular source sprite is fully filled by blue cells plus several red marker cells inside the same bbox.
    - Elsewhere, groups of red single pixels reproduce transformed copies of the source marker mask while the rest of each candidate bbox stays black.

    Rule:
    - Identify the unique blue 4-connected component whose bbox is completely occupied by blue and red cells; this is the source rectangle.
    - Extract the source rectangle's red marker mask and enumerate its unique rotations and mirror+rotation transforms.
    - Wherever one transformed mask exactly matches red pixels and every other cell in that full transformed bbox is black, stamp a transformed copy of the source rectangle after swapping blue to red and red to blue.
    - Keep the original source rectangle unchanged and ignore any candidate whose transformed bbox would not fit on the grid.

    Key insight:
    - The scattered red pixels determine both the stamp location and its orientation, but the stamped content is not the original sprite: it is the same rectangle after swapping the fill and marker colors.
    """
    g = np.asarray(grid, dtype=int)
    source_bbox, source_patch = _find_source(g)
    marker_mask = source_patch == MARK
    variants = _unique_transforms(marker_mask)
    swapped = _swap_colors(source_patch)

    out = g.copy()
    for match in _all_exact_matches(g, variants, source_bbox):
        stamp = _apply_transform(swapped, match["mirror"], match["rot"])
        r0, c0, r1, c1 = match["bbox"]
        if 0 <= r0 and 0 <= c0 and r1 < g.shape[0] and c1 < g.shape[1]:
            out[r0 : r1 + 1, c0 : c1 + 1] = stamp
    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = PLACEMENT_GAP,
) -> bool:
    return bool(
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _random_source_patch() -> np.ndarray | None:
    size = random.choice(SOURCE_SIZES)
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(400):
        mark_count = random.randint(4, 6)
        chosen = random.sample(all_cells, mark_count)
        mask = np.zeros((size, size), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if len(_unique_transforms(mask)) != 8:
            continue
        if not _connected4(~mask):
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 + 1 < size - 1 or c1 - c0 + 1 < size - 1:
            continue
        patch = np.full((size, size), FILL, dtype=int)
        patch[mask] = MARK
        return patch
    return None


def _candidate_boxes(
    grid_shape: tuple[int, int],
    shape: tuple[int, int],
    blocked: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    h_grid, w_grid = grid_shape
    h, w = shape
    options = []
    for r0 in range(h_grid - h + 1):
        for c0 in range(w_grid - w + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_separated(box, other) for other in blocked):
                options.append(box)
    random.shuffle(options)
    return options


def _place_mask_cells(grid: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray) -> None:
    r0, c0, _r1, _c1 = box
    rows, cols = np.nonzero(mask)
    for rr, cc in zip(rows, cols):
        grid[r0 + rr, c0 + cc] = MARK


def _add_partial_distractor(
    grid: np.ndarray,
    blocked: list[tuple[int, int, int, int]],
    variants: list[dict],
) -> bool:
    order = list(range(len(variants)))
    random.shuffle(order)
    for idx in order:
        mask = variants[idx]["grid"]
        if int(mask.sum()) <= 1:
            continue
        options = _candidate_boxes(grid.shape, mask.shape, blocked)
        if not options:
            continue
        for box in options[:12]:
            coords = list(zip(*np.nonzero(mask)))
            random.shuffle(coords)
            keep = coords[:-1]
            if not keep:
                continue
            r0, c0, _r1, _c1 = box
            for rr, cc in keep:
                grid[r0 + rr, c0 + cc] = MARK
            blocked.append(box)
            return True
    return False


def _add_extra_red_distractor(
    grid: np.ndarray,
    blocked: list[tuple[int, int, int, int]],
    variants: list[dict],
) -> bool:
    order = list(range(len(variants)))
    random.shuffle(order)
    for idx in order:
        mask = variants[idx]["grid"]
        zeros = list(zip(*np.nonzero(~mask)))
        if not zeros:
            continue
        options = _candidate_boxes(grid.shape, mask.shape, blocked)
        if not options:
            continue
        for box in options[:12]:
            _place_mask_cells(grid, box, mask)
            r0, c0, _r1, _c1 = box
            rr, cc = random.choice(zeros)
            grid[r0 + rr, c0 + cc] = MARK
            blocked.append(box)
            return True
    return False


def _render_expected_output(
    inp: np.ndarray,
    source_patch: np.ndarray,
    matches: list[dict],
) -> np.ndarray:
    out = inp.copy()
    swapped = _swap_colors(source_patch)
    for match in matches:
        stamp = _apply_transform(swapped, match["mirror"], match["rot"])
        r0, c0, r1, c1 = match["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = stamp
    return out


def _match_signature(match: dict) -> tuple[tuple[int, int, int, int], int]:
    return match["bbox"], match["variant_idx"]


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    source_patch = _random_source_patch()
    if source_patch is None:
        return None
    source_mask = source_patch == MARK
    variants = _unique_transforms(source_mask)
    if len(variants) != 8:
        return None

    source_box_options = _candidate_boxes(grid.shape, source_patch.shape, [])
    if not source_box_options:
        return None
    source_bbox = random.choice(source_box_options)
    sr0, sc0, sr1, sc1 = source_bbox
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = source_patch

    blocked = [source_bbox]
    match_count = random.choice(MATCH_COUNTS)
    chosen_matches: list[dict] = []
    variant_indices = list(range(len(variants)))
    random.shuffle(variant_indices)
    if match_count == 2 and len(variant_indices) >= 2:
        requested = variant_indices[:2]
    else:
        requested = [variant_indices[0]]

    for idx in requested:
        mask = variants[idx]["grid"]
        options = _candidate_boxes(grid.shape, mask.shape, blocked)
        if not options:
            return None
        box = random.choice(options)
        _place_mask_cells(grid, box, mask)
        blocked.append(box)
        chosen_matches.append(
            {
                "bbox": box,
                "variant_idx": idx,
                "mirror": variants[idx]["mirror"],
                "rot": variants[idx]["rot"],
            }
        )

    distractor_count = random.randint(1, 3)
    distractor_added = 0
    for _ in range(distractor_count):
        if random.random() < 0.5:
            distractor_added += int(_add_partial_distractor(grid, blocked, variants))
        else:
            distractor_added += int(_add_extra_red_distractor(grid, blocked, variants))

    if distractor_added == 0:
        return None

    observed_source_bbox, observed_source_patch = _find_source(grid)
    if observed_source_bbox != source_bbox or not np.array_equal(observed_source_patch, source_patch):
        return None

    observed_matches = _all_exact_matches(grid, variants, source_bbox)
    if sorted(_match_signature(m) for m in observed_matches) != sorted(
        _match_signature(m) for m in chosen_matches
    ):
        return None

    expected = _render_expected_output(grid, source_patch, chosen_matches)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    for match in chosen_matches:
        r0, c0, r1, c1 = match["bbox"]
        before = grid[r0 : r1 + 1, c0 : c1 + 1]
        after = expected[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(before == MARK) < 4:
            return None
        if np.count_nonzero(after == FILL) == 0 or np.count_nonzero(after == MARK) == 0:
            return None
        if np.array_equal(before, after):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
