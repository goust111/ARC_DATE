from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SOURCE_FILL = 3
MARKER = 7
PALETTE_COLORS = (1, 2, 4, 5, 6, 8, 9)

GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 24
MATCH_COUNT_CHOICES = (2, 3, 4)
SOURCE_SHAPES = ((3, 4), (3, 5), (4, 5))
MARKER_COUNT_CHOICES = (2, 3, 4)
PALETTE_AREAS = (2, 3, 4, 5, 6)
GENERATION_ATTEMPTS = 500

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else [d for d in DIRS8 if d != (0, 0)]
    comps: list[np.ndarray] = []
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
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _transform_array(arr: np.ndarray, flip: bool, rot: int) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr.copy()
    return np.rot90(out, rot)


def _unique_transforms(arr: np.ndarray) -> list[dict]:
    unique = []
    seen = set()
    for flip in (False, True):
        for rot in range(4):
            transformed = _transform_array(arr, flip, rot)
            key = (transformed.shape, transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "flip": flip,
                    "rot": rot,
                    "patch": transformed.copy(),
                }
            )
    return unique


def _find_source(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = (grid == SOURCE_FILL) | (grid == MARKER)
    candidates = []
    for coords in _components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((crop == SOURCE_FILL) | (crop == MARKER)):
            continue
        if not np.any(crop == SOURCE_FILL) or not np.any(crop == MARKER):
            continue
        if crop.size != int(coords.shape[0]):
            continue
        candidates.append(((r0, c0, r1, c1), crop.copy()))
    assert len(candidates) == 1, "expected exactly one rectangular source sprite"
    return candidates[0]


def _find_matches(grid: np.ndarray, source_patch: np.ndarray) -> list[dict]:
    h_grid, w_grid = grid.shape
    matches = {}
    for spec in _unique_transforms(source_patch):
        patch = spec["patch"]
        marker_mask = patch == MARKER
        h, w = patch.shape
        for r0 in range(h_grid - h + 1):
            for c0 in range(w_grid - w + 1):
                region = grid[r0 : r0 + h, c0 : c0 + w]
                if np.all(region[marker_mask] == MARKER) and np.all(region[~marker_mask] == BG):
                    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                    existing = matches.get(bbox)
                    if existing is None:
                        matches[bbox] = {"bbox": bbox, "patch": patch.copy()}
                    else:
                        assert np.array_equal(existing["patch"], patch)
    return sorted(matches.values(), key=lambda item: (item["bbox"][0], item["bbox"][1]))


def _palette_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for color in sorted(set(int(v) for v in grid.flatten()) - {BG, SOURCE_FILL, MARKER}):
        for coords in _components_mask(grid == color, connectivity=4):
            bbox = _bbox_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each transformed stamp slot with the source sprite, but color its fill from the area-ranked external palette blobs.

    Input structure:
    - A black background contains one full rectangular source sprite made from green fill cells and orange marker cells.
    - Elsewhere, several orange-only groups reproduce the source marker mask under rotations or reflections while the rest of each transformed rectangle stays black.
    - The grid also contains the same number of disconnected monochrome palette blobs in colors other than black, green, or orange.

    Rule:
    - Identify the unique rectangular source sprite whose bbox is completely filled by green or orange.
    - Enumerate all rotation and reflection transforms of that sprite and find every rectangle where the transformed orange mask matches exactly and every non-marker cell is black.
    - Rank the palette blobs by increasing area, breaking ties by their topmost row and then leftmost column, and rank the match rectangles by their top-left coordinates.
    - Pair the kth match with the kth palette blob, stamp the transformed full sprite there, keep orange markers unchanged, replace green fill with the blob's color, and erase all palette blobs.

    Key insight:
    The detached small blobs act as a hidden color palette: their area ranking determines the stamp colors even though the blobs themselves disappear in the output.
    """
    g = np.asarray(grid, dtype=int)
    _source_bbox, source_patch = _find_source(g)
    matches = _find_matches(g, source_patch)
    palette = _palette_components(g)

    assert matches, "expected at least one transformed marker match"
    assert len(matches) == len(palette), "match count must equal palette-blob count"

    ranked_palette = sorted(
        palette,
        key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]),
    )
    out = g.copy()
    for comp in ranked_palette:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    for match, comp in zip(matches, ranked_palette):
        patch = match["patch"].copy()
        patch[patch == SOURCE_FILL] = comp["color"]
        r0, c0, _r1, _c1 = match["bbox"]
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


def _random_source_patch() -> np.ndarray:
    h, w = random.choice(SOURCE_SHAPES)
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(200):
        marker_count = random.choice(MARKER_COUNT_CHOICES)
        markers = random.sample(cells, marker_count)
        rows = {r for r, _ in markers}
        cols = {c for _, c in markers}
        if len(rows) < 2 or len(cols) < 2:
            continue
        patch = np.full((h, w), SOURCE_FILL, dtype=int)
        for r, c in markers:
            patch[r, c] = MARKER
        if len(_unique_transforms(patch == MARKER)) == 8:
            return patch
    raise RuntimeError("failed to create a non-symmetric source marker mask")


def _connected_shape(area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier -= cells
            candidates = []
            for cell in frontier:
                trial = cells | {cell}
                rs = [r for r, _ in trial]
                cs = [c for _, c in trial]
                if max(rs) - min(rs) <= 3 and max(cs) - min(cs) <= 3:
                    candidates.append(cell)
            if not candidates:
                break
            cells.add(random.choice(candidates))
        if len(cells) != area:
            continue
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        mask = np.zeros((max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        transforms = _unique_transforms(mask.astype(int))
        return random.choice(transforms)["patch"].astype(bool)
    return None


def _region_is_clear(grid: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h_grid, w_grid = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > h_grid or c0 + w > w_grid:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_grid, r0 + h + pad)
    cc1 = min(w_grid, c0 + w + pad)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _shape_is_clear(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    coords = np.argwhere(shape)
    h_grid, w_grid = grid.shape
    for rr, cc in coords:
        r = int(r0 + rr)
        c = int(c0 + cc)
        if not (0 <= r < h_grid and 0 <= c < w_grid):
            return False
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h_grid and 0 <= nc < w_grid and grid[nr, nc] != BG:
                    return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _place_marker_only_match(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    marker_mask = patch == MARKER
    rr, cc = np.where(marker_mask)
    grid[r0 + rr, c0 + cc] = MARKER


def _place_shape(grid: np.ndarray, shape: np.ndarray, color: int, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(shape)
    placed = coords + np.array([[r0, c0]])
    grid[placed[:, 0], placed[:, 1]] = color
    return placed


def _build_expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _generate_case() -> np.ndarray | None:
    h_grid = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w_grid = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    inp = np.zeros((h_grid, w_grid), dtype=int)

    source_patch = _random_source_patch()
    transforms = _unique_transforms(source_patch)
    plain = [spec for spec in transforms if not spec["flip"]]
    mirrored = [spec for spec in transforms if spec["flip"]]
    match_count = random.choice(MATCH_COUNT_CHOICES)

    chosen_specs: list[dict] = []
    if match_count >= 2:
        chosen_specs.append(random.choice(plain))
        chosen_specs.append(random.choice(mirrored))
    pool = [spec for spec in transforms if spec not in chosen_specs]
    random.shuffle(pool)
    while len(chosen_specs) < match_count:
        chosen_specs.append(pool.pop())

    source_h, source_w = source_patch.shape
    source_pos = None
    for _ in range(100):
        r0 = random.randint(0, h_grid - source_h)
        c0 = random.randint(0, w_grid - source_w)
        if _region_is_clear(inp, r0, c0, source_h, source_w, pad=1):
            source_pos = (r0, c0)
            _place_patch(inp, source_patch, r0, c0)
            break
    if source_pos is None:
        return None

    placed_matches = []
    random.shuffle(chosen_specs)
    for spec in chosen_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(h_grid - ph + 1)
            for c0 in range(w_grid - pw + 1)
            if _region_is_clear(inp, r0, c0, ph, pw, pad=1)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _place_marker_only_match(inp, patch, r0, c0)
        placed_matches.append({"bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1), "patch": patch.copy()})

    palette_areas = random.sample(PALETTE_AREAS, match_count)
    palette_colors = random.sample(PALETTE_COLORS, match_count)
    placed_palette = []
    for area, color in zip(palette_areas, palette_colors):
        shape = _connected_shape(area)
        if shape is None:
            return None
        sh, sw = shape.shape
        positions = [
            (r0, c0)
            for r0 in range(h_grid - sh + 1)
            for c0 in range(w_grid - sw + 1)
            if _shape_is_clear(inp, shape, r0, c0, pad=1)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        coords = _place_shape(inp, shape, color, r0, c0)
        placed_palette.append(
            {
                "color": color,
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )

    expected = inp.copy()
    for comp in placed_palette:
        expected[comp["coords"][:, 0], comp["coords"][:, 1]] = BG
    ranked_matches = sorted(placed_matches, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    ranked_palette = sorted(placed_palette, key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]))
    for match, comp in zip(ranked_matches, ranked_palette):
        patch = match["patch"].copy()
        patch[patch == SOURCE_FILL] = comp["color"]
        r0, c0, _r1, _c1 = match["bbox"]
        expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    try:
        found_source_bbox, found_source_patch = _find_source(inp)
        found_match_bboxes = [item["bbox"] for item in _find_matches(inp, found_source_patch)]
        found_palette = _palette_components(inp)
    except AssertionError:
        return None

    if found_source_bbox != (source_pos[0], source_pos[1], source_pos[0] + source_h - 1, source_pos[1] + source_w - 1):
        return None
    if found_match_bboxes != [item["bbox"] for item in ranked_matches]:
        return None
    if len(found_palette) != match_count:
        return None
    if len({item["area"] for item in found_palette}) != match_count:
        return None
    if not np.array_equal(_build_expected_output(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
