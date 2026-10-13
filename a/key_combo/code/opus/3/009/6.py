from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GEN_RETRY_LIMIT = 400


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    dirs = FOUR_DIRS if int(connectivity) == 4 else EIGHT_DIRS
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _marker_pixels_for_box(top: int, left: int, bottom: int, right: int) -> set[tuple[int, int]]:
    return {
        (top - 1, left),
        (top, left - 1),
        (top - 1, right),
        (top, right + 1),
        (bottom, left - 1),
        (bottom + 1, left),
        (bottom, right + 1),
        (bottom + 1, right),
    }


def _region_branch(height: int, width: int) -> str:
    if width > height:
        return "wide"
    if height > width:
        return "tall"
    return "square"


def _rotate_patch_for_branch(patch: np.ndarray, branch: str) -> np.ndarray:
    if branch == "wide":
        return np.rot90(patch, -1)
    if branch == "tall":
        return np.rot90(patch, 1)
    assert branch == "square"
    return np.rot90(patch, 2)


def _center_region(height: int, width: int, patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    ph, pw = map(int, patch.shape)
    if ph > height or pw > width:
        raise ValueError("patch does not fit inside target region")
    r0 = (height - ph) // 2
    c0 = (width - pw) // 2
    out = np.zeros((height, width), dtype=int)
    out[r0 : r0 + ph, c0 : c0 + pw] = patch
    return out, (int(r0), int(c0))


def _place_patch(height: int, width: int, patch: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    r0, c0 = map(int, offset)
    ph, pw = map(int, patch.shape)
    out = np.zeros((height, width), dtype=int)
    out[r0 : r0 + ph, c0 : c0 + pw] = patch
    return out


def _transform_region(region: np.ndarray) -> np.ndarray:
    coords = np.argwhere(region != BG)
    assert coords.size > 0
    patch = _crop_bbox(region, _bbox_coords(coords))
    h, w = map(int, region.shape)
    branch = _region_branch(h, w)
    rotated = _rotate_patch_for_branch(patch, branch)
    centered, _ = _center_region(h, w, rotated)
    return centered


def _marker_region_for_color(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.shape[0] != 8:
        return None

    row_counts = Counter(map(int, coords[:, 0]))
    col_counts = Counter(map(int, coords[:, 1]))
    rows = sorted(row_counts)
    cols = sorted(col_counts)
    if len(rows) != 4 or len(cols) != 4:
        return None
    if set(row_counts.values()) != {2} or set(col_counts.values()) != {2}:
        return None
    if rows[1] != rows[0] + 1 or rows[3] != rows[2] + 1:
        return None
    if cols[1] != cols[0] + 1 or cols[3] != cols[2] + 1:
        return None

    comps = _components_mask(grid == int(color), connectivity=8)
    if len(comps) != 4 or any(comp.shape[0] != 2 for comp in comps):
        return None
    valid_local_shapes = [
        {(0, 0), (1, 1)},
        {(0, 1), (1, 0)},
    ]
    for comp in comps:
        r0, c0, r1, c1 = _bbox_coords(comp)
        if (r1 - r0, c1 - c0) != (1, 1):
            return None
        rel = {(int(r - r0), int(c - c0)) for r, c in comp}
        if rel not in valid_local_shapes:
            return None

    top, bottom = rows[1], rows[2]
    left, right = cols[1], cols[2]
    if bottom <= top or right <= left:
        return None
    marker_pixels = _marker_pixels_for_box(top, left, bottom, right)
    actual_pixels = {tuple(map(int, rc)) for rc in coords}
    if actual_pixels != marker_pixels:
        return None

    return {
        "color": int(color),
        "inner_box": (int(top), int(left), int(bottom), int(right)),
        "marker_pixels": marker_pixels,
        "height": int(bottom - top + 1),
        "width": int(right - left + 1),
    }


def _find_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for color in _colors_present(grid, exclude={BG}):
        spec = _marker_region_for_color(grid, color)
        if spec is not None:
            regions.append(spec)
    regions.sort(key=lambda spec: (spec["inner_box"][0], spec["inner_box"][1], spec["color"]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each region's internal pattern by an angle encoded by the corner-marked rectangle's aspect ratio, then erase the corner marks.

    Input structure:
    - A black background contains several disjoint rectangular regions indicated only by four same-color 2-pixel corner marks.
    - Each marker color appears in exactly four diagonal L-corner pairs around one region, leaving the region interior itself black except for one multicolor connected pattern.
    - Different regions have different interior aspect ratios: wide, tall, and square.

    Rule:
    - For each marker color, recover the rectangle interior bounded by its four corner pairs.
    - Tight-crop the non-black pattern inside that interior and choose the rotation from the interior width/height: wide -> 90 degrees clockwise, tall -> 90 degrees counterclockwise, square -> 180 degrees.
    - Place the rotated tight crop back into the center of the same interior, biased to the upper-left when exact centering is impossible, and erase all corner-marker pixels to black.

    Key insight:
    - The corner markers do two jobs at once: they define the invisible rectangular workspace and also encode the rotation angle through that workspace's aspect ratio.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _find_regions(g)
    assert len(regions) >= 1

    for region in regions:
        for r, c in region["marker_pixels"]:
            out[int(r), int(c)] = BG

    seen_boxes: set[tuple[int, int, int, int]] = set()
    for region in regions:
        box = tuple(map(int, region["inner_box"]))
        assert box not in seen_boxes
        seen_boxes.add(box)
        top, left, bottom, right = box
        transformed = _transform_region(g[top : bottom + 1, left : right + 1])
        out[top : bottom + 1, left : right + 1] = transformed

    return out


def _gen_connected_coords(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    if count > height * width:
        return None
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < count:
        frontier = set()
        for r, c in cells:
            for dr, dc in FOUR_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return list(cells)


def _gen_colored_patch(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    max_cells = min(12, height * width)
    if max_cells < 5:
        return None
    for _ in range(120):
        count = random.randint(5, max_cells)
        coords = _gen_connected_coords(height, width, count)
        if coords is None:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
            continue
        color_count = 2 if count < 7 else random.choice([2, 3])
        color_count = min(color_count, count)
        colors = random.sample(palette, color_count)
        assignments = list(colors)
        while len(assignments) < count:
            assignments.append(random.choice(colors))
        random.shuffle(assignments)
        patch = np.zeros((height, width), dtype=int)
        random.shuffle(coords)
        for (r, c), color in zip(coords, assignments):
            patch[int(r), int(c)] = int(color)
        if len(_components_mask(patch != BG, connectivity=4)) != 1:
            continue
        if len(_colors_present(patch, exclude={BG})) != color_count:
            continue
        return patch
    return None


def _gen_region_dims(branch: str) -> tuple[int, int]:
    if branch == "wide":
        height = random.randint(4, 6)
        width = random.randint(height + 1, min(9, height + 4))
        return height, width
    if branch == "tall":
        width = random.randint(4, 6)
        height = random.randint(width + 1, min(9, width + 4))
        return height, width
    side = random.randint(4, 7)
    return side, side


def _render_rotations(height: int, width: int, patch: np.ndarray) -> dict[str, np.ndarray]:
    rendered = {}
    for branch in ("wide", "tall", "square"):
        rotated = _rotate_patch_for_branch(patch, branch)
        rendered[branch], _ = _center_region(height, width, rotated)
    rendered["orig"], _ = _center_region(height, width, patch)
    return rendered


def _gen_region_spec(branch: str, marker_color: int, pattern_palette: list[int]) -> dict | None:
    for _ in range(160):
        height, width = _gen_region_dims(branch)
        limit = min(height, width)
        options = [
            (ph, pw)
            for ph in range(2, limit + 1)
            for pw in range(2, limit + 1)
            if ph * pw >= 5 and (height - ph + 1) * (width - pw + 1) > 1
        ]
        random.shuffle(options)
        for ph, pw in options:
            patch = _gen_colored_patch(ph, pw, pattern_palette)
            if patch is None:
                continue
            rendered = _render_rotations(height, width, patch)
            chosen = rendered[branch]
            if np.array_equal(chosen, rendered["orig"]):
                continue
            if np.array_equal(rendered["wide"], rendered["tall"]):
                continue
            if np.array_equal(rendered["wide"], rendered["square"]):
                continue
            if np.array_equal(rendered["tall"], rendered["square"]):
                continue

            placements = [
                (r0, c0)
                for r0 in range(height - ph + 1)
                for c0 in range(width - pw + 1)
            ]
            random.shuffle(placements)
            rotated = _rotate_patch_for_branch(patch, branch)
            rph, rpw = map(int, rotated.shape)
            for offset in placements:
                input_region = _place_patch(height, width, patch, offset)
                if np.array_equal(input_region, chosen):
                    continue
                rr, cc = offset
                if rr + rph <= height and cc + rpw <= width:
                    if np.array_equal(_place_patch(height, width, rotated, offset), chosen):
                        continue
                return {
                    "branch": branch,
                    "marker_color": int(marker_color),
                    "height": int(height),
                    "width": int(width),
                    "patch": patch,
                    "input_offset": (int(offset[0]), int(offset[1])),
                    "input_region": input_region,
                    "expected_region": chosen,
                }
    return None


def _pack_blocks(
    grid_height: int,
    grid_width: int,
    block_sizes: list[tuple[int, int]],
    *,
    border: int = 1,
    pad: int = 2,
) -> list[tuple[int, int]] | None:
    used = np.zeros((grid_height, grid_width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(block_sizes)
    order = sorted(range(len(block_sizes)), key=lambda idx: block_sizes[idx][0] * block_sizes[idx][1], reverse=True)
    for idx in order:
        bh, bw = block_sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(border, grid_height - bh - border + 1):
            for c0 in range(border, grid_width - bw - border + 1):
                rr0 = max(0, r0 - pad)
                cc0 = max(0, c0 - pad)
                rr1 = min(grid_height, r0 + bh + pad)
                cc1 = min(grid_width, c0 + bw + pad)
                if not bool(used[rr0:rr1, cc0:cc1].any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = (int(pos[0]), int(pos[1]))
        r0, c0 = pos
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(grid_height, r0 + bh + pad)
        cc1 = min(grid_width, c0 + bw + pad)
        used[rr0:rr1, cc0:cc1] = True
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _render_scene(
    grid_height: int,
    grid_width: int,
    regions: list[dict],
    block_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((grid_height, grid_width), dtype=int)
    out = np.zeros((grid_height, grid_width), dtype=int)
    for spec, (block_r, block_c) in zip(regions, block_positions):
        top = int(block_r + 1)
        left = int(block_c + 1)
        bottom = top + int(spec["height"]) - 1
        right = left + int(spec["width"]) - 1

        rr, cc = spec["input_offset"]
        patch = spec["patch"]
        ph, pw = map(int, patch.shape)
        inp[top + rr : top + rr + ph, left + cc : left + cc + pw] = patch

        for r, c in _marker_pixels_for_box(top, left, bottom, right):
            inp[int(r), int(c)] = int(spec["marker_color"])

        out[top : bottom + 1, left : right + 1] = spec["expected_region"]

        spec["inner_box"] = (int(top), int(left), int(bottom), int(right))
        spec["marker_pixels"] = _marker_pixels_for_box(top, left, bottom, right)

    return inp, out


def _boxes_disjoint(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            rr0, cc0, rr1, cc1 = boxes[j]
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False
    return True


def _validate_scene(inp: np.ndarray, expected: np.ndarray, regions: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if {spec["branch"] for spec in regions} != {"wide", "tall", "square"}:
        return False
    marker_colors = {int(spec["marker_color"]) for spec in regions}
    if len(marker_colors) != 3:
        return False
    if not _boxes_disjoint([tuple(map(int, spec["inner_box"])) for spec in regions]):
        return False

    for spec in regions:
        box = tuple(map(int, spec["inner_box"]))
        top, left, bottom, right = box
        height = int(bottom - top + 1)
        width = int(right - left + 1)
        if _region_branch(height, width) != spec["branch"]:
            return False

        patch = np.asarray(spec["patch"], dtype=int)
        if not (5 <= int(np.count_nonzero(patch)) <= 12):
            return False
        patch_colors = set(_colors_present(patch, exclude={BG}))
        if not (2 <= len(patch_colors) <= 3):
            return False
        if patch_colors & marker_colors:
            return False
        if len(_components_mask(patch != BG, connectivity=4)) != 1:
            return False

        rendered = _render_rotations(height, width, patch)
        if not np.array_equal(spec["expected_region"], rendered[spec["branch"]]):
            return False
        if np.array_equal(rendered[spec["branch"]], rendered["orig"]):
            return False
        if np.array_equal(rendered["wide"], rendered["tall"]):
            return False
        if np.array_equal(rendered["wide"], rendered["square"]):
            return False
        if np.array_equal(rendered["tall"], rendered["square"]):
            return False
        if np.array_equal(spec["input_region"], spec["expected_region"]):
            return False

        rotated = _rotate_patch_for_branch(patch, spec["branch"])
        rr, cc = spec["input_offset"]
        if rr + rotated.shape[0] <= height and cc + rotated.shape[1] <= width:
            if np.array_equal(_place_patch(height, width, rotated, spec["input_offset"]), spec["expected_region"]):
                return False

    detected = _find_regions(inp)
    if len(detected) != 3:
        return False
    detected_boxes = {tuple(map(int, spec["inner_box"])) for spec in detected}
    expected_boxes = {tuple(map(int, spec["inner_box"])) for spec in regions}
    if detected_boxes != expected_boxes:
        return False
    if {int(spec["color"]) for spec in detected} != marker_colors:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    marker_colors = random.sample(range(1, 10), 3)
    pattern_palette = [color for color in range(1, 10) if color not in marker_colors]
    branches = ["wide", "tall", "square"]
    random.shuffle(branches)

    regions: list[dict] = []
    for branch, marker_color in zip(branches, marker_colors):
        spec = _gen_region_spec(branch, marker_color, pattern_palette)
        if spec is None:
            return None
        regions.append(spec)

    block_sizes = [(int(spec["height"]) + 2, int(spec["width"]) + 2) for spec in regions]
    min_height = max(20, max(size[0] for size in block_sizes) + 2)
    min_width = max(20, max(size[1] for size in block_sizes) + 2)
    grid_height = random.randint(min_height, 28)
    grid_width = random.randint(min_width, 28)

    positions = _pack_blocks(grid_height, grid_width, block_sizes, border=1, pad=2)
    if positions is None:
        return None

    inp, expected = _render_scene(grid_height, grid_width, regions, positions)
    if not _validate_scene(inp, expected, regions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-ratio rotation input")
