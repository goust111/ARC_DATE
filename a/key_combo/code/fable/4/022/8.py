from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(sub[1:-1, 1:-1] == bg):
                continue
            frames.append({"color": int(color), "bbox": (r0, c0, r1, c1)})
    assert frames, "expected at least one rectangular frame"
    return frames


def _marker_count(grid: np.ndarray, coords: np.ndarray) -> int:
    colors = Counter(int(grid[r, c]) for r, c in coords)
    assert len(colors) == 2, f"expected exactly two object colors, got {sorted(colors)}"
    marker_count = min(colors.values())
    marker_colors = [color for color, count in colors.items() if count == marker_count]
    assert len(marker_colors) == 1, "marker color must be uniquely sparse"
    assert marker_count < max(colors.values()), "marker color must be sparser than body color"
    return int(marker_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor or erase free objects by matching their marker counts to framed exemplar counts.

    Input structure:
    - A dominant background surrounds several single-color rectangular frames with distinct colors.
    - Each frame encloses one multicolor sample object made of a dense body color plus sparse marker pixels.
    - Outside every frame bbox are multicolor free objects built the same way, with varying shapes and marker counts.

    Rule:
    - Find each rectangular frame and inspect the unique object inside its interior.
    - Count the sparse marker pixels inside that sample object to build a mapping from marker count to frame color.
    - For every free object outside the frames, recolor all of its pixels to the mapped frame color when its marker count matches, otherwise erase the whole object to the background.

    Key insight:
    - The usable legend is implicit: each frame color stands for the marker count of the sample it encloses, so free objects are classified by internal marker count rather than by shape, size, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames = _find_frames(g, bg)
    count_to_color: dict[int, int] = {}
    outside = np.ones(g.shape, dtype=bool)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        sample_comps = _components(inner != bg)
        assert len(sample_comps) == 1, "each frame must contain exactly one sample object"
        sample = sample_comps[0] + np.array([r0 + 1, c0 + 1], dtype=int)
        count = _marker_count(g, sample)
        assert count not in count_to_color, "sample marker counts must be unique"
        count_to_color[count] = int(frame["color"])

    free_comps = _components((g != bg) & outside)
    assert free_comps, "expected at least one free object"

    out = g.copy()
    for comp in free_comps:
        count = _marker_count(g, comp)
        color = count_to_color.get(count, bg)
        out[comp[:, 0], comp[:, 1]] = int(color)
    return out


PALETTE = tuple(range(10))
COUNT_POOL = (1, 2, 3, 4, 5)
FRAME_COUNT_RANGE = (2, 3)
UNMATCHED_FREE_RANGE = (1, 2)
GRID_SIDE_RANGE = (22, 30)
GENERATE_MAX_RETRIES = 400
PLACE_PADDING = 1


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                cells.append((r, c))
    return cells


def _build_object_patch(
    rng: random.Random, bg: int, body_color: int, marker_color: int, marker_count: int
) -> np.ndarray:
    for _ in range(120):
        base_h = rng.randint(4, 5)
        base_w = rng.randint(4, 6)
        if (base_h - 2) * (base_w - 2) < marker_count:
            continue

        mask = np.zeros((base_h + 4, base_w + 4), dtype=bool)
        top = 2
        left = 2
        mask[top : top + base_h, left : left + base_w] = True

        arm_count = rng.randint(0, 2)
        for side in rng.sample(("top", "bottom", "left", "right"), arm_count):
            if side in {"top", "bottom"}:
                arm_w = rng.randint(1, min(3, base_w - 2))
                start_c = left + rng.randint(1, base_w - arm_w - 1)
                if side == "top":
                    mask[top - 1, start_c : start_c + arm_w] = True
                else:
                    mask[top + base_h, start_c : start_c + arm_w] = True
            else:
                arm_h = rng.randint(1, min(3, base_h - 2))
                start_r = top + rng.randint(1, base_h - arm_h - 1)
                if side == "left":
                    mask[start_r : start_r + arm_h, left - 1] = True
                else:
                    mask[start_r : start_r + arm_h, left + base_w] = True

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        interior = _interior_cells(mask)
        if len(interior) <= marker_count:
            continue

        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = int(body_color)
        for r, c in rng.sample(interior, marker_count):
            patch[r, c] = int(marker_color)

        k = rng.randrange(4)
        patch = np.rot90(patch, k)
        if rng.random() < 0.5:
            patch = np.fliplr(patch)

        body_pixels = int(np.count_nonzero(patch == body_color))
        marker_pixels = int(np.count_nonzero(patch == marker_color))
        if marker_pixels == marker_count and body_pixels > marker_pixels:
            return patch
    raise RuntimeError("failed to build object patch")


def _frame_patch(frame_color: int, object_patch: np.ndarray, bg: int) -> np.ndarray:
    oh, ow = object_patch.shape
    patch = np.full((oh + 4, ow + 4), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[2 : 2 + oh, 2 : 2 + ow] = object_patch
    return patch


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, bg: int, rng: random.Random
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    ph, pw = patch.shape
    anchors = [(r, c) for r in range(gh - ph + 1) for c in range(gw - pw + 1)]
    rng.shuffle(anchors)
    for r0, c0 in anchors:
        rr0 = max(0, r0 - PLACE_PADDING)
        cc0 = max(0, c0 - PLACE_PADDING)
        rr1 = min(gh, r0 + ph + PLACE_PADDING)
        cc1 = min(gw, c0 + pw + PLACE_PADDING)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        mask = patch != bg
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)
    return None


def _patch_coords(patch: np.ndarray, bg: int, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(patch != bg)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(PALETTE)
    usable = [color for color in PALETTE if color != bg]
    rng.shuffle(usable)

    n_frames = rng.randint(*FRAME_COUNT_RANGE)
    if len(usable) < n_frames + 2:
        return None
    body_color = usable.pop()
    marker_color = usable.pop()
    frame_colors = [usable.pop() for _ in range(n_frames)]

    sample_counts = rng.sample(list(COUNT_POOL), n_frames)
    unmatched_pool = [count for count in COUNT_POOL if count not in sample_counts]
    n_unmatched = rng.randint(UNMATCHED_FREE_RANGE[0], min(UNMATCHED_FREE_RANGE[1], len(unmatched_pool)))
    free_counts = list(sample_counts) + rng.sample(unmatched_pool, n_unmatched)
    if rng.random() < 0.5:
        free_counts.append(rng.choice(sample_counts))
    rng.shuffle(free_counts)

    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    expected = grid.copy()
    count_to_color = dict(zip(sample_counts, frame_colors))
    free_coords_and_counts: list[tuple[np.ndarray, int]] = []

    frame_specs = list(zip(frame_colors, sample_counts))
    rng.shuffle(frame_specs)
    for frame_color, count in frame_specs:
        obj_patch = _build_object_patch(rng, bg, body_color, marker_color, count)
        patch = _frame_patch(frame_color, obj_patch, bg)
        placed = _place_patch(grid, reserved, patch, bg, rng)
        if placed is None:
            return None

    # Rebuild the frame layout in expected directly from the finished input grid.
    expected[:, :] = grid

    free_patches = []
    for count in free_counts:
        patch = _build_object_patch(rng, bg, body_color, marker_color, count)
        free_patches.append((count, patch))
    rng.shuffle(free_patches)

    for count, patch in free_patches:
        placed = _place_patch(grid, reserved, patch, bg, rng)
        if placed is None:
            return None
        coords = _patch_coords(patch, bg, *placed)
        free_coords_and_counts.append((coords, count))

    matched = 0
    erased = 0
    for coords, count in free_coords_and_counts:
        color = count_to_color.get(count, bg)
        expected[coords[:, 0], coords[:, 1]] = int(color)
        matched += int(color != bg)
        erased += int(color == bg)

    if matched == 0 or erased == 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATE_MAX_RETRIES} attempts")
