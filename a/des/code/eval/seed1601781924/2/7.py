from __future__ import annotations

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_COLORS = list(range(1, 10))
_BG = 0
_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray | None:
    coords = np.argwhere(arr != int(bg))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords_list = []
            while stack:
                rr, cc = stack.pop()
                coords_list.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(coords_list, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(coords.shape[0]),
                    "center_sum": (r0 + r1, c0 + c1),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _is_solid_rect(comp: dict) -> bool:
    return bool(np.all(comp["mask"]))


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return bool(
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _bbox_strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return bool(
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _extract_frames_and_ring(grid: np.ndarray, bg: int) -> tuple[list[dict], list[int]]:
    comps = _components(grid, bg)
    frames = [comp for comp in comps if _is_rect_frame(comp)]
    assert len(frames) >= 2

    frames.sort(
        key=lambda comp: (
            (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1),
            comp["area"],
        ),
        reverse=True,
    )

    center_sum = frames[0]["center_sum"]
    for outer, inner in zip(frames, frames[1:]):
        assert outer["center_sum"] == center_sum
        assert inner["center_sum"] == center_sum
        assert _bbox_strictly_contains(outer["bbox"], inner["bbox"])

    frame_colors = [int(frame["color"]) for frame in frames]
    assert len(frame_colors) == len(set(frame_colors))

    outer_bbox = frames[0]["bbox"]
    ring = [
        comp
        for comp in comps
        if not _bbox_contains(outer_bbox, comp["bbox"]) and _is_solid_rect(comp)
    ]
    assert 3 <= len(ring) <= 5

    ring_colors = [int(comp["color"]) for comp in ring]
    assert len(ring_colors) == len(set(ring_colors))
    assert set(frame_colors).issubset(set(ring_colors))

    center_rows = [comp["bbox"][0] + comp["bbox"][2] for comp in ring]
    center_cols = [comp["bbox"][1] + comp["bbox"][3] for comp in ring]
    row_span = max(center_rows) - min(center_rows)
    col_span = max(center_cols) - min(center_cols)

    if col_span >= row_span:
        ring.sort(key=lambda comp: (comp["bbox"][1] + comp["bbox"][3], comp["bbox"][0] + comp["bbox"][2]))
    else:
        ring.sort(key=lambda comp: (comp["bbox"][0] + comp["bbox"][2], comp["bbox"][1] + comp["bbox"][3]))

    return frames, [int(comp["color"]) for comp in ring]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the innermost pattern according to color-ring distances between nested frames."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames, ring_colors = _extract_frames_and_ring(g, bg)
    ring_pos = {color: i for i, color in enumerate(ring_colors)}

    total_steps = 0
    for outer, inner in zip(frames, frames[1:]):
        start = ring_pos[int(outer["color"])]
        end = ring_pos[int(inner["color"])]
        total_steps += (end - start) % len(ring_colors)

    r0, c0, r1, c1 = frames[-1]["bbox"]
    inner_region = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner_region.size > 0

    rotated = np.rot90(inner_region, (-total_steps) % 4)
    cropped = _crop_non_bg(rotated, bg)
    assert cropped is not None
    return cropped


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    grid[r0, c0 : c0 + side] = int(color)
    grid[r0 + side - 1, c0 : c0 + side] = int(color)
    grid[r0 : r0 + side, c0] = int(color)
    grid[r0 : r0 + side, c0 + side - 1] = int(color)


def _pattern_bbox_requirements(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]]:
    if h == 1:
        return [(0, 0), (0, w - 1)]
    if w == 1:
        return [(0, 0), (h - 1, 0)]
    if _randint(rng, 0, 1) == 0:
        return [(0, 0), (h - 1, w - 1)]
    return [(0, w - 1), (h - 1, 0)]


def _make_pattern_patch(
    rng: np.random.Generator,
    max_side: int,
    available_colors: list[int],
) -> np.ndarray | None:
    feasible_dims = [
        (h, w)
        for h in range(1, min(4, max_side) + 1)
        for w in range(1, min(4, max_side) + 1)
        if h * w >= 2
    ]
    if not feasible_dims:
        return None

    for _ in range(60):
        ph, pw = feasible_dims[_randint(rng, 0, len(feasible_dims) - 1)]
        required = _pattern_bbox_requirements(rng, ph, pw)
        max_cells = min(ph * pw, 7)
        fill_count = _randint(rng, len(required), max_cells)

        coords = list(required)
        pool = [(r, c) for r in range(ph) for c in range(pw) if (r, c) not in coords]
        if fill_count > len(coords):
            picked = rng.choice(len(pool), size=fill_count - len(coords), replace=False)
            coords.extend(pool[int(i)] for i in np.atleast_1d(picked))

        color_count = _randint(rng, 1, min(3, len(available_colors)))
        colors = [int(available_colors[int(i)]) for i in rng.choice(len(available_colors), size=color_count, replace=False)]

        patch = np.full((ph, pw), _BG, dtype=int)
        rng.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if color_count == 1:
                patch[r, c] = int(colors[0])
            else:
                patch[r, c] = int(colors[idx % color_count] if idx < color_count else colors[_randint(rng, 0, color_count - 1)])

        tight = _crop_non_bg(patch, _BG)
        assert tight is not None
        if not np.array_equal(tight, np.rot90(tight, 1)):
            return patch
    return None


def _ring_footprint(length: int, block: int, gap: int, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return block, length * block + (length - 1) * gap
    return length * block + (length - 1) * gap, block


def _place_ring(
    grid: np.ndarray,
    colors: list[int],
    corner: str,
    horizontal: bool,
    block: int,
    gap: int,
    margin_r: int,
    margin_c: int,
) -> tuple[int, int, int, int]:
    h, w = _ring_footprint(len(colors), block, gap, horizontal)
    if corner == "tl":
        r0 = margin_r
        c0 = margin_c
    elif corner == "tr":
        r0 = margin_r
        c0 = grid.shape[1] - margin_c - w
    elif corner == "bl":
        r0 = grid.shape[0] - margin_r - h
        c0 = margin_c
    else:
        r0 = grid.shape[0] - margin_r - h
        c0 = grid.shape[1] - margin_c - w

    for idx, color in enumerate(colors):
        if horizontal:
            rr = r0
            cc = c0 + idx * (block + gap)
        else:
            rr = r0 + idx * (block + gap)
            cc = c0
        grid[rr : rr + block, cc : cc + block] = int(color)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return bool(
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _total_rotation(frame_colors: list[int], ring_colors: list[int]) -> int:
    pos = {color: i for i, color in enumerate(ring_colors)}
    total = 0
    for a, b in zip(frame_colors, frame_colors[1:]):
        total += (pos[b] - pos[a]) % len(ring_colors)
    return total % 4


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ring_len = _randint(rng, 3, 5)
    ring_colors = [int(c) for c in rng.choice(_COLORS, size=ring_len, replace=False)]
    frame_count = _randint(rng, 3, ring_len)
    frame_pick = [int(i) for i in rng.choice(ring_len, size=frame_count, replace=False)]
    frame_colors = [ring_colors[i] for i in frame_pick]
    rng.shuffle(frame_colors)

    total_rot = _total_rotation(frame_colors, ring_colors)
    if _randint(rng, 0, 3) != 0 and total_rot == 0:
        return None

    inner_side = _randint(rng, 4, 8)
    gaps = [_randint(rng, 1, 2) for _ in range(frame_count - 1)]
    outer_side = inner_side + 2 + 2 * sum(gap + 1 for gap in gaps)

    horizontal = bool(_randint(rng, 0, 1))
    corner = ["tl", "tr", "bl", "br"][_randint(rng, 0, 3)]
    block = _randint(rng, 1, 2)
    gap = _randint(rng, 0, 1)
    margin_r = _randint(rng, 0, 1)
    margin_c = _randint(rng, 0, 1)
    ring_h, ring_w = _ring_footprint(ring_len, block, gap, horizontal)

    min_h = max(outer_side + 4, ring_h + margin_r + 1)
    min_w = max(outer_side + 4, ring_w + margin_c + 1)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = _randint(rng, min_h, min(30, min_h + 6))
    grid_w = _randint(rng, min_w, min(30, min_w + 6))
    grid = np.full((grid_h, grid_w), _BG, dtype=int)

    outer_r0 = (grid_h - outer_side) // 2
    outer_c0 = (grid_w - outer_side) // 2
    outer_bbox = (outer_r0, outer_c0, outer_r0 + outer_side - 1, outer_c0 + outer_side - 1)

    ring_bbox = _place_ring(grid, ring_colors, corner, horizontal, block, gap, margin_r, margin_c)
    if not _boxes_separated(outer_bbox, ring_bbox, pad=1):
        return None

    cur_r = outer_r0
    cur_c = outer_c0
    cur_side = outer_side
    for idx, color in enumerate(frame_colors):
        _draw_frame(grid, cur_r, cur_c, cur_side, color)
        if idx < len(gaps):
            delta = gaps[idx] + 1
            cur_r += delta
            cur_c += delta
            cur_side -= 2 * delta

    motif_colors = [color for color in _COLORS if color not in ring_colors]
    if not motif_colors:
        motif_colors = _COLORS[:]
    patch = _make_pattern_patch(rng, inner_side, motif_colors)
    if patch is None:
        return None

    pr, pc = patch.shape
    inner_r0 = cur_r + 1
    inner_c0 = cur_c + 1
    place_r = inner_r0 + _randint(rng, 0, inner_side - pr)
    place_c = inner_c0 + _randint(rng, 0, inner_side - pc)
    patch_mask = patch != _BG
    grid_slice = grid[place_r : place_r + pr, place_c : place_c + pc]
    grid_slice[patch_mask] = patch[patch_mask]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
