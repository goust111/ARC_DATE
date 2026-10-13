from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
GEN_PAIR_COUNTS = (2, 3)
GEN_CORNERS = ("tl", "tr", "bl", "br")
GEN_PALETTE = tuple(range(1, 10))
_RNG = np.random.default_rng()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_dict(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    color = int(grid[coords[0]])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "color": color,
        "coords": coords,
        "bbox": bbox,
        "area": len(coords),
        "mask": mask,
    }


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append(_component_dict(grid, coords))
    return objects


def _find_foreground_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append(_component_dict(grid, coords))
    return objects


def _legend_candidate(component: dict, grid_shape: tuple[int, int]) -> bool:
    h, w = grid_shape
    r0, c0, r1, c1 = component["bbox"]
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    if bw != 3 or component["area"] != bh * bw:
        return False
    if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
        return False
    return True


def _extract_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[frozenset[int], int]]:
    candidates = [obj for obj in _find_foreground_components(grid) if _legend_candidate(obj, grid.shape)]
    assert len(candidates) == 1, "expected exactly one corner legend"
    legend = candidates[0]
    r0, c0, r1, c1 = legend["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mapping: dict[frozenset[int], int] = {}
    for rr in range(patch.shape[0]):
        left, right, mixed = map(int, patch[rr, :])
        assert left != BG and right != BG and mixed != BG
        assert left != right, "legend row must refer to two distinct piece colors"
        key = frozenset((left, right))
        assert key not in mapping, "legend color pair must be unique"
        mapping[key] = mixed
    return legend["bbox"], mapping


def _normalized_coords(component: dict) -> tuple[list[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = component["bbox"]
    coords = [(r - r0, c - c0) for r, c in component["coords"]]
    return coords, r1 - r0 + 1, c1 - c0 + 1


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _fit_pair(component_a: dict, component_b: dict) -> np.ndarray | None:
    coords_a, ha, wa = _normalized_coords(component_a)
    coords_b, hb, wb = _normalized_coords(component_b)
    total_area = int(component_a["area"]) + int(component_b["area"])
    solutions: dict[bytes, np.ndarray] = {}

    for h, w in _factor_pairs(total_area):
        if h < max(ha, hb) or w < max(wa, wb):
            continue
        base = np.zeros((h, w), dtype=np.uint8)
        for ra in range(h - ha + 1):
            for ca in range(w - wa + 1):
                owner_a = base.copy()
                for rr, cc in coords_a:
                    owner_a[ra + rr, ca + cc] = 1
                for rb in range(h - hb + 1):
                    for cb in range(w - wb + 1):
                        overlap = False
                        for rr, cc in coords_b:
                            if owner_a[rb + rr, cb + cc] != 0:
                                overlap = True
                                break
                        if overlap:
                            continue
                        owner = owner_a.copy()
                        for rr, cc in coords_b:
                            owner[rb + rr, cb + cc] = 2
                        if not np.all(owner > 0):
                            continue
                        key = owner.tobytes()
                        solutions[key] = owner
                        if len(solutions) > 1:
                            return None
    if len(solutions) != 1:
        return None
    return next(iter(solutions.values()))


def _match_pairs(components: list[dict], legend: dict[frozenset[int], int]) -> list[tuple[int, int, np.ndarray]]:
    n = len(components)
    fits: dict[tuple[int, int], np.ndarray] = {}
    for i, j in combinations(range(n), 2):
        key = frozenset((int(components[i]["color"]), int(components[j]["color"])))
        if key not in legend:
            continue
        owner = _fit_pair(components[i], components[j])
        if owner is not None:
            fits[(i, j)] = owner

    solutions: list[list[tuple[int, int, np.ndarray]]] = []
    used = [False] * n

    def backtrack(current: list[tuple[int, int, np.ndarray]]) -> None:
        if len(solutions) > 1:
            return
        try:
            first = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            solutions.append(current.copy())
            return
        used[first] = True
        for second in range(first + 1, n):
            if used[second]:
                continue
            owner = fits.get((first, second))
            if owner is None:
                continue
            used[second] = True
            current.append((first, second, owner))
            backtrack(current)
            current.pop()
            used[second] = False
        used[first] = False

    backtrack([])
    assert len(solutions) == 1, "expected a unique complement pairing"
    return solutions[0]


def _render_owner(owner: np.ndarray, color_a: int, color_b: int, mix_color: int) -> np.ndarray:
    patch = np.zeros_like(owner, dtype=int)
    patch[owner == 1] = int(color_a)
    patch[owner == 2] = int(color_b)
    boundary = np.zeros_like(owner, dtype=bool)
    h, w = owner.shape
    for r in range(h):
        for c in range(w):
            if owner[r, c] == 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if owner[nr, nc] != 0 and owner[nr, nc] != owner[r, c]:
                    boundary[r, c] = True
                    boundary[nr, nc] = True
    patch[boundary] = int(mix_color)
    return patch


def _pair_sort_key(pair: tuple[int, int, np.ndarray], components: list[dict]) -> tuple[float, float]:
    i, j, _ = pair
    b1 = components[i]["bbox"]
    b2 = components[j]["bbox"]
    c1 = (b1[1] + b1[3]) / 2.0
    c2 = (b2[1] + b2[3]) / 2.0
    r1 = (b1[0] + b1[2]) / 2.0
    r2 = (b2[0] + b2[2]) / 2.0
    return ((c1 + c2) / 2.0, (r1 + r2) / 2.0)


def _pack_output(patches: list[np.ndarray]) -> np.ndarray:
    height = max(p.shape[0] for p in patches)
    width = sum(p.shape[1] for p in patches) + max(0, len(patches) - 1)
    out = np.full((height, width), BG, dtype=int)
    col = 0
    for idx, patch in enumerate(patches):
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w
        if idx + 1 < len(patches):
            col += 1
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    legend_bbox, legend_map = _extract_legend(grid)
    work = np.array(grid, dtype=int)
    r0, c0, r1, c1 = legend_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = BG
    components = _find_monochrome_components(work)
    assert len(components) == 2 * len(legend_map), "legend row count must match piece count"
    pairs = _match_pairs(components, legend_map)
    pairs.sort(key=lambda item: _pair_sort_key(item, components))

    patches: list[np.ndarray] = []
    for i, j, owner in pairs:
        color_a = int(components[i]["color"])
        color_b = int(components[j]["color"])
        mix_color = int(legend_map[frozenset((color_a, color_b))])
        patches.append(_render_owner(owner, color_a, color_b, mix_color))
    return _pack_output(patches)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered puzzle pieces by shape complement, reassemble each pair into a full
    rectangle, and recolor the contact boundary using the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _make_piece_patch(mask: np.ndarray, color: int) -> np.ndarray:
    cropped = _crop_mask(mask)
    patch = np.zeros(cropped.shape, dtype=int)
    patch[cropped] = int(color)
    return patch


def _build_pair_owner(rng: np.random.Generator) -> np.ndarray | None:
    orient = "v" if int(rng.integers(2)) == 0 else "h"
    height = int(rng.integers(3, 7))
    width = int(rng.integers(4, 8))
    owner = np.zeros((height, width), dtype=np.uint8)

    if orient == "v":
        cuts = rng.integers(1, width, size=height)
        for r in range(height):
            cut = int(cuts[r])
            owner[r, :cut] = 1
            owner[r, cut:] = 2
    else:
        cuts = rng.integers(1, height, size=width)
        for c in range(width):
            cut = int(cuts[c])
            owner[:cut, c] = 1
            owner[cut:, c] = 2

    if int((owner == 1).sum()) < 5 or int((owner == 2).sum()) < 5:
        return None
    if rng.random() < 0.5:
        owner = np.where(owner == 1, 2, 1).astype(np.uint8)
    return owner


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    if np.any(grid[r0:r1, c0:c1] != BG):
        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(target[mask] != BG)
    target[mask] = patch[mask]


def _legend_position(corner: str, grid_shape: tuple[int, int], legend_h: int) -> tuple[int, int]:
    h, w = grid_shape
    row = 0 if corner in ("tl", "tr") else h - legend_h
    col = 0 if corner in ("tl", "bl") else w - 3
    return row, col


def _expected_output(pair_specs: list[dict]) -> np.ndarray:
    patches = [
        _render_owner(spec["owner"], spec["colors"][0], spec["colors"][1], spec["mix"])
        for spec in pair_specs
    ]
    return _pack_output(patches)


def _generate_attempt() -> np.ndarray | None:
    rng = _RNG
    pair_count = int(rng.choice(GEN_PAIR_COUNTS))
    corner = str(rng.choice(GEN_CORNERS))
    colors = list(rng.choice(GEN_PALETTE, size=3 * pair_count, replace=False))

    pair_specs: list[dict] = []
    for pair_idx in range(pair_count):
        owner = _build_pair_owner(rng)
        if owner is None:
            return None
        color_a, color_b, mix = map(int, colors[3 * pair_idx : 3 * pair_idx + 3])
        patch_a = _make_piece_patch(owner == 1, color_a)
        patch_b = _make_piece_patch(owner == 2, color_b)
        pair_specs.append(
            {
                "owner": owner,
                "colors": (color_a, color_b),
                "mix": mix,
                "patches": [patch_a, patch_b],
            }
        )

    left_reserved = 4 if corner in ("tl", "bl") else 1
    right_reserved = 4 if corner in ("tr", "br") else 1
    top_reserved = pair_count + 2 if corner in ("tl", "tr") else 1
    bottom_reserved = pair_count + 2 if corner in ("bl", "br") else 1

    upper_choice = [int(_RNG.integers(2)) for _ in range(pair_count)]
    upper_heights = [
        pair_specs[idx]["patches"][upper_choice[idx]].shape[0]
        for idx in range(pair_count)
    ]
    lower_heights = [
        pair_specs[idx]["patches"][1 - upper_choice[idx]].shape[0]
        for idx in range(pair_count)
    ]
    upper_band = max(upper_heights) + int(rng.integers(1, 4))
    lower_band = max(lower_heights) + int(rng.integers(1, 4))
    band_gap = int(rng.integers(2, 5))
    top_slack = int(rng.integers(1, 4))
    bottom_slack = int(rng.integers(1, 4))
    grid_h = top_reserved + top_slack + upper_band + band_gap + lower_band + bottom_reserved + bottom_slack

    lane_widths = [
        max(pair["patches"][0].shape[1], pair["patches"][1].shape[1]) + int(rng.integers(3, 6))
        for pair in pair_specs
    ]
    lane_gaps = [int(rng.integers(2, 5)) for _ in range(max(0, pair_count - 1))]
    left_slack = int(rng.integers(1, 4))
    right_slack = int(rng.integers(1, 4))
    grid_w = (
        left_reserved
        + left_slack
        + sum(lane_widths)
        + sum(lane_gaps)
        + right_reserved
        + right_slack
    )

    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend_rows = [
        [spec["colors"][0], spec["colors"][1], spec["mix"]]
        for spec in pair_specs
    ]
    rng.shuffle(legend_rows)
    legend_r, legend_c = _legend_position(corner, grid.shape, pair_count)
    for rr, row_vals in enumerate(legend_rows):
        grid[legend_r + rr, legend_c : legend_c + 3] = np.array(row_vals, dtype=int)

    upper_start = top_reserved + top_slack
    lower_start = upper_start + upper_band + band_gap
    cursor = left_reserved + left_slack
    for idx, pair in enumerate(pair_specs):
        lane_w = lane_widths[idx]
        patch_upper = pair["patches"][upper_choice[idx]]
        patch_lower = pair["patches"][1 - upper_choice[idx]]

        upper_y = upper_start + int(rng.integers(0, upper_band - patch_upper.shape[0] + 1))
        lower_y = lower_start + int(rng.integers(0, lower_band - patch_lower.shape[0] + 1))
        upper_x = cursor + int(rng.integers(0, lane_w - patch_upper.shape[1] + 1))
        lower_x = cursor + int(rng.integers(0, lane_w - patch_lower.shape[1] + 1))

        if not _can_place_patch(grid, patch_upper, upper_y, upper_x):
            return None
        if not _can_place_patch(grid, patch_lower, lower_y, lower_x):
            return None

        _place_patch(grid, patch_upper, upper_y, upper_x)
        _place_patch(grid, patch_lower, lower_y, lower_x)

        cursor += lane_w
        if idx + 1 < pair_count:
            cursor += lane_gaps[idx]

    expected = _expected_output(pair_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate ARC-style inputs with 2 or 3 complementary piece pairs and a
    3-column corner legend describing the boundary recolor for each pair.
    """
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-piece puzzle")
