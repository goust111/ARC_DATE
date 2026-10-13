from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 900

_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        comps.append(np.array(coords, dtype=int))

    return comps


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "empty mask has no bbox"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_completion(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    del r0, r1
    out = np.array(mask, dtype=bool, copy=True)
    H, W = out.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        mc = int(c0 + c1 - int(c))
        assert 0 <= rr < H and 0 <= mc < W, "mirror falls outside the grid"
        out[rr, mc] = True
    return out


def _single_component_color(grid: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    assert 3 <= len(colors) <= 5, "expected three to five foreground colors"

    single_colors = []
    for color in colors:
        comps = _components(grid == color)
        if len(comps) == 1:
            single_colors.append(color)

    assert len(single_colors) == 1, "expected exactly one color to be a single 4-connected component"
    return int(single_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the uniquely connected object after absorbing same-outline occluders.

    Input structure:
    - The grid has a black background and three to five non-black colors with distinct pixel counts.
    - Exactly one color forms one 4-connected component; every other color is split into at least two components.
    - The single connected object is meant to be left-right symmetric, but a few of its cells are occupied by other colors.

    Rule:
    - Find the only color whose pixels form one 4-connected component.
    - Reflect that visible component across the vertical axis of its own bounding box to recover the complete outline.
    - Cells in the recovered outline become the object color; all remaining non-black cells become gray in place.

    Key insight:
    - The preserved object is selected by connectivity, not by pixel count; a foreign pixel is absorbed only when
      its position lies in the object's mirror-completed outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    target = _single_component_color(g)
    completed = _mirror_completion(g == target)

    out = g.copy()
    out[(g != BLACK) & ~completed] = GRAY
    out[completed] = target
    return out


def _row_interval(width: int, radius: int) -> tuple[int, int]:
    if width % 2:
        mid = width // 2
        return mid - radius, mid + radius
    left_mid = width // 2 - 1
    right_mid = width // 2
    return left_mid - radius, right_mid + radius


def _make_symmetric_mask(branch: int) -> np.ndarray | None:
    for _ in range(120):
        if branch == 0:
            h = random.randint(8, 12)
            w = random.randint(9, 13)
        else:
            h = random.randint(7, 10)
            w = random.randint(7, 11)

        max_radius = w // 2 if w % 2 else w // 2 - 1
        if max_radius < 2:
            continue

        peak = random.randint(2, max_radius)
        profile: list[int] = []
        center = (h - 1) / 2.0
        denom = max(center, 1.0)
        for r in range(h):
            falloff = abs(r - center) / denom
            envelope = int(round(peak * (1.0 - 0.68 * falloff)))
            envelope += random.choice((-1, 0, 0, 1))
            envelope = max(0, min(peak, envelope))
            low = max(0, envelope - 1)
            profile.append(random.randint(low, envelope) if envelope else 0)

        peak_row = random.randint(max(0, h // 3), min(h - 1, 2 * h // 3))
        profile[peak_row] = peak
        if peak_row + 1 < h and random.random() < 0.7:
            profile[peak_row + 1] = max(profile[peak_row + 1], peak - 1)
        if peak_row - 1 >= 0 and random.random() < 0.7:
            profile[peak_row - 1] = max(profile[peak_row - 1], peak - 1)

        if len(set(profile)) < 3:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, radius in enumerate(profile):
            c0, c1 = _row_interval(w, int(radius))
            mask[r, c0 : c1 + 1] = True

        if len(_components(mask)) != 1:
            continue
        b = _bbox_from_mask(mask)
        bbox_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        area = int(mask.sum())
        if area < 18 or bbox_area - area < 6:
            continue
        return mask

    return None


def _mirror_pairs(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    del r0, r1
    pairs = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        mc = int(c0 + c1 - c)
        assert mask[r, mc], "candidate full mask must already be mirror-symmetric"
        if c == mc:
            continue
        if abs(c - mc) <= 1:
            continue
        if c in (c0, c1) or mc in (c0, c1):
            continue

        a = (r, c)
        b = (r, mc)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))

    return pairs


def _cell_degree(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    H, W = mask.shape
    deg = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            deg += 1
    return deg


def _choose_occluders(mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    pairs = _mirror_pairs(mask)
    random.shuffle(pairs)
    selected: list[tuple[int, int]] = []

    for a, b in pairs:
        choices = [a, b]
        random.shuffle(choices)
        for cell in choices:
            if _cell_degree(mask, cell) <= 1:
                continue
            if any(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) <= 1 for s in selected):
                continue

            test = mask.copy()
            for sr, sc in selected:
                test[sr, sc] = False
            test[cell] = False
            if len(_components(test)) != 1:
                continue

            selected.append(cell)
            break

        if len(selected) == count:
            return selected

    return None


def _assign_occluder_colors(
    positions: list[tuple[int, int]],
    distractors: list[int],
) -> dict[tuple[int, int], int]:
    colors = list(distractors)
    while len(colors) < len(positions):
        colors.append(random.choice(distractors))
    random.shuffle(colors)
    return {pos: int(color) for pos, color in zip(positions, colors)}


def _choose_totals(
    target_count: int,
    occ_counts: dict[int, int],
    branch: int,
) -> dict[int, int] | None:
    colors = list(occ_counts)
    random.shuffle(colors)
    used = {int(target_count)}
    totals: dict[int, int] = {}

    def pick(candidates: list[int], minimum: int) -> int | None:
        values = [v for v in candidates if v not in used and v >= minimum]
        if not values:
            return None
        value = random.choice(values)
        used.add(value)
        return int(value)

    if branch == 0:
        for color in colors:
            value = pick(list(range(3, target_count)), occ_counts[color] + 1)
            if value is None:
                return None
            totals[color] = value
        return totals

    if branch == 1:
        high_candidates = list(range(target_count + 4, min(target_count + 13, 46) + 1))
        for color in colors:
            value = pick(high_candidates, occ_counts[color] + 1)
            if value is None:
                return None
            totals[color] = value
        return totals

    high_color = colors[0]
    high = pick(list(range(target_count + 4, min(target_count + 13, 46) + 1)), occ_counts[high_color] + 1)
    if high is None:
        return None
    totals[high_color] = high

    for color in colors[1:]:
        value = pick(list(range(3, target_count)), occ_counts[color] + 1)
        if value is None:
            return None
        totals[color] = value
    return totals


def _inside_padded_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int], pad: int) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 - pad <= r <= r1 + pad and c0 - pad <= c <= c1 + pad


def _can_place_scatter_pixel(
    grid: np.ndarray,
    r: int,
    c: int,
    reserved_bbox: tuple[int, int, int, int],
) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    if _inside_padded_bbox((r, c), reserved_bbox, 2):
        return False

    H, W = grid.shape
    for nr in range(max(0, r - 1), min(H, r + 2)):
        for nc in range(max(0, c - 1), min(W, c + 2)):
            if int(grid[nr, nc]) != BLACK:
                return False
    return True


def _scatter_pixels(
    grid: np.ndarray,
    color: int,
    count: int,
    reserved_bbox: tuple[int, int, int, int],
) -> bool:
    for _ in range(int(count)):
        candidates = [
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if _can_place_scatter_pixel(grid, r, c, reserved_bbox)
        ]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = int(color)
    return True


def _expected_output(grid: np.ndarray, full_mask: np.ndarray, target_color: int) -> np.ndarray:
    out = grid.copy()
    out[(grid != BLACK) & ~full_mask] = GRAY
    out[full_mask] = int(target_color)
    return out


def _semantic_accepts(
    grid: np.ndarray,
    full_mask: np.ndarray,
    target_color: int,
    branch: int,
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    expected = _expected_output(grid, full_mask, target_color)
    if not np.array_equal(out, expected):
        return False
    if grid.ndim != 2 or out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(out, grid):
        return False

    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    if not (3 <= len(colors) <= 5):
        return False

    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    if len(set(counts.values())) != len(counts):
        return False

    singles = [color for color in colors if len(_components(grid == color)) == 1]
    if singles != [int(target_color)]:
        return False
    if any(len(_components(grid == color)) < 2 for color in colors if color != target_color):
        return False

    target_count = counts[int(target_color)]
    if branch == 0 and target_count != max(counts.values()):
        return False
    if branch == 1 and target_count != min(counts.values()):
        return False
    if branch == 2 and not (min(counts.values()) < target_count < max(counts.values())):
        return False

    completed_from_visible = _mirror_completion(grid == target_color)
    if not np.array_equal(completed_from_visible, full_mask):
        return False

    inside_foreign = (grid != BLACK) & (grid != target_color) & full_mask
    outside_foreign = (grid != BLACK) & (grid != target_color) & ~full_mask
    if int(np.count_nonzero(inside_foreign)) < 2 or int(np.count_nonzero(outside_foreign)) < 4:
        return False

    for color in colors:
        if color == target_color:
            continue
        if not np.any((grid == color) & full_mask):
            return False
        if not np.any((grid == color) & ~full_mask):
            return False

    r0, c0, r1, c1 = _bbox_from_mask(full_mask)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if bbox_area - int(full_mask.sum()) < 6:
        return False

    return True


def _gen_attempt(branch: int) -> np.ndarray | None:
    if branch == 1:
        distractor_count = 2
    else:
        distractor_count = random.randint(2, 4)

    colors = random.sample(INPUT_COLORS, distractor_count + 1)
    target_color = int(colors[0])
    distractors = [int(c) for c in colors[1:]]

    local_mask = _make_symmetric_mask(branch)
    if local_mask is None:
        return None

    occ_total = distractor_count + random.randint(0, min(3, distractor_count))
    occluders = _choose_occluders(local_mask, occ_total)
    if occluders is None:
        return None

    visible_local = local_mask.copy()
    for r, c in occluders:
        visible_local[r, c] = False
    if len(_components(visible_local)) != 1:
        return None

    if branch != 0 and int(visible_local.sum()) > 31:
        return None

    occ_color_at = _assign_occluder_colors(occluders, distractors)
    occ_counts = {color: 0 for color in distractors}
    for color in occ_color_at.values():
        occ_counts[int(color)] += 1

    target_visible_count = int(visible_local.sum())
    totals = _choose_totals(target_visible_count, occ_counts, branch)
    if totals is None:
        return None

    h, w = local_mask.shape
    min_h = max(22, h + 10)
    min_w = max(22, w + 10)
    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)

    top = random.randint(3, H - h - 4)
    left = random.randint(3, W - w - 4)

    grid = np.full((H, W), BLACK, dtype=int)
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[top : top + h, left : left + w] = local_mask

    for r, c in np.argwhere(local_mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = target_color

    for (r, c), color in occ_color_at.items():
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = int(color)

    reserved_bbox = _bbox_from_mask(full_mask)
    scatter_jobs = []
    for color in distractors:
        outside_count = int(totals[color] - occ_counts[color])
        if outside_count <= 0:
            return None
        scatter_jobs.append((outside_count, color))

    random.shuffle(scatter_jobs)
    scatter_jobs.sort(key=lambda item: item[0], reverse=True)
    for outside_count, color in scatter_jobs:
        if not _scatter_pixels(grid, color, outside_count, reserved_bbox):
            return None

    if not _semantic_accepts(grid, full_mask, target_color, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    branch = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mode-mask occlusion input")
