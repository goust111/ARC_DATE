from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_rect(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components_mask(mask)) == 1


def _is_filled_rectangle(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return len(pts) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _is_rect_border_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(mask.sum() == int(border.sum()) and np.all(mask[border]) and not np.any(mask[~border]))


def _is_frame_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and not np.any(patch[~border] == color))


def _find_regions(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    frame_candidates: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        for comp in _components_mask(grid == color):
            if _is_frame_component(grid, color, comp):
                boxes.append(_bbox(comp))
        if boxes:
            frame_candidates[color] = sorted(set(boxes))

    assert frame_candidates, "expected at least one rectangular frame color"
    frame_color, boxes = max(frame_candidates.items(), key=lambda item: (len(item[1]), -item[0]))
    assert len(boxes) >= 2 and len(boxes) % 2 == 0, "frame count must be an even number"

    regions: list[dict[str, object]] = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        regions.append({"bbox": bbox, "inner": inner})
    return int(frame_color), regions


def _template_match(template: np.ndarray, instance: np.ndarray, bg: int) -> tuple[int, int, np.ndarray] | None:
    if template.shape != instance.shape:
        return None

    template_colors = {int(v) for v in np.unique(template) if int(v) != bg}
    if len(template_colors) != 1:
        return None
    pattern_color = next(iter(template_colors))
    if not _is_connected(template == pattern_color):
        return None

    instance_colors = {int(v) for v in np.unique(instance) if int(v) != bg}
    if len(instance_colors) != 2 or pattern_color not in instance_colors:
        return None
    blocker_color = next(color for color in instance_colors if color != pattern_color)

    blocker = instance == blocker_color
    if not _is_filled_rectangle(blocker):
        return None
    if not np.any((template == pattern_color) & blocker):
        return None
    if not np.any((template == pattern_color) & ~blocker):
        return None
    if not np.array_equal(instance[~blocker], template[~blocker]):
        return None
    return pattern_color, blocker_color, blocker


def _pair_regions(regions: list[dict[str, object]], bg: int) -> list[tuple[int, int, int, np.ndarray]]:
    template_options: dict[int, list[tuple[int, int, np.ndarray]]] = {idx: [] for idx in range(len(regions))}
    instance_options: dict[int, list[tuple[int, int, np.ndarray]]] = {idx: [] for idx in range(len(regions))}

    for t_idx, template_region in enumerate(regions):
        template = np.asarray(template_region["inner"], dtype=int)
        for i_idx, instance_region in enumerate(regions):
            if t_idx == i_idx:
                continue
            instance = np.asarray(instance_region["inner"], dtype=int)
            match = _template_match(template, instance, bg)
            if match is None:
                continue
            _pattern_color, blocker_color, blocker = match
            template_options[t_idx].append((i_idx, blocker_color, blocker))
            instance_options[i_idx].append((t_idx, blocker_color, blocker))

    pairs: list[tuple[int, int, int, np.ndarray]] = []
    used_templates: set[int] = set()
    used_instances: set[int] = set()
    for instance_idx, options in instance_options.items():
        if not options:
            continue
        assert len(options) == 1, "each instance must match exactly one template"
        template_idx, blocker_color, blocker = options[0]
        assert len(template_options[template_idx]) == 1, "each template must match exactly one instance"
        assert template_idx not in used_templates and instance_idx not in used_instances
        used_templates.add(template_idx)
        used_instances.add(instance_idx)
        pairs.append((template_idx, instance_idx, blocker_color, blocker))

    assert pairs and len(pairs) * 2 == len(regions), "all framed regions must be paired"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each occluded framed instance by copying the matching framed template, then erase the templates.

    Input structure:
    - bg is the most common color.
    - Several grey one-pixel rectangular frames appear on the grid; each frame encloses either a complete template or an occluded instance.
    - A template interior contains one connected pattern color A on bg.
    - The matching instance has the same frame size and same pattern color A, but one axis-aligned rectangle is overwritten by a different color B.

    Rule:
    - Find all framed regions and pair template/instance regions by exact interior size and by the fact that the instance differs from the template only on one solid rectangular blocker of color B.
    - For each pair, restore the instance cells under the blocker from the template.
    - If a blocked cell corresponds to bg in the template, keep color B there as the output marker.
    - Clear every template region completely to bg, including its grey frame.

    Key insight:
    - The template is identified by being the only single-color connected interior that explains an instance after removing one rectangular occluder; the blocker's non-overlap with the pattern is exactly the marker that survives in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, regions = _find_regions(g, bg)
    pairs = _pair_regions(regions, bg)

    out = g.copy()
    for template_idx, _instance_idx, _blocker_color, _blocker in pairs:
        r0, c0, r1, c1 = tuple(regions[template_idx]["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for template_idx, instance_idx, blocker_color, blocker in pairs:
        template_inner = np.asarray(regions[template_idx]["inner"], dtype=int)
        repaired = template_inner.copy()
        repaired[blocker & (template_inner == bg)] = blocker_color
        r0, c0, r1, c1 = tuple(regions[instance_idx]["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = repaired
        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color

    return out


def _random_growth_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    lo = max(4, (h * w) // 3)
    hi = max(lo, min(h * w - 1, (2 * h * w) // 3 + 1))
    target = rng.randint(lo, hi)
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while int(mask.sum()) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _bar_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    vert_w = rng.randint(1, max(1, min(w - 1, max(1, w // 2))))
    vert_c0 = rng.randint(0, w - vert_w)
    mask[:, vert_c0 : vert_c0 + vert_w] = True

    horiz_h = rng.randint(1, max(1, min(h - 1, max(1, h // 2))))
    horiz_r0 = rng.randint(0, h - horiz_h)
    left = rng.randint(0, vert_c0)
    right = rng.randint(vert_c0 + vert_w - 1, w - 1)
    mask[horiz_r0 : horiz_r0 + horiz_h, left : right + 1] = True
    return mask


def _candidate_blockers(mask: np.ndarray, preserve_marker: bool) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    total_on = int(mask.sum())
    max_area = max(2, int(math.ceil(mask.size * 0.6)))
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < 2 or area > max_area:
                        continue
                    covered = int(mask[r0 : r1 + 1, c0 : c1 + 1].sum())
                    if covered == 0 or covered == total_on:
                        continue
                    if preserve_marker:
                        if covered == area:
                            continue
                    else:
                        if covered != area:
                            continue
                    rects.append((r0, c0, r1, c1))
    return rects


def _make_mask_and_blocker(h: int, w: int, preserve_marker: bool, rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    strategies = ["bars", "growth"] if preserve_marker else ["full", "bars", "growth"]
    for _ in range(MASK_ATTEMPTS):
        strategy = rng.choice(strategies)
        if strategy == "full":
            mask = np.ones((h, w), dtype=bool)
        elif strategy == "bars":
            mask = _bar_mask(h, w, rng)
        else:
            mask = _random_growth_mask(h, w, rng)
        if not _is_connected(mask) or _is_rect_border_mask(mask):
            continue
        rects = _candidate_blockers(mask, preserve_marker)
        if rects:
            return mask, rng.choice(rects)
    raise RuntimeError("failed to build a valid template mask")


def _build_pair(
    h: int,
    w: int,
    pattern_color: int,
    blocker_color: int,
    preserve_marker: bool,
    rng: random.Random,
) -> dict[str, np.ndarray]:
    mask, rect = _make_mask_and_blocker(h, w, preserve_marker, rng)

    template = np.full((h + 2, w + 2), BG, dtype=int)
    template[0, :] = FRAME
    template[-1, :] = FRAME
    template[:, 0] = FRAME
    template[:, -1] = FRAME
    template[1:-1, 1:-1][mask] = pattern_color

    instance = template.copy()
    r0, c0, r1, c1 = rect
    instance[r0 + 1 : r1 + 2, c0 + 1 : c1 + 2] = blocker_color

    repaired = template.copy()
    blocker = _mask_from_rect(mask.shape, rect)
    repaired[1:-1, 1:-1][blocker & ~mask] = blocker_color

    return {"template": template, "instance": instance, "repaired": repaired}


def _place_entries(
    entries: list[dict[str, object]], rng: random.Random
) -> tuple[np.ndarray, np.ndarray] | None:
    total_area = sum(int(np.asarray(entry["patch"]).size) for entry in entries)
    max_h = max(int(np.asarray(entry["patch"]).shape[0]) for entry in entries)
    max_w = max(int(np.asarray(entry["patch"]).shape[1]) for entry in entries)
    base_side = max(18, int(math.ceil(math.sqrt(total_area * 1.9))), max_h + 2, max_w + 2)
    if base_side > 30:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h = rng.randint(base_side, 30)
        min_w = max(base_side - 4, max_w + 2, int(math.ceil((total_area * 1.35) / h)))
        if min_w > 30:
            continue
        w = rng.randint(min_w, 30)

        grid = np.full((h, w), BG, dtype=int)
        expected = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(entries)))
        rng.shuffle(order)
        placed = [False] * len(entries)

        for idx in order:
            patch = np.asarray(entries[idx]["patch"], dtype=int)
            result = np.asarray(entries[idx]["result"], dtype=int)
            ph, pw = patch.shape
            positions: list[tuple[int, int]] = []
            for r0 in range(0, h - ph + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + ph + 1)
                for c0 in range(0, w - pw + 1):
                    cc0 = max(0, c0 - 1)
                    cc1 = min(w, c0 + pw + 1)
                    if not occupied[rr0:rr1, cc0:cc1].any():
                        positions.append((r0, c0))
            if not positions:
                break
            r0, c0 = rng.choice(positions)
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            expected[r0 : r0 + ph, c0 : c0 + pw] = result
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed[idx] = True

        if not all(placed):
            continue
        if _most_common_color(grid) != BG:
            continue
        return grid, expected

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3])
    colors = list(PALETTE)
    rng.shuffle(colors)
    if len(colors) < 2 * pair_count:
        return None

    pattern_colors = colors[:pair_count]
    blocker_colors = colors[pair_count : 2 * pair_count]

    if pair_count == 2:
        branches = [False, True]
    else:
        branches = [False, True, rng.choice([False, True])]
    rng.shuffle(branches)

    entries: list[dict[str, object]] = []
    for idx in range(pair_count):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        pair = _build_pair(
            inner_h,
            inner_w,
            pattern_colors[idx],
            blocker_colors[idx],
            branches[idx],
            rng,
        )
        entries.append({"patch": pair["template"], "result": np.full_like(pair["template"], BG)})
        entries.append({"patch": pair["instance"], "result": pair["repaired"]})

    placed = _place_entries(entries, rng)
    if placed is None:
        return None
    grid, expected = placed
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
