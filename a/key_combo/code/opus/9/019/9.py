from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
MARKER = 8
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _component_coords(grid != bg):
        bbox = _bbox_from_coords(coords)
        comps.append({"coords": coords, "bbox": bbox})
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    if cells == 0:
        return False
    return len(_component_coords(mask)) == 1


def _matches_partial(
    window: np.ndarray,
    template: np.ndarray,
    support_mask: np.ndarray,
    fill_mask: np.ndarray,
    bg: int,
) -> bool:
    return bool(
        np.array_equal(window[support_mask], template[support_mask])
        and np.all(window[fill_mask] == bg)
    )


def _template_fill_color(template: np.ndarray, fill_mask: np.ndarray, bg: int) -> int:
    neighbor_colors: list[int] = []
    for r, c in np.argwhere(fill_mask):
        for dr, dc in N4:
            nr, nc = int(r) + dr, int(c) + dc
            if not _in_bounds(template.shape, nr, nc):
                continue
            color = int(template[nr, nc])
            if color not in (bg, MARKER):
                neighbor_colors.append(color)
    if neighbor_colors:
        colors = set(neighbor_colors)
        assert len(colors) == 1
        return int(neighbor_colors[0])

    support_colors = [int(v) for v in template[(template != bg) & (template != MARKER)]]
    counts = Counter(support_colors)
    color, amount = counts.most_common(1)[0]
    assert sum(v == amount for v in counts.values()) == 1
    return int(color)


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    components = _find_foreground_components(g, bg)
    marked = []
    for comp in components:
        values = {int(g[r, c]) for r, c in comp["coords"]}
        if MARKER in values:
            marked.append(comp)
    assert len(marked) == 1

    bbox = marked[0]["bbox"]
    template = _crop_bbox(g, bbox)
    assert template.shape == (3, 3)
    fill_mask = template == MARKER
    support_mask = (template != bg) & (template != MARKER)
    assert 1 <= int(fill_mask.sum()) <= 2
    assert int(support_mask.sum()) >= 4
    fill_color = _template_fill_color(template, fill_mask, bg)
    return bbox, template, support_mask, fill_mask, fill_color, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the cyan-marked sprite as a 3x3 template and complete every matching partial copy.

    Input structure:
    - A black background contains 4 to 6 separated 3x3 foreground sprites.
    - Exactly one sprite belongs to the only 4-connected component that contains cyan marker cells.
    - Inside that marked 3x3 template, cyan cells indicate the missing positions to transfer; the other non-black cells are the required support pattern.

    Rule:
    - Find the unique foreground component containing cyan, crop its 3x3 bounding box, and treat cyan positions as the fill mask.
    - Determine the completion color from the non-cyan foreground cells orthogonally adjacent to the cyan mask cells in that template.
    - Scan every other 3x3 window: if it matches the template on all non-cyan support cells and has background at every cyan-mask position, it is an incomplete copy.
    - Fill those masked positions with the completion color, while leaving the source template and all distractors unchanged.

    Key insight:
    - Cyan has a double role: it selects which sprite supplies the template and simultaneously marks which cells must be restored in the matching copies.
    """
    g = np.asarray(grid, dtype=int)
    bbox, template, support_mask, fill_mask, fill_color, bg = _extract_template(g)
    out = g.copy()
    h, w = g.shape
    src_origin = (bbox[0], bbox[1])

    for r in range(h - 2):
        for c in range(w - 2):
            if (r, c) == src_origin:
                continue
            window = g[r : r + 3, c : c + 3]
            if not _matches_partial(window, template, support_mask, fill_mask, bg):
                continue
            out[r : r + 3, c : c + 3][fill_mask] = fill_color
    return out


def _grow_connected_mask(size: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    start = (random.randrange(3), random.randrange(3))
    mask[start] = True
    while int(mask.sum()) < size:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in N4:
                nr, nc = int(r) + dr, int(c) + dc
                if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        mask[random.choice(frontier)] = True
    return mask


def _has_full_bbox(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    return _bbox_from_coords(coords) == (0, 0, 2, 2)


def _marker_candidates_inside(full_mask: np.ndarray, marker_count: int) -> list[np.ndarray]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    out: list[np.ndarray] = []
    for picks in combinations(cells, marker_count):
        support = full_mask.copy()
        for r, c in picks:
            support[r, c] = False
        if int(support.sum()) < 4 or not _is_connected(support):
            continue
        marker_mask = np.zeros((3, 3), dtype=bool)
        valid = True
        for r, c in picks:
            marker_mask[r, c] = True
            if not any(
                _in_bounds((3, 3), r + dr, c + dc) and support[r + dr, c + dc]
                for dr, dc in N4
            ):
                valid = False
                break
        if valid:
            out.append(marker_mask)
    return out


def _marker_candidates_outside(support: np.ndarray, marker_count: int) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(support)]
    support_bbox = _bbox_from_coords(coords)
    if support_bbox == (0, 0, 2, 2):
        return []

    empty_neighbors = []
    for r, c in coords:
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if _in_bounds((3, 3), nr, nc) and not support[nr, nc]:
                empty_neighbors.append((nr, nc))
    empty_neighbors = list(dict.fromkeys(empty_neighbors))

    out: list[np.ndarray] = []
    for picks in combinations(empty_neighbors, marker_count):
        marker_mask = np.zeros((3, 3), dtype=bool)
        union = support.copy()
        for r, c in picks:
            marker_mask[r, c] = True
            union[r, c] = True
        if not _has_full_bbox(union):
            continue
        if not _is_connected(union):
            continue
        out.append(marker_mask)
    return out


def _make_template_patch() -> tuple[np.ndarray, int]:
    palette = [c for c in range(1, 10) if c != MARKER]
    for _ in range(200):
        marker_count = random.choice([1, 2])
        if random.random() < 0.5:
            full_mask = _grow_connected_mask(random.randint(5, 7))
            if not _has_full_bbox(full_mask):
                continue
            candidates = _marker_candidates_inside(full_mask, marker_count)
            if not candidates:
                continue
            marker_mask = random.choice(candidates)
            support = full_mask & ~marker_mask
        else:
            support = _grow_connected_mask(random.randint(4, 6))
            candidates = _marker_candidates_outside(support, marker_count)
            if not candidates:
                continue
            marker_mask = random.choice(candidates)
            if not _has_full_bbox(support | marker_mask):
                continue

        primary = random.choice(palette)
        accent_cells = []
        neighbor_of_marker = set()
        for r, c in np.argwhere(marker_mask):
            for dr, dc in N4:
                nr, nc = int(r) + dr, int(c) + dc
                if _in_bounds((3, 3), nr, nc) and support[nr, nc]:
                    neighbor_of_marker.add((nr, nc))

        accent_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(support)
            if (int(r), int(c)) not in neighbor_of_marker
        ]
        use_accent = bool(accent_candidates) and random.random() < 0.5
        accent = None
        if use_accent:
            accent = random.choice([c for c in palette if c != primary])
            random.shuffle(accent_candidates)
            accent_cells = accent_candidates[: random.randint(1, min(2, len(accent_candidates)))]

        patch = np.zeros((3, 3), dtype=int)
        patch[support] = primary
        for r, c in accent_cells:
            patch[r, c] = accent
        patch[marker_mask] = MARKER

        if not _is_connected(patch != BG):
            continue
        if _template_fill_color(patch, marker_mask, BG) != primary:
            continue
        return patch, primary
    raise RuntimeError("failed to create template patch")


def _partial_patch(template: np.ndarray) -> np.ndarray:
    patch = template.copy()
    patch[patch == MARKER] = BG
    return patch


def _support_mismatch_distractor(template: np.ndarray, primary: int) -> np.ndarray:
    patch = _partial_patch(template)
    support_cells = [tuple(map(int, rc)) for rc in np.argwhere((template != BG) & (template != MARKER))]
    r, c = random.choice(support_cells)
    current = int(patch[r, c])
    options = [color for color in range(1, 10) if color not in (MARKER, current)]
    patch[r, c] = random.choice(options)
    return patch


def _fill_mismatch_distractor(template: np.ndarray, primary: int) -> np.ndarray:
    patch = _partial_patch(template)
    fill_cells = [tuple(map(int, rc)) for rc in np.argwhere(template == MARKER)]
    wrong = random.choice([color for color in range(1, 10) if color not in (MARKER, primary)])
    count = random.randint(1, len(fill_cells))
    random.shuffle(fill_cells)
    for r, c in fill_cells[:count]:
        patch[r, c] = wrong
    return patch


def _patches_are_valid(patches: list[np.ndarray]) -> bool:
    return all(_is_connected(patch != BG) for patch in patches)


def _slot_origins(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        offset_r = random.randint(0, 2)
        offset_c = random.randint(0, 2)
        rows = list(range(offset_r, h - 2, 4))
        cols = list(range(offset_c, w - 2, 4))
        slots = [(r, c) for r in rows for c in cols]
        if len(slots) < count:
            continue
        return random.sample(slots, count)
    return None


def _detect_partial_origins(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    bbox, template, support_mask, fill_mask, _fill_color, bg = _extract_template(grid)
    found: set[tuple[int, int]] = set()
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if (r, c) == (bbox[0], bbox[1]):
                continue
            window = grid[r : r + 3, c : c + 3]
            if _matches_partial(window, template, support_mask, fill_mask, bg):
                found.add((r, c))
    return (bbox[0], bbox[1]), found


def _count_objects(grid: np.ndarray) -> int:
    return len(_find_foreground_components(grid, BG))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    template, primary = _make_template_patch()

    partial_count = random.choice([2, 3])
    distractor_count = random.choice([1, 2])
    total = 1 + partial_count + distractor_count
    origins = _slot_origins(height, width, total)
    if origins is None:
        return None
    random.shuffle(origins)
    template_origin = origins[0]
    partial_origins = origins[1 : 1 + partial_count]
    distractor_origins = origins[1 + partial_count :]

    patches = [template]
    patches.extend(_partial_patch(template) for _ in range(partial_count))

    distractors = []
    if distractor_count == 1:
        if random.random() < 0.5:
            distractors.append(_support_mismatch_distractor(template, primary))
        else:
            distractors.append(_fill_mismatch_distractor(template, primary))
    else:
        distractors.append(_support_mismatch_distractor(template, primary))
        distractors.append(_fill_mismatch_distractor(template, primary))
    patches.extend(distractors)

    if not _patches_are_valid(patches):
        return None

    grid = np.zeros((height, width), dtype=int)
    placement = [(template_origin, template)]
    placement.extend((origin, _partial_patch(template)) for origin in partial_origins)
    placement.extend((origin, patch) for origin, patch in zip(distractor_origins, distractors))
    for (r, c), patch in placement:
        grid[r : r + 3, c : c + 3] = patch

    src_origin, detected_partials = _detect_partial_origins(grid)
    intended_partials = set(partial_origins)
    if src_origin != template_origin:
        return None
    if detected_partials != intended_partials:
        return None
    if _count_objects(grid) != total:
        return None

    out = main(grid)
    for r, c in partial_origins:
        if np.any(out[r : r + 3, c : c + 3] == MARKER):
            return None
        if not np.array_equal(out[r : r + 3, c : c + 3][template == MARKER], np.full(int(np.sum(template == MARKER)), primary)):
            return None
    for r, c in distractor_origins:
        if not np.array_equal(out[r : r + 3, c : c + 3], grid[r : r + 3, c : c + 3]):
            return None
    if not np.array_equal(out[template_origin[0] : template_origin[0] + 3, template_origin[1] : template_origin[1] + 3], template):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
