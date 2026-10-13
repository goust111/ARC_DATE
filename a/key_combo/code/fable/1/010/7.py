import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PALETTE = list(range(1, 10))
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _normalize_pattern(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    if coords.size == 0:
        return ()
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _sprite_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True

    hole_parts = []
    for part in _components(~mask, connectivity=4):
        if np.any(part[:, 0] == 0) or np.any(part[:, 0] == mask.shape[0] - 1):
            continue
        if np.any(part[:, 1] == 0) or np.any(part[:, 1] == mask.shape[1] - 1):
            continue
        hole_parts.append(part)

    if hole_parts:
        holes = np.concatenate(hole_parts, axis=0) + np.array([r0, c0], dtype=int)
    else:
        holes = np.zeros((0, 2), dtype=int)

    return {
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "holes": holes,
        "hole_signature": _normalize_pattern(holes),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})

    marker_candidates = []
    sprites = []
    for color in colors:
        mask = g == color
        comps8 = _components(mask, connectivity=8)
        if comps8 and all(part.shape[0] == 1 for part in comps8):
            marker_candidates.append((int(color), np.concatenate(comps8, axis=0)))
            continue

        comps4 = _components(mask, connectivity=4)
        for part in comps4:
            if part.shape[0] > 1:
                sprites.append(_sprite_info(g, int(color), part))

    assert len(marker_candidates) == 1, "expected exactly one singleton-marker color"
    marker_color, marker_coords = marker_candidates[0]
    assert marker_coords.shape[0] >= 2, "marker pattern must contain multiple points"
    assert sprites, "expected at least one sprite"
    return {
        "bg": int(bg),
        "marker_color": int(marker_color),
        "marker_coords": marker_coords,
        "marker_signature": _normalize_pattern(marker_coords),
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the uniquely keyed sprite and refill its enclosed holes with the singleton marker color.

    Input structure:
    - The background is dominant.
    - Several differently colored sprite bodies appear as 4-connected components; each sprite encloses one or more background holes.
    - One extra color appears only as isolated single pixels on empty ground, and the whole set of those pixels forms a translated shape.

    Rule:
    - Collect all singleton marker pixels as one coordinate set.
    - For each sprite, collect every enclosed hole cell inside that sprite and compare the full hole-cell set up to translation only.
    - Exactly one sprite has a hole pattern whose translated copy matches the marker set; crop that sprite to its tight bbox and repaint all of its hole cells with the marker color.

    Key insight:
    - The scattered singletons are not counters. They are a translated imprint of one sprite's internal hole geometry, so matching the full hole pattern is required and counting hole cells alone is insufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _analyze_grid(g)
    marker_count = int(info["marker_coords"].shape[0])
    matches = [
        sprite
        for sprite in info["sprites"]
        if sprite["holes"].shape[0] == marker_count and sprite["hole_signature"] == info["marker_signature"]
    ]
    assert len(matches) == 1, "hole pattern must identify exactly one sprite"
    target = matches[0]

    out = _crop_bbox(g, target["bbox"])
    r0, c0, _r1, _c1 = target["bbox"]
    for r, c in target["holes"]:
        out[int(r - r0), int(c - c0)] = info["marker_color"]
    return out


def _render_sprite(height: int, width: int, color: int, bg: int, holes: list[tuple[int, int]]) -> np.ndarray:
    sprite = np.full((height, width), color, dtype=int)
    for r, c in holes:
        sprite[r, c] = bg
    return sprite


def _sample_spaced_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []

    def search(start: int) -> bool:
        if len(chosen) == count:
            rows = {r for r, _ in chosen}
            cols = {c for _, c in chosen}
            return len(rows) >= 2 and len(cols) >= 2
        need = count - len(chosen)
        if len(cells) - start < need:
            return False
        for idx in range(start, len(cells)):
            r, c = cells[idx]
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            if search(idx + 1):
                return True
            chosen.pop()
        return False

    return chosen.copy() if search(0) else None


def _sample_connected_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    if count > (height - 2) * (width - 2):
        return None
    inner_h = height - 2
    inner_w = width - 2
    start = (random.randrange(inner_h), random.randrange(inner_w))
    chosen = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        rr, cc = cell
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(chosen) < count and frontier:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        chosen.add(nxt)
        add_frontier(nxt)

    if len(chosen) != count:
        return None
    return sorted((r + 1, c + 1) for r, c in chosen)


def _sample_any_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    if len(cells) < count:
        return None
    return sorted(random.sample(cells, count))


def _sample_holes(
    height: int,
    width: int,
    count: int,
    *,
    style: str,
    forbidden: set[tuple[tuple[int, int], ...]] | None = None,
) -> list[tuple[int, int]] | None:
    forbidden = set() if forbidden is None else forbidden
    for _ in range(120):
        if style == "spaced":
            holes = _sample_spaced_cells(height, width, count)
        elif style == "connected":
            holes = _sample_connected_cells(height, width, count)
        elif style == "adjacent":
            holes = _sample_any_cells(height, width, count)
            if holes is not None:
                has_adjacent = False
                hole_set = set(holes)
                for r, c in holes:
                    for dr, dc in DIR4:
                        if (r + dr, c + dc) in hole_set:
                            has_adjacent = True
                            break
                    if has_adjacent:
                        break
                if not has_adjacent:
                    holes = None
        else:
            raise ValueError("unknown hole style")

        if holes is None:
            continue
        signature = _normalize_pattern(np.asarray(holes, dtype=int))
        if signature in forbidden:
            continue
        return holes
    return None


def _place_rectangles(
    canvas_h: int,
    canvas_w: int,
    sizes: list[tuple[int, int]],
    pad: int = 1,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    def search(pos: int, blocked: np.ndarray, placements: list[tuple[int, int] | None]) -> list[tuple[int, int]] | None:
        if pos == len(order):
            return [placements[idx] for idx in range(len(sizes))]

        idx = order[pos]
        height, width = sizes[idx]
        candidates = []
        for r0 in range(canvas_h - height + 1):
            for c0 in range(canvas_w - width + 1):
                rr0 = max(0, r0 - pad)
                cc0 = max(0, c0 - pad)
                rr1 = min(canvas_h, r0 + height + pad)
                cc1 = min(canvas_w, c0 + width + pad)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0, rr0, cc0, rr1, cc1))
        random.shuffle(candidates)
        for r0, c0, rr0, cc0, rr1, cc1 in candidates:
            nxt = blocked.copy()
            nxt[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            result = search(pos + 1, nxt, placements)
            if result is not None:
                return result
            placements[idx] = None
        return None

    return search(0, np.zeros((canvas_h, canvas_w), dtype=bool), [None] * len(sizes))


def _marker_candidates(
    grid: np.ndarray,
    pattern: tuple[tuple[int, int], ...],
    bg: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    max_r = max(r for r, _ in pattern)
    max_c = max(c for _, c in pattern)
    pattern_set = set(pattern)
    candidates = []
    for top in range(1, h - max_r - 1):
        for left in range(1, w - max_c - 1):
            cells = [(top + r, left + c) for r, c in pattern]
            if any(grid[r, c] != bg for r, c in cells):
                continue
            ok = True
            for r, c in cells:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if (nr - top, nc - left) in pattern_set:
                            continue
                        if grid[nr, nc] != bg:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append((top, left))
    return candidates


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    sprite_count = random.randint(3, 4)
    colors = random.sample(PALETTE, sprite_count + 1)
    marker_color = colors[0]
    sprite_colors = colors[1:]

    target_h = random.randint(6, 9)
    target_w = random.randint(6, 9)
    max_spaced = ((target_h - 2) + 1) // 2 * (((target_w - 2) + 1) // 2)
    hole_count = random.randint(3, min(6, max_spaced))
    target_holes = _sample_holes(target_h, target_w, hole_count, style="spaced")
    assert target_holes is not None
    target_signature = _normalize_pattern(np.asarray(target_holes, dtype=int))
    target_patch = _render_sprite(target_h, target_w, sprite_colors[0], bg, target_holes)
    expected = target_patch.copy()
    expected[expected == bg] = marker_color

    distractor_h = random.randint(6, 9)
    distractor_w = random.randint(6, 9)
    if (distractor_h - 2) * (distractor_w - 2) < hole_count:
        distractor_h = max(distractor_h, 5 + hole_count // 2)
        distractor_w = max(distractor_w, 5 + hole_count // 2)
        distractor_h = min(distractor_h, 10)
        distractor_w = min(distractor_w, 10)
    distractor_holes = _sample_holes(
        distractor_h,
        distractor_w,
        hole_count,
        style=random.choice(["connected", "adjacent"]),
        forbidden={target_signature},
    )
    assert distractor_holes is not None
    distractor_patch = _render_sprite(distractor_h, distractor_w, sprite_colors[1], bg, distractor_holes)

    patches = [target_patch, distractor_patch]
    forbidden = {target_signature, _normalize_pattern(np.asarray(distractor_holes, dtype=int))}
    used_counts = {hole_count}
    for idx in range(2, sprite_count):
        h = random.randint(5, 9)
        w = random.randint(5, 9)
        possible_counts = [value for value in range(1, min(7, (h - 2) * (w - 2)) + 1) if value not in used_counts]
        if not possible_counts:
            possible_counts = list(range(1, min(7, (h - 2) * (w - 2)) + 1))
        count = random.choice(possible_counts)
        style = random.choice(["connected", "adjacent", "spaced"])
        holes = _sample_holes(h, w, count, style=style, forbidden=forbidden)
        if holes is None:
            holes = _sample_holes(h, w, count, style="connected", forbidden=forbidden)
        assert holes is not None
        patches.append(_render_sprite(h, w, sprite_colors[idx], bg, holes))
        forbidden.add(_normalize_pattern(np.asarray(holes, dtype=int)))
        used_counts.add(count)

    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    canvas_min = max(18, int(np.ceil(np.sqrt(total_area * 2.4))))
    canvas_h = random.randint(canvas_min, 30)
    canvas_w = random.randint(canvas_min, 30)
    sizes = [tuple(map(int, patch.shape)) for patch in patches]
    placements = _place_rectangles(canvas_h, canvas_w, sizes, pad=1)
    assert placements is not None

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    for patch, (r0, c0) in zip(patches, placements):
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    marker_pattern = _normalize_pattern(np.asarray(target_holes, dtype=int))
    options = _marker_candidates(grid, marker_pattern, bg)
    assert options
    top, left = random.choice(options)
    for r, c in marker_pattern:
        grid[top + r, left + c] = marker_color

    return grid, expected


def _case_is_valid(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if actual.shape != expected.shape:
        return False
    if actual.shape[0] > 30 or actual.shape[1] > 30:
        return False

    bg = info["bg"]
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[values == bg][0])
    if bg_count != int(counts.max()) or int(np.count_nonzero(counts == counts.max())) != 1:
        return False

    marker_count = int(info["marker_coords"].shape[0])
    same_count = [sprite for sprite in info["sprites"] if sprite["holes"].shape[0] == marker_count]
    if len(same_count) < 2:
        return False

    matches = [
        sprite
        for sprite in info["sprites"]
        if sprite["holes"].shape[0] == marker_count and sprite["hole_signature"] == info["marker_signature"]
    ]
    if len(matches) != 1:
        return False

    if not any(
        sprite["hole_signature"] != info["marker_signature"] and sprite["holes"].shape[0] == marker_count
        for sprite in info["sprites"]
    ):
        return False

    marker_coords = info["marker_coords"]
    for i in range(marker_coords.shape[0]):
        for j in range(i + 1, marker_coords.shape[0]):
            if max(abs(int(marker_coords[i, 0] - marker_coords[j, 0])), abs(int(marker_coords[i, 1] - marker_coords[j, 1]))) <= 1:
                return False
        r = int(marker_coords[i, 0])
        c = int(marker_coords[i, 1])
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if nr == r and nc == c:
                    continue
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != bg:
                    return False

    def has_adjacent_hole_pair(sprite: dict) -> bool:
        holes = [tuple(map(int, cell)) for cell in sprite["holes"]]
        hole_set = set(holes)
        for r, c in holes:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in hole_set:
                    return True
        return False

    if not any(
        sprite["holes"].shape[0] == marker_count
        and sprite["hole_signature"] != info["marker_signature"]
        and has_adjacent_hole_pair(sprite)
        for sprite in info["sprites"]
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _build_case()
        except AssertionError:
            continue
        if _case_is_valid(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid hole-key puzzle")
