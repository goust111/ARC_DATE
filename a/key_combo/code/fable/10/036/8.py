from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    if connectivity == 8:
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
    raise ValueError(f"unsupported connectivity: {connectivity}")


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _frame_candidate(
    grid: np.ndarray, coords: np.ndarray, bg: int
) -> tuple[int, int, int, int] | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != bg):
        return None
    return r0, c0, r1, c1


def _find_reference_frame(
    grid: np.ndarray, bg: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components(grid != bg, connectivity=8)
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for comp in comps:
        box = _frame_candidate(grid, comp, bg)
        if box is not None:
            candidates.append((comp, box))
    assert len(candidates) == 1, "expected exactly one hollow reference frame"
    return candidates[0]


def _template_from_frame(
    grid: np.ndarray,
    frame_coords: np.ndarray,
    frame_box: tuple[int, int, int, int],
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], set[int]]:
    del frame_coords
    r0, c0, r1, c1 = frame_box
    inside_mask = np.zeros_like(grid, dtype=bool)
    inside_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    template_mask = inside_mask & (grid != bg)
    template_coords = np.argwhere(template_mask)
    assert len(template_coords) > 0, "frame must contain a template sprite"
    template_box = _bbox(template_coords)
    template = _crop(grid, template_box)
    colors = {int(v) for v in np.unique(template) if int(v) != bg}
    assert len(colors) == 2, "template sprite must use exactly two colors"
    return template, template_box, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each outside top fragment by decoding its two-color translation from the boxed template sprite.

    Input structure:
    - A unique hollow frame encloses one complete sprite that uses exactly two non-background colors.
    - Outside the frame, each other sprite keeps only the top few complete rows of that same template shape.
    - Every outside fragment is recolored with its own two colors, and the visible prefix contains at least one cell from each template role color.

    Rule:
    - Detect the boxed full sprite and crop its bounding box as the template.
    - For each outside fragment, align it with the template by top row and compare every visible cell in the fragment prefix with the corresponding template cell.
    - Use those aligned cells to infer the unique mapping from the template's two colors to that fragment's two colors.
    - Translate every remaining template row through that mapping and write the completed sprite back in place, while leaving the boxed template and the frame unchanged.

    Key insight:
    - The boxed sprite provides the role layout, while each fragment's surviving top rows are only a codebook that tells how the two template roles are recolored in that fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_coords, frame_box = _find_reference_frame(g, bg)
    template, template_box, template_colors = _template_from_frame(g, frame_coords, frame_box, bg)
    template_h, template_w = template.shape

    non_bg_components = _components(g != bg, connectivity=8)
    fr0, fc0, fr1, fc1 = frame_box
    inside_components = [
        comp
        for comp in non_bg_components
        if np.all(
            (comp[:, 0] > fr0)
            & (comp[:, 0] < fr1)
            & (comp[:, 1] > fc0)
            & (comp[:, 1] < fc1)
        )
    ]
    assert len(inside_components) == 1, "expected a single boxed template sprite"

    fragments = [
        comp
        for comp in non_bg_components
        if not np.all(
            (comp[:, 0] >= fr0)
            & (comp[:, 0] <= fr1)
            & (comp[:, 1] >= fc0)
            & (comp[:, 1] <= fc1)
        )
    ]
    assert fragments, "expected at least one outside fragment"

    out = g.copy()
    for comp in fragments:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        frag = _crop(g, box)
        frag_h, frag_w = frag.shape
        assert frag_w == template_w, "fragment width must match the template width"
        assert 1 <= frag_h < template_h, "fragment must keep a strict top prefix of the template"

        mapping: dict[int, int] = {}
        for rr in range(frag_h):
            for cc in range(template_w):
                templ_color = int(template[rr, cc])
                frag_color = int(frag[rr, cc])
                if templ_color == bg:
                    assert frag_color == bg, "fragment must preserve template holes in visible rows"
                    continue
                assert frag_color != bg, "visible fragment rows must keep every visible template cell"
                if templ_color in mapping:
                    assert mapping[templ_color] == frag_color, "fragment recoloring must be consistent"
                else:
                    mapping[templ_color] = frag_color

        assert set(mapping) == template_colors, "visible rows must decode both template colors"

        translated = np.full_like(template, bg)
        for templ_color, frag_color in mapping.items():
            translated[template == templ_color] = frag_color
        out[r0 : r0 + template_h, c0 : c0 + template_w] = translated

    return out


def _interval_row(width: int, left: int, right: int) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    row[left : right + 1] = True
    return row


def _generate_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    rows: list[tuple[int, int]] = [(0, width - 1)]
    mask[0, :] = True

    for r in range(1, height):
        prev_left, prev_right = rows[-1]
        prev_width = prev_right - prev_left + 1
        min_width = 3 if r == 1 else 2
        low_width = max(min_width, prev_width - 2)
        high_width = min(width, prev_width + 1)
        if low_width > high_width:
            return None
        row_width = int(rng.integers(low_width, high_width + 1))

        low_left = max(0, prev_left - 1, prev_left - row_width + 1)
        high_left = min(width - row_width, prev_right + 1)
        if low_left > high_left:
            return None
        left = int(rng.integers(low_left, high_left + 1))
        right = left + row_width - 1
        rows.append((left, right))
        mask[r] = _interval_row(width, left, right)

    carved = 0
    for _ in range(int(rng.integers(1, 4))):
        candidate_rows = [
            r
            for r in range(1, height)
            if np.count_nonzero(mask[r]) >= 5
        ]
        if not candidate_rows:
            break
        rr = int(rng.choice(candidate_rows))
        cols = np.flatnonzero(mask[rr])
        valid = [int(c) for c in cols[1:-1] if mask[rr, c - 1] and mask[rr, c + 1]]
        if not valid:
            continue
        cc = int(rng.choice(valid))
        mask[rr, cc] = False
        if len(_components(mask[: rr + 1], connectivity=8)) != 1:
            mask[rr, cc] = True
            continue
        carved += 1

    if len(_components(mask, connectivity=8)) != 1:
        return None
    if not any(np.count_nonzero(mask[r]) < width for r in range(1, height)):
        return None
    if len({tuple(mask[r].tolist()) for r in range(height)}) < 3:
        return None
    if carved == 0 and all(np.count_nonzero(mask[r]) >= width - 1 for r in range(1, height)):
        return None
    return mask


def _role_line(length: int, rng: np.random.Generator, require_both: bool) -> np.ndarray | None:
    if length <= 0:
        return np.zeros(0, dtype=int)
    if require_both and length < 2:
        return None

    values: list[int] = []
    current = int(rng.integers(2))
    while len(values) < length:
        run = int(rng.integers(1, min(3, length - len(values)) + 1))
        values.extend([current] * run)
        current = 1 - current

    line = np.array(values[:length], dtype=int)
    if require_both and len(set(line.tolist())) < 2:
        pivot = int(rng.integers(1, length))
        line[pivot:] = 1 - line[0]
    return line


def _generate_template(
    height: int,
    width: int,
    rng: np.random.Generator,
    colors: tuple[int, int],
) -> np.ndarray | None:
    mask = _generate_mask(height, width, rng)
    if mask is None:
        return None

    role_grid = np.full((height, width), BG, dtype=int)
    role_grid[0, mask[0]] = 0
    later_both_rows = {1, int(rng.integers(2, height))}
    if height > 5:
        later_both_rows.add(int(rng.integers(2, height)))

    for r in range(1, height):
        cols = np.flatnonzero(mask[r])
        require_both = r in later_both_rows or (r == height - 1)
        line = _role_line(len(cols), rng, require_both=require_both)
        if line is None:
            return None
        role_grid[r, cols] = line

    template = np.full((height, width), BG, dtype=int)
    c0, c1 = colors
    template[role_grid == 0] = c0
    template[role_grid == 1] = c1

    visible_two_colors = {int(v) for v in np.unique(template[1:]) if int(v) != BG}
    if visible_two_colors != set(colors):
        return None
    if any(np.count_nonzero(template[r]) == 0 for r in range(height)):
        return None
    if len({tuple(template[r].tolist()) for r in range(height)}) < 4:
        return None
    return template


def _prefix_heights(height: int, count: int, rng: np.random.Generator) -> list[int]:
    choices = list(range(2, height))
    assert len(choices) >= count
    if count == 2:
        low = int(rng.integers(2, max(3, height // 2 + 1)))
        high = int(rng.integers(max(low + 1, height // 2 + 1), height))
        return [low, high]
    picks = rng.choice(choices, size=count, replace=False)
    return sorted(int(v) for v in picks.tolist())


def _full_translation(template: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    colors = sorted(int(v) for v in np.unique(template) if int(v) != BG)
    mapping = {colors[0]: pair[0], colors[1]: pair[1]}
    out = np.full_like(template, BG)
    for src, dst in mapping.items():
        out[template == src] = dst
    return out


def _reserve_ok(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int
) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    H, W = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_box_and_fragments(
    grid_shape: tuple[int, int],
    template_shape: tuple[int, int],
    prefix_heights: list[int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    H, W = grid_shape
    h, w = template_shape
    occupied = np.zeros((H, W), dtype=bool)

    box_h, box_w = h + 4, w + 4
    box_spots = [
        (r, c)
        for r in range(H - box_h + 1)
        for c in range(W - box_w + 1)
    ]
    rng.shuffle(box_spots)
    if not box_spots:
        return None
    box_r, box_c = box_spots[0]
    _reserve(occupied, box_r, box_c, box_h, box_w, padding=2)

    fragment_spots: list[tuple[int, int]] = []
    for prefix_h in prefix_heights:
        del prefix_h
        options = [
            (r, c)
            for r in range(H - h + 1)
            for c in range(W - w + 1)
            if _reserve_ok(occupied, r, c, h, w, padding=2)
        ]
        if not options:
            return None
        rng.shuffle(options)
        choice = options[0]
        fragment_spots.append(choice)
        _reserve(occupied, choice[0], choice[1], h, w, padding=2)
    return (box_r, box_c), fragment_spots


def _semantic_checks(
    inp: np.ndarray,
    template: np.ndarray,
    fragment_specs: list[dict],
) -> bool:
    if _most_common_color(inp) != BG:
        return False

    try:
        out = main(inp)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False

    solved_fragments = []
    for spec in fragment_specs:
        r, c = spec["pos"]
        h, w = template.shape
        solved = out[r : r + h, c : c + w]
        expected = spec["full"]
        if not np.array_equal(solved, expected):
            return False

        prefix_h = int(spec["prefix_h"])
        if prefix_h >= h:
            return False
        hidden = expected[prefix_h:]
        hidden_colors = {int(v) for v in np.unique(hidden) if int(v) != BG}
        if len(hidden_colors) < 2:
            return False
        if all(np.array_equal(expected[row], expected[prefix_h - 1]) for row in range(prefix_h, h)):
            return False

        repeated = np.repeat(expected[prefix_h - 1 : prefix_h], h - prefix_h, axis=0)
        naive = expected.copy()
        naive[prefix_h:] = repeated
        if np.array_equal(naive, expected):
            return False

        swapped = _full_translation(template, (spec["pair"][1], spec["pair"][0]))
        if np.array_equal(swapped, expected):
            return False
        solved_fragments.append(expected.tobytes())

    if len(set(solved_fragments)) != len(solved_fragments):
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_h = int(rng.integers(5, 9))
    template_w = int(rng.integers(6, 11))
    fragment_count = int(rng.integers(2, 4))

    usable_colors = [c for c in ARC_COLORS if c not in (BG, FRAME_COLOR)]
    template_colors = tuple(int(v) for v in rng.choice(usable_colors, size=2, replace=False).tolist())
    template = _generate_template(template_h, template_w, rng, template_colors)
    if template is None:
        return None

    prefix_heights = _prefix_heights(template_h, fragment_count, rng)
    for prefix_h in prefix_heights:
        visible_colors = {int(v) for v in np.unique(template[:prefix_h]) if int(v) != BG}
        hidden_colors = {int(v) for v in np.unique(template[prefix_h:]) if int(v) != BG}
        if visible_colors != set(template_colors) or hidden_colors != set(template_colors):
            return None

    remaining = [c for c in usable_colors if c not in template_colors]
    ordered_pairs = list(itertools.permutations(remaining, 2))
    if len(ordered_pairs) < fragment_count:
        return None
    chosen_pair_ids = rng.choice(len(ordered_pairs), size=fragment_count, replace=False)
    fragment_pairs = [ordered_pairs[int(i)] for i in chosen_pair_ids.tolist()]

    min_h = max(template_h + 8, 18)
    min_w = max(template_w * 2 + 8, 18)
    grid_h = int(rng.integers(min_h, 31))
    grid_w = int(rng.integers(min_w, 31))
    placed = _place_box_and_fragments((grid_h, grid_w), template.shape, prefix_heights, rng)
    if placed is None:
        return None
    box_pos, fragment_positions = placed

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    box_r, box_c = box_pos
    box_h, box_w = template_h + 4, template_w + 4
    inp[box_r, box_c : box_c + box_w] = FRAME_COLOR
    inp[box_r + box_h - 1, box_c : box_c + box_w] = FRAME_COLOR
    inp[box_r : box_r + box_h, box_c] = FRAME_COLOR
    inp[box_r : box_r + box_h, box_c + box_w - 1] = FRAME_COLOR
    inp[box_r + 2 : box_r + 2 + template_h, box_c + 2 : box_c + 2 + template_w] = template

    fragment_specs: list[dict] = []
    for prefix_h, pair, (r, c) in zip(prefix_heights, fragment_pairs, fragment_positions):
        full = _full_translation(template, pair)
        inp[r : r + prefix_h, c : c + template_w] = full[:prefix_h]
        fragment_specs.append(
            {
                "prefix_h": int(prefix_h),
                "pair": tuple(int(v) for v in pair),
                "pos": (int(r), int(c)),
                "full": full,
            }
        )

    if not _semantic_checks(inp, template, fragment_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-translation puzzle")
