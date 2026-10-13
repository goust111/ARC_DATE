import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

OUTLINE_COLOR = 5
SEPARATOR_COLOR = 9
MAX_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _enclosed_background(mask: np.ndarray) -> np.ndarray:
    """Return bg cells inside a closed outline, using 4-neighbor leakage to the bbox boundary."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~mask) & (~outside)


def _scale_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=int)
    return np.kron(mask.astype(int), block).astype(bool)


def _scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=int)
    return np.kron(np.asarray(grid, dtype=int), block)


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        (r, int(grid[r, 0]))
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected a unique full-width separator row"
    return rows[0]


def _component_patch(region: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return region[r0 : r1 + 1, c0 : c1 + 1] == color


def _find_outline_color(
    grid: np.ndarray, top: np.ndarray, bottom: np.ndarray, bg: int, sep_color: int
) -> int:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, sep_color):
            continue
        top_comps = _components(top == color, connectivity=8)
        bottom_comps = _components(bottom == color, connectivity=8)
        if not top_comps or not bottom_comps:
            continue
        if not all(_enclosed_background(_component_patch(top, comp, color)).any() for comp in top_comps):
            continue
        if not all(_enclosed_background(_component_patch(bottom, comp, color)).any() for comp in bottom_comps):
            continue
        candidates.append((len(top_comps) + len(bottom_comps), color))
    assert candidates, "could not infer outline color"
    return max(candidates)[1]


def _extract_dictionary_entries(
    bottom: np.ndarray, outline_color: int, bg: int, sep_color: int
) -> list[dict]:
    comps = _components(bottom == outline_color, connectivity=8)
    assert comps, "dictionary outlines are missing"
    entries = []
    for comp in sorted(comps, key=lambda pts: _bbox(pts)[1]):
        r0, c0, r1, c1 = _bbox(comp)
        shape_mask = bottom[r0 : r1 + 1, c0 : c1 + 1] == outline_color
        inside = _enclosed_background(shape_mask)
        assert inside.any(), "dictionary outline must enclose an interior"
        h, w = shape_mask.shape
        pc0 = c1 + 1
        pc1 = pc0 + w
        assert pc1 <= bottom.shape[1], "pattern block must sit to the right of its outline"
        pattern = bottom[r0 : r1 + 1, pc0:pc1].copy()
        assert pattern.shape == (h, w)
        assert not np.any(pattern == outline_color)
        assert not np.any(pattern == sep_color)
        assert np.any(pattern != bg), "pattern block must not be empty"
        entries.append({"mask": shape_mask, "inside": inside, "pattern": pattern})
    return entries


def _match_entry(big_mask: np.ndarray, entries: list[dict]) -> tuple[dict, int]:
    bh, bw = big_mask.shape
    matches = []
    for entry in entries:
        sh, sw = entry["mask"].shape
        if bh % sh != 0 or bw % sw != 0:
            continue
        factor = bh // sh
        if factor < 2 or bw // sw != factor:
            continue
        if np.array_equal(_scale_bool(entry["mask"], factor), big_mask):
            matches.append((entry, factor))
    assert len(matches) == 1, "expected each large outline to match exactly one dictionary entry"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill large hollow outlines by matching them to a lower shape dictionary and scaling the paired pattern.

    Input structure:
    - The background is the dominant color, and one full-width horizontal separator row splits the grid into an upper query area and a lower dictionary area.
    - In the lower area, each dictionary entry is a small grey hollow outline with a same-sized colored pattern block directly to its right.
    - In the upper area, each target is a larger grey hollow outline whose shape is an integer-scaled copy of one dictionary outline.

    Rule:
    - Parse each lower outline and read the pattern block adjacent to it.
    - For every upper outline, find the unique dictionary outline whose mask matches after dividing by one common integer scale factor.
    - Scale the paired pattern by that same factor and write it only into the enclosed interior of the large outline, leaving the grey border, separator, and dictionary unchanged.

    Key insight:
    - The task has two linked steps: normalize away size to identify the right dictionary shape, then reuse the inferred scale to enlarge the associated pattern before filling the matching interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_r, sep_color = _find_separator_row(g, bg)
    top = g[:sep_r]
    bottom = g[sep_r + 1 :]
    outline_color = _find_outline_color(g, top, bottom, bg, sep_color)
    entries = _extract_dictionary_entries(bottom, outline_color, bg, sep_color)

    out = g.copy()
    top_comps = _components(top == outline_color, connectivity=8)
    assert top_comps, "expected at least one large outline"
    for comp in top_comps:
        r0, c0, r1, c1 = _bbox(comp)
        big_mask = top[r0 : r1 + 1, c0 : c1 + 1] == outline_color
        entry, factor = _match_entry(big_mask, entries)
        inside = _enclosed_background(big_mask)
        scaled_pattern = _scale_grid(entry["pattern"], factor)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[inside] = scaled_pattern[inside]
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _random_interior_cells() -> set[tuple[int, int]]:
    h = random.choice((2, 3))
    w = random.choice((2, 3))
    area = random.randint(2, min(h * w, 5))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_interior_cells()
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _outline_from_interior(cells: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in cells:
        interior[r + 1, c + 1] = True

    dilated = interior.copy()
    for r, c in np.argwhere(interior):
        r0, r1 = max(0, r - 1), min(interior.shape[0], r + 2)
        c0, c1 = max(0, c - 1), min(interior.shape[1], c + 2)
        dilated[r0:r1, c0:c1] = True

    outline = dilated & ~interior
    inside = _enclosed_background(outline)
    if not np.array_equal(inside, interior):
        return None
    if inside.sum() < 2:
        return None
    return outline, inside


def _shape_related(a: np.ndarray, b: np.ndarray) -> bool:
    if np.array_equal(a, b):
        return True
    for small, big in ((a, b), (b, a)):
        sh, sw = small.shape
        bh, bw = big.shape
        if bh % sh != 0 or bw % sw != 0:
            continue
        factor = bh // sh
        if factor >= 2 and bw // sw == factor and np.array_equal(_scale_bool(small, factor), big):
            return True
    return False


def _make_pattern(inside: np.ndarray, bg: int, color: int) -> np.ndarray:
    h, w = inside.shape
    rr, cc = np.indices((h, w))
    center_r = h // 2
    center_c = w // 2
    off_r = random.randrange(2)
    off_c = random.randrange(2)
    masks = [
        (rr + cc + off_r + off_c) % 2 == 0,
        (cc + off_c) % 2 == 0,
        (rr + off_r) % 2 == 0,
        (rr == cc) | (rr + cc == w - 1),
        np.abs(rr - center_r) + np.abs(cc - center_c) <= max(1, min(h, w) // 3),
        (rr % 2 == 0) & (cc % 2 == 0),
    ]
    random.shuffle(masks)
    for mask in masks:
        for invert in (False, True):
            use = ~mask if invert else mask
            if not use.any() or use.all():
                continue
            visible = use & inside
            if 0 < visible.sum() < int(inside.sum()):
                pattern = np.full((h, w), bg, dtype=int)
                pattern[use] = color
                return pattern

    inside_pts = list(map(tuple, np.argwhere(inside)))
    chosen = set(random.sample(inside_pts, random.randint(1, len(inside_pts) - 1)))
    pattern = np.full((h, w), bg, dtype=int)
    for r, c in chosen:
        pattern[r, c] = color
    return pattern


def _sample_entries(bg: int) -> list[dict] | None:
    n_entries = random.choice((2, 3))
    colors = [c for c in range(10) if c not in {bg, OUTLINE_COLOR, SEPARATOR_COLOR}]
    random.shuffle(colors)
    entries = []
    while len(entries) < n_entries:
        built = _outline_from_interior(_random_interior_cells())
        if built is None:
            continue
        outline, inside = built
        if any(_shape_related(outline, prev["mask"]) for prev in entries):
            continue
        color = colors[len(entries)]
        pattern = _make_pattern(inside, bg, color)
        entries.append({"mask": outline, "inside": inside, "pattern": pattern, "color": color})
    return entries


def _sample_objects(entries: list[dict]) -> list[dict]:
    count = random.choice((2, 3))
    choices = [random.randrange(len(entries)) for _ in range(count)]
    if len(entries) > 1 and len(set(choices)) == 1:
        choices[-1] = (choices[-1] + 1) % len(entries)
    objects = []
    for entry_idx in choices:
        scale = random.choice((2, 3))
        sh, sw = entries[entry_idx]["mask"].shape
        objects.append(
            {
                "entry": entry_idx,
                "scale": scale,
                "h": sh * scale,
                "w": sw * scale,
            }
        )
    random.shuffle(objects)
    return objects


def _draw_outline(canvas: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    patch = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = OUTLINE_COLOR


def _layout_width(widths: list[int], gaps: list[int]) -> int:
    return 2 + sum(widths) + sum(gaps)


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {OUTLINE_COLOR, SEPARATOR_COLOR}]
    bg = random.choice(bg_choices)
    entries = _sample_entries(bg)
    if entries is None:
        return None
    objects = _sample_objects(entries)

    dict_gaps = [random.randint(1, 2) for _ in range(len(entries) - 1)]
    top_gaps = [random.randint(1, 3) for _ in range(len(objects) - 1)]
    need_dict_w = _layout_width([2 * entry["mask"].shape[1] for entry in entries], dict_gaps)
    need_top_w = _layout_width([obj["w"] for obj in objects], top_gaps)
    need_w = max(need_dict_w, need_top_w)
    if need_w > 30:
        return None

    max_small_h = max(entry["mask"].shape[0] for entry in entries)
    max_big_h = max(obj["h"] for obj in objects)
    top_h = max_big_h + random.randint(2, 5)
    bottom_h = max_small_h + random.randint(2, 4)
    total_h = top_h + 1 + bottom_h
    if total_h > 30:
        return None

    width = need_w + random.randint(0, min(30 - need_w, 6))
    inp = np.full((total_h, width), bg, dtype=int)
    out = inp.copy()

    sep_r = top_h
    inp[sep_r, :] = SEPARATOR_COLOR
    out[sep_r, :] = SEPARATOR_COLOR

    dict_top = sep_r + 1 + random.randint(1, bottom_h - max_small_h - 1)
    left_extra = width - need_dict_w
    dict_left = 1 + random.randint(0, left_extra)
    x = dict_left
    for i, entry in enumerate(entries):
        h, w = entry["mask"].shape
        _draw_outline(inp, dict_top, x, entry["mask"])
        _draw_outline(out, dict_top, x, entry["mask"])
        inp[dict_top : dict_top + h, x + w : x + 2 * w] = entry["pattern"]
        out[dict_top : dict_top + h, x + w : x + 2 * w] = entry["pattern"]
        x += 2 * w
        if i + 1 < len(entries):
            x += dict_gaps[i]

    left_extra = width - need_top_w
    top_left = 1 + random.randint(0, left_extra)
    x = top_left
    for i, obj in enumerate(objects):
        entry = entries[obj["entry"]]
        factor = obj["scale"]
        big_mask = _scale_bool(entry["mask"], factor)
        big_pattern = _scale_grid(entry["pattern"], factor)
        y = random.randint(1, top_h - big_mask.shape[0] - 1)
        _draw_outline(inp, y, x, big_mask)
        _draw_outline(out, y, x, big_mask)
        inside = _enclosed_background(big_mask)
        patch = out[y : y + big_mask.shape[0], x : x + big_mask.shape[1]]
        patch[inside] = big_pattern[inside]
        x += big_mask.shape[1]
        if i + 1 < len(objects):
            x += top_gaps[i]

    if _bg_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
