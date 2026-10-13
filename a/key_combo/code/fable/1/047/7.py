import random

import numpy as np


BG = 0
GRAY = 5
ORANGE = 7
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    """Most common color in the grid."""
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    """4-connected multicolor components using the ARC (r, c) convention."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _palette_from_top_row(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    cols = np.flatnonzero(grid[0] != bg).tolist()
    assert cols, "expected at least one palette slot on the top row"
    return [int(grid[0, c]) for c in cols], cols


def _body_start_row(grid: np.ndarray, bg: int) -> int:
    for r in range(1, grid.shape[0]):
        if np.any(grid[r] != bg):
            assert r >= 2, "expected an empty separator row below the palette"
            return int(r)
    raise AssertionError("expected a foreground body below the palette")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked object by indexing the top palette with its signed marker count.

    Input structure:
    - The top row contains an ordered palette of isolated color slots, separated from the body by at least one empty row.
    - Below the separator, the foreground consists of several separated 4-connected objects sharing one base color.
    - Some cells inside each object are replaced by gray and orange marker pixels.

    Rule:
    - Read the palette colors from left to right along the top row and keep that row unchanged.
    - For every body object, compute k = (number of gray markers) - (number of orange markers).
    - If k > 0, repaint the entire object, including marker cells, with the kth palette color from the left; if k = 0, erase the whole object to background.

    Key insight:
    - Gray and orange markers act as positive and negative signs rather than two independent decorations, so mixed-marker objects must be algebraically cancelled before the palette lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    palette, _palette_cols = _palette_from_top_row(g, bg)
    body_start = _body_start_row(g, bg)

    out = g.copy()
    components = find_components(g[body_start:], bg=bg)
    assert components, "expected at least one body object"

    for comp in components:
        coords = comp["coords"].copy()
        coords[:, 0] += body_start
        values = g[coords[:, 0], coords[:, 1]]
        gray_count = int(np.count_nonzero(values == GRAY))
        orange_count = int(np.count_nonzero(values == ORANGE))
        k = gray_count - orange_count
        assert 0 <= k <= len(palette), "signed marker count must index the palette or erase"
        if k == 0:
            out[coords[:, 0], coords[:, 1]] = bg
        else:
            out[coords[:, 0], coords[:, 1]] = int(palette[k - 1])
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen[start] = True
    reached = 0
    while stack:
        r, c = stack.pop()
        reached += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return reached == int(mask.sum())


def _random_object_mask(min_area: int) -> np.ndarray:
    for _ in range(100):
        h = random.randint(3, 6)
        w_min = max(4, (min_area + h - 1) // h)
        if w_min > 7:
            continue
        w = random.randint(w_min, 7)
        mask = np.ones((h, w), dtype=bool)

        boundary = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        random.shuffle(boundary)
        max_removals = min(4, int(mask.sum()) - min_area)
        removals = random.randint(0, max_removals)
        removed = 0
        for r, c in boundary:
            if removed >= removals:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) >= min_area and _is_connected(trial):
                mask = trial
                removed += 1

        if int(mask.sum()) >= min_area and _is_connected(mask):
            return mask
    raise RuntimeError("failed to generate a connected object mask")


def _marker_slots(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    rich = []
    for r, c in coords:
        degree = sum(1 for nr, nc in neighbors4(r, c) if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc])
        if degree >= 2:
            rich.append((r, c))
    pool = rich if len(rich) >= count else coords
    return random.sample(pool, count)


def _render_object(mask: np.ndarray, base_color: int, gray_count: int, orange_count: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(base_color)
    slots = _marker_slots(mask, gray_count + orange_count)
    for r, c in slots[:gray_count]:
        patch[r, c] = GRAY
    for r, c in slots[gray_count:]:
        patch[r, c] = ORANGE
    return patch


def _object_specs(palette_len: int, object_count: int) -> list[dict]:
    zero_pair = random.randint(1, 2)
    mixed_k = random.randint(1, max(1, palette_len - 1))
    mixed_orange = random.randint(1, 2)
    pure_k = palette_len

    specs = [
        {"gray": zero_pair, "orange": zero_pair, "kind": "zero"},
        {"gray": mixed_k + mixed_orange, "orange": mixed_orange, "kind": "mixed"},
        {"gray": pure_k, "orange": 0, "kind": "pure"},
    ]

    while len(specs) < object_count:
        kind = random.choice(["zero", "mixed", "pure"])
        if kind == "zero":
            pair = random.randint(1, 2)
            specs.append({"gray": pair, "orange": pair, "kind": kind})
        elif kind == "mixed":
            k = random.randint(1, palette_len)
            orange = random.randint(1, 2)
            specs.append({"gray": k + orange, "orange": orange, "kind": kind})
        else:
            k = random.randint(1, palette_len)
            specs.append({"gray": k, "orange": 0, "kind": kind})

    random.shuffle(specs)
    return specs


def _build_patches(base_color: int, specs: list[dict]) -> list[dict]:
    patches = []
    for spec in specs:
        marker_total = spec["gray"] + spec["orange"]
        min_area = max(8, marker_total + random.randint(3, 6))
        mask = _random_object_mask(min_area)
        patch = _render_object(mask, base_color, spec["gray"], spec["orange"])
        patches.append({"patch": patch, "gray": spec["gray"], "orange": spec["orange"], "kind": spec["kind"]})
    patches.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    return patches


def _free_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occupied.shape[0], r0 + h + gap)
    cc1 = min(occupied.shape[1], c0 + w + gap)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patches(body_h: int, width: int, patches: list[dict]) -> np.ndarray | None:
    body = np.full((body_h, width), BG, dtype=int)
    occupied = np.zeros((body_h, width), dtype=bool)

    for item in patches:
        patch = item["patch"]
        h, w = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(body_h - h + 1)
            for c0 in range(width - w + 1)
            if _free_rect(occupied, r0, c0, h, w, gap=1)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        region = body[r0:r0 + h, c0:c0 + w]
        region[patch != BG] = patch[patch != BG]
        occupied[r0:r0 + h, c0:c0 + w] = True
    return body


def _semantic_ok(inp: np.ndarray) -> bool:
    out = main(inp)
    bg = get_bg(inp)
    palette, _ = _palette_from_top_row(inp, bg)
    body_start = _body_start_row(inp, bg)
    comps = find_components(inp[body_start:], bg=bg)

    if len(comps) < 3:
        return False
    if not np.array_equal(inp[0], out[0]):
        return False

    object_mask = np.zeros(inp.shape, dtype=bool)
    saw_zero = False
    saw_mixed_positive = False
    saw_pure_positive = False
    positive_ks = set()

    for comp in comps:
        coords = comp["coords"].copy()
        coords[:, 0] += body_start
        object_mask[coords[:, 0], coords[:, 1]] = True
        vals = inp[coords[:, 0], coords[:, 1]]
        gray_count = int(np.count_nonzero(vals == GRAY))
        orange_count = int(np.count_nonzero(vals == ORANGE))
        k = gray_count - orange_count

        if gray_count > 0 and orange_count > 0 and k == 0:
            saw_zero = True
        if gray_count > 0 and orange_count > 0 and k > 0:
            saw_mixed_positive = True
        if gray_count > 0 and orange_count == 0 and k > 0:
            saw_pure_positive = True

        if k == 0:
            if np.any(out[coords[:, 0], coords[:, 1]] != bg):
                return False
        else:
            if not np.all(out[coords[:, 0], coords[:, 1]] == palette[k - 1]):
                return False
            positive_ks.add(k)

    if not np.array_equal(out[~object_mask], inp[~object_mask]):
        return False
    return saw_zero and saw_mixed_positive and saw_pure_positive and len(positive_ks) >= 2


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    object_count = random.randint(3, 5)

    base_color = random.choice(COLOR_POOL)
    palette_choices = [color for color in COLOR_POOL if color != base_color]
    palette = random.sample(palette_choices, palette_len)

    specs = _object_specs(palette_len, object_count)
    patches = _build_patches(base_color, specs)

    min_width = max(
        palette_len + 2,
        max(item["patch"].shape[1] for item in patches) + 2,
        12,
    )
    min_body_h = max(max(item["patch"].shape[0] for item in patches) + 2, 10)
    width = random.randint(min_width, 28)
    body_h = random.randint(min_body_h, 24)

    body = _place_patches(body_h, width, patches)
    if body is None:
        return None

    grid = np.full((body_h + 2, width), BG, dtype=int)
    palette_start = random.randint(0, width - palette_len)
    grid[0, palette_start:palette_start + palette_len] = np.array(palette, dtype=int)
    grid[2:, :] = body

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signed-count palette input")
