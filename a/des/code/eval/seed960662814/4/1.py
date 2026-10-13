import random
from collections import deque

import numpy as np


GREY = 5
DEMO_ACCENT = 1
GEN_RETRIES = 400
QUADRANTS = ("tl", "tr", "bl", "br")
ACCENT_TO_TRANSFORM = {
    (1, 1): "identity",
    (1, 0): "hflip",
    (0, 1): "vflip",
    (0, 0): "rot180",
}


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color == col_color:
                return int(r), int(c), int(row_color)
    raise AssertionError("could not find grey separator cross")


def _corner_origins(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, w - 2),
        "bl": (h - 2, 0),
        "br": (h - 2, w - 2),
    }


def _read_demo_info(grid: np.ndarray) -> dict[int, dict[str, str]]:
    h, w = grid.shape
    info: dict[int, dict[str, str]] = {}
    for quadrant, (r0, c0) in _corner_origins(h, w).items():
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        vals, counts = np.unique(patch, return_counts=True)
        if len(vals) != 2 or sorted(map(int, counts)) != [1, 3]:
            raise AssertionError("corner demo block must use a 3+1 color split")
        demo_color = int(vals[np.argmax(counts)])
        accent_color = int(vals[np.argmin(counts)])
        accent_pos = tuple(map(int, np.argwhere(patch == accent_color)[0]))
        if accent_pos not in ACCENT_TO_TRANSFORM:
            raise AssertionError("invalid accent position in demo block")
        info[demo_color] = {
            "quadrant": quadrant,
            "transform": ACCENT_TO_TRANSFORM[accent_pos],
        }
    if len(info) != 4:
        raise AssertionError("demo colors must uniquely identify quadrants")
    return info


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _apply_transform(patch: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return patch.copy()
    if transform == "hflip":
        return np.fliplr(patch)
    if transform == "vflip":
        return np.flipud(patch)
    if transform == "rot180":
        return np.rot90(patch, 2)
    raise AssertionError(f"unknown transform {transform}")


def _paste_transparent(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > out.shape[0] or left + w > out.shape[1]:
        raise AssertionError("placement out of bounds")
    mask = patch != bg
    view = out[top : top + h, left : left + w]
    if np.any(view[mask] != bg):
        raise AssertionError("placement overlaps fixed content")
    view[mask] = patch[mask]


def _place_from_center(
    out: np.ndarray,
    patches: list[np.ndarray],
    quadrant: str,
    cross_r: int,
    cross_c: int,
    bg: int,
) -> None:
    gap = 1
    if quadrant == "tl":
        cursor_r, cursor_c = cross_r - 1, cross_c - 1
        for patch in patches:
            h, w = patch.shape
            top = cursor_r - h + 1
            left = cursor_c - w + 1
            _paste_transparent(out, patch, top, left, bg)
            cursor_r = top - gap - 1
            cursor_c = left - gap - 1
        return

    if quadrant == "tr":
        cursor_r, cursor_c = cross_r - 1, cross_c + 1
        for patch in patches:
            h, w = patch.shape
            top = cursor_r - h + 1
            left = cursor_c
            _paste_transparent(out, patch, top, left, bg)
            cursor_r = top - gap - 1
            cursor_c = left + w + gap
        return

    if quadrant == "bl":
        cursor_r, cursor_c = cross_r + 1, cross_c - 1
        for patch in patches:
            h, w = patch.shape
            top = cursor_r
            left = cursor_c - w + 1
            _paste_transparent(out, patch, top, left, bg)
            cursor_r = top + h + gap
            cursor_c = left - gap - 1
        return

    if quadrant == "br":
        cursor_r, cursor_c = cross_r + 1, cross_c + 1
        for patch in patches:
            h, w = patch.shape
            top = cursor_r
            left = cursor_c
            _paste_transparent(out, patch, top, left, bg)
            cursor_r = top + h + gap
            cursor_c = left + w + gap
        return

    raise AssertionError(f"unknown quadrant {quadrant}")


def _extract_sprites(
    grid: np.ndarray,
    bg: int,
    grey: int,
    cross_r: int,
    cross_c: int,
    demo_info: dict[int, dict[str, str]],
) -> list[dict]:
    h, w = grid.shape
    mask = (grid != bg) & (grid != grey)
    for r0, c0 in _corner_origins(h, w).values():
        mask[r0 : r0 + 2, c0 : c0 + 2] = False

    sprites: list[dict] = []
    demo_colors = set(demo_info)
    for comp in _components(mask):
        r0, r1, c0, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        marker_hits: list[tuple[int, int, int]] = []
        for r, c in comp:
            color = int(grid[r, c])
            patch[r - r0, c - c0] = color
            if color in demo_colors:
                marker_hits.append((int(r), int(c), color))
        if len(marker_hits) != 1:
            raise AssertionError("each sprite must have exactly one address marker")
        mr, mc, marker_color = marker_hits[0]
        sprites.append(
            {
                "coords": comp,
                "patch": patch,
                "marker_color": int(marker_color),
                "sort_key": (
                    int((mr - cross_r) ** 2 + (mc - cross_c) ** 2),
                    int(mr),
                    int(mc),
                ),
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 2x2 corner demos, use their accent-cell orientation to decode the
    quadrant transforms, then move each marked sprite into its addressed quadrant.

    Within each destination quadrant, sprites are ordered by their original marker
    distance to the grey-line intersection and packed diagonally away from center.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    cross_r, cross_c, grey = _find_cross(g, bg)
    demo_info = _read_demo_info(g)

    sprites = _extract_sprites(g, bg, grey, cross_r, cross_c, demo_info)
    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg

    by_quadrant: dict[str, list[tuple[tuple[int, int, int], np.ndarray]]] = {q: [] for q in QUADRANTS}
    for sprite in sprites:
        info = demo_info[int(sprite["marker_color"])]
        transformed = _apply_transform(sprite["patch"], info["transform"])
        by_quadrant[info["quadrant"]].append((sprite["sort_key"], transformed))

    for quadrant in QUADRANTS:
        ordered = [patch for _key, patch in sorted(by_quadrant[quadrant], key=lambda item: item[0])]
        _place_from_center(out, ordered, quadrant, cross_r, cross_c, bg)
    return out


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return (
        any(r == 0 for r, _c in cells)
        and any(r == h - 1 for r, _c in cells)
        and any(c == 0 for _r, c in cells)
        and any(c == w - 1 for _r, c in cells)
    )


def _connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _random_shape(h: int, w: int) -> np.ndarray | None:
    full = {(r, c) for r in range(h) for c in range(w)}
    low = max(5, (h * w + 1) // 2)
    target = random.randint(low, h * w)
    cells = set(full)
    while len(cells) > target:
        candidates = list(cells)
        random.shuffle(candidates)
        removed = False
        for cell in candidates:
            trial = cells - {cell}
            if _touches_all_sides(trial, h, w) and _connected_cells(trial):
                cells = trial
                removed = True
                break
        if not removed:
            break
    if len(cells) != target or not _touches_all_sides(cells, h, w):
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite_patch(bg: int, marker_color: int, body_palette: list[int], transform: str) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = _random_shape(h, w)
        if mask is None:
            continue
        cells = list(map(tuple, np.argwhere(mask)))
        body_colors = random.sample(body_palette, k=random.randint(2, min(3, len(body_palette))))
        patch = np.full((h, w), bg, dtype=int)
        random.shuffle(cells)
        for idx, (r, c) in enumerate(cells):
            patch[r, c] = body_colors[idx % len(body_colors)]
        marker_r, marker_c = random.choice(cells)
        patch[marker_r, marker_c] = marker_color
        non_marker = [int(patch[r, c]) for r, c in cells if (r, c) != (marker_r, marker_c)]
        if len(set(non_marker)) < 2:
            continue
        if transform != "identity" and np.array_equal(patch, _apply_transform(patch, transform)):
            continue
        return patch
    return None


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, margin: int = 1) -> None:
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(blocked.shape[0], top + h + margin)
    c1 = min(blocked.shape[1], left + w + margin)
    blocked[r0:r1, c0:c1] = True


def _scatter_candidates(blocked: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch.shape
    candidates = []
    for top in range(1, h - ph):
        for left in range(1, w - pw):
            if blocked[top : top + ph, left : left + pw].any():
                continue
            if blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any():
                continue
            candidates.append((int(top), int(left)))
    random.shuffle(candidates)
    return candidates


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = patch[mask]


def _demo_patch(color: int, accent_pos: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    patch[accent_pos] = DEMO_ACCENT
    return patch


def _base_canvas(h: int, w: int, bg: int, cross_r: int, cross_c: int, demo_colors: dict[str, int]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    grid[cross_r, :] = GREY
    grid[:, cross_c] = GREY
    accents = {
        "tl": (1, 1),
        "tr": (1, 0),
        "bl": (0, 1),
        "br": (0, 0),
    }
    for quadrant, (r0, c0) in _corner_origins(h, w).items():
        grid[r0 : r0 + 2, c0 : c0 + 2] = _demo_patch(demo_colors[quadrant], accents[quadrant])
    return grid


def _expected_output(
    base: np.ndarray,
    specs: list[dict],
    cross_r: int,
    cross_c: int,
    bg: int,
) -> np.ndarray | None:
    out = base.copy()
    by_quadrant: dict[str, list[tuple[tuple[int, int, int], np.ndarray]]] = {q: [] for q in QUADRANTS}
    for spec in specs:
        by_quadrant[spec["quadrant"]].append((spec["sort_key"], _apply_transform(spec["patch"], spec["transform"])))
    try:
        for quadrant in QUADRANTS:
            ordered = [patch for _key, patch in sorted(by_quadrant[quadrant], key=lambda item: item[0])]
            _place_from_center(out, ordered, quadrant, cross_r, cross_c, bg)
    except AssertionError:
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {GREY, DEMO_ACCENT}]
    bg = int(random.choice(bg_choices))
    demo_pool = [c for c in range(10) if c not in {bg, GREY, DEMO_ACCENT}]
    random.shuffle(demo_pool)
    demo_colors = {
        "tl": int(demo_pool[0]),
        "tr": int(demo_pool[1]),
        "bl": int(demo_pool[2]),
        "br": int(demo_pool[3]),
    }
    body_palette = [c for c in range(10) if c not in {bg, GREY, DEMO_ACCENT, *demo_colors.values()}]
    if len(body_palette) < 3:
        return None

    h = random.randint(27, 32)
    w = random.randint(27, 32)
    cross_r = random.randint(13, h - 14)
    cross_c = random.randint(13, w - 14)

    base = _base_canvas(h, w, bg, cross_r, cross_c, demo_colors)
    inp = base.copy()
    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, cross_r, 0, 1, w)
    _mark_blocked(blocked, 0, cross_c, h, 1)
    for r0, c0 in _corner_origins(h, w).values():
        _mark_blocked(blocked, r0, c0, 2, 2)

    specs: list[dict] = []
    extras = {q: int(random.random() < 0.5) for q in QUADRANTS}
    for quadrant in QUADRANTS:
        count = 1 + extras[quadrant]
        transform = {
            "tl": "identity",
            "tr": "hflip",
            "bl": "vflip",
            "br": "rot180",
        }[quadrant]
        marker_color = demo_colors[quadrant]
        for _ in range(count):
            patch = _make_sprite_patch(bg, marker_color, body_palette, transform)
            if patch is None:
                return None
            candidates = _scatter_candidates(blocked, patch)
            if not candidates:
                return None
            top, left = candidates[0]
            _paint_patch(inp, patch, top, left, bg)
            _mark_blocked(blocked, top, left, patch.shape[0], patch.shape[1])
            marker_rel = tuple(map(int, np.argwhere(patch == marker_color)[0]))
            marker_abs = (top + marker_rel[0], left + marker_rel[1])
            specs.append(
                {
                    "quadrant": quadrant,
                    "transform": transform,
                    "patch": patch,
                    "sort_key": (
                        int((marker_abs[0] - cross_r) ** 2 + (marker_abs[1] - cross_c) ** 2),
                        int(marker_abs[0]),
                        int(marker_abs[1]),
                    ),
                }
            )

    expected = _expected_output(base, specs, cross_r, cross_c, bg)
    if expected is None:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
