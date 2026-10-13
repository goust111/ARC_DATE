from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, "expected a unique dominant background"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords.tolist()}


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _component_pattern(g: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords.tolist():
        out[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
    return out


def _frame_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    colors = {int(g[r, c]) for r, c in coords.tolist()}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(sub[0, :] == color)):
        return None
    if not bool(np.all(sub[-1, :] == color)):
        return None
    if not bool(np.all(sub[:, 0] == color)):
        return None
    if not bool(np.all(sub[:, -1] == color)):
        return None
    if not bool(np.all(sub[1:-1, 1:-1] == bg)):
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_shape": (int(h - 2), int(w - 2)),
    }


def _key_info(
    g: np.ndarray,
    coords: np.ndarray,
    bg: int,
    container_colors: set[int],
) -> dict | None:
    if int(coords.shape[0]) > 8:
        return None

    pattern = _component_pattern(g, coords, bg)
    colors = sorted(set(map(int, pattern.ravel())) - {int(bg)})
    if len(colors) != 2:
        return None

    hits = [color for color in colors if color in container_colors]
    if len(hits) != 1:
        return None
    match_color = int(hits[0])
    fill_color = int(colors[0] if colors[1] == match_color else colors[1])

    match_cells = {
        (int(r), int(c))
        for r, c in coords.tolist()
        if int(g[int(r), int(c)]) == match_color
    }
    fill_cells = {
        (int(r), int(c))
        for r, c in coords.tolist()
        if int(g[int(r), int(c)]) == fill_color
    }
    if not _is_connected_cells(match_cells) or not _is_connected_cells(fill_cells):
        return None

    r0, c0, r1, c1 = _bbox(coords)
    if not any(r in {r0, r1} or c in {c0, c1} for r, c in match_cells):
        return None
    if not any(
        any((nr, nc) in match_cells for nr, nc in _neighbors4(r, c))
        for r, c in fill_cells
    ):
        return None

    return {
        "match_color": int(match_color),
        "fill_color": int(fill_color),
        "pattern": pattern,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each rectangular frame with the sprite that shares its border color, then tile the
    sprite's bbox pattern inside the frame and erase the sprite itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    components = _components(g != bg)
    containers: dict[int, dict] = {}
    unknown_components: list[np.ndarray] = []
    for coords in components:
        info = _frame_info(g, coords, bg)
        if info is None:
            unknown_components.append(coords)
            continue
        color = int(info["color"])
        assert color not in containers, "expected one frame per border color"
        containers[color] = info

    assert containers, "expected at least one frame"
    container_colors = set(containers)

    keys: dict[int, dict] = {}
    key_coords: list[tuple[int, int]] = []
    for coords in unknown_components:
        info = _key_info(g, coords, bg, container_colors)
        assert info is not None, "every non-frame component must be a valid key sprite"
        match_color = int(info["match_color"])
        assert match_color not in keys, "expected one key per frame color"
        keys[match_color] = info
        key_coords.extend(_coord_set(coords))

    assert set(keys) == container_colors, "every frame must have exactly one matching key"

    out = g.copy()
    for r, c in key_coords:
        out[r, c] = bg

    for color, frame in containers.items():
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        pattern = np.asarray(keys[color]["pattern"], dtype=int)
        fill_color = int(keys[color]["fill_color"])
        ph, pw = map(int, pattern.shape)

        for rr in range(inner_h):
            for cc in range(inner_w):
                cell = int(pattern[rr % ph, cc % pw])
                out[r0 + 1 + rr, c0 + 1 + cc] = fill_color if cell == fill_color else bg

    return out


# --------------- generator ---------------

_PATTERN_LIBRARY = (
    np.array([[1, 1], [1, 2]], dtype=int),
    np.array([[1, 1, 1], [1, 2, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 2, 2]], dtype=int),
    np.array([[1, 1], [1, 2], [1, 2], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 2, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 2, 2], [1, 1, 0]], dtype=int),
)

_GRID_MIN = 18
_GRID_MAX = 30
_CONTAINER_COUNT = (2, 4)
_INNER_H_MAX = 9
_INNER_W_MAX = 10
_PLACEMENT_GAP = 1
_BORDER_MARGIN = 1
_GEN_ATTEMPTS = 400


def _random_pattern() -> np.ndarray:
    pattern = np.array(random.choice(_PATTERN_LIBRARY), dtype=int, copy=True)
    if random.choice((False, True)):
        pattern = np.fliplr(pattern)
    if random.choice((False, True)):
        pattern = np.flipud(pattern)
    k = random.randint(0, 3)
    if k:
        pattern = np.rot90(pattern, k)
    return np.asarray(pattern, dtype=int)


def _nudge_period(length: int, period: int, limit: int) -> int:
    if period <= 1 or length % period != 0 or random.random() >= 0.7:
        return int(length)
    for delta in random.sample((-1, 1), 2):
        new_length = int(length + delta)
        if 2 <= new_length <= limit:
            return new_length
    return int(length)


def _sample_inner_shape(pattern: np.ndarray) -> tuple[int, int]:
    ph, pw = map(int, pattern.shape)
    mode = random.choice(("tight", "mixed", "roomy"))
    if mode == "tight":
        h = random.randint(max(2, ph - 1), min(_INNER_H_MAX, ph + 1))
        w = random.randint(max(2, pw - 1), min(_INNER_W_MAX, pw + 1))
    elif mode == "mixed":
        h = random.randint(max(2, ph - 1), min(_INNER_H_MAX, ph + 4))
        w = random.randint(max(2, pw - 1), min(_INNER_W_MAX, pw + 4))
    else:
        h = random.randint(max(2, ph), _INNER_H_MAX)
        w = random.randint(max(2, pw), _INNER_W_MAX)
    h = _nudge_period(h, ph, _INNER_H_MAX)
    w = _nudge_period(w, pw, _INNER_W_MAX)
    return int(h), int(w)


def _area_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(occupied.shape[0], int(r0) + int(h) + pad)
    cc1 = min(occupied.shape[1], int(c0) + int(w) + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_box(
    occupied: np.ndarray,
    h: int,
    w: int,
    pad: int,
) -> tuple[int, int] | None:
    H, W = occupied.shape
    positions = [
        (r, c)
        for r in range(_BORDER_MARGIN, H - h - _BORDER_MARGIN + 1)
        for c in range(_BORDER_MARGIN, W - w - _BORDER_MARGIN + 1)
    ]
    random.shuffle(positions)
    for r, c in positions:
        if _area_free(occupied, r, c, h, w, pad):
            occupied[r : r + h, c : c + w] = True
            return int(r), int(c)
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _draw_sprite(
    grid: np.ndarray,
    pattern: np.ndarray,
    r0: int,
    c0: int,
    match_color: int,
    fill_color: int,
) -> None:
    sprite = np.full(pattern.shape, -1, dtype=int)
    sprite[pattern == 1] = int(match_color)
    sprite[pattern == 2] = int(fill_color)
    for rr in range(sprite.shape[0]):
        for cc in range(sprite.shape[1]):
            if int(sprite[rr, cc]) >= 0:
                grid[r0 + rr, c0 + cc] = int(sprite[rr, cc])


def _render_expected(grid: np.ndarray, records: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rec in records:
        kr, kc = rec["key_pos"]
        pattern = rec["pattern"]
        for rr in range(pattern.shape[0]):
            for cc in range(pattern.shape[1]):
                if int(pattern[rr, cc]) != 0:
                    out[kr + rr, kc + cc] = int(bg)

    for rec in records:
        fr, fc = rec["frame_pos"]
        inner_h = int(rec["inner_h"])
        inner_w = int(rec["inner_w"])
        pattern = rec["pattern"]
        fill_color = int(rec["fill_color"])
        ph, pw = map(int, pattern.shape)
        for rr in range(inner_h):
            for cc in range(inner_w):
                out[fr + 1 + rr, fc + 1 + cc] = fill_color if int(pattern[rr % ph, cc % pw]) == 2 else int(bg)
    return out


def _sample_records() -> tuple[list[dict], int, int, int] | None:
    n = random.randint(*_CONTAINER_COUNT)
    H = random.randint(max(_GRID_MIN, 16 + 2 * n), _GRID_MAX)
    W = random.randint(max(_GRID_MIN, 16 + 2 * n), _GRID_MAX)

    colors = list(range(10))
    bg = random.choice(colors)
    non_bg = [c for c in colors if c != bg]
    random.shuffle(non_bg)
    border_colors = non_bg[:n]
    fill_pool = non_bg[n:]
    if len(fill_pool) < n:
        return None
    if random.choice((False, True)):
        fill_colors = random.sample(fill_pool, n)
    else:
        fill_colors = [random.choice(fill_pool) for _ in range(n)]

    records = []
    for i in range(n):
        pattern = _random_pattern()
        inner_h, inner_w = _sample_inner_shape(pattern)
        records.append(
            {
                "match_color": int(border_colors[i]),
                "fill_color": int(fill_colors[i]),
                "pattern": pattern,
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
            }
        )

    if not any(
        rec["inner_h"] % rec["pattern"].shape[0] != 0 or rec["inner_w"] % rec["pattern"].shape[1] != 0
        for rec in records
    ):
        rec = random.choice(records)
        ph, pw = map(int, rec["pattern"].shape)
        rec["inner_h"] = _nudge_period(int(rec["inner_h"]), ph, _INNER_H_MAX)
        rec["inner_w"] = _nudge_period(int(rec["inner_w"]), pw, _INNER_W_MAX)

    return records, int(bg), int(H), int(W)


def _generate_once() -> np.ndarray | None:
    sampled = _sample_records()
    if sampled is None:
        return None
    records, bg, H, W = sampled

    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for rec in sorted(records, key=lambda item: (item["inner_h"] + 2) * (item["inner_w"] + 2), reverse=True):
        outer_h = int(rec["inner_h"]) + 2
        outer_w = int(rec["inner_w"]) + 2
        pos = _place_box(occupied, outer_h, outer_w, _PLACEMENT_GAP)
        if pos is None:
            return None
        rec["frame_pos"] = pos
        _draw_frame(grid, pos[0], pos[1], outer_h, outer_w, int(rec["match_color"]))

    for rec in records:
        ph, pw = map(int, rec["pattern"].shape)
        pos = _place_box(occupied, ph, pw, _PLACEMENT_GAP)
        if pos is None:
            return None
        rec["key_pos"] = pos
        _draw_sprite(
            grid,
            rec["pattern"],
            pos[0],
            pos[1],
            int(rec["match_color"]),
            int(rec["fill_color"]),
        )

    expected = _render_expected(grid, records, bg)
    if _most_common_color(grid) != bg:
        return None
    if len(_components(grid != bg)) != 2 * len(records):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
