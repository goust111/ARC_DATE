from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_from_cells(cells: set[tuple[int, int]]) -> list[np.ndarray]:
    remaining = set((int(r), int(c)) for r, c in cells)
    comps: list[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(np.array(sorted(comp), dtype=int))
    return comps


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    return _components_from_cells(cells)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _width(coords: np.ndarray) -> int:
    _r0, c0, _r1, c1 = _bbox(coords)
    return c1 - c0 + 1


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _find_divider_and_markers(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(np.isin(row, [RED, YELLOW])) and np.any(row == YELLOW):
            candidates.append(int(r))
    assert len(candidates) == 1, "expected exactly one divider row with yellow markers"
    divider_r = candidates[0]
    markers = [int(c) for c in np.flatnonzero(grid[divider_r] == YELLOW)]
    assert 2 <= len(markers) <= 3
    return divider_r, markers


def _channel_index(col: int, markers: list[int]) -> int:
    return int(np.searchsorted(np.array(markers, dtype=int), int(col), side="left"))


def _split_sprite(sprite: np.ndarray, markers: list[int]) -> list[np.ndarray]:
    buckets: dict[int, set[tuple[int, int]]] = {}
    for rr, cc in sprite:
        key = _channel_index(int(cc), markers)
        buckets.setdefault(key, set()).add((int(rr), int(cc)))

    fragments: list[np.ndarray] = []
    for cells in buckets.values():
        fragments.extend(_components_from_cells(cells))
    return fragments


def _ordered_fragments(sprites: list[np.ndarray], markers: list[int]) -> list[np.ndarray]:
    fragments: list[np.ndarray] = []
    for sprite in sprites:
        parts = _split_sprite(sprite, markers)
        widths = [_width(part) for part in parts]
        assert any(w == 1 for w in widths), "each sprite must create a single-column fragment"
        assert any(w > 1 for w in widths), "each sprite must create a multi-column fragment"
        fragments.extend(parts)
    fragments.sort(key=lambda frag: (-int(frag[:, 0].max()), int(frag[:, 1].min())))
    return fragments


def _drop_fragment(
    fragment: np.ndarray,
    divider_r: int,
    occupied: np.ndarray,
) -> np.ndarray:
    coords = fragment.copy()
    width = _width(coords)
    H, _W = occupied.shape
    shift = 0
    while True:
        moved = coords.copy()
        moved[:, 0] += shift + 1
        if width == 1:
            if np.any(moved[:, 0] >= H):
                break
        else:
            if np.any(moved[:, 0] >= divider_r):
                break
        if np.any(occupied[moved[:, 0], moved[:, 1]]):
            break
        shift += 1

    placed = coords.copy()
    placed[:, 0] += shift
    occupied[placed[:, 0], placed[:, 1]] = True
    return placed


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H <= 30 and W <= 30
    assert _get_bg(g) == BLACK
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, YELLOW})

    divider_r, markers = _find_divider_and_markers(g)
    assert np.all(g[divider_r + 1 :, :] != BLUE), "input sprites must stay above the divider"

    sprites = _components_from_mask(g == BLUE)
    assert 2 <= len(sprites) <= 4
    for sprite in sprites:
        r0, _c0, r1, _c1 = _bbox(sprite)
        assert 3 <= len(sprite) <= 8
        assert r1 <= divider_r - 2
        assert r0 >= 0

    fragments = _ordered_fragments(sprites, markers)
    occupied = np.zeros((H, W), dtype=bool)
    placed: list[dict[str, np.ndarray | int]] = []
    for fragment in fragments:
        moved = _drop_fragment(fragment, divider_r, occupied)
        placed.append({"source": fragment, "moved": moved, "width": _width(fragment)})

    out = g.copy()
    out[g == BLUE] = BLACK
    out[divider_r, :] = RED
    for item in placed:
        moved = item["moved"]
        out[moved[:, 0], moved[:, 1]] = BLUE

    meta = {
        "divider_r": divider_r,
        "markers": markers,
        "sprites": sprites,
        "fragments": fragments,
        "placed": placed,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut blue sprites into ray-defined channels, then drop narrow and wide fragments to different floors.

    Input structure:
    - The grid has a black background.
    - One horizontal divider row is red except for 2-3 yellow marker pixels embedded in it.
    - Blue 4-connected sprites sit strictly above the divider and may cross the marker columns.

    Rule:
    - Treat each yellow marker column as an upward vertical ray ending just above the divider.
    - These rays partition the columns into channels, with each ray column belonging to the channel on its left.
    - Split every blue sprite into channel-local 4-connected fragments, erase the original blue pixels, and restore the marker cells back to red.
    - Drop each fragment independently: a one-column fragment passes through the divider and falls as far as possible, while a wider fragment falls only within the space above the divider; process fragments from lower source fragments to higher ones, breaking ties by leftmost column.

    Key insight:
    - The yellow markers are not moved objects but hidden cutting guides: they define channel boundaries that change fragment widths, and width alone decides whether a fragment stops above the divider or passes through it.
    """
    out, _meta = _solve_details(grid)
    return out


def _touching_or_overlapping(a: np.ndarray, b: np.ndarray) -> bool:
    aset = {tuple(map(int, rc)) for rc in a}
    for r, c in b:
        rr, cc = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in aset:
                    return True
    return False


def _build_local_sprite(orientation: str, multi_width: int) -> np.ndarray:
    for _ in range(80):
        height = random.randint(2, 4)
        connector = random.randrange(height)
        single_len = random.randint(1, min(3, height))
        seg_start_min = max(0, connector - single_len + 1)
        seg_start_max = min(connector, height - single_len)
        seg_start = random.randint(seg_start_min, seg_start_max)
        single_rows = range(seg_start, seg_start + single_len)

        cells: set[tuple[int, int]] = set()
        if orientation == "left_single":
            cells.update((r, 0) for r in single_rows)
            cells.update((connector, c) for c in range(1, multi_width + 1))
            candidates = []
            if connector > 0:
                candidates.extend([(connector - 1, 1), (connector - 1, multi_width)])
                if multi_width == 3:
                    candidates.append((connector - 1, 2))
            if connector + 1 < height:
                candidates.extend([(connector + 1, 1), (connector + 1, multi_width)])
                if multi_width == 3:
                    candidates.append((connector + 1, 2))
        else:
            cells.update((r, multi_width) for r in single_rows)
            cells.update((connector, c) for c in range(multi_width))
            candidates = []
            if connector > 0:
                candidates.extend([(connector - 1, 0), (connector - 1, multi_width - 1)])
                if multi_width == 3:
                    candidates.append((connector - 1, 1))
            if connector + 1 < height:
                candidates.extend([(connector + 1, 0), (connector + 1, multi_width - 1)])
                if multi_width == 3:
                    candidates.append((connector + 1, 1))

        random.shuffle(candidates)
        extra_cap = min(2, len(candidates))
        extra_count = random.randint(0, extra_cap)
        for extra in candidates[:extra_count]:
            cells.add(extra)

        sprite = np.array(sorted(cells), dtype=int)
        if 3 <= len(sprite) <= 8:
            return sprite
    raise RuntimeError("failed to build sprite")


def _choose_layout(n_used: int) -> tuple[int, list[dict[str, int | str]], list[int]]:
    specs = []
    for _ in range(n_used):
        specs.append(
            {
                "orientation": random.choice(["left_single", "right_single"]),
                "multi_width": random.choice([2, 3]),
            }
        )

    base_width = 2 + sum(int(spec["multi_width"]) + 1 for spec in specs) + 2 * (n_used - 1)
    if base_width > 19:
        raise RuntimeError("layout too wide")
    W = random.randint(max(15, base_width), 19)

    lengths = [int(spec["multi_width"]) + 1 for spec in specs]
    gaps = [1] + [2] * (n_used - 1) + [1]
    extra = W - (sum(lengths) + sum(gaps))
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    starts = []
    c = gaps[0]
    for i, length in enumerate(lengths):
        starts.append(c)
        c += length
        if i < n_used - 1:
            c += gaps[i + 1]

    used_markers = []
    for start, spec in zip(starts, specs):
        multi_width = int(spec["multi_width"])
        if spec["orientation"] == "left_single":
            marker = start
        else:
            marker = start + multi_width - 1
        used_markers.append(marker)

    return W, specs, used_markers


def _add_extra_marker(W: int, occupied_cols: set[int], used_markers: list[int]) -> int | None:
    free = []
    for c in range(1, W - 1):
        if c in occupied_cols:
            continue
        if any(abs(c - m) <= 1 for m in used_markers):
            continue
        free.append(c)
    if not free:
        return None
    return int(random.choice(free))


def _choose_sprite_tops(heights: list[int], divider_r: int) -> list[int]:
    max_bottom = divider_r - 2
    mins = [h - 1 for h in heights]
    bottoms = [random.randint(min_bottom, max_bottom) for min_bottom in mins]

    if len(heights) >= 2 and random.random() < 0.45:
        i, j = random.sample(range(len(heights)), 2)
        shared_min = max(mins[i], mins[j])
        shared = random.randint(shared_min, max_bottom)
        bottoms[i] = shared
        bottoms[j] = shared

    if random.random() < 0.6:
        k = random.randrange(len(heights))
        bottoms[k] = max_bottom

    return [bottom - h + 1 for bottom, h in zip(bottoms, heights)]


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert _get_bg(grid) == BLACK

    divider_r, markers = _find_divider_and_markers(grid)
    assert np.all(np.isin(grid[divider_r], [RED, YELLOW]))
    assert np.all(grid[divider_r + 1 :, :] != BLUE)

    sprites = _components_from_mask(grid == BLUE)
    assert 2 <= len(sprites) <= 4

    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for sprite in sprites:
        assert 3 <= len(sprite) <= 8
        assert int(sprite[:, 0].max()) <= divider_r - 2
        shape = _normalize(sprite)
        assert shape not in seen_shapes
        seen_shapes.add(shape)

        parts = _split_sprite(sprite, markers)
        widths = [_width(part) for part in parts]
        assert any(w == 1 for w in widths)
        assert any(w > 1 for w in widths)

    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            assert not _touching_or_overlapping(sprites[i], sprites[j])


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    divider_r, _markers = _find_divider_and_markers(inp)
    assert out.shape == inp.shape
    assert np.all(out[divider_r] == RED)
    assert not np.any(out == YELLOW)
    assert not np.any(out[divider_r] == BLUE)
    assert int(np.count_nonzero(out == BLUE)) == int(np.count_nonzero(inp == BLUE))

    blue_out = _components_from_mask(out == BLUE)
    for comp in blue_out:
        width = _width(comp)
        if width == 1:
            assert int(comp[:, 0].min()) > divider_r
            assert int(comp[:, 0].max()) == out.shape[0] - 1 or np.any(out[int(comp[:, 0].max()) + 1, comp[:, 1]] == BLUE)
        else:
            assert int(comp[:, 0].max()) < divider_r

    expected, meta = _solve_details(inp)
    assert np.array_equal(expected, out)
    for item in meta["placed"]:
        src = item["source"]
        moved = item["moved"]
        width = int(item["width"])
        assert np.array_equal(np.sort(src[:, 1]), np.sort(moved[:, 1]))
        if width == 1:
            moved_next = moved.copy()
            moved_next[:, 0] += 1
            assert np.any(moved_next[:, 0] >= out.shape[0]) or np.any(expected[moved_next[:, 0], moved_next[:, 1]] == BLUE)
        else:
            moved_next = moved.copy()
            moved_next[:, 0] += 1
            blocked_by_divider = np.any(moved_next[:, 0] >= meta["divider_r"])
            blocked_by_blue = np.any(
                (moved_next[:, 0] < out.shape[0]) & (expected[moved_next[:, 0], moved_next[:, 1]] == BLUE)
            )
            assert blocked_by_divider or blocked_by_blue


def _gen_attempt() -> np.ndarray | None:
    try:
        H = random.randint(14, 18)
        divider_r = random.randint(7, H - 5)
        n_used = random.choice([2, 3])
        W, specs, used_markers = _choose_layout(n_used)

        occupied_cols: set[int] = set()
        local_sprites = []
        shape_keys: set[tuple[tuple[int, int], ...]] = set()
        for spec, marker in zip(specs, used_markers):
            multi_width = int(spec["multi_width"])
            if spec["orientation"] == "left_single":
                interval = range(marker, marker + multi_width + 1)
            else:
                interval = range(marker - multi_width + 1, marker + 2)
            occupied_cols.update(int(c) for c in interval)

        for spec in specs:
            sprite = _build_local_sprite(str(spec["orientation"]), int(spec["multi_width"]))
            key = _normalize(sprite)
            if key in shape_keys:
                sprite = _build_local_sprite(str(spec["orientation"]), int(spec["multi_width"]))
                key = _normalize(sprite)
            if key in shape_keys:
                return None
            shape_keys.add(key)
            local_sprites.append(sprite)

        marker_total = 3 if (n_used == 2 and random.random() < 0.55) else n_used
        markers = sorted(used_markers)
        if marker_total == 3 and len(markers) == 2:
            extra_marker = _add_extra_marker(W, occupied_cols, markers)
            if extra_marker is None:
                return None
            markers = sorted(markers + [extra_marker])

        heights = [int(sprite[:, 0].max()) + 1 for sprite in local_sprites]
        tops = _choose_sprite_tops(heights, divider_r)

        grid = np.zeros((H, W), dtype=int)
        grid[divider_r, :] = RED
        grid[divider_r, markers] = YELLOW

        placed_sprites = []
        for top, spec, marker, local in zip(tops, specs, used_markers, local_sprites):
            if spec["orientation"] == "left_single":
                left = marker
            else:
                left = marker - int(spec["multi_width"]) + 1
            sprite = local.copy()
            sprite[:, 0] += top
            sprite[:, 1] += left
            if int(sprite[:, 0].max()) >= divider_r - 1:
                return None
            if np.any(grid[sprite[:, 0], sprite[:, 1]] != BLACK):
                return None
            placed_sprites.append(sprite)
            grid[sprite[:, 0], sprite[:, 1]] = BLUE

        if not (2 <= len(placed_sprites) <= 4):
            return None
        _validate_input(grid)
        out = main(grid)
        _validate_output(grid, out)
        return grid
    except AssertionError:
        return None
    except RuntimeError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-cut gravity input")
