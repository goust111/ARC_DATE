from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == RED)]
    assert len(rows) == 1, "expected exactly one full-width red divider row"
    return int(rows[0])


def _detect_period(region: np.ndarray, ignore: set[int]) -> tuple[int, int]:
    h, w = region.shape
    best = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            seen: dict[tuple[int, int], int] = {}
            ok = True
            for r in range(h):
                for c in range(w):
                    color = int(region[r, c])
                    if color in ignore:
                        continue
                    key = (r % pr, c % pc)
                    prev = seen.get(key)
                    if prev is None:
                        seen[key] = color
                    elif prev != color:
                        ok = False
                        break
                if not ok:
                    break
            if ok and len(seen) == pr * pc:
                cand = (pr * pc, pr, pc)
                if best is None or cand < best:
                    best = cand
    assert best is not None, "no reconstructible translational period found"
    return int(best[1]), int(best[2])


def _reconstruct_periodic(region: np.ndarray, pr: int, pc: int, ignore: set[int]) -> np.ndarray:
    h, w = region.shape
    orbit_color: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color in ignore:
                continue
            key = (r % pr, c % pc)
            prev = orbit_color.get(key)
            if prev is None:
                orbit_color[key] = color
            else:
                assert prev == color, "period orbit is not uniquely colored"
    assert len(orbit_color) == pr * pc, "some period classes are unobserved"

    out = np.empty_like(region)
    for r in range(h):
        for c in range(w):
            out[r, c] = orbit_color[(r % pr, c % pc)]
    return out


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _yellow_components(top_region: np.ndarray) -> list[np.ndarray]:
    return _components4(top_region == YELLOW)


def _apply_gravity(repaired_grid: np.ndarray, base_top: np.ndarray, divider_r: int) -> np.ndarray:
    g = np.asarray(repaired_grid, dtype=int)
    h, w = g.shape
    comps = _yellow_components(g[:divider_r])

    out = g.copy()
    out[:divider_r] = base_top
    occupied_yellow = out == YELLOW

    wide = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) >= 2]
    narrow = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) == 1]

    for comp in wide:
        shift = divider_r - 1 - int(comp[:, 0].max())
        moved = comp.copy()
        moved[:, 0] += shift
        assert int(moved[:, 0].max()) == divider_r - 1
        for r, c in moved:
            rr = int(r)
            cc = int(c)
            assert not occupied_yellow[rr, cc], "wide yellow clusters overlap after falling"
            out[rr, cc] = YELLOW
            occupied_yellow[rr, cc] = True

    narrow.sort(key=lambda comp: (-int(comp[:, 0].max()), int(comp[0, 1])))
    for comp in narrow:
        c = int(comp[0, 1])
        below_rows = np.where(occupied_yellow[divider_r + 1 :, c])[0]
        floor = h - 1 if below_rows.size == 0 else divider_r + 1 + int(below_rows.min())
        dest_bottom = floor - 1 if below_rows.size else floor
        shift = dest_bottom - int(comp[:, 0].max())
        moved = comp.copy()
        moved[:, 0] += shift
        assert int(moved[:, 0].min()) >= divider_r + 1
        out[divider_r, c] = BLACK
        for r, c2 in moved:
            rr = int(r)
            cc = int(c2)
            assert not occupied_yellow[rr, cc], "vertical yellow clusters overlap after falling"
            out[rr, cc] = YELLOW
            occupied_yellow[rr, cc] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded periodic upper panel, then drop the revealed yellow clusters according to their width.

    Input structure:
    - A single full-width red row divides the grid into an upper patterned panel and a lower landing area.
    - Above the divider, the visible colors form a translationally periodic pattern in blue, green, and yellow, with some cells replaced by black occluders.
    - The lower area is not occluded; it serves as the landing region for the falling yellow clusters.

    Rule:
    - Detect the upper panel's translational period while ignoring black cells, and fill every black cell from its unique periodic orbit.
    - In the repaired upper panel, detect the underlying non-yellow periodic background by treating yellow as temporary foreground and reconstruct the base color beneath every yellow source cell.
    - Find all 4-connected yellow clusters above the divider. Width-1 clusters punch through the red row and fall to the bottom or onto previously settled yellow in the same column; wider clusters stop with their lowest row immediately above the divider.
    - Replace each vacated yellow source cell by its reconstructed background color, and turn every pierced divider cell black.

    Key insight:
    - The gravity targets are defined only after the periodic repair: some crucial yellow cells are hidden by black, and the vacated source pixels must be restored from a second periodic reconstruction that ignores yellow.
    """
    g = np.asarray(grid, dtype=int)
    divider_r = _find_divider_row(g)
    top = g[:divider_r]

    full_pr, full_pc = _detect_period(top, ignore={BLACK})
    repaired_top = _reconstruct_periodic(top, full_pr, full_pc, ignore={BLACK})

    base_pr, base_pc = _detect_period(repaired_top, ignore={YELLOW})
    base_top = _reconstruct_periodic(repaired_top, base_pr, base_pc, ignore={YELLOW})

    repaired = g.copy()
    repaired[:divider_r] = repaired_top
    return _apply_gravity(repaired, base_top, divider_r)


def _tile_value(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[r % tile.shape[0], c % tile.shape[1]])


def _random_base_tile(br: int, bc: int) -> np.ndarray:
    while True:
        tile = np.random.choice([BLUE, GREEN], size=(br, bc))
        if BLUE not in tile or GREEN not in tile:
            continue
        if _detect_period(tile, ignore=set()) == (br, bc):
            return tile.astype(int)


def _visible_templates(period: int) -> list[dict[str, list[tuple[int, int]]]]:
    if period == 3:
        return [
            {
                "vertical": [(0, 2), (1, 2)],
                "wide": [(2, 0), (2, 1)],
            },
            {
                "vertical": [(0, 0), (1, 0)],
                "wide": [(2, 1), (2, 2)],
            },
        ]
    return [
        {
            "vertical": [(0, 3), (1, 3)],
            "wide": [(3, 1), (3, 2)],
        },
        {
            "vertical": [(1, 0), (2, 0)],
            "wide": [(3, 2), (3, 3)],
        },
        {
            "vertical": [(0, 1), (1, 1)],
            "wide": [(3, 0), (3, 1)],
        },
    ]


def _build_visible_tile(period: int, base_tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_view = np.empty((period, period), dtype=int)
    for r in range(period):
        for c in range(period):
            base_view[r, c] = _tile_value(base_tile, r, c)

    templates = _visible_templates(period)
    random.shuffle(templates)
    for template in templates:
        yellow_mask = np.zeros((period, period), dtype=bool)
        for r, c in template["vertical"] + template["wide"]:
            yellow_mask[r, c] = True

        # Keep every base orbit visible somewhere after removing yellow.
        br, bc = base_tile.shape
        ok = True
        for rr in range(br):
            for cc in range(bc):
                samples = [
                    (r, c)
                    for r in range(period)
                    for c in range(period)
                    if r % br == rr and c % bc == cc
                ]
                if all(yellow_mask[r, c] for r, c in samples):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        visible = base_view.copy()
        visible[yellow_mask] = YELLOW
        if _detect_period(visible, ignore=set()) != (period, period):
            continue
        if _detect_period(visible, ignore={YELLOW}) != base_tile.shape:
            continue
        return visible, yellow_mask
    raise AssertionError("failed to build a primitive visible tile")


def _build_repaired_grid(
    top_h: int,
    width: int,
    bottom_h: int,
    period: int,
    base_tile: np.ndarray,
    visible_tile: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    divider_r = top_h
    height = top_h + 1 + bottom_h
    grid = np.empty((height, width), dtype=int)

    for r in range(height):
        if r == divider_r:
            grid[r, :] = RED
            continue
        base_r = r if r < divider_r else r - 1
        for c in range(width):
            grid[r, c] = _tile_value(base_tile, base_r, c)

    top_visible = np.empty((top_h, width), dtype=int)
    for r in range(top_h):
        for c in range(width):
            top_visible[r, c] = _tile_value(visible_tile, r, c)
    grid[:divider_r] = top_visible

    base_top = np.empty((top_h, width), dtype=int)
    for r in range(top_h):
        for c in range(width):
            base_top[r, c] = _tile_value(base_tile, r, c)
    return grid, divider_r, top_visible, base_top


def _choose_black_positions(repaired_top: np.ndarray, period: int) -> set[tuple[int, int]]:
    h, w = repaired_top.shape
    comps = _yellow_components(repaired_top)
    wide = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) >= 2]
    narrow = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) == 1]
    assert wide and narrow, "need both wide and narrow yellow components"

    wide_comp = random.choice(wide)
    narrow_comp = random.choice(narrow)
    mandatory = {
        tuple(map(int, wide_comp[random.randrange(len(wide_comp))])),
        tuple(map(int, narrow_comp[random.randrange(len(narrow_comp))])),
    }
    non_yellow = list(map(tuple, np.argwhere(repaired_top != YELLOW)))
    mandatory.add(tuple(map(int, random.choice(non_yellow))))

    orbit_cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            orbit_cells[(r % period, c % period)].append((r, c))

    used_per_orbit = defaultdict(int)
    blacks = set()
    for cell in mandatory:
        key = (cell[0] % period, cell[1] % period)
        assert used_per_orbit[key] < len(orbit_cells[key]) - 1
        blacks.add(cell)
        used_per_orbit[key] += 1

    target = random.randint(int(np.ceil(0.15 * h * w)), int(np.floor(0.25 * h * w)))
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for cell in cells:
        if len(blacks) >= target:
            break
        if cell in blacks:
            continue
        key = (cell[0] % period, cell[1] % period)
        if used_per_orbit[key] >= len(orbit_cells[key]) - 1:
            continue
        blacks.add(cell)
        used_per_orbit[key] += 1

    assert any(repaired_top[r, c] == YELLOW for r, c in blacks)
    assert any(repaired_top[r, c] != YELLOW for r, c in blacks)
    return blacks


def _visible_only_output(inp: np.ndarray) -> np.ndarray:
    divider_r = _find_divider_row(inp)
    out = inp.copy()
    top = out[:divider_r].copy()
    top[top == YELLOW] = BLACK
    out[:divider_r] = top
    occupied_yellow = out == YELLOW
    comps = _yellow_components(inp[:divider_r])

    wide = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) >= 2]
    narrow = [comp for comp in comps if int(comp[:, 1].max() - comp[:, 1].min() + 1) == 1]

    for comp in wide:
        shift = divider_r - 1 - int(comp[:, 0].max())
        moved = comp.copy()
        moved[:, 0] += shift
        for r, c in moved:
            out[int(r), int(c)] = YELLOW
            occupied_yellow[int(r), int(c)] = True

    narrow.sort(key=lambda comp: (-int(comp[:, 0].max()), int(comp[0, 1])))
    for comp in narrow:
        c = int(comp[0, 1])
        below_rows = np.where(occupied_yellow[divider_r + 1 :, c])[0]
        floor = inp.shape[0] - 1 if below_rows.size == 0 else divider_r + 1 + int(below_rows.min())
        dest_bottom = floor - 1 if below_rows.size else floor
        shift = dest_bottom - int(comp[:, 0].max())
        moved = comp.copy()
        moved[:, 0] += shift
        out[divider_r, c] = BLACK
        for r, c2 in moved:
            out[int(r), int(c2)] = YELLOW
            occupied_yellow[int(r), int(c2)] = True
    return out


def _validate_instance(
    inp: np.ndarray,
    repaired: np.ndarray,
    expected: np.ndarray,
    period: int,
    base_shape: tuple[int, int],
    base_top: np.ndarray,
) -> None:
    divider_r = _find_divider_row(inp)
    repaired_top = repaired[:divider_r]

    assert _detect_period(inp[:divider_r], ignore={BLACK}) == (period, period)
    assert np.array_equal(
        _reconstruct_periodic(inp[:divider_r], period, period, ignore={BLACK}),
        repaired_top,
    )
    assert _detect_period(repaired_top, ignore={YELLOW}) == base_shape
    assert np.array_equal(
        _reconstruct_periodic(repaired_top, base_shape[0], base_shape[1], ignore={YELLOW}),
        base_top,
    )

    comps = _yellow_components(repaired_top)
    widths = {int(comp[:, 1].max() - comp[:, 1].min() + 1) for comp in comps}
    assert 1 in widths and any(width >= 2 for width in widths)

    assert np.any(inp[:divider_r] == BLACK)
    assert np.any((inp[:divider_r] == BLACK) & (repaired_top == YELLOW))
    assert not np.array_equal(_visible_only_output(inp), expected)
    assert np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(300):
        period = random.choice([3, 4])
        top_h = random.randint(period + 2, 2 * period - 1)
        bottom_h = random.randint(4, 7)
        width = period * random.randint(3, 4) + random.randint(0, period - 1)
        if width > 18:
            width = 18

        base_shape = random.choice(
            [(1, 2), (2, 1), (2, 2)] if period == 3 else [(1, 2), (2, 1), (2, 2), (1, 3), (2, 3)]
        )
        if base_shape[0] > top_h or base_shape[1] > width:
            continue

        base_tile = _random_base_tile(*base_shape)
        try:
            visible_tile, _yellow_mask = _build_visible_tile(period, base_tile)
        except AssertionError:
            continue

        repaired, divider_r, repaired_top, base_top = _build_repaired_grid(
            top_h, width, bottom_h, period, base_tile, visible_tile
        )

        try:
            full_period = _detect_period(repaired_top, ignore=set())
            base_period = _detect_period(repaired_top, ignore={YELLOW})
        except AssertionError:
            continue

        if full_period != (period, period):
            continue
        if base_period != base_shape:
            continue

        blacks = _choose_black_positions(repaired_top, period)
        inp = repaired.copy()
        for r, c in blacks:
            inp[r, c] = BLACK

        expected = _apply_gravity(repaired, base_top, divider_r)
        try:
            _validate_instance(inp, repaired, expected, period, base_shape, base_top)
        except AssertionError:
            continue
        return inp

    raise AssertionError("failed to generate a valid puzzle instance")
