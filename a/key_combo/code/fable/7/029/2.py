from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in patch.tolist())


def _orbit_cells(
    r: int,
    c: int,
    row_s: int,
    col_s: int,
    row_count: int,
    col_count: int,
) -> tuple[tuple[int, int], ...]:
    rows = {int(r)}
    rr = int(row_s) - int(r)
    if 0 <= rr < row_count:
        rows.add(rr)

    cols = {int(c)}
    cc = int(col_s) - int(c)
    if 0 <= cc < col_count:
        cols.add(cc)

    return tuple(sorted((orow, ocol) for orow in rows for ocol in cols))


def _partition_orbits(
    row_count: int,
    col_count: int,
    row_s: int,
    col_s: int,
) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(row_count):
        for c in range(col_count):
            if (r, c) in seen:
                continue
            orbit = _orbit_cells(r, c, row_s, col_s, row_count, col_count)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _lattice_spec_from_color(
    grid: np.ndarray,
    sep_color: int,
) -> tuple[list[int], list[int], int, int] | None:
    height, width = map(int, grid.shape)
    row_lines = [r for r in range(height) if np.all(grid[r, :] == sep_color)]
    col_lines = [c for c in range(width) if np.all(grid[:, c] == sep_color)]
    if len(row_lines) < 2 or len(col_lines) < 2:
        return None
    if row_lines[0] != 0 or row_lines[-1] != height - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != width - 1:
        return None
    row_steps = [b - a for a, b in zip(row_lines, row_lines[1:])]
    col_steps = [b - a for a, b in zip(col_lines, col_lines[1:])]
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None
    if row_steps[0] <= 1 or col_steps[0] <= 1:
        return None
    return row_lines, col_lines, row_steps[0] - 1, col_steps[0] - 1


def _extract_tiles(
    grid: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[list[np.ndarray]], list[list[tuple[tuple[int, ...], ...] | None]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    specs = []
    for sep_color in map(int, np.unique(g)):
        spec = _lattice_spec_from_color(g, sep_color)
        if spec is not None:
            specs.append((int(sep_color), *spec))
    assert len(specs) == 1, "expected one separator lattice color"

    sep_color, row_lines, col_lines, cell_h, cell_w = specs[0]
    non_sep = g[g != sep_color]
    assert non_sep.size > 0
    bg = _most_common(non_sep)

    tiles: list[list[np.ndarray]] = []
    tile_keys: list[list[tuple[tuple[int, ...], ...] | None]] = []
    for i in range(len(row_lines) - 1):
        row_tiles = []
        row_keys = []
        r0 = row_lines[i] + 1
        r1 = row_lines[i + 1]
        for j in range(len(col_lines) - 1):
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1]
            patch = g[r0:r1, c0:c1].copy()
            assert patch.shape == (cell_h, cell_w)
            row_tiles.append(patch)
            row_keys.append(None if np.all(patch == bg) else _patch_key(patch))
        tiles.append(row_tiles)
        tile_keys.append(row_keys)
    return sep_color, bg, row_lines, col_lines, tiles, tile_keys


def _valid_axis_pairs(
    tile_keys: list[list[tuple[tuple[int, ...], ...] | None]],
) -> list[tuple[int, int]]:
    row_count = len(tile_keys)
    col_count = len(tile_keys[0])
    valid: list[tuple[int, int]] = []

    for row_s in range(2 * row_count - 1):
        for col_s in range(2 * col_count - 1):
            ok = True
            seen: set[tuple[int, int]] = set()
            for r in range(row_count):
                for c in range(col_count):
                    if (r, c) in seen:
                        continue
                    orbit = _orbit_cells(r, c, row_s, col_s, row_count, col_count)
                    seen.update(orbit)
                    known = {
                        tile_keys[orow][ocol]
                        for orow, ocol in orbit
                        if tile_keys[orow][ocol] is not None
                    }
                    if len(known) > 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid.append((row_s, col_s))
    return valid


def _infer_axis_constants(
    tile_keys: list[list[tuple[tuple[int, ...], ...] | None]],
) -> tuple[int, int]:
    groups: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = {}
    for r, row in enumerate(tile_keys):
        for c, key in enumerate(row):
            if key is not None:
                groups.setdefault(key, []).append((r, c))

    row_votes: Counter[int] = Counter()
    col_votes: Counter[int] = Counter()
    for cells in groups.values():
        for i in range(len(cells)):
            r1, c1 = cells[i]
            for j in range(i + 1, len(cells)):
                r2, c2 = cells[j]
                if c1 == c2 and r1 != r2:
                    row_votes[int(r1 + r2)] += 1
                if r1 == r2 and c1 != c2:
                    col_votes[int(c1 + c2)] += 1

    assert row_votes and col_votes, "need repeated known stamps to infer both lattice axes"
    row_ranked = row_votes.most_common()
    col_ranked = col_votes.most_common()
    assert len(row_ranked) == 1 or row_ranked[0][1] > row_ranked[1][1], "row axis is ambiguous"
    assert len(col_ranked) == 1 or col_ranked[0][1] > col_ranked[1][1], "column axis is ambiguous"
    return int(row_ranked[0][0]), int(col_ranked[0][0])


def _orbit_consistent(
    tile_keys: list[list[tuple[tuple[int, ...], ...] | None]],
    row_s: int,
    col_s: int,
) -> bool:
    row_count = len(tile_keys)
    col_count = len(tile_keys[0])
    seen: set[tuple[int, int]] = set()
    for r in range(row_count):
        for c in range(col_count):
            if (r, c) in seen:
                continue
            orbit = _orbit_cells(r, c, row_s, col_s, row_count, col_count)
            seen.update(orbit)
            known = {
                tile_keys[orow][ocol]
                for orow, ocol in orbit
                if tile_keys[orow][ocol] is not None
            }
            if len(known) > 1:
                return False
    return True


def _rebuild_grid(
    sep_color: int,
    row_lines: list[int],
    col_lines: list[int],
    tiles: list[list[np.ndarray]],
) -> np.ndarray:
    height = row_lines[-1] + 1
    width = col_lines[-1] + 1
    out = np.full((height, width), int(sep_color), dtype=int)
    for i in range(len(row_lines) - 1):
        r0 = row_lines[i] + 1
        r1 = row_lines[i + 1]
        for j in range(len(col_lines) - 1):
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1]
            out[r0:r1, c0:c1] = tiles[i][j]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the missing stamp tiles by symmetry on the separator-defined lattice.

    Input structure:
    - Full-width and full-height separator lines partition the canvas into equal-sized tiles.
    - Some tiles contain a multicolor stamp on a uniform tile background, while other tiles are pure background.
    - The set of occupied tile coordinates is symmetric under one row-index reflection and one column-index reflection, but those reflection constants must be inferred from the tile lattice rather than from pixel geometry.

    Rule:
    - Detect the separator lattice and treat each tile interior as one stamp slot.
    - Infer the unique row and column reflection constants for which every orbit of tiles contains at most one non-empty stamp pattern.
    - For each orbit that contains a known stamp, copy that exact tile patch by translation into every empty tile in the same orbit, without mirroring or rotating it.
    - Leave orbits that are entirely background unchanged.

    Key insight:
    - The symmetry lives in tile-coordinate space, not in raw pixel space, and the missing stamp must be copied unchanged from any known member of the full two-axis orbit.
    """
    sep_color, bg, row_lines, col_lines, tiles, tile_keys = _extract_tiles(grid)
    row_s, col_s = _infer_axis_constants(tile_keys)
    assert _orbit_consistent(tile_keys, row_s, col_s), "inferred axes conflict with known stamp orbits"

    row_count = len(tiles)
    col_count = len(tiles[0])
    out_tiles = [[patch.copy() for patch in row] for row in tiles]
    seen: set[tuple[int, int]] = set()

    for r in range(row_count):
        for c in range(col_count):
            if (r, c) in seen:
                continue
            orbit = _orbit_cells(r, c, row_s, col_s, row_count, col_count)
            seen.update(orbit)
            known_patches = [
                tiles[orow][ocol]
                for orow, ocol in orbit
                if tile_keys[orow][ocol] is not None
            ]
            if not known_patches:
                continue
            stamp = known_patches[0]
            for patch in known_patches[1:]:
                assert np.array_equal(patch, stamp)
            for orow, ocol in orbit:
                if tile_keys[orow][ocol] is None:
                    out_tiles[orow][ocol] = stamp.copy()

    return _rebuild_grid(sep_color, row_lines, col_lines, out_tiles)


_BG = 0
_SEP = 5
_STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_MAX_GEN_ATTEMPTS = 500


def _lattice_options() -> list[tuple[int, int, int, int]]:
    options = []
    for row_count in range(4, 7):
        for col_count in range(4, 7):
            for cell_h in range(3, 6):
                for cell_w in range(3, 6):
                    height = row_count * cell_h + row_count + 1
                    width = col_count * cell_w + col_count + 1
                    if height <= 30 and width <= 30:
                        options.append((row_count, col_count, cell_h, cell_w))
    return options


_LATTICE_OPTIONS = _lattice_options()


def _choose_axis_constant(n: int) -> int:
    all_values = list(range(2 * n - 1))
    off_center = [value for value in all_values if value != n - 1]
    if off_center and random.random() < 0.75:
        return random.choice(off_center)
    return random.choice(all_values)


def _make_pattern(
    cell_h: int,
    cell_w: int,
    used: set[tuple[tuple[int, ...], ...]],
) -> np.ndarray | None:
    area = cell_h * cell_w
    for _ in range(200):
        patch = np.full((cell_h, cell_w), _BG, dtype=int)
        color_count = random.randint(2, min(4, len(_STAMP_COLORS)))
        palette = random.sample(_STAMP_COLORS, color_count)
        fill_count = random.randint(3, min(area - 1, max(4, area - 2)))
        cells = random.sample(range(area), fill_count)
        for idx in cells:
            r, c = divmod(idx, cell_w)
            patch[r, c] = random.choice(palette)

        colored = patch != _BG
        box = _bbox(colored)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r0 == r1 or c0 == c1:
            continue
        if np.array_equal(patch, patch[::-1, :]):
            continue
        if np.array_equal(patch, patch[:, ::-1]):
            continue
        if np.array_equal(patch, patch[::-1, ::-1]):
            continue

        key = _patch_key(patch)
        if key in used:
            continue
        used.add(key)
        return patch
    return None


def _blank_tiles(row_count: int, col_count: int, cell_h: int, cell_w: int) -> list[list[np.ndarray]]:
    return [[np.full((cell_h, cell_w), _BG, dtype=int) for _ in range(col_count)] for _ in range(row_count)]


def _render_tiles(
    row_count: int,
    col_count: int,
    occupied: dict[tuple[tuple[int, int], ...], np.ndarray],
    visible: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]],
    *,
    full: bool,
    cell_h: int,
    cell_w: int,
) -> list[list[np.ndarray]]:
    tiles = _blank_tiles(row_count, col_count, cell_h, cell_w)
    for orbit, patch in occupied.items():
        cells = orbit if full else tuple(sorted(visible[orbit]))
        for r, c in cells:
            tiles[r][c] = patch.copy()
    return tiles


def _render_grid_from_tiles(
    row_count: int,
    col_count: int,
    cell_h: int,
    cell_w: int,
    tiles: list[list[np.ndarray]],
) -> np.ndarray:
    height = row_count * cell_h + row_count + 1
    width = col_count * cell_w + col_count + 1
    out = np.full((height, width), _SEP, dtype=int)
    for r in range(row_count):
        r0 = 1 + r * (cell_h + 1)
        for c in range(col_count):
            c0 = 1 + c * (cell_w + 1)
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = tiles[r][c]
    return out


def _choose_occupied_orbits(
    orbits: list[tuple[tuple[int, int], ...]],
) -> tuple[list[tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    size4 = [orbit for orbit in orbits if len(orbit) == 4]
    if len(size4) < 2:
        raise ValueError("need at least two size-4 orbits")

    special = random.choice(size4)
    witness = random.choice([orbit for orbit in size4 if orbit != special])

    max_count = min(len(orbits) - 1, 7)
    min_count = min(max(4, len(orbits) // 2), max_count)
    occupy_count = random.randint(min_count, max_count)

    chosen = {special, witness}
    extra_pool = [orbit for orbit in orbits if orbit not in chosen]
    random.shuffle(extra_pool)
    for orbit in extra_pool:
        if len(chosen) >= occupy_count:
            break
        chosen.add(orbit)

    occupied_orbits = list(chosen)
    random.shuffle(occupied_orbits)
    return occupied_orbits, special, witness


def _choose_visible_cells(
    occupied_orbits: list[tuple[tuple[int, int], ...]],
    special: tuple[tuple[int, int], ...],
    witness: tuple[tuple[int, int], ...],
) -> dict[tuple[tuple[int, int], ...], set[tuple[int, int]]]:
    visible: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {}
    for orbit in occupied_orbits:
        if orbit == special:
            visible[orbit] = {random.choice(orbit)}
            continue
        if orbit == witness:
            visible[orbit] = set(orbit)
            continue
        size = len(orbit)
        if size == 1:
            visible[orbit] = {orbit[0]}
            continue
        if size == 2:
            count = random.choices([1, 2], weights=[3, 2], k=1)[0]
        else:
            count = random.choices([1, 2, 4], weights=[3, 3, 2], k=1)[0]
            count = min(count, size)
        visible[orbit] = set(random.sample(list(orbit), count))

    hidden_total = sum(len(orbit) - len(visible[orbit]) for orbit in occupied_orbits)
    if hidden_total == 0:
        orbit = max((orbit for orbit in occupied_orbits if orbit != witness), key=len, default=witness)
        if orbit != witness and len(orbit) > 1:
            visible[orbit] = set(list(orbit)[:-1])
    return visible


def _requires_full_orbit_reasoning(
    special: tuple[tuple[int, int], ...],
    visible: set[tuple[int, int]],
    row_s: int,
    col_s: int,
) -> bool:
    if len(special) != 4 or len(visible) != 1:
        return False
    known = next(iter(visible))
    for cell in special:
        if cell in visible:
            continue
        rr = (row_s - cell[0], cell[1])
        cc = (cell[0], col_s - cell[1])
        both = (row_s - cell[0], col_s - cell[1])
        if rr in special and cc in special and both in special:
            if rr not in visible and cc not in visible and both in visible:
                return True
    return False


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(_MAX_GEN_ATTEMPTS):
        row_count, col_count, cell_h, cell_w = random.choice(_LATTICE_OPTIONS)
        row_s = _choose_axis_constant(row_count)
        col_s = _choose_axis_constant(col_count)
        orbits = _partition_orbits(row_count, col_count, row_s, col_s)
        if sum(len(orbit) == 4 for orbit in orbits) < 2:
            continue

        try:
            occupied_orbits, special, witness = _choose_occupied_orbits(orbits)
        except ValueError:
            continue

        visible = _choose_visible_cells(occupied_orbits, special, witness)
        if not _requires_full_orbit_reasoning(special, visible[special], row_s, col_s):
            continue

        used: set[tuple[tuple[int, ...], ...]] = set()
        occupied: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
        ok = True
        for orbit in occupied_orbits:
            patch = _make_pattern(cell_h, cell_w, used)
            if patch is None:
                ok = False
                break
            occupied[orbit] = patch
        if not ok:
            continue

        input_tiles = _render_tiles(
            row_count,
            col_count,
            occupied,
            visible,
            full=False,
            cell_h=cell_h,
            cell_w=cell_w,
        )
        output_tiles = _render_tiles(
            row_count,
            col_count,
            occupied,
            visible,
            full=True,
            cell_h=cell_h,
            cell_w=cell_w,
        )
        inp = _render_grid_from_tiles(row_count, col_count, cell_h, cell_w, input_tiles)
        out = _render_grid_from_tiles(row_count, col_count, cell_h, cell_w, output_tiles)

        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue

        _sep, _bg, _rows, _cols, _tiles, tile_keys = _extract_tiles(inp)
        inferred = _infer_axis_constants(tile_keys)
        if inferred != (row_s, col_s):
            continue
        if not _orbit_consistent(tile_keys, row_s, col_s):
            continue

        hidden_cells = [
            cell
            for orbit in occupied_orbits
            for cell in orbit
            if cell not in visible[orbit]
        ]
        if not hidden_cells:
            continue
        if len(occupied_orbits) >= len(orbits):
            continue

        meta = {
            "row_count": row_count,
            "col_count": col_count,
            "cell_h": cell_h,
            "cell_w": cell_w,
            "row_s": row_s,
            "col_s": col_s,
            "orbit_sizes": [len(orbit) for orbit in occupied_orbits],
            "hidden_count": len(hidden_cells),
            "special_size4_single_known": special,
            "occupied_count": len(occupied_orbits),
            "total_orbits": len(orbits),
        }
        return inp, out, meta
    raise RuntimeError("failed to generate a unique valid lattice-stamp case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
