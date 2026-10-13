import numpy as np


GEN_RETRIES = 400
TILE_RETRIES = 500
SIZE_RETRIES = 200
RNG = np.random.default_rng()


def _variants(tile: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(tile, dtype=int)
    lr = np.fliplr(g)
    ud = np.flipud(g)
    rot = np.flipud(lr)
    return [g, lr, ud, rot]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if sep <= start or sep >= length:
            return []
        bands.append((start, sep))
        start = sep + 1
    if start >= length:
        return []
    bands.append((start, length))
    return bands


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"

    height, width = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue
        for tile_h in rows:
            for tile_w in cols:
                if tile_h <= 0 or tile_w <= 0:
                    continue
                if tile_h + 1 >= height or tile_w + 1 >= width:
                    continue

                tile = g[:tile_h, :tile_w]
                board = g[tile_h + 1 :, tile_w + 1 :]
                if tile.size == 0 or board.size == 0:
                    continue

                board_sep_rows = [r for r in range(board.shape[0]) if np.all(board[r, :] == color)]
                board_sep_cols = [c for c in range(board.shape[1]) if np.all(board[:, c] == color)]
                row_bands = _bands(board.shape[0], board_sep_rows)
                col_bands = _bands(board.shape[1], board_sep_cols)
                if not row_bands or not col_bands:
                    continue
                if any(r1 - r0 != tile_h for r0, r1 in row_bands):
                    continue
                if any(c1 - c0 != tile_w for c0, c1 in col_bands):
                    continue

                non_sep = board[board != color]
                if non_sep.size == 0:
                    continue
                vals, counts = np.unique(non_sep, return_counts=True)
                bg = int(vals[np.argmax(counts)])
                candidates.append(
                    {
                        "separator": int(color),
                        "tile_h": int(tile_h),
                        "tile_w": int(tile_w),
                        "tile": tile.copy(),
                        "board": board.copy(),
                        "background": bg,
                        "row_bands": row_bands,
                        "col_bands": col_bands,
                    }
                )

    assert len(candidates) == 1, "expected exactly one valid template/board decomposition"
    return candidates[0]


def _candidate_maps(tile: np.ndarray) -> tuple[list[np.ndarray], dict[int, list[tuple[int, int, int]]], dict[tuple[int, int], list[tuple[int, int]]], dict[int, list[tuple[int, int, int]]]]:
    variants = _variants(tile)
    stack = np.stack(variants, axis=0)
    by_orientation: dict[int, list[tuple[int, int, int]]] = {idx: [] for idx in range(4)}
    by_position: dict[tuple[int, int], list[tuple[int, int]]] = {}
    by_color: dict[int, list[tuple[int, int, int]]] = {}

    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            values = stack[:, r, c]
            for ori, color in enumerate(values):
                if int(np.count_nonzero(values == color)) != 1:
                    continue
                color = int(color)
                by_orientation[ori].append((int(r), int(c), color))
                by_position.setdefault((int(r), int(c)), []).append((ori, color))
                by_color.setdefault(color, []).append((ori, int(r), int(c)))
    return variants, by_orientation, by_position, by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each stamped tile orientation from a single residue pixel and fill the separator-delimited board.

    Input structure:
    - The upper-left corner contains a multicolor rectangular template tile.
    - One full separator row and one full separator column isolate that template from a lower-right board.
    - The board is further partitioned by same-color separator lines into tile-sized regions; each non-empty region contains exactly one non-background residue pixel.

    Rule:
    - Remove the template area and its isolating separator row and column, keeping only the lower-right board.
    - For every non-empty region, compare the residue pixel against the template's four allowed orientations: original, left-right mirror, up-down mirror, and 180-degree rotation.
    - Find the unique orientation whose color matches at that residue coordinate, then stamp that full oriented tile into the region; empty regions stay background and separator lines stay unchanged.

    Key insight:
    - The residue pixel is informative only through its coordinate-color pair: the position narrows the possibilities, and the color at that same position resolves the unique orientation.
    """
    info = _parse_grid(grid)
    tile = info["tile"]
    board = info["board"]
    bg = info["background"]
    row_bands = info["row_bands"]
    col_bands = info["col_bands"]
    variants = _variants(tile)

    out = board.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = board[r0:r1, c0:c1]
            residue_coords = np.argwhere(region != bg)
            if residue_coords.size == 0:
                continue
            assert residue_coords.shape[0] == 1, "expected exactly one residue pixel per non-empty region"
            rr, cc = map(int, residue_coords[0])
            residue_color = int(region[rr, cc])
            matches = [idx for idx, variant in enumerate(variants) if int(variant[rr, cc]) == residue_color]
            assert len(matches) == 1, "expected residue coordinate and color to determine a unique orientation"
            out[r0:r1, c0:c1] = variants[matches[0]]

    return out


def _pick_sizes() -> tuple[int, int, int, int]:
    for _ in range(SIZE_RETRIES):
        tile_h = int(RNG.integers(3, 6))
        tile_w = int(RNG.integers(3, 6))
        board_rows = int(RNG.integers(2, 5))
        board_cols = int(RNG.integers(2, 5))
        if board_rows * board_cols < 8:
            continue
        total_h = (board_rows + 1) * tile_h + board_rows
        total_w = (board_cols + 1) * tile_w + board_cols
        if total_h <= 30 and total_w <= 30:
            return tile_h, tile_w, board_rows, board_cols
    raise RuntimeError("failed to sample valid grid dimensions")


def _build_tile(tile_h: int, tile_w: int, tile_colors: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[int, list[tuple[int, int, int]]], list[tuple[tuple[int, int], list[tuple[int, int]]]], list[tuple[int, list[tuple[int, int, int]]]]]:
    for _ in range(TILE_RETRIES):
        tile = RNG.choice(tile_colors, size=(tile_h, tile_w)).astype(int)
        if np.unique(tile).size < min(3, tile_colors.size):
            continue

        variants, by_orientation, by_position, by_color = _candidate_maps(tile)
        if len({variant.tobytes() for variant in variants}) != 4:
            continue
        if any(len(by_orientation[idx]) < 2 for idx in range(4)):
            continue

        position_groups = [
            (coord, entries)
            for coord, entries in by_position.items()
            if len({ori for ori, _color in entries}) >= 2
        ]
        color_groups = [
            (color, entries)
            for color, entries in by_color.items()
            if len({ori for ori, _r, _c in entries}) >= 2
        ]
        if not position_groups or not color_groups:
            continue
        return tile, variants, by_orientation, position_groups, color_groups
    raise RuntimeError("failed to sample a non-degenerate template tile")


def _add_unique(assignments: list[tuple[int, int, int, int]], item: tuple[int, int, int, int]) -> None:
    if item not in assignments:
        assignments.append(item)


def _has_position_collision(assignments: list[tuple[int, int, int, int]]) -> bool:
    seen: dict[tuple[int, int], set[int]] = {}
    for ori, r, c, _color in assignments:
        seen.setdefault((r, c), set()).add(ori)
    return any(len(orientations) >= 2 for orientations in seen.values())


def _has_color_collision(assignments: list[tuple[int, int, int, int]]) -> bool:
    seen: dict[int, set[int]] = {}
    for ori, _r, _c, color in assignments:
        seen.setdefault(color, set()).add(ori)
    return any(len(orientations) >= 2 for orientations in seen.values())


def _build_assignments(
    by_orientation: dict[int, list[tuple[int, int, int]]],
    position_groups: list[tuple[tuple[int, int], list[tuple[int, int]]]],
    color_groups: list[tuple[int, list[tuple[int, int, int]]]],
    max_nonempty: int,
) -> list[tuple[int, int, int, int]]:
    assignments: list[tuple[int, int, int, int]] = []
    for ori in range(4):
        r, c, color = by_orientation[ori][int(RNG.integers(len(by_orientation[ori])))]
        assignments.append((ori, int(r), int(c), int(color)))

    coord, position_entries = position_groups[int(RNG.integers(len(position_groups)))]
    pos_choices = list(position_entries)
    RNG.shuffle(pos_choices)
    used_orientations: set[int] = set()
    for ori, color in pos_choices:
        if ori in used_orientations:
            continue
        _add_unique(assignments, (int(ori), int(coord[0]), int(coord[1]), int(color)))
        used_orientations.add(int(ori))
        if len(used_orientations) >= 2:
            break

    color, color_entries = color_groups[int(RNG.integers(len(color_groups)))]
    color_choices = list(color_entries)
    RNG.shuffle(color_choices)
    used_orientations.clear()
    for ori, r, c in color_choices:
        if ori in used_orientations:
            continue
        _add_unique(assignments, (int(ori), int(r), int(c), int(color)))
        used_orientations.add(int(ori))
        if len(used_orientations) >= 2:
            break

    if len(assignments) > max_nonempty:
        raise RuntimeError("mandatory residue assignments exceed board capacity")

    target_nonempty = int(RNG.integers(max(6, len(assignments)), max_nonempty + 1))
    attempts = 0
    while len(assignments) < target_nonempty and attempts < 200:
        ori = int(RNG.integers(4))
        r, c, picked_color = by_orientation[ori][int(RNG.integers(len(by_orientation[ori])))]
        _add_unique(assignments, (ori, int(r), int(c), int(picked_color)))
        attempts += 1

    if len(assignments) < target_nonempty:
        raise RuntimeError("failed to sample enough diverse residues")
    if not _has_position_collision(assignments):
        raise RuntimeError("missing anti-shortcut position collision")
    if not _has_color_collision(assignments):
        raise RuntimeError("missing anti-shortcut color collision")
    if {ori for ori, _r, _c, _color in assignments} != {0, 1, 2, 3}:
        raise RuntimeError("expected all four orientations to appear")
    return assignments


def _make_board(board_rows: int, board_cols: int, tile_h: int, tile_w: int, bg: int, separator: int) -> np.ndarray:
    board_h = board_rows * tile_h + (board_rows - 1)
    board_w = board_cols * tile_w + (board_cols - 1)
    board = np.full((board_h, board_w), bg, dtype=int)
    for k in range(1, board_rows):
        board[k * tile_h + (k - 1), :] = separator
    for k in range(1, board_cols):
        board[:, k * tile_w + (k - 1)] = separator
    return board


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        tile_h, tile_w, board_rows, board_cols = _pick_sizes()
        palette = RNG.permutation(10)
        bg = int(palette[0])
        separator = int(palette[1])
        tile_color_count = int(RNG.integers(3, 6))
        tile_colors = palette[2 : 2 + tile_color_count].astype(int)

        try:
            tile, variants, by_orientation, position_groups, color_groups = _build_tile(tile_h, tile_w, tile_colors)
            assignments = _build_assignments(
                by_orientation,
                position_groups,
                color_groups,
                max_nonempty=board_rows * board_cols - 1,
            )
        except RuntimeError:
            continue

        board = _make_board(board_rows, board_cols, tile_h, tile_w, bg, separator)
        expected = board.copy()

        regions = [(rr, cc) for rr in range(board_rows) for cc in range(board_cols)]
        RNG.shuffle(regions)
        used_regions = regions[: len(assignments)]

        for (ori, rr, cc, color), (region_r, region_c) in zip(assignments, used_regions):
            r0 = region_r * (tile_h + 1)
            c0 = region_c * (tile_w + 1)
            board[r0 + rr, c0 + cc] = color
            expected[r0 : r0 + tile_h, c0 : c0 + tile_w] = variants[ori]

        total_h = tile_h + 1 + board.shape[0]
        total_w = tile_w + 1 + board.shape[1]
        full = np.full((total_h, total_w), separator, dtype=int)
        full[:tile_h, :tile_w] = tile
        full[tile_h + 1 :, tile_w + 1 :] = board

        produced = main(full)
        if not np.array_equal(produced, expected):
            continue

        parsed = _parse_grid(full)
        if parsed["background"] != bg or parsed["separator"] != separator:
            continue

        nonempty_regions = 0
        for r0, r1 in parsed["row_bands"]:
            for c0, c1 in parsed["col_bands"]:
                region = parsed["board"][r0:r1, c0:c1]
                residue_coords = np.argwhere(region != bg)
                if residue_coords.size == 0:
                    continue
                nonempty_regions += 1
                if residue_coords.shape[0] != 1:
                    nonempty_regions = -1
                    break
                rr, cc = map(int, residue_coords[0])
                residue_color = int(region[rr, cc])
                match_count = sum(int(variant[rr, cc]) == residue_color for variant in variants)
                if match_count != 1:
                    nonempty_regions = -1
                    break
            if nonempty_regions == -1:
                break

        if nonempty_regions != len(assignments):
            continue
        if board_rows * board_cols == nonempty_regions:
            continue
        return full

    raise RuntimeError("failed to generate a valid residue-orientation tiling input")
