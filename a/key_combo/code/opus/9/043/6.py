from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for stop in [*separators, length]:
        spans.append((start, stop - 1))
        start = stop + 1
    return spans


def _find_separator_structure(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    greyish = np.isin(grid, (GREY, ORANGE))
    sep_rows = [r for r in range(grid.shape[0]) if np.all(greyish[r, :])]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(greyish[:, c])]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 separator grid"

    row_spans = _spans(grid.shape[0], sep_rows)
    col_spans = _spans(grid.shape[1], sep_cols)
    assert len(row_spans) == 3 and len(col_spans) == 3

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "content cells must be equal-sized"
    return sep_rows, sep_cols, row_spans, col_spans


def _cell_bounds(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], row_idx: int, col_idx: int) -> tuple[int, int, int, int]:
    r0, r1 = row_spans[row_idx]
    c0, c1 = col_spans[col_idx]
    return r0, c0, r1, c1


def _masked_stamp(template: np.ndarray, excluded_rows: set[int], excluded_cols: set[int]) -> np.ndarray:
    stamped = template.copy()
    if excluded_rows:
        stamped[np.array(sorted(excluded_rows)), :] = BLACK
    if excluded_cols:
        stamped[:, np.array(sorted(excluded_cols))] = BLACK
    return stamped


def _control_masks(
    grid: np.ndarray,
    row_idx: int,
    col_idx: int,
    sep_rows: list[int],
    sep_cols: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[set[int], set[int]]:
    r0, c0, r1, c1 = _cell_bounds(row_spans, col_spans, row_idx, col_idx)
    excluded_rows: set[int] = set()
    excluded_cols: set[int] = set()

    if row_idx > 0:
        sep_r = sep_rows[row_idx - 1]
        for c in range(c0, c1 + 1):
            if grid[sep_r, c] == ORANGE:
                excluded_cols.add(c - c0)
    if row_idx < 2:
        sep_r = sep_rows[row_idx]
        for c in range(c0, c1 + 1):
            if grid[sep_r, c] == ORANGE:
                excluded_cols.add(c - c0)

    if col_idx > 0:
        sep_c = sep_cols[col_idx - 1]
        for r in range(r0, r1 + 1):
            if grid[r, sep_c] == ORANGE:
                excluded_rows.add(r - r0)
    if col_idx < 2:
        sep_c = sep_cols[col_idx]
        for r in range(r0, r1 + 1):
            if grid[r, sep_c] == ORANGE:
                excluded_rows.add(r - r0)

    return excluded_rows, excluded_cols


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    sep_rows, sep_cols, row_spans, col_spans = _find_separator_structure(g)
    template_cell: tuple[int, int] | None = None
    template: np.ndarray | None = None
    targets: list[dict] = []

    for row_idx in range(3):
        for col_idx in range(3):
            r0, c0, r1, c1 = _cell_bounds(row_spans, col_spans, row_idx, col_idx)
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(tile)))
            has_template_color = bool(colors & {BLUE, GREEN})
            has_target = RED in colors

            assert not (has_template_color and has_target), "template and target markers must be separate"

            if has_template_color:
                assert colors.issubset({BLACK, BLUE, GREEN}), "template tile may only use black, blue, and green"
                assert BLUE in colors and GREEN in colors, "template must show both non-black colors"
                assert template is None, "expected exactly one template tile"
                assert np.all(np.any(tile != BLACK, axis=1)), "each template row must contain non-black pixels"
                assert np.all(np.any(tile != BLACK, axis=0)), "each template column must contain non-black pixels"
                template = tile.copy()
                template_cell = (row_idx, col_idx)
            elif has_target:
                assert colors.issubset({BLACK, RED}), "target tiles may only contain a red marker"
                reds = np.argwhere(tile == RED)
                assert len(reds) == 1, "each target tile must contain exactly one red marker"
                excluded_rows, excluded_cols = _control_masks(g, row_idx, col_idx, sep_rows, sep_cols, row_spans, col_spans)
                assert excluded_rows or excluded_cols, "each target must have at least one orange control"
                targets.append(
                    {
                        "cell": (row_idx, col_idx),
                        "bounds": (r0, c0, r1, c1),
                        "marker_pos": tuple(map(int, reds[0])),
                        "excluded_rows": excluded_rows,
                        "excluded_cols": excluded_cols,
                    }
                )
            else:
                assert colors == {BLACK}, "non-template, non-target tiles must be empty"

    assert template is not None and template_cell is not None, "expected one template tile"
    assert 2 <= len(targets) <= 4, "expected two to four target tiles"

    target_cells = {target["cell"] for target in targets}
    orange_positions = np.argwhere(g == ORANGE)
    for r, c in orange_positions:
        r = int(r)
        c = int(c)
        assert r not in sep_rows or c not in sep_cols, "orange controls cannot sit on separator intersections"

        if r in sep_rows:
            col_idx = next(idx for idx, (c0, c1) in enumerate(col_spans) if c0 <= c <= c1)
            row_pair = (0, 1) if r == sep_rows[0] else (1, 2)
            adjacent = {(row_pair[0], col_idx), (row_pair[1], col_idx)}
        else:
            row_idx = next(idx for idx, (r0, r1) in enumerate(row_spans) if r0 <= r <= r1)
            col_pair = (0, 1) if c == sep_cols[0] else (1, 2)
            adjacent = {(row_idx, col_pair[0]), (row_idx, col_pair[1])}

        assert adjacent & target_cells, "every orange control must border at least one target tile"
        assert template_cell not in adjacent, "template must not share a controlled separator"

    return {
        "grid": g,
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "template": template,
        "template_cell": template_cell,
        "targets": targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the template into each red-marked tile while orange separator pixels erase local template rows and columns.

    Input structure:
    - A black background is split into a 3x3 grid of equal content cells by full grey separator rows and columns.
    - Exactly one cell contains a blue-and-green template sprite, two to four other cells each contain one red marker pixel, and all remaining cells are empty.
    - Some separator positions next to target cells are orange controls instead of grey; horizontal controls encode template columns and vertical controls encode template rows.

    Rule:
    - Detect the unique template cell and every red-marked target cell.
    - For each target, collect orange positions on its adjacent separator segments: orange on a horizontal separator removes the template column at that offset, while orange on a vertical separator removes the template row at that offset.
    - Copy the template into each target cell aligned to the cell's upper-left corner, replace every excluded template row and column with black, overwrite the red marker with that masked copy, and restore all orange separator pixels back to grey.

    Key insight:
    - The same template is reused everywhere, but each target gets a different local row/column erasure mask defined by the orange pixels on the separator lines surrounding that cell.
    """
    parsed = _parse_grid(grid)
    out = parsed["grid"].copy()
    template = parsed["template"]

    out[out == ORANGE] = GREY
    for target in parsed["targets"]:
        r0, c0, r1, c1 = target["bounds"]
        masked = _masked_stamp(template, target["excluded_rows"], target["excluded_cols"])
        out[r0 : r1 + 1, c0 : c1 + 1] = masked

    return out


def _random_template(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    for _ in range(200):
        fill_mask = rng.random((height, width)) < rng.uniform(0.45, 0.7)
        colors = rng.choice([BLUE, GREEN], size=(height, width))
        template = np.where(fill_mask, colors, BLACK)

        if not np.all(np.any(template != BLACK, axis=1)):
            continue
        if not np.all(np.any(template != BLACK, axis=0)):
            continue
        if BLUE not in template or GREEN not in template:
            continue
        if len({tuple(row.tolist()) for row in template}) < height - 1:
            continue
        if len({tuple(col.tolist()) for col in template.T}) < width - 1:
            continue
        return template
    return None


def _choose_targets(rng: np.random.Generator, template_cell: tuple[int, int]) -> list[tuple[int, int]]:
    parity = (template_cell[0] + template_cell[1]) % 2
    candidates = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if (r, c) != template_cell and (r + c) % 2 == parity
    ]
    assert len(candidates) in (3, 4)

    if len(candidates) == 4:
        target_count = int(rng.choice([2, 3, 4], p=[0.15, 0.45, 0.40]))
    else:
        target_count = int(rng.choice([2, 3], p=[0.4, 0.6]))
    pick = rng.choice(len(candidates), size=target_count, replace=False)
    return [candidates[int(i)] for i in pick]


def _target_modes(rng: np.random.Generator, count: int) -> list[str]:
    if count == 2:
        modes = ["row_only", "col_only"]
    elif count == 3:
        modes = ["row_only", "col_only", "both"]
    else:
        modes = ["row_only", "col_only", "both", str(rng.choice(["row_only", "col_only", "both"]))]
    rng.shuffle(modes)
    return modes


def _pick_mask(
    rng: np.random.Generator,
    row_size: int,
    col_size: int,
    mode: str,
) -> tuple[set[int], set[int]]:
    max_rows = min(3, row_size - 1)
    max_cols = min(3, col_size - 1)
    row_count = 0
    col_count = 0

    if mode in {"row_only", "both"}:
        row_count = int(rng.integers(1, max_rows + 1))
    if mode in {"col_only", "both"}:
        col_count = int(rng.integers(1, max_cols + 1))

    rows = set(map(int, rng.choice(row_size, size=row_count, replace=False))) if row_count else set()
    cols = set(map(int, rng.choice(col_size, size=col_count, replace=False))) if col_count else set()
    return rows, cols


def _place_controls(
    grid: np.ndarray,
    rng: np.random.Generator,
    cell: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
    excluded_rows: set[int],
    excluded_cols: set[int],
) -> None:
    row_idx, col_idx = cell
    r0, c0, r1, c1 = _cell_bounds(row_spans, col_spans, row_idx, col_idx)

    vertical_sides: list[int] = []
    horizontal_sides: list[int] = []
    if col_idx > 0:
        vertical_sides.append(sep_cols[col_idx - 1])
    if col_idx < 2:
        vertical_sides.append(sep_cols[col_idx])
    if row_idx > 0:
        horizontal_sides.append(sep_rows[row_idx - 1])
    if row_idx < 2:
        horizontal_sides.append(sep_rows[row_idx])

    for row_offset in excluded_rows:
        side = int(rng.choice(vertical_sides))
        grid[r0 + row_offset, side] = ORANGE
    for col_offset in excluded_cols:
        side = int(rng.choice(horizontal_sides))
        grid[side, c0 + col_offset] = ORANGE

    assert np.any(grid[r0 : r1 + 1, [c for c in vertical_sides]] == ORANGE) or np.any(
        grid[[r for r in horizontal_sides], c0 : c1 + 1] == ORANGE
    )


def _pick_marker_position(
    rng: np.random.Generator,
    template: np.ndarray,
    excluded_rows: set[int],
    excluded_cols: set[int],
    preferred_kind: str,
) -> tuple[int, int]:
    stamped = _masked_stamp(template, excluded_rows, excluded_cols)
    keep_nonblack = np.argwhere(stamped != BLACK)
    erased_nonblack = np.argwhere((template != BLACK) & (stamped == BLACK))
    template_black = np.argwhere(template == BLACK)

    pools = {
        "kept": keep_nonblack,
        "erased": erased_nonblack,
        "black": template_black,
    }
    order = [preferred_kind, "kept", "erased", "black"]
    for key in order:
        if len(pools[key]) > 0:
            choice = pools[key][int(rng.integers(len(pools[key])))]
            return int(choice[0]), int(choice[1])
    raise AssertionError("unable to place red marker")


def _validate_generated(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    template = parsed["template"]
    targets = parsed["targets"]

    assert any(target["excluded_rows"] for target in targets), "need at least one row-exclusion example"
    assert any(target["excluded_cols"] for target in targets), "need at least one column-exclusion example"

    control_signatures = set()
    tile_signatures = set()
    for target in targets:
        masked = _masked_stamp(template, target["excluded_rows"], target["excluded_cols"])
        assert not np.array_equal(masked, template), "each target must visibly differ from the full template"

        control_sig = (
            tuple(sorted(target["excluded_rows"])),
            tuple(sorted(target["excluded_cols"])),
        )
        tile_sig = masked.tobytes()
        control_signatures.add(control_sig)
        tile_signatures.add(tile_sig)

    assert len(control_signatures) == len(targets), "target control masks must be pairwise distinct"
    assert len(tile_signatures) == len(targets), "masked stamp results must be pairwise distinct"

    out = main(grid)
    assert np.all(out[np.isin(grid, (GREY, ORANGE))] == GREY), "separator lines must normalize to grey"

    seen_change = False
    for target in targets:
        r0, c0, r1, c1 = target["bounds"]
        masked = _masked_stamp(template, target["excluded_rows"], target["excluded_cols"])
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], masked)
        if not np.array_equal(masked, grid[r0 : r1 + 1, c0 : c1 + 1]):
            seen_change = True
    assert seen_change, "output must differ from input"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        cell_h = int(rng.integers(5, 9))
        cell_w = int(rng.integers(5, 9))
        grid = np.full((3 * cell_h + 2, 3 * cell_w + 2), BLACK, dtype=int)

        sep_rows = [cell_h, 2 * cell_h + 1]
        sep_cols = [cell_w, 2 * cell_w + 1]
        grid[sep_rows, :] = GREY
        grid[:, sep_cols] = GREY

        row_spans = _spans(grid.shape[0], sep_rows)
        col_spans = _spans(grid.shape[1], sep_cols)

        template = _random_template(rng, cell_h, cell_w)
        if template is None:
            continue

        template_cell = (int(rng.integers(3)), int(rng.integers(3)))
        targets = _choose_targets(rng, template_cell)
        modes = _target_modes(rng, len(targets))

        target_specs: list[tuple[tuple[int, int], set[int], set[int]]] = []
        used_masks: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        used_tiles: set[bytes] = set()
        ok = True
        for cell, mode in zip(targets, modes):
            for _mask_try in range(80):
                excluded_rows, excluded_cols = _pick_mask(rng, cell_h, cell_w, mode)
                masked = _masked_stamp(template, excluded_rows, excluded_cols)
                mask_sig = (tuple(sorted(excluded_rows)), tuple(sorted(excluded_cols)))
                if np.array_equal(masked, template):
                    continue
                if mask_sig in used_masks or masked.tobytes() in used_tiles:
                    continue
                used_masks.add(mask_sig)
                used_tiles.add(masked.tobytes())
                target_specs.append((cell, excluded_rows, excluded_cols))
                break
            else:
                ok = False
                break
        if not ok:
            continue

        tr, tc = template_cell
        r0, c0, r1, c1 = _cell_bounds(row_spans, col_spans, tr, tc)
        grid[r0 : r1 + 1, c0 : c1 + 1] = template

        marker_kinds = ["kept", "erased", "black", "kept"]
        rng.shuffle(marker_kinds)
        for idx, (cell, excluded_rows, excluded_cols) in enumerate(target_specs):
            _place_controls(grid, rng, cell, row_spans, col_spans, sep_rows, sep_cols, excluded_rows, excluded_cols)

            cr, cc = cell
            r0, c0, r1, c1 = _cell_bounds(row_spans, col_spans, cr, cc)
            marker_r, marker_c = _pick_marker_position(
                rng,
                template,
                excluded_rows,
                excluded_cols,
                marker_kinds[idx % len(marker_kinds)],
            )
            grid[r0 + marker_r, c0 + marker_c] = RED

        try:
            _validate_generated(grid)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid input for local masked stamping")
