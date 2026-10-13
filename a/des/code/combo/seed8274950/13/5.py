from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
ARC_COLORS = tuple(range(10))


def _most_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _uniform_step(indices: list[int]) -> int | None:
    if len(indices) < 2:
        return None
    diffs = [b - a for a, b in zip(indices, indices[1:])]
    if len(set(diffs)) != 1:
        return None
    return int(diffs[0])


def _all_same(values: np.ndarray) -> bool:
    return bool(np.all(values == values.flat[0]))


def _parse_password_panel(panel: np.ndarray) -> tuple[int, list[int]] | None:
    if panel.ndim != 2 or panel.shape[0] < 2 or panel.shape[1] < 4:
        return None

    bg = _most_common_color(panel)
    counts: list[int] = []
    for row in panel:
        count = int(np.count_nonzero(row != bg))
        if count < 2 or count > 4:
            return None
        counts.append(count)
    return bg, counts


def _parse_decoder_panel(panel: np.ndarray, bg: int) -> dict | None:
    if panel.ndim != 2 or panel.shape[0] < 5 or panel.shape[1] < 5:
        return None
    if not _all_same(panel[0]) or not _all_same(panel[-1]):
        return None

    line_color = int(panel[0, 0])
    if line_color == bg or int(panel[-1, 0]) != line_color:
        return None

    line_rows = [r for r in range(panel.shape[0]) if _all_same(panel[r]) and int(panel[r, 0]) == line_color]
    line_cols = [c for c in range(panel.shape[1]) if _all_same(panel[:, c]) and int(panel[0, c]) == line_color]
    row_step = _uniform_step(line_rows)
    col_step = _uniform_step(line_cols)
    if row_step is None or col_step is None or row_step < 2 or col_step < 2:
        return None
    if line_rows[0] != 0 or line_rows[-1] != panel.shape[0] - 1:
        return None
    if line_cols[0] != 0 or line_cols[-1] != panel.shape[1] - 1:
        return None

    cell_h = row_step - 1
    cell_w = col_step - 1
    cell_colors = np.full((len(line_rows) - 1, len(line_cols) - 1), bg, dtype=int)

    for rr in range(len(line_rows) - 1):
        r0 = line_rows[rr] + 1
        r1 = line_rows[rr + 1]
        if r1 - r0 != cell_h:
            return None
        for cc in range(len(line_cols) - 1):
            c0 = line_cols[cc] + 1
            c1 = line_cols[cc + 1]
            if c1 - c0 != cell_w:
                return None
            patch = panel[r0:r1, c0:c1]
            if patch.shape != (cell_h, cell_w) or not _all_same(patch):
                return None
            color = int(patch[0, 0])
            if color == line_color:
                return None
            cell_colors[rr, cc] = color

    markers: list[int | None] = []
    for cc in range(cell_colors.shape[1]):
        colors = {int(v) for v in cell_colors[:, cc] if int(v) != bg}
        if len(colors) > 1:
            return None
        markers.append(next(iter(colors)) if colors else None)

    if cell_colors.shape[0] < 4 or cell_colors.shape[1] < 2:
        return None

    return {
        "line_color": line_color,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "cell_rows": int(cell_colors.shape[0]),
        "cell_cols": int(cell_colors.shape[1]),
        "cell_colors": cell_colors,
        "markers": markers,
    }


def _decode_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    candidates: list[dict] = []
    for sep_row in range(1, g.shape[0] - 2):
        if not _all_same(g[sep_row]):
            continue
        password = _parse_password_panel(g[:sep_row])
        if password is None:
            continue
        bg, counts = password
        sep_color = int(g[sep_row, 0])
        if sep_color == bg:
            continue
        decoder = _parse_decoder_panel(g[sep_row + 1 :], bg)
        if decoder is None:
            continue
        if decoder["line_color"] == sep_color:
            continue
        if len(counts) != decoder["cell_cols"]:
            continue
        candidates.append(
            {
                "bg": bg,
                "sep_row": sep_row,
                "sep_color": sep_color,
                "counts": counts,
                "decoder": decoder,
            }
        )

    if len(candidates) != 1:
        raise AssertionError("could not uniquely parse the puzzle structure")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the lower grid column-by-column using the pixel counts from the upper panel.

    Each row in the top panel contains 2-4 colored pixels; that count becomes the
    fill height for the corresponding decoder-panel column. A marked decoder
    column is rewritten so exactly the bottom N cells use that column's marker
    color; unmarked columns are left unchanged.
    """
    g = np.asarray(grid, dtype=int)
    info = _decode_structure(g)
    out = g.copy()

    bg = int(info["bg"])
    sep_row = int(info["sep_row"])
    counts = [int(v) for v in info["counts"]]
    decoder = info["decoder"]
    line_rows = decoder["line_rows"]
    line_cols = decoder["line_cols"]
    cell_rows = int(decoder["cell_rows"])

    bottom = out[sep_row + 1 :]
    for cc, marker in enumerate(decoder["markers"]):
        if marker is None:
            continue
        c0 = line_cols[cc] + 1
        c1 = line_cols[cc + 1]
        for rr in range(cell_rows):
            r0 = line_rows[rr] + 1
            r1 = line_rows[rr + 1]
            bottom[r0:r1, c0:c1] = bg
        fill_from = cell_rows - counts[cc]
        for rr in range(fill_from, cell_rows):
            r0 = line_rows[rr] + 1
            r1 = line_rows[rr + 1]
            bottom[r0:r1, c0:c1] = int(marker)

    return out


def _random_palette() -> tuple[int, int, int, list[int]]:
    bg, sep_color, line_color = random.sample(ARC_COLORS, 3)
    remaining = [c for c in ARC_COLORS if c not in {bg, sep_color, line_color}]
    return int(bg), int(sep_color), int(line_color), remaining


def _pick_marker_row(cell_rows: int, fill_height: int) -> int:
    low_rows = list(range(max(0, cell_rows - fill_height), cell_rows))
    high_rows = list(range(0, max(0, cell_rows - fill_height)))
    if high_rows and random.random() < 0.5:
        return int(random.choice(high_rows))
    return int(random.choice(low_rows))


def _draw_decoder_panel(
    cell_rows: int,
    cell_cols: int,
    cell_size: int,
    bg: int,
    line_color: int,
    marked_cols: set[int],
    counts: list[int],
    marker_colors: dict[int, int],
) -> np.ndarray:
    height = cell_rows * (cell_size + 1) + 1
    width = cell_cols * (cell_size + 1) + 1
    panel = np.full((height, width), bg, dtype=int)

    for r in range(0, height, cell_size + 1):
        panel[r, :] = line_color
    for c in range(0, width, cell_size + 1):
        panel[:, c] = line_color

    for cc in marked_cols:
        rr = _pick_marker_row(cell_rows, counts[cc])
        r0 = rr * (cell_size + 1) + 1
        c0 = cc * (cell_size + 1) + 1
        panel[r0 : r0 + cell_size, c0 : c0 + cell_size] = int(marker_colors[cc])

    return panel


def _draw_password_panel(
    rows: int,
    width: int,
    bg: int,
    row_colors: list[int],
    counts: list[int],
) -> np.ndarray:
    panel = np.full((rows, width), bg, dtype=int)
    for rr in range(rows):
        cols = sorted(random.sample(range(width), counts[rr]))
        panel[rr, cols] = int(row_colors[rr])
    return panel


def _gen_attempt() -> np.ndarray | None:
    cell_cols = random.randint(3, 6)
    cell_rows = random.randint(4, 7)
    cell_size = random.randint(1, 3)

    bg, sep_color, line_color, palette = _random_palette()
    if len(palette) < cell_cols:
        return None

    counts = [random.randint(2, 4) for _ in range(cell_cols)]
    if len(set(counts)) == 1:
        idx = random.randrange(cell_cols)
        choices = [v for v in (2, 3, 4) if v != counts[idx]]
        counts[idx] = random.choice(choices)

    marked_count = random.randint(1, cell_cols - 1)
    marked_cols = set(random.sample(range(cell_cols), marked_count))

    row_colors = random.sample(palette, cell_cols)
    marker_colors = {cc: int(random.choice(palette)) for cc in marked_cols}

    decoder = _draw_decoder_panel(
        cell_rows=cell_rows,
        cell_cols=cell_cols,
        cell_size=cell_size,
        bg=bg,
        line_color=line_color,
        marked_cols=marked_cols,
        counts=counts,
        marker_colors=marker_colors,
    )
    password = _draw_password_panel(
        rows=cell_cols,
        width=decoder.shape[1],
        bg=bg,
        row_colors=row_colors,
        counts=counts,
    )

    grid = np.vstack(
        [
            password,
            np.full((1, decoder.shape[1]), sep_color, dtype=int),
            decoder,
        ]
    )

    try:
        info = _decode_structure(grid)
    except AssertionError:
        return None

    if info["counts"] != counts:
        return None
    if int(info["decoder"]["cell_rows"]) != cell_rows or int(info["decoder"]["cell_cols"]) != cell_cols:
        return None
    if sum(marker is None for marker in info["decoder"]["markers"]) == 0:
        return None
    if sum(marker is not None for marker in info["decoder"]["markers"]) == 0:
        return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid decoder-panel puzzle")
