from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
CELL_SIZE = 5
MAX_ATTEMPTS = 400
SIDES = ("left", "right", "top", "bottom")
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _spans_from_separator_lines(full_lines: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in full_lines + [limit]:
        if idx > start:
            spans.append((start, idx - 1))
        start = idx + 1
    return spans


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    row_spans = _spans_from_separator_lines(row_lines, grid.shape[0])
    col_spans = _spans_from_separator_lines(col_lines, grid.shape[1])
    assert row_spans and col_spans, "expected a separator-partitioned lattice"
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "lattice cells must have uniform size"
    assert row_sizes == {CELL_SIZE} and col_sizes == {CELL_SIZE}, "expected 5x5 cells"
    expected_h = len(row_spans) * CELL_SIZE + (len(row_spans) - 1)
    expected_w = len(col_spans) * CELL_SIZE + (len(col_spans) - 1)
    assert grid.shape == (expected_h, expected_w), "separator spacing must match regular 5x5 cells"
    return row_spans, col_spans


def _cell_patch(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _target_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {BACKGROUND, SEPARATOR})
    assert len(colors) == 1, "expected exactly one non-background non-separator payload color"
    return colors[0]


def _tab_side_and_body(cell: np.ndarray, color: int) -> tuple[str, np.ndarray]:
    mask = cell == color
    comps = _components4(mask)
    assert len(comps) == 2, "template cell must contain a body and a separate tab"
    comps.sort(key=len)
    assert len(comps[0]) == 1 and 3 <= len(comps[1]) <= 5, "invalid tab/body sizes"
    tr, tc = comps[0][0]
    body = np.zeros_like(mask, dtype=bool)
    for r, c in comps[1]:
        body[r, c] = True

    side_hits = []
    if tc == 0:
        side_hits.append("left")
    if tc == CELL_SIZE - 1:
        side_hits.append("right")
    if tr == 0:
        side_hits.append("top")
    if tr == CELL_SIZE - 1:
        side_hits.append("bottom")
    assert len(side_hits) == 1, "tab must lie on exactly one cell edge"
    side = side_hits[0]

    if side == "left":
        assert tc + 1 < CELL_SIZE and not mask[tr, tc + 1] and body[tr, tc + 2]
    elif side == "right":
        assert tc - 1 >= 0 and not mask[tr, tc - 1] and body[tr, tc - 2]
    elif side == "top":
        assert tr + 1 < CELL_SIZE and not mask[tr + 1, tc] and body[tr + 2, tc]
    else:
        assert tr - 1 >= 0 and not mask[tr - 1, tc] and body[tr - 2, tc]

    return side, body


def _reconstruct_template(body: np.ndarray, side: str) -> np.ndarray:
    if side in ("left", "right"):
        full = body | np.fliplr(body)
    else:
        full = body | np.flipud(body)
    assert np.any(full != body), "mirror step must matter"
    return full


def _render_template(full_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(full_mask.shape, BACKGROUND, dtype=int)
    patch[full_mask] = color
    return patch


def _stamp_into_cell(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], patch: np.ndarray) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a tab-indicated half-template by mirroring it inside its lattice cell, then stamp the restored template into every matching marker cell.

    Input structure
    - Grey separator lines partition the canvas into a regular lattice of 5x5 cells on a black background.
    - Exactly one cell contains a same-color broken template: a 4-connected body plus one isolated same-color tab pixel on a cell edge with a one-pixel gap to the body.
    - Several other cells contain exactly one center marker pixel of that same color, while all remaining cells are empty.

    Rule
    - Find the unique cell with more than one payload pixel and split its payload into the body component and the one-pixel tab.
    - The tab edge chooses the mirror direction: left/right tab means flip the body left-right inside the 5x5 cell, top/bottom tab means flip it up-down; remove the tab and merge the body with its mirror to obtain the full template.
    - Replace the template cell and every single-center marker cell with that restored 5x5 template, while leaving black background and grey separator lines unchanged.

    Key insight
    - The tab is not part of the final picture: it only tells which half of the hidden symmetric template is missing, and the recovered template then propagates to all same-color markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_spans, col_spans = _parse_lattice(g)
    color = _target_color(g)
    center = CELL_SIZE // 2

    template_cell: tuple[int, int] | None = None
    marker_cells: list[tuple[int, int]] = []
    for rr, row_span in enumerate(row_spans):
        for cc, col_span in enumerate(col_spans):
            patch = _cell_patch(g, row_span, col_span)
            payload = int(np.sum(patch == color))
            other = set(map(int, np.unique(patch))) - {BACKGROUND, SEPARATOR, color}
            assert not other, "unexpected extra colors inside lattice cells"
            if payload > 1:
                assert template_cell is None, "expected exactly one template cell"
                template_cell = (rr, cc)
            elif payload == 1 and patch[center, center] == color:
                marker_cells.append((rr, cc))
            else:
                assert payload == 0, "non-template payload cells must be single centered markers"

    assert template_cell is not None, "missing template cell"
    assert 2 <= len(marker_cells) <= 4, "expected 2-4 marker cells"

    template_patch = _cell_patch(g, row_spans[template_cell[0]], col_spans[template_cell[1]])
    side, body = _tab_side_and_body(template_patch, color)
    full_mask = _reconstruct_template(body, side)
    rendered = _render_template(full_mask, color)

    out = g.copy()
    for rr, cc in [template_cell, *marker_cells]:
        _stamp_into_cell(out, row_spans[rr], col_spans[cc], rendered)
    return out


def _cell_origin(cell_r: int, cell_c: int) -> tuple[int, int]:
    return cell_r * (CELL_SIZE + 1), cell_c * (CELL_SIZE + 1)


def _body_region(side: str) -> list[tuple[int, int]]:
    if side == "right":
        return [(r, c) for r in range(CELL_SIZE) for c in range(3)]
    if side == "left":
        return [(r, c) for r in range(CELL_SIZE) for c in range(2, CELL_SIZE)]
    if side == "bottom":
        return [(r, c) for r in range(3) for c in range(CELL_SIZE)]
    return [(r, c) for r in range(2, CELL_SIZE) for c in range(CELL_SIZE)]


def _anchor_for_side(side: str, tab_line: int) -> tuple[int, int]:
    if side in ("left", "right"):
        return tab_line, 2
    return 2, tab_line


def _tab_for_side(side: str, tab_line: int) -> tuple[int, int]:
    if side == "left":
        return tab_line, 0
    if side == "right":
        return tab_line, CELL_SIZE - 1
    if side == "top":
        return 0, tab_line
    return CELL_SIZE - 1, tab_line


def _random_connected_body(side: str) -> tuple[np.ndarray, tuple[int, int]]:
    allowed = set(_body_region(side))
    tab_line = random.randint(1, CELL_SIZE - 2)
    anchor = _anchor_for_side(side, tab_line)
    target_size = random.randint(3, 5)

    for _ in range(120):
        cells = {anchor}
        while len(cells) < target_size:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in allowed and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_size:
            continue
        body = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        for r, c in cells:
            body[r, c] = True
        if side in ("left", "right") and np.all(np.argwhere(body)[:, 1] == 2):
            continue
        if side in ("top", "bottom") and np.all(np.argwhere(body)[:, 0] == 2):
            continue
        return body, _tab_for_side(side, tab_line)
    raise RuntimeError("failed to build a non-degenerate body")


def _build_expected_output(
    grid: np.ndarray,
    full_mask: np.ndarray,
    color: int,
    template_cell: tuple[int, int],
    marker_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    patch = _render_template(full_mask, color)
    for rr, cc in [template_cell, *marker_cells]:
        r0, c0 = _cell_origin(rr, cc)
        out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = patch
    return out


def _semantic_checks(
    grid: np.ndarray,
    body: np.ndarray,
    side: str,
    full_mask: np.ndarray,
    color: int,
    template_cell: tuple[int, int],
    marker_cells: list[tuple[int, int]],
) -> bool:
    expected = _build_expected_output(grid, full_mask, color, template_cell, marker_cells)
    if not np.array_equal(main(grid), expected):
        return False
    if side in ("left", "right"):
        wrong = body | np.flipud(body)
    else:
        wrong = body | np.fliplr(body)
    if np.array_equal(wrong, full_mask):
        return False
    if np.array_equal(body, full_mask):
        return False

    row_spans, col_spans = _parse_lattice(grid)
    template_like = 0
    center = CELL_SIZE // 2
    for row_span in row_spans:
        for col_span in col_spans:
            patch = _cell_patch(grid, row_span, col_span)
            payload = int(np.sum(patch == color))
            if payload > 1:
                template_like += 1
            elif payload == 1 and patch[center, center] != color:
                return False
    return template_like == 1


def _generate_case() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    color = random.choice(PAYLOAD_COLORS)
    side = random.choice(SIDES)
    marker_count = random.randint(2, 4)

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_cells)
    template_cell = all_cells[0]
    marker_cells = sorted(all_cells[1 : 1 + marker_count])
    if len(marker_cells) != marker_count:
        return None

    body, tab = _random_connected_body(side)
    full_mask = _reconstruct_template(body, side)

    height = rows * CELL_SIZE + (rows - 1)
    width = cols * CELL_SIZE + (cols - 1)
    grid = np.full((height, width), SEPARATOR, dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            r0, c0 = _cell_origin(rr, cc)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = BACKGROUND

    tr0, tc0 = _cell_origin(*template_cell)
    template_patch = np.full((CELL_SIZE, CELL_SIZE), BACKGROUND, dtype=int)
    template_patch[body] = color
    template_patch[tab] = color
    grid[tr0 : tr0 + CELL_SIZE, tc0 : tc0 + CELL_SIZE] = template_patch

    center = CELL_SIZE // 2
    for rr, cc in marker_cells:
        r0, c0 = _cell_origin(rr, cc)
        grid[r0 + center, c0 + center] = color

    if not _semantic_checks(grid, body, side, full_mask, color, template_cell, marker_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tab-mirror stamp puzzle")
