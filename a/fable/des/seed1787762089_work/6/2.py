import random
from collections import Counter

import numpy as np


BLACK = 0
ALL_COLORS = tuple(range(10))
PANEL_BACKGROUNDS = (1, 2, 3, 4, 7, 8)
GEN_RETRY_LIMIT = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, cell)) for cell in coords}
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(shifted)


def _split_panels(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    separator_cols = [c for c in range(w) if np.all(grid[:, c] == BLACK)]
    spans: list[tuple[int, int]] = []
    start = 0
    for c in separator_cols + [w]:
        if c > start:
            spans.append((start, c))
        start = c + 1
    assert spans, "expected at least one panel"
    assert separator_cols == [spans[i][1] for i in range(len(spans) - 1)], "separators must be single black columns"
    panel_widths = {c1 - c0 for c0, c1 in spans}
    assert len(panel_widths) == 1, "panels must have equal width"
    panel_size = panel_widths.pop()
    assert h == panel_size and panel_size in (6, 7), "expected 6x6 or 7x7 square panels"
    assert len(spans) in (3, 4), "expected 3 or 4 panels"
    return spans


def _extract_panel(panel: np.ndarray) -> dict[str, object]:
    bg = _most_common_color(panel)
    motif = np.argwhere(panel != bg)
    assert 3 <= len(motif) <= 6, "motif must contain 3-6 cells"
    assert _is_connected(motif), "motif must be 4-connected"
    assert np.unique(panel[motif[:, 0], motif[:, 1]]).size == 1, "motif must be monochrome"
    assert motif[:, 0].min() > 0 and motif[:, 0].max() < panel.shape[0] - 1
    assert motif[:, 1].min() > 0 and motif[:, 1].max() < panel.shape[1] - 1
    return {"bg": int(bg), "motif": motif.astype(int)}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    spans = _split_panels(g)
    panels = [_extract_panel(g[:, c0:c1]) for c0, c1 in spans]

    backgrounds = [panel["bg"] for panel in panels]
    assert len(set(backgrounds)) == len(backgrounds), "panel backgrounds must be unique"

    out = g.copy()
    for (c0, c1), panel in zip(spans, panels):
        out[:, c0:c1] = int(panel["bg"])

    for i, (c0, c1) in enumerate(spans):
        shape = panels[(i + 1) % len(panels)]["motif"]
        color = int(panels[(i - 1) % len(panels)]["bg"])
        shape = np.asarray(shape, dtype=int)
        out[shape[:, 0], c0 + shape[:, 1]] = color

    return out


def _random_polyomino(max_h: int, max_w: int, area: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        if area > h * w:
            continue
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords = np.array(_normalize(coords), dtype=int)
        if coords[:, 0].max() + 1 > max_h or coords[:, 1].max() + 1 > max_w:
            continue
        return coords
    return None


def _sample_unique_shapes(panel_size: int, count: int) -> list[np.ndarray] | None:
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[np.ndarray] = []
    max_dim = panel_size - 2
    for _ in range(500):
        if len(shapes) == count:
            return shapes
        area = random.randint(3, 6)
        shape = _random_polyomino(max_dim, max_dim, area)
        if shape is None:
            continue
        key = _normalize(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    return None


def _place_shape(panel_size: int, shape: np.ndarray) -> np.ndarray:
    h = int(shape[:, 0].max()) + 1
    w = int(shape[:, 1].max()) + 1
    top = random.randint(1, panel_size - h - 1)
    left = random.randint(1, panel_size - w - 1)
    return shape + np.array([top, left], dtype=int)


def _build_grid(
    panel_size: int,
    panel_colors: list[int],
    motif_colors: list[int],
    motifs: list[np.ndarray],
) -> np.ndarray:
    panel_count = len(panel_colors)
    width = panel_count * panel_size + (panel_count - 1)
    grid = np.full((panel_size, width), BLACK, dtype=int)
    for i, (bg, motif_color, motif) in enumerate(zip(panel_colors, motif_colors, motifs)):
        c0 = i * (panel_size + 1)
        c1 = c0 + panel_size
        grid[:, c0:c1] = bg
        grid[motif[:, 0], c0 + motif[:, 1]] = motif_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice((3, 4))
    panel_size = random.choice((6, 7))

    panel_colors = random.sample(PANEL_BACKGROUNDS, panel_count)
    motif_color_pool = [color for color in ALL_COLORS if color not in panel_colors]
    motif_colors = random.sample(motif_color_pool, panel_count)

    base_shapes = _sample_unique_shapes(panel_size, panel_count)
    if base_shapes is None:
        return None

    motifs = [_place_shape(panel_size, shape) for shape in base_shapes]
    if len({_normalize(motif) for motif in motifs}) != panel_count:
        return None

    grid = _build_grid(panel_size, panel_colors, motif_colors, motifs)

    spans = _split_panels(grid)
    parsed = [_extract_panel(grid[:, c0:c1]) for c0, c1 in spans]
    if [item["bg"] for item in parsed] != panel_colors:
        return None
    parsed_shapes = [_normalize(np.asarray(item["motif"], dtype=int)) for item in parsed]
    expected_shapes = [_normalize(motif) for motif in motifs]
    if parsed_shapes != expected_shapes:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
