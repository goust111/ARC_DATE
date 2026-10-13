from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

_GEN_MAX_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append({"coords": coords, "bbox": _bbox(coords), "area": len(cells)})
    return components


def _is_frame(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * h + 2 * w - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == GREY):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == GREY):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == GREY):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == GREY):
        return False
    return not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY)


def _extract_panels(grid: np.ndarray) -> list[dict]:
    panels = [comp for comp in _find_color_components(grid, GREY) if _is_frame(grid, comp)]
    assert len(panels) == 3, "expected exactly three grey frames"
    panels.sort(key=lambda comp: comp["bbox"][1])
    dims = {(comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1) for comp in panels}
    assert len(dims) == 1, "expected equal-sized panels"
    top_rows = {comp["bbox"][0] for comp in panels}
    bottom_rows = {comp["bbox"][2] for comp in panels}
    assert len(top_rows) == 1 and len(bottom_rows) == 1, "expected panels aligned by rows"
    return panels


def _extract_sprite(panel: dict, grid: np.ndarray) -> dict:
    r0, c0, r1, c1 = panel["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior != BLACK)
    assert coords.size > 0, "expected a sprite inside each panel"
    colors = {int(interior[r, c]) for r, c in coords}
    assert len(colors) == 1, "expected a monochrome sprite"
    sr0, sc0, sr1, sc1 = _bbox(coords)
    rel_coords = sorted((int(r - sr0), int(c - sc0)) for r, c in coords)
    return {
        "top": int(sr0),
        "left": int(sc0),
        "shape_h": int(sr1 - sr0 + 1),
        "shape_w": int(sc1 - sc0 + 1),
        "rel_coords": rel_coords,
        "color": colors.pop(),
        "interior_h": interior.shape[0],
        "interior_w": interior.shape[1],
    }


def _extract_legend(grid: np.ndarray, panels: list[dict]) -> tuple[set[int], dict[int, int]]:
    r0, _, r1, _ = panels[0]["bbox"]
    last_c1 = panels[-1]["bbox"][3]
    region = grid[r0 + 1 : r1, last_c1 + 1 :]
    assert region.size > 0, "expected legend region to the right of the panels"

    obstacle_rows: set[int] = set()
    row_to_color: dict[int, int] = {}
    for local_r in range(region.shape[0]):
        row = region[local_r]
        if np.any(row == PINK):
            obstacle_rows.add(local_r)
        band_colors = {int(v) for v in row if int(v) in (BLUE, GREEN, YELLOW)}
        assert len(band_colors) == 1, "expected exactly one legend color on each interior row"
        row_to_color[local_r] = band_colors.pop()

    assert obstacle_rows, "expected at least one obstacle row"
    assert len(row_to_color) == region.shape[0], "legend color must cover every interior row"
    return obstacle_rows, row_to_color


def _draw_panel(panel_h: int, panel_w: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY
    return panel


def _stamp_sprite(panel: np.ndarray, top: int, left: int, rel_coords: list[tuple[int, int]], color: int) -> None:
    for dr, dc in rel_coords:
        panel[1 + top + dr, 1 + left + dc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the sprite to the next frame, bounce it off periodic obstacle rows if needed, and recolor it from the legend band at the final row.

    Input structure:
    - The input has three equal grey framed panels on a black background, each containing the same interior sprite at a fixed local column.
    - The sprite's row changes by a constant amount from panel to panel.
    - A right-side legend region marks periodic obstacle rows with pink horizontal segments and partitions the panel interior rows into blue, green, and yellow color bands.

    Rule:
    - Read the sprite's top-row positions from the three panels and infer the constant vertical step.
    - Extrapolate one more frame by adding that step to the last sprite row.
    - If the extrapolated row lands on a pink obstacle row, reverse the vertical step and recompute the next row from the last panel instead.
    - Output a single standalone grey frame with the sprite at the final row, unchanged shape and column, recolored to the legend band color of that final row.

    Key insight:
    - The right-side pink rows are not decoration; they are collision rows that can invalidate the naive linear extrapolation and force a reflected step before the final recoloring lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    panels = _extract_panels(g)
    sprites = [_extract_sprite(panel, g) for panel in panels]
    template = sprites[0]
    for sprite in sprites[1:]:
        assert sprite["left"] == template["left"], "expected a fixed sprite column"
        assert sprite["shape_h"] == template["shape_h"] and sprite["shape_w"] == template["shape_w"]
        assert sprite["rel_coords"] == template["rel_coords"], "expected identical sprite shape in every panel"

    obstacle_rows, row_to_color = _extract_legend(g, panels)
    tops = [sprite["top"] for sprite in sprites]
    delta1 = tops[1] - tops[0]
    delta2 = tops[2] - tops[1]
    assert delta1 == delta2 and delta1 != 0, "expected constant non-zero vertical motion"
    predicted_top = tops[-1] + delta1
    final_top = tops[-1] - delta1 if predicted_top in obstacle_rows else predicted_top

    interior_h = template["interior_h"]
    interior_w = template["interior_w"]
    assert 0 <= final_top <= interior_h - template["shape_h"], "final row leaves the frame interior"
    final_color = row_to_color[final_top]

    panel_h = panels[0]["bbox"][2] - panels[0]["bbox"][0] + 1
    panel_w = panels[0]["bbox"][3] - panels[0]["bbox"][1] + 1
    out = _draw_panel(panel_h, panel_w)
    _stamp_sprite(out, final_top, template["left"], template["rel_coords"], final_color)
    return out


def _sample_sprite() -> tuple[np.ndarray, list[tuple[int, int]]]:
    length = random.choice((2, 3))
    sprite = np.full((1, length), BLACK, dtype=int)
    sprite[0, :] = RED
    rel_coords = [(0, c) for c in range(length)]
    return sprite, rel_coords


def _partition_rows(total: int) -> tuple[int, int, int]:
    cuts = []
    for a in range(2, total - 3):
        for b in range(2, total - a - 1):
            c = total - a - b
            if c >= 2:
                cuts.append((a, b, c))
    return random.choice(cuts)


def _choose_obstacle_rows(interior_h: int) -> list[int]:
    candidates: list[list[int]] = []
    step = 4
    for offset in range(1, step):
        rows = list(range(offset, interior_h, step))
        if 2 <= len(rows) <= 4:
            candidates.append(rows)
    return random.choice(candidates)


def _row_to_color_map(lengths: tuple[int, int, int]) -> dict[int, int]:
    row_to_color: dict[int, int] = {}
    start = 0
    for color, size in zip((BLUE, GREEN, YELLOW), lengths):
        for row in range(start, start + size):
            row_to_color[row] = color
        start += size
    return row_to_color


def _enumerate_motion_configs(
    top_limit: int,
    obstacle_rows: set[int],
    row_to_color: dict[int, int],
) -> list[dict]:
    configs: list[dict] = []
    for delta in (-3, -2, -1, 1, 2, 3):
        for final_top in range(top_limit + 1):
            t1 = final_top - 3 * delta
            t2 = final_top - 2 * delta
            t3 = final_top - delta
            free_rows = (t1, t2, t3, final_top)
            if all(0 <= row <= top_limit for row in free_rows) and obstacle_rows.isdisjoint(free_rows):
                configs.append(
                    {
                        "branch": "free",
                        "delta": delta,
                        "rows": (t1, t2, t3),
                        "final_top": final_top,
                        "final_color": row_to_color[final_top],
                    }
                )

            c1 = final_top - delta
            c2 = final_top
            c3 = final_top + delta
            predicted = final_top + 2 * delta
            visible_rows = (c1, c2, c3, final_top)
            if (
                all(0 <= row <= top_limit for row in (c1, c2, c3, predicted))
                and obstacle_rows.isdisjoint(visible_rows)
                and predicted in obstacle_rows
            ):
                configs.append(
                    {
                        "branch": "collision",
                        "delta": delta,
                        "rows": (c1, c2, c3),
                        "final_top": final_top,
                        "final_color": row_to_color[final_top],
                    }
                )
    return configs


def _build_input_grid(
    panel_h: int,
    panel_w: int,
    sprite_left: int,
    rel_coords: list[tuple[int, int]],
    input_rows: tuple[int, int, int],
    obstacle_rows: list[int],
    row_to_color: dict[int, int],
) -> np.ndarray:
    panel_gap = 1
    legend_gap = 2
    legend_w = 5
    total_w = 3 * panel_w + 2 * panel_gap + legend_gap + legend_w
    grid = np.full((panel_h, total_w), BLACK, dtype=int)

    for panel_idx, top in enumerate(input_rows):
        c0 = panel_idx * (panel_w + panel_gap)
        panel = _draw_panel(panel_h, panel_w)
        _stamp_sprite(panel, top, sprite_left, rel_coords, RED)
        grid[:, c0 : c0 + panel_w] = panel

    legend_c0 = 3 * panel_w + 2 * panel_gap + legend_gap
    for local_r, color in row_to_color.items():
        rr = 1 + local_r
        if local_r in obstacle_rows:
            grid[rr, legend_c0 : legend_c0 + 2] = PINK
        grid[rr, legend_c0 + 3 : legend_c0 + 5] = color
    return grid


def _attempt_generate(preferred_branch: str, preferred_color: int) -> np.ndarray | None:
    panel_h = random.randint(9, 11)
    panel_w = random.randint(6, 7)
    interior_h = panel_h - 2
    interior_w = panel_w - 2

    sprite, rel_coords = _sample_sprite()
    sprite_h, sprite_w = sprite.shape
    if sprite_h > interior_h or sprite_w > interior_w:
        return None

    sprite_left = random.randint(0, interior_w - sprite_w)
    obstacle_rows = _choose_obstacle_rows(interior_h)
    row_to_color = _row_to_color_map(_partition_rows(interior_h))
    configs = _enumerate_motion_configs(interior_h - sprite_h, set(obstacle_rows), row_to_color)
    if not configs:
        return None

    configs = [cfg for cfg in configs if cfg["branch"] == preferred_branch]
    if not configs:
        return None
    color_matched = [cfg for cfg in configs if cfg["final_color"] == preferred_color]
    chosen = random.choice(color_matched if color_matched else configs)

    inp = _build_input_grid(
        panel_h,
        panel_w,
        sprite_left,
        rel_coords,
        chosen["rows"],
        obstacle_rows,
        row_to_color,
    )

    expected = _draw_panel(panel_h, panel_w)
    _stamp_sprite(expected, chosen["final_top"], sprite_left, rel_coords, chosen["final_color"])
    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    preferred_branch = "collision" if _GEN_CALL_COUNT % 2 == 0 else "free"
    preferred_color = (BLUE, GREEN, YELLOW)[_GEN_CALL_COUNT % 3]
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _attempt_generate(preferred_branch, preferred_color)
        if inp is not None:
            _GEN_CALL_COUNT += 1
            return inp
    raise RuntimeError("failed to generate a valid bouncing-extrapolation input")
