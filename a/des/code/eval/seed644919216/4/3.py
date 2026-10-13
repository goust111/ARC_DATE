import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_safes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    safes: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if not (5 <= height <= 8 and 5 <= width <= 8):
                continue
            if len(comp) != 2 * (height + width) - 4:
                continue
            if not np.all(g[r0, c0 : c1 + 1] == color):
                continue
            if not np.all(g[r1, c0 : c1 + 1] == color):
                continue
            if not np.all(g[r0 : r1 + 1, c0] == color):
                continue
            if not np.all(g[r0 : r1 + 1, c1] == color):
                continue
            if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
                continue
            safes.append(
                {
                    "border": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner_h": height - 2,
                    "inner_w": width - 2,
                }
            )
    border_colors = [safe["border"] for safe in safes]
    assert len(border_colors) == len(set(border_colors)), "safe border colors must be unique"
    return safes


def _find_panels(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    panels: list[dict] = []
    seen_segments: set[tuple[int, int, int]] = set()

    for r, c in map(tuple, np.argwhere(g == GREY)):
        r = int(r)
        c = int(c)
        if r == 0:
            continue
        if int(g[r - 1, c]) in (BLACK, GREY):
            continue

        c0 = c
        while c0 - 1 >= 0 and int(g[r - 1, c0 - 1]) not in (BLACK, GREY):
            c0 -= 1
        c1 = c
        while c1 + 1 < w and int(g[r - 1, c1 + 1]) not in (BLACK, GREY):
            c1 += 1

        key = (r - 1, c0, c1)
        if key in seen_segments:
            continue
        seen_segments.add(key)

        colors = [int(v) for v in g[r - 1, c0 : c1 + 1].tolist()]
        if not (3 <= len(colors) <= 5):
            continue
        if any(color in (BLACK, GREY) for color in colors):
            continue
        if len(colors) != len(set(colors)):
            continue
        if int(np.count_nonzero(g[r, c0 : c1 + 1] == GREY)) != 1:
            continue

        panels.append(
            {
                "row": r - 1,
                "c0": c0,
                "c1": c1,
                "colors": colors,
            }
        )

    return panels


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    safes = _find_safes(g)
    panels = _find_panels(g)
    assert safes, "expected at least one safe"
    assert panels, "expected at least one panel"

    safe_by_color = {safe["border"]: safe for safe in safes}
    out = g.copy()
    matched_colors: set[int] = set()

    for panel in panels:
        overlap = [color for color in panel["colors"] if color in safe_by_color]
        assert len(overlap) == 1, "each panel must share exactly one color with the safes"
        border = overlap[0]
        assert border not in matched_colors, "each safe should match exactly one panel"
        matched_colors.add(border)

        safe = safe_by_color[border]
        key_idx = panel["colors"].index(border)
        left_colors = panel["colors"][:key_idx]
        right_colors = panel["colors"][key_idx + 1 :]
        assert len(left_colors) + len(right_colors) <= safe["inner_w"], "side bands must fit the safe interior"

        r0, c0, r1, c1 = safe["bbox"]
        for offset, color in enumerate(left_colors):
            out[r0 + 1 : r1, c0 + 1 + offset] = color
        for offset, color in enumerate(right_colors):
            out[r0 + 1 : r1, c1 - 1 - offset] = color

    assert len(matched_colors) == len(safes), "every safe should be matched by a panel"
    return out


def _partition_items(items: list[dict], row_count: int) -> list[list[dict]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    if row_count == 1:
        return [shuffled]

    if len(shuffled) == 2:
        counts = [1, 1]
    elif len(shuffled) == 3:
        counts = random.choice(([1, 2], [2, 1]))
    else:
        counts = [2, len(shuffled) - 2]

    rows: list[list[dict]] = []
    start = 0
    for count in counts:
        rows.append(shuffled[start : start + count])
        start += count
    return rows


def _prepare_row_layout(items: list[dict], row_count: int, gap_range: tuple[int, int], height_key: str, width_key: str) -> dict:
    rows = _partition_items(items, row_count)
    row_specs = []
    for row_items in rows:
        gaps = [random.randint(*gap_range) for _ in range(max(0, len(row_items) - 1))]
        content_w = sum(int(item[width_key]) for item in row_items) + sum(gaps)
        row_h = max(int(item[height_key]) for item in row_items)
        row_specs.append({"items": row_items, "gaps": gaps, "content_w": content_w, "row_h": row_h})

    vertical_gaps = [random.randint(1, 3) for _ in range(max(0, len(row_specs) - 1))]
    total_h = sum(spec["row_h"] for spec in row_specs) + sum(vertical_gaps)
    max_w = max(spec["content_w"] for spec in row_specs)
    return {"rows": row_specs, "vertical_gaps": vertical_gaps, "total_h": total_h, "max_w": max_w}


def _place_layout(layout: dict, zone_top: int, zone_height: int, grid_w: int, height_key: str, width_key: str, top_key: str, left_key: str) -> None:
    rows = layout["rows"]
    total_h = int(layout["total_h"])
    vertical_gaps = layout["vertical_gaps"]
    assert total_h <= zone_height
    y = zone_top + random.randint(0, zone_height - total_h)

    for row_idx, row_spec in enumerate(rows):
        content_w = int(row_spec["content_w"])
        row_h = int(row_spec["row_h"])
        x = random.randint(1, grid_w - content_w - 1)

        for item_idx, item in enumerate(row_spec["items"]):
            item_h = int(item[height_key])
            item_w = int(item[width_key])
            item[top_key] = y + random.randint(0, row_h - item_h)
            item[left_key] = x
            x += item_w
            if item_idx < len(row_spec["gaps"]):
                x += int(row_spec["gaps"][item_idx])

        y += row_h
        if row_idx < len(vertical_gaps):
            y += int(vertical_gaps[row_idx])


def _paint_safe(grid: np.ndarray, safe: dict) -> None:
    r0 = int(safe["top"])
    c0 = int(safe["left"])
    h = int(safe["outer_h"])
    w = int(safe["outer_w"])
    color = int(safe["border"])

    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    safe["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)


def _paint_panel(grid: np.ndarray, panel: dict) -> None:
    row = int(panel["top"])
    col = int(panel["left"])
    seq = [int(color) for color in panel["colors"]]
    grid[row, col : col + len(seq)] = seq
    grid[row + 1, col + int(panel["support_idx"])] = GREY


def _expected_output(grid: np.ndarray, safes: list[dict], panels: list[dict]) -> np.ndarray:
    out = grid.copy()
    panel_by_border = {int(panel["border"]): panel for panel in panels}
    for safe in safes:
        border = int(safe["border"])
        panel = panel_by_border[border]
        seq = [int(color) for color in panel["colors"]]
        key_idx = seq.index(border)
        r0, c0, r1, c1 = safe["bbox"]
        for offset, color in enumerate(seq[:key_idx]):
            out[r0 + 1 : r1, c0 + 1 + offset] = color
        for offset, color in enumerate(seq[key_idx + 1 :]):
            out[r0 + 1 : r1, c1 - 1 - offset] = color
    return out


def _row_count_for(count: int, allow_single_for_three: bool) -> int:
    if count == 1:
        return 1
    if count == 2:
        return random.choice((1, 2))
    if count == 3 and allow_single_for_three:
        return random.choice((1, 2))
    return 2


def _build_safe_and_panel_specs() -> tuple[list[dict], list[dict]]:
    safe_count = random.choices((1, 2, 3, 4), weights=(1, 4, 4, 2), k=1)[0]
    safe_colors = random.sample(list(DRAW_COLORS), safe_count)
    filler_palette = [color for color in DRAW_COLORS if color not in safe_colors]

    safes: list[dict] = []
    panels: list[dict] = []

    for border in safe_colors:
        full_fill = random.random() < 0.5
        if full_fill:
            panel_len = random.choice((4, 5))
            inner_w = panel_len - 1
        else:
            panel_len = random.choice((3, 4, 5))
            inner_w = random.randint(max(3, panel_len), 6)

        fillers = random.sample(filler_palette, panel_len - 1)
        key_idx = random.randrange(panel_len)
        sequence = fillers[:key_idx] + [border] + fillers[key_idx:]

        safes.append(
            {
                "border": int(border),
                "outer_h": random.randint(5, 8),
                "outer_w": inner_w + 2,
            }
        )
        panels.append(
            {
                "border": int(border),
                "colors": sequence,
                "support_idx": random.randrange(panel_len),
                "panel_h": 2,
                "panel_w": panel_len,
            }
        )

    return safes, panels


def _generate_attempt() -> np.ndarray | None:
    safes, panels = _build_safe_and_panel_specs()
    safe_layout = _prepare_row_layout(
        safes,
        row_count=_row_count_for(len(safes), allow_single_for_three=True),
        gap_range=(2, 4),
        height_key="outer_h",
        width_key="outer_w",
    )
    panel_layout = _prepare_row_layout(
        panels,
        row_count=_row_count_for(len(panels), allow_single_for_three=True),
        gap_range=(3, 6),
        height_key="panel_h",
        width_key="panel_w",
    )

    half_h = max(int(safe_layout["total_h"]) + 2, int(panel_layout["total_h"]) + 2) + random.randint(0, 3)
    grid_h = 2 * half_h
    grid_w = max(int(safe_layout["max_w"]), int(panel_layout["max_w"])) + random.randint(4, 10)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    _place_layout(safe_layout, zone_top=0, zone_height=half_h, grid_w=grid_w, height_key="outer_h", width_key="outer_w", top_key="top", left_key="left")
    _place_layout(panel_layout, zone_top=half_h, zone_height=half_h, grid_w=grid_w, height_key="panel_h", width_key="panel_w", top_key="top", left_key="left")

    for safe in safes:
        _paint_safe(grid, safe)
    for panel in panels:
        _paint_panel(grid, panel)

    expected = _expected_output(grid, safes, panels)
    try:
        found_safes = _find_safes(grid)
        found_panels = _find_panels(grid)
        if len(found_safes) != len(safes) or len(found_panels) != len(panels):
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
