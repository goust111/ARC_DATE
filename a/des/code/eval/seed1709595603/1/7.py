from __future__ import annotations

import random

import numpy as np


GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) >= 0:
                continue
            stack = [(r, c)]
            labels[r, c] = next_id
            while stack:
                rr, cc = stack.pop()
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) < 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            next_id += 1
    return labels, next_id


def _legend_row(grid: np.ndarray, bg: int) -> int | None:
    for r in range(grid.shape[0]):
        if np.any(grid[r] != bg):
            return int(r)
    return None


def _priority_colors(grid: np.ndarray, legend_row: int, bg: int) -> list[int]:
    order = []
    seen = set()
    for c in range(grid.shape[1]):
        color = int(grid[legend_row, c])
        if color == bg or color in seen:
            continue
        order.append(color)
        seen.add(color)
    return order


def _detect_terminal_color(grid: np.ndarray, legend_row: int, bg: int) -> int | None:
    edge_colors = [int(grid[r, -1]) for r in range(legend_row + 1, grid.shape[0]) if int(grid[r, -1]) != bg]
    if not edge_colors:
        return None
    values, counts = np.unique(np.array(edge_colors, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_grey_color(grid: np.ndarray, legend_row: int, bg: int, terminal_color: int | None) -> int | None:
    if grid.shape[1] < 3:
        return None
    interior = grid[legend_row + 1 :, 1:-1]
    values = [int(v) for v in interior.ravel() if int(v) != bg and (terminal_color is None or int(v) != terminal_color)]
    if not values:
        return None
    uniq, counts = np.unique(np.array(values, dtype=int), return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _adjacent_component_ids(labels: np.ndarray, row: int, col: int, target_color_mask: np.ndarray) -> set[int]:
    h, w = labels.shape
    comps = set()
    for dr, dc in DIRS4:
        nr = row + dr
        nc = col + dc
        if 0 <= nr < h and 0 <= nc < w and bool(target_color_mask[nr, nc]):
            comp_id = int(labels[nr, nc])
            if comp_id >= 0:
                comps.add(comp_id)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each grey pipe component, the highest-priority source eventually wins every
    intersection in that component, so every reachable terminal on that component
    takes that source color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)

    legend_row = _legend_row(g, bg)
    if legend_row is None:
        return out

    priority = _priority_colors(g, legend_row, bg)
    if not priority:
        return out
    rank = {color: idx for idx, color in enumerate(priority)}

    terminal_color = _detect_terminal_color(g, legend_row, bg)
    grey_color = _detect_grey_color(g, legend_row, bg, terminal_color)
    if terminal_color is None or grey_color is None:
        return out

    grey_mask = g == grey_color
    labels, _ = _label_components(grey_mask)

    best_by_component: dict[int, int] = {}
    for r in range(legend_row + 1, g.shape[0]):
        color = int(g[r, 0])
        if color in (bg, grey_color, terminal_color):
            continue
        if color not in rank:
            continue
        for comp_id in _adjacent_component_ids(labels, r, 0, grey_mask):
            prev = best_by_component.get(comp_id)
            if prev is None or rank[color] < rank[prev]:
                best_by_component[comp_id] = color

    for r in range(legend_row + 1, g.shape[0]):
        if int(g[r, -1]) != terminal_color:
            continue
        winner = None
        for comp_id in _adjacent_component_ids(labels, r, g.shape[1] - 1, grey_mask):
            color = best_by_component.get(comp_id)
            if color is None:
                continue
            if winner is None or rank[color] < rank[winner]:
                winner = color
        if winner is not None:
            out[r, -1] = winner

    return out


def _place_blocks(height: int, block_heights: list[int], top_margin: int) -> list[list[int]] | None:
    blocked = np.zeros(height, dtype=bool)
    blocks = []
    for block_h in block_heights:
        choices = []
        for start in range(top_margin, height - block_h + 1):
            lo = max(top_margin, start - 1)
            hi = min(height, start + block_h + 1)
            if not bool(blocked[lo:hi].any()):
                choices.append(start)
        if not choices:
            return None
        start = random.choice(choices)
        blocks.append(list(range(start, start + block_h)))
        lo = max(top_margin, start - 1)
        hi = min(height, start + block_h + 1)
        blocked[lo:hi] = True
    return blocks


def _sample_component_specs(width: int, height: int) -> list[dict] | None:
    mode = random.choice(["active", "dormant", "both"])
    specs = [{"kind": "competitive", "rows_n": random.randint(2, 3)}]
    if mode in ("active", "both"):
        specs.append({"kind": "active", "rows_n": random.randint(1, 2)})
    if mode in ("dormant", "both"):
        specs.append({"kind": "dormant", "rows_n": random.randint(1, 2)})

    blocks = _place_blocks(height, [int(spec["rows_n"]) for spec in specs], top_margin=3)
    if blocks is None:
        return None

    for spec, rows in zip(specs, blocks, strict=False):
        spec["rows"] = rows
        nrows = len(rows)
        if spec["kind"] == "competitive":
            src_count = random.randint(2, nrows)
            term_count = random.randint(1, nrows)
            spec["source_rows"] = sorted(random.sample(rows, src_count))
            spec["terminal_rows"] = sorted(random.sample(rows, term_count))
        elif spec["kind"] == "active":
            spec["source_rows"] = [random.choice(rows)]
            spec["terminal_rows"] = sorted(random.sample(rows, random.randint(1, nrows)))
        else:
            spec["source_rows"] = []
            spec["terminal_rows"] = sorted(random.sample(rows, random.randint(1, nrows)))

        if nrows == 1:
            spec["connector_cols"] = []
        else:
            max_connectors = min(2, width - 6)
            connector_count = random.randint(1, max_connectors)
            cols = list(range(3, width - 3))
            spec["connector_cols"] = sorted(random.sample(cols, connector_count))

    return specs


def _draw_component(grid: np.ndarray, rows: list[int], connector_cols: list[int]) -> None:
    min_row = min(rows)
    max_row = max(rows)
    for row in rows:
        grid[row, 1:-1] = GREY
    for col in connector_cols:
        grid[min_row : max_row + 1, col] = GREY


def _place_legend(grid: np.ndarray, colors: list[int], bg: int) -> bool:
    slots = list(range(1, grid.shape[1] - 1, 2))
    if len(slots) < len(colors):
        return False
    cols = sorted(random.sample(slots, len(colors)))
    grid[0, :] = bg
    for col, color in zip(cols, colors, strict=False):
        grid[0, col] = int(color)
    return True


def _terminals_after(grid: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    changed = 0
    unchanged = 0
    for r in range(grid.shape[0]):
        if int(grid[r, -1]) != MAROON:
            continue
        if int(out[r, -1]) == MAROON:
            unchanged += 1
        else:
            changed += 1
    return changed, unchanged


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(16, 26)
    height = random.randint(12, 20)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8])

    specs = _sample_component_specs(width, height)
    if specs is None:
        return None

    total_sources = sum(len(spec["source_rows"]) for spec in specs)
    source_pool = [c for c in range(10) if c not in (bg, GREY, MAROON)]
    if total_sources < 2 or total_sources > len(source_pool):
        return None

    source_colors = random.sample(source_pool, total_sources)
    random.shuffle(source_colors)
    legend_colors = source_colors.copy()
    random.shuffle(legend_colors)

    grid = np.full((height, width), bg, dtype=int)
    grid[1:3, :] = bg

    color_iter = iter(source_colors)
    has_dormant = False
    for spec in specs:
        _draw_component(grid, spec["rows"], spec["connector_cols"])
        for row in spec["source_rows"]:
            grid[row, 0] = int(next(color_iter))
        for row in spec["terminal_rows"]:
            grid[row, -1] = MAROON
        if not spec["source_rows"] and spec["terminal_rows"]:
            has_dormant = True

    if not _place_legend(grid, legend_colors, bg):
        return None

    if _most_common_color(grid) != bg:
        return None

    out = main(grid)
    changed, unchanged = _terminals_after(grid, out)
    if changed == 0:
        return None
    if has_dormant and unchanged == 0:
        return None

    right_edge_colors = {int(grid[r, -1]) for r in range(3, height) if int(grid[r, -1]) != bg}
    if right_edge_colors != {MAROON}:
        return None

    left_edge_colors = [int(grid[r, 0]) for r in range(3, height) if int(grid[r, 0]) != bg]
    if len(left_edge_colors) != len(set(left_edge_colors)):
        return None

    interior_non_bg = {int(v) for v in grid[3:, 1:-1].ravel() if int(v) != bg}
    if interior_non_bg != {GREY}:
        return None

    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-priority sample")
