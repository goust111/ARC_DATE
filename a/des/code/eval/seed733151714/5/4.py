from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _separator_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == YELLOW)]
    assert len(rows) == 1, "expected exactly one full-width yellow separator row"
    return int(rows[0])


def _find_towers(g: np.ndarray, sep_row: int) -> list[dict[str, int]]:
    towers: list[dict[str, int]] = []
    for c in range(g.shape[1]):
        r = 0
        while r < sep_row:
            if g[r, c] == BLACK:
                r += 1
                continue
            start = r
            while r < sep_row and g[r, c] != BLACK:
                r += 1
            end = r - 1
            vals = g[start : end + 1, c]
            ok = (
                5 <= len(vals) <= 7
                and np.all(vals[1:-1] == GREY)
                and int(vals[0]) not in (BLACK, YELLOW, GREY)
                and int(vals[-1]) not in (BLACK, YELLOW, GREY)
            )
            assert ok, "unexpected upper-half object: each tower must be color-grey-grey...-color"
            towers.append(
                {
                    "col": int(c),
                    "top": int(vals[0]),
                    "bottom": int(vals[-1]),
                }
            )
    assert towers, "expected at least one signal tower"
    return towers


def _find_receivers(g: np.ndarray, sep_row: int) -> list[dict[str, object]]:
    lower_mask = np.zeros_like(g, dtype=bool)
    lower_mask[sep_row + 1 :, :] = g[sep_row + 1 :, :] != BLACK
    receivers: list[dict[str, object]] = []
    for coords in _components(lower_mask):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = int(g[coords[0]])
        solid = all(int(g[r, c]) == color for r, c in coords)
        assert solid, "receiver must be monochromatic"
        assert h in (2, 3) and w in (2, 3), "receiver must be a 2x2, 2x3, 3x2, or 3x3 solid block"
        assert len(coords) == h * w, "receiver must be a solid rectangle"
        receivers.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    assert receivers, "expected at least one receiver block"
    return receivers


def _resolve_color(color: int, mapping: dict[int, int]) -> int:
    cur = int(color)
    seen: set[int] = set()
    while cur in mapping and cur not in seen and mapping[cur] != cur:
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify towers above a yellow separator, read each tower as a color mapping,
    and recolor each lower receiver block by repeatedly applying only the mappings
    whose tower columns overlap that receiver's column span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep_row = _separator_row(g)
    towers = _find_towers(g, sep_row)
    receivers = _find_receivers(g, sep_row)

    out = g.copy()
    for receiver in receivers:
        r0, c0, r1, c1 = receiver["bbox"]
        active = [tower for tower in towers if c0 <= tower["col"] <= c1]
        mapping: dict[int, int] = {}
        for tower in active:
            top = int(tower["top"])
            bottom = int(tower["bottom"])
            if top in mapping:
                assert mapping[top] == bottom, "active towers must not disagree on a top color"
            else:
                mapping[top] = bottom
        final_color = _resolve_color(int(receiver["color"]), mapping)
        out[r0 : r1 + 1, c0 : c1 + 1] = final_color
    return out


# --------------- generator ---------------

GRID_W_RANGE = (12, 18)
PRIMARY_CHAIN_LENGTHS = (2, 3)
EXTRA_GROUP_CHOICES = (0, 1, 1, 2)
EXTRA_GROUP_LENGTHS = (1, 2)
GROUP_GAP = 4
OUTER_MARGIN = 2

TOWER_BODY_H_RANGE = (3, 5)
TOWER_BOTTOM_GAP_MIN = 1
SEP_EXTRA_HEADROOM = 4

RECEIVER_H_RANGE = (2, 3)
RECEIVER_V_GAP = 1
LOWER_TOP_PAD = 1
LOWER_BOTTOM_PAD = 1
EXTRA_RECEIVER_RANGE = (0, 2)

MAX_GRID_H = 30
GENERATION_ATTEMPTS = 400


def _rand_partition(total: int, parts: int) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[random.randrange(parts)] += 1
    return vals


def _group_columns(width: int, lengths: list[int]) -> list[list[int]] | None:
    base_gaps = [OUTER_MARGIN] + [GROUP_GAP] * (len(lengths) - 1) + [OUTER_MARGIN]
    used = sum(lengths) + sum(base_gaps)
    if used > width:
        return None
    extra = _rand_partition(width - used, len(base_gaps))
    gaps = [base + add for base, add in zip(base_gaps, extra)]
    groups: list[list[int]] = []
    col = gaps[0]
    for i, length in enumerate(lengths):
        groups.append(list(range(col, col + length)))
        col += length
        if i + 1 < len(lengths):
            col += gaps[i + 1]
    return groups


def _prefix_span(order_cols: list[int], prefix_len: int) -> tuple[int, int]:
    assert 1 <= prefix_len <= len(order_cols)
    if prefix_len == 1:
        edge = order_cols[0]
        if len(order_cols) == 1:
            if random.random() < 0.5:
                return edge - 1, edge
            return edge, edge + 1
        inward = order_cols[1]
        if inward > edge:
            return edge - 1, edge
        return edge, edge + 1
    cols = order_cols[:prefix_len]
    return min(cols), max(cols)


def _free_spans(width: int, tower_cols: set[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for span_w in (2, 3):
        for c0 in range(0, width - span_w + 1):
            if all(col not in tower_cols for col in range(c0, c0 + span_w)):
                spans.append((c0, c0 + span_w - 1))
    return spans


def _sample_group(length: int, cols: list[int]) -> dict[str, object]:
    order_cols = list(cols) if random.random() < 0.5 else list(reversed(cols))
    chain = random.sample(list(COLOR_POOL), length + 1)
    towers = []
    for i, col in enumerate(order_cols):
        towers.append({"col": int(col), "top": int(chain[i]), "bottom": int(chain[i + 1])})
    return {
        "cols": list(cols),
        "order_cols": order_cols,
        "chain": chain,
        "towers": towers,
    }


def _extra_receiver_specs(
    groups: list[dict[str, object]],
    free_spans: list[tuple[int, int]],
    count: int,
) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    used: set[tuple[int, int, int]] = set()
    for _ in range(count * 10):
        if len(specs) >= count:
            break
        if random.random() < 0.7:
            group = random.choice(groups)
            order_cols = list(group["order_cols"])
            chain = list(group["chain"])
            prefix_len = random.randint(1, len(order_cols))
            c0, c1 = _prefix_span(order_cols, prefix_len)
            if random.random() < 0.75:
                start_index = random.randint(0, prefix_len - 1)
                color = int(chain[start_index])
            else:
                color = int(chain[prefix_len])
        else:
            if not free_spans:
                continue
            c0, c1 = random.choice(free_spans)
            color = random.choice(COLOR_POOL)
        key = (int(c0), int(c1), int(color))
        if key in used:
            continue
        used.add(key)
        specs.append({"c0": int(c0), "c1": int(c1), "color": int(color)})
    return specs


def _receiver_rows(count: int) -> tuple[list[tuple[int, int]], int]:
    heights = [random.randint(*RECEIVER_H_RANGE) for _ in range(count)]
    spans: list[tuple[int, int]] = []
    cursor = LOWER_TOP_PAD
    for height in heights:
        spans.append((cursor, cursor + height - 1))
        cursor += height + RECEIVER_V_GAP
    lower_height = cursor - RECEIVER_V_GAP + LOWER_BOTTOM_PAD
    return spans, lower_height


def _render_towers(
    grid: np.ndarray,
    towers: list[dict[str, int]],
    sep_row: int,
) -> None:
    for tower in towers:
        body_h = int(tower["body_h"])
        total_h = body_h + 2
        top_row = random.randint(0, sep_row - total_h - TOWER_BOTTOM_GAP_MIN)
        c = int(tower["col"])
        grid[top_row, c] = int(tower["top"])
        grid[top_row + 1 : top_row + 1 + body_h, c] = GREY
        grid[top_row + 1 + body_h, c] = int(tower["bottom"])


def _render_receivers(
    grid: np.ndarray,
    sep_row: int,
    row_spans: list[tuple[int, int]],
    specs: list[dict[str, int]],
) -> None:
    for (r0_off, r1_off), spec in zip(row_spans, specs):
        r0 = sep_row + 1 + r0_off
        r1 = sep_row + 1 + r1_off
        grid[r0 : r1 + 1, spec["c0"] : spec["c1"] + 1] = int(spec["color"])


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_W_RANGE)
    primary_len = random.choice(PRIMARY_CHAIN_LENGTHS)
    extra_group_count = random.choice(EXTRA_GROUP_CHOICES)
    lengths = [primary_len]
    for _ in range(extra_group_count):
        lengths.append(random.choice(EXTRA_GROUP_LENGTHS))

    group_cols = _group_columns(width, lengths)
    if group_cols is None:
        return None
    groups = [_sample_group(length, cols) for length, cols in zip(lengths, group_cols)]
    primary = groups[0]

    tower_cols = {col for group in group_cols for col in group}
    free = _free_spans(width, tower_cols)
    if not free:
        return None

    start_color = int(primary["chain"][0])
    base_specs = []

    one_c0, one_c1 = _prefix_span(list(primary["order_cols"]), 1)
    base_specs.append({"c0": int(one_c0), "c1": int(one_c1), "color": start_color})

    multi_len = random.randint(2, len(primary["order_cols"]))
    mul_c0, mul_c1 = _prefix_span(list(primary["order_cols"]), multi_len)
    base_specs.append({"c0": int(mul_c0), "c1": int(mul_c1), "color": start_color})

    out_c0, out_c1 = random.choice(free)
    base_specs.append({"c0": int(out_c0), "c1": int(out_c1), "color": start_color})

    extra_count = random.randint(*EXTRA_RECEIVER_RANGE)
    extra_specs = _extra_receiver_specs(groups, free, extra_count)
    receiver_specs = base_specs + extra_specs

    row_spans, lower_height = _receiver_rows(len(receiver_specs))

    all_towers = [dict(tower) for group in groups for tower in group["towers"]]
    for tower in all_towers:
        tower["body_h"] = random.randint(*TOWER_BODY_H_RANGE)
    max_total_h = max(int(tower["body_h"]) + 2 for tower in all_towers)
    sep_row = random.randint(max_total_h + TOWER_BOTTOM_GAP_MIN, max_total_h + SEP_EXTRA_HEADROOM)
    height = sep_row + 1 + lower_height
    if height > MAX_GRID_H:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[sep_row, :] = YELLOW
    _render_towers(grid, all_towers, sep_row)
    _render_receivers(grid, sep_row, row_spans, receiver_specs)

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    one_bbox = (sep_row + 1 + row_spans[0][0], one_c0, sep_row + 1 + row_spans[0][1], one_c1)
    multi_bbox = (sep_row + 1 + row_spans[1][0], mul_c0, sep_row + 1 + row_spans[1][1], mul_c1)
    out_bbox = (sep_row + 1 + row_spans[2][0], out_c0, sep_row + 1 + row_spans[2][1], out_c1)

    one_val = int(out[one_bbox[0], one_bbox[1]])
    multi_val = int(out[multi_bbox[0], multi_bbox[1]])
    outside_val = int(out[out_bbox[0], out_bbox[1]])
    if one_val == start_color or multi_val == start_color or outside_val != start_color:
        return None
    if one_val == multi_val:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
