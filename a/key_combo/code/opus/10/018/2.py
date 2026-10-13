from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
START_COLOR = 1
WALL_COLOR = 2
COLOR_POOL = [3, 4, 5, 6, 7, 8, 9]

GEN_HEIGHT_RANGE = (15, 19)
GEN_WIDTH_RANGE = (15, 19)
GEN_CYCLE_LEN_RANGE = (2, 4)
GEN_MAX_ATTEMPTS = 200
GEN_TEMPLATES = [
    {
        "height": 16,
        "width": 19,
        "start": (0, 0),
        "boxes": [
            {"top": 2, "left": 2, "height": 3, "width": 3, "traversed": True},
            {"top": 5, "left": 7, "height": 3, "width": 3, "traversed": True},
            {"top": 4, "left": 12, "height": 3, "width": 3, "traversed": True},
            {"top": 11, "left": 2, "height": 3, "width": 3, "traversed": False},
        ],
    },
    {
        "height": 16,
        "width": 19,
        "start": (0, 18),
        "boxes": [
            {"top": 2, "left": 6, "height": 3, "width": 3, "traversed": True},
            {"top": 7, "left": 2, "height": 3, "width": 3, "traversed": True},
            {"top": 2, "left": 12, "height": 3, "width": 3, "traversed": True},
            {"top": 8, "left": 11, "height": 3, "width": 3, "traversed": False},
        ],
    },
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _start_and_cycle(grid: np.ndarray) -> tuple[tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    start_coords = np.argwhere(g == START_COLOR)
    assert start_coords.shape == (1, 2), "expected exactly one blue start pixel"
    start = tuple(map(int, start_coords[0]))
    assert start[0] == 0 and start[1] in (0, g.shape[1] - 1), "start must be at the upper-left or upper-right corner"

    cycle = [(r, int(g[r, 0])) for r in range(g.shape[0]) if int(g[r, 0]) not in (BACKGROUND, START_COLOR, WALL_COLOR)]
    assert 2 <= len(cycle) <= 4, "expected 2-4 legend colors in the leftmost column"
    assert len({color for _r, color in cycle}) == len(cycle), "legend colors must be distinct"
    return start, [color for _r, color in sorted(cycle)]


def _detect_u_containers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    containers: list[dict[str, int]] = []
    for comp in _find_components(g == WALL_COLOR):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        height = bottom - top + 1
        width = right - left + 1
        local = np.zeros((height, width), dtype=bool)
        for r, c in comp:
            local[r - top, c - left] = True

        template = np.zeros_like(local)
        template[:, 0] = True
        template[:, -1] = True
        template[-1, :] = True

        assert np.array_equal(local, template), "red component must be a clean U-shaped container"
        assert np.all(g[top:bottom, left + 1 : right] == BACKGROUND), "container interior must be hollow"

        containers.append(
            {
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "height": int(height),
                "width": int(width),
            }
        )
    return sorted(containers, key=lambda box: (box["top"], box["left"]))


def _trace_ray(height: int, width: int, start: tuple[int, int], cycle_len: int) -> list[dict[str, int]]:
    r, c = start
    dr, dc = 1, 1
    bounce_count = 0
    color_index = 0
    seen: set[tuple[int, int, int, int, int]] = set()
    records: list[dict[str, int]] = []

    while (r, c, dr, dc, color_index) not in seen:
        seen.add((r, c, dr, dc, color_index))
        records.append(
            {
                "r": int(r),
                "c": int(c),
                "bounce_count": int(bounce_count),
                "color_index": int(color_index),
            }
        )

        nr, nc = r + dr, c + dc
        bounced = False
        if nr < 0 or nr >= height:
            dr = -dr
            nr = r + dr
            bounced = True
        if nc < 0 or nc >= width:
            dc = -dc
            nc = c + dc
            bounced = True
        if bounced:
            bounce_count += 1
            color_index = bounce_count % cycle_len
        r, c = nr, nc

    return records


def _fill_box(out: np.ndarray, box: dict[str, int], color: int) -> None:
    out[box["top"] : box["bottom"], box["left"] + 1 : box["right"]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each first-crossed U-shaped container according to the diagonal ray's bounce-driven color cycle.

    Input structure:
    - The grid has a black background, one blue start pixel in the upper-left or upper-right corner, and 2-4 non-red legend pixels in the leftmost column.
    - Several disjoint red U-shaped containers are scattered in the main field; each container has two vertical side walls, a solid bottom edge, and a hollow rectangular interior.
    - The legend pixels read top-to-bottom define a repeating fill-color cycle, while the red containers and the blue start pixel define the ray-tracing geometry.

    Rule:
    - Remove the blue start pixel from the output and trace a diagonal ray that begins with step (+1, +1).
    - Whenever the next diagonal step would leave the grid, reflect the needed direction(s); a corner hit reverses both directions but still counts as a single bounce event.
    - The current legend index starts at 0 and advances by 1 modulo the legend length after each bounce event.
    - At every visited ray cell, if its row lies within a red U-container's row span and its column lies between that container's side walls inclusively, fill that container's hollow interior with the current legend color the first time this happens.
    - Containers never refill on later crossings, and any container never crossed by the ray stays hollow.

    Key insight:
    - The fill color is controlled by bounce segments, not by container order: two containers crossed before the next reflection receive the same color, while a later bounce changes the color even if the geometry is otherwise similar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start, cycle = _start_and_cycle(g)
    containers = _detect_u_containers(g)
    records = _trace_ray(g.shape[0], g.shape[1], start, len(cycle))

    out = g.copy()
    out[start] = BACKGROUND
    filled: set[int] = set()

    for record in records:
        color = cycle[record["color_index"]]
        for idx, box in enumerate(containers):
            if idx in filled:
                continue
            if box["top"] <= record["r"] <= box["bottom"] and box["left"] <= record["c"] <= box["right"]:
                _fill_box(out, box, color)
                filled.add(idx)
        if len(filled) == len(containers):
            break

    return out


def _choose_legend_rows(height: int, cycle_len: int) -> list[int]:
    for _ in range(100):
        rows = sorted(random.sample(range(1, height - 1), cycle_len))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    raise RuntimeError("failed to place legend rows")


def _make_box(top: int, left: int, height: int, width: int) -> dict[str, object]:
    return {
        "top": int(top),
        "bottom": int(top + height - 1),
        "left": int(left),
        "right": int(left + width - 1),
        "height": int(height),
        "width": int(width),
    }


def _record_hits_box(record: dict[str, int], box: dict[str, object]) -> bool:
    return (
        int(box["top"]) <= int(record["r"]) <= int(box["bottom"])
        and int(box["left"]) <= int(record["c"]) <= int(box["right"])
    )


def _box_options_for_record(height: int, width: int, records: list[dict[str, int]], idx: int) -> list[dict[str, object]]:
    record = records[idx]
    r = int(record["r"])
    c = int(record["c"])
    options: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for box_h in (3, 4):
        for box_w in (3, 4, 5):
            anchors: list[tuple[int, int]] = []

            if 2 <= r <= height - box_h - 1:
                left_min = max(2, c - box_w + 1)
                left_max = min(c, width - box_w - 1)
                anchors.extend((r, left) for left in range(left_min, left_max + 1))

            if 2 <= c <= width - box_w - 1:
                top_min = max(2, r - box_h + 1)
                top_max = min(r, height - box_h - 1)
                anchors.extend((top, c) for top in range(top_min, top_max + 1))

            for top, left in anchors:
                key = (top, left, box_h, box_w)
                if key in seen:
                    continue
                box = _make_box(top, left, box_h, box_w)
                if any(_record_hits_box(prev, box) for prev in records[:idx]):
                    continue
                box["first_index"] = int(idx)
                box["first"] = dict(record)
                box["side_hit"] = bool(c in (int(box["left"]), int(box["right"])))
                options.append(box)
                seen.add(key)

    random.shuffle(options)
    return options[:12]


def _compatible(a: dict[str, object], b: dict[str, object]) -> bool:
    atop, abot = int(a["top"]) - 1, int(a["bottom"]) + 1
    aleft, aright = int(a["left"]) - 1, int(a["right"]) + 1
    btop, bbot = int(b["top"]) - 1, int(b["bottom"]) + 1
    bleft, bright = int(b["left"]) - 1, int(b["right"]) + 1
    return abot < btop or bbot < atop or aright < bleft or bright < aleft


def _pick_untouched_box(
    height: int,
    width: int,
    records: list[dict[str, int]],
    occupied: list[dict[str, object]],
) -> dict[str, object] | None:
    for _ in range(200):
        box_h = random.choice((3, 4))
        box_w = random.choice((3, 4, 5))
        top = random.randint(2, height - box_h - 1)
        left = random.randint(2, width - box_w - 1)
        box = _make_box(top, left, box_h, box_w)
        if not all(_compatible(box, other) for other in occupied):
            continue
        if any(_record_hits_box(record, box) for record in records):
            continue
        return box

    for box_h in (3, 4):
        for box_w in (3, 4, 5):
            for top in range(2, height - box_h):
                for left in range(2, width - box_w):
                    box = _make_box(top, left, box_h, box_w)
                    if not all(_compatible(box, other) for other in occupied):
                        continue
                    if any(_record_hits_box(record, box) for record in records):
                        continue
                    return box
    return None


def _pick_boxes(height: int, width: int, records: list[dict[str, int]]) -> list[dict[str, object]] | None:
    option_cache: dict[int, list[dict[str, object]]] = {}

    def options_for(idx: int) -> list[dict[str, object]]:
        if idx not in option_cache:
            option_cache[idx] = _box_options_for_record(height, width, records, idx)
        return option_cache[idx]

    by_bounce: dict[int, list[int]] = {}
    for idx, record in enumerate(records):
        by_bounce.setdefault(int(record["bounce_count"]), []).append(idx)

    eligible_bounces = [
        bounce_count
        for bounce_count, indices in by_bounce.items()
        if sum(1 for idx in indices if options_for(idx)) >= 2
    ]
    if not eligible_bounces:
        return None

    valid_indices = [idx for idx, _record in enumerate(records) if options_for(idx)]

    for _ in range(96):
        bounce_count = random.choice(eligible_bounces)
        same_segment = [idx for idx in by_bounce[bounce_count] if options_for(idx)]
        if len(same_segment) < 2:
            continue

        idx_a, idx_b = random.sample(same_segment, 2)
        pair: list[dict[str, object]] | None = None
        for box_a in options_for(idx_a)[:8]:
            for box_b in options_for(idx_b)[:8]:
                if _compatible(box_a, box_b):
                    pair = [box_a, box_b]
                    break
            if pair is not None:
                break
        if pair is None:
            continue

        pair_color = int(pair[0]["first"]["color_index"])
        third_indices = [
            idx for idx in valid_indices if int(records[idx]["color_index"]) != pair_color
        ]
        random.shuffle(third_indices)

        third_box = None
        for idx_c in third_indices[:16]:
            for box_c in options_for(idx_c)[:8]:
                if all(_compatible(box_c, box) for box in pair):
                    third_box = box_c
                    break
            if third_box is not None:
                break
        if third_box is None:
            continue

        chosen = pair + [third_box]
        blank = _pick_untouched_box(height, width, records, chosen)
        if blank is not None:
            return chosen + [blank]
    return None


def _draw_u(grid: np.ndarray, box: dict[str, object]) -> None:
    top = int(box["top"])
    bottom = int(box["bottom"])
    left = int(box["left"])
    right = int(box["right"])
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR


def _manual_expected(
    grid: np.ndarray,
    start: tuple[int, int],
    cycle_colors: list[int],
    traversed_boxes: list[dict[str, object]],
    untouched_boxes: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[start] = BACKGROUND
    for box in traversed_boxes:
        first = box["first"]
        assert isinstance(first, dict)
        _fill_box(out, box, cycle_colors[int(first["color_index"])])
    for box in untouched_boxes:
        _fill_box(out, box, BACKGROUND)
    return out


def _annotate_template_boxes(
    height: int,
    width: int,
    start: tuple[int, int],
    cycle_len: int,
    template_boxes: list[dict[str, int | bool]],
) -> list[dict[str, object]] | None:
    records = _trace_ray(height, width, start, cycle_len)
    chosen: list[dict[str, object]] = []

    for spec in template_boxes:
        box = _make_box(int(spec["top"]), int(spec["left"]), int(spec["height"]), int(spec["width"]))
        if bool(spec["traversed"]):
            first = next((record for record in records if _record_hits_box(record, box)), None)
            if first is None:
                return None
            box["first"] = dict(first)
        else:
            if any(_record_hits_box(record, box) for record in records):
                return None
        chosen.append(box)

    return chosen


def _validate_generated(
    grid: np.ndarray,
    start: tuple[int, int],
    cycle_colors: list[int],
    chosen_boxes: list[dict[str, object]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    traversed = [box for box in chosen_boxes if "first" in box]
    untouched = [box for box in chosen_boxes if "first" not in box]
    if len(traversed) != 3 or len(untouched) != 1:
        return False

    counts_by_segment: dict[int, int] = {}
    for box in traversed:
        seg = int(box["first"]["bounce_count"])
        counts_by_segment[seg] = counts_by_segment.get(seg, 0) + 1
    if max(counts_by_segment.values()) < 2:
        return False
    if len({int(box["first"]["color_index"]) for box in traversed}) < 2:
        return False

    start2, cycle2 = _start_and_cycle(grid)
    if start2 != start or cycle2 != cycle_colors:
        return False

    detected = _detect_u_containers(grid)
    if len(detected) != 4:
        return False

    records = _trace_ray(grid.shape[0], grid.shape[1], start, len(cycle_colors))
    by_box = {
        (int(box["top"]), int(box["left"]), int(box["height"]), int(box["width"])): box
        for box in chosen_boxes
    }
    for det in detected:
        key = (det["top"], det["left"], det["height"], det["width"])
        if key not in by_box:
            return False
        first = None
        for record in records:
            if det["top"] <= record["r"] <= det["bottom"] and det["left"] <= record["c"] <= det["right"]:
                first = record
                break
        chosen = by_box[key]
        if "first" not in chosen:
            if first is not None:
                return False
        else:
            if first is None:
                return False
            expected_first = chosen["first"]
            if int(first["bounce_count"]) != int(expected_first["bounce_count"]):
                return False
            if int(first["color_index"]) != int(expected_first["color_index"]):
                return False

    expected = _manual_expected(grid, start, cycle_colors, traversed, untouched)
    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    template = random.choice(GEN_TEMPLATES)
    height = int(template["height"])
    width = int(template["width"])
    cycle_len = random.randint(max(3, GEN_CYCLE_LEN_RANGE[0]), GEN_CYCLE_LEN_RANGE[1])
    cycle_colors = random.sample(COLOR_POOL, cycle_len)
    start = tuple(template["start"])
    chosen = _annotate_template_boxes(height, width, start, cycle_len, template["boxes"])
    if chosen is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[start] = START_COLOR

    legend_rows = _choose_legend_rows(height, cycle_len)
    for row, color in zip(legend_rows, cycle_colors):
        grid[row, 0] = int(color)

    for box in chosen:
        _draw_u(grid, box)

    if not _validate_generated(grid, start, cycle_colors, chosen):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate ray/container puzzle")
