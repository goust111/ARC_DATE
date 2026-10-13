import random
from collections import Counter

import numpy as np


GEN_ATTEMPTS = 500
PALETTE = list(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    """4-connected monochromatic components as dicts with coords/bbox metadata."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_bar(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    if height != 1 or width < 2:
        return False
    return component["area"] == width


def _is_container(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    if height < 3 or width < 3:
        return False
    mask = _component_mask(component)
    template = np.zeros_like(mask)
    template[:, 0] = True
    template[:, -1] = True
    template[-1, :] = True
    return bool(np.array_equal(mask, template))


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)

    containers = []
    bars = []
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        if _is_bar(component):
            bars.append(
                {
                    **component,
                    "row": r0,
                    "c0": c0,
                    "length": c1 - c0 + 1,
                }
            )
        elif _is_container(component):
            containers.append(
                {
                    **component,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "inner_width": c1 - c0 - 1,
                    "opening_c0": c0 + 1,
                    "seat_row": r1 - 1,
                }
            )
        else:
            raise AssertionError("every foreground component must be either a bar or a hollow U container")

    containers.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    for idx, container in enumerate(containers):
        container["id"] = idx
    bars.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    assert containers, "expected at least one container"
    assert bars, "expected at least one bar"
    assert len({int(container["color"]) for container in containers}) == 1, "all containers must share one wall color"
    assert len({int(container["inner_width"]) for container in containers}) == len(containers), "container inner widths must be unique"
    assert len({int(bar["length"]) for bar in bars}) == len(bars), "bar lengths must be unique"
    assert len({int(bar["color"]) for bar in bars}) == len(bars), "bar colors must be unique"

    return bg, containers, bars


def _drop_row(occupied: np.ndarray, row: int, c0: int, length: int) -> int:
    """Return the final row of a horizontal bar falling vertically in fixed columns."""
    h, _w = occupied.shape
    r = int(row)
    while r + 1 < h and not bool(occupied[r + 1, c0 : c0 + length].any()):
        r += 1
    return r


def _support_type(grid: np.ndarray, final_row: int, c0: int, length: int, bg: int, wall_color: int) -> str:
    if final_row == grid.shape[0] - 1:
        return "floor"
    below = grid[final_row + 1, c0 : c0 + length]
    if np.any((below != bg) & (below != wall_color)):
        return "bar"
    if np.any(below == wall_color):
        return "wall"
    raise AssertionError("falling bar must stop on floor, a container, or a settled bar")


def _transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg, containers, bars = _parse_scene(g)
    wall_color = int(containers[0]["color"])
    width_to_container = {int(container["inner_width"]): container for container in containers}

    matched = []
    unmatched = []
    for bar in bars:
        if int(bar["length"]) in width_to_container:
            matched.append(bar)
        else:
            unmatched.append(bar)

    out = g.copy()
    for bar in bars:
        out[int(bar["row"]), int(bar["c0"]) : int(bar["c0"]) + int(bar["length"])] = bg

    occupied = out != bg
    matched_info = []
    for bar in matched:
        container = width_to_container[int(bar["length"])]
        row = int(container["seat_row"])
        c0 = int(container["opening_c0"])
        sl = slice(c0, c0 + int(bar["length"]))
        assert not bool(occupied[row, sl].any()), "matched bar seat must be empty"
        out[row, sl] = int(bar["color"])
        occupied[row, sl] = True
        matched_info.append({"bar": bar, "container": container, "final_row": row, "final_c0": c0})

    unmatched_info = []
    for bar in sorted(unmatched, key=lambda item: (int(item["row"]), int(item["c0"])), reverse=True):
        final_row = _drop_row(occupied, int(bar["row"]), int(bar["c0"]), int(bar["length"]))
        sl = slice(int(bar["c0"]), int(bar["c0"]) + int(bar["length"]))
        support = _support_type(out, final_row, int(bar["c0"]), int(bar["length"]), bg, wall_color)
        out[final_row, sl] = int(bar["color"])
        occupied[final_row, sl] = True
        unmatched_info.append({"bar": bar, "final_row": final_row, "final_c0": int(bar["c0"]), "support": support})

    return out, {
        "bg": bg,
        "wall_color": wall_color,
        "containers": containers,
        "bars": bars,
        "matched": matched_info,
        "unmatched": unmatched_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each horizontal bar into the uniquely matching U container by width, otherwise let it fall straight down.

    Input structure:
    - The grid has a dominant black background with several disconnected monochromatic foreground components.
    - Lower components are open-upward hollow U containers that all share one wall color and have distinct inner widths.
    - Upper components are one-pixel-thick horizontal bars with distinct colors and distinct lengths.

    Rule:
    - Clear every bar from its original position.
    - Any bar whose length equals a container's inner width slides horizontally to that container, then settles on the inner bottom between the two walls.
    - After all matching bars are seated, every non-matching bar falls vertically in its original columns until the next step would hit a container, a settled bar, or the canvas bottom.

    Key insight:
    - Two motion systems coexist: width equality acts like a key-to-lock teleport for matched bars, while only the leftover bars obey ordinary vertical gravity.
    """
    return _transform(np.asarray(grid, dtype=int))[0]


def _paint_container(grid: np.ndarray, container: dict, color: int) -> None:
    r0, c0, r1, c1 = container["bbox"]
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r1, c0 : c1 + 1] = color


def _sample_layout() -> tuple[int, int, list[dict]] | None:
    n_containers = random.randint(2, 4)
    width_pool = [2, 3, 4, 5, 6]
    inner_widths = random.sample(width_pool, n_containers)
    random.shuffle(inner_widths)
    heights = [random.randint(4, 7) for _ in range(n_containers)]
    gaps = [random.randint(1, 3) for _ in range(n_containers - 1)]
    total_width = sum(width + 2 for width in inner_widths) + sum(gaps)
    min_width = total_width + 2
    if min_width > 28:
        return None

    width = random.randint(max(18, min_width), 28)
    min_height = max(16, max(heights) + 8)
    if min_height > 24:
        return None
    height = random.randint(min_height, 24)

    extra = width - total_width - 2
    if extra < 0:
        return None
    left_margin = 1 + random.randint(0, extra)

    containers = []
    c = left_margin
    for idx, (inner_width, box_height) in enumerate(zip(inner_widths, heights)):
        outer_width = inner_width + 2
        r0 = height - box_height
        bbox = (r0, c, height - 1, c + outer_width - 1)
        containers.append(
            {
                "id": idx,
                "bbox": bbox,
                "height": box_height,
                "width": outer_width,
                "inner_width": inner_width,
                "opening_c0": c + 1,
                "seat_row": height - 2,
            }
        )
        c += outer_width
        if idx < len(gaps):
            c += gaps[idx]

    return height, width, containers


def _build_support_grid(height: int, width: int, containers: list[dict], matched_ids: set[int]) -> np.ndarray:
    """0=bg, 1=container wall, 2=already seated matched bar."""
    grid = np.zeros((height, width), dtype=int)
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        grid[r0 : r1 + 1, c0] = 1
        grid[r0 : r1 + 1, c1] = 1
        grid[r1, c0 : c1 + 1] = 1
    for container in containers:
        if int(container["id"]) not in matched_ids:
            continue
        c0 = int(container["opening_c0"])
        grid[int(container["seat_row"]), c0 : c0 + int(container["inner_width"])] = 2
    return grid


def _candidate_unmatched_bars(
    support_grid: np.ndarray,
    max_bar_row: int,
    available_lengths: list[int],
) -> dict[str, list[dict]]:
    occupied = support_grid != 0
    height, width = support_grid.shape
    candidates = {"floor": [], "wall": [], "bar": []}

    for length in available_lengths:
        if length >= width:
            continue
        for row in range(1, max_bar_row + 1):
            for c0 in range(0, width - length + 1):
                if bool(occupied[row, c0 : c0 + length].any()):
                    continue
                final_row = _drop_row(occupied, row, c0, length)
                if final_row - row < 2:
                    continue
                if final_row == height - 1:
                    kind = "floor"
                else:
                    below = support_grid[final_row + 1, c0 : c0 + length]
                    kind = "bar" if np.any(below == 2) else "wall"
                candidates[kind].append(
                    {
                        "row": row,
                        "c0": c0,
                        "length": length,
                        "final_row": final_row,
                        "kind": kind,
                    }
                )

    for kind in candidates:
        random.shuffle(candidates[kind])
    return candidates


def _sample_unmatched_types(candidates: dict[str, list[dict]]) -> list[str]:
    available = [kind for kind, options in candidates.items() if options]
    if not available:
        return []

    weighted = []
    for kind, weight in (("bar", 4), ("wall", 3), ("floor", 3)):
        if kind in available:
            weighted.extend([kind] * weight)

    chosen = [random.choice(weighted)]
    remaining = [kind for kind in available if kind not in chosen]
    if remaining and random.random() < 0.65:
        chosen.append(random.choice(remaining))
        remaining = [kind for kind in remaining if kind not in chosen]
    if remaining and random.random() < 0.25:
        chosen.append(random.choice(remaining))
    return chosen


def _pick_unmatched_bars(candidates: dict[str, list[dict]], reserved_lengths: set[int]) -> list[dict] | None:
    for _ in range(100):
        desired_types = _sample_unmatched_types(candidates)
        if not desired_types:
            return None

        chosen = []
        used_lengths = set(reserved_lengths)
        used_cols: set[int] = set()
        ok = True

        for kind in desired_types:
            pick = None
            for candidate in candidates[kind]:
                cols = set(range(int(candidate["c0"]), int(candidate["c0"]) + int(candidate["length"])))
                if int(candidate["length"]) in used_lengths or cols & used_cols:
                    continue
                pick = candidate
                break
            if pick is None:
                ok = False
                break
            chosen.append(pick)
            used_lengths.add(int(pick["length"]))
            used_cols |= set(range(int(pick["c0"]), int(pick["c0"]) + int(pick["length"])))

        if not ok or not chosen:
            continue

        extras = [kind for kind in ("bar", "wall", "floor") if kind not in desired_types and candidates[kind]]
        if extras and len(chosen) < 3 and random.random() < 0.35:
            kind = random.choice(extras)
            for candidate in candidates[kind]:
                cols = set(range(int(candidate["c0"]), int(candidate["c0"]) + int(candidate["length"])))
                if int(candidate["length"]) in used_lengths or cols & used_cols:
                    continue
                chosen.append(candidate)
                break

        return chosen

    return None


def _place_original_bar(
    grid: np.ndarray,
    length: int,
    max_bar_row: int,
    blocked_cells: np.ndarray,
    avoid_c0: int | None = None,
) -> tuple[int, int] | None:
    positions = []
    height, width = grid.shape
    for row in range(1, max_bar_row + 1):
        for c0 in range(0, width - length + 1):
            if avoid_c0 is not None and c0 == avoid_c0:
                continue
            if bool(blocked_cells[row, c0 : c0 + length].any()):
                continue
            if bool((grid[row, c0 : c0 + length] != 0).any()):
                continue
            positions.append((row, c0))
    if not positions:
        return None
    return random.choice(positions)


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    height, width, containers = layout

    matched_count = random.randint(1, len(containers) - 1)
    matched_ids = set(random.sample([int(container["id"]) for container in containers], matched_count))
    matched_containers = [container for container in containers if int(container["id"]) in matched_ids]
    reserved_lengths = {int(container["inner_width"]) for container in matched_containers}

    top_limit = min(container["bbox"][0] for container in containers) - 4
    if top_limit < 3:
        return None
    max_bar_row = random.randint(3, min(7, top_limit))

    support_grid = _build_support_grid(height, width, containers, matched_ids)
    available_lengths = [length for length in range(2, min(9, width)) if length not in {int(container["inner_width"]) for container in containers}]
    candidates = _candidate_unmatched_bars(support_grid, max_bar_row, available_lengths)
    unmatched_bars = _pick_unmatched_bars(candidates, reserved_lengths)
    if not unmatched_bars:
        return None

    total_bars = matched_count + len(unmatched_bars)
    if total_bars > len(PALETTE) - 1:
        return None

    wall_color = random.choice(PALETTE)
    bar_colors = random.sample([color for color in PALETTE if color != wall_color], total_bars)
    input_grid = np.zeros((height, width), dtype=int)

    for container in containers:
        _paint_container(input_grid, container, wall_color)

    blocked_cells = np.zeros((height, width), dtype=bool)
    original_bars = []
    color_iter = iter(bar_colors)

    for container in matched_containers:
        length = int(container["inner_width"])
        slot = _place_original_bar(
            input_grid,
            length,
            max_bar_row,
            blocked_cells,
            avoid_c0=int(container["opening_c0"]),
        )
        if slot is None:
            return None
        row, c0 = slot
        color = next(color_iter)
        input_grid[row, c0 : c0 + length] = color
        blocked_cells[row, c0 : c0 + length] = True
        original_bars.append({"row": row, "c0": c0, "length": length, "color": color, "matched": True})

    for candidate in unmatched_bars:
        row = int(candidate["row"])
        c0 = int(candidate["c0"])
        length = int(candidate["length"])
        if bool(blocked_cells[row, c0 : c0 + length].any()):
            return None
        color = next(color_iter)
        input_grid[row, c0 : c0 + length] = color
        blocked_cells[row, c0 : c0 + length] = True
        original_bars.append(
            {
                "row": row,
                "c0": c0,
                "length": length,
                "color": color,
                "matched": False,
                "expected_support": candidate["kind"],
            }
        )

    if not _validate_generated(input_grid):
        return None
    return input_grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out, info = _transform(grid)
    except AssertionError:
        return False

    containers = info["containers"]
    bars = info["bars"]
    matched = info["matched"]
    unmatched = info["unmatched"]
    if not matched or not unmatched:
        return False
    if len({int(entry["container"]["id"]) for entry in matched}) >= len(containers):
        return False
    if np.array_equal(out, grid):
        return False

    min_container_top = min(int(container["bbox"][0]) for container in containers)
    if max(int(bar["row"]) for bar in bars) >= min_container_top - 2:
        return False

    width_to_container = {int(container["inner_width"]): container for container in containers}
    for entry in matched:
        bar = entry["bar"]
        container = width_to_container[int(bar["length"])]
        if int(bar["c0"]) == int(container["opening_c0"]):
            return False
        row = int(container["seat_row"])
        c0 = int(container["opening_c0"])
        if not np.all(out[row, c0 : c0 + int(bar["length"])] == int(bar["color"])):
            return False

    unmatched_bars = [entry["bar"] for entry in unmatched]
    for i, first in enumerate(unmatched_bars):
        first_cols = set(range(int(first["c0"]), int(first["c0"]) + int(first["length"])))
        for second in unmatched_bars[i + 1 :]:
            second_cols = set(range(int(second["c0"]), int(second["c0"]) + int(second["length"])))
            if first_cols & second_cols:
                return False

    support_types = {str(entry["support"]) for entry in unmatched}
    if not support_types:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
