from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


GREY = Color.GREY
MAX_GENERATION_ATTEMPTS = 400
BASE_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _boundary_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _inner_line_near_to_far(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        assert c0 < gc < c1
        return [(r, gc) for r in range(r0 + 1, r1)]
    if gr == r1:
        assert c0 < gc < c1
        return [(r, gc) for r in range(r1 - 1, r0, -1)]
    if gc == c0:
        assert r0 < gr < r1
        return [(gr, c) for c in range(c0 + 1, c1)]
    if gc == c1:
        assert r0 < gr < r1
        return [(gr, c) for c in range(c1 - 1, c0, -1)]
    raise AssertionError("gap must lie on exactly one wall")


def _outward_direction(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        return BASE_DIRECTIONS["top"]
    if gr == r1:
        return BASE_DIRECTIONS["bottom"]
    if gc == c0:
        return BASE_DIRECTIONS["left"]
    if gc == c1:
        return BASE_DIRECTIONS["right"]
    raise AssertionError("gap must lie on the boundary")


def _scan_outside_line(
    grid: np.ndarray,
    gap: tuple[int, int],
    outward: tuple[int, int],
    bg: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    dr, dc = outward
    rr, cc = gap[0] + dr, gap[1] + dc
    cells = []
    while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(bg):
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    obstacle = (int(rr), int(cc)) if in_bounds(grid.shape, rr, cc) else None
    return cells, obstacle


def _extract_containers(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )

    containers = []
    wall_union: set[tuple[int, int]] = set()
    colored_inside: set[tuple[int, int]] = set()
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        boundary = set(_boundary_coords(bbox))
        wall = {tuple(map(int, rc)) for rc in comp["coords"]}
        if not wall.issubset(boundary):
            continue
        missing = sorted(boundary - wall)
        if len(missing) != 1:
            continue

        gap = missing[0]
        inner_line = _inner_line_near_to_far(bbox, gap)
        outward = _outward_direction(bbox, gap)

        interior = {
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
        }
        inner_line_set = set(inner_line)
        bead_slots = []
        for cell in inner_line:
            color = int(g[cell])
            if color != int(bg):
                assert color != int(GREY), "inner bead line may not contain wall color"
                bead_slots.append(cell)

        non_bg_interior = {cell for cell in interior if int(g[cell]) != int(bg)}
        assert non_bg_interior.issubset(inner_line_set), "all interior colored pixels must lie on the gap axis"
        assert bead_slots, "each container needs at least one bead"

        outer_line, obstacle = _scan_outside_line(g, gap, outward, bg)
        containers.append(
            {
                "bbox": bbox,
                "gap": gap,
                "inner_line": inner_line,
                "outer_line": outer_line,
                "obstacle": obstacle,
                "beads": [int(g[cell]) for cell in bead_slots],
                "bead_slots": bead_slots,
                "wall": wall,
            }
        )
        wall_union |= wall
        colored_inside |= set(bead_slots)

    assert containers, "expected at least one gapped grey container"

    all_colored = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
    }
    assert all_colored == wall_union | colored_inside, "unexpected non-background pixels outside containers"

    path_cells: set[tuple[int, int]] = set()
    for container in containers:
        for cell in container["outer_line"]:
            assert cell not in path_cells, "leak corridors must be disjoint"
            path_cells.add(cell)
    containers.sort(key=lambda item: item["bbox"])
    return bg, containers


def _render_from_container_info(grid: np.ndarray, bg: int, containers: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

    for container in containers:
        beads = list(container["beads"])
        capacity = len(container["outer_line"])
        leaked = beads[:capacity]
        retained = beads[capacity:]

        for cell, color in zip(container["inner_line"], retained):
            out[cell] = int(color)

        if leaked:
            outer_slots = container["outer_line"][-len(leaked) :]
            for color, cell in zip(leaked, reversed(outer_slots)):
                out[cell] = int(color)
    return out


def analyze_input(grid: np.ndarray) -> dict:
    bg, containers = _extract_containers(grid)
    summary = {
        "background": int(bg),
        "container_count": len(containers),
        "blocked_count": 0,
        "boundary_count": 0,
        "retained_count": 0,
        "full_leak_count": 0,
        "zero_capacity_count": 0,
        "gapped_sequence_count": 0,
        "directions": set(),
    }
    for container in containers:
        bbox = container["bbox"]
        gap = container["gap"]
        outward = _outward_direction(bbox, gap)
        summary["directions"].add(outward)
        if container["obstacle"] is None:
            summary["boundary_count"] += 1
        else:
            summary["blocked_count"] += 1
        capacity = len(container["outer_line"])
        if capacity == 0:
            summary["zero_capacity_count"] += 1
        if len(container["beads"]) > capacity:
            summary["retained_count"] += 1
        else:
            summary["full_leak_count"] += 1
        line_positions = {cell: idx for idx, cell in enumerate(container["inner_line"])}
        used = sorted(line_positions[cell] for cell in container["bead_slots"])
        if used and any(b - a > 1 for a, b in zip(used, used[1:])):
            summary["gapped_sequence_count"] += 1
    return summary


def validate_transformation(grid: np.ndarray, output: np.ndarray) -> None:
    bg, containers = _extract_containers(grid)
    expected = _render_from_container_info(grid, bg, containers)
    assert np.array_equal(np.asarray(output, dtype=int), expected), "output does not match the puzzle rule"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each container's bead sequence out through its unique wall gap and compact it against the exterior obstacle, keeping overflow packed just inside the gap.

    Input structure:
    - The background is the dominant color.
    - One or more gray hollow rectangles act as containers; each gray frame has exactly one single-cell gap on one wall.
    - Inside each container, the only colored pixels form a sparse one-dimensional bead sequence on the row or column aligned with that gap; the leak corridor outside the gap is otherwise empty until the grid edge or another container wall.

    Rule:
    - For every container, use the gap location to determine the unique outward leak direction.
    - Order the beads by distance from the gap, nearest first. Those beads leave first, travel furthest, and occupy the exterior corridor as a contiguous block pressed against the obstacle while preserving that front-to-back order.
    - If the exterior corridor is too short, the remaining beads stay inside the container and are compacted contiguously against the gap on the interior side, again preserving order. Keep all walls and the gap itself unchanged, and clear all other interior cells to background.

    Key insight:
    - The single missing wall cell is both the direction cue and the local reference point: it defines which interior line contains the ordered beads and how much exterior capacity is available before compaction.
    """
    g = np.asarray(grid, dtype=int)
    bg, containers = _extract_containers(g)
    return _render_from_container_info(g, bg, containers)


def _choose_canvas_and_boxes() -> tuple[int, int, list[dict]] | None:
    n = random.choices([2, 3, 4], weights=[2, 5, 3])[0]
    widths = [random.randint(5, 8) for _ in range(n)]
    heights = [random.randint(4, 7) for _ in range(n)]
    gaps = [random.randint(2, 5) for _ in range(n - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(0, 4)

    min_width = left_margin + right_margin + sum(widths) + sum(gaps)
    if min_width > 30:
        return None
    width = random.randint(max(16, min_width), min(30, min_width + 4))

    min_height = max(max(heights) + 2, 14)
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 6))

    boxes = []
    c0 = left_margin
    for w, h in zip(widths, heights):
        r0 = random.randint(1, height - h - 1)
        boxes.append({"r0": r0, "c0": c0, "h": h, "w": w})
        c0 += w
        if len(boxes) < n:
            c0 += gaps[len(boxes) - 1]
    return height, width, boxes


def _blocked_by_later(boxes: list[dict], idx: int, gap_row: int) -> int | None:
    my_right = boxes[idx]["c0"] + boxes[idx]["w"] - 1
    blocker_col = None
    for later in boxes[idx + 1 :]:
        later_r0 = later["r0"]
        later_r1 = later_r0 + later["h"] - 1
        if later_r0 <= gap_row <= later_r1:
            blocker_col = later["c0"]
            break
    if blocker_col is None:
        return None
    assert blocker_col > my_right
    return blocker_col


def _choose_gap_rows(boxes: list[dict]) -> list[int] | None:
    for _ in range(200):
        chosen = []
        used = set()
        ok = True
        for box in boxes:
            candidates = [r for r in range(box["r0"] + 1, box["r0"] + box["h"] - 1) if r not in used]
            if not candidates:
                ok = False
                break
            row = random.choice(candidates)
            chosen.append(row)
            used.add(row)
        if not ok:
            continue

        blocked_flags = [(_blocked_by_later(boxes, i, row) is not None) for i, row in enumerate(chosen)]
        if not any(blocked_flags):
            continue
        if not any(not flag for flag in blocked_flags):
            continue
        return chosen
    return None


def _choose_bead_count(line_len: int, capacity: int, mode: str) -> int:
    if mode == "partial":
        assert line_len > capacity
        return random.randint(capacity + 1, line_len)
    assert capacity > 0
    return random.randint(1, min(line_len, capacity))


def _choose_bead_counts(specs: list[dict]) -> bool:
    partial_candidates = [i for i, spec in enumerate(specs) if spec["line_len"] > spec["capacity"]]
    full_candidates = [i for i, spec in enumerate(specs) if spec["capacity"] > 0]

    force_partial = random.choice(partial_candidates) if partial_candidates else None
    remaining_full = [i for i in full_candidates if i != force_partial]
    if remaining_full:
        force_full = random.choice(remaining_full)
    elif full_candidates:
        force_full = random.choice(full_candidates)
    else:
        force_full = None

    for i, spec in enumerate(specs):
        modes = []
        if spec["line_len"] > spec["capacity"]:
            modes.append("partial")
        if spec["capacity"] > 0:
            modes.append("full")
        if not modes:
            modes = ["partial"]

        if i == force_partial and "partial" in modes:
            mode = "partial"
        elif i == force_full and "full" in modes:
            mode = "full"
        else:
            mode = random.choice(modes)
        spec["count"] = _choose_bead_count(spec["line_len"], spec["capacity"], mode)
    return True


def _sample_slots(line_len: int, count: int, prefer_gap: bool) -> list[int]:
    for _ in range(50):
        slots = sorted(random.sample(range(line_len), count))
        if not prefer_gap:
            return slots
        if count < line_len and any(b - a > 1 for a, b in zip(slots, slots[1:])):
            return slots
    return sorted(random.sample(range(line_len), count))


def _sample_colors(count: int, palette: list[int]) -> list[int]:
    colors = [random.choice(palette) for _ in range(count)]
    if count > 1 and len(set(colors)) == 1 and len(palette) > 1:
        idx = random.randrange(count)
        colors[idx] = random.choice([color for color in palette if color != colors[0]])
    return colors


def _render_base_instance(
    height: int,
    width: int,
    bg: int,
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        inp[r0, c0 : c1 + 1] = GREY
        inp[r1, c0 : c1 + 1] = GREY
        inp[r0 : r1 + 1, c0] = GREY
        inp[r0 : r1 + 1, c1] = GREY
        inp[spec["gap"]] = int(bg)

    out[:, :] = inp
    for spec in specs:
        slots = _sample_slots(spec["line_len"], spec["count"], spec["prefer_gap"])
        colors = _sample_colors(spec["count"], spec["palette"])
        spec["slots"] = slots
        spec["colors"] = colors
        for slot, color in zip(slots, colors):
            inp[spec["inner_line"][slot]] = int(color)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        leaked = spec["colors"][: spec["capacity"]]
        retained = spec["colors"][spec["capacity"] :]
        for cell, color in zip(spec["inner_line"], retained):
            out[cell] = int(color)
        if leaked:
            outer_slots = spec["outer_line"][-len(leaked) :]
            for color, cell in zip(leaked, reversed(outer_slots)):
                out[cell] = int(color)
    return inp, out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _choose_canvas_and_boxes()
    if layout is None:
        return None
    height, width, boxes = layout

    gap_rows = _choose_gap_rows(boxes)
    if gap_rows is None:
        return None

    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    palette = [color for color in Color.NOT_BLACK if color not in {bg, GREY}]
    random.shuffle(palette)

    specs = []
    for i, (box, gap_row) in enumerate(zip(boxes, gap_rows)):
        r0, c0, h, w = box["r0"], box["c0"], box["h"], box["w"]
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        gap = (gap_row, bbox[3])
        blocker_col = _blocked_by_later(boxes, i, gap_row)
        if blocker_col is None:
            outer_line = [(gap_row, c) for c in range(bbox[3] + 1, width)]
        else:
            outer_line = [(gap_row, c) for c in range(bbox[3] + 1, blocker_col)]
        specs.append(
            {
                "bbox": bbox,
                "gap": gap,
                "line_len": w - 2,
                "capacity": len(outer_line),
                "outer_line": outer_line,
                "inner_line": _inner_line_near_to_far(bbox, gap),
                "prefer_gap": False,
                "palette": palette[: max(2, min(5, len(palette)))],
            }
        )

    if not _choose_bead_counts(specs):
        return None

    gap_pref_candidates = [spec for spec in specs if spec["count"] < spec["line_len"]]
    if gap_pref_candidates:
        random.choice(gap_pref_candidates)["prefer_gap"] = True
    for spec in specs:
        if not spec["prefer_gap"]:
            spec["prefer_gap"] = random.random() < 0.45 and spec["count"] < spec["line_len"]

    inp, out = _render_base_instance(height, width, bg, specs)
    summary = analyze_input(inp)
    if summary["blocked_count"] < 1 or summary["boundary_count"] < 1:
        return None
    if summary["retained_count"] < 1:
        return None
    if summary["full_leak_count"] < 1 and any(spec["capacity"] > 0 for spec in specs):
        return None
    if summary["gapped_sequence_count"] < 1:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected = built
        k = random.randint(0, 3)
        rotated_inp = np.rot90(inp, k)
        rotated_expected = np.rot90(expected, k)
        try:
            actual = main(rotated_inp)
            validate_transformation(rotated_inp, actual)
        except Exception:
            continue
        if np.array_equal(actual, rotated_expected):
            return rotated_inp
    raise RuntimeError("failed to generate a valid leaking-container ARC input")
