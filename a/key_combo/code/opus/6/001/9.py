from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, find_components, in_bounds, neighbors4, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import components_mask, find_components, in_bounds, neighbors4, paste


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

PANEL_SIZE = 7
INNER_SIZE = 5
LEGEND_SIZE = 3
SLOT_SIZE = 4
INDEX_COLORS = (BLUE, RED, GREEN)
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 20
GRID_MAX = 24
GEN_ATTEMPTS = 400


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _panel_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    panels = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    ):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (PANEL_SIZE, PANEL_SIZE):
            continue
        if int(comp["area"]) != PANEL_SIZE * PANEL_SIZE - INNER_SIZE * INNER_SIZE:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.ones((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        border[1:-1, 1:-1] = False
        if not np.all(patch[border] == GREY):
            continue
        panels.append((r0, c0, r1, c1))
    panels.sort()
    assert 2 <= len(panels) <= 3
    return panels


def _nonzero_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for nr, nc in neighbors4(r, c):
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
            out.append((int(nr), int(nc)))
    return out


def _legend_block_from_label(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    if len(_nonzero_neighbors(grid, r, c)) != 1:
        return None

    h, w = grid.shape
    matches: list[tuple[int, int]] = []

    for k in range(LEGEND_SIZE):
        rr = r - k
        cc = c + 1
        if 0 <= rr and rr + LEGEND_SIZE <= h and 0 <= cc and cc + LEGEND_SIZE <= w:
            block = grid[rr : rr + LEGEND_SIZE, cc : cc + LEGEND_SIZE]
            if np.all(block != BG):
                matches.append((int(rr), int(cc)))

    for k in range(LEGEND_SIZE):
        rr = r + 1
        cc = c - k
        if 0 <= rr and rr + LEGEND_SIZE <= h and 0 <= cc and cc + LEGEND_SIZE <= w:
            block = grid[rr : rr + LEGEND_SIZE, cc : cc + LEGEND_SIZE]
            if np.all(block != BG):
                matches.append((int(rr), int(cc)))

    uniq = sorted(set(matches))
    if len(uniq) != 1:
        return None
    return uniq[0]


def _extract_legends(grid: np.ndarray) -> dict[int, np.ndarray]:
    legends: dict[int, np.ndarray] = {}
    used_blocks: set[tuple[int, int]] = set()
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in INDEX_COLORS:
                continue
            block = _legend_block_from_label(grid, r, c)
            if block is None or block in used_blocks:
                continue
            rr, cc = block
            legends[color] = grid[rr : rr + LEGEND_SIZE, cc : cc + LEGEND_SIZE].copy()
            used_blocks.add(block)

    assert 2 <= len(legends) <= 3
    return legends


def _panel_info(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    mask = inner != BG
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1
    coords = comps[0]
    assert 3 <= int(coords.shape[0]) <= 8

    touched = []
    if bool(mask[0].any()):
        touched.append("top")
    if bool(mask[-1].any()):
        touched.append("bottom")
    if bool(mask[:, 0].any()):
        touched.append("left")
    if bool(mask[:, -1].any()):
        touched.append("right")
    assert len(touched) == 1
    side = touched[0]

    if side == "top":
        side_colors = inner[0, mask[0]]
    elif side == "bottom":
        side_colors = inner[-1, mask[-1]]
    elif side == "left":
        side_colors = inner[mask[:, 0], 0]
    else:
        side_colors = inner[mask[:, -1], -1]

    uniq = sorted(set(map(int, side_colors.tolist())))
    assert len(uniq) == 1
    side_color = uniq[0]
    assert side_color in INDEX_COLORS

    return {
        "box": box,
        "inner": inner,
        "side": side,
        "side_color": side_color,
        "area": int(coords.shape[0]),
    }


def _fill_inner(inner: np.ndarray, legend: np.ndarray) -> np.ndarray:
    out = inner.copy()
    for r in range(INNER_SIZE):
        for c in range(INNER_SIZE):
            if int(out[r, c]) == BG:
                out[r, c] = int(legend[r % LEGEND_SIZE, c % LEGEND_SIZE])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey panel interior with the tiled position legend selected by the object's touching-side color.

    Input structure:
    - Black background surrounds 2 or 3 solid 7x7 grey panels.
    - Each panel has a 5x5 inner window containing one 4-connected foreground object; the object touches exactly one side of that 5x5 window.
    - Elsewhere, 2 or 3 separate 3x3 non-black legend blocks are marked by a single adjacent index pixel whose color is blue, red, or green.

    Rule:
    - For each panel, find the unique touched side of the inner 5x5 window and read the uniform color of the object pixels on that touched side.
    - Use that color as the key for the legend whose adjacent index pixel has the same color.
    - Keep the object unchanged, but replace every black cell of the 5x5 window by the legend color at the corresponding position modulo 3 in both row and column.

    Key insight:
    - The touching-side color does not recolor the object itself; it chooses which periodic 3x3 position-to-color legend should fill the panel's missing cells.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legends = _extract_legends(g)
    for box in _panel_boxes(g):
        info = _panel_info(g, box)
        legend = legends[info["side_color"]]
        filled = _fill_inner(info["inner"], legend)
        r0, c0, r1, c1 = box
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _side_cells(side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, 4)]
    if side == "bottom":
        return [(4, c) for c in range(1, 4)]
    if side == "left":
        return [(r, 0) for r in range(1, 4)]
    return [(r, 4) for r in range(1, 4)]


def _allowed_cells(side: str) -> set[tuple[int, int]]:
    return set(_side_cells(side)) | {(r, c) for r in range(1, 4) for c in range(1, 4)}


def _contiguous_side_run(side: str) -> list[tuple[int, int]]:
    cells = _side_cells(side)
    run_len = random.randint(1, min(3, len(cells)))
    start = random.randint(0, len(cells) - run_len)
    return cells[start : start + run_len]


def _grow_object_mask(side: str) -> np.ndarray | None:
    allowed = _allowed_cells(side)
    target = random.randint(3, 8)
    seeds = _contiguous_side_run(side)
    if len(seeds) >= target:
        return None

    cells = set(seeds)
    frontier = set()
    for r, c in seeds:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in allowed and (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))

    while len(cells) < target:
        if not frontier:
            return None
        choice = random.choice(sorted(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nr, nc in neighbors4(*choice):
            if (nr, nc) in allowed and (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))

    mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return None
    side_mask = np.zeros_like(mask)
    for r, c in _side_cells(side):
        side_mask[r, c] = True
    if not bool((mask & side_mask).any()) or int(np.count_nonzero(mask & ~side_mask)) == 0:
        return None
    if side != "top" and bool(mask[0].any()):
        return None
    if side != "bottom" and bool(mask[-1].any()):
        return None
    if side != "left" and bool(mask[:, 0].any()):
        return None
    if side != "right" and bool(mask[:, -1].any()):
        return None
    return mask


def _paint_object(side: str, side_color: int) -> tuple[np.ndarray, dict] | None:
    mask = _grow_object_mask(side)
    if mask is None:
        return None

    side_mask = np.zeros_like(mask)
    for r, c in _side_cells(side):
        side_mask[r, c] = True
    contact = mask & side_mask
    other = mask & ~side_mask
    other_coords = [tuple(map(int, rc)) for rc in np.argwhere(other)]
    if not other_coords:
        return None

    inner = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
    inner[contact] = side_color

    alt_colors = [c for c in INDEX_COLORS if c != side_color]
    used_alts = random.sample(alt_colors, random.randint(1, len(alt_colors)))
    random.shuffle(other_coords)
    for i, (r, c) in enumerate(other_coords):
        inner[r, c] = used_alts[i % len(used_alts)]

    return inner, {"side": side, "side_color": side_color, "mask": mask}


def _random_legend(index_color: int, existing: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        legend = np.array(random.sample(range(1, 10), 9), dtype=int).reshape(3, 3)
        if any(np.array_equal(legend, old) for old in existing):
            continue
        orient = random.choice(["left", "above"])
        slot = np.zeros((SLOT_SIZE, SLOT_SIZE), dtype=int)
        if orient == "left":
            rows = [r for r in range(3) if int(legend[r, 0]) != index_color]
            if not rows:
                continue
            rr = random.choice(rows)
            slot[:3, 1:4] = legend
            slot[rr, 0] = index_color
        else:
            cols = [c for c in range(3) if int(legend[0, c]) != index_color]
            if not cols:
                continue
            cc = random.choice(cols)
            slot[1:4, :3] = legend
            slot[0, cc] = index_color
        return legend, slot
    return np.zeros((3, 3), dtype=int), np.zeros((4, 4), dtype=int)


def _legend_strip(colors: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    slots = []
    legends: dict[int, np.ndarray] = {}
    existing: list[np.ndarray] = []
    for color in colors:
        legend, slot = _random_legend(color, existing)
        existing.append(legend)
        legends[color] = legend
        slots.append(slot)

    width = len(slots) * SLOT_SIZE + (len(slots) - 1)
    strip = np.zeros((SLOT_SIZE, width), dtype=int)
    col = 0
    for slot in slots:
        paste(strip, slot, 0, col, transparent=BG)
        col += SLOT_SIZE + 1
    return strip, legends


def _sample_panel_specs(n_panels: int) -> tuple[list[dict], list[int]]:
    if n_panels == 2:
        colors = random.sample(list(INDEX_COLORS), 2)
        side = random.choice(SIDES)
        specs = [
            {"side_color": colors[0], "side": side},
            {"side_color": colors[1], "side": side},
        ]
        random.shuffle(specs)
        return specs, sorted(colors)

    if random.random() < 0.75:
        colors = random.sample(list(INDEX_COLORS), 2)
        side_a, side_b = random.sample(list(SIDES), 2)
        specs = [
            {"side_color": colors[0], "side": side_a},
            {"side_color": colors[0], "side": side_b},
            {"side_color": colors[1], "side": side_a},
        ]
        random.shuffle(specs)
        return specs, sorted(colors)

    colors = random.sample(list(INDEX_COLORS), 3)
    shared_side = random.choice(SIDES)
    other_side = random.choice([side for side in SIDES if side != shared_side])
    specs = [
        {"side_color": colors[0], "side": shared_side},
        {"side_color": colors[1], "side": shared_side},
        {"side_color": colors[2], "side": other_side},
    ]
    random.shuffle(specs)
    return specs, sorted(colors)


def _place_panels(
    shape: tuple[int, int], legend_box: tuple[int, int, int, int], n_panels: int
) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r in range(h - PANEL_SIZE + 1):
        for c in range(w - PANEL_SIZE + 1):
            box = (r, c, r + PANEL_SIZE - 1, c + PANEL_SIZE - 1)
            if _boxes_far(box, legend_box, pad=1):
                candidates.append((r, c))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    chosen_boxes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        box = (r, c, r + PANEL_SIZE - 1, c + PANEL_SIZE - 1)
        if all(_boxes_far(box, old, pad=1) for old in chosen_boxes):
            chosen.append((r, c))
            chosen_boxes.append(box)
            if len(chosen) == n_panels:
                return chosen
    return []


def _render_panel(spec: dict) -> tuple[np.ndarray, dict] | None:
    painted = _paint_object(spec["side"], spec["side_color"])
    if painted is None:
        return None
    inner, obj_meta = painted
    panel = np.full((PANEL_SIZE, PANEL_SIZE), GREY, dtype=int)
    panel[1:6, 1:6] = inner
    return panel, obj_meta


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    for panel in meta["panels"]:
        r0, c0 = panel["top_left"]
        inner = grid[r0 + 1 : r0 + 6, c0 + 1 : c0 + 6]
        legend = meta["legends"][panel["side_color"]]
        out[r0 + 1 : r0 + 6, c0 + 1 : c0 + 6] = _fill_inner(inner, legend)
    return out


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    legends = _extract_legends(grid)
    if set(legends) != set(meta["legends"]):
        return False
    for color, legend in legends.items():
        if not np.array_equal(legend, meta["legends"][color]):
            return False

    parsed_panels = {info["box"]: info for info in (_panel_info(grid, box) for box in _panel_boxes(grid))}
    if len(parsed_panels) != len(meta["panels"]):
        return False

    touched_sides = []
    side_colors = []
    panel_outputs = []
    changed = 0
    for panel in meta["panels"]:
        r0, c0 = panel["top_left"]
        box = (r0, c0, r0 + 6, c0 + 6)
        if box not in parsed_panels:
            return False
        info = parsed_panels[box]
        if info["side"] != panel["side"] or info["side_color"] != panel["side_color"]:
            return False
        touched_sides.append(info["side"])
        side_colors.append(info["side_color"])
        out_inner = expected[r0 + 1 : r0 + 6, c0 + 1 : c0 + 6]
        panel_outputs.append(out_inner.tobytes())
        changed += int(np.count_nonzero(out_inner != grid[r0 + 1 : r0 + 6, c0 + 1 : c0 + 6]))
        fill_colors = sorted(set(map(int, out_inner[grid[r0 + 1 : r0 + 6, c0 + 1 : c0 + 6] == BG].tolist())))
        if len(fill_colors) < 3:
            return False

    if changed == 0 or len(set(panel_outputs)) < 2:
        return False

    if len(meta["panels"]) == 2:
        if not (len(set(side_colors)) == 2 and len(set(touched_sides)) == 1):
            return False
    elif len(set(side_colors)) == 2:
        same_color_diff_side = any(
            side_colors[i] == side_colors[j] and touched_sides[i] != touched_sides[j]
            for i in range(len(side_colors))
            for j in range(i + 1, len(side_colors))
        )
        same_side_diff_color = any(
            touched_sides[i] == touched_sides[j] and side_colors[i] != side_colors[j]
            for i in range(len(side_colors))
            for j in range(i + 1, len(side_colors))
        )
        if not (same_color_diff_side and same_side_diff_color):
            return False
    else:
        if not (len(set(side_colors)) == 3 and len(set(touched_sides)) < 3):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_panels = random.choices([2, 3], weights=[1, 4], k=1)[0]
    panel_specs, legend_colors = _sample_panel_specs(n_panels)

    legend_strip, legends = _legend_strip(legend_colors)
    strip_h, strip_w = legend_strip.shape
    if strip_w > w:
        return None

    top = random.choice([True, False])
    left = random.choice([True, False])
    strip_r = 0 if top else h - strip_h
    strip_c = 0 if left else w - strip_w
    legend_box = (strip_r, strip_c, strip_r + strip_h - 1, strip_c + strip_w - 1)

    panel_positions = _place_panels((h, w), legend_box, n_panels)
    if len(panel_positions) != n_panels:
        return None

    grid = np.zeros((h, w), dtype=int)
    paste(grid, legend_strip, strip_r, strip_c, transparent=BG)

    panels_meta = []
    for (r0, c0), spec in zip(panel_positions, panel_specs):
        rendered = _render_panel(spec)
        if rendered is None:
            return None
        panel, obj_meta = rendered
        paste(grid, panel, r0, c0, transparent=None)
        panels_meta.append(
            {
                "top_left": (r0, c0),
                "side": obj_meta["side"],
                "side_color": obj_meta["side_color"],
            }
        )

    meta = {"legends": legends, "panels": panels_meta}
    expected = _expected_from_meta(grid, meta)
    if not _semantic_ok(grid, expected, meta):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
