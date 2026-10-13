import random
from typing import Iterable

import numpy as np


BG = 0
MARKER_A = 1
MARKER_B = 2
TEMPLATE_FG = 3
ANCHOR_COLORS = (4, 6, 7, 8, 9)
GREY = 5
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rect_frame(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return set(comp) == _rect_border_cells(r0, c0, r1, c1)


def _template_info(box: np.ndarray) -> tuple[str, bool]:
    assert box.shape[0] >= 4 and box.shape[1] >= 4, "template box too small"
    interior = box[1:-1, 1:-1]
    vals = set(map(int, np.unique(interior))) - {BG}
    assert len(vals) == 1, "template interior must use one foreground color"
    fg = next(iter(vals))

    row_uniform = all(np.all(interior[r, :] == interior[r, 0]) for r in range(interior.shape[0]))
    col_uniform = all(np.all(interior[:, c] == interior[0, c]) for c in range(interior.shape[1]))

    horizontal = (
        row_uniform
        and interior.shape[0] >= 2
        and any(int(interior[r, 0]) == fg for r in range(interior.shape[0]))
        and any(int(interior[r, 0]) == BG for r in range(interior.shape[0]))
        and all(int(interior[r, 0]) != int(interior[r + 1, 0]) for r in range(interior.shape[0] - 1))
    )
    vertical = (
        col_uniform
        and interior.shape[1] >= 2
        and any(int(interior[0, c]) == fg for c in range(interior.shape[1]))
        and any(int(interior[0, c]) == BG for c in range(interior.shape[1]))
        and all(int(interior[0, c]) != int(interior[0, c + 1]) for c in range(interior.shape[1] - 1))
    )

    assert horizontal ^ vertical, "template must be uniquely horizontal or vertical"
    if horizontal:
        return "horizontal", bool(int(interior[0, 0]) == fg)
    return "vertical", bool(int(interior[0, 0]) == fg)


def _read_templates(grid: np.ndarray, panel_bbox: tuple[int, int, int, int]) -> dict[str, bool]:
    r0, c0, r1, c1 = panel_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    grey_rows = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == GREY)]
    grey_cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == GREY)]

    boxes: list[np.ndarray]
    if grey_cols and not grey_rows:
        assert len(grey_cols) == 1, "panel should have one vertical divider"
        d = grey_cols[0]
        boxes = [inner[:, :d], inner[:, d + 1 :]]
    elif grey_rows and not grey_cols:
        assert len(grey_rows) == 1, "panel should have one horizontal divider"
        d = grey_rows[0]
        boxes = [inner[:d, :], inner[d + 1 :, :]]
    else:
        raise AssertionError("panel divider not found")

    info = dict(_template_info(box) for box in boxes)
    assert set(info) == {"horizontal", "vertical"}, "panel must provide both stripe orientations"
    return info


def _fill_room_stripes(
    out: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    anchor_color: int,
    stripe_kind: str,
    phase_colored_first: bool,
) -> None:
    r0, c0, r1, c1 = room_bbox
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape
    if stripe_kind == "horizontal":
        for rr in range(ih):
            interior[rr, :] = anchor_color if ((rr % 2) == 0) == phase_colored_first else BG
    else:
        for cc in range(iw):
            interior[:, cc] = anchor_color if ((cc % 2) == 0) == phase_colored_first else BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the stripe orientation selected by the direction of its external paired line.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick rectangular walls enclose empty rooms; each room contains exactly one colored anchor pixel.
    - A grey-framed reference panel sits in one corner and contains one horizontal-stripe template and one vertical-stripe template.
    - Some pairs of rooms are linked by monochrome single-pixel paths drawn outside the walls.

    Rule:
    - Detect the rooms, their anchor colors, and the corner reference panel.
    - For each external colored path, find the two room walls it touches and measure whether the path uses more horizontal or more vertical steps.
    - A horizontally dominated pair makes both rooms use vertical stripes; a vertically dominated pair makes both rooms use horizontal stripes.
    - Recolor the chosen room interiors with the anchor's own color on the colored stripe rows or columns and black on the alternating rows or columns; unpaired rooms stay unchanged.

    Key insight:
    - The colored external path identifies which rooms are paired, and the room fill direction is the stripe template perpendicular to that path's dominant direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_comps = _components(g == GREY)
    rooms: list[dict[str, object]] = []
    panel_bbox: tuple[int, int, int, int] | None = None

    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if _is_rect_frame(comp):
            inside = g[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = np.argwhere(inside != BG)
            if non_bg.shape[0] == 1:
                ar, ac = map(int, non_bg[0])
                color = int(inside[ar, ac])
                if color != GREY:
                    rooms.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "anchor_color": color,
                        }
                    )
                    continue
        touches_boundary = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if touches_boundary:
            panel_bbox = (r0, c0, r1, c1)

    assert panel_bbox is not None, "reference panel not found"
    assert rooms, "no rooms found"
    templates = _read_templates(g, panel_bbox)

    wall_to_room: dict[tuple[int, int], int] = {}
    forbidden = np.zeros_like(g, dtype=bool)
    for idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[index]
        forbidden[r0 : r1 + 1, c0 : c1 + 1] = True
        for cell in _rect_border_cells(r0, c0, r1, c1):
            wall_to_room[cell] = idx
    pr0, pc0, pr1, pc1 = panel_bbox
    forbidden[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    line_mask = (g != BG) & (g != GREY) & ~forbidden
    line_comps = _components(line_mask)

    paired_direction: dict[int, str] = {}
    for comp in line_comps:
        comp_set = set(comp)
        touched_rooms: set[int] = set()
        h_steps = 0
        v_steps = 0
        for r, c in comp:
            if (r, c + 1) in comp_set:
                h_steps += 1
            if (r + 1, c) in comp_set:
                v_steps += 1
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in wall_to_room:
                    touched_rooms.add(wall_to_room[(nr, nc)])

        assert len(touched_rooms) == 2, "each line must connect exactly two rooms"
        assert h_steps != v_steps, "pair direction must have a unique dominant axis"
        pair_dir = "horizontal" if h_steps > v_steps else "vertical"
        for room_idx in touched_rooms:
            assert room_idx not in paired_direction, "room may belong to at most one pair"
            paired_direction[room_idx] = pair_dir

    out = g.copy()
    for idx, room in enumerate(rooms):
        pair_dir = paired_direction.get(idx)
        if pair_dir is None:
            continue
        stripe_kind = "vertical" if pair_dir == "horizontal" else "horizontal"
        _fill_room_stripes(
            out,
            room["bbox"],  # type: ignore[arg-type]
            int(room["anchor_color"]),
            stripe_kind,
            templates[stripe_kind],
        )
    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment endpoints must align")


def _concat_segments(*segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for seg in segments:
        if not seg:
            continue
        if not path:
            path.extend(seg)
        else:
            path.extend(seg[1:])
    return path


def _horizontal_path(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    bend_c = rng.randint(sc + 1, ec - 1)
    return _concat_segments(
        _segment(start, (sr, bend_c)),
        _segment((sr, bend_c), (er, bend_c)),
        _segment((er, bend_c), end),
    )


def _vertical_path(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    bend_r = rng.randint(sr + 1, er - 1)
    return _concat_segments(
        _segment(start, (bend_r, sc)),
        _segment((bend_r, sc), (bend_r, ec)),
        _segment((bend_r, ec), end),
    )


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    oh: int,
    ow: int,
    anchor_color: int,
    rng: random.Random,
) -> dict[str, object]:
    r1, c1 = r0 + oh - 1, c0 + ow - 1
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    ar = rng.randint(r0 + 1, r1 - 1)
    ac = rng.randint(c0 + 1, c1 - 1)
    grid[ar, ac] = anchor_color
    return {"bbox": (r0, c0, r1, c1), "anchor_color": anchor_color}


def _draw_template_box(
    grid: np.ndarray,
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    kind: str,
    phase_colored_first: bool,
    border_color: int,
) -> None:
    bottom = top + box_h - 1
    right = left + box_w - 1
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    inner = grid[top + 1 : bottom, left + 1 : right]
    if kind == "horizontal":
        for r in range(inner.shape[0]):
            inner[r, :] = TEMPLATE_FG if ((r % 2) == 0) == phase_colored_first else BG
    else:
        for c in range(inner.shape[1]):
            inner[:, c] = TEMPLATE_FG if ((c % 2) == 0) == phase_colored_first else BG


def _draw_panel(
    grid: np.ndarray,
    box_h: int,
    box_w: int,
    arrangement: str,
    order: list[str],
    phases: dict[str, bool],
) -> tuple[int, int, int, int]:
    if arrangement == "side":
        panel_h, panel_w = box_h + 2, box_w * 2 + 3
    else:
        panel_h, panel_w = box_h * 2 + 3, box_w + 2

    r0 = 0
    c0 = 0
    r1 = panel_h - 1
    c1 = panel_w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if arrangement == "side":
        divider_c = 1 + box_w
        grid[1 : panel_h - 1, divider_c] = GREY
        positions = [(1, 1), (1, divider_c + 1)]
    else:
        divider_r = 1 + box_h
        grid[divider_r, 1 : panel_w - 1] = GREY
        positions = [(1, 1), (divider_r + 1, 1)]

    for idx, kind in enumerate(order):
        border_color = MARKER_A if idx == 0 else MARKER_B
        top, left = positions[idx]
        _draw_template_box(grid, top, left, box_h, box_w, kind, phases[kind], border_color)
    return (r0, c0, r1, c1)


def _bbox_overlaps(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in path:
        grid[r, c] = color
    return True


def _flip_to_corner(grid: np.ndarray, corner: str) -> np.ndarray:
    out = grid
    if "b" in corner:
        out = np.flipud(out)
    if "r" in corner:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(28, 30)
    w = rng.randint(28, 30)
    grid = np.full((h, w), BG, dtype=int)

    box_h = 4
    box_w = 4
    arrangement = rng.choice(["side", "stack"])
    order = ["horizontal", "vertical"]
    rng.shuffle(order)
    phases = {"horizontal": bool(rng.randint(0, 1)), "vertical": bool(rng.randint(0, 1))}
    panel_bbox = _draw_panel(grid, box_h, box_w, arrangement, order, phases)
    _, _, panel_r1, panel_c1 = panel_bbox
    panel_h = panel_r1 + 1
    panel_w = panel_c1 + 1

    colors = list(ANCHOR_COLORS)
    rng.shuffle(colors)
    room_count = 4 + rng.randint(0, 1)
    colors = colors[:room_count]

    sizes = [(5, 5) for _ in range(room_count)]
    (ha_h, ha_w), (hb_h, hb_w), (va_h, va_w), (vb_h, vb_w) = sizes[:4]

    max_h1_c0 = w - 26
    if max_h1_c0 < 1:
        return None
    h1_c0 = rng.randint(1, max_h1_c0)
    h2_c0 = rng.randint(h1_c0 + 12, w - 14)
    h1_r0 = rng.randint(h - 7, h - 6)
    h2_r0 = rng.randint(max(h - 7, h1_r0 - 1), h - 6)

    max_v1_r0 = h - 25
    if max_v1_r0 < 1:
        return None
    v1_r0 = rng.randint(1, max_v1_r0)
    v2_r0 = rng.randint(v1_r0 + 12, h - 13)
    v1_c0 = rng.randint(w - 8, w - 6)
    v2_c0 = rng.randint(max(w - 8, v1_c0 - 2), min(w - 6, v1_c0 + 2))

    room_specs = [
        (h1_r0, h1_c0, ha_h, ha_w, colors[0]),
        (h2_r0, h2_c0, hb_h, hb_w, colors[1]),
        (v1_r0, v1_c0, va_h, va_w, colors[2]),
        (v2_r0, v2_c0, vb_h, vb_w, colors[3]),
    ]

    if room_count == 5:
        u_h, u_w = sizes[4]
        u_r0_min = panel_h + 2
        u_r0_max = min(h // 2, h - u_h - 1)
        u_c0_min = panel_w + 2
        u_c0_max = min(w // 2 - u_w - 1, w - u_w - 1)
        if u_r0_min > u_r0_max or u_c0_min > u_c0_max:
            return None
        u_r0 = rng.randint(u_r0_min, u_r0_max)
        u_c0 = rng.randint(u_c0_min, u_c0_max)
        room_specs.append((u_r0, u_c0, u_h, u_w, colors[4]))

    panel_box = panel_bbox
    room_boxes = [(r0, c0, r0 + oh - 1, c0 + ow - 1) for r0, c0, oh, ow, _ in room_specs]
    for box in room_boxes:
        if _bbox_overlaps(box, panel_box, pad=1):
            return None
    for i in range(len(room_boxes)):
        for j in range(i + 1, len(room_boxes)):
            if _bbox_overlaps(room_boxes[i], room_boxes[j], pad=1):
                return None

    rooms = []
    for r0, c0, oh, ow, color in room_specs:
        if not np.all(grid[r0 : r0 + oh, c0 : c0 + ow] == BG):
            return None
        rooms.append(_draw_room(grid, r0, c0, oh, ow, color, rng))

    h_start = (rng.randint(h1_r0 + 1, h1_r0 + ha_h - 2), h1_c0 + ha_w)
    h_end = (rng.randint(h2_r0 + 1, h2_r0 + hb_h - 2), h2_c0 - 1)
    if h_end[1] - h_start[1] <= abs(h_end[0] - h_start[0]):
        return None
    h_path = _horizontal_path(h_start, h_end, rng)
    h_line_color = rng.choice([int(rooms[0]["anchor_color"]), int(rooms[1]["anchor_color"])])
    if not _paint_path(grid, h_path, h_line_color):
        return None

    v_start = (v1_r0 + va_h, rng.randint(v1_c0 + 1, v1_c0 + va_w - 2))
    v_end = (v2_r0 - 1, rng.randint(v2_c0 + 1, v2_c0 + vb_w - 2))
    if v_end[0] - v_start[0] <= abs(v_end[1] - v_start[1]):
        return None
    v_path = _vertical_path(v_start, v_end, rng)
    v_line_color = rng.choice([int(rooms[2]["anchor_color"]), int(rooms[3]["anchor_color"])])
    if not _paint_path(grid, v_path, v_line_color):
        return None

    corner = rng.choice(["tl", "tr", "bl", "br"])
    final_grid = _flip_to_corner(grid, corner)
    try:
        out = main(final_grid)
    except Exception:
        return None
    if out.shape != final_grid.shape:
        return None
    if np.array_equal(out, final_grid):
        return None
    return final_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-room input")
