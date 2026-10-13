from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BG = int(Color.BLACK)
SRC = int(Color.GREEN)
CTRL = int(Color.RED)
RAY = int(Color.GREY)
BLOCK = int(Color.TEAL)

GEN_ATTEMPTS = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_source_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color in {BG, CTRL, RAY, BLOCK}:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 4 or width < 4:
            continue
        border = _border_mask(height, width)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if (
            coords.shape[0] == int(border.sum())
            and bool(np.all(sub[border] == color))
            and bool(np.all(sub[~border] == BG))
        ):
            candidates.append((int(color), (int(r0), int(c0), int(r1), int(c1))))
    assert len(candidates) == 1, "expected exactly one visible source frame"
    return candidates[0]


def _all_black_windows(
    grid: np.ndarray, height: int, width: int
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for r0 in range(grid.shape[0] - int(height) + 1):
        for c0 in range(grid.shape[1] - int(width) + 1):
            if bool(np.all(grid[r0 : r0 + height, c0 : c0 + width] == BG)):
                windows.append((int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)))
    return windows


def _find_hidden_windows(
    grid: np.ndarray,
    height: int,
    width: int,
    source_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    hidden: list[tuple[int, int, int, int]] = []
    for r0 in range(grid.shape[0] - int(height) + 1):
        for c0 in range(grid.shape[1] - int(width) + 1):
            bbox = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
            if bbox == source_bbox:
                continue
            sub = grid[r0 : r0 + height, c0 : c0 + width]
            if bool(np.all(sub == BG)):
                hidden.append(bbox)
    return hidden


def _apply_color_transfer(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    frame_color, source_bbox = _find_source_frame(g)
    r0, c0, r1, c1 = source_bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    border = _border_mask(height, width)

    hidden = _find_hidden_windows(g, height, width, source_bbox)
    out = g.copy()
    for hr0, hc0, hr1, hc1 in hidden:
        window = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        window[border] = int(frame_color)
    return out, source_bbox, hidden


def _control_points(grid: np.ndarray) -> list[tuple[int, int, str]]:
    H, W = map(int, grid.shape)
    controls: list[tuple[int, int, str]] = []
    for r, c in np.argwhere(grid == CTRL):
        rr = int(r)
        cc = int(c)
        if rr == 0 and 0 < cc < W - 1:
            controls.append((rr, cc, "top"))
        elif rr == H - 1 and 0 < cc < W - 1:
            controls.append((rr, cc, "bottom"))
        elif cc == 0 and 0 < rr < H - 1:
            controls.append((rr, cc, "left"))
        elif cc == W - 1 and 0 < rr < H - 1:
            controls.append((rr, cc, "right"))
    assert 1 <= len(controls) <= 2, "expected one or two border controls"
    return controls


def _draw_rays(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    g = np.asarray(grid, dtype=int).copy()
    H, W = map(int, g.shape)
    controls = _control_points(g)

    for r, c, edge in controls:
        if edge == "top":
            for rr in range(1, H):
                if int(g[rr, c]) != CTRL:
                    g[rr, c] = RAY
        elif edge == "bottom":
            for rr in range(0, H - 1):
                if int(g[rr, c]) != CTRL:
                    g[rr, c] = RAY
        elif edge == "left":
            for cc in range(1, W):
                if int(g[r, cc]) != CTRL:
                    g[r, cc] = RAY
        else:
            for cc in range(0, W - 1):
                if int(g[r, cc]) != CTRL:
                    g[r, cc] = RAY

    for r, c, _edge in controls:
        g[r, c] = CTRL
    return g, controls


def _keep_partition_mask(grid_with_rays: np.ndarray, controls: list[tuple[int, int, str]]) -> np.ndarray:
    H, W = map(int, grid_with_rays.shape)
    passable = (grid_with_rays != RAY) & (grid_with_rays != CTRL)
    keep = np.zeros_like(passable, dtype=bool)
    control_edges = {edge for _r, _c, edge in controls}

    for comp in components_mask(passable, connectivity=4):
        edges: set[str] = set()
        if bool(np.any(comp[:, 0] == 0)):
            edges.add("top")
        if bool(np.any(comp[:, 0] == H - 1)):
            edges.add("bottom")
        if bool(np.any(comp[:, 1] == 0)):
            edges.add("left")
        if bool(np.any(comp[:, 1] == W - 1)):
            edges.add("right")
        if edges & control_edges:
            keep[comp[:, 0], comp[:, 1]] = True
    return keep


def _solve(grid: np.ndarray) -> np.ndarray:
    recolored, _source_bbox, _hidden = _apply_color_transfer(grid)
    cut, controls = _draw_rays(recolored)
    keep = _keep_partition_mask(cut, controls)

    out = np.full_like(cut, BG)
    out[keep] = cut[keep]
    out[cut == RAY] = RAY
    out[np.asarray(grid, dtype=int) == CTRL] = CTRL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal same-size hidden frames, then keep only the control-edge ray partition.

    Input structure:
    - The grid is mostly black.
    - One visible monochrome hollow rectangular frame supplies the template color and outer size.
    - One or two same-size rectangular windows are entirely black, so their frame borders are camouflaged; unrelated clutter may appear elsewhere.
    - One or two red control pixels sit on the outer border and define inward full-length rays.

    Rule:
    - Detect the unique visible hollow frame and read its color and outer dimensions.
    - Repaint the border of every same-size all-black window with that source color.
    - From each red border control, overwrite the whole inward row or column with grey until the opposite edge, leaving the control pixel red.
    - Treat the grey rays as cuts through the whole grid. Keep every non-ray cell in a region that touches at least one control edge, and clear every region that touches only non-control edges. Preserve the grey rays and red controls.

    Key insight:
    - The hidden frames can only be recovered before the cut: first transfer the template frame color onto the matching empty windows, then use the control-edge ray partition of the entire completed grid to decide what survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _boxes_separate(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    return (
        a[2] + int(pad) < b[0]
        or b[2] + int(pad) < a[0]
        or a[3] + int(pad) < b[1]
        or b[3] + int(pad) < a[1]
    )


def _control_config(height: int, width: int, frame_h: int, frame_w: int) -> dict | None:
    n_ctrl = random.choices([1, 2], weights=[2, 3], k=1)[0]
    v_lo = int(frame_w) + 1
    v_hi = int(width - frame_w - 2)
    h_lo = int(frame_h) + 1
    h_hi = int(height - frame_h - 2)
    if v_lo > v_hi or h_lo > h_hi:
        return None

    if n_ctrl == 1:
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge in {"top", "bottom"}:
            cut_c = random.randint(v_lo, v_hi)
            controls = [(0 if edge == "top" else height - 1, cut_c, edge)]
            return {
                "n_ctrl": 1,
                "vertical_edge": edge,
                "vertical_col": cut_c,
                "horizontal_edge": None,
                "horizontal_row": None,
                "controls": controls,
            }
        cut_r = random.randint(h_lo, h_hi)
        controls = [(cut_r, 0 if edge == "left" else width - 1, edge)]
        return {
            "n_ctrl": 1,
            "vertical_edge": None,
            "vertical_col": None,
            "horizontal_edge": edge,
            "horizontal_row": cut_r,
            "controls": controls,
        }

    cut_c = random.randint(v_lo, v_hi)
    cut_r = random.randint(h_lo, h_hi)
    v_edge = random.choice(["top", "bottom"])
    h_edge = random.choice(["left", "right"])
    controls = [
        (0 if v_edge == "top" else height - 1, cut_c, v_edge),
        (cut_r, 0 if h_edge == "left" else width - 1, h_edge),
    ]
    return {
        "n_ctrl": 2,
        "vertical_edge": v_edge,
        "vertical_col": cut_c,
        "horizontal_edge": h_edge,
        "horizontal_row": cut_r,
        "controls": controls,
    }


def _positions_for_category(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    config: dict,
    category: str,
) -> list[tuple[int, int]]:
    cut_r = config["horizontal_row"]
    cut_c = config["vertical_col"]
    positions: list[tuple[int, int]] = []
    for r0 in range(1, int(height - frame_h)):
        for c0 in range(1, int(width - frame_w)):
            r1 = int(r0 + frame_h - 1)
            c1 = int(c0 + frame_w - 1)
            ok = False
            if category == "TL":
                ok = cut_r is not None and cut_c is not None and r1 < cut_r and c1 < cut_c
            elif category == "TR":
                ok = cut_r is not None and cut_c is not None and r1 < cut_r and c0 > cut_c
            elif category == "BL":
                ok = cut_r is not None and cut_c is not None and r0 > cut_r and c1 < cut_c
            elif category == "BR":
                ok = cut_r is not None and cut_c is not None and r0 > cut_r and c0 > cut_c
            elif category == "L":
                ok = cut_c is not None and r0 >= 1 and c1 < cut_c
            elif category == "R":
                ok = cut_c is not None and r0 >= 1 and c0 > cut_c
            elif category == "T":
                ok = cut_r is not None and r1 < cut_r and c0 >= 1
            elif category == "B":
                ok = cut_r is not None and r0 > cut_r and c0 >= 1
            elif category == "VCROSS":
                ok = cut_c is not None and c0 < cut_c < c1
            elif category == "VCROSS_TOP":
                ok = cut_r is not None and cut_c is not None and r1 < cut_r and c0 < cut_c < c1
            elif category == "VCROSS_BOTTOM":
                ok = cut_r is not None and cut_c is not None and r0 > cut_r and c0 < cut_c < c1
            elif category == "HCROSS":
                ok = cut_r is not None and r0 < cut_r < r1
            elif category == "HCROSS_LEFT":
                ok = cut_r is not None and cut_c is not None and c1 < cut_c and r0 < cut_r < r1
            elif category == "HCROSS_RIGHT":
                ok = cut_r is not None and cut_c is not None and c0 > cut_c and r0 < cut_r < r1
            if ok:
                positions.append((int(r0), int(c0)))
    random.shuffle(positions)
    return positions


def _kept_and_dropped_categories(config: dict) -> tuple[str, str]:
    keep_row = "T" if config["vertical_edge"] == "top" else "B"
    keep_col = "L" if config["horizontal_edge"] == "left" else "R"
    drop_row = "B" if keep_row == "T" else "T"
    drop_col = "R" if keep_col == "L" else "L"
    return keep_row + keep_col, drop_row + drop_col


def _source_cross_category(config: dict) -> str | None:
    if config["n_ctrl"] == 1:
        return "VCROSS" if config["vertical_col"] is not None else "HCROSS"

    keep_row = "TOP" if config["vertical_edge"] == "top" else "BOTTOM"
    keep_col = "LEFT" if config["horizontal_edge"] == "left" else "RIGHT"
    options = [f"VCROSS_{keep_row}", f"HCROSS_{keep_col}"]
    random.shuffle(options)
    return options[0]


def _window_plan(config: dict, n_hidden: int, cross_frame: bool) -> list[str]:
    n_windows = 1 + int(n_hidden)
    if config["n_ctrl"] == 1:
        if config["vertical_col"] is not None:
            side_choices = ["L", "R"]
        else:
            side_choices = ["T", "B"]
        plan: list[str] = []
        plan.append(_source_cross_category(config) if cross_frame else random.choice(side_choices))
        if n_windows == 2:
            other = random.choice(side_choices)
            if other == plan[0]:
                other = side_choices[1 - side_choices.index(other)]
            plan.append(other)
            return plan
        if all(side not in plan for side in side_choices):
            plan.append(random.choice(side_choices))
        else:
            missing = [side for side in side_choices if side not in plan]
            plan.append(missing[0] if missing else random.choice(side_choices))
        while len(plan) < n_windows:
            plan.append(random.choice(side_choices))
        return plan

    kept, dropped = _kept_and_dropped_categories(config)
    if n_windows == 2:
        return [(_source_cross_category(config) if cross_frame else kept), dropped]
    return [(_source_cross_category(config) if cross_frame else kept), dropped, kept]


def _place_windows(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    config: dict,
    plan: list[str],
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    categories = list(plan)
    for category in categories:
        positions = _positions_for_category(height, width, frame_h, frame_w, config, category)
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            bbox = (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1))
            if all(_boxes_separate(bbox, other, pad=1) for other in boxes):
                boxes.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _false_windows_to_cover(
    grid: np.ndarray,
    frame_h: int,
    frame_w: int,
    true_boxes: set[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    return [bbox for bbox in _all_black_windows(grid, frame_h, frame_w) if bbox not in true_boxes]


def _place_blockers(
    grid: np.ndarray,
    frame_h: int,
    frame_w: int,
    true_boxes: list[tuple[int, int, int, int]],
    controls: list[tuple[int, int, str]],
) -> bool:
    true_box_set = set(true_boxes)
    reserved = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in true_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c, _edge in controls:
        reserved[r, c] = True

    false_windows = _false_windows_to_cover(grid, frame_h, frame_w, true_box_set)
    if not false_windows:
        return True

    cell_to_windows: dict[tuple[int, int], set[int]] = {}
    for idx, (r0, c0, r1, c1) in enumerate(false_windows):
        available = []
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if not reserved[rr, cc]:
                    available.append((int(rr), int(cc)))
                    cell_to_windows.setdefault((int(rr), int(cc)), set()).add(idx)
        if not available:
            return False

    uncovered = set(range(len(false_windows)))
    blockers: list[tuple[int, int]] = []
    while uncovered:
        best_cover = 0
        best_cells: list[tuple[int, int]] = []
        for cell, covered in cell_to_windows.items():
            score = len(covered & uncovered)
            if score > best_cover:
                best_cover = score
                best_cells = [cell]
            elif score == best_cover and score > 0:
                best_cells.append(cell)
        if best_cover == 0:
            return False
        choice = random.choice(best_cells)
        blockers.append(choice)
        uncovered -= cell_to_windows[choice]
        cell_to_windows.pop(choice, None)

    if len(blockers) > 18:
        return False
    for r, c in blockers:
        grid[r, c] = BLOCK
    expected_hidden = set(true_boxes[1:])
    return set(_all_black_windows(grid, frame_h, frame_w)) == expected_hidden


def _frame_has_green(out: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.any(out[r0 : r1 + 1, c0 : c1 + 1] == SRC))


def _frame_has_grey(out: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.any(out[r0 : r1 + 1, c0 : c1 + 1] == RAY))


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    hidden_boxes: list[tuple[int, int, int, int]],
    config: dict,
    cross_frame: bool,
) -> bool:
    try:
        _color, found_source = _find_source_frame(inp)
    except AssertionError:
        return False
    if found_source != source_bbox:
        return False

    sh = int(source_bbox[2] - source_bbox[0] + 1)
    sw = int(source_bbox[3] - source_bbox[1] + 1)
    all_black = set(_all_black_windows(inp, sh, sw))
    expected_black = set(hidden_boxes)
    if all_black != expected_black:
        return False

    hidden_found = set(_find_hidden_windows(inp, sh, sw, source_bbox))
    if hidden_found != set(hidden_boxes):
        return False

    if not any(_frame_has_green(out, bbox) for bbox in [source_bbox, *hidden_boxes]):
        return False

    if config["n_ctrl"] == 2:
        if not any(not _frame_has_green(out, bbox) for bbox in hidden_boxes):
            return False

    if cross_frame and not any(_frame_has_grey(out, bbox) for bbox in [source_bbox, *hidden_boxes]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 19)
    width = random.randint(15, 19)
    frame_h = random.randint(4, 6)
    frame_w = random.randint(4, 6)
    n_hidden = random.choice([1, 2])
    cross_frame = bool(random.random() < 0.55)

    config = _control_config(height, width, frame_h, frame_w)
    if config is None:
        return None
    plan = _window_plan(config, n_hidden, cross_frame)

    for _ in range(60):
        boxes = _place_windows(height, width, frame_h, frame_w, config, plan)
        if boxes is None:
            continue

        source_bbox = boxes[0]
        hidden_boxes = boxes[1:]
        grid = np.full((height, width), BG, dtype=int)

        sr0, sc0, sr1, sc1 = source_bbox
        border = _border_mask(frame_h, frame_w)
        source_window = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
        source_window[:, :] = BG
        source_window[border] = SRC

        for r, c, _edge in config["controls"]:
            grid[int(r), int(c)] = CTRL

        if not _place_blockers(grid, frame_h, frame_w, boxes, config["controls"]):
            continue

        out = _solve(grid)
        if not _semantic_checks(grid, out, source_bbox, hidden_boxes, config, cross_frame):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle after repeated attempts")
