from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_for_fable_6",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
crop_bbox = _ARC_UTILS.crop_bbox


BG = 0
WATER = 4
WALL_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _parse_container(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    patch = crop_bbox(grid, (r0, c0, r1, c1))
    wall = patch == color
    if np.any(wall[1:-1, 1:-1]):
        return None

    top_open = [j for j in range(1, w - 1) if int(patch[0, j]) != color]
    left_open = [i for i in range(1, h - 1) if int(patch[i, 0]) != color]
    right_open = [i for i in range(1, h - 1) if int(patch[i, w - 1]) != color]
    if len(top_open) > 1 or len(left_open) + len(right_open) > 1:
        return None
    if np.any(patch[-1, :] != color):
        return None
    if int(patch[0, 0]) != color or int(patch[0, -1]) != color:
        return None
    if int(patch[-1, 0]) != color or int(patch[-1, -1]) != color:
        return None

    for j in range(1, w - 1):
        v = int(patch[0, j])
        if j in top_open:
            if v != bg:
                return None
        elif v != color:
            return None
    for i in range(1, h - 1):
        lv = int(patch[i, 0])
        rv = int(patch[i, -1])
        if i in left_open:
            if lv != bg:
                return None
        elif lv != color:
            return None
        if i in right_open:
            if rv != bg:
                return None
        elif rv != color:
            return None

    interior = patch[1:-1, 1:-1]
    if np.any((interior != bg) & (interior != WATER)):
        return None

    inner_h, inner_w = interior.shape
    row_fill = np.sum(interior == WATER, axis=1)
    if np.any((row_fill != 0) & (row_fill != inner_w)):
        return None
    water_rows = 0
    while water_rows < inner_h and int(row_fill[inner_h - 1 - water_rows]) == inner_w:
        water_rows += 1
    if np.any(row_fill[: inner_h - water_rows] != 0):
        return None

    side = None
    notch_row_abs = None
    retain_rows = inner_h
    exit_col = None
    if left_open:
        side = "left"
        notch_rel = int(left_open[0])
        notch_row_abs = r0 + notch_rel
        retain_rows = inner_h - notch_rel
        exit_col = c0 - 1
    elif right_open:
        side = "right"
        notch_rel = int(right_open[0])
        notch_row_abs = r0 + notch_rel
        retain_rows = inner_h - notch_rel
        exit_col = c1 + 1

    top_open_abs = None if not top_open else c0 + int(top_open[0])
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "coords": np.asarray(component["coords"], dtype=int),
        "inner_h": inner_h,
        "inner_w": inner_w,
        "water_rows": int(water_rows),
        "top_open_abs": top_open_abs,
        "top_open_rel": None if not top_open else int(top_open[0]),
        "side": side,
        "notch_row_abs": notch_row_abs,
        "retain_rows": int(retain_rows),
        "exit_col": exit_col,
    }


def _extract_containers(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], np.ndarray]:
    bg = int(get_bg(grid))
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, WATER)]
    containers: list[dict] = []
    top_open_map: dict[tuple[int, int], int] = {}
    wall_mask = np.zeros(grid.shape, dtype=bool)

    for color in colors:
        for component in find_components(
            grid,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
        ):
            info = _parse_container(grid, component, bg)
            if info is None:
                continue
            cid = len(containers)
            info["id"] = cid
            containers.append(info)
            wall_mask[info["coords"][:, 0], info["coords"][:, 1]] = True
            if info["top_open_abs"] is not None:
                key = (info["bbox"][0], int(info["top_open_abs"]))
                assert key not in top_open_map
                top_open_map[key] = cid

    assert containers, "expected at least one container"
    return containers, top_open_map, wall_mask


def _settle(grid: np.ndarray, *, want_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert bg == BG

    containers, top_open_map, occupied = _extract_containers(g)
    rows = [int(c["water_rows"]) for c in containers]
    initial_rows = list(rows)
    received_any = [False] * len(containers)
    queue: deque[dict] = deque()

    trace = {
        "containers": containers,
        "initial_rows": initial_rows,
        "entered": [],
        "initial_sources": [],
        "cascade_sources": [],
        "pile_events": [],
    }

    def emit_overflow(cid: int, reason: str) -> None:
        container = containers[cid]
        if container["side"] is None:
            return
        retain_rows = int(container["retain_rows"])
        if rows[cid] <= retain_rows:
            return
        amount = (rows[cid] - retain_rows) * int(container["inner_w"])
        rows[cid] = retain_rows
        queue.append(
            {
                "source": cid,
                "start_row": int(container["notch_row_abs"]),
                "col": int(container["exit_col"]),
                "amount": int(amount),
            }
        )
        if reason == "initial":
            trace["initial_sources"].append(cid)
        elif received_any[cid]:
            trace["cascade_sources"].append(cid)

    order = sorted(range(len(containers)), key=lambda cid: containers[cid]["bbox"][:2])
    for cid in order:
        emit_overflow(cid, "initial")

    while queue:
        event = queue.popleft()
        start_row = int(event["start_row"])
        col = int(event["col"])
        amount = int(event["amount"])
        assert 0 <= col < w
        support = None
        dest_cid = None

        for r in range(start_row + 1, h):
            if occupied[r, col]:
                support = r - 1
                break
            dest_cid = top_open_map.get((r, col))
            if dest_cid is not None:
                break

        if dest_cid is not None:
            container = containers[int(dest_cid)]
            total = rows[dest_cid] * int(container["inner_w"]) + amount
            assert total % int(container["inner_w"]) == 0
            new_rows = total // int(container["inner_w"])
            assert new_rows <= int(container["inner_h"])
            rows[dest_cid] = int(new_rows)
            received_any[dest_cid] = True
            trace["entered"].append({"cid": int(dest_cid), "amount": amount})
            emit_overflow(int(dest_cid), "received")
            continue

        if support is None:
            support = h - 1
        top = support - amount + 1
        assert top >= 0
        assert not np.any(occupied[top : support + 1, col])
        occupied[top : support + 1, col] = True
        trace["pile_events"].append(
            {"col": col, "top": top, "support": support, "amount": amount}
        )

    out = g.copy()
    out[out == WATER] = bg
    for container, water_rows in zip(containers, rows):
        r0, c0, r1, c1 = map(int, container["bbox"])
        if water_rows <= 0:
            continue
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[-water_rows:, :] = WATER
    for pile in trace["pile_events"]:
        out[int(pile["top"]) : int(pile["support"]) + 1, int(pile["col"])] = WATER

    if not want_trace:
        return out
    trace["final_rows"] = list(rows)
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle cascading leaks through a vertical chain of rectangular containers.

    Input structure:
    - Black background with several monochrome rectangular containers whose interiors may hold yellow water in full bottom rows.
    - A container may have one 1-cell top opening, one 1-cell side-wall notch, both, or neither.
    - Top openings line up with possible falling columns; side notches sit on the left or right wall.

    Rule:
    - Any container with a side notch leaks until only the rows strictly below the notch remain; the lost water count is conserved.
    - The leaked water exits one cell horizontally beyond the notch and falls straight down that column.
    - If the falling column first reaches a container's top opening, all leaked water enters that container, refills it from the inner bottom in whole rows, and may trigger another leak from that container's own side notch; otherwise the water stacks as a vertical exterior column on the first support below.

    Key insight:
    - The notch row is the hidden waterline, while each top opening is a receiver; solving any one container locally is insufficient because conserved water can refill lower containers and trigger further overflow.
    """
    return _settle(grid, want_trace=False)


def _make_container(
    *,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    color: int,
    water_rows: int,
    top_open_rel: int | None,
    side: str | None,
    retain_rows: int | None,
) -> dict:
    assert 0 <= water_rows <= inner_h
    if side is None:
        notch_rel = None
    else:
        assert retain_rows is not None and 0 <= retain_rows < inner_h
        notch_rel = inner_h - int(retain_rows)
        assert 1 <= notch_rel <= inner_h
    if top_open_rel is not None:
        assert 1 <= top_open_rel <= inner_w
    return {
        "r0": int(r0),
        "c0": int(c0),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "color": int(color),
        "water_rows": int(water_rows),
        "top_open_rel": top_open_rel,
        "side": side,
        "retain_rows": retain_rows,
        "notch_rel": notch_rel,
    }


def _draw_container(grid: np.ndarray, spec: dict) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    color = int(spec["color"])
    h = inner_h + 2
    w = inner_w + 2
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if spec["top_open_rel"] is not None:
        grid[r0, c0 + int(spec["top_open_rel"])] = BG
    if spec["side"] == "left":
        grid[r0 + int(spec["notch_rel"]), c0] = BG
    elif spec["side"] == "right":
        grid[r0 + int(spec["notch_rel"]), c1] = BG

    water_rows = int(spec["water_rows"])
    if water_rows > 0:
        grid[r1 - water_rows : r1, c0 + 1 : c1] = WATER


def _main_chain_specs() -> list[dict] | None:
    inner_w = random.randint(2, 4)
    spill_count = random.choice([2, 3])
    if spill_count == 2:
        overflow_rows = [random.randint(2, 3), 1]
    else:
        overflow_rows = [3, 2, 1]

    specs: list[dict] = []
    row = 0
    col = 0
    source_side = random.choice(["left", "right"])
    if spill_count == 2:
        spill_sides = [source_side, random.choice(["left", "right"])]
    else:
        flip = "left" if source_side == "right" else "right"
        spill_sides = [source_side, flip, source_side]
    pending_top_open = None
    for idx in range(spill_count + 1):
        is_spill = idx < spill_count
        if idx == 0:
            inner_h_min = overflow_rows[0] + 1
        elif is_spill:
            inner_h_min = max(3, overflow_rows[idx - 1])
        else:
            inner_h_min = 3
        inner_h = random.randint(inner_h_min, 5)
        if idx == 0:
            retain_rows = random.randint(1, inner_h - overflow_rows[0])
            water_rows = retain_rows + overflow_rows[0]
            side = source_side
            top_open_rel = None
        elif is_spill:
            delta = overflow_rows[idx - 1] - overflow_rows[idx]
            retain_rows = random.randint(max(1, delta), inner_h - overflow_rows[idx])
            water_rows = retain_rows - delta
            top_open_rel = int(pending_top_open)
            side = spill_sides[idx]
        else:
            retain_rows = None
            water_rows = random.randint(0, inner_h - overflow_rows[-1])
            top_open_rel = int(pending_top_open)
            side = None

        specs.append(
            _make_container(
                r0=row,
                c0=col,
                inner_h=inner_h,
                inner_w=inner_w,
                color=-1,
                water_rows=water_rows,
                top_open_rel=top_open_rel,
                side=side,
                retain_rows=retain_rows,
            )
        )

        if not is_spill:
            break
        outer_w = inner_w + 2
        fall_col = col - 1 if side == "left" else col + outer_w
        pending_top_open = random.randint(1, inner_w)
        next_col = fall_col - pending_top_open
        row += inner_h + 2 + random.randint(2, 3)
        col = next_col
        if row > 23:
            return None
    return specs


def _roof_pile_specs(start_col: int) -> tuple[list[dict], dict] | None:
    src_inner_w = random.randint(2, 3)
    src_inner_h = random.randint(3, 4)
    spill_rows = random.randint(1, 2)
    src_retain = random.randint(1, src_inner_h - spill_rows)
    src_water = src_retain + spill_rows
    src = _make_container(
        r0=random.randint(1, 3),
        c0=start_col,
        inner_h=src_inner_h,
        inner_w=src_inner_w,
        color=-1,
        water_rows=src_water,
        top_open_rel=None,
        side="right",
        retain_rows=src_retain,
    )
    fall_col = start_col + src_inner_w + 2

    roof_inner_w = random.randint(3, 5)
    roof_inner_h = random.randint(3, 5)
    top_span_start = fall_col - random.randint(0, roof_inner_w + 1)
    roof = _make_container(
        r0=src["r0"] + src_inner_h + 2 + random.randint(4, 7),
        c0=top_span_start,
        inner_h=roof_inner_h,
        inner_w=roof_inner_w,
        color=-1,
        water_rows=random.randint(0, roof_inner_h - 1),
        top_open_rel=None,
        side=None,
        retain_rows=None,
    )
    if not (roof["c0"] <= fall_col <= roof["c0"] + roof_inner_w + 1):
        return None
    pile_height = spill_rows * src_inner_w
    support = roof["r0"] - 1
    if support - pile_height + 1 < 0:
        return None
    return [src, roof], {"fall_col": fall_col, "pile_height": pile_height}


def _embed_specs(specs: list[dict]) -> np.ndarray | None:
    all_r0 = min(int(s["r0"]) for s in specs)
    all_c0 = min(int(s["c0"]) for s in specs)
    all_r1 = max(int(s["r0"]) + int(s["inner_h"]) + 1 for s in specs)
    all_c1 = max(int(s["c0"]) + int(s["inner_w"]) + 1 for s in specs)

    box_h = all_r1 - all_r0 + 1
    box_w = all_c1 - all_c0 + 1
    if box_h > 26 or box_w > 26:
        return None

    pad_top = random.randint(1, 30 - box_h - 1)
    pad_left = random.randint(1, 30 - box_w - 1)
    height = box_h + pad_top + random.randint(1, 30 - box_h - pad_top)
    width = box_w + pad_left + random.randint(1, 30 - box_w - pad_left)
    if height > 30 or width > 30:
        return None

    shift_r = pad_top - all_r0
    shift_c = pad_left - all_c0
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        spec["r0"] += shift_r
        spec["c0"] += shift_c
        _draw_container(grid, spec)
    return grid


def _validate_generated_case(grid: np.ndarray) -> None:
    out, trace = _settle(grid, want_trace=True)
    assert out.shape[0] <= 30 and out.shape[1] <= 30
    assert np.any(grid != out)
    assert len(trace["entered"]) >= 2
    assert len(set(trace["cascade_sources"])) >= 1
    assert any(evt["support"] < grid.shape[0] - 1 for evt in trace["pile_events"])
    containers = trace["containers"]
    initial_rows = trace["initial_rows"]
    final_rows = trace["final_rows"]
    assert any(
        c["side"] is None and c["top_open_abs"] is None and initial_rows[i] == final_rows[i]
        for i, c in enumerate(containers)
    )
    assert any(
        c["side"] is None and c["top_open_abs"] is not None and final_rows[i] > initial_rows[i]
        for i, c in enumerate(containers)
    )


def _gen_attempt() -> np.ndarray | None:
    colors = WALL_COLORS[:]
    random.shuffle(colors)

    chain_specs = _main_chain_specs()
    if chain_specs is None:
        return None
    chain_right = max(int(s["c0"]) + int(s["inner_w"]) + 1 for s in chain_specs)
    roof_pack = _roof_pile_specs(chain_right + random.randint(5, 8))
    if roof_pack is None:
        return None
    roof_specs, _roof_meta = roof_pack

    specs = chain_specs + roof_specs
    if len(specs) > len(colors):
        return None
    for spec, color in zip(specs, colors):
        spec["color"] = color

    grid = _embed_specs(specs)
    if grid is None:
        return None

    try:
        _validate_generated_case(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
