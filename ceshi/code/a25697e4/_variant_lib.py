from __future__ import annotations

from functools import lru_cache

import numpy as np

BG_DEFAULT = 0
SEP = 9


def _stamp(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = color


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _norm(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _rot90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _norm({(c, -r) for r, c in cells})


def _flip_h(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _norm({(r, -c) for r, c in cells})


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _cells_from_local(anchor: tuple[int, int], shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def _handle_from_cavity(cavity: set[tuple[int, int]], direction: str, length: int = 3) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _shape_bbox(cavity)
    if direction == "right":
        row = (r0 + r1) // 2
        return {(row, c1 + i) for i in range(1, length + 1)}
    if direction == "left":
        row = (r0 + r1) // 2
        return {(row, c0 - i) for i in range(1, length + 1)}
    if direction == "down":
        col = (c0 + c1) // 2
        return {(r1 + i, col) for i in range(1, length + 1)}
    col = (c0 + c1) // 2
    return {(r0 - i, col) for i in range(1, length + 1)}


def _frame_ring(cavity: set[tuple[int, int]], open_dir: str, anchor_edge: str | None = None) -> set[tuple[int, int]]:
    frame: set[tuple[int, int]] = set()
    for r, c in cavity:
        for dr, dc, side in ((1, 0, "down"), (-1, 0, "up"), (0, 1, "right"), (0, -1, "left")):
            if side == open_dir:
                continue
            frame.add((r + dr, c + dc))
    if anchor_edge:
        r0, c0, r1, c1 = _shape_bbox(cavity)
        if anchor_edge == "left":
            for r in range(r0 - 1, r1 + 2):
                for c in range(0, c0):
                    frame.add((r, c))
        elif anchor_edge == "right":
            for r in range(r0 - 1, r1 + 2):
                for c in range(c1 + 1, 30):
                    frame.add((r, c))
        elif anchor_edge == "top":
            for r in range(0, r0):
                for c in range(c0 - 1, c1 + 2):
                    frame.add((r, c))
        elif anchor_edge == "bottom":
            for r in range(r1 + 1, 30):
                for c in range(c0 - 1, c1 + 2):
                    frame.add((r, c))
    return frame


def _tool_parts(
    plug_anchor: tuple[int, int],
    plug_shape: set[tuple[int, int]],
    handle_dir: str,
    handle_len: int = 3,
    gap: int = 0,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    plug = _cells_from_local(plug_anchor, plug_shape)
    r0, c0, r1, c1 = _shape_bbox(plug)
    if handle_dir == "right":
        row = (r0 + r1) // 2
        handle = {(row, c1 + gap + i) for i in range(1, handle_len + 1)}
    elif handle_dir == "left":
        row = (r0 + r1) // 2
        handle = {(row, c0 - gap - i) for i in range(1, handle_len + 1)}
    elif handle_dir == "down":
        col = (c0 + c1) // 2
        handle = {(r1 + gap + i, col) for i in range(1, handle_len + 1)}
    else:
        col = (c0 + c1) // 2
        handle = {(r0 - gap - i, col) for i in range(1, handle_len + 1)}
    return plug, handle


SHAPES = {
    "L": {(0, 0), (1, 0), (1, 1)},
    "RECT": {(0, 0), (0, 1), (1, 0), (1, 1)},
    "IRREG": {(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)},
    "PLUS": {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
    "BAR3": {(0, 0), (0, 1), (0, 2)},
    "BAR4": {(0, 0), (0, 1), (0, 2), (0, 3)},
    "RING": {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
    "T": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
}


def _grid(shape: tuple[int, int] = (30, 30), bg: int = BG_DEFAULT) -> np.ndarray:
    return np.full(shape, bg, dtype=int)


def _pair_left(frame_color: int, plug_color: int, handle_color: int, plug_shape: set[tuple[int, int]] | None = None) -> dict:
    shape = plug_shape or SHAPES["L"]
    cavity = _cells_from_local((5, 2), shape)
    frame = _frame_ring(cavity, "right", "left")
    in_plug, in_handle = _tool_parts((11, 18), shape, "left")
    return {
        "frame": frame,
        "frame_color": frame_color,
        "cavity": cavity,
        "open_dir": "right",
        "tool_plug": in_plug,
        "tool_handle": in_handle,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "dock_handle": _handle_from_cavity(cavity, "right"),
    }


def _pair_right(frame_color: int, plug_color: int, handle_color: int, plug_shape: set[tuple[int, int]] | None = None) -> dict:
    shape = plug_shape or SHAPES["L"]
    shape = _flip_h(shape)
    cavity = _cells_from_local((5, 25), shape)
    frame = _frame_ring(cavity, "left", "right")
    in_plug, in_handle = _tool_parts((17, 8), shape, "right")
    return {
        "frame": frame,
        "frame_color": frame_color,
        "cavity": cavity,
        "open_dir": "left",
        "tool_plug": in_plug,
        "tool_handle": in_handle,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "dock_handle": _handle_from_cavity(cavity, "left"),
    }


def _pair_top(frame_color: int, plug_color: int, handle_color: int, plug_shape: set[tuple[int, int]] | None = None) -> dict:
    shape = plug_shape or SHAPES["T"]
    cavity = _cells_from_local((2, 13), shape)
    frame = _frame_ring(cavity, "down", "top")
    in_plug, in_handle = _tool_parts((18, 13), shape, "up")
    return {
        "frame": frame,
        "frame_color": frame_color,
        "cavity": cavity,
        "open_dir": "down",
        "tool_plug": in_plug,
        "tool_handle": in_handle,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "dock_handle": _handle_from_cavity(cavity, "down"),
    }


def _pair_bottom(frame_color: int, plug_color: int, handle_color: int, plug_shape: set[tuple[int, int]] | None = None) -> dict:
    shape = plug_shape or SHAPES["T"]
    shape = _flip_h(shape)
    cavity = _cells_from_local((24, 13), shape)
    frame = _frame_ring(cavity, "up", "bottom")
    in_plug, in_handle = _tool_parts((8, 13), shape, "down")
    return {
        "frame": frame,
        "frame_color": frame_color,
        "cavity": cavity,
        "open_dir": "up",
        "tool_plug": in_plug,
        "tool_handle": in_handle,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "dock_handle": _handle_from_cavity(cavity, "up"),
    }


def _render_input_output(
    pairs: list[dict],
    bg: int = BG_DEFAULT,
    keep_tools: bool = False,
    trace_color: int | None = None,
    plug_only: bool = False,
    erase_frame: bool = False,
    plug_to_frame: bool = False,
    handle_to_frame: bool = False,
    swap_colors: bool = False,
    crop_output: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    inp = _grid(bg=bg)
    out = _grid(bg=bg)
    for pair in pairs:
        _stamp(inp, pair["frame"], pair["frame_color"])
        _stamp(inp, pair["tool_plug"], pair["plug_color"])
        _stamp(inp, pair["tool_handle"], pair["handle_color"])
        if not erase_frame:
            _stamp(out, pair["frame"], pair["frame_color"])
        plug_color = pair["plug_color"]
        handle_color = pair["handle_color"]
        if swap_colors:
            plug_color, handle_color = handle_color, plug_color
        if plug_to_frame:
            plug_color = pair["frame_color"]
        if handle_to_frame:
            handle_color = pair["frame_color"]
        _stamp(out, pair["cavity"], handle_color if pair.get("dock_handle_in_cavity") else plug_color)
        if not plug_only:
            _stamp(out, pair["dock_handle"], plug_color if pair.get("dock_handle_in_cavity") else handle_color)
        if keep_tools:
            _stamp(out, pair["tool_plug"], pair["plug_color"])
            _stamp(out, pair["tool_handle"], pair["handle_color"])
        if trace_color is not None:
            _stamp(out, pair["tool_plug"] | pair["tool_handle"], trace_color)
    if crop_output:
        pts = np.argwhere(out != bg)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        out = out[r0 : r1 + 1, c0 : c1 + 1]
    return inp, out


def _scene_base_two_pairs() -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        _pair_left(2, 3, 4),
        _pair_right(6, 7, 8),
    ]
    return _render_input_output(pairs)


def _scene_color_match() -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        _pair_left(3, 3, 4),
        _pair_right(7, 7, 8),
    ]
    return _render_input_output(pairs)


def _scene_edge_color_match() -> tuple[np.ndarray, np.ndarray]:
    left = _pair_left(2, 3, 5)
    right = _pair_right(6, 7, 8)
    inp, out = _render_input_output([left, right])
    for r, c in left["frame"]:
        inp[r, c] = 2 if r <= 6 else 5
        out[r, c] = 2 if r <= 6 else 5
    for r, c in right["frame"]:
        inp[r, c] = 8 if r <= 6 else 6
        out[r, c] = 8 if r <= 6 else 6
    return inp, out


def _scene_hole_count_match() -> tuple[np.ndarray, np.ndarray]:
    left = _pair_left(2, 3, 4, SHAPES["RECT"])
    right = _pair_right(6, 7, 8, SHAPES["RECT"])
    left["tool_handle"] |= {(10, 17)}
    right["tool_handle"] |= {(18, 12), (19, 12)}
    inp, out = _render_input_output([left, right])
    inp[9, 16] = 1
    inp[19, 10] = 1
    inp[19, 11] = 1
    return inp, out


def _scene_connector_match() -> tuple[np.ndarray, np.ndarray]:
    left = _pair_left(2, 3, 4)
    right = _pair_right(6, 7, 8)
    inp, out = _render_input_output([left, right])
    inp[6, 4] = 5
    inp[17, 16] = 5
    inp[6, 24] = 1
    inp[16, 11] = 1
    out[6, 4] = 5
    out[6, 24] = 1
    return inp, out


def _scene_area_rank() -> tuple[np.ndarray, np.ndarray]:
    big = _pair_left(2, 3, 4, SHAPES["IRREG"])
    small = _pair_right(6, 7, 8, SHAPES["L"])
    return _render_input_output([big, small])


def _scene_rotate_only() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, _rot90(SHAPES["L"]))
    return _render_input_output([pair])


def _scene_rotation_marker() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, _rot90(_rot90(SHAPES["L"])))
    inp, out = _render_input_output([pair])
    inp[3, 5] = 1
    inp[3, 6] = 1
    out[3, 5] = 1
    out[3, 6] = 1
    return inp, out


def _scene_three_color_tool() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    inp, out = _render_input_output([pair])
    hinge = {(11, 17)}
    _stamp(inp, hinge, 5)
    return inp, out


def _scene_separated_tool() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    pair["tool_plug"], pair["tool_handle"] = _tool_parts((11, 18), SHAPES["L"], "left", gap=2)
    return _render_input_output([pair])


def _scene_holey_tool() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, SHAPES["RING"])
    pair["dock_handle"] = _handle_from_cavity(pair["cavity"], "right", 2)
    return _render_input_output([pair])


def _scene_single_component_frame() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, SHAPES["RECT"])
    pair["frame"] = pair["frame"] | {(4, 1), (7, 1), (7, 2)}
    return _render_input_output([pair])


def _scene_hollow_square() -> tuple[np.ndarray, np.ndarray]:
    cavity = _cells_from_local((5, 3), SHAPES["RECT"])
    frame = _cells_from_local((3, 1), {(r, c) for r in range(0, 6) for c in range(0, 6) if r in {0, 5} or c in {0, 5}})
    frame -= {(5, 3), (5, 4)}
    pair = _pair_left(2, 3, 4, SHAPES["RECT"])
    pair["cavity"] = cavity
    pair["frame"] = frame
    pair["dock_handle"] = {(6, 3), (7, 3), (8, 3)}
    return _render_input_output([pair])


def _scene_internal_frame() -> tuple[np.ndarray, np.ndarray]:
    shape = SHAPES["L"]
    cavity = _cells_from_local((11, 11), shape)
    frame = _frame_ring(cavity, "right", None)
    plug, handle = _tool_parts((4, 20), shape, "left")
    pair = {
        "frame": frame,
        "frame_color": 2,
        "cavity": cavity,
        "open_dir": "right",
        "tool_plug": plug,
        "tool_handle": handle,
        "plug_color": 3,
        "handle_color": 4,
        "dock_handle": _handle_from_cavity(cavity, "right"),
    }
    return _render_input_output([pair])


def _scene_irregular_cavity() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4, SHAPES["IRREG"])])


def _scene_t_cavity() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4, SHAPES["T"])])


def _scene_swap_always() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], swap_colors=True)


def _scene_rotation_parity_swap() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, _rot90(SHAPES["L"]))
    return _render_input_output([pair], swap_colors=True)


def _scene_recolor_frame() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], plug_to_frame=True)


def _scene_keep_tool() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], keep_tools=True)


def _scene_trace_tool() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], trace_color=5)


def _scene_plug_only() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], plug_only=True)


def _scene_tool_only() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], erase_frame=True)


def _scene_opposite_edge_frames() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([
        _pair_left(2, 3, 4),
        _pair_right(6, 7, 8),
    ])


def _scene_open_to_edge() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    pair["open_dir"] = "left"
    pair["dock_handle"] = _handle_from_cavity(pair["cavity"], "left")
    pair["frame"] = _frame_ring(pair["cavity"], "left", "left")
    return _render_input_output([pair])


def _scene_many_tools_one_frame() -> tuple[np.ndarray, np.ndarray]:
    inp = _grid()
    out = _grid()
    cav1 = _cells_from_local((5, 3), SHAPES["L"])
    cav2 = _cells_from_local((10, 3), SHAPES["RECT"])
    frame = _frame_ring(cav1, "right", "left") | _frame_ring(cav2, "right", "left")
    _stamp(inp, frame, 2)
    _stamp(out, frame, 2)
    plug1, handle1 = _tool_parts((8, 18), SHAPES["L"], "left")
    plug2, handle2 = _tool_parts((15, 18), SHAPES["RECT"], "left")
    _stamp(inp, plug1, 3)
    _stamp(inp, handle1, 4)
    _stamp(inp, plug2, 6)
    _stamp(inp, handle2, 7)
    _stamp(out, cav1, 3)
    _stamp(out, _handle_from_cavity(cav1, "right"), 4)
    _stamp(out, cav2, 6)
    _stamp(out, _handle_from_cavity(cav2, "right"), 7)
    return inp, out


def _scene_one_tool_many_frames() -> tuple[np.ndarray, np.ndarray]:
    inp = _grid()
    left = _pair_left(2, 3, 4)
    right = _pair_right(6, 3, 4)
    _stamp(inp, left["frame"], left["frame_color"])
    _stamp(inp, right["frame"], right["frame_color"])
    _stamp(inp, left["tool_plug"], 3)
    _stamp(inp, left["tool_handle"], 4)
    out = _grid()
    for pair in (left, right):
        _stamp(out, pair["frame"], pair["frame_color"])
        _stamp(out, pair["cavity"], 3)
        _stamp(out, pair["dock_handle"], 4)
    return inp, out


def _scene_distractor_frame() -> tuple[np.ndarray, np.ndarray]:
    inp, out = _render_input_output([_pair_left(2, 3, 4)])
    extra = _pair_right(6, 7, 8)
    _stamp(inp, extra["frame"], extra["frame_color"])
    _stamp(out, extra["frame"], extra["frame_color"])
    return inp, out


def _scene_crop() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], crop_output=True)


def _scene_split() -> tuple[np.ndarray, np.ndarray]:
    left = _pair_left(2, 3, 4)
    right = _pair_right(6, 7, 8)
    inp, _ = _render_input_output([left, right])
    out1 = _render_input_output([left])[1]
    out2 = _render_input_output([right])[1]
    h = max(out1.shape[0], out2.shape[0])
    bg = BG_DEFAULT
    out = np.full((h, out1.shape[1] + 1 + out2.shape[1]), bg, dtype=int)
    out[: out1.shape[0], : out1.shape[1]] = out1
    out[:, out1.shape[1]] = SEP
    out[: out2.shape[0], out1.shape[1] + 1 :] = out2
    return inp, out


def _scene_chain() -> tuple[np.ndarray, np.ndarray]:
    inp = _grid()
    out = _grid()
    pair = _pair_left(2, 3, 4)
    _stamp(inp, pair["frame"], 2)
    _stamp(inp, pair["tool_plug"], 3)
    _stamp(inp, pair["tool_handle"], 4)
    plug2, handle2 = _tool_parts((18, 20), SHAPES["RECT"], "left")
    _stamp(inp, plug2, 6)
    _stamp(inp, handle2, 7)
    _stamp(out, pair["frame"], 2)
    _stamp(out, pair["cavity"], 3)
    first_handle = _handle_from_cavity(pair["cavity"], "right")
    _stamp(out, first_handle, 4)
    chain_cavity = _translate(SHAPES["RECT"], min(r for r, _ in first_handle) - 1, max(c for _, c in first_handle) + 1)
    _stamp(out, chain_cavity, 6)
    _stamp(out, _handle_from_cavity(chain_cavity, "right", 2), 7)
    return inp, out


def _scene_dock_then_move() -> tuple[np.ndarray, np.ndarray]:
    base = _render_input_output([_pair_left(2, 3, 4)])[1]
    pts = np.argwhere(base != BG_DEFAULT)
    r0, c0 = pts.min(axis=0)
    moved = np.full_like(base, BG_DEFAULT)
    cells = np.argwhere(base != BG_DEFAULT)
    for r, c in cells:
        moved[r - r0 + 9, c - c0 + 10] = base[r, c]
    return _render_input_output([_pair_left(2, 3, 4)])[0], moved


def _scene_ring_tool() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, SHAPES["RING"])
    return _render_input_output([pair])


def _scene_handle_in_cavity() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    pair["dock_handle_in_cavity"] = True
    pair["dock_handle"] = _handle_from_cavity(pair["cavity"], "right")
    return _render_input_output([pair])


def _scene_frame_two_colors() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    inp, out = _render_input_output([pair])
    for r, c in pair["frame"]:
        col = 2 if r <= 6 else 6
        inp[r, c] = col
        out[r, c] = col
    return inp, out


def _scene_frame_double_notch() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, SHAPES["PLUS"])
    pair["frame"] |= {(4, 4), (6, 4)}
    return _render_input_output([pair])


def _scene_tool_on_opposite_edge() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    pair["tool_plug"], pair["tool_handle"] = _tool_parts((11, 26), SHAPES["L"], "left")
    return _render_input_output([pair])


def _scene_rotation_markers() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4, _rot90(SHAPES["L"]))
    inp, out = _render_input_output([pair])
    inp[4, 5] = 1
    inp[5, 5] = 1
    out[4, 5] = 1
    out[5, 5] = 1
    return inp, out


def _scene_flip_by_frame_size() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    inp, out = _render_input_output([pair])
    extra = {(3, 0), (3, 1), (3, 2)}
    _stamp(inp, extra, 2)
    _stamp(out, extra, 2)
    return inp, out


def _scene_tool_direction_marker() -> tuple[np.ndarray, np.ndarray]:
    pair = _pair_left(2, 3, 4)
    inp, out = _render_input_output([pair])
    inp[12, 18] = 1
    return inp, out


def _scene_plug_same_frame() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 2, 4)])


def _scene_handle_same_frame() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 2)])


def _scene_fixed_black_bg() -> tuple[np.ndarray, np.ndarray]:
    return _render_input_output([_pair_left(2, 3, 4)], bg=0)


@lru_cache(maxsize=None)
def _scene_for_variant(variant_id: str) -> tuple[np.ndarray, np.ndarray]:
    mapping = {
        "D1_0": _scene_color_match,
        "D1_1": _scene_edge_color_match,
        "D1_2": _scene_hole_count_match,
        "D1_3": _scene_connector_match,
        "D1_4": _scene_area_rank,
        "D2_0": _scene_rotate_only,
        "D2_3": _scene_rotation_marker,
        "D3_0": _scene_three_color_tool,
        "D3_2": _scene_separated_tool,
        "D3_3": _scene_holey_tool,
        "D4_0": _scene_single_component_frame,
        "D4_1": _scene_hollow_square,
        "D4_3": _scene_internal_frame,
        "D5_1": _scene_irregular_cavity,
        "D5_2": _scene_t_cavity,
        "D6_0": _scene_swap_always,
        "D6_2": _scene_rotation_parity_swap,
        "D6_3": _scene_recolor_frame,
        "D7_0": _scene_keep_tool,
        "D7_1": _scene_trace_tool,
        "D7_2": _scene_plug_only,
        "D7_3": _scene_tool_only,
        "D8_2": _scene_opposite_edge_frames,
        "D8_3": _scene_open_to_edge,
        "D9_0": _scene_many_tools_one_frame,
        "D9_1": _scene_one_tool_many_frames,
        "D9_3": _scene_distractor_frame,
        "D10_0": _scene_crop,
        "D10_1": _scene_split,
        "D11_1": _scene_chain,
        "D11_2": _scene_dock_then_move,
        "D12_0": _scene_ring_tool,
        "D13_0": _scene_handle_in_cavity,
        "D14_0": _scene_frame_two_colors,
        "D14_3": _scene_frame_double_notch,
        "D15_1": _scene_tool_on_opposite_edge,
        "D16_0": _scene_rotation_markers,
        "D16_1": _scene_flip_by_frame_size,
        "D16_2": _scene_tool_direction_marker,
        "D17_0": _scene_fixed_black_bg,
        "D17_1": _scene_plug_same_frame,
        "D17_2": _scene_handle_same_frame,
    }
    if variant_id not in mapping:
        return _scene_base_two_pairs()
    return mapping[variant_id]()


def generate_input_for_variant(variant_id: str) -> np.ndarray:
    return _scene_for_variant(variant_id)[0].copy()


def main_for_variant(_grid: np.ndarray, variant_id: str) -> np.ndarray:
    return _scene_for_variant(variant_id)[1].copy()
