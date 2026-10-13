from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
ANCHOR_COLORS = (1, 2, 3)
BODY_COLORS = (4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_SIDES = ("top", "bottom", "left", "right")
OUTER_SIZE_RANGE = range(4, 9)
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _apply_transform(arr: np.ndarray, flip: bool, turns: int) -> np.ndarray:
    out = np.asarray(arr)
    if flip:
        out = np.fliplr(out)
    if turns:
        out = np.rot90(out, int(turns))
    return out.copy()


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    tmp = out.copy()
    out[tmp == int(a)] = int(b)
    out[tmp == int(b)] = int(a)
    return out


def _frame_border_coords(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _build_frame_patch(cavity_h: int, cavity_w: int, side: str, offset: int, marker_color: int) -> dict:
    fh, fw = cavity_h + 2, cavity_w + 2
    patch = np.full((fh, fw), FRAME, dtype=int)
    patch[1 : 1 + cavity_h, 1 : 1 + cavity_w] = BG
    if side == "top":
        marker = (0, 1 + offset)
    elif side == "bottom":
        marker = (fh - 1, 1 + offset)
    elif side == "left":
        marker = (1 + offset, 0)
    elif side == "right":
        marker = (1 + offset, fw - 1)
    else:
        raise ValueError("invalid side")
    patch[marker] = int(marker_color)
    cavity = {(r, c) for r in range(1, 1 + cavity_h) for c in range(1, 1 + cavity_w)}
    return {
        "patch": patch,
        "marker_local": marker,
        "cavity_local": cavity,
        "cavity_shape": (int(cavity_h), int(cavity_w)),
        "marker_side": side,
        "marker_offset": int(offset),
    }


def _build_docked_tool_patch(cavity_h: int, cavity_w: int, side: str, offset: int, body_color: int, anchor_color: int) -> np.ndarray:
    if side == "top":
        patch = np.full((cavity_h + 1, cavity_w), BG, dtype=int)
        patch[1:, :] = int(body_color)
        patch[0, offset] = int(anchor_color)
    elif side == "bottom":
        patch = np.full((cavity_h + 1, cavity_w), BG, dtype=int)
        patch[:cavity_h, :] = int(body_color)
        patch[cavity_h, offset] = int(anchor_color)
    elif side == "left":
        patch = np.full((cavity_h, cavity_w + 1), BG, dtype=int)
        patch[:, 1:] = int(body_color)
        patch[offset, 0] = int(anchor_color)
    elif side == "right":
        patch = np.full((cavity_h, cavity_w + 1), BG, dtype=int)
        patch[:, :cavity_w] = int(body_color)
        patch[offset, cavity_w] = int(anchor_color)
    else:
        raise ValueError("invalid side")
    return patch


def _offset_choices(side: str, cavity_h: int, cavity_w: int) -> list[int]:
    length = cavity_w if side in ("top", "bottom") else cavity_h
    offsets = list(range(length))
    if length % 2 == 1:
        mid = length // 2
        offsets = [v for v in offsets if v != mid]
    return offsets or [0]


def _parse_frame_marker(rect: tuple[int, int, int, int], marker_pos: tuple[int, int]) -> tuple[str, int]:
    r0, c0, r1, c1 = rect
    mr, mc = marker_pos
    if mr == r0 and c0 < mc < c1:
        return "top", int(mc - (c0 + 1))
    if mr == r1 and c0 < mc < c1:
        return "bottom", int(mc - (c0 + 1))
    if mc == c0 and r0 < mr < r1:
        return "left", int(mr - (r0 + 1))
    if mc == c1 and r0 < mr < r1:
        return "right", int(mr - (r0 + 1))
    raise AssertionError("marker must sit on a non-corner border cell")


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = []
    seen_rects = set()
    for r0 in range(h):
        for c0 in range(w):
            for fh in OUTER_SIZE_RANGE:
                r1 = r0 + fh - 1
                if r1 >= h:
                    continue
                for fw in OUTER_SIZE_RANGE:
                    c1 = c0 + fw - 1
                    if c1 >= w:
                        continue
                    if fh < 4 or fw < 4:
                        continue
                    inner = g[r0 + 1 : r1, c0 + 1 : c1]
                    if inner.size == 0 or not np.all(inner == BG):
                        continue
                    border = _frame_border_coords((r0, c0, r1, c1))
                    vals = [int(g[r, c]) for r, c in border]
                    if any(v == BG for v in vals):
                        continue
                    marker_cells = [(r, c, v) for (r, c), v in zip(border, vals) if v != FRAME]
                    if len(marker_cells) != 1:
                        continue
                    mr, mc, marker_color = marker_cells[0]
                    if marker_color not in ANCHOR_COLORS:
                        continue
                    rect = (int(r0), int(c0), int(r1), int(c1))
                    if rect in seen_rects:
                        continue
                    side, offset = _parse_frame_marker(rect, (mr, mc))
                    cavity = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
                    frames.append(
                        {
                            "rect": rect,
                            "marker_pos": (int(mr), int(mc)),
                            "marker_color": int(marker_color),
                            "marker_side": side,
                            "marker_offset": int(offset),
                            "cavity_coords": cavity,
                            "cavity_shape": (int(fh - 2), int(fw - 2)),
                        }
                    )
                    seen_rects.add(rect)
    frames.sort(key=lambda f: (f["rect"][0], f["rect"][1]))
    assert 2 <= len(frames) <= 3, f"expected 2-3 frames, got {len(frames)}"
    marker_colors = [f["marker_color"] for f in frames]
    assert len(set(marker_colors)) == len(marker_colors), "frame marker colors must be distinct"
    used = set()
    for frame in frames:
        rect = frame["rect"]
        border_cells = set(_frame_border_coords(rect))
        assert not (used & border_cells), "frames must not overlap"
        used |= border_cells
    return frames


def _extract_tools(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        for r, c in _frame_border_coords(frame["rect"]):
            g[r, c] = BG
    comps = _components_mask(g != BG)
    tools = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        mask = np.zeros(patch.shape, dtype=bool)
        r0, c0, _r1, _c1 = bbox
        for r, c in coords:
            mask[int(r - r0), int(c - c0)] = True
        vals = Counter(int(patch[r, c]) for r, c in np.argwhere(mask))
        assert len(vals) == 2, f"tool must use exactly two colors: {vals}"
        anchor_color, anchor_count = min(vals.items(), key=lambda kv: (kv[1], kv[0]))
        assert anchor_count == 1, f"anchor must be singleton: {vals}"
        body_colors = [col for col in vals if col != anchor_color]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        anchor_local = tuple(map(int, np.argwhere((patch == int(anchor_color)) & mask)[0]))
        tools.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "mask": mask,
                "anchor_color": int(anchor_color),
                "body_color": body_color,
                "anchor_local": anchor_local,
            }
        )
    tools.sort(key=lambda t: (t["bbox"][0], t["bbox"][1]))
    assert 2 <= len(tools) <= 3, f"expected 2-3 tools, got {len(tools)}"
    return tools


def _match_transforms(tool: dict, frame: dict) -> list[dict]:
    marker = frame["marker_pos"]
    cavity = frame["cavity_coords"]
    matches = []
    for flip in (False, True):
        for turns in range(4):
            tpatch = _apply_transform(tool["patch"], flip, turns)
            tmask = tpatch != BG
            anchor_cells = np.argwhere((tpatch == int(tool["anchor_color"])) & tmask)
            if anchor_cells.shape[0] != 1:
                continue
            ar, ac = map(int, anchor_cells[0])
            top = int(marker[0] - ar)
            left = int(marker[1] - ac)
            placed = {(top + int(r), left + int(c)) for r, c in np.argwhere(tmask)}
            if marker not in placed:
                continue
            if placed - {marker} != cavity:
                continue
            matches.append(
                {
                    "flip": flip,
                    "turns": int(turns),
                    "patch": tpatch,
                    "top": top,
                    "left": left,
                }
            )
    return matches


def _local_match_transforms(tool_patch: np.ndarray, anchor_color: int, cavity_local: set[tuple[int, int]], marker_local: tuple[int, int]) -> list[dict]:
    dummy_tool = {"patch": tool_patch, "anchor_color": int(anchor_color)}
    dummy_frame = {"cavity_coords": set(cavity_local), "marker_pos": tuple(marker_local)}
    return _match_transforms(dummy_tool, dummy_frame)


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    tools = _extract_tools(g, frames)
    tool_by_anchor = {tool["anchor_color"]: tool for tool in tools}
    frame_by_marker = {frame["marker_color"]: frame for frame in frames}
    assert set(tool_by_anchor) == set(frame_by_marker), "anchor colors must pair tools with frames"

    matches = []
    for color in sorted(tool_by_anchor):
        tool = tool_by_anchor[color]
        frame = frame_by_marker[color]
        options = _match_transforms(tool, frame)
        assert len(options) == 1, f"expected unique transform for color {color}, got {len(options)}"
        match = options[0]
        matches.append({"tool": tool, "frame": frame, **match})

    out = g.copy()
    for tool in tools:
        rr, cc = tool["coords"][:, 0], tool["coords"][:, 1]
        out[rr, cc] = BG

    for match in matches:
        tool = match["tool"]
        patch = match["patch"].copy()
        if match["flip"]:
            patch = _swap_colors(patch, tool["anchor_color"], tool["body_color"])
        tmask = patch != BG
        top = match["top"]
        left = match["left"]
        for r, c in np.argwhere(tmask):
            out[top + int(r), left + int(c)] = int(patch[int(r), int(c)])

    return out, frames, tools, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each two-color tool with the frame that shares its singleton anchor color, then dock it with the unique rotation/reflection that fills the frame cavity.

    Input structure:
    - Black background with 2-3 grey rectangular frames near the edges; each frame has a black rectangular cavity and one non-grey marker pixel on a wall cell.
    - The marker colors are distinct and each marker color also appears as the singleton rare color of exactly one central two-color tool.
    - Every tool is one 4-connected component whose non-anchor cells form a rectangle and whose singleton anchor protrudes from one side.

    Rule:
    - Detect every frame, its cavity, and its wall marker, and detect every tool and its singleton anchor color.
    - Match tool to frame by anchor color = marker color, then test all 8 dihedral transforms of that tool.
    - Keep the unique transform whose anchor lands on the marker cell and whose remaining cells exactly fill the cavity.
    - If the chosen transform uses reflection, swap the tool's two colors before stamping it into the frame; otherwise keep its colors.
    - Remove the original tool from the center and place the transformed tool into the matched frame.

    Key insight:
    - Color chooses the target frame, geometry chooses the unique docking pose, and reflection parity controls whether the two tool colors must be swapped.
    """
    out, _frames, _tools, _matches = _solve(np.asarray(grid, dtype=int))
    return out


def _bbox_can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = grid[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _choose_layout_dims(num_pairs: int) -> list[tuple[int, int]]:
    if num_pairs == 2:
        pair_family = random.choice([[(2, 3), (3, 2)], [(2, 4), (4, 2)]])
        return pair_family
    first = random.choice([[(2, 3), (3, 2)], [(2, 4), (4, 2)]])
    remaining = [(2, 3), (3, 2), (2, 4), (4, 2)]
    extra_choices = [dims for dims in remaining if dims not in first]
    return first + [random.choice(extra_choices)]


def _choose_flip_flags(num_pairs: int) -> list[bool]:
    if num_pairs == 2:
        flags = [False, True]
    else:
        flags = [False, True, random.choice([False, True])]
    random.shuffle(flags)
    return flags


def _frame_positions_for_side(side: str, frame_h: int, frame_w: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    positions = []
    if side == "top":
        rows = [r for r in (0, 1, 2) if r + frame_h <= grid_h]
        cols = list(range(0, grid_w - frame_w + 1))
        for r in rows:
            for c in cols:
                positions.append((r, c))
    elif side == "bottom":
        rows = [r for r in (grid_h - frame_h, grid_h - frame_h - 1, grid_h - frame_h - 2) if r >= 0]
        cols = list(range(0, grid_w - frame_w + 1))
        for r in rows:
            for c in cols:
                positions.append((r, c))
    elif side == "left":
        cols = [c for c in (0, 1, 2) if c + frame_w <= grid_w]
        rows = list(range(0, grid_h - frame_h + 1))
        for r in rows:
            for c in cols:
                positions.append((r, c))
    elif side == "right":
        cols = [c for c in (grid_w - frame_w, grid_w - frame_w - 1, grid_w - frame_w - 2) if c >= 0]
        rows = list(range(0, grid_h - frame_h + 1))
        for r in rows:
            for c in cols:
                positions.append((r, c))
    else:
        raise ValueError("invalid side")
    random.shuffle(positions)
    return positions


def _tool_positions(tool_h: int, tool_w: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    rows = list(range(3, max(4, grid_h - tool_h - 2)))
    cols = list(range(3, max(4, grid_w - tool_w - 2)))
    positions = [(r, c) for r in rows for c in cols]
    positions = [
        (r, c)
        for r, c in positions
        if grid_h // 5 <= r <= grid_h - tool_h - grid_h // 5
        and grid_w // 5 <= c <= grid_w - tool_w - grid_w // 5
    ] or positions
    random.shuffle(positions)
    return positions


def _semantic_parse(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    frames = _detect_frames(grid)
    tools = _extract_tools(grid, frames)
    tool_by_anchor = {tool["anchor_color"]: tool for tool in tools}
    frame_by_marker = {frame["marker_color"]: frame for frame in frames}
    assert set(tool_by_anchor) == set(frame_by_marker)
    matches = []
    for color in sorted(tool_by_anchor):
        options = _match_transforms(tool_by_anchor[color], frame_by_marker[color])
        assert len(options) == 1
        matches.append({"tool": tool_by_anchor[color], "frame": frame_by_marker[color], **options[0]})
    return frames, tools, matches


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        frames, tools, matches = _semantic_parse(grid)
    except Exception:
        return False

    if len(frames) != len(tools):
        return False
    if not any(match["flip"] for match in matches):
        return False
    if not any(not match["flip"] for match in matches):
        return False

    # Color matching must be necessary: at least one tool should geometrically fit
    # more than one frame if color is ignored.
    geometric_options = 0
    for tool in tools:
        count = sum(1 for frame in frames if _match_transforms(tool, frame))
        geometric_options = max(geometric_options, count)
    if geometric_options < 2:
        return False

    # Every paired frame must have a unique transform, but the non-paired search
    # should still admit at least one false-colored geometric competitor overall.
    false_color_match = False
    for tool in tools:
        for frame in frames:
            if tool["anchor_color"] == frame["marker_color"]:
                if len(_match_transforms(tool, frame)) != 1:
                    return False
            elif _match_transforms(tool, frame):
                false_color_match = True
    if not false_color_match:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_pair_specs(num_pairs: int) -> list[dict] | None:
    dims_list = _choose_layout_dims(num_pairs)
    anchor_colors = random.sample(list(ANCHOR_COLORS), num_pairs)
    body_colors = random.sample(list(BODY_COLORS), num_pairs)
    flip_flags = _choose_flip_flags(num_pairs)
    specs = []
    for idx in range(num_pairs):
        cavity_h, cavity_w = dims_list[idx]
        marker_side = random.choice(FRAME_SIDES)
        offset = random.choice(_offset_choices(marker_side, cavity_h, cavity_w))
        frame_info = _build_frame_patch(cavity_h, cavity_w, marker_side, offset, anchor_colors[idx])
        docked_patch = _build_docked_tool_patch(
            cavity_h,
            cavity_w,
            marker_side,
            offset,
            body_colors[idx],
            anchor_colors[idx],
        )

        input_choices = []
        for flip in (False, True):
            for turns in range(4):
                candidate = _apply_transform(docked_patch, flip, turns)
                options = _local_match_transforms(
                    candidate,
                    anchor_colors[idx],
                    frame_info["cavity_local"],
                    frame_info["marker_local"],
                )
                if len(options) == 1 and bool(options[0]["flip"]) == bool(flip_flags[idx]):
                    input_choices.append(candidate)
        if not input_choices:
            return None
        specs.append(
            {
                "anchor_color": anchor_colors[idx],
                "body_color": body_colors[idx],
                "frame_patch": frame_info["patch"],
                "frame_shape": frame_info["patch"].shape,
                "marker_side": marker_side,
                "marker_offset": int(offset),
                "cavity_shape": (cavity_h, cavity_w),
                "docked_patch": docked_patch,
                "input_patch": random.choice(input_choices),
                "required_flip": bool(flip_flags[idx]),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.choice([2, 3])
    grid_h = random.randint(18, 24) if num_pairs == 2 else random.randint(21, 28)
    grid_w = random.randint(18, 24) if num_pairs == 2 else random.randint(21, 28)
    specs = _gen_pair_specs(num_pairs)
    if specs is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    frame_sides = random.sample(list(FRAME_SIDES), num_pairs)
    for spec, outer_side in zip(specs, frame_sides):
        fh, fw = spec["frame_shape"]
        placed = False
        for top, left in _frame_positions_for_side(outer_side, fh, fw, grid_h, grid_w):
            if not _bbox_can_place(occupied, top, left, fh, fw, padding=1):
                continue
            spec["frame_top_left"] = (top, left)
            _place_patch(grid, spec["frame_patch"], top, left)
            _mark_bbox(occupied, top, left, fh, fw)
            placed = True
            break
        if not placed:
            return None

    for spec in sorted(specs, key=lambda d: -np.count_nonzero(d["input_patch"])):
        patch = spec["input_patch"]
        th, tw = patch.shape
        placed = False
        for top, left in _tool_positions(th, tw, grid_h, grid_w):
            if not _bbox_can_place(occupied, top, left, th, tw, padding=2):
                continue
            spec["tool_top_left"] = (top, left)
            _place_patch(grid, patch, top, left)
            _mark_bbox(occupied, top, left, th, tw)
            placed = True
            break
        if not placed:
            return None

    if not _validate_semantics(grid):
        return None

    frames, _tools, matches = _semantic_parse(grid)
    parsed_shapes = sorted(frame["cavity_shape"] for frame in frames)
    if len(set(parsed_shapes)) != len(parsed_shapes):
        return None

    # Branch coverage and spec compliance checks.
    if len(matches) != num_pairs:
        return None
    flips = [match["flip"] for match in matches]
    if not (any(flips) and any(not v for v in flips)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after retries")
