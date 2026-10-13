import random
from typing import Iterable

import numpy as np


BBox = tuple[int, int, int, int]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_height(bbox: BBox) -> int:
    return bbox[2] - bbox[0] + 1


def _bbox_width(bbox: BBox) -> int:
    return bbox[3] - bbox[1] + 1


def _bbox_area(bbox: BBox) -> int:
    return _bbox_height(bbox) * _bbox_width(bbox)


def _translate_bbox(bbox: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rectangles_overlap(a: BBox, b: BBox) -> bool:
    return _intervals_overlap(a[0], a[2], b[0], b[2]) and _intervals_overlap(a[1], a[3], b[1], b[3])


def _rectangles_too_close(a: BBox, b: BBox, padding: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _paint_rect(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rect_cells(bbox: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand_cells(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _consecutive(values: np.ndarray) -> bool:
    vals = np.sort(np.unique(values.astype(int)))
    return vals.size > 0 and int(vals[-1] - vals[0] + 1) == int(vals.size)


def _stem_signature(rem_rows: np.ndarray, rem_cols: np.ndarray, body_bbox: BBox) -> tuple[str, int] | None:
    r0, c0, r1, c1 = body_bbox
    if rem_rows.size == 0:
        return None

    if int(rem_rows.min()) == int(rem_rows.max()):
        rr = int(rem_rows[0])
        if not (r0 <= rr <= r1) or not _consecutive(rem_cols):
            return None
        cmin = int(rem_cols.min())
        cmax = int(rem_cols.max())
        if cmin == c1 + 1:
            return "right", cmax - cmin + 1
        if cmax == c0 - 1:
            return "left", cmax - cmin + 1
        return None

    if int(rem_cols.min()) == int(rem_cols.max()):
        cc = int(rem_cols[0])
        if not (c0 <= cc <= c1) or not _consecutive(rem_rows):
            return None
        rmin = int(rem_rows.min())
        rmax = int(rem_rows.max())
        if rmin == r1 + 1:
            return "down", rmax - rmin + 1
        if rmax == r0 - 1:
            return "up", rmax - rmin + 1
        return None

    return None


def _parse_moving_object(mask: np.ndarray, color: int) -> dict:
    coords = np.argwhere(mask)
    bbox = _bbox_from_coords(coords)
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    best: dict | None = None

    for r0 in range(bbox[0], bbox[2] + 1):
        for r1 in range(r0 + 1, bbox[2] + 1):
            for c0 in range(bbox[1], bbox[3] + 1):
                for c1 in range(c0 + 1, bbox[3] + 1):
                    body_bbox = (r0, c0, r1, c1)
                    area = _bbox_area(body_bbox)
                    inside = (rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)
                    if int(inside.sum()) != area:
                        continue
                    signature = _stem_signature(rows[~inside], cols[~inside], body_bbox)
                    if signature is None:
                        continue
                    direction, stem_len = signature
                    candidate = {
                        "color": int(color),
                        "body_bbox": body_bbox,
                        "direction": direction,
                        "stem_len": int(stem_len),
                        "area": int(area),
                    }
                    if best is None or candidate["area"] > best["area"]:
                        best = candidate

    if best is None:
        raise AssertionError(f"could not parse moving object for color {color}")
    return best


def _analyze_grid(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    bg = _background_color(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    anchors: list[dict] = []
    moving: list[dict] = []

    for color in colors:
        mask = grid == color
        coords = np.argwhere(mask)
        bbox = _bbox_from_coords(coords)
        if int(coords.shape[0]) == _bbox_area(bbox):
            anchors.append({"color": color, "bbox": bbox})
        else:
            moving.append(_parse_moving_object(mask, color))

    if len(anchors) != 1:
        raise AssertionError("expected exactly one rectangular anchor")
    if not moving:
        raise AssertionError("expected at least one moving object")
    return bg, anchors[0], moving


def _ahead_gap(body_bbox: BBox, fixed_bbox: BBox, direction: str) -> int | None:
    br0, bc0, br1, bc1 = body_bbox
    fr0, fc0, fr1, fc1 = fixed_bbox

    if direction == "right":
        if fc0 <= bc1 or not _intervals_overlap(br0, br1, fr0, fr1):
            return None
        return fc0 - bc1 - 1
    if direction == "left":
        if fc1 >= bc0 or not _intervals_overlap(br0, br1, fr0, fr1):
            return None
        return bc0 - fc1 - 1
    if direction == "down":
        if fr0 <= br1 or not _intervals_overlap(bc0, bc1, fc0, fc1):
            return None
        return fr0 - br1 - 1
    if direction == "up":
        if fr1 >= br0 or not _intervals_overlap(bc0, bc1, fc0, fc1):
            return None
        return br0 - fr1 - 1
    raise AssertionError(f"unknown direction {direction}")


def _move_to_contact(body_bbox: BBox, fixed_bbox: BBox, direction: str) -> tuple[BBox, int] | None:
    gap = _ahead_gap(body_bbox, fixed_bbox, direction)
    if gap is None:
        return None
    dr, dc = DIRS[direction]
    return _translate_bbox(body_bbox, dr * gap, dc * gap), gap


def _nearest_target(body_bbox: BBox, direction: str, fixed_boxes: list[BBox]) -> tuple[int, BBox, int] | None:
    best: tuple[int, BBox, int] | None = None
    for idx, fixed_bbox in enumerate(fixed_boxes):
        moved = _move_to_contact(body_bbox, fixed_bbox, direction)
        if moved is None:
            continue
        new_bbox, gap = moved
        if best is None or gap < best[2]:
            best = (idx, new_bbox, gap)
    return best


def _settle_sequence(anchor: dict, moving: list[dict], require_unique: bool = False) -> list[dict]:
    fixed = [{"color": anchor["color"], "bbox": anchor["bbox"]}]
    remaining = [
        {
            "color": obj["color"],
            "body_bbox": obj["body_bbox"],
            "direction": obj["direction"],
        }
        for obj in moving
    ]
    settled: list[dict] = []

    while remaining:
        fixed_boxes = [obj["bbox"] for obj in fixed]
        eligible = []
        for idx, obj in enumerate(remaining):
            target = _nearest_target(obj["body_bbox"], obj["direction"], fixed_boxes)
            if target is None:
                continue
            _target_idx, new_bbox, gap = target
            eligible.append(
                (
                    new_bbox[0],
                    new_bbox[1],
                    obj["body_bbox"][0],
                    obj["body_bbox"][1],
                    gap,
                    idx,
                    new_bbox,
                )
            )

        if not eligible:
            raise AssertionError("no movable object can reach the current fixed set")
        if require_unique and len(eligible) != 1:
            raise AssertionError("generator must create a unique cascade order")

        eligible.sort()
        chosen_idx = eligible[0][5]
        new_bbox = eligible[0][6]
        obj = remaining.pop(chosen_idx)
        settled_obj = {
            "color": obj["color"],
            "bbox": new_bbox,
            "direction": obj["direction"],
        }
        fixed.append(settled_obj)
        settled.append(settled_obj)

    return fixed


def _render_output(shape: tuple[int, int], bg: int, settled: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for obj in settled:
        _paint_rect(out, obj["bbox"], obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each rectangular body along its stem direction until it touches the current fixed block chain.

    Input structure:
    - A black background contains one solid rectangular anchor of a unique color.
    - Every other color forms one connected object made of a solid rectangular body plus
      a straight one-cell-wide stem attached to one side of that body.
    - Only one moving object can initially reach the anchor along its stem direction; the
      others must wait for earlier bodies to settle and become new fixed blocks.

    Rule:
    - Keep the original anchor in place and erase every stem.
    - Repeatedly find a body whose slide direction reaches an already fixed rectangle,
      then translate that body straight along the stem direction until its face becomes
      edge-adjacent to the nearest such fixed rectangle.
    - The newly settled body immediately joins the fixed set; continue until all moving
      objects have been placed.

    Key insight:
    - The stem gives only the slide direction, while the stopping distance is determined
      dynamically by whichever earlier rectangle is currently the nearest reachable anchor.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")

    bg, anchor, moving = _analyze_grid(g)
    settled = _settle_sequence(anchor, moving, require_unique=False)
    return _render_output(g.shape, bg, settled)


def _candidate_final_positions(parent_bbox: BBox, height: int, width: int, direction: str) -> list[BBox]:
    pr0, pc0, pr1, pc1 = parent_bbox
    candidates: list[BBox] = []

    if direction == "right":
        c0 = pc0 - width
        for r0 in range(pr0 - height + 1, pr1 + 1):
            candidates.append((r0, c0, r0 + height - 1, c0 + width - 1))
    elif direction == "left":
        c0 = pc1 + 1
        for r0 in range(pr0 - height + 1, pr1 + 1):
            candidates.append((r0, c0, r0 + height - 1, c0 + width - 1))
    elif direction == "down":
        r0 = pr0 - height
        for c0 in range(pc0 - width + 1, pc1 + 1):
            candidates.append((r0, c0, r0 + height - 1, c0 + width - 1))
    elif direction == "up":
        r0 = pr1 + 1
        for c0 in range(pc0 - width + 1, pc1 + 1):
            candidates.append((r0, c0, r0 + height - 1, c0 + width - 1))
    else:
        raise AssertionError(f"unknown direction {direction}")

    return candidates


def _lane_hits_only_parent(child_bbox: BBox, direction: str, nodes: list[dict], parent_idx: int) -> bool:
    hits = [
        idx
        for idx, node in enumerate(nodes)
        if _ahead_gap(child_bbox, node["bbox"], direction) is not None
    ]
    return len(hits) == 1 and hits[0] == parent_idx


def _valid_final_child(child_bbox: BBox, direction: str, nodes: list[dict], parent_idx: int) -> bool:
    if not _lane_hits_only_parent(child_bbox, direction, nodes, parent_idx):
        return False

    for idx, node in enumerate(nodes):
        bbox = node["bbox"]
        if _rectangles_overlap(child_bbox, bbox):
            return False
        if idx != parent_idx and _rectangles_too_close(child_bbox, bbox, padding=1):
            return False
    return True


def _build_final_nodes() -> list[dict] | None:
    moving_count = random.randint(2, 5)
    palette = random.sample(range(1, 10), moving_count + 1)
    anchor_h = random.randint(2, 5)
    anchor_w = random.randint(2, 6)
    nodes = [{"color": palette[0], "bbox": (0, 0, anchor_h - 1, anchor_w - 1)}]

    for i in range(moving_count):
        parent_idx = len(nodes) - 1
        candidates: list[dict] = []
        for height in range(2, 5):
            for width in range(2, 5):
                directions = list(DIRS)
                random.shuffle(directions)
                for direction in directions:
                    for child_bbox in _candidate_final_positions(nodes[parent_idx]["bbox"], height, width, direction):
                        if _valid_final_child(child_bbox, direction, nodes, parent_idx):
                            candidates.append(
                                {
                                    "color": palette[i + 1],
                                    "bbox": child_bbox,
                                    "parent": parent_idx,
                                    "direction": direction,
                                }
                            )
        if not candidates:
            return None
        nodes.append(random.choice(candidates))

    return nodes


def _stem_cells(body_bbox: BBox, direction: str, stem_len: int, offset: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = body_bbox
    if direction == "right":
        return {(offset, c) for c in range(c1 + 1, c1 + stem_len + 1)}
    if direction == "left":
        return {(offset, c) for c in range(c0 - stem_len, c0)}
    if direction == "down":
        return {(r, offset) for r in range(r1 + 1, r1 + stem_len + 1)}
    if direction == "up":
        return {(r, offset) for r in range(r0 - stem_len, r0)}
    raise AssertionError(f"unknown direction {direction}")


def _attachment_offsets(body_bbox: BBox, direction: str) -> list[int]:
    r0, c0, r1, c1 = body_bbox
    if direction in {"right", "left"}:
        return list(range(r0, r1 + 1))
    return list(range(c0, c1 + 1))


def _build_input_nodes(final_nodes: list[dict]) -> list[dict] | None:
    input_nodes = [{"color": final_nodes[0]["color"], "bbox": final_nodes[0]["bbox"]}]
    occupied = _rect_cells(final_nodes[0]["bbox"])
    forbidden = _expand_cells(occupied)

    for node in final_nodes[1:]:
        direction = node["direction"]
        dr, dc = DIRS[OPPOSITE[direction]]
        distances = list(range(2, 7))
        random.shuffle(distances)
        placed = False

        for distance in distances:
            body_bbox = _translate_bbox(node["bbox"], dr * distance, dc * distance)
            body_cells = _rect_cells(body_bbox)
            if body_cells & forbidden:
                continue

            offsets = _attachment_offsets(body_bbox, direction)
            random.shuffle(offsets)
            for stem_len in range(1, distance):
                for offset in offsets:
                    stem = _stem_cells(body_bbox, direction, stem_len, offset)
                    cells = body_cells | stem
                    if cells & forbidden:
                        continue
                    input_nodes.append(
                        {
                            "color": node["color"],
                            "bbox": body_bbox,
                            "direction": direction,
                            "stem_len": stem_len,
                            "stem_cells": stem,
                        }
                    )
                    occupied |= cells
                    forbidden |= _expand_cells(cells)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            return None

    return input_nodes


def _normalize_layout(input_nodes: list[dict], final_nodes: list[dict]) -> tuple[list[dict], list[dict], tuple[int, int]] | None:
    all_cells: set[tuple[int, int]] = set()
    for node in input_nodes:
        all_cells |= _rect_cells(node["bbox"])
        if "stem_cells" in node:
            all_cells |= set(node["stem_cells"])

    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)

    content_h = max_r - min_r + 1
    content_w = max_c - min_c + 1
    if content_h > 30 or content_w > 30:
        return None

    top_margin = random.randint(1, min(3, 30 - content_h))
    left_margin = random.randint(1, min(3, 30 - content_w))
    extra_bottom = random.randint(0, 30 - content_h - top_margin)
    extra_right = random.randint(0, 30 - content_w - left_margin)
    shape = (content_h + top_margin + extra_bottom, content_w + left_margin + extra_right)
    shift_r = top_margin - min_r
    shift_c = left_margin - min_c

    def shift_bbox(bbox: BBox) -> BBox:
        return _translate_bbox(bbox, shift_r, shift_c)

    shifted_input: list[dict] = []
    for node in input_nodes:
        shifted = {k: v for k, v in node.items() if k != "stem_cells"}
        shifted["bbox"] = shift_bbox(node["bbox"])
        if "stem_cells" in node:
            shifted["stem_cells"] = {(r + shift_r, c + shift_c) for r, c in node["stem_cells"]}
        shifted_input.append(shifted)

    shifted_final = []
    for node in final_nodes:
        shifted = dict(node)
        shifted["bbox"] = shift_bbox(node["bbox"])
        shifted_final.append(shifted)

    return shifted_input, shifted_final, shape


def _render_input(shape: tuple[int, int], input_nodes: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for node in input_nodes:
        _paint_rect(grid, node["bbox"], node["color"])
        for r, c in node.get("stem_cells", set()):
            grid[r, c] = node["color"]
    return grid


def _render_final(shape: tuple[int, int], final_nodes: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for node in final_nodes:
        _paint_rect(grid, node["bbox"], node["color"])
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, anchor, moving = _analyze_grid(grid)
        settled = _settle_sequence(anchor, moving, require_unique=True)
        actual = _render_output(grid.shape, bg, settled)
    except AssertionError:
        return False
    return np.array_equal(actual, expected) and np.array_equal(main(grid), expected)


def _generate_candidate() -> np.ndarray | None:
    final_nodes = _build_final_nodes()
    if final_nodes is None:
        return None

    input_nodes = _build_input_nodes(final_nodes)
    if input_nodes is None:
        return None

    normalized = _normalize_layout(input_nodes, final_nodes)
    if normalized is None:
        return None
    shifted_input, shifted_final, shape = normalized

    grid = _render_input(shape, shifted_input)
    expected = _render_final(shape, shifted_final)
    if not _validate_generated(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-anchor sliding input")
