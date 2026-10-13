from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg


FRAME_COLOR = int(Color.GREY)
BG_COLOR = int(Color.BLACK)
PALETTE = np.array(
    [
        int(Color.BLUE),
        int(Color.RED),
        int(Color.GREEN),
        int(Color.YELLOW),
        int(Color.ORANGE),
        int(Color.PINK),
        int(Color.TEAL),
        int(Color.MAROON),
    ],
    dtype=int,
)

GRID_SIDE_RANGE = (22, 30)
ROOT_PADDING = 2
LEAF_SIDE_VALUES = tuple(range(5, 10))
PARENT_EXTRA_RANGE = (0, 3)
GENERATION_RETRIES = 400

FOREST_TEMPLATES = (
    ((), ((),)),
    ((), (((),),)),
    (((), ()), (((),),)),
    ((), (((),), ())),
    (((),), (((),), ())),
)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _center_distance_fill(shape: tuple[int, int], depth: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    center_r = (h - 1) // 2
    center_c = (w - 1) // 2
    dist = np.abs(rr - center_r) + np.abs(cc - center_c)
    return PALETTE[(dist + depth - 1) % len(PALETTE)]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG_COLOR
    assert set(map(int, np.unique(g))) <= {BG_COLOR, FRAME_COLOR}

    frames: list[dict] = []
    for coords in components_mask(g == FRAME_COLOR, connectivity=4):
        bbox = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5
        patch = np.zeros((h, w), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(patch, expected), "gray components must be hollow 1-pixel rectangles"
        frames.append({"bbox": bbox, "parent": None, "children": [], "depth": None})

    assert len(frames) >= 2
    order = sorted(range(len(frames)), key=lambda i: (
        (frames[i]["bbox"][2] - frames[i]["bbox"][0] + 1) * (frames[i]["bbox"][3] - frames[i]["bbox"][1] + 1),
        frames[i]["bbox"],
    ))
    for idx in order:
        r0, c0, r1, c1 = frames[idx]["bbox"]
        parent_candidates = []
        for j, frame in enumerate(frames):
            if j == idx:
                continue
            pr0, pc0, pr1, pc1 = frame["bbox"]
            if pr0 < r0 and c0 > pc0 and r1 < pr1 and c1 < pc1:
                parent_candidates.append(j)
        if parent_candidates:
            parent = min(
                parent_candidates,
                key=lambda j: (
                    (frames[j]["bbox"][2] - frames[j]["bbox"][0] + 1)
                    * (frames[j]["bbox"][3] - frames[j]["bbox"][1] + 1),
                    frames[j]["bbox"],
                ),
            )
            frames[idx]["parent"] = parent
            frames[parent]["children"].append(idx)

    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    assert len(roots) >= 2

    def assign_depth(idx: int, depth: int) -> None:
        frames[idx]["depth"] = depth
        for child in sorted(frames[idx]["children"], key=lambda k: frames[k]["bbox"]):
            assign_depth(child, depth + 1)

    for idx in sorted(roots, key=lambda k: frames[k]["bbox"]):
        assign_depth(idx, 1)

    assert max(int(frame["depth"]) for frame in frames) <= 3
    return frames


def _render_from_records(shape: tuple[int, int], records: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in records:
        _draw_frame(out, frame["bbox"])

    def paint(idx: int) -> None:
        frame = records[idx]
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        children = list(frame["children"])
        depth = int(frame["depth"])
        if not children:
            inner[:, :] = _center_distance_fill(inner.shape, depth)
            return

        fill_mask = np.ones(inner.shape, dtype=bool)
        for child_idx in children:
            cr0, cc0, cr1, cc1 = records[child_idx]["bbox"]
            fill_mask[
                cr0 - (r0 + 1) : cr1 - (r0 + 1) + 1,
                cc0 - (c0 + 1) : cc1 - (c0 + 1) + 1,
            ] = False
        inner[fill_mask] = int(PALETTE[(depth - 1) % len(PALETTE)])
        for child_idx in sorted(children, key=lambda k: records[k]["bbox"]):
            paint(child_idx)

    roots = [i for i, frame in enumerate(records) if frame["parent"] is None]
    for idx in sorted(roots, key=lambda k: records[k]["bbox"]):
        paint(idx)
    return out


def _flatten_nodes(roots: list[dict]) -> list[dict]:
    records: list[dict] = []

    def visit(node: dict, parent: int | None) -> int:
        idx = len(records)
        record = {
            "bbox": node["bbox"],
            "parent": parent,
            "children": [],
            "depth": int(node["depth"]),
        }
        records.append(record)
        for child in node["children"]:
            child_idx = visit(child, idx)
            record["children"].append(child_idx)
        return idx

    for root in sorted(roots, key=lambda node: node["bbox"]):
        visit(root, None)
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular frame interiors with depth-shifted concentric palettes.

    Input structure:
    - Black is the dominant background and gray pixels form one-pixel-thick hollow rectangular frames.
    - Frames are axis-aligned, may be nested up to three levels deep, and there are at least two independent outer roots.
    - A frame interior may be empty or may contain one or more disjoint child frames entirely inside it.

    Rule:
    - Recover the frame containment forest: outer frames have depth 1 and each child depth is parent depth plus one.
    - Keep every gray frame border unchanged and keep all pixels outside every frame black.
    - For a leaf frame, color every interior cell by Manhattan distance from the floored interior center using the cyclic palette
      [1,2,3,4,7,6,8,9], offset by depth-1.
    - For a non-leaf frame, color only the part of its interior that is outside every child frame bbox with the single palette color at index depth-1.

    Key insight:
    - The same concentric distance pattern appears at multiple scales, but its starting color is controlled by containment depth, so solving requires building the nesting tree rather than treating each frame in isolation.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    return _render_from_records(grid.shape, frames)


def _make_node(spec: tuple) -> dict:
    children = [_make_node(child) for child in spec]
    return {
        "children": children,
        "layout": random.choice(("h", "v")) if children else None,
        "h": None,
        "w": None,
        "bbox": None,
        "depth": None,
    }


def _assign_sizes(node: dict, depth: int, used_leaf_sizes: dict[int, set[tuple[int, int]]]) -> bool:
    node["depth"] = depth
    if not node["children"]:
        used = used_leaf_sizes.setdefault(depth, set())
        candidates = [
            (h, w)
            for h in LEAF_SIDE_VALUES
            for w in LEAF_SIDE_VALUES
            if (h - 2, w - 2) not in used
        ]
        if not candidates:
            return False
        node["h"], node["w"] = random.choice(candidates)
        used.add((node["h"] - 2, node["w"] - 2))
        return True

    for child in node["children"]:
        if not _assign_sizes(child, depth + 1, used_leaf_sizes):
            return False

    if node["layout"] == "h":
        min_h = max(int(child["h"]) for child in node["children"]) + 4
        min_w = sum(int(child["w"]) for child in node["children"]) + len(node["children"]) - 1 + 4
    else:
        min_h = sum(int(child["h"]) for child in node["children"]) + len(node["children"]) - 1 + 4
        min_w = max(int(child["w"]) for child in node["children"]) + 4

    extra_h = random.randint(*PARENT_EXTRA_RANGE)
    extra_w = random.randint(*PARENT_EXTRA_RANGE)
    if len(node["children"]) == 1:
        extra_h += 1
        extra_w += 1
    node["h"] = min_h + extra_h
    node["w"] = min_w + extra_w
    return True


def _random_partition(total: int, slots: int) -> list[int]:
    if slots == 1:
        return [total]
    cuts = sorted(random.sample(range(total + slots - 1), slots - 1))
    values = []
    prev = -1
    for cut in cuts + [total + slots - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _place_children(node: dict) -> bool:
    if not node["children"]:
        return True

    r0, c0, r1, c1 = node["bbox"]
    usable_h = r1 - r0 - 3
    usable_w = c1 - c0 - 3
    children = list(node["children"])

    if node["layout"] == "h":
        base_w = sum(int(child["w"]) for child in children) + len(children) - 1
        extra_w = usable_w - base_w
        if extra_w < 0:
            return False
        offsets = _random_partition(extra_w, len(children) + 1)
        x = c0 + 2 + offsets[0]
        for i, child in enumerate(children):
            room_h = usable_h - int(child["h"])
            if room_h < 0:
                return False
            y = r0 + 2 + random.randint(0, room_h)
            child["bbox"] = (y, x, y + int(child["h"]) - 1, x + int(child["w"]) - 1)
            x += int(child["w"])
            if i < len(children) - 1:
                x += 1 + offsets[i + 1]
    else:
        base_h = sum(int(child["h"]) for child in children) + len(children) - 1
        extra_h = usable_h - base_h
        if extra_h < 0:
            return False
        offsets = _random_partition(extra_h, len(children) + 1)
        y = r0 + 2 + offsets[0]
        for i, child in enumerate(children):
            room_w = usable_w - int(child["w"])
            if room_w < 0:
                return False
            x = c0 + 2 + random.randint(0, room_w)
            child["bbox"] = (y, x, y + int(child["h"]) - 1, x + int(child["w"]) - 1)
            y += int(child["h"])
            if i < len(children) - 1:
                y += 1 + offsets[i + 1]

    bboxes = [child["bbox"] for child in children]
    for i, box_a in enumerate(bboxes):
        if not (r0 + 1 < box_a[0] and box_a[2] < r1 - 1 and c0 + 1 < box_a[1] and box_a[3] < c1 - 1):
            return False
        for j in range(i + 1, len(bboxes)):
            if _boxes_overlap(box_a, bboxes[j], pad=1):
                return False
    return all(_place_children(child) for child in children)


def _place_roots(roots: list[dict], shape: tuple[int, int]) -> bool:
    H, W = shape
    placed: list[tuple[int, int, int, int]] = []
    for root in sorted(roots, key=lambda node: int(node["h"]) * int(node["w"]), reverse=True):
        h = int(root["h"])
        w = int(root["w"])
        if h > H or w > W:
            return False
        for _ in range(250):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap(bbox, other, pad=ROOT_PADDING) for other in placed):
                continue
            root["bbox"] = bbox
            placed.append(bbox)
            break
        else:
            return False
    return True


def _leaf_records(records: list[dict]) -> list[dict]:
    return [record for record in records if not record["children"]]


def _same_structure(a_records: list[dict], b_records: list[dict]) -> bool:
    if len(a_records) != len(b_records):
        return False
    a_map = {record["bbox"]: record for record in a_records}
    b_map = {record["bbox"]: record for record in b_records}
    if set(a_map) != set(b_map):
        return False
    for bbox, record_a in a_map.items():
        record_b = b_map[bbox]
        parent_a = None if record_a["parent"] is None else a_records[record_a["parent"]]["bbox"]
        parent_b = None if record_b["parent"] is None else b_records[record_b["parent"]]["bbox"]
        children_a = sorted(a_records[idx]["bbox"] for idx in record_a["children"])
        children_b = sorted(b_records[idx]["bbox"] for idx in record_b["children"])
        if record_a["depth"] != record_b["depth"] or parent_a != parent_b or children_a != children_b:
            return False
    return True


def _validate_semantics(grid: np.ndarray, expected_records: list[dict]) -> bool:
    try:
        parsed = _extract_frames(grid)
        if not _same_structure(expected_records, parsed):
            return False
        if not any(record["children"] for record in parsed):
            return False

        leaves = _leaf_records(parsed)
        leaf_depths = {int(record["depth"]) for record in leaves}
        if len(leaf_depths) < 2:
            return False

        sizes_by_depth: dict[int, set[tuple[int, int]]] = {}
        for record in leaves:
            r0, c0, r1, c1 = record["bbox"]
            size = (r1 - r0 - 1, c1 - c0 - 1)
            seen = sizes_by_depth.setdefault(int(record["depth"]), set())
            if size in seen:
                return False
            seen.add(size)

        expected = _render_from_records(grid.shape, expected_records)
        actual = main(grid)
        if np.array_equal(actual, grid) or not np.array_equal(actual, expected):
            return False

        center_colors = {}
        for record in parsed:
            r0, c0, r1, c1 = record["bbox"]
            depth = int(record["depth"])
            inner = actual[r0 + 1 : r1, c0 + 1 : c1]
            if not record["children"]:
                target = _center_distance_fill(inner.shape, depth)
                if not np.array_equal(inner, target):
                    return False
                center = ((inner.shape[0] - 1) // 2, (inner.shape[1] - 1) // 2)
                center_colors.setdefault(depth, set()).add(int(inner[center]))
            else:
                mask = np.ones(inner.shape, dtype=bool)
                for child_idx in record["children"]:
                    cr0, cc0, cr1, cc1 = parsed[child_idx]["bbox"]
                    mask[
                        cr0 - (r0 + 1) : cr1 - (r0 + 1) + 1,
                        cc0 - (c0 + 1) : cc1 - (c0 + 1) + 1,
                    ] = False
                if not np.any(mask):
                    return False
                if not np.all(inner[mask] == int(PALETTE[(depth - 1) % len(PALETTE)])):
                    return False

        if len({next(iter(colors)) for colors in center_colors.values()}) != len(center_colors):
            return False
        return True
    except AssertionError:
        return False


def _attempt_generate() -> np.ndarray | None:
    forest_spec = random.choice(FOREST_TEMPLATES)
    roots = [_make_node(spec) for spec in forest_spec]
    if random.random() < 0.45:
        roots.append(_make_node(()))

    used_leaf_sizes: dict[int, set[tuple[int, int]]] = {}
    for root in roots:
        if not _assign_sizes(root, 1, used_leaf_sizes):
            return None

    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    if not _place_roots(roots, (H, W)):
        return None
    if not all(_place_children(root) for root in roots):
        return None

    grid = np.zeros((H, W), dtype=int)
    for record in _flatten_nodes(roots):
        _draw_frame(grid, record["bbox"])

    expected_records = _flatten_nodes(roots)
    if not _validate_semantics(grid, expected_records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth palette frame puzzle")
