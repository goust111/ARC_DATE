from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 600


def _bbox_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _global_mask(
    shape: tuple[int, int], top: int, left: int, mask: np.ndarray
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def _complete_visible_object(g: np.ndarray, comp: dict) -> dict:
    color = int(comp["color"])
    box = tuple(map(int, comp["bbox"]))
    visible = crop_bbox(g == color, box)
    assert visible.any(), "empty visible object"
    assert visible[:, 0].any() and visible[:, -1].any(), "shape side columns must remain visible"
    full = visible | np.fliplr(visible)
    return {"color": color, "bbox": box, "visible": visible, "full": full}


def _depth_edges(g: np.ndarray, objects: list[dict], bg: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            inter = _bbox_intersection(a["bbox"], b["bbox"])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            a_r0, a_c0, _a_r1, _a_c1 = a["bbox"]
            b_r0, b_c0, _b_r1, _b_c1 = b["bbox"]
            mask_a = a["full"][r0 - a_r0 : r1 - a_r0 + 1, c0 - a_c0 : c1 - a_c0 + 1]
            mask_b = b["full"][r0 - b_r0 : r1 - b_r0 + 1, c0 - b_c0 : c1 - b_c0 + 1]
            overlap = mask_a & mask_b
            if not overlap.any():
                continue
            seen = set(map(int, np.unique(g[r0 : r1 + 1, c0 : c1 + 1][overlap])))
            seen.discard(int(bg))
            assert seen <= {a["color"], b["color"]}, "overlap must show one of the two colors"
            assert len(seen) == 1, "pair overlap must reveal a unique front color"
            top_color = next(iter(seen))
            if top_color == a["color"]:
                edges.add((b["color"], a["color"]))
            else:
                edges.add((a["color"], b["color"]))
    return edges


def _unique_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for a, b in edges:
        if b not in outgoing[a]:
            outgoing[a].add(b)
            indeg[b] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        roots = sorted(color for color in remaining if indeg[color] == 0)
        assert len(roots) == 1, "depth constraints must determine a unique total order"
        cur = roots[0]
        order.append(cur)
        remaining.remove(cur)
        for nxt in outgoing[cur]:
            indeg[nxt] -= 1
    return order


def _render_output(objects: list[dict], order: list[int]) -> np.ndarray:
    by_color = {obj["color"]: obj for obj in objects}
    parts = [by_color[color]["full"] for color in order]
    height = max(mask.shape[0] for mask in parts)
    width = sum(mask.shape[1] for mask in parts)
    out = np.zeros((height, width), dtype=int)
    c0 = 0
    for color, mask in zip(order, parts):
        h, w = mask.shape
        r0 = height - h
        patch = out[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = int(color)
        c0 += w
    return out


def _solve(g: np.ndarray, *, return_debug: bool = False):
    bg = BG
    assert np.any(g == bg), "expected black background cells"
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(comps) >= 2, "expected multiple colored symmetric shapes"
    objects = [_complete_visible_object(g, comp) for comp in comps]
    colors = [obj["color"] for obj in objects]
    assert len(set(colors)) == len(colors), "shape colors must be unique"
    edges = _depth_edges(g, objects, bg)
    assert len(edges) == len(objects) - 1, "chain overlaps should yield a path of depth edges"
    order = _unique_order(colors, edges)
    out = _render_output(objects, order)
    if return_debug:
        return out, {"objects": objects, "edges": edges, "order": order}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially hidden bilateral shape, sort the shapes by burial depth, and redraw them side by side.

    Input structure:
    - The grid has a black background and several single-color foreground shapes.
    - Each foreground shape is left-right symmetric about its own vertical axis, but chain-like overlaps hide part of deeper shapes.
    - In every overlap, only the shallower color is visible, and every hidden cell has its horizontal mirror cell still visible.

    Rule:
    - Treat each visible same-color component as one shape and infer its full bounding box from the visible extent.
    - Complete each shape by reflecting its visible mask across the vertical axis of its own bbox.
    - For every pair of completed shapes that overlap, use the visible color inside the overlap to derive which shape is shallower, recover the unique order from deepest to shallowest, and place the completed shapes left to right with their bottoms aligned.

    Key insight:
    - The overlaps encode depth order, while the missing geometry is recovered independently from each shape's own bilateral symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    return _solve(g)


def _centered_row(width: int, span: int) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    left = (width - span) // 2
    row[left : left + span] = True
    return row


def _random_symmetric_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    min_span = 1 if w % 2 else 2
    top_full = rng.randint(0, min(1, h - 1))
    bottom_full = rng.randint(max(h // 2, top_full + 1), h - 1)
    max_shrink = (w - min_span) // 2
    spans: list[int] = []
    for r in range(h):
        dist = min(abs(r - top_full), abs(r - bottom_full))
        limit = min(max_shrink, dist + rng.randint(0, 1))
        shrink = 2 * rng.randint(0, limit)
        spans.append(w - shrink)
    spans[top_full] = w
    spans[bottom_full] = w
    if len(set(spans)) == 1:
        tweak_rows = [r for r in range(h) if r not in {top_full, bottom_full}]
        if tweak_rows and w - 2 >= min_span:
            spans[rng.choice(tweak_rows)] = w - 2
    mask = np.zeros((h, w), dtype=bool)
    for r, span in enumerate(spans):
        mask[r] = _centered_row(w, span)
    return mask


def _make_shapes(rng: random.Random, count: int) -> list[dict] | None:
    width_ranges = {2: (6, 10), 3: (5, 8), 4: (5, 7)}
    low_w, high_w = width_ranges[count]
    widths = [rng.randint(low_w, high_w) for _ in range(count)]
    if sum(widths) > 28:
        return None
    heights = [rng.randint(6, 10) for _ in range(count)]
    colors = rng.sample(PALETTE, count)
    shapes = []
    for color, h, w in zip(colors, heights, widths):
        shapes.append({"color": int(color), "mask": _random_symmetric_mask(rng, h, w)})
    return shapes


def _side_sequence(rng: random.Random, count: int) -> list[int]:
    if count == 2:
        return [rng.choice([-1, 1])]
    if rng.random() < 0.6:
        first = rng.choice([-1, 1])
        seq = [first]
        for _ in range(count - 2):
            seq.append(-seq[-1] if rng.random() < 0.8 else seq[-1])
        return seq
    return [rng.choice([-1, 1]) for _ in range(count - 1)]


def _place_shapes(rng: random.Random, shapes: list[dict]) -> bool:
    sides = _side_sequence(rng, len(shapes))
    overlaps = []
    for i in range(len(shapes) - 1):
        w0 = shapes[i]["mask"].shape[1]
        w1 = shapes[i + 1]["mask"].shape[1]
        max_overlap = min(3, w0 // 2, w1 // 2)
        if max_overlap < 1:
            return False
        overlaps.append(rng.randint(1, max_overlap))

    shapes[0]["top"] = 0
    shapes[0]["left"] = 0
    for i in range(1, len(shapes)):
        prev = shapes[i - 1]
        cur = shapes[i]
        h0 = prev["mask"].shape[0]
        h1 = cur["mask"].shape[0]
        delta_top = rng.randint(2, min(4, h0 - 1, h1 - 1))
        cur["top"] = prev["top"] + delta_top
        ov = overlaps[i - 1]
        if sides[i - 1] > 0:
            cur["left"] = prev["left"] + prev["mask"].shape[1] - ov
        else:
            cur["left"] = prev["left"] - (cur["mask"].shape[1] - ov)
        cur["occluded_side"] = None
        prev["occluded_side"] = "right" if sides[i - 1] > 0 else "left"
    shapes[-1]["occluded_side"] = None

    min_left = min(int(shape["left"]) for shape in shapes)
    if min_left < 0:
        for shape in shapes:
            shape["left"] -= min_left
    max_bottom = max(int(shape["top"]) + shape["mask"].shape[0] for shape in shapes)
    max_right = max(int(shape["left"]) + shape["mask"].shape[1] for shape in shapes)
    if max_bottom > 24 or max_right > 24:
        return False

    boxes = []
    for shape in shapes:
        h, w = shape["mask"].shape
        box = (shape["top"], shape["left"], shape["top"] + h - 1, shape["left"] + w - 1)
        shape["bbox"] = box
        boxes.append(box)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter = _bbox_intersection(boxes[i], boxes[j])
            if inter is None:
                continue
            if j != i + 1:
                return False
    height = max(int(shape["top"]) + shape["mask"].shape[0] for shape in shapes)
    width = max(int(shape["left"]) + shape["mask"].shape[1] for shape in shapes)
    full_globals = [
        _global_mask((height, width), int(shape["top"]), int(shape["left"]), shape["mask"])
        for shape in shapes
    ]
    for i in range(len(shapes) - 1):
        if not (full_globals[i] & full_globals[i + 1]).any():
            return False
    return True


def _visible_masks(shapes: list[dict]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    height = max(int(shape["top"]) + shape["mask"].shape[0] for shape in shapes)
    width = max(int(shape["left"]) + shape["mask"].shape[1] for shape in shapes)
    full_globals = [
        _global_mask((height, width), int(shape["top"]), int(shape["left"]), shape["mask"])
        for shape in shapes
    ]
    grid = np.zeros((height, width), dtype=int)
    visible_globals: list[np.ndarray] = []
    front_union = np.zeros((height, width), dtype=bool)
    for shape, full in zip(reversed(shapes), reversed(full_globals)):
        visible_globals.append(full & ~front_union)
        front_union |= full
    visible_globals.reverse()
    for shape, visible in zip(shapes, visible_globals):
        grid[visible] = int(shape["color"])
    return grid, full_globals, visible_globals


def _expected_output(shapes: list[dict]) -> np.ndarray:
    objects = []
    for shape in shapes:
        h, w = shape["mask"].shape
        objects.append(
            {
                "color": int(shape["color"]),
                "bbox": (0, 0, h - 1, w - 1),
                "visible": shape["mask"].copy(),
                "full": shape["mask"].copy(),
            }
        )
    order = [int(shape["color"]) for shape in shapes]
    return _render_output(objects, order)


def _validate_case(inp: np.ndarray, shapes: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    expected = _expected_output(shapes)
    if max(expected.shape) > 30:
        return False
    produced, debug = _solve(inp, return_debug=True)
    if not np.array_equal(produced, expected):
        return False
    if debug["order"] != [int(shape["color"]) for shape in shapes]:
        return False

    _, full_globals, visible_globals = _visible_masks(shapes)
    for idx, shape in enumerate(shapes):
        full = full_globals[idx]
        visible = visible_globals[idx]
        hidden = full & ~visible
        top = int(shape["top"])
        left = int(shape["left"])
        h, w = shape["mask"].shape
        occ_side = shape["occluded_side"]
        if idx < len(shapes) - 1 and not hidden.any():
            return False
        if occ_side is None:
            continue
        edge_col = w - 1 if occ_side == "right" else 0
        local_visible = visible[top : top + h, left : left + w]
        if not local_visible[:, edge_col].any():
            return False
        hidden_coords = np.argwhere(hidden)
        axis = (w - 1) / 2.0
        for r, c in hidden_coords:
            local_r = int(r - top)
            local_c = int(c - left)
            if occ_side == "right" and not (local_c > axis):
                return False
            if occ_side == "left" and not (local_c < axis):
                return False
            mirror_c = w - 1 - local_c
            if not local_visible[local_r, mirror_c]:
                return False
    return True


def _generate_case(*, return_meta: bool = False):
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        count = rng.choices([2, 3, 4], weights=[1, 3, 3], k=1)[0]
        shapes = _make_shapes(rng, count)
        if shapes is None:
            continue
        if not _place_shapes(rng, shapes):
            continue
        tight, _full, _visible = _visible_masks(shapes)
        pad_top = rng.randint(0, 2)
        pad_bottom = rng.randint(0, 2)
        pad_left = rng.randint(0, 2)
        pad_right = rng.randint(0, 2)
        inp = np.zeros(
            (tight.shape[0] + pad_top + pad_bottom, tight.shape[1] + pad_left + pad_right),
            dtype=int,
        )
        inp[pad_top : pad_top + tight.shape[0], pad_left : pad_left + tight.shape[1]] = tight

        shifted = []
        for shape in shapes:
            copied = {
                "color": int(shape["color"]),
                "mask": shape["mask"].copy(),
                "top": int(shape["top"]) + pad_top,
                "left": int(shape["left"]) + pad_left,
                "occluded_side": shape["occluded_side"],
            }
            h, w = copied["mask"].shape
            copied["bbox"] = (
                copied["top"],
                copied["left"],
                copied["top"] + h - 1,
                copied["left"] + w - 1,
            )
            shifted.append(copied)

        if not _validate_case(inp, shifted):
            continue
        if return_meta:
            return inp, shifted
        return inp
    raise RuntimeError("failed to generate a valid excavation-order case")


def generate_input() -> np.ndarray:
    return _generate_case()
