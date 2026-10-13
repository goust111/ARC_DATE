import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BG = 0
_SLOT_NAMES = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
)
_QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_regions(mask: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _components(~cropped):
        rr = comp[:, 0]
        cc = comp[:, 1]
        touches_border = np.any((rr == 0) | (rr == cropped.shape[0] - 1) | (cc == 0) | (cc == cropped.shape[1] - 1))
        if not touches_border:
            holes.append(comp + np.array([r0, c0], dtype=int))
    return holes


def _parse_panel(panel: np.ndarray) -> dict:
    colors, counts = np.unique(panel[panel != _BG], return_counts=True)
    assert len(colors) >= 1
    main_color = int(colors[np.argmax(counts)])
    mask = panel == main_color
    comps = _components(mask)
    assert len(comps) == 1, "each panel must contain one 4-connected main shape"

    holes = _hole_regions(mask)
    assert holes, "each shape must contain at least one hole"

    extras = np.argwhere((panel != _BG) & (panel != main_color))
    assert extras.shape[0] <= 1, "each panel may contain at most one pointer pixel"

    pointer_color = None
    pointer_coord = None
    if extras.shape[0] == 1:
        pointer_coord = tuple(map(int, extras[0]))
        pointer_color = int(panel[pointer_coord])
        assert any(np.any(np.all(hole == np.array(pointer_coord), axis=1)) for hole in holes), "pointer must lie inside a hole"

    return {
        "color": main_color,
        "mask": mask,
        "holes": holes,
        "pointer_color": pointer_color,
        "pointer_coord": pointer_coord,
    }


def _chain_order(infos: list[dict]) -> list[int]:
    color_to_idx = {info["color"]: i for i, info in enumerate(infos)}
    assert len(color_to_idx) == 4, "panel colors must be distinct"

    outgoing = [None] * 4
    indegree = [0] * 4
    for i, info in enumerate(infos):
        pointer_color = info["pointer_color"]
        if pointer_color is None:
            continue
        assert pointer_color in color_to_idx, "pointer color must reference another panel color"
        j = color_to_idx[pointer_color]
        assert j != i, "pointer may not target its own panel"
        assert outgoing[i] is None, "each panel may have at most one outgoing pointer"
        outgoing[i] = j
        indegree[j] += 1

    assert sum(dst is not None for dst in outgoing) == 3, "exactly three pointers are required"
    assert indegree.count(0) == 1 and indegree.count(1) == 3, "pointers must form one chain"
    head = indegree.index(0)

    order = []
    cur = head
    seen = set()
    while cur is not None:
        assert cur not in seen, "chain may not cycle"
        seen.add(cur)
        order.append(cur)
        cur = outgoing[cur]
    assert len(order) == 4, "the chain must visit all four panels"
    assert len(seen) == 4
    return order


def _composite(infos: list[dict], order: list[int]) -> np.ndarray:
    h, w = infos[0]["mask"].shape
    out = np.full((h, w), _BG, dtype=int)
    for idx in order:
        info = infos[idx]
        out[info["mask"]] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the hidden hole pointers and overlay the four panel shapes in chain order to form one panel.

    Input structure:
    - The input is a 2x2 array of equal-size panels on a shared black background.
    - Each panel contains one unique-color 4-connected shape with one or more enclosed holes, and some holes may contain a single foreign-color pointer pixel.
    - Exactly three panels contain one pointer pixel whose color matches another panel's shape color; the fourth panel is the chain tail and has no pointer.

    Rule:
    - Split the grid into its four equal quadrants and identify each panel's main shape color.
    - Interpret each hole pointer as "this panel is followed by the panel whose shape has the same color", then find the unique shape that is pointed to by nobody and follow the full chain.
    - Create an output canvas the size of one panel and paint the four main shapes in chain order, drawing only the shape pixels and ignoring the pointer pixels; later shapes overwrite earlier ones.

    Key insight:
    - The overlay depth is encoded only by the foreign-color pixels hidden inside holes, so the correct occlusion order comes from reading the pointer chain rather than from panel position, color id, or hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0
    assert _most_common_color(g) == _BG, "background must be black"

    ph, pw = h // 2, w // 2
    panels = [
        g[0:ph, 0:pw],
        g[0:ph, pw:w],
        g[ph:h, 0:pw],
        g[ph:h, pw:w],
    ]
    infos = [_parse_panel(panel) for panel in panels]
    order = _chain_order(infos)
    return _composite(infos, order)


def _ring_block() -> np.ndarray:
    block = np.ones((3, 3), dtype=bool)
    block[1, 1] = False
    return block


def _slot_bboxes(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    r0, c0 = 3, 3
    r1, c1 = h - 4, w - 4
    return {
        "top_left": (r0 - 3, c0, r0 - 1, c0 + 2),
        "top_right": (r0 - 3, c1 - 2, r0 - 1, c1),
        "bottom_left": (r1 + 1, c0, r1 + 3, c0 + 2),
        "bottom_right": (r1 + 1, c1 - 2, r1 + 3, c1),
        "left_top": (r0, c0 - 3, r0 + 2, c0 - 1),
        "left_bottom": (r1 - 2, c0 - 3, r1, c0 - 1),
        "right_top": (r0, c1 + 1, r0 + 2, c1 + 3),
        "right_bottom": (r1 - 2, c1 + 1, r1, c1 + 3),
    }


def _paint_block(mask: np.ndarray, bbox: tuple[int, int, int, int], ring: bool) -> None:
    r0, c0, r1, c1 = bbox
    block = _ring_block() if ring else np.ones((3, 3), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = np.maximum(mask[r0 : r1 + 1, c0 : c1 + 1], block)


def _build_masks(h: int, w: int, layer_order: list[int]) -> tuple[list[np.ndarray], dict]:
    masks = [np.zeros((h, w), dtype=bool) for _ in range(4)]
    core_r0, core_c0 = 3, 3
    core_r1, core_c1 = h - 4, w - 4
    hole_r0, hole_c0 = core_r0 + 2, core_c0 + 2
    hole_r1, hole_c1 = core_r1 - 2, core_c1 - 2
    assert hole_r0 <= hole_r1 and hole_c0 <= hole_c1

    for mask in masks:
        mask[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1] = True
        mask[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1] = False

    slot_names = list(_SLOT_NAMES)
    random.shuffle(slot_names)
    exclusive_slots = {layer_order[i]: slot_names[i] for i in range(4)}
    witness_slots = {
        (layer_order[0], layer_order[1]): slot_names[4],
        (layer_order[1], layer_order[2]): slot_names[5],
        (layer_order[2], layer_order[3]): slot_names[6],
    }
    slot_boxes = _slot_bboxes(h, w)
    ring_slots: set[str] = set()

    for panel_idx, slot_name in exclusive_slots.items():
        ring = random.random() < 0.5
        _paint_block(masks[panel_idx], slot_boxes[slot_name], ring)
        if ring:
            ring_slots.add(slot_name)

    for pair, slot_name in witness_slots.items():
        ring = random.random() < 0.45
        for panel_idx in pair:
            _paint_block(masks[panel_idx], slot_boxes[slot_name], ring)
        if ring:
            ring_slots.add(slot_name)

    return masks, {
        "exclusive_slots": exclusive_slots,
        "witness_slots": witness_slots,
        "ring_slots": ring_slots,
    }


def _pick_pointer_cells(masks: list[np.ndarray], layer_order: list[int]) -> dict[int, tuple[int, int]]:
    pointer_cells = {}
    for src in layer_order[:-1]:
        holes = _hole_regions(masks[src])
        assert holes
        chosen_hole = random.choice(holes)
        coord = tuple(map(int, random.choice(chosen_hole.tolist())))
        pointer_cells[src] = coord
    return pointer_cells


def _render_input(
    masks: list[np.ndarray],
    panel_colors: list[int],
    layer_order: list[int],
    pointer_cells: dict[int, tuple[int, int]],
) -> np.ndarray:
    h, w = masks[0].shape
    grid = np.full((2 * h, 2 * w), _BG, dtype=int)
    target_by_src = {layer_order[i]: layer_order[i + 1] for i in range(3)}
    for panel_idx, (qr, qc) in enumerate(_QUADRANTS):
        panel = np.full((h, w), _BG, dtype=int)
        panel[masks[panel_idx]] = int(panel_colors[panel_idx])
        if panel_idx in pointer_cells:
            pr, pc = pointer_cells[panel_idx]
            panel[pr, pc] = int(panel_colors[target_by_src[panel_idx]])
        grid[qr * h : (qr + 1) * h, qc * w : (qc + 1) * w] = panel
    return grid


def _parse_rendered_infos(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    ph, pw = h // 2, w // 2
    panels = [
        grid[0:ph, 0:pw],
        grid[0:ph, pw:w],
        grid[ph:h, 0:pw],
        grid[ph:h, pw:w],
    ]
    return [_parse_panel(panel) for panel in panels]


def _all_pairwise_overlap(masks: list[np.ndarray]) -> bool:
    for i in range(4):
        for j in range(i + 1, 4):
            if not np.any(masks[i] & masks[j]):
                return False
    return True


def _order_changes_matter(infos: list[dict], order: list[int]) -> bool:
    expected = _composite(infos, order)
    for k in range(3):
        alt = order.copy()
        alt[k], alt[k + 1] = alt[k + 1], alt[k]
        if np.array_equal(expected, _composite(infos, alt)):
            return False

    naive_orders = [
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        sorted(range(4), key=lambda idx: infos[idx]["color"]),
        sorted(range(4), key=lambda idx: infos[idx]["color"], reverse=True),
    ]
    return all(not np.array_equal(expected, _composite(infos, alt)) for alt in naive_orders if alt != order)


def _valid_generated_scene(grid: np.ndarray, intended_order: list[int]) -> bool:
    try:
        infos = _parse_rendered_infos(grid)
        solved_order = _chain_order(infos)
    except Exception:
        return False

    if solved_order != intended_order:
        return False
    if not _all_pairwise_overlap([info["mask"] for info in infos]):
        return False

    out = _composite(infos, solved_order)
    if len(set(map(int, out[out != _BG]))) != 4:
        return False
    if not _order_changes_matter(infos, solved_order):
        return False

    for info in infos:
        extras = np.argwhere((info["mask"] == 0) & (info["pointer_coord"] is not None))
        _ = extras
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _attempt() -> np.ndarray | None:
    h = random.randint(12, 14)
    w = random.randint(12, 14)
    panel_colors = random.sample(list(range(1, 10)), 4)
    layer_order = random.sample(range(4), 4)
    masks, _meta = _build_masks(h, w, layer_order)
    pointer_cells = _pick_pointer_cells(masks, layer_order)
    grid = _render_input(masks, panel_colors, layer_order, pointer_cells)
    return grid if _valid_generated_scene(grid, layer_order) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-pointer-chain input")
