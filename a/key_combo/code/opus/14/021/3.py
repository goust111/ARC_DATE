from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
ORANGE = 7

BOX = 4
SUPPORT_RELS = ((0, 0), (0, 3), (3, 0), (3, 3))

# A 4x4 connected stamp whose four quarter-turns are all distinct and which never
# touches a corner, so corner supports remain valid under every rotation.
BASE_RED_MASK = np.array(
    [
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
    ],
    dtype=bool,
)

CORNER_LAYOUTS = {
    "tl": {
        "support": (0, 0),
        "diag": [(-1, -1)],
        "orth": [(-1, 0), (0, -1)],
        "marker_boxes": [(-2, 0), (0, -2)],
    },
    "tr": {
        "support": (0, 3),
        "diag": [(-1, 1)],
        "orth": [(-1, 0), (0, 1)],
        "marker_boxes": [(-2, 2), (0, 4)],
    },
    "br": {
        "support": (3, 3),
        "diag": [(1, 1)],
        "orth": [(1, 0), (0, 1)],
        "marker_boxes": [(4, 2), (2, 4)],
    },
    "bl": {
        "support": (3, 0),
        "diag": [(1, -1)],
        "orth": [(1, 0), (0, -1)],
        "marker_boxes": [(4, 0), (2, -2)],
    },
}

# Missing-corner encoding inside a 2x2 orange L marker.
# We interpret the missing corner in clockwise order as up/right/down/left.
ROT_TO_MISSING = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
}
MISSING_TO_ROT = {v: k for k, v in ROT_TO_MISSING.items()}

GEN_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 24
SLOT_COUNT_RANGE = (4, 5)
REACHABLE_RANGE = (2, 4)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    dirs = _neighbors(connectivity)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            labels[r, c] = next_label
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
    return labels, next_label


def _rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask.astype(int), -int(turns)).astype(bool)


def _marker_cells(top_left: tuple[int, int], turns: int) -> list[tuple[int, int]]:
    mr, mc = top_left
    missing = ROT_TO_MISSING[int(turns)]
    cells = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.append((mr + dr, mc + dc))
    return cells


def _decode_marker(component: np.ndarray) -> tuple[tuple[int, int, int, int], int] | None:
    if component.shape[0] != 3:
        return None
    coords = {(int(r), int(c)) for r, c in component}
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) != 2 or len(cols) != 2:
        return None
    r0, r1 = rows
    c0, c1 = cols
    missing = None
    for dr in range(2):
        for dc in range(2):
            cell = (r0 + dr, c0 + dc)
            if cell not in coords:
                if missing is not None:
                    return None
                missing = (dr, dc)
    if missing not in MISSING_TO_ROT:
        return None
    return (r0, c0, r1, c1), MISSING_TO_ROT[missing]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = 0
    if ar1 < br0:
        dr = br0 - ar1 - 1
    elif br1 < ar0:
        dr = ar0 - br1 - 1
    dc = 0
    if ac1 < bc0:
        dc = bc0 - ac1 - 1
    elif bc1 < ac0:
        dc = ac0 - bc1 - 1
    return max(dr, dc)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    red_box = _bbox(g == RED)
    assert red_box is not None, "expected a red template"
    tr0, tc0, tr1, tc1 = red_box
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    red_mask = template == RED
    support_mask = template == GREY
    h, w = template.shape
    slot_signature = np.where(support_mask, GREY, BG)

    candidates: list[tuple[int, int]] = []
    for r0 in range(g.shape[0] - h + 1):
        for c0 in range(g.shape[1] - w + 1):
            if np.array_equal(g[r0 : r0 + h, c0 : c0 + w], slot_signature):
                candidates.append((r0, c0))

    orange_labels, orange_n = _label_components(g == ORANGE, connectivity=4)
    markers = []
    for label in range(1, orange_n + 1):
        comp = np.argwhere(orange_labels == label)
        decoded = _decode_marker(comp)
        if decoded is not None:
            bbox, turns = decoded
            markers.append({"bbox": bbox, "turns": turns, "cells": comp})

    grey_labels, _ = _label_components(g == GREY, connectivity=8)
    source_labels = {
        int(grey_labels[tr0 + dr, tc0 + dc])
        for dr, dc in np.argwhere(support_mask)
        if int(grey_labels[tr0 + dr, tc0 + dc]) != 0
    }

    reachable: list[dict] = []
    unreachable: list[dict] = []
    for r0, c0 in candidates:
        support_cells = [(r0 + dr, c0 + dc) for dr, dc in np.argwhere(support_mask)]
        labels = {int(grey_labels[r, c]) for r, c in support_cells if int(grey_labels[r, c]) != 0}
        slot = {"origin": (r0, c0), "labels": labels}
        if labels & source_labels:
            reachable.append(slot)
        else:
            unreachable.append(slot)

    all_slots = reachable + unreachable
    for slot in all_slots:
        sr0, sc0 = slot["origin"]
        sb = (sr0, sc0, sr0 + h - 1, sc0 + w - 1)
        nearby = [m for m in markers if _bbox_gap(sb, m["bbox"]) <= 2]
        assert len(nearby) == 1, "each slot should have one nearby marker"
        slot["marker_turns"] = int(nearby[0]["turns"])

    return {
        "template_bbox": red_box,
        "template": template,
        "red_mask": red_mask,
        "support_mask": support_mask,
        "slot_signature": slot_signature,
        "reachable": reachable,
        "unreachable": unreachable,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the template through the grey network and stamp each reachable matching slot with the marker-indicated rotation.

    Input structure:
    - The grid has black background, one red template sprite inside a square support window, several empty grey support slots of the same window shape, grey path cells, and one orange 3-cell L marker near each slot.
    - The template window contains red stamp cells and grey support cells; each true empty slot contains the same grey support signature with black where the template is red.
    - Only some slots are 8-connected to the template through the grey path network.

    Rule:
    - Find the template window from the red sprite and derive both its red stamp mask and its grey support signature.
    - Scan for empty slot windows whose entire contents exactly match the template window after replacing template-red cells by black.
    - Keep only the matching slots whose grey support cells lie in an 8-connected grey component that also contains template support.
    - Decode the nearby orange 2x2 L marker by its missing corner: top-left/up = 0 degrees, top-right = 90 degrees clockwise, bottom-right = 180 degrees, bottom-left = 270 degrees clockwise.
    - Stamp the correspondingly rotated red mask into each reachable slot, then erase all grey path cells and all orange markers while preserving the template and every grey support structure.

    Key insight:
    - A slot is valid only when three independent tests agree: its support window matches the template signature, the grey network actually reaches that slot, and the local orange L orientation chooses which rotation to stamp there.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    red_mask = info["red_mask"]
    support_mask = info["support_mask"]
    tr0, tc0, tr1, tc1 = info["template_bbox"]
    h, w = red_mask.shape

    out = g.copy()
    out[out == ORANGE] = BG

    preserve_grey = np.zeros(g.shape, dtype=bool)
    for dr, dc in np.argwhere(support_mask):
        preserve_grey[tr0 + dr, tc0 + dc] = True
    for slot in info["reachable"] + info["unreachable"]:
        sr0, sc0 = slot["origin"]
        for dr, dc in np.argwhere(support_mask):
            preserve_grey[sr0 + dr, sc0 + dc] = True
    out[(out == GREY) & (~preserve_grey)] = BG

    for slot in info["reachable"]:
        sr0, sc0 = slot["origin"]
        rotated = _rotate_mask_cw(red_mask, slot["marker_turns"])
        for dr, dc in np.argwhere(rotated):
            out[sr0 + dr, sc0 + dc] = RED

    return out


def _paint_patch(grid: np.ndarray, top_left: tuple[int, int], red_mask: np.ndarray, empty: bool) -> None:
    r0, c0 = top_left
    for dr, dc in SUPPORT_RELS:
        grid[r0 + dr, c0 + dc] = GREY
    if not empty:
        for dr, dc in np.argwhere(red_mask):
            grid[r0 + dr, c0 + dc] = RED


def _support_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr, dc in SUPPORT_RELS]


def _box_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(BOX) for dc in range(BOX)}


def _separated(a: tuple[int, int], b: tuple[int, int], gap: int = 3) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + BOX - 1 + gap < br
        or br + BOX - 1 + gap < ar
        or ac + BOX - 1 + gap < bc
        or bc + BOX - 1 + gap < ac
    )


def _pick_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    positions = [(r, c) for r in range(2, h - 5) for c in range(2, w - 5)]
    for _ in range(200):
        random.shuffle(positions)
        picked: list[tuple[int, int]] = []
        for pos in positions:
            if all(_separated(pos, old) for old in picked):
                picked.append(pos)
                if len(picked) == count:
                    return picked
    return None


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _choose_layout(
    shape: tuple[int, int],
    box_top_left: tuple[int, int],
    rotation: int,
    reserved_boxes: set[tuple[int, int]],
    require_diag: bool | None,
) -> dict | None:
    r0, c0 = box_top_left
    corner_names = list(CORNER_LAYOUTS)
    random.shuffle(corner_names)
    for name in corner_names:
        layout = CORNER_LAYOUTS[name]
        sr, sc = layout["support"]
        endpoint_offsets = list(layout["diag"])
        if require_diag is not True:
            endpoint_offsets += layout["orth"]
        if require_diag is False:
            endpoint_offsets = list(layout["orth"])
        random.shuffle(endpoint_offsets)
        marker_boxes = list(layout["marker_boxes"])
        random.shuffle(marker_boxes)
        for edr, edc in endpoint_offsets:
            endpoint = (r0 + sr + edr, c0 + sc + edc)
            if not _inside(shape, *endpoint):
                continue
            if endpoint in reserved_boxes:
                continue
            for mdr, mdc in marker_boxes:
                marker_tl = (r0 + mdr, c0 + mdc)
                mr, mc = marker_tl
                marker_box = {(mr + dr, mc + dc) for dr in range(2) for dc in range(2)}
                if not all(_inside(shape, rr, cc) for rr, cc in marker_box):
                    continue
                if marker_box & reserved_boxes:
                    continue
                if endpoint in marker_box:
                    continue
                orange_cells = set(_marker_cells(marker_tl, rotation))
                if orange_cells & reserved_boxes:
                    continue
                return {
                    "corner": name,
                    "support": (r0 + sr, c0 + sc),
                    "endpoint": endpoint,
                    "diag": abs(edr) == 1 and abs(edc) == 1,
                    "marker_top_left": marker_tl,
                    "marker_cells": orange_cells,
                    "marker_box": marker_box,
                }
    return None


def _choose_template_endpoint(
    shape: tuple[int, int],
    box_top_left: tuple[int, int],
    reserved_boxes: set[tuple[int, int]],
    require_diag: bool | None,
) -> dict | None:
    r0, c0 = box_top_left
    corner_names = list(CORNER_LAYOUTS)
    random.shuffle(corner_names)
    for name in corner_names:
        layout = CORNER_LAYOUTS[name]
        sr, sc = layout["support"]
        endpoint_offsets = list(layout["diag"])
        if require_diag is not True:
            endpoint_offsets += layout["orth"]
        if require_diag is False:
            endpoint_offsets = list(layout["orth"])
        random.shuffle(endpoint_offsets)
        for edr, edc in endpoint_offsets:
            endpoint = (r0 + sr + edr, c0 + sc + edc)
            if _inside(shape, *endpoint) and endpoint not in reserved_boxes:
                return {
                    "corner": name,
                    "support": (r0 + sr, c0 + sc),
                    "endpoint": endpoint,
                    "diag": abs(edr) == 1 and abs(edc) == 1,
                }
    return None


def _route_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing: set[tuple[int, int]],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    q = deque(existing)
    parents: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in existing}
    dirs = list(_neighbors(8))
    random.shuffle(dirs)
    while q:
        cur = q.popleft()
        if cur == target:
            path = []
            node = cur
            while node is not None and node not in existing:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in parents:
                continue
            if nxt in blocked and nxt != target:
                continue
            parents[nxt] = cur
            q.append(nxt)
    return None


def _expected_output(
    grid: np.ndarray,
    red_mask: np.ndarray,
    template_pos: tuple[int, int],
    slots: list[dict],
    path_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in path_cells | marker_cells:
        out[r, c] = BG
    for slot in slots:
        if not slot["reachable"]:
            continue
        rotated = _rotate_mask_cw(red_mask, slot["rotation"])
        r0, c0 = slot["pos"]
        for dr, dc in np.argwhere(rotated):
            out[r0 + dr, c0 + dc] = RED
    return out


def _reachable_slots_by_labels(
    grid: np.ndarray,
    template_pos: tuple[int, int],
    slots: list[dict],
    connectivity: int,
) -> set[int]:
    labels, _ = _label_components(grid == GREY, connectivity=connectivity)
    tr0, tc0 = template_pos
    source = {
        int(labels[tr0 + dr, tc0 + dc])
        for dr, dc in SUPPORT_RELS
        if int(labels[tr0 + dr, tc0 + dc]) != 0
    }
    ans = set()
    for idx, slot in enumerate(slots):
        r0, c0 = slot["pos"]
        slot_labels = {
            int(labels[r0 + dr, c0 + dc])
            for dr, dc in SUPPORT_RELS
            if int(labels[r0 + dr, c0 + dc]) != 0
        }
        if slot_labels & source:
            ans.add(idx)
    return ans


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    slot_count = random.randint(*SLOT_COUNT_RANGE)

    red_mask = BASE_RED_MASK.copy()
    if random.random() < 0.5:
        red_mask = np.fliplr(red_mask)
    if random.random() < 0.5:
        red_mask = np.flipud(red_mask)
    red_mask = _rotate_mask_cw(red_mask, random.randint(0, 3))

    positions = _pick_positions(h, w, slot_count + 1)
    if positions is None:
        return None
    random.shuffle(positions)
    template_pos = positions[0]
    slot_positions = positions[1:]

    max_reachable = min(REACHABLE_RANGE[1], slot_count - 1)
    reachable_count = random.randint(REACHABLE_RANGE[0], max_reachable)
    reachable_indices = set(random.sample(range(slot_count), reachable_count))
    reachable_rotations = random.sample(range(4), reachable_count)
    slot_data = []
    reach_i = 0
    for i, pos in enumerate(slot_positions):
        is_reachable = i in reachable_indices
        if is_reachable:
            rotation = reachable_rotations[reach_i]
            reach_i += 1
        else:
            rotation = random.randint(0, 3)
        slot_data.append({"pos": pos, "rotation": rotation, "reachable": is_reachable})

    _paint_patch(grid, template_pos, red_mask, empty=False)
    for slot in slot_data:
        _paint_patch(grid, slot["pos"], red_mask, empty=True)

    reserved_boxes: set[tuple[int, int]] = set()
    for pos in [template_pos] + slot_positions:
        reserved_boxes |= _box_cells(pos)

    diag_needed = True
    template_layout = _choose_template_endpoint((h, w), template_pos, reserved_boxes, require_diag=None)
    if template_layout is None:
        return None

    marker_cells: set[tuple[int, int]] = set()
    for slot in slot_data:
        want_diag = True if diag_needed and slot["reachable"] else None
        layout = _choose_layout((h, w), slot["pos"], slot["rotation"], reserved_boxes, want_diag)
        if layout is None:
            layout = _choose_layout((h, w), slot["pos"], slot["rotation"], reserved_boxes, None)
        if layout is None:
            return None
        slot["layout"] = layout
        reserved_boxes |= layout["marker_box"]
        marker_cells |= layout["marker_cells"]
        if slot["reachable"] and layout["diag"]:
            diag_needed = False

    if diag_needed and not template_layout["diag"]:
        template_layout = _choose_template_endpoint((h, w), template_pos, reserved_boxes, require_diag=True)
        if template_layout is None:
            return None

    blocked = set(reserved_boxes)
    support_halo = set()
    all_supports = []
    for pos in [template_pos] + slot_positions:
        all_supports.extend(_support_cells(pos))
    for sr, sc in all_supports:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = sr + dr, sc + dc
                if _inside((h, w), rr, cc) and (rr, cc) not in reserved_boxes:
                    support_halo.add((rr, cc))
    blocked |= support_halo

    path_cells = {template_layout["endpoint"]}
    blocked.discard(template_layout["endpoint"])
    for slot in slot_data:
        if slot["reachable"]:
            blocked.discard(slot["layout"]["endpoint"])

    targets = [slot["layout"]["endpoint"] for slot in slot_data if slot["reachable"]]
    random.shuffle(targets)
    for target in targets:
        segment = _route_path((h, w), blocked, path_cells, target)
        if segment is None:
            return None
        path_cells.update(segment)

    for r, c in path_cells:
        grid[r, c] = GREY
    for r, c in marker_cells:
        grid[r, c] = ORANGE

    expected = _expected_output(grid, red_mask, template_pos, slot_data, path_cells, marker_cells)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    analysis = _analyze(grid)
    if len(analysis["reachable"]) != reachable_count or len(analysis["unreachable"]) != slot_count - reachable_count:
        return None
    if len(analysis["reachable"]) < 2 or len(analysis["unreachable"]) < 1:
        return None

    reachable8 = _reachable_slots_by_labels(grid, template_pos, slot_data, connectivity=8)
    reachable4 = _reachable_slots_by_labels(grid, template_pos, slot_data, connectivity=4)
    if reachable8 != reachable_indices:
        return None
    if reachable4 == reachable8:
        return None

    local_outputs = set()
    for slot in slot_data:
        if not slot["reachable"]:
            continue
        patch = expected[slot["pos"][0] : slot["pos"][0] + BOX, slot["pos"][1] : slot["pos"][1] + BOX]
        local_outputs.add(tuple(map(int, patch.ravel())))
    if len(local_outputs) != len(reachable8):
        return None

    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
