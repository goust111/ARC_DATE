from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import bbox_coords, find_components, get_bg

        return bbox_coords, find_components, get_bg
    except ImportError:
        arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("_arc_utils_fallback", arc_utils_path)
        assert spec is not None and spec.loader is not None, "failed to load arc_utils"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.bbox_coords, module.find_components, module.get_bg


bbox_coords, find_components, get_bg = _load_arc_utils()

BG = 0
PIECE_COLORS = (1, 2, 3, 4)
GREY = 5
ORANGE = 7
SIDE_ROWS = (0, 1, 2)

SIDE_SHAPES = {
    6: [
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)),
    ],
}

CENTER_SHAPES = {
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ],
}

RECT_OPTIONS = [(6, 12)]
CANVAS_ROWS = (26, 28, 30)
CANVAS_COLS = (26, 28, 30)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _shape_info(coords: tuple[tuple[int, int], ...]) -> dict:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))
    return {
        "coords": rel,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": len(rel),
    }


def _third_bounds(length: int, idx: int) -> tuple[int, int]:
    start = (idx * length) // 3
    end = ((idx + 1) * length) // 3 - 1
    return int(start), int(end)


def _third_center(start: int, end: int, idx: int, axis: str) -> int:
    length = int(end - start + 1)
    lo, hi = _third_bounds(length, idx)
    if axis == "row":
        return int(start + hi)
    if idx == 1:
        return int(start + (lo + hi + 1) // 2)
    if idx == 0:
        return int(start + max(lo, hi - 1))
    if idx == 2:
        return int(start + max(lo, hi - 1))
    return int(start + (lo + hi) // 2)


def _place_local_shape(shape: dict, rect: tuple[int, int, int, int], vband: int, hband: int, hshift: int = 0) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    center_r = _third_center(r0, r1, vband, "row")
    center_c = _third_center(c0, c1, hband, "col") + int(hshift)
    top = int(np.floor(center_r - (shape["h"] - 1) / 2.0))
    left = int(np.floor(center_c - (shape["w"] - 1) / 2.0))
    return {(top + dr, left + dc) for dr, dc in shape["coords"]}


def _mirror_coords(coords: set[tuple[int, int]], rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    _r0, c0, _r1, c1 = rect
    return {(r, int(c0 + c1 - c)) for r, c in coords}


def _same_color_conflicts(instances: list[dict]) -> set[int]:
    owners = {}
    colors = {}
    for inst in instances:
        for r, c in inst["coords"]:
            key = (int(r), int(c))
            assert key not in owners, "piece overlap in assembled rectangle"
            owners[key] = int(inst["entity_id"])
            colors[key] = int(inst["color"])

    bad: set[int] = set()
    for (r, c), ent_id in owners.items():
        for nr, nc in _neighbors4(r, c):
            other = owners.get((nr, nc))
            if other is None or other == ent_id:
                continue
            if colors[(nr, nc)] == colors[(r, c)]:
                bad.add(int(ent_id))
                bad.add(int(other))
    return bad


def _build_instances(entities: list[dict], rect: tuple[int, int, int, int], shift_ids: set[int]) -> list[dict]:
    instances: list[dict] = []
    for ent in entities:
        color = int(ent.get("color", PIECE_COLORS[0]))
        hshift = 0
        if ent["kind"] == "side" and ent["id"] in shift_ids:
            hshift = -1 if ent["hband"] == 0 else 1
        original = _place_local_shape(ent["shape"], rect, ent["vband"], ent["hband"], hshift=hshift)
        if ent["kind"] == "center":
            instances.append({"entity_id": ent["id"], "color": color, "coords": original})
        else:
            mirrored = _mirror_coords(original, rect)
            assert original.isdisjoint(mirrored), "side piece must stay off the mirror axis"
            instances.append({"entity_id": ent["id"], "color": color, "coords": original})
            instances.append({"entity_id": ent["id"], "color": color, "coords": mirrored})
    return instances


def _render_instances(canvas_shape: tuple[int, int], instances: list[dict]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    H, W = canvas_shape
    for inst in instances:
        color = int(inst["color"])
        for r, c in inst["coords"]:
            assert 0 <= r < H and 0 <= c < W, "assembled piece out of bounds"
            assert out[r, c] == 0, "assembled pieces overlap"
            out[r, c] = color
    return out


def _extract_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    corners = np.argwhere(grid == ORANGE)
    assert corners.shape == (4, 2), "expected exactly four corner markers"
    r0, c0 = map(int, corners.min(axis=0))
    r1, c1 = map(int, corners.max(axis=0))
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert expected == {tuple(map(int, p)) for p in corners}, "orange markers must be rectangle corners"
    return r0, c0, r1, c1


def _extract_boxes(grid: np.ndarray) -> list[dict]:
    boxes: list[dict] = []
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[GREY]):
        if int(comp["area"]) != 16:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((5, 5), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == GREY):
            continue
        if np.any(sub[1:4, 1:4] == GREY):
            continue
        pts = np.argwhere((sub[1:4, 1:4] != BG) & (sub[1:4, 1:4] != GREY))
        if pts.shape != (1, 2):
            continue
        mr, mc = map(int, pts[0])
        marker_color = int(sub[1 + mr, 1 + mc])
        if marker_color not in PIECE_COLORS:
            continue
        border_coords = {
            (r0 + rr, c0 + cc)
            for rr in range(5)
            for cc in range(5)
            if rr in (0, 4) or cc in (0, 4)
        }
        boxes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker": (mr, mc),
                "marker_color": marker_color,
                "border_coords": border_coords,
            }
        )
    boxes.sort(key=lambda d: d["bbox"])
    return boxes


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    parts: list[dict] = []
    for color in PIECE_COLORS:
        comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[color])
        for comp in comps:
            if int(comp["area"]) < 3:
                continue
            coords = np.array(comp["coords"], dtype=int)
            r0, c0, r1, c1 = map(int, comp["bbox"])
            rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
            parts.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "shape": {"coords": rel, "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1), "area": len(rel)},
                }
            )
    parts.sort(key=lambda d: d["bbox"])
    return parts


def _pair_fragments_to_boxes(fragments: list[dict], boxes: list[dict]) -> list[dict]:
    assert len(fragments) == len(boxes), "fragment and anchor-box counts must match"
    used_boxes: set[int] = set()
    paired: list[dict] = []
    for frag in fragments:
        scored = []
        for idx, box in enumerate(boxes):
            if box["marker_color"] != frag["color"]:
                continue
            box_pts = np.array(sorted(box["border_coords"]), dtype=int)
            dists = np.abs(frag["coords"][:, None, :] - box_pts[None, :, :]).sum(axis=2)
            scored.append((int(dists.min()), idx))
        assert scored, "every fragment must have a matching-color box"
        scored.sort()
        assert len(scored) == 1 or scored[0][0] < scored[1][0], "nearest box must be unique"
        dist, idx = scored[0]
        assert 1 <= dist <= 4, "anchor box must be adjacent to its fragment"
        assert idx not in used_boxes, "anchor boxes must map one-to-one to fragments"
        used_boxes.add(idx)
        paired.append(
            {
                "shape": frag["shape"],
                "color": frag["color"],
                "vband": int(boxes[idx]["marker"][0]),
                "hband": int(boxes[idx]["marker"][1]),
            }
        )
    return paired


def _solve_entities(grid: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int], set[int]]:
    rect = _extract_rect(grid)
    boxes = _extract_boxes(grid)
    fragments = _extract_fragments(grid)
    paired = _pair_fragments_to_boxes(fragments, boxes)

    entities: list[dict] = []
    for idx, ent in enumerate(paired):
        kind = "center" if ent["hband"] == 1 else "side"
        entities.append(
            {
                "id": idx,
                "kind": kind,
                "shape": ent["shape"],
                "color": ent["color"],
                "vband": ent["vband"],
                "hband": ent["hband"],
            }
        )

    raw_instances = _build_instances(entities, rect, set())
    raw_bad = _same_color_conflicts(raw_instances)
    shift_ids = {ent["id"] for ent in entities if ent["kind"] == "side" and ent["id"] in raw_bad}
    shifted_instances = _build_instances(entities, rect, shift_ids)
    assert not _same_color_conflicts(shifted_instances), "one-step outward correction must resolve color conflicts"
    return entities, rect, shift_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble anchored fragments into a mirror-symmetric rectangle inside the corner-marked frame.

    Input structure:
    - Black background contains four orange singleton corner markers that define the target rectangle.
    - Outside that rectangle sit 4-6 small 4-connected monochrome fragments in blue/red/green/yellow.
    - Each fragment has a nearby grey 5x5 border box whose 3x3 interior contains one same-colored marker pixel encoding a vertical third and a horizontal third.

    Rule:
    - Read the corner markers to recover the output rectangle in the original canvas coordinates.
    - Pair every fragment with its nearest same-colored grey box, then decode the marker pixel position as the target third-row and third-column region.
    - Place the fragment so its bbox center aligns with that region center inside the rectangle.
    - If the decoded horizontal position is left or right, also place the fragment's left-right mirror across the rectangle axis; if it is center, place only the fragment itself.
    - If same-colored entities become 4-neighbor adjacent after the initial placement, shift the conflicting side-anchored entity pair one cell away from the mirror axis.
    - Clear everything outside the rectangle to black and remove the markers/boxes.

    Key insight:
    - The box marker does double duty: its row picks the vertical third, while its column simultaneously picks both coarse horizontal placement and whether the fragment should be self-symmetric or mirrored as a pair.
    """
    g = np.asarray(grid, dtype=int)
    entities, rect, shift_ids = _solve_entities(g)
    instances = _build_instances(entities, rect, shift_ids)
    return _render_instances(g.shape, instances)


def _all_area_plans(target: int, center_count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    plans = []
    side_choices = SIDE_SHAPES.keys()
    center_choices = CENTER_SHAPES.keys()
    def rec_side(i: int, acc: list[int]):
        if i == 3:
            rem = target - sum(acc)
            def rec_center(j: int, center_acc: list[int]):
                if j == center_count:
                    if sum(center_acc) == rem:
                        plans.append((tuple(acc), tuple(center_acc)))
                    return
                for area in center_choices:
                    center_acc.append(int(area))
                    rec_center(j + 1, center_acc)
                    center_acc.pop()
            rec_center(0, [])
            return
        for area in side_choices:
            acc.append(int(area))
            rec_side(i + 1, acc)
            acc.pop()
    rec_side(0, [])
    uniq = sorted(set(plans))
    return uniq


def _side_entity(row: int, hband: int, area: int) -> dict:
    shape = _shape_info(random.choice(SIDE_SHAPES[area]))
    return {"kind": "side", "vband": row, "hband": hband, "shape": shape}


def _center_entity(row: int, area: int) -> dict:
    shape = _shape_info(random.choice(CENTER_SHAPES[area]))
    return {"kind": "center", "vband": row, "hband": 1, "shape": shape}


def _assign_entity_ids(entities: list[dict]) -> list[dict]:
    out = []
    for idx, ent in enumerate(entities):
        ent2 = dict(ent)
        ent2["id"] = idx
        out.append(ent2)
    return out


def _fit_check(entities: list[dict], rect: tuple[int, int, int, int]) -> bool:
    try:
        inst = _build_instances(entities, rect, set())
        _render_instances((rect[2] + 1, rect[3] + 1), inst)
        return True
    except AssertionError:
        return False


def _candidate_shift_rows(entities: list[dict], rect: tuple[int, int, int, int]) -> list[int]:
    raw = _build_instances(entities, rect, set())
    by_id = {inst["entity_id"]: [] for inst in raw}
    for inst in raw:
        by_id[inst["entity_id"]].append(inst["coords"])

    candidates = []
    centers_by_row = {ent["vband"]: ent for ent in entities if ent["kind"] == "center"}
    for ent in entities:
        if ent["kind"] != "side" or ent["vband"] not in centers_by_row:
            continue
        side_coords = set().union(*by_id[ent["id"]])
        center_coords = set().union(*by_id[centers_by_row[ent["vband"]]["id"]])
        touching = any((nr, nc) in center_coords for r, c in side_coords for nr, nc in _neighbors4(r, c))
        if not touching:
            continue
        shifted = _build_instances(entities, rect, {ent["id"]})
        try:
            _render_instances((rect[2] + 1, rect[3] + 1), shifted)
        except AssertionError:
            continue
        shifted_union = set()
        for inst in shifted:
            if inst["entity_id"] == ent["id"]:
                shifted_union |= inst["coords"]
        still_touching = any((nr, nc) in center_coords for r, c in shifted_union for nr, nc in _neighbors4(r, c))
        if not still_touching:
            candidates.append(ent["id"])
    return candidates


def _try_coloring(entities: list[dict], rect: tuple[int, int, int, int], shift_target: int | None) -> list[dict] | None:
    row_to_center = {ent["vband"]: ent["id"] for ent in entities if ent["kind"] == "center"}
    for _ in range(300):
        colors = {}
        for ent in entities:
            colors[ent["id"]] = random.choice(PIECE_COLORS)

        if shift_target is not None:
            row = next(ent["vband"] for ent in entities if ent["id"] == shift_target)
            colors[row_to_center[row]] = colors[shift_target]

        if random.random() < 0.7:
            a, b = random.sample([ent["id"] for ent in entities], 2)
            colors[b] = colors[a]

        colored = [dict(ent, color=int(colors[ent["id"]])) for ent in entities]
        raw = _build_instances(colored, rect, set())
        raw_bad = _same_color_conflicts(raw)
        if shift_target is None:
            if raw_bad:
                continue
            if len(set(colors.values())) < 2:
                continue
            return colored
        if shift_target not in raw_bad:
            continue
        if any(ent_id != shift_target and next(ent["kind"] for ent in colored if ent["id"] == ent_id) == "center" for ent_id in raw_bad):
            rows = {next(ent["vband"] for ent in colored if ent["id"] == ent_id) for ent_id in raw_bad}
            target_row = next(ent["vband"] for ent in colored if ent["id"] == shift_target)
            if rows != {target_row}:
                continue
        final = _build_instances(colored, rect, {shift_target})
        if _same_color_conflicts(final):
            continue
        return colored
    return None


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _make_cluster(ent: dict) -> dict:
    shape = ent["shape"]
    gap = random.randint(1, 2)
    direction = random.choice(("N", "S"))
    if direction == "N":
        top = -gap - shape["h"]
        left = random.randint(0, 5 - shape["w"])
    elif direction == "S":
        top = 5 + gap
        left = random.randint(0, 5 - shape["w"])
    elif direction == "W":
        top = random.randint(0, 5 - shape["h"])
        left = -gap - shape["w"]
    else:
        top = random.randint(0, 5 - shape["h"])
        left = 5 + gap

    cells = [(rr, cc, GREY) for rr in range(5) for cc in range(5) if rr in (0, 4) or cc in (0, 4)]
    mr, mc = ent["vband"], ent["hband"]
    cells.append((1 + mr, 1 + mc, ent["color"]))
    for dr, dc in ent["shape"]["coords"]:
        cells.append((top + dr, left + dc, ent["color"]))

    min_r = min(r for r, _c, _col in cells)
    min_c = min(c for _r, c, _col in cells)
    norm = [(r - min_r, c - min_c, col) for r, c, col in cells]
    max_r = max(r for r, _c, _col in norm)
    max_c = max(c for _r, c, _col in norm)
    return {"cells": norm, "bbox": (0, 0, max_r, max_c)}


def _build_plan() -> tuple[np.ndarray, np.ndarray, dict] | None:
    rect_h, rect_w = RECT_OPTIONS[0]
    rect = (0, 0, rect_h - 1, rect_w - 1)
    want_shift = False

    for _ in range(150):
        entities = []
        for row in SIDE_ROWS:
            entities.append(_side_entity(row, random.choice((0, 2)), 6))
        side_hbands = {ent["hband"] for ent in entities}
        if side_hbands != {0, 2}:
            continue
        for row in SIDE_ROWS:
            entities.append(_center_entity(row, 6))
        entities = _assign_entity_ids(entities)
        if not _fit_check(entities, rect):
            continue
        candidates = _candidate_shift_rows(entities, rect)
        if want_shift and not candidates:
            continue
        shift_target = random.choice(candidates) if want_shift and candidates else None
        colored = _try_coloring(entities, rect, shift_target)
        if colored is None:
            continue

        canvas_h = random.choice(CANVAS_ROWS)
        canvas_w = random.choice(CANVAS_COLS)
        if canvas_h < rect_h + 8 or canvas_w < rect_w + 8:
            continue
        rect_r0 = random.randint(4, canvas_h - rect_h - 4)
        rect_c0 = random.randint(4, canvas_w - rect_w - 4)
        abs_rect = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

        instances = _build_instances(colored, abs_rect, {shift_target} if shift_target is not None else set())
        expected = _render_instances((canvas_h, canvas_w), instances)

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        rr0, cc0, rr1, cc1 = abs_rect
        for r, c in ((rr0, cc0), (rr0, cc1), (rr1, cc0), (rr1, cc1)):
            grid[r, c] = ORANGE

        forbidden = [abs_rect]
        clusters = [_make_cluster(ent) for ent in colored]
        order = list(range(len(colored)))
        random.shuffle(order)
        placed_cluster_boxes = []
        ok = True
        for idx in order:
            cluster = clusters[idx]
            bh = cluster["bbox"][2] + 1
            bw = cluster["bbox"][3] + 1
            candidates_xy = []
            for top in range(canvas_h - bh + 1):
                for left in range(canvas_w - bw + 1):
                    box = (top, left, top + bh - 1, left + bw - 1)
                    if any(not _rect_separated(box, other, gap=1) for other in forbidden):
                        continue
                    if np.any(grid[top : top + bh, left : left + bw] != 0):
                        continue
                    candidates_xy.append((top, left))
            if not candidates_xy:
                ok = False
                break
            top, left = random.choice(candidates_xy)
            for r, c, color in cluster["cells"]:
                grid[top + r, left + c] = int(color)
            forbidden.append((top, left, top + bh - 1, left + bw - 1))
            placed_cluster_boxes.append((top, left, top + bh - 1, left + bw - 1))
        if not ok:
            continue

        meta = {
            "rect": abs_rect,
            "entities": colored,
            "shift_target": shift_target,
        }
        return grid, expected, meta
    return None


def _semantic_check(inp: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    rr0, cc0, rr1, cc1 = meta["rect"]
    rect_area = int((rr1 - rr0 + 1) * (cc1 - cc0 + 1))
    entities = meta["entities"]
    assert rect_area % 2 == 0 and rect_area // 2 == sum(ent["shape"]["area"] for ent in entities)
    assert any(ent["kind"] == "center" for ent in entities)
    assert any(ent["kind"] == "side" and ent["hband"] == 0 for ent in entities)
    assert any(ent["kind"] == "side" and ent["hband"] == 2 for ent in entities)

    local_rect = (rr0, cc0, rr1, cc1)
    raw = _build_instances(entities, local_rect, set())
    raw_bad = _same_color_conflicts(raw)
    shift_target = meta["shift_target"]
    if shift_target is None:
        assert not raw_bad
    else:
        assert shift_target in raw_bad
        shifted = _build_instances(entities, local_rect, {shift_target})
        assert not _same_color_conflicts(shifted)

    out = main(inp)
    assert np.array_equal(out, expected)

    solved_entities, solved_rect, solved_shift_ids = _solve_entities(inp)
    assert solved_rect == local_rect
    assert len(solved_entities) == len(entities)
    if shift_target is None:
        assert solved_shift_ids == set()
    else:
        assert solved_shift_ids == {shift_target}


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_plan()
        if built is None:
            continue
        inp, expected, meta = built
        try:
            _semantic_check(inp, expected, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid anchored-mirror puzzle")
