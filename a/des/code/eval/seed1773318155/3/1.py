from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ARC_COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
}


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _anchor_rank(color: int) -> tuple[str, int]:
    return _ARC_COLOR_NAMES.get(int(color), f"color{int(color)}"), int(color)


def _paint_translated_fragments(
    grid_shape: tuple[int, int],
    bg: int,
    fragments: list[dict],
) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    owner = np.full(grid_shape, -1, dtype=int)

    for frag_id, frag in enumerate(fragments):
        ar, ac = frag["anchor"]
        tr, tc = frag["target"]
        dr = int(tr - ar)
        dc = int(tc - ac)
        for r, c in frag["coords"]:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < grid_shape[0] and 0 <= cc < grid_shape[1]
            assert int(owner[rr, cc]) == -1, "translated fragments may not overlap"
            owner[rr, cc] = frag_id
            out[rr, cc] = int(frag["body_color"])

    for component in _mask_components(owner >= 0):
        fragment_ids = {int(owner[r, c]) for r, c in component}
        if len(fragment_ids) < 2:
            continue
        r0, c0, r1, c1 = _bbox(component)
        if len(component) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        winner = min((int(fragments[idx]["anchor_color"]) for idx in fragment_ids), key=_anchor_rank)
        for r, c in component:
            out[r, c] = winner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each fragment's anchor with the same-colored target marker, translate the fragment so
    the anchor lands on that marker, then recolor any multi-fragment filled rectangle by the
    alphabetically earliest anchor color in the group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)

    components = _find_monochrome_components(g, bg)
    fragments_raw = [comp for comp in components if 2 <= int(comp["area"]) <= 5]
    marker_raw = [comp for comp in components if int(comp["area"]) == 1]
    assert fragments_raw and marker_raw, "expected fragments plus single-pixel markers"

    cell_to_fragment: dict[tuple[int, int], int] = {}
    fragments: list[dict] = []
    for frag_id, comp in enumerate(fragments_raw):
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        for cell in coords:
            assert cell not in cell_to_fragment
            cell_to_fragment[cell] = frag_id
        fragments.append(
            {
                "coords": coords,
                "body_color": int(comp["color"]),
                "anchor": None,
                "target": None,
                "anchor_color": None,
            }
        )

    markers_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in marker_raw:
        point = comp["coords"][0]
        markers_by_color[int(comp["color"])].append((int(point[0]), int(point[1])))

    for color, points in markers_by_color.items():
        assert len(points) == 2, "each anchor color must appear exactly twice"
        anchor_point: tuple[int, int] | None = None
        target_point: tuple[int, int] | None = None
        anchor_fragment: int | None = None
        for r, c in points:
            adjacent = {
                cell_to_fragment[(r + dr, c + dc)]
                for dr, dc in _DIR4
                if (r + dr, c + dc) in cell_to_fragment
            }
            if len(adjacent) == 1:
                frag_id = next(iter(adjacent))
                assert anchor_point is None, "anchor color must have a unique fragment-side marker"
                anchor_point = (r, c)
                anchor_fragment = int(frag_id)
            else:
                assert len(adjacent) == 0, "target marker must be isolated from fragment bodies"
                assert target_point is None, "anchor color must have a unique target marker"
                target_point = (r, c)
        assert anchor_point is not None and target_point is not None and anchor_fragment is not None
        frag = fragments[anchor_fragment]
        assert frag["anchor"] is None, "each fragment must have exactly one anchor"
        frag["anchor"] = anchor_point
        frag["target"] = target_point
        frag["anchor_color"] = int(color)

    assert all(frag["anchor"] is not None for frag in fragments), "every fragment needs one anchor"
    return _paint_translated_fragments(g.shape, bg, fragments)


# --------------- generator ---------------

_GEN_BG = 0
_GEN_ANCHOR_COLORS = [1, 2, 3]  # blue, red, green; alphabetical rank is handled by name lookup.
_GEN_BODY_COLORS = [4, 5, 6, 7, 8, 9]
_GEN_GRID_H_RANGE = (16, 22)
_GEN_GRID_W_RANGE = (16, 22)
_GEN_MAX_ATTEMPTS = 400
_FINAL_PAD = 1
_SCATTER_PAD = 1
_MIN_MOVE_CHEBYSHEV = 4

_GROUP_TEMPLATES = [
    np.array([[1, 1], [2, 2]], dtype=int),
    np.array([[1, 1, 1], [2, 2, 2]], dtype=int),
    np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=int),
    np.array([[1, 1, 2], [1, 2, 2], [1, 2, 2]], dtype=int),
    np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=int),
    np.array([[1, 2, 2, 3], [1, 2, 2, 3]], dtype=int),
    np.array([[1, 1, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=int),
    np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=int),
]

_FREE_SHAPES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    arr = np.array(mask, copy=True)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, random.randrange(4))
    return np.array(arr, copy=True)


def _reserve_box(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int, pad: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(r0) - pad),
        max(0, int(c0) - pad),
        min(shape[0] - 1, int(r1) + pad),
        min(shape[1] - 1, int(c1) + pad),
    )


def _touches_existing(point: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = point
    return any(max(abs(r - orow), abs(c - ocol)) <= 1 for orow, ocol in others)


def _entity_from_group(template: np.ndarray, body_colors: list[int], anchor_colors: list[int]) -> dict | None:
    grid = _random_transform(template)
    h, w = grid.shape
    labels = sorted(int(v) for v in np.unique(grid) if int(v) != 0)
    if len(labels) != len(body_colors):
        return None

    fragments = []
    for label, body_color, anchor_color in zip(labels, body_colors, anchor_colors):
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == label)]
        pr0, pc0, pr1, pc1 = _bbox(coords)
        local_cells = [(r - pr0, c - pc0) for r, c in coords]
        candidates = []
        for r, c in coords:
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    candidates.append((r - pr0, c - pc0, dr, dc))
        candidates = sorted(set(candidates))
        if not candidates or not (2 <= len(local_cells) <= 5):
            return None
        fragments.append(
            {
                "offset": (pr0, pc0),
                "cells": local_cells,
                "body_color": int(body_color),
                "anchor_color": int(anchor_color),
                "candidates": candidates,
            }
        )

    return {"h": h, "w": w, "fragments": fragments, "area": int(np.count_nonzero(grid))}


def _entity_from_free_shape(mask: np.ndarray, body_color: int, anchor_color: int) -> dict | None:
    grid = _random_transform(mask)
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid != 0)]
    h, w = grid.shape
    candidates = []
    for r, c in coords:
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                candidates.append((r, c, dr, dc))
    candidates = sorted(set(candidates))
    if not candidates or not (2 <= len(coords) <= 5):
        return None
    return {
        "h": h,
        "w": w,
        "fragments": [
            {
                "offset": (0, 0),
                "cells": coords,
                "body_color": int(body_color),
                "anchor_color": int(anchor_color),
                "candidates": candidates,
            }
        ],
        "area": len(coords),
    }


def _assign_entity_targets(
    entity: dict,
    top: int,
    left: int,
    used_targets: set[tuple[int, int]],
) -> list[dict] | None:
    chosen_targets = set(used_targets)
    chosen: list[dict | None] = [None] * len(entity["fragments"])
    order = sorted(range(len(entity["fragments"])), key=lambda idx: len(entity["fragments"][idx]["candidates"]))

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        frag_idx = order[i]
        fragment = entity["fragments"][frag_idx]
        candidates = list(fragment["candidates"])
        random.shuffle(candidates)
        off_r, off_c = fragment["offset"]
        for local_r, local_c, dr, dc in candidates:
            target = (int(top + off_r + local_r + dr), int(left + off_c + local_c + dc))
            if target in chosen_targets or _touches_existing(target, chosen_targets):
                continue
            chosen_targets.add(target)
            chosen[frag_idx] = {
                "target": target,
                "anchor_local": (int(local_r), int(local_c)),
                "anchor_dir": (int(dr), int(dc)),
            }
            if backtrack(i + 1):
                return True
            chosen_targets.remove(target)
            chosen[frag_idx] = None
        return False

    return [entry for entry in chosen] if backtrack(0) else None


def _place_final_entities(shape: tuple[int, int], entities: list[dict]) -> list[dict] | None:
    order = sorted(range(len(entities)), key=lambda idx: entities[idx]["area"], reverse=True)

    def backtrack(i: int, reserved: np.ndarray, used_targets: set[tuple[int, int]], placed: list[dict]) -> list[dict] | None:
        if i == len(order):
            return placed

        entity = entities[order[i]]
        positions = [
            (r, c)
            for r in range(1, shape[0] - entity["h"])
            for c in range(1, shape[1] - entity["w"])
        ]
        random.shuffle(positions)
        for top, left in positions:
            rr0, cc0, rr1, cc1 = _reserve_box(shape, top, left, top + entity["h"] - 1, left + entity["w"] - 1, _FINAL_PAD)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            assignments = _assign_entity_targets(entity, top, left, used_targets)
            if assignments is None:
                continue

            next_reserved = reserved.copy()
            next_reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            next_targets = set(used_targets)
            placed_entity = {
                "top": int(top),
                "left": int(left),
                "fragments": [],
                "reserve_box": (rr0, cc0, rr1, cc1),
            }
            for fragment, assignment in zip(entity["fragments"], assignments):
                next_targets.add(assignment["target"])
                placed_entity["fragments"].append(
                    {
                        "offset": fragment["offset"],
                        "cells": fragment["cells"],
                        "body_color": fragment["body_color"],
                        "anchor_color": fragment["anchor_color"],
                        "anchor_local": assignment["anchor_local"],
                        "anchor_dir": assignment["anchor_dir"],
                        "target": assignment["target"],
                    }
                )

            result = backtrack(i + 1, next_reserved, next_targets, placed + [placed_entity])
            if result is not None:
                return result
        return None

    return backtrack(0, np.zeros(shape, dtype=bool), set(), [])


def _fragment_envelope_cells(fragment: dict, top: int, left: int) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    body = [(int(top + r), int(left + c)) for r, c in fragment["cells"]]
    alr, alc = fragment["anchor_local"]
    dr, dc = fragment["anchor_dir"]
    anchor = (int(top + alr + dr), int(left + alc + dc))
    all_cells = body + [anchor]
    r0, c0, r1, c1 = _bbox(all_cells)
    return body, anchor, (r0, c0, r1, c1)


def _scatter_fragments(shape: tuple[int, int], fragments: list[dict], blocked: np.ndarray) -> list[dict] | None:
    order = sorted(range(len(fragments)), key=lambda idx: len(fragments[idx]["cells"]), reverse=True)

    def backtrack(i: int, reserved: np.ndarray, placed: list[dict]) -> list[dict] | None:
        if i == len(order):
            return placed
        frag = fragments[order[i]]
        h = 1 + max(r for r, _ in frag["cells"])
        w = 1 + max(c for _, c in frag["cells"])
        positions = [(r, c) for r in range(shape[0] - h + 1) for c in range(shape[1] - w + 1)]
        random.shuffle(positions)
        for top, left in positions:
            body, anchor, bbox = _fragment_envelope_cells(frag, top, left)
            ar, ac = anchor
            if not (0 <= ar < shape[0] and 0 <= ac < shape[1]):
                continue
            if max(abs(ar - frag["target"][0]), abs(ac - frag["target"][1])) < _MIN_MOVE_CHEBYSHEV:
                continue
            if any(bool(reserved[r, c]) for r, c in body) or bool(reserved[ar, ac]):
                continue
            rr0, cc0, rr1, cc1 = _reserve_box(shape, *bbox, _SCATTER_PAD)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            next_reserved = reserved.copy()
            next_reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed_fragment = {
                "coords": body,
                "body_color": frag["body_color"],
                "anchor": anchor,
                "target": frag["target"],
                "anchor_color": frag["anchor_color"],
            }
            result = backtrack(i + 1, next_reserved, placed + [placed_fragment])
            if result is not None:
                return result
        return None

    return backtrack(0, blocked.copy(), [])


def _scenario_entities(scenario: str, anchor_colors: list[int], body_colors: list[int]) -> list[dict] | None:
    if scenario == "group2":
        template = random.choice([tpl for tpl in _GROUP_TEMPLATES if int(np.max(tpl)) == 2])
        return [_entity_from_group(template, body_colors[:2], anchor_colors[:2])]
    if scenario == "group2_plus_leftover":
        template = random.choice([tpl for tpl in _GROUP_TEMPLATES if int(np.max(tpl)) == 2])
        group = _entity_from_group(template, body_colors[:2], anchor_colors[:2])
        leftover = _entity_from_free_shape(random.choice(_FREE_SHAPES), body_colors[2], anchor_colors[2])
        return [group, leftover]
    if scenario == "group3":
        template = random.choice([tpl for tpl in _GROUP_TEMPLATES if int(np.max(tpl)) == 3])
        return [_entity_from_group(template, body_colors[:3], anchor_colors[:3])]
    count = 2 if scenario == "none2" else 3
    return [_entity_from_free_shape(random.choice(_FREE_SHAPES), body_colors[i], anchor_colors[i]) for i in range(count)]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GEN_GRID_H_RANGE)
    w = random.randint(*_GEN_GRID_W_RANGE)
    shape = (h, w)
    scenario = random.choice(["group2", "group2_plus_leftover", "group3", "none2", "none3"])
    n_fragments = 2 if scenario in {"group2", "none2"} else 3
    anchor_colors = random.sample(_GEN_ANCHOR_COLORS, n_fragments)
    body_colors = random.sample(_GEN_BODY_COLORS, n_fragments)
    random.shuffle(anchor_colors)
    random.shuffle(body_colors)

    entities = _scenario_entities(scenario, anchor_colors, body_colors)
    if not entities or any(entity is None for entity in entities):
        return None
    total_fragments = sum(len(entity["fragments"]) for entity in entities)
    if total_fragments != n_fragments:
        return None

    placed_entities = _place_final_entities(shape, entities)
    if placed_entities is None:
        return None

    final_reserved = np.zeros(shape, dtype=bool)
    source_fragments: list[dict] = []
    target_cells: set[tuple[int, int]] = set()
    for entity in placed_entities:
        rr0, cc0, rr1, cc1 = entity["reserve_box"]
        final_reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        top = entity["top"]
        left = entity["left"]
        for fragment in entity["fragments"]:
            off_r, off_c = fragment["offset"]
            coords = [(int(top + off_r + r), int(left + off_c + c)) for r, c in fragment["cells"]]
            target = tuple(fragment["target"])
            source_fragments.append(
                {
                    "cells": fragment["cells"],
                    "body_color": fragment["body_color"],
                    "anchor_color": fragment["anchor_color"],
                    "anchor_local": fragment["anchor_local"],
                    "anchor_dir": fragment["anchor_dir"],
                    "target": target,
                    "final_coords": coords,
                }
            )
            target_cells.add(target)

    scattered = _scatter_fragments(shape, source_fragments, final_reserved)
    if scattered is None:
        return None

    grid = np.full(shape, _GEN_BG, dtype=int)
    for fragment in scattered:
        for r, c in fragment["coords"]:
            if int(grid[r, c]) != _GEN_BG:
                return None
            grid[r, c] = int(fragment["body_color"])
        ar, ac = fragment["anchor"]
        if int(grid[ar, ac]) != _GEN_BG:
            return None
        grid[ar, ac] = int(fragment["anchor_color"])
    for fragment in scattered:
        tr, tc = fragment["target"]
        if int(grid[tr, tc]) != _GEN_BG:
            return None
        grid[tr, tc] = int(fragment["anchor_color"])

    expected = _paint_translated_fragments(shape, _GEN_BG, scattered)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, actual):
        return None
    if _majority_color(grid) != _GEN_BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
