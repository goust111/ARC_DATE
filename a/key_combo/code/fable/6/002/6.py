from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 28
FRAME_SIDE_MIN = 4
FRAME_SIDE_MAX = 7
OBJECT_COUNT_CHOICES = (4, 5, 6)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
FRAME_KEEP_OUT = 1
GEN_RETRIES = 400

_SEEN_OUTPUTS: set[tuple[tuple[int, int], tuple[int, ...]]] = set()


def _coord_bounds(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _coord_bounds(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _border_cycle(h: int, w: int) -> tuple[tuple[int, int], ...]:
    coords = [(0, c) for c in range(w)]
    coords += [(r, w - 1) for r in range(1, h)]
    coords += [(h - 1, c) for c in range(w - 2, -1, -1)]
    coords += [(r, 0) for r in range(h - 2, 0, -1)]
    return tuple(coords)


def _marker_rel_coords(h: int, w: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (0, 0), (0, w - 1)
    if side == "bottom":
        return (h - 1, 0), (h - 1, w - 1)
    if side == "left":
        return (0, 0), (h - 1, 0)
    if side == "right":
        return (0, w - 1), (h - 1, w - 1)
    raise ValueError(f"unknown side {side}")


def _arc_kind(coords: tuple[tuple[int, int], ...]) -> str | None:
    rel, h, w = _normalize_coords(coords)
    area = len(rel)
    if area < 2:
        return None
    if h == 1 or w == 1:
        return "bar"
    if area != h + w - 1:
        return None
    actual = set(rel)
    for elbow_r, elbow_c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        expected = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if r == elbow_r or c == elbow_c
        }
        if actual == expected:
            return "L"
    return None


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int, min_len: int, max_len: int) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, n_parts: int, prefix: list[int]) -> None:
        if n_parts == 1:
            if min_len <= remaining <= max_len:
                out.append(tuple(prefix + [remaining]))
            return
        lo = max(min_len, remaining - max_len * (n_parts - 1))
        hi = min(max_len, remaining - min_len * (n_parts - 1))
        for value in range(lo, hi + 1):
            rec(remaining - value, n_parts - 1, prefix + [value])

    rec(int(total), int(parts), [])
    return tuple(out)


@lru_cache(maxsize=None)
def _valid_plans(h: int, w: int, count: int) -> tuple[tuple[tuple[tuple[int, int], ...], tuple[str, ...]], ...]:
    border = _border_cycle(h, w)
    perimeter = len(border)
    max_len = h + w - 1
    raw: list[tuple[tuple[tuple[int, int], ...], tuple[str, ...]]] = []

    for start in range(perimeter):
        for lengths in _compositions(perimeter, count, 2, max_len):
            pos = start
            arcs: list[tuple[tuple[int, int], ...]] = []
            kinds: list[str] = []
            for length in lengths:
                coords = tuple(border[(pos + step) % perimeter] for step in range(length))
                kind = _arc_kind(coords)
                if kind is None:
                    break
                arcs.append(tuple(sorted(coords)))
                kinds.append(kind)
                pos = (pos + length) % perimeter
            else:
                plan = tuple(arcs)
                raw.append((plan, tuple(kinds)))

    dedup: dict[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[tuple[int, int], ...], tuple[str, ...]]] = {}
    for plan, kinds in raw:
        key = tuple(sorted(plan))
        dedup.setdefault(key, (plan, kinds))
    return tuple(dedup.values())


def _object_from_component(comp: dict) -> dict:
    coords = tuple((int(r), int(c)) for r, c in comp["coords"])
    rel, h, w = _normalize_coords(coords)
    kind = _arc_kind(coords)
    return {
        "bbox": tuple(map(int, comp["bbox"])),
        "rel": rel,
        "h": int(h),
        "w": int(w),
        "area": int(len(rel)),
        "kind": kind,
    }


def _extract_instance(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    counts = {color: int(np.sum(g == color)) for color in colors}
    marker_colors = [color for color, cnt in counts.items() if cnt == 1]
    neutral_colors = [color for color, cnt in counts.items() if cnt > 1]

    assert len(marker_colors) == 2, "expected exactly two marker colors"
    assert len(neutral_colors) == 1, "expected exactly one neutral object color"
    neutral = int(neutral_colors[0])

    markers = {
        tuple(map(int, np.argwhere(g == color)[0])): int(color)
        for color in sorted(marker_colors)
    }

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[neutral],
        return_format="dict",
    )
    assert components, "expected at least one neutral object"

    objects = [_object_from_component(comp) for comp in components]
    for obj in objects:
        assert obj["kind"] in {"bar", "L"}, "neutral objects must be bars or elbows"

    objects.sort(
        key=lambda obj: (
            len(obj["rel"]),
            obj["h"],
            obj["w"],
            obj["bbox"][0],
            obj["bbox"][1],
        ),
        reverse=True,
    )
    return bg, neutral, markers, objects


def _candidate_frames(
    shape: tuple[int, int],
    markers: dict[tuple[int, int], int],
    side_len: int,
    other_len: int,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    (p0, p1) = sorted(markers.keys())
    h_grid, w_grid = shape
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []

    if p0[0] == p1[0]:
        r = int(p0[0])
        c0, c1 = sorted((int(p0[1]), int(p1[1])))
        assert c1 - c0 + 1 == side_len
        candidates = [
            ("top", (r, c0, r + other_len - 1, c1)),
            ("bottom", (r - other_len + 1, c0, r, c1)),
        ]
    else:
        c = int(p0[1])
        r0, r1 = sorted((int(p0[0]), int(p1[0])))
        assert r1 - r0 + 1 == side_len
        candidates = [
            ("left", (r0, c, r1, c + other_len - 1)),
            ("right", (r0, c - other_len + 1, r1, c)),
        ]

    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for side, bbox in candidates:
        r0, c0, r1, c1 = bbox
        if 0 <= r0 <= r1 < h_grid and 0 <= c0 <= c1 < w_grid:
            out.append((side, tuple(map(int, bbox))))
    return out


def _border_cells_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return tuple((r0 + r, c0 + c) for r, c in _border_cycle(h, w))


def _placements_for_object(
    obj: dict,
    bbox: tuple[int, int, int, int],
    border_set: frozenset[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    placements: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for top in range(r0, r1 - obj["h"] + 2):
        for left in range(c0, c1 - obj["w"] + 2):
            cells = frozenset((top + rr, left + cc) for rr, cc in obj["rel"])
            if cells <= border_set and cells not in seen:
                placements.append(cells)
                seen.add(cells)
    return placements


def _solve_frame(
    shape: tuple[int, int],
    bg: int,
    neutral: int,
    markers: dict[tuple[int, int], int],
    objects: list[dict],
    side: str,
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    border_cells = _border_cells_from_bbox(bbox)
    border_set = frozenset(border_cells)
    if any(marker not in border_set for marker in markers):
        return []

    total_area = sum(obj["area"] for obj in objects)
    if total_area != len(border_cells):
        return []

    placements_by_object = [_placements_for_object(obj, bbox, border_set) for obj in objects]
    if any(not placements for placements in placements_by_object):
        return []

    marker_coords = tuple(sorted(markers.keys()))
    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            len(placements_by_object[idx]),
            -objects[idx]["area"],
            objects[idx]["h"],
            objects[idx]["w"],
            objects[idx]["bbox"][0],
            objects[idx]["bbox"][1],
        ),
    )

    solutions: dict[tuple[int, ...], dict] = {}

    def build_solution(chosen: list[tuple[int, frozenset[tuple[int, int]]]]) -> None:
        out = np.full(shape, bg, dtype=int)
        marker_owner: dict[tuple[int, int], int] = {}
        for obj_idx, cells in chosen:
            hits = [marker for marker in marker_coords if marker in cells]
            if len(hits) > 1:
                return
            color = int(markers[hits[0]]) if hits else int(neutral)
            for r, c in cells:
                out[r, c] = color
            if hits:
                marker_owner[hits[0]] = int(obj_idx)
        if set(marker_owner.keys()) != set(marker_coords):
            return
        if len(set(marker_owner.values())) != len(marker_owner):
            return
        key = tuple(map(int, out.ravel()))
        solutions.setdefault(
            key,
            {
                "output": out,
                "bbox": bbox,
                "side": side,
                "marker_owner": marker_owner,
                "placements": {obj_idx: cells for obj_idx, cells in chosen},
            },
        )

    def search(
        depth: int,
        used: frozenset[tuple[int, int]],
        chosen: list[tuple[int, frozenset[tuple[int, int]]]],
    ) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            if used == border_set:
                build_solution(chosen)
            return

        obj_idx = order[depth]
        for cells in placements_by_object[obj_idx]:
            if cells & used:
                continue
            if all(marker not in cells for marker in marker_coords):
                pass
            elif sum(marker in cells for marker in marker_coords) > 1:
                continue
            search(depth + 1, used | cells, chosen + [(obj_idx, cells)])
            if len(solutions) > 1:
                return

    search(0, frozenset(), [])
    return list(solutions.values())


def _solve_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, neutral, markers, objects = _extract_instance(g)

    marker_points = sorted(markers.keys())
    same_row = marker_points[0][0] == marker_points[1][0]
    same_col = marker_points[0][1] == marker_points[1][1]
    assert same_row ^ same_col, "markers must define a horizontal or vertical side"

    side_len = (
        abs(marker_points[0][1] - marker_points[1][1]) + 1
        if same_row
        else abs(marker_points[0][0] - marker_points[1][0]) + 1
    )
    total_area = sum(obj["area"] for obj in objects)
    assert total_area % 2 == 0, "frame perimeter must be even"

    other_len = total_area // 2 - side_len + 2
    assert other_len >= 2, "derived side length must be at least 2"
    assert 2 * side_len + 2 * other_len - 4 == total_area

    candidate_frames = _candidate_frames(g.shape, markers, side_len, other_len)
    outputs: dict[tuple[int, ...], dict] = {}
    all_solutions: list[dict] = []
    for side, bbox in candidate_frames:
        sols = _solve_frame(g.shape, bg, neutral, markers, objects, side, bbox)
        all_solutions.extend(sols)
        for sol in sols:
            key = tuple(map(int, sol["output"].ravel()))
            outputs.setdefault(key, sol)

    assert outputs, "no valid frame assembly found"
    assert len(outputs) == 1, "frame assembly must determine a unique output"

    solved = next(iter(outputs.values()))
    solved["candidate_frames"] = candidate_frames
    solved["bg"] = bg
    solved["neutral"] = neutral
    solved["markers"] = markers
    solved["objects"] = objects
    solved["all_solutions"] = all_solutions
    return solved


def _patch_from_arc(
    arc: tuple[tuple[int, int], ...],
    bg: int,
    color: int,
) -> np.ndarray:
    rel, h, w = _normalize_coords(arc)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in rel:
        patch[r, c] = color
    return patch


def _render_expected_output(
    shape: tuple[int, int],
    bg: int,
    neutral: int,
    marker_map: dict[tuple[int, int], int],
    arcs: tuple[tuple[tuple[int, int], ...], ...],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, _r1, _c1 = bbox
    out = np.full(shape, bg, dtype=int)
    for arc in arcs:
        abs_cells = [(r0 + r, c0 + c) for r, c in arc]
        hits = [coord for coord in marker_map if coord in abs_cells]
        assert len(hits) <= 1
        color = int(marker_map[hits[0]]) if hits else int(neutral)
        for r, c in abs_cells:
            out[r, c] = color
    return out


def _eligible_plans(
    h: int,
    w: int,
    count: int,
    side: str,
) -> list[tuple[tuple[tuple[int, int], ...], tuple[str, ...]]]:
    marker_cells = set(_marker_rel_coords(h, w, side))
    eligible: list[tuple[tuple[tuple[int, int], ...], tuple[str, ...]]] = []
    for arcs, kinds in _valid_plans(h, w, count):
        if kinds.count("L") < 2 or kinds.count("bar") < 1:
            continue
        marker_arc_indices = [idx for idx, arc in enumerate(arcs) if any(cell in arc for cell in marker_cells)]
        if len(marker_arc_indices) != 2 or len(set(marker_arc_indices)) != 2:
            continue
        if any(len(arcs[idx]) < 3 for idx in marker_arc_indices):
            continue
        shape_sigs = {(kinds[idx],) + _normalize_coords(arcs[idx])[1:] for idx in range(len(arcs))}
        if len(shape_sigs) < 3:
            continue
        eligible.append((arcs, kinds))
    return eligible


def _sample_bbox(h_grid: int, w_grid: int, h: int, w: int, side: str) -> tuple[int, int, int, int]:
    side_margin_r = 1
    side_margin_c = 1

    if side in {"top", "bottom"}:
        assert w_grid >= w + 2 * side_margin_c + 1
        c0 = random.randint(side_margin_c, w_grid - w - side_margin_c)
        if side == "top":
            marker_r = random.randint(1, min(2, h - 2))
            r0 = marker_r
        else:
            marker_r = h_grid - 1 - random.randint(1, min(2, h - 2))
            r0 = marker_r - h + 1
        return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)

    assert h_grid >= h + 2 * side_margin_r + 1
    r0 = random.randint(side_margin_r, h_grid - h - side_margin_r)
    if side == "left":
        marker_c = random.randint(1, min(2, w - 2))
        c0 = marker_c
    else:
        marker_c = w_grid - 1 - random.randint(1, min(2, w - 2))
        c0 = marker_c - w + 1
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _scatter_patches(
    shape: tuple[int, int],
    bg: int,
    patches: list[np.ndarray],
    reserved_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    h_grid, w_grid = shape
    occ = np.full((h_grid, w_grid), bg, dtype=int)
    inp = np.full((h_grid, w_grid), bg, dtype=int)

    r0, c0, r1, c1 = reserved_bbox
    rr0 = max(0, r0 - FRAME_KEEP_OUT)
    cc0 = max(0, c0 - FRAME_KEEP_OUT)
    rr1 = min(h_grid, r1 + FRAME_KEEP_OUT + 1)
    cc1 = min(w_grid, c1 + FRAME_KEEP_OUT + 1)
    occ[rr0:rr1, cc0:cc1] = -1

    order = list(range(len(patches)))
    random.shuffle(order)
    try:
        for idx in order:
            patch = patches[idx]
            top, left = random_free_location(
                occ,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
            paste(occ, patch, top, left, transparent=bg)
            paste(inp, patch, top, left, transparent=bg)
    except ValueError:
        return None
    return inp


def _generate_one() -> np.ndarray | None:
    bg, neutral, marker_a, marker_b = random.sample(range(10), 4)
    side = random.choice(("top", "bottom", "left", "right"))

    hw_choices = [(h, w) for h in range(FRAME_SIDE_MIN, FRAME_SIDE_MAX + 1) for w in range(FRAME_SIDE_MIN, FRAME_SIDE_MAX + 1)]
    random.shuffle(hw_choices)

    for h, w in hw_choices:
        count_choices = list(OBJECT_COUNT_CHOICES)
        random.shuffle(count_choices)
        for count in count_choices:
            plans = _eligible_plans(h, w, count, side)
            if not plans:
                continue

            arcs, kinds = random.choice(plans)
            h_grid = random.randint(max(GRID_SIDE_MIN, h + 8), min(GRID_SIDE_MAX, h + 14))
            w_grid = random.randint(max(GRID_SIDE_MIN, w + 8), min(GRID_SIDE_MAX, w + 14))

            bbox = _sample_bbox(h_grid, w_grid, h, w, side)
            r0, c0, _r1, _c1 = bbox
            marker_rel = _marker_rel_coords(h, w, side)
            marker_map = {
                (r0 + marker_rel[0][0], c0 + marker_rel[0][1]): int(marker_a),
                (r0 + marker_rel[1][0], c0 + marker_rel[1][1]): int(marker_b),
            }

            expected = _render_expected_output(
                (h_grid, w_grid),
                bg,
                neutral,
                marker_map,
                arcs,
                bbox,
            )
            expected_key = (expected.shape, tuple(map(int, expected.ravel())))
            if expected_key in _SEEN_OUTPUTS:
                continue

            patches = [_patch_from_arc(arc, bg, neutral) for arc in arcs]
            inp = _scatter_patches((h_grid, w_grid), bg, patches, bbox)
            if inp is None:
                continue

            for (mr, mc), color in marker_map.items():
                if inp[mr, mc] != bg:
                    break
                inp[mr, mc] = color
            else:
                try:
                    solved = _solve_instance(inp)
                except Exception:
                    continue
                if len(solved["candidate_frames"]) != 1:
                    continue
                if solved["side"] != side or solved["bbox"] != bbox:
                    continue
                if not np.array_equal(solved["output"], expected):
                    continue
                if len(solved["marker_owner"]) != 2 or len(set(solved["marker_owner"].values())) != 2:
                    continue
                if kinds.count("L") < 2 or kinds.count("bar") < 1:
                    continue
                _SEEN_OUTPUTS.add(expected_key)
                return inp

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble neutral bars and elbows onto an implied rectangular frame, then recolor the corner-covering pieces.

    Input structure:
    - bg = most frequent color.
    - Two differently colored isolated single pixels mark the endpoints of one side of an unseen rectangle.
    - Every other non-bg cell belongs to a same-colored 4-connected one-pixel-thick object: a horizontal/vertical bar or an L-shaped elbow.

    Rule:
    - Let the distance between the two marker pixels determine the marked side length.
    - Let the total number of neutral object pixels be the perimeter of the missing one-cell-thick rectangular frame, so the perpendicular side length is solved from perimeter conservation.
    - Among the in-bounds rectangles compatible with the marked side, translate every object without rotation or reflection so the frame border is covered exactly once and erase the scattered copies.
    - Recolor the entire placed object that covers a marked corner cell to that marker's color; objects that cover no marked corner stay neutral.

    Key insight:
    The frame is absent from the input: one side is given by the two corner markers, the other side must be recovered from the neutral pixel count, and only then can the pieces be uniquely assembled and recolored.
    """
    solved = _solve_instance(np.asarray(grid, dtype=int))
    return np.asarray(solved["output"], dtype=int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
