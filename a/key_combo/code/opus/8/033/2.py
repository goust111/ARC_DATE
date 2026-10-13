from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components  # noqa: E402


BG = Color.BLACK
BLUE = Color.BLUE
TEAL = Color.TEAL

CORNERS = ("tl", "tr", "bl", "br")
CORNER_TO_DIR = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
PREFERRED_MODE_CYCLE = ("row2", "col2", "chain3")

MIN_SIDE = 4
MAX_SIDE = 6
BASE_GAP = 3
MAX_GRID_SIDE = 30
GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merge_fragment_groups(components: list[dict]) -> list[dict]:
    groups = [{"coords": comp["coords"], "bbox": comp["bbox"]} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": coords, "bbox": bbox_coords(coords)}
                    del groups[j]
                    changed = True
                    break
    return groups


def _control_corner(bbox: tuple[int, int, int, int], coord: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    corner_map = {
        (r0, c0): "tl",
        (r0, c1): "tr",
        (r1, c0): "bl",
        (r1, c1): "br",
    }
    return corner_map.get(tuple(map(int, coord)))


def _group_moves(grid: np.ndarray) -> list[dict]:
    blue_components = find_components(grid, bg=BG, colors=[BLUE], connectivity=4)
    assert blue_components, "expected blue fragments"

    teal_coords = {tuple(map(int, p)) for p in np.argwhere(grid == TEAL)}
    assert teal_coords, "expected teal control pixels"

    groups = _merge_fragment_groups(blue_components)
    moves: list[dict] = []
    used_teals: set[tuple[int, int]] = set()

    for group in sorted(groups, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        r0, c0, r1, c1 = map(int, group["bbox"])
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        teal_inside = np.argwhere(region == TEAL)
        assert teal_inside.shape == (1, 2), "each merged group bbox must contain one teal control"
        control = (r0 + int(teal_inside[0, 0]), c0 + int(teal_inside[0, 1]))
        corner = _control_corner((r0, c0, r1, c1), control)
        assert corner is not None, "teal control must sit on a merged-bbox corner"
        dr, dc = CORNER_TO_DIR[corner]

        for nr, nc in ((control[0] - 1, control[1]), (control[0] + 1, control[1]), (control[0], control[1] - 1), (control[0], control[1] + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                assert int(grid[nr, nc]) != BLUE, "teal control must be 4-disconnected from blue fragments"

        tr0, tc0 = r0 + dr, c0 + dc
        tr1, tc1 = r1 + dr, c1 + dc
        assert 0 <= tr0 <= tr1 < grid.shape[0] and 0 <= tc0 <= tc1 < grid.shape[1], "shifted rectangle must stay in bounds"

        used_teals.add(control)
        moves.append(
            {
                "bbox": (r0, c0, r1, c1),
                "target_bbox": (tr0, tc0, tr1, tc1),
                "corner": corner,
                "dir": (dr, dc),
                "coords": group["coords"],
            }
        )

    assert used_teals == teal_coords, "every teal control must belong to exactly one merged group"
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """Merge span-aligned blue fragments into rectangles, then shift each completed rectangle one diagonal step toward its teal-marked merged-bbox corner.

    Input structure:
    - The background is black.
    - Blue cells form several 4-connected fragments; fragments from the same hidden rectangle may be disconnected.
    - Each hidden rectangle group has exactly one teal control pixel on one corner of the group's merged bounding box, and that teal pixel is not 4-connected to any blue fragment.

    Rule:
    - Find all blue 4-connected fragments.
    - Merge fragments transitively when their bounding boxes overlap in row span or in column span.
    - For each merged group, use the union bbox as the rectangle to complete.
    - Read the unique teal corner on that bbox: top-left means shift (-1,-1), top-right (-1,+1), bottom-left (+1,-1), bottom-right (+1,+1).
    - Erase the original bbox and redraw the fully filled blue rectangle one cell diagonally in that direction.

    Key insight:
    - The teal control only becomes interpretable after fragment groups are merged by row/column-span overlap, so rectangle completion and motion direction must be inferred from the same merged bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BG, BLUE, TEAL}, "input uses only black, blue, teal"

    moves = _group_moves(g)

    out = g.copy()
    for move in moves:
        r0, c0, r1, c1 = move["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for move in moves:
        tr0, tc0, tr1, tc1 = move["target_bbox"]
        assert np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == BG), "shifted rectangles must not overlap"
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BLUE

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(int(total)):
        out[random.randrange(parts)] += 1
    return out


def _reflect_cells(cells: set[tuple[int, int]], h: int, w: int, corner: str) -> set[tuple[int, int]]:
    reflected: set[tuple[int, int]] = set()
    for r, c in cells:
        rr, cc = int(r), int(c)
        if corner in {"bl", "br"}:
            rr = h - 1 - rr
        if corner in {"tr", "br"}:
            cc = w - 1 - cc
        reflected.add((rr, cc))
    return reflected


def _reflect_point(point: tuple[int, int], h: int, w: int, corner: str) -> tuple[int, int]:
    return next(iter(_reflect_cells({point}, h, w, corner)))


def _mode_candidates(h: int, w: int) -> list[str]:
    modes = ["row2", "col2"]
    if h >= 5 and w >= 5:
        modes.append("chain3")
    return modes


def _pattern_row2(h: int, w: int) -> list[set[tuple[int, int]]]:
    top_start = max(2, w - 3)
    f1 = {(0, c) for c in range(top_start, w)} | {(1, top_start), (2, top_start)}
    if random.random() < 0.5 and w >= 5:
        f1.add((1, w - 1))

    f2 = {(r, 0) for r in range(2, h)} | {(h - 1, 1)}
    if h >= 5 and random.random() < 0.5:
        f2.add((h - 2, 1))
    return [f1, f2]


def _pattern_col2(h: int, w: int) -> list[set[tuple[int, int]]]:
    f1 = {(0, c) for c in range(2, w)} | {(1, w - 1)}
    if random.random() < 0.5 and w >= 5:
        f1.add((1, w - 2))

    f2 = {(r, 0) for r in range(2, h)} | {(h - 1, 1), (h - 1, 2)}
    if h >= 5 and random.random() < 0.5:
        f2.add((h - 2, 2))
    return [f1, f2]


def _pattern_chain3(h: int, w: int) -> list[set[tuple[int, int]]]:
    assert h >= 5 and w >= 5
    f1 = {(0, w - 1), (1, w - 1), (2, w - 1), (0, w - 2)}

    f2 = {(r, 2) for r in range(1, h - 1)}

    f3 = {(h - 2, 0), (h - 1, 0), (h - 1, 1)}
    return [f1, f2, f3]


def _build_local_fragments(h: int, w: int, corner: str, mode: str) -> list[set[tuple[int, int]]]:
    if mode == "row2":
        canonical = _pattern_row2(h, w)
    elif mode == "col2":
        canonical = _pattern_col2(h, w)
    elif mode == "chain3":
        canonical = _pattern_chain3(h, w)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [_reflect_cells(fragment, h, w, corner) for fragment in canonical]


def _control_local(h: int, w: int, corner: str) -> tuple[int, int]:
    return _reflect_point((0, 0), h, w, corner)


def _component_relation_signature(components: list[dict]) -> tuple[int, int, list[int]]:
    degrees = [0] * len(components)
    row_edges = 0
    col_edges = 0
    for i, a in enumerate(components):
        for j in range(i + 1, len(components)):
            ar0, ac0, ar1, ac1 = a["bbox"]
            br0, bc0, br1, bc1 = components[j]["bbox"]
            share_row = _spans_overlap(ar0, ar1, br0, br1)
            share_col = _spans_overlap(ac0, ac1, bc0, bc1)
            if share_row or share_col:
                degrees[i] += 1
                degrees[j] += 1
            if share_row:
                row_edges += 1
            if share_col:
                col_edges += 1
    return row_edges, col_edges, sorted(degrees)


def _validate_local_group(h: int, w: int, corner: str, mode: str, fragments: list[set[tuple[int, int]]]) -> None:
    patch = np.zeros((h, w), dtype=int)
    control = _control_local(h, w, corner)
    patch[control] = TEAL
    for fragment in fragments:
        for r, c in fragment:
            patch[r, c] = BLUE

    components = find_components(patch, bg=BG, colors=[BLUE], connectivity=4)
    assert len(components) == len(fragments), "fragment templates must stay disconnected"
    assert all(3 <= int(comp["area"]) <= 8 for comp in components), "each fragment must be 3..8 cells"
    assert _merge_fragment_groups(components)[0]["bbox"] == (0, 0, h - 1, w - 1), "merged bbox must recover the full rectangle"
    assert len(_merge_fragment_groups(components)) == 1, "fragments must merge into exactly one group"

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = control[0] + dr, control[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            assert patch[rr, cc] != BLUE, "control corner must be 4-disconnected"

    for comp in components:
        corners = {
            (comp["bbox"][0], comp["bbox"][1]),
            (comp["bbox"][0], comp["bbox"][3]),
            (comp["bbox"][2], comp["bbox"][1]),
            (comp["bbox"][2], comp["bbox"][3]),
        }
        assert control not in corners, "control must not be readable from any single fragment"

    row_edges, col_edges, degrees = _component_relation_signature(components)
    if mode == "row2":
        assert len(components) == 2 and row_edges == 1 and col_edges == 0, "row2 must rely on row-span overlap only"
    elif mode == "col2":
        assert len(components) == 2 and row_edges == 0 and col_edges == 1, "col2 must rely on column-span overlap only"
    else:
        assert len(components) == 3 and row_edges + col_edges == 2 and degrees == [1, 1, 2], "chain3 must require transitive merging"


def _shortcut_without_completion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves = _group_moves(g)
    for move in moves:
        r0, c0, r1, c1 = move["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for move in moves:
        dr, dc = move["dir"]
        for r, c in map(tuple, move["coords"]):
            out[r + dr, c + dc] = BLUE
    return out


def _render_sample(group_specs: list[dict], grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid_shape
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for spec in group_specs:
        r0, c0 = spec["r0"], spec["c0"]
        gh, gw = spec["h"], spec["w"]
        control = _control_local(gh, gw, spec["corner"])
        inp[r0 + control[0], c0 + control[1]] = TEAL
        for fragment in spec["fragments"]:
            for r, c in fragment:
                inp[r0 + r, c0 + c] = BLUE

        dr, dc = CORNER_TO_DIR[spec["corner"]]
        expected[r0 + dr : r0 + dr + gh, c0 + dc : c0 + dc + gw] = BLUE

    return inp, expected


def _validate_sample(inp: np.ndarray, expected: np.ndarray, group_specs: list[dict]) -> None:
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output must match constructed target"
    assert not np.array_equal(_shortcut_without_completion(inp), expected), "completion step must matter"

    all_blue_components = find_components(inp, bg=BG, colors=[BLUE], connectivity=4)
    for comp in all_blue_components:
        corners = {
            (comp["bbox"][0], comp["bbox"][1]),
            (comp["bbox"][0], comp["bbox"][3]),
            (comp["bbox"][2], comp["bbox"][1]),
            (comp["bbox"][2], comp["bbox"][3]),
        }
        assert all(inp[r, c] != TEAL for r, c in corners), "single-fragment shortcut must not see the control"

    control_count = int(np.count_nonzero(inp == TEAL))
    assert control_count == len(group_specs), "exactly one control per group"
    assert int(np.count_nonzero(inp == BLUE)) < int(np.count_nonzero(expected == BLUE)), "input must be incomplete"


def _sample_group_count() -> int:
    return random.choice([2, 3, 3, 4])


def _sample_sizes(group_count: int) -> tuple[list[int], list[int]]:
    if group_count == 4:
        heights = [random.randint(4, 5) for _ in range(group_count)]
        widths = [random.randint(4, 5) for _ in range(group_count)]
    else:
        heights = [random.randint(MIN_SIDE, MAX_SIDE) for _ in range(group_count)]
        widths = [random.randint(MIN_SIDE, MAX_SIDE) for _ in range(group_count)]
    return heights, widths


def _place_bands(lengths: list[int], total: int) -> list[int]:
    extras = _random_partition(total - (2 + sum(lengths) + BASE_GAP * (len(lengths) - 1)), len(lengths) + 1)
    starts: list[int] = []
    cursor = 1 + extras[0]
    for i, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if i < len(lengths) - 1:
            cursor += BASE_GAP + extras[i + 1]
    assert cursor + 1 + extras[-1] == total
    return starts


def _build_group_specs(preferred_mode: str) -> tuple[list[dict], tuple[int, int]] | None:
    group_count = _sample_group_count()
    heights, widths = _sample_sizes(group_count)

    row_need = 2 + sum(heights) + BASE_GAP * (group_count - 1)
    col_need = 2 + sum(widths) + BASE_GAP * (group_count - 1)
    if row_need > MAX_GRID_SIDE or col_need > MAX_GRID_SIDE:
        return None

    grid_h = random.randint(max(16, row_need), min(MAX_GRID_SIDE, row_need + 4))
    grid_w = random.randint(max(16, col_need), min(MAX_GRID_SIDE, col_need + 4))
    row_starts = _place_bands(heights, grid_h)
    col_starts = _place_bands(widths, grid_w)

    focus_idx = random.randrange(group_count)
    group_specs: list[dict] = []
    preferred_used = False

    for i in range(group_count):
        h, w = heights[i], widths[i]
        feasible = _mode_candidates(h, w)
        if i == focus_idx:
            if preferred_mode not in feasible:
                return None
            mode = preferred_mode
            preferred_used = True
        else:
            weighted = feasible + [m for m in feasible if m != preferred_mode]
            mode = random.choice(weighted)

        corner = random.choice(CORNERS)
        fragments = _build_local_fragments(h, w, corner, mode)
        _validate_local_group(h, w, corner, mode, fragments)
        group_specs.append(
            {
                "r0": row_starts[i],
                "c0": col_starts[i],
                "h": h,
                "w": w,
                "corner": corner,
                "mode": mode,
                "fragments": fragments,
            }
        )

    if not preferred_used:
        return None
    return group_specs, (grid_h, grid_w)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    preferred_mode = PREFERRED_MODE_CYCLE[(_GEN_CALL_COUNT - 1) % len(PREFERRED_MODE_CYCLE)]

    for _ in range(GENERATION_ATTEMPTS):
        built = _build_group_specs(preferred_mode)
        if built is None:
            continue
        group_specs, shape = built
        inp, expected = _render_sample(group_specs, shape)
        try:
            _validate_sample(inp, expected, group_specs)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid fragmented-merge diagonal-shift sample")
