from __future__ import annotations

import random
from collections import defaultdict, deque
from functools import lru_cache

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PIECE_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_RETRIES = 400

TEMPLATES_BY_SIZE: dict[tuple[int, int], tuple[tuple[str, ...], ...]] = {
    (4, 4): (
        ("ACCC", "ACBB", "ACAB", "AAAB"),
        ("AAAA", "AABB", "CCBC", "CCCC"),
        ("BBBB", "ACCB", "AACB", "AABB"),
        ("BBAA", "CBAA", "CBBA", "CCAA"),
    ),
    (4, 5): (
        ("AAAAC", "AACAC", "BBCCC", "BBBBC"),
        ("CCCCC", "BCAAC", "BBAAC", "BBBBB"),
        ("AACCB", "AACBB", "ACCBB", "AABBB"),
        ("AAAAA", "CABBB", "CCBCB", "CCCCB"),
        ("BBBBA", "BAAAA", "CAACC", "CCCCC"),
    ),
    (5, 4): (
        ("CAAA", "CCBA", "CBBA", "CCBA", "CCAA"),
        ("AAAA", "ABAA", "CBBB", "CCCB", "CCCC"),
        ("BBBA", "CBAA", "CBBA", "CCAA", "CCCC"),
        ("CCCC", "ABBC", "AABC", "ABBC", "AAAA"),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _is_hollow_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int, bg: int) -> bool:
    if color == bg or len(coords) < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _find_reference_frame(grid: np.ndarray, bg: int) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.ravel())) - {bg, YELLOW}):
        for comp in _components(g == int(color)):
            if not _is_hollow_frame(g, comp, int(color), bg):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            candidates.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "cells": frozenset((int(r), int(c)) for r, c in comp),
                    "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                }
            )
    assert candidates, "expected a grey reference frame"
    candidates.sort(key=lambda item: (item["bbox"][2] + item["bbox"][3], item["bbox"][0], item["bbox"][1]))
    return candidates[-1]


def _extract_pieces(grid: np.ndarray, frame_cells: frozenset[tuple[int, int]]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    piece_of: dict[tuple[int, int], int] = {}
    pieces: list[dict[str, object]] = []
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if bool(seen[r, c]) or v in (BG, YELLOW) or (r, c) in frame_cells:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and (nr, nc) not in frame_cells
                        and int(g[nr, nc]) == v
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(comp)
            local_cells = tuple(sorted((int(rr - r0), int(cc - c0)) for rr, cc in comp))
            piece = {
                "color": int(v),
                "cells_abs": tuple(sorted((int(rr), int(cc)) for rr, cc in comp)),
                "input_top_left": (int(r0), int(c0)),
                "local_cells": local_cells,
                "bbox_shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                "area": int(len(comp)),
                "markers_abs": [],
            }
            piece_idx = len(pieces)
            for cell in comp:
                piece_of[cell] = piece_idx
            pieces.append(piece)

    for r, c in np.argwhere(g == YELLOW):
        rr = int(r)
        cc = int(c)
        owners = {piece_of[(nr, nc)] for nr, nc in _neighbors4(rr, cc) if (nr, nc) in piece_of}
        if len(owners) != 1:
            continue
        owner = next(iter(owners))
        pieces[owner]["markers_abs"].append((rr, cc))

    for piece in pieces:
        r0, c0 = piece["input_top_left"]
        markers_abs = sorted(set((int(r), int(c)) for r, c in piece["markers_abs"]))
        piece["markers_abs"] = tuple(markers_abs)
        piece["local_markers"] = tuple((int(r - r0), int(c - c0)) for r, c in markers_abs)
        assert piece["markers_abs"], "each puzzle fragment should have yellow interfaces"

    return pieces


def _bit_index(r: int, c: int, width: int) -> int:
    return int(r * width + c)


def _make_piece_placements(piece: dict[str, object], target_h: int, target_w: int) -> list[dict[str, object]]:
    cells = tuple((int(r), int(c)) for r, c in piece["local_cells"])
    markers = tuple((int(r), int(c)) for r, c in piece["local_markers"])
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    out: list[dict[str, object]] = []
    for dr in range(target_h - h + 1):
        for dc in range(target_w - w + 1):
            placed_cells = tuple((int(r + dr), int(c + dc)) for r, c in cells)
            placed_cell_set = set(placed_cells)
            placed_markers = tuple((int(r + dr), int(c + dc)) for r, c in markers)
            if not all(0 <= r < target_h and 0 <= c < target_w for r, c in placed_markers):
                continue
            if any((r, c) in placed_cell_set for r, c in placed_markers):
                continue
            mask = 0
            for r, c in placed_cells:
                mask |= 1 << _bit_index(int(r), int(c), target_w)
            out.append(
                {
                    "dr": int(dr),
                    "dc": int(dc),
                    "cells": placed_cells,
                    "markers": placed_markers,
                    "mask": int(mask),
                }
            )
    assert out, "every piece should have at least one legal placement"
    return out


def _first_empty_index(occ_mask: int, full_mask: int) -> int:
    remaining = full_mask & ~occ_mask
    bit = remaining & -remaining
    return int(bit.bit_length() - 1)


def _group_solution(
    pieces: list[dict[str, object]],
    placements_used: dict[int, dict[str, object]],
    target_h: int,
    target_w: int,
) -> dict[str, object] | None:
    piece_ids = tuple(sorted(placements_used))
    owner_of: dict[tuple[int, int], int] = {}
    image = np.full((target_h, target_w), BG, dtype=int)
    for piece_idx in piece_ids:
        color = int(pieces[piece_idx]["color"])
        for r, c in placements_used[piece_idx]["cells"]:
            owner_of[(int(r), int(c))] = int(piece_idx)
            image[int(r), int(c)] = color

    marker_map: dict[tuple[int, int], set[int]] = defaultdict(set)
    for piece_idx in piece_ids:
        for r, c in placements_used[piece_idx]["markers"]:
            coord = (int(r), int(c))
            owner = owner_of.get(coord)
            if owner is None or owner == piece_idx:
                return None
            marker_map[coord].add(int(piece_idx))

    adjacency = {piece_idx: set() for piece_idx in piece_ids}
    for coord_set in marker_map.values():
        members = sorted(coord_set)
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                adjacency[left].add(int(right))
                adjacency[right].add(int(left))

    for piece_idx in piece_ids:
        for coord in placements_used[piece_idx]["markers"]:
            if len(marker_map[(int(coord[0]), int(coord[1]))]) < 2:
                return None

    start = piece_ids[0]
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if seen != set(piece_ids):
        return None

    return {
        "piece_ids": piece_ids,
        "placements": {int(k): v for k, v in placements_used.items()},
        "image": image,
        "color_count": int(len({int(pieces[idx]["color"]) for idx in piece_ids})),
        "input_anchor": min(tuple(map(int, pieces[idx]["input_top_left"])) for idx in piece_ids),
    }


def _solve_partition(pieces: list[dict[str, object]], target_h: int, target_w: int) -> list[dict[str, object]]:
    area = int(target_h * target_w)
    full_mask = (1 << area) - 1
    placements_by_piece = [_make_piece_placements(piece, target_h, target_w) for piece in pieces]
    cover_lookup: list[dict[int, list[dict[str, object]]]] = []
    for placements in placements_by_piece:
        by_cell: dict[int, list[dict[str, object]]] = defaultdict(list)
        for placement in placements:
            for r, c in placement["cells"]:
                by_cell[_bit_index(int(r), int(c), target_w)].append(placement)
        cover_lookup.append(dict(by_cell))

    @lru_cache(maxsize=None)
    def _candidate_groups(remaining: tuple[int, ...]) -> tuple[dict[str, object], ...]:
        anchor = int(remaining[0])
        remaining_set = set(map(int, remaining))
        solutions: list[dict[str, object]] = []
        seen_sigs: set[tuple[tuple[int, int, int], ...]] = set()

        def backtrack(occ_mask: int, used: dict[int, dict[str, object]]) -> None:
            if occ_mask == full_mask:
                solution = _group_solution(pieces, used, target_h, target_w)
                if solution is None:
                    return
                sig = tuple(
                    sorted(
                        (int(piece_idx), int(data["dr"]), int(data["dc"]))
                        for piece_idx, data in solution["placements"].items()
                    )
                )
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    solutions.append(solution)
                return

            cell_idx = _first_empty_index(occ_mask, full_mask)
            for piece_idx in remaining:
                if piece_idx in used:
                    continue
                for placement in cover_lookup[piece_idx].get(cell_idx, []):
                    if int(placement["mask"]) & int(occ_mask):
                        continue
                    used[piece_idx] = placement
                    backtrack(int(occ_mask | int(placement["mask"])), used)
                    used.pop(piece_idx)

        for anchor_placement in placements_by_piece[anchor]:
            used_map = {anchor: anchor_placement}
            backtrack(int(anchor_placement["mask"]), used_map)

        solutions.sort(key=lambda item: (len(item["piece_ids"]), item["input_anchor"], item["piece_ids"]))
        return tuple(solutions)

    @lru_cache(maxsize=None)
    def _dfs(remaining: tuple[int, ...]) -> tuple[dict[str, object], ...] | None:
        if not remaining:
            return tuple()
        remaining_set = set(map(int, remaining))
        for group in _candidate_groups(remaining):
            next_remaining = tuple(sorted(remaining_set - set(group["piece_ids"])))
            tail = _dfs(next_remaining)
            if tail is not None:
                return (group,) + tail
        return None

    result = _dfs(tuple(range(len(pieces))))
    assert result is not None, "failed to partition fragments into valid rectangles"
    return list(result)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble same-group fragments into full rectangles using shared yellow interface points,
    then sort the rectangles by the number of distinct colors they contain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _find_reference_frame(g, bg)
    target_h, target_w = map(int, frame["shape"])
    pieces = _extract_pieces(g, frame["cells"])
    groups = _solve_partition(pieces, target_h, target_w)
    groups.sort(key=lambda item: (int(item["color_count"]), item["input_anchor"][0], item["input_anchor"][1]))

    out_w = len(groups) * target_w + max(0, len(groups) - 1)
    out = np.full((target_h, out_w), bg, dtype=int)
    col = 0
    for group in groups:
        out[:, col : col + target_w] = group["image"]
        col += target_w + 1
    return out


def _apply_grid_transform(rows: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    arr = np.array([list(row) for row in rows], dtype="<U1")
    if rng.random() < 0.5:
        arr = np.flipud(arr)
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    labels = ["A", "B", "C"]
    perm = labels[:]
    rng.shuffle(perm)
    mapping = {labels[i]: perm[i] for i in range(3)}
    out = np.empty_like(arr)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[r, c] = mapping[str(arr[r, c])]
    return tuple("".join(out[r]) for r in range(out.shape[0]))


def _cells_from_template(rows: tuple[str, ...]) -> dict[str, list[tuple[int, int]]]:
    out = {"A": [], "B": [], "C": []}
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            out[str(value)].append((int(r), int(c)))
    return out


def _shared_connector_candidates(
    cells_by_label: dict[str, list[tuple[int, int]]],
    owner: str,
    left: str,
    right: str,
) -> list[tuple[int, int]]:
    left_cells = set(cells_by_label[left])
    right_cells = set(cells_by_label[right])
    out: list[tuple[int, int]] = []
    for r, c in cells_by_label[owner]:
        neighbors = {(r + dr, c + dc) for dr, dc in DIRS4}
        if neighbors & left_cells and neighbors & right_cells:
            out.append((int(r), int(c)))
    return out


def _color_assignment(kind: int, rng: random.Random) -> dict[str, int]:
    if kind == 1:
        color = int(rng.choice(PIECE_COLORS))
        return {"A": color, "B": color, "C": color}
    if kind == 2:
        shared, distinct = rng.sample(PIECE_COLORS, 2)
        odd_label = rng.choice(["A", "B", "C"])
        mapping = {"A": int(shared), "B": int(shared), "C": int(shared)}
        mapping[str(odd_label)] = int(distinct)
        return mapping
    colors = rng.sample(PIECE_COLORS, 3)
    return {"A": int(colors[0]), "B": int(colors[1]), "C": int(colors[2])}


def _build_group(size: tuple[int, int], color_kind: int, rng: random.Random) -> dict[str, object]:
    rows = _apply_grid_transform(rng.choice(TEMPLATES_BY_SIZE[size]), rng)
    cells_by_label = _cells_from_template(rows)

    connectors = {
        ("A", "B"): rng.choice(_shared_connector_candidates(cells_by_label, "C", "A", "B")),
        ("A", "C"): rng.choice(_shared_connector_candidates(cells_by_label, "B", "A", "C")),
        ("B", "C"): rng.choice(_shared_connector_candidates(cells_by_label, "A", "B", "C")),
    }
    colors = _color_assignment(color_kind, rng)
    h, w = size
    assembled = np.full((h, w), BG, dtype=int)
    for label, color in colors.items():
        for r, c in cells_by_label[label]:
            assembled[int(r), int(c)] = int(color)

    markers_by_label = {
        "A": {tuple(connectors[("A", "B")]), tuple(connectors[("A", "C")])},
        "B": {tuple(connectors[("A", "B")]), tuple(connectors[("B", "C")])},
        "C": {tuple(connectors[("A", "C")]), tuple(connectors[("B", "C")])},
    }

    pieces: list[dict[str, object]] = []
    for label in ("A", "B", "C"):
        cells = tuple(sorted((int(r), int(c)) for r, c in cells_by_label[label]))
        markers = tuple(sorted((int(r), int(c)) for r, c in markers_by_label[label]))
        cell_set = set(cells)
        assert all(marker not in cell_set for marker in markers)
        assert all(
            any((marker[0] + dr, marker[1] + dc) in cell_set for dr, dc in DIRS4)
            for marker in markers
        )
        pieces.append({"color": int(colors[label]), "cells": cells, "markers": markers})

    return {"pieces": pieces, "assembled": assembled, "color_count": int(len(set(colors.values())))}


def _footprint(piece: dict[str, object], height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    for r, c in piece["markers"]:
        patch[int(r), int(c)] = YELLOW
    for r, c in piece["cells"]:
        patch[int(r), int(c)] = int(piece["color"])
    return patch


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    grid[r0, c0 : c0 + width] = GREY
    grid[r0 + height - 1, c0 : c0 + width] = GREY
    grid[r0 : r0 + height, c0] = GREY
    grid[r0 : r0 + height, c0 + width - 1] = GREY


def _compose_output(groups: list[dict[str, object]], height: int, width: int) -> np.ndarray:
    ordered = sorted(groups, key=lambda item: item["color_count"])
    out = np.full((height, len(ordered) * width + len(ordered) - 1), BG, dtype=int)
    col = 0
    for group in ordered:
        out[:, col : col + width] = group["assembled"]
        col += width + 1
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    size = rng.choice(list(TEMPLATES_BY_SIZE))
    height, width = map(int, size)
    groups = [_build_group(size, color_kind, rng) for color_kind in (1, 2, 3)]

    row_gaps = [rng.randint(2, 4) for _ in range(2)]
    col_gaps = [rng.randint(2, 4) for _ in range(2)]
    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    slot_rows = [top_margin]
    slot_cols = [left_margin]
    for gap in row_gaps:
        slot_rows.append(int(slot_rows[-1] + height + gap))
    for gap in col_gaps:
        slot_cols.append(int(slot_cols[-1] + width + gap))

    frame_gap_r = rng.randint(3, 5)
    frame_gap_c = rng.randint(3, 5)
    frame_r0 = int(slot_rows[-1] + height + frame_gap_r)
    frame_c0 = int(slot_cols[-1] + width + frame_gap_c)
    canvas_h = int(frame_r0 + height)
    canvas_w = int(frame_c0 + width)
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    flat_pieces = []
    for group_idx, group in enumerate(groups):
        for piece in group["pieces"]:
            flat_pieces.append({"group_idx": int(group_idx), "piece": piece})
    rng.shuffle(flat_pieces)

    slots = [(int(r), int(c)) for r in slot_rows for c in slot_cols]
    assert len(slots) == len(flat_pieces)
    for item, (r0, c0) in zip(flat_pieces, slots, strict=True):
        patch = _footprint(item["piece"], height, width)
        grid[r0 : r0 + height, c0 : c0 + width] = np.where(
            patch != BG,
            patch,
            grid[r0 : r0 + height, c0 : c0 + width],
        )

    _draw_frame(grid, frame_r0, frame_c0, height, width)
    expected = _compose_output(groups, height, width)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        inp, expected = _generate_case(rng)
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            pass
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
