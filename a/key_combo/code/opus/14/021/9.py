from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CORNER_ORDER = ("TL", "TR", "BL", "BR")
CORNER_OFFSETS = {
    "TL": {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)},
    "TR": {"TL": (0, -1), "TR": (0, 0), "BL": (1, -1), "BR": (1, 0)},
    "BL": {"TL": (-1, 0), "TR": (-1, 1), "BL": (0, 0), "BR": (0, 1)},
    "BR": {"TL": (-1, -1), "TR": (-1, 0), "BL": (0, -1), "BR": (0, 0)},
}
ALT_OFFSETS = {
    "TL": ((0, -1), (-1, 0), (-1, -1)),
    "TR": ((0, 1), (-1, 0), (-1, 1)),
    "BL": ((0, -1), (1, 0), (1, -1)),
    "BR": ((0, 1), (1, 0), (1, 1)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(counts.argmax())


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS8 if connectivity == 8 else ((-1, 0), (1, 0), (0, -1), (0, 1))
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _infer_square(marker_positions: list[tuple[int, int]]):
    marker_set = set(marker_positions)
    if len(marker_set) != 3:
        return None
    rows = sorted({r for r, _ in marker_set})
    cols = sorted({c for _, c in marker_set})
    if len(rows) != 2 or len(cols) != 2:
        return None
    r0, r1 = rows
    c0, c1 = cols
    if r1 <= r0 or c1 <= c0 or (r1 - r0) != (c1 - c0):
        return None
    if (r0 + r1) % 2 != 0 or (c0 + c1) % 2 != 0:
        return None
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if marker_set - corners:
        return None
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "markers": tuple(sorted(marker_set)),
        "corners": corners,
        "center": center,
        "bbox": (r0, c0, r1, c1),
        "side": r1 - r0,
    }


def _attached_markers(component: list[tuple[int, int]], markers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    comp_set = set(component)
    attached: list[tuple[int, int]] = []
    for pos in markers:
        if any(nb in comp_set for nb in _neighbors8(*pos)):
            attached.append(pos)
    return attached


def _find_valid_square(grid: np.ndarray):
    bg = _dominant_color(grid)
    grey_mask = np.asarray(grid == GREY)
    markers = [tuple(map(int, rc)) for rc in np.argwhere((grid != bg) & (grid != GREY))]
    assert len(markers) >= 3, "expected at least three non-grey markers"

    candidates = []
    for component in _components_from_mask(grey_mask, connectivity=8):
        attached = _attached_markers(component, markers)
        if len(attached) != 3:
            continue
        square = _infer_square(attached)
        if square is not None:
            candidates.append(square)
    assert len(candidates) == 1, "expected exactly one path-connected square-defining marker triple"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter path-connected corner markers, infer their square, and draw a shrunken copy at the center.

    Input structure:
    - Black background with several single-pixel colored markers and one or more grey path components.
    - Exactly one grey 8-connected path component touches exactly three markers, and those three markers occupy three corners of one axis-aligned square.
    - Remaining markers are distractors that are either off the grey network or only touch separate grey fragments.

    Rule:
    - Use grey-path 8-connectivity to keep only the unique marker triple that is mutually linked by the same grey component.
    - Infer the missing fourth square corner and compute the square center from the three kept markers.
    - Erase every grey path cell and every distractor marker.
    - Preserve the three valid corner markers, paint one grey center pixel, and paint each marker's color one diagonal step from the center toward that marker.

    Key insight:
    - The geometry is intentionally ambiguous until grey-path connectivity filters out distractors; only then do the remaining three markers uniquely define the square and its inward miniature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    square = _find_valid_square(g)
    center_r, center_c = square["center"]
    valid_markers = list(square["markers"])

    out = np.full_like(g, bg)
    for r, c in valid_markers:
        color = int(g[r, c])
        out[r, c] = color
        dr = -1 if r < center_r else 1
        dc = -1 if c < center_c else 1
        out[center_r + dr, center_c + dc] = color
    out[center_r, center_c] = GREY
    return out


def _square_triples(markers: list[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    triples: set[tuple[tuple[int, int], ...]] = set()
    for triple in itertools.combinations(sorted(markers), 3):
        if _infer_square(list(triple)) is not None:
            triples.add(tuple(sorted(triple)))
    return triples


def _build_square_from_anchor(anchor_kind: str, anchor: tuple[int, int], side: int) -> dict[str, tuple[int, int]]:
    ar, ac = anchor
    return {
        label: (ar + side * dr, ac + side * dc)
        for label, (dr, dc) in CORNER_OFFSETS[anchor_kind].items()
    }


def _anchor_limits(anchor_kind: str, h: int, w: int, side: int, alt_side: int):
    if anchor_kind == "TL":
        return alt_side + 1, h - side - 2, alt_side + 1, w - side - 2
    if anchor_kind == "TR":
        return alt_side + 1, h - side - 2, side + 1, w - alt_side - 2
    if anchor_kind == "BL":
        return side + 1, h - alt_side - 2, alt_side + 1, w - side - 2
    return side + 1, h - alt_side - 2, side + 1, w - alt_side - 2


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _render_expected(
    shape: tuple[int, int],
    valid_color_map: dict[tuple[int, int], int],
    center: tuple[int, int],
    bg: int = BG,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    center_r, center_c = center
    for (r, c), color in valid_color_map.items():
        out[r, c] = int(color)
        dr = -1 if r < center_r else 1
        dc = -1 if c < center_c else 1
        out[center_r + dr, center_c + dc] = int(color)
    out[center_r, center_c] = GREY
    return out


def _random_free_cell(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    center: tuple[int, int],
    min_center_gap: int,
    existing: list[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in blocked:
                continue
            if _chebyshev(pos, center) <= min_center_gap:
                continue
            if any(_chebyshev(pos, other) <= 1 for other in existing):
                continue
            candidates.append(pos)
    if not candidates:
        return None
    idx = int(np.random.randint(0, len(candidates)))
    return candidates[idx]


def _choose_ports(
    shape: tuple[int, int],
    valid_markers: list[tuple[int, int]],
    center: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int]] | None:
    ports: dict[tuple[int, int], tuple[int, int]] = {}
    taken = set(blocked)
    center_r, center_c = center
    for r, c in valid_markers:
        dr = -1 if r < center_r else 1
        dc = -1 if c < center_c else 1
        candidates = [(r + dr, c), (r, c + dc), (r + dr, c + dc)]
        candidates = [
            pos
            for pos in candidates
            if _in_bounds(shape, pos[0], pos[1]) and pos not in taken
        ]
        if not candidates:
            return None
        port = candidates[int(np.random.randint(0, len(candidates)))]
        ports[(r, c)] = port
        taken.add(port)
    return ports


def _random_bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    reusable: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parents = {start: None}
    dir_order = list(DIRS8)
    np.random.shuffle(dir_order)

    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in dir_order:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if nxt in parents or not _in_bounds(shape, nr, nc):
                continue
            if nxt != goal and nxt not in reusable and nxt in blocked:
                continue
            parents[nxt] = cur
            queue.append(nxt)

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _build_valid_path(
    shape: tuple[int, int],
    valid_markers: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    square_bbox: tuple[int, int, int, int],
    center: tuple[int, int],
) -> set[tuple[int, int]] | None:
    blocked = set(valid_markers) | set(distractors)
    r0, c0, r1, c1 = square_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            blocked.add((r, c))
    for rr in range(center[0] - 1, center[0] + 2):
        for cc in range(center[1] - 1, center[1] + 2):
            if _in_bounds(shape, rr, cc):
                blocked.add((rr, cc))
    for pos in distractors:
        for rr in range(pos[0] - 1, pos[0] + 2):
            for cc in range(pos[1] - 1, pos[1] + 2):
                if _in_bounds(shape, rr, cc):
                    blocked.add((rr, cc))

    ports = _choose_ports(shape, valid_markers, center, blocked)
    if ports is None:
        return None

    hub_candidates = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            pos = (r, c)
            if pos in blocked or pos in ports.values():
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if any(_chebyshev(pos, marker) <= 1 for marker in valid_markers + distractors):
                continue
            hub_candidates.append(pos)
    if not hub_candidates:
        return None

    np.random.shuffle(hub_candidates)
    for hub in hub_candidates[:50]:
        grey_cells = set(ports.values())
        ok = True
        order = list(ports.values())
        np.random.shuffle(order)
        for port in order:
            path = _random_bfs_path(shape, port, hub, blocked, grey_cells | {hub})
            if path is None:
                ok = False
                break
            grey_cells.update(path)
        if ok and len(grey_cells) >= 6:
            return grey_cells
    return None


def _attach_stub(
    grid: np.ndarray,
    marker: tuple[int, int],
    other_markers: set[tuple[int, int]],
    protected_grey: set[tuple[int, int]],
    max_len: int = 2,
) -> bool:
    first_candidates = []
    for pos in _neighbors8(*marker):
        if not _in_bounds(grid.shape, pos[0], pos[1]) or grid[pos] != BG:
            continue
        if any(_chebyshev(pos, other) <= 1 for other in other_markers):
            continue
        if any(_chebyshev(pos, grey) <= 1 for grey in protected_grey):
            continue
        first_candidates.append(pos)
    if not first_candidates:
        return False

    first = first_candidates[int(np.random.randint(0, len(first_candidates)))]
    grid[first] = GREY
    if max_len <= 1 or np.random.randint(0, 2) == 0:
        return True

    second_candidates = []
    for pos in _neighbors8(*first):
        if pos == marker or not _in_bounds(grid.shape, pos[0], pos[1]) or grid[pos] != BG:
            continue
        if any(_chebyshev(pos, other) <= 1 for other in other_markers):
            continue
        if any(_chebyshev(pos, grey) <= 1 for grey in protected_grey):
            continue
        second_candidates.append(pos)
    if second_candidates:
        second = second_candidates[int(np.random.randint(0, len(second_candidates)))]
        grid[second] = GREY
    return True


def _validate_case(
    grid: np.ndarray,
    valid_markers: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    all_markers = [tuple(map(int, rc)) for rc in np.argwhere((grid != BG) & (grid != GREY))]
    if len(all_markers) < 5 or len(all_markers) > 7:
        return False

    square_triples = _square_triples(all_markers)
    if len(square_triples) < 2 or tuple(sorted(valid_markers)) not in square_triples:
        return False

    grey_components = _components_from_mask(grid == GREY, connectivity=8)
    attached_sets = [tuple(sorted(_attached_markers(comp, all_markers))) for comp in grey_components]
    target = tuple(sorted(valid_markers))
    if attached_sets.count(target) != 1:
        return False
    if sum(1 for attached in attached_sets if len(attached) == 3) != 1:
        return False

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if np.count_nonzero(grid == GREY) < 6:
        return False
    if any(produced[pos] != BG for pos in set(all_markers) - set(valid_markers)):
        return False
    return True


def _generate_layout():
    h = int(np.random.randint(18, 25))
    w = int(np.random.randint(18, 25))
    anchor_kind = CORNER_ORDER[int(np.random.randint(0, len(CORNER_ORDER)))]

    choices = []
    for side in (6, 8, 10):
        for alt_side in (6, 8, 10):
            r_min, r_max, c_min, c_max = _anchor_limits(anchor_kind, h, w, side, alt_side)
            if r_min <= r_max and c_min <= c_max:
                choices.append((side, alt_side, r_min, r_max, c_min, c_max))
    if not choices:
        return None

    side, alt_side, r_min, r_max, c_min, c_max = choices[int(np.random.randint(0, len(choices)))]
    anchor = (
        int(np.random.randint(r_min, r_max + 1)),
        int(np.random.randint(c_min, c_max + 1)),
    )
    corners = _build_square_from_anchor(anchor_kind, anchor, side)
    missing = np.random.choice([label for label in CORNER_ORDER if label != anchor_kind])
    valid_markers = [corners[label] for label in CORNER_ORDER if label != missing]
    square = _infer_square(valid_markers)
    assert square is not None

    alt_positions = []
    for dr, dc in ALT_OFFSETS[anchor_kind][:2]:
        alt_positions.append((anchor[0] + alt_side * dr, anchor[1] + alt_side * dc))
    return {
        "shape": (h, w),
        "anchor_kind": anchor_kind,
        "side": side,
        "alt_side": alt_side,
        "anchor": anchor,
        "valid_markers": valid_markers,
        "center": square["center"],
        "square_bbox": square["bbox"],
        "base_distractors": alt_positions,
        "alt_diagonal": (
            anchor[0] + alt_side * ALT_OFFSETS[anchor_kind][2][0],
            anchor[1] + alt_side * ALT_OFFSETS[anchor_kind][2][1],
        ),
    }


def generate_input() -> np.ndarray:
    for _ in range(600):
        layout = _generate_layout()
        if layout is None:
            continue

        target_marker_count = int(np.random.randint(5, 8))
        include_alt_diagonal = target_marker_count >= 6 and bool(np.random.randint(0, 2))
        distractors = list(layout["base_distractors"])
        if include_alt_diagonal:
            distractors.append(layout["alt_diagonal"])
        extra_needed = target_marker_count - 3 - len(distractors)
        if extra_needed < 0:
            continue

        occupied = list(layout["valid_markers"]) + distractors
        blocked_for_extra = set(occupied)
        for r in range(layout["square_bbox"][0] - 1, layout["square_bbox"][2] + 2):
            for c in range(layout["square_bbox"][1] - 1, layout["square_bbox"][3] + 2):
                if _in_bounds(layout["shape"], r, c):
                    blocked_for_extra.add((r, c))

        extras = []
        for _extra_idx in range(extra_needed):
            cell = _random_free_cell(
                layout["shape"],
                blocked_for_extra,
                layout["center"],
                min_center_gap=4,
                existing=occupied + extras,
            )
            if cell is None:
                extras = []
                break
            extras.append(cell)
            blocked_for_extra.add(cell)
        if len(extras) != extra_needed:
            continue

        distractors.extend(extras)
        all_markers = list(layout["valid_markers"]) + distractors
        if len(set(all_markers)) != len(all_markers):
            continue

        grey_cells = _build_valid_path(
            layout["shape"],
            list(layout["valid_markers"]),
            distractors,
            layout["square_bbox"],
            layout["center"],
        )
        if grey_cells is None:
            continue

        grid = np.full(layout["shape"], BG, dtype=int)
        colors = np.random.choice(PALETTE, size=len(all_markers), replace=False)
        valid_color_map = {
            pos: int(color) for pos, color in zip(layout["valid_markers"], colors[:3], strict=True)
        }
        distractor_color_map = {
            pos: int(color) for pos, color in zip(distractors, colors[3:], strict=True)
        }
        for pos, color in valid_color_map.items():
            grid[pos] = color
        for pos, color in distractor_color_map.items():
            grid[pos] = color
        for pos in grey_cells:
            grid[pos] = GREY

        expected = _render_expected(layout["shape"], valid_color_map, layout["center"], bg=BG)

        valid_stub_candidates = list(distractors)
        np.random.shuffle(valid_stub_candidates)
        attach_count = min(len(valid_stub_candidates), int(np.random.randint(0, 3)))
        for marker in valid_stub_candidates[:attach_count]:
            _attach_stub(
                grid,
                marker,
                set(all_markers) - {marker},
                set(grey_cells),
                max_len=2,
            )

        if _validate_case(grid, list(layout["valid_markers"]), expected):
            return grid

    raise RuntimeError("failed to generate a valid path-filtered square case")
