from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
RED = 2
FRAME = 5
FILL_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _line_offsets(sizes: list[int]) -> list[int]:
    offsets = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        offsets.append(pos)
    return offsets


def _slot_positions(slot_shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = slot_shape
    return [(sr, sc) for sr in range(rows) for sc in range(cols)]


def _parse_frames(grid: np.ndarray) -> tuple[list[dict], tuple[int, int]]:
    frames: list[dict] = []
    slot_shape: tuple[int, int] | None = None
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox_coords(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame_mask = sub == FRAME

        row_lines = [r for r in range(frame_mask.shape[0]) if bool(frame_mask[r, :].all())]
        col_lines = [c for c in range(frame_mask.shape[1]) if bool(frame_mask[:, c].all())]
        assert len(row_lines) >= 2 and len(col_lines) >= 2, "frame must contain row and col dividers"
        assert row_lines[0] == 0 and row_lines[-1] == frame_mask.shape[0] - 1
        assert col_lines[0] == 0 and col_lines[-1] == frame_mask.shape[1] - 1
        assert all(a + 1 < b for a, b in zip(row_lines, row_lines[1:]))
        assert all(a + 1 < b for a, b in zip(col_lines, col_lines[1:]))

        expected = np.zeros_like(frame_mask, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        assert np.array_equal(frame_mask, expected), "frame must be a complete lattice"

        abs_rows = [r0 + r for r in row_lines]
        abs_cols = [c0 + c for c in col_lines]
        this_shape = (len(abs_rows) - 1, len(abs_cols) - 1)
        if slot_shape is None:
            slot_shape = this_shape
        else:
            assert slot_shape == this_shape, "all frames must be slot-isomorphic"

        slot_colors: dict[tuple[int, int], int] = {}
        for sr in range(this_shape[0]):
            for sc in range(this_shape[1]):
                rr0, rr1 = abs_rows[sr] + 1, abs_rows[sr + 1]
                cc0, cc1 = abs_cols[sc] + 1, abs_cols[sc + 1]
                region = grid[rr0:rr1, cc0:cc1]
                vals = np.unique(region[(region != BG) & (region != FRAME)])
                assert vals.size <= 1, "a slot can only contain one visible color"
                if vals.size:
                    color = int(vals[0])
                    assert np.all((region == BG) | (region == color)), "slot fill must be uniform"
                    slot_colors[(sr, sc)] = color

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "rows": abs_rows,
                "cols": abs_cols,
                "slot_colors": slot_colors,
            }
        )

    assert frames, "expected at least one frame"
    frames.sort(key=lambda frame: frame["bbox"])
    assert slot_shape is not None
    return frames, slot_shape


def _is_horizontal_corridor(grid: np.ndarray, r: int) -> bool:
    if int(grid[r, 0]) != RED or int(grid[r, -1]) != RED:
        return False
    inner = grid[r, 1:-1]
    return bool(np.all((inner == BG) | (inner == FRAME)))


def _is_vertical_corridor(grid: np.ndarray, c: int) -> bool:
    if int(grid[0, c]) != RED or int(grid[-1, c]) != RED:
        return False
    inner = grid[1:-1, c]
    return bool(np.all((inner == BG) | (inner == FRAME)))


def _corridors(grid: np.ndarray) -> tuple[set[int], set[int]]:
    rows = {r for r in range(grid.shape[0]) if _is_horizontal_corridor(grid, r)}
    cols = {c for c in range(grid.shape[1]) if _is_vertical_corridor(grid, c)}
    return rows, cols


def _group_frames(
    frames: list[dict], corridor_rows: set[int], corridor_cols: set[int]
) -> tuple[list[list[int]], list[bool]]:
    n = len(frames)
    parent = list(range(n))
    touched = [False] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in sorted(corridor_rows):
        hits = [i for i, frame in enumerate(frames) if frame["bbox"][0] <= r <= frame["bbox"][2]]
        for i in hits:
            touched[i] = True
        for i in range(1, len(hits)):
            union(hits[0], hits[i])

    for c in sorted(corridor_cols):
        hits = [i for i, frame in enumerate(frames) if frame["bbox"][1] <= c <= frame["bbox"][3]]
        for i in hits:
            touched[i] = True
        for i in range(1, len(hits)):
            union(hits[0], hits[i])

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if touched[i]:
            groups[find(i)].append(i)

    ordered = sorted((sorted(members) for members in groups.values()), key=lambda members: members[0])
    return ordered, touched


def _infer_group_colors(
    frames: list[dict], group: list[int], slot_shape: tuple[int, int]
) -> dict[tuple[int, int], int]:
    slot_colors: dict[tuple[int, int], int] = {}
    for idx in group:
        for pos, color in frames[idx]["slot_colors"].items():
            if pos in slot_colors:
                assert slot_colors[pos] == color, "group hints contradict each other"
            else:
                slot_colors[pos] = color
    expected = set(_slot_positions(slot_shape))
    assert set(slot_colors) == expected, "group must determine every slot position"
    return slot_colors


def _analyze(grid: np.ndarray) -> dict:
    frames, slot_shape = _parse_frames(grid)
    corridor_rows, corridor_cols = _corridors(grid)
    groups, touched = _group_frames(frames, corridor_rows, corridor_cols)
    group_colors = [_infer_group_colors(frames, group, slot_shape) for group in groups]
    return {
        "frames": frames,
        "slot_shape": slot_shape,
        "corridor_rows": corridor_rows,
        "corridor_cols": corridor_cols,
        "groups": groups,
        "touched": touched,
        "group_colors": group_colors,
    }


def _fill_frame_slots(out: np.ndarray, frame: dict, slot_colors: dict[tuple[int, int], int]) -> None:
    rows = frame["rows"]
    cols = frame["cols"]
    for sr in range(len(rows) - 1):
        for sc in range(len(cols) - 1):
            out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = slot_colors[(sr, sc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each lattice frame from the slot-color scheme shared by frames connected through edge-to-edge corridors.

    Input structure:
    - Black background with several gray one-pixel rectangular lattice frames.
    - Every frame has the same logical slot layout, but only some slot interiors are colored.
    - Red edge markers may pair the left/right ends of a row or the top/bottom ends of a column to mark open corridors.

    Rule:
    - A marked row or column is a corridor iff everything between its two red edge markers is only black background or gray frame line.
    - Any corridor that intersects the bounding boxes of two or more frames connects those frames; take the transitive closure of all such connections.
    - Within each connected group, infer the unique full slot-color assignment from the already colored corresponding slot positions, then fill every empty slot in every frame of that group.
    - Keep gray frame lines, red edge markers, and any frame not touched by any corridor unchanged.

    Key insight:
    - The corridors do not paint anything themselves; they only decide which frames share one logical slot-color codebook.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)
    out = g.copy()
    for group, slot_colors in zip(analysis["groups"], analysis["group_colors"]):
        for idx in group:
            _fill_frame_slots(out, analysis["frames"][idx], slot_colors)
    return out


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    row_lines: list[int],
    col_lines: list[int],
    slot_colors: dict[tuple[int, int], int],
) -> None:
    h = row_lines[-1] + 1
    w = col_lines[-1] + 1
    for rr in row_lines:
        grid[top + rr, left : left + w] = FRAME
    for cc in col_lines:
        grid[top : top + h, left + cc] = FRAME
    for (sr, sc), color in slot_colors.items():
        grid[
            top + row_lines[sr] + 1 : top + row_lines[sr + 1],
            left + col_lines[sc] + 1 : left + col_lines[sc + 1],
        ] = int(color)


def _group_layout(
    size: int,
    frame_h: int,
    frame_w: int,
    *,
    fixed_orient: str | None = None,
    compact: bool = False,
) -> dict:
    if compact:
        gap_r = random.randint(1, 2)
        gap_c = random.randint(1, 2)
    else:
        gap_r = random.randint(2, 4)
        gap_c = random.randint(2, 4)

    if size == 1:
        orient = fixed_orient or random.choice(["H", "V"])
        clusters = [(0, 0)]
        corridors = [(orient, 0)]
    elif size == 2:
        orient = fixed_orient or random.choice(["H", "V"])
        if orient == "H":
            clusters = [(0, 0), (0, 1)]
            corridors = [("H", 0)]
        else:
            clusters = [(0, 0), (1, 0)]
            corridors = [("V", 0)]
    elif size == 3:
        orient = fixed_orient or random.choice(["H", "V"])
        if orient == "H":
            clusters = [(0, 0), (0, 1), (1, 1)]
            corridors = [("H", 0), ("V", 1)]
        else:
            clusters = [(0, 0), (1, 0), (1, 1)]
            corridors = [("V", 0), ("H", 1)]
    else:
        raise ValueError("unsupported group size")

    row_clusters = max(r for r, _ in clusters) + 1
    col_clusters = max(c for _, c in clusters) + 1
    local_positions = [
        (r * (frame_h + gap_r), c * (frame_w + gap_c))
        for r, c in clusters
    ]
    group_h = row_clusters * frame_h + (row_clusters - 1) * gap_r
    group_w = col_clusters * frame_w + (col_clusters - 1) * gap_c
    return {
        "size": size,
        "gap_r": gap_r,
        "gap_c": gap_c,
        "clusters": clusters,
        "local_positions": local_positions,
        "corridors": corridors,
        "group_h": group_h,
        "group_w": group_w,
    }


def _deranged_permutation(colors: list[int]) -> list[int]:
    perm = colors[:]
    while True:
        random.shuffle(perm)
        if all(a != b for a, b in zip(colors, perm)):
            return perm[:]


def _choose_schemes(slot_shape: tuple[int, int]) -> list[dict[tuple[int, int], int]]:
    positions = _slot_positions(slot_shape)
    palette = random.sample(FILL_COLORS, len(positions))
    first = palette[:]
    random.shuffle(first)
    second = _deranged_permutation(first)
    return [
        {positions[i]: int(first[i]) for i in range(len(positions))},
        {positions[i]: int(second[i]) for i in range(len(positions))},
    ]


def _hint_masks(size: int, slot_shape: tuple[int, int]) -> list[set[tuple[int, int]]]:
    slots = _slot_positions(slot_shape)
    if size == 1:
        return [set(slots)]

    while True:
        unique_slots = random.sample(slots, size)
        masks = [set([unique_slots[i]]) for i in range(size)]
        remaining = [slot for slot in slots if slot not in unique_slots]

        for slot in remaining:
            primary = random.randrange(size)
            masks[primary].add(slot)
            others = [idx for idx in range(size) if idx != primary]
            if others and random.random() < 0.35:
                extra = random.choice(others)
                if slot not in unique_slots:
                    masks[extra].add(slot)

        if set().union(*masks) != set(slots):
            continue
        if any(len(mask) == 0 or len(mask) == len(slots) for mask in masks):
            continue
        if not all(
            any(slot in masks[idx] and all(slot not in masks[j] for j in range(size) if j != idx) for slot in slots)
            for idx in range(size)
        ):
            continue
        return masks


def _clear_hidden_slots(
    grid: np.ndarray,
    top: int,
    left: int,
    row_lines: list[int],
    col_lines: list[int],
    visible: set[tuple[int, int]],
) -> None:
    for sr in range(len(row_lines) - 1):
        for sc in range(len(col_lines) - 1):
            if (sr, sc) in visible:
                continue
            grid[
                top + row_lines[sr] + 1 : top + row_lines[sr + 1],
                left + col_lines[sc] + 1 : left + col_lines[sc + 1],
            ] = BG


def _generate_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    n_frames = random.choices([3, 4, 5], weights=[1, 3, 3], k=1)[0]
    group_sizes = {3: [2, 1], 4: [2, 2], 5: [2, 3]}[n_frames]
    random.shuffle(group_sizes)

    if n_frames == 5:
        slot_shape = random.choices([(2, 2), (2, 3), (3, 2)], weights=[6, 2, 2], k=1)[0]
    elif n_frames == 4:
        slot_shape = random.choices([(2, 2), (2, 3), (3, 2)], weights=[4, 3, 3], k=1)[0]
    else:
        slot_shape = random.choice([(2, 2), (2, 3), (3, 2)])

    small_bias = 0.8 if n_frames == 5 else 0.65 if n_frames == 4 else 0.5
    row_sizes = [1 if random.random() < small_bias else 2 for _ in range(slot_shape[0])]
    col_sizes = [1 if random.random() < small_bias else 2 for _ in range(slot_shape[1])]
    row_lines = _line_offsets(row_sizes)
    col_lines = _line_offsets(col_sizes)
    frame_h = row_lines[-1] + 1
    frame_w = col_lines[-1] + 1

    groups = []
    schemes = _choose_schemes(slot_shape)
    compact = n_frames >= 4
    preferred_two = None
    if n_frames == 5:
        if frame_w > frame_h:
            preferred_two = "V"
        elif frame_h > frame_w:
            preferred_two = "H"
        else:
            preferred_two = random.choice(["H", "V"])
    for group_idx, size in enumerate(group_sizes):
        fixed_orient = preferred_two if n_frames == 5 and size == 2 else None
        groups.append(
            {
                "group_id": group_idx,
                "scheme": schemes[group_idx],
                "layout": _group_layout(
                    size,
                    frame_h,
                    frame_w,
                    fixed_orient=fixed_orient,
                    compact=compact,
                ),
                "hint_masks": _hint_masks(size, slot_shape),
            }
        )

    random.shuffle(groups)

    if compact:
        margin_top = random.randint(1, 2)
        margin_left = random.randint(1, 2)
        margin_bottom = random.randint(1, 2)
        margin_right = random.randint(1, 2)
        gap_groups_r = random.randint(1, 3)
        gap_groups_c = random.randint(1, 3)
    else:
        margin_top = random.randint(1, 3)
        margin_left = random.randint(1, 3)
        margin_bottom = random.randint(1, 3)
        margin_right = random.randint(1, 3)
        gap_groups_r = random.randint(2, 5)
        gap_groups_c = random.randint(2, 5)

    first = groups[0]["layout"]
    second = groups[1]["layout"]
    height = margin_top + first["group_h"] + gap_groups_r + second["group_h"] + margin_bottom
    width = margin_left + first["group_w"] + gap_groups_c + second["group_w"] + margin_right
    if height > 30 or width > 30:
        return None

    origins = [
        (margin_top, margin_left),
        (margin_top + first["group_h"] + gap_groups_r, margin_left + first["group_w"] + gap_groups_c),
    ]

    output = np.full((height, width), BG, dtype=int)
    input_grid = np.full((height, width), BG, dtype=int)

    frame_meta: list[dict] = []
    corridor_rows: set[int] = set()
    corridor_cols: set[int] = set()

    for group, (group_top, group_left) in zip(groups, origins):
        layout = group["layout"]
        h_idx = random.randrange(len(row_lines))
        v_idx = random.randrange(len(col_lines))

        for frame_idx, ((local_top, local_left), visible) in enumerate(
            zip(layout["local_positions"], group["hint_masks"])
        ):
            top = group_top + local_top
            left = group_left + local_left
            _draw_frame(output, top, left, row_lines, col_lines, group["scheme"])
            _draw_frame(input_grid, top, left, row_lines, col_lines, group["scheme"])
            _clear_hidden_slots(input_grid, top, left, row_lines, col_lines, visible)
            frame_meta.append(
                {
                    "bbox": (top, left, top + frame_h - 1, left + frame_w - 1),
                    "group_id": group["group_id"],
                    "visible": set(visible),
                    "scheme": dict(group["scheme"]),
                    "size": layout["size"],
                    "frame_idx": frame_idx,
                }
            )

        for kind, cluster_idx in layout["corridors"]:
            if kind == "H":
                corridor_rows.add(group_top + cluster_idx * (frame_h + layout["gap_r"]) + row_lines[h_idx])
            else:
                corridor_cols.add(group_left + cluster_idx * (frame_w + layout["gap_c"]) + col_lines[v_idx])

    for r in corridor_rows:
        output[r, 0] = RED
        output[r, -1] = RED
        input_grid[r, 0] = RED
        input_grid[r, -1] = RED
    for c in corridor_cols:
        output[0, c] = RED
        output[-1, c] = RED
        input_grid[0, c] = RED
        input_grid[-1, c] = RED

    frame_meta.sort(key=lambda item: item["bbox"])
    groups_expected: dict[int, list[int]] = defaultdict(list)
    for idx, item in enumerate(frame_meta):
        groups_expected[item["group_id"]].append(idx)

    meta = {
        "slot_shape": slot_shape,
        "frame_meta": frame_meta,
        "groups_expected": sorted((sorted(v) for v in groups_expected.values()), key=lambda v: v[0]),
        "corridor_rows": corridor_rows,
        "corridor_cols": corridor_cols,
    }
    return input_grid, output, meta


def _validate_generated(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.any(inp != out):
        return False

    try:
        analysis = _analyze(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, out):
        return False
    if analysis["slot_shape"] != meta["slot_shape"]:
        return False
    if analysis["corridor_rows"] != meta["corridor_rows"]:
        return False
    if analysis["corridor_cols"] != meta["corridor_cols"]:
        return False
    if analysis["groups"] != meta["groups_expected"]:
        return False

    parsed_bboxes = [frame["bbox"] for frame in analysis["frames"]]
    expected_bboxes = [frame["bbox"] for frame in meta["frame_meta"]]
    if parsed_bboxes != expected_bboxes:
        return False

    for r in meta["corridor_rows"]:
        if not _is_horizontal_corridor(inp, r):
            return False
    for c in meta["corridor_cols"]:
        if not _is_vertical_corridor(inp, c):
            return False

    all_slots = set(_slot_positions(meta["slot_shape"]))
    visible_by_group: dict[int, dict[tuple[int, int], int]] = defaultdict(dict)
    unique_needed_ok = True
    for idx, frame in enumerate(meta["frame_meta"]):
        group_id = frame["group_id"]
        visible = frame["visible"]
        scheme = frame["scheme"]

        if frame["size"] > 1:
            if not visible or visible == all_slots:
                return False
            if not any(
                slot in visible
                and all(
                    slot not in meta["frame_meta"][j]["visible"]
                    for j in meta["groups_expected"][group_id]
                    if j != idx
                )
                for slot in visible
            ):
                unique_needed_ok = False
        else:
            if visible != all_slots:
                return False

        for slot in visible:
            rr0 = analysis["frames"][idx]["rows"][slot[0]] + 1
            rr1 = analysis["frames"][idx]["rows"][slot[0] + 1]
            cc0 = analysis["frames"][idx]["cols"][slot[1]] + 1
            cc1 = analysis["frames"][idx]["cols"][slot[1] + 1]
            region = inp[rr0:rr1, cc0:cc1]
            if not np.all(region == scheme[slot]):
                return False
            visible_by_group[group_id][slot] = scheme[slot]

    if not unique_needed_ok:
        return False

    if any(set(info.keys()) != all_slots for info in visible_by_group.values()):
        return False

    merged_visible: dict[tuple[int, int], set[int]] = defaultdict(set)
    for info in visible_by_group.values():
        for slot, color in info.items():
            merged_visible[slot].add(int(color))
    if not any(len(colors) > 1 for colors in merged_visible.values()):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        made = _generate_attempt()
        if made is None:
            continue
        inp, out, meta = made
        if _validate_generated(inp, out, meta):
            return inp
    raise RuntimeError("failed to generate a valid corridor-group inference puzzle")
