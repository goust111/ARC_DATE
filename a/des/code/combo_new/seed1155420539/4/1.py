from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPE_BANK = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SLOTS = [(r, c) for r in range(3) for c in range(3)]
MAX_LAYOUT_ATTEMPTS = 300
MAX_COLOR_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = np.asarray(mask, dtype=bool)
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        for var in (rot, np.fliplr(rot)):
            key = (tuple(map(int, var.shape)), var.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(int(a[0]), int(c)) for c in range(int(a[1]), int(b[1]) + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(int(r), int(a[1])) for r in range(int(a[0]), int(b[0]) + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _path_cells(
    center_a: tuple[int, int],
    center_b: tuple[int, int],
    order: str,
) -> list[tuple[int, int]]:
    start, end = sorted((tuple(map(int, center_a)), tuple(map(int, center_b))))
    if order == "hv":
        elbow = (int(start[0]), int(end[1]))
    else:
        elbow = (int(end[0]), int(start[1]))
    first = _segment(start, elbow)
    second = _segment(elbow, end)
    return first + second[1:]


def _is_frame_component(coords: np.ndarray, color: int, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    comp = {tuple(map(int, rc)) for rc in coords.tolist()}
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if comp != expected:
        return False
    return bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)).sum() == len(coords))


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    frames: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _component_coords(g == int(color))
        for coords in comps:
            if not _is_frame_component(coords, int(color), g):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            shape_mask = (inner != int(bg)) & (inner != int(color))
            shape_comps = _component_coords(shape_mask)
            assert len(shape_comps) == 1, "each frame must contain exactly one inner shape"
            rel_coords = shape_comps[0]
            shape_colors = set(map(int, inner[rel_coords[:, 0], rel_coords[:, 1]].tolist()))
            assert len(shape_colors) == 1, "each inner shape must be monochrome"
            shape_color = int(next(iter(shape_colors)))
            mask = _tight_mask(rel_coords)
            frames.append(
                {
                    "frame_color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border": {tuple(map(int, rc)) for rc in coords.tolist()},
                    "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
                    "shape_color": int(shape_color),
                    "shape_area": int(mask.sum()),
                    "shape_key": _canonical_shape(mask),
                }
            )
    assert frames, "expected at least one framed shape"
    frame_colors = {frame["frame_color"] for frame in frames}
    assert len(frame_colors) == 1, "all frames should use a single border color"
    return frames


def _path_hits_other_frames(path: list[tuple[int, int]], pair_indices: set[int], frames: list[dict]) -> bool:
    blocked: set[tuple[int, int]] = set()
    for idx, frame in enumerate(frames):
        if idx not in pair_indices:
            blocked |= frame["border"]
    return any(cell in blocked for cell in path)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-shape framed sprites and color the bridge by the rarer inner color.

    Input structure:
    - The background is black.
    - Several single-pixel-thick grey rectangular frames are scattered in the grid.
    - Each frame encloses exactly one monochrome 4-connected shape whose color is neither black nor grey.
    - Every shape class appears in exactly two frames when rotations and mirror flips are treated as equivalent.

    Rule:
    - Extract the tight binary mask of each inner shape and pair frames by rotation/flip-invariant shape.
    - For each pair, compare the total input-pixel counts of the two inner colors; use the less frequent color as the bridge color.
    - Use the lexicographically earlier frame center as the default start, draw an L path that goes horizontal then vertical to the other center, and switch to vertical then horizontal if that L would cross another frame's grey border.
    - Keep the original frames and inner shapes unchanged.

    Key insight:
    The task depends on two independent deductions: shape matching must ignore rotation and reflection, and bridge color comes from whole-grid color frequency rather than from position or frame order.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    counts = Counter(map(int, g.ravel()))
    frames = _find_frames(g)

    by_shape: dict[tuple[tuple[int, int], bytes], list[int]] = defaultdict(list)
    for idx, frame in enumerate(frames):
        by_shape[frame["shape_key"]].append(idx)

    out = g.copy()
    for pair in by_shape.values():
        assert len(pair) == 2, "each canonical shape must appear exactly twice"
        a_idx, b_idx = pair
        fa = frames[a_idx]
        fb = frames[b_idx]
        ca = int(fa["shape_color"])
        cb = int(fb["shape_color"])
        assert ca != cb, "paired inner colors must differ"
        assert counts[ca] != counts[cb], "paired colors must have different total counts"
        line_color = int(ca if counts[ca] < counts[cb] else cb)

        hv_path = _path_cells(fa["center"], fb["center"], "hv")
        if _path_hits_other_frames(hv_path, {a_idx, b_idx}, frames):
            path = _path_cells(fa["center"], fb["center"], "vh")
            assert not _path_hits_other_frames(path, {a_idx, b_idx}, frames)
        else:
            path = hv_path

        for r, c in path:
            if g[int(r), int(c)] == int(bg):
                out[int(r), int(c)] = int(line_color)

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.choice([False, True]):
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _path_blocked_on_slots(pair: tuple[tuple[int, int], tuple[int, int]], occupied: set[tuple[int, int]], order: str) -> bool:
    start, end = sorted(pair)
    for cell in occupied:
        if cell == start or cell == end:
            continue
        rr, cc = cell
        if order == "hv":
            on_first = rr == start[0] and min(start[1], end[1]) <= cc <= max(start[1], end[1])
            on_second = cc == end[1] and min(start[0], end[0]) <= rr <= max(start[0], end[0])
        else:
            on_first = cc == start[1] and min(start[0], end[0]) <= rr <= max(start[0], end[0])
            on_second = rr == end[0] and min(start[1], end[1]) <= cc <= max(start[1], end[1])
        if on_first or on_second:
            return True
    return False


def _random_pairing(slots: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    if not slots:
        return []
    first = slots[0]
    candidates = [cell for cell in slots[1:] if cell[0] != first[0] and cell[1] != first[1]]
    random.shuffle(candidates)
    for second in candidates:
        rest = [cell for cell in slots[1:] if cell != second]
        sub = _random_pairing(rest)
        if sub is not None:
            return [(first, second)] + sub
    return None


def _choose_layout(n_pairs: int, require_v_branch: bool) -> list[dict] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        chosen = sorted(random.sample(SLOTS, 2 * n_pairs))
        pairing = _random_pairing(chosen)
        if pairing is None:
            continue
        occupied = set(chosen)
        infos: list[dict] = []
        has_h = False
        has_v = False
        ok = True
        for pair in pairing:
            hv_blocked = _path_blocked_on_slots(pair, occupied, "hv")
            vh_blocked = _path_blocked_on_slots(pair, occupied, "vh")
            if hv_blocked and vh_blocked:
                ok = False
                break
            branch = "vh" if hv_blocked else "hv"
            has_h |= branch == "hv"
            has_v |= branch == "vh"
            infos.append(
                {
                    "slots": tuple(sorted(pair)),
                    "branch": branch,
                    "hv_blocked": hv_blocked,
                    "vh_blocked": vh_blocked,
                }
            )
        if not ok:
            continue
        if require_v_branch and not has_v:
            continue
        if not require_v_branch and not has_h:
            continue
        return infos
    return None


def _make_lattice(frame_size: int) -> tuple[list[int], list[int], int, int]:
    radius = frame_size // 2
    row_gap_1 = random.randint(frame_size + 1, min(frame_size + 3, 10))
    row_gap_2 = random.randint(frame_size + 1, min(frame_size + 3, 10))
    col_gap_1 = random.randint(frame_size + 1, min(frame_size + 3, 10))
    col_gap_2 = random.randint(frame_size + 1, min(frame_size + 3, 10))
    rows = [radius + 1, radius + 1 + row_gap_1, radius + 1 + row_gap_1 + row_gap_2]
    cols = [radius + 1, radius + 1 + col_gap_1, radius + 1 + col_gap_1 + col_gap_2]
    height = rows[-1] + radius + 1
    width = cols[-1] + radius + 1
    assert height <= 30 and width <= 30
    return rows, cols, int(height), int(width)


def _paint_frame(grid: np.ndarray, center: tuple[int, int], frame_size: int) -> tuple[int, int, int, int]:
    radius = frame_size // 2
    cr, cc = center
    r0 = cr - radius
    c0 = cc - radius
    r1 = cr + radius
    c1 = cc + radius
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    return int(r0), int(c0), int(r1), int(c1)


def _place_shape(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    mh, mw = mask.shape
    off_r = random.randint(0, inner_h - mh)
    off_c = random.randint(0, inner_w - mw)
    target = grid[r0 + 1 + off_r : r0 + 1 + off_r + mh, c0 + 1 + off_c : c0 + 1 + off_c + mw]
    target[mask] = int(color)


def _assign_colors(frame_records: list[dict]) -> bool:
    palette_size = random.randint(3, min(5, len(GEN_COLORS)))
    palette = random.sample(GEN_COLORS, palette_size)
    for _ in range(MAX_COLOR_ATTEMPTS):
        for pair_id in sorted({record["pair_id"] for record in frame_records}):
            pair_frames = [record for record in frame_records if record["pair_id"] == pair_id]
            c1, c2 = random.sample(palette, 2)
            pair_frames[0]["shape_color"] = int(c1)
            pair_frames[1]["shape_color"] = int(c2)
        counts: Counter[int] = Counter()
        for record in frame_records:
            counts[int(record["shape_color"])] += int(record["shape_area"])
        if all(
            counts[int(pair[0]["shape_color"])] != counts[int(pair[1]["shape_color"])]
            for pair in (
                [record for record in frame_records if record["pair_id"] == pair_id]
                for pair_id in sorted({record["pair_id"] for record in frame_records})
            )
        ):
            return True
    return False


def _expected_output(grid: np.ndarray) -> np.ndarray | None:
    try:
        return main(grid)
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        frame_size = random.choice([5, 7, 9])
        rows, cols, height, width = _make_lattice(frame_size)
        n_pairs = random.choice([2, 3])
        require_v_branch = random.choice([False, True])
        layout = _choose_layout(n_pairs, require_v_branch)
        if layout is None:
            continue

        shape_templates = random.sample(SHAPE_BANK, n_pairs)
        frame_records: list[dict] = []
        for pair_id, pair_info in enumerate(layout):
            template = np.asarray(shape_templates[pair_id], dtype=bool)
            for slot in pair_info["slots"]:
                sr, sc = slot
                frame_records.append(
                    {
                        "pair_id": int(pair_id),
                        "center": (int(rows[sr]), int(cols[sc])),
                        "frame_size": int(frame_size),
                        "shape_mask": _random_transform(template),
                        "shape_area": int(template.sum()),
                    }
                )
        if not _assign_colors(frame_records):
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        for record in frame_records:
            bbox = _paint_frame(grid, record["center"], frame_size)
            record["bbox"] = bbox
            _place_shape(grid, bbox, record["shape_mask"], int(record["shape_color"]))

        expected = _expected_output(grid)
        if expected is None:
            continue

        found_frames = _find_frames(grid)
        if len(found_frames) != len(frame_records):
            continue

        used_bg_path_cells: set[tuple[int, int]] = set()
        by_shape = defaultdict(list)
        for idx, frame in enumerate(found_frames):
            by_shape[frame["shape_key"]].append(idx)
        if len(by_shape) != n_pairs or any(len(v) != 2 for v in by_shape.values()):
            continue

        valid = True
        branch_kinds: set[str] = set()
        for pair in by_shape.values():
            fa = found_frames[pair[0]]
            fb = found_frames[pair[1]]
            hv = _path_cells(fa["center"], fb["center"], "hv")
            hv_bad = _path_hits_other_frames(hv, set(pair), found_frames)
            path = hv
            branch = "hv"
            if hv_bad:
                path = _path_cells(fa["center"], fb["center"], "vh")
                if _path_hits_other_frames(path, set(pair), found_frames):
                    valid = False
                    break
                branch = "vh"
            branch_kinds.add(branch)
            bg_cells = {cell for cell in path if grid[cell] == BLACK}
            if used_bg_path_cells & bg_cells:
                valid = False
                break
            used_bg_path_cells |= bg_cells
        if not valid:
            continue
        if require_v_branch and "vh" not in branch_kinds:
            continue
        if not require_v_branch and "hv" not in branch_kinds:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
