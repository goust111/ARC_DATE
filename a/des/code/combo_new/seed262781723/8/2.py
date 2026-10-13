from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ARC_MAX = 30
COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _most_common_color(arr: np.ndarray) -> int:
    flat = np.asarray(arr, dtype=int).reshape(-1)
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.full((h, w), bg, dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    rel_coords: list[tuple[int, int]] = []
    for r, c in coords:
        rr = r - r0
        cc = c - c0
        patch[rr, cc] = int(grid[r, c])
        mask[rr, cc] = True
        rel_coords.append((rr, cc))
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "coords": coords,
        "rel_coords": rel_coords,
        "shape": (h, w),
    }


def _is_reference_ring(comp: dict) -> bool:
    mask = comp["mask"]
    patch = comp["patch"]
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if int(mask.sum()) != 2 * (h + w) - 4:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    if mask[1:-1, 1:-1].any():
        return False
    vals = patch[mask]
    return len(set(map(int, vals))) == 1


def _contiguous_runs(active: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for val in map(int, idx[1:]):
        if val == prev + 1:
            prev = val
            continue
        runs.append((start, prev + 1))
        start = prev = val
    runs.append((start, prev + 1))
    return runs


def _edge_interfaces(comp: dict, base: int, bg: int) -> list[dict]:
    patch = comp["patch"]
    mask = comp["mask"]
    interfaces: list[dict] = []

    def add_runs(side: str, values: np.ndarray, occ: np.ndarray) -> None:
        code_mask = occ & (values != base) & (values != bg)
        for start, end in _contiguous_runs(code_mask):
            seq = tuple(map(int, values[start:end]))
            interfaces.append(
                {
                    "side": side,
                    "start": int(start),
                    "length": int(end - start),
                    "seq": seq,
                }
            )

    add_runs("top", patch[0, :], mask[0, :])
    add_runs("bottom", patch[-1, :], mask[-1, :])
    add_runs("left", patch[:, 0], mask[:, 0])
    add_runs("right", patch[:, -1], mask[:, -1])
    return interfaces


def _pair_delta(piece_a: dict, run_a: dict, piece_b: dict, run_b: dict) -> tuple[int, int]:
    if run_a["length"] != run_b["length"]:
        raise ValueError("matched interfaces must have the same length")

    sa = run_a["side"]
    sb = run_b["side"]
    oa = int(run_a["start"])
    ob = int(run_b["start"])
    ha, wa = piece_a["shape"]
    hb, wb = piece_b["shape"]

    if sa == "right" and sb == "left":
        return oa - ob, wa
    if sa == "left" and sb == "right":
        return oa - ob, -wb
    if sa == "bottom" and sb == "top":
        return ha, oa - ob
    if sa == "top" and sb == "bottom":
        return -hb, oa - ob
    raise ValueError(f"invalid side pairing: {sa} vs {sb}")


def _recover_rectangle_shape(pieces: list[dict]) -> tuple[int, int]:
    by_seq: dict[tuple[int, ...], list[tuple[int, dict]]] = {}
    for pid, piece in enumerate(pieces):
        piece["interfaces"] = _edge_interfaces(piece, piece["base"], BG)
        if not piece["interfaces"]:
            raise ValueError("every fragment must expose at least one colored interface")
        for run in piece["interfaces"]:
            by_seq.setdefault(run["seq"], []).append((pid, run))

    graph: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in pieces]
    for seq, items in by_seq.items():
        if len(items) != 2:
            raise ValueError(f"interface {seq} is ambiguous")
        (a, run_a), (b, run_b) = items
        delta = _pair_delta(pieces[a], run_a, pieces[b], run_b)
        graph[a].append((b, delta))
        graph[b].append((a, (-delta[0], -delta[1])))

    positions: list[tuple[int, int] | None] = [None] * len(pieces)
    positions[0] = (0, 0)
    q: deque[int] = deque([0])
    while q:
        cur = q.popleft()
        cur_r, cur_c = positions[cur]
        assert cur_r is not None and cur_c is not None
        for nxt, (dr, dc) in graph[cur]:
            cand = (cur_r + dr, cur_c + dc)
            if positions[nxt] is None:
                positions[nxt] = cand
                q.append(nxt)
            elif positions[nxt] != cand:
                raise ValueError("fragment offsets are inconsistent")

    if any(pos is None for pos in positions):
        raise ValueError("fragment graph must be connected")

    occupied: set[tuple[int, int]] = set()
    for piece, pos in zip(pieces, positions):
        pr, pc = pos
        assert pr is not None and pc is not None
        for rr, cc in piece["rel_coords"]:
            cell = (pr + rr, pc + cc)
            if cell in occupied:
                raise ValueError("assembled fragments may not overlap")
            occupied.add(cell)

    min_r = min(r for r, _ in occupied)
    min_c = min(c for _, c in occupied)
    norm = {(r - min_r, c - min_c) for r, c in occupied}
    rect_h = max(r for r, _ in norm) + 1
    rect_w = max(c for _, c in norm) + 1
    if len(norm) != rect_h * rect_w:
        raise ValueError("assembled fragments must fill one complete rectangle")
    return rect_h, rect_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-color fragments into the solid rectangle indicated by the reference frame.

    Input structure:
    - The background is black.
    - Several separated fragments share one main fill color, while short multicolor runs on their cut edges act as interface codes.
    - A small hollow rectangle in the lower-right corner marks the final rectangle's size and destination.

    Rule:
    - Separate the hollow reference frame from the fragments and infer the common fragment base color.
    - Read every colored run on fragment borders; identical sequences on opposite-facing edges identify two sides that must be joined.
    - Use the matched interfaces to recover the fragments' relative translations and verify that their union forms one complete rectangle.
    - Clear the scattered fragments and replace the reference frame with a solid rectangle of the base color.

    Key insight:
    - The extra colors are not output material; they are edge-wise alignment codes whose matching order reconstructs the hidden rectangle.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    bg = _most_common_color(g)

    comps = [_extract_component(g, coords, bg) for coords in _connected_components(g != bg)]
    frame_candidates = [comp for comp in comps if _is_reference_ring(comp)]
    if len(frame_candidates) != 1:
        raise ValueError("expected exactly one reference frame")
    frame = frame_candidates[0]
    pieces = [comp for comp in comps if comp is not frame]
    if not pieces:
        raise ValueError("expected at least one fragment")

    color_counts: Counter[int] = Counter()
    for piece in pieces:
        color_counts.update(map(int, piece["patch"][piece["mask"]]))
    base = color_counts.most_common(1)[0][0]
    for piece in pieces:
        piece["base"] = int(base)

    rect_h, rect_w = _recover_rectangle_shape(pieces)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    if (rect_h, rect_w) != (fr1 - fr0 + 1, fc1 - fc0 + 1):
        raise ValueError("assembled rectangle size must match the reference frame")

    out = np.full_like(g, bg)
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(base)
    return out


LAYOUT_MODES = ("vertical", "horizontal", "grid")
GRID_LAYOUTS = ((2, 2), (2, 3), (3, 2))
SPAN_MIN = 4
SPAN_MAX = 6
STRIP_MAX_TOTAL = 20
RUN_MIN = 2
RUN_MAX = 3
CANVAS_MIN = 18


def _sample_parts(n: int, min_part: int, max_part: int, max_total: int) -> list[int] | None:
    min_total = n * min_part
    max_total = min(max_total, n * max_part)
    if min_total > max_total:
        return None
    total = random.randint(min_total, max_total)
    parts = [min_part] * n
    extra = total - min_total
    capacity = [max_part - min_part] * n
    while extra > 0:
        candidates = [i for i, room in enumerate(capacity) if room > 0]
        if not candidates:
            break
        idx = random.choice(candidates)
        parts[idx] += 1
        capacity[idx] -= 1
        extra -= 1
    random.shuffle(parts)
    return parts


def _choose_layout() -> tuple[list[int], list[int]]:
    mode = random.choice(LAYOUT_MODES)
    if mode == "vertical":
        cols = random.randint(3, 5)
        row_heights = [random.randint(SPAN_MIN, SPAN_MAX)]
        col_widths = _sample_parts(cols, SPAN_MIN, SPAN_MAX, STRIP_MAX_TOTAL)
        if col_widths is None:
            raise RuntimeError("failed to sample vertical strip widths")
        return row_heights, col_widths
    if mode == "horizontal":
        rows = random.randint(3, 5)
        row_heights = _sample_parts(rows, SPAN_MIN, SPAN_MAX, STRIP_MAX_TOTAL)
        if row_heights is None:
            raise RuntimeError("failed to sample horizontal strip heights")
        col_widths = [random.randint(SPAN_MIN, SPAN_MAX)]
        return row_heights, col_widths

    rows, cols = random.choice(GRID_LAYOUTS)
    row_heights = [random.randint(SPAN_MIN, SPAN_MAX - 1) for _ in range(rows)]
    col_widths = [random.randint(SPAN_MIN, SPAN_MAX - 1) for _ in range(cols)]
    return row_heights, col_widths


def _sample_sequence(length: int, palette: list[int], used: set[tuple[int, ...]]) -> tuple[int, ...]:
    for _ in range(200):
        seq = tuple(random.choice(palette) for _ in range(length))
        if len(set(seq)) == 1:
            continue
        if seq not in used:
            used.add(seq)
            return seq
    raise RuntimeError("failed to sample a unique interface sequence")


def _paint_vertical_run(piece: np.ndarray, col: int, start: int, seq: tuple[int, ...]) -> None:
    piece[start : start + len(seq), col] = np.asarray(seq, dtype=int)


def _paint_horizontal_run(piece: np.ndarray, row: int, start: int, seq: tuple[int, ...]) -> None:
    piece[row, start : start + len(seq)] = np.asarray(seq, dtype=int)


def _build_fragments(row_heights: list[int], col_widths: list[int], base: int, frame_color: int) -> list[np.ndarray]:
    palette = [c for c in COLORS if c not in (BG, base, frame_color)]
    used: set[tuple[int, ...]] = set()

    rows = len(row_heights)
    cols = len(col_widths)
    pieces = {
        (r, c): np.full((row_heights[r], col_widths[c]), base, dtype=int)
        for r in range(rows)
        for c in range(cols)
    }

    for r in range(rows):
        for c in range(cols - 1):
            shared = row_heights[r]
            run_len = random.randint(RUN_MIN, min(RUN_MAX, shared - 2))
            start = random.randint(1, shared - run_len - 1)
            seq = _sample_sequence(run_len, palette, used)
            _paint_vertical_run(pieces[(r, c)], -1, start, seq)
            _paint_vertical_run(pieces[(r, c + 1)], 0, start, seq)

    for r in range(rows - 1):
        for c in range(cols):
            shared = col_widths[c]
            run_len = random.randint(RUN_MIN, min(RUN_MAX, shared - 2))
            start = random.randint(1, shared - run_len - 1)
            seq = _sample_sequence(run_len, palette, used)
            _paint_horizontal_run(pieces[(r, c)], -1, start, seq)
            _paint_horizontal_run(pieces[(r + 1, c)], 0, start, seq)

    flat = [pieces[(r, c)] for r in range(rows) for c in range(cols)]
    if any(not np.any(piece == base) for piece in flat):
        raise RuntimeError("every fragment must retain visible base-color cells")
    counts = Counter()
    for piece in flat:
        counts.update(map(int, piece.reshape(-1)))
    if counts[base] <= max(v for k, v in counts.items() if k != base):
        raise RuntimeError("base color must stay globally dominant among fragment colors")
    return flat


def _can_place(occupied: np.ndarray, h: int, w: int, r0: int, c0: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patches(grid_shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    positions: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
            if _can_place(occupied, ph, pw, r0, c0)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True

    return [pos for pos in positions if pos is not None]


def _render_input(
    canvas_h: int,
    canvas_w: int,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    patches: list[np.ndarray],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color
    for patch, (r0, c0) in zip(patches, positions):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return grid


def _expected_output(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], base: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = base
    return out


def _gen_attempt() -> np.ndarray | None:
    row_heights, col_widths = _choose_layout()
    target_h = sum(row_heights)
    target_w = sum(col_widths)
    if target_h > ARC_MAX or target_w > ARC_MAX:
        return None

    base, frame_color = random.sample([c for c in COLORS if c != BG], 2)
    try:
        patches = _build_fragments(row_heights, col_widths, base, frame_color)
    except RuntimeError:
        return None

    max_piece_h = max(int(p.shape[0]) for p in patches)
    max_piece_w = max(int(p.shape[1]) for p in patches)
    min_canvas_h = max(CANVAS_MIN, target_h + max_piece_h + 2)
    min_canvas_w = max(CANVAS_MIN, target_w + max_piece_w + 2)
    if min_canvas_h > ARC_MAX or min_canvas_w > ARC_MAX:
        return None

    canvas_h = random.randint(min_canvas_h, ARC_MAX)
    canvas_w = random.randint(min_canvas_w, ARC_MAX)
    frame_bbox = (canvas_h - target_h, canvas_w - target_w, canvas_h - 1, canvas_w - 1)

    positions = _place_patches((canvas_h, canvas_w), frame_bbox, patches)
    if positions is None:
        return None

    inp = _render_input(canvas_h, canvas_w, frame_bbox, frame_color, patches, positions)
    expected = _expected_output(inp.shape, frame_bbox, base)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
