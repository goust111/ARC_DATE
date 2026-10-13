from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLUE = 1
RED = 2
PIECE_COLOR_POOL = [3, 4, 5, 6, 8]
MAX_GEN_ATTEMPTS = 400
GRID_COMBOS = [(2, 2), (2, 3), (3, 2)]
CORNERS = ["TL", "TR", "BL", "BR"]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[dict] | None:
    pieces = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = bbox_coords(coords)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        if np.any(arr == bg):
            return None
        ph, pw = arr.shape
        if not (2 <= ph <= 4 and 3 <= pw <= 5):
            return None
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "area": int(ph * pw),
            }
        )
    return pieces


def _gap_corner(local_gap: tuple[int, int]) -> str:
    mapping = {
        (0, 0): "BR",
        (0, 1): "BL",
        (1, 0): "TR",
        (1, 1): "TL",
    }
    return mapping[tuple(map(int, local_gap))]


def _compat(edge_a: np.ndarray, edge_b: np.ndarray, wildcard: int) -> bool:
    mask = (edge_a != wildcard) & (edge_b != wildcard)
    return bool(np.all(edge_a[mask] == edge_b[mask]))


def _piece_sig(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _count_assemblies(
    pieces: list[np.ndarray],
    wildcard: int,
    corner: str,
    anchor_idx: int,
    limit: int = 2,
) -> tuple[int, np.ndarray | None]:
    total_area = int(sum(int(p.shape[0] * p.shape[1]) for p in pieces))
    anchor = pieces[anchor_idx]
    ah, aw = map(int, anchor.shape)

    dims = []
    for h in range(1, int(total_area**0.5) + 1):
        if total_area % h == 0:
            dims.append((h, total_area // h))
            if h * h != total_area:
                dims.append((total_area // h, h))
    dims = sorted({(int(h), int(w)) for h, w in dims}, key=lambda t: (t[0] * t[1], t[0], t[1]))

    order = sorted(range(len(pieces)), key=lambda i: (-pieces[i].shape[0] * pieces[i].shape[1], -pieces[i].shape[0], -pieces[i].shape[1], i))
    total_count = 0
    first_solution: np.ndarray | None = None

    def anchor_pos(hh: int, ww: int) -> tuple[int, int] | None:
        if hh < ah or ww < aw:
            return None
        if corner == "TL":
            return 0, 0
        if corner == "TR":
            return 0, ww - aw
        if corner == "BL":
            return hh - ah, 0
        return hh - ah, ww - aw

    for hh, ww in dims:
        pos = anchor_pos(int(hh), int(ww))
        if pos is None:
            continue
        canvas = np.full((hh, ww), -1, dtype=int)
        used = [False] * len(pieces)

        def can_place(idx: int, r0: int, c0: int) -> bool:
            arr = pieces[idx]
            ph, pw = map(int, arr.shape)
            if r0 + ph > hh or c0 + pw > ww:
                return False
            if np.any(canvas[r0 : r0 + ph, c0 : c0 + pw] != -1):
                return False
            if c0 > 0:
                left = canvas[r0 : r0 + ph, c0 - 1]
                mask = left != -1
                if mask.any() and not _compat(left[mask], arr[:, 0][mask], wildcard):
                    return False
            if c0 + pw < ww:
                right = canvas[r0 : r0 + ph, c0 + pw]
                mask = right != -1
                if mask.any() and not _compat(right[mask], arr[:, -1][mask], wildcard):
                    return False
            if r0 > 0:
                up = canvas[r0 - 1, c0 : c0 + pw]
                mask = up != -1
                if mask.any() and not _compat(up[mask], arr[0, :][mask], wildcard):
                    return False
            if r0 + ph < hh:
                down = canvas[r0 + ph, c0 : c0 + pw]
                mask = down != -1
                if mask.any() and not _compat(down[mask], arr[-1, :][mask], wildcard):
                    return False
            return True

        def place(idx: int, r0: int, c0: int) -> None:
            arr = pieces[idx]
            ph, pw = map(int, arr.shape)
            canvas[r0 : r0 + ph, c0 : c0 + pw] = arr
            used[idx] = True

        def unplace(idx: int, r0: int, c0: int) -> None:
            arr = pieces[idx]
            ph, pw = map(int, arr.shape)
            canvas[r0 : r0 + ph, c0 : c0 + pw] = -1
            used[idx] = False

        ar0, ac0 = pos
        if not can_place(anchor_idx, ar0, ac0):
            continue
        place(anchor_idx, ar0, ac0)

        def next_empty() -> tuple[int, int] | None:
            coords = np.argwhere(canvas == -1)
            if coords.size == 0:
                return None
            return int(coords[0, 0]), int(coords[0, 1])

        def bt() -> None:
            nonlocal total_count, first_solution
            if total_count >= limit:
                return
            nxt = next_empty()
            if nxt is None:
                total_count += 1
                if first_solution is None:
                    first_solution = canvas.copy()
                return
            r0, c0 = nxt
            tried: set[tuple[tuple[int, int], bytes]] = set()
            for idx in order:
                if used[idx]:
                    continue
                sig = _piece_sig(pieces[idx])
                if sig in tried:
                    continue
                tried.add(sig)
                if can_place(idx, r0, c0):
                    place(idx, r0, c0)
                    bt()
                    unplace(idx, r0, c0)
                    if total_count >= limit:
                        return

        bt()

    return total_count, first_solution


def _analyze_candidate(
    grid: np.ndarray,
    bg: int,
    marker_cells: list[tuple[int, int]],
    gap: tuple[int, int],
    corner: str,
    marker_color_hint: int,
) -> dict | None:
    work = grid.copy()
    for r, c in marker_cells:
        work[r, c] = bg

    pieces = _extract_rectangles(work, bg)
    if pieces is None or not (6 <= len(pieces) <= 8):
        return None

    anchor_piece_idx = None
    for idx, piece in enumerate(pieces):
        r0, c0, r1, c1 = piece["bbox"]
        if r0 <= gap[0] <= r1 and c0 <= gap[1] <= c1:
            local = (gap[0] - r0, gap[1] - c0)
            if local in ((0, 0), (0, c1 - c0), (r1 - r0, 0), (r1 - r0, c1 - c0)):
                anchor_piece_idx = idx
            break
    if anchor_piece_idx is None:
        return None

    r0, c0, r1, c1 = pieces[anchor_piece_idx]["bbox"]
    anchor_local_corner = {
        (0, 0): "TL",
        (0, c1 - c0): "TR",
        (r1 - r0, 0): "BL",
        (r1 - r0, c1 - c0): "BR",
    }.get((gap[0] - r0, gap[1] - c0))
    if anchor_local_corner != corner:
        return None

    color_counts = Counter()
    for piece in pieces:
        color_counts.update(map(int, piece["arr"].flatten()))
    color_counts.pop(int(bg), None)
    if not color_counts:
        return None
    min_count = min(color_counts.values())
    marker_colors = [col for col, count in color_counts.items() if count == min_count]
    if len(marker_colors) != 1:
        return None
    marker_color = int(marker_colors[0])

    piece_marker_counts = [int(np.sum(piece["arr"] == marker_color)) for piece in pieces]
    decoys = [i for i, count in enumerate(piece_marker_counts) if count > 1]
    survivors = [i for i, count in enumerate(piece_marker_counts) if count <= 1]
    if not (2 <= len(decoys) <= 3 and 4 <= len(survivors) <= 6):
        return None
    if sum(piece_marker_counts[i] == 1 for i in survivors) < 1:
        return None
    if anchor_piece_idx not in survivors:
        return None

    survivor_counts = Counter()
    for idx in survivors:
        arr = pieces[idx]["arr"]
        survivor_counts.update(int(v) for v in arr.flatten() if int(v) not in (bg, marker_color))
    if not survivor_counts:
        return None
    max_count = max(survivor_counts.values())
    wildcard_colors = [col for col, count in survivor_counts.items() if count == max_count]
    if len(wildcard_colors) != 1:
        return None
    wildcard = int(wildcard_colors[0])
    if wildcard != int(marker_color_hint):
        return None

    filtered_pieces = []
    anchor_survivor_idx = None
    for old_idx in survivors:
        arr = pieces[old_idx]["arr"].copy()
        if piece_marker_counts[old_idx] == 1:
            arr[arr == marker_color] = wildcard
        filtered_pieces.append(arr)
        if old_idx == anchor_piece_idx:
            anchor_survivor_idx = len(filtered_pieces) - 1
    assert anchor_survivor_idx is not None

    count, solution = _count_assemblies(filtered_pieces, wildcard, corner, anchor_survivor_idx, limit=2)
    if count != 1 or solution is None:
        return None

    return {
        "marker_color": marker_color,
        "wildcard": wildcard,
        "piece_marker_counts": piece_marker_counts,
        "pieces": pieces,
        "survivors": survivors,
        "decoys": decoys,
        "solution": solution,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter out over-marked rectangular fragments, then assemble the survivors into one rectangle.

    Input structure:
    - A blue background contains 6-8 separated filled rectangular fragments, each 2x3 to 4x5.
    - Fragment pixels use 2-3 visible piece colors, and a few internal pixels use one rarer marker color.
    - One corner fragment touches a 3-pixel L marker whose color matches the eventual wildcard seam color.

    Rule:
    - Find the unique 2x2 L marker candidate: three same-colored cells around one non-background corner cell.
    - Remove those three marker cells, split the remaining foreground into filled rectangles, and identify the rarest piece color as the internal marker color.
    - Erase every fragment containing more than one such marker pixel; on each surviving fragment, replace a lone marker pixel by the most frequent surviving non-marker color.
    - Translate the surviving rectangles without rotation so they tile one filled rectangle; touching edge colors must agree, except that the dominant surviving color is a wildcard.
    - Return only the assembled rectangle, cropped to its bounding box.

    Key insight:
    - The tiny L is not a separate object color cue; it must be peeled off first so the touching corner fragment becomes a normal rectangle, after which marker counting filters decoys and lone markers become wildcard seam cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            block = g[r : r + 2, c : c + 2]
            colors = Counter(map(int, block.flatten()))
            for marker_color_hint, count in colors.items():
                if marker_color_hint == bg or count != 3:
                    continue
                gap_local = [(rr, cc) for rr in range(2) for cc in range(2) if int(block[rr, cc]) != int(marker_color_hint)]
                if len(gap_local) != 1:
                    continue
                gr, gc = gap_local[0]
                gap = (r + gr, c + gc)
                if int(g[gap]) == int(bg):
                    continue
                marker_cells = [(r + rr, c + cc) for rr in range(2) for cc in range(2) if (rr, cc) != (gr, gc)]
                analysis = _analyze_candidate(g, bg, marker_cells, gap, _gap_corner((gr, gc)), marker_color_hint)
                if analysis is None:
                    continue
                solution = analysis["solution"]
                outputs[_piece_sig(solution)] = solution

    assert len(outputs) == 1, "expected exactly one valid marker/filter/assembly interpretation"
    return next(iter(outputs.values()))


def _piece_bboxes(row_segs: tuple[int, ...], col_segs: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    bboxes = []
    r0 = 0
    for rh in row_segs:
        c0 = 0
        for cw in col_segs:
            bboxes.append((r0, c0, r0 + rh - 1, c0 + cw - 1))
            c0 += cw
        r0 += rh
    return bboxes


def _piece_index(row_idx: int, col_idx: int, col_count: int) -> int:
    return row_idx * col_count + col_idx


def _seams_compatible(rect: np.ndarray, row_segs: tuple[int, ...], col_segs: tuple[int, ...], wildcard: int) -> bool:
    row_starts = np.cumsum((0,) + row_segs[:-1])
    col_starts = np.cumsum((0,) + col_segs[:-1])
    for ri, rh in enumerate(row_segs):
        for cj in range(len(col_segs) - 1):
            c = int(col_starts[cj] + col_segs[cj] - 1)
            edge_a = rect[int(row_starts[ri]) : int(row_starts[ri] + rh), c]
            edge_b = rect[int(row_starts[ri]) : int(row_starts[ri] + rh), c + 1]
            if not _compat(edge_a, edge_b, wildcard):
                return False
    for ri in range(len(row_segs) - 1):
        r = int(row_starts[ri] + row_segs[ri] - 1)
        for cj, cw in enumerate(col_segs):
            edge_a = rect[r, int(col_starts[cj]) : int(col_starts[cj] + cw)]
            edge_b = rect[r + 1, int(col_starts[cj]) : int(col_starts[cj] + cw)]
            if not _compat(edge_a, edge_b, wildcard):
                return False
    return True


def _choose_flex_cells(
    row_segs: tuple[int, ...],
    col_segs: tuple[int, ...],
    corner_piece_idx: int,
) -> list[tuple[int, int, int, int]]:
    row_starts = np.cumsum((0,) + row_segs[:-1])
    col_starts = np.cumsum((0,) + col_segs[:-1])
    choices = []
    row_count = len(row_segs)
    col_count = len(col_segs)
    for ri, rh in enumerate(row_segs):
        for cj, cw in enumerate(col_segs):
            idx = _piece_index(ri, cj, col_count)
            if idx == corner_piece_idx:
                continue
            local_choices = []
            if cj + 1 < col_count:
                seam_c = int(col_starts[cj] + cw - 1)
                rows = list(range(int(row_starts[ri]), int(row_starts[ri] + rh)))
                mid_rows = rows[1:-1] if len(rows) > 2 else rows
                if mid_rows:
                    rr = random.choice(mid_rows)
                    local_choices.append((rr, seam_c, rr, seam_c + 1))
            if cj > 0:
                seam_c = int(col_starts[cj])
                rows = list(range(int(row_starts[ri]), int(row_starts[ri] + rh)))
                mid_rows = rows[1:-1] if len(rows) > 2 else rows
                if mid_rows:
                    rr = random.choice(mid_rows)
                    local_choices.append((rr, seam_c, rr, seam_c - 1))
            if ri + 1 < row_count:
                seam_r = int(row_starts[ri] + rh - 1)
                cols = list(range(int(col_starts[cj]), int(col_starts[cj] + cw)))
                mid_cols = cols[1:-1] if len(cols) > 2 else cols
                if mid_cols:
                    cc = random.choice(mid_cols)
                    local_choices.append((seam_r, cc, seam_r + 1, cc))
            if ri > 0:
                seam_r = int(row_starts[ri])
                cols = list(range(int(col_starts[cj]), int(col_starts[cj] + cw)))
                mid_cols = cols[1:-1] if len(cols) > 2 else cols
                if mid_cols:
                    cc = random.choice(mid_cols)
                    local_choices.append((seam_r, cc, seam_r - 1, cc))
            if local_choices:
                choices.append(random.choice(local_choices))
    random.shuffle(choices)
    return choices


def _blocked_place_candidates(blocked: np.ndarray, ph: int, pw: int, border: int = 1) -> list[tuple[int, int]]:
    h, w = blocked.shape
    out = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            if not blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                out.append((r0, c0))
    return out


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, ph: int, pw: int, padding: int = 1) -> None:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + ph + padding)
    cc1 = min(blocked.shape[1], c0 + pw + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _make_target(
    row_segs: tuple[int, ...],
    col_segs: tuple[int, ...],
    wildcard: int,
    extras: list[int],
    corner: str,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[int, tuple[int, int]]] | None:
    row_count = len(row_segs)
    col_count = len(col_segs)
    piece_boxes = _piece_bboxes(row_segs, col_segs)
    corner_piece_idx = {
        "TL": _piece_index(0, 0, col_count),
        "TR": _piece_index(0, col_count - 1, col_count),
        "BL": _piece_index(row_count - 1, 0, col_count),
        "BR": _piece_index(row_count - 1, col_count - 1, col_count),
    }[corner]

    palette = [wildcard] + extras
    probs = [0.62, 0.23, 0.15] if len(extras) == 2 else [0.7, 0.3]

    for _ in range(200):
        rect = np.array(random.choices(palette, weights=probs, k=int(sum(row_segs) * sum(col_segs))), dtype=int).reshape(sum(row_segs), sum(col_segs))
        anchor_box = piece_boxes[corner_piece_idx]
        if corner == "TL":
            anchor_corner = (anchor_box[0], anchor_box[1])
        elif corner == "TR":
            anchor_corner = (anchor_box[0], anchor_box[3])
        elif corner == "BL":
            anchor_corner = (anchor_box[2], anchor_box[1])
        else:
            anchor_corner = (anchor_box[2], anchor_box[3])
        rect[anchor_corner] = random.choice(extras)

        if not _seams_compatible(rect, row_segs, col_segs, wildcard):
            continue

        flex_choices = _choose_flex_cells(row_segs, col_segs, corner_piece_idx)
        if len(flex_choices) < 1:
            continue
        max_single = 2 if len(piece_boxes) >= 6 else 1
        wanted_single = random.randint(1, max_single)
        single_red_cells: dict[int, tuple[int, int]] = {}
        for r0, c0, r1, c1 in flex_choices:
            piece_idx = None
            for idx, (pr0, pc0, pr1, pc1) in enumerate(piece_boxes):
                if pr0 <= r0 <= pr1 and pc0 <= c0 <= pc1:
                    piece_idx = idx
                    break
            if piece_idx is None or piece_idx in single_red_cells:
                continue
            if (r0, c0) == anchor_corner:
                continue
            rect[r0, c0] = wildcard
            rect[r1, c1] = random.choice(extras)
            single_red_cells[piece_idx] = (r0, c0)
            if len(single_red_cells) == wanted_single:
                break
        if len(single_red_cells) != wanted_single:
            continue

        for pr0, pc0, pr1, pc1 in piece_boxes:
            arr = rect[pr0 : pr1 + 1, pc0 : pc1 + 1]
            if len(set(map(int, arr.flatten()))) < 2:
                rr = random.randint(pr0, pr1)
                cc = random.randint(pc0, pc1)
                if (rr, cc) == anchor_corner:
                    continue
                rect[rr, cc] = random.choice(extras)
        if not _seams_compatible(rect, row_segs, col_segs, wildcard):
            continue

        counts = Counter(map(int, rect.flatten()))
        if counts[wildcard] <= max(counts[col] for col in extras):
            continue
        return rect, piece_boxes, single_red_cells
    return None


def _make_decoy(wildcard: int, extras: list[int]) -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(3, 5)
    palette = [wildcard] + extras
    probs = [0.65, 0.2, 0.15] if len(extras) == 2 else [0.72, 0.28]
    arr = np.array(random.choices(palette, weights=probs, k=h * w), dtype=int).reshape(h, w)
    red_count = random.randint(2, 3)
    cells = random.sample([(r, c) for r in range(h) for c in range(w)], red_count)
    for r, c in cells:
        arr[r, c] = RED
    if len(set(map(int, arr.flatten()))) == 1:
        rr, cc = random.choice([(r, c) for r in range(h) for c in range(w) if (r, c) not in cells])
        arr[rr, cc] = random.choice(palette)
    return arr


def _anchor_gap_and_topleft(
    grid_shape: tuple[int, int],
    piece_shape: tuple[int, int],
    corner: str,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    gh, gw = grid_shape
    ph, pw = piece_shape
    if corner == "TL":
        gap_r = random.randint(1, 3)
        gap_c = random.randint(1, 3)
        top_left = (gap_r, gap_c)
        marker = [(gap_r - 1, gap_c - 1), (gap_r - 1, gap_c), (gap_r, gap_c - 1)]
    elif corner == "TR":
        gap_r = random.randint(1, 3)
        gap_c = random.randint(gw - 4, gw - 2)
        top_left = (gap_r, gap_c - pw + 1)
        marker = [(gap_r - 1, gap_c), (gap_r - 1, gap_c + 1), (gap_r, gap_c + 1)]
    elif corner == "BL":
        gap_r = random.randint(gh - 4, gh - 2)
        gap_c = random.randint(1, 3)
        top_left = (gap_r - ph + 1, gap_c)
        marker = [(gap_r, gap_c - 1), (gap_r + 1, gap_c - 1), (gap_r + 1, gap_c)]
    else:
        gap_r = random.randint(gh - 4, gh - 2)
        gap_c = random.randint(gw - 4, gw - 2)
        top_left = (gap_r - ph + 1, gap_c - pw + 1)
        marker = [(gap_r, gap_c + 1), (gap_r + 1, gap_c), (gap_r + 1, gap_c + 1)]
    r0, c0 = top_left
    if r0 < 0 or c0 < 0 or r0 + ph > gh or c0 + pw > gw:
        return None
    if any(not (0 <= r < gh and 0 <= c < gw) for r, c in marker):
        return None
    return (gap_r, gap_c), top_left, marker


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    row_count, col_count = random.choice(GRID_COMBOS)
    row_segs = tuple(random.randint(2, 4) for _ in range(row_count))
    col_segs = tuple(random.randint(3, 5) for _ in range(col_count))
    survivor_count = row_count * col_count
    corner = random.choice(CORNERS)

    wildcard = random.choice(PIECE_COLOR_POOL)
    extra_count = random.choice([1, 2])
    extras = random.sample([c for c in PIECE_COLOR_POOL if c != wildcard], extra_count)

    made = _make_target(row_segs, col_segs, wildcard, extras, corner)
    if made is None:
        return None
    target_rect, piece_boxes, single_red_cells = made

    survivor_raw: list[np.ndarray] = []
    anchor_piece_idx = {
        "TL": 0,
        "TR": col_count - 1,
        "BL": (row_count - 1) * col_count,
        "BR": row_count * col_count - 1,
    }[corner]
    for idx, (r0, c0, r1, c1) in enumerate(piece_boxes):
        arr = target_rect[r0 : r1 + 1, c0 : c1 + 1].copy()
        if idx in single_red_cells:
            gr, gc = single_red_cells[idx]
            rr = gr - r0
            cc = gc - c0
            if idx == anchor_piece_idx and ((corner == "TL" and (rr, cc) == (0, 0)) or (corner == "TR" and (rr, cc) == (0, arr.shape[1] - 1)) or (corner == "BL" and (rr, cc) == (arr.shape[0] - 1, 0)) or (corner == "BR" and (rr, cc) == (arr.shape[0] - 1, arr.shape[1] - 1))):
                return None
            arr[rr, cc] = RED
        survivor_raw.append(arr)

    decoy_count = 2 if survivor_count == 6 else random.choice([2, 3])
    decoys = [_make_decoy(wildcard, extras) for _ in range(decoy_count)]
    all_pieces = survivor_raw + decoys

    max_piece_h = max(arr.shape[0] for arr in all_pieces)
    max_piece_w = max(arr.shape[1] for arr in all_pieces)
    gh = random.randint(max(18, max_piece_h + 8), 24)
    gw = random.randint(max(18, max_piece_w + 8), 24)
    if gh > 30 or gw > 30:
        return None

    grid = np.full((gh, gw), BLUE, dtype=int)
    blocked = np.zeros((gh, gw), dtype=bool)

    anchor_info = _anchor_gap_and_topleft((gh, gw), survivor_raw[anchor_piece_idx].shape, corner)
    if anchor_info is None:
        return None
    gap, anchor_tl, marker_cells = anchor_info
    ar0, ac0 = anchor_tl
    anchor_piece = survivor_raw[anchor_piece_idx]
    aph, apw = anchor_piece.shape
    grid[ar0 : ar0 + aph, ac0 : ac0 + apw] = anchor_piece
    for r, c in marker_cells:
        grid[r, c] = wildcard
    union_r0 = min([ar0] + [r for r, _ in marker_cells])
    union_c0 = min([ac0] + [c for _, c in marker_cells])
    union_r1 = max([ar0 + aph - 1] + [r for r, _ in marker_cells])
    union_c1 = max([ac0 + apw - 1] + [c for _, c in marker_cells])
    _mark_blocked(blocked, union_r0, union_c0, union_r1 - union_r0 + 1, union_c1 - union_c0 + 1, padding=1)

    placement_order = [i for i in range(len(all_pieces)) if i != anchor_piece_idx]
    random.shuffle(placement_order)
    placement_order.sort(key=lambda i: -(all_pieces[i].shape[0] * all_pieces[i].shape[1]))
    for idx in placement_order:
        arr = all_pieces[idx]
        ph, pw = arr.shape
        candidates = _blocked_place_candidates(blocked, ph, pw, border=1)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw] = arr
        _mark_blocked(blocked, r0, c0, ph, pw, padding=1)

    expected = target_rect.copy()
    replaced_survivors = [arr.copy() for arr in survivor_raw]
    for arr in replaced_survivors:
        if np.sum(arr == RED) == 1:
            arr[arr == RED] = wildcard

    anchor_idx_after_filter = anchor_piece_idx
    ok_count, solved = _count_assemblies(replaced_survivors, wildcard, corner, anchor_idx_after_filter, limit=2)
    if ok_count != 1 or solved is None or not np.array_equal(solved, expected):
        return None

    raw_count, _ = _count_assemblies(survivor_raw, wildcard, corner, anchor_idx_after_filter, limit=1)
    if raw_count != 0:
        return None

    wrong_pieces = replaced_survivors + decoys
    wrong_count, _ = _count_assemblies(wrong_pieces, wildcard, corner, anchor_idx_after_filter, limit=1)
    if wrong_count != 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("generate_input: failed after many attempts")
