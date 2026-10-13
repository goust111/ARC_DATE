import random
from collections import deque

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    pts = np.argwhere(grid == int(color))
    for r, c in pts:
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.array(coords, dtype=int)
        comps.append(
            {
                "coords": arr,
                "bbox": _bbox_from_coords(arr),
                "area": int(arr.shape[0]),
            }
        )
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, int, int, int] | None = None
    best_color = -1
    best_bbox = (-1, -1, -1, -1)

    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        pts = np.argwhere(grid == int(color))
        if pts.shape[0] < 12:
            continue

        bbox = _bbox_from_coords(pts)
        r0, c0, r1, c1 = bbox
        h = _bbox_h(bbox)
        w = _bbox_w(bbox)
        if h < 6 or w < 6:
            continue

        if not (
            int(grid[r0, c0]) == int(color)
            and int(grid[r0, c1]) == int(color)
            and int(grid[r1, c0]) == int(color)
            and int(grid[r1, c1]) == int(color)
        ):
            continue

        on_border = (pts[:, 0] == r0) | (pts[:, 0] == r1) | (pts[:, 1] == c0) | (pts[:, 1] == c1)
        if not bool(np.all(on_border)):
            continue

        side_counts = [
            int(np.sum(pts[:, 0] == r0)),
            int(np.sum(pts[:, 0] == r1)),
            int(np.sum(pts[:, 1] == c0)),
            int(np.sum(pts[:, 1] == c1)),
        ]
        if min(side_counts) < 2:
            continue

        score = (int(pts.shape[0]), int(h * w), int(h), int(w), -int(color))
        if best is None or score > best:
            best = score
            best_color = int(color)
            best_bbox = bbox

    if best is None:
        raise ValueError("frame not found")
    return best_color, best_bbox


def _scan_horizontal_gaps(
    grid: np.ndarray,
    row: int,
    c0: int,
    c1: int,
    frame_color: int,
    side: str,
) -> list[dict]:
    gaps: list[dict] = []
    c = int(c0 + 1)
    while c <= int(c1 - 1):
        if int(grid[row, c]) != BG:
            c += 1
            continue
        start = int(c)
        while c <= int(c1 - 1) and int(grid[row, c]) == BG:
            c += 1
        end = int(c - 1)
        left = int(start - 1)
        right = int(end + 1)
        if left >= c0 and right <= c1:
            marker = int(grid[row, left])
            if marker != BG and marker != int(frame_color) and int(grid[row, right]) == marker:
                gaps.append(
                    {
                        "side": side,
                        "color": marker,
                        "start": int(start),
                        "end": int(end),
                        "marker_a": (int(row), int(left)),
                        "marker_b": (int(row), int(right)),
                    }
                )
    return gaps


def _scan_vertical_gaps(
    grid: np.ndarray,
    col: int,
    r0: int,
    r1: int,
    frame_color: int,
    side: str,
) -> list[dict]:
    gaps: list[dict] = []
    r = int(r0 + 1)
    while r <= int(r1 - 1):
        if int(grid[r, col]) != BG:
            r += 1
            continue
        start = int(r)
        while r <= int(r1 - 1) and int(grid[r, col]) == BG:
            r += 1
        end = int(r - 1)
        top = int(start - 1)
        bot = int(end + 1)
        if top >= r0 and bot <= r1:
            marker = int(grid[top, col])
            if marker != BG and marker != int(frame_color) and int(grid[bot, col]) == marker:
                gaps.append(
                    {
                        "side": side,
                        "color": marker,
                        "start": int(start),
                        "end": int(end),
                        "marker_a": (int(top), int(col)),
                        "marker_b": (int(bot), int(col)),
                    }
                )
    return gaps


def _find_gaps(grid: np.ndarray, frame_color: int, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    gaps: list[dict] = []
    gaps.extend(_scan_horizontal_gaps(grid, r0, c0, c1, frame_color, "top"))
    gaps.extend(_scan_horizontal_gaps(grid, r1, c0, c1, frame_color, "bottom"))
    gaps.extend(_scan_vertical_gaps(grid, c0, r0, r1, frame_color, "left"))
    gaps.extend(_scan_vertical_gaps(grid, c1, r0, r1, frame_color, "right"))
    if not gaps:
        raise ValueError("no gaps found")
    return gaps


def _match_piece(grid: np.ndarray, gap: dict) -> dict:
    span = int(gap["end"] - gap["start"] + 1)
    marker_set = {tuple(gap["marker_a"]), tuple(gap["marker_b"])}
    candidates: list[dict] = []
    for comp in _color_components(grid, int(gap["color"])):
        coord_set = {tuple(map(int, p)) for p in comp["coords"]}
        if not marker_set.isdisjoint(coord_set):
            continue
        bbox = comp["bbox"]
        if gap["side"] in ("top", "bottom"):
            if _bbox_w(bbox) != span:
                continue
        else:
            if _bbox_h(bbox) != span:
                continue
        candidates.append(comp)

    if len(candidates) != 1:
        if not candidates:
            raise ValueError("matching piece not found")
        candidates.sort(key=lambda comp: (int(comp["area"]), _bbox_h(comp["bbox"]) * _bbox_w(comp["bbox"])))
    return candidates[-1]


def _shifted_target(
    comp: dict,
    gap: dict,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    br0, bc0, br1, bc1 = comp["bbox"]

    if gap["side"] == "top":
        dr = int(r1 - 1 - br1)
        dc = int(gap["start"] - bc0)
    elif gap["side"] == "bottom":
        dr = int(r0 + 1 - br0)
        dc = int(gap["start"] - bc0)
    elif gap["side"] == "left":
        dr = int(gap["start"] - br0)
        dc = int(c1 - 1 - bc1)
    else:
        dr = int(gap["start"] - br0)
        dc = int(c0 + 1 - bc0)

    return comp["coords"] + np.array([dr, dc], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()

    frame_color, frame_bbox = _find_frame(g)
    gaps = _find_gaps(g, frame_color, frame_bbox)

    moved: list[tuple[np.ndarray, int]] = []
    for gap in gaps:
        comp = _match_piece(g, gap)
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG
        moved.append((_shifted_target(comp, gap, frame_bbox), int(gap["color"])))

    for coords, color in moved:
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_piece_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    min_area = max(4, int(np.ceil(h * w * 0.6)))
    target_area = rng.randint(min_area, h * w)

    while int(mask.sum()) > target_area:
        boundary: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(mask[r, c]) or (r, c) in corners:
                    continue
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                        boundary.append((r, c))
                        break
        if not boundary:
            break
        rng.shuffle(boundary)
        removed = False
        for r, c in boundary:
            mask[r, c] = False
            if _is_connected_mask(mask):
                removed = True
                break
            mask[r, c] = True
        if not removed:
            break

    return mask


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = _mask_cells(mask, r0, c0)
    canvas[coords[:, 0], coords[:, 1]] = int(color)


def _min_side_line(count: int) -> int:
    return int(5 * count + 1)


def _sample_gap_spans(line_len: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_gap_len = min(5, line_len - 4)
    if max_gap_len < 2:
        return None

    for _ in range(100):
        lengths = [rng.randint(2, max_gap_len) for _ in range(count)]
        base_spaces = [1] + [1] * (count - 1) + [1]
        used = int(sum(length + 2 for length in lengths) + sum(base_spaces))
        if used > line_len:
            continue
        spaces = base_spaces[:]
        extra = int(line_len - used)
        for _ in range(extra):
            spaces[rng.randrange(len(spaces))] += 1

        pos = int(spaces[0])
        spans: list[tuple[int, int]] = []
        for idx, length in enumerate(lengths):
            gap_start = int(pos + 1)
            gap_end = int(gap_start + length - 1)
            spans.append((gap_start, gap_end))
            pos = int(gap_end + 2 + spaces[idx + 1])
        return spans
    return None


def _relative_to_gap(side: str, frame_bbox: tuple[int, int, int, int], rel_span: tuple[int, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    start, end = rel_span
    if side in ("top", "bottom"):
        return int(c0 + 1 + start), int(c0 + 1 + end)
    return int(r0 + 1 + start), int(r0 + 1 + end)


def _target_origin(
    side: str,
    frame_bbox: tuple[int, int, int, int],
    gap_start: int,
    mask_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    h, w = mask_shape
    if side == "top":
        return int(r1 - h), int(gap_start)
    if side == "bottom":
        return int(r0 + 1), int(gap_start)
    if side == "left":
        return int(gap_start), int(c1 - w)
    return int(gap_start), int(c0 + 1)


def _outside_positions(
    side: str,
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    gap_start: int,
    mask_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = frame_bbox
    mh, mw = mask_shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for rr in range(0, int(r0 - mh)):
            out.append((int(rr), int(gap_start)))
    elif side == "bottom":
        for rr in range(int(r1 + 2), int(h - mh + 1)):
            out.append((int(rr), int(gap_start)))
    elif side == "left":
        for cc in range(0, int(c0 - mw)):
            out.append((int(gap_start), int(cc)))
    else:
        for cc in range(int(c1 + 2), int(w - mw + 1)):
            out.append((int(gap_start), int(cc)))
    rng = random.Random()
    rng.shuffle(out)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    extra = rng.choice([0, 1, 2])
    gap_counts = {side: 1 for side in SIDES}
    for side in rng.sample(list(SIDES), extra):
        gap_counts[side] += 1

    min_inner_w = max(8, _min_side_line(max(gap_counts["top"], gap_counts["bottom"])))
    min_inner_h = max(8, _min_side_line(max(gap_counts["left"], gap_counts["right"])))
    if min_inner_w > 16 or min_inner_h > 16:
        return None

    interior_w = rng.randint(min_inner_w, min(16, min_inner_w + 4))
    interior_h = rng.randint(min_inner_h, min(16, min_inner_h + 4))

    top_margin = rng.randint(4, 7)
    bottom_margin = rng.randint(4, 7)
    left_margin = rng.randint(4, 7)
    right_margin = rng.randint(4, 7)

    h = int(top_margin + interior_h + 2 + bottom_margin)
    w = int(left_margin + interior_w + 2 + right_margin)
    if h > 30 or w > 30:
        return None

    frame_bbox = (
        int(top_margin),
        int(left_margin),
        int(top_margin + interior_h + 1),
        int(left_margin + interior_w + 1),
    )
    r0, c0, r1, c1 = frame_bbox

    frame_color = rng.choice([c for c in range(1, 10)])
    total_gaps = int(sum(gap_counts.values()))
    piece_colors = rng.sample([c for c in range(1, 10) if c != frame_color], total_gaps)

    spans_by_side: dict[str, list[tuple[int, int]]] = {}
    for side in SIDES:
        line_len = interior_w if side in ("top", "bottom") else interior_h
        spans = _sample_gap_spans(line_len, gap_counts[side], rng)
        if spans is None:
            return None
        spans_by_side[side] = spans

    gaps: list[dict] = []
    color_idx = 0
    for side in SIDES:
        for rel_span in spans_by_side[side]:
            start, end = _relative_to_gap(side, frame_bbox, rel_span)
            gaps.append(
                {
                    "side": side,
                    "color": int(piece_colors[color_idx]),
                    "start": int(start),
                    "end": int(end),
                }
            )
            color_idx += 1

    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    out[r0, c0:c1 + 1] = frame_color
    out[r1, c0:c1 + 1] = frame_color
    out[r0:r1 + 1, c0] = frame_color
    out[r0:r1 + 1, c1] = frame_color
    inp[:, :] = out

    placed_pieces: list[dict] = []
    occupied_final = np.zeros((h, w), dtype=bool)
    for gap in gaps:
        side = gap["side"]
        span = int(gap["end"] - gap["start"] + 1)

        if side == "top":
            max_depth = min(5, int(top_margin - 1), int(interior_h - 1))
            if max_depth < 2:
                return None
            piece_h = rng.randint(2, max_depth)
            piece_w = span
        elif side == "bottom":
            max_depth = min(5, int(bottom_margin - 1), int(interior_h - 1))
            if max_depth < 2:
                return None
            piece_h = rng.randint(2, max_depth)
            piece_w = span
        elif side == "left":
            max_depth = min(5, int(left_margin - 1), int(interior_w - 1))
            if max_depth < 2:
                return None
            piece_h = span
            piece_w = rng.randint(2, max_depth)
        else:
            max_depth = min(5, int(right_margin - 1), int(interior_w - 1))
            if max_depth < 2:
                return None
            piece_h = span
            piece_w = rng.randint(2, max_depth)

        for _ in range(40):
            mask = _random_piece_mask(piece_h, piece_w, rng)
            tr, tc = _target_origin(side, frame_bbox, int(gap["start"]), mask.shape)
            cells = _mask_cells(mask, tr, tc)
            if bool(occupied_final[cells[:, 0], cells[:, 1]].any()):
                continue
            occupied_final[cells[:, 0], cells[:, 1]] = True
            _paint_mask(out, mask, tr, tc, int(gap["color"]))
            placed_pieces.append(
                {
                    "side": side,
                    "color": int(gap["color"]),
                    "mask": mask,
                    "target": (int(tr), int(tc)),
                    "gap_start": int(gap["start"]),
                }
            )
            break
        else:
            return None

    for gap in gaps:
        side = gap["side"]
        color = int(gap["color"])
        if side == "top":
            out[r0, gap["start"]:gap["end"] + 1] = BG
            out[r0, gap["start"] - 1] = color
            out[r0, gap["end"] + 1] = color
        elif side == "bottom":
            out[r1, gap["start"]:gap["end"] + 1] = BG
            out[r1, gap["start"] - 1] = color
            out[r1, gap["end"] + 1] = color
        elif side == "left":
            out[gap["start"]:gap["end"] + 1, c0] = BG
            out[gap["start"] - 1, c0] = color
            out[gap["end"] + 1, c0] = color
        else:
            out[gap["start"]:gap["end"] + 1, c1] = BG
            out[gap["start"] - 1, c1] = color
            out[gap["end"] + 1, c1] = color

    inp[:, :] = out
    inp[occupied_final] = BG

    occupied_input = inp != BG
    rng.shuffle(placed_pieces)
    for piece in placed_pieces:
        positions = _outside_positions(
            piece["side"],
            inp.shape,
            frame_bbox,
            int(piece["gap_start"]),
            piece["mask"].shape,
        )
        if not positions:
            return None
        placed = False
        for rr, cc in positions:
            cells = _mask_cells(piece["mask"], rr, cc)
            if bool(occupied_input[cells[:, 0], cells[:, 1]].any()):
                continue
            occupied_input[cells[:, 0], cells[:, 1]] = True
            _paint_mask(inp, piece["mask"], rr, cc, int(piece["color"]))
            placed = True
            break
        if not placed:
            return None

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, inp):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
