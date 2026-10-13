from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, GREY))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _enclosed_mask(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    exterior = np.zeros_like(boundary_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int) -> None:
        if not boundary_mask[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not boundary_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((int(nr), int(nc)))

    return (~boundary_mask) & (~exterior)


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    segments: list[dict] = []
    for seg_id, comp in enumerate(_components(g != BG)):
        coords = np.asarray(comp, dtype=int)
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each segment component must be monochrome"
        color = int(next(iter(colors)))
        rows = coords[:, 0]
        cols = coords[:, 1]

        if int(rows.min()) == int(rows.max()):
            r = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert len(comp) == c1 - c0 + 1, "horizontal segment must be contiguous"
            endpoints = [(r, c0), (r, c1)]
            continuations = [(r, c0 - 1), (r, c1 + 1)]
        else:
            assert int(cols.min()) == int(cols.max()), "each component must be a straight segment"
            c = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert len(comp) == r1 - r0 + 1, "vertical segment must be contiguous"
            endpoints = [(r0, c), (r1, c)]
            continuations = [(r0 - 1, c), (r1 + 1, c)]

        assert len(comp) >= 3, "segments must have length at least 3"
        segments.append(
            {
                "id": int(seg_id),
                "color": color,
                "coords": [(int(r), int(c)) for r, c in comp],
                "length": int(len(comp)),
                "endpoints": endpoints,
                "continuations": continuations,
            }
        )
    return segments


def _build_connectors(grid: np.ndarray, segments: list[dict]) -> list[dict]:
    h, w = grid.shape
    continuation_map: dict[tuple[int, int], list[int]] = {}
    for seg in segments:
        for cont in seg["continuations"]:
            rr, cc = cont
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                continuation_map.setdefault((rr, cc), []).append(int(seg["id"]))

    connectors: list[dict] = []
    for cell, seg_ids in sorted(continuation_map.items()):
        uniq = sorted(set(int(seg_id) for seg_id in seg_ids))
        if len(seg_ids) == 2 and len(uniq) == 2:
            connectors.append({"cell": cell, "segments": (int(uniq[0]), int(uniq[1]))})
    return connectors


def _eligible_segment_components(segment_count: int, connectors: list[dict]) -> list[set[int]]:
    degree = Counter()
    adj = [set() for _ in range(segment_count)]
    for conn in connectors:
        a, b = conn["segments"]
        degree[int(a)] += 1
        degree[int(b)] += 1
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    eligible = {seg_id for seg_id, count in degree.items() if count == 2}
    comps: list[set[int]] = []
    seen: set[int] = set()
    for seg_id in sorted(eligible):
        if seg_id in seen:
            continue
        q = deque([int(seg_id)])
        seen.add(int(seg_id))
        comp: set[int] = set()
        while q:
            cur = q.popleft()
            comp.add(int(cur))
            for nxt in adj[cur]:
                if nxt in eligible and nxt not in seen:
                    seen.add(int(nxt))
                    q.append(int(nxt))
        comps.append(comp)
    return comps


def _boundary_mask(shape: tuple[int, int], segments: list[dict], connectors: list[dict], seg_ids: set[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for seg_id in seg_ids:
        for r, c in segments[seg_id]["coords"]:
            mask[r, c] = True
    for conn in connectors:
        a, b = conn["segments"]
        if a in seg_ids and b in seg_ids:
            r, c = conn["cell"]
            mask[r, c] = True
    return mask


def _winning_color(colors: list[int], lengths: list[int]) -> int:
    seg_counts = Counter(int(color) for color in colors)
    pixel_counts = Counter()
    for color, length in zip(colors, lengths, strict=False):
        pixel_counts[int(color)] += int(length)

    scores = {color: (int(seg_counts[color]), int(pixel_counts[color])) for color in seg_counts}
    best_score = max(scores.values())
    winners = [color for color, score in scores.items() if score == best_score]
    assert len(winners) == 1, "fill color must be unique after the tie-break"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect the hidden polygon from separated line segments and fill its interior by the dominant segment color.

    Input structure:
    - The background is black.
    - The foreground consists of several one-pixel-thick straight line segments, each
      horizontal or vertical, with lengths between 3 and 8 pixels.
    - Some segment endpoints stop one black pixel short of another segment endpoint,
      while other segments are isolated distractors and do not participate in the polygon.

    Rule:
    - For every pair of segment endpoints whose outward continuation lands on the same
      black cell, recolor that shared gap cell grey to reconnect the hidden outline.
    - Among the connected segment group that now encloses a region, choose the fill color
      by segment-color frequency; if multiple colors use the same number of segments,
      break the tie by the total number of outline pixels contributed by that color.
    - Fill the black region enclosed by that polygon outline with the chosen color using
      4-connectivity, while keeping original segment colors and the new grey connectors unchanged.

    Key insight:
    The missing outline pixels are identified from endpoint continuation directions, and
    only the segment network that closes into an actual enclosed region determines the fill.
    """
    g = np.asarray(grid, dtype=int)
    segments = _extract_segments(g)
    connectors = _build_connectors(g, segments)

    out = g.copy()
    for conn in connectors:
        r, c = conn["cell"]
        out[r, c] = GREY

    seg_comps = _eligible_segment_components(len(segments), connectors)
    if not seg_comps:
        return out

    chosen_ids: set[int] | None = None
    chosen_area = 0
    for seg_ids in seg_comps:
        mask = _boundary_mask(g.shape, segments, connectors, seg_ids)
        area = int(_enclosed_mask(mask).sum())
        if area > chosen_area:
            chosen_area = area
            chosen_ids = seg_ids

    if chosen_ids is None or chosen_area == 0:
        return out

    chosen_colors = [int(segments[seg_id]["color"]) for seg_id in chosen_ids]
    chosen_lengths = [int(segments[seg_id]["length"]) for seg_id in chosen_ids]
    fill_color = _winning_color(chosen_colors, chosen_lengths)
    chosen_mask = _boundary_mask(g.shape, segments, connectors, chosen_ids)

    for comp in _components(out == BG):
        if any(r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in comp):
            continue
        touches_boundary = False
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and chosen_mask[nr, nc]:
                    touches_boundary = True
                    break
            if touches_boundary:
                break
        if touches_boundary:
            for r, c in comp:
                out[r, c] = fill_color

    return out


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(int(r), int(c0)) for r in range(r0, r1 + step, step)]


def _sample_partition(edge_len: int) -> list[int] | None:
    valid_counts = [
        count
        for count in range(1, edge_len)
        if 3 * count <= edge_len - count - 1 <= 8 * count
    ]
    if not valid_counts:
        return None
    count = random.choice(valid_counts)
    remaining = edge_len - count - 1
    parts: list[int] = []
    for idx in range(count):
        left = count - idx - 1
        lo = max(3, remaining - 8 * left)
        hi = min(8, remaining - 3 * left)
        if lo > hi:
            return None
        part = random.randint(lo, hi)
        parts.append(int(part))
        remaining -= int(part)
    if remaining != 0:
        return None
    return parts


def _rectangle_edges() -> list[list[tuple[int, int]]]:
    height = random.randint(8, 15)
    width = random.randint(8, 15)
    return [
        _line_cells(0, 0, 0, width - 1),
        _line_cells(0, width - 1, height - 1, width - 1),
        _line_cells(height - 1, width - 1, height - 1, 0),
        _line_cells(height - 1, 0, 0, 0),
    ]


def _notched_edges() -> list[list[tuple[int, int]]]:
    height = random.randint(10, 16)
    width = random.randint(10, 16)
    cut_r = random.randint(4, height - 5)
    cut_c = random.randint(4, width - 5)
    return [
        _line_cells(0, 0, 0, width - 1),
        _line_cells(0, width - 1, cut_r, width - 1),
        _line_cells(cut_r, width - 1, cut_r, cut_c),
        _line_cells(cut_r, cut_c, height - 1, cut_c),
        _line_cells(height - 1, cut_c, height - 1, 0),
        _line_cells(height - 1, 0, 0, 0),
    ]


def _segmentize_edges(edges: list[list[tuple[int, int]]]) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]] | None:
    segments: list[list[tuple[int, int]]] = []
    seams: set[tuple[int, int]] = set()
    for edge in edges:
        parts = _sample_partition(len(edge))
        if parts is None:
            return None
        seams.add(edge[0])
        idx = 1
        for part in parts:
            seg = edge[idx : idx + part]
            if len(seg) != part:
                return None
            segments.append(list(seg))
            idx += part
            seams.add(edge[idx])
            idx += 1
        if idx != len(edge):
            return None
    return segments, seams


def _assign_unique_colors(lengths: list[int]) -> list[int] | None:
    n = len(lengths)
    if n < 4:
        return None
    winner_count = random.randint(max(2, n // 2), n - 1)
    remaining = n - winner_count
    cap = winner_count - 1
    other_needed = max(1, (remaining + cap - 1) // cap)
    if 1 + other_needed > len(COLOR_POOL):
        return None

    palette = random.sample(COLOR_POOL, 1 + other_needed)
    winner = int(palette[0])
    assignments = [None] * n
    winner_ids = set(random.sample(range(n), winner_count))
    for idx in winner_ids:
        assignments[idx] = winner

    other_counts = Counter()
    for idx in range(n):
        if assignments[idx] is not None:
            continue
        choices = [color for color in palette[1:] if other_counts[color] < cap]
        if not choices:
            return None
        color = int(random.choice(choices))
        assignments[idx] = color
        other_counts[color] += 1

    colors = [int(color) for color in assignments]
    if _winning_color(colors, lengths) != winner:
        return None
    return colors


def _assign_tie_colors(lengths: list[int]) -> list[int] | None:
    n = len(lengths)
    if n < 4 or len(set(lengths)) == 1:
        return None

    valid_ks = [
        k
        for k in range(2, n // 2 + 1)
        if n - 2 * k <= (len(COLOR_POOL) - 2) * (k - 1)
    ]
    random.shuffle(valid_ks)
    ranked_ids = sorted(range(n), key=lambda idx: lengths[idx], reverse=True)

    for k in valid_ks:
        a_ids = ranked_ids[:k]
        b_ids = ranked_ids[k : 2 * k]
        if sum(lengths[idx] for idx in a_ids) == sum(lengths[idx] for idx in b_ids):
            continue

        remaining = n - 2 * k
        cap = k - 1
        other_needed = 0 if remaining == 0 else (remaining + cap - 1) // cap
        if 2 + other_needed > len(COLOR_POOL):
            continue

        palette = random.sample(COLOR_POOL, 2 + other_needed)
        winner = int(palette[0])
        runner = int(palette[1])
        assignments = [None] * n
        for idx in a_ids:
            assignments[idx] = winner
        for idx in b_ids:
            assignments[idx] = runner

        other_counts = Counter()
        for idx in ranked_ids[2 * k :]:
            choices = [color for color in palette[2:] if other_counts[color] < cap]
            if not choices:
                assignments = None
                break
            color = int(random.choice(choices))
            assignments[idx] = color
            other_counts[color] += 1

        if assignments is None:
            continue
        colors = [int(color) for color in assignments]
        if _winning_color(colors, lengths) == winner:
            return colors

    return None


def _build_polygon_patch() -> tuple[np.ndarray, np.ndarray] | None:
    edges = random.choice([_rectangle_edges, _notched_edges])()
    partitioned = _segmentize_edges(edges)
    if partitioned is None:
        return None
    segments0, seams0 = partitioned

    all_cells = [cell for seg in segments0 for cell in seg] + list(seams0)
    max_r = max(r for r, _c in all_cells)
    max_c = max(c for _r, c in all_cells)
    segments = [[(r + 1, c + 1) for r, c in seg] for seg in segments0]
    seams = {(r + 1, c + 1) for r, c in seams0}

    lengths = [len(seg) for seg in segments]
    use_tie_branch = random.random() < 0.5
    colors = _assign_tie_colors(lengths) if use_tie_branch else _assign_unique_colors(lengths)
    if colors is None:
        return None
    fill_color = _winning_color(colors, lengths)

    patch = np.full((max_r + 3, max_c + 3), BG, dtype=int)
    inp = patch.copy()
    out = patch.copy()

    for seg, color in zip(segments, colors, strict=False):
        for r, c in seg:
            inp[r, c] = int(color)
            out[r, c] = int(color)
    for r, c in seams:
        out[r, c] = GREY

    enclosed = _enclosed_mask(out != BG)
    out[enclosed] = fill_color

    rot = random.randrange(4)
    if rot:
        inp = np.rot90(inp, rot)
        out = np.rot90(out, rot)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    return inp, out


def _can_place_distractor(
    inp: np.ndarray,
    out: np.ndarray,
    cells: list[tuple[int, int]],
    endpoints: list[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(inp[r, c]) != BG or int(out[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(inp[nr, nc]) != BG:
                return False

    cand_endpoints = [cells[0], cells[-1]]
    for er, ec in cand_endpoints:
        for fr, fc in endpoints:
            if abs(er - fr) + abs(ec - fc) == 2:
                return False
    return True


def _place_distractors(inp: np.ndarray, out: np.ndarray, count: int) -> bool:
    segments = _extract_segments(inp)
    endpoints = [tuple(endpoint) for seg in segments for endpoint in seg["endpoints"]]
    h, w = inp.shape

    placed = 0
    for _ in range(PLACE_ATTEMPTS):
        if placed == count:
            return True
        vertical = random.random() < 0.5
        length = random.randint(3, 8)
        color = int(random.choice(COLOR_POOL))
        if vertical:
            if h < length:
                continue
            r0 = random.randint(0, h - length)
            c0 = random.randint(0, w - 1)
            cells = [(r0 + dr, c0) for dr in range(length)]
        else:
            if w < length:
                continue
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, w - length)
            cells = [(r0, c0 + dc) for dc in range(length)]

        if not _can_place_distractor(inp, out, cells, endpoints):
            continue

        for r, c in cells:
            inp[r, c] = color
            out[r, c] = color
        endpoints.extend([cells[0], cells[-1]])
        placed += 1

    return placed == count


def _generate_case() -> np.ndarray | None:
    patch = _build_polygon_patch()
    if patch is None:
        return None
    patch_inp, patch_out = patch
    ph, pw = patch_inp.shape
    if ph + 3 > 30 or pw + 3 > 30:
        return None

    grid_h = random.randint(ph + 3, 30)
    grid_w = random.randint(pw + 3, 30)
    r0 = random.randint(0, grid_h - ph)
    c0 = random.randint(0, grid_w - pw)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    inp[r0 : r0 + ph, c0 : c0 + pw] = patch_inp
    out[r0 : r0 + ph, c0 : c0 + pw] = patch_out

    distractor_count = random.randint(1, 4)
    if not _place_distractors(inp, out, distractor_count):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid hidden-polygon instance")
