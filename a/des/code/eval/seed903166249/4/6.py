from __future__ import annotations

import itertools
import random

import numpy as np


AZURE = 8
GREY = 5
GREEN = 3
MARKER_POOL = [0, 1, 2, 4, 6, 7, 9]
BODY_SIZE_RANGE = (22, 30)
PAIR_COUNT_RANGE = (2, 4)
SPAN_CHOICES = (1, 2)
DEPTH_CHOICES = (1, 2)
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _mode(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    counts = np.bincount(flat, minlength=10)
    best = int(np.argmax(counts))
    ordered = np.sort(counts)[::-1]
    assert ordered[0] > 0
    assert len(ordered) == 1 or ordered[0] > ordered[1], "expected a unique mode"
    return best


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_color(body: np.ndarray) -> int:
    edge = np.concatenate((body[0, :], body[-1, :], body[:, 0], body[:, -1]))
    vals = np.unique(edge)
    assert vals.size == 1, "expected a single body border color"
    return int(vals[0])


def _parse_legend(row: np.ndarray, legend_bg: int) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    w = row.shape[0]
    c = 0
    while c <= w - 4:
        if (
            int(row[c]) != legend_bg
            and int(row[c + 1]) != legend_bg
            and int(row[c + 2]) == legend_bg
            and int(row[c + 3]) != legend_bg
            and (c == 0 or int(row[c - 1]) == legend_bg)
        ):
            key = tuple(sorted((int(row[c]), int(row[c + 1]))))
            value = int(row[c + 3])
            prev = mapping.get(key)
            assert prev is None or prev == value, "legend mapping must be consistent"
            mapping[key] = value
            c += 4
        else:
            c += 1
    assert mapping, "expected at least one legend entry"
    return mapping


def _extract_pieces(body: np.ndarray, bg: int, frag: int) -> list[dict]:
    pieces = []
    for pts in _components(body != bg):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        shape = np.zeros((h, w), dtype=bool)
        marker_cells: list[tuple[int, int, int]] = []
        for r, c in pts:
            shape[r - r0, c - c0] = True
            color = int(body[r, c])
            if color != frag:
                marker_cells.append((r - r0, c - c0, color))
        assert len(marker_cells) == 1, "each piece must contain exactly one marker pixel"
        mr, mc, marker = marker_cells[0]
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": shape,
                "marker": int(marker),
                "marker_pos": (mr, mc),
            }
        )
    assert pieces and len(pieces) % 2 == 0, "expected an even number of fragments"
    return pieces


def _pair_pieces(pieces: list[dict], color_map: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    n = len(pieces)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a = pieces[i]
            b = pieces[j]
            if a["shape"].shape != b["shape"].shape:
                continue
            if tuple(sorted((a["marker"], b["marker"]))) not in color_map:
                continue
            if np.all(a["shape"] | b["shape"]):
                neighbors[i].append(j)
                neighbors[j].append(i)

    def search(remaining: frozenset[int]) -> list[tuple[int, int]] | None:
        if not remaining:
            return []
        pivot = min(remaining, key=lambda idx: len([j for j in neighbors[idx] if j in remaining]))
        candidates = [j for j in neighbors[pivot] if j in remaining]
        if not candidates:
            return None
        for j in candidates:
            tail = search(remaining - {pivot, j})
            if tail is not None:
                return [(pivot, j)] + tail
        return None

    pairs = search(frozenset(range(n)))
    assert pairs is not None, "expected a complete complementary pairing"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary green fragments into rectangles, then recolor each rectangle using the
    marker-color combination table from the top legend strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 4, "expected a nonempty 2D grid with a legend row"

    legend = g[0, :]
    body = g[1:, :]
    legend_bg = _mode(legend)
    bg = _border_color(body)
    frag = _mode(body[body != bg])
    color_map = _parse_legend(legend, legend_bg)
    pieces = _extract_pieces(body, bg, frag)
    pairs = _pair_pieces(pieces, color_map)

    out = np.full(body.shape, bg, dtype=int)
    for i, j in pairs:
        a = pieces[i]
        b = pieces[j]
        h, w = a["shape"].shape
        anchor = a if (a["bbox"][0], a["bbox"][1]) <= (b["bbox"][0], b["bbox"][1]) else b
        r0, c0, _, _ = anchor["bbox"]
        result_color = color_map[tuple(sorted((a["marker"], b["marker"])))]
        patch = out[r0 : r0 + h, c0 : c0 + w]
        assert patch.shape == (h, w)
        assert np.all(patch == bg), "assembled rectangles must not overlap"
        patch[:] = result_color
    return out


def _horizontal_pair(h: int, w: int, depth: int, span: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    for r in range(h):
        if offset <= r < offset + span:
            a[r, depth:] = True
            b[r, : w - depth] = True
        else:
            a[r, : w - depth] = True
            b[r, depth:] = True
    return a, b


def _vertical_pair(h: int, w: int, depth: int, span: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    for c in range(w):
        if offset <= c < offset + span:
            a[depth:, c] = True
            b[: h - depth, c] = True
        else:
            a[: h - depth, c] = True
            b[depth:, c] = True
    return a, b


def _marker_choices(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    core = mask_a & mask_b
    pts = [
        (int(r), int(c))
        for r, c in np.argwhere(core)
        if 0 < int(r) < core.shape[0] - 1 and 0 < int(c) < core.shape[1] - 1
    ]
    if pts:
        return pts
    return [(int(r), int(c)) for r, c in np.argwhere(core)]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _find_spot(
    body_h: int,
    body_w: int,
    h: int,
    w: int,
    occupied: list[tuple[int, int, int, int]],
    *,
    min_r: int,
    max_r: int,
    min_c: int,
    max_c: int,
    tries: int = PLACEMENT_TRIES,
) -> tuple[int, int] | None:
    if min_r > max_r or min_c > max_c:
        return None
    for _ in range(tries):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_boxes_overlap(box, other, gap=PLACEMENT_GAP) for other in occupied):
            continue
        return r0, c0
    return None


def _paint_piece(
    body: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    marker_color: int,
    marker_pos: tuple[int, int],
) -> None:
    h, w = mask.shape
    region = body[top : top + h, left : left + w]
    assert region.shape == (h, w)
    assert np.all(region[mask] == AZURE), "pieces must not overlap"
    region[mask] = GREEN
    mr, mc = marker_pos
    assert bool(mask[mr, mc]), "marker must lie on the fragment"
    region[mr, mc] = marker_color


def _legend_row(width: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    row = np.full(width, GREY, dtype=int)
    groups = entries[:]
    random.shuffle(groups)
    starts: list[int] = []
    cursor = random.randint(0, 2)
    for idx, _entry in enumerate(groups):
        starts.append(cursor)
        cursor += 5 + random.randint(0, 2)
        if idx == len(groups) - 1:
            break
    assert starts, "legend needs at least one entry"
    needed = starts[-1] + 4
    assert needed <= width, "legend row is too narrow for the sampled entries"
    for start, (a, b, result) in zip(starts, groups):
        if random.random() < 0.5:
            row[start] = a
            row[start + 1] = b
        else:
            row[start] = b
            row[start + 1] = a
        row[start + 3] = result
    return row


def _sample_pair_spec(marker_pair: tuple[int, int], result_color: int) -> dict | None:
    orientation = random.choice(("horizontal", "vertical"))
    depth = random.choice(DEPTH_CHOICES)
    span = random.choice(SPAN_CHOICES)

    if orientation == "horizontal":
        h = random.randint(5, 8)
        w = random.randint(max(6, 2 * depth + 3), 9)
        if h <= span + 2:
            return None
        offset = random.randint(1, h - span - 1)
        mask_a, mask_b = _horizontal_pair(h, w, depth, span, offset)
    else:
        h = random.randint(max(6, 2 * depth + 3), 9)
        w = random.randint(5, 8)
        if w <= span + 2:
            return None
        offset = random.randint(1, w - span - 1)
        mask_a, mask_b = _vertical_pair(h, w, depth, span, offset)

    marker_cells = _marker_choices(mask_a, mask_b)
    if len(marker_cells) < 2:
        return None
    pos1, pos2 = random.sample(marker_cells, 2)
    return {
        "orientation": orientation,
        "h": h,
        "w": w,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "marker_colors": marker_pair,
        "marker_positions": (pos1, pos2),
        "result_color": result_color,
    }


def _render_pair_input_output(specs: list[dict], body_h: int, body_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    body = np.full((body_h, body_w), AZURE, dtype=int)
    out = np.full((body_h, body_w), AZURE, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []

    # Place larger rectangles first to reduce retries.
    order = sorted(range(len(specs)), key=lambda i: specs[i]["h"] * specs[i]["w"], reverse=True)
    placements: dict[int, dict] = {}
    for idx in order:
        spec = specs[idx]
        h, w = spec["h"], spec["w"]
        anchor = _find_spot(
            body_h,
            body_w,
            h,
            w,
            occupied,
            min_r=1,
            max_r=body_h - h - 2,
            min_c=1,
            max_c=body_w - w - 2,
        )
        if anchor is None:
            return None
        ar, ac = anchor
        occupied.append((ar, ac, ar + h - 1, ac + w - 1))

        second = _find_spot(
            body_h,
            body_w,
            h,
            w,
            occupied,
            min_r=ar + 1,
            max_r=body_h - h - 1,
            min_c=ac + 1,
            max_c=body_w - w - 1,
        )
        if second is None:
            return None
        br, bc = second
        occupied.append((br, bc, br + h - 1, bc + w - 1))
        placements[idx] = {"anchor": (ar, ac), "second": (br, bc)}

    for idx, spec in enumerate(specs):
        ar, ac = placements[idx]["anchor"]
        br, bc = placements[idx]["second"]
        first_mask, second_mask = (spec["mask_a"], spec["mask_b"])
        first_marker, second_marker = spec["marker_colors"]
        first_pos, second_pos = spec["marker_positions"]
        if random.random() < 0.5:
            first_mask, second_mask = second_mask, first_mask
            first_marker, second_marker = second_marker, first_marker
            first_pos, second_pos = second_pos, first_pos

        _paint_piece(body, first_mask, ar, ac, first_marker, first_pos)
        _paint_piece(body, second_mask, br, bc, second_marker, second_pos)
        out[ar : ar + spec["h"], ac : ac + spec["w"]] = spec["result_color"]

    return body, out


def _body_bg_is_unique(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(counts[AZURE]) > int(np.max(counts[[c for c in range(10) if c != AZURE]]))


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    marker_pairs = random.sample(list(itertools.combinations(MARKER_POOL, 2)), pair_count)

    result_colors: list[int] = []
    used_results: set[int] = set()
    for a, b in marker_pairs:
        choices = [c for c in MARKER_POOL if c not in {a, b} and c not in used_results]
        if not choices:
            choices = [c for c in MARKER_POOL if c not in {a, b}]
        result = random.choice(choices)
        used_results.add(result)
        result_colors.append(result)

    specs: list[dict] = []
    for pair, result in zip(marker_pairs, result_colors):
        spec = _sample_pair_spec(pair, result)
        if spec is None:
            return None
        specs.append(spec)

    body_h = random.randint(*BODY_SIZE_RANGE)
    min_legend_w = 6 * pair_count + 3
    body_w = random.randint(max(BODY_SIZE_RANGE[0], min_legend_w), BODY_SIZE_RANGE[1])
    rendered = _render_pair_input_output(specs, body_h, body_w)
    if rendered is None:
        return None
    body, expected = rendered
    if not _body_bg_is_unique(body):
        return None

    legend = _legend_row(body_w, [(a, b, r) for (a, b), r in zip(marker_pairs, result_colors)])
    inp = np.vstack([legend, body])
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid complementary-fragment sample")
