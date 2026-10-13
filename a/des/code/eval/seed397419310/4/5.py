from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


COLOR_ORDER = {
    1: 0,  # blue
    2: 1,  # red
    3: 2,  # green
    4: 3,  # yellow
    5: 4,  # grey
    6: 5,  # magenta
    7: 6,  # orange
    8: 7,  # azure
    9: 8,  # maroon
}

BG = 0
GROUND = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAIR_COUNT_CHOICES = [2, 3, 4]
MAX_GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    values = [int(v) for v in grid.ravel()]
    return int(Counter(values).most_common(1)[0][0])


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = (int(coords[0, 0]), int(coords[0, 1]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return int(len(seen)) == int(mask.sum())


def _bbox_is_full(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    h, w = mask.shape
    return (
        int(rows.min()) == 0
        and int(rows.max()) == h - 1
        and int(cols.min()) == 0
        and int(cols.max()) == w - 1
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rel = [(rr - r0, cc - c0) for rr, cc in coords]
            mask = _mask_from_coords(rel, (r1 - r0 + 1, c1 - c0 + 1))
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "coords": coords,
                }
            )
    return comps


def _rect_from_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> dict | None:
    if mask_a.shape != mask_b.shape:
        return None
    h, w = mask_a.shape
    pts_a = {(int(r), int(c)) for r, c in np.argwhere(mask_a)}
    pts_b0 = [(int(r), int(c)) for r, c in np.argwhere(mask_b)]
    best: dict | None = None

    for dr in range(-h, h + 1):
        for dc in range(-w, w + 1):
            shifted_b = {(int(r + dr), int(c + dc)) for r, c in pts_b0}
            if pts_a & shifted_b:
                continue
            union = pts_a | shifted_b
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            rect_h = int(max(rows) - min(rows) + 1)
            rect_w = int(max(cols) - min(cols) + 1)
            if int(len(union)) != rect_h * rect_w:
                continue
            if best is None or (rect_h * rect_w, rect_h, rect_w) > (
                best["area"],
                best["rect_h"],
                best["rect_w"],
            ):
                best = {"dr": dr, "dc": dc, "rect_h": rect_h, "rect_w": rect_w, "area": rect_h * rect_w}
    return best


def _match_pairs(comps: list[dict]) -> list[tuple[int, int, dict]]:
    n = len(comps)
    assert n % 2 == 0 and n > 0

    pair_info: dict[tuple[int, int], dict] = {}
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i]["mask"].shape != comps[j]["mask"].shape:
                continue
            info = _rect_from_pair(comps[i]["mask"], comps[j]["mask"])
            if info is None:
                continue
            pair_info[(i, j)] = info
            neighbors[i].append(j)
            neighbors[j].append(i)

    def dfs(used: set[int]) -> list[tuple[int, int, dict]] | None:
        if len(used) == n:
            return []
        remaining = [i for i in range(n) if i not in used]
        i = min(remaining, key=lambda idx: len([j for j in neighbors[idx] if j not in used]))
        options = sorted(
            [j for j in neighbors[i] if j not in used],
            key=lambda j: (-pair_info[(min(i, j), max(i, j))]["area"], j),
        )
        for j in options:
            key = (min(i, j), max(i, j))
            rest = dfs(used | {i, j})
            if rest is not None:
                return [(key[0], key[1], pair_info[key])] + rest
        return None

    matched = dfs(set())
    assert matched is not None
    return matched


def _color_rank(color: int) -> tuple[int, int]:
    return (COLOR_ORDER.get(int(color), 99), int(color))


def _build_output(
    grid_shape: tuple[int, int],
    bg: int,
    ground: int,
    pairs: list[dict],
) -> np.ndarray:
    h, w = grid_shape
    out = np.full((h, w), int(bg), dtype=int)
    out[-1, :] = int(ground)
    bottom = h - 2

    ordered = sorted(
        pairs,
        key=lambda item: (
            -int(item["area"]),
            -int(item["rect_h"]),
            -int(item["rect_w"]),
            _color_rank(int(item["colors"][0])),
            _color_rank(int(item["colors"][1])),
        ),
    )

    for item in ordered:
        rect_h = int(item["rect_h"])
        rect_w = int(item["rect_w"])
        top = int(bottom - rect_h + 1)
        left = int((w - rect_w) // 2)
        assert top >= 0 and left >= 0 and left + rect_w <= w

        c1, c2 = sorted((int(item["colors"][0]), int(item["colors"][1])), key=_color_rank)
        split = (rect_w + 1) // 2
        out[top : top + rect_h, left : left + split] = c1
        out[top : top + rect_h, left + split : left + rect_w] = c2
        bottom = top - 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments whose equal-sized shapes can be translated to form a solid rectangle,
    then stack the reconstructed rectangles by descending area above the grey ground line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    ground = int(Counter(int(v) for v in g[-1, :]).most_common(1)[0][0])

    upper = g[: h - 1, :]
    comps = _find_components(upper, bg=bg)
    matched = _match_pairs(comps)

    pairs = []
    for i, j, info in matched:
        pairs.append(
            {
                "rect_h": int(info["rect_h"]),
                "rect_w": int(info["rect_w"]),
                "area": int(info["area"]),
                "colors": (int(comps[i]["color"]), int(comps[j]["color"])),
            }
        )

    return _build_output(g.shape, bg=bg, ground=ground, pairs=pairs)


def _sample_cuts(length: int, max_cut: int) -> list[int]:
    cuts = [random.randint(0, max_cut) for _ in range(length)]
    i0, i1 = random.sample(range(length), 2)
    cuts[i0] = 0
    cuts[i1] = max_cut
    if max_cut > 1 and len(set(cuts)) < 3:
        extra = [i for i in range(length) if i not in (i0, i1)]
        if extra:
            cuts[random.choice(extra)] = random.randint(1, max_cut - 1)
    return cuts


def _make_horizontal_pair(color_a: int, color_b: int) -> dict:
    piece_h = random.randint(3, 6)
    piece_w = random.randint(4, 7)
    shift = random.randint(1, piece_w - 1)
    overlap = piece_w - shift
    cuts = _sample_cuts(piece_h, overlap)

    mask_a = np.zeros((piece_h, piece_w), dtype=bool)
    mask_b = np.zeros((piece_h, piece_w), dtype=bool)
    for r, cut in enumerate(cuts):
        mask_a[r, : shift + cut] = True
        mask_b[r, cut:] = True

    assert _is_connected(mask_a) and _is_connected(mask_b)
    assert _bbox_is_full(mask_a) and _bbox_is_full(mask_b)
    return {
        "orientation": "h",
        "piece_h": piece_h,
        "piece_w": piece_w,
        "rect_h": piece_h,
        "rect_w": piece_w + shift,
        "area": piece_h * (piece_w + shift),
        "mask_a": mask_a,
        "mask_b": mask_b,
        "colors": (int(color_a), int(color_b)),
    }


def _make_vertical_pair(color_a: int, color_b: int) -> dict:
    piece_h = random.randint(4, 7)
    piece_w = random.randint(3, 6)
    shift = random.randint(1, piece_h - 1)
    overlap = piece_h - shift
    cuts = _sample_cuts(piece_w, overlap)

    mask_a = np.zeros((piece_h, piece_w), dtype=bool)
    mask_b = np.zeros((piece_h, piece_w), dtype=bool)
    for c, cut in enumerate(cuts):
        mask_a[: shift + cut, c] = True
        mask_b[cut:, c] = True

    assert _is_connected(mask_a) and _is_connected(mask_b)
    assert _bbox_is_full(mask_a) and _bbox_is_full(mask_b)
    return {
        "orientation": "v",
        "piece_h": piece_h,
        "piece_w": piece_w,
        "rect_h": piece_h + shift,
        "rect_w": piece_w,
        "area": (piece_h + shift) * piece_w,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "colors": (int(color_a), int(color_b)),
    }


def _make_pair(color_a: int, color_b: int) -> dict:
    if random.random() < 0.5:
        return _make_horizontal_pair(color_a, color_b)
    return _make_vertical_pair(color_a, color_b)


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _place_pieces(grid_shape: tuple[int, int], pieces: list[dict]) -> np.ndarray | None:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    grid[-1, :] = GROUND
    blocked = np.zeros((h - 1, w), dtype=bool)

    for piece in sorted(pieces, key=lambda item: item["piece_h"] * item["piece_w"], reverse=True):
        ph = int(piece["piece_h"])
        pw = int(piece["piece_w"])
        candidates = [
            (r, c)
            for r in range(0, h - 1 - ph + 1)
            for c in range(0, w - pw + 1)
            if not blocked[r : r + ph, c : c + pw].any()
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        patch = _mask_to_patch(piece["mask"], int(piece["color"]), BG)
        grid[r0 : r0 + ph, c0 : c0 + pw][piece["mask"]] = int(piece["color"])

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    return grid


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(FRAGMENT_COLORS, 2 * pair_count)

    pairs = []
    used_piece_dims: set[tuple[int, int]] = set()
    used_areas: set[int] = set()
    for k in range(pair_count):
        color_a = int(colors[2 * k])
        color_b = int(colors[2 * k + 1])
        pair = None
        for _ in range(100):
            candidate = _make_pair(color_a, color_b)
            dims = (int(candidate["piece_h"]), int(candidate["piece_w"]))
            if dims in used_piece_dims or int(candidate["area"]) in used_areas:
                continue
            pair = candidate
            used_piece_dims.add(dims)
            used_areas.add(int(candidate["area"]))
            break
        if pair is None:
            return None
        pairs.append(pair)

    stack_height = sum(int(pair["rect_h"]) for pair in pairs)
    if stack_height > 22:
        return None

    total_bbox_area = sum(2 * int(pair["piece_h"]) * int(pair["piece_w"]) for pair in pairs)
    max_rect_w = max(int(pair["rect_w"]) for pair in pairs)

    width_low = max(16, max_rect_w + 6, int(np.ceil(np.sqrt(total_bbox_area * 2.2))))
    width = random.randint(width_low, width_low + 5)
    height_low = max(12, stack_height + 2, int(np.ceil(total_bbox_area / max(1, width - 2))) + 7)
    height = random.randint(height_low, height_low + 5)

    pieces = []
    for pair in pairs:
        color_a, color_b = pair["colors"]
        pieces.append(
            {
                "piece_h": int(pair["piece_h"]),
                "piece_w": int(pair["piece_w"]),
                "mask": pair["mask_a"],
                "color": int(color_a),
            }
        )
        pieces.append(
            {
                "piece_h": int(pair["piece_h"]),
                "piece_w": int(pair["piece_w"]),
                "mask": pair["mask_b"],
                "color": int(color_b),
            }
        )

    inp = _place_pieces((height, width), pieces)
    if inp is None:
        return None

    expected = _build_output(
        inp.shape,
        bg=BG,
        ground=GROUND,
        pairs=[
            {
                "rect_h": int(pair["rect_h"]),
                "rect_w": int(pair["rect_w"]),
                "area": int(pair["area"]),
                "colors": tuple(int(c) for c in pair["colors"]),
            }
            for pair in pairs
        ],
    )
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
