from __future__ import annotations

from collections import Counter, deque
import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
GEN_RETRIES = 500
SCALE_PAIRS = tuple((sx, sy) for sx in range(1, 5) for sy in range(1, 5) if (sx, sy) != (1, 1))
DIFF_SCALE_PAIRS = tuple(pair for pair in SCALE_PAIRS if pair[0] != pair[1])
SAME_SCALE_PAIRS = tuple(pair for pair in SCALE_PAIRS if pair[0] == pair[1])


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return comps


def _frame_info(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    top = 0
    while top < h and bool(local[top].all()):
        top += 1
    bottom = 0
    while bottom < h and bool(local[h - 1 - bottom].all()):
        bottom += 1
    left = 0
    while left < w and bool(local[:, left].all()):
        left += 1
    right = 0
    while right < w and bool(local[:, w - 1 - right].all()):
        right += 1

    if top == 0 or left == 0 or top != bottom or left != right:
        return None
    if h - 2 * top <= 0 or w - 2 * left <= 0:
        return None

    expected = np.zeros_like(local)
    expected[:top, :] = True
    expected[h - top :, :] = True
    expected[:, :left] = True
    expected[:, w - left :] = True
    if not np.array_equal(local, expected):
        return None

    ir0 = r0 + top
    ic0 = c0 + left
    ir1 = r1 - top
    ic1 = c1 - left
    cavity = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    pattern_mask = (cavity != bg) & (cavity != color)
    inside_count = int(pattern_mask.sum())
    if inside_count == 0:
        return None

    total_non_bg_non_frame = int(np.count_nonzero((grid != bg) & (grid != color)))
    outside_count = total_non_bg_non_frame - inside_count

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "sx": left,
        "sy": top,
        "cavity_bbox": (ir0, ic0, ir1, ic1),
        "pattern_mask": pattern_mask,
        "inside_count": inside_count,
        "outside_count": outside_count,
        "area": int(comp["area"]),
    }


def _locate_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for comp in _find_components(grid, bg):
        info = _frame_info(grid, bg, comp)
        if info is not None:
            candidates.append(info)

    if not candidates:
        raise ValueError("no valid rectangular frame found")

    strict = [cand for cand in candidates if cand["outside_count"] == 0]
    pool = strict if strict else candidates
    pool = sorted(
        pool,
        key=lambda cand: (
            cand["outside_count"],
            -cand["inside_count"],
            -(cand["area"]),
        ),
    )
    return pool[0]


def _anisotropic_scale(grid: np.ndarray, sy: int, sx: int) -> np.ndarray:
    return np.repeat(np.repeat(grid, int(sy), axis=0), int(sx), axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a thick rectangular frame as an anisotropic lens and enlarge the enclosed pattern.

    Input structure:
    - A mostly uniform background contains one monochrome hollow rectangular frame.
    - The left and right frame sides share one thickness, and the top and bottom sides share another.
    - Inside the frame cavity sits a smaller pattern made of colors distinct from both background and frame, with spare blank margin around it.

    Rule:
    - Identify the frame and use its left/right thickness as the horizontal scale factor and its top/bottom thickness as the vertical scale factor.
    - Inside the cavity, take the tight bbox of all non-background, non-frame pixels and crop that entire bbox, preserving any background holes inside it.
    - Upscale the cropped pattern by repeating columns with the horizontal factor and rows with the vertical factor, and output only that enlarged crop.

    Key insight:
    - The frame is not the output window: its border thicknesses encode the two independent scaling factors, while the pattern must first be tightly cropped before enlargement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frame = _locate_frame(g, bg)
    ir0, ic0, ir1, ic1 = frame["cavity_bbox"]
    cavity = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    pattern_bbox = _bbox(frame["pattern_mask"])
    assert pattern_bbox is not None

    pattern = _crop_bbox(cavity, pattern_bbox)
    out = _anisotropic_scale(pattern, frame["sy"], frame["sx"])
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _sample_scale_pair() -> tuple[int, int]:
    if np.random.random() < 0.75:
        return DIFF_SCALE_PAIRS[np.random.randint(len(DIFF_SCALE_PAIRS))]
    return SAME_SCALE_PAIRS[np.random.randint(len(SAME_SCALE_PAIRS))]


def _random_side_touching_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    seeds = {
        (0, np.random.randint(w)),
        (h - 1, np.random.randint(w)),
        (np.random.randint(h), 0),
        (np.random.randint(h), w - 1),
    }

    max_fg = area - 1 if area > 4 else area
    min_fg = max(len(seeds), min(max_fg, 4))
    if min_fg > max_fg:
        min_fg = max_fg
    target = np.random.randint(min_fg, max_fg + 1)

    mask = np.zeros((h, w), dtype=bool)
    frontier = []
    for cell in seeds:
        if not mask[cell]:
            mask[cell] = True
            frontier.append(cell)

    while int(mask.sum()) < target:
        if frontier and np.random.random() < 0.8:
            base_r, base_c = frontier[np.random.randint(len(frontier))]
            dr, dc = NEIGHBORS4[np.random.randint(4)]
            nr = base_r + dr
            nc = base_c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                mask[nr, nc] = True
                frontier.append((nr, nc))
                continue

        empties = np.argwhere(~mask)
        nr, nc = map(int, empties[np.random.randint(len(empties))])
        mask[nr, nc] = True
        frontier.append((nr, nc))

    interior = np.argwhere(mask[1:-1, 1:-1]) if h >= 3 and w >= 3 else np.empty((0, 2), dtype=int)
    if interior.size and np.random.random() < 0.6 and int(mask.sum()) > 4:
        rr, cc = map(int, interior[np.random.randint(len(interior))])
        mask[rr + 1, cc + 1] = False

    assert mask[0].any()
    assert mask[-1].any()
    assert mask[:, 0].any()
    assert mask[:, -1].any()
    return mask


def _make_pattern(h: int, w: int, bg: int, frame_color: int) -> np.ndarray:
    mask = _random_side_touching_mask(h, w)
    fg_coords = np.argwhere(mask)
    choices = [color for color in PALETTE if color not in {bg, frame_color}]
    np.random.shuffle(choices)

    max_colors = min(3, len(choices), len(fg_coords))
    n_colors = int(np.random.choice(np.arange(1, max_colors + 1), p=None))
    colors = choices[:n_colors]

    labels = np.random.randint(0, n_colors, size=len(fg_coords))
    for idx in range(n_colors):
        labels[idx % len(labels)] = idx
    np.random.shuffle(labels)

    pattern = np.full((h, w), bg, dtype=int)
    for (r, c), label in zip(fg_coords, labels):
        pattern[int(r), int(c)] = int(colors[int(label)])

    return pattern


def _rand_margin(max_extra: int) -> tuple[int, int]:
    total = 2 + np.random.randint(max_extra + 1)
    first = np.random.randint(1, total)
    second = total - first
    return int(first), int(second)


def _build_example() -> tuple[np.ndarray, np.ndarray, dict] | None:
    sx, sy = _sample_scale_pair()

    max_pattern_h = min(6, 30 // sy)
    max_pattern_w = min(6, 30 // sx)
    if max_pattern_h < 2 or max_pattern_w < 2:
        return None

    pattern_sizes = [(h, w) for h in range(2, max_pattern_h + 1) for w in range(2, max_pattern_w + 1) if h * w >= 5]
    if not pattern_sizes:
        return None
    pattern_h, pattern_w = pattern_sizes[np.random.randint(len(pattern_sizes))]

    bg = int(PALETTE[np.random.randint(len(PALETTE))])
    frame_choices = [color for color in PALETTE if color != bg]
    frame_color = int(frame_choices[np.random.randint(len(frame_choices))])
    pattern = _make_pattern(pattern_h, pattern_w, bg, frame_color)

    top_gap = np.random.randint(1, 4)
    bottom_gap = np.random.randint(1, 4)
    left_gap = np.random.randint(1, 4)
    right_gap = np.random.randint(1, 4)

    inner_h = pattern_h + top_gap + bottom_gap
    inner_w = pattern_w + left_gap + right_gap
    if inner_h == pattern_h * sy or inner_w == pattern_w * sx:
        return None

    outer_h = inner_h + 2 * sy
    outer_w = inner_w + 2 * sx
    if outer_h + 2 > 30 or outer_w + 2 > 30:
        return None

    top_margin, bottom_margin = _rand_margin(min(30 - outer_h - 2, 8))
    left_margin, right_margin = _rand_margin(min(30 - outer_w - 2, 8))

    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    fr0 = top_margin
    fc0 = left_margin
    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1

    grid[fr0 : fr0 + sy, fc0 : fc1 + 1] = frame_color
    grid[fr1 - sy + 1 : fr1 + 1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0 : fc0 + sx] = frame_color
    grid[fr0 : fr1 + 1, fc1 - sx + 1 : fc1 + 1] = frame_color

    pr0 = fr0 + sy + top_gap
    pc0 = fc0 + sx + left_gap
    grid[pr0 : pr0 + pattern_h, pc0 : pc0 + pattern_w] = pattern

    spec = {
        "bg": bg,
        "frame_color": frame_color,
        "sx": sx,
        "sy": sy,
        "pattern": pattern,
        "cavity_bbox": (fr0 + sy, fc0 + sx, fr1 - sy, fc1 - sx),
        "pattern_origin": (pr0, pc0),
    }
    return grid, _anisotropic_scale(pattern, sy, sx), spec


def _validate_example(grid: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if _most_common_color(grid) != spec["bg"]:
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    ir0, ic0, ir1, ic1 = spec["cavity_bbox"]
    cavity = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    if cavity.shape == spec["pattern"].shape:
        return False
    if cavity.shape[0] == expected.shape[0] or cavity.shape[1] == expected.shape[1]:
        return False

    wrong_cavity = _anisotropic_scale(cavity, spec["sy"], spec["sx"])
    if wrong_cavity.shape == expected.shape and np.array_equal(wrong_cavity, expected):
        return False

    wrong_swapped = _anisotropic_scale(spec["pattern"], spec["sx"], spec["sy"])
    if spec["sx"] != spec["sy"] and wrong_swapped.shape == expected.shape and np.array_equal(wrong_swapped, expected):
        return False

    if spec["sx"] == 1 and spec["sy"] == 1:
        return False

    bg_count = Counter(map(int, grid.ravel()))[spec["bg"]]
    if bg_count <= max(count for color, count in Counter(map(int, grid.ravel())).items() if color != spec["bg"]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_example()
        if built is None:
            continue
        grid, expected, spec = built
        if _validate_example(grid, expected, spec):
            return grid
    raise RuntimeError("failed to generate a valid thick-frame lens example")
