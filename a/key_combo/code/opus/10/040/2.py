from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

MAJORITY_COLORS = (BLUE, RED, GREEN, YELLOW, PINK)
DISTRACTOR_SIZES = (
    (3, 2),
    (2, 3),
    (3, 4),
    (4, 3),
    (2, 2),
    (4, 4),
    (2, 5),
    (5, 2),
)
PROBE_NAMES = ("top_left", "center", "bottom_right")
PROBE_INDEX = {
    "top_left": lambda h, w: (0, 0),
    "center": lambda h, w: (h // 2, w // 2),
    "bottom_right": lambda h, w: (h - 1, w - 1),
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def find_components(
    grid: np.ndarray,
    *,
    color: int,
    connectivity: int = 4,
) -> list[dict[str, object]]:
    comps = []
    for coords in components_mask(grid == int(color), connectivity=connectivity):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _unique_majority(block: np.ndarray) -> int:
    counts = Counter(map(int, block.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty block"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _extract_lattice(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    for comp in find_components(grid, color=ORANGE, connectivity=4):
        box = comp["bbox"]
        patch = crop_bbox(grid, box)
        row_lines = [idx for idx in range(patch.shape[0]) if np.all(patch[idx, :] == ORANGE)]
        col_lines = [idx for idx in range(patch.shape[1]) if np.all(patch[:, idx] == ORANGE)]
        if len(row_lines) != 4 or len(col_lines) != 4:
            continue
        if any(b - a <= 1 for a, b in zip(row_lines, row_lines[1:])):
            continue
        if any(b - a <= 1 for a, b in zip(col_lines, col_lines[1:])):
            continue
        row_sizes = [b - a - 1 for a, b in zip(row_lines, row_lines[1:])]
        col_sizes = [b - a - 1 for a, b in zip(col_lines, col_lines[1:])]
        if len(set(row_sizes)) != 1 or len(set(col_sizes)) != 1:
            continue

        expected = np.zeros_like(patch, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        if not np.array_equal(patch == ORANGE, expected):
            continue

        row_bands = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(3)]
        col_bands = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(3)]
        candidates.append((box, row_bands, col_bands))

    assert len(candidates) == 1, "expected exactly one 3x3 orange lattice"
    return candidates[0]


def _compressed_template(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> np.ndarray:
    r_box, c_box = box[0], box[1]
    template = np.zeros((3, 3), dtype=int)
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            block = grid[r_box + r0 : r_box + r1 + 1, c_box + c0 : c_box + c1 + 1]
            assert not np.any((block == BLACK) | (block == GREY) | (block == ORANGE))
            template[rr, cc] = _unique_majority(block)
    return template


def _find_target_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    matches = []
    grey_rectangles = find_components(grid, color=GREY, connectivity=4)
    assert len(grey_rectangles) >= 2, "expected multiple grey rectangles"
    for comp in grey_rectangles:
        box = comp["bbox"]
        r0, c0, r1, c1 = box
        h = bbox_h(box)
        w = bbox_w(box)
        if comp["area"] != h * w:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            continue
        if (h, w) == (3, 3):
            matches.append(box)
    assert len(matches) == 1, "expected exactly one 3x3 grey target"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a noisy 3x3 color lattice to its majority-color template, then stamp that template into the unique matching gray target.

    Input structure:
    - A black background contains one orange separator lattice whose full border
      and internal lines partition a rectangle into a 3x3 grid of equal blocks.
    - Each block is filled with non-black, non-gray, non-orange colors and has a
      unique majority color despite a few noisy pixels.
    - Elsewhere there are multiple solid gray rectangles, and exactly one of them
      is a 3x3 square.

    Rule:
    - Detect the orange lattice and recover the 3x3 block layout from its full
      separator rows and columns.
    - Replace each block by its unique majority color to obtain a 3x3 template.
    - Find the unique solid gray 3x3 rectangle and overwrite it cell-for-cell
      with the recovered template.
    - Leave the lattice and every non-matching gray rectangle unchanged.

    Key insight:
    - The stamped pattern is not directly visible in the input; it must be
      synthesized by compressing each noisy lattice cell to its majority color,
      and only the gray rectangle with exactly matching size accepts that stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    lattice_box, row_bands, col_bands = _extract_lattice(g)
    template = _compressed_template(g, lattice_box, row_bands, col_bands)
    target = _find_target_rect(g)

    out = g.copy()
    r0, c0, r1, c1 = target
    out[r0 : r1 + 1, c0 : c1 + 1] = template
    return out


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_box(
    h: int,
    w: int,
    existing: list[tuple[int, int, int, int]],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    *,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    r_min, r_max = row_range
    c_min, c_max = col_range
    if r_max < r_min or c_max < c_min:
        return None
    candidates = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_clear(box, prev, pad=1) for prev in existing):
                candidates.append(box)
    if not candidates:
        return None
    return rng.choice(candidates)


def _make_template(rng: random.Random) -> np.ndarray:
    while True:
        template = np.array(
            [rng.choice(MAJORITY_COLORS) for _ in range(9)],
            dtype=int,
        ).reshape(3, 3)
        distinct = len(set(map(int, template.reshape(-1))))
        if distinct < 3:
            continue
        if np.array_equal(template, np.fliplr(template)) or np.array_equal(template, np.flipud(template)):
            continue
        if len({tuple(row) for row in template}) < 2:
            continue
        if len({tuple(col) for col in template.T}) < 2:
            continue
        return template


def _noise_budget(block_h: int, block_w: int, rng: random.Random) -> int:
    cap = min(3, (block_h * block_w - 1) // 2)
    return rng.randint(1, cap) if cap > 0 else 0


def _fill_block(
    major: int,
    block_h: int,
    block_w: int,
    *,
    rng: random.Random,
    forced_noise: tuple[int, int] | None = None,
    probe_hits: set[str],
) -> np.ndarray:
    block = np.full((block_h, block_w), int(major), dtype=int)
    noise_count = _noise_budget(block_h, block_w, rng)
    all_cells = [(r, c) for r in range(block_h) for c in range(block_w)]
    chosen: list[tuple[int, int]] = []
    if forced_noise is not None:
        chosen.append(forced_noise)
    remaining = [cell for cell in all_cells if cell not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, noise_count - len(chosen))])

    alt_palette = [color for color in MAJORITY_COLORS if color != major]
    alt_count = 1 if len(chosen) == 1 else rng.randint(1, min(2, len(alt_palette)))
    alt_colors = rng.sample(alt_palette, alt_count)
    for idx, (r, c) in enumerate(chosen):
        block[r, c] = alt_colors[idx % alt_count]

    for name, picker in PROBE_INDEX.items():
        pr, pc = picker(block_h, block_w)
        if int(block[pr, pc]) != int(major):
            probe_hits.add(name)
    return block


def _build_lattice(
    template: np.ndarray,
    block_h: int,
    block_w: int,
    *,
    rng: random.Random,
) -> tuple[np.ndarray, set[str]]:
    h = 3 * block_h + 4
    w = 3 * block_w + 4
    lattice = np.full((h, w), ORANGE, dtype=int)
    probe_hits: set[str] = set()
    probe_cells = {
        "top_left": (0, 0),
        "center": (1, 1),
        "bottom_right": (2, 2),
    }
    for tr in range(3):
        for tc in range(3):
            r0 = 1 + tr * (block_h + 1)
            c0 = 1 + tc * (block_w + 1)
            forced_noise = None
            for name, cell in probe_cells.items():
                if (tr, tc) == cell:
                    forced_noise = PROBE_INDEX[name](block_h, block_w)
                    break
            block = _fill_block(
                int(template[tr, tc]),
                block_h,
                block_w,
                rng=rng,
                forced_noise=forced_noise,
                probe_hits=probe_hits,
            )
            lattice[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return lattice, probe_hits


def _make_rectangles(rng: random.Random) -> list[tuple[int, int]]:
    total = rng.randint(2, 4)
    distractor_count = total - 1
    sizes = [rng.choice(((3, 2), (2, 3), (3, 4), (4, 3)))]
    pool = [size for size in DISTRACTOR_SIZES if size != sizes[0]]
    rng.shuffle(pool)
    sizes.extend(pool[: max(0, distractor_count - 1)])
    return [(3, 3), *sizes]


def _render_case(
    height: int,
    width: int,
    lattice: np.ndarray,
    rect_sizes: list[tuple[int, int]],
    *,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((height, width), BLACK, dtype=int)
    existing: list[tuple[int, int, int, int]] = []
    max_rect_w = max(w for _, w in rect_sizes)
    lattice_c_max = width - lattice.shape[1] - max_rect_w - 5
    if lattice_c_max < 1:
        return None

    lattice_box = _place_box(
        lattice.shape[0],
        lattice.shape[1],
        existing,
        (1, height - lattice.shape[0] - 1),
        (1, lattice_c_max),
        rng=rng,
    )
    if lattice_box is None:
        return None
    existing.append(lattice_box)
    lr0, lc0, lr1, lc1 = lattice_box
    grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = lattice

    placements = []
    for idx, (rh, rw) in enumerate(rect_sizes):
        right_start = lc1 + 3
        if right_start > width - rw - 1:
            return None
        box = _place_box(
            rh,
            rw,
            existing,
            (1, height - rh - 1),
            (right_start, width - rw - 1),
            rng=rng,
        )
        if box is None:
            return None
        existing.append(box)
        placements.append((idx, box))

    out = grid.copy()
    for idx, box in placements:
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        if idx == 0:
            out[r0 : r1 + 1, c0 : c1 + 1] = _majority_from_lattice(lattice)
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid, out


def _majority_from_lattice(lattice: np.ndarray) -> np.ndarray:
    block_h = (lattice.shape[0] - 4) // 3
    block_w = (lattice.shape[1] - 4) // 3
    template = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            r0 = 1 + r * (block_h + 1)
            c0 = 1 + c * (block_w + 1)
            template[r, c] = _unique_majority(lattice[r0 : r0 + block_h, c0 : c0 + block_w])
    return template


def _validate_semantics(inp: np.ndarray, out: np.ndarray, probe_hits: set[str]) -> bool:
    if probe_hits != set(PROBE_NAMES):
        return False

    try:
        lattice_box, row_bands, col_bands = _extract_lattice(inp)
        template = _compressed_template(inp, lattice_box, row_bands, col_bands)
        target = _find_target_rect(inp)
        pred = main(inp)
    except Exception:
        return False

    if not np.array_equal(pred, out):
        return False

    r0, c0, r1, c1 = target
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], template):
        return False

    grey_rects = find_components(inp, color=GREY, connectivity=4)
    near_match_seen = False
    target_count = 0
    for comp in grey_rects:
        box = comp["bbox"]
        h = bbox_h(box)
        w = bbox_w(box)
        rr0, cc0, rr1, cc1 = box
        if comp["area"] != h * w:
            return False
        if not np.all(inp[rr0 : rr1 + 1, cc0 : cc1 + 1] == GREY):
            return False
        if (h, w) == (3, 3):
            target_count += 1
        elif h == 3 or w == 3:
            near_match_seen = True
            if not np.array_equal(out[rr0 : rr1 + 1, cc0 : cc1 + 1], inp[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                return False
        else:
            if not np.array_equal(out[rr0 : rr1 + 1, cc0 : cc1 + 1], inp[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                return False

    if target_count != 1 or not near_match_seen:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template = _make_template(rng)
    block_h = rng.choice((3, 4))
    block_w = rng.choice((3, 4))
    lattice, probe_hits = _build_lattice(template, block_h, block_w, rng=rng)
    rect_sizes = _make_rectangles(rng)

    min_h = max(lattice.shape[0] + 2, max(h for h, _ in rect_sizes) + 4, 20)
    min_w = max(lattice.shape[1] + max(w for _, w in rect_sizes) + 7, 24)
    if min_h > 30 or min_w > 30:
        return None

    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    rendered = _render_case(height, width, lattice, rect_sizes, rng=rng)
    if rendered is None:
        return None
    inp, out = rendered

    if get_bg(inp) != BLACK:
        return None
    if not _validate_semantics(inp, out, probe_hits):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 2")
