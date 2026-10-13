import itertools
import random

import numpy as np


BG = 0
GREY = 5
FRAME_SIZE = 5
INTERIOR_SIZE = 3
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
FRAME_RING_MASK = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
FRAME_RING_MASK[1:-1, 1:-1] = False
UP_TEMPLATE = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ],
    dtype=bool,
)
TEMPLATES = {
    "up": UP_TEMPLATE,
    "left": np.rot90(UP_TEMPLATE, 1),
    "down": np.rot90(UP_TEMPLATE, 2),
    "right": np.rot90(UP_TEMPLATE, 3),
}
GEN_RETRIES = 400
GEN_GRID_MIN = 26
GEN_GRID_MAX = 30
GEN_EXTRA_FRAME_OPTIONS = (0, 1)
GEN_PAD = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR_VECTORS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask)) == 1


def _template_direction(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (3, 3), "template must be 3x3"
    assert int(mask.sum()) == 4, "template must have four colored cells"
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    if int(row_counts.max()) == 3:
        bar_row = int(np.flatnonzero(row_counts == 3)[0])
        extra = np.argwhere(mask & (np.arange(3)[:, None] != bar_row))
        assert extra.shape == (1, 2)
        return "up" if int(extra[0, 0]) < bar_row else "down"
    if int(col_counts.max()) == 3:
        bar_col = int(np.flatnonzero(col_counts == 3)[0])
        extra = np.argwhere(mask & (np.arange(3)[None, :] != bar_col))
        assert extra.shape == (1, 2)
        return "left" if int(extra[0, 1]) < bar_col else "right"
    raise AssertionError("template must be an oriented T shape")


def _fragment_masks(template: np.ndarray) -> list[np.ndarray]:
    cells = [tuple(cell) for cell in np.argwhere(template)]
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for size in (2, 3, 4):
        for subset in itertools.combinations(cells, size):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in subset:
                mask[r, c] = True
            if not _is_connected(mask):
                continue
            key = mask.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(mask)
    return out


FRAGMENTS = {direction: _fragment_masks(template) for direction, template in TEMPLATES.items()}


def _parse_legend_block(block: np.ndarray, bg: int) -> dict[int, np.ndarray] | None:
    h, w = block.shape
    entries: list[np.ndarray] = []
    if h == 5 and w >= 9 and (w - 1) % 4 == 0:
        n = (w - 1) // 4
        sep_cols = list(range(0, w, 4))
        if sep_cols[-1] != w - 1:
            return None
        if not np.all(block[0, :] == GREY) or not np.all(block[-1, :] == GREY):
            return None
        if not np.all(block[:, sep_cols] == GREY):
            return None
        non_sep_cols = [c for c in range(w) if c not in sep_cols]
        if np.any(block[1:-1, non_sep_cols] == GREY):
            return None
        for i in range(n):
            entries.append(block[1:4, 1 + 4 * i : 4 + 4 * i])
    elif w == 5 and h >= 9 and (h - 1) % 4 == 0:
        n = (h - 1) // 4
        sep_rows = list(range(0, h, 4))
        if sep_rows[-1] != h - 1:
            return None
        if not np.all(block[:, 0] == GREY) or not np.all(block[:, -1] == GREY):
            return None
        if not np.all(block[sep_rows, :] == GREY):
            return None
        non_sep_rows = [r for r in range(h) if r not in sep_rows]
        if np.any(block[non_sep_rows, 1:-1] == GREY):
            return None
        for i in range(n):
            entries.append(block[1 + 4 * i : 4 + 4 * i, 1:4])
    else:
        return None

    mapping: dict[int, np.ndarray] = {}
    for cell in entries:
        colors = [int(color) for color in np.unique(cell) if int(color) not in (bg, GREY)]
        if len(colors) != 1:
            return None
        color = colors[0]
        mask = cell == color
        if int(mask.sum()) != 4 or not _is_connected(mask):
            return None
        _template_direction(mask)
        if color in mapping:
            return None
        mapping[color] = mask
    return mapping if len(mapping) >= 2 else None


def _extract_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    bg = _background_color(grid)
    grey_components = _components(grid == GREY)
    h, w = grid.shape
    candidates: list[tuple[int, dict[int, np.ndarray]]] = []
    for idx, coords in enumerate(grey_components):
        r0, c0, r1, c1 = _bbox(coords)
        touches_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if not touches_corner:
            continue
        mapping = _parse_legend_block(grid[r0 : r1 + 1, c0 : c1 + 1], bg)
        if mapping is not None:
            candidates.append((idx, mapping))
    assert len(candidates) == 1, "expected exactly one legend block"
    return candidates[0][1], candidates[0][0]


def _frame_records(grid: np.ndarray, legend_index: int, legend_colors: set[int]) -> list[dict[str, object]]:
    bg = _background_color(grid)
    records: list[dict[str, object]] = []
    for idx, coords in enumerate(_components(grid == GREY)):
        if idx == legend_index:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if block.shape != (FRAME_SIZE, FRAME_SIZE):
            continue
        if not np.array_equal(block == GREY, FRAME_RING_MASK):
            continue
        interior = block[1:-1, 1:-1]
        colors = [int(color) for color in np.unique(interior) if int(color) not in (bg, GREY)]
        assert len(colors) == 1, "each frame interior must contain exactly one fragment color"
        color = colors[0]
        assert color in legend_colors, "fragment color must appear in legend"
        fragment = interior == color
        count = int(fragment.sum())
        assert 2 <= count <= 4, "fragment size must encode stripe length"
        assert _is_connected(fragment), "fragment must be connected"
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "fragment": fragment,
                "count": count,
            }
        )
    assert records, "expected at least one frame"
    return records


def _stripe_slice(bbox: tuple[int, int, int, int], direction: str, length: int) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return slice(r0 - length, r0), slice(c0, c1 + 1)
    if direction == "down":
        return slice(r1 + 1, r1 + 1 + length), slice(c0, c1 + 1)
    if direction == "left":
        return slice(r0, r1 + 1), slice(c0 - length, c0)
    if direction == "right":
        return slice(r0, r1 + 1), slice(c1 + 1, c1 + 1 + length)
    raise AssertionError("unknown direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed fragment to its legend template and grow an exterior stripe in the template's opening direction.

    Input structure:
    - The background is black.
    - One corner contains a grey-bordered legend block made of separated 3x3 template cells.
    - Each legend cell contains exactly one connected non-grey template color forming a full oriented shape.
    - Elsewhere, several 5x5 grey hollow square frames each contain one connected same-color fragment inside the visible 3x3 interior.

    Rule:
    - Parse the legend and map each template color to its full 3x3 shape.
    - For every grey frame, identify the fragment color, verify that the fragment is a subset of that color's legend template, and count the visible fragment pixels.
    - Read the legend template's opening direction from its geometry.
    - Grow a solid stripe of the fragment color outside the corresponding side of the frame; the stripe has width 5 and length equal to the fragment pixel count.
    - Keep the legend block and the grey frames unchanged.

    Key insight:
    - The interior fragment does not determine the growth direction by itself; its color selects a full legend template, and that template's orientation provides the hidden direction while the fragment size provides the stripe length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    assert bg == BG, "background must be black"

    legend, legend_index = _extract_legend(g)
    out = g.copy()

    for record in _frame_records(g, legend_index, set(legend)):
        color = int(record["color"])
        bbox = record["bbox"]
        fragment = np.asarray(record["fragment"], dtype=bool)
        template = legend[color]
        assert np.all(fragment <= template), "fragment must be a visible subset of the legend template"
        direction = _template_direction(template)
        rs, cs = _stripe_slice(bbox, direction, int(record["count"]))
        assert rs.start is not None and cs.start is not None
        assert 0 <= rs.start <= rs.stop <= g.shape[0]
        assert 0 <= cs.start <= cs.stop <= g.shape[1]
        target = out[rs, cs]
        assert np.all(target == bg), "stripe area must be empty background"
        out[rs, cs] = color

    return out


def _legend_dims(horizontal: bool, count: int) -> tuple[int, int]:
    return (5, 4 * count + 1) if horizontal else (4 * count + 1, 5)


def _legend_origin(grid_shape: tuple[int, int], horizontal: bool, corner: str, count: int) -> tuple[int, int]:
    h, w = grid_shape
    lh, lw = _legend_dims(horizontal, count)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - lw
    if corner == "bl":
        return h - lh, 0
    if corner == "br":
        return h - lh, w - lw
    raise AssertionError("unknown corner")


def _render_legend(grid: np.ndarray, entries: list[tuple[int, str]], horizontal: bool, corner: str) -> tuple[int, int, int, int]:
    count = len(entries)
    r0, c0 = _legend_origin(grid.shape, horizontal, corner, count)
    lh, lw = _legend_dims(horizontal, count)
    grid[r0 : r0 + lh, c0 : c0 + lw] = GREY
    if horizontal:
        for i, (color, direction) in enumerate(entries):
            rr, cc = r0 + 1, c0 + 1 + 4 * i
            cell = grid[rr : rr + 3, cc : cc + 3]
            cell[:] = BG
            cell[TEMPLATES[direction]] = color
    else:
        for i, (color, direction) in enumerate(entries):
            rr, cc = r0 + 1 + 4 * i, c0 + 1
            cell = grid[rr : rr + 3, cc : cc + 3]
            cell[:] = BG
            cell[TEMPLATES[direction]] = color
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _union_rect(r0: int, c0: int, direction: str, length: int) -> tuple[int, int, int, int]:
    if direction == "up":
        return r0 - length, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1
    if direction == "down":
        return r0, c0, r0 + FRAME_SIZE - 1 + length, c0 + FRAME_SIZE - 1
    if direction == "left":
        return r0, c0 - length, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1
    if direction == "right":
        return r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1 + length
    raise AssertionError("unknown direction")


def _padded_rect(rect: tuple[int, int, int, int], shape: tuple[int, int], pad: int = GEN_PAD) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _frame_candidates(blocked: np.ndarray, direction: str, length: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    max_r = h - FRAME_SIZE
    max_c = w - FRAME_SIZE
    for r0 in range(max_r + 1):
        for c0 in range(max_c + 1):
            rect = _union_rect(r0, c0, direction, length)
            ur0, uc0, ur1, uc1 = rect
            if ur0 < 0 or uc0 < 0 or ur1 >= h or uc1 >= w:
                continue
            pr0, pc0, pr1, pc1 = _padded_rect(rect, (h, w))
            if blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _draw_frame(grid: np.ndarray, r0: int, c0: int, color: int, fragment: np.ndarray) -> None:
    block = grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
    block[:] = BG
    block[FRAME_RING_MASK] = GREY
    interior = block[1:-1, 1:-1]
    interior[fragment] = color


def _job_record(color: int, direction: str) -> dict[str, object]:
    fragment = random.choice(FRAGMENTS[direction]).copy()
    return {
        "color": color,
        "direction": direction,
        "fragment": fragment,
        "length": int(fragment.sum()),
    }


def _validate_generated_grid(grid: np.ndarray, expected_frames: int) -> None:
    legend, legend_index = _extract_legend(grid)
    records = _frame_records(grid, legend_index, set(legend))
    assert len(records) == expected_frames
    directions_seen = {_template_direction(legend[int(record["color"])]) for record in records}
    assert directions_seen == set(DIRECTIONS)
    out = main(grid)
    assert out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    color_assignment = random.sample(COLOR_POOL, len(DIRECTIONS))
    entries = list(zip(color_assignment, DIRECTIONS))
    random.shuffle(entries)
    horizontal = random.choice((True, False))
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_rect = _render_legend(grid, entries, horizontal, corner)
    lr0, lc0, lr1, lc1 = _padded_rect(legend_rect, grid.shape)
    blocked[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    color_to_direction = {color: direction for color, direction in entries}
    jobs = [_job_record(color, color_to_direction[color]) for color, _direction in entries]
    for _ in range(random.choice(GEN_EXTRA_FRAME_OPTIONS)):
        extra_color = random.choice(color_assignment)
        jobs.append(_job_record(extra_color, color_to_direction[extra_color]))
    jobs.sort(key=lambda job: int(job["length"]), reverse=True)

    for job in jobs:
        candidates = _frame_candidates(blocked, str(job["direction"]), int(job["length"]))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _draw_frame(grid, r0, c0, int(job["color"]), np.asarray(job["fragment"], dtype=bool))
        pr0, pc0, pr1, pc1 = _padded_rect(_union_rect(r0, c0, str(job["direction"]), int(job["length"])), grid.shape)
        blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    _validate_generated_grid(grid, len(jobs))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
