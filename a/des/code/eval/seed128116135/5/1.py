import random
from typing import Optional

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
BRANCHES = ("vertical", "horizontal", "both")
MAX_GENERATION_ATTEMPTS = 300
_GEN_COUNTER = 0


WIDE_TEMPLATES = (
    np.array([[1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=bool),
)
TALL_TEMPLATES = tuple(template.T for template in WIDE_TEMPLATES)
SQUARE_TEMPLATES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=bool),
)


def _mode_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_layout(branch: str, inner_h: int, inner_w: int, seed_h: int, seed_w: int):
    if branch == "vertical":
        row_starts = list(range(0, inner_h, seed_h))
        col_starts = [0]
    elif branch == "horizontal":
        row_starts = [0]
        col_starts = list(range(0, inner_w, seed_w))
    else:
        row_starts = list(range(0, inner_h, seed_h))
        col_starts = list(range(0, inner_w, seed_w))
    row_lens = [min(seed_h, inner_h - start) for start in row_starts]
    col_lens = [min(seed_w, inner_w - start) for start in col_starts]
    return row_starts, row_lens, col_starts, col_lens


def _index_for_offset(offset: int, starts: list[int], lens: list[int]) -> Optional[int]:
    for idx, (start, span) in enumerate(zip(starts, lens)):
        if start <= offset < start + span:
            return idx
    return None


def _render_interior(
    seed_mask: np.ndarray,
    branch: str,
    inner_h: int,
    inner_w: int,
    base_color: int,
    row_labels: dict[int, int],
    col_labels: dict[int, int],
) -> np.ndarray:
    seed_h, seed_w = map(int, seed_mask.shape)
    row_starts, row_lens, col_starts, col_lens = _tile_layout(branch, inner_h, inner_w, seed_h, seed_w)
    out = np.full((inner_h, inner_w), BG, dtype=int)
    for row_idx, row_start in enumerate(row_starts):
        row_span = row_lens[row_idx]
        for col_idx, col_start in enumerate(col_starts):
            col_span = col_lens[col_idx]
            color = row_labels.get(row_idx, col_labels.get(col_idx, base_color))
            patch = out[row_start : row_start + row_span, col_start : col_start + col_span]
            patch[seed_mask[:row_span, :col_span]] = color
    return out


def _extract_labels(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    border_color: int,
    branch: str,
    seed_h: int,
    seed_w: int,
) -> tuple[dict[int, int], dict[int, int]]:
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    row_starts, row_lens, col_starts, col_lens = _tile_layout(branch, inner_h, inner_w, seed_h, seed_w)

    row_labels: dict[int, int] = {}
    col_labels: dict[int, int] = {}

    def add_label(mapping: dict[int, int], idx: Optional[int], color: int):
        if idx is None:
            return
        if idx in mapping:
            assert mapping[idx] == color
        else:
            mapping[idx] = color

    top = [(0, c - c0 - 1, int(grid[r0, c])) for c in range(c0 + 1, c1) if int(grid[r0, c]) != border_color]
    bottom = [(0, c - c0 - 1, int(grid[r1, c])) for c in range(c0 + 1, c1) if int(grid[r1, c]) != border_color]
    left = [(r - r0 - 1, 0, int(grid[r, c0])) for r in range(r0 + 1, r1) if int(grid[r, c0]) != border_color]
    right = [(r - r0 - 1, 0, int(grid[r, c1])) for r in range(r0 + 1, r1) if int(grid[r, c1]) != border_color]

    assert len(top) <= 1 and len(bottom) <= 1 and len(left) <= 1 and len(right) <= 1

    for _, local_c, color in top + bottom:
        add_label(col_labels, _index_for_offset(local_c, col_starts, col_lens), color)
    for local_r, _, color in left + right:
        add_label(row_labels, _index_for_offset(local_r, row_starts, row_lens), color)

    return row_labels, col_labels


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    frames: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) == BG:
                continue
            if r0 > 0 and int(grid[r0 - 1, c0]) != BG:
                continue
            if c0 > 0 and int(grid[r0, c0 - 1]) != BG:
                continue

            c1 = c0
            while c1 + 1 < w and int(grid[r0, c1 + 1]) != BG:
                c1 += 1
            r1 = r0
            while r1 + 1 < h and int(grid[r1 + 1, c0]) != BG:
                r1 += 1

            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if not np.all(grid[r0, c0 : c1 + 1] != BG):
                continue
            if not np.all(grid[r1, c0 : c1 + 1] != BG):
                continue
            if not np.all(grid[r0 : r1 + 1, c0] != BG):
                continue
            if not np.all(grid[r0 : r1 + 1, c1] != BG):
                continue

            if r0 > 0 and np.any(grid[r0 - 1, c0 : c1 + 1] != BG):
                continue
            if c0 > 0 and np.any(grid[r0 : r1 + 1, c0 - 1] != BG):
                continue
            if r1 + 1 < h and np.any(grid[r1 + 1, c0 : c1 + 1] != BG):
                continue
            if c1 + 1 < w and np.any(grid[r0 : r1 + 1, c1 + 1] != BG):
                continue

            rect = (r0, c0, r1, c1)
            if rect not in seen:
                seen.add(rect)
                frames.append(rect)
    assert frames
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        border_values = np.concatenate(
            (
                g[r0, c0 : c1 + 1],
                g[r1, c0 : c1 + 1],
                g[r0 + 1 : r1, c0],
                g[r0 + 1 : r1, c1],
            )
        )
        border_color = _mode_color(border_values)
        assert border_color != BG

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask_full = inner != BG
        box = _bbox(seed_mask_full)
        assert box is not None
        sr0, sc0, sr1, sc1 = box
        assert sr0 == 0 and sc0 == 0

        seed = inner[: sr1 + 1, : sc1 + 1]
        seed_colors = np.unique(seed[seed != BG])
        assert seed_colors.size == 1
        seed_color = int(seed_colors[0])
        assert seed_color != border_color

        assert np.all((seed == BG) | (seed == seed_color))
        assert 2 <= int(np.count_nonzero(seed == seed_color)) <= 4

        seed_mask = seed == seed_color
        seed_h, seed_w = seed_mask.shape

        if seed_w > seed_h:
            branch = "vertical"
        elif seed_h > seed_w:
            branch = "horizontal"
        else:
            branch = "both"

        row_labels, col_labels = _extract_labels(g, r0, c0, r1, c1, border_color, branch, seed_h, seed_w)
        filled = _render_interior(seed_mask, branch, inner.shape[0], inner.shape[1], seed_color, row_labels, col_labels)
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _transform_mask(mask: np.ndarray, branch: str, rng: random.Random) -> np.ndarray:
    result = mask.copy()
    if branch == "both":
        for _ in range(rng.randrange(4)):
            result = np.rot90(result)
    else:
        if rng.random() < 0.5:
            result = np.flipud(result)
        if rng.random() < 0.5:
            result = np.fliplr(result)
    box = _bbox(result)
    assert box is not None
    r0, c0, r1, c1 = box
    return np.asarray(result[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _choose_seed(branch: str, rng: random.Random) -> np.ndarray:
    if branch == "vertical":
        template = rng.choice(WIDE_TEMPLATES)
    elif branch == "horizontal":
        template = rng.choice(TALL_TEMPLATES)
    else:
        template = rng.choice(SQUARE_TEMPLATES)
    return _transform_mask(template, branch, rng)


def _branch_dimensions(branch: str, seed_h: int, seed_w: int, rng: random.Random) -> tuple[int, int]:
    if branch == "vertical":
        inner_h = seed_h * rng.randint(2, 4) + rng.randint(0, seed_h - 1)
        inner_w = seed_w + rng.randint(0, 4)
        if inner_w == seed_w:
            inner_w += 1
    elif branch == "horizontal":
        inner_h = seed_h + rng.randint(0, 4)
        if inner_h == seed_h:
            inner_h += 1
        inner_w = seed_w * rng.randint(2, 4) + rng.randint(0, seed_w - 1)
    else:
        inner_h = seed_h * rng.randint(2, 4) + rng.randint(0, seed_h - 1)
        inner_w = seed_w * rng.randint(2, 4) + rng.randint(0, seed_w - 1)
    return inner_h, inner_w


def _pick_label_specs(
    branch: str,
    row_count: int,
    col_count: int,
    rng: random.Random,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    row_specs: list[tuple[str, int]] = []
    col_specs: list[tuple[str, int]] = []

    row_sides = ["left", "right"]
    col_sides = ["top", "bottom"]
    rng.shuffle(row_sides)
    rng.shuffle(col_sides)

    if branch == "vertical":
        row_n = min(len(row_sides), row_count, rng.randint(1, min(2, row_count)))
        col_n = 1 if rng.random() < 0.6 else 0
    elif branch == "horizontal":
        col_n = min(len(col_sides), col_count, rng.randint(1, min(2, col_count)))
        row_n = 1 if rng.random() < 0.6 else 0
    else:
        row_n = min(len(row_sides), row_count, rng.randint(1, min(2, row_count)))
        col_n = min(len(col_sides), col_count, rng.randint(1, min(2, col_count)))

    row_indices = rng.sample(range(row_count), row_n) if row_n else []
    col_indices = rng.sample(range(col_count), col_n) if col_n else []
    for side, idx in zip(row_sides, row_indices):
        row_specs.append((side, idx))
    for side, idx in zip(col_sides, col_indices):
        col_specs.append((side, idx))
    return row_specs, col_specs


def _make_frame_patch(branch: str, rng: random.Random) -> np.ndarray:
    seed_mask = _choose_seed(branch, rng)
    seed_h, seed_w = map(int, seed_mask.shape)
    inner_h, inner_w = _branch_dimensions(branch, seed_h, seed_w, rng)
    row_starts, row_lens, col_starts, col_lens = _tile_layout(branch, inner_h, inner_w, seed_h, seed_w)

    row_specs, col_specs = _pick_label_specs(branch, len(row_starts), len(col_starts), rng)
    total_labels = len(row_specs) + len(col_specs)

    colors = rng.sample(PALETTE, 2 + total_labels)
    border_color = colors[0]
    seed_color = colors[1]
    label_colors = colors[2:]

    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1 : 1 + seed_h, 1 : 1 + seed_w][seed_mask] = seed_color

    label_iter = iter(label_colors)
    for side, row_idx in row_specs:
        local_r = row_starts[row_idx] + rng.randrange(row_lens[row_idx])
        color = next(label_iter)
        if side == "left":
            patch[1 + local_r, 0] = color
        else:
            patch[1 + local_r, -1] = color
    for side, col_idx in col_specs:
        local_c = col_starts[col_idx] + rng.randrange(col_lens[col_idx])
        color = next(label_iter)
        if side == "top":
            patch[0, 1 + local_c] = color
        else:
            patch[-1, 1 + local_c] = color

    return patch


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    spots = [(r, c) for r in range(gh - ph + 1) for c in range(gw - pw + 1)]
    rng.shuffle(spots)
    for r, c in spots:
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(gh, r + ph + 1)
        c1 = min(gw, c + pw + 1)
        if occ[r0:r1, c0:c1].any():
            continue
        mask = patch != BG
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        occ[r0:r1, c0:c1] = True
        return True
    return False


def _gen_attempt(primary_branch: str) -> Optional[np.ndarray]:
    rng = random.Random()
    grid_h = rng.randint(20, 30)
    grid_w = rng.randint(20, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    frame_count = rng.randint(1, 3)
    branches = [primary_branch] + [rng.choice(BRANCHES) for _ in range(frame_count - 1)]
    rng.shuffle(branches)

    patches = [_make_frame_patch(branch, rng) for branch in branches]
    patches.sort(key=lambda item: item.shape[0] * item.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(grid, occ, patch, rng):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    primary_branch = BRANCHES[_GEN_COUNTER % len(BRANCHES)]
    _GEN_COUNTER += 1
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(primary_branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
