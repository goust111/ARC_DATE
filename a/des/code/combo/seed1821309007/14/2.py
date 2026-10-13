import random
from collections import Counter, deque

import numpy as np


BG = 8
SEP = 5
OBJECT_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
ADDRESSES = [(0, 0), (0, 1), (1, 0), (1, 1)]
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = []
    for r in range(h):
        row = grid[r, :]
        if np.all(row == row[0]) and int(row[0]) != bg:
            row_candidates.append((r, int(row[0])))

    col_candidates = []
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            col_candidates.append((c, int(col[0])))

    for r, row_color in row_candidates:
        if r == 0 or r == h - 1:
            continue
        for c, col_color in col_candidates:
            if c == 0 or c == w - 1 or row_color != col_color:
                continue
            return r, c, row_color
    raise AssertionError("expected one full separator row and column")


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_object(quad: np.ndarray, bg: int) -> dict:
    colors = [int(v) for v in np.unique(quad) if int(v) != bg]
    assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
    color = colors[0]
    mask = quad == color
    assert _component_count(mask) == 1, "quadrant foreground must be one 4-connected object"

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    local = coords - np.array([r0, c0])
    patch[local[:, 0], local[:, 1]] = color
    return {
        "color": color,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
    }


def _target_address(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    qh, qw = shape
    obj_r = (r0 + r1) / 2.0
    obj_c = (c0 + c1) / 2.0
    ctr_r = (qh - 1) / 2.0
    ctr_c = (qw - 1) / 2.0
    assert obj_r != ctr_r and obj_c != ctr_c, "object bbox center must be offset from quadrant center"
    return (0 if obj_r < ctr_r else 1, 0 if obj_c < ctr_c else 1)


def _paste(dest: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    region = dest[top : top + h, left : left + w]
    mask = patch != bg
    assert np.all(region[mask] == bg), "moved objects must not overlap"
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    sep_r, sep_c, sep_color = _find_separator(g, bg)
    quadrants = {
        (0, 0): g[:sep_r, :sep_c],
        (0, 1): g[:sep_r, sep_c + 1 :],
        (1, 0): g[sep_r + 1 :, :sep_c],
        (1, 1): g[sep_r + 1 :, sep_c + 1 :],
    }

    out = np.full_like(g, bg)
    out[sep_r, :] = sep_color
    out[:, sep_c] = sep_color

    objects = []
    for source_addr, quad in quadrants.items():
        obj = _extract_object(quad, bg)
        target_addr = _target_address(obj["bbox"], quad.shape)
        obj["source_addr"] = source_addr
        obj["target_addr"] = target_addr
        objects.append(obj)

    target_seen = Counter(obj["target_addr"] for obj in objects)
    assert set(target_seen) == set(ADDRESSES), "expected one object per target quadrant"
    assert all(v == 1 for v in target_seen.values()), "target quadrants must be unique"

    target_origins = {
        (0, 0): (0, 0),
        (0, 1): (0, sep_c + 1),
        (1, 0): (sep_r + 1, 0),
        (1, 1): (sep_r + 1, sep_c + 1),
    }
    target_shapes = {addr: quadrants[addr].shape for addr in quadrants}

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = obj["patch"]
        dest_addr = obj["target_addr"]
        qh, qw = target_shapes[dest_addr]
        assert r1 < qh and c1 < qw, "same relative placement must fit in target quadrant"
        base_r, base_c = target_origins[dest_addr]
        _paste(out, patch, base_r + r0, base_c + c0, bg)

    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r_shift = min(rs)
    c_shift = min(cs)
    norm = {(r - r_shift, c - c_shift) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _make_shape(max_h: int, max_w: int) -> np.ndarray:
    area_limit = min(max_h * max_w, 8)
    area = random.randint(2, area_limit)
    for _ in range(50):
        h = random.randint(1, min(max_h, area))
        w = random.randint(1, min(max_w, area))
        if h * w < area:
            continue
        mask = _random_connected_mask(h, w, area)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask
    return np.ones((1, 2), dtype=bool)


def _valid_positions(limit: int, span: int, bit: int) -> list[int]:
    positions = []
    center = (limit - 1) / 2.0
    for start in range(limit - span + 1):
        obj_center = start + (span - 1) / 2.0
        if bit == 0 and obj_center < center:
            positions.append(start)
        if bit == 1 and obj_center > center:
            positions.append(start)
    return positions


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(4, 8)
    qw = random.randint(4, 8)
    h, w = 2 * qh + 1, 2 * qw + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[qh, :] = SEP
    grid[:, qw] = SEP

    colors = random.sample(OBJECT_COLORS, 4)
    targets = random.sample(ADDRESSES, 4)
    quad_origins = {
        (0, 0): (0, 0),
        (0, 1): (0, qw + 1),
        (1, 0): (qh + 1, 0),
        (1, 1): (qh + 1, qw + 1),
    }

    for source_addr, color, target_addr in zip(ADDRESSES, colors, targets):
        mask = _make_shape(qh - 1, qw - 1)
        mh, mw = mask.shape
        row_positions = _valid_positions(qh, mh, target_addr[0])
        col_positions = _valid_positions(qw, mw, target_addr[1])
        if not row_positions or not col_positions:
            return None
        top = random.choice(row_positions)
        left = random.choice(col_positions)
        base_r, base_c = quad_origins[source_addr]
        patch = grid[base_r + top : base_r + top + mh, base_c + left : base_c + left + mw]
        patch[mask] = color

    try:
        objects = []
        for addr in ADDRESSES:
            base_r, base_c = quad_origins[addr]
            quad = grid[base_r : base_r + qh, base_c : base_c + qw]
            obj = _extract_object(quad, BG)
            obj["target_addr"] = _target_address(obj["bbox"], quad.shape)
            objects.append(obj)
        counts = Counter(obj["target_addr"] for obj in objects)
        if set(counts) != set(ADDRESSES) or any(v != 1 for v in counts.values()):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
