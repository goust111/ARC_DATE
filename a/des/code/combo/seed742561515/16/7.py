from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
WINDOW_TYPES: dict[str, tuple[int, int, int, int]] = {
    "horizontal": (2, 2, 1, 1),
    "vertical": (1, 1, 2, 2),
    "both": (2, 2, 2, 2),
}
GENERATION_RETRIES = 400


def _same_color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_mask(shape: tuple[int, int], top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if top == 2:
        mask[1, :] = True
    if bottom == 2:
        mask[-2, :] = True
    if left == 2:
        mask[:, 1] = True
    if right == 2:
        mask[:, -2] = True
    return mask


def _identify_window(local_mask: np.ndarray) -> tuple[str, tuple[int, int, int, int]] | None:
    h, w = local_mask.shape
    for kind, (top, bottom, left, right) in WINDOW_TYPES.items():
        if h <= top + bottom or w <= left + right:
            continue
        expected = _frame_mask((h, w), top, bottom, left, right)
        if np.array_equal(local_mask, expected):
            return kind, (top, bottom, left, right)
    return None


def _mirror_fill_horizontal(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    src = np.array(arr, copy=True)
    h, w = src.shape
    for r in range(h):
        rr = h - 1 - r
        for c in range(w):
            a = int(src[r, c])
            b = int(src[rr, c])
            if a == BG and b != BG:
                out[r, c] = b
            elif b == BG and a != BG:
                out[rr, c] = a
    return out


def _mirror_fill_vertical(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    src = np.array(arr, copy=True)
    h, w = src.shape
    for r in range(h):
        for c in range(w):
            cc = w - 1 - c
            a = int(src[r, c])
            b = int(src[r, cc])
            if a == BG and b != BG:
                out[r, c] = b
            elif b == BG and a != BG:
                out[r, cc] = a
    return out


def _complete_interior(arr: np.ndarray, kind: str) -> np.ndarray:
    out = np.array(arr, copy=True)
    if kind in {"horizontal", "both"}:
        out = _mirror_fill_horizontal(out)
    if kind in {"vertical", "both"}:
        out = _mirror_fill_vertical(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed window by mirroring its interior according to which
    sides use a double-thickness border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = np.array(g, copy=True)

    windows: list[tuple[int, int, int, int, str, tuple[int, int, int, int]]] = []
    for _color, coords in _same_color_components(g):
        r0, c0, r1, c1 = _bbox(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local_mask[r - r0, c - c0] = True
        identified = _identify_window(local_mask)
        if identified is None:
            continue
        kind, thickness = identified
        windows.append((r0, c0, r1, c1, kind, thickness))

    for r0, c0, r1, c1, kind, (top, bottom, left, right) in windows:
        ir0 = r0 + top
        ir1 = r1 - bottom
        ic0 = c0 + left
        ic1 = c1 - right
        if ir0 > ir1 or ic0 > ic1:
            continue
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = _complete_interior(interior, kind)

    return out


def _orbit_groups(h: int, w: int, kind: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    groups: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            if kind in {"horizontal", "both"}:
                orbit.add((h - 1 - r, c))
            if kind in {"vertical", "both"}:
                orbit.add((r, w - 1 - c))
            if kind == "both":
                orbit.add((h - 1 - r, w - 1 - c))
            ordered = sorted(orbit)
            for pos in ordered:
                seen.add(pos)
            groups.append(ordered)
    return groups


def _sample_full_pattern(h: int, w: int, kind: str, palette: list[int]) -> np.ndarray:
    groups = _orbit_groups(h, w, kind)
    area = h * w
    min_cells = min(max(3, area // 6), area)
    max_cells = max(min_cells, int(round(area * 0.55)))

    for _ in range(200):
        full = np.zeros((h, w), dtype=int)
        density = random.uniform(0.28, 0.52)
        colored_group_count = 0
        multi_group_count = 0
        for group in groups:
            p = density * (0.7 if len(group) == 1 else 1.0)
            if random.random() >= p:
                continue
            color = random.choice(palette)
            for r, c in group:
                full[r, c] = color
            colored_group_count += 1
            if len(group) > 1:
                multi_group_count += 1
        cells = int(np.count_nonzero(full))
        if colored_group_count >= 2 and multi_group_count >= 1 and min_cells <= cells <= max_cells:
            return full
    raise RuntimeError("failed to sample symmetric pattern")


def _puncture_pattern(full: np.ndarray, kind: str) -> np.ndarray:
    groups = _orbit_groups(full.shape[0], full.shape[1], kind)
    candidates = [group for group in groups if len(group) > 1 and int(full[group[0][0], group[0][1]]) != BG]
    assert candidates, "need at least one non-singleton colored orbit"

    full_cells = int(np.count_nonzero(full))
    for _ in range(200):
        partial = np.array(full, copy=True)
        removed = 0
        random.shuffle(candidates)
        required_cuts = random.randint(1, max(1, min(len(candidates), 3)))
        cut_count = 0
        for group in candidates:
            force = cut_count < required_cuts
            if not force and random.random() >= 0.35:
                continue
            keep_count = random.randint(1, len(group) - 1)
            keep = set(random.sample(group, keep_count))
            for r, c in group:
                if (r, c) not in keep and partial[r, c] != BG:
                    partial[r, c] = BG
                    removed += 1
            cut_count += 1
        if removed == 0:
            continue
        if int(np.count_nonzero(partial)) >= max(1, full_cells // 4):
            return partial
    raise RuntimeError("failed to puncture symmetric pattern")


def _build_window_patch(kind: str, border_color: int, content_pool: list[int]) -> tuple[np.ndarray, np.ndarray]:
    top, bottom, left, right = WINDOW_TYPES[kind]
    for _ in range(200):
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 8)
        h = inner_h + top + bottom
        w = inner_w + left + right

        palette_size = random.randint(1, min(3, len(content_pool)))
        palette = random.sample(content_pool, palette_size)
        full = _sample_full_pattern(inner_h, inner_w, kind, palette)
        partial = _puncture_pattern(full, kind)

        border = _frame_mask((h, w), top, bottom, left, right)
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        inp[border] = border_color
        out[border] = border_color
        inp[top : h - bottom, left : w - right] = partial
        out[top : h - bottom, left : w - right] = full

        if np.array_equal(inp, out):
            continue
        return inp, out
    raise RuntimeError("failed to build window patch")


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(patches)),
        key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1],
        reverse=True,
    )
    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        patch = patches[idx][0]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, height - ph):
            for c in range(1, width - pw):
                rr0 = max(0, r - 1)
                rr1 = min(height, r + ph + 1)
                cc0 = max(0, c - 1)
                cc1 = min(width, c + pw + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (r, c)
        blocked[max(0, r - 1) : min(height, r + ph + 1), max(0, c - 1) : min(width, c + pw + 1)] = True

    assert all(pos is not None for pos in placements)
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _compose_case() -> tuple[np.ndarray, np.ndarray] | None:
    window_count = random.randint(1, 4)
    border_colors = random.sample(list(range(1, 10)), window_count)
    content_pool = [c for c in range(1, 10) if c not in border_colors]
    if not content_pool:
        return None

    kinds = [random.choice(list(WINDOW_TYPES)) for _ in range(window_count)]
    patches = [_build_window_patch(kind, border_color, content_pool) for kind, border_color in zip(kinds, border_colors, strict=False)]
    max_h = max(patch[0].shape[0] for patch in patches)
    max_w = max(patch[0].shape[1] for patch in patches)

    for _ in range(80):
        height = random.randint(max(18, max_h + 2), 30)
        width = random.randint(max(18, max_w + 2), 30)
        placements = _place_patches(patches, height, width)
        if placements is None:
            continue

        inp = np.zeros((height, width), dtype=int)
        out = np.zeros((height, width), dtype=int)
        for (patch_in, patch_out), (r, c) in zip(patches, placements, strict=False):
            ph, pw = patch_in.shape
            inp[r : r + ph, c : c + pw] = patch_in
            out[r : r + ph, c : c + pw] = patch_out
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _compose_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
