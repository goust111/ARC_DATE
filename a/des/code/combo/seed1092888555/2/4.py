import random
from collections import deque

import numpy as np


BG = 0
FRAME = 3
PATTERN_COLORS = (1, 2, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_symmetry(pattern: np.ndarray) -> str:
    horiz = np.array_equal(pattern, np.flipud(pattern))
    vert = np.array_equal(pattern, np.fliplr(pattern))
    if horiz and vert:
        return "both"
    if horiz:
        return "horiz"
    if vert:
        return "vert"
    return "none"


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    comps = _components(grid == FRAME)
    frames: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 3:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(box, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == FRAME):
            continue
        if np.any(box[~border] == FRAME):
            continue
        if len(comp) != int(border.sum()):
            continue
        frames.append((r0, c0, box[1:-1, 1:-1].copy()))
    frames.sort(key=lambda item: (item[0], item[1]))
    return frames


def _stack_patterns(patterns: list[np.ndarray]) -> np.ndarray:
    if not patterns:
        return np.zeros((0, 0), dtype=int)
    width = max(int(p.shape[1]) for p in patterns)
    height = sum(int(p.shape[0]) for p in patterns) + len(patterns) - 1
    out = np.full((height, width), BG, dtype=int)
    r = 0
    for i, pattern in enumerate(patterns):
        ph, pw = pattern.shape
        out[r : r + ph, :pw] = pattern
        r += ph
        if i + 1 < len(patterns):
            r += 1
    return out


def _render_columns(columns: list[list[np.ndarray]]) -> np.ndarray:
    blocks = [_stack_patterns(col) for col in columns if col]
    if not blocks:
        return np.zeros((1, 1), dtype=int)
    height = max(int(b.shape[0]) for b in blocks)
    width = sum(int(b.shape[1]) for b in blocks) + len(blocks) - 1
    out = np.full((height, width), BG, dtype=int)
    c = 0
    for i, block in enumerate(blocks):
        bh, bw = block.shape
        out[:bh, c : c + bw] = block
        c += bw
        if i + 1 < len(blocks):
            c += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group equal-sized framed sprites by interior symmetry type.

    Horizontal-only patterns go to the left column, dual-axis patterns to the
    middle column, and vertical-only patterns to the right column. Patterns
    without either reflection symmetry are discarded.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)

    horiz_only: list[np.ndarray] = []
    both_axes: list[np.ndarray] = []
    vert_only: list[np.ndarray] = []
    for _r, _c, pattern in frames:
        kind = _classify_symmetry(pattern)
        if kind == "horiz":
            horiz_only.append(pattern)
        elif kind == "both":
            both_axes.append(pattern)
        elif kind == "vert":
            vert_only.append(pattern)
    return _render_columns([horiz_only, both_axes, vert_only])


def _random_pattern(size: int, rng: random.Random) -> np.ndarray:
    pattern = np.zeros((size, size), dtype=int)
    fill_count = rng.randint(max(1, size), max(size + 1, (size * size * 2) // 3))
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    palette = rng.sample(PATTERN_COLORS, rng.randint(1, min(3, len(PATTERN_COLORS))))
    for r, c in cells[:fill_count]:
        pattern[r, c] = rng.choice(palette)
    return pattern


def _mirror_horiz(pattern: np.ndarray) -> np.ndarray:
    size = pattern.shape[0]
    out = pattern.copy()
    for r in range(size):
        out[size - 1 - r, :] = out[r, :]
    return out


def _mirror_vert(pattern: np.ndarray) -> np.ndarray:
    size = pattern.shape[0]
    out = pattern.copy()
    for c in range(size):
        out[:, size - 1 - c] = out[:, c]
    return out


def _force_break_vertical(pattern: np.ndarray, rng: random.Random) -> np.ndarray:
    size = pattern.shape[0]
    if size < 2:
        return pattern
    for _ in range(40):
        c = rng.randrange(size // 2)
        partner = size - 1 - c
        r = rng.randrange(size)
        rr = size - 1 - r
        if pattern[r, c] == pattern[r, partner] and pattern[rr, c] == pattern[rr, partner]:
            choices = [BG] + list(PATTERN_COLORS)
            rng.shuffle(choices)
            for val in choices:
                if val != pattern[r, partner] and val != pattern[rr, partner]:
                    pattern[r, c] = val
                    pattern[rr, c] = val
                    return pattern
    return pattern


def _force_break_horizontal(pattern: np.ndarray, rng: random.Random) -> np.ndarray:
    size = pattern.shape[0]
    if size < 2:
        return pattern
    for _ in range(40):
        r = rng.randrange(size // 2)
        partner = size - 1 - r
        c = rng.randrange(size)
        cc = size - 1 - c
        if pattern[r, c] == pattern[partner, c] and pattern[r, cc] == pattern[partner, cc]:
            choices = [BG] + list(PATTERN_COLORS)
            rng.shuffle(choices)
            for val in choices:
                if val != pattern[partner, c] and val != pattern[partner, cc]:
                    pattern[r, c] = val
                    pattern[r, cc] = val
                    return pattern
    return pattern


def _make_pattern(kind: str, size: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        if kind == "both":
            base = np.zeros((size, size), dtype=int)
            palette = rng.sample(PATTERN_COLORS, rng.randint(1, min(3, len(PATTERN_COLORS))))
            for r in range((size + 1) // 2):
                for c in range((size + 1) // 2):
                    orbit = {
                        (r, c),
                        (size - 1 - r, c),
                        (r, size - 1 - c),
                        (size - 1 - r, size - 1 - c),
                    }
                    if rng.random() < 0.45:
                        val = rng.choice(palette)
                        for rr, cc in orbit:
                            base[rr, cc] = val
            pattern = base
        elif kind == "horiz":
            seed = _random_pattern(size, rng)
            pattern = _mirror_horiz(seed)
            pattern = _force_break_vertical(pattern, rng)
        elif kind == "vert":
            seed = _random_pattern(size, rng)
            pattern = _mirror_vert(seed)
            pattern = _force_break_horizontal(pattern, rng)
        else:
            pattern = _random_pattern(size, rng)
            pattern = _force_break_horizontal(pattern, rng)
            pattern = _force_break_vertical(pattern, rng)

        if np.count_nonzero(pattern) == 0:
            continue
        if _classify_symmetry(pattern) == kind:
            return pattern
    raise RuntimeError(f"failed to build pattern for {kind}")


def _sample_layout(frame_size: int, count: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]] | None:
    min_side = max(18, frame_size * 2 + 2)
    max_side = min(30, frame_size * count + 6)
    for _ in range(120):
        height = rng.randint(min_side, max_side)
        width = rng.randint(min_side, max_side)
        coords: list[tuple[int, int]] = []
        candidates = [(r, c) for r in range(height - frame_size + 1) for c in range(width - frame_size + 1)]
        rng.shuffle(candidates)
        for r, c in candidates:
            ok = True
            for rr, cc in coords:
                if not (r + frame_size + 1 < rr or rr + frame_size + 1 < r or c + frame_size + 1 < cc or cc + frame_size + 1 < c):
                    ok = False
                    break
            if ok:
                coords.append((r, c))
                if len(coords) == count:
                    return height, width, coords
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        inner_size = rng.randint(3, 6)
        frame_size = inner_size + 2

        labels = ["horiz", "vert", "both", "none"]
        extra = rng.randint(0, 3)
        labels.extend(rng.choices(["horiz", "vert", "both", "none"], k=extra))
        rng.shuffle(labels)

        layout = _sample_layout(frame_size, len(labels), rng)
        if layout is None:
            continue
        height, width, coords = layout
        grid = np.zeros((height, width), dtype=int)

        used_sig: set[bytes] = set()
        valid = True
        for (r0, c0), kind in zip(coords, labels):
            pattern = _make_pattern(kind, inner_size, rng)
            sig = pattern.tobytes() + kind.encode()
            if sig in used_sig:
                valid = False
                break
            used_sig.add(sig)
            grid[r0 : r0 + frame_size, c0 : c0 + frame_size] = FRAME
            grid[r0 + 1 : r0 + 1 + inner_size, c0 + 1 : c0 + 1 + inner_size] = pattern
        if not valid:
            continue

        out = main(grid)
        if out.size == 0 or np.count_nonzero(out) == 0:
            continue

        found = _find_frames(grid)
        if len(found) != len(labels):
            continue
        kinds = [_classify_symmetry(pattern) for _, _, pattern in found]
        if "horiz" not in kinds or "vert" not in kinds or "both" not in kinds:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
