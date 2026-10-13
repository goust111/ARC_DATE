import numpy as np


BG = 8
SEP = 5
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _panel_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    rows = []
    start = 0
    for r in range(grid.shape[0]):
        if np.all(grid[r] == SEP):
            if start < r:
                rows.append((start, r))
            start = r + 1
    if start < grid.shape[0]:
        rows.append((start, grid.shape[0]))
    return rows


def _panel_object_mask(panel: np.ndarray) -> np.ndarray:
    mask = panel != BG
    comps = _components(mask)
    if not comps:
        return np.zeros_like(mask, dtype=bool)
    largest = max(comps, key=len)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in largest:
        out[r, c] = True
    return out


def _gap_direction(mask: np.ndarray) -> str:
    h, w = mask.shape
    touches = {
        "up": bool(mask[0].any()),
        "down": bool(mask[h - 1].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, w - 1].any()),
    }
    missing = [name for name, hit in touches.items() if not hit]
    if len(missing) != 1:
        raise ValueError(f"expected exactly one gap side, got {touches}")
    return missing[0]


def _hole_count(mask: np.ndarray) -> int:
    bg_mask = ~mask
    count = 0
    h, w = mask.shape
    for comp in _components(bg_mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        count += 1
    return count


def _extrude(mask: np.ndarray, direction: str, amount: int) -> np.ndarray:
    out = mask.copy()
    if amount <= 0:
        return out

    h, w = mask.shape
    if direction == "up":
        for c in range(w):
            rows = np.where(mask[:, c])[0]
            if rows.size == 0:
                continue
            top = int(rows.min())
            out[max(0, top - amount) : top, c] = True
    elif direction == "down":
        for c in range(w):
            rows = np.where(mask[:, c])[0]
            if rows.size == 0:
                continue
            bottom = int(rows.max())
            out[bottom + 1 : min(h, bottom + 1 + amount), c] = True
    elif direction == "left":
        for r in range(h):
            cols = np.where(mask[r])[0]
            if cols.size == 0:
                continue
            left = int(cols.min())
            out[r, max(0, left - amount) : left] = True
    elif direction == "right":
        for r in range(h):
            cols = np.where(mask[r])[0]
            if cols.size == 0:
                continue
            right = int(cols.max())
            out[r, right + 1 : min(w, right + 1 + amount)] = True
    else:
        raise ValueError(direction)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, r1 in _panel_slices(g):
        panel = g[r0:r1]
        mask = _panel_object_mask(panel)
        if not mask.any():
            continue
        direction = _gap_direction(mask)
        holes = _hole_count(mask)
        grown = _extrude(mask, direction, holes)
        color = int(np.unique(panel[mask])[0])
        out[r0:r1][grown] = color
    return out


def _profile_mask(h: int, w: int, direction: str, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    base = int(rng.integers(1, max(2, min(4, (h if direction in ("up", "down") else w) - 2))))
    jitter_cap = 2
    if direction in ("up", "down"):
        profile = []
        cur = base
        for _ in range(w):
            cur = int(np.clip(cur + rng.integers(-1, 2), 1, h - 2))
            profile.append(cur)
        if max(profile) - min(profile) == 0 and h >= 6:
            idx = int(rng.integers(1, w - 1)) if w >= 3 else 0
            profile[idx] = min(h - 2, profile[idx] + 1)
        for c, val in enumerate(profile):
            if direction == "up":
                mask[val:, c] = True
            else:
                mask[: h - val, c] = True
    else:
        profile = []
        cur = base
        for _ in range(h):
            cur = int(np.clip(cur + rng.integers(-1, 2), 1, w - 2))
            profile.append(cur)
        if max(profile) - min(profile) == 0 and w >= 6:
            idx = int(rng.integers(1, h - 1)) if h >= 3 else 0
            profile[idx] = min(w - 2, profile[idx] + 1)
        for r, val in enumerate(profile):
            if direction == "left":
                mask[r, val:] = True
            else:
                mask[r, : w - val] = True
    return mask


def _can_carve(mask: np.ndarray, top: int, left: int, hh: int, ww: int) -> bool:
    h, w = mask.shape
    if top <= 0 or left <= 0 or top + hh >= h or left + ww >= w:
        return False
    if not mask[top : top + hh, left : left + ww].all():
        return False
    for r in range(top - 1, top + hh + 1):
        for c in range(left - 1, left + ww + 1):
            if top <= r < top + hh and left <= c < left + ww:
                continue
            if not mask[r, c]:
                return False
    return True


def _add_holes(mask: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    made = 0
    for _ in range(200):
        if made >= target:
            break
        hh = int(rng.integers(1, min(4, out.shape[0] - 1)))
        ww = int(rng.integers(1, min(4, out.shape[1] - 1)))
        top = int(rng.integers(1, out.shape[0] - hh))
        left = int(rng.integers(1, out.shape[1] - ww))
        if not _can_carve(out, top, left, hh, ww):
            continue
        candidate = out.copy()
        candidate[top : top + hh, left : left + ww] = False
        comps = _components(candidate)
        if len(comps) != 1:
            continue
        if _hole_count(candidate) != made + 1:
            continue
        out = candidate
        made += 1
    return out


def _random_panel(h: int, w: int, direction: str, holes: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        mask = _profile_mask(h, w, direction, rng)
        mask = _add_holes(mask, holes, rng)
        if _gap_direction(mask) != direction:
            continue
        if _hole_count(mask) != holes:
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_count = int(rng.integers(2, 5))
    width = int(rng.integers(8, 16))
    heights = [int(rng.integers(6, 11)) for _ in range(panel_count)]
    total_h = sum(heights) + panel_count - 1
    grid = np.full((total_h, width), BG, dtype=int)

    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    hole_counts = [int(rng.integers(0, 4)) for _ in range(panel_count)]
    if panel_count >= 2:
        hole_counts[0] = 0
        hole_counts[1] = int(rng.integers(1, 4))
    colors = [c for c in range(10) if c not in (BG, SEP)]
    rng.shuffle(colors)

    row = 0
    used_dirs = set()
    for i in range(panel_count):
        h = heights[i]
        direction = directions[i % 4] if i < 4 else str(rng.choice(directions))
        used_dirs.add(direction)
        mask = _random_panel(h, width, direction, hole_counts[i], rng)
        if mask is None:
            return None
        grid[row : row + h][mask] = colors[i]
        row += h
        if i != panel_count - 1:
            grid[row, :] = SEP
            row += 1

    if len(used_dirs) < min(2, panel_count):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid input")
