from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BORDER = 3
COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRANCH_PATTERNS = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
)
GRID_CHOICES = (18, 20, 22, 24)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 80
PADDING = 1


def _find_components_monochrome(grid: np.ndarray, bg: int = BG) -> list[dict]:
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
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _corner_colors(grid: np.ndarray) -> dict[str, int]:
    return {
        "TL": int(grid[0, 0]),
        "TR": int(grid[0, -1]),
        "BL": int(grid[-1, 0]),
        "BR": int(grid[-1, -1]),
    }


def _touched_sides(coords: list[tuple[int, int]], h: int, w: int) -> set[str]:
    sides: set[str] = set()
    if any(r == 1 for r, _ in coords):
        sides.add("top")
    if any(r == h - 2 for r, _ in coords):
        sides.add("bottom")
    if any(c == 1 for _, c in coords):
        sides.add("left")
    if any(c == w - 2 for _, c in coords):
        sides.add("right")
    return sides


def _quadrant_from_coords(coords: list[tuple[int, int]], h: int, w: int) -> str:
    arr = np.array(coords, dtype=float)
    top = float(arr[:, 0].mean()) < (h / 2.0)
    left = float(arr[:, 1].mean()) < (w / 2.0)
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def _transform_coords(coords: list[tuple[int, int]], mode: str) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    cr2 = int(r0 + r1)
    cc2 = int(c0 + c1)
    out: list[tuple[int, int]] = []
    for r, c in coords:
        dr2 = 2 * int(r) - cr2
        dc2 = 2 * int(c) - cc2
        if mode == "rot_cw":
            nr2 = cr2 + dc2
            nc2 = cc2 - dr2
        elif mode == "diag_main":
            nr2 = cr2 + dc2
            nc2 = cc2 + dr2
        elif mode == "diag_anti":
            nr2 = cr2 - dc2
            nc2 = cc2 - dr2
        else:
            raise ValueError(f"unknown transform mode: {mode}")
        if (nr2 % 2) or (nc2 % 2):
            raise ValueError("transform requires an integral bounding-box center placement")
        out.append((nr2 // 2, nc2 // 2))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Transform each interior object according to how many frame sides it touches.

    Input structure:
    - Background is black.
    - The outer ring is a one-cell green frame whose four corners are recolored
      to four distinct non-black, non-green marker colors.
    - Small monochrome 4-connected objects sit inside the frame; some touch one
      side, some sit in a corner touching two sides, and some float fully inside.

    Rule:
    - Treat every monochrome interior component as one object and count which
      frame sides it touches via adjacency to the border (top/bottom/left/right).
    - Objects touching exactly one side rotate 90 degrees clockwise around their
      bounding-box center; corner objects touching two adjacent sides reflect
      across the corner diagonal (main diagonal for TL/BR, anti-diagonal for TR/BL).
    - Objects touching no side keep their shape and position but all pixels are
      recolored to the corner-marker color of their grid quadrant; frame and
      corner markers stay unchanged.

    Key insight:
    - Border contact count chooses the geometric operation, while the four corner
      pixels are quadrant color instructions for floating objects rather than decoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    corner_map = _corner_colors(g)
    interior = g[1:-1, 1:-1]
    comps = _find_components_monochrome(interior, bg=BG)

    out = np.full_like(g, BG)
    out[0, :] = g[0, :]
    out[-1, :] = g[-1, :]
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]

    for comp in comps:
        abs_coords = [(r + 1, c + 1) for r, c in comp["coords"]]
        sides = _touched_sides(abs_coords, h, w)
        color = int(comp["color"])

        if len(sides) == 0:
            new_coords = abs_coords
            color = corner_map[_quadrant_from_coords(abs_coords, h, w)]
        elif len(sides) == 1:
            new_coords = _transform_coords(abs_coords, "rot_cw")
        else:
            side_key = frozenset(sides)
            if side_key in (frozenset({"top", "left"}), frozenset({"bottom", "right"})):
                new_coords = _transform_coords(abs_coords, "diag_main")
            elif side_key in (frozenset({"top", "right"}), frozenset({"bottom", "left"})):
                new_coords = _transform_coords(abs_coords, "diag_anti")
            else:
                raise ValueError(f"unexpected side combination: {sorted(sides)}")

        for r, c in new_coords:
            assert 1 <= r < h - 1 and 1 <= c < w - 1
            out[r, c] = color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _anti_diagonal_reflect(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask.T, 2)


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    uniq: list[np.ndarray] = []
    seen: set[bytes] = set()
    for variant in variants:
        key = variant.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(variant.copy())
    return uniq


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[bytes] = set()
    n = 3
    for bits in range(1, 1 << (n * n)):
        flat = np.array([(bits >> i) & 1 for i in range(n * n)], dtype=bool)
        mask = flat.reshape(n, n)
        area = int(mask.sum())
        if area < 3 or area > 6:
            continue
        if not _is_connected(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.array_equal(mask, np.rot90(mask, -1)):
            continue
        if np.array_equal(mask, mask.T):
            continue
        if np.array_equal(mask, _anti_diagonal_reflect(mask)):
            continue
        canonical = min(_d4_variants(mask), key=lambda arr: arr.tobytes())
        key = canonical.tobytes()
        if key in seen:
            continue
        seen.add(key)
        shapes.append(canonical.copy())
    assert shapes, "shape library must not be empty"
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _sample_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_d4_variants(mask)).copy()


def _sample_bbox(slot_kind: str, slot_name: str, size: int, h: int, w: int) -> tuple[int, int]:
    if slot_kind == "corner":
        if slot_name == "TL":
            return 1, 1
        if slot_name == "TR":
            return 1, w - size - 1
        if slot_name == "BL":
            return h - size - 1, 1
        if slot_name == "BR":
            return h - size - 1, w - size - 1
        raise ValueError(slot_name)

    if slot_kind == "edge":
        if slot_name == "top":
            return 1, random.randint(2, w - size - 2)
        if slot_name == "bottom":
            return h - size - 1, random.randint(2, w - size - 2)
        if slot_name == "left":
            return random.randint(2, h - size - 2), 1
        if slot_name == "right":
            return random.randint(2, h - size - 2), w - size - 1
        raise ValueError(slot_name)

    half_h = h // 2
    half_w = w // 2
    if slot_name == "TL":
        return random.randint(2, half_h - size), random.randint(2, half_w - size)
    if slot_name == "TR":
        return random.randint(2, half_h - size), random.randint(half_w, w - size - 2)
    if slot_name == "BL":
        return random.randint(half_h, h - size - 2), random.randint(2, half_w - size)
    if slot_name == "BR":
        return random.randint(half_h, h - size - 2), random.randint(half_w, w - size - 2)
    raise ValueError(slot_name)


def _fits(blocked: np.ndarray, r0: int, c0: int, size: int, h: int, w: int) -> bool:
    rr0 = max(1, r0 - PADDING)
    cc0 = max(1, c0 - PADDING)
    rr1 = min(h - 2, r0 + size - 1 + PADDING)
    cc1 = min(w - 2, c0 + size - 1 + PADDING)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, size: int, h: int, w: int) -> None:
    rr0 = max(1, r0 - PADDING)
    cc0 = max(1, c0 - PADDING)
    rr1 = min(h - 2, r0 + size - 1 + PADDING)
    cc1 = min(w - 2, c0 + size - 1 + PADDING)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _build_specs() -> list[dict]:
    floating_n, edge_n, corner_n = random.choice(BRANCH_PATTERNS)
    specs = [{"kind": "floating", "slot": slot} for slot in random.sample(("TL", "TR", "BL", "BR"), floating_n)]
    specs.extend({"kind": "edge", "slot": slot} for slot in random.sample(("top", "right", "bottom", "left"), edge_n))
    specs.extend({"kind": "corner", "slot": slot} for slot in random.sample(("TL", "TR", "BL", "BR"), corner_n))
    random.shuffle(specs)
    return specs


def _assign_object_colors(specs: list[dict], corner_map: dict[str, int]) -> list[int] | None:
    available = list(COLOR_POOL)
    random.shuffle(available)
    chosen: list[int] = []
    for spec in specs:
        forbidden = {corner_map[spec["slot"]]} if spec["kind"] == "floating" else set()
        candidates = [color for color in available if color not in forbidden]
        if not candidates:
            return None
        color = random.choice(candidates)
        chosen.append(color)
        available.remove(color)
    return chosen


def _paint_frame(grid: np.ndarray, corner_map: dict[str, int]) -> None:
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER
    grid[0, 0] = corner_map["TL"]
    grid[0, -1] = corner_map["TR"]
    grid[-1, 0] = corner_map["BL"]
    grid[-1, -1] = corner_map["BR"]


def _generate_candidate() -> np.ndarray | None:
    h = random.choice(GRID_CHOICES)
    w = random.choice(GRID_CHOICES)
    specs = _build_specs()
    corner_values = random.sample(COLOR_POOL, 4)
    corner_map = dict(zip(("TL", "TR", "BL", "BR"), corner_values))
    colors = _assign_object_colors(specs, corner_map)
    if colors is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_frame(grid, corner_map)
    blocked = np.zeros((h, w), dtype=bool)

    for spec, color in zip(specs, colors):
        mask = _sample_variant(random.choice(SHAPE_LIBRARY))
        size = int(mask.shape[0])
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0, c0 = _sample_bbox(spec["kind"], spec["slot"], size, h, w)
            if not _fits(blocked, r0, c0, size, h, w):
                continue
            cells = np.argwhere(mask)
            for dr, dc in cells:
                grid[r0 + int(dr), c0 + int(dc)] = color
            _mark_blocked(blocked, r0, c0, size, h, w)
            placed = True
            break
        if not placed:
            return None

    comps = _find_components_monochrome(grid[1:-1, 1:-1], bg=BG)
    if len(comps) != len(specs):
        return None

    branch_counts = Counter()
    for comp in comps:
        abs_coords = [(r + 1, c + 1) for r, c in comp["coords"]]
        sides = _touched_sides(abs_coords, h, w)
        if len(sides) > 2:
            return None
        if len(sides) == 0:
            arr = np.array(abs_coords, dtype=int)
            if not (arr[:, 0].max() < h // 2 or arr[:, 0].min() >= h // 2):
                return None
            if not (arr[:, 1].max() < w // 2 or arr[:, 1].min() >= w // 2):
                return None
        if len(sides) == 2:
            if frozenset(sides) not in (
                frozenset({"top", "left"}),
                frozenset({"top", "right"}),
                frozenset({"bottom", "left"}),
                frozenset({"bottom", "right"}),
            ):
                return None
        branch_counts[len(sides)] += 1

    expected = Counter(
        {
            0: sum(1 for spec in specs if spec["kind"] == "floating"),
            1: sum(1 for spec in specs if spec["kind"] == "edge"),
            2: sum(1 for spec in specs if spec["kind"] == "corner"),
        }
    )
    if branch_counts != expected:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[0, :], grid[0, :]) or not np.array_equal(out[-1, :], grid[-1, :]):
        return None
    if not np.array_equal(out[:, 0], grid[:, 0]) or not np.array_equal(out[:, -1], grid[:, -1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
