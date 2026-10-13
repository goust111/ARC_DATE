from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)
NON_ID_TRANSFORMS = tuple(t for t in TRANSFORMS if t != "id")
SYMMETRY_MODES = ("flip_lr", "flip_ud", "transpose", "anti_transpose")
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
BG = 0
GEN_ATTEMPTS = 500


def _apply_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return grid.copy()
    if name == "rot90":
        return np.rot90(grid, 1).copy()
    if name == "rot180":
        return np.rot90(grid, 2).copy()
    if name == "rot270":
        return np.rot90(grid, 3).copy()
    if name == "flip_lr":
        return np.fliplr(grid).copy()
    if name == "flip_ud":
        return np.flipud(grid).copy()
    if name == "transpose":
        return grid.T.copy()
    if name == "anti_transpose":
        return np.fliplr(np.rot90(grid, 1)).copy()
    raise ValueError(f"unknown transform: {name}")


def _matching_transforms(src: np.ndarray, dst: np.ndarray) -> list[str]:
    return [name for name in TRANSFORMS if np.array_equal(_apply_transform(src, name), dst)]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_divider_columns(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    candidates: list[list[int]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(cols) != 4:
            continue
        if int(np.count_nonzero(grid == color)) != h * len(cols):
            continue
        candidates.append(cols)
    assert len(candidates) == 1, "expected one divider color occupying exactly four full-height columns"
    return candidates[0]


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)
    divs = _find_divider_columns(g, bg)
    widths = [divs[0], divs[1] - divs[0] - 1, divs[2] - divs[1] - 1, divs[3] - divs[2] - 1, w - divs[3] - 1]
    assert len(set(widths)) == 1, "all panels must have equal width"
    panel_n = widths[0]
    assert panel_n == h, "panels must be square"
    spans = [
        (0, divs[0]),
        (divs[0] + 1, divs[1]),
        (divs[1] + 1, divs[2]),
        (divs[2] + 1, divs[3]),
        (divs[3] + 1, w),
    ]
    return [g[:, c0:c1].copy() for c0, c1 in spans]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the shared square symmetry from two example pairs and apply it to the query panel.

    Input structure:
    - The grid is split by four full-height divider columns into five same-size square panels.
    - Panels 1->2 and 3->4 are two input/output examples; panel 5 is the query.
    - Each panel is a square canvas with background and a colored sprite.

    Rule:
    - There is exactly one transform among the eight square-preserving flips and rotations
      that satisfies both example pairs simultaneously.
    - The first example alone is ambiguous because its sprite has partial symmetry.
    - Find the unique transform from the intersection of both example-pair hypotheses,
      then apply that transform to the query panel.

    Key insight:
    - You must intersect the transform candidates from both examples; taking the first
      transform that fits panel 1->2 fails on the deliberate symmetric decoy.
    """
    panels = _extract_panels(np.asarray(grid, dtype=int))
    cand_12 = _matching_transforms(panels[0], panels[1])
    cand_34 = _matching_transforms(panels[2], panels[3])
    common = [name for name in TRANSFORMS if name in cand_12 and name in cand_34]
    assert len(common) == 1, "the two examples must determine a unique D4 transform"
    return _apply_transform(panels[4], common[0])


def _neighbors4(r: int, c: int, n: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    n = mask.shape[0]
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, n):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(np.count_nonzero(mask))


def _mask_stabilizers(mask: np.ndarray) -> list[str]:
    return [name for name in TRANSFORMS if np.array_equal(_apply_transform(mask.astype(int), name) != 0, mask)]


def _has_full_line(mask: np.ndarray) -> bool:
    return bool(np.any(np.all(mask, axis=0)) or np.any(np.all(mask, axis=1)))


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _point_transform(mode: str, r: int, c: int, n: int) -> tuple[int, int]:
    if mode == "id":
        return r, c
    if mode == "rot90":
        return n - 1 - c, r
    if mode == "rot180":
        return n - 1 - r, n - 1 - c
    if mode == "rot270":
        return c, n - 1 - r
    if mode == "flip_lr":
        return r, n - 1 - c
    if mode == "flip_ud":
        return n - 1 - r, c
    if mode == "transpose":
        return c, r
    if mode == "anti_transpose":
        return n - 1 - c, n - 1 - r
    raise ValueError(mode)


def _domain_cells(n: int, mode: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            if mode == "flip_lr" and c <= (n - 1) // 2:
                cells.append((r, c))
            elif mode == "flip_ud" and r <= (n - 1) // 2:
                cells.append((r, c))
            elif mode == "transpose" and r <= c:
                cells.append((r, c))
            elif mode == "anti_transpose" and r + c <= n - 1:
                cells.append((r, c))
    return cells


def _boundary_cells(n: int, mode: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if mode == "flip_lr":
        boundary_c = n // 2 if n % 2 else n // 2 - 1
        for r in range(n):
            cells.append((r, boundary_c))
    elif mode == "flip_ud":
        boundary_r = n // 2 if n % 2 else n // 2 - 1
        for c in range(n):
            cells.append((boundary_r, c))
    elif mode == "transpose":
        for i in range(n):
            cells.append((i, i))
    elif mode == "anti_transpose":
        for i in range(n):
            cells.append((i, n - 1 - i))
    else:
        raise ValueError(mode)
    return cells


def _build_symmetric_mask(n: int, mode: str) -> np.ndarray | None:
    domain = set(_domain_cells(n, mode))
    boundary = _boundary_cells(n, mode)
    target = random.randint(2, min(5, len(domain)))
    canonical = {random.choice(boundary)}

    while len(canonical) < target:
        frontier = set()
        for r, c in canonical:
            for rr, cc in _neighbors4(r, c, n):
                if (rr, cc) in domain and (rr, cc) not in canonical:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        canonical.add(random.choice(tuple(frontier)))

    full = set(canonical)
    for r, c in tuple(canonical):
        full.add(_point_transform(mode, r, c, n))

    mask = np.zeros((n, n), dtype=bool)
    for r, c in full:
        mask[r, c] = True

    stabilizers = _mask_stabilizers(mask)
    if not _is_connected(mask):
        return None
    if _has_full_line(mask):
        return None
    if set(stabilizers) != {"id", mode}:
        return None
    return mask


def _build_asymmetric_mask(n: int) -> np.ndarray | None:
    target = random.randint(max(4, n), min(8, n * n - 2))
    cells = {(random.randrange(n), random.randrange(n))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, n):
                if (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not _is_connected(mask):
        return None
    if _has_full_line(mask):
        return None
    if _mask_stabilizers(mask) != ["id"]:
        return None
    return mask


def _assemble_grid(panels: list[np.ndarray], divider_color: int) -> np.ndarray:
    n = panels[0].shape[0]
    out = np.zeros((n, 5 * n + 4), dtype=int)
    for idx, panel in enumerate(panels):
        c0 = idx * (n + 1)
        out[:, c0:c0 + n] = panel
        if idx < 4:
            out[:, c0 + n] = int(divider_color)
    return out


def _panel_signature(panel: np.ndarray) -> bytes:
    return panel.tobytes() + str(panel.shape).encode()


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice((4, 5))
    true_transform = random.choice(NON_ID_TRANSFORMS)
    divider_color, color_a, color_b, color_q = random.sample(PALETTE, 4)
    symmetry_mode = random.choice(SYMMETRY_MODES)

    panel1_mask = _build_symmetric_mask(n, symmetry_mode)
    if panel1_mask is None:
        return None
    panel3_mask = _build_asymmetric_mask(n)
    if panel3_mask is None:
        return None
    query_mask = _build_asymmetric_mask(n)
    if query_mask is None:
        return None

    panel1 = _colorize(panel1_mask, color_a)
    panel2 = _apply_transform(panel1, true_transform)
    panel3 = _colorize(panel3_mask, color_b)
    panel4 = _apply_transform(panel3, true_transform)
    query = _colorize(query_mask, color_q)
    answer = _apply_transform(query, true_transform)

    cand_12 = _matching_transforms(panel1, panel2)
    cand_34 = _matching_transforms(panel3, panel4)
    if len(cand_12) != 2 or true_transform not in cand_12:
        return None
    if cand_34 != [true_transform]:
        return None
    if set(cand_12) & set(cand_34) != {true_transform}:
        return None
    if len({_panel_signature(_apply_transform(query, name)) for name in TRANSFORMS}) != len(TRANSFORMS):
        return None
    if np.array_equal(answer, query):
        return None
    if np.array_equal(panel3, panel4):
        return None

    grid = _assemble_grid([panel1, panel2, panel3, panel4, query], divider_color)
    if not np.array_equal(main(grid), answer):
        return None
    return grid, answer


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _generate_case()
        if generated is not None:
            grid, _answer = generated
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
