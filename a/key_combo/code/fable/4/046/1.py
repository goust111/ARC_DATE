from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Color:
    BLACK = 0
    NOT_BLACK = tuple(range(1, 10))


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1


def _template_order(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    assert h == w, "expected a square grid"
    n = int(round(h ** 0.5))
    assert n * n == h and n in (3, 4), "expected a 9x9 or 16x16 grid"
    return n


def _rect_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = True
    return mask


def _solid_color_rectangles(grid: np.ndarray) -> list[tuple[int, Rect]]:
    candidates: list[tuple[int, Rect]] = []
    for color in sorted(set(map(int, grid.flatten())) - {Color.BLACK}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = Rect(int(r0), int(c0), int(r1), int(c1))
        if coords.shape[0] != rect.h * rect.w:
            continue
        if np.all(grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] == color):
            candidates.append((color, rect))
    return candidates


def _render(template: np.ndarray) -> np.ndarray:
    n = int(template.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    active = template != Color.BLACK
    for br in range(n):
        for bc in range(n):
            if active[br, bc]:
                out[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = template
    return out


def _visible_observations(grid: np.ndarray, occ_rect: Rect, n: int) -> list[tuple[int, int, int]]:
    occ = _rect_mask(grid.shape, occ_rect)
    obs: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if occ[r, c]:
                continue
            block_idx = (r // n) * n + (c // n)
            inner_idx = (r % n) * n + (c % n)
            obs.append((block_idx, inner_idx, int(grid[r, c])))
    return obs


def _solve_with_occluder(grid: np.ndarray, occ_rect: Rect) -> np.ndarray | None:
    n = _template_order(grid)
    size = n * n
    observations = _visible_observations(grid, occ_rect, n)

    required_bits = 0
    zero_pairs: list[tuple[int, int]] = []
    visible_table: list[list[int | None]] = [[None] * size for _ in range(size)]
    for block_idx, inner_idx, value in observations:
        visible_table[block_idx][inner_idx] = value
        if value == Color.BLACK:
            zero_pairs.append((block_idx, inner_idx))
        else:
            required_bits |= 1 << block_idx
            required_bits |= 1 << inner_idx

    free_positions = [idx for idx in range(size) if ((required_bits >> idx) & 1) == 0]
    solutions: dict[bytes, np.ndarray] = {}

    for mask_bits in range(1 << len(free_positions)):
        active_bits = required_bits
        for bit_offset, pos in enumerate(free_positions):
            if (mask_bits >> bit_offset) & 1:
                active_bits |= 1 << pos

        bad = False
        for block_idx, inner_idx in zero_pairs:
            if ((active_bits >> block_idx) & 1) and ((active_bits >> inner_idx) & 1):
                bad = True
                break
        if bad:
            continue

        colors = [0] * size
        for inner_idx in range(size):
            if ((active_bits >> inner_idx) & 1) == 0:
                continue
            color_values = set()
            saw_visible_active = False
            for block_idx in range(size):
                if ((active_bits >> block_idx) & 1) == 0:
                    continue
                value = visible_table[block_idx][inner_idx]
                if value is None:
                    continue
                saw_visible_active = True
                if value == Color.BLACK:
                    bad = True
                    break
                color_values.add(int(value))
            if bad or len(color_values) > 1 or not saw_visible_active:
                bad = True
                break
            colors[inner_idx] = color_values.pop()
        if bad:
            continue

        template = np.array(colors, dtype=int).reshape(n, n)
        full = _render(template)
        occ = _rect_mask(grid.shape, occ_rect)
        if np.array_equal(full[~occ], grid[~occ]):
            solutions[full.tobytes()] = full

    if len(solutions) != 1:
        return None
    return next(iter(solutions.values()))


def _recover_full_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for _color, rect in _solid_color_rectangles(g):
        full = _solve_with_occluder(g, rect)
        if full is not None:
            candidates.append(full)
    unique = {cand.tobytes(): cand for cand in candidates}
    assert len(unique) == 1, "expected a unique self-similar reconstruction"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a self-stamping fractal image hidden behind one solid-color occluding rectangle.

    Input structure:
    - The grid is 9x9 or 16x16 with black background.
    - It was built from an n-by-n template where each non-black template cell activates one whole n-by-n block that copies the same template, while black template cells activate all-black blocks.
    - One non-black color appears only as a solid axis-aligned rectangle that occludes part of the fractal image.

    Rule:
    - Find the color region that is a filled rectangle and treat it as the occluder.
    - Use the visible cells plus the self-similarity law "active blocks equal the template and the active-block mask equals the template non-black mask" to infer the unique n-by-n template.
    - Re-render the full fractal image from that recovered template and remove the occluder.

    Key insight:
    - Every visible non-black pixel simultaneously proves one template cell is non-black and reveals the color of another template cell, so the template mask and the template content constrain each other in a closed loop.
    """
    return _recover_full_grid(np.asarray(grid, dtype=int))


def _random_template(n: int, rng: np.random.Generator) -> np.ndarray:
    size = n * n
    active_count = int(rng.integers(max(3, n), size))
    active_indices = rng.choice(size, size=active_count, replace=False)
    palette_size = int(rng.integers(2, min(6, active_count) + 1))
    palette = rng.choice(np.array(Color.NOT_BLACK, dtype=int), size=palette_size, replace=False)
    template = np.zeros((n, n), dtype=int)
    for idx in active_indices:
        template[idx // n, idx % n] = int(rng.choice(palette))
    return template


def _covers_nonzero(full: np.ndarray, rect: Rect) -> bool:
    return bool(np.any(full[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] != Color.BLACK))


def _visibility_complete(template: np.ndarray, rect: Rect) -> bool:
    n = int(template.shape[0])
    active = template != Color.BLACK
    for inner_r in range(n):
        for inner_c in range(n):
            if not active[inner_r, inner_c]:
                continue
            seen = False
            for block_r in range(n):
                for block_c in range(n):
                    if not active[block_r, block_c]:
                        continue
                    r = block_r * n + inner_r
                    c = block_c * n + inner_c
                    if not (rect.r0 <= r <= rect.r1 and rect.c0 <= c <= rect.c1):
                        seen = True
                        break
                if seen:
                    break
            if not seen:
                return False
    return True


def _random_rect(size: int, rng: np.random.Generator) -> Rect:
    if rng.random() < 0.45:
        n = int(round(size ** 0.5))
        row_edges = sorted(rng.choice(np.arange(n + 1), size=2, replace=False))
        col_edges = sorted(rng.choice(np.arange(n + 1), size=2, replace=False))
        r0 = int(row_edges[0] * n)
        r1 = int(row_edges[1] * n - 1)
        c0 = int(col_edges[0] * n)
        c1 = int(col_edges[1] * n - 1)
        if r0 <= r1 and c0 <= c1:
            return Rect(r0, c0, r1, c1)
    h = int(rng.integers(1, size))
    w = int(rng.integers(1, size))
    r0 = int(rng.integers(0, size - h + 1))
    c0 = int(rng.integers(0, size - w + 1))
    return Rect(r0, c0, r0 + h - 1, c0 + w - 1)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(600):
        n = int(rng.choice(np.array([3, 4], dtype=int)))
        template = _random_template(n, rng)
        full = _render(template)
        size = full.shape[0]

        used_colors = set(map(int, template.flatten())) - {Color.BLACK}
        available_occ = [color for color in Color.NOT_BLACK if color not in used_colors]
        if not available_occ:
            continue

        rect = _random_rect(size, rng)
        if rect.h * rect.w >= size * size:
            continue
        if not _covers_nonzero(full, rect):
            continue
        if not _visibility_complete(template, rect):
            continue

        occ_color = int(rng.choice(np.array(available_occ, dtype=int)))
        grid = full.copy()
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = occ_color

        if np.array_equal(grid, full):
            continue
        recovered = _recover_full_grid(grid)
        if np.array_equal(recovered, full):
            return grid
    raise RuntimeError("failed to generate a valid fractal-occlusion input")
